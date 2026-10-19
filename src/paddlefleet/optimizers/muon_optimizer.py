# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""``MuonOptimizer`` -- Newton-Schulz-orthogonalised momentum update.

A from-scratch ``InnerOptimizer`` subclass (NOT a wrapper around
``paddle.optimizer.Muon``). State per parameter: a single fp32 momentum buffer
(``moment1``) plus an optional fp32 master weight. Parameters that share a shape
(and slice strategy) are batched into one Newton-Schulz call. Registered as
``muon``; the Hyperball variant lives in ``muon_hyperball_optimizer.py``.
"""

from __future__ import annotations

import math
from collections import defaultdict

import paddle
from paddle.base import framework
from paddle.optimizer.inner_optimizer import InnerOptimizer
from paddle.optimizer.mixed_inner_optimizer import register_inner_optimizer

from .muon_math import (
    NS_COEFFICIENT_SETS,
    auto_ns_matmul_dtype,
    check_symmetric_gemm,
    newton_schulz,
    scaling,
    split_func_key,
)

__all__ = ["MuonOptimizer"]


class MuonOptimizer(InnerOptimizer):
    """Muon: Newton-Schulz-orthogonalised momentum update for 2D/3D matrices."""

    _moment_acc_str = "moment1"
    # Newton-Schulz orthogonalisation needs the whole matrix, so the sharding
    # shell must keep every Muon parameter on a single owner rank (V1).
    keep_whole = True

    def __init__(
        self,
        parameters,
        learning_rate=0.02,
        momentum=0.95,
        ns_steps=5,
        muon_version=3,
        ns_coeff_type="quintic",
        ns_coeffs=None,
        muon_extra_scale_factor=0.2,  # same default as paddle.optimizer.Muon
        muon_max_group_bytes=2 * 1024 * 1024 * 1024,  # same cap as Muon
        muon_epsilon=1e-7,
        nesterov=True,
        ns_matmul_dtype=None,
        # Slice strategy of the fused weights: read by the trainer to build
        # this group's split_concat_func (passed in via param_info_map).
        muon_qkv_update_mode="split_head",
        muon_ffn_split=False,
        use_symmetric_gemm=False,
        symmetric_gemm_min_short_edge=1024,  # same thresholds as Muon
        symmetric_gemm_min_step_flops=2e11,
        weight_decay=None,  # required (InnerOptimizer rejects None)
        grad_clip=None,
        lr_ratio=None,
        apply_decay_param_fun=None,
        multi_precision=True,
        param_info_map=None,
        name=None,
    ):
        super().__init__(
            learning_rate=learning_rate,
            parameters=parameters,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            lr_ratio=lr_ratio,
            apply_decay_param_fun=apply_decay_param_fun,
            multi_precision=multi_precision,
            param_info_map=param_info_map,
            name=name,
        )
        self._momentum = momentum
        self._ns_steps = ns_steps
        self._version = muon_version
        self._extra_scale_factor = muon_extra_scale_factor
        self._max_group_bytes = muon_max_group_bytes
        self._muon_epsilon = muon_epsilon
        self._nesterov = nesterov
        self._wd = weight_decay
        if ns_coeff_type == "custom":
            if ns_coeffs is None:
                raise ValueError(
                    "ns_coeffs must be given when ns_coeff_type='custom'."
                )
            self._ns_coeffs = ns_coeffs
        else:
            if ns_coeff_type not in NS_COEFFICIENT_SETS:
                raise ValueError(f"invalid ns_coeff_type: {ns_coeff_type!r}")
            self._ns_coeffs = NS_COEFFICIENT_SETS[ns_coeff_type]
        self._ns_matmul_dtype = (
            auto_ns_matmul_dtype()
            if ns_matmul_dtype is None
            else ns_matmul_dtype
        )
        self._use_symmetric_gemm = use_symmetric_gemm
        self._symmetric_gemm_min_short_edge = symmetric_gemm_min_short_edge
        self._symmetric_gemm_min_step_flops = symmetric_gemm_min_step_flops
        if use_symmetric_gemm:
            check_symmetric_gemm(
                self._ns_matmul_dtype,
                symmetric_gemm_min_short_edge,
                symmetric_gemm_min_step_flops,
            )

    # -- descriptors read by the framework layer (pretrain.py / trainer) ----

    @classmethod
    def yaml_field_map(cls):
        """YAML group field -> ctor kwarg (required per-group hyperparameters)."""
        return {
            "momentum": "momentum",
            "version": "muon_version",
            "ns_steps": "ns_steps",
            "ns_coeff_type": "ns_coeff_type",
            "extra_scale_factor": "muon_extra_scale_factor",
            "epsilon": "muon_epsilon",
            "qkv_update_mode": "muon_qkv_update_mode",
            "ffn_split": "muon_ffn_split",
        }

    @classmethod
    def optional_field_map(cls):
        """YAML group field -> ctor kwarg; omitted = ctor default."""
        return {
            "ns_coeffs": "ns_coeffs",  # needed iff ns_coeff_type='custom'
            "use_symmetric_gemm": "use_symmetric_gemm",
        }

    # -- accumulators / descriptors -----------------------------------

    def _ensure_accumulators(self, param):
        # Muon parameters keep only a momentum buffer (+ master).
        if (
            self._moment_acc_str in self._accumulators
            and param.name in self._accumulators[self._moment_acc_str]
        ):
            return
        self._ensure_master_weight(param)
        self._add_accumulator(
            self._moment_acc_str,
            param,
            dtype=paddle.float32,
            fill_value=0.0,
            shape=param.shape,
            type=framework.core.VarDesc.VarType.DENSE_TENSOR,
        )

    def optimizer_state_layout(self) -> dict:
        # Only moment1 (+ master); no moment2 / beta powers.
        return {
            "vector": ["moment1_0"],
            "scalar": [],
            "master": "fp32_master_0",
        }

    # -- update math --------------------------------------------------

    def _update_parameters(self, params_grads, lr):
        """Group by (slice strategy, shape) then batch-update each group."""
        # fp32 lr tensor, exactly like paddle.optimizer.Muon: effective_lr =
        # lr * lr_ratio is then rounded in fp32 (a python-float lr would round
        # in fp64 and drift by 1 ulp vs the single-Muon baseline).
        lr = paddle.to_tensor(lr, dtype=paddle.float32)
        muon_groups = defaultdict(list)
        for param, grad in params_grads:
            if len(param.shape) not in (2, 3):
                raise ValueError(
                    f"Muon only supports 2D/3D parameters, got {param.name!r} "
                    f"with shape {param.shape}."
                )
            param_shape = tuple(getattr(param, "original_shape", param.shape))
            info = self._param_info_map.get(param.name)
            scf = info.split_concat_func if info is not None else None
            muon_groups[(split_func_key(scf), param_shape)].append(
                (param, grad)
            )
        for grp in muon_groups.values():
            for sub_grp in self._split_group_by_bytes(grp):
                self._update_shape_group(sub_grp, lr)

    def _split_group_by_bytes(self, group_params_grads):
        """Greedy byte-capped sub-grouping, identical to Muon's (fp32 = 4 B)."""
        if not self._max_group_bytes:
            yield group_params_grads
            return
        sub, sub_bytes = [], 0
        for param, grad in group_params_grads:
            nbytes = math.prod(param.shape) * 4
            if sub and sub_bytes + nbytes > self._max_group_bytes:
                yield sub
                sub, sub_bytes = [], 0
            sub.append((param, grad))
            sub_bytes += nbytes
        if sub:
            yield sub

    def _update_shape_group(self, group_params_grads, lr):
        param0, _ = group_params_grads[0]
        param_shape = getattr(param0, "original_shape", param0.shape)
        is_3d = len(param_shape) == 3
        info0 = self._param_info_map.get(param0.name)
        split_concat_func = (
            info0.split_concat_func if info0 is not None else None
        )

        matrices = []
        find_master = []
        with paddle.no_grad():
            # Pass 1: sequential momentum update + reshape to matrix.
            for param, grad in group_params_grads:
                momentum_buffer = self._get_accumulator(
                    self._moment_acc_str, param
                )
                grad_f32 = (
                    grad.astype(momentum_buffer.dtype)
                    if grad.dtype != momentum_buffer.dtype
                    else grad
                )
                new_momentum = paddle.lerp(
                    momentum_buffer, grad_f32, 1.0 - self._momentum
                )
                paddle.assign(new_momentum, momentum_buffer)
                update_buffer = (
                    paddle.lerp(grad_f32, momentum_buffer, self._momentum)
                    if self._nesterov
                    else momentum_buffer
                )
                matrices.append(update_buffer.reshape(param_shape))
                find_master.append(param.name in self._master_weights)

            # Pass 2: batched Newton-Schulz + scaling.
            if len(matrices) > 1:
                batched = (
                    paddle.concat(matrices, axis=0)
                    if is_3d
                    else paddle.stack(matrices, axis=0)
                )
            else:
                batched = matrices[0]

            def ortho_fn(m):
                ns_out = newton_schulz(
                    m,
                    steps=self._ns_steps,
                    eps=self._muon_epsilon,
                    ns_coeffs=self._ns_coeffs,
                    matmul_dtype=self._ns_matmul_dtype,
                    use_symmetric_gemm=self._use_symmetric_gemm,
                    symmetric_gemm_min_short_edge=self._symmetric_gemm_min_short_edge,
                    symmetric_gemm_min_step_flops=self._symmetric_gemm_min_step_flops,
                )
                return scaling(ns_out, self._version, self._extra_scale_factor)

            if split_concat_func is not None:
                batched_update = split_concat_func(batched, ortho_fn)
            else:
                batched_update = ortho_fn(batched)

            if len(matrices) > 1:
                updates = (
                    paddle.split(
                        batched_update, num_or_sections=len(matrices), axis=0
                    )
                    if is_3d
                    else paddle.unbind(batched_update, axis=0)
                )
            else:
                updates = [batched_update]

            # Pass 3: sequential apply.
            for i, (param, _) in enumerate(group_params_grads):
                self._apply_update(
                    param, updates[i], find_master[i], lr, split_concat_func
                )

    def _apply_update(
        self, param, orthogonal_update, find_master, lr, split_concat_func=None
    ):
        """Weight decay + subtract (MuonH overrides this with the projection)."""
        master = self._master_weights[param.name] if find_master else None
        effective_lr = self._effective_lr(param, lr)

        with_decay = True
        if (
            self._apply_decay_param_fun is not None
            and not self._apply_decay_param_fun(param.name)
        ):
            with_decay = False
        if with_decay and self._wd > 0:
            if find_master:
                master.scale_(1.0 - effective_lr * self._wd)
            else:
                param.scale_(1.0 - effective_lr * self._wd)

        final_step = orthogonal_update * effective_lr
        if find_master:
            master.subtract_(final_step)
            paddle.assign(master.astype(param.dtype), param)
        else:
            param.subtract_(final_step.astype(param.dtype))


register_inner_optimizer("muon", MuonOptimizer)
