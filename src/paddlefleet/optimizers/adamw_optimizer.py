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
"""``AdamWOptimizer`` -- standard decoupled AdamW (fused ``_C_ops.adamw_``).

A from-scratch ``InnerOptimizer`` subclass. State per parameter: fp32
``moment1`` / ``moment2`` + scalar ``beta1_pow`` / ``beta2_pow`` (+ optional
master weight). Element-wise, so the sharding shell may shard it per element
(V2). Registered as ``adamw``; the Hyperball variant lives in
``adam_hyperball_optimizer.py``.
"""

from __future__ import annotations

import paddle
from paddle import _C_ops
from paddle.base import framework
from paddle.optimizer.inner_optimizer import InnerOptimizer
from paddle.optimizer.mixed_inner_optimizer import register_inner_optimizer

__all__ = ["AdamWOptimizer"]


class AdamWOptimizer(InnerOptimizer):
    """Standard decoupled AdamW, using the fused ``_C_ops.adamw_`` kernel."""

    _moment_acc_str = "moment1"
    _moment2_acc_str = "moment2"
    _beta1_pow_acc_str = "beta1_pow_acc"
    _beta2_pow_acc_str = "beta2_pow_acc"
    # Element-wise update: correct both on an element-wise shard (V2) and on a
    # whole tensor (V1), so either sharding mode may be configured. Default is
    # V2, which spreads each parameter across ranks.
    keep_whole = None

    def __init__(
        self,
        parameters,
        learning_rate=0.001,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
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
        self._beta1 = adam_beta1
        self._beta2 = adam_beta2
        self._epsilon = adam_epsilon
        self._wd = weight_decay

    # -- descriptors read by the framework layer (pretrain.py / trainer) ----

    @classmethod
    def yaml_field_map(cls):
        """YAML group field -> ctor kwarg (required per-group hyperparameters)."""
        return {
            "beta1": "adam_beta1",
            "beta2": "adam_beta2",
            "epsilon": "adam_epsilon",
        }

    # -- accumulators / descriptors -----------------------------------

    def _ensure_accumulators(self, param):
        if (
            self._moment_acc_str in self._accumulators
            and param.name in self._accumulators[self._moment_acc_str]
        ):
            return
        self._ensure_master_weight(param)
        for acc_name in (self._moment_acc_str, self._moment2_acc_str):
            self._add_accumulator(
                acc_name,
                param,
                dtype=paddle.float32,
                fill_value=0.0,
                shape=param.shape,
                type=framework.core.VarDesc.VarType.DENSE_TENSOR,
            )
        for acc_name, init_val in (
            (self._beta1_pow_acc_str, self._beta1),
            (self._beta2_pow_acc_str, self._beta2),
        ):
            self._add_accumulator(
                acc_name,
                param,
                dtype=paddle.float32,
                fill_value=init_val,
                shape=[1],
                type=framework.core.VarDesc.VarType.DENSE_TENSOR,
            )

    def optimizer_state_layout(self) -> dict:
        """The state created by ``_ensure_accumulators`` above.

        moment1 / moment2 are per-element (sharded with the parameter);
        beta1/2_pow_acc are replicated scalars.
        """
        return {
            "vector": [
                f"{self._moment_acc_str}_0",
                f"{self._moment2_acc_str}_0",
            ],
            "scalar": [
                f"{self._beta1_pow_acc_str}_0",
                f"{self._beta2_pow_acc_str}_0",
            ],
            "master": "fp32_master_0",
        }

    # -- update math --------------------------------------------------

    def _accs(self, param):
        return (
            self._get_accumulator(self._moment_acc_str, param),
            self._get_accumulator(self._moment2_acc_str, param),
            self._get_accumulator(self._beta1_pow_acc_str, param),
            self._get_accumulator(self._beta2_pow_acc_str, param),
        )

    def _update_parameters(self, params_grads, lr):
        # The fused AdamW op takes a float64 lr tensor (matches Muon's path).
        lr_tensor = paddle.to_tensor(lr, dtype=paddle.float64)
        for param, grad in params_grads:
            self._adamw_one(param, grad, lr_tensor)

    def _adamw_one(self, param, grad, lr):
        moment1, moment2, beta1_pow, beta2_pow = self._accs(param)
        with_decay = True
        if (
            self._apply_decay_param_fun is not None
            and not self._apply_decay_param_fun(param.name)
        ):
            with_decay = False
        lr_ratio = 1.0 if self._lr_ratio is None else self._lr_ratio(param)
        find_master = param.name in self._master_weights
        master_weight = (
            self._master_weights[param.name] if find_master else None
        )
        _C_ops.adamw_(
            param,
            grad,
            lr,
            moment1,
            moment2,
            None,  # moment2_max
            beta1_pow,
            beta2_pow,
            master_weight,
            None,  # found_inf
            self._beta1,
            self._beta2,
            self._epsilon,
            lr_ratio,
            self._wd,
            with_decay,
            False,  # lazy_mode
            1000,
            find_master,
            False,
            False,  # amsgrad
        )


register_inner_optimizer("adamw", AdamWOptimizer)
