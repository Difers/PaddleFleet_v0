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
"""``RMSpropOptimizer`` -- port of ``torch.optim.RMSprop`` (torch 2.14).

A CUSTOM registered ``InnerOptimizer`` (registered as ``rmsprop``, NOT one of
the four built-in Muon-family names), so it exercises the generic
"add an optimizer without rebuilding the paddle image" path: the trainer looks
it up in the registry and reads its duck-typed ``yaml_field_map`` instead of
hard-coded branches. The sharding modes the update math supports are declared
by the class (``keep_whole = None`` -- element-wise math is correct both on a
shard and on a whole tensor); yaml may pick one of them, defaulting to V2.

Algorithm (identical to torch's ``_single_tensor_rmsprop``)::

    g_t   = -g_t if maximize else g_t
    g_t   = g_t + λ·θ_{t-1}                       if λ != 0   (coupled L2)
    v_t   = α·v_{t-1} + (1-α)·g_t²
    ṽ_t   = v_t
    if centered:
        ḡ_t = lerp(ḡ_{t-1}, g_t, 1-α)
        ṽ_t = v_t - ḡ_t²
    if μ > 0:
        b_t = μ·b_{t-1} + g_t / (√ṽ_t + ε)
        θ_t = θ_{t-1} - γ·b_t
    else:
        θ_t = θ_{t-1} - γ·g_t / (√ṽ_t + ε)

Notes vs torch:
  * Weight decay is torch's COUPLED L2 (added to the gradient), not AdamW-style
    decoupled decay; ``apply_decay_param_fun`` selects which params get it.
  * Per-param lr = ``lr * lr_ratio(param)`` (honours per-group lr).
  * Math runs in fp32 on the master weight (bf16 params keep an fp32 master).
  * torch's ``step`` counter is not stored: the update never reads it.
  * ``capturable`` / ``foreach`` / ``differentiable`` are torch execution
    details with no paddle equivalent here.

State per parameter (all fp32, same shape as the local parameter/shard):
``square_avg`` always; ``grad_avg`` iff ``centered``; ``momentum_buffer`` iff
``momentum > 0``. No replicated scalars.
"""

from __future__ import annotations

import paddle
from paddle.base import framework
from paddle.optimizer.inner_optimizer import InnerOptimizer
from paddle.optimizer.mixed_inner_optimizer import register_inner_optimizer

__all__ = ["RMSpropOptimizer"]


class RMSpropOptimizer(InnerOptimizer):
    """torch.optim.RMSprop as an ``InnerOptimizer`` (registered ``rmsprop``)."""

    _square_avg_acc_str = "square_avg"
    _grad_avg_acc_str = "grad_avg"
    _momentum_acc_str = "momentum_buffer"
    # Element-wise update: correct both on an element-wise shard (V2) and on a
    # whole tensor (V1) -> either mode may be configured, default V2.
    keep_whole = None

    def __init__(
        self,
        parameters,
        learning_rate=0.01,
        alpha=0.99,
        epsilon=1e-8,
        momentum=0.0,
        centered=False,
        maximize=False,
        weight_decay=None,  # required (InnerOptimizer rejects None)
        grad_clip=None,
        lr_ratio=None,
        apply_decay_param_fun=None,
        multi_precision=True,
        param_info_map=None,
        name=None,
    ):
        alpha, epsilon, momentum = float(alpha), float(epsilon), float(momentum)
        if weight_decay is not None:
            weight_decay = float(weight_decay)
        # Same argument checks as torch.optim.RMSprop.__init__.
        if epsilon < 0.0:
            raise ValueError(f"Invalid epsilon value: {epsilon}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay is not None and weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if alpha < 0.0:
            raise ValueError(f"Invalid alpha value: {alpha}")
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
        self._alpha = alpha
        self._epsilon = epsilon
        self._momentum = momentum
        self._centered = bool(centered)
        self._maximize = bool(maximize)
        self._wd = weight_decay

    # -- descriptors read by the framework layer (pretrain.py / trainer) ----

    @classmethod
    def yaml_field_map(cls):
        """YAML group field -> ctor kwarg (per-group hyperparameters)."""
        return {
            "alpha": "alpha",
            "epsilon": "epsilon",
            "momentum": "momentum",
            "centered": "centered",
            "maximize": "maximize",
        }

    # -- accumulators / checkpoint layout -----------------------------------

    def _state_acc_names(self):
        names = [self._square_avg_acc_str]
        if self._centered:
            names.append(self._grad_avg_acc_str)
        if self._momentum > 0:
            names.append(self._momentum_acc_str)
        return names

    def _ensure_accumulators(self, param):
        if (
            self._square_avg_acc_str in self._accumulators
            and param.name in self._accumulators[self._square_avg_acc_str]
        ):
            return
        self._ensure_master_weight(param)
        for acc_name in self._state_acc_names():
            self._add_accumulator(
                acc_name,
                param,
                dtype=paddle.float32,
                fill_value=0.0,
                shape=param.shape,
                type=framework.core.VarDesc.VarType.DENSE_TENSOR,
            )

    def optimizer_state_layout(self) -> dict:
        return {
            "vector": [f"{n}_0" for n in self._state_acc_names()],
            "scalar": [],
            "master": "fp32_master_0",
        }

    # -- update math --------------------------------------------------------

    def _update_parameters(self, params_grads, lr):
        for param, grad in params_grads:
            self._rmsprop_one(param, grad, lr)

    def _rmsprop_one(self, param, grad, lr):
        with paddle.no_grad():
            find_master = param.name in self._master_weights
            w = self._master_weights[param.name] if find_master else param
            w32 = w.astype(paddle.float32)

            g = grad.astype(paddle.float32)
            if self._maximize:
                g = -g
            if self._wd != 0.0 and (
                self._apply_decay_param_fun is None
                or self._apply_decay_param_fun(param.name)
            ):
                g = g + self._wd * w32
            new_w = w32 - self._effective_lr(param, lr) * self._direction(
                param, g
            )
            if find_master:
                paddle.assign(new_w, w)
            paddle.assign(new_w.astype(param.dtype), param)

    def _direction(self, param, g):
        """Advance the state with gradient ``g``; return the step direction."""
        square_avg = self._get_accumulator(self._square_avg_acc_str, param)
        paddle.assign(
            square_avg * self._alpha + (1.0 - self._alpha) * g * g,
            square_avg,
        )
        if self._centered:
            grad_avg = self._get_accumulator(self._grad_avg_acc_str, param)
            paddle.assign(paddle.lerp(grad_avg, g, 1.0 - self._alpha), grad_avg)
            avg = paddle.sqrt(square_avg - grad_avg * grad_avg)
        else:
            avg = paddle.sqrt(square_avg)
        avg = avg + self._epsilon
        if self._momentum > 0:
            buf = self._get_accumulator(self._momentum_acc_str, param)
            paddle.assign(buf * self._momentum + g / avg, buf)
            return buf
        return g / avg


register_inner_optimizer("rmsprop", RMSpropOptimizer)
