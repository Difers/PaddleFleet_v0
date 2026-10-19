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
"""``AdamHyperballOptimizer`` -- AdamW direction + Hyperball projection.

Advances the same moments as :class:`AdamWOptimizer`, forms
``u = m / (sqrt(v) + eps)``, then projects onto the fixed Frobenius sphere. The
bias-correction scalar is washed out by the Frobenius normalisation, so it is
not applied to the direction (``beta*_pow`` are still advanced for state
consistency). No weight decay (the constraint replaces it). Needs the whole
matrix (V1). Registered as ``adamh``.
"""

from __future__ import annotations

import paddle
from paddle.optimizer.mixed_inner_optimizer import register_inner_optimizer

from .adamw_optimizer import AdamWOptimizer
from .hyperball import hyperball_step

__all__ = ["AdamHyperballOptimizer"]


class AdamHyperballOptimizer(AdamWOptimizer):
    """AdamW preconditioned direction + Hyperball projection (the "AdamH" mode).

    Overrides the update path: AdamW's fused kernel does not expose the
    direction ``u`` that the projection needs.
    """

    # The Frobenius norm of the whole matrix drives the projection, so unlike
    # plain AdamW this cannot run on an element-wise shard (V1).
    keep_whole = True

    def _update_parameters(self, params_grads, lr):
        # fp32 lr tensor, exactly like paddle.optimizer.Muon's AdamH path, so
        # lr * lr_ratio rounds in fp32 (bit-identical to the single-Muon path).
        lr = paddle.to_tensor(lr, dtype=paddle.float32)
        for param, grad in params_grads:
            self._adamh_one(param, grad, lr)

    def _adamh_one(self, param, grad, lr):
        moment1, moment2, beta1_pow, beta2_pow = self._accs(param)
        with paddle.no_grad():
            g = grad.astype(paddle.float32)
            paddle.assign(
                self._beta1 * moment1 + (1.0 - self._beta1) * g, moment1
            )
            paddle.assign(
                self._beta2 * moment2 + (1.0 - self._beta2) * g * g, moment2
            )
            paddle.assign(beta1_pow * self._beta1, beta1_pow)
            paddle.assign(beta2_pow * self._beta2, beta2_pow)

            update = moment1 / (paddle.sqrt(moment2) + self._epsilon)
            # Slices of a split param are projected independently.
            info = self._param_info_map.get(param.name)
            hyperball_step(
                self,
                param,
                update,
                lr,
                info.split_concat_func if info is not None else None,
            )


register_inner_optimizer("adamh", AdamHyperballOptimizer)
