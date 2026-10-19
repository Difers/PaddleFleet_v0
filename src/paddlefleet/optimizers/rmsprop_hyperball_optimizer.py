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
"""``RMSpropHyperballOptimizer`` -- RMSprop direction + Hyperball projection.

Advances the same state as :class:`RMSpropOptimizer` and uses its step
direction (``g / (sqrt(v) + eps)``, or the momentum buffer), then projects onto
the fixed Frobenius sphere instead of subtracting. No weight decay (the
constraint replaces it, as for muonh / adamh). Needs the whole matrix (V1), so
its checkpoint layout differs from plain ``rmsprop`` (V2). Registered as
``rmsproph``.
"""

from __future__ import annotations

import paddle
from paddle.optimizer.mixed_inner_optimizer import register_inner_optimizer

from .hyperball import hyperball_step
from .rmsprop_optimizer import RMSpropOptimizer

__all__ = ["RMSpropHyperballOptimizer"]


class RMSpropHyperballOptimizer(RMSpropOptimizer):
    """RMSprop direction + Hyperball projection (registered ``rmsproph``)."""

    # The Frobenius norm of the whole matrix drives the projection (V1).
    keep_whole = True

    def _rmsprop_one(self, param, grad, lr):
        with paddle.no_grad():
            g = grad.astype(paddle.float32)
            if self._maximize:
                g = -g
            info = self._param_info_map.get(param.name)
            hyperball_step(
                self,
                param,
                self._direction(param, g),
                lr,
                info.split_concat_func if info is not None else None,
            )


register_inner_optimizer("rmsproph", RMSpropHyperballOptimizer)
