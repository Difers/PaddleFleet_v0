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
"""``MuonHyperballOptimizer`` -- Muon direction + Hyperball projection.

Same direction as :class:`MuonOptimizer`; only the final apply differs: the
weight is projected back onto the fixed Frobenius sphere ``||W||_F = R``
instead of weight decay + subtract (so ``weight_decay`` is ignored). Registered
as ``muonh``.
"""

from __future__ import annotations

from paddle.optimizer.mixed_inner_optimizer import register_inner_optimizer

from .hyperball import hyperball_step
from .muon_optimizer import MuonOptimizer

__all__ = ["MuonHyperballOptimizer"]


class MuonHyperballOptimizer(MuonOptimizer):
    """Muon direction + Hyperball projection onto the fixed Frobenius sphere."""

    def _apply_update(
        self, param, orthogonal_update, find_master, lr, split_concat_func=None
    ):
        # Slices of a split param are projected independently.
        hyperball_step(self, param, orthogonal_update, lr, split_concat_func)


register_inner_optimizer("muonh", MuonHyperballOptimizer)
