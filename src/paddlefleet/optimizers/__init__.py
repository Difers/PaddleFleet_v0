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
"""Inner optimizers for the mixed optimizer (``use_mixed_optimizer: true``).

Each is a from-scratch ``InnerOptimizer`` subclass (NOT a wrapper around
``paddle.optimizer.Muon``); importing this package registers ``muon`` /
``adamw`` / ``rmsprop`` (torch.optim.RMSprop port) and their Hyperball variants
``muonh`` / ``adamh`` / ``rmsproph`` via ``register_inner_optimizer``. Each
variant reuses its parent's direction and only replaces the final apply with
``hyperball.hyperball_apply``.

The sharding shell for a ``MixedInnerOptimizer`` is ``MixedShardingOptimizer``
(``paddle.distributed.fleet.meta_optimizers.mixed_sharding_optimizer``).
"""

# The optimizers subclass ``paddle.optimizer.inner_optimizer.InnerOptimizer``,
# which only exists in a Paddle build carrying the inner-optimizer API. On an
# older Paddle this package still imports (``muon_math`` / ``hyperball`` are
# plain Paddle ops) but registers nothing.
try:
    import paddle.optimizer.mixed_inner_optimizer  # noqa: F401
except (ImportError, ModuleNotFoundError):  # pragma: no cover
    __all__ = []
else:
    from .adam_hyperball_optimizer import AdamHyperballOptimizer
    from .adamw_optimizer import AdamWOptimizer
    from .muon_hyperball_optimizer import MuonHyperballOptimizer
    from .muon_optimizer import MuonOptimizer
    from .rmsprop_hyperball_optimizer import RMSpropHyperballOptimizer
    from .rmsprop_optimizer import RMSpropOptimizer

    __all__ = [
        "MuonOptimizer",
        "MuonHyperballOptimizer",
        "AdamWOptimizer",
        "AdamHyperballOptimizer",
        "RMSpropOptimizer",
        "RMSpropHyperballOptimizer",
    ]
