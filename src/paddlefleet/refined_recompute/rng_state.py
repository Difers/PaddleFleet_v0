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
"""Replay the random-state advance of a boundary the recompute pass skips.
``capture`` right after the boundary in the first pass and ``restore`` in its
place in the second pass put the generators where the first pass left them.
Both touch only the generator part of what ``RecomputeFunction`` saves and
restores for the segment (the default generators and the named states of
Paddle's tracker), so the write is undone with the rest when the segment's
recompute context exits.

Nothing else is recorded, so a boundary must draw only from those: not from
numpy or ``random`` (``RecomputeFunction`` restores them, this does not),
PaddleFleet's own ``CudaRNGStatesTracker`` (its states live in a Python dict
that ``RecomputeFunction`` does not save), or a custom state.

Only valid inside a recompute pass that restores the random state
(``preserve_rng_state=True``); a boundary that draws random numbers is not
recomputable without it in the first place.
"""

import paddle
from paddle.distributed.fleet.layers.mpu.random import get_rng_state_tracker

__all__ = ["capture", "restore"]


def capture():
    """Snapshot the generators a boundary could have advanced."""
    tracker = get_rng_state_tracker()
    return (
        paddle.get_rng_state(),
        tracker.get_states_tracker() if tracker.states_ else None,
    )


def restore(state):
    """Put the generators back to a ``capture`` snapshot."""
    generators, tracked = state
    paddle.set_rng_state(generators)
    if tracked is not None:
        get_rng_state_tracker().set_states_tracker(tracked)
