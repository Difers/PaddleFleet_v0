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
"""Warmup -> hold -> power-law-decay learning-rate schedule.

Ported from the ``MuonH`` schedule of modded-nanogpt PR #345
(https://github.com/KellerJordan/modded-nanogpt/pull/345), which reached the
Track-3 target in fewer steps than the previous linear cooldown. The shape is::

    warmup:  linear from ``min_lr`` up to the peak (``learning_rate``)
    hold:    stay at the peak for ``num_stable_steps`` steps
    decay:   floor + (peak - floor) * (1 - progress ** power)
    after:   stay at ``min_lr`` (the floor)

With ``power < 1`` the decay leaves the peak quickly and spends most of the run
near the floor (a "fast-then-slow" cooldown); ``power = 1`` is a straight line.
PR #345 uses ``power = 0.4``. The reference exactly corresponds to
``num_warmup_steps=100``, ``num_stable_steps=100``, ``num_training_steps=3200``,
``min_lr=1e-5``, peak ``0.03``.

Implemented as a ``BaseLRScheduler`` (a ``LambdaDecay``) returning a
*fraction* of ``learning_rate``. Spec: ``power[:<power>[:<num_stable_steps>]]``.
"""

from .base import (
    BaseLRScheduler,
    parse_finite_float,
    parse_nonneg_int,
    register_lr_scheduler,
)


class PowerLR(BaseLRScheduler):
    """Warmup -> hold -> power-law-decay schedule (see the module docstring)."""

    def __init__(
        self,
        learning_rate: float,
        num_warmup_steps: int,
        num_training_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1,
        power: float = 0.4,
        num_stable_steps: int = 0,
    ):
        if not power > 0:
            raise ValueError(f"PowerLR power must be > 0, got {power!r}.")
        peak = float(learning_rate)
        # Floor as a fraction of the peak (LambdaDecay multiplies the peak by this).
        self.floor_ratio = (min_lr / peak) if peak > 0 else 0.0
        self.warmup = int(num_warmup_steps or 0)
        stable = int(num_stable_steps or 0)
        self.power = power
        # Last step that is still held at the peak (mirrors the PR's boundary math:
        # warmup reaches the peak on its final step, then the hold spans the next
        # ``stable`` steps). Decay runs from here to ``num_training_steps``.
        # Clamp to >= 0 so the "no warmup, no hold" case (warmup=0, stable=0) anchors
        # the decay at step 0 -> step 0 is the peak (progress 0), not already decaying.
        self.hold_end = max(self.warmup + stable - 1, 0)
        self.decay_span = max(1, num_training_steps - self.hold_end)
        super().__init__(
            learning_rate,
            num_warmup_steps,
            num_training_steps,
            min_lr,
            last_epoch,
        )

    @classmethod
    def get_param(cls):
        return {
            "power": parse_finite_float,
            "num_stable_steps": parse_nonneg_int,
        }

    def schedule(self, current_step):
        warmup, floor_ratio = self.warmup, self.floor_ratio
        if warmup > 0 and current_step < warmup:
            frac = current_step / max(1, warmup - 1)
            return floor_ratio + (1.0 - floor_ratio) * min(1.0, frac)
        if current_step <= self.hold_end:
            return 1.0
        if current_step <= self.num_training_steps:
            progress = (current_step - self.hold_end) / self.decay_span
            return floor_ratio + (1.0 - floor_ratio) * (
                1.0 - progress**self.power
            )
        return floor_ratio


register_lr_scheduler("power", PowerLR)


def get_power_schedule_with_warmup(
    learning_rate: float,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr: float = 0.0,
    power: float = 0.4,
    num_stable_steps: int = 0,
    last_epoch: int = -1,
):
    """Build a warmup -> hold -> power-law-decay ``PowerLR`` scheduler.

    Args:
        learning_rate: Peak learning rate.
        num_warmup_steps: Linear warmup length (floor -> peak).
        num_training_steps: Step at which lr reaches the floor (decay end).
        min_lr: Absolute floor lr (both the warmup start and the post-decay lr).
        power: Decay exponent. ``< 1`` leaves the peak fast; ``1`` is linear.
        num_stable_steps: Hold-at-peak length after warmup (0 = no hold).
        last_epoch: Passed through to ``LambdaDecay``.
    """
    return PowerLR(
        learning_rate,
        num_warmup_steps,
        num_training_steps,
        min_lr=min_lr,
        last_epoch=last_epoch,
        power=power,
        num_stable_steps=num_stable_steps,
    )
