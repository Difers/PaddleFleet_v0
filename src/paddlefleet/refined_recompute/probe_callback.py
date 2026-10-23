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
"""Open the probe's window for a few optimizer steps, then write the result."""

from paddlefleet.refined_recompute.probe import PROBE
from paddlefleet.trainer.trainer_callback import TrainerCallback

__all__ = ["ProbeWindowCallback"]


def window(args):
    """``(first, last)``: price optimizer steps ``[first, last)``."""
    calibration = (args.auto_refined_recompute or {}).get("calibration") or {}
    first = int(calibration.get("warmup_steps", 3))
    return first, first + int(calibration.get("measure_steps", 12))


class ProbeWindowCallback(TrainerCallback):
    """Price the optimizer steps of ``auto_refined_recompute.calibration``.

    Counting optimizer steps here, not calls, keeps the window right under
    gradient accumulation: ``on_step_begin`` fires once per optimizer step.
    """

    def on_step_begin(self, args, state, control, **kwargs):
        first, last = window(args)
        # ``global_step`` counts the optimizer steps finished so far.
        if first <= state.global_step < last:
            PROBE.begin_step(state.global_step)

    def on_step_end(self, args, state, control, **kwargs):
        PROBE.end_step()
        if state.global_step >= window(args)[1]:
            PROBE.flush()

    def on_train_end(self, args, state, control, **kwargs):
        # A run shorter than the window still writes what it recorded.
        PROBE.end_step()
        PROBE.flush()
