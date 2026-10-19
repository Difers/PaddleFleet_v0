# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

"""Warmup -> stable -> decay (WSD) lr schedule.

Spec: ``wsd[:<num_steady_steps>]``; ``decay_function`` as a group key.
"""

import math

from .base import BaseLRScheduler, parse_nonneg_int, register_lr_scheduler

_DECAY_FUNCTIONS = ("half_life", "1-sqrt")


def _parse_decay_function(text, spec, field):
    if text not in _DECAY_FUNCTIONS:
        raise ValueError(
            f"lr_scheduler {spec!r}: {field} must be one of "
            f"{list(_DECAY_FUNCTIONS)}, got {text!r}."
        )
    return text


class WSDLR(BaseLRScheduler):
    """Linear warmup, hold at the peak until ``num_steady_steps``, then decay
    (``half_life`` or ``1-sqrt``) to ``min_lr``."""

    def __init__(
        self,
        learning_rate: float,
        num_warmup_steps: int,
        num_training_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1,
        decay_function: str = "half_life",
        num_steady_steps=None,
    ):
        if num_steady_steps is None:
            num_steady_steps = 0.9 * num_training_steps
        self.decay_function = decay_function
        self.num_steady_steps = num_steady_steps
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
            "num_steady_steps": parse_nonneg_int,
            "decay_function": _parse_decay_function,
        }

    def schedule(self, current_step, base=0.05):
        num_warmup_steps = self.num_warmup_steps
        num_steady_steps = self.num_steady_steps
        min_lr, learning_rate = self.min_lr, self.learning_rate
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        if current_step < num_steady_steps:
            return 1.0
        num_decay_steps = self.num_training_steps - num_steady_steps
        progress = float(current_step - num_steady_steps) / float(
            max(1, num_decay_steps)
        )

        if self.decay_function == "half_life":
            ratio = base**progress
            normalize_ratio = (ratio - base) * (1 / (1 - base))
            return (
                normalize_ratio * (1 - min_lr / learning_rate)
                + min_lr / learning_rate
            )
        elif self.decay_function == "1-sqrt":
            ratio = 1 - math.sqrt(progress)
            return ratio * (1 - min_lr / learning_rate) + min_lr / learning_rate
        else:
            raise ValueError(f"Invalid decay function: {self.decay_function}")


register_lr_scheduler("wsd", WSDLR)


def get_wsd_schedule_with_warmup(
    learning_rate: float,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    decay_function: str = "half_life",
    last_epoch: int = -1,
    min_lr: float = 0.0,
    num_steady_steps=None,
):
    """Build a ``WSDLR``. ``num_cycles`` is accepted for compatibility and unused."""
    return WSDLR(
        learning_rate,
        num_warmup_steps,
        num_training_steps,
        min_lr=min_lr,
        last_epoch=last_epoch,
        decay_function=decay_function,
        num_steady_steps=num_steady_steps,
    )
