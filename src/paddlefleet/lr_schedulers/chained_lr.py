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
"""Per-group LR schedulers for the mixed (grouped) optimizer.

The trainer steps ONE ``self.lr_scheduler`` per training step, and each
sub-optimizer reads its own ``_learning_rate()``. So each group gets its OWN
scheduler object and a single ``ChainedLRScheduler`` composite is installed as
``self.lr_scheduler``:

    self.lr_scheduler = ChainedLRScheduler([sched_group0, sched_group1, ...])

- The trainer's single ``self.lr_scheduler.step()`` fans out to every group
  scheduler, so all groups advance together.
- Each sub-optimizer is given its OWN group scheduler as ``learning_rate`` (NOT
  the composite), so ``sub._learning_rate()`` returns that group's current lr
  with that group's shape.
- ``state_dict`` / ``set_state_dict`` cover every group scheduler, so the one
  scheduler-state the trainer/optimizer saves restores all of them (the group
  scheduler objects are shared with the sub-optimizers, so they stay in sync).
"""

from __future__ import annotations

from paddle.optimizer.lr import LRScheduler

from .base import get_lr_scheduler

__all__ = [
    "ChainedLRScheduler",
    "build_group_scheduler",
    "parse_lr_scheduler_spec",
]


def parse_lr_scheduler_spec(spec: str, **group_kwargs) -> tuple[str, dict]:
    """Strictly parse an lr-scheduler spec (+ group keys) into ``(kind, params)``.

    ``kind`` is a registered scheduler name (``register_lr_scheduler``); the
    ``:``-separated fields after it and ``group_kwargs`` are resolved by that
    class's ``get_param()`` (``BaseLRScheduler.parse_params``). Built-in::

        cosine
        wsd[:<num_steady_steps>]                 (+ decay_function key)
        power[:<power>[:<num_stable_steps>]]

    Anything else raises ``ValueError``, so a malformed spec fails loudly at
    config time instead of silently changing the schedule.
    """
    s = str(spec)
    kind, *fields = s.split(":")
    return kind, get_lr_scheduler(kind).parse_params(s, fields, group_kwargs)


def build_group_scheduler(
    *,
    lr_scheduler: str,
    learning_rate: float,
    warmup_steps: int,
    max_steps: int,
    min_lr: float = 0.0,
    **group_kwargs,
):
    """Build ONE group's LRScheduler from its spec (any registered name).

    ``learning_rate`` is this group's PEAK lr. ``group_kwargs`` must be names
    the class declares in ``get_param()``; anything else raises ``ValueError``.
    """
    kind, params = parse_lr_scheduler_spec(lr_scheduler, **group_kwargs)
    return get_lr_scheduler(kind)(
        learning_rate,
        int(warmup_steps or 0),
        max_steps,
        min_lr=(min_lr or 0.0),
        **params,
    )


class ChainedLRScheduler(LRScheduler):
    """Fan a single ``.step()`` out to N per-group LRSchedulers.

    Installed as the trainer's ``self.lr_scheduler``. Each group scheduler is
    ALSO handed to its group's sub-optimizer as ``learning_rate`` (shared
    object), so stepping here advances the sub-optimizers' lr too. ``get_lr`` /
    ``__call__`` return group 0's current lr purely for logging.
    """

    def __init__(self, schedulers: list) -> None:
        self._schedulers = list(schedulers)
        # state_dict keys: the group index as a string.
        self._names = [str(i) for i in range(len(self._schedulers))]
        # Mirror the representative (group 0) so the standard LRScheduler
        # attributes exist WITHOUT calling super().__init__ (which would run an
        # extra step() and double-advance the already-initialised sub schedulers).
        rep = self._schedulers[0]
        self.base_lr = rep.base_lr
        self.last_lr = rep.last_lr
        self.last_epoch = rep.last_epoch
        self.verbose = False

    def step(self, epoch: int | None = None) -> None:
        for s in self._schedulers:
            s.step(epoch)
        rep = self._schedulers[0]
        self.last_epoch = rep.last_epoch
        self.last_lr = rep.last_lr

    def get_lr(self) -> float:
        return self._schedulers[0].get_lr()

    def __call__(self) -> float:
        return self._schedulers[0]()

    def state_dict(self) -> dict:
        return {
            name: s.state_dict()
            for name, s in zip(self._names, self._schedulers)
        }

    def set_state_dict(self, state_dict: dict) -> None:
        if set(state_dict) != set(self._names):
            raise ValueError(
                f"ChainedLRScheduler groups {sorted(self._names)} do not match "
                f"the checkpoint's {sorted(state_dict)}."
            )
        for name, s in zip(self._names, self._schedulers):
            s.set_state_dict(state_dict[name])
        rep = self._schedulers[0]
        self.last_epoch = rep.last_epoch
        self.last_lr = rep.last_lr

    # paddle uses set_dict as an alias of set_state_dict in some call sites.
    set_dict = set_state_dict
