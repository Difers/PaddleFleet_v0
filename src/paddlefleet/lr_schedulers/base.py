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
"""``BaseLRScheduler`` and the lr-scheduler registry.

A custom schedule is ONE new file ``paddlefleet/lr_schedulers/<name>_lr.py``
(the package imports every ``*_lr.py`` module, which runs its registration)::

    from .base import BaseLRScheduler, parse_finite_float, register_lr_scheduler

    class FooLR(BaseLRScheduler):
        def __init__(self, learning_rate, num_warmup_steps,
                     num_training_steps, min_lr=0.0, last_epoch=-1, k=1.0):
            self.k = k                      # own fields BEFORE super().__init__
            super().__init__(learning_rate, num_warmup_steps,
                             num_training_steps, min_lr, last_epoch)

        def schedule(self, step):           # REQUIRED
            return ...                      # fraction of the peak lr

        @classmethod
        def get_param(cls):                 # REQUIRED: own ctor kwargs -> parser
            return {"k": parse_finite_float}  # {} = none

    register_lr_scheduler("foo", FooLR)

A yaml group then writes ``lr_scheduler: foo``, ``foo:<k>`` or ``foo`` plus a
``k: <k>`` key.
"""

from __future__ import annotations

import math
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Any

from paddle.optimizer.lr import LambdaDecay

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "LR_SCHEDULER_REGISTRY",
    "BaseLRScheduler",
    "get_lr_scheduler",
    "parse_finite_float",
    "parse_nonneg_int",
    "register_lr_scheduler",
]


class BaseLRScheduler(LambdaDecay, metaclass=ABCMeta):
    """Base class of the warmup-style schedules (``cosine`` / ``wsd`` / ...).

    The emitted lr is ``learning_rate * schedule(step)``. It IS a
    ``LambdaDecay`` (``lr_lambda`` = ``schedule``), so the math, the
    ``state_dict`` keys and checkpoints match the old ``LambdaDecay`` builders.

    Subclasses MUST implement ``schedule`` and ``get_param``.
    """

    def __init__(
        self,
        learning_rate: float,
        num_warmup_steps: int,
        num_training_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        # LambdaDecay.__init__ already calls schedule (via step()), so
        # every field it reads must be set before this super().__init__.
        self.learning_rate = learning_rate
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.min_lr = min_lr
        super().__init__(learning_rate, self.schedule, last_epoch)

    @abstractmethod
    def schedule(self, step: int) -> float:
        """Fraction of the peak ``learning_rate`` at ``step``."""

    @classmethod
    @abstractmethod
    def get_param(cls) -> dict[str, Callable[[Any, str, str], Any]]:
        """Scheduler-specific ``__init__`` kwargs as an ordered
        ``{name: parser}`` map, besides the shared ``learning_rate`` /
        ``warmup_steps`` / ``min_lr``; ``{}`` = none.

        A yaml group sets each either positionally after the name (in this
        order: ``wsd:95073`` -> ``num_steady_steps``) or as its own key, not
        both; ``parser(value, spec, name)`` converts and validates it. An
        unset one keeps the ``__init__`` default."""

    @classmethod
    def parse_params(
        cls, spec: str, fields: list[str], group_kwargs: dict
    ) -> dict:
        """Resolve the spec ``fields`` and group keys into ctor kwargs."""
        parsers = cls.get_param()
        names = list(parsers)
        if len(fields) > len(names):
            raise ValueError(
                f"lr_scheduler {spec!r} takes at most {len(names)} ':' "
                f"field(s) {names}."
            )
        unknown = sorted(set(group_kwargs) - set(names))
        if unknown:
            raise ValueError(
                f"lr_scheduler {spec!r} does not take {unknown}; its params "
                f"are {names}."
            )
        given = dict(zip(names, fields))
        both = sorted(set(given) & set(group_kwargs))
        if both:
            raise ValueError(
                f"lr_scheduler {spec!r}: {both} set both in the spec and as "
                "group keys."
            )
        given.update(group_kwargs)
        return {k: parsers[k](v, spec, k) for k, v in given.items()}


def parse_nonneg_int(text: Any, spec: str, field: str) -> int:
    """Parse a spec field / group value as an integer ``>= 0``."""
    try:
        # via str so a yaml float (0.5) or bool is rejected, not truncated
        value = int(str(text))
    except (TypeError, ValueError):
        raise ValueError(
            f"lr_scheduler {spec!r}: {field} must be a non-negative integer, "
            f"got {text!r}."
        )
    if value < 0:
        raise ValueError(
            f"lr_scheduler {spec!r}: {field} must be >= 0, got {value}."
        )
    return value


def parse_finite_float(text: Any, spec: str, field: str) -> float:
    """Parse a spec field / group value as a finite float."""
    try:
        value = float(str(text))
    except (TypeError, ValueError):
        raise ValueError(
            f"lr_scheduler {spec!r}: {field} must be a finite float, got {text!r}."
        )
    if not math.isfinite(value):
        raise ValueError(
            f"lr_scheduler {spec!r}: {field} must be finite, got {value}."
        )
    return value


LR_SCHEDULER_REGISTRY: dict[str, type] = {}


def register_lr_scheduler(name: str, cls: type) -> None:
    """Register *cls* (a ``BaseLRScheduler`` subclass) under *name*.

    Re-registering the same class is a no-op; a different class raises.
    """
    if not isinstance(name, str) or not name or ":" in name:
        raise ValueError(
            "lr scheduler name must be a non-empty string without ':'."
        )
    if not (isinstance(cls, type) and issubclass(cls, BaseLRScheduler)):
        raise TypeError(
            f"lr scheduler {name!r} must be registered with a "
            f"BaseLRScheduler subclass, got {cls!r}."
        )
    existing = LR_SCHEDULER_REGISTRY.get(name)
    if existing is not None and existing is not cls:
        raise ValueError(
            f"lr scheduler name {name!r} is already registered to "
            f"{existing!r}; refusing to shadow it with {cls!r}."
        )
    LR_SCHEDULER_REGISTRY[name] = cls


def get_lr_scheduler(name: str) -> type:
    """Return the ``BaseLRScheduler`` subclass registered under *name*."""
    try:
        return LR_SCHEDULER_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unsupported lr_scheduler {name!r}; registered: "
            f"{sorted(LR_SCHEDULER_REGISTRY)}."
        ) from None
