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
"""Learning-rate schedules shared by the trainers.

Every schedule is a ``BaseLRScheduler`` subclass registered by name
(``register_lr_scheduler``); ``cosine`` / ``wsd`` / ``power`` are built in.
``ChainedLRScheduler`` fans one ``.step()`` out to several of them, which is
what a mixed optimizer (one schedule per parameter group) needs;
``parse_lr_scheduler_spec`` parses the per-group spec string from the yaml and
``build_group_scheduler`` turns one parsed spec into a scheduler.

Adding a schedule = adding ``<name>_lr.py`` here: every ``*_lr.py`` module of
this package is imported below, which runs its ``register_lr_scheduler``.
"""

import importlib
import pkgutil

from .base import (
    LR_SCHEDULER_REGISTRY,
    BaseLRScheduler,
    get_lr_scheduler,
    register_lr_scheduler,
)
from .chained_lr import (
    ChainedLRScheduler,
    build_group_scheduler,
    parse_lr_scheduler_spec,
)
from .cosine_lr import CosineLR, get_cosine_schedule_with_warmup
from .power_lr import PowerLR, get_power_schedule_with_warmup
from .wsd_lr import WSDLR, get_wsd_schedule_with_warmup

for _mod in pkgutil.iter_modules(__path__):
    if _mod.name.endswith("_lr"):
        importlib.import_module(f"{__name__}.{_mod.name}")

__all__ = [
    "LR_SCHEDULER_REGISTRY",
    "WSDLR",
    "BaseLRScheduler",
    "ChainedLRScheduler",
    "CosineLR",
    "PowerLR",
    "build_group_scheduler",
    "get_cosine_schedule_with_warmup",
    "get_lr_scheduler",
    "get_power_schedule_with_warmup",
    "get_wsd_schedule_with_warmup",
    "parse_lr_scheduler_spec",
    "register_lr_scheduler",
]
