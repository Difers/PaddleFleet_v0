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
"""muon / muonh / adamw / adamh in a ``MixedInnerOptimizer`` vs one
``paddle.optimizer.Muon`` running the same four modes.

The PaddleFleet optimizers re-implement Muon's math, so with the same
hyperparameters they must be BIT-identical to it: same-shape batching, 3D
grouped experts, a 1D AdamW param, per-group lr (``lr_ratio``), a warmup
schedule and one global grad clip. fp32 Newton-Schulz on CPU = deterministic.

Run:  python -m pytest tests/single_card_tests/test_inner_optimizers_vs_muon.py -v
"""

import unittest
from collections import defaultdict

import numpy as np
import paddle

import paddlefleet.optimizers  # noqa: F401  (registers the optimizers)

try:
    from paddle.optimizer.mixed_inner_optimizer import (
        MixedInnerOptimizer,
        get_inner_optimizer,
    )
    from paddle.optimizer.muon import Muon, MuonParamInfo
except (ImportError, ModuleNotFoundError):
    # Paddle build without the inner-optimizer API.
    MixedInnerOptimizer = None

STEPS = 30
BASE_LR = 1.17e-2
WD = 0.1
ADAM_HP = {"adam_beta1": 0.9, "adam_beta2": 0.95, "adam_epsilon": 1e-15}
MUON_HP = {
    "momentum": 0.95,
    "muon_epsilon": 1e-7,
    "ns_coeff_type": "quintic",
    "muon_extra_scale_factor": 0.2,
    "muon_version": 3,
    "ns_matmul_dtype": paddle.float32,
}
# group -> (per-group lr, param shapes)
GROUPS = {
    "muon": (4.5e-3, [(64, 96), (64, 96), (4, 32, 48)]),
    "muonh": (3.0e-3, [(96, 64), (96, 64), (2, 48, 32)]),
    "adamw": (1.0e-3, [(32, 40), (40,)]),
    "adamh": (7.0e-3, [(48, 80)]),
}


def _run(build):
    paddle.seed(0)
    params, group_of = [], {}
    for g, (_, shapes) in GROUPS.items():
        for shape in shapes:
            p = paddle.create_parameter(list(shape), "float32")
            p.set_value(paddle.randn(list(shape)))
            params.append(p)
            group_of[p.name] = g
    sched = paddle.optimizer.lr.LambdaDecay(
        BASE_LR, lambda s: min(1.0, (s + 1) / 750)
    )
    lr_ratio = lambda p: GROUPS[group_of[p.name]][0] / BASE_LR
    opt = build(params, group_of, sched, lr_ratio)
    for step in range(STEPS):
        paddle.seed(1000 + step)
        grads = [paddle.randn(p.shape) * 3.0 for p in params]
        opt._apply_optimize(None, None, list(zip(params, grads)))
        sched.step()
    return [(group_of[p.name], p.numpy()) for p in params]


def _single_muon(params, group_of, sched, lr_ratio):
    info = {}
    for p in params:
        g = group_of[p.name]
        info[p.name] = MuonParamInfo(use_muon=g in ("muon", "muonh"))
        info[p.name].use_hyperball = g in ("muonh", "adamh")
    return Muon(
        learning_rate=sched,
        parameters=params,
        weight_decay=WD,
        muon_param_info_map=info,
        lr_ratio=lr_ratio,
        grad_clip=paddle.nn.ClipGradByGlobalNorm(1.0),
        **ADAM_HP,
        **MUON_HP,
    )


def _mixed(params, group_of, sched, lr_ratio):
    by_group = defaultdict(list)
    for p in params:
        by_group[group_of[p.name]].append(p)
    subs = [
        # The same wd for every group: muonh / adamh must ignore it.
        get_inner_optimizer(g)(
            parameters=by_group[g],
            learning_rate=sched,
            weight_decay=WD,
            lr_ratio=lr_ratio,
            **(MUON_HP if g.startswith("muon") else ADAM_HP),
        )
        for g in GROUPS
    ]
    return MixedInnerOptimizer(
        subs,
        learning_rate=sched,
        grad_clip=paddle.nn.ClipGradByGlobalNorm(1.0),
    )


@unittest.skipIf(
    MixedInnerOptimizer is None, "Paddle build without the inner-optimizer API"
)
class TestInnerOptimizersVsMuon(unittest.TestCase):
    def setUp(self):
        paddle.set_device("cpu")

    def test_bit_identical(self):
        ref, ours = _run(_single_muon), _run(_mixed)
        for (g, x), (_, y) in zip(ref, ours, strict=True):
            with self.subTest(group=g, shape=x.shape):
                np.testing.assert_array_equal(y, x)


if __name__ == "__main__":
    unittest.main()
