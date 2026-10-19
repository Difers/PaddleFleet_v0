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
"""RMSpropOptimizer vs torch.optim.RMSprop, and its Hyperball variant (rmsproph)
vs a numpy reference.

Run:  python -m pytest tests/single_card_tests/test_rmsprop_optimizer.py -v
"""

import itertools
import unittest

import numpy as np
import paddle

import paddlefleet.optimizers  # noqa: F401  (registers rmsprop / rmsproph)

try:
    from paddle.optimizer.mixed_inner_optimizer import get_inner_optimizer
except (ImportError, ModuleNotFoundError):
    # Paddle build without the inner-optimizer API; nothing to register against.
    get_inner_optimizer = None

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

STEPS = 20
LR = 1e-2


def _run_paddle(w0, grads, **hp):
    p = paddle.create_parameter(
        shape=list(w0.shape),
        dtype="float32",
        default_initializer=paddle.nn.initializer.Assign(w0),
    )
    # weight_decay is required (no default in the optimizer); the tests that
    # care pass their own value through **hp.
    hp.setdefault("weight_decay", 0.0)
    name = hp.pop("name", "rmsprop")
    opt = get_inner_optimizer(name)(parameters=[p], learning_rate=LR, **hp)
    for g in grads:
        opt._apply_optimize(None, None, [(p, paddle.to_tensor(g))])
    return p.numpy(), opt, p


def _run_torch(w0, grads, **hp):
    hp = dict(hp)
    if "epsilon" in hp:
        hp["eps"] = hp.pop("epsilon")
    p = torch.nn.Parameter(torch.tensor(w0))
    opt = torch.optim.RMSprop([p], lr=LR, foreach=False, **hp)
    for g in grads:
        p.grad = torch.tensor(g)
        opt.step()
    return p.detach().numpy()


@unittest.skipIf(
    get_inner_optimizer is None, "Paddle build without the inner-optimizer API"
)
@unittest.skipIf(torch is None, "torch not installed")
class TestRMSpropVsTorch(unittest.TestCase):
    def setUp(self):
        paddle.set_device("cpu")
        rng = np.random.default_rng(0)
        self.w0 = rng.standard_normal((16, 8)).astype("float32")
        self.grads = [
            rng.standard_normal((16, 8)).astype("float32") for _ in range(STEPS)
        ]

    def test_all_combinations(self):
        for momentum, centered, wd, maximize in itertools.product(
            [0.0, 0.9], [False, True], [0.0, 0.1], [False, True]
        ):
            # Non-default alpha/eps ride along on the centered half.
            alpha, eps = (0.9, 1e-3) if centered else (0.99, 1e-8)
            hp = {
                "alpha": alpha,
                "epsilon": eps,
                "momentum": momentum,
                "centered": centered,
                "weight_decay": wd,
                "maximize": maximize,
            }
            with self.subTest(**hp):
                ours, _, _ = _run_paddle(self.w0, self.grads, **hp)
                ref = _run_torch(self.w0, self.grads, **hp)
                np.testing.assert_allclose(ours, ref, rtol=1e-5, atol=1e-6)


@unittest.skipIf(
    get_inner_optimizer is None, "Paddle build without the inner-optimizer API"
)
class TestRMSpropStateLayout(unittest.TestCase):
    def setUp(self):
        paddle.set_device("cpu")
        self.w0 = np.ones((4, 4), dtype="float32")
        self.grads = [np.full((4, 4), 0.5, dtype="float32")]

    def _acc_suffixes(self, opt, p):
        return sorted(
            accs[p.name].name[len(p.name) + 1 :]
            for accs in opt._accumulators.values()
            if p.name in accs
        )

    def test_layout_matches_accumulators(self):
        for momentum, centered in itertools.product([0.0, 0.9], [False, True]):
            with self.subTest(momentum=momentum, centered=centered):
                _, opt, p = _run_paddle(
                    self.w0, self.grads, momentum=momentum, centered=centered
                )
                layout = opt.optimizer_state_layout()
                self.assertEqual(layout["scalar"], [])
                self.assertEqual(layout["master"], "fp32_master_0")
                self.assertEqual(
                    sorted(layout["vector"]), self._acc_suffixes(opt, p)
                )

    def test_apply_decay_param_fun_excludes(self):
        # Excluded param with wd>0 must match the wd=0 trajectory.
        ex, _, _ = _run_paddle(
            self.w0,
            self.grads,
            weight_decay=0.1,
            apply_decay_param_fun=lambda n: False,
        )
        ref, _, _ = _run_paddle(self.w0, self.grads, weight_decay=0.0)
        np.testing.assert_array_equal(ex, ref)


def _ref_rmsproph(w0, grads, alpha, epsilon, momentum, centered):
    """numpy: torch's RMSprop direction, then the Hyperball projection."""

    def fro(x):
        return np.sqrt((x * x).sum(axis=(-2, -1), keepdims=True))

    w = w0.copy()
    v, gbar, buf = np.zeros_like(w), np.zeros_like(w), np.zeros_like(w)
    for g in grads:
        v = alpha * v + (1 - alpha) * g * g
        if centered:
            gbar = gbar + (1 - alpha) * (g - gbar)
            avg = np.sqrt(v - gbar * gbar) + epsilon
        else:
            avg = np.sqrt(v) + epsilon
        buf = momentum * buf + g / avg
        u = buf if momentum > 0 else g / avg
        r = fro(w)
        trial = w - LR * r * u / (fro(u) + 1e-12)
        w = r * trial / (fro(trial) + 1e-12)
    return w


@unittest.skipIf(
    get_inner_optimizer is None, "Paddle build without the inner-optimizer API"
)
class TestRMSpropHyperball(unittest.TestCase):
    def setUp(self):
        paddle.set_device("cpu")

    def test_matches_reference(self):
        rng = np.random.default_rng(0)
        for shape, momentum, centered in itertools.product(
            [(16, 8), (3, 8, 4)], [0.0, 0.9], [False, True]
        ):
            w0 = rng.standard_normal(shape).astype("float32")
            grads = [
                rng.standard_normal(shape).astype("float32")
                for _ in range(STEPS)
            ]
            hp = {
                "alpha": 0.9,
                "epsilon": 1e-3,
                "momentum": momentum,
                "centered": centered,
            }
            with self.subTest(shape=shape, **hp):
                # weight_decay is ignored: the sphere replaces it.
                ours, opt, _ = _run_paddle(
                    w0, grads, name="rmsproph", weight_decay=0.1, **hp
                )
                ref = _ref_rmsproph(w0, grads, **hp)
                np.testing.assert_allclose(ours, ref, rtol=1e-5, atol=1e-6)
                # Each matrix (each expert for 3D) keeps its initial norm.
                mats = (-1, *shape[-2:])
                np.testing.assert_allclose(
                    np.linalg.norm(ours.reshape(mats), axis=(1, 2)),
                    np.linalg.norm(w0.reshape(mats), axis=(1, 2)),
                    rtol=1e-5,
                )
                self.assertIs(opt.declared_keep_whole(), True)


if __name__ == "__main__":
    unittest.main()
