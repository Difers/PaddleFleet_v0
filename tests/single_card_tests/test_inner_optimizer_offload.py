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
"""Optimizer offload (``hack_offload_optimizer`` Step 5) for the mixed
optimizer's sub-optimizers.

* ``TestOffloadGranularity`` (CPU, ``reload``/``offload`` recorded): which
  states are moved, and at which granularity -- per ``_update_shape_group``
  batch when the class defines one (Muon family), otherwise one
  ``_update_parameters`` call; the ``is_offload_opt`` gate; patching twice is
  harmless.
* ``TestOffloadAccuracy`` (GPU): a ``MixedInnerOptimizer`` over all six
  registered sub-optimizers with bf16 params (fp32 master weights), run with
  and without offload. Parameters and every state tensor must be
  bit-identical, and between steps every state must live in pinned memory.

Run:  python -m pytest tests/single_card_tests/test_inner_optimizer_offload.py -v
"""

import sys
import unittest
from collections import defaultdict
from unittest import mock

import numpy as np
import paddle
from paddle import _C_ops
from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer.hybrid_parallel_optimizer import (
    HybridParallelOptimizer,
)
from paddle.optimizer import Optimizer

from paddlefleet.trainer.utils import offload_optimizer as oo

try:
    from paddle.optimizer.inner_optimizer import InnerOptimizer
    from paddle.optimizer.mixed_inner_optimizer import (
        MixedInnerOptimizer,
        get_inner_optimizer,
    )

    import paddlefleet.optimizers  # noqa: F401  (registers the sub-optimizers)
except (ImportError, ModuleNotFoundError):
    # Paddle build without the inner-optimizer API.
    InnerOptimizer = None

try:
    from paddle.optimizer.muon import Muon
except (ImportError, ModuleNotFoundError):
    Muon = None

_needs_api = unittest.skipIf(
    InnerOptimizer is None, "Paddle build without the inner-optimizer API"
)
_MISSING = object()
GROUPS = ("muon", "muonh", "adamw", "adamh", "rmsprop", "rmsproph")
MUON_HP = {"momentum": 0.95, "ns_steps": 5, "ns_matmul_dtype": paddle.float32}
ADAM_HP = {"adam_beta1": 0.9, "adam_beta2": 0.95, "adam_epsilon": 1e-8}
RMSPROP_HP = {"momentum": 0.9, "centered": True}
HP = {"muon": MUON_HP, "adam": ADAM_HP, "rmsp": RMSPROP_HP}


def _hack_snapshot(test):
    """Undo every patch the offload hack installs when *test* finishes."""
    saved = {
        (_C_ops, "adam_"): getattr(_C_ops, "adam_", _MISSING),
        (_C_ops, "adamw_"): getattr(_C_ops, "adamw_", _MISSING),
        (Optimizer, "_add_accumulator"): Optimizer._add_accumulator,
        (HybridParallelOptimizer, "_insert_sync"): (
            HybridParallelOptimizer._insert_sync
        ),
        (InnerOptimizer, "_apply_optimize"): InnerOptimizer._apply_optimize,
    }
    if Muon is not None:
        for attr in ("_muon_update_group", "_muon_update"):
            saved[(Muon, attr)] = vars(Muon).get(attr, _MISSING)

    def restore():
        for (obj, name), val in saved.items():
            if val is _MISSING:
                if name in vars(obj):
                    delattr(obj, name)
            else:
                setattr(obj, name, val)

    test.addCleanup(restore)


def _param(shape, dtype, seed):
    w0 = np.random.default_rng(seed).standard_normal(shape).astype("float32")
    return paddle.create_parameter(
        shape=list(shape),
        dtype=dtype,
        default_initializer=paddle.nn.initializer.Assign(w0),
    )


def _states(opt, param):
    states = [
        accs[param.name]
        for accs in opt._accumulators.values()
        if param.name in accs
    ]
    if param.name in opt._master_weights:
        states.append(opt._master_weights[param.name])
    return states


def _step(opt, params, seed):
    rng = np.random.default_rng(seed)
    grads = [
        paddle.to_tensor(rng.standard_normal(p.shape).astype("float32")).astype(
            p.dtype
        )
        for p in params
    ]
    opt._apply_optimize(None, None, list(zip(params, grads)))


if InnerOptimizer is not None:

    class _Sgd(InnerOptimizer):
        """A custom InnerOptimizer (unknown to the offload hack)."""

        keep_whole = None

        def _ensure_accumulators(self, param):
            if param.name not in self._accumulators.get("velocity", {}):
                self._add_accumulator("velocity", param)

        def _update_parameters(self, params_grads, lr):
            with paddle.no_grad():
                for param, grad in params_grads:
                    v = self._get_accumulator("velocity", param)
                    paddle.assign(v * 0.9 + grad, v)
                    param.subtract_(v * lr)

        def optimizer_state_layout(self):
            return {"vector": ["velocity_0"], "scalar": [], "master": ""}


@_needs_api
class TestOffloadGranularity(unittest.TestCase):
    """Order of reload/offload calls around one (second) optimizer step."""

    def setUp(self):
        paddle.set_device("cpu")
        _hack_snapshot(self)
        self.events = []
        for kind in ("reload", "offload"):
            patcher = mock.patch.object(
                oo,
                kind,
                side_effect=lambda t, kind=kind: self.events.append(
                    (kind, id(t))
                ),
            )
            patcher.start()
            self.addCleanup(patcher.stop)
        oo.hack_offload_optimizer()

    def _second_step_events(self, opt, params):
        _step(opt, params, seed=0)  # creates the accumulators
        self.events.clear()
        _step(opt, params, seed=1)
        self.assertNotIn("_update_parameters", vars(opt))
        self.assertNotIn("_update_shape_group", vars(opt))
        return self.events

    @staticmethod
    def _moves(kind, opt, params):
        return [(kind, id(t)) for p in params for t in _states(opt, p)]

    def _sgd(self, params):
        return _Sgd(parameters=params, learning_rate=0.1, weight_decay=0.0)

    def _whole(self, opt, params):
        return self._moves("reload", opt, params) + self._moves(
            "offload", opt, params
        )

    def test_hack_twice_is_idempotent(self):
        oo.hack_offload_optimizer()  # second time (first one in setUp)
        params = [_param([4, 3], "float32", s) for s in range(2)]
        opt = self._sgd(params)
        events = self._second_step_events(opt, params)
        # The inner wrapper's shadow wins: still one move per state.
        self.assertEqual(events, self._whole(opt, params))

    def test_no_inner_optimizer_api_is_noop(self):
        before = InnerOptimizer._apply_optimize
        with mock.patch.dict(
            sys.modules, {"paddle.optimizer.inner_optimizer": None}
        ):
            self.assertIsNone(oo._hack_offload_inner_optimizers())
        self.assertIs(InnerOptimizer._apply_optimize, before)

    def test_muon_per_newton_schulz_batch(self):
        a0, b0, a1 = (
            _param([8, 4], "float32", 0),
            _param([6, 4], "float32", 1),
            _param([8, 4], "float32", 2),
        )
        params = [a0, b0, a1]
        opt = get_inner_optimizer("muon")(
            parameters=params, learning_rate=0.02, weight_decay=0.0, **MUON_HP
        )
        events = self._second_step_events(opt, params)
        # Batches follow Muon's (shape) grouping, not the parameter order.
        expected = (
            self._moves("reload", opt, [a0, a1])
            + self._moves("offload", opt, [a0, a1])
            + self._moves("reload", opt, [b0])
            + self._moves("offload", opt, [b0])
        )
        self.assertEqual(events, expected)

    def test_update_parameters_one_call(self):
        # A custom InnerOptimizer unknown to the hack: one move per state
        # around its single _update_parameters call.
        params = [_param([4, 3], "float32", s) for s in range(2)]
        opt = self._sgd(params)
        events = self._second_step_events(opt, params)
        self.assertEqual(len(_states(opt, params[0])), 1)
        self.assertEqual(events, self._whole(opt, params))

    def test_is_offload_opt_gate(self):
        params = [_param([4, 3], "float32", s) for s in range(2)]
        params[1].is_offload_opt = False
        opt = self._sgd(params)
        events = self._second_step_events(opt, params)
        expected = self._moves("reload", opt, params) + self._moves(
            "offload", opt, [params[0]]
        )
        self.assertEqual(events, expected)


def _mixed_model():
    """Two same-shape matrices (one Muon batch) + one other shape per group."""
    params, group_of = [], {}
    for gi, g in enumerate(GROUPS):
        for k, shape in enumerate(([32, 16], [32, 16], [16, 8])):
            p = _param(shape, "bfloat16", seed=10 * gi + k)
            params.append(p)
            group_of[p.name] = g
    by_group = defaultdict(list)
    for p in params:
        by_group[group_of[p.name]].append(p)
    subs = [
        get_inner_optimizer(g)(
            parameters=by_group[g],
            learning_rate=1e-2,
            weight_decay=0.1,
            multi_precision=True,
            **HP[g[:4]],
        )
        for g in GROUPS
    ]
    opt = MixedInnerOptimizer(
        subs,
        learning_rate=1e-2,
        grad_clip=paddle.nn.ClipGradByGlobalNorm(1.0),
    )
    return params, opt


def _snapshot(params, opt):
    out = {}
    for i, p in enumerate(params):
        out[f"param{i}"] = p.astype("float32").numpy()
        sub = opt._route[p.name]
        for j, t in enumerate(_states(sub, p)):
            out[f"param{i}.state{j}"] = t.numpy()
    return out


@_needs_api
@unittest.skipUnless(
    paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0,
    "needs a GPU (offload moves state to CUDA pinned memory)",
)
class TestOffloadAccuracy(unittest.TestCase):
    STEPS = 4

    def setUp(self):
        paddle.set_device("gpu")
        _hack_snapshot(self)

    def _run(self, offload):
        params, opt = _mixed_model()
        for step in range(self.STEPS):
            _step(opt, params, seed=100 + step)
            if offload:
                for p in params:
                    self.assertTrue(p.place.is_gpu_place(), p.name)
                    sub = opt._route[p.name]
                    states = _states(sub, p)
                    self.assertTrue(states, p.name)
                    for t in states:
                        self.assertTrue(
                            t.place.is_cuda_pinned_place(),
                            f"step {step}: {t.name} on {t.place}",
                        )
        return _snapshot(params, opt), opt.state_dict()

    def test_offload_bit_identical(self):
        ref, ref_sd = self._run(offload=False)
        oo.hack_offload_optimizer()
        got, got_sd = self._run(offload=True)
        self.assertEqual(ref.keys(), got.keys())
        for key in ref:
            with self.subTest(key=key):
                np.testing.assert_array_equal(got[key], ref[key])
        # state_dict (the checkpoint source) reads the pinned tensors fine.
        self.assertEqual(len(ref_sd), len(got_sd))
        for (k0, v0), (k1, v1) in zip(ref_sd.items(), got_sd.items()):
            if isinstance(v0, paddle.Tensor):
                np.testing.assert_array_equal(v1.numpy(), v0.numpy())


if __name__ == "__main__":
    unittest.main()
