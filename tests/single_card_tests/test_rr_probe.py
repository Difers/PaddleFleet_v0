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

"""Refined-recompute probe: numerics unchanged, memory matches the frame."""

import json
import os
import tempfile
import types
import unittest

import paddle
from paddle.distributed.fleet.recompute.recompute import RecomputeFunction
from paddle.distributed.fleet.utils import recompute

from paddlefleet.recompute_utils import make_refined_recompute
from paddlefleet.refined_recompute import AutoRefinedRecompute
from paddlefleet.refined_recompute.probe import (
    PROBE,
    PhasePeaks,
    RRProbe,
    _layer_of,
)
from paddlefleet.refined_recompute.probe_callback import (
    ProbeWindowCallback,
    window,
)

_SEED = 42
_BATCH, _SEQ, _WIDE = 2, 8, 64
_POINT = "mla_o_proj"
_VETOES = (
    {"recompute_granularity": None},
    {"virtual_pipeline_model_parallel_size": 2},
)


def _config(**overrides):
    defaults = {
        "recompute_granularity": "full",
        "recompute_method": "uniform",
        "recompute_num_layers": 1,
        "recompute_modules": [],
        "virtual_pipeline_model_parallel_size": 1,
    }
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


class _Block(paddle.nn.Layer):
    """One RR point wired like a real call site."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer_number = 1
        self.is_mtp_layer = False
        self.up = paddle.nn.Linear(_WIDE, 4 * _WIDE, bias_attr=False)
        self.down = paddle.nn.Linear(4 * _WIDE, _WIDE, bias_attr=False)
        self.calls = 0
        self.use_rr, self._rr = make_refined_recompute(self, _POINT)

    def _boundary(self, hidden):
        self.calls += 1
        return self.down(self.up(hidden))

    def forward(self, hidden):
        if self.use_rr and self.training:
            hidden = self._rr(self._boundary, hidden)
        else:
            hidden = self._boundary(hidden)
        return paddle.tanh(hidden)


def _build(**overrides):
    paddle.seed(_SEED)
    block = _Block(_config(**overrides))
    block.train()
    return block


def _run(block):
    paddle.seed(_SEED)
    hidden = paddle.randn([_BATCH, _SEQ, _WIDE])
    hidden.stop_gradient = False
    output = recompute(block, hidden)
    output.sum().backward()
    return output.detach(), hidden.grad.detach()


class _ProbeCase(unittest.TestCase):
    def setUp(self):
        PROBE.deactivate()
        self.addCleanup(PROBE.deactivate)

    def activate(self, step=0):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        PROBE.activate(directory.name)
        if step is not None:
            PROBE.begin_step(step)


class TestProbeIsTransparent(_ProbeCase):
    def test_matches_plain_recompute_bitwise(self):
        baseline = _build()
        self.activate()
        probed = _build()
        probed.set_state_dict(baseline.state_dict())
        self.assertIsInstance(probed._rr, RRProbe)

        for expected, actual in zip(_run(baseline), _run(probed), strict=True):
            self.assertTrue(paddle.equal_all(expected, actual).item())
        self.assertEqual(baseline.calls, 2)
        self.assertEqual(probed.calls, 2)


class TestWindow(_ProbeCase):
    def test_prices_only_the_recompute_pass_inside_the_window(self):
        self.activate(step=None)
        block = _build()
        _run(block)
        self.assertIsNone(PROBE._window)

        PROBE.begin_step(7)
        _run(block)
        _run(block)  # a second micro-batch of the same optimizer step
        PROBE.end_step()
        _run(block)
        self.assertEqual(
            [(r.key, r.step) for r in PROBE._window.records],
            [((_POINT, 1), 7)] * 2,
        )
        self.assertEqual(PROBE.summarize()[_POINT, 1].calls_per_step, 2)

    def test_flush_writes_the_step(self):
        self.activate(step=3)
        _run(_build())
        PROBE.end_step()  # closing the step is what yields its phase row
        PROBE.flush()
        self.assertIsNone(PROBE._window)
        rank = os.environ.get("PADDLE_TRAINER_ID", "0")
        with open(os.path.join(PROBE._directory, f"rank{rank}.jsonl")) as f:
            row = json.loads(f.readline())
        got = (row["point"], row["layer"], row["step"])
        self.assertEqual(got, (_POINT, 1, 3))
        self.assertTrue(
            os.path.exists(
                os.path.join(PROBE._directory, f"phase_rank{rank}.jsonl")
            )
        )

    def test_layer_is_the_logical_index(self):
        self.activate()
        owner = _Block(_config(num_empty_layers_add_in_head=1))
        self.assertEqual(owner._rr._key, (_POINT, 0))


class _Holder:
    def __init__(self):
        self.config = _config()
        self.layer_number = 4
        self.is_mtp_layer = False

    def method(self):
        pass

    def closure(self):
        def impl():
            return self

        return impl


class TestPhaseRecorder(_ProbeCase):
    def test_splits_a_step_by_where_the_peak_falls(self):
        peaks = PhasePeaks()
        for kind, layer, peak in (
            ("step_begin", None, 1),
            ("rcF_begin", 1, 10),  # before the first forward: O
            ("rcF_end", 1, 20),
            ("rcF_begin", 2, 15),
            ("rcF_end", 2, 30),
            ("rcB_begin", 2, 40),  # loss and head: F
            ("rcB_end", 2, 55),
            ("rcB_begin", 1, 35),
            ("rcB_end", 1, 45),
            ("step_end", None, 60),  # optimizer: O
        ):
            peaks.cut(kind, layer, peak)
        self.assertEqual((peaks.O, peaks.F, peaks.B), (60, 40, {1: 45, 2: 55}))
        self.assertEqual(peaks.order, [1, 2])

        peaks.cut("rcB_end", None, 99)  # unknown layer: counted as F
        self.assertEqual((peaks.F, peaks.unknown), (99, 1))

    def test_layer_of_a_method_a_layer_and_a_closure(self):
        holder = _Holder()
        for run_function in (holder.method, holder, holder.closure()):
            self.assertEqual(_layer_of(run_function), 4)
        self.assertIsNone(_layer_of(lambda: None))

    def test_window_records_a_step_and_restores_the_hooks(self):
        original = vars(RecomputeFunction)["forward"]
        self.activate(step=None)
        PROBE.begin_step(2)
        self.assertIsNot(vars(RecomputeFunction)["forward"], original)
        _run(_build())
        PROBE.end_step()

        (row,) = PROBE.phase_rows()
        self.assertEqual((row["step"], row["order"]), (2, [1]))
        self.assertEqual(row["unknown_layer_events"], 0)
        self.assertGreater(row["B"][1], 0)
        PROBE.flush()
        self.assertIs(vars(RecomputeFunction)["forward"], original)

    def test_hooks_are_installed_once_per_window(self):
        original = vars(RecomputeFunction)["forward"]
        self.activate(step=1)
        hooked = vars(RecomputeFunction)["forward"]
        PROBE.end_step()
        self.assertIs(vars(RecomputeFunction)["forward"], hooked)
        PROBE.begin_step(2)
        self.assertIs(vars(RecomputeFunction)["forward"], hooked)
        PROBE.end_step()
        PROBE.flush()
        PROBE.flush()  # nothing left to restore
        self.assertIs(vars(RecomputeFunction)["forward"], original)

    def test_deactivate_restores_the_hooks_and_clears_the_window(self):
        original = vars(RecomputeFunction)["forward"]
        self.activate(step=1)
        PROBE.deactivate()
        self.assertIs(vars(RecomputeFunction)["forward"], original)
        self.assertIsNone(PROBE._window)
        self.assertIsNone(PROBE._directory)

    def test_a_recompute_between_steps_is_not_recorded(self):
        self.activate(step=1)
        PROBE.end_step()
        _run(_build())  # e.g. an evaluation
        self.assertEqual(PROBE._window.records, [])
        (row,) = PROBE.phase_rows()
        self.assertEqual(row["order"], [])

    def test_closing_without_an_open_step_does_nothing(self):
        self.activate(step=None)
        PROBE.end_step()
        PROBE.flush()
        self.assertEqual(PROBE.phase_rows(), [])
        PROBE.begin_step(1)
        PROBE.end_step()
        PROBE.end_step()
        self.assertEqual(len(PROBE.phase_rows()), 1)
        self.assertIsNone(PROBE.current_step)


class TestMemoryGaugeMatchesFrame(_ProbeCase):
    def test_price_bytes_equal_the_retained_frame(self):
        self.activate()
        block = _build()
        _run(block)
        measured = PROBE.summarize()[_POINT, 1].bytes
        PROBE.deactivate()

        refined = _build(recompute_modules=[_POINT])
        refined.set_state_dict(block.state_dict())
        self.assertIsInstance(refined._rr, AutoRefinedRecompute)
        paddle.seed(_SEED)
        hidden = paddle.randn([_BATCH, _SEQ, _WIDE])
        paddle.device.synchronize()
        before = paddle.device.cuda.memory_allocated()
        with paddle.no_grad():
            refined(hidden)
        paddle.device.synchronize()
        retained = paddle.device.cuda.memory_allocated() - before
        refined._rr._frames.clear()
        self.assertEqual(measured, retained)


class TestFactory(_ProbeCase):
    def test_every_point_is_priced_when_active(self):
        self.activate()
        for modules in ([], [_POINT]):
            owner = _Block(_config(recompute_modules=modules))
            self.assertIsInstance(owner._rr, RRProbe)

    def test_preconditions_still_veto_when_active(self):
        self.activate()
        for overrides in _VETOES:
            with self.subTest(**overrides):
                self.assertFalse(_Block(_config(**overrides)).use_rr)


class TestHandWrittenPoint(_ProbeCase):
    def _owner(self, **overrides):
        return types.SimpleNamespace(
            config=_config(**overrides), layer_number=3, is_mtp_layer=False
        )

    def test_passthrough_when_inactive_or_vetoed(self):
        function = paddle.tanh
        wrapped = PROBE.wrap(function, "flash_attn", self._owner())
        self.assertIs(wrapped, function)
        self.activate()
        for overrides in _VETOES:
            with self.subTest(**overrides):
                owner = self._owner(**overrides)
                wrapped = PROBE.wrap(function, "flash_attn", owner)
                self.assertIs(wrapped, function)

    def test_only_recompute_pass_is_priced(self):
        self.activate()
        probed = PROBE.wrap(paddle.tanh, "flash_attn", self._owner())
        hidden = paddle.randn([_BATCH, _SEQ, _WIDE])
        hidden.stop_gradient = False
        recompute(probed, hidden).sum().backward()
        self.assertEqual(
            [r.key for r in PROBE._window.records], [("flash_attn", 3)]
        )


class TestYamlSection(_ProbeCase):
    def _args(self, section):
        return types.SimpleNamespace(auto_refined_recompute=section)

    def test_window_defaults(self):
        self.assertEqual(window(self._args(None)), (3, 15))

    def test_callback_opens_only_the_configured_steps(self):
        args = self._args(
            {"calibration": {"warmup_steps": 1, "measure_steps": 2}}
        )
        callback = ProbeWindowCallback()
        with tempfile.TemporaryDirectory() as directory:
            PROBE.activate(directory)
            for global_step, expected in ((0, None), (1, 1), (2, 2), (3, None)):
                state = types.SimpleNamespace(global_step=global_step)
                callback.on_step_begin(args, state, None)
                self.assertEqual(PROBE.current_step, expected)
                callback.on_step_end(args, state, None)

    def test_an_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            PROBE.configure({"mode": "measure"}, "out")

    def test_train_end_writes_an_unfinished_window(self):
        args = self._args(
            {"calibration": {"warmup_steps": 0, "measure_steps": 5}}
        )
        self.activate(step=None)
        block, callback = _build(), ProbeWindowCallback()
        state = types.SimpleNamespace(global_step=0)
        callback.on_step_begin(args, state, None)
        _run(block)
        callback.on_step_end(
            args, state, None
        )  # step 0 of 5: nothing written yet
        rank = os.environ.get("PADDLE_TRAINER_ID", "0")
        path = os.path.join(PROBE._directory, f"rank{rank}.jsonl")
        self.assertFalse(os.path.exists(path))
        callback.on_train_end(args, state, None)
        self.assertTrue(os.path.exists(path))

    def test_only_profile_mode_activates_the_probe(self):
        for section in ({}, {"mode": "train"}):
            PROBE.configure(section, "out")
            self.assertFalse(PROBE.active)
        PROBE.configure({"mode": "profile"}, "out")
        self.assertEqual(PROBE._directory, os.path.join("out", "auto_rr"))


if __name__ == "__main__":
    unittest.main()
