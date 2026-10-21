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

"""AutoRefinedRecompute(uses_rng=True): a boundary that draws random numbers.

The segment is ``lin1 -> boundary(lin2 + dropout) -> dropout -> lin3`` under
``recompute``. The recompute pass skips the boundary, so without the write-back
the dropout after it starts too early in the random stream and gets another
mask. Every gradient is compared bitwise with plain autograd, where each mask is
drawn once.
"""

import unittest
from unittest import mock

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle.distributed.fleet.layers.mpu.random import get_rng_state_tracker
from paddle.distributed.fleet.utils import recompute

from paddlefleet.refined_recompute import AutoRefinedRecompute, rng_state

_SEED = 2026
_SLOT = "rr_rng_test_slot"
_HIDDEN = 16


def _dropout(x, slot):
    if slot is None:
        return F.dropout(x, 0.3, training=True)
    with get_rng_state_tracker().rng_state(slot):
        return F.dropout(x, 0.3, training=True)


class _Segment(paddle.nn.Layer):
    def __init__(self, rr, slot, chain):
        super().__init__()
        self.lin1 = paddle.nn.Linear(_HIDDEN, _HIDDEN)
        self.lin2 = paddle.nn.Linear(_HIDDEN, _HIDDEN)
        self.lin3 = paddle.nn.Linear(_HIDDEN, _HIDDEN)
        self.lin4 = paddle.nn.Linear(_HIDDEN, _HIDDEN)
        self.slot = slot
        self.chain = chain
        self.rr = rr

    def _call(self, index, fn, x):
        return fn(x) if not self.rr else self.rr[index](fn, x)

    def first(self, h):
        return _dropout(self.lin2(h), self.slot)

    def second(self, h):
        return _dropout(self.lin4(h), self.slot)

    def forward(self, x):
        y = self._call(0, self.first, self.lin1(x))
        if self.chain:
            y = self._call(1, self.second, _dropout(y, self.slot))
        return self.lin3(_dropout(y, self.slot))


def _run(mode, slot=None, chain=False, micro_batches=1, uses_rng=True):
    """Gradients of three stacked segments; ``mode`` is ``ref`` (plain
    autograd), ``full`` (recompute) or ``rr`` (recompute + boundary)."""
    tracker = get_rng_state_tracker()
    tracker.reset()
    paddle.seed(_SEED)
    if slot is not None:
        tracker.add(slot, _SEED + 7)
    paddle.seed(_SEED)
    segments = []
    for _ in range(3):
        rr = None
        if mode == "rr":
            rr = [
                AutoRefinedRecompute(f"b{i}", uses_rng=uses_rng)
                for i in range(2 if chain else 1)
            ]
        segments.append(_Segment(rr, slot, chain))

    state = np.random.RandomState(_SEED)
    inputs = []
    for _ in range(micro_batches):
        x = paddle.to_tensor(state.randn(4, _HIDDEN).astype("float32"))
        x.stop_gradient = False
        inputs.append(x)
    # All forwards first, as under 1F1B: micro-batches are in flight together.
    losses = []
    for x in inputs:
        h = x
        for segment in segments:
            h = segment(h) if mode == "ref" else recompute(segment, h)
        losses.append(h.sum())
    for loss in losses:
        loss.backward()

    grads = {
        f"{i}.{name}": param.grad.numpy().copy()
        for i, segment in enumerate(segments)
        for name, param in segment.named_parameters()
        if param.grad is not None  # lin4 is unused unless chained
    }
    for j, x in enumerate(inputs):
        grads[f"x{j}"] = x.grad.numpy().copy()
    pending = sum(r.pending for s in segments for r in s.rr or [])
    return grads, pending


def _max_diff(a, b):
    assert a.keys() == b.keys()
    return max(float(np.abs(a[k] - b[k]).max()) for k in a)


class TestAutoRefinedRecomputeRng(unittest.TestCase):
    def tearDown(self):
        get_rng_state_tracker().reset()

    def _assert_bitwise(self, **kwargs):
        ref, _ = _run("ref", **kwargs)
        full, _ = _run("full", **kwargs)
        self.assertEqual(_max_diff(ref, full), 0.0, "recompute drifted")
        got, pending = _run("rr", **kwargs)
        self.assertEqual(pending, 0, "a retained frame leaked")
        self.assertEqual(_max_diff(ref, got), 0.0)

    def test_default_generator(self):
        self._assert_bitwise()

    def test_tracker_state(self):
        self._assert_bitwise(slot=_SLOT)

    def test_two_boundaries_in_one_segment(self):
        self._assert_bitwise(chain=True)
        self._assert_bitwise(slot=_SLOT, chain=True)

    def test_micro_batches_in_flight(self):
        self._assert_bitwise(micro_batches=2)
        self._assert_bitwise(slot=_SLOT, micro_batches=2)

    def test_undeclared_boundary_misaligns_later_dropout(self):
        """Control: without ``uses_rng`` the gradients differ, so the checks
        above are not vacuous."""
        for slot in (None, _SLOT):
            ref, _ = _run("ref", slot=slot)
            got, _ = _run("rr", slot=slot, uses_rng=False)
            self.assertGreater(_max_diff(ref, got), 0.0, f"slot={slot}")

    def test_snapshot_needs_the_tracker_states(self):
        """Control: a snapshot of the default generators alone is not enough
        when the draws are made in a tracker state."""

        def default_only():
            return paddle.get_rng_state(), None

        ref, _ = _run("ref", slot=_SLOT)
        with mock.patch.object(rng_state, "capture", default_only):
            got, _ = _run("rr", slot=_SLOT)
        self.assertGreater(_max_diff(ref, got), 0.0)

    def test_snapshot_only_when_declared(self):
        for uses_rng, calls in ((False, 0), (True, 3)):
            with mock.patch.object(
                rng_state, "capture", wraps=rng_state.capture
            ) as capture:
                _run("rr", uses_rng=uses_rng)
            self.assertEqual(capture.call_count, calls)


if __name__ == "__main__":
    unittest.main()
