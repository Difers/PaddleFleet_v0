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

"""Unit tests for the deterministic DSpark anchor sampler.

``sample_anchor_positions`` is host-side numpy plus an optional tensor
round-trip, so nothing here needs a fleet init or a GPU.  The tests pin the
three properties the draft loss leans on: the ``(seed, global_step,
sample_id)`` determinism contract, the validation guards, and the short-row
cycling that decides how much weight a row carries in the draft-loss numerator
and denominator.
"""

import unittest
import warnings

import numpy as np
import paddle

from paddlefleet.utils.dspark import anchor_sampler
from paddlefleet.utils.dspark.anchor_sampler import (
    AnchorBatch,
    sample_anchor_positions,
)


def _call(loss_mask, **overrides):
    kwargs = {
        "num_anchors": 2,
        "sample_ids": [f"row{i}" for i in range(len(loss_mask))],
        "global_step": 0,
        "training_seed": 42,
    }
    kwargs.update(overrides)
    return sample_anchor_positions(loss_mask=loss_mask, **kwargs)


class TestAnchorSamplingContract(unittest.TestCase):
    def setUp(self):
        # The once-per-process warning latch is module state; reset it so the
        # cycling tests below do not depend on execution order.
        anchor_sampler._CYCLING_REPORTED = False

    def test_positions_are_sorted_distinct_and_kept(self):
        mask = np.ones((2, 8), dtype=bool)
        batch = _call(mask, num_anchors=3)
        self.assertIsInstance(batch, AnchorBatch)
        self.assertEqual(batch.positions.shape, (2, 3))
        self.assertEqual(batch.cycled_rows, 0)
        self.assertTrue(batch.keep_mask.all())
        for row in batch.positions:
            self.assertEqual(list(row), sorted(row))
            self.assertEqual(len(set(row.tolist())), 3)

    def test_column_zero_is_never_an_anchor(self):
        """Position 0 has no left context, so it is masked out unconditionally."""
        batch = _call(np.ones((1, 4), dtype=bool), num_anchors=3)
        self.assertNotIn(0, batch.positions[0].tolist())

    def test_same_key_reproduces_and_the_key_actually_moves_it(self):
        mask = np.ones((1, 8), dtype=bool)
        base = _call(mask, sample_ids=["a"], num_anchors=2).positions.copy()
        again = _call(mask, sample_ids=["a"], num_anchors=2).positions.copy()
        np.testing.assert_array_equal(base, again)

        other_step = _call(
            mask, sample_ids=["a"], num_anchors=2, global_step=1
        ).positions
        other_seed = _call(
            mask, sample_ids=["a"], num_anchors=2, training_seed=7
        ).positions
        other_id = _call(mask, sample_ids=["b"], num_anchors=2).positions
        self.assertFalse(
            np.array_equal(base, other_step)
            and np.array_equal(base, other_seed)
            and np.array_equal(base, other_id)
        )

    def test_short_rows_cycle_and_warn_exactly_once(self):
        # Row 0 has two usable positions for num_anchors=3 -> cycling; row 1 has
        # plenty.  Only the first call in the process may warn.
        mask = np.zeros((2, 6), dtype=bool)
        mask[0, 1:3] = True
        mask[1, 1:] = True
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            first = _call(mask, num_anchors=3)
            second = _call(mask, num_anchors=3)
        self.assertEqual(first.cycled_rows, 1)
        self.assertEqual(second.cycled_rows, 1)
        cycling = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        self.assertEqual(len(cycling), 1)

    def test_a_row_with_no_valid_position_stays_empty(self):
        mask = np.zeros((2, 5), dtype=bool)
        mask[1, 2:4] = True
        batch = _call(mask, num_anchors=2)
        self.assertFalse(batch.keep_mask[0].any())
        self.assertEqual(batch.positions[0].tolist(), [0, 0])
        self.assertTrue(batch.keep_mask[1].all())


class TestAnchorSamplingValidation(unittest.TestCase):
    def setUp(self):
        anchor_sampler._CYCLING_REPORTED = False

    def test_rejects_bad_scalars(self):
        mask = np.ones((1, 6), dtype=bool)
        cases = [
            ("num_anchors", 0, "num_anchors must be a positive integer"),
            ("num_anchors", -3, "num_anchors must be a positive integer"),
            ("num_anchors", True, "num_anchors must be a positive integer"),
            ("num_anchors", 1.5, "num_anchors must be a positive integer"),
            ("global_step", True, "global_step must be an integer"),
            ("global_step", 2.5, "global_step must be an integer"),
            ("global_step", -1, "global_step must be non-negative"),
            ("training_seed", True, "training_seed must be an integer"),
            ("training_seed", 0.5, "training_seed must be an integer"),
        ]
        for field, value, message in cases:
            with self.subTest(field=field, value=value):
                with self.assertRaises(ValueError) as caught:
                    _call(mask, **{field: value})
                self.assertIn(message, str(caught.exception))

    def test_rejects_bad_loss_mask_shapes(self):
        cases = [
            np.ones(6, dtype=bool),
            np.zeros((0, 6), dtype=bool),
            np.zeros((2, 0), dtype=bool),
        ]
        for mask in cases:
            with self.subTest(shape=mask.shape):
                with self.assertRaises(ValueError) as caught:
                    sample_anchor_positions(
                        loss_mask=mask,
                        num_anchors=2,
                        sample_ids=[f"r{i}" for i in range(mask.shape[0])],
                        global_step=0,
                        training_seed=1,
                    )
                self.assertIn("nonempty shape", str(caught.exception))

    def test_rejects_bad_sample_ids(self):
        mask = np.ones((2, 6), dtype=bool)
        cases = [
            ["only-one"],
            ["a", "b", "c"],
            ["a", ""],
            ["a", 7],
            ["a", None],
        ]
        for ids in cases:
            with self.subTest(sample_ids=ids):
                with self.assertRaises(ValueError) as caught:
                    _call(mask, sample_ids=ids)
                self.assertIn(
                    "one non-empty string per row", str(caught.exception)
                )

    def test_tensor_mask_round_trips_to_tensors(self):
        mask = paddle.ones([2, 6], dtype="bool")
        batch = _call(mask, num_anchors=2)
        self.assertTrue(hasattr(batch.positions, "numpy"))
        self.assertTrue(hasattr(batch.keep_mask, "numpy"))
        self.assertEqual(list(batch.positions.shape), [2, 2])
        self.assertTrue(bool(batch.keep_mask.numpy().all()))

    def test_numpy_mask_returns_numpy(self):
        batch = _call(np.ones((1, 6), dtype=bool), num_anchors=2)
        self.assertIsInstance(batch.positions, np.ndarray)
        self.assertIsInstance(batch.keep_mask, np.ndarray)
