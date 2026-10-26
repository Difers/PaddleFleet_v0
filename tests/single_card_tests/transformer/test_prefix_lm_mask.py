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

"""Unit tests for the prefix-LM three-region mask representations.

The FlashMask row indices are what the packed encoder actually feeds to
``flashmask_attention(causal=False)``; the dense mask is the reference. Every
row-index test is checked either against a hand-derived table or against the
dense mask through an expander that only knows the FlashMask semantics.
"""

import random
import unittest

import paddle

from paddlefleet.transformer.prefix_lm_mask import (
    build_dense_mask,
    build_flashmask_row_indices,
    expand_row_indices_to_dense,
    prefix_lm_pad_len,
    prefix_lm_segment_starts,
)

_GPU = paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0


class TestCheckLens(unittest.TestCase):
    def test_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            build_dense_mask([1, 2], [1])

    def test_empty(self):
        with self.assertRaises(ValueError):
            build_dense_mask([], [])

    def test_negative_prefix(self):
        with self.assertRaises(ValueError):
            build_dense_mask([-1], [2])

    def test_non_positive_suffix(self):
        with self.assertRaises(ValueError):
            build_dense_mask([2], [0])

    def test_negative_pad(self):
        with self.assertRaises(ValueError):
            build_dense_mask([2], [2], pad_len=-1)


class TestSegmentStarts(unittest.TestCase):
    def test_prefix_sum(self):
        self.assertEqual(prefix_lm_segment_starts([2, 3], [1, 1]), [0, 3])


class TestPadLen(unittest.TestCase):
    def test_sp_disabled_uses_align(self):
        # base = 1, align = 128 -> round up to 128
        self.assertEqual(
            prefix_lm_pad_len(100, tp_size=4, sequence_parallel=False), 28
        )

    def test_sp_enabled_uses_lcm(self):
        # base = tp_size = 4, lcm(4, 128) = 128
        self.assertEqual(
            prefix_lm_pad_len(100, tp_size=4, sequence_parallel=True), 28
        )

    def test_already_aligned(self):
        self.assertEqual(
            prefix_lm_pad_len(128, tp_size=1, sequence_parallel=False), 0
        )


class TestBuildDenseMask(unittest.TestCase):
    def test_shape_no_pad(self):
        m = build_dense_mask([2], [2])
        self.assertEqual(m.shape, [1, 1, 4, 4])
        self.assertEqual(m.dtype, paddle.bool)

    def test_zero_prefix_segment(self):
        # c == 0 skips the prefix-open branch but still opens causal suffix.
        m = build_dense_mask([0], [3])
        self.assertEqual(m.shape, [1, 1, 3, 3])

    def test_with_pad_self_loop(self):
        m = build_dense_mask([2], [2], pad_len=2)
        self.assertEqual(m.shape, [1, 1, 6, 6])
        # pad rows only see themselves (diagonal is False = allowed).
        self.assertFalse(bool(m[0, 0, 5, 5]))
        self.assertTrue(bool(m[0, 0, 5, 0]))


class TestFlashMaskRowIndices(unittest.TestCase):
    def test_shape_and_dtype(self):
        for pad in (0, 2):
            with self.subTest(pad=pad):
                ri = build_flashmask_row_indices([2], [2], pad_len=pad)
                self.assertEqual(ri.shape, [1, 1, 4 + pad, 2])
                self.assertEqual(ri.dtype, paddle.int32)

    def test_values_match_hand_derived_table(self):
        # seg0: s=0, C=2, L=4 -> prefix cols 0,1: [LTS=4, UTE=s=0];
        #                        suffix cols 2,3: [4, j]
        # seg1: s=4, C=0, L=3 -> suffix cols 4,5,6: [7, j]
        # pad cols 7,8: [j+1, j] (self only)
        ri = build_flashmask_row_indices([2, 0], [2, 3], pad_len=2)
        expected = [
            [4, 0],
            [4, 0],
            [4, 2],
            [4, 3],
            [7, 4],
            [7, 5],
            [7, 6],
            [8, 7],
            [9, 8],
        ]
        self.assertEqual(ri.reshape([-1, 2]).numpy().tolist(), expected)

    def test_validates_lengths(self):
        with self.assertRaises(ValueError):
            build_flashmask_row_indices([2], [0])
        with self.assertRaises(ValueError):
            build_flashmask_row_indices([2], [2], pad_len=-1)


class TestExpandRowIndices(unittest.TestCase):
    def test_rejects_legacy_four_column_layout(self):
        # The causal 4-column [LTS, LTE, UTS, UTE] form must not be mistaken for
        # the non-causal 2-column form.
        for cols in (1, 3, 4):
            with self.subTest(cols=cols), self.assertRaises(ValueError):
                expand_row_indices_to_dense(
                    paddle.zeros([1, 1, 4, cols], dtype="int32")
                )

    def test_rejects_non_4d(self):
        with self.assertRaises(ValueError):
            expand_row_indices_to_dense(paddle.zeros([4, 2], dtype="int32"))


class TestRepresentationsAgree(unittest.TestCase):
    """Dense mask and FlashMask row indices expand to the same forbidden set."""

    def _check(self, prefix_lens, suffix_lens, pad_len):
        dense = build_dense_mask(prefix_lens, suffix_lens, pad_len)
        via_ri = expand_row_indices_to_dense(
            build_flashmask_row_indices(prefix_lens, suffix_lens, pad_len)
        )
        self.assertEqual(dense.shape, via_ri.shape)
        mismatch = int((dense != via_ri).astype("int32").sum())
        self.assertEqual(
            mismatch, 0, f"{prefix_lens=} {suffix_lens=} {pad_len=}"
        )

    def test_edge_cases(self):
        cases = [
            ([3], [2], 0),  # single segment
            ([2, 3], [2, 1], 4),  # multi segment + pad
            ([0, 2], [3, 2], 2),  # empty prefix in the first segment
            ([2, 0], [1, 1], 0),  # empty prefix in the last segment, Q=1
            ([0], [1], 0),  # smallest legal pack
            ([5, 1, 0, 7], [1, 3, 2, 1], 1),  # 1-token suffixes + 1-token pad
        ]
        for prefix, suffix, pad in cases:
            with self.subTest(prefix=prefix, suffix=suffix, pad=pad):
                self._check(prefix, suffix, pad)

    def test_random_packs(self):
        rng = random.Random(20261022)
        for _ in range(40):
            n = rng.randint(1, 6)
            prefix = [rng.randint(0, 9) for _ in range(n)]
            suffix = [rng.randint(1, 9) for _ in range(n)]
            pad = rng.choice([0, 0, 1, rng.randint(2, 7)])
            with self.subTest(prefix=prefix, suffix=suffix, pad=pad):
                self._check(prefix, suffix, pad)


class TestExplicitPlace(unittest.TestCase):
    """The optional ``place`` puts the result on that device, value-unchanged."""

    def test_cpu_place(self):
        ri = build_flashmask_row_indices(
            [2], [2], pad_len=2, place=paddle.CPUPlace()
        )
        self.assertTrue(ri.place.is_cpu_place())
        m = build_dense_mask([2], [2], pad_len=2, place=paddle.CPUPlace())
        self.assertTrue(m.place.is_cpu_place())

    @unittest.skipUnless(_GPU, "requires a GPU")
    def test_gpu_place_matches_cpu_values(self):
        args = ([3, 0, 2], [2, 4, 1], 3)
        cpu = build_flashmask_row_indices(*args, place=paddle.CPUPlace())
        gpu = build_flashmask_row_indices(*args, place=paddle.CUDAPlace(0))
        self.assertTrue(gpu.place.is_gpu_place())
        self.assertEqual(gpu.numpy().tolist(), cpu.numpy().tolist())


if __name__ == "__main__":
    unittest.main()
