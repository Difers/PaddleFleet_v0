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

"""Pure-function unit test for ``get_doc_start_per_pos`` (CSA-aligned naming).

No fleet/distributed init needed: ``get_doc_start_per_pos`` and its helper
``_doc_start_per_pos_from_startend`` are plain tensor ops. Covers the
``attn_mask_startend_row_indices`` branch (CP=1 and CP>1 for both
``cp_balance_mode`` layouts) plus the ``position_ids``/none fallbacks. Unlike
the pre-rename ``_document_segments``, every branch now returns the real
document START POSITION (matching ``csa_attention.py``'s ``doc_start_per_pos``
semantics), not a mix of start positions and incrementing segment ids.

Under CP>1 the global document starts are sliced with the layout the rest of
the model scatters with, so the values stay in GLOBAL coordinates; the
assertions below therefore compare global starts, and the zigzag/contiguous
cases are chosen to differ.
"""

import unittest

import paddle

from paddlefleet.utils.dspark.dspark_utils import get_doc_start_per_pos


def _startend_from_doc_lens(doc_lens):
    """Build a ``[1, S]`` startend-row-indices row from a list of doc lengths."""
    row = []
    cursor = 0
    for length in doc_lens:
        cursor += length
        row.extend([cursor] * length)
    return paddle.to_tensor([row], dtype="int64")


class TestGetDocStartPerPos(unittest.TestCase):
    def test_startend_cp1_matches_two_documents(self):
        # Doc A = positions [0,5), doc B = positions [5,9).
        startend = _startend_from_doc_lens([5, 4])
        input_ids = paddle.zeros([1, 9], dtype="int64")
        doc_start = get_doc_start_per_pos(
            input_ids,
            None,
            attn_mask_startend_row_indices=startend,
            cp_rank=0,
            cp_size=1,
        )
        start = doc_start.numpy()[0]
        self.assertTrue((start[:5] == 0).all())
        self.assertTrue((start[5:9] == 5).all())

    def test_startend_contiguous_rank0_single_document(self):
        # Global seq len 8, one document, cp_size=2 contiguous -> rank 0 holds
        # global [0,4) of the same document, so every local position maps to
        # the global start 0.
        startend = _startend_from_doc_lens([8])
        local_input_ids = paddle.zeros([1, 4], dtype="int64")
        doc_start = get_doc_start_per_pos(
            local_input_ids,
            None,
            attn_mask_startend_row_indices=startend,
            cp_rank=0,
            cp_size=2,
            cp_balance_mode="contiguous_allgather",
        )
        start = doc_start.numpy()[0]
        self.assertEqual(start.shape[0], 4)
        self.assertTrue((start == 0).all())

    def test_startend_contiguous_rank1_boundary_mid_shard(self):
        # Global seq len 8: doc A = [0,6), doc B = [6,8). cp_size=2 contiguous
        # -> rank 1 holds global [4,8), straddling the boundary at 6. Values
        # stay in global coordinates: [0, 0, 6, 6].
        startend = _startend_from_doc_lens([6, 2])
        local_input_ids = paddle.zeros([1, 4], dtype="int64")
        doc_start = get_doc_start_per_pos(
            local_input_ids,
            None,
            attn_mask_startend_row_indices=startend,
            cp_rank=1,
            cp_size=2,
            cp_balance_mode="contiguous_allgather",
        )
        start = doc_start.numpy()[0]
        self.assertEqual(list(start), [0, 0, 6, 6])

    def test_startend_zigzag_rank1_uses_two_chunks(self):
        # Same data as the contiguous case above, but the default zigzag
        # layout: rank 1 owns global [2,4) + [4,6), i.e. positions [2,3] (doc
        # A, start 0) and [4,5] (doc B, start 4). A contiguous-window slice
        # would have produced [4,4,4,4] here -- wrong layout, silently wrong
        # same-document masking.
        startend = _startend_from_doc_lens([4, 4])
        local_input_ids = paddle.zeros([1, 4], dtype="int64")
        doc_start = get_doc_start_per_pos(
            local_input_ids,
            None,
            attn_mask_startend_row_indices=startend,
            cp_rank=1,
            cp_size=2,
            cp_balance_mode="dualchunk_allgather",
        )
        start = doc_start.numpy()[0]
        self.assertEqual(list(start), [0, 0, 4, 4])

        # The two layouts must actually differ, or the layout-aware slice
        # would be untested (contiguous rank 1 = global [4,8) = all doc B).
        contiguous = get_doc_start_per_pos(
            local_input_ids,
            None,
            attn_mask_startend_row_indices=startend,
            cp_rank=1,
            cp_size=2,
            cp_balance_mode="contiguous_allgather",
        ).numpy()[0]
        self.assertEqual(list(contiguous), [4, 4, 4, 4])

    def test_startend_cp_gt1_requires_balance_mode(self):
        # Without a layout the slice would silently assume one, which is the
        # bug this parameter exists to prevent.
        startend = _startend_from_doc_lens([8])
        local_input_ids = paddle.zeros([1, 4], dtype="int64")
        with self.assertRaises(ValueError):
            get_doc_start_per_pos(
                local_input_ids,
                None,
                attn_mask_startend_row_indices=startend,
                cp_rank=1,
                cp_size=2,
            )

    def test_startend_global_input_ids_with_global_mask_is_named(self):
        # GLOBAL ids (length cp_size * local) under a global mask: ``sq *
        # cp_size`` would ask the mask for cp_size**2 rows. The tell is
        # ``mask_len == sq`` -- ids exactly as long as the mask means they were
        # never down-converted. Without the guard this surfaces as a shape error
        # deep inside the mask walk, cp_size frames from the actual mistake.
        startend = _startend_from_doc_lens([8])
        global_input_ids = paddle.zeros([1, 8], dtype="int64")
        with self.assertRaises(ValueError) as caught:
            get_doc_start_per_pos(
                global_input_ids,
                None,
                attn_mask_startend_row_indices=startend,
                cp_rank=0,
                cp_size=2,
                cp_balance_mode="contiguous_allgather",
            )
        self.assertIn("GLOBAL coordinates", str(caught.exception))

    def test_position_ids_rollover_fallback_returns_start_position(self):
        input_ids = paddle.zeros([1, 6], dtype="int64")
        position_ids = paddle.to_tensor([[0, 1, 2, 0, 1, 2]], dtype="int64")
        doc_start = get_doc_start_per_pos(input_ids, position_ids)
        start = doc_start.numpy()[0]
        self.assertTrue((start[:3] == 0).all())
        self.assertTrue((start[3:] == 3).all())

    def test_no_signal_single_document(self):
        input_ids = paddle.zeros([1, 5], dtype="int64")
        doc_start = get_doc_start_per_pos(input_ids, None)
        self.assertTrue((doc_start.numpy() == 0).all())


if __name__ == "__main__":
    unittest.main()
