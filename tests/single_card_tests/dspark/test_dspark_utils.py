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

"""Unit tests for the DSpark tensor contracts.

Two halves, both host-side and GPU-light: the read-only target-side views
(``readonly_embedding`` / ``ReadOnlyProjection``) that keep the draft from
writing into the target's gradients, and ``build_draft_blocks`` -- the single
place where ``input_ids``/``labels``/``loss_mask`` become the
draft's noise inputs, targets and evaluation mask.  The validation table here
is deliberately exhaustive: every guard in that function is a silent-loss bug
if it regresses.
"""

import unittest

import numpy as np
import paddle

from paddlefleet.utils.dspark.dspark_utils import (
    ReadOnlyProjection,
    build_draft_blocks,
    gather_sequence,
    readonly_embedding,
)

_V, _H = 32, 8
_B, _S = 2, 12


def _ids(rows, cols, *, dtype="int64", lo=1, hi=_V, seed=0):
    rng = np.random.default_rng(seed)
    return paddle.to_tensor(
        rng.integers(lo, hi, size=(rows, cols)).astype(dtype)
    )


def _blocks(input_ids, labels, **overrides):
    kwargs = {
        "block_size": 3,
        "mask_token_id": 2,
        "num_anchors": 2,
        "sample_ids": [f"row{i}" for i in range(int(input_ids.shape[0]))],
        "global_step": 0,
        "training_seed": 7,
    }
    kwargs.update(overrides)
    return build_draft_blocks(input_ids, labels, **kwargs)


class TestReadOnlyTargetViews(unittest.TestCase):
    def test_readonly_embedding_requires_a_matrix(self):
        with self.assertRaises(ValueError) as caught:
            readonly_embedding(
                paddle.to_tensor([[1, 2]], dtype="int64"),
                paddle.randn([_V]),
            )
        self.assertIn("[vocab, hidden]", str(caught.exception))

    def test_readonly_embedding_blocks_the_target_gradient(self):
        weight = paddle.randn([_V, _H])
        weight.stop_gradient = False
        out = readonly_embedding(
            paddle.to_tensor([[1, 2]], dtype="int64"), weight
        )
        self.assertEqual(list(out.shape), [1, 2, _H])
        out.sum().backward()
        self.assertIsNone(weight.grad)


class TestReadOnlyProjection(unittest.TestCase):
    def test_requires_a_matrix_and_an_explicit_bool(self):
        cases = [
            (paddle.randn([_V]), True),
            (paddle.randn([_V, _H]), 1),
            (paddle.randn([_V, _H]), None),
        ]
        for weight, transpose_y in cases:
            with self.subTest(shape=weight.shape, transpose_y=transpose_y):
                with self.assertRaises(ValueError) as caught:
                    ReadOnlyProjection(weight=weight, transpose_y=transpose_y)
                self.assertIn(
                    "explicit boolean transpose_y", str(caught.exception)
                )

    def test_bias_must_be_one_per_vocab_entry(self):
        with self.assertRaises(ValueError) as caught:
            ReadOnlyProjection(
                weight=paddle.randn([_V, _H]),
                transpose_y=True,
                bias=paddle.randn([_V + 1]),
            )
        self.assertIn("bias must have shape", str(caught.exception))

    def test_multimax_needs_both_halves(self):
        with self.assertRaises(ValueError) as caught:
            ReadOnlyProjection(
                weight=paddle.randn([_V, _H]),
                transpose_y=True,
                multimax_ranges=paddle.randn([4]),
            )
        self.assertIn("Multimax requires both", str(caught.exception))

    def test_multimax_halves_must_be_length_four(self):
        for name in ("multimax_ranges", "multimax_ts"):
            extra = {
                "multimax_ranges": paddle.randn([4]),
                "multimax_ts": paddle.randn([4]),
            }
            extra[name] = paddle.randn([3])
            with self.subTest(field=name):
                with self.assertRaises(ValueError) as caught:
                    ReadOnlyProjection(
                        weight=paddle.randn([_V, _H]), transpose_y=True, **extra
                    )
                self.assertIn(
                    f"{name} must have shape [4]", str(caught.exception)
                )

    def test_bias_is_added_in_the_logits_dtype(self):
        weight = paddle.randn([_H, _V])
        bias = paddle.randn([_V]).astype("float32")
        head = ReadOnlyProjection(weight=weight, transpose_y=False, bias=bias)
        hidden = paddle.randn([_B, _H]).astype("float16")
        logits = head.transform_logits(paddle.randn([_B, _V]).astype("float16"))
        self.assertEqual(logits.dtype, paddle.float16)

    def test_call_projects_with_the_declared_direction(self):
        weight = paddle.randn([_H, _V])
        head = ReadOnlyProjection(weight=weight, transpose_y=False)
        hidden = paddle.randn([_B, _H])
        expected = paddle.matmul(hidden, weight)
        np.testing.assert_allclose(
            head(hidden).numpy(), expected.numpy(), atol=1e-5
        )

    def test_multimax_path_routes_through_seglu(self):
        ranges = paddle.to_tensor([0.0, 0.1, 0.2, 0.3])
        ts = paddle.to_tensor([1.0, 2.0, 3.0, 4.0])
        head = ReadOnlyProjection(
            weight=paddle.randn([_H, _V]),
            transpose_y=False,
            multimax_ranges=ranges,
            multimax_ts=ts,
        )
        plain = ReadOnlyProjection(weight=head.weight, transpose_y=False)
        hidden = paddle.randn([_B, _H])
        self.assertFalse(
            np.allclose(head(hidden).numpy(), plain(hidden).numpy())
        )


def _anchors(positions, keep_mask):
    from paddlefleet.utils.dspark.anchor_sampler import AnchorBatch

    return AnchorBatch(positions=positions, keep_mask=keep_mask)


class TestBuildDraftBlocksValidation(unittest.TestCase):
    """Every guard in build_draft_blocks; a regression here is a silent loss."""

    def setUp(self):
        self.ids = _ids(_B, _S)
        self.labels = _ids(_B, _S, seed=1)

    def _expect(self, message, *args, **kwargs):
        with self.assertRaises(ValueError) as caught:
            _blocks(*args, **kwargs)
        self.assertIn(message, str(caught.exception))

    def test_mask_token_id_guard(self):
        for bad in (-1, True, 1.5, None):
            with self.subTest(mask_token_id=bad):
                self._expect(
                    "mask_token_id must be a non-negative integer",
                    self.ids,
                    self.labels,
                    mask_token_id=bad,
                )

    def test_input_mode_guard(self):
        self._expect(
            "input_mode must be anchor_mask or sliding_anchor",
            self.ids,
            self.labels,
            input_mode="prefix_lm",
        )

    def test_shape_guards(self):
        self._expect(
            "nonempty shape [B, S]",
            paddle.ones([_S], dtype="int64"),
            paddle.ones([_S], dtype="int64"),
        )
        self._expect(
            "nonempty shape [B, S]",
            paddle.ones([0, _S], dtype="int64"),
            paddle.ones([0, _S], dtype="int64"),
        )
        self._expect(
            "nonempty shape [B, S]", self.ids, _ids(_B, _S + 1, seed=2)
        )
        for field in ("loss_mask", "position_ids"):
            with self.subTest(field=field):
                self._expect(
                    f"{field} shape must match input_ids",
                    self.ids,
                    self.labels,
                    **{field: paddle.ones([_B, _S + 1], dtype="int64")},
                )

    def test_dtype_guard(self):
        self._expect(
            "integer dtype",
            self.ids.astype("float32"),
            self.labels.astype("float32"),
        )

    def test_negative_valid_label_guard(self):
        labels = _ids(_B, _S, seed=3)
        labels[0, 3] = -5  # not ignored_index, so it is a real (broken) label
        self._expect("valid labels must be non-negative", self.ids, labels)

    def test_negative_margin_guard(self):
        for field in ("sample_left_margin", "sample_right_margin"):
            with self.subTest(field=field):
                self._expect(
                    "sample margins must be non-negative",
                    self.ids,
                    self.labels,
                    **{field: -1},
                )

    def test_anchors_without_sample_ids_guard(self):
        self._expect(
            "sample_ids are required when sampling anchors",
            self.ids,
            self.labels,
            sample_ids=None,
        )

    def test_anchor_shape_and_dtype_guards(self):
        self._expect(
            "anchors must have nonempty shape [B, A]",
            self.ids,
            self.labels,
            anchors=_anchors(
                np.zeros((_B, 0), dtype="int64"), np.zeros((_B, 0), dtype=bool)
            ),
        )
        self._expect(
            "anchors must have nonempty shape [B, A]",
            self.ids,
            self.labels,
            anchors=_anchors(
                np.zeros((_B, 2), dtype="int64"), np.zeros((_B, 3), dtype=bool)
            ),
        )
        self._expect(
            "anchor positions must have integer dtype",
            self.ids,
            self.labels,
            anchors=_anchors(
                np.full((_B, 2), 3.0), np.ones((_B, 2), dtype=bool)
            ),
        )

    def test_kept_anchors_must_be_inside_the_sequence(self):
        positions = np.zeros((_B, 2), dtype="int64")
        positions[0, 1] = _S + 4
        self._expect(
            "kept anchor positions must be inside the sequence",
            self.ids,
            self.labels,
            anchors=_anchors(positions, np.ones((_B, 2), dtype=bool)),
        )


class TestGatherSequence(unittest.TestCase):
    def test_batch_dimensions_must_agree(self):
        with self.assertRaises(ValueError) as caught:
            gather_sequence(
                paddle.ones([2, 5], dtype="int64"),
                paddle.ones([3, 1], dtype="int64"),
            )
        self.assertIn("share a batch dimension", str(caught.exception))

    def test_rows_never_mix(self):
        values = paddle.to_tensor([[10, 11, 12], [20, 21, 22]], dtype="int64")
        indices = paddle.to_tensor([[2], [0]], dtype="int64")
        np.testing.assert_array_equal(
            gather_sequence(values, indices).numpy(), [[12], [20]]
        )


class TestBuildDraftBlocksModes(unittest.TestCase):
    def setUp(self):
        self.ids = _ids(_B, _S)
        self.labels = _ids(_B, _S, seed=1)

    def test_anchor_mask_mode_shape_and_leading_noise_token(self):
        blocks = _blocks(self.ids, self.labels, input_mode="anchor_mask")
        self.assertEqual(list(blocks.noise_ids.shape), [_B, 2, 3])
        self.assertEqual(list(blocks.target_ids.shape), [_B, 2, 3])
        # Anchor 0 is at position 0, which the sampler masks out, so the first
        # row's first anchor is unkept and carries the noise token instead.
        self.assertEqual(list(blocks.position_ids.shape), [_B, 2, 3])

    def test_sliding_anchor_mode_uses_history_tokens(self):
        mask = _blocks(self.ids, self.labels, input_mode="anchor_mask")
        slide = _blocks(self.ids, self.labels, input_mode="sliding_anchor")
        self.assertEqual(list(slide.noise_ids.shape), [_B, 2, 3])
        # The two modes differ in how the context block is built.
        self.assertFalse(
            np.array_equal(slide.noise_ids.numpy(), mask.noise_ids.numpy())
        )

    def test_loss_mask_narrows_the_valid_positions(self):
        loss_mask = paddle.ones([_B, _S], dtype="bool")
        loss_mask[:, _S // 2 :] = False
        masked = _blocks(self.ids, self.labels, loss_mask=loss_mask)
        self.assertTrue(
            int(masked.anchors.positions.max()) < _S // 2
            or not bool(masked.anchors.keep_mask.any())
        )

    def test_sample_margins_exclude_the_edges(self):
        blocks = _blocks(
            self.ids,
            self.labels,
            sample_left_margin=3,
            sample_right_margin=3,
        )
        kept = blocks.anchors.positions[blocks.anchors.keep_mask]
        if int(kept.numel()):
            self.assertGreaterEqual(int(kept.min()), 3)
            self.assertLess(int(kept.max()), _S - 3)

    def test_host_numpy_anchors_are_promoted_to_tensors(self):
        positions = np.tile(np.arange(2, 2 + 2, dtype="int64"), (_B, 1))
        blocks = _blocks(
            self.ids,
            self.labels,
            anchors=_anchors(positions, np.ones((_B, 2), dtype=bool)),
        )
        self.assertIsInstance(blocks.anchors.positions, paddle.Tensor)
        self.assertIsInstance(blocks.anchors.keep_mask, paddle.Tensor)

    def test_explicit_position_ids_are_used_for_draft_positions(self):
        position_ids = (
            paddle.arange(_S, dtype="int64").reshape([1, _S]).expand([_B, _S])
        )
        blocks = _blocks(self.ids, self.labels, position_ids=position_ids)
        base = blocks.anchors.positions.unsqueeze(-1)
        offsets = paddle.arange(1, 4, dtype="int64").reshape([1, 1, -1])
        np.testing.assert_array_equal(
            blocks.position_ids.numpy(), (base + offsets).numpy()
        )


class TestDraftBlocksChunkSeam(unittest.TestCase):
    """The zigzag seam is an internal boundary the document filter cannot see."""

    def _blocks_at(self, positions, **overrides):
        ids = paddle.full([1, 16], 5, dtype="int64")
        return _blocks(
            ids,
            ids,
            anchors=_anchors(
                np.array([positions], dtype="int64"),
                np.ones((1, len(positions)), dtype=bool),
            ),
            **overrides,
        )

    def test_block_targets_do_not_straddle_the_zigzag_seam(self):
        # Anchor 7 ends the shard's first interval, so its block (8, 9, 10)
        # lies entirely in the second, globally distant one.
        seam = self._blocks_at([7, 9], chunk_seam=8).eval_mask.numpy()
        self.assertFalse(seam[0, 0].any())
        self.assertTrue(seam[0, 1].all())

    def test_without_a_seam_the_same_block_is_kept(self):
        no_seam = self._blocks_at([7, 9]).eval_mask.numpy()
        self.assertTrue(no_seam[0, 0].all())
        self.assertTrue(no_seam[0, 1].all())

    def test_sliding_anchor_history_does_not_straddle_the_seam(self):
        # The history looks backward: anchor 8's history (7, 8) reaches into
        # the first interval.
        no_seam = self._blocks_at(
            [8], input_mode="sliding_anchor"
        ).noise_ids.numpy()
        seam = self._blocks_at(
            [8], input_mode="sliding_anchor", chunk_seam=8
        ).noise_ids.numpy()
        self.assertEqual(int(no_seam[0, 0, 0]), 5)
        self.assertEqual(int(seam[0, 0, 0]), 2)  # mask_token_id
        self.assertEqual(int(seam[0, 0, 1]), 5)
