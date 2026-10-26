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

"""Packed (``flashmask``) wiring of the HyperEncoder trunk.

The packed path concatenates ``(context, query)`` segments into one ``[1, T]``
sequence and hands the prefix-LM mask to the trunk as 2-column FlashMask row
indices. These tests pin the contracts around that hand-off, with the trunk
stubbed out (identity), so they need no GPU and no distributed init:

* ``HyperEncoderBlock`` takes exactly one mask carrier and forwards the row
  indices to every layer;
* ``HyperEncoderModel.forward_decoder_packed`` (standalone) and
  ``HyperBodyEncoderFrontEnd._forward_packed`` (HyperBody) build row indices
  that describe the same segments they actually concatenated (incl. padding,
  empty prefixes and, for the standalone model, mixed short/long queries), pass
  no dense mask / no ``packed_seq_params``, and slice the query rows back at the
  right offsets.
"""

import types
import unittest

import paddle

from paddlefleet.transformer.prefix_lm_mask import (
    build_dense_mask,
    build_flashmask_row_indices,
    expand_row_indices_to_dense,
    prefix_lm_pad_len,
)
from paddlefleet.transformers.hyperbody.modeling import (
    HyperBodyEncoderFrontEnd,
)
from paddlefleet.transformers.hyperencoder.modeling_fleet import (
    HyperEncoderBlock,
    HyperEncoderModel,
)

_H, _D = 6, 4
_SHORT_Q, _LONG_Q = 3, 5
_ALIGN = 8


def _bare(cls, **attrs):
    """Instantiate a ``FleetLayer`` subclass without running ``__init__``."""
    obj = cls.__new__(cls)
    obj.__dict__.setdefault("_parameters", {})
    obj.__dict__.setdefault("_buffers", {})
    obj.__dict__.setdefault("_sub_layers", {})
    obj.__dict__.setdefault("_non_persistable_buffers", set())
    for key, value in attrs.items():
        object.__setattr__(obj, key, value)
    return obj


def _table(rows, offset):
    """A query table whose rows are distinguishable from every other row."""
    w = paddle.arange(rows * _H, dtype="float32").reshape([rows, _H]) + offset
    return types.SimpleNamespace(weight=w)


def _config(sequence_parallel=False):
    return types.SimpleNamespace(
        tensor_model_parallel_size=1,
        sequence_parallel=sequence_parallel,
        hyperencoder_seq_align=_ALIGN,
    )


def _rope_stub(max_seqlen, seg_lens):
    return paddle.ones([1, sum(seg_lens), 1, _D])


class _RecordingBlock:
    """Identity trunk that records what it was called with."""

    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return kwargs["hidden_states"]


def _contexts(lens):
    paddle.seed(3)
    return [paddle.randn([c, _H]) * 100.0 + 1e5 for c in lens]


# (context lengths, use_long_query, description)
_PACKS = [
    ([4, 2], False, "homogeneous short"),
    ([3], True, "single long segment"),
    ([2, 0, 5], [True, False, True], "mixed queries + empty prefix"),
]


def _expected_geometry(ctx_lens, use_long_query):
    uql = (
        [use_long_query] * len(ctx_lens)
        if isinstance(use_long_query, bool)
        else list(use_long_query)
    )
    nq = [_LONG_Q if u else _SHORT_Q for u in uql]
    seq_len = sum(c + q for c, q in zip(ctx_lens, nq))
    pad = prefix_lm_pad_len(seq_len, 1, False, align=_ALIGN)
    return uql, nq, seq_len, pad


class TestHyperEncoderBlockMaskCarrier(unittest.TestCase):
    def _block(self):
        seen = []

        def _layer(dict_args):
            seen.append(dict(dict_args))
            return dict_args

        block = _bare(
            HyperEncoderBlock,
            config=types.SimpleNamespace(sequence_parallel=False),
            pre_process=True,
            layers=[_layer, _layer],
            norm=None,
        )
        return block, seen

    def test_requires_exactly_one_mask(self):
        block, _ = self._block()
        h = paddle.zeros([1, 4, _H])
        mask = paddle.zeros([1, 1, 4, 4], dtype="bool")
        rows = paddle.zeros([1, 1, 4, 2], dtype="int32")
        for kw in (
            {},
            {"attention_mask": mask, "attn_mask_startend_row_indices": rows},
        ):
            with self.subTest(keys=sorted(kw)), self.assertRaises(ValueError):
                block.forward(h, **kw)

    def test_row_indices_reach_every_layer(self):
        block, seen = self._block()
        h = paddle.zeros([1, 4, _H])
        rows = build_flashmask_row_indices([1], [3])
        out = block.forward(h, attn_mask_startend_row_indices=rows)
        self.assertEqual(len(seen), 2)
        for d in seen:
            self.assertIs(d["attn_mask_startend_row_indices"], rows)
            self.assertIsNone(d["attention_mask"])
        self.assertIs(out, h)

    def test_dense_mask_reaches_every_layer(self):
        block, seen = self._block()
        mask = build_dense_mask([1], [3])
        block.forward(paddle.zeros([1, 4, _H]), attention_mask=mask)
        for d in seen:
            self.assertIs(d["attention_mask"], mask)
            self.assertIsNone(d["attn_mask_startend_row_indices"])


class TestForwardDecoderPacked(unittest.TestCase):
    """Standalone ``HyperEncoderModel.forward_decoder_packed``."""

    def _model(self, sequence_parallel=False):
        block = _RecordingBlock()
        model = _bare(
            HyperEncoderModel,
            config=_config(sequence_parallel),
            query_short=_table(_SHORT_Q, 1000.0),
            query_long=_table(_LONG_Q, 2000.0),
            block=block,
            _packed_rope=_rope_stub,
            # time-major round trip for the SP branch (TP=1: no real split)
            _sp_scatter=lambda x: x.transpose([1, 0, 2]),
            _sp_gather=lambda h: h,
        )
        return model, block

    def _check(self, sequence_parallel):
        for ctx_lens, uql_in, desc in _PACKS:
            with self.subTest(desc=desc, sp=sequence_parallel):
                model, block = self._model(sequence_parallel)
                ctx = _contexts(ctx_lens)
                out = model.forward_decoder_packed(ctx, uql_in)
                uql, nq, seq_len, pad = _expected_geometry(ctx_lens, uql_in)

                (call,) = block.calls
                self.assertNotIn("attention_mask", call)
                self.assertNotIn("packed_seq_params", call)
                rows = call["attn_mask_startend_row_indices"]
                want = build_flashmask_row_indices(ctx_lens, nq, pad)
                self.assertTrue(bool(paddle.equal_all(rows, want)))
                rope = call["rotary_pos_emb"]
                if sequence_parallel:
                    self.assertEqual(rope.shape, [seq_len + pad, 1, 1, _D])
                    self.assertEqual(
                        call["hidden_states"].shape, [seq_len + pad, 1, _H]
                    )
                else:
                    self.assertEqual(rope.shape, [1, seq_len + pad, 1, _D])
                    self.assertEqual(
                        call["hidden_states"].shape, [1, seq_len + pad, _H]
                    )

                # Identity trunk: each segment's output must be its query table.
                outs = out if isinstance(out, list) else list(out)
                self.assertEqual(len(outs), len(ctx_lens))
                for o, u in zip(outs, uql):
                    table = (
                        model.query_long if u else model.query_short
                    ).weight
                    self.assertTrue(bool(paddle.equal_all(o, table)))

    def test_non_sp(self):
        self._check(sequence_parallel=False)

    def test_sp(self):
        self._check(sequence_parallel=True)

    def test_rectangular_input_matches_list_input(self):
        model, _ = self._model()
        ctx = _contexts([3, 3])
        from_list = model.forward_decoder_packed(ctx, False)
        model2, _ = self._model()
        from_rect = model2.forward_decoder_packed(paddle.stack(ctx), False)
        self.assertEqual(from_rect.shape, [2, _SHORT_Q, _H])
        self.assertTrue(bool(paddle.equal_all(from_list, from_rect)))

    def test_use_long_query_length_mismatch_raises(self):
        model, _ = self._model()
        with self.assertRaises(ValueError):
            model.forward_decoder_packed(_contexts([2, 2]), [True])


class TestFrontEndForwardPacked(unittest.TestCase):
    """HyperBody ``HyperBodyEncoderFrontEnd._forward_packed``.

    The frontend has a single query table, appended after every segment.
    """

    def _frontend(self):
        return _bare(
            HyperBodyEncoderFrontEnd,
            config=_config(),
            query=_table(_SHORT_Q, 1000.0),
            _packed_rope=_rope_stub,
        )

    def test_row_indices_describe_the_concatenated_layout(self):
        for ctx_lens, desc in (
            ([4, 2], "two segments"),
            ([3], "single segment"),
            ([2, 0, 5], "empty prefix in the middle"),
        ):
            with self.subTest(desc=desc):
                out = self._frontend()._forward_packed(_contexts(ctx_lens))
                _, nq, seq_len, pad = _expected_geometry(ctx_lens, False)
                # Only the row indices carry the mask on this path.
                self.assertIsNone(out["attention_mask"])
                self.assertIsNone(out["packed_seq_params"])
                rows = out["attn_mask_startend_row_indices"]
                self.assertEqual(rows.shape, [1, 1, seq_len + pad, 2])
                dense = build_dense_mask(ctx_lens, nq, pad)
                self.assertTrue(
                    bool(
                        paddle.equal_all(
                            expand_row_indices_to_dense(rows), dense
                        )
                    )
                )

                # The bridge slices with _hb_packed: its segments must be the
                # ones the row indices isolate.
                layout = out["_hb_packed"]
                self.assertEqual(list(layout["n_contexts"]), ctx_lens)
                self.assertEqual(list(layout["n_queries"]), nq)
                h = out["hidden_states"]
                self.assertEqual(h.shape, [1, seq_len + pad, _H])
                table = _table(_SHORT_Q, 1000.0).weight
                for s, c, q in zip(layout["segment_starts"], ctx_lens, nq):
                    self.assertTrue(
                        bool(paddle.equal_all(h[0, s + c : s + c + q], table))
                    )
                if pad:
                    self.assertEqual(float(h[0, seq_len:].abs().max()), 0.0)
                self.assertEqual(out["seq_len"], seq_len)


if __name__ == "__main__":
    unittest.main()
