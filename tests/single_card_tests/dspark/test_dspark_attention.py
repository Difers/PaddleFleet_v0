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

"""Tiny-model smoke for the DSpark draft attention (CSA ratio=0 shared-KV path).

Builds a ``DsparkHCAAttention`` on a ratio-0 layer of a tiny dsv4-hybrid config
and checks that ``forward_dspark`` (shared target+noise KV selected by
``build_dspark_topk_idxs``, run through the production CSA sparse kernel) drops
the compressor/indexer, returns ``[B, A*K, H]`` and backpropagates.
"""

import unittest

import numpy as np
import paddle
from paddle.distributed.fleet.meta_parallel import build_spec_layer

from paddlefleet.models.common.embeddings.rope_utils import (
    _apply_rotary_pos_emb_bshd,
)
from paddlefleet.models.gpt.gpt_layer_specs import get_attention_spec
from paddlefleet.tensor_parallel.random import model_parallel_cuda_manual_seed
from paddlefleet.training.initialize import initialize_fleet
from paddlefleet.transformer.dspark_attention import (
    DsparkHCAAttention,
    _apply_hca_rope,
    build_dspark_topk_idxs,
    gather_sequence,
)
from paddlefleet.transformer.enums import AttnMaskType
from paddlefleet.transformer.transformer_config import TransformerConfig
from paddlefleet.utils import init_method_normal, scaled_init_method_normal

initialize_fleet(strategy=paddle.distributed.fleet.DistributedStrategy())

_H, _NH, _VHD, _ROPE = 256, 8, 32, 16


def _make_config():
    cfg = TransformerConfig(
        num_hidden_layers=4,
        hidden_size=_H,
        num_attention_heads=_NH,
        params_dtype=paddle.float32,
        bf16=False,
        use_bias=False,
        multi_latent_attention=True,
        experimental_attention_variant="dsv4_hybrid",
        q_lora_rank=64,
        kv_lora_rank=_VHD - _ROPE,
        qk_nope_head_dim=16,
        qk_rope_head_dim=_ROPE,
        qk_pos_emb_head_dim=_ROPE,
        v_head_dim=_VHD,
        hybrid_mla_q_lora_rank=1536,
        hybrid_mla_kv_lora_rank=512,
        hybrid_mla_qk_nope_head_dim=192,
        hybrid_mla_qk_rope_head_dim=64,
        hybrid_mla_v_head_dim=256,
        hybrid_mla_num_attention_heads=64,
        hybrid_mla_num_key_value_heads=64,
        o_groups=4,
        o_lora_rank=32,
        rope_type="rope",
        rotary_base=10000.0,
        rotary_percent=1.0,
        normalization="RMSNorm",
        use_qk_norm=True,
        csa_compress_ratios=[0, 4, 128, 4],
        csa_window_size=16,
        dsa_index_n_heads=4,
        dsa_index_head_dim=32,
        dsa_index_topk=8,
        dsa_indexer_loss_coeff=1.0,
        dsa_indexer_use_sparse_loss=False,
        dsa_indexer_rotary_interleaved=False,
        apply_rope_fusion=False,
        attention_dropout=0.0,
        attention_softmax_in_fp32=True,
        masked_softmax_fusion=False,
        softmax_type="vanilla",
        csa_indexer_backend="unfused",
        csa_sparse_attn_backend="unfused",
        tensor_model_parallel_size=1,
        context_parallel_size=1,
        csa_dense_mode=False,
        init_method=init_method_normal(0.02),
        output_layer_init_method=scaled_init_method_normal(0.02, 1, 2.0),
        rms_norm_eps=1e-5,
    )
    cfg.dtype = "float32"
    return cfg


def _build_attention(cfg):
    model_parallel_cuda_manual_seed(42)
    spec = get_attention_spec(
        config=cfg,
        attention_layer_type="dsv4_hybrid_attention",
        attn_mask_type=AttnMaskType.causal,
    )
    spec.layer = DsparkHCAAttention
    return build_spec_layer(spec, config=cfg, layer_number=0)  # ratio=0


class TestDsparkAttention(unittest.TestCase):
    def _inputs(self, B=1, S=32, A=4, K=3):
        N = A * K
        ctx = paddle.randn([B, S, _H])
        ctx.stop_gradient = False
        noise = paddle.randn([B, N, _H])
        noise.stop_gradient = False
        anchor = paddle.to_tensor([[8, 15, 20, 27]], dtype="int64")
        keep = paddle.ones([B, A], dtype="bool")
        docseg = paddle.zeros([B, S], dtype="int64")
        ctx_pos = paddle.arange(S, dtype="int64").reshape([1, S]).expand([B, S])
        off = paddle.arange(1, K + 1, dtype="int64").reshape([1, 1, K])
        draft_pos = (anchor.unsqueeze(-1) + off).reshape([B, N])
        return ctx, noise, anchor, keep, docseg, ctx_pos, draft_pos, A, K

    def test_ratio0_drops_compressor_and_indexer(self):
        attn = _build_attention(_make_config())
        self.assertIsNone(attn.core_attention.compressor)
        self.assertIsNone(attn.core_attention.indexer)

    def test_forward_shape_and_backward(self):
        attn = _build_attention(_make_config())
        ctx, noise, anchor, keep, docseg, ctx_pos, draft_pos, A, K = (
            self._inputs()
        )
        out = attn.forward_dspark(
            ctx,
            noise,
            anchor_positions=anchor,
            block_keep_mask=keep,
            doc_start_per_pos=docseg,
            context_position_ids=ctx_pos,
            draft_position_ids=draft_pos,
            block_size=K,
        )
        self.assertEqual(out.shape, [1, A * K, _H])
        out.sum().backward()
        self.assertIsNotNone(noise.grad)

    def test_block_internal_rows_are_identical_bidirectional(self):
        # The K draft queries of a block must share one KV index row (so the
        # block-internal attention is query-invariant / bidirectional).
        anchor = paddle.to_tensor([[8, 15, 20, 27]], dtype="int64")
        keep = paddle.ones([1, 4], dtype="bool")
        docseg = paddle.zeros([1, 32], dtype="int64")
        idx = build_dspark_topk_idxs(
            anchor_positions=anchor,
            block_keep_mask=keep,
            doc_start_per_pos=docseg,
            seq_len=32,
            block_size=3,
            window_size=16,
        ).numpy()
        for a in range(4):
            rows = idx[0, a * 3 : (a + 1) * 3]
            self.assertTrue((rows == rows[0]).all())

    def test_window_does_not_straddle_the_zigzag_seam(self):
        # A zigzag shard concatenates two globally distant intervals, so a
        # window crossing the seam would feed the anchor rows it never
        # precedes. One document spans everything here, so only the seam guard
        # can drop them.
        shared = {
            "anchor_positions": paddle.to_tensor([[7, 9]], dtype="int64"),
            "block_keep_mask": paddle.ones([1, 2], dtype="bool"),
            "doc_start_per_pos": paddle.zeros([1, 16], dtype="int64"),
            "seq_len": 16,
            "block_size": 2,
            "window_size": 4,
        }
        no_seam = build_dspark_topk_idxs(**shared).numpy()
        seam = build_dspark_topk_idxs(**shared, chunk_seam=8).numpy()
        # Windows are [4, 5, 6, 7] and [6, 7, 8, 9]; row 0 / row 2 are the two
        # anchors' first draft query.
        np.testing.assert_array_equal(no_seam[0, 0, :4], [4, 5, 6, 7])
        np.testing.assert_array_equal(no_seam[0, 2, :4], [6, 7, 8, 9])
        # Anchor 7 lies in the first interval: untouched. Anchor 9 lies in the
        # second: the two rows from the first interval are dropped.
        np.testing.assert_array_equal(seam[0, 0, :4], [4, 5, 6, 7])
        np.testing.assert_array_equal(seam[0, 2, :4], [-1, -1, 8, 9])


class TestHcaRopeMlaConvention(unittest.TestCase):
    """``_apply_hca_rope`` must reproduce the MLA channel pairing used by the
    DSv4/CSA reference (adjacent pairs ``(2i, 2i+1)``), not the half-split
    default ``(j, j+D/2)``.

    A missing ``multi_latent_attention``/``mla_output_remove_interleaving``
    silently changes the frequency-to-channel map, so the existing shape /
    backward / index tests cannot catch it. This is a direct numerical
    contract test against the reference call.
    """

    _D = 8  # rope (pos) dim

    def _freqs(self, seq_len=5):
        inv = paddle.randn([self._D // 2]) * 0.5
        pos = paddle.arange(seq_len, dtype="float32").reshape([1, seq_len])
        base = pos.unsqueeze(-1) * inv.reshape([1, 1, -1])
        return paddle.concat([base, base], axis=-1).unsqueeze(2)

    def _reference(self, value, freqs, inverse=False):
        # Same call as dsv4_hybrid_attention / csa_attention / the fused
        # mla_rope kernel (see triton_ops/mla_rope_inplace_fusion.py).
        return _apply_rotary_pos_emb_bshd(
            value,
            freqs,
            mscale=1.0,
            rotary_interleaved=False,
            multi_latent_attention=True,
            mla_output_remove_interleaving=True,
            high_precision_rope=False,
            inverse=inverse,
        )

    def test_matches_mla_reference(self):
        value = paddle.randn([1, 5, 3, self._D])
        freqs = self._freqs()
        for inverse in (False, True):
            got = _apply_hca_rope(value, freqs, inverse=inverse, config=None)
            ref = self._reference(value, freqs, inverse)
            np.testing.assert_allclose(
                got.numpy(),
                ref.numpy(),
                rtol=1e-6,
                atol=1e-6,
                err_msg=(
                    f"inverse={inverse}: _apply_hca_rope != MLA reference "
                    "(channel pairing / interleave convention mismatch)"
                ),
            )

    def test_adjacent_pairing_pulse(self):
        # Only the adjacent pair (0, 1) is non-zero. MLA keeps the energy in
        # channels 0/1; the half-split default would leak it into channel D/2.
        value = paddle.zeros([1, 1, 1, self._D])
        value[0, 0, 0, 0] = 1.0
        value[0, 0, 0, 1] = 1.0
        half = paddle.zeros([1, 1, 1, self._D // 2])
        half[..., 0] = 0.3
        freqs = paddle.concat([half, half], axis=-1)
        out = _apply_hca_rope(value, freqs, config=None).numpy()[0, 0, 0]
        self.assertAlmostEqual(float(out[self._D // 2]), 0.0, places=6)
        self.assertGreater(float(abs(out[0]) + abs(out[1])), 0.5)


class TestDsparkGatherSequence(unittest.TestCase):
    """``gather_sequence`` keeps batch rows apart while retaining trailing dims."""

    def test_requires_a_2d_sequence_and_3d_indices(self):
        cases = [
            (
                paddle.ones([4], dtype="float32"),
                paddle.ones([1, 1, 1], dtype="int64"),
            ),
            (
                paddle.ones([1, 4], dtype="float32"),
                paddle.ones([1, 1], dtype="int64"),
            ),
            (
                paddle.ones([1, 4], dtype="float32"),
                paddle.ones([1, 1, 1, 1], dtype="int64"),
            ),
        ]
        for sequence, indices in cases:
            with self.subTest(seq=sequence.shape, idx=indices.shape):
                with self.assertRaises(ValueError) as caught:
                    gather_sequence(sequence, indices)
                self.assertIn("batch_size,seq_len", str(caught.exception))

    def test_batch_sizes_must_agree(self):
        with self.assertRaises(ValueError) as caught:
            gather_sequence(
                paddle.ones([2, 4, 3], dtype="float32"),
                paddle.ones([1, 1, 1], dtype="int64"),
            )
        self.assertIn("batch sizes differ", str(caught.exception))

    def test_trailing_feature_dims_are_retained(self):
        sequence = paddle.arange(2 * 4 * 3, dtype="float32").reshape([2, 4, 3])
        indices = paddle.to_tensor([[[0, 1]], [[3, 2]]], dtype="int64")
        out = gather_sequence(sequence, indices)
        self.assertEqual(list(out.shape), [2, 1, 2, 3])
        np.testing.assert_array_equal(
            out.numpy()[0, 0], sequence.numpy()[0, [0, 1]]
        )
        np.testing.assert_array_equal(
            out.numpy()[1, 0], sequence.numpy()[1, [3, 2]]
        )


if __name__ == "__main__":
    unittest.main()
