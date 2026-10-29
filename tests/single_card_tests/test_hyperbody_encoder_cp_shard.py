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

"""``hyperencoder_cp_shard``: CP over the HyperBody encoder trunk (no GPU,
no dist init).

* config contract: inert at CP=1, follows the configured allgather mode, widens
  the pad to ``2*cp`` and refuses layouts whose CP math was not checked;
* the frontend scatters x / rope and keeps the row indices global, the bridge
  gathers back before its per-segment slice;
* the CP emulation replays what ``flashmask_attention_cp`` does per rank (local
  queries, all-gathered K/V, ``preprocess_index*`` on the global prefix-LM row
  indices) and checks it against full-sequence attention, CP=2/4, both layouts;
* grad-scale marks: under the shard the trunk is left unmarked.
"""

import math
import types
import unittest
from unittest import mock

import paddle

from paddlefleet.context_parallel_utils import (
    preprocess_index,
    preprocess_index_dual_chunks,
)
from paddlefleet.cp_shard import extract_local_cp_chunks
from paddlefleet.transformer.prefix_lm_mask import (
    build_dense_mask,
    build_flashmask_row_indices,
    prefix_lm_pad_len,
)
from paddlefleet.transformers.hyperbody import modeling as hb

_H, _D = 6, 4
_Q = 3
_ALIGN = 8
_MODES = ("contiguous_allgather", "dualchunk_allgather")


def _bare(cls, **attrs):
    obj = cls.__new__(cls)
    obj.__dict__.setdefault("_parameters", {})
    obj.__dict__.setdefault("_buffers", {})
    obj.__dict__.setdefault("_sub_layers", {})
    obj.__dict__.setdefault("_non_persistable_buffers", set())
    for key, value in attrs.items():
        object.__setattr__(obj, key, value)
    return obj


def _enc_cfg(cp=4, shard=True, mode="contiguous_allgather"):
    """Encoder view as ``_configure_encoder_cp`` leaves it (align already lcm'd)."""
    return types.SimpleNamespace(
        tensor_model_parallel_size=1,
        sequence_parallel=False,
        hyperencoder_seq_align=math.lcm(_ALIGN, 2 * cp) if shard else _ALIGN,
        hyperencoder_cp_shard=shard,
        context_parallel_size=cp,
        cp_balance_mode=mode,
    )


def _local_slice(x, rank, cp, mode, axis=1):
    return extract_local_cp_chunks(x, rank, cp, axis=axis, mode=mode)


def _gather_all(locals_, cp, mode, axis=1):
    """Inverse of the per-rank layout (what ContextParallelGatherOp returns)."""
    if mode.startswith("contiguous"):
        return paddle.concat(locals_, axis=axis)
    firsts, seconds = [], []
    for loc in locals_:
        a, b = paddle.split(loc, 2, axis=axis)
        firsts.append(a)
        seconds.append(b)
    return paddle.concat(firsts + seconds[::-1], axis=axis)


# ---- config contract, through the real HyperBodyConfig -> encoder view ---- #
_DECODER = {
    "vocab_size": 128,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 1,
    "num_attention_heads": 4,
    "n_routed_experts": 4,
    "moe_intermediate_size": 32,
    "num_experts_per_tok": 2,
    "n_shared_experts": 1,
}
_ENCODER = {
    "vocab_size": 128,
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "moe_intermediate_size": 128,
    "n_routed_experts": 8,
    "num_experts_per_tok": 2,
    "n_shared_experts": 1,
    "first_k_dense_replace": 1,
    "hyperencoder_query_length": 16,
    "hyperencoder_attn_backend": "flashmask",
    "hyperencoder_packed_decoder": True,
    "hyperencoder_cp_shard": True,
}


def _view(cp=4, enc=None, **top):
    from paddlefleet.transformers.hyperbody.configuration import HyperBodyConfig

    cfg = HyperBodyConfig(
        decoder_config=dict(_DECODER),
        encoder_config=dict(_ENCODER, **(enc or {})),
        context_parallel_size=cp,
        expert_model_parallel_size=cp,
        **top,
    )
    return hb._build_encoder_view(cfg, _DECODER["hidden_size"])


class TestEncoderCpShardConfig(unittest.TestCase):
    def test_inert_at_cp1_and_off(self):
        for cp, shard in ((1, True), (4, False)):
            with self.subTest(cp=cp, shard=shard):
                view = _view(cp=cp, enc={"hyperencoder_cp_shard": shard})
                self.assertFalse(view.hyperencoder_cp_shard)
                self.assertTrue(view.cp_replicated_sequence)
                self.assertEqual(view.hyperencoder_seq_align, 128)

    def test_follows_configured_mode(self):
        for mode in _MODES:
            with self.subTest(mode=mode):
                view = _view(cp_balance_mode=mode)
                self.assertTrue(view.hyperencoder_cp_shard)
                self.assertFalse(view.cp_replicated_sequence)
                self.assertEqual(view.cp_balance_mode, mode)

    def test_allows_per_token_loss(self):
        view = _view(calculate_per_token_loss=True)
        self.assertTrue(view.hyperencoder_cp_shard)
        self.assertFalse(view.cp_replicated_sequence)

    def test_rejects_flashmask_cp_overlap(self):
        for mode in _MODES:
            with (
                self.subTest(mode=mode),
                self.assertRaises(NotImplementedError),
            ):
                _view(cp_balance_mode=mode, flashmask_cp_overlap=True)
        # CP=1 / shard off stay inert even with the decoder on overlap.
        view = _view(cp=1, flashmask_cp_overlap=True)
        self.assertFalse(view.hyperencoder_cp_shard)

    def test_marks_aux_by_lm_tokens(self):
        """Per-token training only flips the aux marker on the encoder view.

        ``calculate_per_token_loss`` itself must NOT reach the encoder: its
        router aux is a native pack mean over the encoder token pool (context +
        learned query), which is not the LM supervision pool.
        """
        for per_token in (False, True):
            with self.subTest(per_token=per_token):
                view = _view(calculate_per_token_loss=per_token)
                self.assertEqual(
                    bool(view.hyperencoder_aux_by_lm_tokens), per_token
                )
                self.assertFalse(
                    bool(getattr(view, "calculate_per_token_loss", False))
                )

    def test_marks_aux_by_lm_tokens_without_cp_shard(self):
        # The aux correction is a per-token issue, not a CP one: it stays on
        # when the encoder is replicated (cp=1 here) too.
        view = _view(cp=1, calculate_per_token_loss=True)
        self.assertFalse(view.hyperencoder_cp_shard)
        self.assertTrue(view.hyperencoder_aux_by_lm_tokens)

    def test_rejects_pp_gt1_with_per_token(self):
        # The encoder aux scale rides a per-forward module stash, correct only
        # at PP=1 (micro-batches run forward->backward one at a time).
        with self.assertRaises(AssertionError):
            _view(calculate_per_token_loss=True, pipeline_model_parallel_size=2)

    def test_rejects_encoder_z_loss_under_per_token(self):
        # An enabled encoder z-loss would be silently 1/T_global too small.
        with self.assertRaises(NotImplementedError):
            _view(
                calculate_per_token_loss=True,
                enc={"router_z_loss_coef": 0.01},
            )
        # Off (the default) stays allowed.
        _view(calculate_per_token_loss=True, enc={"router_z_loss_coef": 0.0})
        # And z-loss itself is fine when per-token is off.
        _view(calculate_per_token_loss=False, enc={"router_z_loss_coef": 0.01})

    def test_widens_seq_align_to_2cp(self):
        view = _view(cp=4, enc={"hyperencoder_seq_align": 12})
        self.assertEqual(view.hyperencoder_seq_align, 24)  # lcm(12, 8)
        self.assertEqual(_view(cp=4).hyperencoder_seq_align, 128)

    def test_rejects_unchecked_layouts(self):
        cases = {
            "a2a": ({"cp_balance_mode": "contiguous_a2a"}, ValueError),
            "local aux": (
                {"moe_router_load_balancing_type": "aux_loss"},
                NotImplementedError,
            ),
            "tp": ({"tensor_model_parallel_size": 2}, NotImplementedError),
        }
        for name, (top, err) in cases.items():
            with self.subTest(name), self.assertRaises(err):
                _view(**top)

    def test_requires_packed_flashmask(self):
        with self.assertRaises(ValueError):
            _view(
                enc={
                    "hyperencoder_attn_backend": "dp",
                    "hyperencoder_packed_decoder": False,
                }
            )


# ---- cut / merge points ---------------------------------------------------- #
def _contexts(lens, seed=3):
    paddle.seed(seed)
    return [paddle.randn([c, _H]) for c in lens]


def _frontend(cfg):
    w = paddle.arange(_Q * _H, dtype="float32").reshape([_Q, _H]) + 1000.0
    return _bare(
        hb.HyperBodyEncoderFrontEnd,
        config=cfg,
        # Single shared query table (this base has no short/long split).
        query=types.SimpleNamespace(weight=w),
        # ``_bare`` skips ``__init__`` (which sets ``self.packed`` from the
        # config) while ``forward`` / ``_forward_text_only`` read it.
        packed=True,
        _packed_rope=lambda m, seg_lens: paddle.arange(
            sum(seg_lens) * _D, dtype="float32"
        ).reshape([1, sum(seg_lens), 1, _D]),
    )


def _scatter_as(rank, cp):
    """Stand-in for ContextParallelScatterOp.apply on CP rank ``rank``."""
    return mock.patch.object(
        hb.ContextParallelScatterOp,
        "apply",
        side_effect=lambda x, axis, mode: _local_slice(x, rank, cp, mode, axis),
    )


class TestFrontEndCut(unittest.TestCase):
    """The frontend pads to 2*cp and scatters x / rope; rows stay global."""

    def test_cut(self):
        for cp, mode in (
            (4, "contiguous_allgather"),
            (4, "dualchunk_allgather"),
            (3, "contiguous_allgather"),
        ):
            with self.subTest(cp=cp, mode=mode):
                cfg = _enc_cfg(cp=cp, mode=mode)
                ctx_lens = [4, 2]
                ref_cfg = _enc_cfg(cp=cp, shard=False)
                ref_cfg.hyperencoder_seq_align = cfg.hyperencoder_seq_align
                full = _frontend(ref_cfg)._forward_packed(_contexts(ctx_lens))
                total = full["hidden_states"].shape[1]
                self.assertEqual(total % (2 * cp), 0)

                outs = []
                for r in range(cp):
                    with _scatter_as(r, cp):
                        outs.append(
                            _frontend(cfg)._forward_packed(_contexts(ctx_lens))
                        )
                for o in outs:
                    self.assertEqual(
                        o["hidden_states"].shape, [1, total // cp, _H]
                    )
                    self.assertEqual(
                        o["rotary_pos_emb"].shape, [1, total // cp, 1, _D]
                    )
                    # Row indices stay global: the CP kernel slices them itself.
                    self.assertTrue(
                        bool(
                            paddle.equal_all(
                                o["attn_mask_startend_row_indices"],
                                full["attn_mask_startend_row_indices"],
                            )
                        )
                    )
                    self.assertEqual(o["_hb_packed"], full["_hb_packed"])
                for key in ("hidden_states", "rotary_pos_emb"):
                    merged = _gather_all([o[key] for o in outs], cp, mode)
                    self.assertTrue(
                        bool(paddle.equal_all(merged, full[key])), key
                    )

    def test_text_only_dummy_pads_to_2cp(self):
        with _scatter_as(0, 4):
            out = _frontend(_enc_cfg(cp=4))._forward_packed(_contexts([1]))
        self.assertEqual(out["hidden_states"].shape, [1, 2, _H])  # 8 / cp4

    def test_off_does_not_scatter(self):
        with mock.patch.object(hb.ContextParallelScatterOp, "apply") as m:
            out = _frontend(_enc_cfg(shard=False))._forward_packed(
                _contexts([4, 2])
            )
        m.assert_not_called()
        self.assertEqual(out["hidden_states"].shape, [1, 16, _H])


class TestBridgeMerge(unittest.TestCase):
    """The bridge gathers to global before the per-segment query slice."""

    def _bridge(self, cfg):
        return _bare(
            hb.HyperBodyEncoderBridge,
            config=cfg,
            final_norm=lambda h: h,
            _project=lambda latents: latents,
        )

    def _full(self):
        out = _frontend(_enc_cfg(shard=False))._forward_packed(
            _contexts([4, 2])
        )
        return out["hidden_states"], out["_hb_packed"]

    def test_gathers_then_slices_queries(self):
        cfg = _enc_cfg(cp=4)
        full, layout = self._full()
        local = _local_slice(full, 0, 4, cfg.cp_balance_mode)
        with mock.patch.object(
            hb.ContextParallelGatherOp, "apply", return_value=full
        ) as m:
            got = self._bridge(cfg)._forward_packed(
                {"hidden_states": local}, layout
            )
        m.assert_called_once()
        self.assertIs(m.call_args.args[0], local)
        self.assertEqual(
            m.call_args.kwargs, {"axis": 1, "mode": cfg.cp_balance_mode}
        )
        want = paddle.concat(
            [
                full[0, s + c : s + c + q]
                for s, c, q in zip(
                    layout["segment_starts"],
                    layout["n_contexts"],
                    layout["n_queries"],
                )
            ]
        )
        self.assertTrue(bool(paddle.equal_all(got, want)))

    def test_off_does_not_gather(self):
        full, layout = self._full()
        with mock.patch.object(hb.ContextParallelGatherOp, "apply") as m:
            self._bridge(_enc_cfg(shard=False))._forward_packed(
                {"hidden_states": full}, layout
            )
        m.assert_not_called()


# ---- flashmask CP emulation ------------------------------------------------ #
def _kernel_forbidden(rows_lts, rows_ute, n_rows):
    """Non-causal 2-col FlashMask semantics: row r of key j is masked iff
    ``r >= LTS[j]`` or ``r < UTE[j]`` (no diagonal term)."""
    r = paddle.arange(n_rows, dtype="int64").reshape([n_rows, 1])
    lts = rows_lts.astype("int64").reshape([1, -1])
    ute = rows_ute.astype("int64").reshape([1, -1])
    return (r >= lts) | (r < ute)


def _attn(q, k, v, forbidden):
    """fp32 masked attention. q [Tq,h,d], k/v [Tk,h,d], forbidden [Tq,Tk]."""
    s = paddle.einsum("qhd,khd->hqk", q, k) / math.sqrt(q.shape[-1])
    s = paddle.where(
        forbidden.unsqueeze(0), paddle.full_like(s, float("-inf")), s
    )
    return paddle.einsum("hqk,khd->qhd", paddle.nn.functional.softmax(s, -1), v)


def _local_row_indices(rows, rank, cp, mode, q_local):
    """Exactly the per-rank index preprocessing of
    ``cp_flashmask_allgatherkv_balance_forward``."""
    if mode == "dualchunk_allgather":
        bs = q_local // 2
        return preprocess_index_dual_chunks(
            rows, rank, 2 * cp - rank - 1, bs, bs
        )
    return preprocess_index(rows, rank, q_local, q_local)


class TestFlashMaskCpEmulation(unittest.TestCase):
    """Per-rank (local Q, gathered KV, preprocessed global rows) == full attn."""

    _PACKS = [
        [4, 2],
        [0, 7, 1],
        [1],  # text-only dummy
        [13, 5, 9],
    ]

    def _check(self, cp, mode):
        heads = 2
        for ctx in self._PACKS:
            nq = [_Q] * len(ctx)
            seq = sum(ctx) + sum(nq)
            pad = prefix_lm_pad_len(
                seq, 1, False, align=math.lcm(_ALIGN, 2 * cp)
            )
            total = seq + pad
            rows = build_flashmask_row_indices(ctx, nq, pad)  # [1,1,T,2]
            lts, ute = rows[0, 0, :, 0], rows[0, 0, :, 1]
            g_forbid = _kernel_forbidden(lts, ute, total)
            dense = build_dense_mask(ctx, nq, pad).reshape([total, total])
            self.assertTrue(bool(paddle.equal_all(g_forbid, dense)))

            paddle.seed(7)
            q, k, v = (paddle.randn([total, heads, _D]) for _ in range(3))
            ref = _attn(q, k, v, g_forbid)
            q_local = total // cp
            pieces = []
            for r in range(cp):
                lq = _local_slice(q.unsqueeze(0), r, cp, mode)[0]
                li = _local_row_indices(rows, r, cp, mode, q_local)
                f = _kernel_forbidden(li[0, 0, :, 0], li[0, 0, :, 1], q_local)
                out = _attn(lq, k, v, f)
                want = _local_slice(ref.unsqueeze(0), r, cp, mode)[0]
                self.assertTrue(
                    bool(paddle.allclose(out, want, atol=1e-5, rtol=1e-5)),
                    f"ctx={ctx} cp={cp} mode={mode} rank={r}",
                )
                pieces.append(out.unsqueeze(0))
            merged = _gather_all(pieces, cp, mode)[0]
            self.assertTrue(bool(paddle.allclose(merged, ref, atol=1e-5)))

    def test_cp2_cp4(self):
        for cp in (2, 4):
            for mode in _MODES:
                with self.subTest(cp=cp, mode=mode):
                    self._check(cp, mode)


# ---- grad-scale marks ------------------------------------------------------ #
class TestGradScaleMarks(unittest.TestCase):
    _PREFIXES = [
        "encoder",
        "encoder.layers.0",
        "encoder.layers.1",
        "encoder.bridge",
        "model",
    ]

    def _model(self):
        layers = [paddle.nn.Linear(2, 2) for _ in self._PREFIXES]
        layers[2].weight.color = {"color": "moe_expert"}  # trunk expert param
        return types.SimpleNamespace(
            run_function=layers,
            get_sequential_name_prefixes=lambda: {
                str(i): p for i, p in enumerate(self._PREFIXES)
            },
        ), layers

    @staticmethod
    def _disabled(p):
        return bool(getattr(p, "context_parallel_disable_scale_grad", False))

    def test_shard_leaves_trunk_unmarked(self):
        model, layers = self._model()
        hb._mark_encoder_region(model, _enc_cfg(cp=4))
        for idx in (0, 3):  # frontend, bridge: replicated, full grads
            for p in layers[idx].parameters():
                self.assertTrue(self._disabled(p))
        for idx in (1, 2, 4):  # trunk + decoder: default CP handling
            for p in layers[idx].parameters():
                self.assertFalse(self._disabled(p))
                self.assertFalse(getattr(p, "expert_cp_replicated", False))

    def test_replicated_marks_whole_encoder(self):
        model, layers = self._model()
        hb._mark_encoder_region(model, _enc_cfg(cp=4, shard=False))
        self.assertTrue(layers[2].weight.expert_cp_replicated)
        self.assertTrue(self._disabled(layers[2].bias))
        for idx in (0, 1, 3):
            for p in layers[idx].parameters():
                self.assertTrue(self._disabled(p))
        for p in layers[4].parameters():
            self.assertFalse(self._disabled(p))


# ---- encoder aux under calculate_per_token_loss ---------------------------- #
def _per_token_enc_cfg():
    cfg = _enc_cfg(shard=False)
    cfg.hyperencoder_aux_by_lm_tokens = True
    return cfg


class TestEncoderAuxPerToken(unittest.TestCase):
    """The encoder router aux stays a native pack mean; per-token training must
    multiply it by the micro-batch's LM supervision token count (the trainer
    then divides every grad once by the step's global count ``T_global``)."""

    def test_aux_token_count_is_lm_label_count(self):
        labels = paddle.to_tensor(
            [[5, -100, 7, -100], [1, 2, -100, -100]], dtype="int64"
        )
        n_mb = _frontend(_per_token_enc_cfg())._lm_aux_token_count(
            {"labels": labels}
        )
        self.assertEqual(float(n_mb), 4.0)
        # This is exactly the decoder's per-shard ``pt_t_local`` summed over the
        # CP group, i.e. the micro-batch's contribution to ``T_global``.
        per_rank = sum(
            int(
                (
                    _local_slice(labels, r, 4, "contiguous_allgather") != -100
                ).sum()
            )
            for r in range(4)
        )
        self.assertEqual(float(n_mb), float(per_rank))

    def test_aux_token_count_drops_mtp_lookahead_tail(self):
        """MTP keeps K lookahead columns in ``labels``; the main head (and
        ``pt_t_local``/``T_global``) drops them, so ``n_mb`` must drop them too.
        """
        labels = paddle.to_tensor(
            [[1, 2, 3, 4], [5, -100, 7, 8]], dtype="int64"
        )  # [B, S + K] = [2, 3 + 1] (last col = K=1 lookahead)
        mask = paddle.zeros([1, 1, 3], dtype="int32")  # [B, K=1, S=3]
        n_mb = _frontend(_per_token_enc_cfg())._lm_aux_token_count(
            {"labels": labels, "mtp_hidden_inputs_mask_all": mask}
        )
        # main head = labels[:, :-1] = [[1, 2, 3], [5, -100, 7]] -> 3 + 2 = 5 non -100
        # (the full width would be 4 + 3 = 7, so this proves the K tail is dropped)
        self.assertEqual(float(n_mb), 5.0)

    def test_aux_token_count_asserts_on_mtp_layout_mismatch(self):
        # MTP mask must be [B, K, S] with S = labels width - K.
        labels = paddle.to_tensor([[1, 2, 3]], dtype="int64")  # width 3
        for bad_seq in (5, 3):  # 5+1 and 3+1 both != 3
            with self.subTest(mask_seq=bad_seq):
                bad_mask = paddle.zeros([1, 1, bad_seq], dtype="int32")
                with self.assertRaises(AssertionError):
                    _frontend(_per_token_enc_cfg())._lm_aux_token_count(
                        {
                            "labels": labels,
                            "mtp_hidden_inputs_mask_all": bad_mask,
                        }
                    )

    def test_aux_token_count_needs_marker_and_labels(self):
        labels = paddle.to_tensor([[1, -100]], dtype="int64")
        # Marker unset (per-token off): no scaling.
        self.assertIsNone(
            _frontend(_enc_cfg(shard=False))._lm_aux_token_count(
                {"labels": labels}
            )
        )
        # Marker set but no labels in this micro-batch.
        self.assertIsNone(
            _frontend(_per_token_enc_cfg())._lm_aux_token_count({})
        )

    def _mm_frontend(self):
        """Frontend whose context builder is a zero [1, _H] stub."""
        fe = _frontend(_per_token_enc_cfg())
        object.__setattr__(
            fe, "_embed_one_context", lambda *a, **k: paddle.zeros([1, _H])
        )
        return fe

    @staticmethod
    def _dec_args(**extra):
        args = {
            "input_ids": paddle.zeros([1, 8], dtype="int64"),
            "labels": paddle.to_tensor([[1, 2, -100, 3]], dtype="int64"),
            "attn_mask_startend_row_indices": None,
            "position_ids": None,
        }
        args.update(extra)
        return args

    def test_multimodal_forward_stashes_token_count(self):
        from paddlefleet.transformer.moe.moe_layer import (
            get_hyperbody_encoder_aux_n_mb,
        )

        out = self._mm_frontend().forward(
            self._dec_args(context_ids=paddle.zeros([1, 4], dtype="int64"))
        )
        self.assertNotIn("_hb_text_only", out)
        self.assertEqual(float(get_hyperbody_encoder_aux_n_mb()), 3.0)

    def test_text_only_forward_stashes_the_lm_token_count(self):
        """The dummy path reports the batch's LM supervision count.

        The dummy carries no LM tokens of its own, but reporting the batch's
        count keeps its aux on the same caliber as the non-per-token baseline
        (``pack_mean x n_mb / T_global``) instead of leaving it ``n_mb``x weak.
        The alternative -- stashing ``0.0`` to suppress the term outright -- is
        covered by ``test_zero_scale_suppresses_the_aux``.
        """
        from paddlefleet.transformer.moe.moe_layer import (
            get_hyperbody_encoder_aux_n_mb,
        )

        out = self._mm_frontend().forward(self._dec_args())
        self.assertTrue(out["_hb_text_only"])
        scale = get_hyperbody_encoder_aux_n_mb()
        self.assertIsNotNone(scale)
        # _dec_args() labels = [[1, 2, -100, 3]] -> 3 supervised tokens.
        self.assertEqual(float(scale), 3.0)

    def test_zero_scale_suppresses_the_aux(self):
        """A stash of ``0.0`` -> aux value 0 and router grad 0 (edge intact).

        This is the "text-only must not train the encoder" option: the
        ``AddAuxiliaryLoss`` edge stays in the graph, only the contribution is
        zeroed.
        """
        from paddlefleet.transformer.moe.moe_layer import (
            _hyperbody_encoder_aux_scale,
            set_hyperbody_encoder_aux_n_mb,
        )
        from paddlefleet.transformer.moe.moe_utils import AddAuxiliaryLoss

        cfg = types.SimpleNamespace(hyperencoder_aux_by_lm_tokens=True)
        set_hyperbody_encoder_aux_n_mb(0.0)
        try:
            w = paddle.to_tensor([1.0], stop_gradient=False)
            aux = _hyperbody_encoder_aux_scale((w * 2.0).sum(), cfg)
            self.assertEqual(float(aux), 0.0)
            out = AddAuxiliaryLoss.apply(paddle.zeros([2]) * w, aux)
            out.sum().backward()
            self.assertEqual(float(w.grad), 0.0)
        finally:
            set_hyperbody_encoder_aux_n_mb(None)

    def test_aux_grad_scales_with_token_count(self):
        """The stash scale reaches the router grads through ``AddAuxiliaryLoss``.

        The PyLayer back-propagates the aux scalar only (forward is the
        identity), so scaling the aux value scales exactly the aux gradient --
        the router's LM gradient branch is untouched.
        """
        from paddlefleet.transformer.moe.moe_layer import (
            _hyperbody_encoder_aux_scale,
            set_hyperbody_encoder_aux_n_mb,
        )
        from paddlefleet.transformer.moe.moe_utils import AddAuxiliaryLoss

        cfg = types.SimpleNamespace(hyperencoder_aux_by_lm_tokens=True)

        def aux_grad(n):
            set_hyperbody_encoder_aux_n_mb(n)
            w = paddle.to_tensor([1.0], stop_gradient=False)
            aux = _hyperbody_encoder_aux_scale((w * 2.0).sum(), cfg)
            # ``AddAuxiliaryLoss`` needs a grad-carrying activation (forward is
            # ``x.clone()``, backward returns ``grad_output`` for ``x``).
            out = AddAuxiliaryLoss.apply(paddle.zeros([2]) * w, aux)
            out.sum().backward()
            return float(w.grad)

        try:
            base = aux_grad(None)
            self.assertAlmostEqual(
                aux_grad(paddle.to_tensor(100.0, dtype="float32")),
                base * 100.0,
                places=4,
            )
            # A python scalar behaves identically.
            self.assertAlmostEqual(aux_grad(100.0), base * 100.0, places=4)
            self.assertAlmostEqual(
                aux_grad(paddle.to_tensor(0.0)), 0.0, places=6
            )
            # The text-only dummy path stashes the python scalar 0.0.
            self.assertEqual(aux_grad(0.0), 0.0)
        finally:
            set_hyperbody_encoder_aux_n_mb(None)

    def test_aux_scale_gated_on_encoder_marker(self):
        """The stash scales only the encoder; the decoder/off marker is a no-op."""
        from paddlefleet.transformer.moe.moe_layer import (
            _hyperbody_encoder_aux_scale,
            set_hyperbody_encoder_aux_n_mb,
        )

        try:
            set_hyperbody_encoder_aux_n_mb(paddle.to_tensor(8.0))
            aux = paddle.to_tensor(2.0)
            # Marker absent (decoder view) -> identity even with the stash set.
            dec = types.SimpleNamespace()
            self.assertEqual(float(_hyperbody_encoder_aux_scale(aux, dec)), 2.0)
            # Marker on the encoder -> scaled.
            enc = types.SimpleNamespace(hyperencoder_aux_by_lm_tokens=True)
            self.assertEqual(
                float(_hyperbody_encoder_aux_scale(aux, enc)), 16.0
            )
            # Stash cleared -> identity.
            set_hyperbody_encoder_aux_n_mb(None)
            self.assertEqual(float(_hyperbody_encoder_aux_scale(aux, enc)), 2.0)
        finally:
            set_hyperbody_encoder_aux_n_mb(None)


class TestAuxScalingAtAttachSite(unittest.TestCase):
    """``MoELayer`` multiplies the aux by the stash at both attach sites."""

    def _moe(self, marker):
        from paddlefleet.transformer.moe.moe_layer import MoELayer

        return _bare(
            MoELayer,
            config=types.SimpleNamespace(
                hyperencoder_aux_by_lm_tokens=marker,
            ),
            router_aux_loss_coef=1.0,
            training=True,
        )

    def _attach_grad(self, moe, n):
        from paddlefleet.transformer.moe.moe_layer import (
            set_hyperbody_encoder_aux_n_mb,
        )
        from paddlefleet.transformer.moe.moe_utils import AddAuxiliaryLoss

        set_hyperbody_encoder_aux_n_mb(n)
        w = paddle.to_tensor([1.0], stop_gradient=False)
        aux = (w * 2.0).sum()
        aux = moe.router_aux_loss_coef * aux
        from paddlefleet.transformer.moe.moe_layer import (
            _hyperbody_encoder_aux_scale,
        )

        aux = _hyperbody_encoder_aux_scale(aux, moe.config)
        out = AddAuxiliaryLoss.apply(paddle.zeros([2]) * w, aux)
        out.sum().backward()
        return float(w.grad)

    def test_encoder_marker_scales_attach(self):
        try:
            moe = self._moe(marker=True)
            base = self._attach_grad(moe, None)
            self.assertAlmostEqual(
                self._attach_grad(moe, paddle.to_tensor(50.0)),
                base * 50.0,
                places=4,
            )
        finally:
            from paddlefleet.transformer.moe.moe_layer import (
                set_hyperbody_encoder_aux_n_mb,
            )

            set_hyperbody_encoder_aux_n_mb(None)

    def test_decoder_marker_ignores_stash(self):
        from paddlefleet.transformer.moe.moe_layer import (
            set_hyperbody_encoder_aux_n_mb,
        )

        try:
            moe = self._moe(marker=False)
            set_hyperbody_encoder_aux_n_mb(paddle.to_tensor(50.0))
            # Even with the stash set, a non-encoder config must not scale.
            self.assertAlmostEqual(
                self._attach_grad(moe, paddle.to_tensor(50.0)),
                self._attach_grad(moe, None),
                places=6,
            )
        finally:
            set_hyperbody_encoder_aux_n_mb(None)


if __name__ == "__main__":
    unittest.main()
