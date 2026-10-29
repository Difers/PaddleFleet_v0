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

"""HyperBody: unified single-PipelineLayer model (encoder + decoder in one).

Unlike the earlier *composite* design (a ``FleetLayer`` holding two independently
``provide()``-d sub-models glued by an ``image_embeds`` scatter), this is ONE
``PipelineLayer`` whose flat layer-desc list is:

    encoder frontend (NEW)  -> encoder trunk (REUSED HyperEncoder specs)
    -> bridge (NEW)         -> decoder embedding (REUSED GPTEmbedding, scatter)
    -> decoder trunk (REUSED) -> layer_norm (REUSED) -> lm_head (REUSED) -> loss

The layer-to-layer contract is a **dict** (``TransformerLayer.forward`` updates
``hidden_states`` in place and preserves every other key). Two facts drive the
whole design:

* the encoder trunk uses the *same* ``TransformerLayer`` LayerSpec as the
  decoder, so it consumes ``hidden_states`` / ``attention_mask`` /
  ``rotary_pos_emb`` and would ALSO consume ``input_ids`` /
  ``attn_mask_startend_row_indices`` if present -> decoder-bound fields are
  carried through the encoder region under ``_hb_dec_*`` prefixes and renamed
  back by the bridge;
* ``GPTEmbedding.forward`` rebuilds a fresh preproc dict from ``input_ids``
  (ignoring any upstream ``hidden_states``) -> the bridge DROPS the encoder
  hidden state; the decoder embedding regenerates ``[B,S,H_dec]`` from
  ``input_ids`` and scatters the encoder latents (``image_embeds``) at
  ``input_ids == image_token_id`` (CONTEXT_TOKEN) positions.

Scope (this phase): PP=1 single-stage random-init smoke (build/forward/backward).
Cross-stage seg_method / stage-pinning at the encoder->decoder boundary is
deferred (the bridge emits variable-length latents and the hidden shape changes
across the boundary -> cross-stage P2P is not viable yet).
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import paddle
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import (
    LayerDesc,
    LayerSpec,
    PipelineLayer,
    SharedLayerDesc,
    build_spec_layer,
)
from paddle.distributed.flex_checkpoint.aoa.generation import resolve_names

from paddlefleet.context_parallel_utils import (
    ContextParallelGatherOp,
    ContextParallelScatterOp,
)
from paddlefleet.models.common.language_loss.language_loss import LanguageLoss
from paddlefleet.models.gpt.gpt_layer_specs import (
    _get_effective_mtp_layers,
    get_gpt_mtp_layers_spec,
    get_gpt_spec,
)
from paddlefleet.models.hyperbody import (
    MlpProjector,
    get_hyperbody_decoder_layer_specs,
    get_hyperbody_encoder_layer_specs,
)
from paddlefleet.tensor_parallel.layers import (
    gen_linear_aoa_statements,
    gen_linear_inv_aoa_statements,
)
from paddlefleet.transformer.hyper_connection import (
    HyperConnectionContractLayer,
    HyperConnectionModule,
)
from paddlefleet.transformer.layer import FleetLayer
from paddlefleet.transformer.moe.moe_layer import MoELayer
from paddlefleet.transformer.moe.moe_router import TopKRouter
from paddlefleet.transformer.multi_token_prediction import (
    MultiTokenPredictionLayer,
)

from ...nn.pp_model import GeneralModelForCausalLMPipe
from ..model_utils import PretrainedModel
from .configuration import CONTEXT_TOKEN, DECODER_VIEW_KEYS, HyperBodyConfig
from .providers import (
    HyperBodyDecoderModelProvider,
    HyperEncoderConfig,
    HyperEncoderProvider,
    _reject_unsupported_decoder_branches,
)

__all__ = [
    "HyperBodyModelDist",
    "HyperBodyModel",
    "HyperBodyForConditionalGeneration",
    "HyperBodyForCausalLMPipe",
    "HyperBodyModelPipe",
    "HyperBodyPretrainedModel",
]


_HB_DECODER_PASSTHROUGH_KEYS = (
    "mtp_startend_row_indices_all",
    "mtp_hidden_inputs_mask_all",
)

_HB_PASSTHROUGH_LOGGED = False


def _export_decoder_passthrough(out, dict_args):
    """Frontend: stash the passthrough fields under ``_hb_dec_*``.

    Always writes the key (``None`` when absent) so the dict's key set does not
    vary across micro-steps.
    """
    for key in _HB_DECODER_PASSTHROUGH_KEYS:
        out[f"_hb_dec_{key}"] = dict_args.get(key, None)
    return out


def _restore_decoder_passthrough(decoder_args, dict_args):
    """Bridge: rename the ``_hb_dec_*`` fields back onto the decoder dict.

    Only non-``None`` values are written: an absent key and a ``None`` value are
    equivalent for every consumer (``dict_args.get``), and leaving ``None`` out
    keeps the decoder dict free of None-valued entries.
    """
    global _HB_PASSTHROUGH_LOGGED
    for key in _HB_DECODER_PASSTHROUGH_KEYS:
        value = dict_args.get(f"_hb_dec_{key}", None)
        if value is not None:
            decoder_args[key] = value
    if not _HB_PASSTHROUGH_LOGGED:
        _HB_PASSTHROUGH_LOGGED = True
        import logging as _logging

        _logging.getLogger(__name__).info(
            "[HyperBody bridge] decoder passthrough: %s",
            {
                k: (
                    "absent"
                    if decoder_args.get(k, None) is None
                    else tuple(decoder_args[k].shape)
                )
                for k in _HB_DECODER_PASSTHROUGH_KEYS
            },
        )
    return decoder_args


# ======================================================================= #
# NEW wrapper layer 1: encoder frontend (pipeline-hostile logic isolated)  #
# ======================================================================= #
class HyperBodyEncoderFrontEnd(FleetLayer):
    """Encoder front end + query concat + prefix-LM mask + rope.

    Holds the pipeline-hostile pieces of ``HyperEncoderModel`` that cannot be a
    plain single-hidden-state layer: the per-sample multimodal context builder
    (with dummy gradient keepalive), the query-table concat, the prefix-LM dense
    mask construction and the rope table. Emits a dict whose ``hidden_states`` is
    the padded ``[B, C+Q(+pad), H_enc]`` sequence, plus the encoder
    ``attention_mask`` / ``rotary_pos_emb`` that the (shared) TransformerLayer
    trunk consumes, plus the geometry scalars and the decoder-bound fields under
    ``_hb_dec_*`` prefixes so the encoder trunk does not consume them.
    """

    def __init__(self, config, **kwargs) -> None:
        super().__init__(config=config)
        from paddlefleet.models.common.embeddings.rotary_pos_embedding import (
            RotaryEmbedding,
        )
        from paddlefleet.models.hyperbody import (
            AudioEncoderConv,
            ImageEncoderConv,
        )
        from paddlefleet.models.hyperencoder.attn_backend import (
            use_packed_decoder,
        )

        self.config = config
        # dp (dense mask) vs flashmask+packed (single call, [1,1,T,2] row
        # indices). use_packed_decoder rejects the mixed combinations.
        self.packed = use_packed_decoder(config)

        self.embed_tokens = paddle.nn.Embedding(
            config.vocab_size, config.hidden_size
        )
        self.image_encoder = ImageEncoderConv(
            img_size=728,
            patch_size=14,
            in_chans=3,
            embed_dim=768,
            out_chans=256,
        )
        self.audio_encoder = AudioEncoderConv(num_mel_bins=128, embed_dim=768)
        # HyperBodyMlpProjector, not the base MlpProjector: the archive stores
        # this leaf transposed (see the subclass).
        self.projector = HyperBodyMlpProjector(
            {
                "projector_type": "linear",
                "input_dim": 768,
                "n_embed": config.hidden_size,
            }
        )
        # One learnable query table shared by every context segment (each
        # segment carries exactly ``hyperencoder_query_length`` placeholders).
        self.query = paddle.nn.Embedding(
            config.hyperencoder_query_length, config.hidden_size
        )
        self.rotary_pos_emb = RotaryEmbedding(
            head_dim=config.head_dim,
            rotary_percent=config.rotary_percent,
            rotary_base=config.rope_theta,
        )

        for m in (
            self.embed_tokens,
            self.image_encoder,
            self.audio_encoder,
            self.projector,
            self.query,
        ):
            m.to(dtype=config.params_dtype)
            for p in m.parameters():
                p.average_gradients_across_tp_domain = True

    # -- per-segment multimodal context builder (dummy keepalive) --
    def _embed_one_context(self, ids_i, img_i, aud_i):
        """One context segment ``[C_i]`` -> ``[C_i', H]`` with image/audio splice.

        Mirrors the reference ``build_context_embeds`` per-sample body: replace
        the IM_PATCH / AUDIO_PATCH placeholder run with the encoder-tower feature
        rows by "truncate and append"; when a modality is absent, push a dummy
        through the tower and add ``feat.sum()*0`` (a forward identity that keeps
        the tower's gradients alive).
        """
        from .configuration import AUDIO_PATCH_TOKEN, IM_PATCH_TOKEN

        emb_i = self.embed_tokens(ids_i)  # [C_i, H]

        pos = paddle.nonzero(ids_i == IM_PATCH_TOKEN)
        if pos.numel() > 0 and isinstance(img_i, paddle.Tensor):
            feat = self.projector(self.image_encoder(img_i.unsqueeze(0)))
            feat = (
                feat.transpose([0, 3, 1, 2])
                .flatten(2)
                .transpose([0, 2, 1])
                .squeeze(0)
            )
            emb_i = paddle.concat([emb_i[: int(pos[0])], feat], axis=0)
        else:
            dummy = paddle.zeros([1, 3, 128, 128], dtype=emb_i.dtype)
            feat = self.projector(self.image_encoder(dummy))
            emb_i = emb_i + feat.sum() * 0.0

        pos = paddle.nonzero(ids_i == AUDIO_PATCH_TOKEN)
        if pos.numel() > 0 and isinstance(aud_i, paddle.Tensor):
            feat = self.projector(self.audio_encoder(aud_i.unsqueeze(0)))
            feat = (
                feat.transpose([0, 3, 1, 2])
                .flatten(2)
                .transpose([0, 2, 1])
                .squeeze(0)
            )
            emb_i = paddle.concat([emb_i[: int(pos[0])], feat], axis=0)
        else:
            dummy = paddle.zeros([1, 128, 10], dtype=emb_i.dtype)
            feat = self.projector(self.audio_encoder(dummy))
            emb_i = emb_i + feat.sum() * 0.0

        return emb_i

    def _build_context_embeds(
        self, context_ids, image=None, audio=None, cu_seqlens_context=None
    ):
        """context_ids -> context embeds.

        Two shapes are supported:

        * **rectangular** (synthetic / equal-length): ``context_ids`` is
          ``[B, C]`` and ``cu_seqlens_context`` is None; returns a stacked
          ``[B, C, H]`` (every row the same length).
        * **packed / ragged** (real torch dump): ``context_ids`` is
          ``[1, C_total]`` and ``cu_seqlens_context`` is a boundary list
          ``[0, e0, e1, ...]`` splitting it into per-segment contexts of possibly
          different lengths; returns a **list** of ``[C_i', H]``. ``image`` /
          ``audio`` are per-segment lists (with ``None`` where absent).
        """
        if cu_seqlens_context is None:
            bs = context_ids.shape[0]
            imgs = image if image else [None] * bs
            auds = audio if audio else [None] * bs
            out = [
                self._embed_one_context(context_ids[i], imgs[i], auds[i])
                for i in range(bs)
            ]
            return paddle.stack(out, axis=0)

        # Packed / ragged: split [1, C_total] by cu_seqlens_context.
        bounds = [int(x) for x in cu_seqlens_context]
        n_seg = len(bounds) - 1
        imgs = image if image else [None] * n_seg
        auds = audio if audio else [None] * n_seg
        row = context_ids[0]
        return [
            self._embed_one_context(
                row[bounds[s] : bounds[s + 1]], imgs[s], auds[s]
            )
            for s in range(n_seg)
        ]

    def _packed_rope(self, max_seqlen: int, seg_lens):
        """Assemble RoPE freqs for the packed sequence: ``[1, ΣL_i, 1, D]``.

        Each segment restarts its position from 0 (prefix-LM in-segment relative
        position), so this concatenates ``freqs[:, :L_i]`` per segment. At a
        single segment it reduces to ``freqs[:, :L_0]`` (identical to the dense
        path). Mirrors ``HyperEncoderModel._packed_rope`` (non-SP branch).
        """
        freqs = self.rotary_pos_emb(max_seq_len=max_seqlen)
        if len(seg_lens) == 1:
            return freqs[:, : seg_lens[0]]
        return paddle.concat([freqs[:, :L] for L in seg_lens], axis=1)

    def _check_context_token_counts(self, input_ids, cu_seqlens):
        """Each decoder segment must hold a whole number of query tables.

        A decoder segment (``cu_seqlens``) may carry several encoder segments
        (tiled images, multiple images, audio chunks), and their context spans can
        sit back to back in ``input_ids``, so individual encoder segments cannot be
        told apart here. What can be checked is that every decoder segment's
        context-token count is a multiple of ``hyperencoder_query_length``;
        ``GPTEmbedding`` only checks the total, which lets e.g. ``q + 1`` and
        ``q - 1`` in two segments through and misplaces the encoder output.
        """
        q = int(self.config.hyperencoder_query_length)
        ctx_token = getattr(
            self.config, "hyperbody_context_token_id", CONTEXT_TOKEN
        )
        bounds = [int(x) for x in cu_seqlens.reshape([-1]).tolist()]
        tokens = input_ids.reshape([-1])
        if len(bounds) < 2 or int(tokens.shape[0]) < bounds[-1]:
            # Not the global decoder layout cu_seqlens describes (e.g. a
            # sequence-sharded view): nothing reliable to count against.
            return
        # MTP appends lookahead tokens after bounds[-1]; they are not counted.
        is_ctx = (tokens[: bounds[-1]] == ctx_token).astype("int64")
        csum = paddle.concat(
            [paddle.zeros([1], dtype="int64"), paddle.cumsum(is_ctx)]
        ).tolist()
        counts = [csum[e] - csum[s] for s, e in zip(bounds[:-1], bounds[1:])]
        bad = [(i, c) for i, c in enumerate(counts) if c % q != 0]
        if bad:
            raise ValueError(
                "every HyperBody decoder segment must hold a multiple of "
                f"hyperencoder_query_length={q} context tokens "
                f"(id={ctx_token}); offending (segment, count) = {bad}"
            )

    def forward(self, dict_args):
        from paddlefleet.transformer.prefix_lm_mask import (
            build_dense_mask,
            prefix_lm_pad_len,
        )
        from .mm_pack import unpack_hyperbody_mm

        # Text-only short path: when a sample carries no image/audio/context, the
        # encoder has nothing to encode. Skip the multimodal frontend and let the
        # decoder run on pure text (the bridge emits image_embeds=None so the
        # shared GPTEmbedding._merge_multimodal is a no-op). A tiny dummy tensor
        # keeps the (fixed) encoder-trunk layers in the pipeline happy; their
        # output is discarded by the bridge under the _hb_text_only flag.
        def _all_none(x):
            return x is None or (
                isinstance(x, (list, tuple)) and all(e is None for e in x)
            )

        if (
            dict_args.get("context_ids", None) is None
            and _all_none(dict_args.get("image", None))
            and _all_none(dict_args.get("audio", None))
        ):
            return self._forward_text_only(
                dict_args, build_dense_mask, prefix_lm_pad_len
            )

        # The fleet pipeline micro-batch loader only passes Tensors; image/audio
        # are packed as ragged Tensors upstream and restored in place to lists here.
        unpack_hyperbody_mm(dict_args)

        context_ids = dict_args["context_ids"]
        image = dict_args.get("image", None)
        audio = dict_args.get("audio", None)
        cu_seqlens = dict_args.get("cu_seqlens", None)
        if cu_seqlens is not None:
            self._check_context_token_counts(dict_args["input_ids"], cu_seqlens)
        cu_seqlens_context = dict_args.get("cu_seqlens_context", None)

        context_embeds = self._build_context_embeds(
            context_ids, image, audio, cu_seqlens_context
        )

        if self.packed:
            out = self._forward_packed(context_embeds)
        else:
            out = self._forward_dense(
                context_embeds,
                build_dense_mask,
                prefix_lm_pad_len,
            )

        # Decoder-bound fields ride through the encoder trunk under _hb_dec_*
        # prefixes (absorbed by TransformerLayer **kwargs, preserved by
        # rst={**dict_args,**rst}); the bridge renames them back.
        out["_hb_dec_input_ids"] = dict_args["input_ids"]
        out["_hb_dec_labels"] = dict_args.get("labels", None)
        out["_hb_dec_attn_mask_startend_row_indices"] = dict_args.get(
            "attn_mask_startend_row_indices", None
        )
        out["_hb_dec_position_ids"] = dict_args.get("position_ids", None)
        return _export_decoder_passthrough(out, dict_args)

    def _forward_text_only(
        self, dict_args, build_dense_mask, prefix_lm_pad_len
    ):
        """No-multimodal path: run the encoder trunk on a 1-token dummy so the
        fixed encoder layers stay well-fed, then carry the decoder-bound fields
        through. The bridge drops the dummy (image_embeds=None) under the
        ``_hb_text_only`` flag, so the decoder runs on pure text.
        """
        dtype = self.query.weight.dtype
        # Build the 1-token dummy through the real context builder (embed_tokens +
        # zero-weighted image/audio tower + projector keepalive) and zero it, so a
        # text-only rank touches the same parameters as a multimodal rank. Together
        # with the bridge's ``multimodal_keepalive`` this keeps the encoder in the
        # backward graph: every EP rank then issues the same DeepEP dispatch/combine
        # sequence even when only some ranks got a text-only micro-batch (e.g. an
        # image read failure degraded one sample). Values stay exactly zero.
        ctx = self._embed_one_context(
            paddle.zeros([1], dtype="int64"), None, None
        )
        dummy_ctx = (ctx * 0.0).astype(dtype).unsqueeze(0)  # [1, 1, H]
        if self.packed:
            out = self._forward_packed(dummy_ctx)
        else:
            out = self._forward_dense(
                dummy_ctx, build_dense_mask, prefix_lm_pad_len
            )
        out["_hb_text_only"] = True
        out["_hb_dec_input_ids"] = dict_args["input_ids"]
        out["_hb_dec_labels"] = dict_args.get("labels", None)
        out["_hb_dec_attn_mask_startend_row_indices"] = dict_args.get(
            "attn_mask_startend_row_indices", None
        )
        out["_hb_dec_position_ids"] = dict_args.get("position_ids", None)
        return _export_decoder_passthrough(out, dict_args)

    def _forward_dense(
        self,
        context_embeds,
        build_dense_mask,
        prefix_lm_pad_len,
    ):
        # Non-SP [B,S,H] dense-mask path (dp backend): every segment of the
        # rectangular [B, n_ctx, H] block shares the single query table.
        bs, n_context, _ = context_embeds.shape
        queries = self.query.weight.unsqueeze(0).expand([bs, -1, -1])
        n_queries = queries.shape[1]

        x = paddle.concat([context_embeds, queries], axis=1)
        seq_len = x.shape[1]
        pad = prefix_lm_pad_len(
            seq_len,
            self.config.tensor_model_parallel_size,
            self.config.sequence_parallel,
            align=self.config.hyperencoder_seq_align,
        )
        if pad > 0:
            x = paddle.concat(
                [x, paddle.zeros([bs, pad, x.shape[-1]], dtype=x.dtype)], axis=1
            )
        mask = build_dense_mask([n_context], [n_queries], pad)
        rope = self.rotary_pos_emb(max_seq_len=x.shape[1])

        return {
            "hidden_states": x,
            "attention_mask": mask,
            "rotary_pos_emb": rope,
            "packed_seq_params": None,
            "_hb_packed": None,
            "n_context": n_context,
            "n_queries": n_queries,
            "seq_len": seq_len,
        }

    def _forward_packed(self, context_embeds):
        """Single packed call through the flashmask branch of
        ``DotProductAttention``.

        The ``(context, query)`` segments are concatenated end-to-end into one
        ``[1, T, H]`` sequence. The prefix-LM mask goes to the trunk as FlashMask
        row indices ``[1, 1, T, 2]`` under ``attn_mask_startend_row_indices``
        (consumed with ``causal=False``); the segment layout is also carried
        under ``_hb_packed`` for the bridge's per-segment slice.
        ``attention_mask`` / ``packed_seq_params`` are None: the mask semantics
        live entirely in the row indices.

        ``context_embeds`` may be a rectangular ``[B, C, H]`` (synthetic /
        equal-length) or a ragged **list** of ``[C_i, H]`` (real packed dump).
        Every segment is followed by the same single query table.
        """
        from paddlefleet.transformer.prefix_lm_mask import (
            build_flashmask_row_indices,
            prefix_lm_pad_len,
        )

        # ---- Normalize context to a per-segment list [C_i, H] ----
        if isinstance(context_embeds, (list, tuple)):
            seg_ctx = [
                c.squeeze(0) if c.ndim == 3 else c for c in context_embeds
            ]
            n_contexts = [int(c.shape[0]) for c in seg_ctx]
            hidden = int(seg_ctx[0].shape[1])
        else:
            bs0, n_context, hidden = context_embeds.shape
            seg_ctx = [context_embeds[i] for i in range(bs0)]
            n_contexts = [n_context] * bs0
        bs = len(seg_ctx)

        qw = self.query.weight
        nq = int(qw.shape[0])
        nq_list = [nq] * bs

        # ---- Concatenate (context, query) segments end-to-end ----
        segments, starts, running = [], [], 0
        for ci, nc in zip(seg_ctx, n_contexts):
            segments.append(paddle.concat([ci, qw], axis=0))
            starts.append(running)
            running += nc + nq
        x = paddle.concat(segments, axis=0).unsqueeze(0)  # [1, ΣL, H]
        seq_len = x.shape[1]

        pad = prefix_lm_pad_len(
            seq_len,
            self.config.tensor_model_parallel_size,
            self.config.sequence_parallel,
            align=self.config.hyperencoder_seq_align,
        )
        if pad > 0:
            x = paddle.concat(
                [x, paddle.zeros([1, pad, hidden], dtype=x.dtype)], axis=1
            )

        # The pad is folded into the last segment (not a segment of its own) so
        # the rope segmentation and the layout segmentation stay aligned.
        seg_lens = [nc + nq for nc, nq in zip(n_contexts, nq_list)]
        seg_lens[-1] += pad
        max_seqlen = max(seg_lens)

        layout = {
            "segment_starts": tuple(starts),
            "n_contexts": tuple(n_contexts),
            "n_queries": tuple(nq_list),
        }
        row_indices = build_flashmask_row_indices(
            n_contexts, nq_list, pad, place=x.place
        )

        rope = self._packed_rope(max_seqlen, seg_lens)
        if getattr(self.config, "hyperencoder_cp_shard", False):
            # Every CP rank packed the same global sequence, so the cut is a
            # scatter whatever the decoder's data source: forward slices, backward
            # all-gathers, i.e. the replicated frontend sees the full grad. Row
            # indices stay global (flashmask_attention_cp slices them per chunk).
            mode = self.config.cp_balance_mode
            x = ContextParallelScatterOp.apply(x, axis=1, mode=mode)
            rope = ContextParallelScatterOp.apply(rope, axis=1, mode=mode)

        return {
            "hidden_states": x,
            "attention_mask": None,
            "attn_mask_startend_row_indices": row_indices,
            "rotary_pos_emb": rope,
            "packed_seq_params": None,
            "_hb_packed": layout,
            # Ragged pack: per-segment lists (bridge reads _hb_packed layout, not
            # these scalars; kept for symmetry with the dense return dict).
            "n_context": list(n_contexts),
            "n_queries": list(nq_list),
            "seq_len": seq_len,
        }


class _ZeroGradKeepalive(paddle.autograd.PyLayer):
    """Zeros forward, zeros backward.

    Keeps the encoder in a text-only rank's backward graph so every EP rank
    issues the same DeepEP dispatch/combine calls (no deadlock), while carrying
    an exact zero so no gradient changes. Unlike ``x * 0.0``, the output does not
    depend on ``x``, so a non-finite encoder output cannot leak into the decoder.
    """

    @staticmethod
    def forward(ctx, x):
        return paddle.zeros_like(x)

    @staticmethod
    def backward(ctx, dy):
        return paddle.zeros_like(dy)


# ======================================================================= #
# NEW wrapper layer 2: bridge (encoder trunk output -> decoder latents)    #
# ======================================================================= #
class HyperBodyEncoderBridge(FleetLayer):
    """Encoder trunk output -> per-sample out-projected LLM latents.

    Reapplies the encoder block's final ``HyperEncoderRMSNorm`` (lost when the
    ``HyperEncoderBlock`` is decomposed into per-layer specs), drops the pad,
    slices the trailing ``Q`` rows, runs the per-batch ``out_projector``
    (RowParallelLinear -> ``language_hidden_size``, adds bias by hand), and emits
    the *clean decoder* dict. It DROPS the encoder ``hidden_states`` so the
    downstream ``GPTEmbedding`` regenerates ``[B,S,H_dec]`` from ``input_ids``
    and scatters the encoder latents at CONTEXT_TOKEN positions.
    """

    def __init__(self, config, **kwargs) -> None:
        super().__init__(config=config)
        from paddlefleet.models.hyperencoder.norm import HyperEncoderRMSNorm
        from paddlefleet.tensor_parallel.layers import RowParallelLinear

        self.config = config
        self.final_norm = HyperEncoderRMSNorm(config)

        projector_config = copy.copy(config)
        projector_config.sequence_parallel = False
        self.out_projector = RowParallelLinear(
            config.hidden_size,
            config.language_hidden_size,
            config=projector_config,
            init_method=config.init_method,
            bias=True,
            input_is_parallel=False,
            skip_bias_add=False,
        )
        if getattr(self.out_projector, "bias", None) is not None:
            self.out_projector.bias.average_gradients_across_tp_domain = True

    def _project(self, latents):
        """Per-segment ``out_projector`` (RowParallelLinear, adds bias by hand)."""
        projected, bias = self.out_projector(latents)
        if bias is not None:
            projected = projected + bias
        return projected

    def forward(self, dict_args):
        # Text-only short path: encoder produced a discardable dummy; emit
        # image_embeds=None so the decoder embedding's _merge_multimodal skips the
        # scatter. The dummy is still projected and handed over as a zero-valued
        # ``multimodal_keepalive`` scalar that the decoder embedding adds to its
        # input, so the encoder trunk (and its EP collectives) runs backward on
        # every rank, keeping DeepEP call counts symmetric across the EP group.
        # The scalar goes through ``_ZeroGradKeepalive``: the backward edge reaches
        # the encoder (DeepEP call counts stay aligned) but carries an exact 0.
        if dict_args.get("_hb_text_only", False):
            layout = dict_args.get("_hb_packed", None)
            if layout is not None:
                dummy = self._forward_packed(dict_args, layout)
            else:
                dummy = self._forward_dense(dict_args)
            keepalive = _ZeroGradKeepalive.apply(dummy).sum()
            return _restore_decoder_passthrough(
                {
                    "multimodal_keepalive": keepalive,
                    "input_ids": dict_args["_hb_dec_input_ids"],
                    "position_ids": dict_args.get("_hb_dec_position_ids", None),
                    "attention_mask": None,
                    "attn_mask_startend_row_indices": dict_args[
                        "_hb_dec_attn_mask_startend_row_indices"
                    ],
                    "decoder_input": None,
                    "image_embeds": None,
                    "video_embeds": None,
                    "labels": dict_args["_hb_dec_labels"],
                },
                dict_args,
            )
        layout = dict_args.get("_hb_packed", None)
        if layout is not None:
            image_embeds = self._forward_packed(dict_args, layout)
        else:
            image_embeds = self._forward_dense(dict_args)

        return _restore_decoder_passthrough(
            {
                "input_ids": dict_args["_hb_dec_input_ids"],
                "position_ids": dict_args.get("_hb_dec_position_ids", None),
                "attention_mask": None,
                "attn_mask_startend_row_indices": dict_args[
                    "_hb_dec_attn_mask_startend_row_indices"
                ],
                "decoder_input": None,
                "image_embeds": image_embeds,
                "video_embeds": None,
                "labels": dict_args["_hb_dec_labels"],
            },
            dict_args,
        )

    def _forward_dense(self, dict_args):
        h = dict_args["hidden_states"]
        n_context = dict_args["n_context"]
        seq_len = dict_args["seq_len"]

        h = self.final_norm(h)
        h = h[:, :seq_len]  # drop the prefix-LM pad
        latents = h[:, n_context:, :]  # trailing Q rows: [B, Q, H_enc]

        outs = [self._project(latents[i]) for i in range(latents.shape[0])]
        return paddle.concat(outs, axis=0)  # [B*Q, H_dec]

    def _forward_packed(self, dict_args, layout):
        # hidden_states is [1, ΣL_i(+pad), H_enc]; slice each segment's trailing
        # query rows via the packed layout, out-project, concat -> [ΣQ_i, H_dec].
        h = dict_args["hidden_states"]
        if getattr(
            self.config, "hyperencoder_cp_shard", False
        ):  # inverse of the frontend scatter
            h = ContextParallelGatherOp.apply(
                h, axis=1, mode=self.config.cp_balance_mode
            )
        h = self.final_norm(h)[0]  # [T, H_enc]
        starts = layout["segment_starts"]
        n_contexts = layout["n_contexts"]
        n_queries = layout["n_queries"]

        outs = []
        for s, nc, nq in zip(starts, n_contexts, n_queries):
            latents = h[s + nc : s + nc + nq, :]  # [Q_i, H_enc]
            outs.append(self._project(latents))
        return paddle.concat(outs, axis=0)  # [ΣQ_i, H_dec]

    # AOA: a stage-one warm start trains ``out_projector`` from its own init,
    # so its archive has no projector to read -- ``projector_cold_start`` says
    # so. Export always states it; ``final_norm`` goes via the base recursion.

    def gen_aoa_statements(
        self,
        ctx,
        *,
        structured_name_prefix="",
        checkpoint_lookup_drop_segment=None,
    ):
        statements = self.final_norm.gen_aoa_statements(
            ctx,
            structured_name_prefix=f"{structured_name_prefix}final_norm.",
            checkpoint_lookup_drop_segment=checkpoint_lookup_drop_segment,
        )
        projector_statements = gen_linear_aoa_statements(
            self.out_projector,
            ctx,
            structured_name_prefix=f"{structured_name_prefix}out_projector.",
            checkpoint_lookup_drop_segment=checkpoint_lookup_drop_segment,
        )
        # ``getattr`` default covers a view not forwarding this global knob.
        if not getattr(ctx.config, "projector_cold_start", False):
            return statements + projector_statements
        # Naming no source leaves every projector tensor at its init.
        return statements + [
            f"_ -> {statement.rsplit(' -> ', 1)[1]}"
            for statement in projector_statements
        ]

    def gen_inv_aoa_statements(
        self,
        ctx,
        *,
        structured_name_prefix="",
        checkpoint_lookup_drop_segment=None,
    ):
        return self.final_norm.gen_inv_aoa_statements(
            ctx,
            structured_name_prefix=f"{structured_name_prefix}final_norm.",
            checkpoint_lookup_drop_segment=checkpoint_lookup_drop_segment,
        ) + gen_linear_inv_aoa_statements(
            self.out_projector,
            ctx,
            structured_name_prefix=f"{structured_name_prefix}out_projector.",
            checkpoint_lookup_drop_segment=checkpoint_lookup_drop_segment,
        )


# ======================================================================= #
# Unified PipelineLayer (mirrors GPTModel's get_layer_desc_list mechanics)  #
# ======================================================================= #
@dataclass
class HyperBodySublayersSpec:
    """Flat spec list for the unified model (encoder + bridge + decoder)."""

    encoder_frontend: LayerSpec = None
    encoder_layers: list[LayerSpec] = field(default_factory=list)
    bridge: LayerSpec = None
    decoder_embedding: LayerSpec = None
    mhc_expand: LayerSpec = None
    decoder_layers: list[LayerSpec] = field(default_factory=list)
    mhc_contract: LayerSpec = None
    # MTP layer spec(s), structure only (weight sharing + forward wired later).
    mtp: list[LayerSpec] = None
    layer_norm: LayerSpec = None
    lm_head: LayerSpec = None


class HyperBodyUnifiedModel(PretrainedModel, PipelineLayer):
    """One PipelineLayer: encoder frontend/trunk/bridge + decoder emb/trunk/head.

    Mirrors ``GPTModel``'s pipeline mechanics (``get_layer_desc_list`` +
    ``add_sequential_layer`` + ``get_sequential_layers``) but with the HyperBody
    layer order. Encoder-region layers use the ``encoder`` name prefix, decoder
    region uses ``model`` (matching ``GPTModel``), so weight keys do not clash.
    """

    config_class = HyperBodyConfig

    def __init__(self, sublayers_spec, **kwargs) -> None:
        self.config = kwargs["config"]
        pp = self.config.pipeline_model_parallel_size
        # single(logical) <-> pipeline(numeric) key maps, built lazily by
        # ``_set_pipeline_name_mapping`` (mirrors GPTModel / PipelinePretrainedModel).
        self._single_to_pp_mapping = None
        self._pp_to_single_mapping = None

        self._sequential_layers = self.get_layer_desc_list(sublayers_spec)
        self.layers = self.get_sequential_layers()

        del kwargs["config"]
        if "tie_word_embeddings" in kwargs:
            del kwargs["tie_word_embeddings"]

        topology = (
            None if pp == 1 else fleet.get_hybrid_communicate_group().topology()
        )

        PipelineLayer.__init__(
            self,
            layers=self.layers,
            topology=topology,
            num_virtual_pipeline_stages=(
                self.config.virtual_pipeline_model_parallel_size
            ),
            **kwargs,
        )

    @staticmethod
    def add_sequential_layer(layers, layer_desc, name_prefix=""):
        layers.append({"layer": layer_desc, "name_prefix": name_prefix})

    def get_sequential_layers(self):
        return [x["layer"] for x in self._sequential_layers]

    def get_sequential_name_prefixes(self):
        return {
            str(i): x["name_prefix"]
            for i, x in enumerate(self._sequential_layers)
        }

    def _iter_transformer_layers(self):
        from paddlefleet.transformer.multi_token_prediction import (
            MultiTokenPredictionLayer,
        )
        from paddlefleet.transformer.transformer_layer import TransformerLayer

        if getattr(self, "_num_virtual_pipeline_stages", 1) > 1:
            layers = (layer for chunk in self._model_chunks for layer in chunk)
        else:
            layers = self.run_function
        for layer in layers:
            if isinstance(layer, TransformerLayer):
                yield layer
            elif isinstance(layer, MultiTokenPredictionLayer):
                # Mirror GPTModel: MTP's fp8 state lives on its inner body.
                yield layer.transformer_layer

    def fp8_quant_weight(self, batch_mode=False, quant_transpose=True):
        for layer in self._iter_transformer_layers():
            layer.fp8_quant_weight(
                batch_mode=batch_mode, quant_transpose=quant_transpose
            )

    def clear_fp8_quant_weight(self):
        for layer in self._iter_transformer_layers():
            layer.clear_fp8_quant_weight()

    def use_fp8(self):
        for layer in self._iter_transformer_layers():
            if layer.use_fp8():
                return True
        return False

    def get_layer_desc_list(self, spec: HyperBodySublayersSpec):
        layers = []
        # --- encoder region (name_prefix="encoder") ---
        self.add_sequential_layer(
            layers, LayerDesc(spec.encoder_frontend), "encoder"
        )
        for i, enc_layer in enumerate(spec.encoder_layers):
            self.add_sequential_layer(
                layers, LayerDesc(enc_layer), f"encoder.layers.{i}"
            )
        self.add_sequential_layer(
            layers, LayerDesc(spec.bridge), "encoder.bridge"
        )
        # --- decoder region (name_prefix="model") ---
        self.add_sequential_layer(
            layers, LayerDesc(spec.decoder_embedding), "model"
        )
        # hyper-connections expand (after embedding, before trunk); only when
        # the decoder was built with enable_hyper_connections=True.
        if spec.mhc_expand is not None:
            self.add_sequential_layer(
                layers, LayerDesc(spec.mhc_expand), "model.mhc_expand"
            )
        n_dec = len(spec.decoder_layers)
        for i, dec_layer in enumerate(spec.decoder_layers):
            # With mtp_shared_last_layer on, the LAST backbone layer is the pivot
            # of the "mtp_reuse_transformer" shared key; each MTP layer's
            # transformer_layer aliases its weights (see the MTP loop below).
            # Mirrors the standard GPTModel. PP=1 => pivot and aliases co-locate.
            if (
                i == n_dec - 1
                and spec.mtp
                and getattr(self.config, "mtp_shared_last_layer", False)
            ):
                desc = SharedLayerDesc(
                    "mtp_reuse_transformer",
                    dec_layer,
                    shared_weight_attr="transformer_layer_weights",
                )
            else:
                desc = LayerDesc(dec_layer)
            self.add_sequential_layer(layers, desc, f"model.layers.{i}")
        # hyper-connections contract (after trunk, before final norm).
        if spec.mhc_contract is not None:
            self.add_sequential_layer(
                layers, LayerDesc(spec.mhc_contract), "model.mhc_contract"
            )
        self.add_sequential_layer(layers, LayerDesc(spec.layer_norm), "model")
        # MTP layers (structure wiring; MTP forward/loss still deferred). Mirror
        # the standard GPTModel: placed after the final norm, indexed continuing
        # the main decoder layers (first MTP layer is model.layers.{n_dec}, so it
        # lines up with the extra csa_compress_ratios entries). Weight sharing is
        # honored here: mtp_shared_last_layer aliases each MTP body against the
        # backbone pivot ("mtp_reuse_transformer"); mtp_shared_weights aliases the
        # all-depths key ("mtp_shared_all"); otherwise each MTP layer owns weights.
        if spec.mtp:
            for j, mtp_layer in enumerate(spec.mtp):
                if getattr(self.config, "mtp_shared_last_layer", False):
                    desc = SharedLayerDesc(
                        "mtp_reuse_transformer",
                        mtp_layer,
                        shared_submodule_weight_only=True,
                        shared_weight_attr="transformer_layer_weights",
                    )
                elif getattr(self.config, "mtp_shared_weights", False):
                    desc = SharedLayerDesc(
                        "mtp_shared_all",
                        mtp_layer,
                        shared_submodule_weight_only=True,
                        shared_weight_attr="all_weights",
                    )
                else:
                    desc = LayerDesc(mtp_layer)
                self.add_sequential_layer(
                    layers, desc, f"model.layers.{n_dec + j}"
                )
        self.add_sequential_layer(
            layers, LayerDesc(spec.lm_head), "model.lm_head"
        )
        return layers

    # ------------------------------------------------------------------ #
    # Logical <-> pipeline key mapping (for weight save / flex_ckpt load) #
    # ------------------------------------------------------------------ #
    # ``PipelineLayer`` names its params ``{layer_idx}.rest`` (a bare integer
    # per sequential slot). The modular AOA generation and ``save_pretrained``
    # speak LOGICAL names (``encoder.image_encoder.*`` / ``model.layers.i.*`` /
    # ...), so ``state_dict`` / ``sharded_state_dict`` must remap
    # numeric<->logical exactly like ``GPTModel`` / ``PipelinePretrainedModel``.
    # PP=VPP=1, so an ``{idx}.rest`` key reduces to prefixing with
    # ``get_sequential_name_prefixes()[idx]``. MTP weight sharing registers its
    # pivot under ``shared_layers.{key}.rest`` instead of an integer slot; such a
    # key aliases the pivot layer's parameter and must resolve to that layer's
    # logical prefix (``model.layers.{pivot}``), mirroring GPTModel.
    def _shared_layer_prefix(self, shared_key):
        for x in self._sequential_layers:
            layer = x["layer"]
            if (
                isinstance(layer, SharedLayerDesc)
                and layer.layer_name == shared_key
            ):
                return x["name_prefix"]
        raise ValueError(
            f"shared layer {shared_key} not found in sequential layers"
        )

    def _set_pipeline_name_mapping(self):
        single_to_pp = {}
        pp_to_single = {}
        prefixes = self.get_sequential_name_prefixes()
        for k in list(super().state_dict().keys()):
            seg = k.split(".")
            idx = seg[0]
            if idx.isdigit():
                pref = prefixes[idx]
                single = ([] if pref == "" else [pref]) + seg[1:]
                single = ".".join(single)
            elif idx == "shared_layers":
                pref = self._shared_layer_prefix(seg[1])
                single = ([] if pref == "" else [pref]) + seg[2:]
                single = ".".join(single)
            else:
                # non-numeric (e.g. directly-registered) keys pass through.
                single = k
            single_to_pp[single] = k
            pp_to_single[k] = single
        self._single_to_pp_mapping = single_to_pp
        self._pp_to_single_mapping = pp_to_single
        return single_to_pp

    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        if self._pp_to_single_mapping is None:
            self._set_pipeline_name_mapping()
        for k in list(sd.keys()):
            v = sd.pop(k)
            sd[self._pp_to_single_mapping[k]] = v
        return sd

    def set_state_dict(self, state_dict, *args, **kwargs):
        if self._single_to_pp_mapping is None:
            self._set_pipeline_name_mapping()
        for k in list(state_dict.keys()):
            v = state_dict.pop(k)
            if k in self._single_to_pp_mapping:
                state_dict[self._single_to_pp_mapping[k]] = v
        return super().set_state_dict(state_dict, *args, **kwargs)

    def sharded_state_dict(self, *args, **kwargs):
        ssd = super().sharded_state_dict(*args, **kwargs)
        if self._pp_to_single_mapping is None:
            self._set_pipeline_name_mapping()
        for k in list(ssd.keys()):
            v = ssd.pop(k)
            v.key = self._pp_to_single_mapping[k]
            ssd[self._pp_to_single_mapping[k]] = v

        # Expert-parallel: local expert ids -> global ids (no-op when EP=1).
        import re

        def _bump_expert(s, inc):
            return re.sub(
                r"(?<=experts\.)\d+", lambda m: str(int(m.group(0)) + inc), s
            )

        renamed = {}
        for k, v in ssd.items():
            off = getattr(v, "global_expert_id_offset", None)
            if off is not None:
                nk = _bump_expert(k, off)
                v.key = nk
                delattr(v, "global_expert_id_offset")
                renamed[nk] = v
            else:
                renamed[k] = v
        return renamed


# ======================================================================= #
# View builders + top-level builder                                        #
# ======================================================================= #
def _build_decoder_view(config: HyperBodyConfig):
    import types

    merged = {
        k: v
        for k, v in config.__dict__.items()
        if k not in ("decoder_config", "encoder_config")
    }
    dec = config.decoder_config
    for key in DECODER_VIEW_KEYS:
        merged[key] = dec.__dict__[key]
    # When first_k_dense_replace is set, feed the provider an int moe_layer_freq=1
    # so __post_init__ builds the [0] + [1]*(L-1) dense-first table (first_k cannot
    # coexist with a list moe_layer_freq). Otherwise layer 0 would be MoE instead of
    # dense, diverging from the real ernie5_v2 architecture.
    if merged.get("first_k_dense_replace"):
        merged["moe_layer_freq"] = 1
    namespace = types.SimpleNamespace(**merged)

    view = HyperBodyDecoderModelProvider.from_config(namespace)
    # The MLA core attention kernel is selected by _attn_implementation. HyperBodyConfig
    # inherits HF PretrainedConfig's default "eager", whereas ernie5_v2(lite) uses
    # "default" (fused/flash); the two differ by ~1e-5 on the same q/k/v and accumulate
    # from the first MLA layer, breaking bit-exact forward parity with standalone lite.
    # Drive it from config (yaml/json/kwargs), falling back to "default" (matching lite)
    # so HF's "eager" default does not leak in.
    view._attn_implementation = (
        getattr(config, "_attn_implementation", None) or "default"
    )
    view.multimodal_embedding = True
    # The <context> placeholder id is hyperbody_context_token_id
    view.image_token_id = config.hyperbody_context_token_id
    view.video_token_id = config.video_token_id
    # Packed decoder RoPE gate: when the encoder runs packed, the decoder is fed
    # a real packed [1, ΣS] layout, so RoPE must restart per segment. The shared
    # GPTEmbedding rope call reads this flag (default off => no behavior change
    # for any other model). It tracks the encoder's ``hyperencoder_packed_decoder``
    # directly (the two always move together for HyperBody).
    view.packed_decoder_rope = bool(
        config.encoder_config.hyperencoder_packed_decoder
    )
    return view


def _build_encoder_view(config: HyperBodyConfig, decoder_hidden: int):
    """Strip the ``encoder_`` prefix into a transient HyperEncoderConfig, then
    materialize a HyperEncoderProvider view; wire the out-projector output width
    to the decoder hidden size and pick the attention implementation that
    matches the encoder backend (``flashmask`` for the packed path, ``eager``
    for the dense ``dp`` path; see ``models/hyperencoder/layer_specs.py``).
    """
    from paddlefleet.models.hyperencoder.attn_backend import (
        use_flashmask_encoder_attn,
    )

    # Recompute: the encoder trunk supports a single degree of freedom --
    # ``recompute_granularity in (None, "full")`` (HyperEncoderProvider.__post_init__
    # pins method="uniform"/num_layers=1). Mirror the decoder's *intent*: if the
    # run recomputes at all (flat granularity is truthy, e.g. "full"/"selective"),
    # the encoder also does full recompute -- its only mode -- otherwise None.
    # We deliberately forward only the mapped granularity, NOT method/num_layers:
    # forwarding a non-"uniform"/non-1 method or num_layers would trip the
    # provider's strict guard, and the provider derives them itself when on.
    # Recompute is numerically transparent, so mapping "selective" -> "full" here
    # only changes memory/speed, never results.
    _flat_recompute = getattr(config, "recompute_granularity", None)
    enc_recompute_granularity = "full" if _flat_recompute else None

    enc = config.encoder_config
    enc_cfg = HyperEncoderConfig(
        vocab_size=enc.vocab_size,
        hidden_size=enc.hidden_size,
        intermediate_size=enc.intermediate_size,
        num_hidden_layers=enc.num_hidden_layers,
        num_attention_heads=enc.num_attention_heads,
        num_key_value_heads=enc.num_key_value_heads,
        rms_norm_eps=enc.rms_norm_eps,
        rope_theta=enc.rope_theta,
        attention_dropout=enc.attention_dropout,
        hidden_dropout_prob=enc.hidden_dropout_prob,
        attention_bias=enc.attention_bias,
        moe_intermediate_size=enc.moe_intermediate_size,
        n_routed_experts=enc.n_routed_experts,
        num_experts_per_tok=enc.num_experts_per_tok,
        n_shared_experts=enc.n_shared_experts,
        first_k_dense_replace=enc.first_k_dense_replace,
        routed_scaling_factor=enc.routed_scaling_factor,
        n_group=enc.n_group,
        topk_group=enc.topk_group,
        norm_topk_prob=enc.norm_topk_prob,
        scoring_func=enc.scoring_func,
        topk_method=enc.topk_method,
        hyperencoder_query_length=enc.hyperencoder_query_length,
        hyperencoder_seq_align=enc.hyperencoder_seq_align,
        hyperencoder_attn_backend=enc.hyperencoder_attn_backend,
        hyperencoder_packed_decoder=enc.hyperencoder_packed_decoder,
        tensor_model_parallel_size=config.tensor_model_parallel_size,
        # ``pad_token_id``: prefer the encoder-scoped value from
        # ``encoder_config`` (NOT an llm_meta key, so it survives a plain
        # assignment); fall back to the top-level value so the encoder trunk
        # keeps the SAME pad id as the decoder by default.
        # ``HyperEncoderProvider.__post_init__`` falls back to 0 only when this
        # is None, so an unset config keeps the previous behavior.
        pad_token_id=(
            enc.pad_token_id
            if getattr(enc, "pad_token_id", None) is not None
            else getattr(config, "pad_token_id", None)
        ),
        # ---- shared (non-``encoder_``-prefixed) runtime fields --------------- #
        # Geometry uses the ``encoder_`` prefix, but MoE/parallelism/fusion
        # runtime knobs are GLOBAL and SHARED: the encoder and decoder pools read
        # the exact same top-level ``HyperBodyConfig`` values. They must be
        # forwarded here or the encoder view silently falls back to
        # HyperEncoderProvider dataclass defaults, which diverge from the global
        # HyperBodyConfig values:
        #   * moe_expert_fusion / moe_deep_gemm: the encoder MoE kernel path MUST
        #     match the decoder. The modular AOA generation also keys its encoder
        #     branch off ``config.moe_expert_fusion`` (same global), so the
        #     checkpoint weight layout (fused grouped_gemm vs per-expert) stays
        #     consistent between the two pools.
        #   * router_aux_loss_coef: NOT shared -- the decoder runs noaux_tc at
        #     0.0 while the encoder's softmax/greedy router needs a real aux
        #     loss, so it comes from ``encoder_config`` (default 0.001). Not
        #     forwarding at all is wrong too: provider base default is 1e-2.
        #   * expert/context_model_parallel_size: encoder would pin EP/CP=1 even
        #     when the decoder runs EP/CP>1.
        # ``sequence_parallel`` is intentionally NOT forwarded: the provider
        # __post_init__ rejects an explicit value and derives it from tp_size.
        moe_expert_fusion=config.moe_expert_fusion,
        moe_deep_gemm=getattr(config, "moe_deep_gemm", True),
        router_aux_loss_coef=float(enc.router_aux_loss_coef),
        expert_model_parallel_size=config.expert_model_parallel_size,
        context_parallel_size=config.context_parallel_size,
        moe_shared_expert_overlap=config.moe_shared_expert_overlap,
        moe_router_load_balancing_type=config.moe_router_load_balancing_type,
        moe_token_dispatcher_type=config.moe_token_dispatcher_type,
        # fake gate (random router logits) must match the decoder for perf tests
        moe_router_force_load_balancing=getattr(
            config, "moe_router_force_load_balancing", False
        ),
        apply_rope_fusion=False,
        # Mirror the decoder's routed-scaling learnability: when the flat config
        # turns it on, the decoder's moe_router allocates a learnable
        # ``routed_scaling_factor_param``. Not forwarding it here would leave the
        # encoder on a fixed scale while the decoder learns one -> silent divergence.
        routed_scaling_factor_learnable=getattr(
            config.encoder_config, "routed_scaling_factor_learnable", False
        ),
        # Recompute intent, clamped to the encoder's only supported mode (see above).
        recompute_granularity=enc_recompute_granularity,
    )
    view = HyperEncoderProvider.from_config(enc_cfg)
    view.language_hidden_size = decoder_hidden
    view._attn_implementation = (
        "flashmask" if use_flashmask_encoder_attn(view) else "eager"
    )
    # The encoder numbers its layers 0..N-1 like the decoder does, so internal
    # medicine would otherwise write both regions' layer i into the same metric
    # key (and the same cross-layer aggregate). The tag routes the encoder's
    # metrics to ``<monitor>_enc/...``; see internal_medicine LayerId.
    view.metric_layer_region = "enc"
    view.hyperbody_context_token_id = config.hyperbody_context_token_id
    _configure_encoder_cp(view, config)
    return view


def _configure_encoder_cp(view, config):
    """Pick the encoder's CP layout (inert at CP=1).

    * default: the encoder is replicated across CP -- every rank holds the
      whole sequence, so whatever scales a per-rank statistic by cp (aux-loss
      denominator, probs gather, KV gather) must see ``cp_replicated_sequence``.
    * ``hyperencoder_cp_shard``: the trunk is a CP region like the decoder. The
      frontend scatters the packed sequence, the trunk runs
      ``flashmask_attention_cp`` on the global (LTS, UTE) row indices, the
      bridge gathers back. The pad is widened to a multiple of ``2*cp``.

    Only the layouts whose CP math was checked against the replicated encoder
    are accepted; everything else fails here rather than training silently
    wrong.
    """
    cp = config.context_parallel_size or 1
    shard = bool(config.encoder_config.hyperencoder_cp_shard) and cp > 1
    view.hyperencoder_cp_shard = shard
    view.cp_replicated_sequence = not shard
    if not shard:
        return
    from paddlefleet.models.hyperencoder.attn_backend import use_packed_decoder

    mode = config.cp_balance_mode
    # Runtime switches may be setattr'd after __init__, so re-check here.
    if not use_packed_decoder(view):
        raise ValueError(
            "hyperencoder_cp_shard needs the packed flashmask encoder "
            "(hyperencoder_attn_backend='flashmask', "
            "hyperencoder_packed_decoder=True)."
        )
    if mode not in ("dualchunk_allgather", "contiguous_allgather"):
        raise ValueError(
            "hyperencoder_cp_shard needs cp_balance_mode 'dualchunk_allgather' "
            f"or 'contiguous_allgather', got {mode!r}."
        )
    # The encoder derives sequence_parallel from TP: a second sequence split
    # on top of CP that has not been checked yet.
    if view.tensor_model_parallel_size > 1:
        raise NotImplementedError("hyperencoder_cp_shard with TP > 1.")
    if view.moe_router_load_balancing_type != "seq_aux_loss":
        raise NotImplementedError(
            "hyperencoder_cp_shard needs moe_router_load_balancing_type="
            f"'seq_aux_loss', got {view.moe_router_load_balancing_type!r}."
        )
    # The overlapped FlashMask CP kernel only takes a local length in
    # {4096, ..., 131072}, while the packed encoder length varies per batch
    # (padded to seq_align only), and its CP math was not checked here.
    if getattr(config, "flashmask_cp_overlap", False):
        raise NotImplementedError(
            "hyperencoder_cp_shard with flashmask_cp_overlap=True."
        )
    view.cp_balance_mode = mode
    view.flashmask_cp_overlap = False
    # The scatter cuts 2*cp chunks; FlashMask CP also needs an even local length.
    view.hyperencoder_seq_align = math.lcm(view.hyperencoder_seq_align, 2 * cp)


def _is_moe_expert_param(p):
    """Exactly the predicate `paddlefleet/trainer/trainer.py` uses for `is_expert`
    (`color == "moe_expert"`), so the `expert_cp_replicated` mark can never be set
    on a param the trainer does not treat as an expert.

    The color is applied for routed experts whenever EP>1 (at construction time,
    or later in `_color_expert_params` when `mtp_shared_last_layer` defers it), so
    this selects exactly the MoE expert params. Do NOT widen this with
    `is_moe_param`: the two trainers disagree on that case, and a marked param
    that `trainer.py` does not treat as an expert would fall into the x-cp branch
    while its mark is ignored (wrong by cp**2).
    """
    color = getattr(p, "color", None)
    return isinstance(color, dict) and color.get("color") == "moe_expert"


def _mark_encoder_region(model, encoder_view):
    """Mark the replicated part of the encoder for the CP grad scale.

    - MoE expert params -> `expert_cp_replicated` (their grad is inflated by cp;
      the trainer divides it by cp).
    - everything else   -> `context_parallel_disable_scale_grad` (replicated
      across CP ranks -> x1).

    Inert at CP=1 (the scale is skipped entirely), so it is safe to run
    unconditionally. The encoder region is every sequential layer whose name
    prefix starts with "encoder". Under ``hyperencoder_cp_shard`` the trunk
    ("encoder.layers.*") holds per-rank partial grads like the decoder and keeps
    the default CP handling; frontend and bridge stay replicated (the scatter /
    gather backward hands them the full grad) and stay marked.
    """
    from paddlefleet.context_parallel_utils import (
        mark_context_parallel_parameter_disable_scale_grad,
    )

    sharded = ("encoder.layers.",) if encoder_view.hyperencoder_cp_shard else ()
    prefixes = model.get_sequential_name_prefixes()
    assert len(model.run_function) == len(prefixes), (
        "layer/prefix count mismatch — PipelineLayer grouped the layers "
        "(use_cudagraph=True?). The index alignment below would mis-mark."
    )
    for i, layer in enumerate(model.run_function):
        prefix = prefixes.get(str(i), "")
        if not prefix.startswith("encoder") or prefix.startswith(sharded):
            continue
        for p in layer.parameters():
            if _is_moe_expert_param(p):
                p.expert_cp_replicated = True
            else:
                mark_context_parallel_parameter_disable_scale_grad(p)


# AOA component overrides: one minimal subclass per divergence between
# HyperBody's checkpoint layout and what the shared components produce.

# These are held in FP32 while another stack may have written them in its own
# dtype, and the engine forbids an implicit dtype change.
_MHC_HEAD_PARAMS = ("hc_head_fn", "hc_head_base", "hc_head_scale")

# Only the encoder region's archive stores the gate as BF16; Fleet keeps it
# FP32 (``_keep_in_fp32_modules``), and the decoder's archive does too.
_ENCODER_GATE_CHECKPOINT_DTYPE = "bfloat16"


def _dtype_attr(tensor):
    return str(tensor.dtype).rsplit(".", 1)[-1]


def _aoa_names(ctx, structured_name_prefix, local_name, *, drop_segment=None):
    return resolve_names(
        local_name,
        ctx.checkpoint_name_prefix,
        structured_name_prefix,
        ctx.pp_to_single_mapping,
        ctx.checkpoint_name_mapping,
        model_name_prefix=ctx.model_name_prefix,
        checkpoint_lookup_drop_segment=drop_segment,
    )


class HyperBodyHyperConnection(HyperConnectionModule):
    """mHC module carrying HyperBody's checkpoint contract for its tensors.

    Its tensors need the load-direction cast of :data:`_MHC_HEAD_PARAMS`. An
    MTP block's archive keeps the inner ``transformer_layer`` segment for the
    two mHC modules while dropping it for the rest of the layer, which no name
    mapping can express (after the drop the lookup name collides with a plain
    decoder layer's), so this subtree does not forward
    ``checkpoint_lookup_drop_segment``; outside an MTP block it is ``None``
    anyway. ``mapping_proj`` is emitted inline because the Linear family's
    helper transposes unconditionally and this leaf is not transposed."""

    def _aoa_name_pairs(self, ctx, structured_name_prefix):
        return [
            _aoa_names(ctx, structured_name_prefix, local_name)
            for local_name in self.state_dict(structured_name_prefix="")
        ]

    def gen_aoa_statements(
        self,
        ctx,
        *,
        structured_name_prefix="",
        checkpoint_lookup_drop_segment=None,
    ):
        """Checkpoint->model: a cast to the dtype these tensors are held in."""
        dtype = _dtype_attr(self.alpha_pre)
        return [
            f"{checkpoint_name} -> {model_name}, dtype='{dtype}'"
            for checkpoint_name, model_name in self._aoa_name_pairs(
                ctx, structured_name_prefix
            )
        ]

    def gen_inv_aoa_statements(
        self,
        ctx,
        *,
        structured_name_prefix="",
        checkpoint_lookup_drop_segment=None,
    ):
        """Model->checkpoint, under the name the archive really uses."""
        return [
            f"{model_name} -> {checkpoint_name}"
            for checkpoint_name, model_name in self._aoa_name_pairs(
                ctx, structured_name_prefix
            )
            if checkpoint_name != model_name
        ]


def _gen_mhc_head_cast_statements(layer, ctx, structured_name_prefix):
    """Load-direction casts for the mHC head parameters ``layer`` owns, the
    names matching the archive so the base recursion omits them."""
    statements = []
    own_tensors = layer.state_dict(
        structured_name_prefix="", include_sublayers=False
    )
    for local_name in own_tensors:
        if local_name not in _MHC_HEAD_PARAMS:
            continue
        checkpoint_name, model_name = _aoa_names(
            ctx, structured_name_prefix, local_name
        )
        dtype = _dtype_attr(getattr(layer, local_name))
        statements.append(f"{checkpoint_name} -> {model_name}, dtype='{dtype}'")
    return statements


class HyperBodyMhcContract(HyperConnectionContractLayer):
    """Contract layer whose mHC head carries the load-direction cast. Export
    needs no override: names and dtype already match the archive."""

    def gen_aoa_statements(
        self,
        ctx,
        *,
        structured_name_prefix="",
        checkpoint_lookup_drop_segment=None,
    ):
        return super().gen_aoa_statements(
            ctx,
            structured_name_prefix=structured_name_prefix,
            checkpoint_lookup_drop_segment=checkpoint_lookup_drop_segment,
        ) + _gen_mhc_head_cast_statements(self, ctx, structured_name_prefix)


class HyperBodyMTPLayer(MultiTokenPredictionLayer):
    """MTP block whose own mHC head carries the load-direction cast; its inner
    layer's mHC modules are installed separately, by the builder below."""

    def gen_aoa_statements(
        self,
        ctx,
        *,
        structured_name_prefix="",
        checkpoint_lookup_drop_segment=None,
    ):
        return super().gen_aoa_statements(
            ctx,
            structured_name_prefix=structured_name_prefix,
            checkpoint_lookup_drop_segment=checkpoint_lookup_drop_segment,
        ) + _gen_mhc_head_cast_statements(self, ctx, structured_name_prefix)


class HyperBodyGateRouter(TopKRouter):
    """MoE router carrying the archive's contract for the tensors it owns.

    ``multimodal_e_score_correction_bias`` (decoder region only) is read from
    an archive carrying it and left at its zero init -- naming no source --
    for one that does not; ``multimodal_bias_cold_start`` says which, and
    export needs no case, the archive keeping it beside the router. ``weight``
    is cross-dtype only in the encoder region."""

    _AOA_MULTIMODAL_BIAS = "multimodal_e_score_correction_bias"
    # The archive's dtype for ``weight``; None means it matches the model's.
    _AOA_GATE_CHECKPOINT_DTYPE = None

    def gen_aoa_statements(
        self,
        ctx,
        *,
        structured_name_prefix="",
        checkpoint_lookup_drop_segment=None,
    ):
        statements = []
        own_tensors = self.state_dict(
            structured_name_prefix="", include_sublayers=False
        )
        for local_name in own_tensors:
            checkpoint_name, model_name = _aoa_names(
                ctx,
                structured_name_prefix,
                local_name,
                drop_segment=checkpoint_lookup_drop_segment,
            )
            if local_name == "weight" and self._AOA_GATE_CHECKPOINT_DTYPE:
                statements.append(
                    f"{checkpoint_name} -> {model_name}, "
                    f"src_dtype='{self._AOA_GATE_CHECKPOINT_DTYPE}',"
                    f"dst_dtype='{_dtype_attr(self.weight)}'"
                )
            elif local_name == self._AOA_MULTIMODAL_BIAS and getattr(
                ctx.config, "multimodal_bias_cold_start", False
            ):
                statements.append(f"_ -> {model_name}")
            elif checkpoint_name != model_name:
                statements.append(f"{checkpoint_name} -> {model_name}")
        return statements

    def gen_inv_aoa_statements(
        self,
        ctx,
        *,
        structured_name_prefix="",
        checkpoint_lookup_drop_segment=None,
    ):
        statements = []
        own_tensors = self.state_dict(
            structured_name_prefix="", include_sublayers=False
        )
        for local_name in own_tensors:
            checkpoint_name, model_name = _aoa_names(
                ctx,
                structured_name_prefix,
                local_name,
                drop_segment=checkpoint_lookup_drop_segment,
            )
            if local_name == "weight" and self._AOA_GATE_CHECKPOINT_DTYPE:
                statements.append(
                    f"{model_name} -> {checkpoint_name}, "
                    f"src_dtype='{_dtype_attr(self.weight)}',"
                    f"dst_dtype='{self._AOA_GATE_CHECKPOINT_DTYPE}'"
                )
            elif checkpoint_name != model_name:
                statements.append(f"{model_name} -> {checkpoint_name}")
        return statements


class HyperBodyEncoderGateRouter(HyperBodyGateRouter):
    """Encoder MoE router: FP32 gate weight against a BF16 archive, a pair
    both directions must declare, the archive side not being derivable."""

    _AOA_GATE_CHECKPOINT_DTYPE = _ENCODER_GATE_CHECKPOINT_DTYPE


class HyperBodyMoELayer(MoELayer):
    """MoE layer whose router carries HyperBody's archive facts.

    Swapping the gate after ``super().__init__`` is the pattern
    ``Gemma4MoELayer`` already uses, the base building its own ``TopKRouter``
    unconditionally. Building it twice runs ``config.init_method`` an extra
    time, shifting the from-scratch init of every parameter built after the
    first swapped layer; loading overwrites those, so only a never-loaded
    tensor keeps the shift -- which is why the installer swaps only where a
    region needs an override."""

    _AOA_ROUTER_CLASS = HyperBodyGateRouter

    def __init__(self, config, sublayers=None, pg_collection=None):
        super().__init__(config, sublayers, pg_collection)
        self.gate = self._AOA_ROUTER_CLASS(
            config=config, pg_collection=pg_collection
        )


class HyperBodyEncoderMoELayer(HyperBodyMoELayer):
    """Encoder MoE layer: same swap, but the router that casts the gate."""

    _AOA_ROUTER_CLASS = HyperBodyEncoderGateRouter


class HyperBodyMlpProjector(MlpProjector):
    """Modality projector whose ``layers`` leaf is transposed in the archive.
    ``MlpProjector`` holds a bare ``paddle.nn.Linear``, which has no AOA
    override of its own, so the base recursion would emit it untransposed."""

    def gen_aoa_statements(
        self,
        ctx,
        *,
        structured_name_prefix="",
        checkpoint_lookup_drop_segment=None,
    ):
        if not isinstance(self.layers, paddle.nn.Linear):
            return super().gen_aoa_statements(
                ctx,
                structured_name_prefix=structured_name_prefix,
                checkpoint_lookup_drop_segment=checkpoint_lookup_drop_segment,
            )
        return gen_linear_aoa_statements(
            self.layers,
            ctx,
            structured_name_prefix=f"{structured_name_prefix}layers.",
            checkpoint_lookup_drop_segment=checkpoint_lookup_drop_segment,
        )

    def gen_inv_aoa_statements(
        self,
        ctx,
        *,
        structured_name_prefix="",
        checkpoint_lookup_drop_segment=None,
    ):
        if not isinstance(self.layers, paddle.nn.Linear):
            return super().gen_inv_aoa_statements(
                ctx,
                structured_name_prefix=structured_name_prefix,
                checkpoint_lookup_drop_segment=checkpoint_lookup_drop_segment,
            )
        return gen_linear_inv_aoa_statements(
            self.layers,
            ctx,
            structured_name_prefix=f"{structured_name_prefix}layers.",
            checkpoint_lookup_drop_segment=checkpoint_lookup_drop_segment,
        )


def _install_mhc_aoa_override(transformer_layer_spec):
    """A layer with hyper-connections off carries ``IdentityOp``, so skip it."""
    sublayers_spec = transformer_layer_spec.sublayers_spec
    for field_name in (
        "self_attention_hyper_connection",
        "mlp_hyper_connection",
    ):
        sub_spec = getattr(sublayers_spec, field_name, None)
        if (
            isinstance(sub_spec, LayerSpec)
            and sub_spec.layer is HyperConnectionModule
        ):
            sub_spec.layer = HyperBodyHyperConnection


def _install_moe_gate_override(transformer_layer_spec, layer_class):
    """A dense layer carries a plain MLP spec, so skip it."""
    mlp_spec = getattr(transformer_layer_spec.sublayers_spec, "mlp", None)
    if isinstance(mlp_spec, LayerSpec) and mlp_spec.layer is MoELayer:
        mlp_spec.layer = layer_class


def _install_aoa_component_overrides(
    *,
    decoder_layer_specs,
    mtp_layer_specs,
    encoder_layer_specs,
    mhc_contract_spec,
    multimodal_router_bias=False,
):
    """Install every AOA component override into the already-built specs.

    ``get_gpt_mtp_layers_spec`` builds the MTP block's inner transformer layer
    itself rather than reusing a decoder layer, so that one is patched
    separately. The decoder region needs its router swapped only when the
    multimodal router bias exists, that buffer being the region's only
    divergence from the identity rule -- which also keeps the extra
    ``init_method`` call out of runs not needing it."""
    for layer_spec in decoder_layer_specs:
        _install_mhc_aoa_override(layer_spec)
        if multimodal_router_bias:
            _install_moe_gate_override(layer_spec, HyperBodyMoELayer)
    for mtp_spec in mtp_layer_specs or ():
        # ``mtp_load_weight_only`` selects a block class with no mHC head.
        if mtp_spec.layer is MultiTokenPredictionLayer:
            mtp_spec.layer = HyperBodyMTPLayer
        inner_spec = getattr(mtp_spec.sublayers_spec, "transformer_layer", None)
        if isinstance(inner_spec, LayerSpec):
            _install_mhc_aoa_override(inner_spec)
            if multimodal_router_bias:
                _install_moe_gate_override(inner_spec, HyperBodyMoELayer)
    for layer_spec in encoder_layer_specs:
        _install_moe_gate_override(layer_spec, HyperBodyEncoderMoELayer)
    if (
        isinstance(mhc_contract_spec, LayerSpec)
        and mhc_contract_spec.layer is HyperConnectionContractLayer
    ):
        mhc_contract_spec.layer = HyperBodyMhcContract


def build_hyperbody_unified_model(
    config: HyperBodyConfig, *, num_stages=1, loss_fn=None
):
    """Assemble the unified single-PipelineLayer model.

    Mirrors ``build_hyperbody_decoder_model`` (narrowed ``gpt_builder``): build
    the decoder gpt spec, extract its embedding/trunk/norm/lm_head sub-specs,
    build the encoder frontend/trunk/bridge specs, then wrap everything in one
    ``HyperBodyUnifiedModel`` root LayerSpec and materialize via
    ``build_spec_layer``.
    """
    # #1: this phase only supports PP=1. Reject both an explicit num_stages!=1
    # and a config that requests pipeline_model_parallel_size>1 (the builder
    # hardcodes num_stages=1 downstream, so an unguarded PP>1 config would be
    # silently ignored rather than honored).
    pp_size = getattr(config, "pipeline_model_parallel_size", 1)
    if num_stages != 1 or (pp_size is not None and pp_size > 1):
        raise NotImplementedError(
            "HyperBody unified model only supports PP=1 in this phase."
        )

    from .configuration import (
        HyperBodyDecoderConfig,
        HyperBodyEncoderConfig,
    )

    if isinstance(getattr(config, "decoder_config", None), dict):
        config.decoder_config = HyperBodyDecoderConfig(**config.decoder_config)
    if isinstance(getattr(config, "encoder_config", None), dict):
        config.encoder_config = HyperBodyEncoderConfig(**config.encoder_config)

    decoder_view = _build_decoder_view(config)

    # #4: mirror the standalone decoder's rejection guards (MTP /
    # separate_mtp_headloss / EmptyLayer head|tail / ringmoe / meta-device) so
    # the unified builder fails loudly on unsupported branches too. Current
    # production configs trigger none of these => behavior unchanged.
    _reject_unsupported_decoder_branches(decoder_view)

    decoder_layers_spec = get_hyperbody_decoder_layer_specs(decoder_view)
    # MTP: build the MTP layer spec(s) exactly like the standard gpt_builder so
    # dec_sub.mtp gets populated and the MTP layers show up in the unified
    # PipelineLayer. Per-depth weight sharing (mtp_shared_last_layer /
    # mtp_shared_weights) is applied at layer assembly in get_layer_desc_list via
    # SharedLayerDesc; only the MTP forward/loss is still deferred.
    mtp_layers_spec = None
    if _get_effective_mtp_layers(decoder_view) > 0:
        mtp_layers_spec = get_gpt_mtp_layers_spec(
            decoder_view, decoder_layers_spec
        )

    gpt_spec = get_gpt_spec(
        config=decoder_view,
        head_empty_layers_spec=[],
        # HyperBody's own decoder-layers spec. It builds the same layers as the
        # shared GPT spec (via get_gpt_layer_local_spec) and supports the full
        # ernie5_v2 / dsv4_hybrid attention family; it additionally enforces a
        # per-layer 0/1 list moe_layer_freq (rejecting int i%N semantics).
        transformer_layers_spec=decoder_layers_spec,
        tail_empty_layers_spec=[],
        mtp_layers_spec=mtp_layers_spec,
        vocab_size=decoder_view.vocab_size,
        tie_word_embeddings=decoder_view.tie_word_embeddings,
        max_sequence_length=decoder_view.max_sequence_length,
        position_embedding_type=decoder_view.position_embedding_type,
        rotary_percent=decoder_view.rotary_percent,
        rotary_base=decoder_view.rope_theta,
        swa_rotary_base=decoder_view.swa_rope_theta,
        rope_scaling=decoder_view.rope_scaling,
        parallel_output=decoder_view.parallel_output,
    )
    dec_sub = gpt_spec.sublayers_spec

    encoder_view = _build_encoder_view(config, decoder_view.hidden_size)
    enc_layer_specs = get_hyperbody_encoder_layer_specs(encoder_view)

    frontend_spec = LayerSpec(
        HyperBodyEncoderFrontEnd, extra_kwargs={"config": encoder_view}
    )
    bridge_spec = LayerSpec(
        HyperBodyEncoderBridge, extra_kwargs={"config": encoder_view}
    )

    # Every spec exists by now, so the AOA overrides go in here, before the
    # root spec takes them. Read off ``dec_sub``, ``get_gpt_spec`` being what
    # finally decides those fields.
    _install_aoa_component_overrides(
        decoder_layer_specs=dec_sub.transformer_layers,
        mtp_layer_specs=dec_sub.mtp,
        encoder_layer_specs=enc_layer_specs,
        mhc_contract_spec=dec_sub.mhc_contract,
        multimodal_router_bias=bool(
            getattr(config, "moe_multimodal_router_bias", False)
        ),
    )

    sublayers_spec = HyperBodySublayersSpec(
        encoder_frontend=frontend_spec,
        encoder_layers=enc_layer_specs,
        bridge=bridge_spec,
        decoder_embedding=dec_sub.embedding,
        mhc_expand=dec_sub.mhc_expand,
        decoder_layers=dec_sub.transformer_layers,
        mhc_contract=dec_sub.mhc_contract,
        mtp=dec_sub.mtp,
        layer_norm=dec_sub.layer_norm,
        lm_head=dec_sub.lm_head,
    )

    root = LayerSpec(
        HyperBodyUnifiedModel,
        extra_kwargs={"config": decoder_view, "tie_word_embeddings": False},
        sublayers_spec=sublayers_spec,
    )
    model = build_spec_layer(
        root,
        loss_fn=LanguageLoss(decoder_view) if loss_fn is None else loss_fn,
        num_stages=num_stages,
        seg_method="layer:TransformerLayer|EmptyLayer",
    )
    _mark_encoder_region(model, encoder_view)
    return model


# ======================================================================= #
# Registration / entry classes                                            #
# ======================================================================= #
class HyperBodyPretrainedModel(PretrainedModel):
    config_class = HyperBodyConfig
    base_model_prefix = "hyperbody"
    input_modalities = ["image", "audio", "text"]
    # The router gate is fp32 on the fleet side (PaddleFleet forces fp32 while
    # use_accuracy_compatible=False). The encoder-region archive stores it as
    # bf16, so those gate statements carry a src/dst dtype pair; the
    # decoder-region archive stores fp32 and needs none.
    _keep_in_fp32_modules = ["mlp.gate.weight"]


class HyperBodyModelDist(HyperBodyPretrainedModel):
    """Factory returning the unified ``HyperBodyUnifiedModel`` (a PipelineLayer).

    Mirrors ``Qwen3VLModel.__new__``: build the pipe via the unified builder
    (PP=1 in this phase). Returns the pipe directly, not ``cls``.
    """

    config_class = HyperBodyConfig

    def __new__(cls, config: HyperBodyConfig, have_criterion: bool = True):
        # build_spec_layer always appends a loss layer for structure; the PP=1
        # entry applies pipe._loss_fn[0] manually, so have_criterion does not
        # change the assembled graph here.
        pipe = build_hyperbody_unified_model(config, num_stages=1)
        pipe.config_to_save = config
        return pipe


# Alias kept for parity with the composite-era public name.
HyperBodyModel = HyperBodyModelDist


class HyperBodyForConditionalGeneration(HyperBodyPretrainedModel):
    """PP=1 smoke entry.

    Holds ``self.pipe`` (the unified ``HyperBodyUnifiedModel``). The pipe does
    NOT auto-apply its loss (stored in ``pipe._loss_fn[0]``), so this entry
    applies it manually when ``labels`` are given.
    """

    config_class = HyperBodyConfig
    # Fleet model: this wraps a PipelineLayer whose params carry logical fleet
    # names (not HF names). is_fleet=True makes from_pretrained SKIP the identity
    # dtype append (model_utils.py:3421) -- that block would otherwise emit bogus
    # `<pipe.-prefixed key> -> <same>, dtype=...` statements whose LHS has no
    # source in the HF safetensors. The modular AOA generation already carries
    # every needed dtype spec (e.g. gate.weight bf16->fp32). Matches the *Pipe
    # entries.
    is_fleet = True

    def __init__(self, config: HyperBodyConfig):
        super().__init__(config)
        self.pipe = build_hyperbody_unified_model(config, num_stages=1)

    # ---- weight name I/O: delegate straight to the pipe -------------------- #
    # The wrapper holds the graph under ``self.pipe``. The default nn.Layer
    # recursion would call ``self.pipe.sharded_state_dict(structured_name_prefix=
    # "pipe.")`` -> the pipe emits ``pipe.{idx}.rest`` keys, then its own
    # numeric<->logical remap looks up ``pipe.0.embed_tokens.weight`` in a table
    # built from UN-prefixed names and raises KeyError (R38). Delegating here (no
    # prefix) makes the wrapper expose the pipe's LOGICAL names verbatim, which is
    # exactly what the modular AOA generation's RHS + flex_checkpoint expect (and
    # identical to what the bare *Pipe entry produces). This unblocks both
    # flex_checkpoint LOAD (from_pretrained) and SAVE through the wrapper.
    def state_dict(self, *args, **kwargs):
        return self.pipe.state_dict(*args, **kwargs)

    def set_state_dict(self, state_dict, *args, **kwargs):
        return self.pipe.set_state_dict(state_dict, *args, **kwargs)

    def sharded_state_dict(self, *args, **kwargs):
        return self.pipe.sharded_state_dict(*args, **kwargs)

    def forward(
        self,
        input_ids: paddle.Tensor | None = None,
        context_ids: paddle.Tensor | None = None,
        image=None,
        audio=None,
        labels: paddle.Tensor | None = None,
        attn_mask_startend_row_indices: paddle.Tensor | None = None,
        position_ids: paddle.Tensor | None = None,
        cu_seqlens: paddle.Tensor | None = None,
        cu_seqlens_context=None,
        **kwargs,
    ):
        if cu_seqlens is not None:
            # Packed decoder: derive both the block-diagonal causal flashmask
            # boundaries and the per-segment reset RoPE position_ids from the
            # decoder segment boundaries. cu_seqlens is a [num_seg + 1] int32
            # vector of cumulative lengths over the packed decoder token axis
            # (ΣS == input_ids.shape[-1]).
            import numpy as _np

            from paddlefleet.transformer.multi_token_prediction import (
                build_startend_row_indices_from_cu_seqlens,
            )

            total_len = int(input_ids.shape[-1])
            # [1, 1, ΣS, 1] int32: each token records its own segment `end`
            # boundary -> flashmask keeps attention inside the segment while
            # is_causal=True (decoder layer spec) makes it lower-triangular.
            attn_mask_startend_row_indices = (
                build_startend_row_indices_from_cu_seqlens(
                    cu_seqlens, batch_size=1, seq_len=total_len
                )
            )
            cu_np = (
                cu_seqlens.numpy()
                if isinstance(cu_seqlens, paddle.Tensor)
                else _np.asarray(cu_seqlens)
            )
            # Reset position_ids: 0..L_i-1 per segment, concatenated -> [1, ΣS].
            pos_np = _np.concatenate(
                [
                    _np.arange(
                        int(cu_np[j + 1]) - int(cu_np[j]), dtype=_np.int64
                    )
                    for j in range(len(cu_np) - 1)
                ]
            )
            position_ids = paddle.to_tensor(
                pos_np[None, :], place=input_ids.place
            )

        input_dict = {
            "input_ids": input_ids,
            "context_ids": context_ids,
            "image": image,
            "audio": audio,
            "labels": labels,
            "attn_mask_startend_row_indices": attn_mask_startend_row_indices,
            "position_ids": position_ids,
            "cu_seqlens": cu_seqlens,
            "cu_seqlens_context": cu_seqlens_context,
        }
        out = self.pipe(input_dict)
        logits = out["logits"] if isinstance(out, dict) else out
        if labels is None:
            return logits
        loss = self.pipe._loss_fn[0](logits, labels)
        if isinstance(loss, (list, tuple)):
            loss = loss[0]
        return loss


class HyperBodyModelPipe(HyperBodyPretrainedModel):
    config_class = HyperBodyConfig
    is_fleet = True

    def __new__(cls, config: HyperBodyConfig, have_criterion: bool = True):
        return HyperBodyModelDist.__new__(cls, config, have_criterion)


class HyperBodyForCausalLMPipe(
    HyperBodyPretrainedModel, GeneralModelForCausalLMPipe
):
    config_class = HyperBodyConfig
    is_fleet = True

    def __new__(cls, config: HyperBodyConfig, have_criterion: bool = True):
        return HyperBodyModelDist.__new__(cls, config, have_criterion)
