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

"""DSpark block attention over target context and draft noise.

The draft uses the CSA ratio-0 path with a causal context window and
bidirectional attention within each draft block.
"""

from __future__ import annotations

from typing import Any

import paddle
from paddle.distributed.fleet.utils import recompute

from paddlefleet.models.common.embeddings.rope_utils import (
    _apply_rotary_pos_emb_bshd,
)
from paddlefleet.recompute_utils import has_recovered
from paddlefleet.transformer.dsv4_hybrid_attention import (
    DSv4HybridSelfAttention,
)
from paddlefleet.transformer.identity_op import IdentityOp
from paddlefleet.transformer.transformer_layer import (
    HyperConnectionTransformerLayer,
)


def gather_sequence(sequence: Any, indices: Any) -> Any:
    """Gather independent per-batch sequence indices and retain feature dims.

    ``sequence`` is ``[batch_size, seq_len, ...]`` and ``indices`` is ``[batch_size, N, window_size]``; returns
    ``[batch_size, N, window_size, ...]`` without mixing batch rows.
    """
    if len(sequence.shape) < 2 or len(indices.shape) != 3:
        raise ValueError(
            "sequence must be [batch_size,seq_len,...] and indices [batch_size,N,window_size]"
        )
    batch_size, sequence_length = sequence.shape[:2]
    if indices.shape[0] != batch_size:
        raise ValueError("sequence and indices batch sizes differ")
    offsets = paddle.arange(batch_size, dtype="int64").reshape([-1, 1, 1])
    offsets = offsets * int(sequence_length)
    flat_indices = (indices.astype("int64") + offsets).reshape([-1])
    trailing = list(sequence.shape[2:])
    flat = sequence.reshape([batch_size * sequence_length, *trailing])
    return paddle.gather(flat, flat_indices, axis=0).reshape(
        [*indices.shape, *trailing]
    )


def build_dspark_topk_idxs(
    *,
    anchor_positions: Any,
    block_keep_mask: Any,
    doc_start_per_pos: Any,
    seq_len: int,
    block_size: int,
    window_size: int,
    chunk_seam: int | None = None,
) -> Any:
    """Per-draft-query KV row-index table for the shared-KV CSA sparse core.

    Rows index into ``kv_full = concat(context_kv[0:seq_len], noise_kv[seq_len:seq_len+anchor_num*block_size])``.
    For anchor ``a`` and block offset ``j`` the visible set is
    ``[a's causal, same-document window of window_size context rows] + [a's block_size noise rows]``.
    Block-internal attention is **bidirectional**: all block_size queries of a block share
    the same block_size noise rows (query-invariant). ``-1`` marks an empty slot.

    Shapes: ``anchor_positions``/``block_keep_mask`` ``[batch_size, anchor_num]``,
    ``doc_start_per_pos`` ``[batch_size, seq_len]``; returns ``[batch_size, anchor_num*block_size, window_size+block_size]`` int32. Query order
    is anchor-major / block-minor: ``q = a*block_size + j``.
    """
    batch_size, anchor_num = anchor_positions.shape
    ap = anchor_positions.astype("int64")  # [batch_size, anchor_num]
    keep = block_keep_mask.astype("bool")  # [batch_size, anchor_num]

    # --- context window: a-window_size+1 .. a  (causal, clamped, same-document) ---
    win_off = paddle.arange(-window_size + 1, 1, dtype="int64").reshape(
        [1, 1, window_size]
    )
    ctx = ap.unsqueeze(-1) + win_off  # [batch_size, anchor_num, window_size]
    ctx_valid = ctx >= 0
    ctx_safe = paddle.clip(ctx, 0, seq_len - 1)
    seg_ctx = gather_sequence(
        doc_start_per_pos.unsqueeze(-1), ctx_safe
    ).squeeze(-1)  # [batch_size, anchor_num, window_size]
    seg_anchor = paddle.take_along_axis(
        doc_start_per_pos.astype("int64"), ap, axis=1
    ).unsqueeze(-1)  # [batch_size, anchor_num, 1]
    ctx_valid = ctx_valid & (seg_ctx == seg_anchor) & keep.unsqueeze(-1)
    if chunk_seam is not None:
        # A CP zigzag shard is two globally distant intervals concatenated, so
        # local adjacency stops matching global adjacency at ``chunk_seam``. The
        # document filter above cannot see that seam -- when one document spans
        # it both halves carry the same start -- so a window straddling it would
        # feed the anchor rows it never precedes. ``ctx <= ap`` always, so "same
        # chunk" reduces to "the anchor is left of the seam, or the row is
        # right of it".
        ctx_valid = ctx_valid & (
            (ap.unsqueeze(-1) < chunk_seam) | (ctx >= chunk_seam)
        )
    neg = paddle.full_like(ctx_safe, -1)
    ctx_idx = paddle.where(
        ctx_valid, ctx_safe, neg
    )  # [batch_size, anchor_num, window_size]

    # --- block noise: seq_len + a*block_size + [0..block_size-1], all block_size (bidirectional) ---
    a_ids = paddle.arange(anchor_num, dtype="int64").reshape([1, anchor_num, 1])
    blk_off = paddle.arange(block_size, dtype="int64").reshape(
        [1, 1, block_size]
    )
    noise_idx = (seq_len + a_ids * block_size + blk_off).expand(
        [batch_size, anchor_num, block_size]
    )  # [batch_size, anchor_num, block_size]
    noise_valid = keep.unsqueeze(-1).expand(
        [batch_size, anchor_num, block_size]
    )
    noise_idx = paddle.where(
        noise_valid, noise_idx, paddle.full_like(noise_idx, -1)
    )

    # per-anchor [batch_size,anchor_num,window_size+block_size] -> broadcast to block_size queries -> [batch_size, anchor_num*block_size, window_size+block_size]
    per_anchor = paddle.concat(
        [ctx_idx, noise_idx], axis=-1
    )  # [batch_size, anchor_num, window_size+block_size]
    rows = (
        per_anchor.unsqueeze(2)
        .expand([batch_size, anchor_num, block_size, window_size + block_size])
        .reshape(
            [batch_size, anchor_num * block_size, window_size + block_size]
        )
    )
    return rows.astype("int32")


def _explicit_freqs(rotary_embedding: Any, position_ids: Any) -> Any:
    positions = position_ids.astype(rotary_embedding.inv_freq.dtype)
    base = positions.unsqueeze(-1) * rotary_embedding.inv_freq.reshape(
        [1, 1, -1]
    )
    return paddle.concat([base, base], axis=-1).unsqueeze(2)


def _apply_hca_rope(value, freqs, *, inverse=False, config=None):
    return _apply_rotary_pos_emb_bshd(
        value,
        freqs,
        rotary_interleaved=False,
        multi_latent_attention=True,
        mla_output_remove_interleaving=True,
        high_precision_rope=bool(getattr(config, "high_precision_rope", False))
        if config is not None
        else False,
        inverse=inverse,
    )


class DsparkHCAAttention(DSv4HybridSelfAttention):
    """DSpark draft attention over a shared target+noise KV (CSA ratio=0 core).

    Constructed for a draft layer whose ``csa_compress_ratios`` entry is ``0``,
    so ``self.core_attention`` is a ``CompressedSparseAttention`` with
    ``compressor is None`` and ``indexer is None`` (window-only / MQA KV). DSpark
    reuses the inherited KV/Q projections and ``core_attention.compressed_sparse_attn``
    kernel, swapping the causal sliding-window index table for
    ``build_dspark_topk_idxs`` (per-anchor context window + block-internal
    bidirectional noise) over one shared ``kv_full``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # At ratio=0 Fleet already leaves core_attention.compressor/indexer None;
        # guard defensively in case a non-zero ratio layer is ever reused here.
        core = getattr(self, "core_attention", None)
        if getattr(core, "compressor", None) is not None:
            core.compressor = None
            core.is_hca_layer = False
        if getattr(core, "indexer", None) is not None:
            core.indexer = None

    def project_context_kv(self, hidden_states: Any, position_ids: Any) -> Any:
        """``linear_kv_proj -> kv_layernorm -> partial RoPE`` -> single-head KV.

        Applied once over the full target context (positions 0..seq_len-1) and once
        over the draft noise (draft positions); the two are concatenated into the
        shared ``kv_full`` consumed by ``compressed_sparse_attn``.
        """
        key_value, _ = self.linear_kv_proj(hidden_states)
        if self.config.swa_high_precision_norm:
            key_value = self.kv_layernorm(
                key_value,
                high_precision_norm=True,
                return_high_precision_norm=True,
            )
        else:
            key_value = self.kv_layernorm(key_value)
        pos_dim = int(self.qk_pos_emb_head_dim)
        if pos_dim:
            nope_dim = int(self.v_head_dim) - pos_dim
            freqs = _explicit_freqs(self.rotary_pos_emb, position_ids)
            key_value = paddle.concat(
                [
                    key_value[..., :nope_dim],
                    _apply_hca_rope(
                        key_value[..., nope_dim:].unsqueeze(2),
                        freqs,
                        config=self.config,
                    ).squeeze(2),
                ],
                axis=-1,
            )
        return key_value

    def dspark_sparse_attention(self, query, kv_full, topk_idxs):
        """Thin wrapper over the production CSA sparse kernel (ratio=0 path).

        ``query`` ``[batch_size, anchor_num*block_size, num_heads, v_head_dim]``, ``kv_full``
        ``[batch_size, seq_len+anchor_num*block_size, v_head_dim]`` (single-head MQA KV), ``topk_idxs``
        ``[batch_size, anchor_num*block_size, window_size+block_size]`` int32 (``-1`` = empty slot, from
        ``build_dspark_topk_idxs``). Block-internal rows are non-causal, so the
        compact-prefix fast path is disabled (``topk_idxs_compacted=False``).
        """
        core = self.core_attention
        scale = float(
            getattr(core, "softmax_scale", None) or self.v_head_dim**-0.5
        )
        return core.compressed_sparse_attn(
            query,
            kv_full,
            core.attn_sink,
            topk_idxs.astype("int32"),
            scale,
            indexer_topk=0,
            topk_idxs_compacted=False,
        )

    def forward_dspark(
        self,
        context_hidden,
        noise_hidden,
        *,
        anchor_positions,
        block_keep_mask,
        doc_start_per_pos,
        context_position_ids,
        draft_position_ids,
        block_size,
        chunk_seam=None,
    ):
        """Run block attention over shared target and draft KV.

        ``context_hidden`` ``[batch_size,seq_len,H]`` (feature-projected target context),
        ``noise_hidden`` ``[batch_size,anchor_num*block_size,H]`` (readonly noise embeddings). Returns the
        attention output ``[batch_size,anchor_num*block_size,H_out]``. KV = ``concat(context_kv, noise_kv)``,
        selected per draft query by ``build_dspark_topk_idxs``; block-internal
        attention is bidirectional, context is causal/windowed/same-document.
        """
        b, n_draft, _ = noise_hidden.shape
        seq_len = int(context_hidden.shape[1])
        window = int(self.core_attention.window_size)
        pos_dim = int(self.qk_pos_emb_head_dim)
        nope_dim = int(self.v_head_dim) - pos_dim

        # --- query from noise ---
        q_compressed, _ = self.linear_q_down_proj(noise_hidden)
        q_compressed = self.q_layernorm(q_compressed)
        if getattr(self, "use_vha_premix", False):
            groups = self.vha_premix_groups
            q = q_compressed.reshape([b, n_draft, groups, self.vha_premix_dq])
            q = paddle.einsum(
                "btgr,krd->btkgd", q, self.vha_premix_weight
            ).reshape([b, n_draft, self.num_attention_heads, self.v_head_dim])
        else:
            q, _ = self.linear_q_up_proj(q_compressed)
            q = q.reshape(
                [b, n_draft, self.num_attention_heads, self.v_head_dim]
            )
        norm_dtype = q.dtype
        q32 = q.astype("float32") if self.config.swa_high_precision_norm else q
        q32 = q32 * paddle.rsqrt(
            q32.square().mean(axis=-1, keepdim=True)
            + float(self.config.rms_norm_eps)
        )
        q = q32.astype(norm_dtype)
        if pos_dim:
            q_freqs = _explicit_freqs(self.rotary_pos_emb, draft_position_ids)
            q = paddle.concat(
                [
                    q[..., :nope_dim],
                    _apply_hca_rope(
                        q[..., nope_dim:], q_freqs, config=self.config
                    ),
                ],
                axis=-1,
            )

        # --- shared single-head KV: context (0..seq_len-1) + noise (seq_len..seq_len+N-1) ---
        context_kv = self.project_context_kv(
            context_hidden, context_position_ids
        )
        noise_kv = self.project_context_kv(noise_hidden, draft_position_ids)
        kv_full = paddle.concat([context_kv, noise_kv], axis=1)

        topk_idxs = build_dspark_topk_idxs(
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            doc_start_per_pos=doc_start_per_pos,
            seq_len=seq_len,
            block_size=int(block_size),
            window_size=window,
            chunk_seam=chunk_seam,
        )

        out = self.dspark_sparse_attention(q, kv_full, topk_idxs)
        out = out.reshape(
            [b, n_draft, self.num_attention_heads, self.v_head_dim]
        )
        if pos_dim:
            out = paddle.concat(
                [
                    out[..., :nope_dim],
                    _apply_hca_rope(
                        out[..., nope_dim:],
                        q_freqs,
                        inverse=True,
                        config=self.config,
                    ),
                ],
                axis=-1,
            )
        out = out.reshape(
            [b, n_draft, self.num_attention_heads * self.v_head_dim]
        )
        if getattr(self, "use_vha_postmix", False):
            out = self._apply_vha_postmix(out)
        out = out.reshape([b, n_draft, self.o_local_groups, -1])
        group_weight = self.linear_o_group_proj.reshape(
            [self.o_local_groups, self.config.o_lora_rank, -1]
        )
        if group_weight.dtype != out.dtype:
            group_weight = group_weight.astype(out.dtype)
        out = paddle.einsum("...gd,grd->...gr", out, group_weight)
        out = out.reshape([b, n_draft, -1])
        if self.gated_attention:
            out = self._gate(q_compressed.reshape([b, n_draft, -1]), out)
        out, bias = self.o_proj(out)
        if bias is not None:
            out = out + bias
        return out


def _draft_forward_impl(
    layer,
    hidden_states,
    dspark_context,
    dspark_anchor_positions,
    dspark_block_keep_mask,
    dspark_doc_start_per_pos,
    dspark_context_position_ids,
    dspark_draft_position_ids,
    dspark_block_size,
    dspark_chunk_seam,
):
    """Recompute entry point spelling out every ``state`` key.

    ``recompute`` only tracks top-level arguments that are tensors and flattens
    ``**kwargs`` positionally, so passing ``state`` as one dict would both hide
    the grad-carrying inputs and drop the ``dspark_*`` names.
    """
    return layer._forward_impl(
        hidden_states=hidden_states,
        dspark_context=dspark_context,
        dspark_anchor_positions=dspark_anchor_positions,
        dspark_block_keep_mask=dspark_block_keep_mask,
        dspark_doc_start_per_pos=dspark_doc_start_per_pos,
        dspark_context_position_ids=dspark_context_position_ids,
        dspark_draft_position_ids=dspark_draft_position_ids,
        dspark_block_size=dspark_block_size,
        dspark_chunk_seam=dspark_chunk_seam,
    )


class DsparkTransformerLayer(HyperConnectionTransformerLayer):
    """mHC transformer layer whose attention is the DSpark shared-KV attention.

    Reuses the parent mHC machinery (stream aggregate -> input_layernorm ->
    attention -> fused H_res/H_post/BDA -> MoE MLP) but replaces the self-attn
    sub-step with ``self.self_attn.forward_dspark`` over the target+noise KV.
    The DSpark context tensors travel in the ``state`` dict under ``dspark_*``
    keys and reach ``_forward_attention`` via ``_forward_impl(**state)`` kwargs.
    """

    def forward(self, state):
        hidden = state["hidden_states"]
        if self.training and paddle.is_grad_enabled() and hidden.stop_gradient:
            # Ephemeral grad leaf so the mHC inner parameter graph is consumed
            # even though the first draft layer receives a frozen noise stream.
            hidden = hidden.detach()
            hidden.stop_gradient = False
            state = {**state, "hidden_states": hidden}
        # ``_forward_impl`` is the parent's, so the parent's activation
        # checkpointing has to be re-applied here: ``full_recompute`` is only
        # consumed by the wrapper in ``TransformerLayer.forward``, which this
        # override replaces.
        if self.full_recompute or (not has_recovered()):
            output = recompute(
                _draft_forward_impl,
                self,
                state["hidden_states"],
                state["dspark_context"],
                state["dspark_anchor_positions"],
                state["dspark_block_keep_mask"],
                state["dspark_doc_start_per_pos"],
                state["dspark_context_position_ids"],
                state["dspark_draft_position_ids"],
                state["dspark_block_size"],
                state["dspark_chunk_seam"],
                use_reentrant=True,
            )
        else:
            output = self._forward_impl(**state)
        if isinstance(output, tuple):
            output = output[0]
        return {**state, "hidden_states": output}

    def _forward_attention(
        self,
        hidden_states,
        attention_mask=None,
        context=None,
        is_first_fwd=False,
        **kwargs,
    ):
        if not isinstance(self.cross_attention, IdentityOp):
            raise TypeError("DSpark draft does not support cross attention")
        for key in (
            "dspark_context",
            "dspark_anchor_positions",
            "dspark_block_keep_mask",
            "dspark_doc_start_per_pos",
            "dspark_context_position_ids",
            "dspark_draft_position_ids",
            "dspark_block_size",
        ):
            if kwargs.get(key) is None:
                raise ValueError(f"DSpark draft layer missing state key: {key}")

        original_residual = hidden_states
        ori_dtype = hidden_states.dtype
        mhc_manager = self._mhc_block_manager(half_layer=0)
        mhc_recompute_on = mhc_manager is not None or (
            self.recompute_mhc_forward and self.training
        )
        (
            aggregated,
            h_res,
            h_post,
            self._attn_mhc_recompute,
        ) = self._mhc_head(
            self.self_attention_hyper_connection,
            hidden_states,
            mhc_manager,
            mhc_recompute_on,
        )
        aggregated = aggregated.to(ori_dtype)
        input_layernorm_output = self._mhc_layernorm(
            self.input_layernorm,
            aggregated,
            mhc_manager,
            self.mhc_checkpoint_input_layernorm,
            self.recompute_input_layernorm,
        )

        attn_out = self.self_attn.forward_dspark(
            kwargs["dspark_context"],
            input_layernorm_output,
            anchor_positions=kwargs["dspark_anchor_positions"],
            block_keep_mask=kwargs["dspark_block_keep_mask"],
            doc_start_per_pos=kwargs["dspark_doc_start_per_pos"],
            context_position_ids=kwargs["dspark_context_position_ids"],
            draft_position_ids=kwargs["dspark_draft_position_ids"],
            block_size=kwargs["dspark_block_size"],
            chunk_seam=kwargs.get("dspark_chunk_seam"),
        )
        attention_output_with_bias = (attn_out, None)

        hidden_states, bda_recompute = self._fused_h_res_h_post_bda(
            self.self_attention_hyper_connection,
            h_res,
            original_residual,
            h_post,
            attention_output_with_bias,
            mhc_recompute_on,
            manager=mhc_manager,
        )
        if self._attn_mhc_recompute is not None:
            self._attn_mhc_recompute.discard_output_and_register_recompute(
                hidden_states
            )
            self._attn_mhc_recompute = None
        hidden_states = self._cast_and_discard_fused_bda(
            hidden_states, ori_dtype, bda_recompute
        )
        if is_first_fwd:
            hidden_states.stop_gradient = False
        return hidden_states, context
