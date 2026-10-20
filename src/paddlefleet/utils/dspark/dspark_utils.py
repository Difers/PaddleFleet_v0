# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Read-only target tensors and block construction for DSpark."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import paddle
import paddle.nn.functional as F

from paddlefleet.cp_shard import extract_local_cp_chunks

from .anchor_sampler import (
    AnchorBatch,
    _positive_int,
    sample_anchor_positions,
)


def detach_and_reduce_hidden_status(hidden, config):
    """Detach a captured hidden state and collapse residual streams."""
    x = hidden.detach()
    n = (
        int(getattr(config, "num_residual_streams", 1))
        if getattr(config, "enable_hyper_connections", False)
        else 1
    )
    hidden_size = int(config.hidden_size)
    if n > 1 and int(x.shape[-1]) == n * hidden_size:
        x = x.reshape([*x.shape[:-1], n, hidden_size]).mean(axis=-2)
    return x


def readonly_embedding(token_ids, weight):
    """Read the current [vocab, hidden] weight without a target gradient."""
    if weight.ndim != 2:
        raise ValueError("embedding weight must have shape [vocab, hidden]")
    return F.embedding(token_ids, weight.detach())


@dataclass(frozen=True)
class ReadOnlyProjection:
    """Full-vocabulary projection, including optional bias and Multimax.

    ``transpose_y`` is explicit even for square heads: True means [V, H],
    False means [H, V]. Detached views preserve the hidden-state gradient.
    Mutating the underlying target weights before backward is unsupported.
    """

    dspark_readonly = True

    weight: Any
    transpose_y: bool
    bias: Any = None
    multimax_ranges: Any = None
    multimax_ts: Any = None

    def __post_init__(self):
        if self.weight.ndim != 2 or not isinstance(self.transpose_y, bool):
            raise ValueError(
                "provide a matrix weight and explicit boolean transpose_y"
            )
        vocab = self.weight.shape[0 if self.transpose_y else 1]
        if self.bias is not None and list(self.bias.shape) != [vocab]:
            raise ValueError("bias must have shape [vocab]")
        if (self.multimax_ranges is None) != (self.multimax_ts is None):
            raise ValueError("Multimax requires both ranges and ts")
        for name in ("multimax_ranges", "multimax_ts"):
            value = getattr(self, name)
            if value is not None and list(value.shape) != [4]:
                raise ValueError(f"{name} must have shape [4]")
        for name in ("weight", "bias", "multimax_ranges", "multimax_ts"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, value.detach())

    def transform_logits(self, logits):
        """Apply the detached target-head bias and shared SegLU transform."""
        if self.bias is not None:
            # Match linear's autocast bias dtype, including streamed callers
            # that project first and call transform_logits directly.
            logits = logits + self.bias.astype(logits.dtype)
        if self.multimax_ranges is None:
            return logits
        # Local import (NOT module-top): paddlefleet.models.gpt.lm_head sits
        # downstream of dspark_utils in the import graph (gpt_model ->
        # parallel_multi_token_prediction -> dspark_utils), so a module-top
        # import here would close a cycle.
        from paddlefleet.models.gpt.lm_head import SegLU

        # Match the native LM-head AMP path: target SegLU parameters are often
        # fp32 while the projected logits are bf16/fp16.
        r = self.multimax_ranges.astype(logits.dtype)
        t = self.multimax_ts.astype(logits.dtype)
        return SegLU(logits, r, t)

    def __call__(self, hidden_states):
        return self.transform_logits(
            paddle.matmul(
                hidden_states, self.weight, transpose_y=self.transpose_y
            )
        )


# Tensor contracts


@dataclass
class DrafterOutput:
    """Minimal output consumed by the Core loss engine."""

    draft_logits: Any | None
    target_ids: Any
    eval_mask: Any
    block_keep_mask: Any
    confidence_pred: Any | None = None
    aligned_target_logits: Any | None = None
    draft_hidden_states: Any | None = None
    aligned_target_hidden_states: Any | None = None
    previous_token_ids: Any | None = None
    logits_projection: Any | None = None
    markov_head: Any | None = None
    candidate_selector: Any | None = None


# Compatibility for loss callers and checkpoints written before the naming was
# normalized to ``drafter``.
DraftOutput = DrafterOutput


@dataclass
class LossTerms:
    """Loss scalar plus named terms and token counts for logging/reduction."""

    loss: Any
    terms: Mapping[str, Any] = field(default_factory=dict)
    # Keep this as a detached device scalar.  Converting it to a Python int in
    # the loss hot path forces a host-device synchronization; callers that need
    # a host value can explicitly call ``int(valid_tokens)`` at a logging or
    # test boundary.
    valid_tokens: Any = 0


# Host block construction


@dataclass(frozen=True)
class DraftBlocks:
    """DraftBlocks."""

    anchors: AnchorBatch
    noise_ids: Any  # [B, A, K]
    target_ids: Any  # [B, A, K], invalid entries replaced by zero
    eval_mask: Any  # [B, A, K], valid prefixes only
    target_indices: Any  # [B, A, K], safe indices into host final hidden
    previous_token_ids: Any  # [B, A, K]
    position_ids: Any  # [B, A, K], host anchor RoPE position + 1..K


def gather_sequence(values, indices):
    """Gather [B, S, ...] at [B, ...] indices without mixing batch rows."""
    if (
        values.ndim < 2
        or indices.ndim < 2
        or values.shape[0] != indices.shape[0]
    ):
        raise ValueError("values and indices must share a batch dimension")
    with paddle.device.device_guard(values.place):
        rows = (
            paddle.arange(values.shape[0], dtype="int64")
            .reshape([values.shape[0]] + [1] * (indices.ndim - 1))
            .expand(indices.shape)
        )
    return paddle.gather_nd(
        values, paddle.stack([rows, indices.astype("int64")], axis=-1)
    )


def _doc_start_per_pos_from_startend(startend_row_indices, seqlen):
    """Per-batch-row forward-fill of document start positions.

    Vectorized, batch>1 version of ``csa_attention._derive_csa_doc_boundaries``.

    ``startend_row_indices``: ``[B, 1, S, 1]`` (CSA convention) or ``[B, S]``;
    entry ``t`` holds the (exclusive) end row index of token ``t``'s document,
    in GLOBAL sequence coordinates (``S`` here is the global length, before
    any CP-local slicing). Returns ``[B, S]`` int64: each position's own
    document start, also in global coordinates.
    """
    mask = startend_row_indices.reshape(
        [startend_row_indices.shape[0], -1]
    ).astype("int64")
    batch_size, seqlen_check = mask.shape
    if seqlen_check != seqlen:
        raise ValueError(
            f"startend_row_indices seqlen {seqlen_check} != expected {seqlen}"
        )
    positions = paddle.arange(seqlen, dtype="int64").reshape([1, seqlen])
    is_boundary = paddle.concat(
        [
            paddle.ones([batch_size, 1], dtype="bool"),
            (positions[:, 1:] == mask[:, :-1]) & (mask[:, 1:] != mask[:, :-1]),
        ],
        axis=1,
    )
    boundary_positions = paddle.where(
        is_boundary,
        positions.expand([batch_size, seqlen]),
        paddle.zeros_like(mask),
    )
    # Forward-fill via cumulative max: boundary positions increase along the
    # row, so at any position the most recent boundary's position is exactly
    # the running max of "boundary ? position : 0" seen so far -- no gather
    # by document index needed (mirrors the single-row cumsum+gather trick in
    # csa_attention.py, generalized per batch row via cummax directly).
    return paddle.cummax(boundary_positions, axis=1)[0]


def get_doc_start_per_pos(
    input_ids,
    position_ids,
    *,
    attn_mask_startend_row_indices=None,
    cp_rank=0,
    cp_size=1,
    cp_balance_mode=None,
):
    """Per-token document START position, CSA-aligned naming and semantics.

    Mirrors ``csa_attention.py``'s ``doc_start_per_pos``: every position in the
    same document carries the SAME value -- that document's first position -- so
    callers only ever test equality, never compare magnitudes.

    Signal priority: ``attn_mask_startend_row_indices`` > ``position_ids``
    rollover (weak: a non-increasing position marks a new document) > single
    document.

    The mask is built from the GLOBAL sequence and is NOT CP-sliced, while
    ``input_ids``/``position_ids`` must be LOCAL. The start is derived in GLOBAL
    coordinates and then sliced with the layout the rest of the model scatters
    with; assuming a contiguous window silently misaligns every rank but 0 under
    the default zigzag layout, surfacing as cross-document attention leakage
    rather than a crash, so ``cp_balance_mode`` is required when
    ``cp_size > 1``. Whether the caller already sliced is a per-configuration
    contract (:func:`~paddlefleet.cp_shard.cp_shard_source`) this function
    cannot see, hence the check below.

    Returned values stay in GLOBAL coordinates: a global start is
    document-invariant even when a document spans this rank's two zigzag
    chunks, whereas subtracting an offset and clamping would merge two distinct
    documents.
    """
    if attn_mask_startend_row_indices is not None:
        sq = int(input_ids.shape[1])
        seqlen_global = sq * int(cp_size)
        if int(cp_size) > 1:
            mask_len = int(
                attn_mask_startend_row_indices.reshape(
                    [attn_mask_startend_row_indices.shape[0], -1]
                ).shape[1]
            )
            if mask_len == sq:
                raise ValueError(
                    f"input_ids is {sq} long, the same as "
                    "attn_mask_startend_row_indices, but cp_size is "
                    f"{int(cp_size)}: input_ids is in GLOBAL coordinates "
                    "while the mask is global too. Slice input_ids (and "
                    "labels) to this rank's local CP window before calling "
                    "get_doc_start_per_pos."
                )
        doc_start_global = _doc_start_per_pos_from_startend(
            attn_mask_startend_row_indices, seqlen_global
        )
        return extract_local_cp_chunks(
            doc_start_global,
            int(cp_rank),
            int(cp_size),
            axis=1,
            mode=cp_balance_mode,
        )
    if position_ids is not None:
        starts = position_ids[:, 1:] <= position_ids[:, :-1]
    else:
        return paddle.zeros_like(input_ids, dtype="int64")
    seg_id = paddle.cumsum(
        paddle.concat(
            [paddle.zeros_like(input_ids[:, :1]), starts.astype("int64")],
            axis=1,
        ),
        axis=1,
    )
    # Convert the incrementing segment id into an actual start POSITION, to
    # match the attn_mask_startend_row_indices branch's return semantics
    # (both must return the real document-start index, not an arbitrary id,
    # since downstream code only relies on "equal value <=> same document"
    # but a uniform contract is far easier to reason about and test).
    positions = paddle.arange(input_ids.shape[1], dtype="int64").reshape(
        [1, -1]
    )
    is_start = paddle.concat(
        [
            paddle.ones_like(input_ids[:, :1], dtype="bool"),
            starts,
        ],
        axis=1,
    )
    start_positions = paddle.where(
        is_start, positions.expand_as(input_ids), paddle.zeros_like(input_ids)
    )
    return paddle.cummax(start_positions, axis=1)[0]


def build_draft_blocks(
    input_ids,
    labels,
    loss_mask=None,
    *,
    block_size,
    mask_token_id,
    anchors=None,
    num_anchors=None,
    sample_ids=None,
    global_step=0,
    training_seed=0,
    input_mode="anchor_mask",
    position_ids=None,
    ignored_index=-100,
    sample_left_margin=0,
    sample_right_margin=0,
    chunk_seam=None,
):
    """Build draft inputs, targets, masks, and teacher indices."""
    _positive_int("block_size", block_size)
    if (
        not isinstance(mask_token_id, int)
        or isinstance(mask_token_id, bool)
        or mask_token_id < 0
    ):
        raise ValueError("mask_token_id must be a non-negative integer")
    if input_mode not in {"anchor_mask", "sliding_anchor"}:
        raise ValueError("input_mode must be anchor_mask or sliding_anchor")
    if (
        input_ids.ndim != 2
        or min(input_ids.shape) <= 0
        or list(labels.shape) != list(input_ids.shape)
    ):
        raise ValueError("input_ids and labels must have nonempty shape [B, S]")
    for name, value in (
        ("loss_mask", loss_mask),
        ("position_ids", position_ids),
    ):
        if value is not None and list(value.shape) != list(input_ids.shape):
            raise ValueError(f"{name} shape must match input_ids")
    if str(input_ids.dtype).split(".")[-1] not in {"int32", "int64"} or str(
        labels.dtype
    ).split(".")[-1] not in {
        "int32",
        "int64",
    }:
        raise ValueError("input_ids and labels must have integer dtype")
    with paddle.device.device_guard(input_ids.place):
        input_ids, labels = input_ids.astype("int64"), labels.astype("int64")
        valid = labels != ignored_index
        if loss_mask is not None:
            valid = valid & loss_mask.astype("bool")
        if bool(((labels < 0) & valid).any()):
            raise ValueError("valid labels must be non-negative")
        # Keep sampled context and target blocks inside the local chunk.
        lm, rm = int(sample_left_margin), int(sample_right_margin)
        if lm < 0 or rm < 0:
            raise ValueError("sample margins must be non-negative")
        if lm or rm:
            seq = int(input_ids.shape[1])
            pos = paddle.arange(seq, dtype="int64").reshape([1, seq])
            interior = (pos >= lm) & (pos < seq - rm)
            valid = valid & interior
        if anchors is None:
            if sample_ids is None:
                raise ValueError(
                    "sample_ids are required when sampling anchors"
                )
            anchors = sample_anchor_positions(
                loss_mask=valid,
                num_anchors=num_anchors,
                sample_ids=sample_ids,
                global_step=global_step,
                training_seed=training_seed,
            )
        positions = anchors.positions
        keep = anchors.keep_mask
        if not isinstance(positions, paddle.Tensor):
            positions = paddle.to_tensor(positions, place=input_ids.place)
        if not isinstance(keep, paddle.Tensor):
            keep = paddle.to_tensor(keep, place=input_ids.place)
        keep = keep.astype("bool")
        if (
            positions.ndim != 2
            or positions.shape[0] != input_ids.shape[0]
            or positions.shape[1] == 0
            or list(keep.shape) != list(positions.shape)
        ):
            raise ValueError("anchors must have nonempty shape [B, A]")
        if str(positions.dtype).split(".")[-1] not in {"int32", "int64"}:
            raise ValueError("anchor positions must have integer dtype")
        if bool(
            (((positions < 0) | (positions >= labels.shape[1])) & keep).any()
        ):
            raise ValueError(
                "kept anchor positions must be inside the sequence"
            )
        positions = paddle.where(
            keep,
            positions.astype("int64"),
            paddle.zeros_like(positions).astype("int64"),
        )
        keep = keep & gather_sequence(valid, positions)
        # Preserve the sampler's cycling diagnostic across the validation
        # rebuild; a caller-supplied batch may predate the field.
        anchors = AnchorBatch(
            positions, keep, getattr(anchors, "cycled_rows", 0)
        )
        offsets = paddle.arange(1, block_size + 1, dtype="int64").reshape(
            [1, 1, -1]
        )
        indices = positions.unsqueeze(-1) + offsets
        safe = paddle.clip(indices, 0, labels.shape[1] - 1)
        safe = paddle.where(keep.unsqueeze(-1), safe, paddle.zeros_like(safe))
        targets = gather_sequence(labels, safe)
        segments = get_doc_start_per_pos(input_ids, position_ids)
        label_segments = paddle.concat(
            [segments[:, 1:], segments[:, -1:]], axis=1
        )
        anchor_segment = gather_sequence(label_segments, positions).unsqueeze(
            -1
        )
        evaluation = (indices < labels.shape[1]) & keep.unsqueeze(-1)
        evaluation = evaluation & gather_sequence(valid, safe)
        evaluation = evaluation & (
            gather_sequence(label_segments, safe) == anchor_segment
        )
        if chunk_seam is not None:
            # Mirror of the window's seam guard (build_dspark_topk_idxs): a
            # target block straddling the zigzag seam would be supervised
            # against tokens from the rank's other, globally distant interval.
            # The block looks FORWARD from the anchor, so "same chunk" is "the
            # anchor is right of the seam, or the target is left of it".
            evaluation = evaluation & (
                (positions.unsqueeze(-1) >= chunk_seam) | (indices < chunk_seam)
            )
        evaluation = paddle.cumprod(evaluation.astype("int32"), dim=-1).astype(
            "bool"
        )
        targets = paddle.where(evaluation, targets, paddle.zeros_like(targets))
        first = gather_sequence(labels, positions)
        first = paddle.where(
            keep, first, paddle.full_like(first, mask_token_id)
        )
        if input_mode == "anchor_mask":
            noise = paddle.concat(
                [
                    first.unsqueeze(-1),
                    paddle.full(
                        [*positions.shape, block_size - 1],
                        mask_token_id,
                        dtype="int64",
                    ),
                ],
                axis=-1,
            )
        else:
            history = positions.unsqueeze(-1) + paddle.arange(
                2 - block_size, 1, dtype="int64"
            ).reshape([1, 1, -1])
            safe_history = paddle.clip(history, 0, labels.shape[1] - 1)
            history_valid = (history >= 0) & keep.unsqueeze(-1)
            history_valid = history_valid & (
                gather_sequence(segments, safe_history) == anchor_segment
            )
            if chunk_seam is not None:
                history_valid = history_valid & (
                    (positions.unsqueeze(-1) < chunk_seam)
                    | (history >= chunk_seam)
                )
            tokens = gather_sequence(input_ids, safe_history)
            tokens = paddle.where(
                history_valid, tokens, paddle.full_like(tokens, mask_token_id)
            )
            noise = paddle.concat([tokens, first.unsqueeze(-1)], axis=-1)
        previous = paddle.concat(
            [first.unsqueeze(-1), targets[:, :, :-1]], axis=-1
        )
        if position_ids is None:
            position_ids = (
                paddle.arange(labels.shape[1], dtype="int64")
                .unsqueeze(0)
                .expand(labels.shape)
            )
        draft_positions = (
            gather_sequence(position_ids, positions).unsqueeze(-1) + offsets
        )
        return DraftBlocks(
            anchors, noise, targets, evaluation, safe, previous, draft_positions
        )


def gather_target_hidden(hidden_states, blocks):
    """Detached teacher features at the prediction indices of ``blocks``."""
    return gather_sequence(hidden_states.detach(), blocks.target_indices)
