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

"""Composable DSpark loss terms shared by model adapters."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import paddle
from paddle.amp.auto_cast import amp_state
from paddle.autograd import PyLayer

from .dspark_utils import DrafterOutput, LossTerms


@dataclass(frozen=True)
class LossProfile:
    """Explicit coefficients and weighting policy for one training recipe.

    Two-level weighting: the ``*_alpha`` fields weight the sub-terms *inside*
    the dspark loss; the config's ``dspark_loss_scaling_factor`` separately
    weights the whole dspark loss against the target loss.

    ``probability_chunk_tokens`` / ``subbatch_sequence_length`` are **memory
    tiling knobs** (vocab ~=201k, so full-vocab logits for A*K tokens are tiled
    to bound peak activation); they do not change the result, only peak memory.
    """

    ce_alpha: float = 0.1
    probability_l1_alpha: float = 0.9
    confidence_alpha: float = 1.0
    weight_type: str = "decay"
    decay_gamma: float | None = 4.0
    dpace_alpha: float = 0.5  # only used when weight_type == "dpace"
    probability_chunk_tokens: int = 1024  # memory-tiling, not a recipe choice
    subbatch_sequence_length: int = 1024  # memory-tiling, <=0 disables tiling
    # The selector_* / use_fused_distill_loss fields below are inert unless the
    # candidate_selector refiner / Triton fused kernel are enabled (both off by
    # default; the default refiner is MarkovHead).
    selector_loss_alpha: float = 0.0
    selector_stop_gradient: bool = False
    selector_warmup_ratio: float = 0.0
    selector_ramp_ratio: float = 0.0
    use_fused_distill_loss: bool = False

    def validate(self) -> LossProfile:
        """Validate fields and return self."""
        if self.weight_type not in {"decay", "dpace"}:
            raise ValueError("weight_type must be 'decay' or 'dpace'")
        if not 0.0 <= float(self.dpace_alpha) <= 1.0:
            raise ValueError("dpace_alpha must be in [0, 1]")
        if (
            not isinstance(self.probability_chunk_tokens, Integral)
            or isinstance(self.probability_chunk_tokens, bool)
            or self.probability_chunk_tokens <= 0
        ):
            raise ValueError("probability_chunk_tokens must be positive")
        if any(
            not math.isfinite(float(value)) or float(value) < 0
            for value in (
                self.ce_alpha,
                self.probability_l1_alpha,
                self.confidence_alpha,
                self.selector_loss_alpha,
            )
        ):
            raise ValueError(
                "loss coefficients must be finite and non-negative"
            )
        if not isinstance(self.selector_stop_gradient, bool):
            raise ValueError("selector_stop_gradient must be a boolean")
        if not isinstance(self.use_fused_distill_loss, bool):
            raise ValueError("use_fused_distill_loss must be a boolean")
        if not 0.0 <= float(self.selector_warmup_ratio) <= 1.0:
            raise ValueError("selector_warmup_ratio must be in [0, 1]")
        if not 0.0 <= float(self.selector_ramp_ratio) <= 1.0:
            raise ValueError("selector_ramp_ratio must be in [0, 1]")
        if self.decay_gamma is not None and (
            not math.isfinite(float(self.decay_gamma)) or self.decay_gamma <= 0
        ):
            raise ValueError("decay_gamma must be None or finite and positive")
        if not isinstance(
            self.subbatch_sequence_length, Integral
        ) or isinstance(self.subbatch_sequence_length, bool):
            raise ValueError(
                "subbatch_sequence_length must be an integer (<= 0 disables tiling)"
            )
        return self


def effective_selector_alpha(
    profile: LossProfile, *, global_step: int, horizon: int
) -> float:
    """Resolve the selector alpha on optimizer-step coordinates.

    Apply warmup and linear ramping to the configured selector weight.
    """
    configured = float(profile.selector_loss_alpha)
    if configured <= 0.0 or int(horizon) <= 0:
        return configured
    warmup_steps = int(horizon * float(profile.selector_warmup_ratio))
    ramp_steps = int(horizon * float(profile.selector_ramp_ratio))
    step = max(0, int(global_step))
    if step < warmup_steps:
        return 0.0
    if ramp_steps > 0 and step < warmup_steps + ramp_steps:
        progress = float(step - warmup_steps) / float(ramp_steps)
        return configured * min(max(progress, 0.0), 1.0)
    return configured


def _probability_l1(
    draft_logits: Any, target_logits: Any, chunk_tokens: int
) -> Any:
    shape = draft_logits.shape[:-1]
    vocab_size = int(draft_logits.shape[-1])
    draft_flat = draft_logits.reshape([-1, vocab_size])
    target_flat = target_logits.reshape([-1, vocab_size])
    values = []
    for start in range(0, int(draft_flat.shape[0]), int(chunk_tokens)):
        end = start + int(chunk_tokens)
        draft_prob = paddle.nn.functional.softmax(
            draft_flat[start:end].astype("float32"), axis=-1
        )
        target_prob = paddle.nn.functional.softmax(
            target_flat[start:end].astype("float32"), axis=-1
        )
        values.append(paddle.abs(draft_prob - target_prob).sum(axis=-1))
    return paddle.concat(values, axis=0).reshape(shape)


def _dpace_weights(
    target_probability: Any, eval_mask: Any, alpha: float
) -> Any:
    valid = eval_mask.astype("bool")
    smooth = (1.0 - float(alpha)) * target_probability.detach() + float(alpha)
    smooth = paddle.where(valid, smooth, paddle.ones_like(smooth))
    prefix = paddle.cumprod(smooth, dim=-1)
    return paddle.flip(
        paddle.cumsum(
            paddle.flip(prefix * eval_mask.astype("float32"), dims=[-1]), dim=-1
        ),
        dims=[-1],
    )


def _loss_ranges(
    sequence_length: int, subbatch_length: int
) -> list[tuple[int, int]]:
    if int(sequence_length) <= 0:
        raise ValueError("DSpark loss sequence length must be positive")
    if int(subbatch_length) <= 0 or int(subbatch_length) >= int(
        sequence_length
    ):
        return [(0, int(sequence_length))]
    return [
        (start, min(start + int(subbatch_length), int(sequence_length)))
        for start in range(0, int(sequence_length), int(subbatch_length))
    ]


def _materialized_token_losses(
    draft_logits: Any,
    target_ids: Any,
    teacher: Any | None,
    *,
    needs_teacher: bool,
    probability_chunk_tokens: int,
    subbatch_sequence_length: int,
) -> tuple[Any, Any | None]:
    output_shape = target_ids.shape
    batch_size = int(target_ids.shape[0])
    sequence_length = int(target_ids.numel()) // batch_size
    vocab_size = int(draft_logits.shape[-1])
    draft_sequence = draft_logits.reshape(
        [batch_size, sequence_length, vocab_size]
    )
    target_sequence = target_ids.reshape([batch_size, sequence_length])
    teacher_sequence = (
        None
        if teacher is None
        else teacher.reshape([batch_size, sequence_length, vocab_size])
    )
    ce_parts = []
    l1_parts = []
    for start, end in _loss_ranges(sequence_length, subbatch_sequence_length):
        width = end - start
        draft_part = draft_sequence[:, start:end]
        ids_part = target_sequence[:, start:end]
        ce_parts.append(
            paddle.nn.functional.cross_entropy(
                draft_part.astype("float32").reshape([-1, vocab_size]),
                ids_part.reshape([-1]),
                reduction="none",
            ).reshape([batch_size, width])
        )
        if needs_teacher:
            l1_parts.append(
                _probability_l1(
                    draft_part,
                    teacher_sequence[:, start:end],
                    probability_chunk_tokens,
                ).reshape([batch_size, width])
            )
    ce_per_token = paddle.concat(ce_parts, axis=1).reshape(output_shape)
    l1_per_token = (
        paddle.concat(l1_parts, axis=1).reshape(output_shape)
        if l1_parts
        else None
    )
    return ce_per_token, l1_per_token


def _validate_streamed_output(output: DrafterOutput) -> None:
    required = {
        "draft_hidden_states": output.draft_hidden_states,
        "aligned_target_hidden_states": output.aligned_target_hidden_states,
        "logits_projection": output.logits_projection,
    }
    if output.markov_head is not None or output.candidate_selector is not None:
        required["previous_token_ids"] = output.previous_token_ids
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(
            "streamed DSpark logits are missing required fields: "
            + ", ".join(missing)
        )


def _fused_distill_available(reference_tensor: Any) -> bool:
    """True when the Triton fused CE/L1 kernel can run for this tensor.

    Requires Triton importable, a CUDA-compiled Paddle, and the operand living
    on a GPU place. Any miss falls back to the eager streamed path so CPU, no
    Triton, and XPU runs keep working unchanged.
    """
    try:
        import triton  # noqa: F401
    except Exception:
        return False
    if not paddle.device.is_compiled_with_cuda():
        return False
    place = getattr(reference_tensor, "place", None)
    return place is not None and bool(reference_tensor.place.is_gpu_place())


def _streamed_token_losses(
    output: DrafterOutput,
    target_ids: Any,
    *,
    chunk_tokens: int,
    subbatch_sequence_length: int,
    needs_teacher: bool,
    selector_enabled: bool = False,
    selector_stop_gradient: bool = False,
    use_fused: bool = False,
) -> tuple[Any, Any | None, Any, Any, Any]:
    """Project and recompute token chunks without retaining full-vocab logits."""
    _validate_streamed_output(output)
    output_shape = target_ids.shape
    batch_size = int(target_ids.shape[0])
    sequence_length = int(target_ids.numel()) // batch_size
    draft_hidden = output.draft_hidden_states.reshape(
        [batch_size, sequence_length, output.draft_hidden_states.shape[-1]]
    )
    target_hidden = output.aligned_target_hidden_states.detach().reshape(
        [
            batch_size,
            sequence_length,
            output.aligned_target_hidden_states.shape[-1],
        ]
    )
    target_sequence = target_ids.reshape([batch_size, sequence_length])
    previous_sequence = (
        None
        if output.previous_token_ids is None
        else output.previous_token_ids.reshape([batch_size, sequence_length])
    )
    projection_weight = getattr(output.logits_projection, "weight", None)
    if projection_weight is None:
        raise ValueError("streamed DSpark logits require a projection weight")
    # The target LM head is frozen. Detaching it prevents Paddle from allocating
    # a full-vocabulary weight-gradient buffer during the outer backward.
    projection_weight = projection_weight.detach()
    projection_weight.stop_gradient = True
    transpose_y = getattr(output.logits_projection, "transpose_y", None)
    if not isinstance(transpose_y, bool):
        raise ValueError(
            "streamed projection requires explicit transpose_y; use ReadOnlyProjection"
        )
    transform_logits = getattr(
        output.logits_projection, "transform_logits", None
    )
    # Custom bound transforms can capture live target parameters. Only the
    # functional read-only protocol is accepted; never freeze a target module.
    if transform_logits is not None and not getattr(
        output.logits_projection, "dspark_readonly", False
    ):
        raise ValueError(
            "custom transforms require ReadOnlyProjection (including detached bias/Multimax)"
        )
    expected_hidden = int(projection_weight.shape[1 if transpose_y else 0])
    if (
        draft_hidden.shape[-1] != expected_hidden
        or target_hidden.shape[-1] != expected_hidden
    ):
        raise ValueError(
            "projection hidden dimension does not match draft/teacher hidden"
        )
    markov_w1 = (
        None
        if output.markov_head is None
        else output.markov_head.markov_w1.weight
    )
    markov_w2 = (
        None
        if output.markov_head is None
        else output.markov_head.markov_w2.weight
    )
    candidate_selector = output.candidate_selector
    if selector_enabled and candidate_selector is None:
        raise ValueError(
            "selector objective requires output.candidate_selector"
        )
    selector_predecessor = (
        None
        if candidate_selector is None
        else getattr(candidate_selector, "predecessor_codebook", None)
    )
    selector_successor = (
        None
        if candidate_selector is None
        else getattr(candidate_selector, "successor_codebook", None)
    )
    selector_projection = (
        None
        if candidate_selector is None
        else getattr(
            getattr(candidate_selector, "hidden_projection", None),
            "weight",
            None,
        )
    )
    if selector_enabled and any(
        value is None
        for value in (
            selector_predecessor,
            selector_successor,
            selector_projection,
        )
    ):
        raise ValueError(
            "candidate_selector must expose predecessor_codebook, "
            "successor_codebook, and hidden_projection.weight"
        )
    selector_top_k = (
        0 if candidate_selector is None else int(candidate_selector.top_k)
    )
    markov_enabled = output.markov_head is not None

    def calculate(
        draft,
        teacher_hidden,
        ids,
        previous,
        projection,
        w1,
        w2,
        predecessor_codebook,
        successor_codebook,
        selector_projection_weight,
    ):
        unary_logits = paddle.matmul(draft, projection, transpose_y=transpose_y)
        with paddle.no_grad():
            target_logits = paddle.matmul(
                teacher_hidden, projection, transpose_y=transpose_y
            )
        draft_logits = unary_logits
        if markov_enabled:
            previous_embeddings = paddle.nn.functional.embedding(previous, w1)
            draft_logits = draft_logits + paddle.matmul(previous_embeddings, w2)
        if transform_logits is not None:
            draft_logits = transform_logits(draft_logits)
            target_logits = transform_logits(target_logits)
        draft_logits = draft_logits.astype("float32")
        ce_per_token = paddle.nn.functional.cross_entropy(
            draft_logits, ids, reduction="none"
        )
        if needs_teacher:
            draft_probs = paddle.nn.functional.softmax(draft_logits, axis=-1)
            target_probs = paddle.nn.functional.softmax(
                target_logits.detach().astype("float32"), axis=-1
            )
            l1_per_token = paddle.abs(draft_probs - target_probs).sum(axis=-1)
        else:
            l1_per_token = paddle.zeros_like(ce_per_token)

        selector_ce = paddle.zeros_like(ce_per_token)
        selector_covered = paddle.zeros_like(ce_per_token)
        selector_correct = paddle.zeros_like(ce_per_token)
        if selector_enabled:
            # Candidate selector is mutually exclusive with the Markov head,
            # so its unary scores are exactly the already transformed draft
            # logits.  Reuse them instead of running the projection transform
            # a second time in every streamed chunk.
            selector_unary = (
                draft_logits if not markov_enabled else unary_logits
            )
            selector_hidden = draft
            if transform_logits is not None and markov_enabled:
                selector_unary = transform_logits(selector_unary)
            if selector_stop_gradient:
                selector_unary = selector_unary.detach()
                selector_hidden = selector_hidden.detach()
            selector_unary = selector_unary.astype("float32")
            topk_logits, candidate_ids = paddle.topk(
                selector_unary, selector_top_k, axis=-1
            )
            matches = candidate_ids == ids.unsqueeze(-1)
            target_index = matches.astype("int64").argmax(axis=-1)
            predecessor = paddle.nn.functional.embedding(
                previous, predecessor_codebook
            )
            successor = paddle.nn.functional.embedding(
                candidate_ids.astype("int64"), successor_codebook
            )
            projected_hidden = paddle.matmul(
                selector_hidden, selector_projection_weight
            )
            selector_context = predecessor * projected_hidden
            selector_logits = topk_logits + paddle.einsum(
                "...r,...kr->...k", selector_context, successor
            )
            selector_ce = paddle.nn.functional.cross_entropy(
                selector_logits,
                target_index,
                reduction="none",
            )
            selector_covered = matches.any(axis=-1).astype("float32")
            selected = paddle.argmax(selector_logits, axis=-1)
            selected_ids = paddle.take_along_axis(
                candidate_ids, selected.unsqueeze(-1), axis=1
            ).squeeze(-1)
            selector_correct = (selected_ids == ids).astype(
                "float32"
            ) * selector_covered
        return (
            ce_per_token,
            l1_per_token,
            selector_ce,
            selector_covered,
            selector_correct,
        )

    class _ChunkedTokenLoss(PyLayer):
        @staticmethod
        def forward(
            ctx,
            draft,
            teacher_hidden,
            ids,
            previous,
            projection,
            w1,
            w2,
            predecessor_codebook,
            successor_codebook,
            selector_projection_weight,
        ):
            """Forward pass."""
            # Backward usually runs outside the trainer's autocast context.
            # Recompute the same projection precision, not an fp32 variant.
            state = amp_state()
            ctx.amp_kwargs = (
                dict(state) if state is not None else {"enable": False}
            )
            ctx.save_for_backward(
                draft,
                teacher_hidden,
                ids,
                previous,
                projection,
                w1,
                w2,
                predecessor_codebook,
                successor_codebook,
                selector_projection_weight,
            )
            with paddle.no_grad():
                return calculate(
                    draft,
                    teacher_hidden,
                    ids,
                    previous,
                    projection,
                    w1,
                    w2,
                    predecessor_codebook,
                    successor_codebook,
                    selector_projection_weight,
                )

        @staticmethod
        def backward(
            ctx, grad_ce, grad_l1, grad_selector, grad_covered, grad_correct
        ):
            """Backward pass."""
            del grad_covered, grad_correct
            (
                draft,
                teacher_hidden,
                ids,
                previous,
                projection,
                w1,
                w2,
                predecessor_codebook,
                successor_codebook,
                selector_projection_weight,
            ) = ctx.saved_tensor()
            differentiable = []
            positions = []
            for position, tensor in (
                (0, draft),
                (4, projection),
                (5, w1),
                (6, w2),
                (7, predecessor_codebook),
                (8, successor_codebook),
                (9, selector_projection_weight),
            ):
                if not tensor.stop_gradient:
                    clone = tensor.detach()
                    clone.stop_gradient = False
                    differentiable.append(clone)
                    positions.append(position)
            args = [
                draft,
                teacher_hidden,
                ids,
                previous,
                projection,
                w1,
                w2,
                predecessor_codebook,
                successor_codebook,
                selector_projection_weight,
            ]
            for position, clone in zip(positions, differentiable):
                args[position] = clone
            with paddle.enable_grad(), paddle.amp.auto_cast(**ctx.amp_kwargs):
                ce_value, l1_value, selector_value, _, _ = calculate(*args)
            grad_ce = (
                paddle.zeros_like(ce_value) if grad_ce is None else grad_ce
            )
            grad_l1 = (
                paddle.zeros_like(l1_value) if grad_l1 is None else grad_l1
            )
            grad_selector = (
                paddle.zeros_like(selector_value)
                if grad_selector is None
                else grad_selector
            )
            values = [ce_value]
            grad_outputs = [grad_ce]
            if needs_teacher:
                values.append(l1_value)
                grad_outputs.append(grad_l1)
            if selector_enabled:
                values.append(selector_value)
                grad_outputs.append(grad_selector)
            gradients = paddle.grad(
                values,
                differentiable,
                grad_outputs=grad_outputs,
                allow_unused=True,
            )
            result = [None] * 10
            for position, gradient in zip(positions, gradients):
                result[position] = gradient
            return tuple(result)

    if markov_w1 is None:
        dtype = str(projection_weight.dtype).split(".")[-1]
        markov_w1 = paddle.empty([0], dtype=dtype)
        markov_w2 = paddle.empty([0], dtype=dtype)
        markov_w1.stop_gradient = True
        markov_w2.stop_gradient = True
    if selector_predecessor is None:
        dtype = str(projection_weight.dtype).split(".")[-1]
        selector_predecessor = paddle.empty([0], dtype=dtype)
        selector_successor = paddle.empty([0], dtype=dtype)
        selector_projection = paddle.empty([0], dtype=dtype)
        selector_predecessor.stop_gradient = True
        selector_successor.stop_gradient = True
        selector_projection.stop_gradient = True

    # Opt-in Triton fast path: fuse projection + SegLU + 2x softmax + CE + L1
    # into two streaming kernels (per-token ce/l1 only, closed-form backward).
    # Only valid without the selector objective (the kernel emits zeros for it),
    # with an identity/SegLU ReadOnlyProjection and no head bias; otherwise the
    # eager _ChunkedTokenLoss below runs and the numerics are unchanged.
    fused_layer = None
    if use_fused and not selector_enabled:
        projection = output.logits_projection
        # Read the detached transform parameters directly from the projection
        # descriptor.  ReadOnlyProjection applies the same shared SegLU in the
        # eager path; the fused kernel receives its detached parameters here.
        multimax_ranges = getattr(projection, "multimax_ranges", None)
        multimax_ts = getattr(projection, "multimax_ts", None)
        projection_bias = getattr(projection, "bias", None)
        transform_is_fusable = getattr(projection, "dspark_readonly", False)
        if (
            transform_is_fusable
            and projection_bias is None
            and _fused_distill_available(draft_hidden)
        ):
            from .dspark_fused_distill_loss import (
                make_fused_ce_l1,
            )

            fused_layer = make_fused_ce_l1(
                needs_teacher=needs_teacher,
                markov_enabled=markov_enabled,
                use_seglu=multimax_ranges is not None,
                seglu_r=multimax_ranges,
                seglu_t=multimax_ts,
                transpose_y=transpose_y,
            )

    ce_subbatches = []
    l1_subbatches = []
    selector_subbatches = []
    selector_covered_subbatches = []
    selector_correct_subbatches = []
    for subbatch_start, subbatch_end in _loss_ranges(
        sequence_length, subbatch_sequence_length
    ):
        subbatch_width = subbatch_end - subbatch_start
        draft_part = draft_hidden[:, subbatch_start:subbatch_end].reshape(
            [-1, draft_hidden.shape[-1]]
        )
        target_part = target_hidden[:, subbatch_start:subbatch_end].reshape(
            [-1, target_hidden.shape[-1]]
        )
        ids_part = target_sequence[:, subbatch_start:subbatch_end].reshape([-1])
        previous_part = (
            ids_part
            if previous_sequence is None
            else previous_sequence[:, subbatch_start:subbatch_end].reshape([-1])
        )
        ce_chunks = []
        l1_chunks = []
        selector_chunks = []
        selector_covered_chunks = []
        selector_correct_chunks = []
        for start in range(
            0, int(ids_part.shape[0]), max(1, int(chunk_tokens))
        ):
            end = min(start + max(1, int(chunk_tokens)), int(ids_part.shape[0]))
            if fused_layer is not None:
                (
                    ce_chunk,
                    l1_chunk,
                    selector_chunk,
                    selector_covered_chunk,
                    selector_correct_chunk,
                ) = fused_layer.apply(
                    draft_part[start:end],
                    target_part[start:end],
                    ids_part[start:end],
                    previous_part[start:end],
                    projection_weight,
                    markov_w1,
                    markov_w2,
                )
            else:
                (
                    ce_chunk,
                    l1_chunk,
                    selector_chunk,
                    selector_covered_chunk,
                    selector_correct_chunk,
                ) = _ChunkedTokenLoss.apply(
                    draft_part[start:end],
                    target_part[start:end],
                    ids_part[start:end],
                    previous_part[start:end],
                    projection_weight,
                    markov_w1,
                    markov_w2,
                    selector_predecessor,
                    selector_successor,
                    selector_projection,
                )
            ce_chunks.append(ce_chunk)
            if needs_teacher:
                l1_chunks.append(l1_chunk)
            selector_chunks.append(selector_chunk)
            selector_covered_chunks.append(selector_covered_chunk)
            selector_correct_chunks.append(selector_correct_chunk)
        ce_subbatches.append(
            paddle.concat(ce_chunks, axis=0).reshape(
                [batch_size, subbatch_width]
            )
        )
        if needs_teacher:
            l1_subbatches.append(
                paddle.concat(l1_chunks, axis=0).reshape(
                    [batch_size, subbatch_width]
                )
            )
        selector_subbatches.append(
            paddle.concat(selector_chunks, axis=0).reshape(
                [batch_size, subbatch_width]
            )
        )
        selector_covered_subbatches.append(
            paddle.concat(selector_covered_chunks, axis=0).reshape(
                [batch_size, subbatch_width]
            )
        )
        selector_correct_subbatches.append(
            paddle.concat(selector_correct_chunks, axis=0).reshape(
                [batch_size, subbatch_width]
            )
        )

    ce_per_token = paddle.concat(ce_subbatches, axis=1).reshape(output_shape)
    l1_per_token = (
        paddle.concat(l1_subbatches, axis=1).reshape(output_shape)
        if needs_teacher
        else None
    )
    selector_per_token = paddle.concat(selector_subbatches, axis=1).reshape(
        output_shape
    )
    selector_covered_per_token = paddle.concat(
        selector_covered_subbatches, axis=1
    ).reshape(output_shape)
    selector_correct_per_token = paddle.concat(
        selector_correct_subbatches, axis=1
    ).reshape(output_shape)
    return (
        ce_per_token,
        l1_per_token,
        selector_per_token,
        selector_covered_per_token,
        selector_correct_per_token,
    )


def _materialized_selector_terms(
    output: DrafterOutput,
    draft_logits: Any,
    target_ids: Any,
    *,
    stop_gradient: bool = False,
) -> tuple[Any, Any, Any]:
    """Score a materialized unary top-k set for evaluation-sized batches."""
    selector = output.candidate_selector
    if selector is None:
        zeros = paddle.zeros_like(target_ids, dtype="float32")
        return zeros, zeros, zeros
    if output.draft_hidden_states is None or output.previous_token_ids is None:
        raise ValueError(
            "candidate selector loss requires draft_hidden_states and previous_token_ids"
        )
    if list(output.draft_hidden_states.shape) != [
        *list(target_ids.shape),
        output.draft_hidden_states.shape[-1],
    ]:
        raise ValueError(
            "draft_hidden_states leading shape must match target_ids"
        )
    if list(output.previous_token_ids.shape) != list(target_ids.shape):
        raise ValueError("previous_token_ids shape must match target_ids")
    selector_hidden = output.draft_hidden_states
    if stop_gradient:
        draft_logits = draft_logits.detach()
        selector_hidden = selector_hidden.detach()
    topk_logits, candidate_ids = paddle.topk(
        draft_logits.astype("float32"), int(selector.top_k), axis=-1
    )
    matches = candidate_ids == target_ids.unsqueeze(-1)
    target_index = matches.astype("int64").argmax(axis=-1)
    selector_logits = selector.score_candidates(
        candidate_ids=candidate_ids,
        unary_logits=topk_logits,
        hidden_states=selector_hidden,
        predecessor_ids=output.previous_token_ids,
    )
    selector_ce = paddle.nn.functional.cross_entropy(
        selector_logits,
        target_index,
        reduction="none",
    )
    covered = matches.any(axis=-1).astype("float32")
    selected = paddle.argmax(selector_logits, axis=-1)
    selected_ids = paddle.take_along_axis(
        candidate_ids, selected.unsqueeze(-1), axis=-1
    ).squeeze(-1)
    correct = (selected_ids == target_ids).astype("float32") * covered
    return selector_ce, covered, correct


def _validate_output(output: DrafterOutput) -> None:
    if output.target_ids is None or output.eval_mask is None:
        raise ValueError("DrafterOutput requires target_ids and eval_mask")
    if len(output.target_ids.shape) != 3 or list(
        output.eval_mask.shape
    ) != list(output.target_ids.shape):
        raise ValueError(
            "target_ids and eval_mask must both have shape [B, A, K]"
        )
    if min(output.target_ids.shape) <= 0:
        raise ValueError("target_ids must have nonempty dimensions")
    if str(output.target_ids.dtype).split(".")[-1] not in {"int32", "int64"}:
        raise ValueError("target_ids must have integer dtype")
    if output.block_keep_mask is None:
        raise ValueError("block_keep_mask is required")
    if list(output.block_keep_mask.shape) != list(output.target_ids.shape[:2]):
        raise ValueError("block_keep_mask must have shape [B, A]")
    if output.draft_logits is None:
        _validate_streamed_output(output)


def compute_dspark_loss(
    output,
    profile=None,
    *,
    return_terms=False,
    selector_loss_alpha=None,
    normalization_group=None,
):
    """CE + probability L1 + optional confidence/selector loss on valid tokens.

    Streamed mode requires a functional ReadOnlyProjection over the current
    full-vocabulary target head. It recomputes each vocabulary projection in
    backward. Target hidden/logits and head parameters are always detached;
    draft hidden and refiner parameters remain differentiable. Materialized
    draft logits must already have the appropriate target I/O gradient boundary.
    All reductions are local; the trainer owns distributed normalization.
    """
    if output.target_ids is None:
        raise ValueError("target_ids is required")
    _validate_output(output)
    profile = (profile or LossProfile()).validate()
    effective_selector_alpha = (
        float(profile.selector_loss_alpha)
        if selector_loss_alpha is None
        else float(selector_loss_alpha)
    )
    if (
        not math.isfinite(effective_selector_alpha)
        or effective_selector_alpha < 0
    ):
        raise ValueError("selector_loss_alpha must be non-negative")
    if (
        profile.selector_loss_alpha > 0 or effective_selector_alpha > 0
    ) and output.candidate_selector is None:
        raise ValueError("selector objective requires candidate_selector")
    if output.markov_head is not None and output.candidate_selector is not None:
        raise ValueError(
            "Markov and candidate selector are alternative refiners"
        )
    selector_enabled = output.candidate_selector is not None and (
        float(profile.selector_loss_alpha) > 0.0
        or effective_selector_alpha > 0.0
    )
    # A dropped block is outside the DSpark objective altogether.  Keep its
    # tokens out of both the numerator and denominator so padding or filtered
    # blocks cannot dilute the loss of the blocks that are actually trained.
    eval_mask = output.eval_mask.astype("bool") & output.block_keep_mask.astype(
        "bool"
    ).unsqueeze(-1)
    target_ids = paddle.where(
        eval_mask,
        output.target_ids.astype("int64"),
        paddle.zeros_like(output.target_ids).astype("int64"),
    )
    draft_logits = output.draft_logits
    if draft_logits is not None and list(draft_logits.shape[:-1]) != list(
        target_ids.shape
    ):
        raise ValueError("draft_logits leading shape must match target_ids")
    teacher = output.aligned_target_logits
    needs_teacher = (
        float(profile.probability_l1_alpha) > 0
        or output.confidence_pred is not None
    )
    if needs_teacher and teacher is None and draft_logits is not None:
        raise ValueError(
            "aligned_target_logits are required for L1/confidence loss"
        )
    if (
        teacher is not None
        and draft_logits is not None
        and list(teacher.shape) != list(draft_logits.shape)
    ):
        raise ValueError("aligned_target_logits shape must match draft_logits")

    if draft_logits is not None:
        vocab_size = int(draft_logits.shape[-1])
    else:
        projection = output.logits_projection
        transpose_y = getattr(projection, "transpose_y", None)
        if not isinstance(transpose_y, bool):
            raise ValueError(
                "streamed projection requires explicit transpose_y; use ReadOnlyProjection"
            )
        vocab_size = int(projection.weight.shape[0 if transpose_y else 1])
        for name in ("draft_hidden_states", "aligned_target_hidden_states"):
            value = getattr(output, name)
            if list(value.shape[:-1]) != list(target_ids.shape):
                raise ValueError(f"{name} leading shape must match target_ids")
    if vocab_size <= 0 or bool(
        ((target_ids < 0) | (target_ids >= vocab_size)).any()
    ):
        raise ValueError("valid target_ids must be in [0, vocab_size)")
    if output.previous_token_ids is not None:
        if list(output.previous_token_ids.shape) != list(target_ids.shape):
            raise ValueError("previous_token_ids shape must match target_ids")
    if (
        selector_enabled
        and not 0 < int(output.candidate_selector.top_k) <= vocab_size
    ):
        raise ValueError("selector top_k must be in [1, vocab_size]")

    if draft_logits is None:
        (
            ce_per_token,
            l1_per_token,
            selector_ce_per_token,
            selector_covered_per_token,
            selector_correct_per_token,
        ) = _streamed_token_losses(
            output,
            target_ids,
            chunk_tokens=profile.probability_chunk_tokens,
            subbatch_sequence_length=profile.subbatch_sequence_length,
            needs_teacher=needs_teacher,
            selector_enabled=selector_enabled,
            selector_stop_gradient=profile.selector_stop_gradient,
            use_fused=profile.use_fused_distill_loss,
        )
    else:
        ce_per_token, l1_per_token = _materialized_token_losses(
            draft_logits,
            target_ids,
            None if teacher is None else teacher.detach(),
            needs_teacher=needs_teacher,
            probability_chunk_tokens=profile.probability_chunk_tokens,
            subbatch_sequence_length=profile.subbatch_sequence_length,
        )
        if selector_enabled:
            (
                selector_ce_per_token,
                selector_covered_per_token,
                selector_correct_per_token,
            ) = _materialized_selector_terms(
                output,
                draft_logits,
                target_ids,
                stop_gradient=profile.selector_stop_gradient,
            )
        else:
            selector_ce_per_token = paddle.zeros_like(
                target_ids, dtype="float32"
            )
            selector_covered_per_token = paddle.zeros_like(
                target_ids, dtype="float32"
            )
            selector_correct_per_token = paddle.zeros_like(
                target_ids, dtype="float32"
            )
    weights = eval_mask.astype("float32")
    if (
        profile.weight_type == "decay"
        and profile.decay_gamma is not None
        and float(profile.decay_gamma) > 0
    ):
        positions = paddle.arange(
            int(target_ids.shape[-1]), dtype="float32"
        ).reshape([1, 1, -1])
        weights = weights * paddle.exp(-positions / float(profile.decay_gamma))
    elif profile.weight_type == "dpace":
        weights = weights * _dpace_weights(
            paddle.exp(-ce_per_token), eval_mask, profile.dpace_alpha
        )
    local_denominator = weights.sum()
    denominator = local_denominator
    if normalization_group is not None:
        denominator = local_denominator.detach().clone()
        paddle.distributed.all_reduce(
            denominator,
            op=paddle.distributed.ReduceOp.SUM,
            group=normalization_group,
        )
    denominator = denominator + 1e-6
    ce_loss = (ce_per_token * weights).sum() / denominator
    l1_loss = paddle.zeros([], dtype="float32")
    accept_rate = None
    if l1_per_token is not None:
        accept_rate = paddle.clip(1.0 - 0.5 * l1_per_token, min=0.0, max=1.0)
        if float(profile.probability_l1_alpha) != 0:
            l1_loss = (l1_per_token * weights).sum() / denominator

    confidence_loss = paddle.zeros([], dtype="float32")
    confidence_num = paddle.zeros([], dtype="float32")
    confidence_den = paddle.zeros([], dtype="float32")
    if output.confidence_pred is not None:
        if list(output.confidence_pred.shape) != list(target_ids.shape):
            raise ValueError("confidence_pred shape must match target_ids")
        confidence_loss = (
            paddle.nn.functional.binary_cross_entropy_with_logits(
                output.confidence_pred.astype("float32"),
                accept_rate.detach(),
                reduction="none",
            )
            * weights
        ).sum() / denominator
        confidence_probability = paddle.nn.functional.sigmoid(
            output.confidence_pred.astype("float32")
        )
        mask = eval_mask.astype("float32")
        confidence_num = (confidence_probability * mask).sum().detach()
        confidence_den = mask.sum().detach()

    selector_loss = paddle.zeros([], dtype="float32")
    selector_accuracy_num = paddle.zeros([], dtype="float32")
    selector_accuracy_den = paddle.zeros([], dtype="float32")
    selector_coverage_num = paddle.zeros([], dtype="float32")
    selector_coverage_den = paddle.zeros([], dtype="float32")
    selector_probability_num = paddle.zeros([], dtype="float32")
    selector_probability_den = paddle.zeros([], dtype="float32")
    if selector_enabled:
        selector_weights = weights * selector_covered_per_token
        # Selector CE is defined only on covered tokens.  Dividing by the base
        # objective's denominator silently scales the selector gradient by the
        # (often tiny) coverage ratio (about 5% for top-k=16 in ERNIE5), which
        # prevents the candidate selector from learning.  Keep its metric and
        # objective normalization on the same covered-token population.
        local_selector_denominator = selector_weights.sum()
        selector_denominator = local_selector_denominator
        if normalization_group is not None:
            selector_denominator = local_selector_denominator.detach().clone()
            paddle.distributed.all_reduce(
                selector_denominator,
                op=paddle.distributed.ReduceOp.SUM,
                group=normalization_group,
            )
        selector_denominator = selector_denominator + 1e-6
        selector_loss = (
            selector_ce_per_token * selector_weights
        ).sum() / selector_denominator
        metric_mask = eval_mask.astype("float32") * selector_covered_per_token
        selector_accuracy_num = (
            (selector_correct_per_token * metric_mask).sum().detach()
        )
        selector_accuracy_den = metric_mask.sum().detach()
        selector_coverage_num = metric_mask.sum().detach()
        selector_coverage_den = eval_mask.astype("float32").sum().detach()
        selector_probability_num = (
            (paddle.exp(-selector_ce_per_token).detach() * metric_mask)
            .sum()
            .detach()
        )
        selector_probability_den = selector_accuracy_den

    total = (
        float(profile.ce_alpha) * ce_loss
        + float(profile.probability_l1_alpha) * l1_loss
        + float(profile.confidence_alpha) * confidence_loss
        + effective_selector_alpha * selector_loss
    )
    if not return_terms:
        return total
    terms: dict[str, Any] = {
        "loss": total.detach(),
        "ce_loss": ce_loss.detach(),
        "l1_loss": l1_loss.detach(),
        "tv_loss": (0.5 * l1_loss).detach(),
        "confidence_loss": confidence_loss.detach(),
        "selector_loss": selector_loss.detach(),
        "selector_accuracy_num": selector_accuracy_num,
        "selector_accuracy_den": selector_accuracy_den,
        "selector_coverage_num": selector_coverage_num,
        "selector_coverage_den": selector_coverage_den,
        "selector_probability_num": selector_probability_num,
        "selector_probability_den": selector_probability_den,
        "selector_loss_alpha": paddle.to_tensor(
            effective_selector_alpha, dtype="float32"
        ),
        "valid_tokens": eval_mask.astype("int64").sum().detach(),
        "weight_denominator": local_denominator.detach(),
        "normalized_weight_denominator": denominator.detach(),
        "selector_weight_denominator": (
            local_selector_denominator.detach()
            if selector_enabled
            else paddle.zeros([], dtype="float32")
        ),
        "normalized_selector_weight_denominator": (
            selector_denominator.detach()
            if selector_enabled
            else paddle.zeros([], dtype="float32")
        ),
    }
    if accept_rate is not None:
        mask = eval_mask.astype("float32")
        terms["accept_rate_per_pos_num"] = (
            (accept_rate * mask).sum(axis=[0, 1]).detach()
        )
        terms["accept_rate_per_pos_den"] = mask.sum(axis=[0, 1]).detach()
        valid_blocks = (
            output.block_keep_mask.astype("bool")
            & eval_mask.astype("bool").any(axis=-1)
        ).astype("float32")
        tau = paddle.cumprod(accept_rate * mask, dim=-1).sum(axis=-1) + 1.0
        terms["tau_num"] = (tau * valid_blocks).sum().detach()
        terms["tau_den"] = valid_blocks.sum().detach()
        terms["confidence_prob_num"] = confidence_num
        terms["confidence_prob_den"] = confidence_den
    return LossTerms(
        loss=total, terms=terms, valid_tokens=terms["valid_tokens"]
    )
