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

"""DSpark draft layers, projection helpers, and loss utilities."""

from __future__ import annotations

import hashlib
import logging
import os
from copy import deepcopy
from typing import TYPE_CHECKING

import paddle
from paddle import Tensor, nn
from paddle.distributed.fleet.meta_parallel import LayerSpec

from paddlefleet.cp_shard import cp_shard_source, extract_local_cp_chunks
from paddlefleet.transformer.dspark_attention import (
    DsparkHCAAttention,
    DsparkTransformerLayer,
)
from paddlefleet.transformer.hyper_connection import (
    HyperConnectionContractLayer,
    HyperConnectionModule,
)
from paddlefleet.transformer.layer import FleetLayer
from paddlefleet.transformer.paddle_norm import WrappedPaddleNorm
from paddlefleet.utils.dspark.dspark_utils import (
    ReadOnlyProjection,
    build_draft_blocks,
    gather_target_hidden,
    get_doc_start_per_pos,
    readonly_embedding,
)

if TYPE_CHECKING:
    from paddlefleet.transformer.transformer_config import TransformerConfig

logger = logging.getLogger(__name__)


def _dspark_global_step() -> int:
    """Optimizer step for deterministic anchor sampling.

    Reuses the MTP ``TRAINER_GLOBAL_STEP`` / ``PDC_INIT_STEP`` env contract so
    anchor sampling is a pure function of (seed, step, sample_id) and needs no
    collective; absent env -> step 0 (constant sampling).
    """
    for key in ("TRAINER_GLOBAL_STEP", "PDC_INIT_STEP"):
        value = os.environ.get(key)
        if value is not None:
            try:
                return max(0, int(value))
            except ValueError:
                pass
    return 0


class MarkovHead(nn.Layer):
    """Low-rank Markov transition refiner (default dspark refiner)."""

    def __init__(self, vocab_size, rank):
        super().__init__()
        self.markov_w1 = nn.Embedding(vocab_size, rank)
        self.markov_w2 = nn.Linear(
            rank,
            vocab_size,
            bias_attr=False,
            weight_attr=paddle.ParamAttr(
                initializer=nn.initializer.Constant(0.0)
            ),
        )

    def get_prev_embeddings(self, token_ids):
        """Rank-`markov_rank` predecessor embeddings, reused by the confidence head."""
        return self.markov_w1(token_ids.astype("int64"))


class CandidateSelector(nn.Layer):
    """Low-rank scorer for a strict unary top-k candidate set.

    The selector does not produce a full-vocabulary bias. It only re-ranks the
    candidates supplied by the caller and keeping the training path compatible
    with streamed logits.

    When used instead of ``MarkovHead`` its ``state_rank`` reuses the single
    ``dspark_markov_rank`` config knob (no separate selector_rank field).
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        vocab_size: int,
        state_rank: int,
        top_k: int,
        initializer_range: float,
    ) -> None:
        super().__init__()
        hidden_size = int(hidden_size)
        vocab_size = int(vocab_size)
        state_rank = int(state_rank)
        top_k = int(top_k)
        initializer_range = float(initializer_range)
        if hidden_size <= 0:
            raise ValueError("CandidateSelector hidden_size must be positive")
        if vocab_size <= 0:
            raise ValueError("CandidateSelector vocab_size must be positive")
        if state_rank < 1:
            raise ValueError("CandidateSelector state_rank must be >= 1")
        if top_k < 1 or top_k > vocab_size:
            raise ValueError(
                "CandidateSelector top_k must be in [1, vocab_size], got "
                f"{top_k} for vocab_size={vocab_size}"
            )
        if initializer_range < 0:
            raise ValueError(
                "CandidateSelector initializer_range must be non-negative"
            )

        self.vocab_size = vocab_size
        self.state_rank = state_rank
        self.top_k = top_k
        self.predecessor_codebook = self.create_parameter(
            [vocab_size, state_rank],
            default_initializer=nn.initializer.Normal(std=initializer_range),
        )
        # Starting with no transition preserves the unary model at step zero.
        self.successor_codebook = self.create_parameter(
            [vocab_size, state_rank],
            default_initializer=nn.initializer.Constant(0.0),
        )
        self.hidden_projection = nn.Linear(
            hidden_size, state_rank, bias_attr=False
        )

    def score_candidates(
        self,
        *,
        candidate_ids: Tensor,
        unary_logits: Tensor,
        hidden_states: Tensor,
        predecessor_ids: Tensor,
    ) -> Tensor:
        """Add low-rank predecessor transitions to candidate logits."""

        candidate_ids = candidate_ids.astype("int64")
        predecessor_ids = predecessor_ids.astype("int64")
        successor = paddle.nn.functional.embedding(
            candidate_ids, self.successor_codebook
        )
        predecessor = self.get_predecessor_embeddings(predecessor_ids)
        context = predecessor * self.hidden_projection(hidden_states)
        transition = paddle.einsum("...r,...kr->...k", context, successor)
        return unary_logits + transition

    def get_predecessor_embeddings(self, token_ids: Tensor) -> Tensor:
        """Return codebook states for predecessor token ids."""

        return paddle.nn.functional.embedding(
            token_ids.astype("int64"), self.predecessor_codebook
        )


class ConfidenceHead(nn.Layer):
    """Predict a per-draft-token acceptance logit (confidence BCE target).

    Input is the draft hidden, optionally concatenated with the refiner's
    predecessor embedding (``confidence_head_with_markov/selector``); kept as a
    plain bias-free linear to the scalar logit.
    """

    def __init__(self, *, hidden_size: int, extra_rank: int = 0) -> None:
        super().__init__()
        self.proj = nn.Linear(
            int(hidden_size) + int(extra_rank), 1, bias_attr=False
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.proj(features).squeeze(-1)


class DSparkLossLoggingHelper:
    """Helper for logging the (single) DSpark loss, mirroring MTPLossLoggingHelper."""

    tracker: dict = {}

    @staticmethod
    def save_loss_to_tracker(
        loss: paddle.Tensor,
        reduce_group=None,
        avg_group=None,
    ):
        """Accumulate the dspark loss for logging."""
        tracker = DSparkLossLoggingHelper.tracker
        if "value" not in tracker:
            tracker["value"] = paddle.zeros([1])
        tracker["value"] += loss.detach().reshape([1])
        tracker["reduce_group"] = reduce_group
        tracker["avg_group"] = avg_group

    @staticmethod
    def clean_loss_in_tracker():
        tracker = DSparkLossLoggingHelper.tracker
        if "value" in tracker:
            tracker["value"].zero_()
        tracker["reduce_group"] = None
        tracker["avg_group"] = None

    @staticmethod
    def reduce_loss_in_tracker():
        tracker = DSparkLossLoggingHelper.tracker
        if "value" not in tracker:
            return
        value = tracker["value"]
        if tracker.get("reduce_group") is not None:
            paddle.distributed.all_reduce(value, group=tracker["reduce_group"])
        if tracker.get("avg_group") is not None:
            paddle.distributed.all_reduce(
                value,
                group=tracker["avg_group"],
                op=paddle.distributed.ReduceOp.AVG,
            )

    @staticmethod
    def track_dspark_metrics(
        loss_scale, iteration, writer, wandb_writer=None, total_loss_dict=None
    ):
        """Track the DSpark loss metric for logging."""
        DSparkLossLoggingHelper.reduce_loss_in_tracker()
        tracker = DSparkLossLoggingHelper.tracker
        if "value" not in tracker:
            return
        name = "dspark loss"
        loss = tracker["value"][0] * loss_scale
        if total_loss_dict is not None:
            total_loss_dict[name] = total_loss_dict.get(name, 0) + loss
        if writer is not None:
            writer.add_scalar(name, loss, iteration)
        if wandb_writer is not None:
            wandb_writer.log({name: loss}, iteration)
        DSparkLossLoggingHelper.clean_loss_in_tracker()


def drafter_provider(target_config, target_layer_ids):
    """Create a draft config from selected target layers."""
    ids = list(target_layer_ids)
    provider = deepcopy(target_config)
    provider.num_hidden_layers = len(ids)
    # ``recompute_num_layers`` is a depth-relative count, not a strategy:
    # need_recompute_in_first_n / _in_block validate it against the config's own
    # depth, and the draft stack is only len(ids) deep. Inheriting the target's
    # value verbatim can exceed that and trip a bare assert at construction time.
    if (
        provider.recompute_granularity == "full"
        and provider.recompute_num_layers is not None
    ):
        provider.recompute_num_layers = min(
            provider.recompute_num_layers, len(ids)
        )
    # Per-layer patterns are re-indexed to the selected layers.
    for key in ("moe_layer_freq", "window_attn_skip_freq"):
        values = getattr(target_config, key, None)
        if isinstance(values, (list, tuple)):
            setattr(provider, key, [values[i] for i in ids])
    # Every draft layer runs CSA ratio=0 (MQA KV + sliding window).
    provider.csa_compress_ratios = [0] * len(ids)
    # Draft layers carry no leading dense block and no latent-MQA/-2 switches.
    for key, value in {
        "first_k_dense_replace": 0,
        "hybrid_mla_attention": "mha",
        "multi_latent_attention": False,
        "mqa_split_kv_b_proj": False,
        "num_nextn_predict_layers": 0,
        "mtp_shared_last_layer": False,
        "enable_mtp_magic_send": False,
        "use_erndata": False,
        "experimental_dataflow": False,
        "separate_mtp_input": False,
        "separate_mtp_headloss": False,
        "use_dense_mtp": False,
        "num_empty_layers_add_in_head": 0,
        "num_empty_layers_add_in_tail": 0,
        "cpu_offloading": False,
        # The draft module is not itself a dspark target: clear the trigger so
        # its (deepcopied) config does not recurse into dspark construction.
        "dspark_target_layer_ids": None,
    }.items():
        if hasattr(provider, key):
            setattr(provider, key, value)
    return provider


# ---------------------------------------------------------------------------
# Shared backbone weights for the draft
# ---------------------------------------------------------------------------
# The draft reads the backbone embedding / LM head weights but owns no
# parameters for them, and it reads them detached (see ``readonly_embedding``
# and ``ReadOnlyProjection``). The sharing itself is declared in
# ``GPTModel.get_layer_desc_list`` as ``SharedLayerDesc`` entries carrying the
# backbone's key, so paddle hands these two ``forward_func``s the *same*
# instance the backbone uses -- one pipeline step that computes nothing, whose
# only job is to republish the weights on ``dict_args`` for the draft.
#
# ``dict_args`` values must stay tensors: paddle flattens them at the stage
# boundary with ``tensor.key = key`` (pp_utils/utils.py), so the optional head
# tensors are omitted when absent rather than published as ``None``.


def dspark_share_embedding(shared_embedding, dict_args):
    """Republish the shared embedding weight as ``dspark_embedding_weight``.

    Runs as a ``SharedLayerDesc.forward_func`` partial, so it receives the
    shared embedding instance the framework built and must return ``dict_args``
    unchanged apart from the republished weight.
    """
    weight = getattr(shared_embedding, "embedding_weight", None)
    if weight is None:
        weight = getattr(shared_embedding, "weight", None)
    if weight is None:
        raise RuntimeError(
            "DSpark shared embedding weight missing: the shared layer of type "
            f"{type(shared_embedding).__name__} exposes neither "
            "'embedding_weight' nor 'weight'."
        )
    dict_args["dspark_embedding_weight"] = weight
    return dict_args


def dspark_share_lm_head(shared_head, dict_args):
    """Republish the shared LM head weights as ``dspark_head_*``.

    Same contract as :func:`dspark_share_embedding`. ``bias`` / ``multimax_*``
    are only published when the head actually carries them, so the draft's
    ``ReadOnlyProjection`` sees the same optional tensors it used to read off
    the borrowed module.
    """
    weight = getattr(shared_head, "weight", None)
    if weight is None:
        raise RuntimeError(
            "DSpark shared LM head weight missing: the shared layer of type "
            f"{type(shared_head).__name__} exposes no 'weight'."
        )
    dict_args["dspark_head_weight"] = weight
    for key, attr in (
        ("dspark_head_bias", "bias"),
        ("dspark_head_multimax_ranges", "multimax_ranges"),
        ("dspark_head_multimax_ts", "multimax_ts"),
    ):
        value = getattr(shared_head, attr, None)
        if value is not None:
            dict_args[key] = value
    return dict_args


class DsparkPredictionLayer(FleetLayer):
    """Run the DSpark draft branch inside the GPT pipeline."""

    def __init__(self, config, vocab_size, layer_number=0, pg_collection=None):
        super().__init__(config=config)
        # Delay imports to avoid the GPT spec/model import cycle.
        from paddle.distributed.fleet.meta_parallel import build_spec_layer

        from paddlefleet.models.gpt.gpt_layer_specs import (
            get_gpt_decoder_layers_spec,
        )
        from paddlefleet.process_groups_config import ProcessGroupCollection

        self.layer_number = layer_number
        self.vocab_size = int(vocab_size)
        self.target_layer_ids = list(config.dspark_target_layer_ids)
        self.hidden_size = int(config.hidden_size)
        # Populated by GPTModel for the in-pipeline path. The shared backbone
        # embedding / LM head weights are *not* held here: the SharedLayerDesc
        # pass-throughs declared in GPTModel.get_layer_desc_list republish them
        # on dict_args (see ``dspark_share_embedding`` / ``dspark_share_lm_head``).
        self._host_meta = None

        # Draft config: target structure shortened to the selected layers, all
        # forced to CSA ratio=0 (MQA + sliding window); MoE/EP inherited.
        self.draft_config = drafter_provider(config, self.target_layer_ids)

        # Feature projection: concat of per-layer target features -> hidden.
        self.feature_width = len(self.target_layer_ids) * self.hidden_size
        self.feature_projection = nn.Linear(
            self.feature_width, self.hidden_size, bias_attr=False
        )
        self.feature_norm = WrappedPaddleNorm(
            self.draft_config, self.hidden_size, self.draft_config.rms_norm_eps
        )

        # Draft transformer blocks: standard GPT decoder spec with the self-attn
        # swapped to the DSpark CSA ratio=0 attention.
        specs = get_gpt_decoder_layers_spec(self.draft_config)
        for spec in specs:
            spec.layer = DsparkTransformerLayer
            spec.sublayers_spec.self_attn.layer = DsparkHCAAttention
        groups = (
            pg_collection
            if pg_collection is not None
            else ProcessGroupCollection.use_mpu_process_groups()
        )
        # Retained for forward-time CP-rank-aware document-boundary parsing.
        self._pg_collection = groups

        self.layers = nn.LayerList(
            [build_spec_layer(spec, pg_collection=groups) for spec in specs]
        )
        # Draft sublayers never engage their own CP: attention/MLP over a local
        # shard needs no cross-rank communication.
        for sublayer in self.layers.sublayers():
            if hasattr(sublayer, "context_parallel_size"):
                sublayer.context_parallel_size = 1

        self.mhc_contract = HyperConnectionContractLayer(self.draft_config)
        self.final_norm = WrappedPaddleNorm(
            self.draft_config, self.hidden_size, self.draft_config.rms_norm_eps
        )

        # Markov refinement is the default.
        self.markov_head = MarkovHead(
            self.vocab_size, int(config.dspark_markov_rank)
        )
        self.candidate_selector = None
        # Confidence prediction for each draft token.
        self.confidence_head = ConfidenceHead(
            hidden_size=self.hidden_size,
            extra_rank=config.dspark_markov_rank,
        )

    def forward(self, dict_args: dict):
        """Populate ``dict_args["dspark_out"]`` with draft outputs."""
        (
            features,
            final_hidden,
            embedding_weight,
            head,
            input_ids,
            labels,
            attn_mask_startend_row_indices,
        ) = self._resolve_inputs(dict_args)

        # CP frames are torn: features/final_hidden/context are LOCAL, while
        # input_ids/labels/attn_mask_startend_row_indices are GLOBAL. Which one
        # the ids arrive in is a contract read off cp_shard_source, never a
        # shape test: a shape test turns a missing or duplicated transform into
        # a silent error instead of a failure.
        #
        # extract_local_cp_chunks, not to_cp_local: under "scatter" every rank
        # already holds the full global tensor, so this is a communication-free
        # slice matching the layout the embedding scattered with. The mask
        # stays global -- get_doc_start_per_pos takes a global mask.
        sq = int(features.shape[1])
        cp_size = max(1, int(getattr(self._pg_collection.cp, "nranks", 1) or 1))
        cp_rank = int(getattr(self._pg_collection.cp, "rank", 0) or 0)
        # Declared by TrainingArguments; no getattr fallback, so a None here
        # would be a silently missing field rather than a deliberate default.
        mode = self.config.cp_balance_mode
        cp_source = cp_shard_source(self.config, cp_size=cp_size)
        global_ids = cp_source in ("scatter", "local")
        expected = sq * cp_size if global_ids else sq
        for name, tensor in (("input_ids", input_ids), ("labels", labels)):
            if tensor is None:
                continue
            if len(tensor.shape) < 2 or int(tensor.shape[1]) != expected:
                raise ValueError(
                    f"{name} must be delivered in "
                    f"{'GLOBAL' if global_ids else 'LOCAL'} coordinates under "
                    f"cp_shard_source={cp_source!r} (cp_size={cp_size}): "
                    f"expected seq len {expected} = local {sq}"
                    + (f" x cp_size {cp_size}" if global_ids else "")
                    + f", got {tuple(tensor.shape)}."
                )
            if global_ids:
                sliced = extract_local_cp_chunks(
                    tensor, cp_rank, cp_size, axis=1, mode=mode
                )
                if name == "input_ids":
                    input_ids = sliced
                else:
                    labels = sliced

        # position_ids rides the same contract. A 1D-RoPE reader produces no
        # such key, in which case the local frame is stated explicitly; it must
        # be the same tensor for build_draft_blocks and the backbone.
        position_ids = dict_args.get("position_ids")
        if position_ids is not None:
            if int(position_ids.shape[-1]) != expected:
                raise ValueError(
                    "position_ids must be delivered in "
                    f"{'GLOBAL' if global_ids else 'LOCAL'} coordinates, like "
                    f"input_ids: expected {expected} (local {sq}"
                    + (f" x cp_size {cp_size}" if global_ids else "")
                    + f" under cp_shard_source={cp_source!r}, got "
                    f"{int(position_ids.shape[-1])}."
                )
            if global_ids:
                position_ids = extract_local_cp_chunks(
                    position_ids, cp_rank, cp_size, axis=-1, mode=mode
                )
        else:
            position_ids = (
                paddle.arange(sq, dtype="int64")
                .reshape([1, sq])
                .expand([int(input_ids.shape[0]), sq])
            )

        # Zigzag (``dualchunk_allgather``) concatenates two globally distant
        # intervals per rank, so local adjacency stops matching global adjacency
        # at the seam. Every rank but the last has a real gap there; the last
        # rank's two intervals are adjacent, so it has no seam. The draft's
        # window and target block are built from LOCAL indices and the document
        # filter cannot see the seam when one document spans it, so both must be
        # told where it is.
        chunk_seam = None
        if (
            cp_size > 1
            and mode == "dualchunk_allgather"
            and cp_rank < cp_size - 1
        ):
            chunk_seam = sq // 2

        # num_anchors stays at 64 PER RANK, so anchor density scales with
        # cp_size (16x the CP=1 density at cp=16). Deliberate, but a CP run's
        # draft supervision is therefore not the same sampling density as a
        # CP=1 run's.
        blocks = build_draft_blocks(
            input_ids,
            labels,
            block_size=int(self.config.dspark_block_size),
            mask_token_id=int(self.config.dspark_noise_token_id),
            num_anchors=int(self.config.dspark_num_anchors),
            sample_ids=self._sample_ids(dict_args, input_ids),
            global_step=_dspark_global_step(),
            training_seed=int(getattr(self.config, "seed", 42) or 42),
            input_mode="anchor_mask",
            position_ids=position_ids,
            chunk_seam=chunk_seam,
        )
        noise_ids = blocks.noise_ids.reshape([blocks.noise_ids.shape[0], -1])
        noise = readonly_embedding(noise_ids, embedding_weight)

        draft_hidden, confidence_pred = self._run_draft_backbone(
            features,
            final_hidden,
            blocks,
            noise,
            input_ids=input_ids,
            position_ids=position_ids,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            chunk_seam=chunk_seam,
        )

        head_weight = head["weight"]
        # The shared GPT LM head builds its weight as [out, in] =
        # [vocab, hidden] (lm_head.py) and DSpark requires TP==1, so shape[0]
        # is the full vocab and exceeds hidden_size for any real tokenizer:
        # this is a [V, H] head. Reading the direction from the layout (rather
        # than hard-coding True) stays correct if the head ever changes; the
        # one ambiguous case, a square head, is unreachable under TP==1.
        transpose_y = int(head_weight.shape[0]) != self.hidden_size
        dict_args["dspark_out"] = {
            "draft_hidden_states": draft_hidden,
            "aligned_target_hidden_states": gather_target_hidden(
                final_hidden, blocks
            ),
            "target_ids": blocks.target_ids,
            "eval_mask": blocks.eval_mask,
            "block_keep_mask": blocks.anchors.keep_mask,
            "previous_token_ids": blocks.previous_token_ids,
            "logits_projection": ReadOnlyProjection(
                weight=head_weight,
                transpose_y=transpose_y,
                bias=head["bias"],
                multimax_ranges=head["multimax_ranges"],
                multimax_ts=head["multimax_ts"],
            ),
            "markov_head": self.markov_head,
            "candidate_selector": self.candidate_selector,
            "confidence_pred": confidence_pred,
        }
        return dict_args

    def _resolve_inputs(self, dict_args):
        """Resolve captured or directly supplied draft inputs.

        The embedding / LM head weights always come off ``dict_args`` -- in the
        production (capture) path and in the direct-injection path alike. They
        are published by the ``SharedLayerDesc`` pass-throughs
        (:func:`dspark_share_embedding` / :func:`dspark_share_lm_head`) that
        ``GPTModel.get_layer_desc_list`` declares next to the draft, so the
        weights arrive through the pipeline instead of through a private
        attribute poked onto this layer after construction.
        """
        embedding_weight = self._require_shared_weight(
            dict_args, "dspark_embedding_weight", "embedding"
        )
        head = self._shared_head_tensors(dict_args)

        if "dspark_features" in dict_args:
            return (
                dict_args["dspark_features"],
                dict_args["dspark_final_hidden"],
                embedding_weight,
                head,
                dict_args["input_ids"],
                dict_args["labels"],
                dict_args.get("attn_mask_startend_row_indices"),
            )

        missing = [
            lid
            for lid in self.target_layer_ids
            if f"dspark_target_hidden_{lid}" not in dict_args
        ]
        final_hidden = dict_args.get("dspark_target_last_hidden_states")
        if missing or final_hidden is None:
            raise RuntimeError(
                "DSpark enabled but feature capture is incomplete: missing "
                f"dspark_target_hidden layers={missing}, dspark_target_last_hidden_states="
                f"{'present' if final_hidden is not None else 'MISSING'}. A "
                "target layer / final norm likely ran on a forward variant that "
                "does not merge rst into dict_args (overlap ScheduleNode / "
                "HySparse), or lives in a different stage/chunk than the draft. "
                f"dict_args keys={sorted(dict_args.keys())}"
            )
        features = paddle.concat(
            [
                dict_args[f"dspark_target_hidden_{lid}"]
                for lid in self.target_layer_ids
            ],
            axis=-1,
        )
        # Remove capture keys before forwarding to the loss stage.
        for lid in self.target_layer_ids:
            dict_args.pop(f"dspark_target_hidden_{lid}", None)
        dict_args.pop("dspark_target_last_hidden_states", None)

        host = self._host_meta or {}

        def _pick(key):
            v = host.get(key)
            return v if v is not None else dict_args.get(key)

        input_ids = _pick("input_ids")
        labels = _pick("labels")

        return (
            features,
            final_hidden,
            embedding_weight,
            head,
            input_ids,
            labels,
            _pick("attn_mask_startend_row_indices"),
        )

    @staticmethod
    def _require_shared_weight(dict_args, key, what):
        """Fetch a weight published by a SharedLayerDesc pass-through."""
        weight = dict_args.get(key)
        if weight is None:
            raise RuntimeError(
                f"DSpark shared {what} weight missing: dict_args carries no "
                f"{key!r}. The SharedLayerDesc pass-through that publishes it "
                "did not run before the draft -- check the draft's position in "
                "GPTModel.get_layer_desc_list. "
                f"dict_args keys={sorted(dict_args.keys())}"
            )
        return weight

    @staticmethod
    def _shared_head_tensors(dict_args):
        """Collect the LM head tensors the draft projects with.

        Only the keys the pass-through actually published are present: an
        absent ``bias`` / ``multimax_*`` stays absent so ``ReadOnlyProjection``
        keeps its own "not configured" semantics.
        """
        return {
            "weight": DsparkPredictionLayer._require_shared_weight(
                dict_args, "dspark_head_weight", "LM head"
            ),
            "bias": dict_args.get("dspark_head_bias"),
            "multimax_ranges": dict_args.get("dspark_head_multimax_ranges"),
            "multimax_ts": dict_args.get("dspark_head_multimax_ts"),
        }

    def _sample_ids(self, dict_args, input_ids):
        """Deterministic per-row ids for anchor sampling (content hash fallback).

        The preferred source is ``dict_args["sample_ids"]``, a stable per-row id
        injected by the data pipeline; nothing writes it yet, so the fallback
        below runs every step. Hashing the content (rather than a per-step
        counter) is deliberate: anchor sampling stays a pure function of
        ``(seed, global_step, sample_id)`` and resumes without a persisted
        Python counter.
        """
        ids = dict_args.get("sample_ids")
        if ids is not None:
            return ids
        rows = input_ids.astype("int64").numpy()
        return [hashlib.sha256(row.tobytes()).hexdigest() for row in rows]

    def _run_draft_backbone(
        self,
        features,
        final_hidden,
        blocks,
        noise,
        *,
        input_ids,
        position_ids,
        attn_mask_startend_row_indices=None,
        chunk_seam=None,
    ):
        """Run the draft transformer over shared target and noise KV."""
        b, n_anchor, block_k = blocks.noise_ids.shape
        n_streams = int(self.draft_config.num_residual_streams)

        # Context: project the full target feature sequence once (features are
        # already detached -> gradient reaches feature_projection, not target).
        context = self.feature_norm(self.feature_projection(features))

        noise_streams = HyperConnectionModule.input_expand(noise, n_streams)

        # Settled by forward(); a None means a caller bypassed it.
        if position_ids is None:
            raise ValueError(
                "position_ids must be settled by DsparkPredictionLayer.forward "
                "before _run_draft_backbone; got None."
            )
        doc_start_per_pos = get_doc_start_per_pos(
            input_ids,
            position_ids,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            cp_rank=getattr(self._pg_collection.cp, "rank", 0),
            cp_size=getattr(self._pg_collection.cp, "nranks", 1),
            cp_balance_mode=self.config.cp_balance_mode,
        )
        draft_position_ids = blocks.position_ids.reshape(
            [b, n_anchor * block_k]
        )

        state = {
            "hidden_states": noise_streams,
            "dspark_context": context,
            "dspark_anchor_positions": blocks.anchors.positions,
            "dspark_block_keep_mask": blocks.anchors.keep_mask,
            "dspark_doc_start_per_pos": doc_start_per_pos,
            "dspark_context_position_ids": position_ids,
            "dspark_draft_position_ids": draft_position_ids,
            "dspark_block_size": int(block_k),
            "dspark_chunk_seam": chunk_seam,
        }
        for layer in self.layers:
            state = layer(state)
        contracted = self.mhc_contract(state)["hidden_states"]
        contracted = contracted.reshape(
            [b, n_anchor, block_k, self.hidden_size]
        )
        draft_hidden = self.final_norm(contracted)
        confidence = None
        if self.confidence_head is not None:
            prev_emb = self.markov_head.get_prev_embeddings(
                blocks.previous_token_ids
            )
            prev_emb = prev_emb.astype(contracted.dtype)
            contracted = paddle.concat([contracted, prev_emb], axis=-1)
            confidence = self.confidence_head(contracted)
        return draft_hidden, confidence

    def build_schedule_node(self):
        from paddle.distributed.fleet.meta_parallel import ScheduleNode

        return ScheduleNode(self.forward, name="DsparkPredictionLayer")


def get_dspark_layer_spec_for_backend(
    config: TransformerConfig,
    vocab_size: int,
    layer_number: int = 0,
) -> LayerSpec:
    """Build the DSpark prediction layer specification."""
    return LayerSpec(
        layer=DsparkPredictionLayer,
        extra_kwargs={
            "config": config,
            "vocab_size": vocab_size,
            "layer_number": layer_number,
        },
    )


class DSparkLossAutoScaler(paddle.autograd.PyLayer):
    """Inject the DSpark loss gradient through the main loss."""

    main_loss_backward_scale: paddle.Tensor = paddle.tensor(1.0)

    @staticmethod
    def forward(ctx, output: paddle.Tensor, dspark_loss: paddle.Tensor):
        """Keep the DSpark loss alive until backward."""
        ctx.save_for_backward(dspark_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: paddle.Tensor):
        """Return the carrier gradient and DSpark loss gradient."""
        (dspark_loss,) = ctx.saved_tensor()
        scale = DSparkLossAutoScaler.main_loss_backward_scale
        scaled_dspark_loss_grad = paddle.ones_like(dspark_loss) * scale
        return grad_output, scaled_dspark_loss_grad

    @staticmethod
    def set_loss_scale(scale: paddle.Tensor):
        """Set the main-loss scale used for DSpark gradients."""
        DSparkLossAutoScaler.main_loss_backward_scale = scale
