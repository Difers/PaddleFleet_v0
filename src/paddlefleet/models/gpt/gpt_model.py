# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import annotations

import logging
from bisect import bisect_right
from dataclasses import dataclass
from typing import TYPE_CHECKING

from paddle.distributed.fleet.meta_parallel import (
    LayerDesc,
    PipelineLayer,
    SharedLayerDesc,
    dict_to_tuple_helper,
)

if TYPE_CHECKING:
    from paddle.distributed.fleet.meta_parallel import LayerSpec


from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import ScheduleChunk

from paddlefleet.models.gpt.aoa_generator import (
    build_aoa_context,
    gen_whole_model_aoa,
    gen_whole_model_inv_aoa,
)
from paddlefleet.models.gpt.gpt_embedding import GPTEmbedding
from paddlefleet.models.gpt.lm_head import (
    GPTLMHead,
    GPTMainLMHead,
    GPTMTPLMHead,
)
from paddlefleet.transformer.multi_token_prediction import (
    MultiTokenPredictionLayer,
)
from paddlefleet.transformer.paddle_norm import WrappedPaddleNormPipe
from paddlefleet.transformer.parallel_multi_token_prediction import (
    DsparkPredictionLayer,
    dspark_share_embedding,
    dspark_share_lm_head,
)
from paddlefleet.transformer.transformer_config import dspark_layers_active
from paddlefleet.transformer.transformer_layer import (
    HyperConnectionTransformerLayer,
    TransformerLayer,
    TransformerLayerNode,
    TransformerLayerOverlappedScheduleNode,
    TransformerLayerWithOverlap,
)

logger = logging.getLogger(__name__)


_VISION_MERGE_PREFIX = "vision_merge."
_VISION_MODEL_PREFIX = "vision_merge.vision_model."


def is_vision_merge_key(key: str) -> bool:
    """Whether ``key`` belongs to the vision encoder held by ``vision_merge``.

    ``vision_merge`` (qwen3_5 PP mode) is a thin wrapper whose only parameters
    live under ``vision_model``; those keys are dropped from the pipeline name
    mapping and re-added from the vision model's own state dict. A parameter
    placed directly on the wrapper would be dropped without being re-added,
    i.e. silently missing from every checkpoint, so reject it loudly.

    Raises ``ValueError`` (not ``assert``) on purpose: with ``python -O`` an
    assertion would vanish, this function would answer ``True`` and the
    parameter would be silently lost from the checkpoint.
    """
    if not key.startswith(_VISION_MERGE_PREFIX):
        return False
    if not key.startswith(_VISION_MODEL_PREFIX):
        raise ValueError(
            f"Unexpected parameter {key!r} directly on vision_merge. Only "
            f"{_VISION_MODEL_PREFIX}* is re-exported through the vision "
            "model's own state dict; anything else would be silently dropped "
            "from the checkpoint. Move the parameter under vision_model or "
            "extend the state-dict plumbing in GPTModel."
        )
    return True


def build_overlapped_nodes(forward_chunk, backward_chunk):
    """Build overlapped nodes for TransformerLayer."""
    overlap_element_class = TransformerLayerNode
    forward_decoder_layer_num = 0
    backward_decoder_layer_num = 0

    assert isinstance(forward_chunk, ScheduleChunk) and isinstance(
        backward_chunk, ScheduleChunk
    )
    for n in forward_chunk.nodes:
        if isinstance(n, overlap_element_class):
            forward_decoder_layer_num += 1
    for n in reversed(backward_chunk.nodes):
        if isinstance(n, overlap_element_class):
            backward_decoder_layer_num += 1

    overlap_layers_num = min(
        forward_decoder_layer_num, backward_decoder_layer_num
    )

    # construct forward pre- and post-chunks
    forward_pre_layers = []
    forward_post_layers = []
    forward_overlap_layers = []
    is_pre = True
    for n in forward_chunk.nodes:
        if not isinstance(n, overlap_element_class):
            if is_pre:
                forward_pre_layers.append(n)
            else:
                forward_post_layers.append(n)
        else:
            is_pre = False
            if len(forward_overlap_layers) == overlap_layers_num:
                forward_post_layers.append(n)
            else:
                forward_overlap_layers.append(n)

    forward_pre_node = ScheduleChunk(forward_pre_layers)
    forward_post_node = ScheduleChunk(forward_post_layers)

    # construct backward pre- and post-chunks
    backward_pre_layers = []
    backward_post_layers = []
    backward_overlap_layers = []
    is_pre = True
    for n in reversed(backward_chunk.nodes):
        if not isinstance(n, overlap_element_class):
            if is_pre:
                backward_pre_layers.append(n)
            else:
                backward_post_layers.append(n)
        else:
            is_pre = False
            if len(backward_overlap_layers) == overlap_layers_num:
                backward_post_layers.append(n)
            else:
                backward_overlap_layers.append(n)

    backward_pre_node = ScheduleChunk(list(reversed(backward_pre_layers)))
    backward_post_node = ScheduleChunk(list(reversed(backward_post_layers)))

    # construct overlap chunk
    overlap_node = ScheduleChunk(
        [
            TransformerLayerOverlappedScheduleNode(forward_node, backward_node)
            for forward_node, backward_node in zip(
                forward_overlap_layers, backward_overlap_layers
            )
        ]
    )
    return (
        forward_pre_node,
        backward_pre_node,
        overlap_node,
        forward_post_node,
        backward_post_node,
    )


@dataclass
class GPTSublayersSpec:
    """p
    The dataclass for LayerSpecs of GPT sublayers_spec
    including embedding, n * transformer_layer, mtp, lm_head.
    """

    embedding: LayerSpec | None = None
    head_empty_layers: list[LayerSpec] | None = None
    mhc_expand: LayerSpec | None = None
    transformer_layers: list[LayerSpec] | None = None
    mhc_contract: LayerSpec | None = None
    tail_empty_layers: list[LayerSpec] | None = None
    mtp: list[LayerSpec] | None = None
    output_block_attn_res: LayerSpec | None = None
    layer_norm: LayerSpec | None = None
    lm_head: LayerSpec | None = None
    mtp_lm_head: LayerDesc | None = None
    mtp_loss: LayerDesc | None = None
    dspark: list[LayerSpec] | None = None
    dspark_loss: LayerDesc | None = None


class GPTModel(PipelineLayer):
    """GPT Transformer language model.

    Args:
        gpt_layer_desc:
    """

    def __init__(
        self,
        sublayers_spec: GPTSublayersSpec,
        **kwargs,
    ) -> None:
        self.config = kwargs["config"]
        if getattr(self.config, "indexcache_topk_pattern", None):
            # Validate before building layers so every caller can use native PP.
            if self.config.virtual_pipeline_model_parallel_size not in (
                None,
                1,
            ):
                raise NotImplementedError(
                    "IndexCache supports only ordinary 1F1B (VPP=1)"
                )
            if kwargs.get("use_dualpipev", False):
                raise NotImplementedError(
                    "IndexCache does not support DualPipeV scheduling"
                )
            strategy = getattr(fleet.fleet, "_user_defined_strategy", None)
            if strategy is not None:
                pp_config = strategy.hybrid_configs["pp_configs"]
                if (
                    pp_config.use_dualpipev
                    or pp_config.forward_backward_overlap_scheduler
                ):
                    raise NotImplementedError(
                        "IndexCache does not support DualPipeV or compute-overlap scheduling"
                    )
                if (
                    self.config.pipeline_model_parallel_size > 1
                    and strategy.amp
                ):
                    raise NotImplementedError(
                        "IndexCache PP requires Trainer-managed AMP/scaler rather than strategy.amp"
                    )
        if dspark_layers_active(self.config):
            # The draft takes the backbone embedding / LM head weights through
            # SharedLayerDesc, and paddle rejects SharedLayerDesc outright under
            # dualpipev ("dualpipev scheduler does not support SharedLayerDesc
            # yet"). Fail here instead, where the reason is stated.
            strategy = getattr(fleet.fleet, "_user_defined_strategy", None)
            pp_config = (
                strategy.hybrid_configs["pp_configs"]
                if strategy is not None
                else None
            )
            if kwargs.get("use_dualpipev", False) or (
                pp_config is not None and pp_config.use_dualpipev
            ):
                raise NotImplementedError(
                    "DSpark does not support DualPipeV: the draft shares the "
                    "backbone embedding / LM head through SharedLayerDesc, "
                    "which paddle's dualpipev scheduler rejects outright."
                )
        tie_word_embeddings = (
            kwargs["tie_word_embeddings"]
            and self.config.pipeline_model_parallel_size > 1
        )
        skip_weight_param_allocation = (
            self.config.tie_word_embeddings
            and self.config.pipeline_model_parallel_size == 1
        )
        self._pipeline_name_mapping = None
        self._pp_to_single_mapping = None
        self._sequential_layers = self.get_layer_desc_list(
            sublayers_spec,
            tie_word_embeddings,
        )
        self.layers = self.get_sequential_layers()
        del kwargs["tie_word_embeddings"]
        del kwargs["config"]

        topology = (
            None
            if self.config.pipeline_model_parallel_size == 1
            else fleet.get_hybrid_communicate_group().topology()
        )

        super().__init__(
            layers=self.layers,
            topology=topology,
            num_virtual_pipeline_stages=self.config.virtual_pipeline_model_parallel_size,
            **kwargs,
        )

        if skip_weight_param_allocation:
            shared_embed_weight = None
            for layer in self.run_function:
                if isinstance(layer, GPTEmbedding):
                    shared_embed_weight = layer.embedding_weight
                if isinstance(layer, GPTLMHead):
                    layer.weight = shared_embed_weight

        # MTP embedding weight management
        if (
            self.config.enable_mtp_magic_send
            and self.config.num_nextn_predict_layers > 0
        ):
            # Make all MTP layers on this rank share one physical embedding Parameter.
            self._tie_mtp_embed_weights_intra_rank()
            # Create a cross-PP-stage comm group containing only ranks that hold mtp_embed.
            self._create_mtp_embed_global_group()
            # Broadcast embedding weight from stage 0 to MTP stages so they start identical.
            self._synchronize_mtp_embed_weight()
            # Mark which rank is the "primary" for gradient-clipping dedup.
            self._mark_mtp_embed_shared_flags()
            # Sanity check: MTP layer depths across all PP ranks form 0..N-1.
            self._assert_mtp_depth_contiguous()

        # Tie GPTMTPLMHead weight to GPTMainLMHead weight on the same stage.
        self._tie_mtp_lm_head_weight()
        # Configure DSpark capture when the draft layer is present.
        self._setup_dspark_capture()
        if (
            getattr(self.config, "mtp_shared_weights", False)
            and getattr(self.config, "mtp_shared_last_layer", False)
            and self.config.pipeline_model_parallel_size > 1
        ):
            self._assert_mtp_depths_colocated_for_combined_sharing()
        if getattr(self.config, "mtp_shared_weights", False) and getattr(
            self.config, "mtp_shared_last_layer", False
        ):
            self._alias_mtp_fusion_weights()

    def _assert_mtp_depths_colocated_for_combined_sharing(self):
        """Require combined MTP sharing to keep all depths on one PP stage.

        ``mtp_shared_last_layer`` uses the shared-layer key for the transformer
        body and the backbone-last pivot. ``mtp_shared_weights`` additionally
        aliases the per-depth fusion modules. Those fusion parameters have no
        second SharedLayerDesc key, so the combined mode can only synchronize
        them by rank-local aliasing after construction.
        """
        import paddle.distributed

        local_depths = sorted(
            layer.layer_number for layer in self._get_all_mtp_layers()
        )
        hcg = fleet.get_hybrid_communicate_group()
        gathered = []
        paddle.distributed.all_gather_object(
            gathered, local_depths, group=hcg.get_pipe_parallel_group()
        )
        holders = [depths for depths in gathered if depths]
        expected = list(range(self.config.num_nextn_predict_layers))
        if len(holders) != 1 or sorted(holders[0]) != expected:
            raise RuntimeError(
                "mtp_shared_weights + mtp_shared_last_layer requires all MTP "
                f"depths on one pipeline stage, but the layout is {gathered}. "
                "The transformer body is shared through the backbone pivot, while "
                "the per-depth fusion parameters are aliased within that stage."
            )

    def _alias_mtp_fusion_weights(self):
        """Share non-transformer parameters across MTP depths in this PP stage."""
        mtp_layers = [
            layer
            for layer in self.run_function
            if isinstance(layer, MultiTokenPredictionLayer)
        ]
        if len(mtp_layers) < 2:
            return

        source_params = dict(mtp_layers[0].all_weights)
        fusion_params = {
            name: param
            for name, param in source_params.items()
            if not name.startswith("transformer_layer.")
        }
        for dest_layer in mtp_layers[1:]:
            for name, dest_param in dest_layer.all_weights:
                if name.startswith("transformer_layer."):
                    continue
                source_param = fusion_params.get(name)
                if source_param is None:
                    raise RuntimeError(
                        f"MTP fusion parameter {name!r} is missing at depth "
                        f"{mtp_layers[0].layer_number}"
                    )
                if tuple(source_param.shape) != tuple(dest_param.shape):
                    raise RuntimeError(
                        f"MTP fusion parameter {name!r} has incompatible shapes "
                        f"between depths {mtp_layers[0].layer_number} and "
                        f"{dest_layer.layer_number}: {source_param.shape} vs "
                        f"{dest_param.shape}"
                    )
                parts = name.split(".")
                owner = dest_layer
                for part in parts[:-1]:
                    owner = getattr(owner, part)
                leaf = parts[-1]
                owner._parameters[leaf] = source_param

    def _alias_shared_layer(self, dest_layer, src_layer):
        """Alias a whole MTP layer, not just its ``transformer_layer``.

        paddle's ``PipelineLayer._alias_shared_layer`` hardcodes
        ``dest_layer.transformer_layer`` as the aliasing scope. That is correct for
        ``mtp_shared_last_layer``, whose pivot is a backbone TransformerLayer: the
        pivot's parameter names line up with the names under the MTP body. It is too
        narrow for ``mtp_shared_weights``, where every MTP depth must also share the
        fusion modules (enorm / hnorm / eh_proj or e_proj+h_proj / norm).

        When both sides are MTP layers, pivot and destination are the same class, so
        their ``named_parameters()`` live at the same level and aliasing the whole
        layer is both correct and complete. Everything else falls through to paddle,
        so ``mtp_shared_last_layer`` and the embedding / lm_head keys are untouched.

        Only rank-local Parameter identity is established here; the cross-stage
        broadcast and gradient allreduce come from ``PipelineLayer.shared_comm``,
        which is built from the desc's ``shared_weight_attr`` (``all_weights`` for the
        mtp_shared_weights key, so every parameter is covered).

        The widening is gated on ``config.mtp_shared_weights``, not on the operand
        types alone: under ``mtp_shared_last_layer`` the pivot can be an MTP layer too
        (when the backbone-last pivot lands off-stage), but that key's
        ``shared_weight_attr`` is ``transformer_layer_weights``, so widening there
        would alias fusion params that shared_comm never syncs -- silently divergent
        across stages instead of loudly broken.

        TODO(upstream): this override exists only because paddle's
        ``PipelineLayer._alias_shared_layer`` hardcodes
        ``dest_layer.transformer_layer`` and ignores the desc's
        ``shared_weight_attr``, which it already carries. Once the alias scope is
        driven off ``shared_weight_attr`` upstream, delete this method.
        """
        if not (
            getattr(self.config, "mtp_shared_weights", False)
            and isinstance(dest_layer, MultiTokenPredictionLayer)
            and isinstance(src_layer, MultiTokenPredictionLayer)
        ):
            return super()._alias_shared_layer(dest_layer, src_layer)

        # Drive the alias scope off all_weights so it stays identical to what
        # shared_comm syncs. In particular mtp_embed is excluded there (GPTModel
        # owns it through _tie_mtp_embed_weights_intra_rank and the
        # _mtp_embed_global_group broadcast), and aliasing something shared_comm
        # does not sync is exactly the silent cross-stage divergence this override
        # exists to avoid.
        source_params = dict(src_layer.all_weights)
        dest_named = list(dest_layer.all_weights)
        aliased = missing = shape_mismatch = 0
        for name, dest_param in dest_named:
            src_param = source_params.get(name)
            if src_param is None:
                missing += 1
                continue
            if tuple(src_param.shape) != tuple(dest_param.shape):
                shape_mismatch += 1
                continue
            # Overwrite the entry in the owning module's _parameters dict so the
            # Parameter object identity is shared, leaving the LayerDesc tree (and
            # therefore the per-depth checkpoint keys) intact.
            parts = name.split(".")
            owner = dest_layer
            for p in parts[:-1]:
                owner = getattr(owner, p)
            owner._parameters[parts[-1]] = src_param
            aliased += 1

        # Raise, not assert: ``python -O`` strips assertions, and this guards a
        # correctness-critical invariant. If a spec really diverged, an assert-less
        # run would continue with a partially aliased depth while shared_comm still
        # syncs the full all_weights set -- the silent cross-stage divergence this
        # override exists to prevent.
        if aliased != len(dest_named):
            raise RuntimeError(
                f"MTP depth {dest_layer.layer_number} cannot be fully aliased onto "
                f"depth {src_layer.layer_number}: aliased={aliased}/"
                f"{len(dest_named)}, missing={missing}, "
                f"shape_mismatch={shape_mismatch}. All MTP depths are built from the "
                "same spec, so a mismatch means the specs diverged (e.g. "
                "use_dense_mtp differing across depths)."
            )

    def forward(self, input, chunk_id=None):
        # Keep local metadata out of the scheduler's received-tensor ownership.
        if isinstance(input, dict):
            input = input.copy()
        output = super().forward(input, chunk_id=chunk_id)
        if isinstance(output, dict):
            # The DSA top-k holder is a stage-local dict, not a tensor, so it
            # cannot enter PP transport. Recompute closures keep the holder
            # itself alive until backward finishes.
            output.pop("_dsa_topk_holder", None)
        return output

    def _validate_dsa_pipeline_sharing(self):
        """Reject shared-indexer layouts the top-k holder cannot serve.

        Producer and consumer must share a PP segment, because the holder is
        a stage-local dict that never crosses a PP boundary. Overlap
        schedulers bypass the forward path that owns the holder's lifetime.

        Recompute needs no check, pipelined or not: every micro-batch's forward
        creates its own holder and each recompute closure keeps that object, so
        a consumer replayed after later micro-batches' forwards still reads
        its own micro-batch's top-k. Backward replays the consumer before the
        producer it reads from, so the consumer sees the top-k of the original
        forward. The cost is memory: under recompute each producer's int64
        top-k stays alive until its micro-batch's backward.

        Segmentation is already resolved here, but no local layers have been
        constructed. Inspect the global descriptors so every rank rejects the
        same unsupported layout before allocating model parameters or running
        pipeline collectives.
        """
        from paddlefleet.transformer.dsa_attention import (
            DSAttention,
            decoder_dsa_logical_layer,
            resolve_dsa_indexer_layout,
        )

        producers = {}
        consumers = []
        for index, descriptor in enumerate(self._layers_desc):
            # Fleet versions expose the LayerSpec through either
            # ``layer_func`` or ``layer_spec``. Prefer the former when it is
            # itself a spec, then fall back to the latter; assuming only one
            # spelling silently skips every DSA descriptor on the other API.
            spec = getattr(descriptor, "layer_func", None)
            legacy_spec = getattr(descriptor, "layer_spec", None)
            if legacy_spec is not None and not hasattr(spec, "sublayers_spec"):
                spec = legacy_spec
            sublayers = getattr(spec, "sublayers_spec", None)
            mtp_transformer = getattr(sublayers, "transformer_layer", None)
            if mtp_transformer is not None:
                if self.config.mtp_load_weight_only:
                    continue
                spec = mtp_transformer
                sublayers = spec.sublayers_spec
            attention = getattr(sublayers, "self_attn", None)
            core = getattr(
                getattr(attention, "sublayers_spec", None),
                "core_attention",
                None,
            )
            if getattr(core, "layer", None) is not DSAttention:
                continue

            kwargs = getattr(spec, "extra_kwargs", None)
            if not isinstance(kwargs, dict) or "layer_number" not in kwargs:
                raise ValueError(
                    f"DSA layer descriptor {index} has no layer_number in "
                    "extra_kwargs; pipeline sharing cannot be validated."
                )
            layer_number = kwargs["layer_number"]
            is_mtp = kwargs.get("is_mtp_layer", False)
            _, skip_topk, index_share, source = resolve_dsa_indexer_layout(
                self.config, layer_number, is_mtp
            )
            if index_share and (
                self._use_dualpipev
                or issubclass(spec.layer, TransformerLayerWithOverlap)
            ):
                raise ValueError(
                    "DSA top-k sharing is not supported by overlap pipeline "
                    "schedulers. Disable overlap scheduling or configure full "
                    "indexers."
                )
            segment = bisect_right(self.segment_parts, index) - 1
            if skip_topk:
                consumers.append((layer_number, is_mtp, source, segment))
            elif not is_mtp:
                logical_layer = decoder_dsa_logical_layer(
                    self.config, layer_number
                )
                producers[logical_layer] = segment

        for layer_number, is_mtp, source, segment in consumers:
            kind = "MTP" if is_mtp else "decoder"
            producer_segment = producers.get(source)
            if producer_segment != segment:
                raise ValueError(
                    f"DSA shared {kind} layer {layer_number} is in pipeline "
                    f"segment {segment}, but decoder top-k producer {source} "
                    f"is in segment {producer_segment}. Cross-segment DSA "
                    "top-k sharing is not supported. Co-locate the producer "
                    "and consumer in one segment or configure full indexers."
                )

    def _build_layer(self):
        self._validate_dsa_pipeline_sharing()
        return super()._build_layer()

    def _build_chunked_layer(self):
        self._validate_dsa_pipeline_sharing()
        return super()._build_chunked_layer()

    def _get_weight_only_params(self):
        """Get all parameters marked with is_weight_only_mtp flag."""
        return [
            param
            for param in self.state_dict().values()
            if getattr(param, "is_weight_only_mtp", False)
        ]

    def _iter_local_layers(self):
        """Yield this rank's local layers under any pipeline layout.

        ``PipelineLayer`` builds the flat ``run_function`` only for 1F1B; under
        VPP>1 and DualPipeV the layers live in ``_model_chunks``, and
        ``run_function`` is assigned later -- per chunk, by
        ``update_run_function`` inside ``forward`` -- so at ``__init__`` time it
        does not exist yet, and afterwards it only ever points at the chunk most
        recently executed.

        Do not reach for ``get_model_chunks()`` here: it returns ``None``
        whenever ``_num_virtual_pipeline_stages == 1``, which is exactly the
        DualPipeV case, even though ``_model_chunks`` is populated there.
        """
        chunks = getattr(self, "_model_chunks", None)
        if chunks is not None:
            for chunk in chunks:
                yield from chunk.run_function
        else:
            yield from self.run_function

    def _mark_dspark_target_layers(self, target_ids):
        """Stamp ``_dspark_target_id`` on every target layer this rank holds.

        Deliberately independent of where the draft lives. The draft may sit in
        a different VPP chunk, or on a downstream PP stage, from the layers it
        taps; gating this loop on "the draft is local" would leave those layers
        unmarked, and the omission would only resurface much later as a missing
        ``dspark_target_hidden_<lid>`` key inside the draft.
        """
        for layer in self._iter_local_layers():
            if (
                isinstance(
                    layer,
                    (HyperConnectionTransformerLayer, TransformerLayer),
                )
                and not isinstance(layer, DsparkPredictionLayer)
                and not getattr(layer, "is_mtp_layer", False)
                and getattr(layer, "layer_number", None) in target_ids
            ):
                if type(layer).forward is not TransformerLayer.forward:
                    raise NotImplementedError(
                        f"DSpark target layer {layer.layer_number} is a "
                        f"{type(layer).__name__}, whose forward does not carry "
                        f"the feature tap. Only TransformerLayer / "
                        f"HyperConnectionTransformerLayer / Gemma4 are supported "
                        f"this phase; extend the tap to that variant first."
                    )
                layer._dspark_target_id = int(layer.layer_number)

    def _setup_dspark_capture(self):
        """Mark capture layers and validate DSpark inputs on this stage."""
        # Short-circuit on the canonical predicate before touching any layer
        # container: for interleaved (VPP) and DualPipeV layouts PipelineLayer
        # populates neither ``run_function`` nor (until forward) the chunk
        # sequence, so a model that never asks for DSpark must not walk layers.
        if not dspark_layers_active(self.config):
            return

        target_ids = set(self.config.dspark_target_layer_ids)
        # Mark first, and unconditionally -- see _mark_dspark_target_layers.
        self._mark_dspark_target_layers(target_ids)

        dspark_layer = embedding = None
        norm_layers = []
        present_ids = set()
        for layer in self._iter_local_layers():
            if isinstance(layer, DsparkPredictionLayer):
                dspark_layer = layer
            elif isinstance(layer, GPTEmbedding):
                embedding = layer
            elif isinstance(layer, WrappedPaddleNormPipe):
                norm_layers.append(layer)
            lid = getattr(layer, "_dspark_target_id", None)
            if lid is not None:
                present_ids.add(lid)

        if dspark_layer is None:
            # No draft on this rank; the marking above is all this rank owes.
            return

        final_norm = norm_layers[-1] if norm_layers else None
        if final_norm is not None:
            # Capture only the final norm output.
            final_norm._dspark_is_final_norm = True

        # The embedding is still located here for the host-metadata hook below,
        # but it is no longer handed to the draft: the shared embedding / LM
        # head weights reach the draft through dict_args, published by the
        # SharedLayerDesc pass-throughs declared next to it in
        # get_layer_desc_list. Those descs live in the draft's own stage, so
        # unlike the backbone layers they need no co-location check.
        missing = target_ids - present_ids
        if missing or final_norm is None or embedding is None:
            raise RuntimeError(
                "DSpark requires its target layers / final norm / embedding to "
                "be co-located with the draft in the same stage: "
                f"missing_target_layers={sorted(missing)}, "
                f"final_norm={'present' if final_norm else 'MISSING'}, "
                f"embedding={'present' if embedding else 'MISSING'}."
            )

        def _host_meta_pre_hook(module, args, kwargs):
            # Preserve host metadata for block construction. attn_mask_
            # startend_row_indices rides the same side-channel: it is built
            # upstream from the GLOBAL sequence and is the preferred
            # document-boundary source in get_doc_start_per_pos, with the
            # "startend_row_indices" fallback name mirroring GPTEmbedding's own
            # lookup order.
            dict_args = args[0] if args else kwargs.get("dict_args")
            if dict_args is None:
                return
            startend = dict_args.get("attn_mask_startend_row_indices")
            if startend is None:
                startend = dict_args.get("startend_row_indices")
            dspark_layer._host_meta = {
                "input_ids": dict_args.get("input_ids"),
                "labels": dict_args.get("labels"),
                "attn_mask_startend_row_indices": startend,
            }

        embedding.register_forward_pre_hook(
            _host_meta_pre_hook, with_kwargs=True
        )

    def _tie_mtp_lm_head_weight(self):
        """Tie GPTMTPLMHead.weight to GPTMainLMHead.weight on the same rank."""
        main_head = None
        mtp_head = None
        if (
            self._num_virtual_pipeline_stages > 1
            and hasattr(self, "_model_chunks")
            and self._model_chunks
        ):
            for chunk in self._model_chunks:
                for layer in chunk.run_function:
                    if isinstance(layer, GPTMainLMHead):
                        main_head = layer
                    elif isinstance(layer, GPTMTPLMHead):
                        mtp_head = layer
        else:
            for layer in self.run_function:
                if isinstance(layer, GPTMainLMHead):
                    main_head = layer
                elif isinstance(layer, GPTMTPLMHead):
                    mtp_head = layer
        if main_head is None or mtp_head is None:
            return
        if mtp_head.weight is not main_head.weight:
            mtp_head._parameters["weight"] = main_head.weight

    # ========================================
    def offload_weight_only_params(self):
        """Offload all weight-only MTP parameters to CPU pinned memory."""
        for param in self._get_weight_only_params():
            if param.place.is_gpu_place():
                cpu_param = param.pin_memory()
                cpu_param._share_buffer_to(param)

    def reload_weight_only_params(self):
        """Reload weight-only MTP parameters from CPU pinned memory back to GPU."""
        for param in self._get_weight_only_params():
            if not param.place.is_gpu_place():
                gpu_param = param.cuda()
                gpu_param._share_buffer_to(param)

    def _model_name_prefix(self) -> str:
        """The model single-name root prefix (no trailing dot).

        Shared by :meth:`get_layer_desc_list`, which names the pipeline layers
        with it, and by ``build_aoa_context``, so pipeline naming and AOA name
        resolution never diverge.
        """
        model_type = getattr(self.config, "model_type", "")
        if "qwen3_vl" in model_type or "qwen3_5" in model_type:
            return "model.language_model"
        return "model"

    def get_layer_desc_list(self, spec, tie_word_embeddings):
        layers = []
        name_prefix = self._model_name_prefix()
        # The draft reads the backbone embedding / LM head weights. When a draft
        # is present both are declared as SharedLayerDesc under a stable key, and
        # the draft joins those keys further down (see the DSpark section): that
        # is what makes the sharing framework-managed instead of an instance
        # handed over by hand after construction.
        dspark_specs = getattr(spec, "dspark", None) or []
        has_draft = bool(dspark_specs)
        if tie_word_embeddings:
            embed_key = "embed"
        elif self.config.enable_mtp_magic_send:
            embed_key = "mtp_embed"
        elif has_draft:
            # Promote. A plain LayerDesc registers no key, so the draft's desc
            # below would become the pivot and build a second, randomly
            # initialised embedding instead of aliasing this one.
            embed_key = "embed"
        else:
            embed_key = None
        if embed_key is not None:
            self.add_sequential_layer(
                layers,
                SharedLayerDesc(
                    embed_key,
                    spec.embedding,
                    shared_weight_attr="embedding_weight",
                ),
                name_prefix,
            )
        else:
            self.add_sequential_layer(
                layers, LayerDesc(spec.embedding), name_prefix
            )
        i = 0
        empty_index = 0

        def next_empty_layer_name():
            """Name for the next EmptyLayer.

            Empty (placeholder) layers live in their own ``empty_layers.<n>``
            namespace and do not consume a ``layers.<i>`` slot, so real
            transformer/MTP layers count from ``layers.0`` without the
            head/tail empty-layer offset.
            """
            nonlocal empty_index
            name = f"{name_prefix}.empty_layers.{empty_index}"
            empty_index += 1
            return name

        for head_empty_layer in spec.head_empty_layers:
            self.add_sequential_layer(
                layers, LayerDesc(head_empty_layer), next_empty_layer_name()
            )

        if spec.mhc_expand is not None:
            self.add_sequential_layer(
                layers, LayerDesc(spec.mhc_expand), f"{name_prefix}.mhc_expand"
            )

        for idx, transformer_layer_spec in enumerate(spec.transformer_layers):
            is_last = idx == len(spec.transformer_layers) - 1
            if (
                is_last
                and getattr(self.config, "mtp_shared_last_layer", False)
                and spec.mtp
            ):
                # In the combined mode this key shares each MTP body's
                # transformer_layer with the backbone pivot. The fusion parameters
                # are aliased across MTP depths after construction.
                desc = SharedLayerDesc(
                    "mtp_reuse_transformer",
                    transformer_layer_spec,
                    shared_weight_attr="transformer_layer_weights",
                )
            else:
                desc = LayerDesc(transformer_layer_spec)
            self.add_sequential_layer(
                layers,
                desc,
                f"{name_prefix}.layers.{i}",
            )
            i += 1

        if spec.mhc_contract is not None:
            self.add_sequential_layer(
                layers,
                LayerDesc(spec.mhc_contract),
                f"{name_prefix}.mhc_contract",
            )

        # Always place layer_norm after transformer_layers and before tail_empty_layers/MTP,
        # so that the model structure is consistent regardless of whether MTP is enabled.
        if not (
            self.config.gpt_model_use_experimental_version
            and self.config.num_nextn_predict_layers >= 1
        ):
            if spec.output_block_attn_res is not None:
                self.add_sequential_layer(
                    layers,
                    LayerDesc(spec.output_block_attn_res),
                    f"{name_prefix}.output_attn_res",
                )
            self.add_sequential_layer(
                layers, LayerDesc(spec.layer_norm), name_prefix
            )

        if spec.mtp:
            for mtp_spec in spec.mtp:
                # NOTE: test mtp_shared_last_layer FIRST. It used to sit behind an
                # `enable_mtp_magic_send` branch that short-circuited to a plain
                # LayerDesc, so with both flags on the tie silently did nothing.
                # The two are orthogonal -- magic send owns `mtp_embed` (synced
                # via _mtp_embed_global_group below) while SharedLayerDesc(
                # shared_submodule_weight_only=True) aliases only the params
                # under `transformer_layer` -- and TransformerConfig no longer
                # rejects the combination, so the order matters for real.
                #
                # The pivot for this key is emitted above, on the last backbone
                # TransformerLayer; aliasing needs both on the same rank (see the
                # co-location note in transformer_config.py). All K descs reuse
                # the one key, so at K > 1 an off-stage pivot aborts the build
                # rather than degrading to broadcast -- depth 0 becomes the stored
                # shared layer and depth 1 aliases against its
                # `transformer_layer.`-prefixed names.
                #
                # The if/elif below picks which shared KEY this desc carries, not
                # which flag applies at which depth count: paddle keys
                # shared_layers by the desc's single layer_name, so one desc
                # cannot join both groups. With both flags on the backbone key
                # wins here and the per-depth fusion params are aliased afterwards
                # by _alias_mtp_fusion_weights. The flags are orthogonal, with
                # different pivots and scopes -- see
                # TransformerConfig.mtp_shared_weights.
                if getattr(self.config, "mtp_shared_last_layer", False):
                    desc = SharedLayerDesc(
                        "mtp_reuse_transformer",
                        mtp_spec,
                        shared_submodule_weight_only=True,
                        shared_weight_attr="transformer_layer_weights",
                    )
                elif getattr(self.config, "mtp_shared_weights", False):
                    desc = SharedLayerDesc(
                        "mtp_shared_all",
                        mtp_spec,
                        shared_submodule_weight_only=True,
                        shared_weight_attr="all_weights",
                    )
                else:
                    desc = LayerDesc(mtp_spec)
                self.add_sequential_layer(
                    layers, desc, f"{name_prefix}.layers.{i}"
                )
                i += 1

        # multimax
        multimax_mode = getattr(self.config, "multimax_modules", None) or []
        use_multimax_lmhead = "lm_head" in multimax_mode
        if use_multimax_lmhead and getattr(
            self.config, "separate_mtp_headloss", False
        ):
            lmhead_shared_weights = [
                "embedding_weight",
                "multimax_ranges",
                "multimax_ts",
            ]
        else:
            lmhead_shared_weights = ["embedding_weight"]

        if spec.mtp_lm_head:
            self.add_sequential_layer(
                layers,
                SharedLayerDesc(
                    "embed",
                    spec.mtp_lm_head,
                    shared_weight_attr=lmhead_shared_weights,
                ),
                f"{name_prefix}.shared_mtp_lm_head",
            )
        if spec.mtp_loss:
            self.add_sequential_layer(
                layers, LayerDesc(spec.mtp_loss), f"{name_prefix}.mtp_loss"
            )

        # The LM head's key is resolved here, before the DSpark section below,
        # because the draft's head pass-through desc needs it.
        if tie_word_embeddings or spec.mtp_lm_head:
            head_key = "embed"
        elif has_draft:
            # Deliberately distinct from embed_key: paddle keeps only the first
            # instance registered per key and reuses it for every later desc, so
            # one key for both would make the LM head position execute the
            # embedding. The guard below rejects the configuration that would
            # collide.
            head_key = "dspark_lm_head"
        else:
            head_key = None
        if has_draft and head_key == embed_key:
            raise NotImplementedError(
                "DSpark cannot share the embedding and the LM head under one "
                f"SharedLayerDesc key ({head_key!r}): paddle keeps only the "
                "first instance per key, so the LM head position would run the "
                "embedding. That needs tie_word_embeddings with "
                "pipeline_model_parallel_size > 1 or separate_mtp_headloss, "
                "neither of which DSpark's configuration validation allows."
            )

        # ``getattr`` keeps duck-typed test specs (which predate these slots)
        # working; production always passes a full GPTSublayersSpec.
        if has_draft:
            # Declare the draft's use of the backbone embedding / LM head as
            # shared layers instead of poking the instances onto the draft after
            # construction. Each desc reuses the backbone's key, so paddle hands
            # the pass-through the *same* instance the backbone uses, and the
            # draft picks the weights up from dict_args. The partial is one
            # pipeline step that computes nothing.
            #
            # Both descs sit here, before the draft: the pivot for a key is the
            # first desc that registers it, and the draft consumes the weights
            # during its own forward. The embedding's pivot is therefore still
            # the backbone desc above (same key), while the LM head is *built*
            # here and the backbone's own head desc below reuses this instance.
            self.add_sequential_layer(
                layers,
                SharedLayerDesc(
                    embed_key,
                    spec.embedding,
                    forward_func=dspark_share_embedding,
                    shared_weight_attr="embedding_weight",
                ),
                name_prefix,
            )
            self.add_sequential_layer(
                layers,
                SharedLayerDesc(
                    head_key,
                    spec.lm_head,
                    forward_func=dspark_share_lm_head,
                    shared_weight_attr=lmhead_shared_weights,
                ),
                f"{name_prefix}.shared_head",
            )
        for dspark_spec in dspark_specs:
            desc = LayerDesc(dspark_spec)
            self.add_sequential_layer(layers, desc, f"{name_prefix}.layers.{i}")
            i += 1

        dspark_loss_spec = getattr(spec, "dspark_loss", None)
        if dspark_loss_spec:
            self.add_sequential_layer(
                layers,
                LayerDesc(dspark_loss_spec),
                f"{name_prefix}.dspark_loss",
            )

        for tail_empty_layer in spec.tail_empty_layers:
            self.add_sequential_layer(
                layers, LayerDesc(tail_empty_layer), next_empty_layer_name()
            )

        if (
            self.config.gpt_model_use_experimental_version
            and self.config.num_nextn_predict_layers >= 1
        ):
            if spec.output_block_attn_res is not None:
                self.add_sequential_layer(
                    layers,
                    LayerDesc(spec.output_block_attn_res),
                    f"{name_prefix}.output_attn_res",
                )
            self.add_sequential_layer(
                layers, LayerDesc(spec.layer_norm), name_prefix
            )
        if head_key is not None:
            self.add_sequential_layer(
                layers,
                SharedLayerDesc(
                    head_key,
                    spec.lm_head,
                    shared_weight_attr=lmhead_shared_weights,
                ),
                f"{name_prefix}.shared_head",
            )
        else:
            self.add_sequential_layer(
                layers, LayerDesc(spec.lm_head), f"{name_prefix}.lm_head"
            )

        return layers

    def overlapped_forward_backward(
        self,
        forward_chunk,
        forward_inputs,
        forward_loss_fn_node,
        backward_chunk,
        backward_loss_fn_node,
        backward_input_grads,
        scaler,
        p2p_async_handle,
    ):
        if backward_loss_fn_node is not None:
            if scaler:
                backward_input_grads = backward_loss_fn_node.backward(
                    scaler=scaler
                )
            else:
                backward_input_grads = backward_loss_fn_node.backward()

        (
            forward_pre_node,
            backward_pre_node,
            overlap_node,
            forward_post_node,
            backward_post_node,
        ) = build_overlapped_nodes(forward_chunk, backward_chunk)

        if len(overlap_node.nodes) > 0:
            assert not any(
                isinstance(node, TransformerLayerNode)
                for node in overlap_node.nodes
            )
            # origin assert, why ?
            # assert not any(
            #     isinstance(node, TransformerLayerNode)
            #     for node in forward_post_node.nodes
            # )
            # assert not any(
            #     isinstance(node, TransformerLayerNode)
            #     for node in backward_post_node.nodes
            # )

        if p2p_async_handle is not None:
            p2p_async_handle.forward_handle_wait()
            p2p_async_handle.backward_handle_wait()

        forward_inputs = forward_pre_node.forward(forward_inputs)
        backward_input_grads = backward_pre_node.backward(backward_input_grads)

        for i, node in enumerate(overlap_node.nodes):
            forward_inputs, backward_input_grads = node.forward_backward(
                forward_inputs,
                backward_input_grads,
                # split_bw=(i == len(overlap_node.nodes) - 1),
            )

        forward_inputs = forward_post_node.forward(forward_inputs)
        backward_input_grads = backward_post_node.backward(backward_input_grads)

        # forward_inputs = forward_chunk.forward(forward_inputs)

        if p2p_async_handle is not None:
            forward_inputs = dict_to_tuple_helper(forward_inputs)
            p2p_async_handle.forward_async_comm(forward_inputs)
            p2p_async_handle.backward_async_comm(backward_input_grads)

        # backward_input_grads = backward_chunk.backward(backward_input_grads)

        # used for bw split
        # if len(overlap_node.nodes) > 0:
        #     WeightGradStore.pop()
        #     assert WeightGradStore.funcs_queue.empty()

        if forward_loss_fn_node is not None:
            forward_loss = forward_loss_fn_node.forward(forward_inputs)
        else:
            forward_loss = None

        return forward_inputs, forward_loss, backward_input_grads

    def get_hardware_flops(self):
        return 989e3

    def add_sequential_layer(self, layers, layer_desc, name_prefix=""):
        """
        Add a sequential layer to the network with specified description and name prefix.

        Args:
            layers (list): List to store layer descriptions. Each element should be a dict
                with keys "layer" (LayerDesc) and "name_prefix" (str).
            layer_desc (LayerDesc|SharedLayerDesc): Layer description object containing
                layer self.configuration.
            name_prefix (str, optional): Prefix for layer names in the pipeline.
                Defaults to empty string.

        Returns:
            None: The layer description is appended to the input layers list.
        """
        layers.append({"layer": layer_desc, "name_prefix": name_prefix})

    def get_sequential_layers(self):
        """
        Get all layers in the sequential network.

        Returns:
            List[paddle.nn.Layer]: List containing all layers.
        """
        return [x["layer"] for x in self._sequential_layers]

    def get_sequential_name_prefixes(self):
        """
        Retrieve name prefixes for all parallel layers in the sequential network.

        Returns:
            Dict[str, str]: A dictionary mapping layer indices (as strings) to their
                corresponding name prefixes. The indices represent the position of
                each layer in the sequential order.
        """
        return {
            str(index): x["name_prefix"]
            for index, x in enumerate(self._sequential_layers)
        }

    def get_shardlayer_prefix(self, name_splited):
        """_summary_
            This function retrieves the prefix of a shared layer. The process involves:
            1. Identifying all key names of shared layers, like 'shared_weight01', 'shared_weight02', etc.
            2. For instance, given name_splited = ['shared_layers', 'shared_weight01', 'weight'],
                the 'shared_layer_key' would be name_splited[1], which is 'shared_weight01'.
            3. By traversing through all layers, the function checks if the specified
                shared_layer is present in the current stage. If found, it returns the corresponding prefix.

            Note: For retrieving all SharedLayer instances in Paddle, you can refer to the following Paddle code.
            https://github.com/PaddlePaddle/Paddle/blob/2cf724d055679a1a0e48766dfb1708b920273078/python/paddle/distributed/fleet/meta_parallel/parallel_layers/pp_layers.py#L460-L513
        Args:
            name_splited (_type_): _description_

        Returns:
            _type_: _description_
        """
        shared_layer_names = {
            s.layer_name for s in self.layers if isinstance(s, SharedLayerDesc)
        }
        assert name_splited[1] in shared_layer_names, (
            f"The shared layer name {name_splited[1]} must be in prefixes!"
        )
        shared_layer_key = name_splited[1]
        for idx, layer in enumerate(self.layers):
            if (
                isinstance(layer, SharedLayerDesc)
                and layer.layer_name == shared_layer_key
            ):
                if self.get_stage_from_index(idx) == self._stage_id:
                    return self.get_sequential_name_prefixes()[str(idx)]

        # the prefix must be in the current stage, else raise error
        raise ValueError(
            f"The shared layer {shared_layer_key} must be in the current stage!"
        )

    def _set_pipeline_name_mapping(self, mappings=None):
        """
        Set the name mapping for pipeline.

        Args:
            mappings (dict, optional): Dictionary storing name mapping relationships. Default is None, meaning no mapping operation.

        Returns:
            dict: Returns the updated or existing mapping relationship.

        """
        if mappings is not None:
            self._pipeline_name_mapping = mappings
        else:
            single_to_pp_mapping = {}
            pp_to_single_mapping = {}

            state_dict_keys = list(super().state_dict().keys())

            # Whether the layers are chunked is a property of the model, not
            # something the key shapes can tell: a chunk key is
            # `{chunk_start}.{local_idx}.xxx`, but an ordinary PP
            # `LayerDesc(nn.Sequential, ...)` also yields
            # `{global_idx}.{sublayer_idx}.xxx`, and conversely the first key of
            # a chunked stage may be a shared layer alias or a directly added
            # layer, both of which keep a non digit second segment. Ask the
            # pipeline layer itself; dualpipev chunks the layers as well.
            use_virtual_pp_degree = (
                self._num_virtual_pipeline_stages > 1 or self._use_dualpipev
            )

            prefixes = self.get_sequential_name_prefixes()
            shared_layer_names = {
                layer.layer_name
                for layer in self.layers
                if isinstance(layer, SharedLayerDesc)
            }
            for k in state_dict_keys:
                # Skip vision_merge.* keys - they are handled separately
                if is_vision_merge_key(k):
                    continue
                name_splited = k.split(".")
                if use_virtual_pp_degree:
                    if name_splited[0].isdigit():
                        if name_splited[1].isdigit():
                            idx = str(
                                int(name_splited[0]) + int(name_splited[1])
                            )
                            single_name = [prefixes[idx]]
                            single_name.extend(name_splited[2:])
                        elif name_splited[1] in shared_layer_names:
                            # A SharedLayerDesc with `forward_func` is
                            # registered on the chunk itself under VPP, so its
                            # key is `{chunk_start}.{shared_name}.rest`. It
                            # aliases the same parameter as
                            # `shared_layers.{shared_name}.rest` and must
                            # resolve to the same single card name.
                            single_name = [
                                self.get_shardlayer_prefix(name_splited)
                            ]
                            single_name.extend(name_splited[2:])
                        else:
                            # Layers directly added to the PipelineLayer under
                            # VPP (e.g. lm_head) are named `{global_idx}.rest`
                            # instead of `{chunk_start}.{local_idx}.rest`, so
                            # the first segment is already the global index.
                            # Resolve them per layer like the non-VPP branch,
                            # otherwise every such key collapses onto the last
                            # layer prefix, drops its submodule name and
                            # collides with its siblings.
                            idx = name_splited[0]
                            single_name = (
                                [] if prefixes[idx] == "" else [prefixes[idx]]
                            )
                            single_name.extend(name_splited[1:])
                    elif name_splited[0] == "shared_layers":
                        single_name = [self.get_shardlayer_prefix(name_splited)]
                        single_name.extend(name_splited[2:])
                    else:
                        single_to_pp_mapping[k] = k
                        pp_to_single_mapping[k] = k
                        continue
                else:
                    idx = name_splited[0]
                    # for normal pp layer
                    if idx.isdigit():
                        # allow empty prefix
                        single_name = (
                            [] if prefixes[idx] == "" else [prefixes[idx]]
                        )
                        single_name.extend(name_splited[1:])
                    elif idx == "shared_layers":
                        single_name = [self.get_shardlayer_prefix(name_splited)]
                        single_name.extend(name_splited[2:])
                    else:
                        single_to_pp_mapping[k] = k
                        pp_to_single_mapping[k] = k
                        continue

                single_to_pp_mapping[".".join(single_name)] = k
                pp_to_single_mapping[k] = ".".join(single_name)

            self._pipeline_name_mapping = single_to_pp_mapping
            self._pp_to_single_mapping = pp_to_single_mapping

        return self._pipeline_name_mapping

    def state_dict(self, *args, **kwargs):
        """
        Return a dictionary with Pipeline Stage mapping.
        Args:
            *args (tuple): Variable argument list passed to parent method.
            **kwargs (dict): Optional keyword arguments passed to parent method.
        Returns:
            dict: Dictionary containing Pipeline Stage mapping.
        """
        state_dict = super().state_dict(*args, **kwargs)

        model_type = getattr(self.config, "model_type", "")
        if "qwen3_vl" in model_type or "qwen3_5" in model_type:
            name_prefix = "model.language_model."
        else:
            name_prefix = ""
        if self._pipeline_name_mapping is None:
            self._set_pipeline_name_mapping()

        # Remove the vision encoder's keys; they are re-added with the proper
        # mapping below. ``is_vision_merge_key`` also rejects parameters placed
        # directly on the wrapper, which would otherwise vanish silently.
        vision_merge_keys = [
            k for k in state_dict.keys() if is_vision_merge_key(k)
        ]
        for k in vision_merge_keys:
            state_dict.pop(k)

        # assert len(self._pipeline_name_mapping) > 0, "The pipeline stage must have parameters!"
        for k in list(state_dict.keys()):
            v = state_dict.pop(k)
            if name_prefix and k.startswith(name_prefix):
                k = k[len(name_prefix) :]
            if k not in self._pp_to_single_mapping:
                state_dict[k] = v
                continue
            v.key = self._pp_to_single_mapping[k]
            state_dict[self._pp_to_single_mapping[k]] = v

        # Re-add vision model keys with proper mapping
        if hasattr(self, "vision_merge") and self.vision_merge is not None:
            vision_model = self.vision_merge.vision_model
            if hasattr(vision_model, "state_dict"):
                vm_state = vision_model.state_dict()
                state_dict.update(vm_state)

        return state_dict

    def set_state_dict(self, state_dict, *args, **kwargs):
        if self._pipeline_name_mapping is None:
            self._set_pipeline_name_mapping()
        assert len(self._pipeline_name_mapping) > 0, (
            "The pipeline stage must have parameters!"
        )

        # Separate vision model keys for vision_merge sublayer
        vision_state = {}
        if hasattr(self, "vision_merge") and self.vision_merge is not None:
            for k in list(state_dict.keys()):
                if k.startswith("model.vision_model."):
                    vision_state[k] = state_dict.pop(k)
            if not vision_state:
                logger.warning(
                    "This stage owns a vision_merge sublayer but the state dict "
                    "has no 'model.vision_model.*' keys, so the vision encoder "
                    "keeps its initial weights. Check that the checkpoint is a "
                    "multimodal one."
                )

        for k in list(state_dict.keys()):
            v = state_dict.pop(k)
            if k not in self._pipeline_name_mapping:
                continue
            state_dict[self._pipeline_name_mapping[k]] = v

        # Load vision model state into vision_merge.vision_model
        if (
            vision_state
            and hasattr(self, "vision_merge")
            and self.vision_merge is not None
        ):
            vision_model = self.vision_merge.vision_model
            if hasattr(vision_model, "set_state_dict"):
                vision_model.set_state_dict(vision_state)

        ret = super().set_state_dict(state_dict, *args, **kwargs)
        return ret

    def _check_shared_model_state(self):
        if self._pipeline_name_mapping is None:
            self._set_pipeline_name_mapping()

        super_state_dict = super().state_dict()
        structure_name_to_tensor = {}
        for k, v in super_state_dict.items():
            # Skip vision_merge.* keys - handled separately
            if is_vision_merge_key(k):
                continue
            k = self._pp_to_single_mapping[k]
            if k not in structure_name_to_tensor:
                structure_name_to_tensor[k] = v
            else:
                old_v = structure_name_to_tensor[k]
                assert old_v is v, (
                    f"Shared tensor with different structure name: {k}"
                )

        missing_shared_keys = {}
        for k, v in self._pp_to_single_mapping.items():
            mapped_k = self._pipeline_name_mapping[v]
            if k != mapped_k:
                missing_shared_keys[k] = mapped_k
        return missing_shared_keys

    def sharded_state_dict(self, *args, **kwargs):
        """
        sharded_state_dict method for PipelinePretrainedModel.

        Remaps parameter keys according to the pipeline stage mapping, and converts expert indices from local to global.
        """
        sharded_state_dict = super().sharded_state_dict(*args, **kwargs)
        if self._pipeline_name_mapping is None:
            self._set_pipeline_name_mapping()

        model_type = getattr(self.config, "model_type", "")
        if "qwen3_vl" in model_type or "qwen3_5" in model_type:
            name_prefix = "model.language_model."
        else:
            name_prefix = ""

        # For qwen3_5 PP mode: vision_merge is added as sublayer of GPTModel,
        # so its keys appear as "vision_merge.*" in super().sharded_state_dict().
        # Remove them here; we'll re-add properly remapped keys from
        # vision_model.sharded_state_dict() below.
        vision_merge_keys = [
            k for k in sharded_state_dict.keys() if is_vision_merge_key(k)
        ]
        for k in vision_merge_keys:
            sharded_state_dict.pop(k)

        for k in list(sharded_state_dict.keys()):
            v = sharded_state_dict.pop(k)
            # remove name_prefix
            if name_prefix and k.startswith(name_prefix):
                k = k[len(name_prefix) :]
            if k not in self._pp_to_single_mapping:
                sharded_state_dict[k] = v
                continue
            v.key = self._pp_to_single_mapping[k]
            sharded_state_dict[self._pp_to_single_mapping[k]] = v

        # For qwen3_5 PP mode: get properly remapped vision model keys
        # from the VisionModel's own sharded_state_dict (which uses
        # _pp_to_single_mapping to produce "model.vision_model.*" keys).
        if hasattr(self, "vision_merge") and self.vision_merge is not None:
            vision_model = self.vision_merge.vision_model
            if hasattr(vision_model, "sharded_state_dict"):
                vm_sharded = vision_model.sharded_state_dict(
                    structured_name_prefix=""
                )
                sharded_state_dict.update(vm_sharded)

        def increment_expert_number(s, increment):
            import re

            def replace(match):
                original_number = int(match.group(0))
                new_number = original_number + increment
                return str(new_number)

            return re.sub(r"(?<=experts\.)\d+", replace, s)

        renamed_sharded_state_dict = {}
        for k, v in sharded_state_dict.items():
            global_expert_id_offset = getattr(
                v, "global_expert_id_offset", None
            )
            layer_cnt = getattr(v, "layer_cnt", None)
            if global_expert_id_offset is not None:
                new_key = increment_expert_number(k, global_expert_id_offset)
                v.key = new_key
                delattr(v, "global_expert_id_offset")
                renamed_sharded_state_dict[new_key] = v
            elif layer_cnt is not None:
                new_key = k + "_layer_" + str(layer_cnt)
                v.key = new_key
                delattr(v, "layer_cnt")
                renamed_sharded_state_dict[new_key] = v
            else:
                renamed_sharded_state_dict[k] = v

        return renamed_sharded_state_dict

    def fp8_quant_weight(self, batch_mode=False, quant_transpose=True):
        if self._num_virtual_pipeline_stages > 1:
            for idx, chunk in enumerate(self._model_chunks):
                for idx, layer in enumerate(chunk):
                    if isinstance(layer, TransformerLayer):
                        layer.fp8_quant_weight(
                            batch_mode=batch_mode,
                            quant_transpose=quant_transpose,
                        )
                    elif isinstance(layer, MultiTokenPredictionLayer):
                        layer.transformer_layer.fp8_quant_weight(
                            batch_mode=batch_mode,
                            quant_transpose=quant_transpose,
                        )
                    elif isinstance(layer, DsparkPredictionLayer):
                        # The draft blocks live in ``layer.layers``, not in
                        # run_function, so they must be driven explicitly or
                        # their routed-expert weights never get the offline fp8
                        # stack -- while clear_param_storage("moe_expert") still
                        # frees their bf16 storage.
                        for sub in layer.layers:
                            sub.fp8_quant_weight(
                                batch_mode=batch_mode,
                                quant_transpose=quant_transpose,
                            )
        else:
            for idx, layer in enumerate(self.run_function):
                if isinstance(layer, TransformerLayer):
                    layer.fp8_quant_weight(
                        batch_mode=batch_mode, quant_transpose=quant_transpose
                    )
                elif isinstance(layer, MultiTokenPredictionLayer):
                    layer.transformer_layer.fp8_quant_weight(
                        batch_mode=batch_mode, quant_transpose=quant_transpose
                    )
                elif isinstance(layer, DsparkPredictionLayer):
                    for sub in layer.layers:
                        sub.fp8_quant_weight(
                            batch_mode=batch_mode,
                            quant_transpose=quant_transpose,
                        )

    def clear_fp8_quant_weight(self):
        if self._num_virtual_pipeline_stages > 1:
            for idx, chunk in enumerate(self._model_chunks):
                for idx, layer in enumerate(chunk):
                    if isinstance(layer, TransformerLayer):
                        layer.clear_fp8_quant_weight()
                    elif isinstance(layer, MultiTokenPredictionLayer):
                        layer.transformer_layer.clear_fp8_quant_weight()
                    elif isinstance(layer, DsparkPredictionLayer):
                        for sub in layer.layers:
                            sub.clear_fp8_quant_weight()
        else:
            for idx, layer in enumerate(self.run_function):
                if isinstance(layer, TransformerLayer):
                    layer.clear_fp8_quant_weight()
                elif isinstance(layer, MultiTokenPredictionLayer):
                    layer.transformer_layer.clear_fp8_quant_weight()
                elif isinstance(layer, DsparkPredictionLayer):
                    for sub in layer.layers:
                        sub.clear_fp8_quant_weight()

    def use_fp8(self):
        if self._num_virtual_pipeline_stages > 1:
            for idx, chunk in enumerate(self._model_chunks):
                for idx, layer in enumerate(chunk):
                    if isinstance(layer, TransformerLayer) and layer.use_fp8():
                        return True
        else:
            for idx, layer in enumerate(self.run_function):
                if isinstance(layer, TransformerLayer) and layer.use_fp8():
                    return True
            return False

    # ═══════════════════════════════════════════════════════════════════════════
    # MTP Magic Send: embedding weight management
    # ═══════════════════════════════════════════════════════════════════════════

    def _get_all_mtp_layers(self):
        """Traverse all virtual chunks and collect MultiTokenPredictionLayer instances."""
        layers = []
        if (
            self._num_virtual_pipeline_stages > 1
            and hasattr(self, "_model_chunks")
            and self._model_chunks
        ):
            for chunk in self._model_chunks:
                for layer in chunk.run_function:
                    if isinstance(layer, MultiTokenPredictionLayer):
                        layers.append(layer)
        else:
            for layer in self.run_function:
                if isinstance(layer, MultiTokenPredictionLayer):
                    layers.append(layer)
        return layers

    def _get_mtp_embed_primary_weight(self):
        """Get the primary (shared) Parameter for mtp_embed weight on this rank."""
        if "mtp_embed" in self.shared_layers:
            return self.shared_layers["mtp_embed"].embedding_weight
        mtp_layers = self._get_all_mtp_layers()
        if mtp_layers:
            return mtp_layers[0].mtp_embed.weight
        return None

    def _tie_mtp_embed_weights_intra_rank(self):
        """Tie all MTP layers' mtp_embed.weight to the same primary Parameter on this rank."""
        mtp_layers = self._get_all_mtp_layers()
        if not mtp_layers:
            return
        if "mtp_embed" in self.shared_layers:
            primary_weight = self.shared_layers["mtp_embed"].embedding_weight
        else:
            primary_weight = mtp_layers[0].mtp_embed.weight
        for layer in mtp_layers:
            if (
                layer.mtp_embed is not None
                and layer.mtp_embed.weight is not primary_weight
            ):
                layer.mtp_embed._parameters["weight"] = primary_weight

        # Verify weight tie
        primary = self._get_mtp_embed_primary_weight()
        for layer in mtp_layers:
            if layer.mtp_embed is not None:
                assert layer.mtp_embed.weight is primary, (
                    f"MTP embed weight tie failed for layer_number={layer.layer_number}: "
                    f"weight id {id(layer.mtp_embed.weight)} != primary {id(primary)}"
                )
        if "mtp_embed" in self.shared_layers:
            assert primary is self.shared_layers["mtp_embed"].embedding_weight

    def _create_mtp_embed_global_group(self):
        """Create a communication group containing only PP ranks that hold mtp_embed weight.

        All ranks in the world must call new_group in the same order to keep
        gid counters synchronized.
        """
        import paddle.distributed

        hcg = fleet.get_hybrid_communicate_group()
        pipe_group = hcg.get_pipe_parallel_group()
        global_rank = paddle.distributed.get_rank()

        has_weight = self._get_mtp_embed_primary_weight() is not None
        gathered = []
        paddle.distributed.all_gather_object(
            gathered, has_weight, group=pipe_group
        )

        pipe_ranks = list(pipe_group.ranks)
        mtp_ranks = [r for r, has in zip(pipe_ranks, gathered) if has]

        # Gather globally to ensure all ranks call new_group in the same order
        world_size = paddle.distributed.get_world_size()
        all_mtp_ranks_lists = []
        paddle.distributed.all_gather_object(all_mtp_ranks_lists, mtp_ranks)

        # Deduplicate
        seen = set()
        unique_mtp_groups = []
        for ranks_list in all_mtp_ranks_lists:
            key = tuple(sorted(ranks_list))
            if key not in seen and len(key) > 1:
                seen.add(key)
                unique_mtp_groups.append(sorted(ranks_list))

        unique_mtp_groups.sort()

        self._mtp_embed_global_group = None
        for ranks_list in unique_mtp_groups:
            group = paddle.distributed.new_group(ranks=ranks_list)
            if global_rank in ranks_list:
                self._mtp_embed_global_group = group

        self._has_mtp_embed_weight = has_weight

    def _synchronize_mtp_embed_weight(self):
        """Broadcast embedding weight from stage 0 to all PP ranks that hold MTP layers."""
        import paddle
        import paddle.distributed

        hcg = fleet.get_hybrid_communicate_group()
        pipe_group = hcg.get_pipe_parallel_group()
        src_rank = hcg.get_rank_from_stage(0)
        primary_weight = self._get_mtp_embed_primary_weight()
        if primary_weight is not None:
            with paddle.no_grad():
                paddle.distributed.broadcast(
                    primary_weight, src=src_rank, group=pipe_group
                )
        else:
            # Intermediate stage: participate in broadcast with a dummy buffer
            # Match VocabParallelEmbedding exactly: it partitions the configured
            # vocabulary directly and does not apply make_vocab_size_divisible_by.
            tp = self.config.tensor_model_parallel_size
            if self.config.vocab_size % tp != 0:
                raise ValueError(
                    f"vocab_size={self.config.vocab_size} must be divisible by "
                    f"tensor_model_parallel_size={tp}"
                )
            local_vocab = self.config.vocab_size // tp
            dtype = (
                self.config.params_dtype
                if hasattr(self.config, "params_dtype")
                else paddle.float32
            )
            if isinstance(dtype, str):
                dtype = getattr(paddle, dtype, paddle.float32)
            recv_buf = paddle.zeros(
                [local_vocab, self.config.hidden_size], dtype=dtype
            )
            with paddle.no_grad():
                paddle.distributed.broadcast(
                    recv_buf, src=src_rank, group=pipe_group
                )

    def _mark_mtp_embed_shared_flags(self):
        """Set is_firstly_shared flag for gradient clipping deduplication."""
        primary_weight = self._get_mtp_embed_primary_weight()
        if primary_weight is None:
            return
        is_pivot = "mtp_embed" in self.shared_layers
        primary_weight.is_firstly_shared = is_pivot

    def _assert_mtp_depth_contiguous(self):
        """Assert that MTP layer depths are contiguous 0..N-1 across all PP ranks."""
        import paddle.distributed

        mtp_layers = self._get_all_mtp_layers()
        local_depths = sorted(layer.layer_number for layer in mtp_layers)
        assert len(local_depths) == len(set(local_depths)), (
            f"duplicate MTP layer_number on this rank: {local_depths}"
        )
        hcg = fleet.get_hybrid_communicate_group()
        pipe_group = hcg.get_pipe_parallel_group()
        gathered = []
        paddle.distributed.all_gather_object(
            gathered, local_depths, group=pipe_group
        )
        all_depths = sorted(d for sub in gathered for d in sub)
        N = self.config.num_nextn_predict_layers
        assert all_depths == list(range(N)), (
            f"MTP depths not contiguous 0..{N - 1}: got {all_depths}"
        )

    def allreduce_shared_weight_gradients(self):
        """Override: mtp_embed uses a dedicated sub-group allreduce;
        other shared layers keep the framework's original pairwise logic.
        """
        import paddle
        import paddle.distributed
        from paddle.framework import core

        def _attached_grad(param):
            """Return the grad buffer to reduce, allocating a zero one if absent.

            Dygraph-only, like the whole PP backward path this is called from.
            The base class also carries a non-dygraph branch, but it is fluid-era
            dead code (it drives ``_dygraph_tracer().trace_op``), and raising
            there instead of reducing a zero buffer would desynchronize the PP
            group's collectives into a hang.
            """
            if hasattr(param, "main_grad"):
                if param.main_grad is None:
                    param.main_grad = core.eager.Tensor(
                        value=paddle.zeros_like(param, dtype="float32").value(),
                        place=param.place,
                        name="main_grad@" + param.name,
                    )
                return param.main_grad
            if param.grad is None:
                param.grad = core.eager.Tensor(
                    value=paddle.zeros_like(param).value(),
                    place=param.place,
                    name="grad@" + param.name,
                )
            return param.grad

        mtp_embed_weight = None
        if hasattr(self, "_mtp_embed_global_group"):
            mtp_embed_weight = self._get_mtp_embed_primary_weight()
        mtp_embed_param_id = (
            id(mtp_embed_weight) if mtp_embed_weight is not None else None
        )
        reduced_param_groups = set()

        # Pairwise allreduce for other shared layers (skip mtp_embed by Parameter
        # identity, and reduce aliases once per communication group).
        for key, comm in self.shared_comm.items():
            for weight_attr in comm["weight_attr"]:
                obj = getattr(comm["layer"], weight_attr)
                params = (
                    [("", obj)] if isinstance(obj, paddle.Tensor) else list(obj)
                )
                for _, param in params:
                    param_id = id(param)
                    param_group_key = (param_id, id(comm["group"]))
                    if (
                        param_id == mtp_embed_param_id
                        or param_group_key in reduced_param_groups
                    ):
                        continue
                    grad = _attached_grad(param)
                    with paddle.framework.no_grad():
                        paddle.distributed.all_reduce(
                            grad.contiguous(), group=comm["group"]
                        )
                    reduced_param_groups.add(param_group_key)

        # mtp_embed: allreduce within dedicated sub-group
        if (
            mtp_embed_weight is not None
            and hasattr(self, "_mtp_embed_global_group")
            and self._mtp_embed_global_group is not None
        ):
            grad = _attached_grad(mtp_embed_weight)
            with paddle.framework.no_grad():
                paddle.distributed.all_reduce(
                    grad.contiguous(), group=self._mtp_embed_global_group
                )

    # ═══════════════════════════════════════════════════════════════════════════
    # AOA modular generation: whole-model entries
    # ═══════════════════════════════════════════════════════════════════════════

    def gen_aoa_statements(self, config=None):
        """Whole-model entry: checkpoint -> model (forward) AOA statements.

        Builds the read-only context (so consumers never construct it) and hands
        off to the modular generator, from ``config`` when given and from
        ``self.config`` otherwise.

        The signature deliberately does not accept the ``Layer`` recursion
        protocol's ``structured_name_prefix`` /
        ``checkpoint_lookup_drop_segment``: this is a whole-model boundary, so a
        container that recursed into it raises ``TypeError`` instead of silently
        producing partial statements.
        """
        ctx = build_aoa_context(
            self, config if config is not None else self.config
        )
        return gen_whole_model_aoa(self, ctx)

    def gen_inv_aoa_statements(self, config=None):
        """Whole-model entry: model -> checkpoint (inverse) AOA statements.

        The independent mirror of :meth:`gen_aoa_statements`: it builds its own
        context and never derives the inverse from the checkpoint->model
        statements.
        """
        ctx = build_aoa_context(
            self, config if config is not None else self.config
        )
        return gen_whole_model_inv_aoa(self, ctx)
