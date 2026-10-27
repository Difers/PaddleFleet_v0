# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025 DeepSeek
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

import contextlib
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import paddle
from paddle import framework, nn

if TYPE_CHECKING:
    from collections.abc import Callable

    from paddle.distributed.communication.group import Group

logger = logging.getLogger(__name__)

from paddlefleet.utils import use_dsv4_accuracy_compatible

from .fp8_utils import FP8_ALIGN, _w4a8_quant_pack_for_hybridep_dispatch
from .fused_a2a import (
    HYBRIDEP_TOKEN_ALIGNMENT,
    DeepEPCombineAsyncRefinedRecompute,
    fused_combine,
    fused_dispatch,
    get_hybrid_ep_buffer,
    hybrid_ep_combine,
    hybrid_ep_dispatch,
    quantize_activation_blockscaled_fast,
)
from .moe_utils import (
    AllGatherGroupOp,
    FakeClone,
    ReduceScatterGroupOp,
    _AllToAll,
    all_gather_group,
    detach_and_requires_grad_,
    is_tensor,
    manual_backward,
    permute,
    reduce_scatter_group,
    sort_chunks_by_idxs,
    unpermute,
    use_accuracy_compatible_kernel,
)
from .moonep import (
    MoonEPWeightBridge,
    get_moonep_buffer,
    is_moonep_available,
    moonep_combine,
    moonep_dispatch,
    moonep_runtime_weights,
)

# ProcessGroup-level SUM op for direct ``process_group.*`` collective calls.
_PG_REDUCE_SUM = framework.core.ReduceOp.SUM

HAVE_HYBRID_EP = False
HYBRID_EP_LOAD_CACHED_KERNELS = True


def _sort_chunks_like_tokens(
    input: paddle.Tensor,
    split_sizes: list[int],
    sorted_idxs: list[int],
) -> paddle.Tensor:
    chunks = paddle.split(input, split_sizes, axis=0)
    return paddle.concat([chunks[i] for i in sorted_idxs], axis=0)


try:
    from paddlefleet_ops import is_hybrid_ep_available

    HAVE_HYBRID_EP = is_hybrid_ep_available()
except ImportError:
    HAVE_HYBRID_EP = False


def is_hybrid_ep_backend_selected(
    dispatcher_type: str | None = None,
) -> bool:
    selected_dispatcher = dispatcher_type or "deepep"
    if selected_dispatcher not in (
        "allgather",
        "alltoall",
        "deepep",
        "hybridep",
        "moonep",
        "ringmoe",
    ):
        raise ValueError(
            "moe_token_dispatcher_type must be one of: "
            "allgather, alltoall, deepep, hybridep, moonep, ringmoe"
        )
    if selected_dispatcher != "hybridep":
        return False
    if not HAVE_HYBRID_EP:
        raise ImportError(
            "moe_token_dispatcher_type=hybridep but HybridEP runtime is unavailable."
        )
    return True


def _try_setup_router_topk_metadata(
    manager,
    num_tokens: int,
    topk_weights: paddle.Tensor | None,
    topk_indices: paddle.Tensor | None,
) -> bool:
    if topk_weights is None or topk_indices is None:
        return False
    manager.token_probs = topk_weights.reshape(
        [num_tokens, manager.router_topk]
    )
    manager.token_indices = topk_indices.reshape(
        [num_tokens, manager.router_topk]
    )
    manager.token_indices.stop_gradient = True
    return True


class _DispatchManager(ABC):
    """
    A manager class to handle dispatch and combine processes for MoE models.

    DispatcherManager handles token dispatching according to the routing_map of format
    [num_local_tokens, world_size, num_instances]. The routing_map is a 3D tensor where each
    element indicates whether a token should be sent to a specific rank.

    num_instances is the maximum number of tokens instances dispatched into a target rank, it
    can be the number of local experts, or the size of sub_group.
    """

    @abstractmethod
    def setup_metadata(
        self,
        routing_map: paddle.Tensor,
        probs: paddle.Tensor,
        topk_weights: paddle.Tensor | None = None,
        topk_indices: paddle.Tensor | None = None,
    ):
        """Set up metadata of routing_map and probs.

        If ``topk_weights`` and ``topk_indices`` are provided (e.g. produced by
        the router), they will be used directly and the internal ``paddle.topk``
        call will be skipped.
        """
        pass

    @abstractmethod
    def dispatch(
        self,
        hidden_states: paddle.Tensor,
        fp8_dispatch: bool,
        async_finish: bool,
    ) -> paddle.Tensor:
        """Dispatch the hidden_states according to the routing_map."""
        pass

    @abstractmethod
    def combine(
        self, hidden_states: paddle.Tensor, combine_overlap_handle: dict | None
    ) -> paddle.Tensor:
        """Combine the hidden_states after expert processing."""
        pass

    @abstractmethod
    def get_dispatched_metadata(self) -> paddle.Tensor:
        """Get the metadata of the dispatched hidden_states."""
        pass

    @abstractmethod
    def get_permuted_hidden_states_by_experts(
        self, hidden_states: paddle.Tensor
    ) -> paddle.Tensor:
        """Get the permuted hidden states by instances."""
        pass

    @abstractmethod
    def get_restored_hidden_states_by_experts(
        self, hidden_states: paddle.Tensor
    ) -> paddle.Tensor:
        """Get the restored hidden states by instances."""
        pass


class _HybridEPManager(_DispatchManager):
    """
    HybridEP path using dispatch_with_permute/combine_with_unpermute only.

    The manager owns per-layer handles and count metadata. The communication
    buffer is shared at fused_a2a module scope.
    """

    def __init__(
        self,
        group: Group,
        router_topk: int,
        num_experts: int | None = None,
        num_local_experts: int | None = None,
        moe_ep_barrier: bool = True,
        hybridep_buffer_configs: dict | None = None,
        moe_deep_gemm: bool = False,
        use_w4a8: bool = False,
    ):
        if not HAVE_HYBRID_EP:
            raise ImportError("HybridEP runtime is not available.")

        self.group = group
        self.router_topk = router_topk
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.routing_map = None
        self.routing_probs = None
        self.token_indices = None
        self.token_probs = None
        self.dispatched_indices = None
        self.dispatched_probs = None
        self.tokens_per_expert = None
        self.padded_tokens_per_expert = None
        self.num_permuted_tokens = None
        self.handle = None
        self._active_buffer = None
        self.hybridep_buffer_configs = hybridep_buffer_configs or {}
        self._moe_deep_gemm = moe_deep_gemm
        # use_w4a8: quantize activations 1x32 UE8M0 on the source tokens for the
        # W4A8 mxfp8 1x32 dispatch and carry the packed scales in HybridEP's fp32
        # scaling-factor slot (see mxfp8_dispatch.py) instead of the FP8 baseline's
        # 1x128 recipe. The mxfp8 1x32 scale is packed into the blockwise-128 scale
        # slot and handed to the UNMODIFIED hybridep fp8 dispatch (hybridep code is
        # NOT changed). Static config, so it rides on the manager like moe_deep_gemm.
        self._use_w4a8 = use_w4a8
        self._reset_dispatch_state()
        self._num_unpadded_tokens = None

    def _reset_dispatch_state(self):
        self._dispatch_uses_fp8 = None
        self._dispatch_pad_multiple = None

    def _set_dispatch_state(self, use_fp8: bool):
        self._dispatch_uses_fp8 = use_fp8
        self._dispatch_pad_multiple = (
            FP8_ALIGN if use_fp8 or self._moe_deep_gemm else None
        )

    def _get_max_num_tokens_per_rank(self, num_local_tokens: int, place) -> int:
        max_num_tokens = num_local_tokens
        if self.group.nranks > 1:
            max_num_tokens_tensor = paddle.to_tensor(
                [num_local_tokens], dtype="int64", place=place
            )
            paddle.distributed.all_reduce(
                max_num_tokens_tensor,
                op=paddle.distributed.ReduceOp.MAX,
                group=self.group,
            )
            max_num_tokens = int(max_num_tokens_tensor.item())
        return (
            (max_num_tokens + HYBRIDEP_TOKEN_ALIGNMENT - 1)
            // HYBRIDEP_TOKEN_ALIGNMENT
            * HYBRIDEP_TOKEN_ALIGNMENT
        )

    def _pad_tokens_to_rank_max(
        self, tensor: paddle.Tensor | None, max_num_tokens: int
    ) -> paddle.Tensor | None:
        if tensor is None or tensor.shape[0] == max_num_tokens:
            return tensor
        assert tensor.shape[0] < max_num_tokens, (
            f"HybridEP token padding expects local tokens <= EP max, got "
            f"{tensor.shape[0]} > {max_num_tokens}."
        )
        pad_shape = [max_num_tokens - tensor.shape[0], *tensor.shape[1:]]
        padding = paddle.zeros(pad_shape, dtype=tensor.dtype)
        return paddle.concat([tensor, padding], axis=0)

    def _get_buffer(
        self,
        hidden_states: paddle.Tensor,
        max_num_of_tokens_per_rank: int | None = None,
    ):
        hidden_dim = hidden_states.shape[-1]
        if max_num_of_tokens_per_rank is None:
            max_num_of_tokens_per_rank = hidden_states.shape[0]
        self._active_buffer = get_hybrid_ep_buffer(
            group=self.group,
            hidden_dim=hidden_dim,
            max_num_of_tokens_per_rank=max_num_of_tokens_per_rank,
            num_local_experts=self.num_local_experts,
            load_cached_kernels=HYBRID_EP_LOAD_CACHED_KERNELS,
            **self.hybridep_buffer_configs,
        )
        return self._active_buffer

    def _get_num_permuted_tokens_upper_bound(
        self, num_local_tokens: int
    ) -> int:
        total_routed_tokens = (
            num_local_tokens * self.group.nranks * self.router_topk
        )
        if FP8_ALIGN > 1:
            total_routed_tokens += self.num_local_experts * (FP8_ALIGN - 1)
        return total_routed_tokens

    def _indices_to_dense_metadata(
        self,
        token_indices: paddle.Tensor,
        token_weights: paddle.Tensor | None,
    ) -> tuple[paddle.Tensor, paddle.Tensor | None]:
        safe_indices = paddle.where(
            token_indices >= 0,
            token_indices,
            paddle.zeros_like(token_indices),
        ).astype("int64")
        one_hot = paddle.nn.functional.one_hot(
            safe_indices, num_classes=self.num_experts
        )
        valid_mask = (token_indices >= 0).astype(one_hot.dtype).unsqueeze(-1)
        one_hot = one_hot * valid_mask
        routing_map = paddle.sum(one_hot, axis=1).astype("bool")

        probs = None
        if token_weights is not None:
            probs = paddle.sum(
                one_hot.astype(token_weights.dtype)
                * token_weights.unsqueeze(-1),
                axis=1,
            )
            if probs.dtype != paddle.float32:
                probs = probs.astype("float32")
        return routing_map, probs

    def _get_dispatch_metadata(
        self,
        token_indices: paddle.Tensor | None,
        token_weights: paddle.Tensor | None,
    ) -> tuple[paddle.Tensor, paddle.Tensor | None]:
        if self.routing_map is not None:
            return self.routing_map, self.routing_probs
        assert token_indices is not None, (
            "HybridEP dispatch requires routing metadata."
        )
        return self._indices_to_dense_metadata(token_indices, token_weights)

    def setup_metadata(
        self,
        routing_map: paddle.Tensor,
        probs: paddle.Tensor,
        topk_weights: paddle.Tensor | None = None,
        topk_indices: paddle.Tensor | None = None,
    ):
        num_tokens = routing_map.shape[0]
        self.routing_map = routing_map.reshape(
            [num_tokens, self.num_experts]
        ).astype("bool")
        self.routing_probs = probs.reshape([num_tokens, self.num_experts])
        if self.routing_probs.dtype != paddle.float32:
            self.routing_probs = self.routing_probs.astype("float32")
        if _try_setup_router_topk_metadata(
            self, num_tokens, topk_weights, topk_indices
        ):
            return
        self.token_probs, self.token_indices = paddle.topk(
            self.routing_probs, self.router_topk, axis=-1
        )

    def _extract_tokens_per_expert(
        self,
        num_dispatched_tokens: int,
        local_expert_routing_map: paddle.Tensor,
    ):
        return (
            local_expert_routing_map[:num_dispatched_tokens]
            .astype("int64")
            .sum(axis=0)
        )

    def _set_num_permuted_tokens(self, tokens_per_expert: paddle.Tensor) -> int:
        self.num_permuted_tokens = int(
            paddle.sum(tokens_per_expert.astype("int64")).item()
        )
        return self.num_permuted_tokens

    def dispatch_overlap(
        self,
        hidden_states: paddle.Tensor,
        token_indices: paddle.Tensor,
        token_weights: paddle.Tensor,
        fp8_dispatch: bool = False,
        async_finish: bool = False,
        use_ue8m0: bool = False,
    ) -> paddle.Tensor:
        del async_finish
        self.token_indices = token_indices
        self.token_probs = token_weights
        hidden_states, self.dispatched_probs, scale = hybrid_ep_dispatch(
            hidden_states,
            token_indices,
            token_weights,
            self,
            fp8_dispatch,
        )
        self.dispatched_indices = None
        return hidden_states, None if scale is None else {"scale": scale}

    def _dispatch_with_permute_impl(
        self,
        hidden_states: paddle.Tensor,
        token_indices: paddle.Tensor,
        token_weights: paddle.Tensor,
        use_fp8: bool = False,
    ):
        num_unpadded_tokens = hidden_states.shape[0]
        max_num_tokens = self._get_max_num_tokens_per_rank(
            num_unpadded_tokens, hidden_states.place
        )
        self._num_unpadded_tokens = num_unpadded_tokens
        routing_map, probs = self._get_dispatch_metadata(
            token_indices, token_weights
        )
        hidden_states = self._pad_tokens_to_rank_max(
            hidden_states, max_num_tokens
        )
        routing_map = self._pad_tokens_to_rank_max(routing_map, max_num_tokens)
        probs = self._pad_tokens_to_rank_max(probs, max_num_tokens)
        buffer = self._get_buffer(hidden_states, max_num_tokens)
        num_permuted_tokens = self._get_num_permuted_tokens_upper_bound(
            max_num_tokens
        )
        scaling_factor = None
        w4a8_fp8 = use_fp8 and self._use_w4a8
        if w4a8_fp8:
            # W4A8 (H1): quantize 1x32 UE8M0 after padding (like the 1x128
            # branch) so the slot has exactly max_num_tokens * H/128 words, and
            # pack the scales directly into the [T, H/128] token-major slot. NO
            # transpose: unlike fp8_quant_blockwise's output_scale_transpose
            # carrier the packed slot is already token-major. Padded source rows
            # are never routed (their routing_map rows are all False).
            hidden_states, scaling_factor = (
                _w4a8_quant_pack_for_hybridep_dispatch(hidden_states)
            )
        elif use_fp8:
            hidden_states, scaling_factor = (
                paddle.incubate.nn.functional.fp8_quant_blockwise(
                    hidden_states,
                    quant_method="1x128",
                    input_transpose=False,
                    output_scale_transpose=True,
                    return_transpose_only=False,
                )
            )
            scaling_factor = scaling_factor.T.contiguous()
        self._set_dispatch_state(use_fp8)
        (
            hidden_states,
            dispatched_probs,
            scale,
            tokens_per_expert,
            self.handle,
        ) = buffer.dispatch_with_permute(
            hidden=hidden_states,
            routing_map=routing_map,
            probs=probs,
            num_of_experts_per_rank=self.num_local_experts,
            use_fp8=use_fp8,
            scaling_factor=scaling_factor,
            pad_multiple=self._dispatch_pad_multiple,
            num_permuted_tokens=num_permuted_tokens,
            non_blocking=True,
        )
        self.padded_tokens_per_expert = tokens_per_expert
        num_permuted_tokens = self._set_num_permuted_tokens(tokens_per_expert)
        hidden_states = hidden_states[:num_permuted_tokens]
        if dispatched_probs is not None:
            dispatched_probs = dispatched_probs[:num_permuted_tokens]
        if scale is not None:
            scale = scale[:num_permuted_tokens]
        if w4a8_fp8:
            # View the fp32 scale slot back to int32 (bytes unchanged -- no float
            # op touches the packed scale). Both dispatchers now hand the expert
            # node the same packed int32 [M, H/128] carrier, which _fwd_gate_up_w4a8
            # unpacks to the exact float32 1x32 scale.
            assert scale is not None, "HybridEP fp8 dispatch returned no scale"
            scale = scale.view(paddle.int32)
        (
            _sparse_to_dense_map,
            _rdma_to_attn_map,
            _attn_to_rdma_map,
            num_dispatched_tokens_tensor,
            local_expert_routing_map,
            *_,
        ) = self.handle
        num_dispatched_tokens = int(num_dispatched_tokens_tensor.item())
        self.tokens_per_expert = self._extract_tokens_per_expert(
            num_dispatched_tokens,
            local_expert_routing_map,
        )
        return hidden_states, dispatched_probs, scale

    def dispatch(
        self,
        hidden_states: paddle.Tensor,
        fp8_dispatch: bool = False,
        async_finish: bool = False,
        use_ue8m0: bool = False,
        using_sonic_moe: bool = False,
    ) -> paddle.Tensor:
        return self.dispatch_overlap(
            hidden_states,
            self.token_indices,
            self.token_probs,
            fp8_dispatch=fp8_dispatch,
            async_finish=async_finish,
        )

    def combine(
        self,
        hidden_states: paddle.Tensor,
        combine_overlap_handle: dict | None = None,
        async_finish: bool = False,
        use_rr_deepep_combine: bool = False,
        fp8_dispatch: bool = False,
        combine_grad_handle: dict | None = None,
    ) -> paddle.Tensor:
        del async_finish, use_rr_deepep_combine
        if combine_overlap_handle is not None:
            raise NotImplementedError(
                "HybridEP backend does not support combine overlap in PaddleFleet."
            )
        hidden_states = hybrid_ep_combine(
            hidden_states, self, self.num_permuted_tokens
        )
        self.dispatched_probs = None
        self.handle = None
        self.num_permuted_tokens = None
        self._reset_dispatch_state()
        if (
            self._num_unpadded_tokens is not None
            and hidden_states.shape[0] != self._num_unpadded_tokens
        ):
            hidden_states = hidden_states[: self._num_unpadded_tokens]
        self._num_unpadded_tokens = None
        return hidden_states

    def get_dispatched_metadata(self) -> paddle.Tensor:
        if self.dispatched_indices is None or self.dispatched_probs is None:
            raise NotImplementedError(
                "HybridEP backend does not expose fused-node dispatch metadata for the current mode."
            )
        return self.dispatched_indices, self.dispatched_probs

    def get_number_of_tokens_per_expert(self) -> paddle.Tensor:
        return self.tokens_per_expert

    def get_permuted_hidden_states_by_experts(
        self, hidden_states: paddle.Tensor
    ) -> paddle.Tensor:
        return hidden_states

    def get_restored_hidden_states_by_experts(
        self, hidden_states: paddle.Tensor
    ) -> paddle.Tensor:
        if self.dispatched_probs is None:
            return hidden_states
        return hidden_states * self.dispatched_probs.astype(
            hidden_states.dtype
        ).unsqueeze(-1)


class _MoonEPManager(_DispatchManager):
    """MoonEP manager using fixed-capacity E+B expert groups."""

    def __init__(
        self,
        group: Group,
        router_topk: int,
        num_experts: int,
        num_local_experts: int,
        moe_ep_barrier: bool = True,
    ):
        del moe_ep_barrier
        if not is_moonep_available():
            raise ImportError(
                "moe_token_dispatcher_type=moonep but MoonEP is unavailable."
            )
        self.group = group
        self.router_topk = router_topk
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.token_indices = None
        self.token_probs = None
        self.tokens_per_expert = None
        self.dispatched_indices = None
        self.dispatched_probs = None
        self.handle = None
        self._buffer = None
        self._buffer_signature = None
        self._bridge = None
        self._num_dispatched_tokens = None

    def bind_experts(self, grouped_experts) -> None:
        self._bridge = MoonEPWeightBridge(
            group=self.group,
            num_experts=self.num_experts,
            num_local_experts=self.num_local_experts,
            weight1_shape=grouped_experts.weight1.shape,
            weight2_shape=grouped_experts.weight2.shape,
        )

    def setup_metadata(
        self,
        routing_map: paddle.Tensor,
        probs: paddle.Tensor,
        topk_weights: paddle.Tensor | None = None,
        topk_indices: paddle.Tensor | None = None,
    ):
        num_tokens = routing_map.shape[0]
        routing_map = routing_map.reshape(
            [num_tokens, self.num_experts]
        ).astype("bool")
        probs = probs.reshape([num_tokens, self.num_experts])
        if not _try_setup_router_topk_metadata(
            self, num_tokens, topk_weights, topk_indices
        ):
            self.token_probs, self.token_indices = paddle.topk(
                probs, self.router_topk, axis=-1
            )
        self.token_probs = self.token_probs.astype("float32")
        self.token_indices = self.token_indices.astype("int32")
        padding_mask = self.token_indices < 0
        self.token_probs = paddle.where(
            padding_mask,
            paddle.zeros_like(self.token_probs),
            self.token_probs,
        )
        self.token_indices = paddle.where(
            padding_mask,
            paddle.zeros_like(self.token_indices),
            self.token_indices,
        )
        self.token_indices.stop_gradient = True
        self.tokens_per_expert = _tokens_per_expert_histogram(
            self.token_indices, self.num_experts
        )

    def _ensure_buffer(self, hidden_states: paddle.Tensor) -> None:
        if self._bridge is None:
            raise RuntimeError(
                "MoonEP grouped experts must be bound before dispatch."
            )
        signature = (
            int(hidden_states.shape[0]),
            int(hidden_states.shape[1]),
            str(hidden_states.dtype),
            int(self.router_topk),
            int(self.num_experts),
            int(self.num_local_experts),
        )
        if self._buffer is not None:
            if hidden_states.dtype != paddle.bfloat16:
                raise ValueError(
                    "MoonEP dispatch requires BF16 hidden states, "
                    f"got {hidden_states.dtype}."
                )
            if signature != self._buffer_signature:
                raise ValueError(
                    "MoonEP requires a fixed dispatch signature while a layer "
                    f"buffer is live: expected {self._buffer_signature}, "
                    f"got {signature}."
                )
            return
        world_size = paddle.distributed.get_world_size(self.group)
        rank_metadata = [None] * world_size
        paddle.distributed.all_gather_object(
            rank_metadata,
            signature,
            group=self.group,
        )
        if any(metadata != rank_metadata[0] for metadata in rank_metadata):
            raise ValueError(
                "MoonEP requires an identical dispatch signature across its "
                f"ranks, got {rank_metadata}."
            )
        if hidden_states.dtype != paddle.bfloat16:
            raise ValueError(
                "MoonEP dispatch requires BF16 hidden states, "
                f"got {hidden_states.dtype}."
            )
        self._buffer_signature = signature
        self._buffer = get_moonep_buffer(
            S=signature[0],
            H=hidden_states.shape[1],
            K=self.router_topk,
            E=self.num_experts,
            B=self.num_local_experts,
            num_ep_ranks=world_size,
            group=self.group,
        )
        self._bridge.attach_buffer(self._buffer)

    def dispatch_overlap(
        self,
        hidden_states: paddle.Tensor,
        token_indices: paddle.Tensor,
        token_weights: paddle.Tensor,
        fp8_dispatch: bool = False,
        async_finish: bool = False,
        use_ue8m0: bool = False,
    ):
        del (
            hidden_states,
            token_indices,
            token_weights,
            fp8_dispatch,
            async_finish,
            use_ue8m0,
        )
        raise NotImplementedError("MoonEP does not support dispatch overlap.")

    def dispatch(
        self,
        hidden_states: paddle.Tensor,
        fp8_dispatch: bool = False,
        async_finish: bool = False,
        use_ue8m0: bool = False,
        using_sonic_moe: bool = False,
    ):
        if fp8_dispatch or use_ue8m0 or using_sonic_moe:
            raise NotImplementedError(
                "MoonEP currently supports BF16 dispatch."
            )
        del async_finish
        self._ensure_buffer(hidden_states)
        state = {}
        (
            dispatched_hidden,
            self.dispatched_probs,
            self.tokens_per_expert,
        ) = moonep_dispatch(
            hidden_states,
            self.token_probs,
            self.token_indices,
            self.tokens_per_expert,
            self._buffer,
            state,
        )
        self.handle = state["plan"]
        return dispatched_hidden, None

    def combine(
        self,
        hidden_states: paddle.Tensor,
        combine_overlap_handle: dict | None = None,
        async_finish: bool = False,
        use_rr_deepep_combine: bool = False,
        fp8_dispatch: bool = False,
        combine_grad_handle: dict | None = None,
    ):
        if combine_overlap_handle is not None:
            raise NotImplementedError(
                "MoonEP does not support shared-expert combine overlap."
            )
        del (
            async_finish,
            use_rr_deepep_combine,
            fp8_dispatch,
            combine_grad_handle,
        )
        hidden_states = moonep_combine(
            hidden_states, self._buffer, self.handle, self._bridge
        )
        self.handle = None
        self.token_indices = None
        self.token_probs = None
        self.tokens_per_expert = None
        self.dispatched_probs = None
        self._num_dispatched_tokens = None
        return hidden_states

    def runtime_expert_weights(self, grouped_experts):
        if self.handle is None:
            raise RuntimeError("MoonEP runtime weights require an active plan.")
        return moonep_runtime_weights(
            grouped_experts, self._bridge, self.handle
        )

    def get_dispatched_metadata(self):
        raise NotImplementedError(
            "MoonEP exposes E+B group counts instead of DeepEP routing metadata."
        )

    def get_number_of_tokens_per_expert(self):
        return self.tokens_per_expert

    def get_permuted_hidden_states_by_experts(self, hidden_states):
        self._num_dispatched_tokens = int(hidden_states.shape[0])
        num_valid_tokens = int(self.tokens_per_expert.sum().item())
        return hidden_states[:num_valid_tokens]

    def get_restored_hidden_states_by_experts(self, hidden_states):
        if self.dispatched_probs is not None:
            hidden_states = hidden_states * self.dispatched_probs[
                : hidden_states.shape[0]
            ].astype(hidden_states.dtype).unsqueeze(-1)
        if hidden_states.shape[0] != self._num_dispatched_tokens:
            hidden_states = paddle.concat(
                [
                    hidden_states,
                    paddle.zeros(
                        [
                            self._num_dispatched_tokens
                            - hidden_states.shape[0],
                            hidden_states.shape[1],
                        ],
                        dtype=hidden_states.dtype,
                    ),
                ],
                axis=0,
            )
        return hidden_states


class _DeepEPManager(_DispatchManager):
    """
    A manager class to handle fused all-to-all communication processes for MoE models using
    DeepEP backend. See https://github.com/deepseek-ai/deepep for more details.

    The workflow of the DeepEP dispatcher is:
    (1) setup_metadata(): Process routing map and probabilities to prepare dispatch metadata
    (2) dispatch():
        - Use fused kernel to permute tokens and perform all-to-all communication in single step
    (3) get_permuted_hidden_states_by_instances():
        - Convert routing map and probabilities to multihot format
        - Permute tokens using fused kernel
    (4) get_restored_hidden_states_by_instances():
        - Reverse permutation using fused kernel
    (5) combine():
        - Reverse process using fused kernel to unpermute and perform all-to-all in single step

    This implementation uses fused communication kernels (fused_dispatch/fused_combine) that
    combine permutation and communication operations for improved efficiency compared to
    separate permute+alltoall steps.
    """

    def __init__(
        self,
        group: Group,
        router_topk: int,
        num_experts: int | None = None,
        num_local_experts: int | None = None,
        moe_ep_barrier: bool = True,
        use_accuracy_compatible: bool = False,
        use_w4a8: bool = False,
    ):
        self.group = group
        self.router_topk = router_topk
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.moe_ep_barrier = moe_ep_barrier
        self.use_accuracy_compatible = use_accuracy_compatible
        # use_w4a8: quantize activations 1x32 UE8M0 on the source tokens for the
        # W4A8 mxfp8 1x32 dispatch and send fp8 + packed int32 scale through DeepEP
        # instead of BF16. The mxfp8 1x32 scale is packed into the blockwise-128
        # scale slot and handed to the UNMODIFIED deep_ep fp8 dispatch (deep_ep code
        # is NOT changed). Static config, so it rides on the manager (mirrors HybridEP).
        self._use_w4a8 = use_w4a8

        # Metadata
        self.token_indices = None
        self.token_probs = None
        # Handle used for combine operation
        self.handle = None
        self.global_input_probs = None

        if fused_dispatch is None:
            raise ImportError(
                "DeepEP is not supported in your paddlepaddle whl package."
            )
        self._rr_fusedcombined = None

    def setup_metadata(
        self,
        routing_map: paddle.Tensor,
        probs: paddle.Tensor,
        topk_weights: paddle.Tensor | None = None,
        topk_indices: paddle.Tensor | None = None,
    ):
        num_tokens = routing_map.shape[0]

        if _try_setup_router_topk_metadata(
            self, num_tokens, topk_weights, topk_indices
        ):
            return

        routing_map = routing_map.reshape([num_tokens, self.num_experts])
        probs = probs.reshape([num_tokens, self.num_experts])
        # Convert the format of routing map from multihot to indices.
        self.token_probs, self.token_indices = paddle.topk(
            probs, self.router_topk, axis=-1
        )

    def dispatch_overlap(
        self,
        hidden_states: paddle.Tensor,
        token_indices: paddle.Tensor,
        token_weights: paddle.Tensor,
        fp8_dispatch: bool = False,
        async_finish: bool = False,
        use_ue8m0: bool = False,
    ) -> paddle.Tensor:
        hidden_states, dispatched_probs, states, scale = fused_dispatch(
            hidden_states,
            token_indices,
            token_weights,
            self.num_experts,
            self.group,
            fp8_dispatch=fp8_dispatch,
            async_finish=async_finish,
            use_ue8m0=use_ue8m0,
            use_w4a8=self._use_w4a8,
        )
        self.handle = states["handle"]
        self.tokens_per_expert = states["tokens_per_expert"]
        self.dispatched_indices = states["dispatched_indices"]
        self.dispatched_probs = dispatched_probs
        self.global_input_probs = None

        return hidden_states, scale

    def dispatch(
        self,
        hidden_states: paddle.Tensor,
        fp8_dispatch: bool = False,
        async_finish: bool = False,
        use_ue8m0: bool = False,
        using_sonic_moe: bool = False,
    ) -> paddle.Tensor:
        hidden_states, dispatched_probs, states, scale = fused_dispatch(
            hidden_states,
            self.token_indices,
            self.token_probs,
            self.num_experts,
            self.group,
            fp8_dispatch=fp8_dispatch,
            async_finish=async_finish,
            moe_ep_barrier=self.moe_ep_barrier,
            use_ue8m0=use_ue8m0,
            using_sonic_moe=using_sonic_moe,
            use_w4a8=self._use_w4a8,
        )
        self.handle = states["handle"]
        self.tokens_per_expert = states["tokens_per_expert"]
        self.dispatched_indices = states["dispatched_indices"]
        self.dispatched_probs = dispatched_probs
        self.global_input_probs = None

        return hidden_states, scale

    def _indices_to_multihot(self, indices, probs):
        """
        Converts a tensor of indices to a multihot vector.

        Args:
            indices (paddle.Tensor): [num_tokens, topk] token indices, where -1 means masked out.
            probs (paddle.Tensor): [num_tokens, topk] token probabilities.

        Returns:
            tuple[paddle.Tensor, paddle.Tensor]:
                - routing_map: Multihot vector.
                - probs: Multihot probabilities.
        """
        if use_dsv4_accuracy_compatible():
            from paddlefleet.accuracy_compatible_patch import (
                indices_to_multihot,
            )

            return indices_to_multihot(indices, probs, self.num_local_experts)

        batch_size = indices.shape[0]
        multihot_routing_map = paddle.zeros(
            (batch_size, self.num_local_experts), dtype=paddle.int64
        )

        multihot_probs = paddle.zeros(
            (batch_size, self.num_local_experts), dtype=paddle.float32
        )

        mask = indices != -1
        valid_indices = indices[mask]
        row_indices = paddle.arange(batch_size).repeat_interleave(
            mask.sum(axis=1)
        )
        multihot_routing_map[row_indices, valid_indices] = 1
        multihot_probs[row_indices, valid_indices] = probs[mask]
        return multihot_routing_map.cast(paddle.bool), multihot_probs

    def get_dispatched_metadata(self) -> paddle.Tensor:
        return self.dispatched_indices, self.dispatched_probs

    def get_number_of_tokens_per_expert(self) -> paddle.Tensor:
        """
        Get the number of tokens per expert.
        """
        return self.tokens_per_expert

    def combine(
        self,
        hidden_states: paddle.Tensor,
        combine_overlap_handle: dict | None = None,
        async_finish: bool = False,
        use_rr_deepep_combine: bool = False,
        fp8_dispatch: bool = False,
        combine_grad_handle: dict | None = None,
    ) -> paddle.Tensor:
        if combine_overlap_handle is not None and use_rr_deepep_combine:
            if self._rr_fusedcombined is None:
                self._rr_fusedcombined = DeepEPCombineAsyncRefinedRecompute()
            elif not isinstance(
                self._rr_fusedcombined, DeepEPCombineAsyncRefinedRecompute
            ):
                raise RuntimeError(
                    f"_rr_fusedcombined type mismatch: expected DeepEPCombineAsyncRefinedRecompute, "
                    f"got {type(self._rr_fusedcombined).__name__}."
                )
        if fp8_dispatch is True:
            assert combine_grad_handle is not None, (
                "fp8_dispatch=True, but combine_grad_handle is None."
            )
        hidden_states = fused_combine(
            hidden_states,
            self.group,
            self.handle,
            _rr_fusedcombined=self._rr_fusedcombined,
            combine_overlap_handle=combine_overlap_handle,
            async_finish=async_finish,
            moe_ep_barrier=self.moe_ep_barrier,
            use_rr_deepep_combine=use_rr_deepep_combine,
            fp8_dispatch=fp8_dispatch,
            combine_grad_handle=combine_grad_handle,
        )
        # Release the handle and token_indices after combine operation
        self.handle = None
        self.token_indices = None
        self.token_probs = None
        self.dispatched_probs = None
        self.dispatched_indices = None
        return hidden_states

    def get_permuted_hidden_states_by_experts(
        self, hidden_states: paddle.Tensor
    ) -> paddle.Tensor:
        self.dispatched_routing_map, self.dispatched_probs = (
            self._indices_to_multihot(
                self.dispatched_indices, self.dispatched_probs
            )
        )
        if self.use_accuracy_compatible and use_dsv4_accuracy_compatible():
            self.global_input_probs = (
                self.dispatched_probs.T.contiguous().masked_select(
                    self.dispatched_routing_map.T.contiguous().cast(paddle.bool)
                )
            )
        self.hidden_shape_before_permute = hidden_states.shape
        hidden_states, self.reversed_mapping_for_combine = permute(
            hidden_states,
            self.dispatched_routing_map,
            num_out_tokens=sum(self.tokens_per_expert),
            use_accuracy_compatible=self.use_accuracy_compatible,
        )
        return hidden_states

    def get_restored_hidden_states_by_experts(
        self, hidden_states: paddle.Tensor
    ) -> paddle.Tensor:
        input_dtype = hidden_states.dtype
        assert self.dispatched_probs.dtype == paddle.float32, (
            "DeepEP only supports float32 probs"
        )
        hidden_states = unpermute(
            hidden_states,
            self.reversed_mapping_for_combine,
            restore_shape=self.hidden_shape_before_permute,
            routing_map=self.dispatched_routing_map,
            probs=(
                None
                if (
                    self.use_accuracy_compatible
                    and use_dsv4_accuracy_compatible()
                )
                else self.dispatched_probs
            ),
            use_accuracy_compatible=self.use_accuracy_compatible,
        )
        return hidden_states.to(input_dtype)


class MoETokenDispatcher:
    """
    MoE Token Dispatcher
    """

    def __init__(self, ep_group) -> None:
        """
        Initialize the MoE Token Dispatcher.
        """
        self._ep_group = ep_group

    @property
    def ep_group(self):
        """Get expert model parallel group."""
        return self._ep_group

    @property
    def ep_size(self):
        """Get expert model parallel world_size."""
        return self.ep_group.world_size

    @abstractmethod
    def token_permutation(
        self,
        tokens: paddle.Tensor,
        probs: paddle.Tensor,
        routing_map: paddle.Tensor,
    ):
        """Dispatch tokens to experts.

        Args:
            tokens (paddle.Tensor): Input tokens.
            probs (paddle.Tensor): The routing probability tensor [num_tokens, num_experts].
            routing_map (paddle.Tensor): Token to expert mapping tensor.

        Returns:
            paddle.Tensor: Tokens tensor.
        """
        raise NotImplementedError("Dispatch function not implemented.")

    @abstractmethod
    def token_unpermutation(
        self, expert_output: paddle.Tensor, bias: paddle.Tensor = None
    ):
        """Restores the expert output to its original ordering.

        Args:
            expert_output (paddle.Tensor): The output tensor from the expert models.
            bias (paddle.Tensor): The bias tensor.

        Returns:
            (paddle.Tensor, paddle.Tensor): Unpermuted activation and optional bias.
        """
        raise NotImplementedError("Restore function not implemented.")


class MoEFlexTokenDispatcher(MoETokenDispatcher):
    """
    Flexible token dispatcher for MoE models with Efficient-A2A communication kernels.
    """

    def __init__(
        self,
        num_local_experts: int,
        num_experts_per_tok: int,
        n_routed_experts: int,
        ep_group: Group,
        moe_ep_barrier: bool = True,
        dispatcher_type: str | None = None,
        hybridep_buffer_configs: dict | None = None,
        moe_deep_gemm: bool = False,
        use_accuracy_compatible: bool = False,
        use_w4a8: bool = False,
    ):
        super().__init__(ep_group)

        self.use_accuracy_compatible = use_accuracy_compatible
        self.num_local_experts = num_local_experts
        assert self.ep_size > 1, "Flex token dispatcher requires EP > 1"
        if dispatcher_type == "moonep":
            manager_cls = _MoonEPManager
        elif is_hybrid_ep_backend_selected(dispatcher_type):
            manager_cls = _HybridEPManager
        else:
            manager_cls = _DeepEPManager
        manager_kwargs = {
            "group": self.ep_group,
            "router_topk": num_experts_per_tok,
            "num_experts": n_routed_experts,
            "num_local_experts": self.num_local_experts,
            "moe_ep_barrier": moe_ep_barrier,
        }
        # The W4A8 FP8-dispatch knobs are only understood by the DeepEP and
        # HybridEP managers; _MoonEPManager does not accept them, so they must
        # not ride in the common kwargs.
        if manager_cls is _HybridEPManager:
            manager_kwargs["hybridep_buffer_configs"] = hybridep_buffer_configs
            manager_kwargs["moe_deep_gemm"] = moe_deep_gemm
            manager_kwargs["use_w4a8"] = use_w4a8
        elif manager_cls is _DeepEPManager:
            manager_kwargs["use_accuracy_compatible"] = use_accuracy_compatible
            manager_kwargs["use_w4a8"] = use_w4a8
        self._comm_manager = manager_cls(**manager_kwargs)
        self.global_input_probs = None

    def bind_experts(self, grouped_experts) -> None:
        if isinstance(self._comm_manager, _MoonEPManager):
            self._comm_manager.bind_experts(grouped_experts)

    def runtime_expert_weights(self, grouped_experts):
        if not isinstance(self._comm_manager, _MoonEPManager):
            raise RuntimeError(
                "runtime_expert_weights is only provided by the MoonEP manager."
            )
        return self._comm_manager.runtime_expert_weights(grouped_experts)

    def dispatch_preprocess(
        self,
        hidden_states: paddle.Tensor,
        probs: paddle.Tensor,
        routing_map: paddle.Tensor,
        topk_weights: paddle.Tensor | None = None,
        topk_indices: paddle.Tensor | None = None,
    ):
        self.hidden_shape = hidden_states.shape
        self.global_input_probs = None
        hidden_states = hidden_states.view([-1, self.hidden_shape[-1]])
        self._comm_manager.setup_metadata(
            routing_map, probs, topk_weights, topk_indices
        )
        return hidden_states

    def dispatch_preprocess_overlap(
        self,
        hidden_states: paddle.Tensor,
        token_probs: paddle.Tensor,
        token_indices: paddle.Tensor,
    ):
        self.hidden_shape = hidden_states.shape
        self.global_input_probs = None
        hidden_states = hidden_states.view([-1, self.hidden_shape[-1]])
        self._comm_manager.routing_map = None
        self._comm_manager.routing_probs = None
        self._comm_manager.token_probs = token_probs
        self._comm_manager.token_indices = token_indices
        return hidden_states

    def token_dispatch_overlap(
        self,
        hidden_states: paddle.Tensor,
        token_indices: paddle.Tensor,
        token_weights: paddle.Tensor,
        fp8_dispatch: bool,
        async_finish: bool = False,
        use_ue8m0: bool = False,
    ):
        return self._comm_manager.dispatch_overlap(
            hidden_states,
            token_indices,
            token_weights,
            fp8_dispatch,
            async_finish,
            use_ue8m0=use_ue8m0,
        )

    def token_dispatch(
        self,
        hidden_states: paddle.Tensor,
        fp8_dispatch: bool,
        async_finish: bool = False,
        use_ue8m0: bool = False,
        using_sonic_moe: bool = False,
    ):
        return self._comm_manager.dispatch(
            hidden_states,
            fp8_dispatch,
            async_finish,
            use_ue8m0=use_ue8m0,
            using_sonic_moe=using_sonic_moe,
        )

    def dispatch_postprocess(
        self,
        hidden_states: paddle.Tensor,
    ):
        global_input_tokens = (
            self._comm_manager.get_permuted_hidden_states_by_experts(
                hidden_states
            )
        )
        self.global_input_probs = getattr(
            self._comm_manager, "global_input_probs", None
        )
        tokens_per_expert = self._comm_manager.get_number_of_tokens_per_expert()

        return global_input_tokens, tokens_per_expert

    def combine_preprocess(self, hidden_states: paddle.Tensor):
        return self._comm_manager.get_restored_hidden_states_by_experts(
            hidden_states
        )

    def token_combine(self, hidden_states: paddle.Tensor, async_finish=False):
        return self._comm_manager.combine(
            hidden_states, async_finish=async_finish
        )

    def combine_postprocess(self, hidden_states: paddle.Tensor):
        return hidden_states.reshape(self.hidden_shape)

    def get_dispatched_routing(self):
        """Return (dispatched_indices, dispatched_probs, tokens_per_expert)."""
        return (
            self._comm_manager.dispatched_indices,
            self._comm_manager.dispatched_probs,
            self._comm_manager.tokens_per_expert,
        )

    def token_permutation(
        self,
        hidden_states: paddle.Tensor,
        probs: paddle.Tensor,
        routing_map: paddle.Tensor,
        topk_weights: paddle.Tensor | None = None,
        topk_indices: paddle.Tensor | None = None,
    ) -> tuple[paddle.Tensor, paddle.Tensor]:
        self.hidden_shape = hidden_states.shape
        hidden_states = hidden_states.view([-1, self.hidden_shape[-1]])

        self._comm_manager.setup_metadata(
            routing_map, probs, topk_weights, topk_indices
        )
        hidden_states, scale = self._comm_manager.dispatch(hidden_states)
        global_input_tokens = (
            self._comm_manager.get_permuted_hidden_states_by_experts(
                hidden_states
            )
        )
        tokens_per_expert = self._comm_manager.get_number_of_tokens_per_expert()

        return global_input_tokens, tokens_per_expert

    def token_unpermutation(
        self, hidden_states: paddle.Tensor, bias: paddle.Tensor | None = None
    ) -> tuple[paddle.Tensor, paddle.Tensor | None]:
        assert bias is None, "Bias is not supported in MoEFlexTokenDispatcher"
        hidden_states = (
            self._comm_manager.get_restored_hidden_states_by_experts(
                hidden_states
            )
        )
        hidden_states = self._comm_manager.combine(hidden_states)

        hidden_states = hidden_states.reshape(self.hidden_shape)
        return hidden_states, None


class AllToAllTokenDispatcher(nn.Layer):
    """
    All-to-All EP
    """

    def __init__(
        self,
        moe_group: Group,
        expert_model_parallel_size: int,
        num_experts_per_device: int,
        local_expert_indices: list,
        use_accuracy_compatible: bool = False,
    ):
        nn.Layer.__init__(self)
        self.moe_group = moe_group
        self.expert_model_parallel_size = expert_model_parallel_size
        self.num_experts_per_device = num_experts_per_device
        self.local_expert_indices = local_expert_indices
        self.num_local_experts = len(local_expert_indices)
        self.use_accuracy_compatible = use_accuracy_compatible

    def dispatch_preprocess(
        self,
        hidden_states: paddle.Tensor,
        probs: paddle.Tensor,
        mask: paddle.Tensor,  # routing_map
        topk_weights: paddle.Tensor | None = None,
        topk_indices: paddle.Tensor | None = None,
    ) -> tuple[paddle.Tensor, paddle.Tensor]:
        self.routing_map = mask
        self.probs = probs
        self.hidden_states_dtype = hidden_states.dtype
        self.num_experts = (
            self.num_experts_per_device * self.expert_model_parallel_size
        )
        mask = mask.to(paddle.int32)

        if len(hidden_states.shape) == 3:
            batch_size, seq_len, d_model = hidden_states.shape
        else:
            seq_len, d_model = hidden_states.shape
        reshaped_input = hidden_states.reshape([-1, d_model])
        self.d_model = d_model
        self.reshaped_input_shape = reshaped_input.shape
        tokens_per_expert = mask.sum(axis=0)  # Shape: [num_experts]
        tokens_per_expert = tokens_per_expert.detach()
        tokens_per_ep_rank = tokens_per_expert.reshape(
            [self.expert_model_parallel_size, -1]
        ).sum(axis=1)
        # First All-to-All: Exchange expert token counts across ranks
        # Returns `tokens_per_expert_group` is for current rank
        num_global_tokens_per_expert = AllGatherGroupOp.apply(
            tokens_per_expert, group=self.moe_group
        ).reshape(self.expert_model_parallel_size, self.num_experts)
        num_global_tokens_per_local_expert = num_global_tokens_per_expert[
            :, self.local_expert_indices[0] : self.local_expert_indices[-1] + 1
        ].clone()

        # Can also use the two AllToAll functions below instead of the above AllGather
        # It will save memory , but also has more accuracy diff with DeepEP version
        # global_tokens_per_expert = _AllToAll.apply(
        #     [tokens_per_expert.shape[0]],
        #     tokens_per_expert,
        #     group=self.moe_group,
        # )
        # num_global_tokens_per_local_expert = global_tokens_per_expert.reshape(self.expert_model_parallel_size, self.num_local_experts)

        if num_global_tokens_per_local_expert.sum().item() == 0:
            self.is_empty_tokens = True
        else:
            self.is_empty_tokens = False

        self.tokens_per_expert = num_global_tokens_per_local_expert.sum(axis=0)

        num_global_tokens_per_rank = num_global_tokens_per_local_expert.sum(
            axis=1
        )

        self.num_global_tokens_per_local_expert = (
            num_global_tokens_per_local_expert.reshape(
                -1, self.num_local_experts
            )
        )

        self.output_splits = num_global_tokens_per_rank.cpu().tolist()
        num_local_tokens_per_expert = self.routing_map.sum(dim=0)
        self.input_split_sizes = num_local_tokens_per_expert.reshape(
            self.expert_model_parallel_size, self.num_local_experts
        ).sum(axis=1)
        self.output_shape_tokens = [
            num_global_tokens_per_rank.sum().cpu().item(),
            d_model,
        ]

        (
            permutated_local_input_tokens,
            self.reversed_local_input_permutation_mapping,
        ) = permute(
            reshaped_input,
            self.routing_map,
            use_accuracy_compatible=self.use_accuracy_compatible,
        )
        if use_accuracy_compatible_kernel():
            num_routed_tokens = int(tokens_per_expert.sum().item())
            routing_map = self.routing_map.cast(paddle.bool).T.contiguous()
            flat_sorted = paddle.argsort(
                routing_map.reshape([-1]).cast("int32"),
                descending=True,
                stable=True,
            )[:num_routed_tokens]
            self.permuted_local_probs = paddle.index_select(
                self.probs.T.contiguous().reshape([-1]),
                flat_sorted,
                axis=0,
            )
        self.permutated_local_input_tokens_shape = (
            permutated_local_input_tokens.shape
        )

        return permutated_local_input_tokens

    def token_dispatch(
        self,
        permutated_local_input_tokens: paddle.Tensor,
        fp8_dispatch: bool = False,
        async_finish: bool = False,
        use_ue8m0: bool = False,
        using_sonic_moe: bool = False,
    ):
        # Second All-to-All: Exchange expert tokens across ranks. `gathered_tokens` are the tokens that will be processed by current rank
        global_input_tokens = _AllToAll.apply(
            self.output_shape_tokens,
            permutated_local_input_tokens,  # sorted_tokens,
            out_split_sizes=self.output_splits,
            in_split_sizes=self.input_split_sizes,
            group=self.moe_group,
        )
        if use_accuracy_compatible_kernel():
            # Match Megatron's all-to-all backward numerics by routing probs through a
            # 2D [tokens, 1] tensor, like hidden-state dispatch.
            global_input_probs_2d = _AllToAll.apply(
                [self.output_shape_tokens[0], 1],
                self.permuted_local_probs.unsqueeze(-1),
                out_split_sizes=self.output_splits,
                in_split_sizes=self.input_split_sizes,
                group=self.moe_group,
            )
            self.global_input_probs = global_input_probs_2d.squeeze(-1)

        return global_input_tokens, None

    def dispatch_postprocess(
        self,
        global_input_tokens: paddle.Tensor,
    ):
        input_chunk_idxs = paddle.arange(self.num_experts)
        # [num_local_experts, ep_size]. Sort the input chunks by local experts.
        self.sort_input_by_local_experts = input_chunk_idxs.reshape(
            -1, self.num_local_experts
        ).T.ravel()
        # [ep_size, num_local_experts]. Restore the output chunks by local experts.
        self.restore_output_by_local_experts = input_chunk_idxs.reshape(
            self.num_local_experts, -1
        ).T.ravel()

        if self.num_local_experts > 1 and not self.is_empty_tokens:
            split_sizes_list = (
                self.num_global_tokens_per_local_expert.ravel().tolist()
            )
            sorted_idxs_list = self.sort_input_by_local_experts.tolist()
            global_input_tokens, _ = sort_chunks_by_idxs(
                global_input_tokens,
                self.num_global_tokens_per_local_expert.ravel(),
                self.sort_input_by_local_experts,
            )
            if use_accuracy_compatible_kernel():
                self.global_input_probs = _sort_chunks_like_tokens(
                    self.global_input_probs,
                    split_sizes_list,
                    sorted_idxs_list,
                )
        sorted_tokens = global_input_tokens
        self.tokens_per_expert_post_gather = self.tokens_per_expert
        return sorted_tokens, self.tokens_per_expert_post_gather

    def get_dispatched_routing(self):
        """Return (dispatched_indices, dispatched_probs, tokens_per_expert).

        AllToAll uses tokens_per_expert-based expert processing
        (expert_forward), so dispatched_indices and dispatched_probs are None.
        The corresponding branch in ``fusion_moe_forward`` selects
        ``expert_forward`` instead of index-based fusion kernels.
        """
        return (None, None, self.tokens_per_expert)

    def combine_preprocess(self, hidden_states: paddle.Tensor):
        if self.num_local_experts > 1 and not self.is_empty_tokens:
            hidden_states, _ = sort_chunks_by_idxs(
                hidden_states,
                self.num_global_tokens_per_local_expert.T.ravel(),
                self.restore_output_by_local_experts,
            )
        return hidden_states

    def token_combine(
        self,
        hidden_states: paddle.Tensor,
        combine_overlap_handle: dict | None = None,
        async_finish: bool = False,
        fp8_combine_grad_handle: dict | None = None,
    ):
        permutated_local_input_tokens = _AllToAll.apply(
            self.permutated_local_input_tokens_shape,
            hidden_states,
            out_split_sizes=self.input_split_sizes,
            in_split_sizes=self.output_splits,
            group=self.moe_group,
        )
        return permutated_local_input_tokens

    def combine_postprocess(self, permutated_local_input_tokens: paddle.Tensor):
        output = unpermute(
            permutated_local_input_tokens,
            self.reversed_local_input_permutation_mapping,
            restore_shape=self.reshaped_input_shape,
            probs=(None if use_accuracy_compatible_kernel() else self.probs),
            routing_map=self.routing_map,
            use_accuracy_compatible=self.use_accuracy_compatible,
        )
        output_dtype = getattr(self, "hidden_states_dtype", output.dtype)
        return output.cast(output_dtype)


class _RouterAllGather(paddle.autograd.PyLayer):
    """AllGather of router tensors over ``group``, on the calc stream.

    Forward:  ``[T_local, K]`` -> AllGather -> ``[T_global, K]``.
    Backward: ``[T_global, K]`` -> ReduceScatter(SUM) -> ``[T_local, K]``.

    Every rank of ``group`` computes partial expert outputs for all gathered
    tokens, so each holds a partial gradient for every token; backward must
    sum the partials across ranks, not just slice out the local segment.
    """

    @staticmethod
    def forward(ctx, input, group):
        ctx.group = group
        ctx.input_shape = list(input.shape)
        # Kept on the calc stream so it does not queue behind the token
        # gathers already occupying the comm stream.
        return _calc_stream_all_gather(input, group)

    @staticmethod
    def backward(ctx, grad):
        group = ctx.group
        local_shape = ctx.input_shape
        if group is None or group.nranks == 1:
            if list(grad.shape) != local_shape:
                grad = grad.reshape(local_shape)
            return grad
        global_shape = [local_shape[0] * group.nranks, *local_shape[1:]]
        if list(grad.shape) != global_shape:
            expected_numel = 1
            for dim in global_shape:
                expected_numel *= dim
            if int(grad.numel()) != expected_numel:
                raise ValueError(
                    "_RouterAllGather.backward: incoming grad has "
                    f"{int(grad.numel())} elements but the AllGather'd router "
                    f"tensor requires {expected_numel} (global_shape="
                    f"{global_shape})."
                )
            grad = grad.reshape(global_shape)
        out = reduce_scatter_group(grad.contiguous(), group=group)
        if list(out.shape) != local_shape:
            out = out.reshape(local_shape)
        return out


class _RouterPreAllGather(paddle.autograd.PyLayer):
    """Consume a routing-pair AllGather pre-issued on the comm stream.

    Counterpart of :class:`_RouterAllGather` whose gather was issued ahead by
    :func:`_prefetch_routing_ag`, so it overlaps compute instead of blocking the
    calc stream. Forward waits the task and returns the gathered pair; backward
    reduce-scatters the grad on the comm stream and waits it. ``handle`` is a
    plain dict, so backward returns a single grad for ``local_pair``.
    """

    @staticmethod
    def forward(ctx, local_pair, handle):
        handle["task"].wait()
        ctx.group = handle["group"]
        ctx.local_shape = handle["local_shape"]
        out = handle["output"]
        handle["output"] = None
        handle["task"] = None
        return out

    @staticmethod
    def backward(ctx, grad):
        group = ctx.group
        if group is None or group.nranks == 1:
            if list(grad.shape) != ctx.local_shape:
                grad = grad.reshape(ctx.local_shape)
            return grad
        out, task = _reduce_scatter_async(grad.contiguous(), group)
        task.wait()
        if list(out.shape) != ctx.local_shape:
            out = out.reshape(ctx.local_shape)
        return out


def _drain_async_handle(handle: dict | None, where: str) -> None:
    """Wait out a leftover pre-issued collective and discard its handle.

    A prefetch whose consumer was never reached -- an exception between issuing
    and consuming -- leaves an in-flight NCCL task owning an output buffer.
    Waiting before the slot is overwritten keeps the next prefetch from racing
    it. A failed wait is logged rather than raised: there is nothing left to
    recover, and raising here would mask whatever aborted the previous forward.

    Always returns None, so the caller can assign the result back over the slot
    it just drained.
    """
    if handle is None:
        return None
    try:
        handle["task"].wait()
    except (RuntimeError, OSError) as exc:
        logger.warning(
            "%s: leftover async task wait failed (%s), discarding handle.",
            where,
            exc,
        )
    return None


class _PreAllGatherResult(paddle.autograd.PyLayer):
    """Consume a pre-issued async AllGather of hidden_states.

    Forward waits for the async NCCL task and returns the filled buffer.
    Backward is ReduceScatter (dual of AllGather).
    The ``handle`` is a plain dict, not a tensor, so Paddle's PyLayer does not
    expect a gradient for it — backward returns only one value.
    """

    @staticmethod
    def forward(ctx, hidden_states, handle):
        handle["task"].wait()
        ctx.group = handle["group"]
        return handle["output"]

    @staticmethod
    def backward(ctx, grad):
        grad_input = ReduceScatterGroupOp.apply(grad, ctx.group)
        return grad_input


class _PreAllGatherFP8Result(paddle.autograd.PyLayer):
    """FP8 variant of _PreAllGatherResult.

    Forward waits for the single fused async AllGather task (fp8 data ++ block
    scale packed per token) and returns ``(x_fp8_global, scale_global)`` after
    splitting.  The fp8 tensor is consumed directly by SonicMoE's _UpProjection
    via prequant_activation_payload.

    _UpProjection.backward produces a bf16 dx for the activation input, but
    the forward output here is fp8.  set_grad_in_dtype_consistent(False) and
    set_materialize_grads(False) tell Paddle to pass the bf16 grad through
    without dtype coercion.  Backward then ReduceScatters that bf16 grad back
    to the local token shard.
    """

    @staticmethod
    def forward(ctx, hidden_states, handle):
        handle["task"].wait()
        ctx.group = handle["group"]
        x_fp8_global, scale_global = _split_fused_fp8_gather(
            handle["fused_global"],
            handle["H"],
            handle["H128"],
            handle["scale_dtype"],
        )
        ctx.set_grad_in_dtype_consistent(False)
        ctx.set_materialize_grads(False)
        return x_fp8_global, scale_global

    @staticmethod
    def backward(ctx, grad_output, grad_scale=None):
        del grad_scale
        group = ctx.group
        if grad_output is None:
            return None
        if group is None or group.nranks == 1:
            return grad_output
        grad_input = ReduceScatterGroupOp.apply(grad_output, group)
        return grad_input


def _reduce_scatter_async(input, group, out=None):
    """Async ReduceScatter (SUM, axis 0) on the comm stream.

    ``out`` optionally supplies the destination buffer, which must have shape
    ``[rows // nranks, *input.shape[1:]]`` and the input dtype; this lets the
    caller scatter directly into a slice of a larger buffer. Returns
    ``(output, task)``.
    """
    input = input.contiguous()
    out_shape = list(input.shape)
    if out_shape[0] % group.nranks != 0:
        raise ValueError(
            f"ReduceScatter input rows {out_shape[0]} not divisible by "
            f"nranks {group.nranks}"
        )
    out_shape[0] //= group.nranks
    output = (
        paddle.empty(shape=out_shape, dtype=input.dtype) if out is None else out
    )
    # Calls the ProcessGroup directly (async, comm stream) to skip the
    # per-call Python checks of the ``paddle.distributed.stream`` wrappers.
    task = group.process_group.reduce_scatter_tensor(
        output, input, _PG_REDUCE_SUM, False
    )
    return output, task


def _all_gather_async(input, group):
    """Async AllGather (axis 0) on the comm stream.

    Returns ``(output, task)``.
    """
    input = input.contiguous()
    out_shape = list(input.shape)
    out_shape[0] *= group.nranks
    output = paddle.empty(shape=out_shape, dtype=input.dtype)
    # Direct ProcessGroup call, as in `_reduce_scatter_async`.
    task = group.process_group.all_gather_into_tensor(output, input, False)
    return output, task


# Low-SM communication for the ring (paddlefleet_ops zcomm ops, see zcomm_comm.py),
# selected per dispatcher by ``TransformerConfig.ringmoe_use_zcomm``. Each helper
# below takes that switch as ``zcomm`` and otherwise runs the NCCL collective.


_ZCOMM_FALLBACK_WARNED = False


def _warn_zcomm_fallback(reason):
    """Warn (once per process) that ringmoe_use_zcomm falls back to NCCL."""
    global _ZCOMM_FALLBACK_WARNED
    if not _ZCOMM_FALLBACK_WARNED:
        _ZCOMM_FALLBACK_WARNED = True
        logger.warning(
            "ringmoe_use_zcomm is on but the zcomm ops cannot run (%s); the "
            "RingMoE dispatcher uses NCCL. Set ringmoe_use_zcomm=False to "
            "silence this.",
            reason,
        )


def _token_all_gather_async(input, group, zcomm=False):
    """Intra AllGather of a token / output grad; copy-engine pull with ``zcomm``."""
    if not zcomm:
        return _all_gather_async(input, group)
    from .zcomm_comm import IntraCE

    return IntraCE.get(group).all_gather_async(input)


def _intra_rs_async(input, group, out=None, zcomm=False, overlapped=True):
    """Intra ReduceScatter; copy engine + local sum with ``zcomm`` where hidden.

    ``overlapped`` marks a call site whose ReduceScatter a following GEMM hides.
    Only those switch to the copy engine: it leaves the GEMM its SMs, but alone
    it is a little slower than NCCL's ring, so exposed ones stay on NCCL.
    """
    if not (zcomm and overlapped and input.dtype == paddle.bfloat16):
        return _reduce_scatter_async(input, group, out=out)
    from .zcomm_comm import IntraCE

    return IntraCE.get(group).reduce_scatter_async(input, out=out)


def _inter_a2a_single(out_shape, x, out_split, in_split, group, zcomm=False):
    """Async all_to_all_single on the inter group. Returns ``(out, task)``.

    With ``zcomm`` the transfer is NCCL-GIN puts and ``out`` is a view of a
    window slot that is reissued two calls of its size later, so the caller
    must consume it within the ring round (or copy it).
    """
    if zcomm:
        from .zcomm_comm import InterGin

        return InterGin.get(group).all_to_all_single(x, out_split, in_split)
    out = paddle.empty(out_shape, dtype=x.dtype)
    task = group.process_group.all_to_all_single(
        out, x.contiguous(), out_split, in_split, False
    )
    return out, task


def _inter_all_gather_async(input, group, zcomm=False):
    """Async AllGather on the inter group; NCCL-GIN puts with ``zcomm``."""
    if not zcomm:
        return _all_gather_async(input, group)
    from .zcomm_comm import InterGin

    return InterGin.get(group).all_gather_async(input)


def _quantize_and_pack_fp8(x):
    """Quantize ``x`` to fp8 e4m3 + int32 1x128 block scale and pack into a
    single uint8 row ``[T_local, H + 4*num_scale_blocks]``
    (fp8 data ++ scale bytes).

    Returns ``(fused_local, H, num_scale_blocks, scale_dtype)``. The caller
    is responsible for AllGather (sync or async) and for unpacking via
    :func:`_split_fused_fp8_gather`. Bit-identical to two separate gather of
    data and scale: AllGather is a lossless row concat and the 1x128 scale
    lives within a single token's hidden vector, so packing along axis 1
    before the gather yields the same per-rank bytes as gathering separately.
    """
    if quantize_activation_blockscaled_fast is None:
        raise RuntimeError(
            "Cannot find quantize_activation_blockscaled_fast, "
            "please update sonicmoe."
        )
    x = x.contiguous()
    x_fp8, scale = quantize_activation_blockscaled_fast(
        x, scale_dtype=paddle.int32
    )
    _, H = x_fp8.shape
    num_scale_blocks = scale.shape[1]
    scale_dtype = scale.dtype
    fused_local = paddle.concat(
        [x_fp8.view("uint8"), scale.view("uint8")], axis=1
    ).contiguous()
    return fused_local, H, num_scale_blocks, scale_dtype


def _fused_fp8_all_gather_async(x, group):
    """Quantize local tensor to fp8 e4m3 + int32 1x128 block scale, then a SINGLE
    async AllGather of the fused (fp8-data-as-uint8 ++ scale-as-uint8) per-token
    byte row on the comm stream.  Returns
    ``(fused_global, H, num_scale_blocks, scale_dtype, task)``.

    Fusing data and scale into one AllGather (vs two back-to-back collectives)
    is bit-identical: AllGather concatenates rows along axis 0 and every rank
    carries an identical ``T_local``, so packing the two payloads along axis 1
    before the gather yields the same per-rank bytes as gathering them
    separately.  It removes one NCCL kernel launch (and its peer-wait fixed
    cost) per collective.

    Because AllGather is a lossless row concat and the 1x128 scale lives within
    a single token's hidden vector, this is bit-identical to
    bf16-AllGather-then-quantize (halving on-wire bytes vs bf16).  The caller
    waits ``task`` and then calls :func:`_split_fused_fp8_gather` to recover
    ``(data_e4m3_global, scale_global)``.

    Used both by the combine backward (``_AllGatherCombineAsync.backward``) and
    by ``AllGatherTokenDispatcher.pre_allgather`` for the forward activation.
    """
    fused_local, H, num_scale_blocks, scale_dtype = _quantize_and_pack_fp8(x)
    T_global = fused_local.shape[0] * group.nranks
    fused_global = paddle.empty([T_global, fused_local.shape[1]], dtype="uint8")
    task = paddle.distributed.stream.all_gather(
        fused_global,
        fused_local,
        group=group,
        sync_op=False,
        use_calc_stream=False,
    )
    return fused_global, H, num_scale_blocks, scale_dtype, task


def _split_fused_fp8_gather(fused_global, H, num_scale_blocks, scale_dtype):
    """Recover ``(data_e4m3_global, scale_global)`` from a fused gather buffer.

    Inverse of the packing in :func:`_fused_fp8_all_gather_async` /
    :meth:`AllGatherTokenDispatcher.pre_allgather`. Must be called only after the
    gather task has completed.
    """
    T_global = fused_global.shape[0]
    data_global_u8 = fused_global[:, :H].contiguous()
    scale_global = fused_global[:, H:].contiguous().view(scale_dtype)
    return (
        data_global_u8.view("float8_e4m3fn"),
        scale_global.reshape([T_global, num_scale_blocks]),
    )


def _fp8_split_all_gather(x, group):
    """fp8 token gather as TWO separate plain AllGathers (data + scale).

    Each payload lands in its OWN contiguous output, so there is no strided
    unpack: the fused-pack path's :func:`_split_fused_fp8_gather` did a full
    ``[T_global, H]`` ``.contiguous()`` per gather (measured the single biggest
    copy on the ring's critical path). Here the gathered data buffer is already
    contiguous ``[T_global, H]`` -- ``.view("float8_e4m3fn")`` feeds the fp8 GEMM
    directly with zero copy.

    Uses the ordinary ``all_gather_group``: the gathered fp8 data is saved for
    the expert GEMM backward, i.e. retained until this microbatch's backward
    (pipeline depth), and a regular allocator tensor retains fine.

    Bit-identical to the fused path: AllGather is a lossless row concat and each
    1x128 scale block lives within a single token's row, so nothing spans ranks.
    Backward stays with the caller (straight-through ReduceScatter of bf16 grad).
    """
    if quantize_activation_blockscaled_fast is None:
        raise RuntimeError(
            "Cannot find quantize_activation_blockscaled_fast, "
            "please update sonicmoe."
        )
    x = x.contiguous()
    x_fp8, scale = quantize_activation_blockscaled_fast(
        x, scale_dtype=paddle.int32
    )
    num_scale_blocks = scale.shape[1]
    scale_dtype = scale.dtype
    if group is None or group.nranks == 1:
        return x_fp8, scale
    data_global = all_gather_group(x_fp8.view("uint8"), group=group)
    scale_global = all_gather_group(scale.view("uint8"), group=group)
    T_global = data_global.shape[0]
    return (
        data_global.view("float8_e4m3fn"),
        scale_global.view(scale_dtype).reshape([T_global, num_scale_blocks]),
    )


class _AllGatherCombineAsync(paddle.autograd.PyLayer):
    """Fuse the combine ReduceScatter with a shared-expert subgraph for overlap.

    Forward:
      1. Issue async ReduceScatter of expert output on comm stream.
      2. Run shared-expert subgraph on calc stream (concurrent with step 1).
      3. Wait for ReduceScatter, return (combined_x,) + fn_out.

    Backward (dual):
      - bf16 path: async AllGather of grad on comm stream while shared-expert
        backward runs on calc stream.
      - fp8 path (fp8_combine_grad_handle != None): quantize local grad to fp8,
        async AllGather both data and scale on comm stream, write results into
        the handle for _DownProjection.backward to consume directly.

    Overlap is safe because fn's inputs are independent of x (expert output).
    """

    @staticmethod
    def forward(
        ctx,
        x,
        group,
        *fn_args,
        fn,
        is_first_fwd=False,
        fp8_combine_grad_handle=None,
    ):
        if fn is None:
            raise ValueError(
                "_AllGatherCombineAsync requires a non-None fn for overlap."
            )
        ctx.group = group
        ctx.fp8_combine_grad_handle = fp8_combine_grad_handle
        if fp8_combine_grad_handle is not None:
            ctx.set_grad_in_dtype_consistent(False)
            ctx.set_materialize_grads(False)

        if group is None or group.nranks == 1:
            combined_x = x.clone()
            ctx.bwf, fn_out = manual_backward(fn, is_first_fwd, *fn_args)
            return (combined_x,) + fn_out  # noqa: RUF005

        combined_x, task = _reduce_scatter_async(x, group)
        ctx.bwf, fn_out = manual_backward(fn, is_first_fwd, *fn_args)
        task.wait()

        return (combined_x,) + fn_out  # noqa: RUF005

    @staticmethod
    def backward(ctx, grad_output, *fn_out_grads):
        group = ctx.group
        handle = ctx.fp8_combine_grad_handle
        if group is None or group.nranks == 1:
            grad_x = grad_output.clone()
            fn_args_grads = ctx.bwf(*fn_out_grads)
            return (grad_x,) + fn_args_grads  # noqa: RUF005

        if handle is not None:
            fused_global, _H, _num_scale_blocks, _sdt, task = (
                _fused_fp8_all_gather_async(grad_output, group)
            )
            fn_args_grads = ctx.bwf(*fn_out_grads)
            task.wait()
            data_e4m3, scale_global = _split_fused_fp8_gather(
                fused_global, _H, _num_scale_blocks, _sdt
            )
            handle["data"] = data_e4m3
            handle["scale"] = scale_global
            return (data_e4m3,) + fn_args_grads  # noqa: RUF005

        grad_x, task = _all_gather_async(grad_output, group)
        fn_args_grads = ctx.bwf(*fn_out_grads)
        task.wait()
        return (grad_x,) + fn_args_grads  # noqa: RUF005


class _AllGatherCombineNoOverlap(paddle.autograd.PyLayer):
    """ReduceScatter combine without overlap subgraph (sync on calc stream).

    Mirrors ``_AllGatherCombineAsync``'s fp8/bf16 grad collection but skips the
    shared-expert overlap. All collectives use the same sync calc-stream
    wrappers (``reduce_scatter_group`` / ``all_gather_group``) as
    :class:`ReduceScatterGroupOp` and the ``_PreAllGather*Result`` backward
    paths, so they FIFO-order naturally after the expert MLP forward and
    ``_DownProjection.backward`` without cross-stream event synchronization.
    Needed because the no-overlap path must still populate
    ``fp8_combine_grad_handle`` in backward for ``_DownProjection.backward``
    to consume; previously the no-overlap branch returned ``hidden_states``
    directly and dropped the handle.
    """

    @staticmethod
    def forward(ctx, x, group, fp8_combine_grad_handle=None):
        ctx.group = group
        ctx.fp8_combine_grad_handle = fp8_combine_grad_handle
        if fp8_combine_grad_handle is not None:
            ctx.set_grad_in_dtype_consistent(False)
            ctx.set_materialize_grads(False)
        if group is None:
            return x.clone()
        return reduce_scatter_group(x, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        group = ctx.group
        handle = ctx.fp8_combine_grad_handle
        if group is None:
            return grad_output.clone()
        if handle is not None:
            fused_local, H, num_scale_blocks, scale_dtype = (
                _quantize_and_pack_fp8(grad_output)
            )
            fused_global = all_gather_group(fused_local, group=group)
            data_e4m3, scale_global = _split_fused_fp8_gather(
                fused_global, H, num_scale_blocks, scale_dtype
            )
            handle["data"] = data_e4m3
            handle["scale"] = scale_global
            return data_e4m3
        return all_gather_group(grad_output, group=group)


def _raw_inter_shift_async(x, group, dst, src, handle, zcomm=False):
    """Async cyclic ring shift of ``x``: send to ``dst``, receive from ``src``.

    Value-only (no autograd node); the reverse is the same call with ``dst``
    and ``src`` swapped, driven by the ring's hand-written backward. The task
    is stored in ``handle["task"]`` and must be drained before the result is
    read.
    """
    nranks = group.nranks
    rows = x.shape[0]
    in_split = [0] * nranks
    in_split[dst] = rows
    out_split = [0] * nranks
    out_split[src] = rows
    out, handle["task"] = _inter_a2a_single(
        list(x.shape), x, out_split, in_split, group, zcomm=zcomm
    )
    return out


def _add_hop(acc, hop):
    """``acc + hop`` for a received reverse hop, never returning ``hop`` itself.

    ``hop`` may be a view of a zcomm window slot that is reissued a couple of
    transfers later, so it must not become a grad that outlives the backward
    step; a missing ``acc`` copies it instead of aliasing it.
    """
    if hop is None:
        return acc
    return hop.clone() if acc is None else acc + hop


def _raw_fp8_shift_async(data, scale, group, dst, src, handle):
    """Async cyclic ring shift of an fp8 token (data and block scale).

    fp8 counterpart of :func:`_raw_inter_shift_async`, issuing one all-to-all
    per tensor; both tasks are stored in ``handle["tasks"]``. Value-only: the
    backward shifts the bf16 data grad back with
    :func:`_raw_inter_shift_async`, and the scale carries no gradient. Returns
    ``(data_out, scale_out)``.
    """
    tasks = []
    outs = []
    for tensor in (data, scale):
        handle_one = {}
        outs.append(_raw_inter_shift_async(tensor, group, dst, src, handle_one))
        tasks.append(handle_one["task"])
    handle["tasks"] = tasks
    return outs[0], outs[1]


_BYTE_CARRIER_DTYPES = (
    paddle.uint8,
    paddle.float8_e4m3fn,
    paddle.float8_e5m2,
)


class _ByteGradCapture(paddle.autograd.PyLayer):
    """Alias of a byte-carrier (uint8/fp8) leaf that captures its bf16 grad.

    Autograd would cast an incoming grad to the leaf's dtype, corrupting a bf16
    grad on a uint8/fp8 leaf. Instead the grad is accumulated into
    ``box["grad"]`` and the leaf itself receives none. The alias shares the
    input's storage when it is contiguous and is a clone otherwise.
    """

    @staticmethod
    def forward(ctx, input, box):
        ctx.box = box
        ctx.set_grad_in_dtype_consistent(False)
        ctx.set_materialize_grads(False)
        if input.is_contiguous():
            fake_output = paddle.Tensor()
            fake_output.get_tensor()._share_data_nocheck_with(
                input.get_tensor()
            )
        else:
            fake_output = input.clone()
        return fake_output

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is not None:
            prev_grad = ctx.box.get("grad")
            ctx.box["grad"] = (
                grad_output if prev_grad is None else prev_grad + grad_output
            )


class _GradRoot(paddle.autograd.PyLayer):
    """Scalar root over a manual-backward segment's outputs.

    ``paddle.autograd.backward(outs, grads)`` deep-copies every root grad on
    the calc stream. Starting from this root instead, its backward hands the
    grads parked in ``box["grads"]`` to the outputs as they are.
    """

    @staticmethod
    def forward(ctx, box, *outs):
        ctx.box = box
        return paddle.empty([1], dtype="float32")

    @staticmethod
    def backward(ctx, _):
        return tuple(ctx.box.pop("grads"))


def _manual_backward(f: Callable, is_first_fwd: bool, *args: Any):
    """``moe_utils.manual_backward`` with support for byte-carrier leaves.

    Same contract (returns ``(bwd_fn, outputs)``), but uint8/fp8 tensor inputs
    that require grad go through :class:`_ByteGradCapture`, so ``bwd_fn``
    returns their bf16 grads instead of dtype-cast ones.
    """
    tracer = framework._dygraph_tracer()
    orig_has_grad = tracer._has_grad
    if not is_first_fwd:
        tracer._has_grad = True

    detached_args = detach_and_requires_grad_(*args)
    grad_boxes = [
        {}
        if is_tensor(arg)
        and not arg.stop_gradient
        and arg.dtype in _BYTE_CARRIER_DTYPES
        else None
        for arg in detached_args
    ]
    detached_args_clone = [
        (
            _ByteGradCapture.apply(arg, box)
            if box is not None
            else FakeClone.apply(arg)
        )
        if is_tensor(arg)
        else arg
        for arg, box in zip(detached_args, grad_boxes)
    ]
    out = f(*detached_args_clone)
    if isinstance(out, list):
        out = tuple(out)
    elif not isinstance(out, tuple):
        out = (out,)

    if is_first_fwd:
        tracer._has_grad = orig_has_grad
        return None, out

    out_cached = [FakeClone.apply(o) for o in out if o is not None]
    # Root over the grad-carrying outputs; see `_GradRoot`.
    root_box = {}
    root_outs = [o for o in out_cached if not o.stop_gradient]
    root = _GradRoot.apply(root_box, *root_outs) if root_outs else None
    for o in out_cached:
        o._clear_dataptr()
    tracer._has_grad = orig_has_grad

    def bwd_f(*grads):
        nonlocal out_cached, detached_args, f
        grads = [g for g in grads if g is not None]
        assert grads and out_cached, (len(grads), len(out_cached))
        grads, out_cached = zip(
            *[(g, o) for g, o in zip(grads, out_cached) if not o.stop_gradient]
        )
        assert len(grads) == len(out_cached), (len(grads), len(out_cached), f)
        if root is not None and len(grads) == len(root_outs):
            root_box["grads"] = grads
            paddle.autograd.backward([root])
        else:
            paddle.autograd.backward(out_cached, grads)
        return tuple(
            box.get("grad") if box is not None else leaf.grad
            for leaf, box in zip(detached_args, grad_boxes)
            if is_tensor(leaf)
        )

    return bwd_f, out


class _RingRoundsExplicit(paddle.autograd.PyLayer):
    """Ring forward as a single op with a hand-scheduled backward.

    ``plan(*diff_leaves)`` runs the forward and returns ``(out, bwd)``, where
    ``bwd(*out_grads)`` returns one grad per entry of ``diff_leaves``. Driving
    the backward by hand lets each round's reverse collectives overlap an
    adjacent round's expert backward. ``relax_dtype`` disables grad dtype
    consistency and materialization for byte-carrier leaves.
    """

    @staticmethod
    def forward(ctx, *diff_leaves, plan, relax_dtype):
        if relax_dtype:
            ctx.set_grad_in_dtype_consistent(False)
            ctx.set_materialize_grads(False)
        out, ctx.bwd = plan(*diff_leaves)
        return out

    @staticmethod
    def backward(ctx, *out_grads):
        return ctx.bwd(*out_grads)


def _alias(x):
    """``x`` as a new tensor sharing its storage, or a clone if it is not
    contiguous."""
    if x.is_contiguous():
        out = paddle.Tensor()
        out.get_tensor()._share_data_nocheck_with(x.get_tensor())
        return out
    return x.clone()


class _RingTokHopJoin(paddle.autograd.PyLayer):
    """Identity on the ring's token carrier and the router input that
    finishes the ring's last token hop.

    The ring backward's final link hops the remote rounds' token grad back
    home. Waiting for it inside the ring op would hold back the router
    backward, which only needs the routing grad. Instead the ring op
    leaves the hop in flight (with zcomm also the last token-grad
    ReduceScatter, which the hop is ordered after on the comm side) and
    records it in ``slot``. This op aliases both
    the carrier and the router input, so autograd runs its backward only
    after both grads arrived, i.e. after the router backward is enqueued. It
    then waits for the hop and adds its result to the carrier grad.
    """

    @staticmethod
    def forward(ctx, carrier, gate_input, slot):
        ctx.slot = slot
        return _alias(carrier), _alias(gate_input)

    @staticmethod
    def backward(ctx, carrier_grad, gate_grad):
        slot = ctx.slot
        ctx.slot = None
        if slot:
            RingMoETokenDispatcher._drain([slot["hop"]])
            # Waiting for the hop covers everything ordered before it on the
            # comm side: the deferred ReduceScatter and the comm-stream add.
            # Their zcomm tasks withdraw their stream registrations now.
            for kept in slot.get("keep") or ():
                if hasattr(kept, "covered"):
                    kept.covered()
            rev = slot["rev"].reshape(carrier_grad.shape)
            if rev.dtype != carrier_grad.dtype:
                rev = rev.astype(carrier_grad.dtype)
            slot.clear()
            carrier_grad = carrier_grad + rev
        return carrier_grad, gate_grad


@paddle.no_grad()
def _tokens_per_expert_histogram(indices, num_experts):
    """Count tokens per expert without a device-to-host synchronization.

    ``indices`` is a ``[..., K]`` routing tensor where ``-1`` marks a dropped
    slot. Returns an int32 ``[num_experts]`` histogram equal to
    ``bincount(indices[indices >= 0], minlength=num_experts)``. Dropped and
    out-of-range slots go to an extra sink bin that is sliced off (as in the
    Triton kernel), so every shape is static and no host sync is needed
    (unlike ``masked_select``/``bincount``). The ring dispatcher uses the
    Triton kernel ``tokens_per_expert_histogram_triton`` instead; this eager
    version serves the other dispatchers.
    """
    flat = indices.reshape([-1]).cast("int64")
    sink = paddle.full_like(flat, num_experts)
    clamped = paddle.where((flat >= 0) & (flat < num_experts), flat, sink)
    # Non-overwriting scatter accumulates a 1 per slot into its expert's bin.
    counts = paddle.zeros([num_experts + 1], dtype="int32")
    ones = paddle.ones([flat.shape[0]], dtype="int32")
    counts = paddle.scatter(counts, clamped, ones, overwrite=False)
    return counts[:num_experts]


class AllGatherTokenDispatcher(nn.Layer):
    """AllGather + ReduceScatter EP dispatcher (SonicMoE fused-kernel only).

    Every expert is sharded along its intermediate dim into EP partitions;
    every rank holds all experts but only its I/EP shard of each.  The forward
    data flow is:

        [T_local, H] --AllGather--> [T_global, H] --SonicMoE fused
        _UpProjection / _DownProjection (all tokens, partial I)-->
        [T_global, H] --ReduceScatter(SUM)--> [T_local, H]

    Routing metadata (topk_indices, topk_weights) is also AllGathered so every
    rank sees the same global token assignments.  No explicit permute/unpermute
    is needed — the SonicMoE fused kernels handle token gather/scatter
    internally based on the indices.
    """

    def __init__(
        self,
        moe_group: Group,
        expert_model_parallel_size: int,
        num_experts: int,
        fp8_dispatch: bool = False,
        use_ue8m0: bool = False,
    ):
        nn.Layer.__init__(self)
        self.moe_group = moe_group
        self.ep_size = expert_model_parallel_size
        self.num_experts = num_experts
        self.num_local_experts = num_experts  # every rank holds all experts
        self.fp8_dispatch = fp8_dispatch
        self.use_ue8m0 = use_ue8m0
        self._pre_ag_handle: dict | None = None
        self._global_topk_indices = None
        self._global_topk_weights = None
        self._fp8_dispatch_scale = None
        self._overlap_combined = None
        # The routing mask/cast and the histogram use the ring's Triton
        # kernels when Triton is available, else their eager equivalents.
        # Importing any triton_ops submodule runs the package __init__, which
        # imports triton, so without triton the import itself fails; this
        # dispatcher must stay usable then, on the eager path.
        try:
            from paddlefleet.triton_ops.utils import is_triton_available
        except ImportError:
            self._use_triton = False
        else:
            self._use_triton = is_triton_available()

    def pre_allgather(self, hidden_states: paddle.Tensor):
        """Issue an async AllGather of hidden_states on the comm stream.

        Called before gate computation so the AllGather overlaps with the gate
        MLP on the calc stream.  Result is stored in self._pre_ag_handle and
        consumed by dispatch_preprocess.

        bf16 path: single async AllGather.
        fp8 path: quantize + a single fused async AllGather of (data ++ scale)
            via ``_fused_fp8_all_gather_async``.
        """
        if self.moe_group is None or self.moe_group.nranks == 1:
            self._pre_ag_handle = None
            return

        # Drain leftover handle from a possibly-aborted previous forward.
        self._pre_ag_handle = _drain_async_handle(
            self._pre_ag_handle, "pre_allgather"
        )

        if len(hidden_states.shape) == 3:
            _, _, d_model = hidden_states.shape
        else:
            _, d_model = hidden_states.shape
        reshaped_input = hidden_states.reshape([-1, d_model]).contiguous()

        if self.fp8_dispatch:
            (
                fused_global,
                H,
                num_scale_blocks,
                scale_dtype,
                task,
            ) = _fused_fp8_all_gather_async(reshaped_input, self.moe_group)
            self._pre_ag_handle = {
                "fused_global": fused_global,
                "H": H,
                "H128": num_scale_blocks,
                "scale_dtype": scale_dtype,
                "task": task,
                "group": self.moe_group,
                "fp8": True,
            }
            return

        output_shape = list(reshaped_input.shape)
        output_shape[0] = output_shape[0] * self.moe_group.nranks
        global_hidden_states = paddle.empty(
            shape=output_shape, dtype=reshaped_input.dtype
        )
        task = paddle.distributed.stream.all_gather(
            global_hidden_states,
            reshaped_input,
            group=self.moe_group,
            sync_op=False,
            use_calc_stream=False,
        )

        self._pre_ag_handle = {
            "output": global_hidden_states,
            "task": task,
            "group": self.moe_group,
        }

    def dispatch_preprocess(
        self,
        hidden_states: paddle.Tensor,
        probs: paddle.Tensor,
        mask: paddle.Tensor,  # routing_map
        topk_weights: paddle.Tensor | None = None,
        topk_indices: paddle.Tensor | None = None,
    ) -> paddle.Tensor:
        """AllGather hidden_states and routing metadata across the EP group.

        Steps:
        1. Reshape to [T_local, H].
        2. AllGather hidden_states (reuse pre-issued async handle if available).
        3. AllGather topk_indices async on comm stream (int32, no gradient).
        4. AllGather topk_weights via _RouterAllGather on calc stream (has
           gradient, backward = reduce-scatter).
        5. Wait for indices, build padding mask (indices < 0), zero padding weights.
        6. Return global hidden_states (unpermuted — SonicMoE handles gather).

        Caches _global_topk_indices, _global_topk_weights for downstream use.

        Note:
            If ``_pre_ag_handle`` is None on entry (gate-overlap did not fire,
            e.g. ``moe_allgather_gate_overlap=False`` or a direct call that
            bypasses ``_maybe_pre_allgather_overlap``) and ``fp8_dispatch`` is
            True, this method issues the fp8 AllGather inline via
            ``pre_allgather`` and immediately waits on it.  This is correct but
            forfeits the gate-compute overlap that the pre-issued path provides.
        """
        if len(hidden_states.shape) == 3:
            _, _, d_model = hidden_states.shape
        else:
            _, d_model = hidden_states.shape
        reshaped_input = hidden_states.reshape([-1, d_model]).contiguous()

        self._fp8_dispatch_scale = None

        if self._pre_ag_handle is not None:
            if self._pre_ag_handle.get("fp8", False):
                global_hidden_states, self._fp8_dispatch_scale = (
                    _PreAllGatherFP8Result.apply(
                        reshaped_input, self._pre_ag_handle
                    )
                )
            else:
                global_hidden_states = _PreAllGatherResult.apply(
                    reshaped_input, self._pre_ag_handle
                )
            self._pre_ag_handle = None
        elif self.fp8_dispatch:
            self.pre_allgather(reshaped_input)
            global_hidden_states, self._fp8_dispatch_scale = (
                _PreAllGatherFP8Result.apply(
                    reshaped_input, self._pre_ag_handle
                )
            )
            self._pre_ag_handle = None
        else:
            global_hidden_states = AllGatherGroupOp.apply(
                reshaped_input, self.moe_group
            )

        if topk_indices is None or topk_weights is None:
            raise ValueError(
                "AllGatherTokenDispatcher requires topk_indices and "
                "topk_weights to be provided."
            )
        if self._use_triton:
            # One fused kernel casts the indices to int32 and the weights to
            # probs dtype and zeroes the weights (and their grads) of padding
            # lanes (idx < 0), on the local [T_local, K] pair before the
            # gathers instead of on the EP-times larger gathered one.
            from paddlefleet.triton_ops.mask_cast_routing_w import (
                mask_cast_routing_w,
            )

            topk_indices_i32, topk_weights = mask_cast_routing_w(
                topk_indices.detach(), topk_weights, probs.dtype
            )
        else:
            topk_indices_i32 = topk_indices.detach().cast("int32")
            topk_weights = topk_weights.cast(probs.dtype)
        # AllGather indices as int32 on comm stream (async, no gradient).
        # Issued before weights AllGather so both collectives are in flight.
        topk_indices_i32 = topk_indices_i32.contiguous()
        if self.moe_group is None or self.moe_group.nranks == 1:
            self._global_topk_indices = topk_indices_i32.clone()
            _idx_task = None
        else:
            _idx_out_shape = list(topk_indices_i32.shape)
            _idx_out_shape[0] *= self.moe_group.nranks
            self._global_topk_indices = paddle.empty(
                shape=_idx_out_shape, dtype=topk_indices_i32.dtype
            )
            _idx_task = paddle.distributed.stream.all_gather(
                self._global_topk_indices,
                topk_indices_i32,
                group=self.moe_group,
                sync_op=False,
                use_calc_stream=False,
            )
        # AllGather router weights on calc stream (has gradient).
        self._global_topk_weights = _RouterAllGather.apply(
            topk_weights, self.moe_group
        )
        # Wait for indices AllGather right before its first consumer.
        if _idx_task is not None:
            _idx_task.wait()
        if not self._use_triton:
            # Build padding mask and zero corresponding weights.
            padding_mask = self._global_topk_indices < 0
            self._global_topk_weights = paddle.where(
                padding_mask,
                paddle.zeros_like(self._global_topk_weights),
                self._global_topk_weights,
            )
        self.tokens_per_expert = None
        return global_hidden_states

    def token_dispatch(
        self,
        permuted_global_input_tokens: paddle.Tensor,
        fp8_dispatch: bool = False,
        async_finish: bool = False,
        use_ue8m0: bool = False,
        using_sonic_moe: bool = True,
    ):
        """No-op pass-through.  AllGather already happened in dispatch_preprocess,
        so every rank already holds the full global token list.  Returns
        (tokens, fp8_handle) where fp8_handle carries the dispatch scale if
        fp8_dispatch is active."""
        if not using_sonic_moe:
            raise ValueError(
                "AllGatherTokenDispatcher requires using_sonic_moe=True; "
                "the AllGather path is only wired for the fused SonicMoE "
                "expert kernels. Switch dispatcher type or enable SonicMoE."
            )
        fp8_handle = (
            {"scale": self._fp8_dispatch_scale}
            if self._fp8_dispatch_scale is not None
            else None
        )
        return permuted_global_input_tokens, fp8_handle

    def get_dispatched_routing(self):
        """Return (global_indices, global_weights, tokens_per_expert).

        tokens_per_expert uses a sync-free histogram (the fused Triton
        kernel when available, else :func:`_tokens_per_expert_histogram`) —
        no GPU->CPU sync, no full one-hot materialization.
        """
        if self._use_triton:
            from paddlefleet.triton_ops.tokens_per_expert_hist import (
                tokens_per_expert_histogram_triton as histogram,
            )
        else:
            histogram = _tokens_per_expert_histogram
        tokens_per_expert = histogram(
            self._global_topk_indices, self.num_experts
        )
        return (
            self._global_topk_indices,
            self._global_topk_weights,
            tokens_per_expert,
        )

    def dispatch_postprocess(
        self,
        global_input_tokens: paddle.Tensor,
    ):
        """Return (global_tokens, tokens_per_expert). tokens_per_expert is None
        on this path — SonicMoE kernels recompute it from indices."""
        return global_input_tokens, self.tokens_per_expert

    def combine_preprocess(self, hidden_states: paddle.Tensor):
        """No-op pass-through."""
        return hidden_states

    def token_combine(
        self,
        hidden_states: paddle.Tensor,
        combine_overlap_handle: dict | None = None,
        async_finish: bool = False,
        fp8_combine_grad_handle: dict | None = None,
    ):
        """Combine expert outputs via ReduceScatter.

        If combine_overlap_handle is provided, fuse the ReduceScatter with the
        shared-expert subgraph via _AllGatherCombineAsync for overlap.  The
        combined output is cached for combine_postprocess to return.

        fp8_combine_grad_handle, when non-None, enables fp8 quantization of the
        combine backward gradient (halves bandwidth vs bf16).  The gathered fp8
        data+scale are written into the handle for _DownProjection.backward.
        """
        if combine_overlap_handle is None:
            # Must wrap in a PyLayer so backward populates
            # fp8_combine_grad_handle for _DownProjection.backward.
            combined_x = _AllGatherCombineNoOverlap.apply(
                hidden_states, self.moe_group, fp8_combine_grad_handle
            )
            self._overlap_combined = combined_x
            return combined_x
        if not isinstance(combine_overlap_handle, dict):
            raise TypeError(
                "combine_overlap_handle must be a dict, got "
                f"{type(combine_overlap_handle).__name__}"
            )
        if (
            "fn" not in combine_overlap_handle
            or "fn_args" not in combine_overlap_handle
        ):
            raise ValueError(
                "combine_overlap_handle must contain 'fn' and 'fn_args' keys"
            )
        if not isinstance(combine_overlap_handle["fn_args"], tuple):
            raise TypeError(
                "combine_overlap_handle['fn_args'] must be a tuple, got "
                f"{type(combine_overlap_handle['fn_args']).__name__}"
            )
        from paddle import framework as _framework

        combined_x, *fn_out = _AllGatherCombineAsync.apply(
            hidden_states,
            self.moe_group,
            *(combine_overlap_handle["fn_args"]),
            fn=combine_overlap_handle["fn"],
            is_first_fwd=not _framework._dygraph_tracer()._has_grad,
            fp8_combine_grad_handle=fp8_combine_grad_handle,
        )
        combine_overlap_handle["fn_out"] = tuple(fn_out)
        self._overlap_combined = combined_x
        return combined_x

    def combine_postprocess(self, hidden_states: paddle.Tensor):
        """Return cached ReduceScatter result from token_combine.

        token_combine sets _overlap_combined on both paths (overlap and
        no-overlap), so the fallback ReduceScatterGroupOp is defensive only.
        """
        if getattr(self, "_overlap_combined", None) is not None:
            out = self._overlap_combined
            self._overlap_combined = None
            return out
        return ReduceScatterGroupOp.apply(hidden_states, self.moe_group)


_RING_SUBGROUP_CACHE: dict = {}

# Number of GPUs per node (G) used to split the EP group into intra-node and
# inter-node levels. Fixed rather than detected: it must match the NVLink
# domain size, and an overestimate would silently treat inter-node links as
# intra-node.
_RING_GPUS_PER_NODE = 8

# Ensures the "_order_after unavailable" warning is logged only once.
_ORDER_AFTER_WARNED = False

# (id(dst_group), id(src_group)) -> (dst_stream, src_stream, event, dst_group,
# src_group). Caches the stream lookup and event creation off the per-round
# issue path. Reusing one event is safe because ``wait_event`` captures the
# event's state at call time; the group references keep the ids in the key
# from being reused.
_ORDER_AFTER_CACHE: dict = {}


def _order_after(dst_group, src_group):
    """Make ``dst_group``'s comm stream wait for all work on ``src_group``'s.

    A comm-to-comm dependency: unlike ``task.wait()``, the calc stream is not
    blocked. A group's comm stream only exists after its first collective, so
    if either stream is unavailable this logs a warning once and returns False
    so the caller can fall back to a calc-side wait. Returns True on success.
    """
    key = (id(dst_group), id(src_group))
    cached = _ORDER_AFTER_CACHE.get(key)
    if cached is not None:
        dst_stream, src_stream, event = cached[:3]
        event.record(src_stream)
        dst_stream.wait_event(event)
        return True

    from paddle import framework as _framework

    place = _framework._current_expected_place()
    try:
        dst_stream = dst_group.process_group.get_stream(place)
        src_stream = src_group.process_group.get_stream(place)
    except Exception as exc:
        global _ORDER_AFTER_WARNED
        if not _ORDER_AFTER_WARNED:
            logger.warning(
                "RingMoE comm-side ordering unavailable: no comm stream yet "
                "for one of the groups (%s); callers fall back to a wait. The "
                "group needs one collective before get_stream works -- see "
                "the warm-up in _build_ring_subgroups.",
                exc,
            )
            _ORDER_AFTER_WARNED = True
        return False
    dst_stream.wait_stream(src_stream)
    _ORDER_AFTER_CACHE[key] = (
        dst_stream,
        src_stream,
        framework.core.CUDAEvent(),
        dst_group,
        src_group,
    )
    return True


def _calc_stream_all_gather(input, group):
    """Synchronous axis-0 AllGather on the calc stream.

    Same as :func:`all_gather_group`, but also accepts ``group=None``, which
    returns a clone of ``input``.
    """
    if group is None or group.nranks == 1:
        return input.clone()
    return all_gather_group(input, group=group)


def _build_ring_subgroups(moe_group, gpus_per_node: int = _RING_GPUS_PER_NODE):
    """Split the EP group into intra-node and inter-node sub-groups.

    Assumes EP ranks are node-contiguous, with ``G = min(gpus_per_node, EP)``
    ranks per node and ``N = EP // G`` nodes. Returns ``(G, N, intra_group,
    inter_group, intra_rs_group, intra_ag_group, intra_rt_group,
    inter_cb_group)``; groups are None where the level has size 1 or this rank
    is not a member. Results are cached per (EP ranks, G).

    ``all_gather_object`` and ``new_group`` are world collectives, so every
    rank must call this in the same order. Each rank contributes its own EP
    group's partition and the union is deduplicated and sorted, so all ranks
    create identical groups in identical order. Ranks without MoE layers would
    not reach this lazily, hence :func:`init_ring_subgroups`.
    """
    ep_ranks = list(moe_group.ranks)
    ep_size = len(ep_ranks)
    G = min(gpus_per_node, ep_size)
    if ep_size % G != 0:
        raise ValueError(
            f"RingMoE requires EP size ({ep_size}) to be a multiple of the "
            f"per-node GPU count ({G}), so that EP == N*G. Pick an EP degree "
            f"that divides evenly across whole machines."
        )
    N = ep_size // G

    key = (tuple(ep_ranks), G)
    if key in _RING_SUBGROUP_CACHE:
        return _RING_SUBGROUP_CACHE[key]

    intra_lists = (
        [ep_ranks[n * G : (n + 1) * G] for n in range(N)] if G > 1 else []
    )
    inter_lists = (
        [[ep_ranks[n * G + j] for n in range(N)] for j in range(G)]
        if N > 1
        else []
    )

    gathered: list = []
    paddle.distributed.all_gather_object(
        gathered, {"intra": intra_lists, "inter": inter_lists}
    )

    def _unique_sorted(kind: str):
        seen, rank_lists = set(), []
        for proposal in gathered:
            for rank_list in proposal[kind]:
                ranks_key = tuple(sorted(rank_list))
                if len(ranks_key) > 1 and ranks_key not in seen:
                    seen.add(ranks_key)
                    rank_lists.append(list(ranks_key))
        rank_lists.sort()
        return rank_lists

    global_rank = paddle.distributed.get_rank()
    intra_group = inter_group = intra_rs_group = intra_ag_group = (
        intra_rt_group
    ) = inter_cb_group = None
    for rank_list in _unique_sorted("intra"):
        sub_group = paddle.distributed.new_group(ranks=rank_list)
        # Extra groups over the same ranks give the output ReduceScatter and
        # the routing gather-ahead their own communicators and streams, so
        # they do not serialize behind the token AllGather. The token
        # gather-ahead reuses the intra group. Created on every rank
        # unconditionally to keep the world-collective sequence identical.
        rs_group = paddle.distributed.new_group(ranks=rank_list)
        ag_group = sub_group
        rt_group = paddle.distributed.new_group(ranks=rank_list)
        if global_rank in rank_list:
            intra_group = sub_group
            intra_rs_group = rs_group
            intra_ag_group = ag_group
            intra_rt_group = rt_group
    for rank_list in _unique_sorted("inter"):
        sub_group = paddle.distributed.new_group(ranks=rank_list)
        # Separate communicator for the first call's routing pre-gather (and
        # its backward ReduceScatter) and the weight-grad hops, so they
        # run beside the token hops on the inter group's stream instead of
        # queueing behind them. Created unconditionally, as above.
        cb_group = paddle.distributed.new_group(ranks=rank_list)
        if global_rank in rank_list:
            inter_group = sub_group
            inter_cb_group = cb_group

    # Communicators and their streams are created lazily on the first
    # collective, but `_order_after` needs these groups' streams before they
    # issue anything. Run one all_reduce on each now; every member does the
    # same, so this cannot hang.
    for warm_group in (intra_ag_group, intra_rt_group, inter_cb_group):
        if warm_group is not None:
            warm_buf = paddle.zeros([1], dtype="float32")
            paddle.distributed.all_reduce(warm_buf, group=warm_group)

    logger.info("RingMoE ring topology: G=%d N=%d", G, N)

    result = (
        G,
        N,
        intra_group,
        inter_group,
        intra_rs_group,
        intra_ag_group,
        intra_rt_group,
        inter_cb_group,
    )
    _RING_SUBGROUP_CACHE[key] = result
    return result


def init_ring_subgroups(moe_group=None, gpus_per_node: int | None = None):
    """Create the RingMoE sub-groups on every rank ahead of time.

    Group creation is a world collective, but dispatchers only exist on
    pipeline stages that own a MoE layer; creating the groups lazily there
    would deadlock. The model builder calls this on every rank so the groups
    are cached before any dispatcher is built. Idempotent; returns the
    :func:`_build_ring_subgroups` tuple, or None without a multi-rank EP
    group.
    """
    if gpus_per_node is None:
        gpus_per_node = _RING_GPUS_PER_NODE
    if moe_group is None:
        from paddlefleet import parallel_state

        moe_group = parallel_state.get_expert_model_parallel_group(
            check_initialized=False
        )
    if moe_group is None or moe_group.nranks == 1:
        return None
    return _build_ring_subgroups(moe_group, gpus_per_node)


class _FP8TokenQuant(paddle.autograd.PyLayer):
    """Quantize the local token once to fp8 data and block scale.

    Both are returned as uint8 views; the ring moves these bytes through
    every hop and intra gather, so no round re-quantizes. ``meta`` receives
    ``nsb`` (scale blocks per row) and ``sdt`` (scale dtype) to reinterpret
    the scale bytes. Backward is straight-through: the bf16 grad is the token
    grad, and the scale has no gradient.
    """

    @staticmethod
    def forward(ctx, tok, meta):
        if quantize_activation_blockscaled_fast is None:
            raise RuntimeError(
                "Cannot find quantize_activation_blockscaled_fast, "
                "please update sonicmoe."
            )
        x = tok.contiguous()
        x_fp8, scale = quantize_activation_blockscaled_fast(
            x, scale_dtype=paddle.int32
        )
        meta["nsb"] = scale.shape[1]
        meta["sdt"] = scale.dtype
        ctx.tok_shape = list(tok.shape)
        ctx.tok_dtype = tok.dtype
        ctx.set_grad_in_dtype_consistent(False)
        ctx.set_materialize_grads(False)
        return x_fp8.view("uint8"), scale.view("uint8")

    @staticmethod
    def backward(ctx, grad, grad_scale=None):
        del grad_scale
        if grad is None:
            return None
        # The incoming grad is already the bf16 grad of the local token,
        # accumulated by autograd over all its consumers.
        return grad.astype(ctx.tok_dtype).reshape(ctx.tok_shape)


def _fp8_pair(data, scale, meta):
    """Bundle fp8 token data and scale for :func:`_prefetch_tok_ag_fp8`.

    Pure bookkeeping with no copy, so it is safe on buffers whose transfer is
    still in flight as long as the reader is ordered after that transfer.
    """
    return {
        "data": data,
        "scale": scale,
        "sdt": meta["sdt"],
        "nsb": meta["nsb"],
    }


def _combine_a2a_issue(partials, group, handle, send=None, zcomm=False):
    """Issue the all-to-all half of the inter-node combine.

    The combine is a ReduceScatter split into an all-to-all plus a local sum:
    the home chunk ``partials[rank]`` is never sent, and work enqueued before
    the sum overlaps the transfer. The caller later adds the received chunks
    to the home chunk.

    ``send`` optionally holds the peer chunks prebuilt in ascending rank
    order; otherwise it is built from ``partials`` (only shapes are read
    from the home entry). Value-only: the ring backward handles the grad.
    The task, receive buffer and layout are stored in ``handle``.
    """
    nranks, rank = group.nranks, group.rank
    T = partials[0].shape[0]
    peers = [peer for peer in range(nranks) if peer != rank]
    # alltoall_single splits by rank order, so peer chunks go in ascending
    # rank order.
    if send is None:
        send = (
            partials[peers[0]]
            if len(peers) == 1
            else paddle.concat([partials[peer] for peer in peers], axis=0)
        )
    in_split = [0] * nranks
    out_split = [0] * nranks
    for peer in peers:
        in_split[peer] = T
        out_split[peer] = T
    recv, task = _inter_a2a_single(
        [T * len(peers), *partials[0].shape[1:]],
        send,
        out_split,
        in_split,
        group,
        zcomm=zcomm,
    )
    handle.update(
        {
            "task": task,
            "recv": recv,
            "peers": peers,
            "T": T,
            "r0": rank,
            "n": nranks,
        }
    )


def _routing_pair(idx, w):
    """Pack routing indices and router weights into one float32 tensor.

    ``[T, K]`` indices and ``[T, K]`` weights become ``[T, 2K]`` float32, so
    each hop and gather moves a single tensor. The packing is lossless: bf16
    and fp32 weights round-trip through float32 exactly, and expert ids are
    far below 2^24. Built from ``concat``/``cast``, so the grad of ``w`` flows
    through normally.
    """
    return paddle.concat([idx.astype("float32"), w.astype("float32")], axis=1)


def _routing_unpair(pair, k, w_dtype):
    """Split a ``[rows, >= 2K]`` float32 routing pair back into ``(idx, w)``."""
    idx = pair[:, :k].astype("int32")
    # columns past 2K, if any, carry the grouped down projection's plan parts
    w = pair[:, k : 2 * k]
    if w.dtype != w_dtype:
        w = w.astype(w_dtype)
    return idx, w


def _prefetch_intra_ag(pair, group, zcomm=False):
    """Pre-issue an async intra AllGather of a non-fp8 token on the comm stream.

    The gather overlaps compute; :func:`_pre_ag_pair_value` consumes it
    value-only. The caller orders it after any producing hop with
    :func:`_order_after`. Returns a handle, or None for a single-rank group
    (unreachable here: the ring requires ``G > 1``).
    """
    if group is None or group.nranks == 1:
        return None
    out, task = _token_all_gather_async(pair, group, zcomm=zcomm)
    return {
        "out": out,
        "task": task,
        "shape": list(pair.shape),
        "group": group,
    }


def _prefetch_routing_ag(pair, group):
    """Pre-issue the routing-pair intra AllGather on the comm stream.

    Issued early so the gather overlaps compute instead of blocking the calc
    stream; when it follows a routing hop the caller orders it after that hop
    with :func:`_order_after`. Consumed by :class:`_RouterPreAllGather`.
    Returns a handle, or None for a single-rank group (the caller then uses
    :class:`_RouterAllGather` inline).
    """
    if group is None or group.nranks == 1:
        return None
    out, task = _all_gather_async(pair, group)
    return {
        "output": out,
        "task": task,
        "group": group,
        "local_shape": list(pair.shape),
    }


def _pre_ag_fp8_value(handle):
    """Wait for a prefetched fp8 token AllGather and return its result.

    Value-only (no autograd node): the caller uses the gathered token as a
    manual-backward leaf and drives the reverse ReduceScatter itself. Clears
    the handle's buffers and tasks. Returns ``(data, scale, group)``: e4m3
    data, the ``[T_global, nsb]`` scale in ``sdt``, and the intra group to
    reduce-scatter the bf16 grad over.
    """
    handle["data_task"].wait()
    handle["scale_task"].wait()
    gather_group = handle["group"]
    T_global = handle["data_buf"].shape[0]
    data = handle["data_buf"].view("float8_e4m3fn")
    scale = (
        handle["scale_buf"]
        .view(handle["sdt"])
        .reshape([T_global, handle["nsb"]])
    )
    handle["data_buf"] = None
    handle["scale_buf"] = None
    handle["data_task"] = None
    handle["scale_task"] = None
    return data, scale, gather_group


def _pre_ag_pair_value(handle):
    """Wait for a :func:`_prefetch_intra_ag` gather; returns ``(out, group)``.

    Non-fp8 counterpart of :func:`_pre_ag_fp8_value`: value-only, and clears
    the handle.
    """
    handle["task"].wait()
    gather_group = handle["group"]
    out = handle["out"]
    handle["out"] = None
    handle["task"] = None
    return out, gather_group


def _prefetch_tok_ag_fp8(tok_fp8, group):
    """Pre-issue the intra AllGather of an already-quantized fp8 token.

    ``tok_fp8`` is a dict from :func:`_fp8_pair`; its data and scale go out
    as two async AllGathers on the comm stream. Consumed by
    :func:`_pre_ag_fp8_value`. Returns a handle, or None for a single-rank
    group.
    """
    if group is None or group.nranks == 1:
        return None
    data_local = tok_fp8["data"]
    scale_local = tok_fp8["scale"]
    rows = data_local.shape[0]
    data_buf = paddle.empty(
        [rows * group.nranks, *data_local.shape[1:]], dtype="uint8"
    )
    scale_buf = paddle.empty(
        [rows * group.nranks, *scale_local.shape[1:]], dtype="uint8"
    )
    data_task = paddle.distributed.stream.all_gather(
        data_buf,
        data_local,
        group=group,
        sync_op=False,
        use_calc_stream=False,
    )
    scale_task = paddle.distributed.stream.all_gather(
        scale_buf,
        scale_local,
        group=group,
        sync_op=False,
        use_calc_stream=False,
    )
    return {
        "data_buf": data_buf,
        "scale_buf": scale_buf,
        "data_task": data_task,
        "scale_task": scale_task,
        "nsb": tok_fp8["nsb"],
        "sdt": tok_fp8["sdt"],
        "group": group,
    }


# Memoized answer of :func:`_sync_free_metadata_ignores_counts`.
_SYNC_FREE_METADATA_IGNORES_COUNTS: bool | None = None


def _sync_free_metadata_ignores_counts() -> bool:
    """Whether ``sync_free_sizing`` metadata never reads ``tokens_per_expert``.

    ``run_sonic_moe`` forwards the counts only to the metadata builder, which
    under ``sync_free_sizing`` sizes its outputs from the shape bound. Its
    fused CUDA path then never dereferences them, so the caller may pass
    ``tokens_per_expert=None``; its eager fallback still host-prefix-sums
    them, so that path needs them even when the sizing is sync-free.
    ``deepep_metadata._HAS_TOPK_CUDA_KERNEL`` is what selects between the two,
    which is why it is the flag read here.

    Returns ``False`` whenever that cannot be established, which keeps the
    counts being computed and passed as before.
    """
    global _SYNC_FREE_METADATA_IGNORES_COUNTS
    if _SYNC_FREE_METADATA_IGNORES_COUNTS is None:
        try:
            from paddlefleet_ops.sonicmoe.ernie_compat import deepep_metadata

            has_cuda_kernel = bool(
                getattr(deepep_metadata, "_HAS_TOPK_CUDA_KERNEL", False)
            )
        except Exception:
            has_cuda_kernel = False
        _SYNC_FREE_METADATA_IGNORES_COUNTS = has_cuda_kernel
    return _SYNC_FREE_METADATA_IGNORES_COUNTS


class RingMoETokenDispatcher(AllGatherTokenDispatcher):
    """Two-level ring dispatcher for intermediate-sharded experts.

    Computes the same result as :class:`AllGatherTokenDispatcher`: every rank
    holds an intermediate-dim shard of every expert, and a token's output is
    the sum of all shards' contributions. Instead of one AllGather and
    ReduceScatter over the whole EP group, the ``N`` nodes take turns: each
    round, every node runs intra-node AllGather -> expert GEMM -> intra-node
    ReduceScatter over its ``G`` GPUs on one node's tokens, while the next
    round's tokens and routing arrive by single-peer inter-node hops. An
    inter-node all-to-all then returns each node the partial sums for its own
    tokens. Bulk traffic thus stays intra-node, and each hop overlaps a GEMM.

    The first call runs rounds ``0..N-1`` in order. Later calls run the
    remote rounds ``1..N-1`` first and the home round ``0`` last, so round
    1's hop and gather overlap the router and the combine all-to-all
    overlaps the home round's GEMM.

    ``MoELayer`` calls :meth:`ring_forward` instead of the
    dispatch/compute/combine pipeline; the inherited methods keep the
    dispatcher interchangeable with the flat one everywhere else.
    """

    def __init__(
        self,
        moe_group: Group,
        expert_model_parallel_size: int,
        num_experts: int,
        fp8_dispatch: bool = False,
        use_ue8m0: bool = False,
        use_zcomm: bool = False,
    ):
        super().__init__(
            moe_group,
            expert_model_parallel_size,
            num_experts,
            fp8_dispatch=fp8_dispatch,
            use_ue8m0=use_ue8m0,
        )
        # ``use_ue8m0`` needs no handling: as in the flat dispatcher, token
        # quantization always uses an int32 block scale.
        self.fp8_dispatch = fp8_dispatch
        # ``TransformerConfig.ringmoe_use_zcomm``: run the ring's collectives on
        # the paddlefleet_ops zcomm ops. The fp8 ring keeps NCCL (the zcomm
        # ReduceScatter sums bf16 only), and so does a process whose
        # paddlefleet_ops / NCCL cannot run them, after one warning.
        self.use_zcomm = use_zcomm and not fp8_dispatch
        if self.use_zcomm:
            from .zcomm_comm import zcomm_unavailable_reason

            reason = zcomm_unavailable_reason()
            if reason is not None:
                _warn_zcomm_fallback(reason)
                self.use_zcomm = False
        # Set by the first ring_forward, which checks that token counts match
        # across the inter group; gates the remote-first pre-gate.
        self._equal_tokens_checked = False
        # Home round's token AllGather, issued by pre_gate_token_ag.
        self._gate_pf_handle = None
        # Token pack from pre_gate_token_ag, consumed by ring_forward.
        self._gate_pack = None
        # Slot from join_token_hop, consumed by ring_forward.
        self._tok_hop_slot = None
        # Grouped BF16 down projection (enable_grouped_down_proj).
        self._gdp = None
        # A real EP group is required; the multi-node (N > 1) and intra-node
        # (G > 1) checks follow once the sub-groups are built. All three are
        # raised rather than asserted so they survive ``python -O``.
        if moe_group is None or moe_group.nranks <= 1:
            raise RuntimeError(
                "RingMoETokenDispatcher needs a real EP group (nranks>1); got "
                f"nranks={None if moe_group is None else moe_group.nranks}. "
                "Use moe_token_dispatcher_type='allgather' for single-GPU / "
                "no-EP."
            )
        # The routing mask/cast and per-round histogram are Triton kernels.
        # They and this check are imported lazily: importing any triton_ops
        # submodule runs the package __init__, which imports triton, and
        # other dispatchers must stay usable without it.
        from paddlefleet.triton_ops.utils import is_triton_available

        if not is_triton_available():
            raise RuntimeError(
                "RingMoETokenDispatcher requires Triton (a CUDA build of "
                "Paddle and the triton package)."
            )
        (
            self.G,
            self.N,
            self.intra_group,
            self.inter_group,
            self.intra_rs_group,
            self.intra_ag_group,
            self.intra_rt_group,
            self.inter_cb_group,
        ) = _build_ring_subgroups(moe_group, _RING_GPUS_PER_NODE)
        if self.N <= 1:
            raise RuntimeError(
                "RingMoETokenDispatcher requires EP to span >1 node (got "
                f"N={self.N}, G={self.G}). With N==1 the two-level ring is "
                "just a single intra-node AllGather -- set "
                "moe_token_dispatcher_type='allgather' for single-node EP "
                f"(expert_model_parallel_size <= {_RING_GPUS_PER_NODE})."
            )
        # The ring also needs an intra-node level.
        if self.G <= 1 or self.intra_group is None:
            raise RuntimeError(
                "RingMoETokenDispatcher requires a two-level ring with a real "
                f"intra-node level (G>1); got G={self.G}, N={self.N} -- an "
                "inter-only degenerate topology (one GPU per node group). Use "
                "moe_token_dispatcher_type='allgather' for such an EP layout."
            )

    def _drain_stale_pregate(self):
        """Wait out and drop pre-gate collectives left by an aborted forward.

        If ``ring_forward`` never consumed them, their tasks still own their
        output buffers; waiting before the slots are reused keeps the next
        forward from racing them. Wait failures are only logged, so they do
        not mask the error that aborted the previous forward.
        """

        def _tasks_of(handle):
            # fp8 gathers use data_task/scale_task, others task or tasks.
            if not handle:
                return ()
            out = [
                handle.get("data_task"),
                handle.get("scale_task"),
                handle.get("task"),
            ]
            out.extend(handle.get("tasks") or ())
            return out

        pending = list(_tasks_of(self._gate_pf_handle))
        stale_remote = (self._gate_pack or {}).get("remote")
        if stale_remote:
            pending += list(_tasks_of(stale_remote["h"])) + list(
                _tasks_of(stale_remote["pf"])
            )
            for more in (stale_remote.get("pf_more") or {}).values():
                pending += list(_tasks_of(more))
        self._gate_pf_handle = None
        self._gate_pack = None
        for task in pending:
            if task is None:
                continue
            try:
                task.wait()
            except (RuntimeError, OSError) as exc:
                logger.warning(
                    "ringmoe pre-gate: leftover async task wait failed (%s), "
                    "discarding handle.",
                    exc,
                )

    def join_token_hop(self, tokens, gate_input):
        """Route the ring's token carrier and the router input through
        :class:`_RingTokHopJoin`.

        Optional. MoELayer calls this before the router, then passes the
        returned token alias to :meth:`pre_gate_token_ag` and the returned
        router-input alias to the router. The next ring backward then leaves
        its last token hop in flight so the router backward overlaps it; the
        join's backward waits for it. Returns both inputs unchanged when
        either takes no grad: without a router backward there is nothing to
        overlap.
        """
        self._tok_hop_slot = None
        if (
            tokens.stop_gradient
            or gate_input.stop_gradient
            or not framework._dygraph_tracer()._has_grad
        ):
            return tokens, gate_input
        slot = {}
        self._tok_hop_slot = slot
        return _RingTokHopJoin.apply(tokens, gate_input, slot)

    def pre_gate_token_ag(self, tokens):
        """Pack the local token and start the home round's gather.

        MoELayer calls this on every forward, before the router, with the
        same tensor it later passes to :meth:`ring_forward`, so the packed
        token is the autograd carrier for the whole ring. On fp8 the token is
        quantized here once and reused by every hop and gather. The intra
        AllGather issued here overlaps the router; from the second call on
        (remote-first), round 1's token hop and gather are issued as well
        (see :meth:`_pre_gate_remote`). Routing comm needs the router output
        and is issued in :meth:`ring_forward`.
        """
        self._drain_stale_pregate()
        t = tokens.reshape([-1, tokens.shape[-1]]).contiguous()
        meta = {}
        if self.fp8_dispatch:
            data, scale = _FP8TokenQuant.apply(t, meta)
            self._gate_pf_handle = _prefetch_tok_ag_fp8(
                _fp8_pair(data, scale, meta), self.intra_group
            )
        else:
            # bf16: the token itself is the carrier.
            data, scale = t, None
            self._gate_pf_handle = _prefetch_intra_ag(
                t, self.intra_group, zcomm=self.use_zcomm
            )
        self._gate_pack = {
            "meta": meta,
            "data": data,
            "scale": scale,
            "remote": self._pre_gate_remote(data, scale, meta),
        }

    def _pre_gate_remote(self, data, scale, meta):
        """Issue the remote rounds' token transfers and intra gathers.

        The forward runs the remote rounds first and the home round last, so
        the last peer slot's ReduceScatter and the combine all-to-all can
        overlap the home round's GEMM. With zcomm one inter AllGather
        fetches every node's token here and every remote round's intra
        gather is issued right behind it, all overlapping the router: no
        round's gather then waits for the previous round's hop (with the
        copy engine's and GIN's handshakes such a hop -> gather chain
        outgrows the GEMM it should hide behind once there are several
        hops). Otherwise round 1's token (on fp8, its data and scale) is
        hopped and gathered here and later rounds hop in the loop.

        Returns ``{"data", "scale", "h", "pf"}`` (round 1's token, the
        transfer handle, round 1's gather handle) plus, with zcomm, ``"blk"``
        (round k's token) and ``"pf_more"`` (later rounds' gather handles),
        or None before :meth:`_check_equal_tokens` has run (a transfer with
        mismatched sizes hangs instead of raising) or without an intra
        routing group.
        """
        if not (self._equal_tokens_checked and self.intra_rt_group is not None):
            return None
        inter = self.inter_group
        tok_group = self.intra_ag_group or self.intra_group
        inter_rank = inter.rank
        dst, src = (inter_rank + 1) % self.N, (inter_rank - 1) % self.N
        handle = {}
        if self.use_zcomm:
            # Block d of the gather is node d's token; round k processes
            # node (rank - k)'s. Round 1's block is gathered first, then the
            # other remote rounds' together, each followed on the comm side
            # by those rounds' intra gathers: round 1 runs right after the
            # router, so its token should not share the NIC with the later
            # rounds' tokens until all of them land.
            from .zcomm_comm import InterGin

            rows = data.shape[0]
            blocks, pfs = {}, {}

            def _on_part(ks, all_tok, task):
                handle["task"] = task
                if not _order_after(tok_group, inter):
                    self._drain([handle])
                for k in ks:
                    blocks[k] = all_tok[
                        (inter_rank - k) % self.N * rows : (
                            (inter_rank - k) % self.N + 1
                        )
                        * rows
                    ]
                    pfs[k] = _prefetch_intra_ag(
                        blocks[k], tok_group, zcomm=self.use_zcomm
                    )

            InterGin.get(inter).all_gather_by_distance_async(data, _on_part)
            # On zcomm the gather output is a window slot, reissued two
            # gathers of its size later; only its blocks are kept. Every
            # block's intra gather is waited on before its round's GEMM, and
            # any later transfer is issued after that, so ordered after it.
            pf = pfs.pop(1)
            return {
                "data": blocks.pop(1),
                "scale": None,
                "h": handle,
                "pf": pf,
                "blk": blocks,
                "pf_more": pfs,
            }
        if self.fp8_dispatch:
            remote_data, remote_scale = _raw_fp8_shift_async(
                data, scale, inter, dst, src, handle
            )
        else:
            remote_data = _raw_inter_shift_async(
                data, inter, dst, src, handle, zcomm=False
            )
            remote_scale = None
        # Order the gather after the hop on the comm side; if that is not
        # possible, wait for the hop instead.
        if not _order_after(tok_group, inter):
            self._drain([handle])
        return {
            "data": remote_data,
            "scale": remote_scale,
            "h": handle,
            "pf": (
                _prefetch_tok_ag_fp8(
                    _fp8_pair(remote_data, remote_scale, meta), tok_group
                )
                if self.fp8_dispatch
                else _prefetch_intra_ag(remote_data, tok_group, zcomm=False)
            ),
        }

    @staticmethod
    def _drain(handles):
        """Wait for every task of the given collective handles, in order.

        A handle holds a single ``task``, a ``tasks`` list (e.g. the separate
        data and scale transfers of :func:`_raw_fp8_shift_async`), or neither
        when its group issued nothing.
        """
        for handle in handles:
            task = handle.get("task")
            if task is not None:
                task.wait()
            for task in handle.get("tasks") or ():
                if task is not None:
                    task.wait()

    def _check_equal_tokens(self, local_tokens: int):
        """Raise unless every inter-group rank holds the same token count.

        Inter-node hops size their sends and receives from the local token
        count, so unequal counts (unbalanced DP, unpacked variable-length
        sequences) mismatch the transfer sizes and hang or corrupt data
        instead of raising. Costs a gather, so ``ring_forward`` runs it once.
        """
        counts = paddle.full([1], local_tokens, dtype="int64")
        gathered = paddle.empty([self.inter_group.nranks], dtype="int64")
        paddle.distributed.stream.all_gather(
            gathered, counts, group=self.inter_group, sync_op=True
        )
        counts_list = gathered.tolist()
        if len(set(counts_list)) != 1:
            raise ValueError(
                "RingMoE requires an equal local token count on every rank of "
                f"the inter-node group, got {counts_list} (this rank: "
                f"{local_tokens}). Pack sequences to a fixed length or keep DP "
                "balanced; otherwise the inter-node shift mismatches NCCL "
                "buffer sizes."
            )

    def global_tokens_per_expert(self, topk_indices):
        """EP-wide tokens-per-expert histogram, for load-balance logging.

        The ring never builds the global index list, so this sums the
        per-rank histograms over the EP group, which gives the same counts.
        """
        from paddlefleet.triton_ops.tokens_per_expert_hist import (
            tokens_per_expert_histogram_triton,
        )

        counts = tokens_per_expert_histogram_triton(
            topk_indices.detach().cast("int32"), self.num_experts
        )
        if self.moe_group is not None and self.moe_group.nranks > 1:
            paddle.distributed.all_reduce(counts, group=self.moe_group)
        return counts

    def enable_grouped_down_proj(self, config):
        """Run the BF16 down projection of every round as sonic's grouped
        down projection (`paddlefleet.transformer.moe.grouped_down_proj`;
        `config` is its `GroupedDownProjConfig`); fp8 dispatch keeps the
        default path."""
        from .grouped_down_proj.ring import RingGroupedDownProj

        self._gdp = RingGroupedDownProj(config)

    def ring_forward(
        self,
        x_l,
        topk_weights,
        topk_indices,
        expert_fn,
        probs_dtype,
        recompute_moe_gate_up=False,
        combine_overlap_handle=None,
    ):
        """Run the two-level ring and return this rank's combined output.

        Each round handles one node's tokens: intra AllGather -> expert GEMM
        -> intra ReduceScatter yields this node's partial for that node. The
        next round's inputs are sent by async inter-node hops issued before
        the current round's GEMM, so they overlap it. An inter-node
        all-to-all then returns every node its tokens' partials, which are
        summed locally.

        The first call runs the home round first and takes every round's
        routing from one inter AllGather. Later calls run remote-first
        (rounds 1..N-1, then home), hop the routing pair and the token (with
        zcomm, every round's token comes from one inter AllGather issued
        before the router instead), and issue the combine before the home
        GEMM so it overlaps it.

        All rounds and the combine run inside one :class:`_RingRoundsExplicit`
        op whose backward is scheduled by hand. Requires an equal local token
        count across the inter group (see :meth:`_check_equal_tokens`).

        ``combine_overlap_handle`` optionally carries a shared-expert function
        that is run while the combine is in flight. Its residual becomes an
        input and its output an output of the ring op; the output is written
        back to the handle as ``fn_out``.
        """
        from paddle import framework as _framework

        x_l = x_l.reshape([-1, x_l.shape[-1]]).contiguous()
        if self.fp8_dispatch and x_l.shape[-1] % 128 != 0:
            # The fp8 kernel consumes whole block-scale tiles, so the width
            # must be a tile multiple, not just a scale-block multiple.
            raise ValueError(
                f"RingMoE + fp8 requires the MoE hidden width "
                f"({x_l.shape[-1]}, i.e. moe_latent_size when latent MoE is on, "
                f"else hidden_size) to be a multiple of 128 (fp8 block-scale "
                f"tile)."
            )
        tok = x_l
        # One fused kernel casts the indices to int32 and the weights to
        # probs_dtype, and zeroes the weights (and their grads) of dropped
        # lanes (idx < 0). It selects a literal 0 rather than multiplying by a
        # mask, since Inf * 0 would leak NaN through dropped lanes. It stays
        # outside the ring op so the router grad flows through it.
        from paddlefleet.triton_ops.mask_cast_routing_w import (
            mask_cast_routing_w,
        )
        from paddlefleet.triton_ops.tokens_per_expert_hist import (
            tokens_per_expert_histogram_triton,
        )

        # The ring always asks the expert op for `sync_free_sizing`, under
        # which the fused metadata kernel sizes its outputs from the shape
        # bound and never reads the per-expert counts. Where that is the case
        # the per-round histogram is dead work on the calc stream, so skip it
        # and let the expert op take `tokens_per_expert=None`.
        need_hist = not _sync_free_metadata_ignores_counts()

        local_idx, local_w = mask_cast_routing_w(
            topk_indices.detach(), topk_weights, probs_dtype
        )

        if not self._equal_tokens_checked:
            # Set only once the check passes: it gates remote-first, whose
            # hops hang on mismatched sizes.
            self._check_equal_tokens(tok.shape[0])
            self._equal_tokens_checked = True

        intra = self.intra_group
        fp8_ring = self.fp8_dispatch
        # Take over the pre-gate pack and clear the slot, so no later call
        # reuses it; `_drain_stale_pregate` only sees unconsumed packs. On
        # fp8, cur_data (a uint8 tensor) carries the token grad and cur_scale
        # has none; on bf16 the packed token is the carrier.
        gate_pack = self._gate_pack
        self._gate_pack = None
        cur_data = cur_scale = None
        local_tok = tok
        # Raised rather than asserted so the check survives ``python -O``.
        if fp8_ring:
            if gate_pack is None:
                raise RuntimeError(
                    "RingMoE fp8 dispatch requires the pre-gate prefetch: "
                    "MoELayer must call pre_gate_token_ag() before ring_forward()."
                )
            cur_data, cur_scale = gate_pack["data"], gate_pack["scale"]
        else:
            if gate_pack is None:
                raise RuntimeError(
                    "RingMoE dispatch requires the pre-gate prefetch: MoELayer "
                    "must call pre_gate_token_ag() before ring_forward()."
                )
            # Use the tensor the pre-gate gather reads as the carrier.
            local_tok = gate_pack["data"]
        home_tok_ag_handle = self._gate_pf_handle
        self._gate_pf_handle = None
        # Set by `join_token_hop`: the backward then hands its last token hop
        # to the join instead of waiting for it.
        tok_hop_slot = self._tok_hop_slot
        self._tok_hop_slot = None
        # Remote round's pre-issued hop and gather, or None.
        remote = gate_pack.get("remote")

        # Differentiable inputs of the ring op: token carrier, router weights
        # and, if present, the shared-expert residual.
        carrier = cur_data if fp8_ring else local_tok
        has_se = combine_overlap_handle is not None
        se_fn = combine_overlap_handle["fn"] if has_se else None
        se_res = combine_overlap_handle["fn_args"][0] if has_se else None

        leaves = [carrier, local_w]
        if has_se:
            leaves.append(se_res)

        # Sampled outside PyLayer.forward, where the tracer has grad disabled
        # and every forward would look like a first forward.
        is_first_fwd = not _framework._dygraph_tracer()._has_grad

        # Runs the forward schedule on the ring op's inputs and returns
        # ``(out, _bwd)``; see `_RingRoundsExplicit`.
        def _plan(*ring_leaves):
            # Hop outputs are not leaves: every hop is value-only and is
            # reversed by hand in `_bwd`.
            home_carrier, home_w = ring_leaves[0], ring_leaves[1]
            res = ring_leaves[2] if has_se else None

            num_nodes = self.N
            inter_rank = self.inter_group.rank
            intra, inter = self.intra_group, self.inter_group
            # Token and routing gathers use separate communicators so they
            # run concurrently (see `_build_ring_subgroups`).
            tok_group = self.intra_ag_group or intra
            rt_group = self.intra_rt_group
            dst, src = (
                (inter_rank + 1) % num_nodes,
                (inter_rank - 1) % num_nodes,
            )
            meta = gate_pack["meta"]
            k_topk = local_idx.shape[1]
            w_dtype = home_w.dtype
            rs_group = self.intra_rs_group
            zc = self.use_zcomm

            # Grouped down projection: every round's routing goes to the
            # planner as soon as its gather is enqueued; the round's down
            # projection waits for the plan on the GPU (see
            # `grouped_down_proj.ring`).
            gdp = self._gdp if not fp8_ring else None
            if gdp is not None:
                rt_rows = rt_group.nranks if rt_group is not None else 1
                if not gdp.start_forward(
                    local_idx.shape[0] * rt_rows,
                    k_topk,
                    self.num_experts,
                    intra,
                ):
                    gdp = None

            def _gdp_submit(rnd, handle, group):
                # Round `rnd`'s routing, produced by `handle`'s gather on
                # `group`'s comm stream.
                if gdp is not None and handle is not None:
                    gdp.submit(
                        rnd,
                        handle["output"],
                        group.process_group.get_stream(
                            _framework._current_expected_place()
                        ),
                    )

            # partials[slot] is this node's output for the tokens of
            # inter rank ``slot``.
            partials = [None] * num_nodes
            rs_tasks = []
            # Peer slots are reduce-scattered straight into their slice of
            # one combine send buffer (ascending peer order, allocated on
            # the first peer ReduceScatter), so the combine needs no concat.
            peer_ranks = [d for d in range(num_nodes) if d != inter_rank]
            peer_index = {d: j for j, d in enumerate(peer_ranks)}
            combine_send = None
            # Remote-first: rounds 1..N-1 run before round 0, round 1
            # using the hop and gather issued by `_pre_gate_remote`.
            remote_first = remote is not None
            # Otherwise every round's routing comes from one inter
            # AllGather of round 0's gathered pair (round k reads block
            # (rank - k) % N), so later rounds issue no routing comm. That
            # gather runs on the combine communicator so it never queues
            # ahead of a token hop. Remote-first rounds can't wait for it,
            # so they hop the pair on `inter` with the token instead.
            pair_pregather = not remote_first
            pair_inter_group = self.inter_cb_group
            # zcomm: the pre-gate token gather keeps `inter` busy into the
            # first rounds, so routing hops go on the combine communicator,
            # beside it, instead of queueing behind it.
            pair_hop_group = (
                pair_inter_group
                if zc and pair_inter_group is not None
                else inter
            )
            # Every node's gathered routing pair, [N*G*T, 2K].
            all_pairs = None
            steps = (
                [*range(1, num_nodes), 0]
                if remote_first
                else list(range(num_nodes))
            )
            # Combine all-to-all handle; filled early under remote-first.
            combine_handle = {}

            def _rs_issue(y, slot):
                # Async, value-only intra ReduceScatter of a round's output
                # into partials[slot]; `_bwd` drives its reverse.
                nonlocal combine_send
                if rs_group is None or rs_group.nranks == 1:
                    partials[slot] = y
                    return
                # The forward output RS stays on NCCL: on the copy engine its
                # local-sum kernel can only start once the next round's
                # expert GEMM (a persistent kernel holding every SM) has
                # finished, which delays the last peer slot's partial and so
                # the combine behind it. NCCL's kernel is resident before the
                # GEMM starts.
                if slot == inter_rank:
                    # Home slot: stays local, own buffer.
                    out, task = _intra_rs_async(
                        y, rs_group, zcomm=False, overlapped=False
                    )
                    partials[slot] = out
                    rs_tasks.append(task)
                    return
                # Peer slot: scatter into its slice of the combine buffer.
                scatter_rows = y.shape[0] // rs_group.nranks
                if combine_send is None:
                    combine_send = paddle.empty(
                        [len(peer_ranks) * scatter_rows, *y.shape[1:]],
                        dtype=y.dtype,
                    )
                peer_idx = peer_index[slot]
                out, task = _intra_rs_async(
                    y,
                    rs_group,
                    zcomm=False,
                    out=combine_send[
                        peer_idx * scatter_rows : (peer_idx + 1) * scatter_rows
                    ],
                )
                partials[slot] = out
                rs_tasks.append(task)

            # Per-round state for `_bwd`: segment backward fns and the
            # intra group for each round's token grad.
            seg_bwd_fns = [None] * num_nodes
            tok_rs_groups = [None] * num_nodes
            # The home routing pair carries the router-weight grad; other
            # rounds' pairs are value-only (hops of it, or slices of the
            # inter gather). Pair gathers run on the routing comm stream
            # (`_prefetch_routing_ag`), so they overlap compute.
            home_pair = _routing_pair(local_idx, home_w)
            if gdp is not None:
                # This rank's part of the grouped plan rides with its routing
                # to every round (GPU planner; unchanged otherwise).
                home_pair = gdp.extend_pair(local_idx, home_pair)
            home_pair.stop_gradient = False
            pair_ag_handle = _prefetch_routing_ag(home_pair, rt_group)
            # Inputs of the first round: the home round unless remote-first.
            cur_data, cur_scale = home_carrier, gate_pack["scale"]
            cur_pair = home_pair
            tok_ag_handle = home_tok_ag_handle
            home_pair_ag_handle = pair_ag_handle
            _gdp_submit(0, home_pair_ag_handle, rt_group)
            remote_pair_hop_handle = {}
            pregather_blk, pregather_pf = None, {}
            if remote_first:
                # Start with round 1. Its token was hopped and
                # gathered under the router; hop its routing pair now and
                # gather it after the hop (comm-side order if possible).
                cur_pair = _raw_inter_shift_async(
                    home_pair,
                    pair_hop_group,
                    dst,
                    src,
                    remote_pair_hop_handle,
                    zcomm=zc,
                )
                if not _order_after(rt_group, pair_hop_group):
                    self._drain([remote_pair_hop_handle])
                pair_ag_handle = _prefetch_routing_ag(cur_pair, rt_group)
                _gdp_submit(1, pair_ag_handle, rt_group)
                cur_pair.stop_gradient = False
                # Pop rather than read: `remote` lives until the end of the
                # forward, and with zcomm its hop output is a window view
                # that must be released once the rounds move past it.
                cur_data, cur_scale = remote.pop("data"), remote.pop("scale")
                cur_data.stop_gradient = home_carrier.stop_gradient
                tok_ag_handle = remote["pf"]
                # zcomm: every remote round's token is a block of the
                # pre-gate inter AllGather, already gathered intra-node.
                pregather_blk = remote.pop("blk", None)
                pregather_pf = remote.pop("pf_more", None) or {}
            # A round's output ReduceScatter is deferred until just before
            # the next round's GEMM, so it overlaps that GEMM; the last
            # one is issued after the loop.
            pending_rs = None

            def _early_combine_issue():
                # Remote-first: issue the combine all-to-all right after the
                # last peer slot's ReduceScatter, ordered after it on the
                # comm side (or after a wait). Only peer chunks are sent, so
                # the not-yet-produced home slot just needs a placeholder for
                # the shape and dtype.
                if rs_group is not None and not _order_after(inter, rs_group):
                    for pending_task in rs_tasks:
                        pending_task.wait()
                traveling_partial = next(p for p in partials if p is not None)
                _combine_a2a_issue(
                    [
                        p if p is not None else traveling_partial
                        for p in partials
                    ],
                    inter,
                    combine_handle,
                    send=combine_send,
                    zcomm=zc,
                )

            for pos, step in enumerate(steps):
                is_last = pos == len(steps) - 1
                # Remote-first: the last remote round is followed by the
                # home round, whose inputs are local and need no hop.
                to_home = not is_last and steps[pos + 1] == 0
                tok_hop_handle, pair_comm_handle = {}, {}
                next_data = next_scale = next_pair = None
                next_tok_ag_handle = None
                next_pair_ag_handle = None

                # Wait for the token gather outside the segment: the
                # gathered token is the segment's leaf, and `_bwd`
                # reduce-scatters its grad by hand.
                if fp8_ring:
                    gathered_tok, gathered_scale, tok_rs_groups[step] = (
                        _pre_ag_fp8_value(tok_ag_handle)
                    )
                else:
                    gathered_tok, tok_rs_groups[step] = _pre_ag_pair_value(
                        tok_ag_handle
                    )
                    gathered_scale = None
                # Match the carrier: a grad for a stop-gradient leaf would
                # be rejected by the PyLayer.
                gathered_tok.stop_gradient = cur_data.stop_gradient

                def _seg(
                    seg_tok,
                    seg_pair,
                    seg_step=step,
                    seg_scale=gathered_scale,
                    seg_pair_ag_handle=pair_ag_handle,
                    seg_pregathered=pair_pregather and step > 0,
                    seg_next=None if is_last else steps[pos + 1],
                ):
                    # One round as a manual-backward segment: routing gather,
                    # histogram, next round's transfers, previous round's
                    # ReduceScatter, expert GEMM. The comm is value-only
                    # (`_bwd` reverses it) and sets `_plan` state read after
                    # the segment.
                    nonlocal \
                        next_data, \
                        next_scale, \
                        next_pair, \
                        next_tok_ag_handle, \
                        pending_rs
                    nonlocal next_pair_ag_handle, all_pairs
                    # A pre-gathered pair needs no gather; otherwise consume
                    # the comm-stream prefetch, or gather on the calc stream.
                    if seg_pregathered:
                        gathered_pair = seg_pair
                    elif seg_pair_ag_handle is not None:
                        gathered_pair = _RouterPreAllGather.apply(
                            seg_pair, seg_pair_ag_handle
                        )
                    else:
                        gathered_pair = _RouterAllGather.apply(
                            seg_pair, rt_group
                        )
                    if gdp is not None:
                        # No-op if the round's gather was submitted ahead;
                        # otherwise its routing was gathered on this stream.
                        gdp.submit(seg_step, gathered_pair)
                    g_idx, g_w = _routing_unpair(gathered_pair, k_topk, w_dtype)
                    hist = (
                        tokens_per_expert_histogram_triton(
                            g_idx, self.num_experts
                        )
                        if need_hist
                        else None
                    )

                    def _do_gemm():
                        # Issue the previous round's deferred ReduceScatter
                        # right before this GEMM so it overlaps the GEMM.
                        nonlocal pending_rs
                        if pending_rs is not None:
                            _rs_issue(*pending_rs)
                            pending_rs = None
                            if remote_first and seg_step == 0:
                                # Every peer ReduceScatter is issued: start
                                # the combine so it overlaps the home GEMM.
                                _early_combine_issue()
                        with (
                            gdp.round(seg_step, g_idx)
                            if gdp is not None
                            else contextlib.nullcontext()
                        ):
                            return expert_fn(
                                seg_tok,
                                g_idx,
                                g_w,
                                self.fp8_dispatch,
                                tokens_per_expert=hist,
                                fp8_scale=seg_scale,
                                recompute_moe_gate_up=recompute_moe_gate_up,
                                fp8_combine_grad_handle=None,
                                sync_free_sizing=True,
                            )

                    # Issue the next round's transfers before the GEMM is
                    # enqueued, so they overlap it. The home round after the
                    # last remote round needs none.
                    if not is_last and not to_home:
                        if pair_pregather:
                            # The first round gathers every node's
                            # gathered pair over inter; later rounds
                            # slice it and issue no routing comm.
                            if seg_step == 0:
                                all_pairs, pair_comm_handle["task"] = (
                                    _all_gather_async(
                                        gathered_pair, pair_inter_group
                                    )
                                )
                                if gdp is not None:
                                    # Rounds 1..N-1: their blocks of the
                                    # gather, sliced after it on the copy
                                    # stream.
                                    rows = all_pairs.shape[0] // num_nodes
                                    ag_stream = pair_inter_group.process_group.get_stream(
                                        _framework._current_expected_place()
                                    )
                                    for k in range(1, num_nodes):
                                        gdp.submit(
                                            k,
                                            all_pairs,
                                            ag_stream,
                                            row0=((inter_rank - k) % num_nodes)
                                            * rows,
                                        )
                        else:
                            # Routing hop first, then its gather, ordered
                            # after this hop only on the comm side. If
                            # that fails, the next round gathers inline.
                            next_pair = _raw_inter_shift_async(
                                cur_pair,
                                pair_hop_group,
                                dst,
                                src,
                                pair_comm_handle,
                                zcomm=zc,
                            )
                            if _order_after(rt_group, pair_hop_group):
                                next_pair_ag_handle = _prefetch_routing_ag(
                                    next_pair, rt_group
                                )
                                _gdp_submit(
                                    seg_next, next_pair_ag_handle, rt_group
                                )
                        # Token hop, then its gather on `tok_group`,
                        # ordered after the `inter` hops on the comm side.
                        # With the pre-gather both were issued under the
                        # router.
                        if pregather_blk is not None:
                            next_data = pregather_blk.pop(steps[pos + 1])
                            next_tok_ag_handle = pregather_pf.pop(
                                steps[pos + 1]
                            )
                        elif fp8_ring:
                            next_data, next_scale = _raw_fp8_shift_async(
                                cur_data,
                                cur_scale,
                                inter,
                                dst,
                                src,
                                tok_hop_handle,
                            )
                        else:
                            next_data = _raw_inter_shift_async(
                                cur_data,
                                inter,
                                dst,
                                src,
                                tok_hop_handle,
                                zcomm=zc,
                            )
                        if pregather_blk is None and _order_after(
                            tok_group, inter
                        ):
                            next_tok_ag_handle = (
                                _prefetch_tok_ag_fp8(
                                    _fp8_pair(next_data, next_scale, meta),
                                    tok_group,
                                )
                                if fp8_ring
                                else _prefetch_intra_ag(
                                    next_data, tok_group, zcomm=zc
                                )
                            )

                    res = _do_gemm()

                    return res[0] if isinstance(res, (tuple, list)) else res

                seg_pair = cur_pair
                if zc:
                    # The pair is a GIN pool view and, as a leaf of the
                    # manual backward, lives until this layer's backward;
                    # copy it (~1 MiB) so the pool slot is free once the
                    # round's prefetches have read it.
                    with paddle.no_grad():
                        seg_pair = cur_pair.clone()
                    seg_pair.stop_gradient = cur_pair.stop_gradient
                seg_bwd_fn, y = _manual_backward(
                    _seg, is_first_fwd, gathered_tok, seg_pair
                )
                seg_bwd_fns[step] = seg_bwd_fn
                # Round `step` processed the tokens of inter rank
                # (rank - step); its ReduceScatter is issued next round.
                pending_rs = (y[0], (inter_rank - step) % num_nodes)
                if remote_first and step == 1:
                    # Retire the pre-loop hop tasks; round 1 consumed them.
                    self._drain([remote["h"], remote_pair_hop_handle])
                if to_home:
                    # Switch to the home round: local carrier and pair,
                    # and the handles of their pre-issued gathers.
                    cur_data, cur_pair = home_carrier, home_pair
                    cur_scale = gate_pack["scale"]
                    tok_ag_handle, pair_ag_handle = (
                        home_tok_ag_handle,
                        home_pair_ag_handle,
                    )
                elif not is_last:
                    self._drain([tok_hop_handle, pair_comm_handle])
                    if pair_pregather:
                        # The next round processes node (rank - step - 1),
                        # whose gathered pair is that block of all_pairs.
                        block_rows = all_pairs.shape[0] // num_nodes
                        block = (inter_rank - step - 1) % num_nodes
                        next_pair = all_pairs[
                            block * block_rows : (block + 1) * block_rows
                        ]
                    # Hop outputs (and all_pairs slices) are value-only;
                    # mark them so the next segment returns their grads. The
                    # token mirrors the home carrier's stop_gradient.
                    next_pair.stop_gradient = False
                    if next_data is not None:
                        next_data.stop_gradient = home_carrier.stop_gradient
                    # `_order_after` only fails before a comm stream exists:
                    # the hop just created `inter`'s and the subgroup warm-up
                    # created `tok_group`'s.
                    assert next_tok_ag_handle is not None, (
                        "ring gather-ahead (token) not issued: _order_after("
                        "tok_group, inter) returned False -- init_ring_subgroups "
                        "warm-up missing?"
                    )
                    cur_data, cur_scale = next_data, next_scale
                    cur_pair = next_pair
                    tok_ag_handle = next_tok_ag_handle
                    pair_ag_handle = next_pair_ag_handle

            # `_seg` closes over these names and the segments' backward
            # closures keep `_seg` alive until this layer's backward; drop
            # the last hop outputs (pool views on the zcomm paths) now.
            cur_data = cur_scale = cur_pair = None
            # The pre-gathered tokens are a window view on zcomm as well.
            pregather_blk, pregather_pf = None, {}

            # Issue the last ReduceScatter without waiting, so the shared
            # expert below overlaps it as well as the combine.
            if pending_rs is not None:
                _rs_issue(*pending_rs)

            # Inter-node combine. Issuing a collective makes it wait on
            # the calc stream's current tail, so the all-to-all is issued
            # before the shared-expert GEMM is enqueued, ordered after the
            # ReduceScatters on the comm side (or after a wait if that is
            # not possible). Remote-first has already issued it.
            partial_shapes = [list(p.shape) for p in partials]
            partial_rows = partials[0].shape[0]
            if not combine_handle:
                if rs_group is not None and not _order_after(inter, rs_group):
                    for pending_task in rs_tasks:
                        pending_task.wait()
                _combine_a2a_issue(
                    partials, inter, combine_handle, send=combine_send, zcomm=zc
                )
            if has_se:
                se_bwf, se_out = _manual_backward(se_fn, is_first_fwd, res)
                if not isinstance(se_out, tuple):
                    se_out = (se_out,)
            else:
                se_bwf, se_out = None, ()
            # The shared expert is already enqueued, so it overlaps this
            # wait. Partials are value-only; mark them as requiring grad
            # so `_bwd` receives their grads.
            for pending_task in rs_tasks:
                pending_task.wait()
            for p in partials:
                p.stop_gradient = False
            with paddle.no_grad():
                # Home partial plus the received peer partials, in ascending
                # peer order.
                combine_handle["task"].wait()
                # Pop the receive buffer: `combine_handle` is reachable from
                # the rounds' closures until this layer's backward, and with
                # zcomm the buffer is a window view reissued two layers later.
                a2a_recv, chunk_rows = (
                    combine_handle.pop("recv"),
                    combine_handle["T"],
                )
                combined = partials[combine_handle["r0"]]
                for chunk_idx in range(len(combine_handle["peers"])):
                    combined = (
                        combined
                        + a2a_recv[
                            chunk_idx * chunk_rows : (chunk_idx + 1)
                            * chunk_rows
                        ]
                    )
            out = (combined, *tuple(se_out))

            def _bwd(*out_grads):
                # Hand-scheduled backward: reverse every forward collective
                # and overlap each with a neighbouring round's expert
                # backward. Returns one grad per ring-op input.

                def _w_of(g_pair):
                    # Router-weight half of a routing-pair grad.
                    if g_pair is None:
                        return None
                    g_w_half = g_pair[:, k_topk : 2 * k_topk]
                    return (
                        g_w_half.astype(w_dtype)
                        if g_w_half.dtype != w_dtype
                        else g_w_half.contiguous()
                    )

                # Combine backward. Round k's output grad is block (rank - k)
                # of an inter AllGather of g_combined, intra-gathered (reverse
                # of the output ReduceScatter). The home block is the local
                # g_combined, so its gather starts at once and round 0's
                # backward runs first. The remote gathers are ordered after
                # the inter gather on the comm side and issued before any
                # backward GEMM, so they overlap the shared-expert and home
                # backward; without comm-side ordering they are issued after
                # a wait, behind the shared expert.
                g_combined = out_grads[0]
                g_se_out = out_grads[1:] if has_se else ()
                g_round_out = [None] * num_nodes
                g_out_ag_tasks = [None] * num_nodes
                g_round_out[0], g_out_ag_tasks[0] = _token_all_gather_async(
                    g_combined.contiguous(), rs_group, zcomm=zc
                )
                g_all, g_all_task = _inter_all_gather_async(
                    g_combined.contiguous(), inter, zcomm=zc
                )

                def _issue_remote_out_gathers():
                    # Slices of g_all are views; no copy is enqueued. On
                    # zcomm g_all is a window slot: drop it once sliced (each
                    # slice's gather is waited on before its round's expert
                    # backward, which precedes any transfer that could reuse
                    # the slot).
                    nonlocal g_all, g_all_task
                    for k in range(num_nodes - 1, 0, -1):
                        d = (inter_rank - k) % num_nodes
                        g_partial = g_all[
                            d * partial_rows : (d + 1) * partial_rows
                        ]
                        if list(g_partial.shape) != partial_shapes[d]:
                            g_partial = g_partial.reshape(partial_shapes[d])
                        g_round_out[k], g_out_ag_tasks[k] = (
                            _token_all_gather_async(
                                g_partial, rs_group, zcomm=zc
                            )
                        )
                    if zc:
                        g_all = g_all_task = None

                remote_ag_issued = _order_after(rs_group, inter)
                if remote_ag_issued:
                    _issue_remote_out_gathers()
                g_se = None

                def _run_se_bwd():
                    # Shared-expert backward: independent of the ring chain,
                    # so it can be enqueued wherever it covers most comm.
                    nonlocal g_se
                    if se_bwf is None:
                        return
                    se_grads = se_bwf(*g_se_out)
                    g_se = (
                        se_grads[0]
                        if isinstance(se_grads, (tuple, list))
                        else se_grads
                    )

                # zcomm with a token-hop join (see `join_token_hop`): the
                # router backward covers the end of this backward. The last
                # round's token-grad ReduceScatter is not waited on here, the
                # final token hop is ordered after it on the comm side, and
                # the join waits for the hop. The shared expert then has
                # nothing to cover at the end and runs first, covering the
                # home round's output-grad gather, which the home round's
                # expert backward waits for right away. (The NCCL path keeps
                # its schedule.)
                defer_last_rs = zc and tok_hop_slot is not None
                # Otherwise the output gathers above are already ordered on
                # the comm side and the home round's expert backward covers
                # the inter one, while the final reverse hop at the end of
                # this function has no compute left after it. Moving the
                # shared expert there trades a covered gather for a covered
                # hop. Only safe when the remote gathers were issued without
                # waiting; otherwise the wait below is what the shared expert
                # has to cover.
                se_late = remote_ag_issued and not defer_last_rs
                # With a join the router backward already covers the final
                # hop, so the held-back shared expert covers the last remote
                # round's token-grad ReduceScatter instead (see after the
                # loop).
                se_covers_last_tok_rs = se_late and tok_hop_slot is not None
                if not se_late:
                    _run_se_bwd()
                if not remote_ag_issued:
                    g_all_task.wait()
                    _issue_remote_out_gathers()
                # Per-round grads w.r.t. the round's local token and pair
                # (the pair grad is gathered-shaped for pre-gathered rounds).
                local_g_tok = [None] * num_nodes
                local_g_pair = [None] * num_nodes
                # (round, out, task, input) of the in-flight token-grad
                # ReduceScatter of the previous segment.
                pending_tok_rs = None
                # Reverse-hop chain: round j's token grad (and, without the
                # pre-gather, its weight grad) hops back to `src` and is added
                # to round j-1's grads there; g_tok_chain[j] / g_w_chain[j]
                # hold round j's accumulated grads. Each hop is issued once
                # its round is complete, overlapping the remaining GEMMs.
                # Weight hops use `pair_inter_group`, so they run beside the
                # token hops on `inter` instead of queueing behind them.
                g_tok_chain = [None] * num_nodes
                g_w_chain = [None] * num_nodes
                pending_rev_hop = None
                # Pre-gather (forward not remote-first): in-flight inter
                # ReduceScatter of all remote rounds' routing grads.
                rt_rs = {}

                def _rt_rs_issue():
                    # Reverse of the routing pre-gather: stack the remote
                    # rounds' gathered pair grads in node order, with zeros for
                    # this node's block (the home grad goes back through its
                    # own segment), and reduce-scatter over inter.
                    remote_g_pairs = [
                        local_g_pair[k] for k in range(1, num_nodes)
                    ]
                    ref = next(
                        (g for g in remote_g_pairs if g is not None), None
                    )
                    if ref is None:
                        return
                    blocks = []
                    for d in range(num_nodes):
                        g = (
                            local_g_pair[(inter_rank - d) % num_nodes]
                            if d != inter_rank
                            else None
                        )
                        blocks.append(
                            g.contiguous()
                            if g is not None
                            else paddle.zeros_like(ref)
                        )
                    src_buf = paddle.concat(blocks, axis=0)
                    out, task = _reduce_scatter_async(src_buf, pair_inter_group)
                    rt_rs.update({"out": out, "task": task, "src": src_buf})

                def _rt_rs_finish(overlap=None):
                    # Finish with the intra ReduceScatter, giving the remote
                    # rounds' contribution to the home router-weight grad.
                    # `overlap` is enqueued after the transfer is issued and
                    # before it is waited on: issuing a collective orders it
                    # after the calc stream's current tail, so compute
                    # enqueued first would serialize the transfer behind it
                    # instead of hiding it.
                    if not rt_rs:
                        if overlap is not None:
                            overlap()
                        return None
                    rt_rs["task"].wait()
                    local_grad, t = _reduce_scatter_async(
                        rt_rs.pop("out"), rt_group
                    )
                    if overlap is not None:
                        overlap()
                    t.wait()
                    rt_rs.clear()
                    return _w_of(local_grad)

                def _fold(j):
                    # Round j >= 1 is complete: add the reverse hop from round
                    # j+1 to its own grads. For j >= 2, send the result on to
                    # round j-1; the last link (1 -> 0) is issued after the
                    # loop.
                    nonlocal pending_rev_hop
                    g_tok_j = local_g_tok[j]
                    # With the pre-gather, weight grads go back through
                    # `_rt_rs_issue` instead of the hop chain.
                    g_w_j = None if pair_pregather else _w_of(local_g_pair[j])
                    if pending_rev_hop is not None:
                        (tok_hop, tok_rev), (w_hop, w_rev) = pending_rev_hop
                        pending_rev_hop = None
                        self._drain([tok_hop, w_hop])
                        g_tok_j = _add_hop(g_tok_j, tok_rev)
                        g_w_j = _add_hop(g_w_j, w_rev)
                        # With zcomm the hop outputs are window views
                        # reissued two hops of their size later, and the
                        # next hop is issued below: drop them first.
                        del tok_rev, w_rev
                    g_tok_chain[j] = g_tok_j
                    g_w_chain[j] = g_w_j
                    if j >= 2:
                        tok_hop, w_hop = {}, {}
                        # The send buffer stays referenced by the handle: the
                        # 2 -> 1 hop is not waited on the calc stream when the
                        # last RS is deferred (see after the loop).
                        if g_tok_j is not None:
                            tok_hop["src"] = g_tok_j.contiguous()
                        tok_rev = (
                            _raw_inter_shift_async(
                                tok_hop["src"],
                                inter,
                                src,
                                dst,
                                tok_hop,
                                zcomm=zc,
                            )
                            if g_tok_j is not None
                            else None
                        )
                        w_rev = (
                            _raw_inter_shift_async(
                                g_w_j.contiguous(),
                                pair_inter_group,
                                src,
                                dst,
                                w_hop,
                                zcomm=zc,
                            )
                            if g_w_j is not None
                            else None
                        )
                        pending_rev_hop = ((tok_hop, tok_rev), (w_hop, w_rev))

                # Home round first, then the remote rounds in descending
                # order, which is the order the reverse-hop chain folds them.
                bwd_order = [0, *range(num_nodes - 1, 0, -1)]
                for k in bwd_order:
                    g_out_ag_tasks[k].wait()
                    # Expert backward of round k; it overlaps the previous
                    # segment's token-grad ReduceScatter. The routing-pair
                    # grad is reduced inside the segment by the router gather
                    # op (gathered-shaped for pre-gathered rounds).
                    g_tok_gathered, g_pair_local = seg_bwd_fns[k](
                        g_round_out[k]
                    )
                    local_g_pair[k] = g_pair_local
                    if pair_pregather and k == 1:
                        # All remote routing grads are in.
                        _rt_rs_issue()
                    if pending_tok_rs is not None:
                        rs_round, rs_out, rs_task, _rs_input = pending_tok_rs
                        if rs_task is not None:
                            rs_task.wait()
                            local_g_tok[rs_round] = rs_out
                        pending_tok_rs = None
                    if 0 < k < num_nodes - 1:
                        # Round k+1's token grad was just completed.
                        _fold(k + 1)
                    # Reduce-scatter this round's gathered token grad async;
                    # it is waited on after the next segment's backward.
                    tok_rs_group = tok_rs_groups[k]
                    tok_rs_src = (
                        g_tok_gathered.contiguous()
                        if (
                            tok_rs_group is not None
                            and tok_rs_group.nranks > 1
                            and g_tok_gathered is not None
                        )
                        else None
                    )
                    if tok_rs_src is not None:
                        rs_out, rs_task = _intra_rs_async(
                            tok_rs_src,
                            tok_rs_group,
                            zcomm=zc,
                            # The last one is exposed unless deferred.
                            overlapped=k != bwd_order[-1] or defer_last_rs,
                        )
                        pending_tok_rs = (k, rs_out, rs_task, tok_rs_src)
                    else:
                        local_g_tok[k] = g_tok_gathered
                        pending_tok_rs = None
                # The loop ended on round 1: wait for its token-grad
                # ReduceScatter (it overlapped any in-flight 2 -> 1 hop),
                # then fold round 1 before the final link. With a join it is
                # not waited on: the final hop is ordered after it below, and
                # everything the comm streams still read or write (its input
                # and output, the reverse hop and the add below) stays
                # referenced until the join drains the hop; freed early, the
                # calc stream could reuse that memory under them.
                # The shared expert held back for the join is enqueued after
                # that ReduceScatter was issued, so it overlaps it.
                if se_covers_last_tok_rs:
                    _run_se_bwd()
                deferred_rs = None
                if pending_tok_rs is not None:
                    rs_round, rs_out, rs_task, _rs_input = pending_tok_rs
                    if rs_task is not None:
                        if defer_last_rs:
                            deferred_rs = (
                                rs_task,
                                tok_rs_groups[rs_round],
                                _rs_input,
                                rs_out,
                            )
                        else:
                            rs_task.wait()
                        local_g_tok[rs_round] = rs_out
                    pending_tok_rs = None
                # zcomm: tasks of comm-stream adds that only the join's wait
                # for the final hop covers (see `_RingTokHopJoin`).
                add_tasks = []
                if deferred_rs is not None and pending_rev_hop is not None:
                    # Round 1's token grad is the RS output plus the reverse
                    # hop from round 2. Add them on the inter comm stream,
                    # after the RS (comm-side) and after the hop (same
                    # stream), so the calc stream waits for neither.
                    (tok_hop, tok_rev), (w_hop, w_rev) = pending_rev_hop
                    pending_rev_hop = None
                    self._drain([w_hop])
                    g_w_1 = None if pair_pregather else _w_of(local_g_pair[1])
                    g_w_chain[1] = _add_hop(g_w_1, w_rev)
                    del w_rev
                    if not _order_after(inter, deferred_rs[1]):
                        deferred_rs[0].wait()
                    if tok_rev is None:
                        g_tok_chain[1] = local_g_tok[1]
                    else:
                        from .zcomm_comm import stream_add

                        # Same bf16 add (and rounding) as `_add_hop`.
                        g_tok_chain[1] = stream_add(
                            local_g_tok[1], tok_rev, inter, tasks=add_tasks
                        )
                    deferred_rs = (*deferred_rs, tok_hop, tok_rev, *add_tasks)
                    del tok_rev
                else:
                    _fold(1)

                # Rounds 1..N-1 are folded. Seed the home round's grads.
                g_w_chain[0] = _w_of(local_g_pair[0])
                g_tok_chain[0] = local_g_tok[0]
                # Final link (round 1 -> 0). The send buffers stay
                # referenced until the hops are drained.
                last_tok_hop, last_w_hop = {}, {}
                last_tok_src = last_w_src = last_tok_rev = last_w_rev = None
                if deferred_rs is not None and not _order_after(
                    inter, deferred_rs[1]
                ):
                    deferred_rs[0].wait()
                if g_tok_chain[1] is not None:
                    last_tok_src = g_tok_chain[1].contiguous()
                    last_tok_rev = _raw_inter_shift_async(
                        last_tok_src,
                        inter,
                        src,
                        dst,
                        last_tok_hop,
                        zcomm=zc,
                    )
                if g_w_chain[1] is not None:
                    last_w_src = g_w_chain[1].contiguous()
                    last_w_rev = _raw_inter_shift_async(
                        last_w_src,
                        pair_inter_group,
                        src,
                        dst,
                        last_w_hop,
                        zcomm=zc,
                    )
                # The hops are in flight and every expert backward is done, so
                # this is the one place in the backward with nothing left to
                # cover a cross-node transfer. The shared expert, when held
                # back for exactly that, is enqueued inside `_rt_rs_finish`
                # so it also covers the router-grad ReduceScatter of the
                # pre-gather (which must stay fp32, so overlapping it is the
                # only thing left to do with it).
                g_w_pregather = _rt_rs_finish(
                    overlap=_run_se_bwd
                    if se_late and not se_covers_last_tok_rs
                    else None
                )
                g_home_tok = g_tok_chain[0]
                # With a join (see `join_token_hop`), the token hop is left
                # in flight: the join waits for it after the router backward
                # is enqueued, and adds last_tok_rev itself (into a new
                # tensor, so a zcomm window view never becomes the grad).
                defer_tok_hop = (
                    tok_hop_slot is not None
                    and last_tok_rev is not None
                    and g_home_tok is not None
                )
                if defer_tok_hop:
                    tok_hop_slot.update(
                        hop=last_tok_hop,
                        rev=last_tok_rev,
                        src=last_tok_src,
                        keep=deferred_rs,
                    )
                    self._drain([last_w_hop])
                else:
                    self._drain([last_tok_hop, last_w_hop])
                g_w_chain[0] = _add_hop(g_w_chain[0], last_w_rev)
                if g_w_pregather is not None:
                    g_w_chain[0] = (
                        g_w_pregather
                        if g_w_chain[0] is None
                        else g_w_chain[0] + g_w_pregather
                    )
                if not defer_tok_hop:
                    g_home_tok = _add_hop(g_home_tok, last_tok_rev)
                leaf_grads = [g_home_tok, g_w_chain[0]]
                if has_se:
                    leaf_grads.append(g_se)
                return tuple(leaf_grads)

            return out, _bwd

        out = _RingRoundsExplicit.apply(
            *leaves,
            plan=_plan,
            relax_dtype=self.fp8_dispatch,
        )

        if isinstance(out, (tuple, list)):
            combined, se_out = out[0], tuple(out[1:])
        else:
            combined, se_out = out, ()
        if has_se:
            combine_overlap_handle["fn_out"] = se_out
        return combined
