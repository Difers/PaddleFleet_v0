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

# Refer to NVIDIA Megatron-LM https://github.com/NVIDIA/Megatron-LM.git
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import paddle

if TYPE_CHECKING:
    import paddle.distributed.fleet.base.topology as topo
import warnings

from .utils._fleet_utils import GlobalMemoryBuffer

logger = logging.getLogger(__name__)

# Intra-layer model parallel group that the current rank belongs to.
_TENSOR_MODEL_PARALLEL_GROUP = None
_TENSOR_MODEL_PARALLEL_GLOBAL_RANKS = None

### Expert-related parallel states
# Naming convention:
# _EXPERT prefix in group name means it's used for expert layer in MoE models.
# _EXPERT_MODEL denotes expert parallelism which splits number of experts across the group.
# _EXPERT_TENSOR denotes tensor parallelism of expert which splits tensor across the group.
# _EXPERT_DATA denotes data parallelism of expert which replicates weight across the group.

# Expert model parallel group that current rank belongs to.
_EXPERT_MODEL_PARALLEL_GROUP = None
# Expert tensor parallel group that current rank belongs to.
_EXPERT_TENSOR_PARALLEL_GROUP = None
# Expert tensor and model combined parallel group
_EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP = None
# Expert tensor, model, pipeline combined parallel group
_EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP = None
# Expert data parallel group
_EXPERT_DATA_PARALLEL_GROUP = None
_EXPERT_DATA_PARALLEL_GROUP_GLOO = None
_INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP = None
_INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP_GLOO = None
_INTER_PARTIAL_EXPERT_DATA_PARALLEL_GROUP = None
# Parallel state values changed on the fly
_MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_EXPERT_MODEL_PARALLEL_RANK = None
_MPU_EXPERT_TENSOR_PARALLEL_WORLD_SIZE = None
_MPU_EXPERT_TENSOR_PARALLEL_RANK = None
### End of expert related parallel states

# Inter-layer model parallel group that the current rank belongs to.
_PIPELINE_MODEL_PARALLEL_GROUP = None
_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
_PIPELINE_MODEL_PARALLEL_RANK = None

_VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = None
_VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None

# Data parallel group that the current rank belongs to.
_DATA_PARALLEL_GROUP = None

# Expert model parallel group that current rank belongs to.
_EXPERT_MODEL_PARALLEL_GROUP = None

# Expert data parallel group
_EXPERT_DATA_PARALLEL_GROUP = None

# Context parallel group that the current rank belongs to.
# NOTE(dcp): this stays the *static* (global) CP group built by fleet. The group
# actually used by CP collectives is resolved by get_context_parallel_group(),
# which prefers the dynamic group selected for the current microbatch.
_CONTEXT_PARALLEL_GROUP = None

### Dynamic context parallel (DCP) states
# Pre-built CP groups of every allowed size, only the one this rank belongs to
# is kept: {group_size: comm_group}. Size 1 maps to None, which means "no CP
# collective needed" and matches the existing `group is None -> world size 1`
# convention below.
_DYNAMIC_CONTEXT_PARALLEL_GROUPS = {}
# Size of the CP group selected for the current microbatch. None means DCP is
# not driving the group choice, so the static group is used.
_CURRENT_CONTEXT_PARALLEL_SIZE = None
# Effective value of max_dynamic_context_parallel_size (defaults to the pool
# size, i.e. sharding_degree).
_MAX_DYNAMIC_CONTEXT_PARALLEL_SIZE = None
### End of dynamic context parallel states

# Data parallel group information with context parallel combined.
_DATA_PARALLEL_GROUP_WITH_CP = None
# These values enable us to change the mpu sizes on the fly.
_MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_TENSOR_MODEL_PARALLEL_RANK = None


# Memory buffers to avoid dynamic memory allocation
_GLOBAL_MEMORY_BUFFER = None


def initialize_model_parallel(
    hcg: topo.EPHybridCommunicateGroup | topo.HybridCommunicateGroup,
    virtual_pipeline_model_parallel_size: int | None = None,
    enable_dynamic_context_parallel: bool = False,
    max_dynamic_context_parallel_size: int | None = None,
):
    global _TENSOR_MODEL_PARALLEL_GROUP
    global _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS
    _TENSOR_MODEL_PARALLEL_GROUP = hcg._mp_comm_group
    _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS = hcg._mp_group

    global _PIPELINE_MODEL_PARALLEL_GROUP
    _PIPELINE_MODEL_PARALLEL_GROUP = hcg._pp_comm_group

    global _DATA_PARALLEL_GROUP
    _DATA_PARALLEL_GROUP = hcg._sharding_comm_group

    global _EXPERT_MODEL_PARALLEL_GROUP
    global _EXPERT_DATA_PARALLEL_GROUP
    _EXPERT_MODEL_PARALLEL_GROUP = getattr(hcg, "_ep_comm_group", None)
    _EXPERT_DATA_PARALLEL_GROUP = getattr(hcg, "_moe_sharding_comm_group", None)

    global _CONTEXT_PARALLEL_GROUP
    global _DATA_PARALLEL_GROUP_WITH_CP
    _CONTEXT_PARALLEL_GROUP = getattr(hcg, "_cp_comm_group", None)
    _DATA_PARALLEL_GROUP_WITH_CP = getattr(hcg, "_cp_sharding_comm_group", None)

    if enable_dynamic_context_parallel:
        initialize_dynamic_context_parallel_groups(
            hcg, max_dynamic_context_parallel_size
        )

    if virtual_pipeline_model_parallel_size is not None:
        if not hcg._pp_comm_group.nranks > 1:
            raise RuntimeError(
                "pipeline-model-parallel size should be greater than 1 with interleaved schedule"
            )
        global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
        global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
        _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = 0
        _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = (
            virtual_pipeline_model_parallel_size
        )

    # Initialize global memory buffer
    # This isn't really "parallel state" but there isn't another good place to
    # put this. If we end up with a more generic initialization of paddlefleet
    # we could stick it there
    _set_global_memory_buffer()


def get_tensor_model_parallel_group(check_initialized=True):
    """Get the tensor-model-parallel group the caller rank belongs to."""
    if check_initialized:
        assert _TENSOR_MODEL_PARALLEL_GROUP is not None, (
            "tensor model parallel group is not initialized"
        )
    return _TENSOR_MODEL_PARALLEL_GROUP


def get_tensor_model_parallel_world_size():
    """Return world size for the tensor-model-parallel group."""
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    if _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    return (
        1
        if get_tensor_model_parallel_group(False) is None
        else get_tensor_model_parallel_group().nranks
    )


def get_tensor_model_parallel_rank():
    """Return caller's rank for the tensor-model-parallel group."""
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    if _MPU_TENSOR_MODEL_PARALLEL_RANK is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_RANK
    if get_tensor_model_parallel_world_size() == 1:
        return 0
    return get_tensor_model_parallel_group().rank


def get_pipeline_model_parallel_group(check_initialized=True):
    """Get the pipeline-model-parallel group the caller rank belongs to."""
    if check_initialized:
        assert _PIPELINE_MODEL_PARALLEL_GROUP is not None, (
            "pipeline_model parallel group is not initialized"
        )
    return _PIPELINE_MODEL_PARALLEL_GROUP


def get_data_parallel_group(
    check_initialized=True, with_context_parallel=False
):
    """Get the data-parallel group the caller rank belongs to."""
    if with_context_parallel:
        if check_initialized:
            assert _DATA_PARALLEL_GROUP_WITH_CP is not None, (
                "data parallel group with context parallel combined is not initialized"
            )
        return _DATA_PARALLEL_GROUP_WITH_CP
    else:
        if check_initialized:
            assert _DATA_PARALLEL_GROUP is not None, (
                "data parallel group is not initialized"
            )
        return _DATA_PARALLEL_GROUP


def get_expert_model_parallel_group(check_initialized=True):
    """Get the expert-model-parallel group the caller rank belongs to."""
    if check_initialized:
        assert _EXPERT_MODEL_PARALLEL_GROUP is not None, (
            "expert model parallel group is not initialized"
        )
    return _EXPERT_MODEL_PARALLEL_GROUP


def get_expert_data_parallel_group(check_initialized=True):
    """Get expert data parallel group."""
    if check_initialized:
        assert _EXPERT_DATA_PARALLEL_GROUP is not None, (
            "Expert data parallel group is not initialized"
        )
    return _EXPERT_DATA_PARALLEL_GROUP


def get_context_parallel_group(check_initialized=False):
    """Get the context-parallel group the caller rank belongs to.

    When DCP has selected a size for the current microbatch, that dynamic group
    is returned; otherwise the static group built by fleet is returned.
    """
    if _CURRENT_CONTEXT_PARALLEL_SIZE is not None:
        return _DYNAMIC_CONTEXT_PARALLEL_GROUPS[_CURRENT_CONTEXT_PARALLEL_SIZE]
    if check_initialized:
        assert _CONTEXT_PARALLEL_GROUP is not None, (
            "context parallel group is not initialized"
        )
    return _CONTEXT_PARALLEL_GROUP


def get_context_parallel_world_size():
    """Return world size of the CP group currently in effect."""
    if get_context_parallel_group() is None:
        return 1
    else:
        return get_context_parallel_group().world_size


def get_global_context_parallel_group(check_initialized=False):
    """Get the static (whole-run) context-parallel group.

    Use this instead of get_context_parallel_group() whenever the answer must
    not change from microbatch to microbatch under DCP -- e.g. parameter
    sharding/placement decisions and anything baked into layer construction.
    """
    if check_initialized:
        assert _CONTEXT_PARALLEL_GROUP is not None, (
            "context parallel group is not initialized"
        )
    return _CONTEXT_PARALLEL_GROUP


def get_global_context_parallel_world_size():
    """Return world size of the static context-parallel group."""
    if get_global_context_parallel_group() is None:
        return 1
    else:
        return get_global_context_parallel_group().world_size


### Dynamic context parallel (DCP) group management


def _dynamic_context_parallel_pools(hcg):
    """Return the rank pools DCP groups are carved out of, one per sharding group.

    This mirrors exactly how fleet builds the static CP group, so DCP inherits
    the same rank ordering instead of guessing a stride:
    `split_context_comm_list` (paddle/distributed/fleet/base/topology.py)
    takes `merge_inner_comm_list(topo, "moe_sharding", "dense_sharding")` -- one
    list of `sharding_degree` ranks per sharding group -- and slices it into
    contiguous blocks of `cp_degree`, i.e. inside a pool the layout is
    `idx = cp_sharding_idx * cp_degree + cp_idx` with cp innermost.
    DCP only relaxes the block length from a fixed `cp_degree` to any `s`.
    """
    return hcg.merge_inner_comm_list(
        hcg._dense_topo, "moe_sharding", "dense_sharding"
    )


def initialize_dynamic_context_parallel_groups(
    hcg, max_dynamic_context_parallel_size=None
):
    """Pre-build every CP group DCP may switch to at run time.

    NCCL groups cannot be created per step, so all power-of-two group sizes up
    to `max_dynamic_context_parallel_size` are created up front. The default is
    the pool size (== sharding_degree), which allows a single long sample to be
    spread over the whole sharding group.

    Every rank walks `(size, pool, block)` in the same order, because
    `paddle.distributed.new_group` is a collective call and a divergent order
    hangs.
    """
    global _DYNAMIC_CONTEXT_PARALLEL_GROUPS
    global _MAX_DYNAMIC_CONTEXT_PARALLEL_SIZE
    global _CURRENT_CONTEXT_PARALLEL_SIZE

    _DYNAMIC_CONTEXT_PARALLEL_GROUPS = {}
    _CURRENT_CONTEXT_PARALLEL_SIZE = None

    # Only EPHybridCommunicateGroup owns a `context` axis; on a plain
    # HybridCommunicateGroup there is no CP at all.
    if not hasattr(hcg, "_cp_topo"):
        _MAX_DYNAMIC_CONTEXT_PARALLEL_SIZE = 1
        _DYNAMIC_CONTEXT_PARALLEL_GROUPS = {1: None}
        logger.info(
            "[DCP] requested but this HybridCommunicateGroup has no `context` "
            "axis (only EPHybridCommunicateGroup does), so DCP stays disabled "
            "with a single size-1 group"
        )
        return _DYNAMIC_CONTEXT_PARALLEL_GROUPS

    cp_degree = hcg._cp_degree
    # NOTE(dcp): pp/mp are out of scope for now, and silently producing wrong
    # groups is worse than refusing to start.
    #   - mp > 1: the CP group is no longer a stride-1 block of the sharding
    #     group (`model` sits inside `context` in the topology axis order), so
    #     the size-aligned contiguous-block carving below would build groups
    #     whose members are not actually CP peers.
    #   - pp > 1: `merge_inner_comm_list` returns one pool per pipeline stage,
    #     and the scheduler has no notion of per-stage pools yet -- every stage
    #     must agree on the group table, which is untested.
    if getattr(hcg, "_mp_degree", 1) != 1:
        raise NotImplementedError(
            f"dynamic context parallel does not support tensor model parallel "
            f"yet (mp_degree={hcg._mp_degree}); the CP group would stop being a "
            f"contiguous block of the sharding group"
        )
    if getattr(hcg, "_pp_degree", 1) != 1:
        raise NotImplementedError(
            f"dynamic context parallel does not support pipeline parallel yet "
            f"(pp_degree={hcg._pp_degree}); the per-stage rank pools and the "
            f"per-microbatch group switching have not been worked through"
        )

    pools = _dynamic_context_parallel_pools(hcg)
    pool_size = len(pools[0])
    assert pool_size == cp_degree * hcg._cp_sharding_degree, (
        f"pool size {pool_size} != cp_degree {cp_degree} * "
        f"cp_sharding_degree {hcg._cp_sharding_degree}"
    )

    # max_dynamic_context_parallel_size is the largest CP group DCP may use.
    # Groups are size-aligned blocks of power-of-two length, so a legal cap is
    # exactly "a power of two that divides the pool" -- then every power of two
    # up to the cap also divides the pool (size | cap | pool_size), and the cap
    # is literally the biggest group that can be built. Both properties are
    # checked instead of silently rounded down, because rounding down would
    # make the run use a smaller cap than configured without saying so.
    if max_dynamic_context_parallel_size is None:
        max_dynamic_context_parallel_size = 1
        while max_dynamic_context_parallel_size * 2 <= pool_size and (
            pool_size % (max_dynamic_context_parallel_size * 2) == 0
        ):
            max_dynamic_context_parallel_size *= 2
    if not 1 <= max_dynamic_context_parallel_size <= pool_size:
        raise ValueError(
            f"max_dynamic_context_parallel_size "
            f"{max_dynamic_context_parallel_size} must be in [1, {pool_size}]"
        )
    if pool_size % max_dynamic_context_parallel_size != 0:
        raise ValueError(
            f"max_dynamic_context_parallel_size "
            f"{max_dynamic_context_parallel_size} must divide the rank pool "
            f"(sharding_degree) {pool_size}"
        )
    if (
        max_dynamic_context_parallel_size
        & (max_dynamic_context_parallel_size - 1)
    ) != 0:
        raise ValueError(
            f"max_dynamic_context_parallel_size "
            f"{max_dynamic_context_parallel_size} must be a power of two; CP "
            f"groups are power-of-two size-aligned blocks"
        )
    _MAX_DYNAMIC_CONTEXT_PARALLEL_SIZE = max_dynamic_context_parallel_size

    sizes = []
    size = 1
    while size <= max_dynamic_context_parallel_size:
        # size | cap | pool_size holds by the checks above, so every size here
        # tiles the pool; no filtering needed.
        sizes.append(size)
        size *= 2

    global_rank = hcg.global_rank
    n_new_groups = 0
    for size in sizes:
        my_group = None
        my_ranks = None
        for pool in pools:
            for start in range(0, pool_size, size):
                ranks = pool[start : start + size]
                if size == 1:
                    # A one-rank CP group needs no communicator; None keeps the
                    # existing "no group -> world size 1" convention.
                    comm_group = None
                elif size == cp_degree:
                    # Reuse the static CP group instead of duplicating it.
                    comm_group = (
                        hcg._cp_comm_group if global_rank in ranks else None
                    )
                else:
                    comm_group = paddle.distributed.new_group(ranks=ranks)
                    n_new_groups += 1
                if global_rank in ranks:
                    my_group = comm_group
                    my_ranks = ranks
        assert my_ranks is not None, (
            f"rank {global_rank} is in no CP group of size {size}"
        )
        if size == cp_degree:
            # Self-check on the pool layout: our own carving must reproduce the
            # group fleet built. If this fires, the cp-innermost assumption
            # above no longer holds for this topology.
            assert list(my_ranks) == list(hcg._cp_group), (
                f"dcp group of size {cp_degree} is {my_ranks} but fleet's "
                f"static cp group is {hcg._cp_group}"
            )
        _DYNAMIC_CONTEXT_PARALLEL_GROUPS[size] = my_group

    # Warm every communicator up here, the way Megatron does right after building
    # its dynamic dp-cp groups (Megatron's parallel_state). NCCL builds a
    # communicator lazily on first use, so without this the first microbatch that
    # happens to select a given size pays the setup cost mid-step. A plain barrier
    # only rendezvouses the group; it does NOT exercise the two ops DCP actually
    # runs (``all_gather`` for CP KV, ``alltoall_single`` for the reroute), and NCCL
    # picks channels/algorithm/buffers per op-type on first real call. So warm both
    # ops on each size group with tiny fake tensors (a barrier-only warmup still
    # leaves a latency spike on the first real op). Safe as a collective: each
    # rank owns exactly one group per size and walks sizes in the same order, so all
    # members of a group reach each op together.
    _warm = paddle.ones([8], dtype="int64")
    for size in sizes:
        group = _DYNAMIC_CONTEXT_PARALLEL_GROUPS[size]
        if group is None:
            continue
        nranks = group.nranks
        # CP attention path: all_gather, the op CP KV gathering runs.
        gathered = []
        paddle.distributed.all_gather(gathered, _warm, group=group)
        # Reroute path: alltoall_single, called the way the external DCP driver
        # calls it (out, in, send_splits, recv_splits, group -- equal 1-splits).
        send = paddle.ones([nranks], dtype="int64")
        recv = paddle.empty([nranks], dtype="int64")
        paddle.distributed.alltoall_single(
            recv, send, [1] * nranks, [1] * nranks, group=group
        )
    paddle.device.cuda.synchronize()

    # Logged on every rank: the per-rank group table is what a DCP run has to be
    # checked against, and it differs between ranks.
    my_groups = ", ".join(
        f"{size}: "
        + (
            "None(no comm)"
            if _DYNAMIC_CONTEXT_PARALLEL_GROUPS[size] is None
            else str(_DYNAMIC_CONTEXT_PARALLEL_GROUPS[size].ranks)
        )
        for size in sizes
    )
    logger.info(
        f"[DCP] dynamic context parallel groups built: rank={global_rank} "
        f"n_pools={len(pools)} pool_size(sharding_degree)={pool_size} "
        f"cp_degree={cp_degree} cp_sharding_degree={hcg._cp_sharding_degree} "
        f"max_dynamic_cp_size={max_dynamic_context_parallel_size} "
        f"sizes={sizes} new_nccl_groups={n_new_groups} "
        f"(size 1 needs none, size {cp_degree} reuses the static cp group) "
        f"my_groups={{{my_groups}}}"
    )

    return _DYNAMIC_CONTEXT_PARALLEL_GROUPS


def get_dynamic_context_parallel_sizes():
    """Return the CP group sizes that were pre-built, ascending."""
    return sorted(_DYNAMIC_CONTEXT_PARALLEL_GROUPS.keys())


def get_max_dynamic_context_parallel_size():
    """Return the effective max_dynamic_context_parallel_size."""
    return _MAX_DYNAMIC_CONTEXT_PARALLEL_SIZE


def get_dynamic_context_parallel_group(size):
    """Return the pre-built CP group of the given size for this rank."""
    assert size in _DYNAMIC_CONTEXT_PARALLEL_GROUPS, (
        f"no pre-built CP group of size {size}; available sizes are "
        f"{get_dynamic_context_parallel_sizes()}"
    )
    return _DYNAMIC_CONTEXT_PARALLEL_GROUPS[size]


def set_current_context_parallel_size(size):
    """Switch the CP group used by subsequent CP collectives.

    Called once per microbatch with that microbatch's `local_cp_size`.
    """
    global _CURRENT_CONTEXT_PARALLEL_SIZE
    assert size in _DYNAMIC_CONTEXT_PARALLEL_GROUPS, (
        f"no pre-built CP group of size {size}; available sizes are "
        f"{get_dynamic_context_parallel_sizes()}"
    )
    _CURRENT_CONTEXT_PARALLEL_SIZE = size


def get_current_context_parallel_size():
    """Return the CP size selected for the current microbatch, or None."""
    return _CURRENT_CONTEXT_PARALLEL_SIZE


def reset_current_context_parallel_size():
    """Fall back to the static CP group."""
    global _CURRENT_CONTEXT_PARALLEL_SIZE
    _CURRENT_CONTEXT_PARALLEL_SIZE = None


### End of dynamic context parallel group management


def get_context_parallel_rank():
    """Return this rank's index within the context-parallel group (0 if CP is disabled)."""
    if get_context_parallel_group() is None:
        return 0
    return get_context_parallel_group().rank


def get_pipeline_model_parallel_world_size():
    """Return world size for the pipeline-model-parallel group."""
    global _PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    if _PIPELINE_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    return (
        1
        if get_pipeline_model_parallel_group(False) is None
        else get_pipeline_model_parallel_group().nranks
    )


def set_pipeline_model_parallel_world_size(world_size):
    """Set the pipeline-model-parallel size"""
    global _PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _PIPELINE_MODEL_PARALLEL_WORLD_SIZE = world_size


def get_pipeline_model_parallel_rank():
    """Return caller's rank for the pipeline-model-parallel group."""
    global _PIPELINE_MODEL_PARALLEL_RANK
    if _PIPELINE_MODEL_PARALLEL_RANK is not None:
        return _PIPELINE_MODEL_PARALLEL_RANK
    if get_pipeline_model_parallel_world_size() == 1:
        return 0
    if (
        paddle.distributed.is_initialized()
        and get_pipeline_model_parallel_group(False) is not None
    ):
        return get_pipeline_model_parallel_group().rank
    else:
        return 0


def is_pipeline_first_stage(ignore_virtual=True, vp_stage=None):
    """Return True if in the first pipeline model-parallel stage, False otherwise."""
    if (
        not ignore_virtual
        and get_virtual_pipeline_model_parallel_world_size() is not None
    ):
        assert vp_stage is not None, (
            "vp_stage must be passed if virtual pipeline is enabled"
        )

        if vp_stage != 0:
            return False
    return get_pipeline_model_parallel_rank() == 0


def is_pipeline_last_stage(ignore_virtual=True, vp_stage=None):
    """Return True if in the last pipeline-model-parallel stage, False otherwise."""
    if (
        not ignore_virtual
        and get_virtual_pipeline_model_parallel_world_size() is not None
    ):
        assert vp_stage is not None, (
            "vp_stage must be passed if virtual pipeline is enabled"
        )

        if vp_stage != (get_virtual_pipeline_model_parallel_world_size() - 1):
            return False
    return get_pipeline_model_parallel_rank() == (
        get_pipeline_model_parallel_world_size() - 1
    )


def get_virtual_pipeline_model_parallel_rank():
    """Return the virtual pipeline-parallel rank."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    return _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK


def set_virtual_pipeline_model_parallel_rank(rank):
    """Set the virtual pipeline-parallel rank."""
    warnings.warn(
        "set_virtual_pipeline_model_parallel_rank in global scope is deprecated. "
        "Pass vp_stage explicitly to is_pipeline_first_stage, is_pipeline_last_stage, etc.",
        DeprecationWarning,
    )
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = rank


def set_expert_model_parallel_world_size(world_size):
    """Sets the expert-model-parallel world size."""
    global _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE
    _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE = world_size


def get_expert_model_parallel_rank():
    """Return caller's rank in the expert-model-parallel group."""
    if _MPU_EXPERT_MODEL_PARALLEL_RANK is not None:
        return _MPU_EXPERT_MODEL_PARALLEL_RANK
    if (
        paddle.distributed.is_initialized()
        and get_expert_model_parallel_group(False) is not None
    ):
        return get_expert_model_parallel_group().rank
    else:
        return 0


def set_expert_model_parallel_rank(rank):
    """Set expert-model-parallel rank."""
    global _MPU_EXPERT_MODEL_PARALLEL_RANK
    _MPU_EXPERT_MODEL_PARALLEL_RANK = rank


def get_virtual_pipeline_model_parallel_world_size():
    """Return the virtual pipeline-parallel world size."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    return _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE


def get_embedding_group(check_initialized=True):
    raise NotImplementedError("Not supported get_embedding_group yet.")
    '''
    """Get the embedding group the caller rank belongs to."""
    if check_initialized:
        assert _EMBEDDING_GROUP is not None, "embedding group is not initialized"
    return _EMBEDDING_GROUP
    '''


def get_expert_tensor_parallel_group(check_initialized=False):
    """Get the expert-tensor-parallel group the caller rank belongs to."""
    if check_initialized:
        assert _EXPERT_TENSOR_PARALLEL_GROUP is not None, (
            "Expert tensor parallel group is not initialized"
        )
    return _EXPERT_TENSOR_PARALLEL_GROUP


def get_expert_tensor_parallel_world_size():
    """Return world size for the expert tensor parallel group."""
    global _MPU_EXPERT_TENSOR_PARALLEL_WORLD_SIZE
    if _MPU_EXPERT_TENSOR_PARALLEL_WORLD_SIZE is not None:
        return _MPU_EXPERT_TENSOR_PARALLEL_WORLD_SIZE
    # Use tensor parallel group world size for backward compatibility otherwise
    if not _EXPERT_TENSOR_PARALLEL_GROUP:
        return _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    else:
        return get_expert_tensor_parallel_group().size()


def set_expert_tensor_parallel_world_size(world_size):
    "Set expert tensor model parallel size"
    global _MPU_EXPERT_TENSOR_PARALLEL_WORLD_SIZE
    _MPU_EXPERT_TENSOR_PARALLEL_WORLD_SIZE = world_size


def get_expert_tensor_parallel_rank():
    """Return my rank for the expert tensor parallel group."""
    global _MPU_EXPERT_TENSOR_PARALLEL_RANK
    if _MPU_EXPERT_TENSOR_PARALLEL_RANK is not None:
        return _MPU_EXPERT_TENSOR_PARALLEL_RANK
    if _EXPERT_TENSOR_PARALLEL_GROUP is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_RANK
    else:
        return 0


def set_expert_tensor_parallel_rank(rank):
    "Set expert tensor model parallel rank"
    global _MPU_EXPERT_TENSOR_PARALLEL_RANK
    _MPU_EXPERT_TENSOR_PARALLEL_RANK = rank


def get_expert_tensor_and_model_parallel_group(check_initialized=True):
    """Get the expert-tensor and expert-model group the caller rank belongs to."""
    if check_initialized:
        assert _EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP is not None, (
            "Expert tensor and model parallel group is not initialized"
        )
    return _EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP


def get_expert_tensor_and_model_parallel_world_size():
    """Return world size for the expert model parallel group times expert tensor parallel group."""
    if (
        paddle.distributed.is_available()
        and paddle.distributed.is_initialized()
    ):
        world_size = get_expert_tensor_and_model_parallel_group().size()
        return world_size
    else:
        return 0


def get_expert_tensor_and_model_parallel_rank():
    """Return caller's rank in the joint tensor- and expert-model-parallel group."""
    if (
        paddle.distributed.is_available()
        and paddle.distributed.is_initialized()
    ):
        return get_expert_tensor_and_model_parallel_group().rank()
    else:
        return 0


### End of expert-related functions region


def _set_global_memory_buffer():
    """Initialize global buffer."""
    global _GLOBAL_MEMORY_BUFFER
    assert _GLOBAL_MEMORY_BUFFER is None, (
        "global memory buffer is already initialized"
    )
    _GLOBAL_MEMORY_BUFFER = GlobalMemoryBuffer()


def have_global_memory_buffer():
    global _GLOBAL_MEMORY_BUFFER
    return _GLOBAL_MEMORY_BUFFER is not None


def get_global_memory_buffer():
    """Return the global GlobalMemoryBuffer object"""
    global _GLOBAL_MEMORY_BUFFER
    assert _GLOBAL_MEMORY_BUFFER is not None, (
        "global memory buffer is not initialized"
    )
    return _GLOBAL_MEMORY_BUFFER


def destroy_global_memory_buffer():
    """Sets the global memory buffer to None"""
    global _GLOBAL_MEMORY_BUFFER
    _GLOBAL_MEMORY_BUFFER = None
