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

"""Fused Triton tokens-per-expert histogram.

One atomic-histogram kernel, the fused counterpart of the eager
``token_dispatcher._tokens_per_expert_histogram``. Padding (``-1``) and ids
outside ``[0, num_experts)`` are skipped. For in-range ids the result equals
``bincount(indices[indices >= 0], minlength=num_experts)`` exactly, since
integer atomic adds are order-independent. Not differentiable.
"""

import paddle
from paddle import Tensor

from .utils import is_torch_compat_available

if is_torch_compat_available():
    paddle.enable_compat(scope={"triton"})

import triton
import triton.language as tl


@triton.jit
def tokens_per_expert_hist_kernel(
    Idx_ptr,  # [n_elems] flattened routing ids
    Counts_ptr,  # [num_experts] int32, zero-initialized
    n_elems,
    num_experts,
    BLOCK: tl.constexpr,
):
    """Atomically count each valid id of one ``BLOCK`` chunk.

    Ids outside ``[0, num_experts)`` are masked out; their address is clamped
    to 0 so it stays in bounds.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elems
    expert_id = tl.load(Idx_ptr + offs, mask=mask, other=-1).to(tl.int32)
    valid = mask & (expert_id >= 0) & (expert_id < num_experts)
    tl.atomic_add(Counts_ptr + tl.where(valid, expert_id, 0), 1, mask=valid)


def tokens_per_expert_histogram_triton(
    indices: Tensor, num_experts: int
) -> Tensor:
    """Fused equivalent of ``_tokens_per_expert_histogram``.

    Args:
        indices: ``[..., K]`` int32/int64 routing table; ``-1`` marks padding.
        num_experts: number of experts.

    Returns:
        int32 ``[num_experts]`` histogram, identical to the eager path for
        in-range ids.
    """
    # The contiguous buffer is indexed linearly, which avoids a reshape launch.
    flat = indices.contiguous()
    counts = paddle.zeros([num_experts], dtype="int32")
    num_elems = 1
    for dim in flat.shape:
        num_elems *= int(dim)
    if num_elems == 0:
        return counts
    BLOCK = 1024
    grid = (triton.cdiv(num_elems, BLOCK),)
    tokens_per_expert_hist_kernel[grid](
        flat,
        counts,
        num_elems,
        int(num_experts),
        BLOCK=BLOCK,
        num_warps=4,
    )
    return counts
