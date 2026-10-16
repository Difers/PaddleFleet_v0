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

"""Fused Triton cast and mask of the RingMoE routing table.

``RingMoETokenDispatcher.ring_forward`` needs the int32 routing indices and the
router weights cast to the probs dtype with dropped lanes (``idx < 0``) zeroed.
Done eagerly this is a chain of small elementwise kernels on the dispatch
critical path; here it is one forward and one backward kernel.

Dropped lanes select a literal 0 (unlike ``w * mask``, which turns ``Inf * 0``
into NaN), and the mask is applied as a plain select in both directions, so no
traced ``where`` grad node enters the graph.
"""

import paddle
from paddle import Tensor

from .utils import is_torch_compat_available

if is_torch_compat_available():
    paddle.enable_compat(scope={"triton"})

import triton
import triton.language as tl


@triton.jit
def _fused_fwd_kernel(
    Idx_ptr, W_ptr, OutIdx_ptr, OutW_ptr, n_elems, BLOCK: tl.constexpr
):
    """Write the cast routing indices and the masked, cast weights."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elems
    idx = tl.load(Idx_ptr + offs, mask=mask, other=-1)
    weight = tl.load(W_ptr + offs, mask=mask, other=0).to(tl.float32)
    keep = idx >= 0
    tl.store(OutIdx_ptr + offs, idx.to(OutIdx_ptr.dtype.element_ty), mask=mask)
    tl.store(
        OutW_ptr + offs,
        tl.where(keep, weight, 0.0).to(OutW_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def _mask_cast_bwd_kernel(G_ptr, Idx_ptr, GW_ptr, n_elems, BLOCK: tl.constexpr):
    """Zero the weight gradient on dropped lanes and cast it to GW's dtype."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elems
    grad = tl.load(G_ptr + offs, mask=mask, other=0).to(tl.float32)
    idx = tl.load(Idx_ptr + offs, mask=mask, other=-1)
    keep = idx >= 0
    grad_w = tl.where(keep, grad, 0.0)
    tl.store(GW_ptr + offs, grad_w.to(GW_ptr.dtype.element_ty), mask=mask)


def _numel(shape):
    numel = 1
    for dim in shape:
        numel *= int(dim)
    return numel


def _launch(fn, src, idx, out, n):
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    fn[grid](src, idx, out, n, BLOCK=BLOCK, num_warps=4)


class _MaskCastRoutingW(paddle.autograd.PyLayer):
    """PyLayer behind :func:`mask_cast_routing_w`.

    Buffers are indexed linearly over ``numel``, so no reshape is needed.
    """

    @staticmethod
    def forward(ctx, topk_indices, topk_weights, out_dtype):
        idx_in = topk_indices.contiguous()
        weights = topk_weights.contiguous()
        num_elems = _numel(weights.shape)
        cur_idx = paddle.empty(weights.shape, dtype="int32")
        cur_w = paddle.empty(weights.shape, dtype=out_dtype)
        if num_elems > 0:
            BLOCK = 1024
            grid = (triton.cdiv(num_elems, BLOCK),)
            _fused_fwd_kernel[grid](
                idx_in,
                weights,
                cur_idx,
                cur_w,
                num_elems,
                BLOCK=BLOCK,
                num_warps=4,
            )
        ctx.save_for_backward(cur_idx)
        ctx.in_dtype = topk_weights.dtype
        cur_idx.stop_gradient = True
        return cur_idx, cur_w

    @staticmethod
    def backward(ctx, g_idx, g_w):
        # Only cur_w is differentiable; g_idx belongs to the int32 indices.
        if g_w is None:
            return None, None
        (cur_idx,) = ctx.saved_tensor()
        grad = g_w.contiguous()
        num_elems = _numel(grad.shape)
        grad_weights = paddle.empty(grad.shape, dtype=ctx.in_dtype)
        if num_elems > 0:
            _launch(
                _mask_cast_bwd_kernel, grad, cur_idx, grad_weights, num_elems
            )
        return None, grad_weights


def mask_cast_routing_w(topk_indices: Tensor, topk_weights: Tensor, out_dtype):
    """Return ``(cur_idx, cur_w)`` built from the raw gate outputs.

    ``cur_idx`` is ``topk_indices`` as int32 (stop-gradient). ``cur_w`` is
    ``topk_weights`` cast to ``out_dtype`` with dropped lanes (``idx < 0``) set
    to 0. Gradients flow to ``topk_weights`` only, zeroed on dropped lanes.
    """
    return _MaskCastRoutingW.apply(topk_indices, topk_weights, out_dtype)
