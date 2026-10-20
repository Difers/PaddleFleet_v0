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

"""Triton kernels for streamed DSpark CE and distillation loss."""

from __future__ import annotations

from typing import Any

import paddle
import triton
import triton.language as tl

from paddlefleet.triton_ops.triton_compat import enable_compat_on_triton_kernel

# Vocab is streamed in BLOCK_SIZE-wide tiles; sized (like the Liger CE kernel)
# to stay register-resident with no local-memory spill for large V.
_BLOCK_SIZE_CAP = 2048
_ELEMENTS_PER_THREAD = 16


def _launch_config(n_cols: int) -> tuple[int, int]:
    block_size = min(triton.next_power_of_2(n_cols), _BLOCK_SIZE_CAP)
    num_warps = max(1, min(32, block_size // (32 * _ELEMENTS_PER_THREAD)))
    return block_size, num_warps


@triton.jit
def _seglu(
    x, r0, r1, r2, r3, t0, t1, t2, t3
):  # pragma: no cover - compiles to PTX
    """SegLU(x) = x + t0*relu(r0-x) + t1*relu(x-r1)
    + t2*relu(r2-x)^2 + t3*relu(x-r3)^2 (identity when t*=0)."""
    a0 = tl.maximum(r0 - x, 0.0)
    a1 = tl.maximum(x - r1, 0.0)
    a2 = tl.maximum(r2 - x, 0.0)
    a3 = tl.maximum(x - r3, 0.0)
    return x + t0 * a0 + t1 * a1 + t2 * a2 * a2 + t3 * a3 * a3


@triton.jit
def _seglu_dx(x, r0, r1, r2, r3, t0, t1, t2, t3):  # pragma: no cover - PTX
    """d SegLU / dx (used for the chain rule back to the pre-SegLU logits)."""
    a2 = tl.maximum(r2 - x, 0.0)
    a3 = tl.maximum(x - r3, 0.0)
    m0 = (r0 - x > 0.0).to(tl.float32)
    m1 = (x - r1 > 0.0).to(tl.float32)
    return 1.0 - t0 * m0 + t1 * m1 - 2.0 * t2 * a2 + 2.0 * t3 * a3


@enable_compat_on_triton_kernel
@triton.jit
def _distill_fwd_kernel(  # pragma: no cover - compiles to PTX
    D_ptr,
    D_stride,  # draft pre-SegLU logits [rows, V]
    T_ptr,
    T_stride,  # target pre-SegLU logits [rows, V]
    Y_ptr,
    Y_stride,  # target ids [rows]
    ce_ptr,
    ce_stride,  # out: per-token CE [rows]
    l1_ptr,
    l1_stride,  # out: per-token L1 [rows]
    n_cols,
    ignore_index,
    r0,
    r1,
    r2,
    r3,
    t0,
    t1,
    t2,
    t3,
    NEEDS_TEACHER: tl.constexpr,
    USE_SEGLU: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    y = tl.load(Y_ptr + row * Y_stride)
    Dr = D_ptr + row * D_stride
    Tr = T_ptr + row * T_stride

    if y == ignore_index:
        tl.store(ce_ptr + row * ce_stride, 0.0)
        tl.store(l1_ptr + row * l1_stride, 0.0)
        return

    d_y = tl.load(Dr + y).cast(tl.float32)
    if USE_SEGLU:
        d_y = _seglu(d_y, r0, r1, r2, r3, t0, t1, t2, t3)

    # Pass 1: online lse over SegLU(draft) and SegLU(target).
    md = float("-inf")
    dd = 0.0
    mt = float("-inf")
    dt = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        off = i + tl.arange(0, BLOCK_SIZE)
        mask = off < n_cols
        db = tl.load(Dr + off, mask=mask, other=float("-inf")).cast(tl.float32)
        if USE_SEGLU:
            db = tl.where(
                mask, _seglu(db, r0, r1, r2, r3, t0, t1, t2, t3), float("-inf")
            )
        bm = tl.max(db)
        mn = tl.maximum(md, bm)
        dd = dd * tl.exp(md - mn) + tl.sum(tl.exp(db - mn))
        md = mn
        if NEEDS_TEACHER:
            tb = tl.load(Tr + off, mask=mask, other=float("-inf")).cast(
                tl.float32
            )
            if USE_SEGLU:
                tb = tl.where(
                    mask,
                    _seglu(tb, r0, r1, r2, r3, t0, t1, t2, t3),
                    float("-inf"),
                )
            bmt = tl.max(tb)
            mnt = tl.maximum(mt, bmt)
            dt = dt * tl.exp(mt - mnt) + tl.sum(tl.exp(tb - mnt))
            mt = mnt

    lse_d = md + tl.log(dd)
    tl.store(ce_ptr + row * ce_stride, lse_d - d_y)

    # Pass 2: L1 = sum_v |softmax(draft)_v - softmax(target)_v|.
    if NEEDS_TEACHER:
        l1 = 0.0
        for i in range(0, n_cols, BLOCK_SIZE):
            off = i + tl.arange(0, BLOCK_SIZE)
            mask = off < n_cols
            db = tl.load(Dr + off, mask=mask, other=float("-inf")).cast(
                tl.float32
            )
            tb = tl.load(Tr + off, mask=mask, other=float("-inf")).cast(
                tl.float32
            )
            if USE_SEGLU:
                db = _seglu(db, r0, r1, r2, r3, t0, t1, t2, t3)
                tb = _seglu(tb, r0, r1, r2, r3, t0, t1, t2, t3)
            p = tl.exp(db - md) / dd
            q = tl.exp(tb - mt) / dt
            l1 += tl.sum(tl.where(mask, tl.abs(p - q), 0.0))
        tl.store(l1_ptr + row * l1_stride, l1)
    else:
        tl.store(l1_ptr + row * l1_stride, 0.0)


@enable_compat_on_triton_kernel
@triton.jit
def _distill_bwd_kernel(  # pragma: no cover - compiles to PTX
    D_ptr,
    D_stride,  # in: draft pre-SegLU logits; out: grad wrt pre-SegLU draft
    T_ptr,
    T_stride,  # target pre-SegLU logits [rows, V]
    Y_ptr,
    Y_stride,  # target ids [rows]
    GCE_ptr,
    GCE_stride,  # upstream grad wrt ce_per_token [rows]
    GL1_ptr,
    GL1_stride,  # upstream grad wrt l1_per_token [rows]
    gr_ptr,
    gt_ptr,  # out: multimax [4] grad accumulators (atomic)
    n_cols,
    ignore_index,
    r0,
    r1,
    r2,
    r3,
    t0,
    t1,
    t2,
    t3,
    NEEDS_TEACHER: tl.constexpr,
    USE_SEGLU: tl.constexpr,
    HAS_MULTIMAX_GRAD: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    y = tl.load(Y_ptr + row * Y_stride)
    Dr = D_ptr + row * D_stride
    Tr = T_ptr + row * T_stride

    if y == ignore_index:
        for i in range(0, n_cols, BLOCK_SIZE):
            off = i + tl.arange(0, BLOCK_SIZE)
            tl.store(Dr + off, 0.0, mask=off < n_cols)
        return

    gce = tl.load(GCE_ptr + row * GCE_stride).cast(tl.float32)
    gl1 = tl.load(GL1_ptr + row * GL1_stride).cast(tl.float32)

    # Pass 1: online lse for draft and target.
    md = float("-inf")
    dd = 0.0
    mt = float("-inf")
    dt = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        off = i + tl.arange(0, BLOCK_SIZE)
        mask = off < n_cols
        db = tl.load(Dr + off, mask=mask, other=float("-inf")).cast(tl.float32)
        if USE_SEGLU:
            db = tl.where(
                mask, _seglu(db, r0, r1, r2, r3, t0, t1, t2, t3), float("-inf")
            )
        bm = tl.max(db)
        mn = tl.maximum(md, bm)
        dd = dd * tl.exp(md - mn) + tl.sum(tl.exp(db - mn))
        md = mn
        if NEEDS_TEACHER:
            tb = tl.load(Tr + off, mask=mask, other=float("-inf")).cast(
                tl.float32
            )
            if USE_SEGLU:
                tb = tl.where(
                    mask,
                    _seglu(tb, r0, r1, r2, r3, t0, t1, t2, t3),
                    float("-inf"),
                )
            bmt = tl.max(tb)
            mnt = tl.maximum(mt, bmt)
            dt = dt * tl.exp(mt - mnt) + tl.sum(tl.exp(tb - mnt))
            mt = mnt

    # Pass 2: spa = <sign(p - q), p> (per-token scalar for the L1 grad).
    spa = 0.0
    if NEEDS_TEACHER:
        for i in range(0, n_cols, BLOCK_SIZE):
            off = i + tl.arange(0, BLOCK_SIZE)
            mask = off < n_cols
            db = tl.load(Dr + off, mask=mask, other=float("-inf")).cast(
                tl.float32
            )
            tb = tl.load(Tr + off, mask=mask, other=float("-inf")).cast(
                tl.float32
            )
            if USE_SEGLU:
                db = _seglu(db, r0, r1, r2, r3, t0, t1, t2, t3)
                tb = _seglu(tb, r0, r1, r2, r3, t0, t1, t2, t3)
            p = tl.exp(db - md) / dd
            q = tl.exp(tb - mt) / dt
            s = tl.where(p - q > 0.0, 1.0, tl.where(p - q < 0.0, -1.0, 0.0))
            spa += tl.sum(tl.where(mask, s * p, 0.0))

    # Pass 3: grad wrt pre-SegLU draft logits + SegLU/multimax param grads.
    gr0 = 0.0
    gr1 = 0.0
    gr2 = 0.0
    gr3 = 0.0
    gt0 = 0.0
    gt1 = 0.0
    gt2 = 0.0
    gt3 = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        off = i + tl.arange(0, BLOCK_SIZE)
        mask = off < n_cols
        dx = tl.load(Dr + off, mask=mask, other=0.0).cast(
            tl.float32
        )  # pre-SegLU
        db = _seglu(dx, r0, r1, r2, r3, t0, t1, t2, t3) if USE_SEGLU else dx
        p = tl.exp(db - md) / dd
        # CE part: gce * (p - onehot_y)
        grad_dl = gce * tl.where(off != y, p, p - 1.0)
        if NEEDS_TEACHER:
            tb0 = tl.load(Tr + off, mask=mask, other=float("-inf")).cast(
                tl.float32
            )
            tb = (
                _seglu(tb0, r0, r1, r2, r3, t0, t1, t2, t3)
                if USE_SEGLU
                else tb0
            )
            q = tl.exp(tb - mt) / dt
            s = tl.where(p - q > 0.0, 1.0, tl.where(p - q < 0.0, -1.0, 0.0))
            grad_dl += gl1 * p * (s - spa)
        grad_dl = tl.where(mask, grad_dl, 0.0)

        if USE_SEGLU:
            grad_x = grad_dl * _seglu_dx(dx, r0, r1, r2, r3, t0, t1, t2, t3)
            if HAS_MULTIMAX_GRAD:
                a0 = tl.maximum(r0 - dx, 0.0)
                a1 = tl.maximum(dx - r1, 0.0)
                a2 = tl.maximum(r2 - dx, 0.0)
                a3 = tl.maximum(dx - r3, 0.0)
                m0 = (a0 > 0.0).to(tl.float32)
                m1 = (a1 > 0.0).to(tl.float32)
                gt0 += tl.sum(grad_dl * a0)
                gt1 += tl.sum(grad_dl * a1)
                gt2 += tl.sum(grad_dl * a2 * a2)
                gt3 += tl.sum(grad_dl * a3 * a3)
                gr0 += t0 * tl.sum(grad_dl * m0)
                gr1 += -t1 * tl.sum(grad_dl * m1)
                gr2 += 2.0 * t2 * tl.sum(grad_dl * a2)
                gr3 += -2.0 * t3 * tl.sum(grad_dl * a3)
        else:
            grad_x = grad_dl
        tl.store(Dr + off, grad_x, mask=mask)

    if USE_SEGLU and HAS_MULTIMAX_GRAD:
        tl.atomic_add(gr_ptr + 0, gr0)
        tl.atomic_add(gr_ptr + 1, gr1)
        tl.atomic_add(gr_ptr + 2, gr2)
        tl.atomic_add(gr_ptr + 3, gr3)
        tl.atomic_add(gt_ptr + 0, gt0)
        tl.atomic_add(gt_ptr + 1, gt1)
        tl.atomic_add(gt_ptr + 2, gt2)
        tl.atomic_add(gt_ptr + 3, gt3)


def _project(hidden: Any, weight: Any, transpose_y: bool) -> Any:
    """logits = hidden @ weight (transpose_y=False, [H,V]) or hidden @ weight^T
    (transpose_y=True, [V,H]).

    ``transpose_y`` is taken verbatim from ReadOnlyProjection rather than
    inferred from shapes: a square head would make shape inference ambiguous,
    and the streamed eager path (losses.py) also treats transpose_y as explicit.
    """
    return paddle.matmul(hidden, weight, transpose_y=transpose_y)


def _draft_logits(draft, previous, weight, w1, w2, markov_enabled, transpose_y):
    logits = _project(draft, weight, transpose_y)
    if markov_enabled:
        prev_emb = paddle.nn.functional.embedding(previous, w1)
        logits = logits + paddle.matmul(prev_emb, w2)
    return logits


def _seglu_vals(seglu_r, seglu_t):
    r = (
        [float(v) for v in seglu_r]
        if seglu_r is not None
        else [0.0, 0.0, 0.0, 0.0]
    )
    t = (
        [float(v) for v in seglu_t]
        if seglu_t is not None
        else [0.0, 0.0, 0.0, 0.0]
    )
    return r, t


def make_fused_ce_l1(
    *,
    needs_teacher: bool,
    markov_enabled: bool,
    use_seglu: bool,
    seglu_r,
    seglu_t,
    transpose_y: bool,
    ignore_index: int = -100,
):
    """Build a PyLayer computing per-token (ce, l1) with the fused kernels.

    SegLU params are captured as read-only constants (frozen lm-head), matching
    the streamed eager path which never propagates gradient to them.
    ``transpose_y`` matches ReadOnlyProjection: True means a [V, H] head, False
    a [H, V] head.
    """
    r, t = _seglu_vals(seglu_r, seglu_t)

    class _FusedCEL1(paddle.autograd.PyLayer):
        @staticmethod
        def forward(ctx, draft, target_hidden, ids, previous, weight, w1, w2):
            """Forward pass."""
            ctx.save_for_backward(
                draft, target_hidden, ids, previous, weight, w1, w2
            )
            with paddle.no_grad():
                D = _draft_logits(
                    draft, previous, weight, w1, w2, markov_enabled, transpose_y
                )
                T = (
                    _project(target_hidden, weight, transpose_y)
                    if needs_teacher
                    else paddle.empty([1, 1], dtype=D.dtype)
                )
                rows, V = int(D.shape[0]), int(D.shape[1])
                ce = paddle.zeros([rows], dtype=paddle.float32)
                l1 = paddle.zeros([rows], dtype=paddle.float32)
                D = D.contiguous()
                T = T.contiguous()
                ids32 = ids.astype("int64").contiguous()
                bs, nw = _launch_config(V)
                _distill_fwd_kernel[(rows,)](
                    D,
                    D.strides[0],
                    T,
                    T.strides[0] if needs_teacher else 0,
                    ids32,
                    1,
                    ce,
                    1,
                    l1,
                    1,
                    V,
                    ignore_index,
                    r[0],
                    r[1],
                    r[2],
                    r[3],
                    t[0],
                    t[1],
                    t[2],
                    t[3],
                    NEEDS_TEACHER=needs_teacher,
                    USE_SEGLU=use_seglu,
                    BLOCK_SIZE=bs,
                    num_warps=nw,
                )
            zeros = paddle.zeros([rows], dtype=paddle.float32)
            return ce, l1, zeros, zeros, zeros

        @staticmethod
        def backward(ctx, grad_ce, grad_l1, grad_sel, grad_cov, grad_cor):
            """Backward pass."""
            del grad_sel, grad_cov, grad_cor
            draft, target_hidden, ids, previous, weight, w1, w2 = (
                ctx.saved_tensor()
            )
            D = _draft_logits(
                draft, previous, weight, w1, w2, markov_enabled, transpose_y
            )
            T = (
                _project(target_hidden, weight, transpose_y)
                if needs_teacher
                else paddle.empty([1, 1], dtype=D.dtype)
            )
            rows, V = int(D.shape[0]), int(D.shape[1])
            gce = (
                paddle.zeros([rows], dtype=paddle.float32)
                if grad_ce is None
                else grad_ce.astype("float32")
            )
            gl1 = (
                paddle.zeros([rows], dtype=paddle.float32)
                if grad_l1 is None
                else grad_l1.astype("float32")
            )
            D = D.contiguous()
            T = T.contiguous()
            gce = gce.contiguous()
            gl1 = gl1.contiguous()
            ids32 = ids.astype("int64").contiguous()
            gr = paddle.zeros([4], dtype=paddle.float32)
            gt = paddle.zeros([4], dtype=paddle.float32)
            bs, nw = _launch_config(V)
            _distill_bwd_kernel[(rows,)](
                D,
                D.strides[0],
                T,
                T.strides[0] if needs_teacher else 0,
                ids32,
                1,
                gce,
                1,
                gl1,
                1,
                gr,
                gt,
                V,
                ignore_index,
                r[0],
                r[1],
                r[2],
                r[3],
                t[0],
                t[1],
                t[2],
                t[3],
                NEEDS_TEACHER=needs_teacher,
                USE_SEGLU=use_seglu,
                HAS_MULTIMAX_GRAD=False,
                BLOCK_SIZE=bs,
                num_warps=nw,
            )
            # D now holds grad wrt the (pre-SegLU) combined draft logits [rows, V].
            grad_logits = D
            if transpose_y:  # weight [V, H]: D = draft @ weight^T
                grad_draft = paddle.matmul(grad_logits, weight)
            else:  # weight [H, V]: D = draft @ weight
                grad_draft = paddle.matmul(
                    grad_logits, weight, transpose_y=True
                )
            grad_draft = grad_draft.astype(draft.dtype)

            grad_w1 = grad_w2 = None
            if markov_enabled:
                prev_emb = paddle.nn.functional.embedding(previous, w1)
                if not w2.stop_gradient:
                    grad_w2 = paddle.matmul(
                        prev_emb, grad_logits, transpose_x=True
                    ).astype(w2.dtype)
                if not w1.stop_gradient:
                    grad_prev = paddle.matmul(grad_logits, w2, transpose_y=True)
                    grad_w1 = paddle.zeros(w1.shape, dtype="float32")
                    grad_w1 = paddle.scatter(
                        grad_w1,
                        previous.astype("int64"),
                        grad_prev.astype("float32"),
                        overwrite=False,
                    ).astype(w1.dtype)
            return grad_draft, None, None, None, None, grad_w1, grad_w2

    return _FusedCEL1
