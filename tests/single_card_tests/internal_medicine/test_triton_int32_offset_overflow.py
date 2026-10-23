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

"""Regression tests for the int32 offset overflow in PaddleFleet Triton kernels.

Guards the int64-promotion fix in:
  * ``triton_ops/balanced_reorder.py`` — ``balanced_gather_reorder_kernel`` /
    ``balanced_scatter_reorder_kernel`` (out-of-bounds WRITE; exact check).
  * ``internal_medicine/core/triton_qk_kernel.py`` — ``qk_stats_partial_kernel``
    head-base term (out-of-bounds READ; per-head stats check).

Each case addresses a tensor spanning >= 2**31 elements along the indexed dims, so
the pre-fix int32 offsets wrap. Single GPU, no communication. These are large
(tens of GB) and slow ON PURPOSE: the bug only exists past 2**31, so the shapes
cannot be shrunk below it.
"""

import importlib
import unittest

try:
    import numpy as np
    import paddle

    triton = importlib.import_module("triton")
except Exception as exc:  # pragma: no cover - environment guard
    raise unittest.SkipTest(f"paddle or triton unavailable: {exc}") from exc

if not paddle.device.is_compiled_with_cuda():
    raise unittest.SkipTest("CUDA required for Triton kernel tests")

from paddlefleet.internal_medicine.backends.paddlefleet.qk_monitor import (
    _compute_qk_stats_triton,
)
from paddlefleet.triton_ops.balanced_reorder import (
    balanced_gather_reorder_kernel,
    balanced_scatter_reorder_kernel,
)

INT32_MAX = 2**31 - 1
_TOTAL_GIB = paddle.device.cuda.get_device_properties(0).total_memory / 2**30


def _carve(buf, start, shape):
    """1-D view of ``buf`` reshaped to ``shape``; asserts it aliases ``buf``."""
    n = 1
    for s in shape:
        n *= s
    view = buf[start : start + n].reshape(shape)
    assert view.data_ptr() == buf.data_ptr() + start * buf.element_size(), (
        "slice is not a view"
    )
    return view, start + n


def _fill_tagged(t, chunk, tags):
    """Set only the three channels sampled by ``_count_wrong_rows``: ``[h-1,d-1]``
    and ``[h//2,d//2]`` get the row tag, ``[0,0]`` the within-chunk index. The
    kernel moves the whole row, so tagging these channels is enough — and it avoids
    materialising a full ``broadcast_to`` temporary the size of the buffer."""
    h, d = t.shape[-2], t.shape[-1]
    rows = t.shape[-3]
    tag_t = paddle.to_tensor(tags.astype("float32")).astype(t.dtype)
    local = np.broadcast_to(
        np.array([(g % chunk) % 251 for g in range(rows)], dtype="float32"),
        tags.shape,
    )
    local_t = paddle.to_tensor(np.ascontiguousarray(local)).astype(t.dtype)
    t[..., h - 1, d - 1] = tag_t
    t[..., h // 2, d // 2] = tag_t
    t[..., 0, 0] = local_t


def _count_overwritten(t, sentinel=7.0, step=1 << 27):
    """Count elements of ``t`` that differ from ``sentinel``, scanning in chunks so
    the transient (bf16->fp32 + bool) stays bounded instead of materialising over
    the whole 2**31-element guard/victim buffer."""
    flat = t.reshape([t.numel()])
    total = 0
    for i in range(0, flat.shape[0], step):
        total += int((flat[i : i + step] != sentinel).sum())
    return total


def _count_wrong_rows(t, axis_len, chunk, exp_tag_of_row):
    """Compare three sampled channels of every row with the expected tag/index."""
    h, d = t.shape[-2], t.shape[-1]
    flat = t.reshape([-1, axis_len, h, d])
    bad = 0
    for b in range(flat.shape[0]):
        loc = flat[b, :, 0, 0].astype("float32").numpy()
        tag_a = flat[b, :, h - 1, d - 1].astype("float32").numpy()
        tag_b = flat[b, :, h // 2, d // 2].astype("float32").numpy()
        exp_tag = np.array(
            [exp_tag_of_row(b, g) for g in range(axis_len)], dtype="float32"
        )
        exp_loc = np.array(
            [(g % chunk) % 251 for g in range(axis_len)], dtype="float32"
        )
        bad += int(
            ((loc != exp_loc) | (tag_a != exp_tag) | (tag_b != exp_tag)).sum()
        )
    return bad


@unittest.skipUnless(
    _TOTAL_GIB >= 40,
    "reorder int32-overflow test needs >= 40 GiB GPU "
    "(peak ~18 GB, and ci/single_card_test.sh runs 4 files per card)",
)
class ReorderInt32OverflowTest(unittest.TestCase):
    """balanced_{gather,scatter}_reorder write offsets must stay int64 so a buffer
    with >= 2**31 elements does not wrap and silently clobber neighbour tensors."""

    CP = 4
    HEADS = 64
    DIM = 576  # inner_size = 64*576 = 36864 (MLA-absorbed query geometry)
    S_LOCAL = (
        16384  # gathered S = 2*CP*(S_LOCAL//2) = 65536; max offset ~2.42e9
    )

    def _run_reorder(self, kernel):
        inner = self.HEADS * self.DIM
        chunk = self.S_LOCAL // 2
        N = self.CP
        BLOCK = 1024
        nb = triton.cdiv(chunk * inner, BLOCK)
        grid = (nb * 2 * N, 1, 1)  # outer_size = 1 (batch 1, axis 1)
        s_global = 2 * N * chunk
        self.assertGreater(
            s_global * inner, INT32_MAX, "shape must overflow int32"
        )

        if kernel == "gather":
            src_shape = [N, self.S_LOCAL, self.HEADS, self.DIM]
            dst_shape = [1, s_global, self.HEADS, self.DIM]
        else:
            src_shape = [1, s_global, self.HEADS, self.DIM]
            dst_shape = [N, 1, self.S_LOCAL, self.HEADS, self.DIM]
        n_src = n_dst = s_global * inner

        # Pooled layout [guard | src | victim | dst | tail]: wrapped writes land
        # in the neighbouring guard/victim/tail, which we assert stays untouched.
        guard_n = victim_n = 2**31 + 2**20
        tail_n = 2**20
        buf = paddle.empty(
            [guard_n + n_src + victim_n + n_dst + tail_n], "bfloat16"
        )
        pos = 0
        guard, pos = _carve(buf, pos, [guard_n])
        src, pos = _carve(buf, pos, src_shape)
        victim, pos = _carve(buf, pos, [victim_n])
        dst, pos = _carve(buf, pos, dst_shape)
        tail, pos = _carve(buf, pos, [tail_n])
        guard.fill_(7.0)
        victim.fill_(7.0)
        tail.fill_(7.0)
        dst.fill_(0.0)

        if kernel == "gather":
            tags = np.array(
                [
                    [
                        (r if g < chunk else 2 * N - 1 - r) + 1
                        for g in range(self.S_LOCAL)
                    ]
                    for r in range(N)
                ]
            )
            _fill_tagged(src, chunk, tags)
            paddle.device.synchronize()
            balanced_gather_reorder_kernel[grid](
                src,
                dst,
                N,
                chunk,
                inner,
                self.S_LOCAL * inner,
                self.S_LOCAL * inner,
                nb,
                BLOCK_SIZE=BLOCK,
            )
            paddle.device.synchronize()
            bad = _count_wrong_rows(
                dst, s_global, chunk, lambda b, g: g // chunk + 1
            )
        else:
            tags = np.array([[g // chunk + 1 for g in range(s_global)]])
            _fill_tagged(src, chunk, tags)
            paddle.device.synchronize()
            balanced_scatter_reorder_kernel[grid](
                src,
                dst,
                N,
                chunk,
                inner,
                s_global * inner,
                self.S_LOCAL * inner,
                self.S_LOCAL * inner,
                nb,
                BLOCK_SIZE=BLOCK,
            )
            paddle.device.synchronize()
            bad = _count_wrong_rows(
                dst,
                self.S_LOCAL,
                chunk,
                lambda b, g: (b if g < chunk else 2 * N - 1 - b) + 1,
            )
        neighbours = (
            _count_overwritten(guard)
            + _count_overwritten(victim)
            + _count_overwritten(tail)
        )
        return bad, neighbours

    def test_gather_no_overflow(self):
        bad, neighbours = self._run_reorder("gather")
        self.assertEqual(bad, 0, "gather produced wrong output rows")
        self.assertEqual(
            neighbours, 0, "gather overwrote neighbour tensors (OOB write)"
        )

    def test_scatter_no_overflow(self):
        bad, neighbours = self._run_reorder("scatter")
        self.assertEqual(bad, 0, "scatter produced wrong output rows")
        self.assertEqual(
            neighbours, 0, "scatter overwrote neighbour tensors (OOB write)"
        )


@unittest.skipUnless(
    _TOTAL_GIB >= 24,
    "qk head-overflow test needs >= 24 GiB GPU (peak ~13 GB)",
)
class QkStatsPartialHeadOverflowTest(unittest.TestCase):
    """qk_stats_partial_kernel's ``head_idx * stride_q_head`` must stay int64 so a
    contiguous [1, H, S, D] Q whose per-head stride * (H-1) exceeds 2**31 reads the
    right head. Reference: the same kernel on one head at a time (stride term == 0,
    never overflows)."""

    HEADS = 64
    S = 8192
    DIM = 8192  # stride_q_head = S*DIM = 6.7e7; (H-1)*stride ~4.2e9 > 2**31
    CHECK_HEADS = (0, 31, 32, 40, 63)

    def test_head_base_no_overflow(self):
        stride_h = self.S * self.DIM
        self.assertGreater(
            (self.HEADS - 1) * stride_h, INT32_MAX, "shape must overflow int32"
        )
        shape = [1, self.HEADS, self.S, self.DIM]
        # Pooled [poison | q]: a pre-fix negative-wrapped read lands in poison
        # (finite 100.0) -> deterministic wrong stats instead of a CUDA fault.
        poison_n = 2**31 + 2**20
        buf = paddle.empty(
            [poison_n + self.HEADS * self.S * self.DIM], "bfloat16"
        )
        poison, pos = _carve(buf, 0, [poison_n])
        q, _ = _carve(buf, pos, shape)
        poison.fill_(100.0)
        # Fill per head in place (value depends only on head) to avoid a full
        # broadcast_to temporary the size of q.
        for head in range(self.HEADS):
            q[:, head] = 0.001 * (head % 5 + 1)
        k = paddle.full([1, 1, self.S, self.DIM], 0.001, dtype="bfloat16")
        paddle.device.synchronize()

        got = (
            _compute_qk_stats_triton(
                q,
                k,
                causal=True,
                heads_per_group=self.HEADS,
                row_stride=1,
                q_row_offset=0,
            )["max_per_head"]
            .astype("float32")
            .numpy()
            .ravel()
        )
        paddle.device.synchronize()

        for h in self.CHECK_HEADS:
            ref = (
                _compute_qk_stats_triton(
                    q[:, h : h + 1].contiguous(),
                    k,
                    causal=True,
                    heads_per_group=1,
                    row_stride=1,
                    q_row_offset=0,
                )["max_per_head"]
                .astype("float32")
                .numpy()
                .ravel()[0]
            )
            self.assertTrue(
                np.isfinite(got[h])
                and abs(got[h] - ref) <= 1e-3 * abs(ref) + 1e-6,
                f"head {h}: batched={got[h]:.6g} single-head ref={ref:.6g} "
                "(int32 head-stride overflow?)",
            )


def _reorder_plain_check(kernel, cp, s_local):
    """Run gather/scatter with inner_size==1 (and chunk_size==1 when s_local==2)
    on tiny plain buffers; return the number of mismatched elements vs a NumPy
    recompute of the reorder mapping. Guards the value==1 -> constexpr case: with
    the pre-fix ``.to`` the kernel fails to compile here."""
    inner, chunk, N, BLOCK = 1, s_local // 2, cp, 1024
    nb = triton.cdiv(chunk * inner, BLOCK)
    grid = (nb * 2 * N, 1, 1)
    s_global = 2 * N * chunk
    if kernel == "gather":
        ids = np.arange(N * s_local, dtype="float32").reshape(N, s_local)
        src = paddle.zeros([N, s_local, 1, 1], dtype="float32")
        src[:, :, 0, 0] = paddle.to_tensor(ids)
        dst = paddle.zeros([1, s_global, 1, 1], dtype="float32")
        paddle.device.synchronize()
        balanced_gather_reorder_kernel[grid](
            src, dst, N, chunk, inner, s_local, s_local, nb, BLOCK_SIZE=BLOCK
        )
        paddle.device.synchronize()
        got = dst[0, :, 0, 0].numpy()
        exp = np.empty(s_global, dtype="float32")
        for g in range(s_global):
            ci, al = g // chunk, g % chunk
            r, sa = (ci, al) if ci < N else (2 * N - 1 - ci, chunk + al)
            exp[g] = ids[r, sa]
        return int((got != exp).sum())
    ids = np.arange(s_global, dtype="float32")
    src = paddle.zeros([1, s_global, 1, 1], dtype="float32")
    src[0, :, 0, 0] = paddle.to_tensor(ids)
    dst = paddle.zeros([N, 1, s_local, 1, 1], dtype="float32")
    paddle.device.synchronize()
    balanced_scatter_reorder_kernel[grid](
        src,
        dst,
        N,
        chunk,
        inner,
        s_global,
        s_local,
        s_local,
        nb,
        BLOCK_SIZE=BLOCK,
    )
    paddle.device.synchronize()
    got = dst[:, 0, :, 0, 0].numpy()
    exp = np.empty((N, s_local), dtype="float32")
    for r in range(N):
        for a in range(s_local):
            p = (
                r * chunk + a
                if a < chunk
                else (2 * N - 1 - r) * chunk + (a - chunk)
            )
            exp[r, a] = ids[p]
    return int((got != exp).sum())


class ReorderParamSpecializationTest(unittest.TestCase):
    """value==1 ints are specialized to constexpr by Triton; the int64 promotion
    must use tl.cast (constexpr has no .to). inner_size==1 occurs for 2-D [B,S]
    gathers (the default dual-chunk CP path); chunk_size==1 for a 2-element axis.
    Tiny + plain buffers, so this runs even on small GPUs (no memory gate)."""

    CP = 4

    def test_inner_size_1(self):
        self.assertEqual(_reorder_plain_check("gather", self.CP, 8), 0)
        self.assertEqual(_reorder_plain_check("scatter", self.CP, 8), 0)

    def test_chunk_size_1(self):
        self.assertEqual(_reorder_plain_check("gather", self.CP, 2), 0)
        self.assertEqual(_reorder_plain_check("scatter", self.CP, 2), 0)


if __name__ == "__main__":
    unittest.main()
