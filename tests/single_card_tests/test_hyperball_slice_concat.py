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
"""``hyperball_apply`` with a ``split_concat_func`` (fused QKV / gate_up / heads).

A fused weight packs several logically independent matrices into one tensor, so
the Hyperball projection must run on each slice's own sphere. These tests check
that the sliced path reproduces slice-by-slice calls exactly, that every slice
keeps its own Frobenius norm, and that it really differs from projecting the
fused matrix as a whole (which would only hold the total norm).

Run:  python -m pytest tests/single_card_tests/test_hyperball_slice_concat.py -v
"""

import functools
import unittest

import paddle

from paddlefleet.optimizers.hyperball import hyperball_apply
from paddlefleet.transformer.muon_utils import (
    ortho_gate_up,
    ortho_per_head,
    ortho_stacked,
)

LR = 0.3


class TestHyperballSliceConcat(unittest.TestCase):
    def setUp(self):
        paddle.seed(2026)

    def _check(self, split_concat_func, w, u, slice_sizes, transposed=False):
        got = hyperball_apply(w, u, LR, split_concat_func=split_concat_func)

        # Reference: hyperball each (w, u) slice pair on its own, then let
        # split_concat_func put the pieces back in the fused layout.
        w_blocks = paddle.split(w, slice_sizes, axis=-1)
        u_blocks = paddle.split(u, slice_sizes, axis=-1)
        if transposed:
            perm = list(range(w.ndim))
            perm[-2], perm[-1] = perm[-1], perm[-2]
            w_blocks = [paddle.transpose(b, perm) for b in w_blocks]
            u_blocks = [paddle.transpose(b, perm) for b in u_blocks]
        parts = [
            hyperball_apply(wb, ub, LR) for wb, ub in zip(w_blocks, u_blocks)
        ]
        ref = split_concat_func(u, lambda _blk: parts.pop(0))
        self.assertEqual(parts, [])
        self.assertEqual(
            float(paddle.max(paddle.abs(got - ref))),
            0.0,
            "sliced hyperball must match per-slice calls bit-for-bit",
        )

        # Every slice stays on its own sphere R_s = ||W_s||_F.
        before, after = [], []
        split_concat_func(
            w, lambda s: before.append(paddle.norm(s.astype("float32"))) or s
        )
        split_concat_func(got, lambda s: after.append(paddle.norm(s)) or s)
        self.assertEqual(len(before), len(slice_sizes))
        for b, a in zip(before, after):
            # fp32 Frobenius round-off
            self.assertAlmostEqual(float(a) / float(b), 1.0, delta=1e-5)

        # Projecting the fused matrix as a whole is a different update.
        whole = hyperball_apply(w, u, LR)
        self.assertGreater(float(paddle.max(paddle.abs(got - whole))), 1e-4)

    def test_gate_up_2d(self):
        w = paddle.randn([128, 256])
        u = paddle.randn([128, 256])
        self._check(ortho_gate_up, w, u, [128, 128])

    def test_gate_up_3d_experts(self):
        w = paddle.randn([4, 64, 128])
        u = paddle.randn([4, 64, 128])
        self._check(ortho_gate_up, w, u, [64, 64])

    def test_per_head_transposed(self):
        split_concat_func = functools.partial(
            ortho_per_head, heads=4, head_sizes=[16, 8], transposed=True
        )
        w = paddle.randn([64, 96])
        u = paddle.randn([64, 96])
        self._check(split_concat_func, w, u, [16, 8] * 4, transposed=True)

    def test_stacked_is_a_noop_wrapper(self):
        # ortho_stacked hands the whole 3D tensor over as one batch, and
        # hyperball_apply is already per-matrix (axis=[-2, -1]), so the sliced
        # path must be identical to the plain one.
        w = paddle.randn([3, 32, 48])
        u = paddle.randn([3, 32, 48])
        got = hyperball_apply(w, u, LR, split_concat_func=ortho_stacked)
        plain = hyperball_apply(w, u, LR)
        self.assertEqual(float(paddle.max(paddle.abs(got - plain))), 0.0)


if __name__ == "__main__":
    unittest.main()
