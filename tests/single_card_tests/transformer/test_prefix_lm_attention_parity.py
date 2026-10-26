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

"""Numerical parity of the HyperEncoder prefix-LM attention through the real
``DotProductAttention`` branches it uses.

* ``flashmask`` backend: bf16 ``DotProductAttention`` (non-eager,
  ``AttnMaskType.no_mask``) fed the 2-column row indices from
  :func:`build_flashmask_row_indices`, i.e. the ``flashmask_attention`` call with
  ``causal=False``;
* ``dp`` backend: fp32 eager ``DotProductAttention`` fed the dense mask from
  :func:`build_dense_mask`.

Both are compared, forward and backward, against an independent fp32 dense
reference. The cases cover multi-segment packs, padding, an empty prefix and a
mixed short/long query pack. Pad rows must attend only to themselves (output ==
value, no NaN), which is what keeps the padded tail finite.
"""

from __future__ import annotations

import unittest

import numpy as np
import paddle
import paddle.nn.functional as F

from paddlefleet.transformer.dot_product_attention import DotProductAttention
from paddlefleet.transformer.enums import AttnMaskType
from paddlefleet.transformer.prefix_lm_mask import (
    build_dense_mask,
    build_flashmask_row_indices,
)
from paddlefleet.transformer.transformer_config import TransformerConfig
from paddlefleet.utils import init_method_normal, scaled_init_method_normal

try:
    from paddlefleet_ops import is_flash_mask_available

    _FLASHMASK = bool(is_flash_mask_available())
except ImportError:  # pragma: no cover
    _FLASHMASK = False

_GPU = paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
_H, _D = 4, 128

# (prefix_lens, suffix_lens, pad_len, description)
_CASES = [
    ([10], [64], 0, "single segment, no pad"),
    ([10], [64], 54, "single segment + pad"),
    ([16, 24], [32, 32], 24, "two segments + pad"),
    ([5, 5, 5], [8, 8, 8], 25, "three segments + pad"),
    ([0, 8], [16, 16], 0, "first segment has no prefix"),
    ([7, 3], [16, 96], 6, "mixed short/long query pack"),
]
_MIN_COS_FWD, _MAX_ABS_FWD = 0.999, 3e-2
_MIN_COS_BWD, _MAX_ABS_BWD = 0.99, 8e-2


def _config(attn_impl, bf16):
    cfg = TransformerConfig(
        num_hidden_layers=1,
        hidden_size=_H * _D,
        num_attention_heads=_H,
        num_key_value_heads=_H,
        head_dim=_D,
        use_bias=False,
        init_method=init_method_normal(0.02),
        output_layer_init_method=scaled_init_method_normal(0.02, 1, 2.0),
        context_parallel_size=1,
        sequence_parallel=False,
        bf16=bf16,
        params_dtype=paddle.bfloat16 if bf16 else paddle.float32,
        attention_dropout=0.0,
        masked_softmax_fusion=False,
        attention_softmax_in_fp32=True,
    )
    cfg._attn_implementation = attn_impl
    return cfg


def _dpa(attn_impl, bf16):
    return DotProductAttention(
        config=_config(attn_impl, bf16),
        layer_number=1,
        attn_mask_type=AttnMaskType.no_mask,
        attention_type="self",
    )


def _reference(q, k, v, forbidden):
    """fp32 dense attention. q/k/v: ``[B, T, H, D]``; forbidden ``[1,1,T,T]``."""
    qf, kf, vf = (
        t.astype("float32").transpose([0, 2, 1, 3]) for t in (q, k, v)
    )
    score = paddle.matmul(qf, kf, transpose_y=True) / np.sqrt(_D)
    score = paddle.where(
        forbidden, paddle.full_like(score, float("-inf")), score
    )
    out = paddle.matmul(F.softmax(score, axis=-1), vf)  # [B, H, T, D]
    return out.transpose([0, 2, 1, 3]).reshape([q.shape[0], q.shape[1], -1])


def _cos(a, b):
    x, y = a.reshape(-1).astype(np.float64), b.reshape(-1).astype(np.float64)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / denom) if denom > 0 else 1.0


def _leaf(t, dtype):
    t = t.astype(dtype).detach()
    t.stop_gradient = False
    return t


def _run(fn, base, gout, dtype):
    q, k, v = (_leaf(t, dtype) for t in base)
    out = fn(q, k, v)
    (out.astype("float32") * gout).sum().backward()
    return out, q.grad, k.grad, v.grad


class _ParityBase(unittest.TestCase):
    def _compare(self, ref, got, total, desc):
        names = ("out", "dq", "dk", "dv")
        for i, (name, r, g) in enumerate(zip(names, ref, got)):
            # Real-token rows only: the pad tail is sliced off downstream.
            r = r.astype("float32").numpy()[:, :total]
            g = g.astype("float32").numpy()[:, :total]
            min_cos, max_abs = (
                (_MIN_COS_FWD, _MAX_ABS_FWD)
                if i == 0
                else (_MIN_COS_BWD, _MAX_ABS_BWD)
            )
            cos, mabs = _cos(r, g), float(np.abs(r - g).max())
            self.assertGreaterEqual(
                cos, min_cos, f"{desc} {name}: cos={cos:.6f}"
            )
            self.assertLessEqual(
                mabs, max_abs, f"{desc} {name}: max_abs={mabs}"
            )

    def _inputs(self, prefix, suffix, pad):
        total = sum(c + q for c, q in zip(prefix, suffix))
        tp = total + pad
        paddle.seed(20260911)
        base = [paddle.randn([1, tp, _H, _D]) for _ in range(3)]
        gout = paddle.randn([1, tp, _H * _D])
        return total, tp, base, gout


@unittest.skipUnless(_GPU and _FLASHMASK, "requires a GPU with FlashMask")
class TestFlashMaskBackendParity(_ParityBase):
    def test_cases(self):
        dpa = _dpa("flashmask", bf16=True)
        for prefix, suffix, pad, desc in _CASES:
            with self.subTest(desc=desc):
                total, tp, base, gout = self._inputs(prefix, suffix, pad)
                forbidden = build_dense_mask(prefix, suffix, pad)
                ri = build_flashmask_row_indices(prefix, suffix, pad)
                ref = _run(
                    lambda q, k, v: _reference(q, k, v, forbidden),
                    base,
                    gout,
                    "float32",
                )
                got = _run(
                    lambda q, k, v: dpa(
                        q, k, v, None, attn_mask_startend_row_indices=ri
                    ),
                    base,
                    gout,
                    "bfloat16",
                )
                self._compare(ref, got, total, desc)
                out = got[0].astype("float32")
                self.assertFalse(bool(paddle.isnan(out).any()), desc)
                if pad:
                    # Pad rows see only themselves -> output equals value.
                    v = base[2].astype("bfloat16").astype("float32")
                    np.testing.assert_allclose(
                        out.numpy()[:, total:],
                        v.reshape([1, tp, -1]).numpy()[:, total:],
                        atol=1e-2,
                        err_msg=desc,
                    )


class TestDenseBackendParity(_ParityBase):
    """The eager dense path (``dp`` backend) on the same cases, in fp32."""

    def test_cases(self):
        dpa = _dpa("eager", bf16=False)
        for prefix, suffix, pad, desc in _CASES[:4]:
            with self.subTest(desc=desc):
                total, _, base, gout = self._inputs(prefix, suffix, pad)
                forbidden = build_dense_mask(prefix, suffix, pad)
                ref = _run(
                    lambda q, k, v: _reference(q, k, v, forbidden),
                    base,
                    gout,
                    "float32",
                )
                got = _run(
                    lambda q, k, v: dpa(q, k, v, forbidden),
                    base,
                    gout,
                    "float32",
                )
                for r, g in zip(ref, got):
                    np.testing.assert_allclose(
                        g.numpy()[:, :total],
                        r.numpy()[:, :total],
                        rtol=1e-4,
                        atol=1e-5,
                        err_msg=desc,
                    )


if __name__ == "__main__":
    unittest.main()
