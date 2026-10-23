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

"""``calculate_per_token_loss`` scaling of the DSA indexer loss.

Under per-token mode the trainer divides every parameter gradient by
``T_global`` once, at the step boundary, so the in-graph indexer loss must be a
**raw sum over this rank's rows**. These tests pin the two switches that make
that true and the two backward branches they feed:

* ``per_token_indexer_num_rows`` / ``per_token_indexer_backward_coeff``;
* the tilelang and cuDNN branches of
  ``TileLangCSAIndexerLossAutoScaler`` / ``compute_csa_indexer_grads``.

The config guard (which must reject only a *real* CSA indexer) is covered in
``test_transformer_config.py``; the layer-level "logged loss is unchanged"
check lives with the layer harness in ``test_mqa_latent_attention.py``.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import paddle

import paddlefleet.cudnn_ops
import paddlefleet.tilelang_ops  # noqa: F401  (patch target)
from paddlefleet.transformer.csa_attention import (
    TileLangCSAIndexerLossAutoScaler,
    compute_csa_indexer_grads,
)
from paddlefleet.transformer.dsa_attention import DSAIndexerLossAutoScaler
from paddlefleet.transformer.mqa_latent_attention import (
    per_token_indexer_backward_coeff,
    per_token_indexer_num_rows,
)


def _cfg(per_token):
    return SimpleNamespace(calculate_per_token_loss=per_token)


class TestPerTokenHelpers(unittest.TestCase):
    """The two switches that turn the mean reduction into a raw row sum."""

    def test_num_rows_keeps_the_legacy_denominator(self):
        self.assertEqual(per_token_indexer_num_rows(_cfg(False), 123.0), 123.0)
        # ``None`` stays ``None``: the backward then falls back to ``b*s``.
        self.assertIsNone(per_token_indexer_num_rows(_cfg(False), None))

    def test_num_rows_is_one_under_per_token(self):
        # Even with no mask -- otherwise the kernel's ``1/(b*s)`` remains.
        self.assertEqual(per_token_indexer_num_rows(_cfg(True), 123.0), 1.0)
        self.assertEqual(per_token_indexer_num_rows(_cfg(True), None), 1.0)

    def test_backward_coeff_undoes_the_cp_division(self):
        # The unmasked branch pre-divides by cp; per-token CP *sums* into
        # T_global, so that division has to be undone.
        self.assertAlmostEqual(
            per_token_indexer_backward_coeff(_cfg(True), 0.01 / 8, 8, False),
            0.01,
        )
        # The masked branch already normalises by the global row count.
        self.assertEqual(
            per_token_indexer_backward_coeff(_cfg(True), 0.01, 8, True),
            0.01,
        )
        # Legacy mode is untouched on both branches.
        self.assertAlmostEqual(
            per_token_indexer_backward_coeff(_cfg(False), 0.01 / 8, 8, False),
            0.01 / 8,
        )
        self.assertEqual(
            per_token_indexer_backward_coeff(_cfg(False), 0.01, 8, True),
            0.01,
        )


class _ScalerFixture:
    """Shared tensors for the two backward branches (no kernels are run)."""

    b, sq, topk, d = 2, 8, 4, 16
    coeff = 0.01
    cp_size = 8

    def setUp(self):
        paddle.set_device("cpu")
        self._saved_scale = DSAIndexerLossAutoScaler._main_loss_backward_scale
        DSAIndexerLossAutoScaler._main_loss_backward_scale = None
        self.addCleanup(
            setattr,
            DSAIndexerLossAutoScaler,
            "_main_loss_backward_scale",
            self._saved_scale,
        )
        b, sq, topk, d = self.b, self.sq, self.topk, self.d
        # Non-leaf on purpose: the PyLayer returns its ``output`` argument
        # unchanged, which Paddle rejects as an "inplace" return for a leaf.
        self._base = paddle.randn([b, sq, d], dtype="float32")
        self._base.stop_gradient = False
        self.index_q = paddle.randn([b, sq, 4, d], dtype="float32")
        self.weights = paddle.randn([b, sq, 4], dtype="float32")
        self.index_k = paddle.randn([b, sq // 2, d], dtype="float32")
        for tensor in (self.index_q, self.weights, self.index_k):
            tensor.stop_gradient = False
        self.topk_indices = (
            paddle.arange(topk, dtype="int64")
            .reshape([1, 1, topk])
            .expand([b, sq, topk])
            .contiguous()
        )
        logits = paddle.randn([b, sq, topk], dtype="float32")
        self.topk_probs = paddle.nn.functional.softmax(logits, axis=-1)
        self.target = paddle.nn.functional.softmax(logits * 0.5, axis=-1)

    def _zero_grads(self):
        return (
            paddle.zeros_like(self.index_q),
            paddle.zeros_like(self.weights),
            paddle.zeros_like(self.index_k),
        )


class TestTileLangPerTokenScaler(_ScalerFixture, unittest.TestCase):
    """The tilelang branch multiplies by ``loss_coeff / num_rows``."""

    def _captured_score_grad(self, per_token, loss_mask):
        valid_rows = None
        if loss_mask is not None:
            valid_rows = max(float(loss_mask.sum()), 1.0)
        # Same coefficient placement as ``MQALatentAttention``: the unmasked
        # branch folds ``/cp`` into the coefficient, the masked branch does not.
        loss_coeff = (
            self.coeff if loss_mask is not None else self.coeff / self.cp_size
        )
        bwd_coeff = per_token_indexer_backward_coeff(
            _cfg(per_token), loss_coeff, self.cp_size, loss_mask is not None
        )
        bwd_rows = per_token_indexer_num_rows(_cfg(per_token), valid_rows)
        with patch("paddlefleet.tilelang_ops.csa_indexer_bwd") as mock_bwd:
            mock_bwd.return_value = self._zero_grads()
            out = TileLangCSAIndexerLossAutoScaler.apply(
                self._base * 1.0,
                self.target,
                self.index_q,
                self.weights,
                self.index_k,
                self.topk_indices,
                self.topk_probs,
                bwd_coeff,
                "tilelang",
                bwd_rows,
                loss_mask,
            )
            out.sum().backward()
            # ``csa_indexer_bwd(index_q, weights, index_k, topk_indices,
            # grad_index_scores)`` -- captured before the kernel divides.
            return mock_bwd.call_args[0][4].clone()

    def test_unmasked_gradient_is_a_raw_row_sum(self):
        n = float(self.b * self.sq)
        legacy = self._captured_score_grad(False, None)
        per_token = self._captured_score_grad(True, None)

        expected_mean = (
            (self.topk_probs - self.target) * self.coeff / self.cp_size / n
        )
        paddle.testing.assert_close(legacy, expected_mean, rtol=1e-6, atol=1e-8)
        # Per-token drops both the ``1/(b*s)`` and the legacy ``/cp``, so the
        # in-graph sum is the CP sum the trainer then divides by T_global.
        paddle.testing.assert_close(
            per_token, expected_mean * n * self.cp_size, rtol=1e-6, atol=1e-8
        )

    def test_masked_gradient_is_a_raw_row_sum(self):
        loss_mask = paddle.ones([self.b, self.sq], dtype="float32")
        loss_mask[:, self.sq // 2 :] = 0.0
        n = max(float(loss_mask.sum()), 1.0)
        legacy = self._captured_score_grad(False, loss_mask)
        per_token = self._captured_score_grad(True, loss_mask)

        # The mask zeroes the padded rows in both modes, so only the
        # denominator differs: the global valid-row count versus ``1``.
        paddle.testing.assert_close(per_token, legacy * n, rtol=1e-6, atol=1e-8)


class TestCudnnPerTokenScaler(_ScalerFixture, unittest.TestCase):
    """The cuDNN kernel divides by ``B*Sq``; ``num_rows`` overrides it."""

    def _captured_loss_coeff(self, num_rows, loss_mask):
        with patch("paddlefleet.cudnn_ops.csa_indexer_bwd") as mock_bwd:
            mock_bwd.return_value = self._zero_grads()
            compute_csa_indexer_grads(
                self.index_q,
                self.weights,
                self.index_k,
                self.target,
                self.topk_probs,
                self.topk_indices,
                loss_coeff=self.coeff,
                indexer_backend="cudnn",
                num_rows=num_rows,
                loss_mask=loss_mask,
            )
            return float(mock_bwd.call_args.kwargs["loss_coeff"])

    def test_legacy_unmasked_keeps_the_kernel_denominator(self):
        n = float(self.b * self.sq)
        # The auto-scaler always hands over ``b*s`` on its fallback, which the
        # compensation turns into an identity.
        self.assertAlmostEqual(
            self._captured_loss_coeff(n, None) / self.coeff, 1.0
        )

    def test_per_token_compensates_the_raw_sum(self):
        n = float(self.b * self.sq)
        self.assertAlmostEqual(
            self._captured_loss_coeff(1.0, None) / (self.coeff * n), 1.0
        )

    def test_masked_branch_keeps_the_global_row_count(self):
        loss_mask = paddle.ones([self.b, self.sq], dtype="float32")
        loss_mask[:, self.sq // 2 :] = 0.0
        valid = max(float(loss_mask.sum()), 1.0)
        n = float(self.b * self.sq)
        self.assertAlmostEqual(
            self._captured_loss_coeff(valid, loss_mask)
            / (self.coeff * n / valid),
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
