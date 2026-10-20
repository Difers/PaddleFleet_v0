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

"""Equivalence of the Triton fused CE/L1 loss against the eager streamed path.

``dspark_fused_distill_loss.make_fused_ce_l1`` fuses the projection, SegLU, the
two softmaxes, CE and L1 into streaming kernels, and is selected by
``LossProfile.use_fused_distill_loss`` (see
``dspark_losses._streamed_token_losses``).
The eager ``_ChunkedTokenLoss`` path is the reference: the two must agree on the
per-token CE/L1 and on the gradient w.r.t. the draft hidden states and the
Markov refiner parameters, for both ``use_seglu`` branches.

The fused path silently falls back to eager when Triton / CUDA / a GPU place is
missing, so the test skips there and asserts the kernel was actually entered
otherwise (a silent fallback would otherwise make the comparison vacuous).
"""

import types
import unittest

import numpy as np
import paddle

import paddlefleet.utils.dspark.dspark_fused_distill_loss as fused_mod
from paddlefleet.training.initialize import initialize_fleet
from paddlefleet.utils.dspark.dspark_losses import (
    LossProfile,
    compute_dspark_loss,
)
from paddlefleet.utils.dspark.dspark_utils import (
    DrafterOutput,
    ReadOnlyProjection,
)

initialize_fleet(strategy=paddle.distributed.fleet.DistributedStrategy())

_B, _A, _K = 1, 4, 3
_H, _V, _R = 32, 64, 8
_ATOL = 1e-4
_RTOL = 1e-4


def _fused_runnable() -> bool:
    """Whether the fused kernel can run here (mirrors the production guard)."""
    try:
        import triton  # noqa: F401
    except Exception:
        return False
    if not paddle.device.is_compiled_with_cuda():
        return False
    return paddle.device.cuda.device_count() > 0


def _make_inputs(use_seglu: bool, transpose_y: bool, seed: int):
    """Build one deterministic DrafterOutput over a ReadOnlyProjection."""
    paddle.seed(seed)
    draft = paddle.randn([_B, _A, _K, _H], dtype="float32")
    target = paddle.randn([_B, _A, _K, _H], dtype="float32")
    target.stop_gradient = True
    target_ids = paddle.randint(1, _V, [_B, _A, _K], dtype="int64")
    previous = paddle.randint(0, _V, [_B, _A, _K], dtype="int64")
    eval_mask = paddle.ones([_B, _A, _K], dtype="bool")
    block_keep = paddle.ones([_B, _A], dtype="bool")
    # transpose_y=True means a [V, H] head, False a [H, V] head.
    weight = paddle.randn(
        [_V, _H] if transpose_y else [_H, _V], dtype="float32"
    )
    w1 = paddle.randn([_V, _R], dtype="float32")
    w2 = paddle.randn([_R, _V], dtype="float32")
    for tensor in (draft, w1, w2):
        tensor.stop_gradient = False
    ranges = (
        paddle.to_tensor([-1.0, 1.0, -2.0, 2.0], dtype="float32")
        if use_seglu
        else None
    )
    ts = (
        paddle.to_tensor([0.1, 0.2, 0.3, 0.4], dtype="float32")
        if use_seglu
        else None
    )
    projection = ReadOnlyProjection(
        weight=weight,
        transpose_y=transpose_y,
        bias=None,
        multimax_ranges=ranges,
        multimax_ts=ts,
    )
    # A bare attribute container is enough: the loss only reads
    # ``markov_w1.weight`` / ``markov_w2.weight``.
    markov_head = types.SimpleNamespace(
        markov_w1=types.SimpleNamespace(weight=w1),
        markov_w2=types.SimpleNamespace(weight=w2),
    )
    output = DrafterOutput(
        draft_logits=None,
        target_ids=target_ids,
        eval_mask=eval_mask,
        block_keep_mask=block_keep,
        draft_hidden_states=draft,
        aligned_target_hidden_states=target,
        previous_token_ids=previous,
        logits_projection=projection,
        markov_head=markov_head,
    )
    return output, markov_head


def _run(use_fused: bool, use_seglu: bool, transpose_y: bool, seed: int = 0):
    """Run the loss end to end and collect scalars plus input gradients."""
    output, markov = _make_inputs(use_seglu, transpose_y, seed)
    terms = compute_dspark_loss(
        output,
        LossProfile(use_fused_distill_loss=use_fused),
        return_terms=True,
    )
    terms.loss.backward()
    return {
        "loss": float(terms.terms["loss"]),
        "ce": float(terms.terms["ce_loss"]),
        "l1": float(terms.terms["l1_loss"]),
        "draft_grad": output.draft_hidden_states.grad,
        "w1_grad": markov.markov_w1.weight.grad,
        "w2_grad": markov.markov_w2.weight.grad,
    }


class TestFusedDistillLossEquivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not _fused_runnable():
            raise unittest.SkipTest(
                "Triton/CUDA GPU unavailable: fused path cannot run"
            )
        paddle.set_device("gpu")

    def _assert_matches(self, eager, fused):
        # A zero gradient would make the allclose comparison vacuous.
        self.assertGreater(
            float(np.abs(eager["draft_grad"].numpy()).max()),
            0.0,
            "eager draft gradient is all zeros; comparison would be vacuous",
        )
        for name in ("loss", "ce", "l1"):
            self.assertAlmostEqual(
                eager[name],
                fused[name],
                delta=_ATOL,
                msg=f"{name} differs between eager and fused paths",
            )
        for name in ("draft_grad", "w1_grad", "w2_grad"):
            self.assertIsNotNone(eager[name], f"eager {name} is None")
            self.assertIsNotNone(fused[name], f"fused {name} is None")
            np.testing.assert_allclose(
                eager[name].numpy(),
                fused[name].numpy(),
                atol=_ATOL,
                rtol=_RTOL,
                err_msg=f"{name} differs between eager and fused paths",
            )

    def _check(self, use_seglu: bool, transpose_y: bool = True):
        calls = []
        original = fused_mod.make_fused_ce_l1

        def _spy(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        fused_mod.make_fused_ce_l1 = _spy
        try:
            eager = _run(
                use_fused=False, use_seglu=use_seglu, transpose_y=transpose_y
            )
            fused = _run(
                use_fused=True, use_seglu=use_seglu, transpose_y=transpose_y
            )
        finally:
            fused_mod.make_fused_ce_l1 = original

        # Guard against a silent fallback: the fused run must have built the
        # kernel, otherwise the comparison would be eager-vs-eager.
        self.assertEqual(
            len(calls), 1, "fused kernel was not exercised by the fused run"
        )
        self._assert_matches(eager, fused)

    def test_fused_matches_eager_without_seglu(self):
        self._check(use_seglu=False)

    def test_fused_matches_eager_with_seglu(self):
        self._check(use_seglu=True)

    def test_fused_matches_eager_weight_not_transposed(self):
        self._check(use_seglu=False, transpose_y=False)


if __name__ == "__main__":
    unittest.main()
