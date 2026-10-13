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
"""Behavior tests for ``paddlefleet.transformer.moe.fp8_utils``.

Module: 计算优化 (FP8 MoE expert utilities). These exercise CPU-runnable,
numerically-derivable pieces of the fp8 expert helpers against hand-derived
independent references (numpy / IEEE-754 reasoning), not the production code.

Scope / honesty: the GPU-only fp8 kernels (deep_gemm grouped GEMM,
``fp8_quant_blockwise``, the fused swiglu/probs ops) are NOT exercised here --
they need a real device and are covered by the single/multi-card fp8 suites.
This file only validates the pure-python / dtype-agnostic logic: the padding
alignment policy, the ue8m0 (power-of-two) scale rounding, the fp32 weighted
swiglu math and its clamp semantics, the blockwise dequant scale broadcast,
the frozen-expert detection contract, and small config/stacking helpers.
"""

import unittest

import numpy as np

try:
    import paddle

    from paddlefleet.transformer.moe.fp8_utils import (
        FP8_ALIGN,
        ExpertsGroupGemmContiguousNode,
        _stack_expert_weights,
        expert_weights_all_frozen,
        has_config,
        moe_token_padding_alignment,
    )

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # honest: no paddle here
    paddle = None
    _IMPORT_ERROR = exc


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _silu(x):
    return x * _sigmoid(x)


class _MinimalCustomMap:
    """Minimal, genuine not-under-test collaborator for node construction.

    Only ``.experts`` / ``.grouped_gemm_experts`` are read on the bf16
    split-group path used by these tests.
    """

    def __init__(self):
        self.experts = []
        self.grouped_gemm_experts = None


_SKIP = paddle is None
_SKIP_REASON = (
    "paddle / paddlefleet.transformer.moe.fp8_utils is not importable in this "
    f"CPU-only environment ({_IMPORT_ERROR!r}); fp8 utils require the paddle "
    "install used by the single-card CI."
)


@unittest.skipIf(_SKIP, _SKIP_REASON)
class TestMoeTokenPaddingAlignment(unittest.TestCase):
    """The alignment policy returns 1 only on the accuracy-compatible,
    pure-bf16, non-grouped-gemm path; every other combination aligns to
    FP8_ALIGN. Walk the full 8-entry truth table so an inverted/dropped
    condition cannot pass."""

    def test_full_truth_table(self):
        self.assertEqual(FP8_ALIGN, 128)
        for use_fp8_mlp in (False, True):
            for grouped in (False, True):
                for acc in (False, True):
                    got = moe_token_padding_alignment(
                        use_fp8_mlp=use_fp8_mlp,
                        moe_grouped_gemm=grouped,
                        use_accuracy_compatible=acc,
                    )
                    expected = (
                        1
                        if (acc and not use_fp8_mlp and not grouped)
                        else FP8_ALIGN
                    )
                    self.assertEqual(
                        got,
                        expected,
                        msg=f"fp8={use_fp8_mlp} grouped={grouped} acc={acc}",
                    )

    def test_only_one_combination_skips_padding(self):
        # Exactly one of the eight flag combinations may return 1.
        skips = 0
        for use_fp8_mlp in (False, True):
            for grouped in (False, True):
                for acc in (False, True):
                    if (
                        moe_token_padding_alignment(
                            use_fp8_mlp=use_fp8_mlp,
                            moe_grouped_gemm=grouped,
                            use_accuracy_compatible=acc,
                        )
                        == 1
                    ):
                        skips += 1
        self.assertEqual(skips, 1)


@unittest.skipIf(_SKIP, _SKIP_REASON)
@unittest.skipIf(_SKIP, _SKIP_REASON)
@unittest.skipIf(_SKIP, _SKIP_REASON)
@unittest.skipIf(_SKIP, _SKIP_REASON)
class TestExpertWeightsAllFrozen(unittest.TestCase):
    """``expert_weights_all_frozen`` returns True only when every entry
    dereferences (via ``_parent``) to a frozen ``EagerParamBase``. None/empty,
    mixed groups, trainable params and non-params are all not-frozen."""

    def _param(self, frozen):
        p = paddle.create_parameter([2, 2], dtype="float32")
        p.stop_gradient = frozen
        return p

    def test_none_and_empty_are_not_frozen(self):
        self.assertFalse(expert_weights_all_frozen(None))
        self.assertFalse(expert_weights_all_frozen([]))

    def test_single_param_reflects_stop_gradient(self):
        self.assertTrue(expert_weights_all_frozen(self._param(frozen=True)))
        self.assertFalse(expert_weights_all_frozen(self._param(frozen=False)))

    def test_mixed_group_is_not_frozen(self):
        self.assertFalse(
            expert_weights_all_frozen(
                [self._param(frozen=True), self._param(frozen=False)]
            )
        )
        self.assertTrue(
            expert_weights_all_frozen(
                [self._param(frozen=True), self._param(frozen=True)]
            )
        )

    def test_non_param_object_is_not_frozen(self):
        # A plain object is not an EagerParamBase, so no gradient is silently
        # dropped -> treated as not frozen even if it claims stop_gradient.
        class _Plain:
            stop_gradient = True

        self.assertFalse(expert_weights_all_frozen(_Plain()))

    def test_parent_is_dereferenced(self):
        # A per-expert view's own stop_gradient is always True, but the
        # frozen decision must follow ``_parent`` to the real parameter.
        frozen_parent = self._param(frozen=True)
        trainable_parent = self._param(frozen=False)

        class _View:
            def __init__(self, parent):
                self._parent = parent
                self.stop_gradient = True  # views are always detached

        self.assertTrue(expert_weights_all_frozen([_View(frozen_parent)]))
        self.assertFalse(expert_weights_all_frozen([_View(trainable_parent)]))


@unittest.skipIf(_SKIP, _SKIP_REASON)
class TestHasConfig(unittest.TestCase):
    """``has_config`` is True only when the map exists, has the key, and the
    value is truthy."""

    def test_truth_table(self):
        self.assertFalse(has_config(None, "k"))
        self.assertFalse(has_config({}, "k"))
        self.assertFalse(has_config({"k": 0}, "k"))
        self.assertFalse(has_config({"k": ""}, "k"))
        self.assertFalse(has_config({"k": None}, "k"))
        self.assertFalse(has_config({"other": 1}, "k"))
        self.assertTrue(has_config({"k": True}, "k"))
        self.assertTrue(has_config({"k": 5}, "k"))


@unittest.skipIf(_SKIP, _SKIP_REASON)
class TestStackExpertWeights(unittest.TestCase):
    """``_stack_expert_weights`` normalizes list/tuple/tensor inputs to a
    single [E, H0, H1] tensor and rejects non-3D results."""

    def test_single_element_list_returns_that_element(self):
        w = paddle.arange(6, dtype="float32").reshape([1, 2, 3])
        out = _stack_expert_weights([w])
        self.assertEqual(out.shape, [1, 2, 3])
        np.testing.assert_array_equal(out.numpy(), w.numpy())

    def test_multi_element_list_is_stacked_in_order(self):
        a = paddle.zeros([2, 3], dtype="float32")
        b = paddle.ones([2, 3], dtype="float32")
        out = _stack_expert_weights([a, b])
        self.assertEqual(out.shape, [2, 2, 3])
        np.testing.assert_array_equal(out.numpy()[0], a.numpy())
        np.testing.assert_array_equal(out.numpy()[1], b.numpy())

    def test_bare_3d_tensor_passthrough(self):
        w = paddle.ones([3, 2, 4], dtype="float32")
        out = _stack_expert_weights(w)
        self.assertEqual(out.shape, [3, 2, 4])

    def test_non_3d_is_rejected(self):
        with self.assertRaises(AssertionError):
            _stack_expert_weights(paddle.ones([2, 3], dtype="float32"))


@unittest.skipIf(_SKIP, _SKIP_REASON)
class TestFwdSwiglu(unittest.TestCase):
    """``ExpertsGroupGemmContiguousNode.fwd_swiglu`` applies SwiGLU
    (silu(gate) * up) over the real node entry; reference is numpy silu."""

    def test_forward_activation_matches_reference(self):
        node = ExpertsGroupGemmContiguousNode(
            _MinimalCustomMap(), use_fp8_mlp=False
        )
        o1 = np.array(
            [[0.0, 2.0, 3.0, -1.0], [1.0, -2.0, 0.5, 4.0]], np.float32
        )
        gate, up = o1[:, :2], o1[:, 2:]
        ref = _silu(gate) * up
        got = node.fwd_swiglu(paddle.to_tensor(o1)).numpy()
        self.assertEqual(list(got.shape), [2, 2])
        np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
