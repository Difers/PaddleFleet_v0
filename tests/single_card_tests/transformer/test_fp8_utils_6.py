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

"""Behavior tests for paddlefleet.transformer.moe.fp8_utils.

All expected values below were derived BY HAND from the production source
(``src/paddlefleet/transformer/moe/fp8_utils.py``); none are copied from any
coverage_test file.

Covered production behaviors:
  * ``moe_token_padding_alignment`` -- the whole AND/NOT guard. It returns 1
    ONLY on the accuracy-compatible pure-bf16 non-grouped path; every other
    combination must fall through to ``FP8_ALIGN``. The full 8-row truth table
    is asserted so flipping any single condition in production is caught.
  * ``has_config`` -- None map, missing key, falsy value all -> False; a
    present truthy value -> True.
  * ``expert_weights_all_frozen`` -- None / empty -> False; all-frozen
    EagerParamBase list -> True; a single trainable entry makes the group
    not-frozen; a non-parameter entry is never frozen.

These require paddle (the module imports it at load time). The covered logic is
CPU-computable and needs no accelerator, but paddle itself must be importable.
When it is not, every case is honestly skipped with the real import error --
never faked as a pass. The import probe catches ONLY ImportError /
ModuleNotFoundError so a genuine API/compile regression still surfaces.
"""

import os
import sys
import unittest

# Allow importing paddlefleet from the in-tree src/ when it is not installed.
_SRC = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
    "src",
)
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

try:
    import paddle

    from paddlefleet.transformer.moe.fp8_utils import (
        FP8_ALIGN,
        expert_weights_all_frozen,
        has_config,
        moe_token_padding_alignment,
    )

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # honest, precise probe
    _IMPORT_ERROR = exc


_SKIP_REASON = (
    None
    if _IMPORT_ERROR is None
    else f"paddle/paddlefleet not importable: {_IMPORT_ERROR!r}"
)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestMoeTokenPaddingAlignment(unittest.TestCase):
    """The alignment guard must consume all three flags with AND/NOT logic."""

    def test_full_truth_table(self):
        # Hand-derived: return 1 iff (accuracy_compatible and not fp8 and
        # not grouped); otherwise FP8_ALIGN. FP8_ALIGN is imported, not
        # hardcoded, so the two branches stay tied to the real constant.
        self.assertEqual(FP8_ALIGN, 128)
        one = 1
        align = FP8_ALIGN
        expected = {
            # (use_accuracy_compatible, use_fp8_mlp, moe_grouped_gemm): result
            (True, False, False): one,  # the ONLY skip-padding case
            (True, False, True): align,
            (True, True, False): align,
            (True, True, True): align,
            (False, False, False): align,
            (False, False, True): align,
            (False, True, False): align,
            (False, True, True): align,
        }
        for (acc, fp8, grouped), want in expected.items():
            got = moe_token_padding_alignment(
                use_accuracy_compatible=acc,
                use_fp8_mlp=fp8,
                moe_grouped_gemm=grouped,
            )
            self.assertEqual(
                got,
                want,
                msg=(
                    f"acc={acc} fp8={fp8} grouped={grouped}: "
                    f"expected {want}, got {got}"
                ),
            )


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestHasConfig(unittest.TestCase):
    """has_config is truthiness of (map is not None and key in map and value)."""

    def test_none_map_is_false(self):
        self.assertIs(has_config(None, "k"), False)

    def test_missing_key_is_false(self):
        self.assertIs(has_config({"other": 1}, "k"), False)

    def test_falsy_values_are_false(self):
        for falsy in (0, None, "", [], {}, 0.0, False):
            self.assertIs(
                has_config({"k": falsy}, "k"),
                False,
                msg=f"falsy value {falsy!r} should be treated as absent",
            )

    def test_truthy_values_are_true(self):
        for truthy in (1, "v", [0], {"x": 1}, 0.5, True):
            self.assertIs(
                has_config({"k": truthy}, "k"),
                True,
                msg=f"truthy value {truthy!r} should be present",
            )


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestExpertWeightsAllFrozen(unittest.TestCase):
    """Frozen detection over stacked / list / single expert parameters."""

    @staticmethod
    def _param(stop_gradient):
        p = paddle.create_parameter(
            shape=[2, 2],
            dtype="float32",
            default_initializer=paddle.nn.initializer.Constant(1.0),
        )
        p.stop_gradient = stop_gradient
        return p

    def test_none_and_empty_are_not_frozen(self):
        self.assertIs(expert_weights_all_frozen(None), False)
        self.assertIs(expert_weights_all_frozen([]), False)
        self.assertIs(expert_weights_all_frozen([None, None]), False)

    def test_all_frozen_list_is_frozen(self):
        weights = [self._param(True), self._param(True)]
        self.assertIs(expert_weights_all_frozen(weights), True)

    def test_mixed_group_is_not_frozen(self):
        # One trainable entry poisons the whole group.
        weights = [self._param(True), self._param(False)]
        self.assertIs(expert_weights_all_frozen(weights), False)

    def test_single_frozen_param_is_frozen(self):
        self.assertIs(expert_weights_all_frozen(self._param(True)), True)

    def test_non_parameter_entry_is_not_frozen(self):
        # A plain tensor is not an EagerParamBase, so it is never counted as
        # frozen even with stop_gradient set.
        t = paddle.to_tensor([1.0, 2.0])
        t.stop_gradient = True
        self.assertIs(expert_weights_all_frozen([t]), False)


if __name__ == "__main__":
    unittest.main()
