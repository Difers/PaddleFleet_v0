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
"""Behaviour tests for CPU-computable MoE FP8 quantization helpers.

Targets pure-numeric helpers in
``paddlefleet.transformer.moe.fp8_utils`` whose expected outputs can be
derived by hand without a GPU, DeepGEMM, or the ``paddlefleet_ops`` custom
kernels:

* ``moe_token_padding_alignment`` -- per-expert token padding policy.

Every expected value below is an independent hand anchor (literal constants
or an inlined numpy formula), never produced by calling the function under
test. All computation is fp32/int only, so no ``float8_e4m3fn`` runtime
support is required.
"""

import os
import sys
import unittest

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

# Honest capability probe: only a genuinely missing dependency (paddle or the
# paddlefleet package tree) is a legitimate skip. Any other error -- compile
# failure, API change, GPU-only import blowing up with RuntimeError -- must
# surface as a real failure rather than be swallowed into a green run.
try:
    import numpy as np
    import paddle

    from paddlefleet.transformer.moe import fp8_utils

    IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
    np = None
    paddle = None
    fp8_utils = None
    IMPORT_ERROR = repr(exc)


@unittest.skipIf(
    fp8_utils is None,
    f"paddle / paddlefleet not importable in this environment: {IMPORT_ERROR}",
)
class _CpuFp8TestBase(unittest.TestCase):
    """Runs the numeric helpers on CPU and restores the process device.

    Setting the paddle device is process-global mutation, so the original
    device is captured and restored (even on failure) via ``addCleanup``.
    """

    def setUp(self):
        original_device = paddle.device.get_device()
        self.addCleanup(paddle.set_device, original_device)
        paddle.set_device("cpu")


def _silu_ref(x):
    """Independent numpy SiLU: x * sigmoid(x), computed in float64."""
    x = np.asarray(x, dtype=np.float64)
    return x / (1.0 + np.exp(-x))


class TestMoeTokenPaddingAlignment(_CpuFp8TestBase):
    """``moe_token_padding_alignment`` returns 1 only on the single
    accuracy-compatible pure-bf16 non-grouped path; every other flag
    combination keeps the FP8_ALIGN (128) padding."""

    def test_only_accuracy_compatible_bf16_nongrouped_skips_padding(self):
        self.assertEqual(fp8_utils.FP8_ALIGN, 128)
        # The one combination that disables padding.
        self.assertEqual(
            fp8_utils.moe_token_padding_alignment(
                use_fp8_mlp=False,
                moe_grouped_gemm=False,
                use_accuracy_compatible=True,
            ),
            1,
        )
        # Flip each contributing flag in isolation -> back to 128, proving
        # every flag is actually consumed by the branch.
        self.assertEqual(
            fp8_utils.moe_token_padding_alignment(
                use_fp8_mlp=True,
                moe_grouped_gemm=False,
                use_accuracy_compatible=True,
            ),
            128,
        )
        self.assertEqual(
            fp8_utils.moe_token_padding_alignment(
                use_fp8_mlp=False,
                moe_grouped_gemm=True,
                use_accuracy_compatible=True,
            ),
            128,
        )
        self.assertEqual(
            fp8_utils.moe_token_padding_alignment(
                use_fp8_mlp=False,
                moe_grouped_gemm=False,
                use_accuracy_compatible=False,
            ),
            128,
        )


if __name__ == "__main__":
    unittest.main()
