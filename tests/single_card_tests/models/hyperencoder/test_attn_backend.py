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

"""Unit tests for the HyperEncoder attention-backend config helpers.

The two switches (``hyperencoder_attn_backend`` / ``hyperencoder_packed_decoder``)
must move together: a mixed pair would silently feed the wrong mask carrier to
``DotProductAttention`` (dense mask into the flashmask branch, or row indices
into the eager branch), so both mixed combinations must raise.
"""

import types
import unittest

from paddlefleet.models.hyperencoder import attn_backend


def _cfg(**kw):
    """A duck-typed config carrying only the hyperencoder_* fields under test."""
    return types.SimpleNamespace(**kw)


class TestEncoderAttnBackend(unittest.TestCase):
    def test_default_is_dp(self):
        # A config without the field falls back to the "dp" default.
        self.assertEqual(attn_backend.encoder_attn_backend(_cfg()), "dp")
        self.assertFalse(attn_backend.use_flashmask_encoder_attn(_cfg()))

    def test_flashmask_selected_case_insensitive(self):
        for v in ("flashmask", "FlashMask", "FLASHMASK"):
            with self.subTest(v=v):
                c = _cfg(hyperencoder_attn_backend=v)
                self.assertEqual(
                    attn_backend.encoder_attn_backend(c), "flashmask"
                )
                self.assertTrue(attn_backend.use_flashmask_encoder_attn(c))

    def test_retired_and_unknown_backends_raise_with_valid_choices(self):
        # "triton" was the previous packed backend; an old yaml must fail loudly
        # (and say what to use instead) rather than fall back to "dp".
        for v in ("triton", "flex", "nope", ""):
            with self.subTest(v=v):
                with self.assertRaises(ValueError) as cm:
                    attn_backend.encoder_attn_backend(
                        _cfg(hyperencoder_attn_backend=v)
                    )
                self.assertIn("flashmask", str(cm.exception))
                self.assertIn("dp", str(cm.exception))


class TestUsePackedDecoder(unittest.TestCase):
    def test_default_off(self):
        self.assertFalse(attn_backend.use_packed_decoder(_cfg()))

    def test_matching_pairs_are_accepted(self):
        self.assertTrue(
            attn_backend.use_packed_decoder(
                _cfg(
                    hyperencoder_packed_decoder=True,
                    hyperencoder_attn_backend="flashmask",
                )
            )
        )
        self.assertFalse(
            attn_backend.use_packed_decoder(
                _cfg(
                    hyperencoder_packed_decoder=False,
                    hyperencoder_attn_backend="dp",
                )
            )
        )

    def test_mixed_pairs_raise_and_report_both_values(self):
        for packed, backend in ((True, "dp"), (False, "flashmask")):
            with self.subTest(packed=packed, backend=backend):
                with self.assertRaises(RuntimeError) as cm:
                    attn_backend.use_packed_decoder(
                        _cfg(
                            hyperencoder_packed_decoder=packed,
                            hyperencoder_attn_backend=backend,
                        )
                    )
                msg = str(cm.exception)
                self.assertIn(f"packed={packed}", msg)
                self.assertIn(repr(backend), msg)

    def test_unknown_backend_is_reported_before_the_pairing(self):
        # A typo must surface as the backend error, not as a pairing error.
        with self.assertRaises(ValueError):
            attn_backend.use_packed_decoder(
                _cfg(
                    hyperencoder_packed_decoder=True,
                    hyperencoder_attn_backend="triton",
                )
            )


if __name__ == "__main__":
    unittest.main()
