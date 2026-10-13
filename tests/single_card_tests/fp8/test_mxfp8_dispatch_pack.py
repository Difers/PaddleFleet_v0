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

"""CPU-only bit-exactness tests for the W4A8 fp8-dispatch scale carrier.

Exercises ``pack_ue8m0_to_int32`` / ``pack_ue8m0_to_fp32_slot`` /
``unpack_ue8m0_scale`` from ``paddlefleet.transformer.moe.mxfp8_dispatch``.
That module is pure integer bit manipulation on paddle tensors (imports only
paddle), so these run on a CPU-only box with no custom ops.
"""

import unittest

import numpy as np
import paddle

from paddlefleet.transformer.moe.mxfp8_dispatch import (
    GROUPS_PER_SLOT_WORD,
    pack_ue8m0_to_fp32_slot,
    pack_ue8m0_to_int32,
    unpack_ue8m0_scale,
)


def _ue8m0_from_exponents(exps):
    """Build exact UE8M0 fp32 scales (sign 0, mantissa 0) from exponent bytes.

    ``exps`` is an int array in [0, 255]; the fp32 bit pattern is ``exp << 23``.
    """
    exps = np.asarray(exps, dtype=np.int64).astype(np.uint32)
    bits = (exps << 23).astype(np.uint32)
    return paddle.to_tensor(bits.view(np.float32))


def _as_int32_bits(tensor):
    """Raw int32 bit view (for sign-0/mantissa-0 exact comparisons)."""
    return tensor.view(paddle.int32).numpy()


class TestUe8m0PackRoundTrip(unittest.TestCase):
    def setUp(self):
        paddle.set_device("cpu")
        np.random.seed(23)

    def _random_exponents(self, rows, groups, include_padding=False):
        # Real UE8M0 scales carry a biased exponent in [1, 254]; padding rows
        # carry a zero exponent byte (which must unpack to a finite +0.0).
        exps = np.random.randint(1, 255, size=(rows, groups), dtype=np.int64)
        if include_padding and rows >= 2:
            exps[rows // 2] = 0
            exps[-1] = 0
        return exps

    def _assert_bit_equal(self, actual, expected, msg):
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(list(actual.shape), list(expected.shape))
        self.assertTrue(
            np.array_equal(_as_int32_bits(actual), _as_int32_bits(expected)),
            msg,
        )

    def test_int32_round_trip_bit_exact(self):
        for rows, groups in ((1, 4), (3, 8), (17, 32), (256, 128)):
            with self.subTest(rows=rows, groups=groups):
                exps = self._random_exponents(rows, groups)
                sf = _ue8m0_from_exponents(exps)
                packed = pack_ue8m0_to_int32(sf)
                self.assertEqual(packed.dtype, paddle.int32)
                self.assertEqual(
                    list(packed.shape),
                    [rows, groups // GROUPS_PER_SLOT_WORD],
                )
                recovered = unpack_ue8m0_scale(packed, groups)
                self.assertEqual(recovered.dtype, paddle.float32)
                self._assert_bit_equal(
                    recovered, sf, "int32 pack/unpack is not bit-exact"
                )

    def test_zero_and_padding_rows(self):
        exps = self._random_exponents(8, 16, include_padding=True)
        sf = _ue8m0_from_exponents(exps)
        recovered = unpack_ue8m0_scale(pack_ue8m0_to_int32(sf), 16)
        self._assert_bit_equal(
            recovered, sf, "padding/zero rows not preserved bit-exactly"
        )
        # Zero exponent bytes must decode to a finite +0.0, not NaN/Inf.
        recovered_np = recovered.numpy()
        zero_rows = np.asarray(exps) == 0
        self.assertTrue(np.all(recovered_np[zero_rows] == 0.0))
        self.assertTrue(np.all(np.isfinite(recovered_np)))

    def test_fp32_slot_is_byte_identical_to_int32(self):
        exps = self._random_exponents(5, 32, include_padding=True)
        sf = _ue8m0_from_exponents(exps)
        packed_i32 = pack_ue8m0_to_int32(sf)
        packed_slot = pack_ue8m0_to_fp32_slot(sf)
        self.assertEqual(packed_slot.dtype, paddle.float32)
        self.assertTrue(
            np.array_equal(_as_int32_bits(packed_slot), packed_i32.numpy()),
            "fp32 slot carrier is not byte-identical to the int32 carrier",
        )
        # Both carriers must unpack to the same fp32 scale.
        from_i32 = unpack_ue8m0_scale(packed_i32, 32)
        from_slot = unpack_ue8m0_scale(packed_slot, 32)
        self._assert_bit_equal(
            from_slot, from_i32, "int32 and fp32-slot carriers diverge"
        )
        self._assert_bit_equal(from_slot, sf, "fp32-slot round-trip not exact")

    def test_non_contiguous_input(self):
        # A column slice of a wider tensor is non-contiguous; pack must handle it
        # (it contiguifies internally) and still match the contiguous result.
        exps = self._random_exponents(6, 8)
        wide = _ue8m0_from_exponents(
            np.concatenate([exps, self._random_exponents(6, 8)], axis=1)
        )
        sf_nc = wide[:, :8]
        self.assertFalse(sf_nc.is_contiguous())
        sf_contig = _ue8m0_from_exponents(exps)
        packed_nc = pack_ue8m0_to_int32(sf_nc)
        packed_contig = pack_ue8m0_to_int32(sf_contig)
        self.assertTrue(
            np.array_equal(packed_nc.numpy(), packed_contig.numpy()),
            "non-contiguous pack differs from contiguous pack",
        )
        recovered = unpack_ue8m0_scale(packed_nc, 8)
        self._assert_bit_equal(
            recovered, sf_contig, "non-contiguous round-trip not bit-exact"
        )

    def test_empty_rows(self):
        sf = paddle.zeros([0, 16], dtype=paddle.float32)
        packed = pack_ue8m0_to_int32(sf)
        self.assertEqual(list(packed.shape), [0, 16 // GROUPS_PER_SLOT_WORD])
        recovered = unpack_ue8m0_scale(packed, 16)
        self.assertEqual(list(recovered.shape), [0, 16])
        self.assertEqual(recovered.dtype, paddle.float32)


if __name__ == "__main__":
    unittest.main()
