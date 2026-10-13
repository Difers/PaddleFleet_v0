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
"""W4A8 FP8 token-dispatch ("H1") scale carrier -- shared by DeepEP + HybridEP.

W4A8 activations are quantized 1x32 (fp8 e4m3 + a UE8M0 power-of-two fp32 scale,
see ``fp8_utils._mxfp8_quant`` / the ``CeilUe8m0`` kernel). The W4A8 mxfp8 dispatch path
quantizes the activation on the *source* tokens before the all-to-all and sends
fp8 (half the token bytes of bf16) plus the per-row scale. Both dispatchers move
the scale as one fp32 word per 128 hidden elements:

* **DeepEP** carries an ``int32`` scale companion ``[T, H/128]``. The dtype is
  the int32 DeepEP already moves for packed UE8M0 scales (so ``moe_permute``
  keys the per-row gather on ``scale.dtype == int32``); the ``[T, H/128]``
  width matches the SonicMoE 1x32 E8M0 carrier, NOT the FP8 1x128 UE8M0 one
  (which is ``[T, H/512]`` -- four 1x128 scales per word).
* **HybridEP** carries an ``fp32`` scaling-factor slot ``[T, H/128]`` (the only
  ``use_fp8=True`` ``dispatch_with_permute`` buffer).

Because every 1x32 scale is an exact power of two (sign 0, mantissa 0, biased
exponent in [1, 254]), it is fully described by its biased-exponent byte. We pack
the ``H/32`` exponent bytes of a row -- four per word -- into the ``H/128`` slot
the dispatcher already moves, and rebuild the exact float32 power of two on the
receiver with ``byte << 23``. The two carriers hold byte-identical payloads: the
only difference is the container dtype (``int32`` vs ``fp32``), so the receiver
views the HybridEP slot back to ``int32`` and both paths unpack with the same
code.

Everything here is plain integer bit manipulation (sign 0 + mantissa 0 means the
exponent byte is lossless), so pack/unpack is bit-exact for every exponent byte
(including 0, which is what zero-filled padding rows carry) -- nothing is ever
rounded, requantized or read as a float value in between. The module imports only
``paddle`` so it unit-tests on a CPU-only box.
"""

import paddle

# One 1x32 UE8M0 group per fp32 scale element of the W4A8 activation quant. Kept
# as a standalone constant (not imported from fp8_utils) so this module stays
# CPU-only importable for its unit tests; must match fp8_utils.W4A8_QUANT_BLOCK
# and the mxfp8_quantize_1x32 kernel.
W4A8_QUANT_BLOCK = 32
# Both dispatchers move one fp32 scale word per 128 hidden elements.
HYBRIDEP_SCALE_BLOCK = 128
# 1x32 exponent bytes carried by one scale word (4 bytes).
GROUPS_PER_SLOT_WORD = HYBRIDEP_SCALE_BLOCK // W4A8_QUANT_BLOCK


def _packed_groups(sf):
    """Validate a UE8M0 scale ``[T, G]`` and return G (= hidden/32)."""
    if sf.dtype != paddle.float32:
        raise TypeError(f"W4A8 UE8M0 scale must be float32, got {sf.dtype}")
    if len(sf.shape) != 2:
        raise ValueError(f"W4A8 UE8M0 scale must be rank-2, got {sf.shape}")
    groups = sf.shape[-1]
    if groups % GROUPS_PER_SLOT_WORD != 0:
        raise ValueError(
            f"W4A8 scale groups (hidden/{W4A8_QUANT_BLOCK}) must be a multiple "
            f"of {GROUPS_PER_SLOT_WORD} (hidden a multiple of "
            f"{HYBRIDEP_SCALE_BLOCK}), got {groups}"
        )
    return groups


def pack_ue8m0_to_int32(sf):
    """Pack fp32 UE8M0 scales ``[T, G]`` into an int32 carrier ``[T, G // 4]``.

    Group ``j`` of a row lands in byte ``j % 4`` of word ``j // 4`` -- the exact
    layout DeepEP's int32 UE8M0 scale companion uses (and ``moe_permute``
    gathers). The returned tensor is only a byte container; its int32 values are
    arbitrary and must not be read as numbers until :func:`unpack_ue8m0_scale`.
    """
    groups = _packed_groups(sf)
    if sf.shape[0] == 0:
        return paddle.empty(
            [0, groups // GROUPS_PER_SLOT_WORD], dtype=paddle.int32
        )
    if not sf.is_contiguous():
        sf = sf.contiguous()
    # sign bit is 0 for every UE8M0 scale, so the shift leaves exactly the biased
    # exponent in [0, 255]; the uint8 cast is then exact.
    exponent = (sf.view(paddle.int32) >> 23).astype(paddle.uint8)
    return exponent.view(paddle.int32)


def pack_ue8m0_to_fp32_slot(sf):
    """Pack fp32 UE8M0 scales ``[T, G]`` into a fp32 carrier ``[T, G // 4]``.

    Byte-identical to :func:`pack_ue8m0_to_int32`; only the container dtype
    differs so the result can ride HybridEP's fp32 ``scaling_factor`` slot. The
    fp32 values are arbitrary bit patterns (some NaN/Inf when read as floats) and
    must never be touched by a float op before the receiver views them back to
    int32.
    """
    return pack_ue8m0_to_int32(sf).view(paddle.float32)


def unpack_ue8m0_scale(packed, num_groups):
    """Inverse of the pack: carrier ``[M, G // 4]`` -> fp32 ``[M, G]``.

    Accepts an int32 (DeepEP) or fp32 (HybridEP slot) carrier -- both are viewed
    as raw bytes, so the dtype is irrelevant. Each exponent byte ``e`` becomes the
    float32 whose bits are ``e << 23`` (sign 0, mantissa 0), i.e. the exact power
    of two; a zero byte (zero-filled padding rows) gives +0.0, which is finite.
    The result is contiguous float32, bit-identical to the scale ``_mxfp8_quant``
    produced before packing.
    """
    if packed.dtype not in (paddle.int32, paddle.float32):
        raise TypeError(
            f"packed W4A8 scale carrier must be int32/float32, got {packed.dtype}"
        )
    if packed.shape[-1] * GROUPS_PER_SLOT_WORD != num_groups:
        raise ValueError(
            f"packed W4A8 scale carrier width {packed.shape[-1]} * "
            f"{GROUPS_PER_SLOT_WORD} != expected {num_groups} groups "
            f"(hidden/{W4A8_QUANT_BLOCK})"
        )
    if packed.shape[0] == 0:
        return paddle.empty([0, num_groups], dtype=paddle.float32)
    if not packed.is_contiguous():
        packed = packed.contiguous()
    exponent = packed.view(paddle.uint8).astype(paddle.int32)
    return (exponent << 23).view(paddle.float32)
