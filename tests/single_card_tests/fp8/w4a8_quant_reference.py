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

"""CPU numerical reference for the W4A8 1x32 quantization ops (test-only).

Production ``fp8_utils`` only exposes the fused SM100 CUDA ops
(``mxfp8_quantize_1x32`` / ``mxfp4_quantize_1x32`` etc.); the python per-op
reference implementation was removed from production code. These byte-for-byte
copies live here as an independent, non-circular CPU ground truth so the W4A8
GPU tests can validate the fused kernels. Not imported by any production module.
"""

import paddle
import paddle.nn.functional as F

from paddlefleet.transformer.moe.fp8_utils import _stack_expert_weights


def ceil_to_ue8m0(x):
    """将 scale 向上取整到 2 的幂（ue8m0）。

    等价 DeepGEMM math.py 的 ceil_to_ue8m0：
        bits = x.abs().float().view(torch.int32)
        exp = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).bool().int()
        return (exp.clamp(1, 254) << 23).view(torch.float)
    """
    bits = x.abs().astype("float32").view("int32")
    mask_ff = paddle.full_like(bits, 0xFF)
    mask_7fffff = paddle.full_like(bits, 0x7FFFFF)
    shift_23 = paddle.full_like(bits, 23)
    exp = ((bits >> shift_23) & mask_ff) + ((bits & mask_7fffff) != 0).astype(
        "int32"
    )
    return (exp.clip(1, 254) << shift_23).view("float32")


def _quantize_to_fp4_e2m1(x):
    """fp32 -> e2m1 4bit code（int32, 0~15），同 DeepGEMM _quantize_to_fp4_e2m1。

    e2m1 可表示值 {0, 0.5, 1, 1.5, 2, 3, 4, 6}，
    rounding 边界（相邻值中点）：0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0，
    bit3 为符号位。
    """
    ax = x.abs()
    code = paddle.zeros(ax.shape, dtype="int32")
    for boundary in (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0):
        code = code + (ax > boundary).astype("int32")
    sign = ((x < 0) & (code != 0)).astype("int32")
    shift_3 = paddle.full_like(code, 3)
    return code | (sign << shift_3)


def quant_blockwize(
    x, quant_method="1x32", quant_dtype="fp8", using_ue8m0_scale=True
):
    """普通 blockwise 量化散算子（对齐 fp8_quant_blockwise 的 attr 风格）。

    参考 DeepGEMM math.py 的 per_token_cast_to_fp8 / per_token_cast_to_fp4。

    Args:
        x: [M, K] bf16/fp32，K 必须能被块大小整除（fp4 时还需被 2 整除）
        quant_method: "1xN"，块大小（如 "1x32" / "1x128"）
        quant_dtype: "fp8"（e4m3fn）或 "fp4"（e2m1，两值打包成一个 int8）
        using_ue8m0_scale: scale 是否向上取整到 2 的幂（SM100 1D1D kernel
            内部会把 fp32 scale cast 成 packed ue8m0，故 w4a8 必须为 True）

    Returns:
        (q, sf):
          fp8: q [M, K] float8_e4m3fn, sf [M, K/gran_k] fp32
          fp4: q [M, K/2] int8（低 nibble 存前一个元素）, sf [M, K/gran_k] fp32
    """
    gran_m, gran_k = (int(v) for v in quant_method.split("x"))
    assert x.ndim == 2, x.shape
    m, n = x.shape
    assert n % gran_k == 0, f"K={n} 必须是 {gran_k} 的倍数"

    if gran_m > 1:
        # 2D tile 量化（如 128x128，权重量化，同 CUDA fuse_stack_*_fp8_quant）
        assert quant_dtype == "fp8", "2D tile 量化仅支持 fp8"
        assert m % gran_m == 0, f"M={m} 必须是 {gran_m} 的倍数"
        tiles = (
            x.reshape([m // gran_m, gran_m, n // gran_k, gran_k])
            .transpose([0, 2, 1, 3])
            .astype("float32")
        )
        amax = tiles.abs().max(axis=[2, 3]).clip(min=1e-4)
        sf = amax / 448.0
        sf = ceil_to_ue8m0(sf) if using_ue8m0_scale else sf
        q = (
            (tiles * (1.0 / sf.unsqueeze(-1).unsqueeze(-1)))
            .transpose([0, 2, 1, 3])
            .reshape([m, n])
            .astype(paddle.float8_e4m3fn)
        )
        return q, sf

    if m == 0:
        if quant_dtype == "fp8":
            q = paddle.empty([0, n], dtype=paddle.float8_e4m3fn)
        else:
            q = paddle.empty([0, n // 2], dtype=paddle.int8)
        return q, paddle.empty([0, n // gran_k], dtype=paddle.float32)

    x_view = x.reshape([m, n // gran_k, gran_k]).astype("float32")
    x_amax = x_view.abs().max(axis=2).clip(min=1e-4)

    if quant_dtype == "fp8":
        sf = x_amax / 448.0
        sf = ceil_to_ue8m0(sf) if using_ue8m0_scale else sf
        q = (
            (x_view * (1.0 / sf.unsqueeze(2)))
            .reshape([m, n])
            .astype(paddle.float8_e4m3fn)
        )
        return q, sf

    if quant_dtype == "fp4":
        assert n % 2 == 0, n
        sf = x_amax / 6.0
        sf = ceil_to_ue8m0(sf) if using_ue8m0_scale else sf
        x_scaled = x_view * (1.0 / sf.unsqueeze(2))
        codes = _quantize_to_fp4_e2m1(x_scaled).reshape([m, n])
        # 打包：两个 4bit code -> 一个字节，低 nibble 存前一个元素
        # （同 DeepGEMM per_token_cast_to_fp4）
        codes2 = codes.reshape([m, n // 2, 2])
        mask_0f = paddle.full_like(codes2[:, :, 0], 0x0F)
        shift_4 = paddle.full_like(codes2[:, :, 0], 4)
        packed = (codes2[:, :, 0] & mask_0f) | (
            (codes2[:, :, 1] & mask_0f) << shift_4
        )
        # 按 bit pattern 转 int8（>127 的按补码回绕）
        packed = paddle.where(packed > 127, packed - 256, packed).astype("int8")
        return packed, sf

    raise ValueError(f"unsupported quant_dtype: {quant_dtype}")


def fuse_stack_fp8_quant_python(
    w, quant_method="1x32", quant_dtype="fp4", using_ue8m0_scale=True
):
    """fuse_stack_fp8_quant 的 python 散算子版本（不转置）。

    输入 w: [E, R, C] bf16 stacked 权重（或 tensor list，自动 stack），
    沿最后一维（GEMM 收缩维）做 blockwise 量化。
    返回: fp4 时 (w_q [E, R, C/2] int8, sf [E, R, C/gran_k] fp32)
    """
    w = _stack_expert_weights(w)
    e, r, c = w.shape
    q, sf = quant_blockwize(
        w.reshape([e * r, c]),
        quant_method=quant_method,
        quant_dtype=quant_dtype,
        using_ue8m0_scale=using_ue8m0_scale,
    )
    if not quant_method.startswith("1x"):
        # 2D tile（如 128x128）时保持 CUDA 融合算子的 2D 布局：
        # q [E*R, C], sf [E*R/gm, C/gk]
        return q, sf
    return (
        q.reshape([e, r, q.shape[-1]]),
        sf.reshape([e, r, sf.shape[-1]]),
    )


def fuse_stack_transpose_fp8_quant_python(
    w, quant_method="1x32", quant_dtype="fp4", using_ue8m0_scale=True
):
    """fuse_stack_transpose_fp8_quant 的 python 散算子版本。

    输入 w: [E, H0, H1]，先转置成 [E, H1, H0] 再沿最后一维量化。
    """
    w = _stack_expert_weights(w)
    return fuse_stack_fp8_quant_python(
        w.transpose([0, 2, 1]).contiguous(),
        quant_method=quant_method,
        quant_dtype=quant_dtype,
        using_ue8m0_scale=using_ue8m0_scale,
    )


def _weighted_swiglu_fp32(o1, probs, clamp_value=None):
    """fp32 全程计算 swiglu(o1) * probs（与 CUDA 融合算子内部精度一致）。

    CUDA 版 fuse_weighted_swiglu_fp8_quant(_clamp) 在 kernel 内用 fp32
    计算 silu(gate)*up*probs 后直接量化，不落 bf16 中间结果；python 版
    同样保持 fp32，才能与融合算子位级对齐。
    """
    x32 = o1.astype("float32")
    h = x32.shape[-1] // 2
    gate, up = x32[:, :h], x32[:, h:]
    if clamp_value is not None:
        gate = gate.clip(max=clamp_value)
        up = up.clip(min=-clamp_value, max=clamp_value)
    p = probs.astype("float32")
    if p.ndim == 1:
        # 端到端 forward 传入的 unzipped_probs 是 1-D [M]（CUDA 融合算子内部兼容），
        # python 版需显式广播为 [M, 1]
        p = p.reshape([-1, 1])
    return F.silu(gate) * up * p


def fuse_weighted_swiglu_fp8_quant_python(
    o1, probs, quant_method="1x32", using_ue8m0_scale=True
):
    """fuse_weighted_swiglu_fp8_quant 的 python 散算子版本。

    o2 = swiglu(o1) * probs（fp32 中间计算），再做 blockwise fp8 量化。
    quant_method="1x128" 且 using_ue8m0_scale=True 时与 CUDA 融合算子
    （using_pow2_scaling=True）逐位一致。
    返回: (o2_fp8 [M, H] float8_e4m3fn, sf [M, H/gran_k] fp32)
    """
    o2 = _weighted_swiglu_fp32(o1, probs)
    return quant_blockwize(
        o2,
        quant_method=quant_method,
        quant_dtype="fp8",
        using_ue8m0_scale=using_ue8m0_scale,
    )


def fuse_weighted_swiglu_fp8_quant_clamp_python(
    o1, probs, clamp_value, quant_method="1x32", using_ue8m0_scale=True
):
    """fuse_weighted_swiglu_fp8_quant_clamp 的 python 散算子版本。

    clamp 语义与 CUDA 版一致：gate 上界夹 min(x, c)，up 对称夹 clip(y, -c, c)。
    quant_method="1x128" 且 using_ue8m0_scale=True 时与 CUDA 融合算子逐位一致。
    """
    o2 = _weighted_swiglu_fp32(o1, probs, clamp_value)
    return quant_blockwize(
        o2,
        quant_method=quant_method,
        quant_dtype="fp8",
        using_ue8m0_scale=using_ue8m0_scale,
    )


def fused_act_dequant_python(x_fp8, sf):
    """fused_act_dequant 的 python 散算子版本（blockwise 反量化）。

    参考 DeepGEMM math.py 的 cast_back_from_fp8：
        group_idx = arange(n) // gran_k
        return x_fp8.float() * sf[:, group_idx]

    输入: x_fp8 [M, K] float8_e4m3fn, sf [M, K/gran_k] fp32
    返回: bf16 [M, K]
    """
    m, n = x_fp8.shape
    assert sf.shape[0] == m and n % sf.shape[1] == 0, (
        f"{x_fp8.shape} vs {sf.shape}"
    )
    gran_k = n // sf.shape[1]
    group_idx = paddle.arange(n, dtype="int64") // gran_k
    sf_expanded = paddle.index_select(sf.astype("float32"), group_idx, axis=1)
    return (x_fp8.astype("float32") * sf_expanded).astype("bfloat16")
