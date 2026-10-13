# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025 DeepSeek
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
"""FP8 Utils"""

import numpy
import paddle
import paddle.nn.functional as F
from paddle.base.framework import EagerParamBase

from paddlefleet.fusions.fused_swiglu_scale import (
    fused_swiglu_scale_backward,
    fused_swiglu_scale_forward,
)
from paddlefleet.train_infer_consistent_ops.ffn_act import (
    dequant_dispatched_hidden_bf16,
    requant_swiglu_output,
    scatter_dispatched_hidden_bf16,
)
from paddlefleet.train_infer_consistent_ops.inspect_util import inspect_tensor
from paddlefleet.train_infer_consistent_ops.permute import (
    canonical_rows,
    scatter_canonical_rows,
)

from .mxfp8_dispatch import (
    pack_ue8m0_to_fp32_slot,
    pack_ue8m0_to_int32,
    unpack_ue8m0_scale,
)

try:
    from paddlefleet_ops import (
        deep_gemm as paddlefleet_deep_gemm,
        fuse_stack_fp8_quant,
        fuse_stack_transpose_fp8_quant,
        fuse_weighted_swiglu_fp8_quant,
        fuse_weighted_swiglu_fp8_quant_clamp,
        fused_swiglu_weighted_clamp_bwd,
    )
except (ImportError, RuntimeError):
    pass

try:
    from paddlefleet_ops import (
        mxfp4_stack_quantize_1x32,
        mxfp8_dequantize_1x32,
        mxfp8_quantize_1x32,
        mxfp8_weighted_swiglu_quantize_1x32,
    )

    HAS_W4A8_FUSED_QUANT = True
except (ImportError, AttributeError, RuntimeError):
    HAS_W4A8_FUSED_QUANT = False

# 优先从 paddlefleet_ops 导入（算子已重命名为 paddlefleet_fused_swiglu_probs_bwd 避免冲突），
# 仅在 paddlefleet_ops 中不存在时回退到旧的 FusedQuantOps。
try:
    from paddlefleet_ops import (
        fused_swiglu_probs_bwd as _fused_swiglu_probs_bwd,
    )

    USE_INPLACE_SWIGLU_BWD = True
except (ImportError, AttributeError, RuntimeError):
    try:
        import FusedQuantOps as _FQO

        _fused_swiglu_probs_bwd = _FQO.fused_swiglu_probs_bwd
        USE_INPLACE_SWIGLU_BWD = True
    except (ImportError, AttributeError):
        _fused_swiglu_probs_bwd = None
        USE_INPLACE_SWIGLU_BWD = False

# Activation names the expert nodes know how to compute. Every down-projection
# dispatch special-cases the ones it handles and falls through to SwiGLU, so an
# unrecognised name has to be rejected rather than allowed to reach a branch.
SUPPORTED_ACTIVATION_TYPES = ("swiglu", "geglu", "situ")
# Of those, the ones the fp8 down projection implements. GeGLU is bf16-only.
FP8_ACTIVATION_TYPES = ("swiglu", "situ")

try:
    from paddle.nn.functional import swiglu
except ImportError:

    def swiglu(x, y=None):
        """
            使用swiglu函数对输入的张量进行Sigmoid-weighted Linear Unit操作，并返回结果。
        如果没有提供y参数，则将输入的张量分割成两个部分，一个是Sigmoid函数的输入，另一个是Linear Unit的输入。
        否则，将x视为Sigmoid函数的输入，y视为Linear Unit的输入。

        Args:
            x (Tensor): 要进行Sigmoid-weighted Linear Unit操作的输入张量，其形状可以是任意维度。（默认值：None）
            y (Tensor, optional): 要与x相乘的常数项，其形状应该和x相同。（默认值：None）

        Returns:
            Tensor: Sigmoid-weighted Linear Unit后的输出张量，其形状与x相同。

        Raises:
            TypeError: 当x不是Tensor类型时会抛出此类型错误。
            ValueError: 当x和y的形状不匹配时会抛出此值错误。
        """
        if y is None:
            x, y = paddle.chunk(x, chunks=2, axis=-1)
        return F.silu(x) * y


try:
    from paddlefleet_ops import deep_gemm
except:
    pass

try:
    from paddle.incubate.nn.functional import fused_transpose_wlch_split_quant
except ImportError:
    fused_transpose_wlch_split_quant = None

from functools import partial

from paddle.distributed.fleet.meta_parallel.zero_bubble_utils import (
    WeightGradStore,
)

from paddlefleet.transformer.activations import (
    situ_glu_scale_backward,
    situ_glu_scale_forward,
)

__all__ = [
    "ExpertsGroupGemmContiguousNode",
]


FP8_ALIGN = 128


def moe_token_padding_alignment(
    *, use_fp8_mlp: bool, moe_grouped_gemm: bool, use_accuracy_compatible: bool
) -> int:
    # == [MG accuracy-alignment diff] per-expert token padding alignment ==
    # Only when use_accuracy_compatible=True and on the pure-bf16 non-grouped_gemm
    #   path do we skip padding (return 1), so each expert's GEMM M dim == the real
    #   token count and cuBLAS picks the same algorithm as MG SequentialMLP. In all
    #   other cases (including when accuracy compatible is off) align to FP8_ALIGN
    #   to preserve the original behavior.
    if use_accuracy_compatible and not use_fp8_mlp and not moe_grouped_gemm:
        return 1
    return FP8_ALIGN


def _get_fp8_weight_and_scale(
    weight, transpose=False, num_expert=None, use_ue8m0=None
):
    """_get_fp8_weight_and_scale"""
    fp8_weight, fp8_scale = (
        weight.fp8_weight_stacked,
        weight.fp8_scale_stacked,
    )

    if transpose:
        if (
            hasattr(weight, "fp8_weight_stacked_transpose")
            and weight.fp8_weight_stacked_transpose is not None
        ):
            fp8_weight = weight.fp8_weight_stacked_transpose
            fp8_scale = weight.fp8_scale_stacked_transpose
        else:
            # 只有非转置版，on-the-fly reshape+transpose
            assert fp8_weight.shape[0] % weight.shape[0] == 0
            assert fp8_weight.ndim == 2
            if num_expert:
                expert_num = num_expert
            else:
                expert_num = fp8_weight.shape[0] // weight.shape[0]

            def transpose_tensor(tensor):
                assert tensor.ndim == 2
                h0 = tensor.shape[0] // expert_num
                h1 = tensor.shape[1]
                tensor = tensor.reshape([expert_num, h0, h1])
                return (
                    tensor.contiguous()
                    .transpose([0, 2, 1])
                    .reshape([-1, h0])
                    .contiguous()
                )

            transpose_scale = (
                weight.fp8_scale_stacked_transpose
                if use_ue8m0
                else transpose_tensor(fp8_scale)
            )
            fp8_weight, fp8_scale = (
                transpose_tensor(fp8_weight),
                transpose_scale,
            )

    return fp8_weight, fp8_scale


def fused_stack_quant_without_cache(
    expert_weight_list, transpose=False, use_ue8m0=False
):
    use_pow2_scale = False
    if paddle.device.cuda.get_device_capability()[0] == 10:
        # Blackwell GPUs require the use of pow2_scales quantization.
        use_pow2_scale = True
    if transpose:
        w, scale = fuse_stack_transpose_fp8_quant(
            expert_weight_list,
            use_pow2_scale,
            use_ue8m0,
            use_ue8m0,
        )
    else:
        w, scale = fuse_stack_fp8_quant(
            expert_weight_list,
            use_pow2_scale,
            use_ue8m0,
            use_ue8m0,
        )

    if use_ue8m0:
        scale = scale.T.contiguous()
    return w, scale


def fused_stack_quant(
    expert_weight_list, transpose=False, use_ue8m0=False, num_expert=None
):
    if hasattr(expert_weight_list[0], "fp8_weight_stacked"):
        w, scale = _get_fp8_weight_and_scale(
            expert_weight_list[0],
            transpose=transpose,
            num_expert=num_expert,
            use_ue8m0=use_ue8m0,
        )
    else:
        w, scale = fused_stack_quant_without_cache(
            expert_weight_list, transpose, use_ue8m0
        )
    return w, scale


def tilewise_quant(x):
    """
    Tile-wise FP8 quantization: quantize input tensor to FP8 with per-tile (1x128) scaling.
    """
    pow_2_scales = False
    if paddle.device.cuda.get_device_capability()[0] == 10:
        # Blackwell GPUs require the use of pow2_scales quantization.
        pow_2_scales = True
    if x.shape[0] > 0:
        return paddle.incubate.nn.functional.fp8_quant_blockwise(
            x,
            output_scale_transpose=False,
            quant_method="1x128",
            input_transpose=False,
        )
    else:
        shape = list(x.shape)
        x_fp8 = paddle.empty(x.shape, dtype=paddle.float8_e4m3fn)
        assert shape[-1] % FP8_ALIGN == 0, shape
        shape[-1] //= FP8_ALIGN
        x_scale = paddle.empty(shape, dtype=paddle.float32)
        return x_fp8, x_scale


# ---------------------------------------------------------------------------
# W4A8 (1x32 MXFP4 weights + 1x32 MXFP8 activations) online-quant helpers.
# Thin wrappers over the fused SM100 CUDA ops (mxfp8_quantize_1x32 /
# mxfp4_quantize_1x32 etc.). W4A8 always runs the fused path; the python
# per-op reference implementation has been removed.
# ---------------------------------------------------------------------------

W4A8_QUANT_BLOCK = 32


def _stack_expert_weights(w):
    """Stack expert weights (list / stacked tensor) into a single [E, H0, H1]."""
    if isinstance(w, (list, tuple)):
        w = w[0] if len(w) == 1 else paddle.stack(w, axis=0)
    assert w.ndim == 3, w.shape
    return w


def _mxfp8_quant(x):
    return mxfp8_quantize_1x32(x)


def _w4a8_pack_scale_for_deepep(x_sf):
    """Pack a float32 1x32 UE8M0 activation scale ``[T, H/32]`` into DeepEP's
    int32 scale companion ``[T, H/128]`` (4 exponent bytes per word).

    Same int32 dtype as DeepEP's packed UE8M0 scale companion, so its dispatch
    and ``moe_permute``'s per-row scale gather move it unchanged. The ``[T,
    H/128]`` width matches the 1x32 E8M0 carrier, not the H/512-wide FP8 1x128
    UE8M0 one.
    """
    return pack_ue8m0_to_int32(x_sf)


def _w4a8_quant_pack_for_hybridep_dispatch(x):
    """Pre-dispatch W4A8 activation quant for HybridEP fp8 dispatch.

    Runs the *same* 1x32 quant the expert node runs after the all-to-all
    (``_mxfp8_quant(x)``) on the source tokens, then repacks the UE8M0 fp32
    scales ``[T, H/32]`` into HybridEP's fp32 ``scaling_factor`` slot ``[T,
    H/128]`` (4 exponent bytes per word). The slot is a byte container built
    directly in [T, H/128] token-major layout -- it must NOT be transposed like
    the 1x128 ``output_scale_transpose`` carrier. The 1x32 quant is row-local and
    dispatch/permute are pure per-row copies, so the receiver recovers exactly
    the (fp8, scale) the post-dispatch quant would have produced.
    """
    x_fp8, x_sf = _mxfp8_quant(x)
    return x_fp8, pack_ue8m0_to_fp32_slot(x_sf)


def _mxfp4_stack_quant(weights, transpose, use_ue8m0=True):
    if not use_ue8m0:
        raise ValueError(
            "mxfp4 stack quant requires use_ue8m0=True; "
            "the fused SM100 kernels use UE8M0 scales"
        )
    weights = _stack_expert_weights(weights)
    return mxfp4_stack_quantize_1x32(weights, transpose)


def _mxfp8_weighted_swiglu_quant(o1, probs, clamp_value=None):
    return mxfp8_weighted_swiglu_quantize_1x32(
        o1,
        probs,
        0.0 if clamp_value is None else float(clamp_value),
    )


def _mxfp8_dequant(x_fp8, sf):
    return mxfp8_dequantize_1x32(x_fp8, sf)


def split_group_gemm(
    x_fp8, x_scale, w_fp8, w_scale, tokens_per_expert, gemm_out, use_ue8m0=False
):
    """
    将输入的张量分割成多个小的矩阵乘

    Args:
        x_fp8 (paddle.Tensor, shape=(N, T)): 需要进行矩阵乘法的FP8格式的张量。
        x_scale (paddle.Tensor, shape=(N, T)): 与x_fp8对应的缩放因子。
        w_fp8 (List[paddle.Tensor], length=6): 包含6个FP8格式的张量，每个张量代表一个专家的权重。
        w_scale (List[paddle.Tensor], length=6): 与w_fp8对应的缩放因子。
        tokens_per_expert (List[int], length=6): 每个专家处理的token数量。
        gemm_out (paddle.Tensor, shape=(N, T)): 存储结果的张量。
        use_ue8m0 (bool): Whether to use UE8M0 format scales (TMA aligned).

    Returns:
        paddle.Tensor, shape=(N, T): 返回计算结果存储在gemm_out中的张量。
    """
    start_idx = 0
    for i, token_num in enumerate(tokens_per_expert):
        if token_num == 0:
            continue
        end_idx = start_idx + token_num

        x_i = x_fp8[start_idx:end_idx]
        x_scale_tma_align = x_scale[start_idx:end_idx].T.contiguous().T

        if use_ue8m0:
            w_scale_tma_align = w_scale[i].T.contiguous().T
        else:
            w_scale_tma_align = w_scale[i].contiguous()

        deep_gemm.fp8_gemm_nt(
            (x_i, x_scale_tma_align),
            (w_fp8[i].contiguous(), w_scale_tma_align),
            gemm_out[start_idx:end_idx],
        )

        start_idx = end_idx

    return gemm_out


def has_config(config_map, key):
    """
    判断给定的配置字典中是否存在指定键，并且该键对应的值不为空。

    Args:
        config_map (Optional[Dict[str, Any]]): 配置字典，可以为None。
        key (str): 需要查找的键名。

    Returns:
        bool: 如果配置字典不为None，且包含指定键，且该键对应的值不为空，则返回True；否则返回False。
    """
    return bool(
        config_map is not None and key in config_map and config_map[key]
    )


def expert_weights_all_frozen(weights):
    """True when every expert weight in ``weights`` is a frozen parameter.

    The MoE expert backward writes weight gradients straight into
    ``main_grad`` / ``grad`` instead of returning them through autograd, so
    ``stop_gradient`` is not honored automatically. Callers use this to skip the
    wgrad GEMMs and their fp32 buffers when the experts are frozen (for example
    DSv4 phase 2, ``train_indexer_only``).

    ``weights`` is a stacked parameter (grouped path), a list of per-expert
    parameters (split path), or a **per-expert view** of a stacked parameter
    (subbatch / sliced deep_gemm path). A view's own ``stop_gradient`` is always
    True even when its parent parameter is trainable, so each entry is first
    dereferenced through ``_parent`` -- stamped by :func:`slice_expert_weight` and
    stored by :class:`_PerExpertWeightView`. Only an ``EagerParamBase`` then
    counts as frozen; anything else keeps the original behavior so no gradient is
    silently dropped. Mixed groups are treated as not frozen.
    """
    if weights is None:
        return False
    if not isinstance(weights, (list, tuple)):
        weights = [weights]
    entries = [getattr(w, "_parent", w) for w in weights if w is not None]
    return bool(entries) and all(
        isinstance(w, EagerParamBase) and w.stop_gradient for w in entries
    )


def kitchen_gemm(
    x_fp8,
    x_scale,
    w_fp8,
    w_scale,
    is_a_1d_scaled,
    is_b_1d_scaled,
    out=None,
    rtn_dtype=paddle.bfloat16,
):
    # if USE_DS_GEMM:
    #     if out is None:
    #         out = paddle.zeros([x_fp8.shape[0], w_fp8.shape[0]], rtn_dtype)
    #     if numpy.prod(x_fp8.shape) != 0 and numpy.prod(w_fp8.shape) != 0:
    #         deep_gemm.wgrad_gemm_fp8_fp8_fp32_nt((x_fp8, x_scale), (w_fp8, w_scale), out, num_sms=get_sm_num())
    #     return out

    if out is not None:
        accumulate = True
        out_dtype = out.dtype
    else:
        accumulate = False
        out_dtype = rtn_dtype
    if numpy.prod(x_fp8.shape) != 0 and numpy.prod(w_fp8.shape) != 0:
        y = paddle.incubate.nn.functional.fp8_gemm_blockwise(
            a=x_fp8,
            a_decode_scale=x_scale,
            b=w_fp8,
            b_decode_scale=w_scale,
            out_dtype=out_dtype,
            out=out,
            accumulate=accumulate,
            use_split_accumulator=True,
            is_a_1d_scaled=is_a_1d_scaled,
            is_b_1d_scaled=is_b_1d_scaled,
        )
    else:
        y = paddle.zeros([x_fp8.shape[0], w_fp8.shape[0]], out_dtype)
        if out is not None:
            out = out + y
            return out

    return y


def slice_expert_weight(parent_weight, local_id):
    """Per-expert view ``[1, K, N]`` of a stacked expert weight.

    ``_slice`` returns a raw view whose ``stop_gradient`` is always True, even when
    the parent parameter is trainable, so keep a pointer to the parameter it came
    from. :func:`expert_weights_all_frozen` dereferences that ``_parent`` to decide
    whether the expert is frozen. Every per-expert slicing site must go through
    here, otherwise that path silently loses the frozen-expert wgrad skip.
    """
    view = parent_weight._slice(local_id, local_id + 1)
    view._parent = parent_weight
    return view


class _PerExpertWeightView:
    """A lightweight view into a single expert's slice of the stacked fp8 weight.

    After offline fp8 quant, the original bf16 weight storage is cleared.
    This view provides the same interface that ``_get_fp8_weight_and_scale``
    expects (``fp8_weight_stacked``, ``fp8_scale_stacked``, etc.) by slicing
    the parent's stacked fp8 tensors for a single expert.

    The ``shape`` property reports ``[1, K, N]`` so that downstream code
    (e.g., ``_get_fp8_weight_and_scale``) correctly infers ``num_expert = 1``.
    """

    def __init__(self, parent_weight, local_id, num_experts):
        self._parent = parent_weight
        self._local_id = local_id
        self._num_experts = num_experts
        self._shape = [1, *list(parent_weight.shape[1:])]

    def _slice_stacked(self, attr_name):
        """Slice a 2D stacked tensor [E*rows, cols] to this expert's rows."""
        t = getattr(self._parent, attr_name, None)
        if t is None:
            return None
        rows_per_expert = t.shape[0] // self._num_experts
        start = self._local_id * rows_per_expert
        return t._slice(start, start + rows_per_expert)

    @property
    def shape(self):
        return self._shape

    def __len__(self):
        return self._shape[0]

    @property
    def dtype(self):
        return self._parent.dtype

    @property
    def fp8_weight_stacked(self):
        return self._slice_stacked("fp8_weight_stacked")

    @property
    def fp8_scale_stacked(self):
        return self._slice_stacked("fp8_scale_stacked")

    @property
    def fp8_weight_stacked_transpose(self):
        return self._slice_stacked("fp8_weight_stacked_transpose")

    @property
    def fp8_scale_stacked_transpose(self):
        return self._slice_stacked("fp8_scale_stacked_transpose")

    @property
    def main_grad(self):
        mg = getattr(self._parent, "main_grad", None)
        if mg is None:
            return None
        return mg._slice(self._local_id, self._local_id + 1)

    @main_grad.setter
    def main_grad(self, value):
        # When backward tries to allocate main_grad, ensure parent has it
        if getattr(self._parent, "main_grad", None) is None:
            self._parent.main_grad = paddle.zeros(
                self._parent.shape, dtype=paddle.float32
            )

    @property
    def grad(self):
        g = self._parent.grad
        if g is None:
            return None
        return g._slice(self._local_id, self._local_id + 1)

    @grad.setter
    def grad(self, value):
        pass

    def _apply_backward_hook(self):
        if hasattr(self._parent, "_apply_backward_hook"):
            self._parent._apply_backward_hook()

    @property
    def stop_gradient(self):
        return self._parent.stop_gradient


class _PerExpertWeightProxy:
    """Proxy for grouped_gemm_experts that provides per-expert weight views.

    Holds ``weight1`` and ``weight2`` as ``_PerExpertWeightView`` instances
    that lazily slice from the parent's fp8 stacked weights.
    """

    def __init__(self, parent, local_id):
        num_experts = parent.weight1.shape[0]
        self.weight1 = _PerExpertWeightView(
            parent.weight1, local_id, num_experts
        )
        self.weight2 = _PerExpertWeightView(
            parent.weight2, local_id, num_experts
        )


def _check_expert_dw_deferral_supported(
    defer_up_gate,
    defer_down,
    moe_deep_gemm,
    subbatch_token_num,
    use_fp8_mlp,
    use_bf16_gemm_weight_grad,
):
    """Fail fast when a requested expert dW deferral cannot be honoured.

    Both points need the deep_gemm weight-grad path, because that is the only
    branch of ``bf16_weight_grad`` with a ``WeightGradStore`` call; the
    ``batched_gemm`` fallback would ignore the request and compute inline, so
    the feature would be a silent no-op.

    FP8 expert compute additionally needs its BF16 weight-gradient fallback.
    Native FP8 weight-gradient kernels compute inline and do not enqueue work
    on ``WeightGradStore``.

    down_proj additionally needs a non-subbatch backward: a deferred dw2 keeps
    reading ``out_grad``, so dx must not reuse that buffer. Outside the
    subbatch loop we give dx its own buffer, but the loop *needs* the alias:
    it assembles dx by writing each slice back into ``out_grad`` and returns
    ``out_grad`` as dx (``assert tmp_dx is tmp_out_grad`` in
    ``backward_impl_subbatch``), so there the alias cannot be broken.
    """
    if defer_up_gate or defer_down:
        if not moe_deep_gemm:
            raise ValueError(
                "p2p_overlap_dw_calc requests an expert dW deferral "
                "(moe_expert_up_gate_proj / moe_expert_down_proj), but "
                "moe_deep_gemm=False. Only the deep_gemm weight-grad path can "
                "queue dW on WeightGradStore, so the request would be a silent "
                "no-op. Set moe_deep_gemm=True or drop the point."
            )
    if (
        (defer_up_gate or defer_down)
        and use_fp8_mlp
        and not use_bf16_gemm_weight_grad
    ):
        raise ValueError(
            "p2p_overlap_dw_calc requests an expert dW deferral "
            "(moe_expert_up_gate_proj / moe_expert_down_proj), but "
            "fp8_wgrad=True selects the native FP8 weight-gradient path. "
            "That path computes dW inline and cannot queue work on "
            "WeightGradStore. Set fp8_wgrad=False or drop the point."
        )
    if defer_down:
        if subbatch_token_num is not None:
            raise ValueError(
                "p2p_overlap_dw_calc requests moe_expert_down_proj, but "
                f"moe_subbatch_token_num_after_dispatch={subbatch_token_num}. "
                "The subbatch backward returns out_grad itself as dx, so a "
                "deferred dw2 would read the overwritten buffer and produce a "
                "wrong gradient. Unset moe_subbatch_token_num_after_dispatch "
                "or drop the point."
            )


class ExpertsGroupGemmContiguousNode:
    """ExpertsGroupGemmContiguousNode"""

    def __init__(
        self,
        custom_map,
        recompute_moe_gate_up=False,
        dequant_input=False,
        group=None,
        name="experts_group_gemm_contiguous_node",
        expert_id=None,
        moe_subbatch_token_num_after_dispatch=None,
        use_bf16_gemm_weight_grad=False,
        use_fp8_mlp=True,
        moe_deep_gemm=False,
        use_ue8m0=False,
        defer_expert_up_gate_dw=False,
        defer_expert_down_dw=False,
        moe_expert_fusion=False,
        clamp_value=None,
        activation_type="swiglu",
        use_accuracy_compatible=False,
        use_w4a8=False,
        w4a8_route_factor_post_w2=None,
    ):
        """
            Initializes the experts group gemm contiguous node.

        Args:
            custom_map (CustomMapping): Custom mapping for the model.
            recompute_moe_gate_up (bool, optional): Whether to recompute forward gate up. Defaults to False.
            dequant_input (bool, optional): Whether to dequantize input. Defaults to False.
            name (str, optional): Name of the node. Defaults to "experts_group_gemm_contiguous_node".
            activation_type (str, optional): Activation function type. "swiglu", "geglu", or "situ". Defaults to "swiglu".
        """
        if moe_deep_gemm and expert_id is not None:
            # Per-expert node for deep_gemm: slice stacked weight to [1, K, N]
            parent = custom_map.grouped_gemm_experts
            moe_rank = getattr(custom_map, "moe_rank", 0)
            num_experts_per_device = getattr(
                custom_map, "num_experts_per_device", parent.weight1.shape[0]
            )
            local_id = expert_id - moe_rank * num_experts_per_device
            if hasattr(parent.weight1, "fp8_weight_stacked"):
                # Offline quant: bf16 weight may have been cleared, use proxy
                self.grouped_gemm_experts = _PerExpertWeightProxy(
                    parent, local_id
                )
            else:
                # Normal: bf16 weight is valid, slice directly
                sliced = type("_SlicedGroupedExpert", (), {})()
                sliced.weight1 = slice_expert_weight(parent.weight1, local_id)
                sliced.weight2 = slice_expert_weight(parent.weight2, local_id)
                sliced._parent = parent
                sliced._local_id = local_id
                self.grouped_gemm_experts = sliced
            self.experts = None
            moe_expert_fusion = True
        elif not moe_expert_fusion or (use_fp8_mlp and not moe_deep_gemm):
            if expert_id is None:
                self.experts = custom_map.experts
            else:
                self.experts = [custom_map.experts[expert_id]]
        else:
            self.grouped_gemm_experts = custom_map.grouped_gemm_experts
        self.expert_id = expert_id

        self.layer_number = getattr(custom_map, "layer_number", None)
        self.recompute_moe_gate_up = recompute_moe_gate_up
        self.dequant_input = dequant_input
        self.tokens_per_expert = None
        self.m_indices = None
        self.input = None
        self.input_fp8 = None
        self.input_scale = None
        self.o1 = None
        self.fp8_fused_ops_configs = {}
        self.group = group
        self.moe_subbatch_token_num_after_dispatch = (
            moe_subbatch_token_num_after_dispatch
        )
        if self.moe_subbatch_token_num_after_dispatch is not None:
            assert (
                self.moe_subbatch_token_num_after_dispatch > 0
                and self.moe_subbatch_token_num_after_dispatch % FP8_ALIGN == 0
            ), self.moe_subbatch_token_num_after_dispatch
        self.use_bf16_gemm_weight_grad = use_bf16_gemm_weight_grad
        self.use_fp8_mlp = use_fp8_mlp
        self.moe_deep_gemm = moe_deep_gemm
        self.use_ue8m0 = use_ue8m0
        self.is_split_group_gemm = not moe_expert_fusion
        self.defer_expert_up_gate_dw = defer_expert_up_gate_dw
        self.defer_expert_down_dw = defer_expert_down_dw
        self.moe_expert_fusion = moe_expert_fusion
        self.clamp_value = clamp_value
        if activation_type not in SUPPORTED_ACTIVATION_TYPES:
            # Every dispatch below this point special-cases the activations it
            # knows and lets everything else fall through to SwiGLU, so a typo
            # would silently train the wrong function. Reject the name once,
            # here, instead of at each branch.
            raise ValueError(
                "activation_type must be one of "
                f"{sorted(SUPPORTED_ACTIVATION_TYPES)}, but got "
                f"{activation_type!r}."
            )
        self.activation_type = activation_type
        config = getattr(custom_map, "config", None)
        self.activation_situ_beta = getattr(config, "activation_situ_beta", 1.0)
        self.activation_situ_linear_beta = getattr(
            config, "activation_situ_linear_beta", None
        )
        self.situ_glu_fusion = getattr(config, "situ_glu_fusion", False)
        self.use_accuracy_compatible = use_accuracy_compatible
        self.token_padding_alignment = moe_token_padding_alignment(
            use_fp8_mlp=use_fp8_mlp,
            moe_grouped_gemm=not self.is_split_group_gemm,
            use_accuracy_compatible=use_accuracy_compatible,
        )
        self.use_w4a8 = use_w4a8
        # Scalar the router deliberately did not fold into probs; applied after
        # the down projection so the FP8 activation quantizer sees the same
        # tensor the inference stack quantizes. None means the router folded it.
        self.w4a8_route_factor_post_w2 = w4a8_route_factor_post_w2
        if w4a8_route_factor_post_w2 is not None and not use_w4a8:
            # The router already dropped the factor from probs; only the W4A8
            # down path puts it back. Any other path would silently emit
            # outputs scaled down by routed_scaling_factor, so fail loudly.
            raise ValueError(
                "w4a8_route_factor_post_w2 is only supported on the W4A8 down "
                f"path, but use_w4a8={use_w4a8}"
            )
        if use_w4a8:
            try:
                capability = paddle.device.cuda.get_device_capability()
            except Exception as exc:
                raise RuntimeError(
                    "use_w4a8 requires a CUDA SM100 device; unable to query "
                    "the current device capability."
                ) from exc
            if capability[0] != 10:
                raise RuntimeError(
                    "use_w4a8 requires an SM100 (compute capability 10.x) "
                    f"device, got {capability[0]}.{capability[1]}."
                )
            # use_ue8m0 and the static-subbatch rejection are validated once at
            # config time in MoELayer.__init__; the node only re-checks the
            # device and the structural requirements its GEMM relies on.
            assert moe_expert_fusion and moe_deep_gemm and use_fp8_mlp, (
                "use_w4a8 requires moe_expert_fusion + moe_deep_gemm + use_fp8_mlp"
            )
            assert hasattr(
                paddlefleet_deep_gemm, "m_grouped_fp8_fp4_gemm_nt_contiguous"
            ), (
                "use_w4a8 requires paddlefleet_ops.deep_gemm to provide the "
                "fp8*fp4 grouped GEMM"
            )
            # Under w4a8, all dW goes through the bf16 grouped gemm
            self.use_bf16_gemm_weight_grad = True
        _check_expert_dw_deferral_supported(
            defer_expert_up_gate_dw,
            defer_expert_down_dw,
            moe_deep_gemm,
            self.moe_subbatch_token_num_after_dispatch,
            use_fp8_mlp,
            self.use_bf16_gemm_weight_grad,
        )

    def cached_tensors(self):
        """
        cached_tensors
        """
        return [
            self.tokens_per_expert,
            self.m_indices,
            self.input,
            self.input_fp8,
            self.input_scale,
            self.o1,
        ]

    def set_cached_tensors(self, tensors):
        """
        set_cached_tensors
        """
        (
            self.tokens_per_expert,
            self.m_indices,
            self.input,
            self.input_fp8,
            self.input_scale,
            self.o1,
        ) = tensors

    def clear_cached_tensors(self):
        """
        clear_cached_tensors
        """
        self.set_cached_tensors([None] * len(self.cached_tensors()))

    def reset_state(self):
        """
        reset_state
        """
        self.tokens_per_expert = None
        self.m_indices = None
        self.clear_activation_tensors()

    def clear_activation_tensors(self):
        """
        clear_activation_tensors
        """
        self.input = None
        self.input_fp8 = None
        self.input_scale = None
        self.o1 = None

    def gen_m_indices(self, tokens_per_expert):
        """
        generate m indices
        """
        if isinstance(tokens_per_expert, paddle.Tensor):
            counts = tokens_per_expert.cast("int32")
        else:
            counts = paddle.to_tensor(tokens_per_expert, dtype="int32")
        if counts.shape[0] == 0:
            return paddle.empty([0], dtype="int32")
        return paddle.repeat_interleave(
            paddle.arange(counts.shape[0], dtype="int32"),
            counts,
        )

    def fwd_gate_up_bf16(self, x, expert_w1):
        """
        fwd_gate_up bf16
        """

        if x is None:
            assert self.input is not None
            x = self.input
        if numpy.prod(x.shape) != 0:
            if self.moe_expert_fusion:
                if self.moe_deep_gemm:
                    o1 = paddle.zeros(
                        [x.shape[0], expert_w1.shape[2]], dtype="bfloat16"
                    )
                    paddlefleet_deep_gemm.m_grouped_bf16_gemm_nn_contiguous(
                        x,
                        expert_w1,
                        o1,
                        self.m_indices,
                    )
                else:
                    o1 = paddle.incubate.nn.functional.batched_gemm(
                        x,
                        expert_w1,
                        self.tokens_per_expert,
                    )
            else:
                expert_output_list = []
                start_idx = 0
                for i, token_num in enumerate(self.tokens_per_expert):
                    if token_num == 0:
                        continue
                    end_idx = start_idx + token_num
                    x_i = x[start_idx:end_idx].contiguous()
                    expert_w1_i = expert_w1[i]
                    expert_output_list.append(
                        F.linear(x=x_i, weight=expert_w1_i)
                    )
                    start_idx = end_idx
                o1 = paddle.concat(expert_output_list, axis=0)
        else:
            if self.moe_expert_fusion:
                o1 = paddle.empty(
                    [x.shape[0], expert_w1.shape[2]], dtype=expert_w1[0].dtype
                )
            else:
                o1 = paddle.empty(
                    [x.shape[0], expert_w1[0].shape[1]],
                    dtype=expert_w1[0].dtype,
                )
        self.input = x
        return o1

    def fwd_gate_up(
        self, x, expert_w1, num_expert, tokens_per_expert, scale=None
    ):
        self.tokens_per_expert = tokens_per_expert
        if self.moe_deep_gemm or self.moe_expert_fusion:
            self.m_indices = self.gen_m_indices(self.tokens_per_expert)
        else:
            self.m_indices = None
        if not self.use_fp8_mlp:
            return self.fwd_gate_up_bf16(x, expert_w1)
        else:
            return self.fwd_gate_up_fp8(
                x, expert_w1, num_expert, tokens_per_expert, scale
            )

    def fwd_gate_up_fp8(
        self, x, expert_w1, num_expert, tokens_per_expert, scale=None
    ):
        """
        o1 = x * w1
        [m_sum, n] = [m_sum, k] * [num_groups, k, n] (m_sum = sum(tokens_per_expert))
        """
        if self.use_w4a8:
            # Backward storage (dequant_input: fp8 1x32 / else bf16) is done
            # inside _fwd_gate_up_w4a8, reusing the forward quant result directly
            return self._fwd_gate_up_w4a8(x, expert_w1, scale=scale)
        # concat w1, shape is [num_groups, n, k]

        if hasattr(self, "grouped_gemm_experts"):
            offline_quant = hasattr(
                self.grouped_gemm_experts.weight1,
                "fp8_weight_stacked_transpose",
            ) or hasattr(
                self.grouped_gemm_experts.weight1, "fp8_weight_stacked"
            )
            if not offline_quant:
                local_expert_num = expert_w1.shape[0]
                expert_w1 = [
                    expert_w1[i, :, :] for i in range(local_expert_num)
                ]
            else:
                expert_w1 = [expert_w1]

        w1_t_quant, w1_t_scale = fused_stack_quant(
            expert_w1,
            transpose=True,
            num_expert=num_expert,
            use_ue8m0=self.use_ue8m0,
        )
        w1_t_quant = w1_t_quant.reshape([num_expert, -1, w1_t_quant.shape[-1]])
        w1_t_scale = w1_t_scale.reshape([num_expert, -1, w1_t_scale.shape[-1]])

        if x is None:
            x_fp8, x_scale = self.input_fp8, self.input_scale
            assert x_fp8 is not None and x_scale is not None
            x_scale = paddle.transpose(
                paddle.transpose(x_scale, [1, 0]).contiguous(), [1, 0]
            )
        elif scale is not None:
            x_fp8, x_scale = x, scale
            assert self.dequant_input, (
                "如果传入了scale, 说明a2a使用了fp8,。必须开启dequant_input"
            )
            x_scale = paddle.transpose(
                paddle.transpose(x_scale, [1, 0]).contiguous(), [1, 0]
            )
        else:
            # quant x_bf16
            x_fp8, x_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
                x,
                output_scale_transpose=True,
                quant_method="1x128",
                input_transpose=False,
                using_ue8m0_scale=self.use_ue8m0,
            )
            x_scale = x_scale.T

        # compute gemm
        o1 = paddle.empty(
            [x_fp8.shape[0], w1_t_quant.shape[1]], dtype=expert_w1[0].dtype
        )
        if numpy.prod(x_fp8.shape) != 0:
            if not self.moe_expert_fusion:
                split_group_gemm(
                    x_fp8,
                    x_scale,
                    w1_t_quant,
                    w1_t_scale,
                    tokens_per_expert,
                    o1,
                    use_ue8m0=self.use_ue8m0,
                )
            else:
                if self.use_ue8m0:
                    w1_t_scale = (
                        w1_t_scale.transpose([0, 2, 1])
                        .contiguous()
                        .transpose([0, 2, 1])
                    )
                paddlefleet_deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                    (x_fp8, x_scale),
                    (w1_t_quant, w1_t_scale),
                    o1,
                    self.m_indices,
                )

        if self.dequant_input:
            self.input_fp8 = x_fp8
            self.input_scale = x_scale
        else:
            self.input = x
        return o1

    # ------------------------------------------------------------------
    # W4A8 path: online fp4 1x32 weight quant + fp8 1x32 activation quant +
    # DeepGEMM fp8*fp4
    # ------------------------------------------------------------------

    def _w4a8_grouped_gemm(self, x_fp8, x_sf, w_fp4, w_sf, out):
        """Call DeepGEMM's m_grouped fp8*fp4 contiguous GEMM (SM100 1D1D).

        x_fp8/x_sf: fp8 1x32 activation quant result [M, K] / [M, K/32]
        w_fp4/w_sf: fp4 1x32 weight quant result [E, N, K/2] / [E, N, K/32]
        out: [M, N] bf16; grouped_layout is self.m_indices (per-row group id).
        """
        paddlefleet_deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
            (x_fp8, x_sf),
            (w_fp4, w_sf),
            out,
            self.m_indices,
            recipe_a=(1, W4A8_QUANT_BLOCK),
            recipe_b=(1, W4A8_QUANT_BLOCK),
        )
        return out

    def _fwd_gate_up_w4a8(self, x, expert_w1, scale=None):
        """W4A8 forward up_gate: o1 = x(fp8 1x32) @ w1(fp4 1x32)^T

        [m_sum, n] = [m_sum, k] * [num_groups, k, n]

        ``scale`` is non-None only under the W4A8 mxfp8 dispatch path (DeepEP or
        HybridEP): ``x`` is then the already-quantized fp8 activation and
        ``scale`` its per-row 1x32 UE8M0 scale packed into an int32 carrier
        ``[M, K/128]`` (the DeepEP companion directly; the HybridEP fp32 slot
        viewed back to int32 by the dispatcher). We unpack it to the exact
        float32 ``[M, K/32]`` the post-dispatch ``_mxfp8_quant`` would have
        produced, so the GEMM input and the tensors stored for backward are
        byte-for-byte identical to the quantize-after-dispatch path.
        """
        x_fp8 = x_sf = None
        if scale is not None:
            assert self.dequant_input, (
                "a scale implies the all-to-all used fp8, so dequant_input "
                "must be enabled"
            )
            assert x is not None and x.dtype == paddle.float8_e4m3fn, (
                "w4a8 fp8 dispatch expects an fp8 activation input, got "
                f"{None if x is None else x.dtype}"
            )
            x_fp8 = x if x.is_contiguous() else x.contiguous()
            x_sf = unpack_ue8m0_scale(scale, x.shape[-1] // W4A8_QUANT_BLOCK)
        elif x is None:
            # backward recompute: under dequant_input reuse the stored 1x32 fp8
            # directly, saving a dequantize + requantize round-trip
            if self.dequant_input:
                assert self.input_fp8 is not None
                x_fp8, x_sf = self.input_fp8, self.input_scale
            else:
                assert self.input is not None
                x = self.input
        # [E, K, N] -> [E, N, K/2]
        w1_fp4, w1_sf = _mxfp4_stack_quant(
            expert_w1,
            transpose=True,
            use_ue8m0=self.use_ue8m0,
        )
        if x_fp8 is None:
            x_fp8, x_sf = _mxfp8_quant(x)
        # Store for backward: under dequant_input keep the fp8 1x32 quant result
        # (saves memory; the bf16 wgrad dequantizes it with _mxfp8_dequant),
        # otherwise store bf16
        if self.dequant_input:
            self.input_fp8, self.input_scale = x_fp8, x_sf
        else:
            self.input = x

        o1 = paddle.empty(
            [x_fp8.shape[0], w1_fp4.shape[1]], dtype=paddle.bfloat16
        )
        if numpy.prod(x_fp8.shape) != 0:
            self._w4a8_grouped_gemm(x_fp8, x_sf, w1_fp4, w1_sf, o1)
        return o1

    def _mxfp8_act_quant(self, o1, unzipped_probs):
        """Activation + router-scale + fp8 1x32 quant of the gate/up output for
        the w4a8 down path.

        SwiGLU uses the fused weighted-swiglu quant kernel (SiLU hardcoded).
        SiTU-GLU has no fused kernel, so it
        reuses the validated ``situ_glu_scale_forward`` (fp32 intermediates,
        identical to the non-w4a8 fp8 situ path in ``fwd_down_fp8``) and then
        applies the standard w4a8 1x32 fp8 quant. ``clamp_value`` does not apply
        to situ, matching the fp8/bf16 situ branches.
        """
        if self.activation_type == "situ":
            o2 = situ_glu_scale_forward(
                o1,
                unzipped_probs,
                self.activation_situ_beta,
                self.activation_situ_linear_beta,
                situ_glu_fusion=self.situ_glu_fusion,
            )
            o2_fp8, o2_sf = _mxfp8_quant(o2)
            del o2
            return o2_fp8, o2_sf
        return _mxfp8_weighted_swiglu_quant(
            o1,
            unzipped_probs,
            self.clamp_value,
        )

    def _fwd_down_w4a8(
        self, o1, unzipped_probs, expert_w2, o3=None, clear_o1=False
    ):
        """W4A8 forward down: o3 = swiglu(o1)*probs (fp8 1x32) @ w2(fp4 1x32)^T

        [m_sum, k] = [m_sum, n] * [num_groups, n, k]
        """
        # Online weight quant: [E, H, K] -> [E, K, H/2]
        w2_fp4, w2_sf = _mxfp4_stack_quant(
            expert_w2,
            transpose=True,
            use_ue8m0=self.use_ue8m0,
        )
        o2_fp8, o2_sf = self._mxfp8_act_quant(o1, unzipped_probs)
        if clear_o1:
            o1._clear_to_zero_allocation()

        o3_shape = [o2_fp8.shape[0], w2_fp4.shape[1]]
        if o3 is not None:
            assert o3.shape == o3_shape, f"{o3.shape} vs {o3_shape}"
        else:
            o3 = paddle.empty(o3_shape, dtype=paddle.bfloat16)
        if numpy.prod(o2_fp8.shape) != 0:
            self._w4a8_grouped_gemm(o2_fp8, o2_sf, w2_fp4, w2_sf, o3)
        if self.w4a8_route_factor_post_w2 is not None:
            # Apply the deferred post-w2 scale (see the __init__ comment on
            # self.w4a8_route_factor_post_w2). In place because o3 may be a
            # caller-provided output buffer (see fwd_down's o3 arg).
            o3.scale_(self.w4a8_route_factor_post_w2)
        return o3

    def _bwd_down_input_w4a8(
        self, expert_w2, unzipped_grad, o1, unzipped_probs
    ):
        """W4A8 backward down dgrad: do2 = do3(fp8 1x32) @ w2(fp4 1x32)

        [m_sum, n] = [m_sum, k] * [num_groups, k, n]
        w2 [E, N, K] not transposed; the contracting dim is K (do3's columns).
        """
        w2_fp4, w2_sf = _mxfp4_stack_quant(
            expert_w2,
            transpose=False,
            use_ue8m0=self.use_ue8m0,
        )
        grad_fp8, grad_sf = _mxfp8_quant(unzipped_grad)

        do2_s = paddle.empty(
            [grad_fp8.shape[0], w2_fp4.shape[1]], dtype=unzipped_grad.dtype
        )
        if numpy.prod(grad_fp8.shape) != 0:
            self._w4a8_grouped_gemm(grad_fp8, grad_sf, w2_fp4, w2_sf, do2_s)

        # SiTU-GLU: reuse the validated
        # situ_glu_scale_backward on bf16; it returns (do1, o2_s, probs_grad),
        # matching this method's return contract. do1/o2_s are freshly allocated
        # (no o1 inplace reuse); backward_impl_fp8's inplace-release guard
        # already excludes situ (used_inplace_swiglu checks activation_type), so
        # o1 is not freed early under the GPU-async read.
        if self.activation_type == "situ":
            return situ_glu_scale_backward(
                o1,
                unzipped_probs,
                do2_s,
                self.activation_situ_beta,
                self.activation_situ_linear_beta,
                situ_glu_fusion=self.situ_glu_fusion,
            )

        # swiglu backward reuses the existing logic (clamp / inplace /
        # out-of-place branches)
        with paddle.amp.auto_cast(False):
            if self.clamp_value is not None and self.clamp_value > 0:
                do1, probs_grad, o2_s = fused_swiglu_weighted_clamp_bwd(
                    o1,
                    unzipped_probs,
                    do2_s,
                    float(self.clamp_value),
                )
            elif USE_INPLACE_SWIGLU_BWD:
                do1, probs_grad, o2_s = _fused_swiglu_probs_bwd(
                    o1, do2_s, unzipped_probs, True
                )
            else:
                do1, probs_grad, o2_s = (
                    paddle.incubate.nn.functional.fused_swiglu_weighted_bwd(
                        o1, do2_s, unzipped_probs
                    )
                )

        return do1, o2_s, probs_grad

    def _bwd_gate_up_input_w4a8(self, do1, expert_w1, dx=None):
        """W4A8 backward up_gate dgrad: dx = do1(fp8 1x32) @ w1(fp4 1x32)

        [m_sum, k] = [m_sum, n] * [num_groups, n, k]
        w1 [E, K, N] not transposed; the contracting dim is N (do1's columns).
        """
        w1_fp4, w1_sf = _mxfp4_stack_quant(
            expert_w1,
            transpose=False,
            use_ue8m0=self.use_ue8m0,
        )
        do1_fp8, do1_sf = _mxfp8_quant(do1)

        dx_shape = [do1_fp8.shape[0], w1_fp4.shape[1]]
        if dx is None:
            dx = paddle.empty(shape=dx_shape, dtype=do1.dtype)
        else:
            assert dx.shape == dx_shape, f"{dx.shape} vs {dx_shape}"
        if numpy.prod(do1_fp8.shape) != 0:
            self._w4a8_grouped_gemm(do1_fp8, do1_sf, w1_fp4, w1_sf, dx)
        return dx

    def fwd_swiglu(self, o1):
        o2 = swiglu(o1)
        return o2

    def fwd_down_bf16(self, o1, unzipped_probs, expert_w2, clear_o1=False):
        """
        fwd_down_bf16
        """
        # == [MG accuracy-alignment diff · fwd SwiGLU×probs] ==
        # Original impl: `o2 = fused_swiglu_scale_forward(o1, unzipped_probs)`.
        #   The fused kernel computes SwiGLU and the probs multiply along its own
        #   precision path, which differs from MG SequentialMLP and leaves the
        #   post-swiglu-scale output misaligned in the last bits.
        # Change (only active when use_accuracy_compatible=True): compute the
        #   equivalent by hand -- promote both SwiGLU and the per-token router
        #   scale to fp32, then cast back to the original dtype once, replicating
        #   MG's "compute in fp32, round once"; otherwise keep the fused kernel.
        # NOTE: the fp32 path hard-codes the SwiGLU (silu) formula, so GeGLU
        #   experts must NOT take it -- they use a dedicated gelu branch inside
        #   the accuracy-compatible block (kept consistent with the non-
        #   accuracy-compatible GeGLU branch below).
        # NOTE: the SwiGLU fp32 path is additionally gated on is_split_group_gemm
        #   to stay paired with the backward: bwd_down_input_bf16's fp32-autograd
        #   branch also requires is_split_group_gemm, so grouped-gemm
        #   (moe_expert_fusion=True) must keep the fused forward/backward pair
        #   here -- otherwise forward would be fp32 round-once while backward
        #   stays on the fused-kernel precision path. GeGLU is unaffected: its
        #   forward is gelu on both paths and pairs with the analytic GeGLU
        #   backward, which is not gated on is_split_group_gemm.
        # ==================================================================
        if (
            self.use_accuracy_compatible
            and self.activation_type != "situ"
            and (self.activation_type == "geglu" or self.is_split_group_gemm)
        ):
            x_glu, x_linear = paddle.chunk(o1, chunks=2, axis=-1)
            probs = unzipped_probs
            if len(probs.shape) == 1:
                probs = probs.unsqueeze(-1)
            if self.activation_type == "geglu":
                # GeGLU: gelu_tanh(gate) * up, then scale by probs. Keep the
                # gelu math identical to the non-accuracy-compatible GeGLU
                # branch below (and to bwd_down_input_bf16's GeGLU recompute)
                # so forward and backward share one activation path.
                o2 = (
                    (F.gelu(x_glu, approximate=True) * x_linear) * probs
                ).cast(o1.dtype)
            else:
                # SwiGLU: promote to fp32 and round once. Apply the same
                # activation_func_clamp_value semantics as the fused kernel
                # (clamp gate to max, value to [-clamp, clamp]) so this fp32
                # path matches fused_swiglu_scale_forward when clamp is set.
                gate_f = x_glu.astype("float32")
                val_f = x_linear.astype("float32")
                if self.clamp_value is not None and self.clamp_value > 0:
                    cv = float(self.clamp_value)
                    gate_f = paddle.clip(gate_f, max=cv)
                    val_f = paddle.clip(val_f, min=-cv, max=cv)
                o2 = (F.silu(gate_f) * val_f * probs.astype("float32")).astype(
                    o1.dtype
                )
        else:
            if self.activation_type == "situ":
                o2 = situ_glu_scale_forward(
                    o1,
                    unzipped_probs,
                    self.activation_situ_beta,
                    self.activation_situ_linear_beta,
                    situ_glu_fusion=self.situ_glu_fusion,
                )
            elif self.activation_type == "geglu":
                # GeGLU: gelu_tanh(gate) * up, then scale by probs
                # F.gelu promotes bf16 to float32, cast back to bf16 for downstream ops
                gate, up = paddle.chunk(o1, 2, dim=-1)
                o2 = F.gelu(gate, approximate=True) * up
                o2 = (o2 * unzipped_probs.unsqueeze(-1)).cast(o1.dtype)
            elif self.clamp_value is not None and self.clamp_value > 0:
                o2 = fused_swiglu_scale_forward(
                    o1, unzipped_probs, self.clamp_value
                )
            else:
                o2 = fused_swiglu_scale_forward(o1, unzipped_probs)

        if clear_o1:
            o1._clear_to_zero_allocation()

        # down proj
        if numpy.prod(o2.shape) != 0:
            if self.moe_expert_fusion:
                if self.moe_deep_gemm:
                    o3 = paddle.zeros(
                        [o2.shape[0], expert_w2.shape[2]], dtype="bfloat16"
                    )
                    paddlefleet_deep_gemm.m_grouped_bf16_gemm_nn_contiguous(
                        o2,
                        expert_w2,
                        o3,
                        self.m_indices,
                    )
                else:
                    o3 = paddle.incubate.nn.functional.batched_gemm(
                        o2,
                        expert_w2,
                        self.tokens_per_expert,
                    )
            else:
                expert_output_list = []
                start_idx = 0
                for i, token_num in enumerate(self.tokens_per_expert):
                    if token_num == 0:
                        continue
                    end_idx = start_idx + token_num
                    o1_i = o2[start_idx:end_idx].contiguous()
                    expert_w2_i = expert_w2[i]
                    expert_output_list.append(
                        F.linear(x=o1_i, weight=expert_w2_i)
                    )
                    start_idx = end_idx
                o3 = paddle.concat(expert_output_list, axis=0)
        else:
            if self.moe_expert_fusion:
                o3_shape = [o2.shape[0], expert_w2.shape[2]]
            else:
                o3_shape = [o2.shape[0], expert_w2[0].shape[1]]
            o3 = paddle.empty(o3_shape, dtype=o1.dtype)
        return o3

    def fwd_down(
        self, o1, unzipped_probs, expert_w2, num_expert, o3=None, clear_o1=False
    ):
        if not self.use_fp8_mlp:
            return self.fwd_down_bf16(o1, unzipped_probs, expert_w2, clear_o1)
        else:
            if self.activation_type not in FP8_ACTIVATION_TYPES:
                # Allowlist, not a geglu denylist: fwd_down_fp8's final branch
                # is the SwiGLU quantizer, so anything unrecognised reaching it
                # would be computed as SwiGLU. __init__ already rejects unknown
                # names; this also covers nodes built via __new__.
                raise ValueError(
                    "FP8 MoE path only supports activation_type='swiglu' "
                    f"or 'situ' yet, but got {self.activation_type!r}. Please "
                    "disable fp8 or implement the corresponding FP8 "
                    "activation kernel."
                )
            return self.fwd_down_fp8(
                o1, unzipped_probs, expert_w2, num_expert, o3, clear_o1
            )

    def fwd_down_fp8(
        self, o1, unzipped_probs, expert_w2, num_expert, o3=None, clear_o1=False
    ):
        """
        o3 = o2 * w2
        [m_sum, k] = [m_sum, n] * [num_groups, n, k]
        """
        if self.use_w4a8:
            return self._fwd_down_w4a8(
                o1, unzipped_probs, expert_w2, o3=o3, clear_o1=clear_o1
            )
        # concat and transpose w2

        if (
            hasattr(self, "grouped_gemm_experts")
            and self.grouped_gemm_experts is not None
        ):
            offline_quant = hasattr(
                self.grouped_gemm_experts.weight2,
                "fp8_weight_stacked_transpose",
            ) or hasattr(
                self.grouped_gemm_experts.weight2, "fp8_weight_stacked"
            )
            if not offline_quant:
                local_expert_num = expert_w2.shape[0]
                expert_w2 = [
                    expert_w2[i, :, :] for i in range(local_expert_num)
                ]
            else:
                expert_w2 = [expert_w2]

        w2_quant, w2_scale = fused_stack_quant(
            expert_w2,
            transpose=True,
            num_expert=num_expert,
            use_ue8m0=self.use_ue8m0,
        )
        w2_quant = w2_quant.reshape([num_expert, -1, w2_quant.shape[-1]])
        w2_scale = w2_scale.reshape([num_expert, -1, w2_scale.shape[-1]])

        if self.activation_type == "situ":
            # SiTU-GLU 没有 activation+scale+quant 的融合算子，拆成两步：
            # 先在 bf16 上算 SiTU-GLU×probs，再走通用 blockwise 量化。量化参数与
            # 下面的 SwiGLU 融合算子对齐（pow2 scale + 可选 ue8m0），产出的
            # o2_fp8 / o2_scale 与融合算子同形同 dtype，后续 layout 处理可直接复用。
            # clamp_value 在这条路径上不生效，与 fwd_down_bf16 的 situ 分支一致。
            o2 = situ_glu_scale_forward(
                o1,
                unzipped_probs,
                self.activation_situ_beta,
                self.activation_situ_linear_beta,
                situ_glu_fusion=self.situ_glu_fusion,
            )
            o2_fp8, o2_scale = (
                paddle.incubate.nn.functional.fp8_quant_blockwise(
                    o2,
                    output_scale_transpose=False,
                    quant_method="1x128",
                    input_transpose=False,
                    using_pow2_scale=True,
                    using_ue8m0_scale=self.use_ue8m0,
                )
            )
            del o2
        elif self.clamp_value is not None and self.clamp_value > 0:
            o2_fp8, o2_scale = fuse_weighted_swiglu_fp8_quant_clamp(
                o1,
                unzipped_probs,
                using_pow2_scaling=True,
                use_ue8m0=self.use_ue8m0,
                clamp_value=float(self.clamp_value),
            )
        else:
            o2_fp8, o2_scale = fuse_weighted_swiglu_fp8_quant(
                o1,
                unzipped_probs,
                using_pow2_scaling=True,
                use_ue8m0=self.use_ue8m0,
            )
        o2_scale = paddle.transpose(
            paddle.transpose(o2_scale, [1, 0]).contiguous(), [1, 0]
        )

        o2_fp8, o2_scale = inspect_tensor(
            "moe_act_quant_output",
            self.layer_number,
            (o2_fp8, o2_scale),
            pre_save_func=lambda pair: canonical_rows(
                dequant_dispatched_hidden_bf16(*pair)
            ),
            post_load_func=lambda canon: requant_swiglu_output(
                o2_fp8, o2_scale, canon, self.use_ue8m0
            ),
        )

        if clear_o1:
            o1._clear_to_zero_allocation()
        # fused_weighted_swiglu_act_quant 已消费完 o1 产出 o2_fp8，此时 o1 可以安全释放。
        o3_shape = [o2_fp8.shape[0], w2_quant.shape[1]]
        if o3 is not None:
            assert o3.shape == o3_shape, f"{o3.shape} vs {o3_shape}"
            o3.zero_()
        else:
            o3 = paddle.empty(o3_shape, dtype=o1.dtype)
        if numpy.prod(o2_fp8.shape) != 0:
            if not self.moe_expert_fusion:
                split_group_gemm(
                    o2_fp8,
                    o2_scale,
                    w2_quant,
                    w2_scale,
                    self.tokens_per_expert,
                    o3,
                    use_ue8m0=self.use_ue8m0,
                )
            else:
                if self.use_ue8m0:
                    w2_scale = (
                        w2_scale.transpose([0, 2, 1])
                        .contiguous()
                        .transpose([0, 2, 1])
                    )
                paddlefleet_deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                    (o2_fp8, o2_scale),
                    (w2_quant, w2_scale),
                    o3,
                    self.m_indices,
                )
        return o3

    def bwd_down_input_bf16(self, expert_w2, unzipped_grad, o1, unzipped_probs):
        """
        bwd_down_input_bf16
        """
        if numpy.prod(unzipped_grad.shape) != 0:
            if self.moe_expert_fusion and not self.use_fp8_mlp:
                if self.moe_deep_gemm:
                    do2_s = paddle.zeros(
                        [unzipped_grad.shape[0], expert_w2.shape[1]],
                        dtype=paddle.bfloat16,
                    )
                    paddlefleet_deep_gemm.m_grouped_bf16_gemm_nt_contiguous(
                        unzipped_grad,
                        expert_w2,
                        do2_s,
                        self.m_indices,
                    )
                else:
                    do2_s = paddle.incubate.nn.functional.batched_gemm(
                        unzipped_grad,
                        expert_w2,
                        self.tokens_per_expert,
                        trans_rhs=True,
                    )
            else:
                do2_s_list = []
                start_idx = 0
                for i, token_num in enumerate(self.tokens_per_expert):
                    if token_num == 0:
                        continue
                    end_idx = start_idx + token_num
                    unzipped_grad_i = unzipped_grad[
                        start_idx:end_idx
                    ].contiguous()
                    expert_w2_i = expert_w2[i].T.contiguous()
                    if self.use_accuracy_compatible:
                        do2_s_list.append(
                            paddle.matmul(unzipped_grad_i, expert_w2_i)
                        )
                    else:
                        do2_s_list.append(
                            F.linear(x=unzipped_grad_i, weight=expert_w2_i)
                        )
                    start_idx = end_idx
                do2_s = paddle.concat(do2_s_list, axis=0)
        else:
            if self.moe_expert_fusion and not self.use_fp8_mlp:
                do2_s_shape = [unzipped_grad.shape[0], expert_w2.shape[1]]
            else:
                do2_s_shape = [unzipped_grad.shape[0], expert_w2[0].shape[1]]
            do2_s = paddle.empty(do2_s_shape, dtype=unzipped_grad.dtype)

        # == [MG accuracy-alignment diff · bwd SwiGLU×probs] ==
        # Original impl: `fused_swiglu_scale_forward` + `fused_swiglu_scale_backward`.
        #   This fused-kernel pair recomputes o2_s and derives do1 / probs_grad, but
        #   its backward precision path differs from the forward hand-written fp32
        #   path (fwd_down_bf16), leaving backward gradients misaligned with MG in
        #   the last bits.
        # Change (only when use_accuracy_compatible=True and non-grouped_gemm and
        #   non-empty grad): rebuild the compute graph with the exact same fp32
        #   expression as the forward (F.silu(gate) * val * scale) and let autograd
        #   run the backward, so forward and backward share one fp32 math path;
        #   otherwise keep the original fused kernel (preserve existing behavior).
        # NOTE: this fp32 path hard-codes the SwiGLU (silu) formula, so it must
        #   NOT be taken for GeGLU experts -- GeGLU falls through to its own
        #   branch below, which uses gelu.
        # ==================================================================
        if (
            self.use_accuracy_compatible
            and self.activation_type not in ("geglu", "situ")
            and self.is_split_group_gemm
            and numpy.prod(unzipped_grad.shape) != 0
        ):
            x_glu, x_linear = paddle.chunk(o1, chunks=2, axis=-1)
            probs_v = (
                unzipped_probs
                if unzipped_probs.ndim > 1
                else unzipped_probs.unsqueeze(-1)
            )
            with paddle.enable_grad():
                gate_g = x_glu.astype("float32").detach()
                val_g = x_linear.astype("float32").detach()
                scale_g = probs_v.astype("float32").detach()
                gate_g.stop_gradient = False
                val_g.stop_gradient = False
                scale_g.stop_gradient = False
                # Apply the same activation_func_clamp_value semantics as the
                # forward / fused kernel (clamp gate to max, value to
                # [-clamp, clamp]). autograd through paddle.clip masks the
                # gradient where saturated, matching fused_swiglu_weighted_clamp_bwd.
                if self.clamp_value is not None and self.clamp_value > 0:
                    cv = float(self.clamp_value)
                    gate_c = paddle.clip(gate_g, max=cv)
                    val_c = paddle.clip(val_g, min=-cv, max=cv)
                else:
                    gate_c = gate_g
                    val_c = val_g
                o2_f32 = F.silu(gate_c) * val_c * scale_g
                paddle.autograd.backward(
                    [o2_f32], [do2_s.astype("float32").detach()]
                )
                d_gate_f32 = gate_g.grad
                d_up_f32 = val_g.grad

                # This ffn-dimension reduction uses a different accumulation order in fp32 than MG(torch) -> dL/dprobs.
                # The last-bit split causes a 1-ULP gate wgrad difference (isolated test: fp32 vs fp64 differs by ~3.5e-8).
                # Summing only in fp64 removes the accumulation-order noise and makes both sides converge to the same value; do1 still follows the fp32 path above
                # while autograd stays unchanged, keeping expert wgrad aligned.
                probs_grad_fp64 = (
                    F.silu(gate_c.detach().astype("float64"))
                    * val_c.detach().astype("float64")
                    * do2_s.astype("float64").detach()
                ).sum(axis=-1, keepdim=True)
                d_scale_f32 = probs_grad_fp64.reshape(
                    unzipped_probs.shape
                ).astype("float32")

            do1 = paddle.concat([d_gate_f32, d_up_f32], axis=-1).astype(
                o1.dtype
            )
            o2_s = o2_f32.detach().astype(o1.dtype)
            probs_grad = d_scale_f32.astype(unzipped_probs.dtype)
            return do1, o2_s, probs_grad

        if self.activation_type == "situ":
            return situ_glu_scale_backward(
                o1,
                unzipped_probs,
                do2_s,
                self.activation_situ_beta,
                self.activation_situ_linear_beta,
                situ_glu_fusion=self.situ_glu_fusion,
            )

        if self.activation_type == "geglu":
            # GeGLU forward recompute (needed for backward weight computation)
            gate, up = paddle.chunk(o1, 2, dim=-1)
            gate_act = F.gelu(gate, approximate=True)
            o2_s_no_scale = gate_act * up
            o2_s = (o2_s_no_scale * unzipped_probs).cast(o1.dtype)

            # probs_grad: d(loss)/d(probs) = do2_s * o2_no_scale, summed over hidden dim
            probs_grad = (
                do2_s.cast(paddle.float32) * o2_s_no_scale.cast(paddle.float32)
            ).sum(-1, keepdim=True)

            # Propagate gradient through probs scaling (aligned with Megatron):
            # do2 = do2_s * probs  (chain rule through o2_s = o2_no_scale * probs)
            do2 = do2_s * unzipped_probs

            # GeGLU backward through activation
            # d_up = do2 * gelu(gate)
            d_up = do2 * gate_act.cast(do2.dtype)
            # d_gate = do2 * up * gelu'(gate)  (tanh-approximate gelu derivative)
            import math

            kAlpha = math.sqrt(2.0 / math.pi)
            inner = kAlpha * (
                gate.cast(paddle.float32)
                + 0.044715 * paddle.pow(gate.cast(paddle.float32), 3)
            )
            tanh_inner = paddle.tanh(inner)
            d_gate = (
                do2
                * up.cast(do2.dtype)
                * (
                    0.5 * (1.0 + tanh_inner)
                    + 0.5
                    * gate.cast(paddle.float32)
                    * (1.0 - tanh_inner * tanh_inner)
                    * kAlpha
                    * (
                        1.0
                        + 0.134145 * paddle.pow(gate.cast(paddle.float32), 2)
                    )
                ).cast(do2.dtype)
            )
            do1 = paddle.concat([d_gate, d_up], dim=-1).cast(o1.dtype)
        elif self.clamp_value is not None and self.clamp_value > 0:
            do1, probs_grad, o2_s = fused_swiglu_weighted_clamp_bwd(
                o1, unzipped_probs, do2_s, float(self.clamp_value)
            )
        else:
            o2_s = fused_swiglu_scale_forward(o1, unzipped_probs)
            do1, probs_grad = fused_swiglu_scale_backward(
                o1, unzipped_probs, do2_s
            )

        return do1, o2_s, probs_grad

    def bwd_down_input_fp8(
        self,
        expert_w2,
        unzipped_grad,
        o1,
        unzipped_probs,
        inplace_swiglu_prob=False,
    ):
        """
        do2 = do3 * w2_t
        [m_sum, n] = [m_sum, k] * [num_groups, k, n]
        """
        if self.use_w4a8:
            return self._bwd_down_input_w4a8(
                expert_w2, unzipped_grad, o1, unzipped_probs
            )
        # recompute concated_w2_2d

        if hasattr(self, "grouped_gemm_experts"):
            offline_quant = hasattr(
                self.grouped_gemm_experts.weight2,
                "fp8_weight_stacked_transpose",
            ) or hasattr(
                self.grouped_gemm_experts.weight2, "fp8_weight_stacked"
            )
            local_expert_num = expert_w2.shape[0]
            if not offline_quant:
                expert_w2 = [
                    expert_w2[i, :, :] for i in range(local_expert_num)
                ]
            else:
                expert_w2 = [expert_w2]
        else:
            local_expert_num = len(expert_w2)

        # fp8_gemm_nt(do3[m,k], w2[n,k]) = do3 @ w2^T = do3 @ [k,n]
        bw_w2_quant, bw_w2_scale = fused_stack_quant(
            expert_w2,
            transpose=False,
            num_expert=local_expert_num,
            use_ue8m0=self.use_ue8m0,
        )
        bw_w2_quant = bw_w2_quant.reshape(
            [local_expert_num, -1, bw_w2_quant.shape[-1]]
        )
        bw_w2_scale = bw_w2_scale.reshape(
            [local_expert_num, -1, bw_w2_scale.shape[-1]]
        )
        if hasattr(
            expert_w2[0], "fp8_weight_stacked_transpose"
        ) and not hasattr(expert_w2[0], "fp8_weight_stacked"):
            bw_w2_quant = (
                bw_w2_quant.contiguous().transpose([0, 2, 1]).contiguous()
            )
            bw_w2_scale = (
                bw_w2_scale.contiguous().transpose([0, 2, 1]).contiguous()
            )

        # Pre-allocate do2_s before fp8 quant to avoid fragmentation:
        # do2_s (long-lived, inplace becomes o2_s) should be at lower address,
        # so that the short-lived unzipped_grad_fp8 is freed at the tail.
        do2_s = paddle.empty(
            [unzipped_grad.shape[0], bw_w2_quant.shape[1]],
            dtype=unzipped_grad.dtype,
        )

        # compute gemm
        if self.use_ue8m0 and self.moe_expert_fusion:
            unzipped_grad_fp8, unzipped_grad_scale = (
                paddle.incubate.nn.functional.fp8_quant_blockwise(
                    unzipped_grad,
                    output_scale_transpose=True,
                    quant_method="1x128",
                    input_transpose=False,
                    using_ue8m0_scale=True,
                )
            )
            unzipped_grad_scale = unzipped_grad_scale.T
        else:
            unzipped_grad_fp8, unzipped_grad_scale = (
                paddle.incubate.nn.functional.fp8_quant_blockwise(
                    unzipped_grad,
                    output_scale_transpose=False,
                    quant_method="1x128",
                    input_transpose=False,
                    using_ue8m0_scale=self.use_ue8m0,
                )
            )

        if numpy.prod(unzipped_grad_fp8.shape) != 0:
            if not self.moe_expert_fusion:
                split_group_gemm(
                    unzipped_grad_fp8,
                    unzipped_grad_scale,
                    bw_w2_quant,
                    bw_w2_scale,
                    self.tokens_per_expert,
                    do2_s,
                    use_ue8m0=self.use_ue8m0,
                )
            else:
                if self.use_ue8m0:
                    bw_w2_scale = (
                        bw_w2_scale.transpose([0, 2, 1])
                        .contiguous()
                        .transpose([0, 2, 1])
                    )
                paddlefleet_deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                    (unzipped_grad_fp8, unzipped_grad_scale),
                    (bw_w2_quant, bw_w2_scale),
                    do2_s,
                    self.m_indices,
                )

        if self.activation_type == "situ":
            # SiTU-GLU 的反向在 bf16 上复用 situ_glu_scale_backward，返回
            # (do1, 重算的 o2_s, probs_grad)，与下面 SwiGLU 各分支同型。o2_s 之后由
            # bwd_down_weight 重新量化，与激活类型无关。
            # 注意：这条路径不复用 o1 的 inplace buffer，do1 / o2_s 都是新分配的，
            # 所以 backward_impl_fp8 里的 inplace 释放时序判断必须排除 situ。
            return situ_glu_scale_backward(
                o1,
                unzipped_probs,
                do2_s,
                self.activation_situ_beta,
                self.activation_situ_linear_beta,
                situ_glu_fusion=self.situ_glu_fusion,
            )

        with paddle.amp.auto_cast(False):
            if self.clamp_value is not None and self.clamp_value > 0:
                do1, probs_grad, o2_s = fused_swiglu_weighted_clamp_bwd(
                    o1,
                    unzipped_probs,
                    do2_s,
                    float(self.clamp_value),
                )
            elif USE_INPLACE_SWIGLU_BWD:
                # inplace，do1 复用 o1 的 GPU buffer（data_ptr 相同）。
                # del o1 后 do1 仍持有引用，refcount 不归零，物理页不会被 VMM 提前回收。
                # 显存峰值：o1/do1(2H) + do2_s(H) + o2_s(H) = 4H（C 点），
                #           do1(2H) + o2_s(H) + n2_s(2H) = 5H（D 点峰值）
                do1, probs_grad, o2_s = _fused_swiglu_probs_bwd(
                    o1, do2_s, unzipped_probs, True
                )
            else:
                # out-of-place，do1 是全新分配的 buffer。
                # del o1 必须推迟到 bwd_gate_up_input_fp8 的 synchronize 之后，
                # 否则 GPU 异步读 o1 时物理页已被 VMM 回收（Bug 2）。
                # 显存峰值：o1(2H) + do2_s(H) + do1(2H) + o2_s(H) = 6H（C 点），
                #           o1(2H) + do1(2H) + o2_s(H) + n2_s(2H) = 7H（D 点峰值）
                do1, probs_grad, o2_s = (
                    paddle.incubate.nn.functional.fused_swiglu_weighted_bwd(
                        o1, do2_s, unzipped_probs
                    )
                )

        return do1, o2_s, probs_grad

    def bwd_swiglu(self, o1, do2):
        do1, _ = paddle._C_ops.swiglu_grad(o1, None, do2)
        return do1

    def bwd_gate_up_input_bf16(self, do1, expert_w1):
        """
        bwd_gate_up_input_bf16
        """
        if numpy.prod(do1.shape) != 0:
            if self.moe_expert_fusion and not self.use_fp8_mlp:
                if self.moe_deep_gemm:
                    dx = paddle.zeros(
                        [do1.shape[0], expert_w1.shape[1]],
                        dtype=paddle.bfloat16,
                    )
                    paddlefleet_deep_gemm.m_grouped_bf16_gemm_nt_contiguous(
                        do1,
                        expert_w1,
                        dx,
                        self.m_indices,
                    )
                else:
                    dx = paddle.incubate.nn.functional.batched_gemm(
                        do1,
                        expert_w1,
                        self.tokens_per_expert,
                        trans_rhs=True,
                    )
            else:
                dx_list = []
                start_idx = 0
                for i, token_num in enumerate(self.tokens_per_expert):
                    if token_num == 0:
                        continue
                    end_idx = start_idx + token_num
                    do1_i = do1[start_idx:end_idx].contiguous()
                    expert_w1_i = expert_w1[i].T.contiguous()
                    if self.use_accuracy_compatible:
                        dx_list.append(paddle.matmul(do1_i, expert_w1_i))
                    else:
                        dx_list.append(F.linear(x=do1_i, weight=expert_w1_i))
                    start_idx = end_idx
                dx = paddle.concat(dx_list, axis=0)
        else:
            if self.moe_expert_fusion and not self.use_fp8_mlp:
                dx_shape = [do1.shape[0], expert_w1.shape[1]]
            else:
                dx_shape = [do1.shape[0], expert_w1[0].shape[0]]
            dx = paddle.empty(shape=dx_shape, dtype=do1.dtype)
        return dx

    def bwd_gate_up_input_fp8(self, do1, expert_w1, dx=None):
        """
        dx = do1 * w1_t
        [m_sum, k] = [m_sum, n] * [num_groups, n, k]
        """
        if self.use_w4a8:
            return self._bwd_gate_up_input_w4a8(do1, expert_w1, dx=dx)
        # recompute concated_w1_t

        if hasattr(self, "grouped_gemm_experts"):
            offline_quant = hasattr(
                self.grouped_gemm_experts.weight1,
                "fp8_weight_stacked_transpose",
            ) or hasattr(
                self.grouped_gemm_experts.weight1, "fp8_weight_stacked"
            )
            local_expert_num = expert_w1.shape[0]
            if not offline_quant:
                expert_w1 = [
                    expert_w1[i, :, :] for i in range(local_expert_num)
                ]
            else:
                expert_w1 = [expert_w1]
        else:
            local_expert_num = len(expert_w1)

        bw_w1_quant, bw_w1_scale = fused_stack_quant(
            expert_w1,
            transpose=False,
            num_expert=local_expert_num,
            use_ue8m0=self.use_ue8m0,
        )
        bw_w1_quant = bw_w1_quant.reshape(
            [local_expert_num, -1, bw_w1_quant.shape[-1]]
        )
        bw_w1_scale = bw_w1_scale.reshape(
            [local_expert_num, -1, bw_w1_scale.shape[-1]]
        )

        # quant do1
        if self.use_ue8m0 and self.moe_expert_fusion:
            do1_fp8, do1_scale = (
                paddle.incubate.nn.functional.fp8_quant_blockwise(
                    do1,
                    output_scale_transpose=True,
                    quant_method="1x128",
                    input_transpose=False,
                    using_ue8m0_scale=True,
                )
            )
            do1_scale = do1_scale.T
        else:
            do1_fp8, do1_scale = (
                paddle.incubate.nn.functional.fp8_quant_blockwise(
                    do1,
                    output_scale_transpose=False,
                    quant_method="1x128",
                    input_transpose=False,
                    using_ue8m0_scale=self.use_ue8m0,
                )
            )

        # compute gemm
        dx_shape = [do1_fp8.shape[0], bw_w1_quant.shape[1]]
        if dx is None:
            dx = paddle.empty(shape=dx_shape, dtype=do1.dtype)
        else:
            assert dx.shape == dx_shape, f"{dx.shape} vs {dx_shape}"
            dx.zero_()
        if numpy.prod(do1_fp8.shape) != 0:
            if not self.moe_expert_fusion:
                split_group_gemm(
                    do1_fp8,
                    do1_scale,
                    bw_w1_quant,
                    bw_w1_scale,
                    self.tokens_per_expert,
                    dx,
                    use_ue8m0=self.use_ue8m0,
                )
            else:
                if self.use_ue8m0:
                    bw_w1_scale = (
                        bw_w1_scale.transpose([0, 2, 1])
                        .contiguous()
                        .transpose([0, 2, 1])
                    )
                paddlefleet_deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                    (do1_fp8, do1_scale),
                    (bw_w1_quant, bw_w1_scale),
                    dx,
                    self.m_indices,
                )

        return dx

    def fused_transpose_split_quant(
        self, x, scale, tokens_per_expert, pow_2_scales
    ):
        out, scale = paddle.incubate.nn.functional.fused_transpose_split_quant(
            x, scale, tokens_per_expert, pow_2_scales
        )
        return out, scale

    def bwd_down_weight(self, do3, o2, expert_w2):
        """
        dw2 = do2_t * do3
        [n, k] = [n, m_sum] * [m_sum, k] (m_sum = sum(tokens_per_expert))
        """
        if expert_weights_all_frozen(expert_w2):
            return
        o2_t_fp8, o2_t_scale = self.fused_transpose_split_quant(
            o2, None, self.tokens_per_expert, True
        )
        do3_t_fp8, do3_t_scale = self.fused_transpose_split_quant(
            do3, None, self.tokens_per_expert, True
        )

        for i in range(len(expert_w2)):
            if hasattr(expert_w2[i], "main_grad"):
                if expert_w2[i].main_grad is None:
                    expert_w2[i].main_grad = paddle.zeros(
                        shape=expert_w2[i].shape, dtype=paddle.float32
                    )
                if self.use_ue8m0:
                    deep_gemm.fp8_gemm_nt(
                        (o2_t_fp8[i], o2_t_scale[i].T),
                        (do3_t_fp8[i], do3_t_scale[i].T),
                        expert_w2[i].main_grad,
                        expert_w2[i].main_grad,
                    )
                else:
                    kitchen_gemm(
                        o2_t_fp8[i],
                        o2_t_scale[i],
                        do3_t_fp8[i],
                        do3_t_scale[i],
                        True,
                        True,
                        expert_w2[i].main_grad,
                        paddle.float32,
                    )
            else:
                if expert_w2[i].grad is None:
                    expert_w2[i].grad = paddle.zeros(
                        shape=expert_w2[i].shape, dtype=paddle.float32
                    )
                if self.use_ue8m0:
                    deep_gemm.fp8_gemm_nt(
                        (o2_t_fp8[i], o2_t_scale[i].T),
                        (do3_t_fp8[i], do3_t_scale[i].T),
                        expert_w2[i].grad,
                        expert_w2[i].grad,
                    )
                else:
                    kitchen_gemm(
                        o2_t_fp8[i],
                        o2_t_scale[i],
                        do3_t_fp8[i],
                        do3_t_scale[i],
                        True,
                        True,
                        expert_w2[i].grad,
                        paddle.float32,
                    )
            if (
                hasattr(expert_w2[i], "_apply_backward_hook")
                and not expert_w2[i].stop_gradient
            ):
                expert_w2[i]._apply_backward_hook()

    def bwd_gate_up_weight(self, do1, input_x, expert_w1, clear_input=False):
        """
        dw1 = dx_t * do1
        [k, n] = [k, m_sum] * [m_sum, n] (m_sum = sum(tokens_per_expert))
        """
        if expert_weights_all_frozen(expert_w1):
            if clear_input:
                self.input = None
                self.input_fp8 = None
                self.input_scale = None
            return

        if input_x is None:
            if self.dequant_input:
                input_x_t_fp8, input_x_t_scale = (
                    self.fused_transpose_split_quant(
                        self.input_fp8,
                        self.input_scale,
                        self.tokens_per_expert,
                        True,
                    )
                )
            else:
                input_x_t_fp8, input_x_t_scale = (
                    self.fused_transpose_split_quant(
                        self.input, None, self.tokens_per_expert, True
                    )
                )
        else:
            input_x_t_fp8, input_x_t_scale = self.fused_transpose_split_quant(
                input_x, None, self.tokens_per_expert, True
            )

        if clear_input:
            self.input = None
            self.input_fp8 = None
            self.input_scale = None

        do1_t_fp8, do1_t_scale = self.fused_transpose_split_quant(
            do1, None, self.tokens_per_expert, True
        )

        for i in range(len(expert_w1)):
            if hasattr(expert_w1[i], "main_grad"):
                if expert_w1[i].main_grad is None:
                    expert_w1[i].main_grad = paddle.zeros(
                        shape=expert_w1[i].shape, dtype=paddle.float32
                    )
                if self.use_ue8m0:
                    deep_gemm.fp8_gemm_nt(
                        (input_x_t_fp8[i], input_x_t_scale[i].T),
                        (do1_t_fp8[i], do1_t_scale[i].T),
                        expert_w1[i].main_grad,
                        expert_w1[i].main_grad,
                    )
                else:
                    kitchen_gemm(
                        input_x_t_fp8[i],
                        input_x_t_scale[i],
                        do1_t_fp8[i],
                        do1_t_scale[i],
                        True,
                        True,
                        expert_w1[i].main_grad,
                        paddle.float32,
                    )
            else:
                if expert_w1[i].grad is None:
                    expert_w1[i].grad = paddle.zeros(
                        shape=expert_w1[i].shape, dtype=paddle.float32
                    )
                if self.use_ue8m0:
                    deep_gemm.fp8_gemm_nt(
                        (input_x_t_fp8[i], input_x_t_scale[i].T),
                        (do1_t_fp8[i], do1_t_scale[i].T),
                        expert_w1[i].grad,
                        expert_w1[i].grad,
                    )
                else:
                    kitchen_gemm(
                        input_x_t_fp8[i],
                        input_x_t_scale[i],
                        do1_t_fp8[i],
                        do1_t_scale[i],
                        True,
                        True,
                        expert_w1[i].grad,
                        paddle.float32,
                    )
            if (
                hasattr(expert_w1[i], "_apply_backward_hook")
                and not expert_w1[i].stop_gradient
            ):
                expert_w1[i]._apply_backward_hook()

    @paddle.no_grad()
    def forward(
        self,
        hs_out,
        unzipped_probs,
        tokens_per_expert,
        output=None,
        scale=None,
    ):
        """如果传入了scale, 说明在a2a之前就做了quant, 这里的hs_out就是fp8。否则, hs_out是bf16"""
        if hs_out is None:
            assert self.input_fp8 is not None
            assert self.input_scale is not None
            shape = self.input_fp8.shape
            dtype = paddle.bfloat16
        elif scale is not None:
            shape = hs_out.shape
            dtype = paddle.bfloat16
        else:
            shape = hs_out.shape
            dtype = hs_out.dtype

        if shape[0] == 0:
            o3 = paddle.zeros(shape, dtype=dtype)
            return o3
        # get w1/w2
        if self.moe_expert_fusion and (
            not self.use_fp8_mlp or self.moe_deep_gemm
        ):
            expert_w1 = self.grouped_gemm_experts.weight1
            expert_w2 = self.grouped_gemm_experts.weight2
        else:
            expert_w1 = [
                x.up_gate_proj.weight for x in self.experts if x is not None
            ]
            expert_w2 = [
                x.down_proj.weight for x in self.experts if x is not None
            ]

        num_expert = len(expert_w1)

        if self.use_w4a8 and scale is not None:
            # W4A8 mxfp8 dispatch path: (hs_out, scale) = (fp8, packed int32 1x32
            # UE8M0 scale carrier) handed over by the dispatcher. The 1x128-only
            # fused_act_dequant cannot read the 1x32 scale, and the packed carrier
            # must not be touched by any float op, so the probe views it back to
            # int32, unpacks to the exact float32 1x32 scale and uses the W4A8
            # dequant; a loaded dump comes back as bf16 + scale=None and is then
            # quantized after the fact exactly like the bf16-dispatch path.
            hs_out, scale = inspect_tensor(
                "moe_dispatched_hidden",
                self.layer_number,
                (hs_out, scale),
                pre_save_func=lambda pair: canonical_rows(
                    _mxfp8_dequant(
                        pair[0],
                        unpack_ue8m0_scale(
                            pair[1], pair[0].shape[-1] // W4A8_QUANT_BLOCK
                        ),
                    )
                ),
                post_load_func=lambda canon: (
                    scatter_canonical_rows(
                        _mxfp8_dequant(
                            hs_out,
                            unpack_ue8m0_scale(
                                scale, hs_out.shape[-1] // W4A8_QUANT_BLOCK
                            ),
                        ),
                        canon,
                    ),
                    None,
                ),
            )
        else:
            hs_out, scale = inspect_tensor(
                "moe_dispatched_hidden",
                self.layer_number,
                (hs_out, scale),
                pre_save_func=lambda pair: canonical_rows(
                    dequant_dispatched_hidden_bf16(*pair)
                ),
                post_load_func=lambda canon: (
                    scatter_dispatched_hidden_bf16(hs_out, scale, canon),
                    None,
                ),
            )

        # o1
        o1 = self.fwd_gate_up(
            hs_out, expert_w1, num_expert, tokens_per_expert, scale=scale
        )
        o1 = inspect_tensor(
            "moe_ffn1_output",
            self.layer_number,
            o1,
            pre_save_func=canonical_rows,
            post_load_func=lambda canon: scatter_canonical_rows(o1, canon),
        )
        if not self.recompute_moe_gate_up:
            self.o1 = o1
            clear_o1 = False
        else:
            clear_o1 = True

        # o3
        # 只有 output 是 bf16/float32 时才传给 fwd_down（auto_subbatch 场景）
        # FP8 的 output 是复用给 gate_up 的，不应作为 down proj 输出 buffer
        fwd_down_output = (
            output
            if output is not None
            and output.dtype in (paddle.bfloat16, paddle.float32)
            else None
        )
        o3 = self.fwd_down(
            o1,
            unzipped_probs,
            expert_w2,
            num_expert,
            o3=fwd_down_output,
            clear_o1=clear_o1,
        )
        # This side folds the routing weight into the SwiGLU kernel, so o3 is
        # only comparable with the inference side's down-GEMM output while that
        # weight is forced to 1. `inspect_tensor_force_unit_probs()` does that for
        # the `moe_act_quant_output` probe, so o3 is the comparable one exactly in
        # the runs where that tag is live -- whitelisting `moe_ffn2_output` alone
        # leaves the weights in place and this dump incomparable.
        o3 = inspect_tensor(
            "moe_ffn2_output",
            self.layer_number,
            o3,
            pre_save_func=canonical_rows,
            post_load_func=lambda canon: scatter_canonical_rows(o3, canon),
        )
        return o3

    @paddle.no_grad()
    def backward(self, out_grad, unzipped_probs, a2a_async_fn=None):
        """
        反向传播函数，用于计算输入的梯度和参数的梯度。
            该函数会根据输出梯度更新模型的参数，并返回输入的梯度和隐藏状态的梯度。

            Args:
                out_grad (Tensor, optional): 输出梯度张量，默认为None，表示没有输出梯度。
                    shape为（batch_size, ...），dtype为float32。如果不为None，则需要保证batch_size大于等于1。

            Returns:
                tuple (dx, probs_grad) (Tensor, Tensor):
                    - dx (Tensor) - 输入的梯度张量，shape为（batch_size, ...），dtype为float32。
                    - probs_grad (Tensor) - 隐藏状态的梯度张量，shape为（batch_size, hidden_size），dtype为float32。
        """
        unzipped_probs = unzipped_probs.unsqueeze(-1)
        if out_grad.shape[0] == 0:
            # for cornet case, Get 0 teken in full train step
            dx = paddle.zeros_like(out_grad)
            probs_grad = paddle.zeros_like(unzipped_probs)

            if not (
                self.moe_expert_fusion
                and (not self.use_fp8_mlp or self.moe_deep_gemm)
            ):
                for expert in self.experts:
                    if expert is None:
                        continue

                    if expert_weights_all_frozen(
                        [expert.down_proj.weight, expert.up_gate_proj.weight]
                    ):
                        continue

                    if hasattr(expert.down_proj.weight, "main_grad"):
                        if expert.down_proj.weight.main_grad is None:
                            expert.down_proj.weight.main_grad = paddle.zeros(
                                shape=expert.down_proj.weight.shape,
                                dtype=paddle.float32,
                            )
                    else:
                        if expert.down_proj.weight.grad is None:
                            expert.down_proj.weight.grad = paddle.zeros(
                                shape=expert.down_proj.weight.shape,
                                dtype=paddle.float32,
                            )

                    if hasattr(expert.up_gate_proj.weight, "main_grad"):
                        if expert.up_gate_proj.weight.main_grad is None:
                            expert.up_gate_proj.weight.main_grad = paddle.zeros(
                                shape=expert.up_gate_proj.weight.shape,
                                dtype=paddle.float32,
                            )
                    else:
                        if expert.up_gate_proj.weight.grad is None:
                            expert.up_gate_proj.weight.grad = paddle.zeros(
                                shape=expert.up_gate_proj.weight.shape,
                                dtype=paddle.float32,
                            )

                    # [FIX-719] see the comment in the fused branch below.
                    for w in (
                        expert.down_proj.weight,
                        expert.up_gate_proj.weight,
                    ):
                        if (
                            hasattr(w, "_apply_backward_hook")
                            and not w.stop_gradient
                        ):
                            w._apply_backward_hook()
            else:
                for weight in (
                    self.grouped_gemm_experts.weight1,
                    self.grouped_gemm_experts.weight2,
                ):
                    # `weight` may be a per-expert view; the predicate resolves it
                    # to the parent parameter. Skipping matters most for the
                    # offline-fp8 view: its main_grad setter allocates an fp32
                    # buffer for the whole parent, not just this expert's slice.
                    if expert_weights_all_frozen(weight):
                        continue
                    if hasattr(weight, "main_grad"):
                        if weight.main_grad is None:
                            weight.main_grad = paddle.zeros(
                                shape=weight.shape, dtype=paddle.float32
                            )
                    elif weight.grad is None:
                        weight.grad = paddle.zeros(
                            shape=weight.shape, dtype=paddle.float32
                        )

                    # [FIX-719] Check the weight in to its sharding comm buffer
                    # even though this rank got zero tokens. Allocating main_grad
                    # is not enough: under sharding stage1 comm-overlap a
                    # FusedCommBuffer only launches its collective once *every*
                    # param it owns has checked in, so a rank that skips this
                    # silently omits the collective while its peers block in
                    # NCCL forever -- and it races ahead into the next DeepEP
                    # barrier alone, which traps after ~100s and surfaces as
                    # CUDA 719 on its peers.
                    # This fires at the same graph position as the non-zero-token
                    # path below (both inside FusionMoePyLayer.backward), so the
                    # collective launch order stays identical across ranks.
                    if (
                        hasattr(weight, "_apply_backward_hook")
                        and not weight.stop_gradient
                    ):
                        weight._apply_backward_hook()

            if a2a_async_fn:
                dx, task = a2a_async_fn(dx)
                task.wait()
            return dx, probs_grad

        subbatch_rows = self.moe_subbatch_token_num_after_dispatch
        if subbatch_rows is None:
            self.tokens_per_expert_tensor = paddle.to_tensor(
                self.tokens_per_expert, dtype="int32"
            )
            return self.backward_impl(
                out_grad, unzipped_probs, a2a_async_fn=a2a_async_fn
            )

        assert a2a_async_fn is None, (
            "a2a_async_fn should be None when moe_subbatch_token_num_after_dispatch is not None"
        )
        assert self.expert_id is not None, self.expert_id

        rows, _ = out_grad.shape
        nparts = (rows + subbatch_rows - 1) // subbatch_rows
        if nparts <= 1:
            self.tokens_per_expert_tensor = paddle.to_tensor(
                self.tokens_per_expert, dtype="int32"
            )
            return self.backward_impl(
                out_grad, unzipped_probs, a2a_async_fn=a2a_async_fn
            )

        input = self.input
        input_fp8 = self.input_fp8
        input_scale = self.input_scale.contiguous()
        o1 = self.o1
        tokens_per_expert = self.tokens_per_expert

        probs_grad = []
        for i in range(nparts):
            s_idx = subbatch_rows * i
            e_idx = min(rows, subbatch_rows * (i + 1))
            if input is not None:
                self.input = input._slice(s_idx, e_idx)

            if input_fp8 is not None:
                self.input_fp8 = input_fp8._slice(s_idx, e_idx)
                self.input_scale = input_scale._slice(s_idx, e_idx)

            if o1 is not None:
                self.o1 = o1._slice(s_idx, e_idx)
            self.tokens_per_expert = [e_idx - s_idx]
            self.tokens_per_expert_tensor = paddle.to_tensor(
                self.tokens_per_expert, dtype="int32"
            )
            if self.moe_deep_gemm:
                self.m_indices = self.gen_m_indices(self.tokens_per_expert)

            tmp_out_grad = out_grad._slice(s_idx, e_idx)
            tmp_unzipped_probs = unzipped_probs._slice(s_idx, e_idx)

            tmp_dx, tmp_probs_grad = self.backward_impl(
                tmp_out_grad, tmp_unzipped_probs
            )
            assert tmp_dx is tmp_out_grad
            probs_grad.append(tmp_probs_grad)

        if self.input is not None:
            self.input = input

        if self.input_fp8 is not None:
            self.input_fp8 = input_fp8
            self.input_scale = input_scale

        if self.o1 is not None:
            self.o1 = o1

        self.tokens_per_expert = tokens_per_expert
        if self.moe_deep_gemm:
            self.m_indices = self.gen_m_indices(self.tokens_per_expert)
        probs_grad = paddle.concat(probs_grad, axis=0)
        return out_grad, probs_grad

    def _lora_weight_grad(
        self, dw, lora_A, lora_B, scaling, grad_attr="main_grad"
    ):
        """
        Given dw (gradient w.r.t. effective weight = w + lora_A @ lora_B * scaling),
        compute and accumulate gradients for lora_A and lora_B.
        dw shape: [E, in_features, out_features]
        lora_A:   [E, in_features, r]
        lora_B:   [E, r, out_features]
        d_lora_B = lora_A.transpose(1,2) @ dw * scaling  -> [E, r, out_features]
        d_lora_A = dw @ lora_B.transpose(1,2) * scaling  -> [E, in_features, r]
        """
        dw_f32 = dw.cast("float32")
        # d_lora_B: [E, r, out] = [E, r, in] @ [E, in, out]
        d_lora_B = (
            paddle.bmm(lora_A.cast("float32").transpose([0, 2, 1]), dw_f32)
            * scaling
        )
        # d_lora_A: [E, in, r] = [E, in, out] @ [E, out, r]
        d_lora_A = (
            paddle.bmm(dw_f32, lora_B.cast("float32").transpose([0, 2, 1]))
            * scaling
        )

        if not hasattr(self, "_lora_grad_log_count"):
            self._lora_grad_log_count = 0
        if self._lora_grad_log_count < 3:
            self._lora_grad_log_count += 1
            import logging as _logging

            _log = _logging.getLogger(__name__)
            _log.info(
                f"[LORA GRAD EP] step={self._lora_grad_log_count}: "
                f"dw norm={float(dw_f32.norm()):.6f} "
                f"d_lora_A norm={float(d_lora_A.norm()):.6f} amax={float(d_lora_A.abs().max()):.6f} "
                f"d_lora_B norm={float(d_lora_B.norm()):.6f} amax={float(d_lora_B.abs().max()):.6f}"
            )

        def _accumulate(param, dgrad):
            dgrad = dgrad.cast(param.dtype)
            if hasattr(param, "main_grad"):
                if param.main_grad is None:
                    param.main_grad = paddle.zeros(
                        param.shape, dtype=paddle.float32
                    )
                param.main_grad.add_(dgrad.cast(paddle.float32))
            else:
                if param.grad is None:
                    param.grad = paddle.zeros(param.shape, dtype=paddle.float32)
                param.grad.add_(dgrad.cast(paddle.float32))

        _accumulate(lora_A, d_lora_A)
        _accumulate(lora_B, d_lora_B)

    def backward_impl_bf16(self, out_grad, unzipped_probs, a2a_async_fn=None):
        """
        backward_impl_bf16
        """
        if a2a_async_fn is not None:
            raise NotImplementedError(
                "bf16 fuse node do not support a2a_async_fn currently"
            )
        # Detect LoRA on grouped_gemm_experts
        _ge = (
            getattr(self, "grouped_gemm_experts", None)
            if self.moe_expert_fusion
            else None
        )
        _has_lora = (
            _ge is not None
            and hasattr(_ge, "get_delta_weight")
            and not getattr(_ge, "disable_lora", False)
            and not getattr(_ge, "merged", False)
        )

        if self.moe_expert_fusion and not self.use_fp8_mlp:
            if _has_lora:
                expert_w1 = _ge.weight1 + _ge.get_delta_weight(
                    _ge.weight1_lora_A, _ge.weight1_lora_B
                )
                expert_w2 = _ge.weight2 + _ge.get_delta_weight(
                    _ge.weight2_lora_A, _ge.weight2_lora_B
                )
            else:
                expert_w1 = self.grouped_gemm_experts.weight1
                expert_w2 = self.grouped_gemm_experts.weight2
        else:
            expert_w2 = [
                x.down_proj.weight for x in self.experts if x is not None
            ]
            expert_w1 = [
                x.up_gate_proj.weight for x in self.experts if x is not None
            ]
        if self.recompute_moe_gate_up:
            o1 = self.fwd_gate_up(
                None, expert_w1, len(expert_w1), self.tokens_per_expert
            )
        else:
            o1 = self.o1

        do1, o2_s, probs_grad = self.bwd_down_input_bf16(
            expert_w2, out_grad, o1, unzipped_probs
        )
        del o1
        self.o1 = None

        # dw1 / lora grads for w1
        if _has_lora and self.moe_expert_fusion:
            # compute dw_eff into a temporary tensor instead of accumulating to frozen weight
            if self.input is not None:
                _input = self.input
            elif self.dequant_input and self.input_fp8 is not None:
                _input = paddle.incubate.nn.functional.fused_act_dequant(
                    self.input_fp8, self.input_scale
                )
            else:
                _input = None
            if _input is not None and _input.shape[0] > 0:
                dw1 = paddle.incubate.nn.functional.batched_gemm(
                    _input, do1, self.tokens_per_expert, trans_lhs=True
                )
                self._lora_weight_grad(
                    dw1, _ge.weight1_lora_A, _ge.weight1_lora_B, _ge.scaling
                )
            self.input = None
        else:
            self.bf16_weight_grad(do1, self.input, expert_w1)
            self.input = None

        # dw2 / lora grads for w2
        if _has_lora and self.moe_expert_fusion:
            if o2_s is not None and o2_s.shape[0] > 0:
                dw2 = paddle.incubate.nn.functional.batched_gemm(
                    o2_s, out_grad, self.tokens_per_expert, trans_lhs=True
                )
                self._lora_weight_grad(
                    dw2, _ge.weight2_lora_A, _ge.weight2_lora_B, _ge.scaling
                )
        else:
            # dx here allocates its own buffer (no dx=out_grad aliasing), so a
            # deferred dw2 can keep reading out_grad safely.
            self.bf16_weight_grad(
                out_grad, o2_s, expert_w2, self.defer_expert_down_dw
            )

        # dx
        dx = self.bwd_gate_up_input_bf16(do1, expert_w1)
        del do1
        self.reset_state()
        return dx, probs_grad

    def backward_impl(self, out_grad, unzipped_probs, a2a_async_fn=None):
        if not self.use_fp8_mlp:
            return self.backward_impl_bf16(
                out_grad, unzipped_probs, a2a_async_fn
            )
        else:
            return self.backward_impl_fp8(
                out_grad, unzipped_probs, a2a_async_fn
            )

    def backward_impl_fp8(self, out_grad, unzipped_probs, a2a_async_fn=None):
        """
        backward_impl
        """
        if hasattr(self, "grouped_gemm_experts"):
            expert_w1 = self.grouped_gemm_experts.weight1
            expert_w2 = self.grouped_gemm_experts.weight2
            num_expert = expert_w1.shape[0]
        else:
            # recompute expert_w2 and expert_w1
            expert_w2 = [
                x.down_proj.weight for x in self.experts if x is not None
            ]
            expert_w1 = [
                x.up_gate_proj.weight for x in self.experts if x is not None
            ]
            num_expert = len(expert_w1)

        if self.recompute_moe_gate_up:
            o1 = self.fwd_gate_up(
                None, expert_w1, num_expert, self.tokens_per_expert
            )
        else:
            o1 = self.o1

        if self.w4a8_route_factor_post_w2 is not None:
            # routed_scaling_factor must be applied before out_grad fans out to
            # the dgrad (bwd_down_input_fp8) and the w2 wgrad (bwd_down_weight);
            # scaling only one would leave the other missing the factor. In
            # place because out_grad is later reused as the dx output buffer.
            out_grad.scale_(self.w4a8_route_factor_post_w2)

        # do2
        do1, o2_s, probs_grad = self.bwd_down_input_fp8(
            expert_w2, out_grad, o1, unzipped_probs, inplace_swiglu_prob=True
        )
        # del o1 时机：
        #   inplace（USE_INPLACE_SWIGLU_BWD=True 且无 clamp）：do1 与 o1 共用 buffer，
        #     refcount 不归零，立即 del 安全。
        #   out-of-place（USE_INPLACE_SWIGLU_BWD=False 或 clamp_value 已设置）：
        #     do1 是独立 buffer，GPU 异步 kernel 仍在读 o1，
        #     必须等 bwd_gate_up_input_fp8 的 synchronize 后再 del。
        #   SiTU：走 situ_glu_scale_backward，do1 恒为新分配的 buffer，同 out-of-place。
        used_inplace_swiglu = (
            USE_INPLACE_SWIGLU_BWD
            and self.clamp_value is None
            and self.activation_type != "situ"
        )
        if used_inplace_swiglu:
            del o1
        self.o1 = None
        # dw2 deferral additionally needs the bf16 weight-grad GEMM; the
        # bwd_down_weight fallback has no WeightGradStore path.
        defer_dw2 = self.defer_expert_down_dw and self.use_bf16_gemm_weight_grad
        if a2a_async_fn is None:
            # dw1
            if self.use_bf16_gemm_weight_grad:
                self.bf16_weight_grad(
                    do1, None, expert_w1, self.defer_expert_up_gate_dw
                )
            else:
                self.bwd_gate_up_weight(do1, None, expert_w1, clear_input=True)
            # 不调用 _record_stream，直接 None。
            # _record_stream 会触发 VMM 积极回收物理页，在 nparts loop 中
            # slice 被释放后原始 input_fp8 的物理页可能被提前回收，
            # 导致后续 npart 访问时 CUDA_ERROR_ILLEGAL_ADDRESS。
            self.input_fp8 = None
            self.input_scale = None
            self.input = None

            # dw2 / dx
            #
            # bwd_gate_up_input_fp8(dx=out_grad) writes dx into out_grad's
            # buffer, so out_grad is gone by the time dx returns. That is fine
            # while dw2 runs inline before dx, but a deferred dw2 would read the
            # clobbered buffer. When deferring, let dx allocate its own output
            # and keep out_grad alive for the deferred GEMM. Cost: one extra
            # [m_sum, hidden_size] activation, plus o2_s [m_sum,
            # moe_intermediate_size] held until the scheduler pops the store.
            if defer_dw2:
                self.bf16_weight_grad(out_grad, o2_s, expert_w2, True)
                dx = self.bwd_gate_up_input_fp8(do1, expert_w1)
            else:
                if self.use_bf16_gemm_weight_grad:
                    self.bf16_weight_grad(out_grad, o2_s, expert_w2)
                else:
                    self.bwd_down_weight(out_grad, o2_s, expert_w2)
                dx = self.bwd_gate_up_input_fp8(do1, expert_w1, dx=out_grad)

            # out-of-place 路径下 fused_swiglu_weighted_bwd 异步读 o1，但此时
            # 中间已经执行了 dw1、dw2 等多个 GEMM kernel（同一 stream 顺序入队），
            # 到达此处时 o1 的读取早已完成，del 安全。
            if not used_inplace_swiglu:
                del o1
            del do1
        else:
            # 为了更充分地overlap, 将dx提前。不过这样可能会增加峰值显存。

            # dx
            # NOTE: 这里复用 out_grad 作为 dx 的输出 buffer，会覆盖 out_grad；
            # 下面的 dw2 仍然读 out_grad，疑似已有缺陷（见 dw2 处注释）。
            # defer_expert_down_dw 打开时必须给 dx 单独的 buffer，否则延后的 dw2
            # 读到的是被覆盖后的数据。
            if defer_dw2:
                dx = self.bwd_gate_up_input_fp8(do1, expert_w1)
            else:
                dx = self.bwd_gate_up_input_fp8(do1, expert_w1, dx=out_grad)

            dx, task = a2a_async_fn(dx)
            # dw1
            if self.use_bf16_gemm_weight_grad:
                self.bf16_weight_grad(do1, None, expert_w1)
            else:
                self.bwd_gate_up_weight(do1, None, expert_w1, clear_input=True)
            self.input_fp8 = None
            self.input_scale = None
            self.input = None
            del do1

            # dw2
            if defer_dw2:
                self.bf16_weight_grad(out_grad, o2_s, expert_w2, True)
            elif self.use_bf16_gemm_weight_grad:
                self.bf16_weight_grad(out_grad, o2_s, expert_w2)
            else:
                self.bwd_down_weight(out_grad, o2_s, expert_w2)

            task.wait()
            # task.wait() 后所有异步 kernel（含 out-of-place 路径下
            # fused_swiglu_weighted_bwd 对 o1 的读取）已完成，安全释放。
            if not used_inplace_swiglu:
                del o1

        self.reset_state()
        return dx, probs_grad

    def bf16_weight_grad(self, dy, x, weights, defer_dw=False):
        """
        BF16 GEMM for weight grad
        """
        # `weights` may be a per-expert view; the predicate resolves it to the
        # parent parameter. The GEMM below still uses the per-expert `weights`.
        if expert_weights_all_frozen(weights):
            return
        if x is None:
            if self.dequant_input:
                if self.use_w4a8:
                    # w4a8 input is fp8 1x32 quant (fused_act_dequant only
                    # supports 1x128)
                    x = _mxfp8_dequant(
                        self.input_fp8,
                        self.input_scale,
                    )
                else:
                    x = paddle.incubate.nn.functional.fused_act_dequant(
                        self.input_fp8, self.input_scale
                    )
            else:
                x = self.input

        # grouped path: weights 是 stacked tensor，用 grouped/batched gemm 计算 weight_grad
        # 条件须与 __init__ 一致，否则 split path 下 weights 是 list 会报错
        # TODO: auto_subbatch 支持 grouped 模式后应该优化判断条件
        if self.moe_expert_fusion and (
            not self.use_fp8_mlp or self.moe_deep_gemm
        ):

            def _compute_weight_grad(
                x,
                dy,
                weights,
                weight_grad,
                tokens_per_expert,
                tokens_per_expert_tensor,
                overlap_gemm,
            ):
                if overlap_gemm:
                    deep_gemm.set_num_sms(118)
                    deep_gemm.k_grouped_bf16_gemm_tn_contiguous(
                        x,
                        dy,
                        weight_grad,
                        tokens_per_expert,
                        tokens_per_expert_tensor,
                        weight_grad,
                    )
                    deep_gemm.set_num_sms(0)
                else:
                    deep_gemm.k_grouped_bf16_gemm_tn_contiguous(
                        x,
                        dy,
                        weight_grad,
                        tokens_per_expert,
                        tokens_per_expert_tensor,
                        weight_grad,
                    )

                if (
                    hasattr(weights, "_apply_backward_hook")
                    and not weights.stop_gradient
                ):
                    weights._apply_backward_hook()

            if hasattr(weights, "main_grad"):
                if weights.main_grad is None:
                    weights.main_grad = paddle.zeros(
                        weights.shape, dtype=paddle.float32
                    )
                if self.moe_deep_gemm:
                    # Use WeightGradStore for deferred execution to overlap with P2P communication
                    if defer_dw:
                        WeightGradStore.enabled = True
                        WeightGradStore.put(
                            partial(
                                _compute_weight_grad,
                                x,
                                dy,
                                weights,
                                weights.main_grad,
                                self.tokens_per_expert,
                                self.tokens_per_expert_tensor,
                                defer_dw,
                            )
                        )
                        WeightGradStore.enabled = False
                    else:
                        _compute_weight_grad(
                            x,
                            dy,
                            weights,
                            weights.main_grad,
                            self.tokens_per_expert,
                            self.tokens_per_expert_tensor,
                            defer_dw,
                        )

                else:
                    assert not self.use_fp8_mlp, (
                        "batched_gemm is not supported when use_fp8_mlp=True"
                    )
                    weights_res = paddle.incubate.nn.functional.batched_gemm(
                        x,
                        dy,
                        self.tokens_per_expert,
                        trans_lhs=True,
                    )
                    weights.main_grad.add_(
                        weights_res.cast(weights.main_grad.dtype)
                    )

                    if (
                        hasattr(weights, "_apply_backward_hook")
                        and not weights.stop_gradient
                    ):
                        weights._apply_backward_hook()

            else:
                if weights.grad is None:
                    weights.grad = paddle.zeros(
                        weights.shape, dtype=paddle.float32
                    )
                if self.moe_deep_gemm:
                    # Use WeightGradStore for deferred execution to overlap with P2P communication
                    if defer_dw:
                        WeightGradStore.enabled = True
                        WeightGradStore.put(
                            partial(
                                _compute_weight_grad,
                                x,
                                dy,
                                weights,
                                weights.grad,
                                self.tokens_per_expert,
                                self.tokens_per_expert_tensor,
                                defer_dw,
                            )
                        )
                        WeightGradStore.enabled = False
                    else:
                        _compute_weight_grad(
                            x,
                            dy,
                            weights,
                            weights.grad,
                            self.tokens_per_expert,
                            self.tokens_per_expert_tensor,
                            defer_dw,
                        )
                else:
                    assert not self.use_fp8_mlp, (
                        "batched_gemm is not supported when use_fp8_mlp=True"
                    )
                    weights_res = paddle.incubate.nn.functional.batched_gemm(
                        x,
                        dy,
                        self.tokens_per_expert,
                        trans_lhs=True,
                    )
                    weights.grad.add_(weights_res.cast(weights.grad.dtype))

                    if (
                        hasattr(weights, "_apply_backward_hook")
                        and not weights.stop_gradient
                    ):
                        weights._apply_backward_hook()
        else:
            # split path: weights 是 list，逐专家计算 weight_grad (支持 auto_subbatch fallback)
            start_idx = 0
            for i, n in enumerate(self.tokens_per_expert):
                if hasattr(weights[i], "main_grad"):
                    if weights[i].main_grad is None:
                        weights[i].main_grad = paddle.zeros(
                            weights[i].shape, dtype=paddle.float32
                        )
                    grad_attr = weights[i].main_grad
                else:
                    if weights[i].grad is None:
                        weights[i].grad = paddle.zeros(
                            weights[i].shape, dtype=paddle.float32
                        )
                    grad_attr = weights[i].grad

                if n > 0:
                    end_idx = start_idx + n
                    # == [MG accuracy-alignment diff · expert wgrad] ==
                    # Only active when use_accuracy_compatible=True: on the non-fp8
                    #   path, cast the x/dy slices to fp32 before accumulating via
                    #   fused_linear_param_grad_add, matching MG SequentialMLP's fp32
                    #   expert wgrad. Otherwise keep the original direct accumulation.
                    # Note: this local tree already pre-pads tokens_per_expert
                    #   upstream (padding_token_per_experts, which is alignment-aware),
                    #   so the FP8_ALIGN re-padding from the upstream PR is a no-op
                    #   here and is intentionally omitted.
                    # ============================================================
                    if self.use_accuracy_compatible and not self.use_fp8_mlp:
                        paddle._C_ops.fused_linear_param_grad_add(
                            x._slice(start_idx, end_idx).astype("float32"),
                            dy._slice(start_idx, end_idx).astype("float32"),
                            grad_attr,
                            None,
                            True,
                            False,
                        )
                    else:
                        paddle._C_ops.fused_linear_param_grad_add(
                            x._slice(start_idx, end_idx),
                            dy._slice(start_idx, end_idx),
                            grad_attr,
                            None,
                            True,
                            False,
                        )
                    start_idx = end_idx

                if (
                    hasattr(weights[i], "_apply_backward_hook")
                    and not weights[i].stop_gradient
                ):
                    weights[i]._apply_backward_hook()
