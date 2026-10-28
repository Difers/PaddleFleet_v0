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

"""CPU tests of real fallback selection and cached-weight slicing methods.

Cache tensors contain distinguishable bytes for every expert and orientation;
these tests cover cache plumbing, not FP4 quantization or GPU GEMM accuracy.
Only the W4A8 hardware capability check is mocked. No distributed launch or
parallel-state initialization is needed.
"""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import paddle
import pytest

from paddlefleet.transformer.moe import fp8_utils
from paddlefleet.transformer.moe.fusion_layer_utils import MlpNode

FP4_ATTRS = (
    "fp4_weight_stacked",
    "fp4_scale_stacked",
    "fp4_weight_stacked_transpose",
    "fp4_scale_stacked_transpose",
)
FP8_ATTRS = tuple(name.replace("fp4", "fp8") for name in FP4_ATTRS)


@pytest.fixture(autouse=True)
def cpu_device():
    previous = paddle.get_device()
    paddle.set_device("cpu")
    yield
    paddle.set_device(previous)


def tensor(shape, offset=0, dtype="float32"):
    values = np.arange(np.prod(shape)).reshape(shape) + offset
    return paddle.to_tensor(values, place=paddle.CPUPlace()).cast(dtype)


def assert_tensor(actual, expected):
    assert actual.dtype == expected.dtype
    assert list(actual.shape) == list(expected.shape)
    np.testing.assert_array_equal(actual.numpy(), expected.numpy())


def cache(weight, kind, count=3, offset=0):
    attrs = FP4_ATTRS if kind == "fp4" else FP8_ATTRS
    # The transposed payload is independently quantized, not a transpose view.
    shapes = (
        (count, 4, 3),
        (count, 4, 1),
        (count, 6, 2),
        (count, 6, 1),
    )
    for i, (name, shape) in enumerate(zip(attrs, shapes)):
        if kind == "fp8":
            shape = (shape[0] * shape[1], shape[2])
        setattr(weight, name, tensor(shape, offset + i * 29, "uint8"))
    return {name: getattr(weight, name) for name in attrs}


def grouped_weights(kind):
    parent = SimpleNamespace(
        weight1=tensor((3, 4, 6), dtype="bfloat16"),
        weight2=tensor((3, 6, 4), 20, "bfloat16"),
    )
    for i, weight in enumerate((parent.weight1, parent.weight2)):
        weight.stop_gradient = False
        if kind in ("fp4", "fp8"):
            cache(weight, kind, offset=i * 37)
            weight._clear_data()
            assert not weight._is_initialized()
        elif kind == "none_attr":
            weight.fp4_weight_stacked = None
    return parent


def gemm_node(parent, kind, expert_id=None):
    custom_map = SimpleNamespace(
        grouped_gemm_experts=parent, moe_rank=1, num_experts_per_device=3
    )
    with patch.object(
        paddle.device.cuda, "get_device_capability", return_value=(10, 0)
    ):
        return fp8_utils.ExpertsGroupGemmContiguousNode(
            custom_map,
            expert_id=expert_id,
            moe_deep_gemm=True,
            moe_expert_fusion=True,
            use_fp8_mlp=True,
            use_ue8m0=True,
            use_w4a8=kind == "fp4",
        )


def assert_selected_weights(selected, parent, kind, local_id):
    for name in ("weight1", "weight2"):
        original, actual = getattr(parent, name), getattr(selected, name)
        assert list(actual.shape) == [1, *original.shape[1:]]
        assert actual._parent is original
        if kind in ("fp4", "fp8"):
            assert isinstance(actual, fp8_utils._PerExpertWeightView)
            attrs = FP4_ATTRS if kind == "fp4" else FP8_ATTRS
            for attr in attrs:
                full = getattr(original, attr)
                rows = 1 if kind == "fp4" else full.shape[0] // 3
                assert_tensor(
                    getattr(actual, attr),
                    full[local_id * rows : (local_id + 1) * rows],
                )
            assert not original._is_initialized()
            if kind == "fp4":
                for transpose in (False, True):
                    suffix = "_transpose" if transpose else ""
                    payload, scale = fp8_utils.mxfp4_stack_quant(
                        [actual], transpose
                    )
                    for value, attr in zip(
                        (payload, scale),
                        ("fp4_weight_stacked", "fp4_scale_stacked"),
                    ):
                        assert_tensor(
                            value,
                            getattr(original, attr + suffix)[
                                local_id : local_id + 1
                            ],
                        )
        else:
            assert isinstance(actual, paddle.Tensor)
            assert_tensor(actual, original[local_id : local_id + 1])
            assert actual._parent.stop_gradient is False


@pytest.mark.parametrize("kind", ["fp4", "fp8", "missing", "none_attr"])
@pytest.mark.parametrize("local_id", [0, 2])
def test_constructor_selects_cache_proxy_or_live_weight(kind, local_id):
    parent = grouped_weights(kind)
    node = gemm_node(parent, kind, expert_id=3 + local_id)
    assert node.expert_id == 3 + local_id
    assert node.experts is None
    assert node.moe_expert_fusion
    assert_selected_weights(node.grouped_gemm_experts, parent, kind, local_id)


@pytest.mark.parametrize("kind", ["fp4", "fp8", "missing", "none_attr"])
@pytest.mark.parametrize("saved_forward", [False, True])
def test_fallback_selects_weights_and_splits_saved_activations(
    kind, saved_forward
):
    parent = grouped_weights(kind)
    fused = gemm_node(parent, kind)
    if saved_forward:
        fused.input_fp8 = tensor((6, 4), dtype="uint8")
        fused.input_scale = tensor((6, 1), 10)
        fused.o1 = tensor((6, 6), 20)
    mlp = MlpNode.__new__(MlpNode)
    mlp.experts = None
    mlp.moe_rank = 1
    mlp.num_experts_per_device = 3
    mlp.moe_expert_fusion = True
    mlp.experts_group_gemm_node = fused
    mlp.padding_token_per_experts = [2, 1, 3]
    mlp.token_offsets = [0, 2, 3, 6]

    mlp.fallback_to_no_expert_fusion()

    assert not mlp.moe_expert_fusion
    assert len(mlp.experts_group_gemm_node) == 3
    assert fused.grouped_gemm_experts is parent
    for local_id, node in enumerate(mlp.experts_group_gemm_node):
        assert node is not fused
        assert node.expert_id == 3 + local_id
        assert node.is_split_group_gemm and node.moe_expert_fusion
        assert node.recompute_moe_gate_up is (not saved_forward)
        count = mlp.padding_token_per_experts[local_id]
        assert node.tokens_per_expert == [count]
        np.testing.assert_array_equal(node.m_indices.numpy(), [0] * count)
        assert_selected_weights(
            node.grouped_gemm_experts, parent, kind, local_id
        )
        for attr in ("input_fp8", "input_scale", "o1"):
            if saved_forward:
                start, end = mlp.token_offsets[local_id : local_id + 2]
                assert_tensor(
                    getattr(node, attr), getattr(fused, attr)[start:end]
                )
            else:
                assert getattr(node, attr) is None


@pytest.mark.parametrize("present_none", [False, True])
def test_fp4_view_missing_attributes_return_none(present_none):
    parent = tensor((3, 4, 6), dtype="bfloat16")
    if present_none:
        for attr in FP4_ATTRS:
            setattr(parent, attr, None)
    view = fp8_utils._PerExpertWeightView(parent, 2, 3)
    for attr in FP4_ATTRS:
        assert getattr(view, attr) is None


def list_mlp(count=3):
    mlp = MlpNode.__new__(MlpNode)
    mlp.moe_rank = 1
    mlp.num_experts_per_device = count
    # Nonlocal holes catch accidental use of a local ID as a global list index.
    mlp.experts = [None] * count
    for i in range(count):
        mlp.experts.append(
            SimpleNamespace(
                up_gate_proj=SimpleNamespace(
                    weight=tensor((4, 6), i, "bfloat16")
                ),
                down_proj=SimpleNamespace(
                    weight=tensor((6, 4), i + 10, "bfloat16")
                ),
            )
        )
    owner = mlp.experts[count]
    expected = []
    for i, proj in enumerate(("up_gate_proj", "down_proj")):
        weight = getattr(owner, proj).weight
        expected.append(cache(weight, "fp8", count, i * 37))
    for expert in mlp.experts[count:]:
        expert.up_gate_proj.weight._clear_data()
        expert.down_proj.weight._clear_data()
    return mlp, expected


@pytest.mark.parametrize("count", [1, 3])
def test_individual_fp8_caches_are_not_sliced_or_removed(count):
    mlp, _ = list_mlp(count=count)
    expected = []
    for local_id in range(count):
        expert = mlp.experts[mlp._global_expert_id(local_id)]
        for proj in ("up_gate_proj", "down_proj"):
            weight = getattr(expert, proj).weight
            expected.append((weight, cache(weight, "fp8", 1, local_id * 37)))
    for local_id in range(count):
        with mlp.slice_fp8_weight(local_id):
            for weight, attrs in expected:
                for attr, value in attrs.items():
                    assert getattr(weight, attr) is value
        for weight, attrs in expected:
            for attr, value in attrs.items():
                assert getattr(weight, attr) is value


@pytest.mark.parametrize("expert_id", [0, 2])
@pytest.mark.parametrize("fail", [False, True])
def test_list_without_cache_keeps_live_weights(expert_id, fail):
    mlp = MlpNode.__new__(MlpNode)
    mlp.moe_rank = 1
    mlp.num_experts_per_device = 3
    mlp.experts = [None] * 3 + [
        SimpleNamespace(
            up_gate_proj=SimpleNamespace(weight=tensor((4, 6), i)),
            down_proj=SimpleNamespace(weight=tensor((6, 4), i + 10)),
        )
        for i in range(3)
    ]
    weights = [
        getattr(expert, proj).weight
        for expert in mlp.experts[3:]
        for proj in ("up_gate_proj", "down_proj")
    ]
    expected = [weight.numpy().copy() for weight in weights]
    entered = False
    try:
        with mlp.slice_fp8_weight(expert_id):
            entered = True
            for weight, value in zip(weights, expected):
                np.testing.assert_array_equal(weight.numpy(), value)
            if fail:
                raise RuntimeError("fallback body failed")
    except RuntimeError as exc:
        assert fail and str(exc) == "fallback body failed"
    assert entered
    for weight, value in zip(weights, expected):
        np.testing.assert_array_equal(weight.numpy(), value)
        for attr in FP4_ATTRS + FP8_ATTRS:
            assert not hasattr(weight, attr)


@pytest.mark.parametrize("expert_id", [0, 2])
@pytest.mark.parametrize("fail", [False, True])
def test_list_fp8_cache_slicing_restores_owner_and_removes_temporary_attrs(
    expert_id, fail
):
    mlp, expected = list_mlp(count=3)
    # FP8's existing contract retains only forward caches on the owner.
    for proj in ("up_gate_proj", "down_proj"):
        owner = getattr(mlp.experts[3], proj).weight
        for attr in FP8_ATTRS[2:]:
            delattr(owner, attr)
    selected = mlp.experts[mlp._global_expert_id(expert_id)]

    class BodyFailure(Exception):
        pass

    reached = False
    try:
        with mlp.slice_fp8_weight(expert_id):
            reached = True
            for proj, original in zip(("up_gate_proj", "down_proj"), expected):
                current = getattr(selected, proj).weight
                assert not current._is_initialized()
                for attr in FP8_ATTRS:
                    if attr.endswith("_transpose"):
                        assert getattr(current, attr) is None
                        continue
                    rows = original[attr].shape[0] // 3
                    assert_tensor(
                        getattr(current, attr),
                        original[attr][
                            expert_id * rows : (expert_id + 1) * rows
                        ],
                    )
            if fail:
                raise BodyFailure
    except BodyFailure:
        assert fail
    assert reached
    for local_id, expert in enumerate(mlp.experts[3:]):
        for proj, original in zip(("up_gate_proj", "down_proj"), expected):
            current = getattr(expert, proj).weight
            for attr in FP8_ATTRS:
                if local_id == 0 and not attr.endswith("_transpose"):
                    assert getattr(current, attr) is original[attr]
                else:
                    assert not hasattr(current, attr)


def test_grouped_slice_context_does_not_touch_already_sliced_proxy():
    mlp = MlpNode.__new__(MlpNode)
    mlp.experts = None
    # Missing rank/list fields deliberately fail if the early return regresses.
    with mlp.slice_fp8_weight(2):
        assert mlp.experts is None
