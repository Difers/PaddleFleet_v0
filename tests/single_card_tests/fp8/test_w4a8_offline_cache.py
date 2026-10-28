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

"""W4A8 offline-cache lifecycle and bit-exact online quantization parity.

The dispatch tests allocate no device tensors. The Blackwell tests exercise
real FP4 quantization, then clear BF16 storage just as the offline callback
can do. End-to-end GEMM/gradient parity lives in test_w4a8_group_gemm.py.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import paddle

from paddlefleet.transformer.moe import fp8_utils
from paddlefleet.transformer.moe.moe_layer import MoELayer

FP4_ATTRS = (
    "fp4_weight_stacked",
    "fp4_scale_stacked",
    "fp4_weight_stacked_transpose",
    "fp4_scale_stacked_transpose",
)
FP8_ATTRS = (
    "fp8_weight_stacked",
    "fp8_scale_stacked",
    "fp8_weight_stacked_transpose",
    "fp8_scale_stacked_transpose",
)


def _layer(**kwargs):
    fields = {
        "moe_use_fusion_node": True,
        "fp8": "e4m3",
        "use_w4a8": True,
        "use_ue8m0": True,
    }
    fields.update(kwargs)
    return SimpleNamespace(**fields)


def _expert(w1, w2):
    return SimpleNamespace(
        up_gate_proj=SimpleNamespace(weight=w1),
        down_proj=SimpleNamespace(weight=w2),
    )


class TestW4A8CacheDispatch(unittest.TestCase):
    def test_stack_single_expert_preserves_values_and_expert_axis(self):
        weight = paddle.to_tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], place=paddle.CPUPlace()
        )
        stacked = weight.unsqueeze(0)
        self.assertIs(fp8_utils._stack_expert_weights(stacked), stacked)
        with self.assertRaises(AssertionError):
            fp8_utils._stack_expert_weights(weight.reshape([-1]))

    def test_live_quant_preserves_expert_axis_and_forwards_orientation(self):
        weight = paddle.to_tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], place=paddle.CPUPlace()
        )
        stacked = weight.unsqueeze(0)
        cases = (
            (stacked, stacked),
            ([stacked], stacked),
            ((stacked,), stacked),
            ([weight], stacked),
            ((weight,), stacked),
            ([weight, weight * 2], paddle.stack([weight, weight * 2])),
            ((weight, weight * 2), paddle.stack([weight, weight * 2])),
        )
        for quantize in (
            fp8_utils.mxfp4_stack_quant_without_cache,
            fp8_utils.mxfp4_stack_quant,
        ):
            for case_id, (weights, expected) in enumerate(cases):
                for transpose in (False, True):
                    with self.subTest(
                        helper=quantize.__name__,
                        case=case_id,
                        transpose=transpose,
                    ):
                        result = (object(), object())
                        with patch.object(
                            fp8_utils,
                            "mxfp4_stack_quantize_1x32",
                            return_value=result,
                            create=True,
                        ) as kernel:
                            self.assertIs(quantize(weights, transpose), result)
                        kernel.assert_called_once()
                        actual, direction = kernel.call_args.args
                        self.assertEqual(actual.shape, expected.shape)
                        self.assertEqual(
                            actual.numpy().tolist(), expected.numpy().tolist()
                        )
                        self.assertIs(direction, transpose)
        with patch.object(
            fp8_utils, "mxfp4_stack_quantize_1x32", create=True
        ) as kernel:
            with self.assertRaises(AssertionError):
                fp8_utils.mxfp4_stack_quant_without_cache(weight, False)
            kernel.assert_not_called()

    def test_uncached_quant_rejects_non_ue8m0_before_reading_weights(self):
        with patch.object(
            fp8_utils, "mxfp4_stack_quantize_1x32", create=True
        ) as quant:
            with self.assertRaisesRegex(ValueError, "use_ue8m0=True"):
                fp8_utils.mxfp4_stack_quant_without_cache(
                    None, transpose=False, use_ue8m0=False
                )
            quant.assert_not_called()

    def test_disabled_quant_and_clear_leave_existing_caches_untouched(self):
        for fusion, fp8 in ((False, "e4m3"), (True, None)):
            with self.subTest(fusion=fusion, fp8=fp8):
                weight = SimpleNamespace()
                cached = {name: object() for name in FP4_ATTRS + FP8_ATTRS}
                for name, value in cached.items():
                    setattr(weight, name, value)
                layer = _layer(
                    moe_use_fusion_node=fusion,
                    fp8=fp8,
                    grouped_gemm_experts=SimpleNamespace(
                        weight1=weight, weight2=weight
                    ),
                )
                with patch(
                    "paddlefleet.transformer.moe.moe_layer."
                    "mxfp4_stack_quant_without_cache"
                ) as quant:
                    MoELayer.fp8_quant_weight(layer, quant_transpose=False)
                    MoELayer.clear_fp8_quant_weight(layer)
                    quant.assert_not_called()
                for name, value in cached.items():
                    self.assertIs(getattr(weight, name), value)

    def test_false_transpose_rejected_before_touching_weights(self):
        layer = _layer()
        with patch.object(
            fp8_utils, "mxfp4_stack_quantize_1x32", create=True
        ) as quant:
            with self.assertRaisesRegex(ValueError, "quant_transpose"):
                MoELayer.fp8_quant_weight(
                    layer, batch_mode=True, quant_transpose=False
                )
            quant.assert_not_called()

    def test_cache_hit_does_not_access_bf16_or_online_quant(self):
        class ClearedWeight(SimpleNamespace):
            @property
            def shape(self):
                raise AssertionError("cache hit read BF16 shape")

            @property
            def ndim(self):
                raise AssertionError("cache hit tried to stack BF16")

        weight = ClearedWeight()
        for name in FP4_ATTRS:
            setattr(weight, name, SimpleNamespace(shape=[1, 64, 64]))
        with patch.object(
            fp8_utils,
            "mxfp4_stack_quant_without_cache",
            side_effect=AssertionError("cache hit used online quant"),
        ):
            for transpose in (False, True):
                for weights in (weight, [weight], (weight,)):
                    with self.subTest(
                        transpose=transpose, container=type(weights).__name__
                    ):
                        suffix = "_transpose" if transpose else ""
                        payload, scale = fp8_utils.mxfp4_stack_quant(
                            weights, transpose
                        )
                        self.assertIs(
                            payload,
                            getattr(weight, "fp4_weight_stacked" + suffix),
                        )
                        self.assertIs(
                            scale,
                            getattr(weight, "fp4_scale_stacked" + suffix),
                        )

    def test_cache_lookup_only_reads_first_weight(self):
        class UnreadableWeight:
            def __getattribute__(self, name):
                raise AssertionError("cache lookup read another expert")

        owner = SimpleNamespace(
            fp4_weight_stacked=object(),
            fp4_scale_stacked=object(),
            fp4_weight_stacked_transpose=object(),
            fp4_scale_stacked_transpose=object(),
        )
        for transpose in (False, True):
            suffix = "_transpose" if transpose else ""
            with self.subTest(transpose=transpose):
                pair = fp8_utils.mxfp4_stack_quant(
                    [owner, UnreadableWeight()], transpose
                )
                self.assertIs(
                    pair[0], getattr(owner, "fp4_weight_stacked" + suffix)
                )
                self.assertIs(
                    pair[1], getattr(owner, "fp4_scale_stacked" + suffix)
                )

    def test_missing_cached_scale_does_not_fall_back_to_bf16(self):
        weight = SimpleNamespace(fp4_weight_stacked=object())
        with (
            patch.object(
                fp8_utils,
                "mxfp4_stack_quant_without_cache",
                side_effect=AssertionError("cache hit read BF16"),
            ),
            self.assertRaises(AttributeError),
        ):
            fp8_utils.mxfp4_stack_quant(weight, False)

    def test_cache_does_not_bypass_ue8m0_requirement(self):
        weight = SimpleNamespace(**{name: object() for name in FP4_ATTRS})
        for transpose in (False, True):
            with (
                self.subTest(transpose=transpose),
                self.assertRaisesRegex(ValueError, "use_ue8m0=True"),
            ):
                fp8_utils.mxfp4_stack_quant(weight, transpose, use_ue8m0=False)

    def test_fp8_quantization_retains_orientation_and_cache_contract(self):
        w1 = paddle.to_tensor(
            [[[1.0, 2.0]], [[3.0, 4.0]]], place=paddle.CPUPlace()
        )
        w2 = paddle.to_tensor(
            [[[5.0], [6.0]], [[7.0], [8.0]]], place=paddle.CPUPlace()
        )
        for grouped in (False, True):
            for transpose in (True, None, False):
                with self.subTest(grouped=grouped, transpose=transpose):
                    if grouped:
                        owners = (w1.clone(), w2.clone())
                        layer = _layer(
                            grouped_gemm_experts=SimpleNamespace(
                                weight1=owners[0], weight2=owners[1]
                            )
                        )
                    else:
                        groups = (
                            [w1[i] for i in range(2)],
                            [w2[i] for i in range(2)],
                        )
                        owners = (groups[0][0], groups[1][0])
                        layer = _layer(
                            experts=[None]
                            + [_expert(a, b) for a, b in zip(*groups)]
                            + [None]
                        )
                    layer.use_w4a8 = False
                    layer.use_ue8m0 = False
                    pairs = [(object(), object()) for _ in range(4)]
                    with (
                        patch(
                            "paddlefleet.transformer.moe.moe_layer."
                            "fused_stack_quant_without_cache",
                            side_effect=pairs,
                        ) as quant,
                        patch(
                            "paddlefleet.transformer.moe.moe_layer."
                            "mxfp4_stack_quant_without_cache",
                            side_effect=AssertionError(
                                "FP8 used FP4 quantizer"
                            ),
                        ),
                    ):
                        MoELayer.fp8_quant_weight(
                            layer, batch_mode=True, quant_transpose=transpose
                        )
                    directions = (
                        (False,) if transpose is False else (False, True)
                    )
                    self.assertEqual(quant.call_count, 2 * len(directions))
                    for i, (owner, expected) in enumerate(
                        zip(owners, (w1, w2))
                    ):
                        for j, direction in enumerate(directions):
                            index = i * len(directions) + j
                            call = quant.call_args_list[index]
                            self.assertEqual(
                                call.kwargs,
                                {"transpose": direction, "use_ue8m0": False},
                            )
                            self.assertEqual(
                                paddle.stack(call.args[0]).numpy().tolist(),
                                expected.numpy().tolist(),
                            )
                            suffix = "_transpose" if direction else ""
                            self.assertIs(
                                getattr(owner, "fp8_weight_stacked" + suffix),
                                pairs[index][0],
                            )
                            self.assertIs(
                                getattr(owner, "fp8_scale_stacked" + suffix),
                                pairs[index][1],
                            )
                        if transpose is False:
                            self.assertIsNone(
                                owner.fp8_weight_stacked_transpose
                            )
                            self.assertIsNone(owner.fp8_scale_stacked_transpose)
                        for name in FP4_ATTRS:
                            self.assertFalse(hasattr(owner, name))

    def test_clear_removes_fp4_and_fp8_caches_and_is_idempotent(self):
        for grouped in (False, True):
            with self.subTest(grouped=grouped):
                weights = [SimpleNamespace() for _ in range(4)]
                for weight in weights:
                    weight.unrelated = "keep"
                    for name in FP4_ATTRS + FP8_ATTRS:
                        setattr(weight, name, object())
                if grouped:
                    weights = weights[:2]
                    layer = _layer(
                        grouped_gemm_experts=SimpleNamespace(
                            weight1=weights[0], weight2=weights[1]
                        )
                    )
                else:
                    layer = _layer(
                        experts=[
                            None,
                            _expert(*weights[:2]),
                            None,
                            _expert(*weights[2:]),
                        ]
                    )
                MoELayer.clear_fp8_quant_weight(layer)
                MoELayer.clear_fp8_quant_weight(layer)
                for weight in weights:
                    self.assertEqual(weight.unrelated, "keep")
                    for name in FP4_ATTRS + FP8_ATTRS:
                        self.assertFalse(hasattr(weight, name), name)


def _is_blackwell():
    if (
        not paddle.is_compiled_with_cuda()
        or paddle.device.cuda.device_count() == 0
    ):
        return False
    try:
        return paddle.device.cuda.get_device_capability()[0] >= 10
    except (RuntimeError, ValueError):
        return False


@unittest.skipUnless(_is_blackwell(), "requires a Blackwell CUDA device")
class TestW4A8OfflineQuantParity(unittest.TestCase):
    def setUp(self):
        self.previous_device = paddle.get_device()
        paddle.set_device("gpu:0")
        self.addCleanup(paddle.set_device, self.previous_device)
        paddle.seed(2026)

    def make_layer(self, grouped=True, count=3):
        # Rectangular matrices catch orientation/shape errors hidden by K=N.
        w1 = paddle.randn([count, 64, 256], dtype="bfloat16")
        w2 = paddle.randn([count, 128, 64], dtype="bfloat16")
        if grouped:
            return _layer(
                grouped_gemm_experts=SimpleNamespace(weight1=w1, weight2=w2)
            ), (w1, w2)
        weights1 = [w1[i].clone() for i in range(count)]
        weights2 = [w2[i].clone() for i in range(count)]
        return _layer(
            experts=[None]
            + [_expert(a, b) for a, b in zip(weights1, weights2)]
            + [None]
        ), (weights1, weights2)

    def assert_pair_equal(self, actual, expected):
        for got, ref in zip(actual, expected):
            self.assertEqual(got.dtype, ref.dtype)
            self.assertEqual(list(got.shape), list(ref.shape))
            # Paddle dtype views can mishandle nonzero slice byte offsets.
            # Copy logical values first, then compare every FP4/scale bit.
            self.assertEqual(got.numpy().tobytes(), ref.numpy().tobytes())

    def online_pairs(self, weights):
        return {
            (i, transpose): fp8_utils.mxfp4_stack_quant_without_cache(
                weight, transpose
            )
            for i, weight in enumerate(weights)
            for transpose in (False, True)
        }

    def assert_cached_pairs(self, weights, expected, individually=False):
        if individually:
            # Non-batched caches are consumed one expert at a time, as in FP8.
            for expert_id in range(len(weights[0])):
                self.assert_cached_pairs(
                    tuple([group[expert_id]] for group in weights),
                    {
                        key: tuple(t[expert_id : expert_id + 1] for t in pair)
                        for key, pair in expected.items()
                    },
                )
            return
        with patch.object(
            fp8_utils,
            "mxfp4_stack_quant_without_cache",
            side_effect=AssertionError("offline path requantized BF16"),
        ):
            for (i, transpose), pair in expected.items():
                self.assert_pair_equal(
                    fp8_utils.mxfp4_stack_quant(weights[i], transpose), pair
                )

    def clear_bf16(self, weights):
        for weight in weights:
            members = weight if isinstance(weight, list) else [weight]
            for member in members:
                shape = tuple(member.shape)
                member._clear_data()
                self.assertFalse(member._is_initialized())
                self.assertEqual(tuple(member.shape), shape)

    def test_grouped_and_list_caches_match_both_online_orientations(self):
        for grouped, batch_mode, count in (
            (True, True, 3),
            (True, False, 3),
            (False, True, 3),
            (False, False, 3),
            (False, True, 1),
            (False, False, 1),
        ):
            for quant_transpose in (True, None):
                with self.subTest(
                    grouped=grouped,
                    batch_mode=batch_mode,
                    count=count,
                    quant_transpose=quant_transpose,
                ):
                    layer, weights = self.make_layer(grouped, count)
                    # Start without a cache and take the actual runtime path.
                    with patch.object(
                        fp8_utils,
                        "mxfp4_stack_quant_without_cache",
                        wraps=fp8_utils.mxfp4_stack_quant_without_cache,
                    ) as raw:
                        expected = {
                            (i, transpose): fp8_utils.mxfp4_stack_quant(
                                weight, transpose
                            )
                            for i, weight in enumerate(weights)
                            for transpose in (False, True)
                        }
                        self.assertEqual(raw.call_count, 4)
                    MoELayer.fp8_quant_weight(
                        layer,
                        batch_mode=batch_mode,
                        quant_transpose=quant_transpose,
                    )
                    self.assert_cached_pairs(
                        weights,
                        expected,
                        individually=not grouped and not batch_mode,
                    )
                    self.clear_bf16(weights)
                    self.assert_cached_pairs(
                        weights,
                        expected,
                        individually=not grouped and not batch_mode,
                    )

    def test_refresh_uses_updated_bf16_not_stale_fp4(self):
        for grouped, batch_mode, count in (
            (True, True, 3),
            (True, False, 3),
            (False, True, 3),
            (False, False, 3),
            (False, True, 1),
            (False, False, 1),
        ):
            with self.subTest(
                grouped=grouped, batch_mode=batch_mode, count=count
            ):
                layer, weights = self.make_layer(grouped, count)
                MoELayer.fp8_quant_weight(layer, batch_mode=batch_mode)
                before = self.online_pairs(weights)
                for weight in weights:
                    members = weight if isinstance(weight, list) else [weight]
                    for member in members:
                        # Change payload signs and every nonzero block scale.
                        member.set_value(member * -4.0)
                expected = self.online_pairs(weights)
                for old, new in zip(before[(0, False)], expected[(0, False)]):
                    self.assertTrue(bool((old != new).any()))
                # No clear here: step-begin must overwrite the old cache.
                MoELayer.fp8_quant_weight(
                    layer, batch_mode=batch_mode, quant_transpose=None
                )
                self.assert_cached_pairs(
                    weights,
                    expected,
                    individually=not grouped and not batch_mode,
                )
                MoELayer.clear_fp8_quant_weight(layer)
                with patch.object(
                    fp8_utils,
                    "mxfp4_stack_quant_without_cache",
                    wraps=fp8_utils.mxfp4_stack_quant_without_cache,
                ) as raw:
                    for (i, transpose), pair in expected.items():
                        self.assert_pair_equal(
                            fp8_utils.mxfp4_stack_quant(weights[i], transpose),
                            pair,
                        )
                    self.assertEqual(raw.call_count, 4)
                MoELayer.fp8_quant_weight(layer, batch_mode=batch_mode)
                self.clear_bf16(weights)
                self.assert_cached_pairs(
                    weights,
                    expected,
                    individually=not grouped and not batch_mode,
                )

    def test_grouped_proxy_keeps_both_orientations_after_bf16_clear(self):
        layer, weights = self.make_layer()
        expected = self.online_pairs(weights)
        MoELayer.fp8_quant_weight(layer, batch_mode=True)
        self.clear_bf16(weights)
        for expert_id in range(3):
            proxy = fp8_utils._PerExpertWeightProxy(
                layer.grouped_gemm_experts, expert_id
            )
            sliced = {
                key: tuple(t[expert_id : expert_id + 1] for t in pair)
                for key, pair in expected.items()
            }
            self.assertEqual(tuple(proxy.weight1.shape), (1, 64, 256))
            self.assertEqual(tuple(proxy.weight2.shape), (1, 128, 64))
            self.assert_cached_pairs((proxy.weight1, proxy.weight2), sliced)


if __name__ == "__main__":
    unittest.main()
