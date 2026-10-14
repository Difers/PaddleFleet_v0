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

import unittest
from contextlib import redirect_stdout
from io import StringIO
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import paddle

import paddlefleet.transformer.transformer_layer as layer_module
from paddlefleet.transformer.transformer_layer import (
    HyperConnectionTransformerLayer,
    TransformerLayer,
)

_STATE_NOT_UPDATED = object()


class _FakeSelfAttention:
    def __init__(self, state_update):
        self.state_update = state_update

    def __call__(self, hidden_states, **_kwargs):
        output = hidden_states * 1.0
        if self.state_update is _STATE_NOT_UPDATED:
            return output, None
        return output, None, self.state_update


class _FakeHyperConnection:
    def __call__(self, hidden_states):
        return hidden_states, hidden_states, hidden_states

    @staticmethod
    def fused_h_res_h_post_bda(
        *,
        h_res,
        original_residual,
        h_post,
        layer_output_with_bias,
        dropout_prob,
        training,
        fused,
    ):
        del h_res, original_residual, h_post, dropout_prob, training, fused
        return layer_output_with_bias[0]


def _bias_dropout_add(_training, _fused):
    def apply(output_with_bias, _residual, _dropout_prob):
        return output_with_bias[0]

    return apply


def _fake_fused_h_res_h_post_bda(
    _hyper_connection,
    _h_res,
    _original_residual,
    _h_post,
    layer_output_with_bias,
    _enable_recompute,
    manager=None,
):
    return layer_output_with_bias[0], None


def _fake_forward_mlp(hidden_states, input_ids=None, **_kwargs):
    del input_ids
    return hidden_states


def _make_attention_layer(layer_type, state_update):
    layer = SimpleNamespace(
        recompute_input_layernorm=False,
        recompute_mhc_forward=False,
        input_layernorm=lambda value: value,
        layer_number=1,
        self_attn=_FakeSelfAttention(state_update),
        self_attn_bda=_bias_dropout_add,
        pre_cross_attn_layernorm=lambda value: value,
        cross_attention=lambda value, **_kwargs: (value, None),
        cross_attn_bda=_bias_dropout_add,
        training=True,
        config=SimpleNamespace(bias_dropout_fusion=False),
        hidden_dropout_prob=0.0,
        _log_md5=lambda *_args, **_kwargs: None,
    )
    if layer_type is HyperConnectionTransformerLayer:
        layer.recompute_mhc_block = False
        layer.mhc_checkpoint_input_layernorm = False
        for name in ("_mhc_block_manager", "_mhc_head", "_mhc_layernorm"):
            setattr(
                layer,
                name,
                MethodType(
                    getattr(HyperConnectionTransformerLayer, name), layer
                ),
            )
        layer.self_attention_hyper_connection = _FakeHyperConnection()
        layer._fused_h_res_h_post_bda = _fake_fused_h_res_h_post_bda
        layer._cast_and_discard_fused_bda = (
            lambda output, _ori_dtype, _span: output
        )
    return layer


def _topk_state(value):
    return (
        paddle.to_tensor([value], dtype="int32"),
        paddle.to_tensor([value], dtype="int64"),
        paddle.to_tensor([0], dtype="int64"),
    )


def _make_forward_impl_layer(attention_result):
    return SimpleNamespace(
        training=True,
        layer_number=1,
        full_recompute=False,
        mlp=object(),
        config=SimpleNamespace(
            block_attention_residuals=False,
            multi_latent_attention=False,
        ),
        _log_md5=lambda *_args, **_kwargs: None,
        _forward_attention=lambda **_kwargs: attention_result,
        _forward_mlp=_fake_forward_mlp,
    )


class TestIndexCacheTransformerLayerStateTransitions(unittest.TestCase):
    def test_forward_preserves_mtp_ids_with_indexcache_update_and_clear(self):
        hidden_states = paddle.ones([1, 1, 4], dtype="float32")
        mtp_ids = paddle.to_tensor([[1, 2]], dtype="int64")
        old_state = _topk_state(1)
        for state in (_topk_state(2), None):
            with self.subTest(clear=state is None):
                forwarded = {}

                def forward_impl(**kwargs):
                    forwarded.update(kwargs)
                    return hidden_states, None, state

                layer = SimpleNamespace(
                    config=SimpleNamespace(
                        num_nextn_predict_layers=1,
                        mtp_load_weight_only=False,
                        enable_mtp_magic_send=True,
                        block_attention_residuals=False,
                        indexcache_topk_pattern="FS",
                    ),
                    layer_number=1,
                    full_recompute=False,
                    _docmask_meta_kwargs=lambda: {},
                    _dsa_topk_holder_kwargs=lambda dict_args: {},
                    _forward_impl=forward_impl,
                )
                with (
                    patch(
                        "paddlefleet.transformer.transformer_layer.has_recovered",
                        return_value=True,
                    ),
                    redirect_stdout(StringIO()),
                ):
                    result = TransformerLayer.forward(
                        layer,
                        {
                            "hidden_states": hidden_states,
                            "mtp_full_input_ids": mtp_ids,
                            "indexcache_state": old_state,
                        },
                    )
                self.assertIs(result["mtp_full_input_ids"], mtp_ids)
                self.assertNotIn("mtp_full_input_ids", forwarded)
                if state is None:
                    self.assertNotIn("indexcache_state", result)
                else:
                    self.assertIs(result["indexcache_state"], state)

    def test_forward_attention_preserves_no_update_replace_and_clear(self):
        hidden_states = paddle.ones([1, 1, 4], dtype="float32")
        old_state = _topk_state(1)
        new_state = _topk_state(2)
        cases = (
            ("no_update", _STATE_NOT_UPDATED, old_state, True, old_state),
            ("replace", new_state, old_state, True, new_state),
            ("explicit_clear", None, old_state, True, None),
            ("empty_no_update", _STATE_NOT_UPDATED, None, False, None),
        )

        for layer_type in (
            TransformerLayer,
            HyperConnectionTransformerLayer,
        ):
            for name, update, incoming, has_state_output, expected in cases:
                with self.subTest(layer=layer_type.__name__, case=name):
                    layer = _make_attention_layer(layer_type, update)
                    result = layer_type._forward_attention(
                        layer,
                        hidden_states,
                        indexcache_state=incoming,
                    )
                    if has_state_output:
                        self.assertEqual(len(result), 3)
                        self.assertIs(result[2], expected)
                    else:
                        self.assertEqual(len(result), 2)

    def test_forward_impl_distinguishes_explicit_clear_from_no_update(self):
        hidden_states = paddle.ones([1, 1, 4], dtype="float32")
        old_state = _topk_state(1)

        clear_layer = _make_forward_impl_layer((hidden_states, None, None))
        cleared = TransformerLayer._forward_impl(
            clear_layer,
            hidden_states=hidden_states,
            indexcache_state=old_state,
        )
        self.assertIsInstance(cleared, paddle.Tensor)

        no_update_layer = _make_forward_impl_layer((hidden_states, None))
        retained = TransformerLayer._forward_impl(
            no_update_layer,
            hidden_states=hidden_states,
            indexcache_state=old_state,
        )
        self.assertEqual(len(retained), 3)
        self.assertIs(retained[2], old_state)


class TestIndexCacheRecomputeTransport(unittest.TestCase):
    def test_state_roundtrip_preserves_context_values_and_probability_gradient(
        self,
    ):
        for size in (3, 8):
            for with_context in (False, True):
                with self.subTest(size=size, context=with_context):
                    state = tuple(
                        paddle.full([1], i + 1.0) for i in range(size)
                    )
                    output = paddle.ones([2, 4])
                    context = paddle.ones([2, 1]) if with_context else None
                    flat = layer_module._flatten_indexcache_recompute_outputs(
                        (output, context, state)
                    )
                    restored = layer_module._unpack_flattened_indexcache_recompute_outputs(
                        flat
                    )
                    self.assertIs(restored[0], output)
                    self.assertIs(restored[1], context)
                    self.assertEqual(len(restored[2]), size)
                    for i, value in enumerate(restored[2]):
                        self.assertTrue(
                            paddle.equal_all(value, state[i]).item()
                        )
                        self.assertEqual(
                            value.stop_gradient, not (size == 8 and i == 5)
                        )
                    if size == 8:
                        restored[2][5].sum().backward()
                        self.assertTrue(paddle.all(state[5].grad == 1).item())
                    restored[2][0].set_value(paddle.zeros([1]))
                    self.assertEqual(float(state[0].item()), 1.0)

    def test_non_state_outputs_are_not_reinterpreted(self):
        output = paddle.ones([1])
        for value in (output, (output, None), (output, None, "metadata")):
            self.assertIs(
                layer_module._flatten_indexcache_recompute_outputs(value), value
            )
        for value in (output, (output,), (output, "a", "b", "c")):
            self.assertIsNone(
                layer_module._unpack_flattened_indexcache_recompute_outputs(
                    value
                )
            )
        self.assertIsNone(
            layer_module._clone_indexcache_recompute_state_outputs(None)
        )
        self.assertEqual(
            layer_module._mark_indexcache_recompute_state_stop_gradient(
                "metadata"
            ),
            "metadata",
        )

    def test_leaf_conversion_preserves_values_and_gradients(self):
        leaf = paddle.to_tensor([2.0, -3.0], stop_gradient=False)
        converted = layer_module._ensure_recompute_non_leaf_tensor(leaf)
        self.assertFalse(converted.is_leaf)
        self.assertTrue(paddle.equal_all(converted, leaf).item())
        self.assertIs(
            layer_module._ensure_recompute_non_leaf_tensor(converted), converted
        )
        converted.square().sum().backward()
        self.assertTrue(
            paddle.allclose(leaf.grad, paddle.to_tensor([4.0, -6.0])).item()
        )
        detached = leaf.detach()
        self.assertIs(
            layer_module._ensure_recompute_non_leaf_tensor(detached), detached
        )

    def test_diagnostics_describe_optional_state_without_changing_it(self):
        tensor = paddle.ones([2, 3])
        value = (tensor, None, 7)
        description = layer_module._describe_indexcache_recompute_input(value)
        self.assertEqual(
            description,
            [
                {
                    "type": "Tensor",
                    "shape": [2, 3],
                    "dtype": str(tensor.dtype),
                    "stop_gradient": True,
                    "is_leaf": True,
                },
                {"type": "int"},
            ],
        )
        self.assertIsNone(
            layer_module._describe_indexcache_recompute_input(None)
        )
        self.assertTrue(tensor.stop_gradient)


if __name__ == "__main__":
    unittest.main()
