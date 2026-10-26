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

"""Unit tests for the HyperEncoder layer-spec builders."""

import types
import unittest
from unittest import mock

import paddle  # noqa: F401

from paddlefleet.models.hyperencoder import layer_specs
from paddlefleet.models.hyperencoder.layer_specs import (
    _is_moe_layer,
    get_hyperencoder_block_spec,
    get_hyperencoder_layer_specs,
)
from paddlefleet.models.hyperencoder.norm import (
    HyperEncoderRMSNorm,
)
from paddlefleet.transformer.enums import AttnMaskType


def _cfg(**over):
    base = {
        "num_hidden_layers": 2,
        "moe_layer_freq": [0, 1],
        "n_routed_experts": 4,
        "moe_expert_fusion": True,
        "_attn_implementation": "eager",
        "hyperencoder_attn_backend": "dp",
    }
    base.update(over)
    return types.SimpleNamespace(**base)


def _make_spec():
    """Fabricate a spec object shaped like get_gpt_layer_local_spec's return."""
    sub = types.SimpleNamespace(
        input_layernorm=None, post_attention_layernorm=None
    )
    attn_sub = types.SimpleNamespace(core_attention=object())
    sub.self_attn = types.SimpleNamespace(sublayers_spec=attn_sub)
    return types.SimpleNamespace(sublayers_spec=sub)


class TestIsMoeLayer(unittest.TestCase):
    def test_list(self):
        cfg = types.SimpleNamespace(moe_layer_freq=[0, 1, 0])
        self.assertFalse(_is_moe_layer(cfg, 0))
        self.assertTrue(_is_moe_layer(cfg, 1))

    def test_tuple(self):
        cfg = types.SimpleNamespace(moe_layer_freq=(1, 0))
        self.assertTrue(_is_moe_layer(cfg, 0))

    def test_int_raises(self):
        cfg = types.SimpleNamespace(moe_layer_freq=2)
        with self.assertRaises(TypeError):
            _is_moe_layer(cfg, 0)


class TestLayerSpecsDp(unittest.TestCase):
    def test_dp_requires_eager(self):
        # Any non-eager value takes the SDPA branch under bf16 (is_causal=True).
        for impl in ("sdpa", "flashmask", None):
            with self.subTest(impl=impl):
                with self.assertRaises(ValueError) as cm:
                    get_hyperencoder_layer_specs(
                        _cfg(_attn_implementation=impl)
                    )
                self.assertIn("eager", str(cm.exception))

    def test_happy_path_replaces_norms(self):
        cfg = _cfg()
        with mock.patch.object(
            layer_specs,
            "get_gpt_layer_local_spec",
            side_effect=lambda **kw: _make_spec(),
        ) as build:
            specs = get_hyperencoder_layer_specs(cfg)
        self.assertEqual(len(specs), 2)
        for s in specs:
            self.assertIs(s.sublayers_spec.input_layernorm, HyperEncoderRMSNorm)
            self.assertIs(
                s.sublayers_spec.post_attention_layernorm,
                HyperEncoderRMSNorm,
            )
        # MoE layer index 1 should receive num_experts; index 0 should not.
        num_experts = [c.kwargs["num_experts"] for c in build.call_args_list]
        self.assertEqual(num_experts, [None, 4])


class TestLayerSpecsFlashMask(unittest.TestCase):
    def test_flashmask_rejects_eager(self):
        # The eager branch ignores row indices -> unmasked attention.
        cfg = _cfg(hyperencoder_attn_backend="flashmask")
        with self.assertRaises(ValueError) as cm:
            get_hyperencoder_layer_specs(cfg)
        self.assertIn("attn_mask_startend_row_indices", str(cm.exception))

    def test_flashmask_keeps_stock_core_attention(self):
        # FlashMask goes through the stock DotProductAttention: the spec's
        # core_attention must be left exactly as the GPT builder produced it.
        cfg = _cfg(
            hyperencoder_attn_backend="flashmask",
            _attn_implementation="flashmask",
        )
        built = []

        def _build(**kw):
            spec = _make_spec()
            # Remember the core the GPT builder produced, before any rewrite.
            built.append(
                (
                    spec,
                    spec.sublayers_spec.self_attn.sublayers_spec.core_attention,
                )
            )
            return spec

        with mock.patch.object(
            layer_specs, "get_gpt_layer_local_spec", side_effect=_build
        ) as build:
            specs = get_hyperencoder_layer_specs(cfg)
        self.assertEqual(len(specs), 2)
        for s, (orig, orig_core) in zip(specs, built):
            self.assertIs(s, orig)
            self.assertIs(
                s.sublayers_spec.self_attn.sublayers_spec.core_attention,
                orig_core,
            )
            self.assertIs(s.sublayers_spec.input_layernorm, HyperEncoderRMSNorm)
        # Same mask type as dp: semantics come only from the fed mask.
        for c in build.call_args_list:
            self.assertIs(c.kwargs["attn_mask_type"], AttnMaskType.no_mask)


class TestBlockSpec(unittest.TestCase):
    def test_block_spec_wraps_layers_and_final_norm(self):
        cfg = _cfg()
        with mock.patch.object(
            layer_specs,
            "get_gpt_layer_local_spec",
            side_effect=lambda **kw: _make_spec(),
        ):
            block = get_hyperencoder_block_spec(cfg)
        self.assertEqual(len(block.layer_specs), 2)
        self.assertIs(block.layer_norm, HyperEncoderRMSNorm)


if __name__ == "__main__":
    unittest.main()
