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

"""``DotProductAttention`` inside a CP-replicated region (``cp_replicated_sequence``).

The HyperBody encoder holds the whole sequence on every CP rank. Its attention
must therefore behave as if CP were off: no KV gather over the CP group (which
would duplicate the sequence cp times) and no CP-only preconditions. The CP
world size is faked to 4 so the test runs on one card.
"""

import unittest
from unittest import mock

import paddle

from paddlefleet.transformer import dot_product_attention as dpa_mod
from paddlefleet.transformer.dot_product_attention import DotProductAttention
from paddlefleet.transformer.enums import AttnMaskType
from paddlefleet.transformer.prefix_lm_mask import build_dense_mask
from paddlefleet.transformer.transformer_config import TransformerConfig
from paddlefleet.utils import init_method_normal, scaled_init_method_normal

_H, _D, _T = 2, 8, 6


def _config(replicated):
    cfg = TransformerConfig(
        num_hidden_layers=1,
        hidden_size=_H * _D,
        num_attention_heads=_H,
        num_key_value_heads=_H,
        head_dim=_D,
        use_bias=False,
        init_method=init_method_normal(0.02),
        output_layer_init_method=scaled_init_method_normal(0.02, 1, 2.0),
        context_parallel_size=1,
        sequence_parallel=False,
        attention_dropout=0.0,
        masked_softmax_fusion=False,
        attention_softmax_in_fp32=True,
    )
    cfg._attn_implementation = "eager"
    if replicated is not None:
        cfg.cp_replicated_sequence = replicated
    return cfg


def _build(replicated, cp_world):
    with mock.patch.object(
        dpa_mod, "get_context_parallel_world_size", return_value=cp_world
    ):
        return DotProductAttention(
            config=_config(replicated),
            layer_number=1,
            attn_mask_type=AttnMaskType.no_mask,
            attention_type="self",
        )


class TestCpReplicatedSequence(unittest.TestCase):
    def test_replicated_region_runs_as_cp1(self):
        self.assertEqual(_build(True, 4).context_parallel_size, 1)

    def test_other_regions_keep_the_real_cp_size(self):
        # Absent or False flag: unchanged behavior for every other model.
        for flag in (None, False):
            with self.subTest(flag=flag):
                self.assertEqual(_build(flag, 4).context_parallel_size, 4)

    def test_forward_under_cp4_equals_cp1(self):
        # Under a real CP group of 4 the replicated encoder must produce exactly
        # what a CP=1 run produces for the same full sequence.
        paddle.seed(7)
        q, k, v = (paddle.randn([1, _T, _H, _D]) for _ in range(3))
        mask = build_dense_mask([2], [_T - 2])
        replicated = _build(True, 4)
        plain = _build(None, 1)
        with mock.patch.object(
            dpa_mod, "get_context_parallel_world_size", return_value=4
        ):
            got = replicated(q, k, v, mask)
        want = plain(q, k, v, mask)
        self.assertEqual(got.shape, [1, _T, _H * _D])
        self.assertTrue(bool(paddle.equal_all(got, want)))


class TestStandaloneProviderIsCpReplicated(unittest.TestCase):
    """The standalone ``HyperEncoderModelFleet`` path never shards over CP
    either: its provider must carry the same flag as the HyperBody encoder view,
    otherwise a CP>1 run would hand the full local sequence to
    ``flashmask_attention_cp`` as if it were one shard."""

    def test_flag_set_for_both_backends(self):
        from paddlefleet.transformers.hyperencoder.configuration import (
            HyperEncoderConfig,
        )
        from paddlefleet.transformers.hyperencoder.modeling_fleet import (
            HyperEncoderProvider,
        )

        for backend, packed in (("dp", False), ("flashmask", True)):
            with self.subTest(backend=backend):
                cfg = HyperEncoderConfig(
                    vocab_size=128,
                    hidden_size=64,
                    intermediate_size=128,
                    num_hidden_layers=2,
                    num_attention_heads=4,
                    num_key_value_heads=4,
                    moe_intermediate_size=32,
                    n_routed_experts=4,
                    num_experts_per_tok=2,
                    n_shared_experts=1,
                    hyperencoder_query_lengths=(16, 64),
                    hyperencoder_attn_backend=backend,
                    hyperencoder_packed_decoder=packed,
                )
                provider = HyperEncoderProvider.from_config(cfg)
                self.assertIs(provider.cp_replicated_sequence, True)


if __name__ == "__main__":
    unittest.main()
