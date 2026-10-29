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

"""moe_ep_barrier_async: config field -> fused_a2a barrier selection."""

import unittest
from types import SimpleNamespace

import paddle.nn.functional as F

from paddlefleet.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from paddlefleet.tensor_parallel.random import model_parallel_cuda_manual_seed
from paddlefleet.transformer.mlp import MLPSublayersSpec
from paddlefleet.transformer.moe import fused_a2a
from paddlefleet.transformer.moe.moe_layer import MoELayer, MoESublayers
from paddlefleet.transformer.transformer_config import TransformerConfig
from paddlefleet.transformers.configuration_utils import PretrainedConfig


def _make_config(**kwargs):
    return TransformerConfig(
        num_hidden_layers=1,
        hidden_size=16,
        num_attention_heads=2,
        intermediate_size=32,
        n_routed_experts=2,
        n_shared_experts=0,
        num_experts_per_tok=1,
        moe_intermediate_size=16,
        moe_token_dispatcher_type="alltoall",
        moe_expert_fusion=False,
        fp8=None,
        gated_linear_unit=True,
        hidden_act=F.silu,
        tensor_model_parallel_size=1,
        **kwargs,
    )


def _make_layer(config):
    mlp_spec = MLPSublayersSpec(
        up_gate_proj=ColumnParallelLinear,
        down_proj=RowParallelLinear,
    )
    return MoELayer(
        config,
        sublayers=MoESublayers(mlp_spec=mlp_spec),
        pg_collection=SimpleNamespace(ep=None, expt_dp=None),
    )


class TestMoeEpBarrierAsync(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model_parallel_cuda_manual_seed(1234)

    def setUp(self):
        self.addCleanup(
            setattr, fused_a2a, "_EP_BARRIER_ASYNC", fused_a2a._EP_BARRIER_ASYNC
        )

    def test_defaults_enable_async_fence(self):
        self.assertTrue(_make_config().moe_ep_barrier_async)
        self.assertTrue(PretrainedConfig().moe_ep_barrier_async)

    def test_pretrained_config_override(self):
        self.assertFalse(
            PretrainedConfig(moe_ep_barrier_async=False).moe_ep_barrier_async
        )

    def test_moe_layer_applies_config_to_fused_a2a(self):
        fused_a2a.configure_ep_barrier(True)
        _make_layer(_make_config(moe_ep_barrier_async=False))
        self.assertIs(fused_a2a._EP_BARRIER_ASYNC, False)

        _make_layer(_make_config(moe_ep_barrier_async=True))
        self.assertIs(fused_a2a._EP_BARRIER_ASYNC, True)


if __name__ == "__main__":
    unittest.main()
