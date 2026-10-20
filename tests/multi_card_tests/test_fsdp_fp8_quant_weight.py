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

"""FP8 pre-quantization and expert master-weight offload under native FSDP.

Drives ``FP8QuantWeightCallback`` on a real fp8 MoE ``GPTModel`` wrapped by
``fully_shard`` (sharding_degree=2) and checks:

* the paddle-private FSDP surface the callback relies on still exists
  (``unshard_params``, ``buffer_manager.buffer_groups``,
  ``group.is_expert_param``, ``group.params_buffer.data_buffer.name``);
* ``fp8_weight_stacked`` / ``fp8_weight_fwd`` caches built through
  ``unshard_params`` are bit-identical to the non-fsdp path on the same
  weights, and the gather is fully undone afterwards;
* with ``offload_fp8_expert_master_weight=True`` the names from
  ``_fsdp_expert_master_weight_names()`` hit ``_master_weights`` after the
  first real step, and offload/reload restores place and values.

Run with:
    python -m paddle.distributed.launch --devices 0,1 \
        tests/multi_card_tests/test_fsdp_fp8_quant_weight.py
"""

import functools
import unittest
from types import SimpleNamespace

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import NoPipelineParallel
from paddle.distributed.fleet.utils import mix_precision_utils
from paddle.distributed.fsdp import fully_shard_fusion
from paddle.distributed.fsdp._fsdp_context import get_fsdp_context
from paddle.distributed.fsdp.fully_shard import fully_shard

import paddlefleet.parallel_state as ps
from paddlefleet import tensor_parallel
from paddlefleet.gpt_builders import gpt_builder
from paddlefleet.models.gpt import GPTConfig
from paddlefleet.trainer.trainer_callback import (
    FP8QuantWeightCallback,
    _fsdp_expert_master_weight_names,
    _master_weights_of,
)
from paddlefleet.trainer.trainer_utils import ShardingOption

SEED = 2026
# fp8 quant kernels need every GEMM dim to be a multiple of 128.
HIDDEN, VOCAB, SEQ, MICRO_BATCH = 256, 256, 128, 2
CACHE_ATTRS = (
    # MoELayer.fp8_quant_weight -> grouped_gemm_experts.weight1/2
    "fp8_weight_stacked",
    "fp8_scale_stacked",
    # Linear.fp8_quant_weight -> attention / shared-expert weights
    "fp8_weight_fwd",
    "fp8_scale_fwd",
    "fp8_scale_bwd",
)


def _sm_major():
    if not paddle.is_compiled_with_cuda():
        return 0
    return paddle.device.cuda.get_device_capability()[0]


def _paddlefleet_ops_available():
    try:
        from paddlefleet_ops import fuse_stack_fp8_quant  # noqa: F401
    except (ImportError, RuntimeError):
        return False
    return True


def _init_fleet():
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": 1,
        "pp_degree": 1,
        "sharding_degree": dist.get_world_size(),
        "sep_degree": 1,
        "cp_degree": 1,
        "ep_degree": 1,
        "moe_sharding_degree": 1,
        "order": [
            "sharding",
            "moe_sharding",
            "pp",
            "sep",
            "cp",
            "dp",
            "ep",
            "mp",
        ],
    }
    fleet.init(is_collective=True, strategy=strategy)
    ps.initialize_model_parallel(fleet.get_hybrid_communicate_group())
    return strategy


def _build_model():
    paddle.seed(SEED)
    tensor_parallel.model_parallel_cuda_manual_seed(SEED)
    config = GPTConfig(
        num_hidden_layers=2,
        hidden_size=HIDDEN,
        vocab_size=VOCAB,
        max_sequence_length=SEQ,
        num_attention_heads=2,
        intermediate_size=HIDDEN,
        normalization="RMSNorm",
        hidden_dropout_prob=0.0,
        attention_dropout=0.0,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=1,
        moe_intermediate_size=HIDDEN,
        moe_token_dispatcher_type="alltoall",
        # fp8 + fused experts + deep_gemm -> grouped_gemm_experts
        moe_expert_fusion=True,
        use_bias=False,
        fp8="e4m3",
        # also pre-quantize non-expert Linears (fp8_weight_fwd)
        full_fp8_computation=True,
        # the sm90 fp8 wgrad GEMM is not needed to exercise this path
        fp8_wgrad=False,
        params_dtype=paddle.bfloat16,
        tie_word_embeddings=True,
        init_method=functools.partial(paddle.nn.init.xavier_uniform_, gain=1.0),
        output_layer_init_method=functools.partial(
            paddle.nn.init.xavier_uniform_, gain=1.0
        ),
    )
    model = gpt_builder(config, num_stages=1)
    # FSDP slices each rank's own copy, so the ranks must agree on the init.
    for param in model.parameters():
        dist.broadcast(param, src=0)
    return model


def _fp8_caches(model):
    """``{param_name: {attr: float32 ndarray}}`` of every fp8 cache present."""
    caches = {}
    for param in model.parameters():
        found = {}
        for attr in CACHE_ATTRS:
            value = getattr(param, attr, None)
            if value is not None:
                found[attr] = value.astype("float32").numpy()
        if found:
            caches[param.name] = found
    return caches


def _expert_params(model):
    return [
        p for n, p in model.named_parameters() if "grouped_gemm_experts" in n
    ]


def _buffer_state(context):
    groups = context.buffer_manager.buffer_groups
    return (
        [g.params_buffer.status for g in groups],
        [g.params_buffer.tmp_data_buffer is None for g in groups],
        context.comm_manager.buffer_cnt_in_using,
    )


def _clear_fp8_caches(model):
    for param in model.parameters():
        for attr in CACHE_ATTRS:
            if hasattr(param, attr):
                delattr(param, attr)


def _args(offload_master):
    return SimpleNamespace(
        using_sonic_moe=False,
        sharding=[ShardingOption.FSDP],
        sharding_parallel_size=dist.get_world_size(),
        offload_fp8_expert_master_weight=offload_master,
    )


def _batch():
    ids = paddle.randint(0, VOCAB, [MICRO_BATCH, SEQ])
    pos = paddle.arange(SEQ, dtype="int64").tile([MICRO_BATCH, 1])
    mask = paddle.ones((MICRO_BATCH, 1, SEQ, SEQ), dtype=bool)
    return (
        {"input_ids": [ids], "position_ids": [pos], "attention_mask": [mask]},
        [ids],
    )


@unittest.skipUnless(
    _sm_major() >= 9 and _paddlefleet_ops_available(),
    "fp8 grouped GEMM needs sm90+ and paddlefleet_ops",
)
@unittest.skipUnless(
    hasattr(fully_shard_fusion.FullyShardFusion, "unshard_params"),
    "installed paddle lacks FullyShardFusion.unshard_params",
)
class TestFSDPFP8QuantWeight(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert dist.get_world_size() > 1, "needs more than one rank"
        cls.strategy = _init_fleet()
        cls.model = _build_model()

        # Non-fsdp reference: the same weights quantized whole-model.
        cls.model.fp8_quant_weight(True, quant_transpose=False)
        cls.reference = _fp8_caches(cls.model)
        _clear_fp8_caches(cls.model)

        # At ep=1 paddlefleet does not tag experts; fsdp groups them by tag.
        for param in _expert_params(cls.model):
            param.is_moe_param = True
        fully_shard(cls.model, enable_tensor_fusion_and_overlap=True)
        cls.context = get_fsdp_context()
        cls.optimizer = mix_precision_utils.MixPrecisionOptimizer(
            paddle.optimizer.AdamW(
                learning_rate=1e-3,
                parameters=cls.model.parameters(),
                multi_precision=True,
            )
        )
        cls.callback = FP8QuantWeightCallback()

    def _kwargs(self):
        return {"model": self.model, "optimizer": self.optimizer}

    def _expert_groups(self):
        return [
            g
            for g in self.context.buffer_manager.buffer_groups
            if g.is_expert_param
        ]

    def test_a_private_fsdp_surface(self):
        """Fail here, not at the first training step, if paddle renames."""
        self.assertIsNotNone(self.context, "fully_shard registered no context")
        self.assertTrue(callable(getattr(self.context, "unshard_params", None)))
        groups = self.context.buffer_manager.buffer_groups
        self.assertTrue(groups)
        for group in groups:
            self.assertIsInstance(group.is_expert_param, bool)
            self.assertTrue(
                group.params_buffer.data_buffer.name.startswith("fuse_params_")
            )
            self.assertTrue(group.params)
        expert_names = {p.name for g in self._expert_groups() for p in g.params}
        self.assertEqual(
            expert_names, {p.name for p in _expert_params(self.model)}
        )

    def test_b_prequant_matches_non_fsdp(self):
        _clear_fp8_caches(self.model)
        self.assertEqual(_fp8_caches(self.model), {})
        experts = _expert_params(self.model)
        self.assertTrue(experts)
        # Outside forward a sharded fsdp param has no storage of its own.
        self.assertFalse(any(p._is_initialized() for p in experts))
        state_before = _buffer_state(self.context)

        self.callback.on_step_begin(
            _args(offload_master=False), None, None, **self._kwargs()
        )

        got = _fp8_caches(self.model)
        self.assertTrue(
            any("fp8_weight_stacked" in c for c in self.reference.values())
        )
        self.assertTrue(
            any("fp8_weight_fwd" in c for c in self.reference.values())
        )
        self.assertEqual(sorted(got), sorted(self.reference))
        for name, ref_attrs in self.reference.items():
            self.assertEqual(sorted(got[name]), sorted(ref_attrs), msg=name)
            for attr, ref in ref_attrs.items():
                np.testing.assert_array_equal(
                    got[name][attr], ref, err_msg=f"{name}.{attr}"
                )

        # The per-layer gather is fully undone.
        self.assertEqual(_buffer_state(self.context), state_before)
        self.assertFalse(any(p._is_initialized() for p in experts))

    def test_c_offload_reload_round_trip(self):
        args = _args(offload_master=True)
        kwargs = self._kwargs()
        names = _fsdp_expert_master_weight_names()
        for group in self._expert_groups():
            self.assertIn(group.params_buffer.data_buffer.name, names)
            for param in group.params:
                self.assertIn(param.name, names)

        # Step 1: master weights are created lazily by the optimizer step,
        # so there is nothing to offload yet.
        self.assertEqual(dict(_master_weights_of(self.optimizer)), {})
        self.callback.on_step_begin(args, None, None, **kwargs)
        self.assertEqual(self.callback.moe_weights_name, names)
        pipe = NoPipelineParallel(self.model, self.strategy)
        loss = pipe.forward_backward_pipeline(_batch())
        self.assertTrue(np.isfinite(float(loss)))
        self.callback.on_optimizer_begin(args, None, None, **kwargs)
        self.optimizer.step()
        self.optimizer.clear_grad()

        master_weights = _master_weights_of(self.optimizer)
        hit = [n for n in names if n in master_weights]
        self.assertTrue(hit, f"{names} hit none of {list(master_weights)}")
        for group in self._expert_groups():
            self.assertIn(group.params_buffer.data_buffer.name, hit)
        others = [n for n in master_weights if n not in hit]
        self.assertTrue(others, "expected non-expert master weights too")
        for name in master_weights:
            self.assertTrue(master_weights[name].place.is_gpu_place(), name)
        tensors = {n: master_weights[n] for n in master_weights}
        values = {n: t.numpy() for n, t in tensors.items()}

        self.callback.on_step_begin(args, None, None, **kwargs)
        for name in hit:
            place = master_weights[name].place
            self.assertTrue(place.is_cuda_pinned_place(), f"{name}: {place}")
        for name in others:
            self.assertTrue(master_weights[name].place.is_gpu_place(), name)

        self.callback.on_optimizer_begin(args, None, None, **kwargs)
        for name, tensor in tensors.items():
            self.assertIs(master_weights[name], tensor, name)
            self.assertTrue(tensor.place.is_gpu_place(), name)
            np.testing.assert_array_equal(
                tensor.numpy(), values[name], err_msg=name
            )


if __name__ == "__main__":
    unittest.main()
