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
"""2-card integration test: PaddleFleet's HPO + Paddle's ``MixedShardingOptimizer``.

The shell's own guards (TP / PP overlap / comm_group_call_opt / undeclared
``keep_whole`` / missing weight_decay) are unit-tested in Paddle. Here a
MixedInnerOptimizer whose subs use a NON-AdamW state layout goes through
MoEHybridParallelOptimizer, and the test checks that:
  0. the HPO picks MixedShardingOptimizer for a MixedInnerOptimizer;
  1. each sub's declared ``keep_whole`` decides V1 (whole tensor) vs V2;
  2. every rank derives the same V1 owner mapping and parameters stay
     bit-identical across ranks after real steps (``_hybrid_sync_grad``);
  3. set_state_dict / sharded_state_dict follow optimizer_state_layout().

Run with:
  python -m paddle.distributed.launch --gpus=0,1 \
      tests/multi_card_tests/test_mixed_sharding_optimizer.py
"""

import random
import unittest

import numpy as np
import paddle
from paddle.distributed import fleet

from paddlefleet.training.initialize import initialize_fleet
from paddlefleet.utils.moe_hybrid_parallel_optimizer import (
    MoEHybridParallelOptimizer,
)

try:
    from paddle.distributed.fleet.meta_optimizers.mixed_sharding_optimizer import (
        MixedShardingOptimizer,
    )
    from paddle.optimizer.inner_optimizer import InnerOptimizer
    from paddle.optimizer.mixed_inner_optimizer import MixedInnerOptimizer

    HAS_MIXED_OPTIMIZER = True
except (ImportError, ModuleNotFoundError):
    # Paddle build without the inner-optimizer API: the whole test is skipped,
    # so the stubs below only need a base class that can be subclassed.
    HAS_MIXED_OPTIMIZER = False
    InnerOptimizer = object
    MixedInnerOptimizer = MixedShardingOptimizer = None

SHARDING_DEGREE = 2
DIM = 64
BATCH_SIZE = 8
STEPS = 3
STATE_SUFFIXES = {"velocity_0", "step_count_0"}
LAYERS = ("whole_fc", "slice_fc", "whole2_fc")

_strategy = None


def _ensure_fleet():
    global _strategy
    if _strategy is not None:
        return _strategy
    strategy = fleet.DistributedStrategy()
    # Same EP-aware topology as moe/test_muon_ep_slice.py (MoE HPO needs the
    # expert/moe_sharding groups); the model itself is dense.
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": 1,
        "pp_degree": 1,
        "sharding_degree": SHARDING_DEGREE,
        "sep_degree": 1,
        "cp_degree": 1,
        "ep_degree": SHARDING_DEGREE,
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
    sharding_configs = strategy.hybrid_configs["sharding_configs"]
    sharding_configs.accumulate_steps = 1
    sharding_configs.comm_buffer_size_MB = 1
    sharding_configs.split_param = True
    initialize_fleet(strategy=strategy)
    _strategy = strategy
    return strategy


class _MomentumSGD(InnerOptimizer):
    """Element-wise momentum SGD with a vector and a scalar state."""

    def __init__(self, parameters, momentum=0.9, **kwargs):
        super().__init__(parameters=parameters, **kwargs)
        self._momentum = momentum

    def _ensure_accumulators(self, param):
        if (
            "velocity" in self._accumulators
            and param.name in self._accumulators["velocity"]
        ):
            return
        self._ensure_master_weight(param)
        self._add_accumulator(
            "velocity", param, dtype=paddle.float32, shape=param.shape
        )
        self._add_accumulator(
            "step_count", param, dtype=paddle.float32, shape=[1]
        )

    def _update_parameters(self, params_grads, lr):
        with paddle.no_grad():
            for param, grad in params_grads:
                v = self._get_accumulator("velocity", param)
                s = self._get_accumulator("step_count", param)
                paddle.assign(self._momentum * v + grad.astype(v.dtype), v)
                paddle.assign(s + 1.0, s)
                step = self._effective_lr(param, lr) * v
                param.subtract_(step.astype(param.dtype))

    def optimizer_state_layout(self):
        return {
            "vector": ["velocity_0"],
            "scalar": ["step_count_0"],
            "master": "fp32_master_0",
        }


class _WholeSGD(_MomentumSGD):
    """Same math, declared V1 (whole tensor on one owner rank)."""

    keep_whole = True


class _SlicedSGD(_MomentumSGD):
    """Same math, declared V2 (element-wise shards)."""

    keep_whole = False


class _Net(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.whole_fc = paddle.nn.Linear(DIM, DIM, bias_attr=False)
        self.slice_fc = paddle.nn.Linear(DIM, DIM, bias_attr=False)
        self.whole2_fc = paddle.nn.Linear(DIM, DIM, bias_attr=False)

    def forward(self, x):
        return self.whole2_fc(self.slice_fc(self.whole_fc(x))).mean()


@unittest.skipUnless(
    HAS_MIXED_OPTIMIZER, "Paddle build without the inner-optimizer API"
)
class TestMixedShardingOptimizer(unittest.TestCase):
    def setUp(self):
        random.seed(2026)
        np.random.seed(2026)
        paddle.seed(2026)
        self.strategy = _ensure_fleet()
        self.hcg = fleet.get_hybrid_communicate_group()
        self.data = [
            np.random.random_sample((BATCH_SIZE, DIM)).astype("float32")
            for _ in range(STEPS + 1)
        ]

    def _mixed(self, net):
        whole = net.whole_fc.weight
        sliced = net.slice_fc.weight
        whole2 = net.whole2_fc.weight
        sub_a = _WholeSGD(
            parameters=[whole, whole2],
            learning_rate=0.01,
            weight_decay=0.0,
        )
        sub_b = _SlicedSGD(
            parameters=[sliced],
            learning_rate=0.01,
            weight_decay=0.0,
            momentum=0.5,
        )
        return MixedInnerOptimizer([sub_a, sub_b], learning_rate=0.01)

    def _step(self, model, optimizer, idx):
        rank = self.hcg.get_sharding_parallel_rank()
        local_bs = BATCH_SIZE // SHARDING_DEGREE
        start = rank * local_bs
        batch = paddle.to_tensor(self.data[idx][start : start + local_bs])
        model(batch).backward()
        optimizer.step()
        optimizer.clear_grad()

    @staticmethod
    def _accumulator_values(shell):
        return {
            var.name: var.numpy().copy()
            for accs in shell._inner_opt.merged_accumulators().values()
            for var in accs.values()
        }

    def test_mixed_sharding(self):
        net = _Net()
        model = fleet.distributed_model(net)
        optimizer = MoEHybridParallelOptimizer(
            self._mixed(net), self.hcg, self.strategy
        )
        shell = optimizer._inner_opt
        # 0. HPO picks the PaddleFleet shell for a MixedInnerOptimizer.
        self.assertIsInstance(shell, MixedShardingOptimizer)
        whole = net.whole_fc.weight.name
        sliced = net.slice_fc.weight.name
        whole2 = net.whole2_fc.weight.name

        # 1. V1 / V2 classification from each sub's declared keep_whole.
        v1 = {
            p.name
            for params in shell._params_2d_by_color.values()
            for p in params
        }
        v2 = set()
        for buffer in shell._comm_buffer_list:
            v2.update(buffer._sharding_param_grad_view.keys())
        self.assertEqual(v1, {whole, whole2})
        self.assertEqual(v2, {sliced})

        # 2. Same owner mapping on every rank; identical params after steps.
        owner_map = {
            str(color): dict(name2rank)
            for color, name2rank in shell._param2rank_2d_by_color.items()
        }
        gathered = []
        paddle.distributed.all_gather_object(gathered, owner_map)
        for other in gathered:
            self.assertEqual(other, gathered[0])
        for idx in range(STEPS):
            self._step(model, optimizer, idx)
        for name, param in net.named_parameters():
            vals = []
            paddle.distributed.all_gather(vals, param)
            for other in vals:
                np.testing.assert_array_equal(
                    other.numpy(), vals[0].numpy(), err_msg=name
                )

        # 3. set_state_dict restores the layout-named states.
        state = shell.state_dict()
        saved = self._accumulator_values(shell)
        self.assertTrue(saved)
        for key in saved:
            self.assertTrue(
                any(key.endswith("_" + s) for s in STATE_SUFFIXES), key
            )
        snapshot = {k: v.clone() for k, v in state.items()}
        self._step(model, optimizer, STEPS)
        shell.set_state_dict(snapshot)
        restored = self._accumulator_values(shell)
        self.assertEqual(set(restored), set(saved))
        for key, value in saved.items():
            np.testing.assert_array_equal(restored[key], value, err_msg=key)

        # 4. sharded_state_dict uses the custom suffixes; across ranks every
        # (param, state) pair is present exactly where its shard lives.
        sharded = shell.sharded_state_dict(net.sharded_state_dict())
        for key, weight in sharded.items():
            self.assertIn(key.rsplit(".", 1)[1], STATE_SUFFIXES, key)
            if key == "slice_fc.weight.velocity_0":
                self.assertTrue(weight.is_flattened)
        all_keys = []
        paddle.distributed.all_gather_object(all_keys, sorted(sharded))
        self.assertEqual(
            {k for keys in all_keys for k in keys},
            {f"{layer}.weight.{s}" for layer in LAYERS for s in STATE_SUFFIXES},
        )


if __name__ == "__main__":
    unittest.main()
