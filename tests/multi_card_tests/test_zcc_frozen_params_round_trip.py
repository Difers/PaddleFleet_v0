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

"""Multi-card round trip of the ZCC model_state with a frozen backbone.

With `train_projector_only` and friends every model param is frozen, so the
sharding shell's param lists are empty and the filter
`_muon_manipulate_sharded_state_dict` has to classify the params itself.
Both MuonShardingOptimizer and MixedShardingOptimizer are covered.
Routed experts are split over the expert-parallel group, which here is the
sharding axis with moe_sharding_degree 1, so each rank's expert shard is the
only copy.

Every rank runs the real filter, the real `saved_ckptmeta` and the real
worker writer `_save_model_state`, then reads the files back into
sentinel-filled targets with `dist.load_state_dict` and compares every key.
A shard dropped on save comes back as the sentinel.

Run with:
  python -m paddle.distributed.launch --gpus 0,1,2,3 \
      tests/multi_card_tests/test_zcc_frozen_params_round_trip.py
"""

import os
import shutil
import sys
import unittest
import zlib
from collections import Counter
from types import SimpleNamespace

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ),
)

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_optimizers.mixed_sharding_optimizer import (
    MixedShardingOptimizer,
)
from paddle.distributed.fleet.meta_optimizers.muon_sharding_optimizer import (
    MuonShardingOptimizer,
)
from paddle.distributed.flex_checkpoint.dcp.sharded_weight import ShardedWeight
from paddle.optimizer.mixed_inner_optimizer import MixedInnerOptimizer

from paddlefleet.trainer.utils.zero_cost_checkpoint import (
    ZeroCostCheckpointCallbackFcBased,
    ZeroCostCheckpointWorkerFcBased,
    saved_ckptmeta,
)
from paddlefleet.utils.env import MODEL_STATE_DIC

# Replicated 2D params, one per rank so the greedy partition gives every rank
# exactly one to own.
REPLICATED_2D = (
    "model.embed_tokens.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.o_proj.weight",
    "lm_head.weight",
)
NORM = "model.layers.0.input_layernorm.weight"
BUFFER = "model.layers.0.mlp.gate.e_score_correction_bias"
EP_FUSED = "model.layers.0.mlp.grouped_gemm_experts.weight1"
EXPERT_ROWS = 2
COLS = 4
SENTINEL = -1.0
SHELLS = (MuonShardingOptimizer, MixedShardingOptimizer)


def _unfused_expert(expert_id):
    return f"model.layers.0.mlp.experts.{expert_id}.up_gate_proj.weight"


def _reference(key, shape):
    """Deterministic content of the whole tensor, distinct per key."""
    base = (zlib.crc32(key.encode()) % 1000) * 100.0 + 1.0
    numel = int(np.prod(shape))
    return (np.arange(numel, dtype="float32") + base).reshape(shape)


def _layout(rank, world):
    """(key, local_shape, global_shape, global_offset, kind) on this rank.

    kind is "2d" (Muon keeps it whole), "1d" (element-wise), "expert" (2D
    in the moe_expert color group) or "buffer" (not a parameter).
    """
    ep_global = (EXPERT_ROWS * world, COLS)
    out = [(k, (8, COLS), (8, COLS), (0, 0), "2d") for k in REPLICATED_2D]
    out += [
        (NORM, (8,), (8,), (0,), "1d"),
        (BUFFER, (8,), (8,), (0,), "buffer"),
        # moe_expert_fusion=True: one row block per EP rank.
        (EP_FUSED, (EXPERT_ROWS, COLS), ep_global, (rank * EXPERT_ROWS, 0)),
        # moe_expert_fusion=False: experts.{rank} only exists on its owner,
        # with a descriptor that looks like a full replica.
        (_unfused_expert(rank), (COLS, COLS), (COLS, COLS), (0, 0)),
    ]
    return [e if len(e) == 5 else (*e, "expert") for e in out]


def _local_reference(local_shape, global_shape, global_offset, key):
    box = tuple(slice(o, o + s) for o, s in zip(global_offset, local_shape))
    return _reference(key, global_shape)[box]


def _sentinel_state_dict(rank, world):
    """Load targets for this rank, filled with SENTINEL."""
    return {
        key: ShardedWeight(
            key=key,
            local_tensor=paddle.full(local_shape, SENTINEL, dtype="float32"),
            local_shape=local_shape,
            global_shape=global_shape,
            global_offset=offset,
        )
        for key, local_shape, global_shape, offset, _ in _layout(rank, world)
    }


class _Model:
    """Frozen params holding the reference values, plus one buffer."""

    def __init__(self, rank, world):
        # moe_sharding_degree == 1: the expert's comm group is this rank alone.
        expert_group = SimpleNamespace(ranks=[rank], rank=0, nranks=1)
        self._params = []
        self._sharded_state_dict = {}
        for key, local_shape, global_shape, offset, kind in _layout(
            rank, world
        ):
            value = _local_reference(local_shape, global_shape, offset, key)
            if kind == "buffer":
                tensor = paddle.to_tensor(value)
            else:
                tensor = paddle.create_parameter(
                    list(local_shape),
                    "float32",
                    default_initializer=paddle.nn.initializer.Assign(value),
                )
                tensor.stop_gradient = True
                if kind == "expert":
                    tensor.color = {
                        "color": "moe_expert",
                        "group": expert_group,
                    }
                self._params.append(tensor)
            self._sharded_state_dict[key] = ShardedWeight(
                key=key,
                local_tensor=tensor,
                local_shape=local_shape,
                global_shape=global_shape,
                global_offset=offset,
            )

    def parameters(self):
        return list(self._params)

    def sharded_state_dict(self):
        return dict(self._sharded_state_dict)


def _info_map(model):
    return {
        p.name: SimpleNamespace(use_muon=len(p.shape) == 2, use_hyperball=False)
        for p in model.parameters()
    }


def _frozen_backbone_shell(shell_cls, hcg, model):
    """The sharding shell as it is left when every param is frozen.

    Frozen params never enter its param lists and there are no master
    weights. The constructor is skipped; the color-group and partition code
    the filter calls is Paddle's. Under Mixed the per-param info is the full
    map the trainer hands MixedInnerOptimizer.
    """
    shell = object.__new__(shell_cls)
    if shell_cls is MixedShardingOptimizer:
        inner = object.__new__(MixedInnerOptimizer)
        inner._sub_opts = []
        inner._multi_precision = True
        inner._param_info_map = _info_map(model)
        shell._inner_opt = inner
    else:
        shell._inner_opt = SimpleNamespace(
            _multi_precision=True, _master_weights={}
        )
        shell._muon_param_info_map = _info_map(model)
    shell._hcg = hcg
    shell._sharding_rank = hcg.get_sharding_parallel_rank()
    shell._local_2d = []
    shell._params_2d_by_color = {}
    shell._params_1d = []
    # Tiny buffers so the greedy partition spreads params over every rank.
    shell.comm_buffer_size_MB = 1e-9
    return shell


class _HostCopies:
    """What ParamFusionStorageHelper.state_dict() returns after an offload:
    a host tensor per key the callback cached for saving."""

    def __init__(self, sharded_state_dict):
        self._tensors = {
            k: sw.local_tensor.cpu() for k, sw in sharded_state_dict.items()
        }

    def state_dict(self):
        return dict(self._tensors)


class TestZCCFrozenParamsRoundTrip(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Sharding-only: the sharding axis doubles as the EP axis.
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": 1,
            "pp_degree": 1,
            "sharding_degree": dist.get_world_size(),
        }
        fleet.init(is_collective=True, strategy=strategy)
        cls.rank = dist.get_rank()
        cls.world = dist.get_world_size()
        assert cls.world > 1, "needs more than one rank"
        cls.hcg = fleet.get_hybrid_communicate_group()
        master = os.environ.get("PADDLE_MASTER", "local").replace(":", "_")
        cls.tmp_root = os.path.join(
            os.environ.get("TMPDIR", "/tmp"), f"zcc_frozen_round_trip_{master}"
        )

    def _save(self, shell_cls, replicate):
        """Filter, build metadata and write model_state as ZCC does.

        Returns the filtered state dict and the model_state directory.
        """
        callback = object.__new__(ZeroCostCheckpointCallbackFcBased)
        callback.args = SimpleNamespace(replicate_saved_into_local=replicate)
        model = _Model(self.rank, self.world)
        filtered = callback._muon_manipulate_sharded_state_dict(
            model, _frozen_backbone_shell(shell_cls, self.hcg, model)
        )
        data_file = f"{self.rank}_0.distcp"
        meta, filter_map = saved_ckptmeta(
            filtered, data_file, replicate_saved_into_local=replicate
        )

        worker = object.__new__(ZeroCostCheckpointWorkerFcBased)
        worker.distcp_file_name = (data_file, "0.metadata")
        worker.dp_rank = 0
        worker.use_expert_parallel = True
        # Only one writer of 0.metadata.
        worker.device_id = self.rank
        worker.param_fusion_storage_helper = _HostCopies(filtered)
        worker.model_state_filter = filter_map
        worker.model_ckpt_meta = meta

        path = os.path.join(
            self.tmp_root, f"{shell_cls.__name__}_replicate_{replicate}"
        )
        if self.rank == 0:
            shutil.rmtree(path, ignore_errors=True)
            os.makedirs(os.path.join(path, MODEL_STATE_DIC))
        dist.barrier()
        worker._save_model_state(path)
        dist.barrier()
        return filtered, os.path.join(path, MODEL_STATE_DIC)

    def _keys_on_disk(self, model_dir):
        return set(
            paddle.load(os.path.join(model_dir, f"{self.rank}_0.distcp"))
        )

    def _reload(self, model_dir):
        """Load the files into sentinel-filled targets of this rank."""
        target = _sentinel_state_dict(self.rank, self.world)
        dist.load_state_dict(target, model_dir)
        return target

    def _assert_restored(self, target):
        """Every element this rank holds came back with its saved value."""
        for key, local_shape, global_shape, offset, _ in _layout(
            self.rank, self.world
        ):
            np.testing.assert_array_equal(
                target[key].local_tensor.numpy(),
                _local_reference(local_shape, global_shape, offset, key),
                err_msg=f"rank {self.rank}: {key} not restored",
            )

    def _round_trip(self, shell_cls, replicate):
        """Save, then reload. Every rank takes the same path through the
        collectives, so a failing rank cannot leave the others hanging."""
        filtered, model_dir = self._save(shell_cls, replicate)
        keys = self._keys_on_disk(model_dir)
        gathered = []
        dist.all_gather_object(gathered, sorted(keys))
        writers = Counter(key for item in gathered for key in item)
        # Loading with a shard missing does not return, so check first. The
        # gathered lists are the same on every rank, hence so is the verdict.
        every_key = set(REPLICATED_2D) | {NORM, BUFFER, EP_FUSED}
        every_key |= {_unfused_expert(r) for r in range(self.world)}
        missing = every_key - set(writers)
        self.assertFalse(missing, f"never written: {sorted(missing)}")
        self.assertEqual(writers[EP_FUSED], self.world)
        target = self._reload(model_dir)
        return set(filtered), keys, writers, target

    def _own(self):
        return {EP_FUSED, _unfused_expert(self.rank)}

    def test_round_trip_dedups_replicas(self):
        for shell_cls in SHELLS:
            with self.subTest(shell=shell_cls.__name__):
                self._check_dedups_replicas(shell_cls)

    def _check_dedups_replicas(self, shell_cls):
        filtered, keys, writers, target = self._round_trip(
            shell_cls, replicate=False
        )
        # Equal sizes: the greedy partition deals the 2D params round robin.
        expected = (
            self._own() | {NORM} | set(REPLICATED_2D[self.rank :: self.world])
        )
        if self.rank == 0:
            expected.add(BUFFER)
        self.assertEqual(filtered, expected)
        # Experts are each rank's only copy; everything else is written once.
        self.assertLessEqual(self._own(), keys)
        for key in (*REPLICATED_2D, NORM, BUFFER):
            self.assertEqual(writers[key], 1, key)
        self._assert_restored(target)

    def test_round_trip_with_local_replicas(self):
        for shell_cls in SHELLS:
            with self.subTest(shell=shell_cls.__name__):
                filtered, keys, _, target = self._round_trip(
                    shell_cls, replicate=True
                )
                everything = {key for key, *_ in _layout(self.rank, self.world)}
                self.assertEqual(filtered, everything)
                self.assertLessEqual(self._own() | set(REPLICATED_2D), keys)
                self._assert_restored(target)

    @classmethod
    def tearDownClass(cls):
        dist.barrier()
        if cls.rank == 0:
            shutil.rmtree(cls.tmp_root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
