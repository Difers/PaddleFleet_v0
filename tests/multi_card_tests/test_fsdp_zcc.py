# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

"""Multi-card tests for ZCC under native FSDP."""

import builtins
import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import paddle
import paddle.distributed as dist
from paddle import nn
from paddle.distributed import fleet
from paddle.distributed.fleet.utils import mix_precision_utils
from paddle.distributed.flex_checkpoint.dcp.sharded_weight import ShardedWeight
from paddle.distributed.fsdp._fsdp_context import get_fsdp_context
from paddle.distributed.fsdp.fully_shard import fully_shard

from paddlefleet.trainer import trainer_utils
from paddlefleet.trainer.trainer import Trainer
from paddlefleet.trainer.trainer_utils import ShardingOption, init_optimizer
from paddlefleet.trainer.training_args import TrainingArguments
from paddlefleet.trainer.utils import zero_cost_checkpoint as zcc
from paddlefleet.trainer.utils.zero_cost_checkpoint import (
    ZeroCostCheckpointCallbackFcBased,
    ZeroCostCheckpointWorker,
    ZeroCostCheckpointWorkerFcBased,
    _restore_master_weights_from_unified,
    get_fused_param_mappings,
)

HIDDEN, INTER, LAYERS, STEPS = 16, 32, 2, 3


class TransformerLayer(nn.Layer):
    # class name matters: fully_shard picks units by class name along the MRO
    def __init__(self):
        super().__init__()
        self.up = nn.Linear(HIDDEN, INTER)
        self.down = nn.Linear(INTER, HIDDEN)

    def forward(self, x):
        return x + self.down(nn.functional.silu(self.up(x)))


class TinyModel(nn.Layer):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(HIDDEN, HIDDEN)
        self.layers = nn.LayerList([TransformerLayer() for _ in range(LAYERS)])

    def get_input_embeddings(self):
        return self.embed

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return x.mean()


def _build_sharded_model():
    paddle.seed(2026)
    model = paddle.amp.decorate(
        models=TinyModel(), level="O2", dtype="bfloat16"
    )
    model = fully_shard(model, enable_tensor_fusion_and_overlap=True)
    model = mix_precision_utils.MixPrecisionLayer(model, dtype="bfloat16")
    optimizer = mix_precision_utils.MixPrecisionOptimizer(
        paddle.optimizer.AdamW(
            learning_rate=1e-3,
            parameters=[p for p in model.parameters() if p.trainable],
            weight_decay=0.0,
            multi_precision=True,
        )
    )
    for _ in range(STEPS):
        with paddle.amp.auto_cast(level="O2", dtype="bfloat16"):
            loss = model(paddle.randn([8, HIDDEN]))
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
    return model, optimizer


def _init_fleet():
    if getattr(fleet.fleet, "_hcg", None) is not None:
        return
    assert dist.get_world_size() > 1, "needs more than one rank"
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": 1,
        "pp_degree": 1,
        "sharding_degree": dist.get_world_size(),
    }
    fleet.init(is_collective=True, strategy=strategy)


class TestZCCFsdpCheckpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _init_fleet()

        cls.model, cls.optimizer = _build_sharded_model()
        cls.context = get_fsdp_context()

        callback = ZeroCostCheckpointCallbackFcBased.__new__(
            ZeroCostCheckpointCallbackFcBased
        )
        cls.mapping, cls.slice_info = callback._gen_unified_name_fsdp(
            cls.context, cls.optimizer
        )
        cls.handed_out = cls.context.optimizer_sharded_state_dict(cls.optimizer)
        state = dict(cls.optimizer.state_dict())
        cls.master_weights = state.pop("master_weights", None) or {}
        state.pop("LR_Scheduler", None)
        cls.sources = {**state, **cls.master_weights}

        root = [None]
        if dist.get_rank() == 0:
            root = [tempfile.mkdtemp(prefix="zcc_fsdp_ckpt_")]
        dist.broadcast_object_list(root, src=0)
        cls.ckpt_root = root[0]
        cls.save_dir = os.path.join(cls.ckpt_root, "master_weights")
        if dist.get_rank() == 0:
            os.makedirs(cls.save_dir, exist_ok=True)
        dist.barrier()

        cls.saved_w0 = {
            k: v for k, v in cls.handed_out.items() if k.endswith(".w_0")
        }
        dist.save_state_dict(cls.saved_w0, cls.save_dir)
        dist.barrier()

    @classmethod
    def tearDownClass(cls):
        dist.barrier()
        if dist.get_rank() == 0:
            shutil.rmtree(cls.ckpt_root, ignore_errors=True)

    def test_save_resume_round_trip(self):
        self.assertEqual(type(self.context).__name__, "FullyShardFusion")
        self.assertEqual(self.slice_info, {})
        self.assertTrue(self.mapping, "plan must not be empty")

        for src_name, pieces in self.mapping.items():
            source = self.sources[src_name].flatten()
            for unified_key, begin, end in pieces:
                np.testing.assert_array_equal(
                    source[begin:end].numpy(),
                    self.handed_out[unified_key].local_tensor.flatten().numpy(),
                    err_msg=f"{unified_key} offset {begin}:{end} of {src_name}",
                )

        target = {
            k: ShardedWeight(
                key=sw.key,
                local_tensor=paddle.zeros_like(sw.local_tensor),
                local_shape=sw.local_shape,
                global_shape=sw.global_shape,
                global_offset=sw.global_offset,
                is_flattened=getattr(sw, "is_flattened", False),
                flattened_range=getattr(sw, "flattened_range", None),
            )
            for k, sw in self.saved_w0.items()
        }
        dist.load_state_dict(target, self.save_dir)
        for k, sw in self.saved_w0.items():
            np.testing.assert_array_equal(
                target[k].local_tensor.numpy(),
                sw.local_tensor.numpy(),
                err_msg=f"disk round trip changed {k}",
            )

        split = {
            name: pieces
            for name, pieces in self.mapping.items()
            if name in self.master_weights and len(pieces) > 1
        }
        self.assertTrue(split, "expected a fused master-weight bucket")
        name, pieces = max(split.items(), key=lambda kv: len(kv[1]))

        loaded = {uk: target[uk].local_tensor for uk, _, _ in pieces}
        restored = _restore_master_weights_from_unified({name: pieces}, loaded)[
            name
        ]
        covered = max(end for _, _, end in pieces)
        self.assertEqual(int(restored._numel()), covered)
        np.testing.assert_array_equal(
            restored.numpy(),
            self.master_weights[name].flatten()[:covered].numpy(),
        )

    def test_fused_param_plumbing(self):
        state = dict(sorted(self.model.sharded_state_dict().items()))
        holder = []
        mappings, ipc_metas = get_fused_param_mappings(
            self.optimizer, state, staging_holder=holder
        )
        self.assertEqual(set(mappings), set(state))
        self.assertTrue(ipc_metas)
        self.assertEqual(len(holder), 1)
        staging = holder[0]
        for key, meta in mappings.items():
            local = state[key].local_tensor
            buffer = staging.buffers[local.dtype]
            self.assertLessEqual(meta["end"], int(buffer._numel()))
            np.testing.assert_array_equal(
                buffer[meta["start"] : meta["end"]].numpy(),
                local.flatten().numpy(),
                err_msg=f"staged copy mismatch for {key}",
            )

        name, pieces = next(
            (n, p) for n, p in self.mapping.items() if len(p) > 1
        )
        end = max(e for _, _, e in pieces)
        worker = ZeroCostCheckpointWorkerFcBased.__new__(
            ZeroCostCheckpointWorkerFcBased
        )
        worker.unified_name_mapping = {name: pieces}
        renamed = worker._replace_pname_with_unified(
            {name: paddle.arange(0, end, dtype="float32")}
        )
        self.assertEqual(
            list(renamed), [unified_key for unified_key, _, _ in pieces]
        )
        for unified_key, begin, end in pieces:
            np.testing.assert_array_equal(
                renamed[unified_key].numpy(),
                np.arange(begin, end, dtype="float32"),
            )

        fresh = mix_precision_utils.MixPrecisionOptimizer(
            paddle.optimizer.AdamW(
                learning_rate=1e-3,
                parameters=[p for p in self.model.parameters() if p.trainable],
                multi_precision=True,
            )
        )

        def _accumulators(opt):
            return {
                k
                for k in dict(opt.state_dict())
                if k not in ("master_weights", "LR_Scheduler")
            }

        self.assertEqual(_accumulators(fresh), set())
        init_optimizer(fresh, self.model.sharded_state_dict(), {})
        self.assertTrue(_accumulators(fresh))


class TestZCCFsdpUnits(unittest.TestCase):
    """FSDP branches reachable without a live context, driven with doubles."""

    @staticmethod
    def _args(*, sharding=(), fmt=None, auto=False):
        args = TrainingArguments.__new__(TrainingArguments)
        args.enable_auto_parallel = auto
        args.sharding = list(sharding)
        args.sharding_parallel_size = 1
        args.save_checkpoint_format = fmt
        return args

    def test_config_gates(self):
        flex = "flex_checkpoint"
        fsdp = [ShardingOption.FSDP]
        enabled = self._args(sharding=fsdp, fmt=flex)
        self.assertTrue(enabled.zcc_fsdp_mode)
        self.assertTrue(enabled.zcc_supported_sharding)
        for args in (
            self._args(sharding=fsdp, fmt=flex, auto=True),
            self._args(sharding=fsdp, fmt="sharding_io"),
            self._args(),
        ):
            self.assertFalse(args.zcc_fsdp_mode)
            self.assertFalse(args.zcc_supported_sharding)

        real_import = builtins.__import__

        def _blocked(name, *args, **kwargs):
            if name == "paddle.distributed.fsdp._fsdp_context":
                raise ImportError("blocked for this test")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_blocked):
            self.assertIsNone(trainer_utils._get_fsdp_context())

    def test_worker_branches(self):
        callback = ZeroCostCheckpointCallbackFcBased.__new__(
            ZeroCostCheckpointCallbackFcBased
        )
        callback.manager = SimpleNamespace(cache_version=7)
        callback.muon_param_staging = None
        inner_opt = SimpleNamespace(
            fusion_storage=None,
            fused_buffer_version=7,
            use_fusion_storage=MagicMock(),
            _maybe_refuse=MagicMock(),
        )
        source = paddle.arange(0, 2, dtype="float32")
        context = SimpleNamespace(
            optimizer_sharded_state_dict=lambda opt: {
                "u.w_0": SimpleNamespace(local_tensor=source)
            }
        )
        with patch.object(zcc, "_get_fsdp_context", return_value=context):
            callback.maybe_update_zcc_worker(
                SimpleNamespace(), MagicMock(), inner_opt, 1
            )
            mapping, slice_info = callback._gen_unified_name(
                SimpleNamespace(state_dict=lambda: {"m": source}), None
            )
        inner_opt.use_fusion_storage.assert_called_once_with()
        inner_opt._maybe_refuse.assert_called_once_with()
        self.assertEqual(mapping, {"m": [("u.w_0", 0, 2)]})
        self.assertEqual(slice_info, {})

        base = ZeroCostCheckpointWorker.__new__(ZeroCostCheckpointWorker)
        base.optimizer_fusion_storage_helper = SimpleNamespace(
            master_weights_meta={}
        )
        base.unified_name_mapping = {
            "layer.weight": [
                ("fuse_params_0.w_0", 0, 2),
                ("fuse_params_1.w_0", 2, 4),
            ]
        }
        restored = base._reverse_unified_name_for_ema(
            {
                "fuse_params_0.w_0": paddle.to_tensor([1.0, 2.0]),
                "fuse_params_1.w_0": paddle.to_tensor([3.0, 4.0]),
                "other.weight": paddle.to_tensor([9.0]),
            }
        )
        self.assertEqual(sorted(restored), ["master_weights", "other.weight"])
        np.testing.assert_array_equal(
            restored["master_weights"]["layer.weight"].numpy(),
            np.array([1, 2, 3, 4], dtype="float32"),
        )

        fc = ZeroCostCheckpointWorkerFcBased.__new__(
            ZeroCostCheckpointWorkerFcBased
        )
        fc.optimizer_fusion_storage_helper = (
            base.optimizer_fusion_storage_helper
        )
        fc.zcc_ema_processor = MagicMock()
        fc.unified_name_mapping = {
            "layer.weight": "fuse_params_0.w_0",
            "bias": "bias_unified",
        }
        fc.use_expert_parallel = False
        fc.dp_rank = 0
        with (
            patch.object(
                zcc.paddle.base.core.LoDTensor,
                "_new_shared_filename",
                staticmethod(lambda meta: None),
            ),
            patch.object(
                zcc.paddle,
                "to_tensor",
                side_effect=[
                    paddle.to_tensor([1.0, 2.0]),
                    paddle.to_tensor([3.0]),
                ],
            ),
        ):
            fc._load_ema_from_shared_memory(
                {
                    "fuse_params_0.w_0": {"shared_meta": b"m", "shape": [2]},
                    "bias_unified": {"shared_meta": b"p", "shape": [1]},
                }
            )
        loaded = fc.zcc_ema_processor.load_ema_state_dict.call_args[0][0]
        self.assertEqual(sorted(loaded), ["bias_unified", "master_weights"])
        np.testing.assert_array_equal(
            loaded["master_weights"]["layer.weight"].numpy(),
            np.array([1, 2], dtype="float32"),
        )


class TestTrainerFsdpWiring(unittest.TestCase):
    """``Trainer`` side of the FSDP wiring, driven without ``train()``."""

    @classmethod
    def setUpClass(cls):
        _init_fleet()

    @staticmethod
    def _trainer(model, optimizer, **overrides):
        args = {
            "enable_auto_parallel": False,
            "world_size": dist.get_world_size(),
            "use_hybrid_parallel": True,
            "sharding": [ShardingOption.FSDP],
            "pipeline_model_parallel_size": 1,
            "tensor_model_parallel_size": 1,
            "sep_parallel_size": 1,
            "context_parallel_size": 1,
            "gradient_accumulation_steps": 2,
            "amp_master_grad": False,
            # take the real defaults so the default path is pinned too
            "fsdp_prefetch_units": TrainingArguments.fsdp_prefetch_units,
            "fsdp_comm_unit_size": TrainingArguments.fsdp_comm_unit_size,
        }
        args.update(overrides)
        trainer = Trainer.__new__(Trainer)
        trainer.args = SimpleNamespace(**args)
        trainer.sharding = True
        trainer.model = model
        trainer.optimizer = optimizer
        trainer.amp_dtype = "bfloat16"
        trainer.do_grad_scaling = False
        trainer.using_fleet_model = False
        return trainer

    @staticmethod
    def _optimizer(model, muon_map=None):
        optimizer = paddle.optimizer.AdamW(
            learning_rate=1e-3,
            parameters=model.parameters(),
            weight_decay=0.01,
            apply_decay_param_fun=lambda name: "b_0" not in name,
            multi_precision=True,
        )
        optimizer._muon_param_info_map = muon_map
        return optimizer

    def _run_save_model(self, fsdp_context):
        trainer = self._trainer(TinyModel(), None)
        with tempfile.TemporaryDirectory() as tmp:
            trainer.args.output_dir = tmp
            trainer.args.output_signal_dir = tmp
            trainer.args.should_save_model_state = False
            trainer.args.save_checkpoint_format = None
            trainer.args.unified_checkpoint_config = []
            with (
                patch(
                    "paddlefleet.trainer.trainer._get_fsdp_context",
                    return_value=fsdp_context,
                ),
                patch("paddlefleet.trainer.trainer.logger") as log,
            ):
                trainer.save_model(tmp)
        return log

    def test_fully_shard_with_optimizer(self):
        model = TinyModel()
        optimizer = self._optimizer(model)
        returned = fully_shard(model, optimizer=optimizer, grad_accum_steps=2)
        self.assertIs(returned, model)
        self.assertEqual(type(model._fsdp_context).__name__, "FullyShardFusion")
        self.assertIs(optimizer._fsdp_state_dict_ctx, model._fsdp_context)
        self.assertTrue(model.embed.bias.no_weight_decay)
        self.assertFalse(model.embed.weight.no_weight_decay)

        muon = TinyModel()
        muon_map = {
            p.name: SimpleNamespace(use_muon=p.ndim == 2)
            for p in muon.parameters()
        }
        fully_shard(muon, optimizer=self._optimizer(muon, muon_map))
        self.assertTrue(muon.embed.weight.use_muon)
        self.assertFalse(muon.embed.bias.use_muon)

        bad = TinyModel()
        with self.assertRaises(ValueError) as caught:
            fully_shard(bad, optimizer=self._optimizer(bad, {}))
        self.assertIn("misses trainable params", str(caught.exception))

    def test_wrap_model_fsdp_pipeline_branch(self):
        model = TinyModel()
        trainer = self._trainer(
            model, self._optimizer(model), amp_master_grad=True
        )
        trainer.using_fleet_model = True
        with (
            patch.object(fleet, "distributed_model", side_effect=lambda m: m),
            patch(
                "paddlefleet.trainer.trainer.fully_shard",
                side_effect=lambda m, **kwargs: m,
            ) as shard,
        ):
            trainer._wrap_model(model, training=True)

        self.assertIsInstance(
            trainer.optimizer, mix_precision_utils.MixPrecisionOptimizer
        )
        shard.assert_called_once_with(
            model,
            mp_policy=None,
            optimizer=trainer.optimizer,
            grad_accum_steps=2,
            prefetch_units=1,
            comm_unit_size=None,
        )

    def test_wrap_model_fsdp_prefetch_args(self):
        # fsdp_* TrainingArguments must reach fully_shard unchanged
        cases = [
            ({}, 1, None),
            ({"fsdp_prefetch_units": 2}, 2, None),
            ({"fsdp_prefetch_units": 0}, 0, None),
            ({"fsdp_comm_unit_size": 1 << 20}, 1, 1 << 20),
            ({"fsdp_comm_unit_size": 0}, 1, 0),
        ]
        for overrides, units, size in cases:
            with self.subTest(**overrides):
                model = TinyModel()
                trainer = self._trainer(
                    model, self._optimizer(model), **overrides
                )
                trainer.using_fleet_model = True
                with (
                    patch.object(
                        fleet, "distributed_model", side_effect=lambda m: m
                    ),
                    patch(
                        "paddlefleet.trainer.trainer.fully_shard",
                        side_effect=lambda m, **kwargs: m,
                    ) as shard,
                ):
                    trainer._wrap_model(model, training=True)

                kwargs = shard.call_args.kwargs
                self.assertEqual(kwargs["prefetch_units"], units)
                self.assertEqual(kwargs["comm_unit_size"], size)

    def test_wrap_model_sharding_stage1_without_pipeline(self):
        model = TinyModel()
        optimizer = self._optimizer(model)
        trainer = self._trainer(
            model,
            optimizer,
            sharding=[ShardingOption.SHARD_OP],
            tensor_model_parallel_size=2,
            sequence_parallel=False,
            split_param=False,
            stage1_broadcast_overlap=False,
        )
        with (
            patch.object(
                fleet, "distributed_model", side_effect=lambda m: m
            ) as dist_model,
            patch.object(
                trainer, "_wrap_distributed_optimizer", side_effect=lambda o: o
            ) as wrap_opt,
        ):
            trainer._wrap_model(model, training=True)

        dist_model.assert_called_once_with(model)
        wrap_opt.assert_called_once_with(optimizer)

    def test_save_model_guard(self):
        context = MagicMock()
        self._run_save_model(context)
        context.all_gather_params.assert_called_once_with()

        log = self._run_save_model(None)
        self.assertIn("skip param all_gather", log.warning.call_args[0][0])


if __name__ == "__main__":
    unittest.main()
