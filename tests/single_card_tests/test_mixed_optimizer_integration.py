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
"""Single-card checks of the trainer-side hooks for the mixed optimizer:
state names taken from ``optimizer_state_layout()`` (init_optimizer, ZCC
unified names) and recognition of ``MixedShardingOptimizer`` (reshard,
shell unwrapping, HF bit-exact clip).

The real shell needs a sharding group, so the optimizers here are fakes that
only carry the attributes the code under test reads.

Run:  python -m pytest tests/single_card_tests/test_mixed_optimizer_integration.py -v
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import paddle

from paddlefleet.trainer import trainer_utils
from paddlefleet.trainer.utils import zero_cost_checkpoint as zcc
from paddlefleet.trainer.utils.reshard import common as reshard_common

_LAYOUT = {
    "vector": ["velocity_0", "square_avg_0"],
    "scalar": ["step_count_0"],
    "master": "fp32_master_0",
}


class _Inner:
    def optimizer_state_layout(self):
        return dict(_LAYOUT)


def _sharded(mapping):
    """{struct_name: static_name} -> a model sharded_state_dict stand-in."""
    return {
        k: SimpleNamespace(local_tensor=SimpleNamespace(name=v))
        for k, v in mapping.items()
    }


class TestInitOptimizerLayout(unittest.TestCase):
    def test_state_names_follow_layout(self):
        created = []
        opt = SimpleNamespace(
            _inner_opt=_Inner(),
            _parameter_list=[
                SimpleNamespace(name="w_a"),
                SimpleNamespace(name="w_b"),
                SimpleNamespace(name="w_c"),
            ],
            _create_accumulators=lambda block, params: created.extend(
                p.name for p in params
            ),
        )
        meta = _sharded(
            {"a.weight": "w_a", "b.weight": "w_b", "c.weight": "w_c"}
        )
        # Only the layout's custom suffixes are in the checkpoint: a param is
        # initialised iff one of ITS layout states is present.
        metadata = {"a.weight.square_avg_0", "b.weight.step_count_0"}
        with mock.patch.object(
            trainer_utils, "_get_fsdp_context", return_value=None
        ):
            trainer_utils.init_optimizer(opt, meta, metadata)
        self.assertEqual(created, ["w_a", "w_b"])


class TestShellRecognition(unittest.TestCase):
    def test_unwrap_by_name(self):
        class MixedShardingOptimizer:  # matched by class name
            pass

        shell = MixedShardingOptimizer()
        wrapper = SimpleNamespace(_inner_opt=SimpleNamespace(_inner_opt=shell))
        self.assertIs(
            trainer_utils._unwrap_muon_sharding_optimizer(wrapper), shell
        )
        self.assertTrue(trainer_utils._is_muon_sharding_optimizer(wrapper))
        self.assertIsNone(
            trainer_utils._unwrap_muon_sharding_optimizer(SimpleNamespace())
        )

    def test_reshard_is_sharding_opt(self):
        class _FakeShell:
            pass

        wrapper = SimpleNamespace(_inner_opt=_FakeShell())
        with mock.patch.object(reshard_common, "MixedShardingOptimizer", None):
            self.assertFalse(reshard_common.is_sharding_opt(wrapper))
        with mock.patch.object(
            reshard_common, "MixedShardingOptimizer", _FakeShell
        ):
            self.assertTrue(reshard_common.is_sharding_opt(wrapper))

    def test_hf_bitexact_unwraps_mixed_shell(self):
        try:
            from paddle.distributed.fleet.meta_optimizers.mixed_sharding_optimizer import (
                MixedShardingOptimizer,
            )
        except (ImportError, ModuleNotFoundError):
            self.skipTest("Paddle build without MixedShardingOptimizer")
        from paddlefleet.utils import hf_bitexact_hybrid_clip

        self.assertIn(
            MixedShardingOptimizer, hf_bitexact_hybrid_clip._WRAPPER_OPTIMIZERS
        )


class TestZCCUnifiedName(unittest.TestCase):
    def test_layout_suffixes(self):
        t = paddle.zeros([1])
        opt = SimpleNamespace(
            _comm_buffer_list=[],
            _inner_opt=_Inner(),
            state_dict=lambda: {
                "w_a_velocity_0": t,
                "w_a_square_avg_0": t,
                "w_a_step_count_0": t,
                "w_b_moment1_0": t,  # built-in names stay accepted
                "master_weights": {"w_a": t},
                "LR_Scheduler": {},
            },
        )
        meta = _sharded({"a.weight": "w_a", "b.weight": "w_b"})
        with mock.patch.object(zcc, "_get_fsdp_context", return_value=None):
            names, slices = (
                zcc.ZeroCostCheckpointCallbackFcBased._gen_unified_name(
                    SimpleNamespace(), opt, meta
                )
            )
        self.assertEqual(
            names,
            {
                "w_a_velocity_0": "a.weight.velocity_0",
                "w_a_square_avg_0": "a.weight.square_avg_0",
                "w_a_step_count_0": "a.weight.step_count_0",
                "w_b_moment1_0": "b.weight.moment1_0",
                "w_a": "a.weight.w_0",
            },
        )
        self.assertEqual(slices, {})


if __name__ == "__main__":
    unittest.main()
