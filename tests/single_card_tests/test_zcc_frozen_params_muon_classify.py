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

"""ZCC model_state filtering of frozen params under the Muon/Mixed shells.

Frozen params never enter the optimizer, so they used to reach the last
branch of the Muon state-dict filters and were kept by sharding rank 0 only.
An expert sharded over an EP group inside the sharding axis was then lost on
every other rank. They are now classified with the shell's own rules: color
group, whole tensor or element-wise, and the greedy 2D owner partition.

Every case runs under MuonShardingOptimizer and MixedShardingOptimizer. The
shell is a real instance with its constructor skipped, so the color-group and
partition code is Paddle's; under Mixed the per-param info comes from a real
MixedInnerOptimizer's `_param_info_map`. The shell is reached through a
HybridParallelOptimizer-style wrapper, as in training. The sharding group has
two ranks; each test plays one of them.
"""

import os
import pickle
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import paddle
from paddle.distributed.fleet.meta_optimizers.mixed_sharding_optimizer import (
    MixedShardingOptimizer,
)
from paddle.distributed.fleet.meta_optimizers.muon_sharding_optimizer import (
    MuonShardingOptimizer,
)
from paddle.optimizer.inner_optimizer import ParamInfo
from paddle.optimizer.mixed_inner_optimizer import MixedInnerOptimizer

from paddlefleet.trainer.utils.zero_cost_checkpoint import (
    ZeroCostCheckpointCallback,
    ZeroCostCheckpointCallbackFcBased,
    _muon_classify_untracked_params,
    get_fused_param_mappings,
)

WORLD = 2
MUON_2D = ("train_w", "frozen_expert", "frozen_a", "frozen_b")
FROZEN = ("frozen_expert", "frozen_a", "frozen_b", "frozen_norm")


class _Model(paddle.nn.Layer):
    """One trainable 2D param, four frozen params and a persistable buffer."""

    def __init__(self, rank):
        super().__init__()
        for name in MUON_2D:
            setattr(self, name, self.create_parameter([4, 4]))
        self.frozen_norm = self.create_parameter([4])
        for name in FROZEN:
            getattr(self, name).stop_gradient = True
        # moe_sharding_degree == 1: the expert's comm group is this rank alone.
        expert_group = SimpleNamespace(ranks=[rank], rank=0, nranks=1)
        self.frozen_expert.color = {
            "color": "moe_expert",
            "group": expert_group,
        }
        self.register_buffer("bias_buf", paddle.zeros([4]), persistable=True)

    def param(self, attr):
        return getattr(self, attr)

    def sharded_state_dict(self):
        return {
            k: SimpleNamespace(local_tensor=v)
            for k, v in self.state_dict().items()
        }


class _Wrapper:
    """Stands in for HybridParallelOptimizer: forwards attribute reads."""

    def __init__(self, inner):
        self._inner_opt = inner

    def __getattr__(self, item):
        return getattr(self._inner_opt, item)


def _param_info_map(model):
    return {
        p.name: SimpleNamespace(use_muon=attr in MUON_2D, use_hyperball=False)
        for attr, p in model.named_parameters()
    }


def _info_map_of(shell):
    """The per-param info the classifier reads for this shell."""
    if isinstance(shell, MixedShardingOptimizer):
        return shell._inner_opt._param_info_map
    return shell._muon_param_info_map


def _shell(model, rank, shell_cls=MuonShardingOptimizer):
    """The shell as the sharding optimizer leaves it with a frozen backbone."""
    shell = object.__new__(shell_cls)
    if shell_cls is MixedShardingOptimizer:
        # The full map built by the trainer, frozen params included; frozen
        # params have no sub-optimizer, so `keep_whole_of` cannot answer.
        inner = object.__new__(MixedInnerOptimizer)
        inner._multi_precision = False
        inner._param_info_map = _param_info_map(model)
        shell._inner_opt = inner
    else:
        shell._inner_opt = SimpleNamespace(_multi_precision=False)
        shell._muon_param_info_map = _param_info_map(model)
    group = SimpleNamespace(ranks=list(range(WORLD)), rank=rank, nranks=WORLD)
    shell._hcg = SimpleNamespace(get_sharding_parallel_group=lambda: group)
    shell._sharding_rank = rank
    # Tiny buffers so the greedy partition spreads params over every rank.
    shell.comm_buffer_size_MB = 1e-9
    train_w = model.param("train_w")
    shell._params_2d_by_color = {None: [train_w]}
    shell._local_2d = [train_w] if rank == 0 else []
    shell._params_1d = []
    return shell


def _as_trainable(shell, model, rank):
    """The shell MuonShardingOptimizer would build were nothing frozen."""
    by_color = {
        None: [model.param(a) for a in MUON_2D if a != "frozen_expert"],
        "moe_expert": [model.param("frozen_expert")],
    }
    shell._params_2d_by_color = by_color
    shell._local_2d = list(by_color["moe_expert"]) + list(
        shell._partition_2d_parameters(list(by_color[None]), WORLD)[rank]
    )
    shell._params_1d = [model.param("frozen_norm")]
    return shell


def _fc_keys(model, optimizer, replicate=False):
    callback = SimpleNamespace(
        args=SimpleNamespace(replicate_saved_into_local=replicate)
    )
    out = ZeroCostCheckpointCallbackFcBased._muon_manipulate_sharded_state_dict(
        callback, model, optimizer
    )
    return set(out)


def _legacy_state(model, optimizer):
    return ZeroCostCheckpointCallback._muon_manipulate_state_dict(
        None, model, optimizer
    )


def _legacy_keys(model, optimizer):
    return set(_legacy_state(model, optimizer))


_REOPEN = """
import pickle, sys
import numpy as np
import paddle
paddle.set_device("gpu")
metas = pickle.load(open(sys.argv[1], "rb"))
out = {
    k: np.array(paddle.to_tensor(paddle.base.core.DenseTensor._new_shared_cuda(m)))
    for k, m in metas.items()
}
pickle.dump(out, open(sys.argv[2], "wb"))
"""


_NO_PIDFD = (
    "pidfd_open is unavailable here; reopening a VMM tensor in another "
    "process needs it"
)


def _can_import_ipc_across_processes():
    """Cross-process VMM import calls pidfd_open, which some containers block."""
    try:
        fd = os.pidfd_open(os.getpid())
    except (AttributeError, OSError):
        return False
    os.close(fd)
    return True


def _reopen_in_child(ipc_metas):
    """Open CUDA IPC handles in another process, as the ZCC worker does."""
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = os.path.join(tmp, "in.pkl"), os.path.join(tmp, "out.pkl")
        with open(src, "wb") as f:
            pickle.dump(ipc_metas, f)
        subprocess.run(
            [sys.executable, "-c", _REOPEN, src, dst],
            check=True,
            stdin=subprocess.DEVNULL,
        )
        with open(dst, "rb") as f:
            return pickle.load(f)


class _FrozenParamsCases:
    """Cases shared by every shell; subclasses set SHELL."""

    SHELL = None

    def setUp(self):
        paddle.set_device("cpu")
        self.models = [_Model(rank) for rank in range(WORLD)]

    def _shell(self, rank):
        return _shell(self.models[rank], rank, self.SHELL)

    def _opt(self, rank):
        return _Wrapper(self._shell(rank))

    def test_frozen_expert_kept_on_every_rank(self):
        """Each rank is the sole owner of its own expert shard."""
        for rank in range(WORLD):
            for keys in (
                _fc_keys(self.models[rank], self._opt(rank)),
                _legacy_keys(self.models[rank], self._opt(rank)),
            ):
                self.assertIn("frozen_expert", keys, f"rank {rank}")

    def test_frozen_2d_saved_once_across_the_group(self):
        """Replicated frozen 2D params are split over owners, not all on rank 0."""
        owners = {}
        for rank in range(WORLD):
            for key in _fc_keys(self.models[rank], self._opt(rank)):
                if key in ("frozen_a", "frozen_b"):
                    owners.setdefault(key, []).append(rank)
        self.assertEqual(set(owners), {"frozen_a", "frozen_b"})
        for key, ranks in owners.items():
            self.assertEqual(len(ranks), 1, f"{key} saved by {ranks}")
        self.assertEqual(
            sorted(r for ranks in owners.values() for r in ranks), [0, 1]
        )

    def test_frozen_1d_saved_on_every_rank(self):
        for rank in range(WORLD):
            self.assertIn(
                "frozen_norm", _fc_keys(self.models[rank], self._opt(rank))
            )

    def test_buffer_stays_on_rank_zero(self):
        self.assertIn("bias_buf", _fc_keys(self.models[0], self._opt(0)))
        self.assertNotIn("bias_buf", _fc_keys(self.models[1], self._opt(1)))
        self.assertIn("bias_buf", _legacy_keys(self.models[0], self._opt(0)))
        self.assertNotIn("bias_buf", _legacy_keys(self.models[1], self._opt(1)))

    def test_replicate_keeps_every_frozen_2d_param(self):
        for rank in range(WORLD):
            keys = _fc_keys(self.models[rank], self._opt(rank), replicate=True)
            self.assertLessEqual(set(FROZEN), keys)

    def test_frozen_matches_trainable(self):
        """Freezing the whole model does not change which rank saves what."""
        for rank in range(WORLD):
            model = self.models[rank]
            frozen = self._shell(rank)
            frozen._params_2d_by_color = {}
            frozen._local_2d = []
            frozen = _Wrapper(frozen)
            trainable = _Wrapper(_as_trainable(self._shell(rank), model, rank))
            for replicate in (False, True):
                self.assertEqual(
                    _fc_keys(model, frozen, replicate),
                    _fc_keys(model, trainable, replicate),
                    f"rank {rank} replicate={replicate}",
                )
            self.assertEqual(
                _legacy_keys(model, frozen), _legacy_keys(model, trainable)
            )

    def test_trainable_shell_leaves_nothing_untracked(self):
        model = self.models[1]
        shell = _as_trainable(self._shell(1), model, 1)
        self.assertEqual(
            _muon_classify_untracked_params(model, shell), (set(), set(), set())
        )

    def test_optimizer_lists_untouched(self):
        """Classifying is read-only: frozen params never reach the optimizer."""
        shell = self._shell(1)
        before = (
            list(shell._local_2d),
            {k: list(v) for k, v in shell._params_2d_by_color.items()},
            list(shell._params_1d),
        )
        _fc_keys(self.models[1], _Wrapper(shell))
        after = (
            list(shell._local_2d),
            {k: list(v) for k, v in shell._params_2d_by_color.items()},
            list(shell._params_1d),
        )
        self.assertEqual(before, after)
        for attr in FROZEN:
            self.assertTrue(self.models[1].param(attr).stop_gradient)

    def test_missing_param_info_raises(self):
        shell = self._shell(0)
        del _info_map_of(shell)[self.models[0].param("frozen_a").name]
        with self.assertRaisesRegex(ValueError, "no param info"):
            _muon_classify_untracked_params(self.models[0], _Wrapper(shell))

    def _fused_mappings(self):
        """Real get_fused_param_mappings() over a frozen-backbone shell."""
        paddle.set_device("gpu")
        model = _Model(0)
        for _, p in model.named_parameters():
            p.set_value(paddle.rand(p.shape))
        model.bias_buf.set_value(paddle.rand(model.bias_buf.shape))
        shell = _shell(model, 0, self.SHELL)
        shell._comm_buffer_list = []
        shell._parameter_list = [model.param("train_w")]
        state = _legacy_state(model, _Wrapper(shell))
        holder = []
        mappings, ipc_metas = get_fused_param_mappings(
            _Wrapper(shell), state, staging_holder=holder
        )
        return state, mappings, ipc_metas, holder

    @unittest.skipUnless(paddle.is_compiled_with_cuda(), "needs CUDA IPC")
    def test_fused_mappings_share_frozen_params_per_tensor(self):
        """Only the trainable param is staged; frozen keys go to `unshard_`."""
        state, mappings, ipc_metas, holder = self._fused_mappings()

        self.assertEqual(set(mappings), set(state))
        staged = {k for k, m in mappings.items() if "start" in m}
        self.assertEqual(staged, {"train_w"})
        self.assertEqual(len(holder), 1)
        for key in set(state) - staged:
            self.assertEqual(mappings[key]["buffer_index"], f"unshard_{key}")
            self.assertIn(f"unshard_{key}", ipc_metas)
        self.assertLessEqual({"frozen_expert", "bias_buf"}, set(state))

    @unittest.skipUnless(paddle.is_compiled_with_cuda(), "needs CUDA IPC")
    @unittest.skipUnless(_can_import_ipc_across_processes(), _NO_PIDFD)
    def test_frozen_params_reopen_intact_in_another_process(self):
        """The `unshard_` handles reopen to the right values, as in the worker."""
        state, mappings, ipc_metas, _ = self._fused_mappings()
        shared = {
            k: ipc_metas[m["buffer_index"]]
            for k, m in mappings.items()
            if m["buffer_index"].startswith("unshard_")
        }
        reopened = _reopen_in_child(shared)
        for key, values in reopened.items():
            np.testing.assert_array_equal(values, state[key].numpy(), key)


class TestFrozenParamsMuonShell(_FrozenParamsCases, unittest.TestCase):
    SHELL = MuonShardingOptimizer


class TestFrozenParamsMixedShell(_FrozenParamsCases, unittest.TestCase):
    SHELL = MixedShardingOptimizer

    def test_reads_mixed_inner_param_info_map(self):
        """Mixed has no `_muon_param_info_map`; the inner map is the source."""
        shell = self._shell(1)
        self.assertNotIn("_muon_param_info_map", vars(shell))
        self.assertFalse(hasattr(shell._inner_opt, "_muon_param_info_map"))
        _, all_2d, all_1d = _muon_classify_untracked_params(
            self.models[1], _Wrapper(shell)
        )
        names = {attr: self.models[1].param(attr).name for attr in FROZEN}
        self.assertEqual(
            all_2d,
            {names[a] for a in ("frozen_expert", "frozen_a", "frozen_b")},
        )
        self.assertEqual(all_1d, {names["frozen_norm"]})

    def test_plain_param_info_is_element_wise(self):
        """A generic `ParamInfo` has no `use_muon`: saved by every rank."""
        for rank in range(WORLD):
            shell = self._shell(rank)
            model = self.models[rank]
            _info_map_of(shell)[model.param("frozen_a").name] = ParamInfo()
            local_2d, all_2d, all_1d = _muon_classify_untracked_params(
                model, _Wrapper(shell)
            )
            name = model.param("frozen_a").name
            self.assertIn(name, all_1d)
            self.assertNotIn(name, all_2d | local_2d)
            self.assertIn("frozen_a", _fc_keys(model, _Wrapper(shell)))


class TestOtherShells(unittest.TestCase):
    def test_other_shells_classify_nothing(self):
        """A shell other than Muon/Mixed keeps the old behaviour."""
        paddle.set_device("cpu")
        model = _Model(1)

        class OtherShardingOptimizer:
            def __getattr__(self, item):
                return getattr(self._inner_opt, item)

        shell = _shell(model, 1, shell_cls=OtherShardingOptimizer)
        self.assertEqual(
            _muon_classify_untracked_params(model, _Wrapper(shell)),
            (set(), set(), set()),
        )
        self.assertNotIn("frozen_expert", _fc_keys(model, _Wrapper(shell)))


if __name__ == "__main__":
    unittest.main()
