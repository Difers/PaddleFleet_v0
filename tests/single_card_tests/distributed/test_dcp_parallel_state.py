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

"""Single-card behavior tests for the dynamic context parallel (DCP) group
management added to ``paddlefleet.parallel_state``.

Covered production code:
  * ``initialize_dynamic_context_parallel_groups`` -- the power-of-two group
    table, the ``max_dynamic_context_parallel_size`` auto-detection and its
    guards, the ``size == 1 -> None`` / ``size == cp_degree -> reuse the static
    group`` special cases, the pool-layout self-check, the mp/pp-unsupported
    asserts, the no-``_cp_topo`` (CP-less hcg) early return and the per-size
    communicator warm-up.
  * ``get_context_parallel_group`` / ``get_context_parallel_world_size`` now
    preferring the per-microbatch dynamic group.
  * ``get_global_context_parallel_group`` / ``..._world_size`` -- the static
    answer that must NOT follow the microbatch.
  * ``set_current_context_parallel_size`` / ``get_...`` / ``reset_...`` and the
    ``get_dynamic_context_parallel_group(size)`` / ``..._sizes()`` getters.
  * ``initialize_model_parallel``'s ``enable_dynamic_context_parallel`` wiring.
  * ``paddlefleet.training.initialize.initialize_fleet``'s DCP argument
    plumbing, including the ``-1 -> None`` ("use the whole rank pool")
    normalization.

Why no real process group: every rank-pool decision in this code is pure
integer arithmetic over ``hcg.merge_inner_comm_list(...)``, and the only
collectives involved (``new_group`` plus the warm-up ``all_gather`` /
``alltoall_single``) are *what is asserted on*, not what produces the expected
values. They are therefore replaced by recording stubs, and the expected rank
blocks are derived by hand from the documented layout
``idx = cp_sharding_idx * cp_degree + cp_idx`` (cp innermost). Real multi-rank
NCCL numerics belong in the multi-card suite.

Every mutated module global is snapshotted and restored, so the shared
``parallel_state`` state is never left polluted for other test files.
"""

import os
import sys
import unittest
from unittest import mock

# ``src`` layout: make the in-tree package importable when not pip-installed.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

try:
    import paddle

    from paddlefleet import parallel_state as ps

    _IMPORT_ERROR = None
except ImportError as exc:  # genuine missing dependency only
    ps = None
    _IMPORT_ERROR = exc

SKIP_REASON = f"paddle/paddlefleet unavailable: {_IMPORT_ERROR}"


class _Group:
    """Stand-in for a paddle communication group handle.

    Only the attributes ``parallel_state`` reads are provided; instances are
    distinct objects so tests can assert identity and catch handle swaps.
    """

    def __init__(self, ranks, rank=0):
        self.ranks = list(ranks)
        self.nranks = len(self.ranks)
        self.rank = rank
        self.world_size = self.nranks

    def __repr__(self):
        return f"_Group(ranks={self.ranks})"


class _Hcg:
    """Fake hybrid-communicate-group exposing only what DCP init reads.

    The rank pools follow fleet's own layout (``split_context_comm_list``): one list of
    ``sharding_degree`` ranks per sharding group, cp innermost, so the static CP
    group of a rank is the contiguous ``cp_degree``-block that contains it.
    """

    def __init__(
        self,
        cp_degree,
        cp_sharding_degree,
        global_rank=0,
        n_pools=1,
        mp_degree=1,
        pp_degree=1,
        with_cp_topo=True,
        static_cp_group_override=None,
    ):
        pool_size = cp_degree * cp_sharding_degree
        self._pools = [
            list(range(i * pool_size, (i + 1) * pool_size))
            for i in range(n_pools)
        ]
        self._cp_degree = cp_degree
        self._cp_sharding_degree = cp_sharding_degree
        self._mp_degree = mp_degree
        self._pp_degree = pp_degree
        self.global_rank = global_rank
        self._dense_topo = object()
        if with_cp_topo:
            self._cp_topo = object()

        # Static CP group: the cp_degree-wide block holding ``global_rank``.
        pool = next(p for p in self._pools if global_rank in p)
        idx = pool.index(global_rank)
        start = (idx // cp_degree) * cp_degree
        self._cp_group = (
            static_cp_group_override
            if static_cp_group_override is not None
            else pool[start : start + cp_degree]
        )
        self._cp_comm_group = _Group(self._cp_group, rank=idx % cp_degree)

        # Read by initialize_model_parallel (plain attribute copies).
        self._mp_comm_group = _Group([global_rank])
        self._mp_group = [global_rank]
        self._pp_comm_group = _Group([global_rank])
        self._sharding_comm_group = _Group(pool)
        self._cp_sharding_comm_group = _Group(pool)

    def merge_inner_comm_list(self, topo, outer, inner):
        # DCP must ask for exactly fleet's own pool construction.
        assert topo is self._dense_topo
        assert (outer, inner) == ("moe_sharding", "dense_sharding")
        return [list(p) for p in self._pools]


@unittest.skipUnless(_IMPORT_ERROR is None, SKIP_REASON)
class _DcpTestBase(unittest.TestCase):
    """Snapshots every DCP/CP global and restores it, whatever the test does."""

    _GLOBALS = (
        "_DYNAMIC_CONTEXT_PARALLEL_GROUPS",
        "_CURRENT_CONTEXT_PARALLEL_SIZE",
        "_MAX_DYNAMIC_CONTEXT_PARALLEL_SIZE",
        "_CONTEXT_PARALLEL_GROUP",
        "_DATA_PARALLEL_GROUP_WITH_CP",
        "_TENSOR_MODEL_PARALLEL_GROUP",
        "_TENSOR_MODEL_PARALLEL_GLOBAL_RANKS",
        "_PIPELINE_MODEL_PARALLEL_GROUP",
        "_DATA_PARALLEL_GROUP",
        "_EXPERT_MODEL_PARALLEL_GROUP",
        "_EXPERT_DATA_PARALLEL_GROUP",
        "_GLOBAL_MEMORY_BUFFER",
    )

    def setUp(self):
        self._saved = {name: getattr(ps, name) for name in self._GLOBALS}
        self.addCleanup(self._restore)

    def _restore(self):
        for name, value in self._saved.items():
            setattr(ps, name, value)

    def build(self, hcg, max_size=None):
        """Run the DCP group build with every collective replaced by a stub.

        Returns ``(table, calls)``, where ``calls`` records what was asked of
        the communication layer so the test can assert on the rank blocks and
        on the warm-up instead of on NCCL side effects.
        """
        calls = {"new_group": [], "all_gather": [], "alltoall": [], "sync": 0}

        def _new_group(ranks):
            calls["new_group"].append(list(ranks))
            rank = (
                ranks.index(hcg.global_rank) if hcg.global_rank in ranks else 0
            )
            return _Group(ranks, rank=rank)

        def _all_gather(out, tensor, group=None):
            calls["all_gather"].append(group)
            out.append(tensor)

        def _alltoall(out, inp, send_splits, recv_splits, group=None):
            calls["alltoall"].append(
                (group, list(send_splits), list(recv_splits))
            )

        def _sync():
            calls["sync"] += 1

        with (
            mock.patch.object(
                paddle.distributed, "new_group", side_effect=_new_group
            ),
            mock.patch.object(
                paddle.distributed, "all_gather", side_effect=_all_gather
            ),
            mock.patch.object(
                paddle.distributed, "alltoall_single", side_effect=_alltoall
            ),
            mock.patch.object(
                paddle.device.cuda, "synchronize", side_effect=_sync
            ),
        ):
            table = ps.initialize_dynamic_context_parallel_groups(hcg, max_size)
        return table, calls


class TestDcpGroupTable(_DcpTestBase):
    """The pre-built ``{size: group}`` table and how it is carved out."""

    def test_auto_max_size_and_group_table(self):
        # pool = sharding_degree = 2 * 4 = 8 ranks [0..7]; this rank is 5, so
        # its static cp=2 block is [4, 5] (cp innermost).
        hcg = _Hcg(cp_degree=2, cp_sharding_degree=4, global_rank=5)
        table, calls = self.build(hcg)

        # Hand-derived: largest power of two that divides the 8-rank pool.
        self.assertEqual(ps.get_max_dynamic_context_parallel_size(), 8)
        self.assertEqual(ps.get_dynamic_context_parallel_sizes(), [1, 2, 4, 8])

        # size 1 needs no communicator at all.
        self.assertIsNone(table[1])
        # size == cp_degree reuses fleet's static group object (identity!).
        self.assertIs(table[2], hcg._cp_comm_group)
        # The bigger groups are the size-aligned blocks holding rank 5.
        self.assertEqual(table[4].ranks, [4, 5, 6, 7])
        self.assertEqual(table[8].ranks, [0, 1, 2, 3, 4, 5, 6, 7])

        # Only sizes that are neither 1 nor cp_degree cost a new communicator,
        # and every rank must walk the same (size, pool, block) order.
        self.assertEqual(
            calls["new_group"],
            [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]],
        )

    def test_explicit_cap_keeps_groups_inside_static_cp(self):
        # max == cp_degree is the documented "never leave the static CP group"
        # setting: no extra NCCL group is created at all.
        hcg = _Hcg(cp_degree=2, cp_sharding_degree=4, global_rank=3)
        table, calls = self.build(hcg, max_size=2)
        self.assertEqual(sorted(table), [1, 2])
        self.assertEqual(calls["new_group"], [])
        self.assertIs(table[2], hcg._cp_comm_group)
        self.assertEqual(ps.get_max_dynamic_context_parallel_size(), 2)

    def test_second_pool_ranks_are_not_mixed_in(self):
        # Two sharding groups -> two disjoint pools; the groups of rank 9 must
        # stay inside its own pool (ranks 8..15), never straddle both.
        hcg = _Hcg(cp_degree=4, cp_sharding_degree=2, global_rank=9, n_pools=2)
        table, _ = self.build(hcg)
        self.assertEqual(ps.get_dynamic_context_parallel_sizes(), [1, 2, 4, 8])
        self.assertEqual(table[2].ranks, [8, 9])
        self.assertIs(table[4], hcg._cp_comm_group)
        self.assertEqual(table[4].ranks, [8, 9, 10, 11])
        self.assertEqual(table[8].ranks, [8, 9, 10, 11, 12, 13, 14, 15])

    def test_non_power_of_two_pool_stops_at_largest_divisor(self):
        # pool = 3 * 2 = 6: only 1 and 2 tile it, so the auto cap is 2 and
        # cp_degree (3) is never materialized as a dynamic size.
        hcg = _Hcg(cp_degree=3, cp_sharding_degree=2, global_rank=4)
        table, calls = self.build(hcg)
        self.assertEqual(ps.get_dynamic_context_parallel_sizes(), [1, 2])
        self.assertEqual(ps.get_max_dynamic_context_parallel_size(), 2)
        self.assertEqual(table[2].ranks, [4, 5])
        self.assertEqual(calls["new_group"], [[0, 1], [2, 3], [4, 5]])


class TestDcpGuards(_DcpTestBase):
    """Every refusal is deliberate: a wrong group table hangs or corrupts a run."""

    def test_hcg_without_context_axis_disables_dcp(self):
        # A plain HybridCommunicateGroup has no `context` axis -> a single
        # size-1 group, and no collective is attempted.
        hcg = _Hcg(cp_degree=1, cp_sharding_degree=4, with_cp_topo=False)
        table, calls = self.build(hcg)
        self.assertEqual(table, {1: None})
        self.assertEqual(ps.get_max_dynamic_context_parallel_size(), 1)
        self.assertEqual(calls["new_group"], [])
        self.assertEqual(calls["all_gather"], [])

    def test_tensor_model_parallel_is_rejected(self):
        hcg = _Hcg(cp_degree=2, cp_sharding_degree=2, mp_degree=2)
        with self.assertRaises(NotImplementedError) as cm:
            self.build(hcg)
        self.assertIn("tensor model parallel", str(cm.exception))

    def test_pipeline_parallel_is_rejected(self):
        hcg = _Hcg(cp_degree=2, cp_sharding_degree=2, pp_degree=2)
        with self.assertRaises(NotImplementedError) as cm:
            self.build(hcg)
        self.assertIn("pipeline parallel", str(cm.exception))

    def test_cap_larger_than_pool_is_rejected(self):
        hcg = _Hcg(cp_degree=2, cp_sharding_degree=2)  # pool = 4
        with self.assertRaises(ValueError) as cm:
            self.build(hcg, max_size=8)
        self.assertIn("must be in [1, 4]", str(cm.exception))

    def test_cap_not_dividing_pool_is_rejected(self):
        hcg = _Hcg(cp_degree=2, cp_sharding_degree=4)  # pool = 8
        with self.assertRaises(ValueError) as cm:
            self.build(hcg, max_size=3)
        self.assertIn("must divide the rank pool", str(cm.exception))

    def test_cap_not_power_of_two_is_rejected(self):
        # pool = 6 so 6 does divide it, but CP groups are power-of-two blocks.
        hcg = _Hcg(cp_degree=3, cp_sharding_degree=2)
        with self.assertRaises(ValueError) as cm:
            self.build(hcg, max_size=6)
        self.assertIn("must be a power of two", str(cm.exception))

    def test_pool_layout_self_check_catches_cp_innermost_violation(self):
        # If the pool were not cp-innermost, our carving would not reproduce
        # the group fleet actually built -- that must fail loudly instead of
        # silently building groups out of ranks that are not CP peers.
        hcg = _Hcg(
            cp_degree=2,
            cp_sharding_degree=2,
            global_rank=1,
            static_cp_group_override=[1, 3],  # strided, not a contiguous block
        )
        with self.assertRaises(AssertionError) as cm:
            self.build(hcg)
        self.assertIn("static cp group is", str(cm.exception))


class TestDcpWarmup(_DcpTestBase):
    """Both ops DCP really runs are warmed up, on every non-trivial size."""

    def test_warmup_exercises_all_gather_and_alltoall_per_size(self):
        hcg = _Hcg(cp_degree=2, cp_sharding_degree=4, global_rank=5)
        table, calls = self.build(hcg)

        warmed = [table[s] for s in (2, 4, 8)]
        # size 1 maps to None (no communicator), so it must be skipped.
        self.assertEqual(calls["all_gather"], warmed)
        self.assertEqual(
            calls["alltoall"],
            [(g, [1] * g.nranks, [1] * g.nranks) for g in warmed],
        )
        self.assertEqual(calls["sync"], 1)


class TestCurrentGroupSelection(_DcpTestBase):
    """Which group a CP collective sees, and which one must never move."""

    def setUp(self):
        super().setUp()
        self.static = _Group([4, 5], rank=1)
        self.g2 = _Group([4, 5], rank=1)
        self.g4 = _Group([4, 5, 6, 7], rank=1)
        ps._CONTEXT_PARALLEL_GROUP = self.static
        ps._DYNAMIC_CONTEXT_PARALLEL_GROUPS = {1: None, 2: self.g2, 4: self.g4}
        ps._MAX_DYNAMIC_CONTEXT_PARALLEL_SIZE = 4
        ps._CURRENT_CONTEXT_PARALLEL_SIZE = None

    def test_static_group_is_used_until_a_size_is_selected(self):
        self.assertIsNone(ps.get_current_context_parallel_size())
        self.assertIs(ps.get_context_parallel_group(), self.static)
        self.assertEqual(ps.get_context_parallel_world_size(), 2)

    def test_selected_size_redirects_every_cp_getter(self):
        ps.set_current_context_parallel_size(4)
        self.assertEqual(ps.get_current_context_parallel_size(), 4)
        self.assertIs(ps.get_context_parallel_group(), self.g4)
        self.assertEqual(ps.get_context_parallel_world_size(), 4)
        self.assertEqual(ps.get_context_parallel_rank(), 1)

    def test_size_one_means_no_communicator(self):
        ps.set_current_context_parallel_size(1)
        self.assertIsNone(ps.get_context_parallel_group())
        self.assertEqual(ps.get_context_parallel_world_size(), 1)
        self.assertEqual(ps.get_context_parallel_rank(), 0)

    def test_reset_falls_back_to_the_static_group(self):
        ps.set_current_context_parallel_size(4)
        ps.reset_current_context_parallel_size()
        self.assertIsNone(ps.get_current_context_parallel_size())
        self.assertIs(ps.get_context_parallel_group(), self.static)

    def test_selecting_a_size_with_no_pre_built_group_is_rejected(self):
        # NCCL groups cannot be created mid-step, so an unknown size is a bug
        # in the scheduler, not something to build on demand.
        with self.assertRaises(AssertionError) as cm:
            ps.set_current_context_parallel_size(8)
        self.assertIn("no pre-built CP group of size 8", str(cm.exception))
        self.assertIn("[1, 2, 4]", str(cm.exception))
        # A rejected switch must not have moved the current size.
        self.assertIsNone(ps.get_current_context_parallel_size())

    def test_group_lookup_by_size(self):
        self.assertIs(ps.get_dynamic_context_parallel_group(2), self.g2)
        self.assertIsNone(ps.get_dynamic_context_parallel_group(1))
        with self.assertRaises(AssertionError):
            ps.get_dynamic_context_parallel_group(3)

    def test_global_getters_do_not_follow_the_microbatch(self):
        # This is what dsa_attention's DP-average group selection relies on:
        # a per-microbatch CP size must not change a whole-run decision.
        ps.set_current_context_parallel_size(4)
        self.assertIs(ps.get_global_context_parallel_group(), self.static)
        self.assertEqual(ps.get_global_context_parallel_world_size(), 2)

    def test_global_world_size_is_one_without_a_static_group(self):
        ps._CONTEXT_PARALLEL_GROUP = None
        self.assertIsNone(ps.get_global_context_parallel_group())
        self.assertEqual(ps.get_global_context_parallel_world_size(), 1)
        with self.assertRaises(AssertionError):
            ps.get_global_context_parallel_group(check_initialized=True)


class TestInitializeModelParallelWiring(_DcpTestBase):
    """``initialize_model_parallel`` only builds DCP groups when asked to."""

    def test_disabled_by_default(self):
        hcg = _Hcg(cp_degree=2, cp_sharding_degree=2)
        with mock.patch.object(
            ps, "initialize_dynamic_context_parallel_groups"
        ) as built:
            ps.initialize_model_parallel(hcg)
        built.assert_not_called()
        # The static CP handles are still wired up as before.
        self.assertIs(
            ps.get_global_context_parallel_group(), hcg._cp_comm_group
        )

    def test_flag_and_cap_are_forwarded(self):
        hcg = _Hcg(cp_degree=2, cp_sharding_degree=2)
        with mock.patch.object(
            ps, "initialize_dynamic_context_parallel_groups"
        ) as built:
            ps.initialize_model_parallel(
                hcg,
                enable_dynamic_context_parallel=True,
                max_dynamic_context_parallel_size=2,
            )
        built.assert_called_once_with(hcg, 2)


@unittest.skipUnless(_IMPORT_ERROR is None, SKIP_REASON)
class TestInitializeFleetDcpPlumbing(unittest.TestCase):
    """``initialize_fleet`` is where PaddleFormers-style args reach DCP.

    ``fleet.init`` / ``dist`` / the logging setup are collaborators, not the
    subject: they are stubbed so the test can assert exactly what is handed to
    ``parallel_state.initialize_model_parallel``.
    """

    def _call(self, parsed_args, **kwargs):
        from types import SimpleNamespace

        from paddlefleet.training import initialize as init_mod

        with (
            mock.patch.object(init_mod, "fleet") as fake_fleet,
            mock.patch.object(init_mod, "dist"),
            mock.patch.object(init_mod, "set_global_variables"),
            mock.patch.object(init_mod, "set_logging"),
            mock.patch.object(init_mod.ps, "initialize_model_parallel") as imp,
        ):
            fake_fleet.get_hybrid_communicate_group.return_value = "HCG"
            init_mod.initialize_fleet(
                SimpleNamespace(), parsed_args=parsed_args, **kwargs
            )
        self.assertEqual(imp.call_count, 1)
        return imp.call_args

    def test_explicit_arguments_win_over_parsed_args(self):
        from types import SimpleNamespace

        args = SimpleNamespace(
            enable_dynamic_context_parallel=False, max_dynamic_cp_size=2
        )
        call = self._call(
            args, enable_dynamic_context_parallel=True, max_dynamic_cp_size=4
        )
        self.assertEqual(call.args, ("HCG",))
        self.assertEqual(
            call.kwargs,
            {
                "enable_dynamic_context_parallel": True,
                "max_dynamic_context_parallel_size": 4,
            },
        )

    def test_minus_one_cap_means_the_whole_rank_pool(self):
        from types import SimpleNamespace

        # -1 is the "unset" value on the PaddleFormers side; downstream that is
        # spelled None ("use the whole pool"), not -1.
        call = self._call(
            SimpleNamespace(),
            enable_dynamic_context_parallel=True,
            max_dynamic_cp_size=-1,
        )
        self.assertIs(call.kwargs["max_dynamic_context_parallel_size"], None)
        self.assertIs(call.kwargs["enable_dynamic_context_parallel"], True)

    def test_values_are_read_from_parsed_args_when_not_passed(self):
        from types import SimpleNamespace

        args = SimpleNamespace(
            enable_dynamic_context_parallel=True, max_dynamic_cp_size=2
        )
        call = self._call(args)
        self.assertEqual(
            call.kwargs,
            {
                "enable_dynamic_context_parallel": True,
                "max_dynamic_context_parallel_size": 2,
            },
        )

    def test_off_by_default_when_nothing_mentions_dcp(self):
        from types import SimpleNamespace

        call = self._call(SimpleNamespace())
        self.assertEqual(
            call.kwargs,
            {
                "enable_dynamic_context_parallel": False,
                "max_dynamic_context_parallel_size": None,
            },
        )

    def test_explicit_false_overrides_an_enabled_config(self):
        from types import SimpleNamespace

        args = SimpleNamespace(
            enable_dynamic_context_parallel=True, max_dynamic_cp_size=2
        )
        call = self._call(args, enable_dynamic_context_parallel=False)
        self.assertIs(call.kwargs["enable_dynamic_context_parallel"], False)
        # The cap still comes from the config; it is simply unused when off.
        self.assertEqual(call.kwargs["max_dynamic_context_parallel_size"], 2)


@unittest.skipUnless(_IMPORT_ERROR is None, SKIP_REASON)
class TestTrainingArgumentsDcpDefaults(unittest.TestCase):
    """The two new ``TrainingArguments`` knobs must default to "nothing changes".

    Read off the dataclass fields instead of constructing TrainingArguments,
    which would require a full output_dir/strategy setup.
    """

    def test_defaults_keep_dcp_off_and_the_cap_unset(self):
        import dataclasses

        from paddlefleet.trainer.training_args import TrainingArguments

        fields = {f.name: f for f in dataclasses.fields(TrainingArguments)}
        self.assertIs(fields["enable_dynamic_context_parallel"].default, False)
        # -1 means "the whole rank pool"; initialize_fleet turns it into None.
        self.assertEqual(fields["max_dynamic_cp_size"].default, -1)


@unittest.skipUnless(_IMPORT_ERROR is None, SKIP_REASON)
class TestTrainerRejectsDcpWithoutDriver(unittest.TestCase):
    """The built-in Trainer never selects a CP group per microbatch, so it
    must refuse ``enable_dynamic_context_parallel`` instead of silently
    running every microbatch on the static group."""

    def _init(self, trainer_cls, enabled):
        from types import SimpleNamespace

        args = SimpleNamespace(enable_dynamic_context_parallel=enabled)
        trainer_cls.__init__(trainer_cls.__new__(trainer_cls), args=args)

    def test_builtin_trainer_raises_when_dcp_is_enabled(self):
        from paddlefleet.trainer.trainer import Trainer

        with self.assertRaisesRegex(
            NotImplementedError, "supports_dynamic_context_parallel"
        ):
            self._init(Trainer, enabled=True)

    def test_opted_in_subclass_passes_the_guard(self):
        from paddlefleet.trainer.trainer import Trainer

        class DcpTrainer(Trainer):
            supports_dynamic_context_parallel = True

        # The stub args are not a full TrainingArguments, so __init__ fails
        # further down; it just must not be the DCP guard.
        try:
            self._init(DcpTrainer, enabled=True)
        except NotImplementedError as exc:
            self.assertNotIn("supports_dynamic_context_parallel", str(exc))
        except Exception:
            pass


if __name__ == "__main__":
    unittest.main()
