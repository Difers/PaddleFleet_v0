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

"""Behavior + numerical tests for the dynamic context parallel (DCP) plumbing
in ``paddlefleet.context_parallel_utils``.

Covered production code:
  * ``get_current_cp_group`` / ``get_current_cp_world_size`` -- the single place
    every CP primitive now resolves its default group through, including the
    deliberately non-fatal fallback when fleet was never initialized.
  * ``refresh_cp_state`` / ``refresh_context_parallel_size`` -- re-pointing the
    ``cp_group / cp_size / cp_rank / cp_enabled`` (resp. ``context_parallel_size``)
    fields a layer froze in ``__init__`` at the current microbatch's group, and
    the guarantee that they are a *no-op* while DCP is off, and the
    re-derivation from the static group once the selection is reset.
  * ``ProcessGroupCollection.cp`` -- following the DCP selection when it
    holds the static group, and keeping a layer's own group otherwise.
  * ``ContextParallelScatterOp`` / ``...GatherOp`` / ``...AllGatherOp`` /
    ``ContextParallelNormalScatter`` / ``ContextParallelNormalGather`` resolving
    their group per call instead of reading fleet's static one.

Numerics (the "precision" part), and why they are meaningful on one card:
  ``scatter_contiguous`` and ``scatter_balance`` are *purely local* slice
  selections -- they touch no collective, only ``group.nranks`` / ``group.rank``.
  That is exactly the code path the DCP bug lived in: a layer that kept the
  static ``cp_size`` sliced a cp=2 microbatch into cp=4 pieces, so the local
  slice width was off by a factor of two.
  So the tests below switch the selected CP size and compare the produced slice
  *element by element* against a hand-derived expectation (distinguishable
  ``arange`` content, exact equality -- no tolerance needed since these are
  copies, not arithmetic). Cross-rank gather/reduce numerics need a real process
  group and stay in the multi-card suite; nothing here fakes a collective to
  stand in for them.

Every mutated module global is snapshotted and restored.
"""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

# ``src`` layout: make the in-tree package importable when not pip-installed.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

try:
    import numpy as np
    import paddle

    from paddlefleet import context_parallel_utils as cpu, parallel_state as ps
    from paddlefleet.process_groups_config import ProcessGroupCollection

    _IMPORT_ERROR = None
except ImportError as exc:  # genuine missing dependency only
    cpu = ps = None
    _IMPORT_ERROR = exc

SKIP_REASON = f"paddle/paddlefleet unavailable: {_IMPORT_ERROR}"


class _Group:
    """Stand-in for a communication group: only topology integers are read by
    the local code paths under test."""

    def __init__(self, ranks, rank=0):
        self.ranks = list(ranks)
        self.nranks = len(self.ranks)
        self.rank = rank
        self.world_size = self.nranks


@unittest.skipUnless(_IMPORT_ERROR is None, SKIP_REASON)
class _DcpUtilsTestBase(unittest.TestCase):
    """Restores the DCP globals and the probe budget after every test."""

    _GLOBALS = (
        "_DYNAMIC_CONTEXT_PARALLEL_GROUPS",
        "_CURRENT_CONTEXT_PARALLEL_SIZE",
        "_MAX_DYNAMIC_CONTEXT_PARALLEL_SIZE",
        "_CONTEXT_PARALLEL_GROUP",
    )

    def setUp(self):
        self._saved = {name: getattr(ps, name) for name in self._GLOBALS}
        self._saved_budget = cpu._DCP_PROBE_BUDGET[0]
        self.addCleanup(self._restore)
        # Start from "DCP off, no CP at all" so each test states its own setup.
        ps._DYNAMIC_CONTEXT_PARALLEL_GROUPS = {}
        ps._CURRENT_CONTEXT_PARALLEL_SIZE = None
        ps._CONTEXT_PARALLEL_GROUP = None

    def _restore(self):
        for name, value in self._saved.items():
            setattr(ps, name, value)
        cpu._DCP_PROBE_BUDGET[0] = self._saved_budget

    def enable_dcp(self, table, current):
        """Pretend ``initialize_dynamic_context_parallel_groups`` ran."""
        ps._DYNAMIC_CONTEXT_PARALLEL_GROUPS = dict(table)
        ps._MAX_DYNAMIC_CONTEXT_PARALLEL_SIZE = max(table)
        ps.set_current_context_parallel_size(current)


class TestResolveCurrentGroup(_DcpUtilsTestBase):
    """``get_current_cp_group`` is the only group source CP primitives may use."""

    def test_static_group_is_returned_while_dcp_is_off(self):
        static = _Group([0, 1], rank=1)
        ps._CONTEXT_PARALLEL_GROUP = static
        self.assertIs(cpu.get_current_cp_group(), static)
        self.assertEqual(cpu.get_current_cp_world_size(), 2)

    def test_uninitialized_fleet_resolves_to_no_group_instead_of_raising(self):
        # The code this replaced read ``pg_collection.cp`` / ``self.cp_group``
        # and simply got None in standalone op tests;
        # ``fleet.get_hybrid_communicate_group()`` asserts instead, so the
        # fallback has to swallow it and stay equally non-fatal.
        self.assertIsNone(cpu.get_current_cp_group())
        self.assertEqual(cpu.get_current_cp_world_size(), 1)

    def test_selected_size_wins_over_the_static_group(self):
        static = _Group([0, 1, 2, 3], rank=2)
        ps._CONTEXT_PARALLEL_GROUP = static
        g2 = _Group([2, 3], rank=0)
        self.enable_dcp({1: None, 2: g2, 4: static}, current=2)
        self.assertIs(cpu.get_current_cp_group(), g2)
        self.assertEqual(cpu.get_current_cp_world_size(), 2)

    def test_size_one_is_a_real_answer_not_a_missing_group(self):
        # A 1-rank CP group needs no communicator, so None here means "no
        # collective", not "not initialized" -- the world size must be 1 and
        # the static group must NOT be used as a fallback.
        ps._CONTEXT_PARALLEL_GROUP = _Group([0, 1], rank=0)
        self.enable_dcp({1: None, 2: _Group([0, 1])}, current=1)
        self.assertIsNone(cpu.get_current_cp_group())
        self.assertEqual(cpu.get_current_cp_world_size(), 1)


class TestRefreshCpState(_DcpUtilsTestBase):
    """Layers cache CP fields in ``__init__``; DCP has to re-point them."""

    def _layer(self):
        # The four fields csa_attention / mqa_latent_attention freeze, seeded
        # with the *static* cp=4 answer a layer would have captured.
        frozen = _Group([4, 5, 6, 7], rank=3)
        return SimpleNamespace(
            cp_group=frozen, cp_size=4, cp_rank=3, cp_enabled=True
        ), frozen

    def test_no_op_while_dcp_is_off(self):
        layer, frozen = self._layer()
        cpu.refresh_cp_state(layer)
        # Bitwise-unchanged behaviour with DCP off is the whole safety story of
        # this PR: identity, not just equality.
        self.assertIs(layer.cp_group, frozen)
        self.assertEqual((layer.cp_size, layer.cp_rank), (4, 3))
        self.assertTrue(layer.cp_enabled)

    def test_repoints_all_four_fields_at_the_current_group(self):
        layer, frozen = self._layer()
        g2 = _Group([6, 7], rank=1)
        self.enable_dcp({1: None, 2: g2, 4: frozen}, current=2)
        cpu.refresh_cp_state(layer)
        self.assertIs(layer.cp_group, g2)
        self.assertEqual((layer.cp_size, layer.cp_rank), (2, 1))
        self.assertTrue(layer.cp_enabled)

    def test_size_one_disables_cp_on_the_layer(self):
        layer, frozen = self._layer()
        self.enable_dcp({1: None, 4: frozen}, current=1)
        cpu.refresh_cp_state(layer)
        self.assertIsNone(layer.cp_group)
        self.assertEqual((layer.cp_size, layer.cp_rank), (1, 0))
        # Keyed on the size, not on "group is not None".
        self.assertFalse(layer.cp_enabled)


class TestRefreshContextParallelSize(_DcpUtilsTestBase):
    """``dot_product_attention`` / ``moe_router`` only cache the size."""

    def test_no_op_while_dcp_is_off(self):
        layer = SimpleNamespace(context_parallel_size=4)
        cpu.refresh_context_parallel_size(layer)
        self.assertEqual(layer.context_parallel_size, 4)

    def test_tracks_the_current_size(self):
        layer = SimpleNamespace(context_parallel_size=4)
        self.enable_dcp(
            {1: None, 2: _Group([0, 1]), 4: _Group([0, 1, 2, 3])}, 2
        )
        cpu.refresh_context_parallel_size(layer)
        self.assertEqual(layer.context_parallel_size, 2)

    def test_size_one_reads_as_one_not_zero(self):
        layer = SimpleNamespace(context_parallel_size=4)
        self.enable_dcp({1: None, 4: _Group([0, 1, 2, 3])}, 1)
        cpu.refresh_context_parallel_size(layer)
        self.assertEqual(layer.context_parallel_size, 1)


class TestScatterOpFollowsCurrentGroup(_DcpUtilsTestBase):
    """Numerical check: which rows this rank keeps must follow the *current*
    group, since the scatter is a pure local slice of the sequence axis."""

    SEQ = 16

    def _input(self):
        # Distinguishable content: a wrong slice origin or width cannot pass.
        data = np.arange(self.SEQ * 3, dtype="float32").reshape(1, self.SEQ, 3)
        return paddle.to_tensor(data), data

    def test_contiguous_slice_width_follows_the_selected_size(self):
        x, data = self._input()
        # This rank is cp_rank 3 of 4 statically, cp_rank 1 of the size-2 group.
        g4 = _Group([0, 1, 2, 3], rank=3)
        g2 = _Group([2, 3], rank=1)
        self.enable_dcp({1: None, 2: g2, 4: g4}, current=4)

        out4 = cpu.ContextParallelScatterOp.apply(x, 1, "contiguous_allgather")
        # cp=4: chunk = 16/4 = 4 rows, rank 3 -> rows [12, 16).
        np.testing.assert_array_equal(out4.numpy(), data[:, 12:16])

        ps.set_current_context_parallel_size(2)
        out2 = cpu.ContextParallelScatterOp.apply(x, 1, "contiguous_allgather")
        # cp=2: chunk = 16/2 = 8 rows, rank 1 -> rows [8, 16).
        np.testing.assert_array_equal(out2.numpy(), data[:, 8:16])

        # The stale-group bug showed up exactly as a wrong local width.
        self.assertEqual(out4.shape[1], 4)
        self.assertEqual(out2.shape[1], 8)

    def test_balanced_scatter_takes_one_chunk_from_each_end(self):
        x, data = self._input()
        g2 = _Group([2, 3], rank=1)
        self.enable_dcp({1: None, 2: g2}, current=2)
        out = cpu.ContextParallelScatterOp.apply(x, 1, "dualchunk_allgather")
        # interval = 16 // 2 // 2 = 4; rank 1 keeps rows [4, 8) from the front
        # and rows [16-8, 16-4) = [8, 12) from the back.
        expected = np.concatenate([data[:, 4:8], data[:, 8:12]], axis=1)
        np.testing.assert_array_equal(out.numpy(), expected)

    def test_gather_scatter_ops_refuse_a_size_one_group(self):
        x, _ = self._input()
        self.enable_dcp({1: None, 2: _Group([2, 3], rank=1)}, current=1)
        for op in (
            cpu.ContextParallelScatterOp,
            cpu.ContextParallelGatherOp,
            cpu.ContextParallelAllGatherOp,
        ):
            with self.assertRaises(AssertionError):
                op.apply(x, 1, "contiguous_allgather")


class TestSingleRankGroupIsBitwiseIdentity(_DcpUtilsTestBase):
    """With a 1-rank CP group every primitive must hand back the same values.

    This is the single-card contract the DCP refactor had to preserve: the
    helpers no longer read ``fleet.get_hybrid_communicate_group()`` themselves,
    they resolve ``group=None`` through ``get_current_cp_group()``, and a
    1-rank group still short-circuits to a copy. Exact equality, no tolerance:
    these paths copy, they do not compute.
    """

    def setUp(self):
        super().setUp()
        # A cp_degree=1 run really does have a 1-rank static CP group.
        ps._CONTEXT_PARALLEL_GROUP = _Group([0], rank=0)
        data = np.arange(24, dtype="float32").reshape(8, 3)
        self.data = data
        self.x = paddle.to_tensor(data)

    def _assert_independent_copy(self, out):
        np.testing.assert_array_equal(out.numpy(), self.data)
        # A copy, not a view: the callers free the input right after.
        self.assertNotEqual(out.data_ptr(), self.x.data_ptr())

    def test_reduce_scatter_any_axis(self):
        self._assert_independent_copy(cpu.reduce_scatter_any_axis(self.x, 0))

    def test_scatter_contiguous(self):
        self._assert_independent_copy(cpu.scatter_contiguous(self.x, axis=0))

    def test_all_gather_contiguous(self):
        self._assert_independent_copy(cpu.all_gather_contiguous(self.x, axis=0))

    def test_reduce_scatter_contiguous(self):
        self._assert_independent_copy(
            cpu.reduce_scatter_contiguous(self.x, axis=0)
        )

    def test_normal_scatter_and_gather(self):
        self._assert_independent_copy(
            cpu.ContextParallelNormalScatter.apply(self.x, 0, 0)
        )
        self._assert_independent_copy(
            cpu.ContextParallelNormalGather.apply(self.x, 0, 0)
        )

    def test_balanced_reduce_scatter(self):
        try:
            import triton  # noqa: F401
        except ImportError as exc:  # the balanced path is a triton kernel
            self.skipTest(f"triton unavailable: {exc}")
        self._assert_independent_copy(
            cpu.reduce_scatter_any_axis_balance(self.x, 0)
        )


class TestNormalScatterFollowsCurrentGroup(_DcpUtilsTestBase):
    """``ContextParallelNormalScatter`` splits by the current group as well.

    ``scatter_with_padding`` is local (``paddle.split`` plus an optional tail
    pad), so the rows this rank keeps can be checked exactly.
    """

    def test_split_follows_the_selected_size(self):
        data = np.arange(8 * 2, dtype="float32").reshape(8, 2)
        x = paddle.to_tensor(data)
        g2 = _Group([2, 3], rank=1)
        g4 = _Group([0, 1, 2, 3], rank=3)
        ps._DYNAMIC_CONTEXT_PARALLEL_GROUPS = {1: None, 2: g2, 4: g4}
        ps._MAX_DYNAMIC_CONTEXT_PARALLEL_SIZE = 4

        ps.set_current_context_parallel_size(2)
        # avg = 8 // 2 = 4 rows per rank; rank 1 keeps rows [4, 8).
        out2 = cpu.ContextParallelNormalScatter.apply(x, 0, 0)
        np.testing.assert_array_equal(out2.numpy(), data[4:8])

        ps.set_current_context_parallel_size(4)
        # avg = 8 // 4 = 2 rows per rank; rank 3 keeps rows [6, 8).
        out4 = cpu.ContextParallelNormalScatter.apply(x, 0, 0)
        np.testing.assert_array_equal(out4.numpy(), data[6:8])


class TestRefreshAfterReset(_DcpUtilsTestBase):
    """With DCP enabled but no size selected, cached fields go back to static."""

    def test_cp_state_is_rederived_from_the_static_group(self):
        static = _Group([4, 5, 6, 7], rank=3)
        ps._CONTEXT_PARALLEL_GROUP = static
        g2 = _Group([6, 7], rank=1)
        layer = SimpleNamespace(
            cp_group=static, cp_size=4, cp_rank=3, cp_enabled=True
        )
        self.enable_dcp({1: None, 2: g2, 4: static}, current=2)
        cpu.refresh_cp_state(layer)
        self.assertEqual((layer.cp_size, layer.cp_rank), (2, 1))

        ps.reset_current_context_parallel_size()
        cpu.refresh_cp_state(layer)
        self.assertIs(layer.cp_group, static)
        self.assertEqual((layer.cp_size, layer.cp_rank), (4, 3))
        self.assertTrue(layer.cp_enabled)

    def test_context_parallel_size_is_rederived_from_the_static_group(self):
        static = _Group([0, 1, 2, 3])
        ps._CONTEXT_PARALLEL_GROUP = static
        layer = SimpleNamespace(context_parallel_size=4)
        self.enable_dcp({1: None, 2: _Group([0, 1]), 4: static}, current=2)
        cpu.refresh_context_parallel_size(layer)
        self.assertEqual(layer.context_parallel_size, 2)

        ps.reset_current_context_parallel_size()
        cpu.refresh_context_parallel_size(layer)
        self.assertEqual(layer.context_parallel_size, 4)

    def test_layer_with_its_own_group_keeps_it(self):
        # A collection holding a group other than the static CP group is a
        # layer-specific choice and must not be redirected by DCP.
        ps._CONTEXT_PARALLEL_GROUP = _Group([0, 1, 2, 3])
        own = _Group([8, 9], rank=1)
        layer = SimpleNamespace(
            pg_collection=ProcessGroupCollection(cp=own),
            cp_group=own,
            cp_size=2,
            cp_rank=1,
            cp_enabled=True,
        )
        self.enable_dcp(
            {1: None, 4: ps._CONTEXT_PARALLEL_GROUP, 8: _Group(range(8))}, 8
        )
        cpu.refresh_cp_state(layer)
        self.assertIs(layer.cp_group, own)
        self.assertEqual((layer.cp_size, layer.cp_rank), (2, 1))


class TestProcessGroupCollectionCp(_DcpUtilsTestBase):
    """``pg_collection.cp`` is read at use time, so it follows DCP."""

    def test_returns_the_stored_group_while_dcp_is_off(self):
        static = _Group([0, 1])
        ps._CONTEXT_PARALLEL_GROUP = static
        self.assertIs(ProcessGroupCollection(cp=static).cp, static)

    def test_static_group_follows_the_selected_size(self):
        static = _Group([0, 1, 2, 3], rank=2)
        ps._CONTEXT_PARALLEL_GROUP = static
        g2 = _Group([2, 3], rank=0)
        pgc = ProcessGroupCollection(cp=static)
        self.enable_dcp({1: None, 2: g2, 4: static}, current=2)
        self.assertIs(pgc.cp, g2)
        ps.set_current_context_parallel_size(1)
        self.assertIsNone(pgc.cp)
        ps.reset_current_context_parallel_size()
        self.assertIs(pgc.cp, static)

    def test_other_groups_are_not_redirected(self):
        ps._CONTEXT_PARALLEL_GROUP = _Group([0, 1, 2, 3])
        own = _Group([8, 9])
        self.enable_dcp({1: None, 4: ps._CONTEXT_PARALLEL_GROUP}, current=1)
        self.assertIs(ProcessGroupCollection(cp=own).cp, own)

    def test_mpu_collection_follows_later_selections(self):
        # Collections are built with the model, before any size is selected,
        # so they store the static group and follow every later selection.
        static = _Group([0, 1, 2, 3])
        ps._CONTEXT_PARALLEL_GROUP = static
        pgc = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["cp"])
        g2 = _Group([0, 1])
        self.enable_dcp({1: None, 2: g2, 4: static}, current=2)
        self.assertIs(pgc.cp, g2)
        ps.reset_current_context_parallel_size()
        self.assertIs(pgc.cp, static)

    def test_unset_cp_raises_attribute_error(self):
        pgc = ProcessGroupCollection()
        self.assertFalse(hasattr(pgc, "cp"))
        self.assertIsNone(getattr(pgc, "cp", None))


class TestKimiRejectsDynamicSize(_DcpUtilsTestBase):
    """KDA freezes cp_size in __init__, so it refuses a different DCP size."""

    def _forward(self, cp_size, cp_group):
        try:
            from paddlefleet.transformer.kimi_delta_attention import (
                KimiDeltaAttention,
            )
        except ImportError as exc:
            self.skipTest(f"kimi_delta_attention unavailable: {exc}")
        layer = SimpleNamespace(
            pg_collection=ProcessGroupCollection(cp=cp_group), cp_size=cp_size
        )
        with mock.patch(
            "paddlefleet.transformer.kimi_delta_attention.get_pg_size",
            lambda g: 1 if g is None else g.nranks,
        ):
            KimiDeltaAttention.forward(layer, paddle.zeros([1, 2, 4]))

    def test_mismatched_dynamic_size_raises(self):
        static = _Group([0, 1, 2, 3])
        ps._CONTEXT_PARALLEL_GROUP = static
        self.enable_dcp({1: None, 2: _Group([0, 1]), 4: static}, current=2)
        with self.assertRaisesRegex(
            NotImplementedError, "cp_size=4.*cp_size=2"
        ):
            self._forward(4, static)


if __name__ == "__main__":
    unittest.main()
