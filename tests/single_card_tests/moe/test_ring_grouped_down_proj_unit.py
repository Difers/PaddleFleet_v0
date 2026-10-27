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
"""Grouped down projection runtime (``grouped_down_proj.ring``) against
stand-in sonic modules: which plan a round runs with, when the grouped path
is on, and how the layer enables it.  The kernels themselves are covered by
``test_ring_grouped_down_proj_gpu.py`` on builds that have them."""

import contextlib
import sys
import types

import paddle
import pytest

from paddlefleet.transformer.moe.grouped_down_proj import _streams, ring as R

pytestmark = pytest.mark.skipif(
    not paddle.is_compiled_with_cuda(), reason="needs CUDA streams and events"
)

_SONIC = "paddlefleet_ops.sonicmoe."
_GROUPED = _SONIC + "functional.grouped_down_proj"
_PLAN_GPU = _SONIC + "functional.grouped_plan_gpu"
_PIECES = _SONIC + "functional.grouped_plan_pieces"
_PLANNER = _SONIC + "row_group_planner"


class _Group:
    def __init__(self, nranks):
        self.nranks = nranks


class _Sonic:
    """Stand-ins for the sonic modules the runtime imports; records calls."""

    def __init__(self, pieces=True, backward=True, new_plan=True, k_fn=True):
        self.calls = []
        rec = self.calls

        g = types.ModuleType(_GROUPED)
        g.GroupList = lambda index: ("groups", index)
        if backward:

            @contextlib.contextmanager
            def plan_ctx(
                groups, plan, ids, err, ready_event=None, backward=False
            ):
                rec.append(("plan", plan, ids, backward, ready_event))
                yield

        else:

            @contextlib.contextmanager
            def plan_ctx(groups, plan, ids, err, ready_event=None):
                rec.append(("plan", plan, ids, False, ready_event))
                yield

        g.grouped_down_proj_plan = plan_ctx
        if k_fn:
            g.supports_expert_k = lambda I: I % 16 == 0
        self.grouped = g

        gp = types.ModuleType(_PLAN_GPU)
        gp.GPUIndex = lambda index: ("gpu_index", index)

        def plan_gpu(ids, index):
            rec.append(("plan_gpu", index))
            return "round_plan"

        gp.plan_gpu = plan_gpu
        self.plan_gpu = gp

        rgp = types.ModuleType(_PLANNER)
        rgp.load_list = lambda path: ("list", path)
        rgp.build_index = lambda lst, E: ("index", lst, E)
        self.planner = rgp

        self.pieces = None
        if pieces:
            pm = types.ModuleType(_PIECES)

            class PlanPieces:
                NX = 3

                def __init__(self, gpu_index, TL, K, P):
                    if TL < 2:
                        raise ValueError("too few tokens")
                    self.TL, self.K, self.P = TL, K, P
                    rec.append(("pieces", TL, K, P))

                def local_select(self, idx):
                    rec.append(("select", list(idx.shape), idx.dtype))
                    return paddle.full([idx.shape[0], self.NX], 7.0, "float32")

                def merge(self, src, slot, x_col):
                    rec.append(("merge", list(src.shape), slot, x_col))
                    return ("merged", slot)

            if new_plan:
                PlanPieces._new_plan = None
            pm.PlanPieces = PlanPieces
            self.pieces = pm

    def install(self, mp):
        mp.setitem(sys.modules, _GROUPED, self.grouped)
        mp.setitem(sys.modules, _PLAN_GPU, self.plan_gpu)
        mp.setitem(sys.modules, _PLANNER, self.planner)
        # None in sys.modules makes the import raise ImportError (older sonic)
        mp.setitem(sys.modules, _PIECES, self.pieces)
        return self

    def kinds(self, kind):
        return [c for c in self.calls if c[0] == kind]


def _runtime():
    return R.RingGroupedDownProj(R.GroupedDownProjConfig(list_path="g.npz"))


def _pair(T, K, extra=0):
    return paddle.ones([T, 2 * K + extra], dtype="float32")


def _ids(T, K):
    return paddle.zeros([T, K], dtype="int32")


def test_rounds_over_the_token_limit_run_the_default_path(monkeypatch):
    sonic = _Sonic().install(monkeypatch)
    rt = _runtime()
    assert not rt.start_forward(0, 8, 16, _Group(2))
    assert not rt.active
    assert rt.gpu_index is None  # no planner state built
    with rt.round(0, _ids(4, 8)):
        pass
    assert sonic.calls == []
    # More than MAX_TOKENS tokens: only with per-piece plans.
    assert not rt.start_forward(R.MAX_TOKENS + 1, 8, 16, _Group(1))
    assert not rt.active
    T = 2 * R.MAX_TOKENS
    assert rt.start_forward(T, 8, 16, _Group(2))
    assert sonic.kinds("pieces") == [("pieces", T // 2, 8, 2)]
    with rt.round(0, _ids(4, 8)):  # not submitted: no int16-bound round plan
        pass
    assert sonic.kinds("plan_gpu") == [] and sonic.kinds("plan") == []


def test_rounds_over_the_token_limit_without_pieces(monkeypatch):
    _Sonic(pieces=False).install(monkeypatch)
    rt = _runtime()
    assert not rt.start_forward(R.MAX_TOKENS + 2, 8, 16, _Group(2))
    assert rt.start_forward(R.MAX_TOKENS, 8, 16, _Group(2))


def test_per_piece_plans_ride_with_the_routing(monkeypatch):
    sonic = _Sonic().install(monkeypatch)
    rt = _runtime()
    T, K, E, P = 8, 2, 4, 2
    assert rt.start_forward(T, K, E, _Group(P))
    assert rt.gpu_index == ("gpu_index", ("index", ("list", "g.npz"), E))
    assert rt.groups == ("groups", ("index", ("list", "g.npz"), E))
    assert rt._bwd
    assert rt.start_forward(T, K, E, _Group(P))  # same shape: pieces kept
    assert sonic.kinds("pieces") == [("pieces", T // P, K, P)]

    local_idx = paddle.zeros([T // P, K], dtype="int64")
    pair = rt.extend_pair(local_idx, _pair(T // P, K))
    assert pair.shape == [T // P, 2 * K + 3]
    assert sonic.kinds("select") == [("select", [T // P, K], paddle.int32)]

    rows = 2 * T  # two rounds gathered in one tensor
    wide = _pair(rows, K, extra=3)
    rt.submit(0, _pair(rows, K))  # no plan parts in it: ignored
    rt.submit(1, wide, src_stream=_streams._current_stream(), row0=T)
    rt.submit(1, wide)  # later submits of a round are ignored
    rt.submit(0, wide)
    assert sonic.kinds("merge") == [
        ("merge", [T, 2 * K + 3], None, 2 * K),
        ("merge", [T, 2 * K + 3], None, 2 * K),
    ]

    ids = _ids(T, K)
    for rnd in (1, 0):
        with rt.round(rnd, ids):
            pass
    plans = sonic.kinds("plan")
    assert [c[1] for c in plans] == [("merged", None)] * 2
    assert all(c[2] is ids and c[3] for c in plans)  # also serves the backward
    assert all(isinstance(c[4], paddle.device.cuda.Event) for c in plans)
    assert sonic.kinds("plan_gpu") == []


def test_round_plans_inline_without_pieces(monkeypatch):
    sonic = _Sonic(pieces=False).install(monkeypatch)
    rt = _runtime()
    T, K = 8, 2
    assert rt.start_forward(T, K, 4, _Group(1))
    assert rt.pieces is None and not rt._bwd
    pair = _pair(T, K)
    assert rt.extend_pair(_ids(T, K), pair) is pair
    rt.submit(0, pair)  # nothing to merge: built at round entry
    ids = _ids(T, K)
    with rt.round(0, ids):
        pass
    assert len(sonic.kinds("plan_gpu")) == 1
    (_, plan, pids, backward, ev) = sonic.kinds("plan")[0]
    assert plan == "round_plan" and pids is ids and not backward
    assert isinstance(ev, paddle.device.cuda.Event)


def test_pieces_off_for_unsupported_shapes(monkeypatch):
    _Sonic().install(monkeypatch)
    rt = _runtime()
    assert rt.start_forward(9, 2, 4, _Group(2))  # T not a multiple of P
    assert rt.pieces is None
    assert rt.start_forward(2, 2, 4, _Group(2))  # PlanPieces rejects it
    assert rt.pieces is None
    assert rt.start_forward(8, 2, 4, None)  # no intra group: one piece
    assert rt.pieces is None


def test_older_pieces_keep_per_round_slots(monkeypatch):
    sonic = _Sonic(new_plan=False).install(monkeypatch)
    rt = _runtime()
    T, K = 8, 2
    assert rt.start_forward(T, K, 4, _Group(2)) and not rt._bwd
    rt.submit(1, _pair(T, K, extra=3))
    assert sonic.kinds("merge") == [("merge", [T, 2 * K + 3], 1, 2 * K)]
    with rt.round(1, _ids(T, K)):
        pass
    assert not sonic.kinds("plan")[0][3]  # no backward reuse


def test_backward_needs_sonic_support(monkeypatch):
    _Sonic(backward=False).install(monkeypatch)
    rt = _runtime()
    assert rt.start_forward(8, 2, 4, _Group(2))
    assert rt.pieces is not None and not rt._bwd


def test_disabled_runs_the_production_path(monkeypatch):
    sonic = _Sonic().install(monkeypatch)
    rt = _runtime()
    rt.enabled = False
    assert not rt.start_forward(8, 2, 4, _Group(2))
    pair = _pair(4, 2)
    assert rt.extend_pair(_ids(4, 2), pair) is pair
    rt.submit(0, _pair(8, 2, extra=3))
    with rt.round(0, _ids(8, 2)):
        pass
    assert sonic.kinds("merge") == [] and sonic.kinds("plan") == []


def test_expert_k_supported(monkeypatch):
    sonic = _Sonic().install(monkeypatch)
    assert R.expert_k_supported(48) and not R.expert_k_supported(40)
    del sonic.grouped.supports_expert_k  # older sonic: multiples of 64
    assert R.expert_k_supported(128) and not R.expert_k_supported(96)
    assert not R.expert_k_supported(0)


class _Dispatcher:
    def __init__(self):
        self.configs = []

    def enable_grouped_down_proj(self, config):
        self.configs.append(config)


def test_enable_for_layer(monkeypatch):
    _Sonic().install(monkeypatch)
    warnings = []
    monkeypatch.setattr(
        R.logger, "warning", lambda msg, *a: warnings.append(msg % a)
    )
    d = _Dispatcher()
    assert not R.enable_grouped_down_proj(d, None, 64, False)
    assert not R.enable_grouped_down_proj(d, "g.npz", 64, True)  # fp8
    assert warnings == []
    assert not R.enable_grouped_down_proj(d, "g.npz", 40, False)
    assert len(warnings) == 1
    assert "per-rank moe_intermediate_size 40" in warnings[0]
    assert d.configs == []
    assert R.enable_grouped_down_proj(d, "g.npz", 64, False)
    assert d.configs == [R.GroupedDownProjConfig(list_path="g.npz")]


def test_ring_dispatcher_owns_the_runtime(monkeypatch):
    _Sonic().install(monkeypatch)
    from paddlefleet.transformer.moe.token_dispatcher import (
        RingMoETokenDispatcher,
    )

    disp = types.SimpleNamespace()
    cfg = R.GroupedDownProjConfig(list_path="g.npz")
    RingMoETokenDispatcher.enable_grouped_down_proj(disp, cfg)
    assert isinstance(disp._gdp, R.RingGroupedDownProj)
    assert disp._gdp.cfg == cfg


def test_on_stream_restores_the_stream(monkeypatch):
    seen = []
    monkeypatch.setattr(
        _streams, "_set_current_stream", lambda st: seen.append(st) or "prev"
    )
    with _streams._on_stream("side"):
        assert seen == ["side"]
    assert seen == ["side", "prev"]


def test_on_stream_without_the_private_setter(monkeypatch):
    seen = []

    @contextlib.contextmanager
    def guard(stream):
        seen.append(("enter", stream))
        yield
        seen.append(("exit", stream))

    monkeypatch.setattr(_streams, "_set_current_stream", None)
    monkeypatch.setattr(paddle.device.cuda, "stream_guard", guard)
    with _streams._on_stream("side"):
        assert seen == [("enter", "side")]
    assert seen == [("enter", "side"), ("exit", "side")]
