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
"""Per-layer grouped down-projection runtime used by ``RingMoETokenDispatcher``.

Every MoE layer with the grouped down projection owns one
``RingGroupedDownProj``; its plans are built on the GPU.

Per ring forward:
    gdp.start_forward(T, K, E, intra_group)   -> runs the grouped path this forward?
    pair = gdp.extend_pair(local_idx, pair)   this rank's plan part appended to
                                              its routing pair (per-piece plans)
    gdp.submit(round, gathered_pair, comm_stream) as soon as a round's routing
                                              gather is enqueued
    with gdp.round(round, ids):               around the round's expert call
        expert_fn(...)                        sonic calls the merged chain
"""

import contextlib
import importlib
import inspect
import logging
from dataclasses import dataclass

import paddle

from ._streams import _current_stream, _on_stream

logger = logging.getLogger(__name__)

# A per-round plan (`plan_gpu`) stores its token index as int16: a round of
# more than this many gathered tokens needs per-piece plans.
MAX_TOKENS = 65536


@dataclass(frozen=True)
class GroupedDownProjConfig:
    list_path: str  # group list of the layer (``row_group_planner.save_list``)


def _grouped():
    return importlib.import_module(
        "paddlefleet_ops.sonicmoe.functional.grouped_down_proj"
    )


def expert_k_supported(I):
    """Whether the grouped down projection runs with a per-rank intermediate
    size (per expert) of ``I``: sonic's ``supports_expert_k`` when it has one
    (any multiple of 16 with a short-segment quack), else multiples of 64."""
    fn = getattr(_grouped(), "supports_expert_k", None)
    return bool(fn(I)) if fn is not None else I > 0 and I % 64 == 0


def enable_grouped_down_proj(dispatcher, list_path, intermediate, fp8_dispatch):
    """Enable the grouped down projection on a ``RingMoETokenDispatcher`` when
    ``list_path`` is set, the dispatch is not fp8 and the per-rank
    intermediate size ``intermediate`` is supported; otherwise warn (if a list
    was given) and keep the default down projection.  Returns whether it was
    enabled."""
    if not list_path or fp8_dispatch:
        return False
    # Each expert's slice of the down weight is one K segment of the grouped
    # GEMM; the installed sonic / quack decide which segment lengths they run.
    if not expert_k_supported(intermediate):
        logger.warning(
            "moe_ring_grouped_down_proj_list ignored: per-rank "
            "moe_intermediate_size %d is not supported by the grouped down "
            "projection",
            intermediate,
        )
        return False
    dispatcher.enable_grouped_down_proj(
        GroupedDownProjConfig(list_path=list_path)
    )
    return True


class RingGroupedDownProj:
    def __init__(self, cfg: GroupedDownProjConfig):
        _grouped()  # fail early if the sonic build lacks the grouped down projection
        self.cfg = cfg
        self.groups = None
        self.active = False
        # Runtime switch: False runs the production path from the next forward
        # (e.g. for an A/B inside one process).
        self.enabled = True
        self.gpu_index = self._err = None
        self._plans = {}  # round -> (plan, ids, ready_event)
        # Per-piece plans (sonic `grouped_plan_pieces`): each rank selects for
        # its own tokens once per forward, the selection rides with the
        # routing pair, and a round only merges the pieces.
        self.pieces = None
        self._pieces_mod = None
        self._hold = {}
        self._bwd = False

    def _start_gpu(self, E, side):
        gp = importlib.import_module(
            "paddlefleet_ops.sonicmoe.functional.grouped_plan_gpu"
        )
        rgp = importlib.import_module(
            "paddlefleet_ops.sonicmoe.row_group_planner"
        )
        index = rgp.build_index(rgp.load_list(self.cfg.list_path), E)
        self.gpu_index = gp.GPUIndex(index)
        self._plan_gpu = gp.plan_gpu
        self.groups = _grouped().GroupList(index)
        self._err = paddle.zeros([1], dtype=paddle.int32)
        self._side = side
        try:
            self._pieces_mod = importlib.import_module(
                "paddlefleet_ops.sonicmoe.functional.grouped_plan_pieces"
            )
        except ImportError:  # older sonic: plan every round with `plan_gpu`
            self._pieces_mod = None
        # Newer sonic: a per-piece plan can also serve the up projection's
        # input gradient (``backward=True``); the plan must then outlive the
        # forward, so every round merges into a freshly allocated plan.
        self._bwd = (
            self._pieces_mod is not None
            and hasattr(self._pieces_mod.PlanPieces, "_new_plan")
            and "backward"
            in inspect.signature(_grouped().grouped_down_proj_plan).parameters
        )

    def start_forward(self, T, K, E, intra_group):
        """Begin one ring forward; returns whether the grouped path runs in it.
        Rounds of more than ``MAX_TOKENS`` tokens need per-piece plans (a
        sonic whose `PlanPieces` accepts the batch); else they run the
        production path."""
        if T <= 0:
            self.active = False
            return False
        if self.gpu_index is None:
            self._start_gpu(E, paddle.device.cuda.Stream())
        self._plans = {}
        self._hold = {}
        self._T, self._K = T, K
        self._set_pieces(T, K, intra_group)
        self.active = self.enabled and (
            T <= MAX_TOKENS or self.pieces is not None
        )
        return self.active

    def _set_pieces(self, T, K, intra_group):
        P = intra_group.nranks if intra_group is not None else 1
        pp = self.pieces
        if pp is not None and (pp.TL * pp.P, pp.K, pp.P) == (T, K, P):
            return
        self.pieces = None
        if self._pieces_mod is None or P < 2 or T % P:
            return
        try:
            self.pieces = self._pieces_mod.PlanPieces(
                self.gpu_index, T // P, K, P
            )
        except ValueError:  # shape outside its limits: per-round plan
            self.pieces = None

    def extend_pair(self, local_idx, pair):
        """Return this rank's routing pair ([T_l, 2K]) with its part of the
        plan appended (per-piece plans; else unchanged).  Called once per
        forward, after `start_forward`."""
        if not (self.active and self.pieces is not None):
            return pair
        x = self.pieces.local_select(
            local_idx.astype(paddle.int32).contiguous()
        )
        return paddle.concat([pair, x], axis=1)

    def submit(self, rnd, pair, src_stream=None, row0=0):
        """Hand round `rnd`'s gathered routing pair (rows row0 .. row0 + T of
        a [rows, 2K + parts] tensor, ids first) to the planner; once per round
        and forward, later calls are ignored.

        Per-piece plans merge the round's pieces on the side stream as soon as
        its routing is gathered (a round ahead for most rounds).  Otherwise
        the plan is built at `round` entry on the side stream (overlapping the
        round's GEMM1 + activation)."""
        pp = self.pieces
        if (
            not self.active
            or pp is None
            or rnd in self._plans
            or int(pair.shape[1]) != 2 * self._K + pp.NX
        ):
            return
        e_cur = paddle.device.cuda.Event()
        e_cur.record(_current_stream())
        self._side.wait_event(e_cur)  # the slot's previous plan is consumed
        if src_stream is not None:
            e_src = paddle.device.cuda.Event()
            e_src.record(src_stream)
            self._side.wait_event(e_src)
        with _on_stream(self._side):
            plan = pp.merge(
                pair[row0 : row0 + self._T],
                slot=None if self._bwd else rnd,
                x_col=2 * self._K,
            )
        ev = paddle.device.cuda.Event()
        ev.record(self._side)
        self._hold[rnd] = pair  # read on the side stream
        self._plans[rnd] = (plan, None, ev)

    @contextlib.contextmanager
    def round(self, rnd, ids):
        """Run the round's expert call with the merged down projection.
        ids: the round's [T, K] int32 routing ids, checked against the lanes."""
        if not self.active:
            yield
            return
        pe = self._plans.pop(rnd, None)
        if pe is None and self._T > MAX_TOKENS:  # `plan_gpu` is int16-bound
            yield
            return
        if pe is None:  # not submitted ahead: build inline on the side stream
            e0 = paddle.device.cuda.Event()
            e0.record(_current_stream())
            self._side.wait_event(e0)
            with _on_stream(self._side):
                plan = self._plan_gpu(ids, self.gpu_index)
            ev = paddle.device.cuda.Event()
            ev.record(self._side)
            pe = (plan, ids, ev)
        plan, pids, ev = pe
        # A merged per-piece plan is this round's own (fresh) buffers: it
        # also serves the round's input gradient in the backward.
        kw = {"backward": True} if pids is None and self._bwd else {}
        with _grouped().grouped_down_proj_plan(
            self.groups,
            plan,
            ids if pids is None else pids,
            self._err,
            ready_event=ev,
            **kw,
        ):
            yield
