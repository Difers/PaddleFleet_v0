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
"""grouped down projection, GPU planner (one rank standing in for a node):
each piece's `extend_pair` + the round's `submit` of the gathered pair must
install the same plan as `plan_gpu` on the round's routing, for a round
submitted ahead and for one gathered inline."""

import numpy as np
import paddle
import pytest

paddle.enable_compat(silent=True)
try:
    import importlib

    import paddlefleet_ops  # noqa: F401
    import paddlefleet_ops.sonicmoe.row_group_planner as P

    # The ring module before sonic's functional modules, as at runtime.
    from paddlefleet.transformer.moe.grouped_down_proj import ring as R

    _sonic = "paddlefleet_ops.sonicmoe.functional."
    importlib.import_module(_sonic + "grouped_plan_pieces")
    current_grouped_plan = importlib.import_module(
        _sonic + "grouped_down_proj"
    ).current_grouped_plan
    _gp = importlib.import_module(_sonic + "grouped_plan_gpu")
    GPUIndex, plan_gpu = _gp.GPUIndex, _gp.plan_gpu
except (ImportError, RuntimeError) as exc:
    # Needs sonic's grouped down projection with per-piece plans (Blackwell
    # build); skip elsewhere.  A mark, not a module-level skip, so pytest
    # exits 0 when everything is skipped.
    pytestmark = pytest.mark.skip(
        reason=f"grouped down projection unavailable: {exc}"
    )


class _Group:
    def __init__(self, nranks):
        self.nranks, self.rank = nranks, 0


def _pairs(ids, n):
    from collections import Counter
    from itertools import combinations

    cnt = Counter()
    for r in ids[:2048]:
        cnt.update(combinations(sorted(int(e) for e in r if e >= 0), 2))
    out, used = [], set()
    for (a, b), _ in cnt.most_common():
        if len(out) == n:
            break
        if (a, b) not in used:
            out.append([a, b])
            used.add((a, b))
    return out


def test_gpu_pieces_plan_matches_plan_gpu(tmp_path):
    E, K, P_, TL, N = 64, 8, 4, 1024, 2
    T = P_ * TL
    rng = np.random.RandomState(5)
    ids = np.stack([rng.choice(E, K, replace=False) for _ in range(N * T)])
    ids = ids.astype(np.int32)
    ids[rng.rand(N * T, K) < 0.05] = -1
    w = rng.rand(N * T, K).astype(np.float32)
    lp = str(tmp_path / "groups.npz")
    P.save_list(lp, _pairs(ids, 200))
    rt = R.RingGroupedDownProj(R.GroupedDownProjConfig(list_path=lp))
    gidx = GPUIndex(P.build_index(P.load_list(lp), E))
    used = np.zeros(gidx.cu_len, bool)
    for _, s0, s1, at in gidx.gemms:
        used[at : at + s1 - s0 + 1] = True
    for _ in range(2):  # second forward reuses the plan buffers
        assert rt.start_forward(T, K, E, _Group(P_))
        assert rt.pieces is not None
        rounds = []
        for rnd in range(N):
            rows = slice(rnd * T, (rnd + 1) * T)
            idx_r = paddle.to_tensor(ids[rows])
            parts = []
            for j in range(P_):  # every rank of the node extends its own pair
                sl = slice(j * TL, (j + 1) * TL)
                pair = paddle.concat(
                    [
                        idx_r[sl].astype("float32"),
                        paddle.to_tensor(w[rows][sl]),
                    ],
                    axis=1,
                )
                parts.append(rt.extend_pair(idx_r[sl], pair))
            gathered = paddle.concat(parts, axis=0)
            rounds.append((idx_r, gathered))
        # round 1 submitted ahead (comm-stream style), round 0 inline
        rt.submit(1, rounds[1][1], R._current_stream())
        for rnd in (1, 0):
            idx_r, gathered = rounds[rnd]
            if rnd == 0:
                rt.submit(0, gathered)
            with rt.round(rnd, idx_r):
                cur = current_grouped_plan()
                # newer sonic: the merged plan also serves the input gradient
                assert getattr(cur, "backward", False) == rt._bwd
                R._current_stream().wait_event(cur.ready_event)
                paddle.device.synchronize()
                ref = plan_gpu(idx_r, gidx)
                paddle.device.synchronize()
                np.testing.assert_array_equal(
                    cur.plan.cu.numpy()[used], ref.cu.numpy()[used]
                )
                np.testing.assert_array_equal(
                    cur.plan.moff.numpy(), ref.moff.numpy()
                )
                n = int(ref.moff.numpy()[-1])
                tok = np.repeat(np.arange(T), np.diff(cur.plan.moff.numpy()))
                rows_tok = cur.plan.row_tok.numpy()[cur.plan.perm.numpy()[:n]]
                np.testing.assert_array_equal(
                    rows_tok.astype(np.int64) & 0xFFFF, tok
                )
        assert rt._plans == {}


def test_gpu_pieces_off_when_shape_does_not_fit(tmp_path):
    """A single-rank node or more segments than tokens per piece keeps the
    per-round plan and an unchanged routing pair."""
    lp = str(tmp_path / "groups.npz")
    P.save_list(lp, [[0, 1], [2, 3]])
    rt = R.RingGroupedDownProj(R.GroupedDownProjConfig(list_path=lp))
    assert rt.start_forward(64, 4, 16, _Group(1))
    assert rt.pieces is None
    idx = paddle.zeros([64, 4], dtype="int32")
    pair = paddle.zeros([64, 8], dtype="float32")
    assert rt.extend_pair(idx, pair) is pair
    assert rt.start_forward(32, 4, 16, _Group(4))  # 8 tokens < 18 segments
    assert rt.pieces is None


def test_expert_k_supported():
    """64-multiples always run; other sizes follow the installed sonic/quack
    (multiples of 16 with a short-segment segmented B), never non-16."""
    assert R.expert_k_supported(64) and R.expert_k_supported(128)
    assert not R.expert_k_supported(24) and not R.expert_k_supported(0)
    assert R.expert_k_supported(32) == R.expert_k_supported(48)
