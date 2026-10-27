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
"""Learn an expert-pair group list for RingMoE's grouped down projection.

The grouped down projection (``moe_ring_grouped_down_proj_list``) merges, per
token, lane-disjoint pairs of its routed experts that appear in a fixed list;
the fewer rows remain, the more it saves.  This tool learns that list from a
recorded routing of the model (CPU only, no GPU):

  * fit on the first half of the tokens, report on the second half;
  * grow the list greedily: plan the fit half with the current list, count
    expert pairs among the lanes left unmerged, append the most frequent new
    pairs (lowest priority), repeat until ``--pairs`` pairs;
  * finally order the pairs by how many other pairs share their experts
    (fewest first), which is the priority the planner's greedy works best with.

Input: an npz with ``indices`` of shape ``[T, K]`` or ``[layers, T, K]``
(expert ids, -1 = dropped; ``--layer`` picks one layer).  Output: a list file
for ``TransformerConfig.moe_ring_grouped_down_proj_list``.

    python make_group_list.py --routing routing.npz --experts 512 -o list.npz

More pairs merge more rows but split each grouped GEMM into more, smaller
segments; around 2x the expert count is a good default.  Learn the list on the
model's own routing and relearn it when the routing changes.
"""

import argparse
import importlib.util
import os

import numpy as np


def _planner():
    """``paddlefleet_ops.sonicmoe.row_group_planner``, loaded from its file so
    no GPU package initialization runs (the planner is numba + numpy only)."""
    spec = importlib.util.find_spec("paddlefleet_ops")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("paddlefleet_ops is not importable")
    for root in spec.submodule_search_locations:
        path = os.path.join(root, "sonicmoe", "row_group_planner.py")
        if os.path.exists(path):
            s = importlib.util.spec_from_file_location("_rgp", path)
            mod = importlib.util.module_from_spec(s)
            s.loader.exec_module(mod)
            return mod
    raise SystemExit("paddlefleet_ops.sonicmoe.row_group_planner not found")


def _plan(P, groups, ids, E, threads):
    index = P.build_index(groups, E)
    out = P.alloc_out(ids.shape[0], ids.shape[1], index)
    n = P.plan(
        ids.astype(np.int16),
        index,
        P.Scratch(ids.shape[0], ids.shape[1], index, threads),
        out,
    )
    return index, out, n


def _residual_lanes(P, groups, ids, E, threads):
    """[T, K] bool: lanes the planner leaves as single-expert rows."""
    index, out, n = _plan(P, groups, ids, E, threads)
    res = np.zeros(ids.shape, bool)
    _, s0, s1, at = index.gemms[-1]  # the residual GEMM comes last
    lo, hi = int(out["cu"][at]), int(out["cu"][at + s1 - s0])
    tok = out["row_tok"][lo:hi].astype(np.int64) & 0xFFFF
    rank = out["row_code"][lo:hi].astype(np.int64) & 15  # sorted-rank lane
    srt = np.argsort(np.where(ids >= 0, ids, np.iinfo(np.int32).max), axis=1)
    res[tok, srt[tok, rank]] = True
    return res


def _pair_counts(ids, mask, E):
    """Counts of expert pairs (lo * E + hi) among the masked lanes."""
    K = ids.shape[1]
    keys = []
    for i in range(K):
        for j in range(i + 1, K):
            m = mask[:, i] & mask[:, j]
            a, b = ids[m, i].astype(np.int64), ids[m, j].astype(np.int64)
            keys.append(np.minimum(a, b) * E + np.maximum(a, b))
    keys = np.concatenate(keys) if keys else np.zeros(0, np.int64)
    return np.bincount(keys, minlength=E * E)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--routing", required=True, help="npz with indices")
    ap.add_argument("--experts", type=int, required=True, help="E")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument(
        "--pairs", type=int, default=0, help="list size (default 2 x E)"
    )
    ap.add_argument("--step", type=int, default=50, help="pairs per round")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args(argv)
    P = _planner()
    E = a.experts
    idx = np.load(a.routing, allow_pickle=True)["indices"]
    if idx.ndim == 3:
        idx = idx[a.layer]
    idx = np.ascontiguousarray(idx.astype(np.int32))
    if idx.max() >= E:
        raise SystemExit(f"routing has expert id {idx.max()} >= E={E}")
    half = idx.shape[0] // 2
    fit, held = idx[:half], idx[half:]
    target = a.pairs or 2 * E
    groups, have = [], set()

    def ratio(ids):
        return _plan(P, groups, ids, E, a.threads)[2] / max(1, (ids >= 0).sum())

    while len(groups) < target:
        mask = (
            _residual_lanes(P, groups, fit, E, a.threads)
            if groups
            else fit >= 0
        )
        cnt = _pair_counts(fit, mask, E)
        order = np.argsort(-cnt, kind="stable")
        quota, added = min(a.step, target - len(groups)), 0
        for key in order:
            if cnt[key] == 0 or added == quota:
                break
            pair = (int(key // E), int(key % E))
            if pair not in have:
                groups.append(list(pair))
                have.add(pair)
                added += 1
        if added == 0:
            break
        print(
            f"pairs {len(groups):5d}: rows / lanes fit {ratio(fit):.3f} "
            f"held-out {ratio(held):.3f}",
            flush=True,
        )
    # Priority: the planner takes a token's pairs in list order.  Pairs whose
    # experts appear in few other pairs go first, so they are not blocked by
    # a pair that could have been replaced by another one.
    deg = np.bincount(np.asarray(groups).ravel(), minlength=E)
    full = _pair_counts(fit, fit >= 0, E)
    groups.sort(key=lambda p: (deg[p[0]] + deg[p[1]], -full[p[0] * E + p[1]]))
    print(
        f"final order: rows / lanes fit {ratio(fit):.3f} held-out {ratio(held):.3f}"
    )
    P.save_list(a.out, groups)
    print(f"wrote {len(groups)} pairs to {a.out}")


if __name__ == "__main__":
    main()
