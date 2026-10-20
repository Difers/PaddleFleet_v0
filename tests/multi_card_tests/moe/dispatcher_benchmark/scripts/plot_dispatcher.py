#!/usr/bin/env python3

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

"""Plot the dispatcher benchmark: step time vs routing peak-to-average (MaxVio),
one line per dispatcher.  Each EP is a row with a Forward, a Backward and a
Total panel side by side.  Reads the .tsv table(s) from run_dispatcher_bench.sh.

    python scripts/plot_dispatcher.py results/ep16.tsv results/ep32.tsv -o fig.png
"""

import argparse
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# "allgather [EP=16] maxvio=1.00 rank_maxvio=0.65  7.87  10.11  18.03  454,342  9.42"
_LINE = re.compile(
    r"^\s*(\w+)\s*\[([^\]]+)\]\s*maxvio=([\d.]+)\s+rank_maxvio=[\d.]+\s+"
    r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d,]+)\s+([\d.]+)"
)
_NICE = {"allgather": "AllGather", "deepep": "DeepEP", "ringmoe": "RingMoE"}
_COLOR = {"AllGather": "#1f77b4", "DeepEP": "#d62728", "RingMoE": "#2ca02c"}
_MARK = {"AllGather": "o", "DeepEP": "s", "RingMoE": "^"}
_METRICS = [
    ("Forward", 0),
    ("Backward", 1),
    ("Total", 2),
]  # index into (fwd, bwd, total)


def parse(paths):
    """-> {ep: {dispatcher: [(maxvio, fwd, bwd, total)]}}"""
    data = {}
    for p in paths:
        for ln in open(p):
            m = _LINE.match(ln)
            if not m:
                continue
            disp, tag, mv = m.group(1), m.group(2), float(m.group(3))
            fwd, bwd, tot = (
                float(m.group(4)),
                float(m.group(5)),
                float(m.group(6)),
            )
            if tag.startswith("EP="):
                ep = int(tag[3:])
            else:  # "G=8 N=2" -> EP = G*N
                g, n = (int(x.split("=")[1]) for x in tag.split())
                ep = g * n
            data.setdefault(ep, {}).setdefault(
                _NICE.get(disp, disp), []
            ).append((mv, fwd, bwd, tot))
    for ep in data:
        for name in data[ep]:
            data[ep][name].sort()
    return data


def _crossover(a, b):
    """First MaxVio where curves a and b (lists of (x, y), same grid) cross,
    by linear interpolation of their difference; None if they never cross."""
    da, db = dict(a), dict(b)
    xs = sorted(set(da) & set(db))
    for x0, x1 in zip(xs, xs[1:]):
        d0, d1 = da[x0] - db[x0], da[x1] - db[x1]
        if d0 == 0:
            return x0, (da[x0] + db[x0]) / 2
        if d0 * d1 < 0:
            t = d0 / (d0 - d1)
            return x0 + t * (x1 - x0), da[x0] + t * (da[x1] - da[x0])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tables", nargs="+")
    ap.add_argument("-o", "--out", default="dispatcher.png")
    ap.add_argument(
        "-t",
        "--title",
        default=None,
        help="shape/model note appended to the figure title, e.g. "
        "'DeepSeek-V3.1 E=256 top-8, 8192 tok/card'",
    )
    args = ap.parse_args()
    data = parse(args.tables)
    eps = sorted(data)
    if not eps:
        raise SystemExit(
            f"no parseable benchmark rows in {args.tables}; "
            "expected lines like 'ringmoe [G=8 N=2] maxvio=... rank_maxvio=...'"
        )
    nrow, ncol = len(eps), len(_METRICS)
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(5.4 * ncol, 4.5 * nrow), squeeze=False
    )
    for r, ep in enumerate(eps):
        for c, (metric, idx) in enumerate(_METRICS):
            ax = axes[r][c]
            curves = {}
            for name, pts in sorted(data[ep].items()):
                xs = [p[0] for p in pts]
                ys = [p[1 + idx] for p in pts]
                curves[name] = list(zip(xs, ys))
                ax.plot(
                    xs,
                    ys,
                    marker=_MARK.get(name, "o"),
                    color=_COLOR.get(name),
                    lw=2,
                    ms=6,
                    label=name,
                )
            if "DeepEP" in curves and "RingMoE" in curves:
                cr = _crossover(curves["DeepEP"], curves["RingMoE"])
                if cr is not None:
                    cx, cy = cr
                    ax.axvline(cx, color="gray", ls=":", lw=1.3)
                    ax.plot([cx], [cy], "k*", ms=11)
                    ax.annotate(
                        f"RingMoE = DeepEP\nMaxVio ≈ {cx:.2f}",
                        (cx, cy),
                        textcoords="offset points",
                        xytext=(10, 10),
                        fontsize=9,
                        bbox={
                            "boxstyle": "round,pad=0.3",
                            "fc": "white",
                            "ec": "gray",
                            "alpha": 0.85,
                        },
                    )
            ax.set_title(f"EP={ep}  {metric}")
            ax.set_xlabel("MaxVio (expert-load peak-to-average)")
            ax.grid(True, ls="--", alpha=0.4)
            if c == 0:
                ax.set_ylabel("time (ms, lower = better)")
    axes[0][0].legend(frameon=False, fontsize=10)
    title = "MoE dispatcher vs routing imbalance"
    if args.title:
        title += f" ({args.title})"
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.out, dpi=130)
    print("saved", args.out)


if __name__ == "__main__":
    main()
