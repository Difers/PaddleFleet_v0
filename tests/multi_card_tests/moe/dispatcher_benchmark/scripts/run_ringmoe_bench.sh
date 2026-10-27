#!/bin/bash
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

# RingMoE with and without the grouped down projection, on a recorded routing.
#
#   BENCH_NODES="ip0 ip1 ip2 ip3" PY=... PYTHONPATH=... \
#   ROUTING=routing.npz [LIST=list.npz] [MODEL=ernie5.5-flash] \
#     bash scripts/run_ringmoe_bench.sh <nnodes> [out-stem]
#
# Runs the dispatchers `ringmoe` and `ringmoe-gdp` (RingMoE with
# `moe_ring_grouped_down_proj_list`) on the same replayed routing and prints
# both rows and their difference.  See README.md "RingMoE: which optimization
# when" for when each optimization applies.
#
# Environment (plus everything run_dispatcher_bench.sh takes):
#   ROUTING  required: npz with `indices` / `weights` ([T, K] or
#            [layers, T, K]) recorded from the model's router; the grouped
#            down projection only saves anything with real co-occurring
#            routing, so do not use synthetic routing here.
#   LIST     group list; if unset it is learned from ROUTING with
#            make_group_list.py into <out-stem>_list.npz.
#   EXPERTS  expert count for learning the list (default: from MODEL).
#   ITERS / WARMUP   timed / warmup iterations (default 20 / 5).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
NN=${1:?nnodes}
STEM=$(realpath -m "${2:-$HERE/results/ringmoe_ep$((NN * 8))}")
PY=${PY:-python}
: "${ROUTING:?set ROUTING=<recorded routing npz>}"
case "${MODEL:-ernie5.5-flash}" in
  ernie5.5-flash)  E_DEF=512 ;;
  deepseek-v3.1)   E_DEF=256 ;;
  qwen3-30b-a3b|qwen3-235b-a22b) E_DEF=128 ;;
  *) E_DEF= ;;
esac
EXPERTS=${EXPERTS:-$E_DEF}
mkdir -p "$(dirname "$STEM")"
if [ -z "${LIST:-}" ]; then
  : "${EXPERTS:?set EXPERTS=<expert count> to learn the group list}"
  LIST=${STEM}_list.npz
  echo "learning the group list from $ROUTING -> $LIST"
  "$PY" "$HERE/make_group_list.py" --routing "$ROUTING" --experts "$EXPERTS" -o "$LIST" || exit 1
fi
export MODEL=${MODEL:-ernie5.5-flash}
export BENCH_ARGS="--dispatchers ringmoe,ringmoe-gdp --routing-npz $(realpath "$ROUTING") \
--grouped-down-proj-list $(realpath "$LIST") --warmup ${WARMUP:-5} --iters ${ITERS:-20} \
--ring-gpus-per-node 8"
bash "$HERE/run_dispatcher_bench.sh" "$NN" "$STEM.json"
rc=$?
TSV=$STEM.tsv
# rows: name ... fwd bwd total tok/s peak (last five columns)
awk '
  /^ringmoe-gdp / { g_f=$(NF-4); g_b=$(NF-3); g_t=$(NF-2); have_g=1; next }
  /^ringmoe /     { r_f=$(NF-4); r_b=$(NF-3); r_t=$(NF-2); have_r=1 }
  END {
    if (have_r && have_g)
      printf "ringmoe-gdp - ringmoe (ms): fwd %+.3f  bwd %+.3f  total %+.3f\n", g_f-r_f, g_b-r_b, g_t-r_t
    else
      print "ringmoe-gdp row missing (skipped? see the launcher log)"
  }' "$TSV"
echo "note: two separate runs; differences below ~0.1 ms are within run-to-run noise"
exit $rc
