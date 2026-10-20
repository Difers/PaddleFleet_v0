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

# Launch the MoE dispatcher benchmark across <nnodes> nodes, sweeping the
# routing peak-to-average (MaxVio).  One launch runs every dispatcher x every
# MaxVio and writes the result table.
#
#   BENCH_NODES="ip0 ip1 ..." bash scripts/run_dispatcher_bench.sh <nnodes> [out.json]
#
# Environment:
#   BENCH_NODES  space-separated node IPs, first = master (required for nnodes>1)
#   PY           python with paddle + paddlefleet + paddlefleet_ops  (default: python)
#   BENCH_ARGS   benchmark args (default: MaxVio 0..4 step 0.5, 3 dispatchers)
#   MODEL        optional MoE shape preset (ernie5.5-flash | deepseek-v3.1 |
#                qwen3-30b-a3b | qwen3-235b-a22b); sets H/I_moe/E/topk
#   PORT         master port (default 36997)
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
BDIR=$(cd "$HERE/.." && pwd)                 # the dispatcher_benchmark dir
BENCH="$BDIR/moe_dispatcher_bench.py"
NN=${1:?nnodes}
OUT=$(realpath -m "${2:-$HERE/results/ep$((NN * 8)).json}")
PY=${PY:-python}
PORT=${PORT:-36997}
BENCH_NODES=${BENCH_NODES:-$(hostname -I | awk '{print $1}')}
BENCH_ARGS=${BENCH_ARGS:---maxvio 0,0.5,1,1.5,2,2.5,3,3.5,4 --warmup 3 --iters 10 --ring-gpus-per-node 8 --dispatchers allgather,deepep,ringmoe}
# MoE layer shape (H, latent, I_moe, E, topk); empty MODEL keeps the bench default.
# Only ernie5.5-flash has the latent projection; the others run experts at H.
case "${MODEL:-}" in
  "")              MODEL_ARGS="" ;;
  ernie5.5-flash)  MODEL_ARGS="--hidden-size 4096 --moe-latent-size 2048 --moe-intermediate-size 2048 --n-routed-experts 512 --topk 10" ;;
  deepseek-v3.1)   MODEL_ARGS="--hidden-size 7168 --moe-latent-size 0 --moe-intermediate-size 2048 --n-routed-experts 256 --topk 8" ;;
  qwen3-30b-a3b)   MODEL_ARGS="--hidden-size 2048 --moe-latent-size 0 --moe-intermediate-size 768 --n-routed-experts 128 --topk 8" ;;
  qwen3-235b-a22b) MODEL_ARGS="--hidden-size 4096 --moe-latent-size 0 --moe-intermediate-size 1536 --n-routed-experts 128 --topk 8" ;;
  *) echo "error: unknown MODEL='$MODEL' (ernie5.5-flash|deepseek-v3.1|qwen3-30b-a3b|qwen3-235b-a22b)" >&2; exit 2 ;;
esac
read -r -a NODES <<< "$BENCH_NODES"
if [ "${#NODES[@]}" -lt "$NN" ]; then
  echo "error: need $NN nodes in BENCH_NODES but got ${#NODES[@]} (${BENCH_NODES})" >&2
  echo "       (too few would silently under-launch and hang at rendezvous)" >&2
  exit 2
fi
NODE0=${NODES[0]}
LOG=$(dirname "$OUT")/log_$(basename "$OUT" .json); command rm -rf "$LOG"; mkdir -p "$LOG"

# Cluster-injected rank variables override --master/--nnodes/--rank; clear them.
UNSET="PADDLE_TRAINER_ENDPOINTS DISTRIBUTED_TRAINER_ENDPOINTS PADDLE_TRAINER_ID PADDLE_TRAINERS \
PADDLE_TRAINERS_NUM PADDLE_TRAINER_COUNT PADDLE_MASTER PADDLE_NNODES PADDLE_RANK"
# `import routing_inject` needs the benchmark dir on sys.path, so run from there.
# PYTHONPATH is forwarded so ssh'd ranks see the same paddlefleet / paddlefleet_ops.
node_cmd() {
  cat <<EOF
cd "$BDIR" && unset $UNSET && export no_proxy='*' PYTHONPATH="${PYTHONPATH:-}" \
 && $PY -m paddle.distributed.launch --master $NODE0:$PORT --nnodes $NN --rank $1 \
    --devices 0,1,2,3,4,5,6,7 --log_dir $LOG/w$1 \
    "$BENCH" $BENCH_ARGS $MODEL_ARGS > $LOG/node$1.log 2>&1
EOF
}
rank=1
for ip in "${NODES[@]:1:$((NN - 1))}"; do
  ssh -o BatchMode=yes "$ip" "nohup bash -c \"$(node_cmd $rank)\" >/dev/null 2>&1 &"
  rank=$((rank + 1))
done
bash -c "$(node_cmd 0)"
rc=$?
# pull the result table out of the launcher log for the plot script
grep -E "\[(EP|G)=.*maxvio=[0-9].* rank_maxvio" "$LOG/node0.log" | grep -vE "iter|==" > "${OUT%.json}.tsv" || true
echo "rc=$rc  table=${OUT%.json}.tsv ($(wc -l < "${OUT%.json}.tsv" 2>/dev/null || echo 0) rows)"
echo "plot: $PY $HERE/plot_dispatcher.py ${OUT%.json}.tsv -o ${OUT%.json}.png"
exit $rc
