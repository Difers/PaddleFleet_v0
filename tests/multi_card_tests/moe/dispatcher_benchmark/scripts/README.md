# Dispatcher benchmark scripts

Compare the MoE token dispatchers — **AllGather**, **DeepEP**, **RingMoE** — as
the routing's expert-load peak-to-average (**MaxVio**) varies, at a fixed
expert-parallel size (EP). One launch sweeps every dispatcher × every MaxVio and
writes one result table; the plot script turns the table(s) into the figure.

Default shape (from `moe_dispatcher_bench.py`, equivalent to
`MODEL=ernie5.5-flash`): **E=512 experts, top-10, 8192 tokens/card,
hidden=4096 (latent 2048), moe_intermediate=2048**, forward + backward. Other
shapes are selected with the `MODEL` presets below; when comparing presets,
label the figure with `plot_dispatcher.py -t/--title`.

## 1. Run the sweep (needs N nodes × 8 GPU)

`run_dispatcher_bench.sh <nnodes> [out-stem]` launches the sweep across
`<nnodes>` nodes (EP = `nnodes * 8`), parses the launcher log, and writes
`<out-stem>.tsv`.

```bash
# EP16 = 2 nodes; EP32 = 4 nodes (first IP is the master)
BENCH_NODES="ip0 ip1" \
PY=/path/to/python \
PYTHONPATH=/path/to/paddlefleet/src:/path/to/paddlefleet_ops/src \
  bash scripts/run_dispatcher_bench.sh 2 scripts/results/ep16
```

Environment / arguments:
- `<nnodes>` (positional): node count; EP = `nnodes * 8`.
- second positional: output stem (default `scripts/results/ep<EP>`); the table
  is written to `<stem>.tsv`, raw logs to `<stem-dir>/log_<stem>/`.
- `BENCH_NODES`: space-separated node IPs, first = master (default: this host).
- `PY`: python with `paddle` + `paddlefleet` + `paddlefleet_ops` (default
  `python`).
- `PYTHONPATH`: forwarded to every rank (so ssh'd nodes see the same
  paddlefleet / paddlefleet_ops); set it to the `src` dirs of both.
- `BENCH_ARGS`: benchmark args. Default is a 9-point MaxVio grid:
  `--maxvio 0,0.5,1,1.5,2,2.5,3,3.5,4 --warmup 3 --iters 10
  --ring-gpus-per-node 8 --dispatchers allgather,deepep,ringmoe`.
- `MODEL`: optional MoE shape preset — `ernie5.5-flash` (default shape),
  `deepseek-v3.1`, `qwen3-30b-a3b`, `qwen3-235b-a22b` — sets H / latent /
  I_moe / E / topk. Only `ernie5.5-flash` has the latent projection; the other
  presets pass `--moe-latent-size 0` so experts run at the model's hidden size.
  Omit `MODEL` (or pass the individual `--hidden-size` / ... args via
  `BENCH_ARGS`) to use the bench default.

```bash
# sweep a specific model shape
MODEL=deepseek-v3.1 BENCH_NODES="ip0 ip1" PY=... PYTHONPATH=... \
  bash scripts/run_dispatcher_bench.sh 2 scripts/results/dsv31_ep16
```

Example — one EP16 table and one EP32 table:

```bash
BENCH_NODES="ip0 ip1"         PY=... PYTHONPATH=... bash scripts/run_dispatcher_bench.sh 2 scripts/results/ep16
BENCH_NODES="ip0 ip1 ip2 ip3" PY=... PYTHONPATH=... bash scripts/run_dispatcher_bench.sh 4 scripts/results/ep32
```

## 2. Plot (no GPU needed)

Turn the `.tsv` table(s) you just wrote into the figure:

```bash
python scripts/plot_dispatcher.py \
  scripts/results/ep16.tsv scripts/results/ep32.tsv \
  -o scripts/results/dispatcher.png
```

One row per EP, three columns (**Forward**, **Backward**, **Total**), one line
per dispatcher. On each panel the RingMoE×DeepEP crossover MaxVio is marked
(black star + dotted line + `RingMoE = DeepEP  MaxVio ≈ x`). `matplotlib` is the
only dependency; pass any number of `.tsv` tables.

## Table format

Each `.tsv` row (parsed by `plot_dispatcher.py`):

```
<dispatcher> [<EP tag>] maxvio=<x> rank_maxvio=<y>  <fwd_ms>  <bwd_ms>  <total_ms>  <tok/s>  <peak_gib>
```

`<EP tag>` is `EP=16` for AllGather/DeepEP or `G=8 N=2` for RingMoE (EP = G·N).
The plot reads `maxvio` (x) and the forward / backward / total milliseconds (y).

## What the figure shows

AllGather and RingMoE are flat in MaxVio (they move every token regardless of
balance); DeepEP's all-to-all cost grows with imbalance. So below the marked
crossover MaxVio DeepEP is cheapest, and above it RingMoE wins; the crossover
sits at a low MaxVio and moves higher as EP grows. The plot marks the exact
crossover per panel for whatever tables you pass.
