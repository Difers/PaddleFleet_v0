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

## 3. RingMoE: which optimization when

RingMoE (`moe_token_dispatcher_type='ringmoe'`) shards every expert along the
intermediate dim: each card holds all E experts with `I / EP` of their
intermediate size, and each ring round runs one node's gathered tokens. What
is on by default and what has to be switched on:

| Optimization | How it is enabled | When it applies |
|---|---|---|
| Ring pipeline: intra-node AllGather → experts → ReduceScatter per round, single-peer inter-node hops overlapped with the GEMMs, remote-first round order, pre-gathered routing for N ≥ 3 | `moe_token_dispatcher_type='ringmoe'` | ≥ 2 nodes (N ≥ 2) of G ≥ 2 GPUs; equal tokens per rank. Flat in MaxVio: prefer it over DeepEP above the crossover MaxVio of section 2 |
| Shared-expert backward and router backward overlapped with the last inter-node hop | always on with `ringmoe` | — |
| BF16 wgrad on 2-CTA 256×256 tiles (sonic) | automatic | gate-up wgrad with `M >= 2048` and `N = 2·I/EP > 128` |
| Routing-metadata expert offsets as a block-wide scan (sonic) | automatic | every Sonic-MoE call; matters when many experts sit on one card (RingMoE: all E) |
| **Grouped down projection**: per token, lane-disjoint expert pairs of a fixed list share one deeper-K GEMM row (fewer rows written and summed); the plan is built per rank and carried with the routing; the up projection's input gradient reuses it | `moe_ring_grouped_down_proj_list=<list.npz>` (learn it with `make_group_list.py`) | BF16 only (no `fp8`); Sonic-MoE with `grouped_plan_pieces`; `I/EP` a multiple of 16. Worth it when `I/EP <= 128` (e.g. intermediate 2048 at EP ≥ 16); at `I/EP >= 256` it does not pay off, leave it off. The input-gradient part turns on by itself only for `2·I/EP <= 256` |

Notes on the grouped down projection:
- The list must come from the model's **own recorded routing** (it relies on
  experts that are routed together); relearn it when the model or its routing
  changes. Synthetic MaxVio routing has no such co-occurrence.
- It keeps a per-shape work buffer of about `T·K·H·2` bytes (T = tokens per
  round) for the whole process; the benchmark's peak memory includes it.
- Unsupported shapes / builds keep the default down projection and log a
  warning; results match the default path up to bf16 summation order.

### Learn a list and measure it

```bash
# 1. learn the pair list (CPU only)
python scripts/make_group_list.py --routing routing.npz --experts 512 -o list.npz

# 2. ringmoe vs ringmoe-gdp on the same replayed routing (EP32 = 4 nodes)
BENCH_NODES="ip0 ip1 ip2 ip3" PY=... PYTHONPATH=... \
ROUTING=routing.npz LIST=list.npz MODEL=ernie5.5-flash \
  bash scripts/run_ringmoe_bench.sh 4 scripts/results/ringmoe_ep32
```

`run_ringmoe_bench.sh` learns the list itself when `LIST` is unset, runs the
`ringmoe` and `ringmoe-gdp` dispatchers through `run_dispatcher_bench.sh` with
`--routing-npz` (each rank replays its slice of the recorded routing), and
prints both rows plus their difference. The two rows come from separate runs,
so differences of about 0.1 ms or less are noise.

The routing npz holds `indices` (expert ids, -1 = dropped) and `weights`, of
shape `[T, K]` or `[layers, T, K]` (the first layer is replayed); record it from
the router's top-k output during training. `moe_dispatcher_bench.py` also takes
`--routing-npz` / `--grouped-down-proj-list` directly.
