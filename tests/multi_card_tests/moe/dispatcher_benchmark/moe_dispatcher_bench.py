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
"""Forward/backward time and memory of one MoELayer under different dispatchers.

Builds the same MoELayer once per token dispatcher, drives it with synthetic
hidden states and a fixed upstream gradient, and prints, per dispatcher, the
forward time, backward time and peak allocated memory of every timed
iteration followed by their medians. Each per-iteration value is the maximum
over ranks, i.e. the slowest rank and the highest peak. By default the real
router's routing compares allgather and ringmoe; a ``--maxvio`` sweep also
adds deepep.

The default MoE block has 512 routed experts, top-10, hidden 4096, latent
2048, intermediate 2048, one shared expert and 8192 tokens per card. All
dispatchers run fused SonicMoE experts; deepep is reported as skipped when
DeepEP is not in the build.

RingMoE needs an inter-node level (N > 1). On a single node the intra-node
GPU count G is patched to ``world // 2`` so the two-level ring still runs; this
exercises the code path but does not reflect multi-node ring performance. Use
``--ring-gpus-per-node`` on a multi-node launch to set the real G.

Load imbalance: ``--maxvio`` sweeps the router MaxVio. The real router still
runs every step, but its routing decision is replaced by a generated one that
hits each target (``routing_inject.py``) and is identical for every
dispatcher; ``--hot-placement`` picks where the hot experts sit relative to
the per-rank expert blocks.

A manual benchmark, not collected by CI (no ``test_*.py`` name, no
assertions). Run it on the target nodes (all GPUs of the launched nodes form
the EP group), e.g. EP=16 on two nodes, run on each with ``--rank`` 0 and 1:

  python -m paddle.distributed.launch --master <node0_ip>:<port> --nnodes 2 \\
      --rank <0|1> --gpus 0,1,2,3,4,5,6,7 \\
      tests/multi_card_tests/moe/dispatcher_benchmark/moe_dispatcher_bench.py

Rank 0 prints the per-iteration table and the medians of every dispatcher.
Correctness of the ring is covered by
``tests/multi_card_tests/moe/test_ring_dispatcher_two_level.py``.
"""

from __future__ import annotations

import argparse
import gc
import os
import statistics
import sys
from dataclasses import dataclass, field
from unittest import mock

import numpy as np
import paddle
import paddle.distributed as dist
import paddle.nn.functional as F
from paddle.distributed import fleet

# Make the sibling routing_inject module importable.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_GIB = 1024.0**3
_ORDER = ["sharding", "moe_sharding", "pp", "sep", "cp", "dp", "ep", "mp"]
ALL_DISPATCHERS = ("allgather", "ringmoe", "deepep")


@dataclass
class BenchConfig:
    """MoELayer shape + measurement knobs shared by every dispatcher run."""

    tokens: int = 8192  # tokens per card (B=1, S=tokens)
    hidden_size: int = 4096
    moe_intermediate_size: int = 2048
    # Experts run in this projected width; None runs them at hidden_size.
    moe_latent_size: int | None = 2048
    n_routed_experts: int = 512
    num_experts_per_tok: int = 10
    n_shared_experts: int = 1
    bwd: bool = True
    warmup: int = 10
    iters: int = 20
    # "pipeline": no per-iteration barrier, the host runs ahead as in
    # training. "barrier": ranks are aligned before fwd and bwd, and each
    # phase ends with the slowest rank, exposing host launch overhead.
    timing: str = "pipeline"
    # Intra-node GPU count G for ringmoe; None = world // 2 on one node,
    # else the dispatcher default.
    ring_gpus_per_node: int | None = None
    seed: int = 1234
    # MaxVio targets to sweep (see routing_inject.py). Empty keeps the real
    # router's routing, giving one cell per dispatcher.
    maxvio_targets: list = field(default_factory=list)
    hot_placement: str = "clustered"
    # Which MaxVio the targets refer to: "per_expert" or "per_rank" (the
    # load of each rank's expert block, i.e. the per-card imbalance).
    maxvio_metric: str = "per_expert"


@dataclass
class BenchResult:
    """One (dispatcher, routing) cell.

    Per-iteration lists hold the maximum over ranks: times in ms, peak
    allocated memory in GiB.
    """

    dispatcher: str
    status: str = "ok"  # "ok" or "skipped: <reason>"
    topology: str = ""
    target_maxvio: float | None = None  # None = real router routing
    realized_maxvio: float | None = None  # per-expert, EP-wide histogram
    per_rank_maxvio: float | None = None  # under the E // EP block mapping
    sigma: float | None = None
    fwd_ms: list = field(default_factory=list)
    bwd_ms: list = field(default_factory=list)
    peak_gib: list = field(default_factory=list)

    @property
    def total_ms(self) -> list:
        return [f + b for f, b in zip(self.fwd_ms, self.bwd_ms)]

    def median(self, name: str) -> float:
        values = getattr(self, name)
        return statistics.median(values) if values else 0.0


def init_ep_fleet():
    """Return process groups with EP spanning the whole world.

    Reuses an already initialized fleet topology.
    """
    from paddlefleet.process_groups_config import ProcessGroupCollection
    from paddlefleet.training.initialize import initialize_fleet

    world = dist.get_world_size()
    if getattr(fleet.fleet, "_hcg", None) is None:
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": 1,
            "pp_degree": 1,
            "sharding_degree": world,
            "sep_degree": 1,
            "cp_degree": 1,
            "ep_degree": world,
            "moe_sharding_degree": 1,
            "order": _ORDER,
        }
        initialize_fleet(strategy=strategy)
    return ProcessGroupCollection.use_mpu_process_groups()


def _small_init(tensor):
    paddle.nn.initializer.Uniform(-0.01, 0.01)(tensor)


def build_transformer_config(dispatcher: str, cfg: BenchConfig):
    """TransformerConfig for one MoELayer driven by ``dispatcher``."""
    from paddlefleet.transformer.transformer_config import TransformerConfig

    return TransformerConfig(
        hidden_size=cfg.hidden_size,
        num_attention_heads=4,
        n_routed_experts=cfg.n_routed_experts,
        num_experts_per_tok=cfg.num_experts_per_tok,
        n_shared_experts=cfg.n_shared_experts,
        moe_intermediate_size=cfg.moe_intermediate_size,
        moe_latent_size=cfg.moe_latent_size,
        use_cpu_initialization=False,
        tensor_model_parallel_size=1,
        expert_model_parallel_size=dist.get_world_size(),
        sequence_parallel=False,
        bf16=True,
        params_dtype=paddle.bfloat16,
        gated_linear_unit=True,
        hidden_act=F.silu,
        bias_activation_fusion=True,
        # Router: sigmoid scores, aux-free top-k.
        scoring_func="sigmoid",
        topk_method="noaux_tc",
        moe_router_load_balancing_type="seq_aux_loss",
        router_aux_loss_coef=0.0,
        n_group=1,
        topk_group=1,
        moe_shared_expert_overlap=cfg.n_shared_experts > 0,
        moe_token_dispatcher_type=dispatcher,
        moe_expert_fusion=True,
        moe_use_fusion_node=True,
        using_sonic_moe=True,
        init_method=_small_init,
        output_layer_init_method=_small_init,
    )


def ring_gpus_per_node(cfg: BenchConfig) -> int:
    """Intra-node GPU count G for the ring.

    Uses the configured value, else ``world // 2`` on a single node and the
    dispatcher default otherwise.
    """
    if cfg.ring_gpus_per_node is not None:
        return cfg.ring_gpus_per_node
    from paddlefleet.transformer.moe import token_dispatcher as td

    world = dist.get_world_size()
    return (
        world // 2
        if world <= td._RING_GPUS_PER_NODE
        else td._RING_GPUS_PER_NODE
    )


def unavailable_reason(dispatcher: str, cfg: BenchConfig) -> str | None:
    """Reason ``dispatcher`` cannot run here, or None if it can."""
    import paddlefleet_ops

    from paddlefleet.transformer.moe import fused_a2a

    if dispatcher not in ALL_DISPATCHERS:
        return f"unknown dispatcher {dispatcher!r}"
    if not paddlefleet_ops.is_sonic_moe_available():
        return "Sonic-MoE not available"
    if dispatcher == "deepep" and not fused_a2a.HAVE_DEEP_EP:
        return "DeepEP not available"
    if dispatcher == "ringmoe":
        gpus_per_node = ring_gpus_per_node(cfg)
        if gpus_per_node < 2 or dist.get_world_size() // gpus_per_node < 2:
            return "ringmoe needs G > 1 and N > 1"
    return None


def build_moe_layer(dispatcher: str, cfg: BenchConfig, pg_collection):
    """Build a bf16 MoELayer in train mode.

    Gradients accumulate into the parameters' own bf16 ``grad``; no fp32
    ``main_grad`` buffers are attached, so the reported memory is the layer's
    weights, gradients and activations only.
    """
    from paddlefleet.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from paddlefleet.tensor_parallel.random import (
        model_parallel_cuda_manual_seed,
    )
    from paddlefleet.transformer.moe import token_dispatcher as td
    from paddlefleet.transformer.moe.moe_layer import MoELayer

    # Same seed for every dispatcher, so all runs share one initialization.
    np.random.seed(cfg.seed)
    paddle.seed(cfg.seed)
    model_parallel_cuda_manual_seed(cfg.seed)
    transformer_config = build_transformer_config(dispatcher, cfg)
    spec = get_gpt_layer_local_spec(
        transformer_config, num_experts=cfg.n_routed_experts
    )
    sublayers = spec.sublayers_spec.mlp.extra_kwargs["sublayers"]
    # The ring dispatcher builds its sub-groups from this module global at
    # construction time; other dispatchers ignore it.
    with mock.patch.object(td, "_RING_GPUS_PER_NODE", ring_gpus_per_node(cfg)):
        layer = MoELayer(transformer_config, sublayers, pg_collection)
    layer.train()
    return layer


def _topology(layer) -> str:
    dispatcher = getattr(layer, "token_dispatcher", None)
    if (
        dispatcher is not None
        and hasattr(dispatcher, "G")
        and hasattr(dispatcher, "N")
    ):
        return f"G={dispatcher.G} N={dispatcher.N}"
    return f"EP={dist.get_world_size()}"


def _barrier():
    dist.all_reduce(paddle.zeros([1], dtype="float32"))


def _max_over_ranks(values: list) -> list:
    buf = paddle.to_tensor(values, dtype="float64")
    dist.all_reduce(buf, op=dist.ReduceOp.MAX)
    return buf.tolist()


def _make_input(cfg: BenchConfig):
    hidden_states = (
        paddle.randn([1, cfg.tokens, cfg.hidden_size], dtype="bfloat16") * 0.1
    )
    hidden_states.stop_gradient = not cfg.bwd
    # Non-leaf input, like a real norm output feeding the MoE block.
    return hidden_states.clone() if cfg.bwd else hidden_states


def _time_layer(layer, cfg: BenchConfig):
    """Time every iteration of ``layer`` on this rank.

    Returns per-iteration forward and backward times in ms (CUDA events) and
    the peak allocated memory of each iteration in GiB. The allocator tracks
    the peak on the host as kernels are enqueued, so reading it needs no
    device synchronization.
    """
    Event = paddle.device.cuda.Event

    def fwd(hidden_states):
        out = layer(hidden_states)
        return out[0] if isinstance(out, (tuple, list)) else out

    out_grad = None
    for _ in range(cfg.warmup):  # absorbs JIT, autotuning and allocation
        out = fwd(_make_input(cfg))
        if cfg.bwd:
            if out_grad is None:
                out_grad = paddle.randn(out.shape, dtype=out.dtype) * 0.02
            out.backward(out_grad)
            layer.clear_gradients()
    out = None
    gc.collect()
    # Align the ranks so the first timed forward does not include waiting
    # for the last rank to leave warmup.
    _barrier()
    paddle.device.synchronize()

    events, peaks = [], []
    for _ in range(cfg.iters):
        # Reset before the input is made so the peak covers the whole step.
        paddle.device.cuda.reset_max_memory_allocated()
        hidden_states = _make_input(cfg)
        iter_events = [Event(enable_timing=True) for _ in range(3)]
        if cfg.timing == "barrier":
            _barrier()
        iter_events[0].record()
        out = fwd(hidden_states)
        if cfg.timing == "barrier":
            _barrier()
        iter_events[1].record()
        if cfg.bwd:
            if out_grad is None:
                out_grad = paddle.randn(out.shape, dtype=out.dtype) * 0.02
            out.backward(out_grad)
            if cfg.timing == "barrier":
                _barrier()
        iter_events[2].record()
        events.append(iter_events)
        if cfg.bwd:
            layer.clear_gradients()
        out = None
        peaks.append(paddle.device.cuda.max_memory_allocated() / _GIB)
        if cfg.timing == "barrier":
            paddle.device.synchronize()
    paddle.device.synchronize()
    fwd_times = [recorded[0].elapsed_time(recorded[1]) for recorded in events]
    bwd_times = [
        recorded[1].elapsed_time(recorded[2]) if cfg.bwd else 0.0
        for recorded in events
    ]
    return fwd_times, bwd_times, peaks


def _measure_cell(layer, cfg: BenchConfig, base: BenchResult):
    """Time one (dispatcher, routing) cell and fill its results into ``base``."""
    gc.collect()
    paddle.device.synchronize()
    fwd_times, bwd_times, peaks = _time_layer(layer, cfg)
    # Keep the slowest rank and the highest peak: they bound a real step.
    base.fwd_ms = _max_over_ranks(fwd_times)
    base.bwd_ms = _max_over_ranks(bwd_times)
    base.peak_gib = _max_over_ranks(peaks)
    return base


def run_dispatcher(dispatcher: str, cfg: BenchConfig, pg_collection):
    """Benchmark one dispatcher over every MaxVio target.

    The layer is built once and reused for the whole sweep, so no cell sees
    another layer's not-yet-freed weights. Unavailable dispatchers are
    reported as skipped (decided identically on every rank). Runtime errors
    are not caught: swallowing one on a single rank would hang the others in
    the next collective.
    """
    reason = unavailable_reason(dispatcher, cfg)
    if reason is not None:
        return [BenchResult(dispatcher, status=f"skipped: {reason}")]
    gc.collect()
    paddle.device.cuda.empty_cache()
    layer = build_moe_layer(dispatcher, cfg, pg_collection)
    layer_topology = _topology(layer)
    results = []
    if not cfg.maxvio_targets:
        results.append(
            _measure_cell(
                layer, cfg, BenchResult(dispatcher, topology=layer_topology)
            )
        )
    else:
        import routing_inject as ri

        ep_group = pg_collection.ep
        gate_signature = ri.capture_gate_signature(layer, _make_input(cfg))
        for target in cfg.maxvio_targets:
            sigma = ri.solve_sigma_for_maxvio(
                target,
                gate_signature["num_tokens"],
                gate_signature["topk"],
                gate_signature["num_experts"],
                ep_group.nranks,
                cfg.seed,
                cfg.hot_placement,
                metric=cfg.maxvio_metric,
            )
            routing = ri.build_routing(
                gate_signature,
                sigma,
                ep_group,
                ep_group.rank,
                cfg.seed,
                cfg.hot_placement,
            )
            base = BenchResult(
                dispatcher,
                topology=layer_topology,
                target_maxvio=float(target),
                realized_maxvio=routing["per_expert_maxvio"],
                per_rank_maxvio=routing["per_rank_maxvio"],
                sigma=routing["sigma"],
            )
            with ri.injected_gate(layer, gate_signature, routing):
                results.append(_measure_cell(layer, cfg, base))
    del layer
    gc.collect()
    paddle.device.cuda.empty_cache()
    return results


def default_dispatchers(cfg: BenchConfig) -> tuple:
    """Dispatchers compared by default.

    On the real router's routing, the intermediate-sharded dispatchers
    allgather and ringmoe; with a MaxVio sweep also deepep, whose per-device
    expert slices make it sensitive to load imbalance.
    """
    if cfg.maxvio_targets:
        return ALL_DISPATCHERS
    return ("allgather", "ringmoe")


def run_all(dispatchers, cfg: BenchConfig):
    """Benchmark every dispatcher in ``dispatchers``; return all cells."""
    pg_collection = init_ep_fleet()
    return [
        result
        for dispatcher in dispatchers
        for result in run_dispatcher(dispatcher, cfg, pg_collection)
    ]


def _cell_name(result: BenchResult) -> str:
    name = f"{result.dispatcher} [{result.topology}]"
    if result.target_maxvio is not None:
        name += (
            f" maxvio={result.realized_maxvio:.2f}"
            f" rank_maxvio={result.per_rank_maxvio:.2f}"
        )
    return name


def format_report(results, cfg: BenchConfig) -> str:
    """Per-iteration fwd / bwd / total time and peak memory, then medians."""
    lines = [
        f"MoELayer dispatcher benchmark: EP={dist.get_world_size()} "
        f"tokens/card={cfg.tokens} hidden={cfg.hidden_size} "
        f"latent={cfg.moe_latent_size} moe_inter={cfg.moe_intermediate_size} "
        f"experts={cfg.n_routed_experts} topk={cfg.num_experts_per_tok} "
        f"shared={cfg.n_shared_experts} bwd={cfg.bwd} timing={cfg.timing} "
        f"warmup={cfg.warmup} iters={cfg.iters} (max over ranks)"
    ]
    column_header = (
        f"{'iter':>6} {'fwd ms':>9} {'bwd ms':>9} {'total ms':>9} "
        f"{'peak GiB':>9}"
    )
    ok = [result for result in results if result.status == "ok"]
    for result in results:
        lines.append("")
        if result.status != "ok":
            lines.append(f"{result.dispatcher}: {result.status}")
            continue
        lines += [f"== {_cell_name(result)}", column_header]
        for i, (fwd, bwd, peak) in enumerate(
            zip(result.fwd_ms, result.bwd_ms, result.peak_gib)
        ):
            lines.append(
                f"{i:>6} {fwd:>9.3f} {bwd:>9.3f} {fwd + bwd:>9.3f} {peak:>9.2f}"
            )
    summary_header = (
        f"{'dispatcher':<44} {'fwd ms':>9} {'bwd ms':>9} {'total ms':>9} "
        f"{'tok/s/card':>12} {'peak GiB':>9}"
    )
    lines += ["", "Median over iterations:", summary_header]
    lines.append("-" * len(summary_header))
    for result in ok:
        total = result.median("total_ms")
        tokens_per_sec = cfg.tokens / (total / 1e3) if total else 0.0
        lines.append(
            f"{_cell_name(result):<44} {result.median('fwd_ms'):>9.3f} "
            f"{result.median('bwd_ms'):>9.3f} {total:>9.3f} "
            f"{tokens_per_sec:>12,.0f} {result.median('peak_gib'):>9.2f}"
        )
    return "\n".join(lines)


def _parse_args(argv=None):
    default_cfg = BenchConfig()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--dispatchers",
        default="",
        help=f"comma-separated subset of {','.join(ALL_DISPATCHERS)} "
        "(default: allgather,ringmoe on real routing, all with --maxvio)",
    )
    parser.add_argument("--tokens", type=int, default=default_cfg.tokens)
    parser.add_argument(
        "--hidden-size", type=int, default=default_cfg.hidden_size
    )
    parser.add_argument(
        "--moe-intermediate-size",
        type=int,
        default=default_cfg.moe_intermediate_size,
    )
    parser.add_argument(
        "--moe-latent-size",
        type=int,
        default=default_cfg.moe_latent_size,
        help="expert width; 0 runs the experts at hidden size",
    )
    parser.add_argument(
        "--n-routed-experts", type=int, default=default_cfg.n_routed_experts
    )
    parser.add_argument(
        "--topk", type=int, default=default_cfg.num_experts_per_tok
    )
    parser.add_argument(
        "--n-shared-experts", type=int, default=default_cfg.n_shared_experts
    )
    parser.add_argument("--fwd-only", action="store_true", help="skip backward")
    parser.add_argument("--warmup", type=int, default=default_cfg.warmup)
    parser.add_argument("--iters", type=int, default=default_cfg.iters)
    parser.add_argument(
        "--timing", choices=("pipeline", "barrier"), default=default_cfg.timing
    )
    parser.add_argument(
        "--ring-gpus-per-node",
        type=int,
        default=None,
        help="ringmoe intra-node GPU count G (default: world//2 on one node, "
        "else the dispatcher default)",
    )
    parser.add_argument(
        "--maxvio",
        default="",
        help="comma-separated MaxVio targets to sweep, e.g. 0,0.5,1,2,4 "
        "(routing injected, see routing_inject.py); empty = real router",
    )
    parser.add_argument(
        "--hot-placement",
        choices=("clustered", "spread", "random"),
        default=default_cfg.hot_placement,
        help="where hot experts sit vs the E//EP rank blocks",
    )
    parser.add_argument(
        "--maxvio-metric",
        choices=("per_expert", "per_rank"),
        default=default_cfg.maxvio_metric,
        help="MaxVio the --maxvio targets refer to (per_rank = per card)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    cfg = BenchConfig(
        tokens=args.tokens,
        hidden_size=args.hidden_size,
        moe_intermediate_size=args.moe_intermediate_size,
        moe_latent_size=args.moe_latent_size or None,
        n_routed_experts=args.n_routed_experts,
        num_experts_per_tok=args.topk,
        n_shared_experts=args.n_shared_experts,
        bwd=not args.fwd_only,
        warmup=args.warmup,
        iters=args.iters,
        timing=args.timing,
        ring_gpus_per_node=args.ring_gpus_per_node,
        maxvio_targets=[
            float(value) for value in args.maxvio.split(",") if value.strip()
        ],
        hot_placement=args.hot_placement,
        maxvio_metric=args.maxvio_metric,
    )
    paddle.set_default_dtype("bfloat16")
    dispatchers = [
        name.strip() for name in args.dispatchers.split(",") if name.strip()
    ] or default_dispatchers(cfg)
    results = run_all(dispatchers, cfg)
    if dist.get_rank() == 0:
        print(format_report(results, cfg), flush=True)
    return results


if __name__ == "__main__":
    main()
