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
"""Controlled router load imbalance (MaxVio) for the dispatcher benchmark.

A learned router on random input lands at an arbitrary imbalance, and each
dispatcher would see a different draw. This module makes routing a controlled
variable: one ``(seed, sigma, hot_placement)`` yields one routing that is
replayed identically for every dispatcher.

* :func:`generate_topk_routing` draws expert popularity from a lognormal whose
  sigma is shared by all ranks (the same experts are hot everywhere) while the
  per-token draws are rank-local. ``sigma == 0`` is a round-robin, perfectly
  balanced routing.
* ``MaxVio = max_e(load_e) / mean_e(load_e) - 1`` over the EP-wide per-expert
  histogram; :func:`solve_sigma_for_maxvio` solves for the sigma that hits a
  target MaxVio, and the benchmark reports the realized value.
* ``hot_placement`` decides where the hot experts sit relative to the
  ``E // EP`` block-to-rank mapping at the same per-expert MaxVio:
  ``clustered`` (fewest ranks hot, worst straggler), ``spread`` (one per rank,
  no straggler) or ``random``. Per-device-expert dispatchers (deepep in this
  benchmark) see the per-rank imbalance; allgather / ringmoe shard every
  expert over all ranks, so their per-card compute stays flat.
* :func:`injected_gate` runs the real router every step, keeping its forward
  and backward in the timed region, and overrides its routing decision with
  the generated one.

The generator re-seeds paddle's global RNG before each draw, since
``multinomial`` / ``randperm`` take no generator; do not use it in training.
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest import mock

import paddle

HOT_PLACEMENTS = ("clustered", "spread", "random")


def _local_seed(seed, rank):
    return int(seed) * 1000 + int(rank) + 1


def _hot_order(E, R, placement):
    """Expert ids ordered so the first receives the hottest logit."""
    experts_per_rank = max(E // R, 1)
    if placement in ("random", "clustered"):
        # Identity order fills rank 0's expert block first, then rank 1's.
        return list(range(E))
    if placement == "spread":
        order = [
            r * experts_per_rank + local_idx
            for local_idx in range(experts_per_rank)
            for r in range(R)
        ]
        seen = set(order)
        return order + [expert for expert in range(E) if expert not in seen]
    raise ValueError(f"unknown hot_placement {placement!r}")


def generate_topk_routing(
    T, K, E, R, sigma, seed, rank=0, hot_placement="clustered"
):
    """Return this rank's ``(topk_indices [T, K] int32, tokens_per_expert [E])``.

    ``T`` is the per-rank token count, ``K`` the top-k, ``E`` the global
    expert count and ``R`` the EP size. ``tokens_per_expert`` is the rank-local
    histogram. ``sigma == 0`` gives a balanced round-robin routing.
    """
    if K > E:
        raise ValueError(f"K({K}) must be <= E({E})")
    if sigma == 0.0:
        experts_per_rank = max(E // R, 1)
        token_ids = paddle.arange(T, dtype="int64").reshape([-1, 1])
        slot_ids = paddle.arange(K, dtype="int64").reshape([1, -1])
        target_rank = (token_ids + slot_ids) % R
        target_local = ((token_ids // R) + slot_ids) % experts_per_rank
        paddle.seed(_local_seed(seed, rank))
        perm = paddle.randperm(experts_per_rank, dtype="int64")
        local_expert = paddle.take_along_axis(
            perm.reshape([1, -1]).expand([T, experts_per_rank]),
            target_local,
            axis=1,
        )
        topk_indices = (target_rank * experts_per_rank + local_expert).astype(
            "int32"
        )
    else:
        paddle.seed(int(seed))  # popularity is shared by all ranks
        logits = paddle.exp(
            paddle.normal(mean=0.0, std=float(sigma), shape=[E])
        )
        if hot_placement != "random":
            # Permute the logits: per-expert MaxVio is unchanged, only the
            # rank blocks holding the hot experts move.
            order = paddle.to_tensor(_hot_order(E, R, hot_placement))
            logits = paddle.scatter(
                paddle.zeros_like(logits),
                order,
                paddle.sort(logits, descending=True),
            )
        probs = logits.reshape([1, E]).expand([T, E])
        paddle.seed(_local_seed(seed, rank))  # token draws are rank-local
        topk_indices = paddle.multinomial(
            probs, num_samples=K, replacement=False
        )
        topk_indices = topk_indices.astype("int32")
    tokens_per_expert = paddle.bincount(
        topk_indices.flatten().astype("int64"), minlength=E
    )
    return topk_indices.contiguous(), tokens_per_expert.astype("int64")


def maxvio(counts) -> float:
    """``max / mean - 1`` of a load vector (tensor or list); 0 = balanced."""
    loads = (
        counts.astype("float64").numpy().tolist()
        if isinstance(counts, paddle.Tensor)
        else [float(value) for value in counts]
    )
    if not loads or sum(loads) <= 0:
        return 0.0
    return max(loads) / (sum(loads) / len(loads)) - 1.0


def per_rank_load(global_counts, ep_size):
    """Per-expert histogram -> per-rank load under the ``E // EP`` block map."""
    counts = global_counts.astype("int64").numpy().tolist()
    experts_per_rank = max(len(counts) // ep_size, 1)
    return [
        float(
            sum(counts[rank * experts_per_rank : (rank + 1) * experts_per_rank])
        )
        for rank in range(ep_size)
    ]


MAXVIO_METRICS = ("per_expert", "per_rank")


def maxvio_for_sigma(
    sigma, T, K, E, R, seed, hot_placement="clustered", metric="per_expert"
):
    """Realized MaxVio at ``sigma``, simulating every rank locally.

    ``metric`` is ``per_expert`` (EP-wide per-expert histogram) or
    ``per_rank`` (that histogram summed over each rank's ``E // R`` block).
    """
    total = paddle.zeros([E], dtype="int64")
    for rank in range(R):
        _, tokens_per_expert = generate_topk_routing(
            T, K, E, R, sigma, seed, rank, hot_placement
        )
        total = total + tokens_per_expert
    if metric == "per_rank":
        return maxvio(per_rank_load(total, R))
    return maxvio(total)


def solve_sigma_for_maxvio(
    target,
    T,
    K,
    E,
    R,
    seed,
    hot_placement="clustered",
    iters=40,
    metric="per_expert",
):
    """Bisect (in log space) for the sigma whose MaxVio hits ``target``.

    Deterministic, so every rank gets the same sigma without communicating.
    """
    if target <= 0.0:
        return 0.0
    lo, hi = 1e-5, 10.0
    best_sigma, best_err = lo, float("inf")
    for _ in range(iters):
        mid = (lo * hi) ** 0.5
        realized = maxvio_for_sigma(
            mid, T, K, E, R, seed, hot_placement, metric
        )
        if abs(realized - target) < best_err:
            best_sigma, best_err = mid, abs(realized - target)
        if abs(realized - target) <= max(0.02 * target, 0.005):
            break
        lo, hi = (mid, hi) if realized < target else (lo, mid)
    return best_sigma


def capture_gate_signature(layer, hidden_states):
    """Run the real gate once and record the dtypes/shapes to reproduce."""
    with paddle.no_grad():
        out = layer.gate(hidden_states)
    if not isinstance(out, (tuple, list)) or len(out) != 8:
        raise RuntimeError("expected TopKRouter.forward's 8-tuple")
    _, top_gate, top_idx, probs, mask, _, _, _ = out
    return {
        "weights_dtype": top_gate.dtype,
        "indices_dtype": top_idx.dtype,
        "probs_dtype": probs.dtype,
        "mask_dtype": mask.dtype,
        "num_tokens": int(top_gate.shape[0]),
        "topk": int(top_gate.shape[1]),
        "num_experts": int(probs.shape[1]),
    }


def build_routing(sig, sigma, ep_group, rank, seed, hot_placement):
    """Build this rank's replayable routing and its EP-wide load statistics."""
    T, K, E = sig["num_tokens"], sig["topk"], sig["num_experts"]
    R = ep_group.nranks
    top_idx, tokens_per_expert = generate_topk_routing(
        T, K, E, R, sigma, seed, rank, hot_placement
    )
    # Rank- and row-dependent weights, so every dispatcher combines distinct
    # values rather than a uniform constant.
    rank_weight = paddle.full([T, K], 0.25 * (rank + 1), dtype="float32")
    row_offset = 0.01 * paddle.arange(T, dtype="float32").reshape([-1, 1])
    top_gate = (rank_weight + row_offset).astype(sig["weights_dtype"])
    top_gate.stop_gradient = False
    global_tokens_per_expert = tokens_per_expert.clone()
    if R > 1:
        paddle.distributed.all_reduce(global_tokens_per_expert, group=ep_group)
    return {
        "top_gate": top_gate,
        "top_idx": top_idx.astype(sig["indices_dtype"]),
        "sigma": float(sigma),
        "per_expert_maxvio": maxvio(global_tokens_per_expert),
        "per_rank_maxvio": maxvio(per_rank_load(global_tokens_per_expert, R)),
    }


def _scatter_dense(top_gate, top_idx, num_experts, probs_dtype, mask_dtype):
    """Scatter the top-k pair into dense ``(probs[T, E], mask[T, E])``.

    Differentiable in ``top_gate``; padded lanes (``idx < 0``) add nothing.
    """
    num_tokens = top_gate.shape[0]
    idx64 = top_idx.astype("int64")
    valid = (idx64 >= 0).astype(top_gate.dtype)
    idx_safe = paddle.clip(idx64, min=0)
    probs = paddle.put_along_axis(
        paddle.zeros([num_tokens, num_experts], dtype=top_gate.dtype),
        idx_safe,
        top_gate * valid,
        axis=1,
        reduce="add",
    )
    mask_values = paddle.ones_like(idx_safe, dtype=mask_dtype) * valid.astype(
        mask_dtype
    )
    mask = paddle.put_along_axis(
        paddle.zeros([num_tokens, num_experts], dtype=mask_dtype),
        idx_safe,
        mask_values,
        axis=1,
        reduce="add",
    )
    return probs.astype(probs_dtype), mask


@contextmanager
def injected_gate(layer, sig, routing):
    """Run the real router, then replace its routing decision with ``routing``.

    Patches ``forward`` on the gate class, since rebinding ``layer.gate`` would
    drop its parameters from the layer. The injected weights and probs are tied
    to the real outputs by ``+ 0 * real.sum()``: values are unchanged, so the
    MaxVio is exact, while backward still runs through the whole router.
    """
    num_experts = int(sig["num_experts"])
    original_forward = type(layer.gate).forward

    def stub_forward(_self, _input, input_ids=None, origin_input_ids=None):
        real_out = original_forward(
            _self,
            _input,
            input_ids=input_ids,
            origin_input_ids=origin_input_ids,
        )
        _, real_top_gate, _, real_probs, _, _, aux_loss, z_loss = real_out
        top_gate = (
            routing["top_gate"]
            + 0.0 * real_top_gate.astype(routing["top_gate"].dtype).sum()
        )
        probs, mask = _scatter_dense(
            top_gate,
            routing["top_idx"],
            num_experts,
            sig["probs_dtype"],
            sig["mask_dtype"],
        )
        probs = probs + 0.0 * real_probs.astype(probs.dtype).sum()
        return (
            None,
            top_gate,
            routing["top_idx"],
            probs,
            mask,
            None,
            aux_loss,
            z_loss,
        )

    with mock.patch.object(type(layer.gate), "forward", stub_forward):
        yield
