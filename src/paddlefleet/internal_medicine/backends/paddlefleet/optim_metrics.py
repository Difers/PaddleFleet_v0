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

"""Per-parameter optimizer-step metrics for PaddleFleet.

Pure tensor functions, no monitor state. The monitor reads four sums of squares
per parameter around one optimizer step,

    GRAD_SQ = sum(g^2)          the gradient the optimizer consumes
    W0_SQ   = sum(W_t^2)        the weight before the step
    W1_SQ   = sum(W_{t+1}^2)    the weight after the step
    DW_SQ   = sum((W_{t+1} - W_t)^2)

and everything reported is a function of those four, which is what lets a
sharded parameter be summed back across ranks before any ratio is taken: a sum
of squares is additive over shards, a norm or a ratio is not.

Everything is computed in float32 on the optimizer's master weight. The step of
a healthy run moves a weight by ~lr of its norm (1e-3 .. 1e-2), which is the same
order as bf16's relative rounding (2^-8 .. 2^-9); differencing bf16 weights would
measure the rounding, not the update.
"""

from __future__ import annotations

import paddle

GRAD_SQ, W0_SQ, W1_SQ, DW_SQ = range(4)
NUM_SUMS = 4

# Per-parameter series, one key each per parameter.
#   grad_norm     ||g||_F, before clipping (clip happens inside the optimizer)
#   grad_rel      ||g||_F / ||W_t||_F
#   eff_lr        ||W_{t+1} - W_t||_F / ||W_t||_F, the step actually taken,
#                 radial part included.
#   eff_lr_ratio  eff_lr / (lr * lr_ratio): the per-unit-lr relative step.
#   aus           Angular update size, 2 sin(theta / 2) with theta the angle
#                 between W_t and W_{t+1}: how far the weight's direction turned,
#                 its norm change set aside. The closed forms reconstruct theta
#                 from the base update u (before the lr) and c = cos(W, u):
#                 atan2(lr sqrt(1 - c^2), 1 - lr c) under Hyperball, and
#                 atan2(lr ||u|| sqrt(1 - c^2), (1 - lr wd) ||W|| - lr ||u|| c)
#                 with decoupled weight decay. Here it is measured on the step
#                 that was taken, which gives the same angle without having to
#                 reach u inside the optimizer. Under Hyperball the norm is fixed
#                 and aus equals eff_lr.
#   aus_over_lr   aus / (lr * lr_ratio). ~sqrt(1 - c^2) under Hyperball, i.e.
#                 ~1 since an update is nearly orthogonal to a large matrix; at
#                 most sqrt(2 - 2 sqrt(1 - lr^2)) / lr ~= 1 + lr^2 / 8, reached
#                 at c = lr. For plain AdamW the rotation per unit lr, without
#                 the norm change that eff_lr_ratio also counts.
#   param_norm    ||W_{t+1}||_F
#   norm_drift    ||W_{t+1}||_F / ||W_t||_F - 1: ~0 (float32 noise) under
#                 Hyperball, the per-step norm growth for plain AdamW.
PARAM_METRICS = (
    "grad_norm",
    "grad_rel",
    "eff_lr",
    "eff_lr_ratio",
    "aus",
    "aus_over_lr",
    "param_norm",
    "norm_drift",
)

# Per-expert spread of a grouped-expert weight ``[E, in, out]``, over every
# expert of the layer (gathered across the EP group, so the quantiles are exact).
# ``zero_grad_experts`` counts experts whose gradient is exactly zero this step,
# i.e. that received no token; under Muon they still move at full lr on stale
# momentum, which is why eff_lr sits next to it. Per-expert readings exist only
# for Muon / Hyperball weights, where each expert keeps its own norm, so the
# eff_lr spread already is the per-expert aus spread.
EXPERT_METRICS = (
    "grad_rel_min",
    "grad_rel_p10",
    "grad_rel_median",
    "grad_rel_max",
    "eff_lr_min",
    "eff_lr_median",
    "eff_lr_max",
    "zero_grad_experts",
)

_TINY = 1e-30


def _fp32(tensor: paddle.Tensor) -> paddle.Tensor:
    return (
        tensor if tensor.dtype == paddle.float32 else tensor.astype("float32")
    )


def _square_sum(value: paddle.Tensor, per_expert: bool) -> paddle.Tensor:
    """``sum(value^2)`` -> 0-d, or ``[E]`` over the trailing matrix of a 3-D stack."""
    squared = value * value
    if per_expert:
        return squared.sum(axis=[1, 2])
    return squared.sum()


def grad_square(grad: paddle.Tensor, per_expert: bool) -> paddle.Tensor:
    """Sum of squares of a gradient in float32; ``[E]`` for a grouped-expert weight."""
    return _square_sum(_fp32(grad.detach()), per_expert)


def snapshot(weight: paddle.Tensor) -> paddle.Tensor:
    """Float32 copy of ``weight`` that the optimizer's in-place update cannot touch."""
    value = weight.detach()
    if value.dtype == paddle.float32:
        return value.clone()
    return value.astype("float32")


def step_sums(
    grad_sq: paddle.Tensor,
    before: paddle.Tensor,
    after: paddle.Tensor,
    per_expert: bool,
) -> paddle.Tensor:
    """``[4]`` (or ``[E, 4]``) vector of the four sums in ``GRAD_SQ..DW_SQ`` order."""
    now = _fp32(after.detach())
    return paddle.stack(
        [
            grad_sq,
            _square_sum(before, per_expert),
            _square_sum(now, per_expert),
            _square_sum(now - before, per_expert),
        ],
        axis=-1 if per_expert else 0,
    )


def angular_update_size(
    before: paddle.Tensor, after: paddle.Tensor, delta_sq: paddle.Tensor
) -> paddle.Tensor:
    """``2 sin(theta / 2)`` from ``||W_t||``, ``||W_{t+1}||`` and ``||dW||^2``.

    It is the chord between the unit-normalised weights, and the radial part of
    ``dW`` is what separates it from ``||dW||``:
    ``chord^2 = (||dW||^2 - (||W_{t+1}|| - ||W_t||)^2) / (||W_t|| ||W_{t+1}||)``.
    Going through ``1 - cos(theta)`` instead would not survive float32: at a step
    of ~1e-3 it is ~5e-7, under the precision of the sums it would come from,
    while ``||dW||^2`` is summed from the difference itself.
    """
    radial = after - before
    return paddle.sqrt(
        (delta_sq - radial * radial).clip(min=0.0) / (before * after)
    )


def param_metrics(
    sums: paddle.Tensor, scheduled_lr: paddle.Tensor
) -> dict[str, paddle.Tensor]:
    """The ``PARAM_METRICS`` columns from an ``[N, 4]`` table of global sums.

    ``scheduled_lr`` is ``[N]``: this step's lr times each parameter's lr ratio.
    Values that are undefined (zero weight, zero lr) come out non-finite; the
    caller drops them rather than logging a NaN.
    """
    grad = paddle.sqrt(sums[:, GRAD_SQ])
    before = paddle.sqrt(sums[:, W0_SQ])
    after = paddle.sqrt(sums[:, W1_SQ])
    delta = paddle.sqrt(sums[:, DW_SQ])
    eff_lr = delta / before
    aus = angular_update_size(before, after, sums[:, DW_SQ])
    return {
        "grad_norm": grad,
        "grad_rel": grad / before,
        "eff_lr": eff_lr,
        "eff_lr_ratio": eff_lr / scheduled_lr,
        "aus": aus,
        "aus_over_lr": aus / scheduled_lr,
        "param_norm": after,
        "norm_drift": after / before - 1.0,
    }


def expert_metrics(sums: paddle.Tensor) -> dict[str, paddle.Tensor]:
    """The ``EXPERT_METRICS`` columns from an ``[S, E, 4]`` table, one row per weight."""
    before = paddle.sqrt(sums[:, :, W0_SQ]).clip(min=_TINY)
    grad_rel = paddle.sqrt(sums[:, :, GRAD_SQ]) / before
    eff_lr = paddle.sqrt(sums[:, :, DW_SQ]) / before
    grad_q = paddle.quantile(grad_rel, q=[0.1, 0.5], axis=1)
    return {
        "grad_rel_min": grad_rel.min(axis=1),
        "grad_rel_p10": grad_q[0],
        "grad_rel_median": grad_q[1],
        "grad_rel_max": grad_rel.max(axis=1),
        "eff_lr_min": eff_lr.min(axis=1),
        "eff_lr_median": paddle.quantile(eff_lr, q=0.5, axis=1),
        "eff_lr_max": eff_lr.max(axis=1),
        "zero_grad_experts": (sums[:, :, GRAD_SQ] == 0.0)
        .astype("float32")
        .sum(axis=1),
    }
