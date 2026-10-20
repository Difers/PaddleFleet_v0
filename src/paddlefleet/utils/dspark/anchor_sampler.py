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

"""Deterministic anchor sampling, independent of framework and global RNG."""

import hashlib
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np
import paddle


@dataclass(frozen=True)
class AnchorBatch:
    """AnchorBatch."""

    positions: Any
    keep_mask: Any
    # Number of rows whose valid-position count was below ``num_anchors``, so
    # the fixed-size row had to be filled by cycling its shuffled candidates.
    cycled_rows: int = 0


_CYCLING_REPORTED = False


def _report_cycling(rows, total_rows, fewest_valid, num_anchors):
    """Warn once when rows need repeated anchors to fill the fixed grid."""

    global _CYCLING_REPORTED
    if _CYCLING_REPORTED:
        return
    _CYCLING_REPORTED = True
    warnings.warn(
        f"DSpark anchor sampling repeated anchors in {rows}/{total_rows} rows: "
        f"the sparsest row had {fewest_valid} valid positions for num_anchors={num_anchors}. "
        "Repeated anchors count once per occurrence in both the draft-loss numerator and "
        "denominator, so affected rows carry more weight than fully-distinct rows.",
        RuntimeWarning,
        stacklevel=3,
    )


def _positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def sample_anchor_positions(
    *,
    loss_mask: Any,
    num_anchors: int,
    sample_ids: Sequence[str],
    global_step: int,
    training_seed: int,
) -> AnchorBatch:
    """Sample deterministic fixed-size anchor rows.

    Invalid rows use zero positions and false keep masks. Short rows cycle valid
    positions to fill ``num_anchors``.
    """
    _positive_int("num_anchors", num_anchors)
    for name, value in (
        ("global_step", global_step),
        ("training_seed", training_seed),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{name} must be an integer")
    if global_step < 0:
        raise ValueError("global_step must be non-negative")
    is_tensor = hasattr(loss_mask, "place") and hasattr(loss_mask, "numpy")

    def as_numpy(value):
        return (
            value.detach().cpu().numpy()
            if hasattr(value, "detach")
            else np.asarray(value)
        )

    mask = as_numpy(loss_mask).astype(bool, copy=True)
    if mask.ndim != 2 or mask.shape[0] == 0 or mask.shape[1] == 0:
        raise ValueError("loss_mask must have nonempty shape [batch, seq_len]")
    if len(sample_ids) != mask.shape[0] or any(
        not isinstance(s, str) or not s for s in sample_ids
    ):
        raise ValueError("sample_ids must contain one non-empty string per row")
    mask[:, 0] = False
    positions = np.zeros((mask.shape[0], num_anchors), dtype=np.int64)
    keep = np.zeros_like(positions, dtype=bool)
    cycled_rows = 0
    fewest_valid = None
    for row, sample_id in enumerate(sample_ids):
        valid = np.flatnonzero(mask[row])
        if not len(valid):
            continue
        if len(valid) < num_anchors:
            cycled_rows += 1
            fewest_valid = (
                len(valid)
                if fewest_valid is None
                else min(fewest_valid, len(valid))
            )
        material = f"dspark-anchor-v1\0{training_seed}\0{global_step}\0{sample_id}".encode()
        seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "little")
        order = valid[np.random.default_rng(seed).permutation(len(valid))]
        positions[row] = np.sort(order[np.arange(num_anchors) % len(order)])
        keep[row] = True
    if cycled_rows:
        _report_cycling(cycled_rows, mask.shape[0], fewest_valid, num_anchors)
    if is_tensor:
        positions = paddle.to_tensor(positions, place=loss_mask.place)
        keep = paddle.to_tensor(keep, place=loss_mask.place)
    return AnchorBatch(positions, keep, cycled_rows)
