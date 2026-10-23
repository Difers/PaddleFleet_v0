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

"""Real 2-GPU coverage for ``calculate_per_token_loss`` (non-PP).

``calculate_per_token_loss`` rejects PP/VPP, so this runs at world_size=2,
TP=1, PP=1, DP=2 -- the plain branch of ``forward_impl`` -- reusing the inputs
and float64 numpy reference of ``test_language_loss_token_avg_reduction``. The
two ranks keep unequal valid token counts, so a dropped rank or a per-rank
normalization is observable in every cross-rank check.
"""

import numpy as np
import paddle
import paddle.distributed as dist

from paddlefleet.models.common.language_loss.language_loss import LanguageLoss
from tests.multi_card_tests.models.test_language_loss_token_avg_reduction import (
    VALID_PER_RANK,
    _build_config,
    _global_reference,
    _local_inputs,
    _numpy_masked_ce,
)
from tests.multi_card_tests.tensor_parallel.test_utilities import Utils


def per_token_forward(log_avg):
    """One real training forward with per-token on: ``(loss, loss_sum, count)``."""
    loss_fn = LanguageLoss(
        config=_build_config(per_token=True, log_avg=log_avg, pp=1)
    )
    logits_np, labels_np = _local_inputs(Utils.rank)
    logits = paddle.to_tensor(logits_np, dtype="float32").cuda()
    labels = paddle.to_tensor(labels_np, dtype="int64").cuda()
    loss = float(loss_fn(logits, labels).numpy())
    loss_sum, count = _numpy_masked_ce(logits_np, labels_np)
    return loss, loss_sum, count


def test_forward_returns_this_ranks_raw_sum():
    """Per-token forward returns the raw masked sum and counts this rank's
    tokens for T_global; the legacy (S, n) entries are left untouched."""
    LanguageLoss.token_avg_tracker.clear()
    try:
        loss, loss_sum, count = per_token_forward(log_avg=False)

        assert count == VALID_PER_RANK[Utils.rank]
        np.testing.assert_allclose(loss, loss_sum, rtol=1e-5, atol=1e-6)
        # Per-token stats share the tracker under their own keys; (S, n) stay
        # empty because log_global_token_avg_loss is off.
        assert "S" not in LanguageLoss.token_avg_tracker
        assert "n" not in LanguageLoss.token_avg_tracker
        stats = LanguageLoss.pop_per_token_stats()
        assert int(stats["pt_t_local"]) == count
        np.testing.assert_allclose(
            float(stats["pt_lm_sum"]), loss_sum, rtol=1e-5, atol=1e-6
        )
        assert int(stats["pt_lm_tok"]) == count
        assert LanguageLoss.token_avg_tracker == {}
    finally:
        LanguageLoss.token_avg_tracker.clear()


def test_t_global_and_lm_sum_reduce_across_ranks():
    """all_reduce(SUM) of the popped stats gives T_global = Σ valid tokens and
    the global per-token loss ΣS / T_global, which with unequal counts differs
    from the mean of the per-rank means."""
    LanguageLoss.token_avg_tracker.clear()
    try:
        per_token_forward(log_avg=False)
        stats = LanguageLoss.pop_per_token_stats()
        t_global = stats["pt_t_local"].clone()
        lm_sum = stats["pt_lm_sum"].clone()
        dist.all_reduce(t_global, dist.ReduceOp.SUM)
        dist.all_reduce(lm_sum, dist.ReduceOp.SUM)
        token_weighted, mean_of_means = _global_reference()

        assert int(t_global) == sum(VALID_PER_RANK)
        got = float(lm_sum) / int(t_global)
        np.testing.assert_allclose(got, token_weighted, rtol=1e-5, atol=1e-6)
        assert abs(got - mean_of_means) > 0.05
    finally:
        LanguageLoss.token_avg_tracker.clear()


def test_per_token_and_token_avg_log_coexist():
    """With both switches on, one forward fills (S, n) and the per-token keys;
    consume_token_avg_loss reduces only (S, n) and leaves the per-token keys
    for pop_per_token_stats."""
    LanguageLoss.token_avg_tracker.clear()
    try:
        _, loss_sum, count = per_token_forward(log_avg=True)

        got = LanguageLoss.consume_token_avg_loss()["loss"]
        token_weighted, _ = _global_reference()
        assert got is not None
        np.testing.assert_allclose(got, token_weighted, rtol=1e-5, atol=1e-6)
        assert "S" not in LanguageLoss.token_avg_tracker
        assert "n" not in LanguageLoss.token_avg_tracker

        stats = LanguageLoss.pop_per_token_stats()
        assert int(stats["pt_t_local"]) == count
        np.testing.assert_allclose(
            float(stats["pt_lm_sum"]), loss_sum, rtol=1e-5, atol=1e-6
        )
        assert LanguageLoss.token_avg_tracker == {}
    finally:
        LanguageLoss.token_avg_tracker.clear()


if __name__ == "__main__":
    Utils.initialize_model_parallel(
        tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=2
    )
    test_forward_returns_this_ranks_raw_sum()
    test_t_global_and_lm_sum_reduce_across_ranks()
    test_per_token_and_token_avg_log_coexist()
