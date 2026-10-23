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

"""Real 2-GPU coverage for ``consume_token_avg_loss``'s cross-rank reduction.

With ``log_global_token_avg_loss`` on, every training forward stashes this rank's
raw ``(loss_sum, valid_tokens)`` in a class-level tracker; the trainer then calls
``consume_token_avg_loss``, which packs every head's terms into one float64
``all_reduce(SUM)`` and returns the global token-weighted loss ΣΣS/ΣΣn
(matching Megatron ``lm_loss``). The switch is independent of
``calculate_per_token_loss``, which only decides what
``_reduce_loss_by_tokens`` returns for backprop. That collective is skipped
when the world has a single rank, so a multi-card test is the only thing that
can reach it.

Topology: world_size=2, TP=1, PP=2 -- the plain (non parallel-cross-entropy)
branch of ``forward_impl``. The two ranks get deterministic but *unequal* valid
token counts, which is what makes token weighting observable: ΣΣS/ΣΣn must then
differ from the arithmetic mean of the two per-rank means. Every expected value
comes from an independent float64 numpy reference over both ranks' known inputs,
never from the module under test.
"""

import functools

import numpy as np
import paddle

from paddlefleet.models.common.language_loss.language_loss import LanguageLoss
from paddlefleet.transformer.transformer_config import TransformerConfig
from tests.multi_card_tests.tensor_parallel.test_utilities import Utils

IGNORE_INDEX = -100
VOCAB_SIZE = 5
SEQ_LEN = 4
BATCH_SIZE = 1
# Rank 0 keeps 3 of its 4 tokens, rank 1 keeps 1 -- unequal on purpose.
VALID_PER_RANK = (3, 1)


def _build_config(per_token=False, log_avg=True, pp=2):
    """Minimal config for the plain TP=1 loss branch.

    The two switches are independent, so each test turns on only the one it is
    about: ``calculate_per_token_loss`` for the raw-sum forward,
    ``log_global_token_avg_loss`` for the tracker / reduction.
    """
    return TransformerConfig(
        hidden_size=64,
        num_attention_heads=4,
        use_cpu_initialization=True,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=pp,
        sequence_parallel=False,
        params_dtype=paddle.float32,
        parallel_output=True,
        hidden_dropout_prob=0.0,
        attention_dropout=0.0,
        loss_subbatch_sequence_length=0,
        gpt_model_use_experimental_version=False,
        calculate_per_token_loss=per_token,
        log_global_token_avg_loss=log_avg,
        init_method=functools.partial(paddle.nn.init.xavier_uniform_, gain=1.0),
        output_layer_init_method=functools.partial(
            paddle.nn.init.xavier_uniform_, gain=1.0
        ),
    )


def _local_inputs(rank):
    """Deterministic rank-distinguishable logits plus unequal valid counts.

    The per-class tilt breaks softmax shift-invariance across ranks and the
    per-rank scale changes the distribution sharpness, so the two ranks
    contribute genuinely different loss sums; ``VALID_PER_RANK`` makes their
    token counts differ too.
    """
    base = np.arange(
        BATCH_SIZE * SEQ_LEN * VOCAB_SIZE, dtype=np.float64
    ).reshape([BATCH_SIZE, SEQ_LEN, VOCAB_SIZE])
    tilt = np.arange(VOCAB_SIZE, dtype=np.float64) * 0.31 * (rank + 1)
    logits = base * (0.1 + 0.05 * rank) + tilt
    labels = np.full([BATCH_SIZE, SEQ_LEN], IGNORE_INDEX, dtype=np.int64)
    for s in range(VALID_PER_RANK[rank]):
        labels[0, s] = (rank + s + 1) % VOCAB_SIZE
    return logits, labels


def _numpy_masked_ce(logits, labels):
    """Independent float64 reference: ``(loss_sum, valid_count)``."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    log_softmax = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    loss_sum = 0.0
    valid_count = 0
    for b in range(logits.shape[0]):
        for s in range(logits.shape[1]):
            label = int(labels[b, s])
            if label == IGNORE_INDEX:
                continue
            loss_sum += -float(log_softmax[b, s, label])
            valid_count += 1
    return loss_sum, valid_count


def _global_reference():
    """``(ΣΣS/ΣΣn, mean-of-per-rank-means)`` from both ranks' known inputs."""
    sums, counts = [], []
    for rank in range(Utils.world_size):
        loss_sum, count = _numpy_masked_ce(*_local_inputs(rank))
        sums.append(loss_sum)
        counts.append(count)
    token_weighted = sum(sums) / sum(counts)
    mean_of_means = sum(s / n for s, n in zip(sums, counts)) / len(sums)
    return token_weighted, mean_of_means


def _forward_once():
    """One real training forward on this rank: ``(loss, loss_sum, count)``.

    ``calculate_per_token_loss`` is unsupported under PP, so this runs with both
    switches off; the tracker must stay empty -- that is the orthogonality
    assertion in ``test_forward_returns_this_ranks_local_mean``. The per-token
    raw-sum forward is covered at PP=1 in test_language_loss_per_token_reduction.
    """
    loss_fn = LanguageLoss(config=_build_config(per_token=False, log_avg=False))
    logits_np, labels_np = _local_inputs(Utils.rank)
    logits = paddle.to_tensor(logits_np, dtype="float32").cuda()
    labels = paddle.to_tensor(labels_np, dtype="int64").cuda()
    loss = float(loss_fn(logits, labels).numpy())
    loss_sum, count = _numpy_masked_ce(logits_np, labels_np)
    return loss, loss_sum, count


def _stash_local_terms():
    """Feed this rank's numpy (S, n) into the tracker via the real
    ``_reduce_loss_by_tokens``, with ``log_global_token_avg_loss`` on."""
    loss_fn = LanguageLoss(config=_build_config(per_token=False, log_avg=True))
    loss_sum, count = _numpy_masked_ce(*_local_inputs(Utils.rank))
    loss_fn._reduce_loss_by_tokens(
        paddle.to_tensor(loss_sum, dtype="float32"),
        paddle.to_tensor(float(count), dtype="float32"),
    )


def test_forward_returns_this_ranks_local_mean():
    """Without per-token loss the forward returns this rank's masked mean, and
    neither the legacy (S, n) nor the per-token keys are written."""
    LanguageLoss.token_avg_tracker.clear()
    try:
        loss, loss_sum, count = _forward_once()

        assert count == VALID_PER_RANK[Utils.rank]
        np.testing.assert_allclose(loss, loss_sum / count, rtol=1e-5, atol=1e-6)
        assert LanguageLoss.token_avg_tracker == {}
    finally:
        LanguageLoss.token_avg_tracker.clear()


def test_per_token_rejects_pipeline_parallel():
    """calculate_per_token_loss is unsupported under PP, so the PP=2 config
    must be rejected at construction instead of mis-scaling the loss."""
    try:
        _build_config(per_token=True, log_avg=False)
    except ValueError as e:
        assert "pipeline parallel" in str(e)
    else:
        raise AssertionError("calculate_per_token_loss + PP=2 was not rejected")


def test_consume_all_reduces_both_ranks_into_sigma_s_over_sigma_n():
    """The collective under test: one packed ``all_reduce(SUM)`` over 2 GPUs.

    Both ranks are checked against the same global reference, and because their
    token counts differ it cannot coincide with the mean of the per-rank means --
    a dropped rank, a wrong reduce op, or normalizing before reducing all fail.
    """
    LanguageLoss.token_avg_tracker.clear()
    try:
        _stash_local_terms()

        got = LanguageLoss.consume_token_avg_loss()["loss"]
        token_weighted, mean_of_means = _global_reference()

        assert got is not None
        np.testing.assert_allclose(got, token_weighted, rtol=1e-5, atol=1e-6)
        # Token weighting is observable here, not an artifact of equal counts.
        assert abs(token_weighted - mean_of_means) > 0.05
        assert abs(got - mean_of_means) > 0.05
        # Consumption clears the tracker on every rank.
        assert LanguageLoss.token_avg_tracker == {}
    finally:
        LanguageLoss.token_avg_tracker.clear()


def test_second_consume_is_a_no_op_that_still_runs_in_lockstep():
    """After consumption every rank contributes (0, 0) and gets ``None`` back.

    This is the contract the trainer leans on: a second ``log()`` in the same
    step must not desync, and ``None`` tells it to leave the log untouched.
    """
    LanguageLoss.token_avg_tracker.clear()
    try:
        _stash_local_terms()

        assert LanguageLoss.consume_token_avg_loss()["loss"] is not None
        assert LanguageLoss.consume_token_avg_loss()["loss"] is None
        assert LanguageLoss.consume_token_avg_loss()["loss"] is None
    finally:
        LanguageLoss.token_avg_tracker.clear()


def test_mtp_head_reduces_in_lockstep_even_where_absent():
    """Per-head reduction: only rank 0 has an MTP head this interval, yet both
    ranks pass the same ``num_mtp`` and reach one shared collective, and the
    main head stays the main-only ΣΣS/ΣΣn."""
    from paddlefleet.models.common.language_loss import language_loss as ll

    LanguageLoss.token_avg_tracker.clear()
    try:
        _stash_local_terms()
        if Utils.rank == 0:
            with ll._token_avg_head_scope(ll._mtp_token_avg_prefix(0)):
                _stash_local_terms()

        got = LanguageLoss.consume_token_avg_loss(1)
        token_weighted, _ = _global_reference()
        mtp_sum, mtp_n = _numpy_masked_ce(*_local_inputs(0))

        np.testing.assert_allclose(
            got["loss"], token_weighted, rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            got["mtp_1_loss"], mtp_sum / mtp_n, rtol=1e-5, atol=1e-6
        )
    finally:
        LanguageLoss.token_avg_tracker.clear()


if __name__ == "__main__":
    Utils.initialize_model_parallel(
        tensor_parallel_size=1, pipeline_parallel_size=2
    )
    test_forward_returns_this_ranks_local_mean()
    test_per_token_rejects_pipeline_parallel()
    test_consume_all_reduces_both_ranks_into_sigma_s_over_sigma_n()
    test_second_consume_is_a_no_op_that_still_runs_in_lockstep()
    test_mtp_head_reduces_in_lockstep_even_where_absent()
