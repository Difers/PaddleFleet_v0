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

"""Unit coverage for the global token-averaged loss display (loss 口径统一).

``log_global_token_avg_loss`` gates a logging-only convention: every training
forward stashes its raw ``(loss_sum, valid_tokens)`` in a class-level tracker
and the trainer consumes it to log the global token-weighted loss ΣΣS/ΣΣn
(matching Megatron lm_loss). The backprop term stays ``loss_sum / local_valid``.

Eval/predict forwards do not accumulate -- only the training loss is unified,
and the tracker must cover exactly one training logging interval. The
discriminator is ``paddle.is_grad_enabled()``, since both eval forward paths run
under ``paddle.no_grad()``; that way even a trainer which reimplements the eval
loop (ERNIE's ``EvalPPLMixin.evaluate``) cannot disturb an in-flight interval.
``self.training`` cannot serve that role -- it is always ``True`` because
``LanguageLoss`` lands in ``PipelineLayer._loss_fn`` (a plain list) and is never
a registered sublayer.

Single-process tests: no dist init, so ``consume_token_avg_loss`` skips the
all_reduce and just returns ΣS/Σn.
"""

from __future__ import annotations

import types
from unittest.mock import patch

import paddle
import pytest

from paddlefleet.models.common.language_loss import language_loss as ll
from paddlefleet.models.common.language_loss.language_loss import LanguageLoss


class _Cfg:
    def __init__(self, enabled):
        self.log_global_token_avg_loss = enabled


class _Self:
    """Minimal stand-in exposing the only attribute _reduce_loss_by_tokens reads."""

    def __init__(self, enabled):
        self.config = _Cfg(enabled)


def _reduce(loss_sum, local_valid, enabled=True):
    """Call the (self-independent) instance method without building the layer."""
    return LanguageLoss._reduce_loss_by_tokens(
        _Self(enabled), loss_sum, local_valid
    )


def _clear_tracker():
    LanguageLoss.token_avg_tracker.clear()


def test_disabled_returns_local_mean_and_leaves_tracker_empty():
    paddle.set_device("cpu")
    _clear_tracker()

    loss_sum = paddle.to_tensor(20.0)
    valid = paddle.to_tensor(4.0)
    out = _reduce(loss_sum, valid, enabled=False)

    assert abs(float(out) - 5.0) < 1e-6, float(out)
    assert LanguageLoss.token_avg_tracker == {}, LanguageLoss.token_avg_tracker


def test_enabled_backprop_value_unchanged_and_accumulates():
    paddle.set_device("cpu")
    _clear_tracker()
    try:
        loss_sum = paddle.to_tensor(20.0)
        valid = paddle.to_tensor(4.0)
        out = _reduce(loss_sum, valid)

        # Backprop term is still loss_sum / local_valid (bit-for-bit path).
        assert abs(float(out) - 5.0) < 1e-6, float(out)
        # The rank's raw (S, n) is stashed for the trainer.
        assert abs(float(LanguageLoss.token_avg_tracker["S"]) - 20.0) < 1e-6
        assert abs(float(LanguageLoss.token_avg_tracker["n"]) - 4.0) < 1e-6
    finally:
        _clear_tracker()


def test_consume_is_token_weighted_across_microbatches():
    """Two uneven micro-batches: the global loss is ΣS/Σn, NOT the mean of the
    two per-batch means (that is the point of token-weighting).
    """
    paddle.set_device("cpu")
    _clear_tracker()
    try:
        # batch A: mean 5 over 4 tokens ; batch B: mean 2 over 96 tokens.
        _reduce(paddle.to_tensor(20.0), paddle.to_tensor(4.0))
        _reduce(paddle.to_tensor(192.0), paddle.to_tensor(96.0))

        got = LanguageLoss.consume_token_avg_loss()
        expected = (20.0 + 192.0) / (4.0 + 96.0)  # = 2.12
        assert got["loss"] is not None
        assert abs(got["loss"] - expected) < 1e-6, (got, expected)
        # Simple average of the two means would be (5 + 2) / 2 = 3.5 -> differs.
        assert abs(got["loss"] - 3.5) > 0.1
        # Tracker is cleared after consumption.
        assert LanguageLoss.token_avg_tracker == {}
    finally:
        _clear_tracker()


def test_consume_returns_none_when_empty():
    paddle.set_device("cpu")
    _clear_tracker()
    assert LanguageLoss.consume_token_avg_loss() == {"loss": None}
    # Still empty / consumable again.
    assert LanguageLoss.consume_token_avg_loss(2) == {
        "loss": None,
        "mtp_1_loss": None,
        "mtp_2_loss": None,
    }


def test_no_grad_forward_does_not_accumulate():
    """Grad mode is the discriminator: a ``paddle.no_grad()`` forward (what the
    trainer wraps every eval/predict step in) must not touch the tracker."""
    paddle.set_device("cpu")
    _clear_tracker()
    try:
        with paddle.no_grad():
            out = _reduce(paddle.to_tensor(30.0), paddle.to_tensor(10.0))

        # Backprop-shaped value is still loss_sum / local_valid.
        assert abs(float(out) - 3.0) < 1e-6, float(out)
        assert LanguageLoss.token_avg_tracker == {}, (
            LanguageLoss.token_avg_tracker
        )
        # Nothing to consume, so the trainer leaves the log untouched.
        assert LanguageLoss.consume_token_avg_loss() == {"loss": None}
    finally:
        _clear_tracker()


def test_eval_pass_cannot_disturb_an_in_flight_training_interval():
    """The EB-shaped case: an eval pass runs while a training logging interval
    is still accumulating, with no trainer bookkeeping anywhere.

    ``should_log`` and ``should_evaluate`` are on independent schedules, so this
    overlap is normal whenever ``eval_steps`` is not a multiple of
    ``logging_steps``.
    """
    paddle.set_device("cpu")
    _clear_tracker()
    try:
        # Training interval in flight: 2 micro-batches, ΣS=212 over Σn=100.
        _reduce(paddle.to_tensor(20.0), paddle.to_tensor(4.0))
        _reduce(paddle.to_tensor(192.0), paddle.to_tensor(96.0))

        # An eval pass opens mid-interval.
        with paddle.no_grad():
            _reduce(paddle.to_tensor(30.0), paddle.to_tensor(10.0))
            _reduce(paddle.to_tensor(10.0), paddle.to_tensor(10.0))

        # The training interval is intact -- 212/100, not 252/120.
        train_v = LanguageLoss.consume_token_avg_loss()["loss"]
        assert abs(train_v - 212.0 / 100.0) < 1e-6, train_v
    finally:
        _clear_tracker()


def test_heads_are_reduced_separately():
    """Main and each MTP head keep their own (ΣS, Σn): the main loss must not
    be pooled with the MTP heads, and each MTP depth gets its own value."""
    paddle.set_device("cpu")
    _clear_tracker()
    try:
        _reduce(paddle.to_tensor(20.0), paddle.to_tensor(10.0))  # main: 2.0
        with ll._token_avg_head_scope(ll._mtp_token_avg_prefix(0)):
            _reduce(paddle.to_tensor(45.0), paddle.to_tensor(9.0))  # mtp1: 5.0
        with ll._token_avg_head_scope(ll._mtp_token_avg_prefix(1)):
            _reduce(paddle.to_tensor(56.0), paddle.to_tensor(8.0))  # mtp2: 7.0

        got = LanguageLoss.consume_token_avg_loss(2)
        assert abs(got["loss"] - 2.0) < 1e-6, got
        assert abs(got["mtp_1_loss"] - 5.0) < 1e-6, got
        assert abs(got["mtp_2_loss"] - 7.0) < 1e-6, got
        assert LanguageLoss.token_avg_tracker == {}
    finally:
        _clear_tracker()


def test_record_override_wins_over_grad_mode():
    """``_token_avg_record`` (set by the recompute wrapper) decides on its own:
    False skips even with grad on, True records even under no_grad."""
    paddle.set_device("cpu")
    _clear_tracker()
    try:
        token = ll._token_avg_record.set(False)
        try:
            _reduce(paddle.to_tensor(20.0), paddle.to_tensor(4.0))
        finally:
            ll._token_avg_record.reset(token)
        assert LanguageLoss.token_avg_tracker == {}

        token = ll._token_avg_record.set(True)
        try:
            with paddle.no_grad():
                _reduce(paddle.to_tensor(20.0), paddle.to_tensor(4.0))
        finally:
            ll._token_avg_record.reset(token)
        assert abs(LanguageLoss.consume_token_avg_loss()["loss"] - 5.0) < 1e-6
    finally:
        _clear_tracker()


def _logit0_loss_func(logits, labels):
    """Per-token loss = logits[..., 0], so each head's CE is set by its input."""
    return logits[:, :, 0].cast("float32")


def _make_mtp_loss_stub(num_nextn):
    config = types.SimpleNamespace(
        gpt_model_use_experimental_version=False,
        sequence_parallel=False,
        num_nextn_predict_layers=num_nextn,
        mtp_load_weight_only=False,
        use_erndata=False,
        mtp_distillation_loss=False,
        train_mtp_only=False,
        add_mtp_loss=False,
        mtp_loss_scaling_factor=0.1,
        fused_linear_ce_loss_chunk=0,
        log_global_token_avg_loss=True,
    )
    stub = types.SimpleNamespace(
        config=config,
        use_subbatch=False,
        ignored_index=-100,
        use_accuracy_compatible=False,
        pg_collection=None,
        enable_parallel_cross_entropy=False,
        loss_func=_logit0_loss_func,
        calculate_per_token_loss=False,
        training=True,
    )
    for name in (
        "forward",
        "forward_impl",
        "_forward",
        "_global_labels_for_depth",
        "_labels_for_depth",
        "_reduce_loss_by_tokens",
    ):
        setattr(stub, name, types.MethodType(getattr(LanguageLoss, name), stub))
    return stub


def _run_mtp_forward(recompute_loss_fn):
    # main head per-token loss 2.0, MTP depth 1 per-token loss 5.0.
    main = paddle.full([1, 3, 4], 2.0, dtype="float32")
    mtp = paddle.full([1, 3, 4], 5.0, dtype="float32")
    main.stop_gradient = False
    mtp.stop_gradient = False
    # labels carry one extra trailing position for the single MTP depth;
    # one ignored token so the heads see different valid counts.
    labels = paddle.to_tensor([[1, -100, 3, 4]], dtype="int64")
    stub = _make_mtp_loss_stub(num_nextn=1)
    with (
        patch.object(
            ll, "module_needs_recompute", return_value=recompute_loss_fn
        ),
        patch.object(ll, "use_dsv4_accuracy_compatible", return_value=False),
    ):
        loss = stub.forward([main, mtp], labels)
        loss.backward()
    return loss


@pytest.mark.parametrize("recompute_loss_fn", [False, True])
def test_mtp_forward_logs_main_and_mtp_heads_separately(recompute_loss_fn):
    """End to end through ``LanguageLoss.forward`` with one MTP depth: the
    main head reads 2.0 and MTP reads 5.0 -- pooling them would give ~3.5. With
    ``loss_fn`` recomputed, the backward replay must not count a head twice
    nor land an MTP replay on the main head."""
    paddle.set_device("cpu")
    _clear_tracker()
    try:
        loss = _run_mtp_forward(recompute_loss_fn)
        # add_mtp_loss=False: backprop value is the main loss alone.
        assert abs(float(loss) - 2.0) < 1e-6, float(loss)

        tr = LanguageLoss.token_avg_tracker
        assert abs(float(tr["n"]) - 2.0) < 1e-6, tr  # main: 1 of 3 ignored
        assert abs(float(tr["mtp_1_n"]) - 2.0) < 1e-6, tr

        got = LanguageLoss.consume_token_avg_loss(1)
        assert abs(got["loss"] - 2.0) < 1e-6, got
        assert abs(got["mtp_1_loss"] - 5.0) < 1e-6, got
    finally:
        _clear_tracker()


if __name__ == "__main__":
    import sys

    try:
        test_disabled_returns_local_mean_and_leaves_tracker_empty()
        test_enabled_backprop_value_unchanged_and_accumulates()
        test_consume_is_token_weighted_across_microbatches()
        test_consume_returns_none_when_empty()
        test_no_grad_forward_does_not_accumulate()
        test_eval_pass_cannot_disturb_an_in_flight_training_interval()
        test_heads_are_reduced_separately()
        test_record_override_wins_over_grad_mode()
        test_mtp_forward_logs_main_and_mtp_heads_separately(False)
        test_mtp_forward_logs_main_and_mtp_heads_separately(True)
    except AssertionError:
        import traceback

        traceback.print_exc()
        sys.exit(1)
    print("OK")
