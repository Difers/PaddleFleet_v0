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

"""Trainer-side half of the global token-averaged loss display (loss 口径统一).

``Trainer._apply_global_token_avg_loss`` is the only consumer of the criterion's
tracker: with ``log_global_token_avg_loss`` on it swaps ``logs["loss"]`` for the
global token-weighted ΣΣS/ΣΣn and keeps the micro-average as
``logs["loss_microavg"]``. The criterion side is covered by
``transformer/test_language_loss_token_avg.py``; this file pins the gates and the
``log()`` call site.

Both gates run BEFORE ``consume_token_avg_loss``, which does two
``all_reduce``es, so they must be rank-invariant and must leave the tracker
untouched when they fire -- otherwise some ranks reduce and others don't, and
training desyncs. That is why the tests below assert the tracker survives a
skipped call, not just that ``logs`` is unchanged.

``Trainer`` needs no fleet state here: every method is called unbound with a
``types.SimpleNamespace`` standing in for ``self`` (same pattern as
``model_accuracy/test_dsv4_flag_gating_trainer.py``).
"""

from __future__ import annotations

import types
from unittest.mock import patch

from paddlefleet.models.common.language_loss.language_loss import LanguageLoss
from paddlefleet.trainer import trainer

# A no-op ``get_timers`` replacement: raising AssertionError is exactly what an
# uninitialized fleet timer helper does, and ``log()`` documents that branch.
_NO_TIMERS = patch(
    "paddle.distributed.fleet.utils.timer_helper.get_timers",
    side_effect=AssertionError,
)


def _fake_trainer(enabled, num_mtp=0):
    """Minimal ``self``: the method only reads ``self.model.config``."""
    return types.SimpleNamespace(
        model=types.SimpleNamespace(
            config=types.SimpleNamespace(
                log_global_token_avg_loss=enabled,
                num_nextn_predict_layers=num_mtp,
            )
        )
    )


def _seed_tracker(S=212.0, n=100.0):
    """Stand in for a logging interval's worth of accumulated forwards."""
    LanguageLoss.token_avg_tracker.clear()
    LanguageLoss.token_avg_tracker["S"] = S
    LanguageLoss.token_avg_tracker["n"] = n


def _apply(enabled, logs, num_mtp=0):
    trainer.Trainer._apply_global_token_avg_loss(
        _fake_trainer(enabled, num_mtp), logs
    )
    return logs


def test_flag_off_leaves_logs_alone_and_does_not_consume():
    """Config gate: no override, and crucially no collective."""
    _seed_tracker()
    try:
        logs = _apply(False, {"loss": 9.5})

        assert logs == {"loss": 9.5}, logs
        # Tracker untouched -> consume_token_avg_loss was never reached.
        assert LanguageLoss.token_avg_tracker["n"] == 100.0
    finally:
        LanguageLoss.token_avg_tracker.clear()


def test_a_log_without_loss_does_not_consume():
    """The ``"loss" not in logs`` gate keeps eval / checkpoint-save / train-end
    ``log()`` calls out, and is rank-invariant: ``_maybe_log_save_evaluate``
    always sets ``"loss"``, those three sites never do."""
    _seed_tracker()
    try:
        logs = _apply(True, {"eval_loss": 1.25})

        assert logs == {"eval_loss": 1.25}, logs
        assert LanguageLoss.token_avg_tracker["n"] == 100.0
    finally:
        LanguageLoss.token_avg_tracker.clear()


def test_enabled_overrides_loss_and_keeps_the_microavg():
    """The happy path: ΣS/Σn replaces ``loss``, the old value is preserved."""
    _seed_tracker()
    try:
        logs = _apply(True, {"loss": 9.5, "learning_rate": 1e-4})

        assert abs(logs["loss"] - 2.12) < 1e-6, logs["loss"]
        assert logs["loss_microavg"] == 9.5
        assert logs["learning_rate"] == 1e-4  # untouched
        assert LanguageLoss.token_avg_tracker == {}
    finally:
        LanguageLoss.token_avg_tracker.clear()


def test_empty_tracker_degrades_silently_to_the_microavg():
    """``consume_token_avg_loss`` returns ``None`` when nothing accumulated, and
    then ``loss`` must stay the micro-average with no ``loss_microavg`` key."""
    LanguageLoss.token_avg_tracker.clear()

    logs = _apply(True, {"loss": 9.5})

    assert logs == {"loss": 9.5}, logs
    assert "loss_microavg" not in logs


def test_mtp_heads_are_overridden_separately():
    """With MTP, ``loss`` is the main head's ΣS/Σn only and each
    ``mtp_{d}_loss`` is replaced by its own, the old values kept as
    ``loss_microavg`` / ``mtp_{d}_loss_last`` (the MTP tracker holds the last
    micro-batch's value, not a micro-average). A depth with no tokens this
    interval is left alone."""
    _seed_tracker()
    LanguageLoss.token_avg_tracker["mtp_1_S"] = 450.0
    LanguageLoss.token_avg_tracker["mtp_1_n"] = 90.0
    try:
        logs = _apply(
            True,
            {"loss": 9.5, "mtp_1_loss": 7.0, "mtp_2_loss": 8.0},
            num_mtp=2,
        )

        assert abs(logs["loss"] - 2.12) < 1e-6, logs
        assert logs["loss_microavg"] == 9.5
        assert abs(logs["mtp_1_loss"] - 5.0) < 1e-6, logs
        assert logs["mtp_1_loss_last"] == 7.0
        assert "mtp_1_loss_microavg" not in logs
        assert logs["mtp_2_loss"] == 8.0
        assert "mtp_2_loss_last" not in logs
        assert LanguageLoss.token_avg_tracker == {}
    finally:
        LanguageLoss.token_avg_tracker.clear()


def test_a_second_call_in_the_same_step_is_a_no_op():
    """Consumption clears the tracker, so a subclass logging twice per step
    cannot double-override (the second call sees an empty tracker)."""
    _seed_tracker()
    try:
        fake_self = _fake_trainer(True)
        logs = {"loss": 9.5}
        trainer.Trainer._apply_global_token_avg_loss(fake_self, logs)
        first = logs["loss"]
        trainer.Trainer._apply_global_token_avg_loss(fake_self, logs)

        assert logs["loss"] == first
        assert logs["loss_microavg"] == 9.5
    finally:
        LanguageLoss.token_avg_tracker.clear()


def test_a_broken_consume_warns_instead_of_killing_the_step():
    """Display-only means never fatal: a failure inside the criterion must warn
    and leave ``logs`` as the micro-average rather than abort training."""
    _seed_tracker()
    try:
        with (
            patch.object(
                LanguageLoss,
                "consume_token_avg_loss",
                side_effect=RuntimeError("boom"),
            ),
            patch.object(trainer.logger, "warning") as warning,
        ):
            logs = _apply(True, {"loss": 9.5})

        assert logs == {"loss": 9.5}, logs
        assert warning.call_count == 1
        assert "boom" in str(warning.call_args)
    finally:
        LanguageLoss.token_avg_tracker.clear()


def test_log_applies_the_override_at_its_single_call_site():
    """``log()`` is the hook point, so every Trainer subclass inherits the
    override; this pins that the call is actually wired up there."""
    _seed_tracker()
    try:
        seen = {}

        def _on_log(args, state, control, logs, **kwargs):
            seen["logs"] = dict(logs)
            return "new-control"

        fake_self = types.SimpleNamespace(
            args=None,
            timers=None,
            metrics_dumper=None,
            control="old-control",
            state=types.SimpleNamespace(epoch=None, log_history=["stale"]),
            callback_handler=types.SimpleNamespace(on_log=_on_log),
        )
        fake_self.model = types.SimpleNamespace(
            config=types.SimpleNamespace(log_global_token_avg_loss=True)
        )
        fake_self._apply_global_token_avg_loss = types.MethodType(
            trainer.Trainer._apply_global_token_avg_loss, fake_self
        )

        logs = {"loss": 9.5}
        with _NO_TIMERS:
            trainer.Trainer.log(fake_self, logs)

        assert abs(logs["loss"] - 2.12) < 1e-6, logs["loss"]
        assert logs["loss_microavg"] == 9.5
        # The callbacks see the overridden value, not the micro-average.
        assert abs(seen["logs"]["loss"] - 2.12) < 1e-6
        assert fake_self.control == "new-control"
    finally:
        LanguageLoss.token_avg_tracker.clear()


if __name__ == "__main__":
    import sys

    try:
        test_flag_off_leaves_logs_alone_and_does_not_consume()
        test_a_log_without_loss_does_not_consume()
        test_enabled_overrides_loss_and_keeps_the_microavg()
        test_empty_tracker_degrades_silently_to_the_microavg()
        test_mtp_heads_are_overridden_separately()
        test_a_second_call_in_the_same_step_is_a_no_op()
        test_a_broken_consume_warns_instead_of_killing_the_step()
        test_log_applies_the_override_at_its_single_call_site()
    except AssertionError:
        import traceback

        traceback.print_exc()
        sys.exit(1)
    print("OK")
