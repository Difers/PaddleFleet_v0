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

"""Single-card coverage for the ``calculate_per_token_loss`` loss path.

With per-token loss on, the loss layers return a raw masked token *sum*; the
trainer divides by ``T_global`` (main-head non-padded label tokens, summed over
accumulation micro-batches and all-reduced). Covered here, all on CPU with no
collectives (CP / TP sizes are patched where a guard reads them):

  * tracker bookkeeping: ``accumulate_per_token_local_count`` / ``pop_per_token_stats``,
    ``suppress_per_token_count`` / ``_per_token_count_enabled`` (recompute
    replay and MTP heads are not counted), and the display accumulators;
  * ``LanguageLoss.__init__`` config guards;
  * ``forward_impl`` (non-fused and a stubbed fused linear+CE kernel): raw sum,
    pre-CP-gather token counting, the training-only assertion;
  * ``LanguageLoss.forward`` without MTP and with MTP (use_erndata per-doc
    roll), including the MTP renormalisation to ``lm_tok / mtp_tok`` and the
    display stats from ``_record_per_token_display``;
  * ``MainLanguageLoss`` / ``MTPLanguageLoss`` separate head-loss classes.

Expected values come from independent NumPy cross-entropy, never from the code
under test.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import paddle
import paddle.nn.functional as F

import paddlefleet.models.common.language_loss.language_loss as ll_mod
from paddlefleet.models.common.language_loss.language_loss import (
    LanguageLoss,
    MainLanguageLoss,
    MTPLanguageLoss,
)

IGNORED = -100
_FUSED_CE = (
    "paddlefleet.triton_ops.fused_linear_cross_entropy."
    "LigerFusedLinearCrossEntropyFunction"
)


def _cfg(**overrides):
    base = {
        "calculate_per_token_loss": True,
        "parallel_output": False,
        "loss_subbatch_sequence_length": 0,
        "num_nextn_predict_layers": 0,
        "mtp_load_weight_only": False,
        "use_erndata": False,
        "separate_mtp_headloss": False,
        "train_mtp_only": False,
        "mtp_distillation_loss": False,
        "gpt_model_use_experimental_version": False,
        "fused_linear_ce_loss_chunk": 0,
        "experimental_dataflow": False,
        "sequence_parallel": False,
        "use_accuracy_compatible": False,
        "add_mtp_loss": True,
        "mtp_loss_scaling_factor": 0.5,
        "cp_balance_mode": "dualchunk_allgather",
        "recompute_granularity": None,
        "recompute_modules": None,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _make(cls=LanguageLoss, **overrides):
    # pg_collection is only stored; pass a dummy so no mpu groups are needed.
    return cls(_cfg(**overrides), pg_collection=object())


def _reset_counters():
    LanguageLoss.token_avg_tracker.clear()
    LanguageLoss._cu_seqlens_q_stash = None


# Per-token stats live in LanguageLoss.token_avg_tracker under ll_mod._PT_KEYS.
# These helpers pop one group at a time so a test can check the T_global count
# and the display stats separately; the trainer pops all of them together via
# LanguageLoss.pop_per_token_stats().
_DISP_KEYS = {
    "lm_sum": ("pt_lm_sum", float),
    "lm_tok": ("pt_lm_tok", int),
    "mtp_sum": ("pt_mtp_sum", float),
    "mtp_tok": ("pt_mtp_tok", int),
    "loss_mean_sum": ("pt_mom_sum", float),
    "loss_mean_cnt": ("pt_mom_cnt", int),
}


def _pop_count():
    value = LanguageLoss.token_avg_tracker.pop("pt_t_local", None)
    return 0 if value is None else int(value)


def _pop_disp():
    tr = LanguageLoss.token_avg_tracker
    out = {}
    for name, (key, cast) in _DISP_KEYS.items():
        value = tr.pop(key, None)
        out[name] = cast(0) if value is None else cast(value)
    return out


def _np_ce_sum(logits, labels):
    """Masked cross-entropy sum and valid-token count (NumPy reference)."""
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)
    m = logits.max(-1, keepdims=True)
    logp = logits - m - np.log(np.exp(logits - m).sum(-1, keepdims=True))
    mask = labels != IGNORED
    safe = np.where(mask, labels, 0)
    nll = -np.take_along_axis(logp, safe[..., None], -1)[..., 0]
    return float((nll * mask).sum()), int(mask.sum())


def _np_roll_per_doc(labels, cu_list, depth):
    """Roll left ``depth + 1`` times inside each packed doc, tail = IGNORED."""
    out = np.array(labels, copy=True)
    for _ in range(depth + 1):
        rolled = out.copy()
        for s, e in zip(cu_list[:-1], cu_list[1:]):
            seg = np.roll(out[:, s:e], -1, axis=1)
            seg[:, -1] = IGNORED
            rolled[:, s:e] = seg
        out = rolled
    return out


class _FakeFusedCE:
    """Stand-in for the triton fused linear+CE kernel: per-token CE, 'none'."""

    @staticmethod
    def apply(_input, weight, labels, bias, ignore_index, *rest):
        logits = paddle.matmul(_input, weight)
        return F.cross_entropy(
            logits, labels, ignore_index=ignore_index, reduction="none"
        ).reshape([-1])


class _Base(unittest.TestCase):
    def setUp(self):
        # Restore the device afterwards: later test modules in the same
        # process (e.g. the eval observer tests) need the original one.
        self.prev_device = paddle.get_device()
        paddle.set_device("cpu")
        paddle.seed(2026)
        _reset_counters()

    def tearDown(self):
        _reset_counters()
        paddle.set_device(self.prev_device)


class TestModuleCounters(_Base):
    def test_local_count_accumulates_and_pop_resets(self):
        ll_mod.accumulate_per_token_local_count(3)
        ll_mod.accumulate_per_token_local_count(paddle.to_tensor(4))
        # Stored as a device tensor in the tracker, not a python int.
        self.assertIsInstance(
            LanguageLoss.token_avg_tracker["pt_t_local"], paddle.Tensor
        )
        self.assertEqual(_pop_count(), 7)
        self.assertEqual(_pop_count(), 0)

    def test_pop_per_token_stats_returns_all_keys_and_resets(self):
        ll_mod.accumulate_per_token_local_count(5)
        ll_mod._per_token_tracker_add("pt_lm_sum", paddle.to_tensor(6.0))
        ll_mod._per_token_tracker_add("pt_lm_sum", 4.0)
        ll_mod._per_token_tracker_add("pt_mom_cnt", 1)
        stats = LanguageLoss.pop_per_token_stats()
        self.assertEqual(set(stats), set(ll_mod._PT_KEYS))
        for key, dtype in ll_mod._PT_KEYS.items():
            self.assertEqual(stats[key].dtype, getattr(paddle, dtype))
            self.assertEqual(stats[key].shape, [])
        self.assertEqual(int(stats["pt_t_local"]), 5)
        self.assertAlmostEqual(float(stats["pt_lm_sum"]), 10.0)
        self.assertEqual(int(stats["pt_mom_cnt"]), 1)
        # Unwritten keys come back as zeros.
        self.assertEqual(float(stats["pt_mtp_sum"]), 0.0)
        self.assertEqual(int(stats["pt_mtp_tok"]), 0)
        self.assertEqual(LanguageLoss.token_avg_tracker, {})
        empty = LanguageLoss.pop_per_token_stats()
        self.assertTrue(all(float(v) == 0.0 for v in empty.values()))

    def test_per_token_and_token_avg_keys_do_not_clear_each_other(self):
        # log_global_token_avg_loss's (S, n) are consumed per log interval and
        # the per-token keys per step; neither consumer may drop the other's.
        tr = LanguageLoss.token_avg_tracker
        tr["S"] = paddle.to_tensor(8.0)
        tr["n"] = paddle.to_tensor(4.0)
        ll_mod.accumulate_per_token_local_count(3)
        LanguageLoss.pop_per_token_stats()
        self.assertEqual(set(tr), {"S", "n"})

        ll_mod.accumulate_per_token_local_count(3)
        self.assertAlmostEqual(
            LanguageLoss.consume_token_avg_loss()["loss"], 2.0
        )
        self.assertEqual(set(tr), {"pt_t_local"})
        self.assertEqual(_pop_count(), 3)

    def test_count_enabled_respects_config_replay_and_suppress(self):
        on = _cfg()
        self.assertTrue(ll_mod._per_token_count_enabled(on))
        self.assertFalse(
            ll_mod._per_token_count_enabled(
                _cfg(calculate_per_token_loss=False)
            )
        )
        # Recompute replay must not count the same labels twice.
        token = ll_mod._token_loss_replaying.set(True)
        try:
            self.assertFalse(ll_mod._per_token_count_enabled(on))
        finally:
            ll_mod._token_loss_replaying.reset(token)
        # MTP heads are kept out of T_global, and the flag is restored on exit
        # even when the body raises.
        with (
            self.assertRaises(ValueError),
            ll_mod.suppress_per_token_count(),
        ):
            self.assertFalse(ll_mod._per_token_count_enabled(on))
            raise ValueError
        self.assertTrue(ll_mod._per_token_count_enabled(on))

    def test_display_accumulates_and_pop_resets(self):
        loss = _make(mtp_loss_scaling_factor=1.0)
        # Two microbatches, each with one MTP head.
        loss._record_per_token_display(
            paddle.to_tensor(6.0),
            paddle.to_tensor(3),
            [paddle.to_tensor(2.0)],
            [paddle.to_tensor(1)],
        )
        loss._record_per_token_display(
            paddle.to_tensor(4.0),
            paddle.to_tensor(2),
            [paddle.to_tensor(3.0)],
            [paddle.to_tensor(3)],
        )
        d = _pop_disp()
        self.assertAlmostEqual(d["lm_sum"], 10.0)
        self.assertEqual(d["lm_tok"], 5)
        self.assertAlmostEqual(d["mtp_sum"], 5.0)
        self.assertEqual(d["mtp_tok"], 4)
        # (6/3 + 1.0 * 2/1) + (4/2 + 1.0 * 3/3) = 4 + 3
        self.assertAlmostEqual(d["loss_mean_sum"], 7.0)
        self.assertEqual(d["loss_mean_cnt"], 2)
        empty = _pop_disp()
        self.assertEqual(
            empty,
            {
                "lm_sum": 0.0,
                "lm_tok": 0,
                "mtp_sum": 0.0,
                "mtp_tok": 0,
                "loss_mean_sum": 0.0,
                "loss_mean_cnt": 0,
            },
        )


class TestInitGuards(_Base):
    """``LanguageLoss.__init__`` rejects unsupported per-token layouts."""

    def _make_cp(self, cp, **overrides):
        with (
            mock.patch.object(
                ll_mod, "get_context_parallel_world_size", return_value=cp
            ),
            mock.patch.object(
                ll_mod, "get_tensor_model_parallel_world_size", return_value=1
            ),
            mock.patch("paddle.distributed.is_initialized", return_value=True),
        ):
            return _make(**overrides)

    def test_flag_requires_literal_true(self):
        self.assertTrue(_make().calculate_per_token_loss)
        self.assertFalse(
            _make(calculate_per_token_loss=1).calculate_per_token_loss
        )
        self.assertFalse(
            _make(calculate_per_token_loss=False).calculate_per_token_loss
        )

    def test_no_mtp_cp_layouts_pass(self):
        # The loader-sharded source counts display tokens on the gathered
        # labels, so CP>1 no longer requires experimental_dataflow.
        self._make_cp(2)
        self._make_cp(2, experimental_dataflow=True)
        self._make_cp(1)

    def test_mtp_guards(self):
        mtp = {"num_nextn_predict_layers": 1}
        cases = [
            ({"train_mtp_only": True}, "train_mtp_only"),
            ({"mtp_distillation_loss": True}, "mtp_distillation_loss"),
        ]
        for extra, msg in cases:
            with (
                self.subTest(extra=extra),
                self.assertRaisesRegex(AssertionError, msg),
            ):
                self._make_cp(1, **mtp, **extra)

    def test_mtp_supported_layouts_pass(self):
        mtp = {"num_nextn_predict_layers": 1}
        self._make_cp(1, use_erndata=True, **mtp)
        # Fused (non-separate) legacy L+K layout.
        self._make_cp(1, **mtp)
        self._make_cp(2, experimental_dataflow=True, **mtp)
        for chunk in (0, 128):
            self._make_cp(
                1,
                gpt_model_use_experimental_version=True,
                fused_linear_ce_loss_chunk=chunk,
                **mtp,
            )
        # mtp_load_weight_only disables MTP, so the MTP guards are skipped.
        self._make_cp(1, mtp_load_weight_only=True, **mtp)


class TestForwardImpl(_Base):
    B, S, V, H = 2, 6, 7, 5

    def _inputs(self):
        logits = paddle.randn([self.B, self.S, self.V], dtype="float32")
        labels_np = np.random.RandomState(0).randint(
            0, self.V, size=(self.B, self.S)
        )
        labels_np[0, -2:] = IGNORED
        labels_np[1, 0] = IGNORED
        return logits, paddle.to_tensor(labels_np, dtype="int64"), labels_np

    def test_returns_raw_sum_and_counts_local_tokens(self):
        logits, labels, labels_np = self._inputs()
        ref_sum, ref_tok = _np_ce_sum(logits.numpy(), labels_np)
        out = _make().forward_impl(logits, labels)
        np.testing.assert_allclose(float(out), ref_sum, rtol=1e-5)
        self.assertEqual(_pop_count(), ref_tok)
        # Raw sum skips _reduce_loss_by_tokens, so the legacy tracker is empty.
        self.assertEqual(LanguageLoss.token_avg_tracker, {})

    def test_disabled_returns_mean_and_does_not_count(self):
        logits, labels, labels_np = self._inputs()
        ref_sum, ref_tok = _np_ce_sum(logits.numpy(), labels_np)
        out = _make(calculate_per_token_loss=False).forward_impl(logits, labels)
        np.testing.assert_allclose(float(out), ref_sum / ref_tok, rtol=1e-5)
        self.assertEqual(_pop_count(), 0)

    def test_suppressed_call_is_not_counted(self):
        logits, labels, labels_np = self._inputs()
        ref_sum, _ = _np_ce_sum(logits.numpy(), labels_np)
        with ll_mod.suppress_per_token_count():
            out = _make().forward_impl(logits, labels)
        np.testing.assert_allclose(float(out), ref_sum, rtol=1e-5)
        self.assertEqual(_pop_count(), 0)

    def test_eval_forward_is_rejected(self):
        logits, labels, _ = self._inputs()
        loss = _make()
        loss.eval()
        with self.assertRaisesRegex(AssertionError, "training"):
            loss.forward_impl(logits, labels)
        # Disabled: eval forward is fine.
        legacy = _make(calculate_per_token_loss=False)
        legacy.eval()
        legacy.forward_impl(logits, labels)

    def test_fused_path_returns_raw_sum_and_counts(self):
        hidden = paddle.randn([self.B, self.S, self.H], dtype="float32")
        weight = paddle.randn([self.H, self.V], dtype="float32")
        _, labels, labels_np = self._inputs()
        ref_sum, ref_tok = _np_ce_sum(
            paddle.matmul(hidden, weight).numpy(), labels_np
        )
        loss = _make(fused_linear_ce_loss_chunk=128)
        with mock.patch(_FUSED_CE, _FakeFusedCE):
            out = loss.forward_impl((hidden, weight, None), labels)
            np.testing.assert_allclose(float(out), ref_sum, rtol=1e-5)
            self.assertEqual(_pop_count(), ref_tok)

            legacy = _make(
                calculate_per_token_loss=False, fused_linear_ce_loss_chunk=128
            )
            out = legacy.forward_impl((hidden, weight, None), labels)
            np.testing.assert_allclose(float(out), ref_sum / ref_tok, rtol=1e-5)
            self.assertEqual(_pop_count(), 0)


class TestForwardNoMTP(_Base):
    def test_raw_sum_and_display_stats(self):
        logits = paddle.randn([2, 5, 6], dtype="float32")
        labels_np = np.array(
            [[1, 2, 3, IGNORED, IGNORED], [0, 5, 4, 3, 2]], dtype="int64"
        )
        ref_sum, ref_tok = _np_ce_sum(logits.numpy(), labels_np)
        # 1/acc_steps scaling is skipped under per-token even if dsv4-compat.
        with (
            mock.patch.object(
                ll_mod, "use_dsv4_accuracy_compatible", return_value=True
            ),
            mock.patch.object(
                ll_mod.LossScaleBeforeBackward,
                "scale",
                side_effect=AssertionError("must not scale"),
            ),
        ):
            out = _make().forward(logits, paddle.to_tensor(labels_np))
        np.testing.assert_allclose(float(out), ref_sum, rtol=1e-5)
        self.assertEqual(_pop_count(), ref_tok)
        d = _pop_disp()
        np.testing.assert_allclose(d["lm_sum"], ref_sum, rtol=1e-5)
        self.assertEqual(d["lm_tok"], ref_tok)
        self.assertEqual(d["mtp_tok"], 0)
        self.assertEqual(d["loss_mean_cnt"], 1)
        np.testing.assert_allclose(
            d["loss_mean_sum"], ref_sum / ref_tok, rtol=1e-5
        )

    def test_loader_source_counts_gathered_labels(self):
        # Loader-sharded CP: labels arrive CP-local and forward_impl gathers
        # the loss sum. Emulate cp=2 by tiling (both ranks hold the same
        # shard), so the gathered sum and token count are both doubled.
        logits = paddle.randn([1, 4, 6], dtype="float32")
        labels_np = np.array([[1, 2, IGNORED, 3]], dtype="int64")
        _, ref_tok = _np_ce_sum(logits.numpy(), labels_np)
        with (
            mock.patch.object(ll_mod, "cp_shard_source", return_value="loader"),
            mock.patch.object(
                ll_mod, "to_cp_local", side_effect=lambda x, cfg, **kw: x
            ),
            mock.patch.object(
                ll_mod,
                "to_cp_global",
                side_effect=lambda x, cfg, **kw: paddle.concat([x, x], axis=1),
            ),
        ):
            _make().forward(logits, paddle.to_tensor(labels_np))
        self.assertEqual(_pop_disp()["lm_tok"], 2 * ref_tok)
        _pop_count()

    def test_disabled_leaves_display_empty(self):
        logits = paddle.randn([1, 4, 6], dtype="float32")
        labels = paddle.to_tensor([[1, 2, 3, 4]], dtype="int64")
        _make(calculate_per_token_loss=False).forward(logits, labels)
        self.assertEqual(_pop_disp()["loss_mean_cnt"], 0)


class TestForwardMTPErndata(_Base):
    """Fused ``LanguageLoss.forward`` MTP list path, use_erndata per-doc roll."""

    K, B, L, V, H = 2, 1, 8, 6, 4
    CU = [0, 3, 8]
    SCALE = 0.5

    def setUp(self):
        super().setUp()
        LanguageLoss._cu_seqlens_q_stash = paddle.to_tensor(
            self.CU, dtype="int32"
        )
        self.labels_np = np.array([[1, 2, 3, 4, IGNORED, 5, 0, 2]])

    def _reference(self, logits_np_list):
        lm_sum, lm_tok = _np_ce_sum(logits_np_list[0], self.labels_np)
        heads = [
            _np_ce_sum(
                logits_np_list[d + 1],
                _np_roll_per_doc(self.labels_np, self.CU, d),
            )
            for d in range(self.K)
        ]
        return lm_sum, lm_tok, heads

    def _check(self, out, logits_np_list, add_mtp_loss=True):
        lm_sum, lm_tok, heads = self._reference(logits_np_list)
        renorm = sum(s * lm_tok / t for s, t in heads) / self.K
        expected = lm_sum + (self.SCALE * renorm if add_mtp_loss else 0.0)
        np.testing.assert_allclose(float(out), expected, rtol=1e-5)
        # Only the main head feeds T_global.
        self.assertEqual(_pop_count(), lm_tok)
        # mtp_i_loss is displayed as a per-token mean over its own tokens, the
        # same as the separate-headloss path.
        for i, (s, t) in enumerate(heads):
            np.testing.assert_allclose(
                float(LanguageLoss.mtp_loss_tracker[f"mtp_{i + 1}_loss"]),
                s / t,
                rtol=1e-5,
            )

        d = _pop_disp()
        np.testing.assert_allclose(d["lm_sum"], lm_sum, rtol=1e-5)
        self.assertEqual(d["lm_tok"], lm_tok)
        np.testing.assert_allclose(
            d["mtp_sum"], sum(s for s, _ in heads), rtol=1e-5
        )
        self.assertEqual(d["mtp_tok"], sum(t for _, t in heads))
        combined = lm_sum / lm_tok
        if add_mtp_loss:
            combined += self.SCALE * np.mean([s / t for s, t in heads])
        np.testing.assert_allclose(d["loss_mean_sum"], combined, rtol=1e-5)

    def _make_mtp(self, **overrides):
        return _make(
            num_nextn_predict_layers=self.K,
            use_erndata=True,
            mtp_loss_scaling_factor=self.SCALE,
            **overrides,
        )

    def test_plain_branch(self):
        logits = [
            paddle.randn([self.B, self.L, self.V], dtype="float32")
            for _ in range(self.K + 1)
        ]
        out = self._make_mtp().forward(logits, paddle.to_tensor(self.labels_np))
        self._check(out, [x.numpy() for x in logits])

    def test_plain_branch_add_mtp_loss_false(self):
        logits = [
            paddle.randn([self.B, self.L, self.V], dtype="float32")
            for _ in range(self.K + 1)
        ]
        out = self._make_mtp(add_mtp_loss=False).forward(
            logits, paddle.to_tensor(self.labels_np)
        )
        self._check(out, [x.numpy() for x in logits], add_mtp_loss=False)

    def test_experimental_version_fused_branch(self):
        hidden = [
            paddle.randn([self.B, self.L, self.H], dtype="float32")
            for _ in range(self.K + 1)
        ]
        weight = paddle.randn([self.H, self.V], dtype="float32")
        loss = self._make_mtp(
            gpt_model_use_experimental_version=True,
            fused_linear_ce_loss_chunk=128,
        )
        with mock.patch(_FUSED_CE, _FakeFusedCE):
            out = loss.forward(
                [(h, weight, None) for h in hidden],
                paddle.to_tensor(self.labels_np),
            )
        self._check(out, [paddle.matmul(h, weight).numpy() for h in hidden])

    def test_experimental_version_unfused_branch(self):
        # The experimental per-token matrix reduction must return the raw
        # masked sum (not the per-head mean) under per-token loss.
        logits = [
            paddle.randn([self.B, self.L, self.V], dtype="float32")
            for _ in range(self.K + 1)
        ]
        out = self._make_mtp(gpt_model_use_experimental_version=True).forward(
            logits, paddle.to_tensor(self.labels_np)
        )
        self._check(out, [x.numpy() for x in logits])

    def test_experimental_version_accuracy_kernel_keeps_mtp_out_of_value(self):
        hidden = [
            paddle.randn([self.B, self.L, self.H], dtype="float32")
            for _ in range(self.K + 1)
        ]
        weight = paddle.randn([self.H, self.V], dtype="float32")
        loss = self._make_mtp(
            gpt_model_use_experimental_version=True,
            fused_linear_ce_loss_chunk=128,
        )
        with (
            mock.patch(_FUSED_CE, _FakeFusedCE),
            mock.patch.object(
                ll_mod, "_use_accuracy_compatible_kernel", return_value=True
            ),
        ):
            out = loss.forward(
                [(h, weight, None) for h in hidden],
                paddle.to_tensor(self.labels_np),
            )
        # MTP only flows through the gradient: value and display are lm only.
        self._check(
            out,
            [paddle.matmul(h, weight).numpy() for h in hidden],
            add_mtp_loss=False,
        )

    def test_display_counts_full_sequence_under_cp(self):
        # Emulate one CP rank: to_cp_local keeps the first half of the
        # sequence, to_cp_global is the identity (no peer to gather from). The
        # display token counts must still be taken on the full labels, since
        # in a real run the sums they divide are CP-gathered.
        half = self.L // 2
        logits = [
            paddle.randn([self.B, half, self.V], dtype="float32")
            for _ in range(self.K + 1)
        ]
        with (
            mock.patch.object(
                ll_mod,
                "to_cp_local",
                side_effect=lambda x, cfg, **kw: x[:, :half],
            ),
            mock.patch.object(
                ll_mod, "to_cp_global", side_effect=lambda x, cfg, **kw: x
            ),
        ):
            self._make_mtp().forward(logits, paddle.to_tensor(self.labels_np))
        full_lm_tok = int((self.labels_np != IGNORED).sum())
        full_mtp_tok = sum(
            int((_np_roll_per_doc(self.labels_np, self.CU, d) != IGNORED).sum())
            for d in range(self.K)
        )
        d = _pop_disp()
        self.assertEqual(d["lm_tok"], full_lm_tok)
        self.assertEqual(d["mtp_tok"], full_mtp_tok)
        _pop_count()


class TestForwardMTPLegacy(TestForwardMTPErndata):
    """Fused ``LanguageLoss.forward`` MTP list path, legacy L+K slice layout.

    Reuses the erndata checks with the reference swapped: labels carry K
    trailing positions, the main head uses ``labels[:, :L]`` and depth d the
    window ``[d + 1, d + 1 + L)``. In particular the main-head token count
    (renorm numerator and display) must not include the trailing K positions.
    """

    def setUp(self):
        _Base.setUp(self)
        # Valid labels in the trailing K positions, so counting them shows up.
        self.labels_np = np.array([[1, 2, 3, 4, IGNORED, 5, 0, 2, 3, 1]])

    def _reference(self, logits_np_list):
        lm_sum, lm_tok = _np_ce_sum(
            logits_np_list[0], self.labels_np[:, : self.L]
        )
        heads = [
            _np_ce_sum(
                logits_np_list[d + 1],
                self.labels_np[:, d + 1 : d + 1 + self.L],
            )
            for d in range(self.K)
        ]
        return lm_sum, lm_tok, heads

    def _make_mtp(self, **overrides):
        return _make(
            num_nextn_predict_layers=self.K,
            mtp_loss_scaling_factor=self.SCALE,
            **overrides,
        )

    def test_display_counts_full_sequence_under_cp(self):
        self.skipTest("CP emulation is covered by the erndata layout.")

    def test_experimental_fused_never_cp_gathers_a_scalar(self):
        # The per-token fused head returns a 0-d raw sum; the legacy
        # experimental reduction used to CP-gather it along axis 1.
        hidden = [
            paddle.randn([self.B, self.L, self.H], dtype="float32")
            for _ in range(self.K + 1)
        ]
        weight = paddle.randn([self.H, self.V], dtype="float32")
        loss = self._make_mtp(
            gpt_model_use_experimental_version=True,
            fused_linear_ce_loss_chunk=128,
        )
        gathered_ndims = []

        def record_gather(x, cfg, **kw):
            gathered_ndims.append(x.ndim)
            return x

        with (
            mock.patch(_FUSED_CE, _FakeFusedCE),
            mock.patch.object(
                ll_mod, "to_cp_global", side_effect=record_gather
            ),
        ):
            out = loss.forward(
                [(h, weight, None) for h in hidden],
                paddle.to_tensor(self.labels_np),
            )
        self.assertTrue(gathered_ndims)
        self.assertNotIn(0, gathered_ndims)
        self._check(out, [paddle.matmul(h, weight).numpy() for h in hidden])


class TestRecordPerTokenDisplay(_Base):
    def test_train_mtp_only_float_lm_and_accuracy_kernel(self):
        loss = _make(mtp_loss_scaling_factor=1.0)
        mtp = [paddle.to_tensor(6.0), paddle.to_tensor(4.0)]
        toks = [paddle.to_tensor(3), paddle.to_tensor(0)]
        # lm_loss is the python float 0.0 under train_mtp_only.
        loss._record_per_token_display(0.0, paddle.to_tensor(0), mtp, toks)
        d = _pop_disp()
        self.assertEqual((d["lm_sum"], d["lm_tok"]), (0.0, 0))
        self.assertAlmostEqual(d["mtp_sum"], 10.0)
        self.assertEqual(d["mtp_tok"], 3)
        # Zero-token head is clipped to 1: mean(6/3, 4/1) = 3.
        self.assertAlmostEqual(d["loss_mean_sum"], 3.0)

        # Accuracy-compatible kernel keeps MTP out of the value -> lm mean only.
        with mock.patch.object(
            ll_mod, "_use_accuracy_compatible_kernel", return_value=True
        ):
            loss._record_per_token_display(
                paddle.to_tensor(8.0), paddle.to_tensor(4), mtp, toks
            )
        self.assertAlmostEqual(_pop_disp()["loss_mean_sum"], 2.0)


class TestSeparateHeadLoss(_Base):
    """``MainLanguageLoss`` / ``MTPLanguageLoss`` under per-token loss."""

    K, B, L, V = 2, 1, 6, 5
    SCALE = 0.5

    def tearDown(self):
        MainLanguageLoss.mtp_loss_tracker.clear()
        LanguageLoss.mtp_loss_tracker.clear()
        super().tearDown()

    def _make_sep(self, cls, **overrides):
        return _make(
            cls,
            num_nextn_predict_layers=self.K,
            separate_mtp_headloss=True,
            mtp_loss_scaling_factor=self.SCALE,
            **overrides,
        )

    def _check_main(self, out, lm_sum, lm_tok, mtp_sums, mtp_toks):
        renorm = (
            sum(s * lm_tok / t for s, t in zip(mtp_sums, mtp_toks)) / self.K
        )
        np.testing.assert_allclose(
            float(out), lm_sum + self.SCALE * renorm, rtol=1e-5
        )
        self.assertEqual(_pop_count(), lm_tok)
        # mtp_i_loss is displayed as a per-token mean over its own tokens.
        for i, (s, t) in enumerate(zip(mtp_sums, mtp_toks)):
            np.testing.assert_allclose(
                float(MainLanguageLoss.mtp_loss_tracker[f"mtp_{i + 1}_loss"]),
                s / t,
                rtol=1e-6,
            )
        d = _pop_disp()
        self.assertEqual(d["lm_tok"], lm_tok)
        self.assertEqual(d["mtp_tok"], sum(mtp_toks))
        np.testing.assert_allclose(d["mtp_sum"], sum(mtp_sums), rtol=1e-6)

    def test_main_ernie5_slice_layout(self):
        # labels are [B, L + K]; the main head uses labels[:, :-K].
        labels_np = np.array([[1, 2, IGNORED, 3, 4, 0, IGNORED, 2]])
        logits = paddle.randn([self.B, self.L, self.V], dtype="float32")
        mtp_sums = [3.0, 5.0]
        mtp_toks = [
            int((labels_np[:, d + 1 : d + 1 + self.L] != IGNORED).sum())
            for d in range(self.K)
        ]
        lm_sum, lm_tok = _np_ce_sum(logits.numpy(), labels_np[:, : -self.K])
        logs = mock.MagicMock()
        with mock.patch.object(
            ll_mod, "get_global_training_logs", return_value=logs
        ):
            out = self._make_sep(MainLanguageLoss).forward(
                {
                    "logits": logits,
                    "mtp_loss": [paddle.to_tensor(s) for s in mtp_sums],
                },
                paddle.to_tensor(labels_np),
            )
        self._check_main(out, lm_sum, lm_tok, mtp_sums, mtp_toks)
        # global_training_logs gets the same per-token means as the tracker.
        logged = {
            k: float(v)
            for c in logs.update.call_args_list
            for k, v in c.kwargs.items()
        }
        for i, (s, t) in enumerate(zip(mtp_sums, mtp_toks)):
            self.assertAlmostEqual(logged[f"mtp_{i + 1}_loss"], s / t, places=6)

    def test_main_erndata_roll_layout(self):
        cu = [0, 2, 6]
        LanguageLoss._cu_seqlens_q_stash = paddle.to_tensor(cu, dtype="int32")
        labels_np = np.array([[1, 2, 3, IGNORED, 4, 0]])
        logits = paddle.randn([self.B, self.L, self.V], dtype="float32")
        mtp_sums = [3.0, 5.0]
        mtp_toks = [
            int((_np_roll_per_doc(labels_np, cu, d) != IGNORED).sum())
            for d in range(self.K)
        ]
        lm_sum, lm_tok = _np_ce_sum(logits.numpy(), labels_np)
        out = self._make_sep(MainLanguageLoss, use_erndata=True).forward(
            {
                "logits": logits,
                "mtp_loss": [paddle.to_tensor(s) for s in mtp_sums],
            },
            paddle.to_tensor(labels_np),
        )
        self._check_main(out, lm_sum, lm_tok, mtp_sums, mtp_toks)

    def test_mtp_heads_return_raw_sums_and_are_not_counted(self):
        labels_np = np.array([[1, 2, IGNORED, 3, 4, 0, IGNORED, 2]])
        mtp_logits = [
            paddle.randn([self.B, self.L, self.V], dtype="float32")
            for _ in range(self.K)
        ]
        out = self._make_sep(MTPLanguageLoss).forward(
            {"mtp_logits": mtp_logits, "labels": paddle.to_tensor(labels_np)}
        )
        for d in range(self.K):
            ref_sum, _ = _np_ce_sum(
                mtp_logits[d].numpy(), labels_np[:, d + 1 : d + 1 + self.L]
            )
            np.testing.assert_allclose(
                float(out["mtp_loss"][d]), ref_sum, rtol=1e-5
            )
        self.assertEqual(_pop_count(), 0)


if __name__ == "__main__":
    unittest.main()
