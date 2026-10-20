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

"""Unit tests for the DSpark loss engine (``utils/dspark_losses.py``).

The end-to-end tests drive the *streamed* teacher-forced path through a real
``DsparkPredictionLayer``.  This module targets what they cannot reach:

* the **materialized** path (``_materialized_token_losses`` /
  ``_materialized_selector_terms``), taken whenever draft logits already exist;
* the **DPACE** weighting policy (``_dpace_weights``), the alternative to the
  default exponential decay;
* the **candidate-selector** objective, materialized and streamed, which
  re-ranks a strict unary top-k set and is normalized on covered tokens only;
* the validation branches (``LossProfile.validate``, ``_validate_output``,
  ``_validate_streamed_output``, and the guards inside
  ``_streamed_token_losses`` / ``compute_dspark_loss``) that must fail loudly
  instead of silently training the wrong objective;
* the fused-kernel availability guard, which must fall back to eager rather
  than crash on a CPU / no-Triton host.

Materialized and streamed are also cross-checked against each other: with the
same hidden states and projection weight, projecting up front and streaming the
projection must agree.  That pins both implementations to one definition, so a
later edit to either one cannot drift unnoticed.
"""

import sys
import types
import unittest

import numpy as np
import paddle

from paddlefleet.training.initialize import initialize_fleet
from paddlefleet.transformer.parallel_multi_token_prediction import (
    CandidateSelector,
)
from paddlefleet.utils.dspark import dspark_losses as dl
from paddlefleet.utils.dspark.dspark_losses import (
    LossProfile,
    compute_dspark_loss,
    effective_selector_alpha,
)
from paddlefleet.utils.dspark.dspark_utils import (
    DrafterOutput,
    ReadOnlyProjection,
)

initialize_fleet(strategy=paddle.distributed.fleet.DistributedStrategy())

_B, _A, _K = 2, 3, 4
_H, _V, _R = 16, 32, 8
_TOKENS = _B * _A * _K
_ATOL = 1e-4
_RTOL = 1e-4


def _fields(seed=0, *, dtype="int64", drop_last=True):
    """target_ids / eval_mask / block_keep_mask, optionally dropping a block.

    Dropping the last block exercises the contract that a filtered block leaves
    both the numerator and the denominator of the loss.
    """
    paddle.seed(seed)
    target_ids = paddle.randint(1, _V, [_B, _A, _K], dtype=dtype)
    eval_mask = paddle.ones([_B, _A, _K], dtype="bool")
    block_keep = paddle.ones([_B, _A], dtype="bool")
    if drop_last:
        eval_mask[_B - 1, _A - 1, _K - 1] = False
        block_keep[_B - 1, _A - 1] = False
    return target_ids, eval_mask, block_keep


def _selector(seed=0, *, top_k=4, state_rank=_R, vocab_size=_V):
    """A real ``CandidateSelector``: the loss reads its codebooks directly."""
    paddle.seed(seed)
    return CandidateSelector(
        hidden_size=_H,
        vocab_size=vocab_size,
        state_rank=state_rank,
        top_k=top_k,
        initializer_range=0.02,
    )


def _materialized_output(
    seed=0,
    *,
    fields=None,
    with_teacher=True,
    teacher_logits=None,
    draft_logits=None,
    draft_hidden_states=None,
    previous_token_ids=None,
    confidence_pred=None,
    selector=None,
    markov_head=None,
    dtype="int64",
):
    """A ``DrafterOutput`` carrying full-vocabulary draft logits."""
    if fields is None:
        fields = _fields(seed, dtype=dtype)
    target_ids, eval_mask, block_keep = fields
    paddle.seed(seed + 1)
    if draft_logits is None:
        draft_logits = paddle.randn([_B, _A, _K, _V], dtype="float32")
    draft_logits.stop_gradient = False
    if with_teacher and teacher_logits is None:
        teacher_logits = paddle.randn([_B, _A, _K, _V], dtype="float32")
    return DrafterOutput(
        draft_logits=draft_logits,
        target_ids=target_ids,
        eval_mask=eval_mask,
        block_keep_mask=block_keep,
        confidence_pred=confidence_pred,
        aligned_target_logits=teacher_logits,
        draft_hidden_states=draft_hidden_states,
        previous_token_ids=previous_token_ids,
        candidate_selector=selector,
        markov_head=markov_head,
    )


def _streamed_output(
    seed=0,
    *,
    fields=None,
    with_teacher=True,
    draft_hidden_states=None,
    target_hidden_states=None,
    previous_token_ids=None,
    projection=None,
    transpose_y=True,
    markov_head=None,
    selector=None,
    hidden_dim=None,
    dtype="int64",
):
    """A ``DrafterOutput`` that must project its hidden states on the fly."""
    if fields is None:
        fields = _fields(seed, dtype=dtype)
    target_ids, eval_mask, block_keep = fields
    paddle.seed(seed + 1)
    hidden_dim = _H if hidden_dim is None else hidden_dim
    if draft_hidden_states is None:
        draft_hidden_states = paddle.randn(
            [_B, _A, _K, hidden_dim], dtype="float32"
        )
    if target_hidden_states is None:
        target_hidden_states = paddle.randn(
            [_B, _A, _K, hidden_dim], dtype="float32"
        )
    draft_hidden_states.stop_gradient = False
    target_hidden_states.stop_gradient = True
    if previous_token_ids is None:
        previous_token_ids = paddle.randint(0, _V, [_B, _A, _K], dtype="int64")
    if projection is None:
        weight = paddle.randn(
            [_V, hidden_dim] if transpose_y else [hidden_dim, _V],
            dtype="float32",
        )
        projection = ReadOnlyProjection(weight=weight, transpose_y=transpose_y)
    return DrafterOutput(
        draft_logits=None,
        target_ids=target_ids,
        eval_mask=eval_mask,
        block_keep_mask=block_keep,
        draft_hidden_states=draft_hidden_states,
        aligned_target_hidden_states=target_hidden_states,
        previous_token_ids=previous_token_ids,
        logits_projection=projection,
        markov_head=markov_head,
        candidate_selector=selector,
    )


class TestLossProfileValidation(unittest.TestCase):
    """Every rejected field must raise instead of training a wrong objective."""

    def test_accepts_the_default_profile(self):
        self.assertIsInstance(LossProfile().validate(), LossProfile)

    def test_rejects_invalid_fields(self):
        cases = {
            "weight_type": {"weight_type": "cosine"},
            "dpace_alpha_high": {"dpace_alpha": 1.5},
            "dpace_alpha_negative": {"dpace_alpha": -0.5},
            "chunk_tokens_zero": {"probability_chunk_tokens": 0},
            "chunk_tokens_bool": {"probability_chunk_tokens": True},
            "chunk_tokens_float": {"probability_chunk_tokens": 1.5},
            "ce_alpha_negative": {"ce_alpha": -1.0},
            "l1_alpha_not_finite": {"probability_l1_alpha": float("inf")},
            "confidence_alpha_negative": {"confidence_alpha": -0.5},
            "selector_alpha_negative": {"selector_loss_alpha": -1.0},
            "stop_gradient_not_bool": {"selector_stop_gradient": 1},
            "fused_not_bool": {"use_fused_distill_loss": "yes"},
            "warmup_out_of_range": {"selector_warmup_ratio": 1.5},
            "ramp_out_of_range": {"selector_ramp_ratio": -0.1},
            "decay_gamma_zero": {"decay_gamma": 0.0},
            "decay_gamma_negative": {"decay_gamma": -1.0},
            "subbatch_float": {"subbatch_sequence_length": 2.5},
            "subbatch_bool": {"subbatch_sequence_length": True},
        }
        for name, kwargs in cases.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                LossProfile(**kwargs).validate()

    def test_accepts_the_documented_off_switches(self):
        profile = LossProfile(
            decay_gamma=None, subbatch_sequence_length=0
        ).validate()
        self.assertIsNone(profile.decay_gamma)


class TestEffectiveSelectorAlpha(unittest.TestCase):
    """Warmup / ramp resolution on optimizer-step coordinates."""

    @staticmethod
    def _profile(alpha=0.8, warmup=0.1, ramp=0.2):
        return LossProfile(
            selector_loss_alpha=alpha,
            selector_warmup_ratio=warmup,
            selector_ramp_ratio=ramp,
        )

    def test_disabled_alpha_short_circuits(self):
        profile = self._profile(alpha=0.0)
        self.assertEqual(
            effective_selector_alpha(profile, global_step=0, horizon=100), 0.0
        )

    def test_non_positive_horizon_returns_the_configured_alpha(self):
        profile = self._profile()
        self.assertAlmostEqual(
            effective_selector_alpha(profile, global_step=0, horizon=0), 0.8
        )

    def test_warmup_then_linear_ramp_then_plateau(self):
        profile = self._profile()
        # horizon=100 -> 10 warmup steps followed by a 20-step ramp.
        self.assertEqual(
            effective_selector_alpha(profile, global_step=9, horizon=100), 0.0
        )
        self.assertAlmostEqual(
            effective_selector_alpha(profile, global_step=20, horizon=100),
            0.4,
        )
        self.assertAlmostEqual(
            effective_selector_alpha(profile, global_step=29, horizon=100),
            0.76,
        )
        self.assertAlmostEqual(
            effective_selector_alpha(profile, global_step=30, horizon=100),
            0.8,
        )

    def test_negative_step_clamps_to_zero(self):
        profile = self._profile()
        self.assertEqual(
            effective_selector_alpha(profile, global_step=-5, horizon=100), 0.0
        )

    def test_zero_ramp_reaches_the_configured_alpha_after_warmup(self):
        profile = self._profile(warmup=0.0, ramp=0.0)
        self.assertAlmostEqual(
            effective_selector_alpha(profile, global_step=1, horizon=100), 0.8
        )


class TestLossRanges(unittest.TestCase):
    """Sub-batch tiling must partition the sequence without overlap or gap."""

    def test_untiled_when_subbatch_is_disabled_or_too_large(self):
        self.assertEqual(dl._loss_ranges(10, 0), [(0, 10)])
        self.assertEqual(dl._loss_ranges(10, -1), [(0, 10)])
        self.assertEqual(dl._loss_ranges(10, 10), [(0, 10)])
        self.assertEqual(dl._loss_ranges(10, 99), [(0, 10)])

    def test_chunks_cover_the_sequence_exactly(self):
        self.assertEqual(dl._loss_ranges(10, 4), [(0, 4), (4, 8), (8, 10)])
        self.assertEqual(dl._loss_ranges(9, 3), [(0, 3), (3, 6), (6, 9)])

    def test_rejects_an_empty_sequence(self):
        with self.assertRaises(ValueError):
            dl._loss_ranges(0, 4)


class TestMaterializedPath(unittest.TestCase):
    """``_materialized_token_losses`` and the chunked ``_probability_l1``."""

    def test_tiling_is_numerically_transparent(self):
        output = _materialized_output(seed=3)
        chunked = compute_dspark_loss(
            output,
            LossProfile(probability_chunk_tokens=5, subbatch_sequence_length=3),
            return_terms=True,
        )
        untiled = compute_dspark_loss(
            output,
            LossProfile(
                probability_chunk_tokens=4096, subbatch_sequence_length=0
            ),
            return_terms=True,
        )
        for name in ("ce_loss", "l1_loss", "loss"):
            self.assertAlmostEqual(
                float(chunked.terms[name]),
                float(untiled.terms[name]),
                delta=_ATOL,
                msg=f"{name} must not depend on the memory-tiling knobs",
            )

    def test_without_a_teacher_the_l1_term_is_absent(self):
        output = _materialized_output(seed=4, with_teacher=False)
        terms = compute_dspark_loss(
            output, LossProfile(probability_l1_alpha=0.0), return_terms=True
        )
        self.assertEqual(float(terms.terms["l1_loss"]), 0.0)
        self.assertNotIn("accept_rate_per_pos_num", terms.terms)

    def test_confidence_head_is_trained_on_the_accept_rate(self):
        output = _materialized_output(
            seed=5,
            confidence_pred=paddle.randn([_B, _A, _K], dtype="float32"),
        )
        terms = compute_dspark_loss(output, LossProfile(), return_terms=True)
        self.assertGreater(float(terms.terms["confidence_loss"]), 0.0)
        self.assertIn("accept_rate_per_pos_num", terms.terms)
        self.assertIn("tau_num", terms.terms)


class TestWeightingPolicies(unittest.TestCase):
    """Exponential decay, its ``None`` off switch, and the DPACE alternative."""

    def test_decay_and_dpace_weight_differently(self):
        output = _materialized_output(seed=6)
        decay = compute_dspark_loss(
            output, LossProfile(weight_type="decay"), return_terms=True
        )
        dpace = compute_dspark_loss(
            output,
            LossProfile(weight_type="dpace", dpace_alpha=0.5),
            return_terms=True,
        )
        self.assertNotAlmostEqual(
            float(decay.terms["weight_denominator"]),
            float(dpace.terms["weight_denominator"]),
            places=4,
        )
        self.assertNotAlmostEqual(
            float(decay.terms["loss"]), float(dpace.terms["loss"]), places=4
        )

    def test_decay_gamma_none_disables_the_position_decay(self):
        fields = _fields(seed=7, drop_last=True)
        output = _materialized_output(seed=7, fields=fields)
        plain = compute_dspark_loss(
            output, LossProfile(decay_gamma=None), return_terms=True
        )
        decayed = compute_dspark_loss(
            output, LossProfile(decay_gamma=4.0), return_terms=True
        )
        # Without a decay the denominator is exactly the valid-token count: the
        # dropped block removes all _K of its tokens.
        valid = fields[1] & fields[2].unsqueeze(-1)
        self.assertEqual(int(valid.astype("float32").sum()), _TOKENS - _K)
        self.assertAlmostEqual(
            float(plain.terms["weight_denominator"]),
            float(_TOKENS - _K),
            delta=1e-4,
        )
        self.assertNotAlmostEqual(
            float(plain.terms["weight_denominator"]),
            float(decayed.terms["weight_denominator"]),
            places=4,
        )


class TestMaterializedMatchesStreamed(unittest.TestCase):
    """One definition, two implementations: project first vs project on the fly."""

    def test_same_projection_gives_the_same_loss(self):
        seed = 11
        paddle.seed(seed)
        hidden = paddle.randn([_B, _A, _K, _H], dtype="float32")
        teacher_hidden = paddle.randn([_B, _A, _K, _H], dtype="float32")
        weight = paddle.randn([_V, _H], dtype="float32")
        previous = paddle.randint(0, _V, [_B, _A, _K], dtype="int64")
        fields = _fields(seed, drop_last=True)
        projection = ReadOnlyProjection(weight=weight, transpose_y=True)
        profile = LossProfile(
            probability_chunk_tokens=7, subbatch_sequence_length=3
        )

        streamed = compute_dspark_loss(
            _streamed_output(
                seed,
                fields=fields,
                draft_hidden_states=hidden,
                target_hidden_states=teacher_hidden,
                previous_token_ids=previous,
                projection=projection,
            ),
            profile,
            return_terms=True,
        )
        materialized = compute_dspark_loss(
            _materialized_output(
                seed,
                fields=fields,
                draft_logits=paddle.matmul(hidden, weight, transpose_y=True),
                teacher_logits=paddle.matmul(
                    teacher_hidden, weight, transpose_y=True
                ),
            ),
            profile,
            return_terms=True,
        )
        for name in ("ce_loss", "l1_loss", "loss", "weight_denominator"):
            self.assertAlmostEqual(
                float(streamed.terms[name]),
                float(materialized.terms[name]),
                delta=_ATOL,
                msg=f"{name} differs between the streamed and materialized paths",
            )


class TestSelectorObjective(unittest.TestCase):
    """The candidate selector re-ranks a strict unary top-k set."""

    @staticmethod
    def _peaked(target_ids, covered_blocks=None):
        """Logits whose unary top-1 is the true token on the given blocks.

        ``covered_blocks=None`` peaks every block; passing a subset leaves the
        rest noisy, so only that subset lands in the unary top-k.
        """
        peaked = (
            paddle.nn.functional.one_hot(target_ids, _V).astype("float32")
            * 10.0
        )
        if covered_blocks is None:
            return peaked
        keep = paddle.zeros([_A], dtype="float32")
        for index in covered_blocks:
            keep[index] = 1.0
        mask = keep.reshape([1, _A, 1, 1])
        noise = paddle.randn([_B, _A, _K, _V], dtype="float32") * 0.01
        return noise * (1.0 - mask) + peaked * mask

    @staticmethod
    def _previous():
        return paddle.randint(0, _V, [_B, _A, _K], dtype="int64")

    def _selector_output(self, seed, fields, draft_logits):
        return _materialized_output(
            seed=seed,
            fields=fields,
            draft_logits=draft_logits,
            selector=_selector(seed=seed),
            draft_hidden_states=paddle.randn([_B, _A, _K, _H]),
            previous_token_ids=self._previous(),
        )

    def test_materialized_selector_is_exact_on_peaked_logits(self):
        fields = _fields(seed=9, drop_last=True)
        output = self._selector_output(
            seed=9, fields=fields, draft_logits=self._peaked(fields[0])
        )
        terms = compute_dspark_loss(
            output, LossProfile(selector_loss_alpha=0.5), return_terms=True
        )
        # successor_codebook starts at zero, so the re-ranking is a no-op and
        # the unary top-1 wins on every covered token.
        self.assertAlmostEqual(
            float(terms.terms["selector_accuracy_num"]),
            float(terms.terms["selector_accuracy_den"]),
        )
        self.assertAlmostEqual(
            float(terms.terms["selector_coverage_num"]),
            float(terms.terms["selector_coverage_den"]),
        )
        self.assertGreater(float(terms.terms["selector_probability_num"]), 0.0)
        self.assertAlmostEqual(float(terms.terms["selector_loss_alpha"]), 0.5)
        # With full coverage the selector denominator spans every valid token.
        self.assertAlmostEqual(
            float(terms.terms["selector_weight_denominator"]),
            float(terms.terms["weight_denominator"]),
            delta=1e-4,
        )
        self.assertGreater(float(terms.terms["l1_loss"]), 0.0)

    def test_selector_denominator_tracks_coverage(self):
        fields = _fields(seed=11, drop_last=True)
        output = self._selector_output(
            seed=11,
            fields=fields,
            draft_logits=self._peaked(fields[0], covered_blocks=[0]),
        )
        terms = compute_dspark_loss(
            output, LossProfile(selector_loss_alpha=0.5), return_terms=True
        )
        coverage = float(terms.terms["selector_coverage_num"]) / float(
            terms.terms["selector_coverage_den"]
        )
        self.assertGreater(coverage, 0.0)
        self.assertLess(coverage, 1.0)
        # Only covered tokens carry selector weight, so its denominator is a
        # strict subset of the base objective's.
        self.assertLess(
            float(terms.terms["selector_weight_denominator"]),
            float(terms.terms["weight_denominator"]),
        )

    def test_materialized_selector_can_stop_the_gradient(self):
        selector = _selector(seed=10)
        output = _materialized_output(
            seed=10,
            draft_logits=self._peaked(_fields(seed=10)[0]),
            selector=selector,
            draft_hidden_states=paddle.randn([_B, _A, _K, _H]),
            previous_token_ids=self._previous(),
        )
        terms = compute_dspark_loss(
            output,
            LossProfile(selector_loss_alpha=0.5, selector_stop_gradient=True),
            return_terms=True,
        )
        terms.loss.backward()
        # The codebooks are the selector's own parameters and keep training;
        # only the draft-side inputs are detached.
        self.assertIsNotNone(selector.predecessor_codebook.grad)
        self.assertIsNone(output.draft_hidden_states.grad)

    def test_streamed_selector_matches_the_materialized_terms(self):
        seed = 12
        selector = _selector(seed=13)
        fields = _fields(seed=seed, drop_last=True)
        paddle.seed(seed)
        hidden = paddle.randn([_B, _A, _K, _H])
        teacher_hidden = paddle.randn([_B, _A, _K, _H])
        weight = paddle.randn([_V, _H])
        previous = self._previous()
        profile = LossProfile(
            selector_loss_alpha=0.5, subbatch_sequence_length=3
        )

        streamed = compute_dspark_loss(
            _streamed_output(
                seed,
                fields=fields,
                draft_hidden_states=hidden,
                target_hidden_states=teacher_hidden,
                previous_token_ids=previous,
                projection=ReadOnlyProjection(weight=weight, transpose_y=True),
                selector=selector,
            ),
            profile,
            return_terms=True,
        )
        materialized = compute_dspark_loss(
            _materialized_output(
                seed,
                fields=fields,
                draft_logits=paddle.matmul(hidden, weight, transpose_y=True),
                teacher_logits=paddle.matmul(
                    teacher_hidden, weight, transpose_y=True
                ),
                draft_hidden_states=hidden,
                previous_token_ids=previous,
                selector=selector,
            ),
            profile,
            return_terms=True,
        )
        for name in (
            "selector_loss",
            "selector_accuracy_num",
            "selector_coverage_num",
            "selector_probability_num",
        ):
            self.assertAlmostEqual(
                float(streamed.terms[name]),
                float(materialized.terms[name]),
                delta=_ATOL,
                msg=f"{name} differs between the two selector paths",
            )

    def test_streamed_selector_backward_reaches_the_codebooks(self):
        selector = _selector(seed=14)
        output = _streamed_output(seed=15, selector=selector)
        terms = compute_dspark_loss(
            output, LossProfile(selector_loss_alpha=0.5), return_terms=True
        )
        terms.loss.backward()
        self.assertIsNotNone(output.draft_hidden_states.grad)
        self.assertIsNotNone(selector.predecessor_codebook.grad)
        self.assertIsNotNone(selector.successor_codebook.grad)

    def test_streamed_selector_stop_gradient_detaches_the_unary_scores(self):
        selector = _selector(seed=16)
        output = _streamed_output(seed=17, selector=selector)
        terms = compute_dspark_loss(
            output,
            LossProfile(selector_loss_alpha=0.5, selector_stop_gradient=True),
            return_terms=True,
        )
        terms.loss.backward()
        self.assertIsNotNone(selector.predecessor_codebook.grad)


class TestSelectorValidation(unittest.TestCase):
    """The selector terms are defensive about their shape contract."""

    def test_materialized_selector_requires_hidden_and_previous_ids(self):
        output = _materialized_output(seed=20, selector=_selector(seed=20))
        with self.assertRaises(ValueError):
            compute_dspark_loss(output, LossProfile(selector_loss_alpha=0.5))

    def test_materialized_selector_rejects_a_hidden_shape_mismatch(self):
        output = _materialized_output(
            seed=21,
            selector=_selector(seed=21),
            draft_hidden_states=paddle.randn([_B, _A, _K + 1, _H]),
            previous_token_ids=paddle.randint(
                0, _V, [_B, _A, _K], dtype="int64"
            ),
        )
        with self.assertRaises(ValueError):
            compute_dspark_loss(output, LossProfile(selector_loss_alpha=0.5))

    def test_private_entry_point_also_guards_its_shapes(self):
        selector = _selector(seed=23)
        target_ids, eval_mask, block_keep = _fields(seed=23)
        draft_logits = paddle.randn([_B, _A, _K, _V])
        base = {
            "draft_logits": draft_logits,
            "target_ids": target_ids,
            "eval_mask": eval_mask,
            "block_keep_mask": block_keep,
            "draft_hidden_states": paddle.randn([_B, _A, _K, _H]),
            "previous_token_ids": paddle.randint(
                0, _V, [_B, _A, _K], dtype="int64"
            ),
            "candidate_selector": selector,
        }
        # compute_dspark_loss pre-checks these; assert the inner helper does too.
        for broken in (
            {"draft_hidden_states": paddle.randn([_B, _A, _K + 1, _H])},
            {
                "previous_token_ids": paddle.randint(
                    0, _V, [_B, _A, _K + 1], dtype="int64"
                )
            },
        ):
            with (
                self.subTest(field=next(iter(broken))),
                self.assertRaises(ValueError),
            ):
                dl._materialized_selector_terms(
                    DrafterOutput(**{**base, **broken}),
                    draft_logits,
                    target_ids,
                )

    def test_absent_selector_returns_zero_terms(self):
        target_ids, eval_mask, block_keep = _fields(seed=24)
        draft_logits = paddle.randn([_B, _A, _K, _V])
        output = DrafterOutput(
            draft_logits=draft_logits,
            target_ids=target_ids,
            eval_mask=eval_mask,
            block_keep_mask=block_keep,
        )
        ce, covered, correct = dl._materialized_selector_terms(
            output, draft_logits, target_ids
        )
        for term in (ce, covered, correct):
            self.assertEqual(float(term.sum()), 0.0)


class TestOutputValidation(unittest.TestCase):
    """Malformed ``DrafterOutput`` must be rejected before any math runs."""

    def _rejected(self, **overrides):
        target_ids, eval_mask, block_keep = _fields(seed=30)
        kwargs = {
            "draft_logits": paddle.randn([_B, _A, _K, _V]),
            "target_ids": target_ids,
            "eval_mask": eval_mask,
            "block_keep_mask": block_keep,
        }
        kwargs.update(overrides)
        with self.assertRaises(ValueError):
            compute_dspark_loss(DrafterOutput(**kwargs))

    def test_target_ids_are_required(self):
        with self.assertRaises(ValueError):
            compute_dspark_loss(
                DrafterOutput(
                    draft_logits=None,
                    target_ids=None,
                    eval_mask=None,
                    block_keep_mask=None,
                )
            )

    def test_eval_mask_is_required(self):
        self._rejected(eval_mask=None)

    def test_target_ids_must_be_three_dimensional(self):
        self._rejected(target_ids=_fields(seed=31)[0].reshape([_B, _A * _K]))

    def test_eval_mask_shape_must_match_target_ids(self):
        self._rejected(eval_mask=_fields(seed=32)[1][:, :, :1])

    def test_dimensions_must_be_nonempty(self):
        self._rejected(
            target_ids=paddle.zeros([_B, _A, 0], dtype="int64"),
            eval_mask=paddle.zeros([_B, _A, 0], dtype="bool"),
        )

    def test_target_ids_must_be_integral(self):
        self._rejected(target_ids=paddle.randn([_B, _A, _K]))

    def test_block_keep_mask_is_required(self):
        self._rejected(block_keep_mask=None)

    def test_block_keep_mask_must_be_per_block(self):
        self._rejected(block_keep_mask=paddle.ones([_B, _A, _K], dtype="bool"))

    def test_streamed_output_requires_its_hidden_states(self):
        target_ids, eval_mask, block_keep = _fields(seed=33)
        output = DrafterOutput(
            draft_logits=None,
            target_ids=target_ids,
            eval_mask=eval_mask,
            block_keep_mask=block_keep,
            logits_projection=ReadOnlyProjection(
                weight=paddle.randn([_V, _H]), transpose_y=True
            ),
        )
        with self.assertRaises(ValueError) as caught:
            compute_dspark_loss(output)
        self.assertIn("missing required fields", str(caught.exception))


class TestStreamedGuards(unittest.TestCase):
    """The streamed engine refuses ambiguous projections and refiners."""

    def test_projection_must_be_readonly_for_custom_transforms(self):
        projection = types.SimpleNamespace(
            weight=paddle.randn([_V, _H]),
            transpose_y=True,
            transform_logits=lambda logits: logits,
        )
        output = _streamed_output(seed=40, projection=projection)
        with self.assertRaises(ValueError):
            compute_dspark_loss(output)

    def test_hidden_dimension_must_match_the_projection(self):
        output = _streamed_output(
            seed=40,
            hidden_dim=_H + 4,
            projection=ReadOnlyProjection(
                weight=paddle.randn([_V, _H]), transpose_y=True
            ),
        )
        with self.assertRaises(ValueError):
            compute_dspark_loss(output)

    def test_candidate_selector_must_expose_its_codebooks(self):
        output = _streamed_output(
            seed=40, selector=types.SimpleNamespace(top_k=4)
        )
        with self.assertRaises(ValueError):
            compute_dspark_loss(output, LossProfile(selector_loss_alpha=0.5))

    def test_draft_hidden_leading_shape_must_match_target_ids(self):
        output = _streamed_output(
            seed=40,
            draft_hidden_states=paddle.randn([_B, _A, _K + 1, _H]),
        )
        with self.assertRaises(ValueError):
            compute_dspark_loss(output)

    def test_transpose_y_must_be_explicit(self):
        projection = types.SimpleNamespace(
            weight=paddle.randn([_V, _H]), transpose_y=None
        )
        with self.assertRaises(ValueError):
            compute_dspark_loss(
                _streamed_output(seed=40, projection=projection)
            )

    def test_target_ids_must_be_inside_the_projected_vocabulary(self):
        target_ids, eval_mask, block_keep = _fields(seed=42)
        target_ids[0, 0, 0] = _V + 5
        with self.assertRaises(ValueError):
            compute_dspark_loss(
                _streamed_output(
                    seed=42, fields=(target_ids, eval_mask, block_keep)
                )
            )

    def test_previous_ids_shape_must_match_target_ids(self):
        output = _streamed_output(
            seed=40,
            previous_token_ids=paddle.randint(
                0, _V, [_B, _A, _K + 1], dtype="int64"
            ),
        )
        with self.assertRaises(ValueError):
            compute_dspark_loss(output)

    def test_selector_top_k_must_fit_the_vocabulary(self):
        output = _streamed_output(
            seed=43, selector=_selector(seed=43, top_k=_V + 4, vocab_size=64)
        )
        with self.assertRaises(ValueError):
            compute_dspark_loss(output, LossProfile(selector_loss_alpha=0.5))

    def test_private_engine_guards_the_public_path_shadows(self):
        """``compute_dspark_loss`` pre-checks some of these; the engine does too."""
        cases = {
            "no_weight": types.SimpleNamespace(transpose_y=True),
            "transpose_not_bool": types.SimpleNamespace(
                weight=paddle.randn([_V, _H]), transpose_y=None
            ),
        }
        for name, projection in cases.items():
            with self.subTest(name=name):
                output = _streamed_output(seed=41, projection=projection)
                with self.assertRaises(ValueError):
                    dl._streamed_token_losses(
                        output,
                        output.target_ids,
                        chunk_tokens=8,
                        subbatch_sequence_length=0,
                        needs_teacher=True,
                    )
        with self.subTest(name="selector_without_a_candidate_selector"):
            output = _streamed_output(seed=41)
            with self.assertRaises(ValueError):
                dl._streamed_token_losses(
                    output,
                    output.target_ids,
                    chunk_tokens=8,
                    subbatch_sequence_length=0,
                    needs_teacher=True,
                    selector_enabled=True,
                )

    def test_streamed_without_a_teacher_zeroes_the_l1_term(self):
        output = _streamed_output(seed=44)
        terms = compute_dspark_loss(
            output, LossProfile(probability_l1_alpha=0.0), return_terms=True
        )
        self.assertEqual(float(terms.terms["l1_loss"]), 0.0)
        self.assertNotIn("accept_rate_per_pos_num", terms.terms)


class TestComputeDsparkLossGuards(unittest.TestCase):
    """Objective-level guards on the entry point itself."""

    def test_rejects_a_negative_effective_selector_alpha(self):
        with self.assertRaises(ValueError):
            compute_dspark_loss(
                _materialized_output(seed=50), selector_loss_alpha=-0.5
            )

    def test_selector_alpha_requires_a_candidate_selector(self):
        with self.assertRaises(ValueError):
            compute_dspark_loss(
                _materialized_output(seed=50),
                LossProfile(selector_loss_alpha=0.5),
            )

    def test_markov_and_selector_are_alternative_refiners(self):
        w1 = paddle.randn([_V, _R])
        w2 = paddle.randn([_R, _V])
        markov = types.SimpleNamespace(
            markov_w1=types.SimpleNamespace(weight=w1),
            markov_w2=types.SimpleNamespace(weight=w2),
        )
        output = _materialized_output(seed=50, selector=_selector(seed=50))
        output.markov_head = markov
        with self.assertRaises(ValueError):
            compute_dspark_loss(output, LossProfile(selector_loss_alpha=0.5))

    def test_draft_logits_leading_shape_must_match_target_ids(self):
        with self.assertRaises(ValueError):
            compute_dspark_loss(
                _materialized_output(
                    seed=50,
                    draft_logits=paddle.randn([_B, _A, _K + 1, _V]),
                )
            )

    def test_teacher_logits_are_required_for_the_l1_term(self):
        with self.assertRaises(ValueError):
            compute_dspark_loss(
                _materialized_output(seed=50, with_teacher=False)
            )

    def test_teacher_logits_shape_must_match_draft_logits(self):
        with self.assertRaises(ValueError):
            compute_dspark_loss(
                _materialized_output(
                    seed=50,
                    teacher_logits=paddle.randn([_B, _A, _K + 1, _V]),
                )
            )

    def test_confidence_pred_shape_must_match_target_ids(self):
        with self.assertRaises(ValueError):
            compute_dspark_loss(
                _materialized_output(
                    seed=50,
                    confidence_pred=paddle.randn([_B, _A, _K + 1]),
                )
            )


class TestFusedAvailability(unittest.TestCase):
    """The fused kernel is opt-in and must degrade to eager, never crash."""

    def test_falls_back_without_triton(self):
        saved = sys.modules.get("triton", None)
        sys.modules["triton"] = None  # makes ``import triton`` raise
        try:
            self.assertFalse(dl._fused_distill_available(paddle.zeros([2])))
        finally:
            if saved is None:
                sys.modules.pop("triton", None)
            else:
                sys.modules["triton"] = saved

    def test_falls_back_without_a_cuda_build(self):
        original = paddle.device.is_compiled_with_cuda
        paddle.device.is_compiled_with_cuda = lambda: False
        try:
            self.assertFalse(dl._fused_distill_available(paddle.zeros([2])))
        finally:
            paddle.device.is_compiled_with_cuda = original

    def test_falls_back_off_a_gpu_place(self):
        if not paddle.device.is_compiled_with_cuda():
            self.skipTest("CUDA build required to reach the place check")
        with paddle.device.device_guard("cpu"):
            on_cpu = paddle.zeros([2])
        self.assertFalse(dl._fused_distill_available(on_cpu))


class TestNormalizationGroup(unittest.TestCase):
    """A process group folds the denominator across data-parallel ranks."""

    def _group(self):
        # Single-card runs do not always bring up the parallel env, but a
        # world-sized group is still available and reduces to the identity.
        try:
            ranks = list(range(paddle.distributed.get_world_size()))
            return paddle.distributed.new_group(ranks=ranks)
        except Exception as exc:
            self.skipTest(f"no process group available: {exc}")

    def test_denominator_is_all_reduced(self):
        group = self._group()
        terms = compute_dspark_loss(
            _materialized_output(seed=60),
            LossProfile(),
            return_terms=True,
            normalization_group=group,
        )
        self.assertTrue(np.isfinite(float(terms.terms["loss"])))
        self.assertAlmostEqual(
            float(terms.terms["normalized_weight_denominator"]),
            float(terms.terms["weight_denominator"]),
            delta=1e-3,
        )

    def test_selector_denominator_is_all_reduced_separately(self):
        group = self._group()
        selector = _selector(seed=61)
        output = _materialized_output(
            seed=61,
            selector=selector,
            draft_hidden_states=paddle.randn([_B, _A, _K, _H]),
            previous_token_ids=paddle.randint(
                0, _V, [_B, _A, _K], dtype="int64"
            ),
        )
        terms = compute_dspark_loss(
            output,
            LossProfile(selector_loss_alpha=0.5),
            return_terms=True,
            normalization_group=group,
        )
        self.assertAlmostEqual(
            float(terms.terms["normalized_selector_weight_denominator"]),
            float(terms.terms["selector_weight_denominator"]),
            delta=1e-3,
        )


if __name__ == "__main__":
    unittest.main()
