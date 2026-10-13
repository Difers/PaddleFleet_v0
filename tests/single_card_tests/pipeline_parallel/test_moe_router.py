# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

"""CPU-only behavior tests for the device-independent pure logic in
``paddlefleet.transformer.moe.moe_router``.

Covered (all runnable on CPU, no collectives, no ``world_size`` faking):
  * ``StandardMoERouter._capacity`` - per-expert capacity integer arithmetic
    and its min/max clamps and the ``capacity > 0`` assertion.
  * ``StandardMoERouter._cal_aux_loss`` - the load-balancing auxiliary loss
    ``sum(mean(gates) * mean(mask)) * num_experts``.
  * ``StandardMoERouter._priority`` - capacity-limited cumulative priority that
    actually DROPS the later token that overflows an expert.
  * ``StandardMoERouter._topk_group_limited_greedy`` - group pre-selection then
    top-k, verifying the chosen expert ids and their gate weights.
  * ``HFBitexactSoftmax`` - the fp32 softmax PyLayer forward AND its custom
    backward against an independent numpy Jacobian-vector reference.
  * ``FusedGateDetachMatmul`` - the (non-hf, non-accuracy) fused gate matmul
    forward ``x @ w.T`` AND backward (both ``x`` and ``w`` gradients).
  * ``StandardMoERouter._probs_drop_policy`` - per-expert top-``capacity`` token
    selection: for each expert only its highest-scoring tokens survive, and any
    token that was not originally selected (score 0) is zeroed (checked against
    an independent numpy reference).
  * ``StandardMoERouter._cal_seq_aux_loss`` - single-rank (tp=cp=1) line
    aggregation on the plain branch: equal-weight line mean on the legacy path
    versus the token-weighted line sum under ``calculate_per_token_loss`` with
    more than one line.

The pure math methods are invoked as real *unbound* methods with a minimal
stand-in ``self`` that only carries the data attributes they read
(``num_experts``). The production method body runs unchanged; no formula is
re-implemented in the test. Constructing a full ``StandardMoERouter`` needs a
complete ``TransformerConfig`` plus a live parallel context, which is out of
scope for CPU pure-logic verification.

Every expected value is derived by hand / independent numpy, never by calling
the code under test. Distributed routing paths (sequence/context/expert
parallel gathers inside ``_cal_seq_aux_loss``/``_cal_z_loss``) are intentionally
NOT covered: faking ``world_size`` + mocking collectives would only prove local
orchestration, not cross-rank semantics.
"""

import unittest
from types import SimpleNamespace

import numpy as np

try:
    import paddle

    from paddlefleet.transformer.moe.moe_router import (
        FusedGateDetachMatmul,
        HFBitexactSoftmax,
        StandardMoERouter,
    )

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # honest dependency probe
    paddle = None
    FusedGateDetachMatmul = None
    HFBitexactSoftmax = None
    StandardMoERouter = None
    _IMPORT_ERROR = exc

_HAS_DEPS = _IMPORT_ERROR is None
_SKIP_REASON = (
    f"paddle / paddlefleet not importable on this CPU host: {_IMPORT_ERROR!r}"
)


class _RouterStub:
    """Minimal ``self`` carrying only the data attributes the pure methods read.

    It holds no behavior: every method under test is the real, unmodified
    production method invoked as ``StandardMoERouter.<method>(stub, ...)``.
    """

    def __init__(self, num_experts):
        self.num_experts = num_experts
        # _cal_aux_loss now reads getattr(self.config, "calculate_per_token_loss",
        # False); give the stub a config so the guard sees the legacy (False) path.
        self.config = SimpleNamespace(calculate_per_token_loss=False)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestCapacity(unittest.TestCase):
    """``_capacity`` = clamp(int((num_tokens // num_experts) * factor))."""

    def setUp(self):
        paddle.set_device("cpu")

    def _capacity(self, num_tokens, num_experts, factor, max_cap, min_cap):
        gates = paddle.zeros([num_tokens, num_experts], dtype="float32")
        stub = _RouterStub(num_experts)
        return StandardMoERouter._capacity(
            stub, gates, factor, max_cap, min_cap
        )

    def test_plain_value(self):
        # 8 // 2 = 4; int(4 * 1.5) = 6; inside [1, 100] -> 6.
        self.assertEqual(self._capacity(8, 2, 1.5, 100, 1), 6)

    def test_truncates_toward_zero(self):
        # 10 // 3 = 3; int(3 * 1.4) = int(4.2) = 4 (floor, not round).
        self.assertEqual(self._capacity(10, 3, 1.4, 100, 1), 4)

    def test_min_clamp_raises_floor(self):
        # 4 // 4 = 1; int(1 * 0.5) = 0; below min_capacity=2 -> clamped to 2.
        self.assertEqual(self._capacity(4, 4, 0.5, 100, 2), 2)

    def test_max_clamp_lowers_ceiling(self):
        # 100 // 2 = 50; int(50 * 2.0) = 100; above max_capacity=8 -> 8.
        self.assertEqual(self._capacity(100, 2, 2.0, 8, 1), 8)

    def test_zero_capacity_trips_assertion(self):
        # 4 // 4 = 1; int(1 * 0.0) = 0; min=0 keeps it 0 -> assert capacity > 0.
        with self.assertRaises(AssertionError):
            self._capacity(4, 4, 0.0, 100, 0)

    def test_non_2d_gates_rejected(self):
        stub = _RouterStub(2)
        with self.assertRaises(AssertionError):
            StandardMoERouter._capacity(
                stub, paddle.zeros([4], dtype="float32"), 1.0, 100, 1
            )


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestCalAuxLoss(unittest.TestCase):
    """``_cal_aux_loss`` = sum(mean_tok(gates) * mean_tok(mask)) * num_experts."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_matches_independent_reference(self):
        num_experts = 3
        gates = np.array(
            [
                [0.7, 0.2, 0.1],
                [0.1, 0.6, 0.3],
                [0.4, 0.4, 0.2],
                [0.0, 0.5, 0.5],
            ],
            dtype=np.float32,
        )
        mask = np.array(
            [[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 1]], dtype=np.float32
        )
        # Independent hand reference (no call into the code under test).
        me = gates.mean(axis=0)
        ce = mask.mean(axis=0)
        expected = float((me * ce).sum() * num_experts)

        stub = _RouterStub(num_experts)
        out = StandardMoERouter._cal_aux_loss(
            stub,
            paddle.to_tensor(gates),
            paddle.to_tensor(mask),
        )
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6, atol=1e-6)

    def test_scales_with_num_experts_factor(self):
        # Same per-column means but the trailing ``* num_experts`` factor must
        # be consumed: doubling num_experts (with matching width) doubles loss.
        gates = paddle.to_tensor([[0.5, 0.5], [0.25, 0.75]], dtype="float32")
        mask = paddle.to_tensor([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
        me = np.array([0.375, 0.625], dtype=np.float32)
        ce = np.array([0.5, 0.5], dtype=np.float32)
        expected = float((me * ce).sum() * 2)
        out = StandardMoERouter._cal_aux_loss(_RouterStub(2), gates, mask)
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6, atol=1e-6)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestPriority(unittest.TestCase):
    """``_priority`` keeps a token->expert slot only while the running per-expert
    count is within ``capacity``; overflowing later tokens are dropped."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_capacity_drops_overflowing_token(self):
        # 3 tokens, top_k=1, num_experts=3, capacity=1.
        #   token0 -> e0, token1 -> e0 (e0 already full -> DROP), token2 -> e1.
        # Row-major cumsum over the flattened [token*k] slots enforces the
        # first-come rule, so the expected assignment is:
        #   token0: e0 kept, token1: nothing, token2: e1 kept.
        topk_idx = paddle.to_tensor([[0], [0], [1]], dtype="int64")
        out = StandardMoERouter._priority(_RouterStub(3), topk_idx, capacity=1)
        expected = np.array(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(out.numpy(), expected)

    def test_capacity_two_keeps_both(self):
        # capacity=2: the two tokens routed to e0 both fit; the third to e1.
        topk_idx = paddle.to_tensor([[0], [0], [1]], dtype="int64")
        out = StandardMoERouter._priority(_RouterStub(3), topk_idx, capacity=2)
        expected = np.array(
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(out.numpy(), expected)

    def test_topk_two_collapses_to_per_token_expert_set(self):
        # top_k=2 with plenty of capacity: each token's row marks exactly the
        # experts it selected (order within k does not matter after the sum).
        topk_idx = paddle.to_tensor([[0, 2], [1, 2]], dtype="int64")
        out = StandardMoERouter._priority(_RouterStub(3), topk_idx, capacity=8)
        expected = np.array(
            [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=np.float32
        )
        np.testing.assert_array_equal(out.numpy(), expected)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestTopkGroupLimitedGreedy(unittest.TestCase):
    """``_topk_group_limited_greedy``: pick ``topk_group`` groups by their max
    score, zero out the rest, then top-k experts over the masked scores."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_group_preselection_then_topk(self):
        # 4 experts, n_group=2 -> group0={e0,e1}, group1={e2,e3}; topk_group=1.
        #   token0: group maxes = [0.2, 0.9] -> keep group1 -> top1 expert = e2.
        #   token1: group maxes = [0.8, 0.2] -> keep group0 -> top1 expert = e0.
        scores = paddle.to_tensor(
            [[0.1, 0.2, 0.9, 0.3], [0.8, 0.1, 0.2, 0.05]], dtype="float32"
        )
        weight, idx = StandardMoERouter._topk_group_limited_greedy(
            _RouterStub(4), scores, k=1, n_group=2, topk_group=1
        )
        self.assertEqual(idx.numpy().tolist(), [[2], [0]])
        np.testing.assert_allclose(
            weight.numpy(),
            np.array([[0.9], [0.8]], dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_masked_group_expert_cannot_be_selected(self):
        # A large score living in the NON-selected group must be suppressed:
        #   token0: group maxes = [0.95, 0.9] -> keep group0 -> e0 (0.95),
        #   NOT e2 (0.9) even though it is the 2nd highest overall.
        scores = paddle.to_tensor([[0.95, 0.05, 0.9, 0.1]], dtype="float32")
        weight, idx = StandardMoERouter._topk_group_limited_greedy(
            _RouterStub(4), scores, k=1, n_group=2, topk_group=1
        )
        self.assertEqual(idx.numpy().tolist(), [[0]])
        np.testing.assert_allclose(
            weight.numpy(), np.array([[0.95]], dtype=np.float32), atol=1e-6
        )


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestHFBitexactSoftmax(unittest.TestCase):
    """``HFBitexactSoftmax`` fp32 softmax forward and its custom backward."""

    def setUp(self):
        paddle.set_device("cpu")

    @staticmethod
    def _np_softmax(x):
        e = np.exp(x - x.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    def test_forward_matches_reference_softmax(self):
        x_np = np.array(
            [[1.0, 2.0, -1.0, 0.5], [0.0, 0.0, 3.0, -2.0]], dtype=np.float32
        )
        out = HFBitexactSoftmax.apply(paddle.to_tensor(x_np))
        np.testing.assert_allclose(
            out.numpy(), self._np_softmax(x_np), rtol=1e-6, atol=1e-6
        )
        # Softmax rows sum to 1 - guards against an un-normalized regression.
        np.testing.assert_allclose(
            out.numpy().sum(axis=-1), np.ones(2, np.float32), atol=1e-6
        )

    def test_backward_matches_independent_jvp(self):
        # Distinguishable, non-uniform upstream grad so a wrong epilogue (e.g.
        # forgetting the ``- sum(g*p) * p`` term) cannot cancel out.
        x_np = np.array(
            [[1.0, 2.0, -1.0, 0.5], [0.0, 0.0, 3.0, -2.0]], dtype=np.float32
        )
        g_np = np.array(
            [[0.3, -0.7, 0.2, 1.1], [-0.5, 0.4, 0.9, -0.2]], dtype=np.float32
        )
        x = paddle.to_tensor(x_np)
        x.stop_gradient = False
        out = HFBitexactSoftmax.apply(x)
        out.backward(paddle.to_tensor(g_np))

        # Independent softmax Jacobian-vector product: dx = p*(g - sum(g*p)).
        p = self._np_softmax(x_np)
        inner = (g_np * p).sum(axis=-1, keepdims=True)
        dx_ref = g_np * p - inner * p

        self.assertIsNotNone(x.grad)
        np.testing.assert_allclose(x.grad.numpy(), dx_ref, rtol=1e-5, atol=1e-6)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestFusedGateDetachMatmul(unittest.TestCase):
    """``FusedGateDetachMatmul`` default (non-hf, non-accuracy) path computes
    ``x @ w.T`` and returns matching gradients for both ``x`` and ``w``."""

    def setUp(self):
        paddle.set_device("cpu")

    def _inputs(self):
        # n=2, H=3, E=4; distinguishable, non-uniform so a transpose/order
        # mistake shows up in the numbers.
        x_np = np.array([[1.0, 2.0, -1.0], [0.5, -0.5, 2.0]], dtype=np.float32)
        w_np = np.array(
            [
                [1.0, 0.0, -1.0],
                [2.0, 1.0, 0.5],
                [-1.0, 0.5, 1.5],
                [0.25, -2.0, 1.0],
            ],
            dtype=np.float32,
        )
        return x_np, w_np

    def test_forward_is_x_matmul_w_transpose(self):
        x_np, w_np = self._inputs()
        out = FusedGateDetachMatmul.apply(
            paddle.to_tensor(x_np),
            paddle.to_tensor(w_np),
            False,
            False,
        )
        np.testing.assert_allclose(
            out.numpy(), x_np @ w_np.T, rtol=1e-5, atol=1e-5
        )

    def test_backward_x_and_w_grads(self):
        x_np, w_np = self._inputs()
        g_np = np.array(
            [[0.5, -1.0, 2.0, 0.25], [1.5, 0.0, -0.5, 1.0]], dtype=np.float32
        )
        x = paddle.to_tensor(x_np)
        w = paddle.to_tensor(w_np)
        x.stop_gradient = False
        w.stop_gradient = False
        out = FusedGateDetachMatmul.apply(x, w, False, False)
        out.backward(paddle.to_tensor(g_np))

        # y = x @ w.T  ->  dx = g @ w ,  dw = g.T @ x  (independent numpy).
        dx_ref = g_np @ w_np
        dw_ref = g_np.T @ x_np
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(w.grad)
        np.testing.assert_allclose(x.grad.numpy(), dx_ref, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(w.grad.numpy(), dw_ref, rtol=1e-5, atol=1e-5)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestProbsDropPolicy(unittest.TestCase):
    """``_probs_drop_policy`` enforces per-expert capacity: for each expert
    (column) it keeps only the ``capacity`` highest-scoring tokens (top-k over
    the token axis) and then zeroes out any token that was not originally
    selected (score 0). The intended per-expert top-``capacity`` behavior is
    verified against an independent numpy reference.
    """

    def setUp(self):
        paddle.set_device("cpu")

    def test_keeps_top_capacity_tokens_per_expert(self):
        # Already-gated scores (zeros for non-selected). capacity=1 -> each
        # expert keeps only its single highest-scoring token.
        #   e0 column [0.9, 0.1, 0.0] -> token0 ; e1 column [0.0, 0.7, 0.5] -> token1.
        scores = paddle.to_tensor(
            [[0.9, 0.0], [0.1, 0.7], [0.0, 0.5]], dtype="float32"
        )
        out = StandardMoERouter._probs_drop_policy(
            _RouterStub(2), scores, capacity=1
        )
        expected = np.array(
            [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=np.float32
        )
        np.testing.assert_array_equal(out.numpy(), expected)


class _SeqAuxRouterStub:
    """``self`` for ``_cal_seq_aux_loss`` on the single-rank (tp=cp=1) path.

    Carries only the data attributes that method reads. ``routing_type`` is
    checked by the caller, not inside the method, so it is not needed here.
    """

    def __init__(self, num_experts, per_token):
        self.num_experts = num_experts
        self.use_accuracy_compatible = False
        self.tensor_model_parallel_size = 1
        self.context_parallel_size = 1
        self.sequence_parallel = False
        self.config = SimpleNamespace(
            sequence_parallel=False,
            experimental_dataflow=False,
            pad_token_id=0,
            gpt_model_use_experimental_version=False,
            calculate_per_token_loss=per_token,
        )


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestCalSeqAuxLossPerLineWeighting(unittest.TestCase):
    """``_cal_seq_aux_loss`` line aggregation on the plain branch (neither
    ``use_accuracy_compatible`` nor ``gpt_model_use_experimental_version``).

    Each line ``b`` gets its own O(1) value
    ``m_b = (E/K) * sum_e f_be * P_be / t_b**2``, where ``t_b`` is that line's
    non-padded token count. The two aggregations differ only when the lines hold
    different numbers of valid tokens:
      * legacy / non-per-token: ``mean_b(m_b)``; under
        ``calculate_per_token_loss`` that mean is then multiplied by
        ``sum_b t_b``, giving ``mean_b(m_b) * sum_b t_b``.
      * ``calculate_per_token_loss`` with B > 1: the exact token-weighted sum
        ``sum_b m_b * t_b``, which is what the trainer's single 1/T_global
        expects.
    Expected values come from an independent numpy computation, never from the
    code under test. Only the single-rank path is exercised; the TP/CP gathers
    stay out of scope (see the module docstring).
    """

    E = 4
    K = 2

    def setUp(self):
        paddle.set_device("cpu")

    @staticmethod
    def _inputs():
        """Deterministic ragged batch: line 0 has 3 valid tokens, line 1 has 1."""
        ids = np.array([[1, 1, 1, 0], [1, 0, 0, 0]], dtype=np.int64)
        probs = np.zeros((2, 4, 4), dtype=np.float32)
        probs[0, 0] = [0.4, 0.3, 0.2, 0.1]
        probs[0, 1] = [0.1, 0.5, 0.3, 0.1]
        probs[0, 2] = [0.25, 0.25, 0.25, 0.25]
        probs[1, 0] = [0.7, 0.1, 0.1, 0.1]
        routing = np.zeros((2, 4, 4), dtype=np.float32)
        routing[0, 0, [0, 1]] = 1.0
        routing[0, 1, [1, 2]] = 1.0
        routing[0, 2, [0, 3]] = 1.0
        routing[1, 0, [0, 1]] = 1.0
        return ids, probs, routing

    @classmethod
    def _per_line_reference(cls, ids, probs, routing):
        """``m_b`` per line plus each line's token count, by independent numpy."""
        t = (ids != 0).sum(-1).astype(np.float64)
        f = routing.sum(1).astype(np.float64)
        p = probs.sum(1).astype(np.float64)
        cost = f / (t[:, None] * (cls.K / cls.E))
        return (cost * p / t[:, None]).sum(1), t

    def _call(self, ids, probs, routing, per_token):
        bsz, seq_len = ids.shape
        return float(
            StandardMoERouter._cal_seq_aux_loss(
                _SeqAuxRouterStub(self.E, per_token),
                paddle.to_tensor(probs.reshape(bsz * seq_len, self.E)),
                self.K,
                paddle.to_tensor(routing.reshape(bsz * seq_len, self.E)),
                seq_len,
                bsz,
                input_ids=paddle.to_tensor(ids),
            )
        )

    def test_per_token_ragged_lines_use_token_weighted_sum(self):
        ids, probs, routing = self._inputs()
        m, t = self._per_line_reference(ids, probs, routing)
        weighted = float((m * t).sum())
        mean_form = float(m.mean() * t.sum())
        # The two aggregations must actually disagree here, otherwise this test
        # would pass for the wrong reason.
        self.assertGreater(abs(mean_form - weighted) / weighted, 1e-3)
        got = self._call(ids, probs, routing, per_token=True)
        self.assertAlmostEqual(got / weighted, 1.0, places=5)

    def test_per_token_equal_lines_agree_with_mean_form(self):
        # Both lines hold 2 valid tokens -> the weighting is a no-op.
        ids, probs, routing = self._inputs()
        ids = np.array([[1, 1, 0, 0], [1, 1, 0, 0]], dtype=np.int64)
        probs[0, 2] = 0.0
        routing[0, 2] = 0.0
        probs[1, 1] = [0.2, 0.2, 0.3, 0.3]
        routing[1, 1, [2, 3]] = 1.0
        m, t = self._per_line_reference(ids, probs, routing)
        got = self._call(ids, probs, routing, per_token=True)
        self.assertAlmostEqual(got / float((m * t).sum()), 1.0, places=5)
        self.assertAlmostEqual(got / float(m.mean() * t.sum()), 1.0, places=5)

    def test_per_token_single_line_keeps_mean_times_total(self):
        # B == 1: the per-line weighting is skipped, so the value stays on the
        # original mean-then-multiply path.
        ids, probs, routing = self._inputs()
        ids, probs, routing = ids[:1], probs[:1], routing[:1]
        m, t = self._per_line_reference(ids, probs, routing)
        got = self._call(ids, probs, routing, per_token=True)
        self.assertAlmostEqual(got / float(m.mean() * t.sum()), 1.0, places=5)

    def test_legacy_path_still_averages_lines_equally(self):
        ids, probs, routing = self._inputs()
        m, _ = self._per_line_reference(ids, probs, routing)
        got = self._call(ids, probs, routing, per_token=False)
        self.assertAlmostEqual(got / float(m.mean()), 1.0, places=5)

    def test_per_token_without_input_ids_scales_by_seq_len_times_batch(self):
        # No input_ids: every position is valid, so the mean is multiplied by
        # S * B instead of the per-line token counts.
        ids, probs, routing = self._inputs()
        bsz, seq_len = ids.shape

        def call(per_token):
            return float(
                StandardMoERouter._cal_seq_aux_loss(
                    _SeqAuxRouterStub(self.E, per_token),
                    paddle.to_tensor(probs.reshape(bsz * seq_len, self.E)),
                    self.K,
                    paddle.to_tensor(routing.reshape(bsz * seq_len, self.E)),
                    seq_len,
                    bsz,
                )
            )

        legacy = call(False)
        self.assertAlmostEqual(
            call(True) / (legacy * seq_len * bsz), 1.0, places=5
        )


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestPerTokenZLossAndAuxGuard(unittest.TestCase):
    """``calculate_per_token_loss`` on ``_cal_z_loss`` / ``_cal_aux_loss``."""

    def setUp(self):
        paddle.set_device("cpu")

    @staticmethod
    def _stub(per_token):
        stub = _RouterStub(num_experts=3)
        stub.config = SimpleNamespace(
            calculate_per_token_loss=per_token,
            sequence_parallel=False,
            experimental_dataflow=False,
            pad_token_id=0,
            gpt_model_use_experimental_version=False,
        )
        return stub

    def test_z_loss_per_token_is_raw_masked_sum(self):
        logits_np = np.array(
            [[0.1, 0.5, -0.2], [1.0, 0.0, 0.3], [0.2, 0.2, 0.2]],
            dtype=np.float32,
        )
        ids = np.array([[5, 7, 0]], dtype=np.int64)  # last token is padding
        mask = (ids.reshape(-1) != 0).astype(np.float64)
        lse = np.log(np.exp(logits_np.astype(np.float64)).sum(-1))
        raw = float((lse**2 * mask).sum())

        def call(per_token):
            return float(
                StandardMoERouter._cal_z_loss(
                    self._stub(per_token),
                    paddle.to_tensor(logits_np),
                    input_ids=paddle.to_tensor(ids),
                )
            )

        self.assertAlmostEqual(call(True) / raw, 1.0, places=5)
        self.assertAlmostEqual(call(False) / (raw / mask.sum()), 1.0, places=5)

    def test_z_loss_per_token_requires_input_ids(self):
        logits = paddle.randn([3, 3], dtype="float32")
        with self.assertRaisesRegex(AssertionError, "input_ids"):
            StandardMoERouter._cal_z_loss(self._stub(True), logits)
        # Legacy path still accepts a missing input_ids.
        StandardMoERouter._cal_z_loss(self._stub(False), logits)

    def test_aux_loss_rejects_per_token(self):
        gates = paddle.full([2, 3], 1.0 / 3.0, dtype="float32")
        mask = paddle.ones([2, 3], dtype="int64")
        with self.assertRaisesRegex(AssertionError, "seq_aux_loss"):
            StandardMoERouter._cal_aux_loss(self._stub(True), gates, mask)


if __name__ == "__main__":
    unittest.main()
