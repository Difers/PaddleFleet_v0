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

"""moe_multimodal_router_bias: multimodal tokens route with their own bias."""

import types
import unittest
from unittest.mock import patch

import numpy as np
import paddle
import paddle.nn.functional as F

from paddlefleet.trainer.trainer_callback import MoEMultimodalBiasAdjustCallback
from paddlefleet.transformer.moe.moe_router import (
    StandardMoERouter,
    TopKRouter,
)

from .test_router import MockTransformerConfig

CTX = 41
PAD = 0
# [2, 6]: CONTEXT tokens, text tokens and padding.
INPUT_IDS = [[CTX, CTX, 5, 6, CTX, 0], [7, CTX, CTX, 8, 0, 0]]


def _config(**overrides):
    cfg = MockTransformerConfig()
    cfg.pad_token_id = PAD
    cfg.hyperbody_context_token_id = CTX
    cfg.moe_multimodal_router_bias = True
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _router(layer=1, is_mtp_layer=False, **overrides):
    router = TopKRouter(_config(**overrides))
    router.set_layer_number(layer, is_mtp_layer=is_mtp_layer)
    return router


def _flat(ids):
    return np.asarray(ids).reshape(-1)


def _topk_sets(x, k):
    return [set(row) for row in np.argsort(-x, axis=-1)[:, :k].tolist()]


class _CPPatch(unittest.TestCase):
    def setUp(self):
        patcher = patch(
            "paddlefleet.transformer.moe.moe_router.get_context_parallel_world_size",
            return_value=1,
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        paddle.seed(7)
        np.random.seed(7)


class TestSetup(_CPPatch):
    def test_disabled_by_default(self):
        router = _router(moe_multimodal_router_bias=False)
        self.assertFalse(router.multimodal_bias_enabled)
        self.assertFalse(hasattr(router, "multimodal_e_score_correction_bias"))

    def test_registered_on_learned_mtp_and_hash_layers(self):
        learned = _router()
        mtp = _router(layer=5, is_mtp_layer=True)
        hashed = _router(layer=0, moe_n_hash_layers=1, actual_vocab_size=64)
        for router in (learned, mtp, hashed):
            self.assertEqual(
                router.multimodal_e_score_correction_bias.shape, [8]
            )
            self.assertIn(
                "multimodal_e_score_correction_bias", router.state_dict()
            )
            self.assertEqual(router.multimodal_token_id, CTX)
        # Hash layers drop the text bias but keep the multimodal one.
        self.assertTrue(hashed.is_hash_layer)
        self.assertFalse(hasattr(hashed, "e_score_correction_bias"))

    def test_set_layer_number_twice_keeps_the_buffer(self):
        router = _router()
        router.multimodal_e_score_correction_bias.set_value(
            paddle.full([8], 0.5)
        )
        router.set_layer_number(1)
        self.assertEqual(
            float(router.multimodal_e_score_correction_bias.sum()), 4.0
        )

    def test_token_id_comes_from_hyperbody_context_token_id(self):
        # image_token_id is not a fallback: only the <context> placeholder counts.
        self.assertEqual(
            _router(
                hyperbody_context_token_id=9, image_token_id=7
            ).multimodal_token_id,
            9,
        )
        with self.assertRaises(ValueError):
            _router(hyperbody_context_token_id=None, image_token_id=9)

    def test_standard_router_set_layer_number_registers_bias(self):
        router = StandardMoERouter(_config())
        router.set_layer_number(1)
        self.assertTrue(router.multimodal_bias_enabled)
        self.assertIn("multimodal_e_score_correction_bias", router.state_dict())

    def test_choice_bias_only_for_noaux_tc(self):
        router = _router()
        gates = paddle.rand([4, 8])
        with self.assertRaisesRegex(ValueError, "choice_bias"):
            router._call_topk_method(
                "greedy", gates, k=2, choice_bias=paddle.zeros([4, 8])
            )

    def test_learned_layer_requires_noaux_tc(self):
        with self.assertRaises(ValueError):
            _router(topk_method="greedy")


class TestLearnedLayer(_CPPatch):
    def _pair(self, **overrides):
        ref = _router(moe_multimodal_router_bias=False, **overrides)
        mm = _router(**overrides)
        mm.weight.set_value(ref.weight)
        text_bias = paddle.to_tensor(np.random.randn(8).astype("float32"))
        mm_bias = paddle.to_tensor(np.random.randn(8).astype("float32"))
        ref.e_score_correction_bias.set_value(text_bias)
        mm.e_score_correction_bias.set_value(text_bias)
        mm.multimodal_e_score_correction_bias.set_value(mm_bias)
        return ref, mm, mm_bias.numpy()

    def _check(self, **overrides):
        ref, mm, mm_bias = self._pair(**overrides)
        hidden = paddle.randn([2, 6, 64])
        input_ids = paddle.to_tensor(INPUT_IDS)
        _, _, idx_ref, _, _, _, _, _ = ref(hidden, input_ids=input_ids)
        _, _, idx_mm, _, _, _, _, _ = mm(hidden, input_ids=input_ids)

        ids = _flat(INPUT_IDS)
        is_mm, is_text = ids == CTX, (ids != CTX) & (ids != PAD)
        # Text rows route exactly as before; padding stays -1.
        np.testing.assert_array_equal(
            idx_mm.numpy()[~is_mm], idx_ref.numpy()[~is_mm]
        )
        # Multimodal rows pick topk(score + multimodal bias).
        logits = hidden.reshape([-1, 64]).matmul(mm.weight, transpose_y=True)
        scores = F.softmax(logits.cast("float32"), axis=-1).numpy()
        want = _topk_sets(scores + mm_bias[None], 2)
        got = [set(r) for r in idx_mm.numpy().tolist()]
        for row in np.nonzero(is_mm)[0]:
            self.assertEqual(got[row], want[row])

        # Each bias counts only its own tokens.
        k = 2
        self.assertEqual(
            int(ref.expert_usage.sum()), (is_mm | is_text).sum() * k
        )
        self.assertEqual(int(mm.expert_usage.sum()), is_text.sum() * k)
        self.assertEqual(int(mm.multimodal_expert_usage.sum()), is_mm.sum() * k)
        mm_counts = np.zeros(8, dtype=np.int64)
        for row in np.nonzero(is_mm)[0]:
            mm_counts[list(got[row])] += 1
        np.testing.assert_array_equal(
            mm.multimodal_expert_usage.numpy(), mm_counts
        )

    def test_eager_topk(self):
        self._check()

    def test_fused_topk(self):
        try:
            from paddlefleet.transformer.moe.moe_router import (
                _get_moe_topk_fusion,
            )

            _get_moe_topk_fusion()
        except ImportError as e:  # triton not installed
            self.skipTest(f"MoETopkFusion unavailable: {e}")
        self._check(moe_topk_fusion=True, scoring_func="sigmoid")

    def test_sequence_parallel_row_order(self):
        """SP: hidden is [S, B, H] and router rows are sequence-major, so the
        multimodal mask must follow input_ids.T, not input_ids."""
        overrides = {
            "sequence_parallel": True,
            "expert_model_parallel_size": 1,
            "experimental_dataflow": False,
        }
        ref, mm, mm_bias = self._pair(**overrides)
        hidden_bsh = paddle.randn([2, 6, 64])
        hidden = hidden_bsh.transpose([1, 0, 2])  # [S, B, H]
        input_ids = paddle.to_tensor(INPUT_IDS)  # still [B, S]
        _, _, idx_ref, _, _, _, _, _ = ref(hidden, input_ids=input_ids)
        _, _, idx_mm, _, _, _, _, _ = mm(hidden, input_ids=input_ids)

        ids = np.asarray(INPUT_IDS).T.reshape(-1)  # sequence-major rows
        is_mm = ids == CTX
        np.testing.assert_array_equal(
            idx_mm.numpy()[~is_mm], idx_ref.numpy()[~is_mm]
        )
        logits = hidden.reshape([-1, 64]).matmul(mm.weight, transpose_y=True)
        scores = F.softmax(logits.cast("float32"), axis=-1).numpy()
        want = _topk_sets(scores + mm_bias[None], 2)
        got = [set(r) for r in idx_mm.numpy().tolist()]
        mm_counts = np.zeros(8, dtype=np.int64)
        for row in np.nonzero(is_mm)[0]:
            self.assertEqual(got[row], want[row])
            mm_counts[list(got[row])] += 1
        np.testing.assert_array_equal(
            mm.multimodal_expert_usage.numpy(), mm_counts
        )

    def test_no_counting_without_grad(self):
        _, mm, _ = self._pair()
        with paddle.no_grad():
            mm(paddle.randn([2, 6, 64]), input_ids=paddle.to_tensor(INPUT_IDS))
        self.assertEqual(int(mm.multimodal_expert_usage.sum()), 0)

    def test_requires_input_ids(self):
        _, mm, _ = self._pair()
        with self.assertRaises(ValueError):
            mm(paddle.randn([2, 6, 64]))


class TestHashLayer(_CPPatch):
    def test_multimodal_rows_leave_tid2eid(self):
        overrides = {
            "moe_n_hash_layers": 1,
            "actual_vocab_size": 64,
            "scoring_func": "sigmoid",
        }
        ref = _router(layer=0, moe_multimodal_router_bias=False, **overrides)
        mm = _router(layer=0, **overrides)
        mm.weight.set_value(ref.weight)
        mm_bias = np.random.randn(8).astype("float32")
        mm.multimodal_e_score_correction_bias.set_value(
            paddle.to_tensor(mm_bias)
        )

        hidden = paddle.randn([2, 6, 64])
        input_ids = paddle.to_tensor(INPUT_IDS)
        _, gate_ref, idx_ref, _, _, _, _, _ = ref(hidden, input_ids=input_ids)
        _, gate_mm, idx_mm, _, _, _, _, _ = mm(hidden, input_ids=input_ids)

        ids = _flat(INPUT_IDS)
        is_mm = ids == CTX
        np.testing.assert_array_equal(
            idx_mm.numpy()[~is_mm], idx_ref.numpy()[~is_mm]
        )
        np.testing.assert_allclose(
            gate_mm.numpy()[~is_mm], gate_ref.numpy()[~is_mm]
        )
        logits = hidden.reshape([-1, 64]).matmul(mm.weight, transpose_y=True)
        scores = F.sigmoid(logits.cast("float32")).numpy()
        want = _topk_sets(scores + mm_bias[None], 2)
        for row in np.nonzero(is_mm)[0]:
            chosen = idx_mm.numpy()[row]
            self.assertEqual(set(chosen.tolist()), want[row])
            # Weights come from the bias-free scores, normalized (sigmoid).
            w = scores[row, chosen]
            np.testing.assert_allclose(
                gate_mm.numpy()[row], w / w.sum(), rtol=1e-5
            )
        self.assertEqual(int(mm.multimodal_expert_usage.sum()), is_mm.sum() * 2)


class TestHashLayerAuxLoss(_CPPatch):
    HASH = {
        "moe_n_hash_layers": 1,
        "actual_vocab_size": 64,
        "scoring_func": "sigmoid",
    }

    def _run(self, **overrides):
        router = _router(layer=0, **{**self.HASH, **overrides})
        if getattr(router, "multimodal_bias_enabled", False):
            router.multimodal_e_score_correction_bias.set_value(
                paddle.to_tensor(np.random.randn(8).astype("float32"))
            )
        hidden = paddle.randn([2, 6, 64])
        out = router(hidden, input_ids=paddle.to_tensor(INPUT_IDS))
        logits = hidden.reshape([-1, 64]).matmul(
            router.weight, transpose_y=True
        )
        valid = (_flat(INPUT_IDS) != PAD).astype("float32")[:, None]
        gates = F.sigmoid(logits.cast("float32")).numpy() * valid
        return router, out, gates, out[4].numpy(), valid

    def test_off_without_multimodal_bias(self):
        _, out, _, _, _ = self._run(moe_multimodal_router_bias=False)
        self.assertIsNone(out[6])

    def test_off_when_coef_is_zero(self):
        _, out, _, _, _ = self._run(router_aux_loss_coef=0.0)
        self.assertIsNone(out[6])

    def test_aux_loss_on_actual_routing_map(self):
        router, out, gates, mask, _ = self._run()
        want = 8 * float((gates.mean(0) * mask.mean(0)).sum())
        np.testing.assert_allclose(float(out[6]), want, rtol=1e-5)
        out[6].backward()
        self.assertGreater(float(router.weight.grad.abs().sum()), 0.0)

    def test_accuracy_compatible_and_experimental_normalization(self):
        """The two alternate normalization paths give the same aux loss as
        the default one for these inputs (sigmoid, no zero rows)."""
        router, out, _, mask, _ = self._run()
        logits = paddle.randn([12, 8])
        m = paddle.to_tensor(mask)
        want = float(
            router._hash_layer_aux_loss(logits, m, None, 6, 2, None, None)
        )
        router.use_accuracy_compatible = True
        router.config.gpt_model_use_experimental_version = True
        got = float(
            router._hash_layer_aux_loss(logits, m, None, 6, 2, None, None)
        )
        np.testing.assert_allclose(got, want, rtol=1e-5)

    def test_seq_aux_loss_on_actual_routing_map(self):
        router, out, gates, mask, valid = self._run(
            moe_router_load_balancing_type="seq_aux_loss"
        )
        k, e = 2, 8
        probs = gates / (gates.sum(-1, keepdims=True) + 1e-20)
        probs, mask, valid = (
            probs.reshape(2, 6, e),
            mask.reshape(2, 6, e),
            valid.reshape(2, 6),
        )
        denom = valid.sum(-1, keepdims=True)
        cost = mask.sum(1) / (denom * k / e)
        want = (cost * probs.sum(1) / denom).sum(1).mean()
        np.testing.assert_allclose(float(out[6]), want, rtol=1e-5)
        out[6].backward()
        self.assertGreater(float(router.weight.grad.abs().sum()), 0.0)


class _Group:
    def __init__(self, nranks):
        self.nranks = nranks


class _HCG:
    def __init__(self, mp=1, dp=1, sd=1):
        self.groups = {"mp": _Group(mp), "dp": _Group(dp), "sd": _Group(sd)}

    def get_model_parallel_group(self):
        return self.groups["mp"]

    def get_data_parallel_group(self):
        return self.groups["dp"]

    def get_sharding_parallel_group(self):
        return self.groups["sd"]


class TestCallback(unittest.TestCase):
    USAGE = np.array([[5, 0, 1, 1, 1, 1, 1, 6], [2] * 8], dtype=np.int64)

    def setUp(self):
        with patch(
            "paddlefleet.transformer.moe.moe_router.get_context_parallel_world_size",
            return_value=1,
        ):
            self.routers = paddle.nn.LayerList([_router(), _router(layer=2)])
        # Frozen gate weights must not block the multimodal update.
        for r, u in zip(self.routers, self.USAGE):
            r.weight.stop_gradient = True
            r.multimodal_expert_usage.set_value(paddle.to_tensor(u))
        self.text_before = [
            r.e_score_correction_bias.numpy().copy() for r in self.routers
        ]
        self.reduced = []

    def _run(self, hcg=None, args=None, use_mp=False, dsv4=False, model=None):
        fleet = types.SimpleNamespace()
        if hcg is not None:
            fleet._hcg = hcg
            fleet.get_hybrid_communicate_group = lambda: hcg
        callback = MoEMultimodalBiasAdjustCallback(lr=0.01, use_mp=use_mp)
        with (
            patch("paddlefleet.trainer.trainer_callback.fleet", fleet),
            patch(
                "paddlefleet.trainer.trainer_callback.dist.all_reduce",
                lambda t, group=None: self.reduced.append(group),
            ),
            patch(
                "paddlefleet.trainer.trainer_callback.use_dsv4_accuracy_compatible",
                return_value=dsv4,
            ),
        ):
            callback.on_optimizer_end(
                args
                or types.SimpleNamespace(
                    freeze_training=False, logging_steps=1
                ),
                types.SimpleNamespace(global_step=1),
                None,
                model=self.routers if model is None else model,
            )

    def _assert_unchanged(self, usage_kept=True):
        for r, u, t in zip(self.routers, self.USAGE, self.text_before):
            self.assertEqual(
                float(r.multimodal_e_score_correction_bias.abs().sum()), 0.0
            )
            if usage_kept:
                np.testing.assert_array_equal(
                    r.multimodal_expert_usage.numpy(), u
                )
            np.testing.assert_array_equal(r.e_score_correction_bias.numpy(), t)

    def test_update_and_reset(self):
        hcg = _HCG(mp=2, dp=2, sd=4)
        self._run(hcg=hcg, use_mp=True)
        g = hcg.groups
        # Both routers have 8 experts -> one stacked reduce per group.
        self.assertEqual(self.reduced, [g["mp"], g["dp"], g["sd"]])
        for r, u, t in zip(self.routers, self.USAGE, self.text_before):
            want = np.sign(u.mean() - u) * 0.01
            np.testing.assert_allclose(
                r.multimodal_e_score_correction_bias.numpy(), want, rtol=1e-6
            )
            self.assertEqual(int(r.multimodal_expert_usage.abs().sum()), 0)
            np.testing.assert_array_equal(r.e_score_correction_bias.numpy(), t)

    def test_single_rank_groups_skip_reduce(self):
        self._run(hcg=_HCG(), use_mp=True)
        self.assertEqual(self.reduced, [])
        self.assertGreater(
            float(
                self.routers[0].multimodal_e_score_correction_bias.abs().sum()
            ),
            0.0,
        )

    def test_non_logging_step_still_updates(self):
        self._run(
            hcg=_HCG(),
            args=types.SimpleNamespace(freeze_training=False, logging_steps=2),
        )
        for r, u in zip(self.routers, self.USAGE):
            np.testing.assert_allclose(
                r.multimodal_e_score_correction_bias.numpy(),
                np.sign(u.mean() - u) * 0.01,
                rtol=1e-6,
            )

    def test_without_hcg_only_reduces(self):
        # Mirrors MoECorrectionBiasAdjustCallback: all_reduce, no bias update.
        self._run(hcg=None)
        self.assertEqual(self.reduced, [None])
        self._assert_unchanged()

    def test_dsv4_accuracy_compatible_only_reduces(self):
        self._run(hcg=_HCG(dp=2), dsv4=True)
        self.assertEqual(self.reduced, [None])
        self._assert_unchanged()

    def test_freeze_training_skips(self):
        self._run(
            hcg=_HCG(dp=2),
            args=types.SimpleNamespace(freeze_training=True, logging_steps=1),
        )
        self.assertEqual(self.reduced, [])
        self._assert_unchanged()

    def test_model_without_multimodal_routers(self):
        self._run(hcg=_HCG(dp=2), model=paddle.nn.Linear(2, 2))
        self.assertEqual(self.reduced, [])
        self._assert_unchanged()


if __name__ == "__main__":
    unittest.main()
