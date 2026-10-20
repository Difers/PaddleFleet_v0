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

"""DSpark prediction layer: end-to-end smoke plus capture-side guard tests.

Builds a tiny dsv4-hybrid + mHC + MoE config, constructs a DsparkPredictionLayer
(draft layers forced to CSA ratio=0), feeds a synthetic capture (features /
final hidden / shared embedding + LM head weights), and checks that the forward
produces a well-shaped ``dspark_out``, the composite loss is finite, and the
gradient boundary holds (draft params trainable; shared target weights get no
gradient).

The guard tests cover the branches the happy path never takes: an incomplete
feature capture, a missing shared weight, the refiner/selector helpers,
the loss-logging helper, and the global-step env contract.
"""

import os
import types
import unittest

import numpy as np
import paddle

import paddlefleet.transformer.parallel_multi_token_prediction as pm
from paddlefleet.tensor_parallel.random import model_parallel_cuda_manual_seed
from paddlefleet.training.initialize import initialize_fleet
from paddlefleet.transformer.transformer_config import TransformerConfig
from paddlefleet.utils import init_method_normal, scaled_init_method_normal
from paddlefleet.utils.dspark.dspark_losses import (
    LossProfile,
    compute_dspark_loss,
)
from paddlefleet.utils.dspark.dspark_utils import DrafterOutput

initialize_fleet(strategy=paddle.distributed.fleet.DistributedStrategy())

_H, _NH, _NRS, _V = 256, 8, 4, 320


def _make_config(**overrides):
    cfg = {
        "num_hidden_layers": 3,
        "hidden_size": _H,
        "num_attention_heads": _NH,
        "num_key_value_heads": _NH,
        "intermediate_size": 128,
        "params_dtype": paddle.float32,
        "bf16": False,
        "use_bias": False,
        "hidden_dropout_prob": 0.0,
        "attention_dropout": 0.0,
        "normalization": "RMSNorm",
        "rms_norm_eps": 1e-5,
        "init_method": init_method_normal(0.02),
        "output_layer_init_method": scaled_init_method_normal(0.02, 1, 2.0),
        "sequence_parallel": False,
        "tensor_model_parallel_size": 1,
        "context_parallel_size": 1,
        "multi_latent_attention": True,
        "experimental_attention_variant": "dsv4_hybrid",
        "gated_attention": True,
        "gated_attn_use_q_lora": True,
        "use_qk_norm": True,
        "q_lora_rank": 64,
        "kv_lora_rank": 16,
        "qk_nope_head_dim": 16,
        "qk_rope_head_dim": 16,
        "qk_pos_emb_head_dim": 16,
        "v_head_dim": 32,
        "hybrid_mla_q_lora_rank": 64,
        "hybrid_mla_kv_lora_rank": 16,
        "hybrid_mla_qk_nope_head_dim": 16,
        "hybrid_mla_qk_rope_head_dim": 16,
        "hybrid_mla_v_head_dim": 32,
        "hybrid_mla_num_attention_heads": _NH,
        "hybrid_mla_num_key_value_heads": _NH,
        "o_groups": 4,
        "o_lora_rank": 32,
        "rope_type": "rope",
        "rotary_base": 10000.0,
        "rotary_percent": 1.0,
        "csa_compress_ratios": [0, 0, 0],
        "csa_window_size": 16,
        "dsa_index_n_heads": 4,
        "dsa_index_head_dim": 32,
        "dsa_index_topk": 8,
        "dsa_indexer_loss_coeff": 1.0,
        "dsa_indexer_use_sparse_loss": False,
        "dsa_indexer_rotary_interleaved": False,
        "apply_rope_fusion": False,
        "attention_softmax_in_fp32": True,
        "masked_softmax_fusion": False,
        "softmax_type": "vanilla",
        "csa_indexer_backend": "unfused",
        "csa_sparse_attn_backend": "unfused",
        "csa_dense_mode": False,
        "moe_layer_freq": [0, 1, 1],
        "n_routed_experts": 8,
        "moe_intermediate_size": 64,
        "num_experts_per_tok": 2,
        "n_shared_experts": 1,
        "moe_token_dispatcher_type": "allgather",
        "scoring_func": "sigmoid",
        "topk_method": "noaux_tc",
        "n_group": 1,
        "topk_group": 1,
        "norm_topk_prob": True,
        "routed_scaling_factor": 1.0,
        "moe_router_load_balancing_type": "seq_aux_loss",
        "enable_hyper_connections": True,
        "num_residual_streams": _NRS,
        "mhc_sinkhorn_iterations": 5,
        "mhc_init_gating_factor": 0.01,
        "high_precision_mhc": True,
        "use_fused_mhc": False,
        "dspark_target_layer_ids": [1, 2],
        "dspark_block_size": 3,
        "dspark_noise_token_id": 2,
        "dspark_markov_rank": 16,
        "dspark_loss_scaling_factor": 1.0,
        "add_dspark_loss": True,
    }
    cfg.update(overrides)
    config = TransformerConfig(**cfg)
    config.dtype = "float32"
    return config


class TestDsparkConfigGuards(unittest.TestCase):
    """``TransformerConfig.__post_init__`` guards around the DSpark switch."""

    def test_mtp_target_is_rejected_when_dspark_is_active(self):
        # The draft reads main-stream input_ids/labels, but an MTP target
        # appends its K-token tail to them; the combination is refused rather
        # than left to the draft's coordinate handling.
        with self.assertRaises(ValueError) as caught:
            _make_config(num_nextn_predict_layers=2)
        message = str(caught.exception)
        self.assertIn("num_nextn_predict_layers==0", message)
        self.assertIn("DSpark", message)

    def test_dspark_without_mtp_builds(self):
        config = _make_config()
        self.assertEqual(int(config.num_nextn_predict_layers), 0)


class TestDsparkRecomputeTier(unittest.TestCase):
    """The target's recompute tier is inherited at the *draft's* depth.

    ``recompute_num_layers`` is validated against the config's own layer count
    (``need_recompute_in_first_n`` / ``_in_block``), while ``drafter_provider``
    rewrites ``num_hidden_layers`` to the draft length. A target-side value
    deeper than the draft stack is legal for the target and impossible for the
    draft, so it has to be clamped rather than inherited verbatim.
    """

    @staticmethod
    def _draft(**overrides):
        # A 48-layer target whose draft keeps only two layers: any
        # recompute_num_layers above 2 is fine for the target, not for the draft.
        config = _make_config(
            num_hidden_layers=48,
            csa_compress_ratios=[0] * 48,
            moe_layer_freq=[0] + [1] * 47,
            dspark_target_layer_ids=[20, 40],
            **overrides,
        )
        return pm.DsparkPredictionLayer(config, vocab_size=_V, layer_number=0)

    def _tier(self, **overrides):
        layer = self._draft(**overrides)
        return layer.draft_config.recompute_num_layers, [
            sub.full_recompute for sub in layer.layers
        ]

    def test_a_tier_deeper_than_the_draft_is_clamped(self):
        # Unclamped this raises a bare AssertionError out of
        # need_recompute_in_first_n / _in_block during build_spec_layer.
        for method in ("first_n", "block"):
            with self.subTest(method=method):
                num_layers, flags = self._tier(
                    recompute_granularity="full",
                    recompute_method=method,
                    recompute_num_layers=8,
                )
                self.assertEqual(num_layers, 2)
                self.assertEqual(flags, [True, True])

    def test_a_tier_within_the_draft_is_kept(self):
        # The clamp is a ceiling, not a rewrite: a shallower tier still means
        # "the first layer only", exactly as it would on the target.
        num_layers, flags = self._tier(
            recompute_granularity="full",
            recompute_method="first_n",
            recompute_num_layers=1,
        )
        self.assertEqual(num_layers, 1)
        self.assertEqual(flags, [True, False])

    def test_uniform_is_untouched(self):
        # uniform pins recompute_num_layers to 1, so it is already depth-free.
        num_layers, flags = self._tier(
            recompute_granularity="full",
            recompute_method="uniform",
            recompute_num_layers=1,
        )
        self.assertEqual(num_layers, 1)
        self.assertEqual(flags, [True, True])

    def test_a_disabled_tier_stays_disabled(self):
        num_layers, flags = self._tier()
        self.assertIsNone(num_layers)
        self.assertEqual(flags, [False, False])


class TestDsparkPredictionLayer(unittest.TestCase):
    def test_forward_shapes_loss_and_gradient_boundary(self):
        model_parallel_cuda_manual_seed(42, tp_rank=0, ep_rank=0, etp_rank=0)
        paddle.seed(42)
        config = _make_config()

        config.dspark_num_anchors = 4
        layer = pm.DsparkPredictionLayer(config, vocab_size=_V, layer_number=0)
        self.assertEqual(len(layer.layers), 2)  # target_layer_ids=[1,2]

        B, S = 1, 16
        F = len(config.dspark_target_layer_ids) * _H
        rng = np.random.default_rng(0)
        input_ids = paddle.to_tensor(
            rng.integers(1, _V, size=(B, S)).astype("int64")
        )
        labels = paddle.to_tensor(
            rng.integers(1, _V, size=(B, S)).astype("int64")
        )
        position_ids = (
            paddle.arange(S, dtype="int64").reshape([1, S]).expand([B, S])
        )
        feats = paddle.randn([B, S, F])
        feats.stop_gradient = True
        final_hidden = paddle.randn([B, S, _H])
        final_hidden.stop_gradient = True
        emb_w = paddle.randn([_V, _H])
        emb_w.stop_gradient = False
        head_w = paddle.randn([_V, _H])
        head_w.stop_gradient = False

        dict_args = {
            "input_ids": input_ids,
            "labels": labels,
            "position_ids": position_ids,
            "dspark_features": feats,
            "dspark_final_hidden": final_hidden,
            "dspark_embedding_weight": emb_w,
            "dspark_head_weight": head_w,
            "hidden_states": paddle.randn([B, S, _NRS * _H]),
        }
        out = layer.forward(dict_args)["dspark_out"]
        A, K = 4, 3
        self.assertEqual(out["draft_hidden_states"].shape, [B, A, K, _H])
        self.assertEqual(out["target_ids"].shape, [B, A, K])

        loss = compute_dspark_loss(
            DrafterOutput(
                draft_logits=None,
                target_ids=out["target_ids"],
                eval_mask=out["eval_mask"],
                block_keep_mask=out["block_keep_mask"],
                confidence_pred=out["confidence_pred"],
                draft_hidden_states=out["draft_hidden_states"],
                aligned_target_hidden_states=out[
                    "aligned_target_hidden_states"
                ],
                previous_token_ids=out["previous_token_ids"],
                logits_projection=out["logits_projection"],
                markov_head=out["markov_head"],
            ),
            LossProfile(),
        )
        self.assertTrue(np.isfinite(float(loss)))
        loss.backward()
        self.assertIsNotNone(layer.feature_projection.weight.grad)
        self.assertIsNone(head_w.grad)  # shared, read-only
        self.assertIsNone(emb_w.grad)  # shared, read-only


class TestDsparkForwardCpFrame(unittest.TestCase):
    """``DsparkPredictionLayer.forward`` must hand the draft a LOCAL window.

    A single card cannot run a real CP group, but the slice is a
    communication-free index, so the honest claim here is the equivalence below:
    the global-ids run and the pre-sharded run must produce identical draft
    supervision. A wrong layout destroys exactly that observable, silently.
    """

    def test_global_ids_are_sliced_to_the_pre_sharded_frame(self):
        model_parallel_cuda_manual_seed(42, tp_rank=0, ep_rank=0, etp_rank=0)
        paddle.seed(42)
        config = _make_config()
        config.dspark_num_anchors = 4
        # Pinned so the test owns the layout instead of inheriting the default.
        config.cp_balance_mode = "contiguous_allgather"
        layer = pm.DsparkPredictionLayer(config, vocab_size=_V, layer_number=0)
        # cp_size=2 with no collective ever issued: only the frame is under
        # test, and the slice is deliberately communication-free.
        layer._pg_collection = types.SimpleNamespace(
            cp=types.SimpleNamespace(rank=0, nranks=2)
        )

        B, S = 1, 16
        F = len(config.dspark_target_layer_ids) * _H
        rng = np.random.default_rng(0)
        local_ids = paddle.to_tensor(
            rng.integers(1, _V, size=(B, S)).astype("int64")
        )
        local_labels = paddle.to_tensor(
            rng.integers(1, _V, size=(B, S)).astype("int64")
        )
        # Rank 0 of a contiguous 2-way split owns the first half, so the global
        # tensor is built as [this rank's window | someone else's window].
        global_ids = paddle.concat([local_ids, local_ids + 1], axis=1)
        global_labels = paddle.concat([local_labels, local_labels + 1], axis=1)
        global_position_ids = paddle.arange(2 * S, dtype="int64").reshape(
            [1, 2 * S]
        )
        # One document spanning the global sequence: the mask is global under
        # every cp_shard_source, so it must NOT be sliced alongside the ids.
        mask = paddle.full([1, 2 * S], 2 * S, dtype="int64")
        feats = paddle.randn([B, S, F])
        feats.stop_gradient = True
        final_hidden = paddle.randn([B, S, _H])
        final_hidden.stop_gradient = True
        emb_w = paddle.randn([_V, _H])
        head_w = paddle.randn([_V, _H])

        def _run(ids, labels, position_ids, dataflow, erndata):
            # The two flags are what cp_shard_source() reads. The draft config
            # was deepcopied at __init__ with both forced False, so flipping
            # them here moves the ids' frame and nothing else.
            config.experimental_dataflow = dataflow
            config.use_erndata = erndata
            args = {
                "input_ids": ids,
                "labels": labels,
                "position_ids": position_ids,
                "attn_mask_startend_row_indices": mask,
                "dspark_features": feats,
                "dspark_final_hidden": final_hidden,
                "dspark_embedding_weight": emb_w,
                "dspark_head_weight": head_w,
                "hidden_states": paddle.randn([B, S, _NRS * _H]),
            }
            return layer.forward(args)["dspark_out"]

        # "scatter": the model was handed GLOBAL ids (it scatters only
        # inputs_embeds), so the draft slices them.
        scattered = _run(
            global_ids, global_labels, global_position_ids, True, False
        )
        # "loader": the trainer already sharded the batch, so these ids are
        # local and must pass through untouched.
        pre_sharded = _run(
            local_ids, local_labels, global_position_ids[:, :S], False, False
        )

        for key in ("target_ids", "eval_mask", "previous_token_ids"):
            np.testing.assert_array_equal(
                scattered[key].numpy(), pre_sharded[key].numpy()
            )
        np.testing.assert_allclose(
            scattered["draft_hidden_states"].numpy(),
            pre_sharded["draft_hidden_states"].numpy(),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_zigzag_global_ids_slice_to_the_local_frame(self):
        # ``dualchunk_allgather`` concatenates two globally distant intervals,
        # so the seam has to reach the draft's window/block guards. Rank 0 of a
        # 2-way zigzag owns global [0, 8) and [24, 32) of a 32-long sequence.
        model_parallel_cuda_manual_seed(42, tp_rank=0, ep_rank=0, etp_rank=0)
        paddle.seed(42)
        config = _make_config()
        config.dspark_num_anchors = 4
        config.cp_balance_mode = "dualchunk_allgather"
        layer = pm.DsparkPredictionLayer(config, vocab_size=_V, layer_number=0)
        layer._pg_collection = types.SimpleNamespace(
            cp=types.SimpleNamespace(rank=0, nranks=2)
        )

        B, S = 1, 16
        half = S // 2
        F = len(config.dspark_target_layer_ids) * _H
        rng = np.random.default_rng(0)
        local_ids = paddle.to_tensor(
            rng.integers(1, _V, size=(B, S)).astype("int64")
        )
        local_labels = paddle.to_tensor(
            rng.integers(1, _V, size=(B, S)).astype("int64")
        )
        # [this rank's first interval | other ranks' | this rank's last one].
        filler = paddle.full([B, S], _V - 1, dtype="int64")
        global_ids = paddle.concat(
            [local_ids[:, :half], filler, local_ids[:, half:]], axis=1
        )
        global_labels = paddle.concat(
            [local_labels[:, :half], filler, local_labels[:, half:]], axis=1
        )
        global_position_ids = paddle.arange(2 * S, dtype="int64").reshape(
            [1, 2 * S]
        )
        local_position_ids = paddle.concat(
            [
                global_position_ids[:, :half],
                global_position_ids[:, 2 * S - half :],
            ],
            axis=1,
        )
        # One document spanning the global sequence, so the document filter
        # cannot separate the seam and only the seam guard can.
        mask = paddle.full([1, 2 * S], 2 * S, dtype="int64")
        feats = paddle.randn([B, S, F])
        feats.stop_gradient = True
        final_hidden = paddle.randn([B, S, _H])
        final_hidden.stop_gradient = True
        emb_w = paddle.randn([_V, _H])
        head_w = paddle.randn([_V, _H])

        def _run(ids, labels, position_ids, dataflow):
            config.experimental_dataflow = dataflow
            config.use_erndata = False
            args = {
                "input_ids": ids,
                "labels": labels,
                "position_ids": position_ids,
                "attn_mask_startend_row_indices": mask,
                "dspark_features": feats,
                "dspark_final_hidden": final_hidden,
                "dspark_embedding_weight": emb_w,
                "dspark_head_weight": head_w,
                "hidden_states": paddle.randn([B, S, _NRS * _H]),
            }
            return layer.forward(args)["dspark_out"]

        scattered = _run(global_ids, global_labels, global_position_ids, True)
        pre_sharded = _run(local_ids, local_labels, local_position_ids, False)

        for key in ("target_ids", "eval_mask", "previous_token_ids"):
            np.testing.assert_array_equal(
                scattered[key].numpy(), pre_sharded[key].numpy()
            )
        np.testing.assert_allclose(
            scattered["draft_hidden_states"].numpy(),
            pre_sharded["draft_hidden_states"].numpy(),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_local_ids_under_a_scatter_source_are_rejected_not_sliced(self):
        # The mirror-image mistake: local ids under a config whose source says
        # they are global. A shape-sniffing guard would skip the slice and
        # supervise against a cp_size-times-wrong sequence; refuse instead.
        config = _make_config()
        config.dspark_num_anchors = 4
        layer = pm.DsparkPredictionLayer(config, vocab_size=_V, layer_number=0)
        layer._pg_collection = types.SimpleNamespace(
            cp=types.SimpleNamespace(rank=0, nranks=2)
        )
        config.experimental_dataflow = True
        config.use_erndata = False

        B, S = 1, 16
        args = {
            "input_ids": paddle.zeros([B, S], dtype="int64"),
            "labels": paddle.zeros([B, S], dtype="int64"),
            "dspark_features": paddle.randn(
                [B, S, len(config.dspark_target_layer_ids) * _H]
            ),
            "dspark_final_hidden": paddle.randn([B, S, _H]),
            "dspark_embedding_weight": paddle.randn([_V, _H]),
            "dspark_head_weight": paddle.randn([_V, _H]),
        }
        with self.assertRaises(ValueError) as caught:
            layer.forward(args)
        message = str(caught.exception)
        self.assertIn(
            "input_ids must be delivered in GLOBAL coordinates", message
        )
        self.assertIn("expected seq len 32", message)


_STEP_KEYS = ("TRAINER_GLOBAL_STEP", "PDC_INIT_STEP")
_R = 16  # dspark_markov_rank in _make_config()


class TestDsparkGlobalStep(unittest.TestCase):
    """``_dspark_global_step`` reads the MTP env contract, defaulting to 0."""

    def test_reads_the_env_contract(self):
        cases = [
            ({"TRAINER_GLOBAL_STEP": "17"}, 17),
            ({"PDC_INIT_STEP": "5"}, 5),
            ({"TRAINER_GLOBAL_STEP": "-3"}, 0),
            ({"TRAINER_GLOBAL_STEP": "not-a-number"}, 0),
            ({"TRAINER_GLOBAL_STEP": "abc", "PDC_INIT_STEP": "9"}, 9),
            ({}, 0),
        ]
        for env, expected in cases:
            with self.subTest(env=env):
                saved = {key: os.environ.pop(key, None) for key in _STEP_KEYS}
                try:
                    os.environ.update(env)
                    self.assertEqual(pm._dspark_global_step(), expected)
                finally:
                    for key in _STEP_KEYS:
                        os.environ.pop(key, None)
                    os.environ.update(
                        {k: v for k, v in saved.items() if v is not None}
                    )


class TestDsparkRefiners(unittest.TestCase):
    """The two refiners are alternative; both expose predecessor states."""

    @classmethod
    def setUpClass(cls):
        model_parallel_cuda_manual_seed(42, tp_rank=0, ep_rank=0, etp_rank=0)

    def test_markov_head_exposes_predecessor_embeddings(self):
        head = pm.MarkovHead(vocab_size=_V, rank=_R)
        token_ids = paddle.randint(0, _V, [2, 3], dtype="int32")
        embeddings = head.get_prev_embeddings(token_ids)
        self.assertEqual(list(embeddings.shape), [2, 3, _R])

    def test_candidate_selector_rejects_invalid_dimensions(self):
        cases = {
            "hidden_size": {"hidden_size": 0},
            "vocab_size": {"vocab_size": 0},
            "state_rank": {"state_rank": 0},
            "top_k_above_vocab": {"top_k": _V + 1},
            "top_k_zero": {"top_k": 0},
            "initializer_range": {"initializer_range": -0.1},
        }
        for name, override in cases.items():
            with self.subTest(name=name):
                kwargs = {
                    "hidden_size": _H,
                    "vocab_size": _V,
                    "state_rank": _R,
                    "top_k": 4,
                    "initializer_range": 0.02,
                }
                kwargs.update(override)
                with self.assertRaises(ValueError):
                    pm.CandidateSelector(**kwargs)


class _RecordingWriter:
    """Minimal tensorboard / wandb stand-in."""

    def __init__(self):
        self.scalars = {}

    def add_scalar(self, name, value, iteration):
        self.scalars[name] = float(value)

    def log(self, payload, iteration):
        self.scalars.update({k: float(v) for k, v in payload.items()})


class TestDSparkLossLoggingHelper(unittest.TestCase):
    """Accumulate -> reduce -> log -> clean, across every consumer."""

    def setUp(self):
        pm.DSparkLossLoggingHelper.tracker.clear()

    def tearDown(self):
        pm.DSparkLossLoggingHelper.tracker.clear()

    def test_accumulates_then_cleans(self):
        helper = pm.DSparkLossLoggingHelper
        helper.save_loss_to_tracker(paddle.to_tensor([2.0]))
        helper.save_loss_to_tracker(paddle.to_tensor([3.0]))
        self.assertAlmostEqual(float(helper.tracker["value"][0]), 5.0)
        helper.clean_loss_in_tracker()
        self.assertAlmostEqual(float(helper.tracker["value"][0]), 0.0)
        self.assertIsNone(helper.tracker["reduce_group"])
        self.assertIsNone(helper.tracker["avg_group"])

    def test_reduce_is_a_noop_without_a_tracked_value(self):
        helper = pm.DSparkLossLoggingHelper
        helper.reduce_loss_in_tracker()  # nothing saved yet -> early return
        helper.save_loss_to_tracker(paddle.to_tensor([1.0]))
        helper.reduce_loss_in_tracker()  # no groups -> no collective
        self.assertAlmostEqual(float(helper.tracker["value"][0]), 1.0)

    def test_reduce_all_reduces_when_groups_are_set(self):
        helper = pm.DSparkLossLoggingHelper
        group = paddle.distributed.new_group(
            ranks=list(range(paddle.distributed.get_world_size()))
        )
        helper.save_loss_to_tracker(
            paddle.to_tensor([4.0]), reduce_group=group, avg_group=group
        )
        helper.reduce_loss_in_tracker()
        # A world-sized group reduces to the identity on a single card.
        self.assertAlmostEqual(float(helper.tracker["value"][0]), 4.0)

    def test_track_writes_to_every_consumer_and_resets(self):
        helper = pm.DSparkLossLoggingHelper
        helper.save_loss_to_tracker(paddle.to_tensor([4.0]))
        writer, wandb_writer, total = (
            _RecordingWriter(),
            _RecordingWriter(),
            {},
        )
        helper.track_dspark_metrics(0.5, 7, writer, wandb_writer, total)
        self.assertAlmostEqual(writer.scalars["dspark loss"], 2.0)
        self.assertAlmostEqual(wandb_writer.scalars["dspark loss"], 2.0)
        self.assertAlmostEqual(float(total["dspark loss"]), 2.0)
        self.assertAlmostEqual(float(helper.tracker["value"][0]), 0.0)

    def test_track_is_a_noop_without_a_recorded_loss(self):
        pm.DSparkLossLoggingHelper.track_dspark_metrics(1.0, 0, None)


class TestDSparkLossAutoScaler(unittest.TestCase):
    def test_carries_the_dspark_gradient_scaled_by_the_main_loss(self):
        source = paddle.randn([2, 3])
        source.stop_gradient = False
        main_loss = source * 2.0  # non-leaf, as in the trainer
        dspark_loss = paddle.to_tensor([1.0])
        dspark_loss.stop_gradient = False
        pm.DSparkLossAutoScaler.set_loss_scale(paddle.to_tensor(0.25))
        carried = pm.DSparkLossAutoScaler.apply(main_loss, dspark_loss)
        carried.backward()
        np.testing.assert_allclose(dspark_loss.grad.numpy(), [0.25], rtol=1e-6)
        np.testing.assert_allclose(
            source.grad.numpy(), 2.0 * np.ones([2, 3]), rtol=1e-6
        )


class TestDsparkCaptureGuards(unittest.TestCase):
    """Capture-side guards: what happens when the target side under-delivers."""

    @classmethod
    def setUpClass(cls):
        model_parallel_cuda_manual_seed(42, tp_rank=0, ep_rank=0, etp_rank=0)
        paddle.seed(42)
        cls.config = _make_config()
        cls.config.dspark_num_anchors = 4
        cls.layer = pm.DsparkPredictionLayer(
            cls.config, vocab_size=_V, layer_number=0
        )

    @classmethod
    def _captured_args(cls):
        """dict_args shaped like a *successful* capture, minus the features."""
        rng = np.random.default_rng(0)
        args = {
            "input_ids": paddle.to_tensor(
                rng.integers(1, _V, size=(1, 16)).astype("int64")
            ),
            "labels": paddle.to_tensor(
                rng.integers(1, _V, size=(1, 16)).astype("int64")
            ),
            "dspark_embedding_weight": paddle.randn([_V, _H]),
            "dspark_head_weight": paddle.randn([_V, _H]),
        }
        for layer_id in cls.config.dspark_target_layer_ids:
            args[f"dspark_target_hidden_{layer_id}"] = paddle.randn([1, 16, _H])
        args["dspark_target_last_hidden_states"] = paddle.randn([1, 16, _H])
        return args

    def test_incomplete_capture_is_reported(self):
        args = self._captured_args()
        del args[
            f"dspark_target_hidden_{self.config.dspark_target_layer_ids[-1]}"
        ]
        with self.assertRaises(RuntimeError) as caught:
            self.layer._resolve_inputs(args)
        message = str(caught.exception)
        self.assertIn("feature capture is incomplete", message)
        self.assertIn("dspark_target_hidden layers=", message)

    def test_missing_final_hidden_is_reported(self):
        args = self._captured_args()
        del args["dspark_target_last_hidden_states"]
        with self.assertRaises(RuntimeError) as caught:
            self.layer._resolve_inputs(args)
        self.assertIn("MISSING", str(caught.exception))

    def test_missing_shared_embedding_weight_is_reported(self):
        args = self._captured_args()
        del args["dspark_embedding_weight"]
        with self.assertRaises(RuntimeError) as caught:
            self.layer._resolve_inputs(args)
        message = str(caught.exception)
        self.assertIn("shared embedding weight missing", message)
        self.assertIn("dspark_embedding_weight", message)

    def test_missing_shared_head_weight_is_reported(self):
        args = self._captured_args()
        del args["dspark_head_weight"]
        with self.assertRaises(RuntimeError) as caught:
            self.layer._resolve_inputs(args)
        message = str(caught.exception)
        self.assertIn("shared LM head weight missing", message)
        self.assertIn("dspark_head_weight", message)

    def test_share_embedding_rejects_a_layer_without_a_weight(self):
        """The pass-through must fail loudly rather than publish nothing."""
        args = {}
        # Neither ``embedding_weight`` nor ``weight``: the shared layer cannot
        # be used and the draft must not silently proceed.
        with self.assertRaises(RuntimeError) as caught:
            pm.dspark_share_embedding(types.SimpleNamespace(), args)
        message = str(caught.exception)
        self.assertIn("shared embedding weight missing", message)
        self.assertIn("SimpleNamespace", message)
        self.assertEqual(args, {})

    def test_share_lm_head_publishes_only_present_tensors(self):
        """Absent bias / multimax stay absent: dict_args values must be tensors."""
        args = self._captured_args()
        weight = paddle.randn([_V, _H])
        pm.dspark_share_lm_head(
            types.SimpleNamespace(weight=weight, bias=None), args
        )
        self.assertIs(args["dspark_head_weight"], weight)
        self.assertNotIn("dspark_head_bias", args)
        self.assertNotIn("dspark_head_multimax_ranges", args)

    def test_share_embedding_prefers_embedding_weight_over_weight(self):
        """``GPTEmbedding`` exposes ``embedding_weight``; a bare Linear does not."""
        args = self._captured_args()
        preferred = paddle.randn([_V, _H])
        fallback = paddle.randn([_V, _H])
        pm.dspark_share_embedding(
            types.SimpleNamespace(embedding_weight=preferred, weight=fallback),
            args,
        )
        self.assertIs(args["dspark_embedding_weight"], preferred)
        pm.dspark_share_embedding(types.SimpleNamespace(weight=fallback), args)
        self.assertIs(args["dspark_embedding_weight"], fallback)

    def test_forward_without_position_ids_uses_the_default_arange(self):
        """``dict_args`` need not carry ``position_ids``; the backbone defaults it."""
        args = self._captured_args()
        self.assertNotIn("position_ids", args)
        out = self.layer.forward(args)
        self.assertIn("dspark_out", out)
        draft_hidden = out["dspark_out"]["draft_hidden_states"]
        self.assertEqual(int(draft_hidden.shape[0]), 1)
        self.assertEqual(int(draft_hidden.shape[-1]), _H)

    def test_layer_spec_builder_returns_the_draft_layer(self):
        spec = pm.get_dspark_layer_spec_for_backend(self.config, _V, 0)
        self.assertIs(spec.layer, pm.DsparkPredictionLayer)
        self.assertEqual(spec.extra_kwargs["vocab_size"], _V)
        self.assertEqual(spec.extra_kwargs["layer_number"], 0)

    def test_resolved_capture_round_trips(self):
        args = self._captured_args()
        features, final_hidden, weight, head, ids, labels, mask = (
            self.layer._resolve_inputs(args)
        )
        self.assertEqual(list(features.shape), [1, 16, 2 * _H])
        self.assertEqual(list(final_hidden.shape), [1, 16, _H])
        self.assertIs(weight, args["dspark_embedding_weight"])
        self.assertIs(head["weight"], args["dspark_head_weight"])
        self.assertIsNone(head["bias"])
        self.assertIsNone(head["multimax_ranges"])
        self.assertIsNone(head["multimax_ts"])
        self.assertIs(ids, args["input_ids"])
        self.assertIs(labels, args["labels"])
        self.assertIsNone(mask)
        # The capture keys are consumed, not left behind for the loss stage.
        self.assertNotIn("dspark_target_last_hidden_states", args)

    def test_sample_ids_prefers_the_pipeline_supplied_ids(self):
        args = {"input_ids": paddle.to_tensor([[1, 2, 3]], dtype="int64")}
        self.assertEqual(
            self.layer._sample_ids(
                {"sample_ids": ["row-a", "row-b"]}, args["input_ids"]
            ),
            ["row-a", "row-b"],
        )

    def test_build_schedule_node_wraps_forward(self):
        node = self.layer.build_schedule_node()
        self.assertIsNotNone(node)


if __name__ == "__main__":
    unittest.main()
