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

"""End-to-end tiny-model smoke: full ``GPTModel`` pipeline with dspark enabled.

Unlike ``test_dspark_prediction_layer.py`` (which constructs a standalone
``DsparkPredictionLayer`` and hand-feeds ``dict_args["dspark_features"]``),
this test goes through ``gpt_builder`` so the feature-capture bridge
(``GPTModel._setup_dspark_capture``) runs for real: hooks on the backbone
layers / final norm populate the capture dict that ``DsparkPredictionLayer``
reads, exercising the full capture path.
"""

import random
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import paddle
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import NoPipelineParallel

import paddlefleet.parallel_state as ps
from paddlefleet.gpt_builders import gpt_builder
from paddlefleet.models.gpt import GPTConfig, gpt_model as gm
from paddlefleet.models.gpt.gpt_embedding import GPTEmbedding
from paddlefleet.models.gpt.lm_head import GPTLMHead
from paddlefleet.tensor_parallel.random import model_parallel_cuda_manual_seed
from paddlefleet.transformer import parallel_multi_token_prediction as pm

_H, _NH, _NRS, _V = 256, 8, 4, 320


def _make_config(add_dspark_loss: bool) -> GPTConfig:
    config = GPTConfig(
        num_hidden_layers=3,
        hidden_size=_H,
        num_attention_heads=_NH,
        num_key_value_heads=_NH,
        intermediate_size=128,
        vocab_size=_V,
        max_sequence_length=64,
        params_dtype=paddle.float32,
        bf16=False,
        use_bias=False,
        hidden_dropout_prob=0.0,
        attention_dropout=0.0,
        normalization="RMSNorm",
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
        sequence_parallel=False,
        tensor_model_parallel_size=1,
        context_parallel_size=1,
        multi_latent_attention=True,
        experimental_attention_variant="dsv4_hybrid",
        gated_attention=True,
        gated_attn_use_q_lora=True,
        use_qk_norm=True,
        q_lora_rank=64,
        kv_lora_rank=16,
        qk_nope_head_dim=16,
        qk_rope_head_dim=16,
        qk_pos_emb_head_dim=16,
        v_head_dim=32,
        hybrid_mla_q_lora_rank=64,
        hybrid_mla_kv_lora_rank=16,
        hybrid_mla_qk_nope_head_dim=16,
        hybrid_mla_qk_rope_head_dim=16,
        hybrid_mla_v_head_dim=32,
        hybrid_mla_num_attention_heads=_NH,
        hybrid_mla_num_key_value_heads=_NH,
        o_groups=4,
        o_lora_rank=32,
        position_embedding_type="rope",
        rope_type="rope",
        rotary_base=10000.0,
        rotary_percent=1.0,
        csa_compress_ratios=[0, 0, 0],
        csa_window_size=16,
        dsa_index_n_heads=4,
        dsa_index_head_dim=32,
        dsa_index_topk=8,
        dsa_indexer_loss_coeff=1.0,
        dsa_indexer_use_sparse_loss=False,
        dsa_indexer_rotary_interleaved=False,
        apply_rope_fusion=False,
        attention_softmax_in_fp32=True,
        masked_softmax_fusion=False,
        softmax_type="vanilla",
        csa_indexer_backend="unfused",
        csa_sparse_attn_backend="unfused",
        csa_dense_mode=False,
        moe_layer_freq=[0, 1, 1],
        n_routed_experts=8,
        moe_intermediate_size=64,
        num_experts_per_tok=2,
        n_shared_experts=1,
        moe_token_dispatcher_type="allgather",
        scoring_func="sigmoid",
        topk_method="noaux_tc",
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
        moe_router_load_balancing_type="seq_aux_loss",
        enable_hyper_connections=True,
        num_residual_streams=_NRS,
        mhc_sinkhorn_iterations=5,
        mhc_init_gating_factor=0.01,
        high_precision_mhc=True,
        use_fused_mhc=False,
        dspark_target_layer_ids=[1, 2],
        dspark_block_size=3,
        dspark_num_anchors=4,
        dspark_noise_token_id=2,
        dspark_markov_rank=16,
        dspark_loss_scaling_factor=1.0,
        add_dspark_loss=add_dspark_loss,
    )
    config.dtype = "float32"
    return config


_FLEET_READY = False
_STRATEGY = None


def _ensure_fleet_ready():
    """Initialize fleet + model-parallel once per process (idempotent).

    ``fleet.init`` / ``initialize_model_parallel`` can only run once per
    process (a second call trips ``global memory buffer is already
    initialized``), but both test methods below need the setup. Returns the
    ``DistributedStrategy`` for reuse by the caller.
    """
    global _FLEET_READY, _STRATEGY
    if _FLEET_READY:
        return _STRATEGY
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": 1,
        "pp_degree": 1,
        "sharding_degree": 1,
        "sep_degree": 1,
        "cp_degree": 1,
        "ep_degree": 1,
        "moe_sharding_degree": 1,
        "order": [
            "sharding",
            "moe_sharding",
            "pp",
            "sep",
            "cp",
            "dp",
            "ep",
            "mp",
        ],
    }
    fleet.init(is_collective=True, strategy=strategy)
    hcg = fleet.get_hybrid_communicate_group()
    ps.initialize_model_parallel(hcg)
    _STRATEGY = strategy
    _FLEET_READY = True
    return _STRATEGY


class TestDsparkEndToEndModel(unittest.TestCase):
    def _build_and_run(self, add_dspark_loss: bool):
        seed = 46
        random.seed(seed)
        np.random.seed(seed)
        paddle.seed(seed)
        strategy = _ensure_fleet_ready()
        model_parallel_cuda_manual_seed(seed, tp_rank=0, ep_rank=0, etp_rank=0)

        config = _make_config(add_dspark_loss)
        model = gpt_builder(config, num_stages=1)

        B, S = 1, 16
        rng = np.random.default_rng(0)
        input_ids = paddle.to_tensor(
            rng.integers(1, _V, size=(B, S)).astype("int64")
        )
        position_ids = (
            paddle.arange(S, dtype="int64").reshape([1, S]).expand([B, S])
        )
        attention_mask = paddle.ones((B, 1, S, S), dtype=bool)
        labels = paddle.to_tensor(
            rng.integers(1, _V, size=(B, S)).astype("int64")
        )

        gpt_pipe_model = NoPipelineParallel(model, strategy)
        data = (
            {
                "input_ids": [input_ids],
                "position_ids": [position_ids],
                "attention_mask": [attention_mask],
                # DsparkPredictionLayer builds draft blocks from labels at the
                # embedding stage (captured by the pre-hook), independent of
                # the loss stage's own `labels` argument below.
                "labels": [labels],
            },
            [labels],
        )
        loss = gpt_pipe_model.forward_backward_pipeline(data)
        return model, loss

    def test_forward_without_dspark_grad_injection_still_produces_dspark_out(
        self,
    ):
        """``add_dspark_loss=False`` only gates the gradient carrier (language_loss.py
        DSparkLanguageLoss.forward), not whether the draft runs. The capture bridge,
        draft backbone and loss computation must still execute without error; dspark
        params legitimately get no gradient since the loss never enters backward.
        """
        model, loss = self._build_and_run(add_dspark_loss=False)
        self.assertTrue(np.isfinite(float(loss)))

    def test_forward_backward_with_dspark_grad_injection(self):
        model, loss = self._build_and_run(add_dspark_loss=True)
        self.assertTrue(np.isfinite(float(loss)))

        dspark_grad_params = [
            (name, param)
            for name, param in model.named_parameters()
            if "dspark" in name.lower()
            or "markov" in name.lower()
            or "feature_projection" in name
        ]
        self.assertTrue(len(dspark_grad_params) > 0)
        for name, param in dspark_grad_params:
            self.assertIsNotNone(
                param.grad, f"dspark param {name} has no gradient"
            )

        # Target embedding stays read-only to the draft: the only non-None
        # grad on it should come from the target's own LM loss, not from
        # dspark (checked indirectly: embedding grad is finite, not from
        # borrowed-weight corruption such as NaN/inf from the draft path).
        embed_param = next(
            param
            for name, param in model.named_parameters()
            if name.endswith("embed_tokens.weight")
        )
        self.assertIsNotNone(embed_param.grad)
        self.assertTrue(bool(paddle.isfinite(embed_param.grad).all()))


class TestDsparkSharedLayerAssembly(unittest.TestCase):
    """The draft must reuse the backbone embedding / LM head *objects*.

    ``GPTModel.get_layer_desc_list`` promotes both to ``SharedLayerDesc`` and
    the draft only reads the weights back off ``dict_args``. Three implicit
    contracts keep that correct, none of which the loss-finiteness assertions
    above can observe: paddle builds one instance per shared key, the key
    promotion is right for the (tie, MTP, dspark) matrix, and the draft gets
    that same parameter rather than a freshly initialised copy. These tests
    pin all three so a desc reorder cannot silently decouple draft from
    target.
    """

    def test_assembly_shares_one_embedding_and_head_instance(self):
        seed = 47
        random.seed(seed)
        np.random.seed(seed)
        paddle.seed(seed)
        strategy = _ensure_fleet_ready()
        model_parallel_cuda_manual_seed(seed, tp_rank=0, ep_rank=0, etp_rank=0)

        model = gpt_builder(_make_config(add_dspark_loss=False), num_stages=1)

        # One key each, and exactly one built instance behind it: a second
        # GPTEmbedding / GPTLMHead anywhere means a desc re-created the layer
        # instead of reusing the pivot.
        self.assertEqual(
            sorted(model.shared_layers.keys()), ["dspark_lm_head", "embed"]
        )
        sublayer_types = [type(m) for m in model.sublayers()]
        self.assertEqual(sublayer_types.count(GPTEmbedding), 1)
        self.assertEqual(sublayer_types.count(GPTLMHead), 1)

        embed = model.shared_layers["embed"]
        head = model.shared_layers["dspark_lm_head"]
        # The tail head desc re-registers the key rather than a new instance,
        # so the head position runs the very object built next to the draft.
        self.assertEqual(sum(1 for rf in model.run_function if rf is head), 1)
        self.assertIsNot(embed.embedding_weight, head.weight)

        # Capture what the draft is actually handed during a real forward.
        captured = {}
        original = pm.DsparkPredictionLayer._resolve_inputs

        def spy(dspark_self, dict_args):
            captured.update(dict_args)
            return original(dspark_self, dict_args)

        B, S = 1, 16
        rng = np.random.default_rng(1)
        input_ids = paddle.to_tensor(
            rng.integers(1, _V, size=(B, S)).astype("int64")
        )
        position_ids = (
            paddle.arange(S, dtype="int64").reshape([1, S]).expand([B, S])
        )
        attention_mask = paddle.ones((B, 1, S, S), dtype=bool)
        labels = paddle.to_tensor(
            rng.integers(1, _V, size=(B, S)).astype("int64")
        )

        with mock.patch.object(
            pm.DsparkPredictionLayer, "_resolve_inputs", spy
        ):
            pipe = NoPipelineParallel(model, strategy)
            pipe.forward_backward_pipeline(
                (
                    {
                        "input_ids": [input_ids],
                        "position_ids": [position_ids],
                        "attention_mask": [attention_mask],
                        "labels": [labels],
                    },
                    [labels],
                )
            )

        # Identity, not numerical equality: a copy holding the same values
        # would pass ``==`` and then diverge as soon as the optimizer touches
        # one of the two.
        self.assertIs(
            captured.get("dspark_embedding_weight"), embed.embedding_weight
        )
        self.assertIs(captured.get("dspark_head_weight"), head.weight)


class _RecordingGPT:
    """Stub ``self`` that records each desc's shared key, if it has one."""

    def __init__(self, config):
        self.config = config

    def _model_name_prefix(self):
        return "model"

    def add_sequential_layer(self, layers, desc, name_prefix):
        layers.append(getattr(desc, "layer_name", None))


def _assembly_config(**overrides):
    config = SimpleNamespace(
        enable_mtp_magic_send=False,
        mtp_shared_last_layer=False,
        gpt_model_use_experimental_version=False,
        num_nextn_predict_layers=0,
        num_hidden_layers=2,
        multimax_modules=[],
        separate_mtp_headloss=False,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _assembly_spec(has_draft, mtp_lm_head=None):
    return SimpleNamespace(
        embedding=object(),
        head_empty_layers=[],
        mhc_expand=None,
        transformer_layers=[object(), object()],
        mhc_contract=None,
        output_block_attn_res=None,
        layer_norm=object(),
        mtp=None,
        mtp_lm_head=mtp_lm_head,
        mtp_loss=None,
        tail_empty_layers=[],
        lm_head=object(),
        dspark=[object()] if has_draft else None,
        dspark_loss=None,
    )


class TestDsparkSharedKeyPromotion(unittest.TestCase):
    """Key promotion matrix and the collision guard, without building layers.

    ``get_layer_desc_list`` is driven with a recording stub, so this is a
    CPU-only unit test of the desc order and the keys it assigns.
    """

    def _keys(self, config, spec, tie_word_embeddings):
        with (
            mock.patch.object(gm, "LayerDesc", lambda *a, **k: None),
            mock.patch.object(
                gm,
                "SharedLayerDesc",
                lambda *a, **k: SimpleNamespace(layer_name=a[0] if a else None),
            ),
        ):
            return gm.GPTModel.get_layer_desc_list(
                _RecordingGPT(config), spec, tie_word_embeddings
            )

    def test_no_draft_keeps_embedding_and_head_unshared(self):
        keys = self._keys(
            _assembly_config(), _assembly_spec(has_draft=False), False
        )
        self.assertEqual([k for k in keys if k], [])

    def test_draft_promotes_embedding_and_head_to_distinct_keys(self):
        keys = self._keys(
            _assembly_config(), _assembly_spec(has_draft=True), False
        )
        # Four keyed descs: the backbone embedding, the draft's embedding and
        # head pass-throughs, and the tail head desc that reuses the head key
        # instead of building a second head.
        self.assertEqual(
            [k for k in keys if k],
            ["embed", "embed", "dspark_lm_head", "dspark_lm_head"],
        )
        # The draft's two pass-throughs precede the draft, and the tail head
        # desc is the last keyed slot.
        self.assertEqual(
            [i for i, k in enumerate(keys) if k == "embed"][-1] < len(keys) - 1,
            True,
        )
        self.assertEqual(keys[-1], "dspark_lm_head")

    def test_tie_words_embeddings_with_draft_is_rejected(self):
        with self.assertRaisesRegex(NotImplementedError, "cannot share"):
            self._keys(_assembly_config(), _assembly_spec(has_draft=True), True)

    def test_mtp_lm_head_with_draft_is_rejected(self):
        # An MTP head claims "embed" for the LM head position, colliding with
        # the embedding key the draft already needs.
        with self.assertRaisesRegex(NotImplementedError, "cannot share"):
            self._keys(
                _assembly_config(),
                _assembly_spec(has_draft=True, mtp_lm_head=object()),
                False,
            )


if __name__ == "__main__":
    unittest.main()
