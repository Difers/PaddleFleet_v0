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

"""Unit coverage for the fp8 weight hooks on ``HyperBodyUnifiedModel``.

Covers ``_iter_transformer_layers`` / ``fp8_quant_weight`` /
``clear_fp8_quant_weight`` / ``use_fp8``:

  * dispatch logic (non-VPP ``run_function`` path and VPP ``_model_chunks``
    path), exercised on a lightweight stand-in so no GPU / distributed init is
    needed;
  * an end-to-end check on a tiny real unified model: every encoder + decoder
    ``TransformerLayer`` is reached, non-transformer slots are skipped, and the
    real bf16 hooks run without error;
  * the same on tiny models with MTP layers (own weights and
    ``mtp_shared_last_layer``): each ``MultiTokenPredictionLayer`` is reached
    through its inner ``transformer_layer``, mirroring ``GPTModel``.
"""

import random
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import paddle

from paddlefleet.transformer.multi_token_prediction import (
    MultiTokenPredictionLayer,
)
from paddlefleet.transformer.transformer_layer import (
    HyperConnectionTransformerLayer,
    TransformerLayer,
)
from paddlefleet.transformers.hyperbody.modeling import HyperBodyUnifiedModel


class _FakeUnifiedModel:
    """Borrows the methods under test without building a PipelineLayer."""

    _iter_transformer_layers = HyperBodyUnifiedModel._iter_transformer_layers
    fp8_quant_weight = HyperBodyUnifiedModel.fp8_quant_weight
    clear_fp8_quant_weight = HyperBodyUnifiedModel.clear_fp8_quant_weight
    use_fp8 = HyperBodyUnifiedModel.use_fp8

    def __init__(self, run_function=None, model_chunks=None, vpp=1):
        self.run_function = run_function or []
        self._num_virtual_pipeline_stages = vpp
        if model_chunks is not None:
            self._model_chunks = model_chunks


def _tl(use_fp8=False, cls=TransformerLayer):
    layer = MagicMock(spec=cls)
    layer.use_fp8.return_value = use_fp8
    return layer


def _mtp(use_fp8=False):
    mtp = MagicMock(spec=MultiTokenPredictionLayer)
    mtp.transformer_layer = _tl(use_fp8)
    return mtp


def _other():
    # Stand-in for frontend / bridge / embedding / norm / lm_head slots. Not a
    # TransformerLayer, but still exposes fp8 hooks so a wrong dispatch shows up.
    return MagicMock()


class TestIterTransformerLayers(unittest.TestCase):
    def test_non_vpp_filters_and_keeps_order(self):
        enc0, enc1, dec0, dec1 = _tl(), _tl(), _tl(), _tl()
        model = _FakeUnifiedModel(
            run_function=[
                _other(),  # encoder frontend
                enc0,
                enc1,
                _other(),  # bridge
                _other(),  # decoder embedding
                dec0,
                dec1,
                _other(),  # final norm
                _other(),  # lm_head
            ]
        )
        self.assertEqual(
            list(model._iter_transformer_layers()), [enc0, enc1, dec0, dec1]
        )

    def test_hyper_connection_layer_is_included(self):
        mhc = _tl(cls=HyperConnectionTransformerLayer)
        model = _FakeUnifiedModel(run_function=[_other(), mhc, _other()])
        self.assertEqual(list(model._iter_transformer_layers()), [mhc])

    def test_vpp_iterates_all_chunks(self):
        a, b, c = _tl(), _tl(), _tl()
        model = _FakeUnifiedModel(
            run_function=[_tl()],  # must be ignored on the VPP path
            model_chunks=[[_other(), a], [], [b, _other(), c]],
            vpp=2,
        )
        self.assertEqual(list(model._iter_transformer_layers()), [a, b, c])

    def test_missing_vpp_attr_falls_back_to_run_function(self):
        layer = _tl()
        model = _FakeUnifiedModel(run_function=[layer])
        del model._num_virtual_pipeline_stages
        self.assertEqual(list(model._iter_transformer_layers()), [layer])

    def test_empty_model(self):
        self.assertEqual(
            list(_FakeUnifiedModel()._iter_transformer_layers()), []
        )

    def test_mtp_yields_inner_transformer_layer(self):
        dec, mtp0, mtp1 = _tl(), _mtp(), _mtp()
        model = _FakeUnifiedModel(
            run_function=[_other(), dec, _other(), mtp0, mtp1, _other()]
        )
        self.assertEqual(
            list(model._iter_transformer_layers()),
            [dec, mtp0.transformer_layer, mtp1.transformer_layer],
        )

    def test_vpp_mtp_yields_inner_transformer_layer(self):
        dec, mtp = _tl(), _mtp()
        model = _FakeUnifiedModel(model_chunks=[[dec], [_other(), mtp]], vpp=2)
        self.assertEqual(
            list(model._iter_transformer_layers()),
            [dec, mtp.transformer_layer],
        )


class TestFp8QuantWeightDispatch(unittest.TestCase):
    def _models(self):
        """Yield (model, transformer_layers, other_layers) for both paths."""
        layers, others = [_tl(), _tl()], [_other(), _other()]
        yield (
            _FakeUnifiedModel(
                run_function=[others[0], layers[0], others[1], layers[1]]
            ),
            layers,
            others,
        )
        layers, others = [_tl(), _tl()], [_other(), _other()]
        yield (
            _FakeUnifiedModel(
                model_chunks=[[others[0], layers[0]], [layers[1], others[1]]],
                vpp=2,
            ),
            layers,
            others,
        )

    def test_default_args_forwarded(self):
        for model, layers, others in self._models():
            model.fp8_quant_weight()
            for layer in layers:
                layer.fp8_quant_weight.assert_called_once_with(
                    batch_mode=False, quant_transpose=True
                )
            for o in others:
                o.fp8_quant_weight.assert_not_called()

    def test_positional_args_from_callback_forwarded(self):
        # FP8QuantWeightCallback calls model.fp8_quant_weight(True, quant_transpose=False)
        for model, layers, _ in self._models():
            model.fp8_quant_weight(True, quant_transpose=False)
            for layer in layers:
                layer.fp8_quant_weight.assert_called_once_with(
                    batch_mode=True, quant_transpose=False
                )

    def test_clear_fp8_quant_weight(self):
        for model, layers, others in self._models():
            model.clear_fp8_quant_weight()
            for layer in layers:
                layer.clear_fp8_quant_weight.assert_called_once_with()
            for o in others:
                o.clear_fp8_quant_weight.assert_not_called()

    def test_quant_then_clear_called_once_each(self):
        for model, layers, _ in self._models():
            model.fp8_quant_weight()
            model.clear_fp8_quant_weight()
            for layer in layers:
                self.assertEqual(layer.fp8_quant_weight.call_count, 1)
                self.assertEqual(layer.clear_fp8_quant_weight.call_count, 1)

    def test_mtp_delegates_to_inner_transformer_layer(self):
        for vpp in (1, 2):
            dec, mtp = _tl(), _mtp()
            if vpp == 1:
                model = _FakeUnifiedModel(run_function=[dec, _other(), mtp])
            else:
                model = _FakeUnifiedModel(
                    model_chunks=[[dec], [_other(), mtp]], vpp=2
                )
            model.fp8_quant_weight(True, quant_transpose=False)
            model.clear_fp8_quant_weight()
            for layer in (dec, mtp.transformer_layer):
                layer.fp8_quant_weight.assert_called_once_with(
                    batch_mode=True, quant_transpose=False
                )
                layer.clear_fp8_quant_weight.assert_called_once_with()


class TestUseFp8(unittest.TestCase):
    def test_all_bf16_returns_false(self):
        model = _FakeUnifiedModel(run_function=[_other(), _tl(), _tl()])
        self.assertIs(model.use_fp8(), False)

    def test_any_fp8_returns_true_and_short_circuits(self):
        first, fp8, after = _tl(False), _tl(True), _tl(False)
        model = _FakeUnifiedModel(run_function=[first, fp8, after])
        self.assertIs(model.use_fp8(), True)
        first.use_fp8.assert_called_once()
        fp8.use_fp8.assert_called_once()
        after.use_fp8.assert_not_called()

    def test_vpp_returns_bool(self):
        # GPTModel.use_fp8 returns None on its VPP path; HyperBody must not.
        bf16 = _FakeUnifiedModel(model_chunks=[[_tl()], [_tl()]], vpp=2)
        self.assertIs(bf16.use_fp8(), False)
        fp8 = _FakeUnifiedModel(model_chunks=[[_tl()], [_tl(True)]], vpp=2)
        self.assertIs(fp8.use_fp8(), True)

    def test_no_transformer_layers_returns_false(self):
        model = _FakeUnifiedModel(run_function=[_other()])
        self.assertIs(model.use_fp8(), False)

    def test_fp8_mtp_body_returns_true(self):
        model = _FakeUnifiedModel(run_function=[_tl(False), _mtp(True)])
        self.assertIs(model.use_fp8(), True)


# --------------------------------------------------------------------------- #
# End-to-end on a tiny real unified model (needs GPU + single-rank fleet).    #
# --------------------------------------------------------------------------- #
_DECODER_GEOMETRY = {
    "vocab_size": 128,
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "intermediate_size": 128,
    "n_routed_experts": 4,
    "moe_intermediate_size": 32,
    "num_experts_per_tok": 2,
    "n_shared_experts": 1,
}
_ENCODER_GEOMETRY = {
    "vocab_size": 128,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "moe_intermediate_size": 32,
    "n_routed_experts": 4,
    "num_experts_per_tok": 2,
    "n_shared_experts": 1,
    "first_k_dense_replace": 1,
    "hyperencoder_query_length": 16,
}

_fleet_ready = False


def _ensure_fleet_init():
    global _fleet_ready
    if _fleet_ready:
        return
    from paddle.distributed import fleet

    import paddlefleet.parallel_state as ps

    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": 1,
        "pp_degree": 1,
        "sharding_degree": 1,
    }
    fleet.init(is_collective=True, strategy=strategy)
    ps.initialize_model_parallel(fleet.get_hybrid_communicate_group())
    _fleet_ready = True


@unittest.skipUnless(
    paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0,
    "requires GPU",
)
class TestHyperBodyUnifiedModelFp8Hooks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        random.seed(46)
        np.random.seed(46)
        paddle.seed(46)
        _ensure_fleet_init()
        from paddlefleet.transformers.hyperbody.configuration import (
            HyperBodyConfig,
        )
        from paddlefleet.transformers.hyperbody.modeling import (
            build_hyperbody_unified_model,
        )

        cfg = HyperBodyConfig(
            decoder_config={**_DECODER_GEOMETRY},
            encoder_config={**_ENCODER_GEOMETRY},
        )
        cls.model = build_hyperbody_unified_model(cfg)

    def test_non_vpp_path(self):
        self.assertIsInstance(self.model, HyperBodyUnifiedModel)
        self.assertEqual(self.model._num_virtual_pipeline_stages, 1)

    def test_iter_covers_encoder_and_decoder_trunks(self):
        prefixes = self.model.get_sequential_name_prefixes()
        found = list(self.model._iter_transformer_layers())
        expected_idx = [
            i
            for i, layer in enumerate(self.model.run_function)
            if isinstance(layer, TransformerLayer)
        ]
        self.assertEqual(
            found, [self.model.run_function[i] for i in expected_idx]
        )
        self.assertEqual(
            [prefixes[str(i)] for i in expected_idx],
            [
                "encoder.layers.0",
                "encoder.layers.1",
                "model.layers.0",
                "model.layers.1",
            ],
        )

    def test_dispatches_to_every_transformer_layer(self):
        layers = list(self.model._iter_transformer_layers())
        with (
            patch.object(
                TransformerLayer, "fp8_quant_weight", autospec=True
            ) as q,
            patch.object(
                TransformerLayer, "clear_fp8_quant_weight", autospec=True
            ) as c,
        ):
            self.model.fp8_quant_weight(True, quant_transpose=False)
            self.model.clear_fp8_quant_weight()
        self.assertEqual([call.args[0] for call in q.call_args_list], layers)
        for call in q.call_args_list:
            self.assertEqual(
                call.kwargs, {"batch_mode": True, "quant_transpose": False}
            )
        self.assertEqual([call.args[0] for call in c.call_args_list], layers)

    def test_bf16_model_reports_no_fp8(self):
        self.assertIs(self.model.use_fp8(), False)

    def test_real_hooks_run_on_bf16_model(self):
        # bf16 Linear / MoE fp8 hooks are no-ops; must not raise.
        self.model.fp8_quant_weight()
        self.model.fp8_quant_weight(True, quant_transpose=False)
        self.model.clear_fp8_quant_weight()


def _build_mtp_model(**extra):
    from paddlefleet.transformers.hyperbody.configuration import (
        HyperBodyConfig,
    )
    from paddlefleet.transformers.hyperbody.modeling import (
        build_hyperbody_unified_model,
    )

    cfg = HyperBodyConfig(
        decoder_config={**_DECODER_GEOMETRY},
        encoder_config={**_ENCODER_GEOMETRY},
        num_nextn_predict_layers=2,
        **extra,
    )
    return build_hyperbody_unified_model(cfg)


@unittest.skipUnless(
    paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0,
    "requires GPU",
)
class TestHyperBodyUnifiedModelFp8HooksWithMTP(unittest.TestCase):
    """MTP layers must be reached through their inner ``transformer_layer``."""

    @classmethod
    def setUpClass(cls):
        paddle.seed(46)
        _ensure_fleet_init()
        cls.models = {
            "own_weights": _build_mtp_model(),
            "shared_last_layer": _build_mtp_model(mtp_shared_last_layer=True),
        }

    def _mtp_layers(self, model):
        prefixes = model.get_sequential_name_prefixes()
        return [
            (prefixes[str(i)], layer)
            for i, layer in enumerate(model.run_function)
            if isinstance(layer, MultiTokenPredictionLayer)
        ]

    def test_iter_includes_mtp_transformer_layers(self):
        for name, model in self.models.items():
            with self.subTest(name):
                mtps = self._mtp_layers(model)
                self.assertEqual(
                    [p for p, _ in mtps], ["model.layers.2", "model.layers.3"]
                )
                found = list(model._iter_transformer_layers())
                trunk = [
                    layer
                    for layer in model.run_function
                    if isinstance(layer, TransformerLayer)
                ]
                self.assertEqual(len(trunk), 4)
                self.assertEqual(
                    found, trunk + [m.transformer_layer for _, m in mtps]
                )

    def test_shared_last_layer_aliases_backbone_weights(self):
        # Sanity check on the sharing mode: the MTP body is a distinct module
        # (so it carries its own per-Linear fp8 walk) whose parameters are the
        # last decoder layer's parameters.
        model = self.models["shared_last_layer"]
        last_dec = [
            layer
            for layer in model.run_function
            if isinstance(layer, TransformerLayer)
        ][-1]
        last_params = {id(p) for p in last_dec.parameters()}
        for _, mtp in self._mtp_layers(model):
            self.assertIsNot(mtp.transformer_layer, last_dec)
            self.assertTrue(
                all(
                    id(p) in last_params
                    for p in mtp.transformer_layer.parameters()
                )
            )

    def test_dispatches_to_mtp_transformer_layers(self):
        for name, model in self.models.items():
            with self.subTest(name):
                mtp_bodies = [
                    m.transformer_layer for _, m in self._mtp_layers(model)
                ]
                with (
                    patch.object(
                        TransformerLayer, "fp8_quant_weight", autospec=True
                    ) as q,
                    patch.object(
                        TransformerLayer,
                        "clear_fp8_quant_weight",
                        autospec=True,
                    ) as c,
                ):
                    model.fp8_quant_weight(True, quant_transpose=False)
                    model.clear_fp8_quant_weight()
                quantized = [call.args[0] for call in q.call_args_list]
                cleared = [call.args[0] for call in c.call_args_list]
                self.assertEqual(len(quantized), 6)
                self.assertEqual(quantized[-2:], mtp_bodies)
                self.assertEqual(cleared[-2:], mtp_bodies)
                for call in q.call_args_list:
                    self.assertEqual(
                        call.kwargs,
                        {"batch_mode": True, "quant_transpose": False},
                    )

    def test_real_hooks_run_on_bf16_model(self):
        for name, model in self.models.items():
            with self.subTest(name):
                self.assertIs(model.use_fp8(), False)
                model.fp8_quant_weight()
                model.clear_fp8_quant_weight()


if __name__ == "__main__":
    unittest.main()
