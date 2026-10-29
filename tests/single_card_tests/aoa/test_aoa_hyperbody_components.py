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
#
# Scope: HyperBody's own AOA component overrides -- the minimal subclasses
# carrying the few places where its archive diverges from what the shared
# components produce, plus the installer that puts them into the layer spec.
# Building the real layers needs a live config plus TP / PP process groups, so
# each unit is a subclass that keeps the class (hence the overrides and their
# ``super()`` chain) and only the tensor / child layout that decides the names.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import unittest
from types import SimpleNamespace

import paddle
from paddle.distributed.fleet.meta_parallel import LayerSpec
from paddle.distributed.flex_checkpoint.aoa.generation import AOAContext

from paddlefleet.transformer.hyper_connection import (
    HyperConnectionContractLayer,
    HyperConnectionModule,
)
from paddlefleet.transformer.moe.moe_layer import MoELayer
from paddlefleet.transformer.multi_token_prediction import (
    MultiTokenPredictionLayer,
)
from paddlefleet.transformers.hyperbody import modeling as hb

# An MTP block holds its transformer one segment deeper than the archive does.
_LAYER = "model.layers.3."
_DROP = "transformer_layer"
_MLP_HC = f"{_LAYER}mlp_hyper_connection."
_MTP_HC = f"{_LAYER}{_DROP}.self_attention_hyper_connection."
_CONTRACT = "model.mhc_contract."
_HC_HEAD = ("hc_head_fn", "hc_head_base", "hc_head_scale")

# The encoder region is rooted at ``encoder`` and reads the archive's inner
# backbone root; ``_CK`` names are what the identity fallback produces, and
# the decoder region shares its root with the archive so its gate names are
# identities. The multimodal bias and the bridge's tensors are the ones the
# archive keeps elsewhere, hence the model-declared mapping entries.
_ENC_ROOT = "model.encoder.decoder.model.model"
_GATE = "encoder.layers.1.mlp.gate."
_GATE_CK = f"{_ENC_ROOT}.layers.1.mlp.gate."
_MODALITY = "encoder.projector."
_MODALITY_CK = "model.encoder.projector."
_MM_BIAS = "multimodal_e_score_correction_bias"
_DEC_GATE = "model.layers.1.mlp.gate."
_DEC_MM_BIAS_CK = f"model.layers.1.block_sparse_moe.{_MM_BIAS}"
_BRIDGE = "encoder.bridge."
_PROJ_CK = "model.projector."
_BRIDGE_MAPPING = {
    f"{_BRIDGE}final_norm.weight": f"{_ENC_ROOT}.norm.weight",
    f"{_BRIDGE}out_projector.weight": f"{_PROJ_CK}weight",
    f"{_BRIDGE}out_projector.bias": f"{_PROJ_CK}bias",
}


def _ctx(
    *,
    model_name_prefix="model",
    checkpoint_name_prefix="model",
    mapping=None,
    config=None,
):
    return AOAContext(
        config=config,
        checkpoint_name_prefix=checkpoint_name_prefix,
        pp_to_single_mapping={},
        checkpoint_name_mapping=mapping or {},
        model_name_prefix=model_name_prefix,
    )


_ENCODER_CTX = _ctx(
    model_name_prefix="encoder", checkpoint_name_prefix=_ENC_ROOT
)
_MODALITY_CTX = _ctx(
    model_name_prefix="encoder", checkpoint_name_prefix="model.encoder"
)


def _decoder_gate_ctx(**config_fields):
    return _ctx(
        mapping={f"{_DEC_GATE}{_MM_BIAS}": _DEC_MM_BIAS_CK},
        config=SimpleNamespace(**config_fields) if config_fields else None,
    )


def _bridge_ctx(**config_fields):
    return _ctx(
        model_name_prefix="encoder",
        checkpoint_name_prefix=_ENC_ROOT,
        mapping=_BRIDGE_MAPPING,
        config=SimpleNamespace(**config_fields) if config_fields else None,
    )


def _fwd(layer, prefix, ctx=None, *, drop=None):
    return layer.gen_aoa_statements(
        ctx if ctx is not None else _ctx(),
        structured_name_prefix=prefix,
        checkpoint_lookup_drop_segment=drop,
    )


def _inv(layer, prefix, ctx=None):
    return layer.gen_inv_aoa_statements(
        ctx if ctx is not None else _ctx(), structured_name_prefix=prefix
    )


class _ParamLeaf(paddle.nn.Layer):
    def __init__(self, dtype="float32"):
        super().__init__()
        self.weight = self.create_parameter(shape=[2, 2], dtype=dtype)


def _hyper_connection(dtype="float32"):
    class _StandIn(hb.HyperBodyHyperConnection):
        def __init__(self):
            paddle.nn.Layer.__init__(self)
            self.alpha_pre = self.create_parameter(shape=[1], dtype=dtype)
            self.alpha_post = self.create_parameter(shape=[1], dtype=dtype)
            self.alpha_res = self.create_parameter(shape=[1], dtype=dtype)
            self.bias = self.create_parameter(shape=[6], dtype=dtype)
            # The real module sets the default dtype around this construction.
            previous = paddle.get_default_dtype()
            try:
                paddle.set_default_dtype(dtype)
                self.mapping_proj = paddle.nn.Linear(4, 6, bias_attr=False)
            finally:
                paddle.set_default_dtype(previous)

    return _StandIn()


def _mhc_contract():
    class _StandIn(hb.HyperBodyMhcContract):
        def __init__(self):
            paddle.nn.Layer.__init__(self)
            for name in _HC_HEAD:
                setattr(self, name, self.create_parameter(shape=[2]))

    return _StandIn()


def _mtp_block():
    class _StandIn(hb.HyperBodyMTPLayer):
        def __init__(self):
            paddle.nn.Layer.__init__(self)
            self.config = SimpleNamespace(
                gpt_model_use_experimental_version=False
            )
            self.enorm = _ParamLeaf()
            self.hnorm = _ParamLeaf()
            self.eh_proj = None
            self.norm = _ParamLeaf()
            self.mtp_embed = None
            for name in _HC_HEAD:
                setattr(self, name, self.create_parameter(shape=[2]))

    return _StandIn()


def _gate_router(router_class=hb.HyperBodyGateRouter, *, multimodal_bias=False):
    class _StandIn(router_class):
        def __init__(self):
            paddle.nn.Layer.__init__(self)
            self.weight = self.create_parameter(shape=[2, 2])
            self.register_buffer(
                "e_score_correction_bias",
                paddle.zeros([2]),
                persistable=True,
            )
            if multimodal_bias:
                self.register_buffer(
                    _MM_BIAS, paddle.zeros([2]), persistable=True
                )

    return _StandIn()


def _projector(linear=True):
    class _StandIn(hb.HyperBodyMlpProjector):
        def __init__(self):
            paddle.nn.Layer.__init__(self)
            self.layers = paddle.nn.Linear(4, 6) if linear else _ParamLeaf()

    return _StandIn()


def _bridge():
    class _StandIn(hb.HyperBodyEncoderBridge):
        def __init__(self):
            paddle.nn.Layer.__init__(self)
            self.final_norm = _ParamLeaf()
            self.out_projector = paddle.nn.Linear(4, 6)

    return _StandIn()


class TestHyperConnection(unittest.TestCase):
    """The mHC subtree: cast on load, segment kept, projection not transposed."""

    def test_load_casts_every_owned_tensor(self):
        for dtype in ("float32", "bfloat16"):
            with self.subTest(dtype=dtype):
                self.assertEqual(
                    sorted(_fwd(_hyper_connection(dtype), _MLP_HC)),
                    sorted(
                        f"{_MLP_HC}{name} -> {_MLP_HC}{name}, dtype='{dtype}'"
                        for name in (
                            "alpha_pre",
                            "alpha_post",
                            "alpha_res",
                            "bias",
                            "mapping_proj.weight",
                        )
                    ),
                )

    def test_mapping_proj_is_not_transposed(self):
        """The Linear helper transposes unconditionally; this leaf must not."""
        self.assertFalse(
            any("^T" in s for s in _fwd(_hyper_connection(), _MLP_HC))
        )

    def test_drop_segment_is_not_applied(self):
        """The archive keeps ``transformer_layer`` for these two modules while
        the MTP block drops it for the rest, so the base would miss the key."""
        statements = _fwd(_hyper_connection(), _MTP_HC, drop=_DROP)
        base = HyperConnectionModule.gen_aoa_statements(
            _hyper_connection(),
            _ctx(),
            structured_name_prefix=_MTP_HC,
            checkpoint_lookup_drop_segment=_DROP,
        )
        self.assertTrue(statements and base)
        for statement in statements:
            self.assertIn(f".{_DROP}.", statement.split(" -> ")[0])
        for statement in base:
            self.assertNotIn(f".{_DROP}.", statement.split(" -> ")[0])

    def test_export_emits_nothing_for_identical_names(self):
        self.assertEqual(_inv(_hyper_connection(), _MLP_HC), [])


class TestMhcContract(unittest.TestCase):
    """The contract layer's mHC head: cast on load, nothing on export."""

    def test_load_casts_the_three_head_parameters(self):
        """The names match, so the casts are additive, not a replacement."""
        self.assertEqual(
            sorted(_fwd(_mhc_contract(), _CONTRACT)),
            sorted(
                f"{_CONTRACT}{name} -> {_CONTRACT}{name}, dtype='float32'"
                for name in _HC_HEAD
            ),
        )
        self.assertEqual(
            HyperConnectionContractLayer.gen_aoa_statements(
                _mhc_contract(), _ctx(), structured_name_prefix=_CONTRACT
            ),
            [],
        )

    def test_export_emits_nothing(self):
        self.assertEqual(_inv(_mhc_contract(), _CONTRACT), [])


class TestMTPBlock(unittest.TestCase):
    """The block's own mHC head, on top of what the base override emits."""

    def test_load_adds_the_casts_to_the_base_statements(self):
        block = _mtp_block()
        base = MultiTokenPredictionLayer.gen_aoa_statements(
            block, _ctx(), structured_name_prefix=_LAYER
        )
        statements = _fwd(block, _LAYER)
        self.assertEqual(statements[: len(base)], base)
        self.assertEqual(
            statements[len(base) :],
            [
                f"{_LAYER}{name} -> {_LAYER}{name}, dtype='float32'"
                for name in _HC_HEAD
            ],
        )

    def test_export_is_unchanged_from_the_base(self):
        block = _mtp_block()
        self.assertEqual(
            _inv(block, _LAYER),
            MultiTokenPredictionLayer.gen_inv_aoa_statements(
                block, _ctx(), structured_name_prefix=_LAYER
            ),
        )


class TestGateRouter(unittest.TestCase):
    """Encoder gate casts; bias has two modes; the rest is identity."""

    def test_only_the_encoder_subclass_declares_an_archive_dtype(self):
        self.assertIsNone(hb.HyperBodyGateRouter._AOA_GATE_CHECKPOINT_DTYPE)
        self.assertEqual(
            hb.HyperBodyEncoderGateRouter._AOA_GATE_CHECKPOINT_DTYPE,
            "bfloat16",
        )

    def test_encoder_gate_weight_declares_the_dtype_pair(self):
        router = _gate_router(hb.HyperBodyEncoderGateRouter)
        self.assertIn(
            f"{_GATE_CK}weight -> {_GATE}weight, "
            f"src_dtype='bfloat16',dst_dtype='float32'",
            _fwd(router, _GATE, _ENCODER_CTX),
        )
        self.assertIn(
            f"{_GATE}weight -> {_GATE_CK}weight, "
            f"src_dtype='float32',dst_dtype='bfloat16'",
            _inv(router, _GATE, _ENCODER_CTX),
        )

    def test_decoder_gate_declares_no_dtype(self):
        """The decoder archive holds the gate in the model's dtype, so the
        identity rule applies: a declared pair would cast the export."""
        router = _gate_router()
        ctx = _decoder_gate_ctx()
        self.assertEqual(_fwd(router, _DEC_GATE, ctx), [])
        self.assertEqual(_inv(router, _DEC_GATE, ctx), [])

    def test_other_tensors_take_the_plain_rename(self):
        """The ``noaux_tc`` bias differs only by root: identity rule."""
        name = "e_score_correction_bias"
        self.assertIn(
            f"{_GATE_CK}{name} -> {_GATE}{name}",
            _fwd(_gate_router(), _GATE, _ENCODER_CTX),
        )
        self.assertIn(
            f"{_GATE}{name} -> {_GATE_CK}{name}",
            _inv(_gate_router(), _GATE, _ENCODER_CTX),
        )

    def test_multimodal_bias_load_modes(self):
        """Default reads it from the archive; under cold start the add
        primitive names no source and keeps the registered zeros. Export
        always states it, the archive name not being the model name."""
        export = [f"{_DEC_GATE}{_MM_BIAS} -> {_DEC_MM_BIAS_CK}"]
        for ctx, expected in (
            (
                _decoder_gate_ctx(),
                f"{_DEC_MM_BIAS_CK} -> {_DEC_GATE}{_MM_BIAS}",
            ),
            (
                _decoder_gate_ctx(multimodal_bias_cold_start=True),
                f"_ -> {_DEC_GATE}{_MM_BIAS}",
            ),
        ):
            with self.subTest(statement=expected):
                router = _gate_router(multimodal_bias=True)
                self.assertEqual(_fwd(router, _DEC_GATE, ctx), [expected])
                self.assertEqual(_inv(router, _DEC_GATE, ctx), export)

    def test_moe_layer_subclasses_only_swap_the_gate(self):
        """``MoELayer.gen_aoa_statements`` delegates to ``self.gate``."""
        for layer_class, router_class in (
            (hb.HyperBodyMoELayer, hb.HyperBodyGateRouter),
            (hb.HyperBodyEncoderMoELayer, hb.HyperBodyEncoderGateRouter),
        ):
            self.assertTrue(issubclass(layer_class, MoELayer))
            self.assertIs(layer_class._AOA_ROUTER_CLASS, router_class)
            for name in ("gen_aoa_statements", "gen_inv_aoa_statements"):
                self.assertNotIn(name, layer_class.__dict__)


class TestMlpProjector(unittest.TestCase):
    """The modality projector's ``layers`` leaf is transposed in the archive."""

    def test_weight_is_transposed_and_bias_is_not(self):
        self.assertEqual(
            _fwd(_projector(), _MODALITY, _MODALITY_CTX),
            [
                f"{_MODALITY_CK}layers.weight^T -> {_MODALITY}layers.weight",
                f"{_MODALITY_CK}layers.bias -> {_MODALITY}layers.bias",
            ],
        )
        self.assertEqual(
            _inv(_projector(), _MODALITY, _MODALITY_CTX),
            [
                f"{_MODALITY}layers.weight^T -> {_MODALITY_CK}layers.weight",
                f"{_MODALITY}layers.bias -> {_MODALITY_CK}layers.bias",
            ],
        )

    def test_non_linear_variant_falls_back_to_the_base(self):
        """``identity`` / ``mlp_gelu`` are not the transposed single Linear."""
        statements = _fwd(_projector(linear=False), _MODALITY, _MODALITY_CTX)
        self.assertFalse(any("^T" in s for s in statements))


class TestEncoderBridge(unittest.TestCase):
    """``out_projector`` is what stage one trains, so it can be cold-started."""

    def test_load_follows_the_cold_start_switch(self):
        """Stage one must keep its own init even when the archive has a copy."""
        norm = f"{_ENC_ROOT}.norm.weight -> {_BRIDGE}final_norm.weight"
        for ctx, projector in (
            (
                _bridge_ctx(),
                [
                    f"{_PROJ_CK}weight^T -> {_BRIDGE}out_projector.weight",
                    f"{_PROJ_CK}bias -> {_BRIDGE}out_projector.bias",
                ],
            ),
            (
                _bridge_ctx(projector_cold_start=True),
                [
                    f"_ -> {_BRIDGE}out_projector.weight",
                    f"_ -> {_BRIDGE}out_projector.bias",
                ],
            ),
        ):
            with self.subTest(statements=projector):
                self.assertEqual(
                    _fwd(_bridge(), _BRIDGE, ctx), [norm, *projector]
                )

    def test_export_states_the_projector_whatever_the_switch(self):
        """Left to the passthrough it would land under its fleet name."""
        expected = [
            f"{_BRIDGE}final_norm.weight -> {_ENC_ROOT}.norm.weight",
            f"{_BRIDGE}out_projector.weight^T -> {_PROJ_CK}weight",
            f"{_BRIDGE}out_projector.bias -> {_PROJ_CK}bias",
        ]
        for ctx in (_bridge_ctx(), _bridge_ctx(projector_cold_start=True)):
            self.assertEqual(_inv(_bridge(), _BRIDGE, ctx), expected)


def _layer_spec(**fields):
    return LayerSpec(layer=object, sublayers_spec=SimpleNamespace(**fields))


_MHC_FIELDS = ("self_attention_hyper_connection", "mlp_hyper_connection")


def _moe_layer_spec():
    return _layer_spec(
        mlp=LayerSpec(layer=MoELayer),
        **{f: LayerSpec(layer=HyperConnectionModule) for f in _MHC_FIELDS},
    )


class TestInstaller(unittest.TestCase):
    """Every substitution is keyed off the original class."""

    def _install(self, *, multimodal_router_bias=False):
        """One MoE layer per region plus an MTP block holding its own inner
        layer, which ``get_gpt_mtp_layers_spec`` builds itself."""
        decoder, encoder, inner = (_moe_layer_spec() for _ in range(3))
        mtp = LayerSpec(
            layer=MultiTokenPredictionLayer,
            sublayers_spec=SimpleNamespace(transformer_layer=inner),
        )
        contract = LayerSpec(layer=HyperConnectionContractLayer)
        hb._install_aoa_component_overrides(
            decoder_layer_specs=[decoder],
            mtp_layer_specs=[mtp],
            encoder_layer_specs=[encoder],
            mhc_contract_spec=contract,
            multimodal_router_bias=multimodal_router_bias,
        )
        self.assertIs(mtp.layer, hb.HyperBodyMTPLayer)
        self.assertIs(contract.layer, hb.HyperBodyMhcContract)
        return decoder, encoder, inner

    def test_mhc_specs_are_installed_in_decoder_and_mtp_inner(self):
        decoder, _, inner = self._install()
        for spec in (decoder, inner):
            for field in _MHC_FIELDS:
                self.assertIs(
                    getattr(spec.sublayers_spec, field).layer,
                    hb.HyperBodyHyperConnection,
                )

    def test_moe_router_swap_follows_the_region_and_the_switch(self):
        """Only the encoder archive holds the gate in another dtype, so that
        region is swapped unconditionally; the decoder's one divergence is the
        multimodal buffer, so there the swap follows the switch creating it."""
        for bias, decoder_class in (
            (False, MoELayer),
            (True, hb.HyperBodyMoELayer),
        ):
            with self.subTest(multimodal_router_bias=bias):
                decoder, encoder, inner = self._install(
                    multimodal_router_bias=bias
                )
                self.assertIs(
                    encoder.sublayers_spec.mlp.layer,
                    hb.HyperBodyEncoderMoELayer,
                )
                self.assertIs(decoder.sublayers_spec.mlp.layer, decoder_class)
                self.assertIs(inner.sublayers_spec.mlp.layer, decoder_class)

    def test_unrelated_classes_are_left_alone(self):
        """mHC off carries a bare class, a dense mlp is not ``MoELayer``, and
        ``mtp_load_weight_only`` selects a different block class."""
        specs = [_layer_spec(self_attention_hyper_connection=object)]
        encoder = [_layer_spec(mlp=LayerSpec(layer=object))]
        mtp = LayerSpec(
            layer=object, sublayers_spec=SimpleNamespace(transformer_layer=None)
        )
        hb._install_aoa_component_overrides(
            decoder_layer_specs=specs,
            mtp_layer_specs=[mtp],
            encoder_layer_specs=encoder,
            mhc_contract_spec=LayerSpec(layer=object),
        )
        self.assertIs(
            specs[0].sublayers_spec.self_attention_hyper_connection, object
        )
        self.assertIs(encoder[0].sublayers_spec.mlp.layer, object)
        self.assertIs(mtp.layer, object)


if __name__ == "__main__":
    unittest.main()
