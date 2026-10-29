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
# Scope: HyperBody's multi-tower AOA declaration -- the region a live unit is
# placed in, the towers built from those regions, the extra tower an MTP
# warm-start needs, and the branches the modular path refuses. What the generic
# multi-tower walk then does with the towers belongs to its own test.
#
# Building the real layer needs a topology and a PipelineLayer, so the unit is
# allocated without ``__init__`` and carries only the sequential-layer list and
# the two provider views the methods under test read.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import unittest
from types import SimpleNamespace
from unittest import mock

import paddle
from paddle.distributed.fleet.meta_parallel import SharedLayerDesc

from paddlefleet.transformers.hyperbody import modeling as hb

# The regions ``get_layer_desc_list`` names the pipeline layers with, trimmed
# to one layer per region: encoder front end / trunk / bridge, then the decoder
# embedding, trunk, final norm and head.
_REGIONS = [
    "encoder",
    "encoder.layers.0",
    "encoder.bridge",
    "model",
    "model.layers.0",
    "model.lm_head",
]

# One entry outside any layer, one layer-scoped, one also expert-scoped.
_DECODER_MAPPING = {
    "model.embedding.embed_tokens.weight": "model.embed_tokens.weight",
    "model.layers.$LAYER_ID.mlp.gate.weight": (
        "model.layers.$LAYER_ID.block_sparse_moe.gate.weight"
    ),
    "model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.down_proj.weight": (
        "model.layers.$LAYER_ID.block_sparse_moe.experts.$EXPERT_ID.w2.weight"
    ),
}


def _encoder_view(**fields):
    fields.setdefault("aoa_checkpoint_name_mapping", {"encoder.a.weight": "x"})
    return SimpleNamespace(**fields)


def _decoder_view(**fields):
    fields.setdefault("experimental_attention_variant", "dsv4_hybrid")
    fields.setdefault("aoa_checkpoint_name_mapping", _DECODER_MAPPING)
    fields.setdefault("num_hidden_layers", 2)
    fields.setdefault("num_nextn_predict_layers", 0)
    return SimpleNamespace(**fields)


def _model(*, regions=None, config=None, encoder_view=-1, shared=None):
    """A stand-in carrying only the sequential layers and the two views.

    Allocated without running ``__init__``: the real one builds a whole
    ``PipelineLayer``, and ``PretrainedModel``'s post-init hook then
    synchronizes shared weights over a communication group this test has no use
    for. ``Layer.__init__`` still runs, because the attribute assignments go
    through ``Layer.__setattr__``.
    """
    model = hb.HyperBodyUnifiedModel.__new__(hb.HyperBodyUnifiedModel)
    paddle.nn.Layer.__init__(model)
    model.config = _decoder_view() if config is None else config
    model._encoder_view = (
        _encoder_view() if encoder_view == -1 else encoder_view
    )
    model._sequential_layers = [
        {"layer": paddle.nn.Layer, "name_prefix": prefix}
        for prefix in (_REGIONS if regions is None else regions)
    ]
    if shared is not None:
        key, prefix = shared
        desc = SharedLayerDesc(key, paddle.nn.Layer)
        model._sequential_layers.append({"layer": desc, "name_prefix": prefix})
    return model


def _owner(towers, structured_prefix):
    """The single tower that claims a unit, as the generic walk requires."""
    owners = [t for t in towers if t.owns_unit(structured_prefix)]
    assert len(owners) == 1, (structured_prefix, len(owners))
    return owners[0]


class TestUnitRegion(unittest.TestCase):
    """A structured prefix is a pipeline slot; the region is its name_prefix."""

    def test_numeric_slot_resolves_through_the_name_prefixes(self):
        model = _model()
        self.assertEqual(model._aoa_unit_region("2."), "encoder.bridge")
        self.assertEqual(model._aoa_unit_region("4."), "model.layers.0")

    def test_shared_layer_resolves_through_its_layer_desc(self):
        """``mtp_shared_last_layer`` registers the pivot only under this path."""
        model = _model(shared=("mtp_reuse_transformer", "model.layers.1"))
        self.assertEqual(
            model._aoa_unit_region("shared_layers.mtp_reuse_transformer."),
            "model.layers.1",
        )

    def test_unknown_shape_is_an_error(self):
        """A third shape means the pipeline grouped the layers differently."""
        with self.assertRaises(RuntimeError):
            _model()._aoa_unit_region("0.0.")


class TestTowers(unittest.TestCase):
    """One tower per region root, each carrying that region's view."""

    def test_two_towers_rooted_at_the_two_regions(self):
        towers = _model()._aoa_towers(fan_out_mtp=True)
        self.assertEqual(
            [t.model_name_prefix for t in towers], ["encoder", "model"]
        )

    def test_each_unit_is_claimed_by_exactly_one_tower(self):
        model = _model()
        towers = model._aoa_towers(fan_out_mtp=True)
        for slot, region in enumerate(_REGIONS):
            expected = "encoder" if region.startswith("encoder") else "model"
            self.assertEqual(
                _owner(towers, f"{slot}.").model_name_prefix, expected
            )


# Two MTP depths behind two decoder layers; the archive has only the first.
_MTP_REGIONS = [
    "encoder",
    "model",
    "model.layers.0",
    "model.layers.1",
    "model.layers.2",
    "model.layers.3",
]


def _warm_start_model(source=1, depth=2, mapping=_DECODER_MAPPING):
    return _model(
        regions=_MTP_REGIONS,
        config=_decoder_view(
            num_hidden_layers=2,
            num_nextn_predict_layers=depth,
            source_mtp_layers=source,
            aoa_checkpoint_name_mapping=mapping,
        ),
    )


class TestMTPWarmStart(unittest.TestCase):
    """A depth the archive lacks reads the last one it has, as its own tower."""

    def test_no_fan_out_without_source_mtp_layers(self):
        self.assertEqual(_model()._aoa_mtp_fan_out(), {})

    def test_no_fan_out_when_the_archive_has_every_depth(self):
        self.assertEqual(_warm_start_model(source=2)._aoa_mtp_fan_out(), {})

    def test_fanned_out_depth_reads_the_last_archived_one(self):
        self.assertEqual(
            _warm_start_model()._aoa_mtp_fan_out(),
            {"model.layers.3": "model.layers.2"},
        )

    def test_renumbered_layers_are_an_error_not_a_silent_miss(self):
        """The computed name must be one a pipeline layer actually carries."""
        model = _model(
            regions=["model", "model.layers.0"],
            config=_decoder_view(
                num_hidden_layers=2,
                num_nextn_predict_layers=2,
                source_mtp_layers=1,
            ),
        )
        with self.assertRaises(RuntimeError):
            model._aoa_mtp_fan_out()

    def test_extra_tower_reads_the_archived_layer_under_its_own_root(self):
        """Its own root, the archive's layer, and the layer-scoped mapping
        entries pinned to a different number on each side."""
        towers = _warm_start_model()._aoa_towers(fan_out_mtp=True)
        self.assertEqual(len(towers), 3)
        tower = towers[-1]
        self.assertEqual(tower.model_name_prefix, "model.layers.3")
        self.assertEqual(tower.checkpoint_name_prefix, "model.layers.2")
        self.assertTrue(tower.shares_checkpoint_root)
        self.assertEqual(
            tower.checkpoint_name_mapping,
            {
                "model.layers.3.mlp.gate.weight": (
                    "model.layers.2.block_sparse_moe.gate.weight"
                ),
                "model.layers.3.mlp.experts.$EXPERT_ID.down_proj.weight": (
                    "model.layers.2.block_sparse_moe.experts."
                    "$EXPERT_ID.w2.weight"
                ),
            },
        )
        # Every unit has exactly one owner, so the roots cannot overlap.
        self.assertEqual(
            _owner(towers, "5.").model_name_prefix, "model.layers.3"
        )
        self.assertEqual(_owner(towers, "4.").model_name_prefix, "model")

    def test_a_tower_without_a_mapping_is_not_an_error(self):
        """The entries are declared per archive; this one declares none."""
        model = _warm_start_model(mapping=None)
        tower = model._aoa_towers(fan_out_mtp=True)[-1]
        self.assertEqual(tower.checkpoint_name_mapping, {})

    def test_export_keeps_every_depth_under_its_own_name(self):
        """Redirecting on export would write ``model.layers.3`` over
        ``model.layers.2`` and leave the archive without the extra depth."""
        towers = _warm_start_model()._aoa_towers(fan_out_mtp=False)
        self.assertEqual(
            [t.model_name_prefix for t in towers], ["encoder", "model"]
        )
        for slot in ("4.", "5."):
            owner = _owner(towers, slot)
            self.assertEqual(owner.model_name_prefix, "model")
            self.assertIsNone(owner.checkpoint_name_prefix)


class TestRejectedBranches(unittest.TestCase):
    """Configurations whose statements only the hand-written generator states.

    All of them are checked inside the AOA entry, never in the constructor: a
    model built for the hand-written path still supports them.
    """

    def test_branches_without_a_checkpoint_source_are_rejected(self):
        for field, value in (
            ("indexer_init_from_scratch", True),
            ("topk_method", "quantile_balancing"),
            ("tie_word_embeddings", True),
        ):
            with self.subTest(field=field):
                model = _model(config=_decoder_view(**{field: value}))
                with self.assertRaises(NotImplementedError):
                    model._reject_unsupported_aoa_branches()

    def test_missing_decoder_mapping_is_rejected(self):
        """Without it every decoder name resolves through the identity
        fallback and names keys this archive does not have."""
        model = _model(config=_decoder_view(aoa_checkpoint_name_mapping=None))
        with self.assertRaises(RuntimeError):
            model._reject_unsupported_aoa_branches()

    def test_a_plain_decoder_needs_no_mapping(self):
        """Its layout matches what the components already produce."""
        model = _model(
            config=_decoder_view(
                experimental_attention_variant=None,
                aoa_checkpoint_name_mapping=None,
            )
        )
        model._reject_unsupported_aoa_branches()

    def test_missing_encoder_mapping_is_rejected(self):
        model = _model(
            encoder_view=_encoder_view(aoa_checkpoint_name_mapping=None)
        )
        with self.assertRaises(RuntimeError):
            model._reject_unsupported_aoa_branches()

    def test_missing_encoder_view_is_an_error(self):
        """The builder passes it; a model built without it has no encoder."""
        with self.assertRaises(RuntimeError):
            _model(encoder_view=None)._reject_unsupported_aoa_branches()


class TestWholeModelEntries(unittest.TestCase):
    """Guard first, then hand the towers to the generic multi-tower walk."""

    def test_both_directions_delegate_with_their_own_towers(self):
        """Only the load entry fans out MTP, and each tower carries the view of
        its own region -- the ``config`` argument the trainer passes is the
        decoder's, so it is ignored."""
        model = _warm_start_model()
        for entry, target, expected in (
            (
                "gen_aoa_statements",
                "gen_multi_tower_aoa",
                ["encoder", "model", "model.layers.3"],
            ),
            (
                "gen_inv_aoa_statements",
                "gen_multi_tower_inv_aoa",
                ["encoder", "model"],
            ),
        ):
            with self.subTest(entry=entry):
                with mock.patch.object(hb, target) as generate:
                    getattr(model, entry)(SimpleNamespace(bogus=True))
                generate.assert_called_once()
                walked, towers = generate.call_args.args
                self.assertIs(walked, model)
                self.assertEqual(
                    [t.model_name_prefix for t in towers], expected
                )
                self.assertIs(towers[0].config, model._encoder_view)
                self.assertIs(towers[1].config, model.config)

    def test_the_guard_runs_before_any_generation(self):
        model = _model(config=_decoder_view(tie_word_embeddings=True))
        for entry, target in (
            ("gen_aoa_statements", "gen_multi_tower_aoa"),
            ("gen_inv_aoa_statements", "gen_multi_tower_inv_aoa"),
        ):
            with (
                mock.patch.object(hb, target) as generate,
                self.assertRaises(NotImplementedError),
            ):
                getattr(model, entry)()
            generate.assert_not_called()


_DECODER_GEOMETRY = {
    "vocab_size": 128,
    "hidden_size": 64,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "intermediate_size": 128,
    "n_routed_experts": 4,
    "moe_intermediate_size": 32,
    "num_experts_per_tok": 2,
    "n_shared_experts": 1,
}
# The encoder is the same geometry, two layers deep, with its own query length.
_ENCODER_GEOMETRY = {
    **_DECODER_GEOMETRY,
    "num_hidden_layers": 2,
    "first_k_dense_replace": 1,
    "hyperencoder_query_length": 16,
}


class TestConfigDeclarations(unittest.TestCase):
    """The fields the AOA entries read are declared on ``HyperBodyConfig`` and
    reach the view each region's tower reads them from."""

    def test_declarations_reach_the_view_of_their_own_region(self):
        mapping = {"encoder.a.weight": "model.encoder.a.weight"}
        cfg = hb.HyperBodyConfig(
            decoder_config=dict(_DECODER_GEOMETRY),
            encoder_config=dict(_ENCODER_GEOMETRY),
            source_mtp_layers=1,
            encoder_aoa_checkpoint_name_prefix="model.encoder",
            encoder_aoa_checkpoint_name_mapping=mapping,
        )
        self.assertEqual(hb._build_decoder_view(cfg).source_mtp_layers, 1)
        view = hb._build_encoder_view(cfg, decoder_hidden=64)
        self.assertEqual(view.aoa_checkpoint_name_prefix, "model.encoder")
        self.assertEqual(view.aoa_checkpoint_name_mapping, mapping)

    def test_nothing_is_declared_by_default(self):
        cfg = hb.HyperBodyConfig(
            decoder_config=dict(_DECODER_GEOMETRY),
            encoder_config=dict(_ENCODER_GEOMETRY),
        )
        self.assertIsNone(hb._build_decoder_view(cfg).source_mtp_layers)
        view = hb._build_encoder_view(cfg, decoder_hidden=64)
        self.assertIsNone(view.aoa_checkpoint_name_prefix)
        self.assertIsNone(view.aoa_checkpoint_name_mapping)


if __name__ == "__main__":
    unittest.main()
