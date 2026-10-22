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
# Scope: multi-tower modular AOA generation (aoa_multi_tower) -- the per-tower context, the
# single walk that dispatches each live unit to its owner, the ownership
# errors, the two directions' independence, and the copy path a fanned-out
# checkpoint root needs. Lightweight fakes only; no pipeline / fleet init.
import os
import sys

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

import unittest
from unittest import mock

import numpy as np
import paddle

from paddlefleet.models.gpt import aoa_multi_tower as mt


class _Leaf(paddle.nn.Layer):
    """One-parameter unit, so the base recursion does the name resolution."""

    def __init__(self):
        super().__init__()
        self.weight = self.create_parameter(shape=[2])


class _Cfg:
    """Stand-in for a region's provider view."""

    def __init__(self, **kwargs):
        self.aoa_checkpoint_name_prefix = kwargs.get(
            "aoa_checkpoint_name_prefix", "hf"
        )
        self.aoa_checkpoint_name_mapping = kwargs.get(
            "aoa_checkpoint_name_mapping", {}
        )
        self.model_type = kwargs.get("model_type", "")
        self.pipeline_model_parallel_size = kwargs.get(
            "pipeline_model_parallel_size", 1
        )


class _FakeWholeModel:
    """Duck-types what the multi-tower entries read off a live model."""

    def __init__(self, sub_layers, pp_mapping, config=None):
        self._sub_layers = sub_layers
        self._pp_to_single_mapping = pp_mapping
        self.config = config if config is not None else _Cfg()

    def _model_name_prefix(self):  # pragma: no cover - towers override it
        return "model"

    def _set_pipeline_name_mapping(self):  # pragma: no cover - never hit
        raise AssertionError("mapping already set")


def _two_region_model():
    """``enc`` rooted at ``encoder``, ``dec`` rooted at ``model``."""
    return _FakeWholeModel(
        {"enc": _Leaf(), "dec": _Leaf()},
        {
            "enc.weight": "encoder.a.weight",
            "dec.weight": "model.b.weight",
        },
    )


def _prefix_owner(*names):
    return lambda structured_prefix: structured_prefix.split(".")[0] in names


def _two_towers():
    return [
        mt.AOATower(
            model_name_prefix="encoder",
            config=_Cfg(aoa_checkpoint_name_prefix="hfenc"),
            owns_unit=_prefix_owner("enc"),
        ),
        mt.AOATower(
            model_name_prefix="model",
            config=_Cfg(aoa_checkpoint_name_prefix="hfdec"),
            owns_unit=_prefix_owner("dec"),
        ),
    ]


_ENC = "hfenc.a.weight -> encoder.a.weight"
_DEC = "hfdec.b.weight -> model.b.weight"


class TestPerTowerContext(unittest.TestCase):
    """Each unit is generated under its own tower's roots."""

    def test_each_region_resolves_under_its_own_roots(self):
        model, towers = _two_region_model(), _two_towers()
        self.assertEqual(
            mt.gen_multi_tower_aoa(model, towers)["aoa_statements"],
            [_ENC, _DEC],
        )

    def test_inverse_is_the_reversed_walk_with_swapped_endpoints(self):
        model, towers = _two_region_model(), _two_towers()
        self.assertEqual(
            mt.gen_multi_tower_inv_aoa(model, towers)["aoa_statements"],
            [
                "model.b.weight -> hfdec.b.weight",
                "encoder.a.weight -> hfenc.a.weight",
            ],
        )

    def test_tower_overrides_win_over_the_config_declaration(self):
        """A tower's roots override what its config declares.

        MTP warm-start needs this: the region's config is the decoder view,
        whose declaration names the region's own layer, while the tower must
        read a different one.
        """
        model = _two_region_model()
        towers = _two_towers()
        towers[0] = mt.AOATower(
            model_name_prefix="encoder",
            config=_Cfg(aoa_checkpoint_name_prefix="ignored"),
            owns_unit=_prefix_owner("enc"),
            checkpoint_name_prefix="override",
            checkpoint_name_mapping={},
        )
        self.assertEqual(
            mt.gen_multi_tower_aoa(model, towers)["aoa_statements"],
            ["override.a.weight -> encoder.a.weight", _DEC],
        )

    def test_mapping_is_validated_against_the_tower_root(self):
        """A mapping key outside the tower's model root is an error."""
        model = _two_region_model()
        towers = _two_towers()
        towers[0] = mt.AOATower(
            model_name_prefix="encoder",
            config=_Cfg(),
            owns_unit=_prefix_owner("enc"),
            checkpoint_name_mapping={"model.a.weight": "hf.a.weight"},
        )
        with self.assertRaises(ValueError):
            mt.gen_multi_tower_aoa(model, towers)


class TestOwnership(unittest.TestCase):
    """Every live unit must belong to exactly one tower."""

    def test_unowned_unit_is_an_error(self):
        """Zero owners would silently drop that region's parameters."""
        model = _two_region_model()
        towers = _two_towers()[:1]
        with self.assertRaises(RuntimeError) as caught:
            mt.gen_multi_tower_aoa(model, towers)
        self.assertIn("dec.", str(caught.exception))

    def test_doubly_owned_unit_is_an_error(self):
        """Several owners would generate it twice under different roots."""
        model = _two_region_model()
        towers = _two_towers()
        towers[1] = mt.AOATower(
            model_name_prefix="model",
            config=_Cfg(),
            owns_unit=_prefix_owner("enc", "dec"),
        )
        with self.assertRaises(RuntimeError) as caught:
            mt.gen_multi_tower_aoa(model, towers)
        self.assertIn("enc.", str(caught.exception))

    def test_inverse_enforces_ownership_too(self):
        model = _two_region_model()
        with self.assertRaises(RuntimeError):
            mt.gen_multi_tower_inv_aoa(model, _two_towers()[:1])


class TestPipelineGather(unittest.TestCase):
    """The gather is whole-model, so it runs once off the model's config."""

    def test_globalize_runs_once_over_the_concatenation(self):
        model, towers = _two_region_model(), _two_towers()
        with mock.patch.object(
            mt, "globalize_statements", side_effect=lambda cfg, st: st
        ) as globalize:
            mt.gen_multi_tower_aoa(model, towers)
        globalize.assert_called_once()
        self.assertEqual(globalize.call_args.args[1], [_ENC, _DEC])

    def test_globalize_reads_the_model_config_not_a_tower_view(self):
        """A region's view need not carry the pipeline size.

        HyperBody's encoder view is built from an explicit field list, so
        taking the size from a tower would turn the "declared pipeline without
        a group" error into a silent rank-local config.
        """
        model = _two_region_model()
        model.config = _Cfg(pipeline_model_parallel_size=2)
        with self.assertRaises(RuntimeError):
            mt.gen_multi_tower_aoa(model, _two_towers())


def _fanout_towers(shared):
    """``dec`` reads the root ``enc`` is named after (MTP warm-start shape)."""
    return [
        mt.AOATower(
            model_name_prefix="encoder",
            config=_Cfg(),
            owns_unit=_prefix_owner("enc"),
            checkpoint_name_prefix="shared",
            shares_checkpoint_root=shared,
        ),
        mt.AOATower(
            model_name_prefix="model",
            config=_Cfg(),
            owns_unit=_prefix_owner("dec"),
            checkpoint_name_prefix="shared",
            shares_checkpoint_root=shared,
        ),
    ]


class TestFannedOutCopyPath(unittest.TestCase):
    """A plain rename out of a shared checkpoint root needs a copy path.

    ``dist.load_state_dict`` loads a plain rename straight into the destination
    keyed by the source name, so one checkpoint tensor feeding several
    destinations reaches only one of them unless it is routed through a temp.
    """

    def test_plain_rename_out_of_a_shared_root_is_rewritten(self):
        statements = mt.gen_multi_tower_aoa(
            _two_region_model(), _fanout_towers(shared=True)
        )["aoa_statements"]
        temp = f"encoder.a.weight{mt._SHARED_SOURCE_TEMP_SUFFIX}"
        self.assertEqual(
            statements[:2],
            [f"shared.a.weight^T -> {temp}", f"{temp}^T -> encoder.a.weight"],
        )

    def test_not_rewritten_when_the_root_is_not_declared_shared(self):
        """The default must stay the plain rename: the rewrite costs a copy."""
        statements = mt.gen_multi_tower_aoa(
            _two_region_model(), _fanout_towers(shared=False)
        )["aoa_statements"]
        self.assertEqual(
            statements,
            [
                "shared.a.weight -> encoder.a.weight",
                "shared.b.weight -> model.b.weight",
            ],
        )

    def test_statements_that_already_have_a_copy_path_are_untouched(self):
        """Transform / multi-source / attributed statements go through a temp.

        Rewriting them would be wrong, not just wasteful: appending a second
        transform changes what the statement computes.
        """
        towers = _fanout_towers(shared=True)
        already = [
            "shared.a^T -> model.a",
            "shared.a, shared.b -> model.c, fused_ffn",
            "shared.d -> model.d, dtype='float32'",
            "_ -> model.e",
        ]
        self.assertEqual(
            mt._copy_path_for_shared_checkpoint_roots(already, towers, set()),
            already,
        )

    def test_no_shared_tower_leaves_every_statement_alone(self):
        statements = ["shared.a -> model.a"]
        self.assertIs(
            mt._copy_path_for_shared_checkpoint_roots(
                statements, _fanout_towers(shared=False), set()
            ),
            statements,
        )

    def test_inverse_direction_has_no_copy_path_rewrite(self):
        """Export has one source per model tensor, so the hazard cannot arise."""
        statements = mt.gen_multi_tower_inv_aoa(
            _two_region_model(), _fanout_towers(shared=True)
        )["aoa_statements"]
        self.assertEqual(
            statements,
            [
                "model.b.weight -> shared.b.weight",
                "encoder.a.weight -> shared.a.weight",
            ],
        )

    def test_passthrough_end_of_a_shared_source_gets_a_self_copy(self):
        """The reader whose name matches the checkpoint emits no statement, so
        the rewrite synthesizes its copy path from the live model names."""
        suffix = mt._SHARED_SOURCE_TEMP_SUFFIX
        rewritten = mt._copy_path_for_shared_checkpoint_roots(
            ["shared.a.weight -> model.b.weight"],
            _fanout_towers(shared=True),
            {"shared.a.weight"},
        )
        self.assertEqual(
            rewritten,
            [
                f"shared.a.weight^T -> shared.a.weight{suffix}",
                f"shared.a.weight{suffix}^T -> shared.a.weight",
                f"shared.a.weight^T -> model.b.weight{suffix}",
                f"model.b.weight{suffix}^T -> model.b.weight",
            ],
        )

    def test_pure_fanout_source_without_a_passthrough_reader_is_left(self):
        """A source no parameter is named after has no passthrough end, so it
        gets the rename rewrite but no self-copy into a name nobody owns."""
        suffix = mt._SHARED_SOURCE_TEMP_SUFFIX
        rewritten = mt._copy_path_for_shared_checkpoint_roots(
            ["shared.a.weight -> model.b.weight"],
            _fanout_towers(shared=True),
            set(),
        )
        self.assertEqual(
            rewritten,
            [
                f"shared.a.weight^T -> model.b.weight{suffix}",
                f"model.b.weight{suffix}^T -> model.b.weight",
            ],
        )


class TestFannedOutLoadReachesBothEnds(unittest.TestCase):
    """The generated copy path must survive a real ``dist.load_state_dict``.

    ``dist.load_state_dict`` keys its load off the source name, so a shared
    checkpoint tensor read by a passthrough end (same name, no statement) and a
    rewritten end collide on that key -- the load's ``sorted`` order lets one
    win the slot and the other keeps an unread buffer. A string comparison
    cannot see this; the tensors have to be loaded and read back.
    """

    def setUp(self):
        self._device = paddle.get_device()
        paddle.set_device("cpu")
        self._env = {}
        for key, value in {
            "PADDLE_TRAINER_ID": "0",
            "PADDLE_TRAINERS_NUM": "1",
            "PADDLE_TRAINER_ENDPOINTS": "127.0.0.1:60001",
            "PADDLE_CURRENT_ENDPOINT": "127.0.0.1:60001",
        }.items():
            self._env[key] = os.environ.get(key)
            os.environ[key] = value

    def tearDown(self):
        paddle.set_device(self._device)
        for key, value in self._env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _load_two_ends(self, statements):
        import tempfile

        import paddle.distributed as dist
        from paddle.distributed.flex_checkpoint.dcp.sharded_weight import (
            build_sharded_state_dict,
        )

        want = np.arange(6, dtype="float32").reshape(2, 3)
        path = tempfile.mkdtemp()
        dist.save_state_dict(
            build_sharded_state_dict(
                {"shared.a.weight": paddle.to_tensor(want.copy())}
            ),
            path,
        )
        # 'encoder.a.weight' sorts before 'shared.a.weight', so the passthrough
        # end wins the shared slot unless its self-copy routes it through a temp.
        sharded = build_sharded_state_dict(
            {
                "shared.a.weight": paddle.full([2, 3], -7.0, dtype="float32"),
                "encoder.a.weight": paddle.full([2, 3], -7.0, dtype="float32"),
            }
        )
        dist.load_state_dict(
            sharded, path, aoa_config={"aoa_statements": statements}
        )
        got = {
            name: shard.local_tensor.numpy() for name, shard in sharded.items()
        }
        return want, got

    def test_self_copy_feeds_the_passthrough_and_the_rewritten_end(self):
        suffix = mt._SHARED_SOURCE_TEMP_SUFFIX
        want, got = self._load_two_ends(
            [
                f"shared.a.weight^T -> shared.a.weight{suffix}",
                f"shared.a.weight{suffix}^T -> shared.a.weight",
                f"shared.a.weight^T -> encoder.a.weight{suffix}",
                f"encoder.a.weight{suffix}^T -> encoder.a.weight",
            ]
        )
        np.testing.assert_array_equal(got["shared.a.weight"], want)
        np.testing.assert_array_equal(got["encoder.a.weight"], want)

    def test_without_the_self_copy_the_rewritten_end_misses_the_value(self):
        """Pins the hazard: the rename rewrite alone lets the two ends collide,
        so the rewritten end does not receive the checkpoint value."""
        suffix = mt._SHARED_SOURCE_TEMP_SUFFIX
        want, got = self._load_two_ends(
            [
                f"shared.a.weight^T -> encoder.a.weight{suffix}",
                f"encoder.a.weight{suffix}^T -> encoder.a.weight",
            ]
        )
        np.testing.assert_array_equal(got["shared.a.weight"], want)
        self.assertFalse(
            np.array_equal(got["encoder.a.weight"], want),
            "rewritten end should miss the value without a self-copy",
        )


if __name__ == "__main__":
    unittest.main()
