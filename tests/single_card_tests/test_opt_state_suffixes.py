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
"""Optimizer state name -> parameter name resolution follows
``optimizer_state_layout()`` instead of a hard-coded AdamW suffix table.

Regression: resuming an RMSprop run from a flex checkpoint failed in the bf16
master-weight write-back with ``cannot find embedding_0.w_0_square_avg_0``,
because ``GroupGetter`` only knew the AdamW suffixes.

Run:  python -m pytest tests/single_card_tests/test_opt_state_suffixes.py -v
"""

import unittest
from collections import OrderedDict
from types import SimpleNamespace
from unittest import mock

import paddle

from paddlefleet.trainer.utils import sharding_io, zero_cost_checkpoint as zcc
from paddlefleet.trainer.utils.reshard import (
    common,
    get_opt_state_suffixes,
    get_optimizer_state_layout,
    sharding_v2,
)

_ADAMW_SUFFIXES = (
    "_fp32_master_0_beta1_pow_acc_0",
    "_fp32_master_0_beta2_pow_acc_0",
    "_fp32_master_0_moment1_0",
    "_fp32_master_0_moment2_0",
    "_beta1_pow_acc_0",
    "_beta2_pow_acc_0",
    "_moment1_0",
    "_moment2_0",
)

_RMSPROP_LAYOUT = {
    "vector": ["square_avg_0", "momentum_buffer_0"],
    "scalar": ["step_count_0"],
    "master": "fp32_master_0",
}


class _LayoutOpt:
    """Inner optimizer that declares its state layout."""

    def __init__(self, layout):
        self._layout = layout

    def optimizer_state_layout(self):
        return dict(self._layout)


def _wrapped(inner, depth=2):
    """Mimic HybridParallelOptimizer -> sharding shell -> inner optimizer."""
    opt = inner
    for _ in range(depth):
        opt = SimpleNamespace(_inner_opt=opt)
    return opt


class _Sentinel(Exception):
    pass


class TestStateLayoutLookup(unittest.TestCase):
    def test_none_and_no_layout(self):
        self.assertIsNone(get_optimizer_state_layout(None))
        self.assertIsNone(get_optimizer_state_layout(_wrapped(object())))

    def test_walks_inner_opt_chain(self):
        layout = get_optimizer_state_layout(
            _wrapped(_LayoutOpt(_RMSPROP_LAYOUT))
        )
        self.assertEqual(layout, _RMSPROP_LAYOUT)

    def test_outermost_layout_wins(self):
        outer = _LayoutOpt({"vector": ["outer_0"]})
        outer._inner_opt = _LayoutOpt({"vector": ["inner_0"]})
        layout = get_optimizer_state_layout(outer)
        self.assertEqual(layout["vector"], ["outer_0"])

    def test_cycle_terminates(self):
        a = SimpleNamespace()
        b = SimpleNamespace(_inner_opt=a)
        a._inner_opt = b
        self.assertIsNone(get_optimizer_state_layout(a))

    def test_non_callable_attribute_is_skipped(self):
        opt = SimpleNamespace(
            optimizer_state_layout={"vector": ["x_0"]},
            _inner_opt=_LayoutOpt(_RMSPROP_LAYOUT),
        )
        self.assertEqual(get_optimizer_state_layout(opt), _RMSPROP_LAYOUT)


class TestStateSuffixes(unittest.TestCase):
    def test_default_is_builtin_adamw(self):
        self.assertEqual(set(get_opt_state_suffixes()), set(_ADAMW_SUFFIXES))
        self.assertEqual(
            set(get_opt_state_suffixes(_wrapped(object()))),
            set(_ADAMW_SUFFIXES),
        )

    def test_layout_names_added_both_forms(self):
        sfx = get_opt_state_suffixes(_wrapped(_LayoutOpt(_RMSPROP_LAYOUT)))
        for n in ("square_avg_0", "momentum_buffer_0", "step_count_0"):
            self.assertIn(f"_{n}", sfx)
            self.assertIn(f"_fp32_master_0_{n}", sfx)
        # Built-in names stay accepted (mixed optimizers keep AdamW groups).
        self.assertTrue(set(_ADAMW_SUFFIXES) <= set(sfx))
        self.assertEqual(len(sfx), len(set(sfx)))

    def test_longest_first(self):
        sfx = get_opt_state_suffixes(_LayoutOpt(_RMSPROP_LAYOUT))
        lens = [len(s) for s in sfx]
        self.assertEqual(lens, sorted(lens, reverse=True))

    def test_custom_master_and_missing_keys(self):
        sfx = get_opt_state_suffixes(
            _LayoutOpt({"vector": ["velocity_0"], "master": "master_x"})
        )
        self.assertIn("_master_x_velocity_0", sfx)
        self.assertIn("_master_x_moment1_0", sfx)
        self.assertNotIn("_fp32_master_0_moment1_0", sfx)
        # Empty master falls back to the default name.
        sfx = get_opt_state_suffixes(_LayoutOpt({"master": ""}))
        self.assertEqual(set(sfx), set(_ADAMW_SUFFIXES))

    def test_empty_layout_is_default(self):
        self.assertEqual(
            set(get_opt_state_suffixes(_LayoutOpt({}))), set(_ADAMW_SUFFIXES)
        )


def _tensor(name):
    t = paddle.zeros([2])
    t.name = name
    return t


class TestConvertOptName(unittest.TestCase):
    def test_default_adamw(self):
        got = common.convert_opt_name_to_tname(
            ["w_0"], ["w_0_fp32_master_0_moment1_0", "w_0_beta1_pow_acc_0"]
        )
        self.assertEqual(
            got,
            {
                "w_0_fp32_master_0_moment1_0": "w_0",
                "w_0_beta1_pow_acc_0": "w_0",
            },
        )

    def test_custom_state_needs_layout(self):
        with self.assertRaises(AssertionError):
            common.convert_opt_name_to_tname(["w_0"], ["w_0_square_avg_0"])
        sfx = get_opt_state_suffixes(_LayoutOpt(_RMSPROP_LAYOUT))
        got = common.convert_opt_name_to_tname(
            ["w_0"], ["w_0_square_avg_0", "w_0_step_count_0"], sfx
        )
        self.assertEqual(
            got, {"w_0_square_avg_0": "w_0", "w_0_step_count_0": "w_0"}
        )

    def test_falls_through_to_next_suffix(self):
        # Same case as TestGroupGetter.test_falls_through_to_next_suffix: the
        # longest suffix leaves "p" (not a tensor), the shorter one leaves the
        # real tensor "p_fp32_master_0" -- both resolvers must agree.
        sfx = get_opt_state_suffixes(_LayoutOpt(_RMSPROP_LAYOUT))
        got = common.convert_opt_name_to_tname(
            ["p_fp32_master_0", "q"],
            ["p_fp32_master_0_square_avg_0", "q_fp32_master_0_square_avg_0"],
            sfx,
        )
        self.assertEqual(
            got,
            {
                "p_fp32_master_0_square_avg_0": "p_fp32_master_0",
                "q_fp32_master_0_square_avg_0": "q",
            },
        )

    def test_no_known_tensor_keeps_longest_match(self):
        got = common.convert_opt_name_to_tname(
            ["other"], ["w_0_fp32_master_0_moment1_0"]
        )
        self.assertEqual(got["w_0_fp32_master_0_moment1_0"], "w_0")

    def test_pack_keys_uses_suffixes(self):
        sfx = get_opt_state_suffixes(_LayoutOpt(_RMSPROP_LAYOUT))
        nms = common.NodeModelState(group=None)
        nms.add_opts({"w_0_square_avg_0": _tensor("w_0_square_avg_0")})
        nms.pack_keys({"a.weight": "w_0"}, opt_state_suffixes=sfx)
        self.assertEqual(
            list(nms._opt_state.keys()),
            [("a.weight", "w_0", "w_0_square_avg_0")],
        )

    def test_pack_keys_default_rejects_custom_state(self):
        nms = common.NodeModelState(group=None)
        nms.add_opts({"w_0_square_avg_0": _tensor("w_0_square_avg_0")})
        with self.assertRaises(AssertionError):
            nms.pack_keys({"a.weight": "w_0"})


class TestParameterNameRemapper(unittest.TestCase):
    def _remapper(self):
        return sharding_io.ParameterNameRemapper(
            {"emb.weight": "embedding_0.w_0"},
            {"emb.weight": "embedding_1.w_0"},
            "ckpt",
        )

    def test_custom_state_renamed(self):
        name = "embedding_0.w_0_square_avg_0"
        out = self._remapper().remap_optimizer_state(
            {name: _tensor(name)},
            get_opt_state_suffixes(_LayoutOpt(_RMSPROP_LAYOUT)),
        )
        self.assertEqual(list(out.keys()), ["embedding_1.w_0_square_avg_0"])

    def test_default_suffixes_reject_custom_state(self):
        name = "embedding_0.w_0_square_avg_0"
        with self.assertRaises(AssertionError):
            self._remapper().remap_optimizer_state({name: _tensor(name)})


def _group_getter(optimizer=None):
    groups = {
        "embedding_0.w_0": SimpleNamespace(id=0),
        "linear_0.w_0": SimpleNamespace(id=1),
        "p_fp32_master_0": SimpleNamespace(id=1),
    }
    params = OrderedDict(
        (k, SimpleNamespace(name=v))
        for k, v in [
            ("emb.weight", "embedding_0.w_0"),
            ("fc.weight", "linear_0.w_0"),
            ("odd.weight", "p_fp32_master_0"),
        ]
    )
    model = SimpleNamespace(state_dict=lambda: params)
    with mock.patch.object(
        sharding_io,
        "get_param_sharding_group",
        side_effect=lambda v, hcg: groups[v.name],
    ):
        return sharding_io.GroupGetter(model, hcg=object(), optimizer=optimizer)


class TestGroupGetter(unittest.TestCase):
    def test_default_adamw_names(self):
        gg = _group_getter()
        self.assertEqual(gg.get_group("linear_0.w_0").id, 1)
        self.assertEqual(gg.get_group("fc.weight").id, 1)
        self.assertEqual(
            gg.get_group("embedding_0.w_0_fp32_master_0_moment2_0").id, 0
        )
        self.assertEqual(gg.get_group("linear_0.w_0_beta2_pow_acc_0").id, 1)
        self.assertEqual(gg.get_group_ids(), [0, 1])
        self.assertEqual(gg.get_group_by_id(1).id, 1)

    def test_custom_state_without_layout_raises(self):
        gg = _group_getter()
        with self.assertRaisesRegex(ValueError, "optimizer_state_layout"):
            gg.get_group("embedding_0.w_0_square_avg_0")

    def test_custom_state_with_layout(self):
        gg = _group_getter(_wrapped(_LayoutOpt(_RMSPROP_LAYOUT)))
        self.assertEqual(gg.get_group("embedding_0.w_0_square_avg_0").id, 0)
        self.assertEqual(gg.get_group("linear_0.w_0_step_count_0").id, 1)
        self.assertEqual(
            gg.get_group("linear_0.w_0_fp32_master_0_momentum_buffer_0").id, 1
        )
        # AdamW names still resolve under a custom layout.
        self.assertEqual(gg.get_group("linear_0.w_0_moment1_0").id, 1)

    def test_falls_through_to_next_suffix(self):
        # "_fp32_master_0_square_avg_0" strips to "p" (unknown); the shorter
        # "_square_avg_0" then yields the real parameter "p_fp32_master_0".
        gg = _group_getter(_LayoutOpt(_RMSPROP_LAYOUT))
        self.assertEqual(
            gg._get_parameter_name("p_fp32_master_0_square_avg_0"),
            "p_fp32_master_0",
        )

    def test_unknown_parameter_raises(self):
        gg = _group_getter(_LayoutOpt(_RMSPROP_LAYOUT))
        with self.assertRaisesRegex(ValueError, "missing_0.w_0_square_avg_0"):
            gg.get_group("missing_0.w_0_square_avg_0")

    def test_split_opt_state_regression(self):
        # The failing step of the RMSprop flex resume.
        gg = _group_getter(_wrapped(_LayoutOpt(_RMSPROP_LAYOUT)))
        opt_state = {
            "embedding_0.w_0_square_avg_0": _tensor("a"),
            "linear_0.w_0_square_avg_0": _tensor("b"),
            "master_weights": {"linear_0.w_0": _tensor("c")},
            "LR_Scheduler": {"last_epoch": 3},
        }
        res = common.split_opt_state(opt_state, gg)
        self.assertEqual(sorted(res), [0, 1])
        self.assertIn("embedding_0.w_0_square_avg_0", res[0])
        self.assertIn("linear_0.w_0_square_avg_0", res[1])
        self.assertIn("linear_0.w_0", res[1]["master_weights"])


class TestRealRMSpropLayout(unittest.TestCase):
    """The real RMSprop layout (not the hand-written fixture) resolves."""

    def test_centered_momentum_states(self):
        from paddlefleet.optimizers.rmsprop_optimizer import RMSpropOptimizer

        opt = RMSpropOptimizer(
            parameters=paddle.nn.Linear(2, 2).parameters(),
            weight_decay=0.0,
            centered=True,
            momentum=0.9,
        )
        names = ("square_avg_0", "grad_avg_0", "momentum_buffer_0")
        self.assertEqual(
            sorted(opt.optimizer_state_layout()["vector"]), sorted(names)
        )
        wrapped = _wrapped(opt)
        sfx = get_opt_state_suffixes(wrapped)
        for n in names:
            self.assertIn(f"_{n}", sfx)
            self.assertIn(f"_fp32_master_0_{n}", sfx)
        gg = _group_getter(wrapped)
        for n in names:
            self.assertEqual(gg.get_group(f"embedding_0.w_0_{n}").id, 0)
            self.assertEqual(
                gg.get_group(f"linear_0.w_0_fp32_master_0_{n}").id, 1
            )


class _FakeNodeModelState:
    """Captures the per-item callback that sharding_v2 hands to NodeModelState."""

    def __init__(self, nranks=2):
        self.group = SimpleNamespace(rank=0, nranks=nranks)
        self.func = None

    def _capture(self, func):
        self.func = func
        return self

    split_items = merge_items = _capture

    def _same(self, *args):
        return self

    flatten_key = reshard = drop_rank = even_distribute = collapse_key = _same


class TestShardingV2ScalarStates(unittest.TestCase):
    """Layout ``scalar`` states are kept whole by V2 reshard, like beta acc."""

    def setUp(self):
        self.opt = _wrapped(_LayoutOpt(_RMSPROP_LAYOUT))

    def test_scalar_state_checker(self):
        is_scalar = sharding_v2.scalar_state_checker(self.opt)
        self.assertTrue(is_scalar("w_0_step_count_0"))
        self.assertTrue(is_scalar("w_0_fp32_master_0_step_count_0"))
        self.assertTrue(is_scalar("w_0_beta1_pow_acc_0"))
        self.assertFalse(is_scalar("w_0_square_avg_0"))
        default = sharding_v2.scalar_state_checker(None)
        self.assertTrue(default("w_0_beta2_pow_acc_0"))
        self.assertFalse(default("w_0_step_count_0"))

    def _split(self, opt_name, value):
        nms = _FakeNodeModelState()
        # param starts at index 2 of an 8-wide buffer split over 2 ranks, so a
        # vector state straddles both ranks.
        infos = {"w_0": (2, 4, 8, True)}
        with (
            mock.patch.object(
                sharding_v2, "collect_split_info", return_value=infos
            ),
            mock.patch.object(
                sharding_v2, "DygraphShardingOptimizerV2", object
            ),
        ):
            sharding_v2.shard(nms, object(), self.opt)
        return nms.func(("a.weight", "w_0", opt_name), value)

    def test_shard_keeps_scalar_whole(self):
        step = paddle.to_tensor([7.0])
        parts = self._split("w_0_step_count_0", step)
        self.assertEqual([r for r, _ in parts], [0, 1])
        for _, t in parts:
            self.assertIs(t, step)

    def test_shard_pads_and_slices_vector(self):
        sq = paddle.to_tensor([1.0, 2.0, 3.0])
        parts = self._split("w_0_square_avg_0", sq)
        self.assertEqual([r for r, _ in parts], [0, 1])
        self.assertEqual(parts[0][1].tolist(), [1.0, 2.0])
        self.assertEqual(parts[1][1].tolist(), [3.0, 0.0])

    def _merge(self, opt_name, tensors):
        nms = _FakeNodeModelState()
        model = SimpleNamespace(
            state_dict=lambda: {"a.weight": paddle.zeros([3])}
        )
        sharding_v2.restore(nms, model, self.opt)
        return nms.func(("a.weight", "w_0", opt_name), list(enumerate(tensors)))

    def test_restore_keeps_scalar_whole(self):
        a, b = paddle.to_tensor([7.0]), paddle.to_tensor([7.0])
        self.assertIs(self._merge("w_0_step_count_0", [a, b]), a)

    def test_restore_merges_vector(self):
        out = self._merge(
            "w_0_square_avg_0",
            [paddle.to_tensor([1.0, 2.0]), paddle.to_tensor([3.0, 0.0])],
        )
        self.assertEqual(out.tolist(), [1.0, 2.0, 3.0])


class TestCallSitesPassOptimizer(unittest.TestCase):
    """Every checkpoint path builds GroupGetter with the live optimizer, so its
    state layout is honoured. GroupGetter is replaced by a sentinel that stops
    the method right after construction."""

    def setUp(self):
        self.opt = _wrapped(_LayoutOpt(_RMSPROP_LAYOUT))
        self.model = object()

    def _assert_built_with_opt(self, gg_mock):
        gg_mock.assert_called_once()
        args, kwargs = gg_mock.call_args
        self.assertIs(args[0], self.model)
        self.assertIs(kwargs["optimizer"], self.opt)

    def _sharding_io_self(self, **kw):
        return SimpleNamespace(model=self.model, optimizer=self.opt, **kw)

    def test_load_state_dict_with_reshard(self):
        fake = self._sharding_io_self()
        with (
            mock.patch.object(
                sharding_io, "GroupGetter", side_effect=_Sentinel
            ) as gg,
            self.assertRaises(_Sentinel),
        ):
            sharding_io.ShardingIO.load_state_dict_from_checkpoint_with_reshard(
                fake, "ckpt", "w", None
            )
        self._assert_built_with_opt(gg)

    def test_load_optimizer_state_with_reshard(self):
        fake = self._sharding_io_self(
            _load_distributed_strategy=lambda c: {
                "pp_degree": 1,
                "mp_degree": 1,
                "sharding_degree": 2,
            },
            _load_sharding_meta=lambda c: {},
            args=SimpleNamespace(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                sharding_parallel_size=2,
            ),
        )
        with (
            mock.patch.object(
                sharding_io.reshard_util,
                "get_sharding_strategy",
                return_value=sharding_io.SHARDING_STRATEGY_V1,
            ),
            mock.patch.object(
                sharding_io, "GroupGetter", side_effect=_Sentinel
            ) as gg,
            self.assertRaises(_Sentinel),
        ):
            sharding_io.ShardingIO.load_optimizer_state_with_reshard(
                fake, "ckpt", "opt", None
            )
        self._assert_built_with_opt(gg)

    def test_manipulate_state_dict_and_config(self):
        fake = self._sharding_io_self(
            args=SimpleNamespace(sharded_name_suffix=lambda: "")
        )
        with (
            mock.patch.object(
                sharding_io, "GroupGetter", side_effect=_Sentinel
            ) as gg,
            self.assertRaises(_Sentinel),
        ):
            sharding_io.ShardingIO.manipulate_state_dict_and_config(
                fake, self.model
            )
        self._assert_built_with_opt(gg)

    def test_remap_parameter_name_passes_suffixes(self):
        remapper = mock.Mock()
        fake = self._sharding_io_self(_get_remapper=lambda c: remapper)
        sharding_io.ShardingIO._remap_parameter_name(fake, "ckpt", {}, True)
        args, _ = remapper.remap_optimizer_state.call_args
        self.assertEqual(args[1], get_opt_state_suffixes(self.opt))
        self.assertIn("_square_avg_0", args[1])

    def test_zcc_manipulate_state_dict(self):
        cb = zcc.ZeroCostCheckpointCallbackFcBased
        with (
            mock.patch.object(zcc, "GroupGetter", side_effect=_Sentinel) as gg,
            self.assertRaises(_Sentinel),
        ):
            cb._manipulate_state_dict_and_config(
                SimpleNamespace(), self.model, self.opt
            )
        self._assert_built_with_opt(gg)


if __name__ == "__main__":
    unittest.main()
