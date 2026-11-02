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
"""FP8 expert-weight callbacks under paddle-native FSDP.

``fully_shard`` owns parameter storage: outside the owning unit's forward a
``.weight`` has no storage, and the optimizer is never wrapped into a sharding
optimizer, so it has no ``clear_param_storage``. Contract:

* ``FP8QuantWeightCallback`` pre-quantizes per step by unsharding one
  ``TransformerLayer`` at a time (never the whole model, never
  ``clear_param_storage``), and offloads / reloads the master weights of
  fsdp-managed expert buffers.
* ``SonicMoELayoutSwitchCallback`` rejects fsdp, since its layout switch
  writes parameter storage outside forward.

Run with:
    python -m pytest tests/single_card_tests/fp8/test_fp8_fsdp_callbacks.py -v
"""

from __future__ import annotations

import contextlib
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import paddle

from paddlefleet.trainer import trainer_callback
from paddlefleet.trainer.trainer_callback import (
    FP8QuantWeightCallback,
    SonicMoELayoutSwitchCallback,
)
from paddlefleet.trainer.trainer_utils import ShardingOption
from paddlefleet.transformer.moe import fp8_utils, moe_expert
from paddlefleet.transformer.moe.moe_layer import MoELayer
from paddlefleet.transformer.transformer_layer import TransformerLayer

_GET_FSDP_CONTEXT = "paddle.distributed.fsdp._fsdp_context.get_fsdp_context"
_PADDLE_FP8 = "paddle.distributed.fsdp._fp8_utils"
_TRANSFORMER_LAYER = (
    "paddlefleet.transformer.transformer_layer.TransformerLayer"
)
_STAGE1_COLORS = [
    "moe_expert",
    "rms_linear",
    "memory_attn",
    "attn_out_project",
    "shared_expert",
]


class _ExplodingModel:
    """Whole-model weight access outside forward is a bug under FSDP."""

    def fp8_quant_weight(self, *args, **kwargs):
        raise AssertionError("fp8_quant_weight must not run under fsdp")

    def apply(self, fn):
        raise AssertionError("model.apply must not run under fsdp")


class _ExplodingOptimizer:
    def clear_param_storage(self, color):
        raise AssertionError("clear_param_storage must not run under fsdp")


def _args(sharding, using_sonic_moe=False, fp8="e4m3", offload_master=True):
    return SimpleNamespace(
        sharding=sharding,
        sharding_parallel_size=8,
        using_sonic_moe=using_sonic_moe,
        fp8=fp8,
        offload_fp8_expert_master_weight=offload_master,
    )


class _FakeParam:
    def __init__(self, name):
        self.name = name


class _FakeTransformerLayer:
    """Stands in for ``TransformerLayer``; checks it is quantized unsharded."""

    def __init__(self, ctx, prefix, use_w4a8=False):
        self._ctx = ctx
        self.expert_params = [
            _FakeParam(f"{prefix}.up_gate_proj"),
            _FakeParam(f"{prefix}.down_proj"),
        ]
        experts = SimpleNamespace(parameters=lambda: list(self.expert_params))
        self.mlp = SimpleNamespace(
            grouped_gemm_experts=experts, use_w4a8=use_w4a8
        )
        self.quant_calls = []

    def sublayers(self, include_self=False):
        return []

    def fp8_quant_weight(self, batch_mode, quant_transpose):
        assert self._ctx.unsharded == self.expert_params, (
            "fp8_quant_weight ran on params that are not unsharded"
        )
        self.quant_calls.append((batch_mode, quant_transpose))


class _FakeFSDPContext:
    def __init__(self):
        self.unsharded = None
        self.unshard_log = []
        self.reshard_log = []
        self.buffer_manager = SimpleNamespace(buffer_groups=[])

    @contextlib.contextmanager
    def unshard_params(self, params):
        assert self.unsharded is None, "unshard_params must not nest"
        self.unsharded = list(params)
        self.unshard_log.append([p.name for p in params])
        try:
            yield
        finally:
            self.reshard_log.append([p.name for p in params])
            self.unsharded = None

    def add_group(self, buffer_name, param_names, is_expert):
        self.buffer_manager.buffer_groups.append(
            SimpleNamespace(
                is_expert_param=is_expert,
                params_buffer=SimpleNamespace(
                    data_buffer=SimpleNamespace(name=buffer_name)
                ),
                params=[_FakeParam(n) for n in param_names],
            )
        )


class _FSDPModel(_ExplodingModel):
    def __init__(self, layers):
        self._layers = layers

    def sublayers(self, include_self=False):
        # A non-transformer sublayer must be skipped, not unsharded.
        return [self, object(), *self._layers]

    def parameters(self):
        return [p for layer in self._layers for p in layer.expert_params]


class _WrappedOptimizer:
    """Mimics MixPrecisionOptimizer: master weights live on ``_inner_opt``."""

    def __init__(self, master_weights):
        self._inner_opt = SimpleNamespace(_master_weights=master_weights)


class TestFP8QuantWeightCallbackUnderFSDP(unittest.TestCase):
    def setUp(self):
        self.callback = FP8QuantWeightCallback()
        self.ctx = _FakeFSDPContext()
        self.ctx.add_group(
            "fuse_params_expert_0", ["e.w1", "e.w2"], is_expert=True
        )
        self.ctx.add_group("fuse_params_dense_0", ["d.w"], is_expert=False)
        self.layers = [
            _FakeTransformerLayer(self.ctx, "layer0", use_w4a8=False),
            _FakeTransformerLayer(self.ctx, "layer1", use_w4a8=True),
        ]
        self.model = _FSDPModel(self.layers)
        self.master_weights = {
            "fuse_params_expert_0": "mw_expert_fused",
            "e.w1": "mw_expert_w1",
            "fuse_params_dense_0": "mw_dense_fused",
            "d.w": "mw_dense_w",
        }
        self.optimizer = _WrappedOptimizer(self.master_weights)
        self.offloaded, self.reloaded = [], []
        self._patch(_GET_FSDP_CONTEXT, lambda: self.ctx)
        self._patch(_TRANSFORMER_LAYER, _FakeTransformerLayer)
        self._patch_obj("offload", self.offloaded.append)
        self._patch_obj("reload", self.reloaded.append)

    def _patch(self, target, new):
        patcher = mock.patch(target, new)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _patch_obj(self, attr, new):
        patcher = mock.patch.object(trainer_callback, attr, new)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _step(self, args):
        kwargs = {"model": self.model, "optimizer": self.optimizer}
        self.callback.on_step_begin(args, None, None, **kwargs)
        self.callback.on_optimizer_begin(args, None, None, **kwargs)

    def test_quantizes_layer_by_layer_under_unshard(self):
        # _FSDPModel.fp8_quant_weight raises and _WrappedOptimizer has no
        # clear_param_storage, so the stage1 path would fail here.
        self._step(_args([ShardingOption.FSDP], offload_master=False))
        expected_params = [
            [p.name for p in layer.expert_params] for layer in self.layers
        ]
        self.assertEqual(self.ctx.unshard_log, expected_params)
        self.assertEqual(self.ctx.reshard_log, expected_params)
        self.assertIsNone(self.ctx.unsharded)
        self.assertEqual(self.layers[0].quant_calls, [(True, False)])
        self.assertEqual(self.layers[1].quant_calls, [(True, False)])

    def test_w4a8_rejection_propagates_and_reshards(self):
        layer = self.layers[0]
        moe = MoELayer.__new__(MoELayer)
        paddle.nn.Layer.__init__(moe)
        moe.moe_use_fusion_node = True
        moe.fp8 = "e4m3"
        moe.use_w4a8 = True
        moe.grouped_gemm_experts = layer.mlp.grouped_gemm_experts
        layer.mlp = moe

        def quantize(batch_mode, quant_transpose):
            self.assertEqual(self.ctx.unsharded, layer.expert_params)
            TransformerLayer.fp8_quant_weight(
                layer, batch_mode=batch_mode, quant_transpose=quant_transpose
            )

        layer.fp8_quant_weight = quantize
        with self.assertRaisesRegex(
            ValueError, "Offline W4A8 requires quant_transpose=True or None"
        ):
            self._step(_args([ShardingOption.FSDP]))
        expected = [[p.name for p in layer.expert_params]]
        self.assertEqual(self.ctx.unshard_log, expected)
        self.assertEqual(self.ctx.reshard_log, expected)
        self.assertIsNone(self.ctx.unsharded)
        self.assertEqual(self.layers[1].quant_calls, [])
        self.assertEqual(self.offloaded, [])

    def test_marked_experts_skip_pre_quant(self):
        """fsdp builds the fp8 caches of marked experts when gathering."""
        self.layers[0].expert_params[0]._fsdp_fp8_quantizer = object()
        self._step(_args([ShardingOption.FSDP], offload_master=False))
        self.assertEqual(self.ctx.unshard_log, [])
        for layer in self.layers:
            self.assertEqual(layer.quant_calls, [])

    def test_offloads_and_reloads_expert_master_weights_only(self):
        self._step(_args([ShardingOption.FSDP]))
        expected = ["mw_expert_fused", "mw_expert_w1"]
        self.assertEqual(self.offloaded, expected)
        self.assertEqual(self.reloaded, expected)

    def test_first_step_without_master_weights(self):
        """fsdp creates master weights lazily in the first optimizer step."""
        self.master_weights.clear()
        self._step(_args([ShardingOption.FSDP]))
        self.assertEqual(self.offloaded, [])
        self.assertEqual(self.reloaded, [])
        self.assertEqual(self.layers[0].quant_calls, [(True, False)])
        self.assertEqual(self.layers[1].quant_calls, [(True, False)])

    def test_no_offload_when_disabled(self):
        self._step(_args([ShardingOption.FSDP], offload_master=False))
        self.assertEqual(self.offloaded, [])
        self.assertEqual(self.reloaded, [])

    def test_missing_fsdp_context_fails_loudly(self):
        self._patch(_GET_FSDP_CONTEXT, lambda: None)
        with self.assertRaisesRegex(AssertionError, "no fsdp context"):
            self._step(_args([ShardingOption.FSDP]))
        for layer in self.layers:
            self.assertEqual(layer.quant_calls, [])

    def test_stage1_still_quantizes(self):
        """Negative control: stage1 keeps the whole-model + clear-storage path."""
        calls = []
        model = SimpleNamespace(
            fp8_quant_weight=lambda *a, **kw: calls.append(("quant", a, kw))
        )
        expert = SimpleNamespace(
            name="s1.expert", color={"color": "moe_expert"}
        )
        dense = SimpleNamespace(name="s1.dense", color=-1)
        optimizer = SimpleNamespace(
            clear_param_storage=lambda color: calls.append(("clear", color)),
            _inner_opt=SimpleNamespace(_parameter_list=[expert, dense]),
            _master_weights={"s1.expert": "mw_s1_expert", "s1.dense": "mw_d"},
        )
        self._patch(_GET_FSDP_CONTEXT, mock.Mock(side_effect=AssertionError))
        args = _args([ShardingOption.SHARD_OP])
        kwargs = {"model": model, "optimizer": optimizer}
        self.callback.on_step_begin(args, None, None, **kwargs)
        self.callback.on_optimizer_begin(args, None, None, **kwargs)
        self.assertEqual(
            calls,
            [("quant", (True,), {"quant_transpose": False})]
            + [("clear", c) for c in _STAGE1_COLORS],
        )
        self.assertEqual(self.ctx.unshard_log, [])
        self.assertEqual(self.offloaded, ["mw_s1_expert"])
        self.assertEqual(self.reloaded, ["mw_s1_expert"])


class TestFP8QuantTargetParams(unittest.TestCase):
    """Which params ``_fp8_quant_target_params`` hands to ``unshard_params``."""

    @staticmethod
    def _sub(name, is_expert=False, fp8=True, quant=True, has_weight=True):
        return SimpleNamespace(
            is_expert=is_expert,
            fp8=fp8,
            weight_quant_func=(lambda w: w) if quant else None,
            weight=_FakeParam(name) if has_weight else None,
        )

    @staticmethod
    def _layer(mlp, subs):
        return SimpleNamespace(
            mlp=mlp, sublayers=lambda include_self=False: list(subs)
        )

    def _names(self, layer):
        return [
            p.name for p in trainer_callback._fp8_quant_target_params(layer)
        ]

    def test_collects_experts_and_fp8_dense_weights(self):
        experts = SimpleNamespace(
            parameters=lambda: [_FakeParam("e.w1"), _FakeParam("e.w2")]
        )
        subs = [
            self._sub("expert_linear", is_expert=True),
            self._sub("bf16_linear", fp8=False),
            self._sub("no_quant_func", quant=False),
            self._sub("no_weight", has_weight=False),
            self._sub("qkv"),
            self._sub("proj"),
        ]
        layer = self._layer(SimpleNamespace(grouped_gemm_experts=experts), subs)
        self.assertEqual(self._names(layer), ["e.w1", "e.w2", "qkv", "proj"])

    def test_plain_experts_attribute(self):
        experts = SimpleNamespace(parameters=lambda: [_FakeParam("e.w")])
        layer = self._layer(SimpleNamespace(experts=experts), [])
        self.assertEqual(self._names(layer), ["e.w"])

    def test_dense_only_layer_without_mlp(self):
        layer = self._layer(None, [self._sub("qkv")])
        self.assertEqual(self._names(layer), ["qkv"])


class TestFSDPExpertMasterWeightNames(unittest.TestCase):
    def test_no_fsdp_context_returns_empty(self):
        with mock.patch(_GET_FSDP_CONTEXT, lambda: None):
            self.assertEqual(
                trainer_callback._fsdp_expert_master_weight_names(), []
            )

    def test_only_expert_groups(self):
        ctx = _FakeFSDPContext()
        ctx.add_group("fuse_params_expert_0", ["e.w1", "e.w2"], is_expert=True)
        ctx.add_group("fuse_params_dense_0", ["d.w"], is_expert=False)
        with mock.patch(_GET_FSDP_CONTEXT, lambda: ctx):
            self.assertEqual(
                trainer_callback._fsdp_expert_master_weight_names(),
                ["fuse_params_expert_0", "e.w1", "e.w2"],
            )


class TestMarkFSDPFP8Params(unittest.TestCase):
    """``on_init_end`` marks grouped experts for fsdp fp8 param gather."""

    class _Grouped:
        def __init__(self, prefix):
            self.weight1 = _FakeParam(f"{prefix}.w1")
            self.weight2 = _FakeParam(f"{prefix}.w2")

    class _Sonic(_Grouped):
        pass

    class _Weights(list):
        shape = (2,)

    def setUp(self):
        for name, new in (
            ("GroupedMLPExpert", self._Grouped),
            ("SonicMoEExpert", self._Sonic),
        ):
            patcher = mock.patch.object(moe_expert, name, new)
            patcher.start()
            self.addCleanup(patcher.stop)
        # Pretend paddle supports fsdp fp8 gather.
        patcher = mock.patch.dict(sys.modules, {_PADDLE_FP8: SimpleNamespace()})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.grouped = self._Grouped("grouped")
        self.sonic = self._Sonic("sonic")
        self.subs = [object(), self.grouped, self.sonic]
        self.model = SimpleNamespace(sublayers=lambda: list(self.subs))

    @staticmethod
    def _args(offline=True, **kwargs):
        return SimpleNamespace(offline_quant_expert_weight=offline, **kwargs)

    def test_on_init_end_marks_only_under_fsdp(self):
        callback = FP8QuantWeightCallback()
        with mock.patch.object(fp8_utils, "mark_fsdp_fp8_params") as mark:
            for args in (
                _args([ShardingOption.FSDP], using_sonic_moe=True),
                _args([ShardingOption.SHARD_OP]),
            ):
                callback.on_init_end(args, None, None, model=self.model)
            mark.assert_not_called()
            args = _args([ShardingOption.FSDP])
            callback.on_init_end(args, None, None, model=self.model)
        mark.assert_called_once_with(args, self.model)

    def test_marks_nothing_when_off_or_unsupported(self):
        mark = fp8_utils.mark_fsdp_fp8_params
        mark(self._args(offline=False), self.model)
        with mock.patch.dict(sys.modules, {_PADDLE_FP8: None}):
            mark(self._args(), self.model)
        self.assertFalse(hasattr(self.grouped.weight1, "_fsdp_fp8_quantizer"))
        self.subs.append(SimpleNamespace(use_ue8m0=True))
        with self.assertRaises(NotImplementedError):
            mark(self._args(), self.model)

    def test_marks_grouped_experts_not_sonic(self):
        fp8_utils.mark_fsdp_fp8_params(self._args(), self.model)
        quantizer = self.grouped.weight1._fsdp_fp8_quantizer
        self.assertFalse(hasattr(self.sonic.weight1, "_fsdp_fp8_quantizer"))
        self.assertEqual(
            [transpose for _, _, transpose in quantizer.attrs], [False, True]
        )
        with mock.patch.object(
            fp8_utils, "fused_stack_quant_without_cache"
        ) as quant:
            quantizer(self._Weights(["e0", "e1"]), True)
        quant.assert_called_once_with(["e0", "e1"], transpose=True)

        args = self._args(using_cache_weight_quant_transpose=False)
        fp8_utils.mark_fsdp_fp8_params(args, self.model)
        attrs = self.grouped.weight1._fsdp_fp8_quantizer.attrs
        self.assertEqual([transpose for _, _, transpose in attrs], [False])


class TestSonicMoECallbackUnderFSDP(unittest.TestCase):
    def setUp(self):
        self.callback = SonicMoELayoutSwitchCallback()
        self.kwargs = {
            "model": _ExplodingModel(),
            "optimizer": _ExplodingOptimizer(),
        }

    def test_on_step_begin_rejects_fsdp(self):
        args = _args([ShardingOption.FSDP], using_sonic_moe=True)
        with self.assertRaises(NotImplementedError):
            self.callback.on_step_begin(args, None, None, **self.kwargs)

    def test_on_optimizer_begin_rejects_fsdp(self):
        args = _args([ShardingOption.FSDP], using_sonic_moe=True)
        with self.assertRaises(NotImplementedError):
            self.callback.on_optimizer_begin(args, None, None, **self.kwargs)

    def test_no_op_when_sonic_moe_disabled(self):
        args = _args([ShardingOption.FSDP], using_sonic_moe=False)
        self.callback.on_step_begin(args, None, None, **self.kwargs)
        self.callback.on_optimizer_begin(args, None, None, **self.kwargs)


if __name__ == "__main__":
    unittest.main()
