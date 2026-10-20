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

"""No-card tests for the fp8 expert dispatch and the DSpark taps.

Three hand-written contracts live here, all of which fail far away from their
cause when they regress:

* which layers ``GPTModel.fp8_quant_weight`` drives.  Wrappers hide the layers
  that own the routed-expert weights, and a layer the pass skips keeps its
  bf16 storage freed by ``clear_param_storage("moe_expert")`` -- it then dies
  later inside ``fused_stack_quant`` on a 0-byte tensor, which is exactly what
  happened to the DSpark draft experts.
* the P2P key carry that lets the draft still resolve the target features it
  was trained to consume.
* the log line that makes the draft loss visible.

The layer objects are real instances of the real classes, built through
``__new__``, so ``isinstance`` dispatch is genuinely exercised while no
``__init__`` runs -- which is what keeps the whole file free of cards, fleet
init and pipeline setup.
"""

import unittest

import paddle

from paddlefleet.models.common.language_loss.language_loss import (
    _publish_dspark_loss,
)
from paddlefleet.models.gpt.gpt_model import GPTModel
from paddlefleet.training.global_vars import (
    get_global_training_logs,
    set_global_training_logs,
)
from paddlefleet.transformer.multi_token_prediction import (
    MultiTokenPredictionLayer,
    _carry_dspark_capture_keys,
)
from paddlefleet.transformer.parallel_multi_token_prediction import (
    DsparkPredictionLayer,
)
from paddlefleet.transformer.transformer_layer import TransformerLayer


def stub(cls, **attrs):
    """A real ``cls`` instance whose ``__init__`` never ran."""
    obj = cls.__new__(cls)
    paddle.nn.Layer.__init__(obj)
    for name, value in attrs.items():
        setattr(obj, name, value)
    return obj


class _Chunk:
    """Stand-in for ``PipelineLayerChunk``: iterating it yields its layers."""

    def __init__(self, layers):
        self._layers = list(layers)

    def __iter__(self):
        return iter(self._layers)


class _Model:
    """The minimal ``GPTModel`` surface: the fp8 pass over the layer tree."""

    fp8_quant_weight = GPTModel.fp8_quant_weight
    clear_fp8_quant_weight = GPTModel.clear_fp8_quant_weight

    def __init__(self, *, stages=1, chunks=None, flat=None):
        self._num_virtual_pipeline_stages = stages
        if chunks is not None:
            self._model_chunks = chunks
        self.run_function = list(flat or [])


def spy(layer, calls):
    """Attach recording ``fp8_quant_weight`` / ``clear_fp8_quant_weight``."""
    layer.fp8_quant_weight = lambda **kw: calls.append((layer, "quant", kw))
    layer.clear_fp8_quant_weight = lambda: calls.append((layer, "clear", {}))
    return layer


def called(calls, index):
    return [call[index] for call in calls]


class TestDsparkCaptureKeyCarry(unittest.TestCase):
    """``_carry_dspark_capture_keys`` -- the P2P rebuild whitelist."""

    def setUp(self):
        self.tap = object()
        self.last = object()
        self.dict_args = {
            "hidden_states": object(),
            "labels": object(),
            "dspark_target_hidden_3": self.tap,
            "dspark_target_last_hidden_states": self.last,
        }

    def test_the_taps_survive_the_rebuild(self):
        new_args = {"hidden_states": self.dict_args["hidden_states"]}
        _carry_dspark_capture_keys(self.dict_args, new_args)
        self.assertIs(new_args["dspark_target_hidden_3"], self.tap)
        self.assertIs(new_args["dspark_target_last_hidden_states"], self.last)

    def test_the_taps_are_carried_by_reference(self):
        """A copy would break the gradient path back to the tapped layer."""
        new_args = {}
        _carry_dspark_capture_keys(self.dict_args, new_args)
        self.assertIs(
            new_args["dspark_target_hidden_3"],
            self.dict_args["dspark_target_hidden_3"],
        )

    def test_unrelated_keys_are_not_carried(self):
        new_args = {}
        _carry_dspark_capture_keys(self.dict_args, new_args)
        self.assertEqual(
            sorted(new_args),
            [
                "dspark_target_hidden_3",
                "dspark_target_last_hidden_states",
            ],
        )

    def test_absent_and_none_taps_are_skipped(self):
        new_args = {}
        _carry_dspark_capture_keys(
            {
                "dspark_target_hidden_1": None,
                "dspark_target_hiddenness": object(),
            },
            new_args,
        )
        self.assertEqual(new_args, {})


class TestPublishDsparkLoss(unittest.TestCase):
    """``_publish_dspark_loss`` -- making the draft loss observable."""

    def tearDown(self):
        set_global_training_logs(None)

    def test_the_draft_loss_lands_in_the_training_logs(self):
        logs = {}
        set_global_training_logs(logs)
        _publish_dspark_loss(paddle.to_tensor(0.25))
        self.assertEqual(sorted(logs), ["dspark_loss"])
        self.assertEqual(float(logs["dspark_loss"]), 0.25)

    def test_the_published_value_is_detached(self):
        """Logging must not extend the graph or leak into the main loss."""
        logs = {}
        set_global_training_logs(logs)
        _publish_dspark_loss(paddle.to_tensor(0.25, stop_gradient=False))
        self.assertTrue(logs["dspark_loss"].stop_gradient)
        self.assertIsInstance(logs["dspark_loss"], paddle.Tensor)

    def test_no_registered_logs_is_not_an_error(self):
        set_global_training_logs(None)
        self.assertIsNone(get_global_training_logs())
        self.assertIsNone(_publish_dspark_loss(paddle.to_tensor(1.0)))

    def test_a_logs_object_without_update_is_tolerated(self):
        set_global_training_logs(object())
        self.assertIsNone(_publish_dspark_loss(paddle.to_tensor(1.0)))


class TestFp8QuantWeightDispatch(unittest.TestCase):
    """``fp8_quant_weight`` / ``clear_fp8_quant_weight`` reach the real owners.

    Both wrappers hide the layer that owns the routed-expert weights:
    ``MultiTokenPredictionLayer`` keeps its transformer layer in
    ``.transformer_layer``, and the DSpark draft blocks sit in
    ``DsparkPredictionLayer.layers``. Driving the wrapper instead of the owner
    is the bug that left the draft experts without an offline fp8 stack.
    """

    def _model_with_calls(self):
        calls = []
        backbone = spy(stub(TransformerLayer, layer_number=1), calls)
        inner = spy(stub(TransformerLayer, layer_number=9), calls)
        mtp = stub(
            MultiTokenPredictionLayer, layer_number=8, transformer_layer=inner
        )
        draft_subs = [
            spy(stub(TransformerLayer, layer_number=100), calls),
            spy(stub(TransformerLayer, layer_number=101), calls),
        ]
        draft = stub(DsparkPredictionLayer, layer_number=43, layers=draft_subs)
        model = _Model(flat=[backbone, mtp, draft])
        return model, calls

    @staticmethod
    def _owners(model):
        """The layers that own the weights, in the order the pass visits them."""
        backbone, mtp, draft = model.run_function
        return [backbone, mtp.transformer_layer, *draft.layers]

    def test_the_pass_drives_every_owner_exactly_once(self):
        model, calls = self._model_with_calls()
        model.fp8_quant_weight(True, quant_transpose=False)
        self.assertEqual(
            [id(layer) for layer in called(calls, 0)],
            [id(layer) for layer in self._owners(model)],
        )

    def test_vpp_layout_drives_every_chunk(self):
        calls = []
        first = spy(stub(TransformerLayer, layer_number=0), calls)
        second = spy(stub(TransformerLayer, layer_number=1), calls)
        model = _Model(
            stages=2, chunks=[_Chunk([first]), _Chunk([second])], flat=[]
        )
        model.fp8_quant_weight()
        self.assertEqual(
            [id(layer) for layer in called(calls, 0)], [id(first), id(second)]
        )

    def test_the_pass_forwards_its_arguments_unchanged(self):
        model, calls = self._model_with_calls()
        model.fp8_quant_weight(batch_mode=True, quant_transpose=False)
        self.assertEqual(len(calls), 4)
        for layer, action, kwargs in calls:
            with self.subTest(layer=layer.layer_number):
                self.assertEqual(action, "quant")
                self.assertEqual(
                    kwargs, {"batch_mode": True, "quant_transpose": False}
                )

    def test_the_clear_pass_mirrors_the_quantise_pass(self):
        model, calls = self._model_with_calls()
        model.clear_fp8_quant_weight()
        self.assertEqual(called(calls, 1), ["clear"] * 4)
        self.assertEqual(
            [id(layer) for layer in called(calls, 0)],
            [id(layer) for layer in self._owners(model)],
        )

    def test_the_draft_blocks_are_not_reached_through_the_wrapper(self):
        """The wrapper has no weights of its own; its sublayers do."""
        model, calls = self._model_with_calls()
        model.fp8_quant_weight()
        draft = model.run_function[2]
        reached = [id(layer) for layer in called(calls, 0)]
        self.assertNotIn(id(draft), reached)
        for sub in draft.layers:
            self.assertIn(id(sub), reached)


if __name__ == "__main__":
    unittest.main()
