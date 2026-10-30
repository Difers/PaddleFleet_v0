# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

"""CPU regression tests for ``unshard_`` buffers in ZCC EMA reshard load.

``ZeroCostCheckpointEMAProcessor.load_ema_state_dict`` walks the fused param
buffers and, for ``unshard_`` entries, used to write the whole buffer with
``buffer[:] = value.flatten()``. ``unshard_`` buffers deliberately keep the
parameter's original shape -- ``init_buffer`` only flattens the fused comm
buffers -- so that write broke at both ends of the shape range:

  * 0-D scalar buffers, e.g. the quantile-balancing router's ``qb_bin_min`` /
    ``qb_bin_max``, raise ``Too many indices (1) for tensor of dimension 0``;
  * buffers with ndim >= 2, e.g. a frozen MoE ``gate.weight`` of shape
    ``[n_routed_experts, hidden_size]`` -- freezing keeps it out of every
    optimizer buffer, so it falls back to the ``unshard_`` tier -- raise
    ``The shape of tensor assigned value must match the shape of target
    shape``, which killed the ZCC EMA worker during a resharded resume.

Both are now served by one ``paddle.assign(value.reshape(dst.shape), dst)``.
These tests pin that behaviour without a GPU or the fleet stack: the value
lands in the same buffer object, the buffer keeps its own shape, and the source
may arrive flattened or already shaped.
"""

import unittest
from types import SimpleNamespace

import numpy as np
import paddle

from paddlefleet.trainer.utils.zero_cost_checkpoint import (
    ZeroCostCheckpointEMAProcessor,
)


def _make_processor(model_weights_metas, ema_buffer_model_params):
    """A processor carrying only what ``load_ema_state_dict`` reads.

    ``__init__`` builds real fused buffers from an optimizer / param helper; the
    loader only touches ``param_fusion_storage_helper.model_weights_metas`` and
    ``ema_buffer_model_params``, so bypass ``__init__`` and set those two.
    """
    proc = ZeroCostCheckpointEMAProcessor.__new__(
        ZeroCostCheckpointEMAProcessor
    )
    proc.param_fusion_storage_helper = SimpleNamespace(
        model_weights_metas=model_weights_metas
    )
    proc.ema_buffer_model_params = ema_buffer_model_params
    # After the model-weight loop the loader also walks master weights; keep that
    # portion a no-op so these tests stay focused on the unshard_ buffer branch.
    proc.optimizer_fusion_storage_helper = SimpleNamespace(
        master_weights_meta={}
    )
    proc.master_min_offset = 0
    proc.ema_buffer = paddle.zeros([1], dtype="float32")
    return proc


class TestZCCEMAScalarBuffer(unittest.TestCase):
    def setUp(self):
        paddle.set_device("cpu")

    def _run_scalar(self, key, source, expected):
        buffer_index = f"unshard_{key}"
        dst = paddle.zeros([], dtype="float32")  # 0-D scalar buffer
        proc = _make_processor(
            {key: {"buffer_index": buffer_index, "name": key, "shape": []}},
            {buffer_index: dst},
        )

        proc.load_ema_state_dict(
            {key: source, "master_weights": {}}
        )  # must not raise

        loaded = proc.ema_buffer_model_params[buffer_index]
        self.assertIs(
            loaded, dst, "scalar must be written into the same object"
        )
        self.assertEqual(loaded.ndim, 0, "dst must stay 0-D")
        self.assertEqual(float(loaded.item()), expected)

    def test_zero_dim_source(self):
        # Source already 0-D -- reshape(dst.shape) is reshape([]), a no-op copy.
        self._run_scalar(
            "model.layers.0.mlp.gate.qb_bin_max",
            paddle.to_tensor(3.5, dtype="float32"),
            3.5,
        )

    def test_one_element_source(self):
        # The reshard / flex path can hand the scalar back as shape [1]; it must
        # still land in the 0-D buffer without a "too many indices" error.
        self._run_scalar(
            "model.layers.0.mlp.gate.qb_bin_min",
            paddle.to_tensor([-2.25], dtype="float32"),
            -2.25,
        )

    def test_1d_unshard_buffer(self):
        # A 1-D unshard buffer was already served correctly by the old
        # dst[:] = value.flatten() write; it must keep working.
        key = "model.layers.0.self_attn.norm.weight"
        buffer_index = f"unshard_{key}"
        dst = paddle.zeros([4], dtype="float32")
        proc = _make_processor(
            {key: {"buffer_index": buffer_index, "name": key, "shape": [4]}},
            {buffer_index: dst},
        )
        source = paddle.to_tensor([1.0, 2.0, 3.0, 4.0], dtype="float32")

        proc.load_ema_state_dict({key: source, "master_weights": {}})

        loaded = proc.ema_buffer_model_params[buffer_index]
        self.assertIs(loaded, dst)
        self.assertEqual(list(loaded.shape), [4])
        np.testing.assert_array_equal(
            loaded.numpy(), np.array([1.0, 2.0, 3.0, 4.0], dtype="float32")
        )

    def test_missing_key_is_noop(self):
        # A buffer whose key is absent from the incoming state dict is skipped,
        # leaving the pre-existing 0-D value untouched (no crash on the guard).
        key = "gate.qb_bin_max"
        buffer_index = f"unshard_{key}"
        dst = paddle.full([], 7.0, dtype="float32")
        proc = _make_processor(
            {key: {"buffer_index": buffer_index, "name": key, "shape": []}},
            {buffer_index: dst},
        )

        proc.load_ema_state_dict({"master_weights": {}})  # key not present

        self.assertEqual(
            float(proc.ema_buffer_model_params[buffer_index].item()), 7.0
        )


class TestZCCEMAUnshardNDBuffer(unittest.TestCase):
    """Buffers with ndim >= 2, the shape the old ``[:]`` write could not hit."""

    def setUp(self):
        paddle.set_device("cpu")

    def _run_nd(self, key, shape, source_shape):
        buffer_index = f"unshard_{key}"
        dst = paddle.zeros(shape, dtype="float32")
        proc = _make_processor(
            {key: {"buffer_index": buffer_index, "name": key, "shape": shape}},
            {buffer_index: dst},
        )
        expected = np.arange(np.prod(shape), dtype="float32").reshape(shape)
        source = paddle.to_tensor(
            expected.reshape(source_shape), dtype="float32"
        )

        proc.load_ema_state_dict({key: source, "master_weights": {}})

        loaded = proc.ema_buffer_model_params[buffer_index]
        self.assertIs(loaded, dst, "value must land in the same buffer object")
        self.assertEqual(list(loaded.shape), list(shape))
        np.testing.assert_array_equal(loaded.numpy(), expected)

    def test_2d_gate_weight_flattened_source(self):
        # The observed failure: an encoder MoE gate.weight is fp32
        # [n_routed_experts=16, hidden_size=1280] and its EMA buffer keeps that
        # shape, while the old code flattened the source to [20480] first.
        self._run_nd("encoder.layers.1.mlp.gate.weight", [16, 1280], (20480,))

    def test_2d_gate_weight_shaped_source(self):
        # The non-resharded resume path hands the tensor back already shaped.
        self._run_nd("encoder.layers.1.mlp.gate.weight", [16, 1280], (16, 1280))

    def test_3d_buffer(self):
        # Guards against a fix that only special-cases ndim == 2.
        self._run_nd("model.layers.0.mlp.gate.weight", [2, 3, 4], (24,))


if __name__ == "__main__":
    unittest.main()
