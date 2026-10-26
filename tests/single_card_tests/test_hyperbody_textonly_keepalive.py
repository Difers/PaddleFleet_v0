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
"""Unit coverage for the HyperBody text-only encoder keepalive.

A text-only micro-batch (no image/audio/context -- e.g. one sample degraded by
an image-read failure) used to drop the encoder out of the backward graph on
that rank only. The encoder's MoE layers then issued ZERO DeepEP
dispatch/combine calls there while every other EP rank issued the full set, so
the group deadlocked until the ~100 s CPU-side timeout fired.

The fix keeps the encoder in the backward graph on text-only ranks while
contributing an exact zero. Covered here:

  * ``_ZeroGradKeepalive``: zeros forward, zeros backward, and -- the property
    that actually prevents the hang -- the producer stays in the backward graph
    (its params get grad ``0``, NOT ``None``);
  * non-finite robustness: an ``inf`` encoder output cannot leak into the
    decoder input or the gradients (which ``x * 0.0`` would do via ``0*inf``);
  * ``HyperBodyEncoderBridge.forward`` text-only branch: emits the keepalive
    scalar, keeps ``image_embeds=None``, and routes dense vs packed;
  * ``HyperBodyEncoderFrontEnd._forward_text_only``: the dummy context is built
    through the real context builder (so a text-only rank touches the same
    params as a multimodal rank) and zeroed;
  * ``GPTEmbedding._merge_multimodal``: adds the keepalive without changing any
    value, and is an exact no-op when the key is absent (other models unaffected);
  * ``moe_router_force_load_balancing`` survives config construction and reaches
    the encoder view.

Pure single-card dynamic graph: layers are built via ``__new__`` with stubbed
sub-modules, so no GPU topology / distributed init is required.
"""

import os
import sys
import unittest

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 2)
)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import numpy as np
import paddle

from paddlefleet.models.gpt.gpt_embedding import GPTEmbedding
from paddlefleet.transformers.hyperbody.modeling import (
    HyperBodyEncoderBridge,
    HyperBodyEncoderFrontEnd,
    _ZeroGradKeepalive,
)


def _bare(cls, **attrs):
    """Instantiate a ``FleetLayer`` subclass without running ``__init__``.

    Only the attributes a single method touches are populated, so these tests
    stay on one card with no distributed init.
    """
    obj = cls.__new__(cls)
    obj.__dict__.setdefault("_parameters", {})
    obj.__dict__.setdefault("_buffers", {})
    obj.__dict__.setdefault("_sub_layers", {})
    obj.__dict__.setdefault("_non_persistable_buffers", set())
    for key, value in attrs.items():
        object.__setattr__(obj, key, value)
    return obj


def _make_embedding():
    """``GPTEmbedding`` stub that only supports ``_merge_multimodal``."""
    return _bare(
        GPTEmbedding,
        config=None,
        multimodal_embedding=True,
        sequence_parallel=False,
        training=False,
    )


class _Producer(paddle.nn.Layer):
    """Stand-in for the encoder trunk + bridge projector."""

    def __init__(self, hidden=4):
        super().__init__()
        self.linear = paddle.nn.Linear(hidden, hidden)

    def forward(self, x):
        return self.linear(x)

    def poison(self):
        """Force a non-finite output to emulate a diverged encoder."""
        with paddle.no_grad():
            self.linear.weight.set_value(
                paddle.full_like(self.linear.weight, float("inf"))
            )


class TestZeroGradKeepalive(unittest.TestCase):
    """``_ZeroGradKeepalive``: zeros both ways, but the edge survives."""

    def test_forward_is_zeros_with_input_shape_and_dtype(self):
        for dtype in ("float32", "bfloat16"):
            x = paddle.full([3, 5], 2.5, dtype=dtype)
            out = _ZeroGradKeepalive.apply(x)
            self.assertEqual(list(out.shape), [3, 5])
            self.assertEqual(out.dtype, x.dtype)
            self.assertEqual(float(out.astype("float32").abs().max()), 0.0)

    def test_producer_stays_in_backward_graph_with_zero_grad(self):
        """The anti-deadlock property: grad is 0, NOT None.

        ``None`` would mean autograd pruned the branch -- i.e. the encoder
        backward never ran and this rank would skip its DeepEP calls.
        """
        prod = _Producer()
        x = paddle.ones([2, 4])
        keep = _ZeroGradKeepalive.apply(prod(x)).sum()
        (x + keep).sum().backward()

        for name, param in prod.named_parameters():
            self.assertIsNotNone(
                param.grad, f"{name} was pruned from the graph"
            )
            self.assertEqual(float(param.grad.abs().max()), 0.0, name)

    def test_keepalive_value_is_exactly_zero(self):
        prod = _Producer()
        keep = _ZeroGradKeepalive.apply(prod(paddle.ones([2, 4]))).sum()
        self.assertEqual(float(keep), 0.0)

    def test_non_finite_producer_does_not_leak(self):
        """An ``inf`` encoder output must not reach the decoder or the grads.

        This is the reason for a PyLayer over ``x * 0.0``: the forward output
        does not depend on ``x``, so ``0 * inf = nan`` never happens.
        """
        prod = _Producer()
        prod.poison()
        x = paddle.ones([2, 4])
        keep = _ZeroGradKeepalive.apply(prod(x)).sum()
        dec = x + keep
        (dec * dec).sum().backward()

        self.assertEqual(float(keep), 0.0)
        self.assertTrue(bool(paddle.isfinite(dec).all()))
        np.testing.assert_allclose(dec.numpy(), x.numpy())
        for name, param in prod.named_parameters():
            self.assertIsNotNone(param.grad, name)
            self.assertTrue(bool(paddle.isfinite(param.grad).all()), name)
            self.assertEqual(float(param.grad.abs().max()), 0.0, name)

    def test_mul_zero_would_leak_nan(self):
        """Contrast: the ``x * 0.0`` formulation propagates nan once poisoned."""
        prod = _Producer()
        prod.poison()
        x = paddle.ones([2, 4])
        dec = x + prod(x).sum() * 0.0
        self.assertFalse(bool(paddle.isfinite(dec).all()))


def _dec_fields(hidden=4, seq=6):
    """The decoder-bound fields the bridge renames back out of ``_hb_dec_*``."""
    return {
        "_hb_dec_input_ids": paddle.ones([1, seq], dtype="int64"),
        "_hb_dec_labels": paddle.zeros([1, seq], dtype="int64"),
        "_hb_dec_attn_mask_startend_row_indices": None,
        "_hb_dec_position_ids": None,
    }


class TestBridgeTextOnlyBranch(unittest.TestCase):
    """``HyperBodyEncoderBridge.forward`` under ``_hb_text_only``."""

    def _run(self, layout=None):
        prod = _Producer()
        calls = {}

        def _dense(dict_args):
            calls["dense"] = True
            return prod(paddle.ones([2, 4]))

        def _packed(dict_args, got_layout):
            calls["packed"] = got_layout
            return prod(paddle.ones([2, 4]))

        bridge = _bare(
            HyperBodyEncoderBridge,
            _forward_dense=_dense,
            _forward_packed=_packed,
        )
        args = dict(_dec_fields(), _hb_text_only=True)
        if layout is not None:
            args["_hb_packed"] = layout
        return bridge.forward(args), calls, prod, args

    def test_dense_path_emits_zero_keepalive_and_no_image_embeds(self):
        out, calls, _, args = self._run()
        self.assertTrue(calls.get("dense"))
        self.assertNotIn("packed", calls)
        # keepalive present and exactly zero
        self.assertIn("multimodal_keepalive", out)
        self.assertEqual(float(out["multimodal_keepalive"]), 0.0)
        # the decoder must stay on pure text: no scatter, no reused hidden state
        self.assertIsNone(out["image_embeds"])
        self.assertIsNone(out["video_embeds"])
        self.assertIsNone(out["decoder_input"])
        # decoder-bound fields are renamed back
        self.assertIs(out["input_ids"], args["_hb_dec_input_ids"])
        self.assertIs(out["labels"], args["_hb_dec_labels"])

    def test_packed_path_receives_layout(self):
        layout = {"segment_starts": [0], "n_contexts": [1], "n_queries": [1]}
        out, calls, _, _ = self._run(layout=layout)
        self.assertIs(calls.get("packed"), layout)
        self.assertNotIn("dense", calls)
        self.assertEqual(float(out["multimodal_keepalive"]), 0.0)

    def test_text_only_keeps_the_decoder_passthrough_fields(self):
        """The MTP passthrough must survive the text-only branch too."""
        out, _, _, _ = self._run()
        self.assertIn("multimodal_keepalive", out)
        # absent passthrough stays absent (bridge writes only non-None values)
        self.assertNotIn("mtp_startend_row_indices_all", out)

        prod = _Producer()
        bridge = _bare(
            HyperBodyEncoderBridge,
            _forward_dense=lambda d: prod(paddle.ones([2, 4])),
            _forward_packed=lambda d, layout: prod(paddle.ones([2, 4])),
        )
        mtp_rows = paddle.zeros([1, 5, 2], dtype="int32")
        args = dict(
            _dec_fields(),
            _hb_text_only=True,
            _hb_dec_mtp_startend_row_indices_all=mtp_rows,
        )
        out = bridge.forward(args)
        self.assertIs(out["mtp_startend_row_indices_all"], mtp_rows)
        self.assertEqual(float(out["multimodal_keepalive"]), 0.0)

    def test_keepalive_keeps_encoder_in_backward_graph(self):
        """Backward from the keepalive must reach the encoder with grad 0."""
        out, _, prod, _ = self._run()
        out["multimodal_keepalive"].backward()
        for name, param in prod.named_parameters():
            self.assertIsNotNone(param.grad, f"{name} pruned -> would deadlock")
            self.assertEqual(float(param.grad.abs().max()), 0.0, name)


class TestFrontEndTextOnlyDummy(unittest.TestCase):
    """``HyperBodyEncoderFrontEnd._forward_text_only`` builds a zeroed dummy."""

    def _run(self, packed=False, dtype="float32"):
        hidden = 4
        prod = _Producer(hidden)
        seen = {}

        def _embed_one_context(ids_i, img_i, aud_i):
            seen["ids"] = ids_i
            seen["img"] = img_i
            seen["aud"] = aud_i
            # a *non-zero* context, so that zeroing is observable
            return prod(paddle.ones([1, hidden])) + 3.0

        def _trunk(dummy_ctx, *rest):
            seen["dummy"] = dummy_ctx
            seen["rest"] = rest
            return {}

        frontend = _bare(
            HyperBodyEncoderFrontEnd,
            packed=packed,
            query=_bare(
                paddle.nn.Layer, weight=paddle.zeros([1, hidden], dtype=dtype)
            ),
            _embed_one_context=_embed_one_context,
            _forward_packed=_trunk,
            _forward_dense=_trunk,
        )
        dict_args = {
            "input_ids": paddle.ones([1, 6], dtype="int64"),
            "labels": paddle.zeros([1, 6], dtype="int64"),
        }
        out = frontend._forward_text_only(dict_args, "mask_fn", "pad_fn")
        return out, seen, prod, dict_args

    def test_dummy_goes_through_real_context_builder_with_zero_ids(self):
        """Param-set alignment: the real builder is used, on a zero int64 id."""
        _, seen, _, _ = self._run()
        self.assertEqual(seen["ids"].dtype, paddle.int64)
        self.assertEqual(list(seen["ids"].shape), [1])
        self.assertEqual(float(seen["ids"].astype("float32").abs().max()), 0.0)
        # no image / audio on a text-only rank
        self.assertIsNone(seen["img"])
        self.assertIsNone(seen["aud"])

    def test_dummy_context_is_zeroed_and_shaped_for_the_trunk(self):
        _, seen, _, _ = self._run()
        dummy = seen["dummy"]
        self.assertEqual(list(dummy.shape), [1, 1, 4])
        self.assertEqual(float(dummy.abs().max()), 0.0)

    def test_dummy_keeps_frontend_params_in_the_graph(self):
        """``ctx * 0.0`` (not ``zeros_like``) so frontend params get grad 0."""
        _, seen, prod, _ = self._run()
        seen["dummy"].sum().backward()
        for name, param in prod.named_parameters():
            self.assertIsNotNone(
                param.grad, f"{name} pruned -> grad set mismatch"
            )
            self.assertEqual(float(param.grad.abs().max()), 0.0, name)

    def test_dummy_follows_query_table_dtype(self):
        _, seen, _, _ = self._run(dtype="bfloat16")
        self.assertEqual(seen["dummy"].dtype, paddle.bfloat16)

    def test_text_only_flag_and_decoder_fields_are_carried(self):
        out, _, _, dict_args = self._run()
        self.assertTrue(out["_hb_text_only"])
        self.assertIs(out["_hb_dec_input_ids"], dict_args["input_ids"])
        self.assertIs(out["_hb_dec_labels"], dict_args["labels"])
        self.assertIsNone(out["_hb_dec_attn_mask_startend_row_indices"])
        self.assertIsNone(out["_hb_dec_position_ids"])

    def test_packed_layout_routes_to_forward_packed(self):
        _, seen, _, _ = self._run(packed=True)
        # dense passes (mask_fn, pad_fn) through; packed takes neither
        self.assertEqual(seen["rest"], ())


class TestMergeMultimodalKeepalive(unittest.TestCase):
    """``GPTEmbedding._merge_multimodal`` consumes the keepalive scalar."""

    def setUp(self):
        self.hidden = 4
        self.input_ids = paddle.ones([1, 6], dtype="int64")
        self.decoder_input = paddle.full(
            [1, 6, self.hidden], 1.5, dtype="float32"
        )

    def test_absent_key_is_an_exact_noop(self):
        """Other models (and multimodal ranks) must be untouched."""
        emb = _make_embedding()
        out, masks, deepstack = emb._merge_multimodal(
            {}, self.input_ids, self.decoder_input, None, None
        )
        self.assertIs(out, self.decoder_input)
        self.assertIsNone(masks)
        self.assertIsNone(deepstack)

    def test_keepalive_does_not_change_any_value(self):
        emb = _make_embedding()
        keep = _ZeroGradKeepalive.apply(paddle.ones([2, self.hidden])).sum()
        out, masks, deepstack = emb._merge_multimodal(
            {"multimodal_keepalive": keep},
            self.input_ids,
            self.decoder_input,
            None,
            None,
        )
        self.assertEqual(list(out.shape), [1, 6, self.hidden])
        np.testing.assert_array_equal(out.numpy(), self.decoder_input.numpy())
        self.assertIsNone(masks)
        self.assertIsNone(deepstack)

    def test_keepalive_is_cast_to_the_decoder_dtype(self):
        emb = _make_embedding()
        keep = _ZeroGradKeepalive.apply(
            paddle.ones([2], dtype="bfloat16")
        ).sum()
        out, _, _ = emb._merge_multimodal(
            {"multimodal_keepalive": keep},
            self.input_ids,
            self.decoder_input,
            None,
            None,
        )
        self.assertEqual(out.dtype, self.decoder_input.dtype)

    def test_keepalive_connects_the_encoder_to_the_loss(self):
        """End-to-end: loss -> decoder_input -> keepalive -> encoder, grad 0."""
        emb = _make_embedding()
        prod = _Producer(self.hidden)
        keep = _ZeroGradKeepalive.apply(
            prod(paddle.ones([2, self.hidden]))
        ).sum()
        out, _, _ = emb._merge_multimodal(
            {"multimodal_keepalive": keep},
            self.input_ids,
            self.decoder_input,
            None,
            None,
        )
        (out * out).sum().backward()
        for name, param in prod.named_parameters():
            self.assertIsNotNone(param.grad, f"{name} pruned -> would deadlock")
            self.assertEqual(float(param.grad.abs().max()), 0.0, name)


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


class TestForceLoadBalancingPassthrough(unittest.TestCase):
    """``moe_router_force_load_balancing`` survives config build and reaches the view.

    The flag is consumed by ``set_expected_keys`` before ``PretrainedConfig``
    pops it, so popping with a ``False`` default used to silently reset it and
    the fake-gate switch never took effect.
    """

    @staticmethod
    def _config(**top_level):
        from paddlefleet.transformers.hyperbody.configuration import (
            HyperBodyConfig,
        )

        return HyperBodyConfig(
            decoder_config={**_DECODER_GEOMETRY},
            encoder_config={**_ENCODER_GEOMETRY},
            **top_level,
        )

    def test_flag_is_not_reset_by_pretrained_config(self):
        for flag in (True, False):
            cfg = self._config(moe_router_force_load_balancing=flag)
            self.assertIs(cfg.moe_router_force_load_balancing, flag)

    def test_flag_reaches_the_encoder_view(self):
        from paddlefleet.transformers.hyperbody.modeling import (
            _build_encoder_view,
        )

        for flag in (True, False):
            cfg = self._config(moe_router_force_load_balancing=flag)
            view = _build_encoder_view(cfg, _DECODER_GEOMETRY["hidden_size"])
            self.assertIs(view.moe_router_force_load_balancing, flag)

    def test_flag_defaults_to_false_when_absent(self):
        from paddlefleet.transformers.hyperbody.modeling import (
            _build_encoder_view,
        )

        cfg = self._config()
        self.assertIs(cfg.moe_router_force_load_balancing, False)
        view = _build_encoder_view(cfg, _DECODER_GEOMETRY["hidden_size"])
        self.assertIs(view.moe_router_force_load_balancing, False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
