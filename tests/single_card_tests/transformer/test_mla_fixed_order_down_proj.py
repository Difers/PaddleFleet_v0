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
"""``_FixedOrderDownProj`` keeps ``d hidden_states`` independent of
``mla_qkv_recompute``.

``hidden_states`` feeds q_a_proj, kv_a_proj_with_mqa and (gated attention
without q_lora) gate_proj. Folding the up-projection into one
``RecomputeWithoutOutput`` node reorders how those three bf16 gradients are
summed. The layout below mirrors MLASelfAttention: down projections, a
recomputable up-projection segment with a few extra nodes on the q branch,
the core-attention stand-in, then the gate.
"""

import types
import unittest
from unittest import mock

import paddle

from paddlefleet.tensor_parallel.random import RecomputeWithoutOutput
from paddlefleet.transformer import multi_latent_attention as mla_module
from paddlefleet.transformer.multi_latent_attention import _FixedOrderDownProj

from .test_mla_gate_attn import _build_mla

H, R, ROPE, T = 256, 128, 32, 2048


def _param(*shape):
    p = (paddle.randn(shape) * 0.02).astype("bfloat16")
    p.stop_gradient = False
    return p


def _bitwise_equal(a, b):
    # bf16 -> fp32 is exact, so this compares the bf16 bits.
    return bool(paddle.equal_all(a.astype("float32"), b.astype("float32")))


@unittest.skipUnless(
    paddle.is_compiled_with_cuda(), "bf16 accumulation order needs GPU"
)
class TestFixedOrderDownProj(unittest.TestCase):
    def setUp(self):
        self.prev_device = paddle.get_device()
        paddle.set_device("gpu")
        paddle.seed(2026)
        self.w = {
            "q_a": _param(H, R),
            "kv_a": _param(H, R + ROPE),
            "q_b": _param(R, H),
            "kv_b": _param(R, H - ROPE),
            "gate_h": _param(H, H),
            "gate_q": _param(R, H),
            "q": _param(H, H),
        }
        self.x0 = paddle.randn([T, H]).astype("bfloat16")
        self.dout = paddle.randn([T, H]).astype("bfloat16")
        w = self.w

        def down_proj(_self, hidden_states):
            q_compressed = hidden_states @ w["q_a"]
            kv_compressed, k_pos_emb = paddle.split(
                hidden_states @ w["kv_a"], [R, ROPE], axis=-1
            )
            return q_compressed, kv_compressed, k_pos_emb

        def down_proj_no_q_lora(_self, hidden_states):
            # Mirrors MultiLatentAttention._down_proj with q_lora_rank=None.
            kv_compressed, k_pos_emb = paddle.split(
                hidden_states @ w["kv_a"], [R, ROPE], axis=-1
            )
            return hidden_states, kv_compressed, k_pos_emb

        self.module = types.SimpleNamespace()
        self.module._down_proj = types.MethodType(down_proj, self.module)
        self.module_no_q_lora = types.SimpleNamespace()
        self.module_no_q_lora._down_proj = types.MethodType(
            down_proj_no_q_lora, self.module_no_q_lora
        )

    def tearDown(self):
        paddle.set_device(self.prev_device)

    def _up_proj(self, q_compressed, kv_compressed, k_pos_emb):
        # q_compressed is [T, R] with q_lora, [T, H] (hidden_states) without.
        q_w = self.w["q_b"] if q_compressed.shape[-1] == R else self.w["q"]
        q = q_compressed @ q_w
        for _ in range(3):
            q = (q * 1.0001).reshape([T, 8, -1]).reshape([T, H])
        key = paddle.concat([kv_compressed @ self.w["kv_b"], k_pos_emb], -1)
        return q.contiguous(), key.contiguous()

    def _run(self, fixed_order, recompute, gate, q_lora=True):
        module = self.module if q_lora else self.module_no_q_lora
        x = self.x0.clone()
        x.stop_gradient = False
        hidden_states = x * 1.0
        if fixed_order:
            q_compressed, kv_compressed, k_pos_emb = _FixedOrderDownProj.apply(
                module, hidden_states, paddle.is_grad_enabled()
            )
        else:
            q_compressed, kv_compressed, k_pos_emb = module._down_proj(
                hidden_states
            )
        span = None
        if recompute:
            span = RecomputeWithoutOutput()
            query, key = span.recompute(
                self._up_proj,
                q_compressed,
                kv_compressed,
                k_pos_emb,
                preserve_rng_state=False,
                share_grad_holder=True,
            )
        else:
            query, key = self._up_proj(q_compressed, kv_compressed, k_pos_emb)
        core_attn_out = query * key
        if span is not None:
            span.discard_output_and_register_recompute(core_attn_out)
        if gate == "hidden_states":
            gate_logits = hidden_states @ self.w["gate_h"]
        elif gate == "q_compressed":
            gate_logits = q_compressed @ self.w["gate_q"]
        else:
            gate_logits = None
        out = core_attn_out
        if gate_logits is not None:
            out = out * paddle.nn.functional.sigmoid(gate_logits)
        (out.astype("float32") * self.dout.astype("float32")).sum().backward()

        grads = {"hidden_states": x.grad.clone()}
        for name, p in self.w.items():
            if p.grad is not None:
                grads[name] = p.grad.clone()
            p.clear_gradient()
        return grads

    def _assert_all_equal(self, a, b, msg):
        self.assertEqual(set(a), set(b))
        for name in a:
            self.assertTrue(_bitwise_equal(a[name], b[name]), f"{msg}: {name}")

    def test_recompute_is_bitwise_for_every_gate_source(self):
        for gate in ("none", "q_compressed", "hidden_states"):
            with self.subTest(gate=gate):
                plain = self._run(True, False, gate)
                replayed = self._run(True, True, gate)
                self._assert_all_equal(plain, replayed, f"gate={gate}")

    def test_two_consumer_layouts_keep_the_old_numerics(self):
        # Without the gate on hidden_states there are only two down-projection
        # consumers, so wrapping them must not move a single bit.
        for gate in ("none", "q_compressed"):
            with self.subTest(gate=gate):
                self._assert_all_equal(
                    self._run(False, False, gate),
                    self._run(True, False, gate),
                    f"gate={gate}",
                )

    def test_eval_forward_builds_no_graph(self):
        with paddle.no_grad():
            outs = _FixedOrderDownProj.apply(self.module, self.x0, False)
        self.assertTrue(all(o.stop_gradient for o in outs))

    def test_no_q_lora_hidden_states_gate_is_bitwise(self):
        # q_lora_rank=None + gate on hidden_states: q_proj, kv_a_proj and
        # gate_proj all consume hidden_states.
        plain = self._run(True, False, "hidden_states", q_lora=False)
        replayed = self._run(True, True, "hidden_states", q_lora=False)
        self._assert_all_equal(plain, replayed, "no q_lora")

    def test_no_q_lora_returns_a_copy_of_its_input(self):
        hidden_states = self.x0.clone()
        hidden_states.stop_gradient = False
        q_compressed, _, _ = _FixedOrderDownProj.apply(
            self.module_no_q_lora, hidden_states, True
        )
        self.assertIsNot(q_compressed, hidden_states)
        self.assertTrue(_bitwise_equal(q_compressed, hidden_states))

    def test_no_q_lora_two_consumer_layout_keeps_the_old_numerics(self):
        self._assert_all_equal(
            self._run(False, False, "none", q_lora=False),
            self._run(True, False, "none", q_lora=False),
            "no q_lora, gate=none",
        )


@unittest.skipUnless(
    paddle.is_compiled_with_cuda(), "bf16 accumulation order needs GPU"
)
class TestMLASelfAttentionQKVRecompute(unittest.TestCase):
    """Real ``MLASelfAttention``: dispatch at the call site and the real
    ``_down_proj`` (layernorms, ``deferrable_linear``) inside the PyLayer."""

    def setUp(self):
        self.prev_device = paddle.get_device()
        paddle.set_device("gpu")
        # Without it the attention backward accumulates dq non-deterministically
        # at longer sequences, so two identical runs already differ.
        self.prev_flags = paddle.get_flags(["FLAGS_cudnn_deterministic"])
        paddle.set_flags({"FLAGS_cudnn_deterministic": True})

    def tearDown(self):
        paddle.set_flags(self.prev_flags)
        paddle.set_device(self.prev_device)

    def _build(self, recompute, **overrides):
        paddle.seed(2026)
        if recompute:
            overrides.update(
                recompute_granularity="selective",
                recompute_modules=["mla_qkv_recompute"],
            )
        attn = _build_mla(**overrides)
        attn.to(dtype="bfloat16")
        attn.train()
        return attn

    def _run(self, attn, x0, dout):
        x = x0.clone()
        x.stop_gradient = False
        out, _ = attn(x * 1.0, attention_mask=None)
        (out.astype("float32") * dout).sum().backward()
        grads = {"hidden_states": x.grad.clone()}
        for name, p in attn.named_parameters():
            if p.grad is not None:
                grads[name] = p.grad.clone()
            p.clear_gradient()
        return grads

    def _assert_recompute_is_bitwise(self, seq_len, seed, **overrides):
        plain = self._build(False, **overrides)
        replay = self._build(True, **overrides)
        self.assertFalse(plain.recompute_qkv_up_porj_and_rope)
        self.assertTrue(replay.recompute_qkv_up_porj_and_rope)
        replay.set_state_dict(plain.state_dict())
        paddle.seed(seed)
        x0 = paddle.randn([2, seq_len, 128]).astype("bfloat16")
        dout = paddle.randn([2, seq_len, 128])
        a, b = self._run(plain, x0, dout), self._run(replay, x0, dout)
        self.assertEqual(set(a), set(b))
        self.assertIn("gate_proj.linear.weight", a)
        for name in a:
            self.assertTrue(_bitwise_equal(a[name], b[name]), name)

    def test_q_lora_hidden_states_gate(self):
        # The plain path drifts by an ULP on hidden_states here.
        self._assert_recompute_is_bitwise(256, 2026, gated_attention=True)

    def test_no_q_lora_hidden_states_gate(self):
        # In this MLA graph the plain path happens to sum in the same order
        # with and without recompute; this guards the copy path, while
        # test_dispatch_condition guards that the layout takes it.
        self._assert_recompute_is_bitwise(
            256, 2026, gated_attention=True, q_lora_rank=None
        )

    def _dispatch_count(self, attn, stop_gradient=False):
        x = paddle.randn([2, 16, 128]).astype("bfloat16")
        x.stop_gradient = stop_gradient
        with mock.patch.object(
            mla_module._FixedOrderDownProj,
            "apply",
            wraps=mla_module._FixedOrderDownProj.apply,
        ) as spy:
            attn(x * 1.0, attention_mask=None)
        return spy.call_count

    def test_dispatch_condition(self):
        cases = [
            # (overrides, training, stop_gradient, expected calls)
            ({"gated_attention": True}, True, False, 1),
            ({"gated_attention": False}, True, False, 1),
            ({"gated_attention": True, "q_lora_rank": None}, True, False, 1),
            ({"gated_attention": False, "q_lora_rank": None}, True, False, 0),
            ({"gated_attention": True}, False, False, 0),
            ({"gated_attention": True}, True, True, 0),
        ]
        for overrides, training, stop_gradient, expected in cases:
            with self.subTest(
                overrides=overrides, training=training, sg=stop_gradient
            ):
                attn = self._build(False, **overrides)
                if not training:
                    attn.eval()
                self.assertEqual(
                    self._dispatch_count(attn, stop_gradient), expected
                )


class TestFixedOrderDownProjBackwardGuard(unittest.TestCase):
    """``backward`` early-outs; called directly with a stand-in ``ctx``."""

    def _ctx(self, hidden_stop_gradient, build_graph, saved=()):
        return types.SimpleNamespace(
            hidden_stop_gradient=hidden_stop_gradient,
            build_graph=build_graph,
            saved_tensor=lambda: saved,
        )

    def test_no_gradient_without_graph_or_for_frozen_input(self):
        grad = paddle.ones([2, 4])
        for sg, build_graph in ((True, True), (False, False)):
            with self.subTest(hidden_stop_gradient=sg, build_graph=build_graph):
                ctx = self._ctx(sg, build_graph)
                self.assertIsNone(
                    _FixedOrderDownProj.backward(ctx, grad, grad, grad)
                )

    def test_all_output_grads_none_skips_inner_backward(self):
        segment_input = paddle.randn([2, 4])
        segment_input.stop_gradient = False
        outputs = [segment_input * 2.0, segment_input * 3.0, segment_input]
        ctx = self._ctx(False, True, (segment_input, *outputs))
        with mock.patch.object(paddle.autograd, "backward") as inner:
            grad = _FixedOrderDownProj.backward(ctx, None, None, None)
        inner.assert_not_called()
        self.assertIsNone(grad)


class TestDownProjParallelBranches(unittest.TestCase):
    """TP-sharded / sequence-parallel branches of ``MLASelfAttention._down_proj``.

    The collectives are replaced with local stand-ins: a TP gather concatenates
    the shard with itself along the last dim, SP scatter / gather are identity.
    """

    Q_LORA, KV_LORA, ROPE, HIDDEN = 8, 6, 4, 16

    def _layer(self, q_out, kv_out, sequence_parallel):
        layer = types.SimpleNamespace(
            config=types.SimpleNamespace(sequence_parallel=sequence_parallel),
            q_lora_rank=self.Q_LORA,
            kv_lora_rank=self.KV_LORA,
            qk_rope_head_dim=self.ROPE,
            pg_collection=types.SimpleNamespace(tp="tp_group"),
            q_a_proj=paddle.nn.Linear(self.HIDDEN, q_out),
            kv_a_proj_with_mqa=paddle.nn.Linear(self.HIDDEN, kv_out),
            q_a_layernorm=lambda t: t,
            kv_a_layernorm=lambda t: t,
        )
        layer._down_proj = types.MethodType(
            mla_module.MLASelfAttention._down_proj, layer
        )
        return layer

    def _run(self, layer, tp_size):
        calls = {"tp_gather": 0, "sp_scatter": 0, "sp_gather": []}

        def tp_gather(t):
            calls["tp_gather"] += 1
            return paddle.concat([t, t], axis=-1)

        def sp_scatter(t):
            calls["sp_scatter"] += 1
            return t

        def sp_gather(t, group=None):
            calls["sp_gather"].append(group)
            return t

        with (
            mock.patch.object(
                mla_module,
                "deferrable_linear",
                lambda cfg, point, proj, x: (proj(x), None),
            ),
            mock.patch.object(
                mla_module,
                "gather_from_tensor_model_parallel_region",
                tp_gather,
            ),
            mock.patch.object(
                mla_module, "scatter_to_sequence_parallel_region", sp_scatter
            ),
            mock.patch.object(
                mla_module, "gather_from_sequence_parallel_region", sp_gather
            ),
            mock.patch.object(mla_module, "get_pg_size", lambda g: tp_size),
        ):
            outs = layer._down_proj(paddle.randn([2, 5, self.HIDDEN]))
        return outs, calls

    def test_column_parallel_outputs_are_gathered_and_scattered(self):
        layer = self._layer(
            self.Q_LORA // 2,
            (self.KV_LORA + self.ROPE) // 2,
            sequence_parallel=True,
        )
        (q, kv, k_pe), calls = self._run(layer, tp_size=2)
        self.assertEqual(q.shape, [2, 5, self.Q_LORA])
        self.assertEqual(kv.shape, [2, 5, self.KV_LORA])
        self.assertEqual(k_pe.shape, [2, 5, self.ROPE])
        self.assertEqual(calls["tp_gather"], 2)
        self.assertEqual(calls["sp_scatter"], 2)
        self.assertEqual(calls["sp_gather"], [])

    def test_column_parallel_without_sequence_parallel_skips_scatter(self):
        layer = self._layer(
            self.Q_LORA // 2,
            (self.KV_LORA + self.ROPE) // 2,
            sequence_parallel=False,
        )
        _, calls = self._run(layer, tp_size=2)
        self.assertEqual(calls["tp_gather"], 2)
        self.assertEqual(calls["sp_scatter"], 0)

    def test_replicated_kv_gathers_rope_under_tp_sequence_parallel(self):
        layer = self._layer(
            self.Q_LORA, self.KV_LORA + self.ROPE, sequence_parallel=True
        )
        (q, kv, k_pe), calls = self._run(layer, tp_size=2)
        self.assertEqual(q.shape, [2, 5, self.Q_LORA])
        self.assertEqual(kv.shape, [2, 5, self.KV_LORA])
        self.assertEqual(k_pe.shape, [2, 5, self.ROPE])
        self.assertEqual(calls["tp_gather"], 0)
        self.assertEqual(calls["sp_scatter"], 0)
        self.assertEqual(calls["sp_gather"], ["tp_group"])


if __name__ == "__main__":
    unittest.main()
