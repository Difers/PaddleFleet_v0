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
"""Two-level (G>1 AND N>1) coverage for RingMoETokenDispatcher.

Patching ``_RING_GPUS_PER_NODE`` below the world size splits the EP group
into an intra-node and an inter-node level even on one node, so the per-round
intra AllGather/ReduceScatter runs inside the inter-node rotation.

With an expert fn linear in the tokens the ring's output is analytic: every
rank's own rows come back scaled by EP (the intra ReduceScatter sums the G
intra partials, the combine all-to-all plus local sum the N node partials),
so forward and backward are checked element-wise against closed forms.

Run with:
  python -m paddle.distributed.launch \
      tests/multi_card_tests/moe/test_ring_dispatcher_two_level.py
"""

import unittest
from unittest import mock

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet

from paddlefleet.process_groups_config import ProcessGroupCollection
from paddlefleet.training.initialize import initialize_fleet
from paddlefleet.transformer.moe.token_dispatcher import (
    quantize_activation_blockscaled_fast as _sonic_fp8_quant,
)

# fp8 ring tests invoke SonicMoE's block-scaled quantizer; skip (not error) on a
# build/GPU where that kernel is unavailable.
_needs_sonicmoe_fp8 = unittest.skipUnless(
    _sonic_fp8_quant is not None,
    "Sonic-MoE fp8 block-scaled quantizer not available",
)

_pg_collection = None


def _ensure_fleet():
    """Bring up a world-sized EP-only topology (or reuse one already standing)."""
    global _pg_collection
    if _pg_collection is not None:
        return _pg_collection
    world_size = dist.get_world_size()
    if world_size < 4 or world_size % 2 != 0:
        raise RuntimeError(
            "two-level RingMoE tests require an even world size of at least 4, "
            f"got {world_size}"
        )
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": 1,
        "pp_degree": 1,
        "sharding_degree": world_size,
        "sep_degree": 1,
        "cp_degree": 1,
        "ep_degree": world_size,
        "moe_sharding_degree": 1,
        "order": [
            "sharding",
            "moe_sharding",
            "pp",
            "sep",
            "cp",
            "dp",
            "ep",
            "mp",
        ],
    }
    if getattr(fleet.fleet, "_hcg", None) is None:
        initialize_fleet(strategy=strategy)
    _pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    ep = _pg_collection.ep
    if ep is None or ep.nranks != world_size:
        # Reusing a foreign topology would silently test a different ring, so
        # fail loudly.
        raise RuntimeError(
            f"expected a {world_size}-rank EP group, got "
            f"{None if ep is None else ep.nranks}"
        )
    return _pg_collection


def _dispatcher(
    ep_group, num_experts, gpus_per_node=2, fp8_dispatch=False, use_zcomm=False
):
    from paddlefleet.transformer.moe import token_dispatcher as td

    with mock.patch.object(td, "_RING_GPUS_PER_NODE", gpus_per_node):
        return td.RingMoETokenDispatcher(
            ep_group,
            ep_group.nranks,
            num_experts=num_experts,
            fp8_dispatch=fp8_dispatch,
            use_zcomm=use_zcomm,
        )


def _zcomm_unavailable():
    from paddlefleet.transformer.moe.zcomm_comm import zcomm_unavailable_reason

    return zcomm_unavailable_reason()


def _scale_expert_fn(scale):
    """Linear in tokens, ignores routing -- output is analytically predictable."""

    def expert_fn(g_tok, g_idx, g_w, use_fp8, **kwargs):
        return g_tok * scale

    return expert_fn


def _weighted_expert_fn():
    """Uses ``g_w`` so the padded-lane mask and router gradient are observable."""

    def expert_fn(g_tok, g_idx, g_w, use_fp8, **kwargs):
        return g_tok * g_w.sum(axis=-1, keepdim=True)

    return expert_fn


class _TwoLevelBase(unittest.TestCase):
    """Fleet once per process, fresh seeds and a two-level EP group per test."""

    @classmethod
    def setUpClass(cls):
        cls.pg_collection = _ensure_fleet()

    def setUp(self):
        paddle.seed(1234)
        np.random.seed(1234)
        self.ep_group = self.__class__.pg_collection.ep
        self.ep_size = self.ep_group.nranks
        self.rank = dist.get_rank(self.ep_group)
        self.num_experts = 4
        self.d_latent = 8
        self.T_local = 4
        self.K = 2

    def _tokens(self, requires_grad=True):
        # Distinct per rank so a mis-routed rotation cannot pass by symmetry.
        x = paddle.full(
            [self.T_local, self.d_latent], float(self.rank + 1), dtype="float32"
        )
        x = x + paddle.arange(self.T_local, dtype="float32").reshape([-1, 1])
        x.stop_gradient = not requires_grad
        return x

    def _routing(self, pad_last=False):
        idx = paddle.to_tensor(
            [
                [i % self.num_experts, (i + 1) % self.num_experts]
                for i in range(self.T_local)
            ],
            dtype="int32",
        )
        if pad_last:
            # Mark the final token's lanes as padding; ring_forward must
            # zero their weights.
            idx[-1, :] = -1
        # Rank- and row-dependent: with a uniform weight, routing taken from
        # the wrong node would be indistinguishable from the right one.
        w = paddle.full(
            [self.T_local, self.K], 0.25 * (self.rank + 1), dtype="float32"
        ) + 0.01 * paddle.arange(self.T_local, dtype="float32").reshape([-1, 1])
        w.stop_gradient = False
        return idx, w

    def _distinct_upstream(self, shape):
        # Upstream cotangent distinct per (rank, element): makes any cross-round
        # or cross-rank mis-routing of the backward observable, unlike a uniform
        # ``out.sum()`` whose all-ones cotangent hides it.
        n = 1
        for d in shape:
            n *= int(d)
        g = paddle.arange(n, dtype="float32").reshape(shape) + float(
            self.rank * n + 1
        )
        g.stop_gradient = True
        return g


class TestTwoLevelRing(_TwoLevelBase):
    use_zcomm = False

    def test_topology_is_two_level(self):
        disp = _dispatcher(
            self.ep_group, self.num_experts, use_zcomm=self.use_zcomm
        )
        self.assertEqual((disp.G, disp.N), (2, self.ep_size // 2))
        self.assertIsNotNone(disp.intra_group)
        self.assertIsNotNone(disp.inter_group)
        self.assertEqual(disp.intra_group.nranks, 2)
        self.assertEqual(disp.inter_group.nranks, self.ep_size // 2)

    def test_forward_returns_own_rows_scaled_by_ep(self):
        """Intra RS sums G partials, the combine sums N -> EP*scale*x."""
        disp = _dispatcher(
            self.ep_group, self.num_experts, use_zcomm=self.use_zcomm
        )
        x = self._tokens()
        idx, w = self._routing()
        disp.pre_gate_token_ag(x)  # required before every ring_forward
        out = disp.ring_forward(x, w, idx, _scale_expert_fn(2.0), w.dtype)
        self.assertEqual(out.shape, [self.T_local, self.d_latent])
        np.testing.assert_allclose(
            out.numpy(),
            (x.detach() * 2.0 * self.ep_size).numpy(),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_backward_scales_token_grad_by_ep(self):
        disp = _dispatcher(
            self.ep_group, self.num_experts, use_zcomm=self.use_zcomm
        )
        x = self._tokens()
        idx, w = self._routing()
        disp.pre_gate_token_ag(x)
        out = disp.ring_forward(x, w, idx, _scale_expert_fn(2.0), w.dtype)
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        np.testing.assert_allclose(
            x.grad.numpy(),
            np.full(
                [self.T_local, self.d_latent],
                2.0 * self.ep_size,
                dtype="float32",
            ),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_join_token_hop_matches_plain(self):
        # join_token_hop leaves the backward's last token hop in flight and
        # has the join on the input add it. Grads must match the plain path,
        # with a router stand-in also reading the input, on the in-order
        # first forward and the remote-first second one.
        # The carrier and the gate input are two DIFFERENT tensors, as in
        # MoELayer's latent wiring: the carrier is a projection of the gate
        # input, so the deferred hop grad belongs to the carrier only and a
        # rev mis-added to the gate input (or dropped) shows up in the
        # projection's weight grad or the token grad.
        grads = {}
        for use_join in (False, True):
            paddle.seed(7)
            proj = paddle.nn.Linear(self.d_latent, self.d_latent)
            disp = _dispatcher(
                self.ep_group, self.num_experts, use_zcomm=self.use_zcomm
            )
            for step in range(2):
                idx, w0 = self._routing()
                x = self._tokens()
                carrier = proj(x)
                carrier, gate_in = (
                    disp.join_token_hop(carrier, x)
                    if use_join
                    else (carrier, x)
                )
                slot = disp._tok_hop_slot
                deferred = []
                if use_join:
                    # Runs after the ring backward, before the join's.
                    carrier.register_hook(
                        lambda _: deferred.append("hop" in slot)
                    )
                w = w0 * gate_in.mean(axis=-1, keepdim=True)
                disp.pre_gate_token_ag(carrier)
                out = disp.ring_forward(
                    carrier, w, idx, _weighted_expert_fn(), w.dtype
                )
                paddle.autograd.backward(
                    [out], [self._distinct_upstream(out.shape)]
                )
                if use_join:
                    self.assertEqual(deferred, [True])
                    self.assertEqual(slot, {})
                    self.assertIsNone(disp._tok_hop_slot)
                grads[use_join, step] = (
                    x.grad.numpy(),
                    w0.grad.numpy(),
                    proj.weight.grad.numpy(),
                )
                proj.clear_gradients()
        for step in range(2):
            for plain, joined in zip(grads[False, step], grads[True, step]):
                np.testing.assert_allclose(joined, plain, rtol=1e-5, atol=1e-5)

    def test_join_skips_router_input_without_grad(self):
        # A router input that takes no grad has no backward to overlap, so
        # the join must hand both inputs back and leave the hop to the ring.
        disp = _dispatcher(
            self.ep_group, self.num_experts, use_zcomm=self.use_zcomm
        )
        x = self._tokens()
        gate_in = self._tokens(requires_grad=False)
        carrier, gate_out = disp.join_token_hop(x, gate_in)
        self.assertIs(carrier, x)
        self.assertIs(gate_out, gate_in)
        self.assertIsNone(disp._tok_hop_slot)

    def test_join_clones_non_contiguous_input(self):
        # The join aliases contiguous inputs and clones strided ones. Without
        # a ring backward its slot stays empty, so its backward is identity.
        disp = _dispatcher(
            self.ep_group, self.num_experts, use_zcomm=self.use_zcomm
        )
        base = paddle.arange(
            self.T_local * self.d_latent, dtype="float32"
        ).reshape([self.d_latent, self.T_local])
        base.stop_gradient = False
        strided = base.transpose([1, 0])
        self.assertFalse(strided.is_contiguous())
        gate_in = self._tokens()
        carrier, gate_out = disp.join_token_hop(strided, gate_in)
        self.assertTrue(carrier.is_contiguous())
        np.testing.assert_array_equal(carrier.numpy(), strided.numpy())
        disp._tok_hop_slot = None
        g = self._distinct_upstream(carrier.shape)
        paddle.autograd.backward([carrier, gate_out], [g, g * 2])
        np.testing.assert_array_equal(base.grad.numpy(), g.numpy().T)
        np.testing.assert_array_equal(gate_in.grad.numpy(), (g * 2).numpy())

    def test_join_in_moe_layer_wiring_without_latent(self):
        # Without latent MoE the carrier and the router input start as the
        # same hidden state; the join aliases them separately and the ring
        # entry still has to hand the carrier alias to the gather.
        from types import SimpleNamespace

        from paddlefleet.transformer.moe import moe_layer

        grads = {}
        for use_join in (False, True):
            disp = _dispatcher(
                self.ep_group, self.num_experts, use_zcomm=self.use_zcomm
            )
            for step in range(2):
                hs = self._tokens()
                idx, w0 = self._routing()
                layer = SimpleNamespace(
                    expert_model_parallel_size=self.ep_size,
                    moe_token_dispatcher_type="ringmoe",
                    use_latent_moe=False,
                    token_dispatcher=disp,
                    _gate_input=hs if use_join else None,
                )
                carrier = {}
                pre_gate = disp.pre_gate_token_ag

                def _record(t, _pre_gate=pre_gate, _carrier=carrier):
                    _carrier["t"] = t
                    return _pre_gate(t)

                with mock.patch.object(disp, "pre_gate_token_ag", _record):
                    moe_layer.MoELayer._maybe_pre_allgather_overlap(layer, hs)
                self.assertIsNone(layer._latent_hidden)
                self.assertEqual(carrier["t"] is not hs, use_join)
                gate_in = layer._gate_input if use_join else hs
                w = w0 * gate_in.mean(axis=-1, keepdim=True)
                out = disp.ring_forward(
                    carrier["t"], w, idx, _weighted_expert_fn(), w.dtype
                )
                paddle.autograd.backward(
                    [out], [self._distinct_upstream(out.shape)]
                )
                grads[use_join, step] = (hs.grad.numpy(), w0.grad.numpy())
        for step in range(2):
            for plain, joined in zip(grads[False, step], grads[True, step]):
                np.testing.assert_allclose(joined, plain, rtol=1e-5, atol=1e-5)

    def test_manual_backward_root_and_partial_grads(self):
        # With every grad given, the segment backward starts from its scalar
        # root; with a None grad it falls back to backward over the outputs
        # that do get one.
        from paddlefleet.transformer.moe import token_dispatcher as td

        def f(a):
            return a * 2.0, a * 3.0

        for grads, expect in (((1.0, 1.0), 5.0), ((1.0, None), 2.0)):
            a = self._tokens()
            bwd, _ = td._manual_backward(f, False, a)
            g = [
                None if s is None else paddle.full(a.shape, s, a.dtype)
                for s in grads
            ]
            (a_grad,) = bwd(*g)
            np.testing.assert_allclose(
                a_grad.numpy(), np.full(a.shape, expect, "float32")
            )

    def test_join_in_moe_layer_wiring_matches_plain(self):
        # MoELayer's ring entry with latent MoE: the carrier is the latent
        # projection and the router input is the hidden state, two different
        # tensors; the router input is swapped for its alias through
        # `_gate_input`. Grads of the hidden state, the projection weight and
        # the router weights must match the path without a join.
        from types import SimpleNamespace

        from paddlefleet.transformer.moe import moe_layer

        grads = {}
        for use_join in (False, True):
            paddle.seed(7)
            proj = paddle.nn.Linear(2 * self.d_latent, self.d_latent)
            disp = _dispatcher(
                self.ep_group, self.num_experts, use_zcomm=self.use_zcomm
            )
            for step in range(2):
                hs = paddle.concat([self._tokens(False)] * 2, axis=-1)
                hs.stop_gradient = False
                idx, w0 = self._routing()
                layer = SimpleNamespace(
                    expert_model_parallel_size=self.ep_size,
                    moe_token_dispatcher_type="ringmoe",
                    use_latent_moe=True,
                    config=None,
                    fc1_latent_proj=proj,
                    token_dispatcher=disp,
                    _gate_input=hs if use_join else None,
                )
                with mock.patch.object(
                    moe_layer,
                    "deferrable_linear_bare",
                    lambda config, point, linear, x: linear(x),
                ):
                    moe_layer.MoELayer._maybe_pre_allgather_overlap(layer, hs)
                gate_in = layer._gate_input if use_join else hs
                self.assertEqual(disp._tok_hop_slot is not None, use_join)
                w = w0 * gate_in.mean(axis=-1, keepdim=True)
                out = disp.ring_forward(
                    layer._latent_hidden,
                    w,
                    idx,
                    _weighted_expert_fn(),
                    w.dtype,
                )
                paddle.autograd.backward(
                    [out], [self._distinct_upstream(out.shape)]
                )
                grads[use_join, step] = (
                    hs.grad.numpy(),
                    proj.weight.grad.numpy(),
                    w0.grad.numpy(),
                )
                proj.clear_gradients()
        for step in range(2):
            for plain, joined in zip(grads[False, step], grads[True, step]):
                np.testing.assert_allclose(joined, plain, rtol=1e-5, atol=1e-5)

    def test_join_with_shared_expert_matches_plain(self):
        # With a join the shared expert's backward moves to overlap the last
        # remote round's token-grad ReduceScatter. Grads of the tokens, router
        # weights, shared-expert input and its weight must match the plain
        # path.
        grads = {}
        for use_join in (False, True):
            paddle.seed(7)
            se = paddle.nn.Linear(self.d_latent, self.d_latent)
            disp = _dispatcher(
                self.ep_group, self.num_experts, use_zcomm=self.use_zcomm
            )
            for step in range(2):
                x = self._tokens()
                # Leading batch dim, like the MoE layer's residual.
                res = (self._tokens(requires_grad=False) * 0.5).unsqueeze(0)
                res.stop_gradient = False
                idx, w0 = self._routing()
                carrier, gate_in = (
                    disp.join_token_hop(x, x) if use_join else (x, x)
                )
                w = w0 * gate_in.mean(axis=-1, keepdim=True)
                handle = {"fn": se, "fn_args": (res,)}
                disp.pre_gate_token_ag(carrier)
                out = disp.ring_forward(
                    carrier,
                    w,
                    idx,
                    _weighted_expert_fn(),
                    w.dtype,
                    combine_overlap_handle=handle,
                )
                out = out + handle["fn_out"][0].squeeze(0)
                paddle.autograd.backward(
                    [out], [self._distinct_upstream(out.shape)]
                )
                grads[use_join, step] = (
                    x.grad.numpy(),
                    w0.grad.numpy(),
                    res.grad.numpy(),
                    se.weight.grad.numpy(),
                )
                se.clear_gradients()
        for step in range(2):
            for plain, joined in zip(grads[False, step], grads[True, step]):
                np.testing.assert_allclose(joined, plain, rtol=1e-5, atol=1e-5)

    def test_rejects_inter_only_topology(self):
        # G==1 (one GPU per node group) is an inter-only ring; construction
        # must raise RuntimeError (survives ``python -O``), not run it.
        with self.assertRaisesRegex(RuntimeError, "two-level ring"):
            _dispatcher(
                self.ep_group,
                self.num_experts,
                gpus_per_node=1,
                use_zcomm=self.use_zcomm,
            )

    def test_token_grad_is_routed_elementwise(self):
        # Stronger than the symmetric 2*EP constant: with expert_fn(t)=t*scale
        # the ring output is out[i]=scale*EP*x[i] elementwise, so a distinct
        # per-(rank,element) cotangent pins x.grad=scale*EP*g_out and exposes a
        # grad routed to the wrong round/rank/row.
        scale = 2.0
        disp = _dispatcher(
            self.ep_group, self.num_experts, use_zcomm=self.use_zcomm
        )
        x = self._tokens()
        idx, w = self._routing()
        disp.pre_gate_token_ag(x)
        out = disp.ring_forward(x, w, idx, _scale_expert_fn(scale), w.dtype)
        g_out = self._distinct_upstream(out.shape)
        paddle.autograd.backward([out], [g_out])
        self.assertIsNotNone(x.grad)
        np.testing.assert_allclose(
            x.grad.numpy(),
            (scale * self.ep_size) * g_out.numpy(),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_token_and_router_grad_match_analytic(self):
        # Weighted expert out[i]=EP*x[i]*sum_k w[i,k] with a distinct cotangent
        # pins BOTH input grads elementwise, numerically checking the routing
        # pair's reverse path (not just its magnitude):
        #   x.grad[i,j] = EP*s[i]*g_out[i,j]        (s[i]=sum_k w[i,k])
        #   w.grad[i,k] = EP*sum_j(g_out[i,j]*x[i,j])
        disp = _dispatcher(
            self.ep_group, self.num_experts, use_zcomm=self.use_zcomm
        )
        x = self._tokens()
        idx, w = self._routing()
        disp.pre_gate_token_ag(x)
        out = disp.ring_forward(x, w, idx, _weighted_expert_fn(), w.dtype)
        g_out = self._distinct_upstream(out.shape)
        paddle.autograd.backward([out], [g_out])
        s = w.sum(axis=-1, keepdim=True)
        np.testing.assert_allclose(
            x.grad.numpy(),
            (self.ep_size * s * g_out).numpy(),
            rtol=1e-5,
            atol=1e-5,
        )
        exp_w = self.ep_size * (g_out * x.detach()).sum(axis=-1, keepdim=True)
        np.testing.assert_allclose(
            w.grad.numpy(),
            np.broadcast_to(exp_w.numpy(), [self.T_local, self.K]),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_router_weights_are_applied_and_masked(self):
        """Padded lanes (idx<0) must contribute nothing.

        ring_forward zeroes their weights once on the local [T, K] pair
        (mask_cast_routing_w), before any gather.
        """
        disp = _dispatcher(
            self.ep_group, self.num_experts, use_zcomm=self.use_zcomm
        )
        x = self._tokens(requires_grad=False)
        idx, w = self._routing(pad_last=True)
        disp.pre_gate_token_ag(x)
        out = disp.ring_forward(x, w, idx, _weighted_expert_fn(), w.dtype)
        s = paddle.where(idx < 0, paddle.zeros_like(w), w).sum(
            axis=-1, keepdim=True
        )
        np.testing.assert_allclose(
            out.numpy(),
            (x * s * self.ep_size).numpy(),
            rtol=1e-5,
            atol=1e-5,
        )
        # The padded row must be exactly zero, not merely small.
        np.testing.assert_array_equal(
            out.numpy()[-1], np.zeros([self.d_latent], dtype="float32")
        )

    def test_router_grad_is_zero_on_padded_lanes(self):
        disp = _dispatcher(
            self.ep_group, self.num_experts, use_zcomm=self.use_zcomm
        )
        x = self._tokens(requires_grad=False)
        idx, w = self._routing(pad_last=True)
        disp.pre_gate_token_ag(x)
        out = disp.ring_forward(x, w, idx, _weighted_expert_fn(), w.dtype)
        out.sum().backward()
        self.assertIsNotNone(w.grad)
        np.testing.assert_array_equal(
            w.grad.numpy()[-1], np.zeros([self.K], dtype="float32")
        )
        # Non-padded lanes must actually receive gradient.
        self.assertGreater(float(np.abs(w.grad.numpy()[:-1]).sum()), 0.0)

    def test_repeated_forwards_are_stable(self):
        """A deferred reduce must not leak state across calls.

        Each round's ReduceScatter is issued async and drained before the
        combine; an undrained task would let the next call read a half-written
        buffer. Calls 2-3 also run remote-first, so they must match call 1.
        """
        disp = _dispatcher(
            self.ep_group, self.num_experts, use_zcomm=self.use_zcomm
        )
        idx, w = self._routing()
        first = None
        for _ in range(3):
            x = self._tokens(requires_grad=False)
            disp.pre_gate_token_ag(x)
            out = disp.ring_forward(x, w, idx, _scale_expert_fn(2.0), w.dtype)
            if first is None:
                first = out.numpy()
            else:
                np.testing.assert_allclose(
                    out.numpy(), first, rtol=1e-6, atol=1e-6
                )


class TestTwoLevelRingZcomm(TestTwoLevelRing):
    """The two-level ring checks again with ``use_zcomm=True``, in particular
    the token-hop join, under which the zcomm backward leaves its last
    token-grad ReduceScatter (and, for N >= 3, the add of the reverse hop
    into it) on the comm streams."""

    use_zcomm = True

    def setUp(self):
        reason = _zcomm_unavailable()
        if reason is not None:
            self.skipTest(f"zcomm unavailable: {reason}")
        super().setUp()


class _Fp8StraightThrough(paddle.autograd.PyLayer):
    """Stand-in for the fp8 expert GEMM: e4m3 in, bf16 out, bf16 grad back.

    The gather PyLayers hand the expert an e4m3 tensor and expect a bf16 dx back
    (their ReduceScatter would otherwise trip NCCL's "float8 not supported for
    reductions"). Reproducing that dtype contract lets the fp8 ring run without
    the real SonicMoE grouped GEMM.
    """

    @staticmethod
    def forward(ctx, tok_fp8):
        ctx.set_grad_in_dtype_consistent(False)
        ctx.set_materialize_grads(False)
        return tok_fp8.astype("bfloat16")

    @staticmethod
    def backward(ctx, grad):
        return grad


def _fp8_expert_fn():
    """Expert fn with the real ring call signature; consumes the gathered e4m3
    tokens straight-through so the fp8 ring is exercised end to end."""

    def expert_fn(
        g_tok,
        g_idx,
        g_w,
        use_fp8,
        tokens_per_expert=None,
        fp8_scale=None,
        recompute_moe_gate_up=False,
        fp8_combine_grad_handle=None,
        sync_free_sizing=False,
    ):
        return _Fp8StraightThrough.apply(g_tok)

    return expert_fn


class TestTwoLevelPreGateOverlap(_TwoLevelBase):
    """bf16 gate overlap: pre_gate_token_ag() then ring_forward() must match
    the closed form and drive the prefetch / gather-ahead / _order_after
    code."""

    def test_pre_gate_matches_closed_form(self):
        disp = _dispatcher(self.ep_group, self.num_experts)
        idx, w = self._routing()
        x = self._tokens()
        disp.pre_gate_token_ag(x)  # home-round gather issued before the gate
        out = disp.ring_forward(x, w, idx, _scale_expert_fn(2.0), w.dtype)
        np.testing.assert_allclose(
            out.numpy(),
            (x.detach() * 2.0 * self.ep_size).numpy(),
            rtol=1e-5,
            atol=1e-5,
        )
        out.sum().backward()
        self.assertIsNotNone(x.grad)

    def test_bf16_ring_requires_pre_gate(self):
        # As on fp8, ring_forward without pre_gate_token_ag must raise
        # RuntimeError (survives ``python -O``), not read a None handle.
        disp = _dispatcher(self.ep_group, self.num_experts)
        idx, w = self._routing()
        with self.assertRaisesRegex(RuntimeError, "pre-gate prefetch"):
            disp.ring_forward(
                self._tokens(), w, idx, _scale_expert_fn(2.0), w.dtype
            )

    def test_stale_pre_gate_is_drained_on_next_call(self):
        # A pre_gate not consumed by ring_forward must be waited out by the next
        # pre_gate_token_ag (drops the in-flight round-0 gather/hop safely).
        disp = _dispatcher(self.ep_group, self.num_experts)
        disp.pre_gate_token_ag(self._tokens(requires_grad=False))
        disp.pre_gate_token_ag(self._tokens(requires_grad=False))
        idx, w = self._routing()
        out = disp.ring_forward(
            self._tokens(), w, idx, _scale_expert_fn(2.0), w.dtype
        )
        self.assertEqual(out.shape, [self.T_local, self.d_latent])


class TestTwoLevelRemoteFirst(_TwoLevelBase):
    """Remote-first forward: from a dispatcher's second call on, the remote
    rounds run before the home round. Checks forward and backward of every
    call against the closed form, at G=2 (N=EP/2) and G=4 (N=EP/4, N=2 on
    8 GPUs)."""

    use_zcomm = False

    def _check_repeated(self, gpus_per_node):
        if self.ep_size % gpus_per_node or self.ep_size // gpus_per_node < 2:
            self.skipTest(f"EP={self.ep_size} has no G={gpus_per_node} ring")
        disp = _dispatcher(
            self.ep_group,
            self.num_experts,
            gpus_per_node=gpus_per_node,
            use_zcomm=self.use_zcomm,
        )
        self.assertEqual(
            (disp.G, disp.N), (gpus_per_node, self.ep_size // gpus_per_node)
        )
        for it in range(3):
            x = self._tokens()
            idx, w = self._routing()
            disp.pre_gate_token_ag(x)
            # The first call runs in order (token counts not checked yet).
            self.assertEqual(disp._gate_pack["remote"] is not None, it > 0)
            out = disp.ring_forward(x, w, idx, _weighted_expert_fn(), w.dtype)
            g_out = self._distinct_upstream(out.shape)
            paddle.autograd.backward([out], [g_out])
            s = w.detach().sum(axis=-1, keepdim=True)
            np.testing.assert_allclose(
                out.numpy(),
                (self.ep_size * s * x.detach()).numpy(),
                rtol=1e-5,
                atol=1e-5,
                err_msg=f"out, call {it}",
            )
            np.testing.assert_allclose(
                x.grad.numpy(),
                (self.ep_size * s * g_out).numpy(),
                rtol=1e-5,
                atol=1e-5,
                err_msg=f"x.grad, call {it}",
            )
            exp_w = self.ep_size * (g_out * x.detach()).sum(
                axis=-1, keepdim=True
            )
            np.testing.assert_allclose(
                w.grad.numpy(),
                np.broadcast_to(exp_w.numpy(), [self.T_local, self.K]),
                rtol=1e-5,
                atol=1e-5,
                err_msg=f"w.grad, call {it}",
            )

    def test_remote_first_matches_analytic_g2(self):
        self._check_repeated(2)

    def test_remote_first_matches_analytic_g4(self):
        self._check_repeated(4)


class TestTwoLevelZcommFallback(_TwoLevelBase):
    """``use_zcomm=True`` where the zcomm ops cannot run (old NCCL, ops not
    built): the dispatcher must fall back to NCCL with a single warning and
    give exactly the results of ``use_zcomm=False``. Runs on any build, the
    unavailable case is forced by patching the capability check."""

    def _fallback_dispatcher(self, **kwargs):
        from paddlefleet.transformer.moe import (
            token_dispatcher as td,
            zcomm_comm,
        )

        with (
            mock.patch.object(
                zcomm_comm,
                "zcomm_unavailable_reason",
                return_value="forced by test",
            ),
            mock.patch.object(td, "_ZCOMM_FALLBACK_WARNED", False),
            mock.patch.object(td.logger, "warning") as warn,
        ):
            disp = _dispatcher(
                self.ep_group, self.num_experts, use_zcomm=True, **kwargs
            )
            again = _dispatcher(
                self.ep_group, self.num_experts, use_zcomm=True, **kwargs
            )
        return disp, again, warn

    def test_falls_back_to_nccl_and_warns_once(self):
        disp, again, warn = self._fallback_dispatcher()
        self.assertFalse(disp.use_zcomm)
        self.assertFalse(again.use_zcomm)
        self.assertEqual(warn.call_count, 1)
        self.assertIn("forced by test", str(warn.call_args))

    def test_fallback_matches_nccl(self):
        # Forward and backward over repeated calls (in-order first call, then
        # remote-first) must equal the use_zcomm=False dispatcher exactly.
        results = []
        fallback, _, _ = self._fallback_dispatcher()
        plain = _dispatcher(self.ep_group, self.num_experts, use_zcomm=False)
        for disp in (fallback, plain):
            calls = []
            for _ in range(3):
                x = self._tokens()
                idx, w = self._routing()
                disp.pre_gate_token_ag(x)
                out = disp.ring_forward(
                    x, w, idx, _weighted_expert_fn(), w.dtype
                )
                paddle.autograd.backward(
                    [out], [self._distinct_upstream(out.shape)]
                )
                calls.append((out.numpy(), x.grad.numpy(), w.grad.numpy()))
            results.append(calls)
        for got, want in zip(*results):
            for a, b in zip(got, want):
                np.testing.assert_array_equal(a, b)


class TestTwoLevelZcomm(TestTwoLevelRemoteFirst):
    """The remote-first checks again with ``use_zcomm=True``: every ring
    collective on the zcomm ops (CE pull AllGather, CE ReduceScatter, GIN
    shifts / combine / AllGather) must give the same closed-form forward and
    backward as the NCCL path, over repeated calls (window reuse)."""

    use_zcomm = True

    def setUp(self):
        reason = _zcomm_unavailable()
        if reason is not None:
            self.skipTest(f"zcomm unavailable: {reason}")
        super().setUp()

    def test_enabled(self):
        disp = _dispatcher(self.ep_group, self.num_experts, use_zcomm=True)
        self.assertTrue(disp.use_zcomm)

    def test_fp8_ring_keeps_nccl(self):
        disp = _dispatcher(
            self.ep_group, self.num_experts, fp8_dispatch=True, use_zcomm=True
        )
        self.assertFalse(disp.use_zcomm)


@_needs_sonicmoe_fp8
class TestTwoLevelFp8Ring(_TwoLevelBase):
    """fp8 dispatch on the two-level ring: needs a 128-aligned hidden width and
    the pre-gate entry point. Expert GEMM is stubbed straight-through."""

    def setUp(self):
        super().setUp()
        self.H = 128  # fp8 block-scale tile alignment

    def _fp8_dispatcher(self):
        return _dispatcher(self.ep_group, self.num_experts, fp8_dispatch=True)

    def _tokens128(self, requires_grad=True):
        x = paddle.full(
            [self.T_local, self.H], float(self.rank + 1), dtype="bfloat16"
        )
        x.stop_gradient = not requires_grad
        return x

    def test_fp8_ring_requires_pre_gate(self):
        disp = self._fp8_dispatcher()
        idx, w = self._routing()
        with self.assertRaisesRegex(RuntimeError, "pre-gate prefetch"):
            disp.ring_forward(
                self._tokens128(), w, idx, _fp8_expert_fn(), w.dtype
            )

    def test_fp8_rejects_unaligned_hidden(self):
        disp = self._fp8_dispatcher()
        idx, w = self._routing()
        x = paddle.full([self.T_local, 8], 1.0, dtype="bfloat16")
        disp.pre_gate_token_ag(x)
        with self.assertRaisesRegex(ValueError, "multiple of 128"):
            disp.ring_forward(x, w, idx, _fp8_expert_fn(), w.dtype)

    def test_fp8_pre_gate_then_ring_forward(self):
        disp = self._fp8_dispatcher()
        idx, w = self._routing()
        x = self._tokens128()
        disp.pre_gate_token_ag(x)
        out = disp.ring_forward(x, w, idx, _fp8_expert_fn(), w.dtype)
        self.assertEqual(out.shape, [self.T_local, self.H])
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, [self.T_local, self.H])

    def test_fp8_grad_matches_straight_through(self):
        # fp8 straight-through expert: forward casts e4m3->bf16, backward passes
        # the cotangent through unchanged, so the token-grad path is identity
        # and x.grad=EP*g_out. A per-(rank,row) integer cotangent (exact in
        # bf16) gives a real numerical reference instead of a not-None check.
        disp = self._fp8_dispatcher()
        idx, w = self._routing()
        x = self._tokens128()
        disp.pre_gate_token_ag(x)
        out = disp.ring_forward(x, w, idx, _fp8_expert_fn(), w.dtype)
        row_id = paddle.arange(self.T_local, dtype="float32").reshape(
            [-1, 1]
        ) + float(self.rank * self.T_local + 1)
        g_out = paddle.tile(row_id, [1, self.H]).astype("bfloat16")
        g_out.stop_gradient = True
        paddle.autograd.backward([out], [g_out])
        self.assertIsNotNone(x.grad)
        np.testing.assert_allclose(
            x.grad.astype("float32").numpy(),
            (self.ep_size * g_out.astype("float32")).numpy(),
            rtol=2e-2,
            atol=2e-2,
        )

    def test_fp8_join_matches_plain(self):
        # fp8 ring with a join: the deferred token grad comes back through the
        # straight-through quantizer and is added in the join. Token and
        # router-weight grads must match the path without a join, on the
        # in-order first forward and the remote-first second one.
        def expert_fn(g_tok, g_idx, g_w, use_fp8, **kwargs):
            scale = g_w.sum(axis=-1, keepdim=True).astype("bfloat16")
            return _Fp8StraightThrough.apply(g_tok) * scale

        grads = {}
        for use_join in (False, True):
            disp = self._fp8_dispatcher()
            for step in range(2):
                x = self._tokens128()
                idx, w0 = self._routing()
                carrier, gate_in = (
                    disp.join_token_hop(x, x) if use_join else (x, x)
                )
                gate = gate_in.astype("float32").mean(axis=-1, keepdim=True)
                w = w0 * gate
                disp.pre_gate_token_ag(carrier)
                out = disp.ring_forward(carrier, w, idx, expert_fn, w.dtype)
                paddle.autograd.backward(
                    [out],
                    [self._distinct_upstream(out.shape).astype("bfloat16")],
                )
                grads[use_join, step] = (
                    x.grad.astype("float32").numpy(),
                    w0.grad.numpy(),
                )
        for step in range(2):
            for plain, joined in zip(grads[False, step], grads[True, step]):
                np.testing.assert_allclose(joined, plain, rtol=2e-2, atol=2e-2)

    def test_join_casts_deferred_grad_to_carrier_dtype(self):
        # The join adds the deferred hop grad in the carrier grad's dtype.
        # The fp8 quantizer only takes bf16 carriers, so a mismatched rev is
        # fed straight into the slot: a bf16 rev on an fp32 carrier, with a
        # hop handle that issued nothing.
        disp = self._fp8_dispatcher()
        x = self._tokens128(False).astype("float32")
        x.stop_gradient = False
        gate_in = self._tokens128()
        carrier, gate_out = disp.join_token_hop(x, gate_in)
        slot = disp._tok_hop_slot
        disp._tok_hop_slot = None
        rev = paddle.full([self.T_local * self.H], 0.5, dtype="bfloat16")
        slot.update(hop={}, rev=rev)
        paddle.autograd.backward(
            [carrier, gate_out],
            [paddle.ones_like(carrier), paddle.ones_like(gate_out)],
        )
        self.assertEqual(slot, {})
        self.assertEqual(x.grad.dtype, paddle.float32)
        np.testing.assert_array_equal(
            x.grad.numpy(), np.full(x.shape, 1.5, "float32")
        )

    def test_fp8_join_distinct_inputs_matches_plain(self):
        # Same fp8 join path, but the carrier and the router input are two
        # DIFFERENT tensors, as in MoELayer: the carrier is a projection of
        # the hidden state and the gate input is the hidden state itself. The
        # deferred hop carries only the carrier's grad, which the join adds
        # after the straight-through quantizer's backward; a rev added on the
        # wrong input (or dropped) shifts exactly one of these grads.
        def expert_fn(g_tok, g_idx, g_w, use_fp8, **kwargs):
            scale = g_w.sum(axis=-1, keepdim=True).astype("bfloat16")
            return _Fp8StraightThrough.apply(g_tok) * scale

        grads = {}
        for use_join in (False, True):
            paddle.seed(7)
            proj = paddle.nn.Linear(self.H, self.H)
            proj.to(dtype="bfloat16")
            disp = self._fp8_dispatcher()
            for step in range(2):
                hs = self._tokens128()
                idx, w0 = self._routing()
                carrier = proj(hs)
                carrier, gate_in = (
                    disp.join_token_hop(carrier, hs)
                    if use_join
                    else (carrier, hs)
                )
                gate = gate_in.astype("float32").mean(axis=-1, keepdim=True)
                w = w0 * gate
                disp.pre_gate_token_ag(carrier)
                out = disp.ring_forward(carrier, w, idx, expert_fn, w.dtype)
                paddle.autograd.backward(
                    [out],
                    [self._distinct_upstream(out.shape).astype("bfloat16")],
                )
                grads[use_join, step] = (
                    hs.grad.astype("float32").numpy(),
                    proj.weight.grad.astype("float32").numpy(),
                    w0.grad.numpy(),
                )
                proj.clear_gradients()
        for step in range(2):
            for plain, joined in zip(grads[False, step], grads[True, step]):
                np.testing.assert_allclose(joined, plain, rtol=2e-2, atol=2e-2)

    def test_fp8_remote_first_repeated_calls(self):
        # From the second call on the fp8 ring runs remote-first, hopping the
        # token's data and scale ahead of the router. Every call must give
        # the first call's output and the straight-through x.grad=EP*g_out.
        row_id = paddle.arange(self.T_local, dtype="float32").reshape(
            [-1, 1]
        ) + float(self.rank * self.T_local + 1)
        g_out = paddle.tile(row_id, [1, self.H]).astype("bfloat16")
        g_out.stop_gradient = True
        for gpus_per_node in (2, 4):
            if (
                self.ep_size % gpus_per_node
                or self.ep_size // gpus_per_node < 2
            ):
                continue
            with self.subTest(G=gpus_per_node):
                disp = _dispatcher(
                    self.ep_group,
                    self.num_experts,
                    gpus_per_node=gpus_per_node,
                    fp8_dispatch=True,
                )
                idx, w = self._routing()
                first = None
                for it in range(3):
                    x = self._tokens128()
                    disp.pre_gate_token_ag(x)
                    self.assertEqual(
                        disp._gate_pack["remote"] is not None, it > 0
                    )
                    out = disp.ring_forward(
                        x, w, idx, _fp8_expert_fn(), w.dtype
                    )
                    paddle.autograd.backward([out], [g_out])
                    out = out.astype("float32").numpy()
                    if first is None:
                        first = out
                    else:
                        np.testing.assert_array_equal(out, first)
                    np.testing.assert_allclose(
                        x.grad.astype("float32").numpy(),
                        (self.ep_size * g_out.astype("float32")).numpy(),
                        rtol=2e-2,
                        atol=2e-2,
                    )


if __name__ == "__main__":
    unittest.main()
