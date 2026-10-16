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


def _dispatcher(ep_group, num_experts, gpus_per_node=2, fp8_dispatch=False):
    from paddlefleet.transformer.moe import token_dispatcher as td

    with mock.patch.object(td, "_RING_GPUS_PER_NODE", gpus_per_node):
        return td.RingMoETokenDispatcher(
            ep_group,
            ep_group.nranks,
            num_experts=num_experts,
            fp8_dispatch=fp8_dispatch,
        )


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
    def test_topology_is_two_level(self):
        disp = _dispatcher(self.ep_group, self.num_experts)
        self.assertEqual((disp.G, disp.N), (2, self.ep_size // 2))
        self.assertIsNotNone(disp.intra_group)
        self.assertIsNotNone(disp.inter_group)
        self.assertEqual(disp.intra_group.nranks, 2)
        self.assertEqual(disp.inter_group.nranks, self.ep_size // 2)

    def test_forward_returns_own_rows_scaled_by_ep(self):
        """Intra RS sums G partials, the combine sums N -> EP*scale*x."""
        disp = _dispatcher(self.ep_group, self.num_experts)
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
        disp = _dispatcher(self.ep_group, self.num_experts)
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

    def test_rejects_inter_only_topology(self):
        # G==1 (one GPU per node group) is an inter-only ring; construction
        # must raise RuntimeError (survives ``python -O``), not run it.
        with self.assertRaisesRegex(RuntimeError, "two-level ring"):
            _dispatcher(self.ep_group, self.num_experts, gpus_per_node=1)

    def test_token_grad_is_routed_elementwise(self):
        # Stronger than the symmetric 2*EP constant: with expert_fn(t)=t*scale
        # the ring output is out[i]=scale*EP*x[i] elementwise, so a distinct
        # per-(rank,element) cotangent pins x.grad=scale*EP*g_out and exposes a
        # grad routed to the wrong round/rank/row.
        scale = 2.0
        disp = _dispatcher(self.ep_group, self.num_experts)
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
        disp = _dispatcher(self.ep_group, self.num_experts)
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
        disp = _dispatcher(self.ep_group, self.num_experts)
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
        disp = _dispatcher(self.ep_group, self.num_experts)
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
        disp = _dispatcher(self.ep_group, self.num_experts)
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

    def _check_repeated(self, gpus_per_node):
        if self.ep_size % gpus_per_node or self.ep_size // gpus_per_node < 2:
            self.skipTest(f"EP={self.ep_size} has no G={gpus_per_node} ring")
        disp = _dispatcher(
            self.ep_group, self.num_experts, gpus_per_node=gpus_per_node
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
