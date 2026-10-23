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
"""Multi-card (EP>1) tests for AllGatherTokenDispatcher and AllToAllTokenDispatcher.

Run with:
  python -m paddle.distributed.launch --gpus=0,1 \
      tests/multi_card_tests/moe/test_allgather_dispatcher_ep.py
"""

import random
import unittest

import numpy as np
import paddle
import paddle.distributed as dist
import paddlefleet_ops
from paddle.distributed import fleet

from paddlefleet.process_groups_config import ProcessGroupCollection
from paddlefleet.tensor_parallel.random import model_parallel_cuda_manual_seed
from paddlefleet.training.initialize import initialize_fleet

# ═══════════════════════════════════════════════════════════════════════════════
#  Shared setup — fleet initialised ONCE for the whole process
# ═══════════════════════════════════════════════════════════════════════════════

_fleet_initialised = False
_pg_collection = None


def _ensure_fleet():
    global _fleet_initialised, _pg_collection
    if _fleet_initialised:
        return _pg_collection
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": 1,
        "pp_degree": 1,
        "sharding_degree": 2,
        "sep_degree": 1,
        "cp_degree": 1,
        "ep_degree": 2,
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
    initialize_fleet(strategy=strategy)
    _pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    _fleet_initialised = True
    return _pg_collection


class _EPTestBase(unittest.TestCase):
    """Base class: initialises fleet once, provides ep_group / rank."""

    @classmethod
    def setUpClass(cls):
        cls.pg_collection = _ensure_fleet()

    def setUp(self):
        self.seed = 42
        random.seed(self.seed)
        np.random.seed(self.seed)
        paddle.seed(self.seed)
        paddle.manual_seed(self.seed)
        model_parallel_cuda_manual_seed(self.seed)
        self.ep_group = self.__class__.pg_collection.ep
        self.ep_size = self.ep_group.nranks
        self.rank = dist.get_rank(self.ep_group)
        self.num_experts = 4
        self.hidden_size = 64
        self.num_experts_per_tok = 2
        self.T_local = 4

    def _make_routing(self):
        """Create routing data. Same on all ranks (deterministic seed)."""
        T, K, E = self.T_local, self.num_experts_per_tok, self.num_experts
        topk_indices = paddle.to_tensor(
            [[i % E, (i + 1) % E] for i in range(T)], dtype="int32"
        )
        topk_weights = paddle.randn([T, K]).abs()
        topk_weights = topk_weights / topk_weights.sum(axis=1, keepdim=True)
        probs = paddle.zeros([T, E], dtype="float32")
        for i in range(T):
            probs[i, i % E] = float(topk_weights[i, 0])
            probs[i, (i + 1) % E] = float(topk_weights[i, 1])
        mask = (probs > 0).cast("float32")
        return topk_indices, topk_weights, probs, mask


# ═══════════════════════════════════════════════════════════════════════════════
#  ReduceScatterGroupOp
# ═══════════════════════════════════════════════════════════════════════════════


class TestReduceScatterGroupOpEP(_EPTestBase):
    def test_forward_scatters_and_sums(self):
        from paddlefleet.transformer.moe.moe_utils import ReduceScatterGroupOp

        T_local, H = 4, 8
        x = paddle.randn([T_local, H]) * (self.rank + 1)
        out = ReduceScatterGroupOp.apply(x, self.ep_group)
        self.assertEqual(out.shape[0], T_local // self.ep_size)
        self.assertEqual(out.shape[1], H)

    def test_backward_all_gathers_grad(self):
        from paddlefleet.transformer.moe.moe_utils import ReduceScatterGroupOp

        T_local, H = 4, 8
        x = paddle.randn([T_local, H])
        x.stop_gradient = False
        out = ReduceScatterGroupOp.apply(x, self.ep_group)
        loss = out.sum()
        loss.backward()
        self.assertEqual(x.grad.shape, [T_local, H])


# ═══════════════════════════════════════════════════════════════════════════════
#  _RouterAllGather
# ═══════════════════════════════════════════════════════════════════════════════


class TestRouterAllGatherEP(_EPTestBase):
    def test_forward_global_shape(self):
        from paddlefleet.transformer.moe.token_dispatcher import (
            _RouterAllGather,
        )

        T_local, K = 4, 2
        x = paddle.randn([T_local, K])
        out = _RouterAllGather.apply(x, self.ep_group)
        self.assertEqual(out.shape[0], T_local * self.ep_size)
        self.assertEqual(out.shape[1], K)

    def test_forward_identical_across_ranks(self):
        from paddlefleet.transformer.moe.token_dispatcher import (
            _RouterAllGather,
        )

        T_local, K = 4, 2
        x = paddle.randn([T_local, K]) * (self.rank + 1)
        out = _RouterAllGather.apply(x, self.ep_group)
        # AllGather output must be identical on every rank
        all_outs = [paddle.empty_like(out) for _ in range(self.ep_size)]
        dist.all_gather(all_outs, out, group=self.ep_group)
        for other in all_outs:
            np.testing.assert_allclose(out.numpy(), other.numpy(), rtol=1e-5)

    def test_backward_reduce_scatter(self):
        from paddlefleet.transformer.moe.token_dispatcher import (
            _RouterAllGather,
        )

        T_local, K = 4, 2
        x = paddle.randn([T_local, K])
        x.stop_gradient = False
        out = _RouterAllGather.apply(x, self.ep_group)
        loss = out.sum()
        loss.backward()
        self.assertEqual(x.grad.shape, [T_local, K])


# ═══════════════════════════════════════════════════════════════════════════════
#  _tokens_per_expert_histogram
# ═══════════════════════════════════════════════════════════════════════════════


class TestTokensPerExpertHistogramEP(_EPTestBase):
    def test_histogram_with_global_indices(self):
        from paddlefleet.transformer.moe.token_dispatcher import (
            _tokens_per_expert_histogram,
        )

        num_experts, T_local, K = 4, 3, 2
        if self.rank == 0:
            indices_local = paddle.to_tensor(
                [[0, 1], [2, 3], [0, -1]], dtype="int32"
            )
        else:
            indices_local = paddle.to_tensor(
                [[1, 2], [3, -1], [0, 1]], dtype="int32"
            )
        recv = [paddle.empty_like(indices_local) for _ in range(self.ep_size)]
        dist.all_gather(recv, indices_local, group=self.ep_group)
        indices_global = paddle.concat(recv, axis=0)

        counts = _tokens_per_expert_histogram(indices_global, num_experts)
        # R0: [0,1],[2,3],[0,-1] → 0:2,1:1,2:1,3:1
        # R1: [1,2],[3,-1],[0,1] → 0:1,1:2,2:1,3:1
        # Total: [3,3,2,2]
        np.testing.assert_array_equal(counts.numpy(), [3, 3, 2, 2])


# ═══════════════════════════════════════════════════════════════════════════════
#  AllGatherTokenDispatcher
# ═══════════════════════════════════════════════════════════════════════════════


@unittest.skipUnless(
    paddlefleet_ops.is_sonic_moe_available(),
    "Sonic-MoE not available",
)
class TestAllGatherDispatcherEP(_EPTestBase):
    def _make_dispatcher(self, fp8=False):
        from paddlefleet.transformer.moe.token_dispatcher import (
            AllGatherTokenDispatcher,
        )

        return AllGatherTokenDispatcher(
            moe_group=self.ep_group,
            expert_model_parallel_size=self.ep_size,
            num_experts=self.num_experts,
            fp8_dispatch=fp8,
            use_ue8m0=False,
        )

    def test_dispatch_preprocess_global_hidden(self):
        dispatcher = self._make_dispatcher()
        ti, tw, probs, mask = self._make_routing()
        x_local = paddle.randn([self.T_local, self.hidden_size])
        global_x = dispatcher.dispatch_preprocess(
            x_local,
            probs,
            mask,
            topk_weights=tw,
            topk_indices=ti,
        )
        self.assertEqual(
            global_x.shape,
            [self.T_local * self.ep_size, self.hidden_size],
        )

    def test_get_dispatched_routing_after_preprocess(self):
        dispatcher = self._make_dispatcher()
        ti, tw, probs, mask = self._make_routing()
        dispatcher.dispatch_preprocess(
            paddle.randn([self.T_local, self.hidden_size]),
            probs,
            mask,
            topk_weights=tw,
            topk_indices=ti,
        )
        indices, weights, counts = dispatcher.get_dispatched_routing()
        T_global = self.T_local * self.ep_size
        self.assertEqual(indices.shape, [T_global, self.num_experts_per_tok])
        self.assertEqual(weights.shape, [T_global, self.num_experts_per_tok])
        self.assertEqual(counts.shape, [self.num_experts])

    def test_token_dispatch_pass_through(self):
        dispatcher = self._make_dispatcher()
        ti, tw, probs, mask = self._make_routing()
        dispatcher.dispatch_preprocess(
            paddle.randn([self.T_local, self.hidden_size]),
            probs,
            mask,
            topk_weights=tw,
            topk_indices=ti,
        )
        x = paddle.randn([self.T_local * self.ep_size, self.hidden_size])
        result, handle = dispatcher.token_dispatch(x, using_sonic_moe=True)
        np.testing.assert_allclose(result.numpy(), x.numpy())
        self.assertIsNone(handle)

    def test_token_combine_reducescatter(self):
        dispatcher = self._make_dispatcher()
        T_global = self.T_local * self.ep_size
        x = paddle.randn([T_global, self.hidden_size])
        combined = dispatcher.token_combine(x, combine_overlap_handle=None)
        self.assertEqual(combined.shape[0], T_global // self.ep_size)

    def test_combine_postprocess_cached(self):
        dispatcher = self._make_dispatcher()
        T_global = self.T_local * self.ep_size
        x = paddle.randn([T_global, self.hidden_size])
        dispatcher.token_combine(x, combine_overlap_handle=None)
        result = dispatcher.combine_postprocess(x)
        self.assertEqual(result.shape[0], T_global // self.ep_size)

    def test_combine_preprocess_noop(self):
        dispatcher = self._make_dispatcher()
        x = paddle.randn([4, 8])
        result = dispatcher.combine_preprocess(x)
        np.testing.assert_allclose(result.numpy(), x.numpy())


# ═══════════════════════════════════════════════════════════════════════════════
#  AllToAllTokenDispatcher
# ═══════════════════════════════════════════════════════════════════════════════


class TestAllToAllDispatcherEP(_EPTestBase):
    def setUp(self):
        super().setUp()
        self.num_local_experts = self.num_experts // self.ep_size
        self.local_expert_indices = list(
            range(
                self.rank * self.num_local_experts,
                (self.rank + 1) * self.num_local_experts,
            )
        )

    def _make_dispatcher(self):
        from paddlefleet.transformer.moe.token_dispatcher import (
            AllToAllTokenDispatcher,
        )

        return AllToAllTokenDispatcher(
            moe_group=self.ep_group,
            expert_model_parallel_size=self.ep_size,
            num_experts_per_device=self.num_local_experts,
            local_expert_indices=self.local_expert_indices,
        )

    def test_dispatch_preprocess_sets_metadata(self):
        dispatcher = self._make_dispatcher()
        _, _, probs, mask = self._make_routing()
        dispatcher.dispatch_preprocess(
            paddle.randn([self.T_local, self.hidden_size]), probs, mask
        )
        self.assertIsNotNone(dispatcher.tokens_per_expert)
        self.assertEqual(
            dispatcher.tokens_per_expert.shape[0], self.num_local_experts
        )

    def test_token_dispatch_returns_tokens(self):
        dispatcher = self._make_dispatcher()
        _, _, probs, mask = self._make_routing()
        permuted = dispatcher.dispatch_preprocess(
            paddle.randn([self.T_local, self.hidden_size]), probs, mask
        )
        result, handle = dispatcher.token_dispatch(permuted)
        self.assertIsNotNone(result)
        self.assertIsNone(handle)

    def test_dispatch_postprocess_sorts_tokens(self):
        dispatcher = self._make_dispatcher()
        _, _, probs, mask = self._make_routing()
        permuted = dispatcher.dispatch_preprocess(
            paddle.randn([self.T_local, self.hidden_size]), probs, mask
        )
        dispatched, _ = dispatcher.token_dispatch(permuted)
        sorted_tokens, tpe = dispatcher.dispatch_postprocess(dispatched)
        self.assertIsNotNone(sorted_tokens)
        self.assertEqual(tpe.shape[0], self.num_local_experts)

    def test_get_dispatched_routing(self):
        dispatcher = self._make_dispatcher()
        _, _, probs, mask = self._make_routing()
        dispatcher.dispatch_preprocess(
            paddle.randn([self.T_local, self.hidden_size]), probs, mask
        )
        indices, probs_out, tpe = dispatcher.get_dispatched_routing()
        self.assertIsNone(indices)
        self.assertIsNone(probs_out)
        self.assertEqual(tpe.shape[0], self.num_local_experts)

    def test_combine_preprocess_preserves_shape(self):
        dispatcher = self._make_dispatcher()
        _, _, probs, mask = self._make_routing()
        permuted = dispatcher.dispatch_preprocess(
            paddle.randn([self.T_local, self.hidden_size]), probs, mask
        )
        dispatched, _ = dispatcher.token_dispatch(permuted)
        sorted_tokens, _ = dispatcher.dispatch_postprocess(dispatched)
        expert_out = paddle.randn_like(sorted_tokens)
        restored = dispatcher.combine_preprocess(expert_out)
        self.assertEqual(restored.shape, expert_out.shape)

    def test_full_round_trip_shape(self):
        dispatcher = self._make_dispatcher()
        _, _, probs, mask = self._make_routing()
        permuted = dispatcher.dispatch_preprocess(
            paddle.randn([self.T_local, self.hidden_size]), probs, mask
        )
        dispatched, _ = dispatcher.token_dispatch(permuted)
        sorted_tokens, _ = dispatcher.dispatch_postprocess(dispatched)
        # Identity expert output
        restored = dispatcher.combine_preprocess(sorted_tokens)
        combined = dispatcher.token_combine(restored)
        output = dispatcher.combine_postprocess(combined)
        self.assertEqual(output.shape[0], self.T_local)
        self.assertEqual(output.shape[1], self.hidden_size)


# ═══════════════════════════════════════════════════════════════════════════════
#  MoELayer.combine() routing
# ═══════════════════════════════════════════════════════════════════════════════


class TestMoELayerCombineEP(_EPTestBase):
    def test_combine_allgather_path(self):
        from unittest.mock import MagicMock

        from paddlefleet.transformer.moe.moe_layer import MoELayer

        layer = MagicMock()
        layer.moe_token_dispatcher_type = "allgather"
        layer.token_dispatcher = MagicMock()
        MoELayer.combine(
            layer, paddle.randn([4, 8]), combine_overlap_handle=None
        )
        layer.token_dispatcher.token_combine.assert_called_once()
        layer.token_dispatcher.combine_postprocess.assert_called_once()

    def test_combine_alltoall_path(self):
        from unittest.mock import MagicMock

        from paddlefleet.transformer.moe.moe_layer import MoELayer

        layer = MagicMock()
        layer.moe_token_dispatcher_type = "alltoall"
        layer.token_dispatcher = MagicMock()
        MoELayer.combine(layer, paddle.randn([4, 8]))
        layer.token_dispatcher.token_combine.assert_called_once()
        layer.token_dispatcher.combine_postprocess.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════════
#  AllGather routing: Triton mask/cast + histogram vs eager
# ═══════════════════════════════════════════════════════════════════════════════


def _triton_available():
    from paddlefleet.triton_ops.utils import is_triton_available

    return is_triton_available()


@unittest.skipUnless(_triton_available(), "Triton not available")
class TestAllGatherTritonRoutingEP(_EPTestBase):
    def test_triton_routing_matches_eager(self):
        # The fused kernel masks and casts the local routing pair before the
        # gathers; the eager path masks the gathered pair. Indices, weights,
        # counts and the router-weight grad must match exactly, padding lanes
        # included.
        from paddlefleet.transformer.moe.token_dispatcher import (
            AllGatherTokenDispatcher,
        )

        ti, tw, probs, mask = self._make_routing()
        ti[-1, 1] = -1  # a padding lane
        x = paddle.randn([self.T_local, self.hidden_size])
        T_global = self.T_local * self.ep_size
        K = self.num_experts_per_tok
        upstream = paddle.arange(T_global * K, dtype="float32").reshape(
            [T_global, K]
        ) + float(self.rank + 1)
        results = {}
        for use_triton in (False, True):
            disp = AllGatherTokenDispatcher(
                moe_group=self.ep_group,
                expert_model_parallel_size=self.ep_size,
                num_experts=self.num_experts,
            )
            disp._use_triton = use_triton
            w = (tw * (self.rank + 1)).detach()
            w.stop_gradient = False
            disp.dispatch_preprocess(
                x, probs, mask, topk_weights=w, topk_indices=ti
            )
            idx, gw, counts = disp.get_dispatched_routing()
            self.assertEqual(idx.dtype, paddle.int32)
            self.assertEqual(gw.dtype, probs.dtype)
            paddle.autograd.backward([gw], [upstream])
            results[use_triton] = (
                idx.numpy(),
                gw.numpy(),
                counts.numpy(),
                w.grad.numpy(),
            )
        for eager, fused in zip(results[False], results[True]):
            np.testing.assert_array_equal(fused, eager)
        idx, gw, counts, w_grad = results[True]
        # Every rank's last token has a padding lane: zero weight, zero grad.
        self.assertTrue((gw[idx < 0] == 0).all())
        self.assertEqual(w_grad[-1, 1], 0.0)
        self.assertEqual(int(counts.sum()), int((idx >= 0).sum()))


# ═══════════════════════════════════════════════════════════════════════════════
#  AllGather MoELayer: sync-free SonicMoE sizing
# ═══════════════════════════════════════════════════════════════════════════════


@unittest.skipUnless(
    paddlefleet_ops.is_sonic_moe_available(),
    "Sonic-MoE not available",
)
class TestAllGatherSyncFreeSizingEP(_EPTestBase):
    """The 'allgather' MoELayer asks the SonicMoE expert for sync-free sizing.
    Output and grads must match the exact count-sized path."""

    def _build_layer(self):
        import paddle.nn.functional as F
        from paddle.distributed.fleet.utils import mix_precision_utils

        from paddlefleet.models.gpt.gpt_layer_specs import (
            get_gpt_layer_local_spec,
        )
        from paddlefleet.transformer.moe.moe_layer import MoELayer
        from paddlefleet.transformer.transformer_config import (
            TransformerConfig,
        )

        config = TransformerConfig(
            hidden_size=256,
            num_attention_heads=4,
            n_routed_experts=8,
            use_cpu_initialization=False,
            num_experts_per_tok=2,
            tensor_model_parallel_size=1,
            expert_model_parallel_size=self.ep_size,
            sequence_parallel=False,
            bf16=True,
            params_dtype=paddle.bfloat16,
            moe_intermediate_size=256,
            gated_linear_unit=True,
            n_shared_experts=0,
            hidden_act=F.silu,
            moe_expert_fusion=True,
            bias_activation_fusion=True,
            moe_token_dispatcher_type="allgather",
            moe_use_fusion_node=True,
            using_sonic_moe=True,
        )
        spec = get_gpt_layer_local_spec(config, num_experts=8)
        layer = MoELayer(
            config,
            spec.sublayers_spec.mlp.extra_kwargs["sublayers"],
            self.__class__.pg_collection,
        )
        mix_precision_utils.MixPrecisionLayer(layer, dtype="bfloat16")
        for param in layer.parameters():
            if hasattr(param, "main_grad") and param.main_grad is None:
                param.main_grad = paddle.zeros_like(param, dtype="float32")
        return layer

    @staticmethod
    def _grads(layer):
        out = {}
        for name, param in layer.named_parameters():
            grad = getattr(param, "main_grad", None)
            if grad is None:
                grad = param.grad
            if grad is not None:
                out[name] = grad.astype("float32").numpy().copy()
                if getattr(param, "main_grad", None) is not None:
                    param.main_grad.zero_()
                if param.grad is not None:
                    param.grad.zero_()
        return out

    def test_sync_free_matches_exact_sizing(self):
        from unittest import mock

        from paddlefleet.transformer.moe import token_dispatcher as td

        layer = self._build_layer()
        self.assertIsInstance(
            layer.token_dispatcher, td.AllGatherTokenDispatcher
        )
        expert = layer.grouped_gemm_experts
        orig_forward = expert.forward
        seen = []

        def exact_forward(*args, sync_free_sizing=False, **kwargs):
            # The old call: exact sizing from the counts.
            seen.append(sync_free_sizing)
            kwargs["tokens_per_expert"] = td._tokens_per_expert_histogram(
                args[1], layer.num_experts
            )
            return orig_forward(*args, **kwargs)

        def spy_forward(*args, **kwargs):
            seen.append(kwargs.get("sync_free_sizing", False))
            if td._sync_free_metadata_ignores_counts():
                self.assertIsNone(kwargs["tokens_per_expert"])
            return orig_forward(*args, **kwargs)

        paddle.seed(1000 + self.rank)
        inp = paddle.randn([2, 64, 256], dtype="bfloat16")
        runs = {}
        for name, fwd in (("exact", exact_forward), ("sync_free", spy_forward)):
            hs = inp.detach().clone()
            hs.stop_gradient = False
            with mock.patch.object(expert, "forward", fwd):
                with paddle.amp.auto_cast(level="O2", dtype="bfloat16"):
                    out = layer(hs)[0]
                    loss = out.astype("float32").square().mean()
                loss.backward()
            runs[name] = (
                out.astype("float32").numpy(),
                hs.grad.astype("float32").numpy(),
                self._grads(layer),
            )
        # The layer asks for sync-free sizing; the reference drops it.
        self.assertEqual(seen, [True, True])
        exact, sync_free = runs["exact"], runs["sync_free"]
        np.testing.assert_allclose(sync_free[0], exact[0], rtol=0, atol=0)
        np.testing.assert_allclose(sync_free[1], exact[1], rtol=0, atol=0)
        self.assertEqual(set(sync_free[2]), set(exact[2]))
        for name in exact[2]:
            np.testing.assert_allclose(
                sync_free[2][name], exact[2][name], rtol=0, atol=0, err_msg=name
            )

    def test_sync_free_keeps_counts_when_metadata_reads_them(self):
        # Where the metadata builder still reads the counts (its eager
        # fallback), the layer keeps passing them alongside sync-free
        # sizing, and the result does not depend on whether it does.
        from unittest import mock

        from paddlefleet.transformer.moe import moe_layer

        layer = self._build_layer()
        expert = layer.grouped_gemm_experts
        orig_forward = expert.forward
        calls = []

        def spy_forward(*args, **kwargs):
            calls.append(
                (
                    kwargs.get("sync_free_sizing", False),
                    kwargs["tokens_per_expert"],
                )
            )
            return orig_forward(*args, **kwargs)

        paddle.seed(2000 + self.rank)
        inp = paddle.randn([2, 64, 256], dtype="bfloat16")
        outs = {}
        for ignores_counts in (False, True):
            hs = inp.detach().clone()
            with (
                mock.patch.object(
                    moe_layer,
                    "_sync_free_metadata_ignores_counts",
                    lambda v=ignores_counts: v,
                ),
                mock.patch.object(expert, "forward", spy_forward),
                paddle.no_grad(),
                paddle.amp.auto_cast(level="O2", dtype="bfloat16"),
            ):
                outs[ignores_counts] = layer(hs)[0].astype("float32").numpy()
        (sf_kept, counts_kept), (sf_dropped, counts_dropped) = calls
        self.assertTrue(sf_kept and sf_dropped)
        self.assertIsNone(counts_dropped)
        self.assertIsNotNone(counts_kept)
        idx = layer.token_dispatcher._global_topk_indices.numpy()
        np.testing.assert_array_equal(
            counts_kept.numpy(),
            np.bincount(idx[idx >= 0], minlength=layer.num_experts),
        )
        np.testing.assert_allclose(outs[False], outs[True], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
