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

"""``moe_dispatch`` refined recompute: config gating (no GPU, no dist init).

The RR runtime needs DeepEP and a live process group, so this covers the part
that decides *whether* it engages: ``MoeLayer._rr_dispatch_update``'s
preconditions, and the queue-pairing contract of
``DeepEPDispatchRefinedRecompute``.
"""

import types
import unittest

from paddlefleet.transformer.moe.moe_layer import MoELayer


def _fake_layer(modules, **over):
    """Bare stand-in carrying only what ``_rr_dispatch_update`` reads."""
    cfg = types.SimpleNamespace(
        recompute_modules=modules,
        recompute_granularity=over.pop("recompute_granularity", "full"),
        recompute_method=over.pop("recompute_method", "uniform"),
        virtual_pipeline_model_parallel_size=over.pop("vpp", None),
        recompute_num_layers=over.pop("recompute_num_layers", 1),
    )
    layer = types.SimpleNamespace(
        config=cfg,
        fp8=over.pop("fp8", None),
        fp8_dispatch=over.pop("fp8_dispatch", False),
        use_w4a8=over.pop("use_w4a8", False),
        moe_token_dispatcher_type=over.pop(
            "moe_token_dispatcher_type", "deepep"
        ),
        layer_number=over.pop("layer_number", 3),
        is_mtp_layer=False,
        use_rr_deepep_dispatch=False,
    )
    assert not over, f"unused kwargs {over}"
    return layer


def _run(layer, in_full=True, in_mlp=False):
    MoELayer._rr_dispatch_update(layer, in_full, in_mlp)
    return layer.use_rr_deepep_dispatch


class TestMoeDispatchRRGating(unittest.TestCase):
    def test_absent_is_inert(self):
        for modules in (None, [], ["moe_gate_up"], {"moe_combine": "all"}):
            with self.subTest(modules=modules):
                self.assertFalse(_run(_fake_layer(modules)))

    def test_list_mode_enables_every_layer(self):
        """List mode carries no layer info, so RR is on everywhere."""
        self.assertTrue(_run(_fake_layer(["moe_dispatch"])))
        self.assertTrue(
            _run(_fake_layer(["moe_dispatch"], layer_number=0)),
        )

    def test_fp8_raises_not_warns(self):
        """A misconfiguration must fail, not silently drop the optimization."""
        for over in (
            {"fp8_dispatch": True, "fp8": "e4m3"},
            {"use_w4a8": True},
        ):
            with self.subTest(**over), self.assertRaises(ValueError) as cm:
                _run(_fake_layer(["moe_dispatch"], **over))
            self.assertIn("moe_dispatch", str(cm.exception))

    def test_non_deepep_raises(self):
        with self.assertRaises(ValueError):
            _run(
                _fake_layer(
                    ["moe_dispatch"], moe_token_dispatcher_type="alltoall"
                )
            )

    def test_recompute_off_raises(self):
        with self.assertRaises(ValueError):
            _run(_fake_layer(["moe_dispatch"], recompute_granularity=None))

    def test_virtual_pipeline_raises(self):
        """VPP reorders replay against the first forward -> frames mispair."""
        with self.assertRaises(ValueError) as cm:
            _run(_fake_layer(["moe_dispatch"], vpp=2))
        self.assertIn("virtual pipeline", str(cm.exception))
        # vpp=1 is the normalized "no VPP" value and must pass.
        self.assertTrue(_run(_fake_layer(["moe_dispatch"], vpp=1)))

    def test_no_recompute_region_raises(self):
        with self.assertRaises(ValueError):
            _run(_fake_layer(["moe_dispatch"]), in_full=False, in_mlp=False)
        # mlp-only recompute is a valid host for the RR frame.
        self.assertTrue(
            _run(_fake_layer(["moe_dispatch"]), in_full=False, in_mlp=True)
        )

    def test_dict_mode_requires_first_n_for_a_count(self):
        with self.assertRaises(ValueError):
            _run(
                _fake_layer(
                    {"moe_dispatch": 2},
                    recompute_method="uniform",
                )
            )

    def test_dict_mode_selector_is_inverted(self):
        """RR runs on the layers the selector does NOT pick."""
        self.assertFalse(_run(_fake_layer({"moe_dispatch": "all"})))
        self.assertFalse(_run(_fake_layer({"moe_dispatch": None})))
        self.assertTrue(
            _run(_fake_layer({"moe_dispatch": [0, 1]}, layer_number=3))
        )
        self.assertFalse(
            _run(_fake_layer({"moe_dispatch": [0, 1]}, layer_number=1))
        )


class TestDispatchRRQueue(unittest.TestCase):
    def test_replay_without_a_cached_frame_raises(self):
        """An unpaired recompute pass must not read a stale/absent frame."""
        import paddle
        from paddle.base import framework

        from paddlefleet.transformer.moe.fused_a2a import (
            DeepEPDispatchRefinedRecompute,
        )

        rr = DeepEPDispatchRefinedRecompute()
        self.assertEqual(rr.pending, 0)
        # Grad on => the recompute branch, which has nothing queued.
        self.assertTrue(framework._dygraph_tracer()._has_grad)
        with self.assertRaises(RuntimeError) as cm:
            rr.forward(paddle.zeros([1, 1]), None, None, 1, None)
        self.assertIn("Queue is empty", str(cm.exception))


class TestDispatchRowBufferIsNotReleased(unittest.TestCase):
    """Static audit: the cached dispatch row buffer must never be released.

    ``moe_dispatch`` RR aliases ``hs_2d_dispatched`` (too big to copy) in its
    cache, so every ``hs_2d_dispatched._clear_to_zero_allocation()`` in
    ``MlpNode`` has to sit behind the ``rr_holds_dispatched`` guard. A missed one
    surfaces only at runtime, deep in the replay, as a bare
    "Tensor's dimension is out of bound ... memory's size is 0" -- hence this
    check rather than a comment.
    """

    def test_every_row_buffer_release_is_guarded(self):
        import ast
        import inspect

        from paddlefleet.transformer.moe import fusion_layer_utils

        tree = ast.parse(inspect.getsource(fusion_layer_utils))
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "MlpNode"
        )
        parent = {}
        for node in ast.walk(cls):
            for child in ast.iter_child_nodes(node):
                parent[child] = node

        def guarded(call):
            node = parent.get(call)
            while node is not None:
                if isinstance(node, ast.If) and "rr_holds_dispatched" in (
                    ast.unparse(node.test)
                ):
                    return True
                node = parent.get(node)
            return False

        unguarded = [
            ast.unparse(n.func.value)
            for n in ast.walk(cls)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "_clear_to_zero_allocation"
            and ast.unparse(n.func.value) == "hs_2d_dispatched"
            and not guarded(n)
        ]
        self.assertEqual(
            unguarded,
            [],
            "unguarded hs_2d_dispatched release(s) in MlpNode: wrap them in "
            "`if not self.rr_holds_dispatched:` or moe_dispatch RR will read "
            "freed memory during the recompute pass",
        )

    def test_small_metadata_is_cached_by_copy(self):
        """The metadata the first pass consumes destructively must be copied.

        ``MlpNode._prepare_forward`` casts ``dispatched_indices`` to int32 and
        then releases the int64 original, so the cache cannot alias it.
        """
        import inspect

        from paddlefleet.transformer.moe.fused_a2a import (
            DeepEPDispatchRefinedRecompute,
        )

        src = inspect.getsource(DeepEPDispatchRefinedRecompute._first_fwd)
        self.assertIn('cached_states["dispatched_indices"]', src)
        self.assertIn(".clone()", src)
        # ...and the row buffer must NOT be copied (that would double it).
        self.assertIn('"recv_x": recv_x.detach(),', src)


def _run_both(layer, in_full=True, in_mlp=False):
    """Drive the full entry point, which gates dispatch and combine in turn."""
    MoELayer.rr_recompute_update(layer, in_full, in_mlp)
    return layer.use_rr_deepep_dispatch, layer.use_rr_deepep_combine


def _fake_both_layer(modules, **over):
    layer = _fake_layer(modules, **over)
    layer.moe_shared_expert_overlap = True
    layer.use_rr_deepep_combine = False
    # ``rr_recompute_update`` delegates through ``self``, which a bare
    # SimpleNamespace does not resolve.
    layer._rr_dispatch_update = types.MethodType(
        MoELayer._rr_dispatch_update, layer
    )
    return layer


class TestMoeCombineRRGating(unittest.TestCase):
    """``moe_combine`` RR gates independently of ``moe_dispatch``."""

    def test_both_entries_enable_both(self):
        self.assertEqual(
            _run_both(_fake_both_layer(["moe_dispatch", "moe_combine"])),
            (True, True),
        )

    def test_each_entry_is_independent(self):
        self.assertEqual(
            _run_both(_fake_both_layer(["moe_combine"])), (False, True)
        )
        self.assertEqual(
            _run_both(_fake_both_layer(["moe_dispatch"])), (True, False)
        )

    def test_requires_shared_expert_overlap(self):
        layer = _fake_both_layer(["moe_combine"])
        layer.moe_shared_expert_overlap = False
        with self.assertRaises(ValueError) as cm:
            _run_both(layer)
        self.assertIn("moe_shared_expert_overlap", str(cm.exception))

    def test_non_deepep_raises(self):
        with self.assertRaises(ValueError):
            _run_both(
                _fake_both_layer(
                    ["moe_combine"], moe_token_dispatcher_type="alltoall"
                )
            )

    def test_recompute_off_raises(self):
        with self.assertRaises(ValueError):
            _run_both(
                _fake_both_layer(["moe_combine"], recompute_granularity=None)
            )

    def test_virtual_pipeline_raises(self):
        """Same mispairing hazard as dispatch RR: must fail, not compute wrong."""
        with self.assertRaises(ValueError) as cm:
            _run_both(_fake_both_layer(["moe_combine"], vpp=2))
        self.assertIn("virtual pipeline", str(cm.exception))
        # vpp=1 is the normalized "no VPP" value and must pass.
        self.assertEqual(
            _run_both(_fake_both_layer(["moe_combine"], vpp=1)), (False, True)
        )

    def test_no_recompute_region_raises(self):
        """RR caches a frame that only a replay pass consumes."""
        with self.assertRaises(ValueError):
            _run_both(
                _fake_both_layer(["moe_combine"]), in_full=False, in_mlp=False
            )

    def test_fp8_is_allowed_unlike_dispatch(self):
        """combine forward is always bf16, so fp8 is not a blocker here."""
        self.assertEqual(
            _run_both(
                _fake_both_layer(["moe_combine"], fp8="e4m3", fp8_dispatch=True)
            ),
            (False, True),
        )


if __name__ == "__main__":
    unittest.main()
