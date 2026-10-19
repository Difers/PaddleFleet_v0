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

"""``MoECorrectionBiasAdjustCallback`` across pools with different expert counts.

A model can hold several MoE pools whose ``n_routed_experts`` differ -- HyperBody
routes its encoder over 64 experts and its decoder over 512 -- and every
``noaux_tc`` router in the model is collected by one ``model.apply``. Their
``expert_usage`` buffers are then shaped per pool, so a single
``paddle.stack`` over all of them raises; the callback groups by expert count
instead and writes each group's updates back in traversal order.
"""

import types
import unittest
from unittest.mock import patch

import paddle

from paddlefleet.trainer import trainer_callback


class _FakeGate:
    """A ``noaux_tc`` gate the callback picks up through ``model.apply``.

    ``expert_usage`` is int64 as the real ``StandardMoERouter`` registers it.
    """

    def __init__(self, usage):
        self.topk_method = "noaux_tc"
        num_experts = len(usage)
        self.e_score_correction_bias = paddle.zeros(
            [num_experts], dtype="float32"
        )
        self.expert_usage = paddle.to_tensor(usage, dtype="int64")
        self.weight = types.SimpleNamespace(stop_gradient=False)


class _FakeModel:
    def __init__(self, layers):
        self._layers = layers

    def apply(self, fn):
        for layer in self._layers:
            fn(layer)


_SINGLE_CARD = types.SimpleNamespace(nranks=1)
_HCG = types.SimpleNamespace(
    get_model_parallel_group=lambda: _SINGLE_CARD,
    get_data_parallel_group=lambda: _SINGLE_CARD,
    get_sharding_parallel_group=lambda: _SINGLE_CARD,
)


class _ParallelGroup:
    """A named stand-in for a Fleet process group; only ``nranks`` is read."""

    def __init__(self, name, nranks):
        self.name = name
        self.nranks = nranks


def _multi_card_hcg(mp=2, dp=2, sd=2):
    """An HCG whose mp/dp/sd groups report ``nranks > 1`` so the grouped
    collective path runs without launching real ranks."""
    mp_group = _ParallelGroup("mp", mp)
    dp_group = _ParallelGroup("dp", dp)
    sd_group = _ParallelGroup("sd", sd)
    hcg = types.SimpleNamespace(
        get_model_parallel_group=lambda: mp_group,
        get_data_parallel_group=lambda: dp_group,
        get_sharding_parallel_group=lambda: sd_group,
    )
    return hcg, (mp_group, dp_group, sd_group)


def _run(gates, lr=0.1, dsv4=False, hcg=_HCG, use_mp=False, has_hcg=True):
    """Drive ``on_optimizer_end`` on a single card, returning the all-reduce mock."""
    callback = trainer_callback.MoECorrectionBiasAdjustCallback(
        lr=lr, use_mp=use_mp
    )
    with (
        patch.object(
            trainer_callback,
            "use_dsv4_accuracy_compatible",
            return_value=dsv4,
        ),
        patch.object(trainer_callback, "PretrainedMoEGate", _FakeGate),
        patch.object(trainer_callback.dist, "all_reduce") as all_reduce,
        patch.object(trainer_callback, "fleet") as fleet,
        patch.object(trainer_callback, "get_lr_ratio_fn", return_value=None),
    ):
        if has_hcg:
            fleet.get_hybrid_communicate_group.return_value = hcg
        else:
            # ``hasattr(fleet, "_hcg")`` must be False to hit the fallback.
            del fleet._hcg
        callback.on_optimizer_end(
            types.SimpleNamespace(freeze_training=False),
            None,
            None,
            model=_FakeModel(gates),
            optimizer=None,
        )
    return all_reduce


def _expected_update(usage, lr):
    """The historical per-layer formula: ``sign(mean - usage) * lr``."""
    usage_tensor = paddle.to_tensor(usage, dtype="int64")
    mean = usage_tensor.mean(-1, keepdim=True)
    return (paddle.sign(mean - usage_tensor) * lr).astype("float32")


class TestHeterogeneousExpertCounts(unittest.TestCase):
    def test_pools_with_different_expert_counts(self):
        """Interleaved pools: a per-group update must land on its own layer, so
        an index-misaligned write-back shows up as a wrong bias here."""
        usages = [[4, 0, 0, 0], [0, 2], [0, 0, 0, 8]]
        gates = [_FakeGate(u) for u in usages]
        _run(gates)
        for gate, usage in zip(gates, usages):
            expected = _expected_update(usage, 0.1)
            self.assertTrue(
                paddle.allclose(gate.e_score_correction_bias, expected),
                f"bias {gate.e_score_correction_bias.tolist()} != "
                f"{expected.tolist()} for usage {usage}",
            )
            # The usage counters are cleared for the next window.
            self.assertEqual(gate.expert_usage.sum().item(), 0)

    def test_single_pool_keeps_the_historical_result(self):
        usages = [[4, 0, 0, 0], [0, 0, 0, 8]]
        gates = [_FakeGate(u) for u in usages]
        _run(gates)
        for gate, usage in zip(gates, usages):
            self.assertTrue(
                paddle.allclose(
                    gate.e_score_correction_bias, _expected_update(usage, 0.1)
                )
            )

    def test_one_all_reduce_per_group_on_the_dsv4_path(self):
        """The frozen DSv4 path only all-reduces and returns; with two expert
        counts that is one call per group, and no bias is touched."""
        gates = [_FakeGate([4, 0, 0, 0]), _FakeGate([0, 2])]
        all_reduce = _run(gates, dsv4=True)
        self.assertEqual(all_reduce.call_count, 2)
        for gate in gates:
            self.assertEqual(gate.e_score_correction_bias.sum().item(), 0.0)
            self.assertNotEqual(gate.expert_usage.sum().item(), 0)

    def test_frozen_gate_is_not_updated(self):
        gates = [_FakeGate([4, 0, 0, 0]), _FakeGate([0, 2])]
        gates[0].weight.stop_gradient = True
        _run(gates)
        self.assertEqual(gates[0].e_score_correction_bias.sum().item(), 0.0)
        self.assertTrue(
            paddle.allclose(
                gates[1].e_score_correction_bias, _expected_update([0, 2], 0.1)
            )
        )

    def test_grouped_all_reduce_targets_every_parallel_group(self):
        """Each expert-count group all-reduces once per parallel group
        (mp/dp/sd), in group order, over its own stacked tensor.

        A missed, misrouted, or reordered collective would hang training or
        sum a bias with the wrong pool's usage, yet the value-only cases above
        would still pass. The all-reduce stand-in only proves the local
        orchestration; the real process groups are not running here.
        """
        hcg, (mp_group, dp_group, sd_group) = _multi_card_hcg(mp=2, dp=2, sd=2)
        # Interleaved 4/2/4 pools -> two groups, (4,) seen before (2,).
        usages = [[4, 0, 0, 0], [0, 2], [0, 0, 0, 8]]
        gates = [_FakeGate(u) for u in usages]
        all_reduce = _run(gates, hcg=hcg, use_mp=True)

        # num_groups (2) x 3 parallel groups.
        self.assertEqual(all_reduce.call_count, 6)
        groups_per_call = [
            call.kwargs["group"] for call in all_reduce.call_args_list
        ]
        self.assertEqual(
            groups_per_call,
            [mp_group, dp_group, sd_group, mp_group, dp_group, sd_group],
        )
        # First three calls reduce the (4,)-pool stack, last three the
        # (2,)-pool stack: group order follows first appearance in model.apply.
        shapes_per_call = [
            tuple(call.args[0].shape) for call in all_reduce.call_args_list
        ]
        self.assertEqual(
            shapes_per_call,
            [(2, 4), (2, 4), (2, 4), (1, 2), (1, 2), (1, 2)],
        )
        # The per-group update still lands on each own layer and clears usage.
        for gate, usage in zip(gates, usages):
            self.assertTrue(
                paddle.allclose(
                    gate.e_score_correction_bias, _expected_update(usage, 0.1)
                )
            )
            self.assertEqual(gate.expert_usage.sum().item(), 0)

    def test_grouped_all_reduce_skips_single_rank_groups(self):
        """A parallel group with ``nranks == 1`` is skipped; only groups with
        more than one rank take part, independent of expert count."""
        hcg, (mp_group, dp_group, sd_group) = _multi_card_hcg(mp=1, dp=2, sd=1)
        gates = [_FakeGate([4, 0, 0, 0]), _FakeGate([0, 2])]
        all_reduce = _run(gates, hcg=hcg, use_mp=True)

        # Two groups x only the dp group participates.
        self.assertEqual(all_reduce.call_count, 2)
        groups_per_call = [
            call.kwargs["group"] for call in all_reduce.call_args_list
        ]
        self.assertEqual(groups_per_call, [dp_group, dp_group])

    def test_all_reduce_falls_back_to_global_group_without_hcg(self):
        """Without ``fleet._hcg`` the callback all-reduces each group's stack
        over the global group and returns before touching the bias."""
        gates = [_FakeGate([4, 0, 0, 0]), _FakeGate([0, 2])]
        all_reduce = _run(gates, has_hcg=False)

        self.assertEqual(all_reduce.call_count, 2)
        for call in all_reduce.call_args_list:
            self.assertNotIn("group", call.kwargs)
        for gate in gates:
            self.assertEqual(gate.e_score_correction_bias.sum().item(), 0.0)
            self.assertNotEqual(gate.expert_usage.sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
