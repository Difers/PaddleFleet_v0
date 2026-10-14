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

"""optim_health: the cross-rank reduction of the per-parameter sums.

``tests/single_card_tests/internal_medicine/test_paddlefleet_optim_health.py``
drives real Muon steps and checks what is read out of them, but with one rank
every group sum in ``_flush_buffers`` is the identity. What is left unproven
there is exactly the part that reads the process groups: which group each
parameter is summed over, that a parameter's contribution lands in its own
bucket and nowhere else, that the per-expert vectors are all-gathered over the
expert-parallel group on the expert axis, and that one rank failing to capture
turns the whole step into "nobody logs it".

This test runs that reduction over real process groups. Every rank derives from
the live groups the slice of every parameter it owns -- the element range its
owner group (the sharding group, or the group its ``color`` names) holds, and
for a grouped-expert weight the experts its expert-parallel rank holds -- and
keeps only that slice of the weight and of the gradient. The whole tensors are
built from a fixed seed, so every rank knows all of them and the sum over the
right group is the whole tensor: the expected values are the global norms,
computed here without a collective, which is what the monitor has to report. A
reduction over the wrong group, a bucket swapped with another, or an expert
vector gathered on the wrong axis all change the answer.

Topology (8 cards): mp=2, pp=2, ep=2 -> sharding=2, moe_sharding=2, so every
axis the monitor reduces over (sharding, the colour group, expert, model and
pipeline parallel) is larger than one. The degrees follow the world size, so the
same file also runs on 4 or 2 cards with a subset of the axes covered. These
topologies all leave dense sharding at one, which gives the expert colour group
the same ranks as the sharding group, so a plan that ignored the colour would
still add up to the same numbers; ``ReductionPlanTest`` closes that gap on the
plan itself, where the colour can be a group sharding never is.

The steps are driven with gradients built here rather than by a backward pass:
the single-card file covers the reading of a real step, and only a gradient that
is known in full on every rank can be checked against its own shards.

Run with:
  python -m paddle.distributed.launch --gpus=0,1,2,3,4,5,6,7 \
      tests/multi_card_tests/internal_medicine/test_optim_health_reduction.py
"""

import math
import re
import unittest
import zlib

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet

from paddlefleet.internal_medicine.backends.paddlefleet import optim_monitor
from paddlefleet.internal_medicine.core.training_logs import training_logs
from paddlefleet.training.initialize import initialize_fleet

HIDDEN, VOCAB, EXPERTS = 8, 16, 4
LR = 0.02
WD = 0.1
PREFIX = "optim_health"
EXPERT_PARAM = "decoder.layers.0.mlp.experts"

_fleet_state = {"done": False, "error": None}


def _nranks(group):
    return int(getattr(group, "nranks", 1) or 1) if group is not None else 1


def _topology(world):
    """Parallel degrees for this world size.

    ``mp=2, pp=2, ep=2`` needs 8 cards; 4 cards drop the pipeline axis and 2
    cards keep only the expert one. ``sharding`` and ``moe_sharding`` follow
    from these (``world / (mp * pp)`` and ``world / (pp * ep)``); an axis left
    at one is simply not part of the reduction plan.
    """
    if world >= 8:
        return 2, 2, 2
    if world >= 4:
        return 2, 1, 2
    return 1, 1, 2


def _ensure_fleet():
    """Initialise fleet once per process; re-raise the original failure after."""
    if _fleet_state["error"] is not None:
        raise _fleet_state["error"]
    if not _fleet_state["done"]:
        world = dist.get_world_size()
        mp_degree, pp_degree, ep_degree = _topology(world)
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "pp_degree": pp_degree,
            "mp_degree": mp_degree,
            # Sharding is whatever the other axes leave, and the degrees are
            # handed to Fleet consistently: it checks that they multiply out.
            "sharding_degree": world // (mp_degree * pp_degree),
            "sep_degree": 1,
            "cp_degree": 1,
            "ep_degree": ep_degree,
            "moe_sharding_degree": world // (pp_degree * ep_degree),
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
        try:
            initialize_fleet(strategy=strategy)
        except Exception as exc:
            _fleet_state["error"] = exc
            raise
        _fleet_state["done"] = True
    return fleet.get_hybrid_communicate_group()


# ----------------------------------------------------------------------
# Toy model: the shapes the single-card test drives, plus the attributes
# the real MoE and TP layers put on their parameters.
# ----------------------------------------------------------------------


class _ToyAttn(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.q_proj = paddle.nn.Linear(HIDDEN, HIDDEN, bias_attr=False)


class _ToyNorm(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.weight = self.create_parameter(
            [HIDDEN], default_initializer=paddle.nn.initializer.Constant(1.0)
        )


class _ToyMoE(paddle.nn.Layer):
    def __init__(self, local_experts):
        super().__init__()
        self.gate = paddle.nn.Linear(HIDDEN, EXPERTS, bias_attr=False)
        self.experts = self.create_parameter(
            [local_experts, HIDDEN, HIDDEN],
            default_initializer=paddle.nn.initializer.Normal(std=0.2),
        )


class _ToyLayer(paddle.nn.Layer):
    def __init__(self, idx, local_experts):
        super().__init__()
        # 1-based, the convention ``resolve_layer_idx`` reads back as ``idx``.
        self.layer_number = idx + 1
        self.input_layernorm = _ToyNorm()
        self.self_attn = _ToyAttn()
        self.mlp = _ToyMoE(local_experts)

    def forward(self, x):
        h = x * self.input_layernorm.weight
        h = h + self.self_attn.q_proj(h)
        scores = paddle.nn.functional.softmax(self.mlp.gate(h), axis=-1)
        experts = self.mlp.experts.astype("float32")
        return h + paddle.einsum("te,ehk,th->tk", scores, experts, h)


class _ToyDecoder(paddle.nn.Layer):
    def __init__(self, layers):
        super().__init__()
        self.layers = paddle.nn.LayerList(layers)


class _ToyModel(paddle.nn.Layer):
    """Embedding + 2 layers + LM head, laid out like PaddleFleet's GPTModel."""

    def __init__(self, local_experts):
        super().__init__()
        self.embed_tokens = paddle.nn.Embedding(VOCAB, HIDDEN)
        self.decoder = _ToyDecoder(
            [_ToyLayer(i, local_experts) for i in range(2)]
        )
        self.lm_head = paddle.nn.Linear(HIDDEN, VOCAB, bias_attr=False)

    def forward(self, ids):
        x = self.embed_tokens(ids)
        for layer in self.decoder.layers:
            x = layer(x)
        return self.lm_head(x)


def _expert_colour(hcg):
    """The group the grouped-expert weights are coloured with.

    What the real MoE layers put in the colour: ``moe_grad_group``, the expert
    data-parallel group, which holds the replicas of the same experts and is
    therefore orthogonal to the expert-parallel group the experts are split
    over. Colouring with any other axis would make the two cuts overlap -- the
    model-parallel group, for one, is the same set of ranks as the expert group
    under these topologies -- and the shards would no longer tile the tensor.

    Whenever dense sharding is one this group has the same ranks as the
    sharding group, so the numbers alone cannot say whether the monitor read
    the colour or just fell back to sharding; ``ReductionPlanTest`` covers that
    on a parameter coloured with a group that is deliberately not sharding's.
    """
    return hcg.get_moe_sharding_parallel_group()


def _odd_colour(hcg):
    """Some group other than sharding, or ``None`` if the world has none.

    Only for the plan test: a colour the reduction plan cannot have arrived at
    by defaulting to the sharding group.
    """
    sharding = sorted(_group_ranks(hcg.get_sharding_parallel_group()))
    for group in (
        hcg.get_model_parallel_group(),
        hcg.get_pipe_parallel_group(),
        hcg.get_expert_parallel_group(),
    ):
        if _nranks(group) > 1 and sorted(_group_ranks(group)) != sharding:
            return group
    return None


def _group_ranks(group):
    return list(getattr(group, "ranks", []) or [])


def _annotate(model, hcg, colour=None):
    """The attributes the real MoE / TP layers set on their parameters.

    Grouped experts carry ``no_sync`` (their gradient is not synced by DDP) and
    the ``moe_expert`` colour, whose group is the expert data-parallel group; a
    tensor-parallel weight carries ``is_distributed`` and no colour.
    """
    colour = colour if colour is not None else _expert_colour(hcg)
    for name, param in model.named_parameters():
        if name.endswith("experts"):
            param.no_sync = True
            param.is_distributed = True
            param.color = {"color": "moe_expert", "group": colour}
        elif name.endswith("q_proj.weight"):
            param.is_distributed = True


def _kind(name):
    if "lm_head" in name:
        return "adamh"
    if "q_proj" in name or "experts" in name:
        return "muonh"
    return "adamw"


def _optimizer(model):
    from paddle.optimizer.muon import Muon, MuonParamInfo

    info = {}
    for name, param in model.named_parameters():
        kind = _kind(name)
        info[param.name] = MuonParamInfo(
            use_muon=kind == "muonh", use_hyperball=kind != "adamw"
        )
    return Muon(
        learning_rate=LR,
        parameters=model.parameters(),
        weight_decay=WD,
        muon_param_info_map=info,
        muon_version=3,
        ns_coeff_type="quintic",
        ns_matmul_dtype=paddle.float32,
        adam_epsilon=1e-15,
    )


def _stage_of(name, pp_degree):
    """Which pipeline stage holds a parameter: the layers are dealt out."""
    if pp_degree <= 1:
        return 0
    match = re.search(r"decoder\.layers\.(\d+)\.", name)
    if match:
        return int(match.group(1)) % pp_degree
    return 0 if "embed" in name else pp_degree - 1


def _owner_group(param, hcg):
    """The group a parameter is summed over: its colour's, else sharding.

    The rule ``MuonShardingOptimizer`` places whole tensors by, and the one
    ``optim_monitor`` reads off the parameter.
    """
    color = getattr(param, "color", -1)
    if isinstance(color, dict):
        return color.get("group")
    return hcg.get_sharding_parallel_group()


def _extra_group(param, hcg):
    """The second axis a parameter is split over, if any."""
    if getattr(param, "no_sync", False):
        return hcg.get_expert_parallel_group()
    if getattr(param, "is_distributed", False):
        return hcg.get_model_parallel_group()
    return None


def _position(group):
    """This rank's place in ``group`` and how many ranks share the tensor."""
    if _nranks(group) <= 1:
        return 0, 1
    return dist.get_rank(group), group.nranks


def _flat_slice(full, index, count):
    """``full`` flattened and cut to this rank's range, zero elsewhere."""
    flat = full.reshape([-1])
    lo = index * flat.numel() // count
    hi = (index + 1) * flat.numel() // count
    parts = [
        paddle.zeros([lo], dtype=full.dtype),
        flat[lo:hi],
        paddle.zeros([flat.numel() - hi], dtype=full.dtype),
    ]
    return paddle.concat([part for part in parts if part.numel()]).reshape(
        full.shape
    )


class _Shards:
    """Where every parameter lives on the ranks, read off the live groups.

    A parameter is cut element-wise over its owner group; a grouped-expert
    weight is additionally cut over the expert-parallel group, which holds a
    different set of experts rather than a different part of the same one. The
    two cuts are orthogonal, so summing a parameter's shards over both groups
    covers it exactly once.
    """

    def __init__(self, model, hcg, pp_degree):
        self.rank_stage = hcg.get_stage_id() if pp_degree > 1 else 0
        self.element = {}
        self.experts = {}
        self.stage = {}
        self.shape = {}
        # The whole tensor and this rank's parameter differ in shape for a
        # grouped-expert weight, and a stage that does not own the parameter
        # still has to be handed something of its own shape to set.
        self.local_shape = {}
        # The monitor keys its slots by ``Parameter.name``, which Paddle builds
        # from the layer class and a counter rather than from the attribute
        # path, so the two have to be carried side by side.
        self.param_name = {}
        for name, param in model.named_parameters():
            self.stage[name] = _stage_of(name, pp_degree)
            self.shape[name] = list(param.shape)
            self.local_shape[name] = list(param.shape)
            self.param_name[name] = param.name
            if getattr(param, "no_sync", False):
                # The owner group cuts each expert, the expert group decides
                # which experts this rank holds; the local tensor is its own.
                self.element[name] = _position(_owner_group(param, hcg))
                expert_rank, expert_count = _position(_extra_group(param, hcg))
                local = param.shape[0]
                self.experts[name] = (
                    expert_rank * local,
                    (expert_rank + 1) * local,
                )
                self.shape[name] = [
                    local * expert_count,
                    *list(param.shape[1:]),
                ]
                continue
            index, count = 0, 1
            for group in (_owner_group(param, hcg), _extra_group(param, hcg)):
                rank, nranks = _position(group)
                index, count = index * nranks + rank, count * nranks
            self.element[name] = (index, count)

    def owned(self, name):
        """Whether this rank's pipeline stage holds the parameter at all."""
        return self.stage[name] == self.rank_stage

    def local(self, name, full):
        """``full`` with everything this rank does not own zeroed out."""
        if not self.owned(name):
            return paddle.zeros(self.local_shape[name], dtype=full.dtype)
        index, count = self.element[name]
        if name not in self.experts:
            return _flat_slice(full, index, count)
        first, last = self.experts[name]
        return paddle.stack(
            [
                _flat_slice(full[expert], index, count)
                for expert in range(first, last)
            ]
        )

    def whole(self, seed_name, name):
        """The tensor every rank knows, of which the parameter is one shard.

        Seeded by ``seed_name`` through a stable hash: ``hash`` is salted per
        process, so it would hand different ranks different tensors.
        """
        seed = 1000 + zlib.crc32(seed_name.encode()) % 100000
        return paddle.to_tensor(
            np.random.RandomState(seed)
            .randn(*self.shape[name])
            .astype("float32")
        )


def _sq(tensor):
    """Sum of squares in float64, so the reference is not the thing measured."""
    return float((tensor.astype("float64") ** 2).sum())


def _norm(tensor):
    return math.sqrt(_sq(tensor))


def _median(values):
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


class _Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hcg = _ensure_fleet()
        cls.pp_degree = _nranks(cls.hcg.get_pipe_parallel_group())
        cls.ep_degree = _nranks(cls.hcg.get_expert_parallel_group())

    def setUp(self):
        training_logs.reset()
        paddle.seed(1234)
        self.model = _ToyModel(EXPERTS // self.ep_degree)
        _annotate(self.model, self.hcg)
        self.optimizer = _optimizer(self.model)
        self.monitor = optim_monitor.PaddleOptimHealthMonitor()
        self.monitor.register_hooks(self.model)
        self.assertTrue(self.monitor.attach_optimizer(self.optimizer))
        self.shards = _Shards(self.model, self.hcg, self.pp_degree)

    def tearDown(self):
        training_logs.reset()

    def _load(self):
        """Build the whole tensors and cut them down to this rank's shards."""
        weights, grads = {}, {}
        for name, param in self.model.named_parameters():
            weights[name] = self.shards.whole(name, name)
            grads[name] = 0.1 * self.shards.whole(f"{name}.grad", name)
            param.set_value(self.shards.local(name, weights[name]))
            param.grad = self.shards.local(name, grads[name])
        return weights, grads

    def _step(self, silent_rank=None):
        """One optimizer step; returns the whole weights, grads and the log.

        ``silent_rank`` drops that rank's captured step, which is what
        ``_apply_optimize`` leaves behind when the capture raised.
        """
        training_logs.reset()
        weights, grads = self._load()
        self.optimizer.step()
        self.optimizer.clear_grad()
        if silent_rank is not None and dist.get_rank() == silent_rank:
            self.monitor._pending = None
        self.monitor.step()
        return weights, grads, training_logs.get_latest(prefix=PREFIX)

    def _key(self, name, metric):
        slot = next(
            s
            for s in self.monitor._slots
            if s.param.name == self.shards.param_name[name]
        )
        return self.monitor._slot_key(slot, metric)

    def _assert_ratio(self, got, want, msg):
        self.assertAlmostEqual(got / want, 1.0, places=4, msg=f"{msg}: {got}")


class OptimHealthReductionTest(_Base):
    def test_every_parameter_is_summed_back_to_its_whole_tensor(self):
        """Per parameter, over the owner group -- and the model and expert axes."""
        _, grads, latest = self._step()
        checked = 0
        for name in self.shards.stage:
            if not self.shards.owned(name):
                continue
            key = self._key(name, "grad_norm")
            self.assertIn(key, latest, key)
            self._assert_ratio(latest[key], _norm(grads[name]), key)
            checked += 1
        self.assertGreater(checked, 0)

    def test_grad_rel_is_taken_after_the_weights_are_summed_back(self):
        """A ratio of two group sums, so a missed shard shows up in both."""
        weights, grads, latest = self._step()
        for name in (
            "decoder.layers.0.self_attn.q_proj.weight",
            "decoder.layers.1.self_attn.q_proj.weight",
            "decoder.layers.0.input_layernorm.weight",
            "embed_tokens.weight",
            "lm_head.weight",
        ):
            if not self.shards.owned(name):
                continue
            key = self._key(name, "grad_rel")
            self.assertIn(key, latest, key)
            self._assert_ratio(
                latest[key], _norm(grads[name]) / _norm(weights[name]), key
            )

    def test_model_totals_split_the_global_norm_over_the_pipeline(self):
        """``_dense`` and ``_expert`` cover the model, ``_total`` their hypot."""
        _, grads, latest = self._step()
        expert = math.sqrt(
            sum(_sq(g) for n, g in grads.items() if n.endswith("experts"))
        )
        dense = math.sqrt(
            sum(_sq(g) for n, g in grads.items() if not n.endswith("experts"))
        )
        model = f"{PREFIX}/model"
        self._assert_ratio(latest[f"{model}/grad_norm_dense"], dense, "dense")
        self._assert_ratio(
            latest[f"{model}/grad_norm_expert"], expert, "expert"
        )
        self.assertAlmostEqual(
            latest[f"{model}/grad_norm_total"],
            math.hypot(dense, expert),
            places=4,
        )

    def test_every_expert_of_the_layer_reaches_the_quantiles(self):
        """The per-expert vectors are all-gathered over the expert group.

        Each expert-parallel rank holds different experts, so a gather on the
        wrong axis or in the wrong order changes the spread the layer reports.
        """
        # Step on every rank before deciding to skip: the step's reduction is a
        # collective, and a stage that skipped it would leave the others waiting.
        weights, grads, latest = self._step()
        if not self.shards.owned(EXPERT_PARAM):
            self.skipTest("the expert layer is on another pipeline stage")
        local = dict(self.model.named_parameters())[EXPERT_PARAM].shape[0]
        # Expert ``k * local + i`` is expert ``i`` of the k-th rank of the
        # expert group, the order the gather concatenates in.
        reference = [
            _norm(grads[EXPERT_PARAM][expert])
            / _norm(weights[EXPERT_PARAM][expert])
            for expert in range(local * self.ep_degree)
        ]
        for metric, want in (
            ("grad_rel_min", min(reference)),
            ("grad_rel_max", max(reference)),
            ("grad_rel_median", _median(reference)),
            (
                "zero_grad_experts",
                float(sum(1 for value in reference if value == 0.0)),
            ),
        ):
            key = self._key(EXPERT_PARAM, metric)
            self.assertIn(key, latest, key)
            if metric == "zero_grad_experts":
                self.assertEqual(latest[key], want, key)
            else:
                self._assert_ratio(latest[key], want, key)

    def test_a_rank_that_failed_to_capture_silences_the_whole_step(self):
        """One rank missing turns into "nobody logs this step", not a mismatch."""
        self._step(silent_rank=0)
        self.assertEqual(training_logs.get_latest(prefix=PREFIX), {})

    def test_every_rank_reports_the_same_model_totals(self):
        """The reduction leaves every rank holding the same model totals.

        Only the totals: a per-parameter value is read on the pipeline stage
        that owns the parameter, so those keys are not shared by every rank.
        """
        _, _, latest = self._step()
        model = f"{PREFIX}/model"
        keys = [
            f"{model}/grad_norm_dense",
            f"{model}/grad_norm_expert",
            f"{model}/grad_norm_total",
            f"{model}/lr",
        ]
        # Missing keys become NaN rather than raising: every rank has to reach
        # the collective below even when the step went wrong on one of them.
        values = paddle.to_tensor(
            [latest.get(key, float("nan")) for key in keys], dtype="float64"
        )
        if dist.get_world_size() > 1:
            spread = values.clone()
            dist.all_reduce(spread, op=dist.ReduceOp.MAX)
            smallest = values.clone()
            dist.all_reduce(smallest, op=dist.ReduceOp.MIN)
            self.assertLess(
                float((spread - smallest).abs().max()),
                1e-6 * float(values.abs().max()),
                keys,
            )


class ReductionPlanTest(_Base):
    def test_a_coloured_parameter_is_summed_over_its_colour_group(self):
        """The bucket's group comes off the colour, not from sharding.

        The numeric tests cannot see this: the expert colour group has the same
        ranks as the sharding group under every topology that fits on 8 cards.
        Here the experts are coloured with a group that is deliberately not
        sharding's, and the plan has to name it. Building a plan is pure local
        bookkeeping -- no collective -- so this stays off the collective path
        and a world that has no such group can skip it.
        """
        colour = _odd_colour(self.hcg)
        if colour is None:
            self.skipTest("every group of this world is the sharding group")
        model = _ToyModel(EXPERTS // self.ep_degree)
        _annotate(model, self.hcg, colour=colour)
        monitor = optim_monitor.PaddleOptimHealthMonitor()
        monitor.register_hooks(model)
        self.assertTrue(monitor.attach_optimizer(_optimizer(model)))
        experts = {s.index for s in monitor._slots if s.is_expert}
        self.assertTrue(experts)
        groups = next(
            groups
            for index, groups in monitor._plan["buckets"]
            if experts & set(index.tolist())
        )
        self.assertIs(groups[0], colour)
        self.assertNotIn(self.hcg.get_sharding_parallel_group(), groups)


if __name__ == "__main__":
    unittest.main()
