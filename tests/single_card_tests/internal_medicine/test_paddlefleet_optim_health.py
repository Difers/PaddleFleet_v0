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

"""optim_health: weight grad norms and effective lr against a real Muon step.

Every test drives ``paddle.optimizer.Muon`` itself -- MuonH, AdamH and plain
AdamW parameters in one model -- because what can break is the reading of the
optimizer (which weight, which gradient, before or after clipping), not the
arithmetic on four sums of squares. Single process: the cross-rank sums are the
identity here, so every number must equal what is computed directly from the
weights and gradients around the step.
"""

import importlib
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

importlib.import_module("_backend_env").skip_unless_backend("paddlefleet")

try:
    paddle = importlib.import_module("paddle")
    nn = importlib.import_module("paddle.nn")
    muon_mod = importlib.import_module("paddle.optimizer.muon")
except (
    Exception
) as exc:  # pragma: no cover - depends on optional backend install
    raise unittest.SkipTest(f"paddle backend unavailable: {exc}") from exc

optim_monitor = importlib.import_module(
    "paddlefleet.internal_medicine.backends.paddlefleet.optim_monitor"
)
training_logs = importlib.import_module(
    "paddlefleet.internal_medicine.core.training_logs"
).training_logs

Muon = muon_mod.Muon
MuonParamInfo = muon_mod.MuonParamInfo

HIDDEN, EXPERTS, VOCAB, TOKENS = 8, 4, 16, 6
LR = 0.02
WD = 0.1
PREFIX = "optim_health"


class ToyAttn(nn.Layer):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(HIDDEN, HIDDEN, bias_attr=False)


class ToyNorm(nn.Layer):
    def __init__(self):
        super().__init__()
        self.weight = self.create_parameter(
            [HIDDEN], default_initializer=nn.initializer.Constant(1.0)
        )


class ToyMoE(nn.Layer):
    def __init__(self, dtype):
        super().__init__()
        self.gate = nn.Linear(HIDDEN, EXPERTS, bias_attr=False)
        self.experts = self.create_parameter(
            [EXPERTS, HIDDEN, HIDDEN],
            dtype=dtype,
            default_initializer=nn.initializer.Normal(std=0.2),
        )
        self.experts.no_sync = True
        self.expert_mask = paddle.ones([EXPERTS])


class ToyLayer(nn.Layer):
    def __init__(self, idx, dtype):
        super().__init__()
        self.layer_number = idx
        self.input_layernorm = ToyNorm()
        self.self_attn = ToyAttn()
        self.mlp = ToyMoE(dtype)

    def forward(self, x):
        h = x * self.input_layernorm.weight
        h = h + self.self_attn.q_proj(h)
        scores = nn.functional.softmax(self.mlp.gate(h), axis=-1)
        scores = scores * self.mlp.expert_mask
        experts = self.mlp.experts.astype("float32")
        routed = paddle.einsum("te,ehk,th->tk", scores, experts, h)
        return h + routed


class ToyDecoder(nn.Layer):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.LayerList(layers)


class ToyModel(nn.Layer):
    """Embedding + 2 layers + LM head, laid out like PaddleFleet's GPTModel."""

    def __init__(self, dtype="float32"):
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB, HIDDEN)
        self.decoder = ToyDecoder([ToyLayer(i, dtype) for i in range(2)])
        self.lm_head = nn.Linear(HIDDEN, VOCAB, bias_attr=False)

    def forward(self, ids):
        x = self.embed_tokens(ids)
        for layer in self.decoder.layers:
            x = layer(x)
        return self.lm_head(x)


class ToyLMHead(nn.Layer):
    """Holds its weight directly, like GPTLMHead: its path is ``<slot>.weight``."""

    def __init__(self):
        super().__init__()
        self.weight = self.create_parameter([HIDDEN, VOCAB])


class ToyPipeModel(nn.Layer):
    """PipelineLayer layout: every stage is a numbered slot of ``run_function``."""

    def __init__(self):
        super().__init__()
        self.run_function = nn.LayerList(
            [nn.Embedding(VOCAB, HIDDEN), ToyLayer(0, "float32"), ToyLMHead()]
        )


def _kind(name):
    if "lm_head" in name:
        return "adamh"
    if "q_proj" in name or "experts" in name:
        return "muonh"
    return "adamw"


def _optimizer(model, grad_clip=None, multi_precision=False):
    info = {}
    for name, p in model.named_parameters():
        kind = _kind(name)
        info[p.name] = MuonParamInfo(
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
        grad_clip=grad_clip,
        multi_precision=multi_precision,
    )


def _monitor(model, optimizer, interval=1):
    monitor = optim_monitor.PaddleOptimHealthMonitor(monitor_interval=interval)
    monitor.register_hooks(model)
    assert monitor.attach_optimizer(optimizer)
    return monitor


def _backward(model, seed=0):
    paddle.seed(seed)
    ids = paddle.randint(0, VOCAB, [TOKENS])
    labels = paddle.randint(0, VOCAB, [TOKENS])
    logits = model(ids)
    loss = nn.functional.cross_entropy(logits, labels).mean()
    loss.backward()


def _fp32(t):
    return t.detach().astype("float32")


def _weights(model, optimizer):
    masters = optimizer._master_weights
    return {
        name: _fp32(masters.get(p.name, p)).clone()
        for name, p in model.named_parameters()
    }


def _norm(t):
    return float(paddle.linalg.norm(t.reshape([-1])))


def _compute_aus(w, u, lr, wd=0.0, hyperball=False):
    """The reference AUS, from the base update ``u`` (before the lr)."""
    w, u = w.reshape([-1]), u.reshape([-1])
    r, q = _norm(w), _norm(u)
    c = max(-1.0, min(1.0, float((w * u).sum()) / (r * q)))
    s = math.sqrt(max(0.0, 1.0 - c * c))
    if hyperball:
        a, b = 1.0 - lr * c, lr * s
    else:
        a, b = (1.0 - lr * wd) * r - lr * q * c, lr * q * s
    return 2.0 * math.sin(math.atan2(b, a) / 2.0)


def _chord(w0, w1):
    """``2 sin(theta / 2)`` between two weights, straight from the definition."""
    return _norm(w1 / _norm(w1) - w0 / _norm(w0))


class _Base(unittest.TestCase):
    def setUp(self):
        training_logs.reset()
        paddle.seed(1234)

    def tearDown(self):
        training_logs.reset()

    def _step(self, model, optimizer, monitor, seed=0):
        """One training step; returns (grads, W_t, W_{t+1}, logged metrics)."""
        training_logs.reset()
        _backward(model, seed)
        grads = {n: _fp32(p.grad).clone() for n, p in model.named_parameters()}
        before = _weights(model, optimizer)
        optimizer.step()
        optimizer.clear_grad()
        monitor.step()
        after = _weights(model, optimizer)
        return grads, before, after, training_logs.get_latest(prefix=PREFIX)


def _key(monitor, name, metric):
    slot = next(s for s in monitor._slots if s.param.name == name)
    return monitor._slot_key(slot, metric)


class OptimHealthStepTest(_Base):
    def test_every_parameter_matches_the_step_it_actually_took(self):
        model = ToyModel()
        optimizer = _optimizer(model)
        monitor = _monitor(model, optimizer)
        grads, before, after, latest = self._step(model, optimizer, monitor)

        params = dict(model.named_parameters())
        for name, p in params.items():
            g, w0, w1 = grads[name], before[name], after[name]
            expect = {
                "grad_norm": _norm(g),
                "grad_rel": _norm(g) / _norm(w0),
                "eff_lr": _norm(w1 - w0) / _norm(w0),
                "eff_lr_ratio": _norm(w1 - w0) / _norm(w0) / LR,
                "param_norm": _norm(w1),
            }
            for metric, value in expect.items():
                key = _key(monitor, p.name, metric)
                self.assertIn(key, latest, key)
                self.assertAlmostEqual(
                    latest[key] / value, 1.0, places=4, msg=key
                )

    def test_keys_follow_the_layer_layout(self):
        model = ToyModel()
        optimizer = _optimizer(model)
        monitor = _monitor(model, optimizer)
        latest = self._step(model, optimizer, monitor)[3]
        for key in (
            f"{PREFIX}/layer_0/self_attn_q_proj_eff_lr",
            f"{PREFIX}/layer_1/input_layernorm_grad_rel",
            f"{PREFIX}/layer_1/mlp_gate_grad_norm",
            f"{PREFIX}/layer_0/mlp_experts_eff_lr_max",
            f"{PREFIX}/model/embed_tokens/eff_lr",
            f"{PREFIX}/model/lm_head/grad_rel",
            f"{PREFIX}/model/grad_norm_dense",
            f"{PREFIX}/model/lr",
        ):
            self.assertIn(key, latest)

    def test_pipeline_slots_are_named_by_their_layer(self):
        """``run_function.2.weight`` must not become the key ``model/2``."""
        model = ToyPipeModel()
        monitor = optim_monitor.PaddleOptimHealthMonitor()
        monitor.register_hooks(model)
        head, embed = model.run_function[2], model.run_function[0]
        self.assertEqual(
            _key(monitor, head.weight.name, "eff_lr"),
            f"{PREFIX}/model/lm_head/eff_lr",
        )
        self.assertEqual(
            _key(monitor, embed.weight.name, "eff_lr"),
            f"{PREFIX}/model/embedding/eff_lr",
        )
        q_proj = model.run_function[1].self_attn.q_proj.weight
        self.assertEqual(
            _key(monitor, q_proj.name, "eff_lr"),
            f"{PREFIX}/layer_0/self_attn_q_proj_eff_lr",
        )

    def test_hyperball_holds_the_norm_and_steps_at_most_lr(self):
        """The chord peaks at ``sqrt(2 - 2 sqrt(1 - lr^2))``, i.e. ~lr (1 + lr^2 / 8)."""
        model = ToyModel()
        optimizer = _optimizer(model)
        monitor = _monitor(model, optimizer)
        latest = self._step(model, optimizer, monitor)[3]
        bound = math.sqrt(2.0 - 2.0 * math.sqrt(1.0 - LR * LR)) / LR
        for name, p in model.named_parameters():
            if _kind(name) == "adamw":
                continue
            drift = latest[_key(monitor, p.name, "norm_drift")]
            ratio = latest[_key(monitor, p.name, "eff_lr_ratio")]
            self.assertLess(abs(drift), 1e-5, name)
            self.assertGreater(ratio, 0.0, name)
            self.assertLessEqual(ratio, bound + 1e-6, name)

    def test_adamh_eff_lr_matches_the_closed_form(self):
        """First AdamH step: the direction is sign(g), so the rotation is exact.

        ``m / sqrt(v) = (1 - b1) / sqrt(1 - b2) * sign(g)`` on step one, which the
        Frobenius normalisation turns into ``u = sign(g) / sqrt(n)``. With
        ``c = <W, u> / R`` the step is a rotation by
        ``phi = atan2(lr sqrt(1 - c^2), 1 - lr c)``, a chord of ``2 sin(phi / 2)``.
        """
        model = ToyModel()
        optimizer = _optimizer(model)
        monitor = _monitor(model, optimizer)
        grads, before, _, latest = self._step(model, optimizer, monitor)
        w0 = before["lm_head.weight"]
        direction = paddle.sign(grads["lm_head.weight"])
        direction = direction / _norm(direction)
        c = float((w0 * direction).sum()) / _norm(w0)
        phi = math.atan2(LR * math.sqrt(1.0 - c * c), 1.0 - LR * c)
        key = _key(monitor, model.lm_head.weight.name, "eff_lr")
        self.assertAlmostEqual(latest[key], 2 * math.sin(phi / 2), places=6)

    def test_grad_is_read_before_clipping(self):
        model = ToyModel()
        optimizer = _optimizer(
            model, grad_clip=nn.ClipGradByGlobalNorm(clip_norm=1e-4)
        )
        monitor = _monitor(model, optimizer)
        grads, _, _, latest = self._step(model, optimizer, monitor)
        total = math.sqrt(sum(_norm(g) ** 2 for g in grads.values()))
        self.assertGreater(total, 1e-2)  # so the clip really rescales
        self.assertAlmostEqual(
            latest[f"{PREFIX}/model/grad_norm_total"] / total, 1.0, places=4
        )
        key = _key(monitor, model.lm_head.weight.name, "grad_norm")
        self.assertAlmostEqual(
            latest[key] / _norm(grads["lm_head.weight"]), 1.0, places=4
        )

    def test_dense_and_expert_totals_split_the_global_norm(self):
        model = ToyModel()
        optimizer = _optimizer(model)
        monitor = _monitor(model, optimizer)
        grads, _, _, latest = self._step(model, optimizer, monitor)
        expert = math.sqrt(
            sum(_norm(g) ** 2 for n, g in grads.items() if "experts" in n)
        )
        dense = math.sqrt(
            sum(_norm(g) ** 2 for n, g in grads.items() if "experts" not in n)
        )
        self.assertAlmostEqual(
            latest[f"{PREFIX}/model/grad_norm_expert"] / expert, 1.0, places=4
        )
        self.assertAlmostEqual(
            latest[f"{PREFIX}/model/grad_norm_dense"] / dense, 1.0, places=4
        )


class OptimHealthAusTest(_Base):
    def test_aus_is_the_angle_the_weight_turned(self):
        """Every parameter, every optimizer kind, over a few steps."""
        model = ToyModel()
        optimizer = _optimizer(model)
        monitor = _monitor(model, optimizer)
        for seed in range(3):
            _, before, after, latest = self._step(
                model, optimizer, monitor, seed=seed
            )
            for name, p in model.named_parameters():
                aus = latest[_key(monitor, p.name, "aus")]
                exact = _chord(before[name], after[name])
                self.assertAlmostEqual(aus / exact, 1.0, places=3, msg=name)
                ratio = latest[_key(monitor, p.name, "aus_over_lr")]
                self.assertAlmostEqual(ratio / (aus / LR), 1.0, places=6)

    def test_aus_matches_the_reference_on_the_first_step(self):
        """Step one has a known base update: sign(g), for AdamH and AdamW alike.

        ``m / sqrt(v)`` with bias correction is ``sign(g)`` on the first step, so
        the reference formula can be evaluated exactly: Hyperball for the AdamH
        lm_head, decoupled weight decay for the AdamW layer norm.
        """
        model = ToyModel()
        optimizer = _optimizer(model)
        monitor = _monitor(model, optimizer)
        grads, before, _, latest = self._step(model, optimizer, monitor)
        for name, param, hyperball in (
            ("lm_head.weight", model.lm_head.weight, True),
            (
                "decoder.layers.0.input_layernorm.weight",
                model.decoder.layers[0].input_layernorm.weight,
                False,
            ),
        ):
            expect = _compute_aus(
                before[name],
                paddle.sign(grads[name]),
                LR,
                wd=WD,
                hyperball=hyperball,
            )
            key = _key(monitor, param.name, "aus")
            self.assertAlmostEqual(latest[key] / expect, 1.0, places=4, msg=key)

    def test_aus_over_lr_is_about_one_under_hyperball(self):
        model = ToyModel()
        optimizer = _optimizer(model)
        monitor = _monitor(model, optimizer)
        self._step(model, optimizer, monitor, seed=0)
        latest = self._step(model, optimizer, monitor, seed=1)[3]
        bound = math.sqrt(2.0 - 2.0 * math.sqrt(1.0 - LR * LR)) / LR
        for name, p in model.named_parameters():
            if _kind(name) == "adamw":
                continue
            aus = latest[_key(monitor, p.name, "aus")]
            ratio = latest[_key(monitor, p.name, "aus_over_lr")]
            self.assertAlmostEqual(
                aus / latest[_key(monitor, p.name, "eff_lr")], 1.0, places=4
            )
            self.assertGreater(ratio, 0.5, name)
            self.assertLessEqual(ratio, bound + 1e-6, name)

    def test_aus_leaves_the_norm_change_out(self):
        """A pure 1% rescale moves 1% (eff_lr) but turns nothing (aus)."""
        from paddlefleet.internal_medicine.backends.paddlefleet import (
            optim_metrics,
        )

        paddle.seed(0)
        w = paddle.randn([HIDDEN, HIDDEN])
        turned = w + LR * _norm(w) * paddle.randn([HIDDEN, HIDDEN]) / HIDDEN
        for w1 in (w * 1.01, turned * 1.01):
            sums = optim_metrics.step_sums(
                paddle.zeros([]), w, w1, per_expert=False
            ).unsqueeze(0)
            out = optim_metrics.param_metrics(sums, paddle.to_tensor([LR]))
            self.assertAlmostEqual(
                float(out["aus"][0]), _chord(w, w1), places=5
            )
        self.assertAlmostEqual(
            float(out["eff_lr"][0]), _norm(w1 - w) / _norm(w), places=5
        )
        sums = optim_metrics.step_sums(
            paddle.zeros([]), w, w * 1.01, per_expert=False
        ).unsqueeze(0)
        out = optim_metrics.param_metrics(sums, paddle.to_tensor([LR]))
        self.assertLess(float(out["aus"][0]), 1e-3)
        self.assertAlmostEqual(float(out["eff_lr"][0]), 0.01, places=5)


class OptimHealthExpertTest(_Base):
    def test_a_tokenless_expert_still_turns_on_stale_momentum(self):
        """Zero gradient, full-lr step: the Muon hazard the per-expert keys expose."""
        model = ToyModel()
        optimizer = _optimizer(model)
        monitor = _monitor(model, optimizer)
        self._step(model, optimizer, monitor, seed=0)

        mask = paddle.ones([EXPERTS])
        mask[0] = 0.0
        for layer in model.decoder.layers:
            layer.mlp.expert_mask = mask
        grads, before, after, latest = self._step(
            model, optimizer, monitor, seed=1
        )

        name = "decoder.layers.0.mlp.experts"
        self.assertEqual(float(paddle.abs(grads[name][0]).sum()), 0.0)
        moved = _norm(after[name][0] - before[name][0]) / _norm(before[name][0])
        self.assertGreater(moved, 0.1 * LR)

        param = model.decoder.layers[0].mlp.experts.name
        self.assertEqual(latest[_key(monitor, param, "zero_grad_experts")], 1.0)
        self.assertEqual(latest[_key(monitor, param, "grad_rel_min")], 0.0)
        self.assertAlmostEqual(
            latest[_key(monitor, param, "eff_lr_min")] / moved,
            1.0,
            delta=0.2,
        )
        per_expert = [
            _norm(after[name][e] - before[name][e]) / _norm(before[name][e])
            for e in range(EXPERTS)
        ]
        self.assertAlmostEqual(
            latest[_key(monitor, param, "eff_lr_max")] / max(per_expert),
            1.0,
            places=4,
        )


class OptimHealthMasterWeightTest(_Base):
    def test_bf16_weights_are_read_from_the_fp32_master(self):
        """The step is ~lr of the weight, the same order as bf16 rounding."""
        model = ToyModel(dtype="bfloat16")
        optimizer = _optimizer(model, multi_precision=True)
        monitor = _monitor(model, optimizer)
        self._step(model, optimizer, monitor, seed=0)  # creates the masters
        param = model.decoder.layers[0].mlp.experts
        self.assertIn(param.name, optimizer._master_weights)
        _, before, after, latest = self._step(model, optimizer, monitor, seed=1)
        name = "decoder.layers.0.mlp.experts"
        exact = _norm(after[name] - before[name]) / _norm(before[name])
        self.assertAlmostEqual(
            latest[_key(monitor, param.name, "eff_lr")] / exact, 1.0, places=4
        )
        drift = latest[_key(monitor, param.name, "norm_drift")]
        self.assertLess(abs(drift), 1e-5)


class OptimHealthSideEffectTest(_Base):
    def test_monitoring_does_not_change_the_update(self):
        paddle.seed(7)
        plain = ToyModel()
        paddle.seed(7)
        watched = ToyModel()
        opt_plain = _optimizer(plain)
        opt_watched = _optimizer(watched)
        monitor = _monitor(watched, opt_watched)
        for seed in range(3):
            _backward(plain, seed)
            opt_plain.step()
            opt_plain.clear_grad()
            self._step(watched, opt_watched, monitor, seed)
        for (name, a), (_, b) in zip(
            plain.named_parameters(), watched.named_parameters(), strict=True
        ):
            self.assertTrue(bool(paddle.equal_all(a, b)), name)

    def test_unsampled_steps_capture_and_log_nothing(self):
        model = ToyModel()
        optimizer = _optimizer(model)
        monitor = _monitor(model, optimizer, interval=2)
        self.assertTrue(self._step(model, optimizer, monitor, seed=0)[3])
        latest = self._step(model, optimizer, monitor, seed=1)[3]
        self.assertEqual(latest, {})
        self.assertTrue(self._step(model, optimizer, monitor, seed=2)[3])

    def test_attach_is_idempotent_and_removable(self):
        model = ToyModel()
        optimizer = _optimizer(model)
        monitor = _monitor(model, optimizer)
        wrapped = optimizer._apply_optimize
        self.assertTrue(monitor.attach_optimizer(optimizer))
        self.assertIs(optimizer._apply_optimize, wrapped)
        monitor.remove_hooks()
        self.assertNotIn("_apply_optimize", optimizer.__dict__)
        self.assertEqual(
            optimizer._apply_optimize.__func__, Muon._apply_optimize
        )

    def test_attach_unwraps_sharding_style_wrappers(self):
        class Wrapper:
            def __init__(self, inner):
                self._inner_opt = inner

            def __getattr__(self, item):
                return getattr(self._inner_opt, item)

        model = ToyModel()
        optimizer = _optimizer(model)
        monitor = optim_monitor.PaddleOptimHealthMonitor()
        monitor.register_hooks(model)
        self.assertTrue(monitor.attach_optimizer(Wrapper(Wrapper(optimizer))))
        self.assertIn("_apply_optimize", optimizer.__dict__)


def _raising(*args, **kwargs):
    raise RuntimeError("probe failure")


class ToyScale(nn.Layer):
    """Holds its weight directly and carries no label of its own."""

    def __init__(self):
        super().__init__()
        self.weight = self.create_parameter([HIDDEN])


class ToyNestedPipe(nn.Layer):
    """A stage whose slots are numbered twice: ``run_function.0.1.weight``."""

    def __init__(self):
        super().__init__()
        self.run_function = nn.LayerList(
            [nn.LayerList([ToyScale(), ToyScale()])]
        )


class ToyDenseOnly(nn.Layer):
    """No grouped experts at all, so the expert half of the model is empty."""

    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB, HIDDEN)
        self.lm_head = ToyLMHead()

    def forward(self, ids):
        return paddle.matmul(self.embed_tokens(ids), self.lm_head.weight)


class ToyMTP(nn.Layer):
    """An MTP wrapper: the layer to read hangs off ``transformer_layer``."""

    def __init__(self, idx, num_hidden_layers):
        super().__init__()
        self.config = SimpleNamespace(num_hidden_layers=num_hidden_layers)
        self.layer_number = idx
        self.transformer_layer = ToyLayer(idx, "float32")


class ToyMTPModel(nn.Layer):
    """One backbone layer and one MTP layer, the layout GPTModel builds."""

    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB, HIDDEN)
        self.decoder = ToyDecoder([ToyLayer(0, "float32"), ToyMTP(1, 1)])


class ToyUnevenExperts(nn.Layer):
    """Two grouped-expert weights of different widths."""

    def __init__(self):
        super().__init__()
        self.decoder = ToyDecoder(
            [ToyLayer(0, "float32"), ToyLayer(1, "float32")]
        )
        narrow = self.decoder.layers[1].mlp
        narrow.experts = narrow.create_parameter([EXPERTS // 2, HIDDEN, HIDDEN])
        narrow.experts.no_sync = True


class BrokenHCG:
    """Every group query fails, as they do while the topology is half up."""

    def __getattr__(self, name):
        return _raising


class ClipWithStat(nn.ClipGradByGlobalNorm):
    """Publishes the norm it computed, the way HybridParallelClipGrad does."""

    GLOBAL_NORM = 3.5

    def __call__(self, params_grads):
        self.stat = {"global_grad_norm": self.GLOBAL_NORM}
        return super().__call__(params_grads)


class OptimHealthDegradedPathTest(_Base):
    """What the monitor does with what it cannot read.

    It wraps the optimizer step of a real run, so each failure it can meet has
    to end as "this step is not logged": never an exception reaching the
    trainer, never a changed update, and -- since every such decision is made
    from the model and the topology alone -- never one rank reducing while
    another has already given up.
    """

    def test_an_optimizer_it_cannot_read_is_declined(self):
        class Bare:
            pass

        model = ToyModel()
        monitor = optim_monitor.PaddleOptimHealthMonitor()
        monitor.register_hooks(model)
        self.assertFalse(monitor.attach_optimizer(None))
        self.assertFalse(monitor.attach_optimizer(Bare()))
        self.assertIn("attach", monitor._failed)
        # Declined once and for all, so a later optimizer is not picked up
        # halfway through training either.
        self.assertFalse(monitor.attach_optimizer(_optimizer(model)))

    def test_a_plan_that_cannot_be_built_disables_the_monitor(self):
        model = ToyModel()
        optimizer = _optimizer(model)
        monitor = optim_monitor.PaddleOptimHealthMonitor()
        monitor.register_hooks(model)
        monitor._build_plan = _raising
        self.assertFalse(monitor.attach_optimizer(optimizer))
        self.assertIn("reduction plan", monitor._failed)
        # The optimizer is left unwrapped: training runs on without metrics.
        self.assertNotIn("_apply_optimize", optimizer.__dict__)
        _backward(model)
        optimizer.step()
        optimizer.clear_grad()
        monitor.step()
        self.assertEqual(training_logs.get_latest(prefix=PREFIX), {})

    def test_an_lr_ratio_that_raises_is_read_as_one(self):
        model = ToyModel()
        optimizer = _optimizer(model)
        optimizer._lr_ratio = _raising
        monitor = _monitor(model, optimizer)
        self.assertEqual(
            monitor._plan["lr_ratio"].tolist(), [1.0] * len(monitor._slots)
        )

    def test_a_pre_step_failure_leaves_the_update_alone(self):
        model = ToyModel()
        optimizer = _optimizer(model)
        monitor = _monitor(model, optimizer)
        monitor._before_step = _raising
        _, before, after, latest = self._step(model, optimizer, monitor)
        self.assertEqual(latest, {})
        self.assertIn("pre-step capture", monitor._failed)
        name = "embed_tokens.weight"
        self.assertGreater(_norm(after[name] - before[name]), 0.0)

    def test_a_post_step_failure_drops_only_the_metrics(self):
        model = ToyModel()
        optimizer = _optimizer(model)
        monitor = _monitor(model, optimizer)
        monitor._after_step = _raising
        _, before, after, latest = self._step(model, optimizer, monitor)
        self.assertEqual(latest, {})
        self.assertIsNone(monitor._pending)
        self.assertIn("post-step capture", monitor._failed)
        name = "embed_tokens.weight"
        self.assertGreater(_norm(after[name] - before[name]), 0.0)

    def test_a_failing_reduction_does_not_reach_the_trainer(self):
        model = ToyModel()
        optimizer = _optimizer(model)
        monitor = _monitor(model, optimizer)
        monitor._emit = _raising
        latest = self._step(model, optimizer, monitor)[3]
        self.assertEqual(latest, {})
        self.assertIn("reduce/emit", monitor._failed)

    def test_the_capture_skips_what_it_cannot_read(self):
        model = ToyModel()
        optimizer = _optimizer(model)
        monitor = _monitor(model, optimizer)
        _backward(model)
        watched = dict(model.named_parameters())["embed_tokens.weight"]
        unwatched = nn.Linear(HIDDEN, HIDDEN).weight
        self.assertEqual(
            monitor._before_step(
                [
                    (watched, None),  # no gradient this step
                    (unwatched, watched.grad),  # not a monitored parameter
                    (watched, paddle.Tensor()),  # an empty gradient slot
                ]
            ),
            [],
        )

    def test_params_grads_passed_positionally_are_still_captured(self):
        """``_apply_optimize`` is called both ways across paddle versions."""
        model = ToyModel()
        optimizer = _optimizer(model)
        monitor = _monitor(model, optimizer)
        _backward(model)
        params_grads = [(p, p.grad) for p in model.parameters()]
        monitor._apply_optimize(
            lambda *args, **kwargs: None, (None, None, params_grads), {}
        )
        self.assertEqual(len(monitor._pending["index"]), len(monitor._slots))

    def test_a_model_without_experts_reports_an_empty_expert_half(self):
        model = ToyDenseOnly()
        optimizer = _optimizer(model)
        monitor = _monitor(model, optimizer)
        latest = self._step(model, optimizer, monitor)[3]
        self.assertGreater(latest[f"{PREFIX}/model/grad_norm_dense"], 0.0)
        self.assertEqual(latest[f"{PREFIX}/model/grad_norm_expert"], 0.0)

    def test_the_clip_norm_is_reported_when_the_clip_publishes_one(self):
        model = ToyModel()
        optimizer = _optimizer(model, grad_clip=ClipWithStat(clip_norm=1.0))
        monitor = _monitor(model, optimizer)
        latest = self._step(model, optimizer, monitor)[3]
        self.assertEqual(
            latest[f"{PREFIX}/model/clip_grad_norm"], ClipWithStat.GLOBAL_NORM
        )

    def test_a_frozen_parameter_is_not_monitored(self):
        model = ToyModel()
        frozen = model.decoder.layers[0].input_layernorm.weight
        frozen.stop_gradient = True
        monitor = optim_monitor.PaddleOptimHealthMonitor()
        monitor.register_hooks(model)
        self.assertNotIn(frozen.name, monitor._slot_by_name)
        self.assertTrue(monitor._slots)

    def test_two_numbered_slots_of_one_class_keep_separate_keys(self):
        """``run_function.0.0`` and ``.0.1`` are both named by their class."""
        monitor = optim_monitor.PaddleOptimHealthMonitor()
        monitor.register_hooks(ToyNestedPipe())
        self.assertEqual(
            [slot.metric_base for slot in monitor._slots],
            ["toy_scale", "toy_scale_1"],
        )

    def test_a_pipeline_rank_that_cannot_be_read_is_taken_as_zero(self):
        parallel_state = importlib.import_module("paddlefleet.parallel_state")
        original = parallel_state.get_pipeline_model_parallel_rank
        parallel_state.get_pipeline_model_parallel_rank = _raising
        try:
            monitor = optim_monitor.PaddleOptimHealthMonitor()
            monitor.register_hooks(ToyModel())
        finally:
            parallel_state.get_pipeline_model_parallel_rank = original
        self.assertEqual(monitor.pp_rank, 0)
        self.assertTrue(monitor._slots)

    def test_groups_that_cannot_be_queried_leave_the_sums_local(self):
        """A half-built topology costs the reduction, not the step."""
        fleet = importlib.import_module("paddle.distributed.fleet")
        parallel_state = importlib.import_module("paddlefleet.parallel_state")
        originals = (
            fleet.get_hybrid_communicate_group,
            parallel_state.get_expert_model_parallel_group,
        )
        fleet.get_hybrid_communicate_group = BrokenHCG
        parallel_state.get_expert_model_parallel_group = _raising
        try:
            model = ToyModel()
            optimizer = _optimizer(model)
            monitor = _monitor(model, optimizer)
            latest = self._step(model, optimizer, monitor)[3]
        finally:
            (
                fleet.get_hybrid_communicate_group,
                parallel_state.get_expert_model_parallel_group,
            ) = originals
        self.assertEqual(monitor._plan["buckets"], [])
        self.assertIsNone(monitor._plan["ep"])
        self.assertIn(f"{PREFIX}/model/grad_norm_total", latest)

    def test_setup_registers_the_monitor_under_its_name(self):
        monitor_dict = {}
        model = ToyModel()
        self.assertIs(
            optim_monitor.setup_optim_monitor(
                model, monitor_dict=monitor_dict, monitor_interval=2
            ),
            model,
        )
        monitor = monitor_dict["optim_health"]
        self.assertEqual(monitor.monitor_interval, 2)
        self.assertTrue(monitor._slots)

    def test_an_mtp_layer_is_marked_in_its_keys(self):
        """An MTP layer shares the metric names, not the layer token."""
        monitor = optim_monitor.PaddleOptimHealthMonitor()
        monitor.register_hooks(ToyMTPModel())
        keys = [
            monitor._slot_key(slot, "grad_norm")
            for slot in monitor._slots
            if slot.layer_idx is not None
        ]
        self.assertTrue(any("_mtp/" in key for key in keys), keys)
        self.assertTrue(any("_mtp/" not in key for key in keys), keys)

    def test_expert_weights_of_two_widths_are_not_read_per_expert(self):
        """Mixed widths cannot share one ``[S, E, 4]`` gather, so none do."""
        model = ToyUnevenExperts()
        monitor = optim_monitor.PaddleOptimHealthMonitor()
        monitor.register_hooks(model)
        self.assertTrue(monitor.attach_optimizer(_optimizer(model)))
        self.assertEqual(monitor._plan["expert_slots"], [])


if __name__ == "__main__":
    unittest.main()
