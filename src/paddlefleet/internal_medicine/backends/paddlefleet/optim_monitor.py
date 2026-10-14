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

"""Optimizer-step health monitor for PaddleFleet: effective lr and weight grads.

Per parameter, per sampled step, this reports the gradient the optimizer
consumed (``grad_norm``, ``grad_rel = ||g|| / ||W||``), the step it actually
took (``eff_lr = ||dW|| / ||W||``) and how far its direction turned (``aus``,
the angular update size, and ``aus_over_lr``), plus ``eff_lr_ratio``,
``param_norm`` and ``norm_drift`` -- see ``optim_metrics`` for the definitions.

Why these. Under Hyperball (MuonH / AdamH) every matrix keeps a fixed Frobenius
norm and moves by a step whose length is set by the lr alone
(``Muon._hyperball_apply`` normalises the update direction), so weight norms
carry no information and a small gradient no longer means a small step. What is
left to read is how big the gradient is relative to the weight, and how far the
weight really turned -- per parameter, since the global grad norm is dominated
by the LM head.

**Where it hooks.** The innermost optimizer's ``_apply_optimize`` is wrapped on
the instance (``MuonShardingOptimizer.step`` reaches it through
``__getattr__``). That call is the only point where all of these hold at once,
checked against Paddle ``optimizer/muon.py`` and
``fleet/meta_optimizers/muon_sharding_optimizer.py``:

- the parameters this rank updates are fixed: its locally owned whole tensors
  (Muon / Hyperball) and its element-wise slices (plain AdamW), a slice carrying
  the name of the parameter it is cut from;
- the gradient is reduced and averaged over gradient accumulation but not yet
  clipped -- ``Muon._apply_optimize`` clips first thing, in place, which is why
  the gradient is read before calling through;
- the fp32 master weight is resident, and it is the only readable copy: the
  bf16 storage of a cleared colour (FP8 experts) is ``empty_like`` garbage here;
- a step the optimizer skips never gets here, so it is never logged.

``HybridParallelOptimizer.step`` runs under ``no_grad``, so sampling is gated on
``_interval_reached`` -- ``PaddleProbe._should_monitor`` also requires grad mode
and would never fire on this path.

**Reduction.** The optimizer path only computes local sums of squares. The
cross-rank sums happen in ``_flush_buffers`` (``on_step_end``, outside the
optimizer), per parameter, over exactly the ranks that split it:

- the owner group is the sharding group, or the group of the parameter's
  ``color`` when it has one -- the same rule ``MuonShardingOptimizer`` uses to
  place whole tensors and element-wise slices, so each element is owned by one
  rank of that group and every other rank contributes zero;
- expert weights are additionally summed over the expert-parallel group, whose
  ranks hold different experts; per-expert vectors are all-gathered over it
  instead, so the expert quantiles are exact.

Every rank then holds identical values, so the cross-rank mean in ``gather.py``
leaves them unchanged. Before any of that, one tiny world-wide MIN agrees that
every rank captured the step: a local failure turns into "nobody logs this
step" rather than into mismatched collectives.

**Cost.** A float32 snapshot of this rank's shard of the weights for the length
of one optimizer step, one pass over the gradients, two over the weights, a few
small collectives. Nothing runs on unsampled steps.
"""

from __future__ import annotations

import logging
import math
import re

import paddle
from paddle import nn

from ...core.training_logs import training_logs
from .base import PaddleProbe
from .layer_discovery import get_decoder_layers, iter_monitor_layers
from .optim_metrics import (
    GRAD_SQ,
    NUM_SUMS,
    PARAM_METRICS,
    expert_metrics,
    grad_square,
    param_metrics,
    snapshot,
    step_sums,
)

logger = logging.getLogger(__name__)

_WRAP_ATTR = "_apply_optimize"
_UNSAFE = re.compile(r"[^0-9A-Za-z_]+")
_NAME_PREFIXES = ("_layers.", "model.", "run_function.")
# A pipeline stage that holds its weight directly (``GPTLMHead``) has a path of
# just ``<slot>.weight``, which leaves nothing but the slot index. The owner's
# class names it instead; the acronym-heavy names get a fixed label so the key
# stays ``lm_head`` rather than ``gptlm_head``.
_CLASS_LABELS = (("LMHead", "lm_head"), ("Embedding", "embedding"))
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _is_transformer_layer(layer) -> bool:
    """Same predicate ``massive_act`` / ``grad_health`` use."""
    return (
        hasattr(layer, "self_attn")
        or hasattr(layer, "self_attention")
        or hasattr(layer, "input_layernorm")
    )


def _metric_base(name: str) -> str:
    """Metric-safe form of a parameter path: ``.`` to ``_``, a trailing ``.weight`` dropped.

    Wrapper prefixes and the leading pipeline-slot indices (``_layers.0.0.``)
    carry no meaning in a key and differ between layouts, so they go too.
    """
    stripped = True
    while stripped:
        stripped = False
        for prefix in _NAME_PREFIXES:
            if name.startswith(prefix):
                name, stripped = name[len(prefix) :], True
    parts = name.removesuffix(".weight").split(".")
    while len(parts) > 1 and parts[0].isdigit():
        parts = parts[1:]
    return _UNSAFE.sub("_", ".".join(parts)).strip("_")


def _class_label(layer) -> str:
    cls = type(layer).__name__
    for marker, label in _CLASS_LABELS:
        if marker in cls:
            return label
    return _UNSAFE.sub("_", _CAMEL.sub("_", cls)).lower()


def _nranks(group) -> int:
    return int(getattr(group, "nranks", 1) or 1) if group is not None else 1


def _optimizer_chain(optimizer):
    """Every wrapper from ``optimizer`` down to the one that owns the update rule.

    Read through ``__dict__``: the sharding wrappers forward unknown attributes
    to the wrapped optimizer, so ``getattr`` would never stop.
    """
    chain, seen = [], set()
    opt = optimizer
    while opt is not None and id(opt) not in seen:
        seen.add(id(opt))
        chain.append(opt)
        opt = getattr(opt, "__dict__", {}).get("_inner_opt")
    return chain


def _innermost_optimizer(optimizer):
    """The optimizer that owns the update rule (``Muon``)."""
    chain = _optimizer_chain(optimizer)
    return chain[-1] if chain else None


def _color_key(param):
    color = getattr(param, "color", -1)
    value = color.get("color", -1) if isinstance(color, dict) else color
    return None if value in (-1, None) else str(value)


def _color_group(param, default):
    color = getattr(param, "color", -1)
    if isinstance(color, dict):
        return color.get("group", default)
    return default


class _Slot:
    """One monitored parameter: its metric key and how it lies across ranks."""

    __slots__ = (
        "index",
        "name",
        "layer_idx",
        "metric_base",
        "param",
        "is_expert",
        "is_tp",
        "color",
        "num_experts",
        "per_expert",
        "in_total",
    )

    def __init__(self, index, layer_idx, metric_base, param):
        self.index = index
        self.name = param.name
        self.layer_idx = layer_idx
        self.metric_base = metric_base
        self.param = param
        self.is_expert = bool(getattr(param, "no_sync", False))
        self.is_tp = (not self.is_expert) and bool(
            getattr(param, "is_distributed", False)
        )
        self.color = _color_key(param)
        shape = list(getattr(param, "original_shape", None) or param.shape)
        self.num_experts = int(shape[0]) if len(shape) == 3 else 0
        # Decided at attach time, once the optimizer says the weight is kept whole.
        self.per_expert = False
        # Mirrors HybridParallelClipGrad: parameters it skips, clips on their own
        # (multimax) or does not own (non-first shared copies) are not part of
        # the global norm this is reconciled against.
        need_clip = getattr(param, "need_clip", True)
        self.in_total = (
            need_clip is not False
            and need_clip != "multimax"
            and getattr(param, "is_firstly_shared", True) is not False
        )

    @property
    def bucket(self):
        """Rank-invariant reduction key: ranks agree on it without talking."""
        return (self.color or "", self.is_expert, self.is_tp)


class PaddleOptimHealthMonitor(PaddleProbe):
    """Per-parameter gradient and effective-lr statistics of the optimizer step."""

    METRIC_PREFIX = "optim_health"

    def __init__(
        self,
        log_per_layer=True,
        log_global=False,
        monitor_interval=1,
        verbose=False,
        exclude_families=None,
    ):
        super().__init__(
            log_per_layer=log_per_layer,
            log_global=log_global,
            monitor_interval=monitor_interval,
            verbose=verbose,
            exclude_families=exclude_families,
        )
        self._slots: list[_Slot] = []
        self._slot_by_name: dict[str, _Slot] = {}
        self._optimizer = None
        self._attach_tried = False
        # ``setup_monitors`` counts a monitor with neither hooks nor wrapped
        # methods as inactive; the optimizer wrap is this monitor's only hook.
        self._wrapped: list = []
        self._plan = None
        self._pending: dict | None = None
        self._failed: set[str] = set()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def register_hooks(self, model: nn.Layer):
        try:
            from paddlefleet.parallel_state import (
                get_pipeline_model_parallel_rank,
            )

            self.pp_rank = get_pipeline_model_parallel_rank()
        except Exception:
            pass

        layer_of: dict[int, tuple[int, str]] = {}
        monitor_layers = iter_monitor_layers(
            get_decoder_layers(model) or [],
            _is_transformer_layer,
            pp_rank=self.pp_rank,
        )
        mtp_ids = [item.idx for item in monitor_layers if item.is_mtp]
        if mtp_ids:
            self.mark_mtp_layers(mtp_ids)
        for item in monitor_layers:
            for rel_name, param in item.layer.named_parameters():
                layer_of.setdefault(id(param), (item.idx, rel_name))

        # ``named_parameters`` order is identical on every rank of a pipeline
        # stage; the reduction layout is built from it and never exchanged.
        owner_of = {
            id(param): layer
            for _, layer in model.named_sublayers(include_self=True)
            for param in layer._parameters.values()
            if param is not None
        }
        used: set[tuple] = set()
        for full_name, param in model.named_parameters():
            if param.stop_gradient or param.name in self._slot_by_name:
                continue
            layer_idx, rel_name = layer_of.get(id(param), (None, full_name))
            base = _metric_base(rel_name)
            if base.isdigit() and id(param) in owner_of:
                base = _class_label(owner_of[id(param)])
            if (layer_idx, base) in used:
                base = f"{base}_{len(self._slots)}"
            used.add((layer_idx, base))
            slot = _Slot(len(self._slots), layer_idx, base, param)
            self._slots.append(slot)
            self._slot_by_name[slot.name] = slot

        logger.info(
            "[PaddleOptimHealthMonitor] %d parameters (%d expert) over %d layers; "
            "attaches to the optimizer at the first optimizer step.",
            len(self._slots),
            sum(s.is_expert for s in self._slots),
            len(monitor_layers),
        )

    def attach_optimizer(self, optimizer) -> bool:
        """Wrap the innermost optimizer's ``_apply_optimize``; idempotent."""
        if self._optimizer is not None:
            return True
        if optimizer is None or not self._slots or self._attach_tried:
            return False
        self._attach_tried = True
        inner = _innermost_optimizer(optimizer)
        original = getattr(inner, _WRAP_ATTR, None)
        if not callable(original) or not hasattr(inner, "_master_weights"):
            if "attach" not in self._failed:
                self._failed.add("attach")
                logger.warning(
                    "[PaddleOptimHealthMonitor] %s has no _apply_optimize / "
                    "_master_weights; optim_health is disabled.",
                    type(inner).__name__,
                )
            return False

        info_map = getattr(inner, "_muon_param_info_map", None) or {}
        ratio_fn = getattr(inner, "_lr_ratio", None)
        # Only MuonShardingOptimizer keeps Muon / Hyperball tensors whole; any
        # other sharding wrapper cuts every tensor element-wise.
        keeps_whole = not any(
            "Sharding" in type(w).__name__
            and type(w).__name__ != "MuonShardingOptimizer"
            for w in _optimizer_chain(optimizer)
        )
        ratios = []
        for slot in self._slots:
            info = info_map.get(slot.name)
            # Per-expert readings need the whole [E, in, out] tensor on its owner,
            # which the sharding optimizer only guarantees for Muon / Hyperball
            # weights; a plain AdamW weight is cut element-wise across ranks.
            slot.per_expert = bool(
                keeps_whole
                and slot.is_expert
                and slot.num_experts > 0
                and info is not None
                and (info.use_muon or getattr(info, "use_hyperball", False))
            )
            try:
                ratios.append(
                    float(ratio_fn(slot.param)) if callable(ratio_fn) else 1.0
                )
            except Exception:
                ratios.append(1.0)
        try:
            self._plan = self._build_plan(ratios)
        except Exception as exc:
            # Deterministic from the model and the topology, so every rank fails
            # alike and none is left waiting in a collective the others skip.
            self._log_failure("reduction plan", exc)
            return False

        monitor = self

        def wrapped(*args, **kwargs):
            return monitor._apply_optimize(original, args, kwargs)

        setattr(inner, _WRAP_ATTR, wrapped)
        self._optimizer = inner
        self._wrapped = [(inner, _WRAP_ATTR)]
        logger.info(
            "[PaddleOptimHealthMonitor] Attached to %s._apply_optimize; %s.",
            type(inner).__name__,
            self._plan["describe"],
        )
        return True

    def _build_plan(self, ratios):
        """Reduction groups per bucket, plus the constant masks ``_emit`` uses."""
        sharding = ep = pp = mp = None
        try:
            from paddle.distributed import fleet

            hcg = fleet.get_hybrid_communicate_group()
        except Exception:
            hcg = None
        if hcg is not None:
            for name, size_fn, group_fn in (
                (
                    "sharding",
                    "get_sharding_parallel_world_size",
                    "get_sharding_parallel_group",
                ),
                (
                    "pp",
                    "get_pipe_parallel_world_size",
                    "get_pipe_parallel_group",
                ),
                (
                    "mp",
                    "get_model_parallel_world_size",
                    "get_model_parallel_group",
                ),
            ):
                try:
                    if getattr(hcg, size_fn)() > 1:
                        group = getattr(hcg, group_fn)()
                        if name == "sharding":
                            sharding = group
                        elif name == "pp":
                            pp = group
                        else:
                            mp = group
                except Exception:
                    pass
        if any(s.is_expert for s in self._slots):
            # The group HybridParallelClipGrad sums the expert norm over; the
            # model's own EP group is the fallback when the hcg has no EP axis.
            try:
                if (
                    hcg is not None
                    and hcg.get_moe_sharding_parallel_world_size() > 0
                ):
                    ep = hcg.get_expert_parallel_group()
            except Exception:
                ep = None
            if ep is None:
                try:
                    from paddlefleet.parallel_state import (
                        get_expert_model_parallel_group,
                    )

                    ep = get_expert_model_parallel_group(
                        check_initialized=False
                    )
                except Exception:
                    ep = None
        ep = ep if _nranks(ep) > 1 else None
        pp = pp if _nranks(pp) > 1 else None

        buckets = {}
        for slot in self._slots:
            buckets.setdefault(slot.bucket, []).append(slot)
        plan_buckets = []
        for key in sorted(buckets):
            members = buckets[key]
            groups = [_color_group(members[0].param, sharding)]
            if key[1]:
                groups.append(ep)
            if key[2]:
                groups.append(mp)
            groups = [g for g in groups if _nranks(g) > 1]
            if groups:
                plan_buckets.append(
                    (
                        paddle.to_tensor(
                            [s.index for s in members], dtype="int64"
                        ),
                        groups,
                    )
                )

        expert_slots = [s for s in self._slots if s.per_expert]
        if len({s.num_experts for s in expert_slots}) > 1:
            expert_slots = []
        expert_owner = None
        if expert_slots:
            owner = _color_group(expert_slots[0].param, sharding)
            expert_owner = owner if _nranks(owner) > 1 else None

        def mask(pred):
            return paddle.to_tensor(
                [float(pred(s)) for s in self._slots], dtype="float32"
            )

        return {
            "buckets": plan_buckets,
            "pp": pp,
            "ep": ep,
            "expert_slots": expert_slots,
            "expert_owner": expert_owner,
            "lr_ratio": paddle.to_tensor(ratios, dtype="float32"),
            "dense_mask": mask(lambda s: s.in_total and not s.is_expert),
            "expert_mask": mask(lambda s: s.in_total and s.is_expert),
            "describe": (
                f"{len(plan_buckets)} reduction buckets; sharding={_nranks(sharding)} "
                f"ep={_nranks(ep)} pp={_nranks(pp)} mp={_nranks(mp)}; "
                f"{len(expert_slots)} grouped-expert weights"
            ),
        }

    def remove_hooks(self):
        super().remove_hooks()
        if (
            self._optimizer is not None
            and _WRAP_ATTR in self._optimizer.__dict__
        ):
            del self._optimizer.__dict__[_WRAP_ATTR]
        self._optimizer = None
        self._wrapped = []
        self._plan = None
        self._pending = None

    # ------------------------------------------------------------------
    # Optimizer path: local sums only, no collective
    # ------------------------------------------------------------------

    def _log_failure(self, where: str, exc: Exception) -> None:
        if where not in self._failed:
            self._failed.add(where)
            logger.warning(
                "[PaddleOptimHealthMonitor] %s failed; affected steps are not logged: %r",
                where,
                exc,
            )

    def _apply_optimize(self, original, args, kwargs):
        if not self._interval_reached():
            return original(*args, **kwargs)
        params_grads = kwargs.get("params_grads")
        if params_grads is None and len(args) >= 3:
            params_grads = args[2]
        try:
            with paddle.no_grad():
                lr = float(self._optimizer.get_lr())
                captured = self._before_step(params_grads or [])
        except Exception as exc:
            self._log_failure("pre-step capture", exc)
            return original(*args, **kwargs)
        result = original(*args, **kwargs)
        try:
            with paddle.no_grad():
                self._pending = self._after_step(captured, lr)
        except Exception as exc:
            self._log_failure("post-step capture", exc)
            self._pending = None
        return result

    def _before_step(self, params_grads):
        masters = getattr(self._optimizer, "_master_weights", None) or {}
        captured = []
        for param, grad in params_grads:
            if grad is None:
                continue
            slot = self._slot_by_name.get(param.name)
            if slot is None:
                continue
            weight = masters.get(param.name, param)
            if not weight._is_initialized() or not grad._is_initialized():
                continue
            per_expert = slot.per_expert and len(weight.shape) == 3
            captured.append(
                (
                    slot,
                    param,
                    snapshot(weight),
                    grad_square(grad, per_expert),
                    per_expert,
                )
            )
        return captured

    def _after_step(self, captured, lr):
        # Looked up again: on the very first step the optimizer creates the fp32
        # master inside the call, as an exact upcast of the bf16 weight the
        # snapshot was taken from, and only the master holds the exact update.
        masters = getattr(self._optimizer, "_master_weights", None) or {}
        rows, index, experts = [], [], {}
        for slot, param, before, grad_sq, per_expert in captured:
            weight = masters.get(param.name, param)
            sums = step_sums(grad_sq, before, weight, per_expert)
            if per_expert:
                experts[slot.index] = sums
                sums = sums.sum(axis=0)
            rows.append(sums)
            index.append(slot.index)
        return {"rows": rows, "index": index, "experts": experts, "lr": lr}

    # ------------------------------------------------------------------
    # Flush: agree, reduce, derive, one D2H
    # ------------------------------------------------------------------

    def _flush_buffers(self) -> None:
        pending, self._pending = self._pending, None
        if (
            self.sampled_this_step
            and self._attach_tried
            and self._plan is not None
        ):
            try:
                with paddle.no_grad():
                    if self._everyone_captured(pending is not None):
                        self._emit(pending)
            except Exception as exc:
                self._log_failure("reduce/emit", exc)
        super()._flush_buffers()

    @staticmethod
    def _everyone_captured(ok: bool) -> bool:
        import paddle.distributed as dist

        if not dist.is_initialized() or dist.get_world_size() <= 1:
            return ok
        flag = paddle.to_tensor([1 if ok else 0], dtype="int32")
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        return bool(int(flag.numpy()[0]))

    def _global_sums(self, pending) -> paddle.Tensor:
        import paddle.distributed as dist

        local = paddle.zeros([len(self._slots), NUM_SUMS], dtype="float32")
        if pending["rows"]:
            local = paddle.scatter(
                local,
                paddle.to_tensor(pending["index"], dtype="int64"),
                paddle.stack(pending["rows"]),
                overwrite=True,
            )
        out = local
        for index, groups in self._plan["buckets"]:
            part = paddle.gather(local, index)
            for group in groups:
                dist.all_reduce(part, group=group)
            out = paddle.scatter(out, index, part, overwrite=True)
        return out

    def _expert_sums(self, pending):
        """``[S, E_global, 4]`` for the grouped-expert weights, or ``None``."""
        import paddle.distributed as dist

        slots = self._plan["expert_slots"]
        if not slots:
            return None
        local = paddle.stack(
            [
                pending["experts"][s.index]
                if s.index in pending["experts"]
                else paddle.zeros([s.num_experts, NUM_SUMS], dtype="float32")
                for s in slots
            ]
        )
        if self._plan["expert_owner"] is not None:
            dist.all_reduce(local, group=self._plan["expert_owner"])
        if self._plan["ep"] is not None:
            parts: list = []
            dist.all_gather(parts, local, group=self._plan["ep"])
            local = paddle.concat(parts, axis=1)
        return local

    def _emit(self, pending) -> None:
        import paddle.distributed as dist

        plan = self._plan
        sums = self._global_sums(pending)
        lr = pending["lr"]
        per_param = param_metrics(sums, plan["lr_ratio"] * lr)

        # Stage totals, then summed over the pipeline, for reconciliation with
        # the clip's global grad norm.
        totals = paddle.stack(
            [
                (sums[:, GRAD_SQ] * plan["dense_mask"]).sum(),
                (sums[:, GRAD_SQ] * plan["expert_mask"]).sum(),
            ]
        )
        if plan["pp"] is not None:
            dist.all_reduce(totals, group=plan["pp"])

        expert_sums = self._expert_sums(pending)
        per_expert = (
            expert_metrics(expert_sums) if expert_sums is not None else {}
        )

        flat = [paddle.sqrt(totals)]
        flat += [per_param[name].reshape([-1]) for name in PARAM_METRICS]
        flat += [per_expert[name].reshape([-1]) for name in per_expert]
        host = paddle.concat(flat).astype("float64").numpy().tolist()

        model = f"{self.METRIC_PREFIX}/model"
        dense_norm, expert_norm = host[0], host[1]
        out = {
            f"{model}/grad_norm_dense": dense_norm,
            f"{model}/grad_norm_expert": expert_norm,
            f"{model}/grad_norm_total": math.hypot(dense_norm, expert_norm),
            f"{model}/lr": lr,
        }
        clip_norm = self._clip_global_norm()
        if clip_norm is not None:
            out[f"{model}/clip_grad_norm"] = clip_norm

        pos, n = 2, len(self._slots)
        for name in PARAM_METRICS:
            for slot, value in zip(
                self._slots, host[pos : pos + n], strict=True
            ):
                out[self._slot_key(slot, name)] = value
            pos += n
        expert_slots = plan["expert_slots"]
        for name in per_expert:
            column = host[pos : pos + len(expert_slots)]
            for slot, value in zip(expert_slots, column, strict=True):
                out[self._slot_key(slot, name)] = value
            pos += len(expert_slots)

        training_logs.update(
            **{
                k: v
                for k, v in out.items()
                if math.isfinite(v) and self.family_allows(k)
            }
        )

    def _slot_key(self, slot: _Slot, metric: str) -> str:
        if slot.layer_idx is None:
            return f"{self.METRIC_PREFIX}/model/{slot.metric_base}/{metric}"
        return self._layer_key(slot.layer_idx, f"{slot.metric_base}_{metric}")

    def _clip_global_norm(self) -> float | None:
        """The clip's own global grad norm of this step, when it exposes one."""
        clip = getattr(self._optimizer, "_grad_clip", None)
        stat = getattr(clip, "stat", None) if clip is not None else None
        if isinstance(stat, dict):
            value = stat.get("global_grad_norm")
            if isinstance(value, int | float):
                return float(value)
        return None


def setup_optim_monitor(
    model,
    log_per_layer=True,
    log_global=False,
    monitor_interval=1,
    verbose=False,
    monitor_dict=None,
    exclude_families=None,
):
    monitor = PaddleOptimHealthMonitor(
        log_per_layer=log_per_layer,
        log_global=log_global,
        monitor_interval=monitor_interval,
        verbose=verbose,
        exclude_families=exclude_families,
    )
    monitor.register_hooks(model)
    if monitor_dict is not None:
        monitor_dict["optim_health"] = monitor
    return model
