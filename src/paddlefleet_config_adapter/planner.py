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

"""Parallelism planning: keep the dims frozen, or shrink EP / PP to fit.

:func:`plan_parallelism` picks one of two planners from the options:

* frozen (the default, and ``--test-performance``) -- every parallel
  dimension stays as the source declares it; an incompatible target scale is
  rejected with the list of legal node counts rather than forced through.
  Shrinking changes the model structure, so it is never a silent default.
* shrink (``--test-accuracy`` without ``--test-performance``) -- candidates
  are tried in order of increasing structural intrusion and the first
  feasible one wins::

      Case A (dims already legal) < EP-only < PP-only < EP+PP joint

  Every candidate must pass the communication-group constraints (C1..C5) and
  the model-structure constraints (M1/M2 for EP, M3/M4/M5 for PP).  Shrinking
  EP scales the routed-expert count, shrinking PP scales
  ``num_hidden_layers`` and realigns VPP / empty tail layers -- those writes
  land in a *separate* copy of ``model_config.json``, never in the source.

  Within a tier the rule is "shrink as little as possible", except for the
  joint tier, where the two axes follow the shrink priority ``EP > PP``: EP
  goes down to its floor first and PP absorbs only the remainder.

TP, SEP and CP are never shrunk.  A smaller TP raises per-card memory and
risks OOM.  A smaller CP is only safe together with a proportionally smaller
``max_seq_length`` -- on its own it *lengthens* the per-card sequence slice --
and rewriting the sequence length silently is not acceptable: it changes what
the run measures, and pre-packed datasets are tokenized at one fixed sequence
length, so a config with a different ``max_seq_length`` cannot read them at
all.  Trading sequence length for cards therefore stays an explicit,
user-driven decision via ``--scale-seq-length``; :func:`_cp_blocks_c4` makes
sure the diagnostic points at it whenever CP is what blocks the shrink.

No dimension that is > 1 in the source is ever reduced to 1, because
that would delete the communication group under test.  VPP is the exception:
whichever planner ran, :func:`enforce_vpp_limit` switches it off when the
final PP is too small for an interleaved schedule.
"""

from __future__ import annotations

import re

from .constraints import (
    MIN_PARALLEL_DEGREE,
    MIN_PP_FOR_VPP,
    WIDTH_ALIGN,
    align_layers,
    check_ep_shrink,
    check_hardware,
    check_pp_shrink,
    ep_candidates,
    min_aligned_width,
    min_legal_layers,
    min_routed_experts,
    min_seq_length,
    min_shrink_cards,
    pp_candidates,
)
from .field_spec import (
    describe_missing,
    nested_find,
    nested_get,
    resolve_encoder_fields,
    resolve_fields,
)
from .layer_fields import (
    LAYER_FIELDS,
    _locate_layer_field,
    effective_mtp_layers,
    min_layers_for_attention_families,
    plan_layer_field_shrink,
)
from .plan import ParallelismPlan
from .topology import TopologyValidator

# Matches a communication-group rejection ("C2 不满足：..."), so the diagnostic
# can put the more actionable model-structure reasons first.
_CONSTRAINT_RE = re.compile(r"C[1-5]\s*不满足")

VPP_FIELD = "virtual_pipeline_model_parallel_size"

#: YAML / model_config keys for the sequence-length shrink.  ``core.SEQ_FIELD``
#: is the same literal, but core imports planner (not the reverse), so the
#: constant is redeclared here rather than imported back.
SEQ_FIELD = "max_seq_length"
WINDOW_FIELD = "csa_window_size"

#: 是否在 --shrink-structure 下把 max_seq_length 缩到下限。
#: 暂时关闭：缩短 max_seq_length 会按更小的窗口把语料切成更多 part（seq 缩 8x ->
#: 数据序列 part 数约涨 64x），导致启动期 data_seq_build_index 从十几秒暴涨到近 20
#: 分钟，反而拖慢整轮 FC 测试。在没有同步缩小数据集之前先不缩序列长度；之后做数据集
#: 瘦身优化时再把它打开。层数/专家数/MoE 宽度等结构瘦身不受影响，照常进行。
SHRINK_MAX_SEQ_LENGTH = False

#: How many candidate rejections the diagnostic prints, and how many of those
#: slots are reserved for communication-group reasons.  C4 ("sharding % CP")
#: is regularly *the* blocking reason, and since CP never shrinks it is also
#: the one the user has to act on -- it must not be crowded out by the
#: model-structure lines that sort first.
DETAIL_LIMIT = 10
MIN_TOPOLOGICAL_DETAIL = 2


def _vpp_off_reason(vpp_old, pp):
    """Why VPP must be 1 at this PP."""
    return (
        f"PP={pp} 时不能开虚拟流水：框架断言 VPP>1 需要 PP>"
        f"{MIN_PP_FOR_VPP - 1}"
        f"（virtual pipeline must run under pp degree > 2），"
        f"VPP {vpp_old} -> 1"
    )


def _encoder_ep_legal(enc_experts, enc_topk, ep_new):
    """Is the (un-shrunk) HyperEncoder expert pool legal under ``ep_new``?

    HyperBody runs its encoder tower on the SAME global EP as the decoder. EP
    adaptation keeps the encoder's expert count fixed on purpose (see the
    field_spec module doc and :func:`plan_structure_minimize`): the encoder pool
    is small, so shrinking it with EP would only raise the smallest feasible EP
    and limit how far cards can be reduced. The one thing we must still check is
    that the shared EP keeps the fixed pool legal --
    ``HyperEncoderProvider._check_divisibility`` asserts ``n_routed_experts %
    ep == 0`` and the router needs at least top-k experts.

    ``enc_experts is None`` means there is no ``encoder_config`` (every
    non-HyperBody model), so the check is vacuously true.
    """
    if enc_experts is None:
        return True
    return enc_experts % ep_new == 0 and enc_experts >= (enc_topk or 1)


def plan_parallelism(
    config, dims, target_cards, cards_per_node, options, context=None
):
    """Plan the final parallel dims. Returns ``(plan, error_or_None)``."""
    if options.freeze_parallel:
        plan, err = plan_frozen(
            dims,
            target_cards,
            cards_per_node,
            frozen_by_performance=options.test_performance,
        )
    else:
        plan, err = ShrinkPlanner().plan(
            config, dims, target_cards, cards_per_node, context
        )
    if err:
        return None, err
    enforce_vpp_limit(config, plan)
    return plan, None


def enforce_vpp_limit(config, plan):
    """Turn VPP off when the planned PP cannot run an interleaved schedule.

    :data:`~.constraints.MIN_PP_FOR_VPP` is a hard framework assert, and the
    PP-shrink planner already respects it, which leaves the plans that never
    touch PP -- a frozen plan, an EP-only shrink, a source that already fits
    the target -- carrying whatever VPP the source declared.  Dropping it to 1
    cannot break the layer alignment: the number of segments goes from
    ``PP * VPP`` down to ``PP``, which divides it.
    """
    if plan.pp >= MIN_PP_FOR_VPP:
        return
    planned = {key: value for key, value, _reason in plan.yaml_changes}
    vpp = planned.get(VPP_FIELD, config.get(VPP_FIELD, 1) or 1)
    if int(vpp) <= 1:
        return
    plan.yaml_changes.append((VPP_FIELD, 1, _vpp_off_reason(vpp, plan.pp)))


STRUCTURE_YAML_KEYS = ("num_empty_layers_add_in_tail", VPP_FIELD)


def plan_structure_minimize(config, model_config, plan, protected_yaml_keys=()):
    """Force the model structure down to its legal floor, in place.

    Runs AFTER :func:`plan_parallelism`, on the plan's *final* dims, so the
    floors use the PP / VPP / EP the shrink (or freeze) actually landed on.
    Only invoked under ``--shrink-structure``: the caller leaves the default
    path untouched, so a run without the switch is byte-for-byte unchanged.

    Four structural axes are pushed to their floor, never grown:

    * ``num_hidden_layers`` -> ``max(pp*vpp, first_k_dense_replace,``
      ``attention-family floor)`` (see :func:`~.constraints.min_legal_layers`);
    * routed experts -> :func:`~.constraints.min_routed_experts` for the final
      EP / grouping;
    * MoE widths (``moe_intermediate_size`` and, when latent MoE is on,
      ``moe_latent_size``) -> the smallest 128-aligned value
      (:func:`~.constraints.min_aligned_width`);
    * ``max_seq_length`` -> :func:`~.constraints.min_seq_length` for the final
      CP and sliding window, unless the config is truly multimodal or the user
      pinned the length (``protected_yaml_keys``).  **目前由模块常量
      ``SHRINK_MAX_SEQ_LENGTH`` 关闭**（缩序列长度会使数据序列 part 数暴涨、拖慢
      启动），其余三条结构轴照常缩。

    ``protected_yaml_keys`` is the set of YAML keys the user already pinned
    (via ``--set`` or ``--scale-seq-length``); the sequence shrink skips any
    key it finds there.  It defaults to empty for backward compatibility.

    Writes are merged into the plan *by key*: any planner write for the same
    target (layer count, expert count, a per-layer list, VPP or the empty
    tail) is dropped and replaced, so the two decisions never fight.  A target
    this step does not touch keeps whatever the planner queued.  Returns
    ``error_or_None``.
    """
    if model_config is None:
        return (
            "强制缩结构需要读取 model_config.json，但本次运行拿不到"
            "（model_name_or_path 指向的配置缺失或无法解析）"
        )

    resolved, missing = resolve_fields(model_config)
    if "num_hidden_layers" in missing:
        return describe_missing(
            "num_hidden_layers", missing["num_hidden_layers"]
        )

    layout, err = ShrinkPlanner._pipeline_layout(
        config, resolved, missing, model_config
    )
    if err:
        return err

    # Final values the plan already decided (later writes win over earlier).
    final_yaml = {key: value for key, value, _reason in plan.yaml_changes}
    final_json = {key: value for key, value, _reason in plan.json_changes}

    pp = int(plan.pp)
    vpp = int(final_yaml.get(VPP_FIELD, layout["vpp"]) or 1)
    tail = int(
        final_yaml.get("num_empty_layers_add_in_tail", layout["tail"]) or 0
    )
    cur_layers = int(final_json.get(layout["layers_key"], layout["layers"]))

    new_yaml_changes = []
    new_json_changes = []
    drop_json_keys = set()

    # ---- layers ---------------------------------------------------------
    family_min = min_layers_for_attention_families(model_config)
    layers_min = min_legal_layers(pp, vpp, layout["first_k"], family_min)
    if layers_min < cur_layers:
        if pp * vpp == 1:
            # A single segment: head + layers + tail is a multiple of 1 whatever
            # the tail is, so there is nothing to realign.  ``align_layers``
            # would refuse here -- its ``pp_new < MIN_PARALLEL_DEGREE`` guard
            # exists for PP shrinking, not for a source already at PP=1.
            vpp_new, tail_new = vpp, tail
        else:
            vpp_new, tail_new = align_layers(
                layers_min,
                layout["head"],
                pp,
                vpp,
                min_tail=layout["min_tail"],
            )
            if vpp_new is None:
                return (
                    f"强制缩结构：层数缩到 {layers_min} 后无法对齐到 "
                    f"PP={pp} × VPP={vpp}"
                )
        layer_changes, err = plan_layer_field_shrink(
            model_config, layout["layers"], layers_min, layout["mtp"]
        )
        if err:
            return err
        new_json_changes.append(
            (
                layout["layers_key"],
                layers_min,
                f"强制缩结构：层数缩到合法下限 {layers_min}"
                f"（= max(PP×VPP={pp}×{vpp}, first_k={layout['first_k']}, "
                f"注意力家族下限={family_min})），{cur_layers} -> {layers_min}",
            )
        )
        new_json_changes.extend(layer_changes)
        drop_json_keys.add(layout["layers_key"])
        for field in LAYER_FIELDS:
            # Same nested scope as plan_layer_field_shrink: drop the per-layer
            # key by its real (possibly decoder_config.*) path so a user --set
            # protection and the merge-by-key dedup both match.
            _value, key = _locate_layer_field(model_config, field.aliases)
            if key is not None:
                drop_json_keys.add(key)
        if vpp_new != vpp:
            new_yaml_changes.append(
                (
                    VPP_FIELD,
                    vpp_new,
                    f"强制缩结构后重新对齐 VPP {vpp} -> {vpp_new}",
                )
            )
        if tail_new != tail:
            new_yaml_changes.append(
                (
                    "num_empty_layers_add_in_tail",
                    tail_new,
                    f"强制缩结构后调整尾部空层，把总层数对齐到 PP×VPP："
                    f"{tail} -> {tail_new}",
                )
            )

    # ---- routed experts -------------------------------------------------
    num_experts = (
        resolved["num_experts"].value if "num_experts" in resolved else None
    )
    topk = (
        resolved["num_experts_per_tok"].value
        if "num_experts_per_tok" in resolved
        else None
    )
    if num_experts is not None and topk is not None:
        experts_key = resolved["num_experts"].writeback_key
        # n_group / topk_group are not in FIELD_SPECS; read them flat-or-nested
        # (HyperBody keeps them under decoder_config) with the same precedence.
        n_group = int(nested_get(model_config, "n_group", 1) or 1)
        topk_group = int(
            nested_get(model_config, "topk_group", n_group) or n_group
        )
        experts_min = min_routed_experts(plan.ep, n_group, topk_group, topk)
        cur_experts = int(final_json.get(experts_key, num_experts))
        if experts_min < cur_experts:
            new_json_changes.append(
                (
                    experts_key,
                    experts_min,
                    f"强制缩结构：路由专家数缩到合法下限 {experts_min}"
                    f"（EP={plan.ep}, n_group={n_group}, "
                    f"topk_group={topk_group}, top-k={topk}），"
                    f"{cur_experts} -> {experts_min}",
                )
            )
            drop_json_keys.add(experts_key)

    # ---- MoE widths -----------------------------------------------------
    # Pure architectural widths: shrinking them only rescales weight shapes
    # (and the derived shared-expert width), with no dependent identity field.
    # The 128 floor keeps the fp8 / DeepGEMM / MoonEP block-scale tiles legal.
    dispatcher = str(
        nested_get(model_config, "moe_token_dispatcher_type") or ""
    )
    intermediate_sharding = dispatcher in ("allgather", "ringmoe")
    # moe_intermediate_size: with intermediate-dim sharding the floor rises to
    # EP*128, which satisfies both ``% EP == 0`` and (under fp8) the
    # ``(//EP) % 128 == 0`` rule.  deepep / hybridep slice whole experts, so
    # the plain 128 alignment is enough there.
    moe_width_align = (
        WIDTH_ALIGN * plan.ep
        if (intermediate_sharding and plan.ep > 1)
        else WIDTH_ALIGN
    )
    for key, field_align in (
        ("moe_intermediate_size", moe_width_align),
        ("moe_latent_size", WIDTH_ALIGN),
    ):
        # Flat-or-nested: HyperBody keeps moe_intermediate_size under
        # decoder_config while moe_latent_size stays top level, so resolve each
        # key's own path and write the shrink back to that exact location.
        raw_width, width_path = nested_find(model_config, key)
        if width_path is None:
            continue
        raw_width = final_json.get(width_path, raw_width)
        # Guard against a field that is declared but not a usable width:
        # ``moe_latent_size: null`` (MiniMax-M2 default) or a list-valued
        # ``moe_intermediate_size`` (ernie). Skip it rather than let int()
        # raise and escape the module's (False, 中文原因) contract.
        if not isinstance(raw_width, int) or isinstance(raw_width, bool):
            continue
        cur_width = raw_width
        if cur_width <= 0:
            continue
        floor = min_aligned_width(cur_width, field_align)
        if floor < cur_width:
            new_json_changes.append(
                (
                    width_path,
                    floor,
                    f"强制缩结构：{key} 缩到 128 对齐下限 {floor}"
                    f"（align={field_align}），{cur_width} -> {floor}",
                )
            )
            drop_json_keys.add(width_path)

    # ---- HyperEncoder (encoder_config) ----------------------------------
    # Policy split for the HyperEncoder tower (see field_spec module doc):
    #   * EP ADAPTATION never shrinks the encoder -- it only *validates* that
    #     the fixed encoder expert pool stays legal under the shared global EP
    #     (ShrinkPlanner._encoder_ep_rejection); shrinking the encoder's small
    #     pool there would only tighten the min-EP and so limit how far cards
    #     can be reduced.
    #   * --shrink-structure (this function) DOES actively shrink the encoder,
    #     because a CI run whose whole point is to be tiny should not keep the
    #     encoder's full expert count + MoE width resident.  The encoder shares
    #     the decoder's final EP (plan.ep), and has no per-layer list fields, so
    #     this is a self-contained experts + MoE-width floor.  The encoder's own
    #     num_hidden_layers is intentionally left untouched: the HyperEncoder is
    #     not pipeline-split and its minimum legal depth is not asserted anywhere
    #     we could verify, so shrinking it is deferred rather than guessed.
    enc_resolved, _enc_missing = resolve_encoder_fields(model_config)
    if "num_experts" in enc_resolved and "num_experts_per_tok" in enc_resolved:
        enc_section = model_config.get("encoder_config", {})
        enc_experts = enc_resolved["num_experts"].value
        enc_topk = enc_resolved["num_experts_per_tok"].value
        enc_experts_key = enc_resolved["num_experts"].writeback_key
        enc_n_group = int(enc_section.get("n_group", 1) or 1)
        enc_topk_group = int(
            enc_section.get("topk_group", enc_n_group) or enc_n_group
        )
        enc_experts_min = min_routed_experts(
            plan.ep, enc_n_group, enc_topk_group, enc_topk
        )
        cur_enc_experts = int(final_json.get(enc_experts_key, enc_experts))
        if enc_experts_min < cur_enc_experts:
            new_json_changes.append(
                (
                    enc_experts_key,
                    enc_experts_min,
                    f"强制缩结构：HyperEncoder 路由专家数缩到合法下限 "
                    f"{enc_experts_min}（共享 EP={plan.ep}, "
                    f"n_group={enc_n_group}, topk_group={enc_topk_group}, "
                    f"top-k={enc_topk}），{cur_enc_experts} -> {enc_experts_min}",
                )
            )
            drop_json_keys.add(enc_experts_key)
        enc_width, enc_width_path = nested_find(
            enc_section, "moe_intermediate_size"
        )
        # enc_section is the encoder sub-dict, so nested_find returns a bare key
        # ("moe_intermediate_size"); re-prefix it to the full encoder path.
        if (
            enc_width_path is not None
            and isinstance(enc_width, int)
            and not isinstance(enc_width, bool)
            and enc_width > 0
        ):
            enc_width_key = f"encoder_config.{enc_width_path}"
            cur_enc_width = final_json.get(enc_width_key, enc_width)
            enc_floor = min_aligned_width(cur_enc_width, WIDTH_ALIGN)
            if enc_floor < cur_enc_width:
                new_json_changes.append(
                    (
                        enc_width_key,
                        enc_floor,
                        f"强制缩结构：HyperEncoder moe_intermediate_size 缩到 "
                        f"128 对齐下限 {enc_floor}，{cur_enc_width} -> {enc_floor}",
                    )
                )
                drop_json_keys.add(enc_width_key)

    # ---- max_seq_length -------------------------------------------------
    # Attention / sequence compute scales with the sequence length.  Shrink it
    # unless the config is truly multimodal (patch ids present) or the user
    # already pinned the length.  ``use_v2_tokenizer_and_bypass_mm`` turns the
    # multimodal token path off, so a config carrying patch ids purely as
    # reserved vocab (e.g. ScalingLadder T6) is still text-only and safe to
    # shrink.
    cur_seq = config.get(SEQ_FIELD)
    patch_keys = ("image_patch_id", "video_patch_id", "audio_patch_id")
    bypass_mm = bool(
        config.get("use_v2_tokenizer_and_bypass_mm")
        or model_config.get("use_v2_tokenizer_and_bypass_mm")
    )
    is_multimodal = (not bypass_mm) and any(
        config.get(k) is not None for k in patch_keys
    )
    if (
        SHRINK_MAX_SEQ_LENGTH
        and not is_multimodal
        and SEQ_FIELD not in protected_yaml_keys
        and isinstance(cur_seq, int)
        and cur_seq > 0
    ):
        window = config.get(WINDOW_FIELD)
        if not isinstance(window, int):
            window = model_config.get(WINDOW_FIELD)
        floor = min_seq_length(
            plan.cp, window if isinstance(window, int) else None
        )
        if floor < cur_seq:
            new_yaml_changes.append(
                (
                    SEQ_FIELD,
                    floor,
                    f"强制缩结构：序列长度缩到下限 {floor}"
                    f"（对齐 lcm(128, 2×CP={plan.cp})，不低于 "
                    f"csa_window_size），{cur_seq} -> {floor}",
                )
            )
            if not config.get("online_generate_data"):
                plan.warnings.append(
                    "强制缩结构：改写 max_seq_length，但该配置不是 "
                    "online_generate_data；预打包数据集按固定序列长度 "
                    "tokenize，改长度后可能读不进来"
                )

    if not new_yaml_changes and not new_json_changes:
        plan.warnings.append(
            "强制缩结构：模型结构已在下限"
            "（层数/专家数/宽度都无可再缩），未做改动"
        )
        return None

    # Merge by key: replace only the targets this step rewrote, so unrelated
    # planner writes (e.g. a VPP / tail the planner already set) survive.
    plan.json_changes = [
        item for item in plan.json_changes if item[0] not in drop_json_keys
    ]
    plan.json_changes.extend(new_json_changes)
    replaced_yaml = {key for key, _value, _reason in new_yaml_changes}
    plan.yaml_changes = [
        item for item in plan.yaml_changes if item[0] not in replaced_yaml
    ]
    plan.yaml_changes.extend(new_yaml_changes)

    # Only weight-shape changes (layers / experts / widths, all JSON) make the
    # original checkpoint stale.  A pure max_seq_length shrink leaves every
    # weight shape intact, so it must not drop the checkpoint reference.
    if new_json_changes:
        plan.structure_shrunk = True
    plan.note = (plan.note + "；" if plan.note else "") + "强制缩结构到合法下限"
    return None


def plan_frozen(dims, target_cards, cards_per_node, frozen_by_performance=True):
    """Keep every parallel dim; fail when the target scale needs changes.

    ``frozen_by_performance`` only shapes the diagnostics: the dims are
    frozen either because ``--test-performance`` demanded it, or because no
    switch was given and shrinking (a model-structure change) requires the
    explicit ``--test-accuracy`` opt-in.
    """
    tp, pp, ep, cp, sep = dims
    validator = TopologyValidator(target_cards, cards_per_node)
    ok, message, _details = validator.validate(tp, pp, ep, cp, sep)
    if not ok:
        if frozen_by_performance:
            why = (
                "--test-performance 只调整 sharding 与 global_batch_size，"
                "不修改 TP/PP/EP/CP/SEP 和 acc，"
                "因此目标机器规模必须与源并行度兼容。"
            )
            hint = (
                "换用上面列出的合法节点数；确要缩小 EP/PP 请去掉 "
                "--test-performance 并加 --test-accuracy"
            )
        else:
            why = (
                "默认不缩小 EP/PP：缩容会等比改写专家数 / 层数，"
                "属于模型结构变更，不应默默发生，"
                "因此目标机器规模必须与源并行度兼容。"
            )
            hint = (
                "换用上面列出的合法节点数，"
                "或加 --test-accuracy 显式允许缩小 EP/PP"
            )
        return None, f"{message}\n  说明：{why}\n  建议：{hint}"
    note = (
        "并行度全部保持不变（--test-performance 冻结 TP/PP/EP/CP/SEP 与 acc）"
        if frozen_by_performance
        else "并行度全部保持不变（默认冻结；缩小 EP/PP 需显式加 "
        "--test-accuracy）"
    )
    return (
        ParallelismPlan(tp, pp, ep, cp, sep, note=note),
        None,
    )


class ShrinkPlanner:
    """Shrinks EP / PP just enough to fit the target scale."""

    def plan(self, config, dims, target_cards, cards_per_node, context=None):
        """Plan the final dims, shrinking EP / PP only as much as needed."""
        tp, pp, ep, cp, sep = dims
        context = context or {}

        ok, why = check_hardware(
            target_cards, cards_per_node, tp, pp, ep, cp, sep
        )
        if not ok:
            return None, why

        validator = TopologyValidator(target_cards, cards_per_node)

        # ---- Case A: the source dims already fit the target scale --------
        ok, _msg, _details = validator.validate(tp, pp, ep, cp, sep)
        if ok:
            return (
                ParallelismPlan(
                    tp,
                    pp,
                    ep,
                    cp,
                    sep,
                    note="源并行度已满足目标机器规模的通信组约束，未做缩容",
                ),
                None,
            )

        # ---- Everything below needs the model structure ------------------
        model_config = context.get("model_config")
        if model_config is None:
            return None, (
                f"需要缩小 EP/PP 才能适配 {target_cards} 卡，"
                f"但读不到 model_config.json："
                f"{context.get('model_config_error') or '未提供'}"
            )

        resolved, missing = resolve_fields(model_config)

        def value_of(name):
            field = resolved.get(name)
            return field.value if field is not None else None

        num_experts = value_of("num_experts")
        topk = value_of("num_experts_per_tok")
        experts_key = (
            resolved["num_experts"].writeback_key
            if "num_experts" in resolved
            else "n_routed_experts"
        )
        # HyperEncoder expert pool, resolved from encoder_config only. Used to
        # VALIDATE (never shrink) every candidate EP: the encoder shares the
        # decoder's global EP, so an ep_new that does not divide the fixed
        # encoder pool would crash HyperEncoderProvider at build time. No
        # encoder_config -> enc_experts stays None -> _encoder_ep_legal passes.
        enc_resolved, _enc_missing = resolve_encoder_fields(model_config)
        enc_experts = (
            enc_resolved["num_experts"].value
            if "num_experts" in enc_resolved
            else None
        )
        enc_topk = (
            enc_resolved["num_experts_per_tok"].value
            if "num_experts_per_tok" in enc_resolved
            else None
        )
        # MoE detection is structural: EP > 1 means expert parallelism is on,
        # regardless of any use_moe flag (MiniMax only sets EP + the flag).
        can_shrink_ep = (
            ep > MIN_PARALLEL_DEGREE
            and num_experts is not None
            and topk is not None
        )

        # ---- Tier 1: EP only ---------------------------------------------
        rejections = []
        if can_shrink_ep:
            pool = []
            for ep_new in ep_candidates(ep, tp, sep):
                ok, why, experts_new = check_ep_shrink(
                    ep, ep_new, num_experts, topk
                )
                if not ok:
                    rejections.append(f"EP {ep} -> {ep_new}：{why}")
                    continue
                ok, why, _d = validator.validate(tp, pp, ep_new, cp, sep)
                if not ok:
                    rejections.append(
                        f"EP {ep} -> {ep_new}：{_first_constraint(why)}"
                    )
                    continue
                if not _encoder_ep_legal(enc_experts, enc_topk, ep_new):
                    rejections.append(
                        f"EP {ep} -> {ep_new}：HyperEncoder 专家池 "
                        f"{enc_experts} 在该 EP 下不合法"
                        f"（% EP != 0 或 < top-k={enc_topk}）；"
                        f"EP 适配不缩 encoder"
                    )
                    continue
                pool.append((ep_new, experts_new))

            if pool:
                # Largest surviving EP = least intrusive within the tier.
                ep_new, experts_new = max(pool, key=lambda item: item[0])
                return (
                    ParallelismPlan(
                        tp,
                        pp,
                        ep_new,
                        cp,
                        sep,
                        yaml_changes=[
                            (
                                "expert_model_parallel_size",
                                ep_new,
                                f"缩小 EP {ep} -> {ep_new} 以满足 "
                                f"{target_cards} 卡下的 C2/C3 约束"
                                f"（PP/TP/CP/SEP 不变）",
                            )
                        ],
                        json_changes=[
                            (
                                experts_key,
                                experts_new,
                                f"随 EP {ep} -> {ep_new} 等比缩减专家数 "
                                f"{num_experts} -> {experts_new}"
                                f"（保证每个 EP rank 专家数相等且 >= top-k"
                                f"={topk}）",
                            )
                        ],
                        note=f"仅缩 EP：{ep} -> {ep_new}",
                    ),
                    None,
                )

        # ---- Tier 2 / Tier 3 need the pipeline layout --------------------
        layout, err = self._pipeline_layout(
            config, resolved, missing, model_config
        )
        if pp > MIN_PARALLEL_DEGREE and err:
            return None, err

        if pp > MIN_PARALLEL_DEGREE and layout is not None:
            plan = self._plan_pp_only(
                validator, dims, layout, target_cards, rejections, model_config
            )
            if plan is not None:
                return plan, None

            if can_shrink_ep:
                plan = self._plan_joint(
                    validator,
                    dims,
                    layout,
                    (num_experts, topk, experts_key),
                    target_cards,
                    rejections,
                    model_config,
                    encoder=(enc_experts, enc_topk),
                )
                if plan is not None:
                    return plan, None

        return None, self._diagnose(
            dims,
            target_cards,
            cards_per_node,
            missing,
            can_shrink_ep,
            layout,
            rejections,
        )

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _pipeline_layout(config, resolved, missing, model_config):
        """Collect everything PP shrinking needs. ``(layout, error)``."""
        if "num_hidden_layers" in missing:
            return None, describe_missing(
                "num_hidden_layers", missing["num_hidden_layers"]
            )

        def value_of(name):
            field = resolved.get(name)
            return field.value if field is not None else None

        mtp = effective_mtp_layers(model_config)

        # What the framework actually segments (gpt_builders.build_gpt_model
        # + GPTModel.get_sequential_layers) is
        #     head_empty + num_hidden_layers + mtp + tail_empty
        # and the MTP layers only join that count when
        # separate_mtp_headloss puts MultiTokenPredictionLayer into
        # seg_method.  In that case the builder also drops exactly one tail
        # empty layer to make room for them
        # (``num_empty_layers_add_in_tail -= 1``), so the yaml tail must stay
        # >= 1 and contributes ``tail - 1`` segmented layers.  Net effect on
        # the total: ``mtp - 1``, which is zero for the usual single MTP
        # layer -- adding the full ``mtp`` here would overshoot by one and
        # produce a config that dies in do_segment().
        head = int(config.get("num_empty_layers_add_in_head", 0) or 0)
        min_tail = 0
        if config.get("separate_mtp_headloss"):
            head += mtp - 1
            min_tail = 1

        return {
            "layers": value_of("num_hidden_layers"),
            "layers_key": resolved["num_hidden_layers"].writeback_key,
            "head": head,
            "tail": int(config.get("num_empty_layers_add_in_tail", 0) or 0),
            "min_tail": min_tail,
            "vpp": int(
                config.get("virtual_pipeline_model_parallel_size", 1) or 1
            ),
            "first_k": int(value_of("first_k_dense_replace") or 0),
            "mtp": mtp,
        }, None

    @staticmethod
    def _pp_changes(
        pp, pp_new, layout, meta, target_cards, layer_changes, note_extra=""
    ):
        """YAML + JSON writes implied by a PP shrink."""
        yaml_changes = [
            (
                "pipeline_model_parallel_size",
                pp_new,
                f"缩小 PP {pp} -> {pp_new} 以满足 {target_cards} 卡下的 "
                f"C1/C2 约束{note_extra}",
            )
        ]
        if meta["vpp_new"] != layout["vpp"]:
            if meta["vpp_new"] == 1 and pp_new < MIN_PP_FOR_VPP:
                reason = _vpp_off_reason(layout["vpp"], pp_new)
            else:
                reason = (
                    f"PP 缩容后重新对齐 VPP "
                    f"{layout['vpp']} -> {meta['vpp_new']}"
                )
            yaml_changes.append((VPP_FIELD, meta["vpp_new"], reason))
        if meta["tail_new"] != layout["tail"]:
            yaml_changes.append(
                (
                    "num_empty_layers_add_in_tail",
                    meta["tail_new"],
                    f"调整尾部空层把总层数对齐到 PP×VPP："
                    f"{layout['tail']} -> {meta['tail_new']}",
                )
            )
        json_changes = [
            (
                layout["layers_key"],
                meta["layers_new"],
                f"随 PP {pp} -> {pp_new} 等比缩减层数 "
                f"{layout['layers']} -> {meta['layers_new']}",
            )
        ]
        # Every per-layer list must follow num_hidden_layers or the framework
        # refuses to start.
        json_changes.extend(layer_changes)
        return yaml_changes, json_changes

    @staticmethod
    def _layer_field_changes(model_config, layout, meta, rejections, label):
        """Per-layer list rewrites for a candidate. ``(changes, ok)``."""
        changes, err = plan_layer_field_shrink(
            model_config,
            layout["layers"],
            meta["layers_new"],
            layout["mtp"],
        )
        if err:
            rejections.append(f"{label}：{err}")
            return [], False
        return changes, True

    def _plan_pp_only(
        self, validator, dims, layout, target_cards, rejections, model_config
    ):
        """Tier 2: shrink PP only. Returns a plan or ``None``."""
        tp, pp, ep, cp, sep = dims
        pool = []
        for pp_new in pp_candidates(pp):
            ok, why, meta = check_pp_shrink(
                pp,
                pp_new,
                layout["layers"],
                layout["head"],
                layout["tail"],
                layout["vpp"],
                first_k_dense_replace=layout["first_k"],
                min_tail=layout["min_tail"],
            )
            if not ok:
                rejections.append(f"PP {pp} -> {pp_new}：{why}")
                continue
            ok, why, _d = validator.validate(tp, pp_new, ep, cp, sep)
            if not ok:
                rejections.append(
                    f"PP {pp} -> {pp_new}：{_first_constraint(why)}"
                )
                continue
            layer_changes, ok = self._layer_field_changes(
                model_config, layout, meta, rejections, f"PP {pp} -> {pp_new}"
            )
            if not ok:
                continue
            pool.append((pp_new, meta, layer_changes))

        if not pool:
            return None

        pp_new, meta, layer_changes = max(pool, key=lambda item: item[0])
        yaml_changes, json_changes = self._pp_changes(
            pp,
            pp_new,
            layout,
            meta,
            target_cards,
            layer_changes,
            "（EP/TP/CP/SEP 不变）",
        )
        return ParallelismPlan(
            tp,
            pp_new,
            ep,
            cp,
            sep,
            yaml_changes=yaml_changes,
            json_changes=json_changes,
            warnings=[meta["warning"]] if meta.get("warning") else [],
            note=f"仅缩 PP：{pp} -> {pp_new}",
        )

    def _plan_joint(
        self,
        validator,
        dims,
        layout,
        moe,
        target_cards,
        rejections,
        model_config,
        encoder=(None, None),
    ):
        """Tier 3: shrink EP and PP together. Returns a plan or ``None``.

        Reached only when neither single-axis shrink survives -- e.g. a source
        ``(pp=8, ep=8)`` targeting 8 cards, where any one-dimension shrink
        still overshoots the card count.

        The two axes are not interchangeable, so they follow the shrink
        priority ``EP > PP``: EP absorbs the reduction down to its floor and PP
        is only shrunk for whatever is left over.  Shrinking EP costs routed
        experts and nothing else, while shrinking PP scales
        ``num_hidden_layers`` -- and with it every per-layer list, the VPP
        degree and the empty tail -- so a shallower pipeline cut is worth a
        deeper expert cut.

        ``encoder`` is the fixed HyperEncoder ``(experts, top-k)`` pool; like
        the EP-only tier, a candidate EP that would make it illegal is rejected
        (the encoder is validated, never shrunk, during EP adaptation).
        """
        tp, pp, ep, cp, sep = dims
        num_experts, topk, experts_key = moe
        enc_experts, enc_topk = encoder

        pool = []
        for ep_new in ep_candidates(ep, tp, sep):
            ok, why, experts_new = check_ep_shrink(
                ep, ep_new, num_experts, topk
            )
            if not ok:
                continue
            if not _encoder_ep_legal(enc_experts, enc_topk, ep_new):
                rejections.append(
                    f"EP {ep} -> {ep_new}：HyperEncoder 专家池 "
                    f"{enc_experts} 在该 EP 下不合法；EP 适配不缩 encoder"
                )
                continue
            for pp_new in pp_candidates(pp):
                ok, why, meta = check_pp_shrink(
                    pp,
                    pp_new,
                    layout["layers"],
                    layout["head"],
                    layout["tail"],
                    layout["vpp"],
                    first_k_dense_replace=layout["first_k"],
                    min_tail=layout["min_tail"],
                )
                if not ok:
                    continue
                ok, why, _d = validator.validate(tp, pp_new, ep_new, cp, sep)
                if not ok:
                    rejections.append(
                        f"EP {ep} -> {ep_new} + PP {pp} -> {pp_new}："
                        f"{_first_constraint(why)}"
                    )
                    continue
                layer_changes, ok = self._layer_field_changes(
                    model_config,
                    layout,
                    meta,
                    rejections,
                    f"PP {pp} -> {pp_new}",
                )
                if not ok:
                    continue
                pool.append((ep_new, pp_new, experts_new, meta, layer_changes))

        if not pool:
            return None

        # Shrink priority EP > PP: smallest feasible EP first, then the
        # largest PP that still fits under it.
        ep_new, pp_new, experts_new, meta, layer_changes = min(
            pool, key=lambda item: (item[0], -item[1])
        )
        yaml_changes, json_changes = self._pp_changes(
            pp,
            pp_new,
            layout,
            meta,
            target_cards,
            layer_changes,
            "（与 EP 联合缩容）",
        )
        yaml_changes.insert(
            0,
            (
                "expert_model_parallel_size",
                ep_new,
                f"缩小 EP {ep} -> {ep_new}：单独缩 EP 或单独缩 PP 都无法"
                f"适配 {target_cards} 卡，改为联合缩容；按 EP 优先于 PP 的"
                f"缩容顺序，先把 EP 压到可行下限，再尽量少缩 PP"
                f"（缩 PP 会连带改层数/VPP/尾部空层/逐层配置）",
            ),
        )
        json_changes.insert(
            0,
            (
                experts_key,
                experts_new,
                f"随 EP {ep} -> {ep_new} 等比缩减专家数 "
                f"{num_experts} -> {experts_new}（>= top-k={topk}）",
            ),
        )
        return ParallelismPlan(
            tp,
            pp_new,
            ep_new,
            cp,
            sep,
            yaml_changes=yaml_changes,
            json_changes=json_changes,
            warnings=[meta["warning"]] if meta.get("warning") else [],
            note=f"EP+PP 联合缩容：EP {ep} -> {ep_new}，PP {pp} -> {pp_new}",
        )

    @staticmethod
    def _diagnose(
        dims,
        target_cards,
        cards_per_node,
        missing,
        can_shrink_ep,
        layout,
        rejections,
    ):
        """Explain why no candidate survived, with an actionable suggestion."""
        tp, pp, ep, cp, sep = dims
        reasons = []

        if ep > 1 and not can_shrink_ep:
            if ep == MIN_PARALLEL_DEGREE:
                reasons.append(
                    f"EP={ep} 已是允许的最小值"
                    f"（不允许把 EP 从 >1 缩到 1，否则等于去掉专家并行）"
                )
            for name in ("num_experts", "num_experts_per_tok"):
                if name in missing:
                    reasons.append(describe_missing(name, missing[name]))

        if pp > 1 and pp == MIN_PARALLEL_DEGREE:
            reasons.append(
                f"PP={pp} 已是允许的最小值"
                f"（不允许把 PP 从 >1 缩到 1，否则等于去掉流水并行）"
            )
        elif pp > MIN_PARALLEL_DEGREE and layout is not None:
            floor_layers = layout["layers"] * MIN_PARALLEL_DEGREE // pp
            reasons.append(
                f"PP 最多缩到 {MIN_PARALLEL_DEGREE}，此时层数为 "
                f"{floor_layers}（={layout['layers']}×"
                f"{MIN_PARALLEL_DEGREE}/{pp}），仍不满足约束"
            )

        floor = tp * sep * MIN_PARALLEL_DEGREE if pp > 1 else tp * sep
        if target_cards % (tp * sep) != 0:
            reasons.append(
                f"C1 约束：target_cards={target_cards} 不能被 TP×SEP="
                f"{tp}×{sep}={tp * sep} 整除"
            )
        elif target_cards < floor:
            reasons.append(
                f"目标卡数 {target_cards} 小于并行度下限 {floor}"
                f"（TP×SEP×PP_min）"
            )

        cp_reason = _cp_blocks_c4(dims, target_cards)
        if cp_reason:
            reasons.append(cp_reason)

        if not reasons:
            reasons.append("所有候选组合都不满足通信组约束或模型结构约束")

        # De-duplicated, truncated candidate rejections: the concrete reason a
        # given (EP, PP) pair was thrown away is usually the fastest way to
        # see what to change.  Model-structure reasons come first -- they are
        # actionable, while the C-constraint ones repeat the same arithmetic.
        # A couple of slots stay reserved for the latter, because C4
        # ("sharding % CP") is often the only real blocker and would otherwise
        # be crowded out entirely.
        unique = list(dict.fromkeys(rejections))
        topological = [r for r in unique if _CONSTRAINT_RE.search(r)]
        structural = [r for r in unique if r not in topological]
        reserved = min(len(topological), MIN_TOPOLOGICAL_DETAIL)
        detail = (structural[: DETAIL_LIMIT - reserved] + topological)[
            :DETAIL_LIMIT
        ]
        detail_block = ""
        if detail:
            detail_block = "\n  候选淘汰明细：\n    " + "\n    ".join(detail)

        min_cards = min_shrink_cards(tp, pp, ep, cp, sep, cards_per_node)
        if min_cards is None:
            advice = (
                "当前维度组合本身不合法（C3 要求 EP % (TP×SEP) == 0，"
                "C5 禁止 SEP 与 CP 同时 >1），任何卡数都无法适配，"
                "必须先修改 TP/SEP/EP/CP"
            )
            return (
                f"{target_cards} 卡下找不到合法的 EP/PP 缩容方案 "
                f"(tp={tp}, pp={pp}, ep={ep}, cp={cp}, sep={sep})。\n"
                f"  阻断原因：\n    "
                + "\n    ".join(reasons)
                + detail_block
                + f"\n  建议：{advice}"
            )
        min_nodes = max(min_cards // cards_per_node, 1)
        if min_cards > target_cards:
            advice = (
                f"改用 --target-nodes {min_nodes}"
                f"（{min_cards} 卡，缩容下限），"
                f"或手动降低源 YAML 的 TP/PP 后重试"
            )
        else:
            advice = (
                f"{target_cards} 卡已达到该配置的缩容下限，"
                f"请调整模型结构（如 --set json:<专家数字段>=<能被目标 EP "
                f"整除的值>），或手动降低源 YAML 的 TP/PP/EP 后重试"
            )
        if cp_reason:
            advice += (
                "；若确实可以牺牲序列长度，可显式加 --scale-seq-length "
                "<新的 max_seq_length>（须与源值成整数倍关系），"
                "它会按同一倍率把 CP 一起缩小"
            )
        return (
            f"{target_cards} 卡下找不到合法的 EP/PP 缩容方案 "
            f"(tp={tp}, pp={pp}, ep={ep}, cp={cp}, sep={sep})。\n"
            f"  阻断原因：\n    " + "\n    ".join(reasons) + detail_block + "\n"
            "  建议：" + advice
        )


def _cp_blocks_c4(dims, target_cards):
    """Diagnostic line when CP alone rules out every candidate, else ``None``.

    C4 needs ``sharding % CP == 0`` with ``sharding = N / (TP*SEP*PP)``.  CP
    never shrinks, and PP is the only axis that moves ``sharding``, so a large
    CP can eliminate the whole search space on its own -- no expert-count or
    layer-count change can rescue it.  Saying so explicitly matters because the
    generic "no candidate survived" wording sends the user off tuning EP and
    PP, which cannot possibly help.
    """
    tp, pp, ep, cp, sep = dims
    if cp <= 1:
        return None
    dense = tp * sep
    best = None
    for pp_new in [pp, *pp_candidates(pp)]:
        denom = dense * pp_new
        if denom < 1 or target_cards % denom:
            continue
        sharding = target_cards // denom
        if sharding % cp == 0:
            return None
        best = sharding if best is None else max(best, sharding)
    if best is None:
        return None
    return (
        f"CP={cp} 不参与自动缩容，而 {target_cards} 卡下 sharding 最大只有 "
        f"{best}（=卡数/(TP×SEP×PP_min)），不能被 CP={cp} 整除，"
        f"C4 对任何 (EP, PP) 组合都无解。"
        f"缩 CP 必须同步缩 max_seq_length，既改变被测序列长度，"
        f"也会让按固定序列长度预打包的数据集读不进来，因此不自动进行"
    )


def _first_constraint(message):
    """Pull the first ``Cx`` line out of a validator message."""
    for line in message.splitlines():
        stripped = line.strip()
        if stripped.startswith("C") and "不满足" in stripped:
            return stripped
    return message.splitlines()[0].strip()
