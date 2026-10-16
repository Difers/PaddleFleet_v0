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

"""Alias registry for the ``model_config.json`` fields shrinking depends on.

The same structural quantity is spelled differently per model family::

    routed experts : n_routed_experts (ERNIE5 / DeepSeek / GLM / Kimi),
                     num_local_experts (MiniMax / GPT-OSS),
                     num_experts (Qwen), moe_num_experts (ERNIE4.5)
    top-k          : num_experts_per_tok (most), moe_k (ERNIE4.5)
    dense prefix   : first_k_dense_replace, moe_layer_start_index
    MTP layers     : num_nextn_predict_layers

Centralising the aliases means supporting a new family is a one-line edit to
:data:`FIELD_SPECS`.  The resolver also remembers which physical key matched,
so a write-back never invents a second spelling of the same field.

Nested configs (HyperBody)
--------------------------
Most families spell the structural fields at the JSON top level, but HyperBody
nests the *trainable decoder's* geometry under ``decoder_config`` and carries a
second, identically-named copy of the HyperEncoder's geometry under
``encoder_config``.  So resolution has two scopes:

* the primary (decoder) scope -- :func:`resolve_fields` -- searches the top
  level first and then every section in :data:`STRUCTURE_SECTIONS`
  (``decoder_config``).  Top-level-first keeps flat models byte-for-byte
  unchanged and never lets the decoder scope accidentally pick up the
  encoder's copy (``encoder_config`` is deliberately NOT in the search path);
* the encoder scope -- :func:`resolve_encoder_fields` -- searches ONLY
  ``encoder_config``.  It exists so the HyperEncoder expert pool can be
  validated (and, under ``--shrink-structure``, shrunk) independently of the
  decoder.

Every resolved field now carries a *dotted path* (``origin`` / ``writeback_key``
e.g. ``decoder_config.n_routed_experts``) instead of a bare key, so a write-back
lands in the exact mapping the value was read from -- writing a stray top-level
``n_routed_experts`` on a HyperBody config would be silently ignored by the
loader.  A top-level field keeps a plain, dot-free key, so flat models are
unaffected.

Pure module: no I/O, no mutation of inputs.
"""

from __future__ import annotations

from collections import namedtuple

#: Sub-configs the *primary* (decoder) scope descends into, after the top
#: level, in order.  HyperBody nests the decoder geometry under
#: ``decoder_config``; flat families leave these fields at the top level, where
#: the top-level-first search finds them unchanged.  ``encoder_config`` is
#: intentionally absent: the decoder scope must never resolve to the encoder's
#: identically-named copy (see :func:`resolve_encoder_fields` for the encoder).
STRUCTURE_SECTIONS = ("decoder_config",)

#: The HyperEncoder sub-config.  Resolved only by :func:`resolve_encoder_fields`.
ENCODER_SECTION = "encoder_config"

# needed_by only phrases diagnostics; ``default is None`` marks the field as
# required for that shrink axis.
FieldSpec = namedtuple("FieldSpec", ["aliases", "default", "needed_by"])

FIELD_SPECS = {
    "num_hidden_layers": FieldSpec(
        aliases=("num_hidden_layers",),
        default=None,
        needed_by="PP",
    ),
    "num_experts": FieldSpec(
        aliases=(
            "n_routed_experts",
            "num_local_experts",
            "num_experts",
            "moe_num_experts",
        ),
        default=None,
        needed_by="EP",
    ),
    "num_experts_per_tok": FieldSpec(
        aliases=("num_experts_per_tok", "moe_k"),
        default=None,
        needed_by="EP",
    ),
    "first_k_dense_replace": FieldSpec(
        aliases=("first_k_dense_replace", "moe_layer_start_index"),
        default=0,
        needed_by="PP",
    ),
    "num_nextn_predict_layers": FieldSpec(
        aliases=("num_nextn_predict_layers",),
        default=0,
        needed_by="PP",
    ),
}

ResolvedField = namedtuple(
    "ResolvedField", ["logical", "value", "origin", "writeback_key"]
)


def _search(model_config, aliases):
    """``(value, dotted_path)`` for the first alias present with a non-``None``
    value, searching the top level first and then each
    :data:`STRUCTURE_SECTIONS` sub-config in order; ``(None, None)`` if absent.

    Top-level-first is what keeps a flat model's path dot-free (and its
    behaviour identical to the pre-nesting resolver); only a HyperBody-style
    config, whose decoder geometry lives under ``decoder_config``, falls
    through to the section search.
    """
    for alias in aliases:
        if model_config.get(alias) is not None:
            return model_config[alias], alias
    for section in STRUCTURE_SECTIONS:
        sub = model_config.get(section)
        if isinstance(sub, dict):
            for alias in aliases:
                if sub.get(alias) is not None:
                    return sub[alias], f"{section}.{alias}"
    return None, None


def _writeback_path(model_config, aliases):
    """First alias *present* (membership, value may be ``None``) as a dotted
    path -- top level first, then each section -- else the canonical top-level
    alias.  Mirrors :func:`_search`'s precedence so a write lands exactly where
    the value was read.
    """
    for alias in aliases:
        if alias in model_config:
            return alias
    for section in STRUCTURE_SECTIONS:
        sub = model_config.get(section)
        if isinstance(sub, dict):
            for alias in aliases:
                if alias in sub:
                    return f"{section}.{alias}"
    return aliases[0]


def nested_find(model_config, key):
    """``(value, dotted_path)`` for a *single* ``key`` at the top level, else in
    each :data:`STRUCTURE_SECTIONS` sub-config; ``(None, None)`` when absent.

    The ad-hoc single-key counterpart to :func:`resolve_fields`, for the planner
    reads/writes of fields that are not in :data:`FIELD_SPECS` (MoE widths,
    routing-group sizes, the dispatcher type).  Same top-level-first precedence,
    so a flat config always yields the bare key it already used.
    """
    if key in model_config:
        return model_config[key], key
    for section in STRUCTURE_SECTIONS:
        sub = model_config.get(section)
        if isinstance(sub, dict) and key in sub:
            return sub[key], f"{section}.{key}"
    return None, None


def nested_get(model_config, key, default=None):
    """Flat-or-nested read of ``key`` (see :func:`nested_find`).  Returns
    ``default`` when the key is absent *or* present with a ``None`` value, so it
    is a drop-in for the planner's ``model_config.get(key, default)`` reads that
    then coerce with ``... or <fallback>``.
    """
    value, _path = nested_find(model_config, key)
    return default if value is None else value


def resolve_fields(model_config):
    """Resolve every registered field against ``model_config``.

    Returns ``(resolved, missing)``: ``resolved`` maps a logical name to a
    :class:`ResolvedField` whose ``origin`` / ``writeback_key`` are dotted
    paths; ``missing`` maps a logical name to its :class:`FieldSpec` for
    required fields that could not be found at the top level or in any
    :data:`STRUCTURE_SECTIONS` sub-config.
    """
    resolved = {}
    missing = {}

    for logical, spec in FIELD_SPECS.items():
        value, origin = _search(model_config, spec.aliases)
        writeback_key = _writeback_path(model_config, spec.aliases)
        if origin is not None:
            resolved[logical] = ResolvedField(
                logical, value, origin, writeback_key
            )
        elif spec.default is not None:
            resolved[logical] = ResolvedField(
                logical, spec.default, "<default>", writeback_key
            )
        else:
            missing[logical] = spec

    return resolved, missing


def resolve_encoder_fields(model_config):
    """Resolve the structural fields inside ``encoder_config`` (the HyperEncoder
    tower), independently of the decoder.

    Returns ``(resolved, missing)`` exactly like :func:`resolve_fields`, except
    every ``origin`` / ``writeback_key`` is prefixed with ``encoder_config.`` and
    *only* that sub-config is searched (never the top level, so it cannot pick
    up a decoder/global field by the same name).  Returns ``({}, {})`` when
    there is no ``encoder_config`` mapping, so every non-HyperBody model is a
    no-op.

    The encoder shares the decoder's global EP, so EP *adaptation* only needs
    this to validate the fixed encoder pool; ``--shrink-structure`` additionally
    uses it to shrink the encoder (see :mod:`.planner`).
    """
    enc = model_config.get(ENCODER_SECTION)
    if not isinstance(enc, dict):
        return {}, {}

    resolved = {}
    missing = {}
    for logical, spec in FIELD_SPECS.items():
        hit = next((a for a in spec.aliases if enc.get(a) is not None), None)
        writeback_alias = next(
            (a for a in spec.aliases if a in enc), spec.aliases[0]
        )
        writeback_key = f"{ENCODER_SECTION}.{writeback_alias}"
        if hit is not None:
            resolved[logical] = ResolvedField(
                logical, enc[hit], f"{ENCODER_SECTION}.{hit}", writeback_key
            )
        elif spec.default is not None:
            resolved[logical] = ResolvedField(
                logical, spec.default, "<default>", writeback_key
            )
        else:
            missing[logical] = spec

    return resolved, missing


def describe_missing(logical, spec):
    """One-line actionable explanation for an unresolved required field."""
    tried = " / ".join(spec.aliases)
    return (
        f"{logical}（{spec.needed_by} 缩容所需）在 model_config.json 中读不到"
        f"（已在顶层及嵌套子配置 {', '.join(STRUCTURE_SECTIONS)} 中尝试字段名 "
        f"{tried}）；请用 --set json:{spec.aliases[0]}=<值> 显式补上"
        f"（嵌套配置写点路径，如 --set json:decoder_config.{spec.aliases[0]}=<值>），"
        f"或在 field_spec.py 的 FIELD_SPECS[{logical!r}] 里登记新别名"
    )
