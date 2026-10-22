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
"""Multi-tower modular AOA generation.

The identity fallback strips exactly one model root, so a model whose regions
are rooted at different single-name prefixes (HyperBody's ``encoder`` region
beside its ``model`` decoder) cannot be described by one ``AOAContext``. Such a
model declares one :class:`AOATower` per region; the entries here build one
context per tower and dispatch each live unit to its owner, reusing the context
builder and the live-unit walk from :mod:`aoa_generator`.

Kept out of :mod:`aoa_generator` so the single-tower generator stays short.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from paddlefleet.models.gpt.aoa_generator import (
    build_aoa_context,
    globalize_statements,
    iter_live_units,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from paddle.distributed.flex_checkpoint.aoa.generation import AOAContext


@dataclass(frozen=True)
class AOATower:
    """One AOA generation scope: a model root paired with a checkpoint root.

    ``model_name_prefix`` is the region's model single-name root (no trailing
    dot) and ``config`` is the region's provider view, which its components
    read. ``owns_unit(structured_prefix) -> bool`` says which live units belong
    to the region.

    The checkpoint overrides default to the config's ``aoa_checkpoint_name_*``
    declarations. A region whose names are already correct needs neither, so
    only a region reading a *different* part of the checkpoint than it is named
    after passes them -- MTP warm-start, where target depth ``d`` reads the last
    depth the checkpoint has.
    """

    model_name_prefix: str
    config: Any
    owns_unit: Callable[[str], bool]
    checkpoint_name_prefix: str | None = None
    checkpoint_name_mapping: dict | None = None
    # Set when another destination also reads this tower's checkpoint root; see
    # :func:`_copy_path_for_shared_checkpoint_roots`.
    shares_checkpoint_root: bool = False


# Suffix for the temporary a plain rename out of a shared source is routed
# through; the double underscore keeps it off any real name.
_SHARED_SOURCE_TEMP_SUFFIX = ".__shared_source_copy"


def _copy_path_for_shared_checkpoint_roots(
    statements: list[str], towers, model_names
) -> list[str]:
    """Routes every reader of a shared checkpoint root through a temporary.

    ``dist.load_state_dict`` keys its load off the *source* name: a plain rename
    (one source, one destination, no transform) loads straight into the
    destination, while a transform routes the source through a temporary that is
    copied to every destination. One checkpoint tensor feeding several
    destinations therefore reaches all of them only when *every* reader goes
    through a temporary -- the readers then share the single loaded source
    buffer. A lone direct reader left in collides with the rewritten ones on
    ``(source_key, offset)``: the load's ``sorted`` order lets whichever it
    visits last win the slot, and the other end keeps either its initial value
    or an unread ``paddle.empty``.

    Two kinds of reader exist. A region whose name differs from the checkpoint
    emits a plain rename, rewritten here into an identity double transpose. A
    region whose name already matches resolves through the engine's passthrough
    and emits *no* statement, yet still direct-loads; so for every shared source
    that is also a live model name a self-copy is synthesized for that
    passthrough end. Keyed off the declared root, not off how many statements
    consume the source, because the passthrough end contributes none.
    """
    prefixes = tuple(
        f"{tower.checkpoint_name_prefix}."
        for tower in towers
        if tower.shares_checkpoint_root and tower.checkpoint_name_prefix
    )
    if not prefixes:
        return statements
    rewritten = []
    fanned_out_sources = []
    seen_sources = set()
    for statement in statements:
        source, separator, destination = statement.partition(" -> ")
        source, destination = source.strip(), destination.strip()
        # A space rules out trailing attributes, a comma multi-source and
        # multi-destination, a caret an existing transform -- all three already
        # go through a temporary.
        plain_rename = separator and not any(
            character in source + destination for character in ",^ "
        )
        if plain_rename and source.startswith(prefixes):
            temp = f"{destination}{_SHARED_SOURCE_TEMP_SUFFIX}"
            rewritten += [f"{source}^T -> {temp}", f"{temp}^T -> {destination}"]
            if source not in seen_sources:
                seen_sources.add(source)
                fanned_out_sources.append(source)
        else:
            rewritten.append(statement)
    # The passthrough reader of a fanned-out source is a model parameter named
    # exactly like the checkpoint key and so direct-loads; route it through a
    # temporary too. Guarded on model membership so a pure fan-out with no
    # passthrough reader does not synthesize a self-copy into a name no
    # parameter owns. Emitted before the renames to match the hand-written
    # order, where the source layer's identity statement preceded its readers'.
    self_copies = []
    for source in fanned_out_sources:
        if source in model_names:
            temp = f"{source}{_SHARED_SOURCE_TEMP_SUFFIX}"
            self_copies += [f"{source}^T -> {temp}", f"{temp}^T -> {source}"]
    return self_copies + rewritten


def _iter_tower_units(
    model, towers
) -> Iterator[tuple[AOAContext, str, object]]:
    """Yields ``(ctx, structured_prefix, unit)`` in the single-tower walk order.

    One walk for all towers, not one walk per tower: statements are an ordered
    program, and keeping the single walk keeps every region's internal order
    identical to what it would be on its own.
    """
    contexts = [
        build_aoa_context(
            model,
            tower.config,
            model_name_prefix=tower.model_name_prefix,
            checkpoint_name_prefix=tower.checkpoint_name_prefix,
            checkpoint_name_mapping=tower.checkpoint_name_mapping,
        )
        for tower in towers
    ]
    for structured_prefix, unit in iter_live_units(model):
        owners = [
            index
            for index, tower in enumerate(towers)
            if tower.owns_unit(structured_prefix)
        ]
        # Zero owners would silently drop the region's parameters from the
        # config; several would generate it twice under different roots.
        if len(owners) != 1:
            raise RuntimeError(
                f"AOA multi-tower generation needs exactly one tower per live "
                f"unit, but unit {structured_prefix!r} is owned by "
                f"{len(owners)}: "
                f"{[towers[index].model_name_prefix for index in owners]}."
            )
        yield contexts[owners[0]], structured_prefix, unit


def gen_multi_tower_aoa(model, towers) -> dict[str, list[str]]:
    """Multi-tower counterpart of ``gen_whole_model_aoa``."""
    statements = []
    for ctx, structured_prefix, unit in _iter_tower_units(model, towers):
        statements += unit.gen_aoa_statements(
            ctx, structured_name_prefix=structured_prefix
        )
    # The single-card model names, the namespace the statements are in before
    # the pipeline gather: a fanned-out source that is also one of these is
    # exactly the passthrough reader to protect.
    model_names = set((model._pp_to_single_mapping or {}).values())
    statements = _copy_path_for_shared_checkpoint_roots(
        statements, towers, model_names
    )
    # The pipeline size is a whole-model property: a region's view may not even
    # carry it (HyperBody builds its encoder view from an explicit field list),
    # which would turn the "declared pipeline without a group" error into a
    # silent rank-local config.
    return {"aoa_statements": globalize_statements(model.config, statements)}


def gen_multi_tower_inv_aoa(model, towers) -> dict[str, list[str]]:
    """Multi-tower counterpart of ``gen_whole_model_inv_aoa``.

    Independently generated, never derived from the checkpoint->model pass. The
    copy-path rewrite is deliberately absent: on export every model tensor is
    its own source, so one source cannot feed several destinations.
    """
    statements = []
    for ctx, structured_prefix, unit in reversed(
        list(_iter_tower_units(model, towers))
    ):
        statements += unit.gen_inv_aoa_statements(
            ctx, structured_name_prefix=structured_prefix
        )
    return {"aoa_statements": globalize_statements(model.config, statements)}
