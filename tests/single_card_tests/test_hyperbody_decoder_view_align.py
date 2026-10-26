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

"""Unit coverage for the HyperBody decoder-view alignment knobs.

Pure config-view assertions (no GPU / no distributed init) for the fixes carried
onto the nested (``sub_configs``) config:

  * ``_attn_implementation`` is yaml/config-configurable and passes through onto
    the decoder view (a fixed decoder kernel is required for bit-exact alignment
    with standalone ernie5_v2 -- the MLA core diverges under ``eager`` vs the
    fused ``default`` path);
  * ``first_k_dense_replace`` truthy no longer collides with a list
    ``moe_layer_freq`` (the view is materialized with an int freq so the
    provider ``__post_init__`` builds the ``[0] + [1]*(L-1)`` dense-first table);
  * nested ``decoder_config`` / ``encoder_config`` dicts are promoted to their
    sub-config objects by ``HyperBodyConfig``.

Geometry is REQUIRED by the sensitive-info policy, so configs are built with
explicit ``decoder_config=`` / ``encoder_config=`` sub-configs.
"""

import sys

from paddlefleet.transformers.hyperbody.configuration import (
    HyperBodyConfig,
    HyperBodyDecoderConfig,
    HyperBodyEncoderConfig,
)
from paddlefleet.transformers.hyperbody.modeling import (
    _build_decoder_view,
    _build_encoder_view,
)

_DECODER_GEOMETRY = {
    "vocab_size": 128,
    "hidden_size": 64,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "intermediate_size": 128,
    "n_routed_experts": 4,
    "moe_intermediate_size": 32,
    "num_experts_per_tok": 2,
    "n_shared_experts": 1,
}
_ENCODER_GEOMETRY = {
    "vocab_size": 128,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "moe_intermediate_size": 32,
    "n_routed_experts": 4,
    "num_experts_per_tok": 2,
    "n_shared_experts": 1,
    "first_k_dense_replace": 1,
    "hyperencoder_query_length": 16,
}


def _make_config(decoder_overrides=None, encoder_overrides=None, **top_level):
    decoder_config = {**_DECODER_GEOMETRY, **(decoder_overrides or {})}
    encoder_config = {**_ENCODER_GEOMETRY, **(encoder_overrides or {})}
    return HyperBodyConfig(
        decoder_config=decoder_config,
        encoder_config=encoder_config,
        **top_level,
    )


def test_nested_dict_promoted_to_subconfigs():
    """Nested ``decoder_config`` / ``encoder_config`` dicts become sub-config objects."""
    cfg = _make_config()
    assert isinstance(cfg.decoder_config, HyperBodyDecoderConfig)
    assert isinstance(cfg.encoder_config, HyperBodyEncoderConfig)
    assert cfg.decoder_config.num_hidden_layers == 4
    assert cfg.encoder_config.num_hidden_layers == 2


def test_attn_implementation_passthrough_to_view():
    """``_attn_implementation`` flows from config onto the decoder view.

    The fused ``default`` kernel is what standalone ernie5_v2(lite) uses; the
    HyperBody decoder view must be able to select it (else the MLA core runs
    ``eager`` and diverges from lite).
    """
    for impl in ("default", "eager"):
        cfg = _make_config(_attn_implementation=impl)
        view = _build_decoder_view(cfg)
        assert view._attn_implementation == impl, (
            impl,
            view._attn_implementation,
        )


def test_attn_implementation_falls_back_to_default_when_absent():
    """When config carries no truthy ``_attn_implementation`` the view uses 'default'."""
    cfg = _make_config()
    # Simulate an absent/empty attribute (older/hand-built configs).
    cfg._attn_implementation = None
    view = _build_decoder_view(cfg)
    assert view._attn_implementation == "default", view._attn_implementation


def test_first_k_dense_replace_builds_dense_first_without_conflict():
    """``first_k_dense_replace`` truthy must not collide with a list moe_layer_freq.

    The view builder passes an int ``moe_layer_freq`` to the provider so the
    provider builds the ``[0] + [1]*(L-1)`` dense-first table (layer 0 dense).
    """
    cfg = _make_config(decoder_overrides={"first_k_dense_replace": 1})
    view = _build_decoder_view(cfg)  # must not raise
    freq = view.moe_layer_freq
    assert isinstance(freq, (list, tuple)), freq
    assert freq[0] == 0, freq  # layer 0 dense
    assert all(x == 1 for x in freq[1:]), freq  # rest MoE


def test_encoder_pad_token_id_configurable():
    """A top-level ``pad_token_id`` flows onto the encoder view.

    ``_build_encoder_view`` forwards the global ``pad_token_id`` into the
    transient ``HyperEncoderConfig``; ``HyperEncoderProvider.__post_init__`` no
    longer hard-pins 0, so the configured value survives onto the view.
    """
    cfg = _make_config(pad_token_id=7)
    view = _build_encoder_view(cfg, decoder_hidden=64)
    assert view.pad_token_id == 7, view.pad_token_id


def test_encoder_pad_token_id_defaults_to_zero_when_absent():
    """With no configured ``pad_token_id`` the encoder view falls back to 0
    (previous behavior), so unset configs are unchanged.
    """
    cfg = _make_config()
    assert getattr(cfg, "pad_token_id", None) is None
    view = _build_encoder_view(cfg, decoder_hidden=64)
    assert view.pad_token_id == 0, view.pad_token_id


def test_encoder_and_decoder_share_pad_token_id():
    """Encoder and decoder views read the SAME top-level ``pad_token_id``."""
    cfg = _make_config(pad_token_id=5)
    dec = _build_decoder_view(cfg)
    enc = _build_encoder_view(cfg, decoder_hidden=64)
    assert dec.pad_token_id == 5, dec.pad_token_id
    assert enc.pad_token_id == 5, enc.pad_token_id


def test_encoder_pad_token_id_encoder_scoped_overrides_top_level():
    """An ``encoder_config.pad_token_id`` wins over the top-level value.

    The encoder view prefers the encoder-scoped id and falls back to the
    top-level one only when it is None, so a cross-seam shared id stays the
    default while per-modality overrides remain possible.
    """
    cfg = _make_config(
        encoder_overrides={"pad_token_id": 9},
        pad_token_id=5,
    )
    dec = _build_decoder_view(cfg)
    enc = _build_encoder_view(cfg, decoder_hidden=64)
    assert enc.pad_token_id == 9, enc.pad_token_id
    assert dec.pad_token_id == 5, dec.pad_token_id


def test_encoder_router_aux_loss_coef_independently_configurable():
    """``encoder_config.router_aux_loss_coef`` drives the encoder view."""
    cfg = _make_config(encoder_overrides={"router_aux_loss_coef": 0.05})
    view = _build_encoder_view(cfg, decoder_hidden=64)
    assert view.router_aux_loss_coef == 0.05, view.router_aux_loss_coef


def test_encoder_router_aux_loss_coef_decoupled_from_top_level():
    """The top-level coef no longer leaks into the encoder view.

    The decoder runs noaux_tc at coef 0.0 while the encoder's softmax/greedy
    router needs a real aux loss, so the encoder coef must come from
    ``encoder_config`` -- NOT from the global ``config.router_aux_loss_coef``.
    If this regresses to the global value the encoder aux loss silently
    follows the decoder-tuned coef.
    """
    cfg = _make_config(
        encoder_overrides={"router_aux_loss_coef": 0.05},
        router_aux_loss_coef=0.5,
    )
    view = _build_encoder_view(cfg, decoder_hidden=64)
    assert view.router_aux_loss_coef == 0.05, view.router_aux_loss_coef


def test_encoder_router_aux_loss_coef_default_keeps_old_global_default():
    """Unset encoder coef keeps the old behavior (the old global default)."""
    cfg = _make_config()
    view = _build_encoder_view(cfg, decoder_hidden=64)
    assert view.router_aux_loss_coef == 0.001, view.router_aux_loss_coef


def test_hyperbody_context_token_id_drives_both_views():
    """``hyperbody_context_token_id`` flows onto BOTH views.

    The decoder scatter id (``view.image_token_id``) and the encoder-side
    ``hyperbody_context_token_id`` must both come from the configured value,
    not from the constant or from ``image_token_id``.
    """
    cfg = _make_config(hyperbody_context_token_id=151665)
    dec = _build_decoder_view(cfg)
    enc = _build_encoder_view(cfg, decoder_hidden=64)
    assert dec.image_token_id == 151665, dec.image_token_id
    assert enc.hyperbody_context_token_id == 151665, (
        enc.hyperbody_context_token_id
    )


def test_hyperbody_context_token_id_defaults_to_context_token():
    """Unset context id keeps the default CONTEXT_TOKEN on both views."""
    from paddlefleet.transformers.hyperbody.configuration import CONTEXT_TOKEN

    cfg = _make_config()
    dec = _build_decoder_view(cfg)
    enc = _build_encoder_view(cfg, decoder_hidden=64)
    assert cfg.hyperbody_context_token_id == CONTEXT_TOKEN
    assert dec.image_token_id == CONTEXT_TOKEN, dec.image_token_id
    assert enc.hyperbody_context_token_id == CONTEXT_TOKEN, (
        enc.hyperbody_context_token_id
    )


def test_hyperbody_context_token_id_falls_back_to_image_token_id():
    """An unset context id falls back to ``image_token_id`` (old semantics).

    Before the knob existed, ``image_token_id`` drove the decoder scatter id;
    configs that only set ``image_token_id`` must keep working unchanged.
    """
    cfg = _make_config(image_token_id=128001)
    dec = _build_decoder_view(cfg)
    enc = _build_encoder_view(cfg, decoder_hidden=64)
    assert cfg.hyperbody_context_token_id == 128001
    assert dec.image_token_id == 128001, dec.image_token_id
    assert enc.hyperbody_context_token_id == 128001, (
        enc.hyperbody_context_token_id
    )


def test_hyperbody_context_token_id_wins_over_image_token_id():
    """An explicit context id wins when both fields are set."""
    cfg = _make_config(
        image_token_id=128001,
        hyperbody_context_token_id=151665,
    )
    dec = _build_decoder_view(cfg)
    assert dec.image_token_id == 151665, dec.image_token_id


if __name__ == "__main__":
    try:
        test_nested_dict_promoted_to_subconfigs()
        test_attn_implementation_passthrough_to_view()
        test_attn_implementation_falls_back_to_default_when_absent()
        test_first_k_dense_replace_builds_dense_first_without_conflict()
        test_encoder_pad_token_id_configurable()
        test_encoder_pad_token_id_defaults_to_zero_when_absent()
        test_encoder_and_decoder_share_pad_token_id()
        test_encoder_pad_token_id_encoder_scoped_overrides_top_level()
        test_encoder_router_aux_loss_coef_independently_configurable()
        test_encoder_router_aux_loss_coef_decoupled_from_top_level()
        test_encoder_router_aux_loss_coef_default_keeps_old_global_default()
        test_hyperbody_context_token_id_drives_both_views()
        test_hyperbody_context_token_id_defaults_to_context_token()
        test_hyperbody_context_token_id_falls_back_to_image_token_id()
        test_hyperbody_context_token_id_wins_over_image_token_id()
    except AssertionError:
        import traceback

        traceback.print_exc()
        sys.exit(1)
    print("OK")
