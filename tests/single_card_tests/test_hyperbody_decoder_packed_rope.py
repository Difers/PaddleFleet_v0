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

"""Unit coverage for the HyperBody packed-path config gates.

Under the nested (``sub_configs``) config, ``packed_decoder_rope`` on the decoder
view tracks ``encoder_config.hyperencoder_packed_decoder`` directly (the two move
together; there is no separate top-level override knob). This module asserts:
  * ``modeling._build_decoder_view`` sets ``packed_decoder_rope`` == the encoder
    flag for both True/False;
  * ``hyperencoder_attn_backend='flashmask'`` and ``hyperencoder_packed_decoder``
    must be set together (both mixed pairs are rejected at config construction,
    case-insensitive);
  * ``modeling._build_encoder_view`` picks the attention implementation that
    matches the backend and marks the encoder as CP-replicated.

These are pure config-view assertions (no GPU / no distributed init). Geometry is
REQUIRED by the sensitive-info policy, so configs are built with explicit
``decoder_config=`` / ``encoder_config=`` sub-configs.
"""

import sys

from paddlefleet.transformers.hyperbody.configuration import HyperBodyConfig
from paddlefleet.transformers.hyperbody.modeling import (
    _build_decoder_view,
    _build_encoder_view,
)

# Minimal-but-complete geometry satisfying the REQUIRED sensitive-info fields.
_DECODER_GEOMETRY = {
    "vocab_size": 128,
    "hidden_size": 64,
    "num_hidden_layers": 2,
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


def _make_config(hyperencoder_packed_decoder, hyperencoder_attn_backend=None):
    if hyperencoder_attn_backend is None:
        hyperencoder_attn_backend = (
            "flashmask" if hyperencoder_packed_decoder else "dp"
        )
    encoder_config = {
        **_ENCODER_GEOMETRY,
        "hyperencoder_attn_backend": hyperencoder_attn_backend,
        "hyperencoder_packed_decoder": hyperencoder_packed_decoder,
    }
    return HyperBodyConfig(
        decoder_config={**_DECODER_GEOMETRY},
        encoder_config=encoder_config,
    )


def test_packed_decoder_rope_tracks_hyperencoder_packed_decoder():
    """``packed_decoder_rope`` on the decoder view mirrors the encoder flag."""
    for packed in (True, False):
        cfg = _make_config(hyperencoder_packed_decoder=packed)
        view = _build_decoder_view(cfg)
        assert view.packed_decoder_rope is packed, (
            packed,
            view.packed_decoder_rope,
        )


def test_flashmask_backend_and_packed_decoder_move_together():
    """Both mixed pairs are rejected at construction; matching pairs pass."""
    for backend, packed in (
        ("flashmask", False),
        ("FlashMask", False),
        ("dp", True),
    ):
        raised = False
        try:
            _make_config(
                hyperencoder_packed_decoder=packed,
                hyperencoder_attn_backend=backend,
            )
        except ValueError as e:
            assert "must be set together" in str(e), str(e)
            raised = True
        assert raised, f"{backend=} {packed=} should raise ValueError"

    for backend, packed in (
        ("flashmask", True),
        ("FLASHMASK", True),
        ("dp", False),
    ):
        cfg = _make_config(
            hyperencoder_packed_decoder=packed,
            hyperencoder_attn_backend=backend,
        )
        assert cfg.encoder_config.hyperencoder_packed_decoder is packed


def test_encoder_view_attn_implementation_follows_backend():
    """flashmask -> non-eager 'flashmask'; dp -> 'eager'; always CP-replicated."""
    for packed, impl in ((True, "flashmask"), (False, "eager")):
        view = _build_encoder_view(
            _make_config(hyperencoder_packed_decoder=packed), decoder_hidden=64
        )
        assert view._attn_implementation == impl, (
            packed,
            view._attn_implementation,
        )
        assert view.cp_replicated_sequence is True


def test_encoder_view_builds_flashmask_layer_specs():
    """The flashmask view passes the layer-spec backend/impl assertion."""
    from paddlefleet.models.hyperencoder.layer_specs import (
        get_hyperencoder_layer_specs,
    )

    view = _build_encoder_view(
        _make_config(hyperencoder_packed_decoder=True), decoder_hidden=64
    )
    specs = get_hyperencoder_layer_specs(view)
    assert len(specs) == view.num_hidden_layers


if __name__ == "__main__":
    try:
        test_packed_decoder_rope_tracks_hyperencoder_packed_decoder()
        test_flashmask_backend_and_packed_decoder_move_together()
        test_encoder_view_attn_implementation_follows_backend()
        test_encoder_view_builds_flashmask_layer_specs()
    except AssertionError:
        import traceback

        traceback.print_exc()
        sys.exit(1)
    print("OK")
