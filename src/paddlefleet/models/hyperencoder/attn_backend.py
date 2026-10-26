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

"""Attention backend selection for HyperEncoder.

The backend and packed-decoder path are driven by config fields rather than
environment variables, so they are declared with defaults, validated centrally
(in ``HyperEncoderProvider.__post_init__``, which calls these functions), and
serialized with the rest of the config:

| field | values | effect |
|---|---|---|
| ``hyperencoder_attn_backend`` | ``dp`` (default) / ``flashmask`` | ``dp``: eager attention with a dense ``[1,1,T,T]`` mask; ``flashmask``: the standard ``DotProductAttention`` flashmask branch with ``[1,1,T,2]`` row indices |
| ``hyperencoder_packed_decoder`` | ``False`` (default) / ``True`` | run the trunk with a single packed call |

The two switches move together: ``flashmask`` is the packed path and ``dp`` is
the dense one (:func:`use_packed_decoder` rejects the mixed combinations). Both
keep the stock ``DotProductAttention`` as ``core_attention`` and differ only in
which mask is fed and in ``_attn_implementation`` (``eager`` for ``dp``,
non-eager for ``flashmask``). The encoder is never sharded over CP: every CP
rank holds the whole sequence (``cp_replicated_sequence``), so
``DotProductAttention`` runs it with a CP size of 1.
"""

from __future__ import annotations

__all__ = [
    "encoder_attn_backend",
    "use_flashmask_encoder_attn",
    "use_packed_decoder",
]

_BACKENDS = ("dp", "flashmask")


def encoder_attn_backend(config) -> str:
    """Return ``dp`` or ``flashmask``."""
    v = str(getattr(config, "hyperencoder_attn_backend", "dp")).lower()
    if v not in _BACKENDS:
        raise ValueError(
            f"Unknown hyperencoder_attn_backend={v!r}; expected one of {_BACKENDS}"
        )
    return v


def use_flashmask_encoder_attn(config) -> bool:
    """Whether the encoder trunk is fed FlashMask row indices."""
    return encoder_attn_backend(config) == "flashmask"


def use_packed_decoder(config) -> bool:
    """Whether the trunk runs as a single packed call."""
    on = bool(getattr(config, "hyperencoder_packed_decoder", False))
    # Packed means several mutually invisible segments in one [1, T] sequence:
    # only the O(T) FlashMask row indices express that at real lengths. Dense
    # means one rectangular segment fed as a [T, T] mask, which only the eager
    # branch honors. Mixing them would silently run the wrong mask.
    if on != use_flashmask_encoder_attn(config):
        raise RuntimeError(
            "hyperencoder_packed_decoder=True requires "
            "hyperencoder_attn_backend='flashmask', and 'flashmask' requires "
            f"hyperencoder_packed_decoder=True; got packed={on}, "
            f"backend={encoder_attn_backend(config)!r}."
        )
    return on
