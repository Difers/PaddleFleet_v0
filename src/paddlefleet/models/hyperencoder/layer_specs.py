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

"""Layer specs for the HyperEncoder trunk.

## What this builds

The block is a standard GPT decoder block with four HyperEncoder-specific
adjustments:

1. Start from a standard per-layer decoder spec
   (``get_gpt_layer_local_spec`` with ``normalization="RMSNorm"``).
2. Replace each layer's ``input_layernorm`` and ``post_attention_layernorm``
   with :class:`~paddlefleet.models.hyperencoder.norm.HyperEncoderRMSNorm`, an
   RMSNorm that always computes in fp32.
3. Set the attention mask type per layer (see "attn_mask_type" below).
4. Replace the block's final layer norm with ``HyperEncoderRMSNorm`` as well
   (done in :func:`get_hyperencoder_block_spec`).

Note: PaddleFleet has no block-spec object, so the per-layer builder is called
in a loop and returns a plain list. Also note the layer-norm naming: the
pre-MLP norm is called ``post_attention_layernorm`` here.

## How ``attn_mask_type`` is handled

We pass **``AttnMaskType.no_mask``**: the prefix-LM semantics come entirely
from the mask we feed (``prefix_lm_mask``), so the framework must not add a
causal triangle on top. Under ``no_mask`` the flashmask branch of
``DotProductAttention`` runs with ``causal=False``, which is exactly what the
2-column row indices encode.

The other switch that must agree with the backend is ``_attn_implementation``
(set on the config, asserted in :func:`get_hyperencoder_layer_specs`):

* ``dp`` needs ``"eager"``: the dense mask is only honored by the eager branch;
  under bf16 any other value takes the SDPA branch, which forces
  ``is_causal=True`` and silently breaks the bidirectional semantics.
* ``flashmask`` needs a non-eager value: the eager branch ignores row indices,
  which would silently degrade to an unmasked (fully visible) attention.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from paddlefleet.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from paddlefleet.models.hyperencoder.attn_backend import (
    use_flashmask_encoder_attn,
)
from paddlefleet.models.hyperencoder.norm import HyperEncoderRMSNorm
from paddlefleet.transformer.enums import AttnMaskType
from paddlefleet.transformer.transformer_block import (
    TransformerBlockSublayersSpec,
)

if TYPE_CHECKING:
    from paddle.distributed.fleet.meta_parallel import LayerSpec

    from paddlefleet.transformer.transformer_config import TransformerConfig

__all__ = ["get_hyperencoder_layer_specs", "get_hyperencoder_block_spec"]


def _is_moe_layer(config: TransformerConfig, layer_idx: int) -> bool:
    """Whether layer ``layer_idx`` is a MoE layer.

    ``moe_layer_freq`` is a per-layer list of 0/1 flags, so the check is simply
    the ``layer_idx``-th entry; there is no ``i % N`` modulo semantics involved.
    A plain int is rejected because the modulo interpretation would misplace the
    MoE layers.
    """
    freq = config.moe_layer_freq
    if isinstance(freq, (list, tuple)):
        return bool(freq[layer_idx])
    raise TypeError(
        f"HyperEncoder requires moe_layer_freq to be a per-layer list, got "
        f"{type(freq).__name__}. Passing an int would trigger `i % N` or "
        "`(i+1) % N` semantics and misplace the MoE layers."
    )


def get_hyperencoder_layer_specs(config: TransformerConfig) -> list[LayerSpec]:
    """Build the list of per-layer ``LayerSpec`` objects."""
    eager = getattr(config, "_attn_implementation", None) == "eager"
    if use_flashmask_encoder_attn(config):
        if eager:
            raise ValueError(
                "hyperencoder_attn_backend='flashmask' requires a non-eager "
                "config._attn_implementation: the eager branch of "
                "DotProductAttention ignores attn_mask_startend_row_indices and "
                "would run an unmasked attention."
            )
    elif not eager:
        raise ValueError(
            "hyperencoder_attn_backend='dp' requires "
            "config._attn_implementation='eager'. Otherwise, under bf16, "
            "DotProductAttention takes the SDPA branch and forces "
            "is_causal=True, which silently breaks the bidirectional encoder "
            "semantics."
        )

    specs: list[LayerSpec] = []
    for i in range(config.num_hidden_layers):
        spec = get_gpt_layer_local_spec(
            config=config,
            layer_number=i,
            # dense vs. MoE is decided per layer by moe_layer_freq
            num_experts=config.n_routed_experts
            if _is_moe_layer(config, i)
            else None,
            moe_expert_fusion=config.moe_expert_fusion,
            normalization="RMSNorm",
            # See "How attn_mask_type is handled" in the module docstring.
            attn_mask_type=AttnMaskType.no_mask,
        )
        # Replace both norms with the fp32 RMSNorm. Note the pre-MLP norm is
        # named post_attention_layernorm in PaddleFleet.
        spec.sublayers_spec.input_layernorm = HyperEncoderRMSNorm
        spec.sublayers_spec.post_attention_layernorm = HyperEncoderRMSNorm
        specs.append(spec)
    return specs


def get_hyperencoder_block_spec(
    config: TransformerConfig,
) -> TransformerBlockSublayersSpec:
    """Build the spec for the full block (all layers plus the final norm)."""
    return TransformerBlockSublayersSpec(
        layer_specs=get_hyperencoder_layer_specs(config),
        layer_norm=HyperEncoderRMSNorm,
    )
