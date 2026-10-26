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

"""Wiring coverage for the ``log_global_token_avg_loss`` switch.

The switch has to be settable from the training YAML, and in this repo that
means being listed in ``LlmMetaConfig``: ``@llmmetaclass`` turns every listed key
into a field on the args dataclass (so ``PdArgumentParser.parse_dict`` accepts
the YAML key instead of rejecting it as unknown), and
``LlmMetaConfig.set_llm_config`` -- the single funnel from args to model config
-- forwards it onto whatever config object the workflow passes.

The HyperBody case is the one worth pinning. ``HyperBodyConfig.__getattr__``
reads through to ``decoder_config`` only when normal lookup FAILS, so a flag
declared on ``TransformerConfig`` alone would be invisible there; listing it in
``loss_attributes`` is what makes ``set_expected_keys`` seed it on the top-level
``__dict__`` and the funnel overwrite it with the YAML value. Without that
listing a YAML ``true`` would be silently dropped and the trainer would keep
logging the micro-average. Deliberately no ctor parameter on ``HyperBodyConfig``
-- the funnel is the only writer, same as ``moe_logging``.

The loss/trainer halves of the feature are covered by
``transformer/test_language_loss_token_avg.py`` and
``test_trainer_token_avg_log.py``.
"""

from __future__ import annotations

import dataclasses
import inspect
import types

from paddlefleet.transformer.transformer_config import TransformerConfig
from paddlefleet.transformers.configuration_utils import LlmMetaConfig
from paddlefleet.transformers.hyperbody.configuration import HyperBodyConfig

KEY = "log_global_token_avg_loss"

# Minimal geometry: HyperBody requires it (no defaults), but nothing here depends
# on the values.
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


def _make_hyperbody_config():
    return HyperBodyConfig(
        decoder_config=dict(_DECODER_GEOMETRY),
        encoder_config=dict(_ENCODER_GEOMETRY),
    )


def test_switch_is_listed_in_the_args_to_config_funnel():
    """``loss_attributes`` is the single listing that makes the key YAML-settable.

    It is in ``_get_init`` (so ``set_expected_keys`` seeds it), in
    ``_get_all_meta`` (so ``@llmmetaclass`` adds the args field) and in
    ``_get_defaults`` (so ``set_llm_config`` forwards it).
    """
    assert KEY in {attr[0] for attr in LlmMetaConfig.loss_attributes}
    assert LlmMetaConfig._get_init()[KEY] is False
    assert LlmMetaConfig._get_defaults()[KEY] is False
    assert KEY in {attr[0] for attr in LlmMetaConfig._get_all_meta()}


def test_transformer_config_declares_the_field_defaulting_off():
    fields = {f.name: f for f in dataclasses.fields(TransformerConfig)}
    assert KEY in fields
    assert fields[KEY].default is False
    assert getattr(TransformerConfig(), KEY) is False


def test_hyperbody_config_needs_no_ctor_parameter():
    """Funnel-only on HyperBody: no ctor param to keep in sync with the YAML, and
    the pre-funnel value is the off default."""
    assert KEY not in inspect.signature(HyperBodyConfig.__init__).parameters
    assert getattr(_make_hyperbody_config(), KEY) is False


def test_funnel_forwards_the_yaml_value_onto_a_hyperbody_config():
    """The regression this guards: the flag must land in the top-level
    ``__dict__``, where ``getattr`` finds it, and not be left to
    ``__getattr__``'s ``decoder_config`` read-through (which has no such key)."""
    cfg = _make_hyperbody_config()
    LlmMetaConfig.set_llm_config(cfg, types.SimpleNamespace(**{KEY: True}))

    assert cfg.__dict__[KEY] is True
    assert getattr(cfg, KEY) is True


def test_funnel_leaves_the_switch_off_when_the_yaml_omits_it():
    cfg = _make_hyperbody_config()
    LlmMetaConfig.set_llm_config(cfg, types.SimpleNamespace())

    assert getattr(cfg, KEY) is False


if __name__ == "__main__":
    import sys

    try:
        test_switch_is_listed_in_the_args_to_config_funnel()
        test_transformer_config_declares_the_field_defaulting_off()
        test_hyperbody_config_needs_no_ctor_parameter()
        test_funnel_forwards_the_yaml_value_onto_a_hyperbody_config()
        test_funnel_leaves_the_switch_off_when_the_yaml_omits_it()
    except AssertionError:
        import traceback

        traceback.print_exc()
        sys.exit(1)
    print("OK")
