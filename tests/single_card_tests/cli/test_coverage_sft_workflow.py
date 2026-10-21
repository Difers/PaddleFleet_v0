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
"""Single card tests for the SFT CLI workflow wiring.

``run_sft`` is driven with every collaborator stubbed and stopped as soon as
the template is requested, so nothing starts a process group or loads a
checkpoint.
"""

import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import paddle

from paddlefleet.cli.train.sft import workflow as wf
from paddlefleet.transformers.glm_moe_dsa.configuration import (
    GlmMoeDsaConfig,
)


class BiasActivationFusionTests(unittest.TestCase):
    """The option reaches every model config, not only GLM MoE DSA."""

    def _config(self, **values):
        registered = []
        config = SimpleNamespace(
            register_unsavable_keys=registered.append, **values
        )
        return config, registered

    def test_a_configured_value_reaches_any_model_config(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                config, registered = self._config(model_type="qwen2_moe")
                wf.apply_bias_activation_fusion(
                    config, SimpleNamespace(bias_activation_fusion=enabled)
                )
                self.assertIs(config.bias_activation_fusion, enabled)
                # A runtime kernel switch, kept out of the exported config.
                self.assertEqual(registered, ["bias_activation_fusion"])

    def test_the_exported_config_drops_the_switch(self):
        config = GlmMoeDsaConfig()
        wf.apply_bias_activation_fusion(
            config, SimpleNamespace(bias_activation_fusion=False)
        )
        self.assertIs(config.to_dict()["bias_activation_fusion"], False)
        self.assertNotIn(
            "bias_activation_fusion", config.to_dict(saving_file=True)
        )

    def test_none_keeps_the_model_config_value(self):
        config, registered = self._config(bias_activation_fusion=True)
        wf.apply_bias_activation_fusion(
            config, SimpleNamespace(bias_activation_fusion=None)
        )
        self.assertTrue(config.bias_activation_fusion)
        self.assertEqual(registered, [])


class _Stop(Exception):
    """Raised by the template stub to end ``run_sft`` early."""


class _RunSftConfig(SimpleNamespace):
    """Provider config double; ``run_sft`` asks for the text sub-config."""

    def get_text_config(self):
        return self

    def register_unsavable_keys(self, key):
        pass


def _model_args(**overrides):
    values = {
        "model_name_or_path": "/models/glm52",
        "download_hub": None,
        "copy_custom_file_list": None,
        "use_fast_layer_norm": False,
        "pp_seg_method": "layer:GlmMoeDsaDecoderLayer",
        "_attn_implementation": "eager",
        "lora": False,
        "moe_logging": False,
        "mtp_attention_flexible": True,
        "stage": "SFT",
        "continue_training": True,
        "neftune": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _data_args(**overrides):
    values = {
        "max_seq_len": 4,
        "processor_use_fast": False,
        "new_special_tokens_path": None,
        "num_samples_each_epoch": 1,
        "random_shuffle": False,
        "greedy_intokens": False,
        "packing": False,
        "mix_strategy": "concat",
        "encode_one_turn": False,
        "use_template": True,
        "truncate_packing": False,
        "template_backend": "custom",
        "split_multi_turn": False,
        "dataset_type": "offline",
        "binpacking": False,
        "packing_interval": 1,
        "packed_idx_cache_dir": None,
        "template": "glm5_2",
        "truncation_strategy": "right",
        "skip_warmup": True,
        "warmup_only_rank0": True,
        "make_offline_data": False,
        "eval_with_do_generation": False,
        "input_dir": "/data/glm52",
        "padding_free": False,
        "train_dataset_path": None,
        "train_dataset_prob": None,
        "train_dataset_type": None,
        "eval_dataset_path": None,
        "eval_dataset_prob": None,
        "eval_dataset_type": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _training_args(output_dir, **overrides):
    values = {
        "print_config": lambda *args, **kwargs: None,
        "pre_alloc_memory": 0,
        "device": "gpu",
        "seed": 42,
        "local_rank": -1,
        "world_size": 1,
        "fp16": False,
        "bf16": True,
        "fp16_opt_level": "O2",
        "output_dir": output_dir,
        "do_train": True,
        "do_eval": False,
        "overwrite_output_dir": True,
        "resume_from_checkpoint": None,
        "weight_quantize_algo": None,
        "pad_token_id": None,
        "prediction_loss_only": False,
        "use_expert_parallel": False,
        "enable_auto_parallel": False,
        "freeze_config": "",
        "compute_type": "bf16",
        "dataset_world_size": 1,
        "dataset_rank": 0,
        "dataloader_num_workers": 0,
        "dataset_num_proc": 1,
        "convert_from_hf": False,
        "load_via_cpu": False,
        "load_checkpoint_format": "unified",
        "autotuner_benchmark": False,
        "should_load_dataset": True,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "global_batch_size": 1,
        "max_steps": 2,
        "decay_steps": 2,
        "num_train_epochs": 1,
        "save_strategy": wf.IntervalStrategy.STEPS,
        "evaluation_strategy": wf.IntervalStrategy.STEPS,
        "logging_strategy": wf.IntervalStrategy.STEPS,
        "train_mtp_only": False,
        "num_nextn_predict_layers": 0,
        "mtp_num_layers": 0,
        "bias_activation_fusion": None,
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _RunSftHarness(unittest.TestCase):
    """Drive ``run_sft`` with every collaborator stubbed."""

    def setUp(self):
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        self.output_dir = holder.name
        self.record = SimpleNamespace(llm_configs=[], templates=[])
        self.model_config = _RunSftConfig(
            model_type="glm_moe_dsa",
            architectures=["GlmMoeDsaForCausalLM"],
            tie_word_embeddings=False,
            quantization_config=SimpleNamespace(
                is_weight_quantize=lambda: False
            ),
            mtp_num_layers=0,
        )
        self.model = SimpleNamespace(is_fleet=False, parameters=lambda: [])
        self.tokenizer = SimpleNamespace(
            pad_token_id=None, eos_token_id=7, chat_template=None
        )
        patcher = mock.patch.multiple(
            wf,
            AutoConfig=SimpleNamespace(
                from_pretrained=lambda path, **kwargs: self.model_config
            ),
            QuantizationConfig=SimpleNamespace(
                from_dict=lambda payload: SimpleNamespace(
                    is_weight_quantize=lambda: False
                )
            ),
            LlmMetaConfig=SimpleNamespace(set_llm_config=self._llm_config),
            AutoModelForCausalLM=SimpleNamespace(
                from_pretrained=lambda path, **kwargs: self.model,
                from_config=lambda config, **kwargs: self.model,
            ),
            AutoTokenizer=SimpleNamespace(
                from_pretrained=lambda path: self.tokenizer
            ),
            AutoProcessor=SimpleNamespace(
                from_pretrained=lambda path, use_fast=None: SimpleNamespace()
            ),
            add_new_special_tokens=lambda tokenizer, path: None,
            get_template_and_fix_tokenizer=self._template,
            set_random_seed=lambda **kwargs: None,
            set_seed=lambda **kwargs: None,
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        device_patcher = mock.patch.object(
            paddle, "set_device", lambda device: None
        )
        device_patcher.start()
        self.addCleanup(device_patcher.stop)

    def _llm_config(self, config, training_args):
        # bias_activation_fusion must not have been applied yet.
        self.record.llm_configs.append(
            getattr(config, "bias_activation_fusion", "unset")
        )

    def _template(self, dataset_config):
        self.record.templates.append(dict(dataset_config))
        raise _Stop

    def _run(self, model_args=None, data_args=None, **training):
        with self.assertRaises(_Stop):
            wf.run_sft(
                model_args or _model_args(),
                data_args or _data_args(),
                SimpleNamespace(enable_thinking=training.pop("thinking", True)),
                _training_args(self.output_dir, **training),
            )


class RunSftWiringTests(_RunSftHarness):
    """``run_sft`` plumbing up to the template request."""

    def test_bias_activation_fusion_is_applied_after_set_llm_config(self):
        self._run(bias_activation_fusion=False)
        self.assertEqual(self.record.llm_configs, ["unset"])
        self.assertIs(self.model_config.bias_activation_fusion, False)

    def test_the_glm5_2_template_receives_the_thinking_switch(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                self.record.templates.clear()
                self._run(thinking=enabled)
                config = self.record.templates[0]
                self.assertEqual(config["template"], "glm5_2")
                self.assertIs(config["enable_thinking"], enabled)

    def test_other_templates_keep_their_registered_thinking_default(self):
        self._run(data_args=_data_args(template="qwen3_vl"), thinking=False)
        self.assertEqual(self.record.templates[0]["template"], "qwen3_vl")
        self.assertNotIn("enable_thinking", self.record.templates[0])


if __name__ == "__main__":
    unittest.main()
