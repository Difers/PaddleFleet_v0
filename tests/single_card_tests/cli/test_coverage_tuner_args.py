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
"""Stages other than SFT reject bias_activation_fusion instead of dropping it."""

import unittest
from types import SimpleNamespace
from unittest import mock

from paddlefleet.cli.train import tuner


def _args(stage, bias_activation_fusion):
    return (
        SimpleNamespace(stage=stage),
        SimpleNamespace(dataset_type="offline"),
        None,
        None,
        SimpleNamespace(bias_activation_fusion=bias_activation_fusion),
    )


class BiasActivationFusionStageTests(unittest.TestCase):
    def _run(self, stage, value):
        calls = []
        with mock.patch.multiple(
            tuner,
            get_train_args=lambda args: _args(stage, value),
            run_sft=lambda *a: calls.append("sft"),
            run_dpo=lambda *a: calls.append("dpo"),
        ):
            tuner._training_function({"args": {}})
        return calls

    def test_non_sft_stages_reject_the_switch(self):
        with self.assertRaisesRegex(ValueError, "stage='DPO'"):
            self._run("DPO", False)

    def test_sft_and_unset_values_dispatch_as_before(self):
        self.assertEqual(self._run("SFT", False), ["sft"])
        self.assertEqual(self._run("DPO", None), ["dpo"])


if __name__ == "__main__":
    unittest.main()
