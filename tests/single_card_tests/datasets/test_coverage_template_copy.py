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
"""Single card tests for ``get_template_and_fix_tokenizer``.

The helper hands each caller its own copy of the registered template, so the
per-run ``enable_thinking`` switch, ``default_system`` and the eos suffix
backfill never leak into the shared ``TEMPLATES`` registry.
"""

import unittest
from types import SimpleNamespace

from paddlefleet.datasets.template.template import (
    TEMPLATES,
    get_template_and_fix_tokenizer,
)


def _dataset_config(template, **overrides):
    values = {
        "tokenizer": SimpleNamespace(eos_token="<eos>", chat_template=None),
        "template": template,
        "tool_format": None,
        "default_system": None,
    }
    values.update(overrides)
    return values


class TemplateCopyTests(unittest.TestCase):
    def test_enable_thinking_is_applied_to_the_copy_only(self):
        registered = TEMPLATES["glm5_2"]
        default = registered.enable_thinking
        for value in (False, True):
            with self.subTest(enable_thinking=value):
                result = get_template_and_fix_tokenizer(
                    _dataset_config("glm5_2", enable_thinking=value)
                )
                self.assertIsNot(result, registered)
                self.assertIs(type(result), type(registered))
                self.assertIs(result.enable_thinking, value)
                self.assertEqual(registered.enable_thinking, default)

    def test_an_absent_switch_keeps_the_registered_default(self):
        registered = TEMPLATES["glm5_2"]
        for config in (
            _dataset_config("glm5_2"),
            _dataset_config("glm5_2", enable_thinking=None),
        ):
            with self.subTest(config=config.get("enable_thinking", "absent")):
                result = get_template_and_fix_tokenizer(config)
                self.assertEqual(
                    result.enable_thinking, registered.enable_thinking
                )
                self.assertEqual(result.suffix, registered.suffix)

    def test_per_run_overrides_do_not_reach_the_registry(self):
        registered = TEMPLATES["empty"]
        system = registered.default_system
        # "empty" registers no suffix, so the helper backfills the eos token.
        self.assertEqual(registered.suffix, [])
        result = get_template_and_fix_tokenizer(
            _dataset_config(
                "empty", default_system="run system", enable_thinking=False
            )
        )
        self.assertEqual(result.default_system, "run system")
        self.assertFalse(result.enable_thinking)
        self.assertEqual(result.suffix, ["<eos>"])
        # None of it lands on the shared template.
        self.assertEqual(registered.default_system, system)
        self.assertEqual(registered.suffix, [])


if __name__ == "__main__":
    unittest.main()
