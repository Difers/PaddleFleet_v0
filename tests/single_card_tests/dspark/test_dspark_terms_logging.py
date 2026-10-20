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

"""Contract tests for the ``global_training_logs`` dspark metric channel.

Consumers of these keys live outside this repo (ernie5 / rl sum ``_num`` and
``_den`` across ranks to derive global means), so a renamed key is a breaking
change that no training run here can reveal. These tests pin the emitted key
set and the num/den pairing.

CPU only: the payload is built from plain tensors and a dict-like logs object.
"""

import unittest
from unittest import mock

import paddle

from paddlefleet.models.common.language_loss import language_loss as ll
from paddlefleet.training.global_vars import (
    set_global_training_logs,
    unset_global_variables,
)
from paddlefleet.utils.dspark.dspark_utils import LossTerms


class _RecordingLogs:
    """Minimal stand-in for the training logs object."""

    def __init__(self):
        self.payload = None

    def update(self, **kwargs):
        if self.payload is None:
            self.payload = {}
        self.payload.update(kwargs)


def _terms(num_pos=2):
    return LossTerms(
        loss=paddle.to_tensor(1.5),
        terms={
            "accept_rate_per_pos_num": paddle.to_tensor([3.0, 4.0][:num_pos]),
            "accept_rate_per_pos_den": paddle.to_tensor([6.0, 8.0][:num_pos]),
            "tau_num": paddle.to_tensor(1.0),
            "tau_den": paddle.to_tensor(2.0),
            "confidence_prob_num": paddle.to_tensor(5.0),
            "confidence_prob_den": paddle.to_tensor(10.0),
            "loss": paddle.to_tensor(2.0),
            "ce_loss": paddle.to_tensor(1.0),
            "l1_loss": paddle.to_tensor(0.5),
            "tv_loss": paddle.to_tensor(0.25),
            "confidence_loss": paddle.to_tensor(0.125),
            "normalized_weight_denominator": paddle.to_tensor(8.0),
            "weight_denominator": paddle.to_tensor(4.0),
        },
        valid_tokens=paddle.to_tensor(4),
    )


class TestDsparkTermsPublish(unittest.TestCase):
    def setUp(self):
        unset_global_variables()
        self.logs = _RecordingLogs()
        set_global_training_logs(self.logs)
        ll._dspark_terms_publish_failures = 0

    def tearDown(self):
        unset_global_variables()

    def test_emits_the_full_expected_key_set(self):
        ll._publish_dspark_terms(_terms())
        self.assertEqual(
            set(self.logs.payload),
            {
                "dspark_accept_k1_num",
                "dspark_accept_k1_den",
                "dspark_accept_k2_num",
                "dspark_accept_k2_den",
                "dspark_accept_all_num",
                "dspark_accept_all_den",
                "dspark_tau_num",
                "dspark_tau_den",
                "dspark_confidence_prob_num",
                "dspark_confidence_prob_den",
                "dspark_draft_num",
                "dspark_draft_den",
                "dspark_ce_num",
                "dspark_ce_den",
                "dspark_l1_num",
                "dspark_l1_den",
                "dspark_tv_num",
                "dspark_tv_den",
                "dspark_confidence_num",
                "dspark_confidence_den",
                "dspark_weight_tokens_local",
                "dspark_weight_tokens_group",
            },
        )

    def test_every_published_group_is_a_num_den_pair(self):
        ll._publish_dspark_terms(_terms())
        keys = set(self.logs.payload)
        for key in keys:
            if key.endswith("_num"):
                self.assertIn(
                    key[: -len("_num")] + "_den",
                    keys,
                    f"{key} has no paired denominator",
                )
            elif key.endswith("_den"):
                self.assertIn(
                    key[: -len("_den")] + "_num",
                    keys,
                    f"{key} has no paired numerator",
                )

    def test_numerator_is_the_term_times_the_group_denominator(self):
        ll._publish_dspark_terms(_terms())
        # ce_loss 1.0 * normalized_weight_denominator 8.0
        self.assertAlmostEqual(
            float(self.logs.payload["dspark_ce_num"]), 8.0, places=5
        )
        # local weight denominator wins for the published denominator
        self.assertAlmostEqual(
            float(self.logs.payload["dspark_ce_den"]), 4.0, places=5
        )

    def test_missing_optional_terms_are_omitted_not_zeroed(self):
        terms = LossTerms(
            loss=paddle.to_tensor(1.0),
            terms={"tau_num": paddle.to_tensor(1.0)},
        )
        ll._publish_dspark_terms(terms)
        self.assertEqual(self.logs.payload, {})

    def test_no_logs_object_is_a_noop(self):
        unset_global_variables()
        ll._publish_dspark_terms(_terms())  # must not raise

    def test_failure_is_counted_and_reported_periodically(self):
        class _Broken:
            def update(self, **kwargs):
                raise RuntimeError("logs channel unavailable")

        set_global_training_logs(_Broken())
        with mock.patch("builtins.print") as fake_print:
            for _ in range(3):
                ll._publish_dspark_terms(_terms())

            # One line on the first failure, then silence until the interval --
            # but the counter keeps climbing instead of latching off forever.
            self.assertEqual(fake_print.call_count, 1)
            self.assertIn("1 failures so far", fake_print.call_args.args[0])
            self.assertEqual(ll._dspark_terms_publish_failures, 3)

            for _ in range(ll._DSPARK_TERMS_WARN_EVERY - 3):
                ll._publish_dspark_terms(_terms())

            self.assertEqual(fake_print.call_count, 2)
            self.assertIn(
                f"{ll._DSPARK_TERMS_WARN_EVERY} failures so far",
                fake_print.call_args.args[0],
            )


if __name__ == "__main__":
    unittest.main()
