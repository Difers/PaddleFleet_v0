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

"""InternalMedicineCallback: what reaches the jsonl rows and the extra sinks.

The callback is all there is between a monitor's metrics and whoever reports
them, and it runs inside a real training step, so the monitors are mocked out
and each test drives one ``on_*`` event: a sink that raises must not take the
step down with it, a non-scalar entry must not reach a row, and the optimizer
has to be offered to the monitors that cannot exist before it does.

It sits with the single-card tests rather than beside the other
trainer-callback tests because this is the suite that collects coverage for
``src/paddlefleet/trainer``.
"""

import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

importlib.import_module("_backend_env").skip_unless_backend("paddlefleet")

try:
    InternalMedicineCallback = importlib.import_module(
        "paddlefleet.trainer.trainer_callback"
    ).InternalMedicineCallback
except (
    Exception
) as exc:  # pragma: no cover - depends on optional backend install
    raise unittest.SkipTest(f"paddle backend unavailable: {exc}") from exc


def _callback(aggregated, log_dir=""):
    """A callback past setup whose single monitor sampled this step."""
    callback = InternalMedicineCallback(monitor_interval=5, log_dir=log_dir)
    callback._setup_done = True
    callback._monitor_dict = {
        "monitor": SimpleNamespace(sampled_this_step=True)
    }
    callback._training_logs = Mock()
    callback._training_logs.gather_and_aggregate.return_value = aggregated
    return callback


class InternalMedicineCallbackSinkTest(unittest.TestCase):
    def test_a_sink_gets_the_same_scalars_as_the_jsonl_row(self):
        with tempfile.TemporaryDirectory() as log_dir:
            callback = _callback(
                {"qk/global_b": 2, "qk/global_a": 1.5, "qk/vec": [1.0, 2.0]},
                log_dir=log_dir,
            )
            sink = Mock()
            callback.add_sink(sink)
            state = SimpleNamespace(global_step=101)

            callback.on_log("args", state, None)

            rows = [
                json.loads(line)
                for line in Path(callback.log_path).read_text().splitlines()
            ]
        # Float scalars in key order; the non-scalar entry reaches neither.
        metrics = {"qk/global_a": 1.5, "qk/global_b": 2.0}
        self.assertEqual(rows, [{"global_step": 101, **metrics}])
        sink.assert_called_once()
        sink_args, sink_state, sink_metrics = sink.call_args[0]
        self.assertEqual(sink_args, "args")
        self.assertIs(sink_state, state)
        self.assertEqual(list(sink_metrics.items()), list(metrics.items()))

    def test_a_failing_sink_does_not_stop_the_others(self):
        callback = _callback({"qk/global_a": 1.0})
        failing = Mock(side_effect=RuntimeError("tracker down"))
        healthy = Mock()
        callback.add_sink(failing)
        callback.add_sink(healthy)

        callback.on_log(None, SimpleNamespace(global_step=6), None)

        failing.assert_called_once()
        healthy.assert_called_once()

    def test_the_jsonl_row_is_written_with_no_sink_registered(self):
        """Sinks are an addition: with none, the jsonl row is still the row."""
        with tempfile.TemporaryDirectory() as log_dir:
            callback = _callback({"qk/global_a": 1.0}, log_dir=log_dir)

            callback.on_log(None, SimpleNamespace(global_step=7), None)

            rows = [
                json.loads(line)
                for line in Path(callback.log_path).read_text().splitlines()
            ]
        self.assertEqual(rows, [{"global_step": 7, "qk/global_a": 1.0}])

    def test_nothing_aggregated_means_no_sink_call(self):
        callback = _callback({})
        sink = Mock()
        callback.add_sink(sink)

        callback.on_log(None, SimpleNamespace(global_step=6), None)

        sink.assert_not_called()

    def test_the_optimizer_is_offered_to_the_monitors_that_take_one(self):
        """optim_health can only wrap the optimizer once it exists."""
        callback = _callback({})
        attaching = SimpleNamespace(attach_optimizer=Mock(return_value=True))
        plain = SimpleNamespace()
        callback._monitor_dict = {"optim_health": attaching, "qk": plain}
        optimizer = object()

        callback.on_optimizer_begin(
            None, None, None, optimizer=optimizer, scaler=None
        )

        attaching.attach_optimizer.assert_called_once_with(optimizer)


if __name__ == "__main__":
    unittest.main()
