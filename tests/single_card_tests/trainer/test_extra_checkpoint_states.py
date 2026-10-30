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

"""``Trainer._extra_checkpoint_states`` lands in every checkpoint directory.

The regular save writes it into ``output_dir/checkpoint-N``; with zero cost
checkpoint the callback hands it to the worker in the PREPARE payload and the
worker writes it into whichever directory it dumps (flash device and/or
persistent), before ``saved_signal_{rank}``. Methods run on bare objects, so no
process group or GPU is needed.
"""

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import paddle

import paddlefleet.trainer.utils.zero_cost_checkpoint as zcc
from paddlefleet.trainer.trainer import Trainer
from paddlefleet.trainer.utils.zero_cost_checkpoint import (
    ZeroCostCheckpointCallback,
    ZeroCostCheckpointWorker,
)

EXTRA = {"data_state_rank000.pdparams": {"cursor": 42}}


class _RecordingManager:
    def __init__(self):
        self.global_step = None
        self.payloads = []

    def get_idle_worker_for_saving(self, payload):
        self.payloads.append(payload)

    def zcc_pipeline_hook(self, hook_id):
        pass


def _noop_timer():
    return SimpleNamespace(
        start=lambda *a, **k: None, stop=lambda *a, **k: None
    )


class ExtraCheckpointStatesTest(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = self._tmpdir.name

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_default_hook_is_empty(self):
        self.assertEqual(Trainer._extra_checkpoint_states(object()), {})

    def test_regular_save_writes_extra_states(self):
        trainer = Trainer.__new__(Trainer)
        trainer.args = SimpleNamespace(
            enable_zero_cost_checkpoint=False,
            output_dir=self.root,
            output_signal_dir=self.root,
            enable_auto_parallel=False,
            metric_for_best_model=None,
            should_save=False,
            save_rng_states=False,
            ignore_save_lr_and_optim=True,
            unified_checkpoint_config=[],
        )
        trainer.state = SimpleNamespace(global_step=30)
        trainer.model = object()
        trainer.runtime_timer = _noop_timer()
        trainer.save_model = lambda *a, **k: None
        trainer._rotate_saved_checkpoints = lambda *a, **k: None
        trainer._extra_checkpoint_states = lambda: EXTRA

        trainer._save_checkpoint(trainer.model)

        path = os.path.join(
            self.root, "checkpoint-30", "data_state_rank000.pdparams"
        )
        self.assertEqual(paddle.load(path), {"cursor": 42})

    def test_zcc_callback_hands_extra_states_to_worker_on_save(self):
        mgr = _RecordingManager()
        cb = ZeroCostCheckpointCallback(
            SimpleNamespace(zcc_ema_interval=2), mgr, _noop_timer(), None
        )
        cb.maybe_update_zcc_worker = lambda *a, **k: None
        cb.extra_states_fn = lambda: EXTRA
        args = SimpleNamespace(
            zcc_save_ema_coef=0.999,
            pipeline_model_parallel_size=1,
            save_rng_states=False,
            flash_device_save_steps=10,
            save_steps=0,
            output_dir=self.root,
        )
        lr = SimpleNamespace(state_dict=lambda: {})
        for should_save, step in ((True, 10), (False, 12)):
            cb.on_step_end(
                args,
                SimpleNamespace(global_step=step),
                SimpleNamespace(should_save=should_save),
                model=object(),
                lr_scheduler=lr,
                optimizer=object(),
            )
        (_, saved), (_, ema_only) = mgr.payloads
        self.assertEqual(saved[3], EXTRA)
        self.assertIsNone(ema_only[3])  # EMA offload dumps nothing

    def test_setup_zcc_callback_wires_the_trainer_hook(self):
        trainer = Trainer.__new__(Trainer)
        trainer.args = SimpleNamespace(zcc_ema_interval=2)
        trainer.zcc_manager = _RecordingManager()
        trainer.runtime_timer = _noop_timer()
        trainer.sharding_io = None
        trainer._extra_checkpoint_states = lambda: EXTRA
        added = []
        trainer.add_callback = added.append

        trainer._setup_zcc_callback(ZeroCostCheckpointCallback)

        self.assertEqual(added[0].extra_states_fn(), EXTRA)

    def test_zcc_worker_dumps_extra_states(self):
        worker = ZeroCostCheckpointWorker.__new__(ZeroCostCheckpointWorker)
        worker.offloaded_numels = 0
        worker.status = SimpleNamespace(value=None)
        worker.device_id = 1  # skip the device-0-only lr / trainer_state files
        flash = os.path.join(self.root, "flash", "checkpoint-10")
        os.makedirs(flash)
        worker.process_prepare_task(((flash, None), ({}, None, None, EXTRA)))

        worker._dump_args_and_state(worker.flash_device_save_dir)

        path = os.path.join(flash, "data_state_rank000.pdparams")
        self.assertEqual(paddle.load(path), {"cursor": 42})

    def test_zcc_dump_writes_extra_states_before_saved_signal(self):
        worker = ZeroCostCheckpointWorker.__new__(ZeroCostCheckpointWorker)
        worker.offloaded_numels = 0
        worker.status = SimpleNamespace(value=None)
        worker.device_id = 1  # skip the device-0-only static objects
        worker.global_rank = 0
        worker._dump_states = lambda output_dir: None  # needs fused buffers
        flash = os.path.join(self.root, "flash", "checkpoint-10")
        signal = os.path.join(flash, "saved_signal_0")
        worker.process_prepare_task(((flash, None), ({}, None, None, EXTRA)))

        signal_seen_at_save = []
        real_save = paddle.save

        def recording_save(obj, path, *args, **kwargs):
            signal_seen_at_save.append(os.path.exists(signal))
            return real_save(obj, path, *args, **kwargs)

        with patch.object(zcc.paddle, "save", side_effect=recording_save):
            worker.process_dump_task_impl(flash)

        self.assertEqual(signal_seen_at_save, [False])
        self.assertTrue(os.path.exists(signal))
        path = os.path.join(flash, "data_state_rank000.pdparams")
        self.assertEqual(paddle.load(path), {"cursor": 42})


if __name__ == "__main__":
    unittest.main()
