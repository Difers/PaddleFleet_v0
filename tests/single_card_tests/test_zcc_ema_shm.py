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

"""Single-card tests for the ZCC EMA shared-memory HF export path.

The feature spans two processes: the ZCC worker keeps the live EMA private, and
on HF-checkpoint steps copies it into ``/dev/shm`` and hands the handles over;
the main process rebuilds tensors from those handles and writes the HF export.
The tests below follow that pipeline:

* ``TestEmaSnapshotExport`` -- worker allocates, copies, describes; main process
  rebuilds. Includes the bfloat16 degradation and the allocation failure modes.
* ``TestEmaAccumulate`` -- the in-place ``lerp_`` update of the live buffers.
* ``TestEmaMetadataGuards`` -- ``unpack_ema_data`` rejects inconsistent metadata
  instead of silently reinterpreting memory.
* ``TestEmaReadinessCoordination`` -- the wait/ready/gone decision and the
  manager-side readiness and hand-off plumbing.
* ``TestWorkerPublishesSnapshot`` -- which steps publish, and how the EMA slots
  are wired from manager to worker.
* ``TestAssemblerHfExport`` -- the pending-step state machine that drives it.

Everything runs in a single process on CPU (no GPU, no process group), except
one GPU-guarded assertion about device placement.
"""

import os
import queue
import sys
import threading
import unittest
from multiprocessing import shared_memory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import paddle

_SRC = os.path.join(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ),
    "src",
)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from paddlefleet.trainer.trainer_utils import (
    EMAStateAssembler,
    unpack_ema_data,
)
from paddlefleet.trainer.utils.zero_cost_checkpoint import (
    _SHM_DIR,
    ZCCTaskType,
    ZeroCostCheckpointEMAProcessor,
    ZeroCostCheckpointManager,
    ZeroCostCheckpointWorker,
    cleanup_ema_shared_memory,
)

_ZCC = "paddlefleet.trainer.utils.zero_cost_checkpoint"


def _fp32(values):
    return paddle.to_tensor(
        np.asarray(values, dtype=np.float32), place=paddle.CPUPlace()
    )


def _bf16_bits(flat):
    """float32 -> bfloat16 bit pattern, round-to-nearest-even (matches paddle)."""
    u32 = np.ascontiguousarray(flat, dtype=np.float32).view(np.uint32)
    rounded = u32 + 0x7FFF + ((u32 >> 16) & 1)
    return (rounded >> 16).astype(np.uint16)


def _bf16_roundtrip(flat):
    """The float32 values that survive a bfloat16 round-trip."""
    return (_bf16_bits(flat).astype(np.uint32) << 16).view(np.float32)


def _slot(value):
    """Stand-in for a ``multiprocessing.Value``."""
    return SimpleNamespace(value=value)


def _dead_pid():
    """A pid with no live process, so its segments count as stale."""
    for candidate in range(4194303, 4000000, -1):
        if not os.path.exists(f"/proc/{candidate}"):
            return candidate
    raise unittest.SkipTest("no free pid to impersonate")


class _EmaProcessorFixture(unittest.TestCase):
    """Real ``ZeroCostCheckpointEMAProcessor`` objects over fake fusion-storage
    helpers -- enough state for the EMA code, no GPU and no process group.

    The master metas deliberately start at offset 4 so ``shm_package`` has to
    rebase them onto the EMA buffer, and ``buf_bf16`` is a non-float32 buffer that
    must be left out of the EMA entirely.
    """

    MASTER_METAS = {
        "mw1": {"start": 4, "end": 8, "shape": [2, 2], "name": "mw1"},
        "mw2": {"start": 8, "end": 12, "shape": [2, 2], "name": "mw2"},
    }
    PARAM_METAS = {
        "model.A": {
            "buffer_index": "b0",
            "start": 0,
            "end": 6,
            "shape": [2, 3],
            "name": "model.A",
        },
        "model.B": {
            "buffer_index": "b0",
            "start": 6,
            "end": 10,
            "shape": [2, 2],
            "name": "model.B",
        },
        # no start/end: an `unshard_` buffer is used whole
        "model.C": {
            "buffer_index": "unshard_0",
            "shape": [2, 2],
            "name": "model.C",
        },
        # backed by a bfloat16 buffer, so it has no EMA at all
        "model.D": {
            "buffer_index": "buf_bf16",
            "start": 0,
            "end": 4,
            "shape": [2, 2],
            "name": "model.D",
        },
    }

    def _make_processor(self, global_rank=7):
        bf16 = _fp32(np.zeros(4)).astype(paddle.bfloat16)
        param_helper = SimpleNamespace(
            inited_buffers={
                "b0": (_fp32(np.zeros(10)), _fp32(np.zeros(10))),
                "unshard_0": (_fp32(np.zeros(4)), _fp32(np.zeros(4))),
                "buf_bf16": (bf16, bf16),
            },
            model_weights_metas=self.PARAM_METAS,
        )
        proc = ZeroCostCheckpointEMAProcessor(
            SimpleNamespace(master_weights_meta=self.MASTER_METAS),
            param_helper,
            ema_coef=0.99,
            global_rank=global_rank,
        )
        self.addCleanup(proc.unlink_shared_memory)
        return proc

    def setUp(self):
        self.proc = self._make_processor()

    def _fill_live_ema(self, proc=None):
        """Put recognisable values in the live buffers; return the expectations."""
        proc = proc or self.proc
        master = np.arange(8, dtype=np.float32) + 300
        b0 = np.arange(10, dtype=np.float32) + 10
        unshard = np.arange(4, dtype=np.float32) + 200
        paddle.assign(_fp32(master), proc.ema_buffer)
        paddle.assign(_fp32(b0), proc.ema_buffer_model_params["b0"])
        paddle.assign(_fp32(unshard), proc.ema_buffer_model_params["unshard_0"])
        return {
            "model.A": b0[0:6].reshape(2, 3),
            "model.B": b0[6:10].reshape(2, 2),
            "model.C": unshard.reshape(2, 2),
            "mw1": master[0:4].reshape(2, 2),
            "mw2": master[4:8].reshape(2, 2),
        }

    def _shm_paths(self, proc=None):
        proc = proc or self.proc
        return [os.path.join(_SHM_DIR, s.name) for s in proc._shm_segments]


class TestEmaSnapshotExport(_EmaProcessorFixture):
    """The whole worker-to-main-process hand-off, end to end."""

    def test_float32_snapshot_round_trips_to_the_main_process(self):
        expected = self._fill_live_ema()

        # the buffer spans only the master range, and non-float32 buffers are out
        self.assertEqual(
            (self.proc.master_min_offset, self.proc.master_max_offset), (4, 12)
        )
        self.assertEqual(list(self.proc.ema_buffer.shape), [8])
        self.assertEqual(
            set(self.proc.ema_buffer_model_params), {"b0", "unshard_0"}
        )

        self.proc.take_ema_snapshot()
        segments = list(self.proc._shm_segments)
        self.proc.take_ema_snapshot()  # a second HF step reuses the buffers
        self.assertEqual(self.proc._shm_segments, segments)

        self.assertEqual(self.proc.ema_snapshot.dtype, paddle.float32)
        handle = self.proc.ema_buffer_shm_handle
        self.assertEqual((handle["dtype"], handle["shape"]), ("float32", [8]))
        self.assertIn("_r7_", handle["name"])  # per-rank segment naming
        self.assertEqual(
            set(self.proc.ema_buffer_model_params_shm_handles),
            {"b0", "unshard_0"},
        )
        for path in self._shm_paths():
            self.assertTrue(os.path.exists(path), path)

        package = self.proc.shm_package()
        metas = package["metas"]
        # master offsets are rebased onto the EMA buffer
        self.assertEqual(
            metas["master_weights"]["mw1"],
            {"start": 0, "end": 4, "shape": [2, 2], "name": "mw1"},
        )
        self.assertEqual(
            metas["master_weights"]["mw2"],
            {"start": 4, "end": 8, "shape": [2, 2], "name": "mw2"},
        )
        self.assertFalse(metas["model.A"]["whole_buffer"])
        self.assertTrue(metas["model.C"]["whole_buffer"])
        self.assertIsNone(metas["model.C"]["start"])
        self.assertNotIn("model.D", metas)  # bfloat16 buffer has no EMA

        out = unpack_ema_data(package)
        for key in ("model.A", "model.B", "model.C"):
            np.testing.assert_array_equal(out[key].numpy(), expected[key])
            self.assertEqual(list(out[key].shape), list(expected[key].shape))
            self.assertEqual(out[key].name, key)
        for key in ("mw1", "mw2"):
            got = out["master_weights"][key]
            np.testing.assert_array_equal(got.numpy(), expected[key])
            self.assertEqual(got.name, key)

        # the snapshot is a copy: further accumulation must not disturb a read
        before = self.proc.ema_snapshot.numpy().copy()
        paddle.assign(_fp32(np.zeros(8)), self.proc.ema_buffer)
        np.testing.assert_array_equal(self.proc.ema_snapshot.numpy(), before)
        np.testing.assert_array_equal(
            out["master_weights"]["mw1"].numpy(), expected["mw1"]
        )

    def test_snapshot_degrades_to_bfloat16_and_still_round_trips(self):
        expected = self._fill_live_ema()
        real = self.proc._alloc_shared_memory
        seen = []

        def no_room_for_float32_master(key, shape, dtype):
            seen.append((key, dtype))
            if key == "master" and dtype == paddle.float32:
                raise RuntimeError("simulated /dev/shm exhaustion")
            return real(key, shape, dtype)

        self.proc._alloc_shared_memory = no_room_for_float32_master
        self.proc.take_ema_snapshot()

        self.assertEqual(
            seen[:2], [("master", paddle.float32), ("master", paddle.bfloat16)]
        )
        self.assertEqual(self.proc.ema_snapshot.dtype, paddle.bfloat16)
        self.assertEqual(self.proc.ema_buffer_shm_handle["dtype"], "bfloat16")
        # only the master degrades; the model-weight snapshots stay float32
        for dst in self.proc.ema_snapshot_model_params.values():
            self.assertEqual(dst.dtype, paddle.float32)

        out = unpack_ema_data(self.proc.shm_package())
        for key in ("mw1", "mw2"):
            got = out["master_weights"][key]
            self.assertEqual(got.dtype, paddle.bfloat16)
            np.testing.assert_array_equal(
                got.astype("float32").numpy(),
                _bf16_roundtrip(expected[key].reshape(-1)).reshape(2, 2),
            )
        self.assertEqual(out["model.A"].dtype, paddle.float32)
        np.testing.assert_array_equal(
            out["model.A"].numpy(), expected["model.A"]
        )

    def test_allocation_failures_are_reported_not_masked(self):
        with self.subTest("no space at all -> RuntimeError naming /dev/shm"):
            proc = self._make_processor()
            with (
                patch(
                    f"{_ZCC}.shared_memory.SharedMemory",
                    side_effect=OSError("No space left on device"),
                ),
                self.assertRaisesRegex(RuntimeError, _SHM_DIR),
            ):
                proc._alloc_shared_memory("master", [8], paddle.float32)

        with self.subTest("not even bfloat16 fits -> propagate, leave nothing"):
            proc = self._make_processor()
            proc._alloc_shared_memory = lambda *a, **k: (_ for _ in ()).throw(
                RuntimeError("simulated /dev/shm exhaustion")
            )
            with self.assertRaises(RuntimeError):
                proc._ensure_ema_snapshot()
            self.assertIsNone(proc.ema_snapshot)
            self.assertEqual(proc._shm_segments, [])

        with self.subTest("a leftover segment of a recycled pid is reclaimed"):
            proc = self._make_processor()
            name = f"zcc_ema_p{os.getpid()}_r7_master"
            stale = shared_memory.SharedMemory(create=True, size=8, name=name)
            stale.close()
            try:
                tensor = proc._alloc_shared_memory(
                    "master", [8], paddle.float32
                )
            finally:
                try:
                    stale.unlink()
                except FileNotFoundError:
                    pass
            self.assertEqual(proc.ema_buffer_shm_handle["name"], name)
            self.assertEqual(list(tensor.shape), [8])

        with self.subTest("reclaim tolerates the file vanishing first"):
            proc = self._make_processor()
            real_shm = shared_memory.SharedMemory
            raised = []

            def once_exists(*args, **kwargs):
                if not raised:
                    raised.append(True)
                    raise FileExistsError
                return real_shm(*args, **kwargs)

            with (
                patch(f"{_ZCC}.shared_memory.SharedMemory", once_exists),
                patch(f"{_ZCC}.os.unlink", side_effect=FileNotFoundError()),
            ):
                tensor = proc._alloc_shared_memory(
                    "master", [8], paddle.float32
                )
            self.assertEqual(list(tensor.shape), [8])

        with (
            self.subTest("describing an unallocated snapshot fails loudly"),
            self.assertRaises(AssertionError),
        ):
            self._make_processor().shm_package()

    def test_release_unlinks_every_segment(self):
        self.proc._ensure_ema_snapshot()
        paths = self._shm_paths()
        self.assertTrue(paths)

        self.proc.unlink_shared_memory()
        for path in paths:
            self.assertFalse(os.path.exists(path), path)
        self.assertIsNone(self.proc.ema_snapshot)
        self.assertIsNone(self.proc.ema_buffer_shm_handle)
        self.assertEqual(self.proc.ema_snapshot_model_params, {})
        self.assertEqual(self.proc.ema_buffer_model_params_shm_handles, {})
        self.proc.unlink_shared_memory()  # idempotent

        with self.subTest("an unremovable segment only warns"):
            proc = self._make_processor()
            proc._ensure_ema_snapshot()
            proc._shm_segments[0] = SimpleNamespace(
                name="broken",
                unlink=lambda: (_ for _ in ()).throw(PermissionError("nope")),
            )
            proc.unlink_shared_memory()
            self.assertEqual(proc._shm_segments, [])


class TestEmaAccumulate(_EmaProcessorFixture):
    """``ema_accumulate`` folds the model weights into the live buffers in place."""

    def test_update_is_in_place_and_honours_the_loss_threshold(self):
        self.proc.ema_coef = 0.9
        self.proc.optimizer_fusion_storage_helper.cpu_buffer = _fp32(
            np.full(12, 5.0)
        )
        # the model-weight source lives in `inited_buffers[...][1]`
        _, cpu_buf = self.proc.param_fusion_storage_helper.inited_buffers["b0"]
        paddle.assign(_fp32(np.full(10, 7.0)), cpu_buf)

        master_buffer = self.proc.ema_buffer
        self.proc.ema_accumulate(10, loss=0.5, zcc_ema_loss_threshold=None)
        self.assertIs(self.proc.ema_buffer, master_buffer)
        # the EMA starts at zero, so one step leaves (1 - coef) * value
        np.testing.assert_allclose(
            self.proc.ema_buffer.numpy(), np.full(8, 0.5), rtol=1e-6
        )
        np.testing.assert_allclose(
            self.proc.ema_buffer_model_params["b0"].numpy(),
            np.full(10, 0.7),
            rtol=1e-6,
        )

        with self.subTest("a loss spike skips the update entirely"):
            proc = self._make_processor()
            proc.ema_coef = 0.9
            proc.optimizer_fusion_storage_helper = SimpleNamespace(
                master_weights_meta=self.MASTER_METAS,
                cpu_buffer=_fp32(np.full(12, 5.0)),
            )
            proc.ema_accumulate(10, loss=9.0, zcc_ema_loss_threshold=1.0)
            np.testing.assert_array_equal(proc.ema_buffer.numpy(), np.zeros(8))


class TestEmaMetadataGuards(unittest.TestCase):
    """``unpack_ema_data`` reinterprets raw memory through ``_slice`` and
    ``_set_dims``, neither of which validates, so inconsistent metadata has to be
    rejected rather than producing silently corrupt weights."""

    def setUp(self):
        self._segments = []

    def tearDown(self):
        for shm in self._segments:
            shm.close()
            try:
                shm.unlink()
            except FileNotFoundError:
                pass

    def _buffer(self, flat):
        flat = np.ascontiguousarray(flat, dtype=np.float32)
        shm = shared_memory.SharedMemory(create=True, size=flat.nbytes)
        self._segments.append(shm)
        np.ndarray(flat.shape, dtype=flat.dtype, buffer=shm.buf)[:] = flat
        return {"name": shm.name, "shape": [flat.size], "dtype": "float32"}

    def _package(self):
        return {
            "master_shm": self._buffer(np.arange(8, dtype=np.float32)),
            "param_shm": {
                "b0": self._buffer(np.arange(10, dtype=np.float32)),
                "unshard_0": self._buffer(np.arange(4, dtype=np.float32)),
            },
            "metas": {
                "model.A": {
                    "buffer_index": "b0",
                    "whole_buffer": False,
                    "start": 0,
                    "end": 6,
                    "shape": [2, 3],
                    "name": "model.A",
                },
                "model.C": {
                    "buffer_index": "unshard_0",
                    "whole_buffer": True,
                    "start": None,
                    "end": None,
                    "shape": [2, 2],
                    "name": "model.C",
                },
                "master_weights": {
                    "mw1": {
                        "start": 0,
                        "end": 4,
                        "shape": [2, 2],
                        "name": "mw1",
                    },
                },
            },
        }

    def test_consistent_metadata_is_accepted(self):
        unpack_ema_data(self._package())  # the baseline must not raise

    def test_inconsistent_metadata_is_rejected(self):
        corruptions = {
            "param slice shorter than its shape": (
                ["model.A", "shape"],
                [2, 4],
            ),
            "param slice past the end of the buffer": (
                ["model.A", "end"],
                999,
            ),
            "whole buffer smaller than its shape": (
                ["model.C", "shape"],
                [3, 3],
            ),
            "master slice longer than its shape": (
                ["master_weights", "mw1", "end"],
                6,
            ),
        }
        for label, (path, value) in corruptions.items():
            with self.subTest(label):
                package = self._package()
                target = package["metas"]
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                with self.assertRaises(AssertionError):
                    unpack_ema_data(package)


class TestEmaReadinessCoordination(unittest.TestCase):
    """Deciding whether step S's EMA is still readable on *every* rank, and the
    manager-side readiness and hand-off plumbing it is built on."""

    def test_classification_covers_wait_ready_and_gone(self):
        # (newest across ranks, oldest across ranks, requested step) -> state
        cases = {
            (5, 5, 5): "ready",
            (5, 3, 5): "wait",  # slowest rank is behind
            (4, 2, 5): "wait",  # every rank is behind
            (-1, -1, 0): "wait",  # nothing accumulated yet
            (6, 5, 5): "gone",  # someone already overwrote it
            (7, 5, 5): "gone",  # "gone" wins over a matching oldest
        }
        for (newest, oldest, step), expected in cases.items():
            with self.subTest(newest=newest, oldest=oldest, step=step):
                self.assertEqual(
                    EMAStateAssembler._classify_ema_state(newest, oldest, step),
                    expected,
                )

    def test_collective_result_is_fed_in_with_the_right_signs(self):
        """``[local, -local]`` reduced with MAX yields (newest, -oldest). The
        collective is stubbed, so this rank's own pair is the reduced result."""
        for local, step, expected in (
            (5, 5, "ready"),
            (4, 5, "wait"),
            (6, 5, "gone"),
            (-1, 0, "wait"),
        ):
            with self.subTest(local=local, step=step):
                asm = EMAStateAssembler.__new__(EMAStateAssembler)
                asm.zcc_manager = SimpleNamespace(
                    ema_shared_memory_ready_step=lambda: local
                )
                with patch("paddlefleet.trainer.trainer_utils.dist.all_reduce"):
                    self.assertEqual(asm._get_ema_state(step), expected)

    @staticmethod
    def _manager(workers):
        mgr = ZeroCostCheckpointManager.__new__(ZeroCostCheckpointManager)
        mgr.workers = workers
        mgr._ema_shm_package = None
        mgr._ema_shm_generation_seen = 0
        return mgr

    @staticmethod
    def _worker(ready=None, generation=0, with_queue=True):
        return SimpleNamespace(
            ema_ready_step=None if ready is None else _slot(ready),
            ema_meta_queue=queue.Queue() if with_queue else None,
            ema_shm_generation=_slot(generation),
        )

    def test_readiness_is_the_slowest_worker(self):
        self.assertEqual(
            self._manager(
                [self._worker(ready=9), self._worker(ready=4)]
            ).ema_shared_memory_ready_step(),
            4,
        )
        with self.subTest("no EMA workers -> not-ready sentinel"):
            self.assertEqual(
                self._manager([]).ema_shared_memory_ready_step(), -1
            )
            self.assertEqual(
                self._manager(
                    [self._worker(ready=None)]
                ).ema_shared_memory_ready_step(),
                -1,
            )

    def test_package_is_consumed_once_per_generation(self):
        worker = self._worker(ready=5, generation=1)
        worker.ema_meta_queue.put({"metas": {"a": 1}})
        mgr = self._manager([worker])

        first = mgr.get_ema_package()
        self.assertEqual(first, {"metas": {"a": 1}})
        self.assertEqual(mgr._ema_shm_generation_seen, 1)
        # same generation -> served from cache, the queue is left alone
        self.assertIs(mgr.get_ema_package(), first)

        with self.subTest("a newer generation supersedes the cache"):
            worker.ema_meta_queue.put({"metas": {"b": 2}})
            worker.ema_shm_generation.value = 2
            self.assertEqual(mgr.get_ema_package(), {"metas": {"b": 2}})

        with self.subTest("workers without a queue are skipped"):
            self.assertIsNone(
                self._manager(
                    [self._worker(ready=5, with_queue=False)]
                ).get_ema_package()
            )

        with self.subTest("a promised package that never arrives times out"):
            stuck = self._manager([self._worker(ready=5, generation=1)])
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                stuck.get_ema_package(timeout=0.01)

    def test_shutdown_reads_the_package_a_worker_cannot_exit_without(self):
        """The last HF step can publish after the last read, and a worker
        cannot exit while its package is still waiting in the pipe."""
        worker = self._worker(ready=6, generation=1)
        worker.task_queue = queue.Queue()
        worker.ema_meta_queue.put({"metas": {"a": 1}})
        mgr = self._manager([worker])

        def join(timeout=None):
            if timeout is None and not worker.ema_meta_queue.empty():
                raise AssertionError("package never read: join() hangs")

        mgr.processes = [
            SimpleNamespace(
                is_alive=lambda: not worker.ema_meta_queue.empty(), join=join
            )
        ]
        mgr.terminate_workers()
        self.assertEqual(
            worker.task_queue.get_nowait(), (ZCCTaskType.FINISH, None)
        )
        self.assertEqual(mgr.get_ema_package(), {"metas": {"a": 1}})

    def test_released_segments_are_resolved_under_shm(self):
        """Handles are bare segment names, not paths."""
        segment = shared_memory.SharedMemory(create=True, size=8)
        self.addCleanup(segment.close)
        try:
            self.assertEqual(
                ZeroCostCheckpointManager._check_shm_files_released(
                    [segment.name, f"/{segment.name}", "zcc_ema_gone"]
                ),
                [segment.name, f"/{segment.name}"],
            )
        finally:
            segment.unlink()
        self.assertEqual(
            ZeroCostCheckpointManager._check_shm_files_released([segment.name]),
            [],
        )

    def _touch_shm(self, name):
        path = os.path.join(_SHM_DIR, name)
        with open(path, "wb") as f:
            f.write(b"\0" * 16)
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        return path

    def test_startup_reclaims_only_orphaned_segments(self):
        stale = self._touch_shm(f"zcc_ema_p{_dead_pid()}_r0_master")
        mine = self._touch_shm(f"zcc_ema_p{os.getpid()}_r3_master")
        foreign = self._touch_shm("not_a_zcc_ema_segment")

        cleanup_ema_shared_memory()
        self.assertFalse(os.path.exists(stale))
        self.assertTrue(os.path.exists(mine))
        self.assertTrue(os.path.exists(foreign))

        # the sweep races the other ranks, so neither of these may abort it;
        # a fresh orphan is needed each time or the loop finds nothing to unlink
        for label, exc in (
            ("another rank removed it first", FileNotFoundError()),
            ("the segment cannot be removed", PermissionError("read-only")),
        ):
            with self.subTest(label):
                orphan = self._touch_shm(f"zcc_ema_p{_dead_pid()}_r0_master")
                with patch(f"{_ZCC}.os.unlink", side_effect=exc):
                    cleanup_ema_shared_memory()
                self.assertTrue(os.path.exists(orphan))
                os.unlink(orphan)

        with (
            self.subTest("an unreadable /dev/shm is tolerated"),
            patch(f"{_ZCC}.os.listdir", side_effect=OSError("boom")),
        ):
            cleanup_ema_shared_memory()


class _FakeProcess:
    def __init__(self, target=None, args=()):
        self.target, self.args = target, args
        self.started = False

    def start(self):
        self.started = True

    def join(self):
        pass


class _FakeContext:
    """Just enough of a multiprocessing context to build a manager in-process."""

    def Queue(self):
        return queue.Queue()

    def Value(self, typecode, value):
        return _slot(value)

    def Event(self):
        return threading.Event()

    def Process(self, target=None, args=()):
        return _FakeProcess(target, args)


class TestWorkerPublishesSnapshot(_EmaProcessorFixture):
    """Which steps publish the snapshot, and how the EMA slots get wired from the
    manager into the worker."""

    def _worker(self, **overrides):
        worker = ZeroCostCheckpointWorker(
            0,  # worker_id
            0,  # device_id
            7,  # global_rank
            1,  # offload_chunks
            queue.Queue(),  # task_queue
            _slot(0),  # status
            _slot(0),  # global_step
            _slot(0),  # version
            False,  # use_expert_parallel
            0,  # dp_rank
            0,  # mp_rank
            0,  # pp_rank
            ema_coef=0.9,
            save_hf_steps=10,
            ema_ready_step=_slot(-1),
            ema_meta_queue=queue.Queue(),
            ema_shm_generation=_slot(0),
        )
        worker.optimizer_fusion_storage_helper = (
            self.proc.optimizer_fusion_storage_helper
        )
        worker.param_fusion_storage_helper = (
            self.proc.param_fusion_storage_helper
        )
        worker.model_meta_content = None
        self.snapshots = []
        worker.zcc_ema_processor = SimpleNamespace(
            take_ema_snapshot=lambda: self.snapshots.append("snapshot"),
            shm_package=lambda: {"metas": {"a": 1, "b": 2}},
        )
        for key, value in overrides.items():
            setattr(worker, key, value)
        return worker

    def test_constructor_leaves_the_handoff_disarmed(self):
        worker = self._worker()
        self.assertEqual(worker.ema_ready_step.value, -1)
        self.assertEqual(worker.ema_shm_generation.value, 0)
        self.assertIsNotNone(worker.ema_meta_queue)
        self.assertFalse(worker.pending_ema_package)

    def test_only_hf_steps_snapshot_and_publish(self):
        skips = {
            "EMA shared memory disabled": {"ema_ready_step": None},
            "HF export disabled": {"save_hf_steps": -1},
        }
        for label, overrides in skips.items():
            with self.subTest(label):
                worker = self._worker(**overrides)
                worker._mark_ema_ready(10)
                self.assertEqual(self.snapshots, [])
                if worker.ema_ready_step is not None:
                    self.assertEqual(worker.ema_ready_step.value, -1)

        with self.subTest("a step that is not an HF step"):
            worker = self._worker()
            worker._mark_ema_ready(13)
            self.assertEqual(self.snapshots, [])
            self.assertEqual(worker.ema_ready_step.value, -1)

        worker = self._worker(pending_ema_package=True)
        worker._mark_ema_ready(20)
        self.assertEqual(self.snapshots, ["snapshot"])
        self.assertEqual(
            worker.ema_meta_queue.get_nowait(), {"metas": {"a": 1, "b": 2}}
        )
        self.assertEqual(worker.ema_shm_generation.value, 1)
        self.assertFalse(worker.pending_ema_package)
        self.assertEqual(worker.ema_ready_step.value, 20)

        with self.subTest("later HF steps re-snapshot but do not re-publish"):
            worker._mark_ema_ready(30)
            self.assertEqual(self.snapshots, ["snapshot", "snapshot"])
            self.assertTrue(worker.ema_meta_queue.empty())
            self.assertEqual(worker.ema_shm_generation.value, 1)
            self.assertEqual(worker.ema_ready_step.value, 30)

        with self.subTest("expert parallel filters keys on non-zero dp rank"):
            worker = self._worker(
                pending_ema_package=True, use_expert_parallel=True, dp_rank=1
            )
            worker._filter_moe_no_sync_optimizer_params = lambda meta, metas: {
                "kept": metas["a"]
            }
            worker._mark_ema_ready(20)
            self.assertEqual(
                worker.ema_meta_queue.get_nowait(), {"metas": {"kept": 1}}
            )

    def test_cache_rebuild_replaces_the_processor_and_arms_the_handoff(self):
        worker = self._worker(pending_ema_rebuild=True, zcc_ema_processor=None)
        worker._maybe_prepare_ema()
        self.addCleanup(worker.zcc_ema_processor.unlink_shared_memory)
        self.assertIsInstance(
            worker.zcc_ema_processor, ZeroCostCheckpointEMAProcessor
        )
        self.assertFalse(worker.pending_ema_rebuild)
        self.assertTrue(worker.pending_ema_package)

        with self.subTest("the previous snapshot is released first"):
            self.proc._ensure_ema_snapshot()
            paths = self._shm_paths()
            replaced = self._worker(
                pending_ema_rebuild=True, zcc_ema_processor=self.proc
            )
            replaced._maybe_prepare_ema()
            self.addCleanup(replaced.zcc_ema_processor.unlink_shared_memory)
            for path in paths:
                self.assertFalse(os.path.exists(path), path)
            self.assertIsNot(replaced.zcc_ema_processor, self.proc)

    def test_manager_creates_the_slots_only_when_the_export_is_on(self):
        def build(ema_coef, save_hf_steps):
            hcg = SimpleNamespace(
                get_data_parallel_rank=lambda: 0,
                get_model_parallel_rank=lambda: 0,
                _get_pipe_parallel_id=lambda: 0,
            )
            env = {"FLAGS_selected_gpus": "0", "FLAGS_selected_xpus": "0"}
            with (
                patch.dict(os.environ, env),
                patch(
                    f"{_ZCC}.multiprocessing.get_context",
                    return_value=_FakeContext(),
                ),
                patch(
                    f"{_ZCC}.fleet",
                    SimpleNamespace(
                        _hcg=hcg, get_hybrid_communicate_group=lambda: hcg
                    ),
                ),
                patch(f"{_ZCC}.dist.get_rank", return_value=3),
                patch(f"{_ZCC}.atexit.register"),
            ):
                return ZeroCostCheckpointManager(
                    worker_num=1,
                    pipeline_hooks_capacity=4,
                    capacity_usage=1.0,
                    use_expert_parallel=False,
                    ema_coef=ema_coef,
                    save_hf_steps=save_hf_steps,
                )

        mgr = build(ema_coef=0.99, save_hf_steps=10)
        (worker,) = mgr.workers
        self.assertEqual(worker.ema_ready_step.value, -1)
        self.assertEqual(worker.ema_shm_generation.value, 0)
        self.assertIsNotNone(worker.ema_meta_queue)
        self.assertEqual(worker.global_rank, 3)
        self.assertIsNone(mgr._ema_shm_package)
        self.assertEqual(mgr._ema_shm_generation_seen, 0)
        self.assertEqual(mgr.ema_shared_memory_ready_step(), -1)

        for label, kwargs in (
            ("no EMA coefficient", {"ema_coef": None, "save_hf_steps": 10}),
            ("no HF steps", {"ema_coef": 0.99, "save_hf_steps": -1}),
        ):
            with self.subTest(label):
                off = build(**kwargs)
                (worker,) = off.workers
                self.assertIsNone(worker.ema_ready_step)
                self.assertIsNone(worker.ema_meta_queue)
                self.assertIsNone(worker.ema_shm_generation)
                self.assertEqual(off.ema_shared_memory_ready_step(), -1)
                self.assertIsNone(off.get_ema_package())


class TestAssemblerHfExport(_EmaProcessorFixture):
    """The main-process state machine: one pending HF step at a time, cleared only
    once the export is on disk."""

    def _assembler(self):
        states, saved = [], []
        asm = EMAStateAssembler.__new__(EMAStateAssembler)
        asm.rank = 0
        asm.save_hf_steps = 10
        asm.pending_hf_step = None
        asm.zcc_manager = object()
        asm._get_ema_state = lambda step: states.pop(0)
        asm._load_ema_state_dict_from_shared_memory = lambda: {"sd": True}
        asm._build_ema_sharded_state_dict = lambda sd: ("sharded", sd)
        asm._save_full_ema_states = lambda step, sd: saved.append(
            (step, sd, asm.pending_hf_step)
        )
        return asm, states, saved

    def test_wait_retries_the_step_and_ready_exports_it(self):
        asm, states, saved = self._assembler()
        states[:] = ["wait", "ready"]

        asm.run(20)
        self.assertEqual(asm.pending_hf_step, 20)
        self.assertEqual(saved, [])

        asm.run(21)  # not an HF step, but step 20 is still owed
        self.assertEqual(len(saved), 1)
        step, state_dict, pending_during_save = saved[0]
        self.assertEqual(step, 20)
        self.assertEqual(state_dict, ("sharded", {"sd": True}))
        # the pending step must outlive the export: clearing it earlier would let
        # the worker overwrite the snapshot while it is still being read
        self.assertEqual(pending_during_save, 20)
        self.assertIsNone(asm.pending_hf_step)

    def test_pending_step_lifecycle(self):
        scenarios = {
            "HF export disabled": (
                {"save_hf_steps": 0},
                [],
                [10],
                [],
            ),
            "plain step with nothing owed": ({}, [], [13], []),
            "ready right on the HF step": ({}, ["ready"], [20], [20]),
            "gone is dropped, never exported": ({}, ["gone"], [20], []),
            "a newer HF step supersedes an unfinished one": (
                {},
                ["wait", "ready"],
                [20, 30],
                [30],
            ),
            "the final run retries what is still owed": (
                {},
                ["wait", "ready"],
                [20, None],
                [20],
            ),
            "the final run with nothing owed is a no-op": (
                {},
                [],
                [None],
                [],
            ),
        }
        for label, (attrs, flow, steps, expected) in scenarios.items():
            with self.subTest(label):
                asm, states, saved = self._assembler()
                for key, value in attrs.items():
                    setattr(asm, key, value)
                states[:] = flow
                for step in steps:
                    asm.run() if step is None else asm.run(step)
                self.assertEqual([s for s, _, _ in saved], expected)
                self.assertIsNone(asm.pending_hf_step)

    def test_a_missing_manager_fails_loudly(self):
        asm, _, _ = self._assembler()
        asm.zcc_manager = None
        with self.assertRaises(AssertionError):
            asm.run(20)

    def test_hf_interval_must_divide_the_checkpoint_interval(self):
        """Otherwise no checkpoint step would ever line up with an HF step."""
        with self.assertRaisesRegex(ValueError, "multiple of save_steps"):
            EMAStateAssembler(
                output_dir="/tmp",
                save_hf_steps=10,
                save_steps=3,
                model=None,
                optimizer=None,
            )

    def test_loading_places_params_on_device_and_keeps_master_on_cpu(self):
        with self.subTest("a readiness flag without handles fails loudly"):
            asm = EMAStateAssembler.__new__(EMAStateAssembler)
            asm.rank = 0
            asm.zcc_manager = SimpleNamespace(get_ema_package=lambda: None)
            with self.assertRaises(AssertionError):
                asm._load_ema_state_dict_from_shared_memory()

        if paddle.device.cuda.device_count() == 0:
            self.skipTest("device placement needs a GPU")

        expected = self._fill_live_ema()
        self.proc.take_ema_snapshot()
        package = self.proc.shm_package()
        asm = EMAStateAssembler.__new__(EMAStateAssembler)
        asm.rank = 0
        asm.zcc_manager = SimpleNamespace(get_ema_package=lambda: package)

        state_dict = asm._load_ema_state_dict_from_shared_memory()
        for key in ("model.A", "model.B", "model.C"):
            self.assertTrue(state_dict[key].place.is_gpu_place(), key)
            np.testing.assert_array_equal(
                state_dict[key].numpy(), expected[key]
            )
        for key in ("mw1", "mw2"):
            master = state_dict["master_weights"][key]
            # left as a zero-copy CPU view of the snapshot
            self.assertTrue(master.place.is_cpu_place(), key)
            np.testing.assert_array_equal(master.numpy(), expected[key])


class TestZccManagerInjection(unittest.TestCase):
    """``create_zcc_manager`` must hand the manager to an assembler that was built
    before it; without that the assembler cannot reach the workers' shm at all."""

    def _create(self, assembler):
        from paddlefleet.trainer.trainer import Trainer

        manager = object()
        order = []
        trainer = SimpleNamespace(
            model=SimpleNamespace(),
            args=SimpleNamespace(pipeline_model_parallel_size=1),
            zcc_manager=None,
            ema_state_assembler=assembler,
            _get_zcc_implementation_classes=lambda: (None, None),
            _create_zcc_manager_instance=lambda model, cls: manager,
            _setup_zcc_callback=lambda cls: order.append("callback"),
            _handle_checkpoint_resume=lambda ckpt: order.append("resume"),
        )

        class _NotPipeline:
            pass

        module = "paddlefleet.trainer.trainer"
        with (
            patch(f"{module}.PretrainedModel", SimpleNamespace),
            patch(f"{module}.PipelineLayer", _NotPipeline),
        ):
            Trainer.create_zcc_manager(trainer, None)
        self.assertEqual(order, ["callback", "resume"])
        return trainer, manager

    def test_manager_reaches_the_assembler_when_there_is_one(self):
        assembler = SimpleNamespace(zcc_manager=None)
        trainer, manager = self._create(assembler)
        self.assertIs(trainer.zcc_manager, manager)
        self.assertIs(assembler.zcc_manager, manager)

        with self.subTest("no assembler -> nothing to inject"):
            trainer, manager = self._create(None)
            self.assertIs(trainer.zcc_manager, manager)


if __name__ == "__main__":
    unittest.main()
