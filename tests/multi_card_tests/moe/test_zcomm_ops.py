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

"""zcomm ops (paddlefleet_ops) via paddlefleet.transformer.moe.zcomm_comm vs NCCL, bitwise.

The intra-node cases run on one node of 8 GPUs (intra = the 8 GPUs of a node);
the inter-node ones (inter = same-index GPUs across nodes) need >= 2 nodes and
are skipped otherwise:
    python -m paddle.distributed.launch --nnodes 2 ... test_zcomm_ops.py
Everything is skipped when the zcomm ops cannot run (see
zcomm_comm.zcomm_unavailable_reason: paddlefleet_ops built and run with
NCCL >= 2.30.7). NCCL_NVLS_ENABLE=1 is recommended, not required.
"""

import unittest

import paddle
import paddle.distributed as dist

from paddlefleet.transformer.moe.zcomm_comm import (
    InterGin,
    IntraCE,
    stream_add,
    zcomm_unavailable_reason,
)

_SUM = paddle.framework.core.ReduceOp.SUM
_G = 8


def _same(a, b):
    return bool((a.view("uint8") == b.view("uint8")).all().item())


_UNAVAILABLE = zcomm_unavailable_reason()


def _needs_inter(fn):
    """Skip an inter-node case unless the job spans >= 2 nodes of _G GPUs."""

    def wrapper(self):
        if self.inter is None:
            self.skipTest(
                f"needs >= 2 nodes x {_G} GPUs, world={dist.get_world_size()}"
            )
        fn(self)

    wrapper.__name__ = fn.__name__
    return wrapper


@unittest.skipIf(_UNAVAILABLE is not None, f"zcomm unavailable: {_UNAVAILABLE}")
class TestZcommOps(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dist.init_parallel_env()
        world, rank = dist.get_world_size(), dist.get_rank()
        if world < _G or world % _G:
            raise unittest.SkipTest(
                f"needs a multiple of {_G} GPUs (one NVLink node), got {world}"
            )
        cls.rank = rank
        intras = [
            dist.new_group(list(range(k * _G, (k + 1) * _G)))
            for k in range(world // _G)
        ]
        inters = [
            dist.new_group([c + _G * k for k in range(world // _G)])
            for c in range(_G)
        ]
        cls.intra, cls.inter = intras[rank // _G], inters[rank % _G]
        if cls.inter.nranks < 2:
            cls.inter = None
        for g in (cls.intra, cls.inter):
            if g is not None:
                dist.all_reduce(paddle.zeros([1]), group=g)

    def _rows(self):
        return (8192, 4096, 1024)

    def test_intra_all_gather(self):
        for it, rows in enumerate(self._rows() * 3):
            x = (paddle.randn([rows, 2048]) * (self.rank + 1 + it)).astype(
                "bfloat16"
            )
            out, task = IntraCE.get(self.intra).all_gather_async(x)
            task.wait()
            ref = paddle.empty([rows * _G, 2048], dtype="bfloat16")
            self.intra.process_group.all_gather_into_tensor(ref, x, True).wait()
            self.assertTrue(_same(out, ref))

    def test_intra_all_gather_push(self):
        for it, rows in enumerate(self._rows() * 3):
            x = (paddle.randn([rows, 2048]) * (self.rank + 1 + it)).astype(
                "bfloat16"
            )
            out, task = IntraCE.get(self.intra).all_gather_push_async(x)
            task.wait()
            ref = paddle.empty([rows * _G, 2048], dtype="bfloat16")
            self.intra.process_group.all_gather_into_tensor(ref, x, True).wait()
            self.assertTrue(_same(out, ref))
            del out

    def test_intra_all_gather_outputs_outlive_calls(self):
        # Outputs are ordinary tensors: holding many of them (as activations
        # saved for backward across layers are) must not corrupt any.
        comm = IntraCE.get(self.intra)
        kept = []
        for it in range(40):
            x = paddle.full([1024, 2048], float(it + 1), dtype="bfloat16")
            out, task = comm.all_gather_async(x)
            task.wait()
            kept.append((it, out.detach()))
        for it, out in kept:
            self.assertTrue(bool((out == float(it + 1)).all().item()), it)

    def test_intra_reduce_scatter(self):
        for rows in self._rows() * 3:
            x = (paddle.randn([rows * _G, 2048]) * (self.rank + 1)).astype(
                "bfloat16"
            )
            out, task = IntraCE.get(self.intra).reduce_scatter_async(x)
            task.wait()
            ref = paddle.empty([rows, 2048], dtype="bfloat16")
            self.intra.process_group.reduce_scatter_tensor(
                ref, x, _SUM, True
            ).wait()
            self.assertTrue(_same(out, ref))

    def _busy_comm_stream(self):
        # Queue ~10 ms of transfers on the intra comm stream, so work enqueued
        # behind them runs well after the calc stream has moved on.
        comm = IntraCE.get(self.intra)
        x = paddle.ones([32768, 4096], dtype="bfloat16")
        return [comm.all_gather_async(x) for _ in range(12)]

    def _reuse_freed_memory(self, nbytes):
        # Calc-stream allocations of the freed sizes, overwritten at once.
        for _ in range(4):
            t = paddle.full([nbytes // 2], 100.0, dtype="bfloat16")
            del t

    def test_stream_add_inputs_freed_early(self):
        # Inputs of a comm-stream op may be dropped before it runs: the op
        # records the comm stream on them, so the allocator does not hand
        # their memory to calc-stream tensors meanwhile.
        rows = 8192
        busy = self._busy_comm_stream()
        a = paddle.full([rows, 2048], 1.0, dtype="bfloat16")
        b = paddle.full([rows, 2048], 2.0, dtype="bfloat16")
        out = stream_add(a, b, self.intra)
        del a, b
        self._reuse_freed_memory(rows * 2048 * 2)
        paddle.device.synchronize()
        del busy
        self.assertTrue(bool((out == 3.0).all().item()))

    def test_reduce_scatter_input_freed_early(self):
        rows = 8192
        x = (paddle.randn([rows * _G, 2048]) * (self.rank + 1)).astype(
            "bfloat16"
        )
        ref = paddle.empty([rows, 2048], dtype="bfloat16")
        self.intra.process_group.reduce_scatter_tensor(
            ref, x, _SUM, True
        ).wait()
        busy = self._busy_comm_stream()
        out, task = IntraCE.get(self.intra).reduce_scatter_async(x.clone())
        del task, x
        self._reuse_freed_memory(rows * _G * 2048 * 2)
        paddle.device.synchronize()
        del busy
        self.assertTrue(_same(out, ref))

    def _two_rs_on_one_input(self, wait_order):
        # Two ReduceScatters of the same input on one comm stream, behind a
        # long queue; the calc stream waits for them in `wait_order`, then
        # frees the input and overwrites its memory. Waiting for the first
        # does not cover the second, so the input must stay registered until
        # the second is waited for.
        rows = 8192
        x = (paddle.randn([rows * _G, 2048]) * (self.rank + 1)).astype(
            "bfloat16"
        )
        ref = paddle.empty([rows, 2048], dtype="bfloat16")
        self.intra.process_group.reduce_scatter_tensor(
            ref, x, _SUM, True
        ).wait()
        comm = IntraCE.get(self.intra)
        busy = self._busy_comm_stream()
        x2 = x.clone()
        del x
        outs = [comm.reduce_scatter_async(x2) for _ in range(2)]
        outs[wait_order[0]][1].wait()
        del x2
        self._reuse_freed_memory(rows * _G * 2048 * 2)
        outs[wait_order[1]][1].wait()
        paddle.device.synchronize()
        del busy
        for out, _ in outs:
            self.assertTrue(_same(out, ref))

    def test_input_reused_wait_first_only(self):
        self._two_rs_on_one_input((0, 1))

    def test_input_reused_waits_reversed(self):
        self._two_rs_on_one_input((1, 0))

    def test_registrations_released_after_wait(self):
        # Waited calls withdraw their registrations: repeated calls whose
        # inputs are freed right after the wait keep the pool flat.
        comm = IntraCE.get(self.intra)
        rows = 4096

        def step():
            x = paddle.randn([rows * _G, 2048]).astype("bfloat16")
            out, task = comm.reduce_scatter_async(x)
            task.wait()
            del x, out, task

        for _ in range(5):
            step()
        paddle.device.synchronize()
        before = paddle.device.cuda.memory_reserved()
        for _ in range(50):
            step()
        paddle.device.synchronize()
        self.assertEqual(paddle.device.cuda.memory_reserved(), before)

    @_needs_inter
    def test_inter_shift(self):
        n, me = self.inter.nranks, self.inter.rank
        for rows in self._rows() * 3:
            x = (paddle.randn([rows, 2048]) * (self.rank + 1)).astype(
                "bfloat16"
            )
            ins = [0] * n
            ins[(me + 1) % n] = rows
            outs = [0] * n
            outs[(me - 1) % n] = rows
            out, task = InterGin.get(self.inter).all_to_all_single(x, outs, ins)
            task.wait()
            ref = paddle.empty([rows, 2048], dtype="bfloat16")
            self.inter.process_group.all_to_all_single(
                ref, x, outs, ins, True
            ).wait()
            self.assertTrue(_same(out, ref))

    @_needs_inter
    def test_inter_shift_fp32_narrow(self):
        # RingMoE routing pair: [T, 2K] float32, 80-byte rows
        n, me = self.inter.nranks, self.inter.rank
        for rows in (8192, 8192, 4096):
            x = paddle.randn([rows, 20]).astype("float32") * (self.rank + 1)
            ins = [0] * n
            ins[(me + 1) % n] = rows
            outs = [0] * n
            outs[(me - 1) % n] = rows
            out, task = InterGin.get(self.inter).all_to_all_single(x, outs, ins)
            task.wait()
            ref = paddle.empty([rows, 20], dtype="float32")
            self.inter.process_group.all_to_all_single(
                ref, x, outs, ins, True
            ).wait()
            self.assertTrue(_same(out, ref))

    @_needs_inter
    def test_inter_all_gather(self):
        n = self.inter.nranks
        for rows in self._rows() * 3:
            x = (paddle.randn([rows, 2048]) * (self.rank + 1)).astype(
                "bfloat16"
            )
            out, task = InterGin.get(self.inter).all_gather_async(x)
            task.wait()
            ref = paddle.empty([rows * n, 2048], dtype="bfloat16")
            self.inter.process_group.all_gather_into_tensor(ref, x, True).wait()
            self.assertTrue(_same(out, ref))


if __name__ == "__main__":
    unittest.main()
