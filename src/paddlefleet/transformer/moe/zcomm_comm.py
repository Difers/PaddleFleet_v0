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

"""Low-SM communication for the RingMoE dispatcher, on paddlefleet_ops' zcomm ops.

* :class:`IntraCE` -- intra-node AllGather (copy engine, 0 SM) and ReduceScatter
  (copy-engine transpose + a local-sum kernel; NCCL-ring accumulation order, so
  bitwise equal to ``ncclReduceScatter``) over one NVLink domain.
* :class:`InterGin` -- inter-node ``all_to_all_single`` / AllGather as NCCL-GIN
  post-and-leave puts.

Both own a dedicated NCCL communicator (bootstrapped from the Paddle group) and
symmetric-window buffers. Window registration is collective, so the windows are
resident and reused, and which one a call uses must agree across the group; it
is therefore picked by call count, never by when a local tensor is freed.

What a caller keeps must not live in a window, or its lifetime would be tied to
that call count instead of to autograd (an activation saved for backward would
be overwritten by later layers, recompute or pipeline schedules deciding how
many). So:

* the intra AllGather *pulls*: each rank stages its shard in its own window and
  every rank copies the peers' stages into an ordinary ``paddle.empty`` output;
* the intra ReduceScatter's window is a stage consumed by the local sum, whose
  output is an ordinary tensor too;
* the inter ops do return window views, from a ring of :data:`_GIN_RING` slots
  per power-of-two bucket; their callers consume them within one ring round
  (and copy what outlives it), and a reissued slot whose view is still
  referenced raises.

Calls are issued on the Paddle group's comm stream after it waits for the calc
stream, and the returned task's ``wait()`` makes the calc stream wait for the
comm stream -- the same contract as the ``sync_op=False, use_calc_stream=False``
collectives they replace. Every rank of the group must make the same calls in
the same order.
"""

from __future__ import annotations

import collections
import itertools
import math
import weakref

import paddle
from paddle.base import core


def _C(name, *args):
    """Run a paddlefleet_ops custom op; single-output ops return the tensor itself."""
    out = paddle._C_ops._run_custom_op(name, *args)
    return out[0] if isinstance(out, (list, tuple)) and len(out) == 1 else out


__all__ = [
    "IntraCE",
    "InterGin",
    "memory_stats",
    "stream_add",
    "zcomm_unavailable_reason",
]

# Oldest NCCL the zcomm ops are validated with (their device API, e.g.
# ncclGin_VASignalInc, first appears in 2.29.7); the ops are only built against
# such an NCCL, and the one loaded at runtime must be at least as new.
MIN_NCCL_VERSION = 23007
_REQUIRED_OPS = (
    "zcomm_unique_id",
    "zcomm_intra_init",
    "zcomm_intra_alloc",
    "zcomm_intra_all_gather_pull",
    "zcomm_intra_reduce_scatter",
    "zcomm_buffer_view",
    "zcomm_gin_init",
    "zcomm_gin_alloc",
    "zcomm_gin_alltoall",
    "zcomm_stream_add",
    "zcomm_release_streams",
)


def zcomm_unavailable_reason():
    """Why the zcomm ops cannot run in this process, or None if they can.

    The ops exist only when paddlefleet_ops was built against an NCCL with the
    device API (see ``MIN_NCCL_VERSION``), and their struct layouts must match
    the NCCL loaded at runtime, so both are checked.
    """
    try:
        from paddlefleet_ops._extensions import ops
    except ImportError as e:
        return f"paddlefleet_ops custom ops are not importable ({e})"
    missing = [name for name in _REQUIRED_OPS if not hasattr(ops, name)]
    if missing:
        return (
            f"paddlefleet_ops was built without the zcomm ops ({missing[0]} is "
            f"missing); rebuild it with NCCL_HOME pointing at NCCL >= "
            f"{_fmt_version(MIN_NCCL_VERSION)}"
        )
    if not core.is_compiled_with_nccl():
        return "Paddle is not compiled with NCCL"
    runtime = core.nccl_version()
    if runtime < MIN_NCCL_VERSION:
        return (
            f"the loaded NCCL is {_fmt_version(runtime)}, the zcomm ops need "
            f">= {_fmt_version(MIN_NCCL_VERSION)}"
        )
    return None


def _fmt_version(code):
    return f"{code // 10000}.{code // 100 % 100}.{code % 100}"


# Slots per bucket of the inter ops' window ring. A slot is reissued after this
# many further calls of its bucket; their callers consume each output within one
# ring round, so two suffice (it also bounds the windows registered per bucket).
_GIN_RING = 2
# NCCL-GIN launch shape: contexts (QP sets), put-kernel blocks, slices per peer.
_GIN_CTX, _GIN_BLOCKS, _GIN_SLICES = 16, 32, 8
# Slots per bucket of IntraCE's push-AllGather output ring (see
# :meth:`IntraCE.all_gather_push_async`, kept for comparison; the dispatcher pulls).
_PUSH_RING = 8
# Grid of the ReduceScatter's local-sum kernel: fewer blocks leave more SMs to the
# GEMM it overlaps but make the sum slower; 128 was the fastest end to end.
RS_SUM_BLOCKS = 128


_TOKENS = itertools.count(1)


_ANCHOR = []
# comm stream -> tokens of its calls not yet released, oldest first. Work on a
# stream runs in issue order and tokens grow with it, so a wait for one call
# covers every earlier call on that stream: their registrations go too. That
# also releases calls nobody waits on directly (ordered before a later wait on
# the comm side, or whose task was dropped).
_PENDING: dict = {}


def _pending(stream_int, token):
    _PENDING.setdefault(stream_int, collections.deque()).append(token)


def _release_upto(stream_int, token):
    q = _PENDING.get(stream_int)
    while q and q[0] <= token:
        _release(q.popleft())


def _release(token):
    """Withdraw the comm-stream registrations of the call that got ``token``.

    The ops register the comm stream on the allocator-owned tensors they read
    or write (see ``zcomm/record_stream.h``), so dropping such a tensor early
    is safe. Called right after the current stream waited for that call, so
    the allocator may again reuse the blocks in calc-stream order.
    """
    # A custom op needs a tensor input; a private empty one, so no user tensor
    # (possibly an autograd leaf) is touched.
    if not _ANCHOR:
        _ANCHOR.append(paddle.empty([0], dtype="uint8"))
    _C(
        "zcomm_release_streams",
        _ANCHOR[0],
        token,
        int(paddle.device.current_stream().stream_base.raw_stream),
    )


class _Task:
    __slots__ = ("_ev", "_keep", "_token", "_stream")

    def __init__(self, ev, keep, token=0, stream=0):
        self._ev, self._keep = ev, keep
        self._token, self._stream = token, stream

    def wait(self):
        if self._ev is not None:
            paddle.device.current_stream().stream_base.wait_event(self._ev)
            if self._token:
                _release_upto(self._stream, self._token)
                self._token = 0
        self.covered()
        return True

    def covered(self):
        """The current stream already waited for this call some other way.

        For a call the caller does not wait on directly but that a later wait
        covers (e.g. a transfer ordered before one that is waited for).
        """
        if self._token:
            _release(self._token)
            self._token = 0
        self._ev = self._keep = None

    def is_completed(self):
        return self._ev is None or self._ev.query()


def _uid(group):
    """Broadcast an NCCL unique id from the group's first rank."""
    if group.rank == 0:
        uid = _C("zcomm_unique_id", paddle.zeros([1], dtype="uint8"))
        uid = uid.cuda()
    else:
        uid = paddle.zeros([128], dtype="uint8")
    paddle.distributed.broadcast(uid, src=group.ranks[0], group=group)
    return uid.cpu()


_DTYPE_NAME = {
    paddle.bfloat16: "bfloat16",
    paddle.float16: "float16",
    paddle.float32: "float32",
    paddle.int32: "int32",
    paddle.int64: "int64",
    paddle.uint8: "uint8",
    paddle.float8_e4m3fn: "float8_e4m3fn",
}


def _bucket(nbytes):
    """Power-of-two bucket (log2 bytes, at least one 4 KiB page) of ``nbytes``."""
    return max(12, (int(nbytes) - 1).bit_length())


class _Pool:
    """Resident symmetric buffers: per power-of-two bucket a ring of slots."""

    def __init__(self, alloc_fn, ring):
        self._alloc, self._ring = alloc_fn, ring
        self._slots: dict[int, list] = {}
        self._served: dict[int, int] = {}

    def take(self, nbytes):
        bucket = _bucket(nbytes)
        ring = self._slots.setdefault(bucket, [])
        n = self._served.get(bucket, 0)
        self._served[bucket] = n + 1
        slot = n % self._ring
        if slot == len(ring):
            ring.append([self._alloc(1 << bucket), None, {}])
        entry = ring[slot]
        if entry[1] is not None and entry[1]() is not None:
            raise RuntimeError(
                f"zcomm buffer of {1 << bucket} bytes still referenced after "
                f"{self._ring} further calls of its size; the caller must "
                f"consume (or copy) an inter-node output before then"
            )
        return entry

    @staticmethod
    def view(entry, off, nbytes, dtype, shape):
        """``entry``'s buffer [off, off+nbytes) as a ``dtype`` tensor (cached per shape).

        Built by the ``zcomm_buffer_view`` op over the real address rather than by
        slicing the uint8 buffer: a sliced view carries a storage offset, which
        several consumers downstream (Triton / CuTe kernels reached through the
        torch proxy) do not honour, so they would read from the buffer's start.
        """
        key = (off, nbytes, dtype, tuple(shape))
        t = entry[2].get(key)
        if t is None:
            t = _C(
                "zcomm_buffer_view",
                entry[0],
                off,
                list(shape),
                _DTYPE_NAME[dtype],
            )
            entry[2][key] = t
        out = t.detach()
        entry[1] = weakref.ref(out)
        return out


class _CommBase:
    _instances: dict = {}

    @classmethod
    def get(cls, group):
        key = (cls.__name__, group.name)
        inst = cls._instances.get(key)
        if inst is None:
            inst = cls._instances[key] = cls(group)
        return inst

    def __init__(self, group):
        self.group = group
        place = paddle.framework._current_expected_place()
        self._comm_stream = group.process_group.get_stream(place)
        self._stream_int = int(self._comm_stream.raw_stream)
        self._evs = [core.CUDAEvent() for _ in range(64)]
        self._ev_i = 0

    def _enter(self):
        """Order the comm stream after the calc stream; returns the call's token."""
        self._comm_stream.wait_stream(
            paddle.device.current_stream().stream_base
        )
        self._token = next(_TOKENS)
        _pending(self._stream_int, self._token)
        return self._token

    def _leave(self, keep):
        self._ev_i = (self._ev_i + 1) % len(self._evs)
        ev = self._evs[self._ev_i]
        ev.record(self._comm_stream)
        return _Task(ev, keep, self._token, self._stream_int)


class IntraCE(_CommBase):
    """Intra-node CE AllGather / ReduceScatter over ``group`` (one NVLink domain)."""

    def __init__(self, group):
        super().__init__(group)
        self._h = _C("zcomm_intra_init", _uid(group), group.rank, group.nranks)
        # Symmetric stages: two per (op, power-of-two bucket), used in turn.
        # Each pull AllGather / ReduceScatter does a single handshake, which
        # orders the peers' accesses to the stages of earlier calls but not to
        # this call's own; a stage is therefore reused only two calls of its
        # kind later, after at least one other call's handshake (calls on one
        # comm are serialized on its stream).
        self._stages: dict[tuple, list] = {}
        # Output ring of the push AllGather, created on first use.
        self._push_pool = None

    def _stage(self, kind, nbytes):
        bucket = _bucket(nbytes)
        entry = self._stages.get((kind, bucket))
        if entry is None:
            entry = self._stages[(kind, bucket)] = [
                [
                    _C("zcomm_intra_alloc", self._h, 1 << bucket)
                    for _ in range(2)
                ],
                0,
            ]
        entry[1] ^= 1
        return entry[0][entry[1]]

    def all_gather_async(self, x):
        """AllGather along axis 0 into an ordinary tensor. Returns ``(out, task)``.

        ``x`` is copied into this rank's stage, a handshake tells the peers it is
        there, and every rank pulls the peers' stages into ``out`` with the copy
        engine. Only the stage is a window, and only for this call, so ``out``
        may live as long as autograd needs it.
        """
        x = x.contiguous()
        n = self.group.nranks
        shard = math.prod(x.shape) * x.element_size()
        out = paddle.empty([x.shape[0] * n, *x.shape[1:]], dtype=x.dtype)
        stage = self._stage("ag", shard)
        self._enter()
        _C(
            "zcomm_intra_all_gather_pull",
            self._h,
            x,
            stage,
            out,
            self._stream_int,
            self._token,
        )
        return out, self._leave((x, out))

    def all_gather_push_async(self, x):
        """AllGather along axis 0, pushed into a window slot. Returns ``(out, task)``.

        Every rank's copy engine writes ``x`` into the same slot of every peer,
        and ``out`` is a view of that slot. About 10% faster than
        :meth:`all_gather_async` (no staging copy, CE writes beat CE reads), but
        the slot is reissued :data:`_PUSH_RING` calls of its size later whatever
        still refers to it, so ``out`` must be consumed before then. That rules
        it out for activations saved for backward across layers (see
        ``RingMoETokenDispatcher``, which uses the pull); kept for benchmarks.
        """
        if self._push_pool is None:
            self._push_pool = _Pool(
                lambda n: _C("zcomm_intra_alloc", self._h, n), _PUSH_RING
            )
        x = x.contiguous()
        n = self.group.nranks
        shard = math.prod(x.shape) * x.element_size()
        entry = self._push_pool.take(n * shard)
        self._enter()
        _C(
            "zcomm_intra_all_gather",
            self._h,
            x,
            entry[0],
            self._stream_int,
            self._token,
        )
        out = _Pool.view(
            entry, 0, n * shard, x.dtype, [x.shape[0] * n, *x.shape[1:]]
        )
        return out, self._leave((x, out))

    def reduce_scatter_async(self, x, out=None, sum_blocks=RS_SUM_BLOCKS):
        """SUM ReduceScatter along axis 0 (bf16). Returns ``(out, task)``.

        The copy engine writes block p of ``x`` into peer p's stage, then a local
        kernel sums the stage in NCCL's ring order with a bf16 rounding per add,
        so the result is bitwise equal to ``ncclReduceScatter``. ``sum_blocks``
        is that kernel's grid.
        """
        x = x.contiguous()
        n = self.group.nranks
        if out is None:
            out = paddle.empty([x.shape[0] // n, *x.shape[1:]], dtype=x.dtype)
        stage = self._stage("rs", math.prod(x.shape) * x.element_size())
        self._enter()
        _C(
            "zcomm_intra_reduce_scatter",
            self._h,
            x,
            stage,
            out,
            True,  # NCCL ring accumulation order
            sum_blocks,
            self._stream_int,
            self._token,
        )
        return out, self._leave((x, out))


class InterGin(_CommBase):
    """Inter-node ``all_to_all_single`` / AllGather over ``group`` as NCCL-GIN puts."""

    def __init__(self, group):
        super().__init__(group)
        self._h = _C(
            "zcomm_gin_init",
            _uid(group),
            group.rank,
            group.nranks,
            _GIN_CTX,
            _GIN_BLOCKS,
            _GIN_SLICES,
        )
        self._pool = _Pool(
            lambda n: _C("zcomm_gin_alloc", self._h, n), _GIN_RING
        )
        self._plans: dict = {}

    def _a2a_plan(self, in_split, out_split, row, half):
        key = ("a2a", tuple(in_split), tuple(out_split), row, half)
        plan = self._plans.get(key)
        if plan is None:
            n, me = self.group.nranks, self.group.rank
            src_off, acc = [0] * n, 0
            for p in range(n):
                src_off[p] = acc
                acc += in_split[p] * row
            sends = []
            for p in range(n):
                if p == me or in_split[p] == 0:
                    continue
                # Peer p's recv offset of my chunk. Requires the split pattern
                # to be rotation invariant across the group (ring shift and
                # the combine all-to-all are): rank q's in_split[x] equals my
                # in_split[(x - q + me) % n].
                off = sum(in_split[(p - q + me) % n] for q in range(me)) * row
                sends += [p, src_off[p], half + off, in_split[p] * row]
            incoming = [q for q in range(n) if q != me and out_split[q] > 0]
            plan = self._plans[key] = [
                len(sends) // 4,
                *sends,
                len(incoming),
                *incoming,
            ]
        return plan

    def all_to_all_single(self, x, out_split, in_split):
        """``all_to_all_single`` along axis 0 (no self chunk). Returns ``(out, task)``."""
        me = self.group.rank
        if out_split[me] or in_split[me]:
            raise ValueError("InterGin.all_to_all_single does not send to self")
        x = x.contiguous()
        row = math.prod(x.shape[1:]) * x.element_size()
        send_bytes, recv_bytes = sum(in_split) * row, sum(out_split) * row
        half = 1 << _bucket(max(send_bytes, recv_bytes))
        entry = self._pool.take(2 * half)
        plan = self._a2a_plan(in_split, out_split, row, half)
        self._enter()
        _C(
            "zcomm_gin_alltoall",
            self._h,
            x,
            entry[0],
            plan,
            0,
            self._stream_int,
            self._token,
        )
        out = _Pool.view(
            entry, half, recv_bytes, x.dtype, [sum(out_split), *x.shape[1:]]
        )
        return out, self._leave(x)

    def all_gather_async(self, x):
        """AllGather along axis 0. Returns ``(out, task)``."""
        x = x.contiguous()
        n, me = self.group.nranks, self.group.rank
        shard = math.prod(x.shape) * x.element_size()
        half = 1 << _bucket(n * shard)
        entry = self._pool.take(2 * half)
        mine = half + me * shard
        key = ("ag", shard, half)
        plan = self._plans.get(key)
        if plan is None:
            peers = [p for p in range(n) if p != me]
            sends = []
            for p in peers:
                sends += [p, mine, mine, shard]
            plan = self._plans[key] = [len(peers), *sends, len(peers), *peers]
        self._enter()
        # Stage x straight into its own output slot, then put that slot to peers.
        _C(
            "zcomm_gin_alltoall",
            self._h,
            x,
            entry[0],
            plan,
            mine,
            self._stream_int,
            self._token,
        )
        out = _Pool.view(
            entry, half, n * shard, x.dtype, [x.shape[0] * n, *x.shape[1:]]
        )
        return out, self._leave(x)

    def all_gather_by_distance_async(self, x, on_part):
        """:meth:`all_gather_async` as two calls: distance 1, then the rest.

        Call 1 puts this rank's block to rank ``me + 1`` and receives rank
        ``me - 1``'s, so the block a ring needs first lands after one block's
        transfer instead of sharing the NIC with every other block until the
        end; call 2 moves the remaining distances together (one call per
        distance would serialize them and delay whatever follows on this
        stream). Same output layout as :meth:`all_gather_async`. After each
        call ``on_part(ds, out, task)`` runs with the distances it delivered
        (block ``me - d`` for each ``d``), so the caller can order their
        consumers right behind it on their own streams. Returns
        ``(out, task)`` of the last call.
        """
        x = x.contiguous()
        n, me = self.group.nranks, self.group.rank
        shard = math.prod(x.shape) * x.element_size()
        half = 1 << _bucket(n * shard)
        entry = self._pool.take(2 * half)
        mine = half + me * shard
        out = _Pool.view(
            entry, half, n * shard, x.dtype, [x.shape[0] * n, *x.shape[1:]]
        )
        task = None
        for ds in ([1], list(range(2, n))):
            if not ds:
                continue
            key = ("agd", shard, half, ds[0])
            plan = self._plans.get(key)
            if plan is None:
                sends = []
                for d in ds:
                    sends += [(me + d) % n, mine, mine, shard]
                plan = self._plans[key] = [
                    len(ds),
                    *sends,
                    len(ds),
                    *[(me - d) % n for d in ds],
                ]
            self._enter()
            # The first call stages x into its own slot; the second puts it.
            _C(
                "zcomm_gin_alltoall",
                self._h,
                x if ds[0] == 1 else x[:0],
                entry[0],
                plan,
                mine,
                self._stream_int,
                self._token,
            )
            task = self._leave(x)
            on_part(ds, out, task)
        return out, task


def stream_add(a, b, group, tasks=None):
    """``a + b`` (bf16 / fp32) computed on ``group``'s comm stream, after its current
    work and after the calc stream's current tail (``out`` is allocated on the
    calc stream). Lets a sum of values that land on that comm stream feed a
    transfer on it without the calc stream waiting for either input. The comm
    stream is registered on ``a``, ``b`` and ``out`` and withdrawn by the next
    wait for a later call on that stream; with ``tasks`` a task is appended
    whose ``covered()`` withdraws it earlier, once a wait covers the add."""
    place = paddle.framework._current_expected_place()
    stream = group.process_group.get_stream(place)
    stream.wait_stream(paddle.device.current_stream().stream_base)
    out = paddle.empty(a.shape, dtype=a.dtype)
    # No .contiguous() here: a copy would run on the calc stream, before the
    # inputs are ready.
    if not (a.is_contiguous() and b.is_contiguous()):
        raise ValueError("stream_add needs contiguous inputs")
    sid = int(stream.raw_stream)
    token = next(_TOKENS)
    _pending(sid, token)
    _C("zcomm_stream_add", a, b, out, sid, token)
    if tasks is not None:
        tasks.append(_Task(None, (a, b, out), token, sid))
    return out


def memory_stats():
    """Resident symmetric-buffer bytes per (class, group), split into pooled outputs
    and RS stages: ``{(cls, group): {"pool": {bucket_bytes: n_slots}, "stage": bytes, "total": bytes}}``."""
    out = {}
    for (cls, gname), inst in _CommBase._instances.items():
        pool = {}
        tot = 0
        for p in (
            getattr(inst, "_pool", None),
            getattr(inst, "_push_pool", None),
        ):
            if p is None:
                continue
            for bucket, ring in p._slots.items():
                blk = 1 << bucket
                pool[blk] = pool.get(blk, 0) + len(ring)
                tot += blk * len(ring)
        stage = sum(
            len(bufs) << b
            for (_, b), (bufs, _) in getattr(inst, "_stages", {}).items()
        )
        out[(cls, gname)] = {"pool": pool, "stage": stage, "total": tot + stage}
    return out
