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

"""Tests for ``trainer/utils/offload_optimizer.py``.

Module under test: paddlefleet.trainer.utils.offload_optimizer.

Independent oracle
------------------
Every expectation below is hand-derived by reading the offload wrapper's
control flow:

* ``offload(t)`` selects a place from the compile flags (CUDA ->
  CUDAPinnedPlace, XPU -> XPUPinnedPlace, else CPUPlace), calls
  ``to_device(t, place)``, asserts the migration is in-place (returns the same
  object) and returns ``None``.
* ``reload(t)`` calls ``to_device(t)`` with no place, so the target is the
  default device (``get_env_device()``); on a CPU build that is CPUPlace.
* Step 1 (``_add_accumulator``) calls the original, then offloads the freshly
  created accumulator -- state offload happens *at accumulator creation*.
* Step 2 (``adam_``/``adamw_``) reloads *every* tensor argument before the op
  and offloads only arguments at index >= 2 (state; never param/grad), gated by
  ``getattr(args[0], "is_offload_opt", <default>)``. The default is ``True``
  for ``hack_offload_optimizer`` but ``False`` for
  ``hack_offload_optimizer_eb5`` -- an observable behavioral difference.
* Step 3 (``_insert_sync``) reloads the sync var, runs the original, then moves
  it back to its original place when the gate is on (same per-variant default).
* Step 4 (Muon) reloads the momentum buffer and the param's master weight
  before the Python-side momentum update and offloads both afterwards, but only
  when ``param.is_offload_opt`` is truthy. It degrades to a no-op when Muon or
  its update entry points are absent.
* Step 5 (mixed-optimizer sub-optimizers) is covered by
  ``test_inner_optimizer_offload.py``; like Step 4 its gate defaults to
  ``True`` in both variants.

Both hack variants carry textually duplicated Step 1/3/4 code, so tests whose
expectation is identical for both are parameterized over both variants to
exercise both copies.

These are CPU/no-GPU observations of the wrapper's *dispatch/lifecycle* control
logic only. Actual device migration, pinned-memory residency and real memory
release require a GPU and are NOT claimed here; ``to_device`` is spied (or
``reload``/``offload`` replaced by recorders) purely to observe which tensors
the wrapper decides to reload/offload and in what order.
"""

import contextlib
import sys
import unittest
from unittest import mock

try:
    import numpy as np
    import paddle
    from paddle import _C_ops
    from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer.hybrid_parallel_optimizer import (
        HybridParallelOptimizer,
    )
    from paddle.optimizer import Optimizer

    from paddlefleet.trainer.utils import offload_optimizer as oo

    HAS_PADDLE = True
except ImportError:
    HAS_PADDLE = False

try:
    from paddle.optimizer.muon import Muon

    _HAS_MUON = True
except ImportError:  # pragma: no cover - Muon missing on very old Paddle
    Muon = None
    _HAS_MUON = False

try:
    from paddle.optimizer.inner_optimizer import InnerOptimizer

    from paddlefleet.optimizers import MuonOptimizer

    # Step 5 patches (mixed-optimizer sub-optimizers), restored per test.
    _INNER_ATTRS = (
        (InnerOptimizer, "_apply_optimize"),
        (MuonOptimizer, "_update_shape_group"),
    )
except (ImportError, ModuleNotFoundError):  # Paddle without InnerOptimizer
    _INNER_ATTRS = ()

_MISSING = object()
_MUON_ATTRS = ("_muon_update_group", "_muon_update")


class _ToDeviceSpy:
    """Records ``to_device`` invocations and returns the tensor unchanged.

    Returning the same tensor satisfies the in-place ``assert new is old``
    checks inside ``offload``/``reload``. ``reload`` calls ``to_device(t)`` so
    ``place is None``; ``offload`` and the ``_insert_sync`` move-back call
    ``to_device(t, place)`` with a concrete place. That lets us classify each
    recorded call as a reload (place is None) or an offload/move-back.
    """

    def __init__(self):
        self.calls = []

    def __call__(self, tensor, place=None):
        self.calls.append((tensor, place))
        return tensor

    @property
    def reloaded(self):
        return [t for (t, p) in self.calls if p is None]

    @property
    def offloaded(self):
        return [t for (t, p) in self.calls if p is not None]

    @property
    def offload_places(self):
        return [p for (t, p) in self.calls if p is not None]


@contextlib.contextmanager
def _force_cpu():
    """Force the CPU branch of ``offload`` so its target place is CPUPlace."""
    with (
        mock.patch("paddle.is_compiled_with_cuda", return_value=False),
        mock.patch("paddle.is_compiled_with_xpu", return_value=False),
    ):
        yield


def _tensor(values):
    return paddle.to_tensor(values, dtype="float32")


def _ids(tensors):
    return [id(t) for t in tensors]


@unittest.skipUnless(HAS_PADDLE, "paddle / paddlefleet not importable")
class _HackTestCase(unittest.TestCase):
    """Base case that keeps the process hermetic across hack invocations.

    The hacks mutate global/class state (``_C_ops``, ``Optimizer``,
    ``HybridParallelOptimizer`` and optionally ``Muon``). Snapshot the *true*
    originals first, then register cleanup so any locally installed stubs and
    hack patches are undone even if an assertion fails.
    """

    def setUp(self):
        saved = {
            (_C_ops, "adam_"): getattr(_C_ops, "adam_", _MISSING),
            (_C_ops, "adamw_"): getattr(_C_ops, "adamw_", _MISSING),
            (Optimizer, "_add_accumulator"): Optimizer._add_accumulator,
            (HybridParallelOptimizer, "_insert_sync"): (
                HybridParallelOptimizer._insert_sync
            ),
        }
        if _HAS_MUON:
            for attr in _MUON_ATTRS:
                saved[(Muon, attr)] = getattr(Muon, attr, _MISSING)
        for obj, attr in _INNER_ATTRS:
            saved[(obj, attr)] = getattr(obj, attr)

        def restore():
            for (obj, name), val in saved.items():
                if val is _MISSING:
                    if hasattr(obj, name):
                        delattr(obj, name)
                else:
                    setattr(obj, name, val)

        self.addCleanup(restore)


def _hacks():
    """Both hack variants, labeled for ``subTest``.

    Steps 1/3/4 are copy-pasted between the two, so parameterizing over both
    covers both copies of that logic where the expectation is the same.
    """
    return (
        ("plain", oo.hack_offload_optimizer),
        ("eb5", oo.hack_offload_optimizer_eb5),
    )


@unittest.skipUnless(HAS_PADDLE, "paddle / paddlefleet not importable")
class OffloadReloadTest(unittest.TestCase):
    """The ``offload``/``reload`` primitives: place selection + in-placeness."""

    def test_offload_selects_cpu_place_and_is_inplace_on_cpu_build(self):
        x = _tensor([[1.0, 2.0], [3.0, 4.0]])
        before = x.numpy().copy()
        with _force_cpu():
            ret = oo.offload(x)
        # No return value; migration is in-place so x itself is the result.
        self.assertIsNone(ret)
        self.assertTrue(x.place.is_cpu_place())
        np.testing.assert_array_equal(x.numpy(), before)

    def test_reload_targets_default_device_and_is_inplace_on_cpu_build(self):
        x = _tensor([9.0, 8.0, 7.0])
        before = x.numpy().copy()
        with (
            mock.patch("paddle.is_compiled_with_cuda", return_value=False),
            mock.patch("paddle.is_compiled_with_xpu", return_value=False),
            mock.patch("paddle.is_compiled_with_rocm", return_value=False),
        ):
            ret = oo.reload(x)
        self.assertIsNone(ret)
        self.assertTrue(x.place.is_cpu_place())
        np.testing.assert_array_equal(x.numpy(), before)

    def test_reload_passes_no_place_so_target_is_default_device(self):
        # Oracle: reload calls to_device(t) with no place argument.
        x = _tensor([1.0])
        spy = _ToDeviceSpy()
        with mock.patch.object(oo, "to_device", spy):
            oo.reload(x)
        self.assertEqual(len(spy.calls), 1)
        t, place = spy.calls[0]
        self.assertIs(t, x)
        self.assertIsNone(place)

    def test_offload_selects_pinned_place_per_compile_flag(self):
        # Observes only the place-selection arms; the pinned place classes are
        # stubbed with sentinels so this runs on any build without migrating.
        for flag, pinned in (
            ("paddle.is_compiled_with_cuda", "paddle.CUDAPinnedPlace"),
            ("paddle.is_compiled_with_xpu", "paddle.XPUPinnedPlace"),
        ):
            with self.subTest(flag=flag):
                x = _tensor([1.0, 2.0])
                spy = _ToDeviceSpy()
                sentinel = object()
                is_cuda = flag.endswith("cuda")
                with (
                    mock.patch(
                        "paddle.is_compiled_with_cuda", return_value=is_cuda
                    ),
                    mock.patch(
                        "paddle.is_compiled_with_xpu", return_value=not is_cuda
                    ),
                    mock.patch(pinned, return_value=sentinel),
                    mock.patch.object(oo, "to_device", spy),
                ):
                    oo.offload(x)
                self.assertEqual(spy.calls, [(x, sentinel)])


class AddAccumulatorTest(_HackTestCase):
    """Step 1: new optimizer state is offloaded at creation."""

    def test_add_accumulator_offloads_new_state_at_creation(self):
        for label, hack in _hacks():
            with self.subTest(hack=label):
                acc = _tensor([7.0, 8.0])
                recorded = {}

                def stub_add(self_opt, *args, **kwargs):
                    recorded["args"] = args
                    recorded["kwargs"] = kwargs
                    return acc

                Optimizer._add_accumulator = stub_add
                spy = _ToDeviceSpy()
                with _force_cpu(), mock.patch.object(oo, "to_device", spy):
                    hack()
                    ret = Optimizer._add_accumulator(
                        object(), "moment", fill_value=0.0
                    )
                # Call-through returns the fresh accumulator unchanged.
                self.assertIs(ret, acc)
                self.assertEqual(recorded["args"], ("moment",))
                self.assertEqual(recorded["kwargs"], {"fill_value": 0.0})
                # Offloaded exactly once, at creation; nothing is reloaded.
                self.assertEqual(_ids(spy.offloaded), _ids([acc]))
                self.assertEqual(spy.reloaded, [])
                for place in spy.offload_places:
                    self.assertTrue(place.is_cpu_place())


class OptOpWrapperTest(_HackTestCase):
    """Step 2: the ``adam_``/``adamw_`` reload-all / offload-state wrapper."""

    @staticmethod
    def _install_op_stubs():
        """Replace adam_/adamw_ with labeled stubs (the wrapped originals)."""
        labels = []

        def stub_adam(*args):
            labels.append("adam")
            return ("adam_ret", args)

        def stub_adamw(*args):
            labels.append("adamw")
            return ("adamw_ret", args)

        _C_ops.adam_ = stub_adam
        _C_ops.adamw_ = stub_adamw
        return labels

    def test_adamw_reloads_tensors_and_offloads_only_state_by_default(self):
        self._install_op_stubs()
        param, grad = _tensor([1.0]), _tensor([2.0])
        m1, m2 = _tensor([3.0]), _tensor([4.0])
        spy = _ToDeviceSpy()
        with _force_cpu(), mock.patch.object(oo, "to_device", spy):
            oo.hack_offload_optimizer()
            ret = _C_ops.adamw_(param, grad, m1, 0.001, "beta1", None, m2)
        # Reload happens for every tensor arg, in order, before the op;
        # non-tensor args (lr, a str, None) are never moved either way.
        self.assertEqual(_ids(spy.reloaded), _ids([param, grad, m1, m2]))
        # Only state (index >= 2) is offloaded; param/grad never are.
        self.assertEqual(_ids(spy.offloaded), _ids([m1, m2]))
        # Offload target is CPUPlace on a CPU build.
        for place in spy.offload_places:
            self.assertTrue(place.is_cpu_place())
        # Call-through: the wrapped op ran and its return propagates.
        self.assertEqual(ret[0], "adamw_ret")

    def test_eb5_adamw_does_not_offload_state_by_default(self):
        self._install_op_stubs()
        param, grad = _tensor([1.0]), _tensor([2.0])
        m1, m2 = _tensor([3.0]), _tensor([4.0])
        spy = _ToDeviceSpy()
        with _force_cpu(), mock.patch.object(oo, "to_device", spy):
            oo.hack_offload_optimizer_eb5()
            ret = _C_ops.adamw_(param, grad, m1, m2)
        # eb5 defaults is_offload_opt=False: still reloads, offloads nothing.
        self.assertEqual(_ids(spy.reloaded), _ids([param, grad, m1, m2]))
        self.assertEqual(spy.offloaded, [])
        self.assertEqual(ret[0], "adamw_ret")

    def test_eb5_adamw_offloads_state_when_param_opts_in(self):
        # eb5 defaults the gate to False; an explicit is_offload_opt=True on
        # args[0] is the only way to reach its offload arm.
        self._install_op_stubs()
        param, grad = _tensor([1.0]), _tensor([2.0])
        param.is_offload_opt = True
        m1, m2 = _tensor([3.0]), _tensor([4.0])
        spy = _ToDeviceSpy()
        with _force_cpu(), mock.patch.object(oo, "to_device", spy):
            oo.hack_offload_optimizer_eb5()
            _C_ops.adamw_(param, grad, m1, 0.5, m2)
        self.assertEqual(_ids(spy.reloaded), _ids([param, grad, m1, m2]))
        self.assertEqual(_ids(spy.offloaded), _ids([m1, m2]))
        for place in spy.offload_places:
            self.assertTrue(place.is_cpu_place())

    def test_mode_eb5_delegates_to_eb5_variant(self):
        # Observed via the eb5-specific default (no offload), not just
        # "something got patched".
        self._install_op_stubs()
        param, grad = _tensor([1.0]), _tensor([2.0])
        m1, m2 = _tensor([3.0]), _tensor([4.0])
        spy = _ToDeviceSpy()
        with _force_cpu(), mock.patch.object(oo, "to_device", spy):
            oo.hack_offload_optimizer(mode="eb5")
            ret = _C_ops.adamw_(param, grad, m1, m2)
        self.assertEqual(spy.offloaded, [])
        self.assertEqual(ret[0], "adamw_ret")

    @unittest.expectedFailure
    def test_adam_wrapper_should_dispatch_to_adam_not_adamw(self):
        """Known production bug: late-binding closure over ``origin_op``.

        Step 2 defines ``new_opt_op`` inside a ``for name in [...]`` loop that
        rebinds ``origin_op`` each iteration. Both installed closures share
        that one cell, so after the loop both dispatch to the *last* original
        (adamw_). Calling the patched ``adam_`` therefore runs adamw_'s
        original. This asserts the correct contract (adam_ -> adam_ original)
        and is expected to fail until the closure captures ``origin_op`` per
        name.
        """
        labels = self._install_op_stubs()
        param, grad = _tensor([1.0]), _tensor([2.0])
        m1, m2 = _tensor([3.0]), _tensor([4.0])
        spy = _ToDeviceSpy()
        with _force_cpu(), mock.patch.object(oo, "to_device", spy):
            oo.hack_offload_optimizer()
            _C_ops.adam_(param, grad, m1, m2)
        self.assertEqual(labels, ["adam"])


class InsertSyncTest(_HackTestCase):
    """Step 3: reload the sync var, run the original, then move it back."""

    @staticmethod
    def _install_stub_sync():
        recorded = {}

        def stub(self_opt, sync_var, *args, **kwargs):
            recorded["sync_var"] = sync_var
            recorded["args"] = args
            return "sync_ret"

        HybridParallelOptimizer._insert_sync = stub
        return recorded

    def test_move_back_follows_the_per_variant_is_offload_opt_gate(self):
        # Oracle: the gate defaults to True for the plain hack and False for
        # eb5, so each variant needs the opposite explicit value to reach its
        # other arm. The move-back targets origin_place, captured *before* the
        # original ran -- that is the tensor's own device (CPUPlace on a CPU
        # build, CUDAPlace on GPU CI), so compare against sync_var.place
        # rather than hard-coding CPU.
        cases = (
            ("plain", oo.hack_offload_optimizer, _MISSING, True),
            ("plain", oo.hack_offload_optimizer, False, False),
            ("eb5", oo.hack_offload_optimizer_eb5, _MISSING, False),
            ("eb5", oo.hack_offload_optimizer_eb5, True, True),
        )
        for label, hack, gate, expect_move_back in cases:
            with self.subTest(hack=label, is_offload_opt=gate):
                recorded = self._install_stub_sync()
                sync_var = _tensor([5.0])
                if gate is not _MISSING:
                    sync_var.is_offload_opt = gate
                origin_place = sync_var.place
                spy = _ToDeviceSpy()
                with _force_cpu(), mock.patch.object(oo, "to_device", spy):
                    hack()
                    ret = HybridParallelOptimizer._insert_sync(
                        object(), sync_var, "extra"
                    )
                self.assertEqual(ret, "sync_ret")
                self.assertIs(recorded["sync_var"], sync_var)
                self.assertEqual(recorded["args"], ("extra",))
                # The reload always happens, before the original runs.
                self.assertEqual(_ids(spy.reloaded), _ids([sync_var]))
                if expect_move_back:
                    self.assertEqual(_ids(spy.offloaded), _ids([sync_var]))
                    self.assertEqual(
                        str(spy.offload_places[0]), str(origin_place)
                    )
                else:
                    self.assertEqual(spy.offloaded, [])


class _Param:
    """Stand-in Muon param exposing only ``name`` and the offload gate."""

    def __init__(self, name, is_offload_opt=True):
        self.name = name
        self.is_offload_opt = is_offload_opt


class _MuonSelf:
    """Minimal ``self`` for the wrapped Muon update entry points."""

    _moment_acc_str = "moment1"

    def __init__(self, master_weights, momentums):
        self._master_weights = master_weights
        self._momentums = momentums

    def _get_accumulator(self, acc_str, param):
        return self._momentums[param.name]


@unittest.skipUnless(_HAS_MUON, "Muon optimizer unavailable")
class MuonOffloadTest(_HackTestCase):
    """Step 4: per-update-group momentum + master-weight lifecycle.

    Muon's momentum update is pure Python (paddle.lerp + paddle.assign), so it
    bypasses the ``_C_ops.adamw_`` patch and needs its own reload/offload. Both
    entry points are covered: the batched ``_muon_update_group`` on newer
    Paddle and the per-param ``_muon_update`` fallback on older releases.
    """

    @staticmethod
    def _record(events):
        """Patch module-level reload/offload to log tagged (op, id) events."""
        return (
            mock.patch.object(
                oo, "reload", lambda t: events.append(("reload", id(t)))
            ),
            mock.patch.object(
                oo, "offload", lambda t: events.append(("offload", id(t)))
            ),
        )

    @staticmethod
    def _drop_muon_update_entry_points():
        for attr in _MUON_ATTRS:
            if attr in Muon.__dict__:
                delattr(Muon, attr)

    def _drive_group(self, stub, params_grads, hack):
        """Install a recording original ``_muon_update_group``, apply the hack
        and run the resulting wrapper. Returns the ordered event log."""
        events = []
        Muon._muon_update_group = lambda self, pg, *a, **k: events.append(
            ("update",)
        )
        hack()
        wrapper = Muon._muon_update_group
        r, o = self._record(events)
        with r, o:
            wrapper(stub, params_grads)
        return events

    def _drive_update(self, stub, param, momentum, hack):
        """Force the per-param ``_muon_update`` fallback (no
        ``_muon_update_group``), apply the hack and run the wrapper."""
        events = []
        self._drop_muon_update_entry_points()
        Muon._muon_update = lambda self, *a, **k: events.append(("update",))
        hack()
        wrapper = Muon._muon_update
        r, o = self._record(events)
        with r, o:
            wrapper(
                stub,
                param,
                paddle.zeros([2, 2]),
                0.1,
                momentum,
                0.95,
                5,
                True,
                1e-9,
                0.0,
                3,
            )
        return events

    @staticmethod
    def _moves(events):
        return [e for e in events if e[0] in ("reload", "offload")]

    # ----- _muon_update_group (batched, newer Paddle) --------------------

    def test_group_reloads_before_and_offloads_after_when_enabled(self):
        for label, hack in _hacks():
            with self.subTest(hack=label):
                mw, mom = paddle.zeros([2, 2]), paddle.zeros([2, 2])
                param = _Param("w0")
                stub = _MuonSelf({"w0": mw}, {"w0": mom})

                events = self._drive_group(
                    stub, [(param, paddle.zeros([2, 2]))], hack
                )

                i = events.index(("update",))
                self.assertIn(("reload", id(mw)), events[:i])
                self.assertIn(("reload", id(mom)), events[:i])
                self.assertIn(("offload", id(mw)), events[i + 1 :])
                self.assertIn(("offload", id(mom)), events[i + 1 :])

    def test_group_reloads_but_never_offloads_when_disabled(self):
        for label, hack in _hacks():
            with self.subTest(hack=label):
                mw, mom = paddle.zeros([2, 2]), paddle.zeros([2, 2])
                param = _Param("w0", is_offload_opt=False)
                stub = _MuonSelf({"w0": mw}, {"w0": mom})

                events = self._drive_group(
                    stub, [(param, paddle.zeros([2, 2]))], hack
                )

                i = events.index(("update",))
                self.assertIn(("reload", id(mw)), events[:i])
                self.assertIn(("reload", id(mom)), events[:i])
                self.assertEqual([e for e in events if e[0] == "offload"], [])

    def test_group_offloads_only_the_enabled_param_in_a_mixed_group(self):
        mw_on, mw_off = paddle.zeros([2, 2]), paddle.zeros([2, 2])
        mom_on, mom_off = paddle.zeros([2, 2]), paddle.zeros([2, 2])
        p_on = _Param("on")
        p_off = _Param("off", is_offload_opt=False)
        stub = _MuonSelf(
            {"on": mw_on, "off": mw_off}, {"on": mom_on, "off": mom_off}
        )

        events = self._drive_group(
            stub,
            [(p_on, paddle.zeros([2, 2])), (p_off, paddle.zeros([2, 2]))],
            oo.hack_offload_optimizer,
        )

        i = events.index(("update",))
        # Reload is unconditional for the whole group; offload is per-param.
        self.assertIn(("reload", id(mw_on)), events[:i])
        self.assertIn(("reload", id(mw_off)), events[:i])
        self.assertIn(("offload", id(mw_on)), events[i + 1 :])
        self.assertIn(("offload", id(mom_on)), events[i + 1 :])
        self.assertNotIn(("offload", id(mw_off)), events)
        self.assertNotIn(("offload", id(mom_off)), events)

    def test_group_moves_momentum_only_without_a_usable_master_weight(self):
        # Two misses of the same arm: no master-weight dict at all, and a dict
        # that holds some other param's master weight.
        for label, hack in _hacks():
            for case, mw_dict in (
                ("no dict", None),
                ("other param", {"someone_else": paddle.zeros([2, 2])}),
            ):
                with self.subTest(hack=label, master_weights=case):
                    mom = paddle.zeros([2, 2])
                    param = _Param("w0")
                    stub = _MuonSelf(mw_dict, {"w0": mom})

                    events = self._drive_group(
                        stub, [(param, paddle.zeros([2, 2]))], hack
                    )

                    # Nothing but the momentum buffer is ever touched.
                    self.assertEqual(
                        self._moves(events),
                        [("reload", id(mom)), ("offload", id(mom))],
                    )

    # ----- _muon_update (per-param fallback, older Paddle) ---------------

    def test_update_reloads_before_and_offloads_after_when_enabled(self):
        for label, hack in _hacks():
            with self.subTest(hack=label):
                mw, mom = paddle.zeros([2, 2]), paddle.zeros([2, 2])
                param = _Param("w0")
                stub = _MuonSelf({"w0": mw}, {"w0": mom})

                events = self._drive_update(stub, param, mom, hack)

                i = events.index(("update",))
                self.assertIn(("reload", id(mw)), events[:i])
                self.assertIn(("reload", id(mom)), events[:i])
                self.assertIn(("offload", id(mw)), events[i + 1 :])
                self.assertIn(("offload", id(mom)), events[i + 1 :])

    def test_update_reloads_but_never_offloads_when_disabled(self):
        for label, hack in _hacks():
            with self.subTest(hack=label):
                mw, mom = paddle.zeros([2, 2]), paddle.zeros([2, 2])
                param = _Param("w0", is_offload_opt=False)
                stub = _MuonSelf({"w0": mw}, {"w0": mom})

                events = self._drive_update(stub, param, mom, hack)

                i = events.index(("update",))
                self.assertIn(("reload", id(mw)), events[:i])
                self.assertEqual([e for e in events if e[0] == "offload"], [])

    def test_update_moves_momentum_only_without_a_usable_master_weight(self):
        for label, hack in _hacks():
            for case, mw_dict in (
                ("no dict", None),
                ("other param", {"someone_else": paddle.zeros([2, 2])}),
            ):
                with self.subTest(hack=label, master_weights=case):
                    mom = paddle.zeros([2, 2])
                    param = _Param("w0")
                    stub = _MuonSelf(mw_dict, {"w0": mom})

                    events = self._drive_update(stub, param, mom, hack)

                    self.assertEqual(
                        self._moves(events),
                        [("reload", id(mom)), ("offload", id(mom))],
                    )

    # ----- neither entry point exists -----------------------------------

    def test_hacks_skip_step_4_when_muon_has_no_update_entry_point(self):
        self._drop_muon_update_entry_points()
        if any(hasattr(Muon, attr) for attr in _MUON_ATTRS):
            self.skipTest("Muon inherits an update entry point")
        for _, hack in _hacks():
            hack()
            # Step 4 found nothing to wrap and installed nothing.
            for attr in _MUON_ATTRS:
                self.assertFalse(hasattr(Muon, attr))


class MuonModuleUnavailableTest(_HackTestCase):
    """Step 4 is optional: both hacks tolerate Muon being absent."""

    def test_hacks_still_patch_steps_1_to_3_without_muon_module(self):
        with mock.patch.dict(sys.modules, {"paddle.optimizer.muon": None}):
            for label, hack in _hacks():
                with self.subTest(hack=label):
                    before = _C_ops.adamw_
                    hack()
                    self.assertIsNot(_C_ops.adamw_, before)


if __name__ == "__main__":
    unittest.main()
