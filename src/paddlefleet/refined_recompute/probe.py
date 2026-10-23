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
"""Measure the time and memory of every refined-recompute point, RR off.

Each point's boundary runs once in the recompute pass; that execution is what
RR would skip, so its span and allocated-memory delta are the point's price.
The time is an upper bound: it excludes RR's own cost and includes any wait on
another stream. Off unless the yaml sets ``auto_refined_recompute.mode: profile``
(``TrainingArguments`` then calls ``activate``); when on, the candidate points
(flash_attn, mla_o_proj) are measured and none runs RR, whatever
``recompute_modules`` lists. ``moe_combine`` is not a candidate: it follows
``recompute_modules``, so its frames are part of the baseline.

Only the optimizer steps inside the window are priced: ``begin_step(step)``
opens one, ``end_step()`` closes it, and ``flush()`` ends the window. The
trainer callback in ``probe_callback`` drives it from
``auto_refined_recompute.calibration``. ``summarize()`` reduces the window to
one ``Price`` per (point, layer); the layer is the logical index
(``recompute_utils.layer_index_of``).

When the window ends each rank writes one JSON line per point, layer and call
to ``rank{N}.jsonl``: ``{"point", "layer", "step", "ms", "bytes"}``.

While the window is open the allocator peak is also split by where it falls
in the step (``PhasePeaks``), which is what a memory check that knows the
frames' lifetimes needs. One line per step goes to ``phase_rank{N}.jsonl``:
``{"step", "O", "F", "B", "order", "unknown_layer_events", "reserved_peak"}``.
The peak counter is reset at every event in the window, so the Trainer's logged
``max_memory_allocated`` is not the step peak there; ``max(O, F, max B)`` is.
"""

import json
import logging
import os
import statistics
from collections import Counter, defaultdict, namedtuple

logger = logging.getLogger(__name__)

__all__ = ["PROBE", "Price"]

Record = namedtuple("Record", "key step start end delta")
# ms: median of one call's recompute-pass span; bytes: largest frame seen;
# calls_per_step: median number of calls per optimizer step.
Price = namedtuple("Price", "ms bytes calls_per_step")


def _in_recompute_pass():
    from paddle.base import framework

    return framework._dygraph_tracer()._has_grad


def _layer_of(run_function):
    """Logical index of the layer a recompute segment belongs to, else None.

    A layer reaches ``recompute`` as a bound method, as itself, or as a closure
    over ``self`` (``TransformerLayer.forward`` wraps its body in one).
    """
    layer = getattr(run_function, "__self__", None)
    if layer is None and hasattr(run_function, "layer_number"):
        layer = run_function
    code = getattr(run_function, "__code__", None)
    if layer is None and code is not None and "self" in code.co_freevars:
        cell = run_function.__closure__[code.co_freevars.index("self")]
        layer = cell.cell_contents
    if getattr(layer, "layer_number", None) is None:
        return None
    from paddlefleet.recompute_utils import layer_index_of

    return layer_index_of(layer)


class PhasePeaks:
    """Where in one step the allocator peak falls; no event list is kept.

    ``cut`` is called at every recompute event with the allocator peak since
    the previous event. ``O``: outside the forward/backward span (before the
    first forward, after the last backward: start-up, optimizer). ``F``:
    inside it but in no layer's backward. ``B[layer]``: that layer's backward,
    recompute included. A frame lives from its layer's forward to its
    backward, so every frame is alive in ``F`` and only the earlier layers'
    are in ``B``. A segment of unknown layer (None) is counted in ``F``, which
    assumes at least as many frames as it can have.
    """

    def __init__(self):
        self.O = self.F = self.unknown = 0
        self.B = {}
        self.order = []  # layers in the order their first forward ran
        self._started = self._tail = False

    def cut(self, kind, layer, peak):
        if layer is None and kind.startswith("rc"):
            self.unknown += 1
        if kind == "step_begin":  # the segment before it is not in the step
            self._started = self._tail = False
        elif kind == "rcB_end":
            if layer is None:
                self.F = max(self.F, peak)
            else:
                self.B[layer] = max(self.B.get(layer, 0), peak)
            self._tail = True
        elif kind in ("rcF_begin", "rcF_end", "rcB_begin"):
            if kind == "rcF_begin" and layer not in (None, *self.order):
                self.order.append(layer)
            if kind == "rcF_begin" and not self._started:
                self.O = max(self.O, peak)
                self._started = True
            else:
                self.F = max(self.F, peak)
            self._tail = False
        elif (
            self._tail or not self._started
        ):  # step_end after the last backward
            self.O = max(self.O, peak)
        else:
            self.F = max(self.F, peak)


class RRProbe:
    """Drop-in for ``AutoRefinedRecompute`` that measures instead of skipping."""

    def __init__(self, probe, point, layer):
        self._probe = probe
        self._key = (point, layer)

    def __call__(self, function, *inputs):
        if not _in_recompute_pass():
            return function(*inputs)
        return self._probe.measure(self._key, function, inputs, {})


class _Window:
    """What one measuring window holds, from its first step to ``flush``."""

    def __init__(self, restore):
        self.restore = restore  # puts the recompute hooks back
        self.records = []
        self.rows = []  # one phase row per finished step
        self.step = None  # optimizer step being priced; None between steps
        self.phase = None  # PhasePeaks of the open step


class _Probe:
    def __init__(self):
        self._directory = None
        self._window = None

    def configure(self, section, output_dir):
        """Activate for ``auto_refined_recompute.mode: profile``; other modes do nothing."""
        mode = section.get("mode", "train")
        if mode not in ("train", "profile"):
            raise ValueError(
                f"auto_refined_recompute.mode must be train or profile, got {mode!r}"
            )
        if mode == "profile":
            self.activate(os.path.join(output_dir, "auto_rr"))

    def activate(self, directory):
        self._directory = directory
        logger.warning(
            f"[RR-PROBE] measuring every refined-recompute point with RR off, "
            f"writing to {directory}"
        )

    def deactivate(self):
        self._drop_window()
        self._directory = None

    def _drop_window(self):
        if self._window is not None:
            self._window.restore()
            self._window = None

    @property
    def current_step(self):
        return None if self._window is None else self._window.step

    def begin_step(self, step):
        """Price calls made until ``end_step``; the first call opens the window."""
        if self._window is None:
            self._window = _Window(self._install_hooks())
        self._window.step, self._window.phase = step, PhasePeaks()
        self._event("step_begin")

    def end_step(self):
        window = self._window
        if window is None or window.step is None:
            return
        import paddle

        self._event("step_end")
        phase = window.phase
        window.rows.append(
            {
                "step": window.step,
                "O": phase.O,
                "F": phase.F,
                "B": phase.B,
                "order": phase.order,
                "unknown_layer_events": phase.unknown,
                "reserved_peak": paddle.device.cuda.max_memory_reserved(),
            }
        )
        window.step = window.phase = None

    def phase_rows(self):
        """The finished steps' rows, as written to ``phase_rank{N}.jsonl``."""
        return [] if self._window is None else list(self._window.rows)

    def _event(self, kind, layer=None):
        window = self._window
        if window is None or window.step is None:
            return  # hooks stay installed between steps, e.g. during eval
        import paddle

        cuda = paddle.device.cuda
        peak = cuda.max_memory_allocated()
        cuda.reset_max_memory_allocated()
        window.phase.cut(kind, layer, peak)

    def _install_hooks(self):
        """Mark the recompute segments: first forward, and backward."""
        from paddle.distributed.fleet.recompute.recompute import (
            RecomputeFunction as cls,
        )

        saved = vars(cls)["forward"], vars(cls)["backward"]
        forward, backward = cls.forward, cls.backward

        def marked_forward(ctx, run_function, *args, **kwargs):
            ctx._rr_layer = _layer_of(run_function)
            self._event("rcF_begin", ctx._rr_layer)
            outputs = forward(ctx, run_function, *args, **kwargs)
            self._event("rcF_end", ctx._rr_layer)
            return outputs

        def marked_backward(ctx, *args):
            layer = getattr(ctx, "_rr_layer", None)
            self._event("rcB_begin", layer)
            grads = backward(ctx, *args)
            self._event("rcB_end", layer)
            return grads

        cls.forward = staticmethod(marked_forward)
        cls.backward = staticmethod(marked_backward)

        def restore():
            cls.forward, cls.backward = saved

        return restore

    @property
    def active(self):
        return self._directory is not None

    def boundary(self, point, layer):
        return RRProbe(self, point, layer)

    def wrap(self, function, point, owner):
        """For points with a hand-written RR, which the probe keeps off."""
        if not self.active:
            return function
        from paddlefleet.recompute_utils import (
            layer_index_of,
            need_full_recompute,
        )

        vpp = getattr(
            owner.config, "virtual_pipeline_model_parallel_size", None
        )
        if not need_full_recompute(owner.layer_number, owner.config) or (
            vpp is not None and vpp > 1
        ):
            return function
        key = (point, layer_index_of(owner))

        def probed(*args, **kwargs):
            if not _in_recompute_pass():
                return function(*args, **kwargs)
            return self.measure(key, function, args, kwargs)

        return probed

    def measure(self, key, function, args, kwargs):
        window = self._window
        if window is None or window.step is None:
            return function(*args, **kwargs)
        import paddle

        before = paddle.device.cuda.memory_allocated()
        start = paddle.device.Event(enable_timing=True)
        end = paddle.device.Event(enable_timing=True)
        start.record()
        outputs = function(*args, **kwargs)
        end.record()
        # elapsed_time synchronizes, so spans are resolved in _rows().
        delta = paddle.device.cuda.memory_allocated() - before
        window.records.append(Record(key, window.step, start, end, delta))
        return outputs

    def _rows(self):
        """The records as plain dicts; the only place events are resolved."""
        import paddle

        paddle.device.synchronize()
        records = [] if self._window is None else self._window.records
        return [
            {
                "point": r.key[0],
                "layer": r.key[1],
                "step": r.step,
                "ms": r.start.elapsed_time(r.end),
                "bytes": r.delta,
            }
            for r in records
        ]

    def summarize(self):
        """``{(point, layer): Price}`` over everything recorded so far."""
        rows = defaultdict(list)
        for row in self._rows():
            rows[row["point"], row["layer"]].append(row)
        return {
            slot: Price(
                ms=statistics.median(r["ms"] for r in calls),
                bytes=max(r["bytes"] for r in calls),
                calls_per_step=int(
                    statistics.median(
                        Counter(r["step"] for r in calls).values()
                    )
                ),
            )
            for slot, calls in rows.items()
        }

    def flush(self):
        window = self._window
        if window is None:
            return
        if window.records or window.rows:
            rank = os.environ.get("PADDLE_TRAINER_ID", "0")
            os.makedirs(self._directory, exist_ok=True)
            for name, lines in (
                ("rank", self._rows()),
                ("phase_rank", window.rows),
            ):
                if lines:
                    path = os.path.join(self._directory, f"{name}{rank}.jsonl")
                    with open(path, "w") as stream:
                        stream.writelines(
                            json.dumps(row) + "\n" for row in lines
                        )
        self._drop_window()


PROBE = _Probe()
