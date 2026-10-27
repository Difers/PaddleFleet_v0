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
"""Cheap current-stream helpers shared by ``ring`` and ``service``."""

import contextlib

import paddle

# ``paddle.device.cuda.current_stream`` and ``stream_guard`` carry @deprecated
# wrappers, and the guard additionally walks signature_safe_contextmanager and
# calls ``current_stream`` itself.  Measured on this build (us per call):
# current_stream 7.2 vs 0.3 unwrapped, stream_guard 30.9 vs 16.2 for the two
# ``_set_current_stream`` calls it ends up making.  The new-API namespace is
# not a drop-in replacement (its Stream rejects an old-API Event), so stay on
# the old API and skip the wrappers.
_current_stream = getattr(
    paddle.device.cuda.current_stream,
    "__wrapped__",
    paddle.device.cuda.current_stream,
)
_set_current_stream = getattr(paddle.device.cuda, "_set_current_stream", None)


@contextlib.contextmanager
def _on_stream(stream):
    """``paddle.device.cuda.stream_guard(stream)``, cheaper; falls back to it
    if this paddle has no private stream setter."""
    if _set_current_stream is None:
        with paddle.device.cuda.stream_guard(stream):
            yield
        return
    prev = _set_current_stream(stream)
    try:
        yield
    finally:
        _set_current_stream(prev)
