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

"""``Trainer._inner_training_loop`` rejects ``calculate_per_token_loss``.

The loss module returns a raw token sum under that flag, and this training loop
has no 1/T_global gradient normalization for it, so it must fail loudly before
any step runs. Called unbound with a ``types.SimpleNamespace`` as ``self``: the
guard is the first statement, so nothing else on ``self`` is touched when it
fires.
"""

from __future__ import annotations

import types

import pytest

from paddlefleet.trainer import trainer


def _fake_trainer(**config):
    return types.SimpleNamespace(
        model=types.SimpleNamespace(config=types.SimpleNamespace(**config))
    )


def _run(fake):
    trainer.Trainer._inner_training_loop(
        fake, None, None, None, 0, 0, 0, 0, 0, None, None
    )


def test_per_token_loss_is_rejected():
    with pytest.raises(NotImplementedError, match="calculate_per_token_loss"):
        _run(_fake_trainer(calculate_per_token_loss=True))


def test_per_token_loss_is_rejected_through_wrappers():
    # config only on the inner layer, as with fleet.distributed_model / DDP.
    inner = types.SimpleNamespace(
        config=types.SimpleNamespace(calculate_per_token_loss=True)
    )
    for fake in (
        types.SimpleNamespace(
            model=types.SimpleNamespace(),
            model_wrapped=types.SimpleNamespace(_layers=inner),
        ),
        types.SimpleNamespace(
            model=types.SimpleNamespace(
                module=types.SimpleNamespace(_layers=inner)
            )
        ),
    ):
        with pytest.raises(
            NotImplementedError, match="calculate_per_token_loss"
        ):
            _run(fake)


@pytest.mark.parametrize(
    "fake",
    [
        _fake_trainer(calculate_per_token_loss=False),
        _fake_trainer(),
        types.SimpleNamespace(model=types.SimpleNamespace()),
    ],
)
def test_guard_passes_when_flag_is_off_or_absent(fake):
    # Past the guard the loop needs real trainer state, so it fails on the next
    # attribute access instead -- anything but the guard's NotImplementedError.
    with pytest.raises(Exception) as excinfo:
        _run(fake)
    assert "calculate_per_token_loss" not in str(excinfo.value)
