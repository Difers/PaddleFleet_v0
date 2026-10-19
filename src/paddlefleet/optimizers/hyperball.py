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
"""Hyperball projection shared by the ``*h`` optimizers (muonh / adamh / rmsproph).

Each of those optimizers computes its own update direction and then calls
:func:`hyperball_apply` instead of weight decay + subtract.
"""

from __future__ import annotations

import paddle

__all__ = ["hyperball_apply", "hyperball_step"]


def hyperball_apply(w, update, lr, eps=1e-12, split_concat_func=None):
    r"""Hyperball step (``Muon._hyperball_apply``): project onto a fixed-R sphere.

        R       = ||w||_F                      (recomputed every step)
        W_{t+1} = R * Normalize(w - lr * R * Normalize(update))

    ``R == ||W_0||_F`` is held constant by the projection and is intentionally
    NOT stored in the checkpoint. For 3D grouped-expert params ``[E, H, I]`` the
    Frobenius norm is taken per expert (last two dims). All math is fp32; the
    fp32 result is returned (the caller casts back to the parameter dtype).

    With ``split_concat_func`` (fused QKV, gate_up, MLA heads, ...) the step is
    applied to the same slices that were orthogonalised, each on its own sphere
    ``R_s = ||W_s||_F``; projecting the fused matrix as a whole would only hold
    the total norm and let the slice norms drift. ``split_concat_func`` takes
    one matrix, so it is run on ``w`` to collect the weight slices, then on
    ``update`` with a callback that pairs slices in call order (the slicing
    depends only on the shape).
    """
    if len(w.shape) not in (2, 3):
        raise ValueError(
            f"Hyperball only supports 2D/3D parameters, got shape {w.shape}."
        )
    if split_concat_func is not None:
        w_slices = []
        split_concat_func(w, lambda s: w_slices.append(s) or s)
        w_iter = iter(w_slices)
        return split_concat_func(
            update,
            lambda u: hyperball_apply(next(w_iter), u, lr, eps),
        )

    def _fro(x):
        return paddle.norm(x, p="fro", axis=[-2, -1], keepdim=True)

    w32 = w.astype(paddle.float32)
    u32 = update.astype(paddle.float32)
    radius = _fro(w32)
    normed_update = u32 / (_fro(u32) + eps)
    trial_w = w32 - lr * radius * normed_update
    return radius * (trial_w / (_fro(trial_w) + eps))


def hyperball_step(opt, param, update, lr, split_concat_func=None):
    """Apply :func:`hyperball_apply` to *param* (its fp32 master weight when
    *opt* keeps one) and write the result back to both."""
    find_master = param.name in opt._master_weights
    w = opt._master_weights[param.name] if find_master else param
    new_w = hyperball_apply(
        w,
        update,
        opt._effective_lr(param, lr),
        split_concat_func=split_concat_func,
    )
    if find_master:
        paddle.assign(new_w, w)
    paddle.assign(new_w.astype(param.dtype), param)
