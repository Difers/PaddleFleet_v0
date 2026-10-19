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
"""Muon math (Newton-Schulz / scaling) used by ``MuonOptimizer``.

Free functions, kept apart from the optimizer class so there is a single source
of truth for the reimplemented Muon math. Ported to be numerically identical to
the corresponding paths of ``paddle.optimizer.muon.Muon`` (which stays the
reference and is unchanged); verified by
``tests/single_card_tests/test_inner_optimizers_vs_muon.py``.
"""

from __future__ import annotations

import paddle
import paddle.nn.functional as F

__all__ = [
    "NS_COEFFICIENT_SETS",
    "split_func_key",
    "newton_schulz",
    "check_symmetric_gemm",
    "scaling",
    "auto_ns_matmul_dtype",
]

# ----------------------------------------------------------------------
# Newton-Schulz coefficient sets (verbatim from paddle.optimizer.muon)
# ----------------------------------------------------------------------
NS_COEFFICIENT_SETS = {
    "simple": [
        (3.4445, -4.7750, 2.0315),
    ],
    "quintic": [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ],
    "polar_express": [
        (8.2051, -22.9019, 16.4607),
        (4.0664, -2.8612, 0.5184),
        (3.9096, -2.8234, 0.5250),
        (3.2856, -2.4153, 0.4853),
        (2.2779, -1.6198, 0.3985),
        (1.8726, -1.2307, 0.3585),
        (1.8564, -1.2132, 0.3568),
        (1.8750, -1.2500, 0.3750),
    ],
    "aol": [
        (4.0098, -7.0585, 2.4635),
        (3.4585, -5.5479, 2.5959),
        (2.7573, -3.2939, 1.4254),
        (2.7215, -3.0494, 1.3169),
    ],
    "deepseekv4": [(3.4445, -4.7750, 2.0315)] * 8 + [(2.0, -1.5, 0.5)] * 2,
}


def _to_hashable(value):
    """Recursively convert lists/tuples/dicts into hashable tuples."""
    if isinstance(value, (list, tuple)):
        return tuple(_to_hashable(v) for v in value)
    if isinstance(value, dict):
        return tuple((k, _to_hashable(v)) for k, v in sorted(value.items()))
    return value


def split_func_key(split_concat_func):
    """Hashable grouping key for a ``split_concat_func`` (matches muon.py).

    Parameters with the same shape AND the same slice strategy can be batched
    into a single Newton-Schulz call. ``functools.partial`` is unwrapped so two
    partials of the same function with the same bound args group together.
    """
    if split_concat_func is None:
        return None
    if hasattr(split_concat_func, "func"):
        return (
            split_concat_func.func,
            tuple(_to_hashable(v) for v in split_concat_func.args),
            tuple(
                (k, _to_hashable(v))
                for k, v in sorted((split_concat_func.keywords or {}).items())
            ),
        )
    return split_concat_func


# ----------------------------------------------------------------------
# Newton-Schulz orthogonalisation + scaling + Hyperball
# ----------------------------------------------------------------------
def _newton_schulz_step(X, a, b, c):
    """One Newton-Schulz iteration for a 2D matrix."""
    A = paddle.matmul(X, X, transpose_y=True)
    B = paddle.addmm(input=A, x=A, y=A, beta=b, alpha=c)
    return paddle.addmm(input=X, x=B, y=X, beta=a, alpha=1.0)


def _newton_schulz_step_batched(X, a, b, c):
    """One Newton-Schulz iteration for a 3D (batched) stack of matrices."""
    A = paddle.matmul(X, X, transpose_y=True)
    B = paddle.baddbmm(A, A, A, beta=b, alpha=c)
    return paddle.baddbmm(X, B, X, beta=a, alpha=1.0)


def _newton_schulz_step_symmetric(X, a, b, c):
    """2D Newton-Schulz step, symmetric products on quack's SYRK kernel."""
    from paddle.optimizer.muon import _load_symmetric_gemm

    gemm_symmetric = _load_symmetric_gemm()
    A = gemm_symmetric(X, paddle.transpose(X, perm=[1, 0]))
    B = gemm_symmetric(A, A, C=A, beta=b, alpha=c)
    return paddle.addmm(input=X, x=B, y=X, beta=a, alpha=1.0)


def _newton_schulz_step_batched_symmetric(X, a, b, c):
    """3D Newton-Schulz step, symmetric products on quack's SYRK kernel."""
    from paddle.optimizer.muon import _load_symmetric_gemm

    gemm_symmetric = _load_symmetric_gemm()
    A = gemm_symmetric(X, paddle.transpose(X, perm=[0, 2, 1]))
    B = gemm_symmetric(A, A, C=A, beta=b, alpha=c)
    return paddle.baddbmm(X, B, X, beta=a, alpha=1.0)


def check_symmetric_gemm(matmul_dtype, min_short_edge, min_step_flops):
    """Eagerly validate ``use_symmetric_gemm=True`` (same checks as Muon's ctor)."""
    from paddle.optimizer.muon import (
        _SYRK_SUPPORTED_DTYPES,
        _SYRK_SUPPORTED_MAJOR_CAPABILITIES,
        _load_symmetric_gemm,
    )

    if min_short_edge <= 0 or min_step_flops <= 0:
        raise ValueError(
            "symmetric_gemm_min_short_edge and symmetric_gemm_min_step_flops "
            f"must be positive, got {min_short_edge} and {min_step_flops}."
        )
    if matmul_dtype not in _SYRK_SUPPORTED_DTYPES:
        raise ValueError(
            "use_symmetric_gemm=True requires bfloat16 or float16 "
            f"Newton-Schulz matmuls, got ns_matmul_dtype={matmul_dtype}."
        )
    cap = (
        paddle.device.cuda.get_device_capability()
        if paddle.is_compiled_with_cuda()
        else (0, 0)
    )
    if cap[0] not in _SYRK_SUPPORTED_MAJOR_CAPABILITIES:
        raise ValueError(
            "use_symmetric_gemm=True requires a GPU with compute capability "
            f"9.x/10.x/11.x, got {cap[0]}.{cap[1]}."
        )
    _load_symmetric_gemm()


def newton_schulz(
    X,
    steps,
    eps,
    ns_coeffs,
    matmul_dtype,
    use_symmetric_gemm=False,
    symmetric_gemm_min_short_edge=None,
    symmetric_gemm_min_step_flops=None,
):
    """Approximate the matrix sign function via Newton-Schulz iteration.

    Faithful port of ``Muon._zeropower_via_newtonschulz5``. Accepts a 2D
    ``(M, N)`` matrix or a 3D ``(B, M, N)`` batch, row-normalises it, then
    applies ``steps`` iterations of the (a, b, c) polynomial (coefficients
    cycled). Returns the orthogonalised matrix in ``matmul_dtype``. With
    ``use_symmetric_gemm`` the two symmetric products of each step go through
    quack's SYRK kernel, for shapes that clear Muon's profitability gate.
    """
    if X.ndim < 2 or X.ndim > 3:
        raise ValueError(f"X must be 2D or 3D (batched), got {X.ndim}D")
    coeff_sets = (
        ns_coeffs if ns_coeffs is not None else NS_COEFFICIENT_SETS["simple"]
    )

    # Newton-Schulz assumes rows <= cols; transpose tall matrices, undo at the end.
    if X.shape[-2] > X.shape[-1]:
        X = paddle.transpose(X, perm=[1, 0] if X.ndim == 2 else [0, 2, 1])
        transposed = True
    else:
        transposed = False

    orig_shape = X.shape
    X_flat = X.flatten(start_axis=-2)
    X_flat = F.normalize(X_flat, p=2, axis=-1, epsilon=eps)
    X = X_flat.reshape(orig_shape).astype(matmul_dtype)

    symmetric = False
    if use_symmetric_gemm:
        from paddle.optimizer.muon import _symmetric_gemm_is_profitable

        symmetric = _symmetric_gemm_is_profitable(
            X.shape,
            X.dtype,
            symmetric_gemm_min_short_edge,
            symmetric_gemm_min_step_flops,
        )
    if X.ndim == 3:
        step_fn = (
            _newton_schulz_step_batched_symmetric
            if symmetric
            else _newton_schulz_step_batched
        )
    else:
        step_fn = (
            _newton_schulz_step_symmetric if symmetric else _newton_schulz_step
        )
    for i in range(steps):
        a, b, c = coeff_sets[i % len(coeff_sets)]
        X = step_fn(X, a, b, c)

    if transposed:
        X = paddle.transpose(X, perm=[1, 0] if X.ndim == 2 else [0, 2, 1])
    return X


def scaling(orthogonal_update, version, extra_scale_factor=1.0):
    """Dimension-dependent scaling of the orthogonal update (Muon._scaling_fn)."""
    din, dout = orthogonal_update.shape[-2], orthogonal_update.shape[-1]
    if version == 1:
        scale = max(1, dout / din) ** 0.5
    elif version == 2:
        scale = (dout / din) ** 0.5
    else:  # version == 3 (default)
        scale = max(dout, din) ** 0.5
    return orthogonal_update * scale * extra_scale_factor


def auto_ns_matmul_dtype():
    """bfloat16 on Ampere+ (capability >= 8.0), float32 on older / CPU."""
    cap = (
        paddle.device.cuda.get_device_capability()
        if paddle.is_compiled_with_cuda()
        else (0, 0)
    )
    return paddle.bfloat16 if cap[0] >= 8 else paddle.float32
