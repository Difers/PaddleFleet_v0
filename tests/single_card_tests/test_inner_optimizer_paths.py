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
"""Branches of the PaddleFleet inner optimizers / Muon math that the
bit-exact comparison tests do not reach: 2D Newton-Schulz, the symmetric-GEMM
path (with a matmul stand-in for quack's SYRK), scaling versions, argument
validation, byte-capped grouping, split_concat_func, master weights,
maximize / coupled weight decay and the yaml descriptors.

Run:  python -m pytest tests/single_card_tests/test_inner_optimizer_paths.py -v
"""

import functools
import unittest
from unittest import mock

import numpy as np
import paddle

from paddlefleet.optimizers import hyperball, muon_math

try:
    import paddle.optimizer.muon as paddle_muon
except (ImportError, ModuleNotFoundError):
    # muon_math imports the symmetric-GEMM helpers from here.
    paddle_muon = None

try:
    from paddle.optimizer.mixed_inner_optimizer import get_inner_optimizer
    from paddle.optimizer.muon import MuonParamInfo

    import paddlefleet.optimizers  # noqa: F401  (registers the optimizers)
except (ImportError, ModuleNotFoundError):
    # Paddle build without the inner-optimizer API.
    get_inner_optimizer = None

_needs_api = unittest.skipIf(
    get_inner_optimizer is None, "Paddle build without the inner-optimizer API"
)
_needs_paddle_muon = unittest.skipIf(
    paddle_muon is None, "Paddle build without paddle.optimizer.muon"
)


def _fake_gemm_symmetric(A, B, C=None, beta=1.0, alpha=1.0):
    """Dense stand-in for quack's ``gemm_symmetric``: alpha*A@B + beta*C."""
    out = paddle.matmul(A, B) * alpha
    return out if C is None else out + C * beta


def _param(shape, dtype="float32", seed=0):
    rng = np.random.default_rng(seed)
    w0 = rng.standard_normal(shape).astype("float32")
    p = paddle.create_parameter(
        shape=list(shape),
        dtype=dtype,
        default_initializer=paddle.nn.initializer.Assign(w0),
    )
    return p


def _grads(shape, steps=3, seed=1):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(shape).astype("float32") for _ in range(steps)]


def _step(opt, params, grads):
    for g in grads:
        opt._apply_optimize(
            None,
            None,
            [(p, paddle.to_tensor(g).astype(p.dtype)) for p in params],
        )


class TestMuonMath(unittest.TestCase):
    def setUp(self):
        paddle.set_device("cpu")

    def test_split_func_key(self):
        def fn(x, cb, n=1):
            return cb(x)

        self.assertIsNone(muon_math.split_func_key(None))
        self.assertIs(muon_math.split_func_key(fn), fn)
        k1 = muon_math.split_func_key(
            functools.partial(fn, [1, (2, 3)], n={"a": [4]})
        )
        k2 = muon_math.split_func_key(
            functools.partial(fn, [1, (2, 3)], n={"a": [4]})
        )
        self.assertEqual(k1, k2)
        hash(k1)
        self.assertEqual(
            muon_math._to_hashable({"b": [1], "a": 2}), (("a", 2), ("b", (1,)))
        )

    def test_newton_schulz_2d_matches_3d(self):
        x = paddle.randn([24, 16])  # tall: transposed internally
        kw = {
            "steps": 5,
            "eps": 1e-7,
            "ns_coeffs": None,
            "matmul_dtype": paddle.float32,
        }
        out2d = muon_math.newton_schulz(x, **kw)
        out3d = muon_math.newton_schulz(x.unsqueeze(0), **kw)
        self.assertEqual(out2d.shape, [24, 16])
        np.testing.assert_allclose(
            out2d.numpy(), out3d[0].numpy(), rtol=1e-5, atol=1e-6
        )
        for bad in ([4], [2, 3, 4, 5]):
            with self.assertRaises(ValueError):
                muon_math.newton_schulz(paddle.randn(bad), **kw)

    @_needs_paddle_muon
    def test_newton_schulz_symmetric_path(self):
        kw = {
            "steps": 3,
            "eps": 1e-7,
            "ns_coeffs": muon_math.NS_COEFFICIENT_SETS["quintic"],
            "matmul_dtype": paddle.float32,
        }
        with (
            mock.patch.object(
                paddle_muon,
                "_load_symmetric_gemm",
                lambda: _fake_gemm_symmetric,
            ),
            mock.patch.object(
                paddle_muon,
                "_symmetric_gemm_is_profitable",
                lambda *a: True,
            ),
        ):
            for shape in ([16, 24], [2, 16, 24]):
                x = paddle.randn(shape)
                sym = muon_math.newton_schulz(
                    x,
                    use_symmetric_gemm=True,
                    symmetric_gemm_min_short_edge=1,
                    symmetric_gemm_min_step_flops=1,
                    **kw,
                )
                ref = muon_math.newton_schulz(x, **kw)
                np.testing.assert_allclose(
                    sym.numpy(), ref.numpy(), rtol=1e-4, atol=1e-5
                )

    @_needs_paddle_muon
    def test_check_symmetric_gemm(self):
        with self.assertRaises(ValueError):
            muon_math.check_symmetric_gemm(paddle.bfloat16, 0, 1)
        with self.assertRaises(ValueError):
            muon_math.check_symmetric_gemm(paddle.float32, 1, 1)
        with (
            mock.patch.object(paddle, "is_compiled_with_cuda", lambda: False),
            self.assertRaises(ValueError),
        ):
            muon_math.check_symmetric_gemm(paddle.bfloat16, 1, 1)
        loaded = []
        with (
            mock.patch.object(paddle, "is_compiled_with_cuda", lambda: True),
            mock.patch.object(
                paddle.device.cuda, "get_device_capability", lambda: (9, 0)
            ),
            mock.patch.object(
                paddle_muon, "_load_symmetric_gemm", lambda: loaded.append(1)
            ),
        ):
            muon_math.check_symmetric_gemm(paddle.bfloat16, 1, 1)
        self.assertEqual(loaded, [1])

    def test_scaling_versions(self):
        u = paddle.ones([4, 16])
        for version, scale in ((1, 2.0), (2, 2.0), (3, 4.0)):
            out = muon_math.scaling(u, version, extra_scale_factor=0.5)
            np.testing.assert_allclose(out.numpy(), 0.5 * scale, rtol=1e-6)
        out = muon_math.scaling(paddle.ones([16, 4]), 2)
        np.testing.assert_allclose(out.numpy(), 0.5, rtol=1e-6)

    def test_auto_ns_matmul_dtype(self):
        with mock.patch.object(paddle, "is_compiled_with_cuda", lambda: False):
            self.assertEqual(muon_math.auto_ns_matmul_dtype(), paddle.float32)
        with (
            mock.patch.object(paddle, "is_compiled_with_cuda", lambda: True),
            mock.patch.object(
                paddle.device.cuda, "get_device_capability", lambda: (8, 0)
            ),
        ):
            self.assertEqual(muon_math.auto_ns_matmul_dtype(), paddle.bfloat16)

    def test_hyperball_rejects_1d(self):
        with self.assertRaises(ValueError):
            hyperball.hyperball_apply(paddle.ones([4]), paddle.ones([4]), 0.1)


@_needs_api
class TestMuonOptimizer(unittest.TestCase):
    def setUp(self):
        paddle.set_device("cpu")

    def _muon(self, params, name="muon", **kw):
        kw.setdefault("weight_decay", 0.1)
        kw.setdefault("ns_matmul_dtype", paddle.float32)
        return get_inner_optimizer(name)(
            parameters=params, learning_rate=0.02, **kw
        )

    def test_ctor_and_descriptors(self):
        p = _param([8, 4])
        with self.assertRaises(ValueError):
            self._muon([p], ns_coeff_type="custom")
        with self.assertRaises(ValueError):
            self._muon([p], ns_coeff_type="nope")
        coeffs = [(3.0, -3.2, 1.2)]
        opt = self._muon([p], ns_coeff_type="custom", ns_coeffs=coeffs)
        self.assertEqual(opt._ns_coeffs, coeffs)
        cls = type(opt)
        self.assertIn("momentum", cls.yaml_field_map())
        self.assertIn("use_symmetric_gemm", cls.optional_field_map())
        self.assertEqual(opt.optimizer_state_layout()["vector"], ["moment1_0"])
        with mock.patch(
            "paddlefleet.optimizers.muon_optimizer.check_symmetric_gemm"
        ) as chk:
            self._muon([p], use_symmetric_gemm=True)
        chk.assert_called_once()

    def test_rejects_1d(self):
        p = _param([8])
        opt = self._muon([p])
        with self.assertRaises(ValueError):
            _step(opt, [p], _grads([8], steps=1))

    def test_byte_cap_does_not_change_result(self):
        # Splitting a same-shape group into one-param sub-groups (cap) or not
        # splitting at all (cap = 0) must give the same per-param update.
        out = []
        for cap in (0, 1, 2 * 1024**3):
            ps = [_param([8, 4], seed=s) for s in range(3)]
            opt = self._muon(ps, muon_max_group_bytes=cap, nesterov=False)
            for g in _grads([8, 4]):
                opt._apply_optimize(
                    None, None, [(p, paddle.to_tensor(g)) for p in ps]
                )
            out.append([p.numpy() for p in ps])
        for other in out[1:]:
            for a, b in zip(out[0], other):
                np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)

    def test_split_concat_func_and_no_decay(self):
        def halves(x, fn):
            a, b = paddle.split(x, 2, axis=-1)
            return paddle.concat([fn(a), fn(b)], axis=-1)

        for name in ("muon", "muonh"):
            p = _param([8, 8])
            info = MuonParamInfo(use_muon=True)
            info.split_concat_func = halves
            opt = self._muon(
                [p],
                name=name,
                param_info_map={p.name: info},
                apply_decay_param_fun=lambda n: False,
            )
            w0 = p.numpy()
            _step(opt, [p], _grads([8, 8]))
            self.assertFalse(np.array_equal(p.numpy(), w0))
            if name == "muonh":
                # Each slice keeps its own norm.
                for s in (slice(0, 4), slice(4, 8)):
                    np.testing.assert_allclose(
                        np.linalg.norm(p.numpy()[:, s]),
                        np.linalg.norm(w0[:, s]),
                        rtol=1e-5,
                    )

    def test_master_weights(self):
        # bf16 params keep an fp32 master weight; the update runs on it.
        # (adamw is left out: the fused CPU adamw kernel has no bf16.)
        for name in ("muon", "muonh", "adamh", "rmsprop", "rmsproph"):
            with self.subTest(name=name):
                p = _param([8, 4], dtype="bfloat16")
                kw = {"weight_decay": 0.1}
                if name.startswith("muon"):
                    kw["ns_matmul_dtype"] = paddle.float32
                opt = get_inner_optimizer(name)(
                    parameters=[p], learning_rate=0.02, **kw
                )
                _step(opt, [p], _grads([8, 4]))
                self.assertIn(p.name, opt._master_weights)
                master = opt._master_weights[p.name]
                self.assertEqual(master.dtype, paddle.float32)
                np.testing.assert_array_equal(
                    master.astype(paddle.bfloat16).astype("float32").numpy(),
                    p.astype("float32").numpy(),
                )


@_needs_api
class TestAdamWAndRMSprop(unittest.TestCase):
    def setUp(self):
        paddle.set_device("cpu")

    def _run(self, name, grads, **kw):
        p = _param([6, 4])
        kw.setdefault("weight_decay", 0.0)
        opt = get_inner_optimizer(name)(
            parameters=[p], learning_rate=1e-2, **kw
        )
        _step(opt, [p], grads)
        return p.numpy(), opt

    def test_adamw_descriptors_and_no_decay(self):
        grads = _grads([6, 4])
        ex, opt = self._run(
            "adamw",
            grads,
            weight_decay=0.1,
            apply_decay_param_fun=lambda n: False,
        )
        ref, _ = self._run("adamw", grads)
        np.testing.assert_array_equal(ex, ref)
        self.assertIn("beta1", type(opt).yaml_field_map())
        layout = opt.optimizer_state_layout()
        self.assertEqual(len(layout["vector"]), 2)
        self.assertEqual(len(layout["scalar"]), 2)

    def test_rmsprop_argument_checks(self):
        cls = get_inner_optimizer("rmsprop")
        p = _param([2, 2])
        for bad in ("epsilon", "momentum", "weight_decay", "alpha"):
            kw = {"weight_decay": 0.0, bad: -1.0}
            with self.subTest(arg=bad), self.assertRaises(ValueError):
                cls(parameters=[p], **kw)
        self.assertIn("centered", cls.yaml_field_map())

    def test_maximize_equals_negated_grads(self):
        grads = _grads([6, 4])
        neg = [-g for g in grads]
        for name in ("rmsprop", "rmsproph"):
            with self.subTest(name=name):
                ours, _ = self._run(name, grads, maximize=True, momentum=0.9)
                ref, _ = self._run(name, neg, momentum=0.9)
                np.testing.assert_array_equal(ours, ref)

    def test_rmsprop_coupled_weight_decay(self):
        # Coupled L2: grad += wd * w, so wd > 0 must change the trajectory.
        grads = _grads([6, 4])
        wd, _ = self._run("rmsprop", grads, weight_decay=0.1)
        ref, _ = self._run("rmsprop", grads)
        self.assertFalse(np.array_equal(wd, ref))


if __name__ == "__main__":
    unittest.main()
