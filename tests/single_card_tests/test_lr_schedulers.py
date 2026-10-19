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
"""``paddlefleet.lr_schedulers``: per-group schedules and the chained wrapper.

A mixed optimizer runs one schedule per parameter group, so the trainer holds a
``ChainedLRScheduler`` that fans its single ``.step()`` out to the per-group
schedulers it shares with the sub-optimizers. These tests check the spec parser
(what a yaml group may write), the builder, and that chaining neither
double-advances nor loses state across a checkpoint round-trip.

Run:  python -m pytest tests/single_card_tests/test_lr_schedulers.py -v
"""

import unittest

from paddlefleet.lr_schedulers import (
    LR_SCHEDULER_REGISTRY,
    BaseLRScheduler,
    ChainedLRScheduler,
    PowerLR,
    build_group_scheduler,
    get_cosine_schedule_with_warmup,
    get_power_schedule_with_warmup,
    get_wsd_schedule_with_warmup,
    parse_lr_scheduler_spec,
    register_lr_scheduler,
)
from paddlefleet.lr_schedulers.base import parse_finite_float

LR = 1e-3
WARMUP = 2
MAX_STEPS = 10


def _build(spec, learning_rate=LR):
    return build_group_scheduler(
        lr_scheduler=spec,
        learning_rate=learning_rate,
        warmup_steps=WARMUP,
        max_steps=MAX_STEPS,
        min_lr=0.0,
    )


class TestParseSpec(unittest.TestCase):
    """The spec string a yaml group may write"""

    def test_accepted(self):
        self.assertEqual(parse_lr_scheduler_spec("cosine"), ("cosine", {}))
        # Unset params are omitted: the ctor default applies.
        self.assertEqual(parse_lr_scheduler_spec("wsd"), ("wsd", {}))
        self.assertEqual(
            parse_lr_scheduler_spec("wsd:100"),
            ("wsd", {"num_steady_steps": 100}),
        )
        self.assertEqual(parse_lr_scheduler_spec("power"), ("power", {}))
        self.assertEqual(
            parse_lr_scheduler_spec("power:0.5"), ("power", {"power": 0.5})
        )
        self.assertEqual(
            parse_lr_scheduler_spec("power:0.4:100"),
            ("power", {"power": 0.4, "num_stable_steps": 100}),
        )
        # A declared param may also come as a group key (same parser).
        self.assertEqual(
            parse_lr_scheduler_spec(
                "wsd", num_steady_steps=7, decay_function="1-sqrt"
            ),
            ("wsd", {"num_steady_steps": 7, "decay_function": "1-sqrt"}),
        )
        self.assertEqual(
            parse_lr_scheduler_spec("power", num_stable_steps=3),
            ("power", {"num_stable_steps": 3}),
        )

    def test_rejected(self):
        for spec in (
            "nope",  # unknown name
            "cosine:1",  # takes no field
            "wsd:-1",  # negative steady steps
            "wsd:0.5",  # steady steps must be an int
            "wsd:1:2",  # one field only
            "power:nan",  # non-finite exponent
            "power:0.4:1:2",  # two fields only
        ):
            with self.assertRaises(ValueError, msg=spec):
                parse_lr_scheduler_spec(spec)

    def test_group_keys_rejected(self):
        for spec, kwargs in (
            ("wsd:5", {"num_steady_steps": 5}),  # both spec and key
            ("wsd", {"decay_function": "linear"}),  # not a choice
            ("wsd", {"num_steady_steps": 0.5}),  # float is not truncated
            ("wsd", {"num_steady_steps": True}),  # nor is a bool
            ("cosine", {"num_cycles": 1.0}),  # not declared
            ("power", {"decay_function": "1-sqrt"}),  # another class's param
        ):
            with self.assertRaises(ValueError, msg=f"{spec} {kwargs}"):
                parse_lr_scheduler_spec(spec, **kwargs)


class TestBuildGroupScheduler(unittest.TestCase):
    """Every accepted spec builds a scheduler that warms up to the peak"""

    def test_warmup_reaches_peak(self):
        # The exact step at which the peak lands differs per kind (cosine/wsd
        # hit it one step after power does), so only require that the run
        # starts at the floor and touches the peak during warmup.
        for spec in ("cosine", "wsd", "wsd:3", "power", "power:0.4:3"):
            sched = _build(spec)
            self.assertAlmostEqual(sched.get_lr(), 0.0, places=8, msg=spec)
            seen = []
            for _ in range(WARMUP + 1):
                sched.step()
                seen.append(sched.get_lr())
            self.assertAlmostEqual(max(seen), LR, places=8, msg=spec)

    def test_decays_after_warmup(self):
        for spec in ("cosine", "wsd", "power:0.4"):
            sched = _build(spec)
            for _ in range(MAX_STEPS - 1):
                sched.step()
            self.assertLess(sched.get_lr(), LR, msg=spec)

    def test_group_kwargs_follow_get_param(self):
        # wsd declares decay_function; cosine/power do not.
        sched = build_group_scheduler(
            lr_scheduler="wsd",
            learning_rate=LR,
            warmup_steps=WARMUP,
            max_steps=MAX_STEPS,
            decay_function="1-sqrt",
        )
        self.assertEqual(sched.decay_function, "1-sqrt")
        self.assertEqual(_build("wsd").decay_function, "half_life")
        for spec in ("cosine", "power"):
            with self.assertRaises(ValueError, msg=spec):
                build_group_scheduler(
                    lr_scheduler=spec,
                    learning_rate=LR,
                    warmup_steps=WARMUP,
                    max_steps=MAX_STEPS,
                    decay_function="1-sqrt",
                )


class TestPowerSchedule(unittest.TestCase):
    def test_pointwise(self):
        # warmup 3 (0.1 -> 1), hold through step 4, decay over 6 steps to 0.1.
        sched = get_power_schedule_with_warmup(
            1.0, 3, 10, min_lr=0.1, power=0.5, num_stable_steps=2
        )
        decay = [0.1 + 0.9 * (1 - (k / 6) ** 0.5) for k in range(1, 7)]
        want = [0.1, 0.55, 1.0, 1.0, 1.0, *decay, 0.1]
        got = []
        for _ in want:
            got.append(sched.get_lr())
            sched.step()
        for step, (g, w) in enumerate(zip(got, want, strict=True)):
            self.assertAlmostEqual(g, w, places=12, msg=f"step {step}")


class TestChainedLRScheduler(unittest.TestCase):
    """One .step() drives N group schedulers, exactly once each"""

    def setUp(self):
        self.specs = ["cosine", "wsd", "power:0.4"]
        self.lrs = [LR, LR / 2, LR / 4]
        self.subs = [
            _build(s, lr) for s, lr in zip(self.specs, self.lrs, strict=True)
        ]
        self.chained = ChainedLRScheduler(self.subs)

    def test_steps_every_sub_once(self):
        # Standalone copies stepped the same number of times are the reference:
        # the chained wrapper must not double-advance (it skips super().step()).
        refs = [
            _build(s, lr) for s, lr in zip(self.specs, self.lrs, strict=True)
        ]
        for _ in range(WARMUP + 3):
            self.chained.step()
            for r in refs:
                r.step()
        for sub, ref, spec in zip(self.subs, refs, self.specs, strict=True):
            self.assertEqual(sub.last_epoch, ref.last_epoch, msg=spec)
            self.assertAlmostEqual(sub.get_lr(), ref.get_lr(), 10, msg=spec)
        # Each group keeps its own peak, so the lrs stay distinct.
        self.assertEqual(
            len({round(s.get_lr(), 10) for s in self.subs}), len(self.subs)
        )
        # The chain mirrors group 0 for logging.
        rep = self.subs[0]
        self.assertAlmostEqual(self.chained.get_lr(), rep.get_lr(), places=10)
        self.assertAlmostEqual(self.chained(), rep(), places=10)
        self.assertEqual(self.chained.last_epoch, rep.last_epoch)

    def test_state_dict_round_trip(self):
        for _ in range(WARMUP + 3):
            self.chained.step()
        sd = self.chained.state_dict()
        self.assertEqual(sorted(sd), ["0", "1", "2"])
        expected = [s.get_lr() for s in self.subs]
        # Restore into a fresh (un-stepped) chain: it must resume mid-schedule.
        fresh_subs = [
            _build(s, lr) for s, lr in zip(self.specs, self.lrs, strict=True)
        ]
        fresh = ChainedLRScheduler(fresh_subs)
        fresh.set_state_dict(sd)
        for sub, want, spec in zip(
            fresh_subs, expected, self.specs, strict=True
        ):
            self.assertAlmostEqual(sub.get_lr(), want, places=10, msg=spec)
        self.assertEqual(fresh.last_epoch, self.chained.last_epoch)

    def test_set_state_dict_rejects_group_mismatch(self):
        sd = self.chained.state_dict()
        del sd["2"]
        with self.assertRaises(ValueError):
            self.chained.set_state_dict(sd)

    def test_set_dict_is_chained_restore(self):
        # The base LRScheduler binds set_dict to ITS set_state_dict; without
        # the alias every set_dict call would bypass the per-group restore.
        self.assertIs(
            ChainedLRScheduler.set_dict, ChainedLRScheduler.set_state_dict
        )


class _ConstLR(BaseLRScheduler):
    """``const:<scale>``: no warmup, ``scale`` x peak throughout."""

    def __init__(self, *args, scale=1.0, **kwargs):
        self.scale = scale
        super().__init__(*args, **kwargs)

    def schedule(self, step):
        return self.scale

    @classmethod
    def get_param(cls):
        return {"scale": parse_finite_float}


class TestCustomScheduler(unittest.TestCase):
    """A registered BaseLRScheduler subclass is usable from a yaml spec"""

    def setUp(self):
        register_lr_scheduler("const", _ConstLR)

    def tearDown(self):
        LR_SCHEDULER_REGISTRY.pop("const", None)

    def test_end_to_end(self):
        self.assertEqual(
            parse_lr_scheduler_spec("const:0.5"), ("const", {"scale": 0.5})
        )
        chained = ChainedLRScheduler([_build("const:0.5"), _build("cosine")])
        for _ in range(WARMUP + 1):
            chained.step()
        self.assertAlmostEqual(chained.get_lr(), LR * 0.5, places=12)

    def test_missing_abstract_methods_cannot_instantiate(self):
        class NoSchedule(BaseLRScheduler):
            @classmethod
            def get_param(cls):
                return {}

        class NoGetParam(BaseLRScheduler):
            def schedule(self, step):
                return 1.0

        for bad in (NoSchedule, NoGetParam):
            with self.assertRaises(TypeError, msg=bad.__name__):
                bad(LR, WARMUP, MAX_STEPS)

    def test_register_rejects_shadowing_and_non_subclass(self):
        with self.assertRaises(ValueError):
            register_lr_scheduler("cosine", _ConstLR)
        with self.assertRaises(TypeError):
            register_lr_scheduler("plain", object)


class TestSchedulerEdges(unittest.TestCase):
    def test_parse_finite_float_rejects_non_number(self):
        with self.assertRaises(ValueError):
            parse_finite_float("abc", "cosine", "min_lr")

    def test_register_rejects_bad_name(self):
        for name in ("", "a:b", 3):
            with self.subTest(name=name), self.assertRaises(ValueError):
                register_lr_scheduler(name, BaseLRScheduler)

    def test_power_must_be_positive(self):
        with self.assertRaises(ValueError):
            PowerLR(LR, WARMUP, MAX_STEPS, power=0.0)

    def test_cosine_factory(self):
        sched = get_cosine_schedule_with_warmup(LR, WARMUP, MAX_STEPS)
        self.assertEqual(sched.get_lr(), 0.0)
        self.assertEqual(sched.schedule(WARMUP), 1.0)

    def test_wsd_factory_and_decay_functions(self):
        steady = 6
        half = get_wsd_schedule_with_warmup(
            LR, WARMUP, MAX_STEPS, num_steady_steps=steady
        )
        sqrt = get_wsd_schedule_with_warmup(
            LR,
            WARMUP,
            MAX_STEPS,
            decay_function="1-sqrt",
            num_steady_steps=steady,
        )
        self.assertEqual(sqrt.schedule(steady), 1.0)
        self.assertAlmostEqual(sqrt.schedule(MAX_STEPS), 0.0)
        # progress = 1/4 -> 1 - sqrt(1/4) = 0.5; half_life is a different curve.
        self.assertAlmostEqual(sqrt.schedule(steady + 1), 0.5)
        self.assertNotAlmostEqual(half.schedule(steady + 1), 0.5)
        sqrt.decay_function = "nope"
        with self.assertRaises(ValueError):
            sqrt.schedule(MAX_STEPS)


if __name__ == "__main__":
    unittest.main()
