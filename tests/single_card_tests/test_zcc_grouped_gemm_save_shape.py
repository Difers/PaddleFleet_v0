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

"""CPU tests for the shapes the ZCC worker writes for grouped-GEMM experts.

A flex_checkpoint descriptor declares the shape of the bytes stored for a key:
`local_shape` when it stores the per-rank N-D block, and a 1-D tensor of
`flattened_range` elements when it stores an element-wise (sharding-v2) slice.
In the latter case `local_shape` describes the parameter the slice came out of,
not the bytes.

The worker reshapes grouped-GEMM expert tensors to the declared `local_shape`
so that fused 2-D buffers agree with an N-D descriptor. Applying that to a
flattened descriptor instead makes the bytes contradict it, and because the
load side assigns the file tensor straight into the optimizer slot
(`paddle.assign` resizes its destination) the slot silently turns N-D and
ShardingV2's merge then slices past its end on the next resume -- a corruption
that only surfaces when the checkpoint is read back.

So: flattened descriptors must keep their 1-D bytes, N-D descriptors must still
be reshaped, and a disagreement must fail the save instead of reaching disk.
The last class closes the loop on the load side, running both byte layouts
through `paddle.assign` and ShardingV2's merge.
"""

import os
import tempfile
import unittest

import paddle
from paddle.distributed.flex_checkpoint.dcp.metadata import (
    LocalTensorMetadata,
    Metadata,
)

from paddlefleet.trainer.utils.reshard.sharding_v2 import merge_tensors
from paddlefleet.trainer.utils.zero_cost_checkpoint import (
    ZeroCostCheckpointWorkerFcBased,
    _zcc_check_shapes_match_meta,
    _zcc_declared_shapes,
    _zcc_declares_flattened,
    _zcc_lookup_local_shape,
)

# One MoE expert weight per flavour, both grouped-GEMM, both 12 elements.
# `FLAT_*` is element-wise sharded (flattened descriptor), `BLOCK_*` is owned
# whole by this rank (N-D descriptor). `PLAIN_W` is not grouped-GEMM at all.
# `SCALAR_S` is a scalar accumulator of the element-wise sharded parameter: it
# has an entry in `param_slice_info` carrying that parameter's element range,
# while its own descriptor and tensor hold a single element.
FLAT_W = "enc.experts.weight1.w_0"
FLAT_M = "enc.experts.weight1.moment1_0"
SCALAR_S = "enc.experts.weight1.beta1_pow_acc_0"
BLOCK_W = "dec.experts.weight1.w_0"
BLOCK_M = "dec.experts.weight1.moment1_0"
PLAIN_W = "enc.norm.w_0"
GROUPED_GEMM = {"enc.experts.weight1", "dec.experts.weight1"}
# The slot holds 16 elements, 12 of which belong to this rank; the worker
# slices the padding off before saving.
PADDED_NUMEL = 16
NUMEL = 12
BLOCK_SHAPE = (6, 2)


def _flat_desc(numel=NUMEL, local_shape=BLOCK_SHAPE):
    """A descriptor storing a 1-D slice of `numel` elements."""
    return LocalTensorMetadata(
        global_offset=(0, 0),
        local_shape=local_shape,
        dtype="float32",
        global_shape=(local_shape[0] * 2, *tuple(local_shape[1:])),
        is_flattened=True,
        flattened_range=(0, numel),
    )


def _block_desc(local_shape=BLOCK_SHAPE):
    """A descriptor storing the per-rank block as-is."""
    return LocalTensorMetadata(
        global_offset=tuple(0 for _ in local_shape),
        local_shape=local_shape,
        dtype="float32",
        global_shape=local_shape,
    )


def _meta(entries):
    meta = Metadata()
    meta.state_dict_metadata = {
        key: value if isinstance(value, list) else [value]
        for key, value in entries.items()
    }
    meta.storage_metadata = {}
    meta.flat_mapping = {}
    return meta


def _worker(**attrs):
    """A worker with only the attributes the `_save_*` methods read.

    `__init__` spawns a process and needs fleet, so bypass it: the save
    methods touch a handful of plain attributes plus two storage helpers.
    """
    worker = ZeroCostCheckpointWorkerFcBased.__new__(
        ZeroCostCheckpointWorkerFcBased
    )
    worker.distcp_file_name = ("0_0.distcp", "0.metadata")
    worker.dp_rank = 0
    worker.device_id = 0
    worker.use_expert_parallel = False
    worker.grouped_gemm_params = set(GROUPED_GEMM)
    worker.param_slice_info = {}
    worker.unified_name_mapping = {}
    worker.ema_coef = None
    for name, value in attrs.items():
        setattr(worker, name, value)
    return worker


class _FakeStorageHelper:
    """Stands in for the fusion storage helper and the EMA processor, which
    only have to hand the worker the tensors it is about to save."""

    def __init__(self, state_dict):
        self._state_dict = state_dict

    def state_dict(self):
        return dict(self._state_dict)

    def ema_state_dict(self):
        return dict(self._state_dict)


def _saved(path):
    return {k: tuple(v.shape) for k, v in paddle.load(path).items()}


class TestDescriptorQueries(unittest.TestCase):
    """`_zcc_declares_flattened` / `_zcc_declared_shapes` answer which flavour
    a key is stored in and what shape its bytes must have."""

    def test_flattened_descriptor_is_recognised(self):
        meta = _meta({FLAT_W: _flat_desc()})
        self.assertTrue(_zcc_declares_flattened(meta, FLAT_W))

    def test_block_descriptor_is_not_flattened(self):
        meta = _meta({BLOCK_W: _block_desc()})
        self.assertFalse(_zcc_declares_flattened(meta, BLOCK_W))

    def test_scalar_descriptor_is_not_flattened(self):
        meta = _meta({SCALAR_S: _block_desc((1,))})
        self.assertFalse(_zcc_declares_flattened(meta, SCALAR_S))

    def test_unknown_key_is_not_flattened(self):
        meta = _meta({BLOCK_W: _block_desc()})
        self.assertFalse(_zcc_declares_flattened(meta, "absent"))

    def test_missing_metadata_is_not_flattened(self):
        # `_merge_ema_ckpt_meta` returns None when no EMA metadata exists.
        self.assertFalse(_zcc_declares_flattened(None, FLAT_W))

    def test_local_slice_narrows_a_flattened_declaration(self):
        # The declared shape is this rank's element count, *not* `local_shape`,
        # even though `local_shape` here is 2-D with the same count.
        meta = _meta({FLAT_W: _flat_desc()})
        slice_info = {FLAT_W: slice(0, NUMEL)}
        self.assertEqual(
            _zcc_declared_shapes(meta, FLAT_W, slice_info), [(NUMEL,)]
        )

    def test_local_slice_wins_over_other_ranks_lengths(self):
        # Slice lengths legitimately differ between ranks, so the all-rank
        # gather must not be allowed to answer for this one.
        meta = _meta({FLAT_W: [_flat_desc(NUMEL), _flat_desc(NUMEL - 4)]})
        slice_info = {FLAT_W: slice(0, NUMEL - 4)}
        self.assertEqual(
            _zcc_declared_shapes(meta, FLAT_W, slice_info), [(NUMEL - 4,)]
        )

    def test_local_slice_does_not_answer_for_a_scalar(self):
        # `param_slice_info` holds an entry per scalar accumulator too,
        # carrying the element range of the *parameter*. Letting it answer
        # would declare `(NUMEL,)` for a 1-element tensor and fail every save.
        meta = _meta({SCALAR_S: _block_desc((1,))})
        slice_info = {SCALAR_S: slice(0, NUMEL)}
        self.assertEqual(
            _zcc_declared_shapes(meta, SCALAR_S, slice_info), [(1,)]
        )

    def test_uncorroborated_local_slice_leaves_the_descriptors_alone(self):
        # Narrowing only; a length no descriptor records keeps the cross-check
        # between the metadata and the bytes rather than replacing it.
        meta = _meta({FLAT_W: _flat_desc(NUMEL)})
        slice_info = {FLAT_W: slice(0, NUMEL - 4)}
        self.assertEqual(
            _zcc_declared_shapes(meta, FLAT_W, slice_info), [(NUMEL,)]
        )

    def test_block_declares_its_local_shape(self):
        meta = _meta({BLOCK_W: _block_desc()})
        self.assertEqual(_zcc_declared_shapes(meta, BLOCK_W), [BLOCK_SHAPE])

    def test_every_recorded_block_shard_is_declared(self):
        # No rank-local descriptor exists for an N-D block on the worker side,
        # so the gather answers; in practice all shards agree.
        meta = _meta({BLOCK_W: [_block_desc((6, 2)), _block_desc((3, 4))]})
        self.assertEqual(_zcc_declared_shapes(meta, BLOCK_W), [(6, 2), (3, 4)])

    def test_flattened_descriptor_without_local_slice_falls_back(self):
        meta = _meta({FLAT_W: _flat_desc()})
        self.assertEqual(_zcc_declared_shapes(meta, FLAT_W), [(NUMEL,)])

    def test_no_descriptor_declares_nothing(self):
        self.assertEqual(_zcc_declared_shapes(_meta({}), FLAT_W), [])
        self.assertEqual(_zcc_declared_shapes(None, FLAT_W), [])

    def test_lookup_local_shape_still_matches_by_numel(self):
        meta = _meta({BLOCK_W: _block_desc()})
        self.assertEqual(
            _zcc_lookup_local_shape(meta, BLOCK_W, NUMEL), list(BLOCK_SHAPE)
        )

    def test_lookup_local_shape_rejects_other_numels(self):
        meta = _meta({BLOCK_W: _block_desc()})
        self.assertIsNone(_zcc_lookup_local_shape(meta, BLOCK_W, NUMEL + 1))
        self.assertIsNone(_zcc_lookup_local_shape(None, BLOCK_W, NUMEL))


class TestCheckShapesMatchMeta(unittest.TestCase):
    """The pre-save check compares the bytes against their descriptor."""

    def test_accepts_bytes_matching_either_flavour(self):
        meta = _meta({FLAT_W: _flat_desc(), BLOCK_W: _block_desc()})
        state_dict = {
            FLAT_W: paddle.zeros([NUMEL]),
            BLOCK_W: paddle.zeros(list(BLOCK_SHAPE)),
        }
        _zcc_check_shapes_match_meta(
            state_dict, meta, "master_weight", {FLAT_W: slice(0, NUMEL)}
        )

    def test_rejects_flattened_bytes_folded_to_2d(self):
        # Exactly the regression: 1-D slice reshaped to `local_shape`.
        meta = _meta({FLAT_W: _flat_desc()})
        state_dict = {FLAT_W: paddle.zeros(list(BLOCK_SHAPE))}
        with self.assertRaises(RuntimeError) as caught:
            _zcc_check_shapes_match_meta(
                state_dict, meta, "master_weight", {FLAT_W: slice(0, NUMEL)}
            )
        message = str(caught.exception)
        self.assertIn(FLAT_W, message)
        self.assertIn("master_weight", message)
        self.assertIn(str(BLOCK_SHAPE), message)
        self.assertIn(f"({NUMEL},)", message)

    def test_rejects_another_ranks_slice_length(self):
        meta = _meta({FLAT_W: [_flat_desc(NUMEL), _flat_desc(NUMEL - 4)]})
        with self.assertRaises(RuntimeError) as caught:
            _zcc_check_shapes_match_meta(
                {FLAT_W: paddle.zeros([NUMEL])},
                meta,
                "master_weight",
                {FLAT_W: slice(0, NUMEL - 4)},
            )
        self.assertIn(FLAT_W, str(caught.exception))

    def test_accepts_any_recorded_block_shard_shape(self):
        # For an N-D block only the key is known, not which rank's shard this
        # is, so a match against any recorded shard counts.
        meta = _meta({BLOCK_W: [_block_desc((6, 2)), _block_desc((3, 4))]})
        _zcc_check_shapes_match_meta(
            {BLOCK_W: paddle.zeros([3, 4])}, meta, "master_weight"
        )

    def test_skips_keys_without_a_descriptor(self):
        # `_save_ema_state` saves tensors FC metadata does not cover.
        _zcc_check_shapes_match_meta(
            {"unmapped": paddle.zeros([3])}, _meta({}), "ema_state"
        )

    def test_tolerates_missing_metadata(self):
        _zcc_check_shapes_match_meta(
            {FLAT_W: paddle.zeros([NUMEL])}, None, "ema_state"
        )

    def test_accepts_0d_scalar_against_a_0d_descriptor(self):
        # 0-D buffers (e.g. the quantile-balancing router's bin bounds) have an
        # empty shape on both sides and must not be read as a mismatch.
        meta = _meta({"gate.qb_bin_min": _block_desc(())})
        _zcc_check_shapes_match_meta(
            {"gate.qb_bin_min": paddle.zeros([])}, meta, "model_state"
        )

    def test_accepts_a_scalar_accumulator_of_a_sharded_param(self):
        # `param_slice_info` carries the parameter's range for the scalar too,
        # so letting it declare would reject every `(1,)` accumulator and fail
        # the save of any sharding-v2 job.
        meta = _meta({SCALAR_S: _block_desc((1,))})
        _zcc_check_shapes_match_meta(
            {SCALAR_S: paddle.zeros([1])},
            meta,
            "optimizer_state",
            {SCALAR_S: slice(0, NUMEL)},
        )

    def test_reports_every_offender_at_once(self):
        meta = _meta({FLAT_W: _flat_desc(), FLAT_M: _flat_desc()})
        state_dict = {
            FLAT_W: paddle.zeros(list(BLOCK_SHAPE)),
            FLAT_M: paddle.zeros(list(BLOCK_SHAPE)),
        }
        with self.assertRaises(RuntimeError) as caught:
            _zcc_check_shapes_match_meta(state_dict, meta, "optimizer_state")
        self.assertIn("2 tensor(s)", str(caught.exception))


def _opt_worker(**overrides):
    attrs = {
        "optimizer_fusion_storage_helper": _FakeStorageHelper(
            {
                "master_weights": {
                    FLAT_W: paddle.zeros([PADDED_NUMEL]),
                    BLOCK_W: paddle.zeros([2, 3, 2]),
                    PLAIN_W: paddle.zeros([4]),
                },
                FLAT_M: paddle.zeros([PADDED_NUMEL]),
                BLOCK_M: paddle.zeros([2, 3, 2]),
                SCALAR_S: paddle.zeros([1]),
            }
        ),
        "unified_name_mapping": {
            key: key
            for key in (
                FLAT_W,
                BLOCK_W,
                PLAIN_W,
                FLAT_M,
                BLOCK_M,
                SCALAR_S,
            )
        },
        "param_slice_info": {
            FLAT_W: slice(0, NUMEL),
            FLAT_M: slice(0, NUMEL),
            # `_gen_unified_name` records the *parameter's* range here.
            SCALAR_S: slice(0, NUMEL),
        },
        "opt_state_filter": {
            FLAT_M: False,
            BLOCK_M: False,
            SCALAR_S: False,
        },
        "master_weights_filter": {
            FLAT_W: False,
            BLOCK_W: False,
            PLAIN_W: False,
        },
        "opt_ckpt_meta": _meta(
            {
                FLAT_M: _flat_desc(),
                BLOCK_M: _block_desc(),
                SCALAR_S: _block_desc((1,)),
            }
        ),
        "master_weight_ckpt_meta": _meta(
            {
                FLAT_W: _flat_desc(),
                BLOCK_W: _block_desc(),
                PLAIN_W: _block_desc((4,)),
            }
        ),
    }
    attrs.update(overrides)
    return _worker(**attrs)


class TestSaveOptState(unittest.TestCase):
    """`_save_opt_state` reshapes grouped-GEMM experts only where the
    descriptor stores an N-D block."""

    def test_flattened_experts_keep_1d_bytes(self):
        worker = _opt_worker()
        with tempfile.TemporaryDirectory() as output_dir:
            for sub in ("optimizer_state", "master_weight"):
                os.makedirs(os.path.join(output_dir, sub))
            worker._save_opt_state(output_dir)

            self.assertEqual(
                _saved(worker.opt_state_path),
                {
                    FLAT_M: (NUMEL,),
                    BLOCK_M: BLOCK_SHAPE,
                    # Scalar accumulators ride along untouched: their entry in
                    # `param_slice_info` must not be read as a declaration.
                    SCALAR_S: (1,),
                },
            )
            self.assertEqual(
                _saved(worker.master_weight_path),
                {
                    FLAT_W: (NUMEL,),
                    BLOCK_W: BLOCK_SHAPE,
                    PLAIN_W: (4,),
                },
            )

    def test_bytes_contradicting_the_descriptor_fail_the_save(self):
        # PLAIN_W is not a grouped-GEMM expert, so nothing reshapes it and the
        # mismatch below stands in for any future break of the invariant.
        worker = _opt_worker(
            master_weight_ckpt_meta=_meta(
                {
                    FLAT_W: _flat_desc(),
                    BLOCK_W: _block_desc(),
                    PLAIN_W: _block_desc((2, 2)),
                }
            )
        )
        with tempfile.TemporaryDirectory() as output_dir:
            for sub in ("optimizer_state", "master_weight"):
                os.makedirs(os.path.join(output_dir, sub))
            with self.assertRaises(RuntimeError) as caught:
                worker._save_opt_state(output_dir)
            self.assertIn(PLAIN_W, str(caught.exception))
            self.assertFalse(os.path.exists(worker.master_weight_path))


MODEL_P = "enc.gate.weight"
UNMAPPED_P = "enc.routed_scaling_factor_param"


class TestSaveEmaState(unittest.TestCase):
    """`_save_ema_state` applies the same rule to the EMA copy."""

    def test_flattened_experts_keep_1d_bytes(self):
        worker = _worker(
            ema_coef=0.9,
            zcc_ema_processor=_FakeStorageHelper(
                {
                    "master_weights": {
                        FLAT_W: paddle.zeros([PADDED_NUMEL]),
                        BLOCK_W: paddle.zeros([2, 3, 2]),
                    },
                    MODEL_P: paddle.zeros([3]),
                    UNMAPPED_P: paddle.zeros([2]),
                }
            ),
            unified_name_mapping={
                key: key for key in (FLAT_W, BLOCK_W, MODEL_P)
            },
            param_slice_info={FLAT_W: slice(0, NUMEL)},
            ema_master_weights_filter={FLAT_W: False, BLOCK_W: False},
            ema_model_state_filter={MODEL_P: False},
            ema_master_weight_ckpt_meta=_meta(
                {FLAT_W: _flat_desc(), BLOCK_W: _block_desc()}
            ),
            ema_model_params_ckpt_meta=_meta({MODEL_P: _block_desc((3,))}),
        )
        with tempfile.TemporaryDirectory() as output_dir:
            os.makedirs(os.path.join(output_dir, "ema_state"))
            worker._save_ema_state(output_dir)

            self.assertEqual(
                _saved(worker.ema_name_path),
                {
                    FLAT_W: (NUMEL,),
                    BLOCK_W: BLOCK_SHAPE,
                    MODEL_P: (3,),
                    # No descriptor covers this one; it is saved as-is.
                    UNMAPPED_P: (2,),
                },
            )


class TestSaveModelState(unittest.TestCase):
    """Model parameters are never reshaped, but are checked all the same."""

    def _worker_for(self, declared):
        return _worker(
            param_fusion_storage_helper=_FakeStorageHelper(
                {MODEL_P: paddle.zeros([2, 2])}
            ),
            model_state_filter={MODEL_P: False},
            model_ckpt_meta=_meta({MODEL_P: _block_desc(declared)}),
        )

    def test_matching_bytes_are_saved(self):
        worker = self._worker_for((2, 2))
        with tempfile.TemporaryDirectory() as output_dir:
            os.makedirs(os.path.join(output_dir, "model_state"))
            worker._save_model_state(output_dir)
            self.assertEqual(
                _saved(worker.model_states_path), {MODEL_P: (2, 2)}
            )

    def test_bytes_contradicting_the_descriptor_fail_the_save(self):
        worker = self._worker_for((4,))
        with tempfile.TemporaryDirectory() as output_dir:
            os.makedirs(os.path.join(output_dir, "model_state"))
            with self.assertRaises(RuntimeError) as caught:
                worker._save_model_state(output_dir)
            self.assertIn(MODEL_P, str(caught.exception))
            self.assertFalse(os.path.exists(worker.model_states_path))


class TestRecoveryFromSavedBytes(unittest.TestCase):
    """What the saved shape costs on the way back in.

    The resume path is `local_load_state_dict` assigning the file tensor into
    the optimizer's master weight slot, then ShardingV2 merging the slot back
    into the parameter shape. Both ends run here on the real production
    functions, over the two byte layouts the save side can produce.
    """

    PARAM_SHAPE = [2, 3, 2]

    def _assign_into_slot(self, file_tensor):
        """The one line `local_load_state_dict` runs per key."""
        slot = paddle.zeros([NUMEL], dtype="float32")
        paddle.assign(file_tensor, slot)
        return slot

    def test_1d_bytes_merge_back_into_the_parameter_shape(self):
        values = paddle.arange(NUMEL).astype("float32")
        slot = self._assign_into_slot(values)
        self.assertEqual(tuple(slot.shape), (NUMEL,))

        merged = merge_tensors(FLAT_W, [slot], self.PARAM_SHAPE)
        self.assertEqual(tuple(merged.shape), tuple(self.PARAM_SHAPE))
        self.assertTrue(bool(paddle.equal_all(merged.reshape([NUMEL]), values)))

    def test_2d_bytes_turn_the_slot_nd_and_break_the_merge(self):
        # `paddle.assign` resizes its destination, so the slot silently stops
        # being the 1-D slice ShardingV2 expects.
        slot = self._assign_into_slot(paddle.zeros(list(BLOCK_SHAPE)))
        self.assertEqual(tuple(slot.shape), BLOCK_SHAPE)

        with self.assertRaises(IndexError):
            merge_tensors(FLAT_W, [slot], self.PARAM_SHAPE)


if __name__ == "__main__":
    unittest.main()
