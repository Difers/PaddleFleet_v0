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

"""``RotaryEmbedding.get_rotary_seq_len`` under dynamic context parallel (DCP).

The rotary table must be built for the *global* sequence length, so the local
length has to be scaled by the CP size that is in effect **for this
microbatch**. ``self.cp_group`` is captured when the layer is built and keeps
reporting the static degree, which is why the getter now prefers
``parallel_state.get_current_context_parallel_size()``. MLA validates
``rotary_seq_len == cp_size * local_seq_len`` against the current size (the
CP RoPE check in ``MLASelfAttention.get_query_key_value_tensors``), so a
static cp=4 against a cp=2 microbatch aborts the step.

Precedence under test (highest first):
  1. ``packed_seq_params`` -- returns ``max(max_seqlen_q, max_seqlen_kv)`` and
     never scales.
  2. the DCP size selected for the current microbatch,
  3. the layer's own ``cp_group`` (so DCP-off behaviour, including every
     existing single-card test that passes ``cp_group`` explicitly, is
     unchanged),
  4. ``parallel_state.get_context_parallel_world_size()`` as a last resort.

The numerical test derives the expected frequency table from ``math``/``numpy``
(``outer(positions, base ** (-2i/dim))``) instead of from the function under
test, and checks that the table produced for a cp=2 microbatch covers the full
16 global positions -- the rows a cp=2 rank actually needs -- while a stale cp=4
would have produced 32.
"""

import os
import sys
import unittest

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

try:
    import numpy as np
    import paddle

    from paddlefleet import parallel_state as ps
    from paddlefleet.models.common.embeddings.rotary_pos_embedding import (
        RotaryEmbedding,
    )

    _IMPORT_ERROR = None
except ImportError as exc:  # genuine missing dependency only
    ps = RotaryEmbedding = None
    _IMPORT_ERROR = exc

SKIP_REASON = f"paddle/paddlefleet unavailable: {_IMPORT_ERROR}"

HEAD_DIM = 64


class _Group:
    def __init__(self, nranks, rank=0):
        self.nranks = nranks
        self.world_size = nranks
        self.rank = rank


class _Config:
    """Only the ``get_rotary_seq_len`` fields of TransformerConfig."""

    def __init__(self, sequence_parallel=False, tensor_model_parallel_size=1):
        self.sequence_parallel = sequence_parallel
        self.tensor_model_parallel_size = tensor_model_parallel_size


class _PackedSeqParams:
    def __init__(self, max_seqlen_q, max_seqlen_kv):
        self.max_seqlen_q = max_seqlen_q
        self.max_seqlen_kv = max_seqlen_kv


def ref_inv_freq(dim, rotary_base=10000.0):
    """Independent reference: inv_freq[i] = base ** (-(2i)/dim)."""
    return np.array(
        [1.0 / (rotary_base ** ((2 * i) / dim)) for i in range(dim // 2)]
    )


@unittest.skipUnless(_IMPORT_ERROR is None, SKIP_REASON)
class _RotaryDcpTestBase(unittest.TestCase):
    _GLOBALS = (
        "_DYNAMIC_CONTEXT_PARALLEL_GROUPS",
        "_CURRENT_CONTEXT_PARALLEL_SIZE",
        "_MAX_DYNAMIC_CONTEXT_PARALLEL_SIZE",
        "_CONTEXT_PARALLEL_GROUP",
    )

    def setUp(self):
        self._saved = {name: getattr(ps, name) for name in self._GLOBALS}
        self.addCleanup(self._restore)
        ps._DYNAMIC_CONTEXT_PARALLEL_GROUPS = {}
        ps._CURRENT_CONTEXT_PARALLEL_SIZE = None
        ps._CONTEXT_PARALLEL_GROUP = None

    def _restore(self):
        for name, value in self._saved.items():
            setattr(ps, name, value)

    def select_cp_size(self, size, table=None):
        """Pretend the DCP scheduler picked ``size`` for this microbatch."""
        table = table or {1: None, 2: _Group(2, rank=1), 4: _Group(4, rank=3)}
        ps._DYNAMIC_CONTEXT_PARALLEL_GROUPS = dict(table)
        ps._MAX_DYNAMIC_CONTEXT_PARALLEL_SIZE = max(table)
        ps.set_current_context_parallel_size(size)


class TestGetRotarySeqLenUnderDcp(_RotaryDcpTestBase):
    LOCAL_SEQ = 8

    def _emb(self, cp_group):
        return RotaryEmbedding(
            head_dim=HEAD_DIM, rotary_percent=1.0, cp_group=cp_group
        )

    def _input(self):
        # Not sequence-parallel -> sequence axis is 1.
        return paddle.zeros([2, self.LOCAL_SEQ, HEAD_DIM])

    def test_frozen_group_still_wins_while_dcp_is_off(self):
        # Unchanged behaviour is the contract for every DCP-off run.
        emb = self._emb(_Group(4, rank=3))
        self.assertEqual(
            emb.get_rotary_seq_len(self._input(), _Config()), 8 * 4
        )

    def test_current_size_overrides_the_frozen_group(self):
        emb = self._emb(_Group(4, rank=3))  # built when CP was 4
        self.select_cp_size(2)
        # 8 * 2 = 16, NOT 8 * 4 = 32: the microbatch only spans two ranks.
        self.assertEqual(emb.get_rotary_seq_len(self._input(), _Config()), 16)

    def test_current_size_one_means_no_scaling(self):
        emb = self._emb(_Group(4, rank=3))
        self.select_cp_size(1)
        self.assertEqual(emb.get_rotary_seq_len(self._input(), _Config()), 8)

    def test_falls_back_to_parallel_state_without_a_layer_group(self):
        # Layers that were built without a cp_group (and DCP off) must still
        # see the run's CP degree -- this is the last-resort branch.
        emb = self._emb(None)
        ps._CONTEXT_PARALLEL_GROUP = _Group(2, rank=0)
        self.assertEqual(emb.get_rotary_seq_len(self._input(), _Config()), 16)

    def test_no_cp_at_all_leaves_the_local_length_alone(self):
        emb = self._emb(None)
        self.assertEqual(emb.get_rotary_seq_len(self._input(), _Config()), 8)

    def test_sequence_parallel_multiplies_tp_and_current_cp(self):
        emb = self._emb(_Group(4, rank=3))
        self.select_cp_size(2)
        # Sequence-parallel -> axis 0; 8 * tp(2) * cp(2) = 32.
        x = paddle.zeros([self.LOCAL_SEQ, 2, HEAD_DIM])
        cfg = _Config(sequence_parallel=True, tensor_model_parallel_size=2)
        self.assertEqual(emb.get_rotary_seq_len(x, cfg), 32)

    def test_packed_seq_params_short_circuit_before_any_cp_scaling(self):
        emb = self._emb(_Group(4, rank=3))
        self.select_cp_size(2)
        packed = _PackedSeqParams(max_seqlen_q=100, max_seqlen_kv=250)
        self.assertEqual(
            emb.get_rotary_seq_len(
                self._input(), _Config(), packed_seq_params=packed
            ),
            250,
        )


class TestRotaryTableValuesForCurrentCpSize(_RotaryDcpTestBase):
    """Numerical check: the table really covers the global positions of the
    microbatch, and its values match an independently derived RoPE table."""

    LOCAL_SEQ = 8

    def test_table_matches_hand_derived_global_frequencies(self):
        emb = RotaryEmbedding(
            head_dim=HEAD_DIM,
            rotary_percent=1.0,
            cp_group=_Group(4, rank=3),  # stale: built when CP was 4
        )
        x = paddle.zeros([2, self.LOCAL_SEQ, HEAD_DIM])
        self.select_cp_size(2)

        seq_len = emb.get_rotary_seq_len(x, _Config())
        self.assertEqual(seq_len, 16)

        emb_table = emb(seq_len).numpy()
        self.assertEqual(list(emb_table.shape), [1, 16, 1, HEAD_DIM])

        inv = ref_inv_freq(HEAD_DIM)
        freqs = np.outer(np.arange(16, dtype="float64"), inv)
        half = HEAD_DIM // 2
        block = emb_table[0, :, 0, :]
        # Default (non-interleaved) layout duplicates the half-table.
        np.testing.assert_allclose(block[:, :half], freqs, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(block[:, half:], freqs, rtol=1e-5, atol=1e-6)

        # The rows this rank consumes (cp_rank 1 of 2 -> global [8, 16)) are
        # the second half of the table and are not position-0-shifted copies of
        # the first half; a locally-sized table would have none of them.
        self.assertFalse(np.allclose(block[:8, :half], block[8:, :half]))

        # A stale cp=4 would have built twice the table and shifted every
        # position of the second microbatch half.
        ps.reset_current_context_parallel_size()
        self.assertEqual(emb.get_rotary_seq_len(x, _Config()), 32)


if __name__ == "__main__":
    unittest.main()
