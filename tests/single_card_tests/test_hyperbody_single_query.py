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

"""Unit coverage for the single HyperBody encoder query table.

``encoder_config.hyperencoder_query_length`` (one positive int) sizes the only
query table, ``query``; every context segment is followed by it. The retired
``(short, long)`` ``hyperencoder_query_lengths`` field is no longer read.

Geometry is REQUIRED by the sensitive-info policy, so configs are built with
explicit ``decoder_config=`` / ``encoder_config=`` sub-configs.
"""

import sys
import types

import paddle

from paddlefleet.transformers.hyperbody.configuration import HyperBodyConfig
from paddlefleet.transformers.hyperbody.modeling import (
    HyperBodyEncoderFrontEnd,
    _build_encoder_view,
)

_Q = 4
_CTX = 128830

_DECODER_GEOMETRY = {
    "vocab_size": 128,
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "intermediate_size": 128,
    "n_routed_experts": 4,
    "moe_intermediate_size": 32,
    "num_experts_per_tok": 2,
    "n_shared_experts": 1,
}
_ENCODER_GEOMETRY = {
    "vocab_size": 128,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "moe_intermediate_size": 32,
    "n_routed_experts": 4,
    "num_experts_per_tok": 2,
    "n_shared_experts": 1,
    "first_k_dense_replace": 1,
    "hyperencoder_query_length": _Q,
}


def _encoder_view(packed=False, **overrides):
    encoder_config = dict(_ENCODER_GEOMETRY, **overrides)
    if packed:
        encoder_config["hyperencoder_attn_backend"] = "flashmask"
        encoder_config["hyperencoder_packed_decoder"] = True
    cfg = HyperBodyConfig(
        decoder_config=dict(_DECODER_GEOMETRY),
        encoder_config=encoder_config,
        hyperbody_context_token_id=_CTX,
    )
    return cfg, _build_encoder_view(cfg, cfg.decoder_config.hidden_size)


def _raises(fn, needle):
    try:
        fn()
    except ValueError as exc:
        assert needle in str(exc), str(exc)
        return True
    return False


def test_query_length_reaches_encoder_view():
    cfg, view = _encoder_view()
    assert cfg.encoder_config.hyperencoder_query_length == _Q
    assert view.hyperencoder_query_length == _Q
    assert view.max_sequence_length == _Q + 8192


def test_query_length_is_required_and_scalar():
    """Missing -> required error; the retired ``(short, long)`` pair -> rejected."""
    geometry = dict(_ENCODER_GEOMETRY)
    del geometry["hyperencoder_query_length"]

    def _missing():
        HyperBodyConfig(
            decoder_config=dict(_DECODER_GEOMETRY), encoder_config=geometry
        )

    assert _raises(_missing, "hyperencoder_query_length is required")
    for bad in ([_Q, 8192], (_Q, 8192), 0, -1, True, 1.5):
        assert _raises(
            lambda b=bad: _encoder_view(hyperencoder_query_length=b),
            "single positive integer",
        ), bad
    # An integral float (e.g. from a loosely typed config) is still accepted.
    _, view = _encoder_view(hyperencoder_query_length=float(_Q))
    assert view.hyperencoder_query_length == _Q
    assert type(view.hyperencoder_query_length) is int


def test_frontend_builds_one_query_table():
    _, view = _encoder_view()
    fe = HyperBodyEncoderFrontEnd(config=view)
    names = {n for n, _ in fe.named_parameters()}
    assert "query.weight" in names, names
    assert not any(n.startswith(("query_short", "query_long")) for n in names)
    assert tuple(fe.query.weight.shape) == (_Q, view.hidden_size)


def _context(fe, n_ctx=3, bs=1):
    w = fe.query.weight
    return paddle.ones([bs, n_ctx, w.shape[1]], dtype=w.dtype)


def test_dense_forward_appends_query_and_backprops():
    from paddlefleet.transformer.prefix_lm_mask import (
        build_dense_mask,
        prefix_lm_pad_len,
    )

    _, view = _encoder_view()
    fe = HyperBodyEncoderFrontEnd(config=view)
    out = fe._forward_dense(
        _context(fe, bs=2), build_dense_mask, prefix_lm_pad_len
    )
    assert out["n_queries"] == _Q, out["n_queries"]
    assert "use_long_query" not in out
    h = out["hidden_states"].astype("float32")
    q = fe.query.weight.astype("float32")
    for b in range(2):
        assert bool(paddle.equal_all(h[b, 3 : 3 + _Q], q))
    out["hidden_states"].sum().backward()
    assert fe.query.weight.grad is not None


def test_packed_forward_appends_query_per_segment():
    _, view = _encoder_view(packed=True)
    fe = HyperBodyEncoderFrontEnd(config=view)
    w = fe.query.weight
    ctx = [
        paddle.ones([2, w.shape[1]], dtype=w.dtype),
        paddle.ones([5, w.shape[1]], dtype=w.dtype),
    ]
    out = fe._forward_packed(ctx)
    assert list(out["n_queries"]) == [_Q, _Q], out["n_queries"]
    assert out["_hb_packed"]["segment_starts"] == (0, 2 + _Q)
    assert "use_long_query" not in out
    h = out["hidden_states"][0].astype("float32")
    q = w.astype("float32")
    assert bool(paddle.equal_all(h[2 : 2 + _Q], q))
    assert bool(paddle.equal_all(h[2 + _Q + 5 : 2 + _Q + 5 + _Q], q))
    out["hidden_states"].sum().backward()
    assert fe.query.weight.grad is not None


def _stub_frontend():
    """Frontend without module construction, for the pure count check."""
    fe = object.__new__(HyperBodyEncoderFrontEnd)
    fe.config = types.SimpleNamespace(
        hyperencoder_query_length=_Q, hyperbody_context_token_id=_CTX
    )
    return fe


def _decoder_segments(*counts, pad=3):
    """Flat ids + cu_seqlens; segment ``i`` holds ``counts[i]`` context tokens
    as back-to-back spans (no separator between encoder segments)."""
    ids, bounds = [], [0]
    for c in counts:
        seg = [7] * pad + [_CTX] * c + [7] * pad
        ids += seg
        bounds.append(bounds[-1] + len(seg))
    return (
        paddle.to_tensor([ids], dtype="int64"),
        paddle.to_tensor(bounds, dtype="int32"),
    )


def test_context_count_check_accepts_whole_query_tables():
    """0 (text segment), q, and several back-to-back tables per segment pass;
    lookahead tokens appended after cu_seqlens[-1] (MTP) are not counted."""
    fe = _stub_frontend()
    ids, cu = _decoder_segments(0, _Q, 3 * _Q)
    fe._check_context_token_counts(ids, cu)
    with_lookahead = paddle.concat(
        [ids, paddle.full([1, 2], _CTX, dtype="int64")], axis=1
    )
    fe._check_context_token_counts(with_lookahead, cu)


def test_context_count_check_rejects_misaligned_segments():
    """Same total as two whole tables, but split q+1 / q-1 across segments."""
    fe = _stub_frontend()
    ids, cu = _decoder_segments(_Q + 1, _Q - 1)
    assert _raises(
        lambda: fe._check_context_token_counts(ids, cu),
        "multiple of hyperencoder_query_length",
    )


def test_context_count_check_skips_non_global_layout():
    """input_ids shorter than cu_seqlens[-1] (sharded view): no check."""
    fe = _stub_frontend()
    ids, cu = _decoder_segments(_Q + 1, _Q - 1)
    fe._check_context_token_counts(ids[:, :5], cu)


if __name__ == "__main__":
    try:
        test_query_length_reaches_encoder_view()
        test_query_length_is_required_and_scalar()
        test_frontend_builds_one_query_table()
        test_dense_forward_appends_query_and_backprops()
        test_packed_forward_appends_query_per_segment()
        test_context_count_check_accepts_whole_query_tables()
        test_context_count_check_rejects_misaligned_segments()
        test_context_count_check_skips_non_global_layout()
    except AssertionError:
        import traceback

        traceback.print_exc()
        sys.exit(1)
    print("OK")
