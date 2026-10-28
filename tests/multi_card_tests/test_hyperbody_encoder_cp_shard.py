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
"""``hyperencoder_cp_shard``: the CP-sharded HyperBody encoder equals the
replicated one, forward and backward.

Builds frontend + flashmask trunk (seq-aux-loss MoE, EP == CP) + bridge twice
from the same weights -- replicated and CP-sharded -- runs the same packed
prefix-LM batch on every CP rank and checks, per parameter, the identity the
trainer's grad reduce relies on:

* frontend / bridge (replicated): every rank's grad equals the replicated grad;
* trunk dense params: the sum of the per-rank sharded grads equals the
  replicated grad (the trainer turns its mean reduce into this sum with x cp);
* trunk experts: replicated experts see every token cp times, so their grad is
  cp x the sharded grad (the trainer's ``expert_cp_replicated`` / cp);
* grad-scale marks: the sharded trunk is unmarked, everything replicated is.

The replicated encoder is run twice first; its run-to-run diff is printed as
the noise floor next to the sharded diff. The aux-loss coefficient is large so a
wrong aux gradient is visible. The optimizer is not run.

    python -m paddle.distributed.launch --gpus=0,1,2,3 \
        tests/multi_card_tests/test_hyperbody_encoder_cp_shard.py

Env: ``HB_TEST_CP`` (4), ``HB_TEST_CP_MODE`` (dualchunk_allgather |
contiguous_allgather; default runs both modes).
"""

import os
import sys

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(_repo_root, "src"))

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import LayerSpec, build_spec_layer

from paddlefleet.training.initialize import initialize_fleet

# bf16 noise of two kernel tilings / reduction orders measures <= 6e-3; a wrong
# aux-loss path measures ~7e-2, a cp-factor error >= 0.5.
_RTOL = 1.5e-2
_Q = 8


def _rel(a, b):
    a, b = a.astype("float64"), b.astype("float64")
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def _np(t):
    return t.astype("float32").numpy()  # bf16 -> fp32 is exact


def _build(cp, shard, mode):
    from paddlefleet.models.hyperbody import get_hyperbody_encoder_layer_specs
    from paddlefleet.transformers.hyperbody.configuration import HyperBodyConfig
    from paddlefleet.transformers.hyperbody.modeling import (
        HyperBodyEncoderBridge,
        HyperBodyEncoderFrontEnd,
        _build_encoder_view,
    )

    cfg = HyperBodyConfig(
        decoder_config={
            "vocab_size": 128,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "n_routed_experts": 4,
            "moe_intermediate_size": 32,
            "num_experts_per_tok": 2,
            "n_shared_experts": 1,
        },
        encoder_config={
            "vocab_size": 128,
            "hidden_size": 256,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,  # head_dim 128, as in production
            "moe_intermediate_size": 64,
            "n_routed_experts": 4,
            "num_experts_per_tok": 2,
            "n_shared_experts": 1,
            "first_k_dense_replace": 1,
            "hyperencoder_query_length": _Q,
            "hyperencoder_seq_align": 32,
            "hyperencoder_attn_backend": "flashmask",
            "hyperencoder_packed_decoder": True,
            "hyperencoder_cp_shard": shard,
            "router_aux_loss_coef": 1.0,
        },
        context_parallel_size=cp,
        expert_model_parallel_size=cp,
        cp_balance_mode=mode,
        moe_router_load_balancing_type="seq_aux_loss",
        moe_token_dispatcher_type="deepep",
    )
    view = _build_encoder_view(cfg, 64)
    assert view.hyperencoder_cp_shard is shard
    layers = [
        build_spec_layer(
            LayerSpec(HyperBodyEncoderFrontEnd, extra_kwargs={"config": view})
        )
    ]
    layers += [
        build_spec_layer(s) for s in get_hyperbody_encoder_layer_specs(view)
    ]
    layers.append(
        build_spec_layer(
            LayerSpec(HyperBodyEncoderBridge, extra_kwargs={"config": view})
        )
    )
    return layers, view


class _Region:
    """The run_function / prefix view ``_mark_encoder_region`` reads."""

    def __init__(self, layers):
        self.run_function = layers
        n = len(layers)
        self._prefixes = {
            str(i): "encoder"
            if i == 0
            else ("encoder.bridge" if i == n - 1 else f"encoder.layers.{i - 1}")
            for i in range(n)
        }

    def get_sequential_name_prefixes(self):
        return self._prefixes


def _tag(i, n):
    return (
        "frontend" if i == 0 else ("bridge" if i == n - 1 else f"trunk{i - 1}")
    )


def _run(layers, view, ctx, proj, group):
    """Forward + backward. Returns {name: (kind, value, marked)} with every grad
    already brought to the replicated scale, then clears the grads."""
    from paddlefleet.tensor_parallel import model_parallel_cuda_manual_seed

    model_parallel_cuda_manual_seed(2026)
    cp, shard = view.context_parallel_size, view.hyperencoder_cp_shard
    ctx = [c.detach() for c in ctx]
    for c in ctx:
        c.stop_gradient = False
    d = layers[0]._forward_packed(ctx)
    for layer in layers[1:-1]:
        d = layer(d)
    emb = layers[-1]._forward_packed(d, d["_hb_packed"])  # gathers under shard
    (emb.astype("float32") * proj).sum().backward()

    def summed(t):
        out = []
        dist.all_gather(out, t.astype("float32").contiguous(), group=group)
        return np.sum([x.numpy() for x in out], axis=0)  # fixed rank order

    res = {"image_embeds": ("emb", _np(emb), None)}
    for i, c in enumerate(ctx):
        res[f"ctx[{i}].grad"] = ("ctx", _np(c.grad), None)
    for i, m in enumerate(layers):
        tag = _tag(i, len(layers))
        for name, p in m.named_parameters():
            if p.grad is None:
                continue
            color = getattr(p, "color", None)
            if tag in ("frontend", "bridge"):
                kind, g = "replicated", _np(p.grad)
            elif isinstance(color, dict) and color.get("color") == "moe_expert":
                kind = "expert"
                g = _np(p.grad) * (cp if shard else 1)
            else:
                kind = "dense"
                g = summed(p.grad) if shard else _np(p.grad)
            marked = bool(
                getattr(p, "context_parallel_disable_scale_grad", False)
                or getattr(p, "expert_cp_replicated", False)
            )
            res[f"{tag}.{name}"] = (kind, g, marked)
            p.clear_gradient(False)
    return res


def _worst(a, b):
    """Per kind, the worst relative diff of ``a`` against reference ``b``."""
    assert a.keys() == b.keys(), set(a) ^ set(b)
    worst, rows = {}, []
    for name, (kind, got, _) in a.items():
        r = _rel(got, b[name][1])
        worst[kind] = max(worst.get(kind, 0.0), r)
        rows.append((name, kind, r))
    return worst, rows


def main():
    cp = int(os.environ.get("HB_TEST_CP", "4"))
    supported_modes = ("dualchunk_allgather", "contiguous_allgather")
    requested_mode = os.environ.get("HB_TEST_CP_MODE")
    modes = (requested_mode,) if requested_mode is not None else supported_modes
    if any(mode not in supported_modes for mode in modes):
        raise ValueError(f"Unsupported HB_TEST_CP_MODE: {requested_mode!r}")
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": 1,
        "pp_degree": 1,
        "sharding_degree": cp,
        "sep_degree": 1,
        "cp_degree": cp,
        "ep_degree": cp,
        "moe_sharding_degree": 1,
        "order": [
            "sharding",
            "moe_sharding",
            "pp",
            "sep",
            "cp",
            "dp",
            "ep",
            "mp",
        ],
    }
    initialize_fleet(strategy)
    paddle.set_flags({"FLAGS_flash_attn_version": 4})
    paddle.set_default_dtype("bfloat16")
    group = fleet.get_hybrid_communicate_group().get_context_parallel_group()
    for mode in modes:
        _check_mode(cp, mode, group)
        # Finish all ranks before constructing the next independent model pair.
        dist.barrier(group=group)


def _check_mode(cp, mode, group):
    from paddlefleet.tensor_parallel import model_parallel_cuda_manual_seed
    from paddlefleet.transformers.hyperbody.modeling import _mark_encoder_region

    rank = dist.get_rank()
    paddle.seed(2026)
    model_parallel_cuda_manual_seed(2026)
    rep, rep_view = _build(cp, False, mode)
    sh, sh_view = _build(cp, True, mode)
    for a, b in zip(
        [p for m in rep for p in m.parameters()],
        [p for m in sh for p in m.parameters()],
    ):
        b.set_value(a)
    _mark_encoder_region(_Region(rep), rep_view)
    _mark_encoder_region(_Region(sh), sh_view)

    # Same batch on every CP rank (the HyperBody batch is replicated over CP):
    # 12 context + 2*8 query rows = 28 -> pad 32. Both modes split real
    # segments across ranks; retain dense/MoE layers with a minimal sequence.
    paddle.seed(7)
    lens = [9, 3]
    ctx = [paddle.randn([n, 256]).astype("bfloat16") for n in lens]
    proj = paddle.randn([_Q * len(lens), 64]).astype("float32")

    r1 = _run(rep, rep_view, ctx, proj, group)
    r2 = _run(rep, rep_view, ctx, proj, group)
    s1 = _run(sh, sh_view, ctx, proj, group)

    noise, _ = _worst(r2, r1)
    worst, rows = _worst(s1, r1)
    errs = [f"{n} ({k}): rel {r:.3e}" for n, k, r in rows if not r < _RTOL]
    checked = {"replicated": 0, "dense": 0, "expert": 0}
    for name, (kind, _, marked) in s1.items():
        if kind not in checked:
            continue
        checked[kind] += 1
        if marked != (kind == "replicated"):
            errs.append(f"{name}: shard mark={marked} for kind={kind}")
        if not r1[name][2]:
            errs.append(f"{name}: replicated encoder param left unmarked")

    fmt = lambda d: {k: f"{v:.2e}" for k, v in d.items()}
    print(
        f"[rank {rank}] cp={cp} {mode} checked={checked}\n"
        f"  noise (rep vs rep) {fmt(noise)}\n"
        f"  shard (sh vs rep)  {fmt(worst)}",
        flush=True,
    )
    assert all(checked.values()), checked
    assert not errs, "\n".join(errs[:20])


if __name__ == "__main__":
    main()
