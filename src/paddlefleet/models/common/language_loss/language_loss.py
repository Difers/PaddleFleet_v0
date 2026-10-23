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


import contextlib
import functools
import hashlib
import os
from contextlib import contextmanager
from contextvars import ContextVar

import numpy as np
import paddle
import paddle.distributed as dist
from paddle import Tensor, nn
from paddle.autograd import PyLayer
from paddle.distributed import fleet
from paddle.distributed.fleet.layers.mpu import mp_ops
from paddle.distributed.fleet.meta_parallel import ScheduleNode
from paddle.distributed.fleet.utils import recompute
from paddle.distributed.fleet.utils.sequence_parallel_utils import AllGatherOp

from paddlefleet.accuracy_compatible_patch import LossScaleBeforeBackward
from paddlefleet.context_parallel_utils import (
    ContextParallelGatherOp,
    ContextParallelScatterOp,
    MTPDistillationLossShift,
)
from paddlefleet.cp_shard import cp_shard_source, to_cp_global, to_cp_local
from paddlefleet.parallel_state import (
    get_context_parallel_group,
    get_context_parallel_world_size,
    get_expert_model_parallel_group,
    get_tensor_model_parallel_world_size,
)
from paddlefleet.process_groups_config import ProcessGroupCollection
from paddlefleet.recompute_utils import module_needs_recompute
from paddlefleet.training.global_vars import get_global_training_logs
from paddlefleet.transformer.layer import FleetLayer
from paddlefleet.transformer.transformer_config import (
    TransformerConfig,
    mtp_layers_active,
)
from paddlefleet.utils import use_dsv4_accuracy_compatible

# A replay must not notify either the original observer or one installed by a
# later micro-batch. Context-local state also keeps nested calls isolated.
_token_loss_replaying = ContextVar("token_loss_replaying", default=False)

# Global token-avg loss bookkeeping (log_global_token_avg_loss): which head the
# current CE belongs to, as a log-key prefix ("" = main LM head -> "loss",
# "mtp_1_" -> "mtp_1_loss"), so the main and every MTP head are reduced
# separately instead of being pooled into one ΣS/Σn.
_token_avg_head = ContextVar("token_avg_head", default="")
# Explicit record decision for a recomputed CE (None = decide from grad mode).
# The recompute wrapper fixes it at call time: record on the first pass iff
# training, never on the backward replay, whatever grad mode each pass runs in.
_token_avg_record = ContextVar("token_avg_record", default=None)


def _mtp_token_avg_prefix(depth: int) -> str:
    return f"mtp_{depth + 1}_"


@contextlib.contextmanager
def _token_avg_head_scope(prefix: str):
    token = _token_avg_head.set(prefix)
    try:
        yield
    finally:
        _token_avg_head.reset(token)


# calculate_per_token_loss bookkeeping lives in LanguageLoss.token_avg_tracker,
# next to log_global_token_avg_loss's ("S", "n"), under the keys below (key ->
# dtype). Values are detached device tensors, so no micro-batch forces a D2H
# sync. The two key sets have different lifetimes and never touch each other:
# the trainer pops every _PT_KEYS entry at each accumulation boundary
# (LanguageLoss.pop_per_token_stats), while ("S", "n") are consumed once per
# log interval (LanguageLoss.consume_token_avg_loss).
#   * pt_t_local: this rank's non-padded main-head label tokens, counted before
#     the CP gather; the trainer all-reduces it into T_global (grad scale).
#   * pt_lm_sum / pt_lm_tok, pt_mtp_sum / pt_mtp_tok: post-CP-gather loss sums
#     and token counts, for the global `per_token_loss` / `mtp_per_token_loss`.
#   * pt_mom_sum / pt_mom_cnt: per-microbatch per-token mean of the combined
#     loss, comparable to the legacy `loss`.
# All but pt_t_local are display-only and never affect grads.
_PT_KEYS = {
    "pt_t_local": "int64",
    "pt_lm_sum": "float64",
    "pt_lm_tok": "int64",
    "pt_mtp_sum": "float64",
    "pt_mtp_tok": "int64",
    "pt_mom_sum": "float64",
    "pt_mom_cnt": "int64",
}


def _per_token_tracker_add(key, value):
    """Add a detached scalar (tensor or python number) to tracker[key]."""
    dtype = _PT_KEYS[key]
    if isinstance(value, paddle.Tensor):
        value = value.detach().astype(dtype).reshape([])
    else:
        value = paddle.to_tensor(value, dtype=dtype)
    _tr = LanguageLoss.token_avg_tracker
    prev = _tr.get(key)
    _tr[key] = value if prev is None else prev + value


def accumulate_per_token_local_count(n):
    """Add this microbatch's per-shard non-padded token count to the step total."""
    _per_token_tracker_add("pt_t_local", n)


# T_global counts main-head tokens only (as in Megatron); MTP heads are
# renormalised to it inside the loss. MTP calls into forward_impl are wrapped in
# suppress_per_token_count() so their labels are not counted.
_PT_SUPPRESS_COUNT = ContextVar("pt_suppress_count", default=False)


@contextmanager
def suppress_per_token_count():
    """Keep the enclosed loss call out of the per-token T_global accounting."""
    token = _PT_SUPPRESS_COUNT.set(True)
    try:
        yield
    finally:
        _PT_SUPPRESS_COUNT.reset(token)


def _per_token_count_enabled(config):
    """Whether this call adds to T_global: per-token on, not a recompute replay,
    and not an MTP head."""
    return (
        getattr(config, "calculate_per_token_loss", False)
        and not _token_loss_replaying.get()
        and not _PT_SUPPRESS_COUNT.get()
    )


def _loss_md5_enabled() -> bool:
    return os.environ.get("LOG_LOSS_MD5", "0") == "1"


def _use_accuracy_compatible_kernel() -> bool:
    """Switch for Megatron-aligned (accuracy-compatible) numeric paths.

    Controlled by the ``FLAGS_use_accuracy_compatible_kernel`` env variable.
    """
    return os.environ.get("FLAGS_use_accuracy_compatible_kernel", "0") == "1"


def _tensor_md5(tensor: Tensor, dtype: str = "float32") -> str:
    """Calculate MD5 hash of a tensor, **for debugging only**.

    Note: internally calls .numpy() which triggers GPU→CPU synchronization
    and blocks the async training pipeline. Do NOT use in the forward pass.
    """
    tensor_for_md5 = tensor.detach().cast(dtype)
    return hashlib.md5(tensor_for_md5.numpy().tobytes()).hexdigest()


def _print_scalar_loss_md5(prefix: str, name: str, loss: Tensor) -> None:
    if not _loss_md5_enabled():
        return
    rank = paddle.distributed.get_rank()
    loss_tensor = loss.detach().cast("float32").reshape([1])
    print(
        f"[{prefix}] rank={rank} {name}={loss_tensor.item():.20f} "
        f"{name}_md5={_tensor_md5(loss_tensor)}",
        flush=True,
    )


class DistributedSoftmaxOp(PyLayer):
    @staticmethod
    def forward(ctx, x, axis=-1, mp_group=None):
        ctx.axis = axis
        if mp_group is None:
            hcg = fleet.get_hybrid_communicate_group()
            mp_group = hcg.get_model_parallel_group()

        ctx.mp_group = mp_group

        local_max = paddle.max(x, axis=axis, keepdim=True)

        all_max = AllGatherOp.apply(local_max)

        global_max = paddle.max(all_max, axis=0, keepdim=True)

        x_stable = x - global_max

        exp_x = paddle.exp(x_stable.cast("float32"))

        local_sum_exp = paddle.sum(exp_x, axis=axis, keepdim=True)

        sum_exp = mp_ops._mp_allreduce(
            local_sum_exp,
            group=mp_group,
            use_calc_stream=True,
            use_model_parallel=True,
        )

        softmax_output = exp_x / sum_exp

        ctx.save_for_backward(softmax_output, sum_exp)

        return softmax_output

    @staticmethod
    def backward(ctx, grad_output):
        softmax_output, global_sum_exp = ctx.saved_tensor()
        axis = ctx.axis
        mp_group = ctx.mp_group

        grad_softmax = grad_output * softmax_output

        local_sum_grad = paddle.sum(grad_softmax, axis=axis, keepdim=True)

        all_sum_grad = AllGatherOp.apply(local_sum_grad)
        global_sum_grad = paddle.sum(all_sum_grad, axis=0, keepdim=True)

        grad_input = softmax_output * (grad_output - global_sum_grad)

        return grad_input


def subbatch(
    f, arg_idx, axis, bs, out_idx, use_recompute=False, same_arg_idx={}
):
    """
    Converts a function to one that applies to subbatch of an input dimension.
    This is useful for processing large tensors in smaller chunks to reduce memory usage.

    Args:
        f (Callable): Original function to be converted to subbatch processing.
        arg_idx ([int]): Indices of the inputs to be subbatched.
        axis ([int]): Indices of the dimensions to be subbatched for each input.
        bs (int): Subbatch size (number of elements to process at once).
        out_idx (int): Index of the output dimension that needs stacking.
        use_recompute (bool, optional): Whether to use recomputation for memory savings. Defaults to False.
        same_arg_idx (dict, optional): Mapping of argument indices that share the same tensor.
                                     e.g. {1: 0} means args[1] == args[0], avoiding duplicate slicing.

    Returns:
        Callable: Converted function that processes inputs in subbatches.
    """

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        assert len(arg_idx) == len(axis), (
            "Number of batching args and number of batching dims should match."
        )

        inps = [args[i] for i in arg_idx]
        axis_width = [inp.shape[d] for inp, d in zip(inps, axis)]
        assert len(set(axis_width)) == 1, "Batch sizes should be kept equal."

        inp_axis = dict(zip(inps, axis))

        axis_width = axis_width[0]
        if axis_width < bs:
            return f(*args, **kwargs)

        outs = []
        for slice_at in np.arange(0, axis_width, bs):
            _args = []
            for i, inp in enumerate(args):
                if i in same_arg_idx:
                    assert i > same_arg_idx[i], (
                        f"expect i > same_arg_idx[i], but got i: {i} and same_arg_idx[i]: {same_arg_idx[i]}"
                    )
                    _args.append(_args[same_arg_idx[i]])
                elif i in arg_idx:
                    inp = inp.slice(
                        [inp_axis[inp]],
                        [slice_at],
                        [min(inp.shape[inp_axis[inp]], slice_at + bs)],
                    )
                    _args.append(inp)
                else:
                    _args.append(inp)
            if use_recompute:
                out = paddle.distributed.fleet.utils.recompute(
                    f, *_args, **kwargs
                )
            else:
                out = f(*_args, **kwargs)
            outs.append(out)

        return paddle.cat(outs, out_idx)

    return wrapper


# Publish failures are counted rather than announced once. A channel that
# stays broken (renamed key, logs unavailable in this configuration) would
# otherwise drop every dspark metric for the rest of the run after a single
# line on stdout, and the caller only logs -- it can never notice on its own.
_DSPARK_TERMS_WARN_EVERY = 100
_dspark_terms_publish_failures = 0


def _report_dspark_terms_failure(exc):
    """Count a failed publish and re-report it periodically."""
    global _dspark_terms_publish_failures
    _dspark_terms_publish_failures += 1
    failures = _dspark_terms_publish_failures
    if failures == 1 or failures % _DSPARK_TERMS_WARN_EVERY == 0:
        print(
            f"[dspark-terms] publish skipped ({failures} failures so far): "
            f"{exc!r}",
            flush=True,
        )


def _publish_dspark_terms(terms):
    """Surface accept-rate and the draft-loss breakdown per step.

    ``compute_dspark_loss`` already builds ``accept_rate_per_pos_num/_den``
    and ``tau_num/_den``, but its default ``return_terms=False`` dropped them,
    so the only accept-rate ever observed came from print-only probes that
    exist in a different fleet line. Publishing through the same
    ``get_global_training_logs`` channel as ``loss`` / ``dspark_loss`` puts
    them in the per-step JSONL (``metrics_rank0.json``) with no TensorBoard
    wiring.

    The loss terms get the same treatment for the same reason: a micro-batch
    with every block dropped yields loss 0 (``dspark_losses.py:1015``), and
    averaging that in dilutes the draft-loss curve. ``<base>_num`` is the
    weighted token sum -- ``term * denominator``, recovering the numerator
    ``compute_dspark_loss`` divided by -- and ``<base>_den`` is this rank's
    own weight sum, so the derived reading is per-token at any CP size.
    ``dspark_loss`` itself is left alone: it is the pre-existing
    production key every earlier run is compared against.

    Logging only -- every value is detached, and a failure here must never
    take down a training step.
    """
    logs = get_global_training_logs()
    if logs is None or not hasattr(logs, "update"):
        return
    try:
        values = dict(terms.terms)
        payload = {}
        num = values.get("accept_rate_per_pos_num")
        den = values.get("accept_rate_per_pos_den")
        if num is not None and den is not None:
            num = num.astype("float32").detach()
            den = den.astype("float32").detach()
            for k in range(int(num.shape[0])):
                payload[f"dspark_accept_k{k + 1}_num"] = num[k]
                payload[f"dspark_accept_k{k + 1}_den"] = den[k]
            payload["dspark_accept_all_num"] = num.sum()
            payload["dspark_accept_all_den"] = den.sum()
        for base in ("tau", "confidence_prob"):
            pair_num = values.get(base + "_num")
            pair_den = values.get(base + "_den")
            if pair_num is None or pair_den is None:
                continue
            payload[f"dspark_{base}_num"] = pair_num.astype("float32").detach()
            payload[f"dspark_{base}_den"] = pair_den.astype("float32").detach()
        den_w = values.get("normalized_weight_denominator")
        if den_w is not None:
            den_w = den_w.astype("float32").detach()
            local_w = values.get("weight_denominator")
            pub_den = (
                local_w.astype("float32").detach()
                if local_w is not None
                else den_w
            )
            for name, base in (
                ("loss", "draft"),
                ("ce_loss", "ce"),
                ("l1_loss", "l1"),
                ("tv_loss", "tv"),
                ("confidence_loss", "confidence"),
            ):
                value = values.get(name)
                if value is None:
                    continue
                payload[f"dspark_{base}_num"] = (
                    value.astype("float32").detach() * den_w
                )
                payload[f"dspark_{base}_den"] = pub_den
            if local_w is not None:
                payload["dspark_weight_tokens_local"] = pub_den
                payload["dspark_weight_tokens_group"] = den_w
        logs.update(**payload)
    except Exception as exc:  # logging must not kill a training step
        _report_dspark_terms_failure(exc)


def _publish_dspark_loss(scaled):
    """Surface the draft loss next to the main one in the training logs.

    The main ``loss`` is deliberately left as a clean lm_loss: the draft
    contributes gradient only, through ``DSparkLossAutoScaler``, which is the
    same ``main + draft - draft.detach()`` contract the MTP path spells out
    explicitly. But nothing reads the loss tracker, so without this the draft
    loss is invisible and a healthy-looking lm_loss cannot tell you whether the
    draft is training at all. Mirrors how the MTP path publishes
    ``mtp_{i}_loss``.
    """
    logs = get_global_training_logs()
    if logs is not None and hasattr(logs, "update"):
        logs.update(dspark_loss=scaled.detach())


class LanguageLoss(FleetLayer):
    """Language loss with an optional ``_eval_token_loss_hook(loss, labels)``.

    The observer receives unreduced CE before masking/normalization, in the
    same token layout as the accompanying labels. It runs for the main head
    and each CE-based MTP head (not the distillation objective), including
    all-masked inputs. Selective ``loss_fn`` backward recomputation does not
    notify again. Direct ``forward_impl`` calls each notify independently.

    Observers must not mutate the tensors; detach any values retained for
    metrics. Return values are ignored and observer exceptions propagate.
    Head selection/aggregation belongs to the caller.
    """

    # Class-level tracker for MTP loss, read by trainer for logging.
    mtp_loss_tracker: dict[str, float] = {}

    # Class-level accumulator for the global token-weighted loss (logging only),
    # filled per micro-batch by _reduce_loss_by_tokens and reduced by
    # consume_token_avg_loss, whose only caller is
    # ``Trainer._apply_global_token_avg_loss``. Keys are ``<head prefix>S`` /
    # ``<head prefix>n``: ``S``/``n`` for the main head, ``mtp_1_S``/``mtp_1_n``
    # for MTP depth 1, and so on. Also holds the calculate_per_token_loss step
    # stats under the ``_PT_KEYS`` keys, popped by pop_per_token_stats; each
    # consumer only removes its own keys.
    token_avg_tracker: dict = {}

    # Class-level stash for cu_seqlens_q under use_erndata=True.
    # Populated on every rank by the dataloader (ernie5
    # dist_data_loader.py — right after the three broadcast_data_obj calls),
    # and additionally by gpt_embedding.forward on the embedding stage as a
    # PP=1 safety net. Consumed here to drive strict per-doc `paddle.roll`
    # in the MTP label-rolling loop so EOS positions are zero-masked
    # correctly and match the embedding-side per-doc roll bit-exactly.
    #
    # PP=1: each micro-batch runs embedding→loss end-to-end before the next
    #   arrives, so the stash is race-free even without thread-locals.
    # PP>1: the last stage never runs embedding; the dataloader stash on every
    #   rank is what enables the loss stage to see cu_seqlens_q. cu_seqlens_q
    #   travels alongside `mtp_startend_row_indices_all` in every
    #   `broadcast_data_obj` tuple (shuffle / main / pp_data).
    _cu_seqlens_q_stash: "paddle.Tensor | None" = None

    def __init__(
        self,
        config: TransformerConfig,
        pg_collection=None,
    ) -> None:
        super().__init__(config=config)
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.pg_collection = pg_collection

        self.config = config
        # Per-token loss returns a raw token sum; the trainer divides by T_global.
        # `is True` rather than bool() so a MagicMock config does not enable it.
        self.calculate_per_token_loss = (
            getattr(config, "calculate_per_token_loss", False) is True
        )
        self.use_accuracy_compatible = getattr(
            config, "use_accuracy_compatible", False
        )
        self.ignored_index = -100
        self.enable_parallel_cross_entropy = (
            paddle.distributed.is_initialized()
            and get_tensor_model_parallel_world_size() > 1
            and config.parallel_output
        )

        if self.enable_parallel_cross_entropy:
            self.loss_func = (
                paddle.distributed.fleet.meta_parallel.ParallelCrossEntropy()
            )
        else:
            self.loss_func = paddle.nn.CrossEntropyLoss(
                reduction="none",
            )

        self.loss_subbatch_sequence_length = (
            config.loss_subbatch_sequence_length
        )
        self.use_subbatch = self.loss_subbatch_sequence_length > 0

        # Per-token loss + MTP: reject configs whose per-head token
        # normalisation is not supported, instead of silently mis-scaling.
        num_mtp = getattr(config, "num_nextn_predict_layers", 0)
        mtp_enabled = (
            isinstance(num_mtp, int)
            and num_mtp > 0
            and not getattr(config, "mtp_load_weight_only", False)
        )
        if self.calculate_per_token_loss and mtp_enabled:
            assert not config.train_mtp_only, (
                "calculate_per_token_loss + train_mtp_only is unsupported: the main "
                "head is skipped, so T_global would be 0."
            )
            assert not config.mtp_distillation_loss, (
                "calculate_per_token_loss + mtp_distillation_loss is unsupported: "
                "that path returns a per-token mean rather than a raw sum."
            )

    def _reduce_loss_by_tokens(self, loss_sum, local_valid):
        """Normalize this rank's (loss_sum, valid_tokens) into a scalar loss.

        The two switches are orthogonal and neither reads the other:

        * ``log_global_token_avg_loss`` -- a *training* forward stashes the raw
          ``(loss_sum, valid_tokens)`` in a class-level tracker so the trainer
          can log the global token-averaged loss. No collective communication
          here, and no effect on the returned value.
        * ``calculate_per_token_loss`` -- decides what this returns:
          ``loss_sum`` untouched (the trainer divides once by T_global), versus
          ``loss_sum / local_valid`` for the legacy micro-average.

        Scope -- ``forward_impl`` routes through this helper on the fused
        linear+CE path and the default non-fused path. Two other terminal
        branches normalize on their own and so never accumulate: the
        ``gpt_model_use_experimental_version`` line-wise reduction (per-sample
        mean, then averaged over samples) and the ``use_accuracy_compatible``
        EP-replicated fp64 reduction. Neither is ΣΣS/ΣΣn, so folding them in
        would change what the logged number means. Under those configurations
        ``log_global_token_avg_loss`` therefore leaves the tracker empty,
        ``consume_token_avg_loss`` maps that head to ``None`` and the trainer
        keeps logging its previous value (the micro-average for ``loss``, the
        last micro-batch's per-rank value for ``mtp_{d}_loss``) -- the absence
        of a ``loss_microavg`` (``mtp_{d}_loss_last``) key is how to tell that
        apart from the feature being active. The branch is
        chosen by config, identically on every rank, so there is no desync
        risk. (The all-ignored early returns skip the helper as well, but a
        micro-batch with zero valid tokens would contribute (0, 0) anyway.)

        Eval/predict forwards deliberately do not accumulate: the logged value
        covers one training logging interval and only the training loss is
        unified (``evaluate()`` keeps reporting its micro-average). Grad mode is
        the discriminator -- both eval forward paths (``prediction_step``,
        ``prediction_pipeline_step``) run under ``paddle.no_grad()`` -- so even a
        subclass that reimplements the eval loop (ERNIE's
        ``EvalPPLMixin.evaluate``) stays correct with no trainer coordination.
        A recomputed CE (``loss_fn`` in ``recompute_modules``) instead follows
        ``_token_avg_record``, fixed by ``_forward`` at call time, so it is
        counted exactly once whichever recompute pass runs with grad.

        ``self.training`` cannot serve as that discriminator: the criterion is
        stored in ``PipelineLayer._loss_fn`` (a plain ``list``), never a
        registered sublayer, so ``model.eval()`` never reaches it and the flag
        is always ``True``.

        Heads are kept apart: the terms go under the ``_token_avg_head`` prefix
        (main head ``""``, MTP depth d ``mtp_{d}_``), so the logged main loss
        is ΣΣS/ΣΣn of the main head alone and each MTP head gets its own.
        """
        # Accumulate this rank's device scalars; the trainer produces the
        # logging value. Grad mode keeps eval/predict out of the interval.
        # A recompute replay re-runs forward_impl, so it must not be counted
        # twice: a recomputed CE carries an explicit ``_token_avg_record``
        # (False on the replay); otherwise grad mode decides and any replay is
        # skipped. MTP heads go to their own prefixed keys, so the main-head
        # ΣΣS/ΣΣn still matches Megatron ``lm loss``.
        if getattr(self.config, "log_global_token_avg_loss", False):
            record = _token_avg_record.get()
            if record is None:
                record = (
                    paddle.is_grad_enabled() and not _token_loss_replaying.get()
                )
            if record:
                _tr = LanguageLoss.token_avg_tracker
                _p = _token_avg_head.get()
                _tr[_p + "S"] = _tr.get(_p + "S", 0.0) + loss_sum.detach()
                _tr[_p + "n"] = _tr.get(_p + "n", 0.0) + local_valid.detach()
        # Per-token: raw token sum, the trainer divides once by T_global.
        if getattr(self, "calculate_per_token_loss", False):
            return loss_sum
        # Backprop: each rank normalizes by its local valid tokens. The clip is
        # defensive -- both call sites return early when every label is ignored,
        # so ``local_valid >= 1``.
        return loss_sum / paddle.clip(local_valid, min=1.0)

    @classmethod
    def consume_token_avg_loss(cls, num_mtp: int = 0) -> dict:
        """all_reduce(SUM) the accumulated per-head (ΣS, Σn) and return each
        head's global token-weighted loss ΣΣS/ΣΣn, clearing those (S, n) keys.

        Returns ``{"loss": main, "mtp_1_loss": ..., "mtp_{num_mtp}_loss": ...}``;
        a head with no valid tokens this interval (non-last pipeline stage,
        ``train_mtp_only``, sampled-out MTP depth, already consumed) maps to
        ``None``.

        COLLECTIVE: one ``all_reduce`` on the default group, so every rank must
        reach this the same number of times with the same ``num_mtp`` -- gate
        only on rank-invariant state (not on e.g. ``evaluation_loop``'s
        per-rank ``all_losses``) and decide what to log after it returns. The
        head set comes from ``num_mtp`` (config), never from this rank's
        tracker contents, so ranks that accumulated nothing still contribute
        zeros in lockstep.

        Only the per-head (S, n) keys are cleared: the calculate_per_token_loss
        keys share the tracker but are popped per accumulation step by
        pop_per_token_stats.
        """
        _tr = cls.token_avg_tracker
        prefixes = [""] + [_mtp_token_avg_prefix(d) for d in range(num_mtp)]
        packed = paddle.to_tensor(
            [float(_tr.get(p + k, 0.0)) for p in prefixes for k in ("S", "n")],
            dtype="float64",
        )
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(packed, dist.ReduceOp.SUM)
        # Not ``_tr.clear()``: that would also drop the _PT_KEYS entries.
        for p in prefixes:
            _tr.pop(p + "S", None)
            _tr.pop(p + "n", None)
        vals = packed.numpy().tolist()
        out = {}
        for i, p in enumerate(prefixes):
            S, n = vals[2 * i], vals[2 * i + 1]
            out[p + "loss"] = round(S / n, 8) if n > 0 else None
        return out

    @classmethod
    def pop_per_token_stats(cls):
        """Pop this accumulation step's calculate_per_token_loss stats.

        Returns ``{key: 0-d tensor}`` for every ``_PT_KEYS`` key (zeros for keys
        not written this step, e.g. non-last pipeline stages or no MTP) and
        removes them from the tracker. No communication and no D2H sync; the
        trainer reduces and reads them at the step boundary.
        """
        _tr = cls.token_avg_tracker
        out = {}
        for key, dtype in _PT_KEYS.items():
            value = _tr.pop(key, None)
            out[key] = value if value is not None else paddle.zeros([], dtype)
        return out

    def forward_impl(self, logits: Tensor | tuple, labels: Tensor) -> Tensor:
        # Per-token loss is training-only: the returned raw sum is not a valid
        # eval loss, and eval would leak tokens into the next step's T_global.
        assert self.training or not self.calculate_per_token_loss, (
            "eval / non-training forward is unsupported under "
            "calculate_per_token_loss: the returned loss is a raw token sum and "
            "the token counters are only popped by the training step."
        )
        # Fused linear + cross-entropy path: `logits` is actually a
        # (hidden_states, weight, bias) tuple emitted by GPTLMHead when
        # config.fused_linear_ce_loss_chunk > 0. Dispatch to the fused kernel
        # to avoid materializing the full [B, S, V] logits tensor.
        if isinstance(logits, tuple):
            assert not self.enable_parallel_cross_entropy, (
                "fused_linear_ce_loss_chunk is incompatible with tensor parallel "
                "parallel_output=True (ParallelCrossEntropy path)."
            )
            from paddlefleet.triton_ops.fused_linear_cross_entropy import (
                LigerFusedLinearCrossEntropyFunction,
            )

            hidden_states, weight, bias = logits[:3]
            # Multimax lm_head fused path: GPTLMHead emits a 5-tuple
            # (hidden_states, weight, bias, multimax_ranges, multimax_ts)
            # so SegLU is applied inside the chunked CE kernel without
            # materializing full [B, S, V] logits.
            multimax_ranges = logits[3] if len(logits) > 3 else None
            multimax_ts = logits[4] if len(logits) > 4 else None
            B, S, H = hidden_states.shape
            _input = hidden_states.reshape([-1, H])
            _labels = labels.reshape([-1])

            apply_args = [
                _input,
                weight,
                _labels,
                bias,
                self.ignored_index,
                "none",
                self.config.fused_linear_ce_loss_chunk,
                getattr(
                    self.config, "gpt_model_use_experimental_version", False
                ),
            ]
            if multimax_ranges is not None and multimax_ts is not None:
                apply_args.append(multimax_ranges)
                apply_args.append(multimax_ts)
            loss_1d = LigerFusedLinearCrossEntropyFunction.apply(*apply_args)
            # Reshape back to [B, S] so downstream CP gather / lossmask
            # handling matches the non-fused path exactly.
            loss = loss_1d.reshape([B, S])

            # Per-token: count this shard's tokens before the CP gather.
            if _per_token_count_enabled(self.config):
                accumulate_per_token_local_count(
                    (labels != self.ignored_index).sum()
                )

            loss = to_cp_global(loss, self.config)
            labels = to_cp_global(labels, self.config)

            observer = getattr(self, "_eval_token_loss_hook", None)
            if observer is not None and not _token_loss_replaying.get():
                observer(loss, labels)
            lossmask = labels != self.ignored_index
            if (~lossmask).all():
                return paddle.mean(loss) * 0.0

            lossmask = lossmask.reshape([-1]).cast(paddle.float32)
            loss = paddle.sum(
                loss.cast(paddle.float32).reshape([-1]) * lossmask
            )
            loss = self._reduce_loss_by_tokens(loss, lossmask.sum())
            return loss

        seq_len = logits.shape[1]

        # Loss-path MD5 probe: logits and labels before cross-entropy
        import os

        if (
            os.environ.get("LOG_LAYER_MD5", "0") == "1"
            or os.environ.get("LOG_LOSS_MD5", "0") == "1"
        ):
            import hashlib

            rank = paddle.distributed.get_rank()
            lg_md5 = hashlib.md5(
                logits.cast("float32").numpy().tobytes()
            ).hexdigest()
            lb_md5 = hashlib.md5(
                labels.cast("int64").numpy().tobytes()
            ).hexdigest()
            print(
                f"[LOSS_PATH_MD5] rank={rank} loss_input_logits shape={list(logits.shape)} md5={lg_md5}",
                flush=True,
            )
            print(
                f"[LOSS_PATH_MD5] rank={rank} loss_input_labels shape={list(labels.shape)} md5={lb_md5}",
                flush=True,
            )

        if self.use_subbatch and seq_len > self.loss_subbatch_sequence_length:

            def _cast_loss_func(logits, labels):
                return self.loss_func(logits.cast("float32"), labels)

            sb_loss_func = subbatch(
                _cast_loss_func,
                arg_idx=[0, 1],
                axis=[1, 1],
                bs=self.loss_subbatch_sequence_length,
                out_idx=1,
            )
            loss = sb_loss_func(logits, labels)
        else:
            if (
                self.config.gpt_model_use_experimental_version
                and self.config.sequence_parallel
            ):
                logits = logits.reshape([labels.shape[0], -1, logits.shape[-1]])
            loss = self.loss_func(logits.cast("float32"), labels)

        # Per-token: count this shard's tokens before the CP gather.
        if _per_token_count_enabled(self.config):
            accumulate_per_token_local_count(
                (labels != self.ignored_index).sum()
            )

        loss = to_cp_global(loss, self.config)
        labels = to_cp_global(labels, self.config)

        if _use_accuracy_compatible_kernel():
            # 定位锚点 1：CP gather 后、mask/归一化前的 per-token CE，
            # 两侧语义唯一，未掺入归一化差异。
            print(
                f"\nper_token_loss: rank={dist.get_rank()} "
                f"shape={list(loss.shape)} md5={loss.cast('float32')._md5sum()}",
                flush=True,
            )

        observer = getattr(self, "_eval_token_loss_hook", None)
        if observer is not None and not _token_loss_replaying.get():
            observer(loss, labels)
        lossmask = labels != self.ignored_index
        if (~lossmask).all():
            loss = paddle.mean(loss) * 0.0
        else:
            lossmask = lossmask.reshape([-1]).cast(paddle.float32)

            # Loss-path MD5 probe: per-token loss and lossmask
            if (
                os.environ.get("LOG_LAYER_MD5", "0") == "1"
                or os.environ.get("LOG_LOSS_MD5", "0") == "1"
            ):
                import hashlib

                rank = paddle.distributed.get_rank()
                pt_md5 = hashlib.md5(
                    loss.cast("float32").reshape([-1]).numpy().tobytes()
                ).hexdigest()
                lm_md5 = hashlib.md5(lossmask.numpy().tobytes()).hexdigest()
                valid_count = lossmask.sum().item()
                loss_sum_val = paddle.sum(
                    loss.cast("float32").reshape([-1]) * lossmask
                ).item()
                print(
                    f"[LOSS_PATH_MD5] rank={rank} per_token_loss md5={pt_md5}",
                    flush=True,
                )
                print(
                    f"[LOSS_PATH_MD5] rank={rank} lossmask md5={lm_md5} valid_tokens={valid_count}",
                    flush=True,
                )
                print(
                    f"[LOSS_PATH_MD5] rank={rank} loss_sum={loss_sum_val} final_loss={loss_sum_val / valid_count}",
                    flush=True,
                )
                # Also compute line-wise loss (matches EC's _line_wise_loss) for exact comparison
                if self.config.gpt_model_use_experimental_version:
                    _probe_loss_2d = loss.cast(
                        paddle.float32
                    ) * lossmask.reshape(labels.shape)
                    _probe_lm_2d = lossmask.reshape(labels.shape)
                    _probe_tc = _probe_lm_2d.sum(-1)
                    _probe_inv = (_probe_tc == 0).astype(paddle.float32)
                    _probe_lpl = _probe_loss_2d.sum(-1) / (
                        _probe_tc + 1e-6 * _probe_inv
                    )
                    _probe_lpl = _probe_lpl * (1 - _probe_inv)
                    _probe_lw = _probe_lpl.sum() / (
                        (1 - _probe_inv).sum() + 1e-6
                    )
                    print(
                        f"[LOSS_PATH_MD5] rank={rank} line_wise_loss={_probe_lw.item():.20f}",
                        flush=True,
                    )

            # EC-compat: line-wise loss (per-sample mean then average across samples)
            # EC's ErniemmPretrainingCriterion recomputes loss as line-wise when task_id
            # is present, which changes the value due to division by (count + 1e-6).
            if self.calculate_per_token_loss:
                # Per-token: raw masked sum; skip the mean and EC line-wise
                # reweighting, the trainer divides by T_global. Routed through
                # the helper so log_global_token_avg_loss still sees (S, n).
                loss = self._reduce_loss_by_tokens(
                    paddle.sum(
                        loss.cast(paddle.float32).reshape([-1]) * lossmask
                    ),
                    lossmask.sum(),
                )
            elif self.config.gpt_model_use_experimental_version:
                if max(get_tensor_model_parallel_world_size(), 1) > 1:
                    loss = loss.squeeze(-1)
                loss_2d = loss.cast(paddle.float32) * lossmask.reshape(
                    labels.shape
                )
                lossmask_2d = lossmask.reshape(labels.shape)
                token_count_per_line = lossmask_2d.sum(-1)
                is_invalid_line_float = (token_count_per_line == 0).astype(
                    paddle.float32
                )
                loss_per_line = loss_2d.sum(-1) / (
                    token_count_per_line + 1e-6 * is_invalid_line_float
                )
                loss_per_line = loss_per_line * (1 - is_invalid_line_float)
                loss = loss_per_line.sum() / (
                    (1 - is_invalid_line_float).sum() + 1e-6
                )
            else:
                if self.use_accuracy_compatible and not (
                    use_dsv4_accuracy_compatible()
                    and self.config.experimental_attention_variant
                    == "dsv4_hybrid"
                ):
                    _flat = loss.cast(paddle.float32).reshape([-1]) * lossmask
                    loss_sum = (
                        _flat.cast(paddle.float64).sum().cast(paddle.float32)
                    )
                    _count = lossmask.sum()
                    import paddle.distributed as _pdist

                    _pg_collection = getattr(self, "pg_collection", None)
                    _ep_group = getattr(_pg_collection, "ep", None)
                    if _ep_group is None:
                        _ep_group = get_expert_model_parallel_group(
                            check_initialized=False
                        )
                    _ep_size = (
                        _pdist.get_world_size(group=_ep_group)
                        if _ep_group is not None
                        else 1
                    )
                    _acc_sum = paddle.zeros([1], dtype=paddle.float32)
                    for _ in range(_ep_size):
                        _acc_sum = _acc_sum + loss_sum
                    loss = _acc_sum[0] / (_count * _ep_size)
                else:
                    loss = paddle.sum(
                        loss.cast(paddle.float32).reshape([-1]) * lossmask
                    )
                    loss = self._reduce_loss_by_tokens(loss, lossmask.sum())

        if _use_accuracy_compatible_kernel():
            # 定位锚点 2：mask + 归一化后的标量 loss，与锚点 1 配合可切开
            # 「CE 上游差异」和「lossmask / valid_token / 除法差异」。
            print(
                f"\nfinal_loss: rank={dist.get_rank()} "
                f"val={float(loss):.20f} md5={loss.cast('float32')._md5sum()}",
                flush=True,
            )

        return loss

    def _forward(self, logits: Tensor | tuple, labels: Tensor):
        # Consumer: ``labels`` is already in local coordinates, produced by
        # _labels_for_depth (or to_cp_local on the no-MTP path).
        if module_needs_recompute("loss_fn", None, self.config):
            # One lifetime per CE invocation, not a flag on the shared layer:
            # several heads/micro-batches may await backward simultaneously.
            # Do not retain token losses or their autograd graphs in the closure.
            forward_active = True
            # Pin the token-avg head and record decision now: a reentrant
            # replay runs inside backward, outside the caller's head scope and
            # with grad on, while its first pass ran under no_grad.
            head = _token_avg_head.get()
            record = paddle.is_grad_enabled()

            def loss_forward(logits, labels):
                token = _token_loss_replaying.set(
                    _token_loss_replaying.get() or not forward_active
                )
                head_token = _token_avg_head.set(head)
                record_token = _token_avg_record.set(record and forward_active)
                try:
                    return self.forward_impl(logits, labels)
                finally:
                    _token_avg_record.reset(record_token)
                    _token_avg_head.reset(head_token)
                    _token_loss_replaying.reset(token)

            try:
                return recompute(loss_forward, logits, labels)
            finally:
                forward_active = False
        return self.forward_impl(logits, labels)

    def _global_labels_for_depth(self, labels, depth):
        """Labels of one prediction head, in global coordinates.

        ``depth < 0`` is the main head, ``depth >= 0`` MTP depth ``depth``.
        This is the labels' global work, which needs the whole sequence:

        * ``use_erndata``: labels arrive length ``L``. MTP depth ``d`` rolls
          them left ``d + 1`` times per packed document (``cu_seqlens_q``
          stashed by the embedding / LM head), filling ``ignored_index`` at
          each boundary so no position trains on the next document.
        * otherwise (L+K contract): labels arrive length ``L + K``. The main
          head drops the trailing ``K``; depth ``d`` takes the window
          ``[d + 1, d + 1 + L)``.
        """
        if getattr(self.config, "use_erndata", False):
            if depth < 0:
                return labels
            _cu = LanguageLoss._cu_seqlens_q_stash
            if _cu is None:
                # A plain roll would leak labels across packed documents.
                raise RuntimeError(
                    "use_erndata=True requires cu_seqlens_q to be "
                    "stashed on LanguageLoss._cu_seqlens_q_stash before the loss "
                    "stage, but it is None on this rank. It should be set by "
                    "GPTEmbedding.forward (PP=1) or the LM head on the last PP "
                    "stage (GPTLMHead / GPTMainLMHead / GPTMTPLMHead)."
                )
            from paddlefleet.transformer.multi_token_prediction import (
                _roll_tensor_packed_seq,
            )

            for _ in range(depth + 1):
                labels, _ = _roll_tensor_packed_seq(
                    labels,
                    shifts=-1,
                    dims=1,
                    cu_seqlens_q=_cu,
                    pad_value=self.ignored_index,
                )
            return labels
        num_mtp = self.config.num_nextn_predict_layers
        if depth < 0:
            return labels[:, :-num_mtp]
        seq_length = labels.shape[1] - num_mtp
        return labels[:, (depth + 1) : (depth + 1 + seq_length)]

    def _labels_for_depth(self, labels, depth):
        """The only producer of MTP-path CE labels: global work, then local.

        See :meth:`_global_labels_for_depth` for the global work. The result
        matches the local logits of that head.
        """
        return to_cp_local(
            self._global_labels_for_depth(labels, depth), self.config
        )

    def forward(self, logits: Tensor | list, labels: Tensor) -> Tensor:
        if isinstance(logits, list):
            assert mtp_layers_active(self.config)
            assert len(logits) == self.config.num_nextn_predict_layers + 1
            labels_ori = labels
            # Under use_erndata labels_ori is sequence-global ([B, L]); the
            # legacy contract appends K trailing positions ([B, L + K]).
            _mtp_is_megatron = getattr(self.config, "use_erndata", False)
            lm_labels = self._labels_for_depth(labels_ori, -1)

            mtp_loss = []
            mtp_logits = logits[1:]
            per_token = self.calculate_per_token_loss

            # Per-token MTP renorm and display: full-sequence (pre-CP-extract)
            # token counts, matching the CP-gathered sums from forward_impl and
            # identical on every CP rank. The main head goes through
            # _global_labels_for_depth so the legacy L+K layout drops its
            # trailing K positions.
            lm_tok_full = (
                (
                    self._global_labels_for_depth(labels_ori, -1)
                    != self.ignored_index
                ).sum()
                if per_token
                else None
            )
            mtp_tok_full_list = []

            if not self.config.mtp_distillation_loss:
                if self.config.train_mtp_only:
                    lm_loss = 0.0
                else:
                    lm_loss = self._forward(logits[0], lm_labels)

                for depth in range(self.config.num_nextn_predict_layers):
                    # MTP depth sampling: the LM head emits None for depths that were
                    # not computed this step; skip them so mtp_loss holds only the K
                    # computed depths (the `/ num_mtp` reduction below then averages
                    # over K). No effect when sampling is off — logits are never None.
                    if mtp_logits[depth] is None:
                        continue
                    logits_cur_depth = mtp_logits[depth]
                    # Per-token: count each head's tokens in the full
                    # (pre-CP-extract) domain, matching the CP-gathered loss
                    # sums from forward_impl.
                    _lbl_full = self._global_labels_for_depth(labels_ori, depth)
                    if per_token:
                        mtp_tok_full_list.append(
                            (_lbl_full != self.ignored_index).sum()
                        )
                    labels_cur_depth = to_cp_local(_lbl_full, self.config)
                    # Per-token + fused CE: _forward already returns the
                    # CP-gathered raw masked sum, so the experimental per-token
                    # matrix reduction below has nothing to do (and would
                    # CP-gather that scalar under the legacy layout). Take the
                    # plain branch instead.
                    if self.config.gpt_model_use_experimental_version and not (
                        per_token and self.config.fused_linear_ce_loss_chunk > 0
                    ):
                        # Align with EB: compute per-token loss matrix and reduce
                        # with global sum/count instead of going through forward_impl
                        # which applies line-wise loss.

                        if self.config.fused_linear_ce_loss_chunk > 0:
                            # MTP head: keep its labels out of T_global.
                            with (
                                suppress_per_token_count(),
                                _token_avg_head_scope(
                                    _mtp_token_avg_prefix(depth)
                                ),
                            ):
                                loss_matrix_cur_depth = self._forward(
                                    logits_cur_depth,
                                    labels_cur_depth,
                                )
                        else:
                            if (
                                self.config.gpt_model_use_experimental_version
                                and self.config.sequence_parallel
                            ):
                                logits_cur_depth = logits_cur_depth.reshape(
                                    [
                                        labels_cur_depth.shape[0],
                                        -1,
                                        logits_cur_depth.shape[-1],
                                    ]
                                )
                            loss_matrix_cur_depth = self.loss_func(
                                logits_cur_depth.cast("float32"),
                                labels_cur_depth,
                            )

                        if not _mtp_is_megatron:
                            # Legacy dataflow normalises over the whole
                            # sequence. Under use_erndata labels stay local:
                            # the lossmask/sum reduction below is per-rank
                            # (allreduce happens implicitly via DP
                            # grad-averaging).
                            loss_matrix_cur_depth = to_cp_global(
                                loss_matrix_cur_depth, self.config
                            )
                            labels_cur_depth = to_cp_global(
                                labels_cur_depth, self.config
                            )

                        # Fused CE already notified inside _forward and returned
                        # a scalar. Only the direct, unreduced CE path needs this
                        # notification; callers may choose to consume main only.
                        if self.config.fused_linear_ce_loss_chunk <= 0:
                            observer = getattr(
                                self, "_eval_token_loss_hook", None
                            )
                            if (
                                observer is not None
                                and not _token_loss_replaying.get()
                            ):
                                observer(
                                    loss_matrix_cur_depth, labels_cur_depth
                                )

                        lossmask_cur_depth = (
                            labels_cur_depth != self.ignored_index
                        ).cast(paddle.float32)
                        loss_matrix_cur_depth = loss_matrix_cur_depth.cast(
                            paddle.float32
                        ).reshape([-1]) * lossmask_cur_depth.reshape([-1])
                        if per_token:
                            # Raw masked sum, as forward_impl returns; the
                            # trainer divides once by T_global.
                            loss_cur_depth = loss_matrix_cur_depth.sum()
                        elif lossmask_cur_depth.sum().item() > 0:
                            loss_cur_depth = (
                                loss_matrix_cur_depth.sum()
                                / lossmask_cur_depth.sum()
                            )
                        else:
                            loss_cur_depth = loss_matrix_cur_depth.sum() * 0.0
                    else:
                        # MTP head: keep its labels out of T_global.
                        with (
                            suppress_per_token_count(),
                            _token_avg_head_scope(_mtp_token_avg_prefix(depth)),
                        ):
                            loss_cur_depth = self._forward(
                                logits_cur_depth,
                                labels_cur_depth,
                            )
                    mtp_loss.append(loss_cur_depth)
            else:
                lm_loss = self._forward(logits[0], lm_labels)
                if get_tensor_model_parallel_world_size() > 1:
                    target_p_self_op_dist = DistributedSoftmaxOp.apply(
                        logits[0], axis=2
                    )
                else:
                    target_p_self_op_dist = nn.Softmax(axis=2)(logits[0])
                if get_context_parallel_world_size() > 1:
                    cp_balance_mode = self.config.cp_balance_mode
                    if cp_balance_mode == "contiguous_allgather":
                        target_p_self_op_dist = MTPDistillationLossShift.apply(
                            target_p_self_op_dist,
                            self.config.num_nextn_predict_layers,
                            mode=cp_balance_mode,
                        )
                    else:
                        target_p_self_op_dist = ContextParallelGatherOp.apply(
                            target_p_self_op_dist,
                            axis=1,
                            mode=cp_balance_mode,
                        )

                def padding(tensor, left=False, pad_len=1):
                    zeropadding = paddle.zeros_like(tensor[:, -pad_len:, :])
                    if left:
                        tensor = paddle.concat((zeropadding, tensor), axis=1)
                    else:
                        tensor = paddle.concat((tensor, zeropadding), axis=1)
                    return tensor

                if (
                    self.config.num_nextn_predict_layers > 0
                    and mtp_logits is not None
                ):
                    for depth in range(len(mtp_logits)):
                        prediction_scores_cur_depth = mtp_logits[depth]
                        # Global: the lossmask sum below is the global
                        # token count, and the mask enters local
                        # coordinates on its own.
                        labels_cur_depth = self._global_labels_for_depth(
                            labels_ori, depth
                        )
                        lossmask = (
                            labels_cur_depth != self.ignored_index
                        ).cast(paddle.float32)
                        if get_tensor_model_parallel_world_size() > 1:
                            out_logp = paddle.log(
                                DistributedSoftmaxOp.apply(
                                    prediction_scores_cur_depth, axis=2
                                )
                            )
                        else:
                            out_logp = nn.LogSoftmax(axis=2)(
                                prediction_scores_cur_depth
                            )

                        if not (
                            get_context_parallel_world_size() > 1
                            and cp_balance_mode == "contiguous_allgather"
                        ):
                            target_p = target_p_self_op_dist[
                                :, (depth + 1) :, :
                            ].clone()
                            target_p = padding(
                                target_p, left=False, pad_len=depth + 1
                            )
                        if get_context_parallel_world_size() > 1:
                            if cp_balance_mode == "contiguous_allgather":
                                target_p = target_p_self_op_dist[
                                    :, depth : depth + out_logp.shape[1]
                                ]
                            else:
                                target_p = ContextParallelScatterOp.apply(
                                    target_p,
                                    axis=1,
                                    mode=cp_balance_mode,
                                )
                        plogp = target_p * out_logp

                        lossmask = lossmask[..., None]
                        xishu = lossmask.sum() + 1e-5
                        lossmask = to_cp_local(lossmask, self.config)

                        ploss = -paddle.sum(lossmask * plogp)
                        if get_tensor_model_parallel_world_size() > 1:
                            dist.all_reduce(
                                ploss,
                                group=fleet.get_hybrid_communicate_group().get_model_parallel_group(),
                            )

                        if get_context_parallel_world_size() > 1:
                            dist.all_reduce(
                                ploss,
                                group=fleet.get_hybrid_communicate_group().get_context_parallel_group(),
                            )

                        ploss = ploss / xishu
                        mtp_loss.append(ploss)

            # Store detached MTP loss tensors into class-level tracker and global_training_logs.
            # Use .detach() instead of .item() to avoid GPU synchronization on every
            # micro-batch. The trainer will call .item() only at logging steps.
            # Per-token: mtp_loss[i] is a raw token sum; divide by the head's own
            # full-sequence token count so mtp_i_loss stays a per-token mean
            # (same as MainLanguageLoss.forward).
            if per_token and not self.config.mtp_distillation_loss:
                mtp_disp_vals = [
                    mtp_loss[i].detach()
                    / paddle.clip(
                        mtp_tok_full_list[i].astype("float32"), min=1.0
                    )
                    for i in range(len(mtp_loss))
                ]
            else:
                mtp_disp_vals = [loss_val.detach() for loss_val in mtp_loss]

            for i, disp_val in enumerate(mtp_disp_vals):
                LanguageLoss.mtp_loss_tracker[f"mtp_{i + 1}_loss"] = disp_val
                _print_scalar_loss_md5(
                    "MTP_LOSS_PATH_MD5",
                    f"mtp{i + 1}.final_loss",
                    mtp_loss[i],
                )

            logs = get_global_training_logs()
            if logs is not None and hasattr(logs, "update"):
                for i, disp_val in enumerate(mtp_disp_vals):
                    logs.update(**{f"mtp_{i + 1}_loss": disp_val})

            # MTP depth sampling: mtp_loss holds only the computed prefix
            # (depths 0..len-1). Drop stale tracker entries for the sampled-out deeper
            # depths so the trainer does not re-log a stale value on those steps —
            # per-depth curves stay sparse-but-correct instead of flat.
            for _d in range(
                len(mtp_loss), self.config.num_nextn_predict_layers
            ):
                LanguageLoss.mtp_loss_tracker.pop(f"mtp_{_d + 1}_loss", None)

            def add_loss(main_loss, loss):
                if _use_accuracy_compatible_kernel():
                    # Megatron-aligned: MTP loss gradient flows but loss scalar unchanged.
                    # This matches Megatron's behavior where MTP contributes to training
                    # gradients without affecting the reported loss value.
                    if self.config.add_mtp_loss:
                        if use_dsv4_accuracy_compatible():
                            return loss - loss.detach() + main_loss
                        return main_loss + loss - loss.detach()
                    else:
                        return main_loss
                else:
                    # Original behavior
                    if self.config.add_mtp_loss:
                        return main_loss + loss
                    else:
                        return main_loss + loss - loss.detach()

            def renorm_mtp_head(idx, mtp_l):
                """Per-token: scale MTP head `idx` by main/own token count so that,
                after the trainer's 1/T_global, it is a mean over its own tokens
                (Megatron process_mtp_loss). Identity otherwise.
                """
                if not per_token:
                    return mtp_l
                return mtp_l * (
                    lm_tok_full.astype("float32")
                    / paddle.clip(
                        mtp_tok_full_list[idx].astype("float32"), min=1.0
                    )
                )

            if self.config.gpt_model_use_experimental_version:
                # Align with EB: accumulate inside loop to match float32
                # arithmetic order: loss += scaling * loss_i / N
                loss = lm_loss
                if _use_accuracy_compatible_kernel():
                    # Megatron-aligned: only add MTP loss when add_mtp_loss=True.
                    # Use add_loss() to keep single maintenance point for compat
                    # behavior (loss + val - val.detach() for gradient-only flow).
                    if self.config.add_mtp_loss:
                        num_mtp = len(mtp_loss)
                        for i, mtp_l in enumerate(mtp_loss):
                            mtp_val = (
                                self.config.mtp_loss_scaling_factor
                                * renorm_mtp_head(i, mtp_l)
                                / num_mtp
                            )
                            loss = add_loss(loss, mtp_val)
                else:
                    # Original behavior: always use add_loss
                    num_mtp = len(mtp_loss)
                    for i, mtp_l in enumerate(mtp_loss):
                        loss = add_loss(
                            loss,
                            self.config.mtp_loss_scaling_factor
                            * renorm_mtp_head(i, mtp_l)
                            / num_mtp,
                        )
            else:
                loss = add_loss(
                    lm_loss,
                    self.config.mtp_loss_scaling_factor
                    * sum(
                        renorm_mtp_head(i, mtp_l)
                        for i, mtp_l in enumerate(mtp_loss)
                    )
                    / len(mtp_loss),
                )
            # Per-token: T_global already spans all acc microbatches, so skip
            # the 1/acc_steps scaling.
            if use_dsv4_accuracy_compatible() and not per_token:
                loss = LossScaleBeforeBackward.scale(loss)

            # Skipped for distillation, whose mtp_loss is a mean, not a sum.
            # Full-sequence counts: the sums are CP-gathered.
            if per_token and not self.config.mtp_distillation_loss:
                self._record_per_token_display(
                    lm_loss, lm_tok_full, mtp_loss, mtp_tok_full_list
                )

            return loss
        else:
            # MTP inactive: the labels have no global work to do.
            loss = self._forward(logits, to_cp_local(labels, self.config))
            if self.calculate_per_token_loss:
                # Per-token: no 1/acc_steps scaling (see above). Count on the
                # full-sequence labels, matching the CP-gathered sum from
                # forward_impl. Labels arrive global except under the
                # loader-sharded source, which pre-shards them per CP rank.
                if cp_shard_source(self.config) == "loader":
                    labels = to_cp_global(labels, self.config)
                self._record_per_token_display(
                    loss, (labels != self.ignored_index).sum(), [], []
                )
            elif use_dsv4_accuracy_compatible():
                loss = LossScaleBeforeBackward.scale(loss)
            return loss

    def _record_per_token_display(
        self, lm_loss, lm_tok, mtp_loss, mtp_tok_list
    ):
        """Stash detached per-token display stats (never affect grads).

        lm_loss / mtp_loss[i] are raw masked sums from forward_impl and lm_tok /
        mtp_tok_list[i] their token counts. Accumulated into token_avg_tracker as
        float64 / int64 device tensors (same arithmetic as python floats, no
        D2H sync); popped by pop_per_token_stats.
        """

        def _f64(x):
            if isinstance(x, paddle.Tensor):
                return x.detach().astype("float64").reshape([])
            return paddle.to_tensor(float(x), dtype="float64")

        # lm head (lm_loss is 0.0 when train_mtp_only).
        lm_sum = _f64(lm_loss)
        lm_tok_f = _f64(lm_tok)
        _per_token_tracker_add("pt_lm_sum", lm_sum)
        _per_token_tracker_add("pt_lm_tok", lm_tok_f)
        lm_mean = lm_sum / paddle.clip(lm_tok_f, min=1.0)

        # mtp heads, aggregated over depths.
        mtp_sum = None
        mtp_tok = None
        mtp_means = []
        for ml, mt in zip(mtp_loss, mtp_tok_list):
            s = _f64(ml)
            t = _f64(mt)
            mtp_sum = s if mtp_sum is None else mtp_sum + s
            mtp_tok = t if mtp_tok is None else mtp_tok + t
            mtp_means.append(s / paddle.clip(t, min=1.0))
        if mtp_means:
            _per_token_tracker_add("pt_mtp_sum", mtp_sum)
            _per_token_tracker_add("pt_mtp_tok", mtp_tok)

        # Old-caliber combined `loss`: value includes mtp only when add_loss puts it
        # in the value (add_mtp_loss and not the accuracy-compatible kernel).
        combined = lm_mean
        if (
            mtp_means
            and self.config.add_mtp_loss
            and not _use_accuracy_compatible_kernel()
        ):
            combined = lm_mean + self.config.mtp_loss_scaling_factor * (
                sum(mtp_means) / len(mtp_means)
            )
        _per_token_tracker_add("pt_mom_sum", combined)
        _per_token_tracker_add("pt_mom_cnt", 1)

    def build_schedule_node(self):
        return ScheduleNode(self.forward, name="LanguageLoss")


class MainLanguageLoss(LanguageLoss):
    # Class-level tracker for MTP loss, read by trainer for logging.
    mtp_loss_tracker: dict[str, float] = {}

    def __init__(
        self,
        config: TransformerConfig,
        pg_collection=None,
    ) -> None:
        super().__init__(config=config, pg_collection=pg_collection)

    def forward(self, dict_args: dict | list, labels: Tensor) -> Tensor:
        assert mtp_layers_active(self.config)
        lm_labels = self._labels_for_depth(labels, -1)

        mtp_loss = dict_args["mtp_loss"]
        logits = dict_args["logits"]

        assert not self.config.mtp_distillation_loss, (
            "separate mtp head & loss don't support mtp_distillation_loss"
        )

        if self.config.train_mtp_only:
            lm_loss = 0.0
        else:
            lm_loss = self._forward(logits, lm_labels)

        # Per-token: full-sequence token counts of the main head and each MTP
        # head (same slice/roll as MTPLanguageLoss), used for the mtp_i_loss
        # display and renorm_mtp_head.
        per_token = self.calculate_per_token_loss
        mtp_tok_full_list = []
        lm_tok_full = None
        if per_token:
            lm_tok_full = (
                self._global_labels_for_depth(labels, -1) != self.ignored_index
            ).sum()
            for depth in range(len(mtp_loss)):
                labels_full_depth = self._global_labels_for_depth(labels, depth)
                mtp_tok_full_list.append(
                    (labels_full_depth != self.ignored_index).sum()
                )

        # Store detached MTP loss tensors into class-level tracker and global_training_logs.
        # Use .detach() instead of .item() to avoid GPU synchronization on every
        # micro-batch. The trainer will call .item() only at logging steps.
        if per_token:
            mtp_disp_vals = [
                mtp_loss[i].detach()
                / paddle.clip(mtp_tok_full_list[i].astype("float32"), min=1.0)
                for i in range(len(mtp_loss))
            ]
        else:
            mtp_disp_vals = [loss_val.detach() for loss_val in mtp_loss]

        for i, disp_val in enumerate(mtp_disp_vals):
            MainLanguageLoss.mtp_loss_tracker[f"mtp_{i + 1}_loss"] = disp_val
            _print_scalar_loss_md5(
                "MTP_LOSS_PATH_MD5",
                f"mtp{i + 1}.final_loss",
                mtp_loss[i],
            )

        # Also write to global_training_logs to read
        logs = get_global_training_logs()
        if logs is not None and hasattr(logs, "update"):
            for i, disp_val in enumerate(mtp_disp_vals):
                logs.update(**{f"mtp_{i + 1}_loss": disp_val})

        def add_loss(main_loss, loss):
            if _use_accuracy_compatible_kernel():
                # Megatron-aligned: MTP loss gradient flows but loss scalar unchanged.
                # This matches Megatron's behavior
                if self.config.add_mtp_loss:
                    return main_loss + loss - loss.detach()
                else:
                    return main_loss
            else:
                # Original behavior
                if self.config.add_mtp_loss:
                    return main_loss + loss
                else:
                    return main_loss + loss - loss.detach()

        def renorm_mtp_head(idx, mtp_l):
            """Same as LanguageLoss.forward's renorm_mtp_head."""
            if not per_token:
                return mtp_l
            return mtp_l * (
                lm_tok_full.astype("float32")
                / paddle.clip(mtp_tok_full_list[idx].astype("float32"), min=1.0)
            )

        loss = add_loss(
            lm_loss,
            self.config.mtp_loss_scaling_factor
            * sum(renorm_mtp_head(i, mtp_l) for i, mtp_l in enumerate(mtp_loss))
            / len(mtp_loss),
        )

        if per_token:
            self._record_per_token_display(
                lm_loss, lm_tok_full, mtp_loss, mtp_tok_full_list
            )

        return loss

    def build_schedule_node(self):
        return ScheduleNode(self.forward, name="MainLanguageLoss")


class MTPLanguageLoss(LanguageLoss):
    def __init__(
        self,
        config: TransformerConfig,
        pg_collection=None,
    ) -> None:
        super().__init__(config=config, pg_collection=pg_collection)

    def forward(self, dict_args: dict):
        mtp_logits = dict_args.get("mtp_logits")
        labels = dict_args.get("labels")
        assert mtp_logits is not None, (
            "separate mtp loss must provide mtp_logits"
        )
        assert labels is not None, "separate mtp loss must provide labels"
        assert mtp_layers_active(self.config)
        mtp_loss = []

        assert not self.config.mtp_distillation_loss, (
            "separate mtp head & loss don't support mtp_distillation_loss"
        )

        for depth in range(self.config.num_nextn_predict_layers):
            logits_cur_depth = mtp_logits[depth]
            labels_cur_depth = self._labels_for_depth(labels, depth)
            # MTP head: keep its labels out of T_global.
            with (
                suppress_per_token_count(),
                _token_avg_head_scope(_mtp_token_avg_prefix(depth)),
            ):
                loss_cur_depth = self._forward(
                    logits_cur_depth,
                    labels_cur_depth,
                )
            mtp_loss.append(loss_cur_depth)

        dict_args.pop("mtp_logits")
        dict_args["mtp_loss"] = mtp_loss

        return dict_args

    def build_schedule_node(self):
        return ScheduleNode(self.forward, name="MTPLanguageLoss")


class DSparkLanguageLoss(LanguageLoss):
    """Compute and record the DSpark draft loss."""

    def __init__(
        self,
        config: TransformerConfig,
        pg_collection=None,
    ) -> None:
        super().__init__(config=config, pg_collection=pg_collection)

    def forward(self, dict_args: dict):
        dspark_out = dict_args.get("dspark_out")
        if dspark_out is None:
            return dict_args
        from paddlefleet.transformer.parallel_multi_token_prediction import (
            DSparkLossAutoScaler,
            DSparkLossLoggingHelper,
        )
        from paddlefleet.utils.dspark.dspark_losses import (
            LossProfile,
            compute_dspark_loss,
        )
        from paddlefleet.utils.dspark.dspark_utils import DrafterOutput

        output = DrafterOutput(
            draft_logits=dspark_out.get("draft_logits"),
            target_ids=dspark_out["target_ids"],
            eval_mask=dspark_out["eval_mask"],
            block_keep_mask=dspark_out["block_keep_mask"],
            confidence_pred=dspark_out.get("confidence_pred"),
            draft_hidden_states=dspark_out.get("draft_hidden_states"),
            aligned_target_hidden_states=dspark_out.get(
                "aligned_target_hidden_states"
            ),
            previous_token_ids=dspark_out.get("previous_token_ids"),
            logits_projection=dspark_out.get("logits_projection"),
            markov_head=dspark_out.get("markov_head"),
            candidate_selector=dspark_out.get("candidate_selector"),
        )
        # CP>1: the draft sees only this rank's window, so its ``weights.sum()``
        # denominator is a 1/cp_size partial of the global token count, while
        # the trainer scales non-expert grads by cp_worldsize -- together a
        # cp_size-times-too-large draft gradient. All-reducing the denominator
        # restores the CP=1 gradient. Skipped at cp_size==1 (no-op group) to
        # keep CP=1 numerics identical.
        normalization_group = None
        if get_context_parallel_world_size() > 1:
            normalization_group = get_context_parallel_group()
        dspark_terms = compute_dspark_loss(
            output,
            LossProfile(),
            return_terms=True,
            normalization_group=normalization_group,
        )
        dspark_loss = dspark_terms.loss
        scaled = (
            float(getattr(self.config, "dspark_loss_scaling_factor", 1.0))
            * dspark_loss
        )
        DSparkLossLoggingHelper.save_loss_to_tracker(scaled)
        dict_args["dspark_loss"] = scaled
        _publish_dspark_loss(scaled)
        _publish_dspark_terms(dspark_terms)
        # Carry the draft gradient through the main loss path.
        if getattr(self.config, "add_dspark_loss", False) and (
            "hidden_states" in dict_args
        ):
            dict_args["hidden_states"] = DSparkLossAutoScaler.apply(
                dict_args["hidden_states"], scaled
            )
        dict_args.pop("dspark_out", None)
        return dict_args

    def build_schedule_node(self):
        return ScheduleNode(self.forward, name="DSparkLanguageLoss")
