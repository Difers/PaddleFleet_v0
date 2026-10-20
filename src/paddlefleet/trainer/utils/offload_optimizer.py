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

import paddle
from paddle import _C_ops
from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer.hybrid_parallel_optimizer import (
    HybridParallelOptimizer,
)
from paddle.optimizer import Optimizer

from .sharding_io import to_device


def offload(tensor):
    if paddle.is_compiled_with_cuda():
        place = paddle.CUDAPinnedPlace()
    elif paddle.is_compiled_with_xpu():
        place = paddle.XPUPinnedPlace()
    else:
        place = paddle.CPUPlace()

    new_tensor = to_device(tensor, place)
    assert new_tensor is tensor, "to_device must be inplace operation"


def reload(tensor):
    new_tensor = to_device(tensor)
    assert new_tensor is tensor, "to_device must be inplace operation"


def _inner_param_states(opt, param):
    """Every state tensor *opt* keeps for *param*: accumulators + master weight.

    Generic over the state layout, so it covers any ``InnerOptimizer``
    subclass (moment1/moment2/beta_pow, square_avg/momentum_buffer, ...).
    """
    states = [
        accs[param.name]
        for accs in opt._accumulators.values()
        if param.name in accs
    ]
    master = opt._master_weights.get(param.name)
    if isinstance(master, paddle.Tensor):
        states.append(master)
    return states


def _reload_inner_states(opt, params):
    for param in params:
        for t in _inner_param_states(opt, param):
            reload(t)


def _offload_inner_states(opt, params):
    for param in params:
        if getattr(param, "is_offload_opt", True):
            for t in _inner_param_states(opt, param):
                offload(t)


def _hack_offload_inner_optimizers():
    """Reload/offload the state of mixed-optimizer sub-optimizers.

    ``InnerOptimizer`` subclasses (``use_mixed_optimizer``) mostly update in
    pure Python, bypassing the ``_C_ops.adamw_`` patch, and their master
    weights are not created through ``_add_accumulator``. Around every update
    unit the touched parameters' accumulators and master weights are reloaded
    to the device and offloaded again afterwards. The unit is chosen from the
    optimizer's interface, never from its concrete class:

    - a class defining ``_update_shape_group(group_params_grads, ...)``
      (Muon family): one batch, which must read/write all of its state inside
      that call;
    - any other ``InnerOptimizer``: one ``_update_parameters`` call, which is
      correct for any update math.

    Only *when* states are moved changes, never the values the update reads,
    so offload on/off stays bit-identical.

    Like Step 4 (Muon), the offload gate defaults to ``is_offload_opt=True`` in
    both ``hack_offload_optimizer`` and ``hack_offload_optimizer_eb5``: these
    pure-Python updates follow Step 4, not the eb5 ``False`` default of the
    ``_C_ops.adamw_`` / ``_insert_sync`` patches (Steps 2/3).
    """
    try:
        from paddle.optimizer.inner_optimizer import InnerOptimizer
    except ImportError:
        return

    origin_apply_optimize = InnerOptimizer._apply_optimize

    def new_apply_optimize(self, loss, startup_program, params_grads):
        # Wrapped after the template's _ensure_accumulators (newly created
        # accumulators are already offloaded by Step 1) and before any read.
        name = (
            "_update_shape_group"
            if hasattr(type(self), "_update_shape_group")
            else "_update_parameters"
        )
        cls_fn = getattr(type(self), name)

        def wrapped(group_params_grads, *args, **kwargs):
            params = [param for param, _ in group_params_grads]
            _reload_inner_states(self, params)
            ret = cls_fn(self, group_params_grads, *args, **kwargs)
            _offload_inner_states(self, params)
            return ret

        # Shadow on the instance for this call only, so subclasses registered
        # later are covered without touching this file.
        setattr(self, name, wrapped)
        try:
            return origin_apply_optimize(
                self, loss, startup_program, params_grads
            )
        finally:
            # pop, not del: when patched twice the inner wrapper has already
            # removed the attribute (keeps the hack idempotent like Steps 1-4).
            self.__dict__.pop(name, None)

    InnerOptimizer._apply_optimize = new_apply_optimize


def hack_offload_optimizer(mode=None):
    if mode == "eb5":
        return hack_offload_optimizer_eb5()

    # Step 1: mock _add_accumulator
    origin_add_accumulator = getattr(Optimizer, "_add_accumulator")

    def new_add_accumulator(self, *args, **kwargs):
        x = origin_add_accumulator(self, *args, **kwargs)
        offload(x)
        return x

    setattr(Optimizer, "_add_accumulator", new_add_accumulator)

    # Step 2: mock _C_ops.adamw_ and _C_ops.adamw
    for name in ["adam_", "adamw_"]:
        origin_op = getattr(_C_ops, name)

        def new_opt_op(*args):
            for arg in args:
                if isinstance(arg, paddle.Tensor):
                    reload(arg)

            ret = origin_op(*args)

            is_offload_opt = getattr(args[0], "is_offload_opt", True)
            for i, arg in enumerate(args):
                if (
                    i >= 2 and isinstance(arg, paddle.Tensor) and is_offload_opt
                ):  # do not offload parameter and gradient
                    offload(arg)
            return ret

        setattr(_C_ops, name, new_opt_op)

    # Step 3: mock _insert_sync
    opt_type = HybridParallelOptimizer
    origin_insert_sync = getattr(opt_type, "_insert_sync")

    def new_insert_sync(self, sync_var, *args, **kwargs):
        origin_place = sync_var.place
        reload(sync_var)
        ret = origin_insert_sync(self, sync_var, *args, **kwargs)
        is_offload_opt = getattr(sync_var, "is_offload_opt", True)
        if is_offload_opt:
            new_sync_var = to_device(sync_var, origin_place)
        else:
            new_sync_var = sync_var
        assert new_sync_var is sync_var, "to_device must be inplace operation"
        return ret

    setattr(opt_type, "_insert_sync", new_insert_sync)

    # Step 4: mock Muon's momentum update
    # Muon's momentum update is pure Python (paddle.lerp + paddle.assign),
    # so it bypasses the _C_ops.adamw_ patch above. We need explicit
    # reload/offload for Muon's momentum_buffer and master_weights.
    try:
        from paddle.optimizer.muon import Muon

        # 4a: Patch the Muon momentum update. Newer Paddle provides the
        # batched _muon_update_group, while stable releases (e.g.
        # release/3.3) still have the per-param _muon_update. Prefer
        # patching _muon_update_group and fall back to _muon_update.
        if hasattr(Muon, "_muon_update_group"):
            # Batched momentum + master_weight offload. _muon_update_group
            # fetches momentum buffers and master weights internally, so
            # reload them for every param in the group before the update and then
            # offload them back afterwards. This keeps the master_weight
            # residency scoped to a single update_group (same granularity as
            # the first-order momentum), instead of loading every master
            # weight at once in _apply_optimize.
            origin_muon_update_group = Muon._muon_update_group

            def new_muon_update_group(
                self, group_params_grads, *args, **kwargs
            ):
                mw_dict = getattr(self, "_master_weights", None)
                for param, _ in group_params_grads:
                    momentum_buffer = self._get_accumulator(
                        self._moment_acc_str, param
                    )
                    reload(momentum_buffer)
                    if mw_dict:
                        mw = mw_dict.get(param.name)
                        if mw is not None and isinstance(mw, paddle.Tensor):
                            reload(mw)
                ret = origin_muon_update_group(
                    self, group_params_grads, *args, **kwargs
                )

                for param, _ in group_params_grads:
                    is_offload_opt = getattr(param, "is_offload_opt", True)
                    if is_offload_opt:
                        momentum_buffer = self._get_accumulator(
                            self._moment_acc_str, param
                        )
                        offload(momentum_buffer)
                        if mw_dict:
                            mw = mw_dict.get(param.name)
                            if mw is not None and isinstance(mw, paddle.Tensor):
                                offload(mw)
                return ret

            Muon._muon_update_group = new_muon_update_group
        elif hasattr(Muon, "_muon_update"):
            # Per-param momentum + master_weight offload for older Paddle
            # where _muon_update receives momentum_buffer as an argument.
            origin_muon_update = Muon._muon_update

            def new_muon_update(
                self,
                param,
                grad,
                lr,
                momentum_buffer,
                momentum_beta,
                ns_steps,
                nesterov,
                epsilon,
                weight_decay,
                version,
            ):
                mw_dict = getattr(self, "_master_weights", None)
                mw = mw_dict.get(param.name) if mw_dict else None
                reload(momentum_buffer)
                if mw is not None and isinstance(mw, paddle.Tensor):
                    reload(mw)
                ret = origin_muon_update(
                    self,
                    param,
                    grad,
                    lr,
                    momentum_buffer,
                    momentum_beta,
                    ns_steps,
                    nesterov,
                    epsilon,
                    weight_decay,
                    version,
                )
                is_offload_opt = getattr(param, "is_offload_opt", True)
                if is_offload_opt:
                    offload(momentum_buffer)
                    if mw is not None and isinstance(mw, paddle.Tensor):
                        offload(mw)
                return ret

            Muon._muon_update = new_muon_update

    except ImportError:
        pass

    # Step 5: mixed-optimizer sub-optimizers (use_mixed_optimizer)
    _hack_offload_inner_optimizers()


def hack_offload_optimizer_eb5():
    # Step 1: mock _add_accumulator
    origin_add_accumulator = getattr(Optimizer, "_add_accumulator")

    def new_add_accumulator(self, *args, **kwargs):
        x = origin_add_accumulator(self, *args, **kwargs)
        offload(x)
        return x

    setattr(Optimizer, "_add_accumulator", new_add_accumulator)

    # Step 2: mock _C_ops.adamw_ and _C_ops.adamw
    for name in ["adam_", "adamw_"]:
        origin_op = getattr(_C_ops, name)

        def new_opt_op(*args):
            for arg in args:
                if isinstance(arg, paddle.Tensor):
                    reload(arg)

            ret = origin_op(*args)
            is_offload_opt = getattr(args[0], "is_offload_opt", False)
            for i, arg in enumerate(args):
                if (
                    i >= 2 and isinstance(arg, paddle.Tensor) and is_offload_opt
                ):  # do not offload parameter and gradient
                    offload(arg)
            return ret

        setattr(_C_ops, name, new_opt_op)

    # Step 3: mock _insert_sync
    opt_type = HybridParallelOptimizer
    origin_insert_sync = getattr(opt_type, "_insert_sync")

    def new_insert_sync(self, sync_var, *args, **kwargs):
        origin_place = sync_var.place
        reload(sync_var)
        ret = origin_insert_sync(self, sync_var, *args, **kwargs)
        is_offload_opt = getattr(sync_var, "is_offload_opt", False)
        if is_offload_opt:
            new_sync_var = to_device(sync_var, origin_place)
        else:
            new_sync_var = sync_var
        assert new_sync_var is sync_var, "to_device must be inplace operation"
        return ret

    setattr(opt_type, "_insert_sync", new_insert_sync)

    # Step 4: mock Muon's momentum update
    # Muon's momentum update is pure Python (paddle.lerp + paddle.assign),
    # so it bypasses the _C_ops.adamw_ patch above. We need explicit
    # reload/offload for Muon's momentum_buffer and master_weights.
    try:
        from paddle.optimizer.muon import Muon

        # 4a: Patch the Muon momentum update. Newer Paddle provides the
        # batched _muon_update_group, while stable releases (e.g.
        # release/3.3) still have the per-param _muon_update. Prefer
        # patching _muon_update_group and fall back to _muon_update.
        if hasattr(Muon, "_muon_update_group"):
            # Batched momentum + master_weight offload. _muon_update_group
            # fetches momentum buffers and master weights internally, so
            # reload them for every param in the group before the update and then
            # offload them back afterwards. This keeps the master_weight
            # residency scoped to a single update_group (same granularity as
            # the first-order momentum), instead of loading every master
            # weight at once in _apply_optimize.
            origin_muon_update_group = Muon._muon_update_group

            def new_muon_update_group(
                self, group_params_grads, *args, **kwargs
            ):
                mw_dict = getattr(self, "_master_weights", None)
                for param, _ in group_params_grads:
                    momentum_buffer = self._get_accumulator(
                        self._moment_acc_str, param
                    )
                    reload(momentum_buffer)
                    if mw_dict:
                        mw = mw_dict.get(param.name)
                        if mw is not None and isinstance(mw, paddle.Tensor):
                            reload(mw)

                ret = origin_muon_update_group(
                    self, group_params_grads, *args, **kwargs
                )

                for param, _ in group_params_grads:
                    is_offload_opt = getattr(param, "is_offload_opt", True)
                    if is_offload_opt:
                        momentum_buffer = self._get_accumulator(
                            self._moment_acc_str, param
                        )
                        offload(momentum_buffer)
                        if mw_dict:
                            mw = mw_dict.get(param.name)
                            if mw is not None and isinstance(mw, paddle.Tensor):
                                offload(mw)
                return ret

            Muon._muon_update_group = new_muon_update_group
        elif hasattr(Muon, "_muon_update"):
            # Per-param momentum + master_weight offload for older Paddle
            # where _muon_update receives momentum_buffer as an argument.
            origin_muon_update = Muon._muon_update

            def new_muon_update(
                self,
                param,
                grad,
                lr,
                momentum_buffer,
                momentum_beta,
                ns_steps,
                nesterov,
                epsilon,
                weight_decay,
                version,
            ):
                mw_dict = getattr(self, "_master_weights", None)
                mw = mw_dict.get(param.name) if mw_dict else None
                reload(momentum_buffer)
                if mw is not None and isinstance(mw, paddle.Tensor):
                    reload(mw)
                ret = origin_muon_update(
                    self,
                    param,
                    grad,
                    lr,
                    momentum_buffer,
                    momentum_beta,
                    ns_steps,
                    nesterov,
                    epsilon,
                    weight_decay,
                    version,
                )
                is_offload_opt = getattr(param, "is_offload_opt", True)
                if is_offload_opt:
                    offload(momentum_buffer)
                    if mw is not None and isinstance(mw, paddle.Tensor):
                        offload(mw)
                return ret

            Muon._muon_update = new_muon_update

    except ImportError:
        pass

    # Step 5: mixed-optimizer sub-optimizers (use_mixed_optimizer)
    _hack_offload_inner_optimizers()
