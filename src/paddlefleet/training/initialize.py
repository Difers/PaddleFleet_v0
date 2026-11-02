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

import logging

import paddle.distributed as dist
from paddle.distributed import fleet

from paddlefleet import parallel_state as ps
from paddlefleet.training.arguments import parse_args
from paddlefleet.training.global_vars import set_global_variables


def initialize_fleet(
    strategy: fleet.DistributedStrategy,
    parsed_args=None,
    enable_dynamic_context_parallel: bool | None = None,
    max_dynamic_cp_size: int | None = None,
    **kwargs,
):
    # Parse arguments
    if parsed_args is None:
        args = parse_args(**kwargs, ignore_unknown_args=True)
    else:
        args = parsed_args

    set_global_variables(args)
    set_logging(args)

    fleet.init(is_collective=True, strategy=strategy)
    hcg = fleet.get_hybrid_communicate_group()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Dynamic context parallel: pre-build the multi-size CP groups. Explicit
    # arguments win over yaml/CLI args, so callers that do not go through
    # parse_args (e.g. PaddleFormers' training_args) can still turn it on.
    # Off by default -- nothing extra is created unless asked.
    if enable_dynamic_context_parallel is None:
        enable_dynamic_context_parallel = getattr(
            args, "enable_dynamic_context_parallel", False
        )
    if max_dynamic_cp_size is None:
        max_dynamic_cp_size = getattr(args, "max_dynamic_cp_size", None)
    # -1 is the "unset" value on the PaddleFormers side; it means "use the whole
    # rank pool", which is what None means downstream.
    if max_dynamic_cp_size is not None and max_dynamic_cp_size <= 0:
        max_dynamic_cp_size = None

    ps.initialize_model_parallel(
        hcg,
        enable_dynamic_context_parallel=enable_dynamic_context_parallel,
        max_dynamic_context_parallel_size=max_dynamic_cp_size,
    )
    # The DCP group table is logged by
    # parallel_state.initialize_dynamic_context_parallel_groups itself, which is
    # where all the numbers live -- don't duplicate it here.
    print(f"fleet initialize successfully: {rank=} {world_size=}")


def set_logging(args):
    """Set the logging level and logging format"""
    logger = logging.getLogger("paddlefleet")

    # set logging level
    logging_level = getattr(args, "logging_level", logging.INFO)
    logger.setLevel(logging_level)

    # set logging format
    import colorlog

    format_string = "%(log_color)s[%(asctime)-15s] [%(levelname)8s] %(filename)s:%(lineno)d%(reset)s - %(message)s"
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(format_string))
    logger.addHandler(handler)
