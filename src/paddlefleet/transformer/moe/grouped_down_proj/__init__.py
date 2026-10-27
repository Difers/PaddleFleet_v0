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
"""Grouped down projection for the RingMoE dispatcher.

Drives the grouped BF16 down projection of the sonic package
(``paddlefleet_ops.sonicmoe.functional.grouped_down_proj``) from the ring:
every round's routing is turned into a row-group plan on the GPU and the
round's expert call runs the grouped down projection with that plan.  With
sonic's per-piece plans (``...functional.grouped_plan_pieces``) each rank
selects for its own tokens once per forward, the result travels with its
routing pair, and a round only merges the node's pieces; otherwise every
round is planned with ``...functional.grouped_plan_gpu``.  Enabled by
``TransformerConfig.moe_ring_grouped_down_proj_list``.

- ``ring``: per-layer runtime used by ``RingMoETokenDispatcher``; the sonic
  modules are imported on first use.
"""
