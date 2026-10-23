// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// record_stream.h -- tell Paddle's allocator that a comm stream uses a tensor,
// and withdraw that once the calc stream has waited for the work.
//
// The zcomm ops enqueue copies and kernels on a comm stream directly, not
// through a ProcessGroup. Paddle's stream-safe allocator orders a freed block
// only after the stream that allocated it (the calc stream), so a tensor the
// comm stream still reads or writes could be handed to a new calc-stream
// tensor as soon as its last reference is dropped. `record()` registers the
// comm stream on the tensor's allocation after the work is enqueued (as
// ProcessGroupNCCL does): a block freed before that work is done is then
// reclaimed only once it is.
//
// Left registered, every such block waits for the comm stream when freed even
// when the calc stream already waited for it, and the pool grows to cover the
// blocks in limbo (+3..7 GiB per GPU measured). So each call's registrations
// form a token, and `release(token, wait_stream)` withdraws them right after
// `wait_stream` waited for the call. A registration is withdrawn only if
//   - the allocation is still alive (once freed its own events govern it),
//   - it is the latest registration of that allocation on that comm stream
//     (comm-stream work runs in order, so waiting for the latest call covers
//     the earlier ones, while waiting for an earlier one does not cover a
//     later one; Paddle keeps one event per stream, so withdrawing on an
//     earlier wait would drop the later call's protection), and
//   - `wait_stream` is the allocation's own stream (the one the allocator
//     orders reuse after).
// Anything else keeps the registration: the block is then just reclaimed
// later. A token that is never released keeps its registrations, so the only
// way to get it wrong is to release before the wait, which the caller does in
// one place (`_Task.wait`). Resident symmetric buffers and their views are not
// allocator-owned and are skipped.
#pragma once

#include <cuda_runtime.h>
#include <paddle/extension.h>

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/memory/malloc.h"

namespace zcomm_ops {

struct Registration {
  std::weak_ptr<phi::Allocation> alloc;
  cudaStream_t stream;
  uint64_t seq;
};

struct LatestReg {
  std::weak_ptr<phi::Allocation> alloc;  // tells a reused address apart
  uint64_t seq;
};

struct RecordRegistry {
  std::mutex mu;
  // (allocation, comm stream) -> its latest registration
  std::map<std::pair<const void*, cudaStream_t>, LatestReg> latest;
  std::unordered_map<int64_t, std::vector<Registration>> tokens;
  uint64_t next_seq = 1;
  size_t inserts = 0;
};

inline RecordRegistry& registry() {
  static RecordRegistry r;
  return r;
}

inline bool same_owner(const std::weak_ptr<phi::Allocation>& a,
                       const std::weak_ptr<phi::Allocation>& b) {
  return !a.owner_before(b) && !b.owner_before(a);
}

// Register stream `s` on `t`'s allocation and add it to `token` (0: register
// only, never withdrawn).
inline void record(const paddle::Tensor& t, cudaStream_t s, int64_t token) {
  if (!t.initialized() || !t.is_gpu()) return;
  auto* d = dynamic_cast<phi::DenseTensor*>(t.impl().get());
  if (d == nullptr || d->Holder() == nullptr) return;
  const std::shared_ptr<phi::Allocation>& a = d->Holder();
  if (!paddle::memory::RecordStream(a, s)) return;  // not stream-safe owned
  if (token == 0) return;
  auto& r = registry();
  std::lock_guard<std::mutex> lk(r.mu);
  const uint64_t seq = r.next_seq++;
  r.latest[{a.get(), s}] = {a, seq};
  r.tokens[token].push_back({a, s, seq});
  if (++r.inserts % 4096 == 0) {
    // Drop entries of freed allocations, and tokens never released (a task
    // dropped without a wait) once all their allocations are freed.
    for (auto it = r.latest.begin(); it != r.latest.end();)
      it = it->second.alloc.expired() ? r.latest.erase(it) : std::next(it);
    for (auto it = r.tokens.begin(); it != r.tokens.end();) {
      bool live = false;
      for (auto& g : it->second) live = live || !g.alloc.expired();
      it = live ? std::next(it) : r.tokens.erase(it);
    }
  }
}

// Withdraw `token`'s registrations; `wait_stream` has just waited for the
// call that made them (see the file comment for when one is kept).
inline void release(int64_t token, cudaStream_t wait_stream) {
  auto& r = registry();
  std::vector<Registration> regs;
  {
    std::lock_guard<std::mutex> lk(r.mu);
    auto it = r.tokens.find(token);
    if (it == r.tokens.end()) return;
    regs.swap(it->second);
    r.tokens.erase(it);
  }
  for (auto& g : regs) {
    std::shared_ptr<phi::Allocation> a = g.alloc.lock();
    if (!a) continue;
    {
      std::lock_guard<std::mutex> lk(r.mu);
      auto it = r.latest.find({a.get(), g.stream});
      if (it == r.latest.end() || !same_owner(it->second.alloc, g.alloc) ||
          it->second.seq != g.seq)
        continue;  // a later call on that stream still uses it
      r.latest.erase(it);
    }
    if (paddle::memory::GetStream(a) == wait_stream)
      paddle::memory::EraseStream(a, g.stream);
  }
}

}  // namespace zcomm_ops
