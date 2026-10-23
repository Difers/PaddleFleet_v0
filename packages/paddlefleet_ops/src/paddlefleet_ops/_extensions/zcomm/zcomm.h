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

// zcomm.h -- intra-node (NVLink/NVSwitch) AllGather / ReduceScatter.
//
// Two layers:
//   * Transport owns "symmetric memory" only: the peer pointer of every rank's
//     buffer (NVLink ld/st reaches it directly) and, optionally, a multicast
//     (multimem) pointer. Implemented with NCCL symmetric windows
//     (ncclMemAlloc + ncclCommWindowRegister; peer pointers from
//     ncclGetLsaPointer, multicast pointers need NCCL_NVLS_ENABLE=1).
//   * IntraComm is the data path (copy / reduce kernels, CE scheduling) and the
//     synchronization. All of it is device-side: block b of a rank handshakes
//     with block b of every peer through a flag holding a monotonic epoch that
//     is never reset.
//
// Calling convention matches NCCL: every rank of the group calls with the same
// order and the same arguments (byte counts, block counts).
#pragma once
#include <cuda_runtime.h>
#include <nccl.h>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace zcomm {

constexpr int kMaxRanks = 8;
constexpr int kMaxBlocks =
    256;  // grid cap of the SM path (flags reserve slots for it)

struct PeerPtrs {
  void* p[kMaxRanks];
};

// A symmetric buffer: the local pointer, the peer pointer of every rank
// (peers.p[rank] == local) and the multicast pointer (nullptr when the
// transport has none or it was not requested).
struct SymmBuffer {
  void* local = nullptr;
  PeerPtrs peers{};
  void* mc = nullptr;
  size_t bytes = 0;
  void* impl = nullptr;  // transport-private (e.g. ncclWindow_t)
};

class Transport {
 public:
  virtual ~Transport() = default;
  virtual const char* name() const = 0;
  virtual int rank() const = 0;
  virtual int nranks() const = 0;
  virtual bool supports_multicast() const = 0;
  // Collective: every rank calls it with the same bytes. The buffer comes back
  // zeroed and visible to the whole group.
  virtual SymmBuffer alloc(size_t bytes, bool multicast) = 0;
  virtual void free(SymmBuffer& b) = 0;
  virtual void barrier() = 0;  // host-side group barrier
};

// comm must be single-node (one LSA team for the whole group). Its lifetime
// stays with the caller.
std::unique_ptr<Transport> make_nccl_transport(ncclComm_t comm);

enum class DType { kBF16, kFP32 };

// Reduction accumulation order.
//   kFP32     : accumulate in fp32, round once at the end (more accurate than
//               NCCL, deterministic)
//   kRingBF16 : accumulate in NCCL's ring order (r+1, r+2, ..., r) and round
//               back to bf16 after every add, which reproduces NCCL's ring
//               ReduceScatter bit for bit (bf16 only)
enum class Acc { kFP32, kRingBF16 };

class IntraComm {
 public:
  explicit IntraComm(std::unique_ptr<Transport> t);
  ~IntraComm();
  IntraComm(const IntraComm&) = delete;
  IntraComm& operator=(const IntraComm&) = delete;

  Transport& transport() { return *t_; }
  int rank() const { return rank_; }
  int nranks() const { return nranks_; }

  SymmBuffer alloc(size_t bytes, bool multicast = false) {
    return t_->alloc(bytes, multicast);
  }
  void free(SymmBuffer& b) { t_->free(b); }

  // ---------------- AllGather：out[r*shard .. (r+1)*shard) = in_r
  // ---------------- push: an SM kernel reads the local `in` and stores it into
  // every peer's `out` (NVLink store). `in` may be any local pointer; when it
  // is already out.local + rank*shard the self copy is skipped.
  void all_gather_push(const void* in,
                       const SymmBuffer& out,
                       size_t shard_bytes,
                       cudaStream_t s,
                       int blocks);
  // pull: an SM kernel loads every peer's `in` into the local `out` (NVLink
  // load). `in` must be a symmetric buffer.
  void all_gather_pull(const SymmBuffer& in,
                       void* out,
                       size_t shard_bytes,
                       cudaStream_t s,
                       int blocks);
  // mc: multimem.st, one store that NVSwitch replicates to the group.
  // `out` must carry a multicast pointer.
  void all_gather_mc(const void* in,
                     const SymmBuffer& out,
                     size_t shard_bytes,
                     cudaStream_t s,
                     int blocks);
  // ce: pushed by copy engines, 0 SMs. Synchronized with stream memops
  // (cuStreamWrite/WaitValue64).
  void all_gather_ce(const void* in,
                     const SymmBuffer& out,
                     size_t shard_bytes,
                     cudaStream_t s);
  // ce_mc: a single cudaMemcpyAsync into the multicast address of `out`, which
  // NVSwitch replicates to the group (`out` needs a multicast pointer). One
  // copy API on the host instead of nranks (or one batch of ~23 us).
  // ce_pull: copy x into this rank's symmetric stage, handshake, then pull the
  //          peers' stages into `out` (any device memory) on the copy engine.
  //          The stage is held only for the duration of one call, so `out`'s
  //          lifetime is independent of the symmetric-buffer pool. One
  //          handshake per call: consecutive calls must alternate between two
  //          stages (a stage is reused only two calls later on this comm).
  void all_gather_ce_pull(const void* in,
                          const SymmBuffer& stage,
                          void* out,
                          size_t shard_bytes,
                          cudaStream_t s);
  void all_gather_ce_mc(const void* in,
                        const SymmBuffer& out,
                        size_t shard_bytes,
                        cudaStream_t s);

  // ---------------- ReduceScatter(SUM)：out = Σ_p in_p[rank*shard ..)
  // ---------------- pull (one-shot): read shard `rank` of every peer's `in`,
  // accumulate in registers and write the local `out`.
  void reduce_scatter_pull(const SymmBuffer& in,
                           void* out,
                           size_t shard_bytes,
                           DType dt,
                           Acc acc,
                           cudaStream_t s,
                           int blocks);
  // push (the split form: transposed copy + local sum in one kernel): block p
  // of `in` goes into peer p's stage[rank], then a handshake and the local sum.
  // `stage` is a symmetric buffer of nranks*shard bytes.
  void reduce_scatter_push(const void* in,
                           const SymmBuffer& stage,
                           void* out,
                           size_t shard_bytes,
                           DType dt,
                           Acc acc,
                           cudaStream_t s,
                           int blocks);
  // mc: multimem.ld_reduce, reduced inside NVSwitch (fp32 accumulation) so the
  // SMs only issue loads. `in` must carry a multicast pointer.
  void reduce_scatter_mc(const SymmBuffer& in,
                         void* out,
                         size_t shard_bytes,
                         DType dt,
                         cudaStream_t s,
                         int blocks);
  // ce (split form, 0 SMs for the transfer): copy engines transpose, then a
  // local-sum kernel. One handshake per call; consecutive calls must
  // alternate between two stages, as for all_gather_ce_pull.
  void reduce_scatter_ce(const void* in,
                         const SymmBuffer& stage,
                         void* out,
                         size_t shard_bytes,
                         DType dt,
                         Acc acc,
                         cudaStream_t s,
                         int sum_blocks);

 private:
  void ce_sync(cudaStream_t s, uint64_t val);
  void ce_fork(cudaStream_t s);
  void ce_join(cudaStream_t s);

  std::unique_ptr<Transport> t_;
  int rank_, nranks_;
  SymmBuffer flags_;  // [kMaxBlocks][kMaxRanks] SM + [kMaxRanks] CE handshakes
  uint64_t epoch_sm_ = 0;  // +1 per SM call; flag value = 2*epoch+{1,2}
  uint64_t ce_seq_ = 0;    // value of the last CE handshake
};

}  // namespace zcomm
