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

// intra_comm.cu -- host side of IntraComm: symmetric memory, epochs and the
// kernel / CE scheduling.
#include <algorithm>
#include <cstring>
#include "zcomm/kernels.cuh"

namespace zcomm {

namespace {

template <bool kRound>
void launch_rs_pull(DType dt,
                    int blocks,
                    cudaStream_t s,
                    const PeerPtrs& in,
                    int4* out,
                    size_t nvec,
                    const PeerPtrs& fl,
                    int rank,
                    int n,
                    uint64_t base) {
  if (dt == DType::kBF16)
    kern::rs_pull<__nv_bfloat16, kRound>
        <<<blocks, kern::kThreads, 0, s>>>(in, out, nvec, fl, rank, n, base);
  else
    kern::rs_pull<float, false>
        <<<blocks, kern::kThreads, 0, s>>>(in, out, nvec, fl, rank, n, base);
}

template <bool kRound>
void launch_rs_push(DType dt,
                    int blocks,
                    cudaStream_t s,
                    const int4* in,
                    const PeerPtrs& st,
                    int4* out,
                    size_t nvec,
                    const PeerPtrs& fl,
                    int rank,
                    int n,
                    uint64_t base) {
  if (dt == DType::kBF16)
    kern::rs_push<__nv_bfloat16, kRound><<<blocks, kern::kThreads, 0, s>>>(
        in, st, out, nvec, fl, rank, n, base);
  else
    kern::rs_push<float, false><<<blocks, kern::kThreads, 0, s>>>(
        in, st, out, nvec, fl, rank, n, base);
}

template <bool kRound>
void launch_rs_sum(DType dt,
                   int blocks,
                   cudaStream_t s,
                   const int4* stage,
                   const int4* self,
                   int4* out,
                   size_t nvec,
                   int rank,
                   int n) {
  if (dt == DType::kBF16)
    kern::rs_local_sum<__nv_bfloat16, kRound>
        <<<blocks, kern::kThreads, 0, s>>>(stage, self, out, nvec, rank, n);
  else
    kern::rs_local_sum<float, false>
        <<<blocks, kern::kThreads, 0, s>>>(stage, self, out, nvec, rank, n);
}

size_t to_vec(size_t bytes) {
  if (bytes % 16) {
    fprintf(
        stderr, "[zcomm] shard of %zu bytes is not a multiple of 16\n", bytes);
    abort();
  }
  return bytes / 16;
}

int clamp_blocks(int b) { return std::max(1, std::min(b, kMaxBlocks)); }

}  // namespace

IntraComm::IntraComm(std::unique_ptr<Transport> t)
    : t_(std::move(t)), rank_(t_->rank()), nranks_(t_->nranks()) {
  if (nranks_ < 1 || nranks_ > kMaxRanks) {
    fprintf(
        stderr, "[zcomm] nranks=%d is outside [1, %d]\n", nranks_, kMaxRanks);
    abort();
  }
  flags_ =
      t_->alloc(kern::kFlagSlots * sizeof(uint64_t), t_->supports_multicast());
}

IntraComm::~IntraComm() {
  cudaDeviceSynchronize();
  t_->free(flags_);
}

// ---------------------------------------------------------------- SM path
void IntraComm::all_gather_push(const void* in,
                                const SymmBuffer& out,
                                size_t shard_bytes,
                                cudaStream_t s,
                                int blocks) {
  const size_t nvec = to_vec(shard_bytes);
  const uint64_t base = 2 * epoch_sm_++;
  const int self =
      (const char*)in != (const char*)out.local + rank_ * shard_bytes;
  kern::ag_push<<<clamp_blocks(blocks), kern::kThreads, 0, s>>>((const int4*)in,
                                                                out.peers,
                                                                nvec,
                                                                self,
                                                                flags_.peers,
                                                                rank_,
                                                                nranks_,
                                                                base);
  ZC_CUDA_CHECK(cudaGetLastError());
}

void IntraComm::all_gather_pull(const SymmBuffer& in,
                                void* out,
                                size_t shard_bytes,
                                cudaStream_t s,
                                int blocks) {
  const size_t nvec = to_vec(shard_bytes);
  const uint64_t base = 2 * epoch_sm_++;
  const int self =
      (const char*)in.local != (const char*)out + rank_ * shard_bytes;
  kern::ag_pull<<<clamp_blocks(blocks), kern::kThreads, 0, s>>>(
      in.peers, (int4*)out, nvec, self, flags_.peers, rank_, nranks_, base);
  ZC_CUDA_CHECK(cudaGetLastError());
}

void IntraComm::reduce_scatter_pull(const SymmBuffer& in,
                                    void* out,
                                    size_t shard_bytes,
                                    DType dt,
                                    Acc acc,
                                    cudaStream_t s,
                                    int blocks) {
  const size_t nvec = to_vec(shard_bytes);
  const uint64_t base = 2 * epoch_sm_++;
  if (acc == Acc::kRingBF16)
    launch_rs_pull<true>(dt,
                         clamp_blocks(blocks),
                         s,
                         in.peers,
                         (int4*)out,
                         nvec,
                         flags_.peers,
                         rank_,
                         nranks_,
                         base);
  else
    launch_rs_pull<false>(dt,
                          clamp_blocks(blocks),
                          s,
                          in.peers,
                          (int4*)out,
                          nvec,
                          flags_.peers,
                          rank_,
                          nranks_,
                          base);
  ZC_CUDA_CHECK(cudaGetLastError());
}

void IntraComm::reduce_scatter_push(const void* in,
                                    const SymmBuffer& stage,
                                    void* out,
                                    size_t shard_bytes,
                                    DType dt,
                                    Acc acc,
                                    cudaStream_t s,
                                    int blocks) {
  const size_t nvec = to_vec(shard_bytes);
  const uint64_t base = 2 * epoch_sm_++;
  if (acc == Acc::kRingBF16)
    launch_rs_push<true>(dt,
                         clamp_blocks(blocks),
                         s,
                         (const int4*)in,
                         stage.peers,
                         (int4*)out,
                         nvec,
                         flags_.peers,
                         rank_,
                         nranks_,
                         base);
  else
    launch_rs_push<false>(dt,
                          clamp_blocks(blocks),
                          s,
                          (const int4*)in,
                          stage.peers,
                          (int4*)out,
                          nvec,
                          flags_.peers,
                          rank_,
                          nranks_,
                          base);
  ZC_CUDA_CHECK(cudaGetLastError());
}

void IntraComm::all_gather_mc(const void* in,
                              const SymmBuffer& out,
                              size_t shard_bytes,
                              cudaStream_t s,
                              int blocks) {
  if (!out.mc) {
    fprintf(stderr, "[zcomm] all_gather_mc: out has no multicast pointer\n");
    abort();
  }
  const size_t nvec = to_vec(shard_bytes);
  const uint64_t base = 2 * epoch_sm_++;
  kern::ag_mc<<<clamp_blocks(blocks), kern::kThreads, 0, s>>>(
      (const int4*)in, (int4*)out.mc, nvec, flags_.peers, rank_, nranks_, base);
  ZC_CUDA_CHECK(cudaGetLastError());
}

void IntraComm::reduce_scatter_mc(const SymmBuffer& in,
                                  void* out,
                                  size_t shard_bytes,
                                  DType dt,
                                  cudaStream_t s,
                                  int blocks) {
  if (!in.mc) {
    fprintf(stderr, "[zcomm] reduce_scatter_mc: in has no multicast pointer\n");
    abort();
  }
  const size_t nvec = to_vec(shard_bytes);
  const uint64_t base = 2 * epoch_sm_++;
  if (dt == DType::kBF16)
    kern::rs_mc<__nv_bfloat16>
        <<<clamp_blocks(blocks), kern::kThreads, 0, s>>>((const int4*)in.mc,
                                                         (int4*)out,
                                                         nvec,
                                                         flags_.peers,
                                                         rank_,
                                                         nranks_,
                                                         base);
  else
    kern::rs_mc<float>
        <<<clamp_blocks(blocks), kern::kThreads, 0, s>>>((const int4*)in.mc,
                                                         (int4*)out,
                                                         nvec,
                                                         flags_.peers,
                                                         rank_,
                                                         nranks_,
                                                         base);
  ZC_CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------- CE path (0
// SMs). Synchronization uses stream memops: cuStreamWriteValue64 to write a
// peer flag and cuStreamWaitValue64 to wait on a local one, both executed by
// the GPU front end without SMs or the host.
//   1. arrive: tell the group "I entered this call" (so my out/stage may be
//      written) and wait until everyone did
//   2. one cudaMemcpyBatchAsync for all copies (the driver spreads them over
//      several copy engines; an earlier version used one stream per peer with
//      cudaMemcpyAsync and only reached ~370 GB/s, the batch matches NCCL's CE)
//   3. write the peer's done flag (the default flag carries a memory barrier,
//   so
//      the copies on the same stream land before it) and wait for everyone's
namespace {
CUdeviceptr ce_flag(const PeerPtrs& f, int owner, int src) {
  return (CUdeviceptr)((uint64_t*)f.p[owner] + kern::kCeFlagBase + src);
}

void memcpy_batch(
    void** dst, void** src, size_t* sz, size_t n, cudaStream_t s) {
  if (!n) return;
#if CUDART_VERSION >= 12080
  cudaMemcpyAttributes attr = {};
  attr.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
  attr.flags = cudaMemcpyFlagPreferOverlapWithCompute;
  size_t idx = 0;
#if CUDART_VERSION >= 13000
  ZC_CUDA_CHECK(cudaMemcpyBatchAsync(dst, src, sz, n, &attr, &idx, 1, s));
#else
  // CUDA 12.8/12.9 take an extra `failIdx` before the stream.
  size_t fail_idx = 0;
  ZC_CUDA_CHECK(
      cudaMemcpyBatchAsync(dst, src, sz, n, &attr, &idx, 1, &fail_idx, s));
#endif
#else
  // No batch API before CUDA 12.8: one copy per peer (slower, see above).
  for (size_t i = 0; i < n; ++i)
    ZC_CUDA_CHECK(
        cudaMemcpyAsync(dst[i], src[i], sz[i], cudaMemcpyDeviceToDevice, s));
#endif
}
}  // namespace

// One CE handshake: write val into the local seq slot, copy it with a CE into
// every peer's flag[rank] (the same trick as NCCL's UC sync in ce_coll.cc: a
// remote cuStreamWriteValue64 costs ~7us, an 8B CE copy much less) and wait for
// the whole group with one cuStreamBatchMemOp. The copies share the stream with
// the data copies, so they are ordered after them.
void IntraComm::ce_sync(cudaStream_t s, uint64_t val) {
  uint64_t* seq = (uint64_t*)flags_.local + kern::kCeSeqSlot + (val & 1);
  ZC_CU_CHECK(cuStreamWriteValue64((CUstream)s, (CUdeviceptr)seq, val, 0));
  if (flags_.mc) {  // multicast: one 8B copy replicated by NVSwitch (NCCL's MC
                    // sync)
    uint64_t* mc = (uint64_t*)flags_.mc + kern::kCeFlagBase + rank_;
    ZC_CUDA_CHECK(cudaMemcpyAsync(
        mc, seq, sizeof(uint64_t), cudaMemcpyDeviceToDevice, s));
  } else {  // no multicast: 7 unicast 8B copies in one batch (a per-copy
            // cudaMemcpyAsync pays a fixed cost each)
    void* dst[kMaxRanks];
    void* src[kMaxRanks];
    size_t sz[kMaxRanks];
    size_t m = 0;
    for (int k = 1; k < nranks_; ++k) {
      const int p = (rank_ + k) % nranks_;
      dst[m] = (void*)ce_flag(flags_.peers, p, rank_);
      src[m] = seq;
      sz[m++] = sizeof(uint64_t);
    }
    memcpy_batch(dst, src, sz, m, s);
  }
  CUstreamBatchMemOpParams ops[kMaxRanks] = {};
  unsigned n = 0;
  for (int p = 0; p < nranks_; ++p) {
    if (p == rank_) continue;
    ops[n].waitValue.operation = CU_STREAM_MEM_OP_WAIT_VALUE_64;
    ops[n].waitValue.address = ce_flag(flags_.peers, rank_, p);
    ops[n].waitValue.value64 = val;
    ops[n].waitValue.flags = CU_STREAM_WAIT_VALUE_GEQ;
    ++n;
  }
  if (n) ZC_CU_CHECK(cuStreamBatchMemOp((CUstream)s, n, ops, 0));
}

// Every CE handshake draws the next value of one monotonic counter, so a
// peer's flag only ever grows and a GEQ wait is never satisfied by a stale one.
void IntraComm::ce_fork(cudaStream_t s) { ce_sync(s, ++ce_seq_); }

void IntraComm::ce_join(cudaStream_t s) { ce_sync(s, ++ce_seq_); }

void IntraComm::all_gather_ce(const void* in,
                              const SymmBuffer& out,
                              size_t shard_bytes,
                              cudaStream_t s) {
  ce_fork(s);
  const size_t off = (size_t)rank_ * shard_bytes;
  void* dst[kMaxRanks];
  void* src[kMaxRanks];
  size_t sz[kMaxRanks];
  size_t n = 0;
  if ((const char*)in != (const char*)out.local + off) {
    dst[n] = (char*)out.local + off;
    src[n] = (void*)in;
    sz[n++] = shard_bytes;
  }
  for (int k = 1; k < nranks_; ++k) {
    const int p = (rank_ + k) %
                  nranks_;  // rotate the destination so no peer gets hit by all
    dst[n] = (char*)out.peers.p[p] + off;
    src[n] = (void*)in;
    sz[n++] = shard_bytes;
  }
  memcpy_batch(dst, src, sz, n, s);
  ce_join(s);
}

void IntraComm::all_gather_ce_pull(const void* in,
                                   const SymmBuffer& stage,
                                   void* out,
                                   size_t shard_bytes,
                                   cudaStream_t s) {
  // 1) own shard -> local stage; 2) arrive: every stage is filled; 3) pull 7
  // peer stages into out (own slot straight from in). There is no closing
  // "peers finished reading" handshake: the caller alternates two stages (see
  // zcomm.h), and every peer writes its arrive of the call in between only
  // after its pulls of this call are done (same stream), so step 1 of the
  // next call on this stage is ordered after them.
  ZC_CUDA_CHECK(cudaMemcpyAsync(
      stage.local, in, shard_bytes, cudaMemcpyDeviceToDevice, s));
  ce_fork(s);
  void* dst[kMaxRanks];
  void* src[kMaxRanks];
  size_t sz[kMaxRanks];
  size_t n = 0;
  dst[n] = static_cast<char*>(out) + (size_t)rank_ * shard_bytes;
  src[n] = const_cast<void*>(in);
  sz[n++] = shard_bytes;
  for (int k = 1; k < nranks_; ++k) {
    const int p = (rank_ + k) % nranks_;
    dst[n] = static_cast<char*>(out) + (size_t)p * shard_bytes;
    src[n] = stage.peers.p[p];
    sz[n++] = shard_bytes;
  }
  memcpy_batch(dst, src, sz, n, s);
}

void IntraComm::all_gather_ce_mc(const void* in,
                                 const SymmBuffer& out,
                                 size_t shard_bytes,
                                 cudaStream_t s) {
  if (!out.mc) {
    fprintf(stderr, "[zcomm] all_gather_ce_mc: out has no multicast pointer\n");
    abort();
  }
  ce_fork(s);
  ZC_CUDA_CHECK(cudaMemcpyAsync((char*)out.mc + (size_t)rank_ * shard_bytes,
                                in,
                                shard_bytes,
                                cudaMemcpyDeviceToDevice,
                                s));
  ce_join(s);
}

void IntraComm::reduce_scatter_ce(const void* in,
                                  const SymmBuffer& stage,
                                  void* out,
                                  size_t shard_bytes,
                                  DType dt,
                                  Acc acc,
                                  cudaStream_t s,
                                  int sum_blocks) {
  // Transposed copies into the peers' stages, then one handshake (every
  // peer's copies into this stage are done), then the local sum. No leading
  // "peers finished summing their stage" handshake: as for
  // all_gather_ce_pull the caller alternates two stages, and the handshake
  // of the call in between is written after that call's sum was enqueued.
  const size_t off = (size_t)rank_ * shard_bytes;
  void* dst[kMaxRanks];
  void* src[kMaxRanks];
  size_t sz[kMaxRanks];
  size_t n = 0;
  for (int k = 1; k < nranks_; ++k) {
    const int p = (rank_ + k) % nranks_;
    dst[n] = (char*)stage.peers.p[p] + off;
    src[n] = (char*)in + (size_t)p * shard_bytes;
    sz[n++] = shard_bytes;
  }
  memcpy_batch(dst, src, sz, n, s);
  ce_join(s);
  const size_t nvec = to_vec(shard_bytes);
  const int4* self = (const int4*)((const char*)in + off);
  if (acc == Acc::kRingBF16)
    launch_rs_sum<true>(dt,
                        std::max(1, sum_blocks),
                        s,
                        (const int4*)stage.local,
                        self,
                        (int4*)out,
                        nvec,
                        rank_,
                        nranks_);
  else
    launch_rs_sum<false>(dt,
                         std::max(1, sum_blocks),
                         s,
                         (const int4*)stage.local,
                         self,
                         (int4*)out,
                         nvec,
                         rank_,
                         nranks_);
  ZC_CUDA_CHECK(cudaGetLastError());
}

}  // namespace zcomm
