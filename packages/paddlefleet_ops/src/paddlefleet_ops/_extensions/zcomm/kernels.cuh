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

// kernels.cuh -- device side of the intra-node AG / RS.
//
// Data partition: a shard is cut into gridDim.x contiguous ranges of 16B
// vectors (int4) and block b owns range b. Every rank uses the same grid, so
// what block b of rank q writes is exactly what block b of rank p reads -- the
// synchronization only needs pairwise handshakes between same-index blocks, no
// group-wide or grid-wide barrier.
//
// Handshake = a monotonic epoch in flags[b][src] that is never reset:
//   arrive(base+1): "I entered this call", so my input is complete and my
//                   output / staging may be written
//   done  (base+2): "my reads / writes of your range are finished"
#pragma once
#include <cuda_bf16.h>
#include "zcomm/common.cuh"
#include "zcomm/zcomm.h"

namespace zcomm {
namespace kern {

constexpr int kThreads = 512;
constexpr int kUnroll = 2;
constexpr int kCeFlagBase =
    kMaxBlocks * kMaxRanks;  // CE handshake slots follow the SM ones
constexpr int kCeSeqSlot =
    kCeFlagBase + kMaxRanks;  // local seq source (2 slots, odd/even)
constexpr size_t kFlagSlots = (size_t)kCeSeqSlot + 2;

// Handshake between same-index blocks: tell every peer "I reached val", then
// wait for every peer to reach it.
__device__ __forceinline__ void block_sync(const PeerPtrs& flags,
                                           int rank,
                                           int n,
                                           uint64_t val) {
  __syncthreads();  // every read/write of this block has been issued
  const int t = threadIdx.x;
  if (t < n && t != rank) {
    __threadfence_system();
    st_release_sys((uint64_t*)flags.p[t] + blockIdx.x * kMaxRanks + rank, val);
    spin_until_geq((const uint64_t*)flags.p[rank] + blockIdx.x * kMaxRanks + t,
                   val);
  }
  __syncthreads();
}

__device__ __forceinline__ void block_range(size_t nvec,
                                            size_t& lo,
                                            size_t& hi) {
  const size_t per = (nvec + gridDim.x - 1) / gridDim.x;
  lo = min(nvec, per * blockIdx.x);
  hi = min(nvec, lo + per);
}

// ---------------------------------------------------------------- reduction
// v[k] is the k-th addend, ordered by the ring accumulation order:
// (r+1), (r+2), ..., (r+n-1), r.
// kRound=true rounds back to T after every add, which is what NCCL's ring
// recvReduceSend does at each hop.
template <typename T, bool kRound>
struct Reducer;

template <bool kRound>
struct Reducer<__nv_bfloat16, kRound> {
  __device__ static int4 run(const int4* v, int n) {
    float acc[8];
    const __nv_bfloat162* x0 = reinterpret_cast<const __nv_bfloat162*>(&v[0]);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float2 f = __bfloat1622float2(x0[j]);
      acc[2 * j] = f.x;
      acc[2 * j + 1] = f.y;
    }
#pragma unroll
    for (int k = 1; k < kMaxRanks; ++k) {
      if (k < n) {
        const __nv_bfloat162* x =
            reinterpret_cast<const __nv_bfloat162*>(&v[k]);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          float2 f = __bfloat1622float2(x[j]);
          acc[2 * j] += f.x;
          acc[2 * j + 1] += f.y;
          if (kRound) {
            acc[2 * j] = __bfloat162float(__float2bfloat16_rn(acc[2 * j]));
            acc[2 * j + 1] =
                __bfloat162float(__float2bfloat16_rn(acc[2 * j + 1]));
          }
        }
      }
    }
    int4 o;
    __nv_bfloat162* y = reinterpret_cast<__nv_bfloat162*>(&o);
#pragma unroll
    for (int j = 0; j < 4; ++j)
      y[j] = __floats2bfloat162_rn(acc[2 * j], acc[2 * j + 1]);
    return o;
  }
};

template <bool kRound>
struct Reducer<float, kRound> {
  __device__ static int4 run(const int4* v, int n) {
    float4 a = *reinterpret_cast<const float4*>(&v[0]);
#pragma unroll
    for (int k = 1; k < kMaxRanks; ++k) {
      if (k < n) {
        float4 b = *reinterpret_cast<const float4*>(&v[k]);
        a.x += b.x;
        a.y += b.y;
        a.z += b.z;
        a.w += b.w;
      }
    }
    return *reinterpret_cast<int4*>(&a);
  }
};

// ---------------------------------------------------------------- AllGather
// push: read the local `in` once and store it into the `out` of n-1 peers
// (NVLink store, fire and forget).
__global__ void __launch_bounds__(kThreads) ag_push(const int4* __restrict__ in,
                                                    PeerPtrs out,
                                                    size_t nvec,
                                                    int copy_self,
                                                    PeerPtrs flags,
                                                    int rank,
                                                    int n,
                                                    uint64_t base) {
  block_sync(flags, rank, n, base + 1);
  size_t lo, hi;
  block_range(nvec, lo, hi);
  const size_t off = (size_t)rank * nvec;
  for (size_t i = lo + threadIdx.x; i < hi; i += kUnroll * kThreads) {
    int4 v[kUnroll];
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      const size_t idx = i + (size_t)u * kThreads;
      if (idx < hi) v[u] = ld_nc(in + idx);
    }
#pragma unroll
    for (int k = 0; k < kMaxRanks; ++k) {
      if (k < n && (k > 0 || copy_self)) {
        int4* dst = (int4*)out.p[(rank + k) % n] + off;
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
          const size_t idx = i + (size_t)u * kThreads;
          if (idx < hi) dst[idx] = v[u];
        }
      }
    }
  }
  block_sync(flags, rank, n, base + 2);
}

// pull: load from the `in` of n ranks (NVLink load) and write the local out.
__global__ void __launch_bounds__(kThreads) ag_pull(PeerPtrs in,
                                                    int4* __restrict__ out,
                                                    size_t nvec,
                                                    int copy_self,
                                                    PeerPtrs flags,
                                                    int rank,
                                                    int n,
                                                    uint64_t base) {
  block_sync(flags, rank, n, base + 1);
  size_t lo, hi;
  block_range(nvec, lo, hi);
  for (size_t i = lo + threadIdx.x; i < hi; i += kUnroll * kThreads) {
    int4 v[kMaxRanks][kUnroll];
#pragma unroll
    for (int k = 0; k < kMaxRanks; ++k) {
      if (k < n && (k > 0 || copy_self)) {
        const int4* src = (const int4*)in.p[(rank + k) % n];
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
          const size_t idx = i + (size_t)u * kThreads;
          if (idx < hi) v[k][u] = ld_cg(src + idx);
        }
      }
    }
#pragma unroll
    for (int k = 0; k < kMaxRanks; ++k) {
      if (k < n && (k > 0 || copy_self)) {
        int4* dst = out + (size_t)((rank + k) % n) * nvec;
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
          const size_t idx = i + (size_t)u * kThreads;
          if (idx < hi) dst[idx] = v[k][u];
        }
      }
    }
  }
  block_sync(
      flags,
      rank,
      n,
      base + 2);  // cannot exit before the peers read my in (it may be reused)
}

// ----------------------------------------------------------------
// ReduceScatter accumulates in the order chunk r takes in NCCL's ring
// ReduceScatter: (r+1), (r+2), ..., (r+n-1), r.
__device__ __forceinline__ int ring_src(int rank, int n, int k) {
  return (rank + 1 + k) % n;
}

// pull (one-shot): read shard `rank` of the n ranks' `in`, accumulate in
// registers and write the local out. Every byte crosses NVLink once and the
// local shard is written once -- no staging.
template <typename T, bool kRound>
__global__ void __launch_bounds__(kThreads) rs_pull(PeerPtrs in,
                                                    int4* __restrict__ out,
                                                    size_t nvec,
                                                    PeerPtrs flags,
                                                    int rank,
                                                    int n,
                                                    uint64_t base) {
  block_sync(flags, rank, n, base + 1);
  size_t lo, hi;
  block_range(nvec, lo, hi);
  const size_t off = (size_t)rank * nvec;
  for (size_t i = lo + threadIdx.x; i < hi; i += kThreads) {
    int4 v[kMaxRanks];
#pragma unroll
    for (int k = 0; k < kMaxRanks; ++k)
      if (k < n)
        v[k] = ld_cg((const int4*)in.p[ring_src(rank, n, k)] + off + i);
    out[i] = Reducer<T, kRound>::run(v, n);
  }
  block_sync(flags, rank, n, base + 2);
}

// Local sum: stage[q] is the shard rank q sent (q != rank); the own shard is
// read from `self`.
template <typename T, bool kRound>
__device__ __forceinline__ void local_sum_range(const int4* stage,
                                                const int4* self,
                                                int4* out,
                                                size_t nvec,
                                                int rank,
                                                int n,
                                                size_t lo,
                                                size_t hi,
                                                size_t step) {
  for (size_t i = lo; i < hi; i += step) {
    int4 v[kMaxRanks];
#pragma unroll
    for (int k = 0; k < kMaxRanks; ++k) {
      if (k < n) {
        const int q = ring_src(rank, n, k);
        v[k] =
            (q == rank) ? ld_nc(self + i) : ld_cg(stage + (size_t)q * nvec + i);
      }
    }
    out[i] = Reducer<T, kRound>::run(v, n);
  }
}

// push (split form): transposed copy into the peers' stage, a handshake and the
// local sum, all in one kernel.
template <typename T, bool kRound>
__global__ void __launch_bounds__(kThreads) rs_push(const int4* __restrict__ in,
                                                    PeerPtrs stage,
                                                    int4* __restrict__ out,
                                                    size_t nvec,
                                                    PeerPtrs flags,
                                                    int rank,
                                                    int n,
                                                    uint64_t base) {
  block_sync(flags, rank, n, base + 1);  // the peers' stage is free again
  size_t lo, hi;
  block_range(nvec, lo, hi);
  const size_t off = (size_t)rank * nvec;
  for (size_t i = lo + threadIdx.x; i < hi; i += kThreads) {
#pragma unroll
    for (int k = 1; k < kMaxRanks; ++k) {
      if (k < n) {
        const int p = (rank + k) % n;
        ((int4*)stage.p[p])[off + i] = ld_nc(in + (size_t)p * nvec + i);
      }
    }
  }
  block_sync(flags, rank, n, base + 2);  // the peers finished writing my stage
  local_sum_range<T, kRound>((const int4*)stage.p[rank],
                             in + off,
                             out,
                             nvec,
                             rank,
                             n,
                             lo + threadIdx.x,
                             hi,
                             kThreads);
}

// Local sum of the CE variant (the transfer was done by copy engines).
template <typename T, bool kRound>
__global__ void __launch_bounds__(kThreads)
    rs_local_sum(const int4* __restrict__ stage,
                 const int4* __restrict__ self,
                 int4* __restrict__ out,
                 size_t nvec,
                 int rank,
                 int n) {
  local_sum_range<T, kRound>(stage,
                             self,
                             out,
                             nvec,
                             rank,
                             n,
                             (size_t)blockIdx.x * kThreads + threadIdx.x,
                             nvec,
                             (size_t)gridDim.x * kThreads);
}

// ----------------------------------------------------------------
// multimem (NVLS): the mc pointer addresses a multicast object; a store is
// replicated to the group by NVSwitch and ld_reduce reads the whole group and
// reduces inside the switch.
__device__ __forceinline__ void mc_st(void* addr, int4 v) {
  asm volatile(
      "multimem.st.relaxed.sys.global.v4.f32 [%0], {%1,%2,%3,%4};" ::"l"(addr),
      "r"(v.x),
      "r"(v.y),
      "r"(v.z),
      "r"(v.w)
      : "memory");
}

template <typename T>
struct McReduce;
template <>
struct McReduce<__nv_bfloat16> {  // bf16 in, fp32 accumulate in the switch,
                                  // result rounded back to bf16
  __device__ static int4 run(const void* addr) {
    int4 v;
    asm volatile(
        "multimem.ld_reduce.relaxed.sys.global.add.acc::f32.v4.bf16x2 "
        "{%0,%1,%2,%3}, [%4];"
        : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
        : "l"(addr)
        : "memory");
    return v;
  }
};
template <>
struct McReduce<float> {
  __device__ static int4 run(const void* addr) {
    float4 v;
    asm volatile(
        "multimem.ld_reduce.relaxed.sys.global.add.v4.f32 {%0,%1,%2,%3}, [%4];"
        : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
        : "l"(addr)
        : "memory");
    return *reinterpret_cast<int4*>(&v);
  }
};

// AG: read the local in once and multimem.st it into out[rank] of the whole
// group (own copy included).
__global__ void __launch_bounds__(kThreads) ag_mc(const int4* __restrict__ in,
                                                  int4* out_mc,
                                                  size_t nvec,
                                                  PeerPtrs flags,
                                                  int rank,
                                                  int n,
                                                  uint64_t base) {
  block_sync(flags, rank, n, base + 1);
  size_t lo, hi;
  block_range(nvec, lo, hi);
  int4* dst = out_mc + (size_t)rank * nvec;
  for (size_t i = lo + threadIdx.x; i < hi; i += kUnroll * kThreads) {
    int4 v[kUnroll];
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      const size_t idx = i + (size_t)u * kThreads;
      if (idx < hi) v[u] = ld_nc(in + idx);
    }
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      const size_t idx = i + (size_t)u * kThreads;
      if (idx < hi) mc_st(dst + idx, v[u]);
    }
  }
  asm volatile("fence.proxy.alias;" ::
                   : "memory");  // multimem (alias proxy) writes become visible
                                 // before the flag
  block_sync(flags, rank, n, base + 2);
}

// RS: multimem.ld_reduce reads shard `rank` (reduced in the switch) and writes
// the local out.
template <typename T>
__global__ void __launch_bounds__(kThreads) rs_mc(const int4* in_mc,
                                                  int4* __restrict__ out,
                                                  size_t nvec,
                                                  PeerPtrs flags,
                                                  int rank,
                                                  int n,
                                                  uint64_t base) {
  block_sync(flags, rank, n, base + 1);
  asm volatile("fence.proxy.alias;" ::: "memory");
  size_t lo, hi;
  block_range(nvec, lo, hi);
  const int4* src = in_mc + (size_t)rank * nvec;
  for (size_t i = lo + threadIdx.x; i < hi; i += kUnroll * kThreads) {
    int4 v[kUnroll];
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      const size_t idx = i + (size_t)u * kThreads;
      if (idx < hi) v[u] = McReduce<T>::run(src + idx);
    }
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      const size_t idx = i + (size_t)u * kThreads;
      if (idx < hi) out[idx] = v[u];
    }
  }
  block_sync(
      flags, rank, n, base + 2);  // cannot exit before the group read my in
}

}  // namespace kern
}  // namespace zcomm
