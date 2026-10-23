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

// common.cuh -- internal helpers of zcomm: error checks, PeerPtrs and the
// cross-GPU visibility primitives.
//
// Only for .cu files (it contains device code); the public API is in zcomm.h.
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define ZC_CUDA_CHECK(x)                  \
  do {                                    \
    cudaError_t e_ = (x);                 \
    if (e_ != cudaSuccess) {              \
      fprintf(stderr,                     \
              "[zcomm] %s:%d %s -> %s\n", \
              __FILE__,                   \
              __LINE__,                   \
              #x,                         \
              cudaGetErrorString(e_));    \
      abort();                            \
    }                                     \
  } while (0)

#define ZC_CU_CHECK(x)                                                         \
  do {                                                                         \
    CUresult r_ = (x);                                                         \
    if (r_ != CUDA_SUCCESS) {                                                  \
      const char* s_ = nullptr;                                                \
      cuGetErrorString(r_, &s_);                                               \
      fprintf(stderr, "[zcomm] %s:%d %s -> %s\n", __FILE__, __LINE__, #x, s_); \
      abort();                                                                 \
    }                                                                          \
  } while (0)

namespace zcomm {

// Timeout of the device-side spin (ns). On expiry the kernel traps instead of
// occupying the GPU forever.
constexpr uint64_t kSpinTimeoutNs = 20ull * 1000 * 1000 * 1000;

__device__ __forceinline__ uint64_t globaltimer() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// sys scope: the flag is written and read from different GPUs, so gpu scope is
// not enough.
__device__ __forceinline__ uint64_t ld_acquire_sys(const uint64_t* p) {
  uint64_t v;
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];"
               : "=l"(v)
               : "l"(p)
               : "memory");
  return v;
}
__device__ __forceinline__ void st_release_sys(uint64_t* p, uint64_t v) {
  asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(p), "l"(v)
               : "memory");
}

// Peer data goes through L2 only (bypassing L1): L1 does not stay coherent with
// a writer on another GPU.
__device__ __forceinline__ int4 ld_cg(const int4* p) { return __ldcg(p); }
// Local read-only input: use the read-only path.
__device__ __forceinline__ int4 ld_nc(const int4* p) { return __ldg(p); }

__device__ __forceinline__ void spin_until_geq(const uint64_t* p,
                                               uint64_t target) {
  if (ld_acquire_sys(p) >= target) return;
  const uint64_t t0 = globaltimer();
  while (ld_acquire_sys(p) < target) {
    if (globaltimer() - t0 > kSpinTimeoutNs) {
      printf("[zcomm] spin timeout: block %d thread %d flag=%" PRIu64
             " target=%" PRIu64 "\n",
             blockIdx.x,
             threadIdx.x,
             (uint64_t)ld_acquire_sys(p),
             (uint64_t)target);
      __trap();
    }
  }
}

}  // namespace zcomm
