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

// transport_nccl.cu -- NCCL symmetric windows (LSA) as the symmetric memory.
//
//   ncclMemAlloc + ncclCommWindowRegister(NCCL_WIN_COLL_SYMMETRIC)
//   peer pointers:      ncclGetLsaDevicePointer
//   multicast pointers: ncclGetLsaMultimemDevicePointer (needs the comm to
//                       have nvlsSupport=1, i.e. NCCL_NVLS_ENABLE=1)

// Only the pointers come from NCCL; the data path stays zcomm's own.

#include <nccl.h>
#include <nccl_device.h>
#include "zcomm/common.cuh"
#include "zcomm/zcomm.h"

#define ZC_NCCL_CHECK(x)                  \
  do {                                    \
    ncclResult_t r_ = (x);                \
    if (r_ != ncclSuccess) {              \
      fprintf(stderr,                     \
              "[zcomm] %s:%d %s -> %s\n", \
              __FILE__,                   \
              __LINE__,                   \
              #x,                         \
              ncclGetErrorString(r_));    \
      abort();                            \
    }                                     \
  } while (0)

namespace zcomm {
namespace {

class NcclTransport final : public Transport {
 public:
  explicit NcclTransport(ncclComm_t comm) : comm_(comm) {
    ZC_NCCL_CHECK(ncclCommUserRank(comm_, &rank_));
    ZC_NCCL_CHECK(ncclCommCount(comm_, &nranks_));
    ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
    ZC_NCCL_CHECK(ncclCommQueryProperties(comm_, &props));
    if (props.nLsaTeams != 1) {
      fprintf(stderr,
              "[zcomm] the nccl transport needs a single-node comm "
              "(nLsaTeams=%d)\n",
              props.nLsaTeams);
      abort();
    }
    mc_ = props.multimemSupport;
    ZC_CUDA_CHECK(cudaStreamCreateWithFlags(&s_, cudaStreamNonBlocking));
    ZC_CUDA_CHECK(cudaMalloc(&scratch_, sizeof(float)));
  }
  ~NcclTransport() override {
    cudaFree(scratch_);
    cudaStreamDestroy(s_);
  }
  const char* name() const override { return "nccl"; }
  int rank() const override { return rank_; }
  int nranks() const override { return nranks_; }
  bool supports_multicast() const override { return mc_; }

  void barrier() override {
    ZC_NCCL_CHECK(
        ncclAllReduce(scratch_, scratch_, 1, ncclFloat, ncclSum, comm_, s_));
    ZC_CUDA_CHECK(cudaStreamSynchronize(s_));
  }

  SymmBuffer alloc(size_t bytes, bool multicast) override {
    if (multicast && !mc_) {
      fprintf(stderr,
              "[zcomm] the comm has no multimem (create it with "
              "NCCL_NVLS_ENABLE=1)\n");
      abort();
    }
    SymmBuffer b;
    b.bytes = bytes;
    const size_t sz = (bytes + NCCL_WIN_REQUIRED_ALIGNMENT - 1) /
                      NCCL_WIN_REQUIRED_ALIGNMENT * NCCL_WIN_REQUIRED_ALIGNMENT;
    ZC_NCCL_CHECK(ncclMemAlloc(&b.local, sz));
    ZC_CUDA_CHECK(cudaMemset(b.local, 0, sz));
    ZC_CUDA_CHECK(cudaDeviceSynchronize());
    auto* win = new ncclWindow_t;
    ZC_NCCL_CHECK(ncclCommWindowRegister(
        comm_, b.local, sz, win, NCCL_WIN_COLL_SYMMETRIC));
    for (int p = 0; p < nranks_; ++p)
      ZC_NCCL_CHECK(ncclGetLsaDevicePointer(*win, 0, p, &b.peers.p[p]));
    if (b.peers.p[rank_] != b.local) {
      // LSA's own mapping need not equal the VA ncclMemAlloc returned; local
      // access uses `local`.
      b.peers.p[rank_] = b.local;
    }
    if (multicast)
      ZC_NCCL_CHECK(ncclGetLsaMultimemDevicePointer(*win, 0, &b.mc));
    b.impl = win;
    barrier();
    return b;
  }

  void free(SymmBuffer& b) override {
    if (!b.local) return;
    ZC_CUDA_CHECK(cudaDeviceSynchronize());
    barrier();
    auto* win = static_cast<ncclWindow_t*>(b.impl);
    ZC_NCCL_CHECK(ncclCommWindowDeregister(comm_, *win));
    delete win;
    ZC_NCCL_CHECK(ncclMemFree(b.local));
    b = SymmBuffer{};
  }

 private:
  ncclComm_t comm_;
  int rank_ = 0, nranks_ = 0;
  bool mc_ = false;
  cudaStream_t s_ = nullptr;
  float* scratch_ = nullptr;
};

}  // namespace

std::unique_ptr<Transport> make_nccl_transport(ncclComm_t comm) {
  return std::make_unique<NcclTransport>(comm);
}

}  // namespace zcomm
