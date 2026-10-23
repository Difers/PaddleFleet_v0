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

// zcomm_intra_ops.cu -- intra-node (NVLink/NVSwitch) communication ops:
// copy-engine AllGather / ReduceScatter.
//
//   zcomm_unique_id(any)                  -> uint8[128], generated on rank 0
//   and
//                                            broadcast by the caller
//   zcomm_intra_init(uid, rank, nranks)   -> int64[1] handle: a CTA_POLICY_ZERO
//                                            NCCL comm + an IntraComm
//   zcomm_intra_alloc(h, nbytes)          -> uint8[nbytes] symmetric buffer
//                                            (ncclMemAlloc + window;
//                                            collective)
//   zcomm_intra_all_gather(h, x, out_buf)    in place: out_buf[r*shard..] =
//   x_r,
//                                            pushed by copy engines, 0 SMs
//   zcomm_intra_reduce_scatter(h, x, stage, out, ring_acc, sum_blocks)
//                                            in place: out = sum_r
//                                            x_r[rank*shard..] (copy engines
//                                            transpose into the peers' stage,
//                                            then a local-sum kernel)
//   zcomm_buffer_view(buf, offset, shape, dtype)
//                                            a byte range of a symmetric buffer
//                                            as a tensor of the given
//                                            dtype/shape
//
// Allocator-owned tensors a communication op reads or writes are registered
// on its stream under the call's `token`; zcomm_release_streams(token)
// withdraws that once the calc stream waited for the call
// (zcomm/record_stream.h).
//
// The communication ops take the stream as a `stream` attribute (the integer
// value of a cudaStream_t, usually the group's comm stream; 0 = current
// stream); every rank of a handle must call them in the same order with the
// same arguments. Symmetric buffers are resident: deregistering a window is a
// collective, so it must not happen in a tensor destructor (ranks would free at
// different times and hang), hence the deleter does nothing and the buffers are
// reclaimed when the process exits. Callers pool them by power-of-two bucket
// with a ring of slots.
#include <cuda_bf16.h>
#include <nccl.h>
#include <paddle/extension.h>
#include <algorithm>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>
#include "zcomm/record_stream.h"
#include "zcomm/zcomm.h"

namespace {

#define ZC_NCCL_ENFORCE(x)                                                \
  do {                                                                    \
    ncclResult_t r_ = (x);                                                \
    PD_CHECK(r_ == ncclSuccess, #x, " failed: ", ncclGetErrorString(r_)); \
  } while (0)

struct IntraHandle {
  ncclComm_t comm = nullptr;
  std::unique_ptr<zcomm::IntraComm> zc;
  std::mutex mu;
  // device pointer of a symmetric buffer -> its SymmBuffer (the collectives
  // look it up by data_ptr)
  std::unordered_map<const void*, zcomm::SymmBuffer> bufs;
};

std::mutex g_mu;
std::unordered_map<int64_t, std::shared_ptr<IntraHandle>> g_handles;
int64_t g_next = 1;

std::shared_ptr<IntraHandle> get_handle(const paddle::Tensor& h) {
  PD_CHECK(h.numel() == 1 && h.dtype() == paddle::DataType::INT64,
           "zcomm handle must be int64[1]");
  auto hc = h.is_gpu() ? h.copy_to(paddle::CPUPlace(), true) : h;
  const int64_t id = hc.data<int64_t>()[0];
  std::lock_guard<std::mutex> lk(g_mu);
  auto it = g_handles.find(id);
  PD_CHECK(it != g_handles.end(), "unknown zcomm intra handle ", id);
  return it->second;
}

const zcomm::SymmBuffer& find_buf(IntraHandle& h,
                                  const paddle::Tensor& t,
                                  size_t need) {
  std::lock_guard<std::mutex> lk(h.mu);
  auto it = h.bufs.find(t.data());
  PD_CHECK(it != h.bufs.end(),
           "tensor is not a zcomm symmetric buffer (allocate it with "
           "zcomm_intra_alloc and pass the "
           "whole buffer, not a slice)");
  PD_CHECK(it->second.bytes >= need,
           "zcomm buffer too small: ",
           it->second.bytes,
           " < ",
           need);
  return it->second;
}

size_t nbytes(const paddle::Tensor& t) {
  return t.numel() * phi::SizeOf(t.dtype());
}

}  // namespace

cudaStream_t pick_stream(int64_t s, const paddle::Tensor& x) {
  return s ? reinterpret_cast<cudaStream_t>(s) : x.stream();
}

std::vector<paddle::Tensor> zcomm_unique_id(const paddle::Tensor& place_hint) {
  (void)place_hint;
  ncclUniqueId id;
  ZC_NCCL_ENFORCE(ncclGetUniqueId(&id));
  auto out = paddle::empty({128}, paddle::DataType::UINT8, paddle::CPUPlace());
  std::memcpy(out.data<uint8_t>(), &id, 128);
  return {out};
}

std::vector<paddle::Tensor> zcomm_intra_init(const paddle::Tensor& uid,
                                             int64_t rank,
                                             int64_t nranks) {
  PD_CHECK(uid.numel() == 128 && uid.dtype() == paddle::DataType::UINT8,
           "uid must be uint8[128]");
  auto uc = uid.is_gpu() ? uid.copy_to(paddle::CPUPlace(), true) : uid;
  ncclUniqueId id;
  std::memcpy(&id, uc.data<uint8_t>(), 128);
  auto h = std::make_shared<IntraHandle>();
  ncclConfig_t cfg = NCCL_CONFIG_INITIALIZER;
  cfg.CTAPolicy = NCCL_CTA_POLICY_ZERO;
  ZC_NCCL_ENFORCE(
      ncclCommInitRankConfig(&h->comm, (int)nranks, id, (int)rank, &cfg));
  h->zc =
      std::make_unique<zcomm::IntraComm>(zcomm::make_nccl_transport(h->comm));
  int64_t hid;
  {
    std::lock_guard<std::mutex> lk(g_mu);
    hid = g_next++;
    g_handles[hid] = h;
  }
  auto out = paddle::empty({1}, paddle::DataType::INT64, paddle::CPUPlace());
  out.data<int64_t>()[0] = hid;
  return {out};
}

std::vector<paddle::Tensor> zcomm_intra_alloc(const paddle::Tensor& handle,
                                              int64_t bytes) {
  auto h = get_handle(handle);
  zcomm::SymmBuffer b =
      h->zc->alloc((size_t)bytes, h->zc->transport().supports_multicast());
  {
    std::lock_guard<std::mutex> lk(h->mu);
    h->bufs[b.local] = b;
  }
  auto t = paddle::from_blob(
      b.local,
      {bytes},
      paddle::DataType::UINT8,
      phi::DataLayout::NCHW,
      phi::GPUPlace(phi::backends::gpu::GetCurrentDeviceId()));
  // No deleter: the buffer is resident (see the file header). A deleter would
  // register the pointer in Paddle's global table, and wrapping the same
  // address again (zcomm_buffer_view) would then conflict.
  return {t};
}

void zcomm_intra_all_gather(const paddle::Tensor& handle,
                            const paddle::Tensor& x,
                            paddle::Tensor& out_buf,
                            int64_t stream,
                            int64_t token) {
  auto h = get_handle(handle);
  PD_CHECK(x.is_gpu() && out_buf.is_gpu(),
           "zcomm_intra_all_gather needs GPU tensors");
  const size_t shard = nbytes(x);
  const auto& ob = find_buf(*h, out_buf, shard * h->zc->nranks());
  const cudaStream_t s = pick_stream(stream, x);
  h->zc->all_gather_ce(x.data(), ob, shard, s);
  zcomm_ops::record(x, s, token);
}

void zcomm_intra_all_gather_pull(const paddle::Tensor& handle,
                                 const paddle::Tensor& x,
                                 const paddle::Tensor& stage,
                                 paddle::Tensor& out,
                                 int64_t stream,
                                 int64_t token) {
  auto h = get_handle(handle);
  PD_CHECK(x.is_gpu() && out.is_gpu(),
           "zcomm_intra_all_gather_pull needs GPU tensors");
  const size_t shard = nbytes(x);
  PD_CHECK(nbytes(out) == shard * h->zc->nranks(),
           "out must hold nranks shards");
  const auto& sb = find_buf(*h, stage, shard);
  const cudaStream_t s = pick_stream(stream, x);
  h->zc->all_gather_ce_pull(x.data(), sb, out.data(), shard, s);
  zcomm_ops::record(x, s, token);
  zcomm_ops::record(out, s, token);
}

void zcomm_intra_reduce_scatter(const paddle::Tensor& handle,
                                const paddle::Tensor& x,
                                const paddle::Tensor& stage,
                                paddle::Tensor& out,
                                bool ring_acc,
                                int64_t sum_blocks,
                                int64_t stream,
                                int64_t token) {
  auto h = get_handle(handle);
  PD_CHECK(x.dtype() == paddle::DataType::BFLOAT16 &&
               out.dtype() == paddle::DataType::BFLOAT16,
           "zcomm_intra_reduce_scatter supports bf16 only");
  const size_t total = nbytes(x);
  const int n = h->zc->nranks();
  PD_CHECK(total % n == 0, "input bytes not divisible by nranks");
  const size_t shard = total / n;
  PD_CHECK(nbytes(out) == shard, "out must hold exactly one shard");
  const auto& sb = find_buf(*h, stage, total);
  const cudaStream_t s = pick_stream(stream, x);
  h->zc->reduce_scatter_ce(x.data(),
                           sb,
                           out.data(),
                           shard,
                           zcomm::DType::kBF16,
                           ring_acc ? zcomm::Acc::kRingBF16 : zcomm::Acc::kFP32,
                           s,
                           (int)sum_blocks);
  zcomm_ops::record(x, s, token);
  zcomm_ops::record(out, s, token);
}

__device__ __forceinline__ float to_f(float x) { return x; }
__device__ __forceinline__ float to_f(__nv_bfloat16 x) {
  return __bfloat162float(x);
}
__device__ __forceinline__ void from_f(float v, float* o) { *o = v; }
__device__ __forceinline__ void from_f(float v, __nv_bfloat16* o) {
  *o = __float2bfloat16(v);
}

template <typename T>
__global__ void stream_add_kernel(const T* __restrict__ a,
                                  const T* __restrict__ b,
                                  T* __restrict__ out,
                                  int64_t n) {
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n;
       i += (int64_t)gridDim.x * blockDim.x)
    from_f(to_f(a[i]) + to_f(b[i]), out + i);
}

// out = a + b on `stream` (bf16: summed in fp32 and rounded to nearest, as
// Paddle's add does). For sums whose inputs land on a comm stream, so the calc
// stream need not wait; `out` must be allocated, and the stream ordered after
// its allocation, by the caller.
void zcomm_stream_add(const paddle::Tensor& a,
                      const paddle::Tensor& b,
                      paddle::Tensor& out,
                      int64_t stream,
                      int64_t token) {
  const auto dt = a.dtype();
  PD_CHECK(
      (dt == paddle::DataType::BFLOAT16 || dt == paddle::DataType::FLOAT32) &&
          b.dtype() == dt && out.dtype() == dt,
      "zcomm_stream_add supports bf16 / fp32 of one dtype");
  PD_CHECK(a.numel() == b.numel() && a.numel() == out.numel(),
           "zcomm_stream_add: size mismatch");
  const int64_t n = a.numel();
  if (n == 0) return;
  const int threads = 256;
  const int blocks = (int)std::min<int64_t>((n + threads - 1) / threads, 1024);
  cudaStream_t s = pick_stream(stream, a);
  if (dt == paddle::DataType::BFLOAT16)
    stream_add_kernel<<<blocks, threads, 0, s>>>(
        reinterpret_cast<const __nv_bfloat16*>(a.data()),
        reinterpret_cast<const __nv_bfloat16*>(b.data()),
        reinterpret_cast<__nv_bfloat16*>(out.data()),
        n);
  else
    stream_add_kernel<<<blocks, threads, 0, s>>>(
        reinterpret_cast<const float*>(a.data()),
        reinterpret_cast<const float*>(b.data()),
        reinterpret_cast<float*>(out.data()),
        n);
  const cudaError_t err = cudaGetLastError();
  PD_CHECK(err == cudaSuccess, "zcomm_stream_add: ", cudaGetErrorString(err));
  zcomm_ops::record(a, s, token);
  zcomm_ops::record(b, s, token);
  zcomm_ops::record(out, s, token);
}

// Withdraw the stream registrations of the call that got `token`; `stream`
// (the caller's current stream) has just waited for that call.
void zcomm_release_streams(paddle::Tensor& any, int64_t token, int64_t stream) {
  (void)any;
  zcomm_ops::release(token, reinterpret_cast<cudaStream_t>(stream));
}

// Wrap buffer[offset, offset + numel*sizeof(dtype)) into a tensor of the given
// dtype and shape (zero copy, it does not own the memory). Used to hand a range
// of a symmetric buffer downstream: it is not a slice, so data_ptr is the real
// address and there is no storage offset.
std::vector<paddle::Tensor> zcomm_buffer_view(const paddle::Tensor& buf,
                                              int64_t offset,
                                              const std::vector<int64_t>& shape,
                                              const std::string& dtype) {
  static const std::unordered_map<std::string, paddle::DataType> kTypes = {
      {"bfloat16", paddle::DataType::BFLOAT16},
      {"float16", paddle::DataType::FLOAT16},
      {"float32", paddle::DataType::FLOAT32},
      {"int32", paddle::DataType::INT32},
      {"int64", paddle::DataType::INT64},
      {"uint8", paddle::DataType::UINT8},
      {"float8_e4m3fn", paddle::DataType::FLOAT8_E4M3FN}};
  auto it = kTypes.find(dtype);
  PD_CHECK(it != kTypes.end(), "zcomm_buffer_view: unsupported dtype ", dtype);
  int64_t numel = 1;
  for (auto d : shape) numel *= d;
  PD_CHECK(offset >= 0 && offset + numel * (int64_t)phi::SizeOf(it->second) <=
                              buf.numel() * (int64_t)phi::SizeOf(buf.dtype()),
           "zcomm_buffer_view out of range");
  return {paddle::from_blob(
      static_cast<char*>(const_cast<void*>(buf.data())) + offset,
      shape,
      it->second,
      phi::DataLayout::NCHW,
      buf.place())};
}

PD_BUILD_OP(zcomm_buffer_view)
    .Inputs({"buf"})
    .Outputs({"out"})
    .Attrs({"offset: int64_t",
            "shape: std::vector<int64_t>",
            "dtype: std::string"})
    .SetKernelFn(PD_KERNEL(zcomm_buffer_view));

PD_BUILD_OP(zcomm_unique_id)
    .Inputs({"place_hint"})
    .Outputs({"uid"})
    .SetKernelFn(PD_KERNEL(zcomm_unique_id));

PD_BUILD_OP(zcomm_intra_init)
    .Inputs({"uid"})
    .Outputs({"handle"})
    .Attrs({"rank: int64_t", "nranks: int64_t"})
    .SetKernelFn(PD_KERNEL(zcomm_intra_init));

PD_BUILD_OP(zcomm_intra_alloc)
    .Inputs({"handle"})
    .Outputs({"buf"})
    .Attrs({"bytes: int64_t"})
    .SetKernelFn(PD_KERNEL(zcomm_intra_alloc));

PD_BUILD_OP(zcomm_intra_all_gather)
    .Inputs({"handle", "x", "out_buf"})
    .Outputs({"out"})
    .SetInplaceMap({{"out_buf", "out"}})
    .Attrs({"stream: int64_t", "token: int64_t"})
    .SetKernelFn(PD_KERNEL(zcomm_intra_all_gather));

PD_BUILD_OP(zcomm_intra_all_gather_pull)
    .Inputs({"handle", "x", "stage", "out_in"})
    .Outputs({"out"})
    .SetInplaceMap({{"out_in", "out"}})
    .Attrs({"stream: int64_t", "token: int64_t"})
    .SetKernelFn(PD_KERNEL(zcomm_intra_all_gather_pull));

PD_BUILD_OP(zcomm_intra_reduce_scatter)
    .Inputs({"handle", "x", "stage", "out_in"})
    .Outputs({"out"})
    .SetInplaceMap({{"out_in", "out"}})
    .Attrs({"ring_acc: bool",
            "sum_blocks: int64_t",
            "stream: int64_t",
            "token: int64_t"})
    .SetKernelFn(PD_KERNEL(zcomm_intra_reduce_scatter));

PD_BUILD_OP(zcomm_stream_add)
    .Inputs({"a", "b", "out_in"})
    .Outputs({"out"})
    .SetInplaceMap({{"out_in", "out"}})
    .Attrs({"stream: int64_t", "token: int64_t"})
    .SetKernelFn(PD_KERNEL(zcomm_stream_add));

PD_BUILD_OP(zcomm_release_streams)
    .Inputs({"any_in"})
    .Outputs({"any"})
    .SetInplaceMap({{"any_in", "any"}})
    .Attrs({"token: int64_t", "stream: int64_t"})
    .SetKernelFn(PD_KERNEL(zcomm_release_streams));
