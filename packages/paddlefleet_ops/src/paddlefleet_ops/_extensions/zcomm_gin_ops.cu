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

// zcomm_gin_ops.cu -- inter-node communication ops: NCCL-GIN post-and-leave
// copies (all_to_all_single / all_gather).
//
//   zcomm_gin_init(uid, rank, nranks, nctx, blocks, slices) -> int64[1] handle
//                                            (NCCL comm + GIN devcomm)
//   zcomm_gin_alloc(h, nbytes)             -> uint8[nbytes] window buffer
//                                            (collective; resident, see the
//                                            header of zcomm_intra_ops.cu)
//   zcomm_gin_alltoall(h, x, buf, plan, stage_dst_off, stream)
//                                            x is first copied into
//                                            buf[stage_dst_off..] by a copy
//                                            engine, then put to the peers
//                                            given by plan
//   plan = int64[1 + 4*nsend + 1 + narrive]:
//          nsend, (peer, src_off, dst_off, bytes) * nsend, narrive, peers...
//          Offsets are byte offsets inside buf, dst_off into the peer's buf.
//          An all_gather is just a plan (one range sent to every peer).
//
// The kernels are the same as in the z-comm experiment: k_post (arrive
// handshake
// + gin.put + VA signal +1, fire and forget) then k_finish (wait until the data
// signals arrived, flush every context). Signals are cumulative and never
// reset; VA signals are used because several contexts add to the same location.
#include <nccl.h>
#include <nccl_device.h>
#include <paddle/extension.h>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "zcomm/record_stream.h"

namespace {

constexpr int kMaxPeers = 16;  // plan is passed by value; smaller is cheaper to
                               // launch (RingMoE inter groups are small)
constexpr size_t kSigBytes = 4096;
// Data signals count every incoming slice. Arrivals are counted per peer: the
// signal of peer p lands at kOffArrive + 8 * p, so an arrival a peer sends for
// a later call can never be taken for the one this call waits for (a single
// counter would allow that whenever a peer can finish a call without this rank,
// e.g. a shift followed by a call to other peers).
constexpr size_t kOffData = 0, kOffArrive = 64;

struct Plan {  // per-peer description of one call, passed to the kernel by
               // value
  int me, npeers_send, npeers_wait_arrive;
  int peer[kMaxPeers];  // peers this rank writes to
  uint64_t src_off[kMaxPeers], dst_off[kMaxPeers],
      bytes[kMaxPeers];            // byte offsets inside the window
  uint64_t arrive_tgt[kMaxPeers];  // peer[k]'s arrivals to wait for
  int arrive_to[kMaxPeers];        // peers that write to me (I signal them)
};

__device__ __forceinline__ uint64_t gtimer() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
constexpr uint64_t kTimeoutNs = 30ull * 1000 * 1000 * 1000;

__device__ __forceinline__ void wait_arrive(ncclGin& gin,
                                            ncclWindow_t wsig,
                                            int peer,
                                            uint64_t target) {
  const uint64_t t0 = gtimer();
  while (gin.readSignal(wsig, kOffArrive + 8 * (size_t)peer) < target)
    if (gtimer() - t0 > kTimeoutNs) {
      printf("[zgin] arrive timeout (peer %d)\n", peer);
      __trap();
    }
}

// A call that sends nothing only tells its senders it has arrived.
__global__ void k_arrive(ncclDevComm dc, ncclWindow_t wsig, Plan pl) {
  if (blockIdx.x) return;
  ncclGin gin(dc, 0);
  const ncclTeam team = ncclTeamWorld(dc);
  for (int i = threadIdx.x; i < pl.npeers_wait_arrive; i += blockDim.x)
    gin.signal(team,
               pl.arrive_to[i],
               ncclGin_VASignalInc{wsig, kOffArrive + 8 * (size_t)pl.me},
               ncclCoopThread());
}

// The arrive handshake is fused into post: block 0 signals the peers and every
// block waits for the arrivals itself before its first put (saves a launch).
__global__ void k_post(ncclDevComm dc,
                       ncclWindow_t wbuf,
                       ncclWindow_t wsig,
                       Plan pl,
                       int slices,
                       int nctx) {
  ncclGin gin(dc, (int)(blockIdx.x % (unsigned)nctx));
  const ncclTeam team = ncclTeamWorld(dc);
  if (blockIdx.x == 0) {
    ncclGin g0(dc, 0);
    for (int i = threadIdx.x; i < pl.npeers_wait_arrive; i += blockDim.x)
      g0.signal(team,
                pl.arrive_to[i],
                ncclGin_VASignalInc{wsig, kOffArrive + 8 * (size_t)pl.me},
                ncclCoopThread());
  }
  // Every block waits for all destinations before its first put, as before;
  // putting to each destination once it arrived measured ~0.7 ms slower.
  if (threadIdx.x == 0) {
    for (int k = 0; k < pl.npeers_send; ++k)
      wait_arrive(gin, wsig, pl.peer[k], pl.arrive_tgt[k]);
  }
  __syncthreads();
  const int total = pl.npeers_send * slices;
  for (int idx = blockIdx.x; idx < total; idx += gridDim.x) {
    const int pi = idx / slices, sj = idx % slices;
    const uint64_t b = pl.bytes[pi];
    const uint64_t sl = (b / slices + 15) / 16 * 16;
    const uint64_t off = (uint64_t)sj * sl;
    const uint64_t cnt = off >= b ? 0 : (off + sl > b ? b - off : sl);
    // Empty slices still need a signal (the peer counts slices) but no 0-byte
    // put.
    if (cnt)
      gin.put(team,
              pl.peer[pi],
              wbuf,
              pl.dst_off[pi] + off,
              wbuf,
              pl.src_off[pi] + off,
              cnt,
              ncclGin_VASignalInc{wsig, kOffData},
              ncclGin_None{},
              ncclCoopCta());
    else
      gin.signal(team,
                 pl.peer[pi],
                 ncclGin_VASignalInc{wsig, kOffData},
                 ncclCoopCta());
  }
}

__global__ void k_finish(ncclDevComm dc,
                         ncclWindow_t wsig,
                         uint64_t target,
                         int nctx) {
  ncclGin gin(dc, (int)blockIdx.x);
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    const uint64_t t0 = gtimer();
    while (gin.readSignal(wsig, kOffData) < target)
      if (gtimer() - t0 > kTimeoutNs) {
        printf("[zgin] data timeout: %" PRIu64 " < %" PRIu64 "\n",
               (uint64_t)gin.readSignal(wsig, kOffData),
               target);
        __trap();
      }
  }
  __syncthreads();
  gin.flush(ncclCoopCta());
}

#define ZG_NCCL_ENFORCE(x)                                                \
  do {                                                                    \
    ncclResult_t r_ = (x);                                                \
    PD_CHECK(r_ == ncclSuccess, #x, " failed: ", ncclGetErrorString(r_)); \
  } while (0)

struct GinHandle {
  ncclComm_t comm = nullptr;
  ncclDevComm dc{};
  void* sig = nullptr;
  ncclWindow_t wsig{};
  int rank = 0, nranks = 0, nctx = 0, blocks = 32, slices = 8;
  uint64_t data_target = 0;
  uint64_t arrive_cnt[kMaxPeers] = {};  // calls so far that sent to each peer
  std::mutex mu;
  std::unordered_map<const void*, std::pair<ncclWindow_t, size_t>>
      wins;  // buffer pointer -> window
};

std::mutex g_mu;
std::unordered_map<int64_t, std::shared_ptr<GinHandle>> g_handles;
int64_t g_next = 1;

std::shared_ptr<GinHandle> get_handle(const paddle::Tensor& h) {
  PD_CHECK(h.numel() == 1 && h.dtype() == paddle::DataType::INT64,
           "zcomm gin handle must be int64[1]");
  auto hc = h.is_gpu() ? h.copy_to(paddle::CPUPlace(), true) : h;
  const int64_t id = hc.data<int64_t>()[0];
  std::lock_guard<std::mutex> lk(g_mu);
  auto it = g_handles.find(id);
  PD_CHECK(it != g_handles.end(), "unknown zcomm gin handle ", id);
  return it->second;
}

}  // namespace

std::vector<paddle::Tensor> zcomm_gin_init(const paddle::Tensor& uid,
                                           int64_t rank,
                                           int64_t nranks,
                                           int64_t nctx,
                                           int64_t blocks,
                                           int64_t slices) {
  PD_CHECK(uid.numel() == 128 && uid.dtype() == paddle::DataType::UINT8,
           "uid must be uint8[128]");
  auto uc = uid.is_gpu() ? uid.copy_to(paddle::CPUPlace(), true) : uid;
  ncclUniqueId id;
  std::memcpy(&id, uc.data<uint8_t>(), 128);
  auto h = std::make_shared<GinHandle>();
  PD_CHECK(nranks <= kMaxPeers, "zcomm gin: at most ", kMaxPeers, " ranks");
  h->rank = (int)rank;
  h->nranks = (int)nranks;
  h->blocks = (int)blocks;
  h->slices = (int)slices;
  ZG_NCCL_ENFORCE(ncclCommInitRank(&h->comm, (int)nranks, id, (int)rank));
  ncclCommProperties_t pr = NCCL_COMM_PROPERTIES_INITIALIZER;
  ZG_NCCL_ENFORCE(ncclCommQueryProperties(h->comm, &pr));
  if (pr.ginType != NCCL_GIN_TYPE_GDAKI)
    fprintf(stderr,
            "[zcomm gin] rank %d: GIN backend=%d is not GDAKI(3); "
            "post-and-leave degrades\n",
            (int)rank,
            (int)pr.ginType);
  ZG_NCCL_ENFORCE(ncclMemAlloc(&h->sig, kSigBytes));
  PD_CHECK(cudaMemset(h->sig, 0, kSigBytes) == cudaSuccess &&
           cudaDeviceSynchronize() == cudaSuccess);
  ZG_NCCL_ENFORCE(ncclCommWindowRegister(
      h->comm, h->sig, kSigBytes, &h->wsig, NCCL_WIN_COLL_SYMMETRIC));
  ncclDevCommRequirements_t req = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  req.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
  req.ginContextCount = (int)nctx;
  ZG_NCCL_ENFORCE(ncclDevCommCreate(h->comm, &req, &h->dc));
  h->nctx = (int)h->dc.ginContextCount;
  PD_CHECK(h->nctx >= 1,
           "GIN is not enabled on this communicator (ginContextCount=0)");
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

std::vector<paddle::Tensor> zcomm_gin_alloc(const paddle::Tensor& handle,
                                            int64_t bytes) {
  auto h = get_handle(handle);
  void* p = nullptr;
  ncclWindow_t w{};
  ZG_NCCL_ENFORCE(ncclMemAlloc(&p, (size_t)bytes));
  ZG_NCCL_ENFORCE(ncclCommWindowRegister(
      h->comm, p, (size_t)bytes, &w, NCCL_WIN_COLL_SYMMETRIC));
  {
    std::lock_guard<std::mutex> lk(h->mu);
    h->wins[p] = {w, (size_t)bytes};
  }
  return {paddle::from_blob(
      p,
      {bytes},
      paddle::DataType::UINT8,
      phi::DataLayout::NCHW,
      phi::GPUPlace(
          phi::backends::gpu::GetCurrentDeviceId()))};  // resident, no deleter
}

void zcomm_gin_alltoall(const paddle::Tensor& handle,
                        const paddle::Tensor& x,
                        paddle::Tensor& buf,
                        const std::vector<int64_t>& plan,
                        int64_t stage_dst_off,
                        int64_t stream,
                        int64_t token) {
  auto h = get_handle(handle);
  ncclWindow_t win;
  size_t win_bytes;
  {
    std::lock_guard<std::mutex> lk(h->mu);
    auto it = h->wins.find(buf.data());
    PD_CHECK(it != h->wins.end(),
             "buf is not a zcomm gin window (use zcomm_gin_alloc, pass the "
             "whole buffer)");
    win = it->second.first;
    win_bytes = it->second.second;
  }
  PD_CHECK(!plan.empty(), "empty plan");
  Plan pl{};
  pl.me = h->rank;
  size_t i = 0;
  pl.npeers_send = (int)plan[i++];
  PD_CHECK(pl.npeers_send <= kMaxPeers &&
               plan.size() >= 2 + 4 * (size_t)pl.npeers_send,
           "bad plan");
  for (int k = 0; k < pl.npeers_send; ++k) {
    pl.peer[k] = (int)plan[i++];
    pl.src_off[k] = plan[i++];
    pl.dst_off[k] = plan[i++];
    pl.bytes[k] = plan[i++];
    PD_CHECK(pl.peer[k] >= 0 && pl.peer[k] < h->nranks && pl.peer[k] != h->rank,
             "bad plan peer");
    PD_CHECK(pl.src_off[k] + pl.bytes[k] <= win_bytes &&
                 pl.dst_off[k] + pl.bytes[k] <= win_bytes,
             "plan segment out of buffer");
  }
  pl.npeers_wait_arrive = (int)plan[i++];
  PD_CHECK(pl.npeers_wait_arrive <= kMaxPeers &&
               plan.size() == i + (size_t)pl.npeers_wait_arrive,
           "bad plan");
  for (int k = 0; k < pl.npeers_wait_arrive; ++k) {
    pl.arrive_to[k] = (int)plan[i++];
    PD_CHECK(pl.arrive_to[k] >= 0 && pl.arrive_to[k] < h->nranks &&
                 pl.arrive_to[k] != h->rank,
             "bad plan arrive peer");
  }
  auto s = stream ? reinterpret_cast<cudaStream_t>(stream) : x.stream();
  const size_t xb = x.numel() * phi::SizeOf(x.dtype());
  if (xb) {
    PD_CHECK(stage_dst_off + xb <= win_bytes,
             "x does not fit the buffer at stage_dst_off");
    PD_CHECK(cudaMemcpyAsync((char*)buf.data() + stage_dst_off,
                             x.data(),
                             xb,
                             cudaMemcpyDeviceToDevice,
                             s) == cudaSuccess);
    // x is only read by this staging copy (the puts read the window).
    zcomm_ops::record(x, s, token);
  }
  for (int k = 0; k < pl.npeers_send; ++k)
    pl.arrive_tgt[k] = ++h->arrive_cnt[pl.peer[k]];
  const int total = pl.npeers_send * h->slices;
  if (total)
    k_post<<<std::min(h->blocks, total), 256, 0, s>>>(
        h->dc, win, h->wsig, pl, h->slices, h->nctx);
  else
    k_arrive<<<1, 64, 0, s>>>(h->dc, h->wsig, pl);
  h->data_target += (uint64_t)pl.npeers_wait_arrive * h->slices;
  k_finish<<<h->nctx, 32, 0, s>>>(h->dc, h->wsig, h->data_target, h->nctx);
  PD_CHECK(cudaGetLastError() == cudaSuccess,
           "zcomm_gin_alltoall launch failed");
}

PD_BUILD_OP(zcomm_gin_init)
    .Inputs({"uid"})
    .Outputs({"handle"})
    .Attrs({"rank: int64_t",
            "nranks: int64_t",
            "nctx: int64_t",
            "blocks: int64_t",
            "slices: int64_t"})
    .SetKernelFn(PD_KERNEL(zcomm_gin_init));

PD_BUILD_OP(zcomm_gin_alloc)
    .Inputs({"handle"})
    .Outputs({"buf"})
    .Attrs({"bytes: int64_t"})
    .SetKernelFn(PD_KERNEL(zcomm_gin_alloc));

PD_BUILD_OP(zcomm_gin_alltoall)
    .Inputs({"handle", "x", "buf_in"})
    .Outputs({"buf"})
    .SetInplaceMap({{"buf_in", "buf"}})
    .Attrs({"plan: std::vector<int64_t>",
            "stage_dst_off: int64_t",
            "stream: int64_t",
            "token: int64_t"})
    .SetKernelFn(PD_KERNEL(zcomm_gin_alltoall));
