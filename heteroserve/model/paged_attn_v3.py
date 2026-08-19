"""v3: context-split paged attention, because the profiler said occupancy.

Nsight on v2 (batch 16, GPT-2, T4):

    Memory Throughput          13.67 %
    Compute (SM) Throughput    11.34 %
    grid (48,1,1) x (32,4,1)
    OPT  This kernel grid is too small to fill the available resources on this
         device, resulting in only 0.1 full waves across all SMs.

Neither memory nor compute was saturated. v2 assigns one warp per (sequence,
head), so 16 sequences x 12 heads = 192 warps = 48 blocks — on a 40-SM card that
is a tenth of one wave. The kernel was not slow, there simply was not enough of
it in flight. No amount of inner-loop tuning fixes that.

v3 adds a *third* axis of parallelism: the context itself.

    split kernel   warp (b, h, s) streams only its slice of the context and
                   emits a partial (m, l, acc) — the online-softmax state, left
                   deliberately un-normalised
    merge kernel   one warp per (b, h) combines the partials:

                       m_g = max_s m_s
                       l_g = sum_s l_s * exp(m_s - m_g)
                       out = sum_s acc_s * exp(m_s - m_g) / l_g

That merge is the same rescale v2 already applies token by token — a running
softmax is associative, which is exactly why it can be split at all. With 8
splits the grid goes from 48 blocks to 384.

Same approach as FlashDecoding, and as vLLM's paged-attention v2 kernel.

`num_splits` is chosen from the device's SM count so small batches split hard and
large batches barely split at all; pass it explicitly to sweep it.
"""

from __future__ import annotations

import os
from typing import Optional

_CPP_DECL = """
#include <torch/extension.h>
torch::Tensor paged_attention_v3(
    torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor block_tables, torch::Tensor context_lens, double scale,
    int64_t num_splits);
int64_t choose_num_splits(int64_t num_seqs, int64_t num_heads, int64_t max_context);
"""

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <algorithm>
#include <c10/cuda/CUDAException.h>

#define WARP_SIZE 32
#define FULL_MASK 0xffffffffu
#define NEG_BIG (-3.0e38f)

__device__ __forceinline__ float warp_reduce_sum(float v) {
  #pragma unroll
  for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
    v += __shfl_down_sync(FULL_MASK, v, off);
  return v;
}

// ---------------------------------------------------------------------------
// pass 1: each warp owns (sequence, head, split) and streams only its slice
// ---------------------------------------------------------------------------
template <typename scalar_t, int HEAD_DIM>
__global__ void paged_attention_split_kernel(
    float* __restrict__ partial_out,          // [B, H, S, D]
    float* __restrict__ partial_m,            // [B, H, S]
    float* __restrict__ partial_l,            // [B, H, S]
    const scalar_t* __restrict__ k_cache,     // [NB, BS, H, D]
    const scalar_t* __restrict__ v_cache,
    const float* __restrict__ q,              // [B, H, D]
    const int* __restrict__ block_tables,     // [B, MB]
    const int* __restrict__ context_lens,     // [B]
    const int num_seqs,
    const int num_heads,
    const int block_size,
    const int max_blocks,
    const int num_splits,
    const float scale) {

  constexpr int VPT = HEAD_DIM / WARP_SIZE;

  const int lane = threadIdx.x;
  const int flat = blockIdx.x * blockDim.y + threadIdx.y;
  if (flat >= num_seqs * num_heads * num_splits) return;

  const int split = flat % num_splits;
  const int hb = flat / num_splits;
  const int h = hb % num_heads;
  const int b = hb / num_heads;

  const int ctx_len = context_lens[b];
  const int chunk = (ctx_len + num_splits - 1) / num_splits;
  const int start = split * chunk;
  const int end = min(start + chunk, ctx_len);

  float m = NEG_BIG;
  float l = 0.0f;
  float acc[VPT];
  #pragma unroll
  for (int i = 0; i < VPT; ++i) acc[i] = 0.0f;

  // An empty slice (short sequence, many splits) falls through with m = -big
  // and l = 0, which the merge weights to exactly zero. No special case needed.
  if (start < end) {
    const float* q_ptr = q + (size_t)(b * num_heads + h) * HEAD_DIM;
    float q_reg[VPT];
    #pragma unroll
    for (int i = 0; i < VPT; ++i) q_reg[i] = q_ptr[lane * VPT + i];

    const int* btab = block_tables + (size_t)b * max_blocks;

    for (int j = start; j < end; ++j) {
      const int blk = btab[j / block_size];
      const int off = j % block_size;
      const size_t base = (((size_t)blk * block_size + off) * num_heads + h) * HEAD_DIM;

      const scalar_t* k_ptr = k_cache + base + lane * VPT;
      float partial = 0.0f;
      #pragma unroll
      for (int i = 0; i < VPT; ++i) partial += q_reg[i] * static_cast<float>(k_ptr[i]);

      float s = warp_reduce_sum(partial);
      s = __shfl_sync(FULL_MASK, s, 0) * scale;

      const float m_new = fmaxf(m, s);
      const float correction = __expf(m - m_new);
      const float p = __expf(s - m_new);
      l = l * correction + p;

      const scalar_t* v_ptr = v_cache + base + lane * VPT;
      #pragma unroll
      for (int i = 0; i < VPT; ++i)
        acc[i] = acc[i] * correction + p * static_cast<float>(v_ptr[i]);

      m = m_new;
    }
  }

  const size_t pidx = (size_t)(b * num_heads + h) * num_splits + split;
  if (lane == 0) {
    partial_m[pidx] = m;
    partial_l[pidx] = l;
  }
  // Deliberately un-normalised: dividing by l here would lose the information
  // the merge needs to reweight this slice against the others.
  float* po = partial_out + pidx * HEAD_DIM + lane * VPT;
  #pragma unroll
  for (int i = 0; i < VPT; ++i) po[i] = acc[i];
}

// ---------------------------------------------------------------------------
// pass 2: combine the per-split softmax states. Cheap: S values per (b, h).
// ---------------------------------------------------------------------------
template <int HEAD_DIM>
__global__ void merge_splits_kernel(
    float* __restrict__ out,                  // [B, H, D]
    const float* __restrict__ partial_out,    // [B, H, S, D]
    const float* __restrict__ partial_m,      // [B, H, S]
    const float* __restrict__ partial_l,      // [B, H, S]
    const int num_seqs,
    const int num_heads,
    const int num_splits) {

  constexpr int VPT = HEAD_DIM / WARP_SIZE;

  const int lane = threadIdx.x;
  const int flat = blockIdx.x * blockDim.y + threadIdx.y;
  if (flat >= num_seqs * num_heads) return;

  const size_t base = (size_t)flat * num_splits;

  float m_g = NEG_BIG;
  for (int s = 0; s < num_splits; ++s) m_g = fmaxf(m_g, partial_m[base + s]);

  float l_g = 0.0f;
  float acc[VPT];
  #pragma unroll
  for (int i = 0; i < VPT; ++i) acc[i] = 0.0f;

  for (int s = 0; s < num_splits; ++s) {
    const float w = __expf(partial_m[base + s] - m_g);
    l_g += partial_l[base + s] * w;
    const float* po = partial_out + (base + s) * HEAD_DIM + lane * VPT;
    #pragma unroll
    for (int i = 0; i < VPT; ++i) acc[i] += po[i] * w;
  }

  const float inv = 1.0f / l_g;
  float* o = out + (size_t)flat * HEAD_DIM + lane * VPT;
  #pragma unroll
  for (int i = 0; i < VPT; ++i) o[i] = acc[i] * inv;
}

// ---------------------------------------------------------------------------

int64_t choose_num_splits(int64_t num_seqs, int64_t num_heads, int64_t max_context) {
  // Enough warps to give every SM a few, but never so many that a slice gets
  // too short to amortise the merge.
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int64_t want_warps = (int64_t)sms * 8;
  const int64_t have = std::max<int64_t>(num_seqs * num_heads, 1);
  int64_t splits = (want_warps + have - 1) / have;
  const int64_t by_ctx = std::max<int64_t>(max_context / 128, 1);
  splits = std::min(splits, by_ctx);
  return std::max<int64_t>(1, std::min<int64_t>(splits, 32));
}

torch::Tensor paged_attention_v3(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor block_tables,
    torch::Tensor context_lens,
    double scale,
    int64_t num_splits) {

  TORCH_CHECK(q.is_cuda(), "q must be on CUDA");
  q = q.contiguous();
  k_cache = k_cache.contiguous();
  v_cache = v_cache.contiguous();
  block_tables = block_tables.to(torch::kInt32).contiguous();
  context_lens = context_lens.to(torch::kInt32).contiguous();

  const int B = q.size(0);
  const int H = q.size(1);
  const int D = q.size(2);
  const int BS = k_cache.size(1);
  const int MB = block_tables.size(1);

  TORCH_CHECK(D % 32 == 0, "head_dim must be a multiple of the warp size");

  if (num_splits <= 0) num_splits = choose_num_splits(B, H, (int64_t)MB * BS);

  auto fopts = q.options().dtype(torch::kFloat32);
  auto out = torch::empty({B, H, D}, fopts);
  auto partial_out = torch::empty({B, H, (int)num_splits, D}, fopts);
  auto partial_m = torch::empty({B, H, (int)num_splits}, fopts);
  auto partial_l = torch::empty({B, H, (int)num_splits}, fopts);

  const int warps_per_block = 4;
  const dim3 threads(WARP_SIZE, warps_per_block);

  const int split_warps = B * H * (int)num_splits;
  const dim3 split_grid((split_warps + warps_per_block - 1) / warps_per_block);
  const int merge_warps = B * H;
  const dim3 merge_grid((merge_warps + warps_per_block - 1) / warps_per_block);

  #define LAUNCH(DIM)                                                           \
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(                                        \
        k_cache.scalar_type(), "paged_attention_v3", ([&] {                     \
          paged_attention_split_kernel<scalar_t, DIM>                           \
              <<<split_grid, threads>>>(                                        \
                  partial_out.data_ptr<float>(), partial_m.data_ptr<float>(),   \
                  partial_l.data_ptr<float>(), k_cache.data_ptr<scalar_t>(),    \
                  v_cache.data_ptr<scalar_t>(), q.data_ptr<float>(),            \
                  block_tables.data_ptr<int>(), context_lens.data_ptr<int>(),   \
                  B, H, BS, MB, (int)num_splits, (float)scale);                 \
        }));                                                                    \
    merge_splits_kernel<DIM><<<merge_grid, threads>>>(                          \
        out.data_ptr<float>(), partial_out.data_ptr<float>(),                   \
        partial_m.data_ptr<float>(), partial_l.data_ptr<float>(),               \
        B, H, (int)num_splits);

  switch (D) {
    case 32:  LAUNCH(32);  break;
    case 64:  LAUNCH(64);  break;
    case 128: LAUNCH(128); break;
    case 256: LAUNCH(256); break;
    default: TORCH_CHECK(false, "unsupported head_dim ", D);
  }
  #undef LAUNCH

  C10_CUDA_CHECK(cudaGetLastError());
  return out;
}
"""

_ext = None
_state = "unbuilt"
_error: Optional[str] = None


def _try_build():
    global _ext, _state, _error
    if _state != "unbuilt":
        return
    try:
        import torch
        from torch.utils.cpp_extension import load_inline
    except ImportError as exc:
        _state, _error = "unavailable", f"torch unavailable: {exc}"
        return
    if not torch.cuda.is_available():
        _state, _error = "unavailable", "no CUDA device"
        return
    try:
        _ext = load_inline(
            name="heteroserve_paged_attn_v3",
            cpp_sources=_CPP_DECL,
            cuda_sources=_CUDA_SRC,
            functions=["paged_attention_v3", "choose_num_splits"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=bool(os.environ.get("HETEROSERVE_VERBOSE_BUILD")),
        )
        _state = "ready"
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        if "ninja is required to load c++ extensions" in msg.lower():
            msg = "ninja is missing -- fix: pip install ninja (or apt install ninja-build)"
        _ext, _state = None, "unavailable"
        _error = f"{type(exc).__name__}: {msg.splitlines()[-1][:200]}"


def is_available() -> bool:
    _try_build()
    return _state == "ready"


def build_error() -> Optional[str]:
    _try_build()
    return _error


def choose_num_splits(num_seqs: int, num_heads: int, max_context: int) -> int:
    _try_build()
    if _state != "ready":
        raise RuntimeError(f"v3 kernel unavailable: {_error}")
    return int(_ext.choose_num_splits(num_seqs, num_heads, max_context))


def paged_attention_v3(q, k_cache, v_cache, block_tables, context_lens, scale,
                       num_splits: int = 0):
    """num_splits <= 0 picks a value from the device's SM count."""
    _try_build()
    if _state != "ready":
        raise RuntimeError(f"v3 kernel unavailable: {_error}")
    return _ext.paged_attention_v3(
        q, k_cache, v_cache, block_tables, context_lens, float(scale), int(num_splits)
    )


def split_merge_reference(q, k, v, scale, num_splits: int):
    """The split-then-merge algorithm in numpy, mirroring both kernels.

    Verifies the associativity claim without a GPU: splitting the context,
    running an independent online softmax per slice, and merging the states must
    equal ordinary attention over the whole context.
    """
    import numpy as np

    H, D = q.shape
    n = k.shape[0]
    chunk = -(-n // num_splits)

    ms, ls, accs = [], [], []
    for s in range(num_splits):
        lo, hi = s * chunk, min((s + 1) * chunk, n)
        m = np.full(H, -np.inf)
        l = np.zeros(H)
        acc = np.zeros((H, D))
        for j in range(lo, hi):
            sc = (q.astype(np.float64) * k[j].astype(np.float64)).sum(-1) * scale
            m_new = np.maximum(m, sc)
            corr = np.where(np.isfinite(m), np.exp(m - m_new), 0.0)
            p = np.exp(sc - m_new)
            l = l * corr + p
            acc = acc * corr[:, None] + p[:, None] * v[j].astype(np.float64)
            m = m_new
        ms.append(m)
        ls.append(l)
        accs.append(acc)

    m_g = np.max(np.stack(ms), axis=0)
    l_g = np.zeros(H)
    out = np.zeros((H, D))
    for m, l, acc in zip(ms, ls, accs):
        w = np.where(np.isfinite(m), np.exp(m - m_g), 0.0)
        l_g += l * w
        out += acc * w[:, None]
    return (out / l_g[:, None]).astype(np.float32)
