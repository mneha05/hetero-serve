"""Optimised paged-attention kernel: online softmax, one warp per (seq, head).

`paged_attn.py` holds v1 — correct, and deliberately naive. It stores the whole
score vector in shared memory, walks the context with scalar loads, and reduces
through a shared-memory tree. This is the version that takes the hardware
seriously.

Four changes, in descending order of how much they matter:

1. **Online softmax (the algorithmic one).** v1 needs O(context) shared memory to
   hold every score before it can normalise, which caps context length and
   forces a second pass over V. v2 keeps a running max `m` and running sum `l`
   and rescales the accumulator as it goes, exactly like FlashAttention:

       m_new = max(m, s)
       l     = l * exp(m - m_new) + exp(s - m_new)
       acc   = acc * exp(m - m_new) + exp(s - m_new) * v

   Shared memory becomes O(1) in context length, K and V are each read once, and
   the kernel stops caring how long the sequence is.

2. **One warp per (sequence, head).** 32 lanes cooperate on one head's 64-wide
   dot product, so every lane is busy. v1 gave a whole 128-thread block to one
   head and left half of it idle in the phase that dominates.

3. **Coalesced per-lane slices.** Each lane owns a *contiguous* `head_dim/32`
   slice, and consecutive lanes own consecutive slices, so one warp reading one
   context position issues a single 128-byte transaction (32 lanes x 2 halves x
   2 bytes at head_dim 64). The unrolled 2-element read also lets the compiler
   emit a single `half2`/`float2` load. v1 read one scalar at a time with a
   stride of head_dim between threads, which is close to the worst case.

4. **Warp-shuffle reductions.** `__shfl_down_sync` instead of a shared-memory
   tree: no `__syncthreads`, no shared traffic, no bank conflicts.

Decode attention is memory-bandwidth bound, so the number that actually matters
is achieved GB/s against the card's peak. `scripts/bench_kernel.py --compare-all`
reports exactly that.
"""

from __future__ import annotations

import os
from typing import Optional

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

#define WARP_SIZE 32
#define FULL_MASK 0xffffffffu

__device__ __forceinline__ float warp_reduce_sum(float v) {
  #pragma unroll
  for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
    v += __shfl_down_sync(FULL_MASK, v, off);
  return v;
}

__device__ __forceinline__ float warp_reduce_max(float v) {
  #pragma unroll
  for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
    v = fmaxf(v, __shfl_down_sync(FULL_MASK, v, off));
  return v;
}

// One warp per (sequence, head). Each warp streams the whole context once,
// maintaining an online softmax so no score vector is ever materialised.
//
// Lane layout: head_dim is split across the warp, so lane i owns dims
// [i*VPT, (i+1)*VPT) where VPT = head_dim / 32. For head_dim 64 that is 2 dims
// per lane, read as one half2/float2.
template <typename scalar_t, int HEAD_DIM>
__global__ void paged_attention_v2_kernel(
    float* __restrict__ out,                  // [B, H, D]
    const scalar_t* __restrict__ k_cache,     // [NB, BS, H, D]
    const scalar_t* __restrict__ v_cache,     // [NB, BS, H, D]
    const float* __restrict__ q,              // [B, H, D]
    const int* __restrict__ block_tables,     // [B, MB]
    const int* __restrict__ context_lens,     // [B]
    const int num_seqs,
    const int num_heads,
    const int block_size,
    const int max_blocks,
    const float scale) {

  constexpr int VPT = HEAD_DIM / WARP_SIZE;   // values per thread

  const int warps_per_block = blockDim.y;
  const int warp_id = threadIdx.y;
  const int lane = threadIdx.x;

  // Flat warp id over (sequence, head). A 1-D grid only: pairing this with a
  // second grid dimension for batch would run every warp once per sequence.
  const int flat = blockIdx.x * warps_per_block + warp_id;
  if (flat >= num_seqs * num_heads) return;
  const int b = flat / num_heads;
  const int h = flat % num_heads;

  const int ctx_len = context_lens[b];
  if (ctx_len <= 0) return;

  const int* btab = block_tables + (size_t)b * max_blocks;
  const float* q_ptr = q + (size_t)(b * num_heads + h) * HEAD_DIM;

  // This lane's slice of q, held in registers for the whole stream.
  float q_reg[VPT];
  #pragma unroll
  for (int i = 0; i < VPT; ++i) q_reg[i] = q_ptr[lane * VPT + i];

  // Online softmax state, plus this lane's slice of the output accumulator.
  float m = -3.0e38f;      // running max
  float l = 0.0f;          // running sum of exp
  float acc[VPT];
  #pragma unroll
  for (int i = 0; i < VPT; ++i) acc[i] = 0.0f;

  for (int j = 0; j < ctx_len; ++j) {
    const int blk = btab[j / block_size];
    const int off = j % block_size;
    const size_t base = (((size_t)blk * block_size + off) * num_heads + h) * HEAD_DIM;

    // ---- score: warp-cooperative dot product, one shuffle reduction ----
    const scalar_t* k_ptr = k_cache + base + lane * VPT;
    float partial = 0.0f;
    #pragma unroll
    for (int i = 0; i < VPT; ++i) partial += q_reg[i] * static_cast<float>(k_ptr[i]);

    float s = warp_reduce_sum(partial);
    s = __shfl_sync(FULL_MASK, s, 0) * scale;   // broadcast lane 0's total

    // ---- online softmax rescale ----
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

  const float inv_l = 1.0f / l;
  float* o_ptr = out + (size_t)(b * num_heads + h) * HEAD_DIM + lane * VPT;
  #pragma unroll
  for (int i = 0; i < VPT; ++i) o_ptr[i] = acc[i] * inv_l;
}

torch::Tensor paged_attention_v2(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor block_tables,
    torch::Tensor context_lens,
    double scale) {

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

  auto out = torch::empty({B, H, D}, q.options().dtype(torch::kFloat32));

  const int warps_per_block = 4;
  const dim3 threads(WARP_SIZE, warps_per_block);
  const int total_warps = B * H;
  const dim3 grid((total_warps + warps_per_block - 1) / warps_per_block);

  #define LAUNCH(DIM)                                                          \
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(                                       \
        k_cache.scalar_type(), "paged_attention_v2", ([&] {                    \
          paged_attention_v2_kernel<scalar_t, DIM><<<grid, threads>>>(         \
              out.data_ptr<float>(), k_cache.data_ptr<scalar_t>(),             \
              v_cache.data_ptr<scalar_t>(), q.data_ptr<float>(),               \
              block_tables.data_ptr<int>(), context_lens.data_ptr<int>(),      \
              B, H, BS, MB, (float)scale);                                        \
        }));

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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("paged_attention_v2", &paged_attention_v2, "paged attention v2 (online softmax)");
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
            name="heteroserve_paged_attn_v2",
            cpp_sources="",
            cuda_sources=_CUDA_SRC,
            functions=["paged_attention_v2"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=bool(os.environ.get("HETEROSERVE_VERBOSE_BUILD")),
        )
        _state = "ready"
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        if "ninja" in msg.lower():
            # By far the most common failure, and easy to misread as "the CUDA
            # code is broken". It is not.
            msg = ("ninja is missing -- torch builds CUDA extensions through it. "
                   "Fix: pip install ninja")
        _ext, _state = None, "unavailable"
        _error = f"{type(exc).__name__}: {msg.splitlines()[-1][:200]}"


def is_available() -> bool:
    _try_build()
    return _state == "ready"


def build_error() -> Optional[str]:
    _try_build()
    return _error


def paged_attention_v2(q, k_cache, v_cache, block_tables, context_lens, scale):
    _try_build()
    if _state != "ready":
        raise RuntimeError(f"v2 kernel unavailable: {_error}")
    return _ext.paged_attention_v2(
        q, k_cache, v_cache, block_tables, context_lens, float(scale)
    )


def online_softmax_reference(q, k, v, scale):
    """The exact recurrence the v2 kernel runs, in numpy.

    Deliberately written as a *scalar streaming loop* rather than vectorised
    maths: it mirrors the kernel line for line, so if the algorithm is wrong the
    CPU test catches it without needing a GPU. The CUDA version is this, with
    the head_dim split across a warp.

        q [H, D]   k, v [n, H, D]  ->  out [H, D]
    """
    import numpy as np

    H, D = q.shape
    n = k.shape[0]
    m = np.full(H, -np.inf, dtype=np.float64)     # running max
    l = np.zeros(H, dtype=np.float64)             # running sum of exp
    acc = np.zeros((H, D), dtype=np.float64)

    for j in range(n):
        s = (q.astype(np.float64) * k[j].astype(np.float64)).sum(-1) * scale   # [H]
        m_new = np.maximum(m, s)
        correction = np.exp(m - m_new)
        # first iteration: m is -inf so correction is exp(-inf - finite) = 0
        correction = np.where(np.isfinite(m), correction, 0.0)
        p = np.exp(s - m_new)
        l = l * correction + p
        acc = acc * correction[:, None] + p[:, None] * v[j].astype(np.float64)
        m = m_new

    return (acc / l[:, None]).astype(np.float32)


# ---------------------------------------------------------------------------
# roofline
# ---------------------------------------------------------------------------


def bytes_moved(batch: int, context: int, n_heads: int, head_dim: int, itemsize: int) -> int:
    """Minimum bytes a decode attention *must* read: K and V, once each.

    Anything above this is the kernel re-reading data it should have kept, so
    achieved-vs-peak against this figure is the honest efficiency number.
    """
    return 2 * batch * context * n_heads * head_dim * itemsize


def peak_bandwidth_gbs() -> Optional[float]:
    """Theoretical peak HBM/GDDR bandwidth of the current device, GB/s."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        p = torch.cuda.get_device_properties(0)
        # memory_clock_rate is in kHz, bus width in bits, DDR => x2
        clock_khz = getattr(p, "memory_clock_rate", None)
        bus_bits = getattr(p, "memory_bus_width", None)
        if not clock_khz or not bus_bits:
            return None
        return (clock_khz * 1e3 * 2 * (bus_bits / 8)) / 1e9
    except Exception:
        return None
