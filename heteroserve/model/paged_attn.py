"""Fused paged-attention CUDA kernel.

Why this exists: profiling the decode step (`scripts/profile_decode.py`) showed
that ~33% of a GPU decode step was *not* accelerator time at all. It was the host
gathering each sequence's KV blocks into contiguous tensors before the engine
could touch them — pure overhead created by paging the cache.

    device    gather   engine   write    total   gather share
    Arc GPU   44.0ms   89.3ms   0.4ms   133.7ms          33%

This kernel removes that gather. It walks each sequence's block table inside the
attention loop, reading K and V straight out of the paged pool, so the
contiguous copy never happens. Same idea as vLLM's paged attention.

Layout it expects, matching `BlockAllocator.pool[layer, 0]`:

    k_cache / v_cache   [num_blocks, block_size, num_heads, head_dim]
    block_tables        [batch, max_blocks]  int32, -1 for unused slots
    context_lens        [batch]              int32

Three implementations, in preference order:
  1. the CUDA kernel below (needs nvcc + a CUDA GPU)
  2. a pure-torch paged path, still gather-free per step but not fused
  3. nothing — callers fall back to the gather-based engine

`which_backend()` reports which one is live, so a benchmark can never silently
claim a kernel result it did not get.
"""

from __future__ import annotations

import os
from typing import Optional

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

// One CUDA block per (sequence, head). Threads cooperate over the context.
//
// The block table indirection is the whole point: position j of sequence b
// lives at block_tables[b][j / block_size], offset j % block_size. Nothing is
// ever copied into a contiguous buffer first.
template <typename scalar_t>
__global__ void paged_attention_kernel(
    float* __restrict__ out,                  // [B, H, D]
    const scalar_t* __restrict__ k_cache,     // [NB, BS, H, D]
    const scalar_t* __restrict__ v_cache,     // [NB, BS, H, D]
    const float* __restrict__ q,              // [B, H, D]
    const int* __restrict__ block_tables,     // [B, MB]
    const int* __restrict__ context_lens,     // [B]
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const int block_size,
    const int max_blocks,
    const int max_context,
    const float scale) {

  const int b = blockIdx.x;
  const int h = blockIdx.y;
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;

  // GQA: query head h reads the KV head it shares with its group.
  const int kv_h = h / (num_heads / num_kv_heads);

  const int ctx_len = context_lens[b];
  if (ctx_len <= 0) return;

  // scores[0..max_context) then a reduction scratchpad of nthreads floats.
  extern __shared__ float smem[];
  float* scores = smem;
  float* red = smem + max_context;

  const float* q_ptr = q + (size_t)(b * num_heads + h) * head_dim;
  const int* btab = block_tables + (size_t)b * max_blocks;

  // ---- 1. q . k for every cached position, straight out of the pages ----
  for (int j = tid; j < ctx_len; j += nthreads) {
    const int blk = btab[j / block_size];
    const int off = j % block_size;
    const scalar_t* k_ptr =
        k_cache + (((size_t)blk * block_size + off) * num_kv_heads + kv_h) * head_dim;
    float acc = 0.f;
    for (int d = 0; d < head_dim; ++d) {
      acc += q_ptr[d] * static_cast<float>(k_ptr[d]);
    }
    scores[j] = acc * scale;
  }
  __syncthreads();

  // ---- 2. softmax, in two block-wide reductions ----
  float local = -3.0e38f;   // effectively -inf for fp32 scores
  for (int j = tid; j < ctx_len; j += nthreads) local = fmaxf(local, scores[j]);
  red[tid] = local;
  __syncthreads();
  for (int s = nthreads >> 1; s > 0; s >>= 1) {
    if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
    __syncthreads();
  }
  const float mx = red[0];
  __syncthreads();

  float partial = 0.f;
  for (int j = tid; j < ctx_len; j += nthreads) {
    const float e = __expf(scores[j] - mx);
    scores[j] = e;
    partial += e;
  }
  red[tid] = partial;
  __syncthreads();
  for (int s = nthreads >> 1; s > 0; s >>= 1) {
    if (tid < s) red[tid] += red[tid + s];
    __syncthreads();
  }
  const float denom = red[0];
  __syncthreads();

  // ---- 3. weighted sum over V, one thread per output dim ----
  for (int d = tid; d < head_dim; d += nthreads) {
    float acc = 0.f;
    for (int j = 0; j < ctx_len; ++j) {
      const int blk = btab[j / block_size];
      const int off = j % block_size;
      const scalar_t* v_ptr =
          v_cache + (((size_t)blk * block_size + off) * num_kv_heads + kv_h) * head_dim;
      acc += scores[j] * static_cast<float>(v_ptr[d]);
    }
    out[(size_t)(b * num_heads + h) * head_dim + d] = acc / denom;
  }
}

torch::Tensor paged_attention(
    torch::Tensor q,             // [B, H, D] float32, contiguous
    torch::Tensor k_cache,       // [NB, BS, H, D]
    torch::Tensor v_cache,       // [NB, BS, H, D]
    torch::Tensor block_tables,  // [B, MB] int32
    torch::Tensor context_lens,  // [B] int32
    double scale) {

  TORCH_CHECK(q.is_cuda(), "q must be on CUDA");
  TORCH_CHECK(k_cache.is_cuda() && v_cache.is_cuda(), "kv cache must be on CUDA");
  TORCH_CHECK(q.dim() == 3, "q must be [B, H, D]");
  TORCH_CHECK(k_cache.dim() == 4, "k_cache must be [NB, BS, H, D]");

  q = q.contiguous();
  k_cache = k_cache.contiguous();
  v_cache = v_cache.contiguous();
  block_tables = block_tables.to(torch::kInt32).contiguous();
  context_lens = context_lens.to(torch::kInt32).contiguous();

  const int B = q.size(0);
  const int H = q.size(1);
  const int D = q.size(2);
  const int BS = k_cache.size(1);
  const int HKV = k_cache.size(2);
  const int MB = block_tables.size(1);

  const int max_context = MB * BS;
  auto out = torch::empty({B, H, D}, q.options().dtype(torch::kFloat32));

  const int threads = 128;
  const dim3 grid(B, H);
  const size_t shmem = (size_t)(max_context + threads) * sizeof(float);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      k_cache.scalar_type(), "paged_attention", ([&] {
        paged_attention_kernel<scalar_t><<<grid, threads, shmem>>>(
            out.data_ptr<float>(),
            k_cache.data_ptr<scalar_t>(),
            v_cache.data_ptr<scalar_t>(),
            q.data_ptr<float>(),
            block_tables.data_ptr<int>(),
            context_lens.data_ptr<int>(),
            H, HKV, D, BS, MB, max_context, (float)scale);
      }));

  C10_CUDA_CHECK(cudaGetLastError());
  return out;
}

"""

# load_inline(functions=[...]) generates its own pybind module, and that
# generated main.cpp needs to *see* the function. The definition lives in
# the .cu source, so the declaration has to be handed over as cpp_sources.
_CPP_DECL = """
#include <torch/extension.h>
torch::Tensor paged_attention(
    torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor block_tables, torch::Tensor context_lens, double scale);
"""

_ext = None
_backend = "none"
_error: Optional[str] = None


def _try_build():
    """Compile the kernel once, on first use. Cached by torch under ~/.cache."""
    global _ext, _backend, _error
    if _backend != "none":
        return

    if os.environ.get("HETEROSERVE_DISABLE_CUDA_KERNEL"):
        _backend, _error = "torch", "disabled by HETEROSERVE_DISABLE_CUDA_KERNEL"
        return

    try:
        import torch
        from torch.utils.cpp_extension import load_inline
    except ImportError as exc:
        _backend, _error = "none", f"torch unavailable: {exc}"
        return

    if not torch.cuda.is_available():
        _backend, _error = "torch", "no CUDA device"
        return

    try:
        _ext = load_inline(
            name="heteroserve_paged_attn",
            cpp_sources=_CPP_DECL,
            cuda_sources=_CUDA_SRC,
            functions=["paged_attention"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=bool(os.environ.get("HETEROSERVE_VERBOSE_BUILD")),
        )
        _backend = "cuda"
    except Exception as exc:  # noqa: BLE001 - compilation is genuinely optional
        msg = str(exc)
        if "ninja is required to load c++ extensions" in msg.lower():
            # By far the most common failure, and the message torch gives is easy
            # to misread as "the CUDA code is broken". It is not.
            msg = ("ninja is missing -- torch builds CUDA extensions through it. "
                   "Fix: pip install ninja")
        _ext, _backend = None, "torch"
        _error = f"{type(exc).__name__}: {msg.splitlines()[-1][:200]}"


def which_backend() -> str:
    """'cuda' (fused kernel), 'torch' (fallback), or 'none' (no torch at all)."""
    _try_build()
    return _backend


def build_error() -> Optional[str]:
    _try_build()
    return _error


def paged_attention_torch(q, k_cache, v_cache, block_tables, context_lens, scale):
    """Reference paged attention in pure torch.

    Still block-table driven — it indexes the pool rather than requiring the
    caller to hand over a pre-gathered contiguous past — but it materialises an
    intermediate, so it does not save the memory traffic the kernel saves. Used
    as the correctness oracle for the kernel and as a fallback.
    """
    import torch

    B, _, D = q.shape
    NB, BS, H, _ = k_cache.shape          # H here is the *KV* head count
    MB = block_tables.shape[1]

    # [B, MB, BS] -> flat positions into a [NB*BS, H, D] view of the pool
    flat_k = k_cache.reshape(NB * BS, H, D)
    flat_v = v_cache.reshape(NB * BS, H, D)

    tables = block_tables.clamp(min=0).to(torch.long)                  # [B, MB]
    slots = tables.unsqueeze(-1) * BS + torch.arange(
        BS, device=q.device, dtype=torch.long
    )                                                                   # [B, MB, BS]
    slots = slots.reshape(B, MB * BS)

    k = flat_k[slots]                       # [B, MB*BS, Hkv, D]
    v = flat_v[slots]

    # GQA: several query heads share one KV head. The reference materialises the
    # expansion for clarity; the CUDA kernels index the shared head instead,
    # which is where the bandwidth saving actually comes from.
    Hq = q.shape[1]
    if Hq != H:
        g = Hq // H
        k = k.repeat_interleave(g, dim=2)
        v = v.repeat_interleave(g, dim=2)

    scores = torch.einsum("bhd,bkhd->bhk", q, k.float()) * scale        # [B, Hq, K]

    positions = torch.arange(MB * BS, device=q.device).unsqueeze(0)     # [1, K]
    valid = positions < context_lens.unsqueeze(1).to(positions.dtype)   # [B, K]
    scores = scores.masked_fill(~valid.unsqueeze(1), float("-inf"))

    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("bhk,bkhd->bhd", probs, v.float())


def paged_attention(q, k_cache, v_cache, block_tables, context_lens, scale):
    """Fused kernel when available, torch reference otherwise."""
    _try_build()
    if _backend == "cuda" and _ext is not None:
        return _ext.paged_attention(
            q, k_cache, v_cache, block_tables, context_lens, float(scale)
        )
    return paged_attention_torch(q, k_cache, v_cache, block_tables, context_lens, scale)
