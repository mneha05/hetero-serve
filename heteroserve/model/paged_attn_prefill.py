"""Paged attention for the *prefill* phase: many query tokens, not one.

v1/v2/v3 all answer the decode question -- one query token against N cached
keys. That is only half of paged attention. Chunked prefill runs S query tokens
at once against the same paged cache, with a causal mask, and until now this
project fell back to gathering the cache into a contiguous tensor for that
phase. Which is precisely the copy the kernels exist to delete.

The shape of the problem is different enough to deserve its own kernel:

    decode    q [B, H, D]        memory-bound GEMV, one row against N keys
    prefill   q [B, S, H, D]     compute-heavy, S rows against N keys, causal

Two consequences follow from that difference.

**Parallelism is free here.** Decode needed v3's context split because B*H warps
could not fill the device -- Nsight measured 0.1 waves. Prefill has B*S*H warps;
at S=256 that is hundreds of times more work in flight, so this kernel keeps the
simple one-warp-per-(sequence, query, head) mapping and the occupancy problem
never arises.

**Causality is per query row.** Query s of sequence b sits at absolute position
`context_len[b] - S + s`, because a chunk is always the trailing tokens of the
context at the moment it is computed. Each warp therefore stops its stream at
its own position, which makes the causal mask a loop bound rather than a
comparison -- no masked work is done at all.

The chunk's KV must already be resident in the pool before this is called: the
caller writes it, then attends. Same ordering the decode path uses, and the
reason the mask is expressible as "stop at my own position".

Tensor cores belong here rather than in decode -- prefill is a real GEMM. The
block size of 16 lines up exactly with a 16x16x16 WMMA fragment, so a K tile is
one page; see the README for why that is the useful half of the idea and what is
still missing.
"""

from __future__ import annotations

import os
from typing import Optional

_CPP_DECL = """
#include <torch/extension.h>
torch::Tensor paged_attention_prefill(
    torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor block_tables, torch::Tensor context_lens, double scale);
"""

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
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

// One warp per (sequence, query token, head).
//
// Same online-softmax streaming as the decode kernels, with two changes: the
// grid carries a query axis, and each warp stops at its own absolute position,
// so the causal mask is the loop bound rather than a comparison.
template <typename scalar_t, int HEAD_DIM>
__global__ void paged_attention_prefill_kernel(
    float* __restrict__ out,                  // [B, S, H, D]
    const scalar_t* __restrict__ k_cache,     // [NB, BS, HKV, D]
    const scalar_t* __restrict__ v_cache,
    const float* __restrict__ q,              // [B, S, H, D]
    const int* __restrict__ block_tables,     // [B, MB]
    const int* __restrict__ context_lens,     // [B] total, including this chunk
    const int num_seqs,
    const int num_queries,
    const int num_heads,
    const int num_kv_heads,
    const int block_size,
    const int max_blocks,
    const float scale) {

  constexpr int VPT = HEAD_DIM / WARP_SIZE;

  const int lane = threadIdx.x;
  const int flat = blockIdx.x * blockDim.y + threadIdx.y;
  if (flat >= num_seqs * num_queries * num_heads) return;

  const int h = flat % num_heads;
  const int s = (flat / num_heads) % num_queries;
  const int b = flat / (num_heads * num_queries);

  const int kv_h = h / (num_heads / num_kv_heads);   // GQA: shared KV head
  const int ctx_len = context_lens[b];

  // A chunk is the trailing tokens of the context, so query s is at
  // ctx_len - num_queries + s and may attend up to and including itself.
  const int q_pos = ctx_len - num_queries + s;
  const int end = q_pos + 1;

  float* o_ptr = out + ((size_t)(b * num_queries + s) * num_heads + h) * HEAD_DIM
                 + lane * VPT;

  if (end <= 0) {                     // padded query slot: nothing to attend to
    #pragma unroll
    for (int i = 0; i < VPT; ++i) o_ptr[i] = 0.0f;
    return;
  }

  const float* q_ptr = q + ((size_t)(b * num_queries + s) * num_heads + h) * HEAD_DIM;
  float q_reg[VPT];
  #pragma unroll
  for (int i = 0; i < VPT; ++i) q_reg[i] = q_ptr[lane * VPT + i];

  const int* btab = block_tables + (size_t)b * max_blocks;

  float m = NEG_BIG;
  float l = 0.0f;
  float acc[VPT];
  #pragma unroll
  for (int i = 0; i < VPT; ++i) acc[i] = 0.0f;

  for (int j = 0; j < end; ++j) {
    const int blk = btab[j / block_size];
    const int off = j % block_size;
    const size_t base = (((size_t)blk * block_size + off) * num_kv_heads + kv_h) * HEAD_DIM;

    const scalar_t* k_ptr = k_cache + base + lane * VPT;
    float partial = 0.0f;
    #pragma unroll
    for (int i = 0; i < VPT; ++i) partial += q_reg[i] * static_cast<float>(k_ptr[i]);

    float sc = warp_reduce_sum(partial);
    sc = __shfl_sync(FULL_MASK, sc, 0) * scale;

    const float m_new = fmaxf(m, sc);
    const float corr = __expf(m - m_new);
    const float p = __expf(sc - m_new);
    l = l * corr + p;

    const scalar_t* v_ptr = v_cache + base + lane * VPT;
    #pragma unroll
    for (int i = 0; i < VPT; ++i)
      acc[i] = acc[i] * corr + p * static_cast<float>(v_ptr[i]);

    m = m_new;
  }

  const float inv = 1.0f / l;
  #pragma unroll
  for (int i = 0; i < VPT; ++i) o_ptr[i] = acc[i] * inv;
}

torch::Tensor paged_attention_prefill(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor block_tables,
    torch::Tensor context_lens,
    double scale) {

  TORCH_CHECK(q.is_cuda(), "q must be on CUDA");
  TORCH_CHECK(q.dim() == 4, "q must be [B, S, H, D]");
  q = q.contiguous();
  k_cache = k_cache.contiguous();
  v_cache = v_cache.contiguous();
  block_tables = block_tables.to(torch::kInt32).contiguous();
  context_lens = context_lens.to(torch::kInt32).contiguous();

  const int B = q.size(0);
  const int S = q.size(1);
  const int H = q.size(2);
  const int D = q.size(3);
  const int BS = k_cache.size(1);
  const int HKV = k_cache.size(2);
  const int MB = block_tables.size(1);

  TORCH_CHECK(H % HKV == 0, "n_head must be divisible by n_kv_head");
  TORCH_CHECK(D % 32 == 0, "head_dim must be a multiple of the warp size");

  auto out = torch::empty({B, S, H, D}, q.options().dtype(torch::kFloat32));

  const int warps_per_block = 4;
  const dim3 threads(WARP_SIZE, warps_per_block);
  const int total = B * S * H;
  const dim3 grid((total + warps_per_block - 1) / warps_per_block);

  #define LAUNCH(DIM)                                                          \
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(                                       \
        k_cache.scalar_type(), "paged_attention_prefill", ([&] {               \
          paged_attention_prefill_kernel<scalar_t, DIM><<<grid, threads>>>(    \
              out.data_ptr<float>(), k_cache.data_ptr<scalar_t>(),             \
              v_cache.data_ptr<scalar_t>(), q.data_ptr<float>(),               \
              block_tables.data_ptr<int>(), context_lens.data_ptr<int>(),      \
              B, S, H, HKV, BS, MB, (float)scale);                             \
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
            name="heteroserve_paged_attn_prefill",
            cpp_sources=_CPP_DECL,
            cuda_sources=_CUDA_SRC,
            functions=["paged_attention_prefill"],
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


def paged_attention_prefill(q, k_cache, v_cache, block_tables, context_lens, scale):
    """Fused kernel when it compiled, the torch reference otherwise.

    Falling back rather than raising keeps the block-table path exercisable on a
    machine with no GPU -- which is how the whole paged prefill route gets tested
    in CI. `is_available()` is what a benchmark must check before claiming a
    kernel number.
    """
    _try_build()
    if _state == "ready":
        return _ext.paged_attention_prefill(
            q, k_cache, v_cache, block_tables, context_lens, float(scale)
        )
    return paged_attention_prefill_torch(
        q, k_cache, v_cache, block_tables, context_lens, scale
    )


def paged_attention_prefill_torch(q, k_cache, v_cache, block_tables, context_lens, scale):
    """Reference: causal paged attention over S query tokens, in pure torch.

    Materialises the gathered cache and the whole mask -- exactly what the kernel
    avoids. Kept obvious rather than fast, because its only job is to be right.
    """
    import torch

    B, S, Hq, D = q.shape
    NB, BS, HKV, _ = k_cache.shape
    MB = block_tables.shape[1]

    flat_k = k_cache.reshape(NB * BS, HKV, D)
    flat_v = v_cache.reshape(NB * BS, HKV, D)

    tables = block_tables.clamp(min=0).to(torch.long)
    slots = tables.unsqueeze(-1) * BS + torch.arange(
        BS, device=q.device, dtype=torch.long
    )
    slots = slots.reshape(B, MB * BS)

    k = flat_k[slots]                    # [B, K, HKV, D]
    v = flat_v[slots]
    if Hq != HKV:                        # GQA: share each KV head across its group
        g = Hq // HKV
        k = k.repeat_interleave(g, dim=2)
        v = v.repeat_interleave(g, dim=2)

    scores = torch.einsum("bshd,bkhd->bshk", q, k.float()) * scale

    K = MB * BS
    ctx = context_lens.to(torch.long).view(B, 1, 1)
    key_pos = torch.arange(K, device=q.device).view(1, 1, K)
    # query s of sequence b sits at ctx_len - S + s
    q_pos = (context_lens.to(torch.long).view(B, 1)
             - S + torch.arange(S, device=q.device).view(1, S)).unsqueeze(-1)
    valid = (key_pos <= q_pos) & (key_pos < ctx)

    scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0)     # fully-masked padded query rows
    return torch.einsum("bshk,bkhd->bshd", probs, v.float())
