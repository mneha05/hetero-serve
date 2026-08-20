"""Tensor-core prefill: FlashAttention tiling on WMMA fragments, off a block table.

Decode is a GEMV and tensor cores cannot help it -- 201 MB moved, 11% compute
throughput, the arithmetic was never the constraint. Prefill is a genuine GEMM:
S query rows against N keys, twice (Q·Kᵀ then P·V). That is where WMMA belongs,
and this is it.

The alignment that makes it work
--------------------------------
A WMMA fragment on sm_70+ is 16x16x16. A KV page in this system is **16 tokens**.
So one key tile is exactly one page: the block table indexes whole fragments
instead of straddling them, and the paging indirection costs one pointer lookup
per tile rather than per element. That is not a coincidence I engineered around
after the fact -- it is why `block_size = 16` was the right default in the first
place, and it is the reason a tensor-core kernel can read a paged cache at all
without a gather.

The loop is FlashAttention's, with the online softmax already proven in v2/v3:

    for each key tile (= one page):
        S = Q·Kᵀ            4 MMAs over head_dim, accumulate in fp32
        mask + rescale      causal bound per row, running max/sum
        O = O·corr + P·V    4 MMAs, accumulated across tiles

Two honest costs
----------------
**Rescaling round-trips through shared memory.** A WMMA accumulator's register
mapping is opaque, so applying a per-row correction means store, scale, reload.
Production kernels (CUTLASS) exploit the known layout and keep it in registers.
This does not, and pays for it.

**fp16 inputs.** MMA on sm_75 takes half operands with fp32 accumulate. The
online softmax and the O accumulator stay fp32, so the numerics are the same as
the other kernels, but Q and P are rounded to half before each GEMM -- which is
what every production attention kernel does, and why the tolerance here is 2e-2
rather than 1e-6.

Requires `block_size == 16` and `head_dim % 16 == 0`. Anything else falls back to
the scalar prefill kernel, which is checked rather than assumed --
`supports()` is what the caller must ask.
"""

from __future__ import annotations

import os
from typing import Optional

_CPP_DECL = """
#include <torch/extension.h>
torch::Tensor paged_attention_prefill_wmma(
    torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor block_tables, torch::Tensor context_lens, double scale);
"""

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAException.h>

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
#define HS_HAS_WMMA 1
#include <mma.h>
using namespace nvcuda;
#else
#define HS_HAS_WMMA 0
#endif

#define WARP_SIZE 32
#define TILE 16                 // WMMA M/N/K, and exactly one KV page
#define WARPS_PER_BLOCK 2
#define NEG_BIG (-3.0e38f)

// One warp per (sequence, 16-query tile, head).
//
// Shared memory per warp, head_dim 64:  Q 2K + K 2K + V 2K + S 1K + P 0.5K
// + O staging 1K = 8.5 KB. Two warps per block keeps a 128-dim head inside
// sm_75's 48 KB without needing the opt-in carveout.
template <int HEAD_DIM>
__global__ void prefill_wmma_kernel(
    float* __restrict__ out,                  // [B, S, H, D]
    const __half* __restrict__ k_cache,       // [NB, 16, HKV, D]
    const __half* __restrict__ v_cache,
    const float* __restrict__ q,              // [B, S, H, D]
    const int* __restrict__ block_tables,     // [B, MB]
    const int* __restrict__ context_lens,     // [B]
    const int num_seqs,
    const int num_queries,
    const int num_heads,
    const int num_kv_heads,
    const int max_blocks,
    const float scale) {
#if HS_HAS_WMMA
  constexpr int NCHUNK = HEAD_DIM / TILE;     // head_dim slices of 16

  extern __shared__ char smem_raw[];
  const int warp = threadIdx.y;
  const int lane = threadIdx.x;

  // per-warp slab
  const size_t per_warp =
      sizeof(__half) * (3 * TILE * HEAD_DIM + TILE * TILE)   // Q, K, V, P
      + sizeof(float) * (TILE * TILE + TILE * TILE + 3 * TILE);  // S, O, m/l/corr
  char* base = smem_raw + warp * per_warp;

  __half* Qs = reinterpret_cast<__half*>(base);
  __half* Ks = Qs + TILE * HEAD_DIM;
  __half* Vs = Ks + TILE * HEAD_DIM;
  __half* Ps = Vs + TILE * HEAD_DIM;
  float*  Ss = reinterpret_cast<float*>(Ps + TILE * TILE);
  float*  Os = Ss + TILE * TILE;
  float*  ms = Os + TILE * TILE;
  float*  ls = ms + TILE;
  float*  cs = ls + TILE;

  const int q_tiles = (num_queries + TILE - 1) / TILE;
  const int flat = blockIdx.x * WARPS_PER_BLOCK + warp;
  if (flat >= num_seqs * q_tiles * num_heads) return;

  const int h  = flat % num_heads;
  const int qt = (flat / num_heads) % q_tiles;
  const int b  = flat / (num_heads * q_tiles);

  const int kv_h = h / (num_heads / num_kv_heads);
  const int ctx_len = context_lens[b];
  const int q0 = qt * TILE;
  // absolute position of query row (q0 + r): a chunk is the trailing tokens
  const int q_abs0 = ctx_len - num_queries + q0;

  // ---- load the Q tile (fp32 -> half), zeroing rows past the chunk ----
  for (int i = lane; i < TILE * HEAD_DIM; i += WARP_SIZE) {
    const int r = i / HEAD_DIM, d = i % HEAD_DIM;
    const int qr = q0 + r;
    Qs[i] = (qr < num_queries)
        ? __float2half(q[((size_t)(b * num_queries + qr) * num_heads + h) * HEAD_DIM + d])
        : __float2half(0.0f);
  }
  if (lane < TILE) { ms[lane] = NEG_BIG; ls[lane] = 0.0f; }
  __syncwarp();

  wmma::fragment<wmma::accumulator, TILE, TILE, TILE, float> o_frag[NCHUNK];
  #pragma unroll
  for (int c = 0; c < NCHUNK; ++c) wmma::fill_fragment(o_frag[c], 0.0f);

  // Only tiles up to the last row's causal bound can contribute.
  const int last_q_abs = q_abs0 + TILE - 1;
  const int key_limit = min(ctx_len, last_q_abs + 1);
  const int n_tiles = (key_limit + TILE - 1) / TILE;

  const int* btab = block_tables + (size_t)b * max_blocks;

  for (int kt = 0; kt < n_tiles; ++kt) {
    const int blk = btab[kt];
    // One key tile is exactly one page -- the whole point of block_size == 16.
    const size_t page = (size_t)blk * TILE * num_kv_heads * HEAD_DIM
                        + (size_t)kv_h * HEAD_DIM;
    for (int i = lane; i < TILE * HEAD_DIM; i += WARP_SIZE) {
      const int t = i / HEAD_DIM, d = i % HEAD_DIM;
      const size_t off = page + (size_t)t * num_kv_heads * HEAD_DIM + d;
      Ks[i] = k_cache[off];
      Vs[i] = v_cache[off];
    }
    __syncwarp();

    // ---- S = Q * K^T  (col_major on K gives the transpose for free) ----
    wmma::fragment<wmma::accumulator, TILE, TILE, TILE, float> s_frag;
    wmma::fill_fragment(s_frag, 0.0f);
    #pragma unroll
    for (int c = 0; c < NCHUNK; ++c) {
      wmma::fragment<wmma::matrix_a, TILE, TILE, TILE, __half, wmma::row_major> a;
      wmma::fragment<wmma::matrix_b, TILE, TILE, TILE, __half, wmma::col_major> bfrag;
      wmma::load_matrix_sync(a, Qs + c * TILE, HEAD_DIM);
      wmma::load_matrix_sync(bfrag, Ks + c * TILE, HEAD_DIM);
      wmma::mma_sync(s_frag, a, bfrag, s_frag);
    }
    wmma::store_matrix_sync(Ss, s_frag, TILE, wmma::mem_row_major);
    __syncwarp();

    // ---- causal mask + online softmax, one row per lane ----
    if (lane < TILE) {
      const int r = lane;
      const int q_abs = q_abs0 + r;
      const bool row_live = (q0 + r) < num_queries && q_abs >= 0;

      float rmax = NEG_BIG;
      #pragma unroll
      for (int j = 0; j < TILE; ++j) {
        const int kpos = kt * TILE + j;
        const bool keep = row_live && kpos <= q_abs && kpos < ctx_len;
        const float v = keep ? Ss[r * TILE + j] * scale : NEG_BIG;
        Ss[r * TILE + j] = v;
        rmax = fmaxf(rmax, v);
      }
      const float m_new = fmaxf(ms[r], rmax);
      const float corr = __expf(ms[r] - m_new);
      float rsum = 0.0f;
      #pragma unroll
      for (int j = 0; j < TILE; ++j) {
        const float p = (Ss[r * TILE + j] <= NEG_BIG) ? 0.0f
                        : __expf(Ss[r * TILE + j] - m_new);
        Ps[r * TILE + j] = __float2half(p);
        rsum += p;
      }
      ls[r] = ls[r] * corr + rsum;
      ms[r] = m_new;
      cs[r] = corr;
    }
    __syncwarp();

    // ---- O = O*corr + P*V ----
    // The accumulator's register mapping is opaque, so the per-row rescale has
    // to round-trip through shared memory. This is the part CUTLASS avoids.
    #pragma unroll
    for (int c = 0; c < NCHUNK; ++c) {
      wmma::store_matrix_sync(Os, o_frag[c], TILE, wmma::mem_row_major);
      __syncwarp();
      if (lane < TILE) {
        const float k = cs[lane];
        #pragma unroll
        for (int j = 0; j < TILE; ++j) Os[lane * TILE + j] *= k;
      }
      __syncwarp();
      wmma::load_matrix_sync(o_frag[c], Os, TILE, wmma::mem_row_major);

      wmma::fragment<wmma::matrix_a, TILE, TILE, TILE, __half, wmma::row_major> pa;
      wmma::fragment<wmma::matrix_b, TILE, TILE, TILE, __half, wmma::row_major> vb;
      wmma::load_matrix_sync(pa, Ps, TILE);
      wmma::load_matrix_sync(vb, Vs + c * TILE, HEAD_DIM);
      wmma::mma_sync(o_frag[c], pa, vb, o_frag[c]);
      __syncwarp();
    }
  }

  // ---- normalise and write out ----
  #pragma unroll
  for (int c = 0; c < NCHUNK; ++c) {
    wmma::store_matrix_sync(Os, o_frag[c], TILE, wmma::mem_row_major);
    __syncwarp();
    if (lane < TILE) {
      const int r = lane;
      const int qr = q0 + r;
      if (qr < num_queries) {
        const float inv = (ls[r] > 0.0f) ? (1.0f / ls[r]) : 0.0f;
        float* o = out + ((size_t)(b * num_queries + qr) * num_heads + h) * HEAD_DIM
                   + c * TILE;
        #pragma unroll
        for (int j = 0; j < TILE; ++j) o[j] = Os[r * TILE + j] * inv;
      }
    }
    __syncwarp();
  }
#endif  // HS_HAS_WMMA
}

torch::Tensor paged_attention_prefill_wmma(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor block_tables,
    torch::Tensor context_lens,
    double scale) {

  TORCH_CHECK(q.is_cuda(), "q must be on CUDA");
  TORCH_CHECK(q.dim() == 4, "q must be [B, S, H, D]");
  TORCH_CHECK(k_cache.scalar_type() == torch::kHalf,
              "the WMMA path needs an fp16 KV cache");
  TORCH_CHECK(k_cache.size(1) == 16,
              "the WMMA path needs block_size == 16 (one page per fragment)");

  q = q.contiguous();
  k_cache = k_cache.contiguous();
  v_cache = v_cache.contiguous();
  block_tables = block_tables.to(torch::kInt32).contiguous();
  context_lens = context_lens.to(torch::kInt32).contiguous();

  const int B = q.size(0);
  const int S = q.size(1);
  const int H = q.size(2);
  const int D = q.size(3);
  const int HKV = k_cache.size(2);
  const int MB = block_tables.size(1);

  TORCH_CHECK(H % HKV == 0, "n_head must be divisible by n_kv_head");
  TORCH_CHECK(D % 16 == 0, "head_dim must be a multiple of 16");

  auto out = torch::empty({B, S, H, D}, q.options().dtype(torch::kFloat32));

  const int q_tiles = (S + 15) / 16;
  const int total = B * q_tiles * H;
  const dim3 threads(32, WARPS_PER_BLOCK);
  const dim3 grid((total + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

  #define LAUNCH(DIM)                                                          \
    {                                                                          \
      const size_t per_warp =                                                  \
          sizeof(__half) * (3 * 16 * (DIM) + 16 * 16)                          \
          + sizeof(float) * (16 * 16 + 16 * 16 + 3 * 16);                      \
      const size_t shmem = per_warp * WARPS_PER_BLOCK;                         \
      prefill_wmma_kernel<DIM><<<grid, threads, shmem>>>(                      \
          out.data_ptr<float>(),                                               \
          reinterpret_cast<const __half*>(k_cache.data_ptr<at::Half>()),       \
          reinterpret_cast<const __half*>(v_cache.data_ptr<at::Half>()),       \
          q.data_ptr<float>(), block_tables.data_ptr<int>(),                   \
          context_lens.data_ptr<int>(), B, S, H, HKV, MB, (float)scale);       \
    }

  switch (D) {
    case 32:  LAUNCH(32);  break;
    case 64:  LAUNCH(64);  break;
    case 128: LAUNCH(128); break;
    default: TORCH_CHECK(false, "unsupported head_dim for the WMMA path: ", D);
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
    major, _minor = torch.cuda.get_device_capability(0)
    if major < 7:
        _state, _error = "unavailable", (
            f"tensor cores need sm_70+, this device is sm_{major}{_minor}")
        return
    try:
        _ext = load_inline(
            name="heteroserve_prefill_wmma",
            cpp_sources=_CPP_DECL,
            cuda_sources=_CUDA_SRC,
            functions=["paged_attention_prefill_wmma"],
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


def supports(block_size: int, head_dim: int, kv_dtype) -> bool:
    """Whether this kernel can serve a given cache geometry.

    Asked rather than assumed: a 16-token page is what makes a key tile a whole
    WMMA fragment, and fp16 is what the MMA takes. Anything else belongs on the
    scalar prefill kernel.
    """
    import torch

    return (
        is_available()
        and block_size == 16
        and head_dim % 16 == 0
        and head_dim in (32, 64, 128)
        and kv_dtype == torch.float16
    )


def paged_attention_prefill_wmma(q, k_cache, v_cache, block_tables, context_lens, scale):
    _try_build()
    if _state != "ready":
        raise RuntimeError(f"WMMA prefill kernel unavailable: {_error}")
    return _ext.paged_attention_prefill_wmma(
        q, k_cache, v_cache, block_tables, context_lens, float(scale)
    )


NEG_BIG = -3.0e38


def tiled_prefill_reference(q, k, v, scale, ctx_len, tile: int = 16):
    """The kernel's loop, in numpy, for one head.

    Mirrors it line for line -- including using a large *finite* sentinel rather
    than -inf, which is what keeps a fully-masked tile from producing
    `exp(-inf - -inf) = nan` on the very first iteration. Written so the tiling,
    the per-row causal bound and the rescale can be checked against dense
    attention without a GPU; only the MMA itself is left untested.

        q [S, D]   k, v [N, D]  ->  out [S, D]
    """
    import numpy as np

    S, D = q.shape
    out = np.zeros((S, D), dtype=np.float64)
    n_qt = -(-S // tile)

    for qt in range(n_qt):
        q0 = qt * tile
        qtile = np.zeros((tile, D))
        live_rows = min(tile, S - q0)
        qtile[:live_rows] = q[q0:q0 + live_rows]

        q_abs0 = ctx_len - S + q0
        m = np.full(tile, NEG_BIG)
        l = np.zeros(tile)
        O = np.zeros((tile, D))

        key_limit = min(ctx_len, q_abs0 + tile)
        for kt in range(max(0, -(-key_limit // tile))):
            lo, hi = kt * tile, min((kt + 1) * tile, len(k))
            kk = np.zeros((tile, D))
            vv = np.zeros((tile, D))
            kk[:hi - lo] = k[lo:hi]
            vv[:hi - lo] = v[lo:hi]

            Sm = (qtile @ kk.T) * scale
            for r in range(tile):
                q_abs = q_abs0 + r
                row_live = (q0 + r) < S and q_abs >= 0
                for j in range(tile):
                    kpos = kt * tile + j
                    if not (row_live and kpos <= q_abs and kpos < ctx_len):
                        Sm[r, j] = NEG_BIG

            rmax = Sm.max(axis=1)
            m_new = np.maximum(m, rmax)
            corr = np.exp(m - m_new)
            P = np.where(Sm <= NEG_BIG, 0.0, np.exp(Sm - m_new[:, None]))
            l = l * corr + P.sum(axis=1)
            O = O * corr[:, None] + P @ vv
            m = m_new

        for r in range(live_rows):
            out[q0 + r] = O[r] / l[r] if l[r] > 0 else 0.0

    return out.astype(np.float32)
