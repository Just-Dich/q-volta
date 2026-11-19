// Device helper: a full warp holds one 32-element chunk c (lane i = element c*32+i). Quantize it into the gemv_q8
// activation format (canonical permutation, see gemv_q8.cu) — lets producer kernels emit the GEMV input directly.
#pragma once
#include "../common.cuh"
#include "gemv.h"

namespace qv::cuda {
__device__ __forceinline__ void q8_store_chunk(float v, int c, const Q8Buf& o) {
    const int i = threadIdx.x & 31;
    const float amax = warp_max(fabsf(v));
    const float dx = amax / 127.f;
    const int qi = dx > 0.f ? __float2int_rn(v / dx) : 0;
    reinterpret_cast<int8_t*>(o.q)[c * 32 + (i & 7) * 4 + (i >> 3)] = (int8_t)qi;
    const float s = warp_sum(v);
    int qs = qi;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) qs += __shfl_xor_sync(0xffffffffu, qs, off);
    if (i == 0) { o.d[c] = dx; o.sum[c] = s; o.qsum[c] = qs; }
}
}  // namespace qv::cuda
