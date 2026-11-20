#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define QV_CUDA_CHECK(x)                                                                              \
    do {                                                                                              \
        cudaError_t e_ = (x);                                                                         \
        if (e_ != cudaSuccess) {                                                                      \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); \
            std::abort();                                                                             \
        }                                                                                             \
    } while (0)

namespace qv::cuda {

// Quantized matrix on device (see tools/qv_format.py for the layout).
struct QMat {
    const uint32_t* q = nullptr;  // [N][K/32][bits]
    const __half2* sm = nullptr;  // [N][K/group] (scale, min)
    int N = 0, K = 0, bits = 0;
    int gshift = 6;               // log2(group size): 6 (64) or 7 (128); sm is [N][K >> gshift]
    // meta8 encoding (sm8 != nullptr, sm unused): sm8 = uint16 [N][K >> gshift] (s8 | z8 << 8), rs = fp16 [N]
    //   scale = s8 * (rs / 255), z = z8 * zstep + zlo, min = -(scale * z)   (tools/qv_quant.py meta8_decode)
    const uint16_t* sm8 = nullptr;
    const __half* rs = nullptr;
    float zlo = 0.f, zstep = 0.f;
};

// (scale, min) of group g of row `row` (G groups per row)
__device__ __forceinline__ float2 load_sm(const QMat& W, size_t row, size_t G, size_t g) {
    if (W.sm8 == nullptr) return __half22float2(W.sm[row * G + g]);
    const unsigned v = W.sm8[row * G + g];
    const float sc = (float)(v & 255u) * (__half2float(W.rs[row]) / 255.0f);
    const float z = (float)(v >> 8) * W.zstep + W.zlo;
    return make_float2(sc, -(sc * z));
}

struct DMat16 {  // dense fp16 row-major [N][K]
    const __half* w = nullptr;
    int N = 0, K = 0;
};

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// Block-wide sum; `red` must hold >= 32 floats. All threads get the result.
__device__ __forceinline__ float block_sum(float v, float* red) {
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) red[w] = v;
    __syncthreads();
    int nw = (blockDim.x + 31) >> 5;
    float t = (threadIdx.x < nw) ? red[threadIdx.x] : 0.f;
    if (w == 0) {
        t = warp_sum(t);
        if (lane == 0) red[0] = t;
    }
    __syncthreads();
    return red[0];
}

__device__ __forceinline__ float silu(float x) { return x / (1.f + __expf(-x)); }

}  // namespace qv::cuda
