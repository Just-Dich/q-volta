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
    const __half2* sm = nullptr;  // [N][K/64] (scale, min)
    int N = 0, K = 0, bits = 0;
};

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
