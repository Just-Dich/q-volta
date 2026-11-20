// Standalone attention-only benchmark with synthetic data: bench_attn_seq <pos0> <T> [iters] [fp32=1] [check=0]
// One layer's KV cache (fp32, [Hkv][max_ctx][D]) with max_ctx = pos0+T; query chunk of T tokens at pos0.
// Prints JSON-ish lines. check=1 compares tensor-core vs fp32 kernel output (rel L2 error).
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../src/backends/cuda/attention/attn_seq.h"

using namespace qv::cuda;
constexpr int Hq = 24, Hkv = 4, D = 256;

__global__ void fill_kernel(float* p, size_t n, uint32_t seed, float scale) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint32_t h = (uint32_t)(i * 2654435761u) ^ (uint32_t)(i >> 32) * 40503u ^ seed;
        h ^= h >> 16; h *= 0x7feb352dU; h ^= h >> 15; h *= 0x846ca68bU; h ^= h >> 16;
        uint32_t h2 = h * 1664525u + 1013904223u; h2 ^= h2 >> 13; h2 *= 0x5bd1e995u; h2 ^= h2 >> 15;
        float u1 = (h >> 8) * (1.f / 16777216.f) + 1e-7f, u2 = (h2 >> 8) * (1.f / 16777216.f);
        p[i] = scale * sqrtf(-2.f * logf(u1)) * cospif(2.f * u2);
    }
}
static float* alloc_fill(size_t n, uint32_t seed, float scale) {
    float* p; QV_CUDA_CHECK(cudaMalloc(&p, n * 4));
    fill_kernel<<<1024, 256>>>(p, n, seed, scale);
    return p;
}

static float time_ms(void (*fn)(const AttnSeqArgs&, cudaStream_t), const AttnSeqArgs& a, int iters) {
    fn(a, 0);
    QV_CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    for (int i = 0; i < iters; ++i) fn(a, 0);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms, e0, e1);
    return ms / iters;
}

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s pos0 T [iters] [fp32=1] [check=0]\n", argv[0]); return 2; }
    const int pos0 = std::atoi(argv[1]), T = std::atoi(argv[2]);
    const int iters = argc > 3 ? std::atoi(argv[3]) : 5;
    const bool do32 = argc > 4 ? std::atoi(argv[4]) != 0 : true;
    const bool check = argc > 5 && std::atoi(argv[5]) != 0;
    const int max_ctx = pos0 + T;
    const size_t csz = (size_t)Hkv * max_ctx * D;
    float* kc = alloc_fill(csz, 1, 1.f);
    float* vc = alloc_fill(csz, 2, 1.f);
    float* q = alloc_fill((size_t)T * Hq * D, 3, 1.f);
    float* qg = alloc_fill((size_t)T * Hq * 2 * D, 4, 1.f);
    float *of = nullptr, *ot = nullptr;
    QV_CUDA_CHECK(cudaMalloc(&ot, (size_t)T * Hq * D * 4));
    __half* oh; QV_CUDA_CHECK(cudaMalloc(&oh, (size_t)T * Hq * D * 2));
    AttnSeqArgs a{q, qg, kc, vc, oh, ot, Hq, Hkv, max_ctx, pos0, T};
    QV_CUDA_CHECK(cudaDeviceSynchronize());
    double keys = (double)T * pos0 + (double)T * (T + 1) / 2;
    double fl = 4.0 * Hq * D * keys;
    float t_tc = time_ms(attn_seq_tc, a, iters);
    std::printf("{\"pos0\": %d, \"T\": %d, \"tc_ms\": %.4f, \"tc_tflops\": %.2f", pos0, T, t_tc, fl / (t_tc * 1e-3) / 1e12);
    if (do32 || check) {
        QV_CUDA_CHECK(cudaMalloc(&of, (size_t)T * Hq * D * 4));
        AttnSeqArgs b = a; b.out_f = of; b.out_h = nullptr;
        float t32 = time_ms(attn_seq_fp32, b, check && !do32 ? 1 : iters);
        std::printf(", \"fp32_ms\": %.4f, \"speedup\": %.2f", t32, t32 / t_tc);
        if (check) {
            std::vector<float> x((size_t)T * Hq * D), y(x.size());
            cudaMemcpy(x.data(), ot, x.size() * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(y.data(), of, y.size() * 4, cudaMemcpyDeviceToHost);
            double e = 0, nn = 0;
            for (size_t i = 0; i < x.size(); ++i) { double d = x[i] - y[i]; e += d * d; nn += (double)y[i] * y[i]; }
            std::printf(", \"rel_err_vs_fp32\": %.3e", std::sqrt(e / nn));
        }
    }
    std::printf("}\n");
    return 0;
}
