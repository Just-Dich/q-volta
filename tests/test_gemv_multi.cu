// Multi-token GEMV must be BIT-IDENTICAL to nt independent single-token GEMVs (same arithmetic per token): MTP verify
// relies on it for exact greedy output. N covers the per-shape default configs of the 1-token kernel (warps 4 and 2).
#include <cstdio>
#include <cstring>
#include <cmath>
#include <random>
#include <vector>
#include "../src/backends/cuda/kernels/gemv.h"
using namespace qv::cuda;
int main() {
  std::mt19937 rng(7); int fails = 0;
  for (int grp : {64, 128, 129}) for (int bits : {3, 4, 6}) for (int K : {5120, 17408}) for (int nt : {2, 3, 4, 8}) for (int N : {1000, 12000}) {
    std::vector<uint32_t> q((size_t)N * K / 32 * bits); for (auto& v : q) v = rng();
    std::vector<__half2> sm((size_t)N * K / (grp == 64 ? 64 : 128)); for (auto& v : sm) v = __floats2half2_rn(0.001f * (rng() % 100 + 1), -0.01f);
    std::vector<float> x((size_t)nt * K); std::normal_distribution<float> nd; for (auto& v : x) v = nd(rng);
    uint32_t* dq; __half2* dsm; float *dx, *y1, *y2; cudaMalloc(&dq, q.size()*4); cudaMalloc(&dsm, sm.size()*4);
    cudaMalloc(&dx, x.size()*4); cudaMalloc(&y1, (size_t)nt*N*4); cudaMalloc(&y2, (size_t)nt*N*4);
    cudaMemcpy(dq, q.data(), q.size()*4, cudaMemcpyHostToDevice); cudaMemcpy(dsm, sm.data(), sm.size()*4, cudaMemcpyHostToDevice);
    cudaMemcpy(dx, x.data(), x.size()*4, cudaMemcpyHostToDevice);
    Q8Buf b; cudaMalloc(&b.q, (size_t)nt*K); cudaMalloc(&b.d, nt*K/32*4); cudaMalloc(&b.sum, nt*K/32*4); cudaMalloc(&b.qsum, nt*K/32*4);
    quantize_q8(dx, nt * K, b, 0);
    QMat W{dq, dsm, N, K, bits, grp == 64 ? 6 : 7};
    if (grp == 129) {  // meta8: reinterpret the random (scale,min) words as s8|z8<<8 codes; any code is a valid group
      W.sm = nullptr; W.sm8 = (const uint16_t*)dsm; W.rs = (const __half*)dsm; W.zlo = -0.25f * ((1 << bits) - 1);
      W.zstep = (float)(1.5 * ((1 << bits) - 1) / 255.0);
    }
    gemv_q8_multi(W, b, nt, y2, nullptr, 0);
    for (int t = 0; t < nt; ++t) {
      Q8Vec v{b.q + (size_t)t*K/4, b.d + t*K/32, b.sum + t*K/32, b.qsum + t*K/32};
      gemv_q8(W, v, y1 + (size_t)t*N, nullptr, 0);
    }
    std::vector<float> a((size_t)nt*N), c((size_t)nt*N);
    cudaMemcpy(a.data(), y1, a.size()*4, cudaMemcpyDeviceToHost); cudaMemcpy(c.data(), y2, c.size()*4, cudaMemcpyDeviceToHost);
    double num = 0, den = 0; for (size_t i = 0; i < a.size(); ++i) { num += (a[i]-c[i])*(a[i]-c[i]); den += a[i]*a[i]; }
    double rel = std::sqrt(num/den); bool ok = std::memcmp(a.data(), c.data(), a.size()*4) == 0; fails += !ok;
    std::printf("g=%d bits=%d K=%5d N=%5d nt=%d rel=%.1e %s\n", grp == 129 ? 1280 : grp, bits, K, N, nt, rel, ok ? "OK" : "FAIL");
  }
  return fails != 0;
}
