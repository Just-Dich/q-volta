// Microbenchmark: quantized GEMV effective HBM bandwidth on Qwen3.8-27B shapes (random weights).
#include <cstdio>
#include <vector>
#include <string>
#include <cstdlib>
#include "../src/backends/cuda/kernels/gemv.h"
using namespace qv::cuda;
int main(int argc, char** argv) {
  bool q8 = argc > 1 && std::string(argv[1]) == "q8";
  int nt = argc > 2 ? std::atoi(argv[2]) : 1;
  Q8Buf xb; cudaMalloc(&xb.q, 17408 * 8); cudaMalloc(&xb.d, 8*17408/32*4); cudaMalloc(&xb.sum, 8*17408/32*4); cudaMalloc(&xb.qsum, 8*17408/32*4); cudaMemset(xb.qsum, 0, 8*17408/32*4);
  cudaMemset(xb.q, 1, 8*17408); cudaMemset(xb.d, 0, 8*17408/32*4); cudaMemset(xb.sum, 0, 8*17408/32*4);
  struct S { const char* name; int N, K, bits; } shapes[] = {
    {"mlp_gate/up 3b", 17408, 5120, 3}, {"mlp_down 3b", 5120, 17408, 3}, {"gdn_qkv 4b", 10240, 5120, 4},
    {"gdn_z 4b", 6144, 5120, 4}, {"gdn_out 4b", 5120, 6144, 4}, {"attn_q 4b", 12288, 5120, 4},
    {"attn_kv 4b", 1024, 5120, 4}, {"lm_head 6b", 248320, 5120, 6}, {"mlp_gate 4b", 17408, 5120, 4}};
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  float *x, *xs, *y; cudaMalloc(&x, 17408*4); cudaMalloc(&xs, 17408/32*4); cudaMalloc(&y, 8*248320*4);
  cudaMemset(x, 0, 17408*4); cudaMemset(xs, 0, 17408/32*4);
  // 1 GB flush buffer so every timed call reads from HBM, not L2
  for (auto& s : shapes) {
    size_t qb = (size_t)s.N * s.K / 8 * s.bits, smb = (size_t)s.N * (s.K / 64) * 4;
    uint32_t* q; __half2* sm; cudaMalloc(&q, qb); cudaMalloc(&sm, smb); cudaMemset(q, 0x5a, qb); cudaMemset(sm, 0, smb);
    QMat W{q, sm, s.N, s.K, s.bits};
    auto run = [&]{ if (q8) gemv_q8_multi(W, xb, nt, y, nullptr, 0); else gemv_q(W, x, xs, y, nullptr, 0); };
    for (int i = 0; i < 5; ++i) run();
    int it = 50; cudaEventRecord(a);
    for (int i = 0; i < it; ++i) run();
    cudaEventRecord(b); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b); ms /= it;
    std::printf("%-16s N=%6d K=%5d  %7.1f us  %6.1f GB/s\n", s.name, s.N, s.K, ms*1e3, (qb+smb)/(ms*1e-3)/1e9);
    cudaFree(q); cudaFree(sm);
  }
}
