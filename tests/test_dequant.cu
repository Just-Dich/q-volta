// Dequant kernel test: bit-exact vs CPU reference (random + real tensors), consistency with gemv_q, bandwidth.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <vector>
#include "../src/backends/cuda/kernels/dequant.h"
#include "../src/backends/cuda/kernels/gemv.h"
#include "qv/model.h"
using namespace qv::cuda;

static const std::vector<std::pair<int,int>> PL[9] = {{}, {}, {{2,0}}, {{2,0},{1,2}}, {{4,0}}, {{4,0},{1,4}}, {{4,0},{2,4}}, {}, {{4,0},{4,4}}};

// host reference: unpack chunk of `bits` words per tools/qv_format.py
static void unpack_chunk(const uint32_t* w, int bits, int* q) {
  for (int i = 0; i < 32; ++i) q[i] = 0;
  int wi = 0;
  for (auto [pw, sh] : PL[bits]) {
    for (int j = 0; j < pw; ++j, ++wi)
      for (int f = 0; f < 32 / pw; ++f) q[j + pw * f] |= (int)((w[wi] >> (pw * f)) & ((1u << pw) - 1)) << sh;
  }
}
static void pack_chunk(const int* q, int bits, uint32_t* w) {
  int wi = 0;
  for (auto [pw, sh] : PL[bits]) {
    for (int j = 0; j < pw; ++j, ++wi) {
      uint32_t v = 0;
      for (int f = 0; f < 32 / pw; ++f) v |= (uint32_t)((q[j + pw * f] >> sh) & ((1 << pw) - 1)) << (pw * f);
      w[wi] = v;
    }
  }
}
static void ref_dequant(const uint32_t* qw, const __half2* sm, int N, int K, int bits, std::vector<__half>& out) {
  out.resize((size_t)N * K);
  int C = K / 32, G = K / 64, q[32];
  for (int n = 0; n < N; ++n)
    for (int c = 0; c < C; ++c) {
      unpack_chunk(qw + ((size_t)n * C + c) * bits, bits, q);
      float2 s = __half22float2(sm[(size_t)n * G + c / 2]);
      for (int i = 0; i < 32; ++i) out[(size_t)n * K + c * 32 + i] = __float2half_rn(s.x * (float)q[i] + s.y);
    }
}
template <class T> T* up(const std::vector<T>& v) { T* d; cudaMalloc(&d, v.size() * sizeof(T)); cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice); return d; }

static size_t cmp_bits(const std::vector<__half>& a, const std::vector<__half>& b) {
  size_t bad = 0;
  for (size_t i = 0; i < a.size(); ++i) { uint16_t x, y; memcpy(&x, &a[i], 2); memcpy(&y, &b[i], 2); bad += x != y; }
  return bad;
}

static int run_case(const char* tag, const std::vector<uint32_t>& qw, const std::vector<__half2>& sm, int N, int K, int bits,
                    bool gemv_check, std::mt19937& rng) {
  std::vector<__half> ref; ref_dequant(qw.data(), sm.data(), N, K, bits, ref);
  QMat W{up(qw), up(sm), N, K, bits};
  __half* d; cudaMalloc(&d, (size_t)N * K * 2);
  dequant_rows_f16(W, 0, N, d, 0);
  QV_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<__half> got((size_t)N * K); cudaMemcpy(got.data(), d, got.size() * 2, cudaMemcpyDeviceToHost);
  size_t bad = cmp_bits(got, ref);
  // sub-range (row0 != 0)
  int r0 = N / 3, nr = std::min(N - r0, 37);
  cudaMemset(d, 0, (size_t)nr * K * 2);
  dequant_rows_f16(W, r0, nr, d, 0);
  std::vector<__half> g2((size_t)nr * K); cudaMemcpy(g2.data(), d, g2.size() * 2, cudaMemcpyDeviceToHost);
  std::vector<__half> r2(ref.begin() + (size_t)r0 * K, ref.begin() + (size_t)(r0 + nr) * K);
  size_t bad2 = cmp_bits(g2, r2);
  double rel = -1;
  if (gemv_check) {
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x(K); for (auto& v : x) v = nd(rng);
    float *dx = up(x), *ds, *dy; cudaMalloc(&ds, K / 32 * 4); cudaMalloc(&dy, N * 4);
    xsum32(dx, ds, K, 0); gemv_q(W, dx, ds, dy, nullptr, 0);
    std::vector<float> y1(N); cudaMemcpy(y1.data(), dy, N * 4, cudaMemcpyDeviceToHost);
    double num = 0, den = 0;
    for (int n = 0; n < N; ++n) {
      double a = 0; for (int k = 0; k < K; ++k) a += (double)__half2float(got[(size_t)n * K + k]) * x[k];
      num += (y1[n] - a) * (y1[n] - a); den += a * a;
    }
    rel = std::sqrt(num / std::max(den, 1e-30));
    cudaFree(dx); cudaFree(ds); cudaFree(dy);
  }
  bool ok = bad == 0 && bad2 == 0 && (rel < 0 || rel <= 1e-3);
  std::printf("%-28s bits=%d N=%5d K=%5d mismatches=%zu sub=%zu gemv_rel=%.2e %s\n", tag, bits, N, K, bad, bad2, rel, ok ? "OK" : "FAIL");
  cudaFree(d); cudaFree((void*)W.q); cudaFree((void*)W.sm);
  return !ok;
}

int main(int argc, char** argv) {
  std::mt19937 rng(1234); int fails = 0;
  // (1) random packed matrices
  for (int bits : {3, 4, 6}) for (int K : {5120, 17408}) for (int N : {256, 1000, 17408}) {
    if ((size_t)N * K > 100000000) continue;  // keep host reference fast
    int C = K / 32, G = K / 64;
    std::vector<uint32_t> qw((size_t)N * C * bits); std::vector<__half2> sm((size_t)N * G);
    int q[32];
    for (size_t i = 0; i < (size_t)N * C; ++i) {
      for (auto& v : q) v = rng() & ((1 << bits) - 1);
      pack_chunk(q, bits, &qw[i * bits]);
    }
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    for (auto& s : sm) s = __floats2half2_rn(u(rng) * 0.1f, u(rng));
    fails += run_case("random", qw, sm, N, K, bits, N <= 1000 || K == 5120, rng);
  }
  // (2) real tensors
  std::string prefix = argc > 1 ? argv[1] : "/workspace/q-volta-models/qv/qwen38-27b-e0";
  try {
    qv::ModelFile mf = qv::ModelFile::open(prefix);
    std::ifstream bin(mf.bin_path, std::ios::binary);
    int got3 = 0, got4 = 0;
    for (auto& [name, ti] : mf.tensors) {
      if (ti.kind != qv::Kind::Quant || ti.shape.size() != 2) continue;
      bool want = (ti.bits == 3 && !got3 && name.find("layers.0.mlp.gate_proj") != std::string::npos) ||
                  (ti.bits == 4 && !got4 && name.find("layers.0.") != std::string::npos);
      if (!want) continue;
      (ti.bits == 3 ? got3 : got4) = 1;
      int N = (int)ti.shape[0], K = (int)ti.shape[1];
      std::vector<uint32_t> qw(ti.nbytes_q / 4); std::vector<__half2> sm(ti.nbytes_sm / 4);
      bin.seekg(ti.off_q); bin.read((char*)qw.data(), ti.nbytes_q);
      bin.seekg(ti.off_sm); bin.read((char*)sm.data(), ti.nbytes_sm);
      fails += run_case(name.c_str(), qw, sm, N, K, ti.bits, N * (size_t)K < 100000000, rng);
    }
    if (!got3 || !got4) { std::printf("FAIL: real tensors not found (3bit=%d 4bit=%d)\n", got3, got4); ++fails; }
  } catch (std::exception& e) { std::printf("FAIL: model load: %s\n", e.what()); ++fails; }
  // (3) bandwidth: N=17408 K=5120 bits=3
  {
    int N = 17408, K = 5120, bits = 3, C = K / 32, G = K / 64;
    std::vector<uint32_t> qw((size_t)N * C * bits); for (auto& v : qw) v = rng();
    std::vector<__half2> sm((size_t)N * G); for (auto& s : sm) s = __floats2half2_rn(0.05f, -0.2f);
    QMat W{up(qw), up(sm), N, K, bits};
    __half* d; cudaMalloc(&d, (size_t)N * K * 2);
    for (int i = 0; i < 5; ++i) dequant_rows_f16(W, 0, N, d, 0);
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int i = 0; i < 20; ++i) dequant_rows_f16(W, 0, N, d, 0);
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b); ms /= 20;
    double bytes = (double)N * C * bits * 4 + (double)N * G * 4 + (double)N * K * 2;
    std::printf("bandwidth N=%d K=%d bits=3: %.3f ms/iter, %.1f MB moved, %.1f GB/s (read+write)\n", N, K, ms, bytes / 1e6, bytes / (ms * 1e-3) / 1e9);
  }
  std::printf(fails ? "TEST FAILED (%d)\n" : "ALL PASSED\n", fails);
  return fails ? 1 : 0;
}
