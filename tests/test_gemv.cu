// GEMV kernel test: reads cases written by tests/gen_gemv_cases.py and compares against fp64 reference.
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
#include "../src/backends/cuda/kernels/gemv.h"
using namespace qv::cuda;
template <class T> std::vector<T> rd(const std::string& p) {
  std::ifstream f(p, std::ios::binary | std::ios::ate); size_t n = f.tellg(); f.seekg(0);
  std::vector<T> v(n / sizeof(T)); f.read((char*)v.data(), n); return v; }
template <class T> T* up(const std::vector<T>& v) { T* d; cudaMalloc(&d, v.size()*sizeof(T)); cudaMemcpy(d, v.data(), v.size()*sizeof(T), cudaMemcpyHostToDevice); return d; }
int main(int argc, char** argv) {
  std::string dir = argv[1]; bool q8 = argc > 2 && std::string(argv[2]) == "q8"; int fails = 0;
  for (int g : {64, 128, 129}) for (int bits : {2,3,4,5,6,8}) for (int K : {5120, 6144, 17408}) {
    std::string p = dir + "/b" + std::to_string(bits) + "_k" + std::to_string(K) + (g == 64 ? "" : g == 128 ? "_g128" : "_m8");
    auto q = rd<uint32_t>(p + ".q"); auto x = rd<float>(p + ".x");
    std::vector<__half2> sm; if (g != 129) sm = rd<__half2>(p + ".sm");  // meta8 metadata is loaded below
    auto yref = rd<double>(p + ".y"); auto res = rd<float>(p + ".r");
    int N = yref.size();
    QMat W{up(q), up(sm), N, K, bits, g == 64 ? 6 : 7};
    if (g == 129) {  // meta8: sm file holds uint16 (s8 | z8<<8); row scales in .rs
      auto sm16 = rd<uint16_t>(p + ".sm"); auto rs = rd<__half>(p + ".rs");
      W.sm = nullptr; W.sm8 = up(sm16); W.rs = up(rs);
      W.zlo = -0.25f * ((1 << bits) - 1); W.zstep = (float)(1.5 * ((1 << bits) - 1) / 255.0);
    }
    float *dx = up(x), *dr = up(res), *dy, *ds; cudaMalloc(&dy, N*4); cudaMalloc(&ds, K/32*4);
    if (q8) { Q8Buf b; cudaMalloc(&b.q, K); cudaMalloc(&b.d, K/32*4); cudaMalloc(&b.sum, K/32*4); cudaMalloc(&b.qsum, K/32*4);
      quantize_q8(dx, K, b, 0); gemv_q8(W, b, dy, dr, 0); }
    else { xsum32(dx, ds, K, 0); gemv_q(W, dx, ds, dy, dr, 0); }
    std::vector<float> y(N); cudaMemcpy(y.data(), dy, N*4, cudaMemcpyDeviceToHost);
    double num = 0, den = 0, mx = 0;
    for (int i = 0; i < N; ++i) { double e = y[i] - yref[i]; num += e*e; den += yref[i]*yref[i]; mx = std::max(mx, std::abs(e)); }
    double rel = std::sqrt(num / den); bool ok = rel < (q8 ? 2e-2 : 1e-5);
    fails += !ok; std::printf("g=%d%s bits=%d K=%5d N=%d rel=%.2e maxabs=%.2e %s\n", g == 129 ? 128 : g, g == 129 ? "+meta8" : "", bits, K, N, rel, mx, ok ? "OK" : "FAIL");
  }
  return fails ? 1 : 0;
}
