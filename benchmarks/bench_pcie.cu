// Pinned host<->device bandwidth and small-copy round-trip latency.
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { std::printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); std::exit(1);} } while (0)
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main() {
    CK(cudaSetDevice(0));
    cudaStream_t s; CK(cudaStreamCreate(&s));
    const size_t MAXB = 256u << 20;
    void *h, *d; CK(cudaHostAlloc(&h, MAXB, cudaHostAllocDefault)); CK(cudaMalloc(&d, MAXB));
    std::memset(h, 1, MAXB);
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    std::printf("bandwidth (pinned, best of reps)\n  size      H2D GB/s   D2H GB/s\n");
    for (size_t mb : {1, 8, 64, 256}) {
        size_t n = mb << 20; double bw[2];
        for (int dir = 0; dir < 2; ++dir) {
            double best = 1e30;
            for (int r = 0; r < 12; ++r) {
                CK(cudaEventRecord(e0, s));
                if (dir == 0) CK(cudaMemcpyAsync(d, h, n, cudaMemcpyHostToDevice, s)); else CK(cudaMemcpyAsync(h, d, n, cudaMemcpyDeviceToHost, s));
                CK(cudaEventRecord(e1, s)); CK(cudaEventSynchronize(e1));
                float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); best = std::min(best, (double)ms);
            }
            bw[dir] = n / (best * 1e-3) / 1e9;
        }
        std::printf("  %3zu MB   %7.2f    %7.2f\n", mb, bw[0], bw[1]);
    }
    std::printf("small-copy latency, wall clock around memcpyAsync + event sync (median of 2000)\n");
    for (size_t b : {6144, 25600}) {
        for (int mode = 0; mode < 3; ++mode) {  // 0: D2H only, 1: H2D only, 2: D2H then H2D (full per-layer exchange)
            std::vector<double> t;
            for (int r = 0; r < 2200; ++r) {
                double t0 = now();
                if (mode != 1) CK(cudaMemcpyAsync(h, d, b, cudaMemcpyDeviceToHost, s));
                if (mode == 2) { CK(cudaEventRecord(e0, s)); CK(cudaEventSynchronize(e0)); }  // host would compute here
                if (mode != 0) CK(cudaMemcpyAsync(d, h, b, cudaMemcpyHostToDevice, s));
                CK(cudaEventRecord(e1, s)); CK(cudaEventSynchronize(e1));
                if (r >= 200) t.push_back((now() - t0) * 1e6);
            }
            std::sort(t.begin(), t.end());
            std::printf("  %5zu B %-22s median %6.1f us  p90 %6.1f us\n", b, mode == 0 ? "D2H" : mode == 1 ? "H2D" : "D2H, sync, H2D, sync", t[t.size() / 2], t[t.size() * 9 / 10]);
        }
    }
    return 0;
}
