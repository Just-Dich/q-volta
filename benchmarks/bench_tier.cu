// Host(pinned)->GPU transfer of scattered int4 KV blocks, the access pattern of sparse top-k retrieval (Gen 6 H-B).
//   unit = one (layer, kv head, 64-token block): K rows + V rows, 64 x 144 B x 2 = 18432 B (int4 group-64 + fp16 scale/min, 4.5 bit)
//   "all-heads" unit = the same block for the 4 kv heads (73728 B), i.e. selection shared across heads.
// Methods: (a) one cudaMemcpyAsync per unit over S streams, (b) zero-copy gather kernel reading mapped pinned memory,
//          (c) one contiguous copy of the same byte count (upper bound), (d) pipeline emulation: 16 attention layers, each
//          preceded by a GPU busy-wait of `layer_us`, prefetch of layer i+1 overlapped with layer i (copy engines).
// Output: one JSON object on stdout.
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); std::exit(1);} } while (0)
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

constexpr size_t kUnit = 64 * 144 * 2;       // 18432
constexpr int kHkv = 4, kLayers = 16;

__global__ void gather_kernel(const uint4* __restrict__ pool, const unsigned long long* __restrict__ off, uint4* __restrict__ dst,
                              int unit_u4, int nunits) {
  // grid-stride over units: a small persistent grid (few SMs) is enough to saturate PCIe and leaves the SMs to the compute kernels
  for (int u = blockIdx.x; u < nunits; u += gridDim.x) {
    const uint4* src = pool + off[u];
    uint4* d = dst + (size_t)u * unit_u4;
    for (int i = threadIdx.x; i < unit_u4; i += blockDim.x * 4) {
        uint4 a[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) { int j = i + u * blockDim.x; if (j < unit_u4) a[u] = src[j]; }
#pragma unroll
        for (int u = 0; u < 4; ++u) { int j = i + u * blockDim.x; if (j < unit_u4) d[j] = a[u]; }
    }
  }
}
__global__ void stream_read(const uint4* __restrict__ p, size_t n, int* sink) {
    uint4 acc = make_uint4(0, 0, 0, 0);
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) { uint4 v = p[i]; acc.x ^= v.x; acc.y += v.y; acc.z ^= v.z; acc.w += v.w; }
    if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x12345678u) *sink = 1;
}
__global__ void spin_kernel(long long cycles, int* sink) {
    long long t0 = clock64();
    while (clock64() - t0 < cycles) {}
    if (threadIdx.x == 0 && blockIdx.x == 0 && cycles < 0) *sink = 1;
}

static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

int main(int argc, char** argv) {
    const int ntok_ctx = argc > 1 ? std::atoi(argv[1]) : 262144;
    const bool pipe_only = argc > 2;
    const int nblk = ntok_ctx / 64;
    CK(cudaSetDevice(0));
    const size_t layer_bytes = (size_t)kHkv * nblk * kUnit, pool_bytes = layer_bytes * kLayers;
    char* pool; CK(cudaHostAlloc((void**)&pool, pool_bytes, cudaHostAllocMapped | cudaHostAllocPortable));
    { // first touch with a pattern, parallel-ish
        for (size_t i = 0; i < pool_bytes; i += 4096) pool[i] = (char)(i >> 12);
    }
    char* dpool; CK(cudaHostGetDevicePointer((void**)&dpool, pool, 0));
    const size_t DMAX = 256u << 20;
    char* ddst; CK(cudaMalloc((void**)&ddst, DMAX));
    unsigned long long* doff; CK(cudaMalloc((void**)&doff, 1 << 20));
    int* dsink; CK(cudaMalloc((void**)&dsink, 4));
    cudaStream_t st[16]; for (auto& s : st) CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    std::mt19937_64 rng(1);
    std::printf("{\"ctx_tokens\": %d, \"pool_gib\": %.2f, \"unit_bytes\": %zu,\n \"scatter\": [\n", ntok_ctx, pool_bytes / 1073741824.0, kUnit);
    bool first = true;
    for (int fetch_tok : {1024, 2048, 4096, 8192}) {
        const int kb = fetch_tok / 64;  // blocks per kv head per layer
        for (int shared : {0, 1}) {      // 0: per-head selection (units of 1 head); 1: all-heads units
            const size_t unit = shared ? kUnit * kHkv : kUnit;
            const int nblk_total = shared ? kb : kb * kHkv;
            const size_t bytes = (size_t)nblk_total * unit;
            auto pick = [&](int layer, std::vector<size_t>& offs) {  // random distinct blocks per head (per-head) or shared
                offs.clear();
                std::vector<int> ids(nblk); std::iota(ids.begin(), ids.end(), 0);
                if (shared) {
                    std::shuffle(ids.begin(), ids.end(), rng);
                    for (int i = 0; i < kb; ++i) offs.push_back((size_t)layer * layer_bytes + (size_t)ids[i] * kHkv * kUnit);  // layout [layer][block][head]
                } else {
                    for (int h = 0; h < kHkv; ++h) {
                        std::shuffle(ids.begin(), ids.end(), rng);
                        for (int i = 0; i < kb; ++i) offs.push_back((size_t)layer * layer_bytes + ((size_t)ids[i] * kHkv + h) * kUnit);
                    }
                }
            };
            std::vector<size_t> offs;
            // (a) memcpyAsync per unit, S streams
            for (int S : {1, 4, 8, 16}) {
                std::vector<double> t;
                for (int r = 0; r < 40; ++r) {
                    pick(r % kLayers, offs);
                    double t0 = now();
                    for (size_t i = 0; i < offs.size(); ++i) CK(cudaMemcpyAsync(ddst + i * unit, pool + offs[i], unit, cudaMemcpyHostToDevice, st[i % S]));
                    for (int s = 0; s < S; ++s) CK(cudaStreamSynchronize(st[s]));
                    if (r >= 5) t.push_back((now() - t0) * 1e3);
                }
                double m = median(t);
                std::printf("%s  {\"method\": \"memcpyAsync\", \"streams\": %d, \"fetch_tokens_per_head\": %d, \"unit_bytes\": %zu, \"units\": %d, \"bytes\": %zu, \"ms_median\": %.4f, \"gbps\": %.2f}", first ? "" : ",\n", S, fetch_tok, unit, nblk_total, bytes, m, bytes / (m * 1e-3) / 1e9);
                first = false;
            }
            // (b) gather kernel zero-copy
            {
                std::vector<double> t;
                for (int r = 0; r < 40; ++r) {
                    pick(r % kLayers, offs);
                    std::vector<unsigned long long> o16(offs.size());
                    for (size_t i = 0; i < offs.size(); ++i) o16[i] = offs[i] / 16;
                    CK(cudaMemcpy(doff, o16.data(), o16.size() * 8, cudaMemcpyHostToDevice));
                    CK(cudaDeviceSynchronize());
                    double t0 = now();
                    gather_kernel<<<(unsigned)std::min<size_t>(offs.size(), 16), 512, 0, st[0]>>>((const uint4*)dpool, doff, (uint4*)ddst, (int)(unit / 16), (int)offs.size());
                    CK(cudaStreamSynchronize(st[0]));
                    if (r >= 5) t.push_back((now() - t0) * 1e3);
                }
                double m = median(t);
                std::printf(",\n  {\"method\": \"zerocopy_gather_kernel_16blocks\", \"fetch_tokens_per_head\": %d, \"unit_bytes\": %zu, \"units\": %d, \"bytes\": %zu, \"ms_median\": %.4f, \"gbps\": %.2f}", fetch_tok, unit, nblk_total, bytes, m, bytes / (m * 1e-3) / 1e9);
            }
            // (c) contiguous copy of the same bytes
            {
                std::vector<double> t;
                for (int r = 0; r < 40; ++r) {
                    double t0 = now();
                    CK(cudaMemcpyAsync(ddst, pool + (size_t)(r % kLayers) * layer_bytes, bytes, cudaMemcpyHostToDevice, st[0]));
                    CK(cudaStreamSynchronize(st[0]));
                    if (r >= 5) t.push_back((now() - t0) * 1e3);
                }
                double m = median(t);
                std::printf(",\n  {\"method\": \"contiguous_memcpy\", \"fetch_tokens_per_head\": %d, \"unit_bytes\": %zu, \"bytes\": %zu, \"ms_median\": %.4f, \"gbps\": %.2f}", fetch_tok, unit, bytes, m, bytes / (m * 1e-3) / 1e9);
            }
        }
    }
    std::printf("\n ],\n \"pipeline\": [\n");
    // (d) pipeline emulation. Stand-in for the GPU work of one attention-layer group (3 GDN + 1 attention layer + MLPs, ~0.8 GB of
    //     weights read per group => ~1.2 ms): a multi-wave (6400-block) bandwidth-bound read kernel like the real GEMV grids, so the fetch really competes for SMs/HBM.
    //     step = 16 groups. Prefetch for group i+1 is issued when group i starts ("overlapped") or just before group i ("critical").
    first = true;
    const size_t RB = 800u << 20;
    char* dread; CK(cudaMalloc((void**)&dread, RB));
    CK(cudaMemset(dread, 1, RB));
    int* dsum; CK(cudaMalloc((void**)&dsum, 4));
    int prio_lo, prio_hi; CK(cudaDeviceGetStreamPriorityRange(&prio_lo, &prio_hi));
    cudaStream_t sc, sf;  // compute stream (normal), fetch stream (highest priority)
    CK(cudaStreamCreateWithPriority(&sc, cudaStreamNonBlocking, prio_lo));
    CK(cudaStreamCreateWithPriority(&sf, cudaStreamNonBlocking, prio_hi));
    for (int fetch_tok : {2048, 4096, 8192}) {
        const int kb = fetch_tok / 64;
        for (int method = 0; method < 4; ++method) {  // 0..2: zero-copy gather kernel with 4 / 16 / 64 blocks of 512 threads, 3: memcpyAsync per unit (8 streams)
            const int gblocks = method == 0 ? 4 : method == 1 ? 16 : 64;
            for (int mode = 0; mode < 3; ++mode) {  // 0: no fetch, 1: prefetch overlapped with the previous group, 2: fetch on the critical path
                if (mode == 0 && (method != 0 || fetch_tok != 2048)) continue;
                std::vector<double> t;
                const int S = 8;
                std::vector<std::vector<size_t>> sel(kLayers);
                for (int r = 0; r < 25; ++r) {
                    for (int l = 0; l < kLayers; ++l) {
                        sel[l].clear();
                        std::vector<int> ids(nblk); std::iota(ids.begin(), ids.end(), 0);
                        for (int h = 0; h < kHkv; ++h) { std::shuffle(ids.begin(), ids.end(), rng); for (int i = 0; i < kb; ++i) sel[l].push_back((size_t)l * layer_bytes + ((size_t)ids[i] * kHkv + h) * kUnit); }
                    }
                    CK(cudaDeviceSynchronize());
                    double t0 = now();
                    auto issue = [&](int l) {
                        if (method == 3) {
                            for (size_t i = 0; i < sel[l].size(); ++i) CK(cudaMemcpyAsync(ddst + (l % 2) * (DMAX / 2) + i * kUnit, pool + sel[l][i], kUnit, cudaMemcpyHostToDevice, st[1 + i % S]));
                        } else {
                            // persistent-style: grid blocks loop over units; offsets precomputed on the host per call (stand-in for the selection kernel's output)
                            std::vector<unsigned long long> o16(sel[l].size());
                            for (size_t i = 0; i < sel[l].size(); ++i) o16[i] = sel[l][i] / 16;
                            CK(cudaMemcpyAsync(doff + (l % 2) * 8192, o16.data(), o16.size() * 8, cudaMemcpyHostToDevice, sf));
                            gather_kernel<<<(unsigned)gblocks, 512, 0, sf>>>((const uint4*)dpool, doff + (l % 2) * 8192, (uint4*)(ddst + (l % 2) * (DMAX / 2)), (int)(kUnit / 16), (int)sel[l].size());
                        }
                    };
                    cudaEvent_t ev[kLayers + 1][8];
                    for (int l = 0; l <= kLayers; ++l) for (int s = 0; s < 8; ++s) CK(cudaEventCreateWithFlags(&ev[l][s], cudaEventDisableTiming));
                    auto record_fetch = [&](int l) {  // right after issue(l): marks the end of layer l's fetch on each stream used
                        if (method == 3) { for (int s = 1; s <= S; ++s) CK(cudaEventRecord(ev[l][s - 1], st[s])); }
                        else CK(cudaEventRecord(ev[l][0], sf));
                    };
                    auto wait_fetch = [&](int l) {
                        if (method == 3) { for (int s = 1; s <= S; ++s) CK(cudaStreamWaitEvent(sc, ev[l][s - 1], 0)); }
                        else CK(cudaStreamWaitEvent(sc, ev[l][0], 0));
                    };
                    cudaEvent_t ce[kLayers];
                    for (int l = 0; l < kLayers; ++l) CK(cudaEventCreateWithFlags(&ce[l], cudaEventDisableTiming));
                    auto gate = [&](int l) {  // the fetch stream(s) may start only after compute group l has finished (l < 0: free)
                        if (l < 0) return;
                        if (method == 3) { for (int s = 1; s <= S; ++s) CK(cudaStreamWaitEvent(st[s], ce[l], 0)); }
                        else CK(cudaStreamWaitEvent(sf, ce[l], 0));
                    };
                    // overlapped: fetch(l+1) may start when compute l starts (= compute l-1 finished); critical: fetch(l) starts when compute l-1 finished
                    if (mode == 1) { issue(0); record_fetch(0); }
                    for (int l = 0; l < kLayers; ++l) {
                        if (mode == 1) { if (l + 1 < kLayers) { gate(l - 1); issue(l + 1); record_fetch(l + 1); } wait_fetch(l); }
                        else if (mode == 2) { gate(l - 1); issue(l); record_fetch(l); wait_fetch(l); }
                        stream_read<<<6400, 256, 0, sc>>>((const uint4*)dread, RB / 16, dsum);
                        CK(cudaEventRecord(ce[l], sc));
                    }
                    CK(cudaStreamSynchronize(sc)); CK(cudaDeviceSynchronize());
                    if (r >= 5) t.push_back((now() - t0) * 1e3);
                    for (int l = 0; l <= kLayers; ++l) for (int s = 0; s < 8; ++s) CK(cudaEventDestroy(ev[l][s]));
                    for (int l = 0; l < kLayers; ++l) CK(cudaEventDestroy(ce[l]));
                }
                std::printf("%s  {\"mode\": \"%s\", \"copy\": \"%s\", \"fetch_tokens_per_head\": %d, \"step_ms_median\": %.3f, \"step_ms_min\": %.3f}", first ? "" : ",\n", mode == 0 ? "no_fetch" : mode == 1 ? "prefetch_overlapped" : "fetch_on_critical_path", mode == 0 ? "-" : method == 3 ? "memcpyAsync_per_unit_8streams" : method == 0 ? "gather_kernel_4blocks" : method == 1 ? "gather_kernel_16blocks" : "gather_kernel_64blocks", fetch_tok, median(t), *std::min_element(t.begin(), t.end()));
                first = false;
            }
        }
    }
    std::printf("\n ]}\n");
    return 0;
}
