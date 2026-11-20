// Host-side (AVX2+FMA+F16C) flash-decoding partial attention over an int4 asymmetric KV cache.
//
// Row format (one token, one kv head, K or V): 144 bytes
//   [0,128)   256 4-bit codes. Per group of 64 elements the group owns 32 bytes; byte j holds
//             element j in the low nibble and element j+32 in the high nibble.
//   [128,144) 4 x { fp16 scale, fp16 min } (one pair per group); value = code*scale + min.
// Buffer layout: [kv_head][token][144 bytes], K and V in separate buffers.
// q / out layout: head-major, q head h uses kv head h / 6 (GQA 6:1).
#pragma once
#include <cstdint>
#include <cstddef>

namespace qv::cpu {

constexpr int kHq = 24, kHkv = 4, kGqa = kHq / kHkv, kD = 256, kGroup = 64;
constexpr int kGroups = kD / kGroup;
constexpr int kRowBytes = kD / 2 + kGroups * 4;  // 144

// Quantize / dequantize one 256-element row (used by tests and by whoever fills the cold cache).
void pack_row(const float* x, uint8_t* row);
void dequant_row(const uint8_t* row, float* x);

// Partial attention over tokens [0, ntok) of every kv head.
//   q   : [kHq][kD] fp32 (unscaled; 1/sqrt(kD) is applied inside)
//   K,V : [kHkv][ntok][kRowBytes]
//   m,l : [kHq]; acc : [kHq][kD]  (unnormalised: out = acc / l once merged with other partials)
// nthreads <= 0 -> omp default. chunk_tokens = tokens per (kv head, chunk) work item. ntok == 0 yields m=-inf, l=0, acc=0.
void attn_cpu_partial(const float* q, const uint8_t* K, const uint8_t* V, int64_t ntok,
                      float* m, float* l, float* acc, int nthreads = 0, int chunk_tokens = 2048);

// Same, restricted to selected 64-token blocks (Gen 6 H-B/H-C): blocks[kvh*nb + i] = block id (tokens [id*64, id*64+64)) of kv head kvh.
// K,V : [kHkv][ntok_per_head][kRowBytes] (the full cold store of one layer). Work item = (kv head, group of blocks_per_item blocks).
void attn_cpu_partial_blocks(const float* q, const uint8_t* K, const uint8_t* V, int64_t ntok_per_head, const int32_t* blocks, int nb,
                             float* m, float* l, float* acc, int nthreads = 0, int blocks_per_item = 8);

// Merge partial b into a (a <- a (+) b); works for CPU/GPU partials alike.
void attn_merge(float* m_a, float* l_a, float* acc_a, const float* m_b, const float* l_b,
                const float* acc_b);

// Straightforward scalar double-precision reference with the same format and semantics.
void attn_cpu_ref(const float* q, const uint8_t* K, const uint8_t* V, int64_t ntok, float* m,
                  float* l, float* acc);

// Fill [kHkv][ntok] rows with quantized pseudo-random data (parallel first-touch, for tests/benchmarks).
void fill_random_cache(uint8_t* buf, int64_t ntok, uint64_t seed);

}  // namespace qv::cpu
