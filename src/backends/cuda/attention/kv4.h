#pragma once
// int4 KV cache for the gated-GQA layers (Qwen3.8: 4 kv heads x head_dim 256).
//
// Storage (per attention layer, per kv head):
//   * quantized BLOCK RECORDS, 64 tokens each, 18432 B, self-contained (this is the unit a later tier - host RAM, cold
//     blocks - moves around; all addressing of records goes through kv4_rec()):
//        [0      , 8192 )  K data   : [64 tokens][32 words]; word w of a token row holds channels 4w..4w+3 in nibbles 0..3
//                                      and 128+4w..128+4w+3 in nibbles 4..7 (a lane of a warp reads its 8 channels with
//                                      one 32-bit load)
//        [8192   , 9216 )  K params : half2 (scale, zero). kmode 0 (per-channel, KIVI style): [256 channels], statistics
//                                      over the 64 tokens of the block. kmode 1 (per-token): [64 tokens][4 groups of 64
//                                      channels]
//        [9216   , 17408)  V data   : [128 byte-columns][64 tokens], byte b = channel b (low nibble) | channel b+128 (high)
//        [17408  , 18432)  V params : half2 (scale, zero) per [64 tokens][4 groups of 64 channels] (per-token groups)
//     value = nibble * scale + zero (asymmetric min/max, 16 levels; scale/zero rounded to fp16 BEFORE quantizing)
//     => 4.5 bits per element, 288 B/token/head, 18 KiB/token for the 16 layers (fp32 KV: 128 KiB, fp16: 64 KiB).
//   * a fp16 RING of the most recent 256 positions (slot = pos % 256) for K and V.
// Window rule (one function of the context length n, shared by prefill, decode and MTP verify so that every path sees
// the same bits): keys [0, kv4_q(n)) are read from quantized blocks, keys [kv4_q(n), n) from the fp16 ring.
// kv4_q(n) = 64 * floor((n-128)/64) (0 for n < 192) => the window holds 128..191 un-quantized tokens.
// Block b is quantized when position 64*(b+3)-1 is written (its tokens are then >= 128 old); the quantization reads the
// ring, is a pure function of the 64 tokens, so MTP roll-back (which only ever rewrites positions newer than that) and
// rewrite is idempotent.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace qv::cuda {

constexpr int KV4_BLK = 64;          // tokens per block
constexpr int KV4_D = 256;           // head_dim (only 256 is supported)
constexpr int KV4_RING = 256;        // fp16 ring slots (positions)
constexpr int KV4_WIN = 128;         // minimum un-quantized window
constexpr int KV4_LAG = 3;           // block b is quantized once position 64*(b+LAG)-1 has been written
constexpr size_t KV4_REC_K = 0, KV4_REC_KP = 8192, KV4_REC_V = 9216, KV4_REC_VP = 17408, KV4_REC_BYTES = 18432;

struct Kv4Layer {
    uint8_t* blocks = nullptr;  // [Hkv][nblocks][KV4_REC_BYTES]
    __half* ring_k = nullptr;   // [Hkv][KV4_RING][D]
    __half* ring_v = nullptr;
    int nblocks = 0;            // blocks per head
    int kmode = 0;              // 0: K params per channel, 1: per token (64-channel groups)
    int ring_n = KV4_RING;      // ring slots per head
    int exact = 0;              // 1: "fp16 KV" mode: ring_n == max_ctx slots, nothing is ever quantized (kv4_q == 0)
};

__host__ __device__ __forceinline__ int kv4_q(const Kv4Layer& l, int n) { return (!l.exact && n >= KV4_WIN) ? ((n - KV4_WIN) >> 6) << 6 : 0; }
__host__ __device__ __forceinline__ int kv4_slot(const Kv4Layer& l, int p) { return l.exact ? p : (p & (KV4_RING - 1)); }

// single place that knows where a block record lives
__host__ __device__ __forceinline__ uint8_t* kv4_rec(const Kv4Layer& l, int kvh, int blk) {
    return l.blocks + ((size_t)kvh * l.nblocks + blk) * KV4_REC_BYTES;
}

// Quantize completed blocks (decode / verify): row z of nt rows, position pos[z]; quantizes block (pos+1)/64 - LAG when
// pos+1 is a multiple of 64 and >= 64*LAG, else does nothing (graph-capturable).
void kv4_quant_decode(const Kv4Layer& l, int n_kv_heads, const int* pos, int nt, cudaStream_t s);
// Prefill: after positions [pos0, pos0+T) were staged (fp16 [T][Hkv*D]); quantizes every block that completes.
void kv4_quant_seq(const Kv4Layer& l, int n_kv_heads, const __half* stage_k, const __half* stage_v, int pos0, int T,
                   cudaStream_t s);
// Prefill: copy the last (<=256) staged positions into the ring.
void kv4_ring_commit(const Kv4Layer& l, int n_kv_heads, const __half* stage_k, const __half* stage_v, int pos0, int T,
                     cudaStream_t s);

}  // namespace qv::cuda
