// Tensor-core (Volta HMMA via WMMA 16x16x16, fp16 in / fp32 accumulate) flash-style causal prefill attention for the
// gated GQA layers. Same contract as attn_seq_fp32. A block owns 8 consecutive queries x the 6 q heads of one kv head
// (48 rows = 3 m16 tiles) and streams 64-key tiles of K and V (converted to fp16 in shared memory) once for all rows.
// Online softmax state and the O accumulator stay fp32; P is rounded to fp16 for the PV product.
// KV-cache reads go through KvSrcF32 only (tile -> fp16 smem), so an int4-block source can replace it.
#include <mma.h>

#include "attn_seq.h"

namespace qv::cuda {
using namespace nvcuda;

namespace {
constexpr int D = 256, G = 6;

// Tile configuration: TQ queries x 6 q heads = R rows; KT keys per tile; one warp per 16x16 score tile.
// SPLITD: 2*KTN warps, each (key tile, D half) over all row tiles with the B fragment reused; two partial score buffers.
// Otherwise one warp per (row tile, key tile) with the full D loop.
template <int TQ_, int KT_, bool SPLITD_>
struct Cfg {
    static constexpr bool SPLITD = SPLITD_;
    static constexpr int TQ = TQ_, KT = KT_, R = TQ_ * G, RT = R / 16, KTN = KT_ / 16, NW = SPLITD_ ? 2 * KTN : RT * KTN, NT = NW * 32;
    static constexpr int LDH = D + 8;    // fp16 smem row stride (528 B) for Q and K/V tiles
    static constexpr int LDS = KT + 4;   // fp32 score stride
    static constexpr int LDP = KT + 8;   // fp16 P stride
    static constexpr int RPW = SPLITD_ ? RT : (RT == 6 ? 2 : 1), RTG = RT / RPW, CTG = NW / RTG, CPW = 16 / CTG;  // PV: warp owns RPW x CPW tiles
    static constexpr int NF = (KT * (D / 4) + NT - 1) / NT;  // float4 per thread per tile
    static constexpr size_t SMEM = sizeof(__half) * ((size_t)R * LDH + (size_t)KT * LDH + (size_t)R * LDP) +
                                   sizeof(float) * ((size_t)R * LDS * (SPLITD_ ? 2 : 1) + 2 * R);
    static_assert(R % 16 == 0 && KT % 16 == 0 && 16 % CTG == 0 && NT <= 1024);
};

// KV tile source. fetch() issues the global loads of this thread's share of a [KT][D] tile into registers (so they overlap
// with tensor-core work); store() converts them to fp16 into the smem tile [KT][LDH]. Keys >= n are zero. An int4-block source
// only has to provide the same Frag/fetch/store triple.
template <class C>
struct KvSrcF32 {
    const float* k;
    const float* v;
    int max_ctx;
    struct Frag { float4 x[C::NF]; };
    __device__ __forceinline__ void fetch(Frag& f, bool is_v, int kvh, int k0, int n, int tid) const {
        const float* base = (is_v ? v : k) + ((size_t)kvh * max_ctx) * D;
#pragma unroll
        for (int u = 0; u < C::NF; ++u) {
            int idx = tid + u * C::NT, j = idx / (D / 4), c4 = idx % (D / 4), key = k0 + j;
            f.x[u] = ((C::NF * C::NT == C::KT * (D / 4) || idx < C::KT * (D / 4)) && key < n) ? *reinterpret_cast<const float4*>(base + (size_t)key * D + c4 * 4)
                                                                  : make_float4(0.f, 0.f, 0.f, 0.f);
        }
    }
    __device__ __forceinline__ void store(const Frag& f, __half* dst, int tid) const {
#pragma unroll
        for (int u = 0; u < C::NF; ++u) {
            int idx = tid + u * C::NT, j = idx / (D / 4), c4 = idx % (D / 4);
            if (C::NF * C::NT != C::KT * (D / 4) && idx >= C::KT * (D / 4)) continue;
            __half2 lo = __floats2half2_rn(f.x[u].x, f.x[u].y), hi = __floats2half2_rn(f.x[u].z, f.x[u].w);
            *reinterpret_cast<uint2*>(dst + j * C::LDH + c4 * 4) = make_uint2(*reinterpret_cast<uint32_t*>(&lo), *reinterpret_cast<uint32_t*>(&hi));
        }
    }
};

// Volta (sm_70) WMMA m16n16k16 fp32 accumulator layout (verified by probe): element i of lane l sits at
//   row = (l&1) + 2*((i>>1)&1) + 4*((l>>4)&1) + 8*((l>>2)&1),  col = 2*((l>>1)&1) + 8*((l>>3)&1) + (i&1) + 4*(i>>2)
__device__ __forceinline__ int acc_row(int l, int i) { return (l & 1) + 2 * ((i >> 1) & 1) + 4 * ((l >> 4) & 1) + 8 * ((l >> 2) & 1); }
__device__ __forceinline__ int acc_col(int l, int i) { return 2 * ((l >> 1) & 1) + 8 * ((l >> 3) & 1) + (i & 1) + 4 * (i >> 2); }

template <class C>
__global__ void __launch_bounds__(C::NT, 1) attn_seq_tc_kernel(AttnSeqArgs a, KvSrcF32<C> src) {
    constexpr int R = C::R, KT = C::KT, LDH = C::LDH, LDS = C::LDS, LDP = C::LDP, NT = C::NT, TQ = C::TQ;
    extern __shared__ __align__(16) unsigned char smraw[];
    __half* Qs = reinterpret_cast<__half*>(smraw);  // [R][LDH]
    __half* KV = Qs + R * LDH;                      // [KT][LDH]
    __half* Ph = KV + KT * LDH;                     // [R][LDP]
    float* S = reinterpret_cast<float*>(Ph + R * LDP);  // [R][LDS]
    float* S1 = S + R * LDS;                             // D-half 1 partial scores (SPLITD only)
    float* alpha_s = S + R * LDS * (C::SPLITD ? 2 : 1);  // [R]
    float* l_s = alpha_s + R;                            // [R]

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int kvh = blockIdx.y;
    const int q0 = (gridDim.x - 1 - blockIdx.x) * TQ;  // heaviest (latest) tiles first
    const int Hq = a.n_heads;
    const int n = a.pos0 + min(q0 + TQ - 1, a.T - 1) + 1;  // keys needed by this block

    for (int idx = tid; idx < R * (D / 4); idx += NT) {
        int r = idx / (D / 4), c4 = idx % (D / 4);
        int qi = q0 + r / G, hq = kvh * G + r % G;
        float4 x = make_float4(0.f, 0.f, 0.f, 0.f);
        if (qi < a.T) x = *reinterpret_cast<const float4*>(a.q + (size_t)qi * Hq * D + hq * D + c4 * 4);
        __half2 lo = __floats2half2_rn(x.x, x.y), hi = __floats2half2_rn(x.z, x.w);
        *reinterpret_cast<uint2*>(Qs + r * LDH + c4 * 4) = make_uint2(*reinterpret_cast<uint32_t*>(&lo), *reinterpret_cast<uint32_t*>(&hi));
    }

    // softmax mapping: 4 threads per row (R*4 == NT or less), KT/4 keys each
    constexpr int SU = KT / 4;
    const int srow = tid >> 2, ssub = tid & 3;
    const bool sm_active = srow < R;
    const int lim = a.pos0 + min(q0 + srow / G, a.T - 1);  // last visible key of this row
    float m_run = -INFINITY, l_run = 0.f;

    // PV ownership: warp -> RPW row tiles x CPW column tiles
    const int rg = warp / C::CTG, cg = warp % C::CTG;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> O[C::RPW][C::CPW];
#pragma unroll
    for (int r = 0; r < C::RPW; ++r)
#pragma unroll
        for (int c = 0; c < C::CPW; ++c) wmma::fill_fragment(O[r][c], 0.f);

    typename KvSrcF32<C>::Frag fr;
    src.fetch(fr, false, kvh, 0, n, tid);  // first K tile
    for (int k0 = 0; k0 < n; k0 += KT) {
        __syncthreads();  // previous PV finished reading KV
        src.store(fr, KV, tid);
        src.fetch(fr, true, kvh, k0, n, tid);  // V tile in flight during S
        __syncthreads();
        if constexpr (C::SPLITD) {  // S = Q K^T: warp -> key tile (warp%KTN), D half (warp/KTN), all row tiles
            const int kt = warp % C::KTN, dh = warp / C::KTN;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> sacc[C::RT];
#pragma unroll
            for (int r = 0; r < C::RT; ++r) wmma::fill_fragment(sacc[r], 0.f);
#pragma unroll
            for (int ks = 0; ks < D / 32; ++ks) {
                const int d0 = dh * (D / 2) + ks * 16;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
                wmma::load_matrix_sync(b, KV + kt * 16 * LDH + d0, LDH);
#pragma unroll
                for (int r = 0; r < C::RT; ++r) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> af;
                    wmma::load_matrix_sync(af, Qs + r * 16 * LDH + d0, LDH);
                    wmma::mma_sync(sacc[r], af, b, sacc[r]);
                }
            }
            float* Sd = dh ? S1 : S;
#pragma unroll
            for (int r = 0; r < C::RT; ++r) wmma::store_matrix_sync(Sd + r * 16 * LDS + kt * 16, sacc[r], LDS, wmma::mem_row_major);
        } else {   // S = Q K^T: warp -> (row tile, key tile), full D
            const int rt = warp / C::KTN, kt = warp % C::KTN;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> s;
            wmma::fill_fragment(s, 0.f);
#pragma unroll
            for (int ks = 0; ks < D / 16; ++ks) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> af;
                wmma::load_matrix_sync(b, KV + kt * 16 * LDH + ks * 16, LDH);
                wmma::load_matrix_sync(af, Qs + rt * 16 * LDH + ks * 16, LDH);
                wmma::mma_sync(s, af, b, s);
            }
            wmma::store_matrix_sync(S + rt * 16 * LDS + kt * 16, s, LDS, wmma::mem_row_major);
        }
        __syncthreads();  // scores complete; K tile free
        src.store(fr, KV, tid);  // V tile
        if (k0 + KT < n) src.fetch(fr, false, kvh, k0 + KT, n, tid);  // next K tile in flight during softmax + PV
        if (sm_active) {
            float sv[SU];
            float mx = -INFINITY;
#pragma unroll
            for (int u = 0; u < SU; ++u) {
                int c = ssub + 4 * u;
                float x = (C::SPLITD ? S[srow * LDS + c] + S1[srow * LDS + c] : S[srow * LDS + c]) * (0.0625f * 1.4426950408889634f);  // log2 domain
                sv[u] = (k0 + c <= lim) ? x : -INFINITY;
                mx = fmaxf(mx, sv[u]);
            }
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
            const float mnew = fmaxf(m_run, mx);
            float alpha = 1.f, sum = 0.f;
            if (mnew == -INFINITY) {
#pragma unroll
                for (int u = 0; u < SU; ++u) Ph[srow * LDP + ssub + 4 * u] = __float2half(0.f);
            } else {
                alpha = exp2f(m_run - mnew);  // m_run=-inf -> 0
#pragma unroll
                for (int u = 0; u < SU; ++u) {
                    float p = exp2f(sv[u] - mnew);
                    Ph[srow * LDP + ssub + 4 * u] = __float2half(p);
                    sum += p;
                }
            }
            sum += __shfl_xor_sync(0xffffffffu, sum, 1);
            sum += __shfl_xor_sync(0xffffffffu, sum, 2);
            l_run = l_run * alpha + sum;
            m_run = mnew;
            if (ssub == 0) { alpha_s[srow] = alpha; l_s[srow] = l_run; }
        }
        __syncthreads();  // V tile + P + alpha visible
#pragma unroll
        for (int r = 0; r < C::RPW; ++r) {
            float al[2];
#pragma unroll
            for (int j = 0; j < 2; ++j) al[j] = alpha_s[(rg * C::RPW + r) * 16 + acc_row(lane, 2 * j)];
#pragma unroll
            for (int c = 0; c < C::CPW; ++c)
#pragma unroll
                for (int i = 0; i < 8; ++i) O[r][c].x[i] *= al[(i >> 1) & 1];
        }
#pragma unroll
        for (int ks = 0; ks < KT / 16; ++ks) {
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b[C::CPW];
#pragma unroll
            for (int c = 0; c < C::CPW; ++c) wmma::load_matrix_sync(b[c], KV + ks * 16 * LDH + (cg * C::CPW + c) * 16, LDH);
#pragma unroll
            for (int r = 0; r < C::RPW; ++r) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> af;
                wmma::load_matrix_sync(af, Ph + (rg * C::RPW + r) * 16 * LDP + ks * 16, LDP);
#pragma unroll
                for (int c = 0; c < C::CPW; ++c) wmma::mma_sync(O[r][c], af, b[c], O[r][c]);
            }
        }
    }
    __syncthreads();  // l_s final
#pragma unroll
    for (int r = 0; r < C::RPW; ++r)
#pragma unroll
        for (int c = 0; c < C::CPW; ++c)
#pragma unroll
            for (int i = 0; i < 8; i += 2) {
                const int row = (rg * C::RPW + r) * 16 + acc_row(lane, i), col = (cg * C::CPW + c) * 16 + acc_col(lane, i);
                const int qi = q0 + row / G, hq = kvh * G + row % G;
                if (qi >= a.T) continue;
                const float inv = 1.f / l_s[row];
                const float2 g = *reinterpret_cast<const float2*>(a.qg + (size_t)qi * Hq * 2 * D + hq * 2 * D + D + col);
                float o0 = O[r][c].x[i] * inv * (1.f / (1.f + expf(-g.x)));
                float o1 = O[r][c].x[i + 1] * inv * (1.f / (1.f + expf(-g.y)));
                const size_t off = (size_t)qi * Hq * D + hq * D + col;
                if (a.out_f) *reinterpret_cast<float2*>(a.out_f + off) = make_float2(o0, o1);
                if (a.out_h)
                    *reinterpret_cast<__half2*>(a.out_h + off) =
                        __floats2half2_rn(fminf(fmaxf(o0, -65504.f), 65504.f), fminf(fmaxf(o1, -65504.f), 65504.f));
            }
}

template <class C>
void launch(const AttnSeqArgs& a, cudaStream_t s) {
    static bool init = false;
    if (!init) {
        QV_CUDA_CHECK(cudaFuncSetAttribute(attn_seq_tc_kernel<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)C::SMEM));
        init = true;
    }
    dim3 grid((a.T + C::TQ - 1) / C::TQ, a.n_kv_heads);
    attn_seq_tc_kernel<C><<<grid, C::NT, C::SMEM, s>>>(a, KvSrcF32<C>{a.k_cache, a.v_cache, a.max_ctx});
}
}  // namespace

void attn_seq_tc(const AttnSeqArgs& a, cudaStream_t s) {
    if (a.n_heads != G * a.n_kv_heads) { std::fprintf(stderr, "attn_seq_tc: need 6 q heads per kv head\n"); std::abort(); }
    if (a.T <= 0) return;
    // Measured alternatives (V100, pos0=8192 T=1024): Cfg<8,64,true> 7.95 ms (chosen), Cfg<8,64,false> 11.1 ms (12 warps, S phase
    // load-bound), Cfg<16,32,false> 8.7 ms (96 rows, spills at 170 regs).
    launch<Cfg<8, 64, true>>(a, s);
}

}  // namespace qv::cuda
