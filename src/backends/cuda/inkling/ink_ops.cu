// Inkling-2-Mini decode kernels. Spec: docs/INKLING_SPEC.md; engine: src/runtime/cuda_inkling_engine.cu.
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "../kernels/q8.cuh"
#include "ink_ops.h"

namespace qv::cuda::ink {

namespace {

constexpr int NT = 768;      // threads of the hidden-size kernels: 3 elements per thread, element = t + 768 j
constexpr int NW = NT / 32;  // 24 warps = 24 chunks per j

template <int BITS>
__device__ __forceinline__ uint32_t qword(const uint32_t* w, int b) {
    if constexpr (BITS == 2) {
        return (w[b & 1] >> (2 * (b >> 1))) & 0x03030303u;
    } else if constexpr (BITS == 3) {
        return ((w[b & 1] >> (2 * (b >> 1))) & 0x03030303u) | (((w[2] >> b) & 0x01010101u) << 2);
    } else if constexpr (BITS == 4) {
        return (w[b & 3] >> (4 * (b >> 2))) & 0x0F0F0F0Fu;
    } else if constexpr (BITS == 5) {
        return ((w[b & 3] >> (4 * (b >> 2))) & 0x0F0F0F0Fu) | (((w[4] >> b) & 0x01010101u) << 4);
    } else if constexpr (BITS == 6) {
        return ((w[b & 3] >> (4 * (b >> 2))) & 0x0F0F0F0Fu) | (((w[4 + (b & 1)] >> (2 * (b >> 1))) & 0x03030303u) << 4);
    } else {
        static_assert(BITS == 8, "unsupported bits");
        return ((w[b & 3] >> (4 * (b >> 2))) & 0x0F0F0F0Fu) | (((w[4 + (b & 3)] >> (4 * (b >> 2))) & 0x0F0F0F0Fu) << 4);
    }
}
__device__ __forceinline__ int dp4a_us(uint32_t a, int b, int c) {
    int d;
    asm("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

// one quantized chunk (32 weights of one row) against one int8 activation chunk
template <int BITS>
__device__ __forceinline__ float qchunk(const uint32_t* wp, const int (&xw)[8], float dx, float xs, int qs, float sc, float mn, float acc) {
    constexpr int ZP = 1 << (BITS - 1);
    uint32_t w[BITS];
#pragma unroll
    for (int b = 0; b < BITS; ++b) w[b] = __ldg(wp + b);
    int d = 0;
#pragma unroll
    for (int b = 0; b < 8; ++b) d = dp4a_us(qword<BITS>(w, b), xw[b], d);
    d -= ZP * qs;
    return fmaf(sc * dx, (float)d, fmaf(fmaf(sc, (float)ZP, mn), xs, acc));
}
__device__ __forceinline__ void load_x(const Q8Vec& xv, size_t c, int (&xw)[8], float& dx, float& xs, int& qs) {
    const int4* xp = reinterpret_cast<const int4*>(xv.q + c * 8);
    const int4 x0 = __ldg(xp), x1 = __ldg(xp + 1);
    xw[0] = x0.x; xw[1] = x0.y; xw[2] = x0.z; xw[3] = x0.w; xw[4] = x1.x; xw[5] = x1.y; xw[6] = x1.z; xw[7] = x1.w;
    dx = __ldg(xv.d + c);
    xs = __ldg(xv.sum + c);
    qs = __ldg(xv.qsum + c);
}
// meta8 (scale, min) of group g of a row
__device__ __forceinline__ void sm8_decode(unsigned v, float rsc, float zlo, float zstep, float& sc, float& mn) {
    sc = (float)(v & 255u) * rsc;
    const float z = (float)(v >> 8) * zstep + zlo;
    mn = -(sc * z);
}

// reduce N values over the block of NT threads (all threads get the sums). sm >= NW*N floats.
template <int N>
__device__ __forceinline__ void bred(float (&v)[N], float* sm) {
    // sm >= NW*N + N floats
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
#pragma unroll
    for (int i = 0; i < N; ++i) v[i] = warp_sum(v[i]);
    __syncthreads();
    if (lane == 0) {
#pragma unroll
        for (int i = 0; i < N; ++i) sm[w * N + i] = v[i];
    }
    __syncthreads();
    if (threadIdx.x < N) {
        float s = 0.f;
#pragma unroll
        for (int k = 0; k < NW; ++k) s += sm[k * N + threadIdx.x];
        sm[NW * N + threadIdx.x] = s;
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < N; ++i) v[i] = sm[NW * N + i];
}

__device__ __forceinline__ Q8Buf q8_stream(const Q8Buf& q, int s) {
    Q8Buf o;
    o.q = q.q + s * (H / 4);
    o.d = q.d + s * HC;
    o.sum = q.sum + s * HC;
    o.qsum = q.qsum + s * HC;
    return o;
}

}  // namespace

// ------------------------------------------------------------------------------------------------------------------
// embedding (4-bit g64 meta8 row dequant) + embed_norm
template <int BITS>
__device__ __forceinline__ float dequant_elem(const QMat& E, size_t row, int c, int lane, size_t C) {
    const uint32_t* wp = E.q + (row * C + c) * BITS;
    uint32_t w[BITS];
#pragma unroll
    for (int b = 0; b < BITS; ++b) w[b] = wp[b];
    const int b = lane & 7, j = lane >> 3;
    const uint32_t code = (qword<BITS>(w, b) >> (8 * j)) & 0xFFu;
    const size_t G = E.K >> E.gshift;
    float sc, mn;
    sm8_decode(E.sm8[row * G + (c >> (E.gshift - 5))], __half2float(E.rs[row]) / 255.0f, E.zlo, E.zstep, sc, mn);
    return fmaf(sc, (float)code, mn);
}

__global__ void __launch_bounds__(NT) k_embed(QMat E, const int* tok, const float* w, float* h, float* blk0) {
    __shared__ float sm[NW];
    const size_t row = (size_t)tok[blockIdx.x];
    h += (size_t)blockIdx.x * H;
    blk0 += (size_t)blockIdx.x * H;
    const size_t C = E.K >> 5;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    float v[3];
    float ss = 0.f;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int c = warp + NW * j;
        switch (E.bits) {
            case 3: v[j] = dequant_elem<3>(E, row, c, lane, C); break;
            case 4: v[j] = dequant_elem<4>(E, row, c, lane, C); break;
            case 5: v[j] = dequant_elem<5>(E, row, c, lane, C); break;
            case 6: v[j] = dequant_elem<6>(E, row, c, lane, C); break;
            default: v[j] = dequant_elem<8>(E, row, c, lane, C); break;
        }
        ss += v[j] * v[j];
    }
    float r = block_sum(ss, sm);
    const float f = rsqrtf(r / H + 1e-6f);
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int i = threadIdx.x + NT * j;
        const float o = w[i] * (v[j] * f);
        h[i] = o;
        blk0[i] = o;
    }
}
void embed_norm(const QMat& E, const int* tok, const float* w, float* h, float* blk0, cudaStream_t s, int T) {
    k_embed<<<T, NT, 0, s>>>(E, tok, w, h, blk0);
}

__global__ void k_softcap(float* x, int n, float cap) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = cap * tanhf(x[i] / cap);
}
void softcap_inplace(float* x, int n, float cap, cudaStream_t s) { k_softcap<<<(n + 255) / 256, 256, 0, s>>>(x, n, cap); }

// ------------------------------------------------------------------------------------------------------------------
// attention residual
template <int QM>
__global__ void __launch_bounds__(NT) k_ar(ArArgs a) {
    __shared__ float sm[NW * 25 + 32];
    const int t = threadIdx.x;
    const float* queries = a.queries + (long)blockIdx.x * a.qbs;
    const long tok = a.allpar ? (long)blockIdx.x : (long)a.tok0, NTK = a.ntok;
    float v[5][3];
#pragma unroll
    for (int b = 0; b < 5; ++b)
#pragma unroll
        for (int j = 0; j < 3; ++j) v[b][j] = 0.f;
    int NB;
    if (a.mode == 0) {
        NB = a.B + 1;
#pragma unroll
        for (int b = 0; b < 4; ++b)
#pragma unroll
            for (int j = 0; j < 3; ++j) v[b][j] = (b < a.B) ? a.blocks[((long)b * NTK + tok) * H + t + NT * j] : 0.f;
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            const float hv = a.h[tok * H + t + NT * j];
            for (int b = 0; b < 5; ++b)
                if (b == a.B) v[b][j] = hv;
        }
    } else {
        float res[3], hin[3];
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            const int c = t + NT * j;
            const float x = a.araw[tok * H + c];
            float s0, s1, s2;
            {
                const long i3 = tok - 3, i2 = tok - 2, i1 = tok - 1;
                s0 = i3 >= 0 ? a.araw[i3 * H + c] : a.cst[(3 + i3) * H + c];
                s1 = i2 >= 0 ? a.araw[i2 * H + c] : a.cst[(3 + i2) * H + c];
                s2 = i1 >= 0 ? a.araw[i1 * H + c] : a.cst[(3 + i1) * H + c];
            }
            const float4 w = *reinterpret_cast<const float4*>(a.cw + 4 * c);
            const float y = w.x * s0 + w.y * s1 + w.z * s2 + w.w * x;
            const float av = (y + x) * 0.44721359549995794f;  // 1/sqrt(5)
            if (NTK == 1) {
                a.cst[c] = s1;
                a.cst[H + c] = s2;
                a.cst[2 * H + c] = x;
            }
            hin[j] = a.h[tok * H + c];
            res[j] = a.boundary ? av : av + hin[j];
            a.res_out[tok * H + c] = res[j];
        }
        NB = a.B + 1 + (a.boundary ? 1 : 0);
#pragma unroll
        for (int b = 0; b < 4; ++b)
#pragma unroll
            for (int j = 0; j < 3; ++j) v[b][j] = (b < a.B) ? a.blocks[((long)b * NTK + tok) * H + t + NT * j] : 0.f;
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            for (int b = 0; b < 5; ++b) {
                if (a.boundary) {
                    if (b == a.B) v[b][j] = hin[j];
                    if (b == a.B + 1) v[b][j] = res[j];
                } else if (b == a.B) {
                    v[b][j] = res[j];
                }
            }
            if (a.boundary) a.blocks[((long)a.B * NTK + tok) * H + t + NT * j] = hin[j];
        }
    }
    float out[QM][3];
    if (!a.mix) {
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            float pv = 0.f;
            for (int b = 0; b < 5; ++b)
                if (b == NB - 1) pv = v[b][j];
            out[0][j] = pv;
        }
    } else {
        float red[5 + 5 * QM];
#pragma unroll
        for (int b = 0; b < 5; ++b) red[b] = (b < NB) ? v[b][0] * v[b][0] + v[b][1] * v[b][1] + v[b][2] * v[b][2] : 0.f;
#pragma unroll
        for (int q = 0; q < QM; ++q) {
            float qv[3];
#pragma unroll
            for (int j = 0; j < 3; ++j) qv[j] = q < a.Q ? queries[(long)q * H + t + NT * j] : 0.f;
#pragma unroll
            for (int b = 0; b < 5; ++b) red[5 + q * 5 + b] = (b < NB && q < a.Q) ? v[b][0] * qv[0] + v[b][1] * qv[1] + v[b][2] * qv[2] : 0.f;
        }
        bred<5 + 5 * QM>(red, sm);
        float rb[5];
#pragma unroll
        for (int b = 0; b < 5; ++b) rb[b] = rsqrtf(red[b] / H + 1e-6f);
        float coef[QM][5];
#pragma unroll
        for (int q = 0; q < QM; ++q) {
            float mx = -1e30f;
#pragma unroll
            for (int b = 0; b < 5; ++b) {
                coef[q][b] = red[5 + q * 5 + b] * rb[b] * (1.0f / 48.0f);  // H^-0.5, H = 2304
                if (b < NB) mx = fmaxf(mx, coef[q][b]);
            }
            float sum = 0.f;
#pragma unroll
            for (int b = 0; b < 5; ++b) {
                coef[q][b] = (b < NB) ? __expf(coef[q][b] - mx) : 0.f;
                sum += coef[q][b];
            }
#pragma unroll
            for (int b = 0; b < 5; ++b) coef[q][b] = coef[q][b] / sum * rb[b];
        }
#pragma unroll
        for (int q = 0; q < QM; ++q)
#pragma unroll
            for (int j = 0; j < 3; ++j) {
                float o = 0.f;
#pragma unroll
                for (int b = 0; b < 5; ++b) o = fmaf(coef[q][b], v[b][j], o);
                out[q][j] = o;
            }
    }
    if (a.epi == 0) {
#pragma unroll
        for (int q = 0; q < QM; ++q)
#pragma unroll
            for (int j = 0; j < 3; ++j) a.raw_out[(long)blockIdx.x * a.rbs + ((long)q * NTK + tok) * H + t + NT * j] = out[q][j];
    } else if (a.epi == 2) {
        float ssq[QM];
#pragma unroll
        for (int q = 0; q < QM; ++q) ssq[q] = out[q][0] * out[q][0] + out[q][1] * out[q][1] + out[q][2] * out[q][2];
        bred<QM>(ssq, sm);
#pragma unroll
        for (int q = 0; q < QM; ++q) {
            const float f = rsqrtf(ssq[q] / H + 1e-6f);
#pragma unroll
            for (int j = 0; j < 3; ++j) {
                const float xv = a.nw[q][t + NT * j] * (out[q][j] * f);
                a.xh_out[((long)q * NTK + tok) * H + t + NT * j] = __float2half(xv);
                if (a.xn_out) a.xn_out[((long)q * NTK + tok) * H + t + NT * j] = xv;
            }
        }
    } else {
        float ssq[QM];
#pragma unroll
        for (int q = 0; q < QM; ++q) ssq[q] = out[q][0] * out[q][0] + out[q][1] * out[q][1] + out[q][2] * out[q][2];
        bred<QM>(ssq, sm);
        const int warp = t >> 5;
#pragma unroll
        for (int q = 0; q < QM; ++q) {
            const float f = rsqrtf(ssq[q] / H + 1e-6f);
            const Q8Buf o = q8_stream(a.q8, q);
#pragma unroll
            for (int j = 0; j < 3; ++j) {
                const float xn = a.nw[q][t + NT * j] * (out[q][j] * f);
                a.xn_out[(long)q * H + t + NT * j] = xn;
                q8_store_chunk(xn, warp + NW * j, o);
            }
        }
    }
}
void attn_res(const ArArgs& a, int nblocks, cudaStream_t s) {
    if (a.Q == 1) k_ar<1><<<nblocks, NT, 0, s>>>(a);
    else if (a.Q == 3) k_ar<3><<<nblocks, NT, 0, s>>>(a);
    else k_ar<4><<<nblocks, NT, 0, s>>>(a);
}

__global__ void __launch_bounds__(NT) k_norm(NormArgs a) {
    __shared__ float sm[NW * 32];
    const int t = threadIdx.x, s = a.T > 1 ? blockIdx.y : blockIdx.x, warp = t >> 5, tk = a.T > 1 ? blockIdx.x : 0;
    float v[3], ss = 0.f;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        v[j] = a.in[s][(long)tk * H + t + NT * j];
        ss += v[j] * v[j];
    }
    float r[1] = {ss};
    bred<1>(r, sm);
    const float f = rsqrtf(r[0] / H + 1e-6f);
    if (a.xh_out) {
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            const float xv = a.nw[s][t + NT * j] * (v[j] * f);
            a.xh_out[((long)s * a.T + tk) * H + t + NT * j] = __float2half(xv);
            if (a.xn_out) a.xn_out[((long)s * a.T + tk) * H + t + NT * j] = xv;
        }
        return;
    }
    const Q8Buf o = q8_stream(a.q8, s);
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const float xn = a.nw[s][t + NT * j] * (v[j] * f);
        a.xn_out[(long)s * H + t + NT * j] = xn;
        q8_store_chunk(xn, warp + NW * j, o);
    }
}
void norm_q8(const NormArgs& a, cudaStream_t s) { k_norm<<<a.T > 1 ? dim3(a.Tg ? a.Tg : a.T, a.S) : dim3(a.S), NT, 0, s>>>(a); }

// ------------------------------------------------------------------------------------------------------------------
// loop message gates: g = down @ [rms_u(h), rms_u(msg)] ; out = (1 - tanh(F g)) h + tanh(G g) msg
__global__ void __launch_bounds__(256) k_gate_down(GateArgs a) {
    __shared__ float xs[2 * H];
    __shared__ float red[32];
    const int s = blockIdx.y, tid = threadIdx.x;
    const float* h = a.h[s];
    const float* m = a.msg[s];
    float s1 = 0.f, s2 = 0.f;
    for (int i = tid; i < H; i += 256) {
        s1 += h[i] * h[i];
        s2 += m[i] * m[i];
    }
    const float r1 = rsqrtf(block_sum(s1, red) / H + 1e-6f);
    const float r2 = rsqrtf(block_sum(s2, red) / H + 1e-6f);
    for (int i = tid; i < H; i += 256) {
        xs[i] = h[i] * r1;
        xs[H + i] = m[i] * r2;
    }
    __syncthreads();
    const int warp = tid >> 5, lane = tid & 31, row = blockIdx.x * 8 + warp;
    const uint4* wp = reinterpret_cast<const uint4*>(a.down[s] + (size_t)row * 2 * H);
    float acc = 0.f;
    for (int i = lane; i < 2 * H / 8; i += 32) {
        const uint4 u = __ldg(wp + i);
        const __half2* hh = reinterpret_cast<const __half2*>(&u);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float2 f = __half22float2(hh[k]);
            acc = fmaf(f.x, xs[i * 8 + 2 * k], fmaf(f.y, xs[i * 8 + 2 * k + 1], acc));
        }
    }
    acc = warp_sum(acc);
    if (lane == 0) a.g[s * 64 + row] = acc;
}
__global__ void __launch_bounds__(256) k_gate_apply(GateArgs a) {
    __shared__ float g[64];
    const int s = blockIdx.y;
    if (threadIdx.x < 64) g[threadIdx.x] = a.g[s * 64 + threadIdx.x];
    __syncthreads();
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= H) return;
    const uint4* fp = reinterpret_cast<const uint4*>(a.forget[s] + (size_t)i * 64);
    const uint4* gp = reinterpret_cast<const uint4*>(a.gain[s] + (size_t)i * 64);
    float f = 0.f, gn = 0.f;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const uint4 u = __ldg(fp + k), w = __ldg(gp + k);
        const __half2* hu = reinterpret_cast<const __half2*>(&u);
        const __half2* hw = reinterpret_cast<const __half2*>(&w);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const float2 a1 = __half22float2(hu[e]), b1 = __half22float2(hw[e]);
            f = fmaf(a1.x, g[k * 8 + 2 * e], fmaf(a1.y, g[k * 8 + 2 * e + 1], f));
            gn = fmaf(b1.x, g[k * 8 + 2 * e], fmaf(b1.y, g[k * 8 + 2 * e + 1], gn));
        }
    }
    a.out[s][i] = (1.f - tanhf(f)) * a.h[s][i] + tanhf(gn) * a.msg[s][i];
}
void gate(const GateArgs& a, cudaStream_t s) {
    k_gate_down<<<dim3(8, a.S), 256, 0, s>>>(a);
    k_gate_apply<<<dim3((H + 255) / 256, a.S), 256, 0, s>>>(a);
}

// ------------------------------------------------------------------------------------------------------------------
// multi-matrix GEMV: 4 warps per block, R rows per warp, each warp owns whole rows (lanes loop over 32-element chunks)
template <int BITS, int R>
__global__ void __launch_bounds__(128) k_mgemv(GDs D) {
    int b = blockIdx.x, di = 0;
#pragma unroll 1
    while (di < D.n - 1 && b >= D.d[di].nblk) {
        b -= D.d[di].nblk;
        ++di;
    }
    const GD& g = D.d[di];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = (b * 4 + warp) * R;
    if (row0 >= g.N) return;
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.f;
    if (g.kind == 0) {
        const int C = g.K >> 5, GS = g.gshift - 5, G = g.K >> g.gshift;
        float rsc[R];
#pragma unroll
        for (int r = 0; r < R; ++r) rsc[r] = __half2float(g.rs[min(row0 + r, g.N - 1)]) / 255.0f;
        for (int c = lane; c < C; c += 32) {
            int xw[8], qs;
            float dx, xs;
            load_x(g.xq, c, xw, dx, xs, qs);
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const int row = row0 + r;
                if (row < g.N) {
                    float sc, mn;
                    sm8_decode(g.sm8[(size_t)row * G + (c >> GS)], rsc[r], g.zlo, g.zstep, sc, mn);
                    acc[r] = qchunk<BITS>(g.q + ((size_t)row * C + c) * BITS, xw, dx, xs, qs, sc, mn, acc[r]);
                }
            }
        }
    } else {
        const int K8 = g.K >> 3;
        for (int i = lane; i < K8; i += 32) {
            const float4 x0 = *reinterpret_cast<const float4*>(g.xf + i * 8), x1 = *reinterpret_cast<const float4*>(g.xf + i * 8 + 4);
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const int row = row0 + r;
                if (row < g.N) {
                    const uint4 u = __ldg(reinterpret_cast<const uint4*>(g.w16 + (size_t)row * g.K) + i);
                    const __half2* hh = reinterpret_cast<const __half2*>(&u);
                    const float2 a0 = __half22float2(hh[0]), a1 = __half22float2(hh[1]), a2 = __half22float2(hh[2]), a3 = __half22float2(hh[3]);
                    acc[r] = fmaf(a0.x, x0.x, fmaf(a0.y, x0.y, fmaf(a1.x, x0.z, fmaf(a1.y, x0.w, fmaf(a2.x, x1.x, fmaf(a2.y, x1.y, fmaf(a3.x, x1.z, fmaf(a3.y, x1.w, acc[r]))))))));
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const float v = warp_sum(acc[r]);
        if (lane == 0 && row0 + r < g.N) g.y[row0 + r] = v;
    }
}
void mgemv(GDs& g, int bits, cudaStream_t s) {
    constexpr int R = 2;
    int tot = 0;
    for (int i = 0; i < g.n; ++i) {
        g.d[i].nblk = (g.d[i].N + 4 * R - 1) / (4 * R);
        tot += g.d[i].nblk;
    }
    switch (bits) {
        case 3: k_mgemv<3, R><<<tot, 128, 0, s>>>(g); break;
        case 4: k_mgemv<4, R><<<tot, 128, 0, s>>>(g); break;
        case 5: k_mgemv<5, R><<<tot, 128, 0, s>>>(g); break;
        case 6: k_mgemv<6, R><<<tot, 128, 0, s>>>(g); break;
        case 8: k_mgemv<8, R><<<tot, 128, 0, s>>>(g); break;
        default: std::fprintf(stderr, "mgemv: unsupported bits %d\n", bits); std::abort();
    }
}

// ------------------------------------------------------------------------------------------------------------------
// attention prep: q/k/v/rel post-processing, short convs, KV write, relative bias table. grid = 8 kv heads, 128 threads
__global__ void __launch_bounds__(128) k_prep(PrepArgs a) {
    __shared__ float rv[48], rv0[48];
    __shared__ float red[32];
    const int g = blockIdx.x, c = threadIdx.x, ch = g * HD + c;
    const int pos = *a.pos;
    float tau = 1.f;
    if (a.glob) tau = 1.f + 0.1f * logf(fmaxf((float)(pos + 1) / 8192.f, 1.f));
    const int slot = a.ring ? (pos & (a.cap - 1)) : pos;
    if (blockIdx.y == 0) {
    // ---- k and v: short conv (state = 3 previous pre-conv inputs), then clip / value norm
    float kx = a.kraw[ch], vx = a.vraw[ch];
    float ko, vo;
    {
        const float k0 = a.kst[ch], k1 = a.kst[1024 + ch], k2 = a.kst[2048 + ch];
        const float v0 = a.vst[ch], v1 = a.vst[1024 + ch], v2 = a.vst[2048 + ch];
        if (a.glob) {
            // taps[l] = static_w[l][ch] + dw[g*4+l]; lag 0 = current
            const float t0 = a.kw[ch] + a.dwk[g * 4 + 0], t1 = a.kw[1024 + ch] + a.dwk[g * 4 + 1], t2 = a.kw[2048 + ch] + a.dwk[g * 4 + 2],
                        t3 = a.kw[3072 + ch] + a.dwk[g * 4 + 3];
            ko = (t0 * kx + t1 * k2 + t2 * k1 + t3 * k0 + kx) * (1.f / 3.f);
            const float u0 = a.vw[ch] + a.dwv[g * 4 + 0], u1 = a.vw[1024 + ch] + a.dwv[g * 4 + 1], u2 = a.vw[2048 + ch] + a.dwv[g * 4 + 2],
                        u3 = a.vw[3072 + ch] + a.dwv[g * 4 + 3];
            vo = (u0 * vx + u1 * v2 + u2 * v1 + u3 * v0 + vx) * (1.f / 3.f);
        } else {
            const float4 wk = *reinterpret_cast<const float4*>(a.kw + 4 * ch), wv = *reinterpret_cast<const float4*>(a.vw + 4 * ch);
            ko = (wk.x * k0 + wk.y * k1 + wk.z * k2 + wk.w * kx + kx) * 0.44721359549995794f;
            vo = (wv.x * v0 + wv.y * v1 + wv.z * v2 + wv.w * vx + vx) * 0.44721359549995794f;
        }
        a.kst[ch] = k1; a.kst[1024 + ch] = k2; a.kst[2048 + ch] = kx;
        a.vst[ch] = v1; a.vst[1024 + ch] = v2; a.vst[2048 + ch] = vx;
    }
    {
        const float nk = block_sum(ko * ko, red);
        const float fk = fminf(1.f, 22.627416997969522f / (sqrtf(nk) + 1e-6f));
        const float nv = block_sum(vo * vo, red);
        const float fv = rsqrtf(nv / HD + 1e-6f);
        if (a.kq) {
            const size_t r = (size_t)g * a.cap + slot;
            kv4_store(ko * fk, c, a.kq + r * 64, a.ks + r * 2, red);
            kv4_store(vo * fv, c, a.vq + r * 64, a.vs + r * 2, red);
        } else {
            a.kc[((size_t)g * a.cap + slot) * HD + c] = __float2half(ko * fk);
            a.vc[((size_t)g * a.cap + slot) * HD + c] = __float2half(vo * fv);
        }
    }
    // ---- q of the 3 heads sharing this kv head: scale, l2 clip, tau, softmax scale
    for (int hh = 0; hh < 3; ++hh) {
        const int h = g * 3 + hh;
        const float qv = a.qraw[h * HD + c] * a.qscale[h * HD + c];
        const float n2 = block_sum(qv * qv, red);
        const float f = fminf(1.f, 22.627416997969522f / (sqrtf(n2) + 1e-6f));
        a.qs[h * HD + c] = qv * f * tau * 0.08838834764831845f;  // 128^-0.5
    }
    }
    // ---- rel: scale, l2 clip (8.0), tau
    if (c < 48) rv0[c] = a.relraw[(g * 3 + c / 16) * 16 + (c & 15)] * a.relscale[(g * 3 + c / 16) * 16 + (c & 15)];
    __syncthreads();
    if (c < 48) {
        float n2 = 0.f;
        for (int k = 0; k < 16; ++k) n2 += rv0[(c / 16) * 16 + k] * rv0[(c / 16) * 16 + k];
        rv[c] = rv0[c] * fminf(1.f, 8.f / (sqrtf(n2) + 1e-6f)) * tau;
    }
    __syncthreads();
    {
        const int d = blockIdx.y * 128 + c;
#pragma unroll
        for (int hh = 0; hh < 3; ++hh) {
            float b = 0.f;
#pragma unroll
            for (int k = 0; k < 16; ++k) b = fmaf(rv[hh * 16 + k], a.relproj[k * REXT + d], b);
            a.rl[(g * 3 + hh) * REXT + d] = b;
        }
    }
}
void attn_prep(const PrepArgs& a, cudaStream_t s) { k_prep<<<dim3(NKV, 8), 128, 0, s>>>(a); }

// ------------------------------------------------------------------------------------------------------------------
// decode attention partials. grid (8 kv heads, nsplit); 128 threads = 4 warps x 4 groups of 8 lanes; a group owns one key at a time,
// lane `sub` holds 16 of the 128 dims. The 3 query heads of the kv head are processed together.
__global__ void __launch_bounds__(128) k_attn_part(AttnArgs a) {
    __shared__ float gacc[16][3][HD];
    __shared__ float gm[16][3], gl[16][3];
    const int g = blockIdx.x, sp = blockIdx.y, nsp = gridDim.y;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, grp = lane >> 3, sub = lane & 7;
    const int gi = warp * 4 + grp;
    const unsigned gmask = 0xFFu << (grp * 8);
    const int pos = *a.pos;
    const int lo = a.window > 0 ? max(0, pos - a.window + 1) : 0;
    const int n = pos - lo + 1;
    const int chunk = (n + nsp - 1) / nsp;
    const int kb = lo + sp * chunk, ke = min(pos + 1, kb + chunk);
    float q[3][16];
#pragma unroll
    for (int hh = 0; hh < 3; ++hh)
#pragma unroll
        for (int i = 0; i < 16; i += 4) {
            const float4 v = *reinterpret_cast<const float4*>(a.qs + (g * 3 + hh) * HD + sub * 16 + i);
            q[hh][i] = v.x; q[hh][i + 1] = v.y; q[hh][i + 2] = v.z; q[hh][i + 3] = v.w;
        }
    float m[3] = {-1e30f, -1e30f, -1e30f}, l[3] = {0.f, 0.f, 0.f}, acc[3][16];
#pragma unroll
    for (int hh = 0; hh < 3; ++hh)
#pragma unroll
        for (int i = 0; i < 16; ++i) acc[hh][i] = 0.f;
    for (int j = kb + gi; j < ke; j += 16) {
        const int slot = a.ring ? (j & (a.cap - 1)) : j;
        const size_t base = ((size_t)g * a.cap + slot) * HD + sub * 16;
        float kf[16], vf[16];
        if (a.kq) {
            const size_t r = (size_t)g * a.cap + slot;
            const uint2 kb8 = __ldg(reinterpret_cast<const uint2*>(a.kq + r * 64) + sub);
            const uint2 vb8 = __ldg(reinterpret_cast<const uint2*>(a.vq + r * 64) + sub);
            const float2 ksc = __half22float2(a.ks[r * 2 + (sub >> 2)]), vsc = __half22float2(a.vs[r * 2 + (sub >> 2)]);
            const uint32_t kw[2] = {kb8.x, kb8.y}, vw[2] = {vb8.x, vb8.y};
#pragma unroll
            for (int w = 0; w < 2; ++w)
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    kf[w * 8 + i] = fmaf((float)((kw[w] >> (4 * i)) & 15u), ksc.x, ksc.y);
                    vf[w * 8 + i] = fmaf((float)((vw[w] >> (4 * i)) & 15u), vsc.x, vsc.y);
                }
        } else {
        const uint4 k0 = __ldg(reinterpret_cast<const uint4*>(a.kc + base)), k1 = __ldg(reinterpret_cast<const uint4*>(a.kc + base) + 1);
        const uint4 v0 = __ldg(reinterpret_cast<const uint4*>(a.vc + base)), v1 = __ldg(reinterpret_cast<const uint4*>(a.vc + base) + 1);
        {
            const __half2* p0 = reinterpret_cast<const __half2*>(&k0);
            const __half2* p1 = reinterpret_cast<const __half2*>(&k1);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float2 f0 = __half22float2(p0[i]), f1 = __half22float2(p1[i]);
                kf[2 * i] = f0.x; kf[2 * i + 1] = f0.y; kf[8 + 2 * i] = f1.x; kf[8 + 2 * i + 1] = f1.y;
            }
            const __half2* q0 = reinterpret_cast<const __half2*>(&v0);
            const __half2* q1 = reinterpret_cast<const __half2*>(&v1);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float2 f0 = __half22float2(q0[i]), f1 = __half22float2(q1[i]);
                vf[2 * i] = f0.x; vf[2 * i + 1] = f0.y; vf[8 + 2 * i] = f1.x; vf[8 + 2 * i + 1] = f1.y;
            }
        }
        }
        float s[3];
#pragma unroll
        for (int hh = 0; hh < 3; ++hh) {
            float d = 0.f;
#pragma unroll
            for (int i = 0; i < 16; ++i) d = fmaf(q[hh][i], kf[i], d);
            d += __shfl_xor_sync(gmask, d, 1);
            d += __shfl_xor_sync(gmask, d, 2);
            d += __shfl_xor_sync(gmask, d, 4);
            const int dist = pos - j;
            s[hh] = d + (dist < REXT ? a.rl[(g * 3 + hh) * REXT + dist] : 0.f);
        }
#pragma unroll
        for (int hh = 0; hh < 3; ++hh) {
            const float mn = fmaxf(m[hh], s[hh]);
            const float corr = __expf(m[hh] - mn), p = __expf(s[hh] - mn);
            l[hh] = l[hh] * corr + p;
            m[hh] = mn;
#pragma unroll
            for (int i = 0; i < 16; ++i) acc[hh][i] = fmaf(p, vf[i], acc[hh][i] * corr);
        }
    }
#pragma unroll
    for (int hh = 0; hh < 3; ++hh) {
#pragma unroll
        for (int i = 0; i < 16; ++i) gacc[gi][hh][sub * 16 + i] = acc[hh][i];
        if (sub == 0) { gm[gi][hh] = m[hh]; gl[gi][hh] = l[hh]; }
    }
    __syncthreads();
    for (int hh = 0; hh < 3; ++hh) {
        float M = -1e30f;
        for (int k = 0; k < 16; ++k) M = fmaxf(M, gm[k][hh]);
        float L = 0.f, o = 0.f;
        for (int k = 0; k < 16; ++k) {
            const float e = gm[k][hh] > -1e29f ? __expf(gm[k][hh] - M) : 0.f;
            L += e * gl[k][hh];
            o += e * gacc[k][hh][tid];
        }
        float* out = a.part + ((size_t)(g * nsp + sp) * 3 + hh) * PART_STRIDE;
        out[2 + tid] = o;
        if (tid == 0) { out[0] = M; out[1] = L; }
    }
}
void attn_part(const AttnArgs& a, cudaStream_t s) { k_attn_part<<<dim3(NKV, a.nsplit), 128, 0, s>>>(a); }

// merge split partials + sink logit (value 0), multiply by sigmoid(gate), quantize to int8 for wo. grid 24 heads, 128 threads
__global__ void __launch_bounds__(128) k_attn_combine(AttnArgs a) {
    const int h = blockIdx.x, g = h / 3, hh = h % 3, d = threadIdx.x, nsp = a.nsplit;
    const float sink = a.sink[h];
    float M = sink;
    for (int s = 0; s < nsp; ++s) {
        const float* p = a.part + ((size_t)(g * nsp + s) * 3 + hh) * PART_STRIDE;
        if (p[1] > 0.f) M = fmaxf(M, p[0]);
    }
    float L = __expf(sink - M), o = 0.f;
    for (int s = 0; s < nsp; ++s) {
        const float* p = a.part + ((size_t)(g * nsp + s) * 3 + hh) * PART_STRIDE;
        if (p[1] > 0.f) {
            const float e = __expf(p[0] - M);
            L += e * p[1];
            o += e * p[2 + d];
        }
    }
    const float gt = a.gate[h * HD + d];
    const float v = o / L * (1.f / (1.f + __expf(-gt)));
    q8_store_chunk(v, h * 4 + (d >> 5), a.out);
}
void attn_combine(const AttnArgs& a, cudaStream_t s) { k_attn_combine<<<NH, 128, 0, s>>>(a); }

// ------------------------------------------------------------------------------------------------------------------
// router
__global__ void __launch_bounds__(64) k_router(const __half* W, const float* bias, const float* x, float* logits, int* ids, float* wts,
                                                int* ticket, float route_scale) {
    __shared__ int s_last;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, row = blockIdx.x * 2 + warp;
    float acc = 0.f;
    for (int i = lane; i < H / 8; i += 32) {
        const float4 x0 = *reinterpret_cast<const float4*>(x + i * 8), x1 = *reinterpret_cast<const float4*>(x + i * 8 + 4);
        const uint4 u = __ldg(reinterpret_cast<const uint4*>(W + (size_t)row * H) + i);
        const __half2* hh = reinterpret_cast<const __half2*>(&u);
        const float2 a0 = __half22float2(hh[0]), a1 = __half22float2(hh[1]), a2 = __half22float2(hh[2]), a3 = __half22float2(hh[3]);
        acc = fmaf(a0.x, x0.x, fmaf(a0.y, x0.y, fmaf(a1.x, x0.z, fmaf(a1.y, x0.w, fmaf(a2.x, x1.x, fmaf(a2.y, x1.y, fmaf(a3.x, x1.z, fmaf(a3.y, x1.w, acc))))))));
    }
    acc = warp_sum(acc);
    if (lane == 0) logits[row] = acc;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) s_last = (atomicAdd(ticket, 1) == (int)gridDim.x - 1);
    __syncthreads();
    if (!s_last || warp != 0) return;
    __threadfence();
    float sv[4], sb[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const float l = __ldcg(logits + lane + 32 * k);
        sv[k] = 1.f / (1.f + __expf(-l));
        sb[k] = sv[k] + bias[lane + 32 * k];
    }
    float chosen_w = 0.f;
    int chosen_i = 0;
    float wsum = 0.f;
    float wsel[TOPK];
    for (int r = 0; r < TOPK; ++r) {
        float best = -1e30f;
        int bi = 1 << 30, bk = 0;
#pragma unroll
        for (int k = 0; k < 4; ++k)
            if (sb[k] > best) { best = sb[k]; bi = lane + 32 * k; bk = k; }
        float ob = best;
        int oi = bi;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float vb = __shfl_xor_sync(0xffffffffu, ob, o);
            const int vi = __shfl_xor_sync(0xffffffffu, oi, o);
            if (vb > ob || (vb == ob && vi < oi)) { ob = vb; oi = vi; }
        }
        // owner lane = oi & 31, k = oi >> 5
        const int kk = oi >> 5;
        float w = 0.f;
        if (lane == (oi & 31)) {
#pragma unroll
            for (int k = 0; k < 4; ++k)
                if (k == kk) { w = sv[k]; sb[k] = -1e30f; }
        }
        w = __shfl_sync(0xffffffffu, w, oi & 31);
        wsel[r] = w;
        wsum += w;
        if (lane == r) { chosen_i = oi; }
        (void)bk;
    }
    if (lane < TOPK) {
        float w = 0.f;
#pragma unroll
        for (int r = 0; r < TOPK; ++r)
            if (r == lane) w = wsel[r];
        ids[lane] = chosen_i;
        wts[lane] = w / (wsum + 1e-20f) * route_scale;
    }
    if (lane == 0) *ticket = 0;
    (void)chosen_w;
}
void router(const __half* W, const float* bias, const float* x, float* logits, int* ids, float* wts, int* ticket, float route_scale, cudaStream_t s) {
    k_router<<<NE / 2, 64, 0, s>>>(W, bias, x, logits, ids, wts, ticket, route_scale);
}

// grouped expert GEMV 1: rows of w13 are interleaved (even = gate, odd = up). block = 64 rows = 32 activations, 16 warps x 4 rows.
template <int BITS>
__global__ void __launch_bounds__(512) k_w13(ExpertW W, Q8Vec x, const int* ids, Q8Buf act, float gcap, float ucap) {
    __shared__ float actv[32];
    const int rb = blockIdx.x, slot = blockIdx.y, e = ids[slot];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = rb * 64 + warp * 4;
    const size_t C = W.K >> 5, G = W.K >> W.gshift;
    const int GS = W.gshift - 5;
    const uint32_t* qb = W.q + (size_t)e * W.N * C * BITS;
    const uint16_t* smb = W.sm8 + (size_t)e * W.N * G;
    const __half* rsb = W.rs + (size_t)e * W.N;
    float acc[4] = {0.f, 0.f, 0.f, 0.f}, rsc[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) rsc[r] = __half2float(rsb[row0 + r]) / 255.0f;
    for (size_t c = lane; c < C; c += 32) {
        int xw[8], qs;
        float dx, xs;
        load_x(x, c, xw, dx, xs, qs);
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            float sc, mn;
            sm8_decode(smb[(size_t)(row0 + r) * G + (c >> GS)], rsc[r], W.zlo, W.zstep, sc, mn);
            acc[r] = qchunk<BITS>(qb + ((size_t)(row0 + r) * C + c) * BITS, xw, dx, xs, qs, sc, mn, acc[r]);
        }
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) acc[r] = warp_sum(acc[r]);
    if (lane == 0) {
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            const float gv = acc[2 * p], uv = acc[2 * p + 1];
            const float gc = gcap * tanhf(gv / gcap), uc = ucap * tanhf(uv / ucap);
            actv[warp * 2 + p] = gc * (1.f / (1.f + __expf(-gv))) * uc;
        }
    }
    __syncthreads();
    if (warp == 0) q8_store_chunk(actv[lane], slot * (W.N / 64) + rb, act);
}
void moe_w13(const ExpertW& W, const Q8Vec& x, const int* ids, Q8Buf act, float gcap, float ucap, cudaStream_t s) {
    const dim3 grid(W.N / 64, TOPK);
    switch (W.bits) {
        case 2: break;
        case 3: k_w13<3><<<grid, 512, 0, s>>>(W, x, ids, act, gcap, ucap); return;
        case 4: k_w13<4><<<grid, 512, 0, s>>>(W, x, ids, act, gcap, ucap); return;
        case 5: k_w13<5><<<grid, 512, 0, s>>>(W, x, ids, act, gcap, ucap); return;
        default: break;
    }
    std::fprintf(stderr, "moe_w13: unsupported bits %d\n", W.bits);
    std::abort();
}

// grouped expert GEMV 2: block = R rows of w2 for all 8 expert slots (warp = slot); the 8 slots are summed in slot order.
template <int BITS, int R>
__global__ void __launch_bounds__(256) k_w2(ExpertW W, Q8Vec act, const int* ids, const float* wts, float* y) {
    __shared__ float ys[TOPK][R];
    const int slot = threadIdx.x >> 5, lane = threadIdx.x & 31, row0 = blockIdx.x * R, e = ids[slot];
    const size_t C = W.K >> 5, G = W.K >> W.gshift;
    const int GS = W.gshift - 5;
    const uint32_t* qb = W.q + (size_t)e * W.N * C * BITS;
    const uint16_t* smb = W.sm8 + (size_t)e * W.N * G;
    const __half* rsb = W.rs + (size_t)e * W.N;
    float acc[R], rsc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) { acc[r] = 0.f; rsc[r] = __half2float(rsb[row0 + r]) / 255.0f; }
    for (size_t c = lane; c < C; c += 32) {
        int xw[8], qs;
        float dx, xs;
        load_x(act, slot * C + c, xw, dx, xs, qs);
#pragma unroll
        for (int r = 0; r < R; ++r) {
            float sc, mn;
            sm8_decode(smb[(size_t)(row0 + r) * G + (c >> GS)], rsc[r], W.zlo, W.zstep, sc, mn);
            acc[r] = qchunk<BITS>(qb + ((size_t)(row0 + r) * C + c) * BITS, xw, dx, xs, qs, sc, mn, acc[r]);
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const float v = warp_sum(acc[r]);
        if (lane == 0) ys[slot][r] = v;
    }
    __syncthreads();
    if (threadIdx.x < R) {
        float o = 0.f;
#pragma unroll
        for (int k = 0; k < TOPK; ++k) o = fmaf(wts[k], ys[k][threadIdx.x], o);
        y[row0 + threadIdx.x] = o;
    }
}
void moe_w2(const ExpertW& W, const Q8Vec& act, const int* ids, const float* wts, float* y, cudaStream_t s) {
    constexpr int R = 4;
    const int grid = W.N / R;
    switch (W.bits) {
        case 3: k_w2<3, R><<<grid, 256, 0, s>>>(W, act, ids, wts, y); return;
        case 4: k_w2<4, R><<<grid, 256, 0, s>>>(W, act, ids, wts, y); return;
        case 5: k_w2<5, R><<<grid, 256, 0, s>>>(W, act, ids, wts, y); return;
        default: break;
    }
    std::fprintf(stderr, "moe_w2: unsupported bits %d\n", W.bits);
    std::abort();
}

__global__ void __launch_bounds__(NT) k_finish(const float* res, const float* y, const float* cw, float* cst, float* h_out, int ntok) {
    const long tok = blockIdx.x;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int c = threadIdx.x + NT * j;
        const float x = y[tok * H + c];
        const long i3 = tok - 3, i2 = tok - 2, i1 = tok - 1;
        const float s0 = i3 >= 0 ? y[i3 * H + c] : cst[(3 + i3) * H + c];
        const float s1 = i2 >= 0 ? y[i2 * H + c] : cst[(3 + i2) * H + c];
        const float s2 = i1 >= 0 ? y[i1 * H + c] : cst[(3 + i1) * H + c];
        const float4 w = *reinterpret_cast<const float4*>(cw + 4 * c);
        const float yy = w.x * s0 + w.y * s1 + w.z * s2 + w.w * x;
        if (ntok == 1) {
            cst[c] = s1;
            cst[H + c] = s2;
            cst[2 * H + c] = x;
        }
        h_out[tok * H + c] = res[tok * H + c] + (yy + x) * 0.44721359549995794f;
    }
}
void finish(const float* res, const float* y, const float* cw, float* cst, float* h_out, cudaStream_t s, int T) {
    k_finish<<<T, NT, 0, s>>>(res, y, cw, cst, h_out, T);
}

// causal-conv state update after a chunk of T rows (row stride `rs` floats, D channels): state = last 3 of [state, rows]
__global__ void k_conv_commit(const float* rows, long rs, int T, int D, float* st) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= D) return;
    float old[3] = {st[c], st[D + c], st[2 * D + c]}, nw[3];
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        const int idx = T + i;
        nw[i] = idx < 3 ? old[idx] : rows[(long)(idx - 3) * rs + c];
    }
    st[c] = nw[0]; st[D + c] = nw[1]; st[2 * D + c] = nw[2];
}
void conv_commit(const float* rows, long rs, int T, int D, float* st, cudaStream_t s) {
    k_conv_commit<<<(D + 255) / 256, 256, 0, s>>>(rows, rs, T, D, st);
}

}  // namespace qv::cuda::ink
