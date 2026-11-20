// C++ re-implementation of tools/qv_format.py pack()/unpack() (test + bench helper, not used by the kernels).
#pragma once
#include <cstdint>
#include <vector>

namespace qv::cpu {
struct PlaneSpec { int w, shift; };
inline int plane_list(int bits, PlaneSpec* p) {
    switch (bits) {
        case 2: p[0] = {2, 0}; return 1;
        case 3: p[0] = {2, 0}; p[1] = {1, 2}; return 2;
        case 4: p[0] = {4, 0}; return 1;
        case 5: p[0] = {4, 0}; p[1] = {1, 4}; return 2;
        case 6: p[0] = {4, 0}; p[1] = {2, 4}; return 2;
        case 8: p[0] = {4, 0}; p[1] = {4, 4}; return 2;
    }
    return 0;
}
// q: uint8 [N*K] -> words uint32 [N][K/32][bits]
inline std::vector<uint32_t> pack_q(const uint8_t* q, int N, int K, int bits) {
    PlaneSpec ps[2];
    const int np = plane_list(bits, ps);
    std::vector<uint32_t> out((size_t)N * (K / 32) * bits);
    for (int n = 0; n < N; ++n)
        for (int c = 0; c < K / 32; ++c) {
            uint32_t* o = &out[((size_t)n * (K / 32) + c) * bits];
            const uint8_t* qc = q + (size_t)n * K + c * 32;
            int wi = 0;
            for (int p = 0; p < np; ++p) {
                const int w = ps[p].w, nf = 32 / w, mask = (1 << w) - 1;
                for (int j = 0; j < w; ++j) {
                    uint32_t word = 0;
                    for (int f = 0; f < nf; ++f)
                        word |= (uint32_t)((qc[j + w * f] >> ps[p].shift) & mask) << (w * f);
                    o[wi++] = word;
                }
            }
        }
    return out;
}
inline void unpack_q(const uint32_t* words, int N, int K, int bits, uint8_t* q) {
    PlaneSpec ps[2];
    const int np = plane_list(bits, ps);
    for (int n = 0; n < N; ++n)
        for (int c = 0; c < K / 32; ++c) {
            const uint32_t* o = words + ((size_t)n * (K / 32) + c) * bits;
            uint8_t* qc = q + (size_t)n * K + c * 32;
            for (int i = 0; i < 32; ++i) qc[i] = 0;
            int wi = 0;
            for (int p = 0; p < np; ++p) {
                const int w = ps[p].w, nf = 32 / w, mask = (1 << w) - 1;
                for (int j = 0; j < w; ++j, ++wi)
                    for (int f = 0; f < nf; ++f) qc[j + w * f] |= (uint8_t)(((o[wi] >> (w * f)) & mask) << ps[p].shift);
            }
        }
}
}  // namespace qv::cpu
