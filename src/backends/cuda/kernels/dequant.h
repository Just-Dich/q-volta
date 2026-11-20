#pragma once
#include "../common.cuh"

namespace qv::cuda {
// Dequantize rows [row0, row0+nrows) of W to fp16, row-major out[nrows][K].
// out[r][k] = __float2half_rn(fp32(scale)*q + fp32(min)). Requires K % 64 == 0, bits in {2,3,4,5,6,8}.
// out must be 16-byte aligned.
void dequant_rows_f16(const QMat& W, int row0, int nrows, __half* out, cudaStream_t s);
}  // namespace qv::cuda
