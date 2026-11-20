#include "linear_prefill.h"

#include <algorithm>

#include "dequant.h"

namespace qv::cuda {

#define QV_CUBLAS_CHECK(x)                                                                          \
    do {                                                                                            \
        cublasStatus_t st_ = (x);                                                                   \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                                         \
            std::fprintf(stderr, "cuBLAS error %d at %s:%d (%s)\n", (int)st_, __FILE__, __LINE__, #x); \
            std::abort();                                                                           \
        }                                                                                           \
    } while (0)

void LinearPrefill::init(cudaStream_t s, size_t slab_bytes) {
    QV_CUBLAS_CHECK(cublasCreate(&cublas));
    QV_CUBLAS_CHECK(cublasSetStream(cublas, s));
    QV_CUBLAS_CHECK(cublasSetMathMode(cublas, CUBLAS_TENSOR_OP_MATH));
    slab_elems = slab_bytes / sizeof(__half);
    QV_CUDA_CHECK(cudaMalloc(&slab, slab_bytes));
}

void LinearPrefill::destroy() {
    if (cublas) cublasDestroy(cublas);
    if (slab) cudaFree(slab);
    cublas = nullptr;
    slab = nullptr;
}

// Row-major C[T][nr] (ld = ldc, written at a column offset of Y) = A[T][K] * B[nr][K]^T.
// In cuBLAS' column-major view: C^T[nr][T] = op_T(B_cm[K][nr]) * A_cm[K][T].
static void gemm_block(cublasHandle_t h, const __half* B, int nr, const __half* A, int T, int K, float* C, int ldc) {
    const float alpha = 1.f, beta = 0.f;
    QV_CUBLAS_CHECK(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, nr, T, K, &alpha, B, CUDA_R_16F, K, A, CUDA_R_16F, K,
                                 &beta, C, CUDA_R_32F, ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

void linear_seq(LinearPrefill& lp, const QMat& W, const __half* X, int T, float* Y, cudaStream_t s) {
    int rows_per = (int)std::min<size_t>(W.N, lp.slab_elems / W.K);
    if (rows_per < W.N && rows_per >= 256) rows_per &= ~255;  // aligned column blocks (cuBLAS picks wide-alignment kernels)
    if (rows_per <= 0) {
        std::fprintf(stderr, "linear_seq: slab too small for K=%d\n", W.K);
        std::abort();
    }
    for (int r0 = 0; r0 < W.N; r0 += rows_per) {
        const int nr = std::min(rows_per, W.N - r0);
        dequant_rows_f16(W, r0, nr, lp.slab, s);
        gemm_block(lp.cublas, lp.slab, nr, X, T, W.K, Y + r0, W.N);
    }
}

void linear_seq_dense(LinearPrefill& lp, const DMat16& W, const __half* X, int T, float* Y, cudaStream_t) {
    gemm_block(lp.cublas, W.w, W.N, X, T, W.K, Y, W.N);
}

}  // namespace qv::cuda
