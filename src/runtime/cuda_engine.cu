// CUDA backend: decode-step executor for Qwen3.5-family hybrid (GDN + gated attention) models.
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "../backends/cuda/attention/attn.h"
#include "../backends/cuda/common.cuh"
#include "../backends/cuda/kernels/elementwise.h"
#include "../backends/cuda/kernels/gemv.h"
#include "../backends/cuda/recurrent/gdn.h"
#include "qv/engine.h"

namespace qv {
using namespace qv::cuda;

namespace {

struct GdnLayer {
    QMat qkv, z, out;
    DMat16 a, b;
    const float *A_neg, *dt_bias, *conv_w, *norm_w;
};
struct AttnLayer {
    QMat q, k, v, o;
    const float *q_norm, *k_norm;
};
struct Layer {
    bool attn;
    int idx_kind;  // index into gdn/attn state arrays
    const float *in_norm, *post_norm;
    QMat gate, up, down;
    GdnLayer g;
    AttnLayer at;
};

class CudaEngine final : public Engine {
   public:
    CudaEngine(const ModelFile& mf, const EngineOptions& opt) : opt_(opt) {
        cfg_ = mf.cfg;
        QV_CUDA_CHECK(cudaStreamCreate(&stream_));
        load_weights(mf);
        bind(mf);
        alloc_state();
        reset();
    }
    ~CudaEngine() override {
        for (void* p : allocs_) cudaFree(p);
        cudaFree(weights_);
        cudaStreamDestroy(stream_);
    }

    void step(int token, int pos) override {
        if (pos >= opt_.max_ctx) throw std::runtime_error("context overflow");
        const Config& c = cfg_;
        cudaStream_t s = stream_;
        QV_CUDA_CHECK(cudaMemcpyAsync(d_tok_, &token, sizeof(int), cudaMemcpyHostToDevice, s));
        embed(embed_, d_tok_, x_, s);
        if (opt_.dump_hidden) dump();
        for (const Layer& L : layers_) {
            rmsnorm(x_, L.in_norm, xn_, xs_, c.hidden, c.eps, s);
            if (!L.attn) {
                const GdnLayer& g = L.g;
                gemv_q(g.qkv, xn_, xs_, qkv_, nullptr, s);
                gemv_q(g.z, xn_, xs_, z_, nullptr, s);
                gemv_f16(g.a, xn_, alpha_, s);
                gemv_f16(g.b, xn_, beta_, s);
                const int C = g.qkv.N;
                conv_step(qkv_, g.conv_w, conv_state_ + (size_t)L.idx_kind * C * 3, conv_out_, C, s);
                GdnStepArgs a{conv_out_, alpha_, beta_, g.A_neg, g.dt_bias, z_, g.norm_w,
                              gdn_state_ + (size_t)L.idx_kind * c.lin_v_heads * c.lin_k_dim * c.lin_v_dim, mix_,
                              c.lin_k_heads, c.lin_v_heads, c.lin_k_dim, c.lin_v_dim, c.eps};
                gdn_step(a, s);
                xsum32(mix_, xs_, g.out.K, s);
                gemv_q(g.out, mix_, xs_, x_, x_, s);  // residual add in place
            } else {
                const AttnLayer& at = L.at;
                gemv_q(at.q, xn_, xs_, qg_, nullptr, s);
                gemv_q(at.k, xn_, xs_, kb_, nullptr, s);
                gemv_q(at.v, xn_, xs_, vb_, nullptr, s);
                const size_t kv_layer = (size_t)c.n_kv_heads * opt_.max_ctx * c.head_dim;
                float* kc = k_cache_ + L.idx_kind * kv_layer;
                float* vc = v_cache_ + L.idx_kind * kv_layer;
                AttnPrepArgs p{qg_, kb_, vb_, at.q_norm, at.k_norm, inv_freq_, qr_, kc, vc,
                               c.n_heads, c.n_kv_heads, c.head_dim, c.rope_dim, opt_.max_ctx, pos, c.eps};
                attn_prep(p, s);
                AttnDecodeArgs d{qr_, qg_, kc, vc, scores_, mix_, c.n_heads, c.n_kv_heads, opt_.max_ctx, pos};
                attn_decode(d, s);
                xsum32(mix_, xs_, at.o.K, s);
                gemv_q(at.o, mix_, xs_, x_, x_, s);
            }
            rmsnorm(x_, L.post_norm, xn_, xs_, c.hidden, c.eps, s);
            gemv_q(L.gate, xn_, xs_, g_, nullptr, s);
            gemv_q(L.up, xn_, xs_, u_, nullptr, s);
            silu_mul(g_, u_, act_, xs_, c.inter, s);
            gemv_q(L.down, act_, xs_, x_, x_, s);
            if (opt_.dump_hidden) dump();
        }
        rmsnorm(x_, final_norm_, xn_, xs_, c.hidden, c.eps, s);
        gemv_q(head_, xn_, xs_, logits_, nullptr, s);
    }

    int argmax() override {
        qv::cuda::argmax(logits_, cfg_.vocab, d_arg_, stream_);
        int r;
        QV_CUDA_CHECK(cudaMemcpyAsync(&r, d_arg_, sizeof(int), cudaMemcpyDeviceToHost, stream_));
        QV_CUDA_CHECK(cudaStreamSynchronize(stream_));
        return r;
    }
    void logits(std::vector<float>& out) override {
        out.resize(cfg_.vocab);
        QV_CUDA_CHECK(cudaMemcpyAsync(out.data(), logits_, cfg_.vocab * 4, cudaMemcpyDeviceToHost, stream_));
        QV_CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
    void reset() override {
        QV_CUDA_CHECK(cudaMemsetAsync(conv_state_, 0, conv_bytes_, stream_));
        QV_CUDA_CHECK(cudaMemsetAsync(gdn_state_, 0, gdn_bytes_, stream_));
        hidden_.clear();
        QV_CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
    const std::vector<float>& hidden_dump() const override { return hidden_; }
    void sync() override { QV_CUDA_CHECK(cudaStreamSynchronize(stream_)); }
    size_t device_bytes_weights() const override { return weights_bytes_; }
    size_t device_bytes_state() const override { return state_bytes_; }

   private:
    EngineOptions opt_;
    cudaStream_t stream_{};
    char* weights_ = nullptr;
    size_t weights_bytes_ = 0, state_bytes_ = 0;
    std::vector<void*> allocs_;
    std::vector<Layer> layers_;
    QMat embed_, head_;
    const float* final_norm_ = nullptr;
    // state
    float *conv_state_ = nullptr, *gdn_state_ = nullptr, *k_cache_ = nullptr, *v_cache_ = nullptr;
    size_t conv_bytes_ = 0, gdn_bytes_ = 0;
    // scratch
    float *x_, *xn_, *xs_, *qkv_, *z_, *alpha_, *beta_, *conv_out_, *mix_, *qg_, *kb_, *vb_, *qr_, *scores_, *g_, *u_,
        *act_, *logits_, *inv_freq_;
    int *d_tok_, *d_arg_;
    std::vector<float> hidden_;

    template <class T>
    T* dalloc(size_t n) {
        void* p;
        QV_CUDA_CHECK(cudaMalloc(&p, n * sizeof(T)));
        QV_CUDA_CHECK(cudaMemset(p, 0, n * sizeof(T)));
        allocs_.push_back(p);
        state_bytes_ += n * sizeof(T);
        return (T*)p;
    }

    void dump() {
        size_t o = hidden_.size();
        hidden_.resize(o + cfg_.hidden);
        QV_CUDA_CHECK(cudaMemcpyAsync(hidden_.data() + o, x_, cfg_.hidden * 4, cudaMemcpyDeviceToHost, stream_));
        QV_CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    void load_weights(const ModelFile& mf) {
        weights_bytes_ = mf.bin_size;
        QV_CUDA_CHECK(cudaMalloc(&weights_, weights_bytes_));
        FILE* f = std::fopen(mf.bin_path.c_str(), "rb");
        if (!f) throw std::runtime_error("cannot open " + mf.bin_path);
        const size_t CH = 256u << 20;
        void* pinned;
        QV_CUDA_CHECK(cudaMallocHost(&pinned, CH));
        size_t off = 0;
        while (off < weights_bytes_) {
            size_t n = std::fread(pinned, 1, std::min(CH, weights_bytes_ - off), f);
            if (n == 0) throw std::runtime_error("short read on weights");
            QV_CUDA_CHECK(cudaMemcpy(weights_ + off, pinned, n, cudaMemcpyHostToDevice));
            off += n;
        }
        cudaFreeHost(pinned);
        std::fclose(f);
    }

    QMat qmat(const ModelFile& mf, const std::string& n) {
        const TensorInfo& t = mf.at(n);
        if (t.kind != Kind::Quant) throw std::runtime_error(n + " is not quantized");
        if (t.group != 64) throw std::runtime_error("group 64 only");
        QMat m;
        m.q = (const uint32_t*)(weights_ + t.off_q);
        m.sm = (const __half2*)(weights_ + t.off_sm);
        m.N = (int)t.shape[0];
        m.K = (int)t.shape[1];
        m.bits = t.bits;
        return m;
    }
    const float* f32(const ModelFile& mf, const std::string& n) {
        const TensorInfo& t = mf.at(n);
        if (t.kind != Kind::F32) throw std::runtime_error(n + " is not f32");
        return (const float*)(weights_ + t.off);
    }
    DMat16 f16(const ModelFile& mf, const std::string& n) {
        const TensorInfo& t = mf.at(n);
        if (t.kind != Kind::F16) throw std::runtime_error(n + " is not f16");
        return DMat16{(const __half*)(weights_ + t.off), (int)t.shape[0], (int)t.shape[1]};
    }

    void bind(const ModelFile& mf) {
        const std::string LM = "model.language_model.";
        embed_ = qmat(mf, LM + "embed_tokens.weight");
        head_ = qmat(mf, "lm_head.weight");
        final_norm_ = f32(mf, LM + "norm.weight");
        int ng = 0, na = 0;
        for (int i = 0; i < cfg_.n_layers; ++i) {
            std::string p = LM + "layers." + std::to_string(i) + ".";
            Layer L{};
            L.attn = cfg_.is_attn[i];
            L.idx_kind = L.attn ? na++ : ng++;
            L.in_norm = f32(mf, p + "input_layernorm.weight");
            L.post_norm = f32(mf, p + "post_attention_layernorm.weight");
            L.gate = qmat(mf, p + "mlp.gate_proj.weight");
            L.up = qmat(mf, p + "mlp.up_proj.weight");
            L.down = qmat(mf, p + "mlp.down_proj.weight");
            if (!L.attn) {
                std::string q = p + "linear_attn.";
                L.g.qkv = qmat(mf, q + "in_proj_qkv.weight");
                L.g.z = qmat(mf, q + "in_proj_z.weight");
                L.g.out = qmat(mf, q + "out_proj.weight");
                L.g.a = f16(mf, q + "in_proj_a.weight");
                L.g.b = f16(mf, q + "in_proj_b.weight");
                L.g.A_neg = f32(mf, q + "A_log");
                L.g.dt_bias = f32(mf, q + "dt_bias");
                L.g.conv_w = f32(mf, q + "conv1d.weight");
                L.g.norm_w = f32(mf, q + "norm.weight");
            } else {
                std::string q = p + "self_attn.";
                L.at.q = qmat(mf, q + "q_proj.weight");
                L.at.k = qmat(mf, q + "k_proj.weight");
                L.at.v = qmat(mf, q + "v_proj.weight");
                L.at.o = qmat(mf, q + "o_proj.weight");
                L.at.q_norm = f32(mf, q + "q_norm.weight");
                L.at.k_norm = f32(mf, q + "k_norm.weight");
            }
            layers_.push_back(L);
        }
        n_gdn_ = ng;
        n_attn_ = na;
    }
    int n_gdn_ = 0, n_attn_ = 0;

    void alloc_state() {
        const Config& c = cfg_;
        const int conv_C = 2 * c.lin_k_heads * c.lin_k_dim + c.lin_v_heads * c.lin_v_dim;
        conv_bytes_ = (size_t)n_gdn_ * conv_C * 3 * 4;
        gdn_bytes_ = (size_t)n_gdn_ * c.lin_v_heads * c.lin_k_dim * c.lin_v_dim * 4;
        conv_state_ = dalloc<float>(conv_bytes_ / 4);
        gdn_state_ = dalloc<float>(gdn_bytes_ / 4);
        size_t kv = (size_t)n_attn_ * c.n_kv_heads * opt_.max_ctx * c.head_dim;
        k_cache_ = dalloc<float>(kv);
        v_cache_ = dalloc<float>(kv);
        x_ = dalloc<float>(c.hidden);
        xn_ = dalloc<float>(c.hidden);
        xs_ = dalloc<float>(std::max(c.inter, 2 * c.n_heads * c.head_dim) / 32 + 32);
        qkv_ = dalloc<float>(conv_C);
        conv_out_ = dalloc<float>(conv_C);
        z_ = dalloc<float>(c.lin_v_heads * c.lin_v_dim);
        alpha_ = dalloc<float>(c.lin_v_heads);
        beta_ = dalloc<float>(c.lin_v_heads);
        mix_ = dalloc<float>(std::max(c.lin_v_heads * c.lin_v_dim, c.n_heads * c.head_dim));
        qg_ = dalloc<float>(2 * c.n_heads * c.head_dim);
        kb_ = dalloc<float>(c.n_kv_heads * c.head_dim);
        vb_ = dalloc<float>(c.n_kv_heads * c.head_dim);
        qr_ = dalloc<float>(c.n_heads * c.head_dim);
        scores_ = dalloc<float>((size_t)c.n_heads * opt_.max_ctx);
        g_ = dalloc<float>(c.inter);
        u_ = dalloc<float>(c.inter);
        act_ = dalloc<float>(c.inter);
        logits_ = dalloc<float>(c.vocab);
        d_tok_ = dalloc<int>(1);
        d_arg_ = dalloc<int>(1);
        // inv_freq computed in fp32 exactly like torch: 1 / base ** (arange(0, dim, 2) / dim)
        std::vector<float> inv(c.rope_dim / 2);
        for (int i = 0; i < c.rope_dim / 2; ++i)
            inv[i] = 1.0f / std::pow((float)c.rope_theta, (float)(2 * i) / (float)c.rope_dim);
        inv_freq_ = dalloc<float>(inv.size());
        QV_CUDA_CHECK(cudaMemcpy(inv_freq_, inv.data(), inv.size() * 4, cudaMemcpyHostToDevice));
    }
};

}  // namespace

std::unique_ptr<Engine> make_cuda_engine(const ModelFile& mf, const EngineOptions& opt) {
    return std::make_unique<CudaEngine>(mf, opt);
}

}  // namespace qv
