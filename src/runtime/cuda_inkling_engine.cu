// CUDA backend: Inkling-2-Mini (inkling2_loop) engine. Decode step = one CUDA graph (no host sync, expert ids stay on the device).
// Spec: docs/INKLING_SPEC.md. Design/measurements: docs/INKLING_ENGINE.md.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "../backends/cuda/common.cuh"
#include "../backends/cuda/inkling/ink_ops.h"
#include "../backends/cuda/kernels/elementwise.h"
#include "../backends/cuda/kernels/gemv.h"
#include "qv/engine.h"

namespace qv {
using namespace qv::cuda;
namespace ik = qv::cuda::ink;

namespace {
static bool g_trace = std::getenv("QV_INK_TRACE") != nullptr;
#define STAGE(name) do { if (g_trace) { cudaError_t e_ = cudaStreamSynchronize(stream_); std::fprintf(stderr, "app %d L%d p%d %s -> %s\n", A.idx, A.L, A.p, name, cudaGetErrorString(e_)); } } while (0)

struct Ctrl { const __half *down = nullptr, *forget = nullptr, *gain = nullptr; };
struct LayerW {
    bool glob = false;
    QMat wq, wk, wv, wo, wg;
    DMat16 wr, sink, kproj, vproj;
    const float *qscale, *relscale, *relproj, *kw, *vw;
    const float *nw[3];  // local: attn_norm in [0]; global: q/k/v_input_norm
    const float *mlp_norm, *attn_sconv, *mlp_sconv;
    DMat16 router;
    const float* router_bias;
    ik::ExpertW w13, w2;
    const float *attn_q, *mlp_q;
    Ctrl ctrl[4];  // attn(q), k, v, mlp
};
struct App { int L, p, idx, kvslot; };

class InklingEngine final : public Engine {
   public:
    InklingEngine(const ModelFile& mf, const EngineOptions& opt) : opt_(opt) {
        cfg_ = mf.cfg;
        const InklingConfig& k = cfg_.ink;
        if (cfg_.hidden != ik::H || cfg_.n_heads != ik::NH || cfg_.n_kv_heads != ik::NKV || cfg_.head_dim != ik::HD || k.n_experts != ik::NE ||
            k.topk != ik::TOPK || k.expert_ff != ik::FF || k.window != 1024 || k.rel_extent != 1024 || k.d_rel != 16 || k.sconv_kernel != 4 ||
            k.attn_res_block != 6 || k.n_shared_experts != 0 || !k.w13_interleaved)
            throw std::runtime_error("unsupported inkling configuration");
        QV_CUDA_CHECK(cudaStreamCreate(&stream_));
        load_weights(mf);
        bind(mf);
        alloc_state();
        reset();
    }
    ~InklingEngine() override {
        if (g_fwd_) cudaGraphExecDestroy(g_fwd_);
        if (g_dec_) cudaGraphExecDestroy(g_dec_);
        if (h_tp_) cudaFreeHost(h_tp_);
        for (void* p : allocs_) cudaFree(p);
        cudaFree(weights_);
        cudaStreamDestroy(stream_);
    }

    void step(int token, int pos) override {
        if (pos >= opt_.max_ctx) throw std::runtime_error("context overflow");
        if (tp_n_ == TP_SLOTS) { QV_CUDA_CHECK(cudaStreamSynchronize(stream_)); tp_n_ = 0; }
        int* tp = h_tp_ + 2 * tp_n_++;
        tp[0] = token;
        tp[1] = pos;
        QV_CUDA_CHECK(cudaMemcpyAsync(d_tok_, tp, 2 * sizeof(int), cudaMemcpyHostToDevice, stream_));
        if (opt_.dump_hidden) {
            forward();
            collect_dump();
        } else {
            if (!g_fwd_) g_fwd_ = capture(false);
            QV_CUDA_CHECK(cudaGraphLaunch(g_fwd_, stream_));
        }
    }

    void prefill(const int* tokens, int T, int pos0) override {
        for (int i = 0; i < T; ++i) step(tokens[i], pos0 + i);  // TODO(batched prefill): token-by-token for now
        if (opt_.row_logits) QV_CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    void decode_greedy(int first, int start_pos, int n, std::vector<int>& out, bool use_graph) override {
        if (start_pos + n > opt_.max_ctx) throw std::runtime_error("context overflow");
        if (tp_n_ == TP_SLOTS) { QV_CUDA_CHECK(cudaStreamSynchronize(stream_)); tp_n_ = 0; }
        int* tp = h_tp_ + 2 * tp_n_++;
        tp[0] = first;
        tp[1] = start_pos;
        QV_CUDA_CHECK(cudaMemcpyAsync(d_tok_, tp, 2 * sizeof(int), cudaMemcpyHostToDevice, stream_));
        if (use_graph && !g_dec_) g_dec_ = capture(true);
        for (int i = 0; i < n; ++i) {
            if (use_graph) {
                QV_CUDA_CHECK(cudaGraphLaunch(g_dec_, stream_));
            } else {
                forward();
                argmax_advance(logits_, cfg_.ink.unpadded_vocab, d_tok_, d_pos_, d_gen_, stream_);
            }
        }
        out.resize(n);
        QV_CUDA_CHECK(cudaMemcpyAsync(out.data(), d_gen_ + start_pos, n * sizeof(int), cudaMemcpyDeviceToHost, stream_));
        QV_CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    int argmax() override {
        argmax_advance(logits_, cfg_.ink.unpadded_vocab, d_arg_, nullptr, nullptr, stream_);
        int r;
        QV_CUDA_CHECK(cudaMemcpyAsync(&r, d_arg_, 4, cudaMemcpyDeviceToHost, stream_));
        QV_CUDA_CHECK(cudaStreamSynchronize(stream_));
        return r;
    }
    void logits(std::vector<float>& out) override {
        out.resize(cfg_.ink.unpadded_vocab);
        QV_CUDA_CHECK(cudaMemcpyAsync(out.data(), logits_, out.size() * 4, cudaMemcpyDeviceToHost, stream_));
        QV_CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
    void reset() override {
        QV_CUDA_CHECK(cudaStreamSynchronize(stream_));
        QV_CUDA_CHECK(cudaMemset(cst_k_, 0, 48 * 3 * 1024 * 4));
        QV_CUDA_CHECK(cudaMemset(cst_v_, 0, 48 * 3 * 1024 * 4));
        QV_CUDA_CHECK(cudaMemset(cst_a_, 0, (size_t)48 * 3 * ik::H * 4));
        QV_CUDA_CHECK(cudaMemset(cst_m_, 0, (size_t)48 * 3 * ik::H * 4));
        QV_CUDA_CHECK(cudaMemset(ticket_, 0, 4));
        QV_CUDA_CHECK(cudaDeviceSynchronize());
    }
    const std::vector<float>& hidden_dump() const override { return hidden_; }
    const std::vector<float>& logits_dump() const override { return logits_dump_; }
    const std::vector<int>& aux_dump() const override { return rids_; }
    void sync() override { QV_CUDA_CHECK(cudaStreamSynchronize(stream_)); }
    size_t device_bytes_weights() const override { return weights_bytes_; }
    size_t device_bytes_state() const override { return state_bytes_; }

   private:
    static constexpr int TP_SLOTS = 4096;
    static constexpr int NSPLIT_L = 8, NSPLIT_G = 64;
    EngineOptions opt_;
    cudaStream_t stream_{};
    uint8_t* weights_ = nullptr;
    size_t weights_bytes_ = 0, state_bytes_ = 0;
    std::vector<void*> allocs_;
    cudaGraphExec_t g_fwd_ = nullptr, g_dec_ = nullptr;
    int *h_tp_ = nullptr, tp_n_ = 0;
    std::vector<LayerW> lw_;
    std::vector<App> apps_;
    QMat embed_, unembed_;
    const float *embed_norm_, *final_norm_, *out_query_, *msgq_ = nullptr;
    // state
    __half *lk_, *lv_, *gk_, *gv_;
    float *cst_k_, *cst_v_, *cst_a_, *cst_m_;
    // scratch
    float *h_, *pre_h_, *blocks_, *res_, *araw_, *ymoe_, *proj_, *xn_f_, *u_, *gs_, *gg_, *xm_f_, *mraw_, *mg_, *msgs_, *rlog_, *wts_, *qs_, *rl_,
        *part_, *logits_;
    int *ids_, *ticket_, *d_tok_, *d_pos_, *d_gen_, *d_arg_;
    Q8Buf q8x_, q8m_, q8act_, q8o_;
    std::vector<float> hidden_, logits_dump_;
    std::vector<int> rids_;
    float *dump_d_ = nullptr;
    int* rid_d_ = nullptr;

    template <class T> T* dalloc(size_t n) {
        void* p = nullptr;
        QV_CUDA_CHECK(cudaMalloc(&p, n * sizeof(T)));
        QV_CUDA_CHECK(cudaMemset(p, 0, n * sizeof(T)));
        allocs_.push_back(p);
        state_bytes_ += n * sizeof(T);
        return (T*)p;
    }
    Q8Buf q8alloc(int chunks) {
        Q8Buf b;
        b.q = dalloc<int>((size_t)chunks * 8);
        b.d = dalloc<float>(chunks);
        b.sum = dalloc<float>(chunks);
        b.qsum = dalloc<int>(chunks);
        return b;
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
        if (t.kind != Kind::Quant || !t.meta8) throw std::runtime_error(n + " is not a meta8 quantized tensor");
        if (t.group != 64 && t.group != 128) throw std::runtime_error("group must be 64 or 128");
        QMat m;
        m.sm8 = (const uint16_t*)(weights_ + t.off_sm);
        m.rs = (const __half*)(weights_ + t.off_rs);
        m.q = (const uint32_t*)(weights_ + t.off_q);
        m.zlo = -0.25f * (float)((1 << t.bits) - 1);
        m.zstep = (float)(1.5 * (double)((1 << t.bits) - 1) / 255.0);
        m.N = (int)t.shape[t.shape.size() - 2];
        m.K = (int)t.shape[t.shape.size() - 1];
        m.bits = t.bits;
        m.gshift = t.group == 128 ? 7 : 6;
        return m;
    }
    ik::ExpertW expert(const ModelFile& mf, const std::string& n) {
        const TensorInfo& t = mf.at(n);
        if (t.kind != Kind::Quant || !t.meta8 || !t.stacked()) throw std::runtime_error(n + " is not a stacked meta8 tensor");
        ik::ExpertW e;
        e.q = (const uint32_t*)(weights_ + t.off_q);
        e.sm8 = (const uint16_t*)(weights_ + t.off_sm);
        e.rs = (const __half*)(weights_ + t.off_rs);
        e.N = (int)t.shape[1];
        e.K = (int)t.shape[2];
        e.bits = t.bits;
        e.gshift = t.group == 128 ? 7 : 6;
        e.zlo = -0.25f * (float)((1 << t.bits) - 1);
        e.zstep = (float)(1.5 * (double)((1 << t.bits) - 1) / 255.0);
        return e;
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
        const InklingConfig& k = cfg_.ink;
        const std::string P = "model.llm.";
        embed_ = qmat(mf, P + "embed.weight");
        unembed_ = qmat(mf, P + "unembed.weight");
        embed_norm_ = f32(mf, P + "embed_norm.weight");
        final_norm_ = f32(mf, P + "norm.weight");
        out_query_ = f32(mf, P + "attn_res.output_query");
        for (int L = 0; L < cfg_.n_layers; ++L) {
            const std::string l = P + "layers." + std::to_string(L) + ".", a = l + "attn.";
            LayerW W;
            W.glob = k.is_global(L);
            W.wq = qmat(mf, a + "wq_du.weight");
            W.wk = qmat(mf, a + "wk_dv.weight");
            W.wv = qmat(mf, a + "wv_dv.weight");
            W.wo = qmat(mf, a + "wo_ud.weight");
            W.wg = qmat(mf, a + "gate_attn.weight");
            W.wr = f16(mf, a + "wr_du.weight");
            W.sink = f16(mf, a + "attn_sink_proj.weight");
            W.qscale = f32(mf, a + "q_l2_clip.scale");
            W.relscale = f32(mf, a + "rel_l2_clip.scale");
            W.relproj = f32(mf, a + "rel_logits_proj.proj");
            if (W.glob) {
                W.kw = f32(mf, a + "k_sconv.static_w");
                W.vw = f32(mf, a + "v_sconv.static_w");
                W.kproj = f16(mf, a + "k_sconv.weight_proj");
                W.vproj = f16(mf, a + "v_sconv.weight_proj");
                W.nw[0] = f32(mf, l + "q_input_norm.weight");
                W.nw[1] = f32(mf, l + "k_input_norm.weight");
                W.nw[2] = f32(mf, l + "v_input_norm.weight");
            } else {
                W.kw = f32(mf, a + "k_sconv.weight");
                W.vw = f32(mf, a + "v_sconv.weight");
                W.nw[0] = W.nw[1] = W.nw[2] = f32(mf, l + "attn_norm.weight");
            }
            W.mlp_norm = f32(mf, l + "mlp_norm.weight");
            W.attn_sconv = f32(mf, l + "attn_sconv.weight");
            W.mlp_sconv = f32(mf, l + "mlp_sconv.weight");
            W.router = f16(mf, l + "mlp.gate.weight");
            W.router_bias = f32(mf, l + "mlp.gate.bias");
            W.w13 = expert(mf, l + "mlp.experts.w13_weight");
            W.w2 = expert(mf, l + "mlp.experts.w2_weight");
            W.attn_q = L >= 1 ? f32(mf, P + "attn_res.attn_queries." + std::to_string(L - 1)) : nullptr;
            W.mlp_q = f32(mf, P + "attn_res.mlp_queries." + std::to_string(L));
            if (k.in_loop(L)) {
                static const char* nm[4] = {"attn", "k", "v", "mlp"};
                for (int s = 0; s < 4; ++s) {
                    const std::string c = l + "controllers." + nm[s] + ".";
                    W.ctrl[s].down = f16(mf, c + "down_weight").w;
                    W.ctrl[s].forget = f16(mf, c + "forget_weight").w;
                    W.ctrl[s].gain = f16(mf, c + "gain_weight").w;
                }
            }
            lw_.push_back(W);
        }
        // schedule: 0..5, loop x3, 18..23 ; KV slots
        int app = 0, lslot = 0, gslot = 0;
        std::vector<std::vector<int>> slots(cfg_.n_layers);
        auto add = [&](int L, int p) {
            int ks;
            if (k.is_global(L)) ks = gslot++; else ks = lslot++;
            apps_.push_back(App{L, p, app++, ks});
        };
        for (int L = 0; L < k.loop_start; ++L) add(L, 0);
        for (int p = 0; p < k.loop_passes; ++p)
            for (int L = k.loop_start; L < k.loop_end; ++L) add(L, p);
        for (int L = k.loop_end; L < cfg_.n_layers; ++L) add(L, 0);
        n_apps_ = app;
        n_lslots_ = lslot;
        n_gslots_ = gslot;
        if (n_apps_ != 48 || n_lslots_ != 40 || n_gslots_ != 8) throw std::runtime_error("unexpected application count");
        // merged message queries [12 loop layers][4 = query,key,value,mlp][H]
        std::vector<float> mq((size_t)12 * 4 * ik::H);
        std::vector<float> a(12 * 2 * ik::H), b(12 * 2 * ik::H);
        QV_CUDA_CHECK(cudaMemcpy(a.data(), weights_ + mf.at(P + "middle_cycle_cross_pass_queries").off, a.size() * 4, cudaMemcpyDeviceToHost));
        QV_CUDA_CHECK(cudaMemcpy(b.data(), weights_ + mf.at(P + "middle_cycle_cross_pass_kv_queries").off, b.size() * 4, cudaMemcpyDeviceToHost));
        for (int li = 0; li < 12; ++li) {
            auto cp = [&](int dst, const float* src) { std::memcpy(&mq[((size_t)li * 4 + dst) * ik::H], src, ik::H * 4); };
            cp(0, &a[((size_t)li * 2 + 0) * ik::H]);
            cp(1, &b[((size_t)li * 2 + 0) * ik::H]);
            cp(2, &b[((size_t)li * 2 + 1) * ik::H]);
            cp(3, &a[((size_t)li * 2 + 1) * ik::H]);
        }
        float* d = dalloc<float>(mq.size());
        QV_CUDA_CHECK(cudaMemcpy(d, mq.data(), mq.size() * 4, cudaMemcpyHostToDevice));
        msgq_ = d;
    }
    int n_apps_ = 0, n_lslots_ = 0, n_gslots_ = 0;

    void alloc_state() {
        const int H = ik::H;
        lk_ = dalloc<__half>((size_t)n_lslots_ * ik::NKV * 1024 * ik::HD);
        lv_ = dalloc<__half>((size_t)n_lslots_ * ik::NKV * 1024 * ik::HD);
        gk_ = dalloc<__half>((size_t)n_gslots_ * ik::NKV * opt_.max_ctx * ik::HD);
        gv_ = dalloc<__half>((size_t)n_gslots_ * ik::NKV * opt_.max_ctx * ik::HD);
        cst_k_ = dalloc<float>(48 * 3 * 1024);
        cst_v_ = dalloc<float>(48 * 3 * 1024);
        cst_a_ = dalloc<float>((size_t)48 * 3 * H);
        cst_m_ = dalloc<float>((size_t)48 * 3 * H);
        h_ = dalloc<float>(H);
        pre_h_ = dalloc<float>(H);
        blocks_ = dalloc<float>(4 * H);
        res_ = dalloc<float>(H);
        araw_ = dalloc<float>(H);
        ymoe_ = dalloc<float>(H);
        proj_ = dalloc<float>(8704);
        xn_f_ = dalloc<float>(3 * H);
        u_ = dalloc<float>(3 * H);
        gs_ = dalloc<float>(3 * H);
        gg_ = dalloc<float>(4 * 64);
        xm_f_ = dalloc<float>(H);
        mraw_ = dalloc<float>(H);
        mg_ = dalloc<float>(H);
        msgs_ = dalloc<float>((size_t)12 * 4 * H);
        rlog_ = dalloc<float>(128);
        wts_ = dalloc<float>(8);
        ids_ = dalloc<int>(8);
        ticket_ = dalloc<int>(1);
        qs_ = dalloc<float>(ik::NH * ik::HD);
        rl_ = dalloc<float>(ik::NH * 1024);
        part_ = dalloc<float>((size_t)ik::NKV * std::max(NSPLIT_L, NSPLIT_G) * 3 * ik::PART_STRIDE);
        logits_ = dalloc<float>(cfg_.vocab);
        q8x_ = q8alloc(3 * ik::HC);
        q8m_ = q8alloc(ik::HC);
        q8act_ = q8alloc(8 * 36);
        q8o_ = q8alloc(96);
        d_tok_ = dalloc<int>(2);
        d_pos_ = d_tok_ + 1;
        d_gen_ = dalloc<int>(opt_.max_ctx + 1);
        d_arg_ = dalloc<int>(1);
        QV_CUDA_CHECK(cudaMallocHost(&h_tp_, 2 * TP_SLOTS * sizeof(int)));
        if (opt_.dump_hidden) {
            dump_d_ = dalloc<float>((size_t)50 * H);
            rid_d_ = dalloc<int>(48 * 8);
        }
    }

    cudaGraphExec_t capture(bool with_argmax) {
        // warm-up outside capture (lazy one-time allocations of the shared argmax scratch)
        argmax_advance(logits_, cfg_.ink.unpadded_vocab, d_arg_, nullptr, nullptr, stream_);
        QV_CUDA_CHECK(cudaStreamSynchronize(stream_));
        QV_CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
        forward();
        if (with_argmax) argmax_advance(logits_, cfg_.ink.unpadded_vocab, d_tok_, d_pos_, d_gen_, stream_);
        cudaGraph_t g;
        QV_CUDA_CHECK(cudaStreamEndCapture(stream_, &g));
        cudaGraphExec_t ge;
        QV_CUDA_CHECK(cudaGraphInstantiate(&ge, g, 0));
        QV_CUDA_CHECK(cudaGraphDestroy(g));
        return ge;
    }

    void collect_dump() {
        QV_CUDA_CHECK(cudaStreamSynchronize(stream_));
        hidden_.resize((size_t)50 * ik::H);
        rids_.resize(48 * 8);
        QV_CUDA_CHECK(cudaMemcpy(hidden_.data(), dump_d_, hidden_.size() * 4, cudaMemcpyDeviceToHost));
        QV_CUDA_CHECK(cudaMemcpy(rids_.data(), rid_d_, rids_.size() * 4, cudaMemcpyDeviceToHost));
    }

    static ik::GD gd_q(const QMat& W, const Q8Buf& x, int stream, float* y) {
        ik::GD g;
        g.kind = 0; g.N = W.N; g.K = W.K; g.gshift = W.gshift; g.q = W.q; g.sm8 = W.sm8; g.rs = W.rs; g.zlo = W.zlo; g.zstep = W.zstep;
        g.xq = Q8Vec{x.q + stream * (ik::H / 4), x.d + stream * ik::HC, x.sum + stream * ik::HC, x.qsum + stream * ik::HC};
        g.y = y;
        return g;
    }
    static ik::GD gd_h(const DMat16& W, const float* x, float* y) {
        ik::GD g;
        g.kind = 1; g.N = W.N; g.K = W.K; g.w16 = W.w; g.xf = x; g.y = y;
        return g;
    }

    void forward() {
        cudaStream_t s = stream_;
        ik::embed_norm(embed_, d_tok_, embed_norm_, h_, blocks_, s);
        if (dump_d_) QV_CUDA_CHECK(cudaMemcpyAsync(dump_d_, h_, ik::H * 4, cudaMemcpyDeviceToDevice, s));
        for (const App& A : apps_) {
            run_app(A);
            if (A.L == cfg_.ink.loop_end - 1 && A.p < cfg_.ink.loop_passes - 1) {  // messages for the next pass
                ik::ArArgs m;
                m.mode = 0; m.B = 1; m.Q = 4; m.mix = 1; m.epi = 0;
                m.h = h_; m.blocks = blocks_ + 2 * ik::H;  // the state that entered layer 12 in this pass
                m.queries = msgq_; m.qbs = 4 * ik::H; m.raw_out = msgs_; m.rbs = 4 * ik::H;
                ik::attn_res(m, 12, s);
            }
        }
        ik::ArArgs f;
        f.mode = 0; f.B = 4; f.Q = 1; f.mix = 1; f.epi = 1;
        f.h = h_; f.blocks = blocks_; f.queries = out_query_; f.nw[0] = final_norm_; f.xn_out = xn_f_; f.q8 = q8x_;
        ik::attn_res(f, 1, s);
        if (dump_d_) QV_CUDA_CHECK(cudaMemcpyAsync(dump_d_ + 49 * ik::H, xn_f_, ik::H * 4, cudaMemcpyDeviceToDevice, s));
        gemv_q8(unembed_, q8x_, logits_, nullptr, s);
        ik::softcap_inplace(logits_, cfg_.ink.unpadded_vocab, cfg_.ink.final_logit_softcap, s);
    }

    void run_app(const App& A) {
        const InklingConfig& K = cfg_.ink;
        const LayerW& W = lw_[A.L];
        cudaStream_t s = stream_;
        const int H = ik::H, L = A.L, p = A.p, li = A.L - K.loop_start;
        const bool glob = W.glob, msg = p > 0;
        const float* hin = (L == K.loop_start) ? pre_h_ : h_;
        const int Battn = (L + 5) / 6;
        // ---- 1. attention-input mix(es), norms, int8 quantization
        const int Qa = glob ? 3 : 1;
        if (!msg) {
            ik::ArArgs a;
            a.mode = 0; a.B = Battn; a.Q = Qa; a.mix = L > 0; a.epi = 1;
            a.h = hin; a.blocks = blocks_; a.queries = W.attn_q;
            for (int q = 0; q < Qa; ++q) a.nw[q] = W.nw[q];
            a.xn_out = xn_f_; a.q8 = q8x_;
            ik::attn_res(a, 1, s);
        } else {
            ik::ArArgs a;
            a.mode = 0; a.B = Battn; a.Q = Qa; a.mix = 1; a.epi = 0;
            a.h = hin; a.blocks = blocks_; a.queries = W.attn_q; a.raw_out = u_;
            ik::attn_res(a, 1, s);
            ik::GateArgs g;
            g.S = 3; g.g = gg_;
            for (int q = 0; q < 3; ++q) {
                g.h[q] = glob ? u_ + q * H : u_;
                g.msg[q] = msgs_ + ((size_t)li * 4 + q) * H;
                g.down[q] = W.ctrl[q].down; g.forget[q] = W.ctrl[q].forget; g.gain[q] = W.ctrl[q].gain;
                g.out[q] = gs_ + q * H;
            }
            ik::gate(g, s);
            ik::NormArgs n;
            n.S = 3;
            for (int q = 0; q < 3; ++q) { n.in[q] = gs_ + q * H; n.nw[q] = W.nw[q]; }
            n.xn_out = xn_f_; n.q8 = q8x_;
            ik::norm_q8(n, s);
        }
        STAGE("attnin");
        const int sk = (glob || msg) ? 1 : 0, sv = (glob || msg) ? 2 : 0;
        // ---- 2. projections: q, k, v, output gate, rel, sink (+ dynamic conv taps for global layers)
        {
            ik::GDs G;
            G.d[G.n++] = gd_q(W.wq, q8x_, 0, proj_ + 0);
            G.d[G.n++] = gd_q(W.wk, q8x_, sk, proj_ + 3072);
            G.d[G.n++] = gd_q(W.wv, q8x_, sv, proj_ + 4096);
            G.d[G.n++] = gd_q(W.wg, q8x_, 0, proj_ + 5120);
            G.d[G.n++] = gd_h(W.wr, xn_f_, proj_ + 8192);
            G.d[G.n++] = gd_h(W.sink, xn_f_, proj_ + 8576);
            if (glob) {
                G.d[G.n++] = gd_h(W.kproj, xn_f_ + (size_t)sk * H, proj_ + 8600);
                G.d[G.n++] = gd_h(W.vproj, xn_f_ + (size_t)sv * H, proj_ + 8632);
            }
            ik::mgemv(G, W.wq.bits, s);
        }
        STAGE("proj");
        // ---- 3. prep (q/k/v postprocessing, convs, KV write, rel bias)
        const bool ring = !glob;
        const int cap = glob ? opt_.max_ctx : 1024;
        __half* kc = glob ? gk_ + (size_t)A.kvslot * ik::NKV * opt_.max_ctx * ik::HD : lk_ + (size_t)A.kvslot * ik::NKV * 1024 * ik::HD;
        __half* vc = glob ? gv_ + (size_t)A.kvslot * ik::NKV * opt_.max_ctx * ik::HD : lv_ + (size_t)A.kvslot * ik::NKV * 1024 * ik::HD;
        {
            ik::PrepArgs P;
            P.qraw = proj_; P.kraw = proj_ + 3072; P.vraw = proj_ + 4096; P.relraw = proj_ + 8192; P.dwk = proj_ + 8600; P.dwv = proj_ + 8632;
            P.qscale = W.qscale; P.relscale = W.relscale; P.relproj = W.relproj; P.kw = W.kw; P.vw = W.vw;
            P.kst = cst_k_ + (size_t)A.idx * 3 * 1024; P.vst = cst_v_ + (size_t)A.idx * 3 * 1024;
            P.glob = glob; P.pos = d_pos_; P.qs = qs_; P.rl = rl_; P.kc = kc; P.vc = vc; P.cap = cap; P.ring = ring;
            ik::attn_prep(P, s);
        }
        STAGE("prep");
        // ---- 4. attention + combine (sink, output gate) -> int8
        {
            ik::AttnArgs T;
            T.qs = qs_; T.rl = rl_; T.kc = kc; T.vc = vc; T.cap = cap; T.ring = ring; T.window = glob ? 0 : K.window;
            T.pos = d_pos_; T.part = part_; T.nsplit = glob ? NSPLIT_G : NSPLIT_L;
            T.sink = proj_ + 8576; T.gate = proj_ + 5120; T.out = q8o_;
            ik::attn_part(T, s);
            STAGE("part");
            ik::attn_combine(T, s);
        }
        STAGE("attn");
        // ---- 5. output projection
        {
            ik::GDs G;
            ik::GD g;
            g.kind = 0; g.N = W.wo.N; g.K = W.wo.K; g.gshift = W.wo.gshift; g.q = W.wo.q; g.sm8 = W.wo.sm8; g.rs = W.wo.rs;
            g.zlo = W.wo.zlo; g.zstep = W.wo.zstep; g.xq = Q8Vec(q8o_); g.y = araw_;
            G.d[G.n++] = g;
            ik::mgemv(G, W.wo.bits, s);
        }
        STAGE("wo");
        // ---- 6. post-attention conv + residual (+ boundary reset), MLP-input mix, norm
        {
            ik::ArArgs b;
            b.mode = 1; b.B = Battn; b.Q = 1; b.mix = 1; b.boundary = (L % 6 == 0);
            b.h = hin; b.araw = araw_; b.cw = W.attn_sconv; b.cst = cst_a_ + (size_t)A.idx * 3 * H; b.res_out = res_; b.blocks = blocks_;
            b.queries = W.mlp_q;
            if (!msg) {
                b.epi = 1; b.nw[0] = W.mlp_norm; b.xn_out = xm_f_; b.q8 = q8m_;
                ik::attn_res(b, 1, s);
            } else {
                b.epi = 0; b.raw_out = mraw_;
                ik::attn_res(b, 1, s);
                ik::GateArgs g;
                g.S = 1; g.g = gg_;
                g.h[0] = mraw_; g.msg[0] = msgs_ + ((size_t)li * 4 + 3) * H;
                g.down[0] = W.ctrl[3].down; g.forget[0] = W.ctrl[3].forget; g.gain[0] = W.ctrl[3].gain; g.out[0] = mg_;
                ik::gate(g, s);
                ik::NormArgs n;
                n.S = 1; n.in[0] = mg_; n.nw[0] = W.mlp_norm; n.xn_out = xm_f_; n.q8 = q8m_;
                ik::norm_q8(n, s);
            }
        }
        STAGE("postattn");
        // ---- 7-8. router + experts
        ik::router(W.router.w, W.router_bias, xm_f_, rlog_, ids_, wts_, ticket_, K.route_scale, s);
        STAGE("router");
        ik::moe_w13(W.w13, Q8Vec(q8m_), ids_, q8act_, K.gate_softcap, K.up_softcap, s);
        STAGE("w13");
        ik::moe_w2(W.w2, Q8Vec(q8act_), ids_, wts_, ymoe_, s);
        STAGE("w2");
        // ---- 9. MLP conv + residual
        float* hout = (L == K.loop_start - 1) ? pre_h_ : h_;
        ik::finish(res_, ymoe_, W.mlp_sconv, cst_m_ + (size_t)A.idx * 3 * H, hout, s);
        if (dump_d_) {
            QV_CUDA_CHECK(cudaMemcpyAsync(dump_d_ + (size_t)(A.idx + 1) * H, hout, H * 4, cudaMemcpyDeviceToDevice, s));
            QV_CUDA_CHECK(cudaMemcpyAsync(rid_d_ + A.idx * 8, ids_, 32, cudaMemcpyDeviceToDevice, s));
        }
    }
};

}  // namespace

std::unique_ptr<Engine> make_cuda_inkling_engine(const ModelFile& mf, const EngineOptions& opt) {
    return std::make_unique<InklingEngine>(mf, opt);
}

}  // namespace qv
