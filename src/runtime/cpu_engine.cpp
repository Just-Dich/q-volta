// CPU backend: decode-step executor for Qwen3.5-family hybrid (GDN + gated attention) models.
// Mirrors the per-layer dataflow of cuda_engine.cu on top of the AVX2 kernels in src/backends/cpu.
//
// Weights are mmap'ed read-only from the .qv.bin and QMatC views point straight into the mapping. Each token runs
// as a short sequence of parallel regions on a spin-barrier pool; the projections of one region are flattened into a
// single list of row chunks (dynamic claiming) and small elementwise/quantize/conv/residual work is fused into the
// chunk epilogues so a GDN layer needs 5 dispatches and an attention layer 7.
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "cpu_gemv.h"
#include "ops.h"
#include "qv/engine.h"

namespace qv {
using namespace qv::cpu;

namespace {

// ---- optional per-phase timers (build with -DQV_CPU_PROF) -------------------------------------------------------------
enum Phase { P_SERIAL, P_PROJ_IN, P_GDN, P_ATTN, P_PROJ_OUT, P_GATEUP, P_DOWN, P_HEAD, P_ARGMAX, P_N };
const char* kPhaseName[P_N] = {"serial(embed,norm,quant,prep)", "in-proj gemv (+conv,a,b / q,k,v)", "gdn step",
                               "attn decode+merge", "out-proj gemv (+res)", "gate/up gemv (+silu,quant)",
                               "down gemv (+res)", "lm_head gemv", "argmax"};
#ifdef QV_CPU_PROF
struct Prof {
    double ns[P_N] = {};
    long steps = 0;
};
struct Scope {
    Prof& p; int ph; std::chrono::steady_clock::time_point t0;
    Scope(Prof& pp, int ph_) : p(pp), ph(ph_), t0(std::chrono::steady_clock::now()) {}
    ~Scope() { p.ns[ph] += std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count(); }
};
#define QV_PHASE(ph) Scope scope_##__LINE__(prof_, ph)
#else
struct Prof { double ns[P_N] = {}; long steps = 0; };
#define QV_PHASE(ph) ((void)0)
#endif

// fp16 dense row dot (F16C + FMA), K % 8 == 0
float dot_f16(const uint16_t* w, const float* x, int K) {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    int k = 0;
    for (; k + 16 <= K; k += 16) {
        a0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(w + k))), _mm256_loadu_ps(x + k), a0);
        a1 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(w + k + 8))), _mm256_loadu_ps(x + k + 8), a1);
    }
    for (; k < K; k += 8)
        a0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(w + k))), _mm256_loadu_ps(x + k), a0);
    a0 = _mm256_add_ps(a0, a1);
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(a0), _mm256_extractf128_ps(a0, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}

// ---- one parallel region = list of row-chunk jobs ---------------------------------------------------------------------
enum class Epi { None, Conv, GateUp, Residual };
struct Job {
    const QMatC* W = nullptr;       // quantized matrix (null: dense fp16 rows)
    const QMatC* W2 = nullptr;      // GateUp: the "up" matrix (W is "gate")
    const uint16_t* dw = nullptr;   // dense fp16 [N][K]
    int N = 0, K = 0;
    const Q8ActC* x = nullptr;      // quantized activation (quantized matrices)
    const float* xf = nullptr;      // fp32 activation (dense rows)
    float* y = nullptr;             // output (GateUp: gate)
    float* y2 = nullptr;            // GateUp: up
    Epi epi = Epi::None;
    int rows_per_chunk = 16;
    // epilogue operands
    const float* conv_w = nullptr; float* conv_state = nullptr; float* conv_out = nullptr;  // Conv
    float* act = nullptr; Q8ActC* xq_act = nullptr;                                          // GateUp
    float* resid = nullptr;                                                                  // Residual: resid += y
    size_t first_chunk = 0, nchunks = 0;
};

struct GdnLayer {
    QMatC qkv, z, out;
    const uint16_t *a, *b;
    const float *A_neg, *dt_bias, *conv_w, *norm_w;
    int a_rows, b_rows;
};
struct AttnLayer {
    QMatC q, k, v, o;
    const float *q_norm, *k_norm;
};
struct Layer {
    bool attn;
    int idx_kind;
    const float *in_norm, *post_norm;
    QMatC gate, up, down;
    GdnLayer g;
    AttnLayer at;
};

class CpuEngine final : public Engine {
   public:
    CpuEngine(const ModelFile& mf, const EngineOptions& opt)
        : opt_(opt), pool_(opt.threads > 0 ? opt.threads : 32, pool_cpus()) {
        cfg_ = mf.cfg;
        map_weights(mf);
        bind(mf);
        alloc_state();
        reset();
    }
    ~CpuEngine() override {
#ifdef QV_CPU_PROF
        if (prof_.steps) {
            double tot = 0;
            for (double v : prof_.ns) tot += v;
            std::fprintf(stderr, "[prof] %ld forward passes, %.2f ms/token total in timed phases\n", prof_.steps,
                         tot / prof_.steps / 1e6);
            for (int i = 0; i < P_N; ++i)
                std::fprintf(stderr, "[prof]   %-34s %8.3f ms/token  (%4.1f%%)\n", kPhaseName[i],
                             prof_.ns[i] / prof_.steps / 1e6, 100.0 * prof_.ns[i] / tot);
        }
#endif
        if (map_) { if (owned_) std::free(map_); else munmap(map_, map_len_); }
        std::free(k_cache_);
        std::free(v_cache_);
    }

    void step(int token, int pos) override {
        if (pos >= opt_.max_ctx) throw std::runtime_error("context overflow");
        forward(token, pos, true);
    }

    void prefill(const int* tokens, int T, int pos0) override {
        if (pos0 + T > opt_.max_ctx) throw std::runtime_error("context overflow");
        for (int t = 0; t < T; ++t) {
            const bool need = opt_.dump_hidden || t == T - 1;
            forward(tokens[t], pos0 + t, need);
            if (opt_.dump_hidden) logits_all_.insert(logits_all_.end(), logits_.begin(), logits_.end());
        }
    }

    void decode_greedy(int first, int start_pos, int n, std::vector<int>& out, bool) override {
        if (start_pos + n > opt_.max_ctx) throw std::runtime_error("context overflow");
        out.resize(n);
        int tok = first;
        for (int i = 0; i < n; ++i) {
            forward(tok, start_pos + i, true);
            tok = argmax();
            out[i] = tok;
        }
    }

    int argmax() override {
        QV_PHASE(P_ARGMAX);
        const int V = cfg_.vocab, nt = pool_.size();
        std::vector<float>& bv = arg_val_;
        std::vector<int>& bi = arg_idx_;
        bv.assign(nt, -INFINITY);
        bi.assign(nt, 0x7fffffff);
        pool_.parallel_for((size_t)V, [&](size_t b, size_t e, int t) { argmax_range(logits_.data(), (int)b, (int)e, &bv[t], &bi[t]); }, 64);
        int r;
        argmax_merge(bv.data(), bi.data(), nt, &r);
        return r;
    }
    void logits(std::vector<float>& out) override { out = logits_; }
    void reset() override {
        std::memset(conv_state_.data(), 0, conv_state_.size() * 4);
        std::memset(gdn_state_.data(), 0, gdn_state_.size() * 4);
        hidden_.clear();
        logits_all_.clear();
    }
    const std::vector<float>& hidden_dump() const override { return hidden_; }
    const std::vector<float>& logits_dump() const override { return logits_all_; }
    void sync() override {}
    size_t device_bytes_weights() const override { return map_len_; }
    size_t device_bytes_state() const override { return state_bytes_; }

   private:
    EngineOptions opt_;
    ThreadPool pool_;
    Prof prof_;
    void* map_ = nullptr;
    size_t map_len_ = 0;
    const char* base_ = nullptr;
    std::vector<Layer> layers_;
    QMatC head_;
    QRows embed_{};
    const float* final_norm_ = nullptr;
    int n_gdn_ = 0, n_attn_ = 0;
    // state
    std::vector<float> conv_state_, gdn_state_;
    float *k_cache_ = nullptr, *v_cache_ = nullptr;
    size_t state_bytes_ = 0;
    // scratch
    std::vector<float> x_, xn_, qkv_, conv_out_, z_, alpha_, beta_, mix_, qg_, kb_, vb_, qr_, part_, g_, u_, act_, tmp_,
        logits_, inv_freq_, rope_cs_, arg_val_;
    std::vector<int> arg_idx_;
    Q8ActC xq_, xq_mix_, xq_act_;
    std::vector<float> hidden_, logits_all_;
    int splits_ = 64;
    std::vector<Job> jobs_;

    static std::vector<int> pool_cpus() {
        std::vector<int> v;
        if (const char* e = std::getenv("QV_CPU_LIST")) {  // comma separated cpu ids
            for (const char* p = e; *p;) { v.push_back(std::atoi(p)); while (*p && *p != ',') ++p; if (*p) ++p; }
        }
        return v;
    }

    // ---------------------------------------------------------------- weights
    // Default: weights are read once into a 2 MiB-aligned anonymous buffer backed by transparent huge pages (the decode
    // loop streams the whole 12.6 GiB per token, so 4 KiB-page TLB misses on the file mapping matter; see
    // QV_WEIGHTS=mmap for the plain read-only file mapping with MAP_POPULATE).
    void map_weights(const ModelFile& mf) {
        int fd = ::open(mf.bin_path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("cannot open " + mf.bin_path);
        const char* mode = std::getenv("QV_WEIGHTS");
        const bool use_mmap = mode && std::string(mode) == "mmap";
        map_len_ = mf.bin_size;
        if (use_mmap) {
            map_ = mmap(nullptr, map_len_, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
            if (map_ == MAP_FAILED) throw std::runtime_error("mmap failed");
            if (std::getenv("QV_HUGEPAGE")) madvise(map_, map_len_, MADV_HUGEPAGE);
        } else {
            const size_t al = 2u << 20, len = (map_len_ + al - 1) / al * al;
            map_ = std::aligned_alloc(al, len);
            if (!map_) throw std::runtime_error("weight buffer allocation failed");
            owned_ = true;
            madvise(map_, len, MADV_HUGEPAGE);
            const size_t CH = 64u << 20, nch = (map_len_ + CH - 1) / CH;
            std::atomic<bool> bad{false};
            pool_.parallel_for_dyn(nch, [&](size_t b, size_t e, int) {
                for (size_t c = b; c < e; ++c) {
                    size_t off = c * CH, end = std::min(map_len_, off + CH);
                    while (off < end) {
                        ssize_t n = ::pread(fd, (char*)map_ + off, end - off, off);
                        if (n <= 0) { bad = true; break; }
                        off += (size_t)n;
                    }
                }
            }, 1);
            if (bad) throw std::runtime_error("short read on weights");
        }
        ::close(fd);
        base_ = (const char*)map_;
    }
    bool owned_ = false;
    QMatC qmat(const ModelFile& mf, const std::string& n) {
        const TensorInfo& t = mf.at(n);
        if (t.kind != Kind::Quant) throw std::runtime_error(n + " is not quantized");
        if (t.group != 64) throw std::runtime_error("group 64 only");
        QMatC m;
        m.q = (const uint32_t*)(base_ + t.off_q);
        m.sm = (const uint16_t*)(base_ + t.off_sm);
        m.N = (int)t.shape[0];
        m.K = (int)t.shape[1];
        m.bits = t.bits;
        return m;
    }
    const float* f32(const ModelFile& mf, const std::string& n) {
        const TensorInfo& t = mf.at(n);
        if (t.kind != Kind::F32) throw std::runtime_error(n + " is not f32");
        return (const float*)(base_ + t.off);
    }
    const uint16_t* f16(const ModelFile& mf, const std::string& n, int& rows) {
        const TensorInfo& t = mf.at(n);
        if (t.kind != Kind::F16) throw std::runtime_error(n + " is not f16");
        rows = (int)t.shape[0];
        return (const uint16_t*)(base_ + t.off);
    }

    void bind(const ModelFile& mf) {
        const std::string LM = "model.language_model.";
        {
            QMatC e = qmat(mf, LM + "embed_tokens.weight");
            embed_ = QRows{e.q, e.sm, e.K, e.bits};
        }
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
                L.g.a = f16(mf, q + "in_proj_a.weight", L.g.a_rows);
                L.g.b = f16(mf, q + "in_proj_b.weight", L.g.b_rows);
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

    static void prep_q8(Q8ActC& a, int K) {
        a.K = K;
        a.q.assign((size_t)K, 0);
        a.ds.assign((size_t)(K / 64) * 4, 0.f);
    }

    void alloc_state() {
        const Config& c = cfg_;
        const int conv_C = 2 * c.lin_k_heads * c.lin_k_dim + c.lin_v_heads * c.lin_v_dim;
        const int VD = c.lin_v_heads * c.lin_v_dim, QD = c.n_heads * c.head_dim;
        conv_state_.assign((size_t)n_gdn_ * conv_C * 3, 0.f);
        gdn_state_.assign((size_t)n_gdn_ * c.lin_v_heads * c.lin_k_dim * c.lin_v_dim, 0.f);
        const size_t kv = (size_t)n_attn_ * c.n_kv_heads * opt_.max_ctx * c.head_dim;
        k_cache_ = (float*)std::calloc(kv, 4);
        v_cache_ = (float*)std::calloc(kv, 4);
        if (!k_cache_ || !v_cache_) throw std::runtime_error("KV cache allocation failed");
        state_bytes_ = (conv_state_.size() + gdn_state_.size()) * 4 + kv * 8;
        x_.assign(c.hidden, 0.f); xn_.assign(c.hidden, 0.f);
        qkv_.assign(conv_C, 0.f); conv_out_.assign(conv_C, 0.f);
        z_.assign(VD, 0.f); alpha_.assign(c.lin_v_heads, 0.f); beta_.assign(c.lin_v_heads, 0.f);
        mix_.assign(std::max(VD, QD), 0.f);
        qg_.assign(2 * QD, 0.f); kb_.assign(c.n_kv_heads * c.head_dim, 0.f); vb_.assign(kb_.size(), 0.f);
        qr_.assign(QD, 0.f);
        part_.assign(attn_part_floats(c.n_heads, splits_, c.head_dim), 0.f);
        g_.assign(c.inter, 0.f); u_.assign(c.inter, 0.f); act_.assign(c.inter, 0.f);
        tmp_.assign(c.hidden, 0.f);
        logits_.assign(c.vocab, 0.f);
        prep_q8(xq_, c.hidden);
        prep_q8(xq_mix_, std::max(VD, QD));
        prep_q8(xq_act_, c.inter);
        inv_freq_.resize(c.rope_dim / 2);
        for (int i = 0; i < c.rope_dim / 2; ++i)  // fp32, exactly like torch / the CUDA engine
            inv_freq_[i] = 1.0f / std::pow((float)c.rope_theta, (float)(2 * i) / (float)c.rope_dim);
        rope_cs_.assign(c.rope_dim, 0.f);
        jobs_.reserve(8);
    }

    // ---------------------------------------------------------------- region runner
    void run_jobs() {
        size_t total = 0;
        for (Job& j : jobs_) {
            j.first_chunk = total;
            j.nchunks = (size_t)(j.N + j.rows_per_chunk - 1) / j.rows_per_chunk;
            total += j.nchunks;
        }
        auto body = [&](size_t cb, size_t ce, int) {
            for (size_t c = cb; c < ce; ++c) {
                const Job* jp = &jobs_[0];
                for (const Job& j : jobs_) if (c >= j.first_chunk) jp = &j;
                const Job& j = *jp;
                const int r0 = (int)(c - j.first_chunk) * j.rows_per_chunk;
                const int r1 = std::min(j.N, r0 + j.rows_per_chunk);
                chunk(j, r0, r1);
            }
        };
        if (total == 1) body(0, 1, 0);
        else pool_.parallel_for_dyn(total, body, 1);
        jobs_.clear();
    }

    void chunk(const Job& j, int r0, int r1) {
        if (!j.W) {  // dense fp16 rows
            for (int r = r0; r < r1; ++r) j.y[r] = dot_f16(j.dw + (size_t)r * j.K, j.xf, j.K);
            return;
        }
        if (diag_fp_ && j.xf) diag_gemv(*j.W, j.xf, j.y, r0, r1); else gemv_q(*j.W, *j.x, j.y, r0, r1);
        switch (j.epi) {
            case Epi::None: break;
            case Epi::Conv: conv_step_range(j.y, j.conv_w, j.conv_state, j.conv_out, r0, r1); break;
            case Epi::Residual: add_range(j.resid, j.y, j.resid, r0, r1); break;
            case Epi::GateUp:
                if (diag_fp_ && j.xf) diag_gemv(*j.W2, j.xf, j.y2, r0, r1); else gemv_q(*j.W2, *j.x, j.y2, r0, r1);
                silu_mul_range(j.y, j.y2, j.act, r0, r1);
                quantize_act_q8_range(j.act, r0 / 64, (r1 + 63) / 64, *j.xq_act);
                break;
        }
    }

    // Diagnostic (QV_DIAG_FP=1): fp32 activations against dequantized weights, scalar and slow. Isolates the error that
    // comes from int8 activation quantization.
    bool diag_fp_ = std::getenv("QV_DIAG_FP") != nullptr;
    void diag_gemv(const QMatC& W, const float* xf, float* y, int r0, int r1) {
        std::vector<float> row(W.K);
        for (int r = r0; r < r1; ++r) {
            embed_row(QRows{W.q, W.sm, W.K, W.bits}, r, row.data());
            double a = 0;
            for (int k = 0; k < W.K; ++k) a += (double)row[k] * xf[k];
            y[r] = (float)a;
        }
    }

    Job qjob(const QMatC& W, const Q8ActC& x, float* y, const float* xf = nullptr) {
        Job j;
        j.W = &W; j.x = &x; j.N = W.N; j.K = W.K; j.y = y; j.xf = xf;
        return j;
    }

    void norm_quant(const float* w) {
        rmsnorm(x_.data(), w, xn_.data(), cfg_.hidden, cfg_.eps);
        quantize_act_q8_range(xn_.data(), 0, cfg_.hidden / 64, xq_);
    }

    void dump() { hidden_.insert(hidden_.end(), x_.begin(), x_.end()); }

    // ---------------------------------------------------------------- one token
    void forward(int token, int pos, bool want_logits) {
        const Config& c = cfg_;
        const int H = c.hidden;
#ifdef QV_CPU_PROF
        ++prof_.steps;
#endif
        {
            QV_PHASE(P_SERIAL);
            embed_row(embed_, token, x_.data());
            attn_rope_table(pos, inv_freq_.data(), c.rope_dim, rope_cs_.data());
            if (opt_.dump_hidden) dump();
        }
        for (const Layer& L : layers_) {
            { QV_PHASE(P_SERIAL); norm_quant(L.in_norm); }
            if (!L.attn) gdn_layer(L); else attn_layer(L, pos);
            // MLP
            { QV_PHASE(P_SERIAL); norm_quant(L.post_norm); }
            {
                QV_PHASE(P_GATEUP);
                Job j = qjob(L.gate, xq_, g_.data(), xn_.data());
                j.W2 = &L.up; j.y2 = u_.data(); j.epi = Epi::GateUp; j.rows_per_chunk = 64;
                j.act = act_.data(); j.xq_act = &xq_act_;
                jobs_.push_back(j);
                run_jobs();
            }
            {
                QV_PHASE(P_DOWN);
                Job j = qjob(L.down, xq_act_, tmp_.data(), act_.data());
                j.epi = Epi::Residual; j.resid = x_.data();
                jobs_.push_back(j);
                run_jobs();
            }
            if (opt_.dump_hidden) dump();
        }
        if (want_logits) {
            { QV_PHASE(P_SERIAL); norm_quant(final_norm_); }
            QV_PHASE(P_HEAD);
            jobs_.push_back(qjob(head_, xq_, logits_.data(), xn_.data()));
            run_jobs();
        }
        (void)H;
    }

    void gdn_layer(const Layer& L) {
        const Config& c = cfg_;
        const GdnLayer& g = L.g;
        const int C = g.qkv.N;
        const int Hv = c.lin_v_heads, dv = c.lin_v_dim;
        {
            QV_PHASE(P_PROJ_IN);
            Job a = qjob(g.qkv, xq_, qkv_.data(), xn_.data());
            a.epi = Epi::Conv; a.conv_w = g.conv_w; a.conv_out = conv_out_.data();
            a.conv_state = conv_state_.data() + (size_t)L.idx_kind * C * 3;
            jobs_.push_back(a);
            jobs_.push_back(qjob(g.z, xq_, z_.data(), xn_.data()));
            Job b; b.dw = g.a; b.N = g.a_rows; b.K = c.hidden; b.xf = xn_.data(); b.y = alpha_.data(); jobs_.push_back(b);
            Job d; d.dw = g.b; d.N = g.b_rows; d.K = c.hidden; d.xf = xn_.data(); d.y = beta_.data(); jobs_.push_back(d);
            run_jobs();
        }
        {
            QV_PHASE(P_GDN);
            GdnArgs a{conv_out_.data(), alpha_.data(), beta_.data(), g.A_neg, g.dt_bias, z_.data(), g.norm_w,
                      gdn_state_.data() + (size_t)L.idx_kind * Hv * c.lin_k_dim * dv, mix_.data(),
                      c.lin_k_heads, Hv, c.lin_k_dim, dv, c.eps};
            pool_.parallel_for_dyn((size_t)Hv, [&](size_t b, size_t e, int) {
                gdn_step_range(a, (int)b, (int)e);
                quantize_act_q8_range(mix_.data(), (int)b * dv / 64, (int)e * dv / 64, xq_mix_);
            }, 1);
        }
        {
            QV_PHASE(P_PROJ_OUT);
            Job j = qjob(g.out, xq_mix_, tmp_.data(), mix_.data());
            j.epi = Epi::Residual; j.resid = x_.data();
            jobs_.push_back(j);
            run_jobs();
        }
    }

    void attn_layer(const Layer& L, int pos) {
        const Config& c = cfg_;
        const AttnLayer& at = L.at;
        const int Hq = c.n_heads, Hkv = c.n_kv_heads, D = c.head_dim;
        {
            QV_PHASE(P_PROJ_IN);
            jobs_.push_back(qjob(at.q, xq_, qg_.data(), xn_.data()));
            jobs_.push_back(qjob(at.k, xq_, kb_.data(), xn_.data()));
            jobs_.push_back(qjob(at.v, xq_, vb_.data(), xn_.data()));
            run_jobs();
        }
        const size_t kv_layer = (size_t)Hkv * opt_.max_ctx * D;
        float* kc = k_cache_ + L.idx_kind * kv_layer;
        float* vc = v_cache_ + L.idx_kind * kv_layer;
        {
            QV_PHASE(P_SERIAL);
            AttnPrepArgs p{qg_.data(), kb_.data(), vb_.data(), at.q_norm, at.k_norm, rope_cs_.data(), qr_.data(), kc, vc,
                           Hq, Hkv, D, c.rope_dim, opt_.max_ctx, pos, c.eps};
            attn_prep_range(p, 0, Hq + Hkv);
        }
        {
            QV_PHASE(P_ATTN);
            AttnDecodeArgs d{qr_.data(), qg_.data(), kc, vc, part_.data(), mix_.data(), Hq, Hkv, D, opt_.max_ctx,
                             pos + 1, splits_};
            const int ns = attn_active_splits(d.n_keys, splits_);
            pool_.parallel_for_dyn((size_t)Hkv * ns, [&](size_t b, size_t e, int) {
                for (size_t w = b; w < e; ++w) {
                    const int item = (int)(w / ns) * splits_ + (int)(w % ns);
                    attn_decode_range(d, item, item + 1);
                }
            }, 1);
            pool_.parallel_for_dyn((size_t)Hq, [&](size_t b, size_t e, int) {
                attn_merge_range(d, (int)b, (int)e);
                quantize_act_q8_range(mix_.data(), (int)b * D / 64, (int)e * D / 64, xq_mix_);
            }, 1);
        }
        {
            QV_PHASE(P_PROJ_OUT);
            Job j = qjob(at.o, xq_mix_, tmp_.data(), mix_.data());
            j.epi = Epi::Residual; j.resid = x_.data();
            jobs_.push_back(j);
            run_jobs();
        }
    }
};

}  // namespace

std::unique_ptr<Engine> make_cpu_engine(const ModelFile& mf, const EngineOptions& opt) {
    return std::make_unique<CpuEngine>(mf, opt);
}

}  // namespace qv
