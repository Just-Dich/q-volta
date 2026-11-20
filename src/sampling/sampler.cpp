#include "sampling/sampler.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <random>

namespace qv {

namespace {
uint64_t splitmix(uint64_t& x) {
    uint64_t z = (x += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
}  // namespace

Sampler::Sampler(const SamplingParams& p, const std::vector<int>& prompt) : p_(p) {
    uint64_t seed = p.has_seed ? p.seed : (std::random_device{}() ^ (uint64_t)std::chrono::steady_clock::now().time_since_epoch().count());
    uint64_t x = seed;
    for (auto& s : s_) s = splitmix(x);
    if (p_.repetition_penalty != 1.0f) for (int t : prompt) seen_[t] = 1;
}

uint64_t Sampler::next() {  // xoshiro256**
    uint64_t r = rotl(s_[1] * 5, 7) * 9, t = s_[1] << 17;
    s_[2] ^= s_[0]; s_[3] ^= s_[1]; s_[1] ^= s_[2]; s_[0] ^= s_[3]; s_[2] ^= t; s_[3] = rotl(s_[3], 45);
    return r;
}
double Sampler::uniform() { return (next() >> 11) * (1.0 / 9007199254740992.0); }

void Sampler::accept(int tok) {
    if (p_.presence_penalty != 0 || p_.frequency_penalty != 0) ++gen_counts_[tok];
    if (p_.repetition_penalty != 1.0f) seen_[tok] = 1;
}

int Sampler::sample(std::vector<float>& lg) {
    const int V = (int)lg.size();
    if (p_.presence_penalty != 0 || p_.frequency_penalty != 0)
        for (auto& [t, c] : gen_counts_) lg[t] -= p_.frequency_penalty * c + p_.presence_penalty;
    if (p_.repetition_penalty != 1.0f)
        for (auto& [t, _] : seen_) lg[t] = lg[t] > 0 ? lg[t] / p_.repetition_penalty : lg[t] * p_.repetition_penalty;
    if (p_.greedy()) return (int)(std::max_element(lg.begin(), lg.end()) - lg.begin());

    const float inv_t = 1.0f / p_.temperature;
    int k = (p_.top_k > 0 && p_.top_k < V) ? p_.top_k : 0;
    const bool need_sorted = k > 0 || p_.top_p < 1.0f || p_.min_p > 0.0f;
    idx_.resize(V);
    std::iota(idx_.begin(), idx_.end(), 0);
    int n = V;
    if (need_sorted) {
        auto gt = [&](int a, int b) { return lg[a] > lg[b] || (lg[a] == lg[b] && a < b); };
        if (k > 0) { std::nth_element(idx_.begin(), idx_.begin() + k, idx_.end(), gt); n = k; std::sort(idx_.begin(), idx_.begin() + n, gt); }
        else std::sort(idx_.begin(), idx_.end(), gt);
    }
    prob_.resize(n);
    float mx = -INFINITY;
    for (int i = 0; i < n; ++i) mx = std::max(mx, lg[idx_[i]]);
    double sum = 0;
    for (int i = 0; i < n; ++i) { prob_[i] = std::exp((lg[idx_[i]] - mx) * inv_t); sum += prob_[i]; }
    for (int i = 0; i < n; ++i) prob_[i] = (float)(prob_[i] / sum);
    if (need_sorted) {
        if (p_.min_p > 0.0f) {
            float thr = p_.min_p * prob_[0];
            while (n > 1 && prob_[n - 1] < thr) --n;
        }
        if (p_.top_p < 1.0f) {
            double c = 0;
            int keep = n;
            for (int i = 0; i < n; ++i) { c += prob_[i]; if (c >= p_.top_p) { keep = i + 1; break; } }
            n = keep;
        }
    }
    double tot = 0;
    for (int i = 0; i < n; ++i) tot += prob_[i];
    double u = uniform() * tot, c = 0;
    for (int i = 0; i < n; ++i) { c += prob_[i]; if (u < c) return idx_[i]; }
    return idx_[n - 1];
}

TokenLogprob Sampler::logprob(const std::vector<float>& raw, int chosen, int top_n) {
    float mx = *std::max_element(raw.begin(), raw.end());
    double s = 0;
    for (float x : raw) s += std::exp((double)x - mx);
    const double lse = mx + std::log(s);
    TokenLogprob r;
    r.token = chosen;
    r.logprob = (float)(raw[chosen] - lse);
    if (top_n > 0) {
        std::vector<int> idx(raw.size());
        std::iota(idx.begin(), idx.end(), 0);
        top_n = std::min<int>(top_n, (int)raw.size());
        std::partial_sort(idx.begin(), idx.begin() + top_n, idx.end(), [&](int a, int b) { return raw[a] > raw[b] || (raw[a] == raw[b] && a < b); });
        for (int i = 0; i < top_n; ++i) r.top.emplace_back(idx[i], (float)(raw[idx[i]] - lse));
    }
    return r;
}

}  // namespace qv
