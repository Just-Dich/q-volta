// qv_vision: encode an image with the Inkling vision tower (CPU or CUDA) and dump / time the result.
//   qv_vision --vision FILE.qv --image PATH [--cuda] [--out emb.f32] [--pix pix.f32] [--bench N] [--synthetic WxH]
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>

#include "vision/vision.h"

using namespace qv::vision;

int main(int argc, char** argv) {
    std::string vf, img, out, pixo;
    bool cuda = false;
    int bench = 0, sw = 0, sh = 0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto nx = [&]() { return std::string(argv[++i]); };
        if (a == "--vision") vf = nx();
        else if (a == "--image") img = nx();
        else if (a == "--cuda") cuda = true;
        else if (a == "--out") out = nx();
        else if (a == "--pix") pixo = nx();
        else if (a == "--bench") bench = std::stoi(nx());
        else if (a == "--synthetic") std::sscanf(nx().c_str(), "%dx%d", &sw, &sh);
    }
    try {
        Rgb8 im;
        if (sw) {  // deterministic gradient + noise image of the requested size (benchmarks)
            im.w = sw; im.h = sh; im.px.resize((size_t)sw * sh * 3);
            uint32_t s = 12345;
            for (int y = 0; y < sh; ++y) for (int x = 0; x < sw; ++x) for (int c = 0; c < 3; ++c) { s = s * 1664525u + 1013904223u; im.px[((size_t)y * sw + x) * 3 + c] = (uint8_t)(((x * (c + 1) + y * (3 - c)) / 3 + (s >> 28) * 4) & 255); }
        } else {
            std::ifstream f(img, std::ios::binary);
            std::vector<uint8_t> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
            im = decode_image(b.data(), b.size(), Limits());
        }
        const auto t0 = std::chrono::steady_clock::now();
        Pixels px = preprocess(im);
        const double pre_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (!pixo.empty()) { std::ofstream o(pixo, std::ios::binary); o.write((const char*)px.data.data(), px.data.size() * 4); }
        const auto t1 = std::chrono::steady_clock::now();
        auto enc = Encoder::create(vf, cuda);
        const double load_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
        std::vector<float> emb((size_t)px.tokens() * enc->cfg().text_hidden);
        std::printf("image %dx%d -> grid %dx%d = %d tokens, backend %s, preprocess %.2f ms, load %.0f ms\n", im.w, im.h, px.rows, px.cols, px.tokens(), enc->backend(), pre_ms, load_ms);
        enc->encode(px, emb.data());
        std::printf("encode (first) %.1f ms\n", enc->last_ms);
        std::vector<double> t;
        for (int i = 0; i < bench; ++i) { enc->encode(px, emb.data()); t.push_back(enc->last_ms); }
        if (bench) { std::sort(t.begin(), t.end()); std::printf("encode median of %d: %.1f ms (min %.1f, max %.1f)\n", bench, t[t.size() / 2], t.front(), t.back()); }
        if (!out.empty()) { std::ofstream o(out, std::ios::binary); o.write((const char*)emb.data(), emb.size() * 4); }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
