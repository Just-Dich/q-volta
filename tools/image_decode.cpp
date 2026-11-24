// qv_image_decode IN OUT.rgb : decode an image exactly like the server (incl. checkerboard alpha composite), write raw RGB8, print "W H".
// On failure prints the error message to stderr and exits 2. Used by tests/test_image_formats.py.
#include <cstdio>
#include <fstream>
#include <iterator>

#include "vision/vision.h"

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: qv_image_decode IN OUT.rgb [max_side]\n"); return 1; }
    std::ifstream f(argv[1], std::ios::binary);
    std::vector<uint8_t> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    qv::vision::Limits lim;
    if (argc > 3) lim.max_side = std::atoi(argv[3]);
    try {
        qv::vision::Rgb8 im = qv::vision::decode_image(b.data(), b.size(), lim);
        std::ofstream o(argv[2], std::ios::binary);
        o.write((const char*)im.px.data(), (std::streamsize)im.px.size());
        std::printf("%d %d\n", im.w, im.h);
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 2; }
    return 0;
}
