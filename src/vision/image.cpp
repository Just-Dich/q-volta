// Image decoding (stb_image, public domain / MIT), preprocessing (HF Inkling2ImageProcessor) and `image_url` source loading.
#include <arpa/inet.h>
#include <netdb.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <fcntl.h>
#include <signal.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_STDIO
#define STBI_NO_LINEAR
#define STBI_NO_HDR
#define STBI_MAX_DIMENSIONS 16384
#include "../../third_party/stb/stb_image.h"
#include "vision.h"

extern "C" {
#include "../../third_party/libwebp/src/webp/decode.h"
#include "../../third_party/libwebp/src/webp/demux.h"
}

extern char** environ;

namespace qv::vision {

static bool has_prefix(const uint8_t* d, size_t n, const char* m, size_t off = 0) {
    const size_t l = std::strlen(m);
    return n >= off + l && !std::memcmp(d + off, m, l);
}

// Name of a recognised but unsupported container, or "" (magic bytes only).
static std::string unsupported_format(const uint8_t* d, size_t n) {
    if (has_prefix(d, n, "ftyp", 4) && n >= 12) {
        const std::string b((const char*)d + 8, 4);
        if (b == "avif" || b == "avis") return "AVIF";
        if (b == "heic" || b == "heix" || b == "hevc" || b == "hevx" || b == "heim" || b == "heis" || b == "mif1" || b == "msf1") return "HEIC";
        return "ISO-BMFF (" + b + ")";
    }
    if (n >= 4 && ((!std::memcmp(d, "II*", 3) && d[3] == 0) || (!std::memcmp(d, "MM", 2) && d[2] == 0 && d[3] == 0x2a) || (!std::memcmp(d, "II+", 3) && d[3] == 0))) return "TIFF";
    if (n >= 2 && d[0] == 0xff && d[1] == 0x0a) return "JPEG XL";
    if (n >= 12 && !std::memcmp(d, "\0\0\0\x0cJXL ", 8)) return "JPEG XL";
    if (n >= 12 && !std::memcmp(d, "\0\0\0\x0cjP  ", 8)) return "JPEG 2000";
    if (n >= 4 && !std::memcmp(d, "\0\0\1\0", 4)) return "ICO";
    if (has_prefix(d, n, "%PDF")) return "PDF";
    if (has_prefix(d, n, "<svg") || has_prefix(d, n, "<?xml")) return "SVG";
    if (has_prefix(d, n, "8BPS")) return "";  // PSD: stb handles it
    return "";
}

static std::string supported_list() { return "PNG, JPEG, GIF, BMP, WebP, PNM, TGA"; }

// "starts with ..." preview of non-image data (HTML / JSON error pages etc.).
static std::string preview(const uint8_t* d, size_t n) {
    std::string o;
    bool text = true;
    for (size_t i = 0; i < n && i < 40; ++i) {
        const uint8_t c = d[i];
        if (c == '\n' || c == '\r' || c == '\t') o += ' ';
        else if (c >= 32 && c < 127) o += (char)c;
        else { text = false; break; }
    }
    if (text) return "'" + o + (n > 40 ? "...'" : "'");
    char buf[16 * 3 + 1] = {0};
    for (size_t i = 0; i < n && i < 16; ++i) std::snprintf(buf + i * 3, 4, "%02x ", d[i]);
    return std::string("bytes ") + buf;
}

static bool looks_like_text(const uint8_t* d, size_t n) {
    size_t i = 0;
    while (i < n && std::isspace(d[i])) ++i;
    if (i >= n) return false;
    const uint8_t c = d[i];
    return c == '<' || c == '{' || c == '[';
}

static bool is_webp(const uint8_t* d, size_t n) { return has_prefix(d, n, "RIFF") && has_prefix(d, n, "WEBP", 8); }

// WebP (libwebp decode-only subset): still images via WebPDecode, animated via WebPAnimDecoder (first frame, composited on the canvas). Output RGBA.
static uint8_t* decode_webp(const uint8_t* data, size_t n, const Limits& lim, int& w, int& h, bool& alpha) {
    WebPBitstreamFeatures f;
    const VP8StatusCode st = WebPGetFeatures(data, n, &f);
    if (st == VP8_STATUS_NOT_ENOUGH_DATA) throw ImageError("decode", "WebP image is truncated");
    if (st != VP8_STATUS_OK) throw ImageError("decode", "corrupt WebP image (header)");
    w = f.width; h = f.height;
    if (w <= 0 || h <= 0) throw ImageError("decode", "image has zero size");
    if (w > lim.max_side || h > lim.max_side)
        throw ImageError("too_large", "image is " + std::to_string(w) + "x" + std::to_string(h) + " px; each side must be at most " + std::to_string(lim.max_side));
    alpha = f.has_alpha;
    uint8_t* out = (uint8_t*)std::malloc((size_t)w * h * 4);
    if (!out) throw ImageError("decode", "out of memory");
    if (!f.has_animation) {
        if (!WebPDecodeRGBAInto(data, n, out, (size_t)w * h * 4, w * 4)) { std::free(out); throw ImageError("decode", "cannot decode WebP image (corrupt or truncated)"); }
        return out;
    }
    std::free(out);
    WebPData wd{data, n};
    WebPAnimDecoderOptions opt;
    WebPAnimDecoderOptionsInit(&opt);
    opt.color_mode = MODE_RGBA;
    opt.use_threads = 0;
    WebPAnimDecoder* dec = WebPAnimDecoderNew(&wd, &opt);
    if (!dec) throw ImageError("decode", "cannot decode animated WebP image (corrupt or truncated)");
    uint8_t* buf = nullptr;
    int ts = 0;
    uint8_t* res = nullptr;
    if (WebPAnimDecoderGetNext(dec, &buf, &ts) && buf) {  // first frame only
        res = (uint8_t*)std::malloc((size_t)w * h * 4);
        if (res) std::memcpy(res, buf, (size_t)w * h * 4);
    }
    WebPAnimDecoderDelete(dec);
    if (!res) throw ImageError("decode", "cannot decode the first frame of the animated WebP image");
    alpha = true;
    return res;
}

Rgb8 decode_image(const uint8_t* data, size_t n, const Limits& lim) {
    if (n == 0) throw ImageError("decode", "empty image data");
    if (n > lim.max_bytes) throw ImageError("too_large", "image data exceeds the " + std::to_string(lim.max_bytes >> 20) + " MiB limit");
    int w = 0, h = 0, comp = 0;
    uint8_t* p = nullptr;
    bool alpha = false, webp = false;
    if (is_webp(data, n)) {
        webp = true;
        p = decode_webp(data, n, lim, w, h, alpha);
    } else {
        if (!stbi_info_from_memory(data, (int)n, &w, &h, &comp)) {
            const std::string fmt = unsupported_format(data, n);
            if (!fmt.empty())
                throw ImageError("decode", "unsupported image format '" + fmt + "' (supported: " + supported_list() + "); convert the image first");
            if (looks_like_text(data, n))
                throw ImageError("decode", "image data is not an image (starts with " + preview(data, n) + "); supported formats: " + supported_list());
            throw ImageError("decode", "unrecognised or corrupt image data (starts with " + preview(data, n) + "); supported formats: " + supported_list() +
                                           " (stb: " + stbi_failure_reason() + ")");
        }
        if (w <= 0 || h <= 0) throw ImageError("decode", "image has zero size");
        if (w > lim.max_side || h > lim.max_side)
            throw ImageError("too_large", "image is " + std::to_string(w) + "x" + std::to_string(h) + " px; each side must be at most " + std::to_string(lim.max_side));
        alpha = comp == 2 || comp == 4;
        int w2, h2, c2;
        p = stbi_load_from_memory(data, (int)n, &w2, &h2, &c2, alpha ? 4 : 3);
        if (!p) throw ImageError("decode", std::string("cannot decode image: ") + stbi_failure_reason());
        w = w2; h = h2;
    }
    const int ch = webp ? 4 : (alpha ? 4 : 3);
    Rgb8 r;
    r.w = w; r.h = h;
    r.px.resize((size_t)w * h * 3);
    if (!alpha) {
        if (ch == 3) std::memcpy(r.px.data(), p, r.px.size());
        else for (size_t i = 0; i < (size_t)w * h; ++i) std::memcpy(&r.px[i * 3], p + i * 4, 3);
    } else {  // HF Inkling2ImageProcessor.convert_to_rgb: composite onto the 8 px checkerboard (204 / 153), integer arithmetic
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                const uint8_t* s = p + ((size_t)y * w + x) * 4;
                const int bg = (((y / 8) + (x / 8)) % 2 == 0) ? 204 : 153, a = s[3];
                for (int c = 0; c < 3; ++c) r.px[((size_t)y * w + x) * 3 + c] = (uint8_t)((s[c] * a + bg * (255 - a) + 127) / 255);
            }
    }
    if (webp) std::free(p); else stbi_image_free(p);
    return r;
}

Pixels preprocess(const Rgb8& img, const VisionConfig& cfg) {
    const int T = cfg.tile();
    Pixels o;
    o.rows = (img.h + T - 1) / T;
    o.cols = (img.w + T - 1) / T;
    o.data.assign((size_t)o.rows * o.cols * T * T * 3, 0.f);
    const float pad = (-1.f / 255.f - 0.5f) / 0.5f;  // fill_value -1 on the 0..255 scale, then rescale + normalise (as HF)
    const float rs = 0.00392156862745098f;
    for (int ty = 0; ty < o.rows; ++ty)
        for (int tx = 0; tx < o.cols; ++tx) {
            float* dst = o.data.data() + ((size_t)ty * o.cols + tx) * T * T * 3;
            for (int y = 0; y < T; ++y)
                for (int x = 0; x < T; ++x) {
                    const int iy = ty * T + y, ix = tx * T + x;
                    for (int c = 0; c < 3; ++c) {
                        float v;
                        if (iy < img.h && ix < img.w) v = ((float)img.px[((size_t)iy * img.w + ix) * 3 + c] * rs - 0.5f) / 0.5f;
                        else v = pad;
                        dst[(y * T + x) * 3 + c] = v;
                    }
                }
        }
    return o;
}

std::vector<uint8_t> base64_decode(const std::string& s) {
    static int8_t tab[256];
    static bool init = false;
    if (!init) {
        std::memset(tab, -1, sizeof tab);
        const char* a = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (int i = 0; i < 64; ++i) tab[(uint8_t)a[i]] = (int8_t)i;
        tab[(uint8_t)'-'] = 62; tab[(uint8_t)'_'] = 63;  // url-safe alphabet too
        init = true;
    }
    std::vector<uint8_t> out;
    out.reserve(s.size() * 3 / 4);
    uint32_t acc = 0;
    int bits = 0;
    size_t nchars = 0, pos = 0;
    for (unsigned char ch : s) {
        ++pos;
        if (ch == '=' || ch == '\n' || ch == '\r' || ch == ' ') continue;
        const int v = tab[ch];
        if (v < 0) {
            char b[96];
            std::snprintf(b, sizeof b, "invalid base64 in image data URL (unexpected byte 0x%02x at offset %zu)", ch, pos - 1);
            std::string m = b;
            if (ch == '%') m += "; the payload looks percent-encoded, send plain base64";
            throw ImageError("bad_base64", m);
        }
        ++nchars;
        acc = (acc << 6) | (uint32_t)v;
        bits += 6;
        if (bits >= 8) { bits -= 8; out.push_back((uint8_t)((acc >> bits) & 0xff)); }
    }
    if (nchars == 0) throw ImageError("bad_base64", "image data URL has an empty base64 payload");
    if (nchars % 4 == 1) throw ImageError("bad_base64", "invalid base64 in image data URL (truncated: " + std::to_string(nchars) + " characters is not a valid length)");
    return out;
}

static std::string percent_decode(const std::string& s) {
    std::string o;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '%' && i + 2 < s.size() + 0 && std::isxdigit((unsigned char)s[i + 1]) && std::isxdigit((unsigned char)s[i + 2])) {
            o.push_back((char)std::strtol(s.substr(i + 1, 2).c_str(), nullptr, 16));
            i += 2;
        } else o.push_back(s[i]);
    }
    return o;
}

static bool addr_is_public(const sockaddr* sa) {
    if (sa->sa_family == AF_INET) {
        const uint32_t a = ntohl(((const sockaddr_in*)sa)->sin_addr.s_addr);
        const uint8_t b0 = a >> 24, b1 = (a >> 16) & 255;
        if (b0 == 0 || b0 == 10 || b0 == 127 || b0 >= 224) return false;
        if (b0 == 100 && b1 >= 64 && b1 <= 127) return false;  // CGNAT
        if (b0 == 169 && b1 == 254) return false;              // link local (cloud metadata)
        if (b0 == 172 && b1 >= 16 && b1 <= 31) return false;
        if (b0 == 192 && b1 == 168) return false;
        if (b0 == 192 && b1 == 0 && ((a >> 8) & 255) == 0) return false;
        if (b0 == 198 && (b1 == 18 || b1 == 19)) return false;
        return true;
    }
    if (sa->sa_family == AF_INET6) {
        const uint8_t* b = ((const sockaddr_in6*)sa)->sin6_addr.s6_addr;
        static const uint8_t zero[16] = {0};
        if (!std::memcmp(b, zero, 16)) return false;
        if (!std::memcmp(b, zero, 15) && b[15] == 1) return false;  // ::1
        if ((b[0] & 0xfe) == 0xfc) return false;                    // fc00::/7
        if (b[0] == 0xfe && (b[1] & 0xc0) == 0x80) return false;   // fe80::/10
        if (b[0] == 0xff) return false;
        static const uint8_t v4m[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff};  // ::ffff:a.b.c.d
        if (!std::memcmp(b, v4m, 12)) {
            sockaddr_in s4{};
            s4.sin_family = AF_INET;
            std::memcpy(&s4.sin_addr, b + 12, 4);
            return addr_is_public((const sockaddr*)&s4);
        }
        return true;
    }
    return false;
}

// http(s) fetch: scheme/host validated, ALL resolved addresses must be public, the connection is pinned to the checked address with
// `curl --resolve` (no DNS rebinding), redirects are refused, size and time are capped. Needs the `curl` binary (no TLS library is linked).
static std::vector<uint8_t> fetch_remote(const std::string& url, const SourceOptions& opt) {
    const bool https = url.rfind("https://", 0) == 0;
    std::string rest = url.substr(https ? 8 : 7);
    const size_t slash = rest.find_first_of("/?#");
    std::string auth = rest.substr(0, slash);
    if (auth.find('@') != std::string::npos) throw ImageError("forbidden", "credentials in image URLs are not allowed");
    std::string host = auth, port = https ? "443" : "80";
    if (!host.empty() && host[0] == '[') {
        const size_t e = host.find(']');
        if (e == std::string::npos) throw ImageError("bad_url", "bad IPv6 host in image URL");
        if (e + 1 < host.size() && host[e + 1] == ':') port = host.substr(e + 2);
        host = host.substr(1, e - 1);
    } else if (const size_t c = host.rfind(':'); c != std::string::npos) {
        port = host.substr(c + 1);
        host = host.substr(0, c);
    }
    if (host.empty() || port.empty() || port.size() > 5 || port.find_first_not_of("0123456789") != std::string::npos) throw ImageError("bad_url", "bad host or port in image URL");
    addrinfo hints{}, *res = nullptr;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host.c_str(), port.c_str(), &hints, &res) != 0 || !res) throw ImageError("fetch", "cannot resolve image host");
    std::string pin;
    for (addrinfo* a = res; a; a = a->ai_next) {
        if (!addr_is_public(a->ai_addr)) { freeaddrinfo(res); throw ImageError("forbidden", "image host resolves to a private, loopback or link-local address (refused)"); }
        if (pin.empty()) {
            char buf[INET6_ADDRSTRLEN];
            const void* src = a->ai_family == AF_INET ? (const void*)&((sockaddr_in*)a->ai_addr)->sin_addr : (const void*)&((sockaddr_in6*)a->ai_addr)->sin6_addr;
            inet_ntop(a->ai_family, src, buf, sizeof buf);
            pin = a->ai_family == AF_INET6 ? std::string("[") + buf + "]" : buf;
        }
    }
    freeaddrinfo(res);
    int fds[2];
    if (pipe(fds) != 0) throw ImageError("fetch", "pipe failed");
    const std::string resolve = host + ":" + port + ":" + pin, maxsize = std::to_string(opt.limits.max_bytes), tmo = std::to_string(opt.fetch_timeout_s);
    std::vector<std::string> args = {"curl", "-sS", "-f", "--proto", https ? "=https" : "=http,https", "--max-redirs", "0", "--max-filesize", maxsize, "--max-time", tmo,
                                     "--resolve", resolve, "--", url};
    std::vector<char*> argv;
    for (auto& a : args) argv.push_back(a.data());
    argv.push_back(nullptr);
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_adddup2(&fa, fds[1], 1);
    posix_spawn_file_actions_addclose(&fa, fds[0]);
    posix_spawn_file_actions_addopen(&fa, 2, "/dev/null", O_WRONLY, 0);
    pid_t pid;
    const int rc = posix_spawnp(&pid, "curl", &fa, nullptr, argv.data(), environ);
    posix_spawn_file_actions_destroy(&fa);
    close(fds[1]);
    if (rc != 0) { close(fds[0]); throw ImageError("fetch", "cannot run curl (needed for http(s) image URLs)"); }
    std::vector<uint8_t> out;
    uint8_t buf[65536];
    ssize_t n;
    while ((n = read(fds[0], buf, sizeof buf)) > 0) {
        out.insert(out.end(), buf, buf + n);
        if (out.size() > opt.limits.max_bytes) { kill(pid, SIGKILL); break; }
    }
    close(fds[0]);
    int st = 0;
    waitpid(pid, &st, 0);
    if (out.size() > opt.limits.max_bytes) throw ImageError("too_large", "remote image exceeds the size limit");
    if (!WIFEXITED(st) || WEXITSTATUS(st) != 0) throw ImageError("fetch", "failed to fetch the image URL (http error, redirect, timeout or size limit)");
    return out;
}

std::vector<uint8_t> load_image_bytes(const std::string& url, const SourceOptions& opt) {
    if (url.rfind("data:", 0) == 0) {
        const size_t comma = url.find(',');
        if (comma == std::string::npos) throw ImageError("bad_url", "malformed data URL (no comma)");
        const std::string head = url.substr(5, comma - 5);
        if (head.find(";base64") == std::string::npos) throw ImageError("bad_url", "only base64 data URLs are supported for images");
        if (head.rfind("image/", 0) != 0 && !head.empty() && head[0] != ';') throw ImageError("bad_url", "data URL media type must be image/*");
        if ((url.size() - comma) / 4 * 3 > opt.limits.max_bytes + 8) throw ImageError("too_large", "image data exceeds the size limit");
        return base64_decode(url.substr(comma + 1));
    }
    if (url.rfind("http://", 0) == 0 || url.rfind("https://", 0) == 0) {
        if (!opt.allow_remote) throw ImageError("forbidden", "remote image URLs are disabled (start the server with --allow-remote-images); use a data: URL or a local file");
        return fetch_remote(url, opt);
    }
    std::string path = url;
    if (path.rfind("file://", 0) == 0) path = percent_decode(path.substr(7));
    else if (path.find("://") != std::string::npos) throw ImageError("bad_url", "unsupported image URL scheme");
    if (opt.local_root.empty()) throw ImageError("forbidden", "local image files are disabled (start the server with --image-root DIR)");
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::path root = fs::weakly_canonical(opt.local_root, ec);
    fs::path full = fs::weakly_canonical(fs::path(path).is_absolute() ? fs::path(path) : root / path, ec);
    if (ec) throw ImageError("bad_url", "cannot resolve the image path");
    auto rel = full.lexically_relative(root);
    if (rel.empty() || *rel.begin() == "..") throw ImageError("forbidden", "image path is outside the allowed image root");
    if (!fs::is_regular_file(full, ec)) throw ImageError("bad_url", "image file not found");
    if (fs::file_size(full, ec) > opt.limits.max_bytes) throw ImageError("too_large", "image file exceeds the size limit");
    std::ifstream f(full, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

}  // namespace qv::vision
