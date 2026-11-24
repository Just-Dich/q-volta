// Inkling-2-Mini vision (image input): preprocessing + ViT encoder + projector. Spec: docs/INKLING_VISION.md.
// Backend-independent API; the CUDA encoder lives in vision_cuda.cu (qv_vision_cuda), the CPU encoder in vision_cpu.cpp.
#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace qv::vision {

struct VisionConfig {
    int hidden = 1024, layers = 23, heads = 16, mlp = 4096, patch = 20, merge = 2, grid = 32, proj = 4096, text_hidden = 2304;
    float eps = 1e-5f, rope_theta = 10000.f, layer_scale_init = 0.1f;
    int image_token_id = 200054;
    int min_side = 40, max_side = 4096;
    int tile() const { return patch * merge; }  // pixels per image token side (40)
};

// Error raised for bad / oversize / undecodable images. `kind` is a stable short code used by the server for its message.
struct ImageError : std::runtime_error {
    std::string kind;  // "decode" | "too_large" | "bad_url" | "forbidden" | "fetch"
    ImageError(const std::string& k, const std::string& msg) : std::runtime_error(msg), kind(k) {}
};

struct Rgb8 {
    int w = 0, h = 0;
    std::vector<uint8_t> px;  // h*w*3
};

struct Limits {
    int max_side = 4096;                 // reject larger images (before allocating pixels)
    size_t max_bytes = 32u << 20;        // encoded size cap
};

// Decode PNG/JPEG/BMP/GIF/TGA/PSD/PNM/PIC via stb_image and WebP (lossy/lossless/alpha/animated first frame) via libwebp; alpha is composited on the 8 px checkerboard (204/153) exactly like the HF processor.
Rgb8 decode_image(const uint8_t* data, size_t n, const Limits& lim);

// pixel_values of the HF Inkling2ImageProcessor: [rows*cols][40][40][3] float in [-1,1] (padding = -1 on the 0..255 scale, bottom/right).
struct Pixels {
    int rows = 0, cols = 0;
    std::vector<float> data;
    int tokens() const { return rows * cols; }
};
Pixels preprocess(const Rgb8& img, const VisionConfig& cfg = VisionConfig());
inline int image_tokens(int w, int h, int tile = 40) { return ((h + tile - 1) / tile) * ((w + tile - 1) / tile); }

// ---- image sources (OpenAI `image_url.url`) ----
struct SourceOptions {
    bool allow_remote = false;           // http(s) fetch (SSRF-guarded, see docs/INKLING_VISION.md section 10)
    std::string local_root;              // local files / file:// URLs are only read below this directory ("" = disabled)
    Limits limits;
    int fetch_timeout_s = 20;
};
std::vector<uint8_t> load_image_bytes(const std::string& url, const SourceOptions& opt);  // data: URL | file | http(s)
std::vector<uint8_t> base64_decode(const std::string& s);

// ---- encoder ----
class Encoder {
   public:
    virtual ~Encoder() = default;
    const VisionConfig& cfg() const { return cfg_; }
    virtual const char* backend() const = 0;
    // out: pix.tokens() * text_hidden floats (the embeddings that REPLACE the placeholder positions of the text residual stream).
    virtual void encode(const Pixels& pix, float* out) = 0;
    // CUDA only: milliseconds of the last encode (device time incl. H2D/D2H); CPU: wall time
    double last_ms = 0;
    // Create from a sidecar file; cuda=true needs the qv_vision_cuda library (make_cuda_encoder).
    static std::unique_ptr<Encoder> create(const std::string& sidecar, bool cuda);

   protected:
    VisionConfig cfg_;
};

// Sidecar container (tools/convert_inkling_vision.py): tensor access with fp16/int8 -> float helpers.
struct VisionFile {
    struct T { std::vector<int64_t> shape; std::string kind; uint64_t off = 0, nbytes = 0, soff = 0; };
    VisionConfig cfg;
    std::string path, quant;
    std::vector<std::pair<std::string, T>> list;
    uint64_t data_off = 0;
    const T& at(const std::string& n) const;
    static std::unique_ptr<VisionFile> open(const std::string& path);
    // dense float copy of a tensor (f16/f32/i8 expanded)
    std::vector<float> floats(const std::string& n) const;
    // fp16 bits of a tensor (f16 as stored; i8 expanded; f32 converted)
    std::vector<uint16_t> half(const std::string& n) const;
};
float half_to_float(uint16_t h);
uint16_t float_to_half(float f);

std::unique_ptr<Encoder> make_cpu_encoder(std::shared_ptr<VisionFile> f);
std::unique_ptr<Encoder> make_cuda_encoder(std::shared_ptr<VisionFile> f);  // defined only when linked with qv_vision_cuda (else returns nullptr)

}  // namespace qv::vision
