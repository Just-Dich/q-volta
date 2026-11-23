// Container blobs / meta (docs/QV_FORMAT.md): round trip, 64-byte alignment, back-compat (no blobs), unknown keys ignored,
// truncated blob rejected, Tokenizer::from_memory on the embedded bytes. Builds synthetic containers in a temp dir (no model needed).
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include "qv/model.h"
#include "tokenizer/tokenizer.h"

using namespace qv;
static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "CHECK failed %s:%d: %s\n", __FILE__, __LINE__, #c); fails++; } } while (0)

// tiny byte-level BPE tokenizer.json: the 256 GPT-2 byte symbols (ids 0..255), merge "a b" -> "ab" (256), added token <|im_end|> (257)
static std::string make_tok() {
    auto utf8 = [](unsigned cp) { std::string s; if (cp < 0x80) s += (char)cp; else { s += (char)(0xC0 | (cp >> 6)); s += (char)(0x80 | (cp & 63)); } return s; };
    std::string v = "{";
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        bool pr = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
        std::string sym = utf8(pr ? (unsigned)b : 256u + n++);
        if (sym == "\"" || sym == "\\") sym = "\\" + sym;
        v += (b ? ",\"" : "\"") + sym + "\":" + std::to_string(b);
    }
    v += ",\"ab\":256}";
    return R"({"model":{"type":"BPE","vocab":)" + v + R"(,"merges":["a b"]},"added_tokens":[{"id":257,"content":"<|im_end|>","special":true}]})";
}

// header JSON text + data bytes -> container file (data_off = 2 MiB like the real writers)
static void write_qv(const std::string& path, const std::string& json, const std::string& data) {
    const uint64_t doff = 2u << 20, jl = json.size();
    std::ofstream f(path, std::ios::binary);
    f.write("QVOLTA1\0", 8);
    f.write((const char*)&jl, 8);
    f.write((const char*)&doff, 8);
    f.write(json.data(), (std::streamsize)jl);
    std::string pad(doff - 24 - jl, '\0');
    f.write(pad.data(), (std::streamsize)pad.size());
    f.write(data.data(), (std::streamsize)data.size());
}

static const char* kCfg = R"("config":{"arch":"inkling2_mtp","hidden_size":8,"num_hidden_layers":1,"vocab_size":4,"num_attention_heads":1,"num_key_value_heads":1,"head_dim":8,"rms_norm_eps":1e-6,"eos_token_id":3})";

int main() {
    namespace fs = std::filesystem;
    fs::path dir = fs::temp_directory_path() / "qv_test_blobs";
    fs::create_directories(dir);
    const std::string tok = make_tok(), tmpl = "{{ messages }} \xE2\x82\xAC", gen = R"({"temperature":0.6,"top_k":20,"eos_token_id":[3,2]})";
    // data area: tensor "t" (32 bytes) | pad to 64 | tokenizer | pad to 64 | template | pad | generation_config
    std::string data(32, 'T');
    auto put = [&](const std::string& b) { data.resize((data.size() + 63) / 64 * 64, '\0'); uint64_t o = data.size(); data += b; return o; };
    uint64_t o_tok = put(tok), o_tmpl = put(tmpl), o_gen = put(gen);
    std::ostringstream js;
    js << R"({"format":"qvolta-v1","format_version":2,)" << kCfg << R"(,"tensors":{"t":{"kind":"f32","shape":[8],"off":0,"nbytes":32}},"blobs":{)"
       << R"("tokenizer.json":{"off":)" << o_tok << R"(,"nbytes":)" << tok.size() << R"(},"chat_template.jinja":{"off":)" << o_tmpl << R"(,"nbytes":)" << tmpl.size()
       << R"(},"generation_config.json":{"off":)" << o_gen << R"(,"nbytes":)" << gen.size() << R"(},"future.thing":{"off":0,"nbytes":4}},)"
       << R"("meta":{"model_name":"unit","context_length":4096,"eos_token_ids":[3],"nested":{"x":[1,2]},"some_future_key":true}})";
    const std::string p = (dir / "a.qv").string();
    write_qv(p, js.str(), data);
    {
        ModelFile m = ModelFile::open(p);
        CHECK(m.format_version == 2);
        CHECK(m.blobs.size() == 4);
        CHECK(m.has_blob("tokenizer.json") && !m.has_blob("nope"));
        CHECK(m.blob("tokenizer.json") == tok);
        CHECK(m.blob("chat_template.jinja") == tmpl);
        CHECK(m.blob("generation_config.json") == gen);
        CHECK(m.blob("nope").empty());
        CHECK(m.blob("future.thing").size() == 4);  // unknown names are kept but nothing interprets them
        for (auto& [n, b] : m.blobs) if (n != "future.thing") CHECK((b.off - (2u << 20)) % 64 == 0);
        CHECK(m.meta_string("model_name") == "unit");
        CHECK(m.meta().find("\"context_length\":4096") != std::string::npos);
        CHECK(m.meta().find("some_future_key") != std::string::npos);
        ModelFile copy = m;  // copies made after the first blob() call share the mappings
        std::string_view v = copy.blob("chat_template.jinja");
        CHECK(v == tmpl && copy.blob("chat_template.jinja").data() == v.data());
        Tokenizer t = Tokenizer::from_memory(m.blob("tokenizer.json"));
        CHECK(t.token_id("<|im_end|>") == 257);
        CHECK(t.encode("ab", true) == std::vector<int>{256});
        CHECK(m.tensors.count("t") == 1);
    }
    {   // back-compat: a version-1 file without blobs / meta
        std::string j1 = std::string(R"({"format":"qvolta-v1",)") + kCfg + R"(,"tensors":{"t":{"kind":"f32","shape":[8],"off":0,"nbytes":32}}})";
        const std::string p1 = (dir / "v1.qv").string();
        write_qv(p1, j1, std::string(32, 'T'));
        ModelFile m = ModelFile::open(p1);
        CHECK(m.format_version == 1 && m.blobs.empty() && m.meta().empty() && m.blob("tokenizer.json").empty());
    }
    {   // a blob that points past the end of the file is a truncated model
        std::string j = std::string(R"({"format":"qvolta-v1","format_version":2,)") + kCfg + R"(,"tensors":{},"blobs":{"tokenizer.json":{"off":0,"nbytes":9999}}})";
        const std::string pt = (dir / "trunc.qv").string();
        write_qv(pt, j, std::string(64, 'x'));
        bool threw = false;
        try { ModelFile::open(pt); } catch (const std::exception&) { threw = true; }
        CHECK(threw);
    }
    fs::remove_all(dir);
    if (fails) { std::fprintf(stderr, "%d check(s) failed\n", fails); return 1; }
    std::printf("test_blobs: ok\n");
    return 0;
}
