// qv_tok — tokenizer CLI.
//   qv_tok TOKENIZER_JSON encode   : stdin lines are JSON strings -> stdout one JSON id array per line
//   qv_tok TOKENIZER_JSON decode   : stdin lines are JSON id arrays -> stdout one JSON string per line
//   TOKENIZER_JSON may be a .qv model file: its embedded tokenizer.json blob is used (docs/QV_FORMAT.md)
//   (add "nospecial" as 3rd arg to disable literal special-token parsing in encode)
#include <cstdio>
#include <iostream>
#include <string>

#include "core/json_doc.h"
#include "qv/model.h"
#include "tokenizer/chat_template.h"
#include "tokenizer/tokenizer.h"

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: qv_tok tokenizer.json encode|decode|chat [nospecial]\n"); return 2; }
    std::string tpath = argv[1];
    qv::Tokenizer tok;
    if (tpath.size() > 3 && tpath.compare(tpath.size() - 3, 3, ".qv") == 0) {
        try {
            qv::ModelFile mf = qv::ModelFile::open(tpath);
            if (!mf.has_blob("tokenizer.json")) { std::fprintf(stderr, "qv_tok: %s has no embedded tokenizer.json (tools/qv_embed.py --list)\n", tpath.c_str()); return 1; }
            tok = qv::Tokenizer::from_memory(mf.blob("tokenizer.json"));
        } catch (std::exception& e) { std::fprintf(stderr, "qv_tok: %s\n", e.what()); return 1; }
    } else {
        tok = qv::Tokenizer::load(tpath);
    }
    std::string mode = argv[2];
    bool special = !(argc > 3 && std::string(argv[3]) == "nospecial");
    std::string line;
    while (std::getline(std::cin, line)) {
        auto v = qv::jd::parse(line);
        if (mode == "chat") {  // {"messages":[...], "enable_thinking":bool?, "reasoning_effort":str?, "preserve_thinking":bool?, "add_generation_prompt":bool?}
            qv::ChatTemplateOptions o;
            if (auto* x = v.find("enable_thinking")) o.enable_thinking = x->as_bool();
            if (auto* x = v.find("preserve_thinking")) o.preserve_thinking = x->as_bool();
            if (auto* x = v.find("reasoning_effort")) o.reasoning_effort = x->as_string();
            if (auto* x = v.find("add_generation_prompt")) o.add_generation_prompt = x->as_bool();
            qv::jd::Value r = qv::jd::Value::object();
            try { r.set("text", qv::render_chat(qv::messages_from_json(*v.find("messages")), o)); }
            catch (std::exception& e) { r.set("error", e.what()); }
            std::printf("%s\n", r.dump().c_str());
        } else if (mode == "encode") {
            auto ids = tok.encode(v.as_string(), special);
            qv::jd::Value a = qv::jd::Value::array();
            for (int i : ids) a.push(i);
            std::printf("%s\n", a.dump().c_str());
        } else {
            std::vector<int> ids;
            for (auto& x : v.items()) ids.push_back((int)x.as_int());
            std::printf("%s\n", qv::jd::Value(tok.decode(ids)).dump().c_str());
        }
        std::fflush(stdout);
    }
}
