// jinja_render TEMPLATE.jinja CTX.json [--now EPOCH]   -> rendered text on stdout; "ERROR[raise]|ERROR[other]: msg" on stderr, exit 3.
// Used by tests/jinja/diff_oracle.py to compare byte-for-byte with Python jinja2.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>

#include "jinja/jinja.h"
#include "core/json_doc.h"

static std::string slurp(const char* p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) { fprintf(stderr, "cannot open %s\n", p); exit(2); }
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: jinja_render TEMPLATE CTX.json [--now EPOCH]\n"); return 2; }
    qv::jinja::Environment env;
    for (int i = 3; i + 1 < argc; ++i)
        if (!strcmp(argv[i], "--now")) { long t = atol(argv[i + 1]); env.set_clock([t] { return (std::time_t)t; }); }
    try {
        auto tpl = env.compile(slurp(argv[1]), argv[1]);
        auto ctx = qv::jinja::from_json(qv::jd::parse(slurp(argv[2])));
        std::string out = tpl.render(ctx);
        fwrite(out.data(), 1, out.size(), stdout);
        return 0;
    } catch (const qv::jinja::RaisedError& e) {
        fprintf(stderr, "ERROR[raise]: %s", e.what());
        return 3;
    } catch (const std::exception& e) {
        fprintf(stderr, "ERROR[other]: %s", e.what());
        return 3;
    }
}
