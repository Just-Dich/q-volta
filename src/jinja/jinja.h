// Jinja2-compatible template engine (C++20, no third-party deps). Targets the subset used by HF/llama.cpp chat templates
// with Python-exact output semantics (str()/repr()/json.dumps formatting, whitespace control, trim_blocks/lstrip_blocks).
//
//   qv::jinja::Environment env;                       // defaults = HF chat-template settings (trim_blocks, lstrip_blocks)
//   auto tpl = env.compile(template_source);          // throws jinja::Error on syntax errors
//   Value ctx = Value::dict(); ctx.set("messages", ...);
//   std::string out = tpl.render(ctx);                // throws jinja::Error / RaisedError (raise_exception)
#pragma once
#include <ctime>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>

#include "jinja/value.h"

namespace qv::jinja {

struct EnvImpl;
struct TemplateAst;

struct EnvOptions {
    bool trim_blocks = true;
    bool lstrip_blocks = true;
    bool keep_trailing_newline = false;
};

using FilterFn = std::function<Value(EnvImpl&, const Value& self, Args& args)>;
using TestFn = std::function<bool(EnvImpl&, const Value& self, Args& args)>;

class Template {
   public:
    Template() = default;
    // ctx: a dict Value (top-level template variables). Output is a UTF-8 string.
    std::string render(const Value& ctx) const;
    const std::string& name() const { return name_; }
    bool valid() const { return ast_ != nullptr; }

   private:
    friend class Environment;
    std::shared_ptr<EnvImpl> env_;
    std::shared_ptr<TemplateAst> ast_;
    std::string name_;
};

class Environment {
   public:
    explicit Environment(EnvOptions opt = {});
    Template compile(std::string_view source, const std::string& name = "<template>") const;

    void set_global(const std::string& name, Value v);
    void add_filter(const std::string& name, FilterFn f);
    void add_test(const std::string& name, TestFn f);
    // strftime_now(): defaults to the wall clock; tests pin it.
    void set_clock(std::function<std::time_t()> now);

   private:
    std::shared_ptr<EnvImpl> impl_;
};

}  // namespace qv::jinja
