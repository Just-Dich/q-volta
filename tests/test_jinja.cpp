// Unit + golden tests for src/jinja (engine, chat rendering, output parser).
//   test_jinja [DIR_WITH_tests/jinja]    (default: QV_JINJA_TEST_DIR)
#include <cstdio>
#include <fstream>
#include <sstream>

#include "jinja/chat.h"
#include "jinja/jinja.h"
#include "tokenizer/chat_template.h"

using namespace qv;
static int g_fail = 0, g_pass = 0;
#define CHECK(c)                                                                  \
    do {                                                                          \
        if (c) ++g_pass;                                                          \
        else { ++g_fail; fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); } \
    } while (0)
#define CHECK_EQ(a, b)                                                                                                         \
    do {                                                                                                                       \
        auto _a = (a);                                                                                                         \
        auto _b = (b);                                                                                                         \
        if (_a == _b) ++g_pass;                                                                                                \
        else { ++g_fail; fprintf(stderr, "FAIL %s:%d: %s\n  got:      %s\n  expected: %s\n", __FILE__, __LINE__, #a, show(_a).c_str(), show(_b).c_str()); } \
    } while (0)

static std::string show(const std::string& s) { return jinja::Value(s).repr(); }
static std::string show(bool b) { return b ? "true" : "false"; }
static std::string show(size_t b) { return std::to_string(b); }
static std::string show(int b) { return std::to_string(b); }

static std::string slurp(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static std::string R(const std::string& tpl, const std::string& ctx_json = "{}") {
    jinja::Environment env;
    return env.compile(tpl).render(jinja::from_json(jd::parse(ctx_json)));
}
static std::string ERR(const std::string& tpl, const std::string& ctx_json = "{}") {
    try { R(tpl, ctx_json); } catch (const jinja::RaisedError& e) { return std::string("raise:") + e.what(); } catch (const jinja::Error& e) { return "error"; }
    return "no-error";
}

static void test_engine() {
    CHECK_EQ(R("{{ 1 + 2 * 3 }}"), "7");
    CHECK_EQ(R("{{ 7 // 2 }} {{ -7 // 2 }} {{ 7 % 3 }} {{ -7 % 3 }} {{ 2 ** 10 }} {{ 1 / 4 }}"), "3 -4 1 2 1024 0.25");
    CHECK_EQ(R("{{ 1.0 }} {{ 1e16 }} {{ 0.1 + 0.2 }} {{ 1e-5 }}"), "1.0 1e+16 0.30000000000000004 1e-05");
    CHECK_EQ(R("{{ [1, 'a', none, true, 2.5] }} {{ {'a': (1, 2)} }}"), "[1, 'a', None, True, 2.5] {'a': (1, 2)}");
    CHECK_EQ(R("{{ 'x' ~ 1 ~ none }} {{ 'a' if 0 else 'b' }} [{{ 'a' if 0 }}]"), "x1None b []");
    CHECK_EQ(R("{{ s[1:3] }} {{ s[::-1] }} {{ l[-1] }} {{ l[1:] }} {{ s[-2:] }}", R"({"s":"héllo","l":[1,2,3]})"), "él olléh 3 [2, 3] lo");
    CHECK_EQ(R("{{ d | tojson }} {{ d | tojson(indent=1) }} {{ d | tojson(sort_keys=true, separators=(',', ':')) }}", R"({"d":{"b":[1,2],"a":"é\n"}})"),
             "{\"b\": [1, 2], \"a\": \"é\\n\"} {\n \"b\": [\n  1,\n  2\n ],\n \"a\": \"é\\n\"\n} {\"a\":\"é\\n\",\"b\":[1,2]}");
    CHECK_EQ(R("{% for x in l %}{{ loop.index }}{{ x }}{{ ',' if not loop.last }}{% endfor %}", R"({"l":["a","b","c"]})"), "1a,2b,3c");
    CHECK_EQ(R("{% for x in [] %}x{% else %}empty{% endfor %}"), "empty");
    CHECK_EQ(R("{% set ns = namespace(n=0) %}{% for i in range(4) %}{% set ns.n = ns.n + i %}{% endfor %}{{ ns.n }}"), "6");
    CHECK_EQ(R("{% macro m(a, b=2) %}{{ a }}-{{ b }}{% endmacro %}{{ m(1) }} {{ m(1, b=3) }}"), "1-2 1-3");
    CHECK_EQ(R("{% macro w() %}[{{ caller() }}]{% endmacro %}{% call w() %}x{% endcall %}"), "[x]");
    CHECK_EQ(R("{% set x %}a{{ 1 }}{% endset %}{{ x | upper }}"), "A1");
    CHECK_EQ(R("{{ 'a,b,,c'.split(',') }} {{ ' x '.strip() }}| {{ 'abc'.startswith('ab') }} {{ 'a b'.title() }}"), "['a', 'b', '', 'c'] x| True A B");
    CHECK_EQ(R("{{ v is defined }} {{ n is none }} {{ s is string }} {{ l is iterable }} {{ d is mapping }} {{ x is number }}", R"({"n":null,"s":"a","l":[1],"d":{},"x":1.5,"v":0})"),
             "True True True True True True");
    CHECK_EQ(R("{{ u is defined }} {{ u | default('d') }} {{ '' | default('e', true) }}"), "False d e");
    CHECK_EQ(R("{{ objs | map(attribute='n') | join(',') }} {{ objs | selectattr('k') | list | length }} {{ objs | rejectattr('k', 'equalto', 1) | list | length }}",
               R"({"objs":[{"n":"a","k":1},{"n":"b","k":0},{"n":"c","k":2}]})"),
             "a,b,c 2 2");
    // whitespace control
    CHECK_EQ(R("a  {%- if true -%}  b  {%- endif -%}  c"), "abc");
    CHECK_EQ(R("x\n{% for i in [1,2] %}\n{{ i }}\n{% endfor %}\ny\n"), "x\n1\n2\ny");
    CHECK_EQ(R("  {% if true %}\n  in\n  {% endif %}\n"), "  in\n");
    CHECK_EQ(R("{% raw %}{{ x }}{% endraw %}|{# c #}z"), "{{ x }}|z");
    // errors
    CHECK_EQ(ERR("{{ raise_exception('bad: ' ~ 1) }}"), "raise:bad: 1");
    CHECK_EQ(ERR("{{ 1 / 0 }}"), "error");
    CHECK_EQ(ERR("{% if %}"), "error");
    CHECK_EQ(ERR("{{ u.x }}"), "error");
    // strftime_now with pinned clock
    jinja::Environment env;
    env.set_clock([] { return (std::time_t)1767323045; });
    setenv("TZ", "UTC", 1);
    tzset();
    CHECK_EQ(env.compile("{{ strftime_now('%Y-%m-%d %H:%M:%S') }}").render(jinja::Value::dict()), "2026-01-02 03:04:05");
}

// ------------------------------------------------------------------------------ golden (Python jinja2 oracle outputs)
static void test_golden(const std::string& dir) {
    std::string tdir = dir + "/templates/";
    int files = 0, cases = 0, bad = 0;
    for (const char* name : {"Qwen3.8-27B", "Inkling-2-Mini", "Qwen3-Coder", "NousResearch-Hermes-2-Pro-Llama-3-8B-tool_use", "meta-llama-Llama-3.1-8B-Instruct",
                              "mistralai-Mistral-Nemo-Instruct-2407", "google-gemma-2-2b-it", "openai-gpt-oss-120b", "deepseek-ai-DeepSeek-V3.1",
                              "Mistral-Small-3.2-24B-Instruct-2506", "Kimi-K2-Instruct"}) {
        std::string src = slurp(tdir + name + ".jinja");
        std::string gold = slurp(dir + "/golden/" + name + ".json");
        if (src.empty() || gold.empty()) { fprintf(stderr, "missing golden/template %s\n", name); ++g_fail; continue; }
        ++files;
        jd::Value g = jd::parse(gold);
        jinja::Environment env;
        long now = (long)g.find("now")->as_int();
        env.set_clock([now] { return (std::time_t)now; });
        jinja::Template tpl = env.compile(src, name);
        for (auto& c : g.find("cases")->items()) {
            ++cases;
            jinja::Value ctx = jinja::from_json(*c.find("ctx"));
            std::string got, err_kind;
            std::string err;
            try { got = tpl.render(ctx); }
            catch (const jinja::RaisedError& e) { err_kind = "raise"; err = e.what(); }
            catch (const std::exception& e) { err_kind = "other"; err = e.what(); }
            const jd::Value* out = c.find("out");
            bool ok = true;
            if (out) ok = err_kind.empty() && got == out->as_string();
            else {
                ok = !err_kind.empty();
                if (ok && c.find("error_kind")->as_string() == "raise" && err_kind == "raise") ok = err == c.find("error")->as_string();
            }
            if (!ok) { ++bad; fprintf(stderr, "golden mismatch: %s / %s\n", name, c.find("name")->as_string().c_str()); }
        }
    }
    fprintf(stderr, "golden: %d templates, %d cases, %d mismatches\n", files, cases, bad);
    CHECK(files == 11);
    CHECK(bad == 0);
}

// ------------------------------------------------------------------------------ chat helpers
static jd::Value J(const char* s) { return jd::parse(s); }

static void test_chat_render(const std::string& dir) {
    auto qwen = chat::ChatTemplate::from_file(dir + "/templates/Qwen3.8-27B.jinja");
    CHECK(qwen.tool_format() == ToolFormat::QwenXml);
    chat::RenderInputs in;
    in.messages = J(R"([{"role":"system","content":"Be brief."},{"role":"user","content":"Hi"},
                        {"role":"assistant","content":"Hello","reasoning_content":"greet"},{"role":"user","content":[{"type":"text","text":"Bye"}]}])");
    std::string p = qwen.render(in);
    CHECK(chat::ChatTemplate::prompt_forces_thinking(p));
    // cross-check against the existing hand-written C++ renderer (src/tokenizer/chat_template.cpp)
    auto msgs = messages_from_json(in.messages);
    ChatTemplateOptions o;
    CHECK_EQ(p, render_chat(msgs, o));
    in.enable_thinking = false;
    CHECK(!chat::ChatTemplate::prompt_forces_thinking(qwen.render(in)));
    o.enable_thinking = false;
    CHECK_EQ(qwen.render(in), render_chat(msgs, o));
    // OpenAI-style tool call: arguments are a JSON *string* and must reach the template as an object
    in = chat::RenderInputs();
    in.messages = J(R"([{"role":"user","content":"w?"},{"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Paris\",\"days\":3}"}}]},{"role":"tool","tool_call_id":"c1","content":"sunny"}])");
    in.tools = J(R"([{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}}}}}])");
    std::string p2 = qwen.render(in);
    CHECK(p2.find("<parameter=city>\nParis\n</parameter>") != std::string::npos);
    CHECK(p2.find("<parameter=days>\n3\n</parameter>") != std::string::npos);
    CHECK(p2.find("<tool_response>\nsunny\n</tool_response>") != std::string::npos);
    // template errors surface as RaisedError with the template's text
    try { in.messages = J("[]"); qwen.render(in); CHECK(false); } catch (const jinja::RaisedError& e) { CHECK_EQ(std::string(e.what()), "No messages provided."); }
    // extra vars
    in.messages = J(R"([{"role":"user","content":"x"}])");
    in.extra = J(R"({"reasoning_effort":"low"})");
    CHECK(qwen.render(in).find("Reasoning effort is set to low.") != std::string::npos);
}

static chat::ParseParams pp(ToolFormat f, bool forced = false) {
    chat::ParseParams p;
    p.format = f;
    p.thinking_forced_open = forced;
    return p;
}

// Streaming contract: successive partial parses of growing prefixes never retract, and diffs reassemble the final message.
static void check_stream(const std::string& text, chat::ParseParams p) {
    p.partial = true;
    chat::Message prev, acc;
    int viol = 0;
    for (size_t n = 0; n <= text.size(); ++n) {
        chat::Message cur = chat::parse_output(text.substr(0, n), p);
        auto pre = [](const std::string& a, const std::string& b) { return b.compare(0, a.size(), a) == 0; };
        if (!pre(prev.content, cur.content) || !pre(prev.reasoning_content, cur.reasoning_content)) {
            ++viol;
            if (viol == 1) fprintf(stderr, "stream retraction at %zu: content %s -> %s | reasoning %s -> %s\n", n, show(prev.content).c_str(), show(cur.content).c_str(), show(prev.reasoning_content).c_str(), show(cur.reasoning_content).c_str());
        }
        if (cur.tool_calls.size() < prev.tool_calls.size()) ++viol;
        for (size_t i = 0; i < prev.tool_calls.size() && i < cur.tool_calls.size(); ++i)
            if (!pre(prev.tool_calls[i].arguments, cur.tool_calls[i].arguments) || (!prev.tool_calls[i].name.empty() && prev.tool_calls[i].name != cur.tool_calls[i].name)) {
                ++viol;
                if (viol == 1) fprintf(stderr, "stream tool retraction at %zu: %s -> %s\n", n, show(prev.tool_calls[i].arguments).c_str(), show(cur.tool_calls[i].arguments).c_str());
            }
        chat::Delta d = chat::diff_messages(prev, cur);
        acc.content += d.content;
        acc.reasoning_content += d.reasoning;
        for (auto& t : d.tool_calls) {
            if (acc.tool_calls.size() <= t.index) acc.tool_calls.resize(t.index + 1);
            if (t.is_new) acc.tool_calls[t.index].name = t.name;
            acc.tool_calls[t.index].arguments += t.arguments;
        }
        prev = cur;
    }
    {  // end of stream: flush the hold-back (no retraction either)
        p.flush = true;
        chat::Message cur = chat::parse_output(text, p);
        chat::Delta d = chat::diff_messages(prev, cur);
        acc.content += d.content;
        acc.reasoning_content += d.reasoning;
        for (auto& t : d.tool_calls) {
            if (acc.tool_calls.size() <= t.index) acc.tool_calls.resize(t.index + 1);
            if (t.is_new) acc.tool_calls[t.index].name = t.name;
            acc.tool_calls[t.index].arguments += t.arguments;
        }
        prev = cur;
    }
    CHECK(viol == 0);
    CHECK(acc == prev);
}

static void test_parse() {
    // ---- reasoning
    {
        std::string r, c;
        chat::extract_reasoning("<think>\nabc\n</think>\n\nanswer", pp(ToolFormat::Generic), r, c);
        CHECK_EQ(r, "abc"); CHECK_EQ(c, "answer");
        chat::extract_reasoning("abc\n</think>\n\nanswer", pp(ToolFormat::Generic, true), r, c);
        CHECK_EQ(r, "abc"); CHECK_EQ(c, "answer");
        chat::extract_reasoning("unterminated thought", pp(ToolFormat::Generic, true), r, c);
        CHECK_EQ(r, "unterminated thought"); CHECK_EQ(c, "");
        chat::extract_reasoning("no think here", pp(ToolFormat::Generic), r, c);
        CHECK_EQ(r, ""); CHECK_EQ(c, "no think here");
        auto p = pp(ToolFormat::Generic); p.reasoning = chat::ReasoningFormat::None;
        chat::extract_reasoning("<think>x</think>y", p, r, c);
        CHECK_EQ(r, ""); CHECK_EQ(c, "<think>x</think>y");
        p.reasoning = chat::ReasoningFormat::DeepseekLegacy;
        chat::extract_reasoning("<think>x</think>y", p, r, c);
        CHECK_EQ(r, "x"); CHECK_EQ(c, "<think>x</think>y");
        CHECK(chat::reasoning_format_from_name("auto") == chat::ReasoningFormat::Deepseek);
    }
    // ---- Hermes
    {
        std::string t = "Let me check.\n\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\", \"n\": 1}}\n</tool_call>\n<tool_call>\n{\"name\":\"b\",\"arguments\":{}}\n</tool_call>";
        auto m = chat::parse_output(t, pp(ToolFormat::Hermes));
        CHECK_EQ(m.content, "Let me check.");
        CHECK_EQ(m.tool_calls.size(), (size_t)2);
        CHECK_EQ(m.tool_calls[0].name, "get_weather");
        CHECK_EQ(m.tool_calls[0].arguments, "{\"city\":\"Paris\",\"n\":1}");
        CHECK_EQ(m.tool_calls[1].arguments, "{}");
        check_stream(t, pp(ToolFormat::Hermes));
        check_stream("<think>\nplan\n</think>\n\n" + t, pp(ToolFormat::Hermes));
        auto m2 = chat::parse_output("<tool_call>\n{\"name\":\"x\",\"arguments\":{\"a\":1}}", pp(ToolFormat::Hermes));  // missing close tag at EOF
        CHECK_EQ(m2.tool_calls.size(), (size_t)1);
        auto m3 = chat::parse_output("just text", pp(ToolFormat::Hermes));
        CHECK_EQ(m3.content, "just text"); CHECK(m3.tool_calls.empty());
    }
    // ---- Qwen XML
    {
        jd::Value tools = J(R"([{"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{"s":{"type":"string"},"n":{"type":"integer"},"o":{"type":"object"},"b":{"type":"boolean"}}}}}])");
        auto p = pp(ToolFormat::QwenXml, true);
        p.tools = &tools;
        std::string t = "plan\n</think>\n\nSure.\n\n<tool_call>\n<function=f>\n<parameter=s>\n007\n</parameter>\n<parameter=n>\n42\n</parameter>\n<parameter=o>\n{\"k\": [1, 2]}\n</parameter>\n<parameter=b>\ntrue\n</parameter>\n</function>\n</tool_call>";
        auto m = chat::parse_output(t, p);
        CHECK_EQ(m.reasoning_content, "plan");
        CHECK_EQ(m.content, "Sure.");
        CHECK_EQ(m.tool_calls.size(), (size_t)1);
        CHECK_EQ(m.tool_calls[0].name, "f");
        CHECK_EQ(m.tool_calls[0].arguments, "{\"s\":\"007\",\"n\":42,\"o\":{\"k\":[1,2]},\"b\":true}");
        check_stream(t, p);
        auto m2 = chat::parse_output("<tool_call>\n<function=g>\n<parameter=x>\nmulti\nline\n</parameter>\n</function>\n</tool_call>", pp(ToolFormat::QwenXml));
        CHECK_EQ(m2.tool_calls[0].arguments, "{\"x\":\"multi\\nline\"}");
    }
    // ---- Inkling
    {
        std::string t = "<|message_model|><|content_thinking|>hmm<|end_message|><|message_model|><|content_text|>Hello!<|end_message|>"
                        "<|message_model|>get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":{\"city\":\"Paris\"}}<|end_message|><|content_model_end_sampling|>";
        auto m = chat::parse_output(t, pp(ToolFormat::Inkling));
        CHECK_EQ(m.reasoning_content, "hmm");
        CHECK_EQ(m.content, "Hello!");
        CHECK_EQ(m.tool_calls.size(), (size_t)1);
        CHECK_EQ(m.tool_calls[0].name, "get_weather");
        CHECK_EQ(m.tool_calls[0].arguments, "{\"city\":\"Paris\"}");
        check_stream(t, pp(ToolFormat::Inkling));
        auto m2 = chat::parse_output("<|content_text|>no header<|end_message|>", pp(ToolFormat::Inkling));
        CHECK_EQ(m2.content, "no header");
    }
    // ---- Mistral / generic
    {
        auto m = chat::parse_output("[TOOL_CALLS][{\"name\":\"f\",\"arguments\":{\"a\":1},\"id\":\"abc\"}]", pp(ToolFormat::Mistral));
        CHECK_EQ(m.tool_calls.size(), (size_t)1); CHECK_EQ(m.tool_calls[0].id, "abc");
        jd::Value tools = J(R"([{"type":"function","function":{"name":"f"}}])");
        auto p = pp(ToolFormat::Generic);
        p.tools = &tools;
        auto g = chat::parse_output("{\"tool_calls\":[{\"name\":\"f\",\"arguments\":{\"x\":2}}]}", p);
        CHECK_EQ(g.tool_calls.size(), (size_t)1); CHECK_EQ(g.tool_calls[0].arguments, "{\"x\":2}");
        auto g2 = chat::parse_output("{\"name\":\"f\",\"parameters\":{\"x\":2}}", p);
        CHECK_EQ(g2.tool_calls.size(), (size_t)1);
        auto g3 = chat::parse_output("{\"not\": \"a call\"}", p);
        CHECK(g3.tool_calls.empty()); CHECK_EQ(g3.content, "{\"not\": \"a call\"}");
    }
    CHECK(chat::detect_tool_format("...<tool_call>...") == ToolFormat::Hermes);
    CHECK(chat::detect_tool_format("<function=x><parameter=y>") == ToolFormat::QwenXml);
    CHECK_EQ(chat::gen_tool_call_id().size(), (size_t)29);
    auto mj = chat::message_to_json(chat::parse_output("<tool_call>{\"name\":\"a\",\"arguments\":{}}</tool_call>", pp(ToolFormat::Hermes)));
    CHECK(mj.find("tool_calls") != nullptr && mj.find("content")->is_null());
}

int main(int argc, char** argv) {
    std::string dir = argc > 1 ? argv[1] : QV_JINJA_TEST_DIR;
    try {
        test_engine();
        test_golden(dir);
        test_chat_render(dir);
        test_parse();
    } catch (const std::exception& e) {
        fprintf(stderr, "uncaught exception: %s\n", e.what());
        return 2;
    }
    printf("test_jinja: %d checks passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
