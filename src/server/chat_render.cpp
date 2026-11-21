#include "server/chat_render.h"

#include <stdexcept>

#include "tokenizer/chat_template.h"

namespace qv::server {

ChatTemplates::ChatTemplates(const std::string& model_template, const std::string& override_template, bool jinja, std::string& warn) {
    source_ = override_template.empty() ? model_template : override_template;
    if (!override_template.empty()) {
        if (override_template == "chatml") { kind_ = Kind::ChatML; source_ = "chatml"; return; }
        throw std::runtime_error("custom --chat-template / --chat-template-file needs the Jinja interpreter, which is not part of this build (only 'chatml' and the model's own Qwen3.8 template are available)");
    }
    kind_ = Kind::Qwen38;
    if (!jinja) warn = "--no-jinja: the built-in Qwen3.8 renderer is used either way (it is not a Jinja evaluation)";
}

jd::Value ChatTemplates::caps() const {
    using jd::Value;
    Value c = Value::object();
    const bool q = kind_ == Kind::Qwen38;
    c.set("supports_object_arguments", jinja_active_);
    c.set("supports_parallel_tool_calls", jinja_active_);
    c.set("supports_preserve_reasoning", q);
    c.set("supports_reasoning_effort", q);
    c.set("supports_string_content", true);
    c.set("supports_system_role", true);
    c.set("supports_tool_calls", jinja_active_);
    c.set("supports_tools", jinja_active_);
    c.set("supports_typed_content", q);
    Value lv = Value::array();
    if (q) for (const char* l : {"low", "medium", "xhigh"}) lv.push(l);
    c.set("reasoning_effort_levels", std::move(lv));
    c.set("reasoning_effort_rejected", Value::array());
    return c;
}

ChatOutput ChatTemplates::apply(const ChatInput& in) const {
    ChatOutput out;
    if (in.tools.is_array() && in.tools.size() > 0) {
        throw std::runtime_error("tools are not supported by the built-in template renderer in this build (the Jinja interpreter is not linked)");
    }
    auto msgs = messages_from_json(in.messages);
    if (kind_ == Kind::ChatML) {
        std::string s;
        for (auto& m : msgs) s += "<|im_start|>" + m.role + "\n" + m.content + "<|im_end|>\n";
        if (in.add_generation_prompt) s += "<|im_start|>assistant\n";
        out.prompt = s;
        return out;
    }
    ChatTemplateOptions co;
    co.enable_thinking = in.enable_thinking;
    co.reasoning_effort = in.reasoning_effort;
    co.add_generation_prompt = in.add_generation_prompt;
    if (in.kwargs.is_object()) {
        if (const jd::Value* v = in.kwargs.find("preserve_thinking"); v && v->is_bool()) co.preserve_thinking = v->as_bool();
    }
    out.prompt = render_chat(msgs, co);
    const std::string tail = "<think>\n";
    out.thinking_open = out.prompt.size() >= tail.size() && out.prompt.compare(out.prompt.size() - tail.size(), tail.size(), tail) == 0;
    return out;
}

}  // namespace qv::server
