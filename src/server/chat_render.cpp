#include "server/chat_render.h"

#include <stdexcept>

#include "tokenizer/chat_template.h"

namespace qv::server {

namespace {
const char* kChatML =
    "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}";
}

ChatTemplates::ChatTemplates(const std::string& model_template, const std::string& override_template, bool jinja, std::string& warn) {
    source_ = override_template.empty() ? model_template : override_template;
    std::string src = source_;
    if (!override_template.empty()) {
        custom_ = true;
        if (override_template == "chatml") { src = kChatML; source_ = src; }
        else if (override_template.find("{%") == std::string::npos && override_template.find("{{") == std::string::npos)
            throw std::runtime_error("the built-in chat template '" + override_template + "' is not available (supported names: chatml; or pass the template text / --chat-template-file)");
    }
    kind_ = Kind::Qwen38;
    if (!jinja) { warn = "--no-jinja: the built-in Qwen3.8 renderer is used (no tools, no custom template evaluation)"; if (custom_) kind_ = Kind::ChatML; return; }
    if (src.empty()) return;
    try {
        jt_ = std::make_unique<chat::ChatTemplate>(chat::ChatTemplate::from_source(src, "", "<|im_end|>"));
        jinja_active_ = true;
        kind_ = Kind::Jinja;
    } catch (const std::exception& e) {
        if (custom_) throw std::runtime_error(std::string("failed to compile the chat template: ") + e.what());
        warn = std::string("the model's chat template could not be compiled by the Jinja interpreter (") + e.what() + "); using the built-in Qwen3.8 renderer";
    }
}

jd::Value ChatTemplates::caps() const {
    using jd::Value;
    Value c = Value::object();
    const bool q = kind_ == Kind::Qwen38 || (jt_ && jt_->source().find("reasoning_effort") != std::string::npos);
    const bool t = supports_tools();
    c.set("supports_object_arguments", t);
    c.set("supports_parallel_tool_calls", t);
    c.set("supports_preserve_reasoning", jt_ && jt_->source().find("preserve_thinking") != std::string::npos);
    c.set("supports_reasoning_effort", q);
    c.set("supports_string_content", true);
    c.set("supports_system_role", true);
    c.set("supports_tool_calls", t);
    c.set("supports_tools", t);
    c.set("supports_typed_content", q);
    Value lv = Value::array();
    if (q) for (const char* l : {"low", "medium", "xhigh"}) lv.push(l);
    c.set("reasoning_effort_levels", std::move(lv));
    c.set("reasoning_effort_rejected", Value::array());
    return c;
}

ChatOutput ChatTemplates::apply(const ChatInput& in) const {
    ChatOutput out;
    const bool has_tools = in.tools.is_array() && in.tools.size() > 0;
    if (jt_) {
        chat::RenderInputs ri;
        ri.messages = in.messages;
        if (has_tools) ri.tools = in.tools;
        ri.add_generation_prompt = in.add_generation_prompt;
        ri.enable_thinking = in.enable_thinking;
        ri.reasoning_effort = in.reasoning_effort;
        if (in.kwargs.is_object()) ri.extra = in.kwargs;
        ri.eos_token = "<|im_end|>";
        try {
            out.prompt = jt_->render(ri);
            out.used_jinja = true;
            out.tool_format = jt_->tool_format();
            out.thinking_open = chat::ChatTemplate::prompt_forces_thinking(out.prompt);
            return out;
        } catch (const jinja::RaisedError& e) {
            throw std::runtime_error(e.what());  // template's own raise_exception text -> 400
        } catch (const jinja::Error& e) {
            if (custom_ || has_tools) throw std::runtime_error(std::string("chat template error: ") + e.what());
            // fall through to the native renderer
        }
    } else if (has_tools) {
        throw std::runtime_error("tools require the Jinja chat template (the model has no usable template, or --no-jinja is set)");
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
