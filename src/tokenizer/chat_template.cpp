#include "tokenizer/chat_template.h"

#include <stdexcept>

namespace qv {

namespace {
// Decode one UTF-8 code point at s[i]; returns length (>=1).
int cp_at(const std::string& s, size_t i, uint32_t& cp) {
    unsigned char c = s[i];
    int n = c < 0x80 ? 1 : c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
    if (i + n > s.size()) n = 1;
    cp = n == 1 ? c : c & (0xFF >> (n + 1));
    for (int k = 1; k < n; ++k) cp = (cp << 6) | ((unsigned char)s[i + k] & 63);
    return n;
}
bool py_space(uint32_t c) {
    return (c >= 9 && c <= 13) || (c >= 0x1C && c <= 0x20) || c == 0x85 || c == 0xA0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200A) ||
           c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}
bool starts_with(const std::string& s, const char* p) { return s.rfind(p, 0) == 0; }
bool ends_with(const std::string& s, const char* p) {
    size_t n = std::char_traits<char>::length(p);
    return s.size() >= n && s.compare(s.size() - n, n, p) == 0;
}
}  // namespace

std::string py_strip(const std::string& s) {
    size_t b = 0;
    while (b < s.size()) {
        uint32_t cp;
        int n = cp_at(s, b, cp);
        if (!py_space(cp)) break;
        b += n;
    }
    size_t e = s.size();
    while (e > b) {
        size_t k = e - 1;
        while (k > b && ((unsigned char)s[k] & 0xC0) == 0x80) --k;
        uint32_t cp;
        cp_at(s, k, cp);
        if (!py_space(cp)) break;
        e = k;
    }
    return s.substr(b, e - b);
}

std::string render_chat(const std::vector<ChatMessage>& msgs, const ChatTemplateOptions& opt) {
    if (msgs.empty()) throw std::runtime_error("No messages provided.");
    const bool thinking = !opt.enable_thinking.has_value() || *opt.enable_thinking;
    std::string reasoning;
    if (thinking) {
        std::string eff = opt.reasoning_effort.empty() ? "xhigh" : opt.reasoning_effort;
        if (eff != "xhigh" && eff != "medium" && eff != "low")
            throw std::runtime_error("Unexpected reasoning effort " + opt.reasoning_effort + ". Supported types are xhigh (default), medium, and low.");
        if (eff == "xhigh")
            reasoning = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
        else if (eff == "low")
            reasoning = "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.";
    }
    std::string out;
    if (msgs[0].role == "system") {
        std::string c = py_strip(msgs[0].content);
        if (!c.empty()) out += "<|im_start|>system\n" + (reasoning.empty() ? "" : reasoning + "\n\n") + c + "<|im_end|>\n";
        else if (!reasoning.empty()) out += "<|im_start|>system\n" + reasoning + "<|im_end|>\n";
    } else if (!reasoning.empty()) {
        out += "<|im_start|>system\n" + reasoning + "<|im_end|>\n";
    }
    // index of the last real user query (a user message that is not purely tool responses)
    long last_query = (long)msgs.size() - 1;
    bool found = false;
    for (long i = (long)msgs.size() - 1; i >= 0; --i) {
        if (msgs[i].role == "user") {
            std::string c = py_strip(msgs[i].content);
            if (!(starts_with(c, "<tool_response>") && ends_with(c, "</tool_response>"))) { last_query = i; found = true; break; }
        }
    }
    if (!found) throw std::runtime_error("No user query found in messages.");
    const size_t n = msgs.size();
    for (size_t i = 0; i < n; ++i) {
        const ChatMessage& m = msgs[i];
        std::string content = py_strip(m.content);
        if (m.role == "system") {
            if (i != 0) throw std::runtime_error("System message must be at the beginning.");
        } else if (m.role == "user") {
            out += "<|im_start|>user\n" + content + "<|im_end|>\n";
        } else if (m.role == "assistant") {
            std::string rc = py_strip(m.reasoning_content);
            out += "<|im_start|>assistant\n";
            if (!opt.preserve_thinking.has_value() || *opt.preserve_thinking || (long)i > last_query)
                out += "<think>\n" + rc + "\n</think>\n\n" + content;
            else
                out += content;
            out += "<|im_end|>\n";
        } else if (m.role == "tool") {
            if (i > 0 && msgs[i - 1].role != "tool") out += "<|im_start|>user";
            out += "\n<tool_response>\n" + content + "\n</tool_response>";
            if ((i + 1 < n && msgs[i + 1].role != "tool") || i + 1 == n) out += "<|im_end|>\n";
        } else {
            throw std::runtime_error("Unexpected message role.");
        }
    }
    if (opt.add_generation_prompt) {
        out += "<|im_start|>assistant\n";
        if (opt.enable_thinking.has_value() && !*opt.enable_thinking) out += "<think>\n\n</think>\n\n";
        else out += "<think>\n";
    }
    return out;
}

std::vector<ChatMessage> messages_from_json(const jd::Value& arr) {
    if (!arr.is_array()) throw std::runtime_error("'messages' must be an array");
    std::vector<ChatMessage> out;
    for (auto& m : arr.items()) {
        if (!m.is_object()) throw std::runtime_error("each message must be an object");
        const jd::Value* role = m.find("role");
        if (!role || !role->is_string()) throw std::runtime_error("message is missing a string 'role'");
        ChatMessage cm;
        cm.role = role->as_string();
        if (cm.role == "developer") cm.role = "system";  // OpenAI newer alias
        if (const jd::Value* tc = m.find("tool_calls"); tc && !tc->is_null() && !(tc->is_array() && tc->size() == 0))
            throw std::runtime_error("assistant 'tool_calls' are not supported by this server");
        const jd::Value* c = m.find("content");
        if (c && c->is_string()) cm.content = c->as_string();
        else if (c && c->is_array()) {
            for (auto& part : c->items()) {
                const jd::Value* ty = part.is_object() ? part.find("type") : nullptr;
                const jd::Value* tx = part.is_object() ? part.find("text") : nullptr;
                if (!ty || !ty->is_string() || ty->as_string() != "text" || !tx || !tx->is_string())
                    throw std::runtime_error("only text content parts are supported (no images, audio or video)");
                cm.content += tx->as_string();
            }
        } else if (c && !c->is_null()) throw std::runtime_error("message 'content' must be a string or an array of text parts");
        if (const jd::Value* r = m.find("reasoning_content"); r && r->is_string()) cm.reasoning_content = r->as_string();
        out.push_back(std::move(cm));
    }
    return out;
}

}  // namespace qv
