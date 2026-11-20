// Native C++ rendering of the Qwen3.8 chat template (chat_template.jinja): system/user/assistant/tool roles,
// enable_thinking / reasoning_effort / preserve_thinking template variables, add_generation_prompt.
// Not supported: `tools` definitions in the system prompt, assistant `tool_calls`, image/video content parts.
#pragma once
#include <optional>
#include <string>
#include <vector>

#include "core/json_doc.h"

namespace qv {

struct ChatMessage {
    std::string role;
    std::string content;            // text parts of the message content already concatenated
    std::string reasoning_content;  // assistant only
};

struct ChatTemplateOptions {
    std::optional<bool> enable_thinking;   // undefined -> thinking enabled
    std::string reasoning_effort;          // "" = undefined -> "xhigh"
    std::optional<bool> preserve_thinking; // undefined -> true
    bool add_generation_prompt = true;
};

// Throws std::runtime_error with the template's own error text for invalid conversations.
std::string render_chat(const std::vector<ChatMessage>& msgs, const ChatTemplateOptions& opt);

// Python str.strip() equivalent on UTF-8 (what jinja's |trim does).
std::string py_strip(const std::string& s);

// Build messages from an OpenAI-style JSON array; throws std::runtime_error with a user-facing message.
std::vector<ChatMessage> messages_from_json(const jd::Value& arr);

}  // namespace qv
