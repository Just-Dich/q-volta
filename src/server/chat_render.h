// Chat template front-end of the server: messages (+ tools) -> prompt text, plus output post-processing metadata.
//   * native Qwen3.8 renderer (src/tokenizer/chat_template.cpp, verified against HF apply_chat_template)
//   * native "chatml" built-in (what `--chat-template chatml` selects in llama-server)
//   * Jinja interpreter (src/jinja, built by branch p15/jinja) for arbitrary templates, tools and tool-call history: enabled with
//     -DQV_HAVE_JINJA; until it is merged `tools` and custom templates are rejected with an explanatory 400.
#pragma once
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "core/json_doc.h"

namespace qv::server {

struct ChatInput {
    jd::Value messages;                    // OpenAI messages array (already validated)
    jd::Value tools;                       // array or null
    std::string tool_choice = "auto";
    bool parallel_tool_calls = false;
    bool add_generation_prompt = true;
    std::optional<bool> enable_thinking;
    std::string reasoning_effort;
    jd::Value kwargs;                      // chat_template_kwargs object (or null)
};

struct ChatOutput {
    std::string prompt;
    bool thinking_open = false;            // generation prompt ends inside an open <think> block
    std::string think_start = "<think>", think_end = "</think>";
    std::vector<std::string> additional_stops;
};

class ChatTemplates {
   public:
    // `model_template`: contents of the model's chat_template.jinja ("" if none). Throws std::runtime_error on unusable settings.
    ChatTemplates(const std::string& model_template, const std::string& override_template, bool jinja, std::string& warn);
    ChatOutput apply(const ChatInput& in) const;  // throws std::runtime_error (message goes to a 400)
    const std::string& source() const { return source_; }
    jd::Value caps() const;
    bool supports_tools() const { return jinja_active_; }

   private:
    enum class Kind { Qwen38, ChatML, Jinja } kind_ = Kind::Qwen38;
    std::string source_;
    bool jinja_active_ = false;
};

}  // namespace qv::server
