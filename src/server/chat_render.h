// Chat template front-end of the server: messages (+ tools) -> prompt text, plus output post-processing metadata.
//   * native Qwen3.8 renderer (src/tokenizer/chat_template.cpp, verified against HF apply_chat_template)
//   * native "chatml" built-in (what `--chat-template chatml` selects in llama-server)
//   * Jinja interpreter (src/jinja, qv::chat::ChatTemplate): the model's own chat_template.jinja, --chat-template / --chat-template-file,
//     tools and tool-call history. The native Qwen3.8 renderer is only the fallback when the Jinja evaluation of the *model's own*
//     template fails (and no tools were requested).
#pragma once
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "core/json_doc.h"
#include "core/tool_format.h"
#include "jinja/chat.h"

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
    ToolFormat tool_format = ToolFormat::Generic;
    bool used_jinja = false;
};

class ChatTemplates {
   public:
    // `model_template`: contents of the model's chat_template.jinja ("" if none). Throws std::runtime_error on unusable settings.
    ChatTemplates(const std::string& model_template, const std::string& override_template, bool jinja, std::string& warn);
    ChatOutput apply(const ChatInput& in) const;  // throws std::runtime_error (message goes to a 400)
    const std::string& source() const { return source_; }
    jd::Value caps() const;
    bool supports_tools() const { return jt_ && jt_->supports_tools(); }
    ToolFormat tool_format() const { return jt_ ? jt_->tool_format() : ToolFormat::Generic; }

   private:
    enum class Kind { Qwen38, ChatML, Jinja } kind_ = Kind::Qwen38;
    std::string source_;
    bool jinja_active_ = false;
    bool custom_ = false;  // --chat-template given: no native fallback
    std::unique_ptr<chat::ChatTemplate> jt_;
};

}  // namespace qv::server
