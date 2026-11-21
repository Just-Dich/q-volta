// Chat helpers on top of the jinja engine: template rendering with llama.cpp-style message normalisation, and the
// model-output parser (reasoning extraction + tool-call extraction, complete and streaming/partial).
#pragma once
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "core/json_doc.h"
#include "core/tool_format.h"
#include "jinja/jinja.h"

namespace qv::chat {

// ------------------------------------------------------------------------------------------------ rendering
struct RenderInputs {
    jd::Value messages = jd::Value::array();  // OpenAI chat messages (content: string | [parts] | null; tool_calls[].function.arguments: JSON string or object)
    jd::Value tools;                          // OpenAI "tools" array ([{type:"function",function:{...}}]); null = none
    bool add_generation_prompt = true;
    std::optional<bool> enable_thinking;      // exposed as `enable_thinking` when set
    std::string reasoning_effort;             // exposed as `reasoning_effort` when non-empty
    jd::Value extra;                          // object: additional template variables (chat_template_kwargs); override the above
    std::string bos_token, eos_token;
    std::optional<std::time_t> now;           // strftime_now() clock override (tests)
};

class ChatTemplate {
   public:
    static ChatTemplate from_source(std::string source, std::string bos = "", std::string eos = "");
    static ChatTemplate from_file(const std::string& path, std::string bos = "", std::string eos = "");  // chat_template.jinja
    // tokenizer_config.json: uses "chat_template" (string, or list of {name,template}: picks "default") + bos/eos tokens.
    static ChatTemplate from_tokenizer_config(const std::string& path);

    // Throws jinja::Error / jinja::RaisedError (template raise_exception) -- the message is the template's own text.
    std::string render(const RenderInputs& in) const;

    const std::string& source() const { return src_; }
    ToolFormat tool_format() const { return fmt_; }           // detected from the template text
    bool supports_tools() const { return supports_tools_; }   // template renders the `tools` variable
    bool supports_thinking() const { return thinking_; }      // template mentions <think> / enable_thinking
    // True when the rendered prompt leaves the model *inside* a reasoning block (ends with "<think>" + whitespace).
    static bool prompt_forces_thinking(std::string_view rendered_prompt, std::string_view tag = "<think>");

   private:
    std::string src_, bos_, eos_;
    ToolFormat fmt_ = ToolFormat::Generic;
    bool supports_tools_ = false, thinking_ = false;
    jinja::Environment env_;
    jinja::Template tpl_;
};

ToolFormat detect_tool_format(std::string_view template_source);

// ------------------------------------------------------------------------------------------------ parsing model output
enum class ReasoningFormat {
    None,            // --reasoning-format none: leave <think> in content
    Deepseek,        // reasoning -> reasoning_content, removed from content (llama-server default "auto")
    DeepseekLegacy,  // reasoning -> reasoning_content AND kept inside content
};
ReasoningFormat reasoning_format_from_name(std::string_view name);  // "none" | "deepseek" | "deepseek-legacy" | "auto"(=deepseek); throws on others

struct ToolCall {
    std::string id;         // empty from the parser; use gen_tool_call_id()
    std::string name;
    std::string arguments;  // JSON text (compact when complete)
    bool complete = true;   // false only in partial parses
};

struct Message {
    std::string content;
    std::string reasoning_content;
    std::vector<ToolCall> tool_calls;
    bool operator==(const Message&) const;
};

struct ParseParams {
    ToolFormat format = ToolFormat::Generic;
    ReasoningFormat reasoning = ReasoningFormat::Deepseek;
    bool thinking_forced_open = false;  // generation prompt already opened <think>: the output starts in the reasoning block
    bool parse_tool_calls = true;       // false: tool markup stays in content
    bool partial = false;               // streaming: input may end anywhere; never retracts previously returned text
    bool flush = false;                 // with partial: the input is complete (end of stream) -> no hold-back, still raw (prefix-stable) arguments
    std::string think_start = "<think>", think_end = "</think>";
    const jd::Value* tools = nullptr;   // tool definitions: used to type XML parameter values (string vs JSON)
};

// Split "<think>r</think>c" -> (reasoning, content) according to `p` (tool calls are not touched).
void extract_reasoning(std::string_view text, const ParseParams& p, std::string& reasoning, std::string& content);
// Full pipeline: reasoning extraction, then tool-call extraction on the content part.
Message parse_output(std::string_view text, const ParseParams& p);

// Streaming deltas between two successive partial parses (cur must extend prev).
struct ToolCallDelta {
    size_t index = 0;
    bool is_new = false;           // first delta of this call: carries name (and the caller assigns the id)
    std::string name;
    std::string arguments;         // appended argument text
};
struct Delta {
    std::string content, reasoning;
    std::vector<ToolCallDelta> tool_calls;
    bool empty() const { return content.empty() && reasoning.empty() && tool_calls.empty(); }
};
Delta diff_messages(const Message& prev, const Message& cur);

std::string gen_tool_call_id();  // "call_" + 24 random alphanumerics
// OpenAI-style assistant message JSON: {role, content|null, reasoning_content?, tool_calls?[{id,type,function:{name,arguments}}]}
jd::Value message_to_json(const Message& m, bool with_ids = true);

// Scan one JSON value starting at s[pos] (after whitespace). Returns end index, npos when incomplete/invalid.
size_t scan_json_value(std::string_view s, size_t pos);

}  // namespace qv::chat
