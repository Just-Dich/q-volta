// Request/response/stream translation for the Anthropic Messages API (/v1/messages) and the OpenAI Responses API (/v1/responses),
// mapped onto the chat-completions path exactly like llama-server's server-chat.cpp / server-task.cpp conversions.
#pragma once
#include <memory>
#include <string>
#include <vector>

#include "core/json_doc.h"

namespace qv::server {

enum class ApiFmt { Oai, Anthropic, Responses };

// Throw std::invalid_argument on malformed input (reported as 400 invalid_request_error).
jd::Value anthropic_to_oai(const jd::Value& body);
jd::Value responses_to_oai(const jd::Value& body);
jd::Value oai_to_anthropic(const jd::Value& oai_chat_completion, const std::string& stopping_word);
jd::Value oai_to_responses(const jd::Value& oai_chat_completion);

// Consumes chat.completion.chunk objects, produces complete SSE records ("event: ...\ndata: ...\n\n").
struct StreamXlate {
    virtual ~StreamXlate() = default;
    virtual std::vector<std::string> feed(const jd::Value& chunk) = 0;
    virtual std::vector<std::string> finish() = 0;  // idempotent
    std::string stopping_word;
};
std::unique_ptr<StreamXlate> make_stream_xlate(ApiFmt fmt, int prompt_tokens);

}  // namespace qv::server
