// Tool-call wire formats a model can emit. Shared by the chat output parser (src/jinja/chat.*) and the tool-call grammar generator (src/grammar).
#pragma once
namespace qv {
enum class ToolFormat {
    Generic,   // JSON: {"tool_calls":[{"name":..,"arguments":{..}}]} / {"tool_call":{..}} / bare {"name":..,"arguments":..}
    Hermes,    // <tool_call>\n{"name": "f", "arguments": {..}}\n</tool_call>   (Qwen2.5/3, Hermes, Llama-3 tool_use variants)
    QwenXml,   // <tool_call>\n<function=f>\n<parameter=k>\nv\n</parameter>\n</function>\n</tool_call>   (Qwen3-Coder, Qwen3.5+)
    Inkling,   // <|message_model|>f<|content_invoke_tool_json|>{"name":"f","args":{..}}<|end_message|>
    Mistral,   // [TOOL_CALLS][{"name":"f","arguments":{..}}, ...]
};
}
