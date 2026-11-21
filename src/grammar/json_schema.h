// JSON Schema -> GBNF (llama.cpp json-schema-to-grammar semantics) and ready-made grammars for response_format / tools.
#pragma once
#include <map>
#include <string>
#include <vector>

#include "core/json_doc.h"
#include "core/tool_format.h"

namespace qv::grammar {

struct SchemaOptions {
    bool dotall = false;    // regex '.' matches newlines
    bool compact = false;   // `space` matches nothing (forces compact JSON like {"a":1}); default allows llama.cpp's optional whitespace
};

// Incremental builder: several schemas / hand-written rules can share one grammar (used by the tool-call grammars).
class SchemaGrammarBuilder {
   public:
    explicit SchemaGrammarBuilder(SchemaOptions opt = {});
    // Adds `rule ::= body` (renamed on collision); returns the final rule name.
    std::string add_rule(const std::string& name, const std::string& body);
    // Converts `schema` (must outlive nothing: it is copied) and returns the name of its rule. `$ref`s resolve against `schema` itself.
    std::string add_schema(const std::string& name, const jd::Value& schema);
    // Adds a builtin json primitive rule (value, object, array, string, number, integer, boolean, null, char, ...) and its deps.
    std::string add_primitive(const std::string& name);
    const std::vector<std::string>& warnings() const { return warnings_; }
    std::string build() const;  // GBNF text of all rules (sorted by name)

   private:
    struct Conv;
    friend struct Conv;
    SchemaOptions opt_;
    std::map<std::string, std::string> rules_;
    std::vector<std::string> warnings_;
};

// Whole-schema conversion; root rule is `root`. Throws std::invalid_argument ("JSON schema conversion failed: ...").
std::string json_schema_to_grammar(const jd::Value& schema, SchemaOptions opt = {});

// GBNF for a literal string ("..." with escapes) and for rule-name sanitising ([^a-zA-Z0-9-]+ -> '-').
std::string gbnf_literal(const std::string& s);
std::string gbnf_rule_name(const std::string& s);

// response_format {"type":"json_object"}: any JSON object (llama.cpp grammars/json.gbnf).
std::string json_object_grammar();

// ---- tool calling
struct ToolGrammarOptions {
    ToolFormat format = ToolFormat::Hermes;
    std::string forced_name;            // tool_choice = {"function":{"name":..}}: only this tool; empty = any of the given tools (required)
    bool parallel = false;              // allow several calls in one answer
    bool thinking_forced_open = false;  // the prompt left the model inside <think>: the grammar first accepts reasoning up to think_end
    std::string think_end = "</think>";
};
struct ToolGrammar {
    std::string gbnf;                       // root rule = [reasoning] tool call(s)
    std::vector<std::string> trigger_words; // for tool_choice=auto (lazy grammar): start of a tool call; the grammar `root_lazy` begins with it
    std::string gbnf_lazy;                  // grammar for lazy mode: starts at the trigger word (no reasoning prefix)
};
// tools: OpenAI array [{"type":"function","function":{"name","description","parameters":{json schema}}}]
ToolGrammar tool_call_grammar(const jd::Value& tools, const ToolGrammarOptions& opt);

// A rule matching any text that does not contain `literal`, ending with `literal` itself (KMP automaton). Returns the rule name.
std::string add_until_rule(SchemaGrammarBuilder& b, const std::string& name, const std::string& literal);

}  // namespace qv::grammar
