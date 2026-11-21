# Grammar-constrained decoding (src/grammar)

Standalone C++20 library (`qv_grammar`, no third-party deps). Semantics follow llama.cpp's `llama-grammar.cpp` and
`json-schema-to-grammar.cpp`. Tests: `tests/test_grammar.cpp` (ctest `grammar`, 807 checks).

## API (`grammar/grammar.h`, namespace `qv::grammar`)
- `Grammar::parse(gbnf, root="root")` throws `GrammarError` (syntax, undefined rule, left recursion, repetition limit). `dump()` gives re-parseable GBNF.
  Supported: literals with escapes, `[a-z]`/`[^..]`, `.`, groups, `* + ? {m} {m,} {m,n}`, comments, `\x \u \U`. Not supported: `<[id]>` token elements.
- `VocabIndex::build(pieces, eog)`: byte trie in pre-order with subtree skip pointers. `vocab_from_tokenizer(tok, eog_ids)` in `sampler_hook.h`.
- `Matcher(grammar, vocab)`: shared across requests; owns an LRU-bounded cache of allowed-token masks keyed by (state id, partial UTF-8). Thread-safe.
- `GrammarState(matcher, Triggers lazy = {})`: `accept(bytes | code points)`, `accept_token(id)`, `is_complete()`, `is_dead()`, `allowed()`, `mask_logits(float*, n)`, `token_allowed(id)`, `fork()`.
  Lazy triggers (trigger words / token ids) buffer text until a trigger appears, then the grammar consumes from the start of the match.
- Token rules: allowed iff every code point advances some stack (trailing partial UTF-8 allowed when completable); EOG tokens only when the grammar can end; empty pieces never allowed.
- `GrammarState(const Grammar&)` needs no vocabulary (accept / is_complete only).

## JSON schema (`grammar/json_schema.h`)
`json_schema_to_grammar(schema, opts)`; `SchemaGrammarBuilder` to combine schemas and hand-written rules; `json_object_grammar()` for `response_format: json_object`.
Covers type (incl. lists), enum, const, oneOf/anyOf, allOf (property merge), `$ref` (local, recursive), definitions/$defs, properties/required/optional (fixed order: required first, then optional, like llama.cpp),
additionalProperties (bool/schema), items/prefixItems/minItems/maxItems, minLength/maxLength, anchored `^...$` patterns (unsupported ones fall back to any string and add a warning),
integer minimum/maximum/exclusive*, formats date/time/date-time/uuid. Number bounds on non-integers are not enforced (best effort).
Whitespace rule is llama.cpp's `space`; `SchemaOptions::compact` removes it.

## Tool calls (`tool_call_grammar`)
Per `qv::ToolFormat` (Hermes, QwenXml, Inkling, Mistral, Generic), forced tool name or any given tool, optional parallel calls, optional reasoning prefix (`thinking_forced_open`, text up to `</think>`).
Returns `gbnf`, `gbnf_lazy` (starts at the trigger word) and `trigger_words` for tool_choice=auto. QwenXml string parameters are raw text up to `\n</parameter>` (KMP-built `add_until_rule`).

## Sampler hook
`qv::LogitMask` (src/sampling/sampler.h) + `Sampler::set_mask()`; the mask is applied after penalties and before temperature/top-k, and `accept()` forwards the committed token.
`GrammarLogitMask(GrammarState)` implements it. Note: constrained requests must sample on host logits (not `engine_argmax_ok()`).

## Mask computation
States are interned stack-sets; transitions cached per state (128-entry ASCII row, uniform non-ASCII successor, hash for others). The mask is a DFS over the vocab trie with dead-state pruning and UTF-8 partial states.
Masks are cached, so repeated states cost one lookup.

## Measured latency (Qwen3.8 tokenizer, 248077 tokens, 591809 trie nodes, single thread, build with -DQV_CUDA=OFF -DQV_CPU=OFF)
See the table printed by `test_grammar` ("[qwen]" lines); numbers are recorded in docs/PROGRESS.md.
