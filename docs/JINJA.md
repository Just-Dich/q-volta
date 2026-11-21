# Jinja2 interpreter + chat helpers (src/jinja)

Standalone C++20 library (no third-party code; only `src/core/json_doc.h`). Target: render the real chat templates of
Qwen3.x and Inkling-2-Mini and llama.cpp's `models/templates/*.jinja` with **Python-jinja2-exact output**, and give the
server the helper layer around them. CMake: `src/jinja/CMakeLists.txt` -> `qv_jinja`, `jinja_render`, `test_jinja` (ctest `jinja`).

## Engine (`jinja/jinja.h`, `jinja/value.h`)
```cpp
qv::jinja::Environment env;                       // defaults = HF chat-template settings: trim_blocks, lstrip_blocks
auto tpl = env.compile(source);                   // throws jinja::Error (syntax)
jinja::Value ctx = jinja::from_json(jd_value);    // Value has Python object semantics (shared list/dict refs, str = code points)
std::string out = tpl.render(ctx);                // throws jinja::RaisedError (raise_exception; what() = bare message) or jinja::Error
env.set_global(name, Value), env.add_filter(name, fn), env.add_test(name, fn), env.set_clock(fn)   // strftime_now pin
```
Supported: all expression forms (arithmetic incl. `//`, `%`, `**`, string `%` formatting, `~`, chained comparisons, `in`/`not in`,
`is [not] test`, ternary with/without else, slicing incl. negative steps, attribute/item access with jinja's getattr->getitem
fallback, calls with kwargs); statements `if/elif/else`, `for` (tuple targets, `if` filter, `else`, `recursive`, `loop.*`
incl. `previtem/nextitem/cycle/changed/depth`), `set` (tuple, `ns.attr`, block form with filter), `macro`, `call`/`caller`,
`filter` blocks, `with`, `do`, `break`/`continue`, `raw`, comments, HF `{% generation %}` (rendered inline), whitespace control
`{%- -%}`, `{%+`, trim_blocks, lstrip_blocks and "single trailing newline dropped" exactly like jinja2's lexer.
Python-faithful `str()`/`repr()` of `None/True/False/float` (shortest round-trip, `1e-05`, `1e+16`), dict/list/tuple repr, and
`json.dumps` formatting for `|tojson` (HF signature `tojson(ensure_ascii=False, indent=None, separators=None, sort_keys=False)`).
Filters: abs attr batch capitalize center count/length default/d dictsort e/escape first float format from_json groupby indent int
items join last list lower map max min reject rejectattr replace reverse round safe select selectattr slice sort string striptags
sum title tojson trim truncate unique upper urlencode wordcount wordwrap. Tests: defined undefined none string number integer
float boolean true false mapping iterable sequence callable even odd divisibleby lower upper eq/ne/lt/le/gt/ge (+symbol
aliases) in sameas escaped filter test. Globals: range dict namespace raise_exception strftime_now joiner.
Methods: str (upper lower strip lstrip rstrip split rsplit splitlines startswith endswith replace find rfind index count join
format zfill center ljust rjust is* partition title capitalize swapcase removeprefix removesuffix), list (append extend insert
pop remove index count reverse copy clear sort), dict (items keys values get update pop setdefault copy clear).
Not supported: `include/import/from/extends/block` (no template loader), `*args/**kwargs` call unpacking, autoescape,
`pprint/random/xmlattr/filesizeformat`, Python's `Markup` type (see below).

Deliberate divergences from python-jinja2 (follow llama.cpp's engine instead):
* `|safe`/`|e` return plain strings. In Python `Markup + str` HTML-escapes the other operand even with autoescape off
  (`meetkai-functionary-medium-v3.1` renders `&#34;` under HF); llama.cpp does not, nor do we. The oracle script patches these two filters.
* `map/select/reject/selectattr/rejectattr/batch/slice` return lists, not generators (`x|map(..)|tojson` works). The oracle wraps them.
* Python raises on `{% for %}` over some non-iterables where we raise our own message; error *text* is only compared for
  `raise_exception` (exact).

## Chat helpers (`jinja/chat.h`, namespace `qv::chat`)
* `ChatTemplate::from_file/from_source/from_tokenizer_config`, `render(RenderInputs)`: OpenAI messages (jd JSON) + tools +
  `add_generation_prompt`, `enable_thinking`, `reasoning_effort`, arbitrary `extra` variables (chat_template_kwargs), bos/eos.
  llama.cpp-style normalisation: null content -> `""`, `developer` -> `system` (unless the template has `<|channel|>`),
  `tool_calls[].function.arguments` JSON string -> object, text-only content-part arrays flattened to a string (joined `\n`)
  unless the template inspects parts (`image`/`text`). `tools` is `none` when absent (HF behaviour).
  `tool_format()`, `supports_tools()`, `supports_thinking()`, `prompt_forces_thinking(prompt)` (prompt ends with `<think>`).
* Output parsing, `ParseParams{format, reasoning, thinking_forced_open, partial, flush, tools}` and
  `Message parse_output(text, params)` -> `{content, reasoning_content, tool_calls[{id,name,arguments}]}`.
  Formats (`core/tool_format.h`): `Hermes` (`<tool_call>{json}</tool_call>`, also Qwen2.5/3), `QwenXml`
  (`<tool_call><function=f><parameter=k>v</parameter></function></tool_call>`, Qwen3-Coder/3.5/3.8; parameter values typed
  from the tool schema: `string` stays text, others parsed as JSON), `Inkling` (message stream per tokenizer_config
  `response_template`: `<|message_model|>[name]<|content_{thinking|text|invoke_tool_json}|>...<|end_message|>`, ends at
  `<|content_model_end_sampling|>`; thinking -> reasoning_content, `{"name","args"}` -> tool call), `Mistral`
  (`[TOOL_CALLS][...]`), `Generic` (`{"tool_calls":[..]}` / `{"tool_call":{..}}` / bare `{"name","arguments"|"parameters"}`,
  only when tools were given). `detect_tool_format(template_source)` picks the format.
  Inkling output must be detokenized **with special tokens visible**.
* Reasoning (`--reasoning-format`): `ReasoningFormat::{None, Deepseek (=auto), DeepseekLegacy}`; `extract_reasoning` handles
  `thinking_forced_open` (prompt already ended in `<think>`), unterminated reasoning, whitespace trimming.
* Streaming: call `parse_output` on the accumulated text with `partial=true` and diff successive results with
  `diff_messages(prev, cur)` -> `Delta{content, reasoning, tool_calls[{index,is_new,name,arguments}]}`. Partial parses never
  retract (tag prefixes and trailing whitespace before a possible `<tool_call>` are held back); tool arguments are raw
  prefix-stable text while streaming (the `partial=false` result re-serialises arguments compactly like llama.cpp, so use the
  streamed text for the stream and `partial=false` only for non-streaming responses). At end of stream call once more with
  `partial=true, flush=true` to release the hold-back. `gen_tool_call_id()`, `message_to_json()` for the OpenAI response.

## Differential testing (Python jinja2 3.1.6 oracle)
`tests/jinja/diff_oracle.py BIN DIR_OR_FILES...` renders each template x 33 generated conversations (system/tool/parallel tool
calls/reasoning/content parts/unicode/enable_thinking/effort/empty/raise cases) with both engines (HF-style environment:
`trim_blocks, lstrip_blocks, loopcontrols, do, {% generation %}`, HF `tojson`, fixed `strftime_now`) and compares
**byte-for-byte**; both-raise cases must have equal `raise_exception` messages. `tests/jinja/snippets.py BIN` does the same for 94
feature snippets. `tests/jinja/gen_golden.py` regenerates `tests/jinja/golden/*.json` (oracle outputs for 11 templates in
`tests/jinja/templates/`) that the ctest `test_jinja` checks without Python.
```
python3 tests/jinja/diff_oracle.py build/src/jinja/jinja_render /path/to/llama.cpp/models/templates real/chat_template.jinja ...
python3 tests/jinja/snippets.py build/src/jinja/jinja_render
```
