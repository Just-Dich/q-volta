#!/usr/bin/env python3
"""Differential test: C++ jinja (build/jinja_render) vs Python jinja2 on chat templates x conversations, byte-for-byte.
usage: diff_oracle.py BIN TEMPLATE_DIR_OR_FILE... [--filter SUBSTR] [--verbose] [--show-fail N]
Oracle = jinja2.Environment(trim_blocks, lstrip_blocks, loopcontrols, do, {% generation %}) with HF-style tojson/raise_exception/strftime_now.
"""
import json, os, subprocess, sys, tempfile, datetime, glob, concurrent.futures as cf
import jinja2
from jinja2.ext import Extension
from jinja2 import nodes

NOW = 1767323045  # 2026-01-02 03:04:05 UTC
os.environ["TZ"] = "UTC"


class GenerationExt(Extension):
    tags = {"generation"}
    def parse(self, parser):
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(["name:endgeneration"], drop_needle=True)
        return body


def make_env():
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True,
                             extensions=["jinja2.ext.loopcontrols", "jinja2.ext.do", GenerationExt])

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
    env.filters["tojson"] = tojson
    # jinja2 returns lazy generators from these; llama.cpp (and this port) return lists. Wrap so `x|map(..)|tojson` etc. is comparable.
    import functools
    def _listify(o):
        @functools.wraps(o)
        def w(*a, **k):
            return list(o(*a, **k))
        return w
    for _n in ("map", "select", "reject", "selectattr", "rejectattr"):
        env.filters[_n] = _listify(env.filters[_n])
    # jinja2's Markup (from |safe / |e) HTML-escapes the other operand of `+`, even with autoescape off. llama.cpp's engine (and this port)
    # has no Markup type, so the oracle returns plain str to compare the rest of the semantics.
    import markupsafe
    env.filters["safe"] = lambda x: str(x)
    env.filters["e"] = env.filters["escape"] = lambda x: str(markupsafe.escape(x))
    env.filters["from_json"] = lambda s: json.loads(s)

    def raise_exception(msg):
        raise jinja2.exceptions.TemplateError(msg)
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = lambda fmt: datetime.datetime.fromtimestamp(NOW, tz=datetime.timezone.utc).strftime(fmt)
    return env


def tool(name, desc, props, required=()):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": list(required)}}}

WEATHER = tool("get_weather", "Get the current weather", {
    "city": {"type": "string", "description": "City name"},
    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
    "days": {"type": "integer"}}, ["city"])
SEARCH = tool("web_search", "Search the web", {"query": {"type": "string"}, "filters": {
    "type": "object", "properties": {"site": {"type": "string"}, "recent": {"type": "boolean"}}},
    "tags": {"type": "array", "items": {"type": "string"}}}, ["query"])


def tc(name, args, id_=None):
    d = {"type": "function", "function": {"name": name, "arguments": args}}
    if id_ is not None:
        d["id"] = id_
    return d


def cases():
    U = lambda c: {"role": "user", "content": c}
    A = lambda c, **k: dict({"role": "assistant", "content": c}, **k)
    S = lambda c: {"role": "system", "content": c}
    T = lambda c, **k: dict({"role": "tool", "content": c}, **k)
    cs = []
    def add(name, messages, **kw):
        d = {"messages": messages, "add_generation_prompt": True, "bos_token": "<s>", "eos_token": "</s>"}
        d.update(kw)
        cs.append((name, d))
    add("single_user", [U("Hello!")])
    add("no_genprompt", [U("Hello!")], add_generation_prompt=False)
    add("system_user", [S("You are a helpful assistant."), U("What is 2+2?")])
    add("multi_turn", [S("Be brief."), U("Hi"), A("Hello there."), U("Tell me a joke"), A("Why did the chicken..."), U("Another")])
    add("unicode_special", [U("Héllo wörld ✓ 日本語 😀 <tag> & \"quotes\" 'single' \\ backslash\nnewline\ttab")])
    add("thinking_off", [U("Think?")], enable_thinking=False)
    add("thinking_on", [U("Think?")], enable_thinking=True)
    add("effort_low", [U("Think?")], reasoning_effort="low")
    add("effort_medium", [U("Think?")], reasoning_effort="medium")
    add("effort_num", [U("Think?")], reasoning_effort=0.5)
    add("assistant_reasoning", [U("2+2?"), A("4", reasoning_content="Adding two and two."), U("and 3+3?")])
    add("assistant_reasoning_last", [U("2+2?"), A("4", reasoning_content="Adding two and two.")], add_generation_prompt=False)
    add("content_parts", [{"role": "user", "content": [{"type": "text", "text": "part one"}, {"type": "text", "text": "part two"}]}])
    add("empty_system", [S(""), U("hi")])
    add("tools_basic", [U("Weather in Paris?")], tools=[WEATHER])
    add("tools_two", [S("Use tools wisely."), U("Weather in Paris and search cats?")], tools=[WEATHER, SEARCH])
    add("tool_roundtrip", [U("Weather in Paris?"),
        A("", tool_calls=[tc("get_weather", {"city": "Paris", "unit": "celsius"}, "call_1")]),
        T("{\"temp\": 21}", tool_call_id="call_1", name="get_weather")], tools=[WEATHER])
    add("tool_roundtrip_final", [U("Weather in Paris?"),
        A("Let me check.", tool_calls=[tc("get_weather", {"city": "Paris", "days": 3, "unit": "celsius"}, "call_1")]),
        T("{\"temp\": 21}", tool_call_id="call_1"), A("It is 21C."), U("thanks")], tools=[WEATHER])
    add("tool_parallel", [U("Both please"),
        A(None, tool_calls=[tc("get_weather", {"city": "Paris"}, "a"), tc("web_search", {"query": "cats", "tags": ["a", "b"], "filters": {"site": "x.org", "recent": True}}, "b")]),
        T("sunny", tool_call_id="a"), T("results...", tool_call_id="b")], tools=[WEATHER, SEARCH])
    add("tool_parallel_think", [U("Both please"),
        A("ok", reasoning_content="I should call two tools.", tool_calls=[tc("get_weather", {"city": "Paris"}, "a"), tc("get_weather", {"city": "Rome", "unit": "fahrenheit"}, "b")]),
        T("sunny", tool_call_id="a"), T("rainy", tool_call_id="b")], tools=[WEATHER])
    add("tool_args_types", [U("go"), A("", tool_calls=[tc("f", {"n": 1, "x": 2.5, "b": False, "z": None, "s": "str\nwith \"q\"", "l": [1, "two", {"k": "v"}], "u": "日本"}, "c1")]), T("done", tool_call_id="c1")], tools=[WEATHER])
    add("tool_no_args", [U("go"), A("", tool_calls=[tc("ping", {}, "c1")]), T("pong", tool_call_id="c1")], tools=[WEATHER])
    add("tool_noid", [U("go"), A("", tool_calls=[tc("ping", {"a": 1})]), T("pong")], tools=[WEATHER])
    add("system_tools", [S("Sys prompt."), U("weather?")], tools=[WEATHER])
    add("user_tool_end", [U("q"), A("a1"), U("q2")], tools=[SEARCH], enable_thinking=False)
    add("assistant_last_prefill", [U("Say hi"), A("Hi")], add_generation_prompt=False)
    add("long_content", [U("line1\n\nline2\n  indented\n" * 5), A("resp " * 20), U("end")])
    add("whitespace_content", [U("  padded  \n"), A("\n\nanswer\n\n"), U("\t tab")])
    add("tool_in_system_mid", [U("a"), S("late system"), U("b")])
    add("two_assistants", [U("a"), A("b"), A("c"), U("d")])
    add("consecutive_users", [U("a"), U("b")])
    add("no_user", [A("only assistant")])
    add("empty_messages", [])
    return cs


def render_py(t, ctx):
    try:
        return ("ok", t.render(**ctx))
    except jinja2.exceptions.TemplateAssertionError as e:
        return ("err", "other", str(e))
    except jinja2.exceptions.TemplateError as e:
        if type(e) is jinja2.exceptions.TemplateError:
            return ("err", "raise", str(e))
        return ("err", "other", str(e))
    except Exception as e:
        return ("err", "other", repr(e))


def render_cpp(binp, path, ctx):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(ctx, f, ensure_ascii=False)
        cp = f.name
    try:
        r = subprocess.run([binp, path, cp, "--now", str(NOW)], capture_output=True, env=dict(os.environ, TZ="UTC"))
    finally:
        os.unlink(cp)
    if r.returncode == 0:
        return ("ok", r.stdout.decode("utf-8"))
    msg = r.stderr.decode("utf-8", "replace")
    if msg.startswith("ERROR[raise]: "):
        return ("err", "raise", msg[len("ERROR[raise]: "):])
    return ("err", "other", msg)


def main():
    args = sys.argv[1:]
    binp = args.pop(0)
    verbose = "--verbose" in args
    flt = None
    show = 5
    paths = []
    i = 0
    while i < len(args):
        if args[i] == "--filter": flt = args[i + 1]; i += 2
        elif args[i] == "--show-fail": show = int(args[i + 1]); i += 2
        elif args[i] == "--verbose": i += 1
        else: paths.append(args[i]); i += 1
    files = []
    for p in paths:
        files += sorted(glob.glob(os.path.join(p, "*.jinja"))) if os.path.isdir(p) else [p]
    if flt:
        files = [f for f in files if flt in f]
    env = make_env()
    cs = cases()
    tot = {"templates": 0, "tpl_pass": 0, "cases": 0, "exact": 0, "both_err": 0, "raise_msg_equal": 0, "fail": 0}
    fails = []
    def run(f, name, ctx):
        src = open(f, encoding="utf-8").read()
        return None
    for f in files:
        src = open(f, encoding="utf-8").read()
        tot["templates"] += 1
        ok_all = True
        # compile check in oracle: if python itself cannot compile the template, record and skip
        try:
            tpl = env.from_string(src)
        except Exception as e:
            print(f"SKIP {os.path.basename(f)}: oracle cannot compile: {e}")
            tot["templates"] -= 1
            continue
        with cf.ThreadPoolExecutor(16) as ex:
            futs = [(n, ex.submit(render_cpp, binp, f, c)) for n, c in cs]
            for (n, c), (_, fu) in zip(cs, futs):
                py = render_py(tpl, c)
                cp = fu.result()
                tot["cases"] += 1
                if py[0] == "ok" and cp[0] == "ok" and py[1] == cp[1]:
                    tot["exact"] += 1
                elif py[0] == "err" and cp[0] == "err":
                    if py[1] == "raise" and cp[1] == "raise":
                        if py[2] == cp[2]: tot["raise_msg_equal"] += 1
                        else:
                            tot["fail"] += 1; ok_all = False
                            fails.append((os.path.basename(f), n, "raise-message differs", py[2], cp[2]))
                            continue
                    tot["both_err"] += 1
                else:
                    tot["fail"] += 1; ok_all = False
                    fails.append((os.path.basename(f), n, "mismatch", py, cp))
        if ok_all: tot["tpl_pass"] += 1
        elif verbose: print("FAIL", os.path.basename(f))
    for fl in fails[:show if show >= 0 else None]:
        print("----", fl[0], fl[1], fl[2])
        a, b = fl[3], fl[4]
        if isinstance(a, tuple) and a[0] == "ok" and b[0] == "ok":
            x, y = a[1], b[1]
            k = next((j for j in range(min(len(x), len(y))) if x[j] != y[j]), min(len(x), len(y)))
            print("first diff at", k, "\n py :", repr(x[max(0, k - 60):k + 80]), "\n cpp:", repr(y[max(0, k - 60):k + 80]))
        else:
            print(" py :", repr(a)[:600], "\n cpp:", repr(b)[:600])
    failed_templates = sorted({f[0] for f in fails})
    print("failed templates:", failed_templates)
    print(json.dumps(tot))


if __name__ == "__main__":
    main()
