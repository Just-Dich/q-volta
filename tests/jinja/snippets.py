#!/usr/bin/env python3
"""Snippet-level differential test: C++ jinja_render vs Python jinja2 (same oracle env as diff_oracle.py).
usage: snippets.py BIN"""
import copy, json, os, sys, tempfile, subprocess
sys.path.insert(0, os.path.dirname(__file__))
import diff_oracle as d

CTX = {
    "s": "Hello, World", "t": "  pad  ", "n": 42, "f": 3.5, "neg": -7, "z": 0, "b": True, "nb": False, "nn": None,
    "l": [3, 1, 2], "ls": ["b", "A", "c"], "e": [], "dd": {"b": 1, "a": 2, "c": {"x": [1, 2]}},
    "objs": [{"name": "x", "v": 1, "t": "a"}, {"name": "y", "v": 2, "t": "b"}, {"name": "z", "v": 3, "t": "a"}],
    "u": "héllo wörld ✓", "csv": "a,b,,c", "multi": "l1\nl2\n\nl4", "words": "the quick  brown fox",
    "msgs": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo", "tool_calls": [{"id": "1", "function": {"name": "f", "arguments": {"q": 1}}}]}],
    "nested": {"a": {"b": {"c": 5}}}, "tup": None,
}

S = [
 # output / literals
 "{{ 1 }} {{ 1.0 }} {{ 1.5e3 }} {{ 1e16 }} {{ 1e-5 }} {{ 0.1 + 0.2 }} {{ 1/3 }} {{ 10/2 }} {{ 7//2 }} {{ -7//2 }} {{ 7%3 }} {{ -7%3 }} {{ 2**10 }} {{ 2**-1 }} {{ 2**0.5 }}",
 "{{ true }} {{ false }} {{ none }} {{ True }} {{ None }} {{ [1, 'a', none, true] }} {{ {'a': 1, 'b': [1,2]} }} {{ (1, 2) }} {{ (1,) }} {{ () }}",
 "{{ 'it\\'s' }} {{ \"say \\\"hi\\\"\" }} {{ 'a\\nb' }} {{ 'tab\\there' }} {{ 'uni\\u00e9' }} {{ 'x' 'y' }} {{ ['it\\'s', 'a\"b'] }}",
 "{{ s }} {{ n }} {{ f }} {{ neg }} {{ l }} {{ dd }} {{ u }} {{ b }} {{ nn }} {{ undefined_var }}|",
 "{{ 3.0 }} {{ 100.0 }} {{ 1e15 }} {{ 123456789012345678 }} {{ 0.000123 }} {{ 5 * 1.0 }} {{ 2.50 }} {{ -0.0 }}",
 # arithmetic & ops
 "{{ n + 1 }} {{ n - f }} {{ n * 2 }} {{ s + '!' }} {{ s * 2 }} {{ l + [9] }} {{ [0] * 3 }} {{ s ~ n ~ b ~ nn }} {{ n // 5 }} {{ -n % 5 }} {{ 'x' ~ 1 + 2 }}",
 "{{ '%s-%d' % ('a', 5) }} {{ '%5.2f|%-4d|%04d' % (3.14159, 7, 42) }} {{ '%(a)s' % {'a': 'v'} }} {{ '%x' % 255 }} {{ '100%%' % () }}",
 "{{ n > 3 }} {{ n == 42 }} {{ n != 42 }} {{ 1 < 2 < 3 }} {{ 1 < 3 < 2 }} {{ 'a' < 'b' }} {{ [1,2] < [1,3] }} {{ 1 == 1.0 }} {{ true == 1 }} {{ 'a' == 'a' }} {{ none == none }} {{ [1] == [1] }} {{ {'a':1} == {'a':1} }}",
 "{{ 1 in l }} {{ 'a' in 'abc' }} {{ 'a' in dd }} {{ 'a' not in dd }} {{ 5 not in l }} {{ 'zz' in s }} {{ nn is none and 1 or 2 }} {{ n and 'yes' }} {{ z or 'dflt' }} {{ not z }} {{ not n }}",
 "{{ 'a' if n else 'b' }} {{ 'a' if z else 'b' }} [{{ 'a' if z }}] {{ (1 if b else 2) + 1 }} {{ 1 if z else 2 if n else 3 }}",
 "{{ -n }} {{ +n }} {{ -f }} {{ - n | abs }} {{ (-n) | abs }} {{ -3 ** 2 }} {{ 2 * 3 + 4 * 5 }} {{ (2 + 3) * 4 }} {{ 10 - 2 - 3 }} {{ 2 ** 3 ** 2 }}",
 # indexing & slicing
 "{{ l[0] }} {{ l[-1] }} {{ l[5] }}| {{ s[0] }} {{ s[-1] }} {{ s[1:4] }} {{ s[:3] }} {{ s[-3:] }} {{ s[::2] }} {{ s[::-1] }} {{ l[1:] }} {{ l[:-1] }} {{ l[::-1] }} {{ l[0:2] }} {{ u[1:5] }} {{ u[::-1] }}",
 "{{ dd['a'] }} {{ dd.a }} {{ dd.c.x[1] }} {{ dd['zz'] }}| {{ nested.a.b.c }} {{ objs[0].name }} {{ objs[-1]['v'] }}",
 "{{ l[1:2:1] }} {{ l[::2] }} {{ s[2:-2] }} {{ s[100:] }}| {{ s[-100:3] }} {{ l[-5:] }} {{ l[1:1] }}|",
 # string methods
 "{{ s.upper() }} {{ s.lower() }} {{ s.title() }} {{ s.capitalize() }} {{ s.swapcase() }} {{ t.strip() }}| {{ t.lstrip() }}| {{ t.rstrip() }}| {{ 'xxhixx'.strip('x') }} {{ 'xxhixx'.lstrip('x') }} {{ 'xxhixx'.rstrip('x') }}",
 "{{ s.split(',') }} {{ s.split() }} {{ words.split() }} {{ csv.split(',') }} {{ csv.split(',', 1) }} {{ csv.rsplit(',', 1) }} {{ multi.splitlines() }} {{ words.split(' ') }} {{ 'a b c d'.rsplit(None, 1) }} {{ 'a b c d'.split(None, 2) }}",
 "{{ s.startswith('Hel') }} {{ s.endswith('ld') }} {{ s.startswith(('x', 'He')) }} {{ s.endswith('x') }} {{ s.startswith('lo', 3) }} {{ s.replace('l', 'L') }} {{ s.replace('l', 'L', 1) }} {{ s.find('o') }} {{ s.find('zz') }} {{ s.rfind('o') }} {{ s.count('l') }} {{ s.index('W') }}",
 "{{ ', '.join(['a', 'b']) }} {{ '-'.join(ls) }} {{ 'abc'.center(9, '*') }} {{ 'ab'.ljust(5, '.') }}| {{ 'ab'.rjust(5) }}| {{ '5'.zfill(3) }} {{ '-5'.zfill(4) }} {{ 'abc'.isalpha() }} {{ '123'.isdigit() }} {{ 'ab1'.isalnum() }} {{ ' '.isspace() }} {{ 'ABC'.isupper() }} {{ 'abc'.islower() }}",
 "{{ '{} and {}'.format('a', 'b') }} {{ '{0}{1}{0}'.format('x', 'y') }} {{ '{name}!'.format(name='bob') }} {{ '{:>5}|{:<5}|{:^5}|{:05d}|{:.2f}|{:,}'.format('a', 'b', 'c', 42, 3.14159, 1234567) }} {{ '{!r}'.format('q') }} {{ '{{x}}'.format() }}",
 "{{ 'a-b'.partition('-') }} {{ 'a-b-c'.rpartition('-') }} {{ 'prefix_x'.removeprefix('prefix_') }} {{ 'x.txt'.removesuffix('.txt') }} {{ u.upper() }} {{ u.title() }} {{ u | length }} {{ u | reverse }}",
 # list / dict methods
 "{% set x = [1, 2] %}{% set _ = x.append(3) %}{{ x }}{% set _ = x.extend([4, 5]) %}{{ x }}{{ x.pop() }}{{ x }}{{ x.pop(0) }}{{ x }}{% set _ = x.insert(1, 9) %}{{ x }}{{ x.index(9) }}{{ x.count(2) }}{% set _ = x.reverse() %}{{ x }}{% set _ = x.remove(9) %}{{ x }}{{ x.copy() }}",
 "{% set d = {'a': 1} %}{% set _ = d.update({'b': 2}) %}{{ d }}{{ d.get('a') }}{{ d.get('zz') }}{{ d.get('zz', 'dflt') }}{{ d.keys() | list }}{{ d.values() | list }}{{ d.items() | list }}{{ d.pop('a') }}{{ d }}{{ d.setdefault('q', 7) }}{{ d }}",
 "{% for k, v in dd.items() %}{{ k }}={{ v }};{% endfor %}|{% for k in dd %}{{ k }},{% endfor %}|{% for k, v in dd | items %}{{ k }}{% endfor %}|{{ dd.keys() | list }}|{{ dd | list }}",
 # filters
 "{{ s | length }} {{ l | length }} {{ dd | length }} {{ e | length }} {{ undefined_var | length }} {{ s | upper }} {{ s | lower }} {{ t | trim }}| {{ s | title }} {{ 'hELLO wORLD' | capitalize }} {{ s | replace('o', '0') }} {{ s | replace('l', 'L', 2) }}",
 "{{ l | first }} {{ l | last }} {{ e | first }}| {{ s | first }} {{ s | last }} {{ l | sort }} {{ l | sort(reverse=true) }} {{ ls | sort }} {{ ls | sort(case_sensitive=true) }} {{ l | reverse | list }} {{ l | min }} {{ l | max }} {{ l | sum }} {{ l | sum(start=10) }} {{ [1.5, 2] | sum }}",
 "{{ l | join(',') }} {{ ls | join }} {{ [1, 2, 3] | join('-') }} {{ objs | join(', ', attribute='name') }} {{ l | list }} {{ s | list }} {{ 'abc' | list | join('|') }} {{ dd | list }}",
 "{{ n | string }} {{ n | string | length }} {{ f | int }} {{ '12' | int }} {{ 'x' | int }} {{ 'x' | int(5) }} {{ '3.9' | int }} {{ ' 7 ' | int }} {{ '2.5' | float }} {{ 'x' | float }} {{ 3 | float }} {{ nn | int }} {{ b | int }} {{ -3.9 | int }} {{ '0x10' | int }} {{ '10' | int(base=2) }}",
 "{{ undefined_var | default('d') }} {{ nn | default('d') }} {{ z | default('d', true) }} {{ z | default('d') }} {{ s | default('d') }} {{ undefined_var | default }}| {{ undefined_var | d('x') }} {{ '' | default('e', boolean=true) }}",
 "{{ dd | tojson }} {{ l | tojson }} {{ s | tojson }} {{ u | tojson }} {{ n | tojson }} {{ f | tojson }} {{ b | tojson }} {{ nn | tojson }} {{ dd | tojson(indent=2) }} {{ dd | tojson(sort_keys=true) }} {{ dd | tojson(separators=(',', ':')) }} {{ {'k': [1, {'z': 'é\"\\\\\\n'}]} | tojson }} {{ 'a<b>&\"' | tojson }} {{ e | tojson }} {{ {} | tojson }} {{ {} | tojson(indent=2) }} {{ [] | tojson(indent=2) }}",
 "{{ dd | tojson(true) }} {{ u | tojson(true) }} {{ [1,[2,[3]]] | tojson(indent=4) }} {{ f | tojson }} {{ 1e20 | tojson }} {{ 0.1 | tojson }} {{ [1.0, 2.5] | tojson }}",
 "{{ objs | map(attribute='name') | list }} {{ objs | map(attribute='name') | join(',') }} {{ ls | map('upper') | list }} {{ ls | map('length') | list }} {{ l | map('string') | join('') }} {{ objs | selectattr('t', 'equalto', 'a') | map(attribute='name') | list }} {{ objs | rejectattr('t', 'equalto', 'a') | map(attribute='name') | list }}",
 "{{ objs | selectattr('v', 'gt', 1) | list | length }} {{ objs | selectattr('v') | list | length }} {{ objs | rejectattr('v', 'lt', 2) | list | length }} {{ l | select('odd') | list }} {{ l | reject('odd') | list }} {{ [0, 1, '', 'a', none] | select | list }} {{ [0, 1, '', 'a', none] | reject | list }} {{ ls | select('lower') | list }}",
 "{{ objs | sort(attribute='v', reverse=true) | map(attribute='name') | join }} {{ objs | map(attribute='v') | max }} {{ objs | map(attribute='v') | min }} {{ objs | sum(attribute='v') }} {{ objs | groupby('t') | map(attribute='grouper') | list }} {{ objs | unique(attribute='t') | list | length }} {{ [1, 2, 1, 3, 2] | unique | list }}",
 "{{ dd | dictsort }} {{ dd | dictsort(reverse=true) }} {{ {'B': 1, 'a': 2} | dictsort }} {{ {'B': 1, 'a': 2} | dictsort(case_sensitive=true) }} {{ {'x': 3, 'y': 1} | dictsort(by='value') }}",
 "{{ 3.14159 | round(2) }} {{ 3.5 | round }} {{ 2.5 | round }} {{ 4.5 | round }} {{ 3.7 | round(method='floor') }} {{ 3.2 | round(method='ceil') }} {{ 5 | round }} {{ -2.5 | round }} {{ neg | abs }} {{ -3.5 | abs }} {{ 1234.5678 | round(-2) }}",
 "{{ '<b>&' | e }} {{ '<b>&' | escape }} {{ '<b>' | safe }} {{ 'x' | forceescape }} {{ '<p>hi <b>there</b></p>' | striptags }} {{ 'a b' | urlencode }} {{ 'a b/c?d' | urlencode }} {{ 'one two three' | wordcount }}",
 "{{ s | indent(2) }}|{{ multi | indent(2) }}|{{ multi | indent(2, true) }}|{{ multi | indent(2, true, true) }}|{{ multi | indent('--') }}|{{ multi | indent(width=3, first=true) }}",
 "{{ 'a b c d e f' | truncate(7) }} {{ 'a b c d e f' | truncate(7, true) }} {{ 'short' | truncate(20) }} {{ 'abcdefghijkl' | truncate(8, true, '..') }} {{ 'hi' | center(10) }}| {{ l | batch(2) | list }} {{ l | slice(2) | list }} {{ [1,2,3,4,5] | batch(2, 0) | list }}",
 "{{ s | attr('upper')() }} {{ objs[0] | attr('name') }} {{ 5 | string | attr('zfill')(3) }} {{ 'x' | format() }} {{ '%s %s' | format('a', 'b') }} {{ 'a' | filesizeformat }}",
 # tests
 "{{ s is string }} {{ n is string }} {{ s is not string }} {{ n is number }} {{ f is number }} {{ b is number }} {{ n is integer }} {{ f is float }} {{ b is boolean }} {{ n is boolean }} {{ nn is none }} {{ s is none }} {{ l is iterable }} {{ s is iterable }} {{ n is iterable }} {{ dd is mapping }} {{ l is mapping }} {{ l is sequence }} {{ s is sequence }} {{ dd is sequence }} {{ n is sequence }}",
 "{{ undefined_var is defined }} {{ undefined_var is undefined }} {{ s is defined }} {{ nn is defined }} {{ dd.zz is defined }} {{ dd.a is defined }} {{ undefined_var is iterable }} {{ undefined_var is sequence }} {{ undefined_var is string }} {{ undefined_var is none }} {{ undefined_var is mapping }}",
 "{{ 4 is even }} {{ 3 is odd }} {{ 9 is divisibleby 3 }} {{ 9 is divisibleby(2) }} {{ n is eq 42 }} {{ n is ne 42 }} {{ n is gt 4 }} {{ n is lt 4 }} {{ n is ge 42 }} {{ n is le 41 }} {{ 3 is in [1, 3] }} {{ none is sameas none }} {{ true is sameas true }} {{ true is true }} {{ false is false }} {{ n is true }} {{ s is lower }} {{ 'ABC' is upper }} {{ s is callable }} {{ s is not none }} {{ nn is not none }}",
 "{{ 1 is defined and 2 is defined }} {{ not s is defined }} {{ s is defined and s is string and s | length > 3 }} {{ 5 is not divisibleby 2 }} {{ (1 + 2) is odd }}",
 # control flow
 "{% for x in l %}{{ x }}{% if not loop.last %},{% endif %}{% endfor %}|{% for x in e %}x{% else %}empty{% endfor %}|{% for x in l if x > 1 %}{{ x }}{% endfor %}|{% for x in l %}{% if x == 1 %}{% continue %}{% endif %}{{ x }}{% endfor %}|{% for x in l %}{% if x == 1 %}{% break %}{% endif %}{{ x }}{% endfor %}",
 "{% for x in ls %}{{ loop.index }}/{{ loop.index0 }}/{{ loop.revindex }}/{{ loop.revindex0 }}/{{ loop.first }}/{{ loop.last }}/{{ loop.length }}/{{ loop.previtem }}/{{ loop.nextitem }};{% endfor %}",
 "{% for x in l %}{{ loop.cycle('a', 'b') }}{{ loop.changed(x) }}{% endfor %}|{% for i in range(3) %}{{ i }}{% endfor %}|{% for i in range(1, 6, 2) %}{{ i }}{% endfor %}|{% for i in range(5, 0, -2) %}{{ i }}{% endfor %}|{{ range(3) | list }}",
 "{% for a, b in [(1, 2), (3, 4)] %}{{ a + b }},{% endfor %}|{% for k, v in dd.items() if v is number %}{{ k }}{% endfor %}|{% for c in 'abc' %}{{ c }}-{% endfor %}|{% for x in undefined_var %}{{ x }}{% endfor %}ok",
 "{% for o in objs %}{% for k in o %}{{ loop.index }}{{ k }}{% endfor %}.{{ loop.index }}{% endfor %}",
 "{% set ns = namespace(c=0, items=[]) %}{% for x in l %}{% set ns.c = ns.c + x %}{% set _ = ns.items.append(x * 2) %}{% endfor %}{{ ns.c }} {{ ns.items }} {% set ns.z = 5 %}{{ ns.z }}",
 "{% set x = 1 %}{% for i in range(3) %}{% set x = x + i %}{{ x }}{% endfor %}-{{ x }}|{% set y = 'a' %}{% if true %}{% set y = 'b' %}{% endif %}{{ y }}",
 "{% set a, b = 1, 2 %}{{ a }}{{ b }}{% set (c, d) = [3, 4] %}{{ c }}{{ d }}{% set blk %}inner {{ n }}{% endset %}[{{ blk }}]{% set up | upper %}shout{% endset %}{{ up }}{% set lst = [1, 2] + [3] %}{{ lst }}",
 "{% if n > 100 %}big{% elif n > 40 %}mid{% else %}small{% endif %} {% if z %}z{% elif nn %}nn{% elif e %}e{% else %}none{% endif %} {% if l and not e %}ok{% endif %} {% if 'a' in dd and dd.a == 2 %}yes{% endif %}",
 "{% macro m(a, b=2, *rest) %}{{ a }}-{{ b }}{% endmacro %}{{ m(1) }} {{ m(1, 3) }} {{ m(b=5, a=0) }} {{ m('x', b='y') }}",
 "{% macro greet(name, greeting='Hello') -%}{{ greeting }}, {{ name }}!{%- endmacro %}{{ greet('Bob') }}|{{ greet('Al', greeting='Hi') }}|{{ greet() }}|{% set g = greet('X') %}{{ g | length }}",
 "{% macro rec(n) %}{% if n > 0 %}{{ n }}{{ rec(n - 1) }}{% endif %}{% endmacro %}{{ rec(4) }}{% macro outer() %}{% macro inner() %}in{% endmacro %}{{ inner() }}{% endmacro %}{{ outer() }}",
 "{% macro wrap() %}[{{ caller() }}]{% endmacro %}{% call wrap() %}body {{ n }}{% endcall %}{% macro lst(items) %}{% for i in items %}{{ caller(i) }}{% endfor %}{% endmacro %}{% call(i) lst([1, 2]) %}<{{ i }}>{% endcall %}",
 "{% filter upper %}hello {{ s }}{% endfilter %}|{% filter replace('a', 'b') | upper %}banana{% endfilter %}|{% with a = 5 %}{{ a }}{% endwith %}{{ a }}|{% do l.append(1) %}{{ l }}",
 "{% raw %}{{ not rendered }} {% if %}{% endraw %}|{# comment #}x{# multi\nline #}y|{{ '{{' }} {{ '{%' }}",
 "{% set d = {'a': {'b': [1, 2, {'c': 3}]}} %}{{ d.a.b[2].c }} {{ d['a']['b'][0] }} {{ d | tojson }} {% set k = 'a' %}{{ d[k].b | length }}",
 # whitespace control
 "a  {%- if true -%}  b  {%- endif -%}  c\n{{- 'd' -}}  \ne",
 "line1\n  {% if true %}\n  inner\n  {% endif %}\nline2\n",
 "x\n{% for i in l %}\n{{ i }}\n{% endfor %}\ny",
 "a {{- 'b' }} c {{ 'd' -}} e {%- set q = 1 -%} f\n\n  {%+ if true %}+{% endif %}\n",
 "  {# c #}\n  text\n    {%- set a = 1 %}\n{{ a }}\n   {% if true %}x{% endif %}   \nend",
 "{%- for x in [1,2] -%}\n  {{ x }}\n{%- endfor %}\n",
 "{% for x in [1,2] %}\n  - {{ x }}\n{% endfor %}\ndone\n\n",
 "a\n\n{% if false %}x{% endif %}\n\nb",
 "{{ 'a' }}\n{{ 'b' }}\n\n{{ 'c' }}\n",
 # misc python-isms
 "{{ [1, 2, 3] | string }} {{ {'a': 'b'} | string }} {{ ['a', \"it's\"] | string }} {{ (1, 'a') | string }} {{ none | string }} {{ true | string }} {{ 1.5 | string }} {{ 'é' | string }} {{ ['é\\n'] }}",
 "{{ dd.items() | list | length }} {{ l | length > 2 }} {{ objs | length }} {{ msgs | length }} {{ msgs[1].tool_calls[0].function.arguments }} {{ msgs[1].tool_calls[0].function.arguments | tojson }} {{ msgs | selectattr('role', 'equalto', 'user') | list | length }}",
 "{{ msgs[0].content.startswith('h') }} {{ msgs[1].tool_calls is defined }} {{ msgs[0].tool_calls is defined }} {{ msgs[0].tool_calls | default([]) | length }} {{ msgs[0].get('role') }} {{ msgs[0].get('nope', 'dflt') }} {{ msgs[0]['nope'] is defined }}",
 "{{ strftime_now('%Y-%m-%d %H:%M:%S') }} {{ strftime_now('%d %B %Y') }} {{ strftime_now('%A') }}",
 "{{ raise_exception('boom: ' ~ n) }}",
 "{{ dict(a=1, b=2) }} {{ dict([('x', 1)]) }} {{ namespace(a=1).a }} {{ range(2) | list }}",
 "{% set x = 5 %}{{ x if x is defined else 'u' }} {{ loop is defined }}",
 "{{ ('a' ~ 'b') | upper }} {{ 'a' ~ ('b' | upper) }} {{ 1 + 2 ~ 3 }} {{ s ~ ': ' ~ n | string }} {{ n | string ~ 'x' }}",
 "{{ l | map('int') | list }} {{ ['1', '2'] | map('int') | sum }} {{ ['a', 'b'] | map('upper') | join }} {{ objs | map(attribute='nope') | list }} {{ objs | map(attribute='nope', default='d') | list }}",
 "{{ x.y }}",
 "{{ n.x }}| {{ nn.x }}| {{ dd.zz.yy }}",
 "{{ l.bad() }}",
 "{{ s + n }}",
 "{{ n + s }}",
 "{{ undefined_var + 1 }}",
 "{{ undefined_var.attr }}",
 "{{ undefined_var() }}",
 "{{ 1 / 0 }}",
 "{{ [1] < 'a' }}",
 "{{ 'abc' | unknown_filter }}",
 "{% if %}x{% endif %}",
 "{% for x in n %}{% endfor %}",
 "{% for x in nn %}{% endfor %}",
 "{{ 'unclosed }}",
 "{% unknown_tag %}",
 "{% for a, b in l %}{% endfor %}",
]

def main():
    binp = sys.argv[1]
    env = d.make_env()
    ok = bad = 0
    fails = []
    for i, src in enumerate(S):
        with tempfile.NamedTemporaryFile("w", suffix=".jinja", delete=False, encoding="utf-8") as f:
            f.write(src); path = f.name
        try:
            try:
                tpl = env.from_string(src)
                py = d.render_py(tpl, copy.deepcopy(CTX))
            except Exception as e:
                py = ("err", "other", "compile: " + str(e))
            cp = d.render_cpp(binp, path, copy.deepcopy(CTX))
        finally:
            os.unlink(path)
        if py[0] == "ok" and cp[0] == "ok" and py[1] == cp[1]: ok += 1
        elif py[0] == "err" and cp[0] == "err" and (py[1] != "raise" or py[2] == cp[2]): ok += 1
        else:
            bad += 1; fails.append((i, src, py, cp))
    for i, src, py, cp in fails:
        print(f"--- #{i}: {src[:200]!r}")
        if py[0] == "ok" and cp[0] == "ok":
            x, y = py[1], cp[1]
            k = next((j for j in range(min(len(x), len(y))) if x[j] != y[j]), min(len(x), len(y)))
            print("  first diff at", k, "\n  py :", repr(x[max(0, k - 50):k + 90]), "\n  cpp:", repr(y[max(0, k - 50):k + 90]))
        else:
            print("  py :", repr(py)[:300], "\n  cpp:", repr(cp)[:300])
    print(json.dumps({"snippets": len(S), "match": ok, "mismatch": bad}))
    return 1 if bad else 0

if __name__ == "__main__":
    sys.exit(main())
