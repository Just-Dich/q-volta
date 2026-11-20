#!/usr/bin/env python3
"""Compare qv_tok (C++ tokenizer) with HF AutoTokenizer on a generated corpus.
usage: python3 -I tests/test_tokenizer.py BUILD/qv_tok [MODEL_DIR]
Checks encode ids (>= 200 strings) and decode text (corpus ids + random id sequences)."""
import json, random, subprocess, sys

qv_tok = sys.argv[1]
model_dir = sys.argv[2] if len(sys.argv) > 2 else "/workspace/q-volta-models/Qwen3.8-27B"
import unicodedata
from transformers import AutoTokenizer
from tokenizers import Tokenizer as RawTokenizer
hf = AutoTokenizer.from_pretrained(model_dir)
# Reference 1: the `tokenizers` library loading tokenizer.json verbatim (the pattern with \\p{M}, as the model was trained).
# Reference 2: transformers AutoTokenizer. NOTE: transformers 5.17's Qwen2Tokenizer silently re-builds the pre-tokenizer
# with a pattern in which \\p{M} never matches, so it differs from tokenizer.json on strings containing combining marks.
raw = RawTokenizer.from_file(model_dir + "/tokenizer.json")

fixed = [
    "", " ", "  ", "\n", "\n\n", "\t", "a", "Hello, world!", "Hello world", " Hello  world ", "The quick brown fox jumps over the lazy dog.",
    "I'm sure it's fine; we'll see, they've said you'd go. HE'S SHE'LL WE'RE 'S 'tis",
    "Привет, мир! Как дела? Это тест токенизатора.", "Съешь ещё этих мягких французских булок, да выпей чаю.",
    "你好，世界！这是一个测试。今天天气很好。", "日本語のテキストです。ひらがなとカタカナ、漢字。", "한국어 텍스트입니다.",
    "مرحبا بالعالم", "שלום עולם", "नमस्ते दुनिया", "สวัสดีชาวโลก",
    "emoji: 😀 🎉 👨‍👩‍👧‍👦 🇷🇺 ❤️ 👍🏽", "😀😀😀😀", "e\u0301 café cafe\u0301 \u00e9 \u0065\u0301\u0323", "Å \u212b A\u030a \uac00 \u1100\u1161\u11a8",
    "def f(x):\n\treturn x * 2\n\n\nclass A:\n    def g(self):\n        pass\n", "int main() {\r\n\tprintf(\"hi\\n\");\r\n\treturn 0;\r\n}\r\n",
    "for (i = 0; i < n; ++i) { a[i] += b[i] * 0.5f; }", "SELECT * FROM t WHERE a >= 10 AND b <> 'x';",
    "1234567890", "3.14159 2.71828 1e-9 -42 +7 1,000,000", "0123 456 7890123456789", "١٢٣ ४५६ ½ ² ③",
    "a   b", "a    b     c", "a \n b", "a\n \nb", "x\n\n\n\ny", "trailing spaces   ", "   leading", " \t \t ", "\r\n\r\n", "a\r\nb\rc",
    "----", "====>", "...!!!???", "a.b.c", "foo_bar baz-qux", "@#$%^&*()", "<html><body>x</body></html>", "{\"a\": [1, 2, {\"b\": null}]}",
    "https://example.com/path?q=1&r=2#frag", "user@example.com", "C:\\Users\\me\\file.txt", "/usr/bin/env python3",
    "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n", "<|endoftext|>", "a<|endoftext|>b", "<think>\nhmm\n</think>\n\nanswer",
    "<|im_start|><|im_end|>", "<tool_call>\n<function=f>\n</function>\n</tool_call>", "<|fim_prefix|>x<|fim_suffix|>y<|fim_middle|>",
    "<|not_a_special|>", "<think", "<|im_start", "< think>", "<<think>>", "<|vision_start|><|image_pad|><|vision_end|>",
    "word" * 50, "a" * 300, " " * 100, "\n" * 40, "ab " * 80, "é" * 64, "你" * 70, "!" * 90, "1" * 33,
    "\u00a0nbsp\u00a0here", "\u2003em\u2003space", "a\u200bb zero width", "\u3000fullwidth space", "tab\there\tand\tthere", "\x0b\x0c form feed",
    "\u0300 lone combining", "a\u0300\u0301\u0302", "ǅ ǆ ǈ ƛ ʰ ˆ", "Ⅷ ⅷ ㈠ ①", "fi ﬁ ﬂ", "ä ö ü ß", "naïve résumé coöperate",
    "'s", "'S", "'re 'RE 'Ve 'LL 'm 'D", "don't", "DON'T", "rock'n'roll", "'x", "''", "'", "\u2019s", "it\u2019s",
    "Mr. Smith's 3rd place — a good one…", "“quoted” ‘single’ «guillemets» „German“", "α β γ δ Ω ∑ ∫ ≠ ≤ → ∞",
    "line1\nline2\r\nline3\n\n\tindented\n    four\n", "\n  \n  x", "a\n\n  b", "x \n", "x  \n\n",
]
rng = random.Random(1234)
ranges = [(0x20, 0x7e), (0x20, 0x7e), (0x20, 0x7e), (0xa1, 0x24f), (0x370, 0x3ff), (0x400, 0x4ff), (0x590, 0x6ff), (0x900, 0x97f),
          (0xe00, 0xe7f), (0x2000, 0x206f), (0x2190, 0x23ff), (0x2500, 0x27bf), (0x3040, 0x30ff), (0x4e00, 0x9fff), (0xac00, 0xd7a3),
          (0x1f300, 0x1f64f), (0x300, 0x36f), (0x1d400, 0x1d7ff), (0x10000, 0x1007f)]
ws = [" ", "  ", "\n", "\t", "\r\n", " \n", "\n\n", "   ", "\u00a0", "\u3000"]
words = ["the", "of", "Hello", "world", "int", "return", "model", "token", "x", "if", "Привет", "мир", "你好", "数据", "'s", "n't", "42", "3.5"]
def rand_char():
    lo, hi = rng.choice(ranges)
    while True:
        c = rng.randint(lo, hi)
        if not (0xd800 <= c <= 0xdfff) and chr(c).isprintable():
            return chr(c)
def rand_string():
    parts = []
    for _ in range(rng.randint(1, 14)):
        r = rng.random()
        if r < 0.35: parts.append(rng.choice(words))
        elif r < 0.55: parts.append(rng.choice(ws))
        elif r < 0.9: parts.append("".join(rand_char() for _ in range(rng.randint(1, 6))))
        else: parts.append(rng.choice(["<|im_start|>", "<|im_end|>", "<think>", "</think>", "<|endoftext|>", "\n", "'s"]))
    return "".join(parts)
corpus = fixed + [rand_string() for _ in range(400)]
assert len(corpus) >= 200

def run(mode, lines, extra=()):
    p = subprocess.run([qv_tok, model_dir + "/tokenizer.json", mode, *extra], input="\n".join(json.dumps(l, ensure_ascii=False) for l in lines) + "\n",
                       capture_output=True, text=True, check=True)
    return [json.loads(x) for x in p.stdout.splitlines()]

got = run("encode", corpus)
bad = 0
auto_diff = auto_diff_nomark = 0
for s, g in zip(corpus, got):
    want = raw.encode(s, add_special_tokens=False).ids
    if g != want:
        bad += 1
        print("ENCODE MISMATCH vs tokenizers", json.dumps(s, ensure_ascii=False)[:120], "\n  ref:", want[:30], "\n  qv:", g[:30])
    if g != hf(s, add_special_tokens=False)["input_ids"]:
        auto_diff += 1
        if not any(unicodedata.category(c).startswith("M") for c in unicodedata.normalize("NFD", s)):
            auto_diff_nomark += 1
            print("ENCODE MISMATCH vs AutoTokenizer (no marks in string)", json.dumps(s, ensure_ascii=False)[:120])
print(f"encode vs tokenizers(tokenizer.json): {len(corpus) - bad}/{len(corpus)} exact ({100.0 * (len(corpus) - bad) / len(corpus):.2f}%)")
print(f"encode vs transformers AutoTokenizer: {len(corpus) - auto_diff}/{len(corpus)} exact; {auto_diff} differ, "
      f"{auto_diff_nomark} of those without any combining mark")
bad += auto_diff_nomark

# decode: corpus ids + random id sequences (random sequences split multi-byte characters -> exercises U+FFFD handling)
seqs = [hf(s, add_special_tokens=False)["input_ids"] for s in corpus[:200]]
V = 248044
for _ in range(100):
    seqs.append([rng.randint(0, V - 1) if rng.random() < 0.9 else rng.randint(248044, 248076) for _ in range(rng.randint(1, 12))])
dec = run("decode", seqs)
dbad = 0
for ids, g in zip(seqs, dec):
    want = hf.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if g != want:
        dbad += 1
        print("DECODE MISMATCH", ids[:12], json.dumps(want)[:100], json.dumps(g)[:100])
print(f"decode: {len(seqs) - dbad}/{len(seqs)} exact")
sys.exit(1 if bad or dbad else 0)
