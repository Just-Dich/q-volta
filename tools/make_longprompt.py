#!/usr/bin/env python3 -I
"""Build long-context probe prompts (token id files, int32) from wikitext-2 test text.
  make_longprompt.py TOKENIZER_JSON QV_TOK WIKITEXT.txt OUT_PREFIX N [--needle-depth 0.5] [--needle-pos-after-tokens K]
Writes OUT_PREFIX.wiki.i32 (first N tokens of wikitext) and OUT_PREFIX.niah.i32 (N tokens of wikitext filler with one needle
sentence at the given depth, followed by the retrieval question), plus OUT_PREFIX.niah.json (needle token position, answer)."""
import json, subprocess, struct, sys

tj, qv_tok, wt, out, N = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
depth = float(sys.argv[sys.argv.index('--needle-depth') + 1]) if '--needle-depth' in sys.argv else 0.5

def enc(strs):
    p = subprocess.run([qv_tok, tj, 'encode', 'nospecial'], input='\n'.join(json.dumps(s) for s in strs) + '\n',
                       capture_output=True, text=True, check=True)
    return [json.loads(l) for l in p.stdout.splitlines()]

lines = open(wt, encoding='utf-8').read().split('\n')
chunks, cur = [], ''
for l in lines:
    cur += l + '\n'
    if len(cur) > 8000:
        chunks.append(cur); cur = ''
chunks.append(cur)
ids = []
for c in enc(chunks):
    ids += c
print('wikitext tokens', len(ids))
if len(ids) < N: sys.exit('not enough text')
open(out + '.wiki.i32', 'wb').write(struct.pack('<%di' % N, *ids[:N]))

needle = '\nThe secret passphrase for the vault is "amber-falcon-7421". Remember it.\n'
question = '\n\nQuestion: What is the secret passphrase for the vault?\nAnswer: The secret passphrase for the vault is "'
nd, qd = enc([needle, question])
filler = ids[:N - len(nd) - len(qd)]
at = int(len(filler) * depth)
niah = filler[:at] + nd + filler[at:] + qd
open(out + '.niah.i32', 'wb').write(struct.pack('<%di' % len(niah), *niah))
json.dump({'n': len(niah), 'needle_start': at, 'needle_len': len(nd), 'answer': 'amber-falcon-7421', 'depth': depth,
           'question_len': len(qd)}, open(out + '.niah.json', 'w'))
print('niah tokens', len(niah), 'needle at', at, 'len', len(nd))
