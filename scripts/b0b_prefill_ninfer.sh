#!/bin/sh
# NInfer prompt throughput on THIS V100: same wikitext prompts as logs/P{512,2048,4088}.i32 (decoded to text), 3 reps, 1 new token.
M=/workspace/q-volta-models/ninfer-iq3s/Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer; cd /workspace/q-volta
CUDA_VISIBLE_DEVICES=0 /workspace/ninfer-all/build/apps/ninfer-serve $M --model-id qwen3.8-27b --host 127.0.0.1 --port 8000 --max-context 8192 --kv-capacity 8192 \
  --kv-dtype rk4v4 --max-concurrency 1 --gdn-state-fp16 --prefill-chunk 1024 --default-max-tokens 8 > logs/b0b-ninfer.server.log 2>&1 &
pid=$!; for i in $(seq 1 120); do curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 && break; sleep 2; done
python3 -I - <<'PY' | tee logs/b0b-ninfer.out
import json,urllib.request,statistics as st
def ask(txt):
    r=urllib.request.Request('http://127.0.0.1:8000/v1/completions',data=json.dumps({'model':'qwen3.8-27b','prompt':txt,'max_tokens':1,'temperature':0}).encode(),headers={'Content-Type':'application/json'})
    return json.load(urllib.request.urlopen(r,timeout=300)).get('timings',{})
ask('warmup '+open('/workspace/q-volta/logs/P512.txt').read())          # untimed: CUDA/cuBLAS init
for n in (512,2048,4088):
    txt=open(f'/workspace/q-volta/logs/P{n}.txt').read(); v=[]
    for rep in range(4):   # unique first tokens each rep => prefix cache cannot hit; prompt_n must be ~n
        t=ask(f'[{n}-{rep}-{rep*7+n}] '+txt); assert t['prompt_n']>n*0.9,t; v.append(t['prompt_per_second'])
    print(n,'prompt_n',t['prompt_n'],'tok/s',[round(x,1) for x in v],'median %.1f'%st.median(v))
PY
kill $pid; wait $pid 2>/dev/null; echo finished > logs/b0b.done
