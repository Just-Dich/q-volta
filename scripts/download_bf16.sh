#!/bin/sh
# Download Qwen3.8-27B BF16 shards (pinned revision) with resume + retries, 3 in parallel.
set -u
D=/workspace/q-volta-models/Qwen3.8-27B
REV=37561816e66b5b9153b1b64c10968c18c1e6f4a0
cd "$D"
seq -f "%02g" 1 18 | xargs -P 3 -I{} sh -c '
  f=model-000{}-of-00018.safetensors
  for t in 1 2 3 4 5 6 7 8 9 10; do
    curl -sfL -C - --retry 5 --retry-delay 5 -o "$f" "https://huggingface.co/Qwen/Qwen3.8-27B/resolve/'$REV'/$f" && { echo "ok $f"; exit 0; }
    echo "retry $t $f"; sleep 5
  done; echo "FAIL $f"'
