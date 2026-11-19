#!/bin/sh
# Wait for all BF16 shards, verify sizes against the HF API, then run the E0 converter + reference pass.
set -u
D=/workspace/q-volta-models/Qwen3.8-27B
while [ "$(grep -c '^ok ' /workspace/q-volta/logs/download.log)" -lt 18 ]; do
  if grep -q '^FAIL' /workspace/q-volta/logs/download.log; then echo "download FAILED"; exit 1; fi
  sleep 20
done
curl -s -m 60 "https://huggingface.co/api/models/Qwen/Qwen3.8-27B/tree/37561816e66b5b9153b1b64c10968c18c1e6f4a0" > /tmp/claude-0/-workspace-research/3056eb9f-7a9f-4763-a99a-2823653ecfd6/scratchpad/tree.json
python3 -I - "$D" /tmp/claude-0/-workspace-research/3056eb9f-7a9f-4763-a99a-2823653ecfd6/scratchpad/tree.json <<'PY'
import json, os, sys
d, tree = sys.argv[1], json.load(open(sys.argv[2]))
bad = [t["path"] for t in tree if t["path"].endswith(".safetensors") and os.path.getsize(os.path.join(d, t["path"])) != t["size"]]
print("size check:", "OK" if not bad else f"MISMATCH {bad}")
sys.exit(1 if bad else 0)
PY
[ $? -eq 0 ] || exit 1
cd /workspace/q-volta && rm -f /workspace/q-volta-models/qv/qwen38-27b-e0.qv.*
CUDA_VISIBLE_DEVICES=0 python3 -I tools/convert.py > logs/convert-e0.log 2>&1
echo "convert exit=$?"; tail -25 logs/convert-e0.log
