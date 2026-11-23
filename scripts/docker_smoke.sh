#!/bin/sh
# Smoke test for the Q-Volta image: build, start with a mounted model dir, query /health and one chat completion.
#   scripts/docker_smoke.sh [cuda|cpu] /path/to/models      (models dir: one .qv file + tokenizer.json)
set -eu
target=${1:-cuda}; models=${2:?usage: docker_smoke.sh [cuda|cpu] MODELS_DIR}
tag=q-volta:smoke-$target
docker build -t "$tag" --target "$target" "$(dirname "$0")/.."
gpu=""; [ "$target" = cuda ] && gpu="--gpus all"
cid=$(docker run -d $gpu -p 18080:8080 -v "$models":/models:ro "$tag")
trap 'docker logs --tail 20 "$cid" || true; docker rm -f "$cid" >/dev/null' EXIT
for i in $(seq 1 120); do curl -fsS http://127.0.0.1:18080/health >/dev/null 2>&1 && break; sleep 5; done
curl -fsS http://127.0.0.1:18080/health; echo
curl -fsS http://127.0.0.1:18080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hi in five words."}],"max_tokens":24,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'
echo; echo "docker smoke OK ($target)"
