#!/bin/sh
# Q-Volta container entrypoint: fills in --model/--tokenizer from /models when they are not given, then runs qv_server.
#   /models/model.qv, or the only/first *.qv file in /models   (override: --model PATH or LLAMA_ARG_MODEL)
#   /models/tokenizer.json                                      (override: --tokenizer PATH)
set -eu
has() { for a in "$@"; do case "$a" in "$HAS"|"$HAS"=*) return 0;; esac; done; return 1; }
args="$*"
extra=""
HAS=--model
if ! has "$@" && [ -z "${LLAMA_ARG_MODEL:-}" ] && ! echo " $args " | grep -q ' -m '; then
  m=/models/model.qv
  [ -f "$m" ] || m=$(ls /models/*.qv 2>/dev/null | head -n1 || true)
  [ -n "$m" ] || { echo "qv: no model: mount a directory with a .qv file at /models (or pass --model PATH)" >&2; exit 2; }
  extra="$extra --model $m"
fi
HAS=--tokenizer
if ! has "$@" && [ -f /models/tokenizer.json ]; then extra="$extra --tokenizer /models/tokenizer.json"; fi
# shellcheck disable=SC2086
exec /usr/local/bin/qv_server --host 0.0.0.0 --port "${QV_PORT:-8080}" $extra "$@"
