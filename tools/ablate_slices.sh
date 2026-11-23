#!/bin/bash
# usage: ablate_slices.sh LOG qv_ablate.py-args...  (relaunch under the shared GPU lock until ALLDONE)
LOG=$1; shift; DIR=$(cd "$(dirname "$0")" && pwd)
while true; do
  flock /tmp/qv_bench.lock env CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True timeout 900 python3 -I "$DIR/qv_ablate.py" "$@" >> "$LOG" 2>&1
  rc=$?; grep -q ALLDONE "$LOG" && break; [ $rc -ne 0 ] && { echo "SLICE FAILED rc=$rc" >> "$LOG"; break; }
done
