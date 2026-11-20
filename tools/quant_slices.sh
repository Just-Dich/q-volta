#!/bin/bash
# Run convert.py in lock-sliced chunks (the V100 and CPU are shared; each slice holds /tmp/qv_bench.lock < 10 min).
# usage: quant_slices.sh LOG LAYERS_PER_SLICE convert.py-args...   (relaunches with --resume until the run prints DONE)
LOG=$1; N=$2; shift 2
DIR=$(cd "$(dirname "$0")" && pwd)
while true; do
  flock /tmp/qv_bench.lock env CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True timeout 900 python3 -I "$DIR/convert.py" "$@" --resume --max-layers "$N" >> "$LOG" 2>&1
  rc=$?
  if grep -q -E "^\[ *[0-9]+s\] DONE|^sens written" "$LOG"; then break; fi
  if [ $rc -ne 0 ]; then echo "SLICE FAILED rc=$rc" >> "$LOG"; break; fi
done
