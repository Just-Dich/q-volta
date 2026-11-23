#!/bin/bash
# Q-Volta e0/e1 quality (ppl 4088 + KL vs the Q8_0 reference, ppl 8x512) then the imatrix, all in ONE lock hold:  flock /tmp/qv_bench.lock bash block_quality_qv.sh
export QV_HELD=1; D=$(dirname "$0")
bash $D/quality_qv.sh /workspace/q-volta-models/qv/qwen38-27b-e0.qv qv_e0
bash $D/quality_qv.sh /workspace/q-volta-models/qv/qwen38-27b-e1.qv qv_e1
bash $D/imatrix.sh
