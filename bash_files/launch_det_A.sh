#!/usr/bin/env bash
# Stage-1 session A: all 344 usable EC voltages + trainable weight vector, detached (setsid nohup), log in logs/det_A.log.
# Run from anywhere: bash bash_files/launch_det_A.sh
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs weights/det_A
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PY:-/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python}
setsid nohup "$PY" -u main_det.py --session A --name det_A --output-dir weights/det_A \
    --epochs 100 --batch_size 2 --accumulate 4 --num_workers 6 \
    > logs/det_A.log 2>&1 < /dev/null &
echo "det_A launched, pid $!"
