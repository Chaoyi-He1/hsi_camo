#!/usr/bin/env bash
# Control run: the raw 133 cube bands (400-800 nm, the EC filter's window) straight into YOLO26s -- identity
# projection, no weight vector, otherwise the session-A recipe. Compares against det_A (spec §12).
# GPU and worker count are env-overridable; fewer workers when another full-resolution job shares the box.
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs weights/raw133_A
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
PY=${PY:-/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python}
setsid nohup "$PY" -u main_det.py --session A --raw-bands --name raw133_A --output-dir weights/raw133_A \
    --epochs 100 --batch_size 2 --accumulate 4 --num_workers "${NUM_WORKERS:-3}" \
    > logs/raw133_A.log 2>&1 < /dev/null &
echo "raw133_A launched, pid $!"
