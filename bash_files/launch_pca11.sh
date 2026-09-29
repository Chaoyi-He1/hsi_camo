#!/usr/bin/env bash
# Decisive follow-up for the filter path (spec §12): the filter model trained from scratch on the top-K PCA-whitened
# response directions (K=11 spans 99.9 % of the response matrix), no gate -- a well-conditioned, information-complete
# input. Parity with raw133_A => the pipeline loss was conditioning; a plateau near AP50 0.5 => the sensor's bandwidth.
# Alternative control: UNIFORM=12 bash bash_files/launch_pca11.sh  (12 uniformly spaced voltages, no gate, no PCA).
cd "$(dirname "$0")/.." || exit 1
K=${K:-11}
if [ -n "${UNIFORM:-}" ]; then
  NAME=uni${UNIFORM}_A; EXTRA="--filter-select uniform --num-filters $UNIFORM --no-gate"
else
  NAME=pca${K}_A; EXTRA="--pca-channels $K"
fi
mkdir -p logs "weights/$NAME"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
PY=${PY:-/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python}
setsid nohup "$PY" -u main_det.py --session A $EXTRA --name "$NAME" --output-dir "weights/$NAME" \
    --epochs 100 --batch_size 2 --accumulate 4 --num_workers "${NUM_WORKERS:-6}" \
    > "logs/$NAME.log" 2>&1 < /dev/null &
echo "$NAME launched, pid $!"
