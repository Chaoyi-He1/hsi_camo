#!/usr/bin/env bash
# Stage-1 session B: top-k voltages from session A, fixed channels, no weight vector, initialised from an A checkpoint.
# The ranking/checkpoint default to the converged epoch-89 artefacts of det_A run #6 (see spec §12); override with
#   A_CKPT=weights/det_A/model_99 A_RANK=weights/det_A/gate_ranking_ep99.csv bash bash_files/launch_det_B.sh
cd "$(dirname "$0")/.." || exit 1
A_CKPT=${A_CKPT:-weights/det_A/model_89}
A_RANK=${A_RANK:-weights/det_A/gate_ranking_ep89.csv}
test -f "$A_RANK" || { echo "$A_RANK missing: session A not finished (or run tmp/rank_from_ckpt.py)"; exit 1; }
mkdir -p logs weights/det_B
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PY:-/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python}
setsid nohup "$PY" -u main_det.py --session B --top_k 10 --ranking "$A_RANK" --resume "$A_CKPT" \
    --name det_B --output-dir weights/det_B \
    --epochs 100 --batch_size 2 --accumulate 4 --num_workers 6 \
    > logs/det_B.log 2>&1 < /dev/null &
echo "det_B launched, pid $!"
