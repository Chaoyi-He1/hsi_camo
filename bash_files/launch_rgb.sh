#!/usr/bin/env bash
# RGB baseline (Stage 1): is hyperspectral / EC input better than a plain camera? The dataset's own RGB/<id>.jpg (3 channels,
# /255, standardised with the train RGB statistics, no gate, no read noise) goes into the same YOLO26s with the unchanged COCO
# stem (3 input channels, so the exact pretrained kernel), the same 100-epoch recipe as every other Stage-1 run, then the
# fixed-checkpoint comparison against the hyperspectral runs. The cube cache is not read by rgb_A; the comparison reads it for
# the four cube runs and the JPEGs for rgb_A, frame by frame.
#   rgb_A  --session A --rgb_images        -> weights/rgb_A, log logs/rgb_A.log
#   main_det_compare.py --runs rgb_A raw133_A sel10g_clean_A sel24g_clean_A pca11_A (model_59..99, clean readings)
#                                          -> results/det/compare_rgb, log logs/compare_rgb.log
#   bash bash_files/launch_rgb.sh          # detached; queue log logs/rgb_queue.log
# A 'final' line in weights/rgb_A/results_rgb_A.txt skips the training (the comparison still runs), so it can be relaunched after
# a crash of the comparison; an interrupted training restarts from epoch 0. PY, GPU (0) and NUM_WORKERS (6) are overridable. One
# full-resolution job fits on the box: do not start another training meanwhile.
SELF=$(readlink -f "$0")
cd "$(dirname "$SELF")/.." || exit 1
PY=${PY:-/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=${GPU:-0}
mkdir -p logs weights/rgb_A
if [ -z "${RGB_QUEUE_CHILD:-}" ]; then
  RGB_QUEUE_CHILD=1 setsid nohup bash "$SELF" "$@" > logs/rgb_queue.log 2>&1 < /dev/null &
  echo "rgb baseline launched, pid $!, queue log logs/rgb_queue.log, run log logs/rgb_A.log, comparison log logs/compare_rgb.log"
  exit 0
fi

finished() { [ -f "weights/$1/results_$1.txt" ] && grep -q '"final": true' "weights/$1/results_$1.txt"; }

if finished rgb_A; then
  echo "$(date) rgb_A already finished, skipping"
else
  echo "$(date) start rgb_A"
  "$PY" -u main_det.py --session A --rgb_images --name rgb_A --output-dir weights/rgb_A \
      --epochs 100 --batch_size 2 --accumulate 4 --num_workers "${NUM_WORKERS:-6}" > logs/rgb_A.log 2>&1 < /dev/null
  rc=$?
  echo "$(date) end rgb_A, exit $rc"
  if [ "$rc" -ne 0 ]; then echo "$(date) rgb_A failed, no comparison"; exit "$rc"; fi
fi

echo "$(date) comparison: rgb_A + raw133_A + sel10g_clean_A + sel24g_clean_A + pca11_A on model_59..99"
"$PY" -u main_det_compare.py --runs rgb_A raw133_A sel10g_clean_A sel24g_clean_A pca11_A --conditions clean \
    --out_dir results/det/compare_rgb --num_workers "${NUM_WORKERS:-6}" > logs/compare_rgb.log 2>&1 < /dev/null
echo "$(date) done, compare exit $?"
