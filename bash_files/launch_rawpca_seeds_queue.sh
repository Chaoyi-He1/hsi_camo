#!/usr/bin/env bash
# Why greedy-10 >= raw 133 (spec §12, 2026-10-18): the controls that separate "whitening" from "which 10 readings", and the
# training-seed spread of the two detectors the claim rests on. Six noise-free runs one after another (one full-resolution
# job fits on the box), then the fixed-checkpoint comparison of everything involved:
#   rawpca10_A          the 133 cube bands whitened to their top-10 principal directions (--raw-bands --pca-channels 10):
#                       the same dimensionality and conditioning as greedy-10, unsupervised PCA instead of selected EC readings
#   rawpca40_A          the bands whitened to 40 directions: whitening without the dimensionality loss
#   sel10g_clean_A_s1/2 greedy-10 (noise-free, sel10g_clean_A's flags) with seeds 1 and 2 (the original is seed 42)
#   raw133_A_s1/2       raw 133 bands (raw133_A's flags) with seeds 1 and 2
# Comparison: the three greedy-10 seeds, the three raw seeds, rawpca10_A, rawpca40_A and pca11_A on model_59..99
#   -> results/det/compare_rawpca_seeds
#   bash bash_files/launch_rawpca_seeds_queue.sh    # detached; queue log logs/rawpca_seeds_queue.log, run logs logs/<run>.log
# A run with a 'final' line in weights/<run>/results_<run>.txt is skipped, so the queue can be relaunched after a crash
# (an interrupted run restarts from epoch 0). RUNS="rawpca10_A rawpca40_A" restricts it. CACHE_DIR defaults to the NVMe
# mirror of the fp16 frame cache (bash_files/copy_cache_to_nvme.sh) when it exists, else the cache next to the data;
# NUM_WORKERS (6), PY and CUDA_VISIBLE_DEVICES (0) are overridable. Do not start another full-resolution training meanwhile.
SELF=$(readlink -f "$0")
cd "$(dirname "$SELF")/.." || exit 1
PY=${PY:-/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
NVME=/home/staff/c/chaoyi_he/hsi_camo_cache/cache_fp16
if [ -z "${CACHE_DIR+x}" ]; then
  if [ -d "$NVME/train" ] && [ -d "$NVME/test" ]; then CACHE_DIR=$NVME; else CACHE_DIR=""; fi
fi
mkdir -p logs
if [ -z "${RAWPCA_QUEUE_CHILD:-}" ]; then
  RAWPCA_QUEUE_CHILD=1 setsid nohup bash "$SELF" "$@" >> logs/rawpca_seeds_queue.log 2>&1 < /dev/null &
  echo "rawpca/seeds queue launched, pid $!, log logs/rawpca_seeds_queue.log, cache ${CACHE_DIR:-<data_path>/cache_fp16}"
  exit 0
fi

# the greedy order of main_select_voltages.py (spec §12), as sel10g_clean_A was trained
GREEDY10="1.75 -0.44 1.36 0.48 -0.65 -0.36 1.70 0.01 -0.86 1.45"
RUNS=${RUNS:-rawpca10_A rawpca40_A sel10g_clean_A_s1 raw133_A_s1 sel10g_clean_A_s2 raw133_A_s2}
NUM_WORKERS=${NUM_WORKERS:-6}
CACHE_FLAG=()
[ -n "$CACHE_DIR" ] && CACHE_FLAG=(--cache-dir "$CACHE_DIR")
echo "$(date) rawpca/seeds queue on GPU $CUDA_VISIBLE_DEVICES, pid $$: $RUNS, cache ${CACHE_DIR:-<data_path>/cache_fp16}"

finished() { [ -f "weights/$1/results_$1.txt" ] && grep -q '"final": true' "weights/$1/results_$1.txt"; }

train() {   # train <name> <flags...>
  local name=$1 rc; shift
  if finished "$name"; then echo "$(date) $name already finished, skipping"; return; fi
  mkdir -p "weights/$name"
  echo "$(date) start $name: $*"
  "$PY" -u main_det.py --session A --read_noise_db 0 --name "$name" --output-dir "weights/$name" \
      --epochs 100 --batch_size 2 --accumulate 4 --num_workers "$NUM_WORKERS" "${CACHE_FLAG[@]}" "$@" > "logs/$name.log" 2>&1 < /dev/null
  rc=$?
  echo "$(date) end $name, exit $rc"
}

for run in $RUNS; do
  case $run in
    rawpca10_A)        train rawpca10_A --raw-bands --pca-channels 10 ;;
    rawpca40_A)        train rawpca40_A --raw-bands --pca-channels 40 ;;
    sel10g_clean_A_s1) train sel10g_clean_A_s1 --filter-select manual --filter-voltages $GREEDY10 --pca-channels 10 --seed 1 ;;
    sel10g_clean_A_s2) train sel10g_clean_A_s2 --filter-select manual --filter-voltages $GREEDY10 --pca-channels 10 --seed 2 ;;
    raw133_A_s1)       train raw133_A_s1 --raw-bands --seed 1 ;;
    raw133_A_s2)       train raw133_A_s2 --raw-bands --seed 2 ;;
    *) echo "unknown run $run"; exit 1 ;;
  esac
done
echo "$(date) comparison: greedy-10 x3 seeds, raw 133 x3 seeds, rawpca10_A, rawpca40_A, pca11_A on model_59..99"
"$PY" -u main_det_compare.py --runs sel10g_clean_A sel10g_clean_A_s1 sel10g_clean_A_s2 raw133_A raw133_A_s1 raw133_A_s2 rawpca10_A rawpca40_A pca11_A \
    --conditions clean --out_dir results/det/compare_rawpca_seeds --num_workers "$NUM_WORKERS" "${CACHE_FLAG[@]}" > logs/compare_rawpca_seeds.log 2>&1
rc=$?
echo "$(date) queue done, compare exit $rc"
