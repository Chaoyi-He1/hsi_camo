#!/usr/bin/env bash
# Training seeds of the final detector recipe (2026-10-21): greedy-10 + scale / photometric jitter + weight EMA with seeds 1 and 2
# (sel10g_aug_A is seed 42), one after another on one GPU, then the fixed-checkpoint comparison of the three aug seeds against the
# three plain greedy-10 seeds, the three raw seeds and the other recipes -> results/det/compare_aug_seeds, then the model_99 export
# of every aug seed at the default operating point (results/det/rois_op/<run>_m99/, the last checkpoint is the export rule of an
# EMA run) and its end-to-end scoring by the seed-0 ec10 segmenters -> results/seg_aug/<run>_m99/compare.json.
#   bash bash_files/launch_aug_seeds_queue.sh    # detached; log logs/aug_seeds_queue.log; prints "queue done" at the end
# CACHE_DIR defaults to the NVMe mirror of the frame cache when it exists (pair it with
# QUEUE_LOG=logs/aug_seeds_queue.log bash bash_files/cleanup_nvme_cache.sh). Finished runs and existing compare.json files are
# skipped on relaunch. RUNS, SEG_RUNS, NUM_WORKERS (6), PY, CUDA_VISIBLE_DEVICES (0) are overridable. One full-resolution job per box.
SELF=$(readlink -f "$0")
cd "$(dirname "$SELF")/.." || exit 1
PY=${PY:-/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} CUDA_DEVICE_ORDER=PCI_BUS_ID
NVME=/home/staff/c/chaoyi_he/hsi_camo_cache/cache_fp16
if [ -z "${CACHE_DIR+x}" ]; then
  if [ -d "$NVME/train" ] && [ -d "$NVME/test" ]; then CACHE_DIR=$NVME; else CACHE_DIR=""; fi
fi
mkdir -p logs
if [ -z "${AUG_SEEDS_CHILD:-}" ]; then
  AUG_SEEDS_CHILD=1 setsid nohup bash "$SELF" "$@" >> logs/aug_seeds_queue.log 2>&1 < /dev/null &
  echo "aug seeds queue launched, pid $!, log logs/aug_seeds_queue.log, cache ${CACHE_DIR:-<data_path>/cache_fp16}"
  exit 0
fi

GREEDY10="1.75 -0.44 1.36 0.48 -0.65 -0.36 1.70 0.01 -0.86 1.45"
GREEDY_FLAGS="--filter-select manual --filter-voltages $GREEDY10 --pca-channels 10"
AUG="--aug_scale 0.67 1.5 --aug_gain 0.2 --aug_tilt 0.1 --ema"
RUNS=${RUNS:-sel10g_aug_A_s1 sel10g_aug_A_s2}
SEG_RUNS=${SEG_RUNS:-"seg_sam2unet_ec10_s0 seg_sam2box_ec10_s0 seg_zoomnext_ec10_s0"}
NUM_WORKERS=${NUM_WORKERS:-6}
CACHE_FLAG=()
[ -n "$CACHE_DIR" ] && CACHE_FLAG=(--cache-dir "$CACHE_DIR")
echo "$(date) aug seeds queue on GPU $CUDA_VISIBLE_DEVICES, pid $$: $RUNS, cache ${CACHE_DIR:-<data_path>/cache_fp16}"

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
    sel10g_aug_A_s1) train sel10g_aug_A_s1 $GREEDY_FLAGS $AUG --seed 1 ;;
    sel10g_aug_A_s2) train sel10g_aug_A_s2 $GREEDY_FLAGS $AUG --seed 2 ;;
    *) echo "unknown run $run"; exit 1 ;;
  esac
done

echo "$(date) comparison: aug x3 seeds, greedy-10 x3 seeds, raw x3 seeds, raw + aug, yolo26m, train + val on model_59..99 -> results/det/compare_aug_seeds"
"$PY" -u main_det_compare.py --runs sel10g_aug_A sel10g_aug_A_s1 sel10g_aug_A_s2 sel10g_clean_A sel10g_clean_A_s1 sel10g_clean_A_s2 \
    raw133_A raw133_A_s1 raw133_A_s2 raw133_aug_A sel10g_aug_m_A sel10g_aug_tv_A \
    --conditions clean --out_dir results/det/compare_aug_seeds --num_workers "$NUM_WORKERS" "${CACHE_FLAG[@]}" > logs/compare_aug_seeds.log 2>&1
echo "$(date) compare exit $?"

for run in sel10g_aug_A sel10g_aug_A_s1 sel10g_aug_A_s2; do
  out=results/det/rois_op/${run}_m99
  if [ ! -f "$out/rois_${run}_test.json" ]; then
    [ -f "weights/$run/model_99" ] || { echo "$(date) $run: no model_99, skipping"; continue; }
    mkdir -p "$out"
    echo "$(date) export $run model_99 -> $out"
    "$PY" -u main_det_rois.py --resume "weights/$run/model_99" --split test --out-dir "$out" --num_workers "$NUM_WORKERS" "${CACHE_FLAG[@]}" >> logs/aug_seeds_exports.log 2>&1 < /dev/null
    echo "$(date) export exit $?: $(tail -n 1 logs/aug_seeds_exports.log)"
  fi
  seg=results/seg_aug/${run}_m99
  if [ -f "$seg/compare.json" ]; then echo "$(date) $seg already evaluated, skipping"; continue; fi
  echo "$(date) evaluate -> $seg"
  "$PY" -u main_seg_eval.py --runs $SEG_RUNS --out_dir "$seg" --roi_run "$run" --roi_dir "$out" --num_workers "$NUM_WORKERS" > "$seg.log" 2>&1 < /dev/null
  echo "$(date) evaluate exit $?: $(grep -m1 'wrote' "$seg.log")"
done
echo "$(date) queue done"
