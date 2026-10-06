#!/usr/bin/env bash
# Stage-2 queue (spec §7, §8): the crop cache once, then every (model, arm, seed) segmentation run in sequence, then the
# end-to-end test evaluation of everything (main_seg_eval.py: oracle ROIs, paste-back, each arm's own detector ROIs).
#   models sam2unet sam2box zoomnext x arms raw ec10 ec24 x seeds 0 1 2 = 27 runs seg_<model>_<arm>_s<seed>, seed-major (the
#   9 (model, arm) pairs of seed 0 first, so a complete first comparison exists after 9 runs); then the control
#   seg_sam2unet_rgb_s<seed> (pseudo-RGB input, raw133_A's boxes; CONTROL_SEEDS, default 0); the zero-shot SAM2.1 box
#   control needs no training and is evaluated by main_seg_eval --zero_shot.
#   bash bash_files/launch_seg_queue.sh    # detached; queue log logs/seg_queue.log, run logs logs/<run>.log,
#                                          # crop cache logs/seg_crop_cache.log, evaluation logs/seg_eval.log
# Prerequisites: bash_files/setup_third_party.sh (sam2, pysodmetrics, ZoomNeXt, weights/pretrained/*) and
# bash_files/launch_rois_all.sh (results/det/rois_<det>_{train,val,test}.json of raw133_A, sel10g_clean_A, sel24g_clean_A).
# The crop cache is built only when <CROP_CACHE>/index.json is missing. A run with a 'final' line in
# weights/<run>/results_<run>.txt is skipped, so the queue can be relaunched after a crash (an interrupted run restarts
# from epoch 0). Batch size and accumulation of every model come from cfg/seg.yaml (models.*), not from this script.
# Env overrides: MODELS, ARMS, SEEDS, CONTROL_SEEDS (empty = no control), DATA_PATH, CROP_CACHE (default
# <DATA_PATH>/crop_cache_seg), NUM_WORKERS (6), PY, CUDA_VISIBLE_DEVICES (0), EVAL_LAST=1 (default; EVAL_LAST= disables)
# also evaluates model_last into results/seg_last, CACHE_ONLY=1 (build the crop cache, then stop), LIST_ONLY=1 (print the
# planned runs, todo/done, and exit without detaching). One full-resolution job per box: do not start another training meanwhile.
SELF=$(readlink -f "$0")
cd "$(dirname "$SELF")/.." || exit 1
PY=${PY:-/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
DATA_PATH=${DATA_PATH:-/data2/chaoyi/HyperCOD/Raw data}
CROP_CACHE=${CROP_CACHE:-$DATA_PATH/crop_cache_seg}
MODELS=${MODELS:-sam2unet sam2box zoomnext}
ARMS=${ARMS:-raw ec10 ec24}
SEEDS=${SEEDS:-0 1 2}
CONTROL_SEEDS=${CONTROL_SEEDS-0}
EVAL_LAST=${EVAL_LAST-1}
NUM_WORKERS=${NUM_WORKERS:-6}

# the arm's detector (spec §2): its front end, its training/val boxes in the crop cache and its test ROIs
det_of() { case $1 in raw|rgb) echo raw133_A ;; ec10) echo sel10g_clean_A ;; ec24) echo sel24g_clean_A ;; *) echo "unknown arm $1" >&2; return 1 ;; esac; }
finished() { [ -f "weights/$1/results_$1.txt" ] && grep -q '"final": true' "weights/$1/results_$1.txt"; }

# the plan, one "name model arm seed" line per run, seed-major
PLAN=()
for s in $SEEDS; do for m in $MODELS; do for a in $ARMS; do PLAN+=("seg_${m}_${a}_s${s} $m $a $s"); done; done; done
for s in $CONTROL_SEEDS; do PLAN+=("seg_sam2unet_rgb_s${s} sam2unet rgb $s"); done

if [ -n "${LIST_ONLY:-}" ]; then
  for p in "${PLAN[@]}"; do
    read -r name _ <<< "$p"
    if finished "$name"; then echo "$name done"; else echo "$name todo"; fi
  done
  exit 0
fi

mkdir -p logs
if [ -z "${SEG_QUEUE_CHILD:-}" ]; then
  SEG_QUEUE_CHILD=1 setsid nohup bash "$SELF" "$@" > logs/seg_queue.log 2>&1 < /dev/null &
  echo "seg queue launched, pid $!, log logs/seg_queue.log"
  exit 0
fi

# prerequisites: the three detectors and their ROI exports for every split
missing=0
for det in raw133_A sel10g_clean_A sel24g_clean_A; do
  [ -f "weights/$det/model_best" ] || { echo "$(date) missing weights/$det/model_best"; missing=1; }
  for split in train val test; do
    [ -f "results/det/rois_${det}_${split}.json" ] || { echo "$(date) missing results/det/rois_${det}_${split}.json: run bash_files/launch_rois_all.sh"; missing=1; }
  done
done
[ "$missing" = 0 ] || exit 1

# the crop cache, once: every train/val window of every arm's boxes (data_loader.roi_crops.build_crop_cache)
if [ ! -f "$CROP_CACHE/index.json" ]; then
  echo "$(date) building the crop cache in $CROP_CACHE (log logs/seg_crop_cache.log)"
  "$PY" -u - "$DATA_PATH" "$CROP_CACHE" "$NUM_WORKERS" > logs/seg_crop_cache.log 2>&1 <<'EOF'
import sys
from data_loader.roi_crops import build_crop_cache
data_path, out_dir, num_workers = sys.argv[1], sys.argv[2], int(sys.argv[3])
dets = {'raw': 'raw133_A', 'ec10': 'sel10g_clean_A', 'ec24': 'sel24g_clean_A'}
roi_files = {arm: {split: f'results/det/rois_{det}_{split}.json' for split in ('train', 'val')} for arm, det in dets.items()}
info = build_crop_cache(data_path, out_dir, roi_files, splits=('train', 'val'), grow=2.0, min_side=512, num_workers=num_workers)
print(f"crop cache: {info['n_windows']} windows, {info['bytes'] / 1e9:.1f} GB -> {out_dir}")
EOF
  rc=$?
  if [ "$rc" != 0 ] || [ ! -f "$CROP_CACHE/index.json" ]; then echo "$(date) crop cache failed, exit $rc"; exit 1; fi
  echo "$(date) $(tail -n 1 logs/seg_crop_cache.log)"
fi
if [ -n "${CACHE_ONLY:-}" ]; then echo "$(date) CACHE_ONLY: done"; exit 0; fi

train() {   # train <name> <model> <arm> <seed>; batch size / accumulation: cfg/seg.yaml models.<model>
  local name=$1 model=$2 arm=$3 seed=$4 det rc
  if finished "$name"; then echo "$(date) $name already finished, skipping"; return; fi
  det=$(det_of "$arm") || return
  mkdir -p "weights/$name"
  echo "$(date) start $name: $model, arm $arm, detector $det, seed $seed"
  "$PY" -u main_seg.py --seg_model "$model" --arm "$arm" --det_ckpt "weights/$det/model_best" --data_path "$DATA_PATH" \
      --crop_cache "$CROP_CACHE" --seed "$seed" --num_workers "$NUM_WORKERS" \
      --name "$name" --output_dir "weights/$name" > "logs/$name.log" 2>&1 < /dev/null
  rc=$?
  echo "$(date) end $name, exit $rc"
}

for p in "${PLAN[@]}"; do
  read -r name model arm seed <<< "$p"
  train "$name" "$model" "$arm" "$seed"
done

# the end-to-end evaluation of every finished run + the zero-shot control (one frame pass per main_seg_eval --runs_per_pass runs)
DONE=()
for p in "${PLAN[@]}"; do
  read -r name _ <<< "$p"
  finished "$name" && DONE+=("$name")
done
echo "$(date) evaluation of ${#DONE[@]}/${#PLAN[@]} finished runs + the zero-shot control (log logs/seg_eval.log)"
"$PY" -u main_seg_eval.py --runs "${DONE[@]}" --zero_shot --data_path "$DATA_PATH" --out_dir results/seg > logs/seg_eval.log 2>&1
rc=$?
echo "$(date) evaluation exit $rc -> results/seg/compare.json"
if [ -n "${EVAL_LAST:-}" ]; then
  "$PY" -u main_seg_eval.py --runs "${DONE[@]}" --ckpt model_last --data_path "$DATA_PATH" --out_dir results/seg_last > logs/seg_eval_last.log 2>&1
  echo "$(date) model_last evaluation exit $? -> results/seg_last/compare.json"
fi
echo "$(date) queue done"
