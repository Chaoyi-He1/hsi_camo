#!/usr/bin/env bash
# Stage-2 queue of the FINAL pipeline (2026-10-21): the three segmenters x seeds 0 1 2 trained on the boxes of the final Stage-1
# detector sel10g_aug_A_m99 (bash_files/prep_final_detector.sh: its ROI exports and crop cache), run names seg_<model>_ec10f_s<seed>
# (arm 'ec10', the detector's 10 whitened EC readings), then the end-to-end test evaluation of the nine runs with that detector's
# boxes -> results/seg_final/compare.json (+ model_last -> results/seg_final_last). Same mechanics as launch_seg_queue.sh: one
# queue per GPU (lock), a run locked while it trains, finished runs skipped, an unfinished run resumed from model_last, so two
# GPUs can share the plan:
#   bash bash_files/launch_seg_final_queue.sh                                  # GPU 0 (evaluates at the end)
#   CUDA_VISIBLE_DEVICES=1 EVAL=0 bash bash_files/launch_seg_final_queue.sh    # GPU 1, same plan, no evaluation
# Env overrides: MODELS, SEEDS, DET (sel10g_aug_A_m99), DATA_PATH, CROP_CACHE (<DATA_PATH>/crop_cache_seg_final), NUM_WORKERS (8),
# PY, CUDA_VISIBLE_DEVICES (0), EVAL (1), EVAL_LAST (1), FORCE_GPU=1 (skip the busy-GPU check), LIST_ONLY=1.
SELF=$(readlink -f "$0")
cd "$(dirname "$SELF")/.." || exit 1
PY=${PY:-/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
DATA_PATH=${DATA_PATH:-/data2/chaoyi/HyperCOD/Raw data}
CROP_CACHE=${CROP_CACHE:-$DATA_PATH/crop_cache_seg_final}
DET=${DET:-sel10g_aug_A_m99}
MODELS=${MODELS:-sam2unet sam2box zoomnext}
SEEDS=${SEEDS:-0 1 2}
EVAL=${EVAL-1}
EVAL_LAST=${EVAL_LAST-1}
NUM_WORKERS=${NUM_WORKERS:-8}
GPU_LOCK=logs/seg_final_queue_gpu${CUDA_VISIBLE_DEVICES}.lock
TRAIN_LOCK=logs/seg_final_train_gpu${CUDA_VISIBLE_DEVICES}.lock
QUEUE_LOG=logs/seg_final_queue_gpu${CUDA_VISIBLE_DEVICES}.log

finished() { [ -f "weights/$1/results_$1.txt" ] && grep -q '"final": true' "weights/$1/results_$1.txt"; }
on() { [ -n "$1" ] && [ "$1" != 0 ]; }
gpu_busy() {
  local out
  on "${FORCE_GPU:-}" && return 1
  command -v nvidia-smi > /dev/null 2>&1 || return 1
  if ! out=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$CUDA_VISIBLE_DEVICES" 2>&1); then
    echo "nvidia-smi -i $CUDA_VISIBLE_DEVICES failed: $out"; return 0
  fi
  [ -n "$out" ] || return 1
  echo "GPU $CUDA_VISIBLE_DEVICES is busy (compute pid(s) $(echo $out))"
}

PLAN=()
for s in $SEEDS; do for m in $MODELS; do PLAN+=("seg_${m}_ec10f_s${s} $m $s"); done; done
if [ -n "${LIST_ONLY:-}" ]; then
  for p in "${PLAN[@]}"; do read -r name _ <<< "$p"; if finished "$name"; then echo "$name done"; else echo "$name todo"; fi; done
  exit 0
fi

mkdir -p logs
if [ -z "${SEG_QUEUE_CHILD:-}" ]; then
  ( flock -n 9 ) 9> "$GPU_LOCK" || { echo "another final seg queue holds GPU $CUDA_VISIBLE_DEVICES ($GPU_LOCK): not launched"; exit 1; }
  if why=$(gpu_busy); then echo "$why: not launched (FORCE_GPU=1 overrides)"; exit 1; fi
  SEG_QUEUE_CHILD=1 setsid nohup bash "$SELF" "$@" >> "$QUEUE_LOG" 2>&1 < /dev/null &
  echo "final seg queue launched on GPU $CUDA_VISIBLE_DEVICES, pid $!, log $QUEUE_LOG"
  exit 0
fi

exec 9> "$GPU_LOCK"
flock -n 9 || { echo "$(date) another final seg queue holds GPU ${CUDA_VISIBLE_DEVICES}"; exit 1; }
if why=$(gpu_busy); then echo "$(date) $why: queue not started (FORCE_GPU=1 overrides)"; exit 1; fi
exec 6> "$TRAIN_LOCK"
flock -n 6 || { echo "$(date) $TRAIN_LOCK is held by another process"; exit 1; }
echo "$(date) final seg queue on GPU $CUDA_VISIBLE_DEVICES, pid $$: ${#PLAN[@]} planned runs, detector $DET, crop cache $CROP_CACHE, NUM_WORKERS $NUM_WORKERS, EVAL ${EVAL:-0}"

missing=0
[ -f "weights/$DET/model_best" ] || { echo "$(date) missing weights/$DET/model_best"; missing=1; }
for split in train val test; do
  [ -f "results/det/rois_${DET}_${split}.json" ] || { echo "$(date) missing results/det/rois_${DET}_${split}.json: run bash_files/prep_final_detector.sh"; missing=1; }
done
[ -f "$CROP_CACHE/index.json" ] || { echo "$(date) missing $CROP_CACHE/index.json: run bash_files/prep_final_detector.sh"; missing=1; }
[ "$missing" = 0 ] || exit 1

train() {   # train <name> <model> <seed>
  local name=$1 model=$2 seed=$3 rc epoch resume=()
  if finished "$name"; then echo "$(date) $name already finished, skipping"; return; fi
  mkdir -p "weights/$name"
  exec 7> "weights/$name/.queue.lock"
  if ! flock -n 7; then echo "$(date) $name is being trained by another seg queue, skipping"; exec 7>&-; return; fi
  if finished "$name"; then echo "$(date) $name finished by another seg queue, skipping"; exec 7>&-; return; fi
  if [ -f "weights/$name/model_last" ]; then
    epoch=$("$PY" -c 'import sys, torch; print(torch.load(sys.argv[1], map_location="cpu", weights_only=False, mmap=True)["epoch"])' \
            "weights/$name/model_last" 2> /dev/null) || epoch='?'
    resume=(--resume "weights/$name/model_last")
    echo "$(date) $name: resume from model_last epoch $epoch ($model, arm ec10, detector $DET, seed $seed)" | tee -a "logs/$name.log"
  else
    echo "$(date) $name: start fresh ($model, arm ec10, detector $DET, seed $seed)" | tee -a "logs/$name.log"
  fi
  "$PY" -u main_seg.py --seg_model "$model" --arm ec10 --det_ckpt "weights/$DET/model_best" --data_path "$DATA_PATH" \
      --crop_cache "$CROP_CACHE" --seed "$seed" --num_workers "$NUM_WORKERS" \
      --name "$name" --output_dir "weights/$name" "${resume[@]}" >> "logs/$name.log" 2>&1 < /dev/null
  rc=$?
  exec 7>&-
  echo "$(date) end $name, exit $rc"
}

for p in "${PLAN[@]}"; do
  read -r name model seed <<< "$p"
  train "$name" "$model" "$seed"
done
exec 6>&-

if ! on "$EVAL"; then echo "$(date) EVAL=${EVAL:-}: no evaluation"; echo "$(date) queue done"; exit 0; fi
for f in logs/seg_final_train_gpu*.lock; do
  [ -e "$f" ] || continue
  flock -n "$f" true || { echo "$(date) waiting for the final seg queue training under $f"; flock "$f" true; }
done
DONE=()
for p in "${PLAN[@]}"; do read -r name _ <<< "$p"; finished "$name" && DONE+=("$name"); done
echo "$(date) evaluation of ${#DONE[@]}/${#PLAN[@]} finished runs (log logs/seg_final_eval.log)"
"$PY" -u main_seg_eval.py --runs "${DONE[@]}" --data_path "$DATA_PATH" --out_dir results/seg_final --num_workers "$NUM_WORKERS" > logs/seg_final_eval.log 2>&1
echo "$(date) evaluation exit $? -> results/seg_final/compare.json"
if on "$EVAL_LAST"; then
  "$PY" -u main_seg_eval.py --runs "${DONE[@]}" --ckpt model_last --data_path "$DATA_PATH" --out_dir results/seg_final_last --num_workers "$NUM_WORKERS" > logs/seg_final_eval_last.log 2>&1
  echo "$(date) model_last evaluation exit $? -> results/seg_final_last/compare.json"
fi
echo "$(date) queue done"
