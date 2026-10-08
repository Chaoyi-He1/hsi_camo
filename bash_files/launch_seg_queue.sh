#!/usr/bin/env bash
# Stage-2 queue (spec §7, §8): the crop cache once, then every (model, arm, seed) segmentation run in sequence, then the
# end-to-end test evaluation of everything (main_seg_eval.py: oracle ROIs, paste-back, each arm's own detector ROIs).
#   models sam2unet sam2box zoomnext x arms raw ec10 ec24 x seeds 0 1 2 = 27 runs seg_<model>_<arm>_s<seed>, seed-major (the
#   9 (model, arm) pairs of seed 0 first, so a complete first comparison exists after 9 runs); then the control
#   seg_sam2unet_rgb_s<seed> (pseudo-RGB input, raw133_A's boxes; CONTROL_SEEDS, default 0); the zero-shot SAM2.1 box
#   control needs no training and is evaluated by main_seg_eval --zero_shot.
#   bash bash_files/launch_seg_queue.sh    # detached; queue log logs/seg_queue_gpu<N>.log (N = CUDA_VISIBLE_DEVICES), run
#                                          # logs logs/<run>.log (appended), crop cache logs/seg_crop_cache.log, evaluation
#                                          # logs/seg_eval.log (model_last: logs/seg_eval_last.log)
# Prerequisites: bash_files/setup_third_party.sh (sam2, pysodmetrics, ZoomNeXt, weights/pretrained/*) and
# bash_files/launch_rois_all.sh (results/det/rois_<det>_{train,val,test}.json of raw133_A, sel10g_clean_A, sel24g_clean_A).
# The crop cache is built only when <CROP_CACHE>/index.json is missing. Relaunch after a crash: a run with a 'final' line
# in weights/<run>/results_<run>.txt is skipped; an unfinished run with weights/<run>/model_last resumes from it
# (main_seg --resume: optimizer, schedule, epoch and best score carried over, results lines past the checkpoint dropped);
# one without model_last starts fresh (main_seg moves a previous attempt's results / TensorBoard dir to .prev). The queue
# log says which ("resume from model_last epoch E" / "start fresh"). Batch size and accumulation of every model come
# from cfg/seg.yaml (models.*), not from this script.
# One queue per GPU: the queue holds logs/seg_queue_gpu<N>.lock while it runs, so a second queue on that GPU refuses to
# start, and it refuses a GPU on which nvidia-smi lists any compute process (e.g. a Stage-1 training; FORCE_GPU=1
# skips this check; without nvidia-smi it is skipped). Every run is locked while it trains (weights/<run>/.queue.lock),
# so two queues never train the same run. Two GPUs can therefore share one plan, each taking the runs the other has
# not started, with one of them evaluating:
#   bash bash_files/launch_seg_queue.sh                                  # GPU 0, evaluates at the end
#   CUDA_VISIBLE_DEVICES=1 EVAL=0 bash bash_files/launch_seg_queue.sh    # GPU 1, same plan, no evaluation
# An evaluating queue first waits until no other queue is still training (logs/seg_train_gpu*.lock), then evaluates the
# finished runs of its own plan (give the queues the same plan, or the evaluation misses the other queue's runs).
# Env overrides: MODELS, ARMS, SEEDS, CONTROL_SEEDS (empty = no control), DATA_PATH, CROP_CACHE (default
# <DATA_PATH>/crop_cache_seg), NUM_WORKERS (8), PY, CUDA_VISIBLE_DEVICES (0; one GPU), EVAL=1 (default; EVAL=0 or empty
# skips both evaluations), EVAL_LAST=1 (default; EVAL_LAST=0 or empty disables) also evaluates model_last into
# results/seg_last, FORCE_GPU=1, CACHE_ONLY=1 (build the crop cache, then stop), LIST_ONLY=1 (print the planned runs,
# todo/done, and exit without detaching, locking or asking nvidia-smi). One full-resolution (Stage-1) job per box: do
# not start one while this queue runs.
SELF=$(readlink -f "$0")
cd "$(dirname "$SELF")/.." || exit 1
PY=${PY:-/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CUDA_DEVICE_ORDER=PCI_BUS_ID      # CUDA numbers the GPUs as nvidia-smi does (the busy check asks nvidia-smi -i N)
DATA_PATH=${DATA_PATH:-/data2/chaoyi/HyperCOD/Raw data}
CROP_CACHE=${CROP_CACHE:-$DATA_PATH/crop_cache_seg}
MODELS=${MODELS:-sam2unet sam2box zoomnext}
ARMS=${ARMS:-raw ec10 ec24}
SEEDS=${SEEDS:-0 1 2}
CONTROL_SEEDS=${CONTROL_SEEDS-0}
EVAL=${EVAL-1}
EVAL_LAST=${EVAL_LAST-1}
NUM_WORKERS=${NUM_WORKERS:-8}
GPU_LOCK=logs/seg_queue_gpu${CUDA_VISIBLE_DEVICES}.lock       # held by the queue for its whole life
TRAIN_LOCK=logs/seg_train_gpu${CUDA_VISIBLE_DEVICES}.lock     # held while it trains (an evaluating queue waits for all)
QUEUE_LOG=logs/seg_queue_gpu${CUDA_VISIBLE_DEVICES}.log

# the arm's detector (spec §2): its front end, its training/val boxes in the crop cache and its test ROIs
det_of() { case $1 in raw|rgb) echo raw133_A ;; ec10) echo sel10g_clean_A ;; ec24) echo sel24g_clean_A ;; *) echo "unknown arm $1" >&2; return 1 ;; esac; }
finished() { [ -f "weights/$1/results_$1.txt" ] && grep -q '"final": true' "weights/$1/results_$1.txt"; }
on() { [ -n "$1" ] && [ "$1" != 0 ]; }                       # a switch: set, and not 0
# why the queue's GPU cannot be used (a compute process on it, or nvidia-smi failing); fails (silently) when it can
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
  # refuse on the terminal, before a second queue would write into a running queue's log; the detached queue checks again
  # for real. The log is appended to, so relaunches (extra seeds, a later evaluation) keep the earlier campaign's timeline.
  ( flock -n 9 ) 9> "$GPU_LOCK" || { echo "another seg queue holds GPU $CUDA_VISIBLE_DEVICES ($GPU_LOCK, log $QUEUE_LOG): not launched"; exit 1; }
  if why=$(gpu_busy); then echo "$why: not launched (FORCE_GPU=1 overrides)"; exit 1; fi
  SEG_QUEUE_CHILD=1 setsid nohup bash "$SELF" "$@" >> "$QUEUE_LOG" 2>&1 < /dev/null &
  echo "seg queue launched on GPU $CUDA_VISIBLE_DEVICES, pid $!, log $QUEUE_LOG"
  exit 0
fi

# the detached queue: one per GPU (the lock is held until it exits), and only on a GPU nobody else computes on. The lock
# descriptors (9 GPU, 6 training, 7 run) are inherited by the python jobs: a run orphaned by a killed queue keeps them.
exec 9> "$GPU_LOCK"
flock -n 9 || { echo "$(date) another seg queue holds GPU ${CUDA_VISIBLE_DEVICES}"; exit 1; }
if why=$(gpu_busy); then echo "$(date) $why: queue not started (FORCE_GPU=1 overrides)"; exit 1; fi
exec 6> "$TRAIN_LOCK"
flock -n 6 || { echo "$(date) $TRAIN_LOCK is held by another process"; exit 1; }
echo "$(date) seg queue on GPU $CUDA_VISIBLE_DEVICES, pid $$: ${#PLAN[@]} planned runs, NUM_WORKERS $NUM_WORKERS, EVAL ${EVAL:-0}"

# prerequisites: the three detectors and their ROI exports for every split
missing=0
for det in raw133_A sel10g_clean_A sel24g_clean_A; do
  [ -f "weights/$det/model_best" ] || { echo "$(date) missing weights/$det/model_best"; missing=1; }
  for split in train val test; do
    [ -f "results/det/rois_${det}_${split}.json" ] || { echo "$(date) missing results/det/rois_${det}_${split}.json: run bash_files/launch_rois_all.sh"; missing=1; }
  done
done
[ "$missing" = 0 ] || exit 1

# the crop cache, once: every train/val window of every arm's boxes (data_loader.roi_crops.build_crop_cache); a second
# queue starting meanwhile waits for it instead of building it twice
exec 8> logs/seg_crop_cache.lock
flock 8
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
exec 8>&-
if [ -n "${CACHE_ONLY:-}" ]; then echo "$(date) CACHE_ONLY: done"; exit 0; fi

train() {   # train <name> <model> <arm> <seed>; batch size / accumulation: cfg/seg.yaml models.<model>
  local name=$1 model=$2 arm=$3 seed=$4 det rc epoch resume=()
  if finished "$name"; then echo "$(date) $name already finished, skipping"; return; fi
  det=$(det_of "$arm") || return
  mkdir -p "weights/$name"
  # the run's lock: a run another queue is training is left to it (checked again for 'final' once the lock is ours)
  exec 7> "weights/$name/.queue.lock"
  if ! flock -n 7; then echo "$(date) $name is being trained by another seg queue, skipping"; exec 7>&-; return; fi
  if finished "$name"; then echo "$(date) $name finished by another seg queue, skipping"; exec 7>&-; return; fi
  # an interrupted run resumes from model_last; the run log is appended to either way
  if [ -f "weights/$name/model_last" ]; then
    epoch=$("$PY" -c 'import sys, torch; print(torch.load(sys.argv[1], map_location="cpu", weights_only=False, mmap=True)["epoch"])' \
            "weights/$name/model_last" 2> /dev/null) || epoch='?'
    resume=(--resume "weights/$name/model_last")
    echo "$(date) $name: resume from model_last epoch $epoch ($model, arm $arm, detector $det, seed $seed)" | tee -a "logs/$name.log"
  else
    echo "$(date) $name: start fresh ($model, arm $arm, detector $det, seed $seed)" | tee -a "logs/$name.log"
  fi
  "$PY" -u main_seg.py --seg_model "$model" --arm "$arm" --det_ckpt "weights/$det/model_best" --data_path "$DATA_PATH" \
      --crop_cache "$CROP_CACHE" --seed "$seed" --num_workers "$NUM_WORKERS" \
      --name "$name" --output_dir "weights/$name" "${resume[@]}" >> "logs/$name.log" 2>&1 < /dev/null
  rc=$?
  exec 7>&-
  echo "$(date) end $name, exit $rc"
}

for p in "${PLAN[@]}"; do
  read -r name model arm seed <<< "$p"
  train "$name" "$model" "$arm" "$seed"
done
exec 6>&-                                                       # training done: an evaluating queue need not wait for us

if ! on "$EVAL"; then echo "$(date) EVAL=${EVAL:-}: no evaluation"; echo "$(date) queue done"; exit 0; fi
# wait until no other seg queue is training (their runs must be finished, or have failed, before the evaluation)
for f in logs/seg_train_gpu*.lock; do
  [ -e "$f" ] || continue
  flock -n "$f" true || { echo "$(date) waiting for the seg queue training under $f"; flock "$f" true; }
done
exec 5> logs/seg_eval.lock
flock 5                                                         # two evaluating queues: one after the other

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
if on "$EVAL_LAST"; then
  "$PY" -u main_seg_eval.py --runs "${DONE[@]}" --ckpt model_last --data_path "$DATA_PATH" --out_dir results/seg_last > logs/seg_eval_last.log 2>&1
  rc=$?
  echo "$(date) model_last evaluation exit $rc -> results/seg_last/compare.json"
fi
echo "$(date) queue done"
