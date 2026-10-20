#!/usr/bin/env bash
# Plan item 3 (2026-10-20): better Stage-1 recall without new data. Four noise-free detector runs with the new training switches,
# one after another on one GPU (one full-resolution job per box), then the fixed-checkpoint comparison against the existing
# seeds, the ROI exports of the new detectors and the Stage-2 end-to-end evaluation of the ec10 segmenters with those boxes:
#   sel10g_aug_A      greedy-10 + scale / crop jitter (0.67-1.5x) + illumination / spectral-tilt jitter + weight EMA
#   raw133_aug_A      the raw 133 bands with the same switches (does the greedy-10 vs raw gap survive augmentation?)
#   sel10g_aug_m_A    sel10g_aug_A with the larger yolo26m backbone
#   sel10g_aug_tv_A   sel10g_aug_A trained on train + val (279 frames; model_best is not held out, so it is exported from model_99)
# Comparison -> results/det/compare_aug; exports -> results/det/rois_<run>_test.json and, at the conf 0.001 / top-15 operating
# point of the box study, results/det/rois_op/<run>_c0.001_k15/; Stage 2 -> results/seg_aug/<run>[_c0.001_k15]/compare.json.
#   bash bash_files/launch_aug_queue.sh     # detached; queue log logs/aug_queue.log, run logs logs/<run>.log
# Finished runs (a 'final' line in weights/<run>/results_<run>.txt) and existing compare.json files are skipped on relaunch;
# RUNS="..." restricts the list. CACHE_DIR defaults to the NVMe mirror of the frame cache (bash_files/copy_cache_to_nvme.sh)
# when it exists -- pair it with  QUEUE_LOG=logs/aug_queue.log bash bash_files/cleanup_nvme_cache.sh, which deletes the mirror
# after the 'queue done' line. NUM_WORKERS (6), PY, CUDA_VISIBLE_DEVICES (0) and SEG_RUNS are overridable. Do not start another
# full-resolution training meanwhile.
SELF=$(readlink -f "$0")
cd "$(dirname "$SELF")/.." || exit 1
PY=${PY:-/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} CUDA_DEVICE_ORDER=PCI_BUS_ID
NVME=/home/staff/c/chaoyi_he/hsi_camo_cache/cache_fp16
if [ -z "${CACHE_DIR+x}" ]; then
  if [ -d "$NVME/train" ] && [ -d "$NVME/test" ]; then CACHE_DIR=$NVME; else CACHE_DIR=""; fi
fi
mkdir -p logs
if [ -z "${AUG_QUEUE_CHILD:-}" ]; then
  AUG_QUEUE_CHILD=1 setsid nohup bash "$SELF" "$@" >> logs/aug_queue.log 2>&1 < /dev/null &
  echo "aug queue launched, pid $!, log logs/aug_queue.log, cache ${CACHE_DIR:-<data_path>/cache_fp16}"
  exit 0
fi

# the greedy order of main_select_voltages.py (spec §12), as sel10g_clean_A was trained
GREEDY10="1.75 -0.44 1.36 0.48 -0.65 -0.36 1.70 0.01 -0.86 1.45"
GREEDY_FLAGS="--filter-select manual --filter-voltages $GREEDY10 --pca-channels 10"
AUG="--aug_scale 0.67 1.5 --aug_gain 0.2 --aug_tilt 0.1 --ema"
RUNS=${RUNS:-sel10g_aug_A raw133_aug_A sel10g_aug_m_A sel10g_aug_tv_A}
SEG_RUNS=${SEG_RUNS:-"seg_sam2unet_ec10_s0 seg_sam2box_ec10_s0 seg_zoomnext_ec10_s0"}
NUM_WORKERS=${NUM_WORKERS:-6}
CACHE_FLAG=()
[ -n "$CACHE_DIR" ] && CACHE_FLAG=(--cache-dir "$CACHE_DIR")
echo "$(date) aug queue on GPU $CUDA_VISIBLE_DEVICES, pid $$: $RUNS, cache ${CACHE_DIR:-<data_path>/cache_fp16}"

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
    sel10g_aug_A)    train sel10g_aug_A $GREEDY_FLAGS $AUG ;;
    raw133_aug_A)    train raw133_aug_A --raw-bands $AUG ;;
    sel10g_aug_m_A)  train sel10g_aug_m_A $GREEDY_FLAGS $AUG --yolo-variant yolo26m ;;
    sel10g_aug_tv_A) train sel10g_aug_tv_A $GREEDY_FLAGS $AUG --train_on_val ;;
    *) echo "unknown run $run"; exit 1 ;;
  esac
done

echo "$(date) comparison: greedy-10 x3 seeds + aug / m / tv, raw 133 x3 seeds + aug on model_59..99 -> results/det/compare_aug"
"$PY" -u main_det_compare.py --runs sel10g_clean_A sel10g_clean_A_s1 sel10g_clean_A_s2 sel10g_aug_A sel10g_aug_m_A sel10g_aug_tv_A \
    raw133_A raw133_A_s1 raw133_A_s2 raw133_aug_A \
    --conditions clean --out_dir results/det/compare_aug --num_workers "$NUM_WORKERS" "${CACHE_FLAG[@]}" > logs/compare_aug.log 2>&1
echo "$(date) compare exit $?"

export_rois() {   # export_rois <det run> <checkpoint> <out dir> [main_det_rois flags...]
  local run=$1 ck=$2 out=$3; shift 3
  mkdir -p "$out"
  echo "$(date) export $run ($ck) -> $out $*"
  "$PY" -u main_det_rois.py --resume "weights/$run/$ck" --split test --out-dir "$out" --num_workers "$NUM_WORKERS" "${CACHE_FLAG[@]}" "$@" >> logs/aug_exports.log 2>&1 < /dev/null
  echo "$(date) export exit $?: $(tail -n 1 logs/aug_exports.log)"
}
evaluate() {   # evaluate <out dir> [main_seg_eval flags...]
  local out=$1; shift
  if [ -f "$out/compare.json" ]; then echo "$(date) $out already evaluated, skipping"; return; fi
  mkdir -p "$(dirname "$out")"
  echo "$(date) evaluate -> $out $*"
  "$PY" -u main_seg_eval.py --runs $SEG_RUNS --out_dir "$out" --num_workers "$NUM_WORKERS" "$@" > "$out.log" 2>&1 < /dev/null
  echo "$(date) evaluate exit $?: $(grep -m1 'wrote' "$out.log")"
}
for run in $RUNS; do
  ck=model_best; [ "$run" = sel10g_aug_tv_A ] && ck=model_99
  [ -f "weights/$run/$ck" ] || { echo "$(date) $run: no $ck, skipping its exports"; continue; }
  export_rois "$run" "$ck" results/det
  export_rois "$run" "$ck" "results/det/rois_op/${run}_c0.001_k15" --roi-conf 0.001 --max-rois 15
done
for run in $RUNS; do
  [ -f "results/det/rois_${run}_test.json" ] || continue
  evaluate "results/seg_aug/$run" --roi_run "$run"
  evaluate "results/seg_aug/${run}_c0.001_k15" --roi_run "$run" --roi_dir "results/det/rois_op/${run}_c0.001_k15"
done
echo "$(date) queue done"
