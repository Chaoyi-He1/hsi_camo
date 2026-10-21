#!/usr/bin/env bash
# The final Stage-1 detector of the pipeline (2026-10-21): greedy-10 + scale / photometric jitter + weight EMA, its LAST checkpoint
# (weights/sel10g_aug_A/model_99; the validation-selected model_best is weaker, spec §12 "Recall without new data"), published as
# the run weights/sel10g_aug_A_m99/model_best (a symlink) so every tool names it sel10g_aug_A_m99. This script exports its ROIs for
# the three splits at the default operating point (results/det/rois_sel10g_aug_A_m99_{train,val,test}.json) and builds the Stage-2
# crop cache of its boxes (CROP_CACHE, default <DATA_PATH>/crop_cache_seg_final: object windows from the GT, false-positive
# windows from these boxes, arm 'ec10'), which bash_files/launch_seg_final_queue.sh trains on. ~25 min on one GPU.
#   bash bash_files/prep_final_detector.sh      # foreground; prints "prep done" at the end. DET, DATA_PATH, CROP_CACHE,
#                                               # NUM_WORKERS (6), PY, CUDA_VISIBLE_DEVICES (1) are overridable.
SELF=$(readlink -f "$0")
cd "$(dirname "$SELF")/.." || exit 1
PY=${PY:-/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1} CUDA_DEVICE_ORDER=PCI_BUS_ID
DET=${DET:-sel10g_aug_A_m99}
DATA_PATH=${DATA_PATH:-/data2/chaoyi/HyperCOD/Raw data}
CROP_CACHE=${CROP_CACHE:-$DATA_PATH/crop_cache_seg_final}
NUM_WORKERS=${NUM_WORKERS:-6}
[ -f "weights/$DET/model_best" ] || { echo "weights/$DET/model_best missing (ln -s ../sel10g_aug_A/model_99 weights/sel10g_aug_A_m99/model_best)"; exit 1; }
mkdir -p logs
for split in test val train; do
  if [ -f "results/det/rois_${DET}_${split}.json" ]; then echo "$(date) rois_${DET}_${split}.json exists, skipping"; continue; fi
  echo "$(date) export $DET $split"
  "$PY" -u main_det_rois.py --resume "weights/$DET/model_best" --split "$split" --out-dir results/det --num_workers "$NUM_WORKERS" >> logs/prep_final_detector.log 2>&1 < /dev/null
  echo "$(date) export exit $?: $(tail -n 1 logs/prep_final_detector.log)"
done
if [ -f "$CROP_CACHE/index.json" ]; then
  echo "$(date) crop cache $CROP_CACHE exists, skipping"
else
  echo "$(date) building the crop cache in $CROP_CACHE (log logs/seg_crop_cache_final.log)"
  "$PY" -u - "$DATA_PATH" "$CROP_CACHE" "$NUM_WORKERS" "$DET" > logs/seg_crop_cache_final.log 2>&1 <<'EOF'
import sys
from data_loader.roi_crops import build_crop_cache
data_path, out_dir, num_workers, det = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
roi_files = {'ec10': {split: f'results/det/rois_{det}_{split}.json' for split in ('train', 'val')}}
info = build_crop_cache(data_path, out_dir, roi_files, splits=('train', 'val'), grow=2.0, min_side=512, num_workers=num_workers)
print(f"crop cache: {info['n_windows']} windows, {info['bytes'] / 1e9:.1f} GB -> {out_dir}")
EOF
  rc=$?
  echo "$(date) crop cache exit $rc: $(tail -n 1 logs/seg_crop_cache_final.log)"
fi
echo "$(date) prep done"
