#!/usr/bin/env bash
# Export Stage-2 ROIs from the session-B detector for both splits (foreground, ~1-1.5 s/frame).
# Operating point = cfg/det.yaml roi_conf / roi_topk. Override the checkpoint with B_CKPT=weights/det_B/model_99.
cd "$(dirname "$0")/.." || exit 1
B_CKPT=${B_CKPT:-weights/det_B/model_best}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PY:-/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python}
mkdir -p logs
for split in test train; do
  "$PY" -u main_det_rois.py --session B --resume "$B_CKPT" --split "$split" --out-dir results/det \
      > "logs/rois_$split.log" 2>&1 || { echo "rois $split FAILED (see logs/rois_$split.log)"; exit 1; }
  tail -n 1 "logs/rois_$split.log"
done
