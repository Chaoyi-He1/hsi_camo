#!/usr/bin/env bash
# Evaluation-only follow-up of the recall study (bash_files/launch_aug_queue.sh): the boxes of sel10g_aug_A's LAST checkpoint
# (model_99 = the EMA weights) at three operating points, scored end to end by the ec10 segmenters. The queue exported model_best
# (epoch 49, chosen on the 28 validation images), which the fixed-checkpoint comparison (model_59..99) showed to be a weaker
# checkpoint than every later one (ROI coverage@op 0.775 vs 0.845). Exports -> results/det/rois_op/sel10g_aug_A_m99*/, Stage 2 ->
# results/seg_aug/sel10g_aug_A_m99*/compare.json. ~20 min on one GPU (reads the test frames from the data-path cache, no mirror
# needed); existing compare.json files are skipped. RUN, CKPT, OPS ("tag:conf:topk" ...), SEG_RUNS, NUM_WORKERS, PY and
# CUDA_VISIBLE_DEVICES (0) are overridable.
#   bash bash_files/launch_aug_followup.sh      # foreground; log lines on stdout, export details in logs/aug_followup_exports.log
SELF=$(readlink -f "$0")
cd "$(dirname "$SELF")/.." || exit 1
PY=${PY:-/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} CUDA_DEVICE_ORDER=PCI_BUS_ID
RUN=${RUN:-sel10g_aug_A}
CKPT=${CKPT:-model_99}
OPS=${OPS:-"m99:0.02:5 m99_c0.01_k5:0.01:5 m99_c0.005_k5:0.005:5"}
SEG_RUNS=${SEG_RUNS:-"seg_sam2unet_ec10_s0 seg_sam2box_ec10_s0 seg_zoomnext_ec10_s0"}
NUM_WORKERS=${NUM_WORKERS:-4}
mkdir -p logs results/seg_aug
for spec in $OPS; do
  tag=${spec%%:*}; rest=${spec#*:}; conf=${rest%%:*}; k=${rest#*:}
  out=results/det/rois_op/${RUN}_$tag
  seg=results/seg_aug/${RUN}_$tag
  if [ -f "$seg/compare.json" ]; then echo "$(date) $seg already evaluated, skipping"; continue; fi
  mkdir -p "$out"
  echo "$(date) export $RUN $CKPT -> $out (conf $conf, top $k)"
  "$PY" -u main_det_rois.py --resume "weights/$RUN/$CKPT" --split test --out-dir "$out" --num_workers "$NUM_WORKERS" --roi-conf "$conf" --max-rois "$k" >> logs/aug_followup_exports.log 2>&1 < /dev/null
  echo "$(date) export exit $?: $(tail -n 1 logs/aug_followup_exports.log)"
  echo "$(date) evaluate -> $seg"
  "$PY" -u main_seg_eval.py --runs $SEG_RUNS --out_dir "$seg" --roi_run "$RUN" --roi_dir "$out" --num_workers "$NUM_WORKERS" > "$seg.log" 2>&1 < /dev/null
  echo "$(date) evaluate exit $?: $(grep -m1 'wrote' "$seg.log")"
done
echo "$(date) followup done"
