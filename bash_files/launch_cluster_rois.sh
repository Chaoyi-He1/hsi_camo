#!/usr/bin/env bash
# The "at most 2 objects per image" prior on the detector boxes (user's question, 2026-10-21): every box of an image is linked
# to the boxes it overlaps (main_fuse_rois.py --merge cluster), each cluster becomes one enclosing box / ROI, and only the two
# most confident clusters are kept. Applied to the greedy-10 detector's default export, its conf 0.001 / top-15 export (where
# 51 of 70 test images carry two or more boxes) and the augmented detector's model_99 export, with "plain top-2 boxes, no
# clustering" as the control; each set is scored end to end by the ec10 segmenters -> results/seg_fuse/<name>/compare.json.
#   bash bash_files/launch_cluster_rois.sh      # foreground, ~25 min on one GPU; existing compare.json files are skipped
# SEG_RUNS, NUM_WORKERS, PY and CUDA_VISIBLE_DEVICES (0) are overridable.
SELF=$(readlink -f "$0")
cd "$(dirname "$SELF")/.." || exit 1
PY=${PY:-/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} CUDA_DEVICE_ORDER=PCI_BUS_ID
SEG_RUNS=${SEG_RUNS:-"seg_sam2unet_ec10_s0 seg_sam2box_ec10_s0 seg_zoomnext_ec10_s0"}
NUM_WORKERS=${NUM_WORKERS:-4}
DEF=results/det/rois_sel10g_clean_A_test.json
LOW=results/det/rois_op/c0.001_k15/rois_sel10g_clean_A_test.json
AUG=results/det/rois_op/sel10g_aug_A_m99/rois_sel10g_aug_A_test.json
mkdir -p logs results/seg_fuse

fuse() {   # fuse <name> <input> <main_fuse_rois flags...>
  local name=$1 input=$2; shift 2
  "$PY" main_fuse_rois.py --inputs "$input" --name "$name" --out_dir results/det "$@" | tee -a logs/cluster_rois.log
}
evaluate() {   # evaluate <name>
  local out=results/seg_fuse/$1
  if [ -f "$out/compare.json" ]; then echo "$(date) $out already evaluated, skipping"; return; fi
  echo "$(date) evaluate -> $out" | tee -a logs/cluster_rois.log
  "$PY" -u main_seg_eval.py --runs $SEG_RUNS --out_dir "$out" --roi_run "$1" --num_workers "$NUM_WORKERS" > "$out.log" 2>&1 < /dev/null
  echo "$(date) evaluate exit $?: $(grep -m1 'wrote' "$out.log")" | tee -a logs/cluster_rois.log
}

fuse ec10_cl2            "$DEF" --merge cluster --max_rois 2
fuse ec10_c0.001_top2    "$LOW" --merge none --iou_dedup 1.0 --max_rois 2
fuse ec10_c0.001_cl2     "$LOW" --merge cluster --max_rois 2
fuse ec10_c0.001_cl2_gap "$LOW" --merge cluster --max_rois 2 --link_gap 0.25
fuse ec10_c0.001_cl1     "$LOW" --merge cluster --max_rois 1
fuse aug_m99_cl2         "$AUG" --merge cluster --max_rois 2
for name in ec10_cl2 ec10_c0.001_top2 ec10_c0.001_cl2 ec10_c0.001_cl2_gap ec10_c0.001_cl1 aug_m99_cl2; do
  evaluate "$name"
done
echo "$(date) cluster rois done" | tee -a logs/cluster_rois.log
