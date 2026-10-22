#!/usr/bin/env bash
# Operating-point and object-count-prior rows of the FINAL pipeline (2026-10-22): the final detector sel10g_aug_A_m99's boxes at the
# default point clustered to <= 2 objects per image, at conf 0.001 / top 15, and the latter clustered, each scored end to end by the
# seed-0 final segmenters seg_{sam2unet,sam2box,zoomnext}_ec10f_s0 -> results/seg_final_op/<name>/compare.json. Evaluation only,
# ~20 min on one GPU, after bash_files/launch_seg_final_queue.sh has trained the segmenters. Existing compare.json files are skipped.
#   bash bash_files/launch_seg_final_op.sh      # foreground; prints "final op done" at the end
SELF=$(readlink -f "$0")
cd "$(dirname "$SELF")/.." || exit 1
PY=${PY:-/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} CUDA_DEVICE_ORDER=PCI_BUS_ID
DET=${DET:-sel10g_aug_A_m99}
SEG_RUNS=${SEG_RUNS:-"seg_sam2unet_ec10f_s0 seg_sam2box_ec10f_s0 seg_zoomnext_ec10f_s0"}
NUM_WORKERS=${NUM_WORKERS:-4}
mkdir -p logs results/seg_final_op "results/det/rois_op/${DET}_c0.001_k15"
LOW="results/det/rois_op/${DET}_c0.001_k15/rois_${DET}_test.json"
if [ ! -f "$LOW" ]; then
  echo "$(date) export $DET at conf 0.001 / top 15"
  "$PY" -u main_det_rois.py --resume "weights/$DET/model_best" --split test --out-dir "results/det/rois_op/${DET}_c0.001_k15" --num_workers "$NUM_WORKERS" --roi-conf 0.001 --max-rois 15 >> logs/seg_final_op.log 2>&1 < /dev/null
  echo "$(date) export exit $?: $(tail -n 1 logs/seg_final_op.log)"
fi
[ -f "results/det/rois_${DET}_cl2_test.json" ] || "$PY" main_fuse_rois.py --inputs "results/det/rois_${DET}_test.json" --name "${DET}_cl2" --merge cluster --max_rois 2 --out_dir results/det | tee -a logs/seg_final_op.log
[ -f "results/det/rois_${DET}_c0.001_cl2_test.json" ] || "$PY" main_fuse_rois.py --inputs "$LOW" --name "${DET}_c0.001_cl2" --merge cluster --max_rois 2 --out_dir results/det | tee -a logs/seg_final_op.log

evaluate() {   # evaluate <name> [main_seg_eval flags...]
  local out=results/seg_final_op/$1; shift
  if [ -f "$out/compare.json" ]; then echo "$(date) $out already evaluated, skipping"; return; fi
  echo "$(date) evaluate -> $out $*"
  "$PY" -u main_seg_eval.py --runs $SEG_RUNS --out_dir "$out" --num_workers "$NUM_WORKERS" "$@" > "$out.log" 2>&1 < /dev/null
  echo "$(date) evaluate exit $?: $(grep -m1 'wrote' "$out.log")"
}
evaluate "${DET}_cl2"        --roi_run "${DET}_cl2"
evaluate "${DET}_c0.001_k15" --roi_run "$DET" --roi_dir "results/det/rois_op/${DET}_c0.001_k15"
evaluate "${DET}_c0.001_cl2" --roi_run "${DET}_c0.001_cl2"
echo "$(date) final op done"
