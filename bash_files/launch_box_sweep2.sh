#!/usr/bin/env bash
# Second box study (2026-10-19, after launch_box_sweep.sh): (a) overlapping boxes merged into one ROI (main_fuse_rois --merge
# union | wbf) for the ec10 detector alone and for the raw + ec10 union, (b) "no detector" exports (main_window_rois.py: the
# whole frame as one ROI, and a 512-px window grid) -- all scored by the three ec10 segmenters (seed 0) and, for (b), the
# zero-shot SAM2.1 control as well. Waits for "box sweep done" in logs/box_sweep.log before touching the GPU.
#   bash bash_files/launch_box_sweep2.sh     # detached; log logs/box_sweep2.log; outputs results/det/rois_<name>_test.json,
#                                            # results/seg_fuse/<name>/compare.json, results/seg_nodet/<name>/compare.json
# SEG_RUNS, NUM_WORKERS (3), PY and CUDA_VISIBLE_DEVICES (1) are overridable; finished evaluations are skipped on relaunch.
SELF=$(readlink -f "$0")
cd "$(dirname "$SELF")/.." || exit 1
PY=${PY:-/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1} CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p logs
if [ -z "${BOX_SWEEP2_CHILD:-}" ]; then
  BOX_SWEEP2_CHILD=1 setsid nohup bash "$SELF" "$@" >> logs/box_sweep2.log 2>&1 < /dev/null &
  echo "box sweep 2 launched on GPU $CUDA_VISIBLE_DEVICES, pid $!, log logs/box_sweep2.log (starts after logs/box_sweep.log says 'box sweep done')"
  exit 0
fi
NUM_WORKERS=${NUM_WORKERS:-3}
SEG_RUNS=${SEG_RUNS:-"seg_sam2unet_ec10_s0 seg_sam2box_ec10_s0 seg_zoomnext_ec10_s0"}
until grep -q "box sweep done" logs/box_sweep.log 2>/dev/null; do sleep 120; done
echo "$(date) box sweep 2 on GPU $CUDA_VISIBLE_DEVICES, pid $$"

evaluate() {   # evaluate <out dir> [main_seg_eval flags...]
  local out=$1; shift
  if [ -f "$out/compare.json" ]; then echo "$(date) $out/compare.json exists, skipping"; return; fi
  mkdir -p "$(dirname "$out")"
  echo "$(date) evaluate -> $out $*"
  "$PY" -u main_seg_eval.py --runs $SEG_RUNS --out_dir "$out" --num_workers "$NUM_WORKERS" "$@" > "$out.log" 2>&1 < /dev/null
  echo "$(date) evaluate exit $?: $(grep -m1 'wrote' "$out.log")"
}
mk() {   # mk <name> <python args...>  (skips an existing export)
  local name=$1; shift
  if [ -f "results/det/rois_${name}_test.json" ]; then echo "$(date) rois_${name}_test.json exists, skipping"; return; fi
  "$PY" "$@" --name "$name" --out_dir results/det | tee -a logs/box_sweep_exports.log
}

# (a) merged overlapping boxes
mk ec10_union  main_fuse_rois.py --inputs results/det/rois_sel10g_clean_A_test.json --merge union --iou_merge 0.5
mk ec10_wbf    main_fuse_rois.py --inputs results/det/rois_sel10g_clean_A_test.json --merge wbf --iou_merge 0.5
mk fuse_raw_ec10_union main_fuse_rois.py --inputs results/det/rois_raw133_A_test.json results/det/rois_sel10g_clean_A_test.json --merge union --iou_merge 0.5
mk ec10_c0.005_union main_fuse_rois.py --inputs results/det/rois_op/c0.005_k5/rois_sel10g_clean_A_test.json --merge union --iou_merge 0.5
for name in ec10_union ec10_wbf fuse_raw_ec10_union ec10_c0.005_union; do
  evaluate "results/seg_fuse/$name" --roi_run "$name"
done

# (b) no detector: the whole frame as one ROI, and a window grid; the zero-shot control is scored too
mk full_frame  main_window_rois.py --mode frame
mk windows_512 main_window_rois.py --mode windows --window 512 --stride 384
for name in full_frame windows_512; do
  evaluate "results/seg_nodet/$name" --roi_run "$name" --zero_shot
done
echo "$(date) box sweep 2 done"
