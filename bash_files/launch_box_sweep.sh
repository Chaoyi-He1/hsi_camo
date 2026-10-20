#!/usr/bin/env bash
# How much end-to-end segmentation the detector boxes cost, and what cheaper boxes buy (plan "1 and 2", 2026-10-19), with no
# training: (1) an operating-point sweep of the ec10 detector sel10g_clean_A (confidence floor x boxes per image), (2) fused
# boxes of several detectors (main_fuse_rois.py: raw + ec10, the three greedy-10 seeds, raw + ec10 + ec24, raw + the three seeds).
# Every box set is scored by the three ec10 segmenters (seed 0) with main_seg_eval.py; only level (c) changes.
#   bash bash_files/launch_box_sweep.sh      # detached; log logs/box_sweep.log; outputs results/det/rois_op/<op>/, results/det/rois_fuse_*,
#                                            # results/seg_op/<op>/compare.json, results/seg_fuse/<name>/compare.json
# OPS ("conf:topk" pairs), SEG_RUNS, NUM_WORKERS (3: the GPUs are shared), PY and CUDA_VISIBLE_DEVICES (1) are overridable.
# Exports and evaluations whose compare.json / export already exist are skipped, so the script can be relaunched.
SELF=$(readlink -f "$0")
cd "$(dirname "$SELF")/.." || exit 1
PY=${PY:-/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1} CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p logs
if [ -z "${BOX_SWEEP_CHILD:-}" ]; then
  BOX_SWEEP_CHILD=1 setsid nohup bash "$SELF" "$@" >> logs/box_sweep.log 2>&1 < /dev/null &
  echo "box sweep launched on GPU $CUDA_VISIBLE_DEVICES, pid $!, log logs/box_sweep.log"
  exit 0
fi
NUM_WORKERS=${NUM_WORKERS:-3}
OPS=${OPS:-"0.01:5 0.005:5 0.002:5 0.02:10 0.005:10 0.001:15"}
SEG_RUNS=${SEG_RUNS:-"seg_sam2unet_ec10_s0 seg_sam2box_ec10_s0 seg_zoomnext_ec10_s0"}
DET=sel10g_clean_A
echo "$(date) box sweep on GPU $CUDA_VISIBLE_DEVICES, pid $$: ops $OPS; fusions raw+ec10, ec10 seeds, raw+ec10+ec24, raw+ec10 seeds"

export_rois() {   # export_rois <det run> <out dir> [extra flags...]
  local run=$1 out=$2; shift 2
  if [ -f "$out/rois_${run}_test.json" ]; then echo "$(date) $out/rois_${run}_test.json exists, skipping"; return; fi
  echo "$(date) export $run -> $out $*"
  "$PY" -u main_det_rois.py --resume "weights/$run/model_best" --split test --out-dir "$out" --num_workers "$NUM_WORKERS" "$@" >> logs/box_sweep_exports.log 2>&1 < /dev/null
  echo "$(date) export exit $?: $(tail -n 1 logs/box_sweep_exports.log)"
}
evaluate() {   # evaluate <out dir> [main_seg_eval flags...]
  local out=$1; shift
  if [ -f "$out/compare.json" ]; then echo "$(date) $out/compare.json exists, skipping"; return; fi
  echo "$(date) evaluate -> $out $*"
  "$PY" -u main_seg_eval.py --runs $SEG_RUNS --out_dir "$out" --num_workers "$NUM_WORKERS" "$@" > "$out.log" 2>&1 < /dev/null
  echo "$(date) evaluate exit $?: $(grep -m1 'wrote' "$out.log")"
}

# (1) the operating-point sweep: one export + one evaluation per (conf, topk)
for op in $OPS; do
  conf=${op%%:*}; k=${op##*:}; tag="c${conf}_k${k}"
  export_rois "$DET" "results/det/rois_op/$tag" --roi-conf "$conf" --max-rois "$k"
  mkdir -p results/seg_op
  evaluate "results/seg_op/$tag" --roi_dir "results/det/rois_op/$tag"
done

# (2) fused boxes at the default operating point: the seeds' exports first
export_rois sel10g_clean_A_s1 results/det
export_rois sel10g_clean_A_s2 results/det
fuse() {   # fuse <name> <inputs...>
  local name=$1; shift
  if [ -f "results/det/rois_${name}_test.json" ]; then echo "$(date) rois_${name}_test.json exists, skipping"; return; fi
  "$PY" main_fuse_rois.py --inputs "$@" --name "$name" --out_dir results/det | tee -a logs/box_sweep_exports.log
}
fuse fuse_raw_ec10 results/det/rois_raw133_A_test.json results/det/rois_sel10g_clean_A_test.json
fuse fuse_ec10_seeds results/det/rois_sel10g_clean_A_test.json results/det/rois_sel10g_clean_A_s1_test.json results/det/rois_sel10g_clean_A_s2_test.json
fuse fuse_raw_ec10_ec24 results/det/rois_raw133_A_test.json results/det/rois_sel10g_clean_A_test.json results/det/rois_sel24g_clean_A_test.json
fuse fuse_raw_ec10_seeds results/det/rois_raw133_A_test.json results/det/rois_sel10g_clean_A_test.json results/det/rois_sel10g_clean_A_s1_test.json results/det/rois_sel10g_clean_A_s2_test.json
mkdir -p results/seg_fuse
for name in fuse_raw_ec10 fuse_ec10_seeds fuse_raw_ec10_ec24 fuse_raw_ec10_seeds; do
  evaluate "results/seg_fuse/$name" --roi_run "$name"
done
echo "$(date) box sweep done"
