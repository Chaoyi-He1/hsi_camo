#!/usr/bin/env bash
# Stage-2 ROI export (spec Stage 2 §5.1): the three arm detectors, each rebuilt from its own checkpoint args, on the
# train (251 ids), val (28 held-out ids) and test (70) frames, one after another on one GPU (~1-1.5 s/frame, ~25 min):
#   raw133_A        raw 133 bands                       -> arm raw
#   sel10g_clean_A  greedy-10 voltages, whitened (10)   -> arm ec10
#   sel24g_clean_A  greedy-24 voltages, whitened (24)   -> arm ec24
# Output results/det/rois_<run>_<split>.json (operating point = cfg/det.yaml roi_conf 0.02 / roi_topk 5).
#   bash bash_files/launch_rois_all.sh    # detached; queue log logs/rois_all.log, per export logs/rois_<run>_<split>.log
# An export whose json already exists is skipped (FORCE=1 redoes it), so the queue can be relaunched after a crash.
# RUNS, SPLITS, CKPT (model_best), NUM_WORKERS (6), PY and CUDA_VISIBLE_DEVICES (0) are overridable.
SELF=$(readlink -f "$0")
cd "$(dirname "$SELF")/.." || exit 1
PY=${PY:-/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
mkdir -p logs results/det
if [ -z "${ROIS_ALL_CHILD:-}" ]; then
  ROIS_ALL_CHILD=1 setsid nohup bash "$SELF" "$@" > logs/rois_all.log 2>&1 < /dev/null &
  echo "ROI export queue launched, pid $!, log logs/rois_all.log"
  exit 0
fi

RUNS=${RUNS:-raw133_A sel10g_clean_A sel24g_clean_A}
SPLITS=${SPLITS:-train val test}
CKPT=${CKPT:-model_best}

rc_all=0
for run in $RUNS; do
  if [ ! -f "weights/$run/$CKPT" ]; then echo "$(date) weights/$run/$CKPT missing, skipping $run"; rc_all=1; continue; fi
  for split in $SPLITS; do
    out="results/det/rois_${run}_${split}.json"
    if [ -f "$out" ] && [ -z "${FORCE:-}" ]; then echo "$(date) $out exists, skipping"; continue; fi
    echo "$(date) start $run $split"
    "$PY" -u main_det_rois.py --resume "weights/$run/$CKPT" --split "$split" --out-dir results/det \
        --num_workers "${NUM_WORKERS:-6}" > "logs/rois_${run}_${split}.log" 2>&1 < /dev/null
    rc=$?
    echo "$(date) end $run $split, exit $rc: $(tail -n 1 "logs/rois_${run}_${split}.log")"
    [ $rc -eq 0 ] || rc_all=1
  done
done
echo "$(date) ROI export queue done, exit $rc_all"
exit $rc_all
