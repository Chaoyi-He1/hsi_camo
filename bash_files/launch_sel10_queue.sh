#!/usr/bin/env bash
# Three fixed-10-voltage runs one after another (one full-resolution job fits on the box), then the fixed-checkpoint
# comparison (spec §12, "how to pick the voltages"). Every run: 10 readings -> 40 dB read-noise floor -> noise-regularised
# whitening (--pca-channels 10) -> YOLO26s, no gate; only the voltages differ:
#   sel10g_A  the greedy set of main_select_voltages.py on the 251 train frames (spec §12; GREEDY="v1 ... v10" overrides)
#   sel10b_A  session B's gate top-10 (the voltages det_B trained on)
#   sel10u_A  10 uniformly spaced usable voltages (--filter-select uniform)
#   bash bash_files/launch_sel10_queue.sh        # detached; queue log logs/sel10_queue.log, run logs logs/<run>.log
# A run with a 'final' line in weights/<run>/results_<run>.txt is skipped, so the queue can be relaunched after a crash
# (an interrupted run restarts from epoch 0). RUNS="sel10g_A sel10u_A" restricts it; SNR_DB (40), NUM_WORKERS (6), PY and
# CUDA_VISIBLE_DEVICES (0) are overridable.
SELF=$(readlink -f "$0")
cd "$(dirname "$SELF")/.." || exit 1
PY=${PY:-/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
mkdir -p logs
if [ -z "${SEL10_QUEUE_CHILD:-}" ]; then
  SEL10_QUEUE_CHILD=1 setsid nohup bash "$SELF" "$@" > logs/sel10_queue.log 2>&1 < /dev/null &
  echo "sel10 queue launched, pid $!, log logs/sel10_queue.log"
  exit 0
fi

# greedy order of main_select_voltages.py (--k 10 --read_noise_db 40, 2026-10-01; results/det/voltages_greedy.json is
# git-ignored, spec §12 records the set); SESSION_B = det_A's gate top-10 (weights/det_A/gate_ranking_ep89.csv)
GREEDY=${GREEDY:-"1.75 -0.44 1.36 0.48 -0.65 -0.36 1.70 0.01 -0.86 1.45"}
SESSION_B="1.32 1.29 1.35 -0.41 -0.38 1.38 -0.44 1.41 1.26 -0.35"
[ -n "$GREEDY" ] || { echo "no greedy voltages"; exit 1; }
SNR_DB=${SNR_DB:-40}
RUNS=${RUNS:-sel10g_A sel10b_A sel10u_A}

finished() { [ -f "weights/$1/results_$1.txt" ] && grep -q '"final": true' "weights/$1/results_$1.txt"; }

train() {   # train <name> <filter flags...>
  local name=$1 rc; shift
  if finished "$name"; then echo "$(date) $name already finished, skipping"; return; fi
  mkdir -p "weights/$name"
  echo "$(date) start $name: $*"
  "$PY" -u main_det.py --session A --pca-channels 10 --read_noise_db "$SNR_DB" --name "$name" --output-dir "weights/$name" \
      --epochs 100 --batch_size 2 --accumulate 4 --num_workers "${NUM_WORKERS:-6}" "$@" > "logs/$name.log" 2>&1 < /dev/null
  rc=$?
  echo "$(date) end $name, exit $rc"
}

for run in $RUNS; do
  case $run in
    sel10g_A) train sel10g_A --filter-select manual --filter-voltages $GREEDY ;;
    sel10b_A) train sel10b_A --filter-select manual --filter-voltages $SESSION_B ;;
    sel10u_A) train sel10u_A --filter-select uniform --num-filters 10 ;;
    *) echo "unknown run $run"; exit 1 ;;
  esac
done
echo "$(date) comparison on fixed checkpoints"
"$PY" -u main_det_compare.py --runs $RUNS --out_dir results/det/compare_sel10 --num_workers "${NUM_WORKERS:-6}" > logs/compare_sel10.log 2>&1
rc=$?
echo "$(date) queue done, compare exit $rc"
