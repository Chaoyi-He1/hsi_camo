#!/usr/bin/env bash
# Follow-up to launch_sel10_queue.sh (spec §12, "how to pick the voltages"): waits until that queue has written
# "queue done" to logs/sel10_queue.log (one full-resolution job fits on the box), then trains one after another
#   sel24g_A        the first 24 voltages of the nested greedy order (main_select_voltages.py --k 24, 40 dB floor),
#                   --pca-channels 24: does a larger reading budget close the gap to PCA-11 / raw? (pixel level: the
#                   set keeps 0.846 -> 0.865 of the raw bands' separability at 40 dB, all 344 readings 0.918)
#   sel10g_clean_A  the 10 greedy voltages WITHOUT read noise (--read_noise_db 0): the pure 10-reading ceiling; the
#                   difference to sel10g_A is the cost of the 40 dB floor
# and then main_det_compare.py over all five sel runs -> results/det/compare_sel_all.
#   bash bash_files/launch_sel_followup.sh       # detached; queue log logs/sel_followup.log, run logs logs/<run>.log
# A run with a 'final' line in weights/<run>/results_<run>.txt is skipped, so the queue can be relaunched after a crash
# (an interrupted run restarts from epoch 0). RUNS="sel24g_A" restricts it; SNR_DB (40), NUM_WORKERS (6), PY and
# CUDA_VISIBLE_DEVICES (0) are overridable.
SELF=$(readlink -f "$0")
cd "$(dirname "$SELF")/.." || exit 1
PY=${PY:-/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
mkdir -p logs
if [ -z "${SEL_FOLLOWUP_CHILD:-}" ]; then
  SEL_FOLLOWUP_CHILD=1 setsid nohup bash "$SELF" "$@" > logs/sel_followup.log 2>&1 < /dev/null &
  echo "sel follow-up queue launched, pid $!, log logs/sel_followup.log"
  exit 0
fi

# nested greedy order of main_select_voltages.py (--k 24 --read_noise_db 40, 2026-10-02; results/det/voltages_greedy24.json
# is git-ignored, spec §12 records the set); the first 10 are the sel10g_A voltages
GREEDY24="1.75 -0.44 1.36 0.48 -0.65 -0.36 1.70 0.01 -0.86 1.45 0.37 2.34 -0.31 1.65 -1.00 -0.60 2.50 1.60 0.19 -0.91 1.80 2.29 1.31 0.53"
GREEDY10="1.75 -0.44 1.36 0.48 -0.65 -0.36 1.70 0.01 -0.86 1.45"
SNR_DB=${SNR_DB:-40}
RUNS=${RUNS:-sel24g_A sel10g_clean_A}

echo "$(date) waiting for the first queue (logs/sel10_queue.log: 'queue done')"
until grep -q "queue done" logs/sel10_queue.log 2>/dev/null; do sleep 120; done
echo "$(date) first queue done"

finished() { [ -f "weights/$1/results_$1.txt" ] && grep -q '"final": true' "weights/$1/results_$1.txt"; }

train() {   # train <name> <filter / noise flags...>
  local name=$1 rc; shift
  if finished "$name"; then echo "$(date) $name already finished, skipping"; return; fi
  mkdir -p "weights/$name"
  echo "$(date) start $name: $*"
  "$PY" -u main_det.py --session A --name "$name" --output-dir "weights/$name" \
      --epochs 100 --batch_size 2 --accumulate 4 --num_workers "${NUM_WORKERS:-6}" "$@" > "logs/$name.log" 2>&1 < /dev/null
  rc=$?
  echo "$(date) end $name, exit $rc"
}

for run in $RUNS; do
  case $run in
    sel24g_A)       train sel24g_A       --filter-select manual --filter-voltages $GREEDY24 --pca-channels 24 --read_noise_db "$SNR_DB" ;;
    sel10g_clean_A) train sel10g_clean_A --filter-select manual --filter-voltages $GREEDY10 --pca-channels 10 --read_noise_db 0 ;;
    *) echo "unknown run $run"; exit 1 ;;
  esac
done
echo "$(date) comparison over all sel runs on fixed checkpoints"
"$PY" -u main_det_compare.py --runs sel10g_A sel10b_A sel10u_A sel24g_A sel10g_clean_A --out_dir results/det/compare_sel_all \
    --num_workers "${NUM_WORKERS:-6}" > logs/compare_sel_all.log 2>&1
rc=$?
echo "$(date) queue done, compare exit $rc"
