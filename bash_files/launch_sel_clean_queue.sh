#!/usr/bin/env bash
# Noise-free redo of the voltage-set study (spec §12, "how to pick the voltages"): the same fixed voltage sets as the 40 dB
# runs, trained WITHOUT read noise (the 40 dB floor was an assumption, not a measurement), one after another, then the
# fixed-checkpoint comparison of every noise-free run. sel10g_clean_A (10 greedy voltages, no noise) already exists.
#   sel10b_clean_A  session B's gate top-10 voltages, --pca-channels 10
#   sel10u_clean_A  10 uniformly spaced usable voltages, --pca-channels 10
#   sel24g_clean_A  the first 24 voltages of the nested greedy order, --pca-channels 24
# Comparison 1: the four sel_clean runs + pca11_A + raw133_A + det_A on model_59..99 -> results/det/compare_clean
# Comparison 2: the four sel_clean runs + det_B (stopped at epoch 91) on model_59..89  -> results/det/compare_clean_detB
#   bash bash_files/launch_sel_clean_queue.sh    # detached; queue log logs/sel_clean_queue.log, run logs logs/<run>.log
# A run with a 'final' line in weights/<run>/results_<run>.txt is skipped, so the queue can be relaunched after a crash
# (an interrupted run restarts from epoch 0). RUNS="sel24g_clean_A" restricts it; NUM_WORKERS (6), PY and
# CUDA_VISIBLE_DEVICES (0) are overridable. One full-resolution job fits on the box: do not start another training meanwhile.
SELF=$(readlink -f "$0")
cd "$(dirname "$SELF")/.." || exit 1
PY=${PY:-/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
mkdir -p logs
if [ -z "${SEL_CLEAN_QUEUE_CHILD:-}" ]; then
  SEL_CLEAN_QUEUE_CHILD=1 setsid nohup bash "$SELF" "$@" > logs/sel_clean_queue.log 2>&1 < /dev/null &
  echo "sel clean queue launched, pid $!, log logs/sel_clean_queue.log"
  exit 0
fi

# the voltage sets of launch_sel10_queue.sh / launch_sel_followup.sh (spec §12 records them)
GREEDY24="1.75 -0.44 1.36 0.48 -0.65 -0.36 1.70 0.01 -0.86 1.45 0.37 2.34 -0.31 1.65 -1.00 -0.60 2.50 1.60 0.19 -0.91 1.80 2.29 1.31 0.53"
SESSION_B="1.32 1.29 1.35 -0.41 -0.38 1.38 -0.44 1.41 1.26 -0.35"
RUNS=${RUNS:-sel10b_clean_A sel10u_clean_A sel24g_clean_A}

finished() { [ -f "weights/$1/results_$1.txt" ] && grep -q '"final": true' "weights/$1/results_$1.txt"; }

train() {   # train <name> <filter flags...>
  local name=$1 rc; shift
  if finished "$name"; then echo "$(date) $name already finished, skipping"; return; fi
  mkdir -p "weights/$name"
  echo "$(date) start $name: $*"
  "$PY" -u main_det.py --session A --read_noise_db 0 --name "$name" --output-dir "weights/$name" \
      --epochs 100 --batch_size 2 --accumulate 4 --num_workers "${NUM_WORKERS:-6}" "$@" > "logs/$name.log" 2>&1 < /dev/null
  rc=$?
  echo "$(date) end $name, exit $rc"
}

for run in $RUNS; do
  case $run in
    sel10b_clean_A) train sel10b_clean_A --filter-select manual --filter-voltages $SESSION_B --pca-channels 10 ;;
    sel10u_clean_A) train sel10u_clean_A --filter-select uniform --num-filters 10 --pca-channels 10 ;;
    sel24g_clean_A) train sel24g_clean_A --filter-select manual --filter-voltages $GREEDY24 --pca-channels 24 ;;
    *) echo "unknown run $run"; exit 1 ;;
  esac
done
echo "$(date) comparison 1: noise-free runs + pca11_A + raw133_A + det_A on model_59..99"
"$PY" -u main_det_compare.py --runs sel10g_clean_A sel10b_clean_A sel10u_clean_A sel24g_clean_A pca11_A raw133_A det_A \
    --conditions clean --out_dir results/det/compare_clean --num_workers "${NUM_WORKERS:-6}" > logs/compare_clean.log 2>&1
rc1=$?
echo "$(date) comparison 2: noise-free runs + det_B on model_59..89"
"$PY" -u main_det_compare.py --runs sel10g_clean_A sel10b_clean_A sel10u_clean_A sel24g_clean_A det_B --ckpt_epochs 59 69 79 89 \
    --conditions clean --out_dir results/det/compare_clean_detB --num_workers "${NUM_WORKERS:-6}" > logs/compare_clean_detB.log 2>&1
rc2=$?
echo "$(date) queue done, compare exits $rc1 $rc2"
