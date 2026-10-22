#!/usr/bin/env bash
# Sequencing helper (2026-10-21): a full-resolution Stage-1 training and a Stage-2 training cannot share the box (memory reclaim
# stall: PSI 68 %, 97 s / step), so the final Stage-2 queue (bash_files/launch_seg_final_queue.sh) is started only after the
# detector seed queue (bash_files/launch_aug_seeds_queue.sh) has printed "queue done" and no python process computes on either
# GPU any more (its exports / evaluations included). Then both GPUs take the plan: GPU 0 evaluates at the end, GPU 1 does not.
#   bash bash_files/launch_seg_final_after_seeds.sh      # detaches; log logs/seg_final_after_seeds.log
SELF=$(readlink -f "$0")
cd "$(dirname "$SELF")/.." || exit 1
mkdir -p logs
if [ -z "${AFTER_SEEDS_CHILD:-}" ]; then
  AFTER_SEEDS_CHILD=1 setsid nohup bash "$SELF" "$@" >> logs/seg_final_after_seeds.log 2>&1 < /dev/null &
  echo "waiter launched, pid $!, log logs/seg_final_after_seeds.log"
  exit 0
fi
echo "$(date) waiting for 'queue done' in logs/aug_seeds_queue.log"
until grep -q "queue done" logs/aug_seeds_queue.log 2>/dev/null; do sleep 120; done
echo "$(date) seed queue done; waiting for the GPUs to be free"
while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; do sleep 60; done
echo "$(date) GPUs free: launching the final seg queue on GPU 0 (evaluating) and GPU 1"
CUDA_VISIBLE_DEVICES=0 bash bash_files/launch_seg_final_queue.sh
sleep 90                                                    # let the first queue take its run before the second picks one
CUDA_VISIBLE_DEVICES=1 EVAL=0 bash bash_files/launch_seg_final_queue.sh
echo "$(date) both final seg queues launched"
