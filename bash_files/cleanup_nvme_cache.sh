#!/usr/bin/env bash
# Delete the NVMe mirror of the fp16 frame cache (bash_files/copy_cache_to_nvme.sh) once the queue that uses it has finished:
# waits for the "queue done" line of the queue log, then for the last main_det / main_det_compare process to exit, then
# removes the mirror and prints the freed space. Detaches itself, so it does not need the launching shell to stay alive.
#   bash bash_files/cleanup_nvme_cache.sh                                   # watches logs/rawpca_seeds_queue.log
#   QUEUE_LOG=logs/other_queue.log bash bash_files/cleanup_nvme_cache.sh    # another queue
# DST is the mirror path (the copy script's default); the script refuses any other path. Log: logs/cleanup_nvme_cache.log.
SELF=$(readlink -f "$0")
cd "$(dirname "$SELF")/.." || exit 1
DST=${DST:-"/home/staff/c/chaoyi_he/hsi_camo_cache/cache_fp16"}
QUEUE_LOG=${QUEUE_LOG:-logs/rawpca_seeds_queue.log}
case "$DST" in
  /home/staff/c/chaoyi_he/hsi_camo_cache/cache_fp16) ;;
  *) echo "refusing to delete $DST: not the cache mirror"; exit 1 ;;
esac
mkdir -p logs
if [ -z "${CLEANUP_CHILD:-}" ]; then
  CLEANUP_CHILD=1 setsid nohup bash "$SELF" "$@" >> logs/cleanup_nvme_cache.log 2>&1 < /dev/null &
  echo "cleanup watcher launched, pid $!, log logs/cleanup_nvme_cache.log (deletes $DST after $QUEUE_LOG says 'queue done')"
  exit 0
fi
echo "$(date) watching $QUEUE_LOG; will delete $DST when the queue is done"
until grep -q "queue done" "$QUEUE_LOG" 2>/dev/null; do sleep 300; done
echo "$(date) queue done; waiting for the last main_det / main_det_compare process to exit"
while pgrep -f "main_det.py|main_det_compare.py" > /dev/null; do sleep 60; done
if [ -d "$DST" ]; then
  echo "$(date) deleting $DST ($(du -sh "$DST" | cut -f1))"
  rm -rf "$DST"
  rmdir "$(dirname "$DST")" 2>/dev/null
  echo "$(date) deleted; $(df -h /home | tail -n 1)"
else
  echo "$(date) $DST already gone"
fi
