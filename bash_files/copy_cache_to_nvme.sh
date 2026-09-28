#!/usr/bin/env bash
# OPTIONAL: mirror the fp16 frame cache (~181 GB) to the NVMe /home partition so training is no longer bound by the
# SATA SSD on /data2 (~380 MB/s to 4-6 workers -> ~2 s/step; NVMe -> GPU-bound ~0.7 s/step).
# Needs the user's OK (shared partition). Afterwards launch with:  main_det.py ... --cache-dir "$DST"
SRC=${SRC:-"/data2/chaoyi/HyperCOD/Raw data/cache_fp16"}
DST=${DST:-"/home/staff/c/chaoyi_he/hsi_camo_cache/cache_fp16"}
set -u
mkdir -p "$DST"
df -h "$(dirname "$DST")" | tail -n 1
date; echo "copying train (144 GB) + test (37 GB) ..."
rsync -a --info=progress2 "$SRC/train" "$SRC/test" "$DST/" || { echo "rsync failed"; exit 1; }
date; du -sh "$DST"; df -h "$(dirname "$DST")" | tail -n 1
