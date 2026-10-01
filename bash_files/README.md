# bash_files

Operational shell scripts for Stage 1 (kept out of the repo root). All of them `cd` to the repo root themselves, so run
them from anywhere: `bash bash_files/<script>.sh`. `PY` (python binary) and the GPU/worker knobs are env-overridable.

| script | what it does |
|---|---|
| `make_env.sh` | creates the `hsi_camo` conda env (torch 2.11 + cu128, ultralytics, wandb, ...) |
| `launch_det_A.sh` | session A: all 344 voltages + weight vector, 100 epochs, detached; log `logs/det_A.log` |
| `launch_det_B.sh` | session B: top-10 voltages from A (`A_CKPT`, `A_RANK` overridable), detached; log `logs/det_B.log` |
| `launch_rois.sh` | Stage-2 ROI export for test and train from `B_CKPT` (default `weights/det_B/model_best`) |
| `launch_raw133.sh` | control run: raw 133 bands (400-800 nm) into YOLO, no filter, no gate (GPU 1 by default) |
| `launch_pca11.sh` | filter model on the K PCA-whitened response directions (default K=11), no gate; `UNIFORM=12` for a uniform-voltage control |
| `launch_sel10_queue.sh` | three fixed-10-voltage runs in sequence (greedy / session-B / uniform voltages, 40 dB read noise, whitened), then `main_det_compare.py` on their fixed checkpoints |
| `copy_cache_to_nvme.sh` | optional: mirror the 181 GB fp16 frame cache to the NVMe `/home` partition (ask first) |

Background runs use `python -u` (redirected stdout is block-buffered otherwise) and `setsid nohup`, so they survive the
launching shell. Only one full-resolution training job fits on the box at a time (see spec §12).
