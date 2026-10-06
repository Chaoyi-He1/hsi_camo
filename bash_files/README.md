# bash_files

Operational shell scripts for Stages 1 and 2 (kept out of the repo root). All of them `cd` to the repo root themselves, so run
them from anywhere: `bash bash_files/<script>.sh`. `PY` (python binary) and the GPU/worker knobs are env-overridable.

| script | what it does |
|---|---|
| `make_env.sh` | creates the `hsi_camo` conda env (torch 2.11 + cu128, ultralytics, wandb, ...) |
| `launch_det_A.sh` | session A: all 344 voltages + weight vector, 100 epochs, detached; log `logs/det_A.log` |
| `launch_det_B.sh` | session B: top-10 voltages from A (`A_CKPT`, `A_RANK` overridable), detached; log `logs/det_B.log` |
| `launch_rois_all.sh` | Stage-2 ROI export: `raw133_A`, `sel10g_clean_A`, `sel24g_clean_A` (rebuilt from their checkpoint args) on train / val / test, sequential on GPU 0, detached; `results/det/rois_<run>_<split>.json`, existing files skipped (`FORCE=1` redoes) |
| `launch_raw133.sh` | control run: raw 133 bands (400-800 nm) into YOLO, no filter, no gate (GPU 1 by default) |
| `launch_pca11.sh` | filter model on the K PCA-whitened response directions (default K=11), no gate; `UNIFORM=12` for a uniform-voltage control |
| `launch_sel10_queue.sh` | three fixed-10-voltage runs in sequence (greedy / session-B / uniform voltages, 40 dB read noise, whitened), then `main_det_compare.py` on their fixed checkpoints |
| `launch_sel_followup.sh` | waits for the queue above, then the 24-voltage greedy run (40 dB) and the 10-voltage greedy run without read noise, then `main_det_compare.py` over all five |
| `launch_sel_clean_queue.sh` | noise-free redo: session-B / uniform / greedy-24 voltage sets without read noise, then `main_det_compare.py` of every noise-free run (incl. PCA-11, raw, sessions A and B) |
| `copy_cache_to_nvme.sh` | optional: mirror the 181 GB fp16 frame cache to the NVMe `/home` partition (ask first) |
| `setup_third_party.sh` | Stage 2, once per machine: pinned `sam2` + hydra-core, `pysodmetrics` (no deps, keeps numpy 2.5.2) + scikit-image/learn, einops + timm; ZoomNeXt clone into git-ignored `third_party/zoomnext/` (no licence: research use only); SAM2 v1 / SAM2.1 Hiera-L and ZoomNeXt-B2 checkpoints into `weights/pretrained/`; verifies every checkpoint strict-loads |
| `launch_seg_queue.sh` | Stage 2: crop cache once (if `index.json` is missing), then the 27 `seg_<model>_<arm>_s<seed>` runs (sam2unet / sam2box / zoomnext x raw / ec10 / ec24 x seeds 0-2, seed-major) and the `seg_sam2unet_rgb` control, then `main_seg_eval.py` over every finished run + the zero-shot SAM2.1 control -> `results/seg/compare.json` (and `model_last` -> `results/seg_last`, `EVAL_LAST=` disables); batch size / accumulation from `cfg/seg.yaml`; detached, log `logs/seg_queue.log`; `LIST_ONLY=1` prints the plan, `CACHE_ONLY=1` stops after the cache |

Background runs use `python -u` (redirected stdout is block-buffered otherwise) and `setsid nohup`, so they survive the
launching shell. Only one full-resolution training job fits on the box at a time (see spec §12).
