# Raw Bands vs EC Responses (report, 2026-09-29)

Why the Stage-1 camouflage detector does better on the 133 raw hyperspectral bands than on the 344 EC filter
responses: the data, what the filter keeps, and the training results.

Published page (private to the owner until shared): https://claude.ai/artifact/3FKb55gD8hdBL7wQyXAKyF

## Main findings

| test split (70 frames, 71 objects) | raw 133 bands | 344 EC responses | learned top-10 |
|---|---|---|---|
| recall at IoU 0.5 | 0.761 | 0.563 | 0.521 |
| AP50 | 0.645 | 0.444 | 0.356 |
| objects fully inside an exported ROI | 0.761 | 0.549 | 0.465 |

- Per pixel, the 344 responses keep a median 93 % of the raw bands' object-vs-surround separability
  (Mahalanobis distance, 12 validation frames, 80–99 %); 11 whitened principal response directions keep 95 %;
  the gate's top-10 keeps 61 %.
- The responses are 344 near-copies (adjacent voltages 0.9998 cosine-similar, 11 directions hold 99.9 % of the
  energy, median bandwidth 112 nm vs 3 nm bands). Session A took 37 epochs to reach val AP50 0.3 (raw: 11) and
  plateaued lower; the gate weights zig-zag between neighbouring voltages. The PCA-11 run (in progress when
  this was written) reached AP50 0.3 by epoch 5, so input conditioning is the larger cost.
- The most discriminative single response is 1.73–1.75 V in 8 of 12 frames, which the top-10 excludes; the best
  raw band sits at 790–800 nm in 6 of 12 frames. Narrow features near 760 nm coincide with the O2-A absorption band.

## Rebuild

Run from the repo root with the `hsi_camo` env; outputs go to `docs/reports/2026-09-29-raw-vs-ec/build/`
(git-ignored) or `$REPORT_OUT`. Needs the fp16 frame cache and the `weights/` of det_A, det_B, raw133_A and
pca11_A (checkpoints are not in git). Only one full-resolution job fits on the machine at a time; the analysis
reads 12 validation and 12 test frames with O_DIRECT and runs a few seconds of GPU-1 inference per test frame.

```bash
PY=/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python
CUDA_VISIBLE_DEVICES="" $PY docs/reports/2026-09-29-raw-vs-ec/analyze_data.py       # ~2-4 min, CPU
CUDA_VISIBLE_DEVICES=1  $PY docs/reports/2026-09-29-raw-vs-ec/analyze_training.py   # ~1-2 min, GPU
CUDA_VISIBLE_DEVICES="" $PY docs/reports/2026-09-29-raw-vs-ec/build_report.py       # -> build/raw-vs-ec.html
```

`template.html` holds the page (inline CSS and JS, no external scripts); `build_report.py` embeds the data and the
images as data URIs.
