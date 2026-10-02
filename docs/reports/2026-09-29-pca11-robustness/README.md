# PCA-11 follow-up: fixed-checkpoint comparison and robustness (2026-09-29)

Inference-only evaluation of the finished Stage-1 checkpoints behind spec §12 ("PCA-11 follow-up"). Nothing is
trained here. Every (model, condition) pair sees the same frames in one pass, so each 0.55 GB frame is read once.

```bash
CUDA_VISIBLE_DEVICES=0 python docs/reports/2026-09-29-pca11-robustness/robust_eval.py \
    --weights-dir /data/chaoyi_he/hsi_camo/.claude/worktrees/det/weights --out results/det/robustness
# --smoke: 2 frames and a reduced plan (pipeline check, ~1 min)
```

Needs `pca11_A/`, `raw133_A/`, `det_A/` and `det_B/` under `--weights-dir` and the fp16 frame cache; about 5 min of
GPU time plus about 10 min of CPU bootstrap. Writes `robust.json` (split summaries for every pair plus the paired
bootstrap comparisons) and `per_image.pkl` (per-frame `BoxMetrics` stats, for further paired tests).
`robust.json` in this folder is the run the spec quotes.

**What is compared.** Test and val summaries of PCA-11 and raw133 at fixed checkpoints (`model_59` … `model_99`)
and at `model_best`, plus det_A `model_89` and `model_99` and det_B `model_best`. The metrics are `main_det.evaluate`'s
own: pooling the per-image stats gives exactly its split summary. Re-evaluating each run's stored checkpoint
reproduces its recorded test numbers (raw133 `model_best` 0.645 / 0.761, det_A `model_89` 0.444 / 0.563,
det_B `model_best` 0.356 / 0.521, pca11 `model_best` 0.620 / 0.746 AP50 / recall50). The 95 % CIs come from a
paired bootstrap over the 70 test frames (2000 resamples); several checkpoints of one run are averaged inside each
resample.

**Conditions** (applied inside the filter bank; the YOLO weights are unchanged):

| condition | meaning |
|---|---|
| `noise<dB>` | i.i.d. Gaussian read noise per bias-voltage reading (per band for raw), sigma = the channel's mean absorbed signal (`|R_V|^T mu`, training band mean in p99 units, i.e. auto-exposure) / 10^(dB/20). For PCA-11 the 344 per-voltage noises pass through its fixed 344 -> 11 map. The only measured device figure (ECHSE SI Fig. S7) is about 0.63 % per reading, about 44 dB, under 20 mW/cm² bench light. |
| `voff<mV>` | every bias voltage lands dv higher (hysteresis or drift): R(V + dv), linear interpolation within a dead-zone branch |
| `gain1` | fixed per-voltage gain error ~ N(0, 1 %), one draw (seed 0) |
| `device1` | the measured Device-1 responses (`R_Device1.mat`, 0.05 V grid, interpolated, per-voltage least-squares gain) instead of the EC_filterV3 interpolant |
| `<above>r` | the same perturbed device with each channel's mean and std re-measured on it (recalibrated standardisation) |
| `drop<a>_<b>` | PCA channels u_a … u_b set to their training mean (0) |

The noise seeds are fixed (`zlib.crc32` of the condition name and the frame id), so reruns are identical.
