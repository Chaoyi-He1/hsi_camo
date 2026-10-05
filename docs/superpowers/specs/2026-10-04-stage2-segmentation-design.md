# Stage 2 — camouflaged-object segmentation inside the Stage-1 ROIs — design

**Date:** 2026-10-04
**Status:** design approved in conversation section by section (goal, input arms, ROI handling, box sources, model families, components, protocol, error handling and testing). Awaiting review of this written spec; no implementation yet.
**Builds on:** `docs/superpowers/specs/2026-09-27-ec-yolo-detector-design.md` (Stage 1: loader, fp16 cache, `FilterBank`, ROI export, §12 results), `docs/superpowers/specs/2026-09-21-hypercod-dataloader-design.md`.

## 1. Goal

Segment the camouflaged object inside each region that a Stage-1 detector exports, and measure two things:

1. **Development metric:** mask quality inside a correct box (oracle GT boxes), at the ROI level and pasted back into the full frame.
2. **Headline result:** the whole sensor chain, input → its own detector's ROIs → segmentation → full-frame mask, compared across three input arms (raw hyperspectral bands vs electrochromic (EC) sensor readings).

Three model families are compared on each arm: **SAM2-UNet** (primary), a **box-prompted SAM2.1 with LoRA** (alternative family) and **ZoomNeXt-B2** (dedicated COD baseline).

## 2. Facts the design relies on (verified)

| Item | Value |
|---|---|
| Frames | 279 train (251 train + 28 val, `data_loader/splits/det_val_ids.json`) / 70 test, 1680 × 1240 (H × W), 133 bands 400–800 nm after the Stage-1 band window |
| Objects | 289 train+val objects, 71 test (`boxes_from_mask`, min area 100 px). Box height p50/p95/max 165/349/1290 px, width 89/216/445 px. Foreground is a median 0.36 % of the frame |
| Frame cache | `<data_path>/cache_fp16/<split>/<id>.npy`, raw (un-scaled) fp16, band-major, 0.55 GB per frame, read with O_DIRECT (`cube_cache.read_npy_direct`); per-frame p99 scale from the loader's csv. Partial reads through mmap stalled DataLoader workers in Stage 1 (spec §12) |
| Stage-1 ROI export | `main_det_rois.py` → `{id: {"rois": [[x1,y1,x2,y2,conf]...], "boxes": [...], "gt_boxes": [...]}}`, float native pixels, `rois` = `boxes` grown by `expand_box(box, 1.5, 256, H, W)` and clipped, operating point conf ≥ 0.02, top 5. Existing files come from det_B and are retired (17 of 70 test frames have no ROI) |
| Detectors per arm (noise-free, spec §12) | raw: `raw133_A` (test AP50 0.645); EC-10: `sel10g_clean_A`, greedy-10 voltages 1.75 −0.44 1.36 0.48 −0.65 −0.36 1.70 0.01 −0.86 1.45 V, `--pca-channels 10` (0.743); EC-24: `sel24g_clean_A`, greedy-24 voltages, `--pca-channels 24` (finishing 2026-10-04) |
| Crop-cache size (measured from GT boxes) | windows of box × 2.0, ≥ 512 px, 133 bands fp16: 22.0 GB for train+val (median 70 MB per object); × 2.5 / ≥ 640 px would be 34 GB |
| Hardware | 2 × RTX A4500 (20 GB); 125 GB RAM (≈ 72 GB free for the page cache); one full-resolution job per machine |
| COD state of the art (research workflow 2026-10-04, numbers from each paper's own table) | COD10K-test S-measure: SAM2-UNeXT .924, BiRefNet .913, SPEGNet .908 (1024 px) / .890 (512 px), ZoomNeXt-B4 .898, SAM2-UNet (Hiera-L) .880. Unverified or unusable: SALT-DINOv3 .932 (not checked), SAM3-Adapter .927 (released code fails at import, no weights) |
| HyperCOD benchmark (AAAI 2026, arXiv 2601.03736, Table 2; full frames, 280 train) | SAM2-UNet best S .805 / E .899 / MAE .0022 / adaptive F .641; the paper's HSC-SAM S .802 (no code released); HGINet S .766 |
| Box-prompted SAM2 (arXiv 2412.01240, Table 4) | zero-shot with GT boxes on COD10K, SAM2-L: S .911, weighted F .902 (not trained) |
| Memory at 512 × 512 (probes by the research workflow on an A4500, to be re-measured in pre-flight) | SAM2-UNet Hiera-L, 133 channels, batch 8 bf16: 12.0 GiB, 0.41 s/step; SAM2.1-L, frozen trunk, batch 8: 10.4 GiB; ZoomNeXt-B2, 133 channels, batch 2: 6.2 GB |
| Licences | SAM2 and SAM2-UNet: Apache-2.0. ZoomNeXt: no licence file (all rights reserved). PySODMetrics: MIT |
| Metric labels | SAM2-UNet/UNeXT report *adaptive* F, ZoomNeXt reports *max* E. Numbers from different papers are not comparable column by column |

## 3. Decisions

- **Input arms.** Three arms, all noise-free like the current Stage-1 study (`--read_noise_db` remains available for later):
  - `raw`: 133 bands, standardised with the training band statistics (as `raw133_A`).
  - `ec10`: the greedy-10 voltages, whitened (`--pca-channels 10`), exactly the `sel10g_clean_A` front end.
  - `ec24`: the greedy-24 voltages, whitened (`--pca-channels 24`), exactly the `sel24g_clean_A` front end.
  Each arm is a complete sensor chain: at evaluation it uses its own detector's ROIs.
- **ROI handling.** Cut the ROI at native resolution and place it on a 512 × 512 canvas: centred and zero-padded when its longer side is ≤ 512 px, otherwise downscaled (aspect kept) to fit and then padded. The model gets N arm channels plus one **box channel** (1 inside the pre-expansion box, 0 elsewhere and on padding).
- **Training boxes: a mix of three sources** (per training item):
  - **50 % expanded GT boxes.** Each side of the GT box is moved *outward* by an independent U(0, 0.15) × (box side), never inward, so the object is always fully inside; then the export rule `expand_box(·, 1.5, 256)` gives the ROI.
  - **40 % matched detector boxes.** The arm's own detector's `rois`/`boxes` on the training frames. A predicted ROI is matched to the object whose mask it covers most, if it covers ≥ 1 % of that mask. Target = the full-frame GT mask cropped to the ROI, so truncated objects are learned as truncated. If an object has no matched ROI, the item falls back to an expanded GT box.
  - **10 % false-positive detector boxes.** ROIs that cover < 1 % of every object mask; target = all zeros.
  Caveat (accepted): the detectors were trained on these frames, so their training-frame boxes are tighter than on test and truncation is under-represented. Follow-up if needed: detector predictions from held-out folds.
- **Crop cache instead of frame reads.** Training reads ~58 k crops per run; reading full frames each time would take ~24 h of disk time. A one-off `build_crop_cache` reads each train/val frame once (O_DIRECT) and stores:
  - per GT object, a window = the smallest rectangle containing the box × 2.0 (≥ 512 px, clipped to the frame) and every detector ROI matched to that object;
  - per false-positive ROI, a window = the ROI itself;
  - raw fp16 133-band data, the GT-mask crop, the frame's p99 scale and all box coordinates, plus one index JSON.
  Estimated 25–50 GB; the actual size is reported by the builder. All arms and models read the same windows; the arm front end runs on the GPU. Test evaluation does not use the cache: it reads each test frame once.
- **Front end = Stage 1's.** `FilterBank` construction is factored out of `build_ec_yolo` into `build_filter_bank(args, dataset)`, used by both stages, so Stage 2 sees bit-identical channels to its detector.
- **N-channel stem: regress-and-fold initialisation.** For each arm, fit by least squares (training pixels) a linear map from the arm's channels z to a pseudo-RGB render normalised with the ImageNet mean/std: `rgb_norm ≈ P z + q` (P: 3 × N, q: 3). The model's pretrained RGB stem `W_rgb [C_out, 3, k, k]` becomes `W_N[o, c] = Σ_k W_rgb[o, k] · P[k, c]` with bias `+ Σ_{k,i,j} W_rgb[o, k, i, j] · q_k`. At initialisation the network therefore sees the pseudo-RGB image; training adds spectral cues. The box channel's kernel slice starts at zero. Pseudo-RGB render: mean of the bands in [600, 700) nm (R), [500, 600) (G), [400, 500) (B) of the p99-scaled cube, clipped to [0, 1]. Same rule for all three arms. Fallback: mean RGB kernel × 3/N.
- **Models** (per-model settings in §7):
  - **SAM2-UNet** (vendored, Apache-2.0): Hiera-L trunk from SAM2 v1 `sam2_hiera_large.pt`, frozen; adapters, RFB modules and U-Net decoder trained; the trunk's patch-embed is the N+1-channel stem.
  - **SAM2.1 + box + LoRA** (`sam2` package, Apache-2.0): `sam2.1_hiera_large.pt`, image encoder frozen except LoRA (r = 8) on the attention projections, prompt encoder frozen, mask decoder trained; the pre-expansion box is the prompt; the image size is set to 512 (positional encodings recomputed — verified in pre-flight).
  - **ZoomNeXt-B2** (fetched at setup into git-ignored `third_party/zoomnext/`, never committed, research use): PVTv2-B2 with its COD checkpoint, full fine-tune; `patch_embed1` (frozen in the original repo) becomes the trainable N+1-channel stem.
- **Controls.** SAM2-UNet trained on the 3-channel pseudo-RGB render (arm `rgb`: does spectral input help at all?); zero-shot SAM2.1 with box prompts on the pseudo-RGB render. Both are trained and matched with the raw arm's boxes (`raw133_A`) and evaluated at all three levels of §8.
- **Splits.** Train on the 251 train frames; the 28 val frames select checkpoints; the 70 test frames are used only for the final report.
- **Code style.** The user's conventions (memory `feedback-match-coding-style`): `main_*.py` with `get_args_parser()` and `main(args)`, new flags snake_case with `type=` and `help=`, `import torch.utils.data as Dataset`, shape comments, `assert cond, f"..."`, module-level collate fn, `cfg/*.yaml`, `bash_files/`.

## 4. Code layout

```
main_det_rois.py                   (changed) rebuild any detector from its checkpoint's own args (as main_det_compare does);
                                   --split train|val|test; output results/det/rois_<run>_<split>.json
main_seg.py                        (new) train + ROI-level evaluation of one (model, arm, seed); get_args_parser(); main(args)
main_seg_eval.py                   (new) end-to-end test evaluation: oracle and detector ROIs, paste-back, full-frame metrics
cfg/seg.yaml                       (new) lr, epochs, batch_size, box_mix, jitter, canvas, loss weights, eval thresholds
data_loader/roi_crops.py           (new) build_crop_cache(...), class HyperCOD_roi(Dataset.Dataset), seg_collate_fn, match_rois
models/filter_bank.py              (changed) + build_filter_bank(args, dataset) factored out of models/ec_yolo.build_ec_yolo
models/ec_yolo.py                  (changed) build_ec_yolo calls build_filter_bank; behaviour unchanged
models/seg_stem.py                 (new) pseudo_rgb(cube, wavelens), fit_rgb_map(z, rgb), fold_stem(conv, P, q, n_extra=1)
models/seg_models.py               (new) build_seg_model(args, n_in) -> SAM2UNetSeg | SAM2BoxSeg | ZoomNeXtSeg (common forward/loss API)
train_eval/train_eval_seg.py       (new) train_one_epoch, evaluate (ROI level), paste_back
train_eval/seg_metrics.py          (new) SegMetrics wrapper around py_sod_metrics; size buckets; FP-ROI false-mask rate
third_party/sam2_unet/             (new, vendored, Apache-2.0, LICENSE + NOTICE of changes)
third_party/zoomnext/              (git-ignored, fetched by bash_files/setup_third_party.sh at a pinned commit)
bash_files/setup_third_party.sh    (new) pip deps, ZoomNeXt clone, checkpoints into weights/pretrained/
bash_files/launch_rois_all.sh      (new) ROI export of raw133_A, sel10g_clean_A, sel24g_clean_A for train/val/test
bash_files/launch_seg_queue.sh     (new) crop cache once, then the (model, arm, seed) runs in sequence, then main_seg_eval
tests/test_roi_crops.py, tests/test_seg_stem.py, tests/test_seg_models.py, tests/test_seg_metrics.py   (new)
weights/seg_<model>_<arm>_s<seed>/ model_best, model_last, results_<name>.txt   (git-ignored)
results/seg/<name>/                eval_roi_oracle.json, eval_full_oracle.json, eval_full_det.json, per_image.pkl (git-ignored)
```

New dependencies: `sam2` (facebookresearch, Apache-2.0) with `hydra-core` 1.3.2, `pysodmetrics` (MIT). Installed by `setup_third_party.sh` into the `hsi_camo` env; compatibility with torch 2.11 is checked in pre-flight.

## 5. Components

### 5.1 `main_det_rois.py` and `bash_files/launch_rois_all.sh`
Rebuild the detector from `ckpt['args']` (arm, voltages, `--pca-channels`, `--raw-bands`) the way `main_det_compare.load_checkpoint_model` does, so any run can export. `--split` accepts `train`, `val` (the 28 held-out ids) and `test`; train and val are written separately. Output name `rois_<run>_<split>.json`, schema unchanged. Operating point unchanged (cfg `roi_conf` 0.02, `roi_topk` 5).

### 5.2 `data_loader/roi_crops.py`
- `match_rois(rois, gt_mask, labels, ids)` → per ROI the matched object id (max mask coverage ≥ 1 %) or −1 (false positive).
- `build_crop_cache(data_path, splits=('train', 'val'), roi_files={arm: path}, out_dir, grow=2.0, min_side=512)`: one O_DIRECT read per frame; writes `<out_dir>/<id>_<k>.npy` (fp16 [133, h, w], un-scaled) and `<out_dir>/index.json` with per window: frame id, split, window xyxy, p99 scale, GT-mask crop file, object boxes, and per arm the ROIs inside it (`roi`, `box`, `conf`, `matched_object` or −1). Prints the total size.
- `class HyperCOD_roi(Dataset.Dataset)`: `__init__(cache_dir, split, arm, box_mix=(0.5, 0.4, 0.1), gt_jitter=0.15, canvas=512, scale_aug=(0.75, 1.25), train=True, seed=None)`; an epoch has one item per training object plus false-positive items amounting to 10 % of the epoch (drawn uniformly from all false-positive ROIs of the arm's detector); each object item uses an expanded GT box with probability 5/9 and its matched detector box with probability 4/9, which gives the 50 / 40 / 10 mix overall. `__getitem__` → `(img [133, 512, 512] fp16 p99-scaled, mask [1, 512, 512] float32, box_map [1, 512, 512] float32, meta)`, where `meta` holds the ROI in frame coordinates, the canvas placement (offset, scale) and the box source. Augmentation in training: horizontal/vertical flips, 90° rotations, scale 0.75–1.25 of the ROI content before placement, per-channel gain ±5 %. Validation items are deterministic: oracle expanded GT ROIs plus the val detector's matched ROIs.
- `seg_collate_fn(batch)` stacks images, masks and box maps, keeps `meta` as a list.

### 5.3 `models/seg_stem.py`
`pseudo_rgb(x, wavelens)` (band-window means, §3); `fit_rgb_map(z, rgb)` → `(P, q)` by least squares on ≤ 2 M sampled training pixels; `fold_stem(conv, P, q, n_extra=1)` → a new `Conv2d(N + n_extra, ...)` initialised as in §3 (box-channel slice zero).

### 5.4 `models/seg_models.py`
`build_seg_model(args, n_in)` returns a module with `forward(x, box_map, box_xyxy)` → logits `[B, 1, 512, 512]` (deep-supervision outputs as a list in training) and `loss(outputs, mask)`:
- `SAM2UNetSeg`: the vendored SAM2-UNet with the Hiera patch-embed replaced by `fold_stem`; trunk frozen.
- `SAM2BoxSeg`: SAM2.1 image encoder (patch-embed folded, LoRA r = 8 on attention q/k/v/proj), prompt encoder fed `box_xyxy` in canvas coordinates, mask decoder trained; the 128 × 128 low-resolution mask is bilinearly upsampled to 512.
- `ZoomNeXtSeg`: ZoomNeXt-B2 from `third_party/zoomnext` with `patch_embed1` folded and unfrozen; its multi-scale input triplet built internally from the 512 canvas.

### 5.5 `train_eval/train_eval_seg.py` and `train_eval/seg_metrics.py`
`train_one_epoch(model, loader, optimizer, device, epoch, scaler, accumulate, logger)` (bf16 autocast, gradient clipping, MetricLogger, same structure as `train_eval_det`); `evaluate(model, loader, device, logger, epoch, tag)` at the ROI level; `paste_back(prob_canvas, meta, H, W)` → full-frame probability map (resize the valid canvas region to the ROI, place it, merge overlapping ROIs by maximum). `SegMetrics` wraps `py_sod_metrics`: S-measure (α = 0.5), E-measure (mean, max, adaptive), weighted F (β = 1), adaptive F, MAE, and IoU of the prediction binarised at 0.5 (computed directly, since PySODMetrics min-max normalises predictions); per object-size bucket by GT mask area (small < 2,000 px, medium 2,000–20,000, large > 20,000); for false-positive ROIs the false-mask rate (share with any predicted pixel > 0.5), since S/E are undefined for an empty GT.

### 5.6 `main_seg.py` and `main_seg_eval.py`
- `main_seg.py`: flags (snake_case) `--seg_model sam2unet|sam2box|zoomnext`, `--arm raw|ec10|ec24|rgb`, `--crop_cache`, `--box_mix`, `--seed`, `--epochs`, `--batch_size`, `--accumulate`, `--lr`, `--stem_lr`, `--output_dir`, `--eval`; the arm's front-end flags are taken from its detector checkpoint (`--det_ckpt`) so they cannot diverge. Saves `model_best` (val mean of S and weighted F) and `model_last`; one JSON line per epoch in `results_<name>.txt`.
- `main_seg_eval.py`: one pass over the 70 test frames (each read once); for each run evaluates (a) ROI level with oracle ROIs, (b) full frame with oracle ROIs, (c) full frame with the arm's detector ROIs (`rois_<det>_test.json`); writes the JSON files of §4; paired frame bootstrap (2000 resamples) between runs, reusing `main_det_compare.bootstrap`.

## 6. Data flow

```
train:  index.json + <id>_<k>.npy (raw 133 bands)  --HyperCOD_roi-->  ROI on 512 canvas [133] + mask + box_map
        --GPU: build_filter_bank (raw: standardise | ec10/ec24: readings -> whitening)-->  [N, 512, 512]
        --concat box_map--> [N+1, 512, 512] --model--> logits --loss vs mask
test:   frame (O_DIRECT, once) -> for each oracle ROI and each detector ROI: cut, canvas, front end, model
        -> probability canvas -> paste_back -> full-frame map [1680, 1240] -> SegMetrics (ROI and full frame)
```

## 7. Training procedure

- **Runs:** 3 models × 3 arms × 3 seeds (0, 1, 2) = 27, plus the 2 controls. About 1 h each at 512 × 512; sequential at first, one per GPU later only if the crop cache is verified to stay in the page cache.
- **Common:** canvas 512, batch 8 (ZoomNeXt: 4 with accumulation 2), bf16 autocast, AdamW (weight decay 1e-4), cosine schedule, 200 epochs (≈ 7 k steps), gradient clipping 1.0, TrainLogger to TensorBoard and W&B (project `hsi_camo`, run name `seg_<model>_<arm>_s<seed>`).
- **SAM2-UNet:** lr 1e-3 for adapters, RFB and decoder, `--stem_lr` 1e-4 for the folded stem; loss = weighted BCE + weighted IoU on the three deep-supervision outputs (the SAM2-UNet structure loss).
- **SAM2.1 + box:** lr 1e-4 for LoRA, stem and mask decoder; loss = Dice + BCE on the upsampled mask.
- **ZoomNeXt-B2:** lr 1e-4 for the whole network, its own loss.
- **Checkpoint choice:** best val mean of S-measure and weighted F over the deterministic val items (oracle ROIs + matched val detector ROIs); `model_last` is reported too.

## 8. Evaluation protocol

On the 70 test frames, per run:
- (a) **ROI level, oracle:** the GT box expanded by the export rule (× 1.5, ≥ 256 px), no jitter; metrics on the ROI crop at native size.
- (b) **Full frame, oracle:** the masks of (a) pasted into 1680 × 1240.
- (c) **Full frame, end to end:** the arm's own detector ROIs at its operating point, merged by maximum; frames without any ROI give an empty mask (counted as misses). Also reported with the HyperCOD paper's metric set (MAE, mean E, S, adaptive F) for comparison with its Table 2.
- Statistics: mean ± std over the 3 seeds; paired frame bootstrap CIs for arm-vs-arm and model-vs-model differences; object-size buckets throughout; false-mask rate on the detector's false-positive ROIs.

## 9. Error handling

- A checkpoint whose arm or voltage set differs from `--arm` / `--det_ckpt`, or an ROI file from a different detector than the arm's, raises an error.
- Every ROI must lie inside its cached window (assert with the offending ids); ROIs at the frame edge are clipped and the canvas is zero-padded. A missing window raises, never skips.
- Frames with no detector ROI → empty full-frame mask, counted. False-positive ROIs are excluded from ROI-level S/E (undefined for an empty GT) and reported through the false-mask rate; they count in the full-frame metrics.
- Pre-flight (`main_seg.py --dry_run`): builds the model, loads one batch per box source, checks channel counts and shapes, prints the peak GPU memory; verifies `sam2` at image size 512 and the torch 2.11 compatibility of the new dependencies.
- The queue skips a run whose results file has a final line, so it can be relaunched after a crash.

## 10. Testing (synthetic fixture, CPU, no network or real data)

- `test_roi_crops.py`: every cached window contains its ROIs; expanded-GT jitter never cuts the object (mask fully inside the box for 1000 draws); matched-ROI targets equal the GT mask cropped to the ROI; false-positive targets are all zero; canvas placement and `paste_back` round-trip a mask exactly (no resize) and to within interpolation error (resize); the box mix is respected over an epoch.
- `test_seg_stem.py`: `fold_stem` with a fitted (P, q) reproduces the RGB stem's response to the pseudo-RGB image (to float tolerance, away from the border); the box-channel slice is zero; `fit_rgb_map` recovers a known linear map.
- `test_seg_models.py`: each model builds and runs forward/backward at N = 3, 10 and 24 with small configurations (tiny Hiera/PVT stubs where the real weights would need the network).
- `test_seg_metrics.py`: `SegMetrics` matches `py_sod_metrics` on hand-made masks; IoU at 0.5 and the size buckets are correct; false-mask rate.
- `build_filter_bank` gives identical outputs to the Stage-1 construction for the same arguments, and Stage-1 checkpoints still load (`test_filter_bank.py`, `test_read_noise.py` extended).
- The existing suite (130 passed, 1 skipped) stays green.

## 11. Open items and risks

- **Optimistic training-frame detector boxes** (§3); follow-up: held-out-fold detector predictions.
- **SAM2 at 512 px:** SAM2 is trained at 1024; Hiera's windowed positional embeddings interpolate, but box-prompted SAM2.1 quality at 512 is unverified. Fallback: 1024 canvas for `SAM2BoxSeg` only (memory re-measured).
- **Frozen RGB trunk on spectral stems:** the folded stem starts in-distribution, but spectral features beyond the pseudo-RGB render must pass through a frozen trunk. Ablation if SAM2-UNet underperforms: LoRA (r = 8) on Hiera stages 3–4.
- **Crop-cache size** with three detectors' ROIs may exceed the free page cache; the builder reports the size, and false-positive windows shared by several detectors are stored once.
- **ZoomNeXt licence:** research use only, never redistributed.
- **Read noise:** this study is noise-free; the 40 dB results of Stage 1 (spec §12) show the voltage choice and the model behaviour change under noise, so a noisy repeat (`--read_noise_db_range`) is a follow-up once the device's SNR is measured.

## 12. Out of scope

Separating multiple objects within one ROI (the target is the binary union of all GT objects inside the ROI); joint training of detector and segmenter; video; on-device deployment and speed optimisation; new voltage selection for segmentation (the arms reuse the detectors' voltages); SAM3, SALT-DINOv3 and other models with unverified numbers or unusable code.

## References

- HyperCOD benchmark: arXiv 2601.03736 (Table 2). SAM2-UNet: arXiv 2408.08870, github.com/WZH0120/SAM2-UNet. SAM2-UNeXT: arXiv 2508.03566. SAM2: github.com/facebookresearch/sam2. Box-prompted SAM2 on COD: arXiv 2412.01240 (Table 4). ZoomNeXt: arXiv 2310.20208, github.com/lartpang/ZoomNeXt. RGB-aware stem initialisation for multispectral COD: MSFormer, arXiv 2608.30355. Metrics: github.com/lartpang/PySODMetrics.
- Research workflow of 2026-10-04 (23 agents, 18 candidates verified against paper tables and repos): summary in the session; the shortlist and refuted claims are reflected in §2.
