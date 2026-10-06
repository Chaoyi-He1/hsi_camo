# Stage 2 — Camouflaged-Object Segmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Segment the camouflaged object inside every Stage-1 detector ROI and compare three input arms (raw 133 bands, 10 and 24 whitened EC readings) end to end, with three model families (SAM2-UNet, box-prompted SAM2.1 + LoRA, ZoomNeXt-B2) and two RGB controls.
**Architecture:** Each arm's front end is rebuilt bit-identically from its Stage-1 detector checkpoint (`build_filter_bank`), detector ROIs are exported for train/val/test, and a one-off crop cache of native-resolution windows feeds ROI items on a 512 canvas (arm channels + box channel) into segmenters whose pretrained RGB stems are folded to N+1 channels by a least-squares pseudo-RGB map. `main_seg.py` trains one (model, arm, seed) and selects on val mean(S, Fw); `main_seg_eval.py` scores oracle ROIs, oracle paste-back and each arm's own detector ROIs on the 70 test frames with seed statistics and paired frame-bootstrap CIs.
**Tech Stack:** Python 3 in the `hsi_camo` conda env: torch 2.11.0+cu128, numpy 2.5.2, opencv 5.0.0 (all unchanged), ultralytics 8.4 (Stage-1 YOLO26), sam2 @ `2b90b9f5ceec907a1c18123530e92e794ad901a4` (dist `SAM-2` 1.0) with hydra-core 1.3.2 / omegaconf 2.3.0 / antlr4-python3-runtime 4.9.3 / iopath 0.1.10 / portalocker 4.4.0, vendored SAM2-UNet @ `01598e5e9912ffb23f965ecbebf4d1dfecbaa56e` (Apache-2.0), ZoomNeXt @ `614af4348808734aaf2ec43937cf827410330b67` (git-ignored clone) with einops 0.8.1 and timm 1.0.30, pysodmetrics 1.6.2 (`--no-deps`) with scikit-image 0.26.0 and scikit-learn 1.9.1, h5py, scipy, PIL, pytest.
**Spec:** `docs/superpowers/specs/2026-10-04-stage2-segmentation-design.md`

## Global Constraints

- Frames 1680 × 1240 (H × W), 133 bands 400–800 nm after the Stage-1 band window; splits 251 train + 28 val (data_loader/splits/det_val_ids.json) / 70 test; val selects checkpoints, test only for the final report.
- Objects = boxes_from_mask components with min area 100 px (289 train+val, 71 test).
- Frame cache: <data_path>/cache_fp16/<split>/<id>.npy, raw (un-scaled) fp16 band-major [133, H, W], read with O_DIRECT (cube_cache.read_npy_direct); per-frame p99 scale from the loader's csv; no partial mmap reads in DataLoader workers.
- Arm detectors (noise-free): raw → raw133_A; ec10 → sel10g_clean_A (greedy-10 voltages 1.75 −0.44 1.36 0.48 −0.65 −0.36 1.70 0.01 −0.86 1.45 V, --pca-channels 10); ec24 → sel24g_clean_A (greedy-24 voltages, --pca-channels 24); the rgb control and the zero-shot control use raw133_A's boxes.
- All arms noise-free (--read_noise_db stays available, unused).
- Front end = Stage 1's: build_filter_bank(args, dataset) shared by build_ec_yolo and Stage 2, bit-identical channels; build_ec_yolo behaviour unchanged.
- ROI export: rois = boxes grown by expand_box(box, 1.5, 256, H, W) and clipped; operating point cfg roi_conf 0.02, roi_topk 5; --split train|val|test; output results/det/rois_<run>_<split>.json with the schema unchanged {id: {rois, boxes, gt_boxes}}.
- Canvas 512 × 512: the ROI at native resolution, centred and zero-padded if its longer side ≤ 512 px, else downscaled (aspect kept) to fit, then padded; input = N arm channels + 1 box channel (1 inside the pre-expansion box, 0 elsewhere and on padding).
- Training box mix 50 / 40 / 10: expanded GT (each side moved OUTWARD by U(0, 0.15) × box side, then expand_box(·, 1.5, 256)) with probability 5/9 per object item, matched detector ROI with 4/9 (fallback to expanded GT), and false-positive items = 10 % of the epoch drawn uniformly from the arm's FP ROIs.
- ROI matching: an ROI matches the object whose mask it covers most if it covers ≥ 1 % of that mask; otherwise it is a false positive (target all zeros); matched target = the full-frame GT mask cropped to the ROI.
- Crop cache: per GT object a window = the smallest rectangle containing box × 2.0 (≥ 512 px, clipped to the frame) and every matched ROI; per FP ROI the ROI itself (shared FP ROIs stored once); fp16 133-band un-scaled + GT crop + p99 scale + index.json; one O_DIRECT read per frame; estimated 25–50 GB, size reported; test evaluation never uses the cache.
- Training augmentation: horizontal/vertical flips, 90° rotations, scale 0.75–1.25 of the ROI content before placement, per-channel gain ±5 %; validation items deterministic (oracle expanded GT ROIs + matched val detector ROIs).
- Stem regress-and-fold: rgb_norm ≈ P z + q (P 3 × N, q 3) by least squares on ≤ 2 M training pixels; W_N[o,c] = Σ_k W_rgb[o,k] P[k,c], bias += Σ_{k,i,j} W_rgb[o,k,i,j] q_k; box-channel slice zero; pseudo-RGB = mean of p99-scaled bands in [600,700) R, [500,600) G, [400,500) B, clipped to [0,1], ImageNet mean/std; fallback mean RGB kernel × 3/N.
- SAM2-UNet: vendored (Apache-2.0, LICENSE + NOTICE), Hiera-L trunk from SAM2 v1 sam2_hiera_large.pt frozen; adapters, RFB and U-Net decoder trained; patch-embed = the N+1-channel stem.
- SAM2.1 + box: sam2.1_hiera_large.pt, image encoder frozen except LoRA r = 8 on attention projections, prompt encoder frozen, mask decoder trained, prompt = pre-expansion box in canvas px, image size 512, 128 × 128 low-res mask bilinearly upsampled to 512.
- ZoomNeXt-B2: PVTv2-B2 COD checkpoint, full fine-tune, patch_embed1 = trainable N+1-channel stem, multi-scale triplet built from the 512 canvas; fetched into git-ignored third_party/zoomnext/, never committed (no licence, research use only).
- Training: 3 models × 3 arms × seeds 0, 1, 2 = 27 runs + 2 controls; canvas 512, batch 8 (ZoomNeXt 4 × accumulation 2), bf16 autocast, AdamW weight decay 1e-4, cosine schedule, 200 epochs, gradient clipping 1.0; sequential, one full-resolution job per machine.
- Per-model optimisation: SAM2-UNet lr 1e-3 (adapters/RFB/decoder), --stem_lr 1e-4, structure loss (weighted BCE + weighted IoU) on the 3 deep-supervision outputs; SAM2.1 + box lr 1e-4 for LoRA, stem and mask decoder, Dice + BCE; ZoomNeXt lr 1e-4 for the whole network, its own loss.
- Checkpoint selection: best val mean of S-measure and weighted F over the deterministic val items; model_last reported too; one JSON line per epoch in results_<name>.txt plus a final line; the queue skips runs whose results file has a final line.
- Logging: TrainLogger to TensorBoard and W&B project hsi_camo, run name seg_<model>_<arm>_s<seed>; weights in weights/seg_<model>_<arm>_s<seed>/ (model_best, model_last, results_<name>.txt).
- Metrics (SegMetrics on py_sod_metrics): S-measure α = 0.5, E-measure mean/max/adaptive, weighted F β = 1, adaptive F, MAE, IoU of the prediction binarised at 0.5 computed directly; size buckets by GT area small < 2,000 px, medium 2,000–20,000, large > 20,000; FP ROIs give the false-mask rate (share with any pixel > 0.5) and are excluded from ROI-level S/E.
- Evaluation on the 70 test frames per run: (a) ROI level with oracle ROIs expand_box(gt, 1.5, 256), no jitter, metrics at native size; (b) (a) pasted into 1680 × 1240, overlaps merged by maximum; (c) the arm's own detector ROIs (rois_<det>_test.json) merged by maximum, frames without ROI = empty mask counted as misses, also with the paper set MAE, mean E, S, adaptive F; mean ± std over 3 seeds; paired frame bootstrap 2000 resamples for arm-vs-arm and model-vs-model; outputs results/seg/<name>/eval_roi_oracle.json, eval_full_oracle.json, eval_full_det.json, per_image.pkl.
- Error handling: a checkpoint whose arm or voltage set differs from --arm/--det_ckpt, or an ROI file from another detector, raises; every ROI lies inside its cached window (assert with ids); a missing window raises; ROIs at the frame edge are clipped and the canvas zero-padded.
- Pre-flight main_seg.py --dry_run: builds the model, one batch per box source, checks channel counts and shapes, prints peak GPU memory, verifies sam2 at image size 512 and torch 2.11 compatibility.
- Dependencies: sam2 (Apache-2.0) with hydra-core 1.3.2 and pysodmetrics (MIT) installed by bash_files/setup_third_party.sh into the hsi_camo env; checkpoints into weights/pretrained/ (plan pins from recon: sam2 @ 2b90b9f5ceec907a1c18123530e92e794ad901a4, SAM2-UNet @ 01598e5e9912ffb23f965ecbebf4d1dfecbaa56e, ZoomNeXt @ 614af4348808734aaf2ec43937cf827410330b67, pysodmetrics 1.6.2 --no-deps; numpy 2.5.2 / cv2 5.0.0 / torch 2.11.0+cu128 unchanged).
- Tests: synthetic fixture, CPU, no network or real data; the existing suite (130 passed, 1 skipped) stays green.
- Code style: main_*.py with get_args_parser() and main(args); new flags snake_case with type= and help=; import torch.utils.data as Dataset; class X(Dataset.Dataset); assert cond, f"..."; shape comments; module-level collate fn; ''' docstrings; cfg/*.yaml; bash_files/.
- Push rule: after each task's suite is green: commit on worktree-det, `git push origin worktree-det`, `git push origin worktree-det:master` (fast-forward only; never force-push master).
- One full-resolution job per machine: no ROI export, crop-cache build, training run or test evaluation runs next to another one on the same box.

## Review Focus

- **(Task 12)** ZoomNeXt-style models that need >= 2 items per training batch (pooled BatchNorm): an epoch whose last batch would hold 1 item, and a --dry_run box source with a single item (the fp source with few fp ROIs). Test: `test_training_never_sees_a_batch_of_one` (Task 12, Steps 5-6).
- **(Task 12)** The crop cache / training ROIs come from a different detector than the arm's (spec §9: an ROI file from a different detector raises). Test: `test_crop_cache_built_from_another_detector_raises` (Task 12, Steps 5-6).
- **(Task 4)** ROIs that contain several objects (target = union, obj_area vs the object's own area) and ROIs clipped at the frame corner (smaller than roi_min, zero-padded canvas). Test: `test_roi_with_two_objects_and_a_frame_corner_roi` (Task 4, Steps 5-6).
- **(Task 13)** Train/test parity of the ROI item: main_seg_eval.make_item (cut from the full p99-scaled test frame) vs HyperCOD_roi's deterministic val item (cut from the crop cache), at s = 1 and at s < 1. Test: `test_test_items_match_the_validation_items` (Task 13, Steps 5-6).
- **(Task 3, training half in Task 4)** Crop-cache build and training items through DataLoader workers (num_workers > 0: CropFrames pickling, parallel np.save, per-worker RNG of HyperCOD_roi); all other unit tests use num_workers=0 while every real run uses 4-6. Tests: `test_build_crop_cache_with_workers_matches_serial` (Task 3, Steps 5-6) and `test_training_items_through_dataloader_workers` (Task 4, Steps 5-6; it needs HyperCOD_roi, which Task 3 does not have yet).

---

## File Structure

| File | Task | Responsibility |
|---|---|---|
| `models/filter_bank.py` (modify) | 1 | gains `pca_whitening`, `pca_whitened_channels`, `read_noise_std` (moved) and `build_filter_bank(args, dataset)`: the one builder of an arm's front end |
| `models/ec_yolo.py` (modify) | 1 | `build_ec_yolo` calls `build_filter_bank`; re-exports the moved helpers; behaviour bit-identical |
| `tests/test_filter_bank.py` (modify) | 1 | `build_filter_bank` reproduces the Stage-1 construction in every branch; ec_yolo re-exports |
| `main_det_rois.py` (rewrite) | 2 | rebuild any detector from its checkpoint (`detector_args`, `load_detector`), `--split train|val|test`, `rois_<run>_<split>.json` |
| `tests/test_main_det_rois.py` (modify) | 2 | checkpoint-driven rebuild, splits, file names |
| `bash_files/launch_rois_all.sh` (new) | 2 | exports raw133_A / sel10g_clean_A / sel24g_clean_A for train, val, test (sequential, detached) |
| `bash_files/launch_rois.sh` (delete) | 2 | retired det_B export |
| `bash_files/README.md` (modify) | 2, 7, 14 | rows for launch_rois_all.sh, setup_third_party.sh (+ header line), launch_seg_queue.sh |
| `data_loader/roi_crops.py` (new) | 3, 4 | ROI matching, crop-cache build (index.json + windows), canvas geometry, `HyperCOD_roi`, `seg_collate_fn` |
| `tests/test_roi_crops.py` (new) | 3, 4 | windows, matching, cache files, canvas round trips, jitter, box mix, fp items, multi-object / corner ROIs, DataLoader workers |
| `train_eval/seg_metrics.py` (new) | 5 | `SegMetrics` on py_sod_metrics (raw-probability and min-max paper protocol), `pool`, `bootstrap_seg` |
| `tests/test_seg_metrics.py` (new) | 5 | equality with py_sod_metrics, IoU, buckets, fp rate, pooling, bootstrap |
| `models/seg_stem.py` (new) | 6 | `pseudo_rgb`, `imagenet_normalize`, `fit_rgb_map`, `fold_stem`, `mean_stem` |
| `tests/test_seg_stem.py` (new) | 6 | band windows, pseudo-RGB, least squares, stem fold exactness |
| `third_party/__init__.py`, `third_party/sam2_unet/{__init__.py, sam2unet.py, LICENSE, NOTICE}` (new) | 7 | vendored SAM2-UNet (Apache-2.0) with documented changes |
| `bash_files/setup_third_party.sh` (new) | 7 | pinned sam2 / hydra / pysodmetrics / einops installs, ZoomNeXt clone, checkpoints, CPU verification |
| `.gitignore` (modify) | 7 | `third_party/zoomnext/` |
| `tests/test_third_party.py` (new) | 7 | licence / notice / no shadowing sam2 copy; structure loss |
| `models/seg_models.py` (new) | 8, 9, 10 | `ArmFrontEnd` / `build_front_end`, `SAM2UNetSeg`, `SAM2BoxSeg`, `ZoomNeXtSeg`, `SEG_MODELS`, `build_seg_model` |
| `tests/test_seg_models.py` (new) | 8 | front ends, SAM2UNetSeg |
| `tests/test_seg_sam2box.py` (new) | 9 | LoRA, Dice+BCE, SAM2BoxSeg |
| `tests/test_seg_zoomnext.py` (new) | 10 | UAL ramp, ZoomNeXtSeg, checkpoint load then fold |
| `train_eval/train_eval_seg.py` (new) | 11 | `seg_inputs`, `train_one_epoch`, `evaluate` (ROI level), `paste_back`, `roi_panel` |
| `tests/test_train_eval_seg.py` (new) | 11 | inputs, paste-back, accumulation, NaN stop, evaluation routing |
| `main_seg.py`, `cfg/seg.yaml` (new) | 12 | train / resume / eval / dry-run one (model, arm, seed); per-model cfg sections |
| `tests/test_main_seg.py` (new) | 12 | cfg merge, detector args, arm and ROI-file checks, dry run, train/resume/eval, no batch of one |
| `main_seg_eval.py` (new) | 13 | test evaluation levels (a)-(c), zero-shot control, per-run files, compare.json |
| `tests/test_main_seg_eval.py` (new) | 13 | levels and compare, downscaled ROIs, zero-shot, missing export, train/test item parity |
| `bash_files/launch_seg_queue.sh` (new) | 14 | crop cache once, 27 runs + rgb control, then main_seg_eval |
| `tests/test_launch_seg_queue.py` (new) | 14 | syntax and the planned run list |
| `docs/superpowers/plans/2026-10-04-stage2-segmentation.md` (this file) | all | the plan; the Results section is filled with the operational outcomes |

## Deviations from the Spec and Notes

These are deliberate and are recorded here so that the spec does not have to be re-read against the code:

- **Renamed / superseded signatures.** Spec §5.2 `match_rois(rois, gt_mask, labels, ids)` is `match_rois(rois, labels, ids, min_cover=0.01)`; the per-ROI index field `matched_object` is `object`. Spec §5.4 `build_seg_model(args, n_in)` / `forward(x, box_map, box_xyxy)` are `build_seg_model(args, n_in, P, q)` / `forward(x, box_xyxy)` (the box map is the last channel of `x`). Spec §5.5 `paste_back(prob_canvas, meta, H, W)` is `paste_back(prob_canvas, meta, out)` (max-merge into a caller-owned frame map). `build_crop_cache(data_path, out_dir, roi_files {arm: {split: path}}, splits, …)` replaces spec §5.2 `(data_path, splits, roi_files {arm: path}, out_dir, …)`; `HyperCOD_roi.__getitem__` returns a 5-tuple with `valid` where the spec has a 4-tuple; `train_one_epoch` / `evaluate` take `front_end` (spec §5.5 has no such argument); `detector_args` lets this run's data paths / device win over the checkpoint's (spec §5.1 says "the way `main_det_compare.load_checkpoint_model` does").
- **Loss weights and thresholds are not in `cfg/seg.yaml`.** Spec §4 lists loss-weight and eval-threshold keys; there are none: the losses are fixed per model, and the 0.5 thresholds (IoU, false-mask rate) are fixed in `train_eval/seg_metrics.py`.
- **Bootstrap.** Spec §5.6 "reusing `main_det_compare.bootstrap`": replaced by `train_eval.seg_metrics.bootstrap_seg`, because `main_det_compare.bootstrap` is hard-wired to BoxMetrics keys and pooling. It is a paired frame (cluster) bootstrap.
- **Deviations from upstream recipes.** SAM2-UNet: AdamW weight decay 1e-4 (upstream 5e-4), 512 canvases (upstream 352), the structure loss with `reduction='none'` (upstream's `reduce='none'` silently computes the plain mean BCE; `legacy_bce=True` restores it; see the vendored NOTICE), biases / norms / stem without decay. ZoomNeXt: one lr 1e-4 for the whole network (`encoder_lr_mult` 1.0; upstream 0.1 for the encoder and Adam without decay), `patch_embed1` trained (upstream frozen). SAM2.1 + box: the IoU and object-score heads frozen, no object-score gating, `apply_postprocessing=False`. SAM2BoxSeg and ZoomNeXtSeg apply AdamW weight decay 1e-4 to biases and norm weights too (only their stem group has decay 0), unlike SAM2UNetSeg's decay / no_decay split and Stage-1's optimizer; spec §7 fixes only "AdamW weight decay 1e-4", and ZoomNeXt's own recipe uses weight decay 0.
- **Third-party installs.** Task 5 hand-installs pysodmetrics 1.6.2 (+ scikit-image 0.26.0 / scikit-learn 1.9.1) before Task 7's `setup_third_party.sh`, which repeats the same pins idempotently; `pip check` reports pysodmetrics' `numpy<2.3.5`, `opencv-python-headless` and `scikit-image<0.26` pins as unmet (expected: numpy stays 2.5.2).
- **timm.** `setup_third_party.sh` also installs `timm==1.0.30` (ZoomNeXt's model code imports it; the plan's Tech Stack lists it but the original script did not install it).
- **Frozen Stage-1 oracle in the tests.** `tests/test_filter_bank.py::_stage1_filter_bank` is an intentional frozen copy of the pre-refactor `ec_yolo.py` logic at commit 1a14827 (the bit-identity oracle of spec §10); never edit it together with `build_filter_bank`.
- **Stem fallback** (spec §3, mean RGB kernel × 3/N): `models/seg_stem.mean_stem` exists and is tested, but no flag selects it; switching a model to it is a manual code change if the fold misbehaves.
- **1024 canvas fallback for SAM2BoxSeg** (spec §11): `--canvas 1024` works in `main_seg.py` (the crop-cache windows are native resolution), but `main_seg_eval.py` requires one canvas per call: evaluate 1024 runs in a separate call with their own `--out_dir`; their compare.json does not cross-compare with the 512 runs.
- **Size buckets.** At ROI level the bucket is the GT pixel count of the scored crop (the union of the GT inside the ROI); at full-frame levels (b) and (c) it is the GT pixel count of the whole frame (the union of all objects), not per object. `meta['area']` (the object's own area) is carried by `HyperCOD_roi` for a later per-object bucketing.
- **Tests.** Spec §10 "`test_read_noise.py` extended": the noise branches (floor, relative, whitening with noise and SNR range) are covered by the parametrised cases added to `tests/test_filter_bank.py`; `test_read_noise.py` is unchanged and green through the re-export. Model tests are split over `test_seg_models.py`, `test_seg_sam2box.py`, `test_seg_zoomnext.py`.
- **NaN in results files.** `results_<name>.txt` and the eval JSONs can hold `NaN` (`fp_false_mask_rate` without fp items, empty size buckets): Python's json reads it back, strict JSON parsers do not.
- **Test frames are read more than once.** The spec says each test frame is read once in total; `main_seg_eval.py` reads them `ceil(n_runs / runs_per_pass)` times (5 passes for the full 29-run queue at the default `--runs_per_pass 6`, about 12 min of extra reading at about 3 min per pass), because a pass keeps its models on the GPU. `--runs_per_pass` >= n_runs restores a single read, at the price of the GPU memory of all models at once.
- **Out of scope (follow-ups):** LoRA (r = 8) on Hiera stages 3–4 for SAM2-UNet (spec §11 ablation), held-out-fold detector boxes, noisy repeats.

### Task 1: Shared front-end builder `build_filter_bank` (refactor out of `build_ec_yolo`)

**Files:**
- Modify: `models/filter_bank.py:18,25` (docstring references), append after line 132 (moved helpers + new `build_filter_bank`)
- Modify: `models/ec_yolo.py:13` (import), delete `models/ec_yolo.py:154-213` (moved helpers), replace `models/ec_yolo.py:216-261` (FilterBank block of `build_ec_yolo`)
- Test: `tests/test_filter_bank.py` (extended)

**Interfaces:**
- Consumes: `FilterBank(R, channel_mean, channel_std, weight_vector, init_logits, noise_std, proj, eval_seed, train_scale_range)` (unchanged); `HyperCOD_data._band_stats()`, `.filter_bank_tensors()`, `.candidate_filter_matrix()`, `.n_bands`, `.wavelens` (unchanged).
- Produces:
  - `models.filter_bank.pca_whitening(R, mean, std, band_cov, k, noise_var=None) -> (V [N, k], lam [k])` (moved, unchanged)
  - `models.filter_bank.pca_whitened_channels(R, mean, std, band_cov, k, noise_var=None) -> (R' [n_bands, k] f32, mean' [k] f32, std' [k] f32, volts = 1..k f64)` (moved, unchanged)
  - `models.filter_bank.read_noise_std(R, mu, cov, snr_db, model='floor', R_ref=None) -> [N] f32` (moved, unchanged)
  - `models.filter_bank.build_filter_bank(args, dataset) -> (fb: FilterBank, volts: np.ndarray [N])`. `args` = detector flags (main_det namespace or a checkpoint's args). Every flag except `session` is read through `getattr` with its Stage-1 default. `dataset` = a `HyperCOD_data` built with `main_det.dataset_kwargs(args)`. `volts` = the voltages, the band centres in nm for `raw_bands`, or `1..K` for whitened channels. **For ec10 and ec24, `volts` is therefore `1..K`. The real voltages are in `det_args.filter_voltages`.**
  - `models.ec_yolo` re-exports `FilterBank, build_filter_bank, pca_whitening, pca_whitened_channels, read_noise_std`, so `from models.ec_yolo import read_noise_std` (main_select_voltages, tests, docs/reports scripts) keeps working. `build_ec_yolo(args, dataset)` now calls `build_filter_bank` and behaves bit-identically.

- [ ] **Step 1: Write the failing test**

In `tests/test_filter_bank.py`, replace the import block (lines 1-7) with:

```python
import os
import types
import numpy as np
import torch
import pytest

import models.ec_yolo as ec_yolo
from models.filter_bank import FilterBank, build_filter_bank, pca_whitening, pca_whitened_channels, read_noise_std
from data_loader.my_dataset import HyperCOD_data
```

Append at the end of `tests/test_filter_bank.py`:

```python


def _fb_ds(root):
    return HyperCOD_data(str(root), split='train', use_filter=False, norm='p99', crop_size=0, num_filters=8, filter_norm='l1', out_dtype='float16')


def _stage1_filter_bank(args, dataset):
    '''
    The FilterBank block of models/ec_yolo.build_ec_yolo at 1a14827, verbatim (minus the print): build_filter_bank must
    reproduce it bit for bit, since Stage 2 relies on seeing exactly the channels its detector was trained on.
    '''
    noise_db = float(getattr(args, 'read_noise_db', 0.0) or 0.0)
    noise_model = getattr(args, 'read_noise_model', None) or 'floor'
    mu, cov = dataset._band_stats()
    noise_std = proj = None
    if getattr(args, 'raw_bands', False):
        R = np.eye(dataset.n_bands, dtype=np.float32)
        mean, std = mu.astype(np.float32), np.sqrt(np.maximum(np.diag(cov), 0.0)).astype(np.float32)
        volts, weight_vector = dataset.wavelens.copy(), False
        if noise_db > 0:
            noise_std = read_noise_std(R, mu, cov, noise_db, noise_model)
    else:
        R, mean, std, volts = dataset.filter_bank_tensors()
        weight_vector = (args.session == 'A') and not getattr(args, 'no_gate', False)
        if noise_db > 0:
            R_all, _ = dataset.candidate_filter_matrix()
            noise_std = read_noise_std(R, mu, cov, noise_db, noise_model, R_ref=R_all)
        k = int(getattr(args, 'pca_channels', 0) or 0)
        if k > 0:
            if noise_std is None:
                R, mean, std, volts = pca_whitened_channels(R, mean, std, cov, k)
            else:
                V, lam = pca_whitening(R, mean, std, cov, k, noise_var=(noise_std / std) ** 2)
                proj, volts = (V / np.sqrt(lam)).astype(np.float32), np.arange(1, k + 1, dtype=np.float64)
            weight_vector = False
    scale_range = None
    db_range = getattr(args, 'read_noise_db_range', None)
    if noise_std is not None and db_range:
        lo_db, hi_db = sorted(float(v) for v in db_range)
        scale_range = (10 ** ((noise_db - hi_db) / 20), 10 ** ((noise_db - lo_db) / 20))
    fb = FilterBank(R, mean, std, weight_vector=weight_vector, noise_std=noise_std, proj=proj, eval_seed=int(getattr(args, 'seed', 0) or 0),
                    train_scale_range=scale_range)
    return fb, volts


# every branch of the construction: raw bands (+ noise), gated / ungated responses, session B (no gate), folded PCA,
# noise-regularised PCA with SNR augmentation, relative read noise on the gated bank
FB_CASES = {
    'raw': dict(raw_bands=True),
    'raw_noise': dict(raw_bands=True, read_noise_db=40.0),
    'gate': dict(),
    'no_gate': dict(no_gate=True),
    'session_b': dict(session='B'),
    'pca_folded': dict(pca_channels=3),
    'pca_noise_range': dict(pca_channels=3, read_noise_db=40.0, read_noise_model='floor', read_noise_db_range=[34.0, 50.0], seed=7),
    'gate_relative_noise': dict(read_noise_db=30.0, read_noise_model='relative', seed=3),
}


@pytest.mark.parametrize('case', sorted(FB_CASES))
def test_build_filter_bank_reproduces_the_stage1_construction(synthetic_root, case):
    root, _, _ = synthetic_root
    ds = _fb_ds(root)
    # a sparse namespace, as the Stage-1 tests and older checkpoints' args have: missing flags keep their getattr defaults
    args = types.SimpleNamespace(**{'session': 'A', **FB_CASES[case]})
    fb, volts = build_filter_bank(args, ds)
    ref, ref_volts = _stage1_filter_bank(args, ds)
    sd, sd_ref = fb.state_dict(), ref.state_dict()
    assert sd.keys() == sd_ref.keys() and all(torch.equal(sd[k], sd_ref[k]) for k in sd)    # same buffers, bit for bit
    assert (fb.n_readings, fb.n_channels, fb.weight_vector, fb.eval_seed, fb.train_scale_range) == \
           (ref.n_readings, ref.n_channels, ref.weight_vector, ref.eval_seed, ref.train_scale_range)
    np.testing.assert_array_equal(np.asarray(volts), np.asarray(ref_volts))
    x = torch.from_numpy(ds[0][0])[None].float()                                           # [1, 133, H, W]
    fb.eval(); ref.eval()                                                                  # eval: the reseeded noise sequence
    assert torch.equal(fb(x), ref(x))


def test_ec_yolo_reexports_and_builds_through_build_filter_bank(synthetic_root, monkeypatch):
    # the moved helpers stay importable from models.ec_yolo (tests, main_select_voltages, docs/reports scripts)
    assert ec_yolo.pca_whitening is pca_whitening and ec_yolo.pca_whitened_channels is pca_whitened_channels
    assert ec_yolo.read_noise_std is read_noise_std and ec_yolo.build_filter_bank is build_filter_bank and ec_yolo.FilterBank is FilterBank
    root, _, _ = synthetic_root
    ds = _fb_ds(root)
    args = types.SimpleNamespace(yolo_variant='yolo26n', pretrained='none', gate_entropy_weight=0.05, contain_weight=1.0, epochs=1,
                                 session='A', pca_channels=3)
    calls, real = [], ec_yolo.build_filter_bank
    monkeypatch.setattr(ec_yolo, 'build_filter_bank', lambda a, d: (calls.append(a), real(a, d))[1])
    m = ec_yolo.build_ec_yolo(args, ds)
    fb, volts = real(args, ds)
    assert len(calls) == 1 and calls[0] is args                                            # one front end, built by the shared builder
    assert m.filter_bank.n_channels == 3 and m.yolo.model[0].conv.weight.shape[1] == 3
    assert all(torch.equal(v, fb.state_dict()[k]) for k, v in m.filter_bank.state_dict().items())
    np.testing.assert_array_equal(m.selected_voltages, np.asarray(volts))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python -m pytest tests/test_filter_bank.py -v`
Expected: FAIL. Collection stops with `ImportError: cannot import name 'build_filter_bank' from 'models.filter_bank'`.

- [ ] **Step 3: Write minimal implementation**

3a. In `models/filter_bank.py`, the FilterBank docstring points at the new location of the helpers:
- line 18: `(ec_yolo.read_noise_std: a floor of eps * s0 for every reading)` becomes `(read_noise_std below: a floor of eps * s0 for every reading)`.
- line 25: `u = proj^T z (ec_yolo.pca_whitening with` becomes `u = proj^T z (pca_whitening below, with`.

3b. Append to the end of `models/filter_bank.py`, after `FilterBank.forward`. The three helpers are moved verbatim from `models/ec_yolo.py:154-213`, with one docstring word changed (`build_ec_yolo` becomes `build_filter_bank`). `build_filter_bank` is `models/ec_yolo.py:222-261` verbatim, plus `return fb, volts`. The module already imports `numpy as np`.

```python


def pca_whitening(R, mean, std, band_cov, k, noise_var=None):
    '''
    Eigen-decomposition of the covariance of the standardised readings z = (R^T x - mean) / std under the training band
    covariance: the top-k eigenvectors V [N, k] and eigenvalues lam [k], so u = V^T z / sqrt(lam) are k uncorrelated
    unit-variance channels. noise_var [N] (per-reading read-noise variance in standardised units, (sigma / std)^2) is
    added to the diagonal first: the directions are then ordered by signal-plus-noise variance and 1/sqrt(lam) never
    amplifies a noise-dominated direction beyond unit output noise (a whitening fitted to noise-free statistics gains up
    to ~1600x on the weakest of 344 responses and breaks on a real readout, see spec §12).
    '''
    R = np.asarray(R, np.float64); mean = np.asarray(mean, np.float64); std = np.asarray(std, np.float64)
    assert 1 <= k <= R.shape[1], f"pca_channels={k} must be in [1, {R.shape[1]}]"
    Dinv = 1.0 / std                                                                  # [N]
    C = (R * Dinv).T @ np.asarray(band_cov, np.float64) @ (R * Dinv)                  # [N, N] covariance of z
    if noise_var is not None:
        nv = np.asarray(noise_var, np.float64).reshape(-1)
        assert nv.shape == (R.shape[1],) and (nv >= 0).all(), f"noise_var must be {R.shape[1]} non-negative variances"
        C = C + np.diag(nv)
    evals, evecs = np.linalg.eigh(C)
    order = np.argsort(evals)[::-1][:k]
    return evecs[:, order], np.maximum(evals[order], 1e-12)                           # [N, k], [k]


def pca_whitened_channels(R, mean, std, band_cov, k, noise_var=None):
    '''
    Top-k principal directions of the standardised responses z = (R^T x - mean) / std under the training band covariance,
    returned as an equivalent (R', mean', std') so the FilterBank stays a linear projection + standardisation:
    u = V^T z = (R D^-1 V)^T x - V^T D^-1 mean, with std' = sqrt(eigenvalues) so every output channel has unit variance.
    Why: the 344 EC responses span an ~11-dimensional subspace (adjacent voltages 0.9998 cosine-similar), so feeding them
    all gives the first conv a near-singular input; k whitened directions keep the information and fix the conditioning.
    noise_var is passed through to pca_whitening (the folded form cannot carry per-reading noise itself: build_filter_bank
    keeps the readings explicit and hands FilterBank the map as `proj` when read noise is simulated).
    '''
    R = np.asarray(R, np.float64); mean = np.asarray(mean, np.float64); std = np.asarray(std, np.float64)
    V, lam = pca_whitening(R, mean, std, band_cov, k, noise_var)
    Dinv = 1.0 / std                                                                  # [N]
    R_new = (R * Dinv) @ V                                                            # [n_bands, k]
    mean_new = V.T @ (Dinv * mean)                                                    # [k]
    std_new = np.sqrt(lam)                                                            # [k]
    return R_new.astype(np.float32), mean_new.astype(np.float32), std_new.astype(np.float32), np.arange(1, k + 1, dtype=np.float64)


def read_noise_std(R, mu, cov, snr_db, model='floor', R_ref=None):
    '''
    Per-reading Gaussian read-noise std [N], in reading units, for the readings y = R^T x (x in p99 units) at snr_db:
      floor     one absolute floor for every reading, sigma = s0 / 10^(dB/20), s0 = median over the reference readings
                of their RMS value sqrt(R_v^T (Sigma + mu mu^T) R_v). R_ref defaults to R; pass the whole usable bank so
                the floor is a property of the device and not of the voltages chosen. Weak readings get the worst SNR,
                as under a real read-noise floor; the voltage selection (main_select_voltages) scores sets under it.
      relative  every reading at the same SNR, sigma_v = |R_v|^T mu / 10^(dB/20) (per-reading auto-exposure; the
                'noise<dB>' condition of docs/reports/2026-09-29-pca11-robustness/robust_eval.py).
    '''
    R = np.asarray(R, np.float64); mu = np.asarray(mu, np.float64); cov = np.asarray(cov, np.float64)
    eps = 10.0 ** (-float(snr_db) / 20.0)
    if model == 'floor':
        Rr = R if R_ref is None else np.asarray(R_ref, np.float64)
        rms = np.sqrt(np.maximum(np.einsum('bn,bc,cn->n', Rr, cov + np.outer(mu, mu), Rr), 0.0))   # [N_ref]
        return np.full(R.shape[1], eps * float(np.median(rms)), dtype=np.float32)
    if model == 'relative':
        return (eps * (np.abs(R).T @ mu)).astype(np.float32)                                      # [N]
    raise ValueError(f"unknown read-noise model '{model}', expected floor | relative")


def build_filter_bank(args, dataset):
    '''
    The input front end of a detector arm, factored out of models.ec_yolo.build_ec_yolo so that Stage 1 (the detector)
    and Stage 2 (models.seg_models.build_front_end) build it from the same flags and see bit-identical channels.
      args     the detector's flags (main_det.get_args_parser(), or a checkpoint's ckpt['args']). Every flag except
               `session` is read through getattr with its Stage-1 default, so older checkpoints (raw133_A has no
               pca_channels / read_noise_* / no_gate) and the tests' sparse namespaces keep working:
                 raw_bands            identity over the cube bands, standardised with the training band statistics
                 (otherwise)          the dataset's selected EC responses (filter_select / filter_voltages / num_filters);
                                      session A carries the trainable weight vector unless no_gate
                 pca_channels K       the top-K whitened principal directions of the responses; no weight vector
                 read_noise_db        per-reading read noise inside the bank (read_noise_std, model read_noise_model)
                 read_noise_db_range  SNR augmentation of the training noise level
                 seed                 the eval-mode noise seed
      dataset  a HyperCOD_data built with main_det.dataset_kwargs(args): band statistics, filter matrices, wavelengths
    Returns (fb, volts): the FilterBank and np.ndarray [N] of what each channel is (voltages; band centres in nm for
    raw_bands; 1..K for whitened channels), which build_ec_yolo stores as model.selected_voltages.
    '''
    noise_db = float(getattr(args, 'read_noise_db', 0.0) or 0.0)
    noise_model = getattr(args, 'read_noise_model', None) or 'floor'
    mu, cov = dataset._band_stats()                                                      # [n_bands], [n_bands, n_bands]
    noise_std = proj = None
    if getattr(args, 'raw_bands', False):
        # control run: the raw cube bands inside band_range go straight into the model (identity projection, standardised
        # with the training band statistics, no weight vector) to compare against the EC filter responses
        R = np.eye(dataset.n_bands, dtype=np.float32)
        mean, std = mu.astype(np.float32), np.sqrt(np.maximum(np.diag(cov), 0.0)).astype(np.float32)
        volts, weight_vector = dataset.wavelens.copy(), False                            # 'voltages' = band centres (nm)
        if noise_db > 0:
            noise_std = read_noise_std(R, mu, cov, noise_db, noise_model)                # per band, floor over the bands
    else:
        R, mean, std, volts = dataset.filter_bank_tensors()
        weight_vector = (args.session == 'A') and not getattr(args, 'no_gate', False)
        if noise_db > 0:
            R_all, _ = dataset.candidate_filter_matrix()                                 # every usable voltage: the floor is the device's
            noise_std = read_noise_std(R, mu, cov, noise_db, noise_model, R_ref=R_all)
        k = int(getattr(args, 'pca_channels', 0) or 0)
        if k > 0:
            if noise_std is None:
                R, mean, std, volts = pca_whitened_channels(R, mean, std, cov, k)        # folded form, as the runs before read noise
            else:
                # the readings stay explicit so the noise lands on them before the whitening (FilterBank docstring)
                V, lam = pca_whitening(R, mean, std, cov, k, noise_var=(noise_std / std) ** 2)
                proj, volts = (V / np.sqrt(lam)).astype(np.float32), np.arange(1, k + 1, dtype=np.float64)
            weight_vector = False
    scale_range = None
    db_range = getattr(args, 'read_noise_db_range', None)
    if noise_std is not None and db_range:
        # SNR augmentation: the noise level seen in training varies log-uniformly between the two dB values (as multipliers
        # of the nominal sigma); the whitening and the evaluation keep the nominal --read_noise_db
        lo_db, hi_db = sorted(float(v) for v in db_range)
        scale_range = (10 ** ((noise_db - hi_db) / 20), 10 ** ((noise_db - lo_db) / 20))
    fb = FilterBank(R, mean, std, weight_vector=weight_vector, noise_std=noise_std, proj=proj, eval_seed=int(getattr(args, 'seed', 0) or 0),
                    train_scale_range=scale_range)
    if noise_std is not None:
        print(f"read noise {noise_db:g} dB ({noise_model}): sigma {noise_std.min():.3g}-{noise_std.max():.3g} per reading"
              + (f", whitening of {fb.n_readings} readings -> {fb.n_channels} channels regularised by it" if proj is not None else "")
              + (f", training level drawn from {lo_db:g}-{hi_db:g} dB per batch" if scale_range is not None else ""))
    return fb, volts
```

3c. In `models/ec_yolo.py`, line 13 `from models.filter_bank import FilterBank` becomes:

```python
from models.filter_bank import FilterBank, build_filter_bank, pca_whitening, pca_whitened_channels, read_noise_std
```

3d. Delete `models/ec_yolo.py:154-213`, which are the old `pca_whitening`, `pca_whitened_channels` and `read_noise_std` with their trailing blank lines. They now come from the import above.

3e. In `models/ec_yolo.py`, replace the old docstring and the FilterBank block of `build_ec_yolo` (old lines 216-261, from `def build_ec_yolo(args, dataset):` through the read-noise `print(...)`) with the code below. The rest of the function (`pretrained = ...` through `return model`) is unchanged:

```python
def build_ec_yolo(args, dataset):
    '''
    Session A: all selected voltages + weight vector; session B: fixed channels, no weight vector (initialised by slice_to_channels).
    The input front end (raw bands / EC responses, --pca-channels whitening, --read_noise_db read noise) comes from
    models.filter_bank.build_filter_bank, the builder Stage 2 shares so its segmenters see the detector's exact channels.
    '''
    fb, volts = build_filter_bank(args, dataset)
    pretrained = None if args.pretrained == 'none' else (download_pretrained(args.yolo_variant) if args.pretrained == 'auto' else args.pretrained)
    yolo, n_matched, n_total = build_detection_model(args.yolo_variant, fb.n_channels, pretrained, nc=1, epochs=args.epochs)
    print(f"{args.yolo_variant}: {fb.n_channels} input channels, pretrained tensors reused {n_matched}/{n_total}")
    model = ECYolo(fb, yolo, gate_entropy_weight=args.gate_entropy_weight, contain_weight=args.contain_weight)
    model.selected_voltages = np.asarray(volts)
    return model
```

`models/ec_yolo.py` goes from 318 to 217 lines. `math` and `np` are still used elsewhere in the file, so the imports stay.

- [ ] **Step 4: Run test to verify it passes**

Run: `OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python -m pytest tests/test_filter_bank.py -v`
Expected: PASS, `12 passed, 1 skipped`. The skip is the opt-in GPU test.

Then run the full suite: `OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python -m pytest tests/ -q`
Expected: `139 passed, 1 skipped`. That is 130 existing plus 9 new; test_read_noise.py, test_pca_channels.py, test_ec_yolo.py and test_raw_bands.py are unchanged and green through the re-export.

- [ ] **Step 5: Check against the three real Stage-2 detector checkpoints (CPU, read-only, ~1 min)**

Run from the worktree. Save as `/tmp/check_fb.py`, then run `CUDA_VISIBLE_DEVICES="" /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python /tmp/check_fb.py 2>&1 | grep -v "^Sensor response\|^Loading sensor"`:

```python
import argparse
import torch
import main_det
from data_loader.my_dataset import HyperCOD_data
from models.filter_bank import build_filter_bank

for run in ['raw133_A', 'sel10g_clean_A', 'sel24g_clean_A']:
    ck = torch.load(f'weights/{run}/model_best', map_location='cpu', weights_only=False)
    a = argparse.Namespace(**{**vars(main_det.get_args_parser().parse_args([])), **ck['args']})
    ds = HyperCOD_data(split='test', **main_det.dataset_kwargs(a))
    fb, volts = build_filter_bank(a, ds)
    ref = {k[len('filter_bank.'):]: v for k, v in ck['model'].items() if k.startswith('filter_bank.')}
    sd = fb.state_dict()
    print(run, sorted(sd) == sorted(ref), all(torch.equal(sd[k], ref[k]) for k in ref), fb.n_channels)
```

Expected (verified on 2026-10-04):
```
raw133_A True True 133
sel10g_clean_A True True 10
sel24g_clean_A True True 24
```

- [ ] **Step 6: Commit and push**

```bash
git add models/filter_bank.py models/ec_yolo.py tests/test_filter_bank.py
git commit -m "refactor(filter_bank): build_filter_bank(args, dataset) shared by the detector and Stage 2

Moves pca_whitening, pca_whitened_channels and read_noise_std into models/filter_bank.py (re-exported from ec_yolo)
so the Stage-2 segmenters rebuild their arm's front end from the detector flags, bit-identical to build_ec_yolo.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QB4DScDwiA2DkVpbDopffV"
git push origin worktree-det
git push origin worktree-det:master
```

---

### Task 2: ROI export from any detector checkpoint, `--split train|val|test`, and `bash_files/launch_rois_all.sh`

**Files:**
- Modify: `main_det_rois.py` (whole file, 75 lines, rewritten below)
- Create: `bash_files/launch_rois_all.sh`
- Delete: `bash_files/launch_rois.sh` (the det_B export is retired; its row in `bash_files/README.md` is replaced)
- Modify: `bash_files/README.md:11`
- Test: `tests/test_main_det_rois.py` (extended; one assert added to `test_export_rois_json`)

**Interfaces:**
- Consumes: `main_det.get_args_parser()`, `main_det.load_cfg(args)`, `main_det.dataset_kwargs(args)`, `main_det.build_model(args, dataset_train, ckpt)`, `data_loader.det_splits.load_det_ids(data_path, path=None) -> (train_ids, val_ids)`, `HyperCOD_data(split, ids, **kw)`, and `build_filter_bank` indirectly through `build_ec_yolo` (Task 1).
- Produces:
  - `main_det_rois.DET_MODEL_KEYS`: the tuple of flags that define a detector. Values come from the checkpoint, or from main_det's parser default when the checkpoint lacks a key.
  - `main_det_rois.detector_args(ckpt_args, args) -> argparse.Namespace`, built as main_det parser defaults < the caller's keys that main_det knows < `DET_MODEL_KEYS` from the checkpoint. Task 12 (`main_seg.det_args_from_ckpt`) reuses it, so the list of model-defining keys lives in one place.
  - `main_det_rois.load_detector(ckpt_path, args, device, dataset=None) -> (model, det_args, run_name)`.
    - `model` is in eval mode on `device`, and the weights load strictly.
    - `det_args.pretrained == 'none'`, and `load_cfg` has been applied.
    - `run_name = basename(dirname(ckpt_path))`.
    - `dataset=None` builds a test-split `HyperCOD_data` from `dataset_kwargs(det_args)`.
    - **For T8/T12/T13:** `det_args` is what `build_filter_bank(det_args, dataset)` needs, with `dataset = HyperCOD_data(split=..., **main_det.dataset_kwargs(det_args))`.
  - `main_det_rois.split_ids(args) -> (hsi_split, ids)`: `('test', None)`; `('train', train_ids)` (251 ids); or `('train', val_ids)` (28 ids, from `--split-file`).
  - The CLI writes `<out-dir>/rois_<run_name>_<split>.json` with the schema unchanged: `{id: {"rois": [[x1,y1,x2,y2,conf],...], "boxes": [[x1,y1,x2,y2,conf],...], "gt_boxes": [[x1,y1,x2,y2],...]}}`.
  - `bash_files/launch_rois_all.sh` writes `results/det/rois_{raw133_A,sel10g_clean_A,sel24g_clean_A}_{train,val,test}.json`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_main_det_rois.py`, replace the import block (lines 1-6) with:

```python
import os
import json
import argparse
import pytest
import yaml
import torch
import main_det, main_det_rois
from tests.test_main_det import _args
from data_loader.cube_cache import build_cube_cache, default_cache_dir
from data_loader.det_splits import load_det_ids
```

In `test_export_rois_json`, add this line directly after `assert len(seen) == 1 and seen[0]['cache_dir'] == default_cache_dir(str(root)) and seen[0]['split'] == 'test'`:

```python
    assert os.path.basename(path) == 'rois_det_A_test.json'                    # rois_<run>_<split>, run = the checkpoint's folder
```

Append at the end of `tests/test_main_det_rois.py`:

```python


def test_detector_args_take_the_model_from_the_checkpoint(tmp_path):
    # the CLI asks for other model flags than the run used, and for its own data / device / operating point
    cli = main_det_rois.get_args_parser().parse_args(
        ['--data-path', str(tmp_path / 'data'), '--device', 'cpu', '--num_workers', '0', '--roi-conf', '0.3', '--pca-channels', '5',
         '--read_noise_db', '40', '--filter-select', 'all', '--split', 'val'])
    ckpt_args = {'session': 'A', 'filter_select': 'manual', 'filter_voltages': [1.75, -0.44, 1.36], 'pca_channels': 3, 'seed': 42,
                 'yolo_variant': 'yolo26s', 'epochs': 100, 'hpy': 'cfg/det.yaml', 'gate_entropy_weight': 0.05, 'contain_weight': 1.0,
                 'data_path': '/old/data', 'cache_dir': '/old/cache', 'device': 'cuda', 'num_workers': 6, 'roi_conf': 0.02, 'roi_topk': 5}
    a = main_det_rois.detector_args(ckpt_args, cli)
    # the model is the checkpoint's ...
    assert (a.filter_select, a.filter_voltages, a.pca_channels, a.seed, a.yolo_variant) == ('manual', [1.75, -0.44, 1.36], 3, 42, 'yolo26s')
    # ... a flag an older checkpoint lacks (raw133_A has no read_noise_*) takes main_det's default, never the CLI value
    assert a.read_noise_db == 0.0 and a.read_noise_db_range is None and a.raw_bands is False and a.no_gate is False
    # ... while data, device and the operating point stay the CLI's (a moved cache or a CPU run is never overridden)
    assert (a.data_path, a.cache_dir, a.device, a.num_workers, a.roi_conf) == (str(tmp_path / 'data'), '', 'cpu', 0, 0.3)
    assert not hasattr(a, 'split') and a.roi_topk is None                     # export-only flags dropped; cfg fills the rest later
    # a Stage-2 namespace (main_seg's own flags, its own cfg) still yields the detector's cfg and parser defaults
    seg = argparse.Namespace(data_path=str(tmp_path / 'data'), device='cpu', hpy='cfg/seg.yaml', lr=1e-3, arm='ec10')
    b = main_det_rois.detector_args(ckpt_args, seg)
    assert b.hpy == 'cfg/det.yaml' and b.lr == 1e-3 and not hasattr(b, 'arm') and b.conf_thres is None and b.cache_dir == ''


def test_export_rebuilds_from_the_checkpoint_for_every_split(synthetic_root, tmp_path, monkeypatch):
    root, _, _ = synthetic_root
    monkeypatch.delenv('RANK', raising=False)
    build_cube_cache(str(root), 'train', num_workers=0); build_cube_cache(str(root), 'test', num_workers=0)
    run_dir = tmp_path / 'pca_run'
    # a whitened detector (6 uniform voltages -> 3 channels, yolo26n): nothing of it is repeated on the export CLI below
    main_det.main(_args(root, tmp_path, **{'--session': 'A', '--name': 'pca_run', '--output-dir': str(run_dir), '--pca-channels': '3'}))
    train_ids, val_ids = load_det_ids(str(root), str(tmp_path / 'val.json'))
    argv = ['--data-path', str(root), '--split-file', str(tmp_path / 'val.json'), '--device', 'cpu', '--amp', '--num_workers', '0',
            '--resume', str(run_dir / 'model_best'), '--out-dir', str(tmp_path / 'rois'), '--roi-min', '0', '--min-area', '10',
            '--roi-conf', '0.0']
    frames = {}
    for split in ['train', 'val', 'test']:
        args = main_det_rois.get_args_parser().parse_args(argv + ['--split', split])
        path = main_det_rois.main(args)
        assert path == str(tmp_path / 'rois' / f'rois_pca_run_{split}.json')
        with open(path) as f:
            data = json.load(f)
        frames[split] = set(data)
        for v in data.values():
            assert v['gt_boxes'] == [[20.0, 10.0, 26.0, 16.0]] and len(v['rois']) <= 5 and all(len(r) == 5 for r in v['rois'])
    # train = the detector's training ids only, val = the held-out ids of the split file, test = the test split
    assert frames == {'train': set(train_ids), 'val': set(val_ids), 'test': {'7'}} and not set(train_ids) & set(val_ids)
    model, det_args, run = main_det_rois.load_detector(str(run_dir / 'model_best'), args, torch.device('cpu'))
    assert run == 'pca_run' and (det_args.pca_channels, det_args.filter_select, det_args.num_filters, det_args.yolo_variant) == (3, 'uniform', 6, 'yolo26n')
    assert model.filter_bank.n_channels == 3 and not model.training and det_args.pretrained == 'none'
    ck = torch.load(run_dir / 'model_best', map_location='cpu', weights_only=False)
    assert all(torch.equal(v, ck['model'][k]) for k, v in model.state_dict().items())   # the checkpoint, loaded strictly
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python -m pytest tests/test_main_det_rois.py -v`
Expected: FAIL in three tests:
- `test_export_rois_json` fails with `AssertionError: assert 'rois_test.json' == 'rois_det_A_test.json'`.
- `test_detector_args_take_the_model_from_the_checkpoint` fails with an argparse error: `--split` has no choice `'val'`, so the parser raises SystemExit 2.
- `test_export_rebuilds_from_the_checkpoint_for_every_split` fails with `RuntimeError: Error(s) in loading state_dict for ECYolo`. Today the model is rebuilt from the CLI flags (yolo26s, all 344 voltages), not from the checkpoint (yolo26n, 3 whitened channels).

- [ ] **Step 3: Write minimal implementation**

Replace `main_det_rois.py` with:

```python
"""
Export candidate camouflage regions (ROIs) from a trained detector for the Stage-2 segmentation (spec Stage 2 §5.1).
The detector is rebuilt from its checkpoint's own args (raw bands / voltages / --pca-channels / read noise / YOLO
variant), so no model flag has to be repeated here and any run can export:
  python main_det_rois.py --resume weights/sel10g_clean_A/model_best --split train
  python main_det_rois.py --resume weights/sel10g_clean_A/model_best --split val
  python main_det_rois.py --resume weights/sel10g_clean_A/model_best --split test
--split train = the detector's 251 training ids, val = the 28 held-out ids of data_loader/splits/det_val_ids.json
(--split-file), test = the 70 test frames. Writes <out-dir>/rois_<run>_<split>.json, run = the checkpoint's folder name:
  {name: {"rois": [[x1,y1,x2,y2,conf],...], "boxes": [[x1,y1,x2,y2,conf],...], "gt_boxes": [[x1,y1,x2,y2],...]}}
(float native-frame pixels). The exported operating point is cfg/det.yaml's roi_conf (0.02) and roi_topk (5) -- the
same subset evaluate() reports its *_op metrics on -- expanded by roi_margin/roi_min and clipped. --max-rois overrides
roi_topk. Data path, cache, device and the operating point come from this command line, never from the checkpoint.
bash_files/launch_rois_all.sh exports the three Stage-2 detectors for all three splits.
"""
import os
if "RANK" not in os.environ and "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import json
from functools import partial
import torch

import util.misc as utils
import main_det
from data_loader.boxes import det_collate_fn, expand_box, boxes_from_mask
from data_loader.det_splits import load_det_ids
from data_loader.my_dataset import HyperCOD_data
from models.ec_yolo import decode_predictions
from train_eval.box_metrics import mask_coverage, filter_to_operating_point

# The flags that define a detector (its FilterBank and its YOLO): taken from the checkpoint's args, or, when an older
# checkpoint lacks one (raw133_A has no pca_channels / read_noise_* / no_gate), from main_det's parser default -- never
# from the caller's command line. hpy too: it is the detector's cfg (the operating point load_cfg fills from it), and a
# Stage-2 caller passes its own cfg/seg.yaml under the same name.
DET_MODEL_KEYS = ('session', 'top_k', 'raw_bands', 'pca_channels', 'no_gate', 'filter_select', 'filter_voltages', 'num_filters',
                  'filter_path', 'band_range', 'read_noise_db', 'read_noise_db_range', 'read_noise_model', 'seed', 'yolo_variant',
                  'gate_entropy_weight', 'contain_weight', 'epochs', 'hpy')


def get_args_parser():
    parser = argparse.ArgumentParser('Export detector ROIs', parents=[main_det.get_args_parser()], add_help=False)
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                        help="train: the detector's training ids; val: the held-out ids of --split-file; test: the test split")
    parser.add_argument('--out-dir', default='results/det')
    parser.add_argument('--max-rois', type=int, default=None, help='detections kept per frame; overrides cfg roi_topk')
    parser.set_defaults(wandb=False)
    return parser


def detector_args(ckpt_args, args):
    '''
    The namespace a detector is rebuilt with: main_det's parser defaults, overridden by the caller's flags that main_det
    knows (data path, cache, device, workers, operating point, split file), overridden by the model-defining flags
    (DET_MODEL_KEYS) from the checkpoint's args. Unlike main_det_compare.load_checkpoint_model's {**vars(args),
    **ckpt['args']}, a checkpoint never overrides where the data lives or which device runs (all three Stage-2 detectors
    store device 'cuda' and the cache of the day), and a CLI model flag never leaks into a checkpoint that lacks it.
    Keys main_det does not know (the export's --split, a Stage-2 caller's --arm) are dropped.
    '''
    defaults = vars(main_det.get_args_parser().parse_args([]))
    cli = {k: v for k, v in vars(args).items() if k in defaults}
    model = {k: ckpt_args.get(k, defaults[k]) for k in DET_MODEL_KEYS}
    return argparse.Namespace(**{**defaults, **cli, **model})


def load_detector(ckpt_path, args, device, dataset=None):
    '''
    Rebuild a trained detector from its checkpoint, in eval mode on device: (model, det_args, run_name).
      ckpt_path  e.g. weights/sel10g_clean_A/model_best; run_name = the checkpoint's folder name (sel10g_clean_A)
      args       the caller's namespace (main_det_rois / main_seg / main_seg_eval flags), see detector_args
      dataset    a HyperCOD_data built with main_det.dataset_kwargs(det_args) to take the filter matrices and band
                 statistics from (the export passes the one it reads its frames from, so no second dataset is built);
                 None builds one on the test split -- filter_bank_tensors() is split-independent, the band statistics
                 always come from the train split's stats file.
    det_args is also what models.filter_bank.build_filter_bank(det_args, dataset) needs to rebuild the arm's front end.
    '''
    assert os.path.isfile(ckpt_path), f"detector checkpoint {ckpt_path} not found"
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    det_args = detector_args(ckpt['args'], args)
    det_args.pretrained, det_args.resume, det_args.eval = 'none', ckpt_path, True       # the weights come from the checkpoint
    main_det.load_cfg(det_args)                                                        # fills only what is still None
    if dataset is None:
        dataset = HyperCOD_data(split='test', **main_det.dataset_kwargs(det_args))
    model = main_det.build_model(det_args, dataset, ckpt).to(device).eval()            # session A, or B sliced by its indices
    run_name = os.path.basename(os.path.dirname(os.path.abspath(ckpt_path)))
    return model, det_args, run_name


def split_ids(args):
    '''(HyperCOD directory split, ids or None) of --split: val frames live in the train directory.'''
    if args.split == 'test':
        return 'test', None
    train_ids, val_ids = load_det_ids(args.data_path, args.split_file or None)
    return 'train', (train_ids if args.split == 'train' else val_ids)


@torch.no_grad()
def main(args):
    utils.init_distributed_mode(args)
    main_det.load_cfg(args)
    assert args.resume, "--resume <detector checkpoint> is required, e.g. --resume weights/sel10g_clean_A/model_best"
    if args.max_rois is not None:
        args.roi_topk = args.max_rois
    # the export operating point is cfg roi_conf/roi_topk, the same one evaluate() reports its *_op metrics at
    print(f"ROI operating point: conf >= {args.roi_conf}, top {args.roi_topk} per frame")
    device = torch.device(args.device if args.device == 'cpu' or torch.cuda.is_available() else 'cpu')
    # One dataset only, built exactly like the training ones (same cache, same numeric path) but with the checkpoint's
    # filter settings (filter_select / filter_voltages / num_filters decide filter_bank_tensors()). It also hands
    # load_detector the filter matrices, so there is no need to build train/val/test just for those.
    det_args = detector_args(torch.load(args.resume, map_location='cpu', weights_only=False)['args'], args)
    hsi_split, ids = split_ids(args)
    dataset = HyperCOD_data(split=hsi_split, ids=ids, **main_det.dataset_kwargs(det_args))
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
                                         collate_fn=partial(det_collate_fn, min_area=args.min_area))
    model, det_args, run_name = load_detector(args.resume, args, device, dataset=dataset)
    print(f"{run_name}: {model.filter_bank.n_channels} input channels, {args.split} split, {len(dataset)} frames")
    out, covered, n_gt = {}, 0, 0
    for batch in loader:
        img = batch['img'].to(device)                                                  # [1, 133, H, W] fp16
        with torch.autocast(device.type, enabled=device.type == 'cuda' and args.amp):
            dets = decode_predictions(model(img), conf_thres=args.conf_thres, iou_thres=args.iou_thres,
                                      max_det=args.max_det, end2end=getattr(model.yolo, 'end2end', None))[0]
        H, W = img.shape[-2:]
        dets = filter_to_operating_point(dets, args.roi_conf, args.roi_topk)         # shared with BoxMetrics' *_op keys
        dets[:, [0, 2]] = dets[:, [0, 2]].clip(0, W); dets[:, [1, 3]] = dets[:, [1, 3]].clip(0, H)
        rois = [[*expand_box(d[:4], args.roi_margin, args.roi_min, H, W).tolist(), float(d[4])] for d in dets]
        gt_boxes, labels, cids = boxes_from_mask(batch['masks'][0], min_area=args.min_area, return_labels=True)
        for gb, cid in zip(gt_boxes, cids):
            n_gt += 1; covered += any(mask_coverage(labels == cid, r[:4]) >= 0.99 for r in rois)
        out[batch['names'][0]] = {'rois': rois, 'boxes': [[float(v) for v in d[:5]] for d in dets], 'gt_boxes': gt_boxes.tolist()}
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f'rois_{run_name}_{args.split}.json')
    with open(path, 'w') as f:
        json.dump(out, f)
    print(f"{run_name} {args.split}: {len(out)} frames, {n_gt} objects, ROI coverage recall@0.99 = {covered / max(n_gt, 1):.3f} -> {path}")
    return path


if __name__ == '__main__':
    main(get_args_parser().parse_args())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python -m pytest tests/test_main_det_rois.py -v`
Expected: PASS (6 passed, about 5 s).

Then run the full suite: `OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python -m pytest tests/ -q`
Expected: `141 passed, 1 skipped`. That is 130 existing, plus 9 from Task 1, plus 2 here.

- [ ] **Step 5: Check the three real detectors and the real splits (CPU, read-only, ~1 min)**

Save as `/tmp/check_rois.py` and run it from the worktree with `CUDA_VISIBLE_DEVICES="" /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python /tmp/check_rois.py 2>&1 | grep -v "^Sensor response\|^Loading sensor\|^Overriding\|pretrained tensors"`:

```python
import torch
import main_det_rois

args = main_det_rois.get_args_parser().parse_args(['--device', 'cpu', '--split', 'val'])
main_det_rois.main_det.load_cfg(args)
print('val', len(main_det_rois.split_ids(args)[1]))
args.split = 'train'; print('train', len(main_det_rois.split_ids(args)[1]))
for run in ['raw133_A', 'sel10g_clean_A', 'sel24g_clean_A']:
    model, a, name = main_det_rois.load_detector(f'weights/{run}/model_best', args, torch.device('cpu'))
    ck = torch.load(f'weights/{run}/model_best', map_location='cpu', weights_only=False)
    same = all(torch.equal(v, ck['model'][k]) for k, v in model.state_dict().items())
    print(name, a.yolo_variant, a.raw_bands, a.pca_channels, a.filter_select, model.filter_bank.n_channels, same, a.roi_conf, a.roi_topk, a.device)
```

Expected (verified on 2026-10-04):
```
val 28
train 251
raw133_A yolo26s True 0 all 133 True 0.02 5 cpu
sel10g_clean_A yolo26s False 10 manual 10 True 0.02 5 cpu
sel24g_clean_A yolo26s False 24 manual 24 True 0.02 5 cpu
```

- [ ] **Step 6: Write the export queue and the README row**

Create `bash_files/launch_rois_all.sh`:

```bash
#!/usr/bin/env bash
# Stage-2 ROI export (spec Stage 2 §5.1): the three arm detectors, each rebuilt from its own checkpoint args, on the
# train (251 ids), val (28 held-out ids) and test (70) frames, one after another on one GPU (~1-1.5 s/frame, ~25 min):
#   raw133_A        raw 133 bands                       -> arm raw
#   sel10g_clean_A  greedy-10 voltages, whitened (10)   -> arm ec10
#   sel24g_clean_A  greedy-24 voltages, whitened (24)   -> arm ec24
# Output results/det/rois_<run>_<split>.json (operating point = cfg/det.yaml roi_conf 0.02 / roi_topk 5).
#   bash bash_files/launch_rois_all.sh    # detached; queue log logs/rois_all.log, per export logs/rois_<run>_<split>.log
# An export whose json already exists is skipped (FORCE=1 redoes it), so the queue can be relaunched after a crash.
# RUNS, SPLITS, CKPT (model_best), NUM_WORKERS (6), PY and CUDA_VISIBLE_DEVICES (0) are overridable.
SELF=$(readlink -f "$0")
cd "$(dirname "$SELF")/.." || exit 1
PY=${PY:-/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
mkdir -p logs results/det
if [ -z "${ROIS_ALL_CHILD:-}" ]; then
  ROIS_ALL_CHILD=1 setsid nohup bash "$SELF" "$@" > logs/rois_all.log 2>&1 < /dev/null &
  echo "ROI export queue launched, pid $!, log logs/rois_all.log"
  exit 0
fi

RUNS=${RUNS:-raw133_A sel10g_clean_A sel24g_clean_A}
SPLITS=${SPLITS:-train val test}
CKPT=${CKPT:-model_best}

rc_all=0
for run in $RUNS; do
  if [ ! -f "weights/$run/$CKPT" ]; then echo "$(date) weights/$run/$CKPT missing, skipping $run"; rc_all=1; continue; fi
  for split in $SPLITS; do
    out="results/det/rois_${run}_${split}.json"
    if [ -f "$out" ] && [ -z "${FORCE:-}" ]; then echo "$(date) $out exists, skipping"; continue; fi
    echo "$(date) start $run $split"
    "$PY" -u main_det_rois.py --resume "weights/$run/$CKPT" --split "$split" --out-dir results/det \
        --num_workers "${NUM_WORKERS:-6}" > "logs/rois_${run}_${split}.log" 2>&1 < /dev/null
    rc=$?
    echo "$(date) end $run $split, exit $rc: $(tail -n 1 "logs/rois_${run}_${split}.log")"
    [ $rc -eq 0 ] || rc_all=1
  done
done
echo "$(date) ROI export queue done, exit $rc_all"
exit $rc_all
```

In `bash_files/README.md`, replace the line 11 row
`| \`launch_rois.sh\` | Stage-2 ROI export for test and train from \`B_CKPT\` (default \`weights/det_B/model_best\`) |`
with:

```markdown
| `launch_rois_all.sh` | Stage-2 ROI export: `raw133_A`, `sel10g_clean_A`, `sel24g_clean_A` (rebuilt from their checkpoint args) on train / val / test, sequential on GPU 0, detached; `results/det/rois_<run>_<split>.json`, existing files skipped (`FORCE=1` redoes) |
```

The retired `bash_files/launch_rois.sh` is removed with `git rm` in Step 7 (once only).

Check: `bash -n bash_files/launch_rois_all.sh && echo syntax ok` should print `syntax ok`. A foreground smoke run with a missing run, `ROIS_ALL_CHILD=1 RUNS=no_such_run bash bash_files/launch_rois_all.sh; echo "exit $?"`, should print `weights/no_such_run/model_best missing, skipping no_such_run`, then `ROI export queue done, exit 1`, then `exit 1`.

Operational launch (after the commit, GPU 0 idle; the export reads ~0.55 GB per frame from the cache): `bash bash_files/launch_rois_all.sh`, then `tail -f logs/rois_all.log`. Expect 9 files `results/det/rois_{raw133_A,sel10g_clean_A,sel24g_clean_A}_{train,val,test}.json` with 251 / 28 / 70 frames each. Each log's last line has the form `<run> <split>: <n> frames, <k> objects, ROI coverage recall@0.99 = ...`.

- [ ] **Step 7: Commit and push**

```bash
git add main_det_rois.py tests/test_main_det_rois.py bash_files/launch_rois_all.sh bash_files/README.md
git rm bash_files/launch_rois.sh
git commit -m "feat(det): ROI export rebuilds any detector from its checkpoint; --split train|val|test

load_detector/detector_args take the model flags from ckpt['args'] (data, device and operating point stay the CLI's),
so the three Stage-2 arm detectors export rois_<run>_<split>.json; launch_rois_all.sh replaces the det_B export.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QB4DScDwiA2DkVpbDopffV"
git push origin worktree-det
git push origin worktree-det:master
```

---

### Task 3: ROI matching and the crop cache (`match_rois`, `build_crop_cache`)

**Files:**
- Create: `data_loader/roi_crops.py`
- Test: `tests/test_roi_crops.py`

**Interfaces:**
- Consumes:
  - `data_loader.boxes.boxes_from_mask(mask, min_area=100, return_labels=False)` and `expand_box(box, margin, min_size, H, W)`
  - `data_loader.cube_cache.read_npy_direct(path)`, `cache_path(cache_dir, split, name)`, `default_cache_dir(data_path)`
  - `data_loader.det_splits.load_det_ids(data_path, path=None) -> (train_ids, val_ids)`
  - `data_loader.my_dataset.HyperCOD_data(...)`, used for `.scale`, `.load_gt`, `.H`, `.W`, `.n_bands`, `.cache_dir`, `.split` and `.img_name`
  - The ROI files of Task 2: `results/det/rois_<run>_<split>.json` = `{id: {"rois": [[x1,y1,x2,y2,conf],...], "boxes": [[x1,y1,x2,y2,conf],...], "gt_boxes": [...]}}`
- Produces:
  - Constants: `ARMS = ('raw', 'ec10', 'ec24')`, `SEG_ARMS = ARMS + ('rgb',)`, `INDEX_NAME = 'index.json'`.
  - `pixel_box(box, H, W) -> (x1, y1, x2, y2)` int: floor x1/y1, ceil x2/y2, clipped. This is the rounding rule of `mask_coverage`, and it is used for every integer window and ROI.
  - `inside(inner, outer) -> bool`
  - `match_rois(rois [K,4], labels [H,W] int32, ids list, min_cover=0.01) -> np.ndarray [K] int64`. Each entry is the object id with the largest mask coverage if that coverage is >= min_cover, else -1.
  - `frame_windows(gt [H,W] bool, frame_rois {arm: {"rois","boxes"}}, grow=2.0, min_side=512, min_area=100, min_cover=0.01) -> list of window dicts` (pure, no I/O)
  - `load_roi_file(path, names) -> {name: entry}`. It asserts that every frame is present.
  - `class CropFrames(Dataset.Dataset)` and the module-level `records_collate_fn(batch)`
  - `build_crop_cache(data_path, out_dir, roi_files, splits=('train', 'val'), grow=2.0, min_side=512, num_workers=4, min_area=100, min_cover=0.01, split_file=None, cache_dir=None, band_range=(400.0, 800.0)) -> {"n_windows", "bytes", "n_object", "n_fp"}`
  - Files written:
    - `<out_dir>/<id>_<k>.npy`: fp16 [133, h, w], un-scaled, row-major (y, x)
    - `<out_dir>/<id>_<k>_gt.npy`: bool [h, w], the raw `load_gt` crop
    - `<out_dir>/index.json`, written last and atomically:
      ```
      {"band_range", "frame_hw", "grow", "min_side", "min_area", "min_cover", "arms", "roi_files",
       "windows": [{"file", "gt_file", "frame", "split", "kind", "window", "scale", "object", "objects", "rois"}]}
      ```
      - `"object"` is the window's own object id, or -1 for a false-positive (fp) window.
      - `"objects"` lists `[{"id", "box", "area"}]` for every kept object whose box intersects the window.
      - `"rois"` is `{arm: [{"roi", "box", "conf", "object"}]}`.
      - `"roi_files"` is `{arm: {split: path}}`: the provenance Task 12 checks against the arm's detector (spec §9).

- [ ] **Step 1: Write the failing test**

Create `tests/test_roi_crops.py`:

```python
import os
import json
import numpy as np
import pytest
from PIL import Image

from data_loader.boxes import expand_box
from data_loader.cube_cache import build_cube_cache, cache_path, default_cache_dir
from data_loader.my_dataset import HyperCOD_data
from data_loader.roi_crops import match_rois, build_crop_cache, frame_windows, pixel_box, inside
from tests.conftest import H, W

OBJ1 = [20.0, 10.0, 26.0, 16.0]          # conftest OBJ_SLICE (rows 10:16, cols 20:26), in every frame
OBJ2 = [4.0, 30.0, 12.0, 40.0]           # second object added to train frame '3' (rows 30:40, cols 4:12, 80 px)
FP_ROI = [28.0, 36.0, 40.0, 48.0]        # touches no object
# hand-written ROI exports (main_det_rois.py schema); boxes = pre-expansion boxes + conf, always inside their roi
ROIS = {
    ('raw', 'train'): {'3': {'rois': [[16, 6, 30, 20, 0.9], FP_ROI + [0.3]], 'boxes': [[19, 9, 27, 17, 0.9], [30, 38, 38, 46, 0.3]]}},
    ('ec10', 'train'): {'3': {'rois': [[0, 26, 8, 44, 0.8], FP_ROI + [0.2]], 'boxes': [[2, 30, 8, 40, 0.8], [30, 38, 38, 46, 0.2]]}},
    ('ec24', 'train'): {'3': {'rois': [], 'boxes': []}},
    ('raw', 'val'): {'10': {'rois': [[15, 5, 31, 21, 0.7]], 'boxes': [[19, 9, 27, 17, 0.7]]}},
    ('ec10', 'val'): {'10': {'rois': [], 'boxes': []}},
    ('ec24', 'val'): {'10': {'rois': [], 'boxes': []}},
}
KW = dict(grow=2.0, min_side=8, min_area=10, num_workers=0)           # fixture frames are 48 x 40, objects 36 / 80 px


def _crop_cache(synthetic_root, tmp_path):
    '''Fixture frames + fp16 frame cache + split file (train ['3'], val ['10']) + ROI files -> built crop cache.'''
    root, _, _ = synthetic_root
    gt = np.zeros((H, W), dtype=np.uint8)
    gt[10:16, 20:26] = 255; gt[30:40, 4:12] = 255
    Image.fromarray(np.stack([gt] * 3, axis=-1)).save(root / 'train' / 'GT' / '3.png')
    build_cube_cache(str(root), 'train', num_workers=0)
    split_file = tmp_path / 'val.json'
    split_file.write_text(json.dumps({'seed': 0, 'n_val': 1, 'val_ids': ['10']}))
    roi_files = {}
    for (arm, split), data in ROIS.items():
        p = tmp_path / f'rois_{arm}_{split}.json'
        p.write_text(json.dumps(data))
        roi_files.setdefault(arm, {})[split] = str(p)
    out = tmp_path / 'crops'
    stats = build_crop_cache(str(root), str(out), roi_files, split_file=str(split_file), **KW)
    return root, out, roi_files, split_file, stats


def test_match_rois_max_coverage_and_threshold():
    labels = np.zeros((60, 60), dtype=np.int32)
    labels[0:20, 0:20] = 1                                                 # 400 px
    labels[30:40, 30:40] = 2                                               # 100 px
    rois = [[0, 0, 20, 20],                    # all of 1
            [10, 10, 40, 40],                  # 25 % of 1, 100 % of 2 -> 2
            [19.2, 0, 21, 2],                  # floor/ceil: cols 19..20, rows 0..1 -> 2 px of 1 = 0.5 % < 1 %
            [19.2, 0, 21, 4],                  # 4 px of 1 = 1 % -> 1 (threshold inclusive)
            [45, 45, 60, 60]]                  # nothing
    np.testing.assert_array_equal(match_rois(rois, labels, [1, 2]), [1, 2, -1, 1, -1])
    np.testing.assert_array_equal(match_rois(rois, labels, [2]), [-1, 2, -1, -1, -1])     # 1 is not a kept object
    assert match_rois(np.zeros((0, 4)), labels, [1, 2]).shape == (0,)
    np.testing.assert_array_equal(match_rois(rois, labels, []), [-1] * 5)


def test_frame_windows_contain_rois_and_dedupe_fp():
    gt = np.zeros((48, 40), dtype=bool)
    gt[10:16, 20:26] = True; gt[30:40, 4:12] = True
    fr = {arm: ROIS[(arm, 'train')]['3'] for arm in ('raw', 'ec10', 'ec24')}
    wins = frame_windows(gt, fr, grow=2.0, min_side=8, min_area=10)
    assert [w['kind'] for w in wins] == ['object', 'object', 'fp']
    o1, o2, fp = wins
    assert o1['object'] == 1 and o2['object'] == 2
    # window = union(expand_box(gt, 2, 8), matched ROIs): obj1 [17,7,29,19] u raw [16,6,30,20]; obj2 [0,25,16,45] u ec10 [0,26,8,44]
    assert o1['window'] == [16, 6, 30, 20] and o2['window'] == [0, 25, 16, 45]
    assert [e['roi'] for e in o1['rois']['raw']] == [[16, 6, 30, 20]] and o1['rois']['ec10'] == []
    assert [e['object'] for e in o2['rois']['ec10']] == [2]
    # the fp ROI shared by raw and ec10 is stored once, with each arm's own confidence
    assert fp['window'] == [28, 36, 40, 48] and fp['object'] == -1 and fp['objects'] == []
    assert [e['conf'] for e in fp['rois']['raw']] == [0.3] and [e['conf'] for e in fp['rois']['ec10']] == [0.2] and fp['rois']['ec24'] == []
    assert {o['id']: o['area'] for o in o1['objects'] + o2['objects']} == {1: 36, 2: 80}


def test_build_crop_cache_index_and_files(synthetic_root, tmp_path):
    root, out, roi_files, split_file, stats = _crop_cache(synthetic_root, tmp_path)
    with open(out / 'index.json') as f:
        index = json.load(f)
    assert index['band_range'] == [400.0, 800.0] and index['frame_hw'] == [H, W] and index['arms'] == ['raw', 'ec10', 'ec24']
    wins = index['windows']
    assert [(w['frame'], w['split'], w['kind']) for w in wins] == [('3', 'train', 'object'), ('3', 'train', 'object'),
                                                                     ('3', 'train', 'fp'), ('10', 'val', 'object')]
    assert stats['n_windows'] == 4 and stats['n_object'] == 3 and stats['n_fp'] == 1
    assert stats['bytes'] == sum(os.path.getsize(out / w[k]) for w in wins for k in ('file', 'gt_file'))
    ds = HyperCOD_data(str(root), split='train', use_filter=False, norm='p99', crop_size=0, filter_norm='none',
                       cache_dir=default_cache_dir(str(root)), out_dtype='float16')
    for w in wins:
        x1, y1, x2, y2 = w['window']
        frame = np.load(cache_path(default_cache_dir(str(root)), 'train', w['frame']))       # [133, H, W] fp16
        cube, gt = np.load(out / w['file']), np.load(out / w['gt_file'])
        assert cube.dtype == np.float16 and cube.shape == (133, y2 - y1, x2 - x1)
        np.testing.assert_array_equal(cube, frame[:, y1:y2, x1:x2])                          # un-scaled, row-major y, x
        assert gt.dtype == bool and np.array_equal(gt, ds.load_gt(w['frame'])[y1:y2, x1:x2])
        assert w['scale'] == ds.scale[w['frame']]
        for arm in index['arms']:                                                             # spec §9
            assert all(inside(pixel_box(e['roi'], H, W), w['window']) for e in w['rois'][arm])
        if w['kind'] == 'object':
            gt_box = next(o['box'] for o in w['objects'] if o['id'] == w['object'])
            assert inside(pixel_box(expand_box(gt_box, 2.0, 8, H, W), H, W), w['window'])
    assert not (out / 'index.json.tmp').exists()


def test_build_crop_cache_raises_on_missing_frame(synthetic_root, tmp_path):
    root, out, roi_files, split_file, _ = _crop_cache(synthetic_root, tmp_path)
    bad = tmp_path / 'rois_raw_val_other.json'
    bad.write_text(json.dumps({'99': {'rois': [], 'boxes': []}}))                 # an ROI file of another split
    roi_files['raw']['val'] = str(bad)
    with pytest.raises(AssertionError, match='lacks frames'):
        build_crop_cache(str(root), str(tmp_path / 'crops2'), roi_files, split_file=str(split_file), **KW)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python -m pytest tests/test_roi_crops.py -v`
Expected: FAIL. Collection errors with `ModuleNotFoundError: No module named 'data_loader.roi_crops'`.

- [ ] **Step 3: Write minimal implementation**

Create `data_loader/roi_crops.py`:

```python
import torch.utils.data as Dataset
import os
import json
import math
import numpy as np
import torch

from data_loader.boxes import boxes_from_mask, expand_box
from data_loader.cube_cache import read_npy_direct, cache_path, default_cache_dir
from data_loader.det_splits import load_det_ids
from data_loader.my_dataset import HyperCOD_data

ARMS = ('raw', 'ec10', 'ec24')          # input arms that have their own Stage-1 detector (and ROI files)
SEG_ARMS = ARMS + ('rgb',)              # 'rgb' = pseudo-RGB control; it reuses the raw arm's ROIs
INDEX_NAME = 'index.json'


def pixel_box(box, H, W):
    '''
    Float xyxy (native pixel edges) -> int (x1, y1, x2, y2): floor x1/y1, ceil x2/y2, clipped to [0, W] x [0, H].
    The same rule as train_eval.box_metrics.mask_coverage, so "ROI inside window" and "mask covered by ROI" agree
    to the pixel. expand_box returns fractional values (e.g. 18.5), hence the rounding.
    '''
    x1 = min(max(0, int(math.floor(float(box[0])))), W)
    y1 = min(max(0, int(math.floor(float(box[1])))), H)
    x2 = min(max(0, int(math.ceil(float(box[2])))), W)
    y2 = min(max(0, int(math.ceil(float(box[3])))), H)
    return x1, y1, x2, y2


def inside(inner, outer):
    '''True if the int xyxy rectangle inner lies inside the int xyxy rectangle outer.'''
    return inner[0] >= outer[0] and inner[1] >= outer[1] and inner[2] <= outer[2] and inner[3] <= outer[3]


# ----------------------------------------------------------------------------------------------------------------
# ROI matching and the crop cache
# ----------------------------------------------------------------------------------------------------------------
def match_rois(rois, labels, ids, min_cover=0.01):
    '''
    Match each detector ROI to the object whose mask it covers most.
    rois [K, 4] float xyxy frame px; labels [H, W] int32 component labels and ids the kept (1-based) labels, both
    from boxes_from_mask(..., return_labels=True). Coverage = share of the object's mask pixels inside the ROI
    (pixel_box rule, as mask_coverage). Returns np.ndarray [K] int64: the object id with the largest coverage if
    that coverage >= min_cover, else -1 (a false-positive ROI). Ties go to the lower id.
    '''
    rois = np.asarray(rois, dtype=np.float64).reshape(-1, 4)          # [K, 4]
    out = -np.ones(len(rois), dtype=np.int64)                         # [K]
    ids = [int(i) for i in ids]
    if len(rois) == 0 or len(ids) == 0:
        return out
    H, W = labels.shape
    n_lab = int(labels.max()) + 1
    totals = np.bincount(labels.ravel(), minlength=n_lab)             # [n_lab] mask pixels per label
    for k, r in enumerate(rois):
        x1, y1, x2, y2 = pixel_box(r, H, W)
        if x2 <= x1 or y2 <= y1:
            continue
        within = np.bincount(labels[y1:y2, x1:x2].ravel(), minlength=n_lab)   # [n_lab] pixels inside the ROI
        cover = np.array([within[i] / totals[i] for i in ids])                # [n_obj]
        j = int(np.argmax(cover))
        if cover[j] >= min_cover:
            out[k] = ids[j]
    return out


def frame_windows(gt, frame_rois, grow=2.0, min_side=512, min_area=100, min_cover=0.01):
    '''
    The cache windows of one frame (no I/O).
    gt: bool [H, W] full-frame GT; frame_rois: {arm: {"rois": [[x1,y1,x2,y2,conf], ...], "boxes": [...]}} of this
    frame from each arm's ROI export. Returns a list of dicts with kind, window (int xyxy), object, objects, rois:
      object window (one per GT object with area >= min_area): the smallest rectangle containing
        expand_box(gt_box, grow, min_side, H, W) and every ROI (any arm) matched to the object; rois[arm] = that
        arm's ROIs matched to the object;
      fp window (one per distinct false-positive ROI): the ROI itself; an fp ROI with identical coordinates in
        several arms is stored once, with each arm's entry under rois[arm].
    Every ROI entry is {"roi", "box", "conf", "object"} (object = matched id or -1).
    '''
    H, W = gt.shape
    gt_boxes, labels, ids = boxes_from_mask(gt, min_area=min_area, return_labels=True)
    areas = np.bincount(labels.ravel(), minlength=max(ids, default=0) + 1)    # pixels per component label
    objects = [{'id': int(i), 'box': [float(v) for v in b], 'area': int(areas[i])} for b, i in zip(gt_boxes, ids)]
    arms = list(frame_rois)
    matched = {o['id']: {arm: [] for arm in arms} for o in objects}
    fps = {}                                                          # rounded roi -> {"roi", "rois": {arm: [...]}}
    for arm in arms:
        rois = np.asarray(frame_rois[arm]['rois'], dtype=np.float64).reshape(-1, 5)     # [K, 5] xyxy + conf
        boxes = np.asarray(frame_rois[arm]['boxes'], dtype=np.float64).reshape(-1, 5)   # [K, 5] pre-expansion boxes
        assert len(rois) == len(boxes), f"arm {arm}: {len(rois)} rois but {len(boxes)} boxes"
        obj = match_rois(rois[:, :4], labels, ids, min_cover)                            # [K]
        for r, b, o in zip(rois, boxes, obj):
            e = {'roi': [float(v) for v in r[:4]], 'box': [float(v) for v in b[:4]], 'conf': float(r[4]), 'object': int(o)}
            if o >= 0:
                matched[int(o)][arm].append(e)
            else:
                key = tuple(np.round(r[:4], 3).tolist())
                fps.setdefault(key, {'roi': e['roi'], 'rois': {a: [] for a in arms}})['rois'][arm].append(e)

    def overlapping(win):
        # objects whose box intersects the window (for an object window this includes the window's own object)
        return [o for o in objects if o['box'][0] < win[2] and o['box'][2] > win[0] and o['box'][1] < win[3] and o['box'][3] > win[1]]

    windows = []
    for o in objects:
        rects = [pixel_box(expand_box(o['box'], grow, min_side, H, W), H, W)]
        rects += [pixel_box(e['roi'], H, W) for arm in arms for e in matched[o['id']][arm]]
        win = [min(r[0] for r in rects), min(r[1] for r in rects), max(r[2] for r in rects), max(r[3] for r in rects)]
        windows.append({'kind': 'object', 'window': win, 'object': o['id'], 'objects': overlapping(win),
                        'rois': matched[o['id']]})
    for key in sorted(fps):
        win = list(pixel_box(fps[key]['roi'], H, W))
        assert win[2] > win[0] and win[3] > win[1], f"degenerate false-positive ROI {fps[key]['roi']}"
        windows.append({'kind': 'fp', 'window': win, 'object': -1, 'objects': overlapping(win), 'rois': fps[key]['rois']})
    for w in windows:                                                 # spec §9: every ROI inside its window
        for arm in arms:
            bad = [e['roi'] for e in w['rois'][arm] if not inside(pixel_box(e['roi'], H, W), w['window'])]
            assert not bad, f"{arm} ROIs {bad} outside window {w['window']}"
    return windows


def load_roi_file(path, names):
    '''{name: {"rois", "boxes", ...}} of an ROI export (main_det_rois.py), restricted to names; every name must be present.'''
    assert os.path.exists(path), f"ROI file {path} not found; export it with main_det_rois.py"
    with open(path) as f:
        data = json.load(f)
    missing = [n for n in names if n not in data]
    assert not missing, f"ROI file {path} lacks frames {missing[:10]} ({len(missing)} missing): wrong split or detector?"
    return {n: data[n] for n in names}


class CropFrames(Dataset.Dataset):
    '''
    One item per frame for build_crop_cache: reads the frame once (O_DIRECT), writes its windows
    (<id>_<k>.npy fp16 [n_bands, h, w] un-scaled, <id>_<k>_gt.npy bool [h, w]) and returns their index records,
    so DataLoader workers do the reading and writing in parallel.
    '''
    def __init__(self, ds, split, frame_rois, out_dir, grow=2.0, min_side=512, min_area=100, min_cover=0.01):
        super(CropFrames, self).__init__()
        self.ds = ds                    # HyperCOD_data (split 'train', ids of this split, norm 'p99', cache_dir set)
        self.split = split              # 'train' or 'val' (both live in the HyperCOD 'train' directory)
        self.frame_rois = frame_rois    # {name: {arm: {"rois", "boxes"}}}
        self.out_dir = out_dir
        self.grow, self.min_side, self.min_area, self.min_cover = grow, min_side, min_area, min_cover

    def __len__(self):
        return len(self.ds.img_name)

    def __getitem__(self, idx):
        name = self.ds.img_name[idx]
        gt = self.ds.load_gt(name)                                    # [H, W] bool (raw GT incl. JPEG specks)
        wins = frame_windows(gt, self.frame_rois[name], self.grow, self.min_side, self.min_area, self.min_cover)
        if not wins:
            return []
        p = cache_path(self.ds.cache_dir, self.ds.split, name)
        assert os.path.exists(p), f"frame cache {p} missing; build it with python -m data_loader.cube_cache"
        arr = read_npy_direct(p)                                      # [n_bands, H, W] fp16, one O_DIRECT read
        assert arr.shape == (self.ds.n_bands, self.ds.H, self.ds.W) and arr.dtype == np.float16, \
            f"cache {p} has {arr.shape} {arr.dtype}, expected ({self.ds.n_bands}, {self.ds.H}, {self.ds.W}) float16"
        records = []
        for k, w in enumerate(wins):
            x1, y1, x2, y2 = w['window']
            f, g = f'{name}_{k}.npy', f'{name}_{k}_gt.npy'
            np.save(os.path.join(self.out_dir, f), np.ascontiguousarray(arr[:, y1:y2, x1:x2]))   # [n_bands, h, w] fp16
            np.save(os.path.join(self.out_dir, g), np.ascontiguousarray(gt[y1:y2, x1:x2]))       # [h, w] bool
            records.append({'file': f, 'gt_file': g, 'frame': name, 'split': self.split, 'kind': w['kind'],
                            'window': [int(v) for v in w['window']], 'scale': float(self.ds.scale[name]),
                            'object': w['object'], 'objects': w['objects'], 'rois': w['rois']})
        return records


def records_collate_fn(batch):
    '''CropFrames items are lists of index records; flatten the batch into one list.'''
    return [r for recs in batch for r in recs]


def build_crop_cache(data_path, out_dir, roi_files, splits=('train', 'val'), grow=2.0, min_side=512, num_workers=4,
                     min_area=100, min_cover=0.01, split_file=None, cache_dir=None, band_range=(400.0, 800.0)):
    '''
    One-off crop cache for Stage-2 training (spec §3): every train/val frame is read once (O_DIRECT, from the fp16
    frame cache) and cut into object windows and false-positive windows (see frame_windows).
    roi_files: {arm: {split: path}} with arm in ARMS, from main_det_rois.py (rois_<run>_<split>.json).
    split_file: the det_val_ids.json that defines train (251) / val (28) ids (default data_loader/splits/...).
    Writes <out_dir>/<id>_<k>.npy, <out_dir>/<id>_<k>_gt.npy and, last and atomically, <out_dir>/index.json =
      {"band_range", "frame_hw", "grow", "min_side", "min_area", "min_cover", "arms", "roi_files",
       "windows": [{"file", "gt_file", "frame", "split", "kind", "window", "scale", "object", "objects", "rois"}]}
    so a crashed build leaves no index.json and is rebuilt. Returns {"n_windows", "bytes", "n_object", "n_fp"}.
    '''
    arms = [a for a in ARMS if a in roi_files]
    assert arms and set(roi_files) <= set(ARMS), f"roi_files arms must be in {ARMS}, got {list(roi_files)}"
    for arm in arms:
        assert all(s in roi_files[arm] for s in splits), f"roi_files[{arm!r}] lacks splits {[s for s in splits if s not in roi_files[arm]]}"
    cache_dir = cache_dir if cache_dir else default_cache_dir(data_path)
    train_ids, val_ids = load_det_ids(data_path, split_file)
    os.makedirs(out_dir, exist_ok=True)
    windows, frame_hw = [], None
    for split in splits:
        assert split in ('train', 'val'), f"split must be 'train' or 'val', got {split!r}"
        # val frames live in the HyperCOD 'train' directory; norm 'p99' only to get the per-frame scale
        ds = HyperCOD_data(data_path, split='train', ids=train_ids if split == 'train' else val_ids, use_filter=False,
                           norm='p99', crop_size=0, band_range=band_range, filter_norm='none', cache_dir=cache_dir,
                           out_dtype='float16')
        frame_hw = [ds.H, ds.W]
        rois = {arm: load_roi_file(roi_files[arm][split], ds.img_name) for arm in arms}
        frame_rois = {name: {arm: rois[arm][name] for arm in arms} for name in ds.img_name}
        loader = torch.utils.data.DataLoader(
            CropFrames(ds, split, frame_rois, out_dir, grow, min_side, min_area, min_cover), batch_size=1,
            shuffle=False, num_workers=num_workers, collate_fn=records_collate_fn)
        print(f"Crop cache: {len(ds)} {split} frames, arms {arms} -> {out_dir}")
        for i, recs in enumerate(loader):
            windows += recs
            if (i + 1) % 25 == 0 or i + 1 == len(ds):
                print(f"  {split} {i + 1}/{len(ds)} frames, {len(windows)} windows")
    n_bytes = sum(os.path.getsize(os.path.join(out_dir, w[k])) for w in windows for k in ('file', 'gt_file'))
    n_obj = sum(w['kind'] == 'object' for w in windows)
    index = {'band_range': [float(band_range[0]), float(band_range[1])], 'frame_hw': frame_hw, 'grow': float(grow),
             'min_side': int(min_side), 'min_area': int(min_area), 'min_cover': float(min_cover), 'arms': arms,
             'roi_files': {a: {s: str(roi_files[a][s]) for s in splits} for a in arms}, 'windows': windows}
    tmp = os.path.join(out_dir, INDEX_NAME + '.tmp')
    with open(tmp, 'w') as f:
        json.dump(index, f)
    os.replace(tmp, os.path.join(out_dir, INDEX_NAME))
    print(f"Crop cache done: {len(windows)} windows ({n_obj} object, {len(windows) - n_obj} fp), {n_bytes / 1e9:.2f} GB")
    return {'n_windows': len(windows), 'bytes': n_bytes, 'n_object': n_obj, 'n_fp': len(windows) - n_obj}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python -m pytest tests/test_roi_crops.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Review Focus test: the cache build through DataLoader workers (write it, run it)**

All other tests build with `num_workers=0`; every real build uses 4-6 workers (CropFrames pickling, parallel `np.save`). Append to `tests/test_roi_crops.py`:

```python


def test_build_crop_cache_with_workers_matches_serial(synthetic_root, tmp_path):
    root, out, roi_files, split_file, stats = _crop_cache(synthetic_root, tmp_path)
    out2 = tmp_path / 'crops_mp'
    stats2 = build_crop_cache(str(root), str(out2), roi_files, split_file=str(split_file), **{**KW, 'num_workers': 2})
    assert stats2 == stats
    a = json.loads((out / 'index.json').read_text())
    b = json.loads((out2 / 'index.json').read_text())
    assert a['windows'] == b['windows']                                         # same windows, same order
    for w in a['windows']:
        np.testing.assert_array_equal(np.load(out / w['file']), np.load(out2 / w['file']))
        np.testing.assert_array_equal(np.load(out / w['gt_file']), np.load(out2 / w['gt_file']))
```

Run: `OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python -m pytest tests/test_roi_crops.py -k workers -v`
Expected: PASS with the implementation of Step 3 (it is a guard, not a red test: a failure means worker pickling or ordering is broken; fix `CropFrames` / `build_crop_cache`, not the test). The training-item half of this Review Focus line lives in Task 4 Step 5, because it needs `HyperCOD_roi`.

- [ ] **Step 6: Run the file and the full suite**

Run: `OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python -m pytest tests/test_roi_crops.py -v`
Expected: PASS (5 passed).

Then: `OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python -m pytest tests/ -q`. It must stay green: the previous count plus 5 new tests.

- [ ] **Step 7: Commit and push**

```bash
git add data_loader/roi_crops.py tests/test_roi_crops.py
git commit -m "feat(seg): crop cache of the Stage-1 ROIs for Stage-2 training (match_rois, build_crop_cache)

Each train/val frame is read once with O_DIRECT and cut into object windows (GT box x2.0 >= 512 px, plus every
matched detector ROI of any arm) and false-positive windows, so training never re-reads 0.55 GB frames.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QB4DScDwiA2DkVpbDopffV"
git push origin worktree-det
git push origin worktree-det:master
```

---

### Task 4: ROI dataset on the canvas (`place_on_canvas`, `canvas_to_roi`, `HyperCOD_roi`, `seg_collate_fn`)

**Files:**
- Modify: `data_loader/roi_crops.py:1-15` (the import block and constants) and append to the end of the file
- Test: `tests/test_roi_crops.py:1-11` (the import block) and append to the end of the file

**Interfaces:**
- Consumes:
  - From Task 3: `pixel_box`, `inside`, `SEG_ARMS`, `INDEX_NAME` and the `index.json` schema (`frame_hw`, `grow`, `min_side`, `arms`, window `object`, object `area`)
  - `data_loader.my_dataset.scale_block(blk, scale)`: fp16 p99 scaling through torch, the same rounding as `HyperCOD_data.__getitem__`
  - `data_loader.boxes.expand_box`
- Produces:
  - `place_on_canvas(crop [C, h, w] np, canvas=512, scale=1.0) -> (out [C, canvas, canvas] in crop dtype, valid [canvas, canvas] bool, (oy, ox), s)`
    - `s = min(1, canvas / max(h, w)) * scale`, clipped so the resized crop fits. The crop is centred and the rest is zeros.
    - The placed size is `placed_size(h, w, s, canvas) = (min(canvas, max(1, round(h*s))), min(canvas, max(1, round(w*s))))`.
    - The crop is copied bit for bit when that size equals (h, w). Otherwise it is resized bilinearly, antialiased when shrinking.
  - `canvas_to_roi(canvas_map [canvas, canvas] np or tensor, (oy, ox), s, (h, w)) -> [h, w] float32`
  - Helpers for Task 11 / Task 13 (paste-back and test-time ROI assembly): `placed_size`, `resize_chw`, `box_to_canvas(box, roi_int, offset, hw, s, canvas)`, `rasterise_box(box, canvas)`, `flip_rot(arrays, box, hflip, vflip, k, canvas)`, `jitter_gt_box(box, gt_jitter, rng, H, W)`, `n_fp_items(n_obj, p_fp)`, `load_window(path)`
  - `class HyperCOD_roi(Dataset.Dataset)` with `__init__(self, cache_dir, split, arm, box_mix=(0.5, 0.4, 0.1), gt_jitter=0.15, roi_margin=1.5, roi_min=256, canvas=512, scale_aug=(0.75, 1.25), gain_aug=0.05, train=True, seed=None)`
    - `__getitem__` returns `(img fp16 [133, c, c] p99-scaled, mask float32 [1, c, c], box_map float32 [1, c, c], valid float32 [1, c, c], meta)`.
    - `meta` is a dict with these keys:
      - From the contract: `frame`, `roi` (int px), `box`, `box_canvas`, `source` ('gt' | 'det' | 'fp'), `offset`, `s`, `roi_hw` and `obj_area`.
      - Added: `object` (the object id, -1 for a false-positive ROI), `area` (the object's own full-frame mask area, for the size buckets) and `aug` (hflip, vflip, k).
    - A missing window file raises with the frame and file in the message (spec §9), never skips.
  - `seg_collate_fn(batch) -> {"img" [B,133,c,c] fp16, "mask" / "box_map" / "valid" [B,1,c,c] float32, "box_xyxy" [B,4] float32 canvas px, "meta": list}`

- [ ] **Step 1: Write the failing test**

Replace the import block at the top of `tests/test_roi_crops.py` (lines 1-11) with:

```python
import os
import json
import random
import numpy as np
import pytest
import torch
from PIL import Image

from data_loader.boxes import expand_box
from data_loader.cube_cache import build_cube_cache, cache_path, default_cache_dir
from data_loader.my_dataset import HyperCOD_data
from data_loader.roi_crops import match_rois, build_crop_cache, frame_windows, pixel_box, inside
from data_loader.roi_crops import (place_on_canvas, canvas_to_roi, rasterise_box, flip_rot, jitter_gt_box, n_fp_items,
                                   HyperCOD_roi, seg_collate_fn)
from tests.conftest import H, W
```

Then append to the end of `tests/test_roi_crops.py`:

```python


DS_KW = dict(roi_margin=1.5, roi_min=0, canvas=32)                       # oracle ROIs of the 6 x 6 object are 10 x 10


def test_place_on_canvas_exact_and_resized_round_trip():
    rng = np.random.default_rng(0)
    crop = rng.random((3, 20, 30)).astype(np.float16)
    out, valid, (oy, ox), s = place_on_canvas(crop, canvas=64)
    assert s == 1.0 and (oy, ox) == (22, 17) and out.dtype == np.float16 and valid.sum() == 600
    np.testing.assert_array_equal(out[:, oy:oy + 20, ox:ox + 30], crop)
    assert not out[:, ~valid].any()
    np.testing.assert_array_equal(canvas_to_roi(out[1].astype(np.float32), (oy, ox), s, (20, 30)), crop[1].astype(np.float32))
    np.testing.assert_array_equal(canvas_to_roi(torch.from_numpy(out[1].astype(np.float32)), (oy, ox), s, (20, 30)),
                                  crop[1].astype(np.float32))
    # larger than the canvas: aspect kept, downscaled to fit, round trip within interpolation error on a smooth map
    yy, xx = np.mgrid[0:100, 0:60]
    smooth = (0.5 + 0.5 * np.sin(yy / 15.0) * np.cos(xx / 11.0)).astype(np.float32)[None]   # [1, 100, 60]
    out, valid, (oy, ox), s = place_on_canvas(smooth, canvas=64)
    assert abs(s - 0.64) < 1e-9 and valid.sum() == 64 * 38 and (oy, ox) == (0, 13)
    back = canvas_to_roi(out[0], (oy, ox), s, (100, 60))
    assert back.shape == (100, 60) and np.abs(back - smooth[0])[4:-4, 4:-4].max() < 0.05
    # scale augmentation never pushes the crop off the canvas
    out, valid, _, s = place_on_canvas(crop, canvas=64, scale=3.0)
    assert s == pytest.approx(64 / 30) and valid.sum() == 43 * 64


def test_flip_rot_keeps_box_and_arrays_consistent():
    c = 16
    box = [3.5, 2.0, 9.0, 6.2]
    for hf in (False, True):
        for vf in (False, True):
            for k in range(4):
                (m,), b = flip_rot([rasterise_box(box, c)], box, hf, vf, k, c)
                np.testing.assert_array_equal(m, rasterise_box(b, c))


def test_jittered_gt_box_never_cuts_the_object():
    rng = random.Random(0)
    gt = [100.0, 50.0, 160.0, 90.0]
    for _ in range(1000):
        b = jitter_gt_box(gt, 0.15, rng, 1680, 1240)
        assert b[0] <= gt[0] and b[1] <= gt[1] and b[2] >= gt[2] and b[3] >= gt[3]
        assert b[2] - b[0] <= 60 * 1.3 + 1e-3 and b[3] - b[1] <= 40 * 1.3 + 1e-3
        # the expanded ROI fits the cached object window (grow 2.0, min_side 512)
        assert inside(pixel_box(expand_box(b, 1.5, 256, 1680, 1240), 1680, 1240),
                      pixel_box(expand_box(gt, 2.0, 512, 1680, 1240), 1680, 1240))
    assert n_fp_items(289, 0.1) == 32 and n_fp_items(2, 0.0) == 0


def test_validation_items_are_deterministic_and_exact(synthetic_root, tmp_path):
    root, out, _, _, _ = _crop_cache(synthetic_root, tmp_path)
    ds = HyperCOD_roi(str(out), 'val', 'raw', train=False, **DS_KW)
    assert len(ds) == 2                                                    # oracle ROI + the matched raw val ROI
    ref = HyperCOD_data(str(root), split='train', ids=['10'], use_filter=False, norm='p99', crop_size=0,
                        filter_norm='none', cache_dir=default_cache_dir(str(root)), out_dtype='float16')
    frame, gt, _ = ref[0]                                                  # [133, H, W] fp16 p99-scaled, [1, H, W]
    for i, source in enumerate(['gt', 'det']):
        img, mask, box_map, valid, meta = ds[i]
        assert meta['source'] == source and meta['frame'] == '10' and meta['s'] == 1.0
        x1, y1, x2, y2 = [int(v) for v in meta['roi']]
        if source == 'gt':
            assert (x1, y1, x2, y2) == pixel_box(expand_box(OBJ1, 1.5, 0, H, W), H, W) == (18, 8, 28, 18)
            assert meta['box'] == OBJ1
        else:
            assert (x1, y1, x2, y2) == (15, 5, 31, 21) and meta['box'] == [19, 9, 27, 17]
        oy, ox = meta['offset']
        h, w = meta['roi_hw']
        assert (h, w) == (y2 - y1, x2 - x1)
        assert img.dtype == np.float16 and img.shape == (133, 32, 32) and mask.shape == box_map.shape == valid.shape == (1, 32, 32)
        np.testing.assert_array_equal(img[:, oy:oy + h, ox:ox + w], frame[:, y1:y2, x1:x2])     # same fp16 p99 values
        np.testing.assert_array_equal(mask[0, oy:oy + h, ox:ox + w], gt[0, y1:y2, x1:x2])
        assert meta['obj_area'] == 36 and meta['area'] == 36 and mask.sum() == 36
        np.testing.assert_array_equal(box_map[0], rasterise_box(meta['box_canvas'], 32))
        bx1, by1, bx2, by2 = meta['box_canvas']
        assert (bx1, by1) == (ox + meta['box'][0] - x1, oy + meta['box'][1] - y1)
        assert valid.sum() == h * w and not img[:, valid[0] == 0].any()
        img2, mask2, _, _, meta2 = ds[i]
        assert np.array_equal(img, img2) and np.array_equal(mask, mask2) and meta == meta2


def test_training_items_mix_jitter_and_targets(synthetic_root, tmp_path):
    root, out, _, _, _ = _crop_cache(synthetic_root, tmp_path)
    ref = HyperCOD_data(str(root), split='train', ids=['3'], use_filter=False, norm='p99', crop_size=0,
                        filter_norm='none', cache_dir=default_cache_dir(str(root)), out_dtype='float16')
    full_gt = ref.load_gt('3')
    # object items only, no augmentation that resizes: check targets and the 5/9 split
    ds = HyperCOD_roi(str(out), 'train', 'raw', scale_aug=(1.0, 1.0), gain_aug=0.0, seed=0, **DS_KW)
    assert len(ds) == 2 and ds.n_fp == 0                                   # 2 objects, round(2 x 0.1 / 0.9) = 0 fp
    counts = {1: {'gt': 0, 'det': 0}, 2: {'gt': 0, 'det': 0}}
    for _ in range(300):
        for i in range(2):
            img, mask, box_map, valid, meta = ds[i]
            counts[meta['object']][meta['source']] += 1
            x1, y1, x2, y2 = [int(v) for v in meta['roi']]
            assert meta['obj_area'] == int(full_gt[y1:y2, x1:x2].sum())
            assert mask.sum() == meta['obj_area']                          # s = 1: the GT crop, only rotated / flipped
            np.testing.assert_array_equal(box_map[0], rasterise_box(meta['box_canvas'], 32))
            assert not mask[valid == 0].any() and not box_map[valid == 0].any()
            if meta['source'] == 'gt':                                     # jittered GT: object never cut
                assert meta['obj_area'] == meta['area']
                ob = OBJ1 if meta['object'] == 1 else OBJ2
                assert inside(pixel_box(ob, H, W), pixel_box(meta['box'], H, W))
            else:                                                          # matched raw ROI of object 1
                assert meta['roi'] == [16, 6, 30, 20] and meta['object'] == 1
    assert counts[2]['det'] == 0 and counts[2]['gt'] == 300               # no matched raw ROI -> GT fallback
    assert abs(counts[1]['gt'] / 300 - 5 / 9) < 0.08


def test_false_positive_items_and_epoch_size(synthetic_root, tmp_path):
    root, out, _, _, _ = _crop_cache(synthetic_root, tmp_path)
    ds = HyperCOD_roi(str(out), 'train', 'ec10', box_mix=(0.25, 0.25, 0.5), seed=1, **DS_KW)
    assert ds.n_fp == 2 and len(ds) == 4 and len(ds.fp_rois) == 1
    for _ in range(20):
        img, mask, box_map, valid, meta = ds[3]
        assert meta['source'] == 'fp' and meta['object'] == -1 and meta['roi'] == FP_ROI
        assert not mask.any() and box_map.any() and img.dtype == np.float16
    assert HyperCOD_roi(str(out), 'train', 'ec24', box_mix=(0.25, 0.25, 0.5), **DS_KW).n_fp == 0   # ec24 has no fp ROI
    assert len(HyperCOD_roi(str(out), 'val', 'ec10', train=False, **DS_KW)) == 1                  # oracle only


def test_rgb_arm_uses_raw_rois(synthetic_root, tmp_path):
    root, out, _, _, _ = _crop_cache(synthetic_root, tmp_path)
    a = HyperCOD_roi(str(out), 'val', 'rgb', train=False, **DS_KW)
    b = HyperCOD_roi(str(out), 'val', 'raw', train=False, **DS_KW)
    assert len(a) == len(b) == 2 and all(np.array_equal(x, y) for x, y in zip(a[1][:4], b[1][:4]))
    with pytest.raises(AssertionError, match='grow'):
        HyperCOD_roi(str(out), 'val', 'raw', train=False, roi_margin=2.0, roi_min=0, canvas=32)


def test_missing_window_raises_with_the_frame(synthetic_root, tmp_path):
    root, out, _, _, _ = _crop_cache(synthetic_root, tmp_path)
    ds = HyperCOD_roi(str(out), 'val', 'raw', train=False, **DS_KW)
    w = ds.items[0][0]
    os.remove(out / w['file'])
    with pytest.raises(AssertionError, match="frame 10"):                   # spec §9: never skipped
        ds[0]


def test_seg_collate_fn(synthetic_root, tmp_path):
    root, out, _, _, _ = _crop_cache(synthetic_root, tmp_path)
    ds = HyperCOD_roi(str(out), 'train', 'raw', seed=0, **DS_KW)
    loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=True, num_workers=0, collate_fn=seg_collate_fn)
    batch = next(iter(loader))
    assert batch['img'].dtype == torch.float16 and batch['img'].shape == (2, 133, 32, 32)
    for k in ('mask', 'box_map', 'valid'):
        assert batch[k].dtype == torch.float32 and batch[k].shape == (2, 1, 32, 32)
    assert batch['box_xyxy'].shape == (2, 4) and batch['box_xyxy'].dtype == torch.float32 and len(batch['meta']) == 2
    for i, m in enumerate(batch['meta']):
        assert torch.equal(batch['box_xyxy'][i], torch.tensor(m['box_canvas']))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python -m pytest tests/test_roi_crops.py -v`
Expected: FAIL. Collection errors with `ImportError: cannot import name 'place_on_canvas' from 'data_loader.roi_crops'`.

- [ ] **Step 3: Write minimal implementation**

Replace lines 1-15 of `data_loader/roi_crops.py` (the import block and constants) with:

```python
import torch.utils.data as Dataset
import os
import json
import math
import random
import numpy as np
import torch
import torch.nn.functional as F

from data_loader.boxes import boxes_from_mask, expand_box
from data_loader.cube_cache import read_npy_direct, cache_path, default_cache_dir
from data_loader.det_splits import load_det_ids
from data_loader.my_dataset import HyperCOD_data, scale_block

ARMS = ('raw', 'ec10', 'ec24')          # input arms that have their own Stage-1 detector (and ROI files)
SEG_ARMS = ARMS + ('rgb',)              # 'rgb' = pseudo-RGB control; it reuses the raw arm's ROIs
INDEX_NAME = 'index.json'
RESIZE_CHUNK = 16                       # channels per F.interpolate call: a 133-band 1680 x 700 crop in fp32 is 0.6 GB
```

Then append to the end of `data_loader/roi_crops.py`:

```python


# ----------------------------------------------------------------------------------------------------------------
# canvas geometry
# ----------------------------------------------------------------------------------------------------------------
def placed_size(h, w, s, canvas):
    '''(nh, nw) of an h x w crop resized by s on the canvas; shared by place_on_canvas and canvas_to_roi.'''
    nh = min(canvas, max(1, int(round(h * s))))
    nw = min(canvas, max(1, int(round(w * s))))
    return nh, nw


def resize_chw(x, size):
    '''
    Bilinear resize of a [C, h, w] array to size = (nh, nw); returns float32 [C, nh, nw].
    Antialiased when shrinking (a 1290 px ROI goes onto a 512 canvas), done in float32 chunks of RESIZE_CHUNK
    channels so a large 133-band crop never needs a full float32 copy.
    '''
    C, h, w = x.shape
    nh, nw = int(size[0]), int(size[1])
    if (h, w) == (nh, nw):
        return np.asarray(x, dtype=np.float32)
    anti = nh < h or nw < w
    out = np.empty((C, nh, nw), dtype=np.float32)
    for c0 in range(0, C, RESIZE_CHUNK):
        t = torch.from_numpy(np.ascontiguousarray(x[c0:c0 + RESIZE_CHUNK], dtype=np.float32))[None]   # [1, c, h, w]
        out[c0:c0 + RESIZE_CHUNK] = F.interpolate(t, size=(nh, nw), mode='bilinear', align_corners=False,
                                                  antialias=anti)[0].numpy()
    return out


def place_on_canvas(crop, canvas=512, scale=1.0):
    '''
    Put a native-resolution crop [C, h, w] on a canvas x canvas grid: s = min(1, canvas / max(h, w)) * scale,
    clipped so the resized crop always fits, resized (bilinear) only when the size changes, centred, zeros elsewhere.
    Returns (out [C, canvas, canvas] in crop's dtype, valid [canvas, canvas] bool = the placed region, (oy, ox), s).
    s = 1 (crop side <= canvas, no scale augmentation) copies the crop bit for bit.
    '''
    C, h, w = crop.shape
    assert h > 0 and w > 0, f"empty crop {crop.shape}"
    s = min(1.0, canvas / max(h, w)) * float(scale)
    s = min(s, canvas / max(h, w))                      # scale augmentation may not push the crop off the canvas
    nh, nw = placed_size(h, w, s, canvas)
    oy, ox = (canvas - nh) // 2, (canvas - nw) // 2     # centred
    out = np.zeros((C, canvas, canvas), dtype=crop.dtype)
    out[:, oy:oy + nh, ox:ox + nw] = crop if (nh, nw) == (h, w) else resize_chw(crop, (nh, nw)).astype(crop.dtype)
    valid = np.zeros((canvas, canvas), dtype=bool)
    valid[oy:oy + nh, ox:ox + nw] = True
    return out, valid, (oy, ox), s


def canvas_to_roi(canvas_map, offset, s, hw):
    '''
    Inverse of place_on_canvas for one map: cut the placed region out of canvas_map [canvas, canvas] and resize it
    back to the ROI's native (h, w). Returns float32 [h, w]; exact when no resize happened (s = 1).
    canvas_map may be a numpy array or a torch tensor (the model's sigmoid output).
    '''
    if torch.is_tensor(canvas_map):
        canvas_map = canvas_map.detach().float().cpu().numpy()
    canvas_map = np.asarray(canvas_map, dtype=np.float32)
    assert canvas_map.ndim == 2 and canvas_map.shape[0] == canvas_map.shape[1], f"expected [c, c], got {canvas_map.shape}"
    h, w = int(hw[0]), int(hw[1])
    oy, ox = int(offset[0]), int(offset[1])
    nh, nw = placed_size(h, w, s, canvas_map.shape[0])
    region = canvas_map[oy:oy + nh, ox:ox + nw]                        # [nh, nw]
    return region.copy() if (nh, nw) == (h, w) else resize_chw(region[None], (h, w))[0]


def box_to_canvas(box, roi, offset, hw, s, canvas):
    '''
    Frame-pixel xyxy box -> canvas-pixel xyxy, given the int ROI (x1, y1, x2, y2) it was placed from. Per-axis
    factors nw / w and nh / h (the rounded placed size), clipped to the placed region.
    '''
    h, w = hw
    oy, ox = offset
    nh, nw = placed_size(h, w, s, canvas)
    sx, sy = nw / w, nh / h
    x1 = ox + (float(box[0]) - roi[0]) * sx; x2 = ox + (float(box[2]) - roi[0]) * sx
    y1 = oy + (float(box[1]) - roi[1]) * sy; y2 = oy + (float(box[3]) - roi[1]) * sy
    return [float(np.clip(x1, ox, ox + nw)), float(np.clip(y1, oy, oy + nh)),
            float(np.clip(x2, ox, ox + nw)), float(np.clip(y2, oy, oy + nh))]


def rasterise_box(box, canvas):
    '''float32 [canvas, canvas] map, 1 on every pixel the xyxy box touches (pixel_box rule), 0 elsewhere.'''
    x1, y1, x2, y2 = pixel_box(box, canvas, canvas)
    m = np.zeros((canvas, canvas), dtype=np.float32)
    m[y1:y2, x1:x2] = 1.0
    return m


def flip_rot(arrays, box, hflip, vflip, k, canvas):
    '''
    Apply horizontal flip, vertical flip, then k x 90 deg counter-clockwise rotation (np.rot90 on the last two axes)
    to every array [..., canvas, canvas] and to the xyxy canvas box, so both stay consistent.
    rot90: out[i, j] = in[j, c - 1 - i]  ->  box (x1, y1, x2, y2) -> (y1, c - x2, y2, c - x1).
    '''
    x1, y1, x2, y2 = [float(v) for v in box]
    c = float(canvas)
    if hflip:
        arrays = [a[..., ::-1] for a in arrays]
        x1, x2 = c - x2, c - x1
    if vflip:
        arrays = [a[..., ::-1, :] for a in arrays]
        y1, y2 = c - y2, c - y1
    for _ in range(int(k) % 4):
        arrays = [np.rot90(a, 1, axes=(-2, -1)) for a in arrays]
        x1, y1, x2, y2 = y1, c - x2, y2, c - x1
    return [np.ascontiguousarray(a) for a in arrays], [x1, y1, x2, y2]


def jitter_gt_box(box, gt_jitter, rng, H, W):
    '''
    Training box from a GT box: each side moved OUTWARD by an independent U(0, gt_jitter) x (box side), never
    inward, so the object stays fully inside; clipped to the frame. rng: a random.Random (or the random module).
    '''
    x1, y1, x2, y2 = [float(v) for v in box]
    bw, bh = x2 - x1, y2 - y1
    return np.array([max(0.0, x1 - rng.uniform(0, gt_jitter) * bw), max(0.0, y1 - rng.uniform(0, gt_jitter) * bh),
                     min(float(W), x2 + rng.uniform(0, gt_jitter) * bw), min(float(H), y2 + rng.uniform(0, gt_jitter) * bh)],
                    dtype=np.float32)


def n_fp_items(n_obj, p_fp):
    '''Number of false-positive items so that they are a share p_fp of an epoch of n_obj object items.'''
    return int(round(n_obj * p_fp / (1.0 - p_fp))) if p_fp > 0 else 0


def load_window(path):
    '''
    Read one cached window whole (np.load through the page cache: the 22-34 GB crop cache is re-read every epoch
    and fits the ~72 GB of free page cache). No partial mmap reads, which stalled DataLoader workers in Stage 1.
    '''
    return np.load(path)


# ----------------------------------------------------------------------------------------------------------------
# training / validation dataset
# ----------------------------------------------------------------------------------------------------------------
class HyperCOD_roi(Dataset.Dataset):
    def __init__(self, cache_dir, split, arm, box_mix=(0.5, 0.4, 0.1), gt_jitter=0.15, roi_margin=1.5, roi_min=256,
                 canvas=512, scale_aug=(0.75, 1.25), gain_aug=0.05, train=True, seed=None):
        '''
        ROI items for Stage-2 segmentation, read from the crop cache of build_crop_cache.
        split: 'train' or 'val'; arm: 'raw', 'ec10', 'ec24' or 'rgb' (rgb uses the raw arm's ROIs).
        train=True: one item per object window plus false-positive items = box_mix[2] of the epoch (drawn uniformly
          from the arm's fp ROIs); an object item uses an expanded jittered GT box with probability
          box_mix[0] / (box_mix[0] + box_mix[1]) (5/9) and otherwise one of its matched detector ROIs (fallback:
          expanded GT if it has none), which gives the 50/40/10 mix overall. Augmentation: hflip, vflip, rot90,
          scale U(scale_aug) of the ROI content before placement, per-channel gain U(1 - gain_aug, 1 + gain_aug).
        train=False: deterministic; per object the oracle ROI expand_box(gt, roi_margin, roi_min) with the GT box as
          the box channel, plus every matched detector ROI of the arm; no augmentation and no fp items.
        '''
        super(HyperCOD_roi, self).__init__()
        assert split in ('train', 'val'), f"split must be 'train' or 'val', got {split!r}"
        assert arm in SEG_ARMS, f"arm must be one of {SEG_ARMS}, got {arm!r}"
        assert len(box_mix) == 3 and min(box_mix) >= 0 and abs(sum(box_mix) - 1.0) < 1e-6, f"box_mix must be 3 shares summing to 1, got {box_mix}"
        assert box_mix[0] + box_mix[1] > 0, f"box_mix {box_mix} has no object items"
        self.cache_dir = cache_dir
        self.split = split
        self.arm = arm
        self.roi_arm = 'raw' if arm == 'rgb' else arm     # rgb control: matched with the raw arm's detector
        self.box_mix = tuple(float(v) for v in box_mix)
        self.p_gt = self.box_mix[0] / (self.box_mix[0] + self.box_mix[1])   # 0.5 / 0.9 = 5/9
        self.gt_jitter = gt_jitter
        self.roi_margin = roi_margin
        self.roi_min = roi_min
        self.canvas = canvas
        self.scale_aug = scale_aug
        self.gain_aug = gain_aug
        self.train = train
        self.seed = seed
        self._rng = random.Random(seed) if seed is not None else None
        self._rng_key = None

        p = os.path.join(cache_dir, INDEX_NAME)
        assert os.path.exists(p), f"{p} missing; build the crop cache first (data_loader.roi_crops.build_crop_cache)"
        with open(p) as f:
            index = json.load(f)
        assert self.roi_arm in index['arms'], f"crop cache {cache_dir} has no ROIs of arm {self.roi_arm!r} (has {index['arms']})"
        # the cached object windows were grown so that every jittered expanded GT ROI fits (spec §3: x 2.0 >= x 1.5 x 1.3)
        assert roi_margin * (1 + 2 * gt_jitter) <= index['grow'] + 1e-9 and roi_min <= index['min_side'], \
            f"roi_margin {roi_margin} x (1 + 2 x {gt_jitter}) / roi_min {roi_min} exceed the cache's grow {index['grow']} / min_side {index['min_side']}"
        self.H, self.W = index['frame_hw']
        self.windows = [w for w in index['windows'] if w['split'] == split]
        self.obj_windows = [w for w in self.windows if w['kind'] == 'object']
        assert self.obj_windows, f"no {split} object windows in {cache_dir}"
        # false-positive ROIs of this arm: (window, roi entry); one fp window can carry several arms' entries
        self.fp_rois = [(w, e) for w in self.windows if w['kind'] == 'fp' for e in w['rois'][self.roi_arm]]
        if train:
            self.n_fp = n_fp_items(len(self.obj_windows), self.box_mix[2]) if self.fp_rois else 0
            self.items = None
        else:
            self.n_fp = 0
            self.items = []                                   # (window, source, roi entry or None), fixed order
            for w in self.obj_windows:
                self.items.append((w, 'gt', None))
                self.items += [(w, 'det', e) for e in w['rois'][self.roi_arm]]

    def __len__(self):
        return len(self.obj_windows) + self.n_fp if self.train else len(self.items)

    @property
    def rng(self):
        '''Same scheme as HyperCOD_data.rng: the random module (re-seeded per worker by PyTorch) or a seeded stream.'''
        if self._rng is None:
            return random
        info = torch.utils.data.get_worker_info()
        if info is not None and self._rng_key != info.seed:
            self._rng = random.Random(f"{self.seed}:{info.seed}")
            self._rng_key = info.seed
        return self._rng

    @staticmethod
    def object_box(w):
        '''GT box (frame px) of the window's own object.'''
        return next(o['box'] for o in w['objects'] if o['id'] == w['object'])

    def choose(self, idx):
        '''(window, source, roi entry or None) of item idx; random in training, fixed in validation.'''
        if not self.train:
            return self.items[idx]
        if idx >= len(self.obj_windows):                          # false-positive item, uniform over the arm's fp ROIs
            w, e = self.fp_rois[self.rng.randrange(len(self.fp_rois))]
            return w, 'fp', e
        w = self.obj_windows[idx]
        dets = w['rois'][self.roi_arm]
        if self.rng.random() < self.p_gt or not dets:             # expanded GT (also the fallback without a matched ROI)
            return w, 'gt', None
        return w, 'det', dets[self.rng.randrange(len(dets))]

    def item_boxes(self, w, source, e):
        '''(roi float xyxy, box float xyxy) in frame px: the ROI cut out and the pre-expansion box of the box channel.'''
        if source == 'gt':
            gt_box = self.object_box(w)
            box = jitter_gt_box(gt_box, self.gt_jitter, self.rng, self.H, self.W) if self.train else np.asarray(gt_box, np.float32)
            return expand_box(box, self.roi_margin, self.roi_min, self.H, self.W), box
        return np.asarray(e['roi'], np.float32), np.asarray(e['box'], np.float32)

    def __getitem__(self, idx):
        w, source, e = self.choose(idx)
        roi_f, box = self.item_boxes(w, source, e)
        wx1, wy1, wx2, wy2 = w['window']
        roi = pixel_box(roi_f, self.H, self.W)
        bpx = pixel_box(box, self.H, self.W)
        if source == 'gt':
            # the window was grown for the jittered ROI (asserted in __init__); clip only absorbs float rounding
            roi = (max(roi[0], wx1), max(roi[1], wy1), min(roi[2], wx2), min(roi[3], wy2))
        assert inside(roi, w['window']) and inside(bpx, roi), \
            f"frame {w['frame']} ({w['file']}): ROI {roi} / box {bpx} outside window {w['window']}"
        x1, y1, x2, y2 = roi
        h, wd = y2 - y1, x2 - x1

        # spec §9: a missing window raises (with the frame), never skips
        p_img, p_gt = os.path.join(self.cache_dir, w['file']), os.path.join(self.cache_dir, w['gt_file'])
        assert os.path.exists(p_img) and os.path.exists(p_gt), \
            f"window {w['file']} / {w['gt_file']} of frame {w['frame']} missing in {self.cache_dir}"
        cube = load_window(p_img)                                             # [n_bands, wh, ww] fp16, un-scaled
        gt = np.load(p_gt)                                                    # [wh, ww] bool
        crop = np.ascontiguousarray(cube[:, y1 - wy1:y2 - wy1, x1 - wx1:x2 - wx1])   # [n_bands, h, w] fp16
        # p99 scaling in fp16 through torch, exactly HyperCOD_data's rounding (so test frames match training crops)
        crop = scale_block(crop, w['scale'])                                  # [n_bands, h, w] fp16
        m = gt[y1 - wy1:y2 - wy1, x1 - wx1:x2 - wx1]                           # [h, w] bool, GT union inside the ROI
        obj_area = int(m.sum())

        s_aug = self.rng.uniform(*self.scale_aug) if self.train else 1.0
        img, valid, offset, s = place_on_canvas(crop, self.canvas, s_aug)     # [C, c, c] fp16, [c, c] bool
        mask = place_on_canvas(m.astype(np.float32)[None], self.canvas, s_aug)[0]   # [1, c, c] float32 (soft if resized)
        box_canvas = box_to_canvas(box, roi, offset, (h, wd), s, self.canvas)
        box_map = rasterise_box(box_canvas, self.canvas)                      # [c, c] float32
        valid = valid.astype(np.float32)                                      # [c, c]
        aug = (False, False, 0)
        if self.train:
            aug = (self.rng.random() < 0.5, self.rng.random() < 0.5, self.rng.randrange(4))
            (img, mask, box_map, valid), box_canvas = flip_rot([img, mask, box_map, valid], box_canvas, *aug, self.canvas)
            if self.gain_aug > 0:
                g = np.array([self.rng.uniform(1 - self.gain_aug, 1 + self.gain_aug) for _ in range(img.shape[0])], np.float32)
                img = (img.astype(np.float32) * g[:, None, None]).astype(np.float16)   # [C, c, c], zero padding stays 0
        area = next((o['area'] for o in w['objects'] if o['id'] == w['object']), 0)
        meta = {'frame': w['frame'], 'roi': [float(v) for v in roi], 'box': [float(v) for v in box],
                'box_canvas': [float(v) for v in box_canvas], 'source': source, 'offset': (int(offset[0]), int(offset[1])),
                's': float(s), 'roi_hw': (int(h), int(wd)), 'obj_area': obj_area, 'object': int(w['object']),
                'area': int(area), 'aug': aug}
        return (np.ascontiguousarray(img, dtype=np.float16), np.ascontiguousarray(mask, dtype=np.float32),
                box_map[None], valid[None], meta)


def seg_collate_fn(batch):
    '''
    HyperCOD_roi items -> {"img" [B, C, c, c] fp16, "mask" / "box_map" / "valid" [B, 1, c, c] float32,
    "box_xyxy" [B, 4] float32 canvas px (the prompt box of SAM2BoxSeg), "meta": list of dicts}.
    '''
    img, mask, box_map, valid, meta = list(zip(*batch))
    return {'img': torch.from_numpy(np.stack(img, axis=0)),                      # [B, C, c, c] fp16
            'mask': torch.from_numpy(np.stack(mask, axis=0)),                    # [B, 1, c, c]
            'box_map': torch.from_numpy(np.stack(box_map, axis=0)),              # [B, 1, c, c]
            'valid': torch.from_numpy(np.stack(valid, axis=0)),                  # [B, 1, c, c]
            'box_xyxy': torch.tensor([m['box_canvas'] for m in meta], dtype=torch.float32).reshape(-1, 4),   # [B, 4]
            'meta': list(meta)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python -m pytest tests/test_roi_crops.py -v`
Expected: PASS (14 passed: 5 from Task 3 + 9 here).

- [ ] **Step 5: Review Focus tests: several objects in one ROI, a frame-corner ROI, and training items through DataLoader workers (write them, run them)**

Append to `tests/test_roi_crops.py`:

```python


def _crop_cache_gt(synthetic_root, tmp_path, gt, min_side):
    '''Crop cache whose val frame '10' has the given GT and no detector ROI in any arm; train frame '3' as in the fixture.'''
    root, _, _ = synthetic_root
    Image.fromarray(np.stack([gt.astype(np.uint8) * 255] * 3, axis=-1)).save(root / 'train' / 'GT' / '10.png')
    build_cube_cache(str(root), 'train', num_workers=0)
    split_file = tmp_path / 'val.json'
    split_file.write_text(json.dumps({'seed': 0, 'n_val': 1, 'val_ids': ['10']}))
    roi_files = {}
    for arm in ('raw', 'ec10', 'ec24'):
        for split, frame in (('train', '3'), ('val', '10')):
            p = tmp_path / f'rois_{arm}_{split}.json'
            p.write_text(json.dumps({frame: {'rois': [], 'boxes': []}}))
            roi_files.setdefault(arm, {})[split] = str(p)
    out = tmp_path / 'crops_gt'
    build_crop_cache(str(root), str(out), roi_files, split_file=str(split_file), grow=2.0, min_side=min_side, min_area=10,
                     num_workers=0)
    return out


def test_roi_with_two_objects_and_a_frame_corner_roi(synthetic_root, tmp_path):
    gt = np.zeros((H, W), dtype=bool)
    gt[10:16, 20:26] = True          # A, 36 px
    gt[18:24, 22:30] = True          # B, 48 px: inside A's 24 px oracle ROI, and A inside B's
    gt[42:48, 0:6] = True            # C, 36 px, in the bottom-left corner of the frame
    out = _crop_cache_gt(synthetic_root, tmp_path, gt, min_side=24)
    ds = HyperCOD_roi(str(out), 'val', 'raw', train=False, roi_margin=1.5, roi_min=24, canvas=32)
    items = [ds[i] for i in range(len(ds))]
    assert len(items) == 3 and all(m['source'] == 'gt' for *_, m in items)
    by_area = {}
    for img, mask, box_map, valid, meta in items:
        x1, y1, x2, y2 = [int(v) for v in meta['roi']]
        assert 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H
        # target = every GT pixel inside the ROI (the union), size bucket = the object's own area
        assert mask.sum() == meta['obj_area'] == int(gt[y1:y2, x1:x2].sum())
        by_area.setdefault(meta['area'], []).append(meta['obj_area'])
        assert not mask[valid == 0].any() and not box_map[valid == 0].any() and not img[:, valid[0] == 0].any()
        np.testing.assert_array_equal(box_map[0], rasterise_box(meta['box_canvas'], 32))
    assert sorted(by_area) == [36, 48] and sorted(by_area[36]) == [36, 84] and by_area[48] == [84]
    corner = next(m for *_, m in items if m['obj_area'] == 36)
    # clipped at the frame edge (expand_box does not shift), so smaller than roi_min and zero-padded on the canvas
    assert corner['roi'][0] == 0 and corner['roi'][3] == H and corner['roi_hw'] == (15, 15) and corner['s'] == 1.0


def test_training_items_through_dataloader_workers(synthetic_root, tmp_path):
    '''
    Review Focus (with Task 3's test_build_crop_cache_with_workers_matches_serial): training items through DataLoader
    workers are always valid, and the seeded per-worker augmentation is not frozen across workers / epochs.
    '''
    root, out, _, _, _ = _crop_cache(synthetic_root, tmp_path)
    ds = HyperCOD_roi(str(out), 'train', 'raw', seed=0, roi_margin=1.5, roi_min=0, canvas=32)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, collate_fn=seg_collate_fn)
    seen = []
    for _ in range(10):
        for batch in loader:
            m = batch['meta'][0]
            assert batch['img'].shape == (1, 133, 32, 32) and not batch['mask'][batch['valid'] == 0].any()
            seen.append(tuple(m['aug']) + (m['source'], m['object']))
    assert len(seen) == 10 * len(ds) and len(set(seen)) > 2
```

Run: `OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python -m pytest tests/test_roi_crops.py -k "two_objects or dataloader_workers" -v`
Expected: PASS with the implementation of Step 3 (guards for the multi-object union, the clipped corner ROI and the worker RNG; a failure is a bug in `HyperCOD_roi`, not in the test).

- [ ] **Step 6: Run the file and the full suite**

Run: `OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python -m pytest tests/test_roi_crops.py -v`
Expected: PASS (16 passed).

Then: `OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python -m pytest tests/ -q`. It must stay green: the previous count plus 11 new tests.

- [ ] **Step 7: Commit and push**

```bash
git add data_loader/roi_crops.py tests/test_roi_crops.py
git commit -m "feat(seg): HyperCOD_roi - ROI items on a fixed canvas with box channel, 50/40/10 box mix and paste-back geometry

Cached windows are cut to the item's ROI, p99-scaled with HyperCOD_data's fp16 rounding, centred on the canvas
(downscaled only when larger) and augmented with flips, rot90, scale and per-channel gain; validation is deterministic.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QB4DScDwiA2DkVpbDopffV"
git push origin worktree-det
git push origin worktree-det:master
```

**Verification done while drafting** (scratch copy of the repo; no repo files were modified): Tasks 3 and 4 were replayed in order; the Task 3 tests passed alone and the Task 4 suite passed with the full suite green. Real-data smoke test: `build_crop_cache` on 2 real val frames (`splits=('val',)`, `num_workers=2`, a fake raw ROI file) produced 4 windows (2 object, 2 fp) totalling 0.19 GB in 2.3 s; `HyperCOD_roi` at canvas 512 then gave `img [4, 133, 512, 512]` fp16 after collation; the oracle item matches `HyperCOD_data(norm='p99', out_dtype='float16')` frame values bit for bit; the mask round trip through `canvas_to_roi` was exact; peak RSS 2.0 GB.

---

### Task 5: Segmentation metrics (`SegMetrics`, `pool`, `bootstrap_seg`)

**Files:**
- Create: `train_eval/seg_metrics.py`
- Test: `tests/test_seg_metrics.py`

**Interfaces:**
- Consumes: `py_sod_metrics` 1.6.2 (installed by Task 7's `setup_third_party.sh`: `pip install --no-deps pysodmetrics==1.6.2` + `pip install scikit-image==0.26.0 scikit-learn==1.9.1`; Task 5 runs before Task 7, so Step 0 installs exactly these pins by hand). Uses `Smeasure(alpha=0.5)`, `Emeasure()`, `WeightedFmeasure(beta=1)`, `MAE()`, `FmeasureV2({'fm': FmeasureHandler(with_dynamic=True, with_adaptive=True, beta=0.3)})`, one `step()` per image, values read back with `[-1]`.
- Produces:
  - `class SegMetrics(size_edges=(2000, 20000), minmax=False)`.
    - `update(pred [h,w] float, gt [h,w] bool, area=None, frame=None, key=None) -> row dict`.
    - `update_fp(pred [h,w], frame=None, key=None) -> row dict`.
    - `summary() -> dict`.
    - `.per_image` (list of row dicts).
  - `summary()` keys:
    - `S, E_mean, E_max, E_adp, Fw, F_adp, F_mean, F_max, MAE, IoU, n`;
    - the same keys suffixed `_small` / `_medium` / `_large`;
    - `fp_false_mask_rate`, `n_fp`.
    - Empty groups give `nan` and `n_* = 0`.
  - `pool(per_image, size_edges=(2000, 20000)) -> summary dict` (identical to `summary()`).
  - `bootstrap_seg(runs_a, runs_b, keys, n=2000, seed=0, size_edges=(2000, 20000)) -> {key: {"diff", "lo", "hi", "p_gt0"}}`.
  - `image_scores(pred, gt, minmax=False) -> dict` and `size_bucket(area, size_edges) -> 'small'|'medium'|'large'`.
  - `SUMMARY_KEYS`.
  - Row schema:
    - object rows: `{"kind": "obj", "frame", "key", "area", "S", "E_adp", "Fw", "F_adp", "MAE", "IoU", "E_curve" [256] f64, "F_curve" [256] f64}`;
    - fp rows: `{"kind": "fp", "frame", "key", "false_mask", "max_prob"}`.
    - Curve index i is threshold 255 - i.
  - Conventions:
    - `minmax=False` (development metric): float64 probabilities, `normalize=False`.
    - `minmax=True` reproduces the SAM2-UNet / HyperCOD Table 2 protocol (per-image min-max to uint8, `normalize=True`).
    - IoU is always on the raw probability > 0.5.
    - `E_max` / `E_mean` / `F_max` / `F_mean` come from the image-averaged curve.
    - `bootstrap_seg` resamples FRAMES when every row has a `frame`, otherwise rows. It asserts that every run holds the same `(kind, frame, key)` rows, so only oracle ROI rows and full-frame rows are pairable across arms.
  - Callers:
    - T11 `evaluate` calls `update(prob_roi, gt_roi)` and `update_fp(...)` for fp items.
    - T13 passes `frame` / `key` on every row (frame bootstrap), keeps `per_image` lists for `per_image.pkl` and `bootstrap_seg`, and uses a second `SegMetrics(minmax=True)` for the paper metric set (MAE, E_mean, S, F_adp).

- [ ] **Step 0: Install the metric package (pins of Task 7, needed before Task 7 runs)**

Run:
```bash
/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python -m pip install -q --no-deps pysodmetrics==1.6.2
/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python -m pip install -q scikit-image==0.26.0 scikit-learn==1.9.1
/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python -c "import py_sod_metrics, numpy, cv2, torch; print(numpy.__version__, cv2.__version__, torch.__version__)"
```
Expected: `2.5.2 5.0.0 2.11.0+cu128` (numpy / opencv / torch unchanged; `pip check` will report pysodmetrics' `numpy<2.3.5` and `opencv-python-headless` pins as unmet, which is expected). Task 7's setup script repeats these installs idempotently.

- [ ] **Step 1: Write the failing test**

`tests/test_seg_metrics.py`:
```python
import math
import warnings
import numpy as np
import pytest

psm = pytest.importorskip('py_sod_metrics')
from train_eval.seg_metrics import SegMetrics, pool, bootstrap_seg, image_scores, size_bucket, SUMMARY_KEYS


def _cases(seed=0, n=8):
    '''Hand-made (prob, gt) pairs: noisy rectangles of different sizes, a low-confidence map, a 1-pixel object.'''
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        h, w = int(rng.integers(24, 80)), int(rng.integers(24, 80))
        gt = np.zeros((h, w), bool)
        y0, x0 = int(rng.integers(0, h // 2)), int(rng.integers(0, w // 2))
        gt[y0:y0 + int(rng.integers(2, h // 2)), x0:x0 + int(rng.integers(2, w // 2))] = True
        prob = np.clip(gt * rng.uniform(0.3, 1.0) + rng.normal(0, 0.2, (h, w)), 0, 1)
        out.append((prob, gt))
    g = np.zeros((40, 40), bool); g[10:20, 10:20] = True
    out.append((0.2 * g + 0.05 * rng.uniform(0, 1, (40, 40)), g))           # low-confidence map, max 0.25
    g1 = np.zeros((30, 30), bool); g1[4, 7] = True
    out.append((0.5 * rng.uniform(0, 1, (30, 30)), g1))                      # 1-pixel object
    return out


def _reference(cases, u8):
    '''py_sod_metrics on its own: one object per metric over all images, get_results().'''
    sm, em, wfm, mae = psm.Smeasure(), psm.Emeasure(), psm.WeightedFmeasure(), psm.MAE()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fm = psm.Fmeasure()                                                  # the deprecated class, as a second reference
    for prob, gt in cases:
        if u8:
            p = (prob - prob.min()) / (prob.max() - prob.min() + 1e-8)
            pred_in, gt_in, norm = (p * 255).astype(np.uint8), gt.astype(np.uint8) * 255, True
        else:
            pred_in, gt_in, norm = prob.astype(np.float64), gt, False
        for m in (sm, em, wfm, mae, fm):
            m.step(pred_in, gt_in, normalize=norm)
    e, f = em.get_results()['em'], fm.get_results()['fm']
    return {'S': sm.get_results()['sm'], 'E_mean': e['curve'].mean(), 'E_max': e['curve'].max(), 'E_adp': e['adp'],
            'Fw': wfm.get_results()['wfm'], 'MAE': mae.get_results()['mae'], 'F_adp': f['adp'],
            'F_mean': f['curve'].mean(), 'F_max': f['curve'].max()}


@pytest.mark.parametrize('minmax', [False, True])
def test_matches_py_sod_metrics(minmax):
    cases = _cases()
    met = SegMetrics(minmax=minmax)
    for prob, gt in cases:
        met.update(prob, gt)
    s, ref = met.summary(), _reference(cases, u8=minmax)
    for k, v in ref.items():
        assert s[k] == pytest.approx(v, abs=1e-12), f"{k}: {s[k]} vs py_sod_metrics {v}"
    assert s['n'] == len(cases) and s['n_fp'] == 0 and math.isnan(s['fp_false_mask_rate'])
    assert set(SUMMARY_KEYS) <= set(s) and all(f'{k}_{b}' in s for k in SUMMARY_KEYS for b in ('small', 'medium', 'large'))


def test_iou_on_raw_probability():
    gt = np.zeros((20, 20), bool); gt[5:15, 5:15] = True                     # 100 px
    pred = np.zeros((20, 20), np.float32); pred[5:15, 10:20] = 0.9           # 50 px inside, 50 px outside
    r = image_scores(pred, gt)
    assert r['IoU'] == pytest.approx(50 / 150)
    low = 0.4 * gt.astype(np.float32)                                        # never above 0.5: IoU 0 in both conventions
    assert image_scores(low, gt)['IoU'] == 0.0 and image_scores(low, gt, minmax=True)['IoU'] == 0.0
    assert image_scores(low, gt, minmax=True)['S'] > image_scores(low, gt)['S']   # min-max inflates a low-confidence map
    assert image_scores(np.zeros((20, 20)), np.zeros((20, 20), bool))['IoU'] == 0.0


def test_input_dtypes_and_clipping():
    gt = np.zeros((16, 16), bool); gt[4:12, 4:12] = True
    p = np.clip(gt + np.random.default_rng(0).normal(0, 0.1, (16, 16)), 0, 1)
    a = image_scores(p.astype(np.float16), gt.astype(np.uint8))              # fp16 pred and 0/1 uint8 gt accepted
    b = image_scores(p.astype(np.float16).astype(np.float64), gt)
    assert a['S'] == b['S'] and a['IoU'] == b['IoU']
    c = image_scores(p * 1.01, gt)                                           # slightly above 1 is clipped, not rejected
    assert np.isfinite(c['S'])
    with pytest.raises(AssertionError):
        image_scores(np.zeros((16, 15)), gt)


def test_size_buckets_and_area_override():
    H = W = 200
    met = SegMetrics(size_edges=(2000, 20000))
    rows = []
    for (h, w) in [(10, 10), (50, 100), (150, 200)]:                         # 100, 5000, 30000 px
        gt = np.zeros((H, W), bool); gt[:h, :w] = True
        rows.append(met.update(np.clip(gt * 0.8 + 0.05, 0, 1), gt))
    s = met.summary()
    assert (s['n_small'], s['n_medium'], s['n_large']) == (1, 1, 1)
    assert s['S_small'] == rows[0]['S'] and s['S_medium'] == rows[1]['S'] and s['S_large'] == rows[2]['S']
    assert s['IoU_large'] == pytest.approx(1.0)
    assert size_bucket(1999) == 'small' and size_bucket(2000) == 'medium' and size_bucket(20000) == 'medium' and size_bucket(20001) == 'large'
    # area override: the caller's object area decides the bucket, not the GT pixels of the crop
    met2 = SegMetrics()
    gt = np.zeros((40, 40), bool); gt[0:10, 0:10] = True
    met2.update(gt.astype(np.float32), gt, area=25000)
    s2 = met2.summary()
    assert s2['n_large'] == 1 and s2['n_small'] == 0 and math.isnan(s2['S_small'])


def test_false_mask_rate_excluded_from_scores():
    met = SegMetrics()
    gt = np.zeros((20, 20), bool); gt[5:10, 5:10] = True
    met.update(gt.astype(np.float32), gt)
    s_obj = met.summary()
    met.update_fp(np.zeros((12, 9)))
    met.update_fp(np.full((12, 9), 0.5))                                     # exactly 0.5 is not a mask pixel
    f = np.zeros((12, 9)); f[3, 3] = 0.51
    met.update_fp(f)
    s = met.summary()
    assert s['n_fp'] == 3 and s['fp_false_mask_rate'] == pytest.approx(1 / 3)
    assert s['n'] == 1 and all(s[k] == s_obj[k] for k in SUMMARY_KEYS)      # fp rows never touch S / E / F / IoU


def test_pool_reproduces_summary_and_curve_averaging():
    met = SegMetrics()
    for i, (prob, gt) in enumerate(_cases(seed=1)):
        met.update(prob, gt, frame=str(i))
    met.update_fp(np.full((8, 8), 0.7), frame='0')
    s, p = met.summary(), pool(met.per_image)
    assert set(s) == set(p)
    for k in s:
        assert (math.isnan(s[k]) and math.isnan(p[k])) or s[k] == p[k], k
    rows = [r for r in met.per_image if r['kind'] == 'obj']
    assert all(r['E_curve'].shape == (256,) and r['F_curve'].shape == (256,) for r in rows)
    # E_max is the max of the image-averaged curve, never larger than the mean of per-image maxima
    assert s['E_max'] == pytest.approx(np.mean([r['E_curve'] for r in rows], axis=0).max())
    assert s['E_max'] <= np.mean([r['E_curve'].max() for r in rows]) + 1e-12
    # pooling two halves together equals one object updated with both
    half = len(met.per_image) // 2
    q = pool(met.per_image[:half] + met.per_image[half:])
    assert q['S'] == s['S'] and q['E_max'] == s['E_max']


def _run(seed, noise, frames=6, rois_per_frame=2):
    '''One "run": per frame a few ROI rows (frame, key) with predictions of a given noise level.'''
    rng = np.random.default_rng(seed)
    met = SegMetrics()
    for f in range(frames):
        for k in range(rois_per_frame):
            gt = np.zeros((24, 24), bool); gt[4 + k:14 + k, 6:16] = True
            met.update(np.clip(gt + rng.normal(0, noise, gt.shape), 0, 1), gt, frame=str(f), key=k)
    return met.per_image


def test_bootstrap_seg_paired_frames():
    good = [_run(s, 0.05) for s in range(2)]
    bad = [_run(10 + s, 0.6) for s in range(2)]
    keys = ['S', 'Fw', 'E_max', 'IoU']
    same = bootstrap_seg(good, good, keys, n=50, seed=0)
    assert all(same[k]['diff'] == 0 and same[k]['lo'] == 0 and same[k]['hi'] == 0 for k in keys)
    ci = bootstrap_seg(good, bad, keys, n=200, seed=0)
    for k in keys:
        assert ci[k]['diff'] > 0 and ci[k]['lo'] > 0 and ci[k]['lo'] <= ci[k]['diff'] <= ci[k]['hi'] and ci[k]['p_gt0'] == 1.0
    # point difference = difference of run-averaged pooled summaries
    exp = np.mean([pool(r)['S'] for r in good]) - np.mean([pool(r)['S'] for r in bad])
    assert ci['S']['diff'] == pytest.approx(exp)
    # same seed -> same draws
    assert bootstrap_seg(good, bad, keys, n=50, seed=3) == bootstrap_seg(good, bad, keys, n=50, seed=3)
    # unpaired rows (a different ROI set) are refused
    with pytest.raises(AssertionError):
        bootstrap_seg(good, [_run(0, 0.05, rois_per_frame=3)], keys, n=10)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python -m pytest tests/test_seg_metrics.py -v`
Expected: FAIL with a collection error: `ModuleNotFoundError: No module named 'train_eval.seg_metrics'`. (Without Step 0 the whole file is reported as SKIPPED, `could not import 'py_sod_metrics'`: do Step 0 first.)

- [ ] **Step 3: Write minimal implementation**

`train_eval/seg_metrics.py`:
```python
'''
Stage-2 segmentation metrics (spec §5.5, §8) on top of py_sod_metrics (PySODMetrics 1.6.2, MIT).

Every prediction is scored once per image (an ROI crop at native size, or a full 1680 x 1240 frame) and kept as one
row of per_image; summary() and the module-level pool() average those rows, so a split can be re-pooled image by image
(size buckets, frame bootstrap) and always gives the same numbers as one SegMetrics updated with those images in order.

Metrics per image (alpha / beta as in the COD literature):
  S       S-measure, alpha = 0.5                                     (py_sod_metrics.Smeasure)
  E_*     E-measure: 256-threshold curve, adaptive value             (py_sod_metrics.Emeasure)
  Fw      weighted F-measure, beta = 1                               (py_sod_metrics.WeightedFmeasure)
  F_*     F-measure, beta^2 = 0.3: 256-threshold curve, adaptive     (py_sod_metrics.FmeasureV2 + FmeasureHandler, which
          gives the same numbers as the deprecated Fmeasure without its warning on every construction)
  MAE     mean |pred - gt|                                           (py_sod_metrics.MAE)
  IoU     IoU of the RAW probability binarised at 0.5 (computed here, never min-max normalised; 0 when pred and GT are
          both empty, the py_sod_metrics divide convention)
Pooling: S, Fw, MAE, E_adp, F_adp and IoU are plain means over images. E_mean / E_max (and F_mean / F_max) are the mean /
max over thresholds of the curve AVERAGED OVER IMAGES first - the mean of per-image maxima is a different, larger number
(0.91 vs 0.81 on the recon test set) - so every row keeps its two [256] curves. Curve index i <-> threshold 255 - i on
floor(pred * 255) with >= (index 0 = threshold 255, index 255 = every pixel foreground).

Input convention (minmax):
  False (default, development metric): pred float64 clipped to [0, 1], gt bool, step(..., normalize=False): the
        probabilities are scored as they are.
  True  (the SAM2-UNet / HyperCOD Table 2 protocol): per-image min-max -> * 255 -> uint8, gt * 255 uint8,
        step(..., normalize=True) (py_sod_metrics min-maxes again); IoU stays on the raw probability. This inflates
        low-confidence maps (S 0.80 vs 0.64 on a map whose maximum is 0.2), so it is only for the paper comparison.

False-positive ROIs (GT empty inside the ROI) have no meaningful S / E / Fw, so they never enter those means; update_fp()
records whether any pixel is > 0.5 and summary() reports the share (fp_false_mask_rate) and their number (n_fp).
Size buckets follow the GT area of each row (small < 2000 px, medium 2000-20000, large > 20000 by default); the caller may
pass the object's own frame-level area, since a crop's GT union can include part of a neighbour.
'''
import numpy as np
import py_sod_metrics as psm

SCORE_KEYS = ('S', 'E_adp', 'Fw', 'F_adp', 'MAE', 'IoU')                  # per-image scalars, pooled by plain mean
SUMMARY_KEYS = ('S', 'E_mean', 'E_max', 'E_adp', 'Fw', 'F_adp', 'F_mean', 'F_max', 'MAE', 'IoU')
BUCKETS = ('small', 'medium', 'large')
N_THRESHOLDS = 256


def _to_pred(pred):
    '''Prediction as float64 in [0, 1] (py_sod_metrics with normalize=False rejects fp16 and values outside [0, 1]).'''
    pred = np.asarray(pred, dtype=np.float64)
    assert pred.ndim == 2, f"pred must be [h, w], got shape {pred.shape}"
    assert np.isfinite(pred).all(), f"pred has {int((~np.isfinite(pred)).sum())} non-finite values"
    return np.clip(pred, 0.0, 1.0)


def image_scores(pred, gt, minmax=False):
    '''
    All metrics of one image: pred [h, w] float in [0, 1] (any float dtype), gt [h, w] bool.
    Returns {S, E_adp, Fw, F_adp, MAE, IoU (floats), E_curve, F_curve ([256] float64)}.
    '''
    pred = _to_pred(pred)
    gt = np.asarray(gt).astype(bool)
    assert pred.shape == gt.shape, f"pred {pred.shape} and gt {gt.shape} differ"
    if minmax:
        # SAM2-UNet test.py: per-image min-max, saved as uint8 PNG, read back and scored with normalize=True
        p = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
        pred_in, gt_in, norm = (p * 255).astype(np.uint8), gt.astype(np.uint8) * 255, True
    else:
        pred_in, gt_in, norm = pred, gt, False
    # fresh objects per image: each step() appends one value per metric, read back with [-1] (identical to reusing one
    # object, verified in recon); Emeasure must go through step(), which sets gt_fg_numel / gt_size before cal_*
    sm, em, wfm, mae = psm.Smeasure(alpha=0.5), psm.Emeasure(), psm.WeightedFmeasure(beta=1), psm.MAE()
    fm_handler = psm.FmeasureHandler(with_dynamic=True, with_adaptive=True, beta=0.3)
    fm = psm.FmeasureV2(metric_handlers={'fm': fm_handler})
    for m in (sm, em, wfm, mae, fm):
        m.step(pred_in, gt_in, normalize=norm)
    # IoU at 0.5 on the RAW probability in both conventions
    b = pred > 0.5
    inter, union = np.count_nonzero(b & gt), np.count_nonzero(b | gt)
    return {'S': float(sm.sms[-1]), 'E_adp': float(em.adaptive_ems[-1]), 'Fw': float(wfm.weighted_fms[-1]),
            'F_adp': float(fm_handler.adaptive_results[-1]), 'MAE': float(mae.maes[-1]),
            'IoU': float(inter / union) if union > 0 else 0.0,
            'E_curve': np.asarray(em.changeable_ems[-1], dtype=np.float64).reshape(N_THRESHOLDS),     # [256]
            'F_curve': np.asarray(fm_handler.dynamic_results[-1], dtype=np.float64).reshape(N_THRESHOLDS)}


def size_bucket(area, size_edges=(2000, 20000)):
    '''small: area < edges[0]; medium: edges[0] <= area <= edges[1]; large: area > edges[1].'''
    return 'small' if area < size_edges[0] else ('medium' if area <= size_edges[1] else 'large')


def _as_arrays(rows):
    '''Columns of a per_image list: object rows (kind 'obj') and false-positive rows (kind 'fp') separately.'''
    obj = [r for r in rows if r['kind'] == 'obj']
    fp = [r for r in rows if r['kind'] == 'fp']
    a = {k: np.array([r[k] for r in obj], dtype=np.float64) for k in SCORE_KEYS}
    a['area'] = np.array([r['area'] for r in obj], dtype=np.float64)
    a['E_curve'] = np.stack([r['E_curve'] for r in obj]) if obj else np.zeros((0, N_THRESHOLDS))   # [n, 256]
    a['F_curve'] = np.stack([r['F_curve'] for r in obj]) if obj else np.zeros((0, N_THRESHOLDS))   # [n, 256]
    a['false_mask'] = np.array([r['false_mask'] for r in fp], dtype=np.float64)                     # [m]
    return a


def _stats(a, idx):
    '''Pooled metrics of the object rows idx (a repeated index counts twice, as the bootstrap needs); nan when empty.'''
    if len(idx) == 0:
        return {**{k: float('nan') for k in SUMMARY_KEYS}, 'n': 0}
    e = a['E_curve'][idx].mean(axis=0)                                     # [256] curve averaged over images
    f = a['F_curve'][idx].mean(axis=0)                                     # [256]
    out = {k: float(a[k][idx].mean()) for k in SCORE_KEYS}
    out.update(E_mean=float(e.mean()), E_max=float(e.max()), F_mean=float(f.mean()), F_max=float(f.max()), n=int(len(idx)))
    return {k: out[k] for k in (*SUMMARY_KEYS, 'n')}


def _pool_arrays(a, obj_idx, fp_idx, size_edges):
    '''Summary of the object rows obj_idx and the false-positive rows fp_idx of the column dict a.'''
    obj_idx, fp_idx = np.asarray(obj_idx, dtype=int), np.asarray(fp_idx, dtype=int)
    out = _stats(a, obj_idx)
    area = a['area'][obj_idx]
    masks = {'small': area < size_edges[0], 'medium': (area >= size_edges[0]) & (area <= size_edges[1]),
             'large': area > size_edges[1]}
    for b in BUCKETS:
        out.update({f'{k}_{b}': v for k, v in _stats(a, obj_idx[masks[b]]).items()})
    out['n_fp'] = int(len(fp_idx))
    out['fp_false_mask_rate'] = float(a['false_mask'][fp_idx].mean()) if len(fp_idx) else float('nan')
    return out


def pool(per_image, size_edges=(2000, 20000)):
    '''Summary of a per_image list (SegMetrics.per_image, possibly several concatenated): identical to summary().'''
    a = _as_arrays(per_image)
    return _pool_arrays(a, np.arange(len(a['S'])), np.arange(len(a['false_mask'])), size_edges)


class SegMetrics(object):
    '''
    Accumulates per-image segmentation scores over a split.
      update(pred, gt, area=None, frame=None, key=None): one object image (an ROI crop, or a full frame); area = the GT
        area used for the size bucket (default gt.sum()); frame / key identify the row for the paired bootstrap
        (bootstrap_seg resamples frames and requires the same (frame, key) rows in every run compared).
      update_fp(pred, frame=None, key=None): one false-positive ROI (no GT inside); only the false-mask flag is kept.
      summary(): S, E_mean, E_max, E_adp, Fw, F_adp, F_mean, F_max, MAE, IoU, n, the same keys suffixed
        _small / _medium / _large, fp_false_mask_rate and n_fp.
    '''

    def __init__(self, size_edges=(2000, 20000), minmax=False):
        assert len(size_edges) == 2 and size_edges[0] <= size_edges[1], f"size_edges must be (lo, hi), got {size_edges}"
        self.size_edges = (float(size_edges[0]), float(size_edges[1]))
        self.minmax = bool(minmax)  # False: raw probabilities (development); True: SAM2-UNet / HyperCOD min-max protocol
        self.per_image = []         # one dict per update() / update_fp() call, in call order

    def update(self, pred, gt, area=None, frame=None, key=None):
        row = image_scores(pred, gt, self.minmax)
        row.update(kind='obj', frame=frame, key=key,
                   area=int(np.count_nonzero(gt)) if area is None else int(area))
        self.per_image.append(row)
        return row

    def update_fp(self, pred, frame=None, key=None):
        pred = _to_pred(pred)
        row = {'kind': 'fp', 'frame': frame, 'key': key, 'false_mask': bool((pred > 0.5).any()), 'max_prob': float(pred.max())}
        self.per_image.append(row)
        return row

    def summary(self):
        return pool(self.per_image, self.size_edges)


def _row_ids(rows):
    return [(r['kind'], r['frame'], r['key']) for r in rows]


def _units(rows):
    '''
    Resampling units of a per_image list: rows grouped by frame (cluster bootstrap: the ROIs of one frame are not
    independent) when every row has a frame, else every row is its own unit. Returns (obj_groups, fp_groups): per unit
    the indices into the object rows and into the false-positive rows (the order _as_arrays uses).
    '''
    by_frame = all(r['frame'] is not None for r in rows)
    order, obj_g, fp_g = [], {}, {}
    n_obj = n_fp = 0
    for j, r in enumerate(rows):
        u = r['frame'] if by_frame else j
        if u not in obj_g:
            order.append(u); obj_g[u], fp_g[u] = [], []
        if r['kind'] == 'obj':
            obj_g[u].append(n_obj); n_obj += 1
        else:
            fp_g[u].append(n_fp); n_fp += 1
    return [np.asarray(obj_g[u], dtype=int) for u in order], [np.asarray(fp_g[u], dtype=int) for u in order]


def bootstrap_seg(runs_a, runs_b, keys, n=2000, seed=0, size_edges=(2000, 20000)):
    '''
    Paired bootstrap of mean(metric over runs_a) - mean(metric over runs_b) (e.g. the 3 seeds of two arms).
    runs_* are lists of per_image lists that must hold the same rows (kind, frame, key) in the same order; frames are
    drawn with replacement (all rows of a drawn frame together; rows without a frame are drawn one by one), the same
    draw for every run, and each draw re-pools the curves (E_max is the max of the resampled mean curve).
    Returns {key: {'diff': point difference, 'lo' / 'hi': 2.5 / 97.5 percentiles, 'p_gt0': share of draws > 0}}.
    '''
    runs = list(runs_a) + list(runs_b)
    assert len(runs_a) and len(runs_b), "bootstrap_seg needs at least one run on each side"
    ref = _row_ids(runs[0])
    for r in runs[1:]:
        ids = _row_ids(r)
        assert ids == ref, (f"runs are not paired: {len(ids)} vs {len(ref)} rows, first difference at "
                            f"{next((i for i, (x, y) in enumerate(zip(ids, ref)) if x != y), min(len(ids), len(ref)))}")
    obj_g, fp_g = _units(runs[0])
    m = len(obj_g)
    arrays = [_as_arrays(r) for r in runs]
    na = len(runs_a)

    def side_means(stats):
        # nan-safe mean over the runs of one side (a bucket can be empty in a draw)
        out = {}
        for k in keys:
            va = [s[k] for s in stats[:na] if not np.isnan(s[k])]
            vb = [s[k] for s in stats[na:] if not np.isnan(s[k])]
            out[k] = (np.mean(va) - np.mean(vb)) if va and vb else np.nan
        return out

    full_obj, full_fp = np.arange(len(arrays[0]['S'])), np.arange(len(arrays[0]['false_mask']))
    point = side_means([_pool_arrays(a, full_obj, full_fp, size_edges) for a in arrays])
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in keys}
    for _ in range(n):
        idx = rng.integers(0, m, m)                                        # [m] units drawn with replacement, shared by all runs
        oi = np.concatenate([obj_g[i] for i in idx]) if m else np.zeros(0, int)
        fi = np.concatenate([fp_g[i] for i in idx]) if m else np.zeros(0, int)
        d = side_means([_pool_arrays(a, oi, fi, size_edges) for a in arrays])
        for k in keys:
            draws[k].append(d[k])
    out = {}
    for k in keys:
        v = np.asarray(draws[k], dtype=np.float64)
        ok = ~np.isnan(v)
        out[k] = dict(diff=float(point[k]), lo=float(np.percentile(v[ok], 2.5)) if ok.any() else float('nan'),
                      hi=float(np.percentile(v[ok], 97.5)) if ok.any() else float('nan'),
                      p_gt0=float(np.mean(v[ok] > 0)) if ok.any() else float('nan'))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python -m pytest tests/test_seg_metrics.py -v`
Expected: PASS (8 passed in about 1 s).
Then run the full suite: `OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python -m pytest tests/ -q`. It stays green.

- [ ] **Step 5: Commit and push**

```bash
git add train_eval/seg_metrics.py tests/test_seg_metrics.py
git commit -m "feat(seg): SegMetrics on py_sod_metrics - per-image rows, curve-averaged E/F max, size buckets, FP false-mask rate, frame bootstrap

Stage-2 development metric (raw probabilities) and the HyperCOD/SAM2-UNet min-max protocol from one per-image record that
pool() and bootstrap_seg() re-pool exactly (spec §5.5, §8).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QB4DScDwiA2DkVpbDopffV"
git push origin worktree-det
git push origin worktree-det:master
```

---

### Task 6: N-channel stem (`pseudo_rgb`, `fit_rgb_map`, `fold_stem`)

**Files:**
- Create: `models/seg_stem.py`
- Test: `tests/test_seg_stem.py`

**Interfaces:**
- Consumes:
  - `data_loader.ec_filter.WAVELENS_200` and `band_indices` (tests only);
  - a front end `nn.Module` mapping `[B, 133, h, w]` to `[B, N, h, w]` (T8-T10 `build_front_end`), only through its callers.
- Produces:
  - `IMAGENET_MEAN`, `IMAGENET_STD` (np float32 [3]) and `RGB_WINDOWS`;
  - `rgb_band_indices(wavelens) -> [idx_R, idx_G, idx_B]`;
  - `pseudo_rgb(x [B,133,h,w] tensor, wavelens np [133]) -> [B,3,h,w] float32 in [0,1]`;
  - `imagenet_normalize(rgb [B,3,h,w]) -> float32 [B,3,h,w]`;
  - `fit_rgb_map(z np [n,N], rgb_norm np [n,3]) -> (P np f32 [3,N], q np f32 [3])`;
  - `fold_stem(conv nn.Conv2d(3,...), P, q, n_extra=1) -> nn.Conv2d(N + n_extra, ...)`, with a bias, extra slices zero and conv unchanged;
  - `mean_stem(conv, n_in, n_extra=1) -> nn.Conv2d` (the spec §3 fallback; not wired to a flag, see Deviations).
  - The pixel sampling for the fit lives in Task 12 (`main_seg.fit_arm_rgb_map`, the only stem-fit sampler). The `rgb` arm's front end is `imagenet_normalize(pseudo_rgb(x, wavelens))`, so its fit gives P ≈ I and q ≈ 0.

- [ ] **Step 1: Write the failing test**

`tests/test_seg_stem.py`:
```python
import numpy as np
import torch
import torch.nn as nn
import pytest

from data_loader.ec_filter import WAVELENS_200, band_indices
from models.seg_stem import (IMAGENET_MEAN, IMAGENET_STD, rgb_band_indices, pseudo_rgb, imagenet_normalize, fit_rgb_map,
                             fold_stem, mean_stem)

WL = WAVELENS_200[band_indices((400.0, 800.0))]                           # [133] the Stage-1 band window


def test_band_windows():
    r, g, b = rgb_band_indices(WL)
    assert (len(r), len(g), len(b)) == (33, 33, 34)
    assert WL[r].min() >= 600 and WL[r].max() < 700 and WL[b].min() >= 400 and WL[b].max() < 500
    with pytest.raises(AssertionError):
        rgb_band_indices(np.linspace(650, 800, 20))                       # no band below 600 nm


def test_pseudo_rgb_window_means_and_clip():
    r, g, b = rgb_band_indices(WL)
    x = torch.zeros(2, 133, 4, 5, dtype=torch.float16)
    x[:, r] = 0.5; x[:, g] = 0.25; x[:, b] = 2.0                           # B is above 1 -> clipped
    x[1, r[0]] = 0.5 + 33 * 0.25                                          # one bright band shifts the R mean by 0.25
    out = pseudo_rgb(x, WL)
    assert out.shape == (2, 3, 4, 5) and out.dtype == torch.float32
    torch.testing.assert_close(out[0], torch.tensor([0.5, 0.25, 1.0]).view(3, 1, 1).expand(3, 4, 5))
    torch.testing.assert_close(out[1, 0], torch.full((4, 5), 0.75))
    n = imagenet_normalize(out)
    torch.testing.assert_close(n[0, :, 0, 0], torch.from_numpy((np.array([0.5, 0.25, 1.0], np.float32) - IMAGENET_MEAN) / IMAGENET_STD))
    with pytest.raises(AssertionError):
        pseudo_rgb(x[:, :100], WL)


def test_fit_rgb_map_recovers_linear_map():
    rng = np.random.default_rng(0)
    for N in (3, 10, 24):
        P0 = rng.normal(size=(3, N)).astype(np.float32); q0 = rng.normal(size=3).astype(np.float32)
        z = rng.normal(size=(5000, N))
        P, q = fit_rgb_map(z, z @ P0.T + q0)
        assert P.shape == (3, N) and q.shape == (3,) and P.dtype == np.float32
        np.testing.assert_allclose(P, P0, atol=1e-5); np.testing.assert_allclose(q, q0, atol=1e-5)
    # rank-deficient z (a duplicated channel, like the near-empty whitened ec24 channels): finite, and still exact on z
    z = rng.normal(size=(2000, 4)); z = np.concatenate([z, z[:, :1]], axis=1)
    rgb = z[:, :4] @ rng.normal(size=(4, 3)) + 0.3
    P, q = fit_rgb_map(z, rgb)
    assert np.isfinite(P).all() and np.allclose(z @ P.T + q, rgb, atol=1e-4)


@pytest.mark.parametrize('N', [3, 10, 24])
def test_fold_stem_reproduces_rgb_stem(N):
    torch.manual_seed(0)
    conv = nn.Conv2d(3, 16, 7, stride=4, padding=3, bias=True)           # the Hiera / PVTv2 stem geometry
    rng = np.random.default_rng(N)
    P = rng.normal(size=(3, N)).astype(np.float32); q = rng.normal(size=3).astype(np.float32)
    z = torch.randn(2, N, 64, 64)                                         # [B, N, h, w]
    rgb_norm = torch.einsum('kc,bchw->bkhw', torch.from_numpy(P), z) + torch.from_numpy(q).view(1, 3, 1, 1)
    w0 = conv.weight.detach().clone()
    new = fold_stem(conv, P, q, n_extra=1)
    assert new.in_channels == N + 1 and new.out_channels == 16 and new.kernel_size == (7, 7)
    assert new.stride == (4, 4) and new.padding == (3, 3) and new.weight.requires_grad and new.bias.requires_grad
    assert torch.equal(new.weight[:, N:], torch.zeros_like(new.weight[:, N:])) and torch.equal(conv.weight, w0)
    box = (torch.rand(2, 1, 64, 64) > 0.5).float()                        # the box channel has no effect at init
    ref, out = conv(rgb_norm), new(torch.cat([z, box], dim=1))           # [2, 16, 16, 16]
    # output row/col 0 is the only one whose 7x7 window (rows 4i-3..4i+3) reaches the zero padding at 64 px
    torch.testing.assert_close(out[:, :, 1:, 1:], ref[:, :, 1:, 1:], atol=1e-4, rtol=1e-4)
    assert not torch.allclose(out[:, :, 0, 0], ref[:, :, 0, 0], atol=1e-3)   # the border differs (folded bias adds q)
    out.sum().backward()
    assert new.weight.grad is not None and new.weight.grad[:, N:].abs().sum() > 0   # the box slice learns


def test_fold_stem_creates_bias_and_rejects_non_rgb():
    conv = nn.Conv2d(3, 8, 3, padding=1, bias=False)
    P, q = np.eye(3, dtype=np.float32), np.array([1.0, -2.0, 0.5], np.float32)
    new = fold_stem(conv, P, q, n_extra=2)
    assert new.in_channels == 5 and new.bias is not None
    exp = torch.einsum('okij,k->o', conv.weight.detach(), torch.from_numpy(q))
    torch.testing.assert_close(new.bias.detach(), exp)
    torch.testing.assert_close(new.weight[:, :3], conv.weight)            # P = I keeps the RGB kernel (the rgb arm)
    with pytest.raises(AssertionError):
        fold_stem(nn.Conv2d(4, 8, 3), P, q)
    with pytest.raises(AssertionError):
        fold_stem(conv, np.eye(4, dtype=np.float32), q)


def test_mean_stem_keeps_grey_response():
    torch.manual_seed(0)
    conv = nn.Conv2d(3, 8, 3, padding=1)
    new = mean_stem(conv, n_in=10, n_extra=1)
    grey = torch.full((1, 3, 6, 6), 0.4)
    x = torch.cat([torch.full((1, 10, 6, 6), 0.4), torch.rand(1, 1, 6, 6)], dim=1)
    torch.testing.assert_close(new(x), conv(grey), atol=1e-5, rtol=1e-5)
    assert torch.equal(new.weight[:, 10:], torch.zeros_like(new.weight[:, 10:]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python -m pytest tests/test_seg_stem.py -v`
Expected: FAIL with a collection error: `ModuleNotFoundError: No module named 'models.seg_stem'`.

- [ ] **Step 3: Write minimal implementation**

`models/seg_stem.py`:
```python
'''
N-channel stems for the Stage-2 segmenters (spec §3 "regress-and-fold", §5.3).

Every pretrained segmenter starts from an RGB stem conv W_rgb [C_out, 3, k, k] that expects ImageNet-normalised RGB.
For an arm with N channels z (raw: 133 standardised bands; ec10 / ec24: the whitened readings; rgb: the normalised
pseudo-RGB render itself), a least-squares map rgb_norm ~ P z + q (P [3, N], q [3]) is fitted on training pixels and
folded into a new conv with N + n_extra inputs:
    W_N[o, c] = sum_k W_rgb[o, k] P[k, c]                      (per kernel tap)
    b_N[o]    = b_rgb[o] + sum_{k, i, j} W_rgb[o, k, i, j] q_k
    W_N[o, N:] = 0                                             (the box channel and any other extra input start silent)
so at initialisation the network sees the pseudo-RGB image (exactly, wherever the kernel does not touch the zero
padding: there the folded bias still adds q for the padded taps while the RGB conv saw zeros), and training adds the
spectral cues on top. The pseudo-RGB render is the same for every arm: mean of the p99-scaled bands in [600, 700) nm (R),
[500, 600) (G), [400, 500) (B), clipped to [0, 1]. The pixel sampling for the fit is main_seg.fit_arm_rgb_map.
'''
import numpy as np
import torch
import torch.nn as nn

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)   # SAM2 / PVTv2 / ZoomNeXt input normalisation
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
RGB_WINDOWS = ((600.0, 700.0), (500.0, 600.0), (400.0, 500.0))      # nm, [lo, hi) for R, G, B


def rgb_band_indices(wavelens):
    '''Band indices of the R, G, B windows (RGB_WINDOWS) in wavelens [n_bands] (nm); each window must hold a band.'''
    wl = np.asarray(wavelens, dtype=np.float64)
    out = []
    for lo, hi in RGB_WINDOWS:
        idx = np.where((wl >= lo) & (wl < hi))[0]
        assert len(idx) > 0, f"no band in [{lo}, {hi}) nm (wavelens {wl.min():.1f}-{wl.max():.1f})"
        out.append(idx)
    return out


def pseudo_rgb(x, wavelens):
    '''
    Pseudo-RGB render of a p99-scaled cube: x [B, n_bands, h, w] tensor (any float dtype), wavelens np [n_bands] nm.
    Returns [B, 3, h, w] float32 in [0, 1] (band-window means, computed in float32, then clipped).
    '''
    assert x.ndim == 4 and x.shape[1] == len(wavelens), \
        f"x must be [B, {len(wavelens)}, h, w] to match wavelens, got {tuple(x.shape)}"
    x = x.float()
    chans = [x.index_select(1, torch.as_tensor(idx, device=x.device)).mean(dim=1) for idx in rgb_band_indices(wavelens)]
    return torch.stack(chans, dim=1).clamp_(0.0, 1.0)                     # [B, 3, h, w]


def imagenet_normalize(rgb):
    '''(rgb - IMAGENET_MEAN) / IMAGENET_STD per channel; rgb [B, 3, h, w] in [0, 1] -> float32 [B, 3, h, w].'''
    mean = torch.as_tensor(IMAGENET_MEAN, device=rgb.device).view(1, 3, 1, 1)
    std = torch.as_tensor(IMAGENET_STD, device=rgb.device).view(1, 3, 1, 1)
    return (rgb.float() - mean) / std


def fit_rgb_map(z, rgb_norm):
    '''
    Least squares with intercept: rgb_norm ~ P z + q.
    z [n, N] arm channels per pixel, rgb_norm [n, 3] ImageNet-normalised pseudo-RGB per pixel.
    Returns P [3, N] float32, q [3] float32. Solved in float64 with numpy's lstsq (minimum-norm solution when z is rank
    deficient, e.g. the near-empty whitened channels of ec24, so those get ~0 weight instead of a blow-up).
    '''
    z = np.asarray(z, dtype=np.float64); rgb_norm = np.asarray(rgb_norm, dtype=np.float64)
    assert z.ndim == 2 and rgb_norm.ndim == 2 and rgb_norm.shape[1] == 3 and len(z) == len(rgb_norm), \
        f"z must be [n, N] and rgb_norm [n, 3] with the same n, got {z.shape} and {rgb_norm.shape}"
    assert len(z) > z.shape[1], f"need more pixels ({len(z)}) than channels + 1 ({z.shape[1] + 1})"
    A = np.concatenate([z, np.ones((len(z), 1))], axis=1)                 # [n, N + 1]
    sol, _, _, _ = np.linalg.lstsq(A, rgb_norm, rcond=None)               # [N + 1, 3]
    P, q = sol[:-1].T, sol[-1]                                            # [3, N], [3]
    return P.astype(np.float32), q.astype(np.float32)


def _new_conv_like(conv, in_channels):
    '''A Conv2d with conv's geometry, in_channels inputs and a bias, on conv's device and dtype.'''
    new = nn.Conv2d(in_channels, conv.out_channels, conv.kernel_size, stride=conv.stride, padding=conv.padding,
                    dilation=conv.dilation, groups=1, bias=True, padding_mode=conv.padding_mode)
    return new.to(device=conv.weight.device, dtype=conv.weight.dtype)


def fold_stem(conv, P, q, n_extra=1):
    '''
    Fold rgb_norm ~ P z + q into the RGB stem conv (nn.Conv2d, 3 inputs, groups 1): returns a new trainable nn.Conv2d
    with N + n_extra inputs (N = P.shape[1]); the n_extra trailing input slices are zero. A bias is created when conv had
    none. The weights of conv itself are not modified.
    '''
    assert isinstance(conv, nn.Conv2d) and conv.in_channels == 3 and conv.groups == 1, \
        f"fold_stem needs an RGB Conv2d (3 inputs, groups 1), got {conv}"
    P = torch.as_tensor(np.asarray(P), dtype=torch.float64); q = torch.as_tensor(np.asarray(q), dtype=torch.float64)
    assert P.ndim == 2 and P.shape[0] == 3 and q.shape == (3,), f"P must be [3, N] and q [3], got {tuple(P.shape)}, {tuple(q.shape)}"
    N = P.shape[1]
    W = conv.weight.detach().double().cpu()                               # [C_out, 3, k, k]
    b = conv.bias.detach().double().cpu() if conv.bias is not None else torch.zeros(conv.out_channels, dtype=torch.float64)
    W_N = torch.einsum('okij,kc->ocij', W, P)                             # [C_out, N, k, k]
    b_N = b + torch.einsum('okij,k->o', W, q)                             # [C_out]
    new = _new_conv_like(conv, N + n_extra)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :N] = W_N.to(new.weight)
        new.bias.copy_(b_N.to(new.bias))
    return new


def mean_stem(conv, n_in, n_extra=1):
    '''
    Fallback initialisation (spec §3): every one of the n_in inputs gets the mean RGB kernel x 3 / n_in (the response to
    a grey image is kept), the bias is copied, the n_extra trailing slices are zero.
    '''
    assert isinstance(conv, nn.Conv2d) and conv.in_channels == 3 and conv.groups == 1, \
        f"mean_stem needs an RGB Conv2d (3 inputs, groups 1), got {conv}"
    new = _new_conv_like(conv, n_in + n_extra)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :n_in] = conv.weight.mean(dim=1, keepdim=True) * (3.0 / n_in)
        new.bias.copy_(conv.bias if conv.bias is not None else torch.zeros_like(new.bias))
    return new
```

- [ ] **Step 4: Run test to verify it passes**

Run: `OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python -m pytest tests/test_seg_stem.py -v`
Expected: PASS (8 passed in about 1.5 s).
Then run the full suite: `OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python -m pytest tests/ -q`. It stays green.

- [ ] **Step 5: Commit and push**

```bash
git add models/seg_stem.py tests/test_seg_stem.py
git commit -m "feat(seg): regress-and-fold N-channel stem - pseudo-RGB render, least-squares (P, q), folded RGB stem conv

Any arm's channels start in the pretrained RGB stem's input distribution; the box channel starts silent (spec §3, §5.3).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QB4DScDwiA2DkVpbDopffV"
git push origin worktree-det
git push origin worktree-det:master
```

---

### Task 7: Third-party setup (sam2, pysodmetrics, ZoomNeXt, checkpoints) and vendored SAM2-UNet

**Files:**
- Create: `third_party/__init__.py`
- Create: `third_party/sam2_unet/__init__.py`
- Create: `third_party/sam2_unet/sam2unet.py` (upstream `SAM2UNet.py` plus `structure_loss` from `train.py`, modified)
- Create: `third_party/sam2_unet/LICENSE` (upstream file, byte-identical, sha256 checked)
- Create: `third_party/sam2_unet/NOTICE`
- Create: `bash_files/setup_third_party.sh`
- Modify: `.gitignore` (append line 13: `third_party/zoomnext/`)
- Modify: `bash_files/README.md:3` (header sentence) and the table (new row after `copy_cache_to_nvme.sh`)
- Test: `tests/test_third_party.py`

**Interfaces:**
- Consumes: nothing from earlier tasks. External pins (from recon): sam2 `2b90b9f5ceec907a1c18123530e92e794ad901a4` (dist `SAM-2` 1.0), SAM2-UNet `01598e5e9912ffb23f965ecbebf4d1dfecbaa56e`, ZoomNeXt `614af4348808734aaf2ec43937cf827410330b67`, pysodmetrics 1.6.2, hydra-core 1.3.2, omegaconf 2.3.0, antlr4-python3-runtime 4.9.3, iopath 0.1.10, portalocker 4.4.0, scikit-image 0.26.0, scikit-learn 1.9.1, einops 0.8.1.
- Produces:
  - `third_party.sam2_unet.sam2unet.build_hiera_trunk(model_cfg="configs/sam2/sam2_hiera_l.yaml", checkpoint_path=None) -> Hiera` (CPU, strict load through `build_sam2`; `checkpoint_path=None` gives a random trunk)
  - `third_party.sam2_unet.sam2unet.SAM2UNet(trunk)`: freezes the trunk, wraps every block in `Adapter`, RFB widths from `trunk.channel_list[::-1]`; `forward(x) -> (out, out1, out2)`, each `[B, 1, H, W]`
  - `third_party.sam2_unet.sam2unet.structure_loss(pred, mask, legacy_bce=False) -> scalar` (weighted BCE + weighted IoU; `legacy_bce=True` gives upstream's value)
  - also `Adapter`, `RFB_modified`, `Up`, `DoubleConv`, `BasicConv2d` (unchanged upstream classes)
  - env after `bash bash_files/setup_third_party.sh`: `import sam2, hydra, py_sod_metrics, skimage, sklearn, einops` works and numpy 2.5.2 / cv2 5.0.0 / torch 2.11.0+cu128 are unchanged; `third_party/zoomnext/` (git-ignored) is at the pinned commit; `weights/pretrained/{sam2_hiera_large.pt, sam2.1_hiera_large.pt, pvtv2-b2-zoomnext.pth}` exist.

- [ ] **Step 1: Write the failing test**

`tests/test_third_party.py`:
```python
import hashlib
import os
import torch
import torch.nn.functional as F

from third_party.sam2_unet.sam2unet import structure_loss

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDOR = os.path.join(REPO, 'third_party', 'sam2_unet')
SAM2UNET_COMMIT = '01598e5e9912ffb23f965ecbebf4d1dfecbaa56e'
LICENSE_SHA256 = 'c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4'   # upstream LICENSE at that commit


def test_vendored_sam2unet_carries_licence_and_notice():
    with open(os.path.join(VENDOR, 'LICENSE'), 'rb') as f:
        lic = f.read()
    assert hashlib.sha256(lic).hexdigest() == LICENSE_SHA256 and b'Apache License' in lic[:200]
    with open(os.path.join(VENDOR, 'NOTICE')) as f:
        notice = f.read()
    assert SAM2UNET_COMMIT in notice and 'Apache License 2.0' in notice and 'CHANGED (hsi_camo)' in notice
    files = sorted(os.listdir(VENDOR))
    # SAM2-UNet's own `sam2` package copy must never be vendored (it shadows the pip sam2), nor any weights
    assert 'sam2' not in files and 'sam2_configs' not in files
    assert not [f for f in files if f.endswith(('.pt', '.pth', '.pyd'))]
    with open(os.path.join(REPO, '.gitignore')) as f:
        assert 'third_party/zoomnext/' in f.read().splitlines()


def _wbce_wiou(pred, mask):
    '''structure_loss written out per pixel: weighted BCE + weighted IoU, mean over the batch.'''
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, 31, stride=1, padding=15) - mask)
    bce = -(mask * F.logsigmoid(pred) + (1 - mask) * F.logsigmoid(-pred))                  # [B, 1, h, w]
    wbce = (weit * bce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    p = torch.sigmoid(pred)
    inter, union = (p * mask * weit).sum(dim=(2, 3)), ((p + mask) * weit).sum(dim=(2, 3))
    return wbce, 1 - (inter + 1) / (union - inter + 1)


def test_structure_loss_is_weighted_bce_plus_weighted_iou():
    torch.manual_seed(0)
    pred = torch.randn(2, 1, 64, 64) * 3
    mask = torch.zeros(2, 1, 64, 64); mask[:, :, 20:44, 10:30] = 1.0
    wbce, wiou = _wbce_wiou(pred, mask)
    torch.testing.assert_close(structure_loss(pred, mask), (wbce + wiou).mean(), rtol=1e-5, atol=1e-6)
    # upstream's reduce='none' means reduction='mean': the plain mean BCE, which differs from the weighted one
    legacy = F.binary_cross_entropy_with_logits(pred, mask) + wiou.mean()
    torch.testing.assert_close(structure_loss(pred, mask, legacy_bce=True), legacy, rtol=1e-5, atol=1e-6)
    assert abs(float(structure_loss(pred, mask)) - float(legacy)) > 1e-3
    good = (mask * 2 - 1) * 20                                                              # confident and right
    assert float(structure_loss(good, mask)) < 0.05 < float(structure_loss(-good, mask))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python -m pytest tests/test_third_party.py -v`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'third_party'`.

- [ ] **Step 3: Vendor SAM2-UNet and git-ignore ZoomNeXt**

Only `SAM2UNet.py` (renamed `sam2unet.py`) and `train.py`'s `structure_loss` are taken. SAM2-UNet also ships its own copy of the `sam2` package (`sam2/`, `sam2_configs/`, a Windows `_C.pyd`). That copy is NOT vendored: it is a second top-level `sam2` package and would shadow the pip `sam2` that SAM2BoxSeg needs. Its Hiera trunk is identical to the pip one, which the recon checked.

`third_party/__init__.py`:
```python
'''
Third-party model code. sam2_unet/ is vendored (Apache-2.0, LICENSE + NOTICE inside). zoomnext/ is NOT part of the
repo: bash_files/setup_third_party.sh clones it at a pinned commit (git-ignored, no licence, research use only) and
models/seg_models.py puts it on sys.path itself (its top-level package is `methods`).
'''
```

`third_party/sam2_unet/__init__.py`:
```python
'''
Vendored SAM2-UNet (Apache-2.0, see LICENSE and NOTICE in this directory). Import from the module:
    from third_party.sam2_unet.sam2unet import SAM2UNet, build_hiera_trunk, structure_loss
'''
```

`third_party/sam2_unet/sam2unet.py`. This is upstream code, so it keeps upstream's style. Every change is marked `# CHANGED (hsi_camo)`:
```python
# Vendored from https://github.com/WZH0120/SAM2-UNet at commit 01598e5e9912ffb23f965ecbebf4d1dfecbaa56e
# (SAM2UNet.py, plus structure_loss from train.py). Licensed under the Apache License 2.0, see LICENSE in this
# directory. Modified for hsi_camo; every change is listed in NOTICE and marked "# CHANGED (hsi_camo)" below.
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Adapter(nn.Module):
    def __init__(self, blk) -> None:
        super(Adapter, self).__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, dim),
            nn.GELU()
        )

    def forward(self, x):
        prompt = self.prompt_learn(x)
        promped = x + prompt
        net = self.block(promped)
        return net


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


# CHANGED (hsi_camo): trunk construction split out of SAM2UNet.__init__. It imports the pip `sam2` package lazily (the
# decoder classes above stay importable without it), passes device='cpu' (upstream used build_sam2's default 'cuda',
# which fails on a CPU-only machine; the caller moves the model afterwards) and apply_postprocessing=False (only the
# trunk is kept, so it changes nothing, but it keeps the call identical to the SAM2.1 box model's), and takes the
# config name as an argument (upstream: "sam2_hiera_l.yaml" from its own bundled sam2_configs module; here the
# identical pip config "configs/sam2/sam2_hiera_l.yaml", or "configs/sam2/sam2_hiera_t.yaml" for the unit tests).
def build_hiera_trunk(model_cfg="configs/sam2/sam2_hiera_l.yaml", checkpoint_path=None):
    from sam2.build_sam import build_sam2
    model = build_sam2(model_cfg, checkpoint_path or None, device="cpu", mode="eval", apply_postprocessing=False)
    trunk = model.image_encoder.trunk
    del model
    return trunk


class SAM2UNet(nn.Module):
    # CHANGED (hsi_camo): takes an already built Hiera trunk (build_hiera_trunk) instead of a checkpoint path, and
    # sizes the RFB modules from trunk.channel_list (upstream hard-coded the Hiera-L widths 144/288/576/1152, which
    # are exactly channel_list[::-1] for Hiera-L, so the full-size model is unchanged).
    def __init__(self, trunk) -> None:
        super(SAM2UNet, self).__init__()
        self.encoder = trunk

        for param in self.encoder.parameters():
            param.requires_grad = False
        blocks = []
        for block in self.encoder.blocks:
            blocks.append(
                Adapter(block)
            )
        self.encoder.blocks = nn.Sequential(
            *blocks
        )
        c = list(trunk.channel_list)[::-1]                                     # CHANGED (hsi_camo): [144, 288, 576, 1152] for Hiera-L
        self.rfb1 = RFB_modified(c[0], 64)
        self.rfb2 = RFB_modified(c[1], 64)
        self.rfb3 = RFB_modified(c[2], 64)
        self.rfb4 = RFB_modified(c[3], 64)
        self.up1 = (Up(128, 64))
        self.up2 = (Up(128, 64))
        self.up3 = (Up(128, 64))
        self.up4 = (Up(128, 64))                                               # unused upstream as well (kept for fidelity)
        self.side1 = nn.Conv2d(64, 1, kernel_size=1)
        self.side2 = nn.Conv2d(64, 1, kernel_size=1)
        self.head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        x1, x2, x3, x4 = self.rfb1(x1), self.rfb2(x2), self.rfb3(x3), self.rfb4(x4)
        x = self.up1(x4, x3)
        out1 = F.interpolate(self.side1(x), scale_factor=16, mode='bilinear')
        x = self.up2(x, x2)
        out2 = F.interpolate(self.side2(x), scale_factor=8, mode='bilinear')
        x = self.up3(x, x1)
        out = F.interpolate(self.head(x), scale_factor=4, mode='bilinear')
        return out, out1, out2


# From train.py. CHANGED (hsi_camo): upstream calls F.binary_cross_entropy_with_logits(pred, mask, reduce='none'); the
# legacy `reduce` argument treats the string 'none' as True, i.e. reduction='mean', so upstream's "weighted BCE" is the
# plain mean BCE (the weit-weighted sum of a scalar divided by the weight sum). reduction='none' gives the weighted BCE
# that the loss is meant to compute (and that the hsi_camo spec §7 describes); legacy_bce=True restores upstream's value.
def structure_loss(pred, mask, legacy_bce=False):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean' if legacy_bce else 'none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()
```

`third_party/sam2_unet/NOTICE`:
```text
SAM2-UNet (vendored)

Source:   https://github.com/WZH0120/SAM2-UNet
Commit:   01598e5e9912ffb23f965ecbebf4d1dfecbaa56e
Licence:  Apache License 2.0 (LICENSE in this directory, copied unchanged from the commit above; the upstream file
          names no copyright holder). Paper: Xiong et al., "SAM2-UNet: Segment Anything 2 Makes Strong Encoder for
          Natural and Medical Image Segmentation", arXiv 2408.08870.

Files taken: SAM2UNet.py (as sam2unet.py) and the structure_loss function of train.py (appended to sam2unet.py).
Files NOT taken: the repo's own copy of the `sam2` package (sam2/, sam2_configs/, a Windows _C.pyd and __pycache__).
It is a second top-level package named `sam2` that would shadow, or be shadowed by, the pip `sam2` package
(facebookresearch/sam2 @ 2b90b9f5ceec907a1c18123530e92e794ad901a4) that hsi_camo installs; its Hiera trunk is identical
to the pip one, so sam2unet.py imports the pip package instead. Not taken either: dataset.py, train.py (except
structure_loss), test.py, eval.py, the shell scripts and the images.

Changes made by hsi_camo (2026-10-04), each marked "# CHANGED (hsi_camo)" in sam2unet.py:
 1. The Hiera trunk is built by a new function build_hiera_trunk(model_cfg, checkpoint_path):
    - it imports `sam2.build_sam.build_sam2` lazily, so the decoder classes import without sam2 installed;
    - it calls build_sam2(..., device="cpu", mode="eval", apply_postprocessing=False); upstream used the default
      device="cuda", which raises "No CUDA GPUs are available" on a CPU-only machine (the caller moves the model);
    - the config name is an argument: "configs/sam2/sam2_hiera_l.yaml" of the pip package (identical trunk to
      upstream's "sam2_hiera_l.yaml"), or "configs/sam2/sam2_hiera_t.yaml" for the CPU unit tests.
 2. SAM2UNet.__init__(trunk) takes the built trunk instead of a checkpoint path, and sizes the four RFB modules from
    trunk.channel_list[::-1] instead of the hard-coded Hiera-L widths 144/288/576/1152 (equal for Hiera-L).
    Everything after the trunk (freezing, Adapter wrapping, RFB, Up, side/head convs, forward) is unchanged,
    including the unused up4.
 3. structure_loss: upstream passes reduce='none' to F.binary_cross_entropy_with_logits. The legacy `reduce`
    argument reads the string as True, i.e. reduction='mean', so upstream's weighted BCE is the plain mean BCE.
    Here reduction='none' (the weighted BCE the loss is written for); structure_loss(..., legacy_bce=True)
    reproduces upstream's value.
 4. The `if __name__ == "__main__"` CUDA smoke test at the end of SAM2UNet.py is removed.

How hsi_camo uses it (models/seg_models.py, SAM2UNetSeg), deliberate departures from the upstream recipe:
 - the trunk's 3-channel patch-embed conv is replaced by an (N + 1)-channel stem folded from it
   (models/seg_stem.fold_stem) and trained at its own learning rate (the rest of the trunk stays frozen);
 - input 512 x 512 canvases instead of 352; bf16 autocast; gradient clipping 1.0; AdamW weight decay 1e-4 (upstream
   5e-4), separate no-decay group for biases / norms and the stem, 200 epochs, cosine schedule (hsi_camo spec §7).
```

`third_party/sam2_unet/LICENSE` is the upstream file, byte for byte. Fetch it and check the hash:
```bash
curl -L --fail -sS -o third_party/sam2_unet/LICENSE https://raw.githubusercontent.com/WZH0120/SAM2-UNet/01598e5e9912ffb23f965ecbebf4d1dfecbaa56e/LICENSE
sha256sum third_party/sam2_unet/LICENSE
```
Expected: `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4  third_party/sam2_unet/LICENSE`

`.gitignore`: append one line at the end. The file is 12 lines and ends with `*.pt\n`, so this becomes line 13:
```text
third_party/zoomnext/
```
`*.pt`, `*.pth` and `weights/` are already ignored, which covers `weights/pretrained/`.

- [ ] **Step 4: Run test to verify it passes**

Run: `OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python -m pytest tests/test_third_party.py -v`
Expected: PASS, 2 passed. sam2 is not needed: `sam2unet.py` imports it only inside `build_hiera_trunk`.
Then run the full suite: `OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python -m pytest tests/ -q`. It stays green.

- [ ] **Step 5: Write the setup script and the README row**

`bash_files/setup_third_party.sh`:
```bash
#!/usr/bin/env bash
# Stage-2 third-party setup (spec §4), once per machine. Idempotent: re-running skips what is already in place.
#   1. sam2 (facebookresearch, Apache-2.0) at a pinned commit, plus hydra-core 1.3.2 and the rest of its runtime deps,
#      all WITHOUT dependency resolution. sam2's pyproject build-requires torch>=2.5.1, so a plain `pip install` builds
#      in an isolated env that downloads a second torch; SAM2_BUILD_CUDA=0 skips the CUDA extension (only the video
#      predictor's hole filling uses it). The wheel also installs a top-level `training` package (no clash here).
#   2. pysodmetrics 1.6.2 (MIT) without its deps: they pin numpy<2.3.5 and opencv-python-headless, which would downgrade
#      numpy 2.5.2 and write a second cv2 over opencv-python 5.0. Its __init__ imports scikit-image and scikit-learn,
#      which install cleanly against numpy 2.5 (pip dry run: numpy untouched).
#   3. einops (ZoomNeXt's model code imports it; nothing else of ZoomNeXt's requirements is needed for the model).
#   4. ZoomNeXt (lartpang, NO licence file: research use only, never committed) cloned at a pinned commit into the
#      git-ignored third_party/zoomnext/. Left pristine: models/seg_models.py works around its CUDA query on CPU.
#   5. checkpoints into weights/pretrained/ (git-ignored): SAM2 v1 Hiera-L (SAM2-UNet starts from v1, not 2.1),
#      SAM2.1 Hiera-L (box-prompted model), ZoomNeXt PVTv2-B2 COD (Google Drive). No ImageNet PVTv2 weights: the COD
#      checkpoint holds the whole encoder.
#   6. verification on CPU: imports, numpy / cv2 / torch versions unchanged, every checkpoint strict-loads into its model.
#   bash bash_files/setup_third_party.sh       # foreground, about 2 GB of downloads
# PY is overridable. Afterwards `pip check` reports pysodmetrics' numpy / opencv-python-headless pins as unmet: expected.
set -euo pipefail
SELF=$(readlink -f "$0")
cd "$(dirname "$SELF")/.." || exit 1
PY=${PY:-/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python}
SAM2_COMMIT=2b90b9f5ceec907a1c18123530e92e794ad901a4
ZOOMNEXT_COMMIT=614af4348808734aaf2ec43937cf827410330b67
ZOOMNEXT_DRIVE_ID=1_h8XZPDtXMKYUDP2r3MIjLp80eQ1LVBB
W=weights/pretrained
mkdir -p "$W" third_party

versions() { "$PY" -c 'import numpy, cv2, torch; print(numpy.__version__, cv2.__version__, torch.__version__)'; }
BEFORE=$(versions)
echo "numpy / cv2 / torch before: $BEFORE"

echo "[1/6] sam2 @ $SAM2_COMMIT + hydra-core 1.3.2"
SAM2_BUILD_CUDA=0 "$PY" -m pip install -q --no-build-isolation --no-deps \
    "git+https://github.com/facebookresearch/sam2@$SAM2_COMMIT" \
    hydra-core==1.3.2 omegaconf==2.3.0 antlr4-python3-runtime==4.9.3 iopath==0.1.10 portalocker==4.4.0

echo "[2/6] pysodmetrics 1.6.2 (no deps) + scikit-image 0.26.0 / scikit-learn 1.9.1"
"$PY" -m pip install -q --no-deps pysodmetrics==1.6.2
"$PY" -m pip install -q scikit-image==0.26.0 scikit-learn==1.9.1

echo "[3/6] einops 0.8.1"
"$PY" -m pip install -q --no-deps einops==0.8.1

echo "[4/6] ZoomNeXt @ $ZOOMNEXT_COMMIT -> third_party/zoomnext (git-ignored, research use only)"
git check-ignore -q third_party/zoomnext/README.md || { echo ".gitignore lacks third_party/zoomnext/: refusing to clone into the repo"; exit 1; }
if [ ! -d third_party/zoomnext/.git ]; then
  git clone -q https://github.com/lartpang/ZoomNeXt third_party/zoomnext
fi
git -C third_party/zoomnext cat-file -e "$ZOOMNEXT_COMMIT^{commit}" 2>/dev/null || git -C third_party/zoomnext fetch -q origin
git -C third_party/zoomnext checkout -q "$ZOOMNEXT_COMMIT"
[ "$(git -C third_party/zoomnext rev-parse HEAD)" = "$ZOOMNEXT_COMMIT" ] || { echo "third_party/zoomnext is not at $ZOOMNEXT_COMMIT"; exit 1; }

echo "[5/6] checkpoints -> $W"
fetch() {   # fetch <url> <dest> <bytes>: kept when present with the right size, else downloaded to .part, size-checked, moved
  local url=$1 dest=$2 size=$3
  if [ -f "$dest" ] && [ "$(stat -c %s "$dest")" = "$size" ]; then echo "  $dest present"; return 0; fi
  curl -L --fail --retry 3 -sS -o "$dest.part" "$url"
  [ "$(stat -c %s "$dest.part")" = "$size" ] || { echo "  $dest: got $(stat -c %s "$dest.part") bytes, expected $size"; exit 1; }
  mv "$dest.part" "$dest"
  echo "  $dest downloaded"
}
fetch https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt "$W/sam2_hiera_large.pt" 897952466
fetch https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt "$W/sam2.1_hiera_large.pt" 898083611
# Google Drive: the usercontent URL with confirm=t skips the virus-scan page; a torch checkpoint is a zip ("PK"), an
# interstitial HTML page would not be (and would have the wrong size anyway)
fetch "https://drive.usercontent.google.com/download?id=$ZOOMNEXT_DRIVE_ID&export=download&confirm=t" "$W/pvtv2-b2-zoomnext.pth" 113020474
[ "$(head -c 2 "$W/pvtv2-b2-zoomnext.pth")" = "PK" ] || { echo "$W/pvtv2-b2-zoomnext.pth is not a torch zip"; exit 1; }

echo "[6/6] verify (CPU)"
AFTER=$(versions)
[ "$AFTER" = "$BEFORE" ] || { echo "numpy / cv2 / torch changed: $BEFORE -> $AFTER"; exit 1; }
CUDA_VISIBLE_DEVICES="" "$PY" - <<'EOF'
import sys
import contextlib
import torch
import numpy, cv2, sam2, hydra, py_sod_metrics, skimage, sklearn, einops
from importlib.metadata import version
from sam2.build_sam import build_sam2

print("numpy", numpy.__version__, "| cv2", cv2.__version__, "| torch", torch.__version__, "| SAM-2", version("SAM-2"),
      "| hydra", hydra.__version__, "| pysodmetrics", version("pysodmetrics"), "| skimage", skimage.__version__,
      "| sklearn", sklearn.__version__, "| einops", einops.__version__)
# build_sam2 loads strictly: any missing or unexpected key raises
m = build_sam2("configs/sam2/sam2_hiera_l.yaml", "weights/pretrained/sam2_hiera_large.pt", device="cpu", apply_postprocessing=False)
print("SAM2 v1 Hiera-L ok: trunk channels", m.image_encoder.trunk.channel_list, "patch embed", m.image_encoder.trunk.patch_embed.proj)
del m
m = build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", "weights/pretrained/sam2.1_hiera_large.pt", device="cpu",
               apply_postprocessing=False, hydra_overrides_extra=["++model.image_size=512"])
print("SAM2.1 Hiera-L at 512 ok: image embedding", m.sam_image_embedding_size, "x", m.sam_image_embedding_size)
del m


@contextlib.contextmanager
def no_cuda_query():
    # pvt_v2_eff.Attention.__init__ asks torch.cuda for the device capability unconditionally; fake one on CPU
    orig = torch.cuda.get_device_properties
    if not torch.cuda.is_available():
        torch.cuda.get_device_properties = lambda *a, **k: type("P", (), {"major": 0, "minor": 0})()
    try:
        yield
    finally:
        torch.cuda.get_device_properties = orig


sys.path.insert(0, "third_party/zoomnext")
from methods.zoomnext.zoomnext import PvtV2B2_ZoomNeXt
with no_cuda_query():
    net = PvtV2B2_ZoomNeXt(pretrained=False, num_frames=1, input_norm=False, use_checkpoint=False)
ck = torch.load("weights/pretrained/pvtv2-b2-zoomnext.pth", map_location="cpu")
ck = {k: v for k, v in ck.items() if not k.startswith("normalizer.")}
sd = net.state_dict()
extra, missing = [k for k in ck if k not in sd], [k for k in sd if k not in ck]
assert not extra and all(k.endswith("num_batches_tracked") for k in missing), f"unexpected {extra[:5]}, missing {missing[:5]}"
sd.update(ck)
net.load_state_dict(sd, strict=True)
print(f"ZoomNeXt PVTv2-B2 COD ok: {len(ck)} tensors, patch_embed1 {net.encoder.patch_embed1.proj}")
EOF
echo "THIRD-PARTY READY"
```

`bash_files/README.md`, line 3. Replace `Operational shell scripts for Stage 1 (kept out of the repo root).` with `Operational shell scripts for Stages 1 and 2 (kept out of the repo root).` Then add this row to the table directly after the `copy_cache_to_nvme.sh` row:
```markdown
| `setup_third_party.sh` | Stage 2, once per machine: pinned `sam2` + hydra-core, `pysodmetrics` (no deps, keeps numpy 2.5.2) + scikit-image/learn, einops; ZoomNeXt clone into git-ignored `third_party/zoomnext/` (no licence: research use only); SAM2 v1 / SAM2.1 Hiera-L and ZoomNeXt-B2 checkpoints into `weights/pretrained/`; verifies every checkpoint strict-loads |
```

Syntax check: `bash -n bash_files/setup_third_party.sh` prints nothing and exits with 0.

- [ ] **Step 6: Run the setup (operational, needs network; it pip-installs into the hsi_camo env)**

Run: `bash bash_files/setup_third_party.sh`
Expected output (download lines read "present" on a re-run):
```text
numpy / cv2 / torch before: 2.5.2 5.0.0 2.11.0+cu128
[1/6] sam2 @ 2b90b9f5ceec907a1c18123530e92e794ad901a4 + hydra-core 1.3.2
[2/6] pysodmetrics 1.6.2 (no deps) + scikit-image 0.26.0 / scikit-learn 1.9.1
[3/6] einops 0.8.1
[4/6] ZoomNeXt @ 614af4348808734aaf2ec43937cf827410330b67 -> third_party/zoomnext (git-ignored, research use only)
[5/6] checkpoints -> weights/pretrained
  weights/pretrained/sam2_hiera_large.pt downloaded
  weights/pretrained/sam2.1_hiera_large.pt downloaded
  weights/pretrained/pvtv2-b2-zoomnext.pth downloaded
[6/6] verify (CPU)
numpy 2.5.2 | cv2 5.0.0 | torch 2.11.0+cu128 | SAM-2 1.0 | hydra 1.3.2 | pysodmetrics 1.6.2 | skimage 0.26.0 | sklearn 1.9.1 | einops 0.8.1
SAM2 v1 Hiera-L ok: trunk channels [1152, 576, 288, 144] patch embed Conv2d(3, 144, kernel_size=(7, 7), stride=(4, 4), padding=(3, 3))
SAM2.1 Hiera-L at 512 ok: image embedding 32 x 32
ZoomNeXt PVTv2-B2 COD ok: 770 tensors, patch_embed1 Conv2d(3, 64, kernel_size=(7, 7), stride=(4, 4), padding=(3, 3))
THIRD-PARTY READY
```
(Deprecation FutureWarnings from timm and `torch.backends.cuda.sdp_kernel` during the ZoomNeXt build are expected.)
Then check:
- `git check-ignore -v third_party/zoomnext/README.md` → `.gitignore:13:third_party/zoomnext/	third_party/zoomnext/README.md`
- `git status --short` lists only this task's files (`.gitignore`, `bash_files/README.md`, `bash_files/setup_third_party.sh`, `third_party/`, `tests/test_third_party.py`) and nothing under `third_party/zoomnext/` or `weights/`.
- `/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python -m pip check`: the only complaints are about pysodmetrics' `numpy<2.3.5` and the missing `opencv-python-headless`. Both are expected.
- The full suite still passes: `OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python -m pytest tests/ -q`.

- [ ] **Step 7: Commit and push**

```bash
git add .gitignore bash_files/setup_third_party.sh bash_files/README.md third_party/__init__.py third_party/sam2_unet/__init__.py third_party/sam2_unet/sam2unet.py third_party/sam2_unet/LICENSE third_party/sam2_unet/NOTICE tests/test_third_party.py
git commit -m "feat(seg): vendored SAM2-UNet + setup_third_party.sh (pinned sam2, pysodmetrics, ZoomNeXt, checkpoints)

Stage 2 needs SAM2's Hiera trunk, the SAM2-UNet decoder and PySODMetrics; the installs are pinned and dependency-free so
numpy/opencv/torch in the hsi_camo env stay as they are, and ZoomNeXt (no licence) stays out of git.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QB4DScDwiA2DkVpbDopffV"
git push origin worktree-det
git push origin worktree-det:master
```

---

### Task 8: Arm front end, seg-model factory and SAM2UNetSeg (`models/seg_models.py`)

**Files:**
- Create: `models/seg_models.py`
- Test: `tests/test_seg_models.py` (T9 adds `tests/test_seg_sam2box.py`, T10 `tests/test_seg_zoomnext.py`)

**Interfaces:**
- Consumes:
  - T1: `models.filter_bank.build_filter_bank(args, dataset) -> (fb: FilterBank, volts: np.ndarray)`
  - T6: `models.seg_stem.IMAGENET_MEAN`, `IMAGENET_STD` (np [3]), `pseudo_rgb(x [B, 133, h, w] tensor, wavelens np [133]) -> [B, 3, h, w]`, `fold_stem(conv, P, q, n_extra=1) -> nn.Conv2d`
  - T7: `third_party.sam2_unet.sam2unet.SAM2UNet(trunk)`, `build_hiera_trunk(model_cfg, checkpoint_path)`, `structure_loss(pred, mask, legacy_bce=False)`; sam2 installed by `setup_third_party.sh`
  - repo: `main_det.get_args_parser()`, `main_det.dataset_kwargs(args)`, `HyperCOD_data`, `cube_cache.build_cube_cache`, `default_cache_dir`
- Produces:
  - `ARMS = ('raw', 'ec10', 'ec24', 'rgb')`, `ARM_PCA = {'ec10': 10, 'ec24': 24}`
  - `class ArmFrontEnd(nn.Module)`: `forward(x [B, 133, h, w] any dtype) -> [B, N, h, w] float32`. It computes in fp32 with autocast off, is always in eval mode and has no trainable params. Attributes: `.n_out`, `.arm`, `.fb` (None for rgb).
  - `build_front_end(arm, det_args, dataset, det_state=None) -> ArmFrontEnd`. It asserts that the arm matches the detector (`raw`/`rgb` need `raw_bands`; `ec10`/`ec24` need `pca_channels` 10/24). With `det_state` (the detector's `ckpt['model']`), the rebuilt FilterBank must equal its `filter_bank.*` tensors. T12 and T13 always pass it for raw / ec10 / ec24 (spec §9).
  - `check_filter_bank(fb, det_state)` (raises AssertionError; the message contains "detector checkpoint")
  - `param_groups_by_name(model, args, stem_prefix, lr=None) -> list of AdamW groups` named `'decay'`, `'no_decay'` and `'stem'`. The stem group runs at `args.stem_lr` with weight decay 0.
  - `class SAM2UNetSeg(nn.Module)`, `STEM = 'net.encoder.patch_embed.proj.'`:
    - `forward(x [B, n_in + 1, c, c], box_xyxy=None) -> [main, side16, side8]`, each `[B, 1, c, c]` logits, with c % 32 == 0;
    - `loss(outputs, mask) -> (total, {'loss', 'loss_main', 'loss_s16', 'loss_s8'})`;
    - `param_groups(args)`;
    - `set_progress(t)` (no-op).
  - `SEG_MODELS = {'sam2unet': SAM2UNetSeg}`: the registry. Task 9 extends this one line to add `'sam2box': SAM2BoxSeg` and Task 10 `'zoomnext': ZoomNeXtSeg`; `build_seg_model` itself is not touched again.
  - `build_seg_model(args, n_in, P, q) -> nn.Module`. It reads `args.seg_model`, `args.sam2unet_cfg` (optional, default `'configs/sam2/sam2_hiera_l.yaml'`), `args.sam2unet_ckpt` (default `weights/pretrained/sam2_hiera_large.pt`, the SAM2 **v1** checkpoint; `''`/`'none'` gives a random trunk, for tests only), `args.lr`, `args.stem_lr` and `args.weight_decay`.

- [ ] **Step 1: Write the failing test**

`tests/test_seg_models.py`:
```python
import argparse
import numpy as np
import pytest
import torch

import main_det
from data_loader.my_dataset import HyperCOD_data
from data_loader.cube_cache import build_cube_cache, default_cache_dir
from models.filter_bank import build_filter_bank
from models.seg_stem import IMAGENET_MEAN, IMAGENET_STD, pseudo_rgb
from models.seg_models import build_front_end, build_seg_model, check_filter_bank

TINY_CFG = 'configs/sam2/sam2_hiera_t.yaml'   # Hiera-T trunk, random init: no checkpoint download in the tests


def _det_args(root, *flags):
    '''Detector args as a checkpoint stores them: main_det's parser defaults on the fixture paths plus the arm flags.'''
    return main_det.get_args_parser().parse_args(['--data-path', str(root), '--cache-dir', default_cache_dir(str(root)), *flags])


RAW = ('--raw-bands',)
EC10 = ('--filter-select', 'uniform', '--num-filters', '12', '--pca-channels', '10')


@pytest.fixture
def frame(synthetic_root):
    '''(root, x [1, 133, 48, 40] fp16 p99-scaled frame as the crop loader hands it over)'''
    root, _, _ = synthetic_root
    build_cube_cache(str(root), 'train', num_workers=0)
    ds = HyperCOD_data(split='train', **main_det.dataset_kwargs(_det_args(root, *RAW)))
    return root, torch.from_numpy(ds[0][0])[None]


def _dataset(root, det):
    return HyperCOD_data(split='train', **main_det.dataset_kwargs(det))


def test_front_end_raw_equals_detector_filter_bank(frame):
    root, x = frame
    det = _det_args(root, *RAW)
    ds = _dataset(root, det)
    fe = build_front_end('raw', det, ds)
    ref, _ = build_filter_bank(det, ds)
    ref.eval()
    y = fe(x)                                                                    # fp16 in, fp32 out
    assert fe.n_out == ds.n_bands == 133 and y.shape == (1, 133, 48, 40) and y.dtype == torch.float32
    assert torch.equal(y, ref(x.float()))
    assert not any(p.requires_grad for p in fe.parameters())
    fe.train()
    assert not fe.training and not fe.fb.training                               # a fixed sensor model stays in eval mode


def test_front_end_ec10_is_fp32_under_bf16_autocast(frame):
    root, x = frame
    det = _det_args(root, *EC10)
    ds = _dataset(root, det)
    fe = build_front_end('ec10', det, ds)
    ref, _ = build_filter_bank(det, ds)
    ref.eval()
    with torch.autocast('cpu', dtype=torch.bfloat16):
        y = fe(x)
    assert fe.n_out == 10 and y.shape == (1, 10, 48, 40) and y.dtype == torch.float32
    assert torch.equal(y, ref(x.float()))                                       # autocast did not touch the front end


def test_front_end_rgb_is_imagenet_normalised_pseudo_rgb(frame):
    root, x = frame
    det = _det_args(root, *RAW)
    ds = _dataset(root, det)
    fe = build_front_end('rgb', det, ds)
    y = fe(x)
    mean = torch.as_tensor(IMAGENET_MEAN).view(1, 3, 1, 1); std = torch.as_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    assert fe.n_out == 3 and y.dtype == torch.float32
    torch.testing.assert_close(y, (pseudo_rgb(x.float(), ds.wavelens) - mean) / std)


def test_front_end_rejects_a_detector_of_another_arm(frame):
    root, _ = frame
    det_raw, det_ec10 = _det_args(root, *RAW), _det_args(root, *EC10)
    ds_raw, ds_ec = _dataset(root, det_raw), _dataset(root, det_ec10)
    for arm, det, ds in (('ec10', det_raw, ds_raw), ('ec24', det_ec10, ds_ec), ('raw', det_ec10, ds_ec), ('rgb', det_ec10, ds_ec)):
        with pytest.raises(AssertionError):
            build_front_end(arm, det, ds)


def test_front_end_checks_the_detector_checkpoint_buffers(frame):
    root, _ = frame
    det = _det_args(root, *EC10)
    ds = _dataset(root, det)
    fb, _ = build_filter_bank(det, ds)
    state = {f'filter_bank.{k}': v.clone() for k, v in fb.state_dict().items()}
    state['yolo.model.0.conv.weight'] = torch.zeros(1)                          # the rest of a detector state dict is ignored
    assert build_front_end('ec10', det, ds, det_state=state).n_out == 10
    state['filter_bank.mean'] = state['filter_bank.mean'] + 1e-3                # e.g. another detector's whitening
    with pytest.raises(AssertionError, match='mean'):
        build_front_end('ec10', det, ds, det_state=state)
    with pytest.raises(AssertionError):
        check_filter_bank(fb, {k: v for k, v in state.items() if not k.endswith('.std')})


def _seg_args(**kw):
    a = dict(seg_model='sam2unet', sam2unet_cfg=TINY_CFG, sam2unet_ckpt='none', lr=1e-3, stem_lr=1e-4, weight_decay=1e-4)
    a.update(kw)
    return argparse.Namespace(**a)


def _stem_map(n_in, seed=0):
    rng = np.random.default_rng(seed)
    return (0.1 * rng.standard_normal((3, n_in))).astype(np.float32), (0.1 * rng.standard_normal(3)).astype(np.float32)


@pytest.mark.parametrize('n_in', [3, 10, 24])
def test_sam2unet_seg_forward_backward(n_in):
    pytest.importorskip('sam2')
    torch.manual_seed(0)
    P, q = _stem_map(n_in)
    model = build_seg_model(_seg_args(), n_in, P, q)
    stem = model.net.encoder.patch_embed.proj
    assert stem.in_channels == n_in + 1 and stem.weight.requires_grad and stem.bias.requires_grad
    assert float(stem.weight.detach()[:, n_in].abs().max()) == 0.0                       # box-channel slice starts at zero
    model.train()
    x = torch.randn(2, n_in + 1, 64, 64)
    box = torch.tensor([[12., 16., 36., 40.]] * 2)                              # canvas px xyxy (unused by SAM2-UNet)
    mask = torch.zeros(2, 1, 64, 64); mask[:, :, 16:40, 12:36] = 1.0
    outs = model(x, box)
    assert len(outs) == 3 and all(o.shape == (2, 1, 64, 64) for o in outs)
    total, items = model.loss(outs, mask)
    assert torch.isfinite(total) and set(items) == {'loss', 'loss_main', 'loss_s16', 'loss_s8'}
    assert all(isinstance(v, float) for v in items.values())
    assert items['loss'] == pytest.approx(items['loss_main'] + items['loss_s16'] + items['loss_s8'], rel=1e-5)
    total.backward()
    assert stem.weight.grad is not None and float(stem.weight.grad[:, n_in].abs().sum()) > 0   # the box slice learns
    named = dict(model.named_parameters())
    trunk = [p for n, p in named.items() if n.startswith('net.encoder.') and '.prompt_learn.' not in n and not n.startswith(model.STEM)]
    adapters = [p for n, p in named.items() if '.prompt_learn.' in n]
    assert trunk and all(not p.requires_grad and p.grad is None for p in trunk)  # Hiera frozen
    assert adapters and all(p.requires_grad and p.grad is not None for p in adapters)
    assert all(p.grad is not None for n, p in named.items() if n.startswith(('net.rfb', 'net.up3', 'net.head', 'net.side')))


def test_sam2unet_seg_stem_is_folded_from_the_trunk_stem():
    pytest.importorskip('sam2')
    from third_party.sam2_unet.sam2unet import build_hiera_trunk
    n_in = 10
    P, q = _stem_map(n_in)
    torch.manual_seed(0)
    rgb_stem = build_hiera_trunk(TINY_CFG, None).patch_embed.proj               # the same random trunk as below
    torch.manual_seed(0)
    model = build_seg_model(_seg_args(), n_in, P, q)
    z = torch.randn(1, n_in, 64, 64)
    rgb = torch.einsum('kc,bchw->bkhw', torch.from_numpy(P), z) + torch.from_numpy(q).view(1, 3, 1, 1)
    with torch.no_grad():
        new = model.net.encoder.patch_embed.proj(torch.cat([z, torch.rand(1, 1, 64, 64)], dim=1))
        old = rgb_stem(rgb)
    # away from the zero-padded border, where the bias fold of q differs (padding is zero in z, not in P z + q)
    torch.testing.assert_close(new[..., 1:-1, 1:-1], old[..., 1:-1, 1:-1], rtol=1e-4, atol=1e-4)


def test_sam2unet_seg_param_groups_and_eval_mode():
    pytest.importorskip('sam2')
    args = _seg_args()
    P, q = _stem_map(10)
    model = build_seg_model(args, 10, P, q)
    groups = model.param_groups(args)
    ids = [id(p) for g in groups for p in g['params']]
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    assert len(ids) == len(set(ids)) and set(ids) == trainable                  # every trainable tensor exactly once
    by = {g['name']: g for g in groups}
    stem = model.net.encoder.patch_embed.proj
    assert {id(p) for p in by['stem']['params']} == {id(stem.weight), id(stem.bias)}
    assert by['stem']['lr'] == 1e-4 and by['stem']['weight_decay'] == 0.0
    assert by['decay']['lr'] == 1e-3 and by['decay']['weight_decay'] == 1e-4 and by['no_decay']['weight_decay'] == 0.0
    assert all(p.ndim > 1 for p in by['decay']['params'])
    assert not any(p.requires_grad for p in model.net.up4.parameters())         # never called upstream
    torch.optim.AdamW(groups, lr=args.lr)
    model.eval()
    with torch.no_grad():
        outs = model(torch.randn(1, 11, 64, 64), torch.zeros(1, 4))
    assert outs[0].shape == (1, 1, 64, 64)
    with pytest.raises(AssertionError):
        model(torch.randn(1, 11, 48, 48), torch.zeros(1, 4))                    # side not a multiple of 32
    with pytest.raises(AssertionError):
        model(torch.randn(1, 10, 64, 64), torch.zeros(1, 4))                    # box channel missing


def test_build_seg_model_checks_name_and_stem_map():
    pytest.importorskip('sam2')
    P, q = _stem_map(10)
    with pytest.raises(AssertionError):
        build_seg_model(_seg_args(seg_model='unet'), 10, P, q)
    with pytest.raises(AssertionError):
        build_seg_model(_seg_args(), 24, P, q)                                   # P fitted for another arm
    with pytest.raises(AssertionError):
        build_seg_model(_seg_args(sam2unet_ckpt='/nonexistent/sam2_hiera_large.pt'), 10, P, q)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python -m pytest tests/test_seg_models.py -v`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'models.seg_models'`. This requires Tasks 1, 6 and 7 to be done first; otherwise the import of `build_filter_bank` / `models.seg_stem` / `third_party` fails earlier.

- [ ] **Step 3: Write minimal implementation**

`models/seg_models.py`:
```python
'''
Stage-2 segmentation models (spec §5.4) and the arm front end that feeds them.

  build_front_end(arm, det_args, dataset)  p99-scaled 133-band canvas [B, 133, c, c] -> the arm's channels [B, N, c, c]
  build_seg_model(args, n_in, P, q)        SAM2UNetSeg | SAM2BoxSeg | ZoomNeXtSeg, all with one API:
      forward(x [B, n_in + 1, c, c], box_xyxy [B, 4] canvas px) -> list of logits [B, 1, c, c]
                                            (index 0 = main output, deep-supervision outputs after it)
      loss(outputs, mask [B, 1, c, c])  -> (total tensor, {name: float})
      param_groups(args)                -> AdamW param groups, the folded stem at args.stem_lr
      set_progress(t)                   -> training progress t in [0, 1] (ZoomNeXt's UAL weight; a no-op elsewhere)

The input x is cat(front_end(img) * valid, box_map) (train_eval_seg.seg_inputs); (P, q) is the arm's least-squares map
from its channels to the ImageNet-normalised pseudo-RGB render (seg_stem.fit_rgb_map), folded into each model's
pretrained RGB stem (seg_stem.fold_stem) so the network sees the pseudo-RGB image at initialisation (spec §3).
'''
import os
import numpy as np
import torch
import torch.nn as nn

from models.filter_bank import build_filter_bank
from models.seg_stem import IMAGENET_MEAN, IMAGENET_STD, pseudo_rgb, fold_stem
from third_party.sam2_unet.sam2unet import SAM2UNet, build_hiera_trunk, structure_loss

ARMS = ('raw', 'ec10', 'ec24', 'rgb')
ARM_PCA = {'ec10': 10, 'ec24': 24}                    # whitened channels of the arm's detector (--pca-channels)

SAM2UNET_CFG = 'configs/sam2/sam2_hiera_l.yaml'       # pip sam2's SAM2 v1 Hiera-L config (SAM2-UNet uses v1, not 2.1)
SAM2UNET_CKPT = 'weights/pretrained/sam2_hiera_large.pt'


class ArmFrontEnd(nn.Module):
    '''
    The arm's fixed sensor model on the GPU: raw / ec10 / ec24 = the detector's FilterBank (standardised raw bands, or
    the EC readings whitened to 10 / 24 channels), rgb = the pseudo-RGB render normalised with the ImageNet mean/std.
    Nothing here trains: every parameter is frozen and the module stays in eval mode whatever train() is called with.
    The computation runs in fp32 with autocast off. Under the bf16 autocast of training the whitened channels would
    otherwise be off by up to 0.34 (ec10) / 0.43 (ec24) standardised units on a real window (recon, fb_check.py); in
    fp32 they are the reference values (the detectors themselves saw fp16-rounded channels).
    '''

    def __init__(self, arm, fb=None, wavelens=None):
        super(ArmFrontEnd, self).__init__()
        assert arm in ARMS, f"arm must be one of {ARMS}, got {arm!r}"
        self.arm = arm
        self.fb = fb
        if arm == 'rgb':
            assert wavelens is not None, "the rgb arm needs the band centres (dataset.wavelens)"
            self.wavelens = np.asarray(wavelens, dtype=np.float64)                         # [133] nm
            self.register_buffer('rgb_mean', torch.as_tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1))
            self.register_buffer('rgb_std', torch.as_tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1))
            self.n_out = 3
        else:
            assert fb is not None, f"arm '{arm}' needs its detector's FilterBank"
            self.n_out = fb.n_channels
        for p in self.parameters():
            p.requires_grad_(False)
        super(ArmFrontEnd, self).train(False)

    def train(self, mode=True):
        # a fixed sensor model: always eval (FilterBank has no train-mode behaviour that Stage 2 wants)
        return super(ArmFrontEnd, self).train(False)

    def forward(self, x):
        '''x: [B, 133, h, w] p99-scaled bands, any float dtype -> [B, N, h, w] float32'''
        with torch.autocast(x.device.type, enabled=False):
            x = x.float()                                                                  # [B, 133, h, w]
            if self.fb is None:
                y = (pseudo_rgb(x, self.wavelens) - self.rgb_mean) / self.rgb_std          # [B, 3, h, w]
            else:
                y = self.fb(x)                                                             # [B, N, h, w]
        return y.float()


def check_filter_bank(fb, det_state):
    '''
    Assert that a rebuilt FilterBank equals the one stored in the detector checkpoint (det_state = ckpt['model'], keys
    'filter_bank.<buffer>'), so the arm's channels are bit-identical to what its detector saw (spec §3, §9).
    '''
    ref = {k[len('filter_bank.'):]: v for k, v in det_state.items() if k.startswith('filter_bank.')}
    own = fb.state_dict()
    assert set(ref) == set(own), f"filter bank tensors differ from the detector checkpoint: rebuilt {sorted(own)}, checkpoint {sorted(ref)}"
    bad = [k for k in own if own[k].shape != ref[k].shape or not torch.equal(own[k].cpu(), ref[k].cpu())]
    assert not bad, f"rebuilt filter bank != detector checkpoint in {bad} (wrong --det_ckpt for this arm, or changed band stats?)"


def build_front_end(arm, det_args, dataset, det_state=None):
    '''
    The arm's front end, rebuilt from its detector's arguments exactly as Stage 1 built it (filter_bank.build_filter_bank).
      arm        'raw' | 'ec10' | 'ec24' | 'rgb'
      det_args   the detector checkpoint's args as a Namespace (main_det_rois.detector_args: parser defaults overlaid
                 with ckpt['args']); rgb uses the raw detector raw133_A (its ROIs; only the band centres matter)
      dataset    a HyperCOD_data built with main_det.dataset_kwargs(det_args): it supplies the band statistics, the EC
                 response matrix of the detector's voltages and wavelens
      det_state  optional detector state dict (ckpt['model']): the rebuilt FilterBank must equal its 'filter_bank.*'
    Returns an ArmFrontEnd with .n_out = N (133 / 10 / 24 / 3).
    '''
    assert arm in ARMS, f"arm must be one of {ARMS}, got {arm!r}"
    raw = bool(getattr(det_args, 'raw_bands', False))
    k = int(getattr(det_args, 'pca_channels', 0) or 0)
    if arm in ('raw', 'rgb'):
        assert raw, f"arm '{arm}' needs the raw-band detector (--raw-bands, raw133_A); this detector uses EC readings ({k} whitened channels)"
    else:
        assert not raw, f"arm '{arm}' needs an EC detector; this one is the raw-band detector"
        assert k == ARM_PCA[arm], f"arm '{arm}' needs a detector with --pca-channels {ARM_PCA[arm]}, this one has {k}"
    if arm == 'rgb':
        fe = ArmFrontEnd(arm, wavelens=dataset.wavelens)
    else:
        fb, volts = build_filter_bank(det_args, dataset)
        if det_state is not None:
            check_filter_bank(fb, det_state)
        fe = ArmFrontEnd(arm, fb=fb)
        expect = dataset.n_bands if arm == 'raw' else ARM_PCA[arm]
        assert fe.n_out == expect, f"arm '{arm}' front end gives {fe.n_out} channels, expected {expect}"
    print(f"front end '{arm}': {fe.n_out} channels" + (" (pseudo-RGB, ImageNet-normalised)" if arm == 'rgb' else
                                                      f" (FilterBank {fe.fb.n_readings} readings -> {fe.fb.n_channels} channels)"))
    return fe


def param_groups_by_name(model, args, stem_prefix, lr=None):
    '''
    AdamW groups over the trainable parameters, by name: 'decay' (weights, args.weight_decay), 'no_decay' (biases and
    1-D tensors such as norms, no decay, as main_det.build_optimizer) and 'stem' (the folded input conv, every tensor
    whose name starts with stem_prefix, at args.stem_lr without decay: it starts from the pretrained RGB kernel, and
    main_det keeps its rebuilt first conv out of the decay too). lr defaults to args.lr. Empty groups are dropped;
    each group carries a 'name' for logging.
    '''
    lr = args.lr if lr is None else lr
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    stem = [p for n, p in named if n.startswith(stem_prefix)]
    assert stem, f"no trainable parameter under '{stem_prefix}'"
    rest = [(n, p) for n, p in named if not n.startswith(stem_prefix)]
    decay = [p for n, p in rest if p.ndim > 1 and not n.endswith('.bias')]
    no_decay = [p for n, p in rest if p.ndim <= 1 or n.endswith('.bias')]
    groups = [{'name': 'decay', 'params': decay, 'lr': lr, 'weight_decay': args.weight_decay},
              {'name': 'no_decay', 'params': no_decay, 'lr': lr, 'weight_decay': 0.0},
              {'name': 'stem', 'params': stem, 'lr': args.stem_lr, 'weight_decay': 0.0}]
    return [g for g in groups if g['params']]


class SAM2UNetSeg(nn.Module):
    '''
    SAM2-UNet (vendored, third_party/sam2_unet): the SAM2 v1 Hiera trunk, frozen, with a trainable bottleneck adapter in
    front of every block, four RFB modules and a U-Net decoder. The trunk's 3-channel patch-embed conv
    (Conv2d(3, 144, 7, stride 4, pad 3) for Hiera-L) is replaced by the folded (n_in + 1)-channel stem, trainable.
    The box enters only through the box channel of x, so box_xyxy is unused.
    Outputs, all [B, 1, c, c] logits: [main (stride-4 head), side1 (stride 16), side2 (stride 8)].
    Loss: SAM2-UNet's structure loss (weighted BCE + weighted IoU) summed over the three outputs.
    args: sam2unet_cfg (hydra config of the pip sam2 package, default SAM2 v1 Hiera-L; the tests pass
          'configs/sam2/sam2_hiera_t.yaml'), sam2unet_ckpt (SAM2 v1 checkpoint, default SAM2UNET_CKPT; '' or 'none' keeps
          the trunk random, for tests only), lr, stem_lr, weight_decay.
    '''
    STEM = 'net.encoder.patch_embed.proj.'

    def __init__(self, args, n_in, P, q):
        super(SAM2UNetSeg, self).__init__()
        cfg = getattr(args, 'sam2unet_cfg', None) or SAM2UNET_CFG
        ckpt = getattr(args, 'sam2unet_ckpt', None)
        ckpt = SAM2UNET_CKPT if ckpt is None else ckpt
        ckpt = None if ckpt in ('', 'none') else ckpt
        assert ckpt is None or os.path.exists(ckpt), f"SAM2 v1 checkpoint {ckpt} not found: run bash bash_files/setup_third_party.sh"
        self.n_in = n_in
        self.net = SAM2UNet(build_hiera_trunk(cfg, ckpt))                       # trunk frozen; adapters, RFB, decoder trainable
        self.net.up4.requires_grad_(False)                                      # defined but never called upstream: keep it out of the optimiser
        old = self.net.encoder.patch_embed.proj                                 # pretrained RGB stem Conv2d(3, C, 7, 4, 3)
        assert old.in_channels == 3, f"expected the RGB patch-embed conv, got {old}"
        self.net.encoder.patch_embed.proj = fold_stem(old, P, q, n_extra=1)    # Conv2d(n_in + 1, C, 7, 4, 3), box slice zero
        self.net.encoder.patch_embed.proj.requires_grad_(True)                  # swapped in AFTER the trunk freeze: trainable
        print(f"SAM2UNetSeg: {cfg}, trunk {'from ' + ckpt if ckpt else 'RANDOM (no checkpoint)'}, stem {self.net.encoder.patch_embed.proj}")

    def forward(self, x, box_xyxy=None):
        '''x: [B, n_in + 1, c, c] (arm channels + box channel), c a multiple of 32 -> [main, side1, side2] logits [B, 1, c, c]'''
        assert x.shape[1] == self.n_in + 1, f"expected {self.n_in} + 1 input channels, got {x.shape[1]}"
        # Hiera tiles its 8 x 8 window position embedding over the stride-4 grid, so the side must be a multiple of 32
        assert x.shape[-2] % 32 == 0 and x.shape[-1] % 32 == 0, f"canvas side must be a multiple of 32, got {tuple(x.shape[-2:])}"
        out, out1, out2 = self.net(x)
        return [out, out1, out2]

    def loss(self, outputs, mask):
        '''structure_loss on every output, in fp32 (outputs come out bf16 under autocast); mask [B, 1, c, c] float in {0, 1}'''
        items, total = {}, 0.0
        with torch.autocast(mask.device.type, enabled=False):
            for name, o in zip(('main', 's16', 's8'), outputs):
                l = structure_loss(o.float(), mask.float())
                items[f'loss_{name}'] = float(l.detach())
                total = total + l
        items['loss'] = float(total.detach())
        return total, items

    def param_groups(self, args):
        # adapters, RFB and decoder at args.lr (spec §7: 1e-3), the folded stem at args.stem_lr (1e-4)
        return param_groups_by_name(self, args, self.STEM)

    def set_progress(self, t):
        pass


# --seg_model name -> class. Task 9 adds 'sam2box': SAM2BoxSeg and Task 10 'zoomnext': ZoomNeXtSeg to this one line.
SEG_MODELS = {'sam2unet': SAM2UNetSeg}


def build_seg_model(args, n_in, P, q):
    '''
    The --seg_model network for n_in arm channels (+ 1 box channel), its stem folded from (P [3, n_in], q [3]).
    Every class is constructed as Cls(args, n_in, P, q) and shares the API of the module docstring.
    '''
    assert args.seg_model in SEG_MODELS, f"--seg_model must be one of {sorted(SEG_MODELS)}, got {args.seg_model!r}"
    P, q = np.asarray(P, dtype=np.float32), np.asarray(q, dtype=np.float32)
    assert P.shape == (3, n_in) and q.shape == (3,), f"stem map must be P [3, {n_in}], q [3]; got {P.shape}, {q.shape}"
    model = SEG_MODELS[args.seg_model](args, n_in, P, q)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"{args.seg_model}: {n_in} + 1 input channels, {n_train / 1e6:.2f}M trainable of {n_all / 1e6:.1f}M parameters")
    return model
```

- [ ] **Step 4: Run test to verify it passes**

Run: `OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python -m pytest tests/test_seg_models.py -v`
Expected: PASS, 11 passed in about 5 s on CPU. With sam2 missing, 5 pass and the 6 SAM2UNetSeg tests are skipped.
Then run the full suite: `OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python -m pytest tests/ -q`. It stays green.

- [ ] **Step 5: Full-size smoke check (CPU, random Hiera-L, 512 canvas; no checkpoint needed)**

Run:
```bash
CUDA_VISIBLE_DEVICES="" /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python - <<'EOF'
import argparse, numpy as np, torch
from models.seg_models import build_seg_model
args = argparse.Namespace(seg_model='sam2unet', sam2unet_ckpt='none', lr=1e-3, stem_lr=1e-4, weight_decay=1e-4)
for n_in in (10, 133):
    m = build_seg_model(args, n_in, np.random.randn(3, n_in).astype(np.float32) * 0.05, np.zeros(3, np.float32)).eval()
    print('groups', [(g['name'], sum(p.numel() for p in g['params'])) for g in m.param_groups(args)])
    with torch.no_grad():
        print(n_in, [tuple(o.shape) for o in m(torch.randn(1, n_in + 1, 512, 512), torch.zeros(1, 4))])
EOF
```
Expected, as measured while drafting:
```text
sam2unet: 10 + 1 input channels, 4.35M trainable of 216.6M parameters
groups [('decay', 4233408), ('no_decay', 36339), ('stem', 77760)]
10 [(1, 1, 512, 512), (1, 1, 512, 512), (1, 1, 512, 512)]
sam2unet: 133 + 1 input channels, 5.22M trainable of 217.5M parameters
groups [('decay', 4233408), ('no_decay', 36339), ('stem', 945648)]
133 [(1, 1, 512, 512), (1, 1, 512, 512), (1, 1, 512, 512)]
```

- [ ] **Step 6: Commit and push**

```bash
git add models/seg_models.py tests/test_seg_models.py
git commit -m "feat(seg): arm front end, seg-model factory and SAM2UNetSeg

Stage 2 feeds each arm's detector FilterBank (fp32, rebuilt and checked against the detector checkpoint) into SAM2-UNet
with the Hiera patch-embed folded to N + 1 channels; SAM2BoxSeg and ZoomNeXtSeg register in the same factory.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QB4DScDwiA2DkVpbDopffV"
git push origin worktree-det
git push origin worktree-det:master
```

---

### Task 9: SAM2BoxSeg (box-prompted SAM2.1 + LoRA) in models/seg_models.py

**Files:**
- Modify: `models/seg_models.py`. Add two imports. Insert `SAM2_CFG`, `LoRALinear`, `dice_bce_loss` and `SAM2BoxSeg` between Task 8's `SAM2UNetSeg` class and its `SEG_MODELS = {...}` line. Extend that one `SEG_MODELS` line; `build_seg_model` stays as Task 8 wrote it.
- Test: `tests/test_seg_sam2box.py` (new)

**Interfaces:**
- Consumes:
  - `models.seg_stem.fold_stem(conv, P, q, n_extra=1) -> nn.Conv2d` (T6, already imported by T8)
  - T8's `SEG_MODELS` registry and `build_seg_model(args, n_in, P, q)` (every class constructed as `Cls(args, n_in, P, q)`)
  - pip `sam2` @ 2b90b9f5ceec907a1c18123530e92e794ad901a4 with hydra-core 1.3.2 (T7): `sam2.build_sam.build_sam2(config_file, ckpt_path, device, mode, hydra_overrides_extra, apply_postprocessing)`
- Produces:
  - `SAM2_CFG = 'configs/sam2.1/sam2.1_hiera_l.yaml'`
  - `LoRALinear(base nn.Linear, r=8, alpha=8)`
  - `dice_bce_loss(logits [B,1,h,w], mask [B,1,h,w], smooth=1.0) -> (bce, dice)` (fp32 scalars)
  - `SAM2BoxSeg(args, n_in, P, q)` with:
    - `forward(x [B, n_in+1, c, c], box_xyxy [B, 4] canvas px) -> [logits [B,1,c,c]]`
    - `loss(outputs, mask) -> (total, {'bce', 'dice'})`
    - `param_groups(args) -> [stem @ args.stem_lr (weight decay 0), lora @ args.lr, decoder @ args.lr]`, each group with a `'name'` key
    - `set_progress(t)`, a no-op
  - `SEG_MODELS = {'sam2unet': SAM2UNetSeg, 'sam2box': SAM2BoxSeg}`
  - Attributes it reads from `args`:
    - required: `seg_model`, `canvas`, `lr`, `stem_lr`, `weight_decay`, `sam2_ckpt` (path to `sam2.1_hiera_large.pt`; `''` or None gives a random init, for tests only)
    - optional through getattr: `sam2_cfg` (default `SAM2_CFG`), `lora_r` (8), `lora_alpha` (= r, so the LoRA scale is 1)
  - Zero-shot control (T13 `--zero_shot`): `build_seg_model(args(seg_model='sam2box'), 3, np.eye(3), np.zeros(3))` with the SAM2.1-L checkpoint is exactly pretrained SAM2.1-L at image size `canvas`, because the identity fold keeps the RGB kernel, the box slice is zero and LoRA's B is zero. Note that the rgb front end (pseudo-RGB, then ImageNet normalisation) is SAM2's own preprocessing.

- [ ] **Step 1: Write the failing test**

Create `tests/test_seg_sam2box.py`:

```python
import types
import numpy as np
import torch
import torch.nn as nn
import pytest

from models.seg_models import LoRALinear, dice_bce_loss, build_seg_model

TINY_CFG = 'configs/sam2.1/sam2.1_hiera_t.yaml'                              # Hiera-T: 12 blocks, embed 96


def _args(**kw):
    '''main_seg-like namespace for a tiny random-init SAM2.1-T at canvas 128 (no checkpoint download).'''
    a = dict(seg_model='sam2box', sam2_cfg=TINY_CFG, sam2_ckpt='', canvas=128, lora_r=8, lr=1e-4, stem_lr=5e-5,
             weight_decay=1e-4)
    a.update(kw)
    return types.SimpleNamespace(**a)


def _pq(n, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((3, n)) * 0.1).astype(np.float32), (rng.standard_normal(3) * 0.1).astype(np.float32)


def test_lora_linear_is_the_base_layer_at_init_and_trains_only_the_adapter():
    torch.manual_seed(0)
    base = nn.Linear(16, 24)
    lora = LoRALinear(base, r=4, alpha=8)
    x = torch.randn(5, 16)
    torch.testing.assert_close(lora(x), base(x))                              # B = 0 -> unchanged
    assert lora.in_features == 16 and lora.out_features == 24 and lora.scale == 2.0
    lora(x).pow(2).sum().backward()
    assert not base.weight.requires_grad and not base.bias.requires_grad and base.weight.grad is None
    assert lora.lora_B.weight.grad is not None and lora.lora_B.weight.grad.abs().sum() > 0
    with torch.no_grad():
        lora.lora_B.weight.fill_(0.01)
    expected = base(x) + (x @ lora.lora_A.weight.t()) @ lora.lora_B.weight.t() * 2.0
    torch.testing.assert_close(lora(x), expected)


def test_dice_bce_loss_values():
    mask = torch.zeros(2, 1, 8, 8)
    mask[0, :, 2:6, 2:6] = 1.0                                                # image 1 is a false-positive item (empty target)
    good = (mask * 2 - 1) * 20.0                                              # confident and correct logits
    bce, dice = dice_bce_loss(good, mask)
    assert float(bce) < 1e-6 and float(dice) < 1e-3
    bad = -good
    bce_b, dice_b = dice_bce_loss(bad, mask)
    assert float(bce_b) > 10.0 and float(dice_b) > 0.45                       # image 0 dice ~1, image 1 ~0 (smooth) -> ~0.5
    logits = torch.randn(2, 1, 8, 8)
    p = torch.sigmoid(logits)
    ref_dice = (1 - (2 * (p * mask).sum((2, 3)) + 1) / (p.sum((2, 3)) + mask.sum((2, 3)) + 1)).mean()
    ref_bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, mask)
    b, d = dice_bce_loss(logits.to(torch.bfloat16), mask)
    assert b.dtype == torch.float32
    torch.testing.assert_close(d, ref_dice, rtol=2e-2, atol=2e-2)              # bf16 logits, fp32 loss
    torch.testing.assert_close(dice_bce_loss(logits, mask)[1], ref_dice)
    torch.testing.assert_close(dice_bce_loss(logits, mask)[0], ref_bce)


def test_sam2box_rejects_a_canvas_not_divisible_by_32():
    pytest.importorskip('sam2')
    P, q = _pq(3)
    with pytest.raises(AssertionError, match='divisible by 32'):
        build_seg_model(_args(canvas=120), 3, P, q)


@pytest.mark.parametrize('n', [3, 10, 24])
def test_sam2box_builds_and_trains_at_n_channels(n):
    pytest.importorskip('sam2')
    torch.manual_seed(0)
    P, q = _pq(n)
    model = build_seg_model(_args(), n, P, q)
    sam = model.sam
    trunk = sam.image_encoder.trunk
    stem = trunk.patch_embed.proj
    assert stem.in_channels == n + 1 and stem.out_channels == 96
    assert torch.equal(stem.weight[:, n], torch.zeros_like(stem.weight[:, n]))  # box-channel slice starts at zero
    # LoRA exactly on attn.qkv / attn.proj of every block, never on the block-level `proj` of stage-change blocks
    assert model.n_lora == 2 * len(trunk.blocks) == 24
    assert all(isinstance(b.attn.qkv, LoRALinear) and isinstance(b.attn.proj, LoRALinear) for b in trunk.blocks)
    assert not any(isinstance(getattr(b, 'proj', None), LoRALinear) for b in trunk.blocks)
    trainable = {k for k, p in model.named_parameters() if p.requires_grad}
    assert all(k.startswith('sam.image_encoder.trunk.patch_embed.') or '.lora_' in k or k.startswith('sam.sam_mask_decoder.')
               for k in trainable)
    assert not any('iou_prediction_head' in k or 'pred_obj_score_head' in k for k in trainable)
    assert not any(p.requires_grad for p in sam.sam_prompt_encoder.parameters())
    assert not any(p.requires_grad for p in sam.image_encoder.neck.parameters())

    model.train()
    x = torch.randn(2, n + 1, 128, 128)
    box = torch.tensor([[10.0, 20.0, 70.0, 90.0], [40.0, 30.0, 127.0, 120.0]])
    outs = model(x, box)
    assert isinstance(outs, list) and len(outs) == 1 and outs[0].shape == (2, 1, 128, 128)
    mask = (torch.rand(2, 1, 128, 128) > 0.8).float()
    total, items = model.loss(outs, mask)
    assert set(items) == {'bce', 'dice'} and all(isinstance(v, float) for v in items.values())
    total.backward()
    no_grad = [k for k, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert no_grad == [], f"trainable parameters without a gradient: {no_grad[:5]}"
    assert stem.weight.grad[:, n].abs().sum() > 0                              # the box channel learns

    groups = model.param_groups(_args())
    assert [g['name'] for g in groups] == ['stem', 'lora', 'decoder']
    assert groups[0]['lr'] == 5e-5 and groups[1]['lr'] == groups[2]['lr'] == 1e-4
    ids = [id(p) for g in groups for p in g['params']]
    assert len(ids) == len(set(ids)) == len(trainable)
    assert len(groups[0]['params']) == 2 and len(groups[1]['params']) == 2 * model.n_lora
    torch.optim.AdamW(groups)                                                  # extra 'name' key is accepted


def test_sam2box_box_prompt_and_bf16_autocast():
    pytest.importorskip('sam2')
    torch.manual_seed(0)
    P, q = _pq(10)
    model = build_seg_model(_args(), 10, P, q).eval()
    x = torch.randn(1, 11, 128, 128)
    with torch.no_grad():
        a = model(x, torch.tensor([[10.0, 10.0, 60.0, 60.0]]))[0]
        b = model(x, torch.tensor([[60.0, 60.0, 120.0, 120.0]]))[0]
        assert not torch.allclose(a, b)                                        # the prompt reaches the decoder
        torch.testing.assert_close(model(x, torch.tensor([[10.0, 10.0, 60.0, 60.0]]))[0], a)   # eval is deterministic
    model.train()
    with torch.autocast('cpu', dtype=torch.bfloat16):
        outs = model(x, torch.tensor([[10.0, 10.0, 60.0, 60.0]]))
    total, _ = model.loss(outs, torch.zeros(1, 1, 128, 128))
    assert total.dtype == torch.float32 and torch.isfinite(total)
    total.backward()


def test_sam2box_identity_fold_keeps_the_pretrained_rgb_stem():
    '''Zero-shot control: P = I, q = 0 leaves the RGB kernel and bias untouched and adds a zero box slice.'''
    pytest.importorskip('sam2')
    from sam2.build_sam import build_sam2
    torch.manual_seed(0)
    ref = build_sam2(TINY_CFG, None, device='cpu', mode='eval', hydra_overrides_extra=['++model.image_size=128'],
                     apply_postprocessing=False).image_encoder.trunk.patch_embed.proj
    torch.manual_seed(0)
    model = build_seg_model(_args(), 3, np.eye(3, dtype=np.float32), np.zeros(3, np.float32))
    stem = model.sam.image_encoder.trunk.patch_embed.proj
    torch.testing.assert_close(stem.weight[:, :3], ref.weight)
    torch.testing.assert_close(stem.bias, ref.bias)
    assert torch.equal(stem.weight[:, 3], torch.zeros_like(stem.weight[:, 3]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python -m pytest tests/test_seg_sam2box.py -v`
Expected: FAIL at collection with `ImportError: cannot import name 'LoRALinear' from 'models.seg_models'`.

- [ ] **Step 3: Write minimal implementation**

3a. Add these two lines to the import block of `models/seg_models.py` (Task 8's imports stay). Do NOT import `sam2` at module level: `import sam2` initialises hydra globally, and the module must import without sam2 so that the tests skip cleanly.

```python
import math
import torch.nn.functional as F
```

3b. Insert after Task 8's `SAM2UNetSeg` class and before its `SEG_MODELS = {...}` line:

```python
# ---------------------------------------------------------------------------------------------------------------------
# SAM2.1 + box prompt + LoRA
# ---------------------------------------------------------------------------------------------------------------------

SAM2_CFG = 'configs/sam2.1/sam2.1_hiera_l.yaml'             # hydra config inside the pip sam2 package (SAM2.1 Hiera-L)


class LoRALinear(nn.Module):
    '''
    Low-rank adapter around a frozen nn.Linear: y = W x + b + (alpha / r) * B(A(x)).
    A starts Kaiming-uniform and B at zero, so at initialisation the wrapped layer IS the pretrained one (the zero-shot
    model is unchanged until training moves B). W and b are frozen here; only A and B train.
    '''

    def __init__(self, base, r=8, alpha=8):
        super(LoRALinear, self).__init__()
        assert isinstance(base, nn.Linear), f"LoRALinear wraps an nn.Linear, got {type(base).__name__}"
        assert r > 0, f"LoRA rank must be positive, got {r}"
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.in_features, self.out_features = base.in_features, base.out_features
        self.lora_A = nn.Linear(base.in_features, r, bias=False)             # [r, in]
        self.lora_B = nn.Linear(r, base.out_features, bias=False)            # [out, r]
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.scale = float(alpha) / float(r)

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(x)) * self.scale


def dice_bce_loss(logits, mask, smooth=1.0):
    '''
    Mean BCE over pixels plus the soft Dice loss per image, averaged over the batch (spec §7, SAM2.1 + box).
    logits, mask: [B, 1, h, w]; computed in fp32 (logits may come out of bf16 autocast). smooth keeps the Dice term
    finite and near 0 for an all-zero target with an all-zero prediction (false-positive items).
    '''
    logits, mask = logits.float(), mask.float()
    bce = F.binary_cross_entropy_with_logits(logits, mask, reduction='mean')
    prob = torch.sigmoid(logits)
    inter = (prob * mask).sum(dim=(2, 3))                                    # [B, 1]
    denom = prob.sum(dim=(2, 3)) + mask.sum(dim=(2, 3))                       # [B, 1]
    dice = 1.0 - (2.0 * inter + smooth) / (denom + smooth)
    return bce, dice.mean()


class SAM2BoxSeg(nn.Module):
    '''
    Box-prompted SAM2.1 (spec §3 "SAM2.1 + box + LoRA", §5.4), built from the pip sam2 package at image size = canvas.

    Trainable: the folded N+1-channel patch-embed stem (sam.image_encoder.trunk.patch_embed.proj), LoRA (r = args.lora_r)
    on every Hiera block's attn.qkv and attn.proj (exact names: stage-change blocks also own a block-level Linear `proj`,
    which stays frozen), and the mask decoder (its conv_s0 / conv_s1 high-resolution projections included).
    Frozen: the rest of the Hiera trunk, the FPN neck, the prompt encoder, no_mem_embed, the memory modules (unused for
    single images), and the decoder's IoU and object-score heads (a mask-only loss gives them no gradient; frozen, the
    trainable set equals the set that receives gradients, which also keeps DDP free of find_unused_parameters).

    The image path is SAM2's own single-image path (sam2_base.forward_image -> _prepare_backbone_features, + no_mem_embed on
    the stride-16 level, as SAM2ImagePredictor.set_image does); the predictor itself is not used because it hard-codes the
    1024-px feature sizes. The box goes in the way SAM2ImagePredictor._predict passes it, as two points labelled 2 / 3 in
    canvas pixels (x first) plus the padding point, the path SAM2 was trained with. The decoder's single 128 x 128 (canvas /
    4) mask is upsampled bilinearly to the canvas. Built with apply_postprocessing=False: the default switches on
    dynamic_multimask_via_stability, which in eval mode can swap the single mask for a multimask one, so training and
    evaluation would see different heads. The decoder's object-score gating (NO_OBJ_SCORE in sam2_base) is not applied.

    Zero-shot control (spec §3): arm 'rgb' with P = eye(3), q = zeros(3) folds the stem into exactly the pretrained RGB
    kernel plus a zero box slice and LoRA's B is zero, so the untrained model is SAM2.1-L at image size canvas.
    '''

    def __init__(self, args, n_in, P, q):
        super(SAM2BoxSeg, self).__init__()
        from sam2.build_sam import build_sam2                                    # here, not at module top: `import sam2` initialises hydra
        canvas = int(args.canvas)
        assert canvas % 32 == 0, f"SAM2 needs a canvas divisible by 32 (Hiera stride 4 x window 8), got {canvas}"
        cfg = getattr(args, 'sam2_cfg', None) or SAM2_CFG
        ckpt = getattr(args, 'sam2_ckpt', None) or None                          # None / '' -> random init (unit tests only)
        assert ckpt is None or os.path.isfile(ckpt), f"--sam2_ckpt {ckpt} not found: run bash_files/setup_third_party.sh"
        sam = build_sam2(cfg, ckpt_path=ckpt, device='cpu', mode='train', hydra_overrides_extra=[f'++model.image_size={canvas}'],
                         apply_postprocessing=False)                             # strict load of the {"model": state_dict} file
        assert sam.image_size == canvas and tuple(sam.sam_prompt_encoder.input_image_size) == (canvas, canvas), \
            f"image size override failed: model {sam.image_size}, prompt encoder {sam.sam_prompt_encoder.input_image_size}"
        for p in sam.parameters():
            p.requires_grad = False

        # LoRA on the attention projections of every Hiera block (48 blocks in L, 12 in T)
        r = int(getattr(args, 'lora_r', 8) or 8)
        alpha = float(getattr(args, 'lora_alpha', None) or r)                   # alpha = r -> LoRA scale 1
        trunk = sam.image_encoder.trunk
        for blk in trunk.blocks:
            blk.attn.qkv = LoRALinear(blk.attn.qkv, r, alpha)
            blk.attn.proj = LoRALinear(blk.attn.proj, r, alpha)
        self.n_lora = 2 * len(trunk.blocks)

        # N + 1-channel stem: swapped in AFTER the freeze, so the new conv's parameters stay trainable
        old = trunk.patch_embed.proj                                             # Conv2d(3, C, 7, stride 4, pad 3, bias=True)
        assert old.in_channels == 3, f"expected the pretrained RGB patch embed, got {old.in_channels} input channels"
        trunk.patch_embed.proj = fold_stem(old, P, q, n_extra=1)                 # Conv2d(n_in + 1, C, 7, 4, 3)
        assert trunk.patch_embed.proj.in_channels == n_in + 1

        # mask decoder trained, except the two heads the mask loss does not reach
        dec = sam.sam_mask_decoder
        for p in dec.parameters():
            p.requires_grad = True
        unused = [dec.iou_prediction_head] + ([dec.pred_obj_score_head] if getattr(dec, 'pred_obj_scores', False) else [])
        for m in unused:
            for p in m.parameters():
                p.requires_grad = False
        self.sam = sam
        self.n_in = int(n_in)
        self.canvas = canvas

    def set_progress(self, t):
        '''Training progress t in [0, 1] (used by ZoomNeXt's loss ramp only; nothing here depends on it).'''
        return None

    def forward(self, x, box_xyxy):
        '''
        x: [B, n_in + 1, c, c] (arm channels * valid, then the box channel); box_xyxy: [B, 4] canvas pixels of the
        pre-expansion box. Returns [logits [B, 1, c, c]] (one output, no deep supervision).
        '''
        B, C, h, w = x.shape
        assert C == self.n_in + 1, f"expected {self.n_in + 1} input channels, got {C}"
        assert h == w == self.canvas, f"SAM2BoxSeg was built for a {self.canvas} canvas, got {h} x {w}"
        assert box_xyxy.shape == (B, 4), f"box_xyxy must be [B, 4], got {tuple(box_xyxy.shape)}"
        sam = self.sam
        backbone_out = sam.forward_image(x)                                      # trunk + FPN; levels 0/1 through conv_s0/conv_s1
        _, vision_feats, _, feat_sizes = sam._prepare_backbone_features(backbone_out)   # list of [HW, B, C]
        if sam.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + sam.no_mem_embed               # single image: "no memory" embedding (predictor path)
        feats = [f.permute(1, 2, 0).reshape(B, -1, *hw) for f, hw in zip(vision_feats, feat_sizes)]
        # feats: [B, 32, c/4, c/4], [B, 64, c/8, c/8], [B, 256, c/16, c/16]
        coords = box_xyxy.to(device=x.device, dtype=torch.float32).reshape(B, 2, 2)      # [B, 2, 2] (x1, y1), (x2, y2)
        labels = torch.tensor([[2, 3]], dtype=torch.int, device=x.device).expand(B, 2)    # box corners: top-left 2, bottom-right 3
        sparse, dense = sam.sam_prompt_encoder(points=(coords, labels), boxes=None, masks=None)   # [B, 3, 256], [B, 256, c/16, c/16]
        low_res, _, _, _ = sam.sam_mask_decoder(image_embeddings=feats[-1], image_pe=sam.sam_prompt_encoder.get_dense_pe(),
                                                sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
                                                multimask_output=False, repeat_image=False, high_res_features=feats[:-1])
        return [F.interpolate(low_res, size=(h, w), mode='bilinear', align_corners=False)]   # [B, 1, c, c]

    def loss(self, outputs, mask):
        bce, dice = dice_bce_loss(outputs[0], mask)
        total = bce + dice
        return total, {'bce': float(bce.detach()), 'dice': float(dice.detach())}

    def param_groups(self, args):
        '''
        AdamW groups: the folded stem at args.stem_lr without weight decay (it starts from the pretrained RGB kernel, the
        convention of Task 8's param_groups_by_name and of main_det's first conv), LoRA and the mask decoder at args.lr
        (spec §7: 1e-4 for all three).
        '''
        stem, lora, decoder = [], [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith('sam.image_encoder.trunk.patch_embed.'):
                stem.append(p)
            elif '.lora_A.' in name or '.lora_B.' in name:
                lora.append(p)
            elif name.startswith('sam.sam_mask_decoder.'):
                decoder.append(p)
            else:
                raise AssertionError(f"unexpected trainable parameter {name}")
        return [dict(params=stem, lr=float(args.stem_lr), weight_decay=0.0, name='stem'),
                dict(params=lora, lr=float(args.lr), weight_decay=float(args.weight_decay), name='lora'),
                dict(params=decoder, lr=float(args.lr), weight_decay=float(args.weight_decay), name='decoder')]
```

3c. Register the model: in `models/seg_models.py` replace Task 8's line

```python
SEG_MODELS = {'sam2unet': SAM2UNetSeg}
```

with

```python
SEG_MODELS = {'sam2unet': SAM2UNetSeg, 'sam2box': SAM2BoxSeg}
```

It stays below the `SAM2BoxSeg` class. `build_seg_model` is unchanged (it already dispatches through `SEG_MODELS`, checks the P/q shapes and prints the parameter count).

- [ ] **Step 4: Run test to verify it passes**

Run: `OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python -m pytest tests/test_seg_sam2box.py -v`

Expected: PASS, `8 passed` in about 6 s once T7 has installed sam2. Without sam2 the result is `2 passed, 6 skipped`. Verified in a scratch mirror against pip sam2 @ 2b90b9f with torch 2.11 on CPU.

Then run the full suite: `OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python -m pytest tests/ -q`. It stays green (`tests/test_seg_models.py` included).

- [ ] **Step 5: Commit and push**

```bash
git add models/seg_models.py tests/test_seg_sam2box.py
git commit -m "feat(seg): SAM2BoxSeg - box-prompted SAM2.1 with LoRA r=8 and a folded N+1-channel stem

Alternative model family of the Stage-2 spec: SAM2.1-L at the 512 canvas, box prompt via the predictor's point
path, mask decoder + LoRA + stem trained with Dice + BCE; registered in SEG_MODELS.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QB4DScDwiA2DkVpbDopffV"
git push origin worktree-det
git push origin worktree-det:master
```

---

### Task 10: ZoomNeXtSeg (ZoomNeXt-B2, COD checkpoint, folded stem) in models/seg_models.py

**Files:**
- Modify: `models/seg_models.py`. Add `sys`, `types`, `warnings` and `contextlib` to the imports. Insert `ZOOMNEXT_DIR`, `ZOOMNEXT_CKPT`, `ZOOMNEXT_SCALES`, `no_cuda_query`, `import_zoomnext`, `zoomnext_ual_coef` and `ZoomNeXtSeg` between `SAM2BoxSeg` and the `SEG_MODELS` line. Register `'zoomnext'` in that line.
- Test: `tests/test_seg_zoomnext.py` (new)

**Interfaces:**
- Consumes:
  - `models.seg_stem.fold_stem(conv, P, q, n_extra=1)` (T6)
  - `SEG_MODELS` / `build_seg_model(args, n_in, P, q)` (T8, extended in T9)
  - the T7 setup: `third_party/zoomnext` = github.com/lartpang/ZoomNeXt @ 614af4348808734aaf2ec43937cf827410330b67 (git-ignored), `einops` in the env, `weights/pretrained/pvtv2-b2-zoomnext.pth`
- Produces:
  - `ZOOMNEXT_DIR` (`<repo>/third_party/zoomnext`)
  - `ZOOMNEXT_CKPT = 'weights/pretrained/pvtv2-b2-zoomnext.pth'` (the default for T12's `--zoomnext_ckpt`)
  - `no_cuda_query()` (a context manager)
  - `import_zoomnext(zoomnext_dir=ZOOMNEXT_DIR) -> PvtV2B2_ZoomNeXt`
  - `zoomnext_ual_coef(t) -> float`
  - `ZoomNeXtSeg(args, n_in, P, q)` with:
    - `forward(x [B, n_in+1, c, c], box_xyxy=None) -> [logits [B,1,c,c]]`
    - `loss(outputs, mask) -> (total, {'bce', 'ual', 'ual_coef'})`
    - `set_progress(t)`
    - `param_groups(args) -> [stem @ args.stem_lr (weight decay 0), encoder @ args.lr * encoder_lr_mult, decoder @ args.lr]`
  - Attributes it reads from `args`:
    - required: `seg_model`, `canvas` (a multiple of 64), `lr`, `stem_lr`, `weight_decay`
    - optional through getattr: `zoomnext_ckpt` (`''` or None gives a random init, for tests only), `zoomnext_dir` (default `ZOOMNEXT_DIR`), `grad_ckpt` (PVT gradient checkpointing, default False), `encoder_lr_mult` (default 1.0 = spec §7; ZoomNeXt's original recipe is 0.1)
  - Contract for T11 and T12:
    - Training calls `model.set_progress((epoch + i / len(loader)) / epochs)` every step. Before the first call the default is 1.0, the full UAL weight.
    - Training batches must have at least 2 items, so the train DataLoader uses `drop_last=True` (T12).

- [ ] **Step 1: Write the failing test**

Create `tests/test_seg_zoomnext.py`:

```python
import os
import math
import types
import numpy as np
import torch
import pytest

from models.seg_models import ZOOMNEXT_DIR, ZoomNeXtSeg, build_seg_model, import_zoomnext, no_cuda_query, zoomnext_ual_coef
from models.seg_stem import fold_stem

# ZoomNeXt is fetched by bash_files/setup_third_party.sh into git-ignored third_party/zoomnext and needs einops
requires_zoomnext = pytest.mark.skipif(not os.path.isfile(os.path.join(ZOOMNEXT_DIR, 'methods', 'zoomnext', 'zoomnext.py')),
                                       reason="third_party/zoomnext missing: run bash_files/setup_third_party.sh")


def _args(**kw):
    '''main_seg-like namespace for a random-init ZoomNeXt-B2 at canvas 128 (no checkpoint download).'''
    a = dict(seg_model='zoomnext', zoomnext_ckpt='', canvas=128, lr=1e-4, stem_lr=5e-5, weight_decay=1e-4)
    a.update(kw)
    return types.SimpleNamespace(**a)


def _pq(n, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((3, n)) * 0.1).astype(np.float32), (rng.standard_normal(3) * 0.1).astype(np.float32)


def test_ual_coef_ramps_from_zero_to_one():
    assert zoomnext_ual_coef(0.0) == 0.0 and zoomnext_ual_coef(1.0) == 1.0
    assert zoomnext_ual_coef(0.5) == pytest.approx(0.5)
    assert zoomnext_ual_coef(-1.0) == 0.0 and zoomnext_ual_coef(2.0) == 1.0
    assert zoomnext_ual_coef(0.25) == pytest.approx((1 - math.cos(math.pi * 0.25)) / 2)


@requires_zoomnext
def test_zoomnext_rejects_a_canvas_not_divisible_by_64():
    pytest.importorskip('einops')
    P, q = _pq(3)
    with pytest.raises(AssertionError, match='divisible by 64'):
        build_seg_model(_args(canvas=96), 3, P, q)


@requires_zoomnext
@pytest.mark.parametrize('n', [3, 10, 24])
def test_zoomnext_builds_and_trains_at_n_channels(n):
    pytest.importorskip('einops')
    torch.manual_seed(0)
    P, q = _pq(n)
    model = build_seg_model(_args(), n, P, q)
    assert isinstance(model, ZoomNeXtSeg)
    stem = model.net.encoder.patch_embed1.proj
    assert stem.in_channels == n + 1 and stem.out_channels == 64 and stem.stride == (4, 4)
    assert torch.equal(stem.weight[:, n], torch.zeros_like(stem.weight[:, n]))  # box-channel slice starts at zero
    assert all(p.requires_grad for p in model.parameters())                      # full fine-tune, stem unfrozen

    model.train()
    x = torch.randn(2, n + 1, 128, 128)
    outs = model(x, torch.zeros(2, 4))
    assert isinstance(outs, list) and len(outs) == 1 and outs[0].shape == (2, 1, 128, 128)
    mask = (torch.rand(2, 1, 128, 128) > 0.8).float()
    model.set_progress(0.3)
    total, items = model.loss(outs, mask)
    assert set(items) == {'bce', 'ual', 'ual_coef'} and all(isinstance(v, float) for v in items.values())
    assert items['ual_coef'] == pytest.approx(zoomnext_ual_coef(0.3))
    assert float(total.detach()) == pytest.approx(items['bce'] + items['ual_coef'] * items['ual'], rel=1e-5)
    total.backward()
    assert stem.weight.grad is not None and stem.weight.grad[:, n].abs().sum() > 0   # the box channel learns

    groups = model.param_groups(_args(encoder_lr_mult=0.1))
    assert [g['name'] for g in groups] == ['stem', 'encoder', 'decoder']
    assert len(groups[0]['params']) == 4                                         # patch_embed1.proj.{weight,bias}, .norm.{weight,bias}
    assert groups[0]['lr'] == 5e-5 and groups[1]['lr'] == pytest.approx(1e-5) and groups[2]['lr'] == 1e-4
    assert model.param_groups(_args())[1]['lr'] == 1e-4                           # default: one lr for the whole network (spec §7)
    ids = [id(p) for g in groups for p in g['params']]
    assert len(ids) == len(set(ids)) == len(list(model.parameters()))
    torch.optim.AdamW(groups)


@requires_zoomnext
def test_zoomnext_progress_zero_is_plain_bce_and_eval_is_deterministic():
    pytest.importorskip('einops')
    torch.manual_seed(0)
    P, q = _pq(10)
    model = build_seg_model(_args(), 10, P, q)
    assert model.progress == 1.0                                                  # ZoomNeXt's own default: the full UAL weight
    model.set_progress(0.0)
    logits = [torch.randn(1, 1, 128, 128)]
    mask = torch.zeros(1, 1, 128, 128)
    total, items = model.loss(logits, mask)
    assert items['ual_coef'] == 0.0
    torch.testing.assert_close(total, torch.nn.functional.binary_cross_entropy_with_logits(logits[0], mask))
    model.eval()
    x = torch.randn(1, 11, 128, 128)
    with torch.no_grad():
        torch.testing.assert_close(model(x)[0], model(x, torch.zeros(1, 4))[0])
    model.train()
    with torch.autocast('cpu', dtype=torch.bfloat16):
        outs = model(torch.randn(2, 11, 128, 128))
    total, _ = model.loss(outs, torch.zeros(2, 1, 128, 128))
    assert total.dtype == torch.float32 and torch.isfinite(total)
    with pytest.raises(AssertionError, match='batch of at least 2'):
        model(torch.randn(1, 11, 128, 128))                                      # SimpleASPP's pooled BatchNorm branch


@requires_zoomnext
def test_zoomnext_loads_the_cod_checkpoint_before_folding_the_stem(tmp_path):
    '''A checkpoint shaped like pvtv2-b2-zoomnext.pth (normalizer.* present, num_batches_tracked absent) loads, and the
    stem is the fold of the CHECKPOINT's RGB kernel.'''
    pytest.importorskip('einops')
    Net = import_zoomnext()
    torch.manual_seed(1)
    with no_cuda_query():
        ref = Net(pretrained=False, num_frames=1, input_norm=True)
    sd = {k: v for k, v in ref.state_dict().items() if not k.endswith('num_batches_tracked')}
    assert any(k.startswith('normalizer.') for k in sd)
    path = str(tmp_path / 'pvtv2-b2-zoomnext.pth')
    torch.save(sd, path)

    torch.manual_seed(2)                                                          # a different random init underneath
    P, q = _pq(10)
    model = build_seg_model(_args(zoomnext_ckpt=path), 10, P, q)
    own = model.net.state_dict()
    for k in ('encoder.block1.0.attn.q.weight', 'encoder.patch_embed2.proj.weight', 'encoder.patch_embed1.norm.weight',
              'tra_5.conv1x1_1.conv.weight', 'tra_5.conv1x1_1.bn.running_var', 'predictor.2.weight'):
        torch.testing.assert_close(own[k], sd[k])
    folded = fold_stem(ref.encoder.patch_embed1.proj, P, q, n_extra=1)
    torch.testing.assert_close(own['encoder.patch_embed1.proj.weight'], folded.weight.data)
    torch.testing.assert_close(own['encoder.patch_embed1.proj.bias'], folded.bias.data)
    assert not any(k.startswith('normalizer.') for k in own)

    bad = dict(sd)
    bad['encoder.block1.0.attn.q.weight'] = torch.zeros(3, 3)
    torch.save(bad, path)
    with pytest.raises(AssertionError, match='shape mismatch'):
        build_seg_model(_args(zoomnext_ckpt=path), 10, P, q)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python -m pytest tests/test_seg_zoomnext.py -v`
Expected: FAIL at collection with `ImportError: cannot import name 'ZOOMNEXT_DIR' from 'models.seg_models'`.

- [ ] **Step 3: Write minimal implementation**

3a. Add the following to the import block of `models/seg_models.py`; the Task 8 / Task 9 imports stay:

```python
import sys
import types
import warnings
import contextlib
```

3b. Insert between the end of `SAM2BoxSeg` and the `SEG_MODELS` line:

```python
# ---------------------------------------------------------------------------------------------------------------------
# ZoomNeXt-B2
# ---------------------------------------------------------------------------------------------------------------------

ZOOMNEXT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'third_party', 'zoomnext')
ZOOMNEXT_CKPT = 'weights/pretrained/pvtv2-b2-zoomnext.pth'                   # COD checkpoint (setup_third_party.sh)
ZOOMNEXT_SCALES = (0.5, 1.0, 1.5)                                            # image_s / image_m / image_l (main_for_image.ms_resize)


@contextlib.contextmanager
def no_cuda_query():
    '''
    ZoomNeXt's PVTv2 (methods/backbone/pvt_v2_eff.py:89, Attention.__init__) calls torch.cuda.get_device_properties('cuda')
    unconditionally to choose its SDPA kernels, so without a GPU (CPU tests, CUDA_VISIBLE_DEVICES="") construction raises
    "No CUDA GPUs are available". Stub the query during construction only, and only when CUDA is absent: the clone stays
    pristine at its pinned commit and a GPU machine takes the original path (A4500 = sm_86 -> math + mem-efficient kernels).
    '''
    orig = torch.cuda.get_device_properties
    if not torch.cuda.is_available():
        torch.cuda.get_device_properties = lambda *a, **k: types.SimpleNamespace(major=0, minor=0)
    try:
        yield
    finally:
        torch.cuda.get_device_properties = orig


def import_zoomnext(zoomnext_dir=ZOOMNEXT_DIR):
    '''
    PvtV2B2_ZoomNeXt from the git-ignored clone (bash_files/setup_third_party.sh, pinned commit). The clone is APPENDED to
    sys.path: its top-level configs/ and utils/ must not shadow anything, and only its `methods` package is imported
    (the model code needs timm and einops; ZoomNeXt's own utils/, mmengine and albumentations are not used).
    '''
    zoomnext_dir = os.path.realpath(zoomnext_dir)
    src = os.path.join(zoomnext_dir, 'methods', 'zoomnext', 'zoomnext.py')
    assert os.path.isfile(src), f"ZoomNeXt code not found at {src}: run bash_files/setup_third_party.sh"
    if zoomnext_dir not in sys.path:
        sys.path.append(zoomnext_dir)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)                       # timm.models.layers is deprecated in timm 1.x
        import methods
        from methods.zoomnext.zoomnext import PvtV2B2_ZoomNeXt
    found = os.path.realpath(os.path.dirname(methods.__file__))
    assert found == os.path.join(zoomnext_dir, 'methods'), f"`methods` was imported from {found}, not from the ZoomNeXt clone {zoomnext_dir}"
    return PvtV2B2_ZoomNeXt


def zoomnext_ual_coef(t):
    '''ZoomNeXt's uncertainty-aware loss weight: (1 - cos(pi t)) / 2, 0 -> 1 over training (get_coef, method 'cos', milestones (0, 1)).'''
    t = min(max(float(t), 0.0), 1.0)
    return (1.0 - math.cos(math.pi * t)) / 2.0


class ZoomNeXtSeg(nn.Module):
    '''
    ZoomNeXt-B2 (PvtV2B2_ZoomNeXt from third_party/zoomnext) with its image-COD checkpoint, fully fine-tuned (spec §3, §5.4).

    - Built with input_norm=False: the original PixelNormalizer holds [3, 1, 1] ImageNet buffers that cannot broadcast to
      N + 1 channels, and the ImageNet normalisation is already inside the folded stem (rgb_norm ~ P z + q).
    - The checkpoint is loaded into the 3-channel model FIRST (normalizer.* dropped; it lacks the BatchNorm
      num_batches_tracked buffers, so it is overlaid on the model's own state dict before a strict load, as ZoomNeXt's
      utils/io/params.py does), THEN encoder.patch_embed1.proj (Conv2d(3, 64, 7, stride 4, pad 3)) is folded into the
      n_in + 1-channel stem, so the fold starts from the COD-trained RGB kernel. patch_embed1.norm keeps its weights.
      Unlike the original (get_grouped_params freezes patch_embed1), every parameter trains, the stem included.
    - forward builds the multi-scale triplet from the canvas, scales 0.5 / 1.0 / 1.5 bilinear (the box channel is
      resized with it), runs ZoomNeXt's body() in both modes and returns [logits [B, 1, c, c]]; box_xyxy is unused (the
      box enters through the box channel). The module's own forward() is not used: in training it needs data['mask'] and
      returns a dict with its own loss.
    - loss = BCE + ual_coef(t) * mean(1 - |2 p - 1|^2), ZoomNeXt's own loss; t is the training progress set by
      set_progress() (train_one_epoch calls it every step), defaulting to 1 (the full weight, ZoomNeXt's forward default).
    The canvas must be divisible by 64 so that the 0.5 scale stays divisible by PVT's stride 32.
    '''

    def __init__(self, args, n_in, P, q):
        super(ZoomNeXtSeg, self).__init__()
        canvas = int(args.canvas)
        assert canvas % 64 == 0, f"ZoomNeXt needs a canvas divisible by 64 (0.5 scale / PVT stride 32), got {canvas}"
        Net = import_zoomnext(getattr(args, 'zoomnext_dir', None) or ZOOMNEXT_DIR)
        with no_cuda_query(), warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            net = Net(pretrained=False, num_frames=1, input_norm=False,
                      use_checkpoint=bool(getattr(args, 'grad_ckpt', False)))   # pretrained=False: the COD ckpt holds the whole encoder
        ckpt = getattr(args, 'zoomnext_ckpt', None) or None                     # None / '' -> random init (unit tests only)
        if ckpt is not None:
            assert os.path.isfile(ckpt), f"--zoomnext_ckpt {ckpt} not found: run bash_files/setup_third_party.sh"
            sd = torch.load(ckpt, map_location='cpu', weights_only=True)
            assert isinstance(sd, dict), f"{ckpt}: expected a plain state_dict, got {type(sd).__name__}"
            sd = {k: v for k, v in sd.items() if not k.startswith('normalizer.')}
            own = net.state_dict()
            unexpected = sorted(set(sd) - set(own))
            missing = sorted(k for k in set(own) - set(sd) if not k.endswith('num_batches_tracked'))
            assert not unexpected and not missing, f"{ckpt}: unexpected keys {unexpected[:5]}, missing keys {missing[:5]}"
            bad = [k for k in sd if tuple(sd[k].shape) != tuple(own[k].shape)]
            assert not bad, f"{ckpt}: shape mismatch on {bad[:5]}"
            own.update(sd)
            net.load_state_dict(own, strict=True)
        old = net.encoder.patch_embed1.proj                                     # Conv2d(3, 64, 7, stride 4, pad 3, bias=True)
        assert old.in_channels == 3, f"expected the RGB patch_embed1, got {old.in_channels} input channels"
        net.encoder.patch_embed1.proj = fold_stem(old, P, q, n_extra=1)         # Conv2d(n_in + 1, 64, 7, 4, 3)
        for p in net.parameters():
            p.requires_grad = True                                               # full fine-tune, stem included
        self.net = net
        self.n_in = int(n_in)
        self.canvas = canvas
        self.progress = 1.0

    def set_progress(self, t):
        '''Training progress t in [0, 1] (iteration / total iterations) for the UAL weight ramp.'''
        self.progress = float(t)

    def forward(self, x, box_xyxy=None):
        '''x: [B, n_in + 1, c, c]; box_xyxy unused (API parity). Returns [logits [B, 1, c, c]].'''
        B, C, h, w = x.shape
        assert C == self.n_in + 1, f"expected {self.n_in + 1} input channels, got {C}"
        assert h % 64 == 0 and w % 64 == 0, f"ZoomNeXt needs sides divisible by 64, got {h} x {w}"
        # SimpleASPP's global-pool branch has a BatchNorm on a [B, 64, 1, 1] map: one item gives a single value per channel,
        # which BatchNorm rejects in training (the train loader uses drop_last=True)
        assert B >= 2 or not self.training, f"ZoomNeXt trains on a batch of at least 2 items, got {B}"
        s, m, l = ZOOMNEXT_SCALES
        data = {'image_s': F.interpolate(x, size=(int(h * s), int(w * s)), mode='bilinear', align_corners=False),   # [B, C, c/2, c/2]
                'image_m': x,                                                                                        # [B, C, c, c]
                'image_l': F.interpolate(x, size=(int(h * l), int(w * l)), mode='bilinear', align_corners=False)}   # [B, C, 1.5c, 1.5c]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)                       # pvt_v2_eff's torch.backends.cuda.sdp_kernel is deprecated
            logits = self.net.body(data)                                         # [B, 1, c, c] (the image_m size)
        return [logits]

    def loss(self, outputs, mask):
        logits = outputs[0].float()
        bce = F.binary_cross_entropy_with_logits(logits, mask.float(), reduction='mean')
        prob = logits.sigmoid()
        ual = (1.0 - (2.0 * prob - 1.0).abs().pow(2)).mean()                   # pushes probabilities away from 0.5
        coef = zoomnext_ual_coef(self.progress)
        total = bce + coef * ual
        return total, {'bce': float(bce.detach()), 'ual': float(ual.detach()), 'ual_coef': float(coef)}

    def param_groups(self, args):
        '''
        AdamW groups: the stem (encoder.patch_embed1.*, the folded conv and its LayerNorm) at args.stem_lr without weight
        decay (the convention of Task 8's param_groups_by_name), the rest of the PVT encoder at args.lr *
        args.encoder_lr_mult (default 1: spec §7, lr 1e-4 for the whole network; the original recipe uses 0.1), the
        decoder at args.lr.
        '''
        mult = float(getattr(args, 'encoder_lr_mult', None) or 1.0)
        stem, encoder, decoder = [], [], []
        for name, p in self.net.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith('encoder.patch_embed1.'):
                stem.append(p)
            elif name.startswith('encoder.'):
                encoder.append(p)
            else:
                decoder.append(p)
        return [dict(params=stem, lr=float(args.stem_lr), weight_decay=0.0, name='stem'),
                dict(params=encoder, lr=float(args.lr) * mult, weight_decay=float(args.weight_decay), name='encoder'),
                dict(params=decoder, lr=float(args.lr), weight_decay=float(args.weight_decay), name='decoder')]
```

3c. Register the model. In `models/seg_models.py` replace

```python
SEG_MODELS = {'sam2unet': SAM2UNetSeg, 'sam2box': SAM2BoxSeg}
```

with

```python
SEG_MODELS = {'sam2unet': SAM2UNetSeg, 'sam2box': SAM2BoxSeg, 'zoomnext': ZoomNeXtSeg}
```

The `build_seg_model` body from Task 8 is unchanged.

- [ ] **Step 4: Run test to verify it passes**

Run: `OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python -m pytest tests/test_seg_zoomnext.py -v`

Expected: PASS, `7 passed` in about 14 s once T7 has cloned `third_party/zoomnext` and installed einops. If either is missing, the result is `1 passed, 6 skipped`. Verified in a scratch mirror against ZoomNeXt @ 614af43 with einops 0.8.1, timm 1.0.30 and torch 2.11 on CPU, without warnings.

Then run the full suite: `OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python -m pytest tests/ -q`. It stays green.

- [ ] **Step 5: Commit and push**

```bash
git add models/seg_models.py tests/test_seg_zoomnext.py
git commit -m "feat(seg): ZoomNeXtSeg - ZoomNeXt-B2 from third_party/zoomnext with the COD checkpoint and a folded stem

Dedicated COD baseline of the Stage-2 spec: checkpoint loaded into the RGB model first, then patch_embed1 folded and
unfrozen; multi-scale triplet built from the canvas; BCE + progress-ramped UAL loss; CPU construction via no_cuda_query.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QB4DScDwiA2DkVpbDopffV"
git push origin worktree-det
git push origin worktree-det:master
```

---

### Task 11: Stage-2 training / evaluation loops and full-frame paste-back

**Files:**
- Create: `train_eval/train_eval_seg.py`
- Test: `tests/test_train_eval_seg.py`

**Interfaces:**
- Consumes:
  - T4 `data_loader.roi_crops`: `place_on_canvas(crop, canvas=512, scale=1.0) -> (out, valid, (oy, ox), s)` (tests only), `canvas_to_roi(canvas_map, (oy, ox), s, (h, w)) -> [h, w] float32`, `seg_collate_fn(batch) -> {"img", "mask", "box_map", "valid", "box_xyxy", "meta"}`. Each item's meta must contain `frame`, `roi`, `source` in `'gt'|'det'|'fp'`, `offset`, `s` and `roi_hw`.
  - T5 `train_eval.seg_metrics.SegMetrics(size_edges)`, with `.update(pred, gt)`, `.update_fp(pred)` and `.summary()`. The summary keys include `S, Fw, MAE, IoU, n, n_fp, fp_false_mask_rate, S_small, n_small`.
  - T8-T10 model API: `forward(x, box_xyxy) -> [logits, ...]`, `loss(outputs, mask) -> (total, {str: float})`, an optional `set_progress(t)`, and `param_groups(args)`, whose groups may carry a `'name'` key. The front end has the attribute `.n_out`.
- Produces (T12 and T13 use these):
  - `seg_inputs(front_end, batch, device) -> x [B, N+1, c, c] float32`. The front end runs in fp32 with autocast off and under `no_grad`. The padding is zeroed by `valid`. The box channel is last.
  - `train_one_epoch(model, front_end, data_loader, optimizer, device, epoch, scaler=None, accumulate=1, max_norm=1.0, logger=None, print_freq=10, amp=True, amp_dtype=torch.bfloat16, total_epochs=None) -> dict`. It returns `loss`, `lr` and every loss item.
  - `evaluate(model, front_end, data_loader, device, logger=None, epoch=0, tag='val', amp=True, amp_dtype=torch.bfloat16, size_edges=(2000, 20000), n_images_log=4) -> SegMetrics.summary()`.
  - `paste_back(prob_canvas [c, c], meta, out [H, W] float32) -> out`. It max-merges in place. The crop origin is `(floor(y1), floor(x1))` of `meta['roi']` and its size is `meta['roi_hw']`. That is the floor/ceil rule of `box_metrics.mask_coverage`, and a `roi_hw` from another rule raises.
  - `roi_panel(prob, gt) -> uint8 [h, 2w+2, 3]`

- [ ] **Step 1: Write the failing test**

`tests/test_train_eval_seg.py`:
```python
import types
import numpy as np
import pytest
pytest.importorskip('py_sod_metrics')        # train_eval_seg -> seg_metrics needs it (bash_files/setup_third_party.sh installs it)
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_loader.roi_crops import place_on_canvas, seg_collate_fn
from train_eval.train_eval_seg import seg_inputs, train_one_epoch, evaluate, paste_back
from util.logger import TrainLogger

C = 32                       # canvas side of the hand-made items (the real canvas is 512)


class FixedFront(nn.Module):
    '''Stand-in for an arm front end: a fixed 133 -> 4 projection (+1, so the padding is non-zero before the valid mask).'''

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(4, 133, generator=torch.Generator().manual_seed(0)) / 133)
        self.n_out = 4
        self.dtypes = []                                   # dtype of every input, to check the fp32 front-end rule

    def forward(self, x):
        self.dtypes.append(x.dtype)
        return torch.einsum('nc,bchw->bnhw', self.weight, x) + 1.0     # [B, 4, h, w]


class TinySeg(nn.Module):
    '''The smallest model with the seg API: conv stem + BN + main head + one deep-supervision side head.'''

    def __init__(self, n_in, nan=False):
        super().__init__()
        self.stem = nn.Conv2d(n_in + 1, 4, 3, padding=1)
        self.bn = nn.BatchNorm2d(4)
        self.head, self.side = nn.Conv2d(4, 1, 1), nn.Conv2d(4, 1, 1)
        self.nan = nan
        self.progress, self.seen = [], []

    def forward(self, x, box_xyxy):
        self.seen.append((tuple(x.shape), tuple(box_xyxy.shape)))
        f = torch.relu(self.bn(self.stem(x)))
        return [self.head(f), self.side(f)]                # main output first

    def loss(self, outputs, mask):
        main = F.binary_cross_entropy_with_logits(outputs[0], mask)
        side = F.binary_cross_entropy_with_logits(outputs[1], mask)
        total = main + side
        if self.nan:
            total = total * float('nan')
        return total, {'main': float(main.detach()), 'side': float(side.detach())}

    def param_groups(self, args):
        rest = [p for n, p in self.named_parameters() if not n.startswith('stem.')]
        return [{'params': list(self.stem.parameters()), 'lr': args.stem_lr, 'name': 'stem'}, {'params': rest, 'name': 'rest'}]

    def set_progress(self, t):
        self.progress.append(t)


class BoxEcho(nn.Module):
    '''Predicts exactly the box channel (logit +-10): the perfect model when the object fills its box; empty=True paints nothing.'''

    def __init__(self, empty=False):
        super().__init__()
        self.empty = empty

    def forward(self, x, box_xyxy):
        if self.empty:
            return [torch.full_like(x[:, -1:], -10.0)]
        return [20.0 * x[:, -1:] - 10.0]


def _item(h, w, box, source='gt', seed=0):
    '''One HyperCOD_roi-style item: an h x w ROI at frame origin (x 7, y 5); the object fills box (empty GT for 'fp').'''
    rng = np.random.default_rng(seed)
    crop = (rng.random((133, h, w)) * 0.5).astype(np.float16)                       # [133, h, w] p99-scaled
    gt, bm = np.zeros((1, h, w), np.float32), np.zeros((1, h, w), np.float32)
    x1, y1, x2, y2 = box
    bm[0, y1:y2, x1:x2] = 1.0
    if source != 'fp':
        gt[0, y1:y2, x1:x2] = 1.0
    img, valid, (oy, ox), s = place_on_canvas(crop, canvas=C)
    mask = (place_on_canvas(gt, canvas=C)[0] > 0.5).astype(np.float32)              # [1, C, C]
    box_map = (place_on_canvas(bm, canvas=C)[0] > 0.5).astype(np.float32)           # [1, C, C]
    meta = {'frame': '3', 'roi': [7.0, 5.0, 7.0 + w, 5.0 + h], 'box': [7 + x1, 5 + y1, 7 + x2, 5 + y2],
            'box_canvas': [ox + x1 * s, oy + y1 * s, ox + x2 * s, oy + y2 * s], 'source': source, 'offset': (oy, ox),
            's': s, 'roi_hw': (h, w), 'obj_area': int(gt.sum())}
    return img.astype(np.float16), mask, box_map, valid[None].astype(np.float32), meta


def test_seg_inputs_zero_the_padding_append_the_box_and_run_the_front_end_in_fp32():
    batch = seg_collate_fn([_item(20, 12, (2, 3, 8, 11)), _item(16, 24, (4, 4, 20, 12), seed=1)])
    fe = FixedFront()
    x = seg_inputs(fe, batch, torch.device('cpu'))
    assert x.shape == (2, 5, C, C) and x.dtype == torch.float32
    assert fe.dtypes == [torch.float32]                                            # fp16 crop cast before the front end
    pad = ~batch['valid'].bool().expand(-1, 4, -1, -1)                             # [B, 4, C, C] canvas padding
    assert pad.any() and (x[:, :4][pad] == 0).all() and (x[:, :4][~pad] != 0).any()
    assert torch.equal(x[:, 4:], batch['box_map'].float())                         # last channel = box map
    assert not x.requires_grad                                                     # no graph into the fixed front end


def test_paste_back_round_trips_merges_by_maximum_and_rejects_bad_geometry():
    crop = np.random.default_rng(0).random((1, 10, 6)).astype(np.float32)          # [1, h, w] ROI probabilities
    canv, _, off, s = place_on_canvas(crop, canvas=16)
    assert s == 1.0
    meta = {'frame': '3', 'roi': [3.0, 2.0, 9.0, 12.0], 'offset': off, 's': s, 'roi_hw': (10, 6)}
    out = np.zeros((20, 15), np.float32)
    assert paste_back(canv[0], meta, out) is out
    np.testing.assert_array_equal(out[2:12, 3:9], crop[0])                          # exact without a resize
    assert out.sum() == pytest.approx(float(crop.sum()))                            # nothing outside the ROI
    half = np.full((1, 4, 4), 0.5, np.float32)
    canv2, _, off2, s2 = place_on_canvas(half, canvas=16)
    before = out.copy()
    paste_back(torch.from_numpy(canv2[0]), {'frame': '3', 'roi': [5.0, 8.0, 9.0, 12.0], 'offset': off2, 's': s2, 'roi_hw': (4, 4)}, out)
    np.testing.assert_array_equal(out[8:12, 5:9], np.maximum(before[8:12, 5:9], 0.5))   # max-merge of overlapping ROIs
    keep = np.ones_like(out, bool); keep[8:12, 5:9] = False
    np.testing.assert_array_equal(out[keep], before[keep])
    # a fractional export ROI is cut with floor x1/y1, ceil x2/y2: [2.5, 1.5, 8.5, 9.5] -> rows 1:10, cols 2:9
    frac = np.random.default_rng(1).random((1, 9, 7)).astype(np.float32)
    canv3, _, off3, s3 = place_on_canvas(frac, canvas=16)
    out3 = paste_back(canv3[0], {'frame': '3', 'roi': [2.5, 1.5, 8.5, 9.5], 'offset': off3, 's': s3, 'roi_hw': (9, 7)},
                      np.zeros((20, 15), np.float32))
    np.testing.assert_array_equal(out3[1:10, 2:9], frac[0])
    with pytest.raises(AssertionError):                                            # roi_hw from another rounding rule
        paste_back(canv[0], {**meta, 'roi_hw': (6, 6)}, out)
    with pytest.raises(AssertionError):                                            # ROI leaves the frame
        paste_back(canv[0], {**meta, 'roi': [10.0, 12.0, 16.0, 22.0]}, out)


def test_paste_back_of_a_downscaled_roi_is_within_interpolation_error():
    ramp = np.broadcast_to(np.linspace(0, 1, 24, dtype=np.float32)[None, None], (1, 40, 24)).copy()   # smooth [1, 40, 24]
    canv, _, off, s = place_on_canvas(ramp, canvas=16)
    assert s == pytest.approx(0.4)
    out = paste_back(canv[0], {'frame': '3', 'roi': [1.0, 0.0, 25.0, 40.0], 'offset': off, 's': s, 'roi_hw': (40, 24)},
                     np.zeros((48, 40), np.float32))
    assert np.abs(out[:40, 1:25] - ramp[0]).max() < 0.1 and out[:, 25:].max() == 0 and out[40:].max() == 0


def test_train_one_epoch_accumulates_reports_progress_and_keeps_the_front_end_fixed(tmp_path):
    items = [_item(20, 12, (2, 3, 8, 11), seed=k) for k in range(4)]
    loader = torch.utils.data.DataLoader(items, batch_size=2, shuffle=False, collate_fn=seg_collate_fn)
    model, fe = TinySeg(4), FixedFront()
    opt = torch.optim.AdamW(model.param_groups(types.SimpleNamespace(stem_lr=1e-2)), lr=1e-3)
    w0 = model.head.weight.detach().clone()
    logger = TrainLogger(types.SimpleNamespace(name='t', wandb=False, runs_dir=str(tmp_path / 'runs'), rank=0), cfg={})
    stats = train_one_epoch(model, fe, loader, opt, torch.device('cpu'), epoch=1, accumulate=2, max_norm=1.0, logger=logger,
                            print_freq=1, total_epochs=4)
    logger.finish()
    assert {'loss', 'lr', 'main', 'side'} <= set(stats) and np.isfinite(stats['loss'])
    assert model.progress == pytest.approx([0.25, 0.375])                          # (epoch + i / n_steps) / total_epochs
    assert int(opt.state[model.head.weight]['step']) == 1                          # 2 batches accumulated into one step
    assert not torch.equal(model.head.weight.detach(), w0)
    assert fe.weight.grad is None                                                  # nothing flows into the front end
    assert model.seen[0] == ((2, 5, C, C), (2, 4))


def test_train_one_epoch_stops_on_a_non_finite_loss():
    loader = torch.utils.data.DataLoader([_item(20, 12, (2, 3, 8, 11))], batch_size=1, collate_fn=seg_collate_fn)
    model = TinySeg(4, nan=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with pytest.raises(AssertionError, match='non-finite loss'):
        train_one_epoch(model, FixedFront(), loader, opt, torch.device('cpu'), epoch=0)


def test_evaluate_scores_objects_at_native_size_and_fp_rois_by_false_mask_rate():
    items = [_item(20, 12, (2, 3, 8, 11)), _item(40, 24, (4, 6, 20, 30), seed=1),     # second one is downscaled (s = 0.8)
             _item(16, 16, (4, 4, 12, 12), source='fp', seed=2)]
    loader = torch.utils.data.DataLoader(items, batch_size=2, shuffle=False, collate_fn=seg_collate_fn)
    summary = evaluate(BoxEcho(), FixedFront(), loader, torch.device('cpu'), epoch=0, tag='val')
    assert summary['n'] == 2 and summary['n_fp'] == 1                              # fp items never enter S / E / F
    assert summary['S'] > 0.95 and summary['IoU'] > 0.95 and summary['MAE'] < 0.05  # box echo = the object (edges of the s = 0.8 item interpolate)
    assert summary['fp_false_mask_rate'] == 1.0                                    # it also paints the fp box
    assert 'S_small' in summary and summary['n_small'] == 2
    empty = evaluate(BoxEcho(empty=True), FixedFront(), loader, torch.device('cpu'), epoch=0, tag='val')
    assert empty['fp_false_mask_rate'] == 0.0 and empty['IoU'] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python -m pytest tests/test_train_eval_seg.py -v`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'train_eval.train_eval_seg'`. If pysodmetrics is not installed yet, the module is SKIPPED instead.

- [ ] **Step 3: Write minimal implementation**

`train_eval/train_eval_seg.py`:
```python
'''
Stage-2 training / evaluation loops (spec §5.5): train_one_epoch, evaluate (ROI level) and paste_back (full frame).

Every model of models/seg_models.py shares one API: forward(x [B, N + 1, c, c], box_xyxy [B, 4]) -> list of logits
[B, 1, c, c] (index 0 = main output, deep-supervision outputs after it), loss(outputs, mask) -> (total, items),
param_groups(args), and optionally set_progress(t) (ZoomNeXt's uncertainty-aware loss ramps with the training progress
t in [0, 1]). The arm's front end (raw: standardised bands; ec10 / ec24: the detector's whitened EC readings; rgb: the
ImageNet-normalised pseudo-RGB render) is fixed: it is the detector's own, so it runs without gradients.
'''
import math
import numpy as np
import torch

from util.misc import MetricLogger, SmoothedValue
from data_loader.roi_crops import canvas_to_roi
from train_eval.seg_metrics import SegMetrics

PRINT_KEYS = ('S', 'E_mean', 'E_max', 'E_adp', 'Fw', 'F_adp', 'MAE', 'IoU', 'n', 'fp_false_mask_rate', 'n_fp')   # the console line of evaluate


def seg_inputs(front_end, batch, device):
    '''
    Model input of one seg_collate_fn batch: x [B, N + 1, c, c] float32 = cat(front_end(img) * valid, box_map).
    The front end runs in float32 with autocast OFF, whatever the caller's autocast context: FilterBank.forward computes
    in its input's dtype, and on a real window bf16 gives errors of 0.34 (ec10) / 0.43 (ec24) in standardised units and
    fp16 already 0.42 on ec24's whitened channel 20 (recon), so the 133-band fp16 crop is cast to float32 first.
    Multiplying by valid zeroes the canvas padding: there the standardised channels would read -mean/std (the padding
    is 0 in band space), so after the mask the padding is 0 = the training mean in every arm, like the box channel.
    No gradient flows into the front end (it is the detector's fixed sensor model).
    '''
    img = batch['img'].to(device, non_blocking=True)                                     # [B, 133, c, c] fp16, p99-scaled
    valid = batch['valid'].to(device, non_blocking=True).float()                         # [B, 1, c, c] 1 = ROI content
    box_map = batch['box_map'].to(device, non_blocking=True).float()                     # [B, 1, c, c] 1 = pre-expansion box
    with torch.no_grad(), torch.autocast(device.type, enabled=False):
        z = front_end(img.float()).float()                                               # [B, N, c, c] float32
    assert z.shape[1] == front_end.n_out, f"front end returned {z.shape[1]} channels, its n_out is {front_end.n_out}"
    assert z.shape[-2:] == box_map.shape[-2:], f"front end output {tuple(z.shape)} does not match the canvas {tuple(box_map.shape)}"
    return torch.cat([z * valid, box_map], dim=1)                                        # [B, N + 1, c, c]


def train_one_epoch(model, front_end, data_loader, optimizer, device, epoch, scaler=None, accumulate=1, max_norm=1.0,
                    logger=None, print_freq=10, amp=True, amp_dtype=torch.bfloat16, total_epochs=None):
    '''
    One pass over data_loader (seg_collate_fn batches). Same structure as train_eval_det.train_one_epoch: MetricLogger,
    loss / accumulate, clipping at max_norm on the accumulated gradient, an optimizer step every `accumulate` batches and
    at the last batch, per-step scalars to logger under 'train/' against the global step.
    Mixed precision: the model forward runs under torch.autocast(amp_dtype) when amp is set and the device is CUDA
    (bf16 by default, spec §7). bf16 needs no GradScaler, so unlike the detector loop (autocast iff a scaler exists)
    autocast is driven by `amp`; pass a torch.amp.GradScaler only with amp_dtype=torch.float16. The loss is computed in
    float32 outside autocast on the float32-cast logits.
    total_epochs: when given and the model has set_progress, it receives the training progress
    t = (epoch + i / n_steps) / total_epochs in [0, 1) before every forward (ZoomNeXt's UAL coefficient).
    Returns {meter: global average} with loss, lr and every item of model.loss.
    '''
    model.train()
    front_end.eval()                                                                     # fixed sensor model, eval-mode noise path
    metric_logger = MetricLogger(delimiter="; ")
    metric_logger.add_meter('loss', SmoothedValue(window_size=10, fmt='{value:.4f}'))
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
    n_steps = len(data_loader)
    use_amp = bool(amp) and device.type == 'cuda'
    params = [p for p in model.parameters() if p.requires_grad]                          # clip only what is trained (frozen trunks have no grad)
    optimizer.zero_grad(set_to_none=True)
    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if total_epochs and hasattr(model, 'set_progress'):
            model.set_progress((epoch + i / n_steps) / total_epochs)
        x = seg_inputs(front_end, batch, device)                                         # [B, N + 1, c, c] float32
        mask = batch['mask'].to(device, non_blocking=True).float()                       # [B, 1, c, c] GT union inside the ROI
        box_xyxy = batch['box_xyxy'].to(device, non_blocking=True).float()               # [B, 4] canvas px (box prompt)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            outputs = model(x, box_xyxy)                                                 # list of [B, 1, c, c] logits
        total, items = model.loss([o.float() for o in outputs], mask)
        # a NaN would silently destroy the run (AdamW spreads it into every trained weight); stop with the frames instead
        assert torch.isfinite(total), \
            f"non-finite loss {float(total.detach())} at epoch {epoch} step {i}, frames {[m['frame'] for m in batch['meta']]}"
        loss = total / accumulate
        (scaler.scale(loss) if scaler is not None else loss).backward()
        if (i + 1) % accumulate == 0 or i + 1 == n_steps:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, max_norm)
            if scaler is not None:
                scaler.step(optimizer); scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        lr = optimizer.param_groups[0]['lr']
        metric_logger.update(loss=float(total.detach()), lr=lr, **{k: float(v) for k, v in items.items()})
        if logger is not None:
            # every AdamW group's lr under its name (param_groups(args) names them, e.g. stem / lora / decoder)
            lrs = {f"lr_{g.get('name', k)}": g['lr'] for k, g in enumerate(optimizer.param_groups)}
            logger.scalars({'loss': float(total.detach()), 'lr': lr, **lrs, **{k: float(v) for k, v in items.items()}},
                           epoch * n_steps + i, prefix='train/')
    return {k: m.global_avg for k, m in metric_logger.meters.items()}


def roi_panel(prob, gt):
    '''Side-by-side uint8 RGB picture [h, 2 w, 3] of a native-size ROI prediction (left) and its GT (right), for logging.'''
    p = (np.clip(prob, 0, 1) * 255).astype(np.uint8)                                     # [h, w]
    g = (np.asarray(gt, dtype=bool) * 255).astype(np.uint8)                              # [h, w]
    sep = np.full((p.shape[0], 2), 128, dtype=np.uint8)                                  # grey divider
    return np.repeat(np.concatenate([p, sep, g], axis=1)[..., None], 3, axis=2)


@torch.no_grad()
def evaluate(model, front_end, data_loader, device, logger=None, epoch=0, tag='val', amp=True, amp_dtype=torch.bfloat16,
             size_edges=(2000, 20000), n_images_log=4):
    '''
    ROI-level evaluation (spec §5.5): every item's main output (index 0) goes through a sigmoid, is cut back from the
    canvas and resized to the ROI's native size (canvas_to_roi), and is scored at that size against the GT mask brought
    back the same way (canvas_to_roi(mask) > 0.5; exact when the ROI fits the canvas, s = 1, which is the case for most
    val ROIs; interpolated for ROIs longer than the canvas). False-positive items (meta['source'] == 'fp', empty GT) go
    to SegMetrics.update_fp (false-mask rate), never into S / E / F. Returns SegMetrics.summary() (S, E_mean, E_max,
    E_adp, Fw, F_adp, MAE, IoU, n, the size-bucket copies, fp_false_mask_rate, n_fp); its scalars are logged under
    '<tag>/' and a few prediction | GT panels under '<tag>/roi_<i>'.
    '''
    model.eval()
    front_end.eval()
    metrics = SegMetrics(size_edges=size_edges)
    metric_logger = MetricLogger(delimiter="; ")
    use_amp = bool(amp) and device.type == 'cuda'
    logged = 0
    for batch in metric_logger.log_every(data_loader, 10, f'Eval {tag}:'):
        x = seg_inputs(front_end, batch, device)                                         # [B, N + 1, c, c]
        box_xyxy = batch['box_xyxy'].to(device, non_blocking=True).float()               # [B, 4]
        with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            outputs = model(x, box_xyxy)
        prob = torch.sigmoid(outputs[0].float())[:, 0].cpu().numpy()                     # [B, c, c] in [0, 1]
        mask = batch['mask'][:, 0].float().numpy()                                       # [B, c, c]
        for b, meta in enumerate(batch['meta']):
            p = canvas_to_roi(prob[b], meta['offset'], meta['s'], meta['roi_hw'])         # [h, w] float32, native ROI size
            if meta['source'] == 'fp':
                metrics.update_fp(p)
                continue
            g = canvas_to_roi(mask[b], meta['offset'], meta['s'], meta['roi_hw']) > 0.5  # [h, w] bool
            metrics.update(p, g)
            if logger is not None and logged < n_images_log:
                logger.image(f'{tag}/roi_{logged}', roi_panel(p, g), epoch); logged += 1
    summary = metrics.summary()
    if logger is not None:
        logger.scalars({k: v for k, v in summary.items() if isinstance(v, (int, float, np.integer, np.floating))}, epoch, prefix=f'{tag}/')
    print(f"{tag} epoch {epoch}: " + ", ".join(f"{k}={summary[k]:.4f}" if isinstance(summary[k], float) else f"{k}={summary[k]}"
                                                for k in PRINT_KEYS if k in summary))
    return summary


def paste_back(prob_canvas, meta, out):
    '''
    Paste one ROI prediction into a full-frame probability map, merging overlaps by maximum (spec §5.5).
      prob_canvas [c, c] probability on the canvas (np or tensor), meta the item's meta dict (offset, s, roi_hw, roi),
      out [H, W] float32 frame map (np.zeros((1680, 1240), np.float32) for a fresh frame), modified in place and returned.
    The ROI's crop starts at (floor(y1), floor(x1)) of meta['roi'] (frame px) and is roi_hw = (h, w) pixels, i.e. the
    floor / ceil rounding of the fractional export ROI (the rule of box_metrics.mask_coverage); a roi_hw inconsistent
    with that rounding means the crop came from another rule and raises instead of pasting at a shifted place.
    '''
    if torch.is_tensor(prob_canvas):
        prob_canvas = prob_canvas.detach().float().cpu().numpy()
    assert out.ndim == 2, f"out must be a [H, W] frame map, got shape {out.shape}"
    h, w = int(meta['roi_hw'][0]), int(meta['roi_hw'][1])
    x1, y1, x2, y2 = [float(v) for v in meta['roi']]
    x0, y0 = int(math.floor(x1)), int(math.floor(y1))
    # floor / ceil rounding makes the integer size exceed the float size by less than 2 px, never fall short of it
    assert -1e-3 <= w - (x2 - x1) < 2 and -1e-3 <= h - (y2 - y1) < 2, \
        f"frame {meta.get('frame')}: roi_hw {(h, w)} does not match roi {meta['roi']} (floor x1/y1, ceil x2/y2)"
    assert x0 >= 0 and y0 >= 0 and y0 + h <= out.shape[0] and x0 + w <= out.shape[1], \
        f"frame {meta.get('frame')}: roi {meta['roi']} with size {(h, w)} leaves the {out.shape} frame"
    p = canvas_to_roi(prob_canvas, meta['offset'], meta['s'], (h, w))                     # [h, w] float32
    np.maximum(out[y0:y0 + h, x0:x0 + w], p, out=out[y0:y0 + h, x0:x0 + w])
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python -m pytest tests/test_train_eval_seg.py -v`
Expected: PASS (6 passed, about 3 s). Then the full suite must stay green: `OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python -m pytest tests/ -q`.

- [ ] **Step 5: Commit and push**

```bash
git add train_eval/train_eval_seg.py tests/test_train_eval_seg.py
git commit -m "feat(seg): Stage-2 train/eval loops and full-frame paste-back

train_one_epoch (bf16 autocast without a scaler, accumulation, clipping, set_progress for ZoomNeXt), ROI-level
evaluate at native size with fp ROIs routed to the false-mask rate, and paste_back with max-merge and the floor/ceil ROI rule.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QB4DScDwiA2DkVpbDopffV"
git push origin worktree-det
git push origin worktree-det:master
```

---

### Task 12: main_seg.py and cfg/seg.yaml

**Files:**
- Create: `main_seg.py`
- Create: `cfg/seg.yaml`
- Test: `tests/test_main_seg.py`

**Interfaces:**
- Consumes:
  - T2 `main_det_rois.detector_args(ckpt_args, args)` (the one list of model-defining detector keys, `DET_MODEL_KEYS`).
  - T1 `models.filter_bank.build_filter_bank(args, dataset) -> (fb, volts)` (tests only).
  - T3 `index.json`: `band_range`, `roi_files` (`{arm: {split: path}}`), `frame_hw`, `grow`, `min_side`, `arms`, windows with `object` and objects with `area`.
  - T4 `HyperCOD_roi(cache_dir, split, arm, box_mix, gt_jitter, roi_margin, roi_min, canvas, scale_aug, gain_aug, train, seed)` and `seg_collate_fn`. `train=False` also works for split `'train'` (used for the deterministic stem fit).
  - T6 `pseudo_rgb(x, wavelens)`, `fit_rgb_map(z, rgb_norm) -> (P, q)`, `IMAGENET_MEAN`, `IMAGENET_STD`.
  - T8-T10 `build_front_end(arm, det_args, dataset, det_state=None)` (with `.n_out`; its `check_filter_bank` raises with "detector checkpoint" in the message) and `build_seg_model(args, n_in, P, q)` with the model API of Task 11.
  - Task 11 `train_one_epoch`, `evaluate`, `seg_inputs`.
  - `main_det.get_args_parser()` and `main_det.dataset_kwargs(args)`.
- Produces (T13 `main_seg_eval.py` and T14 `launch_seg_queue.sh` use these):
  - CLI (snake_case):
    - `--seg_model sam2unet|sam2box|zoomnext`, `--arm raw|ec10|ec24|rgb`, `--det_ckpt`
    - `--name` (default `seg_<model>_<arm>_s<seed>`), `--output_dir` (default `weights/<name>`), `--seed`, `--device`, `--resume`, `--eval`, `--dry_run`, `--hpy cfg/seg.yaml`
    - `--data_path`, `--cache_dir`, `--crop_cache` (default `<data_path>/crop_cache_seg`), `--num_workers`
    - `--box_mix` (3 floats), `--gt_jitter`, `--canvas`, `--epochs`, `--batch_size`, `--accumulate`, `--lr`, `--stem_lr`, `--weight_decay`, `--max_norm`
    - `--sam2_ckpt`, `--sam2unet_ckpt`, `--zoomnext_ckpt`
    - `--runs_dir`, `--wandb/--no-wandb`, `--wandb_entity`, `--wandb_project`, `--wandb_dir`
  - Results: `<output_dir>/results_<name>.txt` holds one JSON line per epoch, `{"epoch", "train", "val", "score", "best", "best_epoch"}`. The last line is `{"epoch", "final": true, "best_epoch", "best", "val", "val_last"}`, and `--eval` appends `{"eval": true, "resume", "epoch", "val"}`.
  - Checkpoints `model_best` and `model_last`: `{"model" (full state_dict), "optimizer", "scaler", "lr_scheduler", "epoch", "args" (vars), "best", "best_epoch", "P" [3, N] f32, "q" [3] f32, "n_in"}`.
  - `load_cfg(args) -> dict`
  - `det_args_from_ckpt(det_ckpt, args) -> (det_args, ckpt)`. `det_args = main_det_rois.detector_args(ckpt['args'], data_path / cache_dir / device of this run)`; `det_args.name` is the detector run, for example `sel10g_clean_A`.
  - `check_arm(arm, det_args)`
  - `build_arm_front_end(args, device) -> (front_end, det_args, dataset)` (front end checked against the detector checkpoint's filter bank through `build_front_end(..., det_state=...)`)
  - `fit_arm_rgb_map(front_end, dataset, wavelens, device, n_pixels, n_items, seed, num_workers) -> (P, q, r2)` (the only stem-fit sampler)
  - `val_score(summary) -> float`, `save_checkpoint(...)`
  - `load_seg_run(ckpt_path, device, **overrides) -> (model, front_end, args, det_args, ckpt)`
  - `dry_run(model, front_end, dataset, args, device, amp_dtype) -> {"n_in", "sources", "peak_mem_gib"}`; a box source with a single item is run as a batch of 2 (ZoomNeXt's pooled BatchNorm).
  - Spec §9 checks in `main`: the crop cache's `band_range` equals the detector's, and the cache's ROI files of the arm (`raw` for `rgb`) are `rois_<det_args.name>_<split>.json`; the train loader uses `drop_last=True` and needs at least one full batch.

- [ ] **Step 1: Write the failing test**

`tests/test_main_seg.py`:
```python
import json
import os
import numpy as np
import pytest
pytest.importorskip('py_sod_metrics')        # main_seg -> train_eval_seg -> seg_metrics needs it
import torch
import yaml

import main_det
import main_seg
from data_loader.my_dataset import HyperCOD_data
from models.filter_bank import build_filter_bank
from tests.test_train_eval_seg import TinySeg

CANVAS = 32


def _det_ckpt(root, path, *argv):
    '''A Stage-1 checkpoint stub: the detector's flags and its filter_bank.* tensors (all main_seg reads from it).'''
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    det_args = main_det.get_args_parser().parse_args(['--data-path', str(root), *argv])
    dataset = HyperCOD_data(split='train', **main_det.dataset_kwargs(det_args))
    fb, _ = build_filter_bank(det_args, dataset)
    torch.save({'args': vars(det_args), 'model': {f'filter_bank.{k}': v for k, v in fb.state_dict().items()}, 'epoch': 0}, path)
    return path


def _crop_cache(cache):
    '''
    A tiny crop cache in build_crop_cache's format (Task 3's index keys; ROI provenance = raw133_A): train frame '3' with
    9 object windows (32 x 32, one object and one matched raw-arm ROI each) and 2 false-positive windows; val frame '10'
    with 2 object windows.
    '''
    rng = np.random.default_rng(0)
    os.makedirs(cache, exist_ok=True)
    windows = []

    def add(frame, split, kind, k, window, objects, rois):
        x1, y1, x2, y2 = window
        img = (rng.random((133, y2 - y1, x2 - x1)) * 0.02).astype(np.float16)            # [133, h, w] un-scaled
        gt = np.zeros((y2 - y1, x2 - x1), dtype=bool)                                    # [h, w]
        for o in objects:
            bx1, by1, bx2, by2 = o['box']
            gt[by1 - y1:by2 - y1, bx1 - x1:bx2 - x1] = True
        np.save(os.path.join(cache, f'{frame}_{k}.npy'), img)
        np.save(os.path.join(cache, f'{frame}_{k}_gt.npy'), gt)
        windows.append({'file': f'{frame}_{k}.npy', 'gt_file': f'{frame}_{k}_gt.npy', 'frame': frame, 'split': split, 'kind': kind,
                        'window': list(window), 'scale': 0.011, 'object': objects[0]['id'] if kind == 'object' else -1,
                        'objects': objects, 'rois': {'raw': rois, 'ec10': [], 'ec24': []}})

    det = lambda oid: [{'roi': [10.0, 13.0, 26.0, 33.0], 'box': [13.0, 17.0, 23.0, 29.0], 'conf': 0.7, 'object': oid}]
    for k in range(9):
        add('3', 'train', 'object', k, (4, 8, 36, 40), [{'id': k + 1, 'box': [14, 18, 22, 28], 'area': 80}], det(k + 1))
    for k in range(9, 11):
        add('3', 'train', 'fp', k, (0, 0, 16, 16), [], [{'roi': [0.0, 0.0, 16.0, 16.0], 'box': [4.0, 4.0, 12.0, 12.0], 'conf': 0.3, 'object': -1}])
    for k in range(2):
        add('10', 'val', 'object', k, (4, 8, 36, 40), [{'id': k + 1, 'box': [14, 18, 22, 28], 'area': 80}], det(k + 1))
    with open(os.path.join(cache, 'index.json'), 'w') as f:
        json.dump({'band_range': [400.0, 800.0], 'frame_hw': [48, 40], 'grow': 2.0, 'min_side': 32, 'min_area': 10,
                   'min_cover': 0.01, 'arms': ['raw', 'ec10', 'ec24'],
                   'roi_files': {a: {s: f'results/det/rois_raw133_A_{s}.json' for s in ('train', 'val')} for a in ('raw', 'ec10', 'ec24')},
                   'windows': windows}, f)


def _hpy(path):
    '''cfg/seg.yaml shrunk to the fixture: 32 px canvas and ROIs, a small stem fit, batch 4, model_last every epoch.'''
    with open('cfg/seg.yaml') as f:
        cfg = yaml.safe_load(f)
    cfg.update(canvas=CANVAS, roi_min=8, fit_items=4, fit_pixels=2000, save_every=1, print_freq=1)
    cfg['models'] = {k: {**v, 'batch_size': 4, 'accumulate': 1} for k, v in cfg['models'].items()}
    with open(path, 'w') as f:
        yaml.safe_dump(cfg, f)


def _args(root, tmp_path, det_ckpt, *extra):
    argv = ['--seg_model', 'sam2unet', '--arm', 'raw', '--det_ckpt', str(det_ckpt), '--data_path', str(root),
            '--crop_cache', str(tmp_path / 'crops'), '--hpy', str(tmp_path / 'seg.yaml'), '--device', 'cpu', '--num_workers', '0',
            '--no-wandb', '--runs_dir', str(tmp_path / 'runs'), '--output_dir', str(tmp_path / 'seg'), '--name', 'tseg', *extra]
    return main_seg.get_args_parser().parse_args(argv)


def _setup(root, tmp_path, monkeypatch):
    '''Raw-arm detector stub (run raw133_A), crop cache, shrunk cfg, and build_seg_model replaced by TinySeg (records n_in, P, q).'''
    det = _det_ckpt(root, tmp_path / 'raw133_A' / 'model_best', '--raw-bands', '--filter-select', 'uniform', '--num-filters', '6')
    _crop_cache(tmp_path / 'crops')
    _hpy(tmp_path / 'seg.yaml')
    built = []
    monkeypatch.setattr(main_seg, 'build_seg_model', lambda args, n_in, P, q: built.append((n_in, np.asarray(P), np.asarray(q))) or TinySeg(n_in))
    return det, built


def test_load_cfg_fills_unset_flags_and_the_model_section_wins():
    args = main_seg.get_args_parser().parse_args(['--seg_model', 'zoomnext', '--lr', '5e-4'])
    main_seg.load_cfg(args)
    assert args.lr == 5e-4                                                         # an explicit flag wins
    assert (args.batch_size, args.accumulate, args.encoder_lr_mult) == (4, 2, 1.0)  # the zoomnext section
    assert args.epochs == 200 and args.canvas == 512 and list(args.box_mix) == [0.5, 0.4, 0.1] and args.gt_jitter == 0.15
    assert isinstance(args.weight_decay, float) and args.weight_decay == 1e-4 and args.max_norm == 1.0 and args.amp_dtype == 'bfloat16'
    sam = main_seg.get_args_parser().parse_args(['--seg_model', 'sam2unet'])
    main_seg.load_cfg(sam)
    assert (sam.lr, sam.stem_lr, sam.batch_size, sam.accumulate) == (1e-3, 1e-4, 8, 1)


def test_det_args_come_from_the_checkpoint_and_must_match_the_arm(synthetic_root, tmp_path):
    root, _, _ = synthetic_root
    ec = _det_ckpt(root, tmp_path / 'sel_det' / 'model_best', '--filter-select', 'uniform', '--num-filters', '12', '--pca-channels', '10')
    ck = torch.load(ec, map_location='cpu', weights_only=False)
    ck['args']['data_path'] = '/nonexistent'                                       # the checkpoint never moves the data
    for k in ('read_noise_db', 'no_gate'):                                          # an older checkpoint without these flags
        ck['args'].pop(k)
    torch.save(ck, ec)
    args = main_seg.get_args_parser().parse_args(['--data_path', str(root), '--device', 'cpu'])
    det_args, _ = main_seg.det_args_from_ckpt(str(ec), args)
    assert det_args.data_path == str(root) and det_args.device == 'cpu' and det_args.name == 'sel_det'
    assert det_args.pca_channels == 10 and det_args.num_filters == 12 and det_args.read_noise_db == 0.0 and det_args.no_gate is False
    main_seg.check_arm('ec10', det_args)
    with pytest.raises(AssertionError, match='--pca-channels 24'):
        main_seg.check_arm('ec24', det_args)
    with pytest.raises(AssertionError, match='raw-bands'):
        main_seg.check_arm('raw', det_args)
    with pytest.raises(AssertionError, match='not found'):
        main_seg.det_args_from_ckpt(str(tmp_path / 'missing'), args)


def test_main_seg_rejects_a_front_end_that_differs_from_the_detector(synthetic_root, tmp_path, monkeypatch):
    root, _, _ = synthetic_root
    det, _ = _setup(root, tmp_path, monkeypatch)
    ck = torch.load(det, map_location='cpu', weights_only=False)
    ck['model']['filter_bank.mean'] = ck['model']['filter_bank.mean'] + 1e-3        # not what the flags rebuild
    bad = tmp_path / 'bad' / 'raw133_A' / 'model_best'                            # same run name, so only the tensors differ
    os.makedirs(bad.parent)
    torch.save(ck, bad)
    with pytest.raises(AssertionError, match='detector checkpoint'):
        main_seg.main(_args(root, tmp_path, bad, '--epochs', '1'))


def test_dry_run_checks_one_batch_per_box_source_and_writes_nothing(synthetic_root, tmp_path, monkeypatch):
    root, _, _ = synthetic_root
    det, built = _setup(root, tmp_path, monkeypatch)
    info = main_seg.main(_args(root, tmp_path, det, '--dry_run'))
    assert info['n_in'] == 133 and info['sources'] and set(info['sources']) <= {'gt', 'det', 'fp'}
    assert all(shape[1:] == (134, CANVAS, CANVAS) for shape in info['sources'].values())
    assert built[0][0] == 133 and built[0][1].shape == (3, 133) and built[0][2].shape == (3,)
    assert not (tmp_path / 'seg').exists()


def test_main_seg_trains_resumes_and_evaluates(synthetic_root, tmp_path, monkeypatch):
    root, _, _ = synthetic_root
    det, built = _setup(root, tmp_path, monkeypatch)
    out = tmp_path / 'seg'
    main_seg.main(_args(root, tmp_path, det, '--epochs', '2'))
    lines = [json.loads(l) for l in (out / 'results_tseg.txt').read_text().strip().splitlines()]
    assert [l['epoch'] for l in lines[:2]] == [0, 1] and lines[-1]['final'] is True and len(lines) == 3
    assert {'S', 'Fw', 'IoU', 'MAE', 'n'} <= set(lines[0]['val']) and lines[0]['val']['n'] == 4   # 2 oracle + 2 matched val ROIs
    assert {'loss', 'lr', 'main', 'side'} <= set(lines[0]['train'])
    ck = torch.load(out / 'model_last', map_location='cpu', weights_only=False)
    assert ck['epoch'] == 1 and ck['P'].shape == (3, 133) and ck['q'].shape == (3,) and ck['n_in'] == 133
    assert ck['args']['arm'] == 'raw' and ck['args']['seg_model'] == 'sam2unet' and (out / 'model_best').exists()
    np.testing.assert_array_equal(ck['P'], built[0][1])

    # resume: one more epoch from model_last with the stored stem fold (no refit) and the best score carried over
    def no_refit(*a, **k):
        raise AssertionError('a resumed run must reuse the stored (P, q)')
    monkeypatch.setattr(main_seg, 'fit_arm_rgb_map', no_refit)
    main_seg.main(_args(root, tmp_path, det, '--epochs', '3', '--resume', str(out / 'model_last')))
    lines = [json.loads(l) for l in (out / 'results_tseg.txt').read_text().strip().splitlines()]
    assert lines[-2]['epoch'] == 2 and lines[-2]['best'] >= lines[1]['best'] and lines[-1]['final'] is True
    np.testing.assert_array_equal(built[1][1], built[0][1])
    with pytest.raises(AssertionError, match='seg_model'):                         # another model's checkpoint
        main_seg.main(_args(root, tmp_path, det, '--seg_model', 'zoomnext', '--resume', str(out / 'model_last')))

    main_seg.main(_args(root, tmp_path, det, '--resume', str(out / 'model_best'), '--eval'))
    last = json.loads((out / 'results_tseg.txt').read_text().strip().splitlines()[-1])
    assert last['eval'] is True and 'S' in last['val'] and last['val']['n'] == 4

    # main_seg_eval's entry point: the run rebuilt from its checkpoint alone
    model, fe, a, det_args, ck = main_seg.load_seg_run(str(out / 'model_best'), torch.device('cpu'))
    assert not model.training and fe.n_out == 133 and a.arm == 'raw' and det_args.raw_bands
    torch.testing.assert_close(model.head.weight, ck['model']['head.weight'])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python -m pytest tests/test_main_seg.py -v`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'main_seg'`.

- [ ] **Step 3: Write minimal implementation**

`cfg/seg.yaml`:
```yaml
# Stage-2 segmentation hyper-parameters (spec §7). main_seg.load_cfg fills every flag left at None and every key that has no
# flag: first the common keys below, then the section of --seg_model under `models` overrides them; an explicit CLI flag
# always wins. Floats are written with a dot (1.0e-4): PyYAML reads a bare 1e-4 as a string.
epochs: 200                   # ~7 k optimizer steps at batch 8 over ~290 objects + fp items per epoch
batch_size: 8
accumulate: 1                 # gradient accumulation steps
lr: 1.0e-4                    # model lr (per-model value below)
stem_lr: 1.0e-4               # the folded N+1-channel stem (its own AdamW group, param_groups(args))
lrf: 0.01                     # final lr = lr * lrf, cosine per epoch (as main_det)
weight_decay: 1.0e-4          # AdamW (SAM2-UNet's own recipe uses 5e-4; spec §7 deliberately uses 1e-4 for all models)
max_norm: 1.0                 # gradient clipping on the accumulated gradient
amp_dtype: bfloat16           # autocast dtype of the model forward on CUDA (bf16: no GradScaler); the front end always runs in fp32
canvas: 512                   # ROI canvas side (px); multiple of 32 (Hiera windows), of 64 for ZoomNeXt (its 0.5x scale)
box_mix: [0.5, 0.4, 0.1]      # training items: expanded GT boxes / matched detector boxes / false-positive detector boxes
gt_jitter: 0.15               # each side of an expanded GT box moves OUTWARD by U(0, gt_jitter) x box side
roi_margin: 1.5               # the Stage-1 export rule: ROI = box grown 1.5x about its centre ...
roi_min: 256                  # ... and at least 256 px per side (cfg/det.yaml roi_margin / roi_min)
scale_aug: [0.75, 1.25]       # training: ROI content rescaled by U(lo, hi) before placement on the canvas
gain_aug: 0.05                # training: per-band gain U(1 - gain_aug, 1 + gain_aug)
fit_pixels: 2000000           # stem fold (P, q): at most this many training pixels ...
fit_items: 64                 # ... drawn evenly from this many deterministic training items (each read once)
size_edges: [2000, 20000]     # object-size buckets by GT area (px): small < 2000 <= medium <= 20000 < large
save_every: 10                # model_last every N epochs and at the end (each full state_dict is ~0.9 GB for the Hiera-L models)
print_freq: 10
models:
  sam2unet:                   # adapters + RFB + U-Net decoder at 1e-3 (SAM2-UNet's recipe), Hiera-L trunk frozen
    lr: 1.0e-3
    batch_size: 8
    accumulate: 1
  sam2box:                    # LoRA (r = 8) + stem + mask decoder at 1e-4; loss = Dice + BCE
    lr: 1.0e-4
    batch_size: 8
    accumulate: 1
    lora_r: 8
  zoomnext:                   # full fine-tune; batch 4 x accumulation 2 (the three-scale encoder pass dominates memory)
    lr: 1.0e-4
    batch_size: 4
    accumulate: 2
    encoder_lr_mult: 1.0      # PVT encoder lr = lr x this (spec §7: 1.0; ZoomNeXt's own recipe uses 0.1)
```

`main_seg.py`:
```python
"""
Stage 2: camouflaged-object segmentation inside the Stage-1 ROIs (spec docs/superpowers/specs/2026-10-04-stage2-segmentation-design.md).
Trains one (model, arm, seed) on the crop cache (data_loader.roi_crops.build_crop_cache) and selects model_best on the
deterministic val ROIs; the test frames are only touched by main_seg_eval.py.
  python main_seg.py --seg_model sam2unet --arm ec10 --det_ckpt weights/sel10g_clean_A/model_best --seed 0
  python main_seg.py --seg_model sam2box --arm raw --det_ckpt weights/raw133_A/model_best --seed 1
  python main_seg.py --seg_model zoomnext --arm ec24 --det_ckpt weights/sel24g_clean_A/model_best --seed 2
Control: SAM2-UNet on the 3-channel pseudo-RGB render, with the raw arm's ROIs:
  python main_seg.py --seg_model sam2unet --arm rgb --det_ckpt weights/raw133_A/model_best --seed 0
Pre-flight (one batch per box source, shapes, peak GPU memory; nothing is written):
  python main_seg.py --seg_model zoomnext --arm raw --det_ckpt weights/raw133_A/model_best --dry_run
Continue a run, or evaluate a checkpoint on the val ROIs:
  python main_seg.py --seg_model sam2unet --arm ec10 --det_ckpt weights/sel10g_clean_A/model_best --resume weights/seg_sam2unet_ec10_s0/model_last
  python main_seg.py --seg_model sam2unet --arm ec10 --det_ckpt weights/sel10g_clean_A/model_best --resume weights/seg_sam2unet_ec10_s0/model_best --eval
Flags left at None come from --hpy cfg/seg.yaml (common keys, then the --seg_model section); explicit flags win.
Spec §9 checks: the detector must be the arm's (check_arm), its filter bank is rebuilt bit-identically (build_front_end with
the checkpoint's state), and the crop cache must hold that detector's ROIs and band window.
Without torchrun this script pins CUDA_VISIBLE_DEVICES=0 unless it is already set (as main_det does); one job per GPU.
"""
import os
if "RANK" not in os.environ and "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import datetime
import json
import math
import random
import time

import yaml
import numpy as np
import torch
import torch.multiprocessing

import util.misc as utils
import main_det
import main_det_rois
from util.logger import TrainLogger
from data_loader.my_dataset import HyperCOD_data
from data_loader.roi_crops import HyperCOD_roi, seg_collate_fn
from models.seg_models import build_front_end, build_seg_model
from models.seg_stem import pseudo_rgb, fit_rgb_map, IMAGENET_MEAN, IMAGENET_STD
from train_eval.train_eval_seg import train_one_epoch, evaluate, seg_inputs

torch.multiprocessing.set_sharing_strategy('file_system')

ARMS = ('raw', 'ec10', 'ec24', 'rgb')
SEG_MODELS = ('sam2unet', 'sam2box', 'zoomnext')
AMP_DTYPES = {'bfloat16': torch.bfloat16, 'float16': torch.float16}


def get_args_parser():
    parser = argparse.ArgumentParser('Stage-2 segmentation inside the Stage-1 ROIs', add_help=False)
    # run
    parser.add_argument('--seg_model', type=str, default='sam2unet', choices=SEG_MODELS,
                        help='sam2unet: SAM2-UNet (Hiera-L trunk frozen, adapters + decoder); sam2box: SAM2.1 + box prompt + LoRA; '
                             'zoomnext: ZoomNeXt-B2 (full fine-tune)')
    parser.add_argument('--arm', type=str, default='raw', choices=ARMS,
                        help='input arm: raw (133 standardised bands), ec10 / ec24 (the detector\'s whitened EC readings), '
                             'rgb (control: ImageNet-normalised pseudo-RGB render, with the raw arm\'s ROIs)')
    parser.add_argument('--det_ckpt', type=str, default='',
                        help="the arm's Stage-1 detector checkpoint (raw / rgb: weights/raw133_A/model_best, ec10: "
                             "weights/sel10g_clean_A/model_best, ec24: weights/sel24g_clean_A/model_best); the front end is rebuilt "
                             "from its flags and checked against its filter_bank tensors")
    parser.add_argument('--name', type=str, default='', help='run name, default seg_<seg_model>_<arm>_s<seed>; results_<name>.txt, runs/<name>, W&B')
    parser.add_argument('--output_dir', type=str, default='', help='model_best / model_last / results_<name>.txt, default weights/<name>')
    parser.add_argument('--seed', type=int, default=0, help='torch / numpy / random seed (runs use 0, 1, 2)')
    parser.add_argument('--device', type=str, default='cuda', help="'cuda', 'cuda:N' or 'cpu'")
    parser.add_argument('--resume', type=str, default='', help='a main_seg checkpoint: continue training from it, or evaluate it with --eval')
    parser.add_argument('--eval', action='store_true', help='only evaluate --resume on the val ROIs')
    parser.add_argument('--dry_run', action='store_true',
                        help='pre-flight: build everything, run one training batch per box source forward + backward, print the '
                             'shapes and the peak GPU memory, write nothing')
    parser.add_argument('--hpy', type=str, default='cfg/seg.yaml', help='hyper-parameter file (flags left at None are filled from it)')
    # data
    parser.add_argument('--data_path', type=str, default='/data2/chaoyi/HyperCOD/Raw data',
                        help='HyperCOD root (band statistics, EC filter file, intensity scales)')
    parser.add_argument('--cache_dir', type=str, default='', help='fp16 frame cache, default <data_path>/cache_fp16 (passed to the detector flags)')
    parser.add_argument('--crop_cache', type=str, default='', help='crop cache of build_crop_cache (index.json + windows), default <data_path>/crop_cache_seg')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    # ROI items (None -> cfg)
    parser.add_argument('--box_mix', type=float, nargs=3, default=None,
                        help='training box sources: expanded GT / matched detector ROI / false-positive ROI fractions (cfg: 0.5 0.4 0.1)')
    parser.add_argument('--gt_jitter', type=float, default=None, help='outward jitter of the expanded GT boxes, fraction of the box side (cfg: 0.15)')
    parser.add_argument('--canvas', type=int, default=None, help='canvas side in px (cfg: 512)')
    # optimisation (None -> cfg, per-model section first)
    parser.add_argument('--epochs', type=int, default=None, help='training epochs (cfg: 200)')
    parser.add_argument('--batch_size', type=int, default=None, help='batch size (cfg: 8; zoomnext 4)')
    parser.add_argument('--accumulate', type=int, default=None, help='gradient accumulation steps (cfg: 1; zoomnext 2)')
    parser.add_argument('--lr', type=float, default=None, help='model lr (cfg: sam2unet 1e-3, sam2box / zoomnext 1e-4)')
    parser.add_argument('--stem_lr', type=float, default=None, help='lr of the folded N+1-channel stem (cfg: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=None, help='AdamW weight decay (cfg: 1e-4)')
    parser.add_argument('--max_norm', type=float, default=None, help='gradient clipping norm (cfg: 1.0)')
    # pretrained weights (bash_files/setup_third_party.sh)
    parser.add_argument('--sam2_ckpt', type=str, default='weights/pretrained/sam2.1_hiera_large.pt', help='SAM2.1 Hiera-L checkpoint (sam2box)')
    parser.add_argument('--sam2unet_ckpt', type=str, default='weights/pretrained/sam2_hiera_large.pt',
                        help='SAM2 v1 Hiera-L checkpoint the SAM2-UNet trunk starts from (the SAM2-UNet README asks for v1, not 2.1)')
    parser.add_argument('--zoomnext_ckpt', type=str, default='weights/pretrained/pvtv2-b2-zoomnext.pth', help='ZoomNeXt PVTv2-B2 COD checkpoint')
    # logging
    parser.add_argument('--runs_dir', type=str, default='runs', help='TensorBoard root')
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=True, help='log to W&B (falls back to offline)')
    parser.add_argument('--wandb_entity', type=str, default='chaoyi-hsi', help='W&B entity')
    parser.add_argument('--wandb_project', type=str, default='hsi_camo', help='W&B project')
    parser.add_argument('--wandb_dir', type=str, default='wandb', help='W&B local directory')
    return parser


def load_cfg(args):
    '''
    Fill args from --hpy: the common keys, overridden by the section of args.seg_model under `models`. A key is applied when
    the flag is unset (None) or has no flag at all (amp_dtype, roi_min, fit_pixels, ... and model knobs such as lora_r or
    encoder_lr_mult that build_seg_model reads from args). Returns the merged flat dict (logged with the run config).
    '''
    with open(args.hpy) as f:
        cfg = yaml.safe_load(f)
    sections = cfg.get('models', {}) or {}
    assert args.seg_model in sections, f"{args.hpy} has no models.{args.seg_model} section (has {sorted(sections)})"
    merged = {**{k: v for k, v in cfg.items() if k != 'models'}, **sections[args.seg_model]}
    for k, v in merged.items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)
    return merged


def det_args_from_ckpt(det_ckpt, args):
    '''
    The detector's flags for the arm's front end, through main_det_rois.detector_args (the one list of model-defining
    detector keys, DET_MODEL_KEYS): main_det's parser defaults (raw133_A predates --pca-channels, read noise and --no-gate,
    so missing keys fall back to the Stage-1 defaults), this run's data_path / cache_dir / device, then the model keys of
    ckpt['args']. The checkpoint never overrides the data location (a cache moved to another disk, the test fixture) or the
    device. Returns (det_args, ckpt); det_args.name = the detector run (the checkpoint's directory, e.g. sel10g_clean_A).
    '''
    assert det_ckpt and os.path.isfile(det_ckpt), f"--det_ckpt {det_ckpt!r} not found (the arm's Stage-1 detector checkpoint)"
    ckpt = torch.load(det_ckpt, map_location='cpu', weights_only=False)
    assert 'args' in ckpt and 'model' in ckpt, f"{det_ckpt} is not a main_det checkpoint (keys {sorted(ckpt)})"
    det_args = main_det_rois.detector_args(ckpt['args'], argparse.Namespace(data_path=args.data_path, cache_dir=args.cache_dir,
                                                                            device=args.device))
    det_args.name = os.path.basename(os.path.dirname(os.path.abspath(det_ckpt)))
    return det_args, ckpt


def check_arm(arm, det_args):
    '''Spec §9: the detector must be the arm's own (raw / rgb: --raw-bands; ec10 / ec24: --pca-channels 10 / 24 EC readings).'''
    raw = bool(getattr(det_args, 'raw_bands', False))
    k = int(getattr(det_args, 'pca_channels', 0) or 0)
    if arm in ('raw', 'rgb'):
        assert raw, f"--arm {arm} needs the --raw-bands detector (raw133_A), got {det_args.name}: raw_bands={raw}, pca_channels={k}"
    else:
        want = {'ec10': 10, 'ec24': 24}[arm]
        assert not raw and k == want, \
            f"--arm {arm} needs an EC detector with --pca-channels {want}, got {det_args.name}: raw_bands={raw}, pca_channels={k}"


def build_arm_front_end(args, device):
    '''
    (front_end, det_args, dataset) of args.arm: the detector's flags from --det_ckpt (det_args_from_ckpt, check_arm), a
    HyperCOD_data built exactly like the detector's (main_det.dataset_kwargs: band statistics, filter matrices, wavelengths)
    and models.seg_models.build_front_end, which checks the rebuilt filter bank against the checkpoint's filter_bank.*
    tensors (spec §3 "Front end = Stage 1's", §9; the rgb control has no filter bank); front_end in eval mode on device.
    '''
    det_args, det_ckpt = det_args_from_ckpt(args.det_ckpt, args)
    check_arm(args.arm, det_args)
    dataset = HyperCOD_data(split='train', **main_det.dataset_kwargs(det_args))
    front_end = build_front_end(args.arm, det_args, dataset, det_state=det_ckpt['model'] if args.arm != 'rgb' else None)
    print(f"arm {args.arm}: front end of {det_args.name} -> {front_end.n_out} channels"
          + (" (filter bank checked against the detector)" if args.arm != 'rgb' else ''))
    return front_end.to(device).eval(), det_args, dataset


@torch.no_grad()
def fit_arm_rgb_map(front_end, dataset, wavelens, device, n_pixels=2000000, n_items=64, seed=0, num_workers=0):
    '''
    (P [3, N], q [3], r2 [3]) of the arm's stem fold (spec §3, §5.3): least squares rgb_norm ~ P z + q on at most n_pixels
    canvas pixels of at most n_items training items. dataset should be deterministic (HyperCOD_roi(split 'train',
    train=False): oracle and matched ROIs, no augmentation); only valid (non-padding) pixels are used, the same number from
    every item. z = the arm's channels from the fp32 front end (the seg_inputs rule), rgb_norm = the ImageNet-normalised
    pseudo-RGB render of the same p99-scaled crop. r2 = the fit's R^2 per R, G, B (1 for the rgb arm), printed.
    '''
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(dataset), size=min(int(n_items), len(dataset)), replace=False))
    per_item = max(1, int(n_pixels) // len(idx))
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, idx.tolist()), batch_size=1, shuffle=False,
                                         num_workers=num_workers, collate_fn=seg_collate_fn)
    mean = torch.as_tensor(IMAGENET_MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.as_tensor(IMAGENET_STD, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    front_end.eval()
    zs, rgbs = [], []
    for batch in loader:
        img = batch['img'].to(device).float()                                            # [1, 133, c, c] p99-scaled
        valid = batch['valid'][0, 0].to(device) > 0.5                                    # [c, c]
        with torch.autocast(device.type, enabled=False):
            z = front_end(img).float()[0]                                                # [N, c, c]
        rgb = ((pseudo_rgb(img, wavelens).float() - mean) / std)[0]                      # [3, c, c]
        z, rgb = z[:, valid], rgb[:, valid]                                              # [N, n_valid], [3, n_valid]
        pick = torch.from_numpy(rng.choice(z.shape[1], size=min(per_item, z.shape[1]), replace=False)).to(device)
        zs.append(z[:, pick].T.double().cpu().numpy()); rgbs.append(rgb[:, pick].T.double().cpu().numpy())
    z, rgb = np.concatenate(zs), np.concatenate(rgbs)                                    # [n, N], [n, 3]
    P, q = fit_rgb_map(z, rgb)                                                           # [3, N], [3]
    resid = rgb - (z @ np.asarray(P, np.float64).T + np.asarray(q, np.float64))
    r2 = 1.0 - resid.var(axis=0) / np.maximum(rgb.var(axis=0), 1e-12)
    print(f"stem fold: {len(z)} pixels of {len(idx)} training items, {z.shape[1]} channels -> pseudo-RGB, R^2 (R, G, B) {np.round(r2, 4).tolist()}")
    return np.asarray(P, np.float32), np.asarray(q, np.float32), r2


def val_score(summary):
    '''Checkpoint selection (spec §7): mean of S-measure and weighted F over the deterministic val ROIs; nan -> 0.'''
    s = 0.5 * (summary['S'] + summary['Fw'])
    return float(s) if np.isfinite(s) else 0.0


def save_checkpoint(path, model, optimizer, scaler, scheduler, epoch, args, best, best_epoch, P, q):
    '''Full model state (frozen trunk included, so a checkpoint needs no pretrained file to load) + optimiser state + the stem fold.'''
    utils.save_on_master({'model': model.state_dict(), 'optimizer': optimizer.state_dict() if optimizer else None,
                          'scaler': scaler.state_dict() if scaler else None, 'lr_scheduler': scheduler.state_dict() if scheduler else None,
                          'epoch': epoch, 'args': vars(args), 'best': float(best), 'best_epoch': int(best_epoch),
                          'P': np.asarray(P, np.float32), 'q': np.asarray(q, np.float32), 'n_in': int(np.shape(P)[1])}, path)


def load_seg_run(ckpt_path, device, **overrides):
    '''
    Rebuild a trained run for evaluation (main_seg_eval): its flags from ckpt['args'] updated with overrides (e.g.
    data_path / cache_dir / det_ckpt / device on another machine), the arm's verified front end, the model built from the
    stored (P, q) and loaded with the checkpoint's weights. Returns (model, front_end, args, det_args, ckpt), both modules in
    eval mode on device.
    '''
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    args = argparse.Namespace(**{**ckpt['args'], **overrides})
    front_end, det_args, _ = build_arm_front_end(args, device)
    assert front_end.n_out == ckpt['n_in'], f"{ckpt_path}: trained on {ckpt['n_in']} arm channels, the front end gives {front_end.n_out}"
    model = build_seg_model(args, front_end.n_out, ckpt['P'], ckpt['q'])
    model.load_state_dict(ckpt['model'])
    return model.to(device).eval(), front_end, args, det_args, ckpt


def dry_run(model, front_end, dataset, args, device, amp_dtype):
    '''
    Pre-flight (spec §9): one training batch per box source (expanded GT 'gt', matched detector ROI 'det', false-positive ROI
    'fp'; at most 20 x batch_size random items are read to find them), each through seg_inputs, forward and backward exactly
    as in training, with the input / output shapes checked. A source with a single item is run as a batch of 2 (ZoomNeXt's
    pooled BatchNorm refuses one item in training). No optimizer step, nothing written.
    Returns {'n_in', 'sources': {source: x shape}, 'peak_mem_gib'} and prints the peak GPU memory.
    '''
    by_source = {'gt': [], 'det': [], 'fp': []}
    order = np.random.default_rng(args.seed).permutation(len(dataset))[:20 * args.batch_size]
    for i in order:
        item = dataset[int(i)]
        src = item[-1]['source']
        if len(by_source[src]) < args.batch_size:
            by_source[src].append(item)
        if all(len(v) == args.batch_size for v in by_source.values()):
            break
    model.train(); front_end.eval()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    info = {'n_in': int(front_end.n_out), 'sources': {}}
    for src, items in by_source.items():
        if not items:
            print(f"dry run: no '{src}' item among {len(order)} items read")
            continue
        if len(items) == 1:
            items = items * 2                                                             # ZoomNeXt's pooled BatchNorm needs 2 items in training
        batch = seg_collate_fn(items)
        x = seg_inputs(front_end, batch, device)                                          # [B, N + 1, c, c]
        assert x.shape[1:] == (front_end.n_out + 1, args.canvas, args.canvas), f"'{src}' input {tuple(x.shape)}, canvas {args.canvas}"
        with torch.autocast(device.type, dtype=amp_dtype, enabled=device.type == 'cuda'):
            outputs = model(x, batch['box_xyxy'].to(device).float())
        assert outputs[0].shape == (len(items), 1, args.canvas, args.canvas), f"'{src}' main output {tuple(outputs[0].shape)}"
        total, items_loss = model.loss([o.float() for o in outputs], batch['mask'].to(device).float())
        total.backward()
        model.zero_grad(set_to_none=True)
        info['sources'][src] = tuple(x.shape)
        print(f"dry run '{src}': x {tuple(x.shape)}, {len(outputs)} output(s) {tuple(outputs[0].shape)}, "
              f"loss {float(total.detach()):.4f} {items_loss}")
    assert info['sources'], f"dry run: no training item found among {len(order)} items"
    info['peak_mem_gib'] = torch.cuda.max_memory_allocated(device) / 2 ** 30 if device.type == 'cuda' else 0.0
    print(f"dry run: peak GPU memory {info['peak_mem_gib']:.2f} GiB at batch {args.batch_size}, canvas {args.canvas}, "
          f"{args.seg_model} / {args.arm} ({front_end.n_out} + 1 channels)")
    return info


def main(args):
    cfg = load_cfg(args)
    args.name = args.name or f'seg_{args.seg_model}_{args.arm}_s{args.seed}'
    args.output_dir = args.output_dir or os.path.join('weights', args.name)
    args.crop_cache = args.crop_cache or os.path.join(args.data_path, 'crop_cache_seg')
    assert len(args.box_mix) == 3 and min(args.box_mix) >= 0 and abs(sum(args.box_mix) - 1.0) < 1e-6, \
        f"--box_mix must be 3 non-negative fractions summing to 1, got {args.box_mix}"
    assert args.amp_dtype in AMP_DTYPES, f"amp_dtype must be one of {sorted(AMP_DTYPES)}, got {args.amp_dtype!r}"
    amp_dtype = AMP_DTYPES[args.amp_dtype]
    device = torch.device(args.device if args.device == 'cpu' or torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    index_path = os.path.join(args.crop_cache, 'index.json')
    assert os.path.isfile(index_path), f"{index_path} not found: build the crop cache first (bash_files/launch_seg_queue.sh)"
    with open(index_path) as f:
        index = json.load(f)
    front_end, det_args, dataset = build_arm_front_end(args, device)
    # spec §9: the crop cache must hold the same band window as the arm's detector ...
    assert [float(v) for v in index['band_range']] == [float(v) for v in det_args.band_range], \
        f"crop cache band range {index['band_range']} != detector {det_args.name} band range {det_args.band_range}"
    # ... and its training / val ROIs must be that detector's export (rgb uses the raw arm's ROIs)
    roi_arm = 'raw' if args.arm == 'rgb' else args.arm
    assert 'roi_files' in index and roi_arm in index['roi_files'], f"{index_path} has no ROI files of arm {roi_arm}"
    for split, p in index['roi_files'][roi_arm].items():
        assert os.path.basename(p) == f'rois_{det_args.name}_{split}.json', \
            f"crop cache {args.crop_cache} holds the {roi_arm} ROIs of {p}, not of the arm's detector {det_args.name}"
    del index                                                                            # the windows are read by HyperCOD_roi

    roi_kw = dict(roi_margin=args.roi_margin, roi_min=args.roi_min, canvas=args.canvas)
    dataset_train = HyperCOD_roi(args.crop_cache, 'train', args.arm, box_mix=tuple(args.box_mix), gt_jitter=args.gt_jitter,
                                 scale_aug=tuple(args.scale_aug), gain_aug=args.gain_aug, train=True, **roi_kw)
    dataset_val = HyperCOD_roi(args.crop_cache, 'val', args.arm, train=False, **roi_kw)
    assert len(dataset_train) > 0 and len(dataset_val) > 0, \
        f"empty split in {args.crop_cache}: {len(dataset_train)} train / {len(dataset_val)} val items"
    # drop_last: ZoomNeXt's pooled BatchNorm refuses a training batch of one item (289 items % 4 == 1 in the real epoch)
    assert len(dataset_train) >= args.batch_size, f"{len(dataset_train)} training items < batch {args.batch_size}"
    print(f"{args.crop_cache}: {len(dataset_train)} training items per epoch, {len(dataset_val)} val items, arm {args.arm}")
    pin = device.type == 'cuda'
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                               collate_fn=seg_collate_fn, pin_memory=pin, drop_last=True)
    loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                             collate_fn=seg_collate_fn, pin_memory=pin, drop_last=False)

    # stem fold (P, q): fitted once per run, or taken from the checkpoint being resumed / evaluated
    ckpt = torch.load(args.resume, map_location='cpu', weights_only=False) if args.resume else None
    if ckpt is not None:
        for k in ('seg_model', 'arm'):
            assert ckpt['args'][k] == getattr(args, k), f"--resume {args.resume} is a {k}={ckpt['args'][k]!r} run, not {getattr(args, k)!r}"
        P, q = ckpt['P'], ckpt['q']
    else:
        dataset_fit = HyperCOD_roi(args.crop_cache, 'train', args.arm, train=False, **roi_kw)   # deterministic training items
        P, q, _ = fit_arm_rgb_map(front_end, dataset_fit, dataset.wavelens, device, n_pixels=args.fit_pixels,
                                  n_items=args.fit_items, seed=args.seed, num_workers=args.num_workers)
    n_in = front_end.n_out
    assert np.shape(P) == (3, n_in) and np.shape(q) == (3,), f"stem fold P {np.shape(P)} / q {np.shape(q)} for {n_in} arm channels"
    model = build_seg_model(args, n_in, P, q).to(device)
    if ckpt is not None:
        model.load_state_dict(ckpt['model'])
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"{args.seg_model} on arm {args.arm}: {n_in} + 1 input channels, {n_train / 1e6:.2f}M trainable of {n_all / 1e6:.1f}M parameters")

    optimizer = torch.optim.AdamW(model.param_groups(args), lr=args.lr, weight_decay=args.weight_decay)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf   # cosine per epoch, as main_det
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' and amp_dtype == torch.float16 else None   # bf16: none needed

    if args.dry_run:
        return dry_run(model, front_end, dataset_train, args, device, amp_dtype)

    os.makedirs(args.output_dir, exist_ok=True)
    logger = TrainLogger(args, cfg)
    results_path = os.path.join(args.output_dir, f'results_{args.name}.txt')
    eval_kw = dict(amp_dtype=amp_dtype, size_edges=tuple(args.size_edges))
    if args.eval:
        assert ckpt is not None, "--eval needs --resume <main_seg checkpoint>"
        val = evaluate(model, front_end, loader_val, device, logger=logger, epoch=ckpt['epoch'], tag='val', **eval_kw)
        with open(results_path, 'a') as f:
            f.write(json.dumps({'eval': True, 'resume': args.resume, 'epoch': ckpt['epoch'], 'val': val}) + '\n')
        logger.finish()
        return results_path

    start_epoch, best, best_epoch = 0, -1.0, -1
    if ckpt is not None and ckpt.get('optimizer'):
        optimizer.load_state_dict(ckpt['optimizer']); scheduler.load_state_dict(ckpt['lr_scheduler'])
        if scaler is not None and ckpt.get('scaler'):
            scaler.load_state_dict(ckpt['scaler'])
        # carry the best score over, or the first post-resume epoch would overwrite model_best with a worse model
        start_epoch, best, best_epoch = ckpt['epoch'] + 1, float(ckpt['best']), int(ckpt['best_epoch'])

    print(f"Start training from epoch {start_epoch}, best so far {best:.4f}"); start = time.time()
    val = None
    for epoch in range(start_epoch, args.epochs):
        train_stats = train_one_epoch(model, front_end, loader_train, optimizer, device, epoch, scaler=scaler, accumulate=args.accumulate,
                                      max_norm=args.max_norm, logger=logger, print_freq=args.print_freq, amp_dtype=amp_dtype,
                                      total_epochs=args.epochs)
        scheduler.step()
        val = evaluate(model, front_end, loader_val, device, logger=logger, epoch=epoch, tag='val', **eval_kw)
        score = val_score(val)
        logger.scalar('val/score', score, epoch)
        if score > best:                                         # before model_last, so model_best includes this epoch
            best, best_epoch = score, epoch
            save_checkpoint(os.path.join(args.output_dir, 'model_best'), model, optimizer, scaler, scheduler, epoch, args, best, best_epoch, P, q)
        if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
            save_checkpoint(os.path.join(args.output_dir, 'model_last'), model, optimizer, scaler, scheduler, epoch, args, best, best_epoch, P, q)
        with open(results_path, 'a') as f:
            f.write(json.dumps({'epoch': epoch, 'train': train_stats, 'val': val, 'score': score, 'best': best, 'best_epoch': best_epoch}) + '\n')
    print(f"Training time {datetime.timedelta(seconds=int(time.time() - start))}, best val mean(S, Fw) {best:.4f} at epoch {best_epoch}")

    # model_best re-evaluated after a reload (checks the checkpoint round trip); the queue skips runs with this final line
    best_ckpt = torch.load(os.path.join(args.output_dir, 'model_best'), map_location='cpu', weights_only=False)
    model.load_state_dict(best_ckpt['model'])
    val_best = evaluate(model, front_end, loader_val, device, logger=logger, epoch=args.epochs, tag='val_best', **eval_kw)
    with open(results_path, 'a') as f:
        f.write(json.dumps({'epoch': best_ckpt['epoch'], 'final': True, 'best_epoch': best_ckpt['epoch'], 'best': best,
                            'val': val_best, 'val_last': val}) + '\n')
    logger.finish()
    return results_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Stage-2 segmentation', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python -m pytest tests/test_main_seg.py -v`
Expected: PASS (5 passed). The fixture index carries Task 3's keys (`frame_hw`, `grow`, `min_side`, `arms`, `roi_files`, window `object`, object `area`), so the real `HyperCOD_roi` of Task 4 reads it. The dry run on the fixture prints one batch for each of `'gt'`, `'det'` and `'fp'` (the single fp item doubled), `x (4, 134, 32, 32)` for the full sources, and a stem fold with R² ≈ 0.95 on the raw arm.

- [ ] **Step 5: Review Focus tests: no training batch of one, and a crop cache built from another detector (write them, run them)**

Append to `tests/test_main_seg.py`:

```python


class PairSeg(TinySeg):
    '''TinySeg that, like ZoomNeXt (BatchNorm on a globally pooled map), refuses a training batch of one item.'''

    def forward(self, x, box_xyxy):
        assert x.shape[0] >= 2 or not self.training, f"batch of {x.shape[0]} item in training"
        return super().forward(x, box_xyxy)


def test_training_never_sees_a_batch_of_one(synthetic_root, tmp_path, monkeypatch):
    root, _, _ = synthetic_root
    det, _ = _setup(root, tmp_path, monkeypatch)
    monkeypatch.setattr(main_seg, 'build_seg_model', lambda args, n_in, P, q: PairSeg(n_in))
    # 9 object items + round(9 x 0.1 / 0.9) = 1 false-positive item = 10 items per epoch: batch 3 leaves a last batch of 1
    main_seg.main(_args(root, tmp_path, det, '--epochs', '1', '--batch_size', '3'))
    lines = [json.loads(l) for l in (tmp_path / 'seg' / 'results_tseg.txt').read_text().strip().splitlines()]
    assert lines[-1]['final'] is True
    # the dry run finds a single 'fp' item: it must still run that source as a batch of >= 2
    info = main_seg.main(_args(root, tmp_path, det, '--dry_run', '--batch_size', '3'))
    assert 'fp' in info['sources'] and all(shape[0] >= 2 for shape in info['sources'].values())


def test_crop_cache_built_from_another_detector_raises(synthetic_root, tmp_path, monkeypatch):
    root, _, _ = synthetic_root
    det, _ = _setup(root, tmp_path, monkeypatch)                         # raw-arm detector stub at <tmp>/raw133_A/model_best
    index_path = tmp_path / 'crops' / 'index.json'
    index = json.loads(index_path.read_text())
    # the cache was built with the ec10 detector's ROIs in the raw slot
    index['roi_files'] = {a: {s: f'results/det/rois_sel10g_clean_A_{s}.json' for s in ('train', 'val')} for a in index['arms']}
    index_path.write_text(json.dumps(index))
    with pytest.raises(AssertionError, match="not of the arm's detector raw133_A"):
        main_seg.main(_args(root, tmp_path, det, '--epochs', '1'))
    assert not (tmp_path / 'seg' / 'results_tseg.txt').exists()         # raised before any training
    # the rgb control reads the raw slot as well
    with pytest.raises(AssertionError, match="not of the arm's detector raw133_A"):
        main_seg.main(_args(root, tmp_path, det, '--epochs', '1', '--arm', 'rgb'))
```

Run: `OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python -m pytest tests/test_main_seg.py -k "batch_of_one or another_detector" -v`
Expected: PASS with the implementation of Step 3 (`drop_last=True`, the doubled single-item dry-run batch and the ROI-file provenance check are what these guard; without them the first test fails with `AssertionError: batch of 1 item in training` and the second does not raise).

- [ ] **Step 6: Run the file and the full suite**

Run: `OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python -m pytest tests/test_main_seg.py -v`
Expected: PASS (7 passed).
Then: `OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python -m pytest tests/ -q`. It must stay green.

- [ ] **Step 7: Commit and push**

```bash
git add main_seg.py cfg/seg.yaml tests/test_main_seg.py
git commit -m "feat(seg): main_seg.py - train one (model, arm, seed) on the crop cache

Front end rebuilt from the detector checkpoint (main_det_rois.detector_args) and checked bit-identical to its filter bank,
the arm and the crop cache's ROI files checked against the detector, the stem fold (P, q) fitted on deterministic training
pixels, no training batch of one, model_best on val mean(S, Fw), resume / --eval / --dry_run, and cfg/seg.yaml with
per-model sections.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QB4DScDwiA2DkVpbDopffV"
git push origin worktree-det
git push origin worktree-det:master
```

---

### Task 13: End-to-end test evaluation (`main_seg_eval.py`)

**Files:**
- Create: `main_seg_eval.py`
- Test: `tests/test_main_seg_eval.py`

**Interfaces:**
- Consumes:
  - `main_det_rois.load_detector(ckpt_path, args, device) -> (model, det_args, run_name)` (T2). `args` is a `main_det.get_args_parser()` namespace.
  - `data_loader.roi_crops.match_rois(rois [K,4], labels [H,W] int32, ids, min_cover=0.01) -> [K] int` (T3), `pixel_box(box, H, W)` (T3).
  - `place_on_canvas(crop, canvas, scale) -> (out [C,c,c], valid [c,c] bool, (oy, ox), s)`, `box_to_canvas(box, roi_int, offset, hw, s, canvas)`, `rasterise_box(box, canvas)` (T4): the same geometry as `HyperCOD_roi`, so test items equal validation items.
  - `canvas_to_roi(canvas_map, (oy, ox), s, (h, w)) -> [h,w] float32` (T4).
  - `seg_collate_fn(batch) -> {"img", "mask", "box_map", "valid", "box_xyxy", "meta"}` (T4). It takes tuples `(img, mask, box_map, valid, meta)` and builds `box_xyxy` from `meta["box_canvas"]`.
  - `train_eval.seg_metrics.SegMetrics(size_edges, minmax=False)` with `.update(pred, gt, area=None, frame=None, key=None)`, `.update_fp(pred, frame=None, key=None)`, `.summary()` and `.per_image` (T5). `update()` appends exactly one row to `per_image`. `minmax=True` is the HyperCOD paper protocol.
  - `bootstrap_seg(runs_a, runs_b, keys, n, seed)` (T5): frame (cluster) bootstrap, since every row carries its frame.
  - `models.seg_models.build_front_end(arm, det_args, dataset, det_state=None) -> module with .n_out` and `build_seg_model(args, n_in, P, q)` (T8–T10). `forward(x, box_xyxy)[0]` is the main output's logits.
  - `train_eval.train_eval_seg.seg_inputs(front_end, batch, device)` and `paste_back(prob_canvas, meta, out) -> out` (T11).
  - `main_seg.get_args_parser()` and `main_seg.load_cfg(args)` (T12).
  - T12 checkpoint layout: `{'model': state_dict, 'args': vars(args) (seg_model, arm, seed, canvas, det_ckpt, ...), 'epoch': int, optional 'P' [3,N] and 'q' [3]}`. The state dict may leave out frozen parameters.
- Produces:
  - `main_seg_eval.get_args_parser()` and `main(args) -> compare dict`.
  - `roi_window(roi, H, W) -> [x1,y1,x2,y2] int` (the `pixel_box` rule, asserts a non-empty ROI).
  - `make_item(img, gt, roi, box, canvas, frame, source) -> ((img, mask, box_map, valid, meta), gt_crop)`.
  - `ZS_NAME = 'zs_sam2box_rgb'` and `ZS_MODEL = 'sam2box_zs'`.
  - Files: `results/seg/<run>/eval_roi_oracle.json`, `eval_full_oracle.json`, `eval_full_det.json` (summary, `paper` {MAE, E_mean, S, F_adp} from `SegMetrics(minmax=True)`, n_frames, n_frames_no_roi, n_det_rois, n_fp_rois), `per_image.pkl` ({level: {"keys", "rows"}}) and `results/seg/compare.json` ({runs, groups {"<model>|<arm>": {runs, seeds, <level>: {key: {mean, std, n}}}}, comparisons {level: {"<model>: <armA> - <armB>" | "<arm>: <modelA> - <modelB>": bootstrap_seg result}}}).
  - Spec §9: every front end (raw / ec10 / ec24) is checked against its detector checkpoint's filter bank (`det_state`); a missing or incomplete ROI export raises.

- [ ] **Step 1: Write the failing test**

`tests/test_main_seg_eval.py`:
```python
import json
import pickle
import numpy as np
import pytest
import torch
import torch.nn as nn

pytest.importorskip('py_sod_metrics')
import main_det
import main_seg
import main_seg_eval
from tests.conftest import write_sample
from tests.test_main_det import _args
from data_loader.cube_cache import build_cube_cache

OBJ_BOX = [20.0, 10.0, 26.0, 16.0]          # conftest OBJ_SLICE (rows 10:16, cols 20:26) as xyxy
FP_BOX = [2.0, 36.0, 8.0, 42.0]             # a detector box far from the object (covers 0 % of its mask)
FRAME_PX = 48 * 40


class BoxEcho(nn.Module):
    '''
    Stand-in segmentation model with the build_seg_model API: logits = +-gain from the box channel (the last input
    channel), so the predicted mask is exactly the prompt box. A wrong canvas placement, box map or paste-back shows up
    as IoU < 1 on the oracle levels; gain starts at 0 (p = 0.5 everywhere) so an unloaded checkpoint shows up too.
    '''
    def __init__(self, n_in, gain=0.0):
        super().__init__()
        self.n_in = n_in
        self.gain = nn.Parameter(torch.tensor(float(gain)))

    def forward(self, x, box_xyxy):
        assert x.shape[1] == self.n_in + 1, f"expected {self.n_in} arm channels + the box channel, got {x.shape[1]}"
        assert box_xyxy.shape == (x.shape[0], 4)
        return [(x[:, -1:] * 2 - 1) * self.gain]


def _setup(synthetic_root, tmp_path, monkeypatch):
    '''Fixture frames + caches, a trained (1 epoch) raw-band detector, its hand-written test ROI export.'''
    root, _, _ = synthetic_root
    monkeypatch.delenv('RANK', raising=False)
    write_sample(str(root), 'test', '8', np.random.default_rng(1))     # a second test frame, left without any detector ROI
    build_cube_cache(str(root), 'train', num_workers=0); build_cube_cache(str(root), 'test', num_workers=0)
    out_a = tmp_path / 'det_A'
    # --raw-bands: the raw and rgb arms (and the zero-shot control) need the raw-band detector (build_front_end's arm check)
    main_det.main(_args(root, tmp_path, **{'--session': 'A', '--raw-bands': None, '--name': 'tA', '--output-dir': str(out_a),
                                           '--top_k': '3'}))
    roi_dir = tmp_path / 'rois'
    roi_dir.mkdir()
    # frame 7: one ROI on the object (expand_box(OBJ_BOX, 1.5, 0)) and one false positive; frame 8: no ROI
    rois = {'7': {'rois': [[18.5, 8.5, 27.5, 17.5, 0.9], [0.5, 34.5, 9.5, 43.5, 0.4]], 'boxes': [OBJ_BOX + [0.9], FP_BOX + [0.4]],
                  'gt_boxes': [OBJ_BOX]},
            '8': {'rois': [], 'boxes': [], 'gt_boxes': [OBJ_BOX]}}
    (roi_dir / 'rois_det_A_test.json').write_text(json.dumps(rois))
    return root, out_a / 'model_best', roi_dir


def _write_run(tmp_path, name, arm, seed, det_ckpt):
    a = main_seg.get_args_parser().parse_args(['--seg_model', 'sam2unet', '--arm', arm, '--det_ckpt', str(det_ckpt), '--canvas', '32',
                                               '--seed', str(seed), '--name', name, '--crop_cache', str(tmp_path / 'crops'),
                                               '--output_dir', str(tmp_path / 'weights' / name)])
    d = tmp_path / 'weights' / name
    d.mkdir(parents=True)
    torch.save({'model': BoxEcho(0, gain=20.0).state_dict(), 'args': vars(a), 'epoch': 3}, d / 'model_best')


def _eval_args(root, tmp_path, roi_dir, *extra):
    return main_seg_eval.get_args_parser().parse_args(
        ['--weights_dir', str(tmp_path / 'weights'), '--out_dir', str(tmp_path / 'seg'), '--roi_dir', str(roi_dir),
         '--data_path', str(root), '--split_file', str(tmp_path / 'val.json'), '--device', 'cpu', '--num_workers', '0',
         '--roi_min', '0', '--min_area', '10', '--n_boot', '50', '--batch_size', '2', '--runs_per_pass', '2', *extra])


def test_roi_window_contains_the_roi():
    assert main_seg_eval.roi_window([18.5, 8.5, 27.5, 17.5], 48, 40) == [18, 8, 28, 18]
    assert main_seg_eval.roi_window([-3.0, 40.2, 12.0, 60.0], 48, 40) == [0, 40, 12, 48]      # clipped to the frame
    with pytest.raises(AssertionError, match='empty ROI'):
        main_seg_eval.roi_window([5.0, 5.0, 5.0, 9.0], 48, 40)


def test_eval_levels_runs_and_compare(synthetic_root, tmp_path, monkeypatch):
    root, det_ckpt, roi_dir = _setup(synthetic_root, tmp_path, monkeypatch)
    monkeypatch.setattr(main_seg_eval, 'build_seg_model', lambda a, n_in, P, q: BoxEcho(n_in))
    for name, arm, seed in (('r_s0', 'raw', 0), ('r_s1', 'raw', 1), ('g_s0', 'rgb', 0)):
        _write_run(tmp_path, name, arm, seed, det_ckpt)
    out = main_seg_eval.main(_eval_args(root, tmp_path, roi_dir, '--runs', 'r_s0', 'r_s1', 'g_s0'))

    run = tmp_path / 'seg' / 'r_s0'
    roi = json.loads((run / 'eval_roi_oracle.json').read_text())
    full = json.loads((run / 'eval_full_oracle.json').read_text())
    det = json.loads((run / 'eval_full_det.json').read_text())
    assert (roi['arm'], roi['seed'], roi['epoch'], roi['det_run']) == ('raw', 0, 3, 'det_A')
    # (a), (b): the oracle box is the object, so the box-echo mask is the GT exactly, at ROI level and pasted back
    assert roi['summary']['n'] == 2 and roi['summary']['IoU'] == pytest.approx(1.0) and roi['summary']['MAE'] < 1e-6
    assert full['summary']['n'] == 2 and full['summary']['IoU'] == pytest.approx(1.0) and full['summary']['S'] == pytest.approx(1.0, abs=1e-4)
    # (c): frame 7 = object + false-positive box (IoU 36/72), frame 8 = no ROI -> empty mask (IoU 0)
    assert det['summary']['IoU'] == pytest.approx(0.25) and det['summary']['fp_false_mask_rate'] == pytest.approx(1.0)
    assert (det['n_frames'], det['n_frames_no_roi'], det['n_det_rois'], det['n_fp_rois']) == (2, 1, 2, 1)
    assert set(det['paper']) == {'MAE', 'E_mean', 'S', 'F_adp'} and det['paper']['MAE'] == pytest.approx(36 / FRAME_PX, abs=1e-6)
    with open(run / 'per_image.pkl', 'rb') as f:
        pi = pickle.load(f)
    assert pi['roi_oracle']['keys'] == [('7', 0), ('8', 0)] and pi['full_det']['keys'] == ['7', '8']
    assert len(pi['full_det']['rows']) == 2                                  # the fp ROI is not a frame row
    assert [r['frame'] for r in pi['roi_oracle']['rows']] == ['7', '8']     # rows carry their frame: frame bootstrap

    # compare.json: seeds pooled per (model, arm); identical models -> zero differences with a degenerate CI
    cmp = json.loads((tmp_path / 'seg' / 'compare.json').read_text())
    assert cmp == json.loads(json.dumps(out, default=main_seg_eval.to_json))
    g = cmp['groups']['sam2unet|raw']
    assert g['seeds'] == [0, 1] and g['full_oracle']['IoU']['n'] == 2 and g['full_oracle']['IoU']['std'] == pytest.approx(0.0)
    assert set(cmp['comparisons']) == {'roi_oracle', 'full_oracle', 'full_det'}
    c = cmp['comparisons']['full_det']['sam2unet: raw - rgb']
    assert c['S']['diff'] == pytest.approx(0.0) and c['IoU']['lo'] == pytest.approx(0.0) and c['IoU']['hi'] == pytest.approx(0.0)


def test_downscaled_rois_paste_back_near_the_gt(synthetic_root, tmp_path, monkeypatch):
    # ROIs of >= 40 px on a 32 px canvas: placed at s < 1 and resized back, so the box-echo mask is the GT up to interpolation
    root, det_ckpt, roi_dir = _setup(synthetic_root, tmp_path, monkeypatch)
    monkeypatch.setattr(main_seg_eval, 'build_seg_model', lambda a, n_in, P, q: BoxEcho(n_in))
    _write_run(tmp_path, 'r_s0', 'raw', 0, det_ckpt)
    main_seg_eval.main(_eval_args(root, tmp_path, roi_dir, '--runs', 'r_s0', '--roi_min', '40'))
    with open(tmp_path / 'seg' / 'r_s0' / 'per_image.pkl', 'rb') as f:
        pi = pickle.load(f)
    full = json.loads((tmp_path / 'seg' / 'r_s0' / 'eval_full_oracle.json').read_text())['summary']
    assert len(pi['roi_oracle']['rows']) == 2 and full['IoU'] > 0.6 and full['S'] > 0.85


def test_zero_shot_control_uses_the_pretrained_stem(synthetic_root, tmp_path, monkeypatch):
    root, det_ckpt, roi_dir = _setup(synthetic_root, tmp_path, monkeypatch)
    calls = []
    def _build(a, n_in, P, q):
        calls.append((a.seg_model, a.arm, a.canvas, n_in, P.copy(), q.copy()))
        return BoxEcho(n_in, gain=20.0)
    monkeypatch.setattr(main_seg_eval, 'build_seg_model', _build)
    main_seg_eval.main(_eval_args(root, tmp_path, roi_dir, '--zero_shot', '--zero_shot_det_ckpt', str(det_ckpt), '--canvas', '32'))
    (model, arm, canvas, n_in, P, q), = calls
    assert (model, arm, canvas, n_in) == ('sam2box', 'rgb', 32, 3)
    assert np.array_equal(P, np.eye(3)) and np.array_equal(q, np.zeros(3))     # fold_stem(conv, I, 0) = the RGB stem itself
    det = json.loads((tmp_path / 'seg' / main_seg_eval.ZS_NAME / 'eval_full_det.json').read_text())
    assert det['seg_model'] == main_seg_eval.ZS_MODEL and det['ckpt'] == 'zero_shot' and det['summary']['IoU'] == pytest.approx(0.25)


def test_missing_roi_export_raises(synthetic_root, tmp_path, monkeypatch):
    root, det_ckpt, roi_dir = _setup(synthetic_root, tmp_path, monkeypatch)
    (roi_dir / 'rois_det_A_test.json').unlink()
    _write_run(tmp_path, 'r_s0', 'raw', 0, det_ckpt)
    with pytest.raises(AssertionError, match='rois_det_A_test.json missing'):
        main_seg_eval.main(_eval_args(root, tmp_path, roi_dir, '--runs', 'r_s0'))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python -m pytest tests/test_main_seg_eval.py -v`

Expected: FAIL. Collection stops with `ModuleNotFoundError: No module named 'main_seg_eval'`.

- [ ] **Step 3: Write minimal implementation**

`main_seg_eval.py`:
```python
"""
Stage 2: end-to-end test evaluation of the segmentation runs (spec §5.6, §8). For every run (weights/<run>/<ckpt>) and the
optional zero-shot SAM2.1 control, on the 70 test frames:
  (a) roi_oracle   ROI level with oracle ROIs: every GT object's box grown by the export rule (x 1.5, >= 256 px), metrics on
                   the ROI crop at native size (target = the GT union inside the ROI)
  (b) full_oracle  the masks of (a) pasted back into the 1680 x 1240 frame (overlapping ROIs merged by maximum)
  (c) full_det     the whole sensor chain: the arm's own detector ROIs (results/det/rois_<det_run>_test.json, exported by
                   main_det_rois.py) pasted back; a frame without any ROI is an empty mask (a miss). The detector's
                   false-positive ROIs (< 1 % of every object mask) give the false-mask rate. (c) is also scored with the
                   HyperCOD paper's protocol (SegMetrics(minmax=True): per-image min-max + uint8, Table 2 columns MAE,
                   mean E, S, adaptive F)
The frames are read once per pass (HyperCOD_data: fp16 cache, O_DIRECT full-frame read, the loader's p99 scaling); the runs
of one pass (--runs_per_pass) keep their models on the GPU, so 29 SAM2-L sized runs need 5 passes instead of 29. Test items
are built with the crop cache's geometry (data_loader.roi_crops: pixel_box, place_on_canvas, box_to_canvas, rasterise_box),
so a test ROI gives exactly the tensors HyperCOD_roi gives for the same ROI. Every row carries its frame, so the paired
bootstrap resamples frames.
  CUDA_VISIBLE_DEVICES=0 python main_seg_eval.py --runs seg_sam2unet_raw_s0 seg_sam2unet_ec10_s0 --zero_shot
  CUDA_VISIBLE_DEVICES=0 python main_seg_eval.py --runs seg_sam2unet_raw_s0 --ckpt model_last --out_dir results/seg_last
Writes <out_dir>/<run>/eval_roi_oracle.json, eval_full_oracle.json, eval_full_det.json, per_image.pkl (per-image rows,
re-poolable with train_eval.seg_metrics.pool) and <out_dir>/compare.json: mean +- std over seeds per (model, arm), and
paired bootstrap CIs (train_eval.seg_metrics.bootstrap_seg) for the same model across arms and the same arm across models.
"""
import os
if "RANK" not in os.environ and "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import time
import pickle
import argparse
import itertools
import numpy as np
import torch
import torch.utils.data as Dataset

import main_det
import main_seg
from main_det_rois import load_detector
from data_loader.my_dataset import HyperCOD_data
from data_loader.boxes import boxes_from_mask, expand_box
from data_loader.roi_crops import (match_rois, pixel_box, place_on_canvas, canvas_to_roi, box_to_canvas, rasterise_box,
                                   seg_collate_fn)
from models.seg_models import build_front_end, build_seg_model
from train_eval.train_eval_seg import seg_inputs, paste_back
from train_eval.seg_metrics import SegMetrics, bootstrap_seg

LEVELS = ['roi_oracle', 'full_oracle', 'full_det']        # spec §8 (a), (b), (c): written per run and bootstrapped
PAPER_KEYS = ['MAE', 'E_mean', 'S', 'F_adp']              # HyperCOD Table 2 columns; its "E" is taken as mean E (inferred
                                                          # from SAM2-UNet's eval.py, which prints exactly this set)
BOOT_KEYS = ['S', 'E_mean', 'E_max', 'Fw', 'F_adp', 'MAE', 'IoU']
ARM_ORDER = ['raw', 'ec10', 'ec24', 'rgb']
ZS_NAME = 'zs_sam2box_rgb'                                # the zero-shot control's run name (and results folder)
ZS_MODEL = 'sam2box_zs'                                   # ... and its model label in compare.json


def get_args_parser():
    parser = argparse.ArgumentParser('Stage-2 end-to-end test evaluation', add_help=False)
    # runs
    parser.add_argument('--runs', type=str, nargs='*', default=[], help='segmentation run names: <weights_dir>/<run>/<ckpt>')
    parser.add_argument('--ckpt', type=str, default='model_best', choices=['model_best', 'model_last'], help='checkpoint evaluated per run')
    parser.add_argument('--weights_dir', type=str, default='weights', help='directory holding the run folders')
    parser.add_argument('--zero_shot', action='store_true',
                        help='also evaluate the untrained control: SAM2.1 + box prompt on the pseudo-RGB render (arm rgb, P = I, q = 0)')
    parser.add_argument('--zero_shot_det_ckpt', type=str, default='weights/raw133_A/model_best',
                        help='detector whose test ROIs and front end the zero-shot control uses')
    parser.add_argument('--sam2_ckpt', type=str, default='', help="zero-shot control's SAM2.1 checkpoint; empty = main_seg's default")
    parser.add_argument('--canvas', type=int, default=512, help='canvas of the zero-shot control when no trained run fixes it')
    # protocol (spec §8)
    parser.add_argument('--roi_dir', type=str, default='results/det', help='where main_det_rois.py wrote rois_<det_run>_test.json')
    parser.add_argument('--roi_margin', type=float, default=1.5, help='oracle ROI = GT box grown by this factor (the export rule)')
    parser.add_argument('--roi_min', type=float, default=256, help='... and at least this many px per side')
    parser.add_argument('--min_area', type=int, default=100, help='GT components below this many px are JPEG specks, not objects')
    parser.add_argument('--min_cover', type=float, default=0.01, help='a detector ROI covering less of every object mask is a false positive')
    parser.add_argument('--size_edges', type=int, nargs=2, default=[2000, 20000], help='object-size buckets by GT area (px)')
    parser.add_argument('--n_boot', type=int, default=2000, help='paired bootstrap resamples')
    parser.add_argument('--seed', type=int, default=0, help='bootstrap seed')
    # data and compute
    parser.add_argument('--data_path', type=str, default='/data2/chaoyi/HyperCOD/Raw data', help='HyperCOD root')
    parser.add_argument('--cache_dir', type=str, default='', help='fp16 frame cache, default <data_path>/cache_fp16')
    parser.add_argument('--split_file', type=str, default='', help='val ids json handed to the detector loader')
    parser.add_argument('--device', type=str, default='cuda', help="'cuda', 'cuda:N' or 'cpu'")
    parser.add_argument('--batch_size', type=int, default=8, help='ROI items per forward pass')
    parser.add_argument('--runs_per_pass', type=int, default=6, help='runs whose models share one pass over the frames (GPU memory)')
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True, help='bf16 autocast for the model (CUDA only)')
    parser.add_argument('--num_workers', type=int, default=2, help='frame-reading DataLoader workers')
    parser.add_argument('--limit', type=int, default=0, help='only the first N test frames (smoke runs)')
    parser.add_argument('--out_dir', type=str, default='results/seg', help='per-run folders and compare.json go here')
    return parser


def frame_collate_fn(batch):
    '''
    batch_size 1 over HyperCOD_data -> (img [C, H, W] fp16 p99-scaled tensor, gt [H, W] bool tensor, name). Tensors, so a
    DataLoader worker hands the 0.55 GB frame over through shared memory instead of pickling a numpy array.
    '''
    img, gt, name = batch[0]
    return torch.from_numpy(img), torch.from_numpy(gt[0] > 0.5), name


def roi_window(roi, H, W):
    '''
    Integer pixel window [x1, y1, x2, y2] of a float ROI: data_loader.roi_crops.pixel_box (floor x1/y1, ceil x2/y2, clipped
    to the frame; the rounding of box_metrics.mask_coverage and of the crop cache), so the window always contains the ROI.
    '''
    x1, y1, x2, y2 = pixel_box(roi, H, W)
    assert x2 > x1 and y2 > y1, f"empty ROI {[float(v) for v in roi[:4]]} in a {H} x {W} frame"
    return [x1, y1, x2, y2]


def make_item(img, gt, roi, box, canvas, frame, source):
    '''
    One test item, built like HyperCOD_roi's deterministic validation items but cut from the full frame (the test frames
    are not in the crop cache): the ROI window at native resolution placed on the canvas (scale 1, no augmentation) and
    the box channel = rasterise_box(box_to_canvas(box)), i.e. HyperCOD_roi's own geometry (box_to_canvas clips the box to
    the placed region, so the box channel is 0 on the padding).
    img: [C, H, W] fp16 p99-scaled numpy; gt: [H, W] bool; roi, box: frame px xyxy; source: 'gt' | 'det' | 'fp'
    Returns (the seg_collate_fn tuple (img, mask, box_map, valid, meta), GT crop [h, w] bool).
    '''
    H, W = gt.shape
    x1, y1, x2, y2 = roi_window(roi, H, W)
    crop = img[:, y1:y2, x1:x2]                                                        # [C, h, w] view
    gt_crop = gt[y1:y2, x1:x2]                                                        # [h, w] bool
    h, w = gt_crop.shape
    out, valid, (oy, ox), s = place_on_canvas(crop, canvas=canvas, scale=1.0)         # [C, c, c], [c, c] bool
    mask = place_on_canvas(gt_crop[None].astype(np.float32), canvas=canvas, scale=1.0)[0]   # [1, c, c]
    # the pre-expansion box, clipped to the window first (detector boxes are only clipped to the frame)
    bx = np.clip(np.asarray(box[:4], dtype=np.float64), [x1, y1, x1, y1], [x2, y2, x2, y2])
    box_canvas = box_to_canvas(bx, (x1, y1, x2, y2), (oy, ox), (h, w), s, canvas)      # canvas px, clipped to the placed region
    box_map = rasterise_box(box_canvas, canvas)                                        # [c, c] float32
    meta = {'frame': frame, 'roi': [x1, y1, x2, y2], 'box': [float(v) for v in bx], 'box_canvas': [float(v) for v in box_canvas],
            'source': source, 'offset': (int(oy), int(ox)), 's': float(s), 'roi_hw': (h, w), 'obj_area': int(gt_crop.sum())}
    item = (np.asarray(out, dtype=np.float16), np.asarray(mask, dtype=np.float32).reshape(1, canvas, canvas), box_map[None],
            valid.astype(np.float32)[None], meta)
    return item, gt_crop


@torch.no_grad()
def predict(model, front_end, items, device, batch_size=8, amp=True):
    '''Sigmoid probabilities [n, c, c] float32 of the model's main output (index 0) for a list of items.'''
    probs = []
    for i in range(0, len(items), batch_size):
        batch = seg_collate_fn(items[i:i + batch_size])
        x = seg_inputs(front_end, batch, device)                                     # [B, n_in + 1, c, c], front end as in training
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp and device.type == 'cuda'):
            logits = model(x, batch['box_xyxy'].to(device))[0]                       # [B, 1, c, c]
        probs.append(torch.sigmoid(logits.float())[:, 0].cpu().numpy())             # [B, c, c]
    return np.concatenate(probs, axis=0) if probs else np.zeros((0, 1, 1), dtype=np.float32)


def det_cli_args(args):
    '''main_det-style namespace for load_detector: data and device keys from this CLI, the model keys come from the checkpoint.'''
    return main_det.get_args_parser().parse_args(['--data-path', args.data_path, '--cache-dir', args.cache_dir, '--split-file', args.split_file,
                                                   '--device', 'cpu', '--num_workers', str(args.num_workers), '--min-area', str(args.min_area),
                                                   '--no-wandb'])


def load_detector_info(args, det_ckpt, cache):
    '''
    Everything a run needs from its detector, once per detector checkpoint: its args (the front end is rebuilt from them),
    its state dict (the rebuilt front end is checked against its filter bank, spec §9), its run name, its exported test
    ROIs and a test HyperCOD_data built with its own filter settings (build_front_end and the frame reads use it). The
    detector network itself is dropped: the ROIs come from main_det_rois.py's export.
    '''
    key = os.path.abspath(det_ckpt)
    if key not in cache:
        model, det_args, det_run = load_detector(det_ckpt, det_cli_args(args), torch.device('cpu'))
        del model
        det_state = torch.load(det_ckpt, map_location='cpu', weights_only=False)['model']
        det_args.data_path, det_args.cache_dir = args.data_path, args.cache_dir   # the frames of this machine, whatever the ckpt says
        dataset = HyperCOD_data(split='test', **main_det.dataset_kwargs(det_args))
        path = os.path.join(args.roi_dir, f'rois_{det_run}_test.json')
        assert os.path.exists(path), f"{path} missing: export it with python main_det_rois.py --resume {det_ckpt} --split test (bash_files/launch_rois_all.sh)"
        with open(path) as f:
            rois = json.load(f)
        missing = [n for n in dataset.img_name if n not in rois]
        assert not missing, f"{path} lacks test frames {missing[:5]} ({len(missing)} in all): not a test-split export of {det_run}?"
        cache[key] = dict(key=key, det_args=det_args, det_state=det_state, det_run=det_run, rois=rois, dataset=dataset, roi_file=path)
    return cache[key]


def load_run(args, run, det_cache):
    '''A trained run: its main_seg args and epoch (the weights are loaded later, per pass) and its detector.'''
    path = os.path.join(args.weights_dir, run, args.ckpt)
    assert os.path.exists(path), f"{path} not found"
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    a, epoch = argparse.Namespace(**ckpt['args']), ckpt.get('epoch')
    del ckpt                                                                            # ~0.9 GB for a SAM2-L run
    assert a.arm in ARM_ORDER, f"{path}: unknown arm {a.arm!r}"
    return dict(name=run, path=path, args=a, zero_shot=False, model_label=a.seg_model, arm=a.arm, seed=int(a.seed),
                canvas=int(a.canvas), epoch=epoch, det=load_detector_info(args, a.det_ckpt, det_cache))


def zero_shot_spec(args, canvas, det_cache):
    '''The zero-shot control: SAM2.1 + box prompt on the pseudo-RGB render, built by main_seg's own parser and cfg, never trained.'''
    argv = ['--seg_model', 'sam2box', '--arm', 'rgb', '--det_ckpt', args.zero_shot_det_ckpt, '--canvas', str(canvas),
            '--seed', '0', '--name', ZS_NAME]
    if args.sam2_ckpt:
        argv += ['--sam2_ckpt', args.sam2_ckpt]
    a = main_seg.get_args_parser().parse_args(argv)
    main_seg.load_cfg(a)
    return dict(name=ZS_NAME, path=None, args=a, zero_shot=True, model_label=ZS_MODEL, arm='rgb', seed=0, canvas=canvas,
                epoch=None, det=load_detector_info(args, args.zero_shot_det_ckpt, det_cache))


def build_run_model(spec, front_end, device):
    '''
    The run's segmentation model in eval mode. Trained run: built with its checkpoint's args, then its weights loaded; a
    checkpoint may leave out frozen parameters (rebuilt from the pretrained file), nothing else. Zero-shot: the stem folded
    with P = I, q = 0 (the rgb front end already is the ImageNet-normalised pseudo-RGB image), so it is the pretrained model.
    '''
    n = front_end.n_out
    if spec['zero_shot']:
        assert n == 3, f"the zero-shot control needs the 3-channel rgb front end, got {n} channels"
        return build_seg_model(spec['args'], n, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)).to(device).eval()
    ckpt = torch.load(spec['path'], map_location='cpu', weights_only=False)
    # (P, q) only initialise the folded stem, whose trained weights the state dict overwrites; zeros when not saved
    P = np.zeros((3, n), dtype=np.float32) if ckpt.get('P') is None else np.asarray(ckpt['P'], dtype=np.float32)
    q = np.zeros(3, dtype=np.float32) if ckpt.get('q') is None else np.asarray(ckpt['q'], dtype=np.float32)
    assert P.shape == (3, n), f"{spec['path']}: P is {P.shape}, the {spec['arm']} front end gives {n} channels"
    model = build_seg_model(spec['args'], n, P, q)
    missing, unexpected = model.load_state_dict(ckpt['model'], strict=False)
    frozen = {k for k, p in model.named_parameters() if not p.requires_grad}
    assert not unexpected and set(missing) <= frozen, \
        f"{spec['path']}: unexpected keys {unexpected[:5]}, missing non-frozen keys {sorted(set(missing) - frozen)[:5]}"
    del ckpt
    return model.to(device).eval()


def new_state(size_edges):
    '''
    Per-run accumulators: one SegMetrics per level (full_det_paper with the HyperCOD paper's min-max protocol), the
    per_image row index and pairing key of every scored image.
    '''
    metrics = {lvl: SegMetrics(size_edges=size_edges) for lvl in LEVELS}
    metrics['full_det_paper'] = SegMetrics(size_edges=size_edges, minmax=True)
    levels = list(metrics)
    return dict(metrics=metrics, rows={lvl: [] for lvl in levels}, keys={lvl: [] for lvl in levels},
                n_frames=0, n_frames_no_roi=0, n_det_rois=0, n_fp_rois=0)


def record(state, level, pred, gt, key):
    '''
    SegMetrics.update plus the bookkeeping that pairs this image across runs (the row index skips update_fp rows). The row
    carries its frame (key = (frame, k) at ROI level, the frame id at full-frame levels), so bootstrap_seg resamples frames.
    '''
    m = state['metrics'][level]
    state['rows'][level].append(len(m.per_image))
    state['keys'][level].append(key)
    m.update(pred, gt, frame=str(key[0]) if isinstance(key, tuple) else str(key), key=key)


def evaluate_frame(spec, oracle, det, gt, name, device, args):
    '''Levels (a), (b), (c) of one frame for one run. oracle / det items: lists of (item, GT crop); det also the matches.'''
    st, (H, W) = spec['state'], gt.shape
    model, front = spec['model'], spec['front']
    # (a) ROI level and (b) full frame, oracle ROIs
    probs = predict(model, front, [it for it, _ in oracle], device, args.batch_size, args.amp)       # [K, c, c]
    full = np.zeros((H, W), dtype=np.float32)
    for k, ((item, gt_crop), p) in enumerate(zip(oracle, probs)):
        meta = item[4]
        record(st, 'roi_oracle', canvas_to_roi(p, meta['offset'], meta['s'], meta['roi_hw']), gt_crop, (name, k))
        full = paste_back(p, meta, full)
    record(st, 'full_oracle', full, gt, name)
    # (c) the arm's own detector ROIs, merged by maximum; no ROI = the empty mask
    items, match = det
    probs = predict(model, front, [it for it, _ in items], device, args.batch_size, args.amp)
    full = np.zeros((H, W), dtype=np.float32)
    for (item, _), p, obj in zip(items, probs, match):
        meta = item[4]
        full = paste_back(p, meta, full)
        if obj < 0:                                       # false positive: S/E undefined, only the false-mask rate
            st['metrics']['full_det'].update_fp(canvas_to_roi(p, meta['offset'], meta['s'], meta['roi_hw']), frame=name)
            st['n_fp_rois'] += 1
    record(st, 'full_det', full, gt, name)
    record(st, 'full_det_paper', full, gt, name)          # the same map, scored with SegMetrics(minmax=True)
    st['n_frames'] += 1; st['n_det_rois'] += len(items); st['n_frames_no_roi'] += int(len(items) == 0)


def to_json(o):
    '''json.dump default: numpy scalars and arrays.'''
    return o.tolist() if hasattr(o, 'tolist') else float(o)


def write_run(args, spec):
    '''The run's three eval_*.json files and per_image.pkl; keeps the summaries and paired rows on spec for compare().'''
    st, d = spec['state'], os.path.join(args.out_dir, spec['name'])
    os.makedirs(d, exist_ok=True)
    summ = {lvl: m.summary() for lvl, m in st['metrics'].items()}
    info = dict(run=spec['name'], seg_model=spec['model_label'], arm=spec['arm'], seed=spec['seed'], ckpt=args.ckpt if not spec['zero_shot'] else 'zero_shot',
                epoch=spec['epoch'], det_run=spec['det']['det_run'], roi_file=spec['det']['roi_file'], canvas=spec['canvas'])
    files = {'roi_oracle': dict(info, level='roi_oracle', roi_margin=args.roi_margin, roi_min=args.roi_min, summary=summ['roi_oracle']),
             'full_oracle': dict(info, level='full_oracle', summary=summ['full_oracle']),
             'full_det': dict(info, level='full_det', summary=summ['full_det'], paper={k: summ['full_det_paper'][k] for k in PAPER_KEYS},
                              n_frames=st['n_frames'], n_frames_no_roi=st['n_frames_no_roi'], n_det_rois=st['n_det_rois'], n_fp_rois=st['n_fp_rois'])}
    for lvl, obj in files.items():
        with open(os.path.join(d, f'eval_{lvl}.json'), 'w') as f:
            json.dump(obj, f, indent=1, default=to_json)
    spec['summary'] = summ
    spec['keys'] = st['keys']
    spec['per_image'] = {lvl: [st['metrics'][lvl].per_image[i] for i in st['rows'][lvl]] for lvl in st['metrics']}
    with open(os.path.join(d, 'per_image.pkl'), 'wb') as f:
        pickle.dump({lvl: dict(keys=spec['keys'][lvl], rows=spec['per_image'][lvl]) for lvl in spec['per_image']}, f)
    s, c, p = summ['roi_oracle'], summ['full_det'], files['full_det']['paper']
    print(f"{spec['name']}: roi_oracle S {s['S']:.3f} Fw {s['Fw']:.3f} IoU {s['IoU']:.3f} | full_det S {c['S']:.3f} Fw {c['Fw']:.3f} "
          f"IoU {c['IoU']:.3f}, false-mask rate {c['fp_false_mask_rate']:.3f} | paper MAE {p['MAE']:.4f} E {p['E_mean']:.3f} "
          f"S {p['S']:.3f} adpF {p['F_adp']:.3f} -> {d}", flush=True)


def seed_stats(summaries):
    '''mean, std (ddof 1; 0 for a single seed) and n over the seeds of one (model, arm) for every numeric summary key.'''
    out = {}
    for k, v in summaries[0].items():
        if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
            vals = np.asarray([float(s[k]) for s in summaries], dtype=np.float64)
            out[k] = dict(mean=float(np.mean(vals)), std=float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, n=len(vals))
    return out


def compare(args, specs):
    '''compare.json: seed statistics per (model, arm) and the paired bootstrap CIs (same model across arms, same arm across models).'''
    groups = {}
    for s in specs:
        groups.setdefault((s['model_label'], s['arm']), []).append(s)
    for g in groups.values():
        g.sort(key=lambda s: s['seed'])
    # pairing: every run scored the same images in the same order (oracle items do not depend on the arm; frames are shared)
    for lvl in LEVELS:
        ref = specs[0]['keys'][lvl]
        for s in specs[1:]:
            assert s['keys'][lvl] == ref, f"{lvl}: {s['name']} scored other images than {specs[0]['name']}; cannot pair them"
    stats = {f"{m}|{a}": dict(runs=[s['name'] for s in g], seeds=[s['seed'] for s in g],
                              **{lvl: seed_stats([s['summary'][lvl] for s in g]) for lvl in LEVELS + ['full_det_paper']})
             for (m, a), g in groups.items()}
    models = sorted({m for m, _ in groups})
    arms = [a for a in ARM_ORDER if any(a == ga for _, ga in groups)]
    pairs = []
    for m in models:
        present = [a for a in arms if (m, a) in groups]
        pairs += [(f"{m}: {a1} - {a2}", (m, a1), (m, a2)) for a1, a2 in itertools.combinations(present, 2)]
    for a in arms:
        present = [m for m in models if (m, a) in groups]
        pairs += [(f"{a}: {m1} - {m2}", (m1, a), (m2, a)) for m1, m2 in itertools.combinations(present, 2)]
    comps = {lvl: {} for lvl in LEVELS}
    for lvl in LEVELS:
        for label, ga, gb in pairs:
            comps[lvl][label] = bootstrap_seg([s['per_image'][lvl] for s in groups[ga]], [s['per_image'][lvl] for s in groups[gb]],
                                              BOOT_KEYS, n=args.n_boot, seed=args.seed)
    out = dict(ckpt=args.ckpt, n_boot=args.n_boot, keys=BOOT_KEYS, paper_keys=PAPER_KEYS,
               runs={s['name']: dict(seg_model=s['model_label'], arm=s['arm'], seed=s['seed'], epoch=s['epoch'], det_run=s['det']['det_run'])
                     for s in specs},
               groups=stats, comparisons=comps)
    path = os.path.join(args.out_dir, 'compare.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=1, default=to_json)
    # print: mean +- std over seeds, then the CIs
    show = ['S', 'E_mean', 'Fw', 'MAE', 'IoU']
    for lvl in LEVELS:
        print(f"\n== {lvl}, mean +- std over seeds: " + ' | '.join(show))
        for g, st in stats.items():
            print(f"{g:24s} n={len(st['runs'])} " + ' '.join(f"{st[lvl][k]['mean']:.3f}+-{st[lvl][k]['std']:.3f}" for k in show))
        print(f"== {lvl}, paired frame bootstrap ({args.n_boot} resamples), diff [95% CI]")
        for label, v in comps[lvl].items():
            print(f"{label:32s} " + '; '.join(f"{k} {v[k]['diff']:+.3f} [{v[k]['lo']:+.3f}, {v[k]['hi']:+.3f}]" for k in ('S', 'Fw', 'IoU')))
    print(f"\nwrote {path}")
    return out


@torch.no_grad()
def main(args):
    device = torch.device(args.device if args.device == 'cpu' or torch.cuda.is_available() else 'cpu')
    assert args.runs or args.zero_shot, "nothing to evaluate: pass --runs and/or --zero_shot"
    assert len(set(args.runs)) == len(args.runs), f"duplicate run names in {args.runs}"
    os.makedirs(args.out_dir, exist_ok=True)

    # the runs (args only; the weights are loaded per pass) and their detectors
    det_cache = {}
    specs = [load_run(args, run, det_cache) for run in args.runs]
    canvases = {s['canvas'] for s in specs}
    assert len(canvases) <= 1, f"runs with different canvases {sorted(canvases)}: the oracle items are shared, evaluate them separately"
    canvas = canvases.pop() if canvases else args.canvas
    if args.zero_shot:
        assert ZS_NAME not in args.runs, f"{ZS_NAME} is the zero-shot control's name"
        specs.append(zero_shot_spec(args, canvas, det_cache))
    for s in specs:
        print(f"{s['name']}: {s['model_label']}, arm {s['arm']}, seed {s['seed']}, epoch {s['epoch']}, detector {s['det']['det_run']} "
              f"({s['det']['roi_file']})", flush=True)

    # the test frames, read through HyperCOD_data exactly as the detectors' ROI export read them
    dataset = specs[0]['det']['dataset']
    if args.limit:
        dataset = Dataset.Subset(dataset, list(range(min(args.limit, len(dataset)))))
    fronts = {}
    for p0 in range(0, len(specs), args.runs_per_pass):
        group = specs[p0:p0 + args.runs_per_pass]
        for s in group:
            key = (s['arm'], s['det']['key'])
            if key not in fronts:                                     # one front end per (arm, detector), shared by its runs
                # spec §9: raw / ec10 / ec24 front ends are checked against the detector checkpoint's filter bank
                det_state = None if s['arm'] == 'rgb' else s['det']['det_state']
                fronts[key] = build_front_end(s['arm'], s['det']['det_args'], s['det']['dataset'], det_state=det_state).to(device).eval()
            s['front'] = fronts[key]
            s['model'] = build_run_model(s, s['front'], device)
            s['state'] = new_state(tuple(args.size_edges))
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
                                             collate_fn=frame_collate_fn)
        t0 = time.time()
        for i, (img, gt, name) in enumerate(loader):
            img, gt = img.numpy(), gt.numpy()                         # [C, H, W] fp16 p99-scaled, [H, W] bool
            H, W = gt.shape
            gt_boxes, labels, ids = boxes_from_mask(gt, min_area=args.min_area, return_labels=True)
            oracle = [make_item(img, gt, expand_box(b, args.roi_margin, args.roi_min, H, W), b, canvas, name, 'gt') for b in gt_boxes]
            det_items = {}                                            # one item list per detector, shared by its runs
            for s in group:
                key = s['det']['key']
                if key in det_items:
                    continue
                entry = s['det']['rois'][name]
                rois = np.asarray(entry['rois'], dtype=np.float32).reshape(-1, 5)     # [R, 5] grown ROI + conf
                boxes = np.asarray(entry['boxes'], dtype=np.float32).reshape(-1, 5)   # [R, 5] pre-expansion box + conf
                assert len(rois) == len(boxes), f"{s['det']['roi_file']} frame {name}: {len(rois)} rois but {len(boxes)} boxes"
                match = match_rois(rois[:, :4], labels, ids, min_cover=args.min_cover) if len(rois) else np.zeros(0, dtype=int)
                det_items[key] = ([make_item(img, gt, r[:4], b[:4], canvas, name, 'det' if m >= 0 else 'fp')
                                   for r, b, m in zip(rois, boxes, match)], match)
            for s in group:
                evaluate_frame(s, oracle, det_items[s['det']['key']], gt, name, device, args)
            if i % 10 == 0 or i == len(loader) - 1:
                print(f"pass {p0 // args.runs_per_pass + 1}: {i + 1}/{len(loader)} frames, {time.time() - t0:.0f}s", flush=True)
        for s in group:
            write_run(args, s)
            del s['model'], s['front'], s['state']
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return compare(args, specs)


if __name__ == '__main__':
    main(get_args_parser().parse_args())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python -m pytest tests/test_main_seg_eval.py -v`

Expected: PASS (5 passed in about 5 s). The main test prints `r_s0: roi_oracle S 1.000 Fw 1.000 IoU 1.000 | full_det S 0.638 Fw 0.251 IoU 0.250, false-mask rate 1.000 | paper MAE 0.0187 ...`. The downscaled test reaches an IoU of about 0.73 against an assertion of > 0.6.

- [ ] **Step 5: Review Focus test: train/test parity of the ROI item (write it, run it)**

Append to `tests/test_main_seg_eval.py`:

```python


from tests.test_roi_crops import _crop_cache
from data_loader.roi_crops import HyperCOD_roi
from data_loader.cube_cache import default_cache_dir
from data_loader.my_dataset import HyperCOD_data


@pytest.mark.parametrize('canvas', [32, 12])
def test_test_items_match_the_validation_items(synthetic_root, tmp_path, canvas):
    '''
    main_seg_eval cuts its ROIs from the full p99-scaled frame, HyperCOD_roi from the un-scaled crop cache. The same ROI
    must give the same tensors (canvas 32: s = 1; canvas 12: the 16 px detector ROI is downscaled), so the model is tested
    on exactly the input it was selected on.
    '''
    root, out, _, _, _ = _crop_cache(synthetic_root, tmp_path)
    ds = HyperCOD_roi(str(out), 'val', 'raw', train=False, roi_margin=1.5, roi_min=0, canvas=canvas)
    ref = HyperCOD_data(str(root), split='train', ids=['10'], use_filter=False, norm='p99', crop_size=0, filter_norm='none',
                        cache_dir=default_cache_dir(str(root)), out_dtype='float16')
    frame, gt, _ = ref[0]                                                      # [133, H, W] fp16 p99-scaled, [1, H, W]
    scales = []
    for i in range(len(ds)):
        img, mask, box_map, valid, meta = ds[i]
        (img2, mask2, box_map2, valid2, meta2), gt_crop = main_seg_eval.make_item(frame, gt[0] > 0.5, meta['roi'], meta['box'],
                                                                                 canvas, '10', meta['source'])
        assert tuple(meta2['offset']) == tuple(meta['offset']) and meta2['s'] == meta['s']
        assert tuple(meta2['roi_hw']) == tuple(meta['roi_hw'])
        assert [float(v) for v in meta2['roi']] == [float(v) for v in meta['roi']]
        np.testing.assert_array_equal(img2, img)                                # same fp16 p99 scaling and resize
        np.testing.assert_array_equal(mask2, mask)
        np.testing.assert_array_equal(valid2, valid)
        np.testing.assert_array_equal(box_map2, box_map)
        np.testing.assert_allclose(meta2['box_canvas'], meta['box_canvas'], atol=1e-6)
        scales.append(meta['s'])
    assert len(ds) == 2
    if canvas == 32:
        assert scales == [1.0, 1.0]
    else:
        assert scales[0] == 1.0 and scales[1] < 1.0                           # oracle 10 px fits, detector 16 px is downscaled
```

Run: `OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python -m pytest tests/test_main_seg_eval.py -k match_the_validation -v`
Expected: PASS (2 passed) with the implementation of Step 3: `make_item` uses `pixel_box`, `place_on_canvas`, `box_to_canvas` and `rasterise_box` exactly as `HyperCOD_roi` does. A failure means the test-time geometry drifted from the training geometry; fix `make_item`, not the test.

- [ ] **Step 6: Run the file and the full suite**

Run: `OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python -m pytest tests/test_main_seg_eval.py -v`
Expected: PASS (7 passed).
Then: `OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python -m pytest tests/ -q`. It must stay green.

- [ ] **Step 7: Commit and push**

```bash
git add main_seg_eval.py tests/test_main_seg_eval.py
git commit -m "feat(seg): main_seg_eval - oracle/detector-ROI test evaluation, paste-back, seed stats and paired frame bootstrap

Spec §8 levels (a)-(c) for every segmentation run plus the zero-shot SAM2.1 control, one frame pass per group of runs,
test items with the crop cache's geometry, front ends checked against their detectors, the HyperCOD paper metric set on
(c) via SegMetrics(minmax=True), compare.json with arm-vs-arm and model-vs-model CIs.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QB4DScDwiA2DkVpbDopffV"
git push origin worktree-det
git push origin worktree-det:master
```

---

### Task 14: Stage-2 queue (`bash_files/launch_seg_queue.sh`), README row and pre-flight on real data

**Files:**
- Create: `bash_files/launch_seg_queue.sh`
- Modify: `bash_files/README.md` (one table row after the `launch_sel_clean_queue.sh` row and after the `launch_rois_all.sh` / `setup_third_party.sh` rows of Tasks 2 and 7; the header line was already changed by Task 7)
- Test: `tests/test_launch_seg_queue.py`

**Interfaces:**
- Consumes:
  - `data_loader.roi_crops.build_crop_cache(data_path, out_dir, roi_files, splits, grow, min_side, num_workers) -> {"n_windows", "bytes", ...}` (T3). It writes `index.json` last.
  - `main_seg.py` flags (T12): `--seg_model`, `--arm`, `--det_ckpt`, `--data_path`, `--crop_cache`, `--seed`, `--batch_size`, `--accumulate`, `--num_workers`, `--name`, `--output_dir`, `--dry_run`, `--epochs`, `--no-wandb`. Its results file is `weights/<name>/results_<name>.txt` with a `"final": true` line.
  - `main_seg_eval.py` (Task 13): `--runs`, `--zero_shot`, `--ckpt`, `--data_path`, `--out_dir`, `--limit`, `--n_boot`.
  - `main_det_rois.py` exports `results/det/rois_<det>_{train,val,test}.json` (T2, via `bash_files/launch_rois_all.sh`).
  - `bash_files/setup_third_party.sh` (T7).
- Produces:
  - `bash bash_files/launch_seg_queue.sh`, with env knobs `MODELS`, `ARMS`, `SEEDS`, `CONTROL_SEEDS`, `DATA_PATH`, `CROP_CACHE`, `NUM_WORKERS`, `PY`, `CUDA_VISIBLE_DEVICES`, `EVAL_LAST`, `CACHE_ONLY` and `LIST_ONLY`.
  - Logs: `logs/seg_queue.log`, `logs/seg_crop_cache.log`, `logs/<run>.log`, `logs/seg_eval.log`.
  - Results: `results/seg/compare.json`.

- [ ] **Step 1: Write the failing test**

`tests/test_launch_seg_queue.py`:
```python
import os
import subprocess

SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'bash_files', 'launch_seg_queue.sh')


def _plan(**env):
    '''The queue's planned run names (LIST_ONLY=1: printed, nothing detached, launched or built).'''
    out = subprocess.run(['bash', SCRIPT], env={**os.environ, 'LIST_ONLY': '1', **env}, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    lines = [l.split() for l in out.stdout.splitlines() if l.strip()]
    assert all(len(l) == 2 and l[1] in ('todo', 'done') for l in lines), out.stdout
    return [l[0] for l in lines]


def test_queue_script_parses():
    assert subprocess.run(['bash', '-n', SCRIPT]).returncode == 0


def test_queue_plan_is_seed_major_with_the_rgb_control():
    names = _plan()
    assert len(names) == 28 and len(set(names)) == 28                       # 3 models x 3 arms x 3 seeds + the control
    assert names[:9] == [f'seg_{m}_{a}_s0' for m in ('sam2unet', 'sam2box', 'zoomnext') for a in ('raw', 'ec10', 'ec24')]
    assert names[-1] == 'seg_sam2unet_rgb_s0'
    assert _plan(MODELS='zoomnext', ARMS='ec10', SEEDS='1', CONTROL_SEEDS='') == ['seg_zoomnext_ec10_s1']
    assert _plan(SEEDS='0', CONTROL_SEEDS='0 1')[-2:] == ['seg_sam2unet_rgb_s0', 'seg_sam2unet_rgb_s1']
```

- [ ] **Step 2: Run test to verify it fails**

Run: `OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python -m pytest tests/test_launch_seg_queue.py -v`

Expected: FAIL. `bash -n` returns 127 (`No such file or directory`), and `_plan` fails its `returncode == 0` assertion.

- [ ] **Step 3: Write minimal implementation**

`bash_files/launch_seg_queue.sh`:
```bash
#!/usr/bin/env bash
# Stage-2 queue (spec §7, §8): the crop cache once, then every (model, arm, seed) segmentation run in sequence, then the
# end-to-end test evaluation of everything (main_seg_eval.py: oracle ROIs, paste-back, each arm's own detector ROIs).
#   models sam2unet sam2box zoomnext x arms raw ec10 ec24 x seeds 0 1 2 = 27 runs seg_<model>_<arm>_s<seed>, seed-major (the
#   9 (model, arm) pairs of seed 0 first, so a complete first comparison exists after 9 runs); then the control
#   seg_sam2unet_rgb_s<seed> (pseudo-RGB input, raw133_A's boxes; CONTROL_SEEDS, default 0); the zero-shot SAM2.1 box
#   control needs no training and is evaluated by main_seg_eval --zero_shot.
#   bash bash_files/launch_seg_queue.sh    # detached; queue log logs/seg_queue.log, run logs logs/<run>.log,
#                                          # crop cache logs/seg_crop_cache.log, evaluation logs/seg_eval.log
# Prerequisites: bash_files/setup_third_party.sh (sam2, pysodmetrics, ZoomNeXt, weights/pretrained/*) and
# bash_files/launch_rois_all.sh (results/det/rois_<det>_{train,val,test}.json of raw133_A, sel10g_clean_A, sel24g_clean_A).
# The crop cache is built only when <CROP_CACHE>/index.json is missing. A run with a 'final' line in
# weights/<run>/results_<run>.txt is skipped, so the queue can be relaunched after a crash (an interrupted run restarts
# from epoch 0). Env overrides: MODELS, ARMS, SEEDS, CONTROL_SEEDS (empty = no control), DATA_PATH, CROP_CACHE (default
# <DATA_PATH>/crop_cache_seg), NUM_WORKERS (6), PY, CUDA_VISIBLE_DEVICES (0), EVAL_LAST=1 (also evaluate model_last into
# results/seg_last), CACHE_ONLY=1 (build the crop cache, then stop), LIST_ONLY=1 (print the planned runs, todo/done, and exit
# without detaching). One full-resolution job per box: do not start another training meanwhile.
SELF=$(readlink -f "$0")
cd "$(dirname "$SELF")/.." || exit 1
PY=${PY:-/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
DATA_PATH=${DATA_PATH:-/data2/chaoyi/HyperCOD/Raw data}
CROP_CACHE=${CROP_CACHE:-$DATA_PATH/crop_cache_seg}
MODELS=${MODELS:-sam2unet sam2box zoomnext}
ARMS=${ARMS:-raw ec10 ec24}
SEEDS=${SEEDS:-0 1 2}
CONTROL_SEEDS=${CONTROL_SEEDS-0}
NUM_WORKERS=${NUM_WORKERS:-6}

# the arm's detector (spec §2): its front end, its training/val boxes in the crop cache and its test ROIs
det_of() { case $1 in raw|rgb) echo raw133_A ;; ec10) echo sel10g_clean_A ;; ec24) echo sel24g_clean_A ;; *) echo "unknown arm $1" >&2; return 1 ;; esac; }
finished() { [ -f "weights/$1/results_$1.txt" ] && grep -q '"final": true' "weights/$1/results_$1.txt"; }

# the plan, one "name model arm seed" line per run, seed-major
PLAN=()
for s in $SEEDS; do for m in $MODELS; do for a in $ARMS; do PLAN+=("seg_${m}_${a}_s${s} $m $a $s"); done; done; done
for s in $CONTROL_SEEDS; do PLAN+=("seg_sam2unet_rgb_s${s} sam2unet rgb $s"); done

if [ -n "${LIST_ONLY:-}" ]; then
  for p in "${PLAN[@]}"; do
    read -r name _ <<< "$p"
    if finished "$name"; then echo "$name done"; else echo "$name todo"; fi
  done
  exit 0
fi

mkdir -p logs
if [ -z "${SEG_QUEUE_CHILD:-}" ]; then
  SEG_QUEUE_CHILD=1 setsid nohup bash "$SELF" "$@" > logs/seg_queue.log 2>&1 < /dev/null &
  echo "seg queue launched, pid $!, log logs/seg_queue.log"
  exit 0
fi

# prerequisites: the three detectors and their ROI exports for every split
missing=0
for det in raw133_A sel10g_clean_A sel24g_clean_A; do
  [ -f "weights/$det/model_best" ] || { echo "$(date) missing weights/$det/model_best"; missing=1; }
  for split in train val test; do
    [ -f "results/det/rois_${det}_${split}.json" ] || { echo "$(date) missing results/det/rois_${det}_${split}.json: run bash_files/launch_rois_all.sh"; missing=1; }
  done
done
[ "$missing" = 0 ] || exit 1

# the crop cache, once: every train/val window of every arm's boxes (data_loader.roi_crops.build_crop_cache)
if [ ! -f "$CROP_CACHE/index.json" ]; then
  echo "$(date) building the crop cache in $CROP_CACHE (log logs/seg_crop_cache.log)"
  "$PY" -u - "$DATA_PATH" "$CROP_CACHE" "$NUM_WORKERS" > logs/seg_crop_cache.log 2>&1 <<'EOF'
import sys
from data_loader.roi_crops import build_crop_cache
data_path, out_dir, num_workers = sys.argv[1], sys.argv[2], int(sys.argv[3])
dets = {'raw': 'raw133_A', 'ec10': 'sel10g_clean_A', 'ec24': 'sel24g_clean_A'}
roi_files = {arm: {split: f'results/det/rois_{det}_{split}.json' for split in ('train', 'val')} for arm, det in dets.items()}
info = build_crop_cache(data_path, out_dir, roi_files, splits=('train', 'val'), grow=2.0, min_side=512, num_workers=num_workers)
print(f"crop cache: {info['n_windows']} windows, {info['bytes'] / 1e9:.1f} GB -> {out_dir}")
EOF
  rc=$?
  if [ "$rc" != 0 ] || [ ! -f "$CROP_CACHE/index.json" ]; then echo "$(date) crop cache failed, exit $rc"; exit 1; fi
  echo "$(date) $(tail -n 1 logs/seg_crop_cache.log)"
fi
if [ -n "${CACHE_ONLY:-}" ]; then echo "$(date) CACHE_ONLY: done"; exit 0; fi

train() {   # train <name> <model> <arm> <seed>
  local name=$1 model=$2 arm=$3 seed=$4 det rc bs=8 acc=1
  if finished "$name"; then echo "$(date) $name already finished, skipping"; return; fi
  det=$(det_of "$arm") || return
  [ "$model" = zoomnext ] && { bs=4; acc=2; }      # spec §7: ZoomNeXt's three-scale encoder, batch 4 x 2
  mkdir -p "weights/$name"
  echo "$(date) start $name: $model, arm $arm, detector $det, seed $seed"
  "$PY" -u main_seg.py --seg_model "$model" --arm "$arm" --det_ckpt "weights/$det/model_best" --data_path "$DATA_PATH" \
      --crop_cache "$CROP_CACHE" --seed "$seed" --batch_size "$bs" --accumulate "$acc" --num_workers "$NUM_WORKERS" \
      --name "$name" --output_dir "weights/$name" > "logs/$name.log" 2>&1 < /dev/null
  rc=$?
  echo "$(date) end $name, exit $rc"
}

for p in "${PLAN[@]}"; do
  read -r name model arm seed <<< "$p"
  train "$name" "$model" "$arm" "$seed"
done

# the end-to-end evaluation of every finished run + the zero-shot control (one frame pass per main_seg_eval --runs_per_pass runs)
DONE=()
for p in "${PLAN[@]}"; do
  read -r name _ <<< "$p"
  finished "$name" && DONE+=("$name")
done
echo "$(date) evaluation of ${#DONE[@]}/${#PLAN[@]} finished runs + the zero-shot control (log logs/seg_eval.log)"
"$PY" -u main_seg_eval.py --runs "${DONE[@]}" --zero_shot --data_path "$DATA_PATH" --out_dir results/seg > logs/seg_eval.log 2>&1
rc=$?
echo "$(date) evaluation exit $rc -> results/seg/compare.json"
if [ -n "${EVAL_LAST:-}" ]; then
  "$PY" -u main_seg_eval.py --runs "${DONE[@]}" --ckpt model_last --data_path "$DATA_PATH" --out_dir results/seg_last > logs/seg_eval_last.log 2>&1
  echo "$(date) model_last evaluation exit $? -> results/seg_last/compare.json"
fi
echo "$(date) queue done"
```

Edit `bash_files/README.md`: insert this row after the `| \`launch_sel_clean_queue.sh\` | ... |` row and after the `setup_third_party.sh` / `launch_rois_all.sh` rows that Tasks 7 and 2 added (the header line already reads "Stages 1 and 2" since Task 7):
```markdown
| `launch_seg_queue.sh` | Stage 2: crop cache once (if `index.json` is missing), then the 27 `seg_<model>_<arm>_s<seed>` runs (sam2unet / sam2box / zoomnext x raw / ec10 / ec24 x seeds 0-2, seed-major) and the `seg_sam2unet_rgb` control, then `main_seg_eval.py` over every finished run + the zero-shot SAM2.1 control -> `results/seg/compare.json`; detached, log `logs/seg_queue.log`; `LIST_ONLY=1` prints the plan, `CACHE_ONLY=1` stops after the cache |
```

- [ ] **Step 4: Run test to verify it passes**

Run: `OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python -m pytest tests/test_launch_seg_queue.py -v`

Expected: PASS (2 passed). `LIST_ONLY` exits before the detach, the prerequisite checks and any build, so the test never launches anything.

Then run the full suite: `OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python -m pytest tests/ -q`. It should stay green.

- [ ] **Step 5: Commit and push**

```bash
git add bash_files/launch_seg_queue.sh bash_files/README.md tests/test_launch_seg_queue.py
git commit -m "feat(seg): launch_seg_queue.sh - crop cache, 27 seg runs + rgb control, then main_seg_eval

One detached, relaunchable queue for spec §7: builds the crop cache once, trains seed-major, skips finished runs and
evaluates everything (plus the zero-shot SAM2.1 control) into results/seg/compare.json.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QB4DScDwiA2DkVpbDopffV"
git push origin worktree-det
git push origin worktree-det:master
```

- [ ] **Step 6: Pre-flight 1. Dependencies (after T7)**

Run from the worktree:
```bash
/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python -c "import sam2, py_sod_metrics, einops, numpy, torch; print(numpy.__version__, torch.__version__)"
ls -la weights/pretrained/
```
Expected:
- The first command prints `2.5.2 2.11.0+cu128`. numpy must not have been downgraded.
- `weights/pretrained/` lists `sam2_hiera_large.pt` (897,952,466 B), `sam2.1_hiera_large.pt` (898,083,611 B) and `pvtv2-b2-zoomnext.pth` (113,020,474 B).

- [ ] **Step 7: Pre-flight 2. ROI exports (after T2)**

Run `bash bash_files/launch_rois_all.sh` and wait for it to finish (see its log). Then check the files:
```bash
/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python -c "import json; [print(d, s, len(json.load(open(f'results/det/rois_{d}_{s}.json')))) for d in ('raw133_A', 'sel10g_clean_A', 'sel24g_clean_A') for s in ('train', 'val', 'test')]"
```
Expected: 9 lines, with frame counts `train 251`, `val 28` and `test 70` for each detector. A different count means the split mapping is wrong; stop and fix T2.

- [ ] **Step 8: Pre-flight 3. Crop cache build and size**

```bash
CACHE_ONLY=1 bash bash_files/launch_seg_queue.sh
tail -f logs/seg_queue.log            # until "CACHE_ONLY: done"
tail -n 1 logs/seg_crop_cache.log
du -sh "/data2/chaoyi/HyperCOD/Raw data/crop_cache_seg"; free -g
```
Expected:
- The cache log ends with `crop cache: <N> windows, <X> GB -> .../crop_cache_seg`. X should be about 25–50 GB (spec §3; 22 GB from the GT boxes alone).
- `du -sh` agrees with X.
- `free -g` shows `available`. If X is larger than available minus about 10 GB, the cache will not stay in the page cache: note it, and keep one job per box (spec §7).

The build reads 279 frames × 0.55 GB once with O_DIRECT, about 10 min.

- [ ] **Step 9: Pre-flight 4. Dry runs and peak memory, one per model, on the widest arm**

The raw arm has 133 + 1 input channels. Each dry run must print the peak GPU memory:
```bash
CC="/data2/chaoyi/HyperCOD/Raw data/crop_cache_seg"
/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python -u main_seg.py --seg_model sam2unet --arm raw --det_ckpt weights/raw133_A/model_best --crop_cache "$CC" --batch_size 8 --accumulate 1 --dry_run --no-wandb --name dry_sam2unet_raw --output_dir /tmp/dry_seg
/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python -u main_seg.py --seg_model sam2box --arm raw --det_ckpt weights/raw133_A/model_best --crop_cache "$CC" --batch_size 8 --accumulate 1 --dry_run --no-wandb --name dry_sam2box_raw --output_dir /tmp/dry_seg
/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python -u main_seg.py --seg_model zoomnext --arm raw --det_ckpt weights/raw133_A/model_best --crop_cache "$CC" --batch_size 4 --accumulate 2 --dry_run --no-wandb --name dry_zoomnext_raw --output_dir /tmp/dry_seg
/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python -u main_seg.py --seg_model sam2unet --arm ec24 --det_ckpt weights/sel24g_clean_A/model_best --crop_cache "$CC" --batch_size 8 --accumulate 1 --dry_run --no-wandb --name dry_sam2unet_ec24 --output_dir /tmp/dry_seg
```
Expected for each command:
- Exit 0 (the front end passes the filter-bank check against the detector checkpoint, and the crop cache's ROI files match the arm's detector).
- One batch from each box source (gt / det / fp).
- Input shape `[B, N+1, 512, 512]` with N = 133 (raw) or 24 (ec24).
- Peak GPU memory below about 19 GiB. The spec §2 probes: SAM2-UNet b8 12.0 GiB; SAM2.1-L b8 10.4 GiB; ZoomNeXt b2 6.2 GB, so b4 is roughly double.

If ZoomNeXt is over budget, use its `use_checkpoint` fallback (`grad_ckpt: true` in the zoomnext section of `cfg/seg.yaml`) or batch 2 × accumulate 4, and record the change in spec §7.

- [ ] **Step 10: Pre-flight 5. Zero-shot sanity of the evaluation path on 10 test frames**

```bash
/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python -u main_seg_eval.py --zero_shot --limit 10 --n_boot 10 --out_dir results/seg_preflight
cat results/seg_preflight/zs_sam2box_rgb/eval_roi_oracle.json
```
Expected: four files under `results/seg_preflight/zs_sam2box_rgb/`. The oracle-ROI S should be clearly above 0.5; box-prompted SAM2-L reaches S .911 on COD10K. S ≈ 0.5 or IoU ≈ 0 means the canvas geometry or the box prompt is wrong (canvas px, x first); stop and debug before training. Also record `full_det` `n_frames_no_roi` for these 10 frames.

- [ ] **Step 11: Pre-flight 6. One-epoch smoke run through training and evaluation**

```bash
/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python -u main_seg.py --seg_model sam2unet --arm ec10 --det_ckpt weights/sel10g_clean_A/model_best --crop_cache "/data2/chaoyi/HyperCOD/Raw data/crop_cache_seg" --epochs 1 --no-wandb --name smoke_seg --output_dir weights/smoke_seg
/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python -u main_seg_eval.py --runs smoke_seg --limit 3 --n_boot 10 --out_dir results/seg_smoke
```
Expected:
- `weights/smoke_seg/results_smoke_seg.txt` ends with a `"final": true` line, and `model_best` / `model_last` exist.
- `main_seg_eval` loads `model_best` with no unexpected or missing non-frozen keys, and writes `results/seg_smoke/smoke_seg/*.json`.

Note the epoch time: it times the 200-epoch budget (spec §7, about 1 h per run). Afterwards, `weights/smoke_seg` and `results/seg_smoke` can be deleted; both are git-ignored.

- [ ] **Step 12: Launch the queue**

```bash
LIST_ONLY=1 bash bash_files/launch_seg_queue.sh      # 28 lines, all "todo" (the smoke run is not in the plan)
bash bash_files/launch_seg_queue.sh
tail -f logs/seg_queue.log
```
Expected:
- `seg queue launched, pid <n>, log logs/seg_queue.log`.
- The queue log shows `start seg_sam2unet_raw_s0 ...`. If the cache from Step 8 exists, it is not rebuilt.
- After the last run, `evaluation exit 0 -> results/seg/compare.json`.

The evaluation of 29 runs makes 5 frame passes (about 3 min of reads each). It also spends about 1.3 s of metric CPU time per frame per run, so expect about 1 h. Record the outcomes in the Results section below.

---

## Results

Pre-flight outcomes of Task 14, run on 2026-10-05 from the worktree on GPU 0 (2x RTX A4500 20 GB), one job at a time. The queue itself (Step 12) is not launched yet. Pre-flights 1-5 passed; pre-flight 6 (smoke run) is BLOCKED by a bug in `train_eval/train_eval_seg.py` (below), so the epoch time and the 200-epoch projection are not measured yet.

**Pre-flight 1, dependencies.** `import sam2, py_sod_metrics, einops, numpy, torch` prints `2.5.2 2.11.0+cu128`. `weights/pretrained/` holds `sam2_hiera_large.pt` (897,952,466 B), `sam2.1_hiera_large.pt` (898,083,611 B), `pvtv2-b2-zoomnext.pth` (113,020,474 B) and the Stage-1 `yolo26s.pt`.

**Pre-flight 2, ROI exports** (`bash bash_files/launch_rois_all.sh`, 07:14-07:32 CDT, 17.7 min, all exports exit 0). Frame counts as expected (251 / 28 / 70) for every detector. The coverage printed by each export is `ROI coverage recall@0.99` (the fraction of GT objects whose box is covered by some exported ROI at 0.99):

| detector (arm) | train (251 frames, 258 objects) | val (28 frames, 31 objects) | test (70 frames, 71 objects) |
|---|---|---|---|
| raw133_A (raw) | 0.872 | 0.839 | 0.775 |
| sel10g_clean_A (ec10) | 0.926 | 0.806 | 0.817 |
| sel24g_clean_A (ec24) | 0.845 | 0.806 | 0.746 |

**Pre-flight 3, crop cache** (`CACHE_ONLY=1 bash bash_files/launch_seg_queue.sh`, 07:32:46-07:37:44 CDT, 5.0 min). `crop cache: 521 windows, 31.4 GB -> /data2/chaoyi/HyperCOD/Raw data/crop_cache_seg` (289 object windows + 232 false-positive windows; `du -sh` 30G = 29.2 GiB, which agrees). In the 25-50 GB band of spec §3. `free -g` after the build: total 125, used 5, free 15, buff/cache 105, **available 119** GB. The cache (31.4 GB) fits in the page cache with a wide margin (119 - 10 > 31.4), so the one-job-per-box rule of spec §7 is enough.

**Pre-flight 4, dry runs** (flags `--batch_size` / `--accumulate` omitted: the values come from `cfg/seg.yaml models.*`; each exits 0, filter bank checked against the detector checkpoint, one batch from each of gt / det / fp, GPU 0 was otherwise idle):

| run | input shape of each source (gt / det / fp) | batch | peak GPU memory | wall time |
|---|---|---|---|---|
| sam2unet / raw | (8, 134, 512, 512) | 8 | 11.97 GiB | 45 s |
| sam2box / raw | (8, 134, 512, 512) | 8 | 10.90 GiB | 39 s |
| zoomnext / raw | (4, 134, 512, 512) | 4 (x accumulate 2) | 10.98 GiB | 47 s |
| sam2unet / ec24 | (8, 25, 512, 512) | 8 | 10.69 GiB | 28 s |

All are below the 19 GiB budget, so no ZoomNeXt fallback was needed (`cfg/seg.yaml models.zoomnext` is unchanged: batch 4 x accumulate 2, no `grad_ckpt`). Items per epoch: 287 train items on every arm; val items 73 (raw), 67 (ec24), 69 (ec10). Logs `logs/preflight_dry_*.log`.

**Pre-flight 5, zero-shot sanity** (`main_seg_eval.py --zero_shot --limit 10 --n_boot 10 --out_dir results/seg_preflight`, 25 s in total, 14 s for the 10-frame pass). Four files under `results/seg_preflight/zs_sam2box_rgb/` (`eval_roi_oracle.json`, `eval_full_oracle.json`, `eval_full_det.json`, `per_image.pkl`). Oracle-ROI (n = 11 objects): S 0.840, Fw 0.754, IoU 0.715 (medium bucket S 0.868, n 7; large bucket S 0.790, n 4). Full-frame oracle: S 0.924, IoU 0.784. Full-frame detector ROIs (arm rgb = raw133_A's test ROIs): S 0.851, Fw 0.683, IoU 0.647, `n_frames = 10`, **`n_frames_no_roi = 1`**, `n_det_rois = 12`, `n_fp_rois = 1`, false-mask rate 1.000 (one fp ROI, so not informative). Far from S 0.5 / IoU 0: the canvas geometry and the box prompt are right.

**Pre-flight 6, smoke run: BLOCKED.** `main_seg.py --seg_model sam2unet --arm ec10 ... --epochs 1 --name smoke_seg --output_dir weights/smoke_seg` builds everything (front end, 287 train / 69 val items, stem fold R^2 [0.9271, 0.9566, 0.9913], 10 + 1 channel SAM2UNetSeg, 4.35M trainable) and then dies in the first training step after 25 s:
```
File "train_eval/train_eval_seg.py", line 89, in train_one_epoch
    metric_logger.update(loss=float(total.detach()), lr=lr, **{k: float(v) for k, v in items.items()})
TypeError: util.misc.MetricLogger.update() got multiple values for keyword argument 'loss'
```
Cause: `SAM2UNetSeg.loss` (`models/seg_models.py:191`) returns `items['loss'] = float(total.detach())` next to `loss_main` / `loss_s16` / `loss_s8`, and `train_one_epoch` passes `loss=` explicitly as well as `**items`. SAM2BoxSeg and ZoomNeXtSeg items have no `loss` key, so only SAM2-UNet runs (10 of the 28 queue runs: 9 + the rgb control) crash. `tests/test_train_eval_seg.py` uses a stub model whose items have no `loss` key, and the dry run never calls `train_one_epoch`, so neither caught it. The same dict-merge collision in the `logger.scalars({'loss': ..., **items})` call is silent (the later key wins, equal value). Pending: the fix (for example merging the dicts, `metric_logger.update(**{'loss': ..., 'lr': lr, **items})`, plus a regression test with a SAM2UNetSeg-shaped items dict), then the smoke run again; the epoch time and the 200-epoch projections (per run and for the 28-run queue) are filled in from that run.
