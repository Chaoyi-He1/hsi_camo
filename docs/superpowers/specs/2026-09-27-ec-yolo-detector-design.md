# Stage 1 — EC-filter YOLO26 detector with learned channel selection — design

**Date:** 2026-09-27
**Status:** design approved in conversation (native resolution, fp16 cache, projection on GPU, N-channel first conv, `(B,N,H,W)×(1,N,1,1)` weight vector, Spec_Occu code layout, TensorBoard + W&B). Open item: W&B entity (see §10).
**Builds on:** `docs/superpowers/specs/2026-09-21-hypercod-dataloader-design.md` (loader, filter alignment, normalisation layers).

## 1. Goal

Reproduce the "top-10 filter responses" recipe of the previous HSI project on HyperCOD, for a detector that later zooms into candidate camouflage regions:

1. **Session A (selection):** feed a pretrained YOLO26 *all* usable EC filter responses of the full frame, multiplied channel-wise by a trainable weight vector `w` (`(B, N, H, W) * (1, N, 1, 1)`), fine-tune it as a single-class box detector on boxes derived from the camouflage masks, then rank channels by `w` and keep the **top 10** voltages.
2. **Session B (fixed channels):** rebuild the model with only the 10 selected responses, no weight vector, initialised from A, and fine-tune again → the detector used to propose regions for Stage 2 (segmentation, separate spec, same recipe).

"Filter response" = the raw HSI cube integrated against the filter matrix, `y_n = Σ_b R[b, n] · cube[b]`, exactly as in the dataloader.

## 2. Facts the design relies on (verified)

| Item | Value |
|---|---|
| Frames | 279 train / 70 test, 1680 × 1240, 200 bands 400–1000 nm (3.015 nm); after step 0 the loader uses **133 bands, 400–800 nm** |
| Objects | 349 masks → 360 boxes: 339 frames with 1 component ≥ 100 px, 9 with 2, 1 with 3; 10 sub-100-px specks (JPEG ringing) dropped. Box height p5/p50/p95 = 55/165/334 px, width 28/89/221 px, min side 11 px, mask/box fill ≈ 0.50 |
| Usable voltages | 344 of 351 (dead zone 0.25–0.31 V excluded) |
| Raw cube read | 9.5 s per `.mat` (gzip, chunks `(200,77,1)`) → cache needed for full-frame training |
| fp16 cache | 133 × 1680 × 1240 × 2 B = **0.55 GB / frame, ≈ 194 GB total** (`/data2` has 814 GB free) |
| YOLO26s (`ultralytics` 8.4.164, `yolo26s.pt` 10.0 M params) | first layer `Conv(conv=Conv2d(3, 32, 3, s=2, p=1, bias=False))`; `DetectionModel('yolo26s.yaml', ch=N, nc=1)` builds; 695/708 pretrained tensors load unchanged (first conv + class-head outputs re-init) |
| Loss API | `model.args = get_cfg(); loss, items = model.loss({'img','batch_idx','cls','bboxes'(norm xywh)})`, `items = {box_loss, cls_loss, l1_loss}` |
| Eval output | tuple, `out[0]` = `[B, 4 + nc, 43407]` raw xywh + scores → decode + `ops.non_max_suppression` |
| Memory | batch 2 × 344 ch × 1696 × 1248 fp16, forward + backward through `model.loss`: **9.0 GB peak** (input with grad 2.9 GB) on an RTX A4500 (20 GB) |
| Logging | TensorBoard available; `wandb` 0.30 installed, API key present, **no entity/team yet** → offline fallback |

## 3. Decisions

- **Step 0 — band window.** Loader default `band_range = (400, 800)` → bands 0–132, `in_channels = 133` in raw mode, `R [133, N]` without zero rows, stats 133×133, p99 recomputed over the 133 bands (`intensity_p99_summary.csv` stays as is; the loader computes the 133-band p99 scale from the cube when the window is narrower — see §5). `band_range = (400, 1000)` remains available.
- **No downsampling anywhere.** YOLO sees the full frame, zero-padded to 1696 × 1248 (stride-32 multiple); boxes are in native pixels.
- **Projection on the GPU.** The loader returns the fp16 133-band cube; `FilterBank` computes the responses on the GPU. At N = 344 the response tensor (1.43 GB fp16/frame) is materialised — measured to fit.
- **Weight vector** `w = N · softmax(θ)`, `θ ∈ R^N` trainable, applied as `y * w.view(1, N, 1, 1)`. Positive, mean 1 (input scale preserved), sparsified by an entropy penalty `λ_H · H(softmax θ)` so the ranking is meaningful (a free scalar per channel would be absorbed by the next conv). Ranking = `w` descending. Alternative (plain `nn.Parameter(ones(1,N,1,1))` + L1) is a two-line switch, not default.
- **N-channel first conv, no adapter.** `model.model[0].conv` is replaced by `Conv2d(N, 32, 3, s=2, p=1, bias=False)` initialised from the pretrained RGB kernel tiled over N channels and scaled by `3/N` (`adapt_input_conv` recipe). All other pretrained weights kept; `nc = 1`.
- **A → B transfer.** Keep channel indices `idx` (top 10 by `w`); new first conv `W_B[:, j] = w[idx_j] · W_A[:, idx_j]`; `FilterBank` rebuilt with `R[:, idx]`, no weight vector. B starts exactly where A left off on the kept channels.
- **Box quality is measured against the mask, not only the GT box.** YOLO26's box loss is CIoU + L1 (no MSE, no DFL): IoU already prefers the tighter of two boxes that both contain the object and CIoU's centre term penalises mis-centred boxes, but IoU is symmetric — a box 10 % too large and a box cutting off 10 % of the object score the same, while only the second one hurts Stage 2. Metrics per GT object (best-matching prediction): **mask coverage** (fraction of GT mask pixels inside the box; reported for the raw box and after the ROI margin), **coverage recall @ 0.99** (share of objects with coverage ≥ 0.99), **tightness** = area(GT box)/area(pred box) for containing boxes (1 = tight), **centre offset** = ‖c_pred − c_GT‖/diag(GT box); plus the standard recall@IoU 0.5, AP50 and matched IoU. **Checkpoint selection: coverage recall @ 0.99 first, mean tightness as tie-breaker.**
- **Asymmetric containment loss (optional, default on).** `ContainBboxLoss(BboxLoss)` adds `contain_weight · Σ_edges relu(gt_edge − pred_edge) / gt_size` (hinge on the part of the GT box outside the prediction, normalised by the GT width/height) to the CIoU + L1 box loss on assigned pairs, injected into both the one-to-many and one-to-one criteria. Under-coverage is penalised, over-coverage is left to CIoU, biasing the detector toward slightly-large-but-complete boxes. `contain_weight` in `cfg/det.yaml` (default 1.0; 0 disables → plain YOLO26 loss for ablation).
- **Validation split:** 28 of the 279 train ids held out (fixed seed 0, list saved to `data_loader/splits/det_val_ids.json`); 70 test frames used only for the final report.
- **Augmentation:** random horizontal / vertical flips (boxes flipped accordingly). Nothing else in Stage 1.
- **Code style:** Spec_Occu / Diffu conventions (DETR-style `util/misc.py`, `main_*.py` with `get_args_parser()` + `main(args)`, `train_eval/train_eval_*.py`, `models/build_*`, `weights/<exp>/`, `runs/`, `cfg/*.yaml`, `results_<name>.txt`).

## 4. Code layout

```
main_det.py                        entry: usage docstring (single GPU / torchrun); get_args_parser(); main(args)
cfg/det.yaml                       lr, lrf, epochs, batch_size, accumulate, gate_entropy_weight, top_k, roi_margin, conf/iou thresholds
data_loader/cube_cache.py          build_cube_cache(data_path, split, band_range, num_workers) -> <data_path>/cache_fp16/<split>/<id>.npy ; also __main__
data_loader/boxes.py               boxes_from_mask(mask, min_area=100) -> [K, 4] xyxy px ; boxes_to_yolo ; flip_boxes ; det_collate_fn
data_loader/my_dataset.py          HyperCOD_data gains cache_dir=, ids=, out_dtype=, band_range= and filter_bank_tensors()  (no new dataset class)
data_loader/splits/det_val_ids.json
models/filter_bank.py              FilterBank(nn.Module)
models/ec_yolo.py                  build_ec_yolo(args, dataset), replace_first_conv, load_pretrained_yolo, select_top_k, slice_to_channels, decode_predictions
train_eval/train_eval_det.py       train_one_epoch, evaluate, box metrics (recall@0.5, AP50, matched IoU)
util/misc.py                       DETR-style utilities copied from Spec_Occu (SmoothedValue, MetricLogger, reduce_dict, init_distributed_mode, save_on_master, is_main_process, ...)
util/distributed_util.py           Custom_DistributedSampler (copied)
util/logger.py                     TrainLogger: TensorBoard + W&B fan-out, main process only
main_det_rois.py                   session-B detector on all frames -> results/det/rois_<split>.json
weights/det_A/, weights/det_B/     model_{epoch}, model_best (git-ignored); runs/ (TensorBoard); wandb/ ; results_<name>.txt
```

## 5. Components

### 5.1 Loader change (step 0) — `data_loader/my_dataset.py`, `ec_filter.py`, `band_stats.py`
- `HyperCOD_data(..., band_range=(400.0, 800.0))`: `self.band_idx = np.where((WAVELENS_200 >= lo) & (WAVELENS_200 <= hi))[0]` (133 for the default), `self.wavelens = WAVELENS_200[self.band_idx]`, `self.n_bands = len(self.band_idx)`. `read_cube_block` slices bands `[band_idx[0]:band_idx[-1]+1]` (contiguous). `check_cube_layout` still expects the file to hold 200 bands. `in_channels = N if use_filter else n_bands`.
- `align_filter_to_wavelens` is called with the windowed `wavelens` → `R [n_bands, N]`; `valid_band_mask` all True for the default window.
- p99 scale: the CSV's p99 is the sum over 200 bands. For a narrower window the loader uses `scale = p99_window / n_bands` where `p99_window` is the 99th percentile of the windowed band sum, computed once per sample from the cube and cached in `<data_path>/intensity_p99_<lo>_<hi>.csv` (same columns as the original CSV; built on first use, ~279 × 9.5 s ≈ 45 min single-threaded → built with 8 workers in ~6 min; the frames cache build (§5.2) writes the same numbers, so running the cache build first makes this free). `band_range = (400, 1000)` uses the original CSV unchanged.
- `band_stats.compute_band_stats` stores `mean [n_bands]`, `cov [n_bands, n_bands]`, `band_range`; `load_band_stats` asserts the file's `band_range` matches the dataset's. File name `band_stats_train_<lo>_<hi>.npz`; the existing `band_stats_train.npz` is kept for `(400, 1000)`.
- Constants: `N_BANDS = 200` stays "bands in the file"; tests use `H, W = 48, 40` fixture with 200 bands and check `n_bands == 133`.

### 5.2 `data_loader/cube_cache.py`
- `build_cube_cache(data_path, split, band_range=(400, 800), num_workers=8, overwrite=False)`: for each `<id>.mat`, read `hypercube` once (full), slice the band window, swap to `[n_bands, H, W]`, store `np.float16` `.npy` at `<data_path>/cache_fp16/<split>/<id>.npy`; also computes the windowed p99 (§5.1) and writes the `intensity_p99_<lo>_<hi>.csv` rows. Worker pool = `torch.utils.data.DataLoader` over a tiny reader dataset (same pattern as `compute_band_stats`). Skips existing files unless `overwrite`. Prints progress every 25 cubes. `__main__` with `--data_path --split {train,test,all} --num_workers --band_range`.
- `cache_path(cache_dir, split, name)` helper used by `HyperCOD_data.read_cube_block`; when `cache_dir` is given the dataset asserts the cache exists and prints the build command otherwise (no silent 45-minute build inside a training script).

### 5.3 `data_loader/boxes.py`
- `boxes_from_mask(mask: bool [H, W], min_area=100) -> np.ndarray [K, 4] float32 xyxy` (x1, y1 inclusive; x2, y2 exclusive pixel edges), via `scipy.ndimage.label`; components with area < `min_area` dropped; `K = 0` allowed (empty array `[0, 4]`).
- `boxes_to_yolo(boxes, H, W) -> [K, 4] normalised cx, cy, w, h` for the loss; `flip_boxes(boxes, H, W, horizontal, vertical)`.
- `det_collate_fn(batch)`: takes the `(img, gt, name)` tuples of `HyperCOD_data` (full frames, `use_filter=False`, `norm='p99'`, `out_dtype='float16'`), derives boxes from each mask with `boxes_from_mask`, and returns `{'img': [B, n_bands, H, W] fp16, 'batch_idx': [ΣK], 'cls': [ΣK, 1] zeros, 'bboxes': [ΣK, 4] normalised cx,cy,w,h (÷ W, H), 'boxes_xyxy': list of [K, 4] px, 'names': list[str]}` — `img/batch_idx/cls/bboxes` are exactly the keys `model.loss` reads. Zero-padding to stride multiples and the random h/v flips (of `img` and boxes together) are done on the GPU in `train_one_epoch` — cheaper than in the workers, and no dataset change.

### 5.4 Reusing `HyperCOD_data` for full frames (no new dataset class)
Additive options on the existing class, all default to today's behaviour:
- `cache_dir=None`: when set, `read_cube_block` loads `cache_path(cache_dir, split, name)` (`np.load(..., mmap_mode='r')`, fp16 `[n_bands, H, W]`, sliced `[:, h0:h0+ch, w0:w0+cw]` — note the cache is stored in `[B, H, W]` order, so no transpose is needed) instead of the `.mat`; asserts the file exists (message names the build command) and has the expected shape/dtype. Works for crops and full frames alike.
- `ids=None`: explicit list of sample ids to keep (train / val split); every id must exist in the split.
- `out_dtype='float32'`: dtype of the returned `img` (`'float16'` for the detector: 0.55 GB per frame through the workers instead of 1.1 GB).
- `band_range=(400., 800.)`: step 0 (§5.1).
- `filter_bank_tensors()` → `(R [n_bands, N] float32 — L1-normalised when filter_norm='l1', channel_mean [N], channel_std [N], selected_voltages [N])` computed from the aligned matrix and the band statistics regardless of the instance's own `use_filter` / `norm` (the detector instance itself runs with `use_filter=False, norm='p99'`, so the standardisation is applied after the projection inside `FilterBank`).
- Detector usage: `HyperCOD_data(data_path, split, ids=train_ids, use_filter=False, norm='p99', crop_size=0, cache_dir=..., out_dtype='float16', filter_select='all', filter_norm='l1')` + `det_collate_fn`.
- Val split file: `data_loader/splits/det_val_ids.json` = 28 ids drawn once with `random.Random(0).sample(sorted train ids, 28)`, committed; `make_det_splits(data_path)` writes it if missing.

### 5.5 `models/filter_bank.py`
```python
class FilterBank(nn.Module):
    def __init__(self, R, channel_mean, channel_std, weight_vector=True, init_logits=None)
    def forward(self, x_fp16 [B, n_bands, H, W]) -> y [B, N, H, W]   # x already p99-scaled by the loader; fp16 under autocast
    @property weights -> w = N * softmax(theta) [N]            (ones when weight_vector=False)
    def entropy(self) -> scalar penalty term  H(softmax theta) / log N  (in [0, 1])
    def ranking(self) -> indices sorted by w descending
```
- Buffers: `R [n_bands, N]`, `mean [N]`, `std [N]` (float32); the projection runs as `torch.einsum('bn,bchw->bnhw', R.T, x)` in autocast (fp16 inputs, fp32 accumulation), then `(y − mean) / std`, then `y * w.view(1, N, 1, 1)`.
- Test: equals the dataloader's `use_filter=True, norm='p99z'` output on the same crop (rtol 1e-2 in fp16).

### 5.6 `models/ec_yolo.py`
- `load_pretrained_yolo(weights='yolo26s.pt')` → the ultralytics `DetectionModel` (downloads to `weights/pretrained/` if missing).
- `replace_first_conv(model, n_channels)`: new `Conv2d(n_channels, 32, 3, 2, 1, bias=False)`, weight = pretrained RGB kernel tiled to `n_channels` (`repeat(1, ceil(N/3), 1, 1)[:, :N]`) × `3/N`; replaces `model.model[0].conv`.
- `build_ec_yolo(args, dataset)` → `nn.Module` `ECYolo(filter_bank, yolo, stride=32)` with `forward(img)` = `yolo(filter_bank(pad_to_stride(img)))` and `loss(batch)` = `yolo.loss({**batch, 'img': filter_bank(pad_to_stride(batch['img']))})` + `λ_H · filter_bank.entropy()` (returned as an extra item `gate_entropy`); `pad_to_stride` zero-pads bottom/right to multiples of 32 (1680×1240 → 1696×1248) and the normalised `bboxes` are rescaled by `(W/Wp, H/Hp)` accordingly. `model.args = get_cfg()` set for the loss gains. Session B: `weight_vector=False`.
- `select_top_k(filter_bank, k=10)` → indices, voltages, weights (also written as `gate_ranking.csv`: rank, voltage, weight, index).
- `slice_to_channels(ecyolo, idx)` → new `ECYolo` with `R[:, idx]`, `mean/std[idx]`, first conv `W[:, idx] * w[idx]`, no weight vector; test: identical detector output on a random input before/after slicing when the dropped channels have `w = 0`.
- `decode_predictions(out, conf_thres, iou_thres, max_det)` → per image `[M, 6]` xyxy, conf, cls via `ultralytics.utils.ops.non_max_suppression`.

### 5.7 `train_eval/train_eval_det.py`
- `train_one_epoch(model, data_loader, optimizer, device, epoch, max_norm, scaler, accumulate, logger)` — `MetricLogger(delimiter="; ")` with meters `loss, box_loss, cls_loss, l1_loss, gate_entropy, lr`; `torch.cuda.amp.autocast(enabled=scaler is not None)`; gradient accumulation over `accumulate` steps; grad-norm clipping; logger scalars per optimizer step.
- `evaluate(model, data_loader, device, conf_thres=0.001, iou_thres=0.6, roi_margin, logger, epoch, tag)` — decodes predictions and, per GT object, picks the best-matching prediction (highest IoU among predictions above `conf_thres`; greedy one-to-one). Metrics (`train_eval/box_metrics.py`, pure numpy, unit-tested):
  - **mask coverage** `cov = |mask ∩ box| / |mask|` for the raw box and for the box expanded by `roi_margin` (the ROI Stage 2 will crop); **coverage recall @ 0.99** = share of GT objects with `cov ≥ 0.99` (raw and margin);
  - **tightness** `area(GT box) / area(pred box)` averaged over objects whose box contains the GT box (1 = tight; reported together with the containment rate);
  - **centre offset** `‖c_pred − c_GT‖ / diag(GT box)` averaged over matched objects;
  - **recall@IoU 0.5**, **AP50** (single class, all-points interpolation over confidence), **matched IoU**, mean detections per image.
  Returns a dict; logs all scalars plus histograms of coverage and tightness, and for the first 4 val frames an image with GT (green), predicted (red) and ROI (dashed) boxes over a false-colour composite of three responses. `model_best` = highest coverage recall @ 0.99 (raw box), ties broken by mean tightness.
- `ContainBboxLoss(BboxLoss)` (`models/ec_yolo.py`): overrides `forward` to add the containment hinge (see §3) to `loss_iou`; `build_ec_yolo` replaces `criterion.one2many.bbox_loss` and `criterion.one2one.bbox_loss` with it when `contain_weight > 0`; the extra term is returned as loss item `contain_loss`.

### 5.8 `main_det.py`
- `get_args_parser()`: `--session {A,B}`, `--top_k 10`, `--ranking weights/det_A/gate_ranking.csv` (B reads A's ranking + `--resume weights/det_A/model_best`), `--data-path`, `--band-range 400 800`, `--filter-select all`, `--hpy cfg/det.yaml`, `--output-dir weights/det_A`, `--name`, `--seed 42`, `--eval`, `--resume`, `--epochs`, `--batch_size 2`, `--accumulate 4`, `--num_workers 4`, `--lr 1e-4`, `--lrf 0.05`, `--weight_decay 5e-4`, `--amp` (`store_false` → on by default, as in Spec_Occu), `--wandb/--no-wandb`, `--wandb-entity`, `--wandb-project hsi_camo`, DDP `--world_size --dist_url`.
- `main(args)`: `utils.init_distributed_mode(args)`; seeds; datasets (train ids = train − val ids; val ids from the split file; test); `Custom_DistributedSampler` when distributed; `build_ec_yolo`; optionally `slice_to_channels` for session B; AdamW (weight vector logits and first conv **without** weight decay, everything else with); `LambdaLR` cosine `lf = ((1 + cos(πx/epochs))/2)(1 − lrf) + lrf`; `GradScaler`; loop: train epoch → evaluate val → log → save `model_{epoch}` (every `save_every`) and `model_best` (by val recall, AP50 tie-break); at the end evaluate `model_best` on val + test and write `results_<name>.txt` lines; session A additionally writes `gate_ranking.csv` and logs the weight-vector bar chart/table.
- Checkpoint dict: `{"model", "optimizer", "scaler", "lr_scheduler", "epoch", "args", "selected_indices", "selected_voltages"}`; `utils.save_on_master`.

### 5.9 `util/logger.py`
- `TrainLogger(args, cfg)`: on the main process creates `SummaryWriter(log_dir=f'runs/{args.name}')` and, if `args.wandb`, `wandb.init(entity=args.wandb_entity, project=args.wandb_project, name=args.name, config={**vars(args), **cfg}, dir='wandb')`; if `wandb_entity` is None or `wandb.init` raises `CommError`, retries with `mode='offline'` and prints the `wandb sync` hint. Methods: `scalar(tag, value, step)`, `scalars(dict, step)`, `histogram(tag, tensor, step)`, `image(tag, hwc_uint8, step)`, `table(tag, columns, rows, step)` (W&B only), `finish()`. Every call is a no-op on non-main ranks.

### 5.10 `main_det_rois.py`
- Loads a session-B checkpoint, runs `evaluate`-style inference on all frames of `--split {train,test}`, expands each box by `roi_margin` (default 1.5× the box side, minimum 256 px, clipped to the frame), and writes `results/det/rois_<split>.json`: `{name: [[x1, y1, x2, y2, conf], ...]}` plus the GT boxes for reference. Stage 2 consumes this file.

## 6. Data flow (session A, train)

```
HyperCOD_data(use_filter=False, norm='p99', crop_size=0, cache_dir, out_dtype='float16')
  cache_fp16/<split>/<id>.npy ──> img fp16 [133, 1680, 1240] (÷ p99 scale) ; GT png ──> mask [1, H, W]
det_collate_fn: boxes_from_mask ──> batch_idx / cls / bboxes (norm xywh), img [B, 133, 1680, 1240]
                                                                                        ▼ GPU (train_one_epoch)
random h/v flips of img + bboxes ──> pad_to_stride ──> [B, 133, 1696, 1248]
FilterBank: y = w ⊙ ((Rᵀ img − m)/s)  ──> [B, 344, 1696, 1248] fp16 ──> YOLO26s (first conv 344→32) ──> model.loss
                                                                              └── + λ_H · entropy(w)
```

## 7. Training procedure

1. `python -m data_loader.cube_cache --split all --num_workers 8` (≈ 1 h once) — also produces the 133-band p99 CSVs; `python -m data_loader.band_stats --band_range 400 800` (≈ 3 min).
2. **Session A:** `python main_det.py --session A --name det_A --output-dir weights/det_A` (single GPU; `torchrun --nproc_per_node=2 main_det.py ...` for DDP). ~100 epochs × 126 steps (251 train frames / batch 2) ≈ 12.6 k iterations; at ~0.6 s/it ≈ 2 h. Outputs: `weights/det_A/model_best`, `gate_ranking.csv`, `results_det_A.txt`.
3. **Session B:** `python main_det.py --session B --top_k 10 --ranking weights/det_A/gate_ranking.csv --resume weights/det_A/model_best --name det_B --output-dir weights/det_B`. Same schedule. Outputs: `weights/det_B/model_best`, `results_det_B.txt`.
4. `python main_det_rois.py --resume weights/det_B/model_best --split train` and `--split test` → `results/det/rois_*.json`.

## 8. Error handling

Asserts with f-string messages (project style): cache file missing → message with the exact build command; cache array shape `(n_bands, H, W)` and dtype fp16 checked on load; `band_range` of stats/cache/CSV must match the dataset's; session B requires `--ranking` and `--resume`; `top_k ≤ N`; boxes clipped to the frame after flipping; empty GT (`K = 0`) frames are legal for the loss (no targets). W&B failures never abort training (offline fallback + printed warning). `torch.multiprocessing.set_sharing_strategy('file_system')` as in Spec_Occu.

## 9. Testing (synthetic fixture, pytest)

- Loader step 0: `band_range=(400, 800)` → `n_bands == 133`, `in_channels == 133` raw, `R.shape == (133, N)`, no zero rows; windowed p99 CSV built and used; `(400, 1000)` reproduces the current behaviour (existing tests updated for the new default where they hard-code 200/133).
- `cube_cache`: build on the fixture (2 train + 1 test cubes) → files exist, `np.load` shape `(133, 48, 40)` fp16, values equal the h5 cube (within fp16), skip-if-exists honoured, p99 CSV rows written.
- `boxes_from_mask`: single 6×6 object → one box `[20, 10, 26, 16]`; two objects → two boxes; a 5-px speck dropped; empty mask → `[0, 4]`. `flip_boxes` round-trips.
- `HyperCOD_data` additions: `cache_dir` returns the same values as the `.mat` path (within fp16) for crops and full frames and asserts on a missing/mis-shaped cache file; `ids` restricts the sample list and rejects unknown ids; `out_dtype='float16'` returns fp16; `filter_bank_tensors()` matches `sensor_R_matrix` / `channel_mean` / `channel_std` of an equivalent `use_filter=True, norm='p99z'` instance. `det_collate_fn` produces the loss keys with the right shapes and normalisation from the fixture masks; GPU-side `pad_to_stride` gives `(133, 64, 64)` for the 48×40 fixture and rescales the normalised boxes; flips move boxes consistently (checked against a flipped mask).
- `FilterBank` ≡ dataloader (`use_filter=True, norm='p99z', filter_norm='l1'`) on the same window; `weights` sum to N; `entropy() ∈ [0, 1]`; `ranking()` sorted.
- `replace_first_conv`: shape `(32, N, 3, 3)`, kernel tiling + `3/N` scaling; `slice_to_channels` preserves outputs when dropped channels have zero weight; `select_top_k` returns k unique indices and writes the CSV.
- `box_metrics`: on hand-made masks/boxes — perfect box → coverage 1, tightness 1, offset 0, recall 1, AP50 1; box cutting 20 % of the mask → coverage 0.8, not counted in coverage recall @ 0.99 even though IoU > 0.5; a 2× larger containing box → coverage 1, tightness 0.25; shifted box → offset equals the hand-computed value; no predictions → recall 0, AP 0, tightness nan; ROI margin turns a slightly-cutting box into coverage 1.
- `ContainBboxLoss`: zero extra loss when the prediction contains the GT box; positive and proportional to the protruding fraction when it does not; `contain_weight = 0` reproduces `BboxLoss` exactly.
- `train_eval_det`: one `train_one_epoch` + `evaluate` step on a tiny `yolo26n.yaml` (`ch=133, nc=1`, random init) with the fixture frames on CPU — runs end to end, loss finite, all metric keys present, checkpoint dict has the required keys.
- `TrainLogger`: with `wandb` disabled writes TensorBoard scalars to a tmp `runs/`; with a forced `CommError` falls back to offline without raising.

## 10. Open items

- **W&B entity — resolved:** the account had no personal entity, so the team **`chaoyi-hsi`** was created through the API (user-authorised); `--wandb-entity` defaults to it and an online run was verified (`https://wandb.ai/chaoyi-hsi/hsi_camo`). Offline fallback stays in place.
- **Environment:** a new conda env **`hsi_camo`** (Python 3.12.14, PyTorch **2.11.0+cu128** — the newest build that runs on driver 575.57 / CUDA ≤ 12.9; PyTorch 2.14 is CUDA-13-only and would need a driver ≥ 580 — torchvision 0.26, numpy 2.5, scipy 1.18, h5py 3.16, pillow 12.3, ultralytics 8.4.164, wandb 0.30, tensorboard, matplotlib, pyyaml, pytest) is used for this stage; `hsi` (torch 2.8) is left untouched for the user's other projects. All commands in this spec run with `/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python`; the existing 56 loader tests pass there and the YOLO26 memory probe reproduces 9.0 GB peak.
- **Weight-vector parameterisation:** `N·softmax(θ)` + entropy penalty is the default; plain `(1, N, 1, 1)` parameter + L1 is a two-line alternative if preferred.
- **Stage 2 (segmentation on ROIs):** separate spec; consumes `results/det/rois_*.json` and the existing crop loader with explicit windows; repeats all-344 → top-10.

## 12. Results (Task 11, 2026-09-28/29)

**Training infrastructure findings.** (1) Full-frame reads must bypass the page cache: inside the training process, mmap or `read()` of the 0.55 GB fp16 frames stalled every DataLoader worker in kernel memory reclaim (memory PSI ≈ 70 %, disk idle), 8–17 s/step; `cube_cache.read_npy_direct` (O_DIRECT) gives a steady ≈ 2 s/step, disk-bound at ≈ 380 MB/s on the SATA SSD (4.2 min/epoch + 30 s val). (2) The gate logits need their own learning rate (`gate_lr` 2e-2): at the model's 1e-4 the 344-way softmax never left uniform (normalised entropy 1.0000 after 12 epochs). (3) `DetectionModel` must be told `end2end = True` or eval runs the one-to-many head + NMS (time-limit warnings at native resolution). (4) Only one full-resolution job fits on the box at a time (two jobs re-trigger the reclaim stall).

**Session A (`det_A` run #6, 100 epochs, 7 h 50 min, W&B run `3nfbc5md`).** Gate: normalised entropy 1.00 → 0.943, weights 0.15–6.65 (33× spread). **Top-10 voltages** (converged gate, identical at epochs 89 and 99): `1.32, 1.29, 1.35, -0.41, -0.38, 1.38, -0.44, 1.41, 1.26, -0.35 V` — two clusters (1.26–1.41 V, −0.35 to −0.44 V); weights 6.65 … 3.80 at ranks 1–10, 3.66 at rank 11, 1.77 at rank 50, 1.07 at rank 100 (`weights/det_A/gate_ranking_ep89.csv`, `top_k_ep89.json`).

*Checkpoint selection deviation.* The spec's rule `(coverage_recall99_raw, tightness)` is computed over all decoded candidates (conf ≥ 0.001, ≤ 300/frame) and is confidence-blind: it selected **epoch 0** (`coverage_recall99_raw` 0.645 from 300 untrained boxes, recall50 0). `cfg select_keys` now defaults to the ROI operating-point pair `(coverage_recall99_roi_op, recall50_op)` — the boxes Stage 2 receives — and the spec pair remains selectable. Session B was initialised from `model_89`, the best saved checkpoint under that pair; the best epoch overall was 75 (0.452), not saved (`save_every` 10).

*ROI operating point.* Calibrated on `model_99` (val): the confidence of the box matched to each object has median 0.003 and 29 % of objects have no candidate at all; `roi_conf` 0.25 keeps 35 % of objects inside an exported ROI (0.46 boxes/frame), 0.10 → 52 % (0.82), 0.05 → 55 % (1.07), **0.02 → 58 % (1.46, chosen)**, 0.001 (top-5 only) → 65 % (2.61).

| Session A, `model_89` | val (28 fr / 31 obj) | test (70 fr / 71 obj) |
|---|---|---|
| recall50 / AP50 | 0.452 / 0.300 | 0.563 / 0.444 |
| matched IoU | 0.727 | 0.722 |
| coverage_recall99_raw / _roi (all candidates) | 0.387 / 0.613 | 0.366 / 0.690 |
| tightness / center_offset (all candidates) | 0.490 / 0.288 | 0.449 / 0.271 |
| recall50_op / coverage_recall99_roi_op (conf ≥ 0.02, top 5) | 0.387 / 0.548 | 0.493 / 0.549 |
| contain_rate_op / tightness_op / center_offset_op | 0.226 / 0.543 / 0.096 | 0.183 / 0.565 / 0.077 |
| candidates per frame (conf ≥ 0.001 / operating point) | 5.9 / 1.21 | 4.6 / 1.16 |

Best single-epoch val values during A (all candidates): recall50 0.58 (ep 44), AP50 0.42 (ep 70), coverage_recall99_roi 0.74 (ep 30); at the operating point coverage_recall99_roi_op 0.45 (ep 75).

**Session B (`det_B`, the 10 voltages above, no weight vector, initialised from A `model_89`, W&B `sx6olkal`).** Stopped by the user at epoch 91/100 (plateau since ≈ epoch 25). With only 10 channels the GPU footprint drops from 15.3 GB to 5.3 GB; the step time stays ≈ 1.6 s because the run is disk-bound. Warm start: val recall50 0.39 / AP50 0.21 after epoch 0, 0.55 / 0.26 after epoch 1. `model_best` under `select_keys` = **epoch 25**. Best single-epoch val values: AP50 0.50 (ep 36, 60), recall50 0.65 (ep 5, 36, 37, 82), matched IoU 0.79 (ep 48), tightness 0.78 (ep 36), coverage_recall99_roi_op 0.61 (ep 25).

| Session B, `model_best` (epoch 25) | val (28 fr / 31 obj) | test (70 fr / 71 obj) |
|---|---|---|
| recall50 / AP50 | 0.613 / 0.423 | 0.521 / 0.356 |
| matched IoU | 0.712 | 0.683 |
| coverage_recall99_raw / _roi (all candidates) | 0.387 / 0.710 | 0.268 / 0.620 |
| tightness / center_offset (all candidates) | 0.690 / 0.114 | 0.608 / 0.177 |
| recall50_op / coverage_recall99_roi_op (conf ≥ 0.02, top 5) | 0.548 / 0.613 | 0.437 / 0.465 |
| contain_rate_op / tightness_op / center_offset_op | 0.226 / 0.680 / 0.089 | 0.169 / 0.633 / 0.110 |
| candidates per frame (conf ≥ 0.001 / operating point) | 9.9 / 1.79 | 8.2 / 1.70 |

*A vs B.* On val the 10-channel model is ahead of the 344-channel one (AP50 0.42 vs 0.30, recall50 0.61 vs 0.45, ROI coverage at the operating point 0.61 vs 0.55); on test it is slightly behind (AP50 0.36 vs 0.44, recall50 0.52 vs 0.56, 0.47 vs 0.55) — within the noise of 31/71 objects, so the top-10 filter responses carry essentially the information the detector uses. Caveat: the 10 voltages form two clusters of near-identical spectral responses (1.26–1.41 V, −0.35 to −0.44 V), so this is closer to "two responses suffice" than to "these ten are optimal"; a control with 10 uniformly spaced or OSP-selected voltages would separate the two readings. Confidence calibration on B `model_best` (val) matches A's: `roi_conf` 0.25 → 36 % of objects inside an exported ROI (0.57 boxes/frame), 0.10 → 52 % (0.93), **0.02 → 61 % (1.79, kept)**, 0.001 (top-5 only) → 71 % (3.57).

**ROI export** (`main_det_rois.py --session B --resume weights/det_B/model_best`, conf ≥ 0.02, top 5, margin 1.5×, min 256 px): `results/det/rois_test.json` — 70 frames, 71 objects, ROI coverage recall@0.99 **0.465**; `results/det/rois_train.json` — 279 frames (train + val), 289 objects, **0.696**. Stage 2 therefore starts with ≈ 1.7 ROIs per frame that fully contain about half of the unseen objects; lowering `roi_conf` to 0.001 raises val coverage to 0.71 at 3.6 ROIs per frame if Stage 2 prefers recall.

**Why the filter path underperforms (diagnostics on 8 val frames, `tmp/filter_diag.py`).** Normalisation is not the cause: the standardised response channels have std 0.21–0.33 (no channel below 5 % of the median, so nothing is noise-amplified), |mean|/std 1.0–1.6, fp16-cache quantisation noise 2·10⁻⁴ of the signal after standardisation, per-frame mean offsets 0.06 std, and the observed/predicted std ratio (0.41–0.88) is the same within-scene shrinkage the raw bands show; the raw control uses the identical p99 scaling and band-statistics standardisation. Two real causes: (a) *spectral resolution* — the aligned response matrix R (133×344, L1 columns) has effective rank 6 (99 % energy) / 11 (99.9 %), median FWHM 112 nm (36–311) against 3 nm bands, adjacent voltages 0.9998 cosine-similar; object-vs-background separability drops from a Mahalanobis distance of 3.21 (raw) to 2.80 (responses; 68–95 % kept per frame, mean 87 %) and the best single-feature Fisher ratio from 1.24 (raw, the 798 nm red edge in half the frames) to 0.89; (b) *conditioning* — 344 channels spanning an 11-dimensional subspace give the first conv a near-singular input, which matches the ≈ 4× slower learning of session A and a gate with no signal to break symmetry. Decisive follow-up: train the filter model from scratch on the 11 PCA-whitened response directions (`--pca-channels 11`, no gate) or on 12 uniformly spaced voltages; parity with the raw model would mean the pipeline loss was conditioning, a plateau near AP50 0.5 would mean it is the sensor's bandwidth.

**Control (`raw133_A`, `--raw-bands`: the 133 raw bands of 400–800 nm straight into YOLO26s, identity projection, no gate, same recipe otherwise, W&B `ku0jkcha`, 100 epochs, 7 h 49 min).** The raw bands learn ≈ 4× faster (val recall50 0.55 at epoch 3 vs epoch ≈ 40 for A) and plateau far higher. `model_best` under `select_keys` = **epoch 57**; mean of the last 10 epochs (val): AP50 0.67, recall50 0.77, coverage_recall99_roi 0.78, coverage_recall99_roi_op 0.67. Best single-epoch val values: AP50 0.72 (ep 84), recall50 0.84 (ep 37/57), matched IoU 0.84 (ep 24), tightness 0.86 (ep 49).

| best checkpoint, test split (70 fr / 71 obj) | A: 344 responses + gate (`model_89`) | B: top-10 responses (ep 25) | raw 133 bands (ep 57) |
|---|---|---|---|
| recall50 / AP50 | 0.563 / 0.444 | 0.521 / 0.356 | **0.761 / 0.645** |
| matched IoU | 0.722 | 0.683 | **0.814** |
| coverage_recall99_raw / _roi (all candidates) | 0.366 / 0.690 | 0.268 / 0.620 | **0.549 / 0.803** |
| tightness / center_offset (all candidates) | 0.449 / 0.271 | 0.608 / 0.177 | **0.753 / 0.072** |
| recall50_op / coverage_recall99_roi_op (conf ≥ 0.02, top 5) | 0.493 / 0.549 | 0.437 / 0.465 | **0.732 / 0.761** |
| contain_rate_op / tightness_op / center_offset_op | 0.183 / 0.565 / 0.077 | 0.169 / 0.633 / 0.110 | **0.324 / 0.739 / 0.075** |
| candidates per frame (conf ≥ 0.001 / operating point) | 4.6 / 1.16 | 8.2 / 1.70 | 12.4 / 1.93 |
| val (28 fr / 31 obj): recall50 / AP50 / cov99_roi_op | 0.452 / 0.300 / 0.548 | 0.613 / 0.423 / 0.613 | **0.839 / 0.623 / 0.774** |

*Reading.* On unseen data the raw bands find 76 % of the camouflaged objects at IoU 0.5 and put 76 % of them fully inside an exported ROI at 1.9 ROIs per frame, against 56 % / 55 % for the 344 filter responses: the EC filter path costs roughly a fifth of the objects and 0.2 AP50 on this task. The raw run is an upper bound (the deployed sensor only delivers responses); the diagnostics above attribute the gap to the responses' spectral resolution (effective rank 6–11, FWHM ≈ 112 nm, 87 % of the object/background separability kept) and to the conditioning of a 344-channel input that spans 11 dimensions. Next experiment (implemented, `bash_files/launch_pca11.sh`): the filter model on 11 whitened principal response directions (or 12 uniform voltages), no gate — parity with the raw run would mean the pipeline loss was conditioning, a plateau near AP50 0.5 would mean it is the sensor's bandwidth.

**PCA-11 follow-up (`pca11_A`, `bash bash_files/launch_pca11.sh`: session A from scratch on the 11 PCA-whitened response directions, no gate, W&B `beuhl194`, 100 epochs, 8 h 03 min, 2026-09-29).** Pre-flight on real frames: fp16 compute and the fp16 cache add ≤ 0.9 % noise to every whitened channel (fp64 comparison), so the weak directions survive the pipeline. The whitened input learns fastest of all runs — 3-epoch-mean val AP50 ≥ 0.3 at epoch 5 (raw 11, A 37) — but the lead over raw is gone by epoch ≈ 35 and the late val plateau is AP50 ≈ 0.62 (raw 0.66–0.68). `model_best` = **epoch 24**: the operating-point rule never improved on (0.710, 0.710) afterwards, so the automatic test line comes from an early model and the comparison below also uses fixed checkpoints. Evaluation: `docs/reports/2026-09-29-pca11-robustness/robust_eval.py` (inference only; re-evaluating each run's stored checkpoint reproduces its recorded test numbers exactly), 95 % CIs from a paired bootstrap over the 70 test frames (2000 resamples).

| test split (70 fr / 71 obj) | PCA-11 | raw 133 bands | A: 344 responses + gate |
|---|---|---|---|
| `model_best`: recall50 / AP50 / matched IoU | 0.746 / 0.620 / 0.791 (ep 24) | 0.761 / 0.645 / 0.814 (ep 57) | 0.563 / 0.444 / 0.722 (`model_89`) |
| `model_best`: recall50_op / coverage_recall99_roi_op / boxes per frame at the operating point | 0.690 / 0.732 / 1.73 | 0.732 / 0.761 / 1.93 | 0.493 / 0.549 / 1.16 |
| fixed checkpoints `model_59` … `model_99`, mean AP50 / recall50 | 0.663 / 0.766 | 0.662 / 0.749 | `model_99` alone: 0.522 / 0.606 |

Paired differences: PCA-11 − raw over the five fixed checkpoints AP50 +0.001 [−0.094, +0.098], recall50 +0.017 [−0.070, +0.107] (at `model_99` −0.023 [−0.136, +0.092]; at `model_best` −0.025 [−0.129, +0.078]); PCA-11 `model_99` − A `model_89` AP50 +0.214 [+0.084, +0.352], recall50 +0.211 [+0.096, +0.329].

*Robustness* (same script; the perturbation acts inside the filter bank, the trained weights are unchanged). Read noise: i.i.d. Gaussian per bias-voltage reading (per band for raw), σ = the channel's mean absorbed signal / 10^(dB/20) in p99 units (auto-exposure); for PCA-11 the 344 per-voltage noises pass through its fixed 344 → 11 map. The only measured device figure (ECHSE SI Fig. S7) is ≈ 0.63 % per reading, ≈ 44 dB, under 20 mW/cm² bench light. Bias offset: every voltage lands dv higher (hysteresis / drift). Gain error: fixed per-voltage N(0, 1 %). Device-1: the measured `R_Device1.mat` responses (per-voltage least-squares gain) instead of the EC_filterV3 interpolant. The perturbed response columns change by a median 1.1 % (+5 mV), 4.4 % (+20 mV), 0.66 % (gain) and 0.36 % (Device-1). "Re-measured" re-standardises each channel with the perturbed device's own mean/std.

| condition, test AP50 / recall50 | PCA-11 `model_99` | A `model_89` | B `model_best` (ep 25) | raw 133 `model_99` |
|---|---|---|---|---|
| clean | 0.658 / 0.775 | 0.444 / 0.563 | 0.356 / 0.521 | 0.681 / 0.761 |
| read noise 50 dB | 0.705 / 0.789 | 0.444 / 0.563 | 0.356 / 0.521 | 0.680 / 0.761 |
| read noise 40 dB | 0.621 / 0.704 | 0.445 / 0.563 | 0.354 / 0.521 | 0.681 / 0.761 |
| read noise 30 dB | **0.148 / 0.155** | 0.450 / 0.563 | 0.343 / 0.507 | 0.676 / 0.761 |
| bias +5 mV | **0.409 / 0.507** | 0.452 / 0.563 | 0.365 / 0.521 | – |
| bias +20 mV | **0.036 / 0.070** | 0.437 / 0.549 | – | – |
| 1 % per-voltage gain error | **0.479 / 0.535** | 0.446 / 0.563 | – | – |
| measured Device-1 response | **0.459 / 0.535** | 0.455 / 0.563 | 0.357 / 0.507 | – |
| +5 mV, channel mean/std re-measured | 0.444 / 0.521 | – | – | – |
| 1 % gain, re-measured | 0.471 / 0.549 | – | – | – |
| Device-1, re-measured | 0.548 / 0.620 | – | – | – |
| u9–u11 set to their training mean | 0.382 / 0.493 | – | – | – |
| u6–u11 set to their training mean | 0.149 / 0.197 | – | – | – |

Paired differences to clean (PCA-11): 30 dB −0.510 AP50 [−0.643, −0.391]; 40 dB −0.037 [−0.123, +0.064]; +5 mV −0.249 [−0.363, −0.124]; gain −0.180 [−0.297, −0.059]; Device-1 −0.199 [−0.322, −0.068], re-measured −0.110 [−0.225, +0.007]; u9–u11 −0.277 [−0.417, −0.135]. For A every perturbation stays within ±0.011 AP50.

*Reading.* On the noiseless simulation the EC responses carry the information: PCA-11 matches the raw bands on test (fixed checkpoints, CI ± 0.1 AP50) and beats A by 0.21 AP50 (`model_99` vs `model_89`, paired) and B by 0.26 (`model_best` vs `model_best`). A's deficit was input conditioning (344 channels in an ≈ 11-dimensional span, condition number 2.75·10⁶ over its top 11 directions against 1 after whitening), B's additionally the gate's redundancy-blind top-10 (two clusters of near-duplicate voltages). This is the "parity → conditioning" outcome, with two caveats. (1) The raw input is itself ill-conditioned (condition number 3.8·10⁴), so the raw ceiling may be beatable by whitened raw bands; a raw-PCA control (e.g. 40 components, `--pca-channels` with `--raw-bands`, not yet supported) completes the {responses, raw} × {standardised, whitened} comparison before any residual gap is blamed on the sensor. (2) Parity holds only without sensor error. Whitening makes the detector rely on directions 8–11 (together 5.9·10⁻⁶ of the response variance; zeroing u9–u11 costs 0.28 AP50), and per-reading noise at 30 dB, a 5 mV bias offset, a 1 % gain error or the device's own measured response cost 0.18–0.51 AP50, while A and B — which never learned to use those directions — are unaffected by all of them. Re-measuring each channel's statistics on the device recovers at most half: even after it, the correlation of u11 with its nominal version is only 0.36 under +5 mV or Device-1 (u8 0.58 under +5 mV, u9 0.79 under Device-1), i.e. the fixed 344 → 11 map mixes channels rather than merely offsetting them. A deployable version therefore needs a handful of well-spread voltages (a pixel-level analysis on 2026-09-29, not in the repo, found that 12 voltages — −0.89, −0.69, −0.35, −0.05, 0.08, 0.14, 0.40, 0.84, 1.37, 1.54, 2.25, 2.43 V — reproduce all 11 whitened directions in the noiseless limit and no 11-voltage set does), a regularised whitening with fewer components, training with read-noise and bias/gain perturbation augmentation, and per-device calibration.

**How to pick the voltages (2026-10-01; analyses in the report `docs/reports/2026-09-29-raw-vs-ec/` and the project memory).** The pixel-level analyses on 56–112 frames settled three points. (1) The 344 responses keep 91 % (un-whitened, ridge 1e-3) to 99 % (whitened) of the raw bands' object-vs-surround separability, so the filter loses little information; session A's gap is the input format (344 near-duplicate channels, adjacent correlation 0.99998, input condition number 8.6e11, versus 7 after PCA whitening). (2) A per-channel weight vector cannot choose voltages: it leaves the information (ratio 0.999) and the near-duplicate structure unchanged, ranked the most discriminative single channels (1.70–1.79 V) only 26th–75th, and its top-10 keeps 0.654 of raw's separability on val; an L1 or stronger entropy penalty makes it worse (blocks of adjacent voltages, 0.46–0.81 at 40 dB). (3) Scoring the SET of readings works: forward greedy selection under a read-noise floor reaches 0.845 at 40 dB, label-free D-optimal design ties it, and every per-channel ranking (gate, L1, group lasso, leverage) fails (0.49–0.74).

*Method (`main_select_voltages.py`).* Per train frame: object pixels (GT mask eroded 2 px, ≤ 6000 samples) vs the 3–60 px surround (≤ 12000), mean difference d and pooled covariance P of the standardised readings z = (R^T x − mean)/std of all 344 usable voltages. Objective: the mean over frames of log d_S^T (P_S + diag(σ²/std²) + 10⁻⁹ tr I)⁻¹ d_S of the chosen set S (its geometric-mean separability) under a per-reading read-noise floor σ = s0·10^(−dB/20), s0 = median RMS reading over the whole bank (0.4507 in p99 units; 40 dB → σ = 0.0045; `models/ec_yolo.read_noise_std`, model `floor`). Greedy forward steps with ≥ 0.05 V between voltages and nothing within 0.05 V of the 0.25–0.31 V dead zone; the order is nested. Selection uses the 251 train frames only; the 28 val frames are reported, the test split untouched (294 s on CPU). Result, greedy order: **1.75, −0.44, 1.36, 0.48, −0.65, −0.36, 1.70, 0.01, −0.86, 1.45 V** (the same basin as the 84-frame analysis, every voltage within 0.03 V), written to `results/det/voltages_greedy.json` (git-ignored) and hard-coded in `bash_files/launch_sel10_queue.sh`. The script takes `main_det.py`'s own `--read_noise_db` / `--read_noise_model` flags (default 40 dB here), so a set is scored under the noise it is then trained with.

| fraction of the raw bands' separability kept, val (28 frames, median) | un-whitened (ridge 1e-3) | whitened (ridge 1e-9) | 40 dB read noise | 30 dB |
|---|---|---|---|---|
| greedy set (10 voltages) | **0.847** | 0.950 | **0.846** | **0.763** |
| session B's gate top-10 | 0.653 | 0.927 | 0.636 | 0.523 |
| 10 uniformly spaced usable voltages (`--filter-select uniform`) | 0.829 | 0.956 | 0.831 | 0.744 |
| all 344 | 0.927 | 0.992 | 0.918 | 0.865 |

*Reading.* Whitened and noise-free, every reasonable set is equivalent (0.93–0.96): session B failed mainly because ten near-copies were fed un-whitened, and the voltage choice matters under read noise (+33 % over the gate's set at 40 dB) but only ≈ 2 % over uniform spacing. A noise-free whitened detector run therefore cannot rank voltage sets; the comparison needs the noise in the loop.

*Read noise in the detector (`--read_noise_db`, `--read_noise_model`).* `FilterBank` adds i.i.d. Gaussian noise of std σ to every reading before the standardisation (fresh draws in training; in eval mode from the bank's own generator, reseeded by `eval()`, so every validation pass and `main_det_compare.py` see identical noise and the numbers reproduce). With `--pca-channels K` the whitening is fitted to the signal-plus-noise covariance (`models/ec_yolo.pca_whitening`, noise variance on the diagonal) and applied after the noise as `FilterBank.proj` ([N, K], fp32): the gain 1/√λ on a noise-dominated direction is then capped at unit output noise, where the noise-free whitening of PCA-11 gained up to 1658× and broke under 30 dB noise or a 5 mV bias (robustness evaluation, 2026-09-29). Without read noise the folded `pca_whitened_channels` path and all earlier checkpoints are unchanged. `main_det_compare.py` evaluates runs on fixed checkpoints (`model_59..99`, `model_best` is AP-blind) under the trained noise and clean, one pass over the frames, with a paired frame bootstrap of the differences.

*Confirming runs (`bash_files/launch_sel10_queue.sh`, sequential, 3 × ≈ 8–9 h).* `sel10g_A` (greedy set), `sel10b_A` (session B's voltages) and `sel10u_A` (uniform 10), each `--pca-channels 10 --read_noise_db 40`, 100 epochs, YOLO26s from the COCO stem, no gate; then `main_det_compare.py --runs sel10g_A sel10b_A sel10u_A` on test. Expected from the pixel analysis: greedy ≈ uniform > session B under the trained noise, all three close when evaluated clean.

*Results (2026-10-02; `results/det/compare_sel10/compare.json`; each run 100 epochs, ≈ 7 h 45 min; 70 test frames, paired frame bootstrap with 2000 resamples).* Fixed checkpoints `model_59..99`, evaluated with the trained 40 dB noise ("trained") and on clean readings ("clean"):

| test split (70 fr / 71 obj) | greedy 10 (`sel10g_A`) | session B's 10 (`sel10b_A`) | uniform 10 (`sel10u_A`) |
|---|---|---|---|
| AP50 / recall50, trained noise, mean of `model_59..99` | 0.519 / 0.617 | 0.375 / 0.583 | **0.609 / 0.704** |
| recall50_op / coverage_recall99_roi_op, trained | 0.563 / 0.614 | 0.513 / 0.592 | **0.639 / 0.668** |
| matched IoU, trained | 0.782 | 0.766 | **0.801** |
| AP50 / recall50, clean readings | 0.008 / 0.025 | 0.030 / 0.104 | 0.102 / 0.161 |
| `model_best`: test AP50 / recall50 / coverage_recall99_roi_op (epoch) | 0.514 / 0.648 / 0.620 (54) | 0.416 / 0.549 / 0.634 (48) | **0.592 / 0.746 / 0.690** (50) |
| last-10-epoch val AP50 / recall50 (noisy) | 0.438 / 0.590 | 0.378 / 0.555 | **0.582 / 0.752** |

Paired differences under the trained noise: greedy − session B AP50 +0.144 [+0.021, +0.250]; greedy − uniform AP50 −0.090 [−0.160, −0.022], recall50 −0.087 [−0.149, −0.032]; session B − uniform AP50 −0.234 [−0.335, −0.120]. Trained − clean: greedy +0.511 [+0.401, +0.619], uniform +0.507 [+0.389, +0.614], session B +0.345 [+0.240, +0.460].

*Reading.* (1) The voltage choice matters on the detector as much as at the pixel level, but the ranking is not the predicted one: uniform spacing beats the greedy set by 0.09 AP50 and 0.09 recall (CIs exclude 0), and both beat session B's clustered set. Ten uniformly spaced voltages at 40 dB (0.609 / 0.704 on fixed checkpoints, 0.592 / 0.746 / 0.690 at `model_best`) are within noise of PCA-11's 344 clean readings (0.663 fixed / 0.620 `model_best`) and ≈ 0.05 AP50 below the raw bands (0.662 / 0.645). (2) The per-frame Mahalanobis objective did not transfer to the detector. It rewards small differences between neighbouring voltages — the greedy set holds three pairs 0.05–0.09 V apart (−0.44/−0.36, 1.70/1.75, 1.36/1.45) and nothing above 1.75 V — which per-frame discriminants exploit but one network under read noise cannot, while uniform spacing covers the whole range including the 2.1–2.5 V plateau responses that reach 798 nm. The pooled-pixel classifier of the 2026-09-29 analysis, which tied uniform with greedy, was the better proxy; per-frame separability over-fits per-frame discriminants. For the detector the 10-voltage recommendation is therefore **uniform spacing** (or a set-level objective pooled across frames), and the greedy order's value is as a baseline-beating ranking, not as the final design. (3) Every noise-trained model collapses on clean readings. At 40 dB four of the ten whitened channels are noise-dominated (SNR < 1), so the network learned the noise statistics as part of its input, and clean readings are out of distribution. The deployed sensor is noisy, so "trained" is the relevant condition, but the dependence on the level is sharp: evaluating `model_99` at other noise levels (test AP50; greedy / uniform), 46 dB 0.12 / 0.16, 42 dB 0.44 / 0.50, **40 dB 0.57 / 0.63**, 38 dB 0.43 / 0.58, 36 dB 0.27 / 0.46, 34 dB 0.14 / 0.34 — a model trained at one fixed level works only at that level, 2 dB either way costs 0.05–0.13 AP50, and *less* noise is as harmful as more. Fix implemented as `--read_noise_db_range LO HI` (`FilterBank.train_scale_range`): the training noise level is drawn log-uniformly per batch, the whitening and the evaluation keep the nominal level. (4) Follow-up queue (`bash_files/launch_sel_followup.sh`, running): 24 greedy voltages at 40 dB and the 10 greedy voltages trained without noise, then `main_det_compare.py` over all five runs; given (1), a uniform-24 run is the natural addition.

*Follow-up results (2026-10-03; `results/det/compare_sel_all/compare.json`; `sel24g_A` 8 h 01 min, `sel10g_clean_A` 7 h 45 min).* Fixed checkpoints `model_59..99`, 70 test frames, each run under its own training condition (the four noisy runs at 40 dB, the clean run without noise):

| test split, mean of `model_59..99` | greedy 10 | session B's 10 | uniform 10 | **greedy 24** | **greedy 10, no read noise** | PCA-11 (344 clean) | raw 133 |
|---|---|---|---|---|---|---|---|
| AP50 | 0.519 | 0.375 | 0.609 | 0.617 | **0.739** | 0.663 | 0.662 |
| recall50 | 0.617 | 0.583 | 0.704 | 0.721 | **0.817** | 0.766 | 0.749 |
| recall50_op / coverage_recall99_roi_op | 0.563 / 0.614 | 0.513 / 0.592 | 0.639 / 0.668 | 0.676 / 0.732 | **0.769 / 0.808** | – | – |
| matched IoU | 0.782 | 0.766 | 0.801 | 0.783 | **0.817** | – | – |
| `model_best`: test AP50 / recall50 / cov99_roi_op (epoch) | 0.514 / 0.648 / 0.620 (54) | 0.416 / 0.549 / 0.634 (48) | 0.592 / 0.746 / 0.690 (50) | 0.629 / 0.746 / 0.761 (52) | **0.743 / 0.817 / 0.803** (70) | 0.620 / 0.746 / 0.732 (24) | 0.645 / 0.761 / 0.761 (57) |
| val AP50, last-20-epoch mean (best epoch) | 0.446 (0.546) | 0.370 (0.502) | 0.585 (0.654) | 0.558 (0.597) | **0.714 (0.790)** | 0.614 (0.688) | 0.678 (0.718) |

Paired differences (trained condition, AP50): greedy 10 − greedy 24 −0.098 [−0.181, −0.013]; uniform 10 − greedy 24 −0.008 [−0.094, +0.078]; greedy 10 − clean 10 −0.220 [−0.324, −0.120]; uniform 10 − clean 10 −0.130 [−0.228, −0.035]; greedy 24 − clean 10 −0.122 [−0.208, −0.033]. On clean readings every noise-trained model stays collapsed (AP50 0.02–0.10; greedy 24 trained − clean +0.597 [+0.495, +0.693]).

*Reading.* (1) **Read noise, not information, is the cost of a 10-reading sensor.** The same ten greedy voltages trained and evaluated without read noise reach AP50 0.739 / recall 0.817 on fixed checkpoints — 0.22 AP50 above their 40 dB twin and *above* both the raw bands (0.662) and PCA-11 (0.663); on val their plateau is 0.71 against raw's 0.68. Ten whitened readings are therefore a sufficient and better-conditioned input than the 133 raw bands for this detector (the raw run is not a ceiling — its 133 standardised bands are ill-conditioned too, condition number 3.8·10⁴), and the 40 dB floor costs 0.22 AP50 at 10 readings and 0.12 at 24. (2) **Reading budget vs layout.** 24 greedy voltages tie 10 uniform ones (−0.008 [−0.094, +0.078]) and both gain ≈ 0.10 over 10 greedy: what the greedy-10 set lacked was coverage above 1.6 V (the 1.6–1.8 V and 2.3–2.5 V responses reaching 798 nm), which the uniform set has with 10 readings and the greedy order only acquires by step 14; beyond that coverage, more readings bring little at 40 dB. (3) **Design consequence.** The per-reading SNR is the lever: raising it (integration time, repeated reads — 3 dB per doubling — or a lower-noise readout) moves a 10-reading sensor from 0.52 towards 0.74 AP50, whereas going from 10 to 24 readings buys 0.10 and a different 10-voltage layout buys the same. A deployable model must also be trained with the noise level varied (`--read_noise_db_range`): every fixed-level model above works only within ≈ 2 dB of its training level. Next runs, in order: uniform 10 with `--read_noise_db_range 34 50` (robust reference design), the same at a 46–50 dB nominal level (where on the SNR curve the knee lies), and uniform 24 at 40 dB (closes the K-curve on the better layout).

*Noise-free redo (2026-10-04; `bash_files/launch_sel_clean_queue.sh`; `results/det/compare_clean/compare.json`, `results/det/compare_clean_detB/compare.json`).* The 40 dB read-noise floor above is an assumption anchored to one bench figure, not a measurement, so every voltage set was retrained without it (`--read_noise_db 0`, `--pca-channels` 10/10/24, ≈ 7 h 45 min each) and compared with the noise-free references on fixed checkpoints (`model_59..99`; `det_B`, stopped at epoch 91, on `model_59..89`), 70 test frames, paired frame bootstrap with 2000 resamples:

| test split, mean of `model_59..99`, no read noise | AP50 | recall50 | recall50_op | coverage_recall99_roi_op | matched IoU | `model_best` AP50 / recall50 / cov99_roi_op (epoch) |
|---|---|---|---|---|---|---|
| **greedy 10, whitened** (`sel10g_clean_A`) | **0.739** | **0.817** | **0.769** | **0.808** | 0.817 | 0.743 / 0.817 / 0.803 (70) |
| greedy 24, whitened (`sel24g_clean_A`) | 0.719 | 0.777 | 0.715 | 0.738 | 0.818 | 0.680 / 0.761 / 0.732 (42) |
| PCA-11 (all 344 readings, `pca11_A`) | 0.663 | 0.766 | 0.682 | 0.713 | 0.814 | 0.620 / 0.746 / 0.732 (24) |
| raw 133 bands (`raw133_A`) | 0.662 | 0.749 | 0.718 | 0.741 | 0.827 | 0.645 / 0.761 / 0.761 (57) |
| session B's (gate) 10, whitened (`sel10b_clean_A`) | 0.582 | 0.645 | 0.580 | 0.620 | 0.810 | 0.586 / 0.648 / 0.620 (81) |
| uniform 10, whitened (`sel10u_clean_A`) | 0.580 | 0.651 | 0.586 | 0.642 | 0.806 | 0.559 / 0.662 / 0.648 (27) |
| session A, 344 + gate (`det_A`) | 0.466 | 0.561 | 0.513 | 0.569 | 0.738 | 0.444 / 0.563 / 0.549 (`model_89`) |
| session B, gate 10 un-whitened (`det_B`, `model_59..89`) | 0.483 | 0.539 | 0.504 | 0.496 | 0.731 | 0.356 / 0.521 / 0.465 (25) |

Paired AP50 differences [95 % CI]: greedy 10 − gate 10 +0.158 [+0.059, +0.257]; greedy 10 − uniform 10 +0.160 [+0.071, +0.255]; gate 10 − uniform 10 +0.002 [−0.092, +0.097]; greedy 10 − greedy 24 +0.020 [−0.046, +0.083] (coverage_recall99_roi_op +0.070 [+0.014, +0.135]); greedy 24 − uniform 10 +0.140 [+0.064, +0.222]; greedy 10 − raw +0.077 [−0.013, +0.162]; greedy 10 − PCA-11 +0.077 [−0.010, +0.153] (recall50_op +0.087 [+0.009, +0.164], coverage_recall99_roi_op +0.096 [+0.022, +0.175]); PCA-11 − raw +0.001 [−0.094, +0.098]; greedy 10 − det_A +0.273 [+0.156, +0.386]; greedy 10 − det_B +0.254 [+0.121, +0.373] (`model_59..89`); gate 10 whitened − det_B +0.097 [−0.015, +0.211].

*Reading.* (1) **Ten well-chosen EC readings carry what the detector needs.** The greedy set matches the raw bands and PCA-11 (point estimate +0.08 AP50, CIs include 0; better ROI coverage than PCA-11), and beats the original gate method by 0.25–0.27 AP50. The "10 readings beat raw" impression of the `model_best` table is not significant on fixed checkpoints. (2) **Without noise the voltage layout decides:** greedy beats both the gate's set and uniform spacing by 0.16 AP50 (significant), while the gate's set and uniform spacing tie. This reverses the 40 dB ranking (uniform 0.609 > greedy 0.519 > gate 0.375): with noise, coverage survives and fine differences between neighbouring voltages drown; without it, the greedy set's fine design wins. (3) **More readings add nothing without noise** (greedy 24 ≈ greedy 10), unlike at 40 dB (+0.10) — the reading budget is a noise-averaging lever, not an information one. (4) **Whitening's gain is smaller than the `model_best` numbers suggested:** on the same gate voltages, whitened-from-scratch vs `det_B` (un-whitened, warm-started from A) is +0.10 AP50 on fixed checkpoints, CI including 0 (`det_B`'s `model_best` at epoch 25 was a weak checkpoint); the large effect of the format is session A (344 near-duplicate channels + gate, 0.466) vs PCA-11 (0.663) and vs any good 10-set. (5) **Design consequence:** the voltage set should be chosen with the selection objective evaluated at the device's measured per-reading SNR — greedy (per-frame separability) when the SNR is high, broad coverage when it is ≈ 40 dB or worse. Measuring the device's SNR is the deciding step.

## 11. Out of scope

Downsampling, multi-scale/mosaic augmentation, other YOLO sizes or the `-p2` head (no pretrained weights), DDP tuning beyond Spec_Occu's pattern, Stage 2 implementation.
