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

**Control (`raw133_A`, `--raw-bands`: the 133 raw bands of 400–800 nm straight into YOLO26s, identity projection, no gate, W&B `ku0jkcha`):** running; its val/test numbers against session A are appended here when finished.

## 11. Out of scope

Downsampling, multi-scale/mosaic augmentation, other YOLO sizes or the `-p2` head (no pretrained weights), DDP tuning beyond Spec_Occu's pattern, Stage 2 implementation.
