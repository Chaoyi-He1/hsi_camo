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
- **Checkpoint selection by val recall@IoU 0.5** (the zoom-in stage must not miss objects); AP50 and matched-IoU reported.
- **Validation split:** 28 of the 279 train ids held out (fixed seed 0, list saved to `data_loader/splits/det_val_ids.json`); 70 test frames used only for the final report.
- **Augmentation:** random horizontal / vertical flips (boxes flipped accordingly). Nothing else in Stage 1.
- **Code style:** Spec_Occu / Diffu conventions (DETR-style `util/misc.py`, `main_*.py` with `get_args_parser()` + `main(args)`, `train_eval/train_eval_*.py`, `models/build_*`, `weights/<exp>/`, `runs/`, `cfg/*.yaml`, `results_<name>.txt`).

## 4. Code layout

```
main_det.py                        entry: usage docstring (single GPU / torchrun); get_args_parser(); main(args)
cfg/det.yaml                       lr, lrf, epochs, batch_size, accumulate, gate_entropy_weight, top_k, roi_margin, conf/iou thresholds
data_loader/cube_cache.py          build_cube_cache(data_path, split, band_range, num_workers) -> <data_path>/cache_fp16/<split>/<id>.npy ; also __main__
data_loader/boxes.py               boxes_from_mask(mask, min_area=100) -> [K, 4] xyxy px
data_loader/frames_dataset.py      HyperCOD_frames(Dataset), det_collate_fn
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
- `cache_path(data_path, split, name)` helper used by the frames dataset; the frames dataset asserts the cache exists and prints the build command otherwise (no silent 45-minute build inside a training script).

### 5.3 `data_loader/boxes.py`
- `boxes_from_mask(mask: bool [H, W], min_area=100) -> np.ndarray [K, 4] float32 xyxy` (x1, y1 inclusive; x2, y2 exclusive pixel edges), via `scipy.ndimage.label`; components with area < `min_area` dropped; `K = 0` allowed (empty array `[0, 4]`).
- `boxes_to_yolo(boxes, H, W) -> [K, 4] normalised cx, cy, w, h` for the loss; `flip_boxes(boxes, H, W, horizontal, vertical)`.

### 5.4 `data_loader/frames_dataset.py`
```python
class HyperCOD_frames(Dataset.Dataset):
    def __init__(self, data_path, split='train', ids=None, band_range=(400., 800.), pad_to=32,
                 flip=True, filter_path=None, num_filters=30, filter_select='all', filter_voltages=None,
                 filter_norm='l1', stats_path=None, seed=None)
```
- Reuses `HyperCOD_data` machinery for the filter matrix and stats: internally builds a `HyperCOD_data(split, use_filter=True, filter_select=..., norm='p99z', band_range=...)` once and exposes `filter_bank_tensors()` → `(R [n_bands, N] float32, channel_mean [N], channel_std [N], selected_voltages [N])` for the model; the frames dataset itself only returns cube tensors.
- `ids`: explicit list of sample ids (train / val split); `None` = all ids of the split.
- `__getitem__(idx)` → dict `{'img': fp16 tensor [n_bands, Hp, Wp] (zero-padded to multiples of pad_to at the bottom/right), 'scale': float (p99 scale of the frame), 'boxes': float32 tensor [K, 4] xyxy px in the padded frame, 'name': str, 'orig_size': (H, W)}`; training flips applied to `img` and `boxes` with probability 0.5 each (h and v).
- `det_collate_fn(batch)` → `{'img': [B, n_bands, Hp, Wp] fp16, 'scale': [B] float32, 'batch_idx': [ΣK], 'cls': [ΣK, 1] zeros, 'bboxes': [ΣK, 4] normalised cx,cy,w,h (÷ Wp, Hp), 'boxes_xyxy': list of [K, 4] px, 'names': list[str]}` — the `img/batch_idx/cls/bboxes` keys are exactly what `model.loss` reads.
- Val split file: `data_loader/splits/det_val_ids.json` = 28 ids drawn once with `random.Random(0).sample(sorted train ids, 28)`, committed; `make_det_splits(data_path)` writes it if missing.

### 5.5 `models/filter_bank.py`
```python
class FilterBank(nn.Module):
    def __init__(self, R, channel_mean, channel_std, weight_vector=True, init_logits=None)
    def forward(self, x_fp16 [B, n_bands, H, W], scale [B]) -> y [B, N, H, W]   # fp16 under autocast
    @property weights -> w = N * softmax(theta) [N]            (ones when weight_vector=False)
    def entropy(self) -> scalar penalty term  H(softmax theta) / log N  (in [0, 1])
    def ranking(self) -> indices sorted by w descending
```
- Buffers: `R [n_bands, N]`, `mean [N]`, `std [N]` (float32); the projection runs as `torch.einsum('bn,bchw->bnhw', R.T, x / scale.view(B,1,1,1))` in autocast (fp16 inputs, fp32 accumulation), then `(y − mean) / std`, then `y * w.view(1, N, 1, 1)`.
- Test: equals the dataloader's `use_filter=True, norm='p99z'` output on the same crop (rtol 1e-2 in fp16).

### 5.6 `models/ec_yolo.py`
- `load_pretrained_yolo(weights='yolo26s.pt')` → the ultralytics `DetectionModel` (downloads to `weights/pretrained/` if missing).
- `replace_first_conv(model, n_channels)`: new `Conv2d(n_channels, 32, 3, 2, 1, bias=False)`, weight = pretrained RGB kernel tiled to `n_channels` (`repeat(1, ceil(N/3), 1, 1)[:, :N]`) × `3/N`; replaces `model.model[0].conv`.
- `build_ec_yolo(args, frames_dataset)` → `nn.Module` `ECYolo(filter_bank, yolo)` with `forward(img, scale)` = `yolo(filter_bank(img, scale))` and `loss(batch)` = `yolo.loss({**batch, 'img': filter_bank(batch['img'], batch['scale'])})` + `λ_H · filter_bank.entropy()` (returned as an extra item `gate_entropy`). `model.args = get_cfg()` set for the loss gains. Session B: `weight_vector=False`.
- `select_top_k(filter_bank, k=10)` → indices, voltages, weights (also written as `gate_ranking.csv`: rank, voltage, weight, index).
- `slice_to_channels(ecyolo, idx)` → new `ECYolo` with `R[:, idx]`, `mean/std[idx]`, first conv `W[:, idx] * w[idx]`, no weight vector; test: identical detector output on a random input before/after slicing when the dropped channels have `w = 0`.
- `decode_predictions(out, conf_thres, iou_thres, max_det)` → per image `[M, 6]` xyxy, conf, cls via `ultralytics.utils.ops.non_max_suppression`.

### 5.7 `train_eval/train_eval_det.py`
- `train_one_epoch(model, data_loader, optimizer, device, epoch, max_norm, scaler, accumulate, logger)` — `MetricLogger(delimiter="; ")` with meters `loss, box_loss, cls_loss, l1_loss, gate_entropy, lr`; `torch.cuda.amp.autocast(enabled=scaler is not None)`; gradient accumulation over `accumulate` steps; grad-norm clipping; logger scalars per optimizer step.
- `evaluate(model, data_loader, device, conf_thres=0.001, iou_thres=0.6, logger, epoch, tag)` — decodes predictions, matches to GT greedily by IoU ≥ 0.5 (one prediction per GT), computes **recall@0.5** (matched GT / all GT), **AP50** (single-class, all-points interpolation over confidence), **matched IoU** (mean IoU of matched pairs), plus mean detections per image; returns a dict; logs scalars and, for the first 4 val frames, an image with GT (green) and predicted (red) boxes over a false-colour composite of three responses.

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
cache_fp16/<split>/<id>.npy  ──np.load──> img fp16 [133, 1680, 1240] ──pad──> [133, 1696, 1248] ──flip──┐
GT png ──boxes_from_mask──> [K,4] xyxy ──pad/flip──> boxes ──> batch_idx / cls / bboxes (norm xywh)     │
                                                                                                        ▼ GPU
FilterBank: y = w ⊙ ((Rᵀ(img/scale) − m)/s)  ──> [B, 344, 1696, 1248] fp16 ──> YOLO26s (first conv 344→32) ──> model.loss
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
- `HyperCOD_frames`: padded shape `(133, 64, 64)` for the 48×40 fixture with `pad_to=32`, boxes inside the padded frame, `scale` equals the loader's, flips move boxes consistently (checked against a flipped mask), `det_collate_fn` produces the loss keys with the right shapes and normalisation.
- `FilterBank` ≡ dataloader (`use_filter=True, norm='p99z', filter_norm='l1'`) on the same window; `weights` sum to N; `entropy() ∈ [0, 1]`; `ranking()` sorted.
- `replace_first_conv`: shape `(32, N, 3, 3)`, kernel tiling + `3/N` scaling; `slice_to_channels` preserves outputs when dropped channels have zero weight; `select_top_k` returns k unique indices and writes the CSV.
- `train_eval_det`: recall/AP50/matched-IoU on hand-made predictions (perfect → 1/1/1; none → 0/0/nan; half matched → 0.5); one `train_one_epoch` + `evaluate` step on a tiny `yolo26n.yaml` (`ch=133, nc=1`, random init) with the fixture frames on CPU — runs end to end, loss finite, checkpoint dict has the required keys.
- `TrainLogger`: with `wandb` disabled writes TensorBoard scalars to a tmp `runs/`; with a forced `CommError` falls back to offline without raising.

## 10. Open items

- **W&B entity:** the API key works but the account has no team/entity yet (`wandb.init` fails with "entity not specified" / "entity … not found"). The user creates a team on wandb.ai and passes `--wandb-entity <team>` (also stored in `cfg/det.yaml`); until then the logger runs offline.
- **Weight-vector parameterisation:** `N·softmax(θ)` + entropy penalty is the default; plain `(1, N, 1, 1)` parameter + L1 is a two-line alternative if preferred.
- **Stage 2 (segmentation on ROIs):** separate spec; consumes `results/det/rois_*.json` and the existing crop loader with explicit windows; repeats all-344 → top-10.

## 11. Out of scope

Downsampling, multi-scale/mosaic augmentation, other YOLO sizes or the `-p2` head (no pretrained weights), DDP tuning beyond Spec_Occu's pattern, Stage 2 implementation.
