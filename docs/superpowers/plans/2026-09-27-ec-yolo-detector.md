# EC-filter YOLO26 Detector (Stage 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A YOLO26s detector fed with all EC filter responses through a trainable per-channel weight vector (session A), a top-10 voltage selection from that vector, and a re-trained 10-channel detector (session B) that proposes camouflage regions for Stage 2 — with mask-based box-quality metrics, TensorBoard + W&B logging, in the Spec_Occu code style.

**Architecture:** `HyperCOD_data` (existing loader) gains a 400–800 nm band window, an fp16 full-frame cache, id subsets and fp16 output; `det_collate_fn` turns its masks into YOLO boxes. On the GPU a `FilterBank` module computes `w ⊙ standardize(Rᵀx)` and feeds a pretrained YOLO26s whose first conv is rebuilt for N input channels; the loss is ultralytics' own `model.loss` plus an entropy penalty on `w` and an asymmetric containment hinge. `main_det.py` runs sessions A/B with DETR-style `util/misc.py` utilities, `train_eval/train_eval_det.py`, checkpoints in `weights/`, logs in `runs/` + W&B.

**Tech Stack:** Python 3.12, PyTorch 2.11 (cu128), ultralytics 8.4.164 (YOLO26), numpy/scipy/h5py/Pillow, wandb 0.30, tensorboard, pytest. Interpreter: `/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python` (alias `PY` below).

## Global Constraints

- Work in the worktree `/data/chaoyi_he/hsi_camo/.claude/worktrees/det` (branch `worktree-det`); run everything from there with `PY=/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi_camo/bin/python`.
- Cube band centres `np.linspace(400, 1000, 200)`; `.mat` key `hypercube`, h5py shape `(200, W, H)`; default band window **400–800 nm = bands 0–132 (133 bands)**; `N_BANDS = 200` stays "bands in the file".
- Filter: `EC_filterV3.mat`, 401 λ × 351 V, dead zone 0.25–0.31 V excluded → **344 usable voltages**; aligned response zero outside the sensor range, never min–max rescaled; `filter_norm='l1'` default.
- Normalisation: `norm='p99'` = cube ÷ (windowed p99 / n_bands); `norm='p99z'` = p99 + per-channel z-score from `band_stats_train_<lo>_<hi>.npz`; **the detector input is `norm='p99'` and standardisation happens after the projection inside `FilterBank`**.
- Weight vector `w = N · softmax(θ)`, applied as `y * w.view(1, N, 1, 1)`; entropy penalty `gate_entropy_weight · H(softmax θ)/log N`.
- YOLO26s pretrained `yolo26s.pt` (`https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt`), `nc = 1`; first conv `Conv2d(3, 32, 3, 2, 1, bias=False)` → `Conv2d(N, 32, 3, 2, 1, bias=False)` initialised by tiling the RGB kernel and scaling by `3/N`; `model.args = get_cfg()`; loss = `model.loss(batch)` with keys `img, batch_idx, cls, bboxes` (normalised cx,cy,w,h); `loss_items = {box_loss, cls_loss, l1_loss}`; criterion `E2ELoss` (`one2many`, `one2one`, each with `.bbox_loss`, and `.update()` per epoch); eval output `out[0]` = `[B, 4+nc, A]` → `ultralytics.utils.nms.non_max_suppression(pred, conf_thres, iou_thres, nc=1, max_det=300)`.
- Frames 1680 × 1240 padded (bottom/right) to 1696 × 1248 on the GPU; boxes in native pixels; no downsampling.
- Metrics: mask coverage (raw + ROI margin), coverage recall @ 0.99, tightness, centre offset, recall@IoU 0.5, AP50, matched IoU; `model_best` = coverage recall @ 0.99 (raw), tie-break mean tightness.
- Containment loss `contain_weight · Σ_edges relu(gt_edge − pred_edge)/gt_size` inside `ContainBboxLoss`, default weight 1.0.
- Logging: `TrainLogger` → TensorBoard `runs/<name>` + W&B `entity='chaoyi-hsi', project='hsi_camo'`, offline fallback on any `wandb` error; main process only.
- Code style: `assert ..., f"..."`, shape comments `# [B, C, H, W]`, triple-single-quote docstrings, `get_args_parser()` / `main(args)`, `MetricLogger(delimiter="; ")`, `utils.save_on_master`, checkpoint dict `{"model","optimizer","scaler","lr_scheduler","epoch","args","selected_indices","selected_voltages"}`.
- Commit after every task; message ends with `Co-Authored-By: Claude <model> <noreply@anthropic.com>` (the model that authored the commit) and `Claude-Session: https://claude.ai/code/session_01QB4DScDwiA2DkVpbDopffV`. Never `git add -A`; never commit `weights/`, `runs/`, `wandb/`, `results/`, `*.pt`, `.superpowers/`, `.claude/`.
- Synthetic fixture (`tests/conftest.py`): `H, W, B = 48, 40, 200`, object `gt[10:16, 20:26]`, train ids `['3','10']`, test id `['7']`, filter file `root/EC_filterV3.mat`, `info[(split, name)] = (cube [200, 40, 48], gt bool [48, 40], p99_200)`.

---

## File structure

| File | Responsibility |
|---|---|
| `data_loader/ec_filter.py` | + `band_indices(band_range)` |
| `data_loader/my_dataset.py` | `HyperCOD_data` + `band_range`, `cache_dir`, `ids`, `out_dtype`, windowed p99, `filter_bank_tensors()` |
| `data_loader/band_stats.py` | band-range aware stats file |
| `data_loader/cube_cache.py` | fp16 full-frame cache builder (+ windowed p99 CSV) |
| `data_loader/boxes.py` | `boxes_from_mask`, `boxes_to_yolo`, `flip_boxes`, `det_collate_fn` |
| `data_loader/det_splits.py` | `make_det_splits`, `load_det_ids` → `data_loader/splits/det_val_ids.json` |
| `util/misc.py`, `util/distributed_util.py` | copied from Spec_Occu |
| `util/logger.py` | `TrainLogger` |
| `models/filter_bank.py` | `FilterBank` |
| `models/ec_yolo.py` | pretrained loading, first-conv surgery, `ContainBboxLoss`, `ECYolo`, `build_ec_yolo`, `select_top_k`, `slice_to_channels`, `decode_predictions` |
| `train_eval/box_metrics.py` | pure-numpy metrics |
| `train_eval/train_eval_det.py` | `train_one_epoch`, `evaluate`, GPU flips/padding |
| `main_det.py`, `cfg/det.yaml` | entry point, hyper-parameters |
| `main_det_rois.py` | ROI export |
| `tests/test_band_window.py`, `tests/test_cube_cache.py`, `tests/test_boxes.py`, `tests/test_logger.py`, `tests/test_filter_bank.py`, `tests/test_ec_yolo.py`, `tests/test_box_metrics.py`, `tests/test_train_eval_det.py`, `tests/test_main_det.py` | tests |

---

### Task 1: Loader band window (step 0) + housekeeping

**Files:**
- Modify: `.gitignore`, `pytest.ini`, `data_loader/ec_filter.py`, `data_loader/my_dataset.py`, `data_loader/band_stats.py`, `tests/test_my_dataset.py`, `tests/test_band_stats.py`
- Create: `tests/test_band_window.py`

**Interfaces:**
- Produces: `band_indices(band_range) -> np.ndarray[int]` (ec_filter); `HyperCOD_data(..., band_range=(400., 800.))` with attributes `band_idx [n_bands]`, `n_bands`, `wavelens [n_bands]`, `band_range`; `read_cube_block` → `[n_bands, cw, ch]`; `in_channels = N | n_bands`; windowed p99 CSV `<data_path>/<split>/intensity map/intensity_p99_<lo>_<hi>.csv` (same columns as the original) built by `compute_window_p99(data_path, split, band_range, num_workers)`; `band_stats.default_stats_path(data_path, band_range)` → `band_stats_train_<lo>_<hi>.npz` (`(400,1000)` keeps `band_stats_train.npz`); `compute_band_stats(..., band_range=...)`, `load_band_stats(path, band_range)`.

- [ ] **Step 1: Housekeeping**

Append to `.gitignore`:
```
.claude/
weights/
runs/
wandb/
results/
*.pt
```
Replace `pytest.ini` with:
```ini
[pytest]
testpaths = tests
pythonpath = .
filterwarnings =
    ignore:This process .* is multi-threaded, use of fork\(\) may lead to deadlocks:DeprecationWarning
```
Run `$PY -m pytest tests/ -q` → `56 passed`, no warnings.

- [ ] **Step 2: Failing tests** — create `tests/test_band_window.py`:

```python
import numpy as np
import pytest

from data_loader.ec_filter import N_BANDS, WAVELENS_200, band_indices
from data_loader.my_dataset import HyperCOD_data
from data_loader.band_stats import default_stats_path, compute_band_stats, load_band_stats
from tests.conftest import H, W


def make(root, **kw):
    kw.setdefault('split', 'train'); kw.setdefault('crop_size', 16); kw.setdefault('norm', 'none')
    kw.setdefault('filter_norm', 'none'); kw.setdefault('seed', 0)
    return HyperCOD_data(data_path=str(root), **kw)


def test_band_indices():
    idx = band_indices((400.0, 800.0))
    assert idx[0] == 0 and idx[-1] == 132 and len(idx) == 133 and np.all(np.diff(idx) == 1)
    assert len(band_indices((400.0, 1000.0))) == N_BANDS
    with pytest.raises(AssertionError, match="band_range"):
        band_indices((900.0, 1200.0))


def test_default_window_is_400_800(synthetic_root):
    root, info, _ = synthetic_root
    ds = make(root, use_filter=False)
    assert ds.band_range == (400.0, 800.0) and ds.n_bands == 133 and ds.in_channels == 133
    np.testing.assert_allclose(ds.wavelens, WAVELENS_200[:133])
    assert ds.sensor_R_matrix.shape == (133, 30) and ds.valid_band_mask.all()
    assert not np.all(ds.sensor_R_matrix == 0, axis=1).any()         # no zero rows inside the window
    img, gt, _ = make(root, split='test', use_filter=False)[0]
    cube = info[('test', '7')][0]                                     # [200, W, H]
    assert img.shape == (133, H, W)
    np.testing.assert_array_equal(img, cube[:133].transpose(0, 2, 1))


def test_full_window_reproduces_200_bands(synthetic_root):
    root, info, _ = synthetic_root
    ds = make(root, use_filter=False, band_range=(400.0, 1000.0))
    assert ds.n_bands == 200 and ds.in_channels == 200 and ds.sensor_R_matrix.shape == (200, 30)
    assert np.all(ds.sensor_R_matrix[133:] == 0.0)
    blk = ds.read_cube_block('3', h0=5, w0=7, ch=16, cw=12)
    np.testing.assert_array_equal(blk, info[('train', '3')][0][:, 7:19, 5:21])


def test_windowed_p99_scale_is_computed_and_cached(synthetic_root):
    root, info, _ = synthetic_root
    csv_path = root / 'train' / 'intensity map' / 'intensity_p99_400_800.csv'
    assert not csv_path.exists()
    ds = make(root, norm='p99', use_filter=False)
    assert csv_path.exists()
    cube = info[('train', '3')][0]                                    # [200, W, H]
    p99_133 = np.percentile(cube[:133].sum(axis=0), 99)
    assert np.isclose(ds.scale['3'], p99_133 / 133, rtol=1e-5)
    mtime = csv_path.stat().st_mtime
    ds2 = make(root, norm='p99', use_filter=False)                    # reuses the csv
    assert csv_path.stat().st_mtime == mtime and ds2.scale == ds.scale
    ds_full = make(root, norm='p99', use_filter=False, band_range=(400.0, 1000.0))
    assert np.isclose(ds_full.scale['3'], info[('train', '3')][2] / 200)


def test_band_stats_are_band_range_aware(synthetic_root, tmp_path):
    root, _, _ = synthetic_root
    assert default_stats_path(str(root), (400.0, 800.0)).endswith('band_stats_train_400_800.npz')
    assert default_stats_path(str(root), (400.0, 1000.0)).endswith('band_stats_train.npz')
    mean, cov = compute_band_stats(str(root), str(tmp_path / 's.npz'), crop_size=0, num_workers=0, band_range=(400.0, 800.0))
    assert mean.shape == (133,) and cov.shape == (133, 133)
    m2, c2 = load_band_stats(str(tmp_path / 's.npz'), band_range=(400.0, 800.0))
    np.testing.assert_array_equal(m2, mean)
    with pytest.raises(AssertionError, match="band_range"):
        load_band_stats(str(tmp_path / 's.npz'), band_range=(400.0, 1000.0))
    ds = make(root, norm='p99z', use_filter=False)                    # default window builds its own stats file
    assert ds.stats_path == default_stats_path(str(root), (400.0, 800.0)) and ds.channel_mean.shape == (133,)
    x = np.concatenate([HyperCOD_data(str(root), norm='p99z', use_filter=False, crop_size=0, filter_norm='none', seed=0)[i][0].reshape(133, -1) for i in range(2)], axis=1)
    np.testing.assert_allclose(x.mean(axis=1), 0.0, atol=1e-4)
    np.testing.assert_allclose(x.std(axis=1), 1.0, rtol=1e-3)
```

Run `$PY -m pytest tests/test_band_window.py -q` → fails (`ImportError: cannot import name 'band_indices'`).

- [ ] **Step 3: `ec_filter.py`** — append:

```python
def band_indices(band_range, wavelens=WAVELENS_200):
    '''Indices of the cube bands whose centre lies inside [lo, hi] nm (inclusive, contiguous).'''
    lo, hi = float(band_range[0]), float(band_range[1])
    idx = np.where((wavelens >= lo - 1e-6) & (wavelens <= hi + 1e-6))[0]
    assert len(idx) > 0, f"band_range {band_range} contains no cube band (cube covers {wavelens[0]:.0f}-{wavelens[-1]:.0f} nm)"
    return idx
```

- [ ] **Step 4: `my_dataset.py`** — changes:

1. Import `band_indices` from `ec_filter`; import `default_stats_path` (now takes `band_range`) from `band_stats`.
2. Signature: add `band_range=(400.0, 800.0)` after `norm`; store `self.band_range = (float(band_range[0]), float(band_range[1]))`, `self.band_idx = band_indices(self.band_range)`, `self.n_bands = len(self.band_idx)`, `self.wavelens = WAVELENS_200[self.band_idx].copy()` (replacing the `WAVELENS_200.copy()` line); `self.stats_path = stats_path if stats_path is not None else default_stats_path(data_path, self.band_range)`.
3. `read_cube_block`: `b0, b1 = int(self.band_idx[0]), int(self.band_idx[-1]) + 1`; hyperslab `f[HYPERCUBE_KEY][b0:b1, w0:w0 + cw, h0:h0 + ch]`; assert shape `(self.n_bands, cw, ch)`.
4. `in_channels = ... else self.n_bands`.
5. `load_intensity_scale()`:
```python
    def load_intensity_scale(self):
        '''
        Per-sample scale = p99 of the band sum inside the window / n_bands. For the full 400-1000 nm window this is
        intensity_p99_valid / 200 from the dataset's csv; for a narrower window the 99th percentile of the windowed
        band sum is computed once per sample from the cubes and cached next to the original csv.
        '''
        if self.n_bands == N_BANDS:
            csv_path = os.path.join(self.intensity_path, 'intensity_p99_summary.csv')
            assert os.path.exists(csv_path), f"{csv_path} not found (needed for norm='p99')"
        else:
            csv_path = window_p99_csv_path(self.data_path, self.split, self.band_range)
            if not os.path.exists(csv_path):
                compute_window_p99(self.data_path, self.split, self.band_range,
                                   num_workers=0 if len(self.img_name) < 8 else min(8, os.cpu_count() or 1))
        scale = {}
        with open(csv_path, newline='') as f:
            for row in csv.DictReader(f):
                scale[row['sample_id']] = float(row['intensity_p99_valid']) / self.n_bands
        for name in self.img_name:
            assert name in scale, f"sample {name} missing from {csv_path}"
            assert scale[name] > 0, f"non-positive intensity p99 for sample {name}"
        return scale
```
6. Module-level helpers (after the class, before `image_collate_fn`):
```python
def window_p99_csv_path(data_path, split, band_range):
    lo, hi = int(round(band_range[0])), int(round(band_range[1]))
    return os.path.join(data_path, split, 'intensity map', f'intensity_p99_{lo}_{hi}.csv')


def compute_window_p99(data_path, split, band_range, num_workers=8):
    '''Write intensity_p99_<lo>_<hi>.csv (same columns as intensity_p99_summary.csv) from the windowed band sums.'''
    ds = HyperCOD_data(data_path, split=split, use_filter=False, norm='none', crop_size=0, band_range=band_range,
                       filter_norm='none')
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=image_collate_fn)
    rows = []
    print(f"Computing {band_range} nm p99 intensity for {len(ds)} {split} cubes...")
    for img, _, name in loader:                                   # img [1, n_bands, H, W]
        s = img[0].sum(dim=0).numpy()                             # [H, W] windowed band sum
        rows.append((name[0], s.shape[0], s.shape[1], ds.n_bands, s.size,
                     float(np.percentile(s, 99)), float(s.min()), float(s.mean()), float(s.max())))
    csv_path = window_p99_csv_path(data_path, split, band_range)
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['sample_id', 'mat_path', 'intensity_map_path', 'height', 'width', 'bands', 'n_valid',
                    'intensity_p99_valid', 'intensity_min_valid', 'intensity_mean_valid', 'intensity_max_valid'])
        for name, h, wd, b, n, p99, mn, me, mx in rows:
            w.writerow([name, '', '', h, wd, b, n, p99, mn, me, mx])
    print(f"Saved {csv_path}")
    return csv_path
```
7. `load_channel_stats`: pass `band_range=self.band_range` to `compute_band_stats` and `load_band_stats`; shapes use `self.n_bands`.
8. `add_dataset_args`: `parser.add_argument('--band_range', type=float, nargs=2, default=[400.0, 800.0], help='wavelength window in nm (default the EC sensor range 400-800; 400 1000 uses all 200 bands)')`; `build_dataset` passes `band_range=tuple(args.band_range)`.

- [ ] **Step 5: `band_stats.py`** — `default_stats_path(data_path, band_range=(400.0, 800.0))`: `band_stats_train.npz` when the window covers all 200 bands (`len(band_indices(band_range)) == N_BANDS`), else `band_stats_train_<lo>_<hi>.npz`. `compute_band_stats(..., band_range=(400.0, 800.0))`: pass `band_range` to the dataset, use `ds.n_bands` for the accumulators, save `band_range=np.array(band_range)`. `load_band_stats(stats_path, band_range=None)`: if given, assert `np.allclose(st['band_range'], band_range)` with message containing `band_range` (files without the key are treated as `(400, 1000)`); assert shapes against `len(band_indices(...))`. `__main__`: `--band_range` two floats.

- [ ] **Step 6: Update existing tests** in `tests/test_my_dataset.py` and `tests/test_band_stats.py` to the new default window: `make()` unchanged (default window); assertions that used 200 bands → `133` (`img.shape == (133, H, W)`, `cube[:133].transpose(0, 2, 1)`, `ds_raw.in_channels == 133`, `sensor_R_matrix.shape == (133, 30)`, `valid_band_mask.sum() == 133` stays, drop `np.all(ds.sensor_R_matrix[133:] == 0.0)`), `test_getitem_filter_uses_only_bands_below_800nm` → assert `ds.sensor_R_matrix.shape[0] == 133`, p99 expectations → windowed p99 (`np.percentile(cube[:133].sum(0), 99) / 133`), `test_intensity_scale` likewise, `test_compute_band_stats_matches_direct` → `cube[:133]` and `/ (p99_133 / 133)` with `band_range=(400., 800.)`, `test_add_dataset_args_defaults` → `args.band_range == [400.0, 800.0]`, `test_p99z_*` shapes `(133,)` / `[:133]`.

- [ ] **Step 7: Run everything** — `$PY -m pytest tests/ -q` → all pass (61 = 56 + 5), no warnings.

- [ ] **Step 8: Commit**
```bash
git add .gitignore pytest.ini data_loader/ec_filter.py data_loader/my_dataset.py data_loader/band_stats.py tests/test_band_window.py tests/test_my_dataset.py tests/test_band_stats.py
git commit -m "Default the loader to the EC sensor window 400-800 nm (133 bands); band-range aware p99 and stats"
```

---

### Task 2: fp16 full-frame cube cache + loader options (`cache_dir`, `ids`, `out_dtype`, `filter_bank_tensors`)

**Files:**
- Create: `data_loader/cube_cache.py`, `tests/test_cube_cache.py`
- Modify: `data_loader/my_dataset.py`

**Interfaces:**
- Consumes: `HyperCOD_data` from Task 1 (`band_idx`, `n_bands`, `read_cube_block`, `compute_window_p99`).
- Produces: `cache_path(cache_dir, split, name) -> str` (`<cache_dir>/<split>/<name>.npy`); `default_cache_dir(data_path) -> <data_path>/cache_fp16`; `build_cube_cache(data_path, split, cache_dir=None, band_range=(400., 800.), num_workers=8, ids=None, overwrite=False) -> list[str]` writing fp16 `[n_bands, H, W]` arrays and the windowed p99 CSV; `HyperCOD_data(..., cache_dir=None, ids=None, out_dtype='float32')`; `HyperCOD_data.filter_bank_tensors() -> (R [n_bands, N] float32, channel_mean [N] float32, channel_std [N] float32, selected_voltages [N])`.

- [ ] **Step 1: Failing tests** — `tests/test_cube_cache.py`:

```python
import os
import numpy as np
import pytest
import torch

from data_loader.cube_cache import build_cube_cache, cache_path, default_cache_dir
from data_loader.my_dataset import HyperCOD_data
from tests.conftest import H, W


def make(root, **kw):
    kw.setdefault('split', 'train'); kw.setdefault('crop_size', 16); kw.setdefault('norm', 'none')
    kw.setdefault('filter_norm', 'none'); kw.setdefault('seed', 0)
    return HyperCOD_data(data_path=str(root), **kw)


def test_build_cache_writes_fp16_bhw_arrays(synthetic_root):
    root, info, _ = synthetic_root
    files = build_cube_cache(str(root), 'train', num_workers=0)
    assert sorted(os.path.basename(f) for f in files) == ['10.npy', '3.npy']
    a = np.load(cache_path(default_cache_dir(str(root)), 'train', '3'), mmap_mode='r')
    assert a.dtype == np.float16 and a.shape == (133, H, W)
    cube = info[('train', '3')][0]                                        # [200, W, H]
    np.testing.assert_allclose(np.asarray(a, dtype=np.float32), cube[:133].transpose(0, 2, 1), rtol=1e-3, atol=1e-5)
    assert (root / 'train' / 'intensity map' / 'intensity_p99_400_800.csv').exists()


def test_build_cache_skips_existing_and_respects_ids(synthetic_root):
    root, _, _ = synthetic_root
    build_cube_cache(str(root), 'train', num_workers=0, ids=['3'])
    p = cache_path(default_cache_dir(str(root)), 'train', '3')
    assert os.path.exists(p) and not os.path.exists(cache_path(default_cache_dir(str(root)), 'train', '10'))
    mtime = os.path.getmtime(p)
    build_cube_cache(str(root), 'train', num_workers=0)                   # fills in 10, leaves 3 untouched
    assert os.path.getmtime(p) == mtime and os.path.exists(cache_path(default_cache_dir(str(root)), 'train', '10'))


def test_dataset_reads_from_cache_identically(synthetic_root):
    root, _, _ = synthetic_root
    build_cube_cache(str(root), 'train', num_workers=0); build_cube_cache(str(root), 'test', num_workers=0)
    cdir = default_cache_dir(str(root))
    for kw in (dict(split='test', use_filter=False), dict(split='test', use_filter=True, num_filters=8),
               dict(split='train', use_filter=False, crop_size=16, obj_crop_prob=1.0, seed=3)):
        a, _, _ = make(root, **kw)[0]
        b, _, _ = make(root, cache_dir=cdir, **kw)[0]
        np.testing.assert_allclose(a, b, rtol=2e-3, atol=1e-4)
    blk = make(root, cache_dir=cdir).read_cube_block('3', h0=5, w0=7, ch=16, cw=12)
    assert blk.shape == (133, 12, 16) and blk.dtype == np.float32


def test_dataset_cache_asserts_when_missing(synthetic_root, tmp_path):
    root, _, _ = synthetic_root
    with pytest.raises(AssertionError, match="cube_cache"):
        make(root, cache_dir=str(tmp_path / 'nocache'))[0]


def test_ids_and_out_dtype(synthetic_root):
    root, _, _ = synthetic_root
    ds = make(root, ids=['10'], use_filter=False, out_dtype='float16')
    assert ds.img_name == ['10'] and len(ds) == 1
    img, gt, name = ds[0]
    assert img.dtype == np.float16 and gt.dtype == np.float32 and name == '10'
    with pytest.raises(AssertionError, match="ids"):
        make(root, ids=['3', '99'])


def test_filter_bank_tensors_match_p99z_instance(synthetic_root):
    root, _, _ = synthetic_root
    ds = make(root, use_filter=False, norm='p99', num_filters=12, filter_norm='l1')
    R, mean, std, volts = ds.filter_bank_tensors()
    ref = make(root, use_filter=True, norm='p99z', num_filters=12, filter_norm='l1')
    np.testing.assert_allclose(R, ref.sensor_R_matrix); np.testing.assert_allclose(mean, ref.channel_mean)
    np.testing.assert_allclose(std, ref.channel_std); np.testing.assert_allclose(volts, ref.selected_voltages)
    assert R.dtype == np.float32 and R.shape == (133, 12)
```

Run → `ImportError` (no `data_loader.cube_cache`).

- [ ] **Step 2: `data_loader/cube_cache.py`**

```python
import os
import argparse
import numpy as np
import torch

from data_loader.ec_filter import band_indices

CACHE_DIRNAME = 'cache_fp16'


def default_cache_dir(data_path):
    return os.path.join(data_path, CACHE_DIRNAME)


def cache_path(cache_dir, split, name):
    return os.path.join(cache_dir, split, f'{name}.npy')


def build_cube_cache(data_path, split, cache_dir=None, band_range=(400.0, 800.0), num_workers=8, ids=None, overwrite=False):
    '''
    Convert every <split>/hyperspectral/<id>.mat into <cache_dir>/<split>/<id>.npy holding the windowed raw cube
    as float16 [n_bands, H, W] (0.55 GB per frame for 133 bands), so full frames load in ~0.3 s instead of 9.5 s.
    Also writes the windowed p99 csv used by norm='p99'. Existing files are skipped unless overwrite=True.
    '''
    from data_loader.my_dataset import HyperCOD_data, image_collate_fn, compute_window_p99, window_p99_csv_path
    cache_dir = cache_dir if cache_dir is not None else default_cache_dir(data_path)
    os.makedirs(os.path.join(cache_dir, split), exist_ok=True)
    ds = HyperCOD_data(data_path, split=split, ids=ids, use_filter=False, norm='none', crop_size=0,
                       band_range=band_range, filter_norm='none', out_dtype='float16')
    todo = [i for i, name in enumerate(ds.img_name) if overwrite or not os.path.exists(cache_path(cache_dir, split, name))]
    print(f"Caching {len(todo)}/{len(ds)} {split} cubes to {cache_dir} ({ds.n_bands} bands, float16)...")
    written = []
    if todo:
        loader = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, todo), batch_size=1, shuffle=False,
                                             num_workers=num_workers, collate_fn=image_collate_fn)
        for k, (img, _, name) in enumerate(loader):                       # img [1, n_bands, H, W] fp16
            out = cache_path(cache_dir, split, name[0])
            np.save(out, img[0].numpy().astype(np.float16))
            written.append(out)
            if (k + 1) % 25 == 0 or k + 1 == len(todo):
                print(f"  {k + 1}/{len(todo)} cubes")
    if not os.path.exists(window_p99_csv_path(data_path, split, band_range)) and len(band_indices(band_range)) < 200:
        compute_window_p99(data_path, split, band_range, num_workers=num_workers)
    return written


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build the fp16 full-frame cube cache')
    parser.add_argument('--data_path', type=str, default='/data2/chaoyi/HyperCOD/Raw data')
    parser.add_argument('--split', type=str, default='all', choices=['train', 'test', 'all'])
    parser.add_argument('--cache_dir', type=str, default=None)
    parser.add_argument('--band_range', type=float, nargs=2, default=[400.0, 800.0])
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--ids', type=str, nargs='*', default=None)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    for split in (['train', 'test'] if args.split == 'all' else [args.split]):
        build_cube_cache(args.data_path, split, args.cache_dir, tuple(args.band_range), args.num_workers, args.ids, args.overwrite)
```

- [ ] **Step 3: `my_dataset.py` options**

Signature: add `cache_dir=None, ids=None, out_dtype='float32'`; store `self.cache_dir`, `self.out_dtype = np.dtype(out_dtype)`; assert `self.out_dtype in (np.float16, np.float32)`.
After `self.img_name = sorted(names, key=int)`:
```python
        if ids is not None:
            ids = [str(i) for i in ids]
            missing = [i for i in ids if i not in self.img_name]
            assert not missing, f"ids not found in {self.hsi_path}: {missing}"
            self.img_name = sorted(ids, key=int)
```
`read_cube_block`:
```python
        if self.cache_dir is not None:
            from data_loader.cube_cache import cache_path
            p = cache_path(self.cache_dir, self.split, name)
            assert os.path.exists(p), f"cache file {p} missing; build it with: python -m data_loader.cube_cache --data_path '{self.data_path}' --split {self.split}"
            arr = np.load(p, mmap_mode='r')                               # [n_bands, H, W] fp16
            assert arr.shape == (self.n_bands, self.H, self.W) and arr.dtype == np.float16, \
                f"cache {p} has {arr.shape} {arr.dtype}, expected ({self.n_bands}, {self.H}, {self.W}) float16"
            blk = np.ascontiguousarray(arr[:, h0:h0 + ch, w0:w0 + cw], dtype=np.float32)   # [n_bands, ch, cw]
            return np.ascontiguousarray(blk.transpose(0, 2, 1))                            # -> [n_bands, cw, ch] like the h5 path
        ... existing h5 code ...
```
`__getitem__` last lines: `img = np.ascontiguousarray(img.transpose(0, 2, 1), dtype=self.out_dtype)` and the p99z standardisation computed in float32 then cast: `img = ((img.astype(np.float32) - mean) / std).astype(self.out_dtype)`.
`filter_bank_tensors()`:
```python
    def filter_bank_tensors(self):
        '''(R [n_bands, N], channel_mean [N], channel_std [N], selected_voltages [N]) for the GPU FilterBank, independent of use_filter/norm.'''
        mu, cov = load_band_stats(self.stats_path, band_range=self.band_range) if os.path.exists(self.stats_path) else (None, None)
        if mu is None:
            from data_loader.band_stats import compute_band_stats
            crop_size = STATS_CROP_SIZE if min(self.H, self.W) >= STATS_CROP_SIZE else 0
            compute_band_stats(self.data_path, self.stats_path, crop_size=crop_size, num_workers=0 if len(self) < 8 else min(8, os.cpu_count() or 1),
                               filter_path=self.filter_path, band_range=self.band_range)
            mu, cov = load_band_stats(self.stats_path, band_range=self.band_range)
        R = self.sensor_R_matrix.astype(np.float64)
        mean = R.T @ mu; std = np.sqrt(np.maximum(np.einsum('bn,bc,cn->n', R, cov, R), 0.0))
        return self.sensor_R_matrix.astype(np.float32), mean.astype(np.float32), std.astype(np.float32), self.selected_voltages.copy()
```
(`load_channel_stats` is refactored to reuse the same stats loading; keep behaviour.)

`add_dataset_args`: `--cache_dir` (default None), `--out_dtype {float32,float16}` (default float32); `build_dataset` passes both.

- [ ] **Step 4: Run** `$PY -m pytest tests/ -q` → all pass. **Step 5: Commit** `git add data_loader/cube_cache.py data_loader/my_dataset.py tests/test_cube_cache.py; git commit -m "Add fp16 full-frame cube cache and loader options cache_dir/ids/out_dtype/filter_bank_tensors"`.

---

### Task 3: Boxes from masks, YOLO collate, validation split

**Files:**
- Create: `data_loader/boxes.py`, `data_loader/det_splits.py`, `tests/test_boxes.py`

**Interfaces:**
- Produces: `boxes_from_mask(mask, min_area=100, return_labels=False) -> boxes [K,4] float32 xyxy (x2,y2 exclusive) | (boxes, labels [H,W] int32, ids list[int])` — **the default 100 px is the real-data floor that drops JPEG specks and must not be lowered; the 36-px fixture object is handled by passing `min_area=10` in tests**; `boxes_to_yolo(boxes, H, W) -> [K,4] float32 normalised cx,cy,w,h`; `yolo_to_xyxy(b, H, W)`; `flip_boxes(boxes, H, W, horizontal, vertical)`; `expand_box(box, margin=1.5, min_size=256, H, W)`; `det_collate_fn(batch, min_area=100) -> dict(img [B,C,H,W] tensor (dtype of the dataset), masks list of bool [H,W], batch_idx [ΣK], cls [ΣK,1], bboxes [ΣK,4], boxes_xyxy list of [K,4], names list[str])`; `make_det_splits(data_path, n_val=28, seed=0, path=None) -> (train_ids, val_ids)` writing `data_loader/splits/det_val_ids.json`; `load_det_ids(data_path, path=None) -> (train_ids, val_ids)`.

- [ ] **Step 1: Failing tests** — `tests/test_boxes.py`:

```python
import json
import numpy as np
import pytest
import torch

from data_loader.boxes import boxes_from_mask, boxes_to_yolo, yolo_to_xyxy, flip_boxes, expand_box, det_collate_fn
from data_loader.det_splits import make_det_splits, load_det_ids
from data_loader.my_dataset import HyperCOD_data
from tests.conftest import H, W, OBJ_SLICE


def test_boxes_from_mask_single_and_specks():
    m = np.zeros((H, W), bool); m[OBJ_SLICE] = True; m[40, 5] = True      # 6x6 object (36 px) + 1-px speck
    b = boxes_from_mask(m, min_area=10)                                      # the fixture object is smaller than the real-data default of 100 px
    np.testing.assert_array_equal(b, [[20, 10, 26, 16]]); assert b.dtype == np.float32
    b2, labels, ids = boxes_from_mask(m, min_area=10, return_labels=True)
    assert labels.shape == (H, W) and ids == [1] and (labels == 1).sum() == 36
    assert boxes_from_mask(m).shape == (0, 4)                                # default min_area=100 drops the 36-px fixture object
    assert boxes_from_mask(np.zeros((H, W), bool)).shape == (0, 4)


def test_boxes_from_mask_two_objects_sorted_by_position():
    m = np.zeros((H, W), bool); m[2:5, 30:34] = True; m[20:30, 2:8] = True
    b = boxes_from_mask(m, min_area=5)
    np.testing.assert_array_equal(b, [[30, 2, 34, 5], [2, 20, 8, 30]])


def test_yolo_conversion_and_flips():
    b = np.array([[20, 10, 26, 16]], np.float32)
    y = boxes_to_yolo(b, H, W)
    np.testing.assert_allclose(y, [[23 / W, 13 / H, 6 / W, 6 / H]])
    np.testing.assert_allclose(yolo_to_xyxy(y, H, W), b)
    np.testing.assert_array_equal(flip_boxes(b, H, W, horizontal=True, vertical=False), [[W - 26, 10, W - 20, 16]])
    np.testing.assert_array_equal(flip_boxes(b, H, W, horizontal=False, vertical=True), [[20, H - 16, 26, H - 10]])
    np.testing.assert_array_equal(flip_boxes(flip_boxes(b, H, W, True, True), H, W, True, True), b)


def test_expand_box_margin_min_size_and_clipping():
    np.testing.assert_array_equal(expand_box(np.array([20, 10, 26, 16.]), margin=1.5, min_size=0, H=H, W=W), [18.5, 8.5, 27.5, 17.5])
    e = expand_box(np.array([20, 10, 26, 16.]), margin=1.0, min_size=20, H=H, W=W)
    assert e[2] - e[0] == 20 and e[3] - e[1] == 20
    e = expand_box(np.array([0, 0, 10, 10.]), margin=3.0, min_size=0, H=H, W=W)
    assert e[0] == 0 and e[1] == 0 and e[2] <= W and e[3] <= H


def test_det_collate_fn_builds_yolo_batch(synthetic_root):
    root, _, _ = synthetic_root
    ds = HyperCOD_data(str(root), split='train', use_filter=False, norm='p99', crop_size=0, out_dtype='float16', filter_norm='none')
    batch = det_collate_fn([ds[0], ds[1]], min_area=10)                       # fixture objects are 36 px
    assert batch['img'].shape == (2, 133, H, W) and batch['img'].dtype == torch.float16
    assert batch['batch_idx'].tolist() == [0.0, 1.0] and batch['cls'].shape == (2, 1) and batch['bboxes'].shape == (2, 4)
    np.testing.assert_allclose(batch['bboxes'][0].numpy(), [23 / W, 13 / H, 6 / W, 6 / H], rtol=1e-6)
    assert batch['names'] == ['3', '10'] and len(batch['masks']) == 2 and batch['masks'][0].dtype == bool
    np.testing.assert_array_equal(batch['boxes_xyxy'][1], [[20, 10, 26, 16]])


def test_make_and_load_det_splits(synthetic_root, tmp_path):
    root, _, _ = synthetic_root
    p = tmp_path / 'det_val_ids.json'
    train_ids, val_ids = make_det_splits(str(root), n_val=1, seed=0, path=str(p))
    assert sorted(train_ids + val_ids) == ['10', '3'] and len(val_ids) == 1
    assert json.load(open(p))['val_ids'] == val_ids
    assert load_det_ids(str(root), path=str(p)) == (train_ids, val_ids)
    assert make_det_splits(str(root), n_val=1, seed=0, path=str(p)) == (train_ids, val_ids)   # idempotent
```

- [ ] **Step 2: `data_loader/boxes.py`**

```python
import numpy as np
import torch
from scipy import ndimage


def boxes_from_mask(mask, min_area=100, return_labels=False):
    '''
    One box per connected foreground component with area >= min_area (JPEG ringing specks are smaller).
    Returns float32 [K, 4] xyxy with x2/y2 exclusive, ordered by component label (row-major first pixel);
    with return_labels also the label map [H, W] int32 and the kept label ids.
    '''
    labels, n = ndimage.label(mask)
    boxes, ids = [], []
    if n > 0:
        sizes = ndimage.sum(mask, labels, index=np.arange(1, n + 1))
        for i, sl in enumerate(ndimage.find_objects(labels), start=1):
            if sl is None or sizes[i - 1] < min_area:
                continue
            boxes.append([sl[1].start, sl[0].start, sl[1].stop, sl[0].stop])
            ids.append(i)
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    return (boxes, labels.astype(np.int32), ids) if return_labels else boxes


def boxes_to_yolo(boxes, H, W):
    '''xyxy pixels -> normalised (cx, cy, w, h) as the ultralytics loss expects.'''
    b = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    out = np.empty_like(b)
    out[:, 0] = (b[:, 0] + b[:, 2]) / 2 / W; out[:, 1] = (b[:, 1] + b[:, 3]) / 2 / H
    out[:, 2] = (b[:, 2] - b[:, 0]) / W;     out[:, 3] = (b[:, 3] - b[:, 1]) / H
    return out


def yolo_to_xyxy(b, H, W):
    b = np.asarray(b, dtype=np.float32).reshape(-1, 4)
    cx, cy, w, h = b[:, 0] * W, b[:, 1] * H, b[:, 2] * W, b[:, 3] * H
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


def flip_boxes(boxes, H, W, horizontal, vertical):
    b = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
    if horizontal:
        b[:, [0, 2]] = W - b[:, [2, 0]]
    if vertical:
        b[:, [1, 3]] = H - b[:, [3, 1]]
    return b


def expand_box(box, margin, min_size, H, W):
    '''Grow a box about its centre by `margin` (1.5 = 50 % larger sides), at least min_size on each side, clipped to the frame.'''
    x1, y1, x2, y2 = [float(v) for v in box]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = max((x2 - x1) * margin, min_size), max((y2 - y1) * margin, min_size)
    return np.array([max(0.0, cx - w / 2), max(0.0, cy - h / 2), min(float(W), cx + w / 2), min(float(H), cy + h / 2)], dtype=np.float32)


def det_collate_fn(batch, min_area=100):
    '''
    HyperCOD_data (img [C, H, W], gt [1, H, W], name) tuples -> the batch dict ultralytics' model.loss reads
    (img, batch_idx, cls, bboxes normalised cx,cy,w,h) plus the masks and pixel boxes for the metrics.
    Use functools.partial(det_collate_fn, min_area=...) as the DataLoader collate_fn to change the box floor.
    '''
    imgs, gts, names = list(zip(*batch))
    img = torch.from_numpy(np.stack(imgs, axis=0))                        # [B, C, H, W], dataset dtype
    H, W = img.shape[-2:]
    masks, boxes_xyxy, batch_idx, bboxes = [], [], [], []
    for i, gt in enumerate(gts):
        m = gt[0] > 0.5                                                   # [H, W] bool
        b = boxes_from_mask(m, min_area=min_area)                         # [K, 4]
        masks.append(m); boxes_xyxy.append(b)
        batch_idx.append(np.full(len(b), i, dtype=np.float32)); bboxes.append(boxes_to_yolo(b, H, W))
    batch_idx = torch.from_numpy(np.concatenate(batch_idx)) if batch_idx else torch.zeros(0)
    bboxes = torch.from_numpy(np.concatenate(bboxes)).reshape(-1, 4) if bboxes else torch.zeros(0, 4)
    return {'img': img, 'masks': masks, 'batch_idx': batch_idx, 'cls': torch.zeros(len(batch_idx), 1),
            'bboxes': bboxes, 'boxes_xyxy': boxes_xyxy, 'names': list(names)}
```

- [ ] **Step 3: `data_loader/det_splits.py`**

```python
import os
import json
import random

SPLIT_PATH = os.path.join(os.path.dirname(__file__), 'splits', 'det_val_ids.json')


def _train_ids(data_path):
    hsi = os.path.join(data_path, 'train', 'hyperspectral')
    return sorted([os.path.splitext(f)[0] for f in os.listdir(hsi) if f.endswith('.mat')], key=int)


def make_det_splits(data_path, n_val=28, seed=0, path=None):
    '''Hold out n_val training ids for validation (fixed seed) and save them; idempotent once the file exists.'''
    path = path or SPLIT_PATH
    ids = _train_ids(data_path)
    if os.path.exists(path):
        return load_det_ids(data_path, path)
    val_ids = sorted(random.Random(seed).sample(ids, n_val), key=int)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump({'seed': seed, 'n_val': n_val, 'val_ids': val_ids}, f, indent=1)
    return [i for i in ids if i not in val_ids], val_ids


def load_det_ids(data_path, path=None):
    path = path or SPLIT_PATH
    assert os.path.exists(path), f"split file {path} missing; run make_det_splits"
    val_ids = json.load(open(path))['val_ids']
    ids = _train_ids(data_path)
    assert all(v in ids for v in val_ids), f"val ids not in {data_path}/train: {[v for v in val_ids if v not in ids]}"
    return [i for i in ids if i not in val_ids], val_ids
```

- [ ] **Step 4: Run** `$PY -m pytest tests/test_boxes.py -q` → 6 passed; full suite passes. Generate the real split once: `$PY -c "from data_loader.det_splits import make_det_splits; print(make_det_splits('/data2/chaoyi/HyperCOD/Raw data')[1])"` → 28 ids written to `data_loader/splits/det_val_ids.json` (commit it).
- [ ] **Step 5: Commit** `git add data_loader/boxes.py data_loader/det_splits.py data_loader/splits/det_val_ids.json tests/test_boxes.py; git commit -m "Add boxes from masks, YOLO collate and the detector validation split"`.

---

### Task 4: Spec_Occu utilities + `TrainLogger` (TensorBoard + W&B)

**Files:**
- Create: `util/__init__.py` (empty), `util/misc.py`, `util/distributed_util.py` (copied), `util/logger.py`, `tests/test_logger.py`

**Interfaces:**
- Produces: `util.misc` (`SmoothedValue`, `MetricLogger`, `reduce_dict`, `init_distributed_mode`, `save_on_master`, `is_main_process`, `get_rank`, `get_world_size`), `util.distributed_util.Custom_DistributedSampler`; `TrainLogger(args, cfg)` with `scalar(tag, value, step)`, `scalars(dict, step, prefix='')`, `histogram(tag, values, step)`, `image(tag, hwc_uint8, step)`, `table(tag, columns, rows, step)`, `finish()`, attribute `wandb_run` (None when disabled/offline-failed) and `mode` in `{'online','offline','disabled'}`.

- [ ] **Step 1: Copy the utilities verbatim** (they import only stdlib/torch/numpy):
```bash
mkdir -p util && touch util/__init__.py
cp /data2/chaoyi/Spec_Occu/util/misc.py util/misc.py
cp /data2/chaoyi/Spec_Occu/util/distributed_util.py util/distributed_util.py
$PY -c "import util.misc as u, util.distributed_util as d; print(u.MetricLogger, d.Custom_DistributedSampler)"
```
Expected: both class reprs print. (`distributed_util.py` imports `from .misc import get_rank, get_world_size` — unchanged.)

- [ ] **Step 2: Failing tests** — `tests/test_logger.py`:

```python
import os
import types
import numpy as np
import pytest

from util.logger import TrainLogger


def _args(tmp_path, **kw):
    a = types.SimpleNamespace(name='t', wandb=False, wandb_entity='chaoyi-hsi', wandb_project='hsi_camo',
                              runs_dir=str(tmp_path / 'runs'), wandb_dir=str(tmp_path / 'wandb'), rank=0, distributed=False)
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def test_tensorboard_only_logs_scalars_and_images(tmp_path):
    lg = TrainLogger(_args(tmp_path), cfg={'lr': 1e-4})
    assert lg.mode == 'disabled' and lg.wandb_run is None
    lg.scalar('train/loss', 1.0, 0); lg.scalars({'a': 1.0, 'b': 2.0}, 0, prefix='val/')
    lg.histogram('w', np.arange(10.0), 0); lg.image('img', np.zeros((8, 8, 3), np.uint8), 0)
    lg.table('rank', ['voltage', 'w'], [[1.0, 0.5]], 0)               # no-op without wandb
    lg.finish()
    assert any(f.startswith('events.out.tfevents') for f in os.listdir(tmp_path / 'runs' / 't'))


def test_wandb_failure_falls_back_to_offline(tmp_path, monkeypatch):
    import wandb
    calls = []
    def fake_init(**kw):
        calls.append(kw.get('mode'))
        if kw.get('mode') != 'offline':
            raise wandb.errors.CommError('entity not found')
        return types.SimpleNamespace(log=lambda *a, **k: None, finish=lambda: None, url='offline')
    monkeypatch.setattr(wandb, 'init', fake_init)
    lg = TrainLogger(_args(tmp_path, wandb=True), cfg={})
    assert calls == [None, 'offline'] and lg.mode == 'offline' and lg.wandb_run is not None
    lg.scalar('x', 1.0, 0); lg.finish()


def test_non_main_process_is_noop(tmp_path):
    lg = TrainLogger(_args(tmp_path, rank=1, distributed=True), cfg={})
    lg.scalar('x', 1.0, 0); lg.finish()
    assert not (tmp_path / 'runs').exists()
```

- [ ] **Step 3: `util/logger.py`**

```python
import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


class TrainLogger(object):
    '''
    Fan every scalar/histogram/image out to TensorBoard (runs/<name>) and Weights & Biases from one place,
    main process only. W&B: entity/project from args; any wandb error (no entity, no network) falls back to
    offline mode so training never blocks; `wandb sync wandb/offline-run-*` pushes those runs later.
    '''

    def __init__(self, args, cfg):
        self.enabled = int(getattr(args, 'rank', 0)) == 0
        self.tb = None
        self.wandb_run = None
        self.mode = 'disabled'
        if not self.enabled:
            return
        self.tb = SummaryWriter(log_dir=os.path.join(getattr(args, 'runs_dir', 'runs'), args.name))
        if getattr(args, 'wandb', False):
            import wandb
            config = {**{k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in vars(args).items()}, **cfg}
            common = dict(entity=args.wandb_entity, project=args.wandb_project, name=args.name, config=config,
                          dir=getattr(args, 'wandb_dir', 'wandb'))
            os.makedirs(common['dir'], exist_ok=True)
            for mode in (None, 'offline'):
                try:
                    self.wandb_run = wandb.init(mode=mode, **common) if mode else wandb.init(**common)
                    self.mode = mode or 'online'
                    break
                except Exception as e:                                    # CommError, UsageError, network errors
                    print(f"wandb.init({mode or 'online'}) failed: {type(e).__name__}: {str(e)[:120]}")
            if self.mode == 'offline':
                print(f"W&B running offline; sync later with: wandb sync {common['dir']}/wandb/offline-run-*")

    def scalar(self, tag, value, step):
        if not self.enabled:
            return
        value = float(value)
        self.tb.add_scalar(tag, value, step)
        if self.wandb_run is not None:
            self.wandb_run.log({tag: value}, step=step)

    def scalars(self, values, step, prefix=''):
        for k, v in values.items():
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                self.scalar(prefix + k, v, step)

    def histogram(self, tag, values, step):
        if not self.enabled:
            return
        v = np.asarray(values, dtype=np.float32).ravel()
        self.tb.add_histogram(tag, v, step)
        if self.wandb_run is not None:
            import wandb
            self.wandb_run.log({tag: wandb.Histogram(v)}, step=step)

    def image(self, tag, hwc_uint8, step):
        if not self.enabled:
            return
        img = np.asarray(hwc_uint8)
        self.tb.add_image(tag, img, step, dataformats='HWC')
        if self.wandb_run is not None:
            import wandb
            self.wandb_run.log({tag: wandb.Image(img)}, step=step)

    def table(self, tag, columns, rows, step):
        if self.enabled and self.wandb_run is not None:
            import wandb
            self.wandb_run.log({tag: wandb.Table(columns=list(columns), data=[list(r) for r in rows])}, step=step)

    def finish(self):
        if self.tb is not None:
            self.tb.flush(); self.tb.close()
        if self.wandb_run is not None:
            self.wandb_run.finish()
```

- [ ] **Step 4: Run** `$PY -m pytest tests/test_logger.py -q` → 3 passed. **Step 5: Commit** `git add util/ tests/test_logger.py; git commit -m "Add Spec_Occu utilities and a TensorBoard + W&B TrainLogger with offline fallback"`.

---

### Task 5: `FilterBank` (GPU projection + standardisation + weight vector)

**Files:**
- Create: `models/__init__.py` (empty), `models/filter_bank.py`, `tests/test_filter_bank.py`

**Interfaces:**
- Consumes: `HyperCOD_data.filter_bank_tensors()` (Task 2).
- Produces:
```python
class FilterBank(nn.Module):
    def __init__(self, R, channel_mean, channel_std, weight_vector=True, init_logits=None)   # numpy or tensors
    forward(x [B, n_bands, H, W]) -> y [B, N, H, W]        # x already p99-scaled by the loader
    weights -> tensor [N]  (N*softmax(theta), or ones)      # property
    entropy() -> scalar tensor in [0, 1]                    # H(softmax theta)/log N, 0 when no weight vector
    ranking() -> LongTensor [N] indices sorted by weight descending
    n_channels -> int
```

- [ ] **Step 1: Failing tests** — `tests/test_filter_bank.py`:

```python
import numpy as np
import torch
import pytest

from models.filter_bank import FilterBank
from data_loader.my_dataset import HyperCOD_data


def make(root, **kw):
    kw.setdefault('split', 'test'); kw.setdefault('crop_size', 0); kw.setdefault('filter_norm', 'l1')
    kw.setdefault('num_filters', 12); kw.setdefault('seed', 0)
    return HyperCOD_data(data_path=str(root), **kw)


def test_filter_bank_matches_dataloader_channels(synthetic_root):
    root, _, _ = synthetic_root
    ds_in = make(root, use_filter=False, norm='p99', out_dtype='float16')       # what the detector loader returns
    ref = make(root, use_filter=True, norm='p99z')                              # loader-side channels, standardised
    fb = FilterBank(*ds_in.filter_bank_tensors()[:3], weight_vector=False)
    x = torch.from_numpy(ds_in[0][0])[None]                                     # [1, 133, H, W] fp16
    y = fb(x.float())[0].numpy()
    np.testing.assert_allclose(y, ref[0][0], rtol=2e-2, atol=2e-2)              # fp16 input vs float32 path
    assert fb.n_channels == 12 and torch.equal(fb.weights, torch.ones(12)) and float(fb.entropy()) == 0.0


def test_weight_vector_softmax_scaling_and_ranking():
    R = np.random.RandomState(0).randn(133, 5).astype(np.float32)
    fb = FilterBank(R, np.zeros(5, np.float32), np.ones(5, np.float32), weight_vector=True,
                    init_logits=np.array([0.0, 2.0, -1.0, 0.5, 0.0], np.float32))
    w = fb.weights
    assert torch.isclose(w.sum(), torch.tensor(5.0)) and (w > 0).all()
    assert fb.ranking().tolist()[0] == 1 and fb.ranking().tolist()[-1] == 2
    assert 0.0 < float(fb.entropy()) < 1.0
    assert float(FilterBank(R, np.zeros(5), np.ones(5), init_logits=np.zeros(5)).entropy()) == pytest.approx(1.0)
    x = torch.randn(2, 133, 4, 6)
    y = fb(x)
    assert y.shape == (2, 5, 4, 6)
    expected = torch.einsum('bn,bchw->bnhw', torch.from_numpy(R).T, x) * w.view(1, 5, 1, 1)
    torch.testing.assert_close(y, expected, rtol=1e-5, atol=1e-5)
    assert fb.theta.requires_grad and list(fb.parameters()) == [fb.theta]


def test_filter_bank_is_fp16_safe_under_autocast():
    if not torch.cuda.is_available():
        pytest.skip("cuda")
    fb = FilterBank(np.random.rand(133, 8).astype(np.float32) / 133, np.zeros(8, np.float32), np.ones(8, np.float32)).cuda()
    x = torch.rand(1, 133, 64, 64, device='cuda', dtype=torch.float16)
    with torch.autocast('cuda', dtype=torch.float16):
        y = fb(x)
    assert y.dtype == torch.float16 and torch.isfinite(y).all()
```

- [ ] **Step 2: `models/filter_bank.py`**

```python
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FilterBank(nn.Module):
    '''
    Filter responses on the GPU: y_n = sum_b R[b, n] * x[b] (the raw HSI integrated against the aligned filter
    matrix, x already p99-scaled), standardised per channel with the training statistics, then multiplied by
    the trainable weight vector w = N * softmax(theta) applied as (B, N, H, W) * (1, N, 1, 1).
    The entropy of softmax(theta) (normalised to [0, 1]) is exposed as a sparsity penalty; ranking() gives the
    channel order used to pick the top-k voltages.
    '''

    def __init__(self, R, channel_mean, channel_std, weight_vector=True, init_logits=None):
        super(FilterBank, self).__init__()
        R = torch.as_tensor(np.asarray(R), dtype=torch.float32)                   # [n_bands, N]
        assert R.ndim == 2, f"R must be [n_bands, N], got {tuple(R.shape)}"
        self.register_buffer('R_t', R.t().contiguous())                           # [N, n_bands]
        self.register_buffer('mean', torch.as_tensor(np.asarray(channel_mean), dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer('std', torch.as_tensor(np.asarray(channel_std), dtype=torch.float32).view(1, -1, 1, 1))
        assert (self.std > 0).all(), "channel_std must be positive"
        self.n_channels = R.shape[1]
        self.weight_vector = weight_vector
        if weight_vector:
            init = torch.zeros(self.n_channels) if init_logits is None else torch.as_tensor(np.asarray(init_logits), dtype=torch.float32)
            assert init.shape == (self.n_channels,), f"init_logits must have shape ({self.n_channels},)"
            self.theta = nn.Parameter(init.clone())                                # [N]

    @property
    def weights(self):
        if not self.weight_vector:
            return torch.ones(self.n_channels, device=self.R_t.device)
        return self.n_channels * F.softmax(self.theta, dim=0)                       # [N], mean 1

    def entropy(self):
        if not self.weight_vector:
            return torch.zeros((), device=self.R_t.device)
        p = F.softmax(self.theta, dim=0)
        return -(p * torch.log(p + 1e-12)).sum() / math.log(self.n_channels)

    def ranking(self):
        return torch.argsort(self.weights.detach(), descending=True)

    def forward(self, x):
        y = torch.einsum('nb,bchw->bnhw', self.R_t.to(x.dtype), x)                # [B, N, H, W] filter responses
        y = (y - self.mean.to(y.dtype)) / self.std.to(y.dtype)                    # per-channel standardisation
        return y * self.weights.to(y.dtype).view(1, -1, 1, 1)                     # (B, N, H, W) * (1, N, 1, 1)
```

- [ ] **Step 3: Run** `$PY -m pytest tests/test_filter_bank.py -q` → 3 passed. **Step 4: Commit** `git add models/ tests/test_filter_bank.py; git commit -m "Add the GPU FilterBank with the trainable per-channel weight vector"`.

---

### Task 6: `models/ec_yolo.py` — YOLO26 with N-channel first conv, containment loss, A→B slicing, decoding

**Files:**
- Create: `models/ec_yolo.py`, `tests/test_ec_yolo.py`

**Interfaces:**
- Consumes: `FilterBank` (Task 5); `det_collate_fn` batches (Task 3).
- Produces:
```python
PRETRAINED_URL = 'https://github.com/ultralytics/assets/releases/download/v8.4.0/{variant}.pt'
download_pretrained(variant='yolo26s', weights_dir='weights/pretrained') -> str path
adapt_first_conv_weight(w_rgb [c1,3,k,k], n_channels) -> [c1,n_channels,k,k]   # tile RGB kernel, scale 3/n
build_detection_model(variant, n_channels, pretrained_path=None, nc=1, epochs=100) -> (DetectionModel, n_matched, n_total)
containment_term(pred_xyxy [n,4], target_xyxy [n,4], weight [n,1], target_scores_sum) -> scalar tensor
class ContainBboxLoss(BboxLoss): __init__(reg_max, contain_weight); attribute last_contain (float)
class ECYolo(nn.Module): __init__(filter_bank, yolo, gate_entropy_weight=0.0, contain_weight=0.0, stride=32)
    pad_to_stride(img) -> (img_padded, (H, W, Hp, Wp)); forward(img) -> yolo output; loss(batch) -> (total scalar, items dict)
    attach_criterion() ; end_epoch()  # calls criterion.update()
build_ec_yolo(args, dataset) -> ECYolo (CPU)          # args: yolo_variant, pretrained ('auto'|path|'none'), gate_entropy_weight, contain_weight, epochs, session
select_top_k(ecyolo, k, csv_path=None) -> (indices list[int], weights np [N], voltages np [N])   # csv: rank,index,voltage,weight
slice_to_channels(ecyolo, indices, variant, pretrained_path=None) -> ECYolo   # 10-channel model initialised from A
decode_predictions(out, conf_thres=0.001, iou_thres=0.6, max_det=300, nc=1) -> list[np.ndarray [M,6] xyxy,conf,cls]
```

- [ ] **Step 1: Failing tests** — `tests/test_ec_yolo.py`:

```python
import os
import types
import numpy as np
import pytest
import torch

from ultralytics.utils.loss import BboxLoss
from models.filter_bank import FilterBank
from models.ec_yolo import (adapt_first_conv_weight, build_detection_model, containment_term, ContainBboxLoss, ECYolo,
                            build_ec_yolo, select_top_k, slice_to_channels, decode_predictions, download_pretrained)
from data_loader.boxes import det_collate_fn
from data_loader.my_dataset import HyperCOD_data
from tests.conftest import H, W


def _tiny(n_channels, weight_vector=True, logits=None, contain_weight=1.0, gate_w=0.1):
    dm, _, _ = build_detection_model('yolo26n', n_channels, pretrained_path=None, nc=1, epochs=2)
    rng = np.random.RandomState(0)
    fb = FilterBank(rng.rand(133, n_channels).astype(np.float32) / 133, np.zeros(n_channels, np.float32),
                    np.ones(n_channels, np.float32), weight_vector=weight_vector, init_logits=logits)
    return ECYolo(fb, dm, gate_entropy_weight=gate_w, contain_weight=contain_weight)


def _batch(root):
    ds = HyperCOD_data(str(root), split='train', use_filter=False, norm='p99', crop_size=0, out_dtype='float16', filter_norm='none')
    return det_collate_fn([ds[0], ds[1]])


def test_adapt_first_conv_weight_tiles_and_scales():
    w = torch.arange(2 * 3 * 3 * 3, dtype=torch.float32).view(2, 3, 3, 3)
    a = adapt_first_conv_weight(w, 7)
    assert a.shape == (2, 7, 3, 3)
    torch.testing.assert_close(a[:, 3], w[:, 0] * (3 / 7)); torch.testing.assert_close(a[:, 6], w[:, 0] * (3 / 7))
    torch.testing.assert_close(adapt_first_conv_weight(w, 3), w)


def test_build_detection_model_with_pretrained_weights():
    path = download_pretrained('yolo26s', weights_dir='weights/pretrained')
    assert os.path.exists(path)
    dm, n_matched, n_total = build_detection_model('yolo26s', 133, pretrained_path=path, nc=1)
    assert n_matched >= 690 and n_total == 708
    conv = dm.model[0].conv
    assert tuple(conv.weight.shape) == (32, 133, 3, 3) and conv.stride == (2, 2) and conv.bias is None
    from ultralytics import YOLO
    w_rgb = YOLO(path).model.float().state_dict()['model.0.conv.weight']
    torch.testing.assert_close(conv.weight.detach(), adapt_first_conv_weight(w_rgb, 133))
    assert dm.args.epochs == 100 and dm.model[-1].nc == 1


def test_containment_term_hinge():
    tb = torch.tensor([[10., 10., 20., 30.]]); w = torch.ones(1, 1); s = torch.tensor(1.0)
    assert float(containment_term(torch.tensor([[8., 9., 22., 31.]]), tb, w, s)) == 0.0          # contains -> no penalty
    assert float(containment_term(torch.tensor([[12., 10., 20., 30.]]), tb, w, s)) == pytest.approx(0.2)   # 2 px of 10 px width cut
    assert float(containment_term(torch.tensor([[10., 10., 20., 25.]]), tb, w, s)) == pytest.approx(0.25)  # 5 px of 20 px height cut
    assert float(containment_term(torch.tensor([[12., 10., 20., 25.]]), tb, w, s)) == pytest.approx(0.45)


def test_contain_bbox_loss_reduces_to_bbox_loss_when_disabled():
    a, b = ContainBboxLoss(1, contain_weight=0.0), BboxLoss(1)
    assert a.contain_weight == 0.0 and a.last_contain == 0.0 and isinstance(a, BboxLoss) and type(a).forward is not BboxLoss.forward


def test_ecyolo_pads_and_computes_loss_on_cpu(synthetic_root):
    root, _, _ = synthetic_root
    model = _tiny(6).train()
    batch = _batch(root)
    img_p, (h, w, hp, wp) = model.pad_to_stride(batch['img'])
    assert (h, w, hp, wp) == (H, W, 64, 64) and img_p.shape == (2, 133, 64, 64)
    total, items = model.loss(batch)
    assert torch.isfinite(total) and total.requires_grad
    assert {'box_loss', 'cls_loss', 'l1_loss', 'gate_entropy', 'contain_loss'} <= set(items)
    total.backward()
    assert model.filter_bank.theta.grad is not None and model.yolo.model[0].conv.weight.grad.shape == (16, 6, 3, 3)
    model.end_epoch()                                                           # criterion.update() must not fail


def test_select_top_k_and_slice_preserve_outputs(tmp_path):
    logits = np.array([5.0, 4.0, 3.0, -30.0, -30.0, -30.0], np.float32)          # channels 3-5 have ~zero weight
    model = _tiny(6, logits=logits).eval()
    idx, weights, volts = select_top_k(model, 3, csv_path=str(tmp_path / 'rank.csv'), voltages=np.arange(6) * 0.1)
    assert idx == [0, 1, 2] and weights.shape == (6,) and (tmp_path / 'rank.csv').read_text().splitlines()[0] == 'rank,index,voltage,weight'
    small = slice_to_channels(model, idx, 'yolo26n').eval()
    assert small.filter_bank.n_channels == 3 and not small.filter_bank.weight_vector
    assert tuple(small.yolo.model[0].conv.weight.shape) == (16, 3, 3, 3)
    x = torch.rand(1, 133, 64, 64)
    with torch.no_grad():
        a, b = model(x), small(x)
    torch.testing.assert_close(a[0], b[0], rtol=1e-3, atol=1e-3)


def test_decode_predictions_returns_xyxy_conf_cls():
    model = _tiny(4, contain_weight=0.0).eval()
    with torch.no_grad():
        out = model(torch.rand(2, 133, 64, 64))
    dets = decode_predictions(out, conf_thres=0.0, iou_thres=0.6, max_det=10)
    assert len(dets) == 2 and all(d.shape[1] == 6 and d.dtype == np.float32 and len(d) <= 10 for d in dets)


def test_build_ec_yolo_from_args_and_dataset(synthetic_root):
    root, _, _ = synthetic_root
    ds = HyperCOD_data(str(root), split='train', use_filter=False, norm='p99', crop_size=0, num_filters=8, filter_norm='l1', out_dtype='float16')
    args = types.SimpleNamespace(yolo_variant='yolo26n', pretrained='none', gate_entropy_weight=0.05, contain_weight=1.0, epochs=3, session='A')
    m = build_ec_yolo(args, ds)
    assert m.filter_bank.n_channels == 8 and m.filter_bank.weight_vector and m.yolo.args.epochs == 3
    args.session = 'B'
    assert not build_ec_yolo(args, ds).filter_bank.weight_vector
```

- [ ] **Step 2: `models/ec_yolo.py`**

```python
import os
import csv
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.tasks import DetectionModel
from ultralytics.cfg import get_cfg
from ultralytics.utils.loss import BboxLoss
from ultralytics.utils.nms import non_max_suppression

from models.filter_bank import FilterBank

PRETRAINED_URL = 'https://github.com/ultralytics/assets/releases/download/v8.4.0/{variant}.pt'
FIRST_CONV_KEY = 'model.0.conv.weight'


def download_pretrained(variant='yolo26s', weights_dir='weights/pretrained'):
    os.makedirs(weights_dir, exist_ok=True)
    path = os.path.join(weights_dir, f'{variant}.pt')
    if not os.path.exists(path):
        print(f"Downloading {variant}.pt to {path}...")
        torch.hub.download_url_to_file(PRETRAINED_URL.format(variant=variant), path, progress=False)
    return path


def adapt_first_conv_weight(w_rgb, n_channels):
    '''Tile the pretrained RGB kernel [c1, 3, k, k] over n_channels and scale by 3/n so the stem's activation scale is kept.'''
    c1, c_in, k1, k2 = w_rgb.shape
    reps = int(math.ceil(n_channels / c_in))
    return (w_rgb.repeat(1, reps, 1, 1)[:, :n_channels] * (c_in / n_channels)).contiguous()  # [c1, n_channels, k, k]


def build_detection_model(variant, n_channels, pretrained_path=None, nc=1, epochs=100):
    '''
    YOLO26 DetectionModel whose first conv takes n_channels. Every pretrained tensor with a matching shape is
    copied (695/708 for yolo26s, nc=1); the first conv is initialised from the RGB kernel (adapt_first_conv_weight);
    the class-head outputs are re-initialised because nc changed. model.args carries the loss gains (get_cfg()).
    '''
    dm = DetectionModel(f'{variant}.yaml', ch=n_channels, nc=nc, verbose=False)
    own = dm.state_dict()
    n_matched = 0
    if pretrained_path is not None:
        from ultralytics import YOLO
        sd = YOLO(pretrained_path).model.float().state_dict()
        matched = {k: v for k, v in sd.items() if k in own and v.shape == own[k].shape}
        n_matched = len(matched)
        dm.load_state_dict(matched, strict=False)
        if FIRST_CONV_KEY in sd and sd[FIRST_CONV_KEY].shape != own[FIRST_CONV_KEY].shape:
            dm.model[0].conv.weight.data.copy_(adapt_first_conv_weight(sd[FIRST_CONV_KEY], n_channels))
    dm.args = get_cfg()
    dm.args.epochs = int(epochs)
    return dm, n_matched, len(own)


def containment_term(pred_xyxy, target_xyxy, weight, target_scores_sum):
    '''Hinge on how far the GT box protrudes from the prediction, per edge, normalised by the GT width/height.'''
    gw = (target_xyxy[:, 2] - target_xyxy[:, 0]).clamp(min=1e-6)
    gh = (target_xyxy[:, 3] - target_xyxy[:, 1]).clamp(min=1e-6)
    protrude = (F.relu(pred_xyxy[:, 0] - target_xyxy[:, 0]) + F.relu(target_xyxy[:, 2] - pred_xyxy[:, 2])) / gw \
             + (F.relu(pred_xyxy[:, 1] - target_xyxy[:, 1]) + F.relu(target_xyxy[:, 3] - pred_xyxy[:, 3])) / gh   # [n]
    return (protrude.unsqueeze(-1) * weight).sum() / target_scores_sum


class ContainBboxLoss(BboxLoss):
    '''ultralytics BboxLoss (CIoU + L1 for YOLO26) plus the asymmetric containment hinge, added to the IoU term.'''

    def __init__(self, reg_max, contain_weight=1.0):
        super(ContainBboxLoss, self).__init__(reg_max)
        self.contain_weight = float(contain_weight)
        self.last_contain = 0.0

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask, imgsz, stride):
        loss_iou, loss_dfl = super(ContainBboxLoss, self).forward(pred_dist, pred_bboxes, anchor_points, target_bboxes,
                                                                  target_scores, target_scores_sum, fg_mask, imgsz, stride)
        self.last_contain = 0.0
        if self.contain_weight > 0 and bool(fg_mask.any()):
            weight = target_scores[fg_mask].sum(-1, keepdim=True)                       # [n, 1] same weighting as the IoU term
            contain = containment_term(pred_bboxes[fg_mask], target_bboxes[fg_mask], weight, target_scores_sum)
            self.last_contain = float(contain.detach())
            loss_iou = loss_iou + self.contain_weight * contain
        return loss_iou, loss_dfl


class ECYolo(nn.Module):
    '''FilterBank -> YOLO26 with an N-channel first conv. loss(batch) = ultralytics loss + gate entropy penalty.'''

    def __init__(self, filter_bank, yolo, gate_entropy_weight=0.0, contain_weight=0.0, stride=32):
        super(ECYolo, self).__init__()
        self.filter_bank = filter_bank
        self.yolo = yolo
        self.gate_entropy_weight = float(gate_entropy_weight)
        self.contain_weight = float(contain_weight)
        self.stride = int(stride)

    def pad_to_stride(self, img):
        H, W = img.shape[-2:]
        Hp, Wp = math.ceil(H / self.stride) * self.stride, math.ceil(W / self.stride) * self.stride
        return F.pad(img, (0, Wp - W, 0, Hp - H)), (H, W, Hp, Wp)                    # zero-pad bottom/right

    def attach_criterion(self):
        crit = self.yolo.init_criterion()
        if self.contain_weight > 0:
            reg_max = self.yolo.model[-1].reg_max
            device = next(self.yolo.parameters()).device
            for sub in (getattr(crit, 'one2many', None), getattr(crit, 'one2one', None), crit if hasattr(crit, 'bbox_loss') else None):
                if sub is not None and hasattr(sub, 'bbox_loss'):
                    sub.bbox_loss = ContainBboxLoss(reg_max, self.contain_weight).to(device)
        self.yolo.criterion = crit
        return crit

    def forward(self, img=None, batch=None):
        '''img -> YOLO raw output; or, with batch=..., the training loss (so DDP's forward wraps the loss and syncs gradients).'''
        if batch is not None:
            return self.loss(batch)
        img_p, _ = self.pad_to_stride(img)
        return self.yolo(self.filter_bank(img_p))

    def loss(self, batch):
        if getattr(self.yolo, 'criterion', None) is None:
            self.attach_criterion()
        img_p, (H, W, Hp, Wp) = self.pad_to_stride(batch['img'])
        bboxes = batch['bboxes'].clone().to(img_p.device)
        if len(bboxes):
            bboxes[:, [0, 2]] *= W / Wp                                              # normalised coords follow the padding
            bboxes[:, [1, 3]] *= H / Hp
        y = self.filter_bank(img_p)                                                  # [B, N, Hp, Wp]
        loss, items = self.yolo.loss({'img': y, 'batch_idx': batch['batch_idx'].to(y.device),
                                      'cls': batch['cls'].to(y.device), 'bboxes': bboxes})
        ent = self.filter_bank.entropy()
        total = loss.sum() + self.gate_entropy_weight * ent
        out = {k: float(v) for k, v in (items.items() if isinstance(items, dict) else enumerate(items))}
        out['gate_entropy'] = float(ent.detach())
        o2o = getattr(self.yolo.criterion, 'one2one', self.yolo.criterion)
        out['contain_loss'] = float(getattr(getattr(o2o, 'bbox_loss', None), 'last_contain', 0.0))
        return total, out

    def end_epoch(self):
        crit = getattr(self.yolo, 'criterion', None)
        if crit is not None and hasattr(crit, 'update'):
            crit.update()


def build_ec_yolo(args, dataset):
    '''Session A: all selected voltages + weight vector; session B: fixed channels, no weight vector (initialised by slice_to_channels).'''
    R, mean, std, volts = dataset.filter_bank_tensors()
    fb = FilterBank(R, mean, std, weight_vector=(args.session == 'A'))
    pretrained = None if args.pretrained == 'none' else (download_pretrained(args.yolo_variant) if args.pretrained == 'auto' else args.pretrained)
    yolo, n_matched, n_total = build_detection_model(args.yolo_variant, fb.n_channels, pretrained, nc=1, epochs=args.epochs)
    print(f"{args.yolo_variant}: {fb.n_channels} input channels, pretrained tensors reused {n_matched}/{n_total}")
    model = ECYolo(fb, yolo, gate_entropy_weight=args.gate_entropy_weight, contain_weight=args.contain_weight)
    model.selected_voltages = np.asarray(volts)
    return model


def select_top_k(ecyolo, k, csv_path=None, voltages=None):
    '''Rank channels by the weight vector; write rank,index,voltage,weight; return the k best indices (weight-descending).'''
    weights = ecyolo.filter_bank.weights.detach().cpu().numpy()
    voltages = np.asarray(voltages if voltages is not None else getattr(ecyolo, 'selected_voltages', np.arange(len(weights))))
    order = np.argsort(-weights)
    assert 1 <= k <= len(order), f"top_k={k} must be in [1, {len(order)}]"
    if csv_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f); w.writerow(['rank', 'index', 'voltage', 'weight'])
            for r, i in enumerate(order, start=1):
                w.writerow([r, int(i), float(voltages[i]), float(weights[i])])
    return [int(i) for i in order[:k]], weights, voltages


def slice_to_channels(ecyolo, indices, variant, pretrained_path=None):
    '''Keep only `indices`: FilterBank columns, first-conv input channels scaled by the gate (w_j * W[:, idx_j]); no weight vector.'''
    fb, idx = ecyolo.filter_bank, list(indices)
    w = fb.weights.detach().cpu()
    R = fb.R_t.t().cpu().numpy()[:, idx]
    new_fb = FilterBank(R, fb.mean.view(-1).cpu().numpy()[idx], fb.std.view(-1).cpu().numpy()[idx], weight_vector=False)
    yolo, _, _ = build_detection_model(variant, len(idx), pretrained_path=None, nc=ecyolo.yolo.model[-1].nc, epochs=ecyolo.yolo.args.epochs)
    sd = {k: v for k, v in ecyolo.yolo.state_dict().items() if k != FIRST_CONV_KEY}
    yolo.load_state_dict(sd, strict=False)
    W_old = ecyolo.yolo.model[0].conv.weight.detach().cpu()                            # [c1, N, k, k]
    yolo.model[0].conv.weight.data.copy_(W_old[:, idx] * w[idx].view(1, -1, 1, 1))
    new = ECYolo(new_fb, yolo, gate_entropy_weight=0.0, contain_weight=ecyolo.contain_weight, stride=ecyolo.stride)
    new.selected_voltages = np.asarray(getattr(ecyolo, 'selected_voltages', np.arange(fb.n_channels)))[idx]
    return new


def decode_predictions(out, conf_thres=0.001, iou_thres=0.6, max_det=300, nc=1):
    '''Raw YOLO output -> per image float32 [M, 6] (x1, y1, x2, y2, conf, cls) in padded-frame pixels.'''
    pred = out[0] if isinstance(out, (list, tuple)) else out
    if pred.ndim == 3 and pred.shape[1] == 4 + nc:                                     # [B, 4+nc, A]: decode + NMS
        dets = non_max_suppression(pred.float(), conf_thres=conf_thres, iou_thres=iou_thres, nc=nc, max_det=max_det)
    else:                                                                              # [B, max_det, 6]: end-to-end head
        dets = [d[d[:, 4] >= conf_thres][:max_det] for d in pred.float()]
    return [d.detach().cpu().numpy().astype(np.float32).reshape(-1, 6) for d in dets]
```

- [ ] **Step 3: Run** `$PY -m pytest tests/test_ec_yolo.py -q` → 8 passed (the pretrained test downloads 19.5 MB once into `weights/pretrained/`, which is git-ignored). Full suite passes.
- [ ] **Step 4: Commit** `git add models/ec_yolo.py tests/test_ec_yolo.py; git commit -m "Add YOLO26 with N-channel first conv, containment loss, top-k slicing and decoding"`.

---

### Task 7: Mask-based box-quality metrics (`train_eval/box_metrics.py`)

**Files:**
- Create: `train_eval/__init__.py` (empty), `train_eval/box_metrics.py`, `tests/test_box_metrics.py`

**Interfaces:**
- Consumes: `boxes_from_mask(mask, return_labels=True)`, `expand_box` (Task 3).
- Produces: `box_iou_matrix(a [K,4], b [M,4]) -> [K,M]`; `match_greedy(gt_boxes [K,4], dets [M,6]) -> (pred_idx [K] int, iou [K] float)` (−1 / 0 when unmatched; one-to-one, IoU-descending); `mask_coverage(comp [H,W] bool, box [4]) -> float`; `contains(pred_box, gt_box, tol=0.5) -> bool`; `tightness(gt_box, pred_box) -> float`; `center_offset(gt_box, pred_box) -> float`; `average_precision(confs [P], tp [P] bool, n_gt) -> float`; `class BoxMetrics(roi_margin=1.5, roi_min=256, min_area=100)` with `update(mask [H,W] bool, dets [M,6])` and `summary() -> dict` with keys `n_images, n_gt, dets_per_image, recall50, ap50, matched_iou, coverage_raw, coverage_recall99_raw, coverage_roi, coverage_recall99_roi, contain_rate, tightness, center_offset` (nan where undefined), `select_score(summary) -> (coverage_recall99_raw, tightness)`.

- [ ] **Step 1: Failing tests** — `tests/test_box_metrics.py`:

```python
import math
import numpy as np
import pytest

from train_eval.box_metrics import (box_iou_matrix, match_greedy, mask_coverage, contains, tightness, center_offset,
                                    average_precision, BoxMetrics, select_score)

H, W = 48, 40


def _mask():
    m = np.zeros((H, W), bool); m[10:20, 20:30] = True                      # 10x10 object, box [20, 10, 30, 20]
    return m


def _det(x1, y1, x2, y2, conf=0.9):
    return np.array([[x1, y1, x2, y2, conf, 0]], np.float32)


def test_iou_matrix_and_greedy_match():
    g = np.array([[20, 10, 30, 20]], np.float32)
    d = np.array([[20, 10, 30, 20, 0.5, 0], [0, 0, 5, 5, 0.9, 0]], np.float32)
    iou = box_iou_matrix(g, d[:, :4]); np.testing.assert_allclose(iou, [[1.0, 0.0]])
    idx, best = match_greedy(g, d); assert idx.tolist() == [0] and best[0] == 1.0
    idx, best = match_greedy(g, np.zeros((0, 6), np.float32)); assert idx.tolist() == [-1] and best[0] == 0.0


def test_pointwise_metrics():
    m, gt = _mask(), np.array([20, 10, 30, 20], np.float32)
    assert mask_coverage(m, gt) == 1.0 and mask_coverage(m, np.array([22, 10, 30, 20])) == pytest.approx(0.8)
    assert contains(np.array([19.5, 10, 30, 20]), gt) and not contains(np.array([22, 10, 30, 20]), gt)
    assert tightness(gt, np.array([15, 5, 35, 25])) == pytest.approx(0.25)
    assert center_offset(gt, gt) == 0.0 and center_offset(gt, np.array([24, 10, 34, 20])) == pytest.approx(4 / math.hypot(10, 10))


def test_average_precision():
    assert average_precision(np.array([0.9, 0.8]), np.array([True, True]), n_gt=2) == pytest.approx(1.0)
    assert average_precision(np.array([0.9, 0.8]), np.array([False, True]), n_gt=1) == pytest.approx(0.5)
    assert average_precision(np.zeros(0), np.zeros(0, bool), n_gt=3) == 0.0


def test_box_metrics_cases():
    m = _mask()
    perfect = BoxMetrics(roi_margin=1.5, roi_min=0); perfect.update(m, _det(20, 10, 30, 20)); s = perfect.summary()
    assert s['n_gt'] == 1 and s['recall50'] == 1 and s['ap50'] == 1 and s['coverage_raw'] == 1 and s['coverage_recall99_raw'] == 1
    assert s['tightness'] == 1 and s['contain_rate'] == 1 and s['center_offset'] == 0 and s['matched_iou'] == 1
    cut = BoxMetrics(roi_margin=1.5, roi_min=0); cut.update(m, _det(22, 10, 30, 20)); s = cut.summary()
    assert s['recall50'] == 1 and s['coverage_raw'] == pytest.approx(0.8) and s['coverage_recall99_raw'] == 0   # IoU 0.8 but 20 % of the object lost
    assert s['coverage_roi'] == 1 and s['coverage_recall99_roi'] == 1 and math.isnan(s['tightness']) and s['contain_rate'] == 0
    big = BoxMetrics(roi_margin=1.0, roi_min=0); big.update(m, _det(15, 5, 35, 25)); s = big.summary()
    assert s['coverage_raw'] == 1 and s['tightness'] == pytest.approx(0.25) and s['recall50'] == 0   # IoU 0.25
    none = BoxMetrics(); none.update(m, np.zeros((0, 6), np.float32)); s = none.summary()
    assert s['recall50'] == 0 and s['ap50'] == 0 and math.isnan(s['tightness']) and s['coverage_raw'] == 0 and s['dets_per_image'] == 0
    two = BoxMetrics(roi_margin=1.0, roi_min=0)
    m2 = m.copy(); m2[30:40, 2:10] = True                                    # second object
    two.update(m2, np.array([[20, 10, 30, 20, 0.9, 0], [2, 30, 10, 40, 0.8, 0], [0, 0, 3, 3, 0.7, 0]], np.float32))
    s = two.summary(); assert s['n_gt'] == 2 and s['recall50'] == 1 and s['ap50'] == pytest.approx(1.0) and s['dets_per_image'] == 3
    assert select_score(s) == (s['coverage_recall99_raw'], s['tightness'])
```

- [ ] **Step 2: `train_eval/box_metrics.py`**

```python
import math
import numpy as np

from data_loader.boxes import boxes_from_mask, expand_box


def box_iou_matrix(a, b):
    a = np.asarray(a, np.float64).reshape(-1, 4); b = np.asarray(b, np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2]); rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)[:, None]; area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)[None, :]
    return inter / np.maximum(area_a + area_b - inter, 1e-12)


def match_greedy(gt_boxes, dets):
    '''One-to-one matching by descending IoU (ties by confidence); returns per-GT prediction index (-1) and IoU (0).'''
    K, M = len(gt_boxes), len(dets)
    idx, best = -np.ones(K, dtype=int), np.zeros(K)
    if K and M:
        iou = box_iou_matrix(gt_boxes, dets[:, :4])
        order = sorted(((iou[k, m], dets[m, 4], k, m) for k in range(K) for m in range(M) if iou[k, m] > 0), reverse=True)
        used_gt, used_pred = set(), set()
        for v, _, k, m in order:
            if k not in used_gt and m not in used_pred:
                idx[k], best[k] = m, v; used_gt.add(k); used_pred.add(m)
    return idx, best


def mask_coverage(comp, box):
    '''Fraction of the object's mask pixels inside the box (pixel edges: floor x1/y1, ceil x2/y2).'''
    H, W = comp.shape
    x1, y1 = max(0, int(math.floor(box[0]))), max(0, int(math.floor(box[1])))
    x2, y2 = min(W, int(math.ceil(box[2]))), min(H, int(math.ceil(box[3])))
    total = comp.sum()
    return float(comp[y1:y2, x1:x2].sum() / total) if total and x2 > x1 and y2 > y1 else 0.0


def contains(pred_box, gt_box, tol=0.5):
    return bool(pred_box[0] <= gt_box[0] + tol and pred_box[1] <= gt_box[1] + tol and pred_box[2] >= gt_box[2] - tol and pred_box[3] >= gt_box[3] - tol)


def tightness(gt_box, pred_box):
    area = lambda b: max(float(b[2] - b[0]), 0.0) * max(float(b[3] - b[1]), 0.0)
    return area(gt_box) / max(area(pred_box), 1e-12)


def center_offset(gt_box, pred_box):
    cg = ((gt_box[0] + gt_box[2]) / 2, (gt_box[1] + gt_box[3]) / 2); cp = ((pred_box[0] + pred_box[2]) / 2, (pred_box[1] + pred_box[3]) / 2)
    return float(math.hypot(cp[0] - cg[0], cp[1] - cg[1]) / max(math.hypot(gt_box[2] - gt_box[0], gt_box[3] - gt_box[1]), 1e-12))


def average_precision(confs, tp, n_gt):
    '''Single-class AP with all-points interpolation over confidence-sorted predictions.'''
    if n_gt == 0 or len(confs) == 0:
        return 0.0
    order = np.argsort(-np.asarray(confs)); tp = np.asarray(tp, bool)[order]
    ctp, cfp = np.cumsum(tp), np.cumsum(~tp)
    recall = ctp / n_gt; precision = ctp / np.maximum(ctp + cfp, 1e-12)
    mrec = np.concatenate([[0.0], recall, [1.0]]); mpre = np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    changes = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[changes + 1] - mrec[changes]) * mpre[changes + 1]))


class BoxMetrics(object):
    '''Accumulates per-object records over a split; every metric is defined against the GT mask component.'''

    def __init__(self, roi_margin=1.5, roi_min=256, min_area=100):
        self.roi_margin, self.roi_min, self.min_area = roi_margin, roi_min, min_area
        self.records, self.pred_conf, self.pred_tp, self.n_images, self.n_dets = [], [], [], 0, 0

    def update(self, mask, dets):
        H, W = mask.shape
        gt_boxes, labels, ids = boxes_from_mask(mask, self.min_area, return_labels=True)
        dets = np.asarray(dets, np.float32).reshape(-1, 6)
        self.n_images += 1; self.n_dets += len(dets)
        idx, best = match_greedy(gt_boxes, dets)
        for k, (gt, cid) in enumerate(zip(gt_boxes, ids)):
            comp = labels == cid
            rec = {'iou': float(best[k]), 'tp50': bool(best[k] >= 0.5), 'coverage_raw': 0.0, 'coverage_roi': 0.0,
                   'contains': False, 'tightness': float('nan'), 'offset': float('nan')}
            if idx[k] >= 0:
                p = dets[idx[k], :4]
                rec['coverage_raw'] = mask_coverage(comp, p)
                rec['coverage_roi'] = mask_coverage(comp, expand_box(p, self.roi_margin, self.roi_min, H, W))
                rec['contains'] = contains(p, gt)
                rec['tightness'] = tightness(gt, p) if rec['contains'] else float('nan')
                rec['offset'] = center_offset(gt, p)
            self.records.append(rec)
        # AP bookkeeping: confidence-ordered greedy TP flags at IoU >= 0.5
        matched = set()
        for m in np.argsort(-dets[:, 4]) if len(dets) else []:
            iou = box_iou_matrix(gt_boxes, dets[m:m + 1, :4])[:, 0] if len(gt_boxes) else np.zeros(0)
            cand = [(v, k) for k, v in enumerate(iou) if v >= 0.5 and k not in matched]
            tp = bool(cand)
            if tp:
                matched.add(max(cand)[1])
            self.pred_conf.append(float(dets[m, 4])); self.pred_tp.append(tp)

    def summary(self):
        r = self.records; n = len(r)
        nanmean = lambda xs: float(np.mean(xs)) if len(xs) else float('nan')
        return {'n_images': self.n_images, 'n_gt': n, 'dets_per_image': self.n_dets / max(self.n_images, 1),
                'recall50': float(np.mean([x['tp50'] for x in r])) if n else 0.0,
                'ap50': average_precision(self.pred_conf, self.pred_tp, n),
                'matched_iou': nanmean([x['iou'] for x in r if x['tp50']]),
                'coverage_raw': float(np.mean([x['coverage_raw'] for x in r])) if n else 0.0,
                'coverage_recall99_raw': float(np.mean([x['coverage_raw'] >= 0.99 for x in r])) if n else 0.0,
                'coverage_roi': float(np.mean([x['coverage_roi'] for x in r])) if n else 0.0,
                'coverage_recall99_roi': float(np.mean([x['coverage_roi'] >= 0.99 for x in r])) if n else 0.0,
                'contain_rate': float(np.mean([x['contains'] for x in r])) if n else 0.0,
                'tightness': nanmean([x['tightness'] for x in r if x['contains']]),
                'center_offset': nanmean([x['offset'] for x in r if not math.isnan(x['offset'])])}


def select_score(summary):
    '''Checkpoint ranking key: coverage recall @ 0.99 (raw box) first, mean tightness as tie-breaker (nan -> 0).'''
    t = summary['tightness']
    return (summary['coverage_recall99_raw'], 0.0 if (t is None or math.isnan(t)) else t)
```

- [ ] **Step 3: Run** `$PY -m pytest tests/test_box_metrics.py -q` → 4 passed. **Step 4: Commit** `git add train_eval/ tests/test_box_metrics.py; git commit -m "Add mask-based box quality metrics (coverage, tightness, centre offset, recall, AP50)"`.

---

### Task 8: `train_eval/train_eval_det.py` — training epoch, evaluation, GPU flips, visualisation

**Files:**
- Create: `train_eval/train_eval_det.py`, `tests/test_train_eval_det.py`

**Interfaces:**
- Consumes: `ECYolo.loss/forward/end_epoch`, `decode_predictions` (Task 6); `BoxMetrics`, `select_score` (Task 7); `util.misc.MetricLogger, SmoothedValue`; `TrainLogger` (Task 4); `expand_box` (Task 3).
- Produces: `random_flips(img [B,C,H,W], bboxes [ΣK,4], batch_idx [ΣK], p=0.5, generator=None) -> (img, bboxes, flags list[(h, v)])`; `train_one_epoch(model, data_loader, optimizer, device, epoch, max_norm=10.0, scaler=None, accumulate=1, logger=None, flip=True, print_freq=10) -> dict of epoch means`; `evaluate(model, data_loader, device, conf_thres=0.001, iou_thres=0.6, max_det=300, roi_margin=1.5, roi_min=256, min_area=100, logger=None, epoch=0, tag='val', n_images_log=4, amp=True) -> summary dict` (`min_area` = GT component floor, 100 px on real data, 10 on the fixture); `false_colour(y [N,H,W] tensor) -> uint8 [H,W,3]`; `draw_boxes(img_uint8, boxes, color, width=3, dashed=False) -> uint8`.

- [ ] **Step 1: Failing tests** — `tests/test_train_eval_det.py`:

```python
import types
from functools import partial
import numpy as np
import torch
import pytest

from train_eval.train_eval_det import random_flips, train_one_epoch, evaluate, false_colour, draw_boxes
from tests.test_ec_yolo import _tiny
from data_loader.boxes import det_collate_fn
from data_loader.my_dataset import HyperCOD_data
from util.logger import TrainLogger
from tests.conftest import H, W


def _loader(root, split='train'):
    ds = HyperCOD_data(str(root), split=split, use_filter=False, norm='p99', crop_size=0, out_dtype='float16', filter_norm='none')
    return torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False, num_workers=0, collate_fn=partial(det_collate_fn, min_area=10))


def test_random_flips_move_boxes_with_pixels():
    img = torch.zeros(2, 1, 8, 10); img[0, 0, 1, 2] = 1.0; img[1, 0, 6, 7] = 1.0
    bboxes = torch.tensor([[0.25, 0.1875, 0.1, 0.125], [0.75, 0.8125, 0.1, 0.125]])      # centred on the lit pixels
    g = torch.Generator().manual_seed(0)
    out, bb, flags = random_flips(img, bboxes, torch.tensor([0., 1.]), p=1.0, generator=g)   # p=1 -> both flips on every sample
    assert flags == [(True, True), (True, True)]
    assert out[0, 0, 6, 7] == 1.0 and out[1, 0, 1, 2] == 1.0
    torch.testing.assert_close(bb[0], torch.tensor([0.75, 0.8125, 0.1, 0.125])); torch.testing.assert_close(bb[1], torch.tensor([0.25, 0.1875, 0.1, 0.125]))
    out2, bb2, flags2 = random_flips(img, bboxes, torch.tensor([0., 1.]), p=0.0, generator=g)
    assert flags2 == [(False, False), (False, False)] and torch.equal(out2, img) and torch.equal(bb2, bboxes)


def test_train_and_evaluate_one_epoch_on_cpu(synthetic_root, tmp_path):
    root, _, _ = synthetic_root
    model = _tiny(6)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    logger = TrainLogger(types.SimpleNamespace(name='t', wandb=False, runs_dir=str(tmp_path / 'runs'), rank=0), cfg={})
    stats = train_one_epoch(model, _loader(root), opt, torch.device('cpu'), epoch=0, scaler=None, accumulate=2, logger=logger, print_freq=1)
    assert {'loss', 'box_loss', 'cls_loss', 'gate_entropy', 'contain_loss', 'lr'} <= set(stats) and np.isfinite(stats['loss'])
    summary = evaluate(model, _loader(root, 'test'), torch.device('cpu'), roi_margin=1.5, roi_min=0, min_area=10, logger=logger, epoch=0, tag='val', amp=False)
    for k in ['recall50', 'ap50', 'coverage_raw', 'coverage_recall99_raw', 'coverage_roi', 'tightness', 'center_offset', 'dets_per_image']:
        assert k in summary
    assert summary['n_images'] == 1 and summary['n_gt'] == 1
    logger.finish()


def test_false_colour_and_draw_boxes():
    y = torch.randn(5, H, W)
    img = false_colour(y); assert img.shape == (H, W, 3) and img.dtype == np.uint8
    out = draw_boxes(img, np.array([[20, 10, 30, 20]], np.float32), color=(255, 0, 0), width=1)
    assert out.shape == img.shape and (out[10, 20:30] == (255, 0, 0)).all()
    out2 = draw_boxes(img, np.array([[5, 5, 30, 30]], np.float32), color=(0, 255, 0), width=1, dashed=True)
    assert out2.shape == img.shape
```

- [ ] **Step 2: `train_eval/train_eval_det.py`**

```python
import math
import numpy as np
import torch
from PIL import Image, ImageDraw

from util.misc import MetricLogger, SmoothedValue
from models.ec_yolo import decode_predictions
from train_eval.box_metrics import BoxMetrics
from data_loader.boxes import expand_box, yolo_to_xyxy, boxes_from_mask


def random_flips(img, bboxes, batch_idx, p=0.5, generator=None):
    '''Per-sample random horizontal/vertical flips of the frames and their normalised (cx, cy, w, h) boxes.'''
    img = img.clone(); bboxes = bboxes.clone(); flags = []
    for b in range(img.shape[0]):
        h = bool(torch.rand((), generator=generator) < p); v = bool(torch.rand((), generator=generator) < p)
        flags.append((h, v))
        sel = batch_idx == b
        if h:
            img[b] = img[b].flip(-1); bboxes[sel, 0] = 1.0 - bboxes[sel, 0]
        if v:
            img[b] = img[b].flip(-2); bboxes[sel, 1] = 1.0 - bboxes[sel, 1]
    return img, bboxes, flags


def train_one_epoch(model, data_loader, optimizer, device, epoch, max_norm=10.0, scaler=None, accumulate=1, logger=None, flip=True, print_freq=10):
    model.train()
    metric_logger = MetricLogger(delimiter="; ")
    for k in ['loss', 'box_loss', 'cls_loss', 'l1_loss', 'gate_entropy', 'contain_loss']:
        metric_logger.add_meter(k, SmoothedValue(window_size=10, fmt='{value:.4f}'))
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
    n_steps = len(data_loader)
    is_ddp = isinstance(model, torch.nn.parallel.DistributedDataParallel)
    core = model.module if is_ddp else model
    loss_fn = (lambda b: model(batch=b)) if is_ddp else core.loss              # DDP: go through forward so gradients sync
    optimizer.zero_grad(set_to_none=True)
    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        img = batch['img'].to(device, non_blocking=True)                                  # [B, C, H, W] fp16
        bboxes, batch_idx = batch['bboxes'].to(device), batch['batch_idx'].to(device)
        if flip:
            img, bboxes, _ = random_flips(img, bboxes, batch_idx)
        with torch.autocast(device.type, enabled=scaler is not None):
            total, items = loss_fn({'img': img, 'bboxes': bboxes, 'batch_idx': batch_idx, 'cls': batch['cls'].to(device)})
        loss = total / accumulate
        (scaler.scale(loss) if scaler is not None else loss).backward()
        if (i + 1) % accumulate == 0 or i + 1 == n_steps:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            if scaler is not None:
                scaler.step(optimizer); scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        lr = optimizer.param_groups[0]['lr']
        metric_logger.update(loss=float(total.detach()), lr=lr, **{k: items.get(k, 0.0) for k in ['box_loss', 'cls_loss', 'l1_loss', 'gate_entropy', 'contain_loss']})
        if logger is not None:
            step = epoch * n_steps + i
            logger.scalars({'loss': float(total.detach()), 'lr': lr, **items}, step, prefix='train/')
    core.end_epoch()
    return {k: m.global_avg for k, m in metric_logger.meters.items()}


def false_colour(y):
    '''Three channels spread over the response stack -> uint8 RGB (percentile-stretched) for logging.'''
    y = y.detach().float().cpu()
    idx = [0, y.shape[0] // 2, y.shape[0] - 1] if y.shape[0] >= 3 else [0] * 3
    rgb = torch.stack([y[i] for i in idx], dim=-1).numpy()
    lo, hi = np.percentile(rgb, 1), np.percentile(rgb, 99)
    return (np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)


def draw_boxes(img_uint8, boxes, color, width=3, dashed=False):
    im = Image.fromarray(np.ascontiguousarray(img_uint8)); d = ImageDraw.Draw(im)
    for b in np.asarray(boxes).reshape(-1, 4):
        x1, y1, x2, y2 = [float(v) for v in b]
        if not dashed:
            d.rectangle([x1, y1, x2 - 1, y2 - 1], outline=color, width=width)
        else:
            pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]
            for (ax, ay), (bx, by) in zip(pts[:-1], pts[1:]):
                n = max(int(math.hypot(bx - ax, by - ay) // 8), 1)
                for s in range(0, n, 2):
                    d.line([(ax + (bx - ax) * s / n, ay + (by - ay) * s / n), (ax + (bx - ax) * (s + 1) / n, ay + (by - ay) * (s + 1) / n)], fill=color, width=width)
    return np.asarray(im)


@torch.no_grad()
def evaluate(model, data_loader, device, conf_thres=0.001, iou_thres=0.6, max_det=300, roi_margin=1.5, roi_min=256,
             min_area=100, logger=None, epoch=0, tag='val', n_images_log=4, amp=True):
    model.eval()
    metrics = BoxMetrics(roi_margin=roi_margin, roi_min=roi_min, min_area=min_area)
    metric_logger = MetricLogger(delimiter="; ")
    logged = 0
    for batch in metric_logger.log_every(data_loader, 10, f'Eval {tag}:'):
        img = batch['img'].to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=amp and device.type == 'cuda'):
            out = model(img)
        dets = decode_predictions(out, conf_thres=conf_thres, iou_thres=iou_thres, max_det=max_det)
        H, W = img.shape[-2:]
        for b, mask in enumerate(batch['masks']):
            d = dets[b].copy()
            d[:, [0, 2]] = d[:, [0, 2]].clip(0, W); d[:, [1, 3]] = d[:, [1, 3]].clip(0, H)   # padding is bottom/right: coords unchanged
            metrics.update(mask, d)
            if logger is not None and logged < n_images_log:
                with torch.autocast(device.type, enabled=amp and device.type == 'cuda'):
                    y = model.filter_bank(model.pad_to_stride(img[b:b + 1])[0])[0, :, :H, :W]
                pic = false_colour(y)
                pic = draw_boxes(pic, boxes_from_mask(mask, min_area=min_area), (0, 255, 0))
                top = d[d[:, 4] >= 0.25][:5]
                pic = draw_boxes(pic, top[:, :4], (255, 0, 0))
                pic = draw_boxes(pic, [expand_box(p, roi_margin, roi_min, H, W) for p in top[:, :4]], (255, 255, 0), dashed=True)
                logger.image(f'{tag}/frame_{batch["names"][b]}', pic, epoch); logged += 1
    summary = metrics.summary()
    if logger is not None:
        logger.scalars(summary, epoch, prefix=f'{tag}/')
        logger.histogram(f'{tag}/coverage_raw_hist', [r['coverage_raw'] for r in metrics.records], epoch)
        logger.histogram(f'{tag}/tightness_hist', [r['tightness'] for r in metrics.records if not math.isnan(r['tightness'])] or [0.0], epoch)
    print(f"{tag} epoch {epoch}: " + ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in summary.items()))
    return summary
```

- [ ] **Step 3: Run** `$PY -m pytest tests/test_train_eval_det.py -q` → 3 passed (the CPU epoch on the tiny model takes ~20 s). **Step 4: Commit** `git add train_eval/train_eval_det.py tests/test_train_eval_det.py; git commit -m "Add detector training epoch, evaluation with mask metrics, GPU flips and box visualisation"`.

---

### Task 9: `main_det.py` + `cfg/det.yaml` — sessions A and B

**Files:**
- Create: `cfg/det.yaml`, `main_det.py`, `tests/test_main_det.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `get_args_parser()`, `main(args)`, helpers `load_cfg(args) -> dict`, `build_datasets(args) -> (train, val, test)`, `build_model(args, dataset_train, ckpt=None) -> ECYolo` (session B slices an A model by `ckpt`/ranking), `read_ranking(csv_path, k) -> list[int]`, `save_checkpoint(path, model, optimizer, scaler, scheduler, epoch, args)`; outputs in `args.output_dir`: `model_{epoch}`, `model_best`, `results_<name>.txt` (one JSON line per epoch), session A also `gate_ranking.csv` and `top_k.json` (`{"top_k": k, "indices": [...], "voltages": [...]}`).

- [ ] **Step 1: `cfg/det.yaml`**
```yaml
# EC-filter YOLO26 detector hyper-parameters (CLI flags override lr/epochs/batch; these are the loss/eval knobs)
gate_entropy_weight: 0.05     # sparsity of the weight vector w = N*softmax(theta); 0 disables
contain_weight: 1.0           # asymmetric containment hinge inside the box loss; 0 = plain YOLO26 loss
max_norm: 10.0                # gradient clipping
conf_thres: 0.001             # decoding threshold for metrics (low: recall/AP need all predictions)
iou_thres: 0.6                # NMS IoU
max_det: 300
roi_margin: 1.5               # Stage-2 crop = box grown 1.5x about its centre ...
roi_min: 256                  # ... and at least 256 px per side
min_area: 100                 # GT mask components below this many pixels are JPEG specks, not objects (tests use 10)
```

- [ ] **Step 2: Failing test** — `tests/test_main_det.py`:

```python
import json
import os
import sys
import types
import torch
import pytest

import main_det
from data_loader.cube_cache import build_cube_cache, default_cache_dir


def _args(root, tmp_path, **kw):
    argv = ['--data-path', str(root), '--cache-dir', default_cache_dir(str(root)), '--split-file', str(tmp_path / 'val.json'),
            '--yolo-variant', 'yolo26n', '--pretrained', 'none', '--filter-select', 'uniform', '--num-filters', '6',
            '--epochs', '1', '--batch_size', '2', '--num_workers', '0', '--device', 'cpu', '--amp', '--no-wandb',
            '--runs-dir', str(tmp_path / 'runs'), '--roi-min', '0', '--min-area', '10', '--save_every', '1']
    for k, v in kw.items():
        argv += [k] + ([] if v is None else [str(v)])
    parser = main_det.get_args_parser()
    return parser.parse_args(argv)


def test_sessions_a_then_b_and_eval(synthetic_root, tmp_path, monkeypatch):
    root, _, _ = synthetic_root
    build_cube_cache(str(root), 'train', num_workers=0); build_cube_cache(str(root), 'test', num_workers=0)
    monkeypatch.setenv('RANK', '')                                    # keep init_distributed_mode in single-process mode
    monkeypatch.delenv('RANK', raising=False)
    out_a, out_b = tmp_path / 'det_A', tmp_path / 'det_B'
    main_det.main(_args(root, tmp_path, **{'--session': 'A', '--name': 'tA', '--output-dir': str(out_a)}))
    assert (out_a / 'model_best').exists() and (out_a / 'model_0').exists()
    assert (out_a / 'gate_ranking.csv').read_text().splitlines()[0] == 'rank,index,voltage,weight'
    top = json.load(open(out_a / 'top_k.json')); assert len(top['indices']) == 3 and len(top['voltages']) == 3   # --top_k default 10 capped to N//2 when N < 10? no: test passes --top_k 3 below
```
Replace the last two lines by passing `'--top_k': '3'` in the A call and asserting `len(top['indices']) == 3`; then continue:
```python
    lines = (out_a / 'results_tA.txt').read_text().strip().splitlines()
    rec = json.loads(lines[-1]); assert rec['epoch'] == 0 and 'val' in rec and 'coverage_recall99_raw' in rec['val']
    ck = torch.load(out_a / 'model_best', map_location='cpu', weights_only=False)
    assert set(ck) >= {'model', 'optimizer', 'scaler', 'lr_scheduler', 'epoch', 'args', 'selected_indices', 'selected_voltages'}
    assert len(ck['selected_indices']) == 6
    main_det.main(_args(root, tmp_path, **{'--session': 'B', '--name': 'tB', '--output-dir': str(out_b), '--top_k': '3',
                                           '--ranking': str(out_a / 'gate_ranking.csv'), '--resume': str(out_a / 'model_best')}))
    ckb = torch.load(out_b / 'model_best', map_location='cpu', weights_only=False)
    assert len(ckb['selected_indices']) == 3 and ckb['model']['yolo.model.0.conv.weight'].shape[1] == 3
    main_det.main(_args(root, tmp_path, **{'--session': 'B', '--name': 'tB_eval', '--output-dir': str(out_b), '--top_k': '3',
                                           '--resume': str(out_b / 'model_best'), '--eval': None}))
    assert (out_b / 'results_tB_eval.txt').exists()
```

- [ ] **Step 3: `main_det.py`**

```python
"""
Stage 1: EC-filter YOLO26 camouflage detector with a trainable per-channel weight vector.

Session A (all usable voltages + weight vector), single GPU:
  python main_det.py --session A --name det_A --output-dir weights/det_A
Session B (top-10 voltages from A, no weight vector), initialised from A:
  python main_det.py --session B --top_k 10 --ranking weights/det_A/gate_ranking.csv --resume weights/det_A/model_best --name det_B --output-dir weights/det_B
Evaluate a checkpoint on val + test:
  python main_det.py --session B --resume weights/det_B/model_best --eval
DDP (2 GPUs):
  torchrun --nproc_per_node=2 main_det.py --session A --name det_A --output-dir weights/det_A
Smoke run on a few cached frames:
  python main_det.py --session A --limit 4 --epochs 1 --name smoke --output-dir weights/smoke --no-wandb
"""
import os
if "RANK" not in os.environ and "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import csv
import datetime
import json
import math
import random
import time
from functools import partial

import yaml
import numpy as np
import torch
import torch.multiprocessing

import util.misc as utils
from util.distributed_util import Custom_DistributedSampler
from util.logger import TrainLogger
from data_loader.my_dataset import HyperCOD_data
from data_loader.boxes import det_collate_fn
from data_loader.det_splits import make_det_splits
from data_loader.cube_cache import default_cache_dir
from models.ec_yolo import build_ec_yolo, select_top_k, slice_to_channels
from train_eval.train_eval_det import train_one_epoch, evaluate
from train_eval.box_metrics import select_score

torch.multiprocessing.set_sharing_strategy('file_system')


def get_args_parser():
    parser = argparse.ArgumentParser('EC-filter YOLO26 camouflage detector (Stage 1)', add_help=False)
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--name', default='', help='run name; results_<name>.txt, runs/<name>, W&B run name')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--eval', action='store_true', help='only evaluate --resume on val + test')
    # session / model
    parser.add_argument('--session', default='A', choices=['A', 'B'], help='A: all voltages + weight vector; B: top_k fixed voltages')
    parser.add_argument('--top_k', default=10, type=int, help='voltages kept for session B')
    parser.add_argument('--ranking', default='', help='gate_ranking.csv from session A (session B)')
    parser.add_argument('--resume', default='', help='checkpoint: session A model to slice (B) or a checkpoint to continue/evaluate')
    parser.add_argument('--hpy', type=str, default='cfg/det.yaml', help='hyper parameters path')
    parser.add_argument('--yolo-variant', default='yolo26s')
    parser.add_argument('--pretrained', default='auto', help="'auto' downloads <variant>.pt, a path, or 'none'")
    parser.add_argument('--gate-entropy-weight', type=float, default=None, help='overrides cfg')
    parser.add_argument('--contain-weight', type=float, default=None, help='overrides cfg')
    # data
    parser.add_argument('--data-path', default='/data2/chaoyi/HyperCOD/Raw data')
    parser.add_argument('--cache-dir', default='', help='fp16 cube cache, default <data-path>/cache_fp16')
    parser.add_argument('--band-range', type=float, nargs=2, default=[400.0, 800.0])
    parser.add_argument('--filter-path', default=None)
    parser.add_argument('--filter-select', default='all', choices=['all', 'uniform', 'osp', 'manual'])
    parser.add_argument('--num-filters', type=int, default=30)
    parser.add_argument('--filter-voltages', type=float, nargs='+', default=None)
    parser.add_argument('--split-file', default='', help='val ids json, default data_loader/splits/det_val_ids.json')
    parser.add_argument('--limit', type=int, default=0, help='use only the first N train / N//4 val / N test frames (smoke runs)')
    # training
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lrf', default=0.05, type=float, help='final lr = lr * lrf (cosine)')
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--accumulate', default=4, type=int, help='gradient accumulation steps')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N')
    parser.add_argument('--amp', action='store_false', help='disable mixed precision (on by default)')
    parser.add_argument('--no-flip', action='store_true', help='disable random h/v flips')
    parser.add_argument('--save_every', default=10, type=int)
    # eval overrides
    parser.add_argument('--conf-thres', type=float, default=None); parser.add_argument('--iou-thres', type=float, default=None)
    parser.add_argument('--roi-margin', type=float, default=None); parser.add_argument('--roi-min', type=float, default=None)
    parser.add_argument('--min-area', type=int, default=None, help='GT component floor in px (cfg: 100; fixture tests use 10)')
    # logging
    parser.add_argument('--output-dir', default='weights/det_A')
    parser.add_argument('--runs-dir', default='runs')
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--wandb-entity', default='chaoyi-hsi'); parser.add_argument('--wandb-project', default='hsi_camo')
    parser.add_argument('--wandb-dir', default='wandb')
    # distributed
    parser.add_argument('--world_size', default=1, type=int); parser.add_argument('--dist_url', default='env://')
    return parser


def load_cfg(args):
    with open(args.hpy) as f:
        cfg = yaml.safe_load(f)
    for k in ['gate_entropy_weight', 'contain_weight', 'conf_thres', 'iou_thres', 'roi_margin', 'roi_min', 'min_area']:
        if getattr(args, k) is None:
            setattr(args, k, cfg[k])
    args.max_norm = cfg.get('max_norm', 10.0); args.max_det = cfg.get('max_det', 300)
    return cfg


def build_datasets(args):
    cache_dir = args.cache_dir or default_cache_dir(args.data_path)
    train_ids, val_ids = make_det_splits(args.data_path, path=args.split_file or None)
    test_ids = None
    if args.limit:
        train_ids, val_ids = train_ids[:args.limit], val_ids[:max(1, args.limit // 4)]
        test_ids = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(args.data_path, 'test', 'hyperspectral')) if f.endswith('.mat'))
        test_ids = sorted(test_ids, key=int)[:max(1, args.limit)]
    kw = dict(data_path=args.data_path, use_filter=False, norm='p99', crop_size=0, cache_dir=cache_dir, out_dtype='float16',
              band_range=tuple(args.band_range), filter_path=args.filter_path, num_filters=args.num_filters,
              filter_select=args.filter_select, filter_voltages=args.filter_voltages, filter_norm='l1')
    return (HyperCOD_data(split='train', ids=train_ids, **kw), HyperCOD_data(split='train', ids=val_ids, **kw),
            HyperCOD_data(split='test', ids=test_ids, **kw))


def read_ranking(csv_path, k):
    with open(csv_path, newline='') as f:
        rows = sorted(csv.DictReader(f), key=lambda r: int(r['rank']))
    assert len(rows) >= k, f"{csv_path} has {len(rows)} channels, top_k={k}"
    return [int(r['index']) for r in rows[:k]]


def build_model(args, dataset_train, ckpt=None):
    '''Session A: all voltages + weight vector. Session B: slice an A model (from --ranking/--resume or a B checkpoint's indices).'''
    a_args = argparse.Namespace(**{**vars(args), 'session': 'A'})
    model = build_ec_yolo(a_args, dataset_train)                                   # A structure, N channels
    n_all = model.filter_bank.n_channels
    if args.session == 'A':
        model.selected_indices = list(range(n_all))
        if ckpt is not None:
            model.load_state_dict(ckpt['model'])
        return model
    if ckpt is not None and ckpt['args'].get('session') == 'B':                      # continue / evaluate a B checkpoint
        idx = list(ckpt['selected_indices'])
        model = slice_to_channels(model, idx, args.yolo_variant)
        model.load_state_dict(ckpt['model'])
    else:
        assert ckpt is not None and args.ranking, "session B needs --resume <session A checkpoint> and --ranking <gate_ranking.csv>"
        model.load_state_dict(ckpt['model'])                                          # session A weights
        idx = read_ranking(args.ranking, args.top_k)
        model = slice_to_channels(model, idx, args.yolo_variant)
    model.selected_indices = idx
    print(f"session B: {len(idx)} voltages {np.round(model.selected_voltages, 2).tolist()}")
    return model


def save_checkpoint(path, model, optimizer, scaler, scheduler, epoch, args):
    utils.save_on_master({'model': model.state_dict(), 'optimizer': optimizer.state_dict() if optimizer else None,
                          'scaler': scaler.state_dict() if scaler else None, 'lr_scheduler': scheduler.state_dict() if scheduler else None,
                          'epoch': epoch, 'args': vars(args), 'selected_indices': list(model.selected_indices),
                          'selected_voltages': [float(v) for v in model.selected_voltages]}, path)


def main(args):
    utils.init_distributed_mode(args)
    cfg = load_cfg(args)
    args.name = args.name or f'det_{args.session}'
    device = torch.device(args.device if args.device == 'cpu' or torch.cuda.is_available() else 'cpu')
    seed = args.seed + utils.get_rank(); torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    os.makedirs(args.output_dir, exist_ok=True)
    logger = TrainLogger(args, cfg)

    dataset_train, dataset_val, dataset_test = build_datasets(args)
    print(f"train {len(dataset_train)} / val {len(dataset_val)} / test {len(dataset_test)} frames, {dataset_train.n_bands} bands")
    if args.distributed:
        sampler_train = Custom_DistributedSampler(dataset_train, shuffle=True)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
    collate = partial(det_collate_fn, min_area=args.min_area)
    mk = lambda ds, sampler, bs, shuffle: torch.utils.data.DataLoader(ds, batch_size=bs, sampler=sampler, shuffle=shuffle, num_workers=args.num_workers,
                                                                     collate_fn=collate, pin_memory=device.type == 'cuda', drop_last=False)
    loader_train = mk(dataset_train, sampler_train, args.batch_size, False)
    loader_val, loader_test = mk(dataset_val, None, 1, False), mk(dataset_test, None, 1, False)

    ckpt = torch.load(args.resume, map_location='cpu', weights_only=False) if args.resume else None
    model = build_model(args, dataset_train, ckpt).to(device)
    model.attach_criterion()
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu]); model_without_ddp = model.module

    no_decay = [p for n, p in model_without_ddp.named_parameters() if p.requires_grad and (n.endswith('.bias') or n == 'filter_bank.theta' or n == 'yolo.model.0.conv.weight' or p.ndim == 1)]
    decay = [p for n, p in model_without_ddp.named_parameters() if p.requires_grad and all(p is not q for q in no_decay)]
    optimizer = torch.optim.AdamW([{'params': decay, 'weight_decay': args.weight_decay}, {'params': no_decay, 'weight_decay': 0.0}], lr=args.lr)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scaler = torch.amp.GradScaler('cuda') if (args.amp and device.type == 'cuda') else None
    if ckpt is not None and ckpt['args'].get('session') == args.session and not args.eval and ckpt.get('optimizer'):
        optimizer.load_state_dict(ckpt['optimizer']); scheduler.load_state_dict(ckpt['lr_scheduler'])
        if scaler is not None and ckpt.get('scaler'):
            scaler.load_state_dict(ckpt['scaler'])
        args.start_epoch = ckpt['epoch'] + 1

    eval_kw = dict(conf_thres=args.conf_thres, iou_thres=args.iou_thres, max_det=args.max_det, roi_margin=args.roi_margin, roi_min=args.roi_min,
                   min_area=args.min_area, amp=scaler is not None)
    results_path = os.path.join(args.output_dir, f'results_{args.name}.txt')
    if args.eval:
        val = evaluate(model_without_ddp, loader_val, device, logger=logger, epoch=args.start_epoch, tag='val', **eval_kw)
        test = evaluate(model_without_ddp, loader_test, device, logger=logger, epoch=args.start_epoch, tag='test', **eval_kw)
        if utils.is_main_process():
            with open(results_path, 'a') as f:
                f.write(json.dumps({'eval': True, 'resume': args.resume, 'val': val, 'test': test}) + '\n')
        logger.finish(); return

    print("Start training"); start = time.time(); best = (-1.0, -1.0)
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(model, loader_train, optimizer, device, epoch, max_norm=args.max_norm, scaler=scaler,
                                      accumulate=args.accumulate, logger=logger, flip=not args.no_flip)
        scheduler.step()
        val = evaluate(model_without_ddp, loader_val, device, logger=logger, epoch=epoch, tag='val', **eval_kw)
        if args.session == 'A' and utils.is_main_process():
            w = model_without_ddp.filter_bank.weights.detach().cpu().numpy(); volts = model_without_ddp.selected_voltages
            logger.histogram('gate/weights', w, epoch); logger.scalars({'gate/max': w.max(), 'gate/entropy': float(model_without_ddp.filter_bank.entropy())}, epoch)
            order = np.argsort(-w)[:20]; logger.table('gate/top20', ['rank', 'voltage', 'weight'], [[r + 1, float(volts[i]), float(w[i])] for r, i in enumerate(order)], epoch)
        score = select_score(val)
        if utils.is_main_process():
            if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
                save_checkpoint(os.path.join(args.output_dir, f'model_{epoch}'), model_without_ddp, optimizer, scaler, scheduler, epoch, args)
            if score > best:
                best = score; save_checkpoint(os.path.join(args.output_dir, 'model_best'), model_without_ddp, optimizer, scaler, scheduler, epoch, args)
            with open(results_path, 'a') as f:
                f.write(json.dumps({'epoch': epoch, 'train': train_stats, 'val': val, 'best': list(best)}) + '\n')
    print(f"Training time {datetime.timedelta(seconds=int(time.time() - start))}, best (coverage_recall99, tightness) = {best}")

    if utils.is_main_process():
        best_ckpt = torch.load(os.path.join(args.output_dir, 'model_best'), map_location='cpu', weights_only=False)
        model_without_ddp.load_state_dict(best_ckpt['model'])
        val = evaluate(model_without_ddp, loader_val, device, logger=logger, epoch=args.epochs, tag='val_best', **eval_kw)
        test = evaluate(model_without_ddp, loader_test, device, logger=logger, epoch=args.epochs, tag='test_best', **eval_kw)
        with open(results_path, 'a') as f:
            f.write(json.dumps({'final': True, 'best_epoch': best_ckpt['epoch'], 'val': val, 'test': test}) + '\n')
        if args.session == 'A':
            k = min(args.top_k, model_without_ddp.filter_bank.n_channels)
            idx, w, volts = select_top_k(model_without_ddp, k, csv_path=os.path.join(args.output_dir, 'gate_ranking.csv'))
            with open(os.path.join(args.output_dir, 'top_k.json'), 'w') as f:
                json.dump({'top_k': k, 'indices': idx, 'voltages': [float(volts[i]) for i in idx], 'weights': [float(w[i]) for i in idx]}, f, indent=1)
            print(f"top-{k} voltages (V): {[round(float(volts[i]), 2) for i in idx]}")
    logger.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('EC-filter YOLO26 detector', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
```
Notes: `Custom_DistributedSampler` + `DataLoader(sampler=..., shuffle=False)`; in the non-distributed branch `shuffle` is provided through the `RandomSampler`. `select_score` returns a tuple compared lexicographically (coverage recall first, tightness second).

- [ ] **Step 4: Run** `$PY -m pytest tests/test_main_det.py -q` → 1 passed (~1–2 min on CPU). Full suite passes.
- [ ] **Step 5: Commit** `git add cfg/det.yaml main_det.py tests/test_main_det.py; git commit -m "Add main_det.py: sessions A/B with checkpointing, metrics-based model selection, TB+W&B logging"`.

---

### Task 10: `main_det_rois.py` — ROI export for Stage 2

**Files:**
- Create: `main_det_rois.py`, `tests/test_main_det_rois.py`

**Interfaces:**
- Consumes: `main_det.build_datasets`, `main_det.build_model`, `decode_predictions`, `expand_box`, `boxes_from_mask`.
- Produces: `results/det/rois_<split>.json` = `{name: {"rois": [[x1, y1, x2, y2, conf], ...], "boxes": [[x1, y1, x2, y2, conf], ...], "gt_boxes": [[x1, y1, x2, y2], ...]}}` (rois = detections above `--conf-thres` (default 0.25), at most `--max-rois` per frame, expanded by `roi_margin`/`roi_min`, clipped; `boxes` = the raw detections); prints per-split coverage recall of the exported ROIs.

- [ ] **Step 1: Failing test** — `tests/test_main_det_rois.py`:

```python
import json
import torch
import main_det, main_det_rois
from tests.test_main_det import _args
from data_loader.cube_cache import build_cube_cache


def test_export_rois_json(synthetic_root, tmp_path, monkeypatch):
    root, _, _ = synthetic_root
    monkeypatch.delenv('RANK', raising=False)
    build_cube_cache(str(root), 'train', num_workers=0); build_cube_cache(str(root), 'test', num_workers=0)
    out_a = tmp_path / 'det_A'
    main_det.main(_args(root, tmp_path, **{'--session': 'A', '--name': 'tA', '--output-dir': str(out_a), '--top_k': '3'}))
    args = main_det_rois.get_args_parser().parse_args(['--data-path', str(root), '--cache-dir', str(root / 'cache_fp16'),
        '--split-file', str(tmp_path / 'val.json'), '--yolo-variant', 'yolo26n', '--pretrained', 'none', '--filter-select', 'uniform',
        '--num-filters', '6', '--device', 'cpu', '--amp', '--no-wandb', '--session', 'A', '--resume', str(out_a / 'model_best'),
        '--split', 'test', '--out-dir', str(tmp_path / 'rois'), '--conf-thres', '0.0', '--max-rois', '2', '--roi-min', '0', '--min-area', '10', '--num_workers', '0'])
    path = main_det_rois.main(args)
    data = json.load(open(path))
    assert set(data) == {'7'} and {'rois', 'boxes', 'gt_boxes'} <= set(data['7']) and len(data['7']['rois']) <= 2
    assert data['7']['gt_boxes'] == [[20.0, 10.0, 26.0, 16.0]]
    for r in data['7']['rois']:
        assert len(r) == 5 and 0 <= r[0] < r[2] <= 40 and 0 <= r[1] < r[3] <= 48
```

- [ ] **Step 2: `main_det_rois.py`**

```python
"""
Export candidate camouflage regions (ROIs) from a trained detector for the Stage-2 segmentation loader.
  python main_det_rois.py --session B --resume weights/det_B/model_best --split train
  python main_det_rois.py --session B --resume weights/det_B/model_best --split test
Writes results/det/rois_<split>.json: {name: {"rois": [[x1,y1,x2,y2,conf],...], "boxes": [...], "gt_boxes": [...]}}
"""
import os
if "RANK" not in os.environ and "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import json
from functools import partial
import numpy as np
import torch

import util.misc as utils
import main_det
from data_loader.boxes import det_collate_fn, expand_box, boxes_from_mask
from data_loader.my_dataset import HyperCOD_data
from models.ec_yolo import decode_predictions
from train_eval.box_metrics import mask_coverage


def get_args_parser():
    parser = argparse.ArgumentParser('Export detector ROIs', parents=[main_det.get_args_parser()], add_help=False)
    parser.add_argument('--split', default='test', choices=['train', 'test'])
    parser.add_argument('--out-dir', default='results/det')
    parser.add_argument('--max-rois', type=int, default=5, help='detections kept per frame (by confidence)')
    parser.set_defaults(conf_thres=0.25, wandb=False)
    return parser


@torch.no_grad()
def main(args):
    utils.init_distributed_mode(args)
    main_det.load_cfg(args)
    device = torch.device(args.device if args.device == 'cpu' or torch.cuda.is_available() else 'cpu')
    dataset_train, _, _ = main_det.build_datasets(args)                              # only for the filter matrices
    kw = dict(data_path=args.data_path, use_filter=False, norm='p99', crop_size=0, cache_dir=args.cache_dir or None, out_dtype='float16',
              band_range=tuple(args.band_range), filter_path=args.filter_path, num_filters=args.num_filters,
              filter_select=args.filter_select, filter_voltages=args.filter_voltages, filter_norm='l1')
    dataset = HyperCOD_data(split=args.split, **kw)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
                                         collate_fn=partial(det_collate_fn, min_area=args.min_area))
    ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
    model = main_det.build_model(args, dataset_train, ckpt).to(device).eval()
    out, covered, n_gt = {}, 0, 0
    for batch in loader:
        img = batch['img'].to(device)
        with torch.autocast(device.type, enabled=device.type == 'cuda' and args.amp):
            dets = decode_predictions(model(img), conf_thres=args.conf_thres, iou_thres=args.iou_thres, max_det=args.max_det)[0]
        H, W = img.shape[-2:]
        dets = dets[np.argsort(-dets[:, 4])][:args.max_rois]
        dets[:, [0, 2]] = dets[:, [0, 2]].clip(0, W); dets[:, [1, 3]] = dets[:, [1, 3]].clip(0, H)
        rois = [[*expand_box(d[:4], args.roi_margin, args.roi_min, H, W).tolist(), float(d[4])] for d in dets]
        gt_boxes, labels, ids = boxes_from_mask(batch['masks'][0], min_area=args.min_area, return_labels=True)
        for gb, cid in zip(gt_boxes, ids):
            n_gt += 1; covered += any(mask_coverage(labels == cid, r[:4]) >= 0.99 for r in rois)
        out[batch['names'][0]] = {'rois': rois, 'boxes': [[float(v) for v in d[:5]] for d in dets], 'gt_boxes': gt_boxes.tolist()}
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f'rois_{args.split}.json')
    with open(path, 'w') as f:
        json.dump(out, f)
    print(f"{args.split}: {len(out)} frames, {n_gt} objects, ROI coverage recall@0.99 = {covered / max(n_gt, 1):.3f} -> {path}")
    return path


if __name__ == '__main__':
    main(get_args_parser().parse_args())
```

- [ ] **Step 3: Run** `$PY -m pytest tests/test_main_det_rois.py -q` → 1 passed. **Step 4: Commit** `git add main_det_rois.py tests/test_main_det_rois.py; git commit -m "Add ROI export from the detector for Stage 2"`.

---

### Task 11: Real-data preparation, GPU smoke run, and launching the sessions

**Files:** none new (operational task; records results in `docs/superpowers/plans/2026-09-27-ec-yolo-detector.md` § Results and in the spec's §10).

- [ ] **Step 1: Smoke cache for 4 frames + 1 test frame** (fast, ~1 min):
```bash
$PY -m data_loader.cube_cache --split train --ids 1 2 3 4 --num_workers 4
$PY -m data_loader.cube_cache --split test --ids 6 --num_workers 1
```
Expected: `Caching 4/… train cubes …`, files `cache_fp16/train/{1,2,3,4}.npy` (0.55 GB each) and `cache_fp16/test/6.npy`; the windowed p99 csv is only written by the full build (step 3) — for the smoke run pass `--band-range 400 800` and let `compute_window_p99` build the csv **only if it is missing**; to avoid a 45-minute csv build here, create it from the 5 cached cubes first: `$PY -c "from data_loader.my_dataset import compute_window_p99; compute_window_p99('/data2/chaoyi/HyperCOD/Raw data','train',(400.,800.),num_workers=4)"` reads all 279 cubes (≈ 6 min with 4 workers) — acceptable; or run step 3 first.
- [ ] **Step 2: GPU smoke run** (session A on 4 frames, yolo26s pretrained, 1 epoch):
```bash
$PY main_det.py --session A --limit 4 --epochs 1 --batch_size 2 --accumulate 1 --num_workers 2 --name smoke_A --output-dir weights/smoke_A --no-wandb --save_every 1
```
Expected: `344 input channels, pretrained tensors reused 695/708`; ~0.6–1.0 s per step; peak GPU memory ≤ 12 GB (`nvidia-smi` in a second shell); `weights/smoke_A/{model_0,model_best,gate_ranking.csv,top_k.json,results_smoke_A.txt}` written; `runs/smoke_A/` has events. Then session B smoke: `--session B --top_k 10 --ranking weights/smoke_A/gate_ranking.csv --resume weights/smoke_A/model_best --name smoke_B --output-dir weights/smoke_B --limit 4 --epochs 1 --no-wandb` → `session B: 10 voltages [...]`. Then `$PY main_det_rois.py --session B --resume weights/smoke_B/model_best --split test --limit 1 --no-wandb` → `results/det/rois_test.json`.
- [ ] **Step 3: Full cache + statistics** (once, ~1 h + 3 min; run in the background with `nohup … > logs/cache.log 2>&1 &`):
```bash
mkdir -p logs
nohup $PY -m data_loader.cube_cache --split all --num_workers 8 > logs/cache.log 2>&1 &
# when finished (tail logs/cache.log shows "Saved .../intensity_p99_400_800.csv" for train and test):
$PY -m data_loader.band_stats --band_range 400 800 --num_workers 8        # -> band_stats_train_400_800.npz
```
Expected: 349 `.npy` files (194 GB), both windowed p99 csvs, `band_stats_train_400_800.npz` (`mean (133,)`, `cov (133,133)`).
- [ ] **Step 4: Session A** (background, ~2–3 h; W&B run `det_A` in `chaoyi-hsi/hsi_camo`):
```bash
nohup $PY main_det.py --session A --name det_A --output-dir weights/det_A --epochs 100 --batch_size 2 --accumulate 4 --num_workers 4 > logs/det_A.log 2>&1 &
```
Monitor: `tail -f logs/det_A.log` (per-epoch `val epoch k: recall50=… coverage_recall99_raw=… tightness=…`), TensorBoard `tensorboard --logdir runs`, W&B. Finished when the log prints `top-10 voltages (V): [...]` and `weights/det_A/{model_best,gate_ranking.csv,top_k.json}` exist.
- [ ] **Step 5: Session B** (background, ~2 h):
```bash
nohup $PY main_det.py --session B --top_k 10 --ranking weights/det_A/gate_ranking.csv --resume weights/det_A/model_best --name det_B --output-dir weights/det_B --epochs 100 --batch_size 2 --accumulate 4 --num_workers 4 > logs/det_B.log 2>&1 &
```
- [ ] **Step 6: ROI export + report**:
```bash
$PY main_det_rois.py --session B --resume weights/det_B/model_best --split train
$PY main_det_rois.py --session B --resume weights/det_B/model_best --split test
```
Record in the spec §10: top-10 voltages, val/test `coverage_recall99_raw`, `coverage_recall99_roi`, `tightness`, `center_offset`, `recall50`, `ap50` for A and B, and the ROI coverage recall printed by `main_det_rois.py`. Commit the docs (`git add docs/...; git commit -m "Record Stage 1 detector results"`), push the branch.

---

## Results

See the spec's §12 for the full tables. Summary (2026-09-29): session A trained 100 epochs (7 h 50 min, ≈ 2 s/step disk-bound after the O_DIRECT loader fix); converged top-10 voltages `1.32, 1.29, 1.35, -0.41, -0.38, 1.38, -0.44, 1.41, 1.26, -0.35 V`; the spec's checkpoint-selection rule picked epoch 0, so `cfg select_keys` now uses the ROI operating-point pair and `roi_conf` was calibrated to 0.02; session B was launched from A's `model_89` with `gate_ranking_ep89.csv` and stopped by the user at epoch 91 (plateau); its `model_best` (epoch 25) gives test AP50 0.36 / recall50 0.52 and the exported ROIs contain 46.5 % of test objects (69.6 % of train+val objects) at 1.7 ROIs per frame. A `--raw-bands` control (133 raw bands, no filter) is running for comparison. Task 11 Steps 4–6 were run with `python -u`, 6 DataLoader workers and the launch scripts in `bash_files/`; Step 5 uses `--resume weights/det_A/model_89 --ranking weights/det_A/gate_ranking_ep89.csv` instead of `model_best`/`gate_ranking.csv` for the reason above.

---

## Self-review against the spec

- **Spec §3 decisions → tasks:** band window (T1); no downsampling / GPU padding (T6 `pad_to_stride`, T8); projection on GPU + materialised responses (T5, T6); weight vector `N·softmax(θ)` + entropy (T5, T6 `loss`); N-channel first conv from tiled RGB kernel (T6 `adapt_first_conv_weight`, `build_detection_model`); A→B transfer with gate folding (T6 `slice_to_channels`, T9 `build_model`); checkpoint selection by coverage recall @ 0.99 + tightness (T7 `select_score`, T9); val split 28 ids seed 0 (T3); flips (T8 `random_flips`); code style (all); mask-based metrics (T7); containment loss (T6 `ContainBboxLoss`, cfg `contain_weight`).
- **Spec §5 components:** 5.1 loader change (T1), 5.2 cache (T2), 5.3 boxes (T3), 5.4 `HyperCOD_data` options + `filter_bank_tensors` (T2), 5.5 `FilterBank` (T5), 5.6 `ec_yolo` (T6), 5.7 `train_eval_det` + `box_metrics` (T7, T8), 5.8 `main_det` (T9), 5.9 `TrainLogger` (T4), 5.10 `main_det_rois` (T10). §7 procedure (T11). §8 error handling: asserts in T1–T3, T6 (`top_k`), T9 (`session B needs --resume/--ranking`), cache-missing message (T2), W&B fallback (T4). §9 tests: every bullet has a test in T1–T10 (windowed p99 csv, cache round-trip/skip, boxes/flips/collate, logger fallback, FilterBank ≡ loader, first-conv tiling, containment hinge, slicing preserves outputs, metric cases incl. ROI rescue, train/eval step on the tiny model, main A→B→eval, ROI export).
- **Placeholder scan:** none (`TBD/TODO` absent; every code step has full code; commands have expected outputs).
- **Type consistency:** `HyperCOD_data.filter_bank_tensors() -> (R, mean, std, voltages)` used by `build_ec_yolo` (T6) and tests (T2, T5); `det_collate_fn` keys `img/masks/batch_idx/cls/bboxes/boxes_xyxy/names` consumed by `ECYolo.loss`, `train_one_epoch`, `evaluate`, `main_det_rois`; `decode_predictions -> list[np [M,6]]` consumed by `evaluate`/`BoxMetrics.update(mask, dets)` and `main_det_rois`; `select_score(summary) -> tuple` compared in `main_det`; `ECYolo.forward(img=None, batch=None)` used by DDP path in T8; `build_model(args, dataset_train, ckpt)` shared by T9/T10; `expand_box(box, margin, min_size, H, W)` signature identical in T3/T7/T8/T10.
- **Known deviation from the spec text:** the spec's `FilterBank.forward(x, scale)` became `forward(x)` because the loader already applies the p99 scale (`norm='p99'`); spec §5.5 updated accordingly in the docs commit of T11.
