# HyperCOD dataloader with EC-filter response — design

**Date:** 2026-09-21
**Status:** approved by user (spatial: native-resolution crops; I/O: direct `.mat` reads; `--norm` default `p99`; repo initialised)

## Goal

A PyTorch `Dataset` for the HyperCOD hyperspectral camouflaged-object dataset that

1. aligns the EC tunable-detector spectral response (`EC_filterV3.mat`) to the cube's wavelength axis once, at `__init__`;
2. in `__getitem__` returns either the simulated detector channels (`use_filter=True`) or the raw 200 bands (`use_filter=False`), plus the binary GT mask;
3. follows the conventions of the previous HSI project (`/data/chaoyi_he/HSI/Diffu/data_loader/my_dataset.py`, `HSI/EC_dataset/Up_seg/my_dataset.py`): `torch.utils.data` subclass with plain keyword arguments, `sio.loadmat` + `scipy.interpolate.interp1d` for the sensor response, `self.sensor_R_matrix [C, N]`, projection by matrix product in `__getitem__`, numpy outputs, module-level collate function, `argparse` snake_case flags, shape comments like `# [C, H, W]`, `__main__` smoke test.

## Data facts the design relies on

| Item | Value | Source |
|---|---|---|
| Samples | 279 train / 70 test, integer ids, all modalities aligned | `/data2/chaoyi/HyperCOD/Raw data/{train,test}` |
| Cube file | `hyperspectral/<id>.mat`, MATLAB v7.3 (HDF5), key `hypercube` | opened with `h5py`, not `scipy.io` |
| Cube layout as seen by h5py | `(200, 1240, 1680)` = `(B, W, H)` float32, gzip, chunks `(200, 77, 1)` | MATLAB column-major |
| Band centres | `np.linspace(400, 1000, 200)` nm (3.015 nm step) | `/data2/chaoyi/HyperCOD/Code/wl_200bands.npy` |
| GT | `GT/<id>.png`, actually JPEG bytes, 3 identical channels; foreground = channel 0 `> 127` | verified on all 349 |
| Intensity stats | `intensity map/intensity_p99_summary.csv`: `sample_id, intensity_p99_valid, ...` where intensity = sum over 200 bands | verified exact |
| Filter | `EC_filterV3.mat` (MATLAB v5): `Wavelength (1,401)` 400–800 nm @1 nm, `Voltage (1,351)` −1.00..2.50 V @0.01 V, `responsivity (401,351)` in [−1,1], each column peak-normalised to \|max\|=1 | verified |
| Filter polarity flip | response changes sign abruptly at V ≈ 0.27–0.29 V (near-zero output there) | verified |
| Timings | full cube 9.5 s; 512² crop hyperslab 1.7 s; `(B,W,H)→(B,H,W)` swap 0.02–0.14 s; 200→30 projection 0.04 s; `(B,W,H)→(H,W,B)` copy 6.6 s (avoid) | measured |

## Decisions

- **Spatial:** no down-sampling. Training returns object-biased random crops of `crop_size × crop_size` at native resolution; test returns the full 1680×1240 frame. (User choice; targets have median 0.36 % of pixels, so down-sampling would erase the smallest.)
- **I/O:** read the crop directly from the `.mat` with an h5py hyperslab in every `__getitem__`; no cache. (User choice; simplest code, ~1.7 s per 512² crop hidden by DataLoader workers.)
- **Layout:** keep the h5py `(B, W, H)` block, project with the filter in that layout, then swap only the last two axes to `(C, H, W)`. Never materialise `(H, W, B)`.
- **Filter alignment:** linear `interp1d` onto the cube axis with `fill_value=0.0` outside 400–800 nm (bands 133–199 get zero response — the file defines nothing there; extrapolation would be wrong). **No per-column min–max normalisation** (unlike the previous project) because the columns are already peak-normalised and min–max would destroy the negative lobes.
- **Channel selection:** candidates exclude the dead zone 0.25–0.31 V; `uniform` (default, `np.linspace` over candidate indices as in the previous code), `osp` (reuse the previous `osp()` for index selection only), `all`, `manual` (nearest voltage to each requested value). `num_filters` is used by `uniform` and `osp` only; `all` yields every candidate and `manual` yields one channel per requested voltage (`filter_voltages`, list of floats).
- **Normalisation:** `--norm p99` (default) divides the cube by `intensity_p99 / 200` of that sample (from the CSV) so bright-pixel band values sit near 1; `--norm none` disables. The scale is a positive scalar, so applying it before or after the filter is identical.
- **Source of truth:** `.mat` files. `HCODh5/` (float16 Blosc cache) is not used: 22 ids missing, `242.h5` truncated, `22.h5` has 400 bands.

## Components

### `data_loader/my_dataset.py`

```python
class HyperCOD_data(Dataset.Dataset):
    def __init__(self, data_path, split='train', use_filter=True, filter_path=None,
                 num_filters=30, filter_select='uniform', filter_voltages=None,
                 crop_size=512, obj_crop_prob=0.5, norm='p99', seed=None)
```

`seed` (optional int) seeds the dataset's own `random.Random` used for crop sampling, for reproducible tests; `None` uses an unseeded generator.

Attributes after init: `img_name` (list of id strings), `wavelens [200]`, `sensor_wavelens [401]`, `voltages [351]`, `sensor_R_matrix [200, N]`, `selected_indices`, `selected_voltages`, `valid_band_mask [200]` (≤ 800 nm), `scale` (dict id → float, or None), `in_channels` (N or 200), `H, W = 1680, 1240`.

Methods:
- `load_sensor_response()` — load `.mat`, validate keys/shapes, align to `wavelens` (see Decisions), select channels.
- `select_filter_channels(R_all)` → indices, per `filter_select`.
- `load_intensity_scale()` — parse CSV into `self.scale` when `norm == 'p99'`.
- `crop_window(gt)` → `(h0, w0, ch, cw)`; training: with prob `obj_crop_prob` and a non-empty mask, pick a random foreground pixel `(hs, ws)` and draw `h0 ~ U[max(0, hs-ch+1), min(hs, H-ch)]` (same for w) so the crop contains it; otherwise uniform. Test or `crop_size == 0`: full frame.
- `load_gt(name)` → bool `[H, W]`.
- `read_cube_block(name, h0, w0, ch, cw)` → float32 `[200, cw, ch]` via `h5py.File(...)['hypercube'][:, w0:w0+cw, h0:h0+ch]` (file opened per call, worker-safe).
- `__getitem__(idx)` → `(img, gt, name)`:
  - `img`: float32 numpy `[C, ch, cw]`, C = N (filter: `np.tensordot(R.T, blk, axes=(1, 0))` then swap axes) or 200 (raw: swap axes only); scaled by `1/scale[name]` when `norm == 'p99'`.
  - `gt`: float32 numpy `[1, ch, cw]` in {0, 1}.
  - `name`: str.
- `__len__` → number of samples.

Module level:
- `image_collate_fn(batch)` → `img [B, C, H, W]` float32 tensor, `gt [B, 1, H, W]` float32 tensor, `names list[str]`.
- `osp(X, num_channels)` — copied from the previous project (index selection).
- `add_dataset_args(parser)` — adds `--data_path`, `--filter_path`, `--use_filter`, `--num_filters`, `--filter_select`, `--filter_voltages`, `--crop_size`, `--obj_crop_prob`, `--norm`.
- `build_dataset(args, split)` — maps the namespace onto the constructor (`use_filter=args.use_filter`, ...).
- `if __name__ == '__main__':` smoke test on one real train sample (prints shapes, selected voltages, timing).

### Constants

`WAVELENS_200 = np.linspace(400.0, 1000.0, 200)`, `FILTER_DEAD_ZONE_V = (0.25, 0.31)`, `GT_THRESHOLD = 127`, `N_BANDS = 200`.

## Data flow (`use_filter=True`, train)

```
GT png ──> bool mask [H,W] ──> crop_window ──> (h0,w0,ch,cw)
                                        │
cube.mat ──h5py hyperslab [:, w0:w0+cw, h0:h0+ch]──> blk [200, cw, ch] float32
                                        │  (÷ scale[name] if p99)
             sensor_R_matrix [200,N] ──tensordot──> [N, cw, ch] ──swap(1,2)──> img [N, ch, cw]
mask[h0:h0+ch, w0:w0+cw] ─────────────────────────────────────────────────> gt [1, ch, cw]
```

## Error handling

Asserts with f-string messages (previous style): `data_path`/split folders/filter file exist; every cube has a GT; filter `.mat` has the three keys with consistent shapes; `hypercube` shape is `(200, W, H)` matching the GT `(H, W)` (checked on the first sample at init); `crop_size <= min(H, W)`; `filter_select == 'manual'` requires `filter_voltages`; `num_filters <= number of candidates`. Printed warnings: a manual voltage inside the dead zone; a sample with an empty mask (falls back to a uniform crop). Files are never modified.

## Testing (`tests/test_my_dataset.py`, pytest)

Synthetic fixture in `tmp_path`: 2 train + 1 test samples with `H=48, W=40`, 200 random bands written with h5py as `(200, 40, 48)` under key `hypercube`; GT png with a 6×6 object; matching `intensity_p99_summary.csv`; synthetic filter `.mat` via `sio.savemat` with the real grid sizes (401 × 351) and analytically known curves (e.g. Gaussians whose centre moves with voltage, negative for V < 0.27).

Tests:
1. `sensor_R_matrix.shape == (200, N)`; row 0 equals the filter's 400 nm row; rows for bands > 800 nm are all zero; one interior band matches a hand-computed linear interpolation.
2. Raw mode `img.shape == (200, cs, cs)`; values equal the h5 block transposed.
3. Filter mode `img.shape == (N, cs, cs)` and equals `einsum('bn,bhw->nhw', R, raw_img)` within 1e-5.
4. `obj_crop_prob=1` → every crop contains at least one foreground pixel (50 draws); `obj_crop_prob=0` still returns valid crops.
5. `split='test'` → full frame `(C, H, W)` regardless of `crop_size`.
6. `image_collate_fn` shapes `(B, C, cs, cs)`, `(B, 1, cs, cs)`, `len(names) == B`.
7. `filter_select='manual'` maps requested voltages to nearest indices; no selected index lies in the dead zone for `uniform`.
8. `norm='p99'` divides by `p99/200`; `norm='none'` leaves values unchanged.
9. `in_channels` is N / 200 accordingly.

Plus the `__main__` smoke test on a real sample (manual).

## Out of scope

Augmentation beyond cropping, caching, RGB/intensity-map outputs, sliding-window inference, model code.
