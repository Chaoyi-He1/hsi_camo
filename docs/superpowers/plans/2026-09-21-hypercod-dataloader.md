# HyperCOD Dataloader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A PyTorch `Dataset` for HyperCOD that aligns the EC tunable-detector response to the cube wavelengths at `__init__` and, per sample, returns either the simulated detector channels (`use_filter=True`) or the raw 200 bands, plus the binary GT mask — read directly from the `.mat` cubes as native-resolution object-biased crops.

**Architecture:** `data_loader/ec_filter.py` holds pure functions for loading, aligning and selecting the sensor response; `data_loader/my_dataset.py` holds `HyperCOD_data(torch.utils.data.Dataset)`, the collate function and the argparse glue, in the style of `/data/chaoyi_he/HSI/Diffu/data_loader/my_dataset.py`. Cubes are MATLAB v7.3 (HDF5) files read with `h5py` hyperslabs in their native `[B, W, H]` layout; the filter projection is applied in that layout and only the last two axes are swapped to give `[C, H, W]`. Tests run on a tiny synthetic dataset written into `tmp_path`.

**Tech Stack:** Python 3.10 (`conda env hsi`), numpy 2.2, scipy 1.15 (`loadmat`, `interp1d`), h5py 3.14, Pillow 11, torch 2.8, pytest.

## Global Constraints

- Cube band centres are `np.linspace(400.0, 1000.0, 200)` nm; the `.mat` variable is `hypercube` with h5py shape `(200, 1240, 1680)` = `(B, W, H)`.
- Filter file `EC_filterV3.mat` variables: `Wavelength (1, 401)` 400–800 nm, `Voltage (1, 351)` −1.00..2.50 V, `responsivity (401, 351)` in [−1, 1], columns peak-normalised.
- Outside 400–800 nm the aligned response is **zero** (never extrapolated); the response is **never** min–max rescaled.
- Voltages in the dead zone `0.25 V ≤ V ≤ 0.31 V` are never selected automatically.
- GT foreground = channel 0 of the png `> 127`.
- `--norm p99` (default) divides the cube by `intensity_p99_valid / 200` from `<split>/intensity map/intensity_p99_summary.csv`.
- Training returns `crop_size × crop_size` crops (default 512) at native resolution; test returns the full frame. No down-sampling.
- Outputs: `img` float32 numpy `[C, H, W]`, `gt` float32 numpy `[1, H, W]` in {0, 1}, `name` str; collate → torch `[B, C, H, W]`, `[B, 1, H, W]`, `list[str]`.
- Code style: `import torch.utils.data as Dataset`, plain keyword arguments, `assert ... , f"..."` messages, shape comments `# [C, H, W]`, snake_case argparse flags, `__main__` smoke test.
- Every commit message ends with the two trailer lines shown in Task 1 Step 8.
- All commands run from the worktree root `/data/chaoyi_he/hsi_camo/.claude/worktrees/dataloader` with the `hsi` conda env active (`python` = `/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python`).

---

## File structure

| File | Responsibility |
|---|---|
| `pytest.ini` | `testpaths = tests`, `pythonpath = .` so `from data_loader...` imports work |
| `data_loader/__init__.py` | empty package marker |
| `data_loader/ec_filter.py` | constants (`N_BANDS`, `WAVELENS_200`, `FILTER_DEAD_ZONE_V`), `load_ec_filter`, `align_filter_to_wavelens`, `candidate_indices`, `osp`, `select_filter_channels` |
| `data_loader/my_dataset.py` | `HyperCOD_data`, `image_collate_fn`, `add_dataset_args`, `build_dataset`, `__main__` smoke test |
| `tests/__init__.py` | empty, so `from tests.conftest import ...` works |
| `tests/conftest.py` | `synthetic_root` fixture: 2 train + 1 test tiny samples, csv, synthetic filter `.mat` |
| `tests/test_ec_filter.py` | tests for `ec_filter.py` |
| `tests/test_my_dataset.py` | tests for the dataset, collate, args |

---

### Task 1: Scaffold, synthetic fixture, filter loading and alignment

**Files:**
- Create: `pytest.ini`, `data_loader/__init__.py`, `data_loader/ec_filter.py`, `tests/__init__.py`, `tests/conftest.py`, `tests/test_ec_filter.py`

**Interfaces:**
- Produces: `N_BANDS = 200`, `WAVELENS_200: np.ndarray [200]`, `FILTER_DEAD_ZONE_V = (0.25, 0.31)`, `load_ec_filter(filter_path) -> (sensor_wavelens [401], voltages [351], R [401, 351])`, `align_filter_to_wavelens(sensor_wavelens, R, wavelens) -> (R_aligned [len(wavelens), N], valid_band_mask [len(wavelens)] bool)`.
- Fixture `synthetic_root -> (root: pathlib.Path, info: dict[(split, name)] -> (cube [200, 40, 48] float32, gt [48, 40] bool, p99: float), filt: (wl [401], volt [351], R [401, 351]))`. Constants in conftest: `H, W, B = 48, 40, 200`; the object occupies `gt[10:16, 20:26]`; train names `['3', '10']`, test name `['7']`.

- [ ] **Step 1: Install pytest into the `hsi` env and create the scaffold**

```bash
python -m pip install -q pytest
mkdir -p data_loader tests
touch data_loader/__init__.py tests/__init__.py
printf '[pytest]\ntestpaths = tests\npythonpath = .\n' > pytest.ini
python -m pytest --version
```

Expected: a line like `pytest 8.x.y`.

- [ ] **Step 2: Write the synthetic fixture**

Create `tests/conftest.py`:

```python
import os
import numpy as np
import h5py
import scipy.io as sio
import pytest
from PIL import Image

# tiny stand-in for the real 1680 x 1240 x 200 cubes
H, W, B = 48, 40, 200
OBJ_SLICE = (slice(10, 16), slice(20, 26))   # 6 x 6 foreground object in every GT


def write_sample(root, split, name, rng):
    # MATLAB v7.3 stores the cube column-major, so h5py sees [B, W, H]
    cube = (rng.random((B, W, H), dtype=np.float32) * 0.02).astype(np.float32)
    with h5py.File(os.path.join(root, split, 'hyperspectral', f'{name}.mat'), 'w') as f:
        f.create_dataset('hypercube', data=cube)
    gt = np.zeros((H, W), dtype=np.uint8)
    gt[OBJ_SLICE] = 255
    # real GTs are 3-channel; write 3 identical channels
    Image.fromarray(np.stack([gt] * 3, axis=-1)).save(os.path.join(root, split, 'GT', f'{name}.png'))
    p99 = float(np.percentile(cube.sum(axis=0), 99))   # intensity = sum over bands
    return cube, gt > 127, p99


@pytest.fixture
def synthetic_root(tmp_path):
    rng = np.random.default_rng(0)
    root = tmp_path / 'HyperCOD'
    info = {}
    for split, names in (('train', ['3', '10']), ('test', ['7'])):
        for sub in ('hyperspectral', 'GT', 'intensity map'):
            (root / split / sub).mkdir(parents=True)
        rows = []
        for name in names:
            cube, gt, p99 = write_sample(str(root), split, name, rng)
            info[(split, name)] = (cube, gt, p99)
            rows.append((name, p99))
        with open(root / split / 'intensity map' / 'intensity_p99_summary.csv', 'w') as f:
            f.write('sample_id,mat_path,intensity_map_path,height,width,bands,n_valid,'
                    'intensity_p99_valid,intensity_min_valid,intensity_mean_valid,intensity_max_valid\n')
            for name, p99 in rows:
                f.write(f'{name},x,y,{H},{W},{B},{H * W},{p99},0,0,0\n')

    # synthetic EC filter on the real grid sizes: a Gaussian whose centre sweeps 400 -> 800 nm
    # with voltage, negative below the 0.27 V polarity flip (like the real device)
    wl = np.arange(400, 801, dtype=np.float64)                  # [401]
    volt = np.round(np.arange(-1.0, 2.5001, 0.01), 2)           # [351]
    centre = 400.0 + (volt - volt.min()) / (volt.max() - volt.min()) * 400.0
    R = np.exp(-0.5 * ((wl[:, None] - centre[None, :]) / 40.0) ** 2)   # [401, 351], peak 1
    R[:, volt < 0.27] *= -1.0
    sio.savemat(str(root / 'EC_filterV3.mat'),
                {'Wavelength': wl[None].astype(np.uint16), 'Voltage': volt[None], 'responsivity': R})
    return root, info, (wl, volt, R)
```

- [ ] **Step 3: Write the failing tests for loading and alignment**

Create `tests/test_ec_filter.py`:

```python
import numpy as np
import pytest

from data_loader.ec_filter import (N_BANDS, WAVELENS_200, load_ec_filter, align_filter_to_wavelens)


def test_wavelens_constant():
    assert N_BANDS == 200
    assert WAVELENS_200.shape == (200,)
    assert WAVELENS_200[0] == 400.0 and WAVELENS_200[-1] == 1000.0
    assert np.isclose(WAVELENS_200[1] - WAVELENS_200[0], 600.0 / 199.0)


def test_load_ec_filter_shapes(synthetic_root):
    root, _, (wl, volt, R) = synthetic_root
    sensor_wavelens, voltages, R_loaded = load_ec_filter(str(root / 'EC_filterV3.mat'))
    assert sensor_wavelens.shape == (401,) and voltages.shape == (351,) and R_loaded.shape == (401, 351)
    assert sensor_wavelens.dtype == np.float64
    np.testing.assert_allclose(sensor_wavelens, wl)
    np.testing.assert_allclose(voltages, volt)
    np.testing.assert_allclose(R_loaded, R)


def test_load_ec_filter_missing_key(tmp_path):
    import scipy.io as sio
    bad = tmp_path / 'bad.mat'
    sio.savemat(str(bad), {'Wavelength': np.arange(3)[None], 'Voltage': np.arange(2)[None]})
    with pytest.raises(AssertionError, match="responsivity"):
        load_ec_filter(str(bad))


def test_align_exact_at_400_zero_above_800(synthetic_root):
    _, _, (wl, volt, R) = synthetic_root
    R200, valid = align_filter_to_wavelens(wl, R, WAVELENS_200)
    assert R200.shape == (200, 351)
    assert valid.dtype == bool and valid.sum() == 133          # bands 0..132 are <= 800 nm
    assert valid[132] and not valid[133]
    np.testing.assert_allclose(R200[0], R[0])                   # 400 nm lies exactly on the sensor grid
    assert np.all(R200[133:] == 0.0)                            # sensor is blind above 800 nm: zero, not extrapolated
    # band 1 sits at 403.015 nm, between sensor rows 403 and 404 -> linear interpolation
    frac = WAVELENS_200[1] - 403.0
    np.testing.assert_allclose(R200[1], R[3] + frac * (R[4] - R[3]), rtol=1e-12, atol=1e-12)


def test_align_keeps_negative_lobes(synthetic_root):
    _, _, (wl, volt, R) = synthetic_root
    R200, _ = align_filter_to_wavelens(wl, R, WAVELENS_200)
    neg_cols = volt < 0.27
    assert R200[:133, neg_cols].max() <= 0.0 and R200[:133, neg_cols].min() < -0.9
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `python -m pytest tests/test_ec_filter.py -v`
Expected: `ModuleNotFoundError: No module named 'data_loader.ec_filter'` (collection error).

- [ ] **Step 5: Implement constants, loading and alignment**

Create `data_loader/ec_filter.py`:

```python
import numpy as np
import scipy.io as sio
from scipy.interpolate import interp1d

# HyperCOD cube band centres: 200 bands, 400 nm to 1000 nm, 3.015 nm step (Code/wl_200bands.npy)
N_BANDS = 200
WAVELENS_200 = np.linspace(400.0, 1000.0, N_BANDS)

# The EC detector's photocurrent flips sign at ~0.27-0.29 V and is ~0 there,
# so these bias voltages are excluded from automatic channel selection
FILTER_DEAD_ZONE_V = (0.25, 0.31)


def load_ec_filter(filter_path):
    '''
    Load the EC tunable detector response (EC_filterV3.mat, MATLAB v5).
    Returns:
        sensor_wavelens: [C_s] nm, 401 points from 400 to 800 nm
        voltages: [N_all] V, 351 points from -1.00 to 2.50 V
        R: [C_s, N_all] responsivity, each column peak-normalized to |max| = 1, may be negative
    '''
    mat = sio.loadmat(filter_path)
    for key in ('Wavelength', 'Voltage', 'responsivity'):
        assert key in mat, \
            f"{filter_path} has no variable '{key}', found {[k for k in mat if not k.startswith('__')]}"
    sensor_wavelens = np.asarray(mat['Wavelength'], dtype=np.float64).reshape(-1)
    voltages = np.asarray(mat['Voltage'], dtype=np.float64).reshape(-1)
    R = np.asarray(mat['responsivity'], dtype=np.float64)
    assert R.shape == (len(sensor_wavelens), len(voltages)), \
        f"responsivity shape {R.shape} does not match ({len(sensor_wavelens)} wavelengths, {len(voltages)} voltages)"
    return sensor_wavelens, voltages, R


def align_filter_to_wavelens(sensor_wavelens, R, wavelens):
    '''
    Linearly resample the sensor response onto the cube band centres.
    Bands outside the measured sensor range get ZERO response (the sensor is blind there);
    extrapolating the edge values would invent a response the device does not have.
    Args:
        sensor_wavelens: [C_s]
        R: [C_s, N]
        wavelens: [C]
    Returns:
        R_aligned: [C, N]
        valid_band_mask: [C] bool, True where the band lies inside the sensor range
    '''
    f = interp1d(sensor_wavelens, R, axis=0, kind='linear', bounds_error=False, fill_value=0.0)
    R_aligned = f(wavelens)
    valid_band_mask = (wavelens >= sensor_wavelens.min()) & (wavelens <= sensor_wavelens.max())
    return R_aligned, valid_band_mask
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `python -m pytest tests/test_ec_filter.py -v`
Expected: `5 passed`.

- [ ] **Step 7: Confirm the alignment against the real filter file**

Run:
```bash
python -c "
from data_loader.ec_filter import *
wl, v, R = load_ec_filter('/data2/chaoyi/HyperCOD/Raw data/EC_filterV3.mat')
R200, valid = align_filter_to_wavelens(wl, R, WAVELENS_200)
print(R200.shape, int(valid.sum()), float(R200[:133].min()), float(R200[:133].max()), float(abs(R200[133:]).max()))"
```
Expected: `(200, 351) 133 -1.0 1.0 0.0` (min/max may be −0.99…/0.99… because band centres fall between the 1 nm grid points).

- [ ] **Step 8: Commit**

```bash
git add pytest.ini data_loader/__init__.py data_loader/ec_filter.py tests/__init__.py tests/conftest.py tests/test_ec_filter.py
git commit -m "Add EC filter loading and wavelength alignment with synthetic test fixture

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QB4DScDwiA2DkVpbDopffV"
```

---

### Task 2: Voltage channel selection

**Files:**
- Modify: `data_loader/ec_filter.py` (append)
- Test: `tests/test_ec_filter.py` (append)

**Interfaces:**
- Consumes: `FILTER_DEAD_ZONE_V`, aligned `R [C, N_all]`, `voltages [N_all]` from Task 1.
- Produces: `candidate_indices(voltages, dead_zone=FILTER_DEAD_ZONE_V) -> np.ndarray[int]`, `osp(X, num_channels) -> (X_selected, selected_indices: list[int])`, `select_filter_channels(R, voltages, num_filters=30, mode='uniform', filter_voltages=None, dead_zone=FILTER_DEAD_ZONE_V) -> list[int]` (indices into the voltage axis; modes `'uniform' | 'osp' | 'all' | 'manual'`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ec_filter.py`:

```python
from data_loader.ec_filter import FILTER_DEAD_ZONE_V, candidate_indices, osp, select_filter_channels


def _aligned(synthetic_root):
    _, _, (wl, volt, R) = synthetic_root
    R200, _ = align_filter_to_wavelens(wl, R, WAVELENS_200)
    return R200, volt


def _in_dead_zone(v):
    lo, hi = FILTER_DEAD_ZONE_V
    return (v >= lo) & (v <= hi)


def test_candidate_indices_exclude_dead_zone(synthetic_root):
    _, volt = _aligned(synthetic_root)
    cand = candidate_indices(volt)
    assert len(cand) == 351 - 7                                  # 0.25, 0.26, ..., 0.31 excluded
    assert not _in_dead_zone(volt[cand]).any()
    assert np.all(np.diff(cand) >= 1)


def test_select_uniform(synthetic_root):
    R200, volt = _aligned(synthetic_root)
    sel = select_filter_channels(R200, volt, num_filters=30, mode='uniform')
    assert len(sel) == 30 and len(set(sel)) == 30
    assert sel[0] == 0 and sel[-1] == 350                        # spans the full voltage range
    assert all(isinstance(i, int) for i in sel)
    assert not _in_dead_zone(volt[sel]).any()


def test_select_uniform_bounds(synthetic_root):
    R200, volt = _aligned(synthetic_root)
    assert select_filter_channels(R200, volt, num_filters=1, mode='uniform') == [0]
    with pytest.raises(AssertionError):
        select_filter_channels(R200, volt, num_filters=400, mode='uniform')


def test_select_all(synthetic_root):
    R200, volt = _aligned(synthetic_root)
    sel = select_filter_channels(R200, volt, mode='all')
    assert sel == candidate_indices(volt).tolist()


def test_select_manual_nearest(synthetic_root):
    R200, volt = _aligned(synthetic_root)
    sel = select_filter_channels(R200, volt, mode='manual', filter_voltages=[-0.5, 1.004, 2.5])
    np.testing.assert_allclose(volt[sel], [-0.5, 1.0, 2.5])
    with pytest.raises(AssertionError, match="filter_voltages"):
        select_filter_channels(R200, volt, mode='manual')


def test_select_manual_warns_in_dead_zone(synthetic_root, capsys):
    R200, volt = _aligned(synthetic_root)
    sel = select_filter_channels(R200, volt, mode='manual', filter_voltages=[0.28])
    assert np.isclose(volt[sel[0]], 0.28)
    assert "dead zone" in capsys.readouterr().out


def test_select_osp(synthetic_root):
    R200, volt = _aligned(synthetic_root)
    sel = select_filter_channels(R200, volt, num_filters=5, mode='osp')
    assert len(sel) == 5 and len(set(sel)) == 5
    assert not _in_dead_zone(volt[sel]).any()


def test_osp_returns_indices_into_input():
    X = np.eye(6)[:, :4]                                        # 4 orthogonal columns
    X_sel, idx = osp(X, 3)
    assert len(idx) == 3 and len(set(idx)) == 3 and X_sel.shape == (6, 3)


def test_select_unknown_mode(synthetic_root):
    R200, volt = _aligned(synthetic_root)
    with pytest.raises(ValueError):
        select_filter_channels(R200, volt, mode='random')
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_ec_filter.py -v`
Expected: `ImportError: cannot import name 'candidate_indices'`.

- [ ] **Step 3: Implement selection**

Append to `data_loader/ec_filter.py`:

```python
def candidate_indices(voltages, dead_zone=FILTER_DEAD_ZONE_V):
    '''Indices of the voltages outside the polarity-flip dead zone (inclusive bounds, 1e-6 V tolerance).'''
    lo, hi = dead_zone
    return np.where((voltages < lo - 1e-6) | (voltages > hi + 1e-6))[0]


def osp(X, num_channels):
    ''' Orthogonal Subspace Projection (OSP) algorithm to
    select a subset of channels from the sensor response matrix X.
    (Copied from HSI/Diffu/data_loader/my_dataset.py.)
    Args:
        X: Sensor response matrix of shape (C, N), where C is the number of channels, N is the number of sensor responses
        num_channels: Number of channels to select

    Returns:
        Selected channels from the sensor response matrix and their indices
    '''
    C, N = X.shape

    X_min, X_max = X.min(), X.max()
    X = (X - X_min) / (X_max - X_min)  # Normalize to [0, 1]

    selected_indices = []
    residual = X.copy().astype(np.float64)  # Use float64 for better numerical stability

    for _ in range(num_channels):
        # Compute the norms of each sensor response in the residual
        norms = np.linalg.norm(residual, axis=0)  # [N]
        selected_index = int(np.argmax(norms))
        selected_indices.append(selected_index)

        # Get the selected sensor response
        selected_response = residual[:, selected_index].reshape(-1, 1)  # [C, 1]

        # Avoid division by zero - compute squared norm (||v||²)
        norm_squared = np.linalg.norm(selected_response) ** 2
        if norm_squared < 1e-12:
            break

        # Project out the selected response from all remaining columns
        projection_matrix = (selected_response @ selected_response.T) / norm_squared  # [C, C]
        residual = residual - projection_matrix @ residual

        # Set the selected column to zero to avoid selecting it again
        residual[:, selected_index] = 0

    return X[:, selected_indices], selected_indices


def select_filter_channels(R, voltages, num_filters=30, mode='uniform', filter_voltages=None,
                           dead_zone=FILTER_DEAD_ZONE_V):
    '''
    Choose which bias voltages (columns of R) become the model's input channels.
    Args:
        R: [C, N_all] aligned response
        voltages: [N_all]
        num_filters: used by 'uniform' and 'osp' only
        mode: 'uniform' - evenly spaced over the candidate voltages (as in the previous project)
              'osp'     - OSP selection among the candidates
              'all'     - every candidate voltage
              'manual'  - nearest voltage to each value in filter_voltages
    Returns:
        selected_indices: list[int] into the voltage axis
    '''
    cand = candidate_indices(voltages, dead_zone)
    assert len(cand) > 0, "no candidate voltages outside the dead zone"
    if mode == 'uniform':
        assert 1 <= num_filters <= len(cand), f"num_filters={num_filters} must be in [1, {len(cand)}]"
        pos = np.linspace(0, len(cand) - 1, num_filters).round().astype(int)
        return cand[pos].tolist()
    if mode == 'osp':
        assert 1 <= num_filters <= len(cand), f"num_filters={num_filters} must be in [1, {len(cand)}]"
        _, sel = osp(R[:, cand], num_filters)
        if len(sel) < num_filters:
            print(f"Warning: OSP stopped at {len(sel)} channels (residual exhausted), requested {num_filters}")
        return cand[np.asarray(sel, dtype=int)].tolist()
    if mode == 'all':
        return cand.tolist()
    if mode == 'manual':
        assert filter_voltages is not None and len(filter_voltages) > 0, \
            "filter_select='manual' requires filter_voltages"
        sel = []
        for v in filter_voltages:
            idx = int(np.argmin(np.abs(voltages - v)))
            if dead_zone[0] - 1e-6 <= voltages[idx] <= dead_zone[1] + 1e-6:
                print(f"Warning: requested voltage {v:+.2f} V maps to {voltages[idx]:+.2f} V inside the dead zone {dead_zone}")
            sel.append(idx)
        return sel
    raise ValueError(f"unknown filter_select mode '{mode}', expected uniform | osp | all | manual")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_ec_filter.py -v`
Expected: `14 passed`.

- [ ] **Step 5: Commit**

```bash
git add data_loader/ec_filter.py tests/test_ec_filter.py
git commit -m "Add dead-zone-aware voltage channel selection (uniform, osp, all, manual)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QB4DScDwiA2DkVpbDopffV"
```

---

### Task 3: `HyperCOD_data.__init__` — samples, layout check, sensor alignment, intensity scale

**Files:**
- Create: `data_loader/my_dataset.py`, `tests/test_my_dataset.py`

**Interfaces:**
- Consumes: everything from `data_loader/ec_filter.py`.
- Produces: `class HyperCOD_data(Dataset.Dataset)` with `__init__(self, data_path, split='train', use_filter=True, filter_path=None, num_filters=30, filter_select='uniform', filter_voltages=None, crop_size=512, obj_crop_prob=0.5, norm='p99', seed=None)`, `__len__`, attributes `img_name: list[str]`, `wavelens [200]`, `sensor_wavelens [401]`, `voltages [351]`, `valid_band_mask [200] bool`, `selected_indices: list[int]`, `selected_voltages [N]`, `sensor_R_matrix [200, N] float32`, `scale: dict[str, float] | None`, `in_channels: int`, `H: int`, `W: int`, `rng` (a `random.Random` when `seed` is given, else the `random` module); methods `load_sensor_response()`, `load_intensity_scale() -> dict`, `load_gt(name) -> bool [H, W]`, `check_cube_layout(name)`. Constants `GT_THRESHOLD = 127`, `HYPERCUBE_KEY = 'hypercube'`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_my_dataset.py`:

```python
import argparse
import numpy as np
import pytest
import torch

from data_loader.ec_filter import N_BANDS, WAVELENS_200, FILTER_DEAD_ZONE_V
from data_loader.my_dataset import HyperCOD_data
from tests.conftest import H, W, OBJ_SLICE


def make(root, **kw):
    kw.setdefault('split', 'train')
    kw.setdefault('crop_size', 16)
    kw.setdefault('norm', 'none')
    kw.setdefault('seed', 0)
    return HyperCOD_data(data_path=str(root), **kw)


def test_init_lists_samples_sorted_numerically(synthetic_root):
    root, _, _ = synthetic_root
    ds = make(root)
    assert ds.img_name == ['3', '10'] and len(ds) == 2
    assert (ds.H, ds.W) == (H, W)
    assert make(root, split='test').img_name == ['7']


def test_init_aligns_filter_and_selects_channels(synthetic_root):
    root, _, (wl, volt, R) = synthetic_root
    ds = make(root, num_filters=30)
    np.testing.assert_allclose(ds.wavelens, WAVELENS_200)
    assert ds.sensor_R_matrix.shape == (N_BANDS, 30) and ds.sensor_R_matrix.dtype == np.float32
    assert ds.valid_band_mask.sum() == 133
    assert np.all(ds.sensor_R_matrix[133:] == 0.0)
    assert len(ds.selected_indices) == 30
    np.testing.assert_allclose(ds.selected_voltages, volt[ds.selected_indices])
    lo, hi = FILTER_DEAD_ZONE_V
    assert not ((ds.selected_voltages >= lo) & (ds.selected_voltages <= hi)).any()
    # the aligned matrix must keep the negative lobes (no min-max rescaling)
    assert ds.sensor_R_matrix.min() < -0.9 and ds.sensor_R_matrix.max() > 0.9


def test_in_channels_follows_use_filter(synthetic_root):
    root, _, _ = synthetic_root
    assert make(root, use_filter=True, num_filters=12).in_channels == 12
    ds_raw = make(root, use_filter=False)
    assert ds_raw.in_channels == N_BANDS
    assert ds_raw.sensor_R_matrix.shape == (N_BANDS, 30)        # aligned at init regardless


def test_init_manual_voltages(synthetic_root):
    root, _, _ = synthetic_root
    ds = make(root, filter_select='manual', filter_voltages=[-0.5, 1.0])
    np.testing.assert_allclose(ds.selected_voltages, [-0.5, 1.0])
    assert ds.in_channels == 2


def test_intensity_scale(synthetic_root):
    root, info, _ = synthetic_root
    ds = make(root, norm='p99')
    assert set(ds.scale) == {'3', '10'}
    assert np.isclose(ds.scale['3'], info[('train', '3')][2] / N_BANDS)
    assert make(root, norm='none').scale is None


def test_default_filter_path(synthetic_root):
    root, _, _ = synthetic_root
    ds = make(root)
    assert ds.filter_path == str(root / 'EC_filterV3.mat')


def test_init_asserts(synthetic_root, tmp_path):
    root, _, _ = synthetic_root
    with pytest.raises(AssertionError, match="split"):
        make(root, split='val')
    with pytest.raises(AssertionError, match="crop_size"):
        make(root, crop_size=W + 1)
    with pytest.raises(AssertionError, match="norm"):
        make(root, norm='minmax')
    with pytest.raises(AssertionError, match="does not exist"):
        make(root, filter_path=str(tmp_path / 'missing.mat'))
    with pytest.raises(AssertionError, match="does not exist"):
        HyperCOD_data(data_path=str(tmp_path / 'nowhere'))


def test_load_gt_thresholds_channel_0(synthetic_root):
    root, info, _ = synthetic_root
    ds = make(root)
    gt = ds.load_gt('3')
    assert gt.dtype == bool and gt.shape == (H, W)
    np.testing.assert_array_equal(gt, info[('train', '3')][1])
    assert gt[OBJ_SLICE].all() and gt.sum() == 36
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_my_dataset.py -v`
Expected: `ModuleNotFoundError: No module named 'data_loader.my_dataset'`.

- [ ] **Step 3: Implement the class skeleton and `__init__`**

Create `data_loader/my_dataset.py`:

```python
import torch.utils.data as Dataset
import os
import csv
import random
import argparse
import numpy as np
import h5py
import torch
from PIL import Image

from data_loader.ec_filter import (N_BANDS, WAVELENS_200, load_ec_filter, align_filter_to_wavelens,
                                   select_filter_channels)

GT_THRESHOLD = 127            # GT pngs are JPEG-compressed with 3 identical channels; foreground = channel 0 > 127
HYPERCUBE_KEY = 'hypercube'   # variable name inside the MATLAB v7.3 (HDF5) .mat cubes


class HyperCOD_data(Dataset.Dataset):
    def __init__(self, data_path, split='train', use_filter=True, filter_path=None, num_filters=30,
                 filter_select='uniform', filter_voltages=None, crop_size=512, obj_crop_prob=0.5,
                 norm='p99', seed=None):
        super(HyperCOD_data, self).__init__()
        self.data_path = data_path
        self.split = split  # 'train' or 'test'
        self.use_filter = use_filter  # True: simulated EC detector channels, False: raw 200 bands
        self.filter_path = filter_path if filter_path is not None else os.path.join(data_path, 'EC_filterV3.mat')
        self.num_filters = num_filters
        self.filter_select = filter_select
        self.filter_voltages = filter_voltages
        self.crop_size = crop_size  # training crop at native resolution, 0 means full frame
        self.obj_crop_prob = obj_crop_prob  # probability that a training crop is placed to contain the object
        self.norm = norm  # 'none' or 'p99'
        # the random module is re-seeded per DataLoader worker by PyTorch; a fixed seed is for reproducible tests
        self.rng = random.Random(seed) if seed is not None else random

        assert self.split in ['train', 'test'], "split must be 'train' or 'test'"
        assert self.norm in ['none', 'p99'], "norm must be 'none' or 'p99'"
        assert os.path.exists(self.data_path), f"Data path {self.data_path} does not exist"
        assert os.path.exists(self.filter_path), f"Filter file {self.filter_path} does not exist"

        # Load data path: <split>/hyperspectral/<id>.mat, <split>/GT/<id>.png, <split>/intensity map/*.csv
        self.hsi_path = os.path.join(self.data_path, self.split, 'hyperspectral')
        self.gt_path = os.path.join(self.data_path, self.split, 'GT')
        self.intensity_path = os.path.join(self.data_path, self.split, 'intensity map')
        assert os.path.isdir(self.hsi_path), f"{self.hsi_path} does not exist"
        assert os.path.isdir(self.gt_path), f"{self.gt_path} does not exist"

        self.img_name = sorted([os.path.splitext(f)[0] for f in os.listdir(self.hsi_path) if f.endswith('.mat')], key=int)
        assert len(self.img_name) > 0, f"no .mat files found in {self.hsi_path}"
        for name in self.img_name:
            assert os.path.exists(os.path.join(self.gt_path, f'{name}.png')), f"GT for sample {name} not found in {self.gt_path}"

        # self.wavelens is shape [200], 400 nm to 1000 nm, uniform 3.015 nm step
        self.wavelens = WAVELENS_200.copy()

        # image size from the first GT; the cube layout is verified against it
        self.H, self.W = self.load_gt(self.img_name[0]).shape
        self.check_cube_layout(self.img_name[0])
        if self.split == 'train' and self.crop_size > 0:
            assert self.crop_size <= min(self.H, self.W), f"crop_size {self.crop_size} exceeds image size ({self.H}, {self.W})"

        print(f"Loading sensor response from {self.filter_path}...")
        self.load_sensor_response()

        self.scale = self.load_intensity_scale() if self.norm == 'p99' else None

        self.in_channels = self.sensor_R_matrix.shape[1] if self.use_filter else N_BANDS

    def __len__(self):
        return len(self.img_name)

    def load_sensor_response(self):
        '''
        Load the EC detector response [401 wavelengths, 351 voltages], resample it onto the cube band
        centres (zero outside 400-800 nm) and keep the selected voltage columns.
        Result: self.sensor_R_matrix [C, N], C = 200 bands, N = number of selected voltages.
        The columns are already peak-normalized to |max| = 1 and contain negative lobes, so unlike the
        previous project there is NO per-column min-max normalization here (it would destroy the signs).
        '''
        self.sensor_wavelens, self.voltages, R_raw = load_ec_filter(self.filter_path)   # [C_s], [N_all], [C_s, N_all]
        R_aligned, self.valid_band_mask = align_filter_to_wavelens(self.sensor_wavelens, R_raw, self.wavelens)  # [C, N_all]
        self.selected_indices = select_filter_channels(R_aligned, self.voltages, num_filters=self.num_filters,
                                                       mode=self.filter_select, filter_voltages=self.filter_voltages)
        self.selected_voltages = self.voltages[self.selected_indices]  # [N]
        self.sensor_R_matrix = R_aligned[:, self.selected_indices].astype(np.float32)  # [C, N]
        print(f"Sensor response aligned: {self.sensor_R_matrix.shape[0]} bands x {self.sensor_R_matrix.shape[1]} channels, "
              f"{int(self.valid_band_mask.sum())} bands inside the sensor range, "
              f"voltages {np.round(self.selected_voltages, 2).tolist()}")

    def load_intensity_scale(self):
        '''
        Per-sample scale = intensity_p99_valid / N_BANDS from intensity_p99_summary.csv, where the
        intensity map is the sum of the 200 bands. Dividing the cube by it puts bright-pixel band values
        near 1 and removes the ~40x scene-to-scene brightness spread.
        '''
        csv_path = os.path.join(self.intensity_path, 'intensity_p99_summary.csv')
        assert os.path.exists(csv_path), f"{csv_path} not found (needed for norm='p99')"
        scale = {}
        with open(csv_path, newline='') as f:
            for row in csv.DictReader(f):
                scale[row['sample_id']] = float(row['intensity_p99_valid']) / N_BANDS
        for name in self.img_name:
            assert name in scale, f"sample {name} missing from {csv_path}"
            assert scale[name] > 0, f"non-positive intensity p99 for sample {name}"
        return scale

    def load_gt(self, name):
        '''GT pngs are JPEG-compressed with 3 identical channels; foreground = channel 0 > 127. Returns bool [H, W].'''
        gt = np.array(Image.open(os.path.join(self.gt_path, f'{name}.png')))
        if gt.ndim == 3:
            gt = gt[:, :, 0]
        return gt > GT_THRESHOLD

    def check_cube_layout(self, name):
        '''MATLAB v7.3 stores the cube column-major, so h5py sees [B, W, H]; verify against the GT size.'''
        with h5py.File(os.path.join(self.hsi_path, f'{name}.mat'), 'r') as f:
            assert HYPERCUBE_KEY in f, f"{name}.mat has no '{HYPERCUBE_KEY}' variable, found {list(f.keys())}"
            shape = f[HYPERCUBE_KEY].shape
        assert shape == (N_BANDS, self.W, self.H), \
            f"cube {name} has shape {shape}, expected (bands, W, H) = ({N_BANDS}, {self.W}, {self.H})"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_my_dataset.py -v`
Expected: `8 passed`.

- [ ] **Step 5: Commit**

```bash
git add data_loader/my_dataset.py tests/test_my_dataset.py
git commit -m "Add HyperCOD_data init: sample listing, cube layout check, filter alignment, p99 scale

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QB4DScDwiA2DkVpbDopffV"
```

---

### Task 4: Crop window, cube block reading, raw-band `__getitem__`

**Files:**
- Modify: `data_loader/my_dataset.py` (append methods to `HyperCOD_data`)
- Test: `tests/test_my_dataset.py` (append)

**Interfaces:**
- Consumes: `HyperCOD_data` attributes from Task 3.
- Produces: `crop_window(gt: bool [H, W]) -> (h0, w0, ch, cw)`, `read_cube_block(name, h0, w0, ch, cw) -> float32 [200, cw, ch]`, `__getitem__(idx) -> (img float32 [C, ch, cw], gt float32 [1, ch, cw], name str)` — raw bands only in this task (`C = 200`); Task 5 adds the filter branch and normalisation.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_my_dataset.py`:

```python
def test_read_cube_block_matches_h5_layout(synthetic_root):
    root, info, _ = synthetic_root
    ds = make(root)
    cube = info[('train', '3')][0]                                # [B, W, H]
    blk = ds.read_cube_block('3', h0=5, w0=7, ch=16, cw=12)
    assert blk.shape == (N_BANDS, 12, 16) and blk.dtype == np.float32
    np.testing.assert_array_equal(blk, cube[:, 7:19, 5:21])


def test_crop_window_full_frame_for_test_split_or_zero_crop(synthetic_root):
    root, info, _ = synthetic_root
    gt = info[('train', '3')][1]
    assert make(root, split='test').crop_window(info[('test', '7')][1]) == (0, 0, H, W)
    assert make(root, crop_size=0).crop_window(gt) == (0, 0, H, W)


def test_crop_window_object_biased_contains_object(synthetic_root):
    root, info, _ = synthetic_root
    ds = make(root, crop_size=8, obj_crop_prob=1.0, seed=1)
    gt = info[('train', '3')][1]
    for _ in range(200):
        h0, w0, ch, cw = ds.crop_window(gt)
        assert (ch, cw) == (8, 8)
        assert 0 <= h0 <= H - 8 and 0 <= w0 <= W - 8
        assert gt[h0:h0 + 8, w0:w0 + 8].any()


def test_crop_window_uniform_stays_in_bounds(synthetic_root):
    root, info, _ = synthetic_root
    ds = make(root, crop_size=16, obj_crop_prob=0.0, seed=2)
    gt = info[('train', '3')][1]
    seen_outside_object = False
    for _ in range(200):
        h0, w0, ch, cw = ds.crop_window(gt)
        assert 0 <= h0 <= H - 16 and 0 <= w0 <= W - 16
        seen_outside_object |= not gt[h0:h0 + 16, w0:w0 + 16].any()
    assert seen_outside_object                                   # uniform crops do miss the object sometimes


def test_crop_window_empty_mask_falls_back_to_uniform(synthetic_root):
    root, _, _ = synthetic_root
    ds = make(root, crop_size=16, obj_crop_prob=1.0, seed=3)
    h0, w0, ch, cw = ds.crop_window(np.zeros((H, W), dtype=bool))
    assert (ch, cw) == (16, 16) and 0 <= h0 <= H - 16 and 0 <= w0 <= W - 16


def test_getitem_raw_full_frame_values(synthetic_root):
    root, info, _ = synthetic_root
    ds = make(root, split='test', use_filter=False)
    img, gt, name = ds[0]
    cube, gt_true, _ = info[('test', '7')]
    assert name == '7'
    assert img.shape == (N_BANDS, H, W) and img.dtype == np.float32 and img.flags['C_CONTIGUOUS']
    np.testing.assert_array_equal(img, cube.transpose(0, 2, 1))   # [B, W, H] -> [B, H, W]
    assert gt.shape == (1, H, W) and gt.dtype == np.float32
    np.testing.assert_array_equal(gt[0], gt_true.astype(np.float32))


def test_getitem_raw_crop(synthetic_root):
    root, info, _ = synthetic_root
    ds = make(root, use_filter=False, crop_size=16, obj_crop_prob=1.0, seed=4)
    img, gt, name = ds[1]
    assert name == '10'
    assert img.shape == (N_BANDS, 16, 16) and gt.shape == (1, 16, 16)
    assert gt.sum() > 0
    assert set(np.unique(gt)) <= {0.0, 1.0}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_my_dataset.py -v`
Expected: the 7 new tests fail with `AttributeError: 'HyperCOD_data' object has no attribute 'read_cube_block'` / `'crop_window'` / `TypeError: 'HyperCOD_data' object is not subscriptable`; the 8 earlier tests still pass.

- [ ] **Step 3: Implement crop window, block read and raw `__getitem__`**

Append inside `class HyperCOD_data` in `data_loader/my_dataset.py` (after `check_cube_layout`):

```python
    def read_cube_block(self, name, h0, w0, ch, cw):
        '''
        Read a spatial window of the cube directly from the .mat (HDF5) file.
        h5py layout is [B, W, H], so W is indexed with w0:w0+cw and H with h0:h0+ch.
        Returns float32 [B, cw, ch]. The file is opened per call so DataLoader workers stay independent.
        '''
        with h5py.File(os.path.join(self.hsi_path, f'{name}.mat'), 'r') as f:
            blk = f[HYPERCUBE_KEY][:, w0:w0 + cw, h0:h0 + ch]
        return np.asarray(blk, dtype=np.float32)

    def crop_window(self, gt):
        '''
        Choose the crop (h0, w0, ch, cw).
        Train with crop_size > 0: with probability obj_crop_prob (and a non-empty mask) pick a random
        foreground pixel and place the crop uniformly among the positions that contain it; otherwise a
        uniform random crop. Test split or crop_size == 0: the full frame.
        gt: bool [H, W]
        '''
        H, W = gt.shape
        if self.split != 'train' or self.crop_size <= 0:
            return 0, 0, H, W
        cs = self.crop_size
        fg_h, fg_w = np.nonzero(gt)
        if len(fg_h) > 0 and self.rng.random() < self.obj_crop_prob:
            k = self.rng.randrange(len(fg_h))
            hs, ws = int(fg_h[k]), int(fg_w[k])
            # top-left must satisfy h0 <= hs <= h0 + cs - 1 and 0 <= h0 <= H - cs (same for w)
            h0 = self.rng.randint(max(0, hs - cs + 1), min(hs, H - cs))
            w0 = self.rng.randint(max(0, ws - cs + 1), min(ws, W - cs))
        else:
            h0 = self.rng.randint(0, H - cs)
            w0 = self.rng.randint(0, W - cs)
        return h0, w0, cs, cs

    def __getitem__(self, idx):
        name = self.img_name[idx]
        gt = self.load_gt(name)  # [H, W] bool
        h0, w0, ch, cw = self.crop_window(gt)

        blk = self.read_cube_block(name, h0, w0, ch, cw)  # [B, cw, ch] float32
        img = blk  # raw bands

        # only the last two axes are swapped: [C, cw, ch] -> [C, ch, cw]; never build [H, W, B] (6 s per crop)
        img = np.ascontiguousarray(img.transpose(0, 2, 1), dtype=np.float32)  # [C, ch, cw]
        gt = gt[h0:h0 + ch, w0:w0 + cw].astype(np.float32)[None]  # [1, ch, cw]
        return img, gt, name
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_my_dataset.py -v`
Expected: `15 passed`.

- [ ] **Step 5: Commit**

```bash
git add data_loader/my_dataset.py tests/test_my_dataset.py
git commit -m "Add object-biased crop window, h5 block reads and raw-band __getitem__

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QB4DScDwiA2DkVpbDopffV"
```

---

### Task 5: Filter response and p99 normalisation in `__getitem__`

**Files:**
- Modify: `data_loader/my_dataset.py` (replace `__getitem__`)
- Test: `tests/test_my_dataset.py` (append)

**Interfaces:**
- Consumes: `self.sensor_R_matrix [200, N] float32`, `self.scale`, `self.use_filter`, `self.norm` from Task 3; `read_cube_block`, `crop_window` from Task 4.
- Produces: final `__getitem__(idx) -> (img float32 [C, ch, cw], gt float32 [1, ch, cw], name str)` with `C = N` when `use_filter` else `200`, scaled by `1 / scale[name]` when `norm == 'p99'`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_my_dataset.py`:

```python
def test_getitem_filter_equals_einsum_on_raw(synthetic_root):
    root, _, _ = synthetic_root
    ds_f = make(root, split='test', use_filter=True, num_filters=30)
    ds_r = make(root, split='test', use_filter=False)
    img_f, gt_f, name = ds_f[0]
    img_r, gt_r, _ = ds_r[0]
    assert img_f.shape == (30, H, W) and img_f.dtype == np.float32 and img_f.flags['C_CONTIGUOUS']
    expected = np.einsum('bn,bhw->nhw', ds_f.sensor_R_matrix, img_r)   # y_n = sum_b R[b, n] * cube[b]
    np.testing.assert_allclose(img_f, expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(gt_f, gt_r)


def test_getitem_filter_uses_only_bands_below_800nm(synthetic_root):
    root, _, _ = synthetic_root
    ds = make(root, split='test', use_filter=True)
    img_r, _, _ = make(root, split='test', use_filter=False)[0]
    img_f, _, _ = ds[0]
    expected = np.einsum('bn,bhw->nhw', ds.sensor_R_matrix[:133], img_r[:133])
    np.testing.assert_allclose(img_f, expected, rtol=1e-5, atol=1e-6)


def test_getitem_filter_crop_shape(synthetic_root):
    root, _, _ = synthetic_root
    ds = make(root, use_filter=True, num_filters=8, crop_size=16, seed=5)
    img, gt, _ = ds[0]
    assert img.shape == (8, 16, 16) and gt.shape == (1, 16, 16)


def test_getitem_p99_norm_scales_by_p99_over_bands(synthetic_root):
    root, info, _ = synthetic_root
    p99 = info[('test', '7')][2]
    for use_filter in (False, True):
        img_none, _, _ = make(root, split='test', use_filter=use_filter, norm='none')[0]
        img_p99, _, _ = make(root, split='test', use_filter=use_filter, norm='p99')[0]
        np.testing.assert_allclose(img_p99, img_none / np.float32(p99 / N_BANDS), rtol=1e-5, atol=1e-6)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_my_dataset.py -v`
Expected: `test_getitem_filter_equals_einsum_on_raw`, `test_getitem_filter_uses_only_bands_below_800nm`, `test_getitem_filter_crop_shape` fail on the shape assertion (`(200, ...)` instead of `(30, ...)`/`(8, ...)`); `test_getitem_p99_norm_scales_by_p99_over_bands` fails on values. 15 earlier tests pass.

- [ ] **Step 3: Replace `__getitem__` with the full version**

In `data_loader/my_dataset.py`, replace the whole `__getitem__` method with:

```python
    def __getitem__(self, idx):
        name = self.img_name[idx]
        gt = self.load_gt(name)  # [H, W] bool
        h0, w0, ch, cw = self.crop_window(gt)

        blk = self.read_cube_block(name, h0, w0, ch, cw)  # [B, cw, ch] float32
        if self.norm == 'p99':
            # positive per-sample scalar, so scaling before or after the filter is identical
            blk /= np.float32(self.scale[name])

        if self.use_filter:
            # simulated detector channels: y_n = sum_b R[b, n] * cube[b], done in the h5 layout -> [N, cw, ch]
            img = np.tensordot(self.sensor_R_matrix.T, blk, axes=(1, 0))
        else:
            img = blk  # raw bands [B, cw, ch]

        # only the last two axes are swapped: [C, cw, ch] -> [C, ch, cw]; never build [H, W, B] (6 s per crop)
        img = np.ascontiguousarray(img.transpose(0, 2, 1), dtype=np.float32)  # [C, ch, cw]
        gt = gt[h0:h0 + ch, w0:w0 + cw].astype(np.float32)[None]  # [1, ch, cw]
        return img, gt, name
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/ -v`
Expected: `33 passed` (14 in `test_ec_filter.py`, 19 in `test_my_dataset.py`).

- [ ] **Step 5: Commit**

```bash
git add data_loader/my_dataset.py tests/test_my_dataset.py
git commit -m "Compute EC filter channels and p99 normalisation in __getitem__

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QB4DScDwiA2DkVpbDopffV"
```

---

### Task 6: Collate function, argparse glue, smoke test on real data, push

**Files:**
- Modify: `data_loader/my_dataset.py` (append module-level functions and `__main__`)
- Test: `tests/test_my_dataset.py` (append)

**Interfaces:**
- Consumes: `HyperCOD_data` (final) from Task 5.
- Produces: `image_collate_fn(batch) -> (img torch.float32 [B, C, H, W], gt torch.float32 [B, 1, H, W], names list[str])`, `add_dataset_args(parser) -> parser` adding `--data_path --filter_path --use_filter --num_filters --filter_select --filter_voltages --crop_size --obj_crop_prob --norm`, `build_dataset(args, split) -> HyperCOD_data`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_my_dataset.py`:

```python
from data_loader.my_dataset import image_collate_fn, add_dataset_args, build_dataset


def test_collate_stacks_to_bchw(synthetic_root):
    root, _, _ = synthetic_root
    ds = make(root, use_filter=True, num_filters=30, crop_size=16, seed=6)
    loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False, collate_fn=image_collate_fn)
    img, gt, names = next(iter(loader))
    assert isinstance(img, torch.Tensor) and img.shape == (2, 30, 16, 16) and img.dtype == torch.float32
    assert isinstance(gt, torch.Tensor) and gt.shape == (2, 1, 16, 16) and gt.dtype == torch.float32
    assert names == ['3', '10']


def test_add_dataset_args_defaults():
    args = add_dataset_args(argparse.ArgumentParser()).parse_args([])
    assert args.data_path == '/data2/chaoyi/HyperCOD/Raw data'
    assert args.filter_path is None and args.use_filter is False
    assert (args.num_filters, args.filter_select, args.filter_voltages) == (30, 'uniform', None)
    assert (args.crop_size, args.obj_crop_prob, args.norm) == (512, 0.5, 'p99')


def test_build_dataset_from_args(synthetic_root):
    root, _, _ = synthetic_root
    parser = add_dataset_args(argparse.ArgumentParser())
    args = parser.parse_args(['--data_path', str(root), '--use_filter', '--crop_size', '16',
                              '--filter_select', 'manual', '--filter_voltages', '-0.5', '1.0', '--norm', 'none'])
    ds = build_dataset(args, split='train')
    assert ds.use_filter and ds.in_channels == 2 and ds.crop_size == 16 and ds.norm == 'none'
    np.testing.assert_allclose(ds.selected_voltages, [-0.5, 1.0])
    img, gt, name = ds[0]
    assert img.shape == (2, 16, 16)
    ds_test = build_dataset(args, split='test')
    assert ds_test.split == 'test' and ds_test[0][0].shape == (2, H, W)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_my_dataset.py -v`
Expected: `ImportError: cannot import name 'image_collate_fn'`.

- [ ] **Step 3: Implement collate, args, builder and smoke test**

Append to the end of `data_loader/my_dataset.py` (module level, after the class):

```python
def image_collate_fn(batch):
    img, gt, name = list(zip(*batch))
    # the dataset gives [C, H, W] / [1, H, W] numpy arrays; stack to [B, C, H, W] / [B, 1, H, W] float32 tensors
    img = torch.from_numpy(np.stack(img, axis=0)).to(dtype=torch.float32)
    gt = torch.from_numpy(np.stack(gt, axis=0)).to(dtype=torch.float32)
    return img, gt, list(name)


def add_dataset_args(parser):
    parser.add_argument('--data_path', type=str, default='/data2/chaoyi/HyperCOD/Raw data',
                        help='HyperCOD root containing train/ and test/')
    parser.add_argument('--filter_path', type=str, default=None,
                        help='EC sensor response .mat, default <data_path>/EC_filterV3.mat')
    parser.add_argument('--use_filter', action='store_true',
                        help='feed the model the simulated EC detector channels instead of the raw 200 bands')
    parser.add_argument('--num_filters', type=int, default=30,
                        help='number of voltage channels for filter_select uniform / osp')
    parser.add_argument('--filter_select', type=str, default='uniform', choices=['uniform', 'osp', 'all', 'manual'],
                        help='how to choose the bias voltages')
    parser.add_argument('--filter_voltages', type=float, nargs='+', default=None,
                        help='explicit bias voltages in V for filter_select manual')
    parser.add_argument('--crop_size', type=int, default=512,
                        help='training crop size at native resolution, 0 means full frame')
    parser.add_argument('--obj_crop_prob', type=float, default=0.5,
                        help='probability that a training crop is placed to contain the object')
    parser.add_argument('--norm', type=str, default='p99', choices=['none', 'p99'],
                        help='per-sample scaling by intensity p99 / 200 from intensity_p99_summary.csv')
    return parser


def build_dataset(args, split):
    return HyperCOD_data(data_path=args.data_path, split=split, use_filter=args.use_filter,
                         filter_path=args.filter_path, num_filters=args.num_filters,
                         filter_select=args.filter_select, filter_voltages=args.filter_voltages,
                         crop_size=args.crop_size, obj_crop_prob=args.obj_crop_prob, norm=args.norm)


if __name__ == '__main__':
    import time
    parser = add_dataset_args(argparse.ArgumentParser(description='HyperCOD dataset smoke test'))
    args = parser.parse_args()
    dataset = build_dataset(args, split='train')
    print(f"{len(dataset)} train samples, in_channels={dataset.in_channels}")
    t = time.time()
    img, gt, name = dataset[0]
    print(f"sample {name}: img {img.shape} {img.dtype} range [{img.min():.4f}, {img.max():.4f}], "
          f"gt {gt.shape} fg={int(gt.sum())} px, {time.time() - t:.2f}s")
```

- [ ] **Step 4: Run the full test suite**

Run: `python -m pytest tests/ -v`
Expected: `36 passed`.

- [ ] **Step 5: Smoke test on the real data (filter mode and raw mode)**

Run:
```bash
python -m data_loader.my_dataset --use_filter --crop_size 512
python -m data_loader.my_dataset --crop_size 512 --norm none
```
Expected (first command): `Loading sensor response from /data2/chaoyi/HyperCOD/Raw data/EC_filterV3.mat...`, `Sensor response aligned: 200 bands x 30 channels, 133 bands inside the sensor range, voltages [-1.0, ...]`, `279 train samples, in_channels=30`, `sample 1: img (30, 512, 512) float32 range [...], gt (1, 512, 512) fg=<n> px, ~2s`.
Expected (second): `in_channels=200`, `img (200, 512, 512)`, range roughly `[0.0000, 0.0x]` (raw radiance scale), `~2s`.

- [ ] **Step 6: Commit and push the branch**

```bash
git add data_loader/my_dataset.py tests/test_my_dataset.py
git commit -m "Add collate function, dataset argparse glue and smoke test

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QB4DScDwiA2DkVpbDopffV"
git push -u origin worktree-dataloader
```

Expected: `worktree-dataloader -> worktree-dataloader` on `git@github.com:Chaoyi-He1/hsi_camo.git`. Merging into `master` is the user's call.

---

## Self-review against the spec

- Spec §Components `HyperCOD_data.__init__` signature, attributes and methods → Tasks 3–5. `select_filter_channels`, `osp`, alignment → Tasks 1–2. `image_collate_fn`, `add_dataset_args`, `build_dataset`, `__main__` → Task 6. Constants → Task 1 (`ec_filter.py`) and Task 3 (`GT_THRESHOLD`, `HYPERCUBE_KEY`).
- Spec §Decisions: zero fill above 800 nm (Task 1 test), no min–max rescale (Task 1 + Task 3 tests), dead zone (Task 2 tests), p99 = p99/200 (Task 5 test), `(B,W,H)` path with last-two-axes swap (Task 4/5 code + value test), direct h5 hyperslab reads (Task 4), `.mat` as source (no HCODh5 anywhere).
- Spec §Error handling: every listed assert appears in Task 3 (`split`, `norm`, paths, GT per sample, cube shape, `crop_size`), Task 2 (`manual` without voltages, `num_filters` bounds, dead-zone warning) — all covered by tests.
- Spec §Testing items 1–9 → Task 1 (1), Task 4 (2, 4, 5), Task 5 (3, 8), Task 6 (6), Task 2/3 (7), Task 3 (9).
- Type consistency: `crop_window` returns `(h0, w0, ch, cw)` and `read_cube_block(name, h0, w0, ch, cw)` returns `[B, cw, ch]`, consumed in that order in `__getitem__`; `selected_indices` is a `list[int]` everywhere; `sensor_R_matrix` is `[200, N] float32` in Task 3 and used as `R.T [N, 200]` in Task 5 and as `'bn,bhw->nhw'` in tests.
