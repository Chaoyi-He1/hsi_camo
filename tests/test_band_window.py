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
        band_indices((1200.0, 1400.0))


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
