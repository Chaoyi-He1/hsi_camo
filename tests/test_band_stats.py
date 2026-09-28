import numpy as np
import pytest

from data_loader.band_stats import compute_band_stats, load_band_stats
from tests.conftest import H, W


def test_compute_band_stats_matches_direct(synthetic_root, tmp_path):
    root, info, _ = synthetic_root
    out = tmp_path / 'stats.npz'
    mean, cov = compute_band_stats(str(root), str(out), crop_size=0, num_workers=0, band_range=(400., 800.))
    # direct computation: all pixels of both training cubes, p99-normalised (windowed cube / (windowed p99 / 133))
    xs = []
    for name in ['3', '10']:
        cube, _, _ = info[('train', name)]                                   # [B, W, H]
        p99_133 = np.percentile(cube[:133].sum(axis=0), 99)
        xs.append(cube[:133].reshape(133, -1).astype(np.float64) / (p99_133 / 133))
    x = np.concatenate(xs, axis=1)                                           # [133, 2*H*W]
    np.testing.assert_allclose(mean, x.mean(axis=1), rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(cov, np.cov(x, bias=True), rtol=1e-5, atol=1e-8)
    m2, c2 = load_band_stats(str(out))
    np.testing.assert_array_equal(m2, mean)
    np.testing.assert_array_equal(c2, cov)
    st = np.load(out)
    assert int(st['n_pixels']) == 2 * H * W and int(st['n_samples']) == 2 and str(st['norm']) == 'p99'
    assert st['wavelens'].shape == (133,)


def test_compute_band_stats_with_crops_and_workers(synthetic_root, tmp_path):
    root, _, _ = synthetic_root
    mean, cov = compute_band_stats(str(root), str(tmp_path / 's.npz'), crop_size=16, num_workers=2, seed=0)
    assert mean.shape == (133,) and cov.shape == (133, 133)
    assert np.allclose(cov, cov.T) and (np.diag(cov) > 0).all()
    assert int(np.load(tmp_path / 's.npz')['n_pixels']) == 2 * 16 * 16


def test_load_band_stats_validates_shapes(tmp_path):
    np.savez(tmp_path / 'bad.npz', mean=np.zeros(3), cov=np.zeros((3, 3)))
    with pytest.raises(AssertionError, match="mean"):
        load_band_stats(str(tmp_path / 'bad.npz'))
