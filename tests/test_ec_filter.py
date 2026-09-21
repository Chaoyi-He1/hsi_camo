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
