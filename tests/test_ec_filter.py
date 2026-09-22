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


def test_osp_stops_on_rank_deficient_input_without_duplicates():
    X = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])          # column 0 == column 1, rank 1
    _, idx = osp(X, 3)
    assert idx == [0]                                          # residual exhausted after one pick: stop, no duplicate


def test_select_osp_warns_when_rank_deficient(synthetic_root, capsys):
    _, _, (wl, volt, R) = synthetic_root
    R_const = np.tile(np.linspace(0.0, 1.0, 200)[:, None], (1, len(volt)))   # every voltage column identical
    sel = select_filter_channels(R_const, volt, num_filters=2, mode='osp')
    assert len(sel) == 1 and sel[0] == int(candidate_indices(volt)[0])
    assert "OSP stopped" in capsys.readouterr().out
