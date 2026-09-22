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
