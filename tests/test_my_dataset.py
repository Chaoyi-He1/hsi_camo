import argparse
import pickle
import random
import numpy as np
import pytest
import torch

from data_loader.ec_filter import N_BANDS, WAVELENS_200, FILTER_DEAD_ZONE_V
from data_loader.my_dataset import HyperCOD_data, image_collate_fn, add_dataset_args, build_dataset
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


def test_dataset_is_picklable_when_unseeded(synthetic_root):
    root, _, _ = synthetic_root
    ds = make(root, seed=None)
    assert ds.rng is random                                       # module resolved at use, never stored
    ds2 = pickle.loads(pickle.dumps(ds))                          # spawn/forkserver workers need this
    assert ds2.img_name == ds.img_name and ds2.rng is random


def test_seeded_dataset_is_deterministic_in_process(synthetic_root):
    root, _, _ = synthetic_root
    a = [make(root, use_filter=False, crop_size=8, seed=7)[i][1] for i in range(2)]
    b = [make(root, use_filter=False, crop_size=8, seed=7)[i][1] for i in range(2)]
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def test_seeded_dataset_varies_crops_across_epochs_with_workers(synthetic_root):
    root, _, _ = synthetic_root
    ds = make(root, use_filter=False, crop_size=8, obj_crop_prob=0.0, seed=0)
    loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False, num_workers=2,
                                         persistent_workers=False, collate_fn=image_collate_fn)
    epochs = [next(iter(loader))[0].clone() for _ in range(3)]    # img tensors [2, 200, 8, 8]
    assert not (torch.equal(epochs[0], epochs[1]) and torch.equal(epochs[1], epochs[2]))


def test_filter_gain(synthetic_root):
    root, _, _ = synthetic_root
    ds = make(root, num_filters=12)
    assert ds.filter_gain.shape == (12,)
    np.testing.assert_allclose(ds.filter_gain, np.abs(ds.sensor_R_matrix).sum(axis=0))
    assert (ds.filter_gain > 0).all()


def test_read_cube_block_asserts_on_clipped_window(synthetic_root):
    root, _, _ = synthetic_root
    ds = make(root)
    with pytest.raises(AssertionError, match="window"):
        ds.read_cube_block('3', h0=H - 4, w0=0, ch=16, cw=12)


def test_init_asserts_non_numeric_mat_name(synthetic_root):
    root, _, _ = synthetic_root
    (root / 'train' / 'hyperspectral' / 'notes.mat').write_bytes(b'')
    with pytest.raises(AssertionError, match="non-numeric"):
        make(root)
