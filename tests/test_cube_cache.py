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


def test_cache_float16_path_is_a_view_and_matches_h5(synthetic_root):
    root, info, _ = synthetic_root
    build_cube_cache(str(root), 'train', num_workers=0); build_cube_cache(str(root), 'test', num_workers=0)
    cdir = default_cache_dir(str(root))
    blk = make(root, cache_dir=cdir, out_dtype='float16').read_cube_block('3', h0=5, w0=7, ch=16, cw=12)
    assert blk.shape == (133, 12, 16) and blk.dtype == np.float16
    assert isinstance(blk, np.memmap) and blk.base is not None                 # zero-copy view of the cache file
    np.testing.assert_allclose(blk.astype(np.float32), info[('train', '3')][0][:133, 7:19, 5:21], rtol=1e-3, atol=1e-6)
    # __getitem__ in float16 must agree with the .mat path (one extra fp16 rounding of the p99 scale allowed)
    for kw in (dict(split='test', use_filter=False, norm='p99'),
               dict(split='test', use_filter=True, num_filters=8, norm='p99'),
               dict(split='train', use_filter=False, crop_size=16, obj_crop_prob=1.0, seed=3, norm='p99')):
        a, _, _ = make(root, out_dtype='float16', **kw)[0]
        b, _, _ = make(root, cache_dir=cdir, out_dtype='float16', **kw)[0]
        assert a.dtype == b.dtype == np.float16 and a.shape == b.shape and b.flags['C_CONTIGUOUS']
        np.testing.assert_allclose(a.astype(np.float32), b.astype(np.float32), rtol=4e-3, atol=1e-3)
    # full frame without filter: the returned image is produced without any float32 intermediate
    ds = make(root, cache_dir=cdir, split='test', use_filter=False, norm='p99', out_dtype='float16')
    img, _, _ = ds[0]
    assert img.dtype == np.float16 and img.shape == (133, H, W) and img.flags['C_CONTIGUOUS']


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
