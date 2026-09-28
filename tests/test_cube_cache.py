import os
import numpy as np
import pytest
import torch

from data_loader.cube_cache import build_cube_cache, cache_path, default_cache_dir, read_npy_direct
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
    # full frame without filter: the returned image is produced without any float32 intermediate; the full frame
    # is read() into memory (not a memmap) and scaled in place, so the returned image owns its data
    ds = make(root, cache_dir=cdir, split='test', use_filter=False, norm='p99', out_dtype='float16')
    full = ds.read_cube_block(ds.img_name[0], 0, 0, H, W)
    assert not isinstance(full, np.memmap) and full.flags.writeable and full.shape == (133, W, H)
    img, _, _ = ds[0]
    assert img.dtype == np.float16 and img.shape == (133, H, W) and img.flags['C_CONTIGUOUS'] and not isinstance(img, np.memmap)


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


def test_read_npy_direct_matches_np_load(tmp_path):
    rng = np.random.default_rng(0)
    arrays = [rng.standard_normal((133, 48, 40)).astype(np.float16),      # 510 KB + header: not a multiple of 4096
              np.arange(3, dtype=np.float32),                                # smaller than one block
              rng.integers(0, 255, (5, 4096), dtype=np.uint8)]                 # data start inside the first block
    for i, a in enumerate(arrays):
        p = tmp_path / f'a{i}.npy'; np.save(p, a)
        b = read_npy_direct(str(p))
        assert b.shape == a.shape and b.dtype == a.dtype and b.flags['C_CONTIGUOUS'] and b.flags.writeable
        np.testing.assert_array_equal(b, np.load(p))
        b[...] = 0                                                             # owns its memory: the file is untouched
        np.testing.assert_array_equal(np.load(p), a)


@pytest.mark.skipif(not os.path.isdir('/dev/shm'), reason='needs a tmpfs mount')
def test_read_npy_direct_falls_back_without_o_direct():
    import uuid
    p = f'/dev/shm/hsi_camo_test_{uuid.uuid4().hex}.npy'
    a = np.arange(1000, dtype=np.float16).reshape(10, 100)
    try:
        np.save(p, a)
        np.testing.assert_array_equal(read_npy_direct(p), a)                   # tmpfs rejects O_DIRECT -> np.load
    finally:
        os.remove(p)
