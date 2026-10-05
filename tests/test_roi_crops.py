import os
import json
import random
import numpy as np
import pytest
import torch
from PIL import Image

from data_loader.boxes import expand_box
from data_loader.cube_cache import build_cube_cache, cache_path, default_cache_dir
from data_loader import roi_crops
from data_loader.my_dataset import HyperCOD_data
from data_loader.roi_crops import match_rois, build_crop_cache, frame_windows, pixel_box, inside
from data_loader.roi_crops import (place_on_canvas, canvas_to_roi, rasterise_box, flip_rot, jitter_gt_box, n_fp_items,
                                   HyperCOD_roi, seg_collate_fn)
from tests.conftest import H, W

OBJ1 = [20.0, 10.0, 26.0, 16.0]          # conftest OBJ_SLICE (rows 10:16, cols 20:26), in every frame
OBJ2 = [4.0, 30.0, 12.0, 40.0]           # second object added to train frame '3' (rows 30:40, cols 4:12, 80 px)
FP_ROI = [28.0, 36.0, 40.0, 48.0]        # touches no object
# hand-written ROI exports (main_det_rois.py schema); boxes = pre-expansion boxes + conf, always inside their roi
ROIS = {
    ('raw', 'train'): {'3': {'rois': [[16, 6, 30, 20, 0.9], FP_ROI + [0.3]], 'boxes': [[19, 9, 27, 17, 0.9], [30, 38, 38, 46, 0.3]]}},
    ('ec10', 'train'): {'3': {'rois': [[0, 26, 8, 44, 0.8], FP_ROI + [0.2]], 'boxes': [[2, 30, 8, 40, 0.8], [30, 38, 38, 46, 0.2]]}},
    ('ec24', 'train'): {'3': {'rois': [], 'boxes': []}},
    ('raw', 'val'): {'10': {'rois': [[15, 5, 31, 21, 0.7]], 'boxes': [[19, 9, 27, 17, 0.7]]}},
    ('ec10', 'val'): {'10': {'rois': [], 'boxes': []}},
    ('ec24', 'val'): {'10': {'rois': [], 'boxes': []}},
}
KW = dict(grow=2.0, min_side=8, min_area=10, num_workers=0)           # fixture frames are 48 x 40, objects 36 / 80 px


def _crop_cache(synthetic_root, tmp_path):
    '''Fixture frames + fp16 frame cache + split file (train ['3'], val ['10']) + ROI files -> built crop cache.'''
    root, _, _ = synthetic_root
    gt = np.zeros((H, W), dtype=np.uint8)
    gt[10:16, 20:26] = 255; gt[30:40, 4:12] = 255
    Image.fromarray(np.stack([gt] * 3, axis=-1)).save(root / 'train' / 'GT' / '3.png')
    build_cube_cache(str(root), 'train', num_workers=0)
    split_file = tmp_path / 'val.json'
    split_file.write_text(json.dumps({'seed': 0, 'n_val': 1, 'val_ids': ['10']}))
    roi_files = {}
    for (arm, split), data in ROIS.items():
        p = tmp_path / f'rois_{arm}_{split}.json'
        p.write_text(json.dumps(data))
        roi_files.setdefault(arm, {})[split] = str(p)
    out = tmp_path / 'crops'
    stats = build_crop_cache(str(root), str(out), roi_files, split_file=str(split_file), **KW)
    return root, out, roi_files, split_file, stats


def test_match_rois_max_coverage_and_threshold():
    labels = np.zeros((60, 60), dtype=np.int32)
    labels[0:20, 0:20] = 1                                                 # 400 px
    labels[30:40, 30:40] = 2                                               # 100 px
    rois = [[0, 0, 20, 20],                    # all of 1
            [10, 10, 40, 40],                  # 25 % of 1, 100 % of 2 -> 2
            [19.2, 0, 21, 2],                  # floor/ceil: cols 19..20, rows 0..1 -> 2 px of 1 = 0.5 % < 1 %
            [19.2, 0, 21, 4],                  # 4 px of 1 = 1 % -> 1 (threshold inclusive)
            [45, 45, 60, 60]]                  # nothing
    np.testing.assert_array_equal(match_rois(rois, labels, [1, 2]), [1, 2, -1, 1, -1])
    np.testing.assert_array_equal(match_rois(rois, labels, [2]), [-1, 2, -1, -1, -1])     # 1 is not a kept object
    assert match_rois(np.zeros((0, 4)), labels, [1, 2]).shape == (0,)
    np.testing.assert_array_equal(match_rois(rois, labels, []), [-1] * 5)


def test_frame_windows_contain_rois_and_dedupe_fp():
    gt = np.zeros((48, 40), dtype=bool)
    gt[10:16, 20:26] = True; gt[30:40, 4:12] = True
    fr = {arm: ROIS[(arm, 'train')]['3'] for arm in ('raw', 'ec10', 'ec24')}
    wins = frame_windows(gt, fr, grow=2.0, min_side=8, min_area=10)
    assert [w['kind'] for w in wins] == ['object', 'object', 'fp']
    o1, o2, fp = wins
    assert o1['object'] == 1 and o2['object'] == 2
    # window = union(expand_box(gt, 2, 8), matched ROIs): obj1 [17,7,29,19] u raw [16,6,30,20]; obj2 [0,25,16,45] u ec10 [0,26,8,44]
    assert o1['window'] == [16, 6, 30, 20] and o2['window'] == [0, 25, 16, 45]
    assert [e['roi'] for e in o1['rois']['raw']] == [[16, 6, 30, 20]] and o1['rois']['ec10'] == []
    assert [e['object'] for e in o2['rois']['ec10']] == [2]
    # the fp ROI shared by raw and ec10 is stored once, with each arm's own confidence
    assert fp['window'] == [28, 36, 40, 48] and fp['object'] == -1 and fp['objects'] == []
    assert [e['conf'] for e in fp['rois']['raw']] == [0.3] and [e['conf'] for e in fp['rois']['ec10']] == [0.2] and fp['rois']['ec24'] == []
    assert {o['id']: o['area'] for o in o1['objects'] + o2['objects']} == {1: 36, 2: 80}


def test_frame_windows_assert_names_the_frame():
    # spec §9: an assert in frame_windows must say which frame it is about
    gt = np.zeros((48, 40), dtype=bool)
    gt[10:16, 20:26] = True
    degenerate = {'raw': {'rois': [[30, 30, 30, 36, 0.5]], 'boxes': [[30, 30, 30, 36, 0.5]]}}   # zero-width fp ROI
    with pytest.raises(AssertionError, match='frame 42'):
        frame_windows(gt, degenerate, grow=2.0, min_side=8, min_area=10, name='42')
    skew = {'raw': {'rois': [[30, 30, 36, 36, 0.5]], 'boxes': []}}                               # 1 roi, 0 boxes
    with pytest.raises(AssertionError, match='frame 42'):
        frame_windows(gt, skew, grow=2.0, min_side=8, min_area=10, name='42')


def test_build_crop_cache_index_and_files(synthetic_root, tmp_path):
    root, out, roi_files, split_file, stats = _crop_cache(synthetic_root, tmp_path)
    with open(out / 'index.json') as f:
        index = json.load(f)
    assert index['band_range'] == [400.0, 800.0] and index['frame_hw'] == [H, W] and index['arms'] == ['raw', 'ec10', 'ec24']
    wins = index['windows']
    assert [(w['frame'], w['split'], w['kind']) for w in wins] == [('3', 'train', 'object'), ('3', 'train', 'object'),
                                                                     ('3', 'train', 'fp'), ('10', 'val', 'object')]
    assert stats['n_windows'] == 4 and stats['n_object'] == 3 and stats['n_fp'] == 1
    assert stats['bytes'] == sum(os.path.getsize(out / w[k]) for w in wins for k in ('file', 'gt_file'))
    ds = HyperCOD_data(str(root), split='train', use_filter=False, norm='p99', crop_size=0, filter_norm='none',
                       cache_dir=default_cache_dir(str(root)), out_dtype='float16')
    for w in wins:
        x1, y1, x2, y2 = w['window']
        frame = np.load(cache_path(default_cache_dir(str(root)), 'train', w['frame']))       # [133, H, W] fp16
        cube, gt = np.load(out / w['file']), np.load(out / w['gt_file'])
        assert cube.dtype == np.float16 and cube.shape == (133, y2 - y1, x2 - x1)
        np.testing.assert_array_equal(cube, frame[:, y1:y2, x1:x2])                          # un-scaled, row-major y, x
        assert gt.dtype == bool and np.array_equal(gt, ds.load_gt(w['frame'])[y1:y2, x1:x2])
        assert w['scale'] == ds.scale[w['frame']]
        for arm in index['arms']:                                                             # spec §9
            assert all(inside(pixel_box(e['roi'], H, W), w['window']) for e in w['rois'][arm])
        if w['kind'] == 'object':
            gt_box = next(o['box'] for o in w['objects'] if o['id'] == w['object'])
            assert inside(pixel_box(expand_box(gt_box, 2.0, 8, H, W), H, W), w['window'])
    assert not (out / 'index.json.tmp').exists()


def test_build_crop_cache_raises_on_missing_frame(synthetic_root, tmp_path):
    root, out, roi_files, split_file, _ = _crop_cache(synthetic_root, tmp_path)
    bad = tmp_path / 'rois_raw_val_other.json'
    bad.write_text(json.dumps({'99': {'rois': [], 'boxes': []}}))                 # an ROI file of another split
    roi_files['raw']['val'] = str(bad)
    with pytest.raises(AssertionError, match='lacks frames'):
        build_crop_cache(str(root), str(tmp_path / 'crops2'), roi_files, split_file=str(split_file), **KW)


def test_rebuild_into_existing_out_dir_never_leaves_a_stale_index(synthetic_root, tmp_path, monkeypatch):
    root, out, roi_files, split_file, stats = _crop_cache(synthetic_root, tmp_path)       # first, complete build
    old = (out / 'index.json').read_text()
    (out / 'index.json.tmp').write_text('{"half-written')                                 # debris of an earlier crash
    reads = []
    real_read = roi_crops.read_npy_direct

    def crash_on_second_frame(path):
        reads.append(path)
        if len(reads) == 2:
            raise RuntimeError('simulated crash while reading the second frame')
        return real_read(path)

    monkeypatch.setattr(roi_crops, 'read_npy_direct', crash_on_second_frame)              # num_workers=0: same process
    with pytest.raises(RuntimeError, match='simulated crash'):
        build_crop_cache(str(root), str(out), roi_files, split_file=str(split_file), **{**KW, 'grow': 3.0})
    assert len(reads) == 2
    assert not (out / 'index.json').exists() and not (out / 'index.json.tmp').exists()   # nothing claims a complete cache
    monkeypatch.setattr(roi_crops, 'read_npy_direct', real_read)                          # relaunch: the rebuild succeeds
    stats2 = build_crop_cache(str(root), str(out), roi_files, split_file=str(split_file), **{**KW, 'grow': 3.0})
    new = (out / 'index.json').read_text()
    assert json.loads(new)['grow'] == 3.0 and new != old and stats2['n_windows'] == 4
    assert not (out / 'index.json.tmp').exists()


def test_build_crop_cache_with_workers_matches_serial(synthetic_root, tmp_path):
    root, out, roi_files, split_file, stats = _crop_cache(synthetic_root, tmp_path)
    out2 = tmp_path / 'crops_mp'
    stats2 = build_crop_cache(str(root), str(out2), roi_files, split_file=str(split_file), **{**KW, 'num_workers': 2})
    assert stats2 == stats
    a = json.loads((out / 'index.json').read_text())
    b = json.loads((out2 / 'index.json').read_text())
    assert a['windows'] == b['windows']                                         # same windows, same order
    for w in a['windows']:
        np.testing.assert_array_equal(np.load(out / w['file']), np.load(out2 / w['file']))
        np.testing.assert_array_equal(np.load(out / w['gt_file']), np.load(out2 / w['gt_file']))


DS_KW = dict(roi_margin=1.5, roi_min=0, canvas=32)                       # oracle ROIs of the 6 x 6 object are 10 x 10


def test_place_on_canvas_exact_and_resized_round_trip():
    rng = np.random.default_rng(0)
    crop = rng.random((3, 20, 30)).astype(np.float16)
    out, valid, (oy, ox), s = place_on_canvas(crop, canvas=64)
    assert s == 1.0 and (oy, ox) == (22, 17) and out.dtype == np.float16 and valid.sum() == 600
    np.testing.assert_array_equal(out[:, oy:oy + 20, ox:ox + 30], crop)
    assert not out[:, ~valid].any()
    np.testing.assert_array_equal(canvas_to_roi(out[1].astype(np.float32), (oy, ox), s, (20, 30)), crop[1].astype(np.float32))
    np.testing.assert_array_equal(canvas_to_roi(torch.from_numpy(out[1].astype(np.float32)), (oy, ox), s, (20, 30)),
                                  crop[1].astype(np.float32))
    # larger than the canvas: aspect kept, downscaled to fit, round trip within interpolation error on a smooth map
    yy, xx = np.mgrid[0:100, 0:60]
    smooth = (0.5 + 0.5 * np.sin(yy / 15.0) * np.cos(xx / 11.0)).astype(np.float32)[None]   # [1, 100, 60]
    out, valid, (oy, ox), s = place_on_canvas(smooth, canvas=64)
    assert abs(s - 0.64) < 1e-9 and valid.sum() == 64 * 38 and (oy, ox) == (0, 13)
    back = canvas_to_roi(out[0], (oy, ox), s, (100, 60))
    assert back.shape == (100, 60) and np.abs(back - smooth[0])[4:-4, 4:-4].max() < 0.05
    # scale augmentation never pushes the crop off the canvas
    out, valid, _, s = place_on_canvas(crop, canvas=64, scale=3.0)
    assert s == pytest.approx(64 / 30) and valid.sum() == 43 * 64


def test_flip_rot_keeps_box_and_arrays_consistent():
    c = 16
    box = [3.5, 2.0, 9.0, 6.2]
    for hf in (False, True):
        for vf in (False, True):
            for k in range(4):
                (m,), b = flip_rot([rasterise_box(box, c)], box, hf, vf, k, c)
                np.testing.assert_array_equal(m, rasterise_box(b, c))


def test_jittered_gt_box_never_cuts_the_object():
    rng = random.Random(0)
    gt = [100.0, 50.0, 160.0, 90.0]
    for _ in range(1000):
        b = jitter_gt_box(gt, 0.15, rng, 1680, 1240)
        assert b[0] <= gt[0] and b[1] <= gt[1] and b[2] >= gt[2] and b[3] >= gt[3]
        assert b[2] - b[0] <= 60 * 1.3 + 1e-3 and b[3] - b[1] <= 40 * 1.3 + 1e-3
        # the expanded ROI fits the cached object window (grow 2.0, min_side 512)
        assert inside(pixel_box(expand_box(b, 1.5, 256, 1680, 1240), 1680, 1240),
                      pixel_box(expand_box(gt, 2.0, 512, 1680, 1240), 1680, 1240))
    assert n_fp_items(289, 0.1) == 32 and n_fp_items(2, 0.0) == 0


def test_validation_items_are_deterministic_and_exact(synthetic_root, tmp_path):
    root, out, _, _, _ = _crop_cache(synthetic_root, tmp_path)
    ds = HyperCOD_roi(str(out), 'val', 'raw', train=False, **DS_KW)
    assert len(ds) == 2                                                    # oracle ROI + the matched raw val ROI
    ref = HyperCOD_data(str(root), split='train', ids=['10'], use_filter=False, norm='p99', crop_size=0,
                        filter_norm='none', cache_dir=default_cache_dir(str(root)), out_dtype='float16')
    frame, gt, _ = ref[0]                                                  # [133, H, W] fp16 p99-scaled, [1, H, W]
    for i, source in enumerate(['gt', 'det']):
        img, mask, box_map, valid, meta = ds[i]
        assert meta['source'] == source and meta['frame'] == '10' and meta['s'] == 1.0
        x1, y1, x2, y2 = [int(v) for v in meta['roi']]
        if source == 'gt':
            assert (x1, y1, x2, y2) == pixel_box(expand_box(OBJ1, 1.5, 0, H, W), H, W) == (18, 8, 28, 18)
            assert meta['box'] == OBJ1
        else:
            assert (x1, y1, x2, y2) == (15, 5, 31, 21) and meta['box'] == [19, 9, 27, 17]
        oy, ox = meta['offset']
        h, w = meta['roi_hw']
        assert (h, w) == (y2 - y1, x2 - x1)
        assert img.dtype == np.float16 and img.shape == (133, 32, 32) and mask.shape == box_map.shape == valid.shape == (1, 32, 32)
        np.testing.assert_array_equal(img[:, oy:oy + h, ox:ox + w], frame[:, y1:y2, x1:x2])     # same fp16 p99 values
        np.testing.assert_array_equal(mask[0, oy:oy + h, ox:ox + w], gt[0, y1:y2, x1:x2])
        assert meta['obj_area'] == 36 and meta['area'] == 36 and mask.sum() == 36
        np.testing.assert_array_equal(box_map[0], rasterise_box(meta['box_canvas'], 32))
        bx1, by1, bx2, by2 = meta['box_canvas']
        assert (bx1, by1) == (ox + meta['box'][0] - x1, oy + meta['box'][1] - y1)
        assert valid.sum() == h * w and not img[:, valid[0] == 0].any()
        img2, mask2, _, _, meta2 = ds[i]
        assert np.array_equal(img, img2) and np.array_equal(mask, mask2) and meta == meta2


def test_training_items_mix_jitter_and_targets(synthetic_root, tmp_path):
    root, out, _, _, _ = _crop_cache(synthetic_root, tmp_path)
    ref = HyperCOD_data(str(root), split='train', ids=['3'], use_filter=False, norm='p99', crop_size=0,
                        filter_norm='none', cache_dir=default_cache_dir(str(root)), out_dtype='float16')
    full_gt = ref.load_gt('3')
    # object items only, no augmentation that resizes: check targets and the 5/9 split
    ds = HyperCOD_roi(str(out), 'train', 'raw', scale_aug=(1.0, 1.0), gain_aug=0.0, seed=0, **DS_KW)
    assert len(ds) == 2 and ds.n_fp == 0                                   # 2 objects, round(2 x 0.1 / 0.9) = 0 fp
    counts = {1: {'gt': 0, 'det': 0}, 2: {'gt': 0, 'det': 0}}
    for _ in range(300):
        for i in range(2):
            img, mask, box_map, valid, meta = ds[i]
            counts[meta['object']][meta['source']] += 1
            x1, y1, x2, y2 = [int(v) for v in meta['roi']]
            assert meta['obj_area'] == int(full_gt[y1:y2, x1:x2].sum())
            assert mask.sum() == meta['obj_area']                          # s = 1: the GT crop, only rotated / flipped
            np.testing.assert_array_equal(box_map[0], rasterise_box(meta['box_canvas'], 32))
            assert not mask[valid == 0].any() and not box_map[valid == 0].any()
            if meta['source'] == 'gt':                                     # jittered GT: object never cut
                assert meta['obj_area'] == meta['area']
                ob = OBJ1 if meta['object'] == 1 else OBJ2
                assert inside(pixel_box(ob, H, W), pixel_box(meta['box'], H, W))
            else:                                                          # matched raw ROI of object 1
                assert meta['roi'] == [16, 6, 30, 20] and meta['object'] == 1
    assert counts[2]['det'] == 0 and counts[2]['gt'] == 300               # no matched raw ROI -> GT fallback
    assert abs(counts[1]['gt'] / 300 - 5 / 9) < 0.08


def test_false_positive_items_and_epoch_size(synthetic_root, tmp_path):
    root, out, _, _, _ = _crop_cache(synthetic_root, tmp_path)
    ds = HyperCOD_roi(str(out), 'train', 'ec10', box_mix=(0.25, 0.25, 0.5), seed=1, **DS_KW)
    assert ds.n_fp == 2 and len(ds) == 4 and len(ds.fp_rois) == 1
    for _ in range(20):
        img, mask, box_map, valid, meta = ds[3]
        assert meta['source'] == 'fp' and meta['object'] == -1 and meta['roi'] == FP_ROI
        assert not mask.any() and box_map.any() and img.dtype == np.float16
    assert HyperCOD_roi(str(out), 'train', 'ec24', box_mix=(0.25, 0.25, 0.5), **DS_KW).n_fp == 0   # ec24 has no fp ROI
    assert len(HyperCOD_roi(str(out), 'val', 'ec10', train=False, **DS_KW)) == 1                  # oracle only


def test_rgb_arm_uses_raw_rois(synthetic_root, tmp_path):
    root, out, _, _, _ = _crop_cache(synthetic_root, tmp_path)
    a = HyperCOD_roi(str(out), 'val', 'rgb', train=False, **DS_KW)
    b = HyperCOD_roi(str(out), 'val', 'raw', train=False, **DS_KW)
    assert len(a) == len(b) == 2 and all(np.array_equal(x, y) for x, y in zip(a[1][:4], b[1][:4]))
    with pytest.raises(AssertionError, match='grow'):
        HyperCOD_roi(str(out), 'val', 'raw', train=False, roi_margin=2.0, roi_min=0, canvas=32)


def test_missing_window_raises_with_the_frame(synthetic_root, tmp_path):
    root, out, _, _, _ = _crop_cache(synthetic_root, tmp_path)
    ds = HyperCOD_roi(str(out), 'val', 'raw', train=False, **DS_KW)
    w = ds.items[0][0]
    os.remove(out / w['file'])
    with pytest.raises(AssertionError, match="frame 10"):                   # spec §9: never skipped
        ds[0]


def test_seg_collate_fn(synthetic_root, tmp_path):
    root, out, _, _, _ = _crop_cache(synthetic_root, tmp_path)
    ds = HyperCOD_roi(str(out), 'train', 'raw', seed=0, **DS_KW)
    loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=True, num_workers=0, collate_fn=seg_collate_fn)
    batch = next(iter(loader))
    assert batch['img'].dtype == torch.float16 and batch['img'].shape == (2, 133, 32, 32)
    for k in ('mask', 'box_map', 'valid'):
        assert batch[k].dtype == torch.float32 and batch[k].shape == (2, 1, 32, 32)
    assert batch['box_xyxy'].shape == (2, 4) and batch['box_xyxy'].dtype == torch.float32 and len(batch['meta']) == 2
    for i, m in enumerate(batch['meta']):
        assert torch.equal(batch['box_xyxy'][i], torch.tensor(m['box_canvas']))


def _crop_cache_gt(synthetic_root, tmp_path, gt, min_side, train_gt=None, train_rois=None):
    '''
    Crop cache whose val frame '10' has the given GT and no detector ROI in any arm; train frame '3' as in the fixture
    unless train_gt / train_rois (the raw arm's entry of frame '3', {"rois": [...], "boxes": [...]}) are given.
    '''
    root, _, _ = synthetic_root
    Image.fromarray(np.stack([gt.astype(np.uint8) * 255] * 3, axis=-1)).save(root / 'train' / 'GT' / '10.png')
    if train_gt is not None:
        Image.fromarray(np.stack([train_gt.astype(np.uint8) * 255] * 3, axis=-1)).save(root / 'train' / 'GT' / '3.png')
    build_cube_cache(str(root), 'train', num_workers=0)
    split_file = tmp_path / 'val.json'
    split_file.write_text(json.dumps({'seed': 0, 'n_val': 1, 'val_ids': ['10']}))
    roi_files = {}
    for arm in ('raw', 'ec10', 'ec24'):
        for split, frame in (('train', '3'), ('val', '10')):
            p = tmp_path / f'rois_{arm}_{split}.json'
            entry = train_rois if (train_rois is not None and arm == 'raw' and split == 'train') else {'rois': [], 'boxes': []}
            p.write_text(json.dumps({frame: entry}))
            roi_files.setdefault(arm, {})[split] = str(p)
    out = tmp_path / 'crops_gt'
    build_crop_cache(str(root), str(out), roi_files, split_file=str(split_file), grow=2.0, min_side=min_side, min_area=10,
                     num_workers=0)
    return out


def test_roi_with_two_objects_and_a_frame_corner_roi(synthetic_root, tmp_path):
    gt = np.zeros((H, W), dtype=bool)
    gt[10:16, 20:26] = True          # A, 36 px
    gt[18:24, 22:30] = True          # B, 48 px: inside A's 24 px oracle ROI, and A inside B's
    gt[42:48, 0:6] = True            # C, 36 px, in the bottom-left corner of the frame
    out = _crop_cache_gt(synthetic_root, tmp_path, gt, min_side=24)
    ds = HyperCOD_roi(str(out), 'val', 'raw', train=False, roi_margin=1.5, roi_min=24, canvas=32)
    items = [ds[i] for i in range(len(ds))]
    assert len(items) == 3 and all(m['source'] == 'gt' for *_, m in items)
    by_area = {}
    for img, mask, box_map, valid, meta in items:
        x1, y1, x2, y2 = [int(v) for v in meta['roi']]
        assert 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H
        # target = every GT pixel inside the ROI (the union), size bucket = the object's own area
        assert mask.sum() == meta['obj_area'] == int(gt[y1:y2, x1:x2].sum())
        by_area.setdefault(meta['area'], []).append(meta['obj_area'])
        assert not mask[valid == 0].any() and not box_map[valid == 0].any() and not img[:, valid[0] == 0].any()
        np.testing.assert_array_equal(box_map[0], rasterise_box(meta['box_canvas'], 32))
    assert sorted(by_area) == [36, 48] and sorted(by_area[36]) == [36, 84] and by_area[48] == [84]
    corner = next(m for *_, m in items if m['obj_area'] == 36)
    # clipped at the frame edge (expand_box does not shift), so smaller than roi_min and zero-padded on the canvas
    assert corner['roi'][0] == 0 and corner['roi'][3] == H and corner['roi_hw'] == (15, 15) and corner['s'] == 1.0


def test_training_items_through_dataloader_workers(synthetic_root, tmp_path):
    '''
    Review Focus (with Task 3's test_build_crop_cache_with_workers_matches_serial): training items through DataLoader
    workers are always valid, and the seeded per-worker augmentation is not frozen across workers / epochs.
    '''
    root, out, _, _, _ = _crop_cache(synthetic_root, tmp_path)
    ds = HyperCOD_roi(str(out), 'train', 'raw', seed=0, roi_margin=1.5, roi_min=0, canvas=32)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, collate_fn=seg_collate_fn)
    seen = []
    for _ in range(10):
        for batch in loader:
            m = batch['meta'][0]
            assert batch['img'].shape == (1, 133, 32, 32) and not batch['mask'][batch['valid'] == 0].any()
            seen.append(tuple(m['aug']) + (m['source'], m['object']))
    assert len(seen) == 10 * len(ds) and len(set(seen)) > 2


def test_false_positive_item_target_is_empty_even_when_it_overlaps_an_object(synthetic_root, tmp_path):
    '''
    Controller note (Task 3): an fp window may list GT objects below min_cover (here 1 of 120 px = 0.8 % < 1 %).
    The window's GT crop then has a foreground pixel, but spec §3 says a false-positive box has an EMPTY target.
    '''
    gt = np.zeros((H, W), dtype=bool)
    gt[10:22, 20:30] = True                                                # 12 x 10 = 120 px
    fp = [29.0, 21.0, 38.0, 30.0]                                          # covers only the pixel (row 21, col 29)
    train_rois = {'rois': [fp + [0.4]], 'boxes': [[31, 23, 36, 28, 0.4]]}
    out = _crop_cache_gt(synthetic_root, tmp_path, gt, min_side=8, train_gt=gt, train_rois=train_rois)
    ds = HyperCOD_roi(str(out), 'train', 'raw', box_mix=(0.25, 0.25, 0.5), seed=0, roi_margin=1.5, roi_min=0, canvas=32)
    assert len(ds.obj_windows) == 1 and ds.n_fp == 1 and len(ds) == 2 and len(ds.fp_rois) == 1
    w, e = ds.fp_rois[0]
    assert [o['id'] for o in w['objects']] == [1]                          # the overlapped object is listed ...
    assert np.load(out / w['gt_file']).sum() == 1                          # ... and its pixel is in the window's GT crop
    for _ in range(20):
        img, mask, box_map, valid, meta = ds[1]
        assert meta['source'] == 'fp' and meta['object'] == -1 and meta['roi'] == fp
        assert not mask.any() and meta['obj_area'] == 0 and meta['area'] == 0     # ... but the target stays empty
        assert box_map.any() and not box_map[valid == 0].any()
