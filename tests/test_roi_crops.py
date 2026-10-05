import os
import json
import numpy as np
import pytest
from PIL import Image

from data_loader.boxes import expand_box
from data_loader.cube_cache import build_cube_cache, cache_path, default_cache_dir
from data_loader.my_dataset import HyperCOD_data
from data_loader.roi_crops import match_rois, build_crop_cache, frame_windows, pixel_box, inside
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
