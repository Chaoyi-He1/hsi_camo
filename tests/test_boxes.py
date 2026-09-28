import json
import numpy as np
import pytest
import torch

from data_loader.boxes import boxes_from_mask, boxes_to_yolo, yolo_to_xyxy, flip_boxes, expand_box, det_collate_fn
from data_loader.det_splits import make_det_splits, load_det_ids
from data_loader.my_dataset import HyperCOD_data
from tests.conftest import H, W, OBJ_SLICE


def test_boxes_from_mask_single_and_specks():
    m = np.zeros((H, W), bool); m[OBJ_SLICE] = True; m[40, 5] = True      # 6x6 object + 1-px speck
    b = boxes_from_mask(m)
    np.testing.assert_array_equal(b, [[20, 10, 26, 16]]); assert b.dtype == np.float32
    b2, labels, ids = boxes_from_mask(m, return_labels=True)
    assert labels.shape == (H, W) and ids == [1] and (labels == 1).sum() == 36
    assert boxes_from_mask(np.zeros((H, W), bool)).shape == (0, 4)


def test_boxes_from_mask_two_objects_sorted_by_position():
    m = np.zeros((H, W), bool); m[2:5, 30:34] = True; m[20:30, 2:8] = True
    b = boxes_from_mask(m, min_area=5)
    np.testing.assert_array_equal(b, [[30, 2, 34, 5], [2, 20, 8, 30]])


def test_yolo_conversion_and_flips():
    b = np.array([[20, 10, 26, 16]], np.float32)
    y = boxes_to_yolo(b, H, W)
    np.testing.assert_allclose(y, [[23 / W, 13 / H, 6 / W, 6 / H]])
    np.testing.assert_allclose(yolo_to_xyxy(y, H, W), b)
    np.testing.assert_array_equal(flip_boxes(b, H, W, horizontal=True, vertical=False), [[W - 26, 10, W - 20, 16]])
    np.testing.assert_array_equal(flip_boxes(b, H, W, horizontal=False, vertical=True), [[20, H - 16, 26, H - 10]])
    np.testing.assert_array_equal(flip_boxes(flip_boxes(b, H, W, True, True), H, W, True, True), b)


def test_expand_box_margin_min_size_and_clipping():
    np.testing.assert_array_equal(expand_box(np.array([20, 10, 26, 16.]), margin=1.5, min_size=0, H=H, W=W), [18.5, 8.5, 27.5, 17.5])
    e = expand_box(np.array([20, 10, 26, 16.]), margin=1.0, min_size=20, H=H, W=W)
    assert e[2] - e[0] == 20 and e[3] - e[1] == 20
    e = expand_box(np.array([0, 0, 10, 10.]), margin=3.0, min_size=0, H=H, W=W)
    assert e[0] == 0 and e[1] == 0 and e[2] <= W and e[3] <= H


def test_det_collate_fn_builds_yolo_batch(synthetic_root):
    root, _, _ = synthetic_root
    ds = HyperCOD_data(str(root), split='train', use_filter=False, norm='p99', crop_size=0, out_dtype='float16', filter_norm='none')
    batch = det_collate_fn([ds[0], ds[1]])
    assert batch['img'].shape == (2, 133, H, W) and batch['img'].dtype == torch.float16
    assert batch['batch_idx'].tolist() == [0.0, 1.0] and batch['cls'].shape == (2, 1) and batch['bboxes'].shape == (2, 4)
    np.testing.assert_allclose(batch['bboxes'][0].numpy(), [23 / W, 13 / H, 6 / W, 6 / H], rtol=1e-6)
    assert batch['names'] == ['3', '10'] and len(batch['masks']) == 2 and batch['masks'][0].dtype == bool
    np.testing.assert_array_equal(batch['boxes_xyxy'][1], [[20, 10, 26, 16]])


def test_make_and_load_det_splits(synthetic_root, tmp_path):
    root, _, _ = synthetic_root
    p = tmp_path / 'det_val_ids.json'
    train_ids, val_ids = make_det_splits(str(root), n_val=1, seed=0, path=str(p))
    assert sorted(train_ids + val_ids) == ['10', '3'] and len(val_ids) == 1
    assert json.load(open(p))['val_ids'] == val_ids
    assert load_det_ids(str(root), path=str(p)) == (train_ids, val_ids)
    assert make_det_splits(str(root), n_val=1, seed=0, path=str(p)) == (train_ids, val_ids)   # idempotent
