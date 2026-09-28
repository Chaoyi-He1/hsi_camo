import math
import numpy as np
import pytest

from train_eval.box_metrics import (box_iou_matrix, match_greedy, mask_coverage, contains, tightness, center_offset,
                                    average_precision, filter_to_operating_point, BoxMetrics, select_score)

OP_KEYS = ['coverage_raw', 'coverage_recall99_raw', 'coverage_roi', 'coverage_recall99_roi',
           'contain_rate', 'tightness', 'center_offset', 'recall50', 'dets_per_image']

H, W = 48, 40


def _mask():
    m = np.zeros((H, W), bool); m[10:20, 20:30] = True                      # 10x10 object, box [20, 10, 30, 20]
    return m


def _det(x1, y1, x2, y2, conf=0.9):
    return np.array([[x1, y1, x2, y2, conf, 0]], np.float32)


def test_iou_matrix_and_greedy_match():
    g = np.array([[20, 10, 30, 20]], np.float32)
    d = np.array([[20, 10, 30, 20, 0.5, 0], [0, 0, 5, 5, 0.9, 0]], np.float32)
    iou = box_iou_matrix(g, d[:, :4]); np.testing.assert_allclose(iou, [[1.0, 0.0]])
    idx, best = match_greedy(g, d); assert idx.tolist() == [0] and best[0] == 1.0
    idx, best = match_greedy(g, np.zeros((0, 6), np.float32)); assert idx.tolist() == [-1] and best[0] == 0.0


def test_pointwise_metrics():
    m, gt = _mask(), np.array([20, 10, 30, 20], np.float32)
    assert mask_coverage(m, gt) == 1.0 and mask_coverage(m, np.array([22, 10, 30, 20])) == pytest.approx(0.8)
    assert contains(np.array([19.5, 10, 30, 20]), gt) and not contains(np.array([22, 10, 30, 20]), gt)
    assert tightness(gt, np.array([15, 5, 35, 25])) == pytest.approx(0.25)
    assert center_offset(gt, gt) == 0.0 and center_offset(gt, np.array([24, 10, 34, 20])) == pytest.approx(4 / math.hypot(10, 10))


def test_average_precision():
    assert average_precision(np.array([0.9, 0.8]), np.array([True, True]), n_gt=2) == pytest.approx(1.0)
    assert average_precision(np.array([0.9, 0.8]), np.array([False, True]), n_gt=1) == pytest.approx(0.5)
    assert average_precision(np.zeros(0), np.zeros(0, bool), n_gt=3) == 0.0


def test_box_metrics_cases():
    m = _mask()
    perfect = BoxMetrics(roi_margin=1.5, roi_min=0); perfect.update(m, _det(20, 10, 30, 20)); s = perfect.summary()
    assert s['n_gt'] == 1 and s['recall50'] == 1 and s['ap50'] == 1 and s['coverage_raw'] == 1 and s['coverage_recall99_raw'] == 1
    assert s['tightness'] == 1 and s['contain_rate'] == 1 and s['center_offset'] == 0 and s['matched_iou'] == 1
    cut = BoxMetrics(roi_margin=1.5, roi_min=0); cut.update(m, _det(22, 10, 30, 20)); s = cut.summary()
    assert s['recall50'] == 1 and s['coverage_raw'] == pytest.approx(0.8) and s['coverage_recall99_raw'] == 0   # IoU 0.8 but 20 % of the object lost
    assert s['coverage_roi'] == 1 and s['coverage_recall99_roi'] == 1 and math.isnan(s['tightness']) and s['contain_rate'] == 0
    big = BoxMetrics(roi_margin=1.0, roi_min=0); big.update(m, _det(15, 5, 35, 25)); s = big.summary()
    assert s['coverage_raw'] == 1 and s['tightness'] == pytest.approx(0.25) and s['recall50'] == 0   # IoU 0.25
    none = BoxMetrics(); none.update(m, np.zeros((0, 6), np.float32)); s = none.summary()
    assert s['recall50'] == 0 and s['ap50'] == 0 and math.isnan(s['tightness']) and s['coverage_raw'] == 0 and s['dets_per_image'] == 0
    two = BoxMetrics(roi_margin=1.0, roi_min=0)
    m2 = m.copy(); m2[30:40, 2:12] = True                                    # second object (10x10)
    two.update(m2, np.array([[20, 10, 30, 20, 0.9, 0], [2, 30, 12, 40, 0.8, 0], [0, 0, 3, 3, 0.7, 0]], np.float32))
    s = two.summary(); assert s['n_gt'] == 2 and s['recall50'] == 1 and s['ap50'] == pytest.approx(1.0) and s['dets_per_image'] == 3
    assert select_score(s) == (s['coverage_recall99_raw'], s['tightness'])


def test_filter_to_operating_point():
    d = np.array([[0, 0, 1, 1, 0.1, 0], [0, 0, 1, 1, 0.9, 0], [0, 0, 1, 1, 0.4, 0]], np.float32)
    assert filter_to_operating_point(d, 0.25, 5)[:, 4].tolist() == pytest.approx([0.9, 0.4])   # conf floor, then sorted
    assert filter_to_operating_point(d, 0.25, 1)[:, 4].tolist() == pytest.approx([0.9])        # top-k truncation
    assert filter_to_operating_point(d, 1.0, 5).shape == (0, 6)
    assert filter_to_operating_point(np.zeros((0, 6), np.float32), 0.25, 5).shape == (0, 6)


def test_operating_point_metrics_ignore_low_confidence_detections():
    '''An object covered only by a sub-threshold box counts in the raw metrics but not at the ROI operating point.'''
    m = _mask()
    b = BoxMetrics(roi_margin=1.5, roi_min=0, roi_conf=0.25, roi_topk=5)
    b.update(m, _det(20, 10, 30, 20, conf=0.1))                          # a perfect box, below the export threshold
    s = b.summary()
    assert s['n_gt'] == 1 and s['dets_per_image'] == 1 and s['recall50'] == 1 and s['coverage_recall99_raw'] == 1
    assert s['dets_per_image_op'] == 0 and s['recall50_op'] == 0 and s['coverage_recall99_raw_op'] == 0
    assert s['coverage_raw_op'] == 0 and s['coverage_roi_op'] == 0 and s['coverage_recall99_roi_op'] == 0
    assert s['contain_rate_op'] == 0 and math.isnan(s['tightness_op']) and math.isnan(s['center_offset_op'])
    assert s['ap50'] == 1 and not {'ap50_op', 'n_gt_op', 'n_images_op', 'matched_iou_op'} & set(s)
    assert select_score(s) == (s['coverage_recall99_raw'], s['tightness'])   # selection untouched by the _op keys


def test_operating_point_metrics_equal_the_plain_ones_when_all_dets_pass():
    m = _mask().copy(); m[30:40, 2:12] = True                            # two objects
    dets = np.array([[20, 10, 30, 20, 0.9, 0], [2, 30, 12, 40, 0.8, 0], [0, 0, 3, 3, 0.7, 0]], np.float32)
    b = BoxMetrics(roi_margin=1.0, roi_min=0, roi_conf=0.25, roi_topk=5)
    b.update(m, dets); s = b.summary()
    for k in OP_KEYS:
        a, o = s[k], s[k + '_op']
        assert (math.isnan(a) and math.isnan(o)) or a == o, f"{k}: {a} != {o}"
    trunc = BoxMetrics(roi_margin=1.0, roi_min=0, roi_conf=0.25, roi_topk=2)
    trunc.update(m, dets); st = trunc.summary()
    assert st['dets_per_image'] == 3 and st['dets_per_image_op'] == 2   # the 0.7 spurious box is cut by top-k
