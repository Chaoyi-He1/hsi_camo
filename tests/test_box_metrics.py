import math
import numpy as np
import pytest

from train_eval.box_metrics import (box_iou_matrix, match_greedy, mask_coverage, contains, tightness, center_offset,
                                    average_precision, BoxMetrics, select_score)

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
