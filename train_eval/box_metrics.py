import math
import numpy as np

from data_loader.boxes import boxes_from_mask, expand_box


def box_iou_matrix(a, b):
    a = np.asarray(a, np.float64).reshape(-1, 4); b = np.asarray(b, np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2]); rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)[:, None]; area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)[None, :]
    return inter / np.maximum(area_a + area_b - inter, 1e-12)


def match_greedy(gt_boxes, dets):
    '''One-to-one matching by descending IoU (ties by confidence); returns per-GT prediction index (-1) and IoU (0).'''
    K, M = len(gt_boxes), len(dets)
    idx, best = -np.ones(K, dtype=int), np.zeros(K)
    if K and M:
        iou = box_iou_matrix(gt_boxes, dets[:, :4])
        order = sorted(((iou[k, m], dets[m, 4], k, m) for k in range(K) for m in range(M) if iou[k, m] > 0), reverse=True)
        used_gt, used_pred = set(), set()
        for v, _, k, m in order:
            if k not in used_gt and m not in used_pred:
                idx[k], best[k] = m, v; used_gt.add(k); used_pred.add(m)
    return idx, best


def mask_coverage(comp, box):
    '''Fraction of the object's mask pixels inside the box (pixel edges: floor x1/y1, ceil x2/y2).'''
    H, W = comp.shape
    x1, y1 = max(0, int(math.floor(box[0]))), max(0, int(math.floor(box[1])))
    x2, y2 = min(W, int(math.ceil(box[2]))), min(H, int(math.ceil(box[3])))
    total = comp.sum()
    return float(comp[y1:y2, x1:x2].sum() / total) if total and x2 > x1 and y2 > y1 else 0.0


def contains(pred_box, gt_box, tol=0.5):
    return bool(pred_box[0] <= gt_box[0] + tol and pred_box[1] <= gt_box[1] + tol and pred_box[2] >= gt_box[2] - tol and pred_box[3] >= gt_box[3] - tol)


def tightness(gt_box, pred_box):
    area = lambda b: max(float(b[2] - b[0]), 0.0) * max(float(b[3] - b[1]), 0.0)
    return area(gt_box) / max(area(pred_box), 1e-12)


def center_offset(gt_box, pred_box):
    cg = ((gt_box[0] + gt_box[2]) / 2, (gt_box[1] + gt_box[3]) / 2); cp = ((pred_box[0] + pred_box[2]) / 2, (pred_box[1] + pred_box[3]) / 2)
    return float(math.hypot(cp[0] - cg[0], cp[1] - cg[1]) / max(math.hypot(gt_box[2] - gt_box[0], gt_box[3] - gt_box[1]), 1e-12))


def average_precision(confs, tp, n_gt):
    '''Single-class AP with all-points interpolation over confidence-sorted predictions.'''
    if n_gt == 0 or len(confs) == 0:
        return 0.0
    order = np.argsort(-np.asarray(confs)); tp = np.asarray(tp, bool)[order]
    ctp, cfp = np.cumsum(tp), np.cumsum(~tp)
    recall = ctp / n_gt; precision = ctp / np.maximum(ctp + cfp, 1e-12)
    mrec = np.concatenate([[0.0], recall, [1.0]]); mpre = np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    changes = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[changes + 1] - mrec[changes]) * mpre[changes + 1]))


class BoxMetrics(object):
    '''Accumulates per-object records over a split; every metric is defined against the GT mask component.'''

    def __init__(self, roi_margin=1.5, roi_min=256, min_area=100):
        self.roi_margin, self.roi_min, self.min_area = roi_margin, roi_min, min_area
        self.records, self.pred_conf, self.pred_tp, self.n_images, self.n_dets = [], [], [], 0, 0

    def update(self, mask, dets):
        H, W = mask.shape
        gt_boxes, labels, ids = boxes_from_mask(mask, self.min_area, return_labels=True)
        dets = np.asarray(dets, np.float32).reshape(-1, 6)
        self.n_images += 1; self.n_dets += len(dets)
        idx, best = match_greedy(gt_boxes, dets)
        for k, (gt, cid) in enumerate(zip(gt_boxes, ids)):
            comp = labels == cid
            rec = {'iou': float(best[k]), 'tp50': bool(best[k] >= 0.5), 'coverage_raw': 0.0, 'coverage_roi': 0.0,
                   'contains': False, 'tightness': float('nan'), 'offset': float('nan')}
            if idx[k] >= 0:
                p = dets[idx[k], :4]
                rec['coverage_raw'] = mask_coverage(comp, p)
                rec['coverage_roi'] = mask_coverage(comp, expand_box(p, self.roi_margin, self.roi_min, H, W))
                rec['contains'] = contains(p, gt)
                rec['tightness'] = tightness(gt, p) if rec['contains'] else float('nan')
                rec['offset'] = center_offset(gt, p)
            self.records.append(rec)
        # AP bookkeeping: confidence-ordered greedy TP flags at IoU >= 0.5
        matched = set()
        for m in np.argsort(-dets[:, 4]) if len(dets) else []:
            iou = box_iou_matrix(gt_boxes, dets[m:m + 1, :4])[:, 0] if len(gt_boxes) else np.zeros(0)
            cand = [(v, k) for k, v in enumerate(iou) if v >= 0.5 and k not in matched]
            tp = bool(cand)
            if tp:
                matched.add(max(cand)[1])
            self.pred_conf.append(float(dets[m, 4])); self.pred_tp.append(tp)

    def summary(self):
        r = self.records; n = len(r)
        nanmean = lambda xs: float(np.mean(xs)) if len(xs) else float('nan')
        return {'n_images': self.n_images, 'n_gt': n, 'dets_per_image': self.n_dets / max(self.n_images, 1),
                'recall50': float(np.mean([x['tp50'] for x in r])) if n else 0.0,
                'ap50': average_precision(self.pred_conf, self.pred_tp, n),
                'matched_iou': nanmean([x['iou'] for x in r if x['tp50']]),
                'coverage_raw': float(np.mean([x['coverage_raw'] for x in r])) if n else 0.0,
                'coverage_recall99_raw': float(np.mean([x['coverage_raw'] >= 0.99 for x in r])) if n else 0.0,
                'coverage_roi': float(np.mean([x['coverage_roi'] for x in r])) if n else 0.0,
                'coverage_recall99_roi': float(np.mean([x['coverage_roi'] >= 0.99 for x in r])) if n else 0.0,
                'contain_rate': float(np.mean([x['contains'] for x in r])) if n else 0.0,
                'tightness': nanmean([x['tightness'] for x in r if x['contains']]),
                'center_offset': nanmean([x['offset'] for x in r if not math.isnan(x['offset'])])}


def select_score(summary):
    '''Checkpoint ranking key: coverage recall @ 0.99 (raw box) first, mean tightness as tie-breaker (nan -> 0).'''
    t = summary['tightness']
    return (summary['coverage_recall99_raw'], 0.0 if (t is None or math.isnan(t)) else t)
