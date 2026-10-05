'''
Stage-2 segmentation metrics (spec §5.5, §8) on top of py_sod_metrics (PySODMetrics 1.6.2, MIT).

Every prediction is scored once per image (an ROI crop at native size, or a full 1680 x 1240 frame) and kept as one
row of per_image; summary() and the module-level pool() average those rows, so a split can be re-pooled image by image
(size buckets, frame bootstrap) and always gives the same numbers as one SegMetrics updated with those images in order.

Metrics per image (alpha / beta as in the COD literature):
  S       S-measure, alpha = 0.5                                     (py_sod_metrics.Smeasure)
  E_*     E-measure: 256-threshold curve, adaptive value             (py_sod_metrics.Emeasure)
  Fw      weighted F-measure, beta = 1                               (py_sod_metrics.WeightedFmeasure)
  F_*     F-measure, beta^2 = 0.3: 256-threshold curve, adaptive     (py_sod_metrics.FmeasureV2 + FmeasureHandler, which
          gives the same numbers as the deprecated Fmeasure without its warning on every construction)
  MAE     mean |pred - gt|                                           (py_sod_metrics.MAE)
  IoU     IoU of the RAW probability binarised at 0.5 (computed here, never min-max normalised; 0 when pred and GT are
          both empty, the py_sod_metrics divide convention)
Pooling: S, Fw, MAE, E_adp, F_adp and IoU are plain means over images. E_mean / E_max (and F_mean / F_max) are the mean /
max over thresholds of the curve AVERAGED OVER IMAGES first - the mean of per-image maxima is a different, larger number
(0.91 vs 0.81 on the recon test set) - so every row keeps its two [256] curves. Curve index i <-> threshold 255 - i on
floor(pred * 255) with >= (index 0 = threshold 255, index 255 = every pixel foreground).

Input convention (minmax):
  False (default, development metric): pred float64 clipped to [0, 1], gt bool, step(..., normalize=False): the
        probabilities are scored as they are.
  True  (the SAM2-UNet / HyperCOD Table 2 protocol): per-image min-max -> * 255 -> uint8, gt * 255 uint8,
        step(..., normalize=True) (py_sod_metrics min-maxes again); IoU stays on the raw probability. This inflates
        low-confidence maps (S 0.80 vs 0.64 on a map whose maximum is 0.2), so it is only for the paper comparison.

False-positive ROIs (GT empty inside the ROI) have no meaningful S / E / Fw, so they never enter those means; update_fp()
records whether any pixel is > 0.5 and summary() reports the share (fp_false_mask_rate) and their number (n_fp).
Size buckets follow the GT area of each row (small < 2000 px, medium 2000-20000, large > 20000 by default); the caller may
pass the object's own frame-level area, since a crop's GT union can include part of a neighbour.
'''
import numpy as np
import py_sod_metrics as psm

SCORE_KEYS = ('S', 'E_adp', 'Fw', 'F_adp', 'MAE', 'IoU')                  # per-image scalars, pooled by plain mean
SUMMARY_KEYS = ('S', 'E_mean', 'E_max', 'E_adp', 'Fw', 'F_adp', 'F_mean', 'F_max', 'MAE', 'IoU')
BUCKETS = ('small', 'medium', 'large')
N_THRESHOLDS = 256


def _to_pred(pred):
    '''Prediction as float64 in [0, 1] (py_sod_metrics with normalize=False rejects fp16 and values outside [0, 1]).'''
    pred = np.asarray(pred, dtype=np.float64)
    assert pred.ndim == 2, f"pred must be [h, w], got shape {pred.shape}"
    assert np.isfinite(pred).all(), f"pred has {int((~np.isfinite(pred)).sum())} non-finite values"
    return np.clip(pred, 0.0, 1.0)


def image_scores(pred, gt, minmax=False):
    '''
    All metrics of one image: pred [h, w] float in [0, 1] (any float dtype), gt [h, w] bool.
    Returns {S, E_adp, Fw, F_adp, MAE, IoU (floats), E_curve, F_curve ([256] float64)}.
    '''
    pred = _to_pred(pred)
    gt = np.asarray(gt).astype(bool)
    assert pred.shape == gt.shape, f"pred {pred.shape} and gt {gt.shape} differ"
    if minmax:
        # SAM2-UNet test.py: per-image min-max, saved as uint8 PNG, read back and scored with normalize=True
        p = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
        pred_in, gt_in, norm = (p * 255).astype(np.uint8), gt.astype(np.uint8) * 255, True
    else:
        pred_in, gt_in, norm = pred, gt, False
    # fresh objects per image: each step() appends one value per metric, read back with [-1] (identical to reusing one
    # object, verified in recon); Emeasure must go through step(), which sets gt_fg_numel / gt_size before cal_*
    sm, em, wfm, mae = psm.Smeasure(alpha=0.5), psm.Emeasure(), psm.WeightedFmeasure(beta=1), psm.MAE()
    fm_handler = psm.FmeasureHandler(with_dynamic=True, with_adaptive=True, beta=0.3)
    fm = psm.FmeasureV2(metric_handlers={'fm': fm_handler})
    for m in (sm, em, wfm, mae, fm):
        m.step(pred_in, gt_in, normalize=norm)
    # IoU at 0.5 on the RAW probability in both conventions
    b = pred > 0.5
    inter, union = np.count_nonzero(b & gt), np.count_nonzero(b | gt)
    return {'S': float(sm.sms[-1]), 'E_adp': float(em.adaptive_ems[-1]), 'Fw': float(wfm.weighted_fms[-1]),
            'F_adp': float(fm_handler.adaptive_results[-1]), 'MAE': float(mae.maes[-1]),
            'IoU': float(inter / union) if union > 0 else 0.0,
            'E_curve': np.asarray(em.changeable_ems[-1], dtype=np.float64).reshape(N_THRESHOLDS),     # [256]
            'F_curve': np.asarray(fm_handler.dynamic_results[-1], dtype=np.float64).reshape(N_THRESHOLDS)}


def size_bucket(area, size_edges=(2000, 20000)):
    '''small: area < edges[0]; medium: edges[0] <= area <= edges[1]; large: area > edges[1].'''
    return 'small' if area < size_edges[0] else ('medium' if area <= size_edges[1] else 'large')


def _as_arrays(rows):
    '''Columns of a per_image list: object rows (kind 'obj') and false-positive rows (kind 'fp') separately.'''
    obj = [r for r in rows if r['kind'] == 'obj']
    fp = [r for r in rows if r['kind'] == 'fp']
    a = {k: np.array([r[k] for r in obj], dtype=np.float64) for k in SCORE_KEYS}
    a['area'] = np.array([r['area'] for r in obj], dtype=np.float64)
    a['E_curve'] = np.stack([r['E_curve'] for r in obj]) if obj else np.zeros((0, N_THRESHOLDS))   # [n, 256]
    a['F_curve'] = np.stack([r['F_curve'] for r in obj]) if obj else np.zeros((0, N_THRESHOLDS))   # [n, 256]
    a['false_mask'] = np.array([r['false_mask'] for r in fp], dtype=np.float64)                     # [m]
    return a


def _stats(a, idx):
    '''Pooled metrics of the object rows idx (a repeated index counts twice, as the bootstrap needs); nan when empty.'''
    if len(idx) == 0:
        return {**{k: float('nan') for k in SUMMARY_KEYS}, 'n': 0}
    e = a['E_curve'][idx].mean(axis=0)                                     # [256] curve averaged over images
    f = a['F_curve'][idx].mean(axis=0)                                     # [256]
    out = {k: float(a[k][idx].mean()) for k in SCORE_KEYS}
    out.update(E_mean=float(e.mean()), E_max=float(e.max()), F_mean=float(f.mean()), F_max=float(f.max()), n=int(len(idx)))
    return {k: out[k] for k in (*SUMMARY_KEYS, 'n')}


def _pool_arrays(a, obj_idx, fp_idx, size_edges):
    '''Summary of the object rows obj_idx and the false-positive rows fp_idx of the column dict a.'''
    obj_idx, fp_idx = np.asarray(obj_idx, dtype=int), np.asarray(fp_idx, dtype=int)
    out = _stats(a, obj_idx)
    area = a['area'][obj_idx]
    masks = {'small': area < size_edges[0], 'medium': (area >= size_edges[0]) & (area <= size_edges[1]),
             'large': area > size_edges[1]}
    for b in BUCKETS:
        out.update({f'{k}_{b}': v for k, v in _stats(a, obj_idx[masks[b]]).items()})
    out['n_fp'] = int(len(fp_idx))
    out['fp_false_mask_rate'] = float(a['false_mask'][fp_idx].mean()) if len(fp_idx) else float('nan')
    return out


def pool(per_image, size_edges=(2000, 20000)):
    '''Summary of a per_image list (SegMetrics.per_image, possibly several concatenated): identical to summary().'''
    a = _as_arrays(per_image)
    return _pool_arrays(a, np.arange(len(a['S'])), np.arange(len(a['false_mask'])), size_edges)


class SegMetrics(object):
    '''
    Accumulates per-image segmentation scores over a split.
      update(pred, gt, area=None, frame=None, key=None): one object image (an ROI crop, or a full frame); area = the GT
        area used for the size bucket (default gt.sum()); frame / key identify the row for the paired bootstrap
        (bootstrap_seg resamples frames and requires the same (frame, key) rows in every run compared).
      update_fp(pred, frame=None, key=None): one false-positive ROI (no GT inside); only the false-mask flag is kept.
      summary(): S, E_mean, E_max, E_adp, Fw, F_adp, F_mean, F_max, MAE, IoU, n, the same keys suffixed
        _small / _medium / _large, fp_false_mask_rate and n_fp.
    '''

    def __init__(self, size_edges=(2000, 20000), minmax=False):
        assert len(size_edges) == 2 and size_edges[0] <= size_edges[1], f"size_edges must be (lo, hi), got {size_edges}"
        self.size_edges = (float(size_edges[0]), float(size_edges[1]))
        self.minmax = bool(minmax)  # False: raw probabilities (development); True: SAM2-UNet / HyperCOD min-max protocol
        self.per_image = []         # one dict per update() / update_fp() call, in call order

    def update(self, pred, gt, area=None, frame=None, key=None):
        row = image_scores(pred, gt, self.minmax)
        row.update(kind='obj', frame=frame, key=key,
                   area=int(np.count_nonzero(gt)) if area is None else int(area))
        self.per_image.append(row)
        return row

    def update_fp(self, pred, frame=None, key=None):
        pred = _to_pred(pred)
        row = {'kind': 'fp', 'frame': frame, 'key': key, 'false_mask': bool((pred > 0.5).any()), 'max_prob': float(pred.max())}
        self.per_image.append(row)
        return row

    def summary(self):
        return pool(self.per_image, self.size_edges)


def _row_ids(rows):
    return [(r['kind'], r['frame'], r['key']) for r in rows]


def _units(rows):
    '''
    Resampling units of a per_image list: rows grouped by frame (cluster bootstrap: the ROIs of one frame are not
    independent) when every row has a frame, else every row is its own unit. Returns (obj_groups, fp_groups): per unit
    the indices into the object rows and into the false-positive rows (the order _as_arrays uses).
    '''
    by_frame = all(r['frame'] is not None for r in rows)
    order, obj_g, fp_g = [], {}, {}
    n_obj = n_fp = 0
    for j, r in enumerate(rows):
        u = r['frame'] if by_frame else j
        if u not in obj_g:
            order.append(u); obj_g[u], fp_g[u] = [], []
        if r['kind'] == 'obj':
            obj_g[u].append(n_obj); n_obj += 1
        else:
            fp_g[u].append(n_fp); n_fp += 1
    return [np.asarray(obj_g[u], dtype=int) for u in order], [np.asarray(fp_g[u], dtype=int) for u in order]


def bootstrap_seg(runs_a, runs_b, keys, n=2000, seed=0, size_edges=(2000, 20000)):
    '''
    Paired bootstrap of mean(metric over runs_a) - mean(metric over runs_b) (e.g. the 3 seeds of two arms).
    runs_* are lists of per_image lists that must hold the same rows (kind, frame, key) in the same order; frames are
    drawn with replacement (all rows of a drawn frame together; rows without a frame are drawn one by one), the same
    draw for every run, and each draw re-pools the curves (E_max is the max of the resampled mean curve).
    Returns {key: {'diff': point difference, 'lo' / 'hi': 2.5 / 97.5 percentiles, 'p_gt0': share of draws > 0}}.
    '''
    runs = list(runs_a) + list(runs_b)
    assert len(runs_a) and len(runs_b), "bootstrap_seg needs at least one run on each side"
    ref = _row_ids(runs[0])
    for r in runs[1:]:
        ids = _row_ids(r)
        assert ids == ref, (f"runs are not paired: {len(ids)} vs {len(ref)} rows, first difference at "
                            f"{next((i for i, (x, y) in enumerate(zip(ids, ref)) if x != y), min(len(ids), len(ref)))}")
    obj_g, fp_g = _units(runs[0])
    m = len(obj_g)
    arrays = [_as_arrays(r) for r in runs]
    na = len(runs_a)

    def side_means(stats):
        # nan-safe mean over the runs of one side (a bucket can be empty in a draw)
        out = {}
        for k in keys:
            va = [s[k] for s in stats[:na] if not np.isnan(s[k])]
            vb = [s[k] for s in stats[na:] if not np.isnan(s[k])]
            out[k] = (np.mean(va) - np.mean(vb)) if va and vb else np.nan
        return out

    full_obj, full_fp = np.arange(len(arrays[0]['S'])), np.arange(len(arrays[0]['false_mask']))
    point = side_means([_pool_arrays(a, full_obj, full_fp, size_edges) for a in arrays])
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in keys}
    for _ in range(n):
        idx = rng.integers(0, m, m)                                        # [m] units drawn with replacement, shared by all runs
        oi = np.concatenate([obj_g[i] for i in idx]) if m else np.zeros(0, int)
        fi = np.concatenate([fp_g[i] for i in idx]) if m else np.zeros(0, int)
        d = side_means([_pool_arrays(a, oi, fi, size_edges) for a in arrays])
        for k in keys:
            draws[k].append(d[k])
    out = {}
    for k in keys:
        v = np.asarray(draws[k], dtype=np.float64)
        ok = ~np.isnan(v)
        out[k] = dict(diff=float(point[k]), lo=float(np.percentile(v[ok], 2.5)) if ok.any() else float('nan'),
                      hi=float(np.percentile(v[ok], 97.5)) if ok.any() else float('nan'),
                      p_gt0=float(np.mean(v[ok] > 0)) if ok.any() else float('nan'))
    return out
