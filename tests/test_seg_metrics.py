import math
import warnings
import numpy as np
import pytest

psm = pytest.importorskip('py_sod_metrics')
from train_eval.seg_metrics import SegMetrics, pool, bootstrap_seg, image_scores, size_bucket, SUMMARY_KEYS


def _cases(seed=0, n=8):
    '''Hand-made (prob, gt) pairs: noisy rectangles of different sizes, a low-confidence map, a 1-pixel object.'''
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        h, w = int(rng.integers(24, 80)), int(rng.integers(24, 80))
        gt = np.zeros((h, w), bool)
        y0, x0 = int(rng.integers(0, h // 2)), int(rng.integers(0, w // 2))
        gt[y0:y0 + int(rng.integers(2, h // 2)), x0:x0 + int(rng.integers(2, w // 2))] = True
        prob = np.clip(gt * rng.uniform(0.3, 1.0) + rng.normal(0, 0.2, (h, w)), 0, 1)
        out.append((prob, gt))
    g = np.zeros((40, 40), bool); g[10:20, 10:20] = True
    out.append((0.2 * g + 0.05 * rng.uniform(0, 1, (40, 40)), g))           # low-confidence map, max 0.25
    g1 = np.zeros((30, 30), bool); g1[4, 7] = True
    out.append((0.5 * rng.uniform(0, 1, (30, 30)), g1))                      # 1-pixel object
    return out


def _reference(cases, u8):
    '''py_sod_metrics on its own: one object per metric over all images, get_results().'''
    sm, em, wfm, mae = psm.Smeasure(), psm.Emeasure(), psm.WeightedFmeasure(), psm.MAE()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fm = psm.Fmeasure()                                                  # the deprecated class, as a second reference
    for prob, gt in cases:
        if u8:
            p = (prob - prob.min()) / (prob.max() - prob.min() + 1e-8)
            pred_in, gt_in, norm = (p * 255).astype(np.uint8), gt.astype(np.uint8) * 255, True
        else:
            pred_in, gt_in, norm = prob.astype(np.float64), gt, False
        for m in (sm, em, wfm, mae, fm):
            m.step(pred_in, gt_in, normalize=norm)
    e, f = em.get_results()['em'], fm.get_results()['fm']
    return {'S': sm.get_results()['sm'], 'E_mean': e['curve'].mean(), 'E_max': e['curve'].max(), 'E_adp': e['adp'],
            'Fw': wfm.get_results()['wfm'], 'MAE': mae.get_results()['mae'], 'F_adp': f['adp'],
            'F_mean': f['curve'].mean(), 'F_max': f['curve'].max()}


@pytest.mark.parametrize('minmax', [False, True])
def test_matches_py_sod_metrics(minmax):
    cases = _cases()
    met = SegMetrics(minmax=minmax)
    for prob, gt in cases:
        met.update(prob, gt)
    s, ref = met.summary(), _reference(cases, u8=minmax)
    for k, v in ref.items():
        assert s[k] == pytest.approx(v, abs=1e-12), f"{k}: {s[k]} vs py_sod_metrics {v}"
    assert s['n'] == len(cases) and s['n_fp'] == 0 and math.isnan(s['fp_false_mask_rate'])
    assert set(SUMMARY_KEYS) <= set(s) and all(f'{k}_{b}' in s for k in SUMMARY_KEYS for b in ('small', 'medium', 'large'))


def test_iou_on_raw_probability():
    gt = np.zeros((20, 20), bool); gt[5:15, 5:15] = True                     # 100 px
    pred = np.zeros((20, 20), np.float32); pred[5:15, 10:20] = 0.9           # 50 px inside, 50 px outside
    r = image_scores(pred, gt)
    assert r['IoU'] == pytest.approx(50 / 150)
    low = 0.4 * gt.astype(np.float32)                                        # never above 0.5: IoU 0 in both conventions
    assert image_scores(low, gt)['IoU'] == 0.0 and image_scores(low, gt, minmax=True)['IoU'] == 0.0
    assert image_scores(low, gt, minmax=True)['S'] > image_scores(low, gt)['S']   # min-max inflates a low-confidence map
    assert image_scores(np.zeros((20, 20)), np.zeros((20, 20), bool))['IoU'] == 0.0


def test_input_dtypes_and_clipping():
    gt = np.zeros((16, 16), bool); gt[4:12, 4:12] = True
    p = np.clip(gt + np.random.default_rng(0).normal(0, 0.1, (16, 16)), 0, 1)
    a = image_scores(p.astype(np.float16), gt.astype(np.uint8))              # fp16 pred and 0/1 uint8 gt accepted
    b = image_scores(p.astype(np.float16).astype(np.float64), gt)
    assert a['S'] == b['S'] and a['IoU'] == b['IoU']
    c = image_scores(p * 1.01, gt)                                           # slightly above 1 is clipped, not rejected
    assert np.isfinite(c['S'])
    with pytest.raises(AssertionError):
        image_scores(np.zeros((16, 15)), gt)


def test_size_buckets_and_area_override():
    H = W = 200
    met = SegMetrics(size_edges=(2000, 20000))
    rows = []
    for (h, w) in [(10, 10), (50, 100), (150, 200)]:                         # 100, 5000, 30000 px
        gt = np.zeros((H, W), bool); gt[:h, :w] = True
        rows.append(met.update(np.clip(gt * 0.8 + 0.05, 0, 1), gt))
    s = met.summary()
    assert (s['n_small'], s['n_medium'], s['n_large']) == (1, 1, 1)
    assert s['S_small'] == rows[0]['S'] and s['S_medium'] == rows[1]['S'] and s['S_large'] == rows[2]['S']
    assert s['IoU_large'] == pytest.approx(1.0)
    assert size_bucket(1999) == 'small' and size_bucket(2000) == 'medium' and size_bucket(20000) == 'medium' and size_bucket(20001) == 'large'
    # area override: the caller's object area decides the bucket, not the GT pixels of the crop
    met2 = SegMetrics()
    gt = np.zeros((40, 40), bool); gt[0:10, 0:10] = True
    met2.update(gt.astype(np.float32), gt, area=25000)
    s2 = met2.summary()
    assert s2['n_large'] == 1 and s2['n_small'] == 0 and math.isnan(s2['S_small'])


def test_false_mask_rate_excluded_from_scores():
    met = SegMetrics()
    gt = np.zeros((20, 20), bool); gt[5:10, 5:10] = True
    met.update(gt.astype(np.float32), gt)
    s_obj = met.summary()
    met.update_fp(np.zeros((12, 9)))
    met.update_fp(np.full((12, 9), 0.5))                                     # exactly 0.5 is not a mask pixel
    f = np.zeros((12, 9)); f[3, 3] = 0.51
    met.update_fp(f)
    s = met.summary()
    assert s['n_fp'] == 3 and s['fp_false_mask_rate'] == pytest.approx(1 / 3)
    assert s['n'] == 1 and all(s[k] == s_obj[k] for k in SUMMARY_KEYS)      # fp rows never touch S / E / F / IoU


def test_pool_reproduces_summary_and_curve_averaging():
    met = SegMetrics()
    for i, (prob, gt) in enumerate(_cases(seed=1)):
        met.update(prob, gt, frame=str(i))
    met.update_fp(np.full((8, 8), 0.7), frame='0')
    s, p = met.summary(), pool(met.per_image)
    assert set(s) == set(p)
    for k in s:
        assert (math.isnan(s[k]) and math.isnan(p[k])) or s[k] == p[k], k
    rows = [r for r in met.per_image if r['kind'] == 'obj']
    assert all(r['E_curve'].shape == (256,) and r['F_curve'].shape == (256,) for r in rows)
    # E_max is the max of the image-averaged curve, never larger than the mean of per-image maxima
    assert s['E_max'] == pytest.approx(np.mean([r['E_curve'] for r in rows], axis=0).max())
    assert s['E_max'] <= np.mean([r['E_curve'].max() for r in rows]) + 1e-12
    # pooling two halves together equals one object updated with both
    half = len(met.per_image) // 2
    q = pool(met.per_image[:half] + met.per_image[half:])
    assert q['S'] == s['S'] and q['E_max'] == s['E_max']


def _run(seed, noise, frames=6, rois_per_frame=2):
    '''One "run": per frame a few ROI rows (frame, key) with predictions of a given noise level.'''
    rng = np.random.default_rng(seed)
    met = SegMetrics()
    for f in range(frames):
        for k in range(rois_per_frame):
            gt = np.zeros((24, 24), bool); gt[4 + k:14 + k, 6:16] = True
            met.update(np.clip(gt + rng.normal(0, noise, gt.shape), 0, 1), gt, frame=str(f), key=k)
    return met.per_image


def test_bootstrap_seg_paired_frames():
    good = [_run(s, 0.05) for s in range(2)]
    bad = [_run(10 + s, 0.6) for s in range(2)]
    keys = ['S', 'Fw', 'E_max', 'IoU']
    same = bootstrap_seg(good, good, keys, n=50, seed=0)
    assert all(same[k]['diff'] == 0 and same[k]['lo'] == 0 and same[k]['hi'] == 0 for k in keys)
    ci = bootstrap_seg(good, bad, keys, n=200, seed=0)
    for k in keys:
        assert ci[k]['diff'] > 0 and ci[k]['lo'] > 0 and ci[k]['lo'] <= ci[k]['diff'] <= ci[k]['hi'] and ci[k]['p_gt0'] == 1.0
    # point difference = difference of run-averaged pooled summaries
    exp = np.mean([pool(r)['S'] for r in good]) - np.mean([pool(r)['S'] for r in bad])
    assert ci['S']['diff'] == pytest.approx(exp)
    # same seed -> same draws
    assert bootstrap_seg(good, bad, keys, n=50, seed=3) == bootstrap_seg(good, bad, keys, n=50, seed=3)
    # unpaired rows (a different ROI set) are refused
    with pytest.raises(AssertionError):
        bootstrap_seg(good, [_run(0, 0.05, rois_per_frame=3)], keys, n=10)
