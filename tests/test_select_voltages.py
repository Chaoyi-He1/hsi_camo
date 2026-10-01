import numpy as np
import scipy.ndimage as ndi

from main_select_voltages import admissible_mask, frame_pixels, mahalanobis2, set_scores, greedy_select


def _problem(n_frames=4, seed=0):
    '''6 readings: 0/1 duplicates of a useless signal, 2 and 4 informative and independent, 3 pure noise, 5 near-copy of 4.'''
    rng = np.random.RandomState(seed)
    volts = np.array([-1.0, -0.99, 0.0, 0.5, 1.0, 1.03])
    d, P = [], []
    for _ in range(n_frames):
        base = rng.randn(1000, 3)                                                       # three independent latent signals
        Y = np.stack([base[:, 0], base[:, 0] + 1e-3 * rng.randn(1000), base[:, 1], rng.randn(1000), base[:, 2], base[:, 2] + 1e-3 * rng.randn(1000)], 1)
        d.append(np.array([0.0, 0.0, 1.0, 0.0, 1.5, 1.5]) + 0.05 * rng.randn(6))       # only latents 1 and 2 separate object from surround
        P.append(np.cov(Y.T))
    return np.array(d), np.array(P), volts


def test_admissible_mask_excludes_the_dead_zone_margin():
    volts = np.round(np.arange(-1.0, 2.5001, 0.01), 2)
    volts = volts[(volts < 0.25 - 1e-6) | (volts > 0.31 + 1e-6)]                        # the 344 usable voltages
    ok = admissible_mask(volts, 0.05)
    for v in (0.20, 0.24, 0.32, 0.36):
        assert not ok[np.isclose(volts, v)].any()
    for v in (0.19, 0.37, -1.0, 2.5):
        assert ok[np.isclose(volts, v)].all()
    assert ok.sum() == len(volts) - 10


def test_frame_pixels_object_and_ring():
    mask = np.zeros((60, 80), bool); mask[20:40, 30:50] = True
    obj, ring = frame_pixels(mask, 2, (3.0, 10.0), np.random.default_rng(0), 10000, 10000)
    flat = mask.ravel()
    assert flat[obj].all() and not flat[ring].any()
    d_in, d_out = ndi.distance_transform_edt(mask).ravel(), ndi.distance_transform_edt(~mask).ravel()
    assert (d_in[obj] > 2).all() and (d_out[ring] > 3).all() and (d_out[ring] <= 10).all()
    assert len(obj) == 16 * 16                                                           # 2 px eroded on every side
    obj_s, ring_s = frame_pixels(mask, 2, (3.0, 10.0), np.random.default_rng(0), 50, 70)
    assert len(obj_s) == 50 and len(ring_s) == 70 and flat[obj_s].all()                  # subsampled


def test_set_scores_matches_direct_mahalanobis():
    d, P, _ = _problem()
    nv = np.full(6, 0.1); idx = [2, 4]
    s = set_scores(d, P, nv, idx, 1e-3)
    for f in range(len(d)):
        PP = P[f][np.ix_(idx, idx)] + np.diag(nv[idx])
        assert np.isclose(s[f], mahalanobis2(d[f, idx], PP, 1e-3))
    assert np.isclose(mahalanobis2(np.array([1.0, 0.0]), np.eye(2), 0.0), 1.0)


def test_greedy_picks_informative_distinct_voltages():
    d, P, volts = _problem()
    allowed = np.ones(6, bool)
    chosen, objective = greedy_select(d, P, np.zeros(6), volts, 3, allowed, 0.05, 1e-9)
    assert chosen[0] in (4, 5) and chosen[1] == 2                                         # the informative readings first
    assert np.diff(np.sort(volts[chosen])).min() >= 0.05 - 1e-9                           # no near-duplicate pair
    assert all(b >= a - 1e-9 for a, b in zip(objective, objective[1:]))                    # the set only gains
    chosen2, _ = greedy_select(d, P, np.zeros(6), volts, 2, allowed & (volts != 0.0), 0.05, 1e-9)
    assert 2 not in chosen2                                                               # an excluded voltage is never chosen
