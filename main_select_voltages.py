"""
Choose the K bias voltages a K-reading EC sensor should use (spec §12, "how to pick the voltages").

Forward greedy selection on the TRAIN frames: every step adds the voltage that most raises the geometric mean over frames
of the object-vs-surround separability of the whole SET of readings (Mahalanobis distance between the object pixels and
their 3-60 px surround under the pooled covariance), scored under a per-reading read-noise floor (--snr_db, the model of
models.ec_yolo.read_noise_std). Scoring the set, not single channels, is what session A's per-channel gate could not do:
the 344 responses are near-duplicates (adjacent voltages correlate at 0.99998), so the information sits in differences
between readings and a voltage is worth only what it adds to the ones already chosen. Constraints: at least --min_spacing V
between chosen voltages (a near-duplicate is a wasted reading) and nothing within --dead_zone_margin V of the polarity-flip
dead zone (0.25-0.31 V), where the measured response is least reliable. The result is nested: the first K' voltages serve
a K'-reading sensor. Validation frames are read for REPORTING only, never for the choice; the test split is not touched.
  CUDA_VISIBLE_DEVICES="" python main_select_voltages.py --k 10 --read_noise_db 40 --out results/det/voltages_greedy.json
Feed the result to the detector with --filter-select manual --filter-voltages <voltages> --pca-channels K --read_noise_db <dB>
(the same --read_noise_db / --read_noise_model flags as main_det.py, so the sets are scored under the noise they are trained with).
"""
import os
import json
import time
import argparse
import numpy as np
import scipy.ndimage as ndi
import torch

import main_det
from data_loader.ec_filter import FILTER_DEAD_ZONE_V, select_filter_channels
from models.ec_yolo import read_noise_std

SESSION_B_VOLTAGES = [1.32, 1.29, 1.35, -0.41, -0.38, 1.38, -0.44, 1.41, 1.26, -0.35]  # det_A's gate top-10 (weights/det_A/gate_ranking_ep89.csv)
RIDGE_RAW = 1e-3  # relative ridge of the raw-band reference separability (the convention of the Stage-1 analyses)


def get_args_parser():
    parser = argparse.ArgumentParser('Greedy EC voltage selection', parents=[main_det.get_args_parser()], add_help=False)
    # selection parameters
    parser.add_argument('--k', type=int, default=10, help='number of voltages (readings per frame) to choose')
    parser.set_defaults(read_noise_db=40.0)  # the sets are scored under main_det's --read_noise_db / --read_noise_model (floor), 40 dB by default
    parser.add_argument('--ridge', type=float, default=1e-9, help='relative ridge on the pooled covariance of a set (the noise already regularises it)')
    parser.add_argument('--min_spacing', type=float, default=0.05, help='minimum distance between chosen voltages (V)')
    parser.add_argument('--dead_zone_margin', type=float, default=0.05, help='exclude voltages within this distance of the dead zone (V)')
    # pixel sampling
    parser.add_argument('--erode', type=int, default=2, help='object pixels = GT mask eroded by this many px')
    parser.add_argument('--ring', type=float, nargs=2, default=[3.0, 60.0], help='surround = pixels this far outside the mask (px)')
    parser.add_argument('--obj_samples', type=int, default=6000, help='object pixels sampled per frame')
    parser.add_argument('--ring_samples', type=int, default=12000, help='surround pixels sampled per frame')
    parser.add_argument('--min_obj_px', type=int, default=200, help='frames with a smaller object are skipped')
    parser.add_argument('--max_frames', type=int, default=0, help='use only the first N frames of each split (smoke runs)')
    parser.add_argument('--no_val', action='store_true', help='skip the validation-frame report')
    # output
    parser.add_argument('--out', type=str, default='results/det/voltages_greedy.json', help='where the chosen voltages and the report go')
    return parser


def admissible_mask(volts, dead_zone_margin, dead_zone=FILTER_DEAD_ZONE_V):
    '''bool [N]: voltages farther than dead_zone_margin from the polarity-flip dead zone (volts are already outside it).'''
    lo, hi = dead_zone
    dist = np.maximum(np.maximum(lo - volts, volts - hi), 0.0)  # [N] distance to the dead zone, 0 inside
    return dist > dead_zone_margin + 1e-9


def frame_pixels(mask, erode, ring, rng, n_obj, n_ring):
    '''
    Flat indices of the object pixels (mask eroded by `erode` px, or the whole mask if fewer than 100 remain) and of the
    surround (ring[0] < distance outside the mask <= ring[1] px), each subsampled to at most n_obj / n_ring pixels.
    mask: bool [H, W]
    '''
    d_in = ndi.distance_transform_edt(mask)  # [H, W] distance to the nearest background pixel
    d_out = ndi.distance_transform_edt(~mask)  # [H, W] distance to the nearest object pixel
    obj = np.flatnonzero((d_in > erode).ravel()) if (d_in > erode).sum() >= 100 else np.flatnonzero(mask.ravel())
    sur = np.flatnonzero(((d_out > ring[0]) & (d_out <= ring[1])).ravel())
    if len(obj) > n_obj:
        obj = rng.choice(obj, n_obj, replace=False)
    if len(sur) > n_ring:
        sur = rng.choice(sur, n_ring, replace=False)
    return obj, sur


def mahalanobis2(d, P, ridge):
    '''d^T (P + ridge * tr(P)/n I)^-1 d'''
    P = P + ridge * np.trace(P) / len(d) * np.eye(len(d))
    return float(d @ np.linalg.solve(P, d))


def frame_stats(img, mask, R, mean, std, mu, band_std, obj, sur):
    '''
    Object-vs-surround statistics of one frame in the standardised reading space z = (R^T x - mean) / std:
    d [N] mean difference, P [N, N] pooled covariance, and the squared separability of the raw bands as reference.
    img: [C, H, W] p99-scaled cube, obj / sur: flat pixel indices (frame_pixels)
    '''
    flat = img.reshape(img.shape[0], -1)  # [C, H*W]
    Xo = np.asarray(flat[:, obj], np.float64).T  # [n_obj, C]
    Xs = np.asarray(flat[:, sur], np.float64).T  # [n_sur, C]
    Yo = (Xo @ R - mean) / std  # [n_obj, N] standardised readings
    Ys = (Xs @ R - mean) / std  # [n_sur, N]
    d = Yo.mean(0) - Ys.mean(0)  # [N]
    P = 0.5 * (np.cov(Yo.T) + np.cov(Ys.T))  # [N, N]
    Zo = (Xo - mu) / band_std  # [n_obj, C] standardised raw bands (the reference representation)
    Zs = (Xs - mu) / band_std  # [n_sur, C]
    raw_d2 = mahalanobis2(Zo.mean(0) - Zs.mean(0), 0.5 * (np.cov(Zo.T) + np.cov(Zs.T)), RIDGE_RAW)
    return d, P, raw_d2


def set_scores(d, P, noise_var, idx, ridge):
    '''
    Squared separability of the reading set idx on every frame: d_S^T (P_S + diag(noise_var_S) + ridge tr/k I)^-1 d_S,
    solved for all frames at once. d [F, N], P [F, N, N], noise_var [N] (standardised units) -> [F]
    '''
    idx = np.asarray(idx, dtype=int)
    k = len(idx)
    dd = d[:, idx]  # [F, k]
    PP = P[:, idx[:, None], idx[None, :]] + np.diag(noise_var[idx])  # [F, k, k]
    tr = np.einsum('fkk->f', PP) / k  # [F]
    PP = PP + (ridge * tr)[:, None, None] * np.eye(k)
    sol = np.linalg.solve(PP, dd[..., None])[..., 0]  # [F, k]
    return np.einsum('fk,fk->f', dd, sol)


def greedy_select(d, P, noise_var, volts, k, allowed, min_spacing, ridge):
    '''
    Forward greedy selection: every step adds the admissible voltage that maximises the mean over frames of log D^2 of the
    set (its geometric-mean separability); voltages closer than min_spacing to a chosen one are skipped.
    Returns the chosen indices in greedy order and the objective after each step.
    '''
    chosen, objective = [], []
    for step in range(k):
        best, best_j = -np.inf, -1
        for j in np.flatnonzero(allowed):
            if j in chosen or (chosen and np.abs(volts[chosen] - volts[j]).min() < min_spacing - 1e-9):
                continue
            score = float(np.mean(np.log(np.maximum(set_scores(d, P, noise_var, chosen + [int(j)], ridge), 1e-300))))
            if score > best:
                best, best_j = score, int(j)
        assert best_j >= 0, f"no admissible voltage left at step {step + 1} (min_spacing {min_spacing} V too large for k={k}?)"
        chosen.append(best_j)
        objective.append(best)
    return chosen, objective


def collect(dataset, n_frames, R, mean, std, mu, band_std, args):
    '''
    Per-frame statistics of the first n_frames frames of a split (frames read in parallel by a DataLoader):
    d [F, N], P [F, N, N], raw_d2 [F], names [F]
    '''
    rng = np.random.default_rng(args.seed)
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, list(range(n_frames))), batch_size=None,
                                         shuffle=False, num_workers=args.num_workers)
    d, P, raw_d2, names = [], [], [], []
    t0 = time.time()
    for i, (img, gt, name) in enumerate(loader):
        img, mask = img.numpy(), gt.numpy()[0] > 0.5  # [C, H, W] fp16, bool [H, W]
        if mask.sum() < args.min_obj_px:
            print(f"  skip frame {name}: {int(mask.sum())} object px")
            continue
        obj, sur = frame_pixels(mask, args.erode, args.ring, rng, args.obj_samples, args.ring_samples)
        d_f, P_f, raw_f = frame_stats(img, mask, R, mean, std, mu, band_std, obj, sur)
        d.append(d_f); P.append(P_f); raw_d2.append(raw_f); names.append(str(name))
        if (i + 1) % 10 == 0 or i + 1 == n_frames:
            print(f"  {dataset.split} {i + 1}/{n_frames} frames, {time.time() - t0:.0f}s", flush=True)
    return np.array(d), np.array(P), np.array(raw_d2), names


def report_set(stats, idx, noise_var, ridge, snr_db):
    '''
    Median (and 10th percentile) over frames of the set's separability as a fraction of the raw bands', under four readouts:
    un-whitened (ridge 1e-3, no noise: what un-whitened channels cost), whitened (ridge 1e-9, no noise: the information
    carried), and whitened under read noise at snr_db and at 30 dB.
    '''
    d, P, raw_d2 = stats
    conditions = {'unwhitened': (RIDGE_RAW, np.zeros_like(noise_var)), 'whitened': (ridge, np.zeros_like(noise_var)), f'{snr_db:g}dB': (ridge, noise_var)}
    if snr_db != 30.0:
        conditions['30dB'] = (ridge, noise_var * 10 ** ((snr_db - 30.0) / 10.0))
    out = {}
    for name, (rdg, nv) in conditions.items():
        ratio = np.sqrt(np.maximum(set_scores(d, P, nv, idx, rdg), 0.0) / raw_d2)  # [F]
        out[name] = {'median': round(float(np.median(ratio)), 4), 'p10': round(float(np.percentile(ratio, 10)), 4)}
    return out


def main(args):
    t0 = time.time()
    # every usable voltage is a candidate; the dataset is only used for its frames, filter matrix and band statistics
    args.filter_select, args.filter_voltages = 'all', None
    dataset_train, dataset_val, _ = main_det.build_datasets(args)
    R, mean, std, volts = [np.asarray(a, np.float64) for a in dataset_train.filter_bank_tensors()]  # [C, N], [N], [N], [N]
    mu, cov = dataset_train._band_stats()  # [C], [C, C]
    band_std = np.sqrt(np.diag(cov))  # [C]
    assert args.read_noise_db > 0, f"--read_noise_db must be positive (the sets are scored under read noise), got {args.read_noise_db}"
    sigma = read_noise_std(R, mu, cov, args.read_noise_db, args.read_noise_model).astype(np.float64)  # [N] reading units (R is the whole bank)
    noise_var = (sigma / std) ** 2  # [N] in standardised units
    allowed = admissible_mask(volts, args.dead_zone_margin)  # [N]
    print(f"{len(volts)} usable voltages, {int(allowed.sum())} admissible (dead-zone margin {args.dead_zone_margin} V); "
          f"read noise {args.read_noise_db:g} dB ({args.read_noise_model}): sigma {sigma.min():.4g}-{sigma.max():.4g}")

    # per-frame statistics: the train split chooses, the val split only reports
    n_train = len(dataset_train) if not args.max_frames else min(args.max_frames, len(dataset_train))
    stats = {'train': collect(dataset_train, n_train, R, mean, std, mu, band_std, args)}
    if not args.no_val:
        n_val = len(dataset_val) if not args.max_frames else min(args.max_frames, len(dataset_val))
        stats['val'] = collect(dataset_val, n_val, R, mean, std, mu, band_std, args)
    print(f"statistics of {len(stats['train'][3])} train" + (f" + {len(stats['val'][3])} val" if 'val' in stats else '') + f" frames in {time.time() - t0:.0f}s")

    # greedy selection on the train frames
    d, P, raw_d2, _ = stats['train']
    chosen, objective = greedy_select(d, P, noise_var, volts, args.k, allowed, args.min_spacing, args.ridge)
    chosen_volts = volts[chosen]
    print(f"greedy order: {np.round(chosen_volts, 2).tolist()} (objective {np.round(objective, 4).tolist()})")

    # baselines on the same frames: session B's gate top-10 and the detector's own uniform selection of k voltages
    uniform_axis = select_filter_channels(dataset_train.R_aligned_all, dataset_train.voltages, num_filters=args.k, mode='uniform')
    axis_to_cand = {int(a): j for j, a in enumerate(dataset_train.selected_indices)}  # voltage-axis index -> candidate column
    baselines = {'session_B': [int(np.abs(volts - v).argmin()) for v in SESSION_B_VOLTAGES[:args.k]],
                 'uniform': [axis_to_cand[int(a)] for a in uniform_axis], 'all': list(range(len(volts)))}
    report = {}
    for split in stats:
        report[split] = {'greedy_prefix': {k: report_set(stats[split][:3], chosen[:k], noise_var, args.ridge, args.read_noise_db) for k in range(1, args.k + 1)},
                         'baselines': {name: report_set(stats[split][:3], idx, noise_var, args.ridge, args.read_noise_db) for name, idx in baselines.items()}}
        rows = [('greedy', report[split]['greedy_prefix'][args.k])] + [(n, report[split]['baselines'][n]) for n in baselines]
        print(f"{split}: fraction of the raw bands' separability kept (median), " + ' / '.join(rows[0][1].keys()))
        for name, r in rows:
            print(f"  {name:10s} " + ' / '.join(f"{v['median']:.3f}" for v in r.values()))

    # write the result
    out = dict(k=args.k, read_noise_db=args.read_noise_db, read_noise_model=args.read_noise_model,
               noise_sigma=np.round(sigma, 6).tolist() if args.read_noise_model != 'floor' else float(sigma[0]),
               ridge=args.ridge, min_spacing=args.min_spacing, dead_zone_margin=args.dead_zone_margin, erode=args.erode, ring=list(args.ring),
               obj_samples=args.obj_samples, ring_samples=args.ring_samples, seed=args.seed,
               n_frames={s: len(v[3]) for s, v in stats.items()}, frames={s: v[3] for s, v in stats.items()},
               voltages=[round(float(v), 2) for v in chosen_volts], candidate_indices=chosen,
               axis_indices=[int(dataset_train.selected_indices[j]) for j in chosen], objective=[round(float(o), 6) for o in objective],
               baseline_voltages={name: [round(float(volts[j]), 2) for j in idx] for name, idx in baselines.items() if name != 'all'},
               report=report, filter_voltages_arg=' '.join(f'{v:.2f}' for v in chosen_volts))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=1)
    print(f"wrote {args.out} in {time.time() - t0:.0f}s; detector flags: --filter-select manual --filter-voltages {out['filter_voltages_arg']} "
          f"--pca-channels {args.k} --read_noise_db {args.read_noise_db:g}" + (f" --read_noise_model {args.read_noise_model}" if args.read_noise_model != 'floor' else ''))
    return out


if __name__ == '__main__':
    main(get_args_parser().parse_args())
