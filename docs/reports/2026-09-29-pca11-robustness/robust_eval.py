"""
Inference-only robustness evaluation of the finished Stage-1 checkpoints (spec §12, "PCA-11 follow-up").

Every (model, condition) pair sees the same frames in one pass, so each 0.55 GB frame is read from the SATA cache once
and reused for every pair on the GPU. One BoxMetrics per image and pair is kept: pooling them reproduces
main_det.evaluate's split summary exactly, and the paired bootstrap over frames only re-pools the stored stats.

Conditions act inside the filter bank; a checkpoint's YOLO weights are shared by all of its conditions:
  clean        the model exactly as trained (its own FilterBank.forward)
  noise<dB>    i.i.d. Gaussian read noise per bias-voltage reading (per band for raw bands), sigma = that channel's
               mean absorbed signal (|R_V|^T mu, mu = training band mean in p99 units, i.e. auto-exposure) / 10^(dB/20);
               for the PCA model the 344 per-voltage noises go through its fixed 344 -> 11 map (correlated noise)
  voff<mV>     every bias voltage lands dv higher (hysteresis / drift): R(V + dv), linear within a dead-zone branch
  gain<pct>    fixed per-voltage gain error ~ N(0, pct %), one draw (seed 0)
  device1      measured Device-1 responses (R_Device1.mat, 0.05 V grid, interpolated, per-voltage least-squares gain)
               instead of the EC_filterV3 interpolant, with the nominal map and standardisation
  <above>r     the same perturbed device, with each channel's mean/std re-measured on it (recalibrated standardisation)
  drop<a>_<b>  PCA channels u_a..u_b (1-based) set to their training mean (0) inside the frame
Run from any checkout (the script imports the repo it lives in); the checkpoints are read from --weights-dir:
  CUDA_VISIBLE_DEVICES=0 python docs/reports/2026-09-29-pca11-robustness/robust_eval.py \
      --weights-dir /data/chaoyi_he/hsi_camo/.claude/worktrees/det/weights --out results/det/robustness
"""
import argparse
import json
import os
import pickle
import sys
import time
import zlib
from functools import partial
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
os.chdir(REPO)
from main_det import get_args_parser as main_det_parser, build_datasets, build_model, load_cfg    # noqa: E402
from data_loader.boxes import det_collate_fn                                                     # noqa: E402
from train_eval.box_metrics import BoxMetrics                                                    # noqa: E402
from models.ec_yolo import decode_predictions                                                    # noqa: E402

KEYS = ['ap50', 'recall50', 'matched_iou', 'recall50_op', 'coverage_recall99_roi_op', 'dets_per_image_op']
PERTURB = ['noise50', 'noise40', 'noise30', 'voff5', 'voff20', 'gain1', 'device1']


def get_args_parser():
    parser = argparse.ArgumentParser('Stage-1 robustness evaluation (inference only)', add_help=True)
    parser.add_argument('--weights-dir', default='weights', help='directory holding pca11_A/, raw133_A/, det_A/, det_B/')
    parser.add_argument('--device1', default='/data/chaoyi_he/HSI/Diffu/dataset/HASCID-Dataset/R_Device1.mat')
    parser.add_argument('--out', required=True, help='output directory (robust.json, per_image.pkl)')
    parser.add_argument('--splits', nargs='+', default=['test', 'val'])
    parser.add_argument('--n-boot', type=int, default=2000)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--smoke', action='store_true', help='2 frames per split and a reduced plan (pipeline check)')
    return parser


# ---------------------------------------------------------------- response perturbations (nominal R: [133, 344])
def r_at_volt(R, volts, dv):
    '''Responses at V + dv, linear interpolation within the dead-zone branch of the nominal voltage (extrapolates at ends).'''
    out = np.empty_like(R)
    lo, hi = volts <= 0.245, volts >= 0.315
    for j, v in enumerate(volts):
        br = lo if v <= 0.245 else hi
        vb, Rb = volts[br], R[:, br]
        k = int(np.clip(np.searchsorted(vb, v + dv) - 1, 0, len(vb) - 2))
        t = (v + dv - vb[k]) / (vb[k + 1] - vb[k])
        out[:, j] = (1 - t) * Rb[:, k] + t * Rb[:, k + 1]
    return out


def r_device1(R, volts, wl, path):
    '''Device-1 measurement on the band grid, interpolated in voltage within each branch, gain-calibrated per voltage.'''
    d1 = sio.loadmat(path)
    D1, D1v, D1w = d1['R'].astype(np.float64), d1['voltage'].ravel(), d1['wavelength'].ravel().astype(float)
    D1b = np.stack([np.interp(wl, D1w, D1[:, j], left=0, right=0) for j in range(D1.shape[1])], 1)   # [133, 71]
    D1i = np.empty_like(R)
    for br, gm in ((volts <= 0.245, D1v <= 0.25), (volts >= 0.315, D1v >= 0.30)):
        for b in range(R.shape[0]):
            D1i[b, br] = np.interp(volts[br], D1v[gm], D1b[b, gm])
    g = (D1i * R).sum(0) / (D1i * D1i).sum(0)
    return D1i * g


def pca_map(R, mean, std, band_cov, k):
    '''The 344 -> k map M of models/ec_yolo.pca_whitened_channels (y' = M^T y, before its mean'/std').'''
    Dinv = 1.0 / std
    C = (R * Dinv).T @ band_cov @ (R * Dinv)
    evals, evecs = np.linalg.eigh(C)
    order = np.argsort(evals)[::-1][:k]
    return Dinv[:, None] * evecs[:, order]                                        # [344, k]


# ---------------------------------------------------------------- perturbed filter bank
class PerturbedFB(nn.Module):
    '''
    FilterBank.forward with an optional replaced projection, replaced standardisation statistics, read noise before
    the standardisation (inside the frame only, not in the stride padding) and zeroed output channels.
    '''

    def __init__(self, fb, R_t=None, mean=None, std=None, noise_L=None, zero_idx=None, seed=0):
        super(PerturbedFB, self).__init__()
        dev = fb.R_t.device
        as_t = lambda a: None if a is None else torch.as_tensor(np.asarray(a), dtype=torch.float32, device=dev)
        self.fb, self.seed, self.zero_idx = fb, int(seed), list(zero_idx or [])
        self.R_t, self.L = as_t(R_t), as_t(noise_L)
        self.mean = fb.mean if mean is None else as_t(mean).view(1, -1, 1, 1)
        self.std = fb.std if std is None else as_t(std).view(1, -1, 1, 1)
        self.diag = noise_L is not None and bool(np.allclose(noise_L, np.diag(np.diag(noise_L))))
        self.frame_key, self.valid_hw = 0, None

    def forward(self, x):
        fb = self.fb
        R_t = fb.R_t if self.R_t is None else self.R_t
        y = torch.einsum('nc,bchw->bnhw', R_t.to(x.dtype), x)                     # [B, N, Hp, Wp], as in training
        H, W = self.valid_hw
        if self.L is not None:
            g = torch.Generator(device=y.device); g.manual_seed(self.seed * 1000003 + self.frame_key)
            B, N = y.shape[:2]
            if self.diag:
                sig = torch.diagonal(self.L).to(y.dtype).view(1, N, 1, 1)
                y[:, :, :H, :W] += torch.randn((B, N, H, W), generator=g, device=y.device, dtype=y.dtype) * sig
            else:
                y = y.float()
                z = torch.randn((B, N, H, W), generator=g, device=y.device, dtype=torch.float32)
                y[:, :, :H, :W] += torch.einsum('nm,bmhw->bnhw', self.L, z)
        y = (y - self.mean.to(y.dtype)) / self.std.to(y.dtype)
        y = y * fb.weights.to(y.dtype).view(1, -1, 1, 1)
        if self.zero_idx:
            y[:, self.zero_idx, :H, :W] = 0
        return y.to(x.dtype)


# ---------------------------------------------------------------- metrics
def image_stats(mask, dets, kw):
    bm = BoxMetrics(roi_margin=kw['roi_margin'], roi_min=kw['roi_min'], min_area=kw['min_area'], roi_conf=kw['roi_conf'], roi_topk=kw['roi_topk'])
    bm.update(mask, dets)
    return dict(records=bm.records, records_op=bm.records_op, pred_conf=bm.pred_conf, pred_tp=bm.pred_tp,
                n_dets=bm.n_dets, n_dets_op=bm.n_dets_op)


def pool(stats, kw):
    '''Split summary of a list of per-image stats: identical to one BoxMetrics updated with those images in order.'''
    bm = BoxMetrics(roi_margin=kw['roi_margin'], roi_min=kw['roi_min'], min_area=kw['min_area'], roi_conf=kw['roi_conf'], roi_topk=kw['roi_topk'])
    for s in stats:
        bm.n_images += 1; bm.n_dets += s['n_dets']; bm.n_dets_op += s['n_dets_op']
        bm.records += s['records']; bm.records_op += s['records_op']; bm.pred_conf += s['pred_conf']; bm.pred_tp += s['pred_tp']
    return bm.summary()


def bootstrap(runs_a, runs_b, kw, n=2000, seed=0):
    '''
    Paired frame bootstrap of mean(metric over runs_a) - mean(metric over runs_b); runs_* are lists of per-image stat
    lists over the same frames in the same order (several checkpoints of one model are averaged).
    '''
    m = len(runs_a[0]); rng = np.random.default_rng(seed)
    point = {k: np.mean([pool(r, kw)[k] for r in runs_a]) - np.mean([pool(r, kw)[k] for r in runs_b]) for k in KEYS}
    draws = {k: [] for k in KEYS}
    for _ in range(n):
        idx = rng.integers(0, m, m)
        sa = [pool([r[i] for i in idx], kw) for r in runs_a]; sb = [pool([r[i] for i in idx], kw) for r in runs_b]
        for k in KEYS:
            vals_a = [s[k] for s in sa if not np.isnan(s[k])]; vals_b = [s[k] for s in sb if not np.isnan(s[k])]
            draws[k].append(np.mean(vals_a) - np.mean(vals_b) if vals_a and vals_b else np.nan)
    return {k: dict(diff=float(point[k]), lo=float(np.nanpercentile(draws[k], 2.5)), hi=float(np.nanpercentile(draws[k], 97.5)),
                    p_gt0=float(np.nanmean(np.asarray(draws[k]) > 0))) for k in KEYS}


# ---------------------------------------------------------------- main
def main(args):
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device('cuda')
    W = args.weights_dir
    base = main_det_parser().parse_args([])
    load_cfg(base)
    kw = dict(conf_thres=base.conf_thres, iou_thres=base.iou_thres, max_det=base.max_det, roi_margin=base.roi_margin,
              roi_min=base.roi_min, min_area=base.min_area, roi_conf=base.roi_conf, roi_topk=base.roi_topk)
    ds_train, ds_val, ds_test = build_datasets(base)
    R, mean, std, volts = [np.asarray(a, np.float64) for a in ds_train.filter_bank_tensors()]
    mu, cov = ds_train._band_stats()
    wl = np.asarray(ds_train.wavelens, np.float64)
    sigma_unit = np.abs(R).T @ mu                                                  # [344] mean absorbed signal per voltage
    R_pert = {'voff5': r_at_volt(R, volts, 0.005), 'voff20': r_at_volt(R, volts, 0.020),
              'gain1': R * (1 + np.random.default_rng(0).normal(0, 0.01, R.shape[1])), 'device1': r_device1(R, volts, wl, args.device1)}
    rel = {k: float(np.median(np.linalg.norm(v - R, axis=0) / np.linalg.norm(R, axis=0))) for k, v in R_pert.items()}
    print('median column change of the perturbed responses:', {k: round(v, 4) for k, v in rel.items()}, flush=True)

    # ------------------------------------------------ what to evaluate: (tag, checkpoint, conditions)
    if args.smoke:
        plan = [('pca11', f'{W}/pca11_A/model_99', ['clean', 'noise40', 'voff5r', 'drop9_11']),
                ('raw133', f'{W}/raw133_A/model_99', ['clean', 'noise40']), ('detA', f'{W}/det_A/model_89', ['clean', 'voff5'])]
    else:
        plan = [('pca11', f'{W}/pca11_A/model_99', ['clean'] + PERTURB + ['voff5r', 'gain1r', 'device1r', 'drop9_11', 'drop6_11']),
                ('raw133', f'{W}/raw133_A/model_99', ['clean', 'noise50', 'noise40', 'noise30']),
                ('detA', f'{W}/det_A/model_89', ['clean'] + PERTURB),
                ('detB', f'{W}/det_B/model_best', ['clean', 'noise50', 'noise40', 'noise30', 'voff5', 'device1'])]
        for e in (59, 69, 79, 89):
            plan += [('pca11', f'{W}/pca11_A/model_{e}', ['clean']), ('raw133', f'{W}/raw133_A/model_{e}', ['clean'])]
        plan += [('pca11', f'{W}/pca11_A/model_best', ['clean']), ('raw133', f'{W}/raw133_A/model_best', ['clean']),
                 ('detA', f'{W}/det_A/model_99', ['clean'])]

    pairs = {}                                                                     # key -> (model, filter-bank module)
    for tag, path, conds in plan:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        a = argparse.Namespace(**{**vars(base), **ckpt['args']}); a.pretrained = 'none'; a.resume = path
        model = build_model(a, ds_train, ckpt).to(dev).eval()
        fb = model.filter_bank
        R_t_ck = fb.R_t.detach().double().cpu().numpy()                            # [N, n_bands]
        if getattr(a, 'raw_bands', False):
            kind, T = 'raw', None
        elif int(getattr(a, 'pca_channels', 0) or 0) > 0:
            kind, T = 'pca', pca_map(R, mean, std, cov, int(a.pca_channels))
            T *= np.sign(np.sum((R @ T) * R_t_ck.T, axis=0))                       # eigenvector signs as in the checkpoint
            err = np.abs(R @ T - R_t_ck.T).max() / np.abs(R_t_ck).max()
            assert err < 1e-4, f"{path}: recomputed PCA map does not reproduce the checkpoint projection (rel err {err:.2e})"
        else:
            idx = list(ckpt.get('selected_indices', range(R.shape[1])))
            kind, T = 'ec', np.eye(R.shape[1])[:, idx]
            err = np.abs(R @ T - R_t_ck.T).max() / np.abs(R_t_ck).max()
            assert err < 1e-5, f"{path}: nominal responses do not match the checkpoint projection (rel err {err:.2e})"
        name = f"{tag}@{os.path.basename(path)}(ep{int(ckpt['epoch'])})"
        print(f"loaded {name}: kind {kind}, {fb.n_channels} channels, gate {fb.weight_vector}", flush=True)
        for c in conds:
            if c == 'clean':
                mod = fb
            elif c.startswith('noise'):
                s = 10 ** (-float(c[5:]) / 20)
                if kind == 'raw':
                    L = np.diag(np.asarray(mu, np.float64) * s)
                else:
                    Sig = T.T @ np.diag((sigma_unit * s) ** 2) @ T
                    L = np.linalg.cholesky(Sig + 1e-18 * np.trace(Sig) * np.eye(len(Sig))) if kind == 'pca' else np.diag(np.sqrt(np.diag(Sig)))
                mod = PerturbedFB(fb, noise_L=L, seed=zlib.crc32(c.encode()) % 997)   # reproducible across runs
            elif c.startswith('drop'):
                lo_, hi_ = [int(v) for v in c[4:].split('_')]
                mod = PerturbedFB(fb, zero_idx=list(range(lo_ - 1, hi_)))
            else:
                assert kind != 'raw', f"{c} does not apply to raw bands"
                recal = c.endswith('r')
                A = R_pert[c[:-1] if recal else c] @ T                              # [n_bands, N] projection of the perturbed device
                if recal:                                                          # channel statistics re-measured on that device
                    mod = PerturbedFB(fb, R_t=A.T, mean=A.T @ mu, std=np.sqrt(np.diag(A.T @ cov @ A)))
                else:
                    mod = PerturbedFB(fb, R_t=A.T)
            pairs[f"{name}|{c}"] = (model, mod)
    print(f"{len(pairs)} (model, condition) pairs", flush=True)

    # ------------------------------------------------ one pass over the frames
    results, per_image = {}, {}
    for split in args.splits:
        ds = ds_test if split == 'test' else ds_val
        if args.smoke:
            ds = torch.utils.data.Subset(ds, [0, len(ds) - 1])
        loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True,
                                             collate_fn=partial(det_collate_fn, min_area=kw['min_area']))
        stats = {k: [] for k in pairs}; names = []; t0 = time.time()
        for i, batch in enumerate(loader):
            img = batch['img'].to(dev, non_blocking=True)                          # [1, 133, H, W] fp16
            mask, fname = batch['masks'][0], batch['names'][0]; names.append(fname)
            H, W_ = img.shape[-2:]
            with torch.no_grad(), torch.autocast('cuda'):
                for key, (model, mod) in pairs.items():
                    if isinstance(mod, PerturbedFB):
                        mod.frame_key, mod.valid_hw = int(fname), (H, W_)
                    img_p, _ = model.pad_to_stride(img)
                    out = model.yolo(mod(img_p))
                    d = decode_predictions(out, conf_thres=kw['conf_thres'], iou_thres=kw['iou_thres'], max_det=kw['max_det'],
                                           end2end=getattr(model.yolo, 'end2end', None))[0].copy()
                    d[:, [0, 2]] = d[:, [0, 2]].clip(0, W_); d[:, [1, 3]] = d[:, [1, 3]].clip(0, H)
                    stats[key].append(image_stats(mask, d, kw))
            if i % 10 == 0 or i == len(loader) - 1:
                print(f"{split} {i + 1}/{len(loader)} frames, {time.time() - t0:.0f}s", flush=True)
        per_image[split] = dict(names=names, stats=stats)
        results[split] = {k: pool(v, kw) for k, v in stats.items()}
    with open(os.path.join(args.out, 'per_image.pkl'), 'wb') as f:
        pickle.dump(per_image, f)

    # ------------------------------------------------ comparisons (paired over test frames)
    comps = {}
    if not args.smoke and 'test' in per_image:
        st = per_image['test']['stats']
        key = lambda tag, ck, c='clean': next(k for k in st if k.startswith(f"{tag}@{ck}(") and k.endswith(f"|{c}"))
        many = lambda tag, cks, c='clean': [st[key(tag, ck, c)] for ck in cks]
        cks = ['model_59', 'model_69', 'model_79', 'model_89', 'model_99']
        comps['pca11_vs_raw133_mean_ep59-99'] = bootstrap(many('pca11', cks), many('raw133', cks), kw, args.n_boot)
        comps['pca11_vs_raw133_ep99'] = bootstrap(many('pca11', ['model_99']), many('raw133', ['model_99']), kw, args.n_boot)
        comps['pca11_vs_raw133_model_best'] = bootstrap(many('pca11', ['model_best']), many('raw133', ['model_best']), kw, args.n_boot)
        comps['pca11_ep99_vs_detA_ep89'] = bootstrap(many('pca11', ['model_99']), many('detA', ['model_89']), kw, args.n_boot)
        for c in PERTURB + ['voff5r', 'gain1r', 'device1r', 'drop9_11', 'drop6_11']:
            comps[f'pca11_ep99_{c}_vs_clean'] = bootstrap(many('pca11', ['model_99'], c), many('pca11', ['model_99']), kw, args.n_boot)
        for c in PERTURB:
            comps[f'detA_ep89_{c}_vs_clean'] = bootstrap(many('detA', ['model_89'], c), many('detA', ['model_89']), kw, args.n_boot)
    json.dump(dict(eval_kw=kw, response_change_median=rel, results=results, comparisons=comps),
              open(os.path.join(args.out, 'robust.json'), 'w'), indent=1)

    for split in results:
        print(f"\n== {split}: " + ' | '.join(KEYS))
        for k, s in results[split].items():
            print(f"{k:55s} " + ' '.join(f"{s[m]:.3f}" for m in KEYS))
    for c, v in comps.items():
        print(f"\n{c}: " + '; '.join(f"{k} {d['diff']:+.3f} [{d['lo']:+.3f},{d['hi']:+.3f}]" for k, d in v.items() if k in ('ap50', 'recall50', 'recall50_op')))


if __name__ == '__main__':
    main(get_args_parser().parse_args())
