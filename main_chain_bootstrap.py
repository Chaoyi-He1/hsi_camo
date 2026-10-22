'''
Paired frame bootstrap of one end-to-end chain against another at level (c) (full_det): the segmenters of chain A (e.g. the final
pipeline, results/seg_final: trained on and scored with the final detector's boxes) against those of chain B (e.g. the reference,
results/seg: the ec10 segmenters with sel10g_clean_A's boxes), per model, seeds paired (s0-s0, s1-s1, s2-s2). The statistic is
the mean over seeds of the per-image metric; the 70 test images are resampled with replacement (default 2000 times); 95 %
percentile CI. Also both chains' means +- std over seeds.
    python main_chain_bootstrap.py --a_dir results/seg_final --a_suffix ec10f --b_dir results/seg --b_suffix ec10
Writes <a_dir>/chain_vs_reference.json (read by main_paper_package.py) and prints a markdown table.
'''
import argparse
import json
import os
import pickle

import numpy as np

MODELS = ['sam2unet', 'sam2box', 'zoomnext']
KEYS = ['S', 'IoU', 'Fw', 'MAE', 'E_adp', 'F_adp']


def get_args_parser():
    parser = argparse.ArgumentParser('Paired bootstrap of two end-to-end chains', add_help=False)
    parser.add_argument('--a_dir', type=str, default='results/seg_final', help='main_seg_eval output of chain A (the final pipeline)')
    parser.add_argument('--a_suffix', type=str, default='ec10f', help="run-name arm suffix of chain A: seg_<model>_<suffix>_s<seed>")
    parser.add_argument('--b_dir', type=str, default='results/seg', help='main_seg_eval output of chain B (the reference)')
    parser.add_argument('--b_suffix', type=str, default='ec10', help='run-name arm suffix of chain B')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2], help='seeds paired between the chains')
    parser.add_argument('--level', type=str, default='full_det', help='evaluation level of per_image.pkl')
    parser.add_argument('--n_boot', type=int, default=2000, help='bootstrap resamples of the test images')
    parser.add_argument('--seed', type=int, default=0, help='bootstrap RNG seed')
    parser.add_argument('--out', type=str, default='', help='output json, default <a_dir>/chain_vs_reference.json')
    return parser


def per_image(run_dir, level):
    '''{image: {metric: value}} of one run at one level (every test image has a row, images without a box score an empty map).'''
    p = pickle.load(open(os.path.join(run_dir, 'per_image.pkl'), 'rb'))[level]
    return {(k if isinstance(k, str) else k[0]): r for k, r in zip(p['keys'], p['rows'])}


def metric_matrix(d, suffix, model, seeds, frames, key, level):
    '''[n_seeds, n_frames] of one metric; an image without a row counts 0.'''
    out = np.zeros((len(seeds), len(frames)))
    for i, s in enumerate(seeds):
        rows = per_image(os.path.join(d, f'seg_{model}_{suffix}_s{s}'), level)
        for j, f in enumerate(frames):
            out[i, j] = rows[f][key] if f in rows else 0.0
    return out


def main(args):
    # 1. the test images: the union of the rows of chain A's first run (every evaluation covers the same 70 images)
    first = os.path.join(args.a_dir, f'seg_{MODELS[0]}_{args.a_suffix}_s{args.seeds[0]}')
    frames = sorted(per_image(first, args.level), key=lambda s: (len(s), s))
    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, len(frames), size=(args.n_boot, len(frames)))
    # 2. per model and metric: seed means, paired per-image difference, bootstrap CI
    result = {}
    print(f'| model | metric | B {args.b_suffix} (mean ± std over seeds) | A {args.a_suffix} | A − B [95 % CI] |')
    print('|---|---|---|---|---|')
    for m in MODELS:
        result[m] = {}
        for k in KEYS:
            A = metric_matrix(args.a_dir, args.a_suffix, m, args.seeds, frames, k, args.level)
            B = metric_matrix(args.b_dir, args.b_suffix, m, args.seeds, frames, k, args.level)
            a_runs, b_runs = A.mean(1), B.mean(1)
            diff_img = A.mean(0) - B.mean(0)
            boots = diff_img[idx].mean(1)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            result[m][k] = dict(reference_mean=float(b_runs.mean()), reference_std=float(b_runs.std(ddof=1)) if len(args.seeds) > 1 else 0.0,
                                final_mean=float(a_runs.mean()), final_std=float(a_runs.std(ddof=1)) if len(args.seeds) > 1 else 0.0,
                                diff=float(diff_img.mean()), lo=float(lo), hi=float(hi), p_gt0=float((boots > 0).mean()), n_boot=args.n_boot, n_frames=len(frames))
            sig = '**' if lo > 0 or hi < 0 else ''
            print(f"| {m} | {k} | {b_runs.mean():.3f} ± {result[m][k]['reference_std']:.3f} | {a_runs.mean():.3f} ± {result[m][k]['final_std']:.3f} | {sig}{diff_img.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]{sig} |")
    # 3. write
    out = args.out or os.path.join(args.a_dir, 'chain_vs_reference.json')
    json.dump(dict(a_dir=args.a_dir, b_dir=args.b_dir, a_suffix=args.a_suffix, b_suffix=args.b_suffix, level=args.level, n_boot=args.n_boot, seeds=args.seeds, result=result),
              open(out, 'w'), indent=1)
    print(f'-> {out}')


if __name__ == '__main__':
    main(get_args_parser().parse_args())
