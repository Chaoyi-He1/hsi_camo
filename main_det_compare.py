"""
Compare finished Stage-1 detector runs on FIXED checkpoints (spec §12). model_best is picked by an AP-blind rule, so the
runs are compared on their model_<epoch> checkpoints (default 59 69 79 89 99): every checkpoint is evaluated under the
read-noise condition it was trained with ('trained': the FilterBank's own reproducible eval-mode noise, so the numbers
match the run's validation lines) and on clean readings ('clean'), and the difference between two runs, averaged over
the checkpoints, gets a paired bootstrap over frames. One pass over the frames serves every (run, checkpoint, condition),
so each 0.55 GB frame is read from the cache once. A run's input is its own: a cube run reads the cached cube, an --rgb_images run
(the RGB baseline, rgb_A) the dataset's RGB/<id>.jpg of the same frame (rgb_images from its checkpoint's args), both read in lockstep
over the same ids, so one comparison can hold both kinds.
  CUDA_VISIBLE_DEVICES=0 python main_det_compare.py --runs sel10g_A sel10b_A sel10u_A --out_dir results/det/compare_sel10
Writes <out_dir>/compare.json (per-checkpoint summaries, run means, pairwise bootstrap CIs) and per_image.pkl (per-frame
stats, re-poolable with pool()).
"""
import os
if "RANK" not in os.environ and "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import time
import pickle
import argparse
import itertools
from functools import partial
import numpy as np
import torch

import main_det
from data_loader.boxes import det_collate_fn
from data_loader.my_dataset import HyperCOD_data
from models.ec_yolo import decode_predictions
from train_eval.box_metrics import BoxMetrics

KEYS = ['ap50', 'recall50', 'matched_iou', 'recall50_op', 'coverage_recall99_roi_op', 'dets_per_image_op']


def get_args_parser():
    parser = argparse.ArgumentParser('Compare detector runs on fixed checkpoints', parents=[main_det.get_args_parser()], add_help=False)
    parser.add_argument('--runs', type=str, nargs='+', required=True, help='run names: <weights_dir>/<run>/model_<epoch>')
    parser.add_argument('--weights_dir', type=str, default='weights', help='directory holding the run folders')
    parser.add_argument('--ckpt_epochs', type=int, nargs='+', default=[59, 69, 79, 89, 99], help='checkpoint epochs compared (averaged per run)')
    parser.add_argument('--splits', type=str, nargs='+', default=['test'], choices=['test', 'val'], help='frames to evaluate on; the first split is compared')
    parser.add_argument('--conditions', type=str, nargs='+', default=['trained', 'clean'], choices=['trained', 'clean'],
                        help='trained: the read noise the run was trained with (= clean for a noise-free run); clean: no read noise')
    parser.add_argument('--n_boot', type=int, default=2000, help='paired frame-bootstrap resamples')
    parser.add_argument('--out_dir', type=str, default='results/det/compare', help='where compare.json and per_image.pkl go')
    parser.set_defaults(wandb=False)
    return parser


def metric_kwargs(args):
    return dict(roi_margin=args.roi_margin, roi_min=args.roi_min, min_area=args.min_area, roi_conf=args.roi_conf, roi_topk=args.roi_topk)


def image_stats(mask, dets, kw):
    '''Per-image BoxMetrics state (records, confidences, counts), so a split can be re-pooled frame by frame for the bootstrap.'''
    bm = BoxMetrics(**kw)
    bm.update(mask, dets)
    return dict(records=bm.records, records_op=bm.records_op, pred_conf=bm.pred_conf, pred_tp=bm.pred_tp, n_dets=bm.n_dets, n_dets_op=bm.n_dets_op)


def pool(stats, kw):
    '''Split summary of a list of per-image stats: identical to one BoxMetrics updated with those images in order.'''
    bm = BoxMetrics(**kw)
    for s in stats:
        bm.n_images += 1; bm.n_dets += s['n_dets']; bm.n_dets_op += s['n_dets_op']
        bm.records += s['records']; bm.records_op += s['records_op']; bm.pred_conf += s['pred_conf']; bm.pred_tp += s['pred_tp']
    return bm.summary()


def bootstrap(runs_a, runs_b, kw, n=2000, seed=0):
    '''
    Paired frame bootstrap of mean(metric over runs_a) - mean(metric over runs_b); runs_* are lists of per-image stat
    lists over the same frames in the same order (the checkpoints of one run are averaged).
    '''
    m = len(runs_a[0])
    rng = np.random.default_rng(seed)
    point = {k: float(np.mean([pool(r, kw)[k] for r in runs_a]) - np.mean([pool(r, kw)[k] for r in runs_b])) for k in KEYS}
    draws = {k: [] for k in KEYS}
    for _ in range(n):
        idx = rng.integers(0, m, m)  # [m] frames drawn with replacement, the same for both sides
        sa = [pool([r[i] for i in idx], kw) for r in runs_a]
        sb = [pool([r[i] for i in idx], kw) for r in runs_b]
        for k in KEYS:
            va = [s[k] for s in sa if not np.isnan(s[k])]; vb = [s[k] for s in sb if not np.isnan(s[k])]
            draws[k].append(np.mean(va) - np.mean(vb) if va and vb else np.nan)
    return {k: dict(diff=point[k], lo=float(np.nanpercentile(draws[k], 2.5)), hi=float(np.nanpercentile(draws[k], 97.5)),
                    p_gt0=float(np.nanmean(np.asarray(draws[k]) > 0))) for k in KEYS}


def load_checkpoint_model(args, run, epoch, device):
    '''The run's model at model_<epoch>, rebuilt from the checkpoint's own args (filter selection, PCA, read noise), in eval mode.'''
    path = os.path.join(args.weights_dir, run, f'model_{epoch}')
    assert os.path.exists(path), f"{path} not found"
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    a = argparse.Namespace(**{**vars(args), **ckpt['args']})
    a.pretrained, a.resume, a.eval = 'none', path, True
    main_det.load_cfg(a)
    # filter_bank_tensors() is split-independent (the band statistics come from the train split's stats file), so one
    # dataset built with the run's own filter settings is enough to rebuild its FilterBank (as in main_det_rois)
    dataset = HyperCOD_data(split='test', **main_det.dataset_kwargs(a))
    model = main_det.build_model(a, dataset, ckpt).to(device).eval()
    return model, a, ckpt


@torch.no_grad()
def main(args):
    main_det.load_cfg(args)
    assert not args.rgb_images, "main_det_compare takes each run's input (cube or RGB frame) from its own checkpoint's rgb_images; do not pass --rgb_images"
    device = torch.device(args.device if args.device == 'cpu' or torch.cuda.is_available() else 'cpu')
    kw = metric_kwargs(args)
    os.makedirs(args.out_dir, exist_ok=True)

    # the models: one per (run, checkpoint), each with its own FilterBank; the frames are shared
    models = {}
    for run in args.runs:
        for e in args.ckpt_epochs:
            model, a, ckpt = load_checkpoint_model(args, run, e, device)
            noise_db = float(getattr(a, 'read_noise_db', 0.0) or 0.0)
            # a noise-free run has no 'trained' noise: its 'trained' condition is its clean evaluation
            conds = {c: ('clean' if c == 'trained' and noise_db == 0 else c) for c in args.conditions}
            rgb = bool(getattr(a, 'rgb_images', False))                                 # the run's input: its RGB frame, or the cube
            models[(run, e)] = dict(model=model, conds=conds, noise_db=noise_db, rgb=rgb)
            fb = model.filter_bank
            readings = ('RGB image' if rgb else 'raw bands' if getattr(a, 'raw_bands', False) else f"manual {a.filter_voltages}" if a.filter_select == 'manual'
                        else f"{a.filter_select} {a.num_filters}" if a.filter_select in ('uniform', 'osp') else 'all voltages')
            print(f"{run}/model_{e}: epoch {ckpt['epoch']}, {fb.n_readings} readings ({readings}) -> {fb.n_channels} channels, "
                  f"read noise {noise_db:g} dB", flush=True)

    # one pass over the frames per split; kinds = the inputs the models need (False: the cube, True: the RGB frame), one loader each,
    # read in lockstep (same ids, same order, same GT)
    kinds = sorted({m['rgb'] for m in models.values()})
    datasets = {k: main_det.build_datasets(argparse.Namespace(**{**vars(args), 'rgb_images': k})) for k in kinds}   # (train, val, test)
    per_image, results = {}, {}
    for split in args.splits:
        loaders = {k: torch.utils.data.DataLoader(datasets[k][2] if split == 'test' else datasets[k][1], batch_size=1, shuffle=False,
                                                  num_workers=args.num_workers, pin_memory=device.type == 'cuda',
                                                  collate_fn=partial(det_collate_fn, min_area=args.min_area)) for k in kinds}
        loader = loaders[kinds[0]]
        for m in models.values():
            m['model'].eval()  # restarts every bank's eval-noise sequence: the same draws as the run's own validation pass
        stats, names, t0 = {}, [], time.time()
        for i, batches in enumerate(zip(*loaders.values())):
            batches = dict(zip(kinds, batches))
            batch = batches[kinds[0]]
            assert all(b['names'] == batch['names'] for b in batches.values()), f"frames out of step: {[b['names'] for b in batches.values()]}"
            imgs = {k: b['img'].to(device, non_blocking=True) for k, b in batches.items()}  # [1, 133, H, W] fp16 cube, [1, 3, H, W] fp16 RGB frame
            mask, name = batch['masks'][0], batch['names'][0]
            names.append(name)
            H, W = imgs[kinds[0]].shape[-2:]
            for (run, e), m in models.items():
                for cond in sorted(set(m['conds'].values()), reverse=True):  # 'trained' before 'clean': one noise draw per frame, as in training
                    m['model'].filter_bank.noise_scale = 1.0 if cond == 'trained' else 0.0
                    with torch.autocast(device.type, enabled=device.type == 'cuda'):
                        out = m['model'](imgs[m['rgb']])
                    d = decode_predictions(out, conf_thres=args.conf_thres, iou_thres=args.iou_thres, max_det=args.max_det,
                                           end2end=getattr(m['model'].yolo, 'end2end', None))[0].copy()
                    d[:, [0, 2]] = d[:, [0, 2]].clip(0, W); d[:, [1, 3]] = d[:, [1, 3]].clip(0, H)
                    stats.setdefault((run, e, cond), []).append(image_stats(mask, d, kw))
            if i % 10 == 0 or i == len(loader) - 1:
                print(f"{split} {i + 1}/{len(loader)} frames, {time.time() - t0:.0f}s", flush=True)
        for (run, e), m in models.items():
            for c, actual in m['conds'].items():
                stats[(run, e, c)] = stats[(run, e, actual)]  # alias 'trained' of a noise-free run to its clean stats
        per_image[split] = dict(names=names, stats={f"{run}/model_{e}|{c}": v for (run, e, c), v in stats.items()})
        results[split] = {key: pool(v, kw) for key, v in per_image[split]['stats'].items()}

    # run means over the checkpoints, and paired comparisons on the first split
    split = args.splits[0]
    st = per_image[split]['stats']
    many = lambda run, cond: [st[f"{run}/model_{e}|{cond}"] for e in args.ckpt_epochs]
    means = {f"{run}|{c}": {k: float(np.mean([results[split][f"{run}/model_{e}|{c}"][k] for e in args.ckpt_epochs])) for k in KEYS}
             for run in args.runs for c in args.conditions}
    comps = {}
    for c in args.conditions:
        for ra, rb in itertools.combinations(args.runs, 2):
            comps[f"{ra} - {rb} | {c}"] = bootstrap(many(ra, c), many(rb, c), kw, args.n_boot)
    if 'trained' in args.conditions and 'clean' in args.conditions:
        for run in args.runs:
            if models[(run, args.ckpt_epochs[0])]['noise_db'] > 0:
                comps[f"{run} | trained - clean"] = bootstrap(many(run, 'trained'), many(run, 'clean'), kw, args.n_boot)

    # write and print
    with open(os.path.join(args.out_dir, 'per_image.pkl'), 'wb') as f:
        pickle.dump(per_image, f)
    with open(os.path.join(args.out_dir, 'compare.json'), 'w') as f:
        json.dump(dict(runs=args.runs, ckpt_epochs=args.ckpt_epochs, conditions=args.conditions, metric_kwargs=kw, results=results,
                       means={split: means}, comparisons={split: comps}), f, indent=1)
    for s in results:
        print(f"\n== {s}: " + ' | '.join(KEYS))
        for key, r in results[s].items():
            print(f"{key:40s} " + ' '.join(f"{r[k]:.3f}" for k in KEYS))
    print(f"\n== {split}, mean over model_{args.ckpt_epochs}: " + ' | '.join(KEYS))
    for key, r in means.items():
        print(f"{key:40s} " + ' '.join(f"{r[k]:.3f}" for k in KEYS))
    print(f"\n== {split}, paired frame bootstrap ({args.n_boot} resamples), diff [95% CI]")
    for key, v in comps.items():
        print(f"{key:40s} " + '; '.join(f"{k} {v[k]['diff']:+.3f} [{v[k]['lo']:+.3f}, {v[k]['hi']:+.3f}]" for k in ('ap50', 'recall50', 'recall50_op', 'coverage_recall99_roi_op')))
    print(f"\nwrote {args.out_dir}/compare.json")
    return results, comps


if __name__ == '__main__':
    main(get_args_parser().parse_args())
