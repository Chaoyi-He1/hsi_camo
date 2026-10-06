"""
Stage 2: end-to-end test evaluation of the segmentation runs (spec §5.6, §8). For every run (weights/<run>/<ckpt>) and the
optional zero-shot SAM2.1 control, on the 70 test frames:
  (a) roi_oracle   ROI level with oracle ROIs: every GT object's box grown by the export rule (x 1.5, >= 256 px), metrics on
                   the ROI crop at native size (target = the GT union inside the ROI)
  (b) full_oracle  the masks of (a) pasted back into the 1680 x 1240 frame (overlapping ROIs merged by maximum)
  (c) full_det     the whole sensor chain: the arm's own detector ROIs (results/det/rois_<det_run>_test.json, exported by
                   main_det_rois.py) pasted back; a frame without any ROI is an empty mask (a miss). The detector's
                   false-positive ROIs (< 1 % of every object mask) give the false-mask rate. (c) is also scored with the
                   HyperCOD paper's protocol (SegMetrics(minmax=True): per-image min-max + uint8, Table 2 columns MAE,
                   mean E, S, adaptive F)
The frames are read once per pass (HyperCOD_data: fp16 cache, O_DIRECT full-frame read, the loader's p99 scaling); the runs
of one pass (--runs_per_pass) keep their models on the GPU, so 29 SAM2-L sized runs need 5 passes instead of 29. Test items
are built with the crop cache's geometry (data_loader.roi_crops: pixel_box, place_on_canvas, box_to_canvas, rasterise_box),
so a test ROI gives exactly the tensors HyperCOD_roi gives for the same ROI. Every row carries its frame, so the paired
bootstrap resamples frames. Before the first pass every run's front end is built (and checked against its detector's
filter bank) and every checkpoint is loaded into its model on the CPU, so a bad run fails in seconds, not after hours.
  CUDA_VISIBLE_DEVICES=0 python main_seg_eval.py --runs seg_sam2unet_raw_s0 seg_sam2unet_ec10_s0 --zero_shot
  CUDA_VISIBLE_DEVICES=0 python main_seg_eval.py --runs seg_sam2unet_raw_s0 --ckpt model_last --out_dir results/seg_last
Writes <out_dir>/<run>/eval_roi_oracle.json, eval_full_oracle.json, eval_full_det.json, per_image.pkl and
<out_dir>/compare.json: mean +- std over seeds per (model, arm), and paired bootstrap CIs
(train_eval.seg_metrics.bootstrap_seg) for the same model across arms and the same arm across models.
per_image.pkl = {level: {'keys', 'rows', 'fp_rows'}}: 'rows' are the scored images (kind 'obj', in 'keys' order: the
paired bootstrap input), 'fp_rows' the detector's false-positive ROIs (kind 'fp', full_det only; each detector has its own,
so they never enter the pairing); train_eval.seg_metrics.pool(rows + fp_rows) gives the level's summary, false-mask
rate included.
"""
import os
if "RANK" not in os.environ and "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import time
import pickle
import argparse
import itertools
import numpy as np
import torch
import torch.utils.data as Dataset

import main_det
import main_seg
from main_det_rois import load_detector
from data_loader.my_dataset import HyperCOD_data
from data_loader.boxes import boxes_from_mask, expand_box
from data_loader.roi_crops import (match_rois, pixel_box, place_on_canvas, canvas_to_roi, box_to_canvas, rasterise_box,
                                   seg_collate_fn)
from models.seg_models import build_front_end, build_seg_model
from train_eval.train_eval_seg import seg_inputs, paste_back
from train_eval.seg_metrics import SegMetrics, bootstrap_seg

LEVELS = ['roi_oracle', 'full_oracle', 'full_det']        # spec §8 (a), (b), (c): written per run and bootstrapped
PAPER_KEYS = ['MAE', 'E_mean', 'S', 'F_adp']              # HyperCOD Table 2 columns; its "E" is taken as mean E (inferred
                                                          # from SAM2-UNet's eval.py, which prints exactly this set)
BOOT_KEYS = ['S', 'E_mean', 'E_max', 'Fw', 'F_adp', 'MAE', 'IoU']
ARM_ORDER = ['raw', 'ec10', 'ec24', 'rgb']
ZS_NAME = 'zs_sam2box_rgb'                                # the zero-shot control's run name (and results folder)
ZS_MODEL = 'sam2box_zs'                                   # ... and its model label in compare.json


def get_args_parser():
    parser = argparse.ArgumentParser('Stage-2 end-to-end test evaluation', add_help=False)
    # runs
    parser.add_argument('--runs', type=str, nargs='*', default=[], help='segmentation run names: <weights_dir>/<run>/<ckpt>')
    parser.add_argument('--ckpt', type=str, default='model_best', choices=['model_best', 'model_last'], help='checkpoint evaluated per run')
    parser.add_argument('--weights_dir', type=str, default='weights', help='directory holding the run folders')
    parser.add_argument('--zero_shot', action='store_true',
                        help='also evaluate the untrained control: SAM2.1 + box prompt on the pseudo-RGB render (arm rgb, P = I, q = 0)')
    parser.add_argument('--zero_shot_det_ckpt', type=str, default='weights/raw133_A/model_best',
                        help='detector whose test ROIs and front end the zero-shot control uses')
    parser.add_argument('--sam2_ckpt', type=str, default='', help="zero-shot control's SAM2.1 checkpoint; empty = main_seg's default")
    parser.add_argument('--canvas', type=int, default=512, help='canvas of the zero-shot control when no trained run fixes it')
    # protocol (spec §8)
    parser.add_argument('--roi_dir', type=str, default='results/det', help='where main_det_rois.py wrote rois_<det_run>_test.json')
    parser.add_argument('--roi_margin', type=float, default=1.5, help='oracle ROI = GT box grown by this factor (the export rule)')
    parser.add_argument('--roi_min', type=float, default=256, help='... and at least this many px per side')
    parser.add_argument('--min_area', type=int, default=100, help='GT components below this many px are JPEG specks, not objects')
    parser.add_argument('--min_cover', type=float, default=0.01, help='a detector ROI covering less of every object mask is a false positive')
    parser.add_argument('--size_edges', type=int, nargs=2, default=[2000, 20000], help='object-size buckets by GT area (px)')
    parser.add_argument('--n_boot', type=int, default=2000, help='paired bootstrap resamples')
    parser.add_argument('--seed', type=int, default=0, help='bootstrap seed')
    # data and compute
    parser.add_argument('--data_path', type=str, default='/data2/chaoyi/HyperCOD/Raw data', help='HyperCOD root')
    parser.add_argument('--cache_dir', type=str, default='', help='fp16 frame cache, default <data_path>/cache_fp16')
    parser.add_argument('--split_file', type=str, default='', help='val ids json handed to the detector loader')
    parser.add_argument('--device', type=str, default='cuda', help="'cuda', 'cuda:N' or 'cpu'")
    parser.add_argument('--batch_size', type=int, default=8, help='ROI items per forward pass')
    parser.add_argument('--runs_per_pass', type=int, default=6, help='runs whose models share one pass over the frames (GPU memory)')
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True, help='bf16 autocast for the model (CUDA only)')
    parser.add_argument('--num_workers', type=int, default=2, help='frame-reading DataLoader workers')
    parser.add_argument('--limit', type=int, default=0, help='only the first N test frames (smoke runs)')
    parser.add_argument('--out_dir', type=str, default='results/seg', help='per-run folders and compare.json go here')
    return parser


def frame_collate_fn(batch):
    '''
    batch_size 1 over HyperCOD_data -> (img [C, H, W] fp16 p99-scaled tensor, gt [H, W] bool tensor, name). Tensors, so a
    DataLoader worker hands the 0.55 GB frame over through shared memory instead of pickling a numpy array.
    '''
    img, gt, name = batch[0]
    return torch.from_numpy(img), torch.from_numpy(gt[0] > 0.5), name


def roi_window(roi, H, W):
    '''
    Integer pixel window [x1, y1, x2, y2] of a float ROI: data_loader.roi_crops.pixel_box (floor x1/y1, ceil x2/y2, clipped
    to the frame; the rounding of box_metrics.mask_coverage and of the crop cache), so the window always contains the ROI.
    '''
    x1, y1, x2, y2 = pixel_box(roi, H, W)
    assert x2 > x1 and y2 > y1, f"empty ROI {[float(v) for v in roi[:4]]} in a {H} x {W} frame"
    return [x1, y1, x2, y2]


def make_item(img, gt, roi, box, canvas, frame, source):
    '''
    One test item, built like HyperCOD_roi's deterministic validation items but cut from the full frame (the test frames
    are not in the crop cache): the ROI window at native resolution placed on the canvas (scale 1, no augmentation) and
    the box channel = rasterise_box(box_to_canvas(box)), i.e. HyperCOD_roi's own geometry (box_to_canvas clips the box to
    the placed region, so the box channel is 0 on the padding).
    img: [C, H, W] fp16 p99-scaled numpy; gt: [H, W] bool; roi, box: frame px xyxy; source: 'gt' | 'det' | 'fp'
    Returns (the seg_collate_fn tuple (img, mask, box_map, valid, meta), GT crop [h, w] bool).
    '''
    H, W = gt.shape
    x1, y1, x2, y2 = roi_window(roi, H, W)
    crop = img[:, y1:y2, x1:x2]                                                        # [C, h, w] view
    gt_crop = gt[y1:y2, x1:x2]                                                        # [h, w] bool
    h, w = gt_crop.shape
    out, valid, (oy, ox), s = place_on_canvas(crop, canvas=canvas, scale=1.0)         # [C, c, c], [c, c] bool
    mask = place_on_canvas(gt_crop[None].astype(np.float32), canvas=canvas, scale=1.0)[0]   # [1, c, c]
    # the pre-expansion box, clipped to the window first (detector boxes are only clipped to the frame)
    bx = np.clip(np.asarray(box[:4], dtype=np.float64), [x1, y1, x1, y1], [x2, y2, x2, y2])
    box_canvas = box_to_canvas(bx, (x1, y1, x2, y2), (oy, ox), (h, w), s, canvas)      # canvas px, clipped to the placed region
    box_map = rasterise_box(box_canvas, canvas)                                        # [c, c] float32
    meta = {'frame': frame, 'roi': [x1, y1, x2, y2], 'box': [float(v) for v in bx], 'box_canvas': [float(v) for v in box_canvas],
            'source': source, 'offset': (int(oy), int(ox)), 's': float(s), 'roi_hw': (h, w), 'obj_area': int(gt_crop.sum())}
    item = (np.asarray(out, dtype=np.float16), np.asarray(mask, dtype=np.float32).reshape(1, canvas, canvas), box_map[None],
            valid.astype(np.float32)[None], meta)
    return item, gt_crop


@torch.no_grad()
def predict(model, front_end, items, device, batch_size=8, amp=True):
    '''Sigmoid probabilities [n, c, c] float32 of the model's main output (index 0) for a list of items.'''
    probs = []
    for i in range(0, len(items), batch_size):
        batch = seg_collate_fn(items[i:i + batch_size])
        x = seg_inputs(front_end, batch, device)                                     # [B, n_in + 1, c, c], front end as in training
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp and device.type == 'cuda'):
            logits = model(x, batch['box_xyxy'].to(device))[0]                       # [B, 1, c, c]
        probs.append(torch.sigmoid(logits.float())[:, 0].cpu().numpy())             # [B, c, c]
    return np.concatenate(probs, axis=0) if probs else np.zeros((0, 1, 1), dtype=np.float32)


def det_cli_args(args):
    '''main_det-style namespace for load_detector: data and device keys from this CLI, the model keys come from the checkpoint.'''
    return main_det.get_args_parser().parse_args(['--data-path', args.data_path, '--cache-dir', args.cache_dir, '--split-file', args.split_file,
                                                   '--device', 'cpu', '--num_workers', str(args.num_workers), '--min-area', str(args.min_area),
                                                   '--no-wandb'])


def load_detector_info(args, det_ckpt, cache):
    '''
    Everything a run needs from its detector, once per detector checkpoint: its args (the front end is rebuilt from them),
    its state dict (the rebuilt front end is checked against its filter bank, spec §9), its run name, its exported test
    ROIs and a test HyperCOD_data built with its own filter settings (build_front_end and the frame reads use it). The
    detector network itself is dropped: the ROIs come from main_det_rois.py's export.
    '''
    key = os.path.abspath(det_ckpt)
    if key not in cache:
        model, det_args, det_run = load_detector(det_ckpt, det_cli_args(args), torch.device('cpu'))
        del model
        det_state = torch.load(det_ckpt, map_location='cpu', weights_only=False)['model']
        det_args.data_path, det_args.cache_dir = args.data_path, args.cache_dir   # the frames of this machine, whatever the ckpt says
        dataset = HyperCOD_data(split='test', **main_det.dataset_kwargs(det_args))
        path = os.path.join(args.roi_dir, f'rois_{det_run}_test.json')
        assert os.path.exists(path), f"{path} missing: export it with python main_det_rois.py --resume {det_ckpt} --split test (bash_files/launch_rois_all.sh)"
        with open(path) as f:
            rois = json.load(f)
        missing = [n for n in dataset.img_name if n not in rois]
        assert not missing, f"{path} lacks test frames {missing[:5]} ({len(missing)} in all): not a test-split export of {det_run}?"
        cache[key] = dict(key=key, det_args=det_args, det_state=det_state, det_run=det_run, rois=rois, dataset=dataset, roi_file=path)
    return cache[key]


def load_run(args, run, det_cache):
    '''A trained run: its main_seg args and epoch (the weights are loaded later, per pass) and its detector.'''
    path = os.path.join(args.weights_dir, run, args.ckpt)
    assert os.path.exists(path), f"{path} not found"
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    a, epoch = argparse.Namespace(**ckpt['args']), ckpt.get('epoch')
    del ckpt                                                                            # ~0.9 GB for a SAM2-L run
    assert a.arm in ARM_ORDER, f"{path}: unknown arm {a.arm!r}"
    return dict(name=run, path=path, args=a, zero_shot=False, model_label=a.seg_model, arm=a.arm, seed=int(a.seed),
                canvas=int(a.canvas), epoch=epoch, det=load_detector_info(args, a.det_ckpt, det_cache))


def zero_shot_spec(args, canvas, det_cache):
    '''The zero-shot control: SAM2.1 + box prompt on the pseudo-RGB render, built by main_seg's own parser and cfg, never trained.'''
    argv = ['--seg_model', 'sam2box', '--arm', 'rgb', '--det_ckpt', args.zero_shot_det_ckpt, '--canvas', str(canvas),
            '--seed', '0', '--name', ZS_NAME]
    if args.sam2_ckpt:
        argv += ['--sam2_ckpt', args.sam2_ckpt]
    a = main_seg.get_args_parser().parse_args(argv)
    main_seg.load_cfg(a)
    return dict(name=ZS_NAME, path=None, args=a, zero_shot=True, model_label=ZS_MODEL, arm='rgb', seed=0, canvas=canvas,
                epoch=None, det=load_detector_info(args, args.zero_shot_det_ckpt, det_cache))


def build_run_model(spec, front_end, device):
    '''
    The run's segmentation model in eval mode. Trained run: built with its checkpoint's args, then its weights loaded; a
    checkpoint may leave out frozen parameters (rebuilt from the pretrained file), nothing else. Zero-shot: the stem folded
    with P = I, q = 0 (the rgb front end already is the ImageNet-normalised pseudo-RGB image), so it is the pretrained model.
    '''
    n = front_end.n_out
    if spec['zero_shot']:
        assert n == 3, f"the zero-shot control needs the 3-channel rgb front end, got {n} channels"
        return build_seg_model(spec['args'], n, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)).to(device).eval()
    ckpt = torch.load(spec['path'], map_location='cpu', weights_only=False)
    # (P, q) only initialise the folded stem, whose trained weights the state dict overwrites; zeros when not saved
    P = np.zeros((3, n), dtype=np.float32) if ckpt.get('P') is None else np.asarray(ckpt['P'], dtype=np.float32)
    q = np.zeros(3, dtype=np.float32) if ckpt.get('q') is None else np.asarray(ckpt['q'], dtype=np.float32)
    assert P.shape == (3, n), f"{spec['path']}: P is {P.shape}, the {spec['arm']} front end gives {n} channels"
    model = build_seg_model(spec['args'], n, P, q)
    missing, unexpected = model.load_state_dict(ckpt['model'], strict=False)
    frozen = {k for k, p in model.named_parameters() if not p.requires_grad}
    assert not unexpected and set(missing) <= frozen, \
        f"{spec['path']}: unexpected keys {unexpected[:5]}, missing non-frozen keys {sorted(set(missing) - frozen)[:5]}"
    del ckpt
    return model.to(device).eval()


def new_state(size_edges):
    '''
    Per-run accumulators: one SegMetrics per level (full_det_paper with the HyperCOD paper's min-max protocol), the
    per_image row index and pairing key of every scored image.
    '''
    metrics = {lvl: SegMetrics(size_edges=size_edges) for lvl in LEVELS}
    metrics['full_det_paper'] = SegMetrics(size_edges=size_edges, minmax=True)
    levels = list(metrics)
    return dict(metrics=metrics, rows={lvl: [] for lvl in levels}, keys={lvl: [] for lvl in levels},
                n_frames=0, n_frames_no_roi=0, n_det_rois=0, n_fp_rois=0)


def record(state, level, pred, gt, key):
    '''
    SegMetrics.update plus the bookkeeping that pairs this image across runs (the row index skips update_fp rows). The row
    carries its frame (key = (frame, k) at ROI level, the frame id at full-frame levels), so bootstrap_seg resamples frames.
    '''
    m = state['metrics'][level]
    state['rows'][level].append(len(m.per_image))
    state['keys'][level].append(key)
    m.update(pred, gt, frame=str(key[0]) if isinstance(key, tuple) else str(key), key=key)


def evaluate_frame(spec, oracle, det, gt, name, device, args):
    '''Levels (a), (b), (c) of one frame for one run. oracle / det items: lists of (item, GT crop); det also the matches.'''
    st, (H, W) = spec['state'], gt.shape
    model, front = spec['model'], spec['front']
    # (a) ROI level and (b) full frame, oracle ROIs
    probs = predict(model, front, [it for it, _ in oracle], device, args.batch_size, args.amp)       # [K, c, c]
    full = np.zeros((H, W), dtype=np.float32)
    for k, ((item, gt_crop), p) in enumerate(zip(oracle, probs)):
        meta = item[4]
        record(st, 'roi_oracle', canvas_to_roi(p, meta['offset'], meta['s'], meta['roi_hw']), gt_crop, (name, k))
        full = paste_back(p, meta, full)
    record(st, 'full_oracle', full, gt, name)
    # (c) the arm's own detector ROIs, merged by maximum; no ROI = the empty mask
    items, match = det
    probs = predict(model, front, [it for it, _ in items], device, args.batch_size, args.amp)
    full = np.zeros((H, W), dtype=np.float32)
    for (item, _), p, obj in zip(items, probs, match):
        meta = item[4]
        full = paste_back(p, meta, full)
        if obj < 0:                                       # false positive: S/E undefined, only the false-mask rate
            st['metrics']['full_det'].update_fp(canvas_to_roi(p, meta['offset'], meta['s'], meta['roi_hw']), frame=name)
            st['n_fp_rois'] += 1
    record(st, 'full_det', full, gt, name)
    record(st, 'full_det_paper', full, gt, name)          # the same map, scored with SegMetrics(minmax=True)
    st['n_frames'] += 1; st['n_det_rois'] += len(items); st['n_frames_no_roi'] += int(len(items) == 0)


def to_json(o):
    '''json.dump default: numpy scalars and arrays.'''
    return o.tolist() if hasattr(o, 'tolist') else float(o)


def write_run(args, spec):
    '''The run's three eval_*.json files and per_image.pkl; keeps the summaries and paired rows on spec for compare().'''
    st, d = spec['state'], os.path.join(args.out_dir, spec['name'])
    os.makedirs(d, exist_ok=True)
    summ = {lvl: m.summary() for lvl, m in st['metrics'].items()}
    info = dict(run=spec['name'], seg_model=spec['model_label'], arm=spec['arm'], seed=spec['seed'], ckpt=args.ckpt if not spec['zero_shot'] else 'zero_shot',
                epoch=spec['epoch'], det_run=spec['det']['det_run'], roi_file=spec['det']['roi_file'], canvas=spec['canvas'])
    files = {'roi_oracle': dict(info, level='roi_oracle', roi_margin=args.roi_margin, roi_min=args.roi_min, summary=summ['roi_oracle']),
             'full_oracle': dict(info, level='full_oracle', summary=summ['full_oracle']),
             'full_det': dict(info, level='full_det', summary=summ['full_det'], paper={k: summ['full_det_paper'][k] for k in PAPER_KEYS},
                              n_frames=st['n_frames'], n_frames_no_roi=st['n_frames_no_roi'], n_det_rois=st['n_det_rois'], n_fp_rois=st['n_fp_rois'])}
    for lvl, obj in files.items():
        with open(os.path.join(d, f'eval_{lvl}.json'), 'w') as f:
            json.dump(obj, f, indent=1, default=to_json)
    spec['summary'] = summ
    spec['keys'] = st['keys']
    # the paired rows (bootstrap input) and, apart, the false-positive ROI rows (update_fp): rows + fp_rows re-pool to summ
    spec['per_image'] = {lvl: [st['metrics'][lvl].per_image[i] for i in st['rows'][lvl]] for lvl in st['metrics']}
    fp_rows = {lvl: [r for r in st['metrics'][lvl].per_image if r['kind'] == 'fp'] for lvl in st['metrics']}
    with open(os.path.join(d, 'per_image.pkl'), 'wb') as f:
        pickle.dump({lvl: dict(keys=spec['keys'][lvl], rows=spec['per_image'][lvl], fp_rows=fp_rows[lvl]) for lvl in spec['per_image']}, f)
    s, c, p = summ['roi_oracle'], summ['full_det'], files['full_det']['paper']
    print(f"{spec['name']}: roi_oracle S {s['S']:.3f} Fw {s['Fw']:.3f} IoU {s['IoU']:.3f} | full_det S {c['S']:.3f} Fw {c['Fw']:.3f} "
          f"IoU {c['IoU']:.3f}, false-mask rate {c['fp_false_mask_rate']:.3f} | paper MAE {p['MAE']:.4f} E {p['E_mean']:.3f} "
          f"S {p['S']:.3f} adpF {p['F_adp']:.3f} -> {d}", flush=True)


def seed_stats(summaries):
    '''mean, std (ddof 1; 0 for a single seed) and n over the seeds of one (model, arm) for every numeric summary key.'''
    out = {}
    for k, v in summaries[0].items():
        if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
            vals = np.asarray([float(s[k]) for s in summaries], dtype=np.float64)
            out[k] = dict(mean=float(np.mean(vals)), std=float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, n=len(vals))
    return out


def compare(args, specs):
    '''compare.json: seed statistics per (model, arm) and the paired bootstrap CIs (same model across arms, same arm across models).'''
    groups = {}
    for s in specs:
        groups.setdefault((s['model_label'], s['arm']), []).append(s)
    for g in groups.values():
        g.sort(key=lambda s: s['seed'])
    # pairing: every run scored the same images in the same order (oracle items do not depend on the arm; frames are shared)
    for lvl in LEVELS:
        ref = specs[0]['keys'][lvl]
        for s in specs[1:]:
            assert s['keys'][lvl] == ref, f"{lvl}: {s['name']} scored other images than {specs[0]['name']}; cannot pair them"
    stats = {f"{m}|{a}": dict(runs=[s['name'] for s in g], seeds=[s['seed'] for s in g],
                              **{lvl: seed_stats([s['summary'][lvl] for s in g]) for lvl in LEVELS + ['full_det_paper']})
             for (m, a), g in groups.items()}
    models = sorted({m for m, _ in groups})
    arms = [a for a in ARM_ORDER if any(a == ga for _, ga in groups)]
    pairs = []
    for m in models:
        present = [a for a in arms if (m, a) in groups]
        pairs += [(f"{m}: {a1} - {a2}", (m, a1), (m, a2)) for a1, a2 in itertools.combinations(present, 2)]
    for a in arms:
        present = [m for m in models if (m, a) in groups]
        pairs += [(f"{a}: {m1} - {m2}", (m1, a), (m2, a)) for m1, m2 in itertools.combinations(present, 2)]
    comps = {lvl: {} for lvl in LEVELS}
    for lvl in LEVELS:
        for label, ga, gb in pairs:
            comps[lvl][label] = bootstrap_seg([s['per_image'][lvl] for s in groups[ga]], [s['per_image'][lvl] for s in groups[gb]],
                                              BOOT_KEYS, n=args.n_boot, seed=args.seed)
    out = dict(ckpt=args.ckpt, n_boot=args.n_boot, keys=BOOT_KEYS, paper_keys=PAPER_KEYS,
               runs={s['name']: dict(seg_model=s['model_label'], arm=s['arm'], seed=s['seed'], epoch=s['epoch'], det_run=s['det']['det_run'])
                     for s in specs},
               groups=stats, comparisons=comps)
    path = os.path.join(args.out_dir, 'compare.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=1, default=to_json)
    # print: mean +- std over seeds, then the CIs
    show = ['S', 'E_mean', 'Fw', 'MAE', 'IoU']
    for lvl in LEVELS:
        print(f"\n== {lvl}, mean +- std over seeds: " + ' | '.join(show))
        for g, st in stats.items():
            print(f"{g:24s} n={len(st['runs'])} " + ' '.join(f"{st[lvl][k]['mean']:.3f}+-{st[lvl][k]['std']:.3f}" for k in show))
        print(f"== {lvl}, paired frame bootstrap ({args.n_boot} resamples), diff [95% CI]")
        for label, v in comps[lvl].items():
            print(f"{label:32s} " + '; '.join(f"{k} {v[k]['diff']:+.3f} [{v[k]['lo']:+.3f}, {v[k]['hi']:+.3f}]" for k in ('S', 'Fw', 'IoU')))
    print(f"\nwrote {path}")
    return out


def check_runs(specs, device):
    '''
    Up-front checks, before the first frame pass: the front end of every (arm, detector) is built, raw / ec10 / ec24 with
    the detector checkpoint's state (build_front_end checks the arm and the filter bank, spec §9), and every run's model is
    built on the CPU with its checkpoint loaded (build_run_model: no unexpected keys, only frozen ones missing), then freed.
    A broken run thus fails in seconds instead of after the passes of the runs before it. Returns the front ends on device,
    {(arm, detector key): front end}, shared by the runs of each pair.
    '''
    fronts = {}
    for s in specs:
        key = (s['arm'], s['det']['key'])
        if key not in fronts:
            det_state = None if s['arm'] == 'rgb' else s['det']['det_state']
            fronts[key] = build_front_end(s['arm'], s['det']['det_args'], s['det']['dataset'], det_state=det_state).to(device).eval()
        model = build_run_model(s, fronts[key], torch.device('cpu'))
        del model
        print(f"{s['name']}: front end ({fronts[key].n_out} channels) and {'pretrained model' if s['zero_shot'] else s['path']} ok", flush=True)
    return fronts


@torch.no_grad()
def main(args):
    device = torch.device(args.device if args.device == 'cpu' or torch.cuda.is_available() else 'cpu')
    assert args.runs or args.zero_shot, "nothing to evaluate: pass --runs and/or --zero_shot"
    assert len(set(args.runs)) == len(args.runs), f"duplicate run names in {args.runs}"
    os.makedirs(args.out_dir, exist_ok=True)

    # the runs (args only; the weights are loaded per pass) and their detectors
    det_cache = {}
    specs = [load_run(args, run, det_cache) for run in args.runs]
    canvases = {s['canvas'] for s in specs}
    assert len(canvases) <= 1, f"runs with different canvases {sorted(canvases)}: the oracle items are shared, evaluate them separately"
    canvas = canvases.pop() if canvases else args.canvas
    if args.zero_shot:
        assert ZS_NAME not in args.runs, f"{ZS_NAME} is the zero-shot control's name"
        specs.append(zero_shot_spec(args, canvas, det_cache))
    for s in specs:
        print(f"{s['name']}: {s['model_label']}, arm {s['arm']}, seed {s['seed']}, epoch {s['epoch']}, detector {s['det']['det_run']} "
              f"({s['det']['roi_file']})", flush=True)

    # check every front end (spec §9 filter bank) and every run's checkpoint before the first pass
    fronts = check_runs(specs, device)                                # one front end per (arm, detector), shared by its runs

    # the test frames, read through HyperCOD_data exactly as the detectors' ROI export read them
    dataset = specs[0]['det']['dataset']
    if args.limit:
        dataset = Dataset.Subset(dataset, list(range(min(args.limit, len(dataset)))))

    # evaluate: one pass over the test frames per runs_per_pass runs, their models on the device
    for p0 in range(0, len(specs), args.runs_per_pass):
        group = specs[p0:p0 + args.runs_per_pass]
        for s in group:
            s['front'] = fronts[(s['arm'], s['det']['key'])]
            s['model'] = build_run_model(s, s['front'], device)
            s['state'] = new_state(tuple(args.size_edges))
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
                                             collate_fn=frame_collate_fn)
        t0 = time.time()
        for i, (img, gt, name) in enumerate(loader):
            img, gt = img.numpy(), gt.numpy()                         # [C, H, W] fp16 p99-scaled, [H, W] bool
            H, W = gt.shape
            gt_boxes, labels, ids = boxes_from_mask(gt, min_area=args.min_area, return_labels=True)
            oracle = [make_item(img, gt, expand_box(b, args.roi_margin, args.roi_min, H, W), b, canvas, name, 'gt') for b in gt_boxes]
            det_items = {}                                            # one item list per detector, shared by its runs
            for s in group:
                key = s['det']['key']
                if key in det_items:
                    continue
                entry = s['det']['rois'][name]
                rois = np.asarray(entry['rois'], dtype=np.float32).reshape(-1, 5)     # [R, 5] grown ROI + conf
                boxes = np.asarray(entry['boxes'], dtype=np.float32).reshape(-1, 5)   # [R, 5] pre-expansion box + conf
                assert len(rois) == len(boxes), f"{s['det']['roi_file']} frame {name}: {len(rois)} rois but {len(boxes)} boxes"
                match = match_rois(rois[:, :4], labels, ids, min_cover=args.min_cover) if len(rois) else np.zeros(0, dtype=int)
                det_items[key] = ([make_item(img, gt, r[:4], b[:4], canvas, name, 'det' if m >= 0 else 'fp')
                                   for r, b, m in zip(rois, boxes, match)], match)
            for s in group:
                evaluate_frame(s, oracle, det_items[s['det']['key']], gt, name, device, args)
            if i % 10 == 0 or i == len(loader) - 1:
                print(f"pass {p0 // args.runs_per_pass + 1}: {i + 1}/{len(loader)} frames, {time.time() - t0:.0f}s", flush=True)
        # write the pass's runs: eval_*.json and per_image.pkl, then free their models
        for s in group:
            write_run(args, s)
            del s['model'], s['front'], s['state']
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # compare: mean +- std over seeds, paired frame bootstrap
    return compare(args, specs)


if __name__ == '__main__':
    main(get_args_parser().parse_args())
