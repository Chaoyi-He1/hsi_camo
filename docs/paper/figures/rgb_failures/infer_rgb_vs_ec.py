'''
Full-frame level-(c) probability maps of the strict RGB chain (rgb_A boxes + seg_sam2unet_rgb_s0, pseudo-RGB segmenter) and the EC
chain (sel10g_clean_A boxes + seg_sam2unet_ec10_s0) on chosen test images, computed exactly as main_seg_eval.evaluate_frame does
(make_item -> predict -> paste_back, overlaps by maximum, no ROI = empty map), checked against the evaluation's stored rows
(results/seg_rgbdet/<run>/per_image.pkl). Writes <out>/<image>.npz (uint8 prob x 255 per chain, GT mask) for the figure script.
    CUDA_VISIBLE_DEVICES=1 python infer_rgb_vs_ec.py --images 196 301 ...
'''
import os
import sys
import time
import pickle
import argparse
import numpy as np
import torch
import torch.utils.data as Dataset

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../..'))   # the repository root
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
os.chdir(REPO)
import main_seg_eval as E
from data_loader.boxes import boxes_from_mask
from data_loader.roi_crops import match_rois
from train_eval.train_eval_seg import paste_back
from train_eval.seg_metrics import image_scores

RUNS = {'rgb': 'seg_sam2unet_rgb_s0', 'ec': 'seg_sam2unet_ec10_s0'}


def get_args_parser():
    parser = argparse.ArgumentParser('RGB vs EC chain maps', add_help=False)
    parser.add_argument('--images', type=str, nargs='+', required=True, help='test image names')
    parser.add_argument('--out', type=str, default=os.path.join(REPO, 'results', 'paper_figures', 'rgb_failures', 'maps'), help='output folder of the per-image npz files')
    return parser


@torch.no_grad()
def main(args):
    t0 = time.time()
    # 1. the evaluation's arguments: the rgb arm takes rgb_A's boxes (the strict RGB chain), the ec10 arm its own detector's
    eargs = E.get_args_parser().parse_args(['--runs', *RUNS.values(), '--rgb_roi_run', 'rgb_A', '--out_dir', os.path.join(REPO, 'results', 'paper_figures', 'rgb_failures', 'scratch_out')])
    device = torch.device('cuda')
    det_cache = {}
    specs = {k: E.load_run(eargs, run, det_cache) for k, run in RUNS.items()}
    assert specs['rgb']['det']['det_run'] == 'rgb_A' and specs['ec']['det']['det_run'] == 'sel10g_clean_A', {k: s['det']['det_run'] for k, s in specs.items()}
    canvas = specs['ec']['canvas']
    assert specs['rgb']['canvas'] == canvas
    fronts = E.check_runs(list(specs.values()), device)
    for s in specs.values():
        s['front'] = fronts[(s['arm'], s['det']['key'])]
        s['model'] = E.build_run_model(s, s['front'], device)
        print(f"{s['name']}: arm {s['arm']}, epoch {s['epoch']}, boxes of {s['det']['det_run']}", flush=True)
    # 2. the frames (the cube is read once; every arm's front end works on it)
    dataset = specs['ec']['det']['dataset']
    subset = Dataset.Subset(dataset, [dataset.img_name.index(n) for n in args.images])
    loader = torch.utils.data.DataLoader(subset, batch_size=1, shuffle=False, num_workers=2, collate_fn=E.frame_collate_fn)
    stored = {k: {kk: r for kk, r in zip(*[pickle.load(open(f'results/seg_rgbdet/{run}/per_image.pkl', 'rb'))['full_det'][x] for x in ('keys', 'rows')])}
              for k, run in RUNS.items()}
    os.makedirs(args.out, exist_ok=True)
    worst = 0.0
    for img, gt, name in loader:
        img, gt = img.numpy(), gt.numpy()                                       # [C, H, W] fp16, [H, W] bool
        H, W = gt.shape
        _, labels, ids = boxes_from_mask(gt, min_area=eargs.min_area, return_labels=True)
        maps = {}
        for k, s in specs.items():
            entry = s['det']['rois'][name]
            rois = np.asarray(entry['rois'], dtype=np.float32).reshape(-1, 5)
            boxes = np.asarray(entry['boxes'], dtype=np.float32).reshape(-1, 5)
            match = match_rois(rois[:, :4], labels, ids, min_cover=eargs.min_cover) if len(rois) else np.zeros(0, dtype=int)
            items = [E.make_item(img, gt, r[:4], b[:4], canvas, name, 'det' if mt >= 0 else 'fp') for r, b, mt in zip(rois, boxes, match)]
            full = np.zeros((H, W), dtype=np.float32)
            for (item, _), p in zip(items, E.predict(s['model'], s['front'], [it for it, _ in items], device, eargs.batch_size, eargs.amp)):
                full = paste_back(p, item[4], full)
            sc = image_scores(full, gt)
            d = max(abs(sc['IoU'] - stored[k][name]['IoU']), abs(sc['S'] - stored[k][name]['S']))
            worst = max(worst, d)
            maps[k] = (np.clip(full * 255 + 0.5, 0, 255).astype(np.uint8), sc)
        np.savez_compressed(os.path.join(args.out, f'{name}.npz'), rgb=maps['rgb'][0], ec=maps['ec'][0], gt=gt.astype(np.uint8),
                            rgb_iou=maps['rgb'][1]['IoU'], ec_iou=maps['ec'][1]['IoU'], rgb_s=maps['rgb'][1]['S'], ec_s=maps['ec'][1]['S'])
        print(f"image {name}: IoU RGB {maps['rgb'][1]['IoU']:.3f} EC {maps['ec'][1]['IoU']:.3f} | S RGB {maps['rgb'][1]['S']:.3f} EC {maps['ec'][1]['S']:.3f} "
              f"| {time.time() - t0:.0f}s", flush=True)
    print(f"max |difference| to the stored evaluation rows: {worst:.2e}" + ("" if worst < 1e-3 else "   <-- does not reproduce the evaluation"))


if __name__ == '__main__':
    main(get_args_parser().parse_args())
