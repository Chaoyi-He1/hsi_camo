'''
Fuse detector ROI exports (Stage-2 question: how much end-to-end recall the boxes cost, and what cheaper boxes buy).
    python main_fuse_rois.py --inputs results/det/rois_raw133_A_test.json results/det/rois_sel10g_clean_A_test.json \
        --name fuse_raw_ec10 --out_dir results/det
    python main_fuse_rois.py --inputs results/det/rois_sel10g_clean_A_test.json --name ec10_union --merge union --iou_merge 0.5
Writes <out_dir>/rois_<name>_<split>.json in the format of main_det_rois.py (per frame: 'rois' [x1, y1, x2, y2, conf] grown
ROIs, 'boxes' the pre-expansion boxes, 'gt_boxes'), so main_seg_eval.py --roi_run <name> scores any segmenter with these boxes.
  Union of inputs   every box of every input (one input = the detector's own boxes, several = a multi-detector union); frames
                    are the union of the inputs' frames, gt_boxes from the first input that has the frame.
  --merge none      a box whose pre-expansion IoU with an already kept (higher-confidence) box exceeds --iou_dedup is dropped.
  --merge union     boxes with pairwise IoU > --iou_merge are clustered (greedy, by confidence) and each cluster becomes its
                    enclosing box: two half boxes of one object give one box that holds the whole object.
  --merge wbf       as union, but the cluster's box is the confidence-weighted mean of its boxes (weighted box fusion).
  A merged box keeps the cluster's highest confidence and is re-grown into its ROI with the export rule (--roi_margin,
  --roi_min, clipped to --frame_hw); unmerged boxes keep their exported ROI. Then at most --max_rois boxes per frame are
  kept by confidence (0 = keep all).
'''
import argparse
import json
import os

import numpy as np

from data_loader.boxes import expand_box


def get_args_parser():
    parser = argparse.ArgumentParser('Fuse detector ROI exports', add_help=False)
    parser.add_argument('--inputs', type=str, nargs='+', required=True, help='ROI export files of main_det_rois.py (same split)')
    parser.add_argument('--name', type=str, required=True, help='name of the fused export: rois_<name>_<split>.json')
    parser.add_argument('--split', type=str, default='test', help='split suffix of the output file')
    parser.add_argument('--out_dir', type=str, default='results/det', help='where the fused export goes')
    parser.add_argument('--merge', type=str, default='none', choices=['none', 'union', 'wbf'], help='how overlapping boxes are combined (see the module docstring)')
    parser.add_argument('--iou_merge', type=float, default=0.5, help='--merge union|wbf: boxes with IoU above this join one cluster')
    parser.add_argument('--iou_dedup', type=float, default=0.7, help='--merge none: a box with IoU above this with a kept box is a duplicate; 1.0 keeps everything')
    parser.add_argument('--max_rois', type=int, default=0, help='boxes kept per frame after fusion, by confidence; 0 keeps all')
    parser.add_argument('--roi_margin', type=float, default=1.5, help='ROI of a merged box = box grown by this factor about its centre (the export rule)')
    parser.add_argument('--roi_min', type=float, default=256, help='... and at least this many px per side')
    parser.add_argument('--frame_hw', type=int, nargs=2, default=[1680, 1240], help='frame height and width the ROIs are clipped to')
    return parser


def box_iou(a, b):
    '''IoU of two xyxy boxes.'''
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def merge_cluster(boxes, mode):
    '''One [x1, y1, x2, y2, conf] from a cluster of boxes: enclosing box (union) or confidence-weighted mean (wbf); conf = max.'''
    b = np.asarray(boxes, dtype=np.float64)                                # [n, 5]
    if mode == 'union':
        xyxy = [b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()]
    else:
        w = b[:, 4] / max(b[:, 4].sum(), 1e-9)
        xyxy = list((b[:, :4] * w[:, None]).sum(0))
    return [float(v) for v in xyxy] + [float(b[:, 4].max())]


def fuse_frame(entries, iou_dedup=0.7, max_rois=0, merge='none', iou_merge=0.5, roi_margin=1.5, roi_min=256, frame_hw=(1680, 1240)):
    '''
    entries: list of per-detector frame dicts {'rois', 'boxes', 'gt_boxes'} (rois[i] and boxes[i] carry the same conf).
    Returns the fused dict: the union of (roi, box) pairs sorted by confidence, duplicates removed (merge 'none') or
    overlapping boxes merged and re-grown into ROIs (merge 'union' | 'wbf'), capped at max_rois when > 0.
    '''
    pairs = []
    for e in entries:
        assert len(e['rois']) == len(e['boxes']), f"{len(e['rois'])} rois but {len(e['boxes'])} boxes"
        pairs += [(list(map(float, r)), list(map(float, b))) for r, b in zip(e['rois'], e['boxes'])]
    pairs.sort(key=lambda p: -p[1][4])                                    # highest confidence first
    kept = []
    if merge == 'none':
        for roi, box in pairs:
            if any(box_iou(box[:4], k[1][:4]) > iou_dedup for k in kept):
                continue
            kept.append((roi, box))
    else:
        H, W = frame_hw
        clusters = []                                                     # each: list of (roi, box), seeded by confidence order
        for roi, box in pairs:
            for c in clusters:
                if box_iou(box[:4], c[0][1][:4]) > iou_merge:             # join the first cluster whose seed box it overlaps
                    c.append((roi, box))
                    break
            else:
                clusters.append([(roi, box)])
        for c in clusters:
            if len(c) == 1:
                kept.append(c[0])
            else:
                box = merge_cluster([p[1] for p in c], merge)
                roi = [*expand_box(box[:4], roi_margin, roi_min, H, W).tolist(), box[4]]
                kept.append((roi, box))
        kept.sort(key=lambda p: -p[1][4])
    if max_rois > 0:
        kept = kept[:max_rois]
    gt = next((e['gt_boxes'] for e in entries if 'gt_boxes' in e), [])
    return {'rois': [k[0] for k in kept], 'boxes': [k[1] for k in kept], 'gt_boxes': gt}


def main(args):
    # 1. read every export; the frame set is their union
    exports = []
    for p in args.inputs:
        assert os.path.isfile(p), f"{p} not found"
        with open(p) as f:
            exports.append(json.load(f))
    frames = sorted(set().union(*[set(e) for e in exports]), key=lambda s: (len(s), s))
    # 2. fuse frame by frame
    fused, n_in, n_out = {}, 0, 0
    for name in frames:
        entries = [e[name] for e in exports if name in e]
        n_in += sum(len(e['rois']) for e in entries)
        fused[name] = fuse_frame(entries, args.iou_dedup, args.max_rois, args.merge, args.iou_merge, args.roi_margin, args.roi_min, tuple(args.frame_hw))
        n_out += len(fused[name]['rois'])
    # 3. write, and report the coverage the fused boxes reach (every GT box inside at least one ROI)
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f'rois_{args.name}_{args.split}.json')
    with open(out, 'w') as f:
        json.dump(fused, f)
    n_gt, n_cov, no_roi = 0, 0, 0
    for v in fused.values():
        no_roi += int(len(v['rois']) == 0)
        for g in v['gt_boxes']:
            n_gt += 1
            n_cov += int(any(r[0] <= g[0] and r[1] <= g[1] and r[2] >= g[2] and r[3] >= g[3] for r in v['rois']))
    print(f"{args.name} (merge {args.merge}): {len(exports)} exports, {n_in} boxes in -> {n_out} out over {len(frames)} frames, {no_roi} frames without any box, "
          f"{n_cov}/{n_gt} GT boxes fully inside an ROI ({n_cov / max(n_gt, 1):.3f}) -> {out}")


if __name__ == '__main__':
    main(get_args_parser().parse_args())
