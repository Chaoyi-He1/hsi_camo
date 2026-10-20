'''
Fuse the test ROI exports of several detectors into one export (Stage-2 question: how much end-to-end recall the boxes cost).
    python main_fuse_rois.py --inputs results/det/rois_raw133_A_test.json results/det/rois_sel10g_clean_A_test.json \
        --name fuse_raw_ec10 --out_dir results/det
Writes <out_dir>/rois_<name>_<split>.json in the format of main_det_rois.py (per frame: 'rois' [x1, y1, x2, y2, conf] grown
ROIs, 'boxes' the pre-expansion boxes, 'gt_boxes'), so main_seg_eval.py --roi_run <name> scores any segmenter with the fused
boxes. Fusion = union of the boxes of all inputs; a box whose pre-expansion IoU with an already kept box exceeds --iou_dedup
is dropped (the higher-confidence one is kept; confidences of different detectors are compared as they are), then at most
--max_rois boxes per frame are kept by confidence (0 = keep all). Frames are the union of the inputs' frames; gt_boxes are
taken from the first input that has the frame.
'''
import argparse
import json
import os

import numpy as np


def get_args_parser():
    parser = argparse.ArgumentParser('Fuse detector ROI exports', add_help=False)
    parser.add_argument('--inputs', type=str, nargs='+', required=True, help='ROI export files of main_det_rois.py (same split)')
    parser.add_argument('--name', type=str, required=True, help='name of the fused export: rois_<name>_<split>.json')
    parser.add_argument('--split', type=str, default='test', help='split suffix of the output file')
    parser.add_argument('--out_dir', type=str, default='results/det', help='where the fused export goes')
    parser.add_argument('--iou_dedup', type=float, default=0.7, help='a box with IoU above this with a kept box (pre-expansion) is a duplicate; 1.0 keeps everything')
    parser.add_argument('--max_rois', type=int, default=0, help='boxes kept per frame after fusion, by confidence; 0 keeps all')
    return parser


def box_iou(a, b):
    '''IoU of two xyxy boxes.'''
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def fuse_frame(entries, iou_dedup=0.7, max_rois=0):
    '''
    entries: list of per-detector frame dicts {'rois', 'boxes', 'gt_boxes'} (rois[i] and boxes[i] carry the same conf).
    Returns the fused dict: the union of (roi, box) pairs sorted by confidence, duplicates by pre-expansion IoU removed,
    capped at max_rois when > 0.
    '''
    pairs = []
    for e in entries:
        assert len(e['rois']) == len(e['boxes']), f"{len(e['rois'])} rois but {len(e['boxes'])} boxes"
        pairs += [(list(map(float, r)), list(map(float, b))) for r, b in zip(e['rois'], e['boxes'])]
    pairs.sort(key=lambda p: -p[1][4])                                    # highest confidence first
    kept = []
    for roi, box in pairs:
        if any(box_iou(box[:4], k[1][:4]) > iou_dedup for k in kept):
            continue
        kept.append((roi, box))
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
        fused[name] = fuse_frame(entries, args.iou_dedup, args.max_rois)
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
    print(f"{args.name}: {len(exports)} exports, {n_in} boxes in -> {n_out} out over {len(frames)} frames, {no_roi} frames without any box, "
          f"{n_cov}/{n_gt} GT boxes fully inside an ROI ({n_cov / max(n_gt, 1):.3f}) -> {out}")


if __name__ == '__main__':
    main(get_args_parser().parse_args())
