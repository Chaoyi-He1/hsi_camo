'''
Paper figures of the images where the RGB pipeline fails and the EC pipeline succeeds (and the reverse cases), from the maps of
infer_rgb_vs_ec.py. One row per test image, six panels on one square window that holds the object and every box of both detectors:
  RGB image | ground truth | RGB pipeline: detection, segmentation | EC pipeline (10 EC readings): detection, segmentation
Detection panels: the camera image with the detector's boxes at the operating point (conf >= 0.02, <= 5 boxes); segmentation
panels: the full-frame level-(c) mask (probability > 0.5) as a binary map, with its IoU against the ground truth.
    python render_rgb_failures.py --images 196 301 ... --out fig_rgb_failures --tags "no detection" ...
Writes <out>.pdf and <out>.png (600 dpi); --panels also writes every panel as its own PNG into <out>_panels/.
'''
import argparse
import json
import os

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../..'))       # the repository root
HERE = os.path.dirname(os.path.abspath(__file__))
RGB_DIR = '/data2/chaoyi/HyperCOD/Raw data/test/RGB'
DET = {'rgb': 'rgb_A', 'ec': 'sel10g_clean_A'}
COL = {'rgb': '#F28A1C', 'ec': '#19A7E0'}              # box colours: RGB pipeline orange, EC pipeline blue
plt.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 7, 'pdf.fonttype': 42, 'ps.fonttype': 42})


def failure_mode(boxes, gt, seg_iou):
    '''What went wrong, judged on the detector's own boxes (not the grown ROIs): no box; no box on the object (every box on
    background); a box on the object but the segmenter masks (almost) nothing of it; otherwise the mask is poor.'''
    if not boxes:
        return 'no detection'
    ys, xs = np.nonzero(gt)
    on = [b for b in boxes if np.count_nonzero(gt[int(max(b[1], 0)):int(np.ceil(b[3])), int(max(b[0], 0)):int(np.ceil(b[2]))]) >= 0.25 * len(xs)]
    if not on:
        return 'box on background'
    return 'boxed, not segmented' if seg_iou < 0.2 else 'poor mask'


def get_args_parser():
    parser = argparse.ArgumentParser('RGB-failure figure', add_help=False)
    parser.add_argument('--images', type=str, nargs='+', required=True, help='test images, one row each, in order')
    parser.add_argument('--out', type=str, required=True, help='output path without extension')
    parser.add_argument('--maps', type=str, default=os.path.join(R, 'results', 'paper_figures', 'rgb_failures', 'maps'), help='folder of infer_rgb_vs_ec.py npz files')
    parser.add_argument('--width', type=float, default=7.2, help='figure width in inches (7.2 = a full-width Nature figure)')
    parser.add_argument('--panels', action='store_true', help='also write each panel as a PNG (no annotations) into <out>_panels/')
    parser.add_argument('--reverse', action='store_true', help='row tags describe the EC failure instead of the RGB one')
    return parser


def window(gt, boxes, H, W):
    '''A square crop holding the ground truth and every detector box: side = max(1.2 x the union, 2.4 x the GT box, 320 px).'''
    ys, xs = np.nonzero(gt)
    g = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
    allb = [g] + [b[:4] for b in boxes]
    x1, y1 = min(b[0] for b in allb), min(b[1] for b in allb)
    x2, y2 = max(b[2] for b in allb), max(b[3] for b in allb)
    side = int(min(max(1.2 * max(x2 - x1, y2 - y1), 2.4 * max(g[2] - g[0], g[3] - g[1]), 320), W, H))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    x0 = int(np.clip(round(cx - side / 2), 0, W - side)); y0 = int(np.clip(round(cy - side / 2), 0, H - side))
    return x0, y0, side


def main(args):
    rois = {k: json.load(open(f'{R}/results/det/rois_{run}_test.json')) for k, run in DET.items()}
    n = len(args.images)
    modes, panel_info = {}, {}
    label_w = 0.70                                                        # inches for the row labels
    pw = (args.width - label_w) / 6                                       # panel width = height
    head = 0.42
    fig = plt.figure(figsize=(args.width, head + n * pw))
    if args.panels:
        os.makedirs(args.out + '_panels', exist_ok=True)
    for i, name in enumerate(args.images):
        d = np.load(os.path.join(args.maps, f'{name}.npz'))
        gt = d['gt'] > 0
        H, W = gt.shape
        img = cv2.imread(os.path.join(RGB_DIR, f'{name}.jpg'))[:, :, ::-1]
        assert img.shape[:2] == (H, W), f"{name}: RGB {img.shape} vs GT {gt.shape}"
        boxes = {k: rois[k][name]['boxes'] for k in DET}
        x0, y0, s = window(gt, boxes['rgb'] + boxes['ec'], H, W)
        crop = lambda a: a[y0:y0 + s, x0:x0 + s]
        panels = [('image', crop(img), None), ('gt', crop(gt), None),
                  ('rgb_det', crop(img), 'rgb'), ('rgb_mask', crop(d['rgb'] >= 128), 'rgb'),
                  ('ec_det', crop(img), 'ec'), ('ec_mask', crop(d['ec'] >= 128), 'ec')]
        top = 1 - (head + (i + 1) * pw) / (head + n * pw)
        for j, (kind, a, chain) in enumerate(panels):
            ax = fig.add_axes([(label_w + j * pw) / args.width + 0.0015, top + 0.0015 * args.width / (head + n * pw),
                               pw / args.width - 0.003, pw / (head + n * pw) - 0.003 * args.width / (head + n * pw)])
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.4); sp.set_color('#808080')
            if a.dtype == bool:
                ax.imshow(a, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
            else:
                ax.imshow(a, interpolation='lanczos')
            ax.set_xlim(-0.5, s - 0.5); ax.set_ylim(s - 0.5, -0.5)
            if args.panels:
                p = (a.astype(np.uint8) * 255) if a.dtype == bool else a[:, :, ::-1].copy()     # BGR for OpenCV
                if kind.endswith('_det'):                                                    # the boxes drawn in, as in the figure
                    t = max(2, s // 200)
                    for b in boxes[chain]:
                        p1, p2 = (int(round(b[0] - x0)), int(round(b[1] - y0))), (int(round(b[2] - x0)), int(round(b[3] - y0)))
                        cv2.rectangle(p, p1, p2, (0, 0, 0), 2 * t, cv2.LINE_AA)
                        cv2.rectangle(p, p1, p2, tuple(int(COL[chain][q:q + 2], 16) for q in (5, 3, 1)), t, cv2.LINE_AA)
                cv2.imwrite(os.path.join(args.out + '_panels', f'{name}_{kind}.png'), p)
                panel_info[name] = dict(crop_xywh=[x0, y0, s, s], frame_hw=[H, W],
                                        boxes_in_crop={k: [[round(b[0] - x0, 1), round(b[1] - y0, 1), round(b[2] - x0, 1), round(b[3] - y0, 1), round(b[4], 3)] for b in boxes[k]] for k in DET},
                                        iou={k: round(float(d[f'{k}_iou']), 4) for k in DET}, S={k: round(float(d[f'{k}_s']), 4) for k in DET})
            if kind.endswith('_det'):
                bx = boxes[chain]
                for b in bx:
                    for lw, c in ((2.2, 'black'), (1.2, COL[chain])):
                        ax.add_patch(Rectangle((b[0] - x0, b[1] - y0), b[2] - b[0], b[3] - b[1], fill=False, lw=lw, ec=c))
                if not bx:
                    ax.text(0.5, 0.95, 'no detection', transform=ax.transAxes, ha='center', va='top', color='white', fontsize=6.2,
                            bbox=dict(boxstyle='square,pad=0.18', fc=COL[chain], ec='none'))
            if kind.endswith('_mask'):
                iou = float(d[f'{chain}_iou'])
                ax.text(0.04, 0.04, f'IoU {iou:.2f}', transform=ax.transAxes, ha='left', va='bottom', fontsize=6.2, color='white',
                        bbox=dict(boxstyle='square,pad=0.18', fc=COL[chain], ec='none'))
        # row label: image id and what went wrong in the failing pipeline
        k = 'ec' if args.reverse else 'rgb'
        mode = failure_mode(boxes[k], gt, float(d[f'{k}_iou']))
        modes[name] = mode
        fig.text((label_w - 0.05) / args.width, top + 0.5 * pw / (head + n * pw), f'#{name}\n' + mode.replace(' on ', ' on\n').replace(', ', ',\n'),
                 ha='right', va='center', fontsize=6.2, linespacing=1.25)
    # column headers: two groups over the pipeline columns
    yh = 1 - 0.13 / (head + n * pw)
    ys = 1 - 0.32 / (head + n * pw)
    xc = lambda j: (label_w + (j + 0.5) * pw) / args.width
    for j, t in enumerate(['RGB image', 'Ground truth']):
        fig.text(xc(j), (yh + ys) / 2, t, ha='center', va='center', fontsize=7, fontweight='bold')
    for j0, t, c in ((2, 'RGB camera pipeline', COL['rgb']), (4, '10 EC readings (ours)', COL['ec'])):
        fig.text((xc(j0) + xc(j0 + 1)) / 2, yh, t, ha='center', va='center', fontsize=7, fontweight='bold', color='black')
        x_a, x_b = (label_w + j0 * pw + 0.04) / args.width, (label_w + (j0 + 2) * pw - 0.04) / args.width
        fig.add_artist(plt.Line2D([x_a, x_b], [yh - 0.075 / (head + n * pw)] * 2, color=c, lw=1.6))
        for k, st in enumerate(['detection', 'segmentation']):
            fig.text(xc(j0 + k), ys, st, ha='center', va='center', fontsize=6.6)
    fig.savefig(args.out + '.pdf')
    fig.savefig(args.out + '.png', dpi=600)
    print('wrote', args.out + '.pdf', args.out + '.png')
    print('failure modes:', modes, {m: sum(v == m for v in modes.values()) for m in sorted(set(modes.values()))})
    if args.panels:
        for name in panel_info:
            panel_info[name]['failure_mode'] = modes[name]
        json.dump(dict(colours=COL, detectors=DET, threshold=0.5, panels=panel_info), open(os.path.join(args.out + '_panels', 'panels.json'), 'w'), indent=1)


if __name__ == '__main__':
    main(get_args_parser().parse_args())
