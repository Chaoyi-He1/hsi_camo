'''
"No detector" ROI exports for Stage 2: the whole frame as one ROI, or a grid of fixed windows over the frame, so main_seg_eval.py
--roi_run <name> scores the segmenters without any detector box (what the detector stage buys).
    python main_window_rois.py --mode frame --name full_frame               # one ROI = the frame (downscaled onto the canvas)
    python main_window_rois.py --mode windows --name windows_512            # 512 x 512 windows, stride 384, native resolution
The frame list and gt_boxes come from --like (an existing export of main_det_rois.py, test split); every box gets confidence 1
and its ROI equals the box (no growth: a window is already the crop). The segmenters never saw such prompts in training (they
were trained on object-centred ROIs), so this is an upper bound on how much the detector adds, not a trained full-frame baseline.
'''
import argparse
import json
import os


def get_args_parser():
    parser = argparse.ArgumentParser('Full-frame / window ROI exports', add_help=False)
    parser.add_argument('--like', type=str, default='results/det/rois_sel10g_clean_A_test.json', help='export whose frames and gt_boxes are reused')
    parser.add_argument('--mode', type=str, default='windows', choices=['frame', 'windows'], help='one ROI per frame, or a window grid')
    parser.add_argument('--name', type=str, required=True, help='name of the export: rois_<name>_<split>.json')
    parser.add_argument('--split', type=str, default='test', help='split suffix of the output file')
    parser.add_argument('--out_dir', type=str, default='results/det', help='where the export goes')
    parser.add_argument('--frame_hw', type=int, nargs=2, default=[1680, 1240], help='frame height and width')
    parser.add_argument('--window', type=int, default=512, help='--mode windows: window side (px)')
    parser.add_argument('--stride', type=int, default=384, help='--mode windows: step between windows; the last window is aligned to the frame edge')
    return parser


def window_grid(H, W, side, stride):
    '''Top-left corners covering the frame: steps of `stride`, the last one aligned to the edge, never beyond it.'''
    def starts(n):
        s = list(range(0, max(n - side, 0) + 1, stride))
        if s[-1] + side < n:
            s.append(n - side)
        return s
    return [(x0, y0) for y0 in starts(H) for x0 in starts(W)]


def frame_rois(mode, H, W, side=512, stride=384):
    '''The [x1, y1, x2, y2, conf] list of one frame for the mode.'''
    if mode == 'frame':
        return [[0.0, 0.0, float(W), float(H), 1.0]]
    return [[float(x0), float(y0), float(min(x0 + side, W)), float(min(y0 + side, H)), 1.0] for x0, y0 in window_grid(H, W, side, stride)]


def main(args):
    with open(args.like) as f:
        like = json.load(f)
    H, W = args.frame_hw
    out = {name: {'rois': frame_rois(args.mode, H, W, args.window, args.stride), 'boxes': frame_rois(args.mode, H, W, args.window, args.stride), 'gt_boxes': v['gt_boxes']}
           for name, v in like.items()}
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f'rois_{args.name}_{args.split}.json')
    with open(path, 'w') as f:
        json.dump(out, f)
    n = len(next(iter(out.values()))['rois'])
    print(f"{args.name}: {args.mode}, {len(out)} frames x {n} ROI(s) of {'the whole frame' if args.mode == 'frame' else f'{args.window} px (stride {args.stride})'} -> {path}")


if __name__ == '__main__':
    main(get_args_parser().parse_args())
