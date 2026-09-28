"""
Export candidate camouflage regions (ROIs) from a trained detector for the Stage-2 segmentation loader.
  python main_det_rois.py --session B --resume weights/det_B/model_best --split train
  python main_det_rois.py --session B --resume weights/det_B/model_best --split test
Writes results/det/rois_<split>.json: {name: {"rois": [[x1,y1,x2,y2,conf],...], "boxes": [...], "gt_boxes": [...]}}
"""
import os
if "RANK" not in os.environ and "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import json
from functools import partial
import numpy as np
import torch

import util.misc as utils
import main_det
from data_loader.boxes import det_collate_fn, expand_box, boxes_from_mask
from data_loader.my_dataset import HyperCOD_data
from models.ec_yolo import decode_predictions
from train_eval.box_metrics import mask_coverage


def get_args_parser():
    parser = argparse.ArgumentParser('Export detector ROIs', parents=[main_det.get_args_parser()], add_help=False)
    parser.add_argument('--split', default='test', choices=['train', 'test'])
    parser.add_argument('--out-dir', default='results/det')
    parser.add_argument('--max-rois', type=int, default=5, help='detections kept per frame (by confidence)')
    parser.set_defaults(conf_thres=0.25, wandb=False)
    return parser


@torch.no_grad()
def main(args):
    utils.init_distributed_mode(args)
    main_det.load_cfg(args)
    assert args.resume, "--resume <detector checkpoint> is required, e.g. --resume weights/det_B/model_best"
    device = torch.device(args.device if args.device == 'cpu' or torch.cuda.is_available() else 'cpu')
    # One dataset only, built exactly like the training ones (same cache, same numeric path). It also hands
    # build_model the filter matrices: filter_bank_tensors() is split-independent (the band statistics always
    # come from the train split's stats file), so there is no need to build train/val/test just for those.
    dataset = HyperCOD_data(split=args.split, **main_det.dataset_kwargs(args))
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
                                         collate_fn=partial(det_collate_fn, min_area=args.min_area))
    ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
    model = main_det.build_model(args, dataset, ckpt).to(device).eval()
    out, covered, n_gt = {}, 0, 0
    for batch in loader:
        img = batch['img'].to(device)
        with torch.autocast(device.type, enabled=device.type == 'cuda' and args.amp):
            dets = decode_predictions(model(img), conf_thres=args.conf_thres, iou_thres=args.iou_thres, max_det=args.max_det)[0]
        H, W = img.shape[-2:]
        dets = dets[np.argsort(-dets[:, 4])][:args.max_rois]
        dets[:, [0, 2]] = dets[:, [0, 2]].clip(0, W); dets[:, [1, 3]] = dets[:, [1, 3]].clip(0, H)
        rois = [[*expand_box(d[:4], args.roi_margin, args.roi_min, H, W).tolist(), float(d[4])] for d in dets]
        gt_boxes, labels, ids = boxes_from_mask(batch['masks'][0], min_area=args.min_area, return_labels=True)
        for gb, cid in zip(gt_boxes, ids):
            n_gt += 1; covered += any(mask_coverage(labels == cid, r[:4]) >= 0.99 for r in rois)
        out[batch['names'][0]] = {'rois': rois, 'boxes': [[float(v) for v in d[:5]] for d in dets], 'gt_boxes': gt_boxes.tolist()}
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f'rois_{args.split}.json')
    with open(path, 'w') as f:
        json.dump(out, f)
    print(f"{args.split}: {len(out)} frames, {n_gt} objects, ROI coverage recall@0.99 = {covered / max(n_gt, 1):.3f} -> {path}")
    return path


if __name__ == '__main__':
    main(get_args_parser().parse_args())
