"""
Export candidate camouflage regions (ROIs) from a trained detector for the Stage-2 segmentation (spec Stage 2 §5.1).
The detector is rebuilt from its checkpoint's own args (raw bands / voltages / --pca-channels / read noise / YOLO
variant), so no model flag has to be repeated here and any run can export:
  python main_det_rois.py --resume weights/sel10g_clean_A/model_best --split train
  python main_det_rois.py --resume weights/sel10g_clean_A/model_best --split val
  python main_det_rois.py --resume weights/sel10g_clean_A/model_best --split test
--split train = the detector's 251 training ids, val = the 28 held-out ids of data_loader/splits/det_val_ids.json
(--split-file), test = the 70 test frames. Writes <out-dir>/rois_<run>_<split>.json, run = the checkpoint's folder name:
  {name: {"rois": [[x1,y1,x2,y2,conf],...], "boxes": [[x1,y1,x2,y2,conf],...], "gt_boxes": [[x1,y1,x2,y2],...]}}
(float native-frame pixels). The exported operating point is cfg/det.yaml's roi_conf (0.02) and roi_topk (5) -- the
same subset evaluate() reports its *_op metrics on -- expanded by roi_margin/roi_min and clipped. --max-rois overrides
roi_topk. Data path, cache, device and the operating point come from this command line, never from the checkpoint.
bash_files/launch_rois_all.sh exports the three Stage-2 detectors for all three splits.
"""
import os
if "RANK" not in os.environ and "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import json
from functools import partial
import torch

import util.misc as utils
import main_det
from data_loader.boxes import det_collate_fn, expand_box, boxes_from_mask
from data_loader.det_splits import load_det_ids
from data_loader.my_dataset import HyperCOD_data
from models.ec_yolo import decode_predictions
from train_eval.box_metrics import mask_coverage, filter_to_operating_point

# The flags that define a detector (its FilterBank and its YOLO): taken from the checkpoint's args, or, when an older
# checkpoint lacks one (raw133_A has no pca_channels / read_noise_* / no_gate), from main_det's parser default -- never
# from the caller's command line. hpy too: it is the detector's cfg (the operating point load_cfg fills from it), and a
# Stage-2 caller passes its own cfg/seg.yaml under the same name.
DET_MODEL_KEYS = ('session', 'top_k', 'raw_bands', 'pca_channels', 'no_gate', 'filter_select', 'filter_voltages', 'num_filters',
                  'filter_path', 'band_range', 'read_noise_db', 'read_noise_db_range', 'read_noise_model', 'seed', 'yolo_variant',
                  'gate_entropy_weight', 'contain_weight', 'epochs', 'hpy')


def get_args_parser():
    parser = argparse.ArgumentParser('Export detector ROIs', parents=[main_det.get_args_parser()], add_help=False)
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                        help="train: the detector's training ids; val: the held-out ids of --split-file; test: the test split")
    parser.add_argument('--out-dir', default='results/det')
    parser.add_argument('--max-rois', type=int, default=None, help='detections kept per frame; overrides cfg roi_topk')
    parser.set_defaults(wandb=False)
    return parser


def detector_args(ckpt_args, args):
    '''
    The namespace a detector is rebuilt with: main_det's parser defaults, overridden by the caller's flags that main_det
    knows (data path, cache, device, workers, operating point, split file), overridden by the model-defining flags
    (DET_MODEL_KEYS) from the checkpoint's args. Unlike main_det_compare.load_checkpoint_model's {**vars(args),
    **ckpt['args']}, a checkpoint never overrides where the data lives or which device runs (all three Stage-2 detectors
    store device 'cuda' and the cache of the day), and a CLI model flag never leaks into a checkpoint that lacks it.
    Keys main_det does not know (the export's --split, a Stage-2 caller's --arm) are dropped.
    '''
    defaults = vars(main_det.get_args_parser().parse_args([]))
    cli = {k: v for k, v in vars(args).items() if k in defaults}
    model = {k: ckpt_args.get(k, defaults[k]) for k in DET_MODEL_KEYS}
    return argparse.Namespace(**{**defaults, **cli, **model})


def load_detector(ckpt_path, args, device, dataset=None):
    '''
    Rebuild a trained detector from its checkpoint, in eval mode on device: (model, det_args, run_name).
      ckpt_path  e.g. weights/sel10g_clean_A/model_best; run_name = the checkpoint's folder name (sel10g_clean_A)
      args       the caller's namespace (main_det_rois / main_seg / main_seg_eval flags), see detector_args
      dataset    a HyperCOD_data built with main_det.dataset_kwargs(det_args) to take the filter matrices and band
                 statistics from (the export passes the one it reads its frames from, so no second dataset is built);
                 None builds one on the test split -- filter_bank_tensors() is split-independent, the band statistics
                 always come from the train split's stats file.
    det_args is also what models.filter_bank.build_filter_bank(det_args, dataset) needs to rebuild the arm's front end.
    '''
    assert os.path.isfile(ckpt_path), f"detector checkpoint {ckpt_path} not found"
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    det_args = detector_args(ckpt['args'], args)
    det_args.pretrained, det_args.resume, det_args.eval = 'none', ckpt_path, True       # the weights come from the checkpoint
    main_det.load_cfg(det_args)                                                        # fills only what is still None
    if dataset is None:
        dataset = HyperCOD_data(split='test', **main_det.dataset_kwargs(det_args))
    model = main_det.build_model(det_args, dataset, ckpt).to(device).eval()            # session A, or B sliced by its indices
    run_name = os.path.basename(os.path.dirname(os.path.abspath(ckpt_path)))
    return model, det_args, run_name


def split_ids(args):
    '''(HyperCOD directory split, ids or None) of --split: val frames live in the train directory.'''
    if args.split == 'test':
        return 'test', None
    train_ids, val_ids = load_det_ids(args.data_path, args.split_file or None)
    return 'train', (train_ids if args.split == 'train' else val_ids)


@torch.no_grad()
def main(args):
    utils.init_distributed_mode(args)
    main_det.load_cfg(args)
    assert args.resume, "--resume <detector checkpoint> is required, e.g. --resume weights/sel10g_clean_A/model_best"
    if args.max_rois is not None:
        args.roi_topk = args.max_rois
    # the export operating point is cfg roi_conf/roi_topk, the same one evaluate() reports its *_op metrics at
    print(f"ROI operating point: conf >= {args.roi_conf}, top {args.roi_topk} per frame")
    device = torch.device(args.device if args.device == 'cpu' or torch.cuda.is_available() else 'cpu')
    # One dataset only, built exactly like the training ones (same cache, same numeric path) but with the checkpoint's
    # filter settings (filter_select / filter_voltages / num_filters decide filter_bank_tensors()). It also hands
    # load_detector the filter matrices, so there is no need to build train/val/test just for those.
    det_args = detector_args(torch.load(args.resume, map_location='cpu', weights_only=False)['args'], args)
    hsi_split, ids = split_ids(args)
    dataset = HyperCOD_data(split=hsi_split, ids=ids, **main_det.dataset_kwargs(det_args))
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
                                         collate_fn=partial(det_collate_fn, min_area=args.min_area))
    model, det_args, run_name = load_detector(args.resume, args, device, dataset=dataset)
    print(f"{run_name}: {model.filter_bank.n_channels} input channels, {args.split} split, {len(dataset)} frames")
    out, covered, n_gt = {}, 0, 0
    for batch in loader:
        img = batch['img'].to(device)                                                  # [1, 133, H, W] fp16
        with torch.autocast(device.type, enabled=device.type == 'cuda' and args.amp):
            dets = decode_predictions(model(img), conf_thres=args.conf_thres, iou_thres=args.iou_thres,
                                      max_det=args.max_det, end2end=getattr(model.yolo, 'end2end', None))[0]
        H, W = img.shape[-2:]
        dets = filter_to_operating_point(dets, args.roi_conf, args.roi_topk)         # shared with BoxMetrics' *_op keys
        dets[:, [0, 2]] = dets[:, [0, 2]].clip(0, W); dets[:, [1, 3]] = dets[:, [1, 3]].clip(0, H)
        rois = [[*expand_box(d[:4], args.roi_margin, args.roi_min, H, W).tolist(), float(d[4])] for d in dets]
        gt_boxes, labels, cids = boxes_from_mask(batch['masks'][0], min_area=args.min_area, return_labels=True)
        for gb, cid in zip(gt_boxes, cids):
            n_gt += 1; covered += any(mask_coverage(labels == cid, r[:4]) >= 0.99 for r in rois)
        out[batch['names'][0]] = {'rois': rois, 'boxes': [[float(v) for v in d[:5]] for d in dets], 'gt_boxes': gt_boxes.tolist()}
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f'rois_{run_name}_{args.split}.json')
    with open(path, 'w') as f:
        json.dump(out, f)
    print(f"{run_name} {args.split}: {len(out)} frames, {n_gt} objects, ROI coverage recall@0.99 = {covered / max(n_gt, 1):.3f} -> {path}")
    return path


if __name__ == '__main__':
    main(get_args_parser().parse_args())
