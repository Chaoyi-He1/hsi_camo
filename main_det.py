"""
Stage 1: EC-filter YOLO26 camouflage detector with a trainable per-channel weight vector.

Session A (all usable voltages + weight vector), single GPU:
  python main_det.py --session A --name det_A --output-dir weights/det_A
Session B (top-10 voltages from A, no weight vector), initialised from A:
  python main_det.py --session B --top_k 10 --ranking weights/det_A/gate_ranking.csv --resume weights/det_A/model_best --name det_B --output-dir weights/det_B
Evaluate a checkpoint on val + test:
  python main_det.py --session B --resume weights/det_B/model_best --eval
DDP (2 GPUs):
  torchrun --nproc_per_node=2 main_det.py --session A --name det_A --output-dir weights/det_A
Without torchrun this script pins CUDA_VISIBLE_DEVICES=0 (see the top of the file) unless it is already set.
Smoke run on a few cached frames:
  python main_det.py --session A --limit 4 --epochs 1 --name smoke --output-dir weights/smoke --no-wandb
"""
import os
if "RANK" not in os.environ and "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import csv
import datetime
import json
import math
import random
import time
from functools import partial

import yaml
import numpy as np
import torch
import torch.multiprocessing

import util.misc as utils
from util.distributed_util import Custom_DistributedSampler
from util.logger import TrainLogger
from data_loader.my_dataset import HyperCOD_data
from data_loader.boxes import det_collate_fn
from data_loader.det_splits import make_det_splits
from data_loader.cube_cache import default_cache_dir
from models.ec_yolo import build_ec_yolo, select_top_k, slice_to_channels
from train_eval.train_eval_det import train_one_epoch, evaluate
from train_eval.box_metrics import select_score

torch.multiprocessing.set_sharing_strategy('file_system')


def get_args_parser():
    parser = argparse.ArgumentParser('EC-filter YOLO26 camouflage detector (Stage 1)', add_help=False)
    parser.add_argument('--device', default='cuda', help="'cuda', 'cuda:N' or 'cpu' (a bare digit is not a torch device string)")
    parser.add_argument('--name', default='', help='run name; results_<name>.txt, runs/<name>, W&B run name')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--eval', action='store_true', help='only evaluate --resume on val + test')
    # session / model
    parser.add_argument('--session', default='A', choices=['A', 'B'], help='A: all voltages + weight vector; B: top_k fixed voltages')
    parser.add_argument('--top_k', default=10, type=int, help='voltages kept for session B')
    parser.add_argument('--ranking', default='', help='gate_ranking.csv from session A (session B)')
    parser.add_argument('--resume', default='', help='checkpoint: session A model to slice (B) or a checkpoint to continue/evaluate')
    parser.add_argument('--hpy', type=str, default='cfg/det.yaml', help='hyper parameters path')
    parser.add_argument('--yolo-variant', default='yolo26s')
    parser.add_argument('--pretrained', default='auto', help="'auto' downloads <variant>.pt, a path, or 'none'")
    parser.add_argument('--gate-entropy-weight', type=float, default=None, help='overrides cfg')
    parser.add_argument('--gate-lr', type=float, default=None, help='lr of the gate logits theta (session A); overrides cfg gate_lr')
    parser.add_argument('--select-keys', nargs='+', default=None, help='val summary keys that rank checkpoints (primary first); overrides cfg select_keys')
    parser.add_argument('--contain-weight', type=float, default=None, help='overrides cfg')
    # data
    parser.add_argument('--data-path', default='/data2/chaoyi/HyperCOD/Raw data')
    parser.add_argument('--cache-dir', default='', help='fp16 cube cache, default <data-path>/cache_fp16')
    parser.add_argument('--band-range', type=float, nargs=2, default=[400.0, 800.0])
    parser.add_argument('--filter-path', default=None)
    parser.add_argument('--filter-select', default='all', choices=['all', 'uniform', 'osp', 'manual'])
    parser.add_argument('--num-filters', type=int, default=30)
    parser.add_argument('--filter-voltages', type=float, nargs='+', default=None)
    parser.add_argument('--split-file', default='', help='val ids json, default data_loader/splits/det_val_ids.json')
    parser.add_argument('--limit', type=int, default=0, help='use only the first N train / N//4 val / N test frames (smoke runs)')
    # training
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lrf', default=0.05, type=float, help='final lr = lr * lrf (cosine)')
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--accumulate', default=4, type=int, help='gradient accumulation steps')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N')
    parser.add_argument('--amp', action='store_false', help='disable mixed precision; ignored on CUDA, where fp16 is required to fit the response tensor')
    parser.add_argument('--no-flip', action='store_true', help='disable random h/v flips')
    parser.add_argument('--save_every', default=10, type=int)
    # eval overrides
    parser.add_argument('--conf-thres', type=float, default=None); parser.add_argument('--iou-thres', type=float, default=None)
    parser.add_argument('--roi-margin', type=float, default=None); parser.add_argument('--roi-min', type=float, default=None)
    parser.add_argument('--roi-conf', type=float, default=None, help='ROI operating point: confidence floor (cfg: 0.25)')
    parser.add_argument('--roi-topk', type=int, default=None, help='ROI operating point: detections kept per frame (cfg: 5)')
    parser.add_argument('--min-area', type=int, default=None, help='GT component floor in px (cfg: 100; fixture tests use 10)')
    # logging
    parser.add_argument('--output-dir', default='weights/det_A')
    parser.add_argument('--runs-dir', default='runs')
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--wandb-entity', default='chaoyi-hsi'); parser.add_argument('--wandb-project', default='hsi_camo')
    parser.add_argument('--wandb-dir', default='wandb')
    # distributed
    parser.add_argument('--world_size', default=1, type=int); parser.add_argument('--dist_url', default='env://')
    return parser


def build_optimizer(model_without_ddp, args):
    '''
    AdamW in three groups: weights (decayed), biases / norms / the rebuilt first conv (no decay), and the gate logits
    filter_bank.theta with their own lr (session A only). The gate needs it: AdamW moves a logit by about lr per
    step, so at the model's 1e-4 the 344-way softmax stays uniform for a whole run (det_A: normalised entropy
    1.0000 after 12 epochs) and the top-k ranking would be noise. cfg gate_lr / --gate-lr set it.
    '''
    named = [(n, p) for n, p in model_without_ddp.named_parameters() if p.requires_grad]
    gate = [p for n, p in named if n == 'filter_bank.theta']
    no_decay = [p for n, p in named if n != 'filter_bank.theta' and (n.endswith('.bias') or n == 'yolo.model.0.conv.weight' or p.ndim == 1)]
    skip = {id(p) for p in no_decay} | {id(p) for p in gate}
    decay = [p for n, p in named if id(p) not in skip]
    groups = [{'params': decay, 'weight_decay': args.weight_decay}, {'params': no_decay, 'weight_decay': 0.0}]
    if gate:
        groups.append({'params': gate, 'weight_decay': 0.0, 'lr': args.gate_lr})
    return torch.optim.AdamW(groups, lr=args.lr)


def load_cfg(args):
    with open(args.hpy) as f:
        cfg = yaml.safe_load(f)
    for k in ['gate_entropy_weight', 'gate_lr', 'contain_weight', 'conf_thres', 'iou_thres', 'roi_margin', 'roi_min',
              'roi_conf', 'roi_topk', 'min_area', 'select_keys']:
        if getattr(args, k) is None:
            setattr(args, k, cfg[k])
    args.max_norm = cfg.get('max_norm', 10.0); args.max_det = cfg.get('max_det', 300)
    return cfg


def dataset_kwargs(args):
    '''
    HyperCOD_data kwargs shared by training and the ROI export (main_det_rois), so both read the same fp16 cache
    (~0.3 s/frame instead of 9.5 s through h5py) and the same numeric path the model was trained on.
    '''
    return dict(data_path=args.data_path, use_filter=False, norm='p99', crop_size=0, out_dtype='float16',
                cache_dir=args.cache_dir or default_cache_dir(args.data_path),
                band_range=tuple(args.band_range), filter_path=args.filter_path, num_filters=args.num_filters,
                filter_select=args.filter_select, filter_voltages=args.filter_voltages, filter_norm='l1')


def build_datasets(args):
    # make_det_splits defaults to holding out 28 ids; the real split (279 train frames) uses that default, but a
    # tiny fixture (a handful of frames, e.g. the test suite) cannot spare 28 for validation, so cap it to leave
    # at least one id on each side. Deviation from the brief, which calls make_det_splits with no n_val override.
    hsi_dir = os.path.join(args.data_path, 'train', 'hyperspectral')
    n_train_ids = len([f for f in os.listdir(hsi_dir) if f.endswith('.mat')])
    n_val = min(28, max(1, n_train_ids - 1))
    train_ids, val_ids = make_det_splits(args.data_path, n_val=n_val, path=args.split_file or None)
    test_ids = None
    if args.limit:
        train_ids, val_ids = train_ids[:args.limit], val_ids[:max(1, args.limit // 4)]
        test_ids = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(args.data_path, 'test', 'hyperspectral')) if f.endswith('.mat'))
        test_ids = sorted(test_ids, key=int)[:max(1, args.limit)]
    kw = dataset_kwargs(args)
    return (HyperCOD_data(split='train', ids=train_ids, **kw), HyperCOD_data(split='train', ids=val_ids, **kw),
            HyperCOD_data(split='test', ids=test_ids, **kw))


def build_train_sampler(dataset, distributed, num_replicas=None, rank=None):
    '''
    Training sampler, one pass over the data per epoch. Custom_DistributedSampler defaults to extend_factor=20
    (20 concatenated permutations per "epoch"), but the cosine LR, the E2E loss gain decay and the per-epoch
    validation/checkpoint cadence all assume a single pass, so DDP pins it to 1 and the length is checked.
    '''
    if not distributed:
        return torch.utils.data.RandomSampler(dataset)
    sampler = Custom_DistributedSampler(dataset, num_replicas=num_replicas, rank=rank, shuffle=True, extend_factor=1)
    expect = math.ceil(len(dataset) / sampler.num_replicas)
    assert len(sampler) == expect, f"DDP sampler yields {len(sampler)} indices per rank, expected one pass ({expect})"
    return sampler


def read_ranking(csv_path, k):
    with open(csv_path, newline='') as f:
        rows = sorted(csv.DictReader(f), key=lambda r: int(r['rank']))
    assert len(rows) >= k, f"{csv_path} has {len(rows)} channels, top_k={k}"
    return [int(r['index']) for r in rows[:k]]


def build_model(args, dataset_train, ckpt=None):
    '''Session A: all voltages + weight vector. Session B: slice an A model (from --ranking/--resume or a B checkpoint's indices).'''
    a_args = argparse.Namespace(**{**vars(args), 'session': 'A'})
    model = build_ec_yolo(a_args, dataset_train)                                   # A structure, N channels
    n_all = model.filter_bank.n_channels
    if args.session == 'A':
        model.selected_indices = list(range(n_all))
        if ckpt is not None:
            model.load_state_dict(ckpt['model'])
        return model
    if ckpt is not None and ckpt['args'].get('session') == 'B':                      # continue / evaluate a B checkpoint
        idx = list(ckpt['selected_indices'])
        model = slice_to_channels(model, idx, args.yolo_variant)
        model.load_state_dict(ckpt['model'])
    else:
        assert ckpt is not None and args.ranking, "session B needs --resume <session A checkpoint> and --ranking <gate_ranking.csv>"
        model.load_state_dict(ckpt['model'])                                          # session A weights
        idx = read_ranking(args.ranking, args.top_k)
        model = slice_to_channels(model, idx, args.yolo_variant)
    model.selected_indices = idx
    print(f"session B: {len(idx)} voltages {np.round(model.selected_voltages, 2).tolist()}")
    return model


def save_checkpoint(path, model, optimizer, scaler, scheduler, epoch, args, best=(-1.0, -1.0)):
    utils.save_on_master({'model': model.state_dict(), 'optimizer': optimizer.state_dict() if optimizer else None,
                          'scaler': scaler.state_dict() if scaler else None, 'lr_scheduler': scheduler.state_dict() if scheduler else None,
                          'epoch': epoch, 'args': vars(args), 'selected_indices': list(model.selected_indices),
                          'selected_voltages': [float(v) for v in model.selected_voltages],
                          'best': [float(v) for v in best]}, path)


def main(args):
    utils.init_distributed_mode(args)
    cfg = load_cfg(args)
    args.name = args.name or f'det_{args.session}'
    device = torch.device(args.device if args.device == 'cpu' or torch.cuda.is_available() else 'cpu')
    seed = args.seed + utils.get_rank(); torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    os.makedirs(args.output_dir, exist_ok=True)
    logger = TrainLogger(args, cfg)

    dataset_train, dataset_val, dataset_test = build_datasets(args)
    print(f"train {len(dataset_train)} / val {len(dataset_val)} / test {len(dataset_test)} frames, {dataset_train.n_bands} bands")
    sampler_train = build_train_sampler(dataset_train, args.distributed)
    collate = partial(det_collate_fn, min_area=args.min_area)
    mk = lambda ds, sampler, bs, shuffle: torch.utils.data.DataLoader(ds, batch_size=bs, sampler=sampler, shuffle=shuffle, num_workers=args.num_workers,
                                                                     collate_fn=collate, pin_memory=device.type == 'cuda', drop_last=False)
    loader_train = mk(dataset_train, sampler_train, args.batch_size, False)
    loader_val, loader_test = mk(dataset_val, None, 1, False), mk(dataset_test, None, 1, False)

    ckpt = torch.load(args.resume, map_location='cpu', weights_only=False) if args.resume else None
    model = build_model(args, dataset_train, ckpt).to(device)
    model.attach_criterion()
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu]); model_without_ddp = model.module

    optimizer = build_optimizer(model_without_ddp, args)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # AMP must always be on when running on CUDA: the 344-channel response tensor has to stay fp16 at full
    # resolution to fit in memory. --amp (store_false) is meant to be an opt-out, but on CUDA we keep it on
    # regardless and just say so; only on CPU (tests) does --amp actually disable the scaler (there is none).
    use_amp = device.type == 'cuda'
    if args.amp is False and device.type == 'cuda':
        print("AMP kept on: full-resolution fp16 is required on CUDA")
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    best = (-1.0, -1.0)
    if ckpt is not None and ckpt['args'].get('session') == args.session and not args.eval and ckpt.get('optimizer'):
        optimizer.load_state_dict(ckpt['optimizer']); scheduler.load_state_dict(ckpt['lr_scheduler'])
        if scaler is not None and ckpt.get('scaler'):
            scaler.load_state_dict(ckpt['scaler'])
        args.start_epoch = ckpt['epoch'] + 1
        # carry the best score over: without it the first post-resume epoch always beats (-1, -1) and overwrites
        # model_best with a worse model. .get() keeps checkpoints written before this field loadable.
        best = tuple(ckpt.get('best', (-1.0, -1.0)))
    if not args.eval and args.start_epoch:
        # E2ELoss decays its one2many/one2one gains once per epoch (ECYolo.end_epoch -> criterion.update()), but
        # attach_criterion() above rebuilt the criterion at update 0; replay the epochs already trained.
        for _ in range(args.start_epoch):
            model_without_ddp.end_epoch()

    eval_kw = dict(conf_thres=args.conf_thres, iou_thres=args.iou_thres, max_det=args.max_det, roi_margin=args.roi_margin, roi_min=args.roi_min,
                   min_area=args.min_area, roi_conf=args.roi_conf, roi_topk=args.roi_topk, amp=scaler is not None)
    results_path = os.path.join(args.output_dir, f'results_{args.name}.txt')
    if args.eval:
        if utils.is_main_process():                              # loader_val/test are not sharded: rank 0 only
            val = evaluate(model_without_ddp, loader_val, device, logger=logger, epoch=args.start_epoch, tag='val', **eval_kw)
            test = evaluate(model_without_ddp, loader_test, device, logger=logger, epoch=args.start_epoch, tag='test', **eval_kw)
            with open(results_path, 'a') as f:
                f.write(json.dumps({'eval': True, 'resume': args.resume, 'val': val, 'test': test}) + '\n')
        logger.finish(); return

    print(f"Start training from epoch {args.start_epoch}, best so far {best}"); start = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(model, loader_train, optimizer, device, epoch, max_norm=args.max_norm, scaler=scaler,
                                      accumulate=args.accumulate, logger=logger, flip=not args.no_flip)
        scheduler.step()
        # loader_val is not sharded, so every rank would evaluate the whole val set and only rank 0 would use
        # the numbers; the other ranks simply wait at the next epoch's first all-reduce.
        if utils.is_main_process():
            val = evaluate(model_without_ddp, loader_val, device, logger=logger, epoch=epoch, tag='val', **eval_kw)
            if args.session == 'A':
                w = model_without_ddp.filter_bank.weights.detach().cpu().numpy(); volts = model_without_ddp.selected_voltages
                logger.histogram('gate/weights', w, epoch); logger.scalars({'gate/max': w.max(), 'gate/entropy': float(model_without_ddp.filter_bank.entropy().detach())}, epoch)
                order = np.argsort(-w)[:20]; logger.table('gate/top20', ['rank', 'voltage', 'weight'], [[r + 1, float(volts[i]), float(w[i])] for r, i in enumerate(order)], epoch)
            score = select_score(val, args.select_keys)
            if score > best:                                     # before model_{epoch}, so it stores the best including this epoch
                best = score; save_checkpoint(os.path.join(args.output_dir, 'model_best'), model_without_ddp, optimizer, scaler, scheduler, epoch, args, best)
            if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
                save_checkpoint(os.path.join(args.output_dir, f'model_{epoch}'), model_without_ddp, optimizer, scaler, scheduler, epoch, args, best)
            with open(results_path, 'a') as f:
                f.write(json.dumps({'epoch': epoch, 'train': train_stats, 'val': val, 'best': list(best)}) + '\n')
    print(f"Training time {datetime.timedelta(seconds=int(time.time() - start))}, best {tuple(args.select_keys)} = {best}")

    if utils.is_main_process():
        best_ckpt = torch.load(os.path.join(args.output_dir, 'model_best'), map_location='cpu', weights_only=False)
        model_without_ddp.load_state_dict(best_ckpt['model'])
        val = evaluate(model_without_ddp, loader_val, device, logger=logger, epoch=args.epochs, tag='val_best', **eval_kw)
        test = evaluate(model_without_ddp, loader_test, device, logger=logger, epoch=args.epochs, tag='test_best', **eval_kw)
        with open(results_path, 'a') as f:
            # 'epoch' mirrors best_ckpt['epoch'] so the last line of results_<name>.txt is always keyed by an
            # epoch number, like every per-epoch line above it (deviation from the brief, which only wrote
            # 'best_epoch' here, leaving the final line without an 'epoch' field).
            f.write(json.dumps({'epoch': best_ckpt['epoch'], 'final': True, 'best_epoch': best_ckpt['epoch'], 'val': val, 'test': test}) + '\n')
        if args.session == 'A':
            k = min(args.top_k, model_without_ddp.filter_bank.n_channels)
            idx, w, volts = select_top_k(model_without_ddp, k, csv_path=os.path.join(args.output_dir, 'gate_ranking.csv'))
            with open(os.path.join(args.output_dir, 'top_k.json'), 'w') as f:
                json.dump({'top_k': k, 'indices': idx, 'voltages': [float(volts[i]) for i in idx], 'weights': [float(w[i]) for i in idx]}, f, indent=1)
            print(f"top-{k} voltages (V): {[round(float(volts[i]), 2) for i in idx]}")
    logger.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('EC-filter YOLO26 detector', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
