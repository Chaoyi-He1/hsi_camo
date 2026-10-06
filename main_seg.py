"""
Stage 2: camouflaged-object segmentation inside the Stage-1 ROIs (spec docs/superpowers/specs/2026-10-04-stage2-segmentation-design.md).
Trains one (model, arm, seed) on the crop cache (data_loader.roi_crops.build_crop_cache) and selects model_best on the
deterministic val ROIs; the test frames are only touched by main_seg_eval.py.
  python main_seg.py --seg_model sam2unet --arm ec10 --det_ckpt weights/sel10g_clean_A/model_best --seed 0
  python main_seg.py --seg_model sam2box --arm raw --det_ckpt weights/raw133_A/model_best --seed 1
  python main_seg.py --seg_model zoomnext --arm ec24 --det_ckpt weights/sel24g_clean_A/model_best --seed 2
Control: SAM2-UNet on the 3-channel pseudo-RGB render, with the raw arm's ROIs:
  python main_seg.py --seg_model sam2unet --arm rgb --det_ckpt weights/raw133_A/model_best --seed 0
Pre-flight (one batch per box source, shapes, peak GPU memory; nothing is written):
  python main_seg.py --seg_model zoomnext --arm raw --det_ckpt weights/raw133_A/model_best --dry_run
Continue a run, or evaluate a checkpoint on the val ROIs:
  python main_seg.py --seg_model sam2unet --arm ec10 --det_ckpt weights/sel10g_clean_A/model_best --resume weights/seg_sam2unet_ec10_s0/model_last
  python main_seg.py --seg_model sam2unet --arm ec10 --det_ckpt weights/sel10g_clean_A/model_best --resume weights/seg_sam2unet_ec10_s0/model_best --eval
Flags left at None come from --hpy cfg/seg.yaml (common keys, then the --seg_model section); explicit flags win.
A resumed or evaluated checkpoint must be this run's (model, arm, detector folder, seed, canvas; --epochs too when training
continues). Resuming drops the results lines written after the checkpoint and keeps a later on-disk model_best; a fresh
start (no --resume / --eval) moves a previous attempt's results file, TensorBoard dir and checkpoints to <path>.prev.
Spec §9 checks: the detector must be the arm's (check_arm), its filter bank is rebuilt bit-identically (build_front_end with
the checkpoint's state), and the crop cache must hold that detector's ROIs and band window.
Without torchrun this script pins CUDA_VISIBLE_DEVICES=0 unless it is already set (as main_det does); one job per GPU.
"""
import os
if "RANK" not in os.environ and "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import datetime
import json
import math
import random
import shutil
import time

import yaml
import numpy as np
import torch
import torch.multiprocessing

import util.misc as utils
import main_det
import main_det_rois
from util.logger import TrainLogger
from data_loader.my_dataset import HyperCOD_data
from data_loader.roi_crops import HyperCOD_roi, seg_collate_fn
from models.seg_models import ARMS, ARM_PCA, build_front_end, build_seg_model
from models.seg_stem import pseudo_rgb, imagenet_normalize, fit_rgb_map
from train_eval.train_eval_seg import train_one_epoch, evaluate, seg_inputs

torch.multiprocessing.set_sharing_strategy('file_system')

SEG_MODELS = ('sam2unet', 'sam2box', 'zoomnext')
AMP_DTYPES = {'bfloat16': torch.bfloat16, 'float16': torch.float16}


def get_args_parser():
    parser = argparse.ArgumentParser('Stage-2 segmentation inside the Stage-1 ROIs', add_help=False)
    # run
    parser.add_argument('--seg_model', type=str, default='sam2unet', choices=SEG_MODELS,
                        help='sam2unet: SAM2-UNet (Hiera-L trunk frozen, adapters + decoder); sam2box: SAM2.1 + box prompt + LoRA; '
                             'zoomnext: ZoomNeXt-B2 (full fine-tune)')
    parser.add_argument('--arm', type=str, default='raw', choices=ARMS,
                        help='input arm: raw (133 standardised bands), ec10 / ec24 (the detector\'s whitened EC readings), '
                             'rgb (control: ImageNet-normalised pseudo-RGB render, with the raw arm\'s ROIs)')
    parser.add_argument('--det_ckpt', type=str, default='',
                        help="the arm's Stage-1 detector checkpoint (raw / rgb: weights/raw133_A/model_best, ec10: "
                             "weights/sel10g_clean_A/model_best, ec24: weights/sel24g_clean_A/model_best); the front end is rebuilt "
                             "from its flags and checked against its filter_bank tensors")
    parser.add_argument('--name', type=str, default='', help='run name, default seg_<seg_model>_<arm>_s<seed>; results_<name>.txt, runs/<name>, W&B')
    parser.add_argument('--output_dir', type=str, default='', help='model_best / model_last / results_<name>.txt, default weights/<name>')
    parser.add_argument('--seed', type=int, default=0, help='torch / numpy / random seed (runs use 0, 1, 2)')
    parser.add_argument('--device', type=str, default='cuda', help="'cuda', 'cuda:N' or 'cpu'")
    parser.add_argument('--resume', type=str, default='', help='a main_seg checkpoint: continue training from it, or evaluate it with --eval')
    parser.add_argument('--eval', action='store_true', help='only evaluate --resume on the val ROIs')
    parser.add_argument('--dry_run', action='store_true',
                        help='pre-flight: build everything, run one training batch per box source forward + backward, print the '
                             'shapes and the peak GPU memory, write nothing')
    parser.add_argument('--hpy', type=str, default='cfg/seg.yaml', help='hyper-parameter file (flags left at None are filled from it)')
    # data
    parser.add_argument('--data_path', type=str, default='/data2/chaoyi/HyperCOD/Raw data',
                        help='HyperCOD root (band statistics, EC filter file, intensity scales)')
    parser.add_argument('--cache_dir', type=str, default='', help='fp16 frame cache, default <data_path>/cache_fp16 (passed to the detector flags)')
    parser.add_argument('--crop_cache', type=str, default='', help='crop cache of build_crop_cache (index.json + windows), default <data_path>/crop_cache_seg')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    # ROI items (None -> cfg)
    parser.add_argument('--box_mix', type=float, nargs=3, default=None,
                        help='training box sources: expanded GT / matched detector ROI / false-positive ROI fractions (cfg: 0.5 0.4 0.1)')
    parser.add_argument('--gt_jitter', type=float, default=None, help='outward jitter of the expanded GT boxes, fraction of the box side (cfg: 0.15)')
    parser.add_argument('--canvas', type=int, default=None, help='canvas side in px (cfg: 512)')
    # optimisation (None -> cfg, per-model section first)
    parser.add_argument('--epochs', type=int, default=None, help='training epochs (cfg: 200)')
    parser.add_argument('--batch_size', type=int, default=None, help='batch size (cfg: 8; zoomnext 4)')
    parser.add_argument('--accumulate', type=int, default=None, help='gradient accumulation steps (cfg: 1; zoomnext 2)')
    parser.add_argument('--lr', type=float, default=None, help='model lr (cfg: sam2unet 1e-3, sam2box / zoomnext 1e-4)')
    parser.add_argument('--stem_lr', type=float, default=None, help='lr of the folded N+1-channel stem (cfg: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=None, help='AdamW weight decay (cfg: 1e-4)')
    parser.add_argument('--max_norm', type=float, default=None, help='gradient clipping norm (cfg: 1.0)')
    # pretrained weights (bash_files/setup_third_party.sh)
    parser.add_argument('--sam2_ckpt', type=str, default='weights/pretrained/sam2.1_hiera_large.pt', help='SAM2.1 Hiera-L checkpoint (sam2box)')
    parser.add_argument('--sam2unet_ckpt', type=str, default='weights/pretrained/sam2_hiera_large.pt',
                        help='SAM2 v1 Hiera-L checkpoint the SAM2-UNet trunk starts from (the SAM2-UNet README asks for v1, not 2.1)')
    parser.add_argument('--zoomnext_ckpt', type=str, default='weights/pretrained/pvtv2-b2-zoomnext.pth', help='ZoomNeXt PVTv2-B2 COD checkpoint')
    # logging
    parser.add_argument('--runs_dir', type=str, default='runs', help='TensorBoard root')
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=True, help='log to W&B (falls back to offline)')
    parser.add_argument('--wandb_entity', type=str, default='chaoyi-hsi', help='W&B entity')
    parser.add_argument('--wandb_project', type=str, default='hsi_camo', help='W&B project')
    parser.add_argument('--wandb_dir', type=str, default='wandb', help='W&B local directory')
    return parser


def load_cfg(args):
    '''
    Fill args from --hpy: the common keys, overridden by the section of args.seg_model under `models`. A key is applied when
    the flag is unset (None) or has no flag at all (amp_dtype, roi_min, fit_pixels, ... and model knobs such as lora_r or
    encoder_lr_mult that build_seg_model reads from args). Returns the merged flat dict (logged with the run config).
    '''
    with open(args.hpy) as f:
        cfg = yaml.safe_load(f)
    sections = cfg.get('models', {}) or {}
    assert args.seg_model in sections, f"{args.hpy} has no models.{args.seg_model} section (has {sorted(sections)})"
    merged = {**{k: v for k, v in cfg.items() if k != 'models'}, **sections[args.seg_model]}
    for k, v in merged.items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)
    return merged


def det_args_from_ckpt(det_ckpt, args):
    '''
    The detector's flags for the arm's front end, through main_det_rois.detector_args (the one list of model-defining
    detector keys, DET_MODEL_KEYS): main_det's parser defaults (raw133_A predates --pca-channels, read noise and --no-gate,
    so missing keys fall back to the Stage-1 defaults), this run's data_path / cache_dir / device, then the model keys of
    ckpt['args']. The checkpoint never overrides the data location (a cache moved to another disk, the test fixture) or the
    device. Returns (det_args, ckpt); det_args.name = the detector run (the checkpoint's directory, e.g. sel10g_clean_A).
    '''
    assert det_ckpt and os.path.isfile(det_ckpt), f"--det_ckpt {det_ckpt!r} not found (the arm's Stage-1 detector checkpoint)"
    ckpt = torch.load(det_ckpt, map_location='cpu', weights_only=False)
    assert 'args' in ckpt and 'model' in ckpt, f"{det_ckpt} is not a main_det checkpoint (keys {sorted(ckpt)})"
    det_args = main_det_rois.detector_args(ckpt['args'], argparse.Namespace(data_path=args.data_path, cache_dir=args.cache_dir,
                                                                            device=args.device))
    det_args.name = det_run_of(det_ckpt)
    return det_args, ckpt


def det_run_of(det_ckpt):
    '''The detector run of a checkpoint path: its folder name (weights/sel10g_clean_A/model_best -> sel10g_clean_A).'''
    return os.path.basename(os.path.dirname(os.path.abspath(det_ckpt)))


def check_arm(arm, det_args):
    '''Spec §9: the detector must be the arm's own (raw / rgb: --raw-bands; ec10 / ec24: --pca-channels 10 / 24 EC readings).'''
    raw = bool(getattr(det_args, 'raw_bands', False))
    k = int(getattr(det_args, 'pca_channels', 0) or 0)
    if arm in ('raw', 'rgb'):
        assert raw, f"--arm {arm} needs the --raw-bands detector (raw133_A), got {det_args.name}: raw_bands={raw}, pca_channels={k}"
    else:
        want = ARM_PCA[arm]
        assert not raw and k == want, \
            f"--arm {arm} needs an EC detector with --pca-channels {want}, got {det_args.name}: raw_bands={raw}, pca_channels={k}"


def build_arm_front_end(args, device):
    '''
    (front_end, det_args, dataset) of args.arm: the detector's flags from --det_ckpt (det_args_from_ckpt, check_arm), a
    HyperCOD_data built exactly like the detector's (main_det.dataset_kwargs: band statistics, filter matrices, wavelengths)
    and models.seg_models.build_front_end, which checks the rebuilt filter bank against the checkpoint's filter_bank.*
    tensors (spec §3 "Front end = Stage 1's", §9; the rgb control has no filter bank); front_end in eval mode on device.
    '''
    det_args, det_ckpt = det_args_from_ckpt(args.det_ckpt, args)
    check_arm(args.arm, det_args)
    dataset = HyperCOD_data(split='train', **main_det.dataset_kwargs(det_args))
    front_end = build_front_end(args.arm, det_args, dataset, det_state=det_ckpt['model'] if args.arm != 'rgb' else None)
    print(f"arm {args.arm}: front end of {det_args.name} -> {front_end.n_out} channels"
          + (" (filter bank checked against the detector)" if args.arm != 'rgb' else ''))
    return front_end.to(device).eval(), det_args, dataset


@torch.no_grad()
def fit_arm_rgb_map(front_end, dataset, wavelens, device, n_pixels=2000000, n_items=64, seed=0, num_workers=0):
    '''
    (P [3, N], q [3], r2 [3]) of the arm's stem fold (spec §3, §5.3): least squares rgb_norm ~ P z + q on at most n_pixels
    canvas pixels of at most n_items training items. dataset should be deterministic (HyperCOD_roi(split 'train',
    train=False): oracle and matched ROIs, no augmentation); only valid (non-padding) pixels are used, the same number from
    every item. z = the arm's channels from the fp32 front end (the seg_inputs rule), rgb_norm = the ImageNet-normalised
    pseudo-RGB render of the same p99-scaled crop (models.seg_stem.imagenet_normalize(pseudo_rgb(...)), the render the
    rgb arm feeds). r2 = the fit's R^2 per R, G, B (1 for the rgb arm), printed.
    '''
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(dataset), size=min(int(n_items), len(dataset)), replace=False))
    per_item = max(1, int(n_pixels) // len(idx))
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, idx.tolist()), batch_size=1, shuffle=False,
                                         num_workers=num_workers, collate_fn=seg_collate_fn)
    front_end.eval()
    zs, rgbs = [], []
    for batch in loader:
        img = batch['img'].to(device).float()                                            # [1, 133, c, c] p99-scaled
        valid = batch['valid'][0, 0].to(device) > 0.5                                    # [c, c]
        with torch.autocast(device.type, enabled=False):
            z = front_end(img).float()[0]                                                # [N, c, c]
        rgb = imagenet_normalize(pseudo_rgb(img, wavelens))[0]                           # [3, c, c]
        z, rgb = z[:, valid], rgb[:, valid]                                              # [N, n_valid], [3, n_valid]
        pick = torch.from_numpy(rng.choice(z.shape[1], size=min(per_item, z.shape[1]), replace=False)).to(device)
        zs.append(z[:, pick].T.double().cpu().numpy()); rgbs.append(rgb[:, pick].T.double().cpu().numpy())
    z, rgb = np.concatenate(zs), np.concatenate(rgbs)                                    # [n, N], [n, 3]
    P, q = fit_rgb_map(z, rgb)                                                           # [3, N], [3]
    resid = rgb - (z @ np.asarray(P, np.float64).T + np.asarray(q, np.float64))
    r2 = 1.0 - resid.var(axis=0) / np.maximum(rgb.var(axis=0), 1e-12)
    print(f"stem fold: {len(z)} pixels of {len(idx)} training items, {z.shape[1]} channels -> pseudo-RGB, R^2 (R, G, B) {np.round(r2, 4).tolist()}")
    return np.asarray(P, np.float32), np.asarray(q, np.float32), r2


def val_score(summary):
    '''Checkpoint selection (spec §7): mean of S-measure and weighted F over the deterministic val ROIs; nan -> 0.'''
    s = 0.5 * (summary['S'] + summary['Fw'])
    return float(s) if np.isfinite(s) else 0.0


def save_checkpoint(path, model, optimizer, scaler, scheduler, epoch, args, best, best_epoch, P, q):
    '''
    Full model state (frozen trunk included, so a checkpoint needs no pretrained file to load) + optimiser state + the stem
    fold. Written to <path>.tmp in the same directory, then os.replace'd over path: a crash or a full disk mid-write
    leaves the previous model_best / model_last intact, never a truncated file (the queue resumes from model_last).
    '''
    if not utils.is_main_process():
        return
    tmp = path + '.tmp'
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict() if optimizer else None,
                'scaler': scaler.state_dict() if scaler else None, 'lr_scheduler': scheduler.state_dict() if scheduler else None,
                'epoch': epoch, 'args': vars(args), 'best': float(best), 'best_epoch': int(best_epoch),
                'P': np.asarray(P, np.float32), 'q': np.asarray(q, np.float32), 'n_in': int(np.shape(P)[1])}, tmp)
    os.replace(tmp, path)


def check_ckpt_args(ckpt_args, args, path, train=True):
    '''
    A checkpoint this run continues (--resume) or evaluates (--eval) must be this run's: same seg_model, arm, seed, canvas
    and detector run (the folder of --det_ckpt: the front end and the ROIs the model was trained on); continuing its
    training (train=True) also needs the same --epochs, which the cosine schedule and ZoomNeXt's loss ramp depend on.
    '''
    for k in ('seg_model', 'arm', 'seed', 'canvas') + (('epochs',) if train else ()):
        assert ckpt_args.get(k) == getattr(args, k), f"{path} is a {k}={ckpt_args.get(k)!r} run, not {getattr(args, k)!r} (--{k})"
    det, want = det_run_of(ckpt_args.get('det_ckpt') or ''), det_run_of(args.det_ckpt)
    assert det == want, \
        f"{path} was trained with detector {det} ({ckpt_args.get('det_ckpt')!r}), this run uses {want} (--det_ckpt {args.det_ckpt})"


def move_aside(paths):
    '''
    Fresh start (neither --resume nor --eval): every existing file or directory of paths (a previous attempt's results
    file, TensorBoard dir, model_best / model_last) is moved to <path>.prev, replacing an older .prev. So a restarted run
    never appends to a crashed run's lines or events, and a stale model_last can never be resumed into the new run.
    Returns the paths moved.
    '''
    moved = []
    for p in paths:
        if not os.path.lexists(p):
            continue
        prev = p + '.prev'
        if os.path.isdir(prev) and not os.path.islink(prev):
            shutil.rmtree(prev)
        elif os.path.lexists(prev):
            os.remove(prev)
        os.replace(p, prev)
        moved.append(p)
    return moved


def truncate_results(path, last_epoch):
    '''
    Resume after epoch last_epoch (the checkpoint's): rewrite results_<name>.txt atomically (<path>.tmp + os.replace),
    keeping, in order,
      - the per-epoch lines of epochs <= last_epoch: the history the checkpoint continues;
      - every --eval line: it describes a checkpoint file, not the training history;
    and dropping
      - the per-epoch lines of later epochs: the interrupted run wrote them after its last checkpoint, they are re-run;
      - every 'final' line: its 'epoch' is the best epoch, not a training epoch, so the epoch rule alone would keep it;
        the resumed run writes its own at the end, and the queue treats a run with a final line as finished;
      - an unreadable line (a write cut by the crash).
    Returns the number of lines dropped.
    '''
    if not os.path.isfile(path):
        return 0
    with open(path) as f:
        lines = [l for l in f.read().splitlines() if l.strip()]
    keep = []
    for l in lines:
        try:
            r = json.loads(l)
        except json.JSONDecodeError:
            continue
        if r.get('eval') or (not r.get('final') and r.get('epoch') is not None and int(r['epoch']) <= last_epoch):
            keep.append(l)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        f.write(''.join(l + '\n' for l in keep))
    os.replace(tmp, path)
    return len(lines) - len(keep)


def later_best(args, last_epoch):
    '''
    (best, best_epoch) of <output_dir>/model_best when it was saved after epoch last_epoch, else None. The interrupted run
    may have improved after its last model_last (saved every save_every epochs); resuming from the checkpoint's older best
    would let the first epoch that beats that older score overwrite the better model_best. The file must be this run's
    (check_ckpt_args); it is opened with mmap, so its tensors are not read.
    '''
    path = os.path.join(args.output_dir, 'model_best')
    if not os.path.isfile(path):
        return None
    ckpt = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    check_ckpt_args(ckpt['args'], args, path)
    found = (float(ckpt['best']), int(ckpt['best_epoch'])) if int(ckpt['epoch']) > last_epoch else None
    del ckpt
    return found


def dry_run(model, front_end, dataset, args, device, amp_dtype):
    '''
    Pre-flight (spec §9): one training batch per box source (expanded GT 'gt', matched detector ROI 'det', false-positive ROI
    'fp'; at most 20 x batch_size random items are read to find them), each through seg_inputs, forward and backward exactly
    as in training, with the input / output shapes checked. A source with a single item is run as a batch of 2 (ZoomNeXt's
    pooled BatchNorm refuses one item in training). No optimizer step, nothing written.
    Returns {'n_in', 'sources': {source: x shape}, 'peak_mem_gib'} and prints the peak GPU memory.
    '''
    by_source = {'gt': [], 'det': [], 'fp': []}
    order = np.random.default_rng(args.seed).permutation(len(dataset))[:20 * args.batch_size]
    for i in order:
        item = dataset[int(i)]
        src = item[-1]['source']
        if len(by_source[src]) < args.batch_size:
            by_source[src].append(item)
        if all(len(v) == args.batch_size for v in by_source.values()):
            break
    model.train(); front_end.eval()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    info = {'n_in': int(front_end.n_out), 'sources': {}}
    for src, items in by_source.items():
        if not items:
            print(f"dry run: no '{src}' item among {len(order)} items read")
            continue
        if len(items) == 1:
            items = items * 2                                                             # ZoomNeXt's pooled BatchNorm needs 2 items in training
        batch = seg_collate_fn(items)
        x = seg_inputs(front_end, batch, device)                                          # [B, N + 1, c, c]
        assert x.shape[1:] == (front_end.n_out + 1, args.canvas, args.canvas), f"'{src}' input {tuple(x.shape)}, canvas {args.canvas}"
        with torch.autocast(device.type, dtype=amp_dtype, enabled=device.type == 'cuda'):
            outputs = model(x, batch['box_xyxy'].to(device).float())
        assert outputs[0].shape == (len(items), 1, args.canvas, args.canvas), f"'{src}' main output {tuple(outputs[0].shape)}"
        total, items_loss = model.loss([o.float() for o in outputs], batch['mask'].to(device).float())
        total.backward()
        model.zero_grad(set_to_none=True)
        info['sources'][src] = tuple(x.shape)
        print(f"dry run '{src}': x {tuple(x.shape)}, {len(outputs)} output(s) {tuple(outputs[0].shape)}, "
              f"loss {float(total.detach()):.4f} {items_loss}")
    assert info['sources'], f"dry run: no training item found among {len(order)} items"
    info['peak_mem_gib'] = torch.cuda.max_memory_allocated(device) / 2 ** 30 if device.type == 'cuda' else 0.0
    print(f"dry run: peak GPU memory {info['peak_mem_gib']:.2f} GiB at batch {args.batch_size}, canvas {args.canvas}, "
          f"{args.seg_model} / {args.arm} ({front_end.n_out} + 1 channels)")
    return info


def main(args):
    # hyper-parameters: flags left at None come from --hpy (common keys, then the --seg_model section)
    cfg = load_cfg(args)
    args.name = args.name or f'seg_{args.seg_model}_{args.arm}_s{args.seed}'
    args.output_dir = args.output_dir or os.path.join('weights', args.name)
    args.crop_cache = args.crop_cache or os.path.join(args.data_path, 'crop_cache_seg')
    assert len(args.box_mix) == 3 and min(args.box_mix) >= 0 and abs(sum(args.box_mix) - 1.0) < 1e-6, \
        f"--box_mix must be 3 non-negative fractions summing to 1, got {args.box_mix}"
    assert args.amp_dtype in AMP_DTYPES, f"amp_dtype must be one of {sorted(AMP_DTYPES)}, got {args.amp_dtype!r}"
    amp_dtype = AMP_DTYPES[args.amp_dtype]
    device = torch.device(args.device if args.device == 'cpu' or torch.cuda.is_available() else 'cpu')

    # set random seed
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    # arm front end = the detector's own (spec §9 checks), and the crop cache that must match it
    index_path = os.path.join(args.crop_cache, 'index.json')
    assert os.path.isfile(index_path), f"{index_path} not found: build the crop cache first (bash_files/launch_seg_queue.sh)"
    with open(index_path) as f:
        index = json.load(f)
    front_end, det_args, dataset = build_arm_front_end(args, device)
    # spec §9: the crop cache must hold the same band window as the arm's detector ...
    assert [float(v) for v in index['band_range']] == [float(v) for v in det_args.band_range], \
        f"crop cache band range {index['band_range']} != detector {det_args.name} band range {det_args.band_range}"
    # ... and its training / val ROIs must be that detector's export (rgb uses the raw arm's ROIs)
    roi_arm = 'raw' if args.arm == 'rgb' else args.arm
    assert 'roi_files' in index and roi_arm in index['roi_files'], f"{index_path} has no ROI files of arm {roi_arm}"
    for split, p in index['roi_files'][roi_arm].items():
        assert os.path.basename(p) == f'rois_{det_args.name}_{split}.json', \
            f"crop cache {args.crop_cache} holds the {roi_arm} ROIs of {p}, not of the arm's detector {det_args.name}"
    del index                                                                            # the windows are read by HyperCOD_roi

    # create dataset and dataloader
    roi_kw = dict(roi_margin=args.roi_margin, roi_min=args.roi_min, canvas=args.canvas)
    dataset_train = HyperCOD_roi(args.crop_cache, 'train', args.arm, box_mix=tuple(args.box_mix), gt_jitter=args.gt_jitter,
                                 scale_aug=tuple(args.scale_aug), gain_aug=args.gain_aug, train=True, **roi_kw)
    dataset_val = HyperCOD_roi(args.crop_cache, 'val', args.arm, train=False, **roi_kw)
    assert len(dataset_train) > 0 and len(dataset_val) > 0, \
        f"empty split in {args.crop_cache}: {len(dataset_train)} train / {len(dataset_val)} val items"
    # drop_last: ZoomNeXt's pooled BatchNorm refuses a training batch of one item, which an epoch's last batch can be
    assert len(dataset_train) >= args.batch_size, f"{len(dataset_train)} training items < batch {args.batch_size}"
    print(f"{args.crop_cache}: {len(dataset_train)} training items per epoch, {len(dataset_val)} val items, arm {args.arm}")
    # persistent workers: re-spawning them for every epoch and val pass left the data wait at 41-48 % of a step
    pin, keep = device.type == 'cuda', args.num_workers > 0
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                               collate_fn=seg_collate_fn, pin_memory=pin, drop_last=True, persistent_workers=keep)
    loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                             collate_fn=seg_collate_fn, pin_memory=pin, drop_last=False, persistent_workers=keep)

    # stem fold (P, q): fitted once per run, or taken from the checkpoint being resumed / evaluated (which must be this run's)
    ckpt = torch.load(args.resume, map_location='cpu', weights_only=False) if args.resume else None
    if ckpt is not None:
        check_ckpt_args(ckpt['args'], args, f"--resume {args.resume}", train=not args.eval)
        P, q = ckpt['P'], ckpt['q']
    else:
        dataset_fit = HyperCOD_roi(args.crop_cache, 'train', args.arm, train=False, **roi_kw)   # deterministic training items
        P, q, _ = fit_arm_rgb_map(front_end, dataset_fit, dataset.wavelens, device, n_pixels=args.fit_pixels,
                                  n_items=args.fit_items, seed=args.seed, num_workers=args.num_workers)

    # build model
    n_in = front_end.n_out
    assert np.shape(P) == (3, n_in) and np.shape(q) == (3,), f"stem fold P {np.shape(P)} / q {np.shape(q)} for {n_in} arm channels"
    model = build_seg_model(args, n_in, P, q).to(device)
    if ckpt is not None:
        model.load_state_dict(ckpt['model'])
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"{args.seg_model} on arm {args.arm}: {n_in} + 1 input channels, {n_train / 1e6:.2f}M trainable of {n_all / 1e6:.1f}M parameters")

    # optimizer and scheduler
    optimizer = torch.optim.AdamW(model.param_groups(args), lr=args.lr, weight_decay=args.weight_decay)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf   # cosine per epoch, as main_det
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' and amp_dtype == torch.float16 else None   # bf16: none needed

    # pre-flight only: one batch per box source, nothing written
    if args.dry_run:
        return dry_run(model, front_end, dataset_train, args, device, amp_dtype)

    # output files: a fresh start moves a previous attempt's results / TensorBoard events / checkpoints to <path>.prev
    results_path = os.path.join(args.output_dir, f'results_{args.name}.txt')
    if not args.resume and not args.eval:
        moved = move_aside([results_path, os.path.join(args.runs_dir, args.name),
                            os.path.join(args.output_dir, 'model_best'), os.path.join(args.output_dir, 'model_last')])
        if moved:
            print(f"fresh start: moved a previous attempt to <path>.prev: {moved}")
    os.makedirs(args.output_dir, exist_ok=True)
    logger = TrainLogger(args, cfg)
    eval_kw = dict(amp_dtype=amp_dtype, size_edges=tuple(args.size_edges))

    # evaluate only
    if args.eval:
        assert ckpt is not None, "--eval needs --resume <main_seg checkpoint>"
        val = evaluate(model, front_end, loader_val, device, logger=logger, epoch=ckpt['epoch'], tag='val', **eval_kw)
        with open(results_path, 'a') as f:
            f.write(json.dumps({'eval': True, 'resume': args.resume, 'epoch': ckpt['epoch'], 'val': val}) + '\n')
        logger.finish()
        return results_path

    # resume: optimizer / scheduler / epoch, the best score, and the results lines up to the checkpoint's epoch
    start_epoch, best, best_epoch = 0, -1.0, -1
    if ckpt is not None and ckpt.get('optimizer'):
        optimizer.load_state_dict(ckpt['optimizer']); scheduler.load_state_dict(ckpt['lr_scheduler'])
        if scaler is not None and ckpt.get('scaler'):
            scaler.load_state_dict(ckpt['scaler'])
        # carry the best score over, or the first post-resume epoch would overwrite model_best with a worse model ...
        start_epoch, best, best_epoch = ckpt['epoch'] + 1, float(ckpt['best']), int(ckpt['best_epoch'])
        # ... and a model_best the interrupted run saved after this checkpoint holds a better score still
        later = later_best(args, ckpt['epoch'])
        if later is not None:
            print(f"model_best of epoch {later[1]} (score {later[0]:.4f}) is later than the checkpoint: kept as the best so far")
            best, best_epoch = later
    if ckpt is not None:
        dropped = truncate_results(results_path, start_epoch - 1)
        print(f"resume at epoch {start_epoch}: dropped {dropped} results line(s) (later epochs, final lines, cut lines)")

    # train
    print(f"Start training from epoch {start_epoch}, best so far {best:.4f}"); start = time.time()
    val = None
    for epoch in range(start_epoch, args.epochs):
        train_stats = train_one_epoch(model, front_end, loader_train, optimizer, device, epoch, scaler=scaler, accumulate=args.accumulate,
                                      max_norm=args.max_norm, logger=logger, print_freq=args.print_freq, amp_dtype=amp_dtype,
                                      total_epochs=args.epochs)
        scheduler.step()
        val = evaluate(model, front_end, loader_val, device, logger=logger, epoch=epoch, tag='val', **eval_kw)
        score = val_score(val)
        logger.scalar('val/score', score, epoch)
        if score > best:                                         # before model_last, so model_best includes this epoch
            best, best_epoch = score, epoch
            save_checkpoint(os.path.join(args.output_dir, 'model_best'), model, optimizer, scaler, scheduler, epoch, args, best, best_epoch, P, q)
        # the epoch's line before model_last: a crash between the two leaves a line that the resume drops, never a gap
        with open(results_path, 'a') as f:
            f.write(json.dumps({'epoch': epoch, 'train': train_stats, 'val': val, 'score': score, 'best': best, 'best_epoch': best_epoch}) + '\n')
        if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
            save_checkpoint(os.path.join(args.output_dir, 'model_last'), model, optimizer, scaler, scheduler, epoch, args, best, best_epoch, P, q)
    print(f"Training time {datetime.timedelta(seconds=int(time.time() - start))}, best val mean(S, Fw) {best:.4f} at epoch {best_epoch}")

    # final: model_best re-evaluated after a reload (checks the checkpoint round trip); the queue skips runs with this line
    best_ckpt = torch.load(os.path.join(args.output_dir, 'model_best'), map_location='cpu', weights_only=False)
    model.load_state_dict(best_ckpt['model'])
    val_best = evaluate(model, front_end, loader_val, device, logger=logger, epoch=args.epochs, tag='val_best', **eval_kw)
    with open(results_path, 'a') as f:
        f.write(json.dumps({'epoch': best_ckpt['epoch'], 'final': True, 'best_epoch': best_ckpt['epoch'], 'best': best,
                            'val': val_best, 'val_last': val}) + '\n')
    logger.finish()
    return results_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Stage-2 segmentation', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
