'''
Stage-2 segmentation models (spec §5.4) and the arm front end that feeds them.

  build_front_end(arm, det_args, dataset)  p99-scaled 133-band canvas [B, 133, c, c] -> the arm's channels [B, N, c, c]
  build_seg_model(args, n_in, P, q)        SAM2UNetSeg | SAM2BoxSeg | ZoomNeXtSeg, all with one API:
      forward(x [B, n_in + 1, c, c], box_xyxy [B, 4] canvas px) -> list of logits [B, 1, c, c]
                                            (index 0 = main output, deep-supervision outputs after it)
      loss(outputs, mask [B, 1, c, c])  -> (total tensor, {name: float})
      param_groups(args)                -> AdamW param groups, the folded stem at args.stem_lr
      set_progress(t)                   -> training progress t in [0, 1] (ZoomNeXt's UAL weight; a no-op elsewhere)

The input x is cat(front_end(img) * valid, box_map) (train_eval_seg.seg_inputs); (P, q) is the arm's least-squares map
from its channels to the ImageNet-normalised pseudo-RGB render (seg_stem.fit_rgb_map), folded into each model's
pretrained RGB stem (seg_stem.fold_stem) so the network sees the pseudo-RGB image at initialisation (spec §3).
'''
import os
import numpy as np
import torch
import torch.nn as nn

from models.filter_bank import build_filter_bank
from models.seg_stem import pseudo_rgb, imagenet_normalize, fold_stem
from third_party.sam2_unet.sam2unet import SAM2UNet, build_hiera_trunk, structure_loss

ARMS = ('raw', 'ec10', 'ec24', 'rgb')
ARM_PCA = {'ec10': 10, 'ec24': 24}                    # whitened channels of the arm's detector (--pca-channels)

SAM2UNET_CFG = 'configs/sam2/sam2_hiera_l.yaml'       # pip sam2's SAM2 v1 Hiera-L config (SAM2-UNet uses v1, not 2.1)
SAM2UNET_CKPT = 'weights/pretrained/sam2_hiera_large.pt'


class ArmFrontEnd(nn.Module):
    '''
    The arm's fixed sensor model on the GPU: raw / ec10 / ec24 = the detector's FilterBank (standardised raw bands, or
    the EC readings whitened to 10 / 24 channels), rgb = the pseudo-RGB render normalised with the ImageNet mean/std.
    Nothing here trains: every parameter is frozen and the module stays in eval mode whatever train() is called with.
    The computation runs in fp32 with autocast off. Under the bf16 autocast of training the whitened channels would
    otherwise be off by up to 0.34 (ec10) / 0.43 (ec24) standardised units on a real window (recon, fb_check.py); in
    fp32 they are the reference values (the detectors themselves saw fp16-rounded channels).
    '''

    def __init__(self, arm, fb=None, wavelens=None):
        super(ArmFrontEnd, self).__init__()
        assert arm in ARMS, f"arm must be one of {ARMS}, got {arm!r}"
        self.arm = arm
        self.fb = fb
        if arm == 'rgb':
            assert wavelens is not None, "the rgb arm needs the band centres (dataset.wavelens)"
            self.wavelens = np.asarray(wavelens, dtype=np.float64)                         # [133] nm
            self.n_out = 3
        else:
            assert fb is not None, f"arm '{arm}' needs its detector's FilterBank"
            self.n_out = fb.n_channels
        for p in self.parameters():
            p.requires_grad_(False)
        super(ArmFrontEnd, self).train(False)

    def train(self, mode=True):
        # a fixed sensor model: always eval (FilterBank has no train-mode behaviour that Stage 2 wants)
        return super(ArmFrontEnd, self).train(False)

    def forward(self, x):
        '''x: [B, 133, h, w] p99-scaled bands, any float dtype -> [B, N, h, w] float32'''
        with torch.autocast(x.device.type, enabled=False):
            x = x.float()                                                                  # [B, 133, h, w]
            if self.fb is None:
                y = imagenet_normalize(pseudo_rgb(x, self.wavelens))                       # [B, 3, h, w]
            else:
                y = self.fb(x)                                                             # [B, N, h, w]
        return y.float()


def check_filter_bank(fb, det_state):
    '''
    Assert that a rebuilt FilterBank equals the one stored in the detector checkpoint (det_state = ckpt['model'], keys
    'filter_bank.<buffer>'), so the arm's channels are bit-identical to what its detector saw (spec §3, §9).
    '''
    ref = {k[len('filter_bank.'):]: v for k, v in det_state.items() if k.startswith('filter_bank.')}
    own = fb.state_dict()
    assert set(ref) == set(own), f"filter bank tensors differ from the detector checkpoint: rebuilt {sorted(own)}, checkpoint {sorted(ref)}"
    bad = [k for k in own if own[k].shape != ref[k].shape or not torch.equal(own[k].cpu(), ref[k].cpu())]
    assert not bad, f"rebuilt filter bank != detector checkpoint in {bad} (wrong --det_ckpt for this arm, or changed band stats?)"


def build_front_end(arm, det_args, dataset, det_state=None):
    '''
    The arm's front end, rebuilt from its detector's arguments exactly as Stage 1 built it (filter_bank.build_filter_bank).
      arm        'raw' | 'ec10' | 'ec24' | 'rgb'
      det_args   the detector checkpoint's args as a Namespace (main_det_rois.detector_args: parser defaults overlaid
                 with ckpt['args']); rgb uses the raw detector raw133_A (its ROIs; only the band centres matter)
      dataset    a HyperCOD_data built with main_det.dataset_kwargs(det_args): it supplies the band statistics, the EC
                 response matrix of the detector's voltages and wavelens
      det_state  optional detector state dict (ckpt['model']): the rebuilt FilterBank must equal its 'filter_bank.*'
    Returns an ArmFrontEnd with .n_out = N (133 / 10 / 24 / 3).
    '''
    assert arm in ARMS, f"arm must be one of {ARMS}, got {arm!r}"
    raw = bool(getattr(det_args, 'raw_bands', False))
    k = int(getattr(det_args, 'pca_channels', 0) or 0)
    if arm in ('raw', 'rgb'):
        assert raw, f"arm '{arm}' needs the raw-band detector (--raw-bands, raw133_A); this detector uses EC readings ({k} whitened channels)"
    else:
        assert not raw, f"arm '{arm}' needs an EC detector; this one is the raw-band detector"
        assert k == ARM_PCA[arm], f"arm '{arm}' needs a detector with --pca-channels {ARM_PCA[arm]}, this one has {k}"
    if arm == 'rgb':
        fe = ArmFrontEnd(arm, wavelens=dataset.wavelens)
    else:
        fb, volts = build_filter_bank(det_args, dataset)
        if det_state is not None:
            check_filter_bank(fb, det_state)
        fe = ArmFrontEnd(arm, fb=fb)
        expect = dataset.n_bands if arm == 'raw' else ARM_PCA[arm]
        assert fe.n_out == expect, f"arm '{arm}' front end gives {fe.n_out} channels, expected {expect}"
    print(f"front end '{arm}': {fe.n_out} channels" + (" (pseudo-RGB, ImageNet-normalised)" if arm == 'rgb' else
                                                      f" (FilterBank {fe.fb.n_readings} readings -> {fe.fb.n_channels} channels)"))
    return fe


def param_groups_by_name(model, args, stem_prefix, lr=None):
    '''
    AdamW groups over the trainable parameters, by name: 'decay' (weights, args.weight_decay), 'no_decay' (biases and
    1-D tensors such as norms, no decay, as main_det.build_optimizer) and 'stem' (the folded input conv, every tensor
    whose name starts with stem_prefix, at args.stem_lr without decay: it starts from the pretrained RGB kernel, and
    main_det keeps its rebuilt first conv out of the decay too). lr defaults to args.lr. Empty groups are dropped;
    each group carries a 'name' for logging.
    '''
    lr = args.lr if lr is None else lr
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    stem = [p for n, p in named if n.startswith(stem_prefix)]
    assert stem, f"no trainable parameter under '{stem_prefix}'"
    rest = [(n, p) for n, p in named if not n.startswith(stem_prefix)]
    decay = [p for n, p in rest if p.ndim > 1 and not n.endswith('.bias')]
    no_decay = [p for n, p in rest if p.ndim <= 1 or n.endswith('.bias')]
    groups = [{'name': 'decay', 'params': decay, 'lr': lr, 'weight_decay': args.weight_decay},
              {'name': 'no_decay', 'params': no_decay, 'lr': lr, 'weight_decay': 0.0},
              {'name': 'stem', 'params': stem, 'lr': args.stem_lr, 'weight_decay': 0.0}]
    return [g for g in groups if g['params']]


class SAM2UNetSeg(nn.Module):
    '''
    SAM2-UNet (vendored, third_party/sam2_unet): the SAM2 v1 Hiera trunk, frozen, with a trainable bottleneck adapter in
    front of every block, four RFB modules and a U-Net decoder. The trunk's 3-channel patch-embed conv
    (Conv2d(3, 144, 7, stride 4, pad 3) for Hiera-L) is replaced by the folded (n_in + 1)-channel stem, trainable.
    The box enters only through the box channel of x, so box_xyxy is unused.
    Outputs, all [B, 1, c, c] logits: [main (stride-4 head), side1 (stride 16), side2 (stride 8)].
    Loss: SAM2-UNet's structure loss (weighted BCE + weighted IoU) summed over the three outputs.
    args: sam2unet_cfg (hydra config of the pip sam2 package, default SAM2 v1 Hiera-L; the tests pass
          'configs/sam2/sam2_hiera_t.yaml'), sam2unet_ckpt (SAM2 v1 checkpoint, default SAM2UNET_CKPT; '' or 'none' keeps
          the trunk random, for tests only), lr, stem_lr, weight_decay.
    '''
    STEM = 'net.encoder.patch_embed.proj.'

    def __init__(self, args, n_in, P, q):
        super(SAM2UNetSeg, self).__init__()
        cfg = getattr(args, 'sam2unet_cfg', None) or SAM2UNET_CFG
        ckpt = getattr(args, 'sam2unet_ckpt', None)
        ckpt = SAM2UNET_CKPT if ckpt is None else ckpt
        ckpt = None if ckpt in ('', 'none') else ckpt
        assert ckpt is None or os.path.exists(ckpt), f"SAM2 v1 checkpoint {ckpt} not found: run bash bash_files/setup_third_party.sh"
        self.n_in = n_in
        self.net = SAM2UNet(build_hiera_trunk(cfg, ckpt))                       # trunk frozen; adapters, RFB, decoder trainable
        self.net.up4.requires_grad_(False)                                      # defined but never called upstream: keep it out of the optimiser
        old = self.net.encoder.patch_embed.proj                                 # pretrained RGB stem Conv2d(3, C, 7, 4, 3)
        assert old.in_channels == 3, f"expected the RGB patch-embed conv, got {old}"
        self.net.encoder.patch_embed.proj = fold_stem(old, P, q, n_extra=1)    # Conv2d(n_in + 1, C, 7, 4, 3), box slice zero
        self.net.encoder.patch_embed.proj.requires_grad_(True)                  # swapped in AFTER the trunk freeze: trainable
        print(f"SAM2UNetSeg: {cfg}, trunk {'from ' + ckpt if ckpt else 'RANDOM (no checkpoint)'}, stem {self.net.encoder.patch_embed.proj}")

    def forward(self, x, box_xyxy=None):
        '''x: [B, n_in + 1, c, c] (arm channels + box channel), c a multiple of 32 -> [main, side1, side2] logits [B, 1, c, c]'''
        assert x.shape[1] == self.n_in + 1, f"expected {self.n_in} + 1 input channels, got {x.shape[1]}"
        # Hiera tiles its 8 x 8 window position embedding over the stride-4 grid, so the side must be a multiple of 32
        assert x.shape[-2] % 32 == 0 and x.shape[-1] % 32 == 0, f"canvas side must be a multiple of 32, got {tuple(x.shape[-2:])}"
        out, out1, out2 = self.net(x)
        return [out, out1, out2]

    def loss(self, outputs, mask):
        '''structure_loss on every output, in fp32 (outputs come out bf16 under autocast); mask [B, 1, c, c] float in {0, 1}'''
        items, total = {}, 0.0
        with torch.autocast(mask.device.type, enabled=False):
            for name, o in zip(('main', 's16', 's8'), outputs):
                l = structure_loss(o.float(), mask.float())
                items[f'loss_{name}'] = float(l.detach())
                total = total + l
        items['loss'] = float(total.detach())
        return total, items

    def param_groups(self, args):
        # adapters, RFB and decoder at args.lr (spec §7: 1e-3), the folded stem at args.stem_lr (1e-4)
        return param_groups_by_name(self, args, self.STEM)

    def set_progress(self, t):
        pass


# --seg_model name -> class. Task 9 adds 'sam2box': SAM2BoxSeg and Task 10 'zoomnext': ZoomNeXtSeg to this one line.
SEG_MODELS = {'sam2unet': SAM2UNetSeg}


def build_seg_model(args, n_in, P, q):
    '''
    The --seg_model network for n_in arm channels (+ 1 box channel), its stem folded from (P [3, n_in], q [3]).
    Every class is constructed as Cls(args, n_in, P, q) and shares the API of the module docstring.
    '''
    assert args.seg_model in SEG_MODELS, f"--seg_model must be one of {sorted(SEG_MODELS)}, got {args.seg_model!r}"
    P, q = np.asarray(P, dtype=np.float32), np.asarray(q, dtype=np.float32)
    assert P.shape == (3, n_in) and q.shape == (3,), f"stem map must be P [3, {n_in}], q [3]; got {P.shape}, {q.shape}"
    model = SEG_MODELS[args.seg_model](args, n_in, P, q)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"{args.seg_model}: {n_in} + 1 input channels, {n_train / 1e6:.2f}M trainable of {n_all / 1e6:.1f}M parameters")
    return model
