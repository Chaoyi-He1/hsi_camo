import copy
import math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from util.misc import MetricLogger, SmoothedValue
from models.ec_yolo import decode_predictions
from train_eval.box_metrics import BoxMetrics, filter_to_operating_point
from data_loader.boxes import expand_box, boxes_from_mask


def random_flips(img, bboxes, batch_idx, p=0.5, generator=None):
    '''Per-sample random horizontal/vertical flips of the frames and their normalised (cx, cy, w, h) boxes.'''
    img = img.clone(); bboxes = bboxes.clone(); flags = []
    for b in range(img.shape[0]):
        h = bool(torch.rand((), generator=generator) < p); v = bool(torch.rand((), generator=generator) < p)
        flags.append((h, v))
        sel = batch_idx == b
        if h:
            img[b] = img[b].flip(-1); bboxes[sel, 0] = 1.0 - bboxes[sel, 0]
        if v:
            img[b] = img[b].flip(-2); bboxes[sel, 1] = 1.0 - bboxes[sel, 1]
    return img, bboxes, flags


def scale_crop_boxes(bboxes, scale_hw, offset, H, W, min_visible=0.3):
    '''
    Normalised (cx, cy, w, h) boxes of one frame after it was rescaled to scale_hw = (Hs, Ws) pixels and either cropped
    (offset = top-left of the H x W crop inside the scaled frame, Hs >= H) or placed on an H x W canvas (offset = where the
    scaled frame's top-left landed, Hs <= H; pass it negated, see random_scale_crop). Boxes are clipped to the frame; keep[k]
    is False for a box that kept less than min_visible of its area, so the caller drops its cls / batch_idx rows too.
    '''
    Hs, Ws = scale_hw; oy, ox = offset
    cx, cy = bboxes[:, 0] * Ws - ox, bboxes[:, 1] * Hs - oy                          # pixel coords in the output frame
    w, h = bboxes[:, 2] * Ws, bboxes[:, 3] * Hs
    x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    x1c, y1c, x2c, y2c = x1.clamp(0, W), y1.clamp(0, H), x2.clamp(0, W), y2.clamp(0, H)
    visible = ((x2c - x1c).clamp(min=0) * (y2c - y1c).clamp(min=0)) / (w * h).clamp(min=1e-6)
    keep = visible >= min_visible
    out = torch.stack([(x1c + x2c) / 2 / W, (y1c + y2c) / 2 / H, (x2c - x1c) / W, (y2c - y1c) / H], dim=1)
    return out[keep], keep


def random_scale_crop(img, bboxes, batch_idx, cls, scale_range=(0.67, 1.5), min_visible=0.3, generator=None):
    '''
    Per-sample scale jitter at native resolution: the frame is rescaled by s ~ log-uniform(scale_range) (bilinear), then a
    random H x W crop of it (s > 1) or a random placement of it on an H x W canvas filled with the frame's per-channel mean
    (s < 1; after standardisation that is a flat mid-grey border). Boxes follow (scale_crop_boxes); cls and batch_idx rows of
    dropped boxes are removed with them. Returns img, bboxes, batch_idx, cls, scales. Works on the fp16 cube on the GPU: one
    frame at a time, and the mean is accumulated in fp32 without an fp32 copy of the frame.
    '''
    B, C, H, W = img.shape
    lo, hi = scale_range
    assert 0 < lo <= hi, f"scale_range must satisfy 0 < lo <= hi, got {scale_range}"
    out = img.clone()
    boxes_out, idx_out, cls_out, scales = [], [], [], []
    for b in range(B):
        s = float(torch.exp(torch.empty((), dtype=torch.float64).uniform_(math.log(lo), math.log(hi), generator=generator)))
        scales.append(s)
        sel = batch_idx == b
        bb = bboxes[sel]
        if abs(s - 1.0) < 1e-3:
            boxes_out.append(bb); idx_out.append(batch_idx[sel]); cls_out.append(cls[sel]); continue
        Hs, Ws = max(int(round(H * s)), 1), max(int(round(W * s)), 1)
        x = img[b:b + 1]
        x = F.interpolate(x.float() if x.device.type == 'cpu' else x, size=(Hs, Ws), mode='bilinear', align_corners=False).to(img.dtype)
        if s > 1.0:                                                                           # crop a native-size window
            oy = int(torch.randint(0, Hs - H + 1, (), generator=generator)); ox = int(torch.randint(0, Ws - W + 1, (), generator=generator))
            out[b] = x[0, :, oy:oy + H, ox:ox + W]
            offset = (oy, ox)
        else:                                                                                 # place on a mean-filled canvas
            oy = int(torch.randint(0, H - Hs + 1, (), generator=generator)); ox = int(torch.randint(0, W - Ws + 1, (), generator=generator))
            out[b] = img[b].mean(dim=(1, 2), dtype=torch.float32).to(img.dtype)[:, None, None]
            out[b, :, oy:oy + Hs, ox:ox + Ws] = x[0]
            offset = (-oy, -ox)
        bb2, keep = scale_crop_boxes(bb, (Hs, Ws), offset, H, W, min_visible)
        boxes_out.append(bb2); idx_out.append(batch_idx[sel][keep]); cls_out.append(cls[sel][keep])
    cat = lambda parts, like: torch.cat(parts) if parts else like[:0]
    return out, cat(boxes_out, bboxes), cat(idx_out, batch_idx), cat(cls_out, cls), scales


def random_gain(img, gain=0.0, tilt=0.0, generator=None):
    '''
    Per-sample photometric jitter of the input cube: a global gain exp(U(-gain, gain)) (illumination level) times a spectral
    tilt 1 + a * ramp(-1..1 over the channels), a ~ U(-tilt, tilt) (illuminant colour / sensor response drift). 0 = off.
    '''
    if gain <= 0 and tilt <= 0:
        return img
    B, C = img.shape[:2]
    out = img.clone()
    ramp = torch.linspace(-1.0, 1.0, C, device=img.device) if C > 1 else torch.zeros(1, device=img.device)
    for b in range(B):
        g = float(torch.exp(torch.empty((), dtype=torch.float64).uniform_(-gain, gain, generator=generator))) if gain > 0 else 1.0
        a = float(torch.empty((), dtype=torch.float64).uniform_(-tilt, tilt, generator=generator)) if tilt > 0 else 0.0
        factor = (g * (1.0 + a * ramp)).to(img.dtype)                                      # [C]
        out[b] = img[b] * factor[:, None, None]
    return out


class ModelEMA:
    '''
    Exponential moving average of the model weights, updated once per optimizer step with the ultralytics-style warm-up
    decay_t = decay * (1 - exp(-t / tau)), so the average tracks the raw weights closely at the start. Floating-point
    buffers (BatchNorm statistics, the filter-bank tensors) are averaged too; integer buffers are copied. The averaged copy
    is evaluated and checkpointed in place of the raw weights (main_det stores the raw ones under 'model_raw').
    '''

    def __init__(self, model, decay=0.992, tau=128.0):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay, self.tau, self.updates = float(decay), float(tau), 0

    def current_decay(self):
        return self.decay * (1.0 - math.exp(-self.updates / self.tau)) if self.tau > 0 else self.decay

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = self.current_decay()
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            src = msd[k].detach()
            if v.dtype.is_floating_point:
                v.mul_(d).add_(src.to(v.dtype), alpha=1.0 - d)
            else:
                v.copy_(src)

    def state_dict(self):
        return {'updates': self.updates, 'decay': self.decay, 'tau': self.tau}

    def load_state_dict(self, state):
        self.updates = int(state.get('updates', 0)); self.decay = float(state.get('decay', self.decay)); self.tau = float(state.get('tau', self.tau))


def train_one_epoch(model, data_loader, optimizer, device, epoch, max_norm=10.0, scaler=None, accumulate=1, logger=None, flip=True, print_freq=10,
                    aug=None, ema=None):
    '''
    One pass over the loader. aug = {'scale': (lo, hi) | None, 'min_visible': f, 'gain': g, 'tilt': t} switches the
    scale / crop and photometric jitter on (random_scale_crop, random_gain; flips stay separate); ema is a ModelEMA
    updated after every optimizer step.
    '''
    model.train()
    metric_logger = MetricLogger(delimiter="; ")
    for k in ['loss', 'box_loss', 'cls_loss', 'l1_loss', 'gate_entropy', 'contain_loss']:
        metric_logger.add_meter(k, SmoothedValue(window_size=10, fmt='{value:.4f}'))
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
    n_steps = len(data_loader)
    is_ddp = isinstance(model, torch.nn.parallel.DistributedDataParallel)
    core = model.module if is_ddp else model
    loss_fn = (lambda b: model(batch=b)) if is_ddp else core.loss              # DDP: go through forward so gradients sync
    aug = aug or {}
    optimizer.zero_grad(set_to_none=True)
    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        img = batch['img'].to(device, non_blocking=True)                                  # [B, C, H, W] fp16
        bboxes, batch_idx, cls = batch['bboxes'].to(device), batch['batch_idx'].to(device), batch['cls'].to(device)
        if flip:
            img, bboxes, _ = random_flips(img, bboxes, batch_idx)
        if aug.get('scale'):
            img, bboxes, batch_idx, cls, _ = random_scale_crop(img, bboxes, batch_idx, cls, aug['scale'], aug.get('min_visible', 0.3))
        if aug.get('gain', 0) > 0 or aug.get('tilt', 0) > 0:
            img = random_gain(img, aug.get('gain', 0.0), aug.get('tilt', 0.0))
        with torch.autocast(device.type, enabled=scaler is not None):
            total, items = loss_fn({'img': img, 'bboxes': bboxes, 'batch_idx': batch_idx, 'cls': cls})
        loss = total / accumulate
        (scaler.scale(loss) if scaler is not None else loss).backward()
        if (i + 1) % accumulate == 0 or i + 1 == n_steps:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            if scaler is not None:
                scaler.step(optimizer); scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(core)
        lr = optimizer.param_groups[0]['lr']
        metric_logger.update(loss=float(total.detach()), lr=lr, **{k: items.get(k, 0.0) for k in ['box_loss', 'cls_loss', 'l1_loss', 'gate_entropy', 'contain_loss']})
        if logger is not None:
            step = epoch * n_steps + i
            logger.scalars({'loss': float(total.detach()), 'lr': lr, **items}, step, prefix='train/')
    core.end_epoch()
    return {k: m.global_avg for k, m in metric_logger.meters.items()}


def false_colour(y):
    '''Three channels spread over the response stack -> uint8 RGB (percentile-stretched) for logging.'''
    idx = [0, y.shape[0] // 2, y.shape[0] - 1] if y.shape[0] >= 3 else [0] * 3
    rgb = y.detach()[idx].float().cpu().permute(1, 2, 0).numpy()   # pick 3 of N channels first: N=344 at native res is 2.9 GB in fp32
    lo, hi = np.percentile(rgb, 1), np.percentile(rgb, 99)
    return (np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)


def draw_boxes(img_uint8, boxes, color, width=3, dashed=False):
    im = Image.fromarray(np.ascontiguousarray(img_uint8)); d = ImageDraw.Draw(im)
    for b in np.asarray(boxes).reshape(-1, 4):
        x1, y1, x2, y2 = [float(v) for v in b]
        x1, x2 = sorted((x1, x2)); y1, y2 = sorted((y1, y2))       # a degenerate/inverted box (early epochs) must not crash the eval
        if not dashed:
            d.rectangle([x1, y1, max(x2 - 1, x1), max(y2 - 1, y1)], outline=color, width=width)
        else:
            pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]
            for (ax, ay), (bx, by) in zip(pts[:-1], pts[1:]):
                n = max(int(math.hypot(bx - ax, by - ay) // 8), 1)
                for s in range(0, n, 2):
                    d.line([(ax + (bx - ax) * s / n, ay + (by - ay) * s / n), (ax + (bx - ax) * (s + 1) / n, ay + (by - ay) * (s + 1) / n)], fill=color, width=width)
    return np.asarray(im)


@torch.no_grad()
def evaluate(model, data_loader, device, conf_thres=0.001, iou_thres=0.6, max_det=300, roi_margin=1.5, roi_min=256,
             min_area=100, roi_conf=0.25, roi_topk=5, logger=None, epoch=0, tag='val', n_images_log=4, amp=True):
    model.eval()
    metrics = BoxMetrics(roi_margin=roi_margin, roi_min=roi_min, min_area=min_area, roi_conf=roi_conf, roi_topk=roi_topk)
    metric_logger = MetricLogger(delimiter="; ")
    logged = 0
    for batch in metric_logger.log_every(data_loader, 10, f'Eval {tag}:'):
        img = batch['img'].to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=amp and device.type == 'cuda'):
            out = model(img)
        dets = decode_predictions(out, conf_thres=conf_thres, iou_thres=iou_thres, max_det=max_det,
                                  end2end=getattr(model.yolo, 'end2end', None))
        H, W = img.shape[-2:]
        for b, mask in enumerate(batch['masks']):
            d = dets[b].copy()
            d[:, [0, 2]] = d[:, [0, 2]].clip(0, W); d[:, [1, 3]] = d[:, [1, 3]].clip(0, H)   # padding is bottom/right: coords unchanged
            metrics.update(mask, d)
            if logger is not None and logged < n_images_log:
                # this extra filter-bank forward must not advance the bank's eval-mode read-noise sequence, or the
                # following frames would see different noise from main_det_compare's pass over the same frames
                state = model.filter_bank.noise_state()
                with torch.autocast(device.type, enabled=amp and device.type == 'cuda'):
                    y = model.filter_bank(model.pad_to_stride(img[b:b + 1])[0])[0, :, :H, :W]
                model.filter_bank.set_noise_state(state)
                pic = false_colour(y)
                pic = draw_boxes(pic, boxes_from_mask(mask, min_area=min_area), (0, 255, 0))
                top = filter_to_operating_point(d, roi_conf, roi_topk)       # the boxes the ROI export would keep
                pic = draw_boxes(pic, top[:, :4], (255, 0, 0))
                pic = draw_boxes(pic, [expand_box(p, roi_margin, roi_min, H, W) for p in top[:, :4]], (255, 255, 0), dashed=True)
                logger.image(f'{tag}/frame_{batch["names"][b]}', pic, epoch); logged += 1
    summary = metrics.summary()
    if logger is not None:
        logger.scalars(summary, epoch, prefix=f'{tag}/')
        logger.histogram(f'{tag}/coverage_raw_hist', [r['coverage_raw'] for r in metrics.records], epoch)
        logger.histogram(f'{tag}/tightness_hist', [r['tightness'] for r in metrics.records if not math.isnan(r['tightness'])] or [0.0], epoch)
    print(f"{tag} epoch {epoch}: " + ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in summary.items()))
    return summary
