import math
import numpy as np
import torch
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


def train_one_epoch(model, data_loader, optimizer, device, epoch, max_norm=10.0, scaler=None, accumulate=1, logger=None, flip=True, print_freq=10):
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
    optimizer.zero_grad(set_to_none=True)
    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        img = batch['img'].to(device, non_blocking=True)                                  # [B, C, H, W] fp16
        bboxes, batch_idx = batch['bboxes'].to(device), batch['batch_idx'].to(device)
        if flip:
            img, bboxes, _ = random_flips(img, bboxes, batch_idx)
        with torch.autocast(device.type, enabled=scaler is not None):
            total, items = loss_fn({'img': img, 'bboxes': bboxes, 'batch_idx': batch_idx, 'cls': batch['cls'].to(device)})
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
        if not dashed:
            d.rectangle([x1, y1, x2 - 1, y2 - 1], outline=color, width=width)
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
                with torch.autocast(device.type, enabled=amp and device.type == 'cuda'):
                    y = model.filter_bank(model.pad_to_stride(img[b:b + 1])[0])[0, :, :H, :W]
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
