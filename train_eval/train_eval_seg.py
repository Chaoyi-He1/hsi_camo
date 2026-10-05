'''
Stage-2 training / evaluation loops (spec §5.5): train_one_epoch, evaluate (ROI level) and paste_back (full frame).

Every model of models/seg_models.py shares one API: forward(x [B, N + 1, c, c], box_xyxy [B, 4]) -> list of logits
[B, 1, c, c] (index 0 = main output, deep-supervision outputs after it), loss(outputs, mask) -> (total, items),
param_groups(args), and optionally set_progress(t) (ZoomNeXt's uncertainty-aware loss ramps with the training progress
t in [0, 1]). The arm's front end (raw: standardised bands; ec10 / ec24: the detector's whitened EC readings; rgb: the
ImageNet-normalised pseudo-RGB render) is fixed: it is the detector's own, so it runs without gradients.
'''
import math
import numpy as np
import torch

from util.misc import MetricLogger, SmoothedValue
from data_loader.roi_crops import canvas_to_roi
from train_eval.seg_metrics import SegMetrics

PRINT_KEYS = ('S', 'E_mean', 'E_max', 'E_adp', 'Fw', 'F_adp', 'MAE', 'IoU', 'n', 'fp_false_mask_rate', 'n_fp')   # the console line of evaluate


def seg_inputs(front_end, batch, device):
    '''
    Model input of one seg_collate_fn batch: x [B, N + 1, c, c] float32 = cat(front_end(img) * valid, box_map).
    The front end runs in float32 with autocast OFF, whatever the caller's autocast context: FilterBank.forward computes
    in its input's dtype, and on a real window bf16 gives errors of 0.34 (ec10) / 0.43 (ec24) in standardised units and
    fp16 already 0.42 on ec24's whitened channel 20 (recon), so the 133-band fp16 crop is cast to float32 first.
    Multiplying by valid zeroes the canvas padding: there the standardised channels would read -mean/std (the padding
    is 0 in band space), so after the mask the padding is 0 = the training mean in every arm, like the box channel.
    No gradient flows into the front end (it is the detector's fixed sensor model).
    '''
    img = batch['img'].to(device, non_blocking=True)                                     # [B, 133, c, c] fp16, p99-scaled
    valid = batch['valid'].to(device, non_blocking=True).float()                         # [B, 1, c, c] 1 = ROI content
    box_map = batch['box_map'].to(device, non_blocking=True).float()                     # [B, 1, c, c] 1 = pre-expansion box
    with torch.no_grad(), torch.autocast(device.type, enabled=False):
        z = front_end(img.float()).float()                                               # [B, N, c, c] float32
    assert z.shape[1] == front_end.n_out, f"front end returned {z.shape[1]} channels, its n_out is {front_end.n_out}"
    assert z.shape[-2:] == box_map.shape[-2:], f"front end output {tuple(z.shape)} does not match the canvas {tuple(box_map.shape)}"
    return torch.cat([z * valid, box_map], dim=1)                                        # [B, N + 1, c, c]


def train_one_epoch(model, front_end, data_loader, optimizer, device, epoch, scaler=None, accumulate=1, max_norm=1.0,
                    logger=None, print_freq=10, amp=True, amp_dtype=torch.bfloat16, total_epochs=None):
    '''
    One pass over data_loader (seg_collate_fn batches). Same structure as train_eval_det.train_one_epoch: MetricLogger,
    loss / accumulate, clipping at max_norm on the accumulated gradient, an optimizer step every `accumulate` batches and
    at the last batch, per-step scalars to logger under 'train/' against the global step.
    Mixed precision: the model forward runs under torch.autocast(amp_dtype) when amp is set and the device is CUDA
    (bf16 by default, spec §7). bf16 needs no GradScaler, so unlike the detector loop (autocast iff a scaler exists)
    autocast is driven by `amp`; pass a torch.amp.GradScaler only with amp_dtype=torch.float16. The loss is computed in
    float32 outside autocast on the float32-cast logits.
    total_epochs: when given and the model has set_progress, it receives the training progress
    t = (epoch + i / n_steps) / total_epochs in [0, 1) before every forward (ZoomNeXt's UAL coefficient).
    Returns {meter: global average} with loss, lr and every item of model.loss.
    '''
    model.train()
    front_end.eval()                                                                     # fixed sensor model, eval-mode noise path
    metric_logger = MetricLogger(delimiter="; ")
    metric_logger.add_meter('loss', SmoothedValue(window_size=10, fmt='{value:.4f}'))
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
    n_steps = len(data_loader)
    use_amp = bool(amp) and device.type == 'cuda'
    params = [p for p in model.parameters() if p.requires_grad]                          # clip only what is trained (frozen trunks have no grad)
    optimizer.zero_grad(set_to_none=True)
    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if total_epochs and hasattr(model, 'set_progress'):
            model.set_progress((epoch + i / n_steps) / total_epochs)
        x = seg_inputs(front_end, batch, device)                                         # [B, N + 1, c, c] float32
        mask = batch['mask'].to(device, non_blocking=True).float()                       # [B, 1, c, c] GT union inside the ROI
        box_xyxy = batch['box_xyxy'].to(device, non_blocking=True).float()               # [B, 4] canvas px (box prompt)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            outputs = model(x, box_xyxy)                                                 # list of [B, 1, c, c] logits
        total, items = model.loss([o.float() for o in outputs], mask)
        # a NaN would silently destroy the run (AdamW spreads it into every trained weight); stop with the frames instead
        assert torch.isfinite(total), \
            f"non-finite loss {float(total.detach())} at epoch {epoch} step {i}, frames {[m['frame'] for m in batch['meta']]}"
        loss = total / accumulate
        (scaler.scale(loss) if scaler is not None else loss).backward()
        if (i + 1) % accumulate == 0 or i + 1 == n_steps:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, max_norm)
            if scaler is not None:
                scaler.step(optimizer); scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        lr = optimizer.param_groups[0]['lr']
        metric_logger.update(loss=float(total.detach()), lr=lr, **{k: float(v) for k, v in items.items()})
        if logger is not None:
            # every AdamW group's lr under its name (param_groups(args) names them, e.g. stem / lora / decoder)
            lrs = {f"lr_{g.get('name', k)}": g['lr'] for k, g in enumerate(optimizer.param_groups)}
            logger.scalars({'loss': float(total.detach()), 'lr': lr, **lrs, **{k: float(v) for k, v in items.items()}},
                           epoch * n_steps + i, prefix='train/')
    return {k: m.global_avg for k, m in metric_logger.meters.items()}


def roi_panel(prob, gt):
    '''Side-by-side uint8 RGB picture [h, 2 w, 3] of a native-size ROI prediction (left) and its GT (right), for logging.'''
    p = (np.clip(prob, 0, 1) * 255).astype(np.uint8)                                     # [h, w]
    g = (np.asarray(gt, dtype=bool) * 255).astype(np.uint8)                              # [h, w]
    sep = np.full((p.shape[0], 2), 128, dtype=np.uint8)                                  # grey divider
    return np.repeat(np.concatenate([p, sep, g], axis=1)[..., None], 3, axis=2)


@torch.no_grad()
def evaluate(model, front_end, data_loader, device, logger=None, epoch=0, tag='val', amp=True, amp_dtype=torch.bfloat16,
             size_edges=(2000, 20000), n_images_log=4):
    '''
    ROI-level evaluation (spec §5.5): every item's main output (index 0) goes through a sigmoid, is cut back from the
    canvas and resized to the ROI's native size (canvas_to_roi), and is scored at that size against the GT mask brought
    back the same way (canvas_to_roi(mask) > 0.5; exact when the ROI fits the canvas, s = 1, which is the case for most
    val ROIs; interpolated for ROIs longer than the canvas). False-positive items (meta['source'] == 'fp', empty GT) go
    to SegMetrics.update_fp (false-mask rate), never into S / E / F. Returns SegMetrics.summary() (S, E_mean, E_max,
    E_adp, Fw, F_adp, MAE, IoU, n, the size-bucket copies, fp_false_mask_rate, n_fp); its scalars are logged under
    '<tag>/' and a few prediction | GT panels under '<tag>/roi_<i>'.
    '''
    model.eval()
    front_end.eval()
    metrics = SegMetrics(size_edges=size_edges)
    metric_logger = MetricLogger(delimiter="; ")
    use_amp = bool(amp) and device.type == 'cuda'
    logged = 0
    for batch in metric_logger.log_every(data_loader, 10, f'Eval {tag}:'):
        x = seg_inputs(front_end, batch, device)                                         # [B, N + 1, c, c]
        box_xyxy = batch['box_xyxy'].to(device, non_blocking=True).float()               # [B, 4]
        with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            outputs = model(x, box_xyxy)
        prob = torch.sigmoid(outputs[0].float())[:, 0].cpu().numpy()                     # [B, c, c] in [0, 1]
        mask = batch['mask'][:, 0].float().numpy()                                       # [B, c, c]
        for b, meta in enumerate(batch['meta']):
            p = canvas_to_roi(prob[b], meta['offset'], meta['s'], meta['roi_hw'])         # [h, w] float32, native ROI size
            if meta['source'] == 'fp':
                metrics.update_fp(p)
                continue
            g = canvas_to_roi(mask[b], meta['offset'], meta['s'], meta['roi_hw']) > 0.5  # [h, w] bool
            metrics.update(p, g)
            if logger is not None and logged < n_images_log:
                logger.image(f'{tag}/roi_{logged}', roi_panel(p, g), epoch); logged += 1
    summary = metrics.summary()
    if logger is not None:
        logger.scalars({k: v for k, v in summary.items() if isinstance(v, (int, float, np.integer, np.floating))}, epoch, prefix=f'{tag}/')
    print(f"{tag} epoch {epoch}: " + ", ".join(f"{k}={summary[k]:.4f}" if isinstance(summary[k], float) else f"{k}={summary[k]}"
                                                for k in PRINT_KEYS if k in summary))
    return summary


def paste_back(prob_canvas, meta, out):
    '''
    Paste one ROI prediction into a full-frame probability map, merging overlaps by maximum (spec §5.5).
      prob_canvas [c, c] probability on the canvas (np or tensor), meta the item's meta dict (offset, s, roi_hw, roi),
      out [H, W] float32 frame map (np.zeros((1680, 1240), np.float32) for a fresh frame), modified in place and returned.
    The ROI's crop starts at (floor(y1), floor(x1)) of meta['roi'] (frame px) and is roi_hw = (h, w) pixels, i.e. the
    floor / ceil rounding of the fractional export ROI (the rule of box_metrics.mask_coverage); a roi_hw inconsistent
    with that rounding means the crop came from another rule and raises instead of pasting at a shifted place.
    '''
    if torch.is_tensor(prob_canvas):
        prob_canvas = prob_canvas.detach().float().cpu().numpy()
    assert out.ndim == 2, f"out must be a [H, W] frame map, got shape {out.shape}"
    h, w = int(meta['roi_hw'][0]), int(meta['roi_hw'][1])
    x1, y1, x2, y2 = [float(v) for v in meta['roi']]
    x0, y0 = int(math.floor(x1)), int(math.floor(y1))
    # floor / ceil rounding makes the integer size exceed the float size by less than 2 px, never fall short of it
    assert -1e-3 <= w - (x2 - x1) < 2 and -1e-3 <= h - (y2 - y1) < 2, \
        f"frame {meta.get('frame')}: roi_hw {(h, w)} does not match roi {meta['roi']} (floor x1/y1, ceil x2/y2)"
    assert x0 >= 0 and y0 >= 0 and y0 + h <= out.shape[0] and x0 + w <= out.shape[1], \
        f"frame {meta.get('frame')}: roi {meta['roi']} with size {(h, w)} leaves the {out.shape} frame"
    p = canvas_to_roi(prob_canvas, meta['offset'], meta['s'], (h, w))                     # [h, w] float32
    np.maximum(out[y0:y0 + h, x0:x0 + w], p, out=out[y0:y0 + h, x0:x0 + w])
    return out
