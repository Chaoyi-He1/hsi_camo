import os
import csv
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.tasks import DetectionModel
from ultralytics.cfg import get_cfg
from ultralytics.utils.loss import BboxLoss
from ultralytics.utils.nms import non_max_suppression

from models.filter_bank import FilterBank

PRETRAINED_URL = 'https://github.com/ultralytics/assets/releases/download/v8.4.0/{variant}.pt'
FIRST_CONV_KEY = 'model.0.conv.weight'


def download_pretrained(variant='yolo26s', weights_dir='weights/pretrained'):
    os.makedirs(weights_dir, exist_ok=True)
    path = os.path.join(weights_dir, f'{variant}.pt')
    if not os.path.exists(path):
        print(f"Downloading {variant}.pt to {path}...")
        torch.hub.download_url_to_file(PRETRAINED_URL.format(variant=variant), path, progress=False)
    return path


def adapt_first_conv_weight(w_rgb, n_channels):
    '''Tile the pretrained RGB kernel [c1, 3, k, k] over n_channels and scale by 3/n so the stem's activation scale is kept.'''
    c1, c_in, k1, k2 = w_rgb.shape
    reps = int(math.ceil(n_channels / c_in))
    return (w_rgb.repeat(1, reps, 1, 1)[:, :n_channels] * (c_in / n_channels)).contiguous()  # [c1, n_channels, k, k]


def build_detection_model(variant, n_channels, pretrained_path=None, nc=1, epochs=100):
    '''
    YOLO26 DetectionModel whose first conv takes n_channels. Every pretrained tensor with a matching shape is
    copied (695/708 for yolo26s, nc=1); the first conv is initialised from the RGB kernel (adapt_first_conv_weight);
    the class-head outputs are re-initialised because nc changed. model.args carries the loss gains (get_cfg()).
    '''
    dm = DetectionModel(f'{variant}.yaml', ch=n_channels, nc=nc, verbose=False)
    own = dm.state_dict()
    n_matched = 0
    if pretrained_path is not None:
        from ultralytics import YOLO
        sd = YOLO(pretrained_path).model.float().state_dict()
        matched = {k: v for k, v in sd.items() if k in own and v.shape == own[k].shape}
        n_matched = len(matched)
        dm.load_state_dict(matched, strict=False)
        if FIRST_CONV_KEY in sd and sd[FIRST_CONV_KEY].shape != own[FIRST_CONV_KEY].shape:
            dm.model[0].conv.weight.data.copy_(adapt_first_conv_weight(sd[FIRST_CONV_KEY], n_channels))
    dm.args = get_cfg()
    dm.args.epochs = int(epochs)
    # YOLO26 is NMS-free: infer with the one-to-one head ([B, max_det, 6] xyxy/conf/cls), as ultralytics' own
    # validator does (nms=False). Training is unchanged (E2ELoss supervises both heads). Left off, the model
    # returns the one-to-many [B, 4+nc, A] and eval falls back to NMS, which times out on native-res frames.
    dm.end2end = True
    return dm, n_matched, len(own)


def containment_term(pred_xyxy, target_xyxy, weight, target_scores_sum):
    '''Hinge on how far the GT box protrudes from the prediction, per edge, normalised by the GT width/height.'''
    gw = (target_xyxy[:, 2] - target_xyxy[:, 0]).clamp(min=1e-6)
    gh = (target_xyxy[:, 3] - target_xyxy[:, 1]).clamp(min=1e-6)
    protrude = (F.relu(pred_xyxy[:, 0] - target_xyxy[:, 0]) + F.relu(target_xyxy[:, 2] - pred_xyxy[:, 2])) / gw \
             + (F.relu(pred_xyxy[:, 1] - target_xyxy[:, 1]) + F.relu(target_xyxy[:, 3] - pred_xyxy[:, 3])) / gh   # [n]
    return (protrude.unsqueeze(-1) * weight).sum() / target_scores_sum


class ContainBboxLoss(BboxLoss):
    '''ultralytics BboxLoss (CIoU + L1 for YOLO26) plus the asymmetric containment hinge, added to the IoU term.'''

    def __init__(self, reg_max, contain_weight=1.0):
        super(ContainBboxLoss, self).__init__(reg_max)
        self.contain_weight = float(contain_weight)
        self.last_contain = 0.0

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask, imgsz, stride):
        loss_iou, loss_dfl = super(ContainBboxLoss, self).forward(pred_dist, pred_bboxes, anchor_points, target_bboxes,
                                                                  target_scores, target_scores_sum, fg_mask, imgsz, stride)
        self.last_contain = 0.0
        if self.contain_weight > 0 and bool(fg_mask.any()):
            weight = target_scores[fg_mask].sum(-1, keepdim=True)                       # [n, 1] same weighting as the IoU term
            contain = containment_term(pred_bboxes[fg_mask], target_bboxes[fg_mask], weight, target_scores_sum)
            self.last_contain = float(contain.detach())
            loss_iou = loss_iou + self.contain_weight * contain
        return loss_iou, loss_dfl


class ECYolo(nn.Module):
    '''FilterBank -> YOLO26 with an N-channel first conv. loss(batch) = ultralytics loss + gate entropy penalty.'''

    def __init__(self, filter_bank, yolo, gate_entropy_weight=0.0, contain_weight=0.0, stride=32):
        super(ECYolo, self).__init__()
        self.filter_bank = filter_bank
        self.yolo = yolo
        self.gate_entropy_weight = float(gate_entropy_weight)
        self.contain_weight = float(contain_weight)
        self.stride = int(stride)

    def pad_to_stride(self, img):
        H, W = img.shape[-2:]
        Hp, Wp = math.ceil(H / self.stride) * self.stride, math.ceil(W / self.stride) * self.stride
        # the dataset may hand out fp16 frames (out_dtype='float16') and the model runs in fp32, so upcast --
        # except under autocast, where FilterBank's einsum casts straight back down and the fp32 copy of a
        # [B, 133, 1696, 1248] batch (2.25 GB) would be pure peak-memory overhead.
        if not torch.is_autocast_enabled(img.device.type):
            img = img.float()
        return F.pad(img, (0, Wp - W, 0, Hp - H)), (H, W, Hp, Wp)                     # zero-pad bottom/right

    def attach_criterion(self):
        crit = self.yolo.init_criterion()
        if self.contain_weight > 0:
            reg_max = self.yolo.model[-1].reg_max
            device = next(self.yolo.parameters()).device
            for sub in (getattr(crit, 'one2many', None), getattr(crit, 'one2one', None), crit if hasattr(crit, 'bbox_loss') else None):
                if sub is not None and hasattr(sub, 'bbox_loss'):
                    sub.bbox_loss = ContainBboxLoss(reg_max, self.contain_weight).to(device)
        self.yolo.criterion = crit
        return crit

    def forward(self, img=None, batch=None):
        '''img -> YOLO raw output; or, with batch=..., the training loss (so DDP's forward wraps the loss and syncs gradients).'''
        if batch is not None:
            return self.loss(batch)
        img_p, _ = self.pad_to_stride(img)
        return self.yolo(self.filter_bank(img_p))

    def loss(self, batch):
        if getattr(self.yolo, 'criterion', None) is None:
            self.attach_criterion()
        img_p, (H, W, Hp, Wp) = self.pad_to_stride(batch['img'])
        bboxes = batch['bboxes'].clone().to(img_p.device)
        if len(bboxes):
            bboxes[:, [0, 2]] *= W / Wp                                              # normalised coords follow the padding
            bboxes[:, [1, 3]] *= H / Hp
        y = self.filter_bank(img_p)                                                  # [B, N, Hp, Wp]
        loss, items = self.yolo.loss({'img': y, 'batch_idx': batch['batch_idx'].to(y.device),
                                      'cls': batch['cls'].to(y.device), 'bboxes': bboxes})
        ent = self.filter_bank.entropy()
        total = loss.sum() + self.gate_entropy_weight * ent
        out = {k: float(v) for k, v in (items.items() if isinstance(items, dict) else enumerate(items))}
        out['gate_entropy'] = float(ent.detach())
        o2o = getattr(self.yolo.criterion, 'one2one', self.yolo.criterion)
        out['contain_loss'] = float(getattr(getattr(o2o, 'bbox_loss', None), 'last_contain', 0.0))
        return total, out

    def end_epoch(self):
        crit = getattr(self.yolo, 'criterion', None)
        if crit is not None and hasattr(crit, 'update'):
            crit.update()


def build_ec_yolo(args, dataset):
    '''Session A: all selected voltages + weight vector; session B: fixed channels, no weight vector (initialised by slice_to_channels).'''
    if getattr(args, 'raw_bands', False):
        # control run: the raw cube bands inside band_range go straight into YOLO (identity projection, standardised
        # with the training band statistics, no weight vector) to compare against the EC filter responses
        mu, cov = dataset._band_stats()                                                  # [n_bands], [n_bands, n_bands]
        R = np.eye(dataset.n_bands, dtype=np.float32)
        mean, std = mu.astype(np.float32), np.sqrt(np.maximum(np.diag(cov), 0.0)).astype(np.float32)
        volts, weight_vector = dataset.wavelens.copy(), False                            # 'voltages' = band centres (nm)
    else:
        R, mean, std, volts = dataset.filter_bank_tensors()
        weight_vector = (args.session == 'A')
    fb = FilterBank(R, mean, std, weight_vector=weight_vector)
    pretrained = None if args.pretrained == 'none' else (download_pretrained(args.yolo_variant) if args.pretrained == 'auto' else args.pretrained)
    yolo, n_matched, n_total = build_detection_model(args.yolo_variant, fb.n_channels, pretrained, nc=1, epochs=args.epochs)
    print(f"{args.yolo_variant}: {fb.n_channels} input channels, pretrained tensors reused {n_matched}/{n_total}")
    model = ECYolo(fb, yolo, gate_entropy_weight=args.gate_entropy_weight, contain_weight=args.contain_weight)
    model.selected_voltages = np.asarray(volts)
    return model


def select_top_k(ecyolo, k, csv_path=None, voltages=None):
    '''Rank channels by the weight vector; write rank,index,voltage,weight; return the k best indices (weight-descending).'''
    weights = ecyolo.filter_bank.weights.detach().cpu().numpy()
    voltages = np.asarray(voltages if voltages is not None else getattr(ecyolo, 'selected_voltages', np.arange(len(weights))))
    order = np.argsort(-weights)
    assert 1 <= k <= len(order), f"top_k={k} must be in [1, {len(order)}]"
    if csv_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f); w.writerow(['rank', 'index', 'voltage', 'weight'])
            for r, i in enumerate(order, start=1):
                w.writerow([r, int(i), float(voltages[i]), float(weights[i])])
    return [int(i) for i in order[:k]], weights, voltages


def slice_to_channels(ecyolo, indices, variant, pretrained_path=None):
    '''Keep only `indices`: FilterBank columns, first-conv input channels scaled by the gate (w_j * W[:, idx_j]); no weight vector.'''
    fb, idx = ecyolo.filter_bank, list(indices)
    w = fb.weights.detach().cpu()
    R = fb.R_t.t().cpu().numpy()[:, idx]
    new_fb = FilterBank(R, fb.mean.view(-1).cpu().numpy()[idx], fb.std.view(-1).cpu().numpy()[idx], weight_vector=False)
    yolo, _, _ = build_detection_model(variant, len(idx), pretrained_path=None, nc=ecyolo.yolo.model[-1].nc, epochs=ecyolo.yolo.args.epochs)
    sd = {k: v for k, v in ecyolo.yolo.state_dict().items() if k != FIRST_CONV_KEY}
    yolo.load_state_dict(sd, strict=False)
    W_old = ecyolo.yolo.model[0].conv.weight.detach().cpu()                            # [c1, N, k, k]
    yolo.model[0].conv.weight.data.copy_(W_old[:, idx] * w[idx].view(1, -1, 1, 1))
    new = ECYolo(new_fb, yolo, gate_entropy_weight=0.0, contain_weight=ecyolo.contain_weight, stride=ecyolo.stride)
    new.selected_voltages = np.asarray(getattr(ecyolo, 'selected_voltages', np.arange(fb.n_channels)))[idx]
    return new


def decode_predictions(out, conf_thres=0.001, iou_thres=0.6, max_det=300, nc=1, end2end=None):
    '''
    Raw YOLO output -> per image float32 [M, 6] (x1, y1, x2, y2, conf, cls) in padded-frame pixels.
    end2end=True reads the one-to-one head's [B, max_det, 6] rows directly (no NMS); False decodes the
    one-to-many [B, 4+nc, A] through NMS. None guesses from the column count - pass the model's own
    yolo.end2end where it is known, since [B, 4+nc, A] is ambiguous when A == 6.
    '''
    pred = out[0] if isinstance(out, (list, tuple)) else out
    if end2end is None:
        end2end = pred.ndim == 3 and pred.shape[-1] == 6 and pred.shape[1] != 4 + nc
    if end2end:                                                                        # [B, max_det, 6]: end-to-end head
        dets = [d[d[:, 4] >= conf_thres][:max_det] for d in pred.float()]
    else:                                                                              # [B, 4+nc, A]: decode + NMS
        dets = non_max_suppression(pred.float(), conf_thres=conf_thres, iou_thres=iou_thres, nc=nc, max_det=max_det)
    return [d.detach().cpu().numpy().astype(np.float32).reshape(-1, 6) for d in dets]
