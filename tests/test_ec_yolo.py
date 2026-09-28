import os
import types
import numpy as np
import pytest
import torch

from ultralytics.utils.loss import BboxLoss
from models.filter_bank import FilterBank
from models.ec_yolo import (adapt_first_conv_weight, build_detection_model, containment_term, ContainBboxLoss, ECYolo,
                            build_ec_yolo, select_top_k, slice_to_channels, decode_predictions, download_pretrained)
from data_loader.boxes import det_collate_fn
from data_loader.my_dataset import HyperCOD_data
from tests.conftest import H, W


def _tiny(n_channels, weight_vector=True, logits=None, contain_weight=1.0, gate_w=0.1):
    dm, _, _ = build_detection_model('yolo26n', n_channels, pretrained_path=None, nc=1, epochs=2)
    rng = np.random.RandomState(0)
    fb = FilterBank(rng.rand(133, n_channels).astype(np.float32) / 133, np.zeros(n_channels, np.float32),
                    np.ones(n_channels, np.float32), weight_vector=weight_vector, init_logits=logits)
    return ECYolo(fb, dm, gate_entropy_weight=gate_w, contain_weight=contain_weight)


def _batch(root):
    ds = HyperCOD_data(str(root), split='train', use_filter=False, norm='p99', crop_size=0, out_dtype='float16', filter_norm='none')
    return det_collate_fn([ds[0], ds[1]], min_area=10)


def test_adapt_first_conv_weight_tiles_and_scales():
    w = torch.arange(2 * 3 * 3 * 3, dtype=torch.float32).view(2, 3, 3, 3)
    a = adapt_first_conv_weight(w, 7)
    assert a.shape == (2, 7, 3, 3)
    torch.testing.assert_close(a[:, 3], w[:, 0] * (3 / 7)); torch.testing.assert_close(a[:, 6], w[:, 0] * (3 / 7))
    torch.testing.assert_close(adapt_first_conv_weight(w, 3), w)


def test_build_detection_model_with_pretrained_weights():
    path = download_pretrained('yolo26s', weights_dir='weights/pretrained')
    assert os.path.exists(path)
    dm, n_matched, n_total = build_detection_model('yolo26s', 133, pretrained_path=path, nc=1)
    assert n_matched >= 690 and n_total == 708
    conv = dm.model[0].conv
    assert tuple(conv.weight.shape) == (32, 133, 3, 3) and conv.stride == (2, 2) and conv.bias is None
    from ultralytics import YOLO
    w_rgb = YOLO(path).model.float().state_dict()['model.0.conv.weight']
    torch.testing.assert_close(conv.weight.detach(), adapt_first_conv_weight(w_rgb, 133))
    assert dm.args.epochs == 100 and dm.model[-1].nc == 1


def test_containment_term_hinge():
    tb = torch.tensor([[10., 10., 20., 30.]]); w = torch.ones(1, 1); s = torch.tensor(1.0)
    assert float(containment_term(torch.tensor([[8., 9., 22., 31.]]), tb, w, s)) == 0.0          # contains -> no penalty
    assert float(containment_term(torch.tensor([[12., 10., 20., 30.]]), tb, w, s)) == pytest.approx(0.2)   # 2 px of 10 px width cut
    assert float(containment_term(torch.tensor([[10., 10., 20., 25.]]), tb, w, s)) == pytest.approx(0.25)  # 5 px of 20 px height cut
    assert float(containment_term(torch.tensor([[12., 10., 20., 25.]]), tb, w, s)) == pytest.approx(0.45)


def _bbox_loss_inputs():
    '''Synthetic BboxLoss.forward arguments: 1 image, 3 anchors, 2 foreground, reg_max=1 (the YOLO26 L1 branch).
    Anchor 0's prediction cuts 2/10 of the GT width and 5/20 of its height; anchor 1's prediction contains the GT.'''
    A = 3
    return dict(pred_dist=torch.zeros(1, A, 4),                                       # [B, A, 4]
                pred_bboxes=torch.tensor([[[12., 10., 20., 25.], [8., 9., 22., 31.], [0., 0., 1., 1.]]]),
                anchor_points=torch.tensor([[5., 5.], [15., 15.], [25., 25.]]),       # [A, 2]
                target_bboxes=torch.tensor([[[10., 10., 20., 30.], [10., 10., 20., 30.], [0., 0., 1., 1.]]]),
                target_scores=torch.tensor([[[1.], [1.], [0.]]]),                     # [B, A, nc]
                target_scores_sum=torch.tensor(2.0), fg_mask=torch.tensor([[True, True, False]]),
                imgsz=torch.tensor([64., 64.]), stride=torch.ones(A, 1))


def _call(loss):
    return loss(**{k: v.clone() for k, v in _bbox_loss_inputs().items()})


def test_contain_bbox_loss_reduces_to_bbox_loss_when_disabled():
    a = ContainBboxLoss(1, contain_weight=0.0)
    assert a.contain_weight == 0.0 and a.last_contain == 0.0 and isinstance(a, BboxLoss)
    base_iou, base_dfl = _call(BboxLoss(1))
    off_iou, off_dfl = _call(a)
    torch.testing.assert_close(off_iou, base_iou); torch.testing.assert_close(off_dfl, base_dfl)
    assert a.last_contain == 0.0


def test_contain_bbox_loss_adds_the_protruding_fraction_to_the_iou_term():
    base_iou, base_dfl = _call(BboxLoss(1))
    crit = ContainBboxLoss(1, contain_weight=1.0)
    on_iou, on_dfl = _call(crit)
    expected = (0.2 + 0.25) * 1.0 / 2.0                                  # weighted by target_scores / target_scores_sum
    torch.testing.assert_close(on_dfl, base_dfl)                         # the hinge only touches the IoU term
    assert crit.last_contain == pytest.approx(expected)
    torch.testing.assert_close(on_iou, base_iou + expected)
    two_iou, _ = _call(ContainBboxLoss(1, contain_weight=2.0))            # proportional to contain_weight
    torch.testing.assert_close(two_iou, base_iou + 2 * expected)


def test_ecyolo_pads_and_computes_loss_on_cpu(synthetic_root):
    root, _, _ = synthetic_root
    model = _tiny(6).train()
    batch = _batch(root)
    img_p, (h, w, hp, wp) = model.pad_to_stride(batch['img'])
    assert (h, w, hp, wp) == (H, W, 64, 64) and img_p.shape == (2, 133, 64, 64)
    total, items = model.loss(batch)
    assert torch.isfinite(total) and total.requires_grad
    assert {'box_loss', 'cls_loss', 'l1_loss', 'gate_entropy', 'contain_loss'} <= set(items)
    total.backward()
    assert model.filter_bank.theta.grad is not None and model.yolo.model[0].conv.weight.grad.shape == (16, 6, 3, 3)
    model.end_epoch()                                                           # criterion.update() must not fail


def test_pad_to_stride_skips_the_fp32_copy_under_autocast():
    model = _tiny(4, contain_weight=0.0)
    x = torch.zeros(1, 133, H, W, dtype=torch.float16)
    out, shape = model.pad_to_stride(x)
    assert out.dtype == torch.float32 and shape == (H, W, 64, 64)      # autocast off: the model runs in fp32
    with torch.autocast('cpu', dtype=torch.bfloat16):
        out_low, shape_low = model.pad_to_stride(x)
    assert out_low.dtype == torch.float16 and shape_low == shape       # the einsum casts down anyway


def test_select_top_k_and_slice_preserve_outputs(tmp_path):
    logits = np.array([5.0, 4.0, 3.0, -30.0, -30.0, -30.0], np.float32)          # channels 3-5 have ~zero weight
    model = _tiny(6, logits=logits).eval()
    idx, weights, volts = select_top_k(model, 3, csv_path=str(tmp_path / 'rank.csv'), voltages=np.arange(6) * 0.1)
    assert idx == [0, 1, 2] and weights.shape == (6,) and (tmp_path / 'rank.csv').read_text().splitlines()[0] == 'rank,index,voltage,weight'
    small = slice_to_channels(model, idx, 'yolo26n').eval()
    assert small.filter_bank.n_channels == 3 and not small.filter_bank.weight_vector
    assert tuple(small.yolo.model[0].conv.weight.shape) == (16, 3, 3, 3)
    x = torch.rand(1, 133, 64, 64)
    with torch.no_grad():
        a, b = model(x), small(x)
    torch.testing.assert_close(a[0], b[0], rtol=1e-3, atol=1e-3)


def test_decode_predictions_returns_xyxy_conf_cls():
    model = _tiny(4, contain_weight=0.0).eval()
    with torch.no_grad():
        out = model(torch.rand(2, 133, 64, 64))
    assert model.yolo.end2end and out[0].ndim == 3 and out[0].shape[-1] == 6          # one-to-one head, no NMS
    dets = decode_predictions(out, conf_thres=0.0, iou_thres=0.6, max_det=10)
    assert len(dets) == 2 and all(d.shape[1] == 6 and d.dtype == np.float32 and len(d) <= 10 for d in dets)
    assert all(np.all(d[:, 2] >= d[:, 0]) and np.all(d[:, 3] >= d[:, 1]) for d in dets)  # xyxy


def test_decode_predictions_falls_back_to_nms_for_the_one_to_many_head():
    pred = torch.zeros(1, 5, 3)                                             # [B, 4+nc, A] xywh + cls, nc=1
    pred[0, :4, 0] = torch.tensor([20., 15., 10., 10.]); pred[0, 4, 0] = 0.9
    pred[0, :4, 1] = torch.tensor([20., 15., 10., 10.]); pred[0, 4, 1] = 0.8   # duplicate: NMS must drop it
    for kw in ({}, {'end2end': False}):                                     # auto-detected and passed explicitly
        dets = decode_predictions(pred, conf_thres=0.5, iou_thres=0.6, max_det=10, **kw)
        assert len(dets) == 1 and dets[0].shape == (1, 6)
        assert dets[0][0].tolist() == pytest.approx([15.0, 10.0, 25.0, 20.0, 0.9, 0.0])


def test_build_ec_yolo_from_args_and_dataset(synthetic_root):
    root, _, _ = synthetic_root
    ds = HyperCOD_data(str(root), split='train', use_filter=False, norm='p99', crop_size=0, num_filters=8, filter_norm='l1', out_dtype='float16')
    args = types.SimpleNamespace(yolo_variant='yolo26n', pretrained='none', gate_entropy_weight=0.05, contain_weight=1.0, epochs=3, session='A')
    m = build_ec_yolo(args, ds)
    assert m.filter_bank.n_channels == 8 and m.filter_bank.weight_vector and m.yolo.args.epochs == 3
    args.session = 'B'
    assert not build_ec_yolo(args, ds).filter_bank.weight_vector
