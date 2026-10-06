import types
import numpy as np
import pytest
pytest.importorskip('py_sod_metrics')        # train_eval_seg -> seg_metrics needs it (bash_files/setup_third_party.sh installs it)
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_loader.roi_crops import place_on_canvas, seg_collate_fn
from train_eval.train_eval_seg import seg_inputs, train_one_epoch, evaluate, paste_back
from util.logger import TrainLogger

C = 32                       # canvas side of the hand-made items (the real canvas is 512)


class FixedFront(nn.Module):
    '''Stand-in for an arm front end: a fixed 133 -> 4 projection (+1, so the padding is non-zero before the valid mask).'''

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(4, 133, generator=torch.Generator().manual_seed(0)) / 133)
        self.n_out = 4
        self.dtypes = []                                   # dtype of every input, to check the fp32 front-end rule

    def forward(self, x):
        self.dtypes.append(x.dtype)
        return torch.einsum('nc,bchw->bnhw', self.weight, x) + 1.0     # [B, 4, h, w]


class TinySeg(nn.Module):
    '''The smallest model with the seg API: conv stem + BN + main head + one deep-supervision side head.'''

    def __init__(self, n_in, nan=False):
        super().__init__()
        self.stem = nn.Conv2d(n_in + 1, 4, 3, padding=1)
        self.bn = nn.BatchNorm2d(4)
        self.head, self.side = nn.Conv2d(4, 1, 1), nn.Conv2d(4, 1, 1)
        self.nan = nan
        self.progress, self.seen = [], []

    def forward(self, x, box_xyxy):
        self.seen.append((tuple(x.shape), tuple(box_xyxy.shape)))
        f = torch.relu(self.bn(self.stem(x)))
        return [self.head(f), self.side(f)]                # main output first

    def loss(self, outputs, mask):
        main = F.binary_cross_entropy_with_logits(outputs[0], mask)
        side = F.binary_cross_entropy_with_logits(outputs[1], mask)
        total = main + side
        if self.nan:
            total = total * float('nan')
        return total, {'main': float(main.detach()), 'side': float(side.detach())}

    def param_groups(self, args):
        rest = [p for n, p in self.named_parameters() if not n.startswith('stem.')]
        return [{'params': list(self.stem.parameters()), 'lr': args.stem_lr, 'name': 'stem'}, {'params': rest, 'name': 'rest'}]

    def set_progress(self, t):
        self.progress.append(t)


class BoxEcho(nn.Module):
    '''Predicts exactly the box channel (logit +-10): the perfect model when the object fills its box; empty=True paints nothing.'''

    def __init__(self, empty=False):
        super().__init__()
        self.empty = empty

    def forward(self, x, box_xyxy):
        if self.empty:
            return [torch.full_like(x[:, -1:], -10.0)]
        return [20.0 * x[:, -1:] - 10.0]


def _item(h, w, box, source='gt', seed=0):
    '''One HyperCOD_roi-style item: an h x w ROI at frame origin (x 7, y 5); the object fills box (empty GT for 'fp').'''
    rng = np.random.default_rng(seed)
    crop = (rng.random((133, h, w)) * 0.5).astype(np.float16)                       # [133, h, w] p99-scaled
    gt, bm = np.zeros((1, h, w), np.float32), np.zeros((1, h, w), np.float32)
    x1, y1, x2, y2 = box
    bm[0, y1:y2, x1:x2] = 1.0
    if source != 'fp':
        gt[0, y1:y2, x1:x2] = 1.0
    img, valid, (oy, ox), s = place_on_canvas(crop, canvas=C)
    mask = (place_on_canvas(gt, canvas=C)[0] > 0.5).astype(np.float32)              # [1, C, C]
    box_map = (place_on_canvas(bm, canvas=C)[0] > 0.5).astype(np.float32)           # [1, C, C]
    meta = {'frame': '3', 'roi': [7.0, 5.0, 7.0 + w, 5.0 + h], 'box': [7 + x1, 5 + y1, 7 + x2, 5 + y2],
            'box_canvas': [ox + x1 * s, oy + y1 * s, ox + x2 * s, oy + y2 * s], 'source': source, 'offset': (oy, ox),
            's': s, 'roi_hw': (h, w), 'obj_area': int(gt.sum())}
    return img.astype(np.float16), mask, box_map, valid[None].astype(np.float32), meta


def test_seg_inputs_zero_the_padding_append_the_box_and_run_the_front_end_in_fp32():
    batch = seg_collate_fn([_item(20, 12, (2, 3, 8, 11)), _item(16, 24, (4, 4, 20, 12), seed=1)])
    fe = FixedFront()
    x = seg_inputs(fe, batch, torch.device('cpu'))
    assert x.shape == (2, 5, C, C) and x.dtype == torch.float32
    assert fe.dtypes == [torch.float32]                                            # fp16 crop cast before the front end
    pad = ~batch['valid'].bool().expand(-1, 4, -1, -1)                             # [B, 4, C, C] canvas padding
    assert pad.any() and (x[:, :4][pad] == 0).all() and (x[:, :4][~pad] != 0).any()
    assert torch.equal(x[:, 4:], batch['box_map'].float())                         # last channel = box map
    assert not x.requires_grad                                                     # no graph into the fixed front end


def test_paste_back_round_trips_merges_by_maximum_and_rejects_bad_geometry():
    crop = np.random.default_rng(0).random((1, 10, 6)).astype(np.float32)          # [1, h, w] ROI probabilities
    canv, _, off, s = place_on_canvas(crop, canvas=16)
    assert s == 1.0
    meta = {'frame': '3', 'roi': [3.0, 2.0, 9.0, 12.0], 'offset': off, 's': s, 'roi_hw': (10, 6)}
    out = np.zeros((20, 15), np.float32)
    assert paste_back(canv[0], meta, out) is out
    np.testing.assert_array_equal(out[2:12, 3:9], crop[0])                          # exact without a resize
    assert out.sum() == pytest.approx(float(crop.sum()))                            # nothing outside the ROI
    half = np.full((1, 4, 4), 0.5, np.float32)
    canv2, _, off2, s2 = place_on_canvas(half, canvas=16)
    before = out.copy()
    paste_back(torch.from_numpy(canv2[0]), {'frame': '3', 'roi': [5.0, 8.0, 9.0, 12.0], 'offset': off2, 's': s2, 'roi_hw': (4, 4)}, out)
    np.testing.assert_array_equal(out[8:12, 5:9], np.maximum(before[8:12, 5:9], 0.5))   # max-merge of overlapping ROIs
    keep = np.ones_like(out, bool); keep[8:12, 5:9] = False
    np.testing.assert_array_equal(out[keep], before[keep])
    # a fractional export ROI is cut with floor x1/y1, ceil x2/y2: [2.5, 1.5, 8.5, 9.5] -> rows 1:10, cols 2:9
    frac = np.random.default_rng(1).random((1, 9, 7)).astype(np.float32)
    canv3, _, off3, s3 = place_on_canvas(frac, canvas=16)
    out3 = paste_back(canv3[0], {'frame': '3', 'roi': [2.5, 1.5, 8.5, 9.5], 'offset': off3, 's': s3, 'roi_hw': (9, 7)},
                      np.zeros((20, 15), np.float32))
    np.testing.assert_array_equal(out3[1:10, 2:9], frac[0])
    with pytest.raises(AssertionError):                                            # roi_hw from another rounding rule
        paste_back(canv[0], {**meta, 'roi_hw': (6, 6)}, out)
    # roi_hw must be exactly (ceil(y2) - floor(y1), ceil(x2) - floor(x1)): 1 px off in either axis raises
    for bad in ((11, 6), (10, 7), (9, 6), (10, 5)):
        with pytest.raises(AssertionError, match='roi_hw'):
            paste_back(canv[0], {**meta, 'roi_hw': bad}, out)
    for bad in ((8, 7), (9, 6)):                                                   # the fractional ROI above needs (9, 7)
        with pytest.raises(AssertionError, match='roi_hw'):
            paste_back(canv3[0], {'frame': '3', 'roi': [2.5, 1.5, 8.5, 9.5], 'offset': off3, 's': s3, 'roi_hw': bad},
                       np.zeros((20, 15), np.float32))
    with pytest.raises(AssertionError):                                            # ROI leaves the frame
        paste_back(canv[0], {**meta, 'roi': [10.0, 12.0, 16.0, 22.0]}, out)


def test_paste_back_of_a_downscaled_roi_is_within_interpolation_error():
    ramp = np.broadcast_to(np.linspace(0, 1, 24, dtype=np.float32)[None, None], (1, 40, 24)).copy()   # smooth [1, 40, 24]
    canv, _, off, s = place_on_canvas(ramp, canvas=16)
    assert s == pytest.approx(0.4)
    out = paste_back(canv[0], {'frame': '3', 'roi': [1.0, 0.0, 25.0, 40.0], 'offset': off, 's': s, 'roi_hw': (40, 24)},
                     np.zeros((48, 40), np.float32))
    assert np.abs(out[:40, 1:25] - ramp[0]).max() < 0.1 and out[:, 25:].max() == 0 and out[40:].max() == 0


def test_train_one_epoch_accumulates_reports_progress_and_keeps_the_front_end_fixed(tmp_path):
    items = [_item(20, 12, (2, 3, 8, 11), seed=k) for k in range(4)]
    loader = torch.utils.data.DataLoader(items, batch_size=2, shuffle=False, collate_fn=seg_collate_fn)
    model, fe = TinySeg(4), FixedFront()
    opt = torch.optim.AdamW(model.param_groups(types.SimpleNamespace(stem_lr=1e-2)), lr=1e-3)
    w0 = model.head.weight.detach().clone()
    logger = TrainLogger(types.SimpleNamespace(name='t', wandb=False, runs_dir=str(tmp_path / 'runs'), rank=0), cfg={})
    stats = train_one_epoch(model, fe, loader, opt, torch.device('cpu'), epoch=1, accumulate=2, max_norm=1.0, logger=logger,
                            print_freq=1, total_epochs=4)
    logger.finish()
    assert {'loss', 'lr', 'main', 'side'} <= set(stats) and np.isfinite(stats['loss'])
    assert model.progress == pytest.approx([0.25, 0.375])                          # (epoch + i / n_steps) / total_epochs
    assert int(opt.state[model.head.weight]['step']) == 1                          # 2 batches accumulated into one step
    assert not torch.equal(model.head.weight.detach(), w0)
    assert fe.weight.grad is None                                                  # nothing flows into the front end
    assert model.seen[0] == ((2, 5, C, C), (2, 4))


class _ScalarLog:
    '''Records logger.scalars calls ({name: value} per step), the only logger method train_one_epoch uses.'''

    def __init__(self):
        self.calls = []

    def scalars(self, d, step, prefix=''):
        self.calls.append((dict(d), step, prefix))


class LossItemSeg(TinySeg):
    '''TinySeg whose loss items carry a 'loss' key, like SAM2UNetSeg ({'loss_main', 'loss_s16', 'loss_s8', 'loss'}); item_loss
    None = the key holds the total, a number = a stand-in that must lose against the explicitly logged total.'''

    def __init__(self, n_in, item_loss=None):
        super().__init__(n_in)
        self.item_loss, self.totals = item_loss, []

    def loss(self, outputs, mask):
        total, items = super().loss(outputs, mask)
        self.totals.append(float(total.detach()))
        items['loss'] = float(total.detach()) if self.item_loss is None else self.item_loss
        return total, items


@pytest.mark.parametrize('item_loss', [None, 123.0])
def test_train_one_epoch_survives_a_loss_item_named_loss_and_logs_the_total(item_loss):
    items = [_item(20, 12, (2, 3, 8, 11), seed=k) for k in range(4)]
    loader = torch.utils.data.DataLoader(items, batch_size=2, shuffle=False, collate_fn=seg_collate_fn)
    model, log = LossItemSeg(4, item_loss), _ScalarLog()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    stats = train_one_epoch(model, FixedFront(), loader, opt, torch.device('cpu'), epoch=0, logger=log, print_freq=1)   # was a TypeError
    assert len(model.totals) == 2 and len(log.calls) == 2
    assert stats['loss'] == pytest.approx(np.mean(model.totals))                   # the returned 'loss' is the total's average
    assert [c[0]['loss'] for c in log.calls] == pytest.approx(model.totals)        # and so is every logged scalar
    assert {'loss', 'lr', 'main', 'side'} <= set(stats)


def test_train_one_epoch_stops_on_a_non_finite_loss():
    loader = torch.utils.data.DataLoader([_item(20, 12, (2, 3, 8, 11))], batch_size=1, collate_fn=seg_collate_fn)
    model = TinySeg(4, nan=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with pytest.raises(AssertionError, match='non-finite loss'):
        train_one_epoch(model, FixedFront(), loader, opt, torch.device('cpu'), epoch=0)


def test_evaluate_scores_objects_at_native_size_and_fp_rois_by_false_mask_rate():
    items = [_item(20, 12, (2, 3, 8, 11)), _item(40, 24, (4, 6, 20, 30), seed=1),     # second one is downscaled (s = 0.8)
             _item(16, 16, (4, 4, 12, 12), source='fp', seed=2)]
    loader = torch.utils.data.DataLoader(items, batch_size=2, shuffle=False, collate_fn=seg_collate_fn)
    summary = evaluate(BoxEcho(), FixedFront(), loader, torch.device('cpu'), epoch=0, tag='val')
    assert summary['n'] == 2 and summary['n_fp'] == 1                              # fp items never enter S / E / F
    assert summary['S'] > 0.95 and summary['IoU'] > 0.95 and summary['MAE'] < 0.05  # box echo = the object (edges of the s = 0.8 item interpolate)
    assert summary['fp_false_mask_rate'] == 1.0                                    # it also paints the fp box
    assert 'S_small' in summary and summary['n_small'] == 2
    empty = evaluate(BoxEcho(empty=True), FixedFront(), loader, torch.device('cpu'), epoch=0, tag='val')
    assert empty['fp_false_mask_rate'] == 0.0 and empty['IoU'] == 0.0
