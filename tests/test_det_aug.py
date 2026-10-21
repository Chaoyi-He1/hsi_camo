'''Plan item 3 (2026-10-20): scale / crop and photometric jitter, the weight EMA and --train_on_val of the Stage-1 detector.'''
import json
import math
import os
import torch

import main_det
from data_loader.cube_cache import build_cube_cache
from train_eval.train_eval_det import scale_crop_boxes, random_scale_crop, random_gain, ModelEMA
from tests.test_main_det import _args


def _lit_block():
    '''One 8 x 10 frame with a 2 x 2 block of ones at rows 2-3, cols 4-5 and its normalised (cx, cy, w, h) box.'''
    img = torch.zeros(1, 1, 8, 10); img[0, 0, 2:4, 4:6] = 1.0
    return img, torch.tensor([[0.5, 0.375, 0.2, 0.25]]), torch.tensor([0.]), torch.zeros(1, 1)


def _mask_box(out):
    '''Normalised box of the pixels > 0.5 in an [H, W] frame and their count.'''
    ys, xs = torch.nonzero(out > 0.5, as_tuple=True)
    H, W = out.shape
    if len(ys) == 0:
        return None, 0
    x1, x2, y1, y2 = xs.min().item(), xs.max().item() + 1, ys.min().item(), ys.max().item() + 1
    return torch.tensor([(x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H]), len(ys)


def test_scale_crop_boxes_clips_and_drops():
    bb = torch.tensor([[0.5, 0.5, 0.2, 0.2]])                                   # 20 x 20 box centred in a 100 x 100 frame
    out, keep = scale_crop_boxes(bb, (200, 200), (100, 100), 100, 100, min_visible=0.3)   # 2x, crop from (100, 100): a quarter is left
    assert keep.tolist() == [False] and out.shape == (0, 4)
    out, keep = scale_crop_boxes(bb, (200, 200), (100, 100), 100, 100, min_visible=0.2)
    assert keep.tolist() == [True]; torch.testing.assert_close(out[0], torch.tensor([0.1, 0.1, 0.2, 0.2]))
    out, keep = scale_crop_boxes(bb, (50, 50), (-25, -25), 100, 100)            # 0.5x placed at (25, 25): whole box, 10 x 10 at (45..55)
    assert keep.tolist() == [True]; torch.testing.assert_close(out[0], torch.tensor([0.5, 0.5, 0.1, 0.1]))
    out, keep = scale_crop_boxes(bb[:0], (50, 50), (0, 0), 100, 100)
    assert out.shape == (0, 4) and keep.shape == (0,)


def test_random_scale_crop_keeps_boxes_on_their_pixels():
    '''Up- and down-scaling with random crops / placements: the returned box is the box of the lit pixels, or the box is
    dropped (with its cls / batch_idx rows) when less than min_visible of it survived the crop.'''
    img, bb, idx, cls = _lit_block()
    for scale in ((2.0, 2.0), (0.5, 0.5)):
        for seed in range(12):
            g = torch.Generator().manual_seed(seed)
            out, bb2, idx2, cls2, scales = random_scale_crop(img, bb, idx, cls, scale, min_visible=0.3, generator=g)
            assert out.shape == img.shape and scales == [scale[0]] and len(bb2) == len(idx2) == len(cls2)
            mbox, n_lit = _mask_box(out[0, 0])
            full = 16 if scale[0] == 2.0 else 1                                     # lit pixels of the whole block at that scale
            if n_lit / full >= 0.3:
                assert len(bb2) == 1, f"scale {scale} seed {seed}: box dropped although {n_lit}/{full} of it is visible"
                torch.testing.assert_close(bb2[0], mbox, atol=1e-6, rtol=0)
            else:
                assert len(bb2) == 0, f"scale {scale} seed {seed}: box kept with {n_lit}/{full} visible"
            if scale[0] == 0.5:                                                     # the canvas outside the placed frame holds the frame mean
                assert (out[0, 0] == img.mean()).sum() >= 80 - 20
    out, bb2, idx2, cls2, scales = random_scale_crop(img, bb, idx, cls, (1.0, 1.0))
    assert torch.equal(out, img) and torch.equal(bb2, bb) and scales == [1.0]       # scale 1: untouched
    out, bb2, idx2, cls2, _ = random_scale_crop(img, bb[:0], idx[:0], cls[:0], (2.0, 2.0))
    assert bb2.shape == (0, 4) and idx2.shape == (0,) and cls2.shape == (0, 1)       # a frame without boxes


def test_random_scale_crop_handles_each_sample_separately():
    img, bb, idx, cls = _lit_block()
    img2 = torch.cat([img, img]); bb2 = torch.cat([bb, bb]); idx2 = torch.tensor([0., 1.]); cls2 = torch.zeros(2, 1)
    g = torch.Generator().manual_seed(3)
    out, b, i, c, scales = random_scale_crop(img2, bb2, idx2, cls2, (0.6, 1.6), generator=g)
    assert len(scales) == 2 and scales[0] != scales[1] and out.shape == img2.shape
    assert all(0.6 <= s <= 1.6 for s in scales) and set(i.tolist()) <= {0.0, 1.0} and len(b) == len(i) == len(c)


def test_random_gain_scales_frames_and_tilts_channels():
    img = torch.ones(2, 4, 3, 3)
    g = torch.Generator().manual_seed(0)
    out = random_gain(img, gain=0.2, tilt=0.0, generator=g)
    for b in range(2):
        v = out[b].unique()
        assert len(v) == 1 and torch.exp(torch.tensor(-0.2)) <= v[0] <= torch.exp(torch.tensor(0.2))   # one gain per frame
    out = random_gain(img, gain=0.0, tilt=0.1, generator=g)
    for b in range(2):
        ch = out[b, :, 0, 0]
        torch.testing.assert_close(ch[1] - ch[0], ch[2] - ch[1]); torch.testing.assert_close(ch.mean(), torch.tensor(1.0))   # linear ramp, mean 1
        assert (ch - 1).abs().max() <= 0.1 + 1e-6
    assert random_gain(img) is img                                                   # both off: untouched
    half = random_gain(img.half(), gain=0.1, generator=g)
    assert half.dtype == torch.float16 and half.shape == img.shape


def test_model_ema_averages_and_round_trips():
    m = torch.nn.Linear(2, 1); torch.nn.init.constant_(m.weight, 1.0); torch.nn.init.constant_(m.bias, 0.0)
    ema = ModelEMA(m, decay=0.5, tau=1e-9)                                          # tau -> 0: no warm-up, decay 0.5 at once
    assert not any(p.requires_grad for p in ema.module.parameters())
    torch.nn.init.constant_(m.weight, 3.0)
    ema.update(m)
    assert ema.updates == 1 and ema.current_decay() == 0.5
    torch.testing.assert_close(ema.module.weight, torch.full((1, 2), 2.0))          # 0.5 * 1 + 0.5 * 3
    torch.testing.assert_close(m.weight, torch.full((1, 2), 3.0))                   # the raw model is untouched
    warm = ModelEMA(m, decay=0.992, tau=128.0)
    warm.update(m)
    assert abs(warm.current_decay() - 0.992 * (1 - math.exp(-1 / 128.0))) < 1e-9
    fresh = ModelEMA(m); fresh.load_state_dict(warm.state_dict())
    assert fresh.updates == 1 and fresh.decay == 0.992 and fresh.tau == 128.0
    # BatchNorm statistics are averaged, a constant buffer is copied bit for bit (a rebuilt Stage-2 front end must equal it)
    bn = torch.nn.Sequential(torch.nn.BatchNorm1d(2)); bn.register_buffer('const', torch.tensor([0.1, 0.2, 0.3]))
    e = ModelEMA(bn, decay=0.5, tau=1e-9)
    bn[0].running_mean.fill_(4.0); bn.const.mul_(3.0)
    e.update(bn)
    torch.testing.assert_close(e.module[0].running_mean, torch.full((2,), 2.0))      # 0.5 * 0 + 0.5 * 4
    assert torch.equal(e.module.const, bn.const) and e.module[0].num_batches_tracked == bn[0].num_batches_tracked


def test_main_det_with_augmentation_ema_and_train_on_val(synthetic_root, tmp_path, monkeypatch, capsys):
    '''A training round trip with every new switch on: train + val frames in the training set, EMA weights under 'model' and the
    raw ones under 'model_raw', a continued run restores both, and --eval loads the averaged checkpoint.'''
    root, _, _ = synthetic_root
    build_cube_cache(str(root), 'train', num_workers=0); build_cube_cache(str(root), 'test', num_workers=0)
    monkeypatch.delenv('RANK', raising=False)
    n_ids = len([f for f in os.listdir(root / 'train' / 'hyperspectral') if f.endswith('.mat')])
    out = tmp_path / 'aug_A'
    base = {'--session': 'A', '--name': 'aug', '--output-dir': str(out), '--top_k': '3', '--aug_gain': '0.2', '--aug_tilt': '0.1',
            '--ema': None, '--train_on_val': None, '--ema_tau': '1e-9'}
    args = _args(root, tmp_path, **base); args.aug_scale = [0.8, 1.25]
    main_det.main(args)
    printed = capsys.readouterr().out
    assert f"train {n_ids} / val" in printed and "--train_on_val" in printed and "augmentation: scale (0.8, 1.25)" in printed
    ck = torch.load(out / 'model_0', map_location='cpu', weights_only=False)
    assert 'model_raw' in ck and ck['ema']['updates'] >= 1 and ck['args']['train_on_val'] is True and ck['args']['aug_scale'] == [0.8, 1.25]
    k = 'yolo.model.0.conv.weight'
    assert not torch.equal(ck['model'][k], ck['model_raw'][k])                      # averaged vs trained weights
    args2 = _args(root, tmp_path, **{**base, '--resume': str(out / 'model_0'), '--epochs': '2'}); args2.aug_scale = [0.8, 1.25]
    main_det.main(args2)
    assert f"{ck['ema']['updates']} updates so far" in capsys.readouterr().out      # the EMA counter was restored
    ck2 = torch.load(out / 'model_1', map_location='cpu', weights_only=False)
    assert ck2['ema']['updates'] > ck['ema']['updates'] and ck2['epoch'] == 1
    lines = [json.loads(l) for l in (out / 'results_aug.txt').read_text().strip().splitlines()]
    assert [l['epoch'] for l in lines if 'train' in l] == [0, 1] and lines[-1].get('final') is True
    main_det.main(_args(root, tmp_path, **{'--session': 'A', '--name': 'aug_eval', '--output-dir': str(out), '--resume': str(out / 'model_best'), '--eval': None}))
    assert (out / 'results_aug_eval.txt').exists()


def test_main_det_without_the_switches_is_unchanged(synthetic_root, tmp_path, monkeypatch, capsys):
    root, _, _ = synthetic_root
    build_cube_cache(str(root), 'train', num_workers=0); build_cube_cache(str(root), 'test', num_workers=0)
    monkeypatch.delenv('RANK', raising=False)
    out = tmp_path / 'plain_A'
    main_det.main(_args(root, tmp_path, **{'--session': 'A', '--name': 'plain', '--output-dir': str(out), '--top_k': '3'}))
    printed = capsys.readouterr().out
    assert 'augmentation:' not in printed and 'EMA' not in printed and '--train_on_val' not in printed
    ck = torch.load(out / 'model_0', map_location='cpu', weights_only=False)
    assert 'model_raw' not in ck and 'ema' not in ck
