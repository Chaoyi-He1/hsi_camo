import json
import pickle
import numpy as np
import pytest
import torch
import torch.nn as nn

pytest.importorskip('py_sod_metrics')
import main_det
import main_seg
import main_seg_eval
from tests.conftest import write_sample
from tests.test_main_det import _args
from tests.test_roi_crops import _crop_cache
from data_loader.cube_cache import build_cube_cache, default_cache_dir
from data_loader.roi_crops import HyperCOD_roi
from data_loader.my_dataset import HyperCOD_data
from train_eval.seg_metrics import pool

OBJ_BOX = [20.0, 10.0, 26.0, 16.0]          # conftest OBJ_SLICE (rows 10:16, cols 20:26) as xyxy
FP_BOX = [2.0, 36.0, 8.0, 42.0]             # a detector box far from the object (covers 0 % of its mask)
FRAME_PX = 48 * 40


class BoxEcho(nn.Module):
    '''
    Stand-in segmentation model with the build_seg_model API: logits = +-gain from the box channel (the last input
    channel), so the predicted mask is exactly the prompt box. A wrong canvas placement, box map or paste-back shows up
    as IoU < 1 on the oracle levels; gain starts at 0 (p = 0.5 everywhere) so an unloaded checkpoint shows up too.
    '''
    def __init__(self, n_in, gain=0.0):
        super().__init__()
        self.n_in = n_in
        self.gain = nn.Parameter(torch.tensor(float(gain)))

    def forward(self, x, box_xyxy):
        assert x.shape[1] == self.n_in + 1, f"expected {self.n_in} arm channels + the box channel, got {x.shape[1]}"
        assert box_xyxy.shape == (x.shape[0], 4)
        return [(x[:, -1:] * 2 - 1) * self.gain]


def _setup(synthetic_root, tmp_path, monkeypatch):
    '''Fixture frames + caches, a trained (1 epoch) raw-band detector, its hand-written test ROI export.'''
    root, _, _ = synthetic_root
    monkeypatch.delenv('RANK', raising=False)
    write_sample(str(root), 'test', '8', np.random.default_rng(1))     # a second test frame, left without any detector ROI
    build_cube_cache(str(root), 'train', num_workers=0); build_cube_cache(str(root), 'test', num_workers=0)
    out_a = tmp_path / 'det_A'
    # --raw-bands: the raw and rgb arms (and the zero-shot control) need the raw-band detector (build_front_end's arm check)
    main_det.main(_args(root, tmp_path, **{'--session': 'A', '--raw-bands': None, '--name': 'tA', '--output-dir': str(out_a),
                                           '--top_k': '3'}))
    roi_dir = tmp_path / 'rois'
    roi_dir.mkdir()
    # frame 7: one ROI on the object (expand_box(OBJ_BOX, 1.5, 0)) and one false positive; frame 8: no ROI
    rois = {'7': {'rois': [[18.5, 8.5, 27.5, 17.5, 0.9], [0.5, 34.5, 9.5, 43.5, 0.4]], 'boxes': [OBJ_BOX + [0.9], FP_BOX + [0.4]],
                  'gt_boxes': [OBJ_BOX]},
            '8': {'rois': [], 'boxes': [], 'gt_boxes': [OBJ_BOX]}}
    (roi_dir / 'rois_det_A_test.json').write_text(json.dumps(rois))
    return root, out_a / 'model_best', roi_dir


def _write_run(tmp_path, name, arm, seed, det_ckpt):
    a = main_seg.get_args_parser().parse_args(['--seg_model', 'sam2unet', '--arm', arm, '--det_ckpt', str(det_ckpt), '--canvas', '32',
                                               '--seed', str(seed), '--name', name, '--crop_cache', str(tmp_path / 'crops'),
                                               '--output_dir', str(tmp_path / 'weights' / name)])
    d = tmp_path / 'weights' / name
    d.mkdir(parents=True)
    torch.save({'model': BoxEcho(0, gain=20.0).state_dict(), 'args': vars(a), 'epoch': 3}, d / 'model_best')


def _eval_args(root, tmp_path, roi_dir, *extra):
    return main_seg_eval.get_args_parser().parse_args(
        ['--weights_dir', str(tmp_path / 'weights'), '--out_dir', str(tmp_path / 'seg'), '--roi_dir', str(roi_dir),
         '--data_path', str(root), '--split_file', str(tmp_path / 'val.json'), '--device', 'cpu', '--num_workers', '0',
         '--roi_min', '0', '--min_area', '10', '--n_boot', '50', '--batch_size', '2', '--runs_per_pass', '2', *extra])


def test_roi_window_contains_the_roi():
    assert main_seg_eval.roi_window([18.5, 8.5, 27.5, 17.5], 48, 40) == [18, 8, 28, 18]
    assert main_seg_eval.roi_window([-3.0, 40.2, 12.0, 60.0], 48, 40) == [0, 40, 12, 48]      # clipped to the frame
    with pytest.raises(AssertionError, match='empty ROI'):
        main_seg_eval.roi_window([5.0, 5.0, 5.0, 9.0], 48, 40)


def test_eval_levels_runs_and_compare(synthetic_root, tmp_path, monkeypatch):
    root, det_ckpt, roi_dir = _setup(synthetic_root, tmp_path, monkeypatch)
    monkeypatch.setattr(main_seg_eval, 'build_seg_model', lambda a, n_in, P, q: BoxEcho(n_in))
    for name, arm, seed in (('r_s0', 'raw', 0), ('r_s1', 'raw', 1), ('g_s0', 'rgb', 0)):
        _write_run(tmp_path, name, arm, seed, det_ckpt)
    out = main_seg_eval.main(_eval_args(root, tmp_path, roi_dir, '--runs', 'r_s0', 'r_s1', 'g_s0'))

    run = tmp_path / 'seg' / 'r_s0'
    roi = json.loads((run / 'eval_roi_oracle.json').read_text())
    full = json.loads((run / 'eval_full_oracle.json').read_text())
    det = json.loads((run / 'eval_full_det.json').read_text())
    assert (roi['arm'], roi['seed'], roi['epoch'], roi['det_run']) == ('raw', 0, 3, 'det_A')
    # (a), (b): the oracle box is the object, so the box-echo mask is the GT exactly, at ROI level and pasted back
    assert roi['summary']['n'] == 2 and roi['summary']['IoU'] == pytest.approx(1.0) and roi['summary']['MAE'] < 1e-6
    assert full['summary']['n'] == 2 and full['summary']['IoU'] == pytest.approx(1.0) and full['summary']['S'] == pytest.approx(1.0, abs=1e-4)
    # (c): frame 7 = object + false-positive box (IoU 36/72), frame 8 = no ROI -> empty mask (IoU 0)
    assert det['summary']['IoU'] == pytest.approx(0.25) and det['summary']['fp_false_mask_rate'] == pytest.approx(1.0)
    assert (det['n_frames'], det['n_frames_no_roi'], det['n_det_rois'], det['n_fp_rois']) == (2, 1, 2, 1)
    assert set(det['paper']) == {'MAE', 'E_mean', 'S', 'F_adp'} and det['paper']['MAE'] == pytest.approx(36 / FRAME_PX, abs=1e-6)
    with open(run / 'per_image.pkl', 'rb') as f:
        pi = pickle.load(f)
    assert pi['roi_oracle']['keys'] == [('7', 0), ('8', 0)] and pi['full_det']['keys'] == ['7', '8']
    assert len(pi['full_det']['rows']) == 2                                  # the fp ROI is not a frame row
    assert [r['frame'] for r in pi['roi_oracle']['rows']] == ['7', '8']     # rows carry their frame: frame bootstrap
    # ... it is kept apart as an fp row, so rows + fp_rows re-pool to the written summaries, false-mask rate included
    assert [r['kind'] for r in pi['full_det']['fp_rows']] == ['fp'] and pi['full_det']['fp_rows'][0]['frame'] == '7'
    assert all(r['kind'] == 'obj' for lvl in pi for r in pi[lvl]['rows'])
    for lvl, summary in (('roi_oracle', roi['summary']), ('full_oracle', full['summary']), ('full_det', det['summary'])):
        pooled = json.loads(json.dumps(pool(pi[lvl]['rows'] + pi[lvl]['fp_rows']), default=main_seg_eval.to_json))
        assert pooled.keys() == summary.keys() and pooled == pytest.approx(summary, nan_ok=True), lvl

    # compare.json: seeds pooled per (model, arm); identical models -> zero differences with a degenerate CI
    cmp = json.loads((tmp_path / 'seg' / 'compare.json').read_text())
    assert cmp == json.loads(json.dumps(out, default=main_seg_eval.to_json))
    g = cmp['groups']['sam2unet|raw']
    assert g['seeds'] == [0, 1] and g['full_oracle']['IoU']['n'] == 2 and g['full_oracle']['IoU']['std'] == pytest.approx(0.0)
    assert set(cmp['comparisons']) == {'roi_oracle', 'full_oracle', 'full_det'}
    c = cmp['comparisons']['full_det']['sam2unet: raw - rgb']
    assert c['S']['diff'] == pytest.approx(0.0) and c['IoU']['lo'] == pytest.approx(0.0) and c['IoU']['hi'] == pytest.approx(0.0)


def test_downscaled_rois_paste_back_near_the_gt(synthetic_root, tmp_path, monkeypatch):
    # ROIs of >= 40 px on a 32 px canvas: placed at s < 1 and resized back, so the box-echo mask is the GT up to interpolation
    root, det_ckpt, roi_dir = _setup(synthetic_root, tmp_path, monkeypatch)
    monkeypatch.setattr(main_seg_eval, 'build_seg_model', lambda a, n_in, P, q: BoxEcho(n_in))
    _write_run(tmp_path, 'r_s0', 'raw', 0, det_ckpt)
    main_seg_eval.main(_eval_args(root, tmp_path, roi_dir, '--runs', 'r_s0', '--roi_min', '40'))
    with open(tmp_path / 'seg' / 'r_s0' / 'per_image.pkl', 'rb') as f:
        pi = pickle.load(f)
    full = json.loads((tmp_path / 'seg' / 'r_s0' / 'eval_full_oracle.json').read_text())['summary']
    # per-axis box mapping at integer rounding (6 x 7 px box map for the 6 x 6 object): IoU about 0.64, S about 0.85, not the 0.73 of a uniform scale
    assert len(pi['roi_oracle']['rows']) == 2 and full['IoU'] > 0.6 and full['S'] > 0.8


def test_zero_shot_control_uses_the_pretrained_stem(synthetic_root, tmp_path, monkeypatch):
    root, det_ckpt, roi_dir = _setup(synthetic_root, tmp_path, monkeypatch)
    calls = []
    def _build(a, n_in, P, q):
        calls.append((a.seg_model, a.arm, a.canvas, n_in, P.copy(), q.copy()))
        return BoxEcho(n_in, gain=20.0)
    monkeypatch.setattr(main_seg_eval, 'build_seg_model', _build)
    main_seg_eval.main(_eval_args(root, tmp_path, roi_dir, '--zero_shot', '--zero_shot_det_ckpt', str(det_ckpt), '--canvas', '32'))
    assert len(calls) == 2                                                     # the up-front check (CPU), then the pass
    (model, arm, canvas, n_in, P, q) = calls[-1]
    assert all(c[:4] == calls[-1][:4] for c in calls)
    assert (model, arm, canvas, n_in) == ('sam2box', 'rgb', 32, 3)
    assert np.array_equal(P, np.eye(3)) and np.array_equal(q, np.zeros(3))     # fold_stem(conv, I, 0) = the RGB stem itself
    det = json.loads((tmp_path / 'seg' / main_seg_eval.ZS_NAME / 'eval_full_det.json').read_text())
    assert det['seg_model'] == main_seg_eval.ZS_MODEL and det['ckpt'] == 'zero_shot' and det['summary']['IoU'] == pytest.approx(0.25)


@pytest.mark.parametrize('broken', ['unexpected_key', 'wrong_detector'])
def test_a_broken_run_fails_before_the_first_frame(synthetic_root, tmp_path, monkeypatch, broken):
    '''Every run's front end and checkpoint are checked up front: a bad last run must not cost the passes before it.'''
    root, det_ckpt, roi_dir = _setup(synthetic_root, tmp_path, monkeypatch)
    monkeypatch.setattr(main_seg_eval, 'build_seg_model', lambda a, n_in, P, q: BoxEcho(n_in))
    _write_run(tmp_path, 'r_s0', 'raw', 0, det_ckpt)
    if broken == 'unexpected_key':
        _write_run(tmp_path, 'r_s1', 'raw', 1, det_ckpt)
        path = tmp_path / 'weights' / 'r_s1' / 'model_best'
        ck = torch.load(path, map_location='cpu', weights_only=False)
        ck['model']['stray.weight'] = torch.zeros(1)
        torch.save(ck, path)
        match = 'unexpected keys'
    else:
        _write_run(tmp_path, 'r_s1', 'ec10', 1, det_ckpt)                        # an EC arm on the raw-band detector
        match = 'needs an EC detector'
    frames = []
    monkeypatch.setattr(main_seg_eval, 'evaluate_frame', lambda spec, oracle, det, gt, name, device, args: frames.append(name))
    with pytest.raises(AssertionError, match=match):
        main_seg_eval.main(_eval_args(root, tmp_path, roi_dir, '--runs', 'r_s0', 'r_s1', '--runs_per_pass', '1'))
    assert frames == [] and not (tmp_path / 'seg' / 'r_s0').exists()            # no pass ran, nothing written


def test_missing_roi_export_raises(synthetic_root, tmp_path, monkeypatch):
    root, det_ckpt, roi_dir = _setup(synthetic_root, tmp_path, monkeypatch)
    (roi_dir / 'rois_det_A_test.json').unlink()
    _write_run(tmp_path, 'r_s0', 'raw', 0, det_ckpt)
    with pytest.raises(AssertionError, match='rois_det_A_test.json missing'):
        main_seg_eval.main(_eval_args(root, tmp_path, roi_dir, '--runs', 'r_s0'))


@pytest.mark.parametrize('canvas', [32, 12])
def test_test_items_match_the_validation_items(synthetic_root, tmp_path, canvas):
    '''
    main_seg_eval cuts its ROIs from the full p99-scaled frame, HyperCOD_roi from the un-scaled crop cache. The same ROI
    must give the same tensors (canvas 32: s = 1; canvas 12: the 16 px detector ROI is downscaled), so the model is tested
    on exactly the input it was selected on.
    '''
    root, out, _, _, _ = _crop_cache(synthetic_root, tmp_path)
    ds = HyperCOD_roi(str(out), 'val', 'raw', train=False, roi_margin=1.5, roi_min=0, canvas=canvas)
    ref = HyperCOD_data(str(root), split='train', ids=['10'], use_filter=False, norm='p99', crop_size=0, filter_norm='none',
                        cache_dir=default_cache_dir(str(root)), out_dtype='float16')
    frame, gt, _ = ref[0]                                                      # [133, H, W] fp16 p99-scaled, [1, H, W]
    scales = []
    for i in range(len(ds)):
        img, mask, box_map, valid, meta = ds[i]
        (img2, mask2, box_map2, valid2, meta2), gt_crop = main_seg_eval.make_item(frame, gt[0] > 0.5, meta['roi'], meta['box'],
                                                                                 canvas, '10', meta['source'])
        assert tuple(meta2['offset']) == tuple(meta['offset']) and meta2['s'] == meta['s']
        assert tuple(meta2['roi_hw']) == tuple(meta['roi_hw'])
        assert [float(v) for v in meta2['roi']] == [float(v) for v in meta['roi']]
        np.testing.assert_array_equal(img2, img)                                # same fp16 p99 scaling and resize
        np.testing.assert_array_equal(mask2, mask)
        np.testing.assert_array_equal(valid2, valid)
        np.testing.assert_array_equal(box_map2, box_map)
        np.testing.assert_allclose(meta2['box_canvas'], meta['box_canvas'], atol=1e-6)
        scales.append(meta['s'])
    assert len(ds) == 2
    if canvas == 32:
        assert scales == [1.0, 1.0]
    else:
        assert scales[0] == 1.0 and scales[1] < 1.0                           # oracle 10 px fits, detector 16 px is downscaled
