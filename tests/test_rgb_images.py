import json
import os
import shutil
import types
import cv2
import numpy as np
import pytest
import torch
from PIL import Image

import main_det
import main_det_compare
import main_det_rois
from data_loader import my_dataset
from data_loader.band_stats import (RGB_STATS_FILENAME, default_rgb_stats_path, compute_rgb_stats, load_rgb_stats)
from data_loader.cube_cache import build_cube_cache
from data_loader.my_dataset import HyperCOD_data
from models.ec_yolo import adapt_first_conv_weight, build_detection_model
from models.filter_bank import build_filter_bank
from tests.conftest import H, W, OBJ_SLICE, RGB_LEVELS
from tests.test_main_det import _args
from tests.test_main_det_rois import _roi_args


def make(root, **kw):
    kw.setdefault('split', 'train'); kw.setdefault('crop_size', 0); kw.setdefault('seed', 0)
    return HyperCOD_data(data_path=str(root), rgb_images=True, **kw)


def frame(root, split, name):
    '''The fixture frame as an independent reader sees it: cv2 (BGR) -> RGB uint8 [H, W, 3].'''
    bgr = cv2.imread(str(root / split / 'RGB' / f'{name}.jpg'))
    assert bgr is not None and bgr.shape == (H, W, 3) and bgr.dtype == np.uint8
    return bgr[:, :, ::-1]


# ---------------------------------------------------------------------------------------------------- the dataset

def test_rgb_item_is_the_jpeg_over_255(synthetic_root):
    root, info, _ = synthetic_root
    ds = make(root, split='test')
    img, gt, name = ds[0]
    ref = frame(root, 'test', '7')                                              # [H, W, 3] uint8
    assert name == '7' and img.shape == (3, H, W) and img.dtype == np.float32
    np.testing.assert_array_equal(img, ref.transpose(2, 0, 1).astype(np.float32) / 255.0)   # /255, nothing else: no p99, no band window
    assert img.mean(axis=(1, 2)).tolist() == sorted(img.mean(axis=(1, 2)).tolist(), reverse=True)   # R > G > B, the fixture's levels
    assert abs(img[0].mean() * 255 - RGB_LEVELS[0]) < 15 and abs(img[2].mean() * 255 - RGB_LEVELS[2]) < 15   # not a BGR frame
    pil = np.asarray(Image.open(root / 'test' / 'RGB' / '7.jpg'))               # PIL reads RGB natively: a second, independent decoder
    assert np.abs(pil.astype(int) - ref.astype(int)).max() <= 3
    assert gt.shape == (1, H, W) and gt.dtype == np.float32
    np.testing.assert_array_equal(gt[0] > 0.5, info[('test', '7')][1])          # the full-frame GT, as in the cube path
    assert ds.in_channels == 3 and (ds.H, ds.W) == (H, W)


def test_rgb_item_honours_out_dtype(synthetic_root):
    root, _, _ = synthetic_root
    img16, _, _ = make(root, split='test', out_dtype='float16')[0]
    img32, _, _ = make(root, split='test', out_dtype='float32')[0]
    assert img16.dtype == np.float16 and img16.shape == (3, H, W)
    np.testing.assert_allclose(img16.astype(np.float32), img32, atol=5e-4, rtol=0)   # fp16 resolves [0, 1] to 5e-4, finer than the 1/255 grid


def test_rgb_train_crop_follows_the_cube_paths_window_and_gt(synthetic_root):
    root, info, _ = synthetic_root
    cs = 16
    ref = make(root, crop_size=cs, obj_crop_prob=0.5)
    h0, w0, ch, cw = ref.crop_window(ref.load_gt('3'))                          # the window the first item of an identically seeded dataset draws
    img, gt, name = make(root, crop_size=cs, obj_crop_prob=0.5)[0]
    assert name == '3' and (ch, cw) == (cs, cs) and img.shape == (3, cs, cs) and gt.shape == (1, cs, cs)
    np.testing.assert_array_equal(img, frame(root, 'train', '3')[h0:h0 + ch, w0:w0 + cw].transpose(2, 0, 1).astype(np.float32) / 255.0)
    np.testing.assert_array_equal(gt[0] > 0.5, info[('train', '3')][1][h0:h0 + ch, w0:w0 + cw])
    # the cube path of an identically seeded dataset draws the very same crop and GT
    _, gt_cube, _ = HyperCOD_data(str(root), split='train', use_filter=False, norm='none', crop_size=cs, obj_crop_prob=0.5, seed=0)[0]
    np.testing.assert_array_equal(gt, gt_cube)


def test_rgb_mode_needs_neither_the_cube_cache_nor_band_statistics(synthetic_root, tmp_path):
    root, _, _ = synthetic_root
    shutil.rmtree(root / 'train' / 'intensity map')                             # no p99 csv either: nothing of the cube is read
    stats = tmp_path / 'never_written.npz'
    ds = make(root, norm='p99z', cache_dir=str(tmp_path / 'no_such_cache'), stats_path=str(stats))
    img, _, _ = ds[0]
    assert img.shape == (3, H, W) and not stats.exists() and not (root / 'band_stats_train_400_800.npz').exists()
    assert ds.scale is None and ds.channel_mean is None and ds.channel_std is None and ds.norm == 'none'
    assert ds.in_channels == 3


def test_rgb_mode_asserts_every_frame_has_its_jpeg(synthetic_root):
    root, _, _ = synthetic_root
    (root / 'train' / 'RGB' / '10.jpg').unlink()
    with pytest.raises(AssertionError, match=r"RGB for sample 10 not found"):
        make(root)


def test_rgb_mode_asserts_the_jpeg_matches_the_gt_size(synthetic_root):
    root, _, _ = synthetic_root
    cv2.imwrite(str(root / 'train' / 'RGB' / '3.jpg'), np.zeros((H + 2, W, 3), np.uint8))   # a transposed / rotated camera frame
    with pytest.raises(AssertionError, match=r"RGB frame 3 has shape"):
        make(root)[0]


def test_dataset_kwargs_carries_rgb_images(synthetic_root, tmp_path):
    root, _, _ = synthetic_root
    assert main_det.dataset_kwargs(_args(root, tmp_path))['rgb_images'] is False
    assert main_det.dataset_kwargs(_args(root, tmp_path, **{'--rgb_images': None}))['rgb_images'] is True
    assert main_det.get_args_parser().parse_args([]).rgb_images is False


# -------------------------------------------------------------------------------------------------------- the stats

def _direct(root, names, crops=None):
    '''Mean [3] and population covariance [3, 3] of the /255 values of the train frames (or of their crops), float64.'''
    xs = []
    for name in names:
        f = frame(root, 'train', name).astype(np.float64) / 255.0              # [H, W, 3]
        if crops is not None:
            h0, w0, ch, cw = crops[name]
            f = f[h0:h0 + ch, w0:w0 + cw]
        xs.append(f.reshape(-1, 3))
    x = np.concatenate(xs).T                                                    # [3, P]
    return x.mean(axis=1), np.cov(x, bias=True)


def test_compute_rgb_stats_full_frames_match_direct(synthetic_root, tmp_path):
    root, _, _ = synthetic_root
    out = tmp_path / 'rgb.npz'
    mean, cov = compute_rgb_stats(str(root), str(out), crop_size=0)
    mu, sigma = _direct(root, ['3', '10'])
    np.testing.assert_allclose(mean, mu, rtol=1e-6); np.testing.assert_allclose(cov, sigma, rtol=1e-5, atol=1e-10)
    assert mean.shape == (3,) and cov.shape == (3, 3) and np.array_equal(cov, cov.T)
    st = np.load(out)
    assert int(st['n_pixels']) == 2 * H * W and int(st['n_samples']) == 2 and int(st['crop_size']) == 0 and int(st['seed']) == 0
    m2, c2 = load_rgb_stats(str(out), crop_size=0, seed=0, n_samples=2)
    np.testing.assert_array_equal(m2, mean); np.testing.assert_array_equal(c2, cov)


def test_compute_rgb_stats_uses_one_seeded_uniform_crop_per_train_frame(synthetic_root, tmp_path):
    root, _, _ = synthetic_root
    cs = 16
    mean, cov = compute_rgb_stats(str(root), str(tmp_path / 'rgb.npz'), crop_size=cs, seed=0)
    ref = make(root, crop_size=cs, obj_crop_prob=0.0, seed=0)                   # the same stream: one uniform crop per frame, in id order
    crops = {name: ref.crop_window(ref.load_gt(name)) for name in ref.img_name}
    mu, sigma = _direct(root, ref.img_name, crops)
    np.testing.assert_allclose(mean, mu, rtol=1e-6); np.testing.assert_allclose(cov, sigma, rtol=1e-5, atol=1e-10)
    assert int(np.load(tmp_path / 'rgb.npz')['n_pixels']) == 2 * cs * cs
    again, _ = compute_rgb_stats(str(root), str(tmp_path / 'rgb2.npz'), crop_size=cs, seed=0)
    np.testing.assert_array_equal(again, mean)                                  # reproducible
    other, _ = compute_rgb_stats(str(root), str(tmp_path / 'rgb3.npz'), crop_size=cs, seed=1)
    assert not np.array_equal(other, mean)


def test_rgb_stats_default_path_is_next_to_the_data(synthetic_root):
    root, _, _ = synthetic_root
    assert RGB_STATS_FILENAME == 'rgb_stats_train.npz' and default_rgb_stats_path(str(root)) == str(root / 'rgb_stats_train.npz')
    assert make(root).rgb_stats_path == str(root / 'rgb_stats_train.npz')


def test_dataset_rgb_stats_are_computed_once_then_loaded(synthetic_root, monkeypatch):
    root, _, _ = synthetic_root
    ds = make(root, split='test')                                               # the stats are always the TRAIN frames', whatever split asks
    assert not (root / 'rgb_stats_train.npz').exists()
    mu, cov = ds._rgb_stats()
    assert (root / 'rgb_stats_train.npz').exists()
    ref_mu, ref_cov = _direct(root, ['3', '10'])                                # fixture frames are smaller than 512: full frames
    np.testing.assert_allclose(mu, ref_mu, rtol=1e-6); np.testing.assert_allclose(cov, ref_cov, rtol=1e-5, atol=1e-10)

    def _recomputed(*a, **k):
        raise AssertionError("the cached RGB statistics were recomputed")
    monkeypatch.setattr(my_dataset, 'compute_rgb_stats', _recomputed)
    mu2, cov2 = make(root, split='train')._rgb_stats()
    np.testing.assert_array_equal(mu2, mu); np.testing.assert_array_equal(cov2, cov)


@pytest.mark.parametrize('key, bad, message', [('crop_size', 512, 'crop_size'), ('seed', 7, 'seed'), ('n_samples', 5, 'n_samples'),
                                               ('norm', 'p99', 'norm')])
def test_a_cached_rgb_stats_file_must_match_its_metadata(synthetic_root, key, bad, message):
    root, _, _ = synthetic_root
    ds = make(root)
    ds._rgb_stats()
    path = root / 'rgb_stats_train.npz'
    st = dict(np.load(path)); st[key] = np.asarray(bad)
    np.savez(path, **st)
    with pytest.raises(AssertionError, match=message):
        make(root)._rgb_stats()


def test_load_rgb_stats_validates_shapes(tmp_path):
    np.savez(tmp_path / 'bad.npz', mean=np.zeros(4), cov=np.zeros((4, 4)), crop_size=0, seed=0, n_samples=1, norm='rgb255')
    with pytest.raises(AssertionError, match='mean'):
        load_rgb_stats(str(tmp_path / 'bad.npz'))


# ------------------------------------------------------------------------------------------------------ the front end

def test_build_filter_bank_rgb_is_identity_with_rgb_statistics_and_no_gate(synthetic_root):
    root, _, _ = synthetic_root
    ds = make(root)
    args = types.SimpleNamespace(session='A', rgb_images=True)                  # a sparse namespace: every other flag keeps its getattr default
    fb, volts = build_filter_bank(args, ds)
    mu, cov = ds._rgb_stats()
    assert fb.n_channels == fb.n_readings == 3 and fb.R_t.shape == (3, 3)
    assert torch.equal(fb.R_t, torch.eye(3))                                    # identity front end
    assert not fb.weight_vector and 'theta' not in fb.state_dict() and list(fb.parameters()) == []   # no gate
    assert fb.noise_std is None and fb.proj is None                             # no read noise, no whitening
    np.testing.assert_allclose(fb.mean.view(-1).numpy(), mu.astype(np.float32))
    np.testing.assert_allclose(fb.std.view(-1).numpy(), np.sqrt(np.diag(cov)).astype(np.float32))
    assert fb.std.dtype == torch.float32 and fb.mean.dtype == torch.float32
    assert isinstance(volts, np.ndarray) and volts.tolist() == [0.0, 1.0, 2.0]  # channel index (R, G, B); numeric like every other branch's
    assert not os.path.exists(ds.stats_path)                                    # the cube's band statistics were never built
    # the standardisation: over the frames the statistics came from, every channel ends up zero-mean, unit-variance
    ys = torch.cat([fb(torch.from_numpy(ds[i][0])[None]).reshape(3, -1) for i in range(len(ds))], dim=1)   # [3, P]
    torch.testing.assert_close(ys.mean(dim=1), torch.zeros(3), atol=1e-4, rtol=0)
    torch.testing.assert_close(ys.var(dim=1, unbiased=False), torch.ones(3), atol=1e-3, rtol=0)
    img = ds[0][0]
    np.testing.assert_allclose(fb(torch.from_numpy(img)[None])[0].numpy(), (img - mu[:, None, None]) / np.sqrt(np.diag(cov))[:, None, None],
                               atol=1e-4, rtol=1e-4)


def test_build_filter_bank_rgb_ignores_the_gate_flag_of_session_a(synthetic_root):
    root, _, _ = synthetic_root
    fb, _ = build_filter_bank(types.SimpleNamespace(session='A', rgb_images=True, no_gate=False), make(root))
    assert not fb.weight_vector and fb.entropy().item() == 0.0 and torch.equal(fb.weights, torch.ones(3))


RGB_CONFLICTS = {
    'raw_bands': (dict(raw_bands=True), 'raw-bands'),
    'pca_channels': (dict(pca_channels=3), 'pca-channels'),
    'read_noise_db': (dict(read_noise_db=40.0), 'read_noise_db'),
    'session_b': (dict(session='B'), 'session B'),
}


@pytest.mark.parametrize('case', sorted(RGB_CONFLICTS))
def test_rgb_images_conflicts_raise_in_build_filter_bank(synthetic_root, case):
    root, _, _ = synthetic_root
    extra, message = RGB_CONFLICTS[case]
    args = types.SimpleNamespace(**{'session': 'A', 'rgb_images': True, **extra})
    with pytest.raises(AssertionError, match=message):
        build_filter_bank(args, make(root))


@pytest.mark.parametrize('flag, value, message', [('--raw-bands', None, 'raw-bands'), ('--pca-channels', '3', 'pca-channels'),
                                                 ('--read_noise_db', '40', 'read_noise_db'), ('--session', 'B', 'session B')])
def test_rgb_images_conflicts_stop_main_before_anything_is_built(synthetic_root, tmp_path, monkeypatch, flag, value, message):
    root, _, _ = synthetic_root
    monkeypatch.delenv('RANK', raising=False)
    out = tmp_path / 'rgb_conflict'
    with pytest.raises(AssertionError, match=message):
        main_det.main(_args(root, tmp_path, **{'--rgb_images': None, flag: value, '--name': 'conflict', '--output-dir': str(out)}))
    assert not out.exists()                                                     # failed before the output folder, W&B or the data loaders


def test_the_rgb_first_conv_is_the_pretrained_coco_kernel(tmp_path):
    from ultralytics import YOLO
    path = str(tmp_path / 'yolo26n_stand_in.pt')
    YOLO('yolo26n.yaml').save(path)                                             # a 3-channel checkpoint in the COCO checkpoint's format, offline
    w_rgb = YOLO(path).model.float().state_dict()['model.0.conv.weight']        # [16, 3, 3, 3]
    torch.testing.assert_close(adapt_first_conv_weight(w_rgb, 3), w_rgb, rtol=0, atol=0)   # tiling i mod 3 with scale 3/3: the identity
    dm, n_matched, _ = build_detection_model('yolo26n', 3, pretrained_path=path, nc=1)
    assert n_matched > 0
    torch.testing.assert_close(dm.model[0].conv.weight.detach(), w_rgb, rtol=0, atol=0)   # exactly the pretrained stem, not a re-initialised one
    assert tuple(dm.model[0].conv.weight.shape) == (16, 3, 3, 3)


# ------------------------------------------------------------------------------------ training, rebuilding, comparing

@pytest.fixture
def rgb_run(synthetic_root, tmp_path, monkeypatch):
    '''One epoch of `--session A --rgb_images` on the fixture, WITHOUT building the cube cache: (root, run folder).'''
    root, _, _ = synthetic_root
    monkeypatch.delenv('RANK', raising=False)
    out = tmp_path / 'rgb_A'
    main_det.main(_args(root, tmp_path, **{'--session': 'A', '--rgb_images': None, '--name': 'rgb_A', '--output-dir': str(out), '--top_k': '3'}))
    assert not os.path.exists(os.path.join(root, 'cache_fp16'))                 # nothing of the cube was ever needed
    return root, out


def test_rgb_session_trains_and_writes_a_three_channel_checkpoint(rgb_run, tmp_path):
    root, out = rgb_run
    ck = torch.load(out / 'model_best', map_location='cpu', weights_only=False)
    assert ck['args']['rgb_images'] is True and ck['model']['yolo.model.0.conv.weight'].shape[1] == 3
    assert 'filter_bank.theta' not in ck['model'] and torch.equal(ck['model']['filter_bank.R_t'], torch.eye(3))
    mu, cov = load_rgb_stats(str(root / 'rgb_stats_train.npz'))
    np.testing.assert_allclose(ck['model']['filter_bank.mean'].view(-1).numpy(), mu.astype(np.float32))
    np.testing.assert_allclose(ck['model']['filter_bank.std'].view(-1).numpy(), np.sqrt(np.diag(cov)).astype(np.float32))
    assert ck['selected_voltages'] == [0.0, 1.0, 2.0] and ck['selected_indices'] == [0, 1, 2]
    assert not (out / 'gate_ranking.csv').exists() and not (out / 'top_k.json').exists()     # no gate, nothing to rank
    recs = [json.loads(line) for line in (out / 'results_rgb_A.txt').read_text().strip().splitlines()]
    assert recs[-1]['final'] is True and recs[-1]['val']['n_images'] >= 1 and recs[-1]['test']['n_images'] == 1
    # --eval of the checkpoint: the model is rebuilt from the CLI flags, so --rgb_images is passed again (as --raw-bands is)
    main_det.main(_args(root, tmp_path, **{'--session': 'A', '--rgb_images': None, '--name': 'rgb_eval', '--output-dir': str(out),
                                           '--resume': str(out / 'model_best'), '--eval': None}))
    rec = json.loads((out / 'results_rgb_eval.txt').read_text().strip().splitlines()[-1])
    assert rec['eval'] is True and rec['test']['n_images'] == 1


def test_det_model_keys_include_rgb_images(tmp_path):
    assert 'rgb_images' in main_det_rois.DET_MODEL_KEYS
    cli = main_det_rois.get_args_parser().parse_args(['--data-path', str(tmp_path), '--rgb_images', '--pretrained', 'none'])
    # a checkpoint without the key (every run before the RGB baseline) stays a cube detector, whatever the CLI says ...
    assert main_det_rois.detector_args({'session': 'A', 'raw_bands': True}, cli).rgb_images is False
    # ... and an RGB checkpoint rebuilds as RGB without the flag on the command line
    assert main_det_rois.detector_args({'session': 'A', 'rgb_images': True}, main_det_rois.get_args_parser().parse_args([])).rgb_images is True


def test_rgb_detector_rebuilds_from_its_checkpoint_in_the_roi_export_and_compare(rgb_run, tmp_path):
    root, out = rgb_run
    ck = torch.load(out / 'model_best', map_location='cpu', weights_only=False)
    # Stage-2 / ROI-export route: nothing of the model repeated on the command line
    args = _roi_args(root, tmp_path, resume=out / 'model_best', **{'--roi-conf': '0.0'})
    model, det_args, run = main_det_rois.load_detector(str(out / 'model_best'), args, torch.device('cpu'))
    assert run == 'rgb_A' and det_args.rgb_images is True and model.filter_bank.n_channels == 3 and not model.training
    assert all(torch.equal(v, ck['model'][k]) for k, v in model.state_dict().items())       # the checkpoint, loaded strictly
    path = main_det_rois.main(args)                                             # reads the test frames as RGB, no cube cache
    with open(path) as f:
        data = json.load(f)
    assert set(data) == {'7'} and data['7']['gt_boxes'] == [[20.0, 10.0, 26.0, 16.0]]
    # main_det_compare.load_checkpoint_model, unchanged: the checkpoint's own args carry rgb_images
    cargs = main_det_compare.get_args_parser().parse_args(
        ['--runs', 'rgb_A', '--weights_dir', str(tmp_path), '--data-path', str(root), '--split-file', str(tmp_path / 'val.json'),
         '--device', 'cpu', '--yolo-variant', 'yolo26n', '--pretrained', 'none', '--roi-min', '0', '--min-area', '10'])
    cmodel, a, _ = main_det_compare.load_checkpoint_model(cargs, 'rgb_A', 0, torch.device('cpu'))
    assert a.rgb_images is True and cmodel.filter_bank.n_channels == 3 and not cmodel.training
    assert all(torch.equal(v, ck['model'][k]) for k, v in cmodel.state_dict().items())
    img = torch.from_numpy(make(root, split='test', out_dtype='float16')[0][0])[None]        # [1, 3, H, W] fp16 RGB frame
    pred = cmodel(img)                                                          # the rebuilt model runs on the dataset's RGB frame
    assert torch.isfinite((pred[0] if isinstance(pred, (list, tuple)) else pred).float()).all()


@pytest.mark.filterwarnings("ignore:All-NaN slice encountered:RuntimeWarning")   # an untrained detector matches no GT: matched_iou is NaN in every draw
def test_compare_gives_every_run_its_own_input(rgb_run, tmp_path, monkeypatch):
    '''A cube run and an RGB run in one comparison: one pass over the frames, each model fed the input it was trained on.'''
    root, out = rgb_run
    build_cube_cache(str(root), 'train', num_workers=0); build_cube_cache(str(root), 'test', num_workers=0)
    main_det.main(_args(root, tmp_path, **{'--session': 'A', '--name': 'ec_A', '--output-dir': str(tmp_path / 'ec_A'), '--top_k': '3'}))
    base = ['--weights_dir', str(tmp_path), '--data-path', str(root), '--cache-dir', os.path.join(str(root), 'cache_fp16'),
            '--split-file', str(tmp_path / 'val.json'), '--device', 'cpu', '--yolo-variant', 'yolo26n', '--pretrained', 'none',
            '--num_workers', '0', '--roi-min', '0', '--min-area', '10', '--ckpt_epochs', '0', '--conditions', 'clean', '--n_boot', '3']
    results, comps = main_det_compare.main(main_det_compare.get_args_parser().parse_args(
        base + ['--runs', 'rgb_A', 'ec_A', '--out_dir', str(tmp_path / 'compare')]))
    assert set(results['test']) == {'rgb_A/model_0|clean', 'ec_A/model_0|clean'}
    assert all(r['n_images'] == 1 for r in results['test'].values()) and 'rgb_A - ec_A | clean' in comps
    with open(tmp_path / 'compare' / 'compare.json') as f:
        assert json.load(f)['runs'] == ['rgb_A', 'ec_A']
    # the RGB run's numbers do not depend on what it is compared with (its own frames, its own checkpoint)
    # ... and a comparison of RGB runs only never touches the cube cache (a later --cache-dir overrides the one in base)
    alone, _ = main_det_compare.main(main_det_compare.get_args_parser().parse_args(
        base + ['--cache-dir', str(tmp_path / 'no_cache'), '--runs', 'rgb_A', '--out_dir', str(tmp_path / 'alone')]))
    np.testing.assert_equal(alone['test']['rgb_A/model_0|clean'], results['test']['rgb_A/model_0|clean'])
    cube_only, _ = main_det_compare.main(main_det_compare.get_args_parser().parse_args(base + ['--runs', 'ec_A', '--out_dir', str(tmp_path / 'cube_only')]))
    np.testing.assert_equal(cube_only['test']['ec_A/model_0|clean'], results['test']['ec_A/model_0|clean'])   # the cube run: the pre-RGB path
    with pytest.raises(AssertionError, match='rgb_images'):                     # the input type is each run's own, not a comparison flag
        main_det_compare.main(main_det_compare.get_args_parser().parse_args(base + ['--runs', 'rgb_A', '--rgb_images', '--out_dir', str(tmp_path / 'x')]))
