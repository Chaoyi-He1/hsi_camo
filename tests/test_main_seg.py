import json
import os
import numpy as np
import pytest
pytest.importorskip('py_sod_metrics')        # main_seg -> train_eval_seg -> seg_metrics needs it
import torch
import yaml

import main_det
import main_seg
from data_loader.my_dataset import HyperCOD_data
from models.filter_bank import build_filter_bank
from tests.test_train_eval_seg import TinySeg

CANVAS = 32


def _det_ckpt(root, path, *argv):
    '''A Stage-1 checkpoint stub: the detector's flags and its filter_bank.* tensors (all main_seg reads from it).'''
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    det_args = main_det.get_args_parser().parse_args(['--data-path', str(root), *argv])
    dataset = HyperCOD_data(split='train', **main_det.dataset_kwargs(det_args))
    fb, _ = build_filter_bank(det_args, dataset)
    torch.save({'args': vars(det_args), 'model': {f'filter_bank.{k}': v for k, v in fb.state_dict().items()}, 'epoch': 0}, path)
    return path


def _crop_cache(cache):
    '''
    A tiny crop cache in build_crop_cache's format (Task 3's index keys; ROI provenance = raw133_A): train frame '3' with
    9 object windows (32 x 32, one object and one matched raw-arm ROI each) and 2 false-positive windows; val frame '10'
    with 2 object windows.
    '''
    rng = np.random.default_rng(0)
    os.makedirs(cache, exist_ok=True)
    windows = []

    def add(frame, split, kind, k, window, objects, rois):
        x1, y1, x2, y2 = window
        img = (rng.random((133, y2 - y1, x2 - x1)) * 0.02).astype(np.float16)            # [133, h, w] un-scaled
        gt = np.zeros((y2 - y1, x2 - x1), dtype=bool)                                    # [h, w]
        for o in objects:
            bx1, by1, bx2, by2 = o['box']
            gt[by1 - y1:by2 - y1, bx1 - x1:bx2 - x1] = True
        np.save(os.path.join(cache, f'{frame}_{k}.npy'), img)
        np.save(os.path.join(cache, f'{frame}_{k}_gt.npy'), gt)
        windows.append({'file': f'{frame}_{k}.npy', 'gt_file': f'{frame}_{k}_gt.npy', 'frame': frame, 'split': split, 'kind': kind,
                        'window': list(window), 'scale': 0.011, 'object': objects[0]['id'] if kind == 'object' else -1,
                        'objects': objects, 'rois': {'raw': rois, 'ec10': [], 'ec24': []}})

    det = lambda oid: [{'roi': [10.0, 13.0, 26.0, 33.0], 'box': [13.0, 17.0, 23.0, 29.0], 'conf': 0.7, 'object': oid}]
    for k in range(9):
        add('3', 'train', 'object', k, (4, 8, 36, 40), [{'id': k + 1, 'box': [14, 18, 22, 28], 'area': 80}], det(k + 1))
    for k in range(9, 11):
        add('3', 'train', 'fp', k, (0, 0, 16, 16), [], [{'roi': [0.0, 0.0, 16.0, 16.0], 'box': [4.0, 4.0, 12.0, 12.0], 'conf': 0.3, 'object': -1}])
    for k in range(2):
        add('10', 'val', 'object', k, (4, 8, 36, 40), [{'id': k + 1, 'box': [14, 18, 22, 28], 'area': 80}], det(k + 1))
    with open(os.path.join(cache, 'index.json'), 'w') as f:
        json.dump({'band_range': [400.0, 800.0], 'frame_hw': [48, 40], 'grow': 2.0, 'min_side': 32, 'min_area': 10,
                   'min_cover': 0.01, 'arms': ['raw', 'ec10', 'ec24'],
                   'roi_files': {a: {s: f'results/det/rois_raw133_A_{s}.json' for s in ('train', 'val')} for a in ('raw', 'ec10', 'ec24')},
                   'windows': windows}, f)


def _hpy(path):
    '''cfg/seg.yaml shrunk to the fixture: 32 px canvas and ROIs, a small stem fit, batch 4, model_last every epoch.'''
    with open('cfg/seg.yaml') as f:
        cfg = yaml.safe_load(f)
    cfg.update(canvas=CANVAS, roi_min=8, fit_items=4, fit_pixels=2000, save_every=1, print_freq=1)
    cfg['models'] = {k: {**v, 'batch_size': 4, 'accumulate': 1} for k, v in cfg['models'].items()}
    with open(path, 'w') as f:
        yaml.safe_dump(cfg, f)


def _args(root, tmp_path, det_ckpt, *extra):
    argv = ['--seg_model', 'sam2unet', '--arm', 'raw', '--det_ckpt', str(det_ckpt), '--data_path', str(root),
            '--crop_cache', str(tmp_path / 'crops'), '--hpy', str(tmp_path / 'seg.yaml'), '--device', 'cpu', '--num_workers', '0',
            '--no-wandb', '--runs_dir', str(tmp_path / 'runs'), '--output_dir', str(tmp_path / 'seg'), '--name', 'tseg', *extra]
    return main_seg.get_args_parser().parse_args(argv)


def _setup(root, tmp_path, monkeypatch):
    '''Raw-arm detector stub (run raw133_A), crop cache, shrunk cfg, and build_seg_model replaced by TinySeg (records n_in, P, q).'''
    det = _det_ckpt(root, tmp_path / 'raw133_A' / 'model_best', '--raw-bands', '--filter-select', 'uniform', '--num-filters', '6')
    _crop_cache(tmp_path / 'crops')
    _hpy(tmp_path / 'seg.yaml')
    built = []
    monkeypatch.setattr(main_seg, 'build_seg_model', lambda args, n_in, P, q: built.append((n_in, np.asarray(P), np.asarray(q))) or TinySeg(n_in))
    return det, built


def test_load_cfg_fills_unset_flags_and_the_model_section_wins():
    args = main_seg.get_args_parser().parse_args(['--seg_model', 'zoomnext', '--lr', '5e-4'])
    main_seg.load_cfg(args)
    assert args.lr == 5e-4                                                         # an explicit flag wins
    assert (args.batch_size, args.accumulate, args.encoder_lr_mult) == (4, 2, 1.0)  # the zoomnext section
    assert args.epochs == 200 and args.canvas == 512 and list(args.box_mix) == [0.5, 0.4, 0.1] and args.gt_jitter == 0.15
    assert isinstance(args.weight_decay, float) and args.weight_decay == 1e-4 and args.max_norm == 1.0 and args.amp_dtype == 'bfloat16'
    sam = main_seg.get_args_parser().parse_args(['--seg_model', 'sam2unet'])
    main_seg.load_cfg(sam)
    assert (sam.lr, sam.stem_lr, sam.batch_size, sam.accumulate) == (1e-3, 1e-4, 8, 1)


def test_det_args_come_from_the_checkpoint_and_must_match_the_arm(synthetic_root, tmp_path):
    root, _, _ = synthetic_root
    ec = _det_ckpt(root, tmp_path / 'sel_det' / 'model_best', '--filter-select', 'uniform', '--num-filters', '12', '--pca-channels', '10')
    ck = torch.load(ec, map_location='cpu', weights_only=False)
    ck['args']['data_path'] = '/nonexistent'                                       # the checkpoint never moves the data
    for k in ('read_noise_db', 'no_gate'):                                          # an older checkpoint without these flags
        ck['args'].pop(k)
    torch.save(ck, ec)
    args = main_seg.get_args_parser().parse_args(['--data_path', str(root), '--device', 'cpu'])
    det_args, _ = main_seg.det_args_from_ckpt(str(ec), args)
    assert det_args.data_path == str(root) and det_args.device == 'cpu' and det_args.name == 'sel_det'
    assert det_args.pca_channels == 10 and det_args.num_filters == 12 and det_args.read_noise_db == 0.0 and det_args.no_gate is False
    main_seg.check_arm('ec10', det_args)
    with pytest.raises(AssertionError, match='--pca-channels 24'):
        main_seg.check_arm('ec24', det_args)
    with pytest.raises(AssertionError, match='raw-bands'):
        main_seg.check_arm('raw', det_args)
    with pytest.raises(AssertionError, match='not found'):
        main_seg.det_args_from_ckpt(str(tmp_path / 'missing'), args)


def test_main_seg_rejects_a_front_end_that_differs_from_the_detector(synthetic_root, tmp_path, monkeypatch):
    root, _, _ = synthetic_root
    det, _ = _setup(root, tmp_path, monkeypatch)
    ck = torch.load(det, map_location='cpu', weights_only=False)
    ck['model']['filter_bank.mean'] = ck['model']['filter_bank.mean'] + 1e-3        # not what the flags rebuild
    bad = tmp_path / 'bad' / 'raw133_A' / 'model_best'                            # same run name, so only the tensors differ
    os.makedirs(bad.parent)
    torch.save(ck, bad)
    with pytest.raises(AssertionError, match='detector checkpoint'):
        main_seg.main(_args(root, tmp_path, bad, '--epochs', '1'))


def test_dry_run_checks_one_batch_per_box_source_and_writes_nothing(synthetic_root, tmp_path, monkeypatch):
    root, _, _ = synthetic_root
    det, built = _setup(root, tmp_path, monkeypatch)
    info = main_seg.main(_args(root, tmp_path, det, '--dry_run'))
    assert info['n_in'] == 133 and info['sources'] and set(info['sources']) <= {'gt', 'det', 'fp'}
    assert all(shape[1:] == (134, CANVAS, CANVAS) for shape in info['sources'].values())
    assert built[0][0] == 133 and built[0][1].shape == (3, 133) and built[0][2].shape == (3,)
    assert not (tmp_path / 'seg').exists()


def test_main_seg_trains_resumes_and_evaluates(synthetic_root, tmp_path, monkeypatch):
    root, _, _ = synthetic_root
    det, built = _setup(root, tmp_path, monkeypatch)
    out = tmp_path / 'seg'
    main_seg.main(_args(root, tmp_path, det, '--epochs', '2'))
    lines = [json.loads(l) for l in (out / 'results_tseg.txt').read_text().strip().splitlines()]
    assert [l['epoch'] for l in lines[:2]] == [0, 1] and lines[-1]['final'] is True and len(lines) == 3
    assert {'S', 'Fw', 'IoU', 'MAE', 'n'} <= set(lines[0]['val']) and lines[0]['val']['n'] == 4   # 2 oracle + 2 matched val ROIs
    assert {'loss', 'lr', 'main', 'side'} <= set(lines[0]['train'])
    ck = torch.load(out / 'model_last', map_location='cpu', weights_only=False)
    assert ck['epoch'] == 1 and ck['P'].shape == (3, 133) and ck['q'].shape == (3,) and ck['n_in'] == 133
    assert ck['args']['arm'] == 'raw' and ck['args']['seg_model'] == 'sam2unet' and (out / 'model_best').exists()
    np.testing.assert_array_equal(ck['P'], built[0][1])

    # resume: one more epoch from model_last with the stored stem fold (no refit) and the best score carried over
    def no_refit(*a, **k):
        raise AssertionError('a resumed run must reuse the stored (P, q)')
    monkeypatch.setattr(main_seg, 'fit_arm_rgb_map', no_refit)
    main_seg.main(_args(root, tmp_path, det, '--epochs', '3', '--resume', str(out / 'model_last')))
    lines = [json.loads(l) for l in (out / 'results_tseg.txt').read_text().strip().splitlines()]
    assert lines[-2]['epoch'] == 2 and lines[-2]['best'] >= lines[1]['best'] and lines[-1]['final'] is True
    np.testing.assert_array_equal(built[1][1], built[0][1])
    with pytest.raises(AssertionError, match='seg_model'):                         # another model's checkpoint
        main_seg.main(_args(root, tmp_path, det, '--seg_model', 'zoomnext', '--resume', str(out / 'model_last')))

    main_seg.main(_args(root, tmp_path, det, '--resume', str(out / 'model_best'), '--eval'))
    last = json.loads((out / 'results_tseg.txt').read_text().strip().splitlines()[-1])
    assert last['eval'] is True and 'S' in last['val'] and last['val']['n'] == 4


class PairSeg(TinySeg):
    '''TinySeg that, like ZoomNeXt (BatchNorm on a globally pooled map), refuses a training batch of one item.'''

    def forward(self, x, box_xyxy):
        assert x.shape[0] >= 2 or not self.training, f"batch of {x.shape[0]} item in training"
        return super().forward(x, box_xyxy)


def test_training_never_sees_a_batch_of_one(synthetic_root, tmp_path, monkeypatch):
    root, _, _ = synthetic_root
    det, _ = _setup(root, tmp_path, monkeypatch)
    monkeypatch.setattr(main_seg, 'build_seg_model', lambda args, n_in, P, q: PairSeg(n_in))
    # 9 object items + round(9 x 0.1 / 0.9) = 1 false-positive item = 10 items per epoch: batch 3 leaves a last batch of 1
    main_seg.main(_args(root, tmp_path, det, '--epochs', '1', '--batch_size', '3'))
    lines = [json.loads(l) for l in (tmp_path / 'seg' / 'results_tseg.txt').read_text().strip().splitlines()]
    assert lines[-1]['final'] is True
    # the dry run finds a single 'fp' item: it must still run that source as a batch of >= 2
    info = main_seg.main(_args(root, tmp_path, det, '--dry_run', '--batch_size', '3'))
    assert 'fp' in info['sources'] and all(shape[0] >= 2 for shape in info['sources'].values())


def test_crop_cache_built_from_another_detector_raises(synthetic_root, tmp_path, monkeypatch):
    root, _, _ = synthetic_root
    det, _ = _setup(root, tmp_path, monkeypatch)                         # raw-arm detector stub at <tmp>/raw133_A/model_best
    index_path = tmp_path / 'crops' / 'index.json'
    index = json.loads(index_path.read_text())
    # the cache was built with the ec10 detector's ROIs in the raw slot
    index['roi_files'] = {a: {s: f'results/det/rois_sel10g_clean_A_{s}.json' for s in ('train', 'val')} for a in index['arms']}
    index_path.write_text(json.dumps(index))
    with pytest.raises(AssertionError, match="not of the arm's detector raw133_A"):
        main_seg.main(_args(root, tmp_path, det, '--epochs', '1'))
    assert not (tmp_path / 'seg' / 'results_tseg.txt').exists()         # raised before any training
    # the rgb control reads the raw slot as well
    with pytest.raises(AssertionError, match="not of the arm's detector raw133_A"):
        main_seg.main(_args(root, tmp_path, det, '--epochs', '1', '--arm', 'rgb'))
