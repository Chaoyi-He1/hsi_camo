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


def _crash_at(monkeypatch):
    '''Make train_one_epoch raise at epoch crash['at'] (an interrupted run); crash['at'] = None disarms it.'''
    crash = {'at': None}
    train = main_seg.train_one_epoch

    def crashing(model, front_end, loader, optimizer, device, epoch, **kw):
        if epoch == crash['at']:
            raise RuntimeError(f'simulated crash in epoch {epoch}')
        return train(model, front_end, loader, optimizer, device, epoch, **kw)
    monkeypatch.setattr(main_seg, 'train_one_epoch', crashing)
    return crash


def _lines(path):
    return [json.loads(l) for l in path.read_text().strip().splitlines()]


def test_main_seg_trains_resumes_and_evaluates(synthetic_root, tmp_path, monkeypatch):
    root, _, _ = synthetic_root
    det, built = _setup(root, tmp_path, monkeypatch)
    crash = _crash_at(monkeypatch)
    out = tmp_path / 'seg'
    crash['at'] = 2                                                                # a 3-epoch run interrupted in epoch 2
    with pytest.raises(RuntimeError, match='simulated crash'):
        main_seg.main(_args(root, tmp_path, det, '--epochs', '3'))
    crash['at'] = None
    lines = _lines(out / 'results_tseg.txt')
    assert [l['epoch'] for l in lines] == [0, 1] and not any(l.get('final') for l in lines)
    assert {'S', 'Fw', 'IoU', 'MAE', 'n'} <= set(lines[0]['val']) and lines[0]['val']['n'] == 4   # 2 oracle + 2 matched val ROIs
    assert {'loss', 'lr', 'main', 'side'} <= set(lines[0]['train'])
    ck = torch.load(out / 'model_last', map_location='cpu', weights_only=False)
    assert ck['epoch'] == 1 and ck['P'].shape == (3, 133) and ck['q'].shape == (3,) and ck['n_in'] == 133
    assert ck['args']['arm'] == 'raw' and ck['args']['seg_model'] == 'sam2unet' and (out / 'model_best').exists()
    np.testing.assert_array_equal(ck['P'], built[0][1])

    # resume: the last epoch from model_last with the stored stem fold (no refit) and the best score carried over
    def no_refit(*a, **k):
        raise AssertionError('a resumed run must reuse the stored (P, q)')
    monkeypatch.setattr(main_seg, 'fit_arm_rgb_map', no_refit)
    main_seg.main(_args(root, tmp_path, det, '--epochs', '3', '--resume', str(out / 'model_last')))
    lines = _lines(out / 'results_tseg.txt')
    assert [l['epoch'] for l in lines[:-1]] == [0, 1, 2] and lines[-1]['final'] is True
    assert lines[-2]['best'] >= lines[1]['best']
    np.testing.assert_array_equal(built[1][1], built[0][1])
    with pytest.raises(AssertionError, match='seg_model'):                         # another model's checkpoint
        main_seg.main(_args(root, tmp_path, det, '--seg_model', 'zoomnext', '--epochs', '3', '--resume', str(out / 'model_last')))

    main_seg.main(_args(root, tmp_path, det, '--resume', str(out / 'model_best'), '--eval'))
    last = json.loads((out / 'results_tseg.txt').read_text().strip().splitlines()[-1])
    assert last['eval'] is True and 'S' in last['val'] and last['val']['n'] == 4


def test_fresh_start_moves_a_previous_attempt_aside(synthetic_root, tmp_path, monkeypatch):
    root, _, _ = synthetic_root
    det, _ = _setup(root, tmp_path, monkeypatch)
    crash = _crash_at(monkeypatch)
    out, tb = tmp_path / 'seg', tmp_path / 'runs' / 'tseg'
    crash['at'] = 1                                                                # attempt 1 dies in epoch 1
    with pytest.raises(RuntimeError, match='simulated crash'):
        main_seg.main(_args(root, tmp_path, det, '--epochs', '2'))
    crash['at'] = None
    first = (out / 'results_tseg.txt').read_text()
    old_events = sorted(p.name for p in tb.iterdir())
    assert len(_lines(out / 'results_tseg.txt')) == 1 and old_events
    (tmp_path / 'runs' / 'tseg.prev').mkdir()                                      # an older .prev is replaced
    (tmp_path / 'runs' / 'tseg.prev' / 'older').write_text('x')
    (out / 'results_tseg.txt.prev').write_text('older\n')

    main_seg.main(_args(root, tmp_path, det, '--epochs', '1', '--dry_run'))        # a dry run moves nothing
    assert (out / 'results_tseg.txt').read_text() == first and (out / 'model_last').exists()

    main_seg.main(_args(root, tmp_path, det, '--epochs', '2'))                     # attempt 2, from epoch 0
    lines = _lines(out / 'results_tseg.txt')
    assert [l['epoch'] for l in lines[:-1]] == [0, 1] and lines[-1]['final'] is True   # no line of attempt 1
    assert (out / 'results_tseg.txt.prev').read_text() == first
    assert sorted(p.name for p in (tmp_path / 'runs' / 'tseg.prev').iterdir()) == old_events
    assert set(p.name for p in tb.iterdir()).isdisjoint(old_events)                # new TensorBoard events only
    prev_last = torch.load(out / 'model_last.prev', map_location='cpu', weights_only=False)
    assert prev_last['epoch'] == 0 and (out / 'model_best.prev').exists()          # attempt 1's checkpoints, aside

    n = len(_lines(out / 'results_tseg.txt'))                                      # --eval appends, moves nothing
    main_seg.main(_args(root, tmp_path, det, '--epochs', '2', '--resume', str(out / 'model_best'), '--eval'))
    assert len(_lines(out / 'results_tseg.txt')) == n + 1 and (out / 'model_last').exists()


def test_checkpoint_save_is_atomic(tmp_path, monkeypatch):
    model = TinySeg(4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: 1.0)
    args = main_seg.get_args_parser().parse_args([])
    P, q = np.ones((3, 4), np.float32), np.zeros(3, np.float32)
    path = str(tmp_path / 'model_last')
    main_seg.save_checkpoint(path, model, optimizer, None, scheduler, 3, args, 0.5, 2, P, q)
    assert sorted(p.name for p in tmp_path.iterdir()) == ['model_last']             # no .tmp left behind
    ck = torch.load(path, map_location='cpu', weights_only=False)
    assert (ck['epoch'], ck['best'], ck['best_epoch'], ck['n_in']) == (3, 0.5, 2, 4)

    # a save that dies mid-write leaves the previous checkpoint intact
    real_save = torch.save

    def dying_save(obj, f, *a, **k):
        with open(f, 'wb') as fh:
            fh.write(b'PK truncated')
        raise OSError('disk full')
    monkeypatch.setattr(torch, 'save', dying_save)
    with pytest.raises(OSError, match='disk full'):
        main_seg.save_checkpoint(path, model, optimizer, None, scheduler, 4, args, 0.6, 4, P, q)
    monkeypatch.setattr(torch, 'save', real_save)
    assert torch.load(path, map_location='cpu', weights_only=False)['epoch'] == 3


def test_resume_drops_later_lines_and_keeps_a_later_model_best(synthetic_root, tmp_path, monkeypatch):
    root, _, _ = synthetic_root
    det, _ = _setup(root, tmp_path, monkeypatch)
    out = tmp_path / 'seg'
    # keep a copy of model_last as saved after epoch 0, then let the run go on to the end
    save, kept = main_seg.save_checkpoint, {}

    def save_and_keep(path, *a):
        save(path, *a)
        if path.endswith('model_last') and a[4] == 0:                              # a[4] = epoch
            kept['e0'] = torch.load(path, map_location='cpu', weights_only=False)
    monkeypatch.setattr(main_seg, 'save_checkpoint', save_and_keep)
    main_seg.main(_args(root, tmp_path, det, '--epochs', '3'))
    main_seg.main(_args(root, tmp_path, det, '--epochs', '3', '--resume', str(out / 'model_best'), '--eval'))
    # the interrupted state: model_last of epoch 0, results lines of epochs 0-2 + final + an --eval line + a cut line, and
    # a model_best saved after epoch 0 whose score no later epoch can beat
    torch.save(kept['e0'], out / 'model_last')
    best = torch.load(out / 'model_best', map_location='cpu', weights_only=False)
    best.update(epoch=2, best=2.0, best_epoch=2)
    torch.save(best, out / 'model_best')
    with open(out / 'results_tseg.txt', 'a') as f:
        f.write('{"epoch": 3, "train": {"lo')
    before = (out / 'results_tseg.txt').read_text().splitlines()
    assert [json.loads(l).get('epoch') for l in before[:3]] == [0, 1, 2] and '"final": true' in before[3] and '"eval": true' in before[4]

    main_seg.main(_args(root, tmp_path, det, '--epochs', '3', '--resume', str(out / 'model_last')))
    after = (out / 'results_tseg.txt').read_text().splitlines()
    assert after[:2] == [before[0], before[4]]                                     # epoch 0 and the --eval line kept, in order
    lines = [json.loads(l) for l in after]
    assert [l.get('epoch') for l in lines[2:4]] == [1, 2] and not lines[2].get('final') and lines[-1]['final'] is True
    assert sum(1 for l in lines if l.get('final')) == 1 and len(lines) == 5        # epochs 1, 2 re-run once, one final line
    assert all((l['best'], l['best_epoch']) == (2.0, 2) for l in lines[2:4])       # seeded from the later model_best
    assert lines[-1]['best'] == 2.0 and lines[-1]['best_epoch'] == 2
    on_disk = torch.load(out / 'model_best', map_location='cpu', weights_only=False)
    assert (on_disk['epoch'], on_disk['best']) == (2, 2.0)                         # never overwritten by a worse epoch
    assert not list(out.glob('*.tmp'))


def test_resume_must_continue_the_same_run(synthetic_root, tmp_path, monkeypatch):
    root, _, _ = synthetic_root
    det, _ = _setup(root, tmp_path, monkeypatch)
    out = tmp_path / 'seg'
    main_seg.main(_args(root, tmp_path, det, '--epochs', '1'))
    last = str(out / 'model_last')
    for flags, what in ((('--seed', '1'), 'seed'), (('--canvas', '64'), 'canvas'), (('--epochs', '5'), 'epochs'),
                        (('--arm', 'rgb'), 'arm')):
        with pytest.raises(AssertionError, match=what):
            main_seg.main(_args(root, tmp_path, det, '--epochs', '1', '--resume', last, *flags))
    # another detector's run (its front end and ROIs): the checkpoint names the detector folder
    ck = torch.load(last, map_location='cpu', weights_only=False)
    ck['args']['det_ckpt'] = '/elsewhere/sel10g_clean_A/model_best'
    torch.save(ck, tmp_path / 'other_last')
    with pytest.raises(AssertionError, match='detector sel10g_clean_A'):
        main_seg.main(_args(root, tmp_path, det, '--epochs', '1', '--resume', str(tmp_path / 'other_last')))
    # --epochs only matters for training: evaluating the checkpoint with another --epochs is fine
    main_seg.main(_args(root, tmp_path, det, '--epochs', '5', '--resume', last, '--eval'))
    assert _lines(out / 'results_tseg.txt')[-1]['eval'] is True


def test_loaders_keep_their_workers_between_epochs(synthetic_root, tmp_path, monkeypatch):
    root, _, _ = synthetic_root
    det, _ = _setup(root, tmp_path, monkeypatch)
    made, real = [], torch.utils.data.DataLoader

    def spy(*a, **k):
        made.append(real(*a, **k))
        return made[-1]
    monkeypatch.setattr(torch.utils.data, 'DataLoader', spy)
    main_seg.main(_args(root, tmp_path, det, '--dry_run', '--num_workers', '2'))
    train, val = [d for d in made if d.batch_size == 4][:2]                       # the stem fit's loader has batch 1
    assert train.drop_last and not val.drop_last and train.persistent_workers and val.persistent_workers
    made.clear()
    main_seg.main(_args(root, tmp_path, det, '--dry_run'))                          # --num_workers 0: no workers to keep
    assert not any(d.persistent_workers for d in made)


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
