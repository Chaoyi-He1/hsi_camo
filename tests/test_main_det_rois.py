import os
import json
import argparse
import pytest
import yaml
import torch
import main_det, main_det_rois
from tests.test_main_det import _args
from data_loader.cube_cache import build_cube_cache, default_cache_dir
from data_loader.det_splits import load_det_ids


def _roi_args(root, tmp_path, resume, **extra):
    argv = ['--data-path', str(root), '--split-file', str(tmp_path / 'val.json'), '--yolo-variant', 'yolo26n',
            '--pretrained', 'none', '--filter-select', 'uniform', '--num-filters', '6', '--device', 'cpu', '--amp',
            '--no-wandb', '--session', 'A', '--resume', str(resume), '--split', 'test', '--out-dir', str(tmp_path / 'rois'),
            '--roi-min', '0', '--min-area', '10', '--num_workers', '0']
    for k, v in extra.items():
        argv += [k] + ([] if v is None else [str(v)])
    return main_det_rois.get_args_parser().parse_args(argv)


def test_export_operating_point_comes_from_cfg(synthetic_root, tmp_path):
    root, _, _ = synthetic_root
    args = _roi_args(root, tmp_path, resume='x')
    main_det.load_cfg(args)
    with open(args.hpy) as f:
        cfg = yaml.safe_load(f)
    assert args.max_rois is None and (args.roi_conf, args.roi_topk) == (cfg['roi_conf'], cfg['roi_topk'])
    assert (args.roi_conf, args.roi_topk) == (0.02, 5)      # the operating point evaluate()'s *_op keys use (calibrated on det_A)


def test_dataset_kwargs_defaults_to_the_training_cache(synthetic_root, tmp_path):
    root, _, _ = synthetic_root
    args = _args(root, tmp_path)
    args.cache_dir = ''
    assert main_det.dataset_kwargs(args)['cache_dir'] == default_cache_dir(str(root))     # never the raw .mat path
    args.cache_dir = str(tmp_path / 'elsewhere')
    assert main_det.dataset_kwargs(args)['cache_dir'] == str(tmp_path / 'elsewhere')


def test_export_requires_a_checkpoint(synthetic_root, tmp_path, monkeypatch):
    root, _, _ = synthetic_root
    monkeypatch.delenv('RANK', raising=False)
    with pytest.raises(AssertionError, match='--resume'):
        main_det_rois.main(_roi_args(root, tmp_path, resume=''))


def test_export_rois_json(synthetic_root, tmp_path, monkeypatch):
    root, _, _ = synthetic_root
    monkeypatch.delenv('RANK', raising=False)
    build_cube_cache(str(root), 'train', num_workers=0); build_cube_cache(str(root), 'test', num_workers=0)
    out_a = tmp_path / 'det_A'
    main_det.main(_args(root, tmp_path, **{'--session': 'A', '--name': 'tA', '--output-dir': str(out_a), '--top_k': '3'}))

    seen, real = [], main_det_rois.HyperCOD_data
    monkeypatch.setattr(main_det_rois, 'HyperCOD_data', lambda **kw: (seen.append(kw), real(**kw))[1])
    def _no_build(*a, **k):
        raise AssertionError("the ROI export must not build the train/val/test datasets")
    monkeypatch.setattr(main_det, 'build_datasets', _no_build)
    # no --cache-dir: it must still come out as the training cache. --max-rois overrides cfg roi_topk.
    args = _roi_args(root, tmp_path, resume=out_a / 'model_best', **{'--roi-conf': '0.0', '--max-rois': '2'})
    path = main_det_rois.main(args)
    assert args.roi_topk == 2
    assert len(seen) == 1 and seen[0]['cache_dir'] == default_cache_dir(str(root)) and seen[0]['split'] == 'test'
    assert os.path.basename(path) == 'rois_det_A_test.json'                    # rois_<run>_<split>, run = the checkpoint's folder

    with open(path) as f:
        data = json.load(f)
    assert set(data) == {'7'} and {'rois', 'boxes', 'gt_boxes'} <= set(data['7']) and len(data['7']['rois']) <= 2
    assert data['7']['gt_boxes'] == [[20.0, 10.0, 26.0, 16.0]]
    for r in data['7']['rois']:
        assert len(r) == 5 and 0 <= r[0] < r[2] <= 40 and 0 <= r[1] < r[3] <= 48


def test_detector_args_take_the_model_from_the_checkpoint(tmp_path):
    # the CLI asks for other model flags than the run used, and for its own data / device / operating point
    cli = main_det_rois.get_args_parser().parse_args(
        ['--data-path', str(tmp_path / 'data'), '--device', 'cpu', '--num_workers', '0', '--roi-conf', '0.3', '--pca-channels', '5',
         '--read_noise_db', '40', '--filter-select', 'all', '--split', 'val', '--pretrained', 'none'])
    ckpt_args = {'session': 'A', 'filter_select': 'manual', 'filter_voltages': [1.75, -0.44, 1.36], 'pca_channels': 3, 'seed': 42,
                 'yolo_variant': 'yolo26s', 'epochs': 100, 'hpy': 'cfg/det.yaml', 'gate_entropy_weight': 0.05, 'contain_weight': 1.0,
                 'data_path': '/old/data', 'cache_dir': '/old/cache', 'device': 'cuda', 'num_workers': 6, 'roi_conf': 0.02, 'roi_topk': 5}
    a = main_det_rois.detector_args(ckpt_args, cli)
    # the model is the checkpoint's ...
    assert (a.filter_select, a.filter_voltages, a.pca_channels, a.seed, a.yolo_variant) == ('manual', [1.75, -0.44, 1.36], 3, 42, 'yolo26s')
    # ... a flag an older checkpoint lacks (raw133_A has no read_noise_*) takes main_det's default, never the CLI value
    assert a.read_noise_db == 0.0 and a.read_noise_db_range is None and a.raw_bands is False and a.no_gate is False
    # ... while data, device and the operating point stay the CLI's (a moved cache or a CPU run is never overridden)
    assert (a.data_path, a.cache_dir, a.device, a.num_workers, a.roi_conf) == (str(tmp_path / 'data'), '', 'cpu', 0, 0.3)
    assert not hasattr(a, 'split') and a.roi_topk is None                     # export-only flags dropped; cfg fills the rest later
    # a Stage-2 namespace (main_seg's own flags, its own cfg) still yields the detector's cfg and parser defaults
    seg = argparse.Namespace(data_path=str(tmp_path / 'data'), device='cpu', hpy='cfg/seg.yaml', lr=1e-3, arm='ec10')
    b = main_det_rois.detector_args(ckpt_args, seg)
    assert b.hpy == 'cfg/det.yaml' and b.lr == 1e-3 and not hasattr(b, 'arm') and b.conf_thres is None and b.cache_dir == ''


def test_export_rebuilds_from_the_checkpoint_for_every_split(synthetic_root, tmp_path, monkeypatch):
    root, _, _ = synthetic_root
    monkeypatch.delenv('RANK', raising=False)
    build_cube_cache(str(root), 'train', num_workers=0); build_cube_cache(str(root), 'test', num_workers=0)
    run_dir = tmp_path / 'pca_run'
    # a whitened detector (6 uniform voltages -> 3 channels, yolo26n): nothing of it is repeated on the export CLI below
    main_det.main(_args(root, tmp_path, **{'--session': 'A', '--name': 'pca_run', '--output-dir': str(run_dir), '--pca-channels': '3'}))
    train_ids, val_ids = load_det_ids(str(root), str(tmp_path / 'val.json'))
    argv = ['--data-path', str(root), '--split-file', str(tmp_path / 'val.json'), '--device', 'cpu', '--amp', '--num_workers', '0',
            '--resume', str(run_dir / 'model_best'), '--out-dir', str(tmp_path / 'rois'), '--roi-min', '0', '--min-area', '10',
            '--roi-conf', '0.0', '--pretrained', 'none']
    frames = {}
    for split in ['train', 'val', 'test']:
        args = main_det_rois.get_args_parser().parse_args(argv + ['--split', split])
        path = main_det_rois.main(args)
        assert path == str(tmp_path / 'rois' / f'rois_pca_run_{split}.json')
        with open(path) as f:
            data = json.load(f)
        frames[split] = set(data)
        for v in data.values():
            assert v['gt_boxes'] == [[20.0, 10.0, 26.0, 16.0]] and len(v['rois']) <= 5 and all(len(r) == 5 for r in v['rois'])
    # train = the detector's training ids only, val = the held-out ids of the split file, test = the test split
    assert frames == {'train': set(train_ids), 'val': set(val_ids), 'test': {'7'}} and not set(train_ids) & set(val_ids)
    model, det_args, run = main_det_rois.load_detector(str(run_dir / 'model_best'), args, torch.device('cpu'))
    assert run == 'pca_run' and (det_args.pca_channels, det_args.filter_select, det_args.num_filters, det_args.yolo_variant) == (3, 'uniform', 6, 'yolo26n')
    assert model.filter_bank.n_channels == 3 and not model.training and det_args.pretrained == 'none'
    ck = torch.load(run_dir / 'model_best', map_location='cpu', weights_only=False)
    assert all(torch.equal(v, ck['model'][k]) for k, v in model.state_dict().items())   # the checkpoint, loaded strictly
