import json
import pytest
import yaml
import main_det, main_det_rois
from tests.test_main_det import _args
from data_loader.cube_cache import build_cube_cache, default_cache_dir


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

    with open(path) as f:
        data = json.load(f)
    assert set(data) == {'7'} and {'rois', 'boxes', 'gt_boxes'} <= set(data['7']) and len(data['7']['rois']) <= 2
    assert data['7']['gt_boxes'] == [[20.0, 10.0, 26.0, 16.0]]
    for r in data['7']['rois']:
        assert len(r) == 5 and 0 <= r[0] < r[2] <= 40 and 0 <= r[1] < r[3] <= 48
