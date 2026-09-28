import json
import torch
import main_det, main_det_rois
from tests.test_main_det import _args
from data_loader.cube_cache import build_cube_cache


def test_export_rois_json(synthetic_root, tmp_path, monkeypatch):
    root, _, _ = synthetic_root
    monkeypatch.delenv('RANK', raising=False)
    build_cube_cache(str(root), 'train', num_workers=0); build_cube_cache(str(root), 'test', num_workers=0)
    out_a = tmp_path / 'det_A'
    main_det.main(_args(root, tmp_path, **{'--session': 'A', '--name': 'tA', '--output-dir': str(out_a), '--top_k': '3'}))
    args = main_det_rois.get_args_parser().parse_args(['--data-path', str(root), '--cache-dir', str(root / 'cache_fp16'),
        '--split-file', str(tmp_path / 'val.json'), '--yolo-variant', 'yolo26n', '--pretrained', 'none', '--filter-select', 'uniform',
        '--num-filters', '6', '--device', 'cpu', '--amp', '--no-wandb', '--session', 'A', '--resume', str(out_a / 'model_best'),
        '--split', 'test', '--out-dir', str(tmp_path / 'rois'), '--conf-thres', '0.0', '--max-rois', '2', '--roi-min', '0', '--min-area', '10', '--num_workers', '0'])
    path = main_det_rois.main(args)
    data = json.load(open(path))
    assert set(data) == {'7'} and {'rois', 'boxes', 'gt_boxes'} <= set(data['7']) and len(data['7']['rois']) <= 2
    assert data['7']['gt_boxes'] == [[20.0, 10.0, 26.0, 16.0]]
    for r in data['7']['rois']:
        assert len(r) == 5 and 0 <= r[0] < r[2] <= 40 and 0 <= r[1] < r[3] <= 48
