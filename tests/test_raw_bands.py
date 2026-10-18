import json
import torch

import main_det
from data_loader.cube_cache import build_cube_cache
from tests.test_main_det import _args


def test_raw_bands_control_session(synthetic_root, tmp_path, monkeypatch):
    root, _, _ = synthetic_root
    build_cube_cache(str(root), 'train', num_workers=0); build_cube_cache(str(root), 'test', num_workers=0)
    monkeypatch.delenv('RANK', raising=False)
    out = tmp_path / 'raw_A'
    main_det.main(_args(root, tmp_path, **{'--session': 'A', '--raw-bands': None, '--name': 'raw', '--output-dir': str(out), '--top_k': '3'}))
    ck = torch.load(out / 'model_best', map_location='cpu', weights_only=False)
    assert ck['model']['yolo.model.0.conv.weight'].shape[1] == 133                      # the 133 raw bands go straight in
    assert 'filter_bank.theta' not in ck['model'] and ck['args']['raw_bands'] is True   # no gate in the control
    assert torch.equal(ck['model']['filter_bank.R_t'], torch.eye(133))                  # identity projection
    assert not (out / 'gate_ranking.csv').exists() and not (out / 'top_k.json').exists()  # nothing to rank
    rec = json.loads((out / 'results_raw.txt').read_text().strip().splitlines()[-1])
    assert 'val' in rec and rec['val']['n_images'] >= 1
    main_det.main(_args(root, tmp_path, **{'--session': 'A', '--raw-bands': None, '--name': 'raw_eval', '--output-dir': str(out),
                                           '--resume': str(out / 'model_best'), '--eval': None}))
    assert (out / 'results_raw_eval.txt').exists()


def test_raw_bands_with_pca_channels_trains_on_the_whitened_bands(synthetic_root, tmp_path, monkeypatch):
    '''raw-PCA-K control: the K whitened band directions go into the first conv, and --eval rebuilds the same front end.'''
    root, _, _ = synthetic_root
    build_cube_cache(str(root), 'train', num_workers=0); build_cube_cache(str(root), 'test', num_workers=0)
    monkeypatch.delenv('RANK', raising=False)
    out = tmp_path / 'rawpca3_A'
    main_det.main(_args(root, tmp_path, **{'--session': 'A', '--raw-bands': None, '--pca-channels': '3', '--name': 'rawpca3',
                                           '--output-dir': str(out), '--top_k': '3'}))
    ck = torch.load(out / 'model_best', map_location='cpu', weights_only=False)
    assert ck['model']['yolo.model.0.conv.weight'].shape[1] == 3                         # 3 whitened band directions
    assert ck['model']['filter_bank.R_t'].shape == (3, 133) and 'filter_bank.theta' not in ck['model']
    assert ck['args']['raw_bands'] is True and ck['args']['pca_channels'] == 3
    main_det.main(_args(root, tmp_path, **{'--session': 'A', '--raw-bands': None, '--pca-channels': '3', '--name': 'rawpca3_eval',
                                           '--output-dir': str(out), '--resume': str(out / 'model_best'), '--eval': None}))
    assert (out / 'results_rawpca3_eval.txt').exists()
