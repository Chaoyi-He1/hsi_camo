import json
import os
import sys
import types
import torch
import pytest

import main_det
from data_loader.cube_cache import build_cube_cache, default_cache_dir


def _args(root, tmp_path, **kw):
    argv = ['--data-path', str(root), '--cache-dir', default_cache_dir(str(root)), '--split-file', str(tmp_path / 'val.json'),
            '--yolo-variant', 'yolo26n', '--pretrained', 'none', '--filter-select', 'uniform', '--num-filters', '6',
            '--epochs', '1', '--batch_size', '2', '--num_workers', '0', '--device', 'cpu', '--amp', '--no-wandb',
            '--runs-dir', str(tmp_path / 'runs'), '--roi-min', '0', '--min-area', '10', '--save_every', '1']
    for k, v in kw.items():
        argv += [k] + ([] if v is None else [str(v)])
    parser = main_det.get_args_parser()
    return parser.parse_args(argv)


def test_sessions_a_then_b_and_eval(synthetic_root, tmp_path, monkeypatch):
    root, _, _ = synthetic_root
    build_cube_cache(str(root), 'train', num_workers=0); build_cube_cache(str(root), 'test', num_workers=0)
    monkeypatch.setenv('RANK', '')                                    # keep init_distributed_mode in single-process mode
    monkeypatch.delenv('RANK', raising=False)
    out_a, out_b = tmp_path / 'det_A', tmp_path / 'det_B'
    main_det.main(_args(root, tmp_path, **{'--session': 'A', '--name': 'tA', '--output-dir': str(out_a), '--top_k': '3'}))
    assert (out_a / 'model_best').exists() and (out_a / 'model_0').exists()
    assert (out_a / 'gate_ranking.csv').read_text().splitlines()[0] == 'rank,index,voltage,weight'
    top = json.load(open(out_a / 'top_k.json')); assert len(top['indices']) == 3 and len(top['voltages']) == 3

    lines = (out_a / 'results_tA.txt').read_text().strip().splitlines()
    rec = json.loads(lines[-1]); assert rec['epoch'] == 0 and 'val' in rec and 'coverage_recall99_raw' in rec['val']
    ck = torch.load(out_a / 'model_best', map_location='cpu', weights_only=False)
    assert set(ck) >= {'model', 'optimizer', 'scaler', 'lr_scheduler', 'epoch', 'args', 'selected_indices', 'selected_voltages'}
    assert len(ck['selected_indices']) == 6
    main_det.main(_args(root, tmp_path, **{'--session': 'B', '--name': 'tB', '--output-dir': str(out_b), '--top_k': '3',
                                           '--ranking': str(out_a / 'gate_ranking.csv'), '--resume': str(out_a / 'model_best')}))
    ckb = torch.load(out_b / 'model_best', map_location='cpu', weights_only=False)
    assert len(ckb['selected_indices']) == 3 and ckb['model']['yolo.model.0.conv.weight'].shape[1] == 3
    main_det.main(_args(root, tmp_path, **{'--session': 'B', '--name': 'tB_eval', '--output-dir': str(out_b), '--top_k': '3',
                                           '--resume': str(out_b / 'model_best'), '--eval': None}))
    assert (out_b / 'results_tB_eval.txt').exists()
