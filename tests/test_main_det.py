import json
import os
import types
import torch

import main_det
from models.ec_yolo import ECYolo
from util.distributed_util import Custom_DistributedSampler
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


def test_train_sampler_is_one_pass_per_epoch():
    ds = list(range(250))
    assert isinstance(main_det.build_train_sampler(ds, distributed=False), torch.utils.data.RandomSampler)
    s = main_det.build_train_sampler(ds, distributed=True, num_replicas=2, rank=0)
    idx = list(iter(s))
    assert len(s) == 125 and len(idx) == 125 and len(set(idx)) == 125        # one pass over this rank's half
    assert len(main_det.build_train_sampler(list(range(251)), True, num_replicas=2, rank=1)) == 126   # ceil(N/world)
    assert len(Custom_DistributedSampler(ds, num_replicas=2, rank=0)) == 125 * 20   # the default this guards against


def test_sessions_a_then_b_and_eval(synthetic_root, tmp_path, monkeypatch):
    root, _, _ = synthetic_root
    build_cube_cache(str(root), 'train', num_workers=0); build_cube_cache(str(root), 'test', num_workers=0)
    monkeypatch.setenv('RANK', '')                                    # keep init_distributed_mode in single-process mode
    monkeypatch.delenv('RANK', raising=False)
    out_a, out_b = tmp_path / 'det_A', tmp_path / 'det_B'
    main_det.main(_args(root, tmp_path, **{'--session': 'A', '--name': 'tA', '--output-dir': str(out_a), '--top_k': '3'}))
    assert (out_a / 'model_best').exists() and (out_a / 'model_0').exists()
    assert (out_a / 'gate_ranking.csv').read_text().splitlines()[0] == 'rank,index,voltage,weight'
    with open(out_a / 'top_k.json') as f:
        top = json.load(f)
    assert len(top['indices']) == 3 and len(top['voltages']) == 3

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


def test_same_session_resume_keeps_best_and_loss_schedule(synthetic_root, tmp_path, monkeypatch):
    root, _, _ = synthetic_root
    build_cube_cache(str(root), 'train', num_workers=0); build_cube_cache(str(root), 'test', num_workers=0)
    monkeypatch.delenv('RANK', raising=False)
    out = tmp_path / 'det_R'
    main_det.main(_args(root, tmp_path, **{'--session': 'A', '--name': 'r0', '--output-dir': str(out), '--top_k': '3'}))
    ck = torch.load(out / 'model_0', map_location='cpu', weights_only=False)
    assert 'best' in ck and len(ck['best']) == 2
    ck['best'] = [1.0, 1.0]                                            # unbeatable: nothing after the resume may top it
    torch.save(ck, out / 'model_0')
    best_bytes = (out / 'model_best').read_bytes()

    calls = []
    real_end_epoch = ECYolo.end_epoch
    monkeypatch.setattr(ECYolo, 'end_epoch', lambda self: (calls.append(1), real_end_epoch(self))[1])
    main_det.main(_args(root, tmp_path, **{'--session': 'A', '--name': 'r1', '--output-dir': str(out), '--top_k': '3',
                                           '--resume': str(out / 'model_0'), '--epochs': '2'}))
    rec = json.loads((out / 'results_r1.txt').read_text().strip().splitlines()[0])
    assert rec['epoch'] == 1 and rec['best'] == [1.0, 1.0]             # carried over, not re-initialised to (-1, -1)
    assert (out / 'model_best').read_bytes() == best_bytes             # the pre-resume best was not clobbered
    assert len(calls) == 2                                             # 1 replayed epoch + 1 trained epoch


def test_build_optimizer_gives_the_gate_its_own_lr():
    class Fake(torch.nn.Module):
        def __init__(self, gate):
            super().__init__()
            self.filter_bank = torch.nn.Module()
            if gate:
                self.filter_bank.theta = torch.nn.Parameter(torch.zeros(6))
            self.yolo = torch.nn.Module(); self.yolo.model = torch.nn.ModuleList([torch.nn.Conv2d(6, 4, 3)])
            self.head = torch.nn.Linear(4, 2)
    args = types.SimpleNamespace(lr=1e-4, gate_lr=1e-2, weight_decay=5e-4)
    opt = main_det.build_optimizer(Fake(gate=True), args)
    assert len(opt.param_groups) == 3
    assert opt.param_groups[0]['weight_decay'] == 5e-4 and opt.param_groups[1]['weight_decay'] == 0.0
    g = opt.param_groups[2]
    assert g['lr'] == 1e-2 and g['weight_decay'] == 0.0 and len(g['params']) == 1 and tuple(g['params'][0].shape) == (6,)
    assert sum(len(g['params']) for g in opt.param_groups) == 5          # theta, conv w/b, linear w/b: each exactly once
    assert len(main_det.build_optimizer(Fake(gate=False), args).param_groups) == 2   # session B: no gate group
