import os
import types
import numpy as np
import pytest

from util.logger import TrainLogger


def _args(tmp_path, **kw):
    a = types.SimpleNamespace(name='t', wandb=False, wandb_entity='chaoyi-hsi', wandb_project='hsi_camo',
                              runs_dir=str(tmp_path / 'runs'), wandb_dir=str(tmp_path / 'wandb'), rank=0, distributed=False)
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def test_tensorboard_only_logs_scalars_and_images(tmp_path):
    lg = TrainLogger(_args(tmp_path), cfg={'lr': 1e-4})
    assert lg.mode == 'disabled' and lg.wandb_run is None
    lg.scalar('train/loss', 1.0, 0); lg.scalars({'a': 1.0, 'b': 2.0}, 0, prefix='val/')
    lg.histogram('w', np.arange(10.0), 0); lg.image('img', np.zeros((8, 8, 3), np.uint8), 0)
    lg.table('rank', ['voltage', 'w'], [[1.0, 0.5]], 0)               # no-op without wandb
    lg.finish()
    assert any(f.startswith('events.out.tfevents') for f in os.listdir(tmp_path / 'runs' / 't'))


def test_wandb_failure_falls_back_to_offline(tmp_path, monkeypatch):
    import wandb
    calls = []
    def fake_init(**kw):
        calls.append(kw.get('mode'))
        if kw.get('mode') != 'offline':
            raise wandb.errors.CommError('entity not found')
        return types.SimpleNamespace(log=lambda *a, **k: None, finish=lambda: None, url='offline')
    monkeypatch.setattr(wandb, 'init', fake_init)
    lg = TrainLogger(_args(tmp_path, wandb=True), cfg={})
    assert calls == [None, 'offline'] and lg.mode == 'offline' and lg.wandb_run is not None
    lg.scalar('x', 1.0, 0); lg.finish()


def test_non_main_process_is_noop(tmp_path):
    lg = TrainLogger(_args(tmp_path, rank=1, distributed=True), cfg={})
    lg.scalar('x', 1.0, 0); lg.finish()
    assert not (tmp_path / 'runs').exists()
