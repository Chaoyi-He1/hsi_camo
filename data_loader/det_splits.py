import os
import json
import random

SPLIT_PATH = os.path.join(os.path.dirname(__file__), 'splits', 'det_val_ids.json')


def _train_ids(data_path):
    hsi = os.path.join(data_path, 'train', 'hyperspectral')
    return sorted([os.path.splitext(f)[0] for f in os.listdir(hsi) if f.endswith('.mat')], key=int)


def make_det_splits(data_path, n_val=28, seed=0, path=None):
    '''Hold out n_val training ids for validation (fixed seed) and save them; idempotent once the file exists.'''
    path = path or SPLIT_PATH
    ids = _train_ids(data_path)
    if os.path.exists(path):
        return load_det_ids(data_path, path)
    val_ids = sorted(random.Random(seed).sample(ids, n_val), key=int)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump({'seed': seed, 'n_val': n_val, 'val_ids': val_ids}, f, indent=1)
    return [i for i in ids if i not in val_ids], val_ids


def load_det_ids(data_path, path=None):
    path = path or SPLIT_PATH
    assert os.path.exists(path), f"split file {path} missing; run make_det_splits"
    with open(path) as f:
        val_ids = json.load(f)['val_ids']
    ids = _train_ids(data_path)
    assert all(v in ids for v in val_ids), f"val ids not in {data_path}/train: {[v for v in val_ids if v not in ids]}"
    return [i for i in ids if i not in val_ids], val_ids
