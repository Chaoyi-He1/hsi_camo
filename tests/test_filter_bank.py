import os
import types
import numpy as np
import torch
import pytest

import models.ec_yolo as ec_yolo
from models.filter_bank import FilterBank, build_filter_bank, pca_whitening, pca_whitened_channels, read_noise_std
from data_loader.my_dataset import HyperCOD_data

# GPU tests are opt-in: this box shares one GPU with the live detector runs, so a test that allocates on CUDA
# OOMs against them and fails the suite. Set HSI_CAMO_GPU_TESTS=1 to run them on an idle GPU.
GPU_TESTS = os.environ.get('HSI_CAMO_GPU_TESTS') == '1'
requires_gpu = pytest.mark.skipif(not (GPU_TESTS and torch.cuda.is_available()),
                                  reason="GPU test: set HSI_CAMO_GPU_TESTS=1 with a free CUDA device")


def make(root, **kw):
    kw.setdefault('split', 'test'); kw.setdefault('crop_size', 0); kw.setdefault('filter_norm', 'l1')
    kw.setdefault('num_filters', 12); kw.setdefault('seed', 0)
    return HyperCOD_data(data_path=str(root), **kw)


def test_filter_bank_matches_dataloader_channels(synthetic_root):
    root, _, _ = synthetic_root
    ds_in = make(root, use_filter=False, norm='p99', out_dtype='float16')       # what the detector loader returns
    ref = make(root, use_filter=True, norm='p99z')                              # loader-side channels, standardised
    fb = FilterBank(*ds_in.filter_bank_tensors()[:3], weight_vector=False)
    x = torch.from_numpy(ds_in[0][0])[None]                                     # [1, 133, H, W] fp16
    y = fb(x.float())[0].numpy()
    np.testing.assert_allclose(y, ref[0][0], rtol=2e-2, atol=2e-2)              # fp16 input vs float32 path
    assert fb.n_channels == 12 and torch.equal(fb.weights, torch.ones(12)) and float(fb.entropy()) == 0.0


def test_weight_vector_softmax_scaling_and_ranking():
    R = np.random.RandomState(0).randn(133, 5).astype(np.float32)
    fb = FilterBank(R, np.zeros(5, np.float32), np.ones(5, np.float32), weight_vector=True,
                    init_logits=np.array([0.0, 2.0, -1.0, 0.5, 0.0], np.float32))
    w = fb.weights
    assert torch.isclose(w.sum(), torch.tensor(5.0)) and (w > 0).all()
    assert fb.ranking().tolist()[0] == 1 and fb.ranking().tolist()[-1] == 2
    assert 0.0 < fb.entropy().item() < 1.0
    assert FilterBank(R, np.zeros(5), np.ones(5), init_logits=np.zeros(5)).entropy().item() == pytest.approx(1.0)
    x = torch.randn(2, 133, 4, 6)
    y = fb(x)
    assert y.shape == (2, 5, 4, 6)
    expected = torch.einsum('nc,bchw->bnhw', torch.from_numpy(R).T, x) * w.view(1, 5, 1, 1)
    torch.testing.assert_close(y, expected, rtol=1e-5, atol=1e-5)
    assert fb.theta.requires_grad and list(fb.parameters()) == [fb.theta]


def _autocast_roundtrip(device, dtype):
    '''FilterBank under autocast: low-precision in, the same low precision out, no inf/nan from the einsum.'''
    fb = FilterBank(np.random.rand(133, 8).astype(np.float32) / 133, np.zeros(8, np.float32), np.ones(8, np.float32)).to(device)
    x = torch.rand(1, 133, 64, 64, device=device, dtype=dtype)
    with torch.autocast(device, dtype=dtype):
        y = fb(x)                                                               # [1, 8, 64, 64]
    assert y.dtype == dtype and torch.isfinite(y).all()


def test_filter_bank_is_low_precision_safe_under_autocast_on_cpu():
    _autocast_roundtrip('cpu', torch.bfloat16)


@requires_gpu
def test_filter_bank_is_fp16_safe_under_autocast():
    _autocast_roundtrip('cuda', torch.float16)


def _fb_ds(root):
    return HyperCOD_data(str(root), split='train', use_filter=False, norm='p99', crop_size=0, num_filters=8, filter_norm='l1', out_dtype='float16')


def _stage1_filter_bank(args, dataset):
    '''
    The FilterBank block of models/ec_yolo.build_ec_yolo at 1a14827, verbatim (minus the print): build_filter_bank must
    reproduce it bit for bit, since Stage 2 relies on seeing exactly the channels its detector was trained on.
    '''
    noise_db = float(getattr(args, 'read_noise_db', 0.0) or 0.0)
    noise_model = getattr(args, 'read_noise_model', None) or 'floor'
    mu, cov = dataset._band_stats()
    noise_std = proj = None
    if getattr(args, 'raw_bands', False):
        R = np.eye(dataset.n_bands, dtype=np.float32)
        mean, std = mu.astype(np.float32), np.sqrt(np.maximum(np.diag(cov), 0.0)).astype(np.float32)
        volts, weight_vector = dataset.wavelens.copy(), False
        if noise_db > 0:
            noise_std = read_noise_std(R, mu, cov, noise_db, noise_model)
    else:
        R, mean, std, volts = dataset.filter_bank_tensors()
        weight_vector = (args.session == 'A') and not getattr(args, 'no_gate', False)
        if noise_db > 0:
            R_all, _ = dataset.candidate_filter_matrix()
            noise_std = read_noise_std(R, mu, cov, noise_db, noise_model, R_ref=R_all)
        k = int(getattr(args, 'pca_channels', 0) or 0)
        if k > 0:
            if noise_std is None:
                R, mean, std, volts = pca_whitened_channels(R, mean, std, cov, k)
            else:
                V, lam = pca_whitening(R, mean, std, cov, k, noise_var=(noise_std / std) ** 2)
                proj, volts = (V / np.sqrt(lam)).astype(np.float32), np.arange(1, k + 1, dtype=np.float64)
            weight_vector = False
    scale_range = None
    db_range = getattr(args, 'read_noise_db_range', None)
    if noise_std is not None and db_range:
        lo_db, hi_db = sorted(float(v) for v in db_range)
        scale_range = (10 ** ((noise_db - hi_db) / 20), 10 ** ((noise_db - lo_db) / 20))
    fb = FilterBank(R, mean, std, weight_vector=weight_vector, noise_std=noise_std, proj=proj, eval_seed=int(getattr(args, 'seed', 0) or 0),
                    train_scale_range=scale_range)
    return fb, volts


# every branch of the construction: raw bands (+ noise), gated / ungated responses, session B (no gate), folded PCA,
# noise-regularised PCA with SNR augmentation, relative read noise on the gated bank
FB_CASES = {
    'raw': dict(raw_bands=True),
    'raw_noise': dict(raw_bands=True, read_noise_db=40.0),
    'gate': dict(),
    'no_gate': dict(no_gate=True),
    'session_b': dict(session='B'),
    'pca_folded': dict(pca_channels=3),
    'pca_noise_range': dict(pca_channels=3, read_noise_db=40.0, read_noise_model='floor', read_noise_db_range=[34.0, 50.0], seed=7),
    'gate_relative_noise': dict(read_noise_db=30.0, read_noise_model='relative', seed=3),
}


@pytest.mark.parametrize('case', sorted(FB_CASES))
def test_build_filter_bank_reproduces_the_stage1_construction(synthetic_root, case):
    root, _, _ = synthetic_root
    ds = _fb_ds(root)
    # a sparse namespace, as the Stage-1 tests and older checkpoints' args have: missing flags keep their getattr defaults
    args = types.SimpleNamespace(**{'session': 'A', **FB_CASES[case]})
    fb, volts = build_filter_bank(args, ds)
    ref, ref_volts = _stage1_filter_bank(args, ds)
    sd, sd_ref = fb.state_dict(), ref.state_dict()
    assert sd.keys() == sd_ref.keys() and all(torch.equal(sd[k], sd_ref[k]) for k in sd)    # same buffers, bit for bit
    assert (fb.n_readings, fb.n_channels, fb.weight_vector, fb.eval_seed, fb.train_scale_range) == \
           (ref.n_readings, ref.n_channels, ref.weight_vector, ref.eval_seed, ref.train_scale_range)
    np.testing.assert_array_equal(np.asarray(volts), np.asarray(ref_volts))
    x = torch.from_numpy(ds[0][0])[None].float()                                           # [1, 133, H, W]
    fb.eval(); ref.eval()                                                                  # eval: the reseeded noise sequence
    assert torch.equal(fb(x), ref(x))


def test_ec_yolo_reexports_and_builds_through_build_filter_bank(synthetic_root, monkeypatch):
    # the moved helpers stay importable from models.ec_yolo (tests, main_select_voltages, docs/reports scripts)
    assert ec_yolo.pca_whitening is pca_whitening and ec_yolo.pca_whitened_channels is pca_whitened_channels
    assert ec_yolo.read_noise_std is read_noise_std and ec_yolo.build_filter_bank is build_filter_bank and ec_yolo.FilterBank is FilterBank
    root, _, _ = synthetic_root
    ds = _fb_ds(root)
    args = types.SimpleNamespace(yolo_variant='yolo26n', pretrained='none', gate_entropy_weight=0.05, contain_weight=1.0, epochs=1,
                                 session='A', pca_channels=3)
    calls, real = [], ec_yolo.build_filter_bank
    monkeypatch.setattr(ec_yolo, 'build_filter_bank', lambda a, d: (calls.append(a), real(a, d))[1])
    m = ec_yolo.build_ec_yolo(args, ds)
    fb, volts = real(args, ds)
    assert len(calls) == 1 and calls[0] is args                                            # one front end, built by the shared builder
    assert m.filter_bank.n_channels == 3 and m.yolo.model[0].conv.weight.shape[1] == 3
    assert all(torch.equal(v, fb.state_dict()[k]) for k, v in m.filter_bank.state_dict().items())
    np.testing.assert_array_equal(m.selected_voltages, np.asarray(volts))
