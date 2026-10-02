import types
import numpy as np
import torch
import pytest

from models.filter_bank import FilterBank
from models.ec_yolo import pca_whitening, pca_whitened_channels, read_noise_std, build_ec_yolo
from data_loader.my_dataset import HyperCOD_data


def _ds(root, **kw):
    kw.setdefault('num_filters', 8)
    return HyperCOD_data(str(root), split='train', use_filter=False, norm='p99', crop_size=0, filter_norm='l1', out_dtype='float16', **kw)


def _bank(n=5, seed=0):
    R = np.random.RandomState(seed).rand(133, n).astype(np.float32) / 133                 # [133, n] positive, L1 ~ 0.5
    return R, (R.T @ np.full(133, 0.5, np.float32)).astype(np.float32), np.full(n, 0.3, np.float32)


def test_filter_bank_without_noise_or_projection_is_unchanged():
    R, mean, std = _bank()
    fb = FilterBank(R, mean, std, weight_vector=False)
    assert fb.noise_std is None and fb.proj is None and fb.n_channels == fb.n_readings == 5
    assert 'noise_std' not in fb.state_dict() and 'proj' not in fb.state_dict()          # checkpoints trained before keep loading
    x = torch.rand(2, 133, 4, 6)
    expected = (torch.einsum('nc,bchw->bnhw', torch.from_numpy(R).T, x) - torch.from_numpy(mean).view(1, -1, 1, 1)) / torch.from_numpy(std).view(1, -1, 1, 1)
    torch.testing.assert_close(fb(x), expected, rtol=1e-5, atol=1e-6)


def test_read_noise_is_fresh_in_training_and_reproducible_in_eval():
    R, mean, std = _bank()
    sig = np.full(5, 0.02, np.float32)
    fb = FilterBank(R, mean, std, weight_vector=False, noise_std=sig, eval_seed=3)
    clean = FilterBank(R, mean, std, weight_vector=False)
    x = torch.rand(2, 133, 32, 32)
    fb.train()
    y1, y2 = fb(x), fb(x)
    assert not torch.allclose(y1, y2)                                                     # a fresh draw every forward
    noise = (y1 - clean(x)).permute(1, 0, 2, 3).reshape(5, -1)                            # [5, B*H*W] in standardised units
    np.testing.assert_allclose(noise.std(1).numpy(), sig / std, rtol=0.1)                 # sigma / std per channel
    fb.eval(); e1 = fb(x); e2 = fb(x)
    fb.eval(); e3 = fb(x)
    assert torch.equal(e1, e3) and not torch.equal(e1, e2) and not torch.equal(e1, y1)   # eval() restarts the sequence, forwards advance it
    fb.noise_scale = 0.0
    torch.testing.assert_close(fb(x), clean(x))                                           # a noise-trained bank on clean readings
    assert fb.state_dict()['noise_std'].shape == (1, 5, 1, 1)


def test_train_scale_range_varies_the_noise_level_in_training_only():
    R, mean, std = _bank()
    sig = np.full(5, 0.02, np.float32)
    fb = FilterBank(R, mean, std, weight_vector=False, noise_std=sig, train_scale_range=(0.5, 2.0))
    clean = FilterBank(R, mean, std, weight_vector=False)
    x = torch.rand(1, 133, 32, 32)
    torch.manual_seed(0)
    fb.train()
    levels = np.array([float((fb(x) - clean(x)).std()) for _ in range(40)]) / float(sig[0] / std[0])   # noise std per batch, in units of sigma / std
    assert levels.min() < 0.7 and levels.max() > 1.5 and (levels > 0.45).all() and (levels < 2.1).all()   # log-uniform in [0.5, 2]
    fb.eval()
    level = float((fb(x) - clean(x)).std()) / float(sig[0] / std[0])
    assert abs(level - 1.0) < 0.1                                                         # evaluation at the nominal level
    with pytest.raises(AssertionError):
        FilterBank(R, mean, std, weight_vector=False, train_scale_range=(0.5, 2.0))        # needs noise_std


def test_build_ec_yolo_read_noise_db_range(synthetic_root):
    root, _, _ = synthetic_root
    ds = _ds(root)
    base = dict(yolo_variant='yolo26n', pretrained='none', gate_entropy_weight=0.05, contain_weight=1.0, epochs=1, session='A', seed=0,
                read_noise_db=40.0, read_noise_model='floor', no_gate=False, pca_channels=3)
    m = build_ec_yolo(types.SimpleNamespace(**base, read_noise_db_range=[34.0, 50.0]), ds)
    lo, hi = m.filter_bank.train_scale_range
    assert abs(lo - 10 ** (-10 / 20)) < 1e-6 and abs(hi - 10 ** (6 / 20)) < 1e-6         # 50 dB -> 0.316x sigma, 34 dB -> 2.0x sigma
    assert build_ec_yolo(types.SimpleNamespace(**base, read_noise_db_range=None), ds).filter_bank.train_scale_range is None


def test_noise_state_rewinds_the_eval_sequence():
    R, mean, std = _bank()
    fb = FilterBank(R, mean, std, weight_vector=False, noise_std=np.full(5, 0.02, np.float32), eval_seed=1)
    x = torch.rand(1, 133, 8, 8)
    assert fb.noise_state() is None                                                      # no draw yet
    fb.eval(); a1 = fb(x); a2 = fb(x)                                                    # the reference sequence
    fb.eval(); b1 = fb(x)
    state = fb.noise_state(); fb(x); fb.set_noise_state(state)                           # an extra forward, rewound (evaluate()'s picture)
    b2 = fb(x)
    assert torch.equal(a1, b1) and torch.equal(a2, b2)


def test_projection_matches_folded_pca_channels(synthetic_root):
    root, _, _ = synthetic_root
    ds = _ds(root)
    R, mean, std, _ = ds.filter_bank_tensors(); mu, cov = ds._band_stats()
    V, lam = pca_whitening(R, mean, std, cov, 3)
    two_stage = FilterBank(R, mean, std, weight_vector=False, proj=(V / np.sqrt(lam)).astype(np.float32))
    folded = FilterBank(*pca_whitened_channels(R, mean, std, cov, 3)[:3], weight_vector=False)
    assert two_stage.n_readings == 8 and two_stage.n_channels == 3 and two_stage.state_dict()['proj'].shape == (3, 8)
    x = torch.from_numpy(ds[0][0])[None].float()                                          # [1, 133, H, W]
    ref = two_stage(x)
    torch.testing.assert_close(ref, folded(x), rtol=1e-2, atol=1e-2)                     # same channels, readings kept explicit
    with torch.autocast('cpu', dtype=torch.bfloat16):
        y = two_stage(x.bfloat16())                                                       # low precision in, same precision out ...
    assert y.dtype == torch.bfloat16 and torch.isfinite(y).all()
    torch.testing.assert_close(y.float(), ref, rtol=5e-2, atol=5e-2)                    # ... with the whitening itself done in fp32


def test_noise_regularised_whitening_has_unit_total_variance(synthetic_root):
    root, _, _ = synthetic_root
    ds = _ds(root)
    R, mean, std, _ = ds.filter_bank_tensors(); mu, cov = ds._band_stats()
    sig = read_noise_std(R, mu, cov, 40.0, 'floor')
    nv = (sig / std) ** 2
    V, lam = pca_whitening(R, mean, std, cov, 8, noise_var=nv)
    W = V / np.sqrt(lam)                                                                  # [8, 8] whitening map
    RD = R.astype(np.float64) / std.astype(np.float64)
    C = RD.T @ cov @ RD + np.diag(nv.astype(np.float64))                                  # signal + noise covariance of z
    np.testing.assert_allclose(W.T @ C @ W, np.eye(8), atol=1e-6)                         # whitened including the noise
    V0, lam0 = pca_whitening(R, mean, std, cov, 8)
    assert (lam >= lam0 - 1e-12).all() and lam[-1] >= nv.min() - 1e-12                   # the floor caps the gain 1/sqrt(lam)


def test_read_noise_std_models():
    R, mean, std = _bank(4)
    mu, cov = np.full(133, 0.5), 0.01 * np.eye(133)
    f40, f20 = read_noise_std(R, mu, cov, 40.0, 'floor'), read_noise_std(R, mu, cov, 20.0, 'floor')
    assert f40.shape == (4,) and np.allclose(f40, f40[0]) and np.allclose(f20, 10 * f40)   # one floor for all readings; 20 dB = 10x
    rms = np.sqrt(np.einsum('bn,bc,cn->n', R, cov + np.outer(mu, mu), R))
    assert np.isclose(f40[0], 1e-2 * np.median(rms))
    assert read_noise_std(R, mu, cov, 40.0, 'floor', R_ref=np.concatenate([R, 5 * R], 1))[0] > f40[0]   # the reference bank sets the floor
    np.testing.assert_allclose(read_noise_std(R, mu, cov, 40.0, 'relative'), 1e-2 * (np.abs(R).T @ mu), rtol=1e-6)
    with pytest.raises(ValueError):
        read_noise_std(R, mu, cov, 40.0, 'other')


def test_build_ec_yolo_with_read_noise_round_trips(synthetic_root):
    root, _, _ = synthetic_root
    ds = _ds(root)
    base = dict(yolo_variant='yolo26n', pretrained='none', gate_entropy_weight=0.05, contain_weight=1.0, epochs=1, session='A', seed=7,
                read_noise_db=40.0, read_noise_model='floor', no_gate=False)
    m = build_ec_yolo(types.SimpleNamespace(**base, pca_channels=3), ds)
    fb = m.filter_bank
    assert fb.n_readings == 8 and fb.n_channels == 3 and not fb.weight_vector and fb.eval_seed == 7
    assert fb.noise_std.shape == (1, 8, 1, 1) and fb.proj.shape == (3, 8) and m.yolo.model[0].conv.weight.shape[1] == 3
    R_all, _ = ds.candidate_filter_matrix(); mu, cov = ds._band_stats()
    np.testing.assert_allclose(fb.noise_std.view(-1).numpy(), read_noise_std(ds.filter_bank_tensors()[0], mu, cov, 40.0, R_ref=R_all))
    m2 = build_ec_yolo(types.SimpleNamespace(**base, pca_channels=3), ds)
    m2.load_state_dict(m.state_dict())                                                    # strict: a checkpoint of this model rebuilds
    m3 = build_ec_yolo(types.SimpleNamespace(**base, pca_channels=0), ds)                 # gate + read noise, no whitening
    assert m3.filter_bank.weight_vector and m3.filter_bank.noise_std is not None and m3.filter_bank.proj is None
    x = torch.from_numpy(ds[0][0])[None].float()                                          # [1, 133, H, W]
    m.eval(); out1 = m(x); m.eval(); out2 = m(x)
    pred = lambda o: o[0] if isinstance(o, (list, tuple)) else o
    assert torch.equal(pred(out1), pred(out2))                                            # reproducible eval through the whole model


def test_candidate_filter_matrix_contains_the_selected_columns(synthetic_root):
    root, _, _ = synthetic_root
    ds = _ds(root)
    R_all, v_all = ds.candidate_filter_matrix()
    assert R_all.shape == (ds.n_bands, 344) and v_all.shape == (344,) and not np.any((v_all >= 0.25 - 1e-6) & (v_all <= 0.31 + 1e-6))
    for j, v in enumerate(ds.selected_voltages):
        np.testing.assert_array_equal(R_all[:, int(np.argmin(np.abs(v_all - v)))], ds.sensor_R_matrix[:, j])
    np.testing.assert_allclose(np.abs(R_all).sum(0), 1.0, rtol=1e-5)                      # L1-normalised like sensor_R_matrix
