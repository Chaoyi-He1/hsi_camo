import types
import numpy as np

from data_loader.my_dataset import HyperCOD_data
from models.ec_yolo import build_ec_yolo, pca_whitened_channels


def _ds(root):
    return HyperCOD_data(str(root), split='train', use_filter=False, norm='p99', crop_size=0, num_filters=8, filter_norm='l1', out_dtype='float16')


def test_pca_whitened_channels_are_unit_variance_and_orthogonal(synthetic_root):
    root, _, _ = synthetic_root
    ds = _ds(root)
    R, mean, std, _ = ds.filter_bank_tensors(); mu, cov = ds._band_stats()
    Rk, mk, sk, labels = pca_whitened_channels(R, mean, std, cov, 3)
    assert Rk.shape == (133, 3) and mk.shape == (3,) and sk.shape == (3,) and (sk > 0).all() and labels.tolist() == [1.0, 2.0, 3.0]
    C = Rk.T.astype(np.float64) @ cov @ Rk.astype(np.float64)                  # covariance of the new channels under the band stats
    np.testing.assert_allclose(np.diag(C), sk.astype(np.float64) ** 2, rtol=1e-3)   # std' = sqrt(variance): whitened after standardisation
    off = C - np.diag(np.diag(C)); assert np.abs(off).max() < 1e-3 * np.diag(C).max()  # principal directions are uncorrelated
    np.testing.assert_allclose(Rk.T.astype(np.float64) @ mu, mk, rtol=1e-3, atol=1e-5)  # mean' is the projected training mean
    assert sk[0] >= sk[1] >= sk[2]                                                  # ordered by variance


def test_build_ec_yolo_pca_and_no_gate(synthetic_root):
    root, _, _ = synthetic_root
    ds = _ds(root)
    base = dict(yolo_variant='yolo26n', pretrained='none', gate_entropy_weight=0.05, contain_weight=1.0, epochs=1, session='A')
    m = build_ec_yolo(types.SimpleNamespace(**base, pca_channels=3, no_gate=False), ds)
    assert m.filter_bank.n_channels == 3 and not m.filter_bank.weight_vector and m.yolo.model[0].conv.weight.shape[1] == 3
    m2 = build_ec_yolo(types.SimpleNamespace(**base, pca_channels=0, no_gate=True), ds)
    assert m2.filter_bank.n_channels == 8 and not m2.filter_bank.weight_vector
    m3 = build_ec_yolo(types.SimpleNamespace(**base, pca_channels=0, no_gate=False), ds)
    assert m3.filter_bank.n_channels == 8 and m3.filter_bank.weight_vector
