import numpy as np
import torch
import pytest

from models.filter_bank import FilterBank
from data_loader.my_dataset import HyperCOD_data


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


def test_filter_bank_is_fp16_safe_under_autocast():
    if not torch.cuda.is_available():
        pytest.skip("cuda")
    fb = FilterBank(np.random.rand(133, 8).astype(np.float32) / 133, np.zeros(8, np.float32), np.ones(8, np.float32)).cuda()
    x = torch.rand(1, 133, 64, 64, device='cuda', dtype=torch.float16)
    with torch.autocast('cuda', dtype=torch.float16):
        y = fb(x)
    assert y.dtype == torch.float16 and torch.isfinite(y).all()
