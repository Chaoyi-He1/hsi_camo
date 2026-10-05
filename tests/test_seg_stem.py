import numpy as np
import torch
import torch.nn as nn
import pytest

from data_loader.ec_filter import WAVELENS_200, band_indices
from models.seg_stem import (IMAGENET_MEAN, IMAGENET_STD, rgb_band_indices, pseudo_rgb, imagenet_normalize, fit_rgb_map,
                             fold_stem, mean_stem)

WL = WAVELENS_200[band_indices((400.0, 800.0))]                           # [133] the Stage-1 band window


def test_band_windows():
    r, g, b = rgb_band_indices(WL)
    assert (len(r), len(g), len(b)) == (33, 33, 34)
    assert WL[r].min() >= 600 and WL[r].max() < 700 and WL[b].min() >= 400 and WL[b].max() < 500
    with pytest.raises(AssertionError):
        rgb_band_indices(np.linspace(650, 800, 20))                       # no band below 600 nm


def test_pseudo_rgb_window_means_and_clip():
    r, g, b = rgb_band_indices(WL)
    x = torch.zeros(2, 133, 4, 5, dtype=torch.float16)
    x[:, r] = 0.5; x[:, g] = 0.25; x[:, b] = 2.0                           # B is above 1 -> clipped
    x[1, r[0]] = 0.5 + 33 * 0.25                                          # one bright band shifts the R mean by 0.25
    out = pseudo_rgb(x, WL)
    assert out.shape == (2, 3, 4, 5) and out.dtype == torch.float32
    torch.testing.assert_close(out[0], torch.tensor([0.5, 0.25, 1.0]).view(3, 1, 1).expand(3, 4, 5))
    torch.testing.assert_close(out[1, 0], torch.full((4, 5), 0.75))
    n = imagenet_normalize(out)
    torch.testing.assert_close(n[0, :, 0, 0], torch.from_numpy((np.array([0.5, 0.25, 1.0], np.float32) - IMAGENET_MEAN) / IMAGENET_STD))
    with pytest.raises(AssertionError):
        pseudo_rgb(x[:, :100], WL)


def test_fit_rgb_map_recovers_linear_map():
    rng = np.random.default_rng(0)
    for N in (3, 10, 24):
        P0 = rng.normal(size=(3, N)).astype(np.float32); q0 = rng.normal(size=3).astype(np.float32)
        z = rng.normal(size=(5000, N))
        P, q = fit_rgb_map(z, z @ P0.T + q0)
        assert P.shape == (3, N) and q.shape == (3,) and P.dtype == np.float32
        np.testing.assert_allclose(P, P0, atol=1e-5); np.testing.assert_allclose(q, q0, atol=1e-5)
    # rank-deficient z (a duplicated channel, like the near-empty whitened ec24 channels): finite, and still exact on z
    z = rng.normal(size=(2000, 4)); z = np.concatenate([z, z[:, :1]], axis=1)
    rgb = z[:, :4] @ rng.normal(size=(4, 3)) + 0.3
    P, q = fit_rgb_map(z, rgb)
    assert np.isfinite(P).all() and np.allclose(z @ P.T + q, rgb, atol=1e-4)


@pytest.mark.parametrize('N', [3, 10, 24])
def test_fold_stem_reproduces_rgb_stem(N):
    torch.manual_seed(0)
    conv = nn.Conv2d(3, 16, 7, stride=4, padding=3, bias=True)           # the Hiera / PVTv2 stem geometry
    rng = np.random.default_rng(N)
    P = rng.normal(size=(3, N)).astype(np.float32); q = rng.normal(size=3).astype(np.float32)
    z = torch.randn(2, N, 64, 64)                                         # [B, N, h, w]
    rgb_norm = torch.einsum('kc,bchw->bkhw', torch.from_numpy(P), z) + torch.from_numpy(q).view(1, 3, 1, 1)
    w0 = conv.weight.detach().clone()
    new = fold_stem(conv, P, q, n_extra=1)
    assert new.in_channels == N + 1 and new.out_channels == 16 and new.kernel_size == (7, 7)
    assert new.stride == (4, 4) and new.padding == (3, 3) and new.weight.requires_grad and new.bias.requires_grad
    assert torch.equal(new.weight[:, N:], torch.zeros_like(new.weight[:, N:])) and torch.equal(conv.weight, w0)
    box = (torch.rand(2, 1, 64, 64) > 0.5).float()                        # the box channel has no effect at init
    ref, out = conv(rgb_norm), new(torch.cat([z, box], dim=1))           # [2, 16, 16, 16]
    # output row/col 0 is the only one whose 7x7 window (rows 4i-3..4i+3) reaches the zero padding at 64 px
    torch.testing.assert_close(out[:, :, 1:, 1:], ref[:, :, 1:, 1:], atol=1e-4, rtol=1e-4)
    assert not torch.allclose(out[:, :, 0, 0], ref[:, :, 0, 0], atol=1e-3)   # the border differs (folded bias adds q)
    out.sum().backward()
    assert new.weight.grad is not None and new.weight.grad[:, N:].abs().sum() > 0   # the box slice learns


def test_fold_stem_creates_bias_and_rejects_non_rgb():
    conv = nn.Conv2d(3, 8, 3, padding=1, bias=False)
    P, q = np.eye(3, dtype=np.float32), np.array([1.0, -2.0, 0.5], np.float32)
    new = fold_stem(conv, P, q, n_extra=2)
    assert new.in_channels == 5 and new.bias is not None
    exp = torch.einsum('okij,k->o', conv.weight.detach(), torch.from_numpy(q))
    torch.testing.assert_close(new.bias.detach(), exp)
    torch.testing.assert_close(new.weight[:, :3], conv.weight)            # P = I keeps the RGB kernel (the rgb arm)
    with pytest.raises(AssertionError):
        fold_stem(nn.Conv2d(4, 8, 3), P, q)
    with pytest.raises(AssertionError):
        fold_stem(conv, np.eye(4, dtype=np.float32), q)


def test_mean_stem_keeps_grey_response():
    torch.manual_seed(0)
    conv = nn.Conv2d(3, 8, 3, padding=1)
    new = mean_stem(conv, n_in=10, n_extra=1)
    grey = torch.full((1, 3, 6, 6), 0.4)
    x = torch.cat([torch.full((1, 10, 6, 6), 0.4), torch.rand(1, 1, 6, 6)], dim=1)
    torch.testing.assert_close(new(x), conv(grey), atol=1e-5, rtol=1e-5)
    assert torch.equal(new.weight[:, 10:], torch.zeros_like(new.weight[:, 10:]))
