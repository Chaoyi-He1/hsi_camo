'''
N-channel stems for the Stage-2 segmenters (spec §3 "regress-and-fold", §5.3).

Every pretrained segmenter starts from an RGB stem conv W_rgb [C_out, 3, k, k] that expects ImageNet-normalised RGB.
For an arm with N channels z (raw: 133 standardised bands; ec10 / ec24: the whitened readings; rgb: the normalised
pseudo-RGB render itself), a least-squares map rgb_norm ~ P z + q (P [3, N], q [3]) is fitted on training pixels and
folded into a new conv with N + n_extra inputs:
    W_N[o, c] = sum_k W_rgb[o, k] P[k, c]                      (per kernel tap)
    b_N[o]    = b_rgb[o] + sum_{k, i, j} W_rgb[o, k, i, j] q_k
    W_N[o, N:] = 0                                             (the box channel and any other extra input start silent)
so at initialisation the network sees the pseudo-RGB image (exactly, wherever the kernel does not touch the zero
padding: there the folded bias still adds q for the padded taps while the RGB conv saw zeros), and training adds the
spectral cues on top. The pseudo-RGB render is the same for every arm: mean of the p99-scaled bands in [600, 700) nm (R),
[500, 600) (G), [400, 500) (B), clipped to [0, 1]. The pixel sampling for the fit is main_seg.fit_arm_rgb_map.
'''
import numpy as np
import torch
import torch.nn as nn

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)   # SAM2 / PVTv2 / ZoomNeXt input normalisation
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
RGB_WINDOWS = ((600.0, 700.0), (500.0, 600.0), (400.0, 500.0))      # nm, [lo, hi) for R, G, B


def rgb_band_indices(wavelens):
    '''Band indices of the R, G, B windows (RGB_WINDOWS) in wavelens [n_bands] (nm); each window must hold a band.'''
    wl = np.asarray(wavelens, dtype=np.float64)
    out = []
    for lo, hi in RGB_WINDOWS:
        idx = np.where((wl >= lo) & (wl < hi))[0]
        assert len(idx) > 0, f"no band in [{lo}, {hi}) nm (wavelens {wl.min():.1f}-{wl.max():.1f})"
        out.append(idx)
    return out


def pseudo_rgb(x, wavelens):
    '''
    Pseudo-RGB render of a p99-scaled cube: x [B, n_bands, h, w] tensor (any float dtype), wavelens np [n_bands] nm.
    Returns [B, 3, h, w] float32 in [0, 1] (band-window means, computed in float32, then clipped).
    '''
    assert x.ndim == 4 and x.shape[1] == len(wavelens), \
        f"x must be [B, {len(wavelens)}, h, w] to match wavelens, got {tuple(x.shape)}"
    x = x.float()
    chans = [x.index_select(1, torch.as_tensor(idx, device=x.device)).mean(dim=1) for idx in rgb_band_indices(wavelens)]
    return torch.stack(chans, dim=1).clamp_(0.0, 1.0)                     # [B, 3, h, w]


def imagenet_normalize(rgb):
    '''(rgb - IMAGENET_MEAN) / IMAGENET_STD per channel; rgb [B, 3, h, w] in [0, 1] -> float32 [B, 3, h, w].'''
    mean = torch.as_tensor(IMAGENET_MEAN, device=rgb.device).view(1, 3, 1, 1)
    std = torch.as_tensor(IMAGENET_STD, device=rgb.device).view(1, 3, 1, 1)
    return (rgb.float() - mean) / std


def fit_rgb_map(z, rgb_norm):
    '''
    Least squares with intercept: rgb_norm ~ P z + q.
    z [n, N] arm channels per pixel, rgb_norm [n, 3] ImageNet-normalised pseudo-RGB per pixel.
    Returns P [3, N] float32, q [3] float32. Solved in float64 with numpy's lstsq (minimum-norm solution when z is rank
    deficient, e.g. the near-empty whitened channels of ec24, so those get ~0 weight instead of a blow-up).
    '''
    z = np.asarray(z, dtype=np.float64); rgb_norm = np.asarray(rgb_norm, dtype=np.float64)
    assert z.ndim == 2 and rgb_norm.ndim == 2 and rgb_norm.shape[1] == 3 and len(z) == len(rgb_norm), \
        f"z must be [n, N] and rgb_norm [n, 3] with the same n, got {z.shape} and {rgb_norm.shape}"
    assert len(z) > z.shape[1], f"need more pixels ({len(z)}) than channels + 1 ({z.shape[1] + 1})"
    A = np.concatenate([z, np.ones((len(z), 1))], axis=1)                 # [n, N + 1]
    sol, _, _, _ = np.linalg.lstsq(A, rgb_norm, rcond=None)               # [N + 1, 3]
    P, q = sol[:-1].T, sol[-1]                                            # [3, N], [3]
    return P.astype(np.float32), q.astype(np.float32)


def _new_conv_like(conv, in_channels):
    '''A Conv2d with conv's geometry, in_channels inputs and a bias, on conv's device and dtype.'''
    new = nn.Conv2d(in_channels, conv.out_channels, conv.kernel_size, stride=conv.stride, padding=conv.padding,
                    dilation=conv.dilation, groups=1, bias=True, padding_mode=conv.padding_mode)
    return new.to(device=conv.weight.device, dtype=conv.weight.dtype)


def fold_stem(conv, P, q, n_extra=1):
    '''
    Fold rgb_norm ~ P z + q into the RGB stem conv (nn.Conv2d, 3 inputs, groups 1): returns a new trainable nn.Conv2d
    with N + n_extra inputs (N = P.shape[1]); the n_extra trailing input slices are zero. A bias is created when conv had
    none. The weights of conv itself are not modified.
    '''
    assert isinstance(conv, nn.Conv2d) and conv.in_channels == 3 and conv.groups == 1, \
        f"fold_stem needs an RGB Conv2d (3 inputs, groups 1), got {conv}"
    P = torch.as_tensor(np.asarray(P), dtype=torch.float64); q = torch.as_tensor(np.asarray(q), dtype=torch.float64)
    assert P.ndim == 2 and P.shape[0] == 3 and q.shape == (3,), f"P must be [3, N] and q [3], got {tuple(P.shape)}, {tuple(q.shape)}"
    N = P.shape[1]
    W = conv.weight.detach().double().cpu()                               # [C_out, 3, k, k]
    b = conv.bias.detach().double().cpu() if conv.bias is not None else torch.zeros(conv.out_channels, dtype=torch.float64)
    W_N = torch.einsum('okij,kc->ocij', W, P)                             # [C_out, N, k, k]
    b_N = b + torch.einsum('okij,k->o', W, q)                             # [C_out]
    new = _new_conv_like(conv, N + n_extra)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :N] = W_N.to(new.weight)
        new.bias.copy_(b_N.to(new.bias))
    return new


def mean_stem(conv, n_in, n_extra=1):
    '''
    Fallback initialisation (spec §3): every one of the n_in inputs gets the mean RGB kernel x 3 / n_in (the response to
    a grey image is kept), the bias is copied, the n_extra trailing slices are zero.
    '''
    assert isinstance(conv, nn.Conv2d) and conv.in_channels == 3 and conv.groups == 1, \
        f"mean_stem needs an RGB Conv2d (3 inputs, groups 1), got {conv}"
    new = _new_conv_like(conv, n_in + n_extra)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :n_in] = conv.weight.mean(dim=1, keepdim=True) * (3.0 / n_in)
        new.bias.copy_(conv.bias if conv.bias is not None else torch.zeros_like(new.bias))
    return new
