import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FilterBank(nn.Module):
    '''
    Filter responses on the GPU: y_n = sum_b R[b, n] * x[b] (the raw HSI integrated against the aligned filter
    matrix, x already p99-scaled), standardised per channel with the training statistics, then multiplied by
    the trainable weight vector w = N * softmax(theta) applied as (B, N, H, W) * (1, N, 1, 1).
    The entropy of softmax(theta) (normalised to [0, 1]) is exposed as a sparsity penalty; ranking() gives the
    channel order used to pick the top-k voltages.

    Two optional stages model a real readout (both off by default, so checkpoints trained without them load unchanged):
      noise_std [N]  i.i.d. Gaussian read noise added to every reading y_n BEFORE the standardisation, in reading units
                     (ec_yolo.read_noise_std: a floor of eps * s0 for every reading). Drawn fresh in training; in eval
                     mode drawn from the bank's own generator, which eval() reseeds, so every validation pass sees the
                     same noise (frames are read in a fixed order). noise_scale (a plain float, not state) multiplies
                     it: 0 evaluates a noise-trained model on clean readings.
      proj [N, K]    fixed linear map applied to the standardised readings, u = proj^T z (ec_yolo.pca_whitening with
                     the noise variance folded in); the bank then outputs K channels and carries no weight vector.
                     Keeping the N readings explicit, instead of folding proj into R, is what puts the noise where a
                     device adds it: on each reading, before the whitening amplifies the weak directions.
    '''

    def __init__(self, R, channel_mean, channel_std, weight_vector=True, init_logits=None, noise_std=None, proj=None, eval_seed=0):
        super(FilterBank, self).__init__()
        R = torch.as_tensor(np.asarray(R), dtype=torch.float32)                   # [n_bands, N]
        assert R.ndim == 2, f"R must be [n_bands, N], got {tuple(R.shape)}"
        self.register_buffer('R_t', R.t().contiguous())                           # [N, n_bands]
        self.register_buffer('mean', torch.as_tensor(np.asarray(channel_mean), dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer('std', torch.as_tensor(np.asarray(channel_std), dtype=torch.float32).view(1, -1, 1, 1))
        assert (self.std > 0).all(), f"channel_std must be positive, got min {float(self.std.min()):.3g}"
        self.n_readings = R.shape[1]
        self.n_channels = self.n_readings
        self.weight_vector = weight_vector
        if weight_vector:
            init = torch.zeros(self.n_channels) if init_logits is None else torch.as_tensor(np.asarray(init_logits), dtype=torch.float32)
            assert init.shape == (self.n_channels,), f"init_logits must have shape ({self.n_channels},)"
            self.theta = nn.Parameter(init.clone())                                # [N]
        if noise_std is not None:
            sig = torch.as_tensor(np.asarray(noise_std), dtype=torch.float32).reshape(-1)
            assert sig.shape == (self.n_readings,), f"noise_std must have shape ({self.n_readings},), got {tuple(sig.shape)}"
            assert (sig >= 0).all(), f"noise_std must be non-negative, got min {float(sig.min()):.3g}"
            self.register_buffer('noise_std', sig.view(1, -1, 1, 1))              # [1, N, 1, 1] reading units
        else:
            self.noise_std = None
        if proj is not None:
            P = torch.as_tensor(np.asarray(proj), dtype=torch.float32)
            assert P.ndim == 2 and P.shape[0] == self.n_readings, f"proj must be [{self.n_readings}, K], got {tuple(P.shape)}"
            assert not weight_vector, "a projected (whitened) filter bank has no per-reading weight vector"
            self.register_buffer('proj', P.t().contiguous())                      # [K, N]
            self.n_channels = P.shape[1]
        else:
            self.proj = None
        self.noise_scale = 1.0
        self.eval_seed = int(eval_seed)
        self._gen = None                                                           # eval-mode noise generator, see train()

    @property
    def weights(self):
        if not self.weight_vector:
            return torch.ones(self.n_channels, device=self.R_t.device)
        return self.n_channels * F.softmax(self.theta, dim=0)                       # [N], mean 1

    def entropy(self):
        if not self.weight_vector or self.n_channels < 2:
            return torch.zeros((), device=self.R_t.device)
        p = F.softmax(self.theta, dim=0)
        return -(p * torch.log(p + 1e-12)).sum() / math.log(self.n_channels)

    def ranking(self):
        return torch.argsort(self.weights.detach(), descending=True)

    def train(self, mode=True):
        # eval() restarts the eval-mode noise sequence, so two evaluation passes over the same frames in the same order
        # (every validation epoch, main_det --eval, main_det_compare) draw identical noise
        super(FilterBank, self).train(mode)
        if not mode:
            self._gen = None
        return self

    def _noise(self, y):
        '''Standard-normal draws shaped like y: the global RNG in training, the reseeded bank generator in eval mode.'''
        if self.training:
            return torch.randn_like(y)
        if self._gen is None or self._gen.device != y.device:
            self._gen = torch.Generator(device=y.device)
            self._gen.manual_seed(self.eval_seed)
        return torch.randn(y.shape, generator=self._gen, device=y.device, dtype=y.dtype)

    def noise_state(self):
        '''State of the eval-mode noise generator (None before its first draw); set_noise_state() rewinds to it, so a forward
        that is not part of the evaluation (e.g. the picture evaluate() logs) does not advance the sequence.'''
        return None if self._gen is None else self._gen.get_state()

    def set_noise_state(self, state):
        if state is not None:
            self._gen.set_state(state)

    def _standardise(self, y):
        '''Read noise on every reading, then the per-channel standardisation.  y: [B, N, H, W] readings'''
        if self.noise_std is not None and self.noise_scale > 0:
            y = y + self._noise(y) * (self.noise_std.to(y.dtype) * self.noise_scale)
        return (y - self.mean.to(y.dtype)) / self.std.to(y.dtype)

    def forward(self, x):
        y = torch.einsum('nc,bchw->bnhw', self.R_t.to(x.dtype), x)                # [B, N, H, W] readings (fp16 under autocast, as in every run)
        if self.proj is None:
            y = self._standardise(y)
            return y if not self.weight_vector else y * self.weights.to(y.dtype).view(1, -1, 1, 1)   # (B, N, H, W) * (1, N, 1, 1)
        # the whitening map cancels near-duplicate readings against each other, which fp16 would round at 1e-3, so from
        # here on fp32 with autocast off (an einsum on .float() inputs would be re-cast by autocast). The readings keep
        # fp16's rounding, 20x below a 40 dB noise floor and capped like the noise by proj's regularisation; the fp32
        # [B, N, H, W] copy is small for the N <= 16 readings a whitened bank has.
        with torch.autocast(x.device.type, enabled=False):
            y = torch.einsum('kn,bnhw->bkhw', self.proj, self._standardise(y.float()))   # [B, K, H, W]
        return y.to(x.dtype)
