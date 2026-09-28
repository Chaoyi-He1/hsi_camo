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
    '''

    def __init__(self, R, channel_mean, channel_std, weight_vector=True, init_logits=None):
        super(FilterBank, self).__init__()
        R = torch.as_tensor(np.asarray(R), dtype=torch.float32)                   # [n_bands, N]
        assert R.ndim == 2, f"R must be [n_bands, N], got {tuple(R.shape)}"
        self.register_buffer('R_t', R.t().contiguous())                           # [N, n_bands]
        self.register_buffer('mean', torch.as_tensor(np.asarray(channel_mean), dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer('std', torch.as_tensor(np.asarray(channel_std), dtype=torch.float32).view(1, -1, 1, 1))
        assert (self.std > 0).all(), "channel_std must be positive"
        self.n_channels = R.shape[1]
        self.weight_vector = weight_vector
        if weight_vector:
            init = torch.zeros(self.n_channels) if init_logits is None else torch.as_tensor(np.asarray(init_logits), dtype=torch.float32)
            assert init.shape == (self.n_channels,), f"init_logits must have shape ({self.n_channels},)"
            self.theta = nn.Parameter(init.clone())                                # [N]

    @property
    def weights(self):
        if not self.weight_vector:
            return torch.ones(self.n_channels, device=self.R_t.device)
        return self.n_channels * F.softmax(self.theta, dim=0)                       # [N], mean 1

    def entropy(self):
        if not self.weight_vector:
            return torch.zeros((), device=self.R_t.device)
        p = F.softmax(self.theta, dim=0)
        return -(p * torch.log(p + 1e-12)).sum() / math.log(self.n_channels)

    def ranking(self):
        return torch.argsort(self.weights.detach(), descending=True)

    def forward(self, x):
        y = torch.einsum('nc,bchw->bnhw', self.R_t.to(x.dtype), x)                # [B, N, H, W] filter responses
        y = (y - self.mean.to(y.dtype)) / self.std.to(y.dtype)                    # per-channel standardisation
        return y * self.weights.to(y.dtype).view(1, -1, 1, 1)                     # (B, N, H, W) * (1, N, 1, 1)
