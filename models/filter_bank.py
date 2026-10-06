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
                     (read_noise_std below: a floor of eps * s0 for every reading). Drawn fresh in training; in eval
                     mode drawn from the bank's own generator, which eval() reseeds, so every validation pass sees the
                     same noise (frames are read in a fixed order). noise_scale (a plain float, not state) multiplies
                     it: 0 evaluates a noise-trained model on clean readings. train_scale_range (lo, hi) draws, in
                     training mode only, a further log-uniform multiplier of the noise std per forward (SNR augmentation):
                     a model trained at one fixed noise level works only at that level (test AP50 halves 2 dB away,
                     spec §12), so the level the network sees has to vary during training.
      proj [N, K]    fixed linear map applied to the standardised readings, u = proj^T z (pca_whitening below, with
                     the noise variance folded in); the bank then outputs K channels and carries no weight vector.
                     Keeping the N readings explicit, instead of folding proj into R, is what puts the noise where a
                     device adds it: on each reading, before the whitening amplifies the weak directions.
    '''

    def __init__(self, R, channel_mean, channel_std, weight_vector=True, init_logits=None, noise_std=None, proj=None, eval_seed=0,
                 train_scale_range=None):
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
        self.train_scale_range = None if train_scale_range is None else (float(train_scale_range[0]), float(train_scale_range[1]))
        if self.train_scale_range is not None:
            assert noise_std is not None, "train_scale_range needs noise_std"
            assert 0 < self.train_scale_range[0] <= self.train_scale_range[1], f"train_scale_range must be 0 < lo <= hi, got {train_scale_range}"
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
            scale = self.noise_scale
            if self.training and self.train_scale_range is not None:
                lo, hi = self.train_scale_range                                    # SNR augmentation: one log-uniform level per batch
                scale = scale * lo * (hi / lo) ** float(torch.rand(()))
            y = y + self._noise(y) * (self.noise_std.to(y.dtype) * scale)
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


def pca_whitening(R, mean, std, band_cov, k, noise_var=None):
    '''
    Eigen-decomposition of the covariance of the standardised readings z = (R^T x - mean) / std under the training band
    covariance: the top-k eigenvectors V [N, k] and eigenvalues lam [k], so u = V^T z / sqrt(lam) are k uncorrelated
    unit-variance channels. noise_var [N] (per-reading read-noise variance in standardised units, (sigma / std)^2) is
    added to the diagonal first: the directions are then ordered by signal-plus-noise variance and 1/sqrt(lam) never
    amplifies a noise-dominated direction beyond unit output noise (a whitening fitted to noise-free statistics gains up
    to ~1600x on the weakest of 344 responses and breaks on a real readout, see spec §12).
    '''
    R = np.asarray(R, np.float64); mean = np.asarray(mean, np.float64); std = np.asarray(std, np.float64)
    assert 1 <= k <= R.shape[1], f"pca_channels={k} must be in [1, {R.shape[1]}]"
    Dinv = 1.0 / std                                                                  # [N]
    C = (R * Dinv).T @ np.asarray(band_cov, np.float64) @ (R * Dinv)                  # [N, N] covariance of z
    if noise_var is not None:
        nv = np.asarray(noise_var, np.float64).reshape(-1)
        assert nv.shape == (R.shape[1],) and (nv >= 0).all(), f"noise_var must be {R.shape[1]} non-negative variances"
        C = C + np.diag(nv)
    evals, evecs = np.linalg.eigh(C)
    order = np.argsort(evals)[::-1][:k]
    return evecs[:, order], np.maximum(evals[order], 1e-12)                           # [N, k], [k]


def pca_whitened_channels(R, mean, std, band_cov, k, noise_var=None):
    '''
    Top-k principal directions of the standardised responses z = (R^T x - mean) / std under the training band covariance,
    returned as an equivalent (R', mean', std') so the FilterBank stays a linear projection + standardisation:
    u = V^T z = (R D^-1 V)^T x - V^T D^-1 mean, with std' = sqrt(eigenvalues) so every output channel has unit variance.
    Why: the 344 EC responses span an ~11-dimensional subspace (adjacent voltages 0.9998 cosine-similar), so feeding them
    all gives the first conv a near-singular input; k whitened directions keep the information and fix the conditioning.
    noise_var is passed through to pca_whitening (the folded form cannot carry per-reading noise itself: build_filter_bank
    keeps the readings explicit and hands FilterBank the map as `proj` when read noise is simulated).
    '''
    R = np.asarray(R, np.float64); mean = np.asarray(mean, np.float64); std = np.asarray(std, np.float64)
    V, lam = pca_whitening(R, mean, std, band_cov, k, noise_var)
    Dinv = 1.0 / std                                                                  # [N]
    R_new = (R * Dinv) @ V                                                            # [n_bands, k]
    mean_new = V.T @ (Dinv * mean)                                                    # [k]
    std_new = np.sqrt(lam)                                                            # [k]
    return R_new.astype(np.float32), mean_new.astype(np.float32), std_new.astype(np.float32), np.arange(1, k + 1, dtype=np.float64)


def read_noise_std(R, mu, cov, snr_db, model='floor', R_ref=None):
    '''
    Per-reading Gaussian read-noise std [N], in reading units, for the readings y = R^T x (x in p99 units) at snr_db:
      floor     one absolute floor for every reading, sigma = s0 / 10^(dB/20), s0 = median over the reference readings
                of their RMS value sqrt(R_v^T (Sigma + mu mu^T) R_v). R_ref defaults to R; pass the whole usable bank so
                the floor is a property of the device and not of the voltages chosen. Weak readings get the worst SNR,
                as under a real read-noise floor; the voltage selection (main_select_voltages) scores sets under it.
      relative  every reading at the same SNR, sigma_v = |R_v|^T mu / 10^(dB/20) (per-reading auto-exposure; the
                'noise<dB>' condition of docs/reports/2026-09-29-pca11-robustness/robust_eval.py).
    '''
    R = np.asarray(R, np.float64); mu = np.asarray(mu, np.float64); cov = np.asarray(cov, np.float64)
    eps = 10.0 ** (-float(snr_db) / 20.0)
    if model == 'floor':
        Rr = R if R_ref is None else np.asarray(R_ref, np.float64)
        rms = np.sqrt(np.maximum(np.einsum('bn,bc,cn->n', Rr, cov + np.outer(mu, mu), Rr), 0.0))   # [N_ref]
        return np.full(R.shape[1], eps * float(np.median(rms)), dtype=np.float32)
    if model == 'relative':
        return (eps * (np.abs(R).T @ mu)).astype(np.float32)                                      # [N]
    raise ValueError(f"unknown read-noise model '{model}', expected floor | relative")


def assert_rgb_compatible(args):
    '''
    The Stage-1 RGB baseline (--rgb_images) is a plain camera input: the dataset's 3-channel frame through an identity front
    end, no gate, no read noise, no whitening, and no session-B slicing (nothing to rank). Asserts that no flag asks for
    more, with the CLI spelling in the message; a no-op without rgb_images. main_det.main calls it before anything is built
    and build_filter_bank again, so a checkpoint rebuilt by Stage 2 or main_det_compare is held to the same rule.
    '''
    if not getattr(args, 'rgb_images', False):
        return
    assert not getattr(args, 'raw_bands', False), "--rgb_images (the RGB camera baseline) cannot be combined with --raw-bands (the cube's bands)"
    k = int(getattr(args, 'pca_channels', 0) or 0)
    assert k == 0, f"--rgb_images has no EC responses to whiten, it cannot be combined with --pca-channels {k}"
    noise_db = float(getattr(args, 'read_noise_db', 0.0) or 0.0)
    assert noise_db == 0.0, f"--rgb_images is the noise-free camera baseline, it cannot be combined with --read_noise_db {noise_db:g}"
    assert args.session == 'A', "--rgb_images has no gate to rank and no channels to slice, it cannot be combined with session B (--session A)"


def rgb_filter_bank(args, dataset):
    '''
    The RGB baseline's front end: the dataset's own camera frame (HyperCOD_data(rgb_images=True), uint8 / 255) through an
    identity FilterBank (R = I_3, no weight vector, no read noise, no whitening) that standardises each of R, G, B with the
    train RGB statistics (dataset._rgb_stats, band_stats.compute_rgb_stats): mean = mu, std = sqrt(diag(cov)). The first conv
    of the YOLO then sees 3 channels, so its init is the exact COCO RGB kernel (ec_yolo.adapt_first_conv_weight with N = 3).
    Returns (fb, volts) like build_filter_bank, volts = [0, 1, 2] (the channel index: R, G, B).
    '''
    assert_rgb_compatible(args)
    mu, cov = dataset._rgb_stats()                                                       # [3], [3, 3] float64
    mean, std = mu.astype(np.float32), np.sqrt(np.maximum(np.diag(cov), 0.0)).astype(np.float32)
    fb = FilterBank(np.eye(3, dtype=np.float32), mean, std, weight_vector=False, eval_seed=int(getattr(args, 'seed', 0) or 0))
    print(f"RGB baseline: identity front end over the 3 camera channels, mean {[round(float(v), 3) for v in mean]}, "
          f"std {[round(float(v), 3) for v in std]} (of /255 values), no gate, no read noise")
    return fb, np.arange(3, dtype=np.float64)


def build_filter_bank(args, dataset):
    '''
    The input front end of a detector arm, factored out of models.ec_yolo.build_ec_yolo so that Stage 1 (the detector)
    and Stage 2 (models.seg_models.build_front_end) build it from the same flags and see bit-identical channels.
      args     the detector's flags (main_det.get_args_parser(), or a checkpoint's ckpt['args']). Every flag except
               `session` is read through getattr with its Stage-1 default, so older checkpoints (raw133_A has no
               pca_channels / read_noise_* / no_gate) and the tests' sparse namespaces keep working:
                 rgb_images           the RGB camera baseline, see rgb_filter_bank (excludes everything below)
                 raw_bands            identity over the cube bands, standardised with the training band statistics
                 (otherwise)          the dataset's selected EC responses (filter_select / filter_voltages / num_filters);
                                      session A carries the trainable weight vector unless no_gate
                 pca_channels K       the top-K whitened principal directions of the responses; no weight vector
                 read_noise_db        per-reading read noise inside the bank (read_noise_std, model read_noise_model)
                 read_noise_db_range  SNR augmentation of the training noise level
                 seed                 the eval-mode noise seed
      dataset  a HyperCOD_data built with main_det.dataset_kwargs(args): band statistics, filter matrices, wavelengths
    Returns (fb, volts): the FilterBank and np.ndarray [N] of what each channel is (voltages; band centres in nm for
    raw_bands; 1..K for whitened channels; 0, 1, 2 = R, G, B for rgb_images), which build_ec_yolo stores as
    model.selected_voltages.
    '''
    if getattr(args, 'rgb_images', False):
        return rgb_filter_bank(args, dataset)     # before dataset._band_stats(): the camera frame never touches the cube
    noise_db = float(getattr(args, 'read_noise_db', 0.0) or 0.0)
    noise_model = getattr(args, 'read_noise_model', None) or 'floor'
    mu, cov = dataset._band_stats()                                                      # [n_bands], [n_bands, n_bands]
    noise_std = proj = None
    if getattr(args, 'raw_bands', False):
        # control run: the raw cube bands inside band_range go straight into the model (identity projection, standardised
        # with the training band statistics, no weight vector) to compare against the EC filter responses
        R = np.eye(dataset.n_bands, dtype=np.float32)
        mean, std = mu.astype(np.float32), np.sqrt(np.maximum(np.diag(cov), 0.0)).astype(np.float32)
        volts, weight_vector = dataset.wavelens.copy(), False                            # 'voltages' = band centres (nm)
        if noise_db > 0:
            noise_std = read_noise_std(R, mu, cov, noise_db, noise_model)                # per band, floor over the bands
    else:
        R, mean, std, volts = dataset.filter_bank_tensors()
        weight_vector = (args.session == 'A') and not getattr(args, 'no_gate', False)
        if noise_db > 0:
            R_all, _ = dataset.candidate_filter_matrix()                                 # every usable voltage: the floor is the device's
            noise_std = read_noise_std(R, mu, cov, noise_db, noise_model, R_ref=R_all)
        k = int(getattr(args, 'pca_channels', 0) or 0)
        if k > 0:
            if noise_std is None:
                R, mean, std, volts = pca_whitened_channels(R, mean, std, cov, k)        # folded form, as the runs before read noise
            else:
                # the readings stay explicit so the noise lands on them before the whitening (FilterBank docstring)
                V, lam = pca_whitening(R, mean, std, cov, k, noise_var=(noise_std / std) ** 2)
                proj, volts = (V / np.sqrt(lam)).astype(np.float32), np.arange(1, k + 1, dtype=np.float64)
            weight_vector = False
    scale_range = None
    db_range = getattr(args, 'read_noise_db_range', None)
    if noise_std is not None and db_range:
        # SNR augmentation: the noise level seen in training varies log-uniformly between the two dB values (as multipliers
        # of the nominal sigma); the whitening and the evaluation keep the nominal --read_noise_db
        lo_db, hi_db = sorted(float(v) for v in db_range)
        scale_range = (10 ** ((noise_db - hi_db) / 20), 10 ** ((noise_db - lo_db) / 20))
    fb = FilterBank(R, mean, std, weight_vector=weight_vector, noise_std=noise_std, proj=proj, eval_seed=int(getattr(args, 'seed', 0) or 0),
                    train_scale_range=scale_range)
    if noise_std is not None:
        print(f"read noise {noise_db:g} dB ({noise_model}): sigma {noise_std.min():.3g}-{noise_std.max():.3g} per reading"
              + (f", whitening of {fb.n_readings} readings -> {fb.n_channels} channels regularised by it" if proj is not None else "")
              + (f", training level drawn from {lo_db:g}-{hi_db:g} dB per batch" if scale_range is not None else ""))
    return fb, volts
