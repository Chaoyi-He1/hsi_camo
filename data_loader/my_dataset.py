import torch.utils.data as Dataset
import os
import csv
import random
import argparse
import numpy as np
import h5py
import torch
from PIL import Image

from data_loader.ec_filter import (N_BANDS, WAVELENS_200, band_indices, load_ec_filter, align_filter_to_wavelens,
                                   select_filter_channels)
from data_loader.band_stats import STATS_CROP_SIZE, default_stats_path, load_band_stats, compute_band_stats

GT_THRESHOLD = 127            # GT pngs are JPEG-compressed with 3 identical channels; foreground = channel 0 > 127
HYPERCUBE_KEY = 'hypercube'   # variable name inside the MATLAB v7.3 (HDF5) .mat cubes


class HyperCOD_data(Dataset.Dataset):
    def __init__(self, data_path, split='train', use_filter=True, filter_path=None, num_filters=30,
                 filter_select='uniform', filter_voltages=None, crop_size=512, obj_crop_prob=0.5,
                 norm='p99z', band_range=(400.0, 800.0), filter_norm='l1', stats_path=None, seed=None):
        super(HyperCOD_data, self).__init__()
        self.data_path = data_path
        self.split = split  # 'train' or 'test'
        self.use_filter = use_filter  # True: simulated EC detector channels, False: raw 200 bands
        self.filter_path = filter_path if filter_path is not None else os.path.join(data_path, 'EC_filterV3.mat')
        self.num_filters = num_filters
        self.filter_select = filter_select
        self.filter_voltages = filter_voltages
        self.crop_size = crop_size  # training crop at native resolution, 0 means full frame
        self.obj_crop_prob = obj_crop_prob  # probability that a training crop is placed to contain the object
        # normalisation layers: 'p99' = per-scene scalar (cube / (intensity_p99 / 200)),
        # 'p99z' = p99 followed by per-channel standardisation with training band statistics
        self.norm = norm  # 'none', 'p99' or 'p99z'
        # 'l1' divides every selected filter column by its L1 norm, so a channel is a weighted average of bands
        # (same scale as a raw band) instead of a weighted sum ~20-65x larger; 'none' keeps the peak-normalised columns
        self.filter_norm = filter_norm
        # wavelength window (nm) the loader restricts the cube bands to; defaults to the EC sensor's own
        # measured range 400-800 nm. (400, 1000) keeps all 200 bands (today's behaviour, unchanged csv/stats file).
        self.band_range = (float(band_range[0]), float(band_range[1]))
        self.band_idx = band_indices(self.band_range)     # [n_bands] contiguous indices into the 200 cube bands
        self.n_bands = len(self.band_idx)
        self.stats_path = stats_path if stats_path is not None else default_stats_path(data_path, self.band_range)
        self.seed = seed
        # never keep the random module itself on the instance: a module is not picklable, which breaks
        # DataLoader workers with the spawn/forkserver start methods; see the rng property
        self._rng = random.Random(seed) if seed is not None else None
        self._rng_key = None

        assert self.split in ['train', 'test'], "split must be 'train' or 'test'"
        assert self.norm in ['none', 'p99', 'p99z'], "norm must be 'none', 'p99' or 'p99z'"
        assert self.filter_norm in ['none', 'l1'], "filter_norm must be 'none' or 'l1'"
        assert os.path.exists(self.data_path), f"Data path {self.data_path} does not exist"
        assert os.path.exists(self.filter_path), f"Filter file {self.filter_path} does not exist"

        # Load data path: <split>/hyperspectral/<id>.mat, <split>/GT/<id>.png, <split>/intensity map/*.csv
        self.hsi_path = os.path.join(self.data_path, self.split, 'hyperspectral')
        self.gt_path = os.path.join(self.data_path, self.split, 'GT')
        self.intensity_path = os.path.join(self.data_path, self.split, 'intensity map')
        assert os.path.isdir(self.hsi_path), f"{self.hsi_path} does not exist"
        assert os.path.isdir(self.gt_path), f"{self.gt_path} does not exist"

        names = [os.path.splitext(f)[0] for f in os.listdir(self.hsi_path) if f.endswith('.mat')]
        assert all(n.isdigit() for n in names), \
            f"non-numeric .mat names in {self.hsi_path}: {[n for n in names if not n.isdigit()]}"
        self.img_name = sorted(names, key=int)
        assert len(self.img_name) > 0, f"no .mat files found in {self.hsi_path}"
        for name in self.img_name:
            assert os.path.exists(os.path.join(self.gt_path, f'{name}.png')), f"GT for sample {name} not found in {self.gt_path}"

        # self.wavelens is shape [n_bands], the cube band centres inside band_range (uniform 3.015 nm step)
        self.wavelens = WAVELENS_200[self.band_idx].copy()

        # image size from the first GT; the cube layout is verified against it
        self.H, self.W = self.load_gt(self.img_name[0]).shape
        self.check_cube_layout(self.img_name[0])
        if self.split == 'train' and self.crop_size > 0:
            assert self.crop_size <= min(self.H, self.W), f"crop_size {self.crop_size} exceeds image size ({self.H}, {self.W})"

        print(f"Loading sensor response from {self.filter_path}...")
        self.load_sensor_response()

        self.scale = self.load_intensity_scale() if self.norm in ['p99', 'p99z'] else None

        self.in_channels = self.sensor_R_matrix.shape[1] if self.use_filter else self.n_bands

        # per-channel mean/std [C] for norm='p99z' (None otherwise), applied after the filter projection
        self.channel_mean, self.channel_std = self.load_channel_stats() if self.norm == 'p99z' else (None, None)

    def __len__(self):
        return len(self.img_name)

    @property
    def rng(self):
        '''
        Random stream used for crop sampling.
        seed is None (training default): the random module, which PyTorch re-seeds in every DataLoader
        worker at every epoch, so crops differ across workers and epochs.
        seed given (reproducible tests): a seeded random.Random in-process; inside a DataLoader worker a
        fresh stream is derived from (seed, worker seed) so each worker and each epoch still gets
        different crops instead of replaying the same ones.
        '''
        if self._rng is None:
            return random
        info = torch.utils.data.get_worker_info()
        if info is not None and self._rng_key != info.seed:
            self._rng = random.Random(f"{self.seed}:{info.seed}")
            self._rng_key = info.seed
        return self._rng

    def load_sensor_response(self):
        '''
        Load the EC detector response [401 wavelengths, 351 voltages], resample it onto the cube band
        centres (zero outside 400-800 nm) and keep the selected voltage columns.
        Result: self.sensor_R_matrix [C, N], C = 200 bands, N = number of selected voltages.
        The columns are already peak-normalized to |max| = 1 and contain negative lobes, so unlike the
        previous project there is NO per-column min-max normalization here (it would destroy the signs).
        '''
        self.sensor_wavelens, self.voltages, R_raw = load_ec_filter(self.filter_path)   # [C_s], [N_all], [C_s, N_all]
        R_aligned, self.valid_band_mask = align_filter_to_wavelens(self.sensor_wavelens, R_raw, self.wavelens)  # [C, N_all]
        self.selected_indices = select_filter_channels(R_aligned, self.voltages, num_filters=self.num_filters,
                                                       mode=self.filter_select, filter_voltages=self.filter_voltages)
        self.selected_voltages = self.voltages[self.selected_indices]  # [N]
        self.sensor_R_matrix = R_aligned[:, self.selected_indices].astype(np.float32)  # [C, N]
        # per-channel gain: with norm='p99' band values are ~1 at bright pixels, so a filter channel is
        # ~filter_gain times larger (about 20-65 on the real EC filter); the training script can divide by it
        self.filter_gain = np.abs(self.sensor_R_matrix).sum(axis=0)  # [N], gain of the peak-normalised columns
        if self.filter_norm == 'l1':
            # unit L1 norm per column: each channel becomes a weighted average of the bands, so filter-mode
            # inputs sit on the same scale as raw bands and the 3x gain difference between voltages (an
            # artifact of the file's peak normalisation, not physics) disappears
            self.sensor_R_matrix = (self.sensor_R_matrix / self.filter_gain).astype(np.float32)  # [C, N]
        print(f"Sensor response aligned: {self.sensor_R_matrix.shape[0]} bands x {self.sensor_R_matrix.shape[1]} channels, "
              f"{int(self.valid_band_mask.sum())} bands inside the sensor range, filter_norm={self.filter_norm}, "
              f"voltages {np.round(self.selected_voltages, 2).tolist()}")

    def load_intensity_scale(self):
        '''
        Per-sample scale = p99 of the band sum inside the window / n_bands. For the full 400-1000 nm window this is
        intensity_p99_valid / 200 from the dataset's csv; for a narrower window the 99th percentile of the windowed
        band sum is computed once per sample from the cubes and cached next to the original csv.
        '''
        if self.n_bands == N_BANDS:
            csv_path = os.path.join(self.intensity_path, 'intensity_p99_summary.csv')
            assert os.path.exists(csv_path), f"{csv_path} not found (needed for norm='p99')"
        else:
            csv_path = window_p99_csv_path(self.data_path, self.split, self.band_range)
            if not os.path.exists(csv_path):
                compute_window_p99(self.data_path, self.split, self.band_range,
                                   num_workers=0 if len(self.img_name) < 8 else min(8, os.cpu_count() or 1))
        scale = {}
        with open(csv_path, newline='') as f:
            for row in csv.DictReader(f):
                scale[row['sample_id']] = float(row['intensity_p99_valid']) / self.n_bands
        for name in self.img_name:
            assert name in scale, f"sample {name} missing from {csv_path}"
            assert scale[name] > 0, f"non-positive intensity p99 for sample {name}"
        return scale

    def load_channel_stats(self):
        '''
        Per-channel mean/std for norm='p99z' from the training band statistics (mean [n_bands], cov
        [n_bands, n_bands] of the p99-normalised bands inside band_range; built from the train split on first
        use, see band_stats.compute_band_stats).
        Standardisation is applied AFTER the filter projection - a real detector integrates over wavelength
        before readout, so per-band means cannot be subtracted first - and the channel statistics follow from
        the band statistics by linearity: mean_y = R^T mu, var_y = diag(R^T Sigma R).
        Returns (channel_mean [C], channel_std [C]) as float32, C = N filter channels or n_bands raw bands.
        '''
        if not os.path.exists(self.stats_path):
            crop_size = STATS_CROP_SIZE if min(self.H, self.W) >= STATS_CROP_SIZE else 0  # tiny cubes: full frames
            num_workers = 0 if len(self) < 8 else min(8, os.cpu_count() or 1)
            compute_band_stats(self.data_path, self.stats_path, crop_size=crop_size, num_workers=num_workers,
                               filter_path=self.filter_path, band_range=self.band_range)
        mu, cov = load_band_stats(self.stats_path, band_range=self.band_range)  # [n_bands], [n_bands, n_bands]
        if self.use_filter:
            R = self.sensor_R_matrix.astype(np.float64)  # [n_bands, N]
            mean = R.T @ mu  # [N]
            var = np.einsum('bn,bc,cn->n', R, cov, R)  # [N]
        else:
            mean, var = mu, np.diag(cov)  # [n_bands]
        std = np.sqrt(np.maximum(var, 0.0))
        assert (std > 0).all(), f"zero-variance channels in {self.stats_path}: {np.where(std == 0)[0].tolist()}"
        return mean.astype(np.float32), std.astype(np.float32)

    def load_gt(self, name):
        '''GT pngs are JPEG-compressed with 3 identical channels; foreground = channel 0 > 127. Returns bool [H, W].'''
        gt = np.array(Image.open(os.path.join(self.gt_path, f'{name}.png')))
        if gt.ndim == 3:
            gt = gt[:, :, 0]
        return gt > GT_THRESHOLD

    def check_cube_layout(self, name):
        '''MATLAB v7.3 stores the cube column-major, so h5py sees [B, W, H]; verify against the GT size.'''
        with h5py.File(os.path.join(self.hsi_path, f'{name}.mat'), 'r') as f:
            assert HYPERCUBE_KEY in f, f"{name}.mat has no '{HYPERCUBE_KEY}' variable, found {list(f.keys())}"
            shape = f[HYPERCUBE_KEY].shape
        assert shape == (N_BANDS, self.W, self.H), \
            f"cube {name} has shape {shape}, expected (bands, W, H) = ({N_BANDS}, {self.W}, {self.H})"

    def read_cube_block(self, name, h0, w0, ch, cw):
        '''
        Read a spatial window of the cube directly from the .mat (HDF5) file, restricted to the bands inside
        band_range (self.band_idx is contiguous, so this is a single hyperslab on the band axis).
        h5py layout is [B, W, H], so W is indexed with w0:w0+cw and H with h0:h0+ch.
        Returns float32 [n_bands, cw, ch]. The file is opened per call so DataLoader workers stay independent.
        '''
        b0, b1 = int(self.band_idx[0]), int(self.band_idx[-1]) + 1
        with h5py.File(os.path.join(self.hsi_path, f'{name}.mat'), 'r') as f:
            blk = f[HYPERCUBE_KEY][b0:b1, w0:w0 + cw, h0:h0 + ch]
        blk = np.asarray(blk, dtype=np.float32)
        assert blk.shape == (self.n_bands, cw, ch), \
            f"cube {name}: window (h0={h0}, w0={w0}, ch={ch}, cw={cw}) returned {blk.shape}, expected ({self.n_bands}, {cw}, {ch}); is this cube smaller than the first sample?"
        return blk

    def crop_window(self, gt):
        '''
        Choose the crop (h0, w0, ch, cw).
        Train with crop_size > 0: with probability obj_crop_prob (and a non-empty mask) pick a random
        foreground pixel and place the crop uniformly among the positions that contain it; otherwise a
        uniform random crop. Test split or crop_size == 0: the full frame.
        gt: bool [H, W]
        '''
        H, W = gt.shape
        if self.split != 'train' or self.crop_size <= 0:
            return 0, 0, H, W
        cs = self.crop_size
        fg_h, fg_w = np.nonzero(gt)
        if len(fg_h) > 0 and self.rng.random() < self.obj_crop_prob:
            k = self.rng.randrange(len(fg_h))
            hs, ws = int(fg_h[k]), int(fg_w[k])
            # top-left must satisfy h0 <= hs <= h0 + cs - 1 and 0 <= h0 <= H - cs (same for w)
            h0 = self.rng.randint(max(0, hs - cs + 1), min(hs, H - cs))
            w0 = self.rng.randint(max(0, ws - cs + 1), min(ws, W - cs))
        else:
            h0 = self.rng.randint(0, H - cs)
            w0 = self.rng.randint(0, W - cs)
        return h0, w0, cs, cs

    def __getitem__(self, idx):
        name = self.img_name[idx]
        gt = self.load_gt(name)  # [H, W] bool
        h0, w0, ch, cw = self.crop_window(gt)

        blk = self.read_cube_block(name, h0, w0, ch, cw)  # [B, cw, ch] float32
        if self.norm in ['p99', 'p99z']:
            # positive per-sample scalar, so scaling before or after the filter is identical
            blk /= np.float32(self.scale[name])

        if self.use_filter:
            # simulated detector channels: y_n = sum_b R[b, n] * cube[b], done in the h5 layout -> [N, cw, ch]
            img = np.tensordot(self.sensor_R_matrix.T, blk, axes=(1, 0))
        else:
            img = blk  # raw bands [B, cw, ch]

        # only the last two axes are swapped: [C, cw, ch] -> [C, ch, cw]; never build [H, W, B] (6 s per crop)
        img = np.ascontiguousarray(img.transpose(0, 2, 1), dtype=np.float32)  # [C, ch, cw]
        if self.norm == 'p99z':
            # per-channel standardisation after the (simulated) readout
            img = (img - self.channel_mean[:, None, None]) / self.channel_std[:, None, None]  # [C, ch, cw]
        gt = gt[h0:h0 + ch, w0:w0 + cw].astype(np.float32)[None]  # [1, ch, cw]
        return img, gt, name


def window_p99_csv_path(data_path, split, band_range):
    lo, hi = int(round(band_range[0])), int(round(band_range[1]))
    return os.path.join(data_path, split, 'intensity map', f'intensity_p99_{lo}_{hi}.csv')


def compute_window_p99(data_path, split, band_range, num_workers=8):
    '''Write intensity_p99_<lo>_<hi>.csv (same columns as intensity_p99_summary.csv) from the windowed band sums.'''
    ds = HyperCOD_data(data_path, split=split, use_filter=False, norm='none', crop_size=0, band_range=band_range,
                       filter_norm='none')
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=image_collate_fn)
    rows = []
    print(f"Computing {band_range} nm p99 intensity for {len(ds)} {split} cubes...")
    for img, _, name in loader:                                   # img [1, n_bands, H, W]
        s = img[0].sum(dim=0).numpy()                             # [H, W] windowed band sum
        rows.append((name[0], s.shape[0], s.shape[1], ds.n_bands, s.size,
                     float(np.percentile(s, 99)), float(s.min()), float(s.mean()), float(s.max())))
    csv_path = window_p99_csv_path(data_path, split, band_range)
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['sample_id', 'mat_path', 'intensity_map_path', 'height', 'width', 'bands', 'n_valid',
                    'intensity_p99_valid', 'intensity_min_valid', 'intensity_mean_valid', 'intensity_max_valid'])
        for name, h, wd, b, n, p99, mn, me, mx in rows:
            w.writerow([name, '', '', h, wd, b, n, p99, mn, me, mx])
    print(f"Saved {csv_path}")
    return csv_path


def image_collate_fn(batch):
    img, gt, name = list(zip(*batch))
    # the dataset gives [C, H, W] / [1, H, W] numpy arrays; stack to [B, C, H, W] / [B, 1, H, W] float32 tensors
    img = torch.from_numpy(np.stack(img, axis=0)).to(dtype=torch.float32)
    gt = torch.from_numpy(np.stack(gt, axis=0)).to(dtype=torch.float32)
    return img, gt, list(name)


def add_dataset_args(parser):
    parser.add_argument('--data_path', type=str, default='/data2/chaoyi/HyperCOD/Raw data',
                        help='HyperCOD root containing train/ and test/')
    parser.add_argument('--filter_path', type=str, default=None,
                        help='EC sensor response .mat, default <data_path>/EC_filterV3.mat')
    parser.add_argument('--use_filter', action='store_true',
                        help='feed the model the simulated EC detector channels instead of the raw 200 bands '
                             '(CLI default off; the HyperCOD_data constructor defaults to use_filter=True)')
    parser.add_argument('--num_filters', type=int, default=30,
                        help='number of voltage channels for filter_select uniform / osp')
    parser.add_argument('--filter_select', type=str, default='uniform', choices=['uniform', 'osp', 'all', 'manual'],
                        help='how to choose the bias voltages')
    parser.add_argument('--filter_voltages', type=float, nargs='+', default=None,
                        help='explicit bias voltages in V for filter_select manual')
    parser.add_argument('--crop_size', type=int, default=512,
                        help='training crop size at native resolution, 0 means full frame')
    parser.add_argument('--obj_crop_prob', type=float, default=0.5,
                        help='probability that a training crop is placed to contain the object')
    parser.add_argument('--norm', type=str, default='p99z', choices=['none', 'p99', 'p99z'],
                        help='p99: per-sample scaling by intensity p99 / 200 from intensity_p99_summary.csv; '
                             'p99z: p99 followed by per-channel standardisation with training band statistics '
                             '(band_stats_train.npz, built on first use)')
    parser.add_argument('--band_range', type=float, nargs=2, default=[400.0, 800.0],
                        help='wavelength window in nm (default the EC sensor range 400-800; 400 1000 uses all 200 bands)')
    parser.add_argument('--filter_norm', type=str, default='l1', choices=['none', 'l1'],
                        help='l1: divide each filter column by its L1 norm so channels are weighted averages of bands; '
                             'none: keep the peak-normalised columns (channels ~20-65x larger, see HyperCOD_data.filter_gain)')
    parser.add_argument('--stats_path', type=str, default=None,
                        help='training band statistics .npz for norm p99z, default <data_path>/band_stats_train.npz')
    return parser


def build_dataset(args, split):
    return HyperCOD_data(data_path=args.data_path, split=split, use_filter=args.use_filter,
                         filter_path=args.filter_path, num_filters=args.num_filters,
                         filter_select=args.filter_select, filter_voltages=args.filter_voltages,
                         crop_size=args.crop_size, obj_crop_prob=args.obj_crop_prob, norm=args.norm,
                         band_range=tuple(args.band_range), filter_norm=args.filter_norm,
                         stats_path=args.stats_path)


if __name__ == '__main__':
    import time
    parser = add_dataset_args(argparse.ArgumentParser(description='HyperCOD dataset smoke test'))
    args = parser.parse_args()
    dataset = build_dataset(args, split='train')
    print(f"{len(dataset)} train samples, in_channels={dataset.in_channels}")
    t = time.time()
    img, gt, name = dataset[0]
    print(f"sample {name}: img {img.shape} {img.dtype} range [{img.min():.4f}, {img.max():.4f}], "
          f"gt {gt.shape} fg={int(gt.sum())} px, {time.time() - t:.2f}s")
