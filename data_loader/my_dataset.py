import torch.utils.data as Dataset
import os
import csv
import random
import argparse
import numpy as np
import h5py
import torch
from PIL import Image

from data_loader.ec_filter import (N_BANDS, WAVELENS_200, load_ec_filter, align_filter_to_wavelens,
                                   select_filter_channels)

GT_THRESHOLD = 127            # GT pngs are JPEG-compressed with 3 identical channels; foreground = channel 0 > 127
HYPERCUBE_KEY = 'hypercube'   # variable name inside the MATLAB v7.3 (HDF5) .mat cubes


class HyperCOD_data(Dataset.Dataset):
    def __init__(self, data_path, split='train', use_filter=True, filter_path=None, num_filters=30,
                 filter_select='uniform', filter_voltages=None, crop_size=512, obj_crop_prob=0.5,
                 norm='p99', seed=None):
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
        self.norm = norm  # 'none' or 'p99'
        # the random module is re-seeded per DataLoader worker by PyTorch; a fixed seed is for reproducible tests
        self.rng = random.Random(seed) if seed is not None else random

        assert self.split in ['train', 'test'], "split must be 'train' or 'test'"
        assert self.norm in ['none', 'p99'], "norm must be 'none' or 'p99'"
        assert os.path.exists(self.data_path), f"Data path {self.data_path} does not exist"
        assert os.path.exists(self.filter_path), f"Filter file {self.filter_path} does not exist"

        # Load data path: <split>/hyperspectral/<id>.mat, <split>/GT/<id>.png, <split>/intensity map/*.csv
        self.hsi_path = os.path.join(self.data_path, self.split, 'hyperspectral')
        self.gt_path = os.path.join(self.data_path, self.split, 'GT')
        self.intensity_path = os.path.join(self.data_path, self.split, 'intensity map')
        assert os.path.isdir(self.hsi_path), f"{self.hsi_path} does not exist"
        assert os.path.isdir(self.gt_path), f"{self.gt_path} does not exist"

        self.img_name = sorted([os.path.splitext(f)[0] for f in os.listdir(self.hsi_path) if f.endswith('.mat')], key=int)
        assert len(self.img_name) > 0, f"no .mat files found in {self.hsi_path}"
        for name in self.img_name:
            assert os.path.exists(os.path.join(self.gt_path, f'{name}.png')), f"GT for sample {name} not found in {self.gt_path}"

        # self.wavelens is shape [200], 400 nm to 1000 nm, uniform 3.015 nm step
        self.wavelens = WAVELENS_200.copy()

        # image size from the first GT; the cube layout is verified against it
        self.H, self.W = self.load_gt(self.img_name[0]).shape
        self.check_cube_layout(self.img_name[0])
        if self.split == 'train' and self.crop_size > 0:
            assert self.crop_size <= min(self.H, self.W), f"crop_size {self.crop_size} exceeds image size ({self.H}, {self.W})"

        print(f"Loading sensor response from {self.filter_path}...")
        self.load_sensor_response()

        self.scale = self.load_intensity_scale() if self.norm == 'p99' else None

        self.in_channels = self.sensor_R_matrix.shape[1] if self.use_filter else N_BANDS

    def __len__(self):
        return len(self.img_name)

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
        print(f"Sensor response aligned: {self.sensor_R_matrix.shape[0]} bands x {self.sensor_R_matrix.shape[1]} channels, "
              f"{int(self.valid_band_mask.sum())} bands inside the sensor range, "
              f"voltages {np.round(self.selected_voltages, 2).tolist()}")

    def load_intensity_scale(self):
        '''
        Per-sample scale = intensity_p99_valid / N_BANDS from intensity_p99_summary.csv, where the
        intensity map is the sum of the 200 bands. Dividing the cube by it puts bright-pixel band values
        near 1 and removes the ~40x scene-to-scene brightness spread.
        '''
        csv_path = os.path.join(self.intensity_path, 'intensity_p99_summary.csv')
        assert os.path.exists(csv_path), f"{csv_path} not found (needed for norm='p99')"
        scale = {}
        with open(csv_path, newline='') as f:
            for row in csv.DictReader(f):
                scale[row['sample_id']] = float(row['intensity_p99_valid']) / N_BANDS
        for name in self.img_name:
            assert name in scale, f"sample {name} missing from {csv_path}"
            assert scale[name] > 0, f"non-positive intensity p99 for sample {name}"
        return scale

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
        Read a spatial window of the cube directly from the .mat (HDF5) file.
        h5py layout is [B, W, H], so W is indexed with w0:w0+cw and H with h0:h0+ch.
        Returns float32 [B, cw, ch]. The file is opened per call so DataLoader workers stay independent.
        '''
        with h5py.File(os.path.join(self.hsi_path, f'{name}.mat'), 'r') as f:
            blk = f[HYPERCUBE_KEY][:, w0:w0 + cw, h0:h0 + ch]
        return np.asarray(blk, dtype=np.float32)

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
        if self.norm == 'p99':
            # positive per-sample scalar, so scaling before or after the filter is identical
            blk /= np.float32(self.scale[name])

        if self.use_filter:
            # simulated detector channels: y_n = sum_b R[b, n] * cube[b], done in the h5 layout -> [N, cw, ch]
            img = np.tensordot(self.sensor_R_matrix.T, blk, axes=(1, 0))
        else:
            img = blk  # raw bands [B, cw, ch]

        # only the last two axes are swapped: [C, cw, ch] -> [C, ch, cw]; never build [H, W, B] (6 s per crop)
        img = np.ascontiguousarray(img.transpose(0, 2, 1), dtype=np.float32)  # [C, ch, cw]
        gt = gt[h0:h0 + ch, w0:w0 + cw].astype(np.float32)[None]  # [1, ch, cw]
        return img, gt, name


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
                        help='feed the model the simulated EC detector channels instead of the raw 200 bands')
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
    parser.add_argument('--norm', type=str, default='p99', choices=['none', 'p99'],
                        help='per-sample scaling by intensity p99 / 200 from intensity_p99_summary.csv')
    return parser


def build_dataset(args, split):
    return HyperCOD_data(data_path=args.data_path, split=split, use_filter=args.use_filter,
                         filter_path=args.filter_path, num_filters=args.num_filters,
                         filter_select=args.filter_select, filter_voltages=args.filter_voltages,
                         crop_size=args.crop_size, obj_crop_prob=args.obj_crop_prob, norm=args.norm)


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
