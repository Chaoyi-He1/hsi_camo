import os
import argparse
import numpy as np
import torch

from data_loader.ec_filter import N_BANDS, band_indices

STATS_FILENAME = 'band_stats_train.npz'   # saved next to the data, like data_range.npz in the previous project
STATS_CROP_SIZE = 512                      # one crop of this size per training cube (full frame if the cube is smaller)
RGB_STATS_FILENAME = 'rgb_stats_train.npz'  # the RGB baseline's statistics (compute_rgb_stats), next to the band statistics
RGB_STATS_NORM = 'rgb255'                   # what the statistics are of: uint8 / 255 values in R, G, B order, nothing else
RGB_STATS_SEED = 0                          # the crop stream of the one crop per frame


def default_stats_path(data_path, band_range=(400.0, 800.0)):
    '''band_stats_train.npz for the full 400-1000 nm window (unchanged from before band windows existed);
    band_stats_train_<lo>_<hi>.npz for a narrower window, so different windows never collide on disk.'''
    if len(band_indices(band_range)) == N_BANDS:
        return os.path.join(data_path, STATS_FILENAME)
    lo, hi = int(round(band_range[0])), int(round(band_range[1]))
    return os.path.join(data_path, f'band_stats_train_{lo}_{hi}.npz')


def compute_band_stats(data_path, stats_path, crop_size=STATS_CROP_SIZE, num_workers=8, seed=0, filter_path=None,
                       band_range=(400.0, 800.0)):
    '''
    Mean [n_bands] and covariance [n_bands, n_bands] of the p99-normalised raw band values inside band_range,
    over the TRAIN split, accumulated in float64 from one uniform random crop per cube (crop_size == 0: the
    full frame).
    Channel statistics for any filter selection follow by linearity (mean_y = R^T mean, var_y = diag(R^T cov R)),
    so the file never needs recomputing when the voltages change.
    Saves mean, cov, n_pixels, n_samples, crop_size, norm, wavelens and band_range to stats_path and returns
    (mean, cov).
    '''
    from data_loader.my_dataset import HyperCOD_data, image_collate_fn   # local import: my_dataset imports this module
    dataset = HyperCOD_data(data_path, split='train', use_filter=False, filter_path=filter_path,
                            crop_size=crop_size, obj_crop_prob=0.0, norm='p99', seed=seed, band_range=band_range)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers,
                                         collate_fn=image_collate_fn)
    n_bands = dataset.n_bands
    s1 = np.zeros(n_bands, dtype=np.float64)              # sum of x       [n_bands]
    s2 = np.zeros((n_bands, n_bands), dtype=np.float64)   # sum of x x^T   [n_bands, n_bands]
    n = 0
    print(f"Computing band statistics from {len(dataset)} training cubes (crop_size={crop_size})...")
    for i, (img, _, _) in enumerate(loader):
        x = img[0].reshape(n_bands, -1).to(torch.float64).numpy()   # [n_bands, P] p99-normalised band values
        s1 += x.sum(axis=1)
        s2 += x @ x.T
        n += x.shape[1]
        if (i + 1) % 25 == 0 or i + 1 == len(dataset):
            print(f"  {i + 1}/{len(dataset)} cubes, {n} pixels")
    mean = s1 / n                                          # [n_bands]
    cov = s2 / n - np.outer(mean, mean)                    # [n_bands, n_bands], population covariance
    cov = 0.5 * (cov + cov.T)                              # exact symmetry against float round-off
    os.makedirs(os.path.dirname(os.path.abspath(stats_path)), exist_ok=True)
    np.savez(stats_path, mean=mean, cov=cov, n_pixels=n, n_samples=len(dataset), crop_size=crop_size,
             norm='p99', wavelens=dataset.wavelens, band_range=np.array(band_range))
    print(f"Band statistics saved to {stats_path}")
    return mean, cov


def default_rgb_stats_path(data_path):
    '''rgb_stats_train.npz next to the data, like band_stats_train.npz.'''
    return os.path.join(data_path, RGB_STATS_FILENAME)


def compute_rgb_stats(data_path, stats_path, crop_size=STATS_CROP_SIZE, seed=RGB_STATS_SEED, filter_path=None):
    '''
    Mean [3] and covariance [3, 3] of the dataset's own RGB frames (<split>/RGB/<id>.jpg, uint8 / 255, channels in R, G, B
    order) over the TRAIN split, accumulated in float64 from one uniform random crop per frame (crop_size == 0: the full
    frame), the same convention as compute_band_stats: it covers every id of the train split directory (all 279 frames, so the
    detector's 251 training frames AND its 28 held-out validation frames, not the detector's own training ids), so one file
    serves every run, and the crop of a frame is drawn with obj_crop_prob=0 from a stream seeded with `seed`, frame by frame in
    id order (in process, so the file is reproducible). The RGB baseline's FilterBank standardises with these statistics
    (models.filter_bank.build_filter_bank).
    Saves mean, cov, n_pixels, n_samples, crop_size, seed and norm to stats_path and returns (mean, cov).
    '''
    from data_loader.my_dataset import HyperCOD_data   # local import: my_dataset imports this module
    dataset = HyperCOD_data(data_path, split='train', filter_path=filter_path, crop_size=crop_size, obj_crop_prob=0.0, seed=seed,
                            rgb_images=True)
    s1 = np.zeros(3, dtype=np.float64)           # sum of x       [3]
    s2 = np.zeros((3, 3), dtype=np.float64)      # sum of x x^T   [3, 3]
    n = 0
    print(f"Computing RGB statistics from {len(dataset)} training frames (crop_size={crop_size})...")
    for i in range(len(dataset)):
        img, _, _ = dataset[i]                                  # [3, ch, cw] float32, /255
        x = img.reshape(3, -1).astype(np.float64)               # [3, P]
        s1 += x.sum(axis=1)
        s2 += x @ x.T
        n += x.shape[1]
        if (i + 1) % 25 == 0 or i + 1 == len(dataset):
            print(f"  {i + 1}/{len(dataset)} frames, {n} pixels")
    mean = s1 / n                                          # [3]
    cov = s2 / n - np.outer(mean, mean)                    # [3, 3], population covariance
    cov = 0.5 * (cov + cov.T)                              # exact symmetry against float round-off
    os.makedirs(os.path.dirname(os.path.abspath(stats_path)), exist_ok=True)
    np.savez(stats_path, mean=mean, cov=cov, n_pixels=n, n_samples=len(dataset), crop_size=crop_size, seed=seed, norm=RGB_STATS_NORM)
    print(f"RGB statistics saved to {stats_path}")
    return mean, cov


def load_rgb_stats(stats_path, crop_size=None, seed=None, n_samples=None):
    '''
    Returns (mean [3], cov [3, 3]) as float64 from a file written by compute_rgb_stats. The file's metadata must match what the
    caller expects: norm always, and crop_size / seed / n_samples when given - a file built from other crops or from a
    different set of frames must not silently be reused.
    '''
    st = np.load(stats_path)
    mean = np.asarray(st['mean'], dtype=np.float64)
    cov = np.asarray(st['cov'], dtype=np.float64)
    assert mean.shape == (3,), f"{stats_path}: mean has shape {mean.shape}, expected (3,)"
    assert cov.shape == (3, 3), f"{stats_path}: cov has shape {cov.shape}, expected (3, 3)"
    assert str(st['norm']) == RGB_STATS_NORM, f"{stats_path}: norm {str(st['norm'])!r} does not match {RGB_STATS_NORM!r}"
    for key, want in (('crop_size', crop_size), ('seed', seed), ('n_samples', n_samples)):
        if want is not None:
            assert int(st[key]) == int(want), f"{stats_path}: {key} {int(st[key])} does not match requested {int(want)}"
    return mean, cov


def load_band_stats(stats_path, band_range=None):
    '''
    Returns (mean [n_bands], cov [n_bands, n_bands]) as float64 from a file written by compute_band_stats.
    The expected n_bands is derived from the file's own band_range (files saved before band windows existed
    have no band_range key and are treated as the full (400, 1000) window). If band_range is given, it must
    match the file's band_range - a stats file built for one window must not silently be reused for another.
    '''
    st = np.load(stats_path)
    mean = np.asarray(st['mean'], dtype=np.float64)
    cov = np.asarray(st['cov'], dtype=np.float64)
    file_range = tuple(st['band_range'].tolist()) if 'band_range' in st else (400.0, 1000.0)
    if band_range is not None:
        assert np.allclose(file_range, band_range), \
            f"{stats_path}: band_range {file_range} does not match requested {tuple(band_range)}"
    n_bands = len(band_indices(file_range))
    assert mean.shape == (n_bands,), f"{stats_path}: mean has shape {mean.shape}, expected ({n_bands},)"
    assert cov.shape == (n_bands, n_bands), f"{stats_path}: cov has shape {cov.shape}, expected ({n_bands}, {n_bands})"
    return mean, cov


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute HyperCOD training band statistics for norm=p99z')
    parser.add_argument('--data_path', type=str, default='/data2/chaoyi/HyperCOD/Raw data')
    parser.add_argument('--stats_path', type=str, default=None, help='default <data_path>/band_stats_train.npz')
    parser.add_argument('--crop_size', type=int, default=STATS_CROP_SIZE, help='0 means full frames')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--band_range', type=float, nargs=2, default=[400.0, 800.0],
                        help='wavelength window in nm (default the EC sensor range 400-800; 400 1000 uses all 200 bands)')
    args = parser.parse_args()
    band_range = tuple(args.band_range)
    compute_band_stats(args.data_path, args.stats_path or default_stats_path(args.data_path, band_range),
                       crop_size=args.crop_size, num_workers=args.num_workers, seed=args.seed,
                       band_range=band_range)
