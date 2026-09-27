import os
import argparse
import numpy as np
import torch

from data_loader.ec_filter import N_BANDS

STATS_FILENAME = 'band_stats_train.npz'   # saved next to the data, like data_range.npz in the previous project
STATS_CROP_SIZE = 512                      # one crop of this size per training cube (full frame if the cube is smaller)


def default_stats_path(data_path):
    return os.path.join(data_path, STATS_FILENAME)


def compute_band_stats(data_path, stats_path, crop_size=STATS_CROP_SIZE, num_workers=8, seed=0, filter_path=None):
    '''
    Mean [200] and covariance [200, 200] of the p99-normalised raw band values over the TRAIN split,
    accumulated in float64 from one uniform random crop per cube (crop_size == 0: the full frame).
    Channel statistics for any filter selection follow by linearity (mean_y = R^T mean, var_y = diag(R^T cov R)),
    so the file never needs recomputing when the voltages change.
    Saves mean, cov, n_pixels, n_samples, crop_size, norm and wavelens to stats_path and returns (mean, cov).
    '''
    from data_loader.my_dataset import HyperCOD_data, image_collate_fn   # local import: my_dataset imports this module
    dataset = HyperCOD_data(data_path, split='train', use_filter=False, filter_path=filter_path,
                            crop_size=crop_size, obj_crop_prob=0.0, norm='p99', seed=seed)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers,
                                         collate_fn=image_collate_fn)
    s1 = np.zeros(N_BANDS, dtype=np.float64)              # sum of x       [200]
    s2 = np.zeros((N_BANDS, N_BANDS), dtype=np.float64)   # sum of x x^T   [200, 200]
    n = 0
    print(f"Computing band statistics from {len(dataset)} training cubes (crop_size={crop_size})...")
    for i, (img, _, _) in enumerate(loader):
        x = img[0].reshape(N_BANDS, -1).to(torch.float64).numpy()   # [200, P] p99-normalised band values
        s1 += x.sum(axis=1)
        s2 += x @ x.T
        n += x.shape[1]
        if (i + 1) % 25 == 0 or i + 1 == len(dataset):
            print(f"  {i + 1}/{len(dataset)} cubes, {n} pixels")
    mean = s1 / n                                          # [200]
    cov = s2 / n - np.outer(mean, mean)                    # [200, 200], population covariance
    cov = 0.5 * (cov + cov.T)                              # exact symmetry against float round-off
    os.makedirs(os.path.dirname(os.path.abspath(stats_path)), exist_ok=True)
    np.savez(stats_path, mean=mean, cov=cov, n_pixels=n, n_samples=len(dataset), crop_size=crop_size,
             norm='p99', wavelens=dataset.wavelens)
    print(f"Band statistics saved to {stats_path}")
    return mean, cov


def load_band_stats(stats_path):
    '''Returns (mean [200], cov [200, 200]) as float64 from a file written by compute_band_stats.'''
    st = np.load(stats_path)
    mean = np.asarray(st['mean'], dtype=np.float64)
    cov = np.asarray(st['cov'], dtype=np.float64)
    assert mean.shape == (N_BANDS,), f"{stats_path}: mean has shape {mean.shape}, expected ({N_BANDS},)"
    assert cov.shape == (N_BANDS, N_BANDS), f"{stats_path}: cov has shape {cov.shape}, expected ({N_BANDS}, {N_BANDS})"
    return mean, cov


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute HyperCOD training band statistics for norm=p99z')
    parser.add_argument('--data_path', type=str, default='/data2/chaoyi/HyperCOD/Raw data')
    parser.add_argument('--stats_path', type=str, default=None, help='default <data_path>/band_stats_train.npz')
    parser.add_argument('--crop_size', type=int, default=STATS_CROP_SIZE, help='0 means full frames')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    compute_band_stats(args.data_path, args.stats_path or default_stats_path(args.data_path),
                       crop_size=args.crop_size, num_workers=args.num_workers, seed=args.seed)
