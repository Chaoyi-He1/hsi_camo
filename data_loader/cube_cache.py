import os
import argparse
import numpy as np
import torch

from data_loader.ec_filter import band_indices

CACHE_DIRNAME = 'cache_fp16'


def default_cache_dir(data_path):
    return os.path.join(data_path, CACHE_DIRNAME)


def cache_path(cache_dir, split, name):
    return os.path.join(cache_dir, split, f'{name}.npy')


def build_cube_cache(data_path, split, cache_dir=None, band_range=(400.0, 800.0), num_workers=8, ids=None, overwrite=False):
    '''
    Convert every <split>/hyperspectral/<id>.mat into <cache_dir>/<split>/<id>.npy holding the windowed raw cube
    as float16 [n_bands, H, W] (0.55 GB per frame for 133 bands), so full frames load in ~0.3 s instead of 9.5 s.
    Also writes the windowed p99 csv used by norm='p99'. Existing files are skipped unless overwrite=True.
    '''
    from data_loader.my_dataset import HyperCOD_data, image_collate_fn, compute_window_p99, window_p99_csv_path
    cache_dir = cache_dir if cache_dir is not None else default_cache_dir(data_path)
    os.makedirs(os.path.join(cache_dir, split), exist_ok=True)
    ds = HyperCOD_data(data_path, split=split, ids=ids, use_filter=False, norm='none', crop_size=0,
                       band_range=band_range, filter_norm='none', out_dtype='float16')
    todo = [i for i, name in enumerate(ds.img_name) if overwrite or not os.path.exists(cache_path(cache_dir, split, name))]
    print(f"Caching {len(todo)}/{len(ds)} {split} cubes to {cache_dir} ({ds.n_bands} bands, float16)...")
    written = []
    if todo:
        loader = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, todo), batch_size=1, shuffle=False,
                                             num_workers=num_workers, collate_fn=image_collate_fn)
        for k, (img, _, name) in enumerate(loader):                       # img [1, n_bands, H, W] fp16
            out = cache_path(cache_dir, split, name[0])
            np.save(out, img[0].numpy().astype(np.float16))
            written.append(out)
            if (k + 1) % 25 == 0 or k + 1 == len(todo):
                print(f"  {k + 1}/{len(todo)} cubes")
    if not os.path.exists(window_p99_csv_path(data_path, split, band_range)) and len(band_indices(band_range)) < 200:
        compute_window_p99(data_path, split, band_range, num_workers=num_workers)
    return written


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build the fp16 full-frame cube cache')
    parser.add_argument('--data_path', type=str, default='/data2/chaoyi/HyperCOD/Raw data')
    parser.add_argument('--split', type=str, default='all', choices=['train', 'test', 'all'])
    parser.add_argument('--cache_dir', type=str, default=None)
    parser.add_argument('--band_range', type=float, nargs=2, default=[400.0, 800.0])
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--ids', type=str, nargs='*', default=None)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    for split in (['train', 'test'] if args.split == 'all' else [args.split]):
        build_cube_cache(args.data_path, split, args.cache_dir, tuple(args.band_range), args.num_workers, args.ids, args.overwrite)
