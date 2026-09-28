import os
import io
import mmap
import errno
import argparse
import numpy as np
import torch

from data_loader.ec_filter import band_indices

CACHE_DIRNAME = 'cache_fp16'
DIRECT_BLOCK = 4096          # O_DIRECT alignment for address, file offset and length


def read_npy_direct(path):
    '''
    Read a whole .npy file with O_DIRECT into a page-aligned anonymous buffer; returns a writable array that owns
    its memory (C order). Why: a cached frame is 0.55 GB and a training epoch streams 139 GB of them; read through
    the page cache inside the training process (cache full, ~1 GB free) every DataLoader worker stalled in kernel
    memory reclaim (memory PSI ~70 %, disk idle at ~10 MB/s, 17 s/step). O_DIRECT leaves the page cache alone and
    the SATA SSD then delivers its ~380-420 MB/s to 4-6 workers. Falls back to np.load where O_DIRECT is not
    supported (tmpfs -> EINVAL).
    '''
    size = os.path.getsize(path)
    length = -(-size // DIRECT_BLOCK) * DIRECT_BLOCK      # aligned; the last read runs past EOF and comes back short
    try:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    except OSError as e:
        if e.errno == errno.EINVAL:
            return np.load(path)
        raise
    buf = mmap.mmap(-1, length)                           # anonymous mapping: page-aligned address
    view = memoryview(buf)
    try:
        got = 0
        while got < size:
            k = os.readv(fd, [view[got:length]])
            assert k > 0, f"O_DIRECT read of {path} stopped at {got}/{size} bytes"
            got += k
    except OSError as e:
        if e.errno == errno.EINVAL:                       # filesystem accepted the flag but not the read
            return np.load(path)
        raise
    finally:
        os.close(fd)
    header = io.BytesIO(view[:min(length, 4 * DIRECT_BLOCK)].tobytes())
    version = np.lib.format.read_magic(header)
    read_header = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
    shape, fortran_order, dtype = read_header(header)
    assert not fortran_order, f"{path}: Fortran-ordered .npy is not supported"
    return np.frombuffer(buf, dtype=dtype, count=int(np.prod(shape)), offset=header.tell()).reshape(shape)


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
