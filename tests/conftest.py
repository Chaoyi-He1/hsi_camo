import os
import numpy as np
import h5py
import scipy.io as sio
import pytest
from PIL import Image

# tiny stand-in for the real 1680 x 1240 x 200 cubes
H, W, B = 48, 40, 200
OBJ_SLICE = (slice(10, 16), slice(20, 26))   # 6 x 6 foreground object in every GT


def write_sample(root, split, name, rng):
    # MATLAB v7.3 stores the cube column-major, so h5py sees [B, W, H]
    cube = (rng.random((B, W, H), dtype=np.float32) * 0.02).astype(np.float32)
    with h5py.File(os.path.join(root, split, 'hyperspectral', f'{name}.mat'), 'w') as f:
        f.create_dataset('hypercube', data=cube)
    gt = np.zeros((H, W), dtype=np.uint8)
    gt[OBJ_SLICE] = 255
    # real GTs are 3-channel; write 3 identical channels
    Image.fromarray(np.stack([gt] * 3, axis=-1)).save(os.path.join(root, split, 'GT', f'{name}.png'))
    p99 = float(np.percentile(cube.sum(axis=0), 99))   # intensity = sum over bands
    return cube, gt > 127, p99


@pytest.fixture
def synthetic_root(tmp_path):
    rng = np.random.default_rng(0)
    root = tmp_path / 'HyperCOD'
    info = {}
    for split, names in (('train', ['3', '10']), ('test', ['7'])):
        for sub in ('hyperspectral', 'GT', 'intensity map'):
            (root / split / sub).mkdir(parents=True)
        rows = []
        for name in names:
            cube, gt, p99 = write_sample(str(root), split, name, rng)
            info[(split, name)] = (cube, gt, p99)
            rows.append((name, p99))
        with open(root / split / 'intensity map' / 'intensity_p99_summary.csv', 'w') as f:
            f.write('sample_id,mat_path,intensity_map_path,height,width,bands,n_valid,'
                    'intensity_p99_valid,intensity_min_valid,intensity_mean_valid,intensity_max_valid\n')
            for name, p99 in rows:
                f.write(f'{name},x,y,{H},{W},{B},{H * W},{p99},0,0,0\n')

    # synthetic EC filter on the real grid sizes: a Gaussian whose centre sweeps 400 -> 800 nm
    # with voltage, negative below the 0.27 V polarity flip (like the real device)
    wl = np.arange(400, 801, dtype=np.float64)                  # [401]
    volt = np.round(np.arange(-1.0, 2.5001, 0.01), 2)           # [351]
    centre = 400.0 + (volt - volt.min()) / (volt.max() - volt.min()) * 400.0
    R = np.exp(-0.5 * ((wl[:, None] - centre[None, :]) / 40.0) ** 2)   # [401, 351], peak 1
    R[:, volt < 0.27] *= -1.0
    sio.savemat(str(root / 'EC_filterV3.mat'),
                {'Wavelength': wl[None].astype(np.uint16), 'Voltage': volt[None], 'responsivity': R})
    return root, info, (wl, volt, R)
