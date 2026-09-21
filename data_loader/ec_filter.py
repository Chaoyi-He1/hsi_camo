import numpy as np
import scipy.io as sio
from scipy.interpolate import interp1d

# HyperCOD cube band centres: 200 bands, 400 nm to 1000 nm, 3.015 nm step (Code/wl_200bands.npy)
N_BANDS = 200
WAVELENS_200 = np.linspace(400.0, 1000.0, N_BANDS)

# The EC detector's photocurrent flips sign at ~0.27-0.29 V and is ~0 there,
# so these bias voltages are excluded from automatic channel selection
FILTER_DEAD_ZONE_V = (0.25, 0.31)


def load_ec_filter(filter_path):
    '''
    Load the EC tunable detector response (EC_filterV3.mat, MATLAB v5).
    Returns:
        sensor_wavelens: [C_s] nm, 401 points from 400 to 800 nm
        voltages: [N_all] V, 351 points from -1.00 to 2.50 V
        R: [C_s, N_all] responsivity, each column peak-normalized to |max| = 1, may be negative
    '''
    mat = sio.loadmat(filter_path)
    for key in ('Wavelength', 'Voltage', 'responsivity'):
        assert key in mat, \
            f"{filter_path} has no variable '{key}', found {[k for k in mat if not k.startswith('__')]}"
    sensor_wavelens = np.asarray(mat['Wavelength'], dtype=np.float64).reshape(-1)
    voltages = np.asarray(mat['Voltage'], dtype=np.float64).reshape(-1)
    R = np.asarray(mat['responsivity'], dtype=np.float64)
    assert R.shape == (len(sensor_wavelens), len(voltages)), \
        f"responsivity shape {R.shape} does not match ({len(sensor_wavelens)} wavelengths, {len(voltages)} voltages)"
    return sensor_wavelens, voltages, R


def align_filter_to_wavelens(sensor_wavelens, R, wavelens):
    '''
    Linearly resample the sensor response onto the cube band centres.
    Bands outside the measured sensor range get ZERO response (the sensor is blind there);
    extrapolating the edge values would invent a response the device does not have.
    Args:
        sensor_wavelens: [C_s]
        R: [C_s, N]
        wavelens: [C]
    Returns:
        R_aligned: [C, N]
        valid_band_mask: [C] bool, True where the band lies inside the sensor range
    '''
    f = interp1d(sensor_wavelens, R, axis=0, kind='linear', bounds_error=False, fill_value=0.0)
    R_aligned = f(wavelens)
    valid_band_mask = (wavelens >= sensor_wavelens.min()) & (wavelens <= sensor_wavelens.max())
    return R_aligned, valid_band_mask
