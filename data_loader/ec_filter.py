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


def candidate_indices(voltages, dead_zone=FILTER_DEAD_ZONE_V):
    '''Indices of the voltages outside the polarity-flip dead zone (inclusive bounds, 1e-6 V tolerance).'''
    lo, hi = dead_zone
    return np.where((voltages < lo - 1e-6) | (voltages > hi + 1e-6))[0]


def osp(X, num_channels):
    ''' Orthogonal Subspace Projection (OSP) algorithm to
    select a subset of channels from the sensor response matrix X.
    (Copied from HSI/Diffu/data_loader/my_dataset.py.)
    Args:
        X: Sensor response matrix of shape (C, N), where C is the number of channels, N is the number of sensor responses
        num_channels: Number of channels to select

    Returns:
        Selected channels from the sensor response matrix and their indices
    '''
    C, N = X.shape

    X_min, X_max = X.min(), X.max()
    X = (X - X_min) / (X_max - X_min)  # Normalize to [0, 1]

    selected_indices = []
    residual = X.copy().astype(np.float64)  # Use float64 for better numerical stability

    for _ in range(num_channels):
        # Compute the norms of each sensor response in the residual
        norms = np.linalg.norm(residual, axis=0)  # [N]
        selected_index = int(np.argmax(norms))
        selected_indices.append(selected_index)

        # Get the selected sensor response
        selected_response = residual[:, selected_index].reshape(-1, 1)  # [C, 1]

        # Avoid division by zero - compute squared norm (||v||²)
        norm_squared = np.linalg.norm(selected_response) ** 2
        if norm_squared < 1e-12:
            break

        # Project out the selected response from all remaining columns
        projection_matrix = (selected_response @ selected_response.T) / norm_squared  # [C, C]
        residual = residual - projection_matrix @ residual

        # Set the selected column to zero to avoid selecting it again
        residual[:, selected_index] = 0

    return X[:, selected_indices], selected_indices


def select_filter_channels(R, voltages, num_filters=30, mode='uniform', filter_voltages=None,
                           dead_zone=FILTER_DEAD_ZONE_V):
    '''
    Choose which bias voltages (columns of R) become the model's input channels.
    Args:
        R: [C, N_all] aligned response
        voltages: [N_all]
        num_filters: used by 'uniform' and 'osp' only
        mode: 'uniform' - evenly spaced over the candidate voltages (as in the previous project)
              'osp'     - OSP selection among the candidates
              'all'     - every candidate voltage
              'manual'  - nearest voltage to each value in filter_voltages
    Returns:
        selected_indices: list[int] into the voltage axis
    '''
    cand = candidate_indices(voltages, dead_zone)
    assert len(cand) > 0, "no candidate voltages outside the dead zone"
    if mode == 'uniform':
        assert 1 <= num_filters <= len(cand), f"num_filters={num_filters} must be in [1, {len(cand)}]"
        pos = np.linspace(0, len(cand) - 1, num_filters).round().astype(int)
        return cand[pos].tolist()
    if mode == 'osp':
        assert 1 <= num_filters <= len(cand), f"num_filters={num_filters} must be in [1, {len(cand)}]"
        _, sel = osp(R[:, cand], num_filters)
        if len(sel) < num_filters:
            print(f"Warning: OSP stopped at {len(sel)} channels (residual exhausted), requested {num_filters}")
        return cand[np.asarray(sel, dtype=int)].tolist()
    if mode == 'all':
        return cand.tolist()
    if mode == 'manual':
        assert filter_voltages is not None and len(filter_voltages) > 0, \
            "filter_select='manual' requires filter_voltages"
        sel = []
        for v in filter_voltages:
            idx = int(np.argmin(np.abs(voltages - v)))
            if dead_zone[0] - 1e-6 <= voltages[idx] <= dead_zone[1] + 1e-6:
                print(f"Warning: requested voltage {v:+.2f} V maps to {voltages[idx]:+.2f} V inside the dead zone {dead_zone}")
            sel.append(idx)
        return sel
    raise ValueError(f"unknown filter_select mode '{mode}', expected uniform | osp | all | manual")
