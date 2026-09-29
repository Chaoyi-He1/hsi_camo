"""
Report data, part 1 (CPU): what separates a camouflaged object from its surroundings in the raw bands, and how much of
that survives the EC filter. Writes data_part1.json + images under img/.

Definitions used throughout (stated again in the report):
  x      raw spectrum of a pixel: the 133 cube bands of 400-800 nm, divided by the frame's p99 scale (exactly the model input)
  z      band-standardised raw spectrum (x - mu) / sigma, mu/sigma from the training band statistics
  y      EC responses R^T x (344 usable voltages, L1-normalised responsivity columns aligned to the cube bands)
  yz     standardised responses (y - R^T mu) / sqrt(diag(R^T Sigma R))  (what FilterBank feeds YOLO)
  object pixels   GT mask eroded by 2 px
  surround        pixels 3-60 px outside the mask (the local background a camouflaged object has to blend into)
  Fisher ratio    (m_obj - m_bg)^2 / (v_obj + v_bg) of one channel
  separability    Mahalanobis distance between object and surround means under the pooled covariance (+1e-3 ridge);
                  invariant to invertible linear maps, so it measures the information a representation keeps
"""
import os, sys, json, time
import numpy as np
import scipy.ndimage as ndi
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
WT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))                 # repo root
sys.path.insert(0, WT)
os.chdir(WT)
import main_det
from models.ec_yolo import pca_whitened_channels

OUT = os.environ.get('REPORT_OUT', os.path.join(HERE, 'build'))
IMG = os.path.join(OUT, 'img'); os.makedirs(IMG, exist_ok=True)
N_VAL = 12
rng = np.random.default_rng(0)
t0 = time.time()

args = main_det.get_args_parser().parse_args([]); main_det.load_cfg(args)
ds_tr, ds_val, ds_te = main_det.build_datasets(args)
R, rmean, rstd, volts = ds_tr.filter_bank_tensors()
R, rmean, rstd, volts = R.astype(np.float64), rmean.astype(np.float64), rstd.astype(np.float64), np.asarray(volts, np.float64)
mu, cov = ds_tr._band_stats(); mu, cov = np.asarray(mu, np.float64), np.asarray(cov, np.float64)
bstd = np.sqrt(np.diag(cov))
wl = np.asarray(ds_tr.wavelens, np.float64)
top = json.load(open(os.path.join(WT, 'weights/det_A/top_k_ep89.json')))
top_idx = [int(i) for i in top['indices']]
Rk, mk, sk, _ = pca_whitened_channels(R, rmean, rstd, cov, 11)
Rk, mk, sk = Rk.astype(np.float64), mk.astype(np.float64), sk.astype(np.float64)
print(f'loaded datasets + filter in {time.time() - t0:.0f}s: R {R.shape}, top-10 {np.round(volts[top_idx], 2).tolist()}', flush=True)

# ---------------------------------------------------------------- 1. the filter matrix itself
Rpk = R / np.abs(R).max(0, keepdims=True)                          # peak-normalised shapes, for display
U, s, Vt = np.linalg.svd(R, full_matrices=False)
energy = s ** 2 / np.sum(s ** 2)
res_R = 1.0 - np.cumsum(energy)
ev = np.sort(np.linalg.eigvalsh(cov))[::-1]; ev = np.maximum(ev, 0)
res_scene = 1.0 - np.cumsum(ev) / ev.sum()
fwhm = []
for j in range(R.shape[1]):
    c = np.abs(Rpk[:, j]); above = np.where(c >= 0.5)[0]; fwhm.append(float((above[-1] - above[0] + 1) * (wl[1] - wl[0])))
fwhm = np.array(fwhm)
cos_adj = np.sum(R[:, 1:] * R[:, :-1], 0) / (np.linalg.norm(R[:, 1:], axis=0) * np.linalg.norm(R[:, :-1], axis=0))
rank99 = int(np.searchsorted(np.cumsum(energy), 0.99) + 1); rank999 = int(np.searchsorted(np.cumsum(energy), 0.999) + 1)
scene99 = int(np.searchsorted(np.cumsum(ev) / ev.sum(), 0.99) + 1); scene999 = int(np.searchsorted(np.cumsum(ev) / ev.sum(), 0.999) + 1)
print(f'R effective rank {rank99}/{rank999}; scene covariance {scene99}/{scene999}; FWHM median {np.median(fwhm):.0f} nm; '
      f'adjacent cosine median {np.median(cos_adj):.5f}', flush=True)


def diverging_png(M, path, mid):
    """M in [-1, 1] -> blue (neg) / neutral mid / red (pos), rows already in display order."""
    neg = np.array([0x2a, 0x78, 0xd6], np.float64); pos = np.array([0xe3, 0x49, 0x48], np.float64); m = np.array(mid, np.float64)
    t = np.clip(M, -1, 1)[..., None]
    rgb = np.where(t < 0, m + (neg - m) * (-t), m + (pos - m) * t)
    Image.fromarray(rgb.round().astype(np.uint8)).save(path, optimize=True)


heat = Rpk[::-1, :]                                                # wavelength increasing upward
diverging_png(heat, os.path.join(IMG, 'R_heat_light.png'), (0xf0, 0xef, 0xec))
diverging_png(heat, os.path.join(IMG, 'R_heat_dark.png'), (0x38, 0x38, 0x35))

# ---------------------------------------------------------------- 2. object vs surround on validation frames
def fisher(a, b):
    return (a.mean(0) - b.mean(0)) ** 2 / (a.var(0) + b.var(0) + 1e-12)


def maha(a, b, ridge=1e-3):
    d = a.mean(0) - b.mean(0)
    P = 0.5 * (np.cov(a.T) + np.cov(b.T)); P = np.atleast_2d(P)
    P = P + ridge * np.trace(P) / len(d) * np.eye(len(d))
    w = np.linalg.solve(P, d)
    return float(np.sqrt(d @ w)), w


def stretch(v, lo=1, hi=99):
    a, b = np.percentile(v, [lo, hi]); return np.clip((v - a) / max(b - a, 1e-9), 0, 1)


def reps(X):
    """X [n,133] raw p99-scaled spectra -> dict of standardised representations."""
    z = (X - mu) / bstd
    y = X @ R; yz = (y - rmean) / rstd
    pca = (X @ Rk - mk) / sk
    t10 = yz[:, top_idx]
    return {'raw133': z, 'resp344': yz, 'pca11': pca, 'top10': t10}


frames = []
fisher_raw, fisher_resp, cohen_raw = [], [], []
for i in range(min(N_VAL, len(ds_val))):
    img, gt, name = ds_val[i]                                      # fp16 [133,H,W], gt [1,H,W]
    m = gt[0] > 0.5
    if m.sum() < 200:
        print(f'skip frame {name}: {int(m.sum())} object px', flush=True); continue
    d_in = ndi.distance_transform_edt(m); d_out = ndi.distance_transform_edt(~m)
    obj = np.flatnonzero((d_in > 2).ravel()) if (d_in > 2).sum() >= 100 else np.flatnonzero(m.ravel())
    ring = np.flatnonzero(((d_out > 3) & (d_out <= 60)).ravel())
    bg = np.flatnonzero((~m).ravel())
    so = obj if len(obj) <= 8000 else rng.choice(obj, 8000, replace=False)
    sr = ring if len(ring) <= 20000 else rng.choice(ring, 20000, replace=False)
    sb = rng.choice(bg, min(len(bg), 20000), replace=False)
    flat = img.reshape(img.shape[0], -1)
    Xo = np.asarray(flat[:, so], np.float64).T; Xr = np.asarray(flat[:, sr], np.float64).T; Xb = np.asarray(flat[:, sb], np.float64).T
    Ro, Rr, Rb = reps(Xo), reps(Xr), reps(Xb)
    f_raw = fisher(Ro['raw133'], Rr['raw133']); f_resp = fisher(Ro['resp344'], Rr['resp344'])
    fisher_raw.append(f_raw); fisher_resp.append(f_resp)
    cohen_raw.append((Ro['raw133'].mean(0) - Rr['raw133'].mean(0)) / np.sqrt(0.5 * (Ro['raw133'].var(0) + Rr['raw133'].var(0)) + 1e-12))
    sep_ring, sep_bg, lda = {}, {}, {}
    for k in Ro:
        sep_ring[k], lda[k] = maha(Ro[k], Rr[k]); sep_bg[k], _ = maha(Ro[k], Rb[k])
    rec = dict(name=str(name), obj_px=int(m.sum()), best_band_nm=float(wl[f_raw.argmax()]), best_band_fisher=float(f_raw.max()),
               best_volt=float(volts[f_resp.argmax()]), best_volt_fisher=float(f_resp.max()),
               sep_ring={k: round(v, 4) for k, v in sep_ring.items()}, sep_bg={k: round(v, 4) for k, v in sep_bg.items()},
               mean_obj=np.round(Xo.mean(0), 5).tolist(), mean_ring=np.round(Xr.mean(0), 5).tolist(),
               fisher_raw=np.round(f_raw, 4).tolist(), fisher_resp=np.round(f_resp, 4).tolist())

    # ---- image panel for this frame: RGB composite, best band, best voltage, LDA maps (all linear in x: a^T x + b)
    H, W = m.shape
    ys, xs = np.nonzero(m); cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    side = int(min(max(2.4 * max(np.ptp(ys), np.ptp(xs)), 300), min(H, W)))
    y0 = int(np.clip(cy - side / 2, 0, H - side)); x0 = int(np.clip(cx - side / 2, 0, W - side))
    crop = (slice(y0, y0 + side), slice(x0, x0 + side))
    xc = np.asarray(img[:, crop[0], crop[1]], np.float32)        # [133, side, side]
    Xc = xc.reshape(133, -1).T.astype(np.float64)
    vecs = {
        'band': np.eye(133)[f_raw.argmax()] / bstd[f_raw.argmax()],
        'volt': R[:, f_resp.argmax()] / rstd[f_resp.argmax()],
        'lda_raw': lda['raw133'] / bstd,
        'lda_resp': R @ (lda['resp344'] / rstd),
        'lda_pca': Rk @ (lda['pca11'] / sk),
        'lda_top10': R[:, top_idx] @ (lda['top10'] / rstd[top_idx]),
    }
    contrast = {}
    mo = m[crop].ravel(); mr = ((d_out[crop] > 3) & (d_out[crop] <= 60)).ravel()
    for key, a in vecs.items():
        v = Xc @ a
        if v[mo].mean() < v[mr].mean():
            v = -v                                                 # object bright in every map
        contrast[key] = round(float(fisher(v[mo][:, None], v[mr][:, None])[0]), 4)
        g = (stretch(v).reshape(side, side) * 255).round().astype(np.uint8)
        Image.fromarray(g).resize((360, 360), Image.LANCZOS).save(os.path.join(IMG, f'val{name}_{key}.jpg'), quality=86)
    rgb_idx = [int(np.abs(wl - t).argmin()) for t in (640, 550, 460)]
    full = np.stack([np.asarray(img[b], np.float32) for b in rgb_idx], -1)
    full = np.clip(full / np.percentile(full, 99.5, axis=(0, 1), keepdims=True), 0, 1) ** (1 / 2.2)
    Image.fromarray((full * 255).round().astype(np.uint8)).resize((W // 2, H // 2), Image.LANCZOS).save(os.path.join(IMG, f'val{name}_rgb.jpg'), quality=84)
    rc = (full[crop] * 255).round().astype(np.uint8)
    Image.fromarray(rc).resize((360, 360), Image.LANCZOS).save(os.path.join(IMG, f'val{name}_rgbcrop.jpg'), quality=86)
    edge = m[crop] & ~ndi.binary_erosion(m[crop], iterations=max(1, side // 180))
    ov = np.zeros((side, side, 4), np.uint8); ov[edge] = (255, 255, 255, 255)
    halo = ndi.binary_dilation(edge, iterations=max(1, side // 240)) & ~edge; ov[halo] = (0, 0, 0, 170)
    Image.fromarray(ov).resize((360, 360), Image.NEAREST).save(os.path.join(IMG, f'val{name}_outline.png'), optimize=True)
    rec.update(crop=[y0, x0, side], H=H, W=W, contrast=contrast)
    frames.append(rec)
    print(f"frame {name}: {int(m.sum())} obj px | best band {rec['best_band_nm']:.0f} nm F={rec['best_band_fisher']:.3f} | best V {rec['best_volt']:.2f} "
          f"F={rec['best_volt_fisher']:.3f} | sep ring raw {sep_ring['raw133']:.2f} resp {sep_ring['resp344']:.2f} pca11 {sep_ring['pca11']:.2f} "
          f"top10 {sep_ring['top10']:.2f} | maps {contrast} [{time.time() - t0:.0f}s]", flush=True)

fr = np.array(fisher_raw); fv = np.array(fisher_resp); cr = np.array(cohen_raw)
q = lambda a: {'q25': np.round(np.percentile(a, 25, 0), 5).tolist(), 'med': np.round(np.median(a, 0), 5).tolist(),
               'q75': np.round(np.percentile(a, 75, 0), 5).tolist()}

# difference spectrum of a representative frame and what the sensor can represent of it
recon = {}
for rec in frames:
    d = np.array(rec['mean_obj']) - np.array(rec['mean_ring'])
    recon[rec['name']] = {'diff': np.round(d, 6).tolist(),
                          'k6': np.round(U[:, :6] @ (U[:, :6].T @ d), 6).tolist(),
                          'k11': np.round(U[:, :11] @ (U[:, :11].T @ d), 6).tolist()}

out = dict(
    wavelengths=np.round(wl, 2).tolist(), voltages=np.round(volts, 3).tolist(), top_idx=top_idx,
    top_volts=np.round(volts[top_idx], 3).tolist(),
    R_peak_top10=[np.round(Rpk[:, j], 4).tolist() for j in top_idx],
    R_peak_context={f'{volts[j]:.2f}': np.round(Rpk[:, j], 4).tolist() for j in range(0, len(volts), 24)},
    residual_R=np.round(res_R[:40], 8).tolist(), residual_scene=np.round(res_scene[:40], 8).tolist(),
    rank99=rank99, rank999=rank999, scene99=scene99, scene999=scene999,
    fwhm=np.round(fwhm, 1).tolist(), fwhm_median=float(np.median(fwhm)), cos_adj=np.round(cos_adj, 6).tolist(),
    fisher_raw=q(fr), fisher_resp=q(fv), cohen_raw=q(np.abs(cr)),
    frames=frames, recon=recon, band_std=np.round(bstd, 5).tolist(), R_peak=np.round(Rpk, 3).tolist(),
)
json.dump(out, open(os.path.join(OUT, 'data_part1.json'), 'w'))
print(f'wrote data_part1.json with {len(frames)} frames in {time.time() - t0:.0f}s', flush=True)
