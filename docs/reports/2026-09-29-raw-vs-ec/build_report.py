"""Assemble raw-vs-ec.html: template + data JSON + embedded images (data URIs)."""
import os, sys, json, base64, io
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.environ.get('REPORT_OUT', os.path.join(HERE, 'build'))                 # data_*.json + img/ from the analyze scripts
WT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))                 # repo root
sys.path.insert(0, WT); os.chdir(WT)
import main_det

p1 = json.load(open(f'{R}/data_part1.json')); p2 = json.load(open(f'{R}/data_part2.json'))


def uri_file(path):
    mime = 'image/png' if path.endswith('.png') else 'image/jpeg'
    return f'data:{mime};base64,' + base64.b64encode(open(path, 'rb').read()).decode()


def uri_png(arr):
    b = io.BytesIO(); Image.fromarray(arr).save(b, 'PNG', optimize=True)
    return 'data:image/png;base64,' + base64.b64encode(b.getvalue()).decode()


# filter energy and scene covariance residuals up to k = 100 (same definitions as analyze_data.py)
args = main_det.get_args_parser().parse_args([]); main_det.load_cfg(args)
ds_tr, _, _ = main_det.build_datasets(args)
Rm, _, _, volts = ds_tr.filter_bank_tensors(); Rm = Rm.astype(np.float64); volts = np.asarray(volts, np.float64)
_, cov = ds_tr._band_stats()
sv = np.linalg.svd(Rm, compute_uv=False); en = sv ** 2 / np.sum(sv ** 2); resR = 1 - np.cumsum(en)
ev = np.maximum(np.sort(np.linalg.eigvalsh(np.asarray(cov, np.float64)))[::-1], 0); resS = 1 - np.cumsum(ev) / ev.sum()
resR = np.maximum(resR, 0)[:100]; resS = np.maximum(resS, 0)[:100]

# heatmap on a uniform 0.01 V grid (351 columns); dead-zone columns get the neutral midpoint
Rpk = np.array(p1['R_peak'])                                         # [133 wl ascending, 344 volts]
grid = np.round(np.arange(-1.0, 2.5001, 0.01), 2); col = {round(float(v), 2): j for j, v in enumerate(volts)}
def heat(mid):
    neg = np.array([0x2a, 0x78, 0xd6], float); pos = np.array([0xe3, 0x49, 0x48], float); m = np.array(mid, float)
    out = np.tile(m, (Rpk.shape[0], len(grid), 1))
    for gi, v in enumerate(grid):
        j = col.get(round(float(v), 2))
        if j is None: continue
        t = np.clip(Rpk[:, j], -1, 1)[:, None]
        out[:, gi] = np.where(t < 0, m + (neg - m) * (-t), m + (pos - m) * t)
    return uri_png(out[::-1].round().astype(np.uint8))              # wavelength increasing upward


def smooth(a, k=5):
    a = np.array(a, float); out = []
    for i in range(len(a)):
        lo, hi = max(0, i - k // 2), min(len(a), i + k // 2 + 1); out.append(round(float(np.nanmean(a[lo:hi])), 4))
    return out


runs = {}
for key, r in p2['runs'].items():
    c = r['curves']; d = {'epoch': c['epoch']}
    for m in ['ap50', 'recall50', 'coverage_recall99_roi_op']:
        d[m] = c[m]; d[m + '_s'] = smooth(c[m])
    runs[key] = d

img = {}
frames = []
for f in p1['frames']:
    n = f['name']
    for k in ['rgbcrop', 'band', 'volt', 'lda_raw', 'lda_resp', 'lda_pca', 'lda_top10']:
        img[f'val{n}_{k}'] = uri_file(f'{R}/img/val{n}_{k}.jpg')
    img[f'val{n}_outline'] = uri_file(f'{R}/img/val{n}_outline.png')
    frames.append({k: f[k] for k in ['name', 'obj_px', 'best_band_nm', 'best_band_fisher', 'best_volt', 'best_volt_fisher', 'sep_ring', 'sep_bg',
                                     'contrast', 'fisher_raw', 'fisher_resp']})
show = [['246', 'all'], ['301', 'raw'], ['212', 'B'], ['196', 'none']]
for n, _ in show:
    img[f'test{n}_rgb'] = uri_file(f'{R}/img/test{n}_rgb.jpg'); img[f'test{n}_outline'] = uri_file(f'{R}/img/test{n}_outline.png')

D = dict(
    wl=p1['wavelengths'], volts=p1['voltages'], topIdx=p1['top_idx'], topVolts=p1['top_volts'],
    bestVoltIdx=int(np.abs(np.array(p1['voltages']) - 1.74).argmin()),
    Rpk=p1['R_peak'], heat={'light': heat((0xf0, 0xef, 0xec)), 'dark': heat((0x38, 0x38, 0x35))},
    fisherRawQ=p1['fisher_raw'], fisherRespQ=p1['fisher_resp'], frames=frames,
    recon={k: {'diff': v['diff'], 'k11': v['k11']} for k, v in p1['recon'].items()},
    residR=[float(f'{v:.3e}') for v in resR], residScene=[float(f'{v:.3e}') for v in resS],
    rank99=p1['rank99'], rank999=p1['rank999'], scene99=p1['scene99'], scene999=p1['scene999'],
    runs=runs, test=p2['test'], gate=p2['gate'], examples=p2['examples'], showExamples=show, img=img,
)
blob = json.dumps(D, separators=(',', ':')).replace('</', '<\\/')
html = open(os.path.join(HERE, 'template.html')).read()
assert html.count('/*__DATA__*/null') == 1
html = html.replace('/*__DATA__*/null', blob)
out = f'{R}/raw-vs-ec.html'
open(out, 'w').write(html)
print(f'wrote {out}: {len(html) / 1e6:.2f} MB (images {sum(len(v) for v in img.values()) / 1e6:.2f} MB); '
      f'rank {p1["rank999"]}, scene {p1["scene999"]}, bestVoltIdx {D["bestVoltIdx"]} -> {p1["voltages"][D["bestVoltIdx"]]} V, pca epochs {len(runs["pca"]["epoch"])}')
