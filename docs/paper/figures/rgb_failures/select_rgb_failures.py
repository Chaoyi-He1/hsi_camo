'''Per test image: the strict RGB chain (rgb_A boxes + SAM2-UNet trained on pseudo-RGB) against the EC chain (sel10g_clean_A
boxes + SAM2-UNet trained on the 10 EC readings), level (c), seed-mean IoU / S over seeds 0-2 (results/seg_rgbdet, one evaluation),
and what each detector did (boxes, false positives, GT box fully inside an ROI). Writes rgb_vs_ec_per_image.csv and prints the
categories and candidate failure examples.'''
import csv
import json
import os
import pickle

import numpy as np

R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../..'))       # the repository root
OUT = os.path.dirname(os.path.abspath(__file__))
SEEDS = (0, 1, 2)


def per_image(run):
    p = pickle.load(open(f'{R}/results/seg_rgbdet/{run}/per_image.pkl', 'rb'))['full_det']
    return {k: r for k, r in zip(p['keys'], p['rows'])}, p['fp_rows']


def chain(arm):
    rows = [per_image(f'seg_sam2unet_{arm}_s{s}') for s in SEEDS]
    names = list(rows[0][0])
    out = {}
    for n in names:
        out[n] = {k: float(np.mean([r[0][n][k] for r in rows])) for k in ('IoU', 'S', 'Fw')}
        out[n]['IoU_s0'] = rows[0][0][n]['IoU']
        out[n]['false_masks_s0'] = sum(1 for f in rows[0][1] if f['frame'] == n and f['false_mask'])
    return out


def det_status(run):
    rois = json.load(open(f'{R}/results/det/rois_{run}_test.json'))
    st = {}
    for n, e in rois.items():
        gts = e['gt_boxes']
        inside = [any(r[0] <= g[0] and r[1] <= g[1] and r[2] >= g[2] and r[3] >= g[3] for r in e['rois']) for g in gts]
        touch = [any(min(r[2], g[2]) > max(r[0], g[0]) and min(r[3], g[3]) > max(r[1], g[1]) for r in e['rois']) for g in gts]
        st[n] = dict(n_boxes=len(e['rois']), all_inside=all(inside), any_touch=any(touch), n_gt=len(gts),
                     n_fp=sum(1 for r in e['rois'] if not any(min(r[2], g[2]) > max(r[0], g[0]) and min(r[3], g[3]) > max(r[1], g[1]) for g in gts)),
                     max_conf=max([r[4] for r in e['rois']], default=0.0))
    return st


rgb, ec = chain('rgb'), chain('ec10')
drgb, dec = det_status('rgb_A'), det_status('sel10g_clean_A')


def category(d):
    if d['n_boxes'] == 0:
        return 'no box'
    if not d['any_touch']:
        return 'boxes elsewhere'
    if not d['all_inside']:
        return 'object partly boxed'
    return 'object boxed'


rows = []
for n in sorted(rgb, key=int):
    rows.append(dict(image=n, rgb_detector=category(drgb[n]), rgb_boxes=drgb[n]['n_boxes'], rgb_fp_boxes=drgb[n]['n_fp'],
                     ec_detector=category(dec[n]), ec_boxes=dec[n]['n_boxes'], ec_fp_boxes=dec[n]['n_fp'],
                     rgb_IoU=round(rgb[n]['IoU'], 4), ec_IoU=round(ec[n]['IoU'], 4), diff_IoU=round(ec[n]['IoU'] - rgb[n]['IoU'], 4),
                     rgb_S=round(rgb[n]['S'], 4), ec_S=round(ec[n]['S'], 4), rgb_IoU_s0=round(rgb[n]['IoU_s0'], 4), ec_IoU_s0=round(ec[n]['IoU_s0'], 4),
                     rgb_false_masks_s0=rgb[n]['false_masks_s0'], ec_false_masks_s0=ec[n]['false_masks_s0']))
with open(os.path.join(OUT, 'rgb_vs_ec_per_image.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

print(f"{len(rows)} test images; mean IoU RGB chain {np.mean([r['rgb_IoU'] for r in rows]):.3f}, EC chain {np.mean([r['ec_IoU'] for r in rows]):.3f}")
for name, d in (('RGB detector', 'rgb_detector'), ('EC detector', 'ec_detector')):
    cats = {}
    for r in rows:
        cats[r[d]] = cats.get(r[d], 0) + 1
    print(name, cats)
print('cross table (RGB detector x EC detector):')
for a in ('object boxed', 'object partly boxed', 'boxes elsewhere', 'no box'):
    print(f"  RGB {a:20s}", {b: sum(1 for r in rows if r['rgb_detector'] == a and r['ec_detector'] == b) for b in ('object boxed', 'object partly boxed', 'boxes elsewhere', 'no box')})
good, bad = 0.5, 0.2
win = [r for r in rows if r['ec_IoU'] >= good and r['rgb_IoU'] <= bad]
lose = [r for r in rows if r['rgb_IoU'] >= good and r['ec_IoU'] <= bad]
print(f"EC IoU >= {good} and RGB IoU <= {bad}: {len(win)} images; the reverse: {len(lose)} images")
print(f"diff IoU > +0.3: {sum(r['diff_IoU'] > 0.3 for r in rows)}, < -0.3: {sum(r['diff_IoU'] < -0.3 for r in rows)}")
print('\nRGB fails, EC succeeds (sorted by IoU gain):')
for r in sorted(win, key=lambda r: -r['diff_IoU']):
    print(f"  {r['image']:>4s}: RGB {r['rgb_detector']:20s} ({r['rgb_boxes']} boxes, {r['rgb_fp_boxes']} fp) IoU {r['rgb_IoU']:.3f} | EC {r['ec_detector']:20s} ({r['ec_boxes']} boxes) IoU {r['ec_IoU']:.3f} (s0 {r['ec_IoU_s0']:.3f}) S {r['rgb_S']:.3f} -> {r['ec_S']:.3f}")
print('\nreverse (EC fails, RGB succeeds):')
for r in sorted(lose, key=lambda r: r['diff_IoU']):
    print(f"  {r['image']:>4s}: RGB {r['rgb_detector']:20s} IoU {r['rgb_IoU']:.3f} | EC {r['ec_detector']:20s} IoU {r['ec_IoU']:.3f}")
