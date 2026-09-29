"""
Report data, part 2: training curves, gate evolution, test metrics, and detection examples (GPU 1, 12 test frames).
Writes data_part2.json + img/test*_rgb.jpg / _outline.png.
"""
import os, sys, json, time
import numpy as np
import scipy.ndimage as ndi
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
WT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))                 # repo root
sys.path.insert(0, WT); os.chdir(WT)
import main_det
from data_loader.boxes import boxes_from_mask, expand_box
from models.ec_yolo import decode_predictions
from train_eval.box_metrics import mask_coverage, filter_to_operating_point, box_iou_matrix

OUT = os.environ.get('REPORT_OUT', os.path.join(HERE, 'build'))
IMG = os.path.join(OUT, 'img'); os.makedirs(IMG, exist_ok=True)
t0 = time.time()
KEYS = ['ap50', 'recall50', 'matched_iou', 'coverage_recall99_roi', 'coverage_recall99_roi_op', 'recall50_op', 'dets_per_image',
        'dets_per_image_op', 'tightness_op', 'center_offset_op', 'contain_rate_op', 'coverage_recall99_raw']


def last_run(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    starts = [i for i, r in enumerate(rows) if r.get('epoch') == 0 and 'val' in r and not r.get('final')]
    seg = rows[starts[-1]:] if starts else rows
    ep = [r for r in seg if 'val' in r and not r.get('final') and not r.get('eval')]
    fin = [r for r in seg if r.get('final')]
    return ep, (fin[-1] if fin else None)


def curves(ep):
    return {'epoch': [r['epoch'] for r in ep],
            **{k: [round(float(r['val'].get(k, float('nan'))), 4) for r in ep] for k in KEYS},
            'train_loss': [round(float(r['train'].get('loss', float('nan'))), 4) for r in ep],
            'gate_entropy': [round(float(r['train'].get('gate_entropy', float('nan'))), 5) for r in ep]}


runs = {}
for key, path in [('raw', 'weights/raw133_A/results_raw133_A.txt'), ('A', 'weights/det_A/results_det_A.txt'),
                  ('B', 'weights/det_B/results_det_B.txt'), ('pca', 'weights/pca11_A/results_pca11_A.txt')]:
    ep, fin = last_run(path)
    runs[key] = {'curves': curves(ep), 'final': fin}
    print(f'{key}: {len(ep)} epochs, final={"yes" if fin else "no"}', flush=True)


def summary(d):
    return {k: round(float(d[k]), 4) for k in KEYS if k in d}


test = {
    'raw': {'ckpt': 'raw133_A/model_best', 'epoch': runs['raw']['final']['best_epoch'], 'val': summary(runs['raw']['final']['val']), 'test': summary(runs['raw']['final']['test'])},
}
for key, path, ck, e in [('A', 'weights/det_A_eval89/results_det_A_eval89.txt', 'det_A/model_89', 89),
                         ('B', 'weights/det_B_eval/results_det_B_eval.txt', 'det_B/model_best', 25)]:
    r = json.loads(open(path).read().strip().splitlines()[-1])
    test[key] = {'ckpt': ck, 'epoch': e, 'val': summary(r['val']), 'test': summary(r['test'])}
print({k: (v['test']['recall50'], v['test']['ap50'], v['test']['coverage_recall99_roi_op']) for k, v in test.items()}, flush=True)

# gate evolution from session A checkpoints
gate = {}
for e in [9, 19, 29, 39, 49, 59, 69, 79, 89, 99]:
    ck = torch.load(f'weights/det_A/model_{e}', map_location='cpu', weights_only=False)
    th = ck['model']['filter_bank.theta'].float()
    gate[str(e)] = np.round((th.numel() * torch.softmax(th, 0)).numpy(), 4).tolist()
print('gate epochs', list(gate), 'max at 99', max(gate['99']), flush=True)

# ---------------------------------------------------------------- detection examples on 12 test frames (GPU 1)
args = main_det.get_args_parser().parse_args([]); main_det.load_cfg(args)
ds_tr, ds_val, ds_te = main_det.build_datasets(args)
areas = [(name, int(ds_te.load_gt(name).sum())) for name in ds_te.img_name]
cand = sorted([a for a in areas if 1500 <= a[1] <= 60000], key=lambda a: a[1])
pick = [cand[int(round(i))] for i in np.linspace(0, len(cand) - 1, 12)]
print('candidates (name, obj px):', pick, flush=True)
device = torch.device('cuda')


def load(session, path, raw=False):
    a = main_det.get_args_parser().parse_args(['--session', session] + (['--raw-bands'] if raw else [])); main_det.load_cfg(a)
    ck = torch.load(path, map_location='cpu', weights_only=False)
    return main_det.build_model(a, ds_tr, ck).to(device).eval(), a


models = {'raw': load('A', 'weights/raw133_A/model_best', raw=True), 'A': load('A', 'weights/det_A/model_89'),
          'B': load('B', 'weights/det_B/model_best')}
print(f'models loaded [{time.time() - t0:.0f}s]', flush=True)
name_to_idx = {n: i for i, n in enumerate(ds_te.img_name)}
examples = []
with torch.no_grad():
    for name, area in pick:
        img, gt, _ = ds_te[name_to_idx[name]]
        mask = gt[0] > 0.5; H, W = mask.shape
        gt_boxes, labels, ids = boxes_from_mask(mask, min_area=args.min_area, return_labels=True)
        x = torch.from_numpy(np.ascontiguousarray(img))[None].to(device)
        rec = {'name': str(name), 'obj_px': area, 'H': H, 'W': W, 'gt_boxes': np.round(gt_boxes, 1).tolist(), 'models': {}}
        for key, (model, a) in models.items():
            with torch.autocast('cuda'):
                out = model(x)
            dets = decode_predictions(out, conf_thres=0.001, iou_thres=a.iou_thres, max_det=a.max_det, end2end=getattr(model.yolo, 'end2end', None))[0]
            dets[:, [0, 2]] = dets[:, [0, 2]].clip(0, W); dets[:, [1, 3]] = dets[:, [1, 3]].clip(0, H)
            op = filter_to_operating_point(dets, a.roi_conf, a.roi_topk)
            iou = box_iou_matrix(gt_boxes, op[:, :4]) if len(gt_boxes) and len(op) else np.zeros((len(gt_boxes), 0))
            rois = [expand_box(d[:4], a.roi_margin, a.roi_min, H, W) for d in op]
            covered = [bool(any(mask_coverage(labels == cid, r) >= 0.99 for r in rois)) for cid in ids]
            rec['models'][key] = {'boxes': np.round(op[:, :5], 3).tolist(), 'rois': [np.round(np.asarray(r), 1).tolist() for r in rois],
                                  'best_iou': np.round(iou.max(1), 3).tolist() if iou.size else [0.0] * len(gt_boxes),
                                  'hit50': bool(iou.size and (iou.max(1) >= 0.5).all()), 'roi_cov': bool(all(covered)) if covered else False,
                                  'top_conf': round(float(dets[:, 4].max()), 4) if len(dets) else 0.0}
        wl = np.asarray(ds_te.wavelens)
        rgb_idx = [int(np.abs(wl - t).argmin()) for t in (640, 550, 460)]
        full = np.stack([np.asarray(img[b], np.float32) for b in rgb_idx], -1)
        full = np.clip(full / np.percentile(full, 99.5, axis=(0, 1), keepdims=True), 0, 1) ** (1 / 2.2)
        Image.fromarray((full * 255).round().astype(np.uint8)).resize((W // 2, H // 2), Image.LANCZOS).save(os.path.join(IMG, f'test{name}_rgb.jpg'), quality=82)
        small = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((W // 2, H // 2), Image.NEAREST)) > 127
        edge = small & ~ndi.binary_erosion(small, iterations=2)
        ov = np.zeros(small.shape + (4,), np.uint8); halo = ndi.binary_dilation(edge, iterations=1) & ~edge
        ov[halo] = (0, 0, 0, 160); ov[edge] = (255, 255, 255, 255)
        Image.fromarray(ov).save(os.path.join(IMG, f'test{name}_outline.png'), optimize=True)
        examples.append(rec)
        print(f"test {name} ({area} px): " + ' | '.join(f"{k} hit50={v['hit50']} roi={v['roi_cov']} iou={v['best_iou']} n={len(v['boxes'])} top={v['top_conf']:.3f}"
                                                     for k, v in rec['models'].items()) + f' [{time.time() - t0:.0f}s]', flush=True)

json.dump({'runs': runs, 'test': test, 'gate': gate, 'examples': examples}, open(os.path.join(OUT, 'data_part2.json'), 'w'))
print(f'wrote data_part2.json [{time.time() - t0:.0f}s]', flush=True)
