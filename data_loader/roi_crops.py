import torch.utils.data as Dataset
import os
import json
import math
import numpy as np
import torch

from data_loader.boxes import boxes_from_mask, expand_box
from data_loader.cube_cache import read_npy_direct, cache_path, default_cache_dir
from data_loader.det_splits import load_det_ids
from data_loader.my_dataset import HyperCOD_data

ARMS = ('raw', 'ec10', 'ec24')          # input arms that have their own Stage-1 detector (and ROI files)
SEG_ARMS = ARMS + ('rgb',)              # 'rgb' = pseudo-RGB control; it reuses the raw arm's ROIs
INDEX_NAME = 'index.json'


def pixel_box(box, H, W):
    '''
    Float xyxy (native pixel edges) -> int (x1, y1, x2, y2): floor x1/y1, ceil x2/y2, clipped to [0, W] x [0, H].
    The same rule as train_eval.box_metrics.mask_coverage, so "ROI inside window" and "mask covered by ROI" agree
    to the pixel. expand_box returns fractional values (e.g. 18.5), hence the rounding.
    '''
    x1 = min(max(0, int(math.floor(float(box[0])))), W)
    y1 = min(max(0, int(math.floor(float(box[1])))), H)
    x2 = min(max(0, int(math.ceil(float(box[2])))), W)
    y2 = min(max(0, int(math.ceil(float(box[3])))), H)
    return x1, y1, x2, y2


def inside(inner, outer):
    '''True if the int xyxy rectangle inner lies inside the int xyxy rectangle outer.'''
    return inner[0] >= outer[0] and inner[1] >= outer[1] and inner[2] <= outer[2] and inner[3] <= outer[3]


# ----------------------------------------------------------------------------------------------------------------
# ROI matching and the crop cache
# ----------------------------------------------------------------------------------------------------------------
def match_rois(rois, labels, ids, min_cover=0.01):
    '''
    Match each detector ROI to the object whose mask it covers most.
    rois [K, 4] float xyxy frame px; labels [H, W] int32 component labels and ids the kept (1-based) labels, both
    from boxes_from_mask(..., return_labels=True). Coverage = share of the object's mask pixels inside the ROI
    (pixel_box rule, as mask_coverage). Returns np.ndarray [K] int64: the object id with the largest coverage if
    that coverage >= min_cover, else -1 (a false-positive ROI). Ties go to the lower id.
    '''
    rois = np.asarray(rois, dtype=np.float64).reshape(-1, 4)          # [K, 4]
    out = -np.ones(len(rois), dtype=np.int64)                         # [K]
    ids = [int(i) for i in ids]
    if len(rois) == 0 or len(ids) == 0:
        return out
    H, W = labels.shape
    n_lab = int(labels.max()) + 1
    totals = np.bincount(labels.ravel(), minlength=n_lab)             # [n_lab] mask pixels per label
    for k, r in enumerate(rois):
        x1, y1, x2, y2 = pixel_box(r, H, W)
        if x2 <= x1 or y2 <= y1:
            continue
        within = np.bincount(labels[y1:y2, x1:x2].ravel(), minlength=n_lab)   # [n_lab] pixels inside the ROI
        cover = np.array([within[i] / totals[i] for i in ids])                # [n_obj]
        j = int(np.argmax(cover))
        if cover[j] >= min_cover:
            out[k] = ids[j]
    return out


def frame_windows(gt, frame_rois, grow=2.0, min_side=512, min_area=100, min_cover=0.01, name=''):
    '''
    The cache windows of one frame (no I/O).
    gt: bool [H, W] full-frame GT; frame_rois: {arm: {"rois": [[x1,y1,x2,y2,conf], ...], "boxes": [...]}} of this
    frame from each arm's ROI export; name: the frame id, only used in assertion messages (spec §9).
    Returns a list of dicts with kind, window (int xyxy), object, objects, rois:
      object window (one per GT object with area >= min_area): the smallest rectangle containing
        expand_box(gt_box, grow, min_side, H, W) and every ROI (any arm) matched to the object; rois[arm] = that
        arm's ROIs matched to the object;
      fp window (one per distinct false-positive ROI): the ROI itself; an fp ROI with identical coordinates in
        several arms is stored once, with each arm's entry under rois[arm].
    Every ROI entry is {"roi", "box", "conf", "object"} (object = matched id or -1).
    '''
    H, W = gt.shape
    gt_boxes, labels, ids = boxes_from_mask(gt, min_area=min_area, return_labels=True)
    areas = np.bincount(labels.ravel(), minlength=max(ids, default=0) + 1)    # pixels per component label
    objects = [{'id': int(i), 'box': [float(v) for v in b], 'area': int(areas[i])} for b, i in zip(gt_boxes, ids)]
    arms = list(frame_rois)
    matched = {o['id']: {arm: [] for arm in arms} for o in objects}
    fps = {}                                                          # rounded roi -> {"roi", "rois": {arm: [...]}}
    for arm in arms:
        rois = np.asarray(frame_rois[arm]['rois'], dtype=np.float64).reshape(-1, 5)     # [K, 5] xyxy + conf
        boxes = np.asarray(frame_rois[arm]['boxes'], dtype=np.float64).reshape(-1, 5)   # [K, 5] pre-expansion boxes
        assert len(rois) == len(boxes), f"frame {name}, arm {arm}: {len(rois)} rois but {len(boxes)} boxes"
        obj = match_rois(rois[:, :4], labels, ids, min_cover)                            # [K]
        for r, b, o in zip(rois, boxes, obj):
            e = {'roi': [float(v) for v in r[:4]], 'box': [float(v) for v in b[:4]], 'conf': float(r[4]), 'object': int(o)}
            if o >= 0:
                matched[int(o)][arm].append(e)
            else:
                key = tuple(np.round(r[:4], 3).tolist())
                fps.setdefault(key, {'roi': e['roi'], 'rois': {a: [] for a in arms}})['rois'][arm].append(e)

    def overlapping(win):
        # objects whose box intersects the window (for an object window this includes the window's own object)
        return [o for o in objects if o['box'][0] < win[2] and o['box'][2] > win[0] and o['box'][1] < win[3] and o['box'][3] > win[1]]

    windows = []
    for o in objects:
        rects = [pixel_box(expand_box(o['box'], grow, min_side, H, W), H, W)]
        rects += [pixel_box(e['roi'], H, W) for arm in arms for e in matched[o['id']][arm]]
        win = [min(r[0] for r in rects), min(r[1] for r in rects), max(r[2] for r in rects), max(r[3] for r in rects)]
        windows.append({'kind': 'object', 'window': win, 'object': o['id'], 'objects': overlapping(win),
                        'rois': matched[o['id']]})
    for key in sorted(fps):
        win = list(pixel_box(fps[key]['roi'], H, W))
        assert win[2] > win[0] and win[3] > win[1], f"frame {name}: degenerate false-positive ROI {fps[key]['roi']}"
        windows.append({'kind': 'fp', 'window': win, 'object': -1, 'objects': overlapping(win), 'rois': fps[key]['rois']})
    for w in windows:                                                 # spec §9: every ROI inside its window
        for arm in arms:
            bad = [e['roi'] for e in w['rois'][arm] if not inside(pixel_box(e['roi'], H, W), w['window'])]
            assert not bad, f"frame {name}: {arm} ROIs {bad} outside window {w['window']}"
    return windows


def load_roi_file(path, names):
    '''{name: {"rois", "boxes", ...}} of an ROI export (main_det_rois.py), restricted to names; every name must be present.'''
    assert os.path.exists(path), f"ROI file {path} not found; export it with main_det_rois.py"
    with open(path) as f:
        data = json.load(f)
    missing = [n for n in names if n not in data]
    assert not missing, f"ROI file {path} lacks frames {missing[:10]} ({len(missing)} missing): wrong split or detector?"
    return {n: data[n] for n in names}


class CropFrames(Dataset.Dataset):
    '''
    One item per frame for build_crop_cache: reads the frame once (O_DIRECT), writes its windows
    (<id>_<k>.npy fp16 [n_bands, h, w] un-scaled, <id>_<k>_gt.npy bool [h, w]) and returns their index records,
    so DataLoader workers do the reading and writing in parallel.
    '''
    def __init__(self, ds, split, frame_rois, out_dir, grow=2.0, min_side=512, min_area=100, min_cover=0.01):
        super(CropFrames, self).__init__()
        self.ds = ds                    # HyperCOD_data (split 'train', ids of this split, norm 'p99', cache_dir set)
        self.split = split              # 'train' or 'val' (both live in the HyperCOD 'train' directory)
        self.frame_rois = frame_rois    # {name: {arm: {"rois", "boxes"}}}
        self.out_dir = out_dir
        self.grow, self.min_side, self.min_area, self.min_cover = grow, min_side, min_area, min_cover

    def __len__(self):
        return len(self.ds.img_name)

    def __getitem__(self, idx):
        name = self.ds.img_name[idx]
        gt = self.ds.load_gt(name)                                    # [H, W] bool (raw GT incl. JPEG specks)
        wins = frame_windows(gt, self.frame_rois[name], self.grow, self.min_side, self.min_area, self.min_cover, name)
        if not wins:
            return []
        p = cache_path(self.ds.cache_dir, self.ds.split, name)
        assert os.path.exists(p), f"frame cache {p} missing; build it with python -m data_loader.cube_cache"
        arr = read_npy_direct(p)                                      # [n_bands, H, W] fp16, one O_DIRECT read
        assert arr.shape == (self.ds.n_bands, self.ds.H, self.ds.W) and arr.dtype == np.float16, \
            f"cache {p} has {arr.shape} {arr.dtype}, expected ({self.ds.n_bands}, {self.ds.H}, {self.ds.W}) float16"
        records = []
        for k, w in enumerate(wins):
            x1, y1, x2, y2 = w['window']
            f, g = f'{name}_{k}.npy', f'{name}_{k}_gt.npy'
            np.save(os.path.join(self.out_dir, f), np.ascontiguousarray(arr[:, y1:y2, x1:x2]))   # [n_bands, h, w] fp16
            np.save(os.path.join(self.out_dir, g), np.ascontiguousarray(gt[y1:y2, x1:x2]))       # [h, w] bool
            records.append({'file': f, 'gt_file': g, 'frame': name, 'split': self.split, 'kind': w['kind'],
                            'window': [int(v) for v in w['window']], 'scale': float(self.ds.scale[name]),
                            'object': w['object'], 'objects': w['objects'], 'rois': w['rois']})
        return records


def records_collate_fn(batch):
    '''CropFrames items are lists of index records; flatten the batch into one list.'''
    return [r for recs in batch for r in recs]


def build_crop_cache(data_path, out_dir, roi_files, splits=('train', 'val'), grow=2.0, min_side=512, num_workers=4,
                     min_area=100, min_cover=0.01, split_file=None, cache_dir=None, band_range=(400.0, 800.0)):
    '''
    One-off crop cache for Stage-2 training (spec §3): every train/val frame is read once (O_DIRECT, from the fp16
    frame cache) and cut into object windows and false-positive windows (see frame_windows).
    roi_files: {arm: {split: path}} with arm in ARMS, from main_det_rois.py (rois_<run>_<split>.json).
    split_file: the det_val_ids.json that defines train (251) / val (28) ids (default data_loader/splits/...).
    Writes <out_dir>/<id>_<k>.npy, <out_dir>/<id>_<k>_gt.npy and, last and atomically, <out_dir>/index.json =
      {"band_range", "frame_hw", "grow", "min_side", "min_area", "min_cover", "arms", "roi_files",
       "windows": [{"file", "gt_file", "frame", "split", "kind", "window", "scale", "object", "objects", "rois"}]}
    so a crashed build leaves no index.json and is rebuilt. Returns {"n_windows", "bytes", "n_object", "n_fp"}.
    '''
    arms = [a for a in ARMS if a in roi_files]
    assert arms and set(roi_files) <= set(ARMS), f"roi_files arms must be in {ARMS}, got {list(roi_files)}"
    for arm in arms:
        assert all(s in roi_files[arm] for s in splits), f"roi_files[{arm!r}] lacks splits {[s for s in splits if s not in roi_files[arm]]}"
    cache_dir = cache_dir if cache_dir else default_cache_dir(data_path)
    train_ids, val_ids = load_det_ids(data_path, split_file)
    os.makedirs(out_dir, exist_ok=True)
    windows, frame_hw = [], None
    for split in splits:
        assert split in ('train', 'val'), f"split must be 'train' or 'val', got {split!r}"
        # val frames live in the HyperCOD 'train' directory; norm 'p99' only to get the per-frame scale
        ds = HyperCOD_data(data_path, split='train', ids=train_ids if split == 'train' else val_ids, use_filter=False,
                           norm='p99', crop_size=0, band_range=band_range, filter_norm='none', cache_dir=cache_dir,
                           out_dtype='float16')
        frame_hw = [ds.H, ds.W]
        rois = {arm: load_roi_file(roi_files[arm][split], ds.img_name) for arm in arms}
        frame_rois = {name: {arm: rois[arm][name] for arm in arms} for name in ds.img_name}
        loader = torch.utils.data.DataLoader(
            CropFrames(ds, split, frame_rois, out_dir, grow, min_side, min_area, min_cover), batch_size=1,
            shuffle=False, num_workers=num_workers, collate_fn=records_collate_fn)
        print(f"Crop cache: {len(ds)} {split} frames, arms {arms} -> {out_dir}")
        for i, recs in enumerate(loader):
            windows += recs
            if (i + 1) % 25 == 0 or i + 1 == len(ds):
                print(f"  {split} {i + 1}/{len(ds)} frames, {len(windows)} windows")
    n_bytes = sum(os.path.getsize(os.path.join(out_dir, w[k])) for w in windows for k in ('file', 'gt_file'))
    n_obj = sum(w['kind'] == 'object' for w in windows)
    index = {'band_range': [float(band_range[0]), float(band_range[1])], 'frame_hw': frame_hw, 'grow': float(grow),
             'min_side': int(min_side), 'min_area': int(min_area), 'min_cover': float(min_cover), 'arms': arms,
             'roi_files': {a: {s: str(roi_files[a][s]) for s in splits} for a in arms}, 'windows': windows}
    tmp = os.path.join(out_dir, INDEX_NAME + '.tmp')
    with open(tmp, 'w') as f:
        json.dump(index, f)
    os.replace(tmp, os.path.join(out_dir, INDEX_NAME))
    print(f"Crop cache done: {len(windows)} windows ({n_obj} object, {len(windows) - n_obj} fp), {n_bytes / 1e9:.2f} GB")
    return {'n_windows': len(windows), 'bytes': n_bytes, 'n_object': n_obj, 'n_fp': len(windows) - n_obj}
