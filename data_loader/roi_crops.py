import torch.utils.data as Dataset
import os
import json
import math
import random
import numpy as np
import torch
import torch.nn.functional as F

from data_loader.boxes import boxes_from_mask, expand_box
from data_loader.cube_cache import read_npy_direct, cache_path, default_cache_dir
from data_loader.det_splits import load_det_ids
from data_loader.my_dataset import HyperCOD_data, scale_block

ARMS = ('raw', 'ec10', 'ec24')          # input arms that have their own Stage-1 detector (and ROI files)
SEG_ARMS = ARMS + ('rgb',)              # 'rgb' = pseudo-RGB control; it reuses the raw arm's ROIs
INDEX_NAME = 'index.json'
RESIZE_CHUNK = 16                       # channels per F.interpolate call: a 133-band 1680 x 700 crop in fp32 is 0.6 GB


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
    index.json exists only after a complete build: a rebuild into an existing out_dir first removes the old
    index.json (and a stale index.json.tmp), so a crashed build leaves no index.json and is rebuilt. The old window
    files are not deleted: the new build overwrites those it reuses and the rest stay unreferenced.
    Returns {"n_windows", "bytes", "n_object", "n_fp"}.
    '''
    arms = [a for a in ARMS if a in roi_files]
    assert arms and set(roi_files) <= set(ARMS), f"roi_files arms must be in {ARMS}, got {list(roi_files)}"
    for arm in arms:
        assert all(s in roi_files[arm] for s in splits), f"roi_files[{arm!r}] lacks splits {[s for s in splits if s not in roi_files[arm]]}"
    cache_dir = cache_dir if cache_dir else default_cache_dir(data_path)
    train_ids, val_ids = load_det_ids(data_path, split_file)
    os.makedirs(out_dir, exist_ok=True)
    for stale in (INDEX_NAME, INDEX_NAME + '.tmp'):                   # an index of an earlier build would point at windows about to be overwritten
        if os.path.exists(os.path.join(out_dir, stale)):
            os.remove(os.path.join(out_dir, stale))
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


# ----------------------------------------------------------------------------------------------------------------
# canvas geometry
# ----------------------------------------------------------------------------------------------------------------
def placed_size(h, w, s, canvas):
    '''(nh, nw) of an h x w crop resized by s on the canvas; shared by place_on_canvas and canvas_to_roi.'''
    nh = min(canvas, max(1, int(round(h * s))))
    nw = min(canvas, max(1, int(round(w * s))))
    return nh, nw


def resize_chw(x, size):
    '''
    Bilinear resize of a [C, h, w] array to size = (nh, nw); returns float32 [C, nh, nw].
    Antialiased when shrinking (a 1290 px ROI goes onto a 512 canvas), done in float32 chunks of RESIZE_CHUNK
    channels so a large 133-band crop never needs a full float32 copy.
    '''
    C, h, w = x.shape
    nh, nw = int(size[0]), int(size[1])
    if (h, w) == (nh, nw):
        return np.asarray(x, dtype=np.float32)
    anti = nh < h or nw < w
    out = np.empty((C, nh, nw), dtype=np.float32)
    for c0 in range(0, C, RESIZE_CHUNK):
        t = torch.from_numpy(np.ascontiguousarray(x[c0:c0 + RESIZE_CHUNK], dtype=np.float32))[None]   # [1, c, h, w]
        out[c0:c0 + RESIZE_CHUNK] = F.interpolate(t, size=(nh, nw), mode='bilinear', align_corners=False,
                                                  antialias=anti)[0].numpy()
    return out


def place_on_canvas(crop, canvas=512, scale=1.0):
    '''
    Put a native-resolution crop [C, h, w] on a canvas x canvas grid: s = min(1, canvas / max(h, w)) * scale,
    clipped so the resized crop always fits, resized (bilinear) only when the size changes, centred, zeros elsewhere.
    Returns (out [C, canvas, canvas] in crop's dtype, valid [canvas, canvas] bool = the placed region, (oy, ox), s).
    s = 1 (crop side <= canvas, no scale augmentation) copies the crop bit for bit.
    '''
    C, h, w = crop.shape
    assert h > 0 and w > 0, f"empty crop {crop.shape}"
    s = min(1.0, canvas / max(h, w)) * float(scale)
    s = min(s, canvas / max(h, w))                      # scale augmentation may not push the crop off the canvas
    nh, nw = placed_size(h, w, s, canvas)
    oy, ox = (canvas - nh) // 2, (canvas - nw) // 2     # centred
    out = np.zeros((C, canvas, canvas), dtype=crop.dtype)
    out[:, oy:oy + nh, ox:ox + nw] = crop if (nh, nw) == (h, w) else resize_chw(crop, (nh, nw)).astype(crop.dtype)
    valid = np.zeros((canvas, canvas), dtype=bool)
    valid[oy:oy + nh, ox:ox + nw] = True
    return out, valid, (oy, ox), s


def canvas_to_roi(canvas_map, offset, s, hw):
    '''
    Inverse of place_on_canvas for one map: cut the placed region out of canvas_map [canvas, canvas] and resize it
    back to the ROI's native (h, w). Returns float32 [h, w]; exact when no resize happened (s = 1).
    canvas_map may be a numpy array or a torch tensor (the model's sigmoid output).
    '''
    if torch.is_tensor(canvas_map):
        canvas_map = canvas_map.detach().float().cpu().numpy()
    canvas_map = np.asarray(canvas_map, dtype=np.float32)
    assert canvas_map.ndim == 2 and canvas_map.shape[0] == canvas_map.shape[1], f"expected [c, c], got {canvas_map.shape}"
    h, w = int(hw[0]), int(hw[1])
    oy, ox = int(offset[0]), int(offset[1])
    nh, nw = placed_size(h, w, s, canvas_map.shape[0])
    region = canvas_map[oy:oy + nh, ox:ox + nw]                        # [nh, nw]
    return region.copy() if (nh, nw) == (h, w) else resize_chw(region[None], (h, w))[0]


def box_to_canvas(box, roi, offset, hw, s, canvas):
    '''
    Frame-pixel xyxy box -> canvas-pixel xyxy, given the int ROI (x1, y1, x2, y2) it was placed from. Per-axis
    factors nw / w and nh / h (the rounded placed size), clipped to the placed region.
    '''
    h, w = hw
    oy, ox = offset
    nh, nw = placed_size(h, w, s, canvas)
    sx, sy = nw / w, nh / h
    x1 = ox + (float(box[0]) - roi[0]) * sx; x2 = ox + (float(box[2]) - roi[0]) * sx
    y1 = oy + (float(box[1]) - roi[1]) * sy; y2 = oy + (float(box[3]) - roi[1]) * sy
    return [float(np.clip(x1, ox, ox + nw)), float(np.clip(y1, oy, oy + nh)),
            float(np.clip(x2, ox, ox + nw)), float(np.clip(y2, oy, oy + nh))]


def rasterise_box(box, canvas):
    '''float32 [canvas, canvas] map, 1 on every pixel the xyxy box touches (pixel_box rule), 0 elsewhere.'''
    x1, y1, x2, y2 = pixel_box(box, canvas, canvas)
    m = np.zeros((canvas, canvas), dtype=np.float32)
    m[y1:y2, x1:x2] = 1.0
    return m


def flip_rot(arrays, box, hflip, vflip, k, canvas):
    '''
    Apply horizontal flip, vertical flip, then k x 90 deg counter-clockwise rotation (np.rot90 on the last two axes)
    to every array [..., canvas, canvas] and to the xyxy canvas box, so both stay consistent.
    rot90: out[i, j] = in[j, c - 1 - i]  ->  box (x1, y1, x2, y2) -> (y1, c - x2, y2, c - x1).
    '''
    x1, y1, x2, y2 = [float(v) for v in box]
    c = float(canvas)
    if hflip:
        arrays = [a[..., ::-1] for a in arrays]
        x1, x2 = c - x2, c - x1
    if vflip:
        arrays = [a[..., ::-1, :] for a in arrays]
        y1, y2 = c - y2, c - y1
    for _ in range(int(k) % 4):
        arrays = [np.rot90(a, 1, axes=(-2, -1)) for a in arrays]
        x1, y1, x2, y2 = y1, c - x2, y2, c - x1
    return [np.ascontiguousarray(a) for a in arrays], [x1, y1, x2, y2]


def jitter_gt_box(box, gt_jitter, rng, H, W):
    '''
    Training box from a GT box: each side moved OUTWARD by an independent U(0, gt_jitter) x (box side), never
    inward, so the object stays fully inside; clipped to the frame. rng: a random.Random (or the random module).
    '''
    x1, y1, x2, y2 = [float(v) for v in box]
    bw, bh = x2 - x1, y2 - y1
    return np.array([max(0.0, x1 - rng.uniform(0, gt_jitter) * bw), max(0.0, y1 - rng.uniform(0, gt_jitter) * bh),
                     min(float(W), x2 + rng.uniform(0, gt_jitter) * bw), min(float(H), y2 + rng.uniform(0, gt_jitter) * bh)],
                    dtype=np.float32)


def n_fp_items(n_obj, p_fp):
    '''Number of false-positive items so that they are a share p_fp of an epoch of n_obj object items.'''
    return int(round(n_obj * p_fp / (1.0 - p_fp))) if p_fp > 0 else 0


def load_window(path):
    '''
    Read one cached window whole (np.load through the page cache: the 22-34 GB crop cache is re-read every epoch
    and fits the ~72 GB of free page cache). No partial mmap reads, which stalled DataLoader workers in Stage 1.
    '''
    return np.load(path)


# ----------------------------------------------------------------------------------------------------------------
# training / validation dataset
# ----------------------------------------------------------------------------------------------------------------
class HyperCOD_roi(Dataset.Dataset):
    def __init__(self, cache_dir, split, arm, box_mix=(0.5, 0.4, 0.1), gt_jitter=0.15, roi_margin=1.5, roi_min=256,
                 canvas=512, scale_aug=(0.75, 1.25), gain_aug=0.05, train=True, seed=None):
        '''
        ROI items for Stage-2 segmentation, read from the crop cache of build_crop_cache.
        split: 'train' or 'val'; arm: 'raw', 'ec10', 'ec24' or 'rgb' (rgb uses the raw arm's ROIs).
        train=True: one item per object window plus false-positive items = box_mix[2] of the epoch (drawn uniformly
          from the arm's fp ROIs); an object item uses an expanded jittered GT box with probability
          box_mix[0] / (box_mix[0] + box_mix[1]) (5/9) and otherwise one of its matched detector ROIs (fallback:
          expanded GT if it has none), which gives the 50/40/10 mix overall. Augmentation: hflip, vflip, rot90,
          scale U(scale_aug) of the ROI content before placement, per-channel gain U(1 - gain_aug, 1 + gain_aug).
        train=False: deterministic; per object the oracle ROI expand_box(gt, roi_margin, roi_min) with the GT box as
          the box channel, plus every matched detector ROI of the arm; no augmentation and no fp items.
        '''
        super(HyperCOD_roi, self).__init__()
        assert split in ('train', 'val'), f"split must be 'train' or 'val', got {split!r}"
        assert arm in SEG_ARMS, f"arm must be one of {SEG_ARMS}, got {arm!r}"
        assert len(box_mix) == 3 and min(box_mix) >= 0 and abs(sum(box_mix) - 1.0) < 1e-6, f"box_mix must be 3 shares summing to 1, got {box_mix}"
        assert box_mix[0] + box_mix[1] > 0, f"box_mix {box_mix} has no object items"
        self.cache_dir = cache_dir
        self.split = split
        self.arm = arm
        self.roi_arm = 'raw' if arm == 'rgb' else arm     # rgb control: matched with the raw arm's detector
        self.box_mix = tuple(float(v) for v in box_mix)
        self.p_gt = self.box_mix[0] / (self.box_mix[0] + self.box_mix[1])   # 0.5 / 0.9 = 5/9
        self.gt_jitter = gt_jitter
        self.roi_margin = roi_margin
        self.roi_min = roi_min
        self.canvas = canvas
        self.scale_aug = scale_aug
        self.gain_aug = gain_aug
        self.train = train
        self.seed = seed
        self._rng = random.Random(seed) if seed is not None else None
        self._rng_key = None

        p = os.path.join(cache_dir, INDEX_NAME)
        assert os.path.exists(p), f"{p} missing; build the crop cache first (data_loader.roi_crops.build_crop_cache)"
        with open(p) as f:
            index = json.load(f)
        assert self.roi_arm in index['arms'], f"crop cache {cache_dir} has no ROIs of arm {self.roi_arm!r} (has {index['arms']})"
        # the cached object windows were grown so that every jittered expanded GT ROI fits (spec §3: x 2.0 >= x 1.5 x 1.3)
        assert roi_margin * (1 + 2 * gt_jitter) <= index['grow'] + 1e-9 and roi_min <= index['min_side'], \
            f"roi_margin {roi_margin} x (1 + 2 x {gt_jitter}) / roi_min {roi_min} exceed the cache's grow {index['grow']} / min_side {index['min_side']}"
        self.H, self.W = index['frame_hw']
        self.windows = [w for w in index['windows'] if w['split'] == split]
        self.obj_windows = [w for w in self.windows if w['kind'] == 'object']
        assert self.obj_windows, f"no {split} object windows in {cache_dir}"
        # false-positive ROIs of this arm: (window, roi entry); one fp window can carry several arms' entries
        self.fp_rois = [(w, e) for w in self.windows if w['kind'] == 'fp' for e in w['rois'][self.roi_arm]]
        if train:
            self.n_fp = n_fp_items(len(self.obj_windows), self.box_mix[2]) if self.fp_rois else 0
            self.items = None
        else:
            self.n_fp = 0
            self.items = []                                   # (window, source, roi entry or None), fixed order
            for w in self.obj_windows:
                self.items.append((w, 'gt', None))
                self.items += [(w, 'det', e) for e in w['rois'][self.roi_arm]]

    def __len__(self):
        return len(self.obj_windows) + self.n_fp if self.train else len(self.items)

    @property
    def rng(self):
        '''Same scheme as HyperCOD_data.rng: the random module (re-seeded per worker by PyTorch) or a seeded stream.'''
        if self._rng is None:
            return random
        info = torch.utils.data.get_worker_info()
        if info is not None and self._rng_key != info.seed:
            self._rng = random.Random(f"{self.seed}:{info.seed}")
            self._rng_key = info.seed
        return self._rng

    @staticmethod
    def object_box(w):
        '''GT box (frame px) of the window's own object.'''
        return next(o['box'] for o in w['objects'] if o['id'] == w['object'])

    def choose(self, idx):
        '''(window, source, roi entry or None) of item idx; random in training, fixed in validation.'''
        if not self.train:
            return self.items[idx]
        if idx >= len(self.obj_windows):                          # false-positive item, uniform over the arm's fp ROIs
            w, e = self.fp_rois[self.rng.randrange(len(self.fp_rois))]
            return w, 'fp', e
        w = self.obj_windows[idx]
        dets = w['rois'][self.roi_arm]
        if self.rng.random() < self.p_gt or not dets:             # expanded GT (also the fallback without a matched ROI)
            return w, 'gt', None
        return w, 'det', dets[self.rng.randrange(len(dets))]

    def item_boxes(self, w, source, e):
        '''(roi float xyxy, box float xyxy) in frame px: the ROI cut out and the pre-expansion box of the box channel.'''
        if source == 'gt':
            gt_box = self.object_box(w)
            box = jitter_gt_box(gt_box, self.gt_jitter, self.rng, self.H, self.W) if self.train else np.asarray(gt_box, np.float32)
            return expand_box(box, self.roi_margin, self.roi_min, self.H, self.W), box
        return np.asarray(e['roi'], np.float32), np.asarray(e['box'], np.float32)

    def __getitem__(self, idx):
        w, source, e = self.choose(idx)
        roi_f, box = self.item_boxes(w, source, e)
        wx1, wy1, wx2, wy2 = w['window']
        roi = pixel_box(roi_f, self.H, self.W)
        bpx = pixel_box(box, self.H, self.W)
        if source == 'gt':
            # the window was grown for the jittered ROI (asserted in __init__); clip only absorbs float rounding
            roi = (max(roi[0], wx1), max(roi[1], wy1), min(roi[2], wx2), min(roi[3], wy2))
        assert inside(roi, w['window']) and inside(bpx, roi), \
            f"frame {w['frame']} ({w['file']}): ROI {roi} / box {bpx} outside window {w['window']}"
        x1, y1, x2, y2 = roi
        h, wd = y2 - y1, x2 - x1

        # spec §9: a missing window raises (with the frame), never skips
        p_img, p_gt = os.path.join(self.cache_dir, w['file']), os.path.join(self.cache_dir, w['gt_file'])
        assert os.path.exists(p_img) and os.path.exists(p_gt), \
            f"window {w['file']} / {w['gt_file']} of frame {w['frame']} missing in {self.cache_dir}"
        cube = load_window(p_img)                                             # [n_bands, wh, ww] fp16, un-scaled
        crop = np.ascontiguousarray(cube[:, y1 - wy1:y2 - wy1, x1 - wx1:x2 - wx1])   # [n_bands, h, w] fp16
        # p99 scaling in fp16 through torch, exactly HyperCOD_data's rounding (so test frames match training crops)
        crop = scale_block(crop, w['scale'])                                  # [n_bands, h, w] fp16
        if source == 'fp':
            # spec §3: a false-positive box has an EMPTY target, even when its ROI grazes an object below min_cover
            m = np.zeros((h, wd), dtype=bool)                                 # [h, w]
        else:
            gt = np.load(p_gt)                                                # [wh, ww] bool
            m = gt[y1 - wy1:y2 - wy1, x1 - wx1:x2 - wx1]                      # [h, w] bool, GT union inside the ROI
        obj_area = int(m.sum())

        s_aug = self.rng.uniform(*self.scale_aug) if self.train else 1.0
        img, valid, offset, s = place_on_canvas(crop, self.canvas, s_aug)     # [C, c, c] fp16, [c, c] bool
        mask = place_on_canvas(m.astype(np.float32)[None], self.canvas, s_aug)[0]   # [1, c, c] float32 (soft if resized)
        box_canvas = box_to_canvas(box, roi, offset, (h, wd), s, self.canvas)
        box_map = rasterise_box(box_canvas, self.canvas)                      # [c, c] float32
        valid = valid.astype(np.float32)                                      # [c, c]
        aug = (False, False, 0)
        if self.train:
            aug = (self.rng.random() < 0.5, self.rng.random() < 0.5, self.rng.randrange(4))
            (img, mask, box_map, valid), box_canvas = flip_rot([img, mask, box_map, valid], box_canvas, *aug, self.canvas)
            if self.gain_aug > 0:
                g = np.array([self.rng.uniform(1 - self.gain_aug, 1 + self.gain_aug) for _ in range(img.shape[0])], np.float32)
                img = (img.astype(np.float32) * g[:, None, None]).astype(np.float16)   # [C, c, c], zero padding stays 0
        area = next((o['area'] for o in w['objects'] if o['id'] == w['object']), 0)
        meta = {'frame': w['frame'], 'roi': [float(v) for v in roi], 'box': [float(v) for v in box],
                'box_canvas': [float(v) for v in box_canvas], 'source': source, 'offset': (int(offset[0]), int(offset[1])),
                's': float(s), 'roi_hw': (int(h), int(wd)), 'obj_area': obj_area, 'object': int(w['object']),
                'area': int(area), 'aug': aug}
        return (np.ascontiguousarray(img, dtype=np.float16), np.ascontiguousarray(mask, dtype=np.float32),
                box_map[None], valid[None], meta)


def seg_collate_fn(batch):
    '''
    HyperCOD_roi items -> {"img" [B, C, c, c] fp16, "mask" / "box_map" / "valid" [B, 1, c, c] float32,
    "box_xyxy" [B, 4] float32 canvas px (the prompt box of SAM2BoxSeg), "meta": list of dicts}.
    '''
    img, mask, box_map, valid, meta = list(zip(*batch))
    return {'img': torch.from_numpy(np.stack(img, axis=0)),                      # [B, C, c, c] fp16
            'mask': torch.from_numpy(np.stack(mask, axis=0)),                    # [B, 1, c, c]
            'box_map': torch.from_numpy(np.stack(box_map, axis=0)),              # [B, 1, c, c]
            'valid': torch.from_numpy(np.stack(valid, axis=0)),                  # [B, 1, c, c]
            'box_xyxy': torch.tensor([m['box_canvas'] for m in meta], dtype=torch.float32).reshape(-1, 4),   # [B, 4]
            'meta': list(meta)}
