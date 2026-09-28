import numpy as np
import torch
from scipy import ndimage


def boxes_from_mask(mask, min_area=100, return_labels=False):
    '''
    One box per connected foreground component with area >= min_area (JPEG ringing specks are smaller than 100 px).
    Returns float32 [K, 4] xyxy with x2/y2 exclusive, ordered by component label (row-major first pixel);
    with return_labels also the label map [H, W] int32 and the kept label ids.
    '''
    labels, n = ndimage.label(mask)
    boxes, ids = [], []
    if n > 0:
        sizes = ndimage.sum(mask, labels, index=np.arange(1, n + 1))
        for i, sl in enumerate(ndimage.find_objects(labels), start=1):
            if sl is None or sizes[i - 1] < min_area:
                continue
            boxes.append([sl[1].start, sl[0].start, sl[1].stop, sl[0].stop])
            ids.append(i)
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    return (boxes, labels.astype(np.int32), ids) if return_labels else boxes


def boxes_to_yolo(boxes, H, W):
    '''xyxy pixels -> normalised (cx, cy, w, h) as the ultralytics loss expects.'''
    b = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    out = np.empty_like(b)
    out[:, 0] = (b[:, 0] + b[:, 2]) / 2 / W; out[:, 1] = (b[:, 1] + b[:, 3]) / 2 / H
    out[:, 2] = (b[:, 2] - b[:, 0]) / W;     out[:, 3] = (b[:, 3] - b[:, 1]) / H
    return out


def yolo_to_xyxy(b, H, W):
    b = np.asarray(b, dtype=np.float32).reshape(-1, 4)
    cx, cy, w, h = b[:, 0] * W, b[:, 1] * H, b[:, 2] * W, b[:, 3] * H
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


def flip_boxes(boxes, H, W, horizontal, vertical):
    b = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
    if horizontal:
        b[:, [0, 2]] = W - b[:, [2, 0]]
    if vertical:
        b[:, [1, 3]] = H - b[:, [3, 1]]
    return b


def expand_box(box, margin, min_size, H, W):
    '''Grow a box about its centre by `margin` (1.5 = 50 % larger sides), at least min_size on each side, clipped to the frame.'''
    x1, y1, x2, y2 = [float(v) for v in box]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = max((x2 - x1) * margin, min_size), max((y2 - y1) * margin, min_size)
    return np.array([max(0.0, cx - w / 2), max(0.0, cy - h / 2), min(float(W), cx + w / 2), min(float(H), cy + h / 2)], dtype=np.float32)


def det_collate_fn(batch, min_area=100):
    '''
    HyperCOD_data (img [C, H, W], gt [1, H, W], name) tuples -> the batch dict ultralytics' model.loss reads
    (img, batch_idx, cls, bboxes normalised cx,cy,w,h) plus the masks and pixel boxes for the metrics.
    Use functools.partial(det_collate_fn, min_area=...) as the DataLoader collate_fn to change the box floor.
    '''
    imgs, gts, names = list(zip(*batch))
    img = torch.from_numpy(np.stack(imgs, axis=0))                        # [B, C, H, W], dataset dtype
    H, W = img.shape[-2:]
    masks, boxes_xyxy, batch_idx, bboxes = [], [], [], []
    for i, gt in enumerate(gts):
        m = gt[0] > 0.5                                                   # [H, W] bool
        b = boxes_from_mask(m, min_area=min_area)                         # [K, 4]
        masks.append(m); boxes_xyxy.append(b)
        batch_idx.append(np.full(len(b), i, dtype=np.float32)); bboxes.append(boxes_to_yolo(b, H, W))
    batch_idx = torch.from_numpy(np.concatenate(batch_idx)) if batch_idx else torch.zeros(0)
    bboxes = torch.from_numpy(np.concatenate(bboxes)).reshape(-1, 4) if bboxes else torch.zeros(0, 4)
    return {'img': img, 'masks': masks, 'batch_idx': batch_idx, 'cls': torch.zeros(len(batch_idx), 1),
            'bboxes': bboxes, 'boxes_xyxy': boxes_xyxy, 'names': list(names)}
