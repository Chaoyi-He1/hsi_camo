import types
from functools import partial
import numpy as np
import torch

from train_eval.train_eval_det import random_flips, train_one_epoch, evaluate, false_colour, draw_boxes
from tests.test_ec_yolo import _tiny
from data_loader.boxes import det_collate_fn
from data_loader.my_dataset import HyperCOD_data
from util.logger import TrainLogger
from tests.conftest import H, W


def _loader(root, split='train'):
    ds = HyperCOD_data(str(root), split=split, use_filter=False, norm='p99', crop_size=0, out_dtype='float16', filter_norm='none')
    return torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False, num_workers=0, collate_fn=partial(det_collate_fn, min_area=10))


def test_random_flips_move_boxes_with_pixels():
    img = torch.zeros(2, 1, 8, 10); img[0, 0, 1, 2] = 1.0; img[1, 0, 6, 7] = 1.0
    bboxes = torch.tensor([[0.25, 0.1875, 0.1, 0.125], [0.75, 0.8125, 0.1, 0.125]])      # centred on the lit pixels
    g = torch.Generator().manual_seed(0)
    out, bb, flags = random_flips(img, bboxes, torch.tensor([0., 1.]), p=1.0, generator=g)   # p=1 -> both flips on every sample
    assert flags == [(True, True), (True, True)]
    assert out[0, 0, 6, 7] == 1.0 and out[1, 0, 1, 2] == 1.0
    torch.testing.assert_close(bb[0], torch.tensor([0.75, 0.8125, 0.1, 0.125])); torch.testing.assert_close(bb[1], torch.tensor([0.25, 0.1875, 0.1, 0.125]))
    out2, bb2, flags2 = random_flips(img, bboxes, torch.tensor([0., 1.]), p=0.0, generator=g)
    assert flags2 == [(False, False), (False, False)] and torch.equal(out2, img) and torch.equal(bb2, bboxes)


def test_train_and_evaluate_one_epoch_on_cpu(synthetic_root, tmp_path):
    root, _, _ = synthetic_root
    model = _tiny(6)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    logger = TrainLogger(types.SimpleNamespace(name='t', wandb=False, runs_dir=str(tmp_path / 'runs'), rank=0), cfg={})
    stats = train_one_epoch(model, _loader(root), opt, torch.device('cpu'), epoch=0, scaler=None, accumulate=2, logger=logger, print_freq=1)
    assert {'loss', 'box_loss', 'cls_loss', 'gate_entropy', 'contain_loss', 'lr'} <= set(stats) and np.isfinite(stats['loss'])
    summary = evaluate(model, _loader(root, 'test'), torch.device('cpu'), roi_margin=1.5, roi_min=0, min_area=10,
                       roi_conf=0.25, roi_topk=5, logger=logger, epoch=0, tag='val', amp=False)
    for k in ['recall50', 'ap50', 'coverage_raw', 'coverage_recall99_raw', 'coverage_roi', 'tightness', 'center_offset', 'dets_per_image']:
        assert k in summary
    for k in ['recall50', 'coverage_raw', 'coverage_recall99_raw', 'coverage_roi', 'coverage_recall99_roi',
              'contain_rate', 'tightness', 'center_offset', 'dets_per_image']:
        assert k + '_op' in summary                                   # the ROI operating point, threaded from cfg
    assert summary['dets_per_image_op'] <= 5 and summary['dets_per_image_op'] <= summary['dets_per_image']
    assert summary['n_images'] == 1 and summary['n_gt'] == 1
    logger.finish()


def test_false_colour_and_draw_boxes():
    y = torch.randn(5, H, W)
    img = false_colour(y); assert img.shape == (H, W, 3) and img.dtype == np.uint8
    out = draw_boxes(img, np.array([[20, 10, 30, 20]], np.float32), color=(255, 0, 0), width=1)
    assert out.shape == img.shape and (out[10, 20:30] == (255, 0, 0)).all()
    out2 = draw_boxes(img, np.array([[5, 5, 30, 30]], np.float32), color=(0, 255, 0), width=1, dashed=True)
    assert out2.shape == img.shape
