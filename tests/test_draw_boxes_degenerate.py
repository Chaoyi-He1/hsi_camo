import numpy as np

from train_eval.train_eval_det import draw_boxes


def test_draw_boxes_tolerates_degenerate_and_inverted_boxes():
    # raw133_A crashed in its first eval: PIL's rectangle() raises when x1 > x0, which a sub-pixel or inverted
    # detection box from an early epoch produces after the x2 - 1 shift.
    img = np.zeros((40, 50, 3), np.uint8)
    boxes = np.array([[30, 20, 10, 10],       # inverted
                      [5, 5, 5, 9],           # zero width
                      [7, 7, 7.4, 7.2],       # sub-pixel
                      [0, 0, 50, 40]], np.float32)
    out = draw_boxes(img, boxes, (255, 0, 0), width=1)
    assert out.shape == img.shape and out.dtype == np.uint8 and out.max() == 255
    out2 = draw_boxes(img, boxes, (0, 255, 0), width=1, dashed=True)
    assert out2.shape == img.shape
