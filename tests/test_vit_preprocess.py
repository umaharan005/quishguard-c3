import random

import cv2
import numpy as np

from quishguard.tamper.generate import photo_effects, recrop
from quishguard.vit.preprocess import SIZE, prepare


def qr(text, scale=6):
    img = cv2.QRCodeEncoder.create().encode(text)
    ys, xs = np.where(img < 128)
    img = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)


def test_output_shape_and_range():
    x = prepare(qr("https://sliit.lk"))
    assert x.shape == (SIZE, SIZE) and x.dtype == np.float32
    assert 0.0 <= x.min() and x.max() <= 1.0


def test_extra_white_border_does_not_change_input():
    a = prepare(qr("https://example.com/x"))
    b = prepare(cv2.copyMakeBorder(qr("https://example.com/x"), 60, 60, 60, 60, cv2.BORDER_CONSTANT, value=255))
    assert np.abs(a - b).mean() < 0.02  # same framing whatever the original margin


def test_photo_image_is_handled():
    rng = random.Random(0)
    x = prepare(photo_effects(recrop(qr("https://example.com/y"), rng), rng))
    assert x.shape == (SIZE, SIZE)
