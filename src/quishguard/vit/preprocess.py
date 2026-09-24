"""Turn any QR image (dataset PNG, tampered image, phone photo) into the
ViT input: a 224 x 224 grayscale float image in [0, 1].

The SAME steps are used for training images, tamper images and real scans,
so the model can never tell them apart by preprocessing alone:
  1. grayscale
  2. crop to the dark content (the code plus anything stuck on it)
  3. add a fixed 6% white margin
  4. resize to 224 x 224
"""
from __future__ import annotations

import cv2
import numpy as np

from quishguard.decode.decoder import ImageLike, to_gray

SIZE = 224
MARGIN = 0.06


def crop_to_content(gray: np.ndarray, thresh: int = 160) -> np.ndarray:
    small = cv2.GaussianBlur(gray, (3, 3), 0)
    ys, xs = np.where(small < thresh)
    if len(ys) < 20:  # blank image: keep as is
        return gray
    return gray[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def prepare(img: ImageLike, size: int = SIZE) -> np.ndarray:
    g = crop_to_content(to_gray(img))
    h, w = g.shape
    side = max(h, w)
    pad = int(round(side * MARGIN))
    # make square (centre the content) then add the margin
    top = (side - h) // 2 + pad
    left = (side - w) // 2 + pad
    g = cv2.copyMakeBorder(g, top, side - h - (side - h) // 2 + pad,
                           left, side - w - (side - w) // 2 + pad,
                           cv2.BORDER_CONSTANT, value=255)
    g = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)
    return g.astype(np.float32) / 255.0
