import random

import cv2
import numpy as np

from quishguard.decode.decoder import decode
from quishguard.tamper.generate import (attack_logo, attack_sticker, photo_effects, recrop)


def qr(text, scale=6):
    img = cv2.QRCodeEncoder.create().encode(text)
    ys, xs = np.where(img < 128)
    img = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)


def test_normal_effects_keep_code_readable():
    rng = random.Random(0)
    ok = sum(decode(photo_effects(recrop(qr("https://sliit.lk/x"), rng), rng)).ok for _ in range(20))
    assert ok >= 18


def test_sticker_changes_decoded_url():
    texts = [decode(attack_sticker(qr("https://real-parking.lk/pay"), qr("http://evil.top/pay"),
                                   random.Random(s))[0]).text for s in range(20)]
    assert texts.count("http://evil.top/pay") >= 10      # scanner now opens the attacker's link
    assert "https://real-parking.lk/pay" not in texts


def test_logo_output_shape():
    rng = random.Random(2)
    base = qr("https://example.com/abc")
    t, info = attack_logo(base, rng)
    assert t.shape == base.shape and 0.3 <= info["logo_frac"] <= 0.55
