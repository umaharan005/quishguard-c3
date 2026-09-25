import io
import socket

import numpy as np
import pytest

from quishguard.decode.decoder import decode, payload_type, rectify

import cv2

try:
    import segno
except ImportError:  # fall back to OpenCV's own encoder
    segno = None


def make_qr(text: str, scale: int = 6) -> np.ndarray:
    """A QR image like the CIC ones: no white quiet zone around it."""
    if segno is not None:
        buf = io.BytesIO()
        segno.make(text, error="m").save(buf, kind="png", scale=scale, border=0)
        return cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), cv2.IMREAD_GRAYSCALE)
    img = cv2.QRCodeEncoder.create().encode(text)  # has a quiet zone: crop it off
    ys, xs = np.where(img < 128)
    img = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)


@pytest.mark.parametrize("text", [
    "http://example.com/login",
    "paypal.com.account-verify.xyz/signin?id=123",
    "WIFI:S:CafeNet;T:WPA;P:secret;;",
])
def test_decodes_codes_without_quiet_zone(text):
    r = decode(make_qr(text))
    assert r.ok and r.text == text


def test_rectify_gives_square():
    r = decode(make_qr("https://sliit.lk"))
    out = rectify(r, size=224)
    assert out is not None and out.shape == (224, 224)


def test_decode_never_uses_network(monkeypatch):
    def blocked(*a, **k):
        raise AssertionError("decoder tried to use the network")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    assert decode(make_qr("http://malicious.example/x")).ok


def test_payload_types():
    assert payload_type("http://a.com") == "url"
    assert payload_type("192.168.1.1:8080/x") == "url"
    assert payload_type("WIFI:S:x;T:WPA;P:y;;") == "wifi"
    assert payload_type("upi://pay?pa=shop@bank") == "payment"
    assert payload_type("BEGIN:VCARD\nFN:A\nEND:VCARD") == "contact"
    assert payload_type("hello world") == "text"


def test_blank_image_fails_cleanly():
    r = decode(np.full((200, 200), 255, np.uint8))
    assert not r.ok and r.payload_type == "none"


def test_opencv_internal_error_is_treated_as_unreadable(monkeypatch):
    class Boom:
        def detectAndDecode(self, *_):
            raise cv2.error("simulated OpenCV 5.0 assertion")
    monkeypatch.setattr(cv2, "QRCodeDetector", lambda: Boom())
    r = decode(np.full((120, 120), 255, np.uint8))
    assert not r.ok


def test_locate_cuts_code_out_of_a_photo():
    from quishguard.decode.decoder import downscale, locate
    qr = make_qr("https://www.sliit.lk/", scale=8)
    qr = cv2.copyMakeBorder(qr, 32, 32, 32, 32, cv2.BORDER_CONSTANT, value=255)   # printed quiet zone
    photo = np.full((1800, 2400), 120, np.uint8)                                   # grey table
    cv2.putText(photo, "MENU", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 8, 30, 20)    # clutter
    h, w = qr.shape
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[900, 600], [1500, 650], [1480, 1250], [880, 1200]])        # slight tilt
    warped = cv2.warpPerspective(qr, cv2.getPerspectiveTransform(src, dst), (2400, 1800), borderValue=0)
    mask = cv2.warpPerspective(np.full_like(qr, 255), cv2.getPerspectiveTransform(src, dst), (2400, 1800))
    photo = np.where(mask > 0, warped, photo).astype(np.uint8)
    r = decode(downscale(photo))
    assert r.ok and r.text == "https://www.sliit.lk/"
    crop = locate(r)
    assert crop is not None and crop.shape == (448, 448)
    assert decode(crop).text == "https://www.sliit.lk/"      # the crop is the code, straightened
    assert (crop[:30] > 200).mean() > 0.9                     # border is white, not the table
