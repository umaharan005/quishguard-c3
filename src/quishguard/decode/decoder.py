"""Module B: safe QR decoding.

Finds, straightens and decodes a QR image and labels what kind of payload it
holds. This module never makes a network request: the decoded link is only
text from here on.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

try:  # optional second decoder
    from pyzbar import pyzbar as _pyzbar
except Exception:  # pragma: no cover - pyzbar missing or zbar DLL missing
    _pyzbar = None

ImageLike = Union[str, Path, np.ndarray, "PIL.Image.Image"]  # noqa: F821

PAD = 24  # CIC images have no white quiet zone; decoders need one


@dataclass
class DecodeResult:
    ok: bool
    text: Optional[str] = None
    payload_type: str = "none"
    method: Optional[str] = None
    corners: Optional[np.ndarray] = field(default=None, repr=False)  # 4x2, in padded image
    image: Optional[np.ndarray] = field(default=None, repr=False)    # padded grayscale


def to_gray(img: ImageLike) -> np.ndarray:
    """Load any supported input as a uint8 grayscale array."""
    if isinstance(img, (str, Path)):
        arr = cv2.imdecode(np.fromfile(str(img), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if arr is None:
            raise ValueError(f"Cannot read image: {img}")
        return arr
    if isinstance(img, np.ndarray):
        arr = img
    else:  # PIL image (mode "1" for CIC images)
        arr = np.array(img.convert("L"))
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    if arr.dtype == bool:
        arr = arr.astype(np.uint8) * 255
    return arr.astype(np.uint8)


def add_quiet_zone(gray: np.ndarray, pad: int = PAD) -> np.ndarray:
    return cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)


def _try_opencv(gray: np.ndarray):
    det = cv2.QRCodeDetector()
    text, pts, _ = det.detectAndDecode(gray)
    if text:
        return text, (pts.reshape(-1, 2) if pts is not None else None)
    return None, None


def _try_pyzbar(gray: np.ndarray):
    if _pyzbar is None:
        return None, None
    for r in _pyzbar.decode(gray, symbols=[_pyzbar.ZBarSymbol.QRCODE]):
        text = r.data.decode("utf-8", errors="replace")
        pts = np.array([(p.x, p.y) for p in r.polygon], dtype=np.float32)
        return text, (pts if len(pts) == 4 else None)
    return None, None


def decode(img: ImageLike) -> DecodeResult:
    """Decode a QR image. Tries OpenCV, then an upscaled copy, then pyzbar."""
    gray = add_quiet_zone(to_gray(img))
    attempts = [("opencv", gray)]
    if max(gray.shape) < 400:
        attempts.append(("opencv_x2", cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)))
    for name, im in attempts:
        text, pts = _try_opencv(im)
        if text:
            if pts is not None and name.endswith("_x2"):
                pts = pts / 2.0
            return DecodeResult(True, text, payload_type(text), name, pts, gray)
    text, pts = _try_pyzbar(gray)
    if text:
        return DecodeResult(True, text, payload_type(text), "pyzbar", pts, gray)
    return DecodeResult(False, image=gray)


_URL_RE = re.compile(
    r"^(?:[a-z][a-z0-9+.\-]*://)?"                    # optional scheme
    r"(?:\[?[0-9a-f:.]+\]?|[\w\-]+(?:\.[\w\-]+)+)"    # IP or dotted host
    r"(?::\d{1,5})?(?:[/?#].*)?$",
    re.IGNORECASE | re.DOTALL,
)


def payload_type(text: str) -> str:
    """url | wifi | payment | email | phone | sms | contact | geo | text"""
    t = (text or "").strip()
    low = t.lower()
    if not t:
        return "none"
    if low.startswith("wifi:"):
        return "wifi"
    if low.startswith(("upi://", "bitcoin:", "ethereum:", "litecoin:", "lightning:")) or "pay?" in low[:20]:
        return "payment"
    if low.startswith(("mailto:", "matmsg:")):
        return "email"
    if low.startswith("tel:"):
        return "phone"
    if low.startswith(("smsto:", "sms:", "mmsto:")):
        return "sms"
    if low.startswith(("begin:vcard", "mecard:", "begin:vevent")):
        return "contact"
    if low.startswith("geo:"):
        return "geo"
    if " " not in t and _URL_RE.match(t):
        return "url"
    return "text"


def rectify(result: DecodeResult, size: int = 224, margin: float = 0.06) -> Optional[np.ndarray]:
    """Warp the detected QR to a straight size x size square (input for the ViT)."""
    if result.image is None or result.corners is None or len(result.corners) != 4:
        return None
    src = result.corners.astype(np.float32)
    m = size * margin
    dst = np.array([[m, m], [size - m, m], [size - m, size - m], [m, size - m]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(result.image, H, (size, size), flags=cv2.INTER_AREA, borderValue=255)
