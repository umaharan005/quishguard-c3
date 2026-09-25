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

try:  # optional third decoder (strongest on real photos and designer codes): pip install zxing-cpp
    import zxingcpp as _zx
except Exception:  # pragma: no cover
    _zx = None

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
    try:
        text, pts, _ = det.detectAndDecode(gray)
    except cv2.error:
        # Some OpenCV builds (e.g. 5.0 on Colab) raise an internal assertion on
        # damaged/tampered codes instead of returning "not found". Treat as unreadable.
        return None, None
    if text:
        return text, (pts.reshape(-1, 2) if pts is not None else None)
    return None, None


def _try_pyzbar(gray: np.ndarray):
    if _pyzbar is None:
        return None, None
    try:
        results = _pyzbar.decode(gray, symbols=[_pyzbar.ZBarSymbol.QRCODE])
    except Exception:
        return None, None
    for r in results:
        text = r.data.decode("utf-8", errors="replace")
        pts = np.array([(p.x, p.y) for p in r.polygon], dtype=np.float32)
        return text, (pts if len(pts) == 4 else None)
    return None, None


def _try_zxing(gray: np.ndarray):
    if _zx is None:
        return None, None
    try:
        results = _zx.read_barcodes(gray, formats=_zx.BarcodeFormat.QRCode)
    except Exception:
        return None, None
    for r in results:
        if r.text:
            p = r.position
            pts = np.array([[p.top_left.x, p.top_left.y], [p.top_right.x, p.top_right.y],
                            [p.bottom_right.x, p.bottom_right.y], [p.bottom_left.x, p.bottom_left.y]], np.float32)
            return r.text, pts
    return None, None


def decode(img: ImageLike) -> DecodeResult:
    """Decode a QR image. Tries OpenCV, then an upscaled copy, then pyzbar, then zxing-cpp."""
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
    text, pts = _try_zxing(gray)
    if text:
        return DecodeResult(True, text, payload_type(text), "zxing", pts, gray)
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


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Top-left, top-right, bottom-right, bottom-left (any decoder's order in)."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def locate(result: DecodeResult, size: int = 448, margin: float = 0.10) -> Optional[np.ndarray]:
    """Cut the QR code out of a larger photo (for the visual model).

    Phone photos contain a table, a wall, a hand... that the visual model never
    saw in training. This warps the code to a straight square with a 10% border,
    so a sticker that spills slightly past the code is still inside the crop.
    Uses the decoder's corners; if the code could not be decoded, tries OpenCV's
    detector alone (it can often find a code it cannot read). None = not found.
    """
    if result.image is None:
        return None
    pts = result.corners
    if pts is None or len(pts) != 4:
        try:
            found, p = cv2.QRCodeDetector().detect(result.image)
        except cv2.error:
            found, p = False, None
        if not found or p is None:
            return None
        pts = p.reshape(-1, 2)
    src = _order_corners(pts)
    if cv2.contourArea(src) < 400:        # too small to be a real detection
        return None
    m = size * margin
    dst = np.array([[m, m], [size - m, m], [size - m, size - m], [m, size - m]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(result.image, H, (size, size), flags=cv2.INTER_AREA, borderValue=255)


def downscale(gray: np.ndarray, max_side: int = 1600) -> np.ndarray:
    """Phone photos are 3000-4000 px; decoding and scoring do not need that."""
    h, w = gray.shape[:2]
    f = max_side / max(h, w)
    return gray if f >= 1 else cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)


def photo_candidates(gray: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Versions of a camera photo to try, from cheapest to most processed.
    Screens and prints give glare, moire and soft edges; one of these usually reads."""
    g = downscale(gray, 1600)
    out = [("photo", g)]
    for side in (1000, 700):
        if max(g.shape) > side:
            out.append((f"photo_{side}", downscale(g, side)))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(downscale(g, 1000))
    out.append(("photo_clahe", clahe))
    blur = cv2.GaussianBlur(downscale(g, 1000), (5, 5), 0)
    out.append(("photo_otsu", cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]))
    out.append(("photo_adaptive", cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                         cv2.THRESH_BINARY, 51, 10)))
    return out


def decode_photo(gray: np.ndarray) -> DecodeResult:
    """decode() for real camera photos: tries several cleaned-up versions.
    If none can be read, the result still carries the image where a code was
    FOUND (not read), so the visual model can check it; image=None means no code at all."""
    cands = photo_candidates(gray)
    for name, c in cands:
        r = decode(c)
        if r.ok:
            r.method = f"{name}+{r.method}"
            return r
    for name, c in cands:                       # nothing readable: is there a code at all?
        r = DecodeResult(False, image=add_quiet_zone(c), method=name)
        crop = locate(r)
        if crop is not None:
            # a straightened crop often reads when the whole photo did not (curved posters,
            # rounded-dot and logo designs); keep r.image/corners so locate() still works
            blur = cv2.GaussianBlur(crop, (3, 3), 0)
            for tag, cc in (("crop", crop), ("crop_otsu", cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
                            ("crop_small", cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA))):
                d = decode(cc)
                if d.ok:
                    r.ok, r.text, r.payload_type, r.method = True, d.text, d.payload_type, f"{name}+{tag}+{d.method}"
                    break
            return r
    return DecodeResult(False, image=None)


def clean_photo(crop: np.ndarray) -> np.ndarray:
    """Turn a straightened photo of a code into a clean black-and-white image, like the
    computer-made codes the visual model learned from (removes glare, grey paper, screen
    moire). Stickers and edits stay visible: they change the black/white pattern itself."""
    blur = cv2.GaussianBlur(crop, (3, 3), 0)
    return cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
