"""What happens when a photo arrives: check it, scan it, save it, raise the alert.

Kept separate from the web server so it can be tested without FastAPI.
"""
from __future__ import annotations

import json
import re
import threading
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from quishguard.app.safety import defang
from quishguard.app.store import ScanStore

MAX_BYTES = 15 * 1024 * 1024          # a phone photo is 2-6 MB
SAFE_NAME = re.compile(r"^[0-9a-f]{12}(_heat\.png|\.jpg)$")


class BadImage(ValueError):
    pass


class ScanService:
    def __init__(self, scanner, store: ScanStore, scans_dir: Path | str, webhook_url: str | None = None):
        self.scanner = scanner                    # QuishGuard (or a fake one in tests)
        self.store = store
        self.scans_dir = Path(scans_dir)
        self.scans_dir.mkdir(parents=True, exist_ok=True)
        self.webhook_url = webhook_url
        self._lock = threading.Lock()             # one scan at a time: the model is not shared-safe

    def handle(self, data: bytes, device: str = "") -> dict:
        if not data:
            raise BadImage("No image received.")
        if len(data) > MAX_BYTES:
            raise BadImage("Image is larger than 15 MB.")
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise BadImage("This file is not an image we can read (use JPG or PNG).")

        scan_id = uuid.uuid4().hex[:12]
        heat = self.scans_dir / f"{scan_id}_heat.png"
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        with self._lock:
            result = self.scanner.scan(gray, heatmap_path=heat, explain=True, photo=True)
        _save_small(img, self.scans_dir / f"{scan_id}.jpg")

        result = dict(result)
        result["scan_id"] = scan_id
        result["decoded"] = dict(result.get("decoded") or {})
        result["decoded"]["text_defanged"] = defang(result["decoded"].get("text"))
        result["heatmap"] = f"/files/{scan_id}_heat.png" if heat.exists() else None
        result["photo"] = f"/files/{scan_id}.jpg"
        row = self.store.add(scan_id, result, device)
        if row["is_alert"] and self.webhook_url:
            threading.Thread(target=_post, args=(self.webhook_url, row), daemon=True).start()
        return row

    def file_path(self, name: str) -> Path | None:
        """Only files this app wrote (12-hex id), so no path tricks like ../../."""
        if not SAFE_NAME.match(name):
            return None
        p = self.scans_dir / name
        return p if p.exists() else None


def _save_small(img: np.ndarray, path: Path, max_side: int = 1200):
    h, w = img.shape[:2]
    f = min(1.0, max_side / max(h, w))
    if f < 1:
        img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 85])


def _post(url: str, row: dict):
    """Send the alert to the group's Final Aggregator (or any webhook). Best effort."""
    body = json.dumps({"source": "C3-QR", "sent_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                       **row}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        urllib.request.urlopen(req, timeout=3).close()
    except Exception as e:  # the portal still shows the alert even if the aggregator is down
        print("alert webhook failed:", e)
