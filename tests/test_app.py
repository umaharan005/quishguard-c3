"""Scan app logic without the web server or the trained models."""
import cv2
import numpy as np
import pytest

from quishguard.app.safety import defang
from quishguard.app.service import BadImage, ScanService
from quishguard.app.store import ScanStore


class FakeScanner:
    def __init__(self, tier="Critical", score=91.0):
        self.tier, self.score_value, self.calls = tier, score, []

    def scan(self, img, heatmap_path=None, explain=True, photo=False):
        self.calls.append(photo)
        if heatmap_path:
            cv2.imwrite(str(heatmap_path), np.zeros((8, 8, 3), np.uint8))
        return {"component": "QR", "score": self.score_value, "tier": self.tier,
                "decoded": {"ok": True, "payload_type": "url", "text": "http://bad.example.xyz/login"},
                "reasons": ["Link: test"]}


def jpg():
    ok, buf = cv2.imencode(".jpg", np.full((60, 80, 3), 200, np.uint8))
    return buf.tobytes()


def test_defang():
    assert defang("https://login.example.com/a.b") == "hxxps://login[.]example[.]com/a.b"
    assert defang("WIFI:S:x;;") == "WIFI:S:x;;"
    assert defang(None) == ""


def test_scan_is_stored_and_alert_raised(tmp_path):
    svc = ScanService(FakeScanner(), ScanStore(tmp_path / "s.db"), tmp_path / "scans")
    row = svc.handle(jpg(), device="phone")
    assert row["is_alert"] and row["status"] == "new" and row["tier"] == "Critical"
    assert row["result"]["decoded"]["text_defanged"] == "hxxp://bad[.]example[.]xyz/login"
    assert svc.scanner.calls == [True]                       # scanned as a real photo
    assert svc.file_path(row["scan_id"] + ".jpg") is not None
    assert svc.file_path(row["scan_id"] + "_heat.png") is not None
    st = svc.store.stats()
    assert st["open_alerts"] == 1 and st["tiers"]["Critical"] == 1
    assert svc.store.acknowledge(row["seq"]) and svc.store.stats()["open_alerts"] == 0


def test_safe_scan_is_not_an_alert(tmp_path):
    svc = ScanService(FakeScanner("Safe", 8.0), ScanStore(tmp_path / "s.db"), tmp_path / "scans")
    row = svc.handle(jpg())
    assert not row["is_alert"] and row["status"] == "none"
    assert not svc.store.acknowledge(row["seq"])             # nothing to acknowledge


def test_since_returns_only_new_scans(tmp_path):
    svc = ScanService(FakeScanner(), ScanStore(tmp_path / "s.db"), tmp_path / "scans")
    a = svc.handle(jpg()); b = svc.handle(jpg())
    assert [r["seq"] for r in svc.store.since(a["seq"])] == [b["seq"]]


@pytest.mark.parametrize("data", [b"", b"not an image", b"x" * (16 * 1024 * 1024)])
def test_bad_uploads_rejected(tmp_path, data):
    svc = ScanService(FakeScanner(), ScanStore(tmp_path / "s.db"), tmp_path / "scans")
    with pytest.raises(BadImage):
        svc.handle(data)


def test_file_names_cannot_escape_scan_folder(tmp_path):
    svc = ScanService(FakeScanner(), ScanStore(tmp_path / "s.db"), tmp_path / "scans")
    for name in ["../s.db", "..\\s.db", "abc.jpg", "/etc/passwd", "0123456789ab.py"]:
        assert svc.file_path(name) is None
