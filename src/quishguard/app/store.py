"""Scan history + alerts in one SQLite file (data/app/scans.db, never in git)."""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

ALERT_TIERS = ("Phishing", "Critical")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS scans (
    seq        INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id    TEXT UNIQUE NOT NULL,
    at_utc     TEXT NOT NULL,
    device     TEXT,
    tier       TEXT NOT NULL,
    score      REAL NOT NULL,
    is_alert   INTEGER NOT NULL,
    status     TEXT NOT NULL,          -- 'new' | 'acknowledged' | 'none' (not an alert)
    result     TEXT NOT NULL           -- the full alert JSON
);
CREATE INDEX IF NOT EXISTS ix_scans_alert ON scans(is_alert, status);
"""


class ScanStore:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(_SCHEMA)

    def add(self, scan_id: str, result: dict, device: str = "") -> dict:
        is_alert = result["tier"] in ALERT_TIERS
        row = {"scan_id": scan_id, "at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "device": (device or "")[:60], "tier": result["tier"], "score": float(result["score"]),
               "is_alert": int(is_alert), "status": "new" if is_alert else "none",
               "result": json.dumps(result)}
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO scans (scan_id, at_utc, device, tier, score, is_alert, status, result) "
                "VALUES (:scan_id, :at_utc, :device, :tier, :score, :is_alert, :status, :result)", row)
            self._db.commit()
            row["seq"] = cur.lastrowid
        return self._out(row)

    def since(self, seq: int = 0, limit: int = 200) -> list[dict]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM scans WHERE seq > ? ORDER BY seq DESC LIMIT ?",
                                    (int(seq), int(limit))).fetchall()
        return [self._out(dict(r)) for r in rows]

    def get(self, seq: int) -> dict | None:
        with self._lock:
            r = self._db.execute("SELECT * FROM scans WHERE seq = ?", (int(seq),)).fetchone()
        return self._out(dict(r)) if r else None

    def acknowledge(self, seq: int) -> bool:
        with self._lock:
            cur = self._db.execute("UPDATE scans SET status = 'acknowledged' WHERE seq = ? AND is_alert = 1",
                                   (int(seq),))
            self._db.commit()
        return cur.rowcount == 1

    def stats(self) -> dict:
        with self._lock:
            tiers = dict(self._db.execute("SELECT tier, COUNT(*) FROM scans GROUP BY tier").fetchall())
            open_alerts = self._db.execute("SELECT COUNT(*) FROM scans WHERE is_alert = 1 AND status = 'new'").fetchone()[0]
            total = self._db.execute("SELECT COUNT(*) FROM scans").fetchone()[0]
        return {"total": total, "open_alerts": open_alerts,
                "tiers": {t: tiers.get(t, 0) for t in ("Safe", "Suspicious", "Phishing", "Critical")}}

    @staticmethod
    def _out(row: dict) -> dict:
        res = row["result"] if isinstance(row["result"], dict) else json.loads(row["result"])
        return {"seq": row["seq"], "scan_id": row["scan_id"], "at_utc": row["at_utc"], "device": row["device"],
                "tier": row["tier"], "score": row["score"], "is_alert": bool(row["is_alert"]),
                "status": row["status"], "result": res}
