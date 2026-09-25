"""Phones that have the scan page open (for the portal's "Connected devices" panel).

The phone page sends a small heartbeat every few seconds. A phone counts as connected
while heartbeats arrive. The analyst can disconnect a phone from the portal; its scans
are then refused until the person on the phone taps "Reconnect".
Nothing here is stored on disk.
"""
from __future__ import annotations

import re
import threading
import time

ONLINE_SECONDS = 20
_ID = re.compile(r"^[A-Za-z0-9_-]{6,40}$")


def name_from_user_agent(ua: str) -> str:
    ua = ua or ""
    if "iPhone" in ua:
        return "iPhone"
    if "iPad" in ua:
        return "iPad"
    if "Android" in ua:
        m = re.search(r"Android [\d.]+; ([^;)]+?)(?: Build|\))", ua)
        model = m.group(1).strip() if m else ""
        return f"Android · {model}" if model and model.upper() != "K" else "Android phone"
    if "Windows" in ua:
        return "Windows browser"
    if "Macintosh" in ua:
        return "Mac browser"
    return "Browser"


class DeviceRegistry:
    def __init__(self):
        self._lock = threading.Lock()
        self._dev: dict[str, dict] = {}

    @staticmethod
    def valid(device_id: str) -> bool:
        return bool(device_id and _ID.match(device_id))

    def ping(self, device_id: str, user_agent: str, reconnect: bool = False) -> dict:
        now = time.time()
        with self._lock:
            d = self._dev.setdefault(device_id, {"id": device_id, "first_seen": now, "scans": 0,
                                                 "disconnected": False})
            d["name"] = name_from_user_agent(user_agent)
            d["last_seen"] = now
            if reconnect:
                d["disconnected"] = False
            return {"connected": not d["disconnected"], "name": d["name"]}

    def allowed(self, device_id: str) -> bool:
        with self._lock:
            d = self._dev.get(device_id)
            return not (d and d["disconnected"])

    def count_scan(self, device_id: str, user_agent: str) -> str:
        self.ping(device_id, user_agent)
        with self._lock:
            self._dev[device_id]["scans"] += 1
            return self._dev[device_id]["name"]

    def disconnect(self, device_id: str) -> bool:
        with self._lock:
            d = self._dev.get(device_id)
            if not d:
                return False
            d["disconnected"] = True
            return True

    def list(self) -> list[dict]:
        now = time.time()
        with self._lock:
            out = []
            for d in self._dev.values():
                online = (now - d["last_seen"]) < ONLINE_SECONDS
                state = "disconnected" if d["disconnected"] else ("connected" if online else "offline")
                out.append({"id": d["id"], "name": d["name"], "state": state, "scans": d["scans"],
                            "last_seen_s": round(now - d["last_seen"])})
        return sorted(out, key=lambda x: (x["state"] != "connected", x["last_seen_s"]))
