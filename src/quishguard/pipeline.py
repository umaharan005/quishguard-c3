"""The full C3 pipeline: one call from a QR image to an explainable alert.

    from quishguard.pipeline import QuishGuard
    qg = QuishGuard(models_dir="/content/drive/MyDrive/quishguard/models")
    alert = qg.scan("photo.jpg")            # dict, ready to send as JSON

Nothing here opens the decoded link. Live checks (WHOIS, redirects) are
Phase 8 and will run in a sandbox.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from quishguard import config
from quishguard.decode.decoder import ImageLike, decode
from quishguard.fusion.scoring import ACTIONS, TAMPER_FLOOR, W_URL, fuse


class QuishGuard:
    def __init__(self, models_dir: Path | str = config.MODELS_DIR, url_model: str = "url_model_v2.joblib",
                 visual_arch: str = "patchcore", w_url: float = W_URL, tamper_floor: float = TAMPER_FLOOR,
                 device: str | None = None):
        from quishguard.url_model.predict import UrlScorer
        from quishguard.vit.predict import VisualScorer
        models_dir = Path(models_dir)
        self.url = UrlScorer(models_dir / url_model)
        self.visual = VisualScorer(visual_arch, models_dir, device=device)
        self.w_url, self.tamper_floor = w_url, tamper_floor

    def scan(self, img: ImageLike, heatmap_path: Path | str | None = None, explain: bool = True) -> dict:
        t0 = time.perf_counter()
        d = decode(img)
        url_res = None
        if d.ok and d.payload_type == "url":
            url_res = self.url.score(d.text, explain=explain)
        vis = self.visual.score(img)
        f = fuse(url_res["url_score"] if url_res else None, vis["visual_score"], self.w_url, self.tamper_floor)

        reasons = []
        if not d.ok:
            reasons.append("The QR code could not be read normally (damaged or altered).")
        if url_res:
            reasons += [f"Link: {r['text']} = {r['value']} ({r['effect']})" for r in url_res.get("reasons", [])[:3]]
        if vis["tampered"]:
            r, c = vis["worst_patch_rowcol"]
            area = ["top", "middle", "bottom"][min(r // 5, 2)] + "-" + ["left", "centre", "right"][min(c // 5, 2)]
            reasons.append(f"Image: unusual pattern in the {area} of the code (possible sticker, overlay or edit).")

        if heatmap_path is not None:
            import cv2
            cv2.imwrite(str(heatmap_path), self.visual.heatmap_image(img, vis))

        return {
            "component": "QR",
            "score": f.score,
            "tier": f.tier,
            "recommended_action": ACTIONS[f.tier],
            "fusion_rule": f.rule,
            "streams": {"url": url_res["url_score"] if url_res else None, "visual": vis["visual_score"]},
            "decoded": {"ok": d.ok, "payload_type": d.payload_type, "text": d.text},
            "visual_tampered": bool(vis["tampered"]),
            "reasons": reasons,
            "heatmap": str(heatmap_path) if heatmap_path else None,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }
