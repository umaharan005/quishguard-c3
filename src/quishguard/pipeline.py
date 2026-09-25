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
from quishguard.decode.decoder import ImageLike, clean_photo, decode, decode_photo, locate, to_gray
from quishguard.fusion.scoring import ACTIONS, TAMPER_FLOOR, W_URL, fuse


class QuishGuard:
    def __init__(self, models_dir: Path | str = config.MODELS_DIR, url_model: str = "url_model_v2.joblib",
                 visual_arch: str = "patchcore", w_url: float = W_URL, tamper_floor: float = TAMPER_FLOOR,
                 device: str | None = None, photo_clean: bool = False,
                 photo_visual: str = "advisory", tranco_csv: Path | str | None = None):
        from quishguard.url_model.predict import UrlScorer
        from quishguard.vit.predict import VisualScorer
        models_dir = Path(models_dir)
        self.url = UrlScorer(models_dir / url_model)
        self.visual = VisualScorer(visual_arch, models_dir, device=device)
        self.w_url, self.tamper_floor = w_url, tamper_floor
        self.photo_clean = photo_clean      # binarise real photos before the visual model
        # "advisory": for real camera photos the image check is shown (score, heatmap) but does not
        # drive the tier, because it is not yet validated on real-world codes (designer codes,
        # posters, screens score as unusual). "full": use it like on the test sets.
        self.photo_visual = photo_visual
        self.reputation = None
        if tranco_csv and Path(tranco_csv).exists():
            from quishguard.url_model.reputation import Reputation
            self.reputation = Reputation(tranco_csv)

    def scan(self, img: ImageLike, heatmap_path: Path | str | None = None, explain: bool = True,
             photo: bool = False) -> dict:
        """photo=True for real camera photos: shrink the photo, find the code in it and
        give the visual model only the code (not the table or wall around it)."""
        t0 = time.perf_counter()
        located = None
        if photo:
            d = decode_photo(to_gray(img))
            located = locate(d)
            if located is None:
                return {"component": "QR", "no_code": True, "decoded": {"ok": False, "payload_type": "none", "text": None},
                        "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
            vis_img = clean_photo(located) if self.photo_clean else located
        else:
            d = decode(img)
            vis_img = img
        url_res, rep_reason, rep_rank = None, None, None
        if d.ok and d.payload_type == "url":
            url_res = self.url.score(d.text, explain=explain)
            if self.reputation is not None:
                capped, rep_reason, rep_rank = self.reputation.adjust(d.text, url_res["url_score"])
                url_res = {**url_res, "url_score_model": url_res["url_score"], "url_score": capped}
        vis = self.visual.score(vis_img)
        advisory = photo and self.photo_visual == "advisory"
        if advisory:
            if url_res:
                # one confirmed signal (the link) -> at most Phishing, as in the tier rule
                f = fuse(min(url_res["url_score"], 75.0), vis["visual_score"], 1.0, 0.0)
                f.rule = "link check, max Phishing (image check is advisory for camera photos)"
            else:   # nothing to check but the image: at most Suspicious, ask for a manual look
                f = fuse(None, min(vis["visual_score"], 50.0))
                f.rule = "image check only, capped at Suspicious (advisory for camera photos)"
        else:
            f = fuse(url_res["url_score"] if url_res else None, vis["visual_score"], self.w_url, self.tamper_floor)

        reasons = []
        if not d.ok:
            reasons.append("The QR code could not be read normally (damaged or altered).")
        if url_res:
            if rep_reason:
                reasons.append(rep_reason)
            reasons += [f"Link: {r['text']} = {r['value']} ({r['effect']})" for r in url_res.get("reasons", [])[:3]]
        if vis["tampered"]:
            r, c = vis["worst_patch_rowcol"]
            area = ["top", "middle", "bottom"][min(r // 5, 2)] + "-" + ["left", "centre", "right"][min(c // 5, 2)]
            reasons.append(f"Image{' (advisory)' if advisory else ''}: unusual pattern in the {area} of the code "
                           "(possible sticker, overlay or edit).")

        if heatmap_path is not None:
            import cv2
            cv2.imwrite(str(heatmap_path), self.visual.heatmap_image(vis_img, vis))

        return {
            "component": "QR",
            "score": f.score,
            "tier": f.tier,
            "recommended_action": ACTIONS[f.tier],
            "fusion_rule": f.rule,
            "streams": {"url": url_res["url_score"] if url_res else None, "visual": vis["visual_score"]},
            "decoded": {"ok": d.ok, "payload_type": d.payload_type, "text": d.text},
            "visual_tampered": bool(vis["tampered"]),
            "visual_mode": "advisory" if advisory else "full",
            "tranco_rank": rep_rank,
            "url_score_model": url_res.get("url_score_model", url_res["url_score"]) if url_res else None,
            "reasons": reasons,
            "heatmap": str(heatmap_path) if heatmap_path else None,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }
