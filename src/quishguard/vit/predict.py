"""Score one QR image with the trained visual model: 0-100 score + heatmap."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import torch

from quishguard import config
from quishguard.decode.decoder import ImageLike
from quishguard.vit.model import anomaly_score
from quishguard.vit.preprocess import prepare
from quishguard.vit.train import load_visual, patch_map


class VisualScorer:
    def __init__(self, arch: str = "patchcore", models_dir: Path | str = config.MODELS_DIR, device: str | None = None):
        models_dir = Path(models_dir)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = load_visual(models_dir / f"visual_{arch}.pt", self.device)
        c = joblib.load(models_dir / f"visual_{arch}_calibrator.joblib")
        self.cal, self.thr = c["calibrator"], c["threshold_error"]
        self.heat_lo, self.heat_hi = c.get("heat_lo"), c.get("heat_hi")

    @torch.no_grad()
    def score(self, img: ImageLike) -> dict:
        x = torch.from_numpy(prepare(img))[None, None].to(self.device)
        pe = patch_map(self.model, x)
        err = float(anomaly_score(pe)[0])
        p = float(self.cal.predict_proba(np.log([[err + 1e-8]]))[0, 1])
        hm = pe[0].cpu().numpy()
        worst = np.unravel_index(np.argmax(hm), hm.shape)
        return {
            "visual_score": round(100 * p, 2),
            "tampered": err >= self.thr,
            "raw_error": err,
            "heatmap": hm,                      # 14 x 14 map (for the alert image)
            "worst_patch_rowcol": [int(worst[0]), int(worst[1])],
        }

    def heatmap_image(self, img: ImageLike, result: dict | None = None):
        """BGR image with the unusual areas coloured (for the alert / dashboard)."""
        from quishguard.vit.heatmap import overlay
        r = result or self.score(img)
        lo = self.heat_lo if self.heat_lo is not None else float(np.percentile(r["heatmap"], 50))
        hi = self.heat_hi if self.heat_hi is not None else float(r["heatmap"].max()) + 1e-6
        return overlay(prepare(img), r["heatmap"], lo, hi)
