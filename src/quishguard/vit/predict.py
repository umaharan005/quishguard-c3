"""Score one QR image with the trained visual model: 0-100 score + heatmap."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import torch

from quishguard import config
from quishguard.decode.decoder import ImageLike
from quishguard.vit.model import anomaly_score, build, patch_errors
from quishguard.vit.preprocess import prepare


class VisualScorer:
    def __init__(self, arch: str = "vit", models_dir: Path | str = config.MODELS_DIR, device: str | None = None):
        models_dir = Path(models_dir)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ck = torch.load(models_dir / f"visual_{arch}.pt", map_location=self.device)
        self.model = build(ck["arch"], pretrained=False).to(self.device).eval()
        self.model.load_state_dict(ck["state_dict"])
        c = joblib.load(models_dir / f"visual_{arch}_calibrator.joblib")
        self.cal, self.thr = c["calibrator"], c["threshold_error"]

    @torch.no_grad()
    def score(self, img: ImageLike) -> dict:
        x = torch.from_numpy(prepare(img))[None, None].to(self.device)
        pe = patch_errors(x, self.model(x))
        err = float(anomaly_score(pe)[0])
        p = float(self.cal.predict_proba(np.log([[err + 1e-8]]))[0, 1])
        hm = pe[0].cpu().numpy()
        worst = np.unravel_index(np.argmax(hm), hm.shape)
        return {
            "visual_score": round(100 * p, 2),
            "tampered": err >= self.thr,
            "raw_error": err,
            "heatmap": hm,                      # 14 x 14 patch errors (for the alert image)
            "worst_patch_rowcol": [int(worst[0]), int(worst[1])],
        }
