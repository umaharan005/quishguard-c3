"""Heatmap images for the explainable alert.

Colours only the patches that are MORE unusual than normal codes ever get:
  lo = 99th percentile of patch scores on normal val images (below: no colour)
  hi = 99.5th percentile of patch scores on tampered val images (full red)
Colour strength also sets transparency, so normal areas stay clean.

Regenerate the report figure from a saved model (no retraining):
    python -m quishguard.vit.heatmap --models-dir .../models --arch patchcore \
        --tamper-dir /content/tamper --out .../reports/patchcore/heatmaps.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

ATTACK_ORDER = ["normal", "sticker", "logo", "module_flip", "warp", "double_print"]


def heat_scale(normal_maps: np.ndarray, tampered_maps: np.ndarray) -> tuple[float, float]:
    lo = float(np.percentile(normal_maps, 99))
    hi = float(np.percentile(tampered_maps, 99.5))
    if hi <= lo:
        hi = lo * 1.5 + 1e-6
    return lo, hi


def overlay(gray01: np.ndarray, pmap: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """gray01: 224x224 float [0,1]; pmap: 14x14 -> BGR uint8 image with heat on top."""
    size = gray01.shape[0]
    h = np.clip((pmap - lo) / (hi - lo), 0, 1).astype(np.float32)
    h = cv2.resize(h, (size, size), interpolation=cv2.INTER_CUBIC).clip(0, 1)
    color = cv2.applyColorMap((h * 255).astype(np.uint8), cv2.COLORMAP_JET).astype(np.float32)
    base = cv2.cvtColor((gray01 * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR).astype(np.float32)
    a = (0.75 * h)[..., None]
    return (base * (1 - a) + color * a).astype(np.uint8)


def render_grid(path: Path, items: list[tuple[str, np.ndarray, np.ndarray]], lo: float, hi: float, per: int = 3):
    """items: (attack, gray01 image, 14x14 map)."""
    rows = []
    for kind in ATTACK_ORDER:
        sel = [it for it in items if it[0] == kind][:per]
        if not sel:
            continue
        tiles = []
        for _, g, m in sel:
            tiles.append(np.hstack([cv2.cvtColor((g * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR),
                                    overlay(g, m, lo, hi),
                                    np.full((g.shape[0], 8, 3), 255, np.uint8)]))
        r = np.hstack(tiles)
        r = cv2.copyMakeBorder(r, 24, 4, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(r, kind, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        rows.append(r)
    w = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 0, 0, w - r.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for r in rows]
    cv2.imwrite(str(path), np.vstack(rows))


def main(argv=None):
    import joblib
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader

    from quishguard.vit.preprocess import prepare
    from quishguard.vit.train import Files, load_visual, score_loader

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-dir", type=Path, required=True)
    ap.add_argument("--arch", default="patchcore")
    ap.add_argument("--tamper-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_visual(args.models_dir / f"visual_{args.arch}.pt", device)
    man = pd.read_csv(args.tamper_dir / "manifest.csv")

    va = pd.concat([g.sample(n=min(len(g), 150), random_state=0)
                    for _, g in man[man["split"] == "val"].groupby("attack")], ignore_index=True)
    _, vmaps = score_loader(model, DataLoader(Files([args.tamper_dir / f for f in va["file"]]), batch_size=64),
                            device, keep_maps=len(va))
    lo, hi = heat_scale(vmaps[(va["attack"] == "normal").values], vmaps[(va["attack"] != "normal").values])

    te = pd.concat([g.sample(n=args.per, random_state=args.seed)
                    for _, g in man[man["split"] == "test"].groupby("attack")], ignore_index=True)
    _, tmaps = score_loader(model, DataLoader(Files([args.tamper_dir / f for f in te["file"]]), batch_size=64),
                            device, keep_maps=len(te))
    items = [(a, prepare(args.tamper_dir / f), tmaps[i]) for i, (a, f) in enumerate(zip(te["attack"], te["file"]))]
    render_grid(args.out, items, lo, hi, per=args.per)

    calp = args.models_dir / f"visual_{args.arch}_calibrator.joblib"
    if calp.exists():
        c = joblib.load(calp); c.update({"heat_lo": lo, "heat_hi": hi}); joblib.dump(c, calp)
    print(f"Heat scale lo={lo:.4f} hi={hi:.4f} (saved to calibrator)\nFigure -> {args.out}")


if __name__ == "__main__":
    main()
