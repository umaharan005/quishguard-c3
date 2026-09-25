"""Check how the visual model sees real photos: original vs adapted visual model.

    python -m quishguard.app.diagnose --models-dir models data/app/scans test_qr

For every photo it prints whether the code was read and the visual score (0-100) of the
original model and, if present, the adapted model (python -m quishguard.vit.adapt).
A clean, untampered code should score low (well under 50).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from quishguard.decode.decoder import decode_photo, locate, to_gray


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+", type=Path)
    ap.add_argument("--models-dir", type=Path, default=Path("models"))
    args = ap.parse_args(argv)

    from quishguard.vit.predict import VisualScorer
    models = {"original": VisualScorer("patchcore", args.models_dir)}
    if (args.models_dir / "visual_patchcore_adapted.pt").exists():
        models["adapted"] = VisualScorer("patchcore_adapted", args.models_dir)
    files = [f for d in args.folders for f in sorted(d.glob("*"))
             if f.suffix.lower() in (".jpg", ".jpeg", ".png") and not f.stem.endswith("_heat")]
    print(f"{'file':34s} {'read':5s} " + " ".join(f"{k:>9s}" for k in models) + "  link   (visual score, raw crop)")
    for f in files:
        d = decode_photo(to_gray(f))
        crop = locate(d)
        if crop is None:
            print(f"{f.name[:34]:34s} {'-':5s} no QR code found")
            continue
        sc = " ".join(f"{m.score(crop)['visual_score']:9.1f}" for m in models.values())
        print(f"{f.name[:34]:34s} {'yes' if d.ok else 'no':5s} {sc}  {(d.text or '')[:45]}")

if __name__ == "__main__":
    main()
