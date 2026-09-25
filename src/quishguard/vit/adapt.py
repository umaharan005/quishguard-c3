"""Adapt the PatchCore visual model to real-world QR codes (domain adaptation, no retraining).

Why: the memory bank only holds patches of CIC-Trap4Phish codes. Codes from other
generators, screens and phone cameras look "unusual" to it, so clean codes score high.
PatchCore's idea of "normal" IS its memory bank, so we add normal patches from:
  * synthetic codes made with another generator (segno), random sizes / error levels /
    quiet zones, shown three ways: digital (screenshot), printed photo, photo of a screen
  * optional real phone photos of NORMAL (untampered) codes you took (--photos folder)
Every image goes through exactly the same steps as the scan app
(decode_photo -> locate -> [clean_photo]) so the memory matches what the app sees.

Then a new threshold and calibrator are fitted on held-out normal codes and synthetic
tampered versions (sticker, logo, module_flip, warp, double_print), and a before/after
table is printed on a separate held-out test half.

    python -m quishguard.vit.adapt --models-dir models --photos my_normal_photos

Writes models/visual_patchcore_adapted.pt + visual_patchcore_adapted_calibrator.joblib
(the originals are not changed). The scan app uses the adapted model when it exists.
"""
from __future__ import annotations

import argparse
import io
import json
import random
import string
import time
from pathlib import Path

import cv2
import numpy as np

from quishguard.decode.decoder import add_quiet_zone, clean_photo, decode_photo, locate, to_gray
from quishguard.tamper.generate import (attack_double_print, attack_logo, attack_module_flip,
                                        attack_sticker, attack_warp, photo_effects, recrop)

ATTACKS = ["sticker", "logo", "module_flip", "warp", "double_print"]
RENDERS = ["digital", "print_photo", "screen_photo"]


# ------------------------------------------------------------ synthetic codes
def random_payload(rng: random.Random) -> str:
    host = "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(4, 14)))
    tld = rng.choice(["com", "lk", "org", "net", "io", "xyz", "top", "edu", "gov.lk", "co.uk"])
    path = "/".join("".join(rng.choice(string.ascii_letters + string.digits + "-_")
                            for _ in range(rng.randint(2, 12))) for _ in range(rng.randint(0, 4)))
    kind = rng.random()
    if kind < 0.75:
        return f"{rng.choice(['https://', 'http://', ''])}{rng.choice(['www.', ''])}{host}.{tld}/{path}"
    if kind < 0.85:
        return f"WIFI:S:{host};T:WPA;P:{path or host};;"
    if kind < 0.93:
        return f"upi://pay?pa={host}@bank&am={rng.randint(10, 9999)}"
    return " ".join(host for _ in range(rng.randint(1, 6)))


def make_code(text: str, rng: random.Random, border: int | None = None) -> np.ndarray:
    """A clean code from a generator other than CIC's: segno or OpenCV (both when available)."""
    scale = rng.randint(3, 10)
    border = rng.randint(0, 4) if border is None else border
    try:
        import segno
        use_segno = rng.random() < 0.6
    except ImportError:
        use_segno = False
    if use_segno:
        buf = io.BytesIO()
        segno.make(text, error=rng.choice("lmqh"), micro=False).save(buf, kind="png", scale=scale, border=border)
        return cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), cv2.IMREAD_GRAYSCALE)
    p = cv2.QRCodeEncoder.Params()
    p.correction_level = rng.choice([cv2.QRCODE_ENCODER_CORRECT_LEVEL_L, cv2.QRCODE_ENCODER_CORRECT_LEVEL_M,
                                     cv2.QRCODE_ENCODER_CORRECT_LEVEL_Q, cv2.QRCODE_ENCODER_CORRECT_LEVEL_H])
    img = cv2.QRCodeEncoder.create(p).encode(text)
    ys, xs = np.where(img < 128)
    img = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]           # drop its own quiet zone
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    return cv2.copyMakeBorder(img, border * scale, border * scale, border * scale, border * scale,
                              cv2.BORDER_CONSTANT, value=255)


def screen_photo(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """A phone photo of a code shown on a laptop / phone screen: grey-white background,
    moire stripes, glare, perspective, blur, noise, JPEG."""
    img = add_quiet_zone(img, max(8, img.shape[0] // 8))
    h, w = img.shape
    S = int(max(h, w) * rng.uniform(1.5, 2.4))
    canvas = np.full((S, S), rng.randint(150, 235), np.float32)
    white = rng.uniform(190, 250)
    black = rng.uniform(10, 70)
    code = black + (img.astype(np.float32) / 255.0) * (white - black)
    y0, x0 = rng.randint(0, S - h), rng.randint(0, S - w)
    canvas[y0:y0 + h, x0:x0 + w] = code
    ys, xs = np.mgrid[0:S, 0:S].astype(np.float32)
    ang = rng.uniform(0, np.pi)
    canvas += rng.uniform(2, 10) * np.sin((xs * np.cos(ang) + ys * np.sin(ang)) * rng.uniform(0.3, 1.2))  # moire
    gx, gy = rng.uniform(0, S), rng.uniform(0, S)
    canvas += rng.uniform(0, 45) * np.exp(-((xs - gx) ** 2 + (ys - gy) ** 2) / (2 * (S * rng.uniform(0.15, 0.4)) ** 2))  # glare
    d = 0.08 * S
    src = np.float32([[0, 0], [S, 0], [S, S], [0, S]])
    dst = src + np.float32([[rng.uniform(-d, d), rng.uniform(-d, d)] for _ in range(4)])
    out = cv2.warpPerspective(np.clip(canvas, 0, 255).astype(np.uint8), cv2.getPerspectiveTransform(src, dst),
                              (S, S), borderMode=cv2.BORDER_REPLICATE)
    out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.6, 1.8))
    out = np.clip(out + np.random.default_rng(rng.randint(0, 1 << 30)).normal(0, rng.uniform(2, 7), out.shape), 0, 255)
    ok, buf = cv2.imencode(".jpg", out.astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, rng.randint(60, 92)])
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)


def render(img: np.ndarray, how: str, rng: random.Random) -> np.ndarray:
    if how == "digital":
        return img
    if how == "print_photo":
        return photo_effects(recrop(img, rng), rng)
    return screen_photo(img, rng)


def tamper(qr: np.ndarray, kind: str, rng: random.Random) -> np.ndarray:
    other = make_code(random_payload(rng), rng, border=0)
    if kind == "sticker":
        return attack_sticker(qr, other, rng)[0]
    if kind == "logo":
        return attack_logo(qr, rng)[0]
    if kind == "module_flip":
        return attack_module_flip(qr, rng)[0]
    if kind == "warp":
        return attack_warp(qr, rng)[0]
    return attack_double_print(qr, other, rng)[0]


# ------------------------------------------------------------ app view + scoring
def app_view(img: np.ndarray, clean: bool) -> np.ndarray | None:
    """Exactly what the scan app gives the visual model (None = the app finds no code)."""
    crop = locate(decode_photo(to_gray(img)))
    if crop is None:
        return None
    return clean_photo(crop) if clean else crop


def batch_maps(model, imgs: list[np.ndarray], device, bs: int = 32) -> np.ndarray:
    import torch
    from quishguard.vit.preprocess import prepare
    from quishguard.vit.train import patch_map
    out = []
    for i in range(0, len(imgs), bs):
        x = torch.from_numpy(np.stack([prepare(im) for im in imgs[i:i + bs]]))[:, None].to(device)
        out.append(patch_map(model, x).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 14, 14), np.float32)


def errors(maps: np.ndarray) -> np.ndarray:
    import torch
    from quishguard.vit.model import anomaly_score
    return anomaly_score(torch.from_numpy(maps)).numpy() if len(maps) else np.zeros(0)


def summary(err_n, err_t, thr, cal, real=None) -> dict:
    from sklearn.metrics import roc_auc_score
    vs = lambda e: 100 * cal.predict_proba(np.log(e[:, None] + 1e-8))[:, 1]
    y = np.r_[np.zeros(len(err_n)), np.ones(len(err_t))]
    s = np.r_[err_n, err_t]
    r = {"auroc": float(roc_auc_score(y, s)),
         "false_alarm_rate": float((err_n >= thr).mean()),
         "detection_rate": float((err_t >= thr).mean()),
         "mean_score_normal": float(vs(err_n).mean()),
         "mean_score_tampered": float(vs(err_t).mean())}
    if real is not None and len(real):
        r["real_photos_mean_score"] = float(vs(real).mean())
        r["real_photos_false_alarm_rate"] = float((real >= thr).mean())
    return r


# ------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-dir", type=Path, default=Path("models"))
    ap.add_argument("--photos", type=Path, default=None, help="folder with phone photos of NORMAL (untampered) QR codes")
    ap.add_argument("--memory-codes", type=int, default=200, help="synthetic normal codes added to memory (x3 renders)")
    ap.add_argument("--val-codes", type=int, default=120, help="synthetic codes for threshold + test (normal and tampered)")
    ap.add_argument("--coreset", type=float, default=0.10)
    ap.add_argument("--clean", action="store_true", help="use clean_photo() like the app's --photo-clean mode")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    import joblib
    import torch
    from sklearn.linear_model import LogisticRegression
    from quishguard.vit.heatmap import heat_scale
    from quishguard.vit.patchcore import greedy_coreset
    from quishguard.vit.train import load_visual

    t0 = time.time()
    rng = random.Random(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_visual(args.models_dir / "visual_patchcore.pt", device)
    old_cal = joblib.load(args.models_dir / "visual_patchcore_calibrator.joblib")
    print(f"Device {device}. Memory bank before: {model.memory.shape[0]:,} patches")

    # --- real photos: half to memory, half to the test
    real_mem, real_test = [], []
    if args.photos and args.photos.exists():
        files = sorted(f for f in args.photos.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
        rng.shuffle(files)
        views = [(f.name, app_view(cv2.imread(str(f)), args.clean)) for f in files]
        skipped = [n for n, v in views if v is None]
        views = [v for n, v in views if v is not None]
        real_mem, real_test = views[0::2], views[1::2]
        print(f"Real photos: {len(views)} usable ({len(real_mem)} memory / {len(real_test)} test)"
              + (f", no code found in: {skipped}" if skipped else ""))

    # --- synthetic normal codes for memory
    def normals(n):
        out = []
        while len(out) < n:
            qr = make_code(random_payload(rng), rng)
            v = app_view(render(qr, RENDERS[len(out) % 3], rng), args.clean)
            if v is not None:
                out.append(v)
        return out

    mem_imgs = normals(3 * args.memory_codes) + real_mem
    print(f"Memory images: {len(mem_imgs)} ({time.time() - t0:.0f}s)")

    # --- validation / test: normal + tampered, split in half (cal / test)
    val_n = normals(2 * args.val_codes)
    val_t, kinds = [], []
    while len(val_t) < 2 * args.val_codes:
        k = ATTACKS[len(val_t) % len(ATTACKS)]
        qr = make_code(random_payload(rng), rng, border=0)
        v = app_view(render(tamper(qr, k, rng), RENDERS[rng.randint(0, 2)], rng), args.clean)
        if v is not None:
            val_t.append(v); kinds.append(k)
    cal_n, test_n = val_n[0::2], val_n[1::2]
    cal_t, test_t = val_t[0::2], val_t[1::2]
    test_kinds = kinds[1::2]
    print(f"Validation: {len(val_n)} normal + {len(val_t)} tampered ({time.time() - t0:.0f}s)")

    # --- BEFORE
    before = summary(errors(batch_maps(model, test_n, device)), errors(batch_maps(model, test_t, device)),
                     old_cal["threshold_error"], old_cal["calibrator"],
                     errors(batch_maps(model, real_test, device)) if real_test else None)

    # --- add normal patches to the memory bank
    feats = []
    for i in range(0, len(mem_imgs), 32):
        from quishguard.vit.preprocess import prepare
        x = torch.from_numpy(np.stack([prepare(im) for im in mem_imgs[i:i + 32]]))[:, None].to(device)
        feats.append(model.features(x).reshape(-1, model.memory.shape[1]))
    feats = torch.cat(feats)
    add = greedy_coreset(feats, max(1000, int(len(feats) * args.coreset)), log=lambda *a: None)
    model.memory = torch.cat([model.memory, add.to(model.memory.device)])
    print(f"Added {add.shape[0]:,} patches -> memory bank {model.memory.shape[0]:,} ({time.time() - t0:.0f}s)")

    # --- new threshold + calibrator on the cal half
    mn, mt = batch_maps(model, cal_n, device), batch_maps(model, cal_t, device)
    en, et = errors(mn), errors(mt)
    thr = float(np.quantile(en, 0.95))
    cal = LogisticRegression().fit(np.log(np.r_[en, et][:, None] + 1e-8), np.r_[np.zeros(len(en)), np.ones(len(et))])
    lo, hi = heat_scale(mn, mt)

    # --- AFTER (test half)
    e_tn, e_tt = errors(batch_maps(model, test_n, device)), errors(batch_maps(model, test_t, device))
    after = summary(e_tn, e_tt, thr, cal, errors(batch_maps(model, real_test, device)) if real_test else None)
    per_attack = {k: float((e_tt[np.array(test_kinds) == k] >= thr).mean()) for k in ATTACKS}

    torch.save({"arch": "patchcore", "state_dict": model.state_dict()}, args.models_dir / "visual_patchcore_adapted.pt")
    info = {"memory_images": len(mem_imgs), "real_photos_memory": len(real_mem), "real_photos_test": len(real_test),
            "patches_added": int(add.shape[0]), "clean": args.clean, "seed": args.seed}
    joblib.dump({"calibrator": cal, "threshold_error": thr, "arch": "patchcore_adapted",
                 "heat_lo": lo, "heat_hi": hi, "adapted": info}, args.models_dir / "visual_patchcore_adapted_calibrator.joblib")
    report = {"before": before, "after": after, "after_detection_by_attack": per_attack, "info": info}
    (args.models_dir / "visual_adaptation_report.json").write_text(json.dumps(report, indent=2))

    print("\nHeld-out test (other generators + phone/screen photos), threshold = 5% false alarms on normals")
    print(f"{'':30s}{'before':>10s}{'after':>10s}")
    for k in before:
        print(f"{k:30s}{before[k]:10.3f}{after[k]:10.3f}")
    print("detection by attack (after): " + ", ".join(f"{k} {v:.2f}" for k, v in per_attack.items()))
    print(f"\nSaved visual_patchcore_adapted.pt (+ calibrator, report) in {args.models_dir}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
