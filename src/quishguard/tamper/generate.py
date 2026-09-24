"""Phase 4: build the tamper test set for the ViT anomaly detector.

CIC-Trap4Phish only has clean, computer-made QR codes, so there is nothing
tampered to test the ViT on. This script makes realistic tampered versions
of clean codes, plus "normal" versions that only have everyday photo effects
(blur, lighting, noise, JPEG, small angle). The ViT must learn to ignore the
normal effects and flag only the tampering.

Attack types
  sticker       another QR pasted over 50-100% of the original (a sticker
                glued over a real parking or payment code); edges of the
                original stay visible
  logo          a solid logo/patch covering the centre beyond what error
                correction can repair
  module_flip   a patch of modules flipped black<->white (scratched or
                edited code)
  warp          part of the code shifted / sheared (misprint, bent sticker)
  double_print  two codes printed on top of each other (misaligned overlay)

Every output image (normal or tampered) also gets random everyday photo
effects, so "has photo effects" never gives the answer away.

    python -m quishguard.tamper.generate --raw-dir /content/raw \
        --subset data/processed/image_subset.csv --out data/tamper
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from quishguard import config
from quishguard.data.prepare import find_class_root
from quishguard.decode.decoder import add_quiet_zone, decode, to_gray

ATTACKS = ("sticker", "logo", "module_flip", "warp", "double_print")


# ------------------------------------------------------------ everyday effects
def photo_effects(img: np.ndarray, rng: random.Random, strength: float = 1.0) -> np.ndarray:
    """Blur, lighting, noise, small rotation/perspective, JPEG. Never tampering."""
    img = add_quiet_zone(img, max(4, int(0.08 * max(img.shape))))  # room so rotation never clips corners
    h, w = img.shape
    out = img.astype(np.float32)
    # lighting: contrast/brightness + soft gradient (shadow)
    a = rng.uniform(0.75, 1.1)
    b = rng.uniform(-25, 25) * strength
    out = out * a + b
    if rng.random() < 0.5:
        gx = np.linspace(0, 1, w)[None, :] if rng.random() < 0.5 else np.linspace(0, 1, h)[:, None]
        out = out - rng.uniform(0, 45) * strength * gx
    out = np.clip(out, 0, 255).astype(np.uint8)
    # small rotation + perspective (camera angle)
    if rng.random() < 0.7:
        d = 0.04 * strength * w
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = src + np.float32([[rng.uniform(-d, d), rng.uniform(-d, d)] for _ in range(4)])
        M = cv2.getPerspectiveTransform(src, dst)
        R = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-6, 6) * strength, 1.0)
        out = cv2.warpAffine(out, R, (w, h), borderValue=255)
        out = cv2.warpPerspective(out, M, (w, h), borderValue=255)
    # blur
    if rng.random() < 0.6:
        k = rng.choice([3, 3, 5])
        out = cv2.GaussianBlur(out, (k, k), rng.uniform(0.3, 1.2) * strength)
    # sensor noise
    if rng.random() < 0.6:
        out = np.clip(out + np.random.default_rng(rng.randint(0, 1 << 30)).normal(0, rng.uniform(2, 9) * strength, out.shape), 0, 255).astype(np.uint8)
    # JPEG
    if rng.random() < 0.7:
        ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(45, 92)])
        out = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    return out


# ------------------------------------------------------------ attacks
def _fit(img: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)


def attack_sticker(qr: np.ndarray, other: np.ndarray, rng: random.Random):
    h, w = qr.shape
    frac = rng.uniform(0.5, 1.0)                      # share of the side covered
    s = int(min(h, w) * rng.uniform(0.85, 1.05) * frac ** 0.5 + 0.5)
    s = max(s, int(0.6 * min(h, w)))
    border = max(2, s // 20)                         # white sticker margin
    stk = cv2.copyMakeBorder(_fit(other, s), border, border, border, border, cv2.BORDER_CONSTANT, value=255)
    S = stk.shape[0]
    canvas = add_quiet_zone(qr, S // 3)
    H, W = canvas.shape
    cx = rng.randint(S // 3, W - S // 3 - S) if W - 2 * (S // 3) - S > 0 else (W - S) // 2
    cy = rng.randint(S // 3, H - S // 3 - S) if H - 2 * (S // 3) - S > 0 else (H - S) // 2
    ang = rng.uniform(-8, 8)
    M = cv2.getRotationMatrix2D((S / 2, S / 2), ang, 1.0)
    rot = cv2.warpAffine(stk, M, (S, S), borderValue=255)
    mask = cv2.warpAffine(np.full((S, S), 255, np.uint8), M, (S, S), borderValue=0)
    # soft shadow under the sticker
    sh = cv2.GaussianBlur(mask, (0, 0), 3)
    roi = canvas[cy:cy + S, cx:cx + S].astype(np.float32)
    roi = roi * (1 - 0.25 * sh[: roi.shape[0], : roi.shape[1]] / 255.0)
    m = mask[: roi.shape[0], : roi.shape[1]] > 127
    roi[m] = rot[: roi.shape[0], : roi.shape[1]][m]
    canvas[cy:cy + S, cx:cx + S] = np.clip(roi, 0, 255).astype(np.uint8)
    return canvas, {"cover_frac": round(frac, 3)}


def attack_logo(qr: np.ndarray, rng: random.Random):
    h, w = qr.shape
    out = qr.copy()
    frac = rng.uniform(0.30, 0.55)                    # side of the logo vs code side
    s = int(min(h, w) * frac)
    x0 = (w - s) // 2 + rng.randint(-s // 6, s // 6)
    y0 = (h - s) // 2 + rng.randint(-s // 6, s // 6)
    logo = np.full((s, s), rng.choice([255, 230, 40]), np.uint8)
    if rng.random() < 0.5:                             # round logo
        logo[:] = 255
        cv2.circle(logo, (s // 2, s // 2), s // 2 - 1, rng.randint(0, 90), -1)
    # a few "letters"/shapes inside
    for _ in range(rng.randint(1, 4)):
        c = rng.randint(0, 255)
        p1 = (rng.randint(0, s - 1), rng.randint(0, s - 1))
        p2 = (rng.randint(0, s - 1), rng.randint(0, s - 1))
        cv2.rectangle(logo, p1, p2, c, -1) if rng.random() < 0.5 else cv2.line(logo, p1, p2, c, max(1, s // 12))
    out[y0:y0 + s, x0:x0 + s] = logo
    return out, {"logo_frac": round(frac, 3)}


def _module_size(qr: np.ndarray) -> int:
    # CIC codes have no quiet zone; top-left finder is 7 modules wide
    row = qr[qr.shape[0] // 14]
    dark = np.where(row < 128)[0]
    if len(dark) == 0:
        return max(2, qr.shape[1] // 25)
    run = 0
    for x in range(dark[0], len(row)):
        if row[x] < 128:
            run += 1
        else:
            break
    return max(2, int(round(run / 7)))


def attack_module_flip(qr: np.ndarray, rng: random.Random):
    out = qr.copy()
    m = _module_size(qr)
    h, w = qr.shape
    n_mod = w // m
    pw = rng.randint(max(3, n_mod // 5), max(4, n_mod // 2))
    ph = rng.randint(max(3, n_mod // 5), max(4, n_mod // 2))
    mx, my = rng.randint(0, n_mod - pw), rng.randint(0, n_mod - ph)
    p = rng.uniform(0.35, 0.7)
    flipped = 0
    for yy in range(my, my + ph):
        for xx in range(mx, mx + pw):
            if rng.random() < p:
                blk = out[yy * m:(yy + 1) * m, xx * m:(xx + 1) * m]
                out[yy * m:(yy + 1) * m, xx * m:(xx + 1) * m] = 255 - blk
                flipped += 1
    return out, {"modules_flipped": flipped}


def attack_warp(qr: np.ndarray, rng: random.Random):
    h, w = qr.shape
    canvas = add_quiet_zone(qr, 20)
    H, W = canvas.shape
    out = canvas.copy()
    if rng.random() < 0.5:     # cut and shift one part (misaligned print / swapped half)
        horizontal = rng.random() < 0.5
        cut = rng.randint(int(0.3 * h), int(0.7 * h)) + 20
        shift = rng.choice([-1, 1]) * rng.randint(max(3, w // 20), max(4, w // 8))
        if horizontal:
            out[cut:, :] = np.roll(canvas[cut:, :], shift, axis=1)
        else:
            out[:, cut:] = np.roll(canvas[:, cut:], shift, axis=0)
        info = {"kind": "shift", "shift_px": int(shift)}
    else:                      # local wave bend (bent / bubbled sticker)
        amp = rng.uniform(0.02, 0.05) * w
        per = rng.uniform(0.4, 1.0) * w
        ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
        y0, y1 = sorted(rng.sample(range(20, H - 20), 2))
        band = ((ys > y0) & (ys < y1)).astype(np.float32)
        mapx = xs + band * amp * np.sin(2 * np.pi * ys / per)
        out = cv2.remap(canvas, mapx, ys, cv2.INTER_LINEAR, borderValue=255)
        info = {"kind": "bend", "amp_px": round(float(amp), 1)}
    return out, info


def attack_double_print(qr: np.ndarray, other: np.ndarray, rng: random.Random):
    h, w = qr.shape
    o = _fit(other, w)
    dx, dy = rng.randint(-w // 6, w // 6), rng.randint(-h // 6, h // 6)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    o = cv2.warpAffine(o, M, (w, h), borderValue=255)
    alpha = rng.uniform(0.35, 0.65)
    out = np.minimum(qr, (o * alpha + 255 * (1 - alpha)).astype(np.uint8))
    return out, {"alpha": round(alpha, 2), "dx": dx, "dy": dy}


def recrop(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Crop to the dark content and add the same random white margin for every
    image, so border size can never tell normal and tampered apart."""
    ys, xs = np.where(img < 160)
    if len(ys):
        img = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return add_quiet_zone(img, rng.randint(0, 16))


# ------------------------------------------------------------ builder
def build(raw_dir: Path, subset_csv: Path, out_dir: Path, split: str,
          n_normal: int, n_per_attack: int, seed: int = config.SEED) -> pd.DataFrame:
    rng = random.Random(seed + hash(split) % 1000)
    sub = pd.read_csv(subset_csv)
    sub = sub[sub["split"] == split].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    need = n_normal + n_per_attack * len(ATTACKS)
    if len(sub) < need + 50:
        raise ValueError(f"Only {len(sub)} images in split '{split}', need {need}")
    roots = {c: find_class_root(raw_dir, c) / "qrs" for c in config.CLASS_DIRS}
    load = lambda r: to_gray(roots[r["class_dir"]] / r["img_file"])  # noqa: E731

    rows, i = [], 0
    donors = sub.iloc[need:].reset_index(drop=True)   # "attacker" codes never used as a base
    (out_dir / split).mkdir(parents=True, exist_ok=True)

    def save(img, kind, base, extra):
        nonlocal i
        name = f"{split}_{kind}_{i:05d}.png"
        cv2.imwrite(str(out_dir / split / name), img)
        d = decode(img)
        rows.append({"file": f"{split}/{name}", "split": split, "tampered": int(kind != "normal"),
                     "attack": kind, "base_img": f"{base['class_dir']}/{base['img_file']}",
                     "base_label": int(base["label"]), "base_url": base["url"],
                     "decodes": int(d.ok), "decoded_text": d.text or "",
                     "decodes_to_base_url": int(d.ok and d.text.strip() == str(base["url"]).strip()),
                     "params": json.dumps(extra)})
        i += 1

    k = 0
    for _ in range(n_normal):
        base = sub.iloc[k]; k += 1
        save(photo_effects(recrop(load(base), rng), rng), "normal", base, {})
    for attack in ATTACKS:
        for _ in range(n_per_attack):
            base = sub.iloc[k]; k += 1
            qr = load(base)
            if attack in ("sticker", "double_print"):
                other = load(donors.iloc[rng.randrange(len(donors))])
                t, extra = (attack_sticker if attack == "sticker" else attack_double_print)(qr, other, rng)
            else:
                t, extra = {"logo": attack_logo, "module_flip": attack_module_flip, "warp": attack_warp}[attack](qr, rng)
            save(photo_effects(recrop(t, rng), rng), attack, base, extra)
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, default=config.RAW_DIR)
    ap.add_argument("--subset", type=Path, default=config.PROCESSED_DIR / "image_subset.csv")
    ap.add_argument("--out", type=Path, default=config.REPO_ROOT / "data" / "tamper")
    ap.add_argument("--test-normal", type=int, default=3000)
    ap.add_argument("--test-per-attack", type=int, default=1000)
    ap.add_argument("--val-normal", type=int, default=900)
    ap.add_argument("--val-per-attack", type=int, default=300)
    ap.add_argument("--examples", action="store_true", help="also save an example grid PNG")
    args = ap.parse_args(argv)

    frames = []
    for split, nn, na in (("val", args.val_normal, args.val_per_attack), ("test", args.test_normal, args.test_per_attack)):
        print(f"Building {split}: {nn} normal + {na} x {len(ATTACKS)} attacks")
        frames.append(build(args.raw_dir, args.subset, args.out, split, nn, na))
    man = pd.concat(frames, ignore_index=True)
    man.to_csv(args.out / "manifest.csv", index=False)
    summ = man.groupby(["split", "attack"]).agg(n=("file", "size"), decodes=("decodes", "mean"),
                                                 same_url=("decodes_to_base_url", "mean")).round(3)
    summ.to_csv(args.out / "summary.csv")
    print(summ.to_string())
    if args.examples:
        save_grid(args.out, man)
    print(f"Manifest -> {args.out / 'manifest.csv'}")


def save_grid(out_dir: Path, man: pd.DataFrame, per: int = 4):
    tiles = []
    for kind in ("normal",) + ATTACKS:
        row = []
        for f in man[man["attack"] == kind]["file"].head(per):
            im = cv2.imread(str(out_dir / f), cv2.IMREAD_GRAYSCALE)
            im = cv2.resize(im, (180, 180), interpolation=cv2.INTER_AREA)
            row.append(cv2.copyMakeBorder(im, 22, 4, 4, 4, cv2.BORDER_CONSTANT, value=255))
        r = np.hstack(row)
        cv2.putText(r, kind, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1, cv2.LINE_AA)
        tiles.append(r)
    cv2.imwrite(str(out_dir / "examples.png"), np.vstack(tiles))


if __name__ == "__main__":
    main()
