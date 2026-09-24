"""Phase 1: clean CIC-Trap4Phish QR data and split it by domain.

Run on the laptop (or Colab):

    python -m quishguard.data.prepare --raw-dir D:\\uma

Writes to data/processed/:
    urls_clean.csv.gz     one row per usable sample, with label, host, domain, split
    image_subset.csv      balanced image subset (only images that exist on disk)
    prepare_report.json   counts to quote in the report
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

from quishguard import config
from quishguard.data.urls import get_host, normalize_url, registered_domain


def find_class_root(raw_dir: Path, class_dir: str) -> Path:
    """Handle both layouts: raw/QR_All_benign/... and raw/QR_All_benign/QR_All_benign/..."""
    for cand in (raw_dir / class_dir / class_dir, raw_dir / class_dir):
        if list(cand.glob("*.csv")) or (cand / "qrs").is_dir():
            return cand
    raise FileNotFoundError(f"Could not find {class_dir} under {raw_dir}")


def load_raw(raw_dir: Path) -> pd.DataFrame:
    frames = []
    for class_dir, label in config.CLASS_DIRS.items():
        root = find_class_root(raw_dir, class_dir)
        csvs = sorted(root.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(f"No CSV in {root}")
        df = pd.read_csv(csvs[0], dtype=str, keep_default_na=False)
        df["label"] = label
        df["class_dir"] = class_dir
        # qr_path looks like "Output\QR_All_benign\qrs\benign_000001.png"
        df["img_file"] = df["qr_path"].str.replace("\\", "/", regex=False).str.split("/").str[-1]
        df["source_row"] = df["index"]
        frames.append(df[["source_row", "url", "label", "class_dir", "img_file"]])
    return pd.concat(frames, ignore_index=True)


def split_of(domain: str) -> str:
    h = int(hashlib.md5((config.SPLIT_SALT + domain).encode("utf-8")).hexdigest(), 16) % 100
    for name, (lo, hi) in config.SPLIT_BUCKETS.items():
        if lo <= h < hi:
            return name
    raise AssertionError(h)


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rep: dict = {"raw_rows": df["label"].value_counts().sort_index().to_dict()}

    df = df.copy()
    df["url"] = df["url"].str.strip()
    df["had_scheme"] = df["url"].str.match(r"^\s*[A-Za-z][A-Za-z0-9+.\-]{0,15}://")
    df["url_norm"] = df["url"].map(normalize_url)
    df["host"] = df["url"].map(get_host)
    df["domain"] = df["host"].map(registered_domain)

    empty = (df["url_norm"] == "") | (df["host"] == "")
    rep["dropped_empty_url_or_host"] = int(empty.sum())
    df = df[~empty]

    # same normalised URL in both classes -> label conflict, drop all copies
    labels_per_url = df.groupby("url_norm")["label"].nunique()
    conflict = df["url_norm"].isin(labels_per_url[labels_per_url > 1].index)
    rep["dropped_label_conflicts"] = int(conflict.sum())
    df = df[~conflict]

    dup = df.duplicated("url_norm", keep="first")
    rep["dropped_duplicates_after_normalising"] = int(dup.sum())
    df = df[~dup]

    df["split"] = df["domain"].map(split_of)

    rep["scheme_share_by_label"] = (
        df.groupby("label")["had_scheme"].mean().round(4).to_dict()
    )
    rep["clean_rows"] = df["label"].value_counts().sort_index().to_dict()
    rep["rows_by_split_and_label"] = {
        s: g["label"].value_counts().sort_index().to_dict() for s, g in df.groupby("split")
    }
    doms = {s: set(g["domain"]) for s, g in df.groupby("split")}
    rep["domain_overlap_train_test"] = len(doms.get("train", set()) & doms.get("test", set()))
    rep["domain_overlap_train_val"] = len(doms.get("train", set()) & doms.get("val", set()))
    rep["unique_domains"] = int(df["domain"].nunique())
    top = df.groupby("label")["domain"].agg(lambda s: s.value_counts().head(5).to_dict())
    rep["top_domains_by_label"] = top.to_dict()
    return df.reset_index(drop=True), rep


def existing_images(raw_dir: Path) -> dict[str, set[str]]:
    """One directory scan per class (fast even for 500k files)."""
    out = {}
    for class_dir in config.CLASS_DIRS:
        qrs = find_class_root(raw_dir, class_dir) / "qrs"
        names: set[str] = set()
        if qrs.is_dir():
            with os.scandir(qrs) as it:
                names = {e.name for e in it if e.name.endswith(".png")}
        out[class_dir] = names
    return out


def image_path(raw_dir: Path, class_dir: str, img_file: str) -> Path:
    return find_class_root(raw_dir, class_dir) / "qrs" / img_file


def verify_decodes(raw_dir: Path, rows: pd.DataFrame) -> pd.Series:
    """True where the image decodes to exactly the URL in the CSV.

    Some CIC images are broken: every 114 px and 126 px image we tested is a
    cropped code with only one finder pattern (~7% of malicious, ~2% of
    benign). A model could learn "broken QR = malicious", so we drop them.
    """
    from quishguard.decode.decoder import decode  # imported here: needs OpenCV

    roots = {c: find_class_root(raw_dir, c) / "qrs" for c in config.CLASS_DIRS}
    ok = []
    for i, (c, f, u) in enumerate(zip(rows["class_dir"], rows["img_file"], rows["url"])):
        try:
            r = decode(roots[c] / f)
            ok.append(r.ok and r.text.strip() == u.strip())
        except Exception:
            ok.append(False)
        if (i + 1) % 5000 == 0:
            print(f"  verified {i + 1:,}/{len(rows):,}")
    return pd.Series(ok, index=rows.index)


def make_image_subset(df: pd.DataFrame, present: dict[str, set[str]],
                      per_class: dict[str, int], raw_dir: Path | None = None,
                      verify: bool = True) -> tuple[pd.DataFrame, dict]:
    has_img = [f in present.get(c, ()) for c, f in zip(df["class_dir"], df["img_file"])]
    pool = df[has_img]
    parts, rejected = [], {}
    for split, n in per_class.items():
        for label in (0, 1):
            g = pool[(pool["split"] == split) & (pool["label"] == label)]
            take = min(len(g), int(n * 1.15) if verify else n)  # oversample to replace broken ones
            s = g.sample(n=take, random_state=config.SEED)
            if verify and raw_dir is not None:
                print(f"Checking {split}/label={label}: {len(s):,} images")
                good = verify_decodes(raw_dir, s)
                rejected[f"{split}_label{label}"] = int((~good).sum())
                s = s[good]
            parts.append(s.head(n))
    sub = pd.concat(parts, ignore_index=True)
    cols = ["class_dir", "img_file", "label", "split", "url", "url_norm", "domain"]
    return sub[cols], rejected


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, default=config.RAW_DIR,
                    help="folder containing QR_All_benign and QR_All_Malicious")
    ap.add_argument("--out-dir", type=Path, default=config.PROCESSED_DIR)
    ap.add_argument("--train-per-class", type=int, default=40_000)
    ap.add_argument("--eval-per-class", type=int, default=5_000,
                    help="images per class for val and for test")
    ap.add_argument("--skip-images", action="store_true", help="do not scan image folders")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip decoding each subset image (faster, keeps broken QRs)")
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Reading CSVs from {args.raw_dir} ...")
    raw = load_raw(args.raw_dir)
    df, rep = clean(raw)

    out_csv = args.out_dir / "urls_clean.csv.gz"
    df.drop(columns=["had_scheme"]).to_csv(out_csv, index=False, compression="gzip")
    print(f"Wrote {len(df):,} rows -> {out_csv}")

    if not args.skip_images:
        print("Scanning image folders ...")
        present = existing_images(args.raw_dir)
        rep["images_on_disk"] = {k: len(v) for k, v in present.items()}
        sub, rejected = make_image_subset(
            df, present,
            {"train": args.train_per_class, "val": args.eval_per_class, "test": args.eval_per_class},
            raw_dir=args.raw_dir, verify=not args.no_verify,
        )
        rep["broken_images_rejected"] = rejected
        sub.to_csv(args.out_dir / "image_subset.csv", index=False)
        rep["image_subset"] = {
            s: g["label"].value_counts().sort_index().to_dict() for s, g in sub.groupby("split")
        }
        print(f"Wrote image subset: {len(sub):,} rows")

    rep_path = args.out_dir / "prepare_report.json"
    rep_path.write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps({k: v for k, v in rep.items() if k != "top_domains_by_label"}, indent=2, default=str))
    print(f"Report -> {rep_path}")


if __name__ == "__main__":
    main()
