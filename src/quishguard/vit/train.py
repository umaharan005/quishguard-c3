"""Phase 5: train and evaluate the visual anomaly detector (Module C1).

    python -m quishguard.vit.train --arch vit --raw-dir /content/subset \
        --subset .../image_subset.csv --tamper-dir /content/tamper --out .../models --reports .../reports/vit

Training uses ONLY clean TRAIN images (with everyday photo effects added on
the fly). Nothing tampered is ever seen in training.
Evaluation:
  * val tamper set  -> choose the threshold (5% false alarms on normal images)
                       and fit a small calibrator: error -> Visual Anomaly Score 0-100
  * test tamper set -> AUROC, F1, per-attack results, and results on the
                       "still scans" subset (the attacks a URL check cannot see)
  * CIC test images, benign vs malicious, clean -> expected AUROC about 0.5
                       (supports Gap 1: pixels do not reveal intent)
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from quishguard import config
from quishguard.data.prepare import find_class_root
from quishguard.decode.decoder import to_gray
from quishguard.tamper.generate import photo_effects, recrop
from quishguard.vit.model import anomaly_score, build, patch_errors
from quishguard.vit.preprocess import prepare


class CleanQR(Dataset):
    """Clean dataset QR codes + random everyday photo effects (no tampering)."""

    def __init__(self, rows: pd.DataFrame, raw_dir: Path, augment: bool, seed: int = 0):
        self.rows = rows.reset_index(drop=True)
        self.roots = {c: find_class_root(raw_dir, c) / "qrs" for c in config.CLASS_DIRS}
        self.augment, self.seed = augment, seed

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows.iloc[i]
        g = to_gray(self.roots[r["class_dir"]] / r["img_file"])
        # training: new random effects every time; val/test: fixed effects per image (repeatable)
        rng = random.Random() if self.augment else random.Random(self.seed * 1_000_003 + i)
        g = photo_effects(recrop(g, rng), rng)
        return torch.from_numpy(prepare(g))[None]


class Files(Dataset):
    def __init__(self, paths):
        self.paths = list(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return torch.from_numpy(prepare(self.paths[i]))[None]


@torch.no_grad()
def score_loader(model, loader, device, keep_maps: int = 0):
    model.eval()
    scores, maps = [], []
    for x in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            rec = model(x)
        pe = patch_errors(x.float(), rec.float())
        scores.append(anomaly_score(pe).cpu())
        if keep_maps and sum(m.shape[0] for m in maps) < keep_maps:
            maps.append(pe.cpu())
    s = torch.cat(scores).numpy()
    return (s, torch.cat(maps).numpy()) if keep_maps else s


def metrics_at(y, s, thr):
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
    p = (s >= thr).astype(int)
    fp = int(((p == 1) & (y == 0)).sum()); tn = int(((p == 0) & (y == 0)).sum())
    return {"auroc": float(roc_auc_score(y, s)) if len(set(y)) > 1 else float("nan"),
            "precision": float(precision_score(y, p, zero_division=0)),
            "recall": float(recall_score(y, p, zero_division=0)),
            "f1": float(f1_score(y, p, zero_division=0)),
            "fpr": fp / max(fp + tn, 1), "n": int(len(y))}


def save_examples(path: Path, tamper_dir: Path, man: pd.DataFrame, maps: np.ndarray, per=3):
    import cv2
    tiles = []
    for kind in ["normal", "sticker", "logo", "module_flip", "warp", "double_print"]:
        idx = man.index[man["attack"] == kind][:per]
        row = []
        for i in idx:
            if i >= len(maps):
                continue
            img = (prepare(tamper_dir / man.loc[i, "file"]) * 255).astype(np.uint8)
            hm = cv2.resize(maps[i] / (maps[: len(man)].max() + 1e-9), (224, 224), interpolation=cv2.INTER_CUBIC)
            hm = cv2.applyColorMap(np.clip(hm * 255 * 3, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
            over = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), 0.55, hm, 0.45, 0)
            row.append(np.hstack([cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), over]))
        if row:
            r = np.hstack(row)
            r = cv2.copyMakeBorder(r, 22, 4, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            cv2.putText(r, kind, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
            tiles.append(r)
    w = max(t.shape[1] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, 0, 0, w - t.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in tiles]
    cv2.imwrite(str(path), np.vstack(tiles))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", choices=["vit", "cnn"], default="vit")
    ap.add_argument("--raw-dir", type=Path, required=True, help="folder with QR_All_benign/ and QR_All_Malicious/ (subset)")
    ap.add_argument("--subset", type=Path, required=True, help="image_subset.csv")
    ap.add_argument("--tamper-dir", type=Path, required=True, help="unzipped tamper set (has manifest.csv)")
    ap.add_argument("--out", type=Path, default=config.MODELS_DIR)
    ap.add_argument("--reports", type=Path, default=config.REPORTS_DIR / "vit")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--train-limit", type=int, default=0, help="use only N train images (0 = all 80k)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--smoke", action="store_true", help="tiny run to check everything works (2 min)")
    args = ap.parse_args(argv)
    if args.smoke:
        args.epochs, args.train_limit = 1, 512
    args.out.mkdir(parents=True, exist_ok=True)
    args.reports.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.SEED); np.random.seed(config.SEED); random.seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    res: dict = {"arch": args.arch, "device": str(device)}

    sub = pd.read_csv(args.subset)
    tr = sub[sub["split"] == "train"]
    if args.train_limit:
        tr = tr.sample(n=min(args.train_limit, len(tr)), random_state=config.SEED)
    va_clean = sub[sub["split"] == "val"].sample(n=2000 if not args.smoke else 128, random_state=config.SEED)
    res["train_images"] = len(tr)

    dl_tr = DataLoader(CleanQR(tr, args.raw_dir, augment=True), batch_size=args.batch, shuffle=True,
                       num_workers=args.workers, pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    dl_va = DataLoader(CleanQR(va_clean, args.raw_dir, augment=False, seed=1), batch_size=args.batch,
                       num_workers=args.workers)

    model = build(args.arch).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    res["parameters"] = int(n_params)
    print(f"{args.arch} autoencoder: {n_params/1e6:.2f} M parameters, {len(tr):,} training images")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    steps = args.epochs * len(dl_tr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=max(steps, 1), pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    history, best, ckpt = [], float("inf"), args.out / f"visual_{args.arch}.pt"
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train(); tot = n = 0
        for x in dl_tr:
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                rec = model(x)
                loss = torch.nn.functional.mse_loss(rec.float(), x)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * x.shape[0]; n += x.shape[0]
        va = float(np.mean(score_loader(model, dl_va, device)))
        history.append({"epoch": ep, "train_mse": tot / n, "val_normal_score": va})
        print(f"epoch {ep:2d}  train MSE {tot/n:.5f}  val normal anomaly score {va:.5f}  ({(time.time()-t0)/60:.1f} min)")
        if va < best:
            best = va
            torch.save({"arch": args.arch, "state_dict": model.state_dict()}, ckpt)
    res["train_minutes"] = (time.time() - t0) / 60
    pd.DataFrame(history).to_csv(args.reports / "history.csv", index=False)

    # ---------------------------------------------------------------- evaluation
    model.load_state_dict(torch.load(ckpt, map_location=device)["state_dict"])
    man = pd.read_csv(args.tamper_dir / "manifest.csv")
    if args.smoke:
        man = man.groupby(["split", "attack"], group_keys=False).head(20)
    out = {}
    for split in ("val", "test"):
        m = man[man["split"] == split].reset_index(drop=True)
        dl = DataLoader(Files([args.tamper_dir / f for f in m["file"]]), batch_size=args.batch, num_workers=args.workers)
        if split == "test":
            s, maps = score_loader(model, dl, device, keep_maps=len(m))
        else:
            s = score_loader(model, dl, device)
        m["error"] = s
        out[split] = m
    va, te = out["val"], out["test"]

    # threshold: 5% false alarms on val normal images
    thr = float(np.quantile(va.loc[va.attack == "normal", "error"], 0.95))
    # calibrator: raw error -> probability of tampering -> Visual Anomaly Score 0-100
    from sklearn.linear_model import LogisticRegression
    cal = LogisticRegression().fit(np.log(va[["error"]].values + 1e-8), va["tampered"].values)
    te["visual_score"] = 100 * cal.predict_proba(np.log(te[["error"]].values + 1e-8))[:, 1]

    y, s = te["tampered"].values, te["error"].values
    res["threshold_error"] = thr
    res["test_overall"] = metrics_at(y, s, thr)
    per = {}
    for a in [k for k in te["attack"].unique() if k != "normal"]:
        mm = te[te["attack"].isin(["normal", a])]
        per[a] = metrics_at(mm["tampered"].values, mm["error"].values, thr)
    res["test_per_attack"] = per
    dec = te[(te["decodes"] == 1)]
    res["test_still_scans_only"] = metrics_at(dec["tampered"].values, dec["error"].values, thr)
    res["note_still_scans"] = "Only images a phone can still read. Here decoding succeeds, so a decode-failure check cannot help; the visual stream has to."

    # Gap 1 check: clean CIC test images, benign vs malicious
    ct = pd.concat([g.sample(n=min(len(g), 2000 if not args.smoke else 64), random_state=config.SEED)
                    for _, g in sub[sub["split"] == "test"].groupby("label")], ignore_index=True)
    dl = DataLoader(CleanQR(ct, args.raw_dir, augment=False, seed=2), batch_size=args.batch, num_workers=args.workers)
    sc = score_loader(model, dl, device)
    from sklearn.metrics import roc_auc_score
    res["cic_benign_vs_malicious_auroc"] = float(roc_auc_score(ct["label"].values, sc))
    res["cic_note"] = "About 0.5 means the anomaly score does not depend on whether the URL is malicious: the visual stream detects tampering, not intent (Gap 1)."

    # latency (single image, includes preprocessing)
    f0 = args.tamper_dir / te["file"].iloc[0]
    x = torch.from_numpy(prepare(f0))[None, None].to(device)
    with torch.no_grad():
        for _ in range(5):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(50):
            x = torch.from_numpy(prepare(f0))[None, None].to(device)
            pe = patch_errors(x, model(x)); anomaly_score(pe)
        if device.type == "cuda":
            torch.cuda.synchronize()
    res["latency_ms_per_image"] = (time.perf_counter() - t) * 1000 / 50

    # save
    import joblib
    joblib.dump({"calibrator": cal, "threshold_error": thr, "arch": args.arch}, args.out / f"visual_{args.arch}_calibrator.joblib")
    te.drop(columns=["params"]).to_csv(args.reports / "test_scores.csv", index=False)
    (args.reports / "metrics.json").write_text(json.dumps(res, indent=2, default=float))
    save_examples(args.reports / "heatmaps.png", args.tamper_dir, te, maps)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for a in ["normal", "sticker", "logo", "module_flip", "warp", "double_print"]:
            ax.hist(np.log10(te.loc[te.attack == a, "error"] + 1e-8), bins=60, alpha=0.5, label=a, density=True)
        ax.axvline(np.log10(thr), color="k", ls="--", lw=1, label="threshold (5% FPR on val)")
        ax.set_xlabel("log10 anomaly score (patch reconstruction error)"); ax.set_ylabel("density")
        ax.set_title(f"{args.arch.upper()} autoencoder: test anomaly scores"); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(args.reports / "score_hist.png", dpi=150); plt.close(fig)
    except Exception as e:
        print("plot skipped:", e)

    table = pd.DataFrame({**{"ALL tampered vs normal": res["test_overall"]},
                          **{f"{k} vs normal": v for k, v in per.items()},
                          "still-scans subset": res["test_still_scans_only"]}).T
    table = table[["auroc", "precision", "recall", "f1", "fpr", "n"]]
    table.round(4).to_csv(args.reports / "test_results_table.csv")
    print("\nTEST RESULTS\n", table.round(4).to_string())
    print("\nCIC benign vs malicious AUROC (expect ~0.5):", round(res["cic_benign_vs_malicious_auroc"], 4))
    print("Latency ms/image:", round(res["latency_ms_per_image"], 2))
    print(f"\nModel -> {ckpt}\nReports -> {args.reports}")


if __name__ == "__main__":
    main()
