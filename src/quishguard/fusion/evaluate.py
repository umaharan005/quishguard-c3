"""Phase 6: evaluate fusion + severity on the tamper val/test sets.

Each image has two truths:
  url_malicious: the link it really opens is malicious (the CIC label of the
                 decoded link; for stickers that is the attacker's link)
  tampered:      the image was physically altered
Reference tier (proposal 3.7.6): both -> Critical, one -> Phishing, none -> Safe.
Alert truth: url_malicious OR tampered.

Compares URL-only, visual-only and weighted fusion (w_url = 0.5 ... 0.8, with
and without the tamper floor), chooses the weight on VAL, and reports TEST.

    python -m quishguard.fusion.evaluate --models-dir .../models --tamper-dir /content/tamper \
        --urls .../processed/urls_clean.csv.gz --reports .../reports/fusion
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quishguard.data.urls import normalize_url
from quishguard.decode.decoder import payload_type
from quishguard.fusion.scoring import fuse, ground_truth_tier

TIER_ORDER = ["Safe", "Suspicious", "Phishing", "Critical"]
WEIGHTS = [0.5, 0.6, 0.65, 0.7, 0.8]


def score_streams(man: pd.DataFrame, models_dir: Path, tamper_dir: Path, url_model: str) -> pd.DataFrame:
    import joblib
    import torch
    from torch.utils.data import DataLoader

    from quishguard.url_model.predict import UrlScorer
    from quishguard.vit.train import Files, load_visual, score_loader

    m = man.copy()
    url = UrlScorer(models_dir / url_model)
    m["payload"] = [payload_type(t) if d else "none" for t, d in zip(m["decoded_text"].fillna(""), m["decodes"])]
    m["url_score"] = [url.score(t, explain=False)["url_score"] if p == "url" else np.nan
                      for t, p in zip(m["decoded_text"].fillna(""), m["payload"])]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vis = load_visual(models_dir / "visual_patchcore.pt", device)
    c = joblib.load(models_dir / "visual_patchcore_calibrator.joblib")
    err = score_loader(vis, DataLoader(Files([tamper_dir / f for f in m["file"]]), batch_size=64, num_workers=2), device)
    m["visual_score"] = 100 * c["calibrator"].predict_proba(np.log(err[:, None] + 1e-8))[:, 1]
    m["visual_tampered_flag"] = err >= c["threshold_error"]
    return m


def add_truth(m: pd.DataFrame, urls: pd.DataFrame) -> pd.DataFrame:
    by_url = dict(zip(urls["url"].astype(str).str.strip(), urls["label"]))
    by_norm = dict(zip(urls["url_norm"].astype(str), urls["label"]))

    def lab(row):
        if row["decodes"] and str(row["decoded_text"]).strip():
            t = str(row["decoded_text"]).strip()
            if t in by_url:
                return int(by_url[t])
            n = normalize_url(t)
            if n in by_norm:
                return int(by_norm[n])
            return np.nan
        return int(row["base_label"])       # unreadable: the link that was originally there

    m = m.copy()
    m["url_malicious"] = m.apply(lab, axis=1)
    m = m[m["url_malicious"].notna()].copy()
    m["url_malicious"] = m["url_malicious"].astype(int)
    m["gt_tier"] = [ground_truth_tier(u == 1, t == 1) for u, t in zip(m["url_malicious"], m["tampered"])]
    m["should_alert"] = ((m["url_malicious"] == 1) | (m["tampered"] == 1)).astype(int)
    return m


def evaluate(m: pd.DataFrame, w: float | str, floor: bool) -> dict:
    from sklearn.metrics import cohen_kappa_score, f1_score, precision_score, recall_score, roc_auc_score
    from scipy.stats import pearsonr

    if w == "url_only":
        s = [(u if not np.isnan(u) else 0.0) for u in m["url_score"]]
        t = [fuse(u if not np.isnan(u) else 0.0, 0.0, w_url=1.0, tamper_floor=0).tier for u in m["url_score"]]
    elif w == "visual_only":
        s = list(m["visual_score"]); t = [fuse(None, v).tier for v in s]
    else:
        fs = [fuse(None if np.isnan(u) else u, v, w_url=w, tamper_floor=0.6 if floor else 0)
              for u, v in zip(m["url_score"], m["visual_score"])]
        s = [f.score for f in fs]; t = [f.tier for f in fs]
    s = np.array(s); t = np.array(t)
    y = m["should_alert"].values
    pred = np.isin(t, ["Phishing", "Critical"]).astype(int)
    gt = m["gt_tier"].values
    gi = np.array([TIER_ORDER.index(g) for g in gt]); ti = np.array([TIER_ORDER.index(x) for x in t])
    fp = int(((pred == 1) & (y == 0)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    return {
        "alert_auroc": float(roc_auc_score(y, s)),
        "alert_precision": float(precision_score(y, pred, zero_division=0)),
        "alert_recall": float(recall_score(y, pred, zero_division=0)),
        "alert_f1": float(f1_score(y, pred, zero_division=0)),
        "alert_fpr": fp / max(fp + tn, 1),
        "tier_accuracy": float((gt == t).mean()),
        "tier_within_one": float((np.abs(gi - ti) <= 1).mean()),
        "tier_kappa_linear": float(cohen_kappa_score(gi, ti, weights="linear")),
        "pearson_score_vs_gt": float(pearsonr(s, gi)[0]),
        "_tiers": t, "_scores": s,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-dir", type=Path, required=True)
    ap.add_argument("--tamper-dir", type=Path, required=True)
    ap.add_argument("--urls", type=Path, required=True, help="urls_clean.csv.gz (to look up the true label of decoded links)")
    ap.add_argument("--reports", type=Path, required=True)
    ap.add_argument("--url-model", default="url_model_v2.joblib")
    args = ap.parse_args(argv)
    args.reports.mkdir(parents=True, exist_ok=True)

    man = pd.read_csv(args.tamper_dir / "manifest.csv", keep_default_na=False)
    man["decoded_text"] = man["decoded_text"].astype(str)
    urls = pd.read_csv(args.urls, usecols=["url", "url_norm", "label"], dtype={"url": str, "url_norm": str}, keep_default_na=False)
    print("Scoring URL and visual streams for", len(man), "images ...")
    scored = add_truth(score_streams(man, args.models_dir, args.tamper_dir, args.url_model), urls)
    scored.drop(columns=["params"], errors="ignore").to_csv(args.reports / "scored_images.csv", index=False)
    print("Images with known truth:", len(scored), "| truth tiers:", scored["gt_tier"].value_counts().to_dict())

    configs = [("url_only", False), ("visual_only", False)] + [(w, False) for w in WEIGHTS] + [(w, True) for w in WEIGHTS]
    rows = {}
    for split in ("val", "test"):
        d = scored[scored["split"] == split]
        for w, fl in configs:
            name = w if isinstance(w, str) else f"w_url={w:.2f}{' +floor' if fl else ''}"
            r = evaluate(d, w, fl)
            rows[(split, name)] = {k: v for k, v in r.items() if not k.startswith("_")}
    tab = pd.DataFrame(rows).T
    tab.index.names = ["split", "config"]
    tab.round(4).to_csv(args.reports / "ablation.csv")

    val = tab.loc["val"]
    best = val.loc[[i for i in val.index if i.startswith("w_url")], "tier_accuracy"].idxmax()
    w_best = float(best.split("=")[1].split()[0]); floor_best = "+floor" in best
    test_d = scored[scored["split"] == "test"]
    r = evaluate(test_d, w_best, floor_best)
    test_d = test_d.assign(pred_tier=r["_tiers"], fused_score=r["_scores"])
    test_d.drop(columns=["params"], errors="ignore").to_csv(args.reports / "test_fused.csv", index=False)
    cm = pd.crosstab(pd.Categorical(test_d["gt_tier"], TIER_ORDER), pd.Categorical(test_d["pred_tier"], TIER_ORDER),
                     rownames=["truth"], colnames=["predicted"], dropna=False)
    cm.to_csv(args.reports / "tier_confusion_test.csv")
    by_attack = pd.DataFrame([{"attack": a, "n": len(g),
                               "alert_rate": float(np.isin(g["pred_tier"], ["Phishing", "Critical"]).mean()),
                               "mean_score": float(g["fused_score"].mean())}
                              for a, g in test_d.groupby("attack")]).set_index("attack")
    by_attack.round(4).to_csv(args.reports / "test_by_attack.csv")

    summary = {"chosen_on_val": best, "w_url": w_best, "tamper_floor": floor_best,
               "test": tab.loc[("test", best)].to_dict(),
               "test_url_only": tab.loc[("test", "url_only")].to_dict(),
               "test_visual_only": tab.loc[("test", "visual_only")].to_dict()}
    (args.reports / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4.8, 4))
        ax.imshow(cm.values, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm.values):
            ax.text(j, i, f"{v:,}", ha="center", va="center", color="white" if v > cm.values.max() / 2 else "black")
        ax.set_xticks(range(4), TIER_ORDER, rotation=20); ax.set_yticks(range(4), TIER_ORDER)
        ax.set_xlabel("Predicted tier"); ax.set_ylabel("Reference tier")
        ax.set_title(f"Severity tiers on test ({best})"); fig.tight_layout()
        fig.savefig(args.reports / "tier_confusion_test.png", dpi=150); plt.close(fig)
    except Exception as e:
        print("plot skipped:", e)

    show = ["alert_auroc", "alert_precision", "alert_recall", "alert_f1", "alert_fpr",
            "tier_accuracy", "tier_within_one", "tier_kappa_linear", "pearson_score_vs_gt"]
    print("\nTEST ABLATION\n", tab.loc["test"][show].round(4).to_string())
    print(f"\nChosen on val: {best}")
    print("\nTier confusion (test):\n", cm.to_string())
    print("\nBy attack (test):\n", by_attack.round(3).to_string())


if __name__ == "__main__":
    main()
