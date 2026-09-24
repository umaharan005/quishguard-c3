"""Phase 3: train and evaluate the URL model (Module C2).

    python -m quishguard.url_model.train --data data/processed/urls_clean.csv.gz

What it does
  1. Loads the domain-split table from Phase 1.
  2. Caps rows per domain in TRAIN (default 100) so a few huge domains
     (duckdns.org, one IP with 10k URLs) do not dominate learning.
  3. Extracts lexical features from the normalised URL.
  4. Shortcut check: how well does "URL starts with http(s)://" alone separate
     the classes? (It is removed from our features.)
  5. Trains baselines (Logistic Regression, Random Forest, char n-gram LR)
     and the main model (XGBoost, early stopping on VAL).
  6. 10-fold StratifiedGroupKFold CV (groups = domain) on TRAIN.
  7. Calibrates XGBoost probabilities on VAL (isotonic) -> URL score 0-100.
  8. Reports TEST metrics, ROC curves, confusion matrix, SHAP/importance,
     latency, and saves the model bundle.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from quishguard import config
from quishguard.url_model.features import UrlFeaturizer

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover
    xgb = None


# ----------------------------------------------------------------- helpers
def cap_per_domain(df: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    if cap <= 0:
        return df
    return (df.sample(frac=1.0, random_state=seed)
              .groupby("domain", group_keys=False).head(cap)
              .sort_index())


def metrics(y, prob, thr=0.5) -> dict:
    pred = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, prob),
        "pr_auc": average_precision_score(y, prob),
        "fpr": fp / max(fp + tn, 1),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "threshold": float(thr),
    }


def threshold_for_fpr(y, prob, target_fpr) -> float:
    fpr, tpr, thr = roc_curve(y, prob)
    ok = np.where(fpr <= target_fpr)[0]
    return float(thr[ok[-1]]) if len(ok) else 1.0


def make_xgb(n_estimators=800, **kw):
    params = dict(n_estimators=n_estimators, learning_rate=0.08, max_depth=8,
                  subsample=0.9, colsample_bytree=0.8, min_child_weight=2,
                  reg_lambda=1.0, tree_method="hist", eval_metric="logloss",
                  n_jobs=-1, random_state=config.SEED)
    params.update(kw)
    if xgb is not None:
        return xgb.XGBClassifier(**params)
    from sklearn.ensemble import HistGradientBoostingClassifier  # fallback
    return HistGradientBoostingClassifier(max_iter=min(n_estimators, 300), learning_rate=0.1,
                                          max_leaf_nodes=63, random_state=config.SEED)


def fit_xgb(model, Xtr, ytr, Xva, yva, weights=None):
    if xgb is not None and isinstance(model, xgb.XGBClassifier):
        model.set_params(early_stopping_rounds=50)
        model.fit(Xtr, ytr, sample_weight=weights, eval_set=[(Xva, yva)], verbose=False)
    else:
        model.fit(Xtr, ytr, sample_weight=weights)
    return model


# ----------------------------------------------------------------- plots
def save_plots(out: Path, y_test, probs: dict, cm, importances: pd.Series, shap_ok: bool):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, p in probs.items():
        f, t, _ = roc_curve(y_test, p)
        ax.plot(f, t, label=f"{name} (AUC {roc_auc_score(y_test, p):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("URL model: ROC on test set"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "roc_test.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Benign", "Malicious"]); ax.set_yticks([0, 1], ["Benign", "Malicious"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("XGBoost: test confusion matrix")
    fig.tight_layout(); fig.savefig(out / "confusion_test.png", dpi=150); plt.close(fig)

    if not shap_ok:
        top = importances.sort_values().tail(20)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.barh(top.index, top.values); ax.set_title("Feature importance (gain)")
        fig.tight_layout(); fig.savefig(out / "feature_importance.png", dpi=150); plt.close(fig)


# ----------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=config.PROCESSED_DIR / "urls_clean.csv.gz")
    ap.add_argument("--model-out", type=Path, default=config.MODELS_DIR / "url_model.joblib")
    ap.add_argument("--reports", type=Path, default=config.REPORTS_DIR / "url_model")
    ap.add_argument("--max-per-domain", type=int, default=100, help="cap rows per domain in TRAIN (0 = no cap)")
    ap.add_argument("--sample", type=int, default=0, help="use only N rows per split (quick test)")
    ap.add_argument("--cv-folds", type=int, default=10)
    ap.add_argument("--cv-max-rows", type=int, default=200_000, help="rows used for CV (speed)")
    ap.add_argument("--baseline-max-rows", type=int, default=200_000)
    ap.add_argument("--gpu", action="store_true", help="XGBoost on CUDA (Colab T4)")
    ap.add_argument("--tranco-top", type=int, default=0,
                    help="add the top-N Tranco domains as benign homepage URLs (fixes the homepage bias)")
    ap.add_argument("--tranco-file", type=Path, default=None, help="local/cached Tranco csv or zip")
    ap.add_argument("--tld-smoothing", type=float, default=300.0,
                    help="pull rare TLDs toward the average risk (country TLDs are rare in the data)")
    ap.add_argument("--stack-char", action="store_true",
                    help="add an out-of-fold char n-gram score as an extra XGBoost feature")
    args = ap.parse_args(argv)
    args.reports.mkdir(parents=True, exist_ok=True)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    rng = config.SEED
    T = {}
    results: dict = {"xgboost_available": xgb is not None}

    t0 = time.time()
    df = pd.read_csv(args.data, dtype={"url": str, "url_norm": str, "domain": str}, keep_default_na=False)
    if args.tranco_top:
        from quishguard.data.tranco import tranco_rows
        mal_domains = set(df.loc[df["label"] == 1, "domain"])
        extra = tranco_rows(args.tranco_top, mal_domains, args.tranco_file)
        extra = extra[~extra["url_norm"].isin(set(df["url_norm"]))]
        results["tranco_rows_added"] = {s: int((extra["split"] == s).sum()) for s in ("train", "val", "test")}
        print("Added Tranco benign homepages:", results["tranco_rows_added"])
        df = pd.concat([df, extra[df.columns.intersection(extra.columns)]], ignore_index=True)
    splits = {s: df[df["split"] == s] for s in ("train", "val", "test")}
    if args.sample:
        splits = {s: g.sample(n=min(args.sample, len(g)), random_state=rng) for s, g in splits.items()}
    before = len(splits["train"])
    splits["train"] = cap_per_domain(splits["train"], args.max_per_domain, rng)
    results["rows"] = {s: {"total": len(g), "benign": int((g.label == 0).sum()), "malicious": int((g.label == 1).sum())}
                       for s, g in splits.items()}
    results["train_rows_before_domain_cap"] = before
    print(json.dumps(results["rows"], indent=1))

    # ---- shortcut check (scheme present in the raw URL)
    te = splits["test"]
    had_scheme = te["url"].str.match(r"^\s*[A-Za-z][A-Za-z0-9+.\-]{0,15}://").astype(int)
    results["shortcut_check"] = {
        "auc_scheme_only_on_test": roc_auc_score(te["label"], had_scheme),
        "note": "AUC of 'URL starts with http(s)://' alone. Removed from features by normalisation.",
    }
    print("Shortcut AUC (scheme only):", round(results["shortcut_check"]["auc_scheme_only_on_test"], 4))

    # ---- features
    print("Extracting features ...")
    ts = time.time()
    raw = {s: UrlFeaturizer.raw_frame(g["url_norm"].values) for s, g in splits.items()}
    T["feature_extraction_s"] = time.time() - ts
    feat = UrlFeaturizer(smoothing=args.tld_smoothing).fit(raw["train"], splits["train"]["label"].values)
    X = {s: feat.transform(r) for s, r in raw.items()}
    y = {s: g["label"].values for s, g in splits.items()}
    groups_train = splits["train"]["domain"].values
    X["train"].describe().T.to_csv(args.reports / "feature_summary_train.csv")

    probs_test: dict[str, np.ndarray] = {}
    results["test"] = {}

    def sub(n):
        idx = np.arange(len(y["train"]))
        if len(idx) > n:
            idx = np.random.default_rng(rng).choice(idx, n, replace=False)
        return idx

    # ---- baselines
    bi = sub(args.baseline_max_rows)
    print("Baseline: Logistic Regression")
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    lr.fit(X["train"].iloc[bi], y["train"][bi])
    probs_test["LogReg (features)"] = lr.predict_proba(X["test"])[:, 1]

    print("Baseline: Random Forest")
    rf = RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_leaf=2,
                                n_jobs=-1, random_state=rng)
    rf.fit(X["train"].iloc[bi], y["train"][bi])
    probs_test["Random Forest"] = rf.predict_proba(X["test"])[:, 1]

    print("Baseline: char n-gram TF-IDF + LR")
    tf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=200_000,
                         sublinear_tf=True, lowercase=True)
    Xt_tr = tf.fit_transform(splits["train"]["url_norm"].values[bi])
    lr_txt = LogisticRegression(max_iter=2000, C=4.0, solver="liblinear")
    lr_txt.fit(Xt_tr, y["train"][bi])
    probs_test["Char n-gram LR"] = lr_txt.predict_proba(tf.transform(splits["test"]["url_norm"].values))[:, 1]

    # ---- optional stacking: char n-gram score becomes one more XGBoost feature
    char_model = None
    if args.stack_char:
        print("Stacking: out-of-fold char n-gram scores on train (5 folds, grouped by domain) ...")
        ts = time.time()
        tf_s = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=300_000,
                               sublinear_tf=True, lowercase=True, dtype=np.float32)
        from quishguard.data.urls import mask_tld
        masked = {k: g["url_norm"].map(mask_tld).values for k, g in splits.items()}
        T_tr = tf_s.fit_transform(masked["train"])
        oof = np.zeros(len(y["train"]), dtype=np.float32)
        for a, b in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=rng).split(T_tr, y["train"], groups_train):
            m_ = LogisticRegression(max_iter=2000, C=4.0, solver="liblinear").fit(T_tr[a], y["train"][a])
            oof[b] = m_.predict_proba(T_tr[b])[:, 1]
        lr_s = LogisticRegression(max_iter=2000, C=4.0, solver="liblinear").fit(T_tr, y["train"])
        X["train"]["char_ngram_prob"] = oof
        for s in ("val", "test"):
            X[s]["char_ngram_prob"] = lr_s.predict_proba(tf_s.transform(masked[s]))[:, 1].astype(np.float32)
        char_model = {"vectorizer": tf_s, "lr": lr_s, "mask_tld": True}
        T["stacking_s"] = time.time() - ts
    feature_names = list(X["train"].columns)
    results["stacked_char_ngram"] = bool(args.stack_char)

    # ---- cross-validation (grouped by domain)
    print(f"{args.cv_folds}-fold StratifiedGroupKFold CV on train ...")
    ci = sub(args.cv_max_rows)
    Xc, yc, gc = X["train"].iloc[ci], y["train"][ci], groups_train[ci]
    cv_rows = []
    extra = {"device": "cuda"} if (args.gpu and xgb is not None) else {}
    for k, (a, b) in enumerate(StratifiedGroupKFold(n_splits=args.cv_folds, shuffle=True, random_state=rng).split(Xc, yc, gc)):
        m = make_xgb(n_estimators=400, **extra)
        m.fit(Xc.iloc[a], yc[a]) if xgb is None else m.fit(Xc.iloc[a], yc[a], verbose=False)
        p = m.predict_proba(Xc.iloc[b])[:, 1]
        r = metrics(yc[b], p); r["fold"] = k + 1
        cv_rows.append(r)
        print(f"  fold {k+1}: F1={r['f1']:.4f} AUC={r['roc_auc']:.4f}")
    cv = pd.DataFrame(cv_rows)
    cv.to_csv(args.reports / "cv_folds.csv", index=False)
    results["cv"] = {m: {"mean": float(cv[m].mean()), "std": float(cv[m].std())}
                     for m in ("accuracy", "precision", "recall", "f1", "roc_auc", "fpr")}

    # ---- main model
    print("Training XGBoost on full train (early stopping on val) ...")
    ts = time.time()
    model = fit_xgb(make_xgb(**extra), X["train"], y["train"], X["val"], y["val"])
    T["xgb_train_s"] = time.time() - ts
    if xgb is not None:
        results["xgb_best_iteration"] = int(getattr(model, "best_iteration", -1))
        model.set_params(device="cpu")  # predict on CPU (deployment)

    p_val_raw = model.predict_proba(X["val"])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p_val_raw, y["val"])
    p_test = iso.predict(model.predict_proba(X["test"])[:, 1])
    p_val = iso.predict(p_val_raw)
    probs_test["XGBoost (calibrated)"] = p_test

    thr_fpr3 = threshold_for_fpr(y["val"], p_val, 0.03)
    thr_fpr1 = threshold_for_fpr(y["val"], p_val, 0.01)
    results["thresholds_from_val"] = {"default": 0.5, "fpr_3pct": thr_fpr3, "fpr_1pct": thr_fpr1}

    for name, p in probs_test.items():
        results["test"][name] = metrics(y["test"], p)
    results["test"]["XGBoost @ val FPR<=3%"] = metrics(y["test"], p_test, thr_fpr3)
    results["test"]["XGBoost @ val FPR<=1%"] = metrics(y["test"], p_test, thr_fpr1)

    # robustness: at most 20 URLs per domain in test, so no single domain dominates
    te_c = cap_per_domain(splits["test"].assign(_p=p_test), 20, rng)
    results["test_capped_20_per_domain"] = metrics(te_c["label"].values, te_c["_p"].values)

    # ---- explanations
    importances = pd.Series(getattr(model, "feature_importances_", np.zeros(X["train"].shape[1])),
                            index=feature_names)
    importances.sort_values(ascending=False).to_csv(args.reports / "feature_importance.csv")
    shap_ok = False
    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = X["test"].sample(n=min(3000, len(X["test"])), random_state=rng)
        sv = shap.TreeExplainer(model).shap_values(xs)
        shap.summary_plot(sv, xs, show=False, max_display=20)
        plt.tight_layout(); plt.savefig(args.reports / "shap_summary.png", dpi=150); plt.close()
        pd.Series(np.abs(sv).mean(0), index=xs.columns).sort_values(ascending=False) \
          .to_csv(args.reports / "shap_mean_abs.csv")
        shap_ok = True
    except Exception as e:  # shap missing or incompatible
        print("SHAP skipped:", e)

    cm = confusion_matrix(y["test"], (p_test >= 0.5).astype(int), labels=[0, 1])
    save_plots(args.reports, y["test"], probs_test, cm, importances, shap_ok)

    # ---- bundle + latency
    bundle = {"featurizer": feat, "model": model, "calibrator": iso, "char_model": char_model,
              "thresholds": results["thresholds_from_val"], "feature_names": feature_names,
              "version": "url-v1-stacked" if char_model else "url-v1", "trained_rows": int(len(y["train"]))}
    joblib.dump(bundle, args.model_out)

    from quishguard.url_model.predict import UrlScorer
    scorer = UrlScorer(args.model_out)
    sample_urls = splits["test"]["url"].sample(n=min(300, len(splits["test"])), random_state=rng).tolist()
    ts = time.perf_counter()
    for u in sample_urls:
        scorer.score(u, explain=False)
    T["latency_ms_per_url"] = (time.perf_counter() - ts) * 1000 / len(sample_urls)
    results["timing"] = T
    results["total_minutes"] = (time.time() - t0) / 60

    (args.reports / "metrics.json").write_text(json.dumps(results, indent=2, default=float))
    table = pd.DataFrame(results["test"]).T[["accuracy", "precision", "recall", "f1", "roc_auc", "fpr"]]
    table.round(4).to_csv(args.reports / "test_results_table.csv")
    print("\nTEST RESULTS\n", table.round(4).to_string())
    print("\nCV (XGBoost, grouped by domain):",
          {k: f"{v['mean']:.4f} ± {v['std']:.4f}" for k, v in results["cv"].items()})
    print(f"\nModel -> {args.model_out}\nReports -> {args.reports}")


if __name__ == "__main__":
    main()
