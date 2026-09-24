"""Score one URL with the trained URL model: 0-100 score plus reasons."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from quishguard import config
from quishguard.data.urls import mask_tld, normalize_url
from quishguard.url_model.features import UrlFeaturizer

READABLE = {
    "len_url": "URL length", "len_host": "host length", "len_path": "path length",
    "len_query": "query length", "digit_ratio": "share of digits", "letter_ratio": "share of letters",
    "symbol_ratio": "share of symbols", "upper_ratio": "share of capitals",
    "entropy_url": "randomness of URL", "entropy_host": "randomness of host name",
    "host_digit_ratio": "digits in host name", "host_hyphens": "hyphens in host name",
    "n_dots": "number of dots", "n_host_labels": "host name parts",
    "n_subdomain_labels": "number of subdomains", "len_subdomain": "subdomain length",
    "len_domain_label": "domain name length", "len_tld": "TLD length", "is_ip": "raw IP address as host",
    "has_port": "explicit port number", "path_depth": "path depth", "n_query_params": "query parameters",
    "n_at": "'@' characters", "n_percent": "'%' encoding", "n_equals": "'=' characters",
    "n_tilde": "'~' characters", "n_underscore": "'_' characters", "has_punycode": "punycode (look-alike letters)",
    "has_double_slash_in_path": "'//' inside path", "n_suspicious_words": "phishing words (login, verify…)",
    "is_shortener": "link shortener", "is_free_host": "free hosting / dynamic DNS",
    "risky_ext": "download file type (.exe, .apk, .sh…)", "web_ext": "script page (.php, .html…)",
    "max_token_len": "longest word in URL", "n_tokens": "number of URL parts",
    "max_char_run": "repeated characters", "host_vowel_ratio": "vowel share in host (gibberish check)",
    "brand_in_subdomain": "brand name in subdomain", "tld_risk": "TLD often used for phishing",
    "char_ngram_prob": "character patterns look like known phishing URLs",
}


class UrlScorer:
    def __init__(self, path: Path | str = config.MODELS_DIR / "url_model.joblib"):
        b = joblib.load(path)
        self.feat: UrlFeaturizer = b["featurizer"]
        self.model = b["model"]
        self.cal = b["calibrator"]
        self.thresholds = b["thresholds"]
        self.names = b["feature_names"]
        self.char = b.get("char_model")
        self._booster = None
        try:
            import xgboost as xgb
            if isinstance(self.model, xgb.XGBClassifier):
                self._booster = self.model.get_booster()
                self._xgb = xgb
        except ImportError:
            pass

    def score(self, url: str, explain: bool = True, top_k: int = 5) -> dict:
        X = self.feat.transform(UrlFeaturizer.raw_frame([url]))
        if self.char is not None:
            text = normalize_url(url)
            if self.char.get("mask_tld"):
                text = mask_tld(text)
            t = self.char["vectorizer"].transform([text])
            X["char_ngram_prob"] = self.char["lr"].predict_proba(t)[:, 1].astype(np.float32)
        X = X[self.names]
        raw_p = float(self.model.predict_proba(X)[0, 1])
        p = float(self.cal.predict([raw_p])[0])
        out = {"url_score": round(100 * p, 2), "probability": p,
               "malicious": p >= self.thresholds.get("default", 0.5)}
        if explain and self._booster is not None:
            contrib = self._booster.predict(self._xgb.DMatrix(X), pred_contribs=True)[0][:-1]
            order = np.argsort(-np.abs(contrib))[:top_k]
            out["reasons"] = [{
                "feature": self.names[i],
                "text": READABLE.get(self.names[i], self.names[i]),
                "value": round(float(X.iloc[0, i]), 4),
                "effect": "raises risk" if contrib[i] > 0 else "lowers risk",
                "weight": round(float(contrib[i]), 4),
            } for i in order]
        return out
