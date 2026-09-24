"""Module C2: lexical URL features.

All features are computed on the NORMALISED URL (scheme and "www." removed),
so the model cannot use "starts with http://" as a shortcut. Nothing here
touches the network.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
import pandas as pd

from quishguard.data.urls import get_host, is_ip, normalize_url, registered_domain

SUSPICIOUS_WORDS = (
    "login", "log-in", "signin", "sign-in", "verify", "verification", "account",
    "update", "secure", "security", "confirm", "banking", "bank", "password",
    "passwd", "wallet", "invoice", "payment", "billing", "support", "unlock",
    "suspend", "webscr", "cmd=", "auth", "token", "reset", "recover", "gift",
    "bonus", "free", "promo", "claim", "paypal", "apple", "microsoft", "office365",
    "outlook", "netflix", "amazon", "dhl", "fedex", "usps",
)
SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly",
    "cutt.ly", "rebrand.ly", "shorturl.at", "rb.gy", "tiny.cc", "t.ly", "s.id",
    "v.gd", "qrco.de", "shorturl.asia", "bl.ink",
}
# free hosting / dynamic DNS / page builders often abused for phishing
FREE_HOSTS = {
    "duckdns.org", "000webhostapp.com", "workers.dev", "pages.dev", "web.app",
    "firebaseapp.com", "weebly.com", "wixsite.com", "blogspot.com", "github.io",
    "glitch.me", "herokuapp.com", "netlify.app", "vercel.app", "ngrok.io",
    "ngrok-free.app", "no-ip.com", "ddns.net", "hopto.org", "zapto.org",
    "sites.google.com", "googleusercontent.com", "square.site", "godaddysites.com",
    "webflow.io", "r2.dev", "ipfs.io", "azurewebsites.net", "appspot.com",
}
RISKY_EXT = (".exe", ".apk", ".scr", ".bat", ".sh", ".bin", ".jar", ".msi", ".dll",
             ".zip", ".rar", ".7z", ".js", ".vbs", ".ps1", ".m", ".mips", ".arm7")
WEB_EXT = (".php", ".html", ".htm", ".asp", ".aspx", ".jsp", ".cgi")

_SYMBOLS = set("-_.~!*'();:@&=+$,/?#[]%")


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in Counter(s).values())


def _max_run(s: str) -> int:
    best = run = 0
    prev = None
    for ch in s:
        run = run + 1 if ch == prev else 1
        best = max(best, run)
        prev = ch
    return best


def extract_one(url: str) -> dict:
    """Features for one URL (raw or already normalised)."""
    u = normalize_url(url)
    host = get_host(u)
    dom = registered_domain(host)
    ip = is_ip(host)
    rest = u[len(host):] if u.lower().startswith(host) else u.split("/", 1)[-1]
    has_port = bool(re.match(r"^:\d{1,5}", rest))
    path = re.split(r"[?#]", rest, maxsplit=1)[0]
    query = rest.split("?", 1)[1] if "?" in rest else ""
    low = u.lower()
    n = max(len(u), 1)
    labels = host.split(".") if host and not ip else []
    sub = host[: -len(dom)].rstrip(".") if dom and host.endswith(dom) and not ip else ""
    tld = labels[-1] if labels else ("ip" if ip else "")
    tokens = [t for t in re.split(r"[./?=&_\-:%]", u) if t]
    last_seg = path.rstrip("/").split("/")[-1].lower() if path else ""

    return {
        "len_url": len(u),
        "len_host": len(host),
        "len_path": len(path),
        "len_query": len(query),
        "digit_ratio": sum(c.isdigit() for c in u) / n,
        "letter_ratio": sum(c.isalpha() for c in u) / n,
        "symbol_ratio": sum(c in _SYMBOLS for c in u) / n,
        "upper_ratio": sum(c.isupper() for c in u) / n,
        "entropy_url": _entropy(u),
        "entropy_host": _entropy(host),
        "host_digit_ratio": sum(c.isdigit() for c in host) / max(len(host), 1),
        "host_hyphens": host.count("-"),
        "n_dots": u.count("."),
        "n_host_labels": len(labels),
        "n_subdomain_labels": len([x for x in sub.split(".") if x]),
        "len_subdomain": len(sub),
        "len_domain_label": len(dom.split(".")[0]) if dom and not ip else 0,
        "len_tld": len(tld) if not ip else 0,
        "is_ip": int(ip),
        "has_port": int(has_port),
        "path_depth": path.count("/"),
        "n_query_params": query.count("&") + (1 if query else 0),
        "n_at": u.count("@"),
        "n_percent": u.count("%"),
        "n_equals": u.count("="),
        "n_tilde": u.count("~"),
        "n_underscore": u.count("_"),
        "has_punycode": int("xn--" in host),
        "has_double_slash_in_path": int("//" in path),
        "n_suspicious_words": sum(w in low for w in SUSPICIOUS_WORDS),
        "is_shortener": int(host in SHORTENERS or dom in SHORTENERS),
        "is_free_host": int(dom in FREE_HOSTS or host in FREE_HOSTS
                            or any(host.endswith("." + f) for f in FREE_HOSTS)),
        "risky_ext": int(last_seg.endswith(RISKY_EXT)),
        "web_ext": int(last_seg.endswith(WEB_EXT)),
        "max_token_len": max((len(t) for t in tokens), default=0),
        "n_tokens": len(tokens),
        "max_char_run": _max_run(u),
        "host_vowel_ratio": sum(c in "aeiou" for c in host) / max(sum(c.isalpha() for c in host), 1),
        "brand_in_subdomain": int(any(b in sub for b in ("paypal", "apple", "microsoft", "google",
                                                        "amazon", "netflix", "bank", "office"))),
        "_tld": tld,  # turned into tld_risk by UrlFeaturizer (fit on train only)
    }


class UrlFeaturizer:
    """Turns URLs into a numeric matrix. `fit` learns TLD risk from TRAIN only."""

    def __init__(self, smoothing: float = 20.0):
        self.smoothing = smoothing
        self.tld_risk_: dict[str, float] = {}
        self.prior_: float = 0.5
        self.feature_names_: list[str] = []

    @staticmethod
    def raw_frame(urls) -> pd.DataFrame:
        return pd.DataFrame([extract_one(u) for u in urls])

    def fit(self, raw: pd.DataFrame, y) -> "UrlFeaturizer":
        y = np.asarray(y)
        self.prior_ = float(y.mean())
        stats = pd.DataFrame({"tld": raw["_tld"].values, "y": y}).groupby("tld")["y"].agg(["sum", "count"])
        k = self.smoothing
        self.tld_risk_ = ((stats["sum"] + k * self.prior_) / (stats["count"] + k)).to_dict()
        self.feature_names_ = [c for c in raw.columns if c != "_tld"] + ["tld_risk"]
        return self

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        X = raw.drop(columns=["_tld"]).copy()
        X["tld_risk"] = raw["_tld"].map(self.tld_risk_).fillna(self.prior_).astype(float)
        return X[self.feature_names_].astype(np.float32)
