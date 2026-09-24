"""Phase 7: the real malicious QR test (50% milestone).

Proves the system works on NEW, real threats, not only the 2025 dataset:
  * malicious: live phishing links from the OpenPhish community feed and
    malware links from URLhaus (abuse.ch), downloaded today
  * benign: popular legitimate sites from Tranco ranks 150,001+ (ranks 1-150,000
    were used in training) plus a hand-made Sri Lankan list
  * any link whose host name appears anywhere in the training data is removed,
    so every test link is unseen (time hold-out + host hold-out). Hosts, not
    registered domains: phishing often sits on shared platforms (vercel.app,
    weebly.com...), and dropping whole platforms would bias the test.

Each link becomes a QR image in four situations:
  clean            computer-made code (like a screenshot or a PDF)
  photo            the same code with phone-photo effects
  sticker_attack   a malicious code stuck over a legitimate code (benign base)
  your_photos      optional: real phone photos you take of printed codes

SAFETY: only the feed LISTS are downloaded. The phishing links themselves are
never opened, fetched, or resolved. They are only turned into QR images and scored.
"""
from __future__ import annotations

import csv
import io
import random
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from quishguard.data.urls import get_host, normalize_url, registered_domain

OPENPHISH = "https://openphish.com/feed.txt"
URLHAUS = "https://urlhaus.abuse.ch/downloads/csv_recent/"
TRANCO = "https://tranco-list.eu/top-1m.csv.zip"

SRI_LANKA_BENIGN = [
    "https://www.sliit.lk/", "https://www.nsbm.ac.lk/", "https://www.cmb.ac.lk/", "https://www.mrt.ac.lk/",
    "https://www.pdn.ac.lk/", "https://www.gov.lk/", "https://www.cbsl.gov.lk/", "https://www.dialog.lk/",
    "https://www.mobitel.lk/", "https://www.slt.lk/", "https://www.combank.lk/", "https://www.hnb.lk/",
    "https://www.sampath.lk/", "https://www.boc.lk/", "https://www.peoplesbank.lk/", "https://www.keells.lk/",
    "https://www.daraz.lk/", "https://www.dailymirror.lk/", "https://www.newsfirst.lk/", "https://www.srilankan.com/",
    "https://www.ikman.lk/", "https://www.pickme.lk/", "https://www.arpico.com/", "https://www.cargillsonline.com/",
    "https://www.ceb.lk/", "https://www.immigration.gov.lk/", "https://www.health.gov.lk/", "https://www.moe.gov.lk/",
    "https://www.pizzahut.lk/", "https://www.hutch.lk/",
]


def _get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "quishguard-research/0.1 (SLIIT J26-IT-424)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def fetch_openphish() -> pd.DataFrame:
    lines = _get(OPENPHISH).decode("utf-8", "replace").splitlines()
    urls = [l.strip() for l in lines if l.strip().startswith("http")]
    return pd.DataFrame({"url": urls, "source": "openphish", "threat": "phishing"})


def fetch_urlhaus() -> pd.DataFrame:
    text = _get(URLHAUS).decode("utf-8", "replace")
    rows = [r for r in csv.reader(line for line in text.splitlines() if line and not line.startswith("#"))]
    # id, dateadded, url, url_status, last_online, threat, tags, urlhaus_link, reporter
    df = pd.DataFrame([{"url": r[2], "added": r[1], "threat": r[5] or "malware"} for r in rows if len(r) > 5])
    df["source"] = "urlhaus"
    return df


def fetch_tranco_unseen(n: int, start_rank: int = 150_001, cache: Path | None = None) -> pd.DataFrame:
    if cache and Path(cache).exists():
        raw = Path(cache).read_bytes()
    else:
        with zipfile.ZipFile(io.BytesIO(_get(TRANCO, 120))) as z:
            raw = z.read(z.namelist()[0])
        if cache:
            Path(cache).write_bytes(raw)
    t = pd.read_csv(io.BytesIO(raw), header=None, names=["rank", "domain"], nrows=start_rank + 200_000)
    t = t[t["rank"] >= start_rank].sample(n=min(n * 3, len(t) - start_rank), random_state=7)
    return pd.DataFrame({"url": "https://" + t["domain"].astype(str) + "/", "source": "tranco_unseen",
                         "threat": "none", "rank": t["rank"].values})


def build_link_set(seen_hosts: set[str], n_malicious: int = 500, n_benign: int = 500,
                   tranco_cache: Path | None = None, seed: int = 7) -> pd.DataFrame:
    parts, notes = [], {}
    for name, fn in [("openphish", fetch_openphish), ("urlhaus", fetch_urlhaus)]:
        try:
            d = fn(); parts.append(d); notes[name] = len(d)
        except Exception as e:  # a feed can be down; the other still works
            notes[name] = f"failed: {e}"
    mal = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["url", "source", "threat"])
    mal["label"] = 1
    ben = pd.concat([fetch_tranco_unseen(n_benign, cache=tranco_cache),
                     pd.DataFrame({"url": SRI_LANKA_BENIGN, "source": "sri_lanka_list", "threat": "none"})],
                    ignore_index=True)
    ben["label"] = 0
    df = pd.concat([mal, ben], ignore_index=True)
    df["url"] = df["url"].astype(str).str.strip()
    df = df[df["url"].str.len().between(8, 900)]
    df["url_norm"] = df["url"].map(normalize_url)
    df["host"] = df["url"].map(get_host)
    df["domain"] = df["host"].map(registered_domain)
    df = df.drop_duplicates("url_norm")
    before = df["label"].value_counts().to_dict()
    df = df[~df["host"].isin(seen_hosts)]                 # unseen hosts only
    sl = df[df["source"] == "sri_lanka_list"]
    rng_mal = df[df["label"] == 1].groupby("source", group_keys=False)
    mal_s = pd.concat([g.sample(n=min(len(g), n_malicious // 2), random_state=seed) for _, g in rng_mal]) \
        if (df["label"] == 1).any() else df.iloc[0:0]
    if len(mal_s) < n_malicious:   # top up from whichever feed has more
        rest = df[(df["label"] == 1) & ~df.index.isin(mal_s.index)]
        mal_s = pd.concat([mal_s, rest.sample(n=min(len(rest), n_malicious - len(mal_s)), random_state=seed)])
    ben_s = df[(df["label"] == 0) & (df["source"] == "tranco_unseen")]
    ben_s = pd.concat([sl, ben_s.sample(n=min(len(ben_s), n_benign - len(sl)), random_state=seed)])
    out = pd.concat([mal_s, ben_s], ignore_index=True)
    out["fetched_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    out.attrs["notes"] = {"feeds": notes, "before_domain_filter": before,
                          "after": out["label"].value_counts().to_dict()}
    return out


# ------------------------------------------------------------------ images
def make_qr(text: str, scale: int, border: int = 4) -> np.ndarray:
    try:
        import segno
        buf = io.BytesIO()
        segno.make(text, error="m").save(buf, kind="png", scale=scale, border=border)
        return cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), cv2.IMREAD_GRAYSCALE)
    except ImportError:
        img = cv2.QRCodeEncoder.create().encode(text)
        return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)


def make_images(links: pd.DataFrame, out_dir: Path, n_sticker: int = 200, seed: int = 7) -> pd.DataFrame:
    from quishguard.tamper.generate import attack_sticker, photo_effects, recrop

    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, r in links.reset_index(drop=True).iterrows():
        qr = make_qr(r["url"], scale=rng.randint(4, 9))
        for kind, img in (("clean", qr), ("photo", photo_effects(recrop(qr, rng), rng))):
            f = out_dir / f"{kind}_{i:04d}.png"
            cv2.imwrite(str(f), img)
            rows.append({"file": f.name, "situation": kind, "url": r["url"], "label": int(r["label"]),
                         "source": r["source"], "tampered": 0})
    ben = links[links["label"] == 0].sample(frac=1, random_state=seed).reset_index(drop=True)
    mal = links[links["label"] == 1].sample(frac=1, random_state=seed).reset_index(drop=True)
    for j in range(min(n_sticker, len(ben), len(mal))):
        base = make_qr(ben.loc[j, "url"], scale=6, border=0)
        stk = make_qr(mal.loc[j, "url"], scale=6, border=0)
        t, _ = attack_sticker(base, stk, rng)
        f = out_dir / f"sticker_attack_{j:04d}.png"
        cv2.imwrite(str(f), photo_effects(recrop(t, rng), rng))
        rows.append({"file": f.name, "situation": "sticker_attack", "url": mal.loc[j, "url"], "label": 1,
                     "source": mal.loc[j, "source"], "tampered": 1, "covered_benign_url": ben.loc[j, "url"]})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ results
def summarize(res: pd.DataFrame) -> pd.DataFrame:
    """res: one row per scanned image with columns label, situation, tier, score, url_score, visual_score, decoded_ok."""
    alert = res["tier"].isin(["Phishing", "Critical"])
    warn = res["tier"].isin(["Suspicious", "Phishing", "Critical"])
    rows = []
    for sit, g in res.groupby("situation"):
        a, w = alert[g.index], warn[g.index]
        mal, ben = g["label"] == 1, g["label"] == 0
        url_alert = g["url_score"].fillna(0) >= 50
        rows.append({
            "situation": sit, "n_malicious": int(mal.sum()), "n_benign": int(ben.sum()),
            "decoded_%": 100 * g["decoded_ok"].mean(),
            "detected_%_(Phishing+Critical)": 100 * a[mal].mean() if mal.any() else np.nan,
            "flagged_%_(Suspicious+)": 100 * w[mal].mean() if mal.any() else np.nan,
            "false_alarm_%_(benign Phishing+)": 100 * a[ben].mean() if ben.any() else np.nan,
            "url_stream_only_detected_%": 100 * url_alert[mal].mean() if mal.any() else np.nan,
            "mean_score_malicious": g.loc[mal, "score"].mean() if mal.any() else np.nan,
            "mean_score_benign": g.loc[ben, "score"].mean() if ben.any() else np.nan,
        })
    return pd.DataFrame(rows).set_index("situation").round(2)
