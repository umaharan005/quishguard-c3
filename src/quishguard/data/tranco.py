"""Extra benign data: popular legitimate domains from the Tranco top-sites list.

Why: in CIC-Trap4Phish 81% of bare-domain URLs (e.g. "sliit.lk") are malicious,
because the benign URLs come from deep pages of an old web crawl. Real QR
codes often point to a homepage, so without this the model flags normal
homepages. Tranco (https://tranco-list.eu) ranks sites by popularity over 30
days; the top of the list is overwhelmingly legitimate.
"""
from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from quishguard.data.prepare import split_of
from quishguard.data.urls import get_host, normalize_url, registered_domain

TRANCO_URL = "https://tranco-list.eu/top-1m.csv.zip"


def load_tranco(top_n: int, path: Path | None = None) -> pd.DataFrame:
    """Read rank,domain from a local csv/zip, or download the latest list."""
    if path is not None and Path(path).exists():
        src = Path(path)
        if src.suffix == ".zip":
            with zipfile.ZipFile(src) as z:
                raw = z.read(z.namelist()[0])
        else:
            raw = src.read_bytes()
    else:
        with urllib.request.urlopen(TRANCO_URL, timeout=60) as r:
            with zipfile.ZipFile(io.BytesIO(r.read())) as z:
                raw = z.read(z.namelist()[0])
        if path is not None:  # cache for next time
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_bytes(raw)
    df = pd.read_csv(io.BytesIO(raw), header=None, names=["rank", "domain"], nrows=top_n)
    return df


def tranco_rows(top_n: int, exclude_domains: set[str], path: Path | None = None) -> pd.DataFrame:
    """Benign rows in the same schema as urls_clean.csv.gz."""
    t = load_tranco(top_n, path)
    url = t["domain"].astype(str).str.strip().str.lower()
    out = pd.DataFrame({"url": url})
    out["url_norm"] = out["url"].map(normalize_url)
    out["host"] = out["url"].map(get_host)
    out["domain"] = out["host"].map(registered_domain)
    out = out[(out["host"] != "") & ~out["domain"].isin(exclude_domains)]
    out = out.drop_duplicates("url_norm")
    out["label"] = 0
    out["class_dir"] = "tranco"
    out["img_file"] = ""
    out["source_row"] = ""
    out["split"] = out["domain"].map(split_of)
    return out.reset_index(drop=True)
