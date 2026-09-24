"""URL normalisation and domain helpers shared by data prep and the URL model."""
from __future__ import annotations

import ipaddress
import re
from functools import lru_cache
from urllib.parse import urlsplit

# Also catches typos seen in the dataset such as "hhttps://".
_SCHEME_RE = re.compile(r"^\s*[a-z][a-z0-9+.\-]{0,15}://", re.IGNORECASE)

# Used only when tldextract is not installed.
_TWO_PART_SUFFIXES = {
    "co.uk", "org.uk", "ac.uk", "gov.uk", "com.au", "net.au", "org.au", "co.nz",
    "co.jp", "ne.jp", "or.jp", "co.in", "net.in", "org.in", "co.za", "com.br",
    "com.cn", "net.cn", "org.cn", "com.tw", "com.hk", "com.sg", "com.my",
    "com.mx", "com.ar", "com.tr", "com.pk", "com.ng", "co.id", "co.kr",
    "ac.lk", "com.lk", "gov.lk", "edu.lk", "org.lk", "co.il", "com.ua", "com.ru",
}

try:  # tldextract gives correct registered domains; use its bundled list, no network
    import tldextract

    _EXTRACT = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)
except Exception:  # pragma: no cover - fallback path
    _EXTRACT = None


def has_scheme(url: str) -> bool:
    return bool(_SCHEME_RE.match(url or ""))


def normalize_url(url: str) -> str:
    """Remove the parts of a URL that reflect how the dataset was collected.

    In CIC-Trap4Phish, 91% of benign URLs have no scheme but only 37% of
    malicious ones do not, so "starts with http://" would be a shortcut.
    We strip the scheme and a leading "www.", lowercase the host, and drop a
    trailing slash. Path and query keep their case.
    """
    u = (url or "").strip()
    u = _SCHEME_RE.sub("", u)
    if u.lower().startswith("www."):
        u = u[4:]
    # lowercase the host part only
    cut = len(u)
    for ch in "/?#":
        i = u.find(ch)
        if i != -1:
            cut = min(cut, i)
    u = u[:cut].lower() + u[cut:]
    return u.rstrip("/")


def get_host(url: str) -> str:
    """Hostname without port or "www.", lowercased. Empty string if none."""
    u = (url or "").strip()
    if not has_scheme(u):
        u = "http://" + u
    try:
        host = urlsplit(u).hostname or ""
    except ValueError:
        # e.g. "Invalid IPv6 URL": take text up to the first / ? # :
        host = re.split(r"[/?#:]", _SCHEME_RE.sub("", u), maxsplit=1)[0]
    host = host.strip(".").lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host.strip("[]"))
        return True
    except ValueError:
        return False


@lru_cache(maxsize=500_000)
def registered_domain(host: str) -> str:
    """"login.paypal.com.evil.co.uk" -> "evil.co.uk". IPs are returned as-is."""
    if not host:
        return ""
    if is_ip(host):
        return host
    if _EXTRACT is not None:
        ext = _EXTRACT(host)
        if ext.domain and ext.suffix:
            return f"{ext.domain}.{ext.suffix}"
        return ext.domain or host
    parts = host.split(".")
    if len(parts) >= 3 and ".".join(parts[-2:]) in _TWO_PART_SUFFIXES:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def mask_tld(url_norm: str) -> str:
    """"shop.example.lk/menu" -> "shop.example.<tld>/menu".

    Used for the character n-gram model so it learns URL *shape*, not country
    endings. TLD risk is handled separately (and smoothed) by tld_risk; the
    dataset has very few Sri Lankan (.lk) URLs and most of them are malicious.
    """
    host = get_host(url_norm)
    if not host or is_ip(host) or "." not in host:
        return url_norm
    tld = host.rsplit(".", 1)[1]
    i = url_norm.lower().find(host)
    if i < 0:
        return url_norm
    end = i + len(host)
    return url_norm[: end - len(tld)] + "<tld>" + url_norm[end:]
