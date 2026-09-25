"""Show decoded links without making them clickable or openable."""
from __future__ import annotations

import re

_SCHEME = re.compile(r"^([a-z][a-z0-9+.\-]*)://", re.IGNORECASE)


def defang(text: str | None) -> str:
    """hxxps://login[.]example[.]com/x : the standard way analysts share bad links.
    Only the host part gets [.], so the path stays readable."""
    if not text:
        return ""
    t = text.strip()
    scheme = ""
    m = _SCHEME.match(t)
    if m:
        scheme = re.sub(r"^http", "hxxp", m.group(1), flags=re.IGNORECASE) + "://"
        t = t[m.end():]
    host, sep, rest = t.partition("/")
    return scheme + host.replace(".", "[.]") + sep + rest
