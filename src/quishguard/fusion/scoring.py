"""Module D/E: fuse the two streams into one Quishing Confidence Score and a tier.

    score = w_url * URL score + (1 - w_url) * Visual score      (w_url = 0.65 by default)

Rules for the cases the weighted sum cannot handle:
  * The payload is not a URL (Wi-Fi, payment, text...), or the code cannot be
    read at all: there is no URL score, so score = min(Visual score, 75).
    Only one signal can be observed, so it can reach Phishing but not
    Critical (proposal 3.7.6: one signal -> Phishing, both -> Critical).
  * Strong tamper evidence (Visual score >= 50, i.e. the visual model says
    "tampered"): score = max(weighted score, 0.6 * Visual score).
    A sticker over a real payment code can hide a link that looks harmless
    (an aged, redirected domain). With this floor, a visual score of 85 or
    more still reaches the Phishing tier on its own. Below 50 the floor is
    off, so a normal-looking code never has its score raised.

Tiers (proposal section 3.7.4): 0-25 Safe, 26-50 Suspicious, 51-75 Phishing, 76-100 Critical.
"""
from __future__ import annotations

from dataclasses import dataclass

W_URL = 0.65
TAMPER_FLOOR = 0.6
TAMPER_MIN = 50.0      # visual score from which the floor applies
VISUAL_ONLY_CAP = 75.0 # one signal alone stays at most Phishing

TIERS = [(25, "Safe"), (50, "Suspicious"), (75, "Phishing"), (100, "Critical")]
ACTIONS = {
    "Safe": "No action needed.",
    "Suspicious": "Warn the user and queue for manual review.",
    "Phishing": "Block the link and alert a Tier-2 analyst.",
    "Critical": "Block immediately, escalate, and check where the code is physically placed.",
}


def tier(score: float) -> str:
    for hi, name in TIERS:
        if score <= hi:
            return name
    return "Critical"


@dataclass
class Fused:
    score: float
    tier: str
    rule: str          # which rule produced the score (shown in the alert)


def fuse(url_score: float | None, visual_score: float, w_url: float = W_URL,
         tamper_floor: float = TAMPER_FLOOR, tamper_min: float = TAMPER_MIN,
         visual_only_cap: float = VISUAL_ONLY_CAP) -> Fused:
    if url_score is None:
        s, rule = min(visual_score, visual_only_cap), "visual only (no URL in the code, or the code could not be read)"
    else:
        s = w_url * url_score + (1 - w_url) * visual_score
        rule = f"weighted: {w_url:.2f} x URL + {1 - w_url:.2f} x visual"
        if tamper_floor and visual_score >= tamper_min and tamper_floor * visual_score > s:
            s, rule = tamper_floor * visual_score, "strong tamper evidence raised the score"
    s = float(max(0.0, min(100.0, s)))
    return Fused(round(s, 2), tier(s), rule)


def ground_truth_tier(url_malicious: bool | None, tampered: bool) -> str:
    """Reference severity for evaluation (proposal section 3.7.6):
    both signals -> Critical, one -> Phishing, none -> Safe."""
    n = int(bool(url_malicious)) + int(bool(tampered))
    return {0: "Safe", 1: "Phishing", 2: "Critical"}[n]
