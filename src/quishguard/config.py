"""Project paths and fixed settings.

Paths can be overridden with environment variables so the same code runs on
the laptop (Windows) and in Colab:

    QG_RAW_DIR        folder that holds QR_All_benign / QR_All_Malicious
    QG_PROCESSED_DIR  where cleaned tables and splits are written
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = Path(os.environ.get("QG_RAW_DIR", REPO_ROOT / "data" / "raw"))
PROCESSED_DIR = Path(os.environ.get("QG_PROCESSED_DIR", REPO_ROOT / "data" / "processed"))
MODELS_DIR = Path(os.environ.get("QG_MODELS_DIR", REPO_ROOT / "models"))
REPORTS_DIR = REPO_ROOT / "reports"

# CIC-Trap4Phish class folders. The label comes ONLY from the folder:
# every image file is named benign_XXXXXX.png, even the malicious ones.
CLASS_DIRS = {"QR_All_benign": 0, "QR_All_Malicious": 1}

# Split by registered domain: md5(salt + domain) -> bucket 0..99
SPLIT_SALT = "quishguard-c3"
SPLIT_BUCKETS = {"train": (0, 70), "val": (70, 80), "test": (80, 100)}

SEED = 42
