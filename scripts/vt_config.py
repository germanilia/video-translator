"""Load config.env from the repo root into the environment (existing env wins).

All secrets and choices live in one file the user edits once; every script
imports this so standalone stage runs see the same configuration.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config.env"


def load() -> None:
    if not CONFIG.exists():
        return
    for line in CONFIG.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("\"'")
        if key and value and key not in os.environ:
            os.environ[key] = value


load()
