"""Pronunciation overrides applied after Dicta niqqud.

Dicta is usually right, but movie names/slang/rare words sometimes need a
project-level lexicon. Keep this generic: add reusable Hebrew vowelized pairs,
not movie-specific hacks in TTS scripts.
"""

import json
import os
from pathlib import Path

DEFAULT_OVERRIDES = {
    "עַכְבְּרוּשׁ": "עַכְבָּרוֹשׁ",
    "קְרָאטָה": "קָרָטֵה",
    "קָרָטֶה": "קָרָטֵה",
    "אממ": "אֶממ",
}


def load_overrides(root: Path, work: Path) -> dict[str, str]:
    paths = [root / "pronunciation_overrides.json", work / "pronunciation_overrides.json"]
    extra = os.environ.get("VT_PRONUNCIATION_OVERRIDES")
    if extra:
        paths.append(Path(extra).expanduser())

    overrides = dict(DEFAULT_OVERRIDES)
    for path in paths:
        if path.exists():
            overrides.update(json.loads(path.read_text()))
    return overrides


def apply_pronunciation_overrides(vowelized: str, overrides: dict[str, str]) -> str:
    for source, target in overrides.items():
        vowelized = vowelized.replace(source, target)
    return vowelized
