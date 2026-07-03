"""Find and repair bad translation lines (explanations, alternatives, mixed script).

Flags lines that contain Latin letters, parentheses/slashes/asterisks, meta words,
runaway repetition, or absurd length — then re-translates each one individually
with a strict prompt and validates the result. Deletes the TTS wav of every
repaired line so the next tts_mlx run regenerates it.

Usage:  VT_WORK=work_full uv run python scripts/fix_translations.py
"""

import json
import os
import re
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / os.environ.get("VT_WORK", "work")
MODEL = os.environ.get("VT_OLLAMA_MODEL", "gemma4:e4b")
MAX_ATTEMPTS = 3

STRICT_PROMPT = """Translate this single movie dialogue line into natural spoken Hebrew for dubbing a kids' movie.

Rules:
- Output ONLY the Hebrew line. No explanations, no alternatives, no parentheses, no notes.
- Hebrew letters only — transliterate English names and acronyms into Hebrew letters.
- Keep it short and punchy, similar length to the original.

Line: {text}"""


def is_bad(he: str, en: str) -> list[str]:
    reasons = []
    if re.search(r"[A-Za-z]", he):
        reasons.append("latin")
    if re.search(r"[()*/]", he):
        reasons.append("markup")
    if len(he) > max(3 * len(en), 60):
        reasons.append("too_long")
    words = he.split()
    if len(words) >= 8 and len(set(words)) <= max(2, len(words) // 4):
        reasons.append("repetition")
    if re.search(r"תרגום|הערה|הסבר|במקור|מילולית|בהתאם להקשר", he):
        reasons.append("meta")
    return reasons


def ollama(prompt: str) -> str:
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0.2}},
        timeout=300,
    )
    r.raise_for_status()
    return r.json()["response"].strip()


def sanitize(he: str) -> str:
    he = re.sub(r"\([^)]*\)", "", he)  # drop parentheticals
    he = he.split("/")[0]  # first alternative only
    he = re.sub(r"[\"'*]", "", he)
    return " ".join(he.split()).strip()


def retranslate(en: str) -> str | None:
    for _ in range(MAX_ATTEMPTS):
        he = sanitize(ollama(STRICT_PROMPT.format(text=en)))
        if he and not is_bad(he, en):
            return he
    return None


def main() -> None:
    segs = {s["id"]: s for s in json.loads((WORK / "segments.json").read_text())}
    trans = json.loads((WORK / "translations.json").read_text())

    fixed, failed = 0, []
    for t in trans:
        en = segs[t["id"]]["text"]
        reasons = is_bad(t["text_he"], en)
        if not reasons:
            continue
        he = retranslate(en)
        if he is None:
            he = sanitize(t["text_he"])  # last resort: strip the junk from the old line
            failed.append(t["id"])
        print(f"id {t['id']} [{','.join(reasons)}]: {t['text_he'][:50]!r} -> {he[:50]!r}")
        t["text_he"] = he
        wav = WORK / f"tts/seg_{t['id']:03d}.wav"
        wav.unlink(missing_ok=True)
        fixed += 1

    (WORK / "translations.json").write_text(json.dumps(trans, indent=2, ensure_ascii=False))
    print(f"\nrepaired {fixed} lines ({len(failed)} sanitized-only: {failed})")
    print("now regenerate: .venv-mlx/bin/python scripts/tts_mlx.py  (only missing segments)")


if __name__ == "__main__":
    main()
