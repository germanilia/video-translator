"""Review pass: correct gender agreement and grammar in the translated script.

A second remote-model pass over translations.json with the full speaker/addressee
gender map. The model returns every line either unchanged or corrected; only
genuinely changed lines are updated, and their stale TTS wavs are deleted so the
next TTS run regenerates exactly those.

Usage:  VT_WORK=work_full uv run python scripts/review_translations.py
Reads:  <work>/segments.json, <work>/speakers.json, <work>/translations.json
Writes: <work>/translations.json (in place), deletes stale tts wavs
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fix_translations import is_bad, sanitize
from translate_remote import (
    HEB,
    PARALLEL,
    addressee_gender,
    detect_reversed,
    parse_numbered,
    pick_provider,
    unreverse,
)

import vt_config  # noqa: F401

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / os.environ.get("VT_WORK", "work")
BATCH = 20

PROMPT = """You are reviewing the Hebrew dub script of a kids' action movie for GENDER AGREEMENT and GRAMMAR only.

Each numbered item shows: [speaker gender, addressing listener gender] | English original | current Hebrew line.

Check every Hebrew line:
- The speaker's self-reference and present-tense verbs must match the SPEAKER's gender.
- Second-person forms (you/your, imperatives, adjectives about the listener) must match the LISTENER's gender.
- Fix grammatical errors (agreement, construct forms, unnatural word order).
- Fix literal word-by-word calques: rephrase into natural spoken Hebrew that carries the same meaning and register.
- Keep the meaning, register, and length. Hebrew letters only. Do NOT re-translate lines that are already correct - return them unchanged.

Output EXACTLY one numbered Hebrew line per item (the corrected line, or the original if already correct):
1. <hebrew>
2. <hebrew>

Items:
{lines}"""


def main() -> None:
    segs = {s["id"]: s for s in json.loads((WORK / "segments.json").read_text())}
    genders = {k: v.get("gender", "unknown") for k, v in json.loads((WORK / "speakers.json").read_text()).items()}
    trans = json.loads((WORK / "translations.json").read_text())
    ordered = sorted(segs.values(), key=lambda s: s["start"])
    index = {s["id"]: i for i, s in enumerate(ordered)}

    name, call = pick_provider()
    print(f"reviewer: {name}, parallel={PARALLEL}")
    rev = detect_reversed(call)

    corrected: dict[int, str] = {}

    def review_batch(batch: list[dict]) -> None:
        lines = []
        for j, t in enumerate(batch):
            s = segs[t["id"]]
            spk = genders.get(s["speaker"], "unknown")
            to = addressee_gender(ordered, index[t["id"]], genders)
            lines.append(f"{j + 1}. [{spk} speaker, addressing {to}] | {s['text']} | {t['text_he']}")
        try:
            parsed = parse_numbered(call(PROMPT.format(lines="\n".join(lines))), len(batch))
        except Exception as e:
            print(f"review batch failed ({e}); keeping originals")
            return
        for j, t in enumerate(batch):
            he = parsed.get(j + 1)
            if he and rev and HEB.search(he):
                he = unreverse(he)
            he = sanitize(he) if he else None
            if he and he != t["text_he"] and not is_bad(he, segs[t["id"]]["text"]):
                corrected[t["id"]] = he

    batches = [trans[i : i + BATCH] for i in range(0, len(trans), BATCH)]
    with ThreadPoolExecutor(max_workers=PARALLEL) as pool:
        list(pool.map(review_batch, batches))

    for t in trans:
        if t["id"] in corrected:
            t["text_he"] = corrected[t["id"]]
            for d in ("tts", "tts_11labs"):
                (WORK / d / f"seg_{t['id']:03d}.wav").unlink(missing_ok=True)
    (WORK / "translations.json").write_text(json.dumps(trans, indent=2, ensure_ascii=False))
    print(f"review corrected {len(corrected)} of {len(trans)} lines (stale wavs deleted)")


if __name__ == "__main__":
    main()
