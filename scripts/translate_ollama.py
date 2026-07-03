"""Translate segments to Hebrew with a local Ollama LLM (better dubbing style than NLLB).

Batches numbered lines per request; falls back to existing translations.json text
(NLLB) for any line the model fails to return. Checkpoints progress so it resumes.

Usage:  VT_WORK=work_full uv run python scripts/translate_ollama.py
Reads:  <work>/segments.json (+ existing translations.json as fallback)
Writes: <work>/translations.json
"""

import json
import os
import re
from pathlib import Path

import requests

from hebrew_gender import addressee_gender
from fix_translations import is_bad, retranslate, sanitize

import vt_config  # noqa: F401  (loads config.env)

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / os.environ.get("VT_WORK", "work")
MODEL = os.environ.get("VT_OLLAMA_MODEL", "gemma4:e4b")
BATCH = 10
CHECKPOINT = WORK / "translations_ollama_progress.json"
SCENE_GAP_SEC = 8.0

PROMPT = """You are dubbing a kids' action movie into Hebrew. Translate each numbered English dialogue line into natural, spoken Hebrew — short, punchy, similar length to the original, appropriate for children.

Strict rules:
- ONE translation per line. Never give alternatives, parentheses, notes, or explanations.
- Hebrew letters only — transliterate English names and acronyms into Hebrew letters.
- Each line is tagged [<speaker gender> speaker, addressing <listener gender>]. Hebrew genders
  BOTH: the speaker's self-reference AND all second-person forms (you/your, imperatives,
  adjectives) must match the LISTENER's gender. Tags never appear in the output.

Reply with EXACTLY one numbered Hebrew line per input line, same numbers, format:
1. <hebrew>
2. <hebrew>

Lines:
{lines}"""


def ollama(prompt: str) -> str:
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0.3}},
        timeout=600,
    )
    r.raise_for_status()
    return r.json()["response"]


def parse_numbered(text: str, n: int) -> dict[int, str]:
    out = {}
    for line in text.splitlines():
        m = re.match(r"\s*(\d+)[.)]\s*(.+)", line.strip())
        if m and 1 <= int(m.group(1)) <= n:
            out[int(m.group(1))] = m.group(2).strip()
    return out





def load_genders() -> dict[str, str]:
    path = WORK / "speakers.json"
    if not path.exists():
        return {}
    return {spk: v.get("gender", "unknown") for spk, v in json.loads(path.read_text()).items()}


def load_reusable() -> dict[tuple[str, int], str]:
    """Old translations keyed by (english_text, rounded_start) — reused for
    male/unknown speakers so their TTS audio stays valid across re-diarization."""
    old_segs = WORK / "segments_old.json"
    old_trans = WORK / "translations_old.json"
    if not (old_segs.exists() and old_trans.exists()):
        return {}
    texts = {s["id"]: s["text"] for s in json.loads(old_segs.read_text())}
    out = {}
    for t in json.loads(old_trans.read_text()):
        en = texts.get(t["id"])
        if en:
            out[(en.strip(), round(t["start"]))] = t["text_he"]
    return out


def main() -> None:
    segs = json.loads((WORK / "segments.json").read_text())
    genders = load_genders()
    reusable = load_reusable()
    fallback = {}
    trans_path = WORK / "translations.json"
    if trans_path.exists():
        fallback = {t["id"]: t["text_he"] for t in json.loads(trans_path.read_text())}

    done: dict[str, str] = json.loads(CHECKPOINT.read_text()) if CHECKPOINT.exists() else {}
    fallback_count = reused = 0

    for seg in segs:
        if str(seg["id"]) in done:
            continue
        if genders.get(seg["speaker"], "unknown") != "female":
            he = reusable.get((seg["text"].strip(), round(seg["start"])))
            if he:
                done[str(seg["id"])] = he
                reused += 1
    if reused:
        CHECKPOINT.write_text(json.dumps(done, ensure_ascii=False))
        print(f"reused {reused} existing translations")

    for i in range(0, len(segs), BATCH):
        batch = [s for s in segs[i : i + BATCH] if str(s["id"]) not in done]
        if not batch:
            continue
        index = {s["id"]: k for k, s in enumerate(segs)}
        lines = "\n".join(
            f"{j + 1}. [{genders.get(s['speaker'], 'unknown')} speaker, addressing "
            f"{addressee_gender(segs, index[s['id']], genders)}] {s['text']}"
            for j, s in enumerate(batch)
        )
        parsed = parse_numbered(ollama(PROMPT.format(lines=lines)), len(batch))
        for j, s in enumerate(batch):
            he = parsed.get(j + 1)
            if he and is_bad(sanitize(he), s["text"]):
                he = retranslate(s["text"])  # strict single-line repair
            elif he:
                he = sanitize(he)
            if not he:
                he = fallback.get(s["id"], s["text"])
                fallback_count += 1
            done[str(s["id"])] = he
        CHECKPOINT.write_text(json.dumps(done, ensure_ascii=False))
        print(f"translated {min(i + BATCH, len(segs))}/{len(segs)}", flush=True)

    out = [
        {"id": s["id"], "start": s["start"], "end": s["end"], "speaker": s["speaker"],
         "text_he": done[str(s["id"])]}
        for s in segs
    ]
    trans_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"wrote {trans_path} ({fallback_count} lines used fallback)")


if __name__ == "__main__":
    main()
