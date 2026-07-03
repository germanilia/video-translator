"""Translate segments to Hebrew with a remote frontier model — provider selectable.

Providers (VT_TRANSLATOR env, or auto-detected in this order):
  pi         - pi CLI print mode (uses whatever provider pi is authenticated with)
  anthropic  - Anthropic API (ANTHROPIC_API_KEY, model VT_ANTHROPIC_MODEL, default claude-sonnet-5)
  openai     - OpenAI API (OPENAI_API_KEY, model VT_OPENAI_MODEL, default gpt-5.4)

Batches are independent (scene context comes from source lines), so they run in
parallel (VT_PI_PARALLEL, default 6). Progress is checkpointed and resumable.
Any line failing validation falls back to the local Gemma repair path.

A startup canary detects providers that return Hebrew in visual (reversed)
order and output is un-reversed automatically.

Usage:  VT_WORK=work_full uv run python scripts/translate_remote.py
Reads:  <work>/segments.json, <work>/speakers.json
Writes: <work>/translations.json
"""

import json
import os
import re
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from hebrew_gender import addressee_gender
from fix_translations import is_bad, retranslate, sanitize

import vt_config  # noqa: F401  (loads config.env)

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / os.environ.get("VT_WORK", "work")
BATCH = 20
SCENE_GAP_SEC = 8.0
PARALLEL = int(os.environ.get("VT_PI_PARALLEL", "6"))
CHECKPOINT = WORK / "translations_remote_progress.json"
LEGACY_CHECKPOINT = WORK / "translations_pi_progress.json"
HEB = re.compile(r"[֐-׿]")

PROMPT = """You are a professional dubbing translator localizing a kids' action movie into Hebrew.
Translate each numbered English dialogue line into natural, punchy SPOKEN Hebrew, as a real Israeli dub would sound.

Rules:
- Translate MEANING and intent, never word-by-word: localize idioms, wordplay,
  and phrasing into what a Hebrew-speaking character would naturally say in that
  moment. A literal calque that sounds like translated English is a WRONG answer
  even if grammatically valid.
- ONE translation per numbered line. No alternatives, no parentheses, no notes, no explanations.
- Hebrew letters only. Transliterate character names and acronyms into Hebrew letters, consistently.
- Write numbers as Hebrew words (text-to-speech reads them aloud).
- Keep each line about the same spoken length as the English or shorter - never longer.
- Each line is tagged [<speaker gender> speaker, addressing <listener gender>]. Hebrew genders
  BOTH: use correct forms for the speaker's self-reference AND for all second-person forms
  (you/your, imperatives, adjectives) matching the LISTENER's gender. Tags never appear in output.
- Lines are consecutive dialogue - keep continuity of tone and references between them.
- Lines marked CONTEXT are for continuity only - do NOT include them in the output.

Output EXACTLY one numbered Hebrew line per non-context input line, same numbers:
1. <hebrew>
2. <hebrew>

Lines:
{lines}"""


def call_pi(prompt: str) -> str:
    res = subprocess.run(
        ["pi", "-p", "--no-session", "--mode", "json", prompt],
        capture_output=True, text=True, timeout=600,
    )
    texts = []
    for line in res.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("type") == "message_end" and e.get("message", {}).get("role") == "assistant":
            for c in e["message"].get("content", []):
                if c.get("type") == "text":
                    texts.append(c["text"])
    if not texts:
        raise RuntimeError(f"pi returned no text (rc={res.returncode}): {res.stderr[:200]}")
    return texts[-1]


def call_anthropic(prompt: str) -> str:
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": os.environ["ANTHROPIC_API_KEY"],
            "anthropic-version": "2023-06-01",
        },
        json={
            "model": os.environ.get("VT_ANTHROPIC_MODEL", "claude-sonnet-5"),
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=300,
    )
    resp.raise_for_status()
    return "".join(c["text"] for c in resp.json()["content"] if c["type"] == "text")


def call_openai(prompt: str) -> str:
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
        json={
            "model": os.environ.get("VT_OPENAI_MODEL", "gpt-5.4"),
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def pick_provider() -> tuple[str, callable]:
    want = os.environ.get("VT_TRANSLATOR")
    table = {"pi": call_pi, "anthropic": call_anthropic, "openai": call_openai}
    if want:
        if want not in table:
            raise SystemExit(f"unknown VT_TRANSLATOR={want}; choose pi|anthropic|openai")
        return want, table[want]
    if shutil.which("pi"):
        return "pi", call_pi
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic", call_anthropic
    if os.environ.get("OPENAI_API_KEY"):
        return "openai", call_openai
    raise SystemExit(
        "no remote translator available: install/authenticate the pi CLI, or set "
        "ANTHROPIC_API_KEY or OPENAI_API_KEY (or use scripts/translate_ollama.py)"
    )


def unreverse(line: str) -> str:
    flipped = line[::-1]
    flipped = re.sub(r"[0-9A-Za-z]+", lambda m: m.group(0)[::-1], flipped)
    swap = {"(": ")", ")": "(", "[": "]", "]": "[", "{": "}", "}": "{"}
    return "".join(swap.get(c, c) for c in flipped)


def detect_reversed(call) -> bool:
    out = call('Translate the single English word "peace" to Hebrew. Reply with ONLY the Hebrew word.')
    word = out.strip().splitlines()[-1].strip()
    rev = word.startswith("ם")
    print(f"canary: {word!r} -> provider {'REVERSED (visual order)' if rev else 'normal'}")
    return rev


def parse_numbered(text: str, n: int) -> dict[int, str]:
    out = {}
    for line in text.splitlines():
        m = re.match(r"\s*(\d+)[.)]\s*(.+)", line.strip())
        if m and 1 <= int(m.group(1)) <= n:
            out[int(m.group(1))] = m.group(2).strip()
    return out





def make_batches(segs: list[dict]) -> list[tuple[list[dict], list[dict]]]:
    """(batch, context_tail_of_previous_batch) pairs; context is source lines only."""
    groups, cur = [], []
    for s in segs:
        if cur and (len(cur) >= BATCH or s["start"] - cur[-1]["end"] > SCENE_GAP_SEC):
            groups.append(cur)
            cur = []
        cur.append(s)
    if cur:
        groups.append(cur)
    return [(b, groups[i - 1][-2:] if i else []) for i, b in enumerate(groups)]


def tag_for(segs: list[dict], seg: dict, genders: dict[str, str], index: dict[int, int]) -> str:
    spk = genders.get(seg["speaker"], "unknown")
    to = addressee_gender(segs, index[seg["id"]], genders)
    return f"[{spk} speaker, addressing {to}]"


def main() -> None:
    segs = json.loads((WORK / "segments.json").read_text())
    genders = {}
    if (WORK / "speakers.json").exists():
        genders = {k: v.get("gender", "unknown") for k, v in json.loads((WORK / "speakers.json").read_text()).items()}

    done: dict[str, str] = {}
    for ckpt in (LEGACY_CHECKPOINT, CHECKPOINT):
        if ckpt.exists():
            done.update(json.loads(ckpt.read_text()))

    name, call = pick_provider()
    print(f"translator: {name}, parallel={PARALLEL}")
    rev = detect_reversed(call)
    lock = threading.Lock()
    stats = {"fallback": 0, "done_batches": 0}
    t0 = time.time()

    index = {s["id"]: i for i, s in enumerate(segs)}

    def run_batch(batch: list[dict], ctx: list[dict]) -> None:
        lines = [f"CONTEXT: {tag_for(segs, c, genders, index)} {c['text']}" for c in ctx]
        lines += [f"{j + 1}. {tag_for(segs, s, genders, index)} {s['text']}" for j, s in enumerate(batch)]
        try:
            parsed = parse_numbered(call(PROMPT.format(lines="\n".join(lines))), len(batch))
        except Exception as e:
            print(f"batch failed ({e}); per-line fallback")
            parsed = {}
        results = {}
        for j, s in enumerate(batch):
            he = parsed.get(j + 1)
            if he and rev and HEB.search(he):
                he = unreverse(he)
            he = sanitize(he) if he else None
            if not he or is_bad(he, s["text"]):
                he = retranslate(s["text"]) or he or s["text"]
                with lock:
                    stats["fallback"] += 1
            results[str(s["id"])] = he
        with lock:
            done.update(results)
            CHECKPOINT.write_text(json.dumps(done, ensure_ascii=False))
            stats["done_batches"] += 1
            print(f"batch done ({stats['done_batches']} this run, {len(done)}/{len(segs)} lines, "
                  f"{time.time() - t0:.0f}s)", flush=True)

    pending = [(b, c) for b, c in make_batches(segs) if any(str(s["id"]) not in done for s in b)]
    with ThreadPoolExecutor(max_workers=PARALLEL) as pool:
        futures = [pool.submit(run_batch, b, c) for b, c in pending]
        for f in as_completed(futures):
            f.result()

    out = [
        {"id": s["id"], "start": s["start"], "end": s["end"], "speaker": s["speaker"],
         "text_he": done[str(s["id"])]}
        for s in segs
    ]
    (WORK / "translations.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"wrote translations.json ({stats['fallback']} lines used gemma fallback)")


if __name__ == "__main__":
    main()
