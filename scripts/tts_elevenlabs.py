"""Generate Hebrew speech with ElevenLabs (eleven_v3) using instant voice clones.

The top-N speakers (by line count) are cloned once from work/refs/<speaker>.wav and
cached in <work>/voices_11labs.json; all remaining minor speakers are mapped
deterministically to ElevenLabs stock voices (no voice slots consumed).

Usage:  VT_WORK=work_full uv run python scripts/tts_elevenlabs.py
Reads:  <work>/translations.json, <work>/refs/<speaker>.wav
Writes: <work>/tts_11labs/seg_<id>.wav
"""

import json
import math
import os
import subprocess
import time
from collections import Counter
from pathlib import Path

import requests
import soundfile as sf
from dicta_onnx import Dicta
from hebrew_gender import addressee_map, feminize_second_person, speaker_overrides

import vt_config  # noqa: F401  (loads config.env)

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / os.environ.get("VT_WORK", "work")
API = "https://api.elevenlabs.io/v1"
KEY = os.environ["ELEVENLABS_API_KEY"]
MODEL = "eleven_v3"
OUT_DIR = WORK / "tts_11labs"
VOICES_CACHE = WORK / "voices_11labs.json"
TOP_N = int(os.environ.get("VT_11LABS_TOP_N", "9"))  # voice slots available for clones
MIN_SAMPLE_SEC = 5.0  # ElevenLabs rejects clone samples under 4.6s; loop short refs

# Premade stock voices for minor speakers (no slots, varied timbres).
STOCK_VOICES = [
    "pNInz6obpgDQGcFmaJgB",  # Adam
    "TxGEqnHWrfWFTfGW9XjX",  # Josh
    "ErXwobaYiN019PkySvjV",  # Antoni
    "yoZ06aMxZJJ28mfd3POQ",  # Sam
    "21m00Tcm4TlvDq8ikWAM",  # Rachel
    "AZnzlk1XvdvUeBnXmlld",  # Domi
]


def request_with_retry(method: str, url: str, **kwargs) -> requests.Response:
    for attempt in range(5):
        resp = requests.request(method, url, headers={"xi-api-key": KEY}, **kwargs)
        if resp.status_code not in (429, 500, 502, 503):
            resp.raise_for_status()
            return resp
        wait = 2**attempt
        print(f"  {resp.status_code}, retrying in {wait}s")
        time.sleep(wait)
    resp.raise_for_status()
    return resp


def prepared_ref(speaker: str) -> Path:
    ref = WORK / f"refs/{speaker}.wav"
    wav, sr = sf.read(str(ref))
    dur = len(wav) / sr
    if dur >= MIN_SAMPLE_SEC:
        return ref
    looped = WORK / f"refs/{speaker}_looped.wav"
    reps = math.ceil(MIN_SAMPLE_SEC / dur)
    sf.write(str(looped), list(wav) * reps, sr)
    return looped


def clone_voice(speaker: str) -> str:
    with open(prepared_ref(speaker), "rb") as f:
        resp = request_with_retry(
            "POST",
            f"{API}/voices/add",
            data={"name": f"vt_movie_{speaker}", "remove_background_noise": "true"},
            files={"files": (f"{speaker}.wav", f, "audio/wav")},
            timeout=180,
        )
    voice_id = resp.json()["voice_id"]
    print(f"cloned {speaker} -> {voice_id}")
    return voice_id


def ensure_voices(segs: list[dict]) -> dict[str, str]:
    counts = Counter(s["speaker"] for s in segs)
    main = [spk for spk, _ in counts.most_common(TOP_N)]
    cache = json.loads(VOICES_CACHE.read_text()) if VOICES_CACHE.exists() else {}
    for spk in main:
        if spk not in cache:
            cache[spk] = clone_voice(spk)
            VOICES_CACHE.write_text(json.dumps(cache, indent=2))
    minor = sorted(set(counts) - set(main))
    for spk in minor:
        if spk not in cache:
            digits = int("".join(c for c in spk if c.isdigit()) or 0)
            cache[spk] = STOCK_VOICES[digits % len(STOCK_VOICES)]
    VOICES_CACHE.write_text(json.dumps(cache, indent=2))
    covered = sum(counts[s] for s in main) / len(segs)
    print(f"{len(main)} cloned mains cover {covered:.0%} of lines; {len(minor)} minors on stock voices")
    return cache


def tts(text: str, voice_id: str, out: Path) -> None:
    resp = request_with_retry(
        "POST",
        f"{API}/text-to-speech/{voice_id}",
        params={"output_format": "mp3_44100_128"},
        json={"text": text, "model_id": MODEL},
        timeout=300,
    )
    mp3 = out.with_suffix(".mp3")
    mp3.write_bytes(resp.content)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3), "-ac", "1", str(out)],
        check=True,
    )
    mp3.unlink()


def main() -> None:
    segs = json.loads((WORK / "translations.json").read_text())
    speaker_overrides(WORK, segs)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    voices = ensure_voices(segs)
    addr = addressee_map(WORK)
    dicta = Dicta(str(ROOT / "work/models/dicta-1.0.onnx"))
    total_chars = 0
    t0 = time.time()
    for i, seg in enumerate(segs):
        out = OUT_DIR / f"seg_{seg['id']:03d}.wav"
        if out.exists():
            continue
        # full niqqud on every line: removes pronunciation guessing entirely;
        # 2nd-person forms are then gender-corrected per the addressee map
        text = dicta.add_diacritics(seg["text_he"])
        if addr.get(seg["id"]) == "female":
            text = feminize_second_person(text)
        tts(text, voices[seg["speaker"]], out)
        total_chars += len(seg["text_he"])
        if (i + 1) % 50 == 0:
            rate = (time.time() - t0) / max(1, i + 1)
            print(f"{i + 1}/{len(segs)} segs, ~{total_chars} chars, {rate:.1f}s/seg", flush=True)
    print(f"done, ~{total_chars} characters used this run")


if __name__ == "__main__":
    main()
