"""Transcribe with mlx-whisper (Apple GPU, ~15x realtime) — much faster than CPU whisper.

Run with the MLX venv:
  VT_WORK=work_full .venv-mlx/bin/python scripts/asr_mlx.py

Reads:  <work>/clip/audio_16k.wav
Writes: <work>/asr.json  [{start, end, text}] — consumed by transcribe.py
"""

import json
import os
import time
from pathlib import Path

import mlx_whisper

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / os.environ.get("VT_WORK", "work")
MODEL = "mlx-community/whisper-large-v3-turbo"


def main() -> None:
    t0 = time.time()
    result = mlx_whisper.transcribe(
        str(WORK / "clip/audio_16k.wav"),
        path_or_hf_repo=MODEL,
        language="en",
        condition_on_previous_text=False,
        word_timestamps=True,
        verbose=None,
    )
    out = []
    for seg in result["segments"]:
        text = seg["text"].strip()
        if not text:
            continue
        words = [
            {"start": round(w["start"], 3), "end": round(w["end"], 3), "word": w["word"]}
            for w in seg.get("words", [])
        ]
        out.append(
            {"start": round(seg["start"], 3), "end": round(seg["end"], 3), "text": text, "words": words}
        )
    (WORK / "asr.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"wrote {WORK / 'asr.json'}: {len(out)} segments in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
