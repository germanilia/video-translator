"""Generate Hebrew speech with the MLX port of Chatterbox (Apple-native, ~9x faster).

Run with the dedicated MLX venv:
  .venv-mlx/bin/python scripts/tts_mlx.py

Reads:  work/translations.json, work/refs/<speaker>.wav
Writes: work/tts/seg_<id>.wav (silence-trimmed)
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from dicta_onnx import Dicta
from hebrew_gender import addressee_map, feminize_second_person, speaker_overrides
from hebrew_pronunciation import apply_pronunciation_overrides, load_overrides
from mlx_audio.tts.generate import generate_audio
from mlx_audio.tts.models.chatterbox import tokenizer as chatterbox_tokenizer
from mlx_audio.tts.utils import load_model

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / os.environ.get("VT_WORK", "work")
MODEL = "mlx-community/chatterbox-fp16"
TOKENS_PER_SEC = 25  # chatterbox S3 speech-token rate
TRIM_REL_THRESH = 0.08  # MLX port pads the tail with low-level noise; trim hard


def trim_silence(wav: np.ndarray, sr: int) -> np.ndarray:
    env = np.abs(wav)
    active = np.nonzero(env > env.max() * TRIM_REL_THRESH)[0]
    if len(active) == 0:
        return wav
    pad = int(0.05 * sr)
    a = max(0, active[0] - pad)
    b = min(len(wav), active[-1] + pad)
    return wav[a:b]


def main() -> None:
    dicta = Dicta(str(ROOT / "work/models/dicta-1.0.onnx"))
    chatterbox_tokenizer._dicta = dicta
    model = load_model(MODEL)
    segs = json.loads((WORK / "translations.json").read_text())
    speaker_overrides(WORK, segs)
    addr = addressee_map(WORK)
    pronunciation_overrides = load_overrides(ROOT, WORK)
    out_dir = WORK / "tts"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for seg in segs:
        out = out_dir / f"seg_{seg['id']:03d}.wav"
        if out.exists():
            continue
        slot = seg["end"] - seg["start"]
        max_tokens = int(TOKENS_PER_SEC * min(max(slot * 1.6 + 2.0, 2.5), 12.0))
        text = dicta.add_diacritics(seg["text_he"])
        if addr.get(seg["id"]) == "female":
            # dicta defaults ambiguous 2nd-person forms to masculine vowels
            text = feminize_second_person(text)
        text = apply_pronunciation_overrides(text, pronunciation_overrides)
        generate_audio(
            text=text,
            model=model,
            max_tokens=max_tokens,
            ref_audio=str(WORK / f"refs/{seg['speaker']}.wav"),
            lang_code="he",
            output_path=str(out_dir),
            file_prefix=f"raw_{seg['id']:03d}",
            verbose=False,
        )
        raw = out_dir / f"raw_{seg['id']:03d}_000.wav"
        wav, sr = sf.read(str(raw))
        raw.unlink()
        wav = trim_silence(wav, sr)
        sf.write(str(out), wav, sr)
        print(f"seg {seg['id']:03d} [{seg['speaker']}] gen {len(wav) / sr:.2f}s / slot {slot:.2f}s")
    print(f"total {time.time() - t0:.0f}s for {len(segs)} segments")


if __name__ == "__main__":
    main()
