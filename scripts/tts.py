"""Generate Hebrew speech with Chatterbox multilingual, zero-shot cloned per speaker.

Usage:
  uv run python scripts/tts.py --smoke            # quick Hebrew sanity check
  uv run python scripts/tts.py                    # generate all segments from translations.json

Reads:  work/translations.json  [{id, start, end, speaker, text_he}]
        work/refs/<speaker>.wav (reference clips)
Writes: work/tts/seg_<id>.wav
"""

import argparse
import json
from pathlib import Path

import torch
import torchaudio

ROOT = Path(__file__).resolve().parent.parent

# Chatterbox checkpoints reference CUDA devices; remap for Apple Silicon.
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
_torch_load = torch.load


def _patched_load(*args, **kwargs):
    kwargs.setdefault("map_location", torch.device(DEVICE))
    return _torch_load(*args, **kwargs)


torch.load = _patched_load

from chatterbox.mtl_tts import ChatterboxMultilingualTTS  # noqa: E402
from chatterbox.models.tokenizers import tokenizer as cb_tokenizer  # noqa: E402
from dicta_onnx import Dicta  # noqa: E402

DICTA_MODEL = ROOT / "work/models/dicta-1.0.onnx"


def load_model() -> ChatterboxMultilingualTTS:
    # chatterbox instantiates Dicta() without the model path its API requires;
    # pre-seed the module global so Hebrew niqqud is actually applied.
    cb_tokenizer._dicta = Dicta(str(DICTA_MODEL))
    print(f"loading Chatterbox multilingual on {DEVICE}...")
    return ChatterboxMultilingualTTS.from_pretrained(device=DEVICE)


def trim_silence(wav: torch.Tensor, sr: int, rel_thresh: float = 0.03) -> torch.Tensor:
    """Cut leading/trailing silence (takes often carry 1-2s of dead tail)."""
    env = wav.abs().mean(dim=0)
    active = (env > env.max() * rel_thresh).nonzero()
    if len(active) == 0:
        return wav
    pad = int(0.05 * sr)
    a = max(0, active[0].item() - pad)
    b = min(wav.shape[-1], active[-1].item() + pad)
    return wav[:, a:b]


def save(wav: torch.Tensor, sr: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), trim_silence(wav.cpu(), sr), sr)


def smoke(model: ChatterboxMultilingualTTS) -> None:
    text = "שלום! אני מדבר עברית עכשיו. זה מבחן של שכפול קול."
    ref = ROOT / "work/refs/smoke.wav"
    kwargs = {"language_id": "he"}
    if ref.exists():
        kwargs["audio_prompt_path"] = str(ref)
    wav = model.generate(text, **kwargs)
    out = ROOT / "work/tts/smoke_he.wav"
    save(wav, model.sr, out)
    print(f"wrote {out} ({wav.shape[-1] / model.sr:.1f}s)")


MAX_ATTEMPTS = 3


def generate_all(model: ChatterboxMultilingualTTS) -> None:
    segs = json.loads((ROOT / "work/translations.json").read_text())
    for seg in segs:
        out = ROOT / f"work/tts/seg_{seg['id']:03d}.wav"
        if out.exists():
            continue
        ref = ROOT / f"work/refs/{seg['speaker']}.wav"
        slot = seg["end"] - seg["start"]
        # short exclamations sometimes ramble; retry and keep the shortest take
        limit = max(slot * 2.0, slot + 1.5)
        best = None
        for attempt in range(MAX_ATTEMPTS):
            wav = model.generate(
                seg["text_he"],
                language_id="he",
                audio_prompt_path=str(ref),
            )
            if best is None or wav.shape[-1] < best.shape[-1]:
                best = wav
            if best.shape[-1] / model.sr <= limit:
                break
        save(best, model.sr, out)
        dur = best.shape[-1] / model.sr
        print(f"seg {seg['id']:03d} [{seg['speaker']}] gen {dur:.2f}s / slot {slot:.2f}s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    model = load_model()
    if args.smoke:
        smoke(model)
    else:
        generate_all(model)


if __name__ == "__main__":
    main()
