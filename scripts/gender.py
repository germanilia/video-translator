"""Classify each speaker's gender with a pretrained voice model (F0 medians fail
on excited movie speech — shouting men read as >240Hz).

Pools up to 20s of each speaker's cleanest segments, runs a wav2vec2 gender
classifier, and updates <work>/speakers.json in place (keeps f0 for reference).

Usage:  VT_WORK=work_full uv run python scripts/gender.py
"""

import json
import os
from pathlib import Path

import torch
import torchaudio
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / os.environ.get("VT_WORK", "work")
VOCALS = WORK / "demucs/htdemucs/audio_44k/vocals.wav"
MODEL = "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech"
SR = 16000
POOL_SEC = 20.0
CONF_FEMALE = float(__import__("os").environ.get("VT_FEMALE_CONF", "0.95"))  # strict: young male voices score falsely high


def main() -> None:
    segs = json.loads((WORK / "segments.json").read_text())
    speakers = json.loads((WORK / "speakers.json").read_text())
    wav, sr = torchaudio.load(str(VOCALS))
    wav = torchaudio.functional.resample(wav.mean(dim=0, keepdim=True), sr, SR)

    extractor = AutoFeatureExtractor.from_pretrained(MODEL)
    model = AutoModelForAudioClassification.from_pretrained(MODEL)
    model.eval()
    labels = {i: l.lower() for i, l in model.config.id2label.items()}

    for spk in sorted(speakers):
        own = sorted(
            (s for s in segs if s["speaker"] == spk),
            key=lambda s: s["end"] - s["start"],
            reverse=True,
        )
        chunks, total = [], 0.0
        for s in own:
            chunks.append(wav[0, int(s["start"] * SR) : int(s["end"] * SR)])
            total += s["end"] - s["start"]
            if total >= POOL_SEC:
                break
        if not chunks:
            continue
        audio = torch.cat(chunks)[: int(POOL_SEC * SR)].numpy()
        inputs = extractor(audio, sampling_rate=SR, return_tensors="pt")
        with torch.no_grad():
            probs = model(**inputs).logits.softmax(-1).squeeze()
        p = {labels[i]: float(probs[i]) for i in range(len(probs))}
        female_p = p.get("female", 0.0)
        gender = "female" if female_p >= CONF_FEMALE else "male"
        speakers[spk]["gender"] = gender
        speakers[spk]["female_prob"] = round(female_p, 3)
        print(f"{spk}: female_p={female_p:.2f} -> {gender}")

    overrides_path = WORK / "gender_overrides.json"
    if overrides_path.exists():
        for spk, g in json.loads(overrides_path.read_text()).items():
            if spk in speakers and speakers[spk]["gender"] != g:
                speakers[spk]["gender"] = g
                print(f"override: {spk} -> {g}")
    (WORK / "speakers.json").write_text(json.dumps(speakers, indent=2))
    print("updated", WORK / "speakers.json")


if __name__ == "__main__":
    main()
