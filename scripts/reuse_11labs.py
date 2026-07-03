"""Carry over ElevenLabs audio from a previous run after re-diarization.

A new segmentation renumbers ids and relabels speakers, but most lines are
unchanged: same English text, same timing, same character, same Hebrew. Their
existing wavs are copied over so only genuinely changed lines cost API credits.

Match rule per new segment:
  old segment with identical English text, |start delta| <= 0.5s,
  identical Hebrew translation, and the old speaker's centroid matches the
  new speaker's centroid (cosine sim >= 0.60 — same character).

Reads:  <work>/{segments,translations}.json, <work>/{segments,translations}_old.json,
        <work>/tts_11labs_old/, demucs vocals
Writes: <work>/tts_11labs/seg_<id>.wav (copies)
"""

import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / os.environ.get("VT_WORK", "work")
VOCALS = WORK / "demucs/htdemucs/audio_44k/vocals.wav"
SR = 16000
MIN_CLUSTER_SIM = 0.60


def centroids(segs: list[dict], wav: torch.Tensor, enc) -> dict[str, np.ndarray]:
    out = {}
    for spk in sorted({s["speaker"] for s in segs}):
        own = sorted((s for s in segs if s["speaker"] == spk), key=lambda s: s["end"] - s["start"], reverse=True)[:12]
        embs, weights = [], []
        for s in own:
            chunk = wav[:, int(s["start"] * SR) : int(s["end"] * SR)]
            if chunk.shape[1] < SR // 2:
                chunk = torch.nn.functional.pad(chunk, (0, SR // 2 - chunk.shape[1]))
            with torch.no_grad():
                e = enc.encode_batch(chunk).squeeze().numpy()
            embs.append(e / np.linalg.norm(e))
            weights.append(s["end"] - s["start"])
        c = np.average(embs, axis=0, weights=weights)
        out[spk] = c / np.linalg.norm(c)
    return out


def main() -> None:
    new_segs = json.loads((WORK / "segments.json").read_text())
    old_segs = json.loads((WORK / "segments_old.json").read_text())
    new_he = {t["id"]: t["text_he"] for t in json.loads((WORK / "translations.json").read_text())}
    old_he = {t["id"]: t["text_he"] for t in json.loads((WORK / "translations_old.json").read_text())}
    old_dir = WORK / "tts_11labs_old"
    new_dir = WORK / "tts_11labs"
    new_dir.mkdir(parents=True, exist_ok=True)

    wav, sr = torchaudio.load(str(VOCALS))
    wav = torchaudio.functional.resample(wav.mean(dim=0, keepdim=True), sr, SR)
    enc = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(ROOT / "work/models/ecapa")
    )
    new_cents = centroids(new_segs, wav, enc)
    old_cents = centroids(old_segs, wav, enc)

    old_index: dict[str, list[dict]] = {}
    for s in old_segs:
        old_index.setdefault(s["text"].strip(), []).append(s)

    copied = missing = 0
    for seg in new_segs:
        dst = new_dir / f"seg_{seg['id']:03d}.wav"
        if dst.exists():
            continue
        match = None
        for old in old_index.get(seg["text"].strip(), []):
            if abs(old["start"] - seg["start"]) > 0.5:
                continue
            if old_he.get(old["id"]) != new_he.get(seg["id"]):
                continue
            sim = float(np.dot(new_cents[seg["speaker"]], old_cents[old["speaker"]]))
            if sim >= MIN_CLUSTER_SIM:
                match = old
                break
        if match:
            src = old_dir / f"seg_{match['id']:03d}.wav"
            if src.exists():
                shutil.copy2(src, dst)
                copied += 1
                continue
        missing += 1
    print(f"reused {copied} wavs, {missing} segments need ElevenLabs generation")


if __name__ == "__main__":
    main()
