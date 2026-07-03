"""Transcribe the clip with faster-whisper and diarize speakers via ECAPA embeddings.

Reads:  work/clip/audio_16k.wav (ASR), work/demucs/htdemucs/audio_44k/vocals.wav (embeddings)
Writes: work/segments.json  [{id, start, end, speaker, text}]
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
import torchaudio
from faster_whisper import WhisperModel
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from speechbrain.inference.speaker import EncoderClassifier

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / os.environ.get("VT_WORK", "work")
ASR_WAV = WORK / "clip/audio_16k.wav"
VOCALS_WAV = WORK / "demucs/htdemucs/audio_44k/vocals.wav"
OUT = WORK / "segments.json"

MIN_SEG_DUR = 0.5  # seconds; shorter segments get merged into neighbors for embedding
CLUSTER_DIST = 0.85  # cosine-distance threshold for same-speaker grouping


def transcribe() -> list[dict]:
    model = WhisperModel("large-v3", device="cpu", compute_type="int8")
    segments, info = model.transcribe(
        str(ASR_WAV),
        language="en",
        vad_filter=True,
        word_timestamps=True,
    )
    out = []
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        out.append({"start": round(seg.start, 3), "end": round(seg.end, 3), "text": text})
    print(f"transcribed {len(out)} segments, detected language={info.language}")
    return out


def embed_segments(segs: list[dict]) -> np.ndarray:
    wav, sr = torchaudio.load(str(VOCALS_WAV))
    wav = wav.mean(dim=0, keepdim=True)  # mono
    wav = torchaudio.functional.resample(wav, sr, 16000)
    sr = 16000

    encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(ROOT / "work/models/ecapa"),
    )
    embs = []
    for seg in segs:
        s = max(0, int(seg["start"] * sr))
        e = min(wav.shape[1], int(seg["end"] * sr))
        chunk = wav[:, s:e]
        if chunk.shape[1] < int(MIN_SEG_DUR * sr):
            pad = int(MIN_SEG_DUR * sr) - chunk.shape[1]
            chunk = torch.nn.functional.pad(chunk, (0, pad))
        with torch.no_grad():
            emb = encoder.encode_batch(chunk).squeeze().numpy()
        embs.append(emb / np.linalg.norm(emb))
    return np.stack(embs)


def cluster_speakers(embs: np.ndarray) -> list[int]:
    if len(embs) == 1:
        return [0]
    dists = pdist(embs, metric="cosine")
    z = linkage(dists, method="average")
    labels = fcluster(z, t=CLUSTER_DIST, criterion="distance")
    # renumber by first appearance
    order: dict[int, int] = {}
    out = []
    for lb in labels:
        if lb not in order:
            order[lb] = len(order)
        out.append(order[lb])
    return out


def main() -> None:
    asr_json = WORK / "asr.json"
    if asr_json.exists():
        segs = json.loads(asr_json.read_text())
        print(f"using {len(segs)} pre-transcribed segments from {asr_json}")
    else:
        segs = transcribe()
    embs = embed_segments(segs)
    speakers = cluster_speakers(embs)
    for i, (seg, spk) in enumerate(zip(segs, speakers)):
        seg["id"] = i
        seg["speaker"] = f"S{spk}"
    OUT.write_text(json.dumps(segs, indent=2, ensure_ascii=False))
    n_speakers = len(set(speakers))
    print(f"wrote {OUT} — {len(segs)} segments, {n_speakers} speakers")
    for seg in segs:
        print(f"[{seg['speaker']}] {seg['start']:7.2f}–{seg['end']:7.2f}  ({len(seg['text'])} chars)")


if __name__ == "__main__":
    main()
