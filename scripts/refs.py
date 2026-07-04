"""Merge over-split speaker clusters and build one long reference clip per speaker.

Diarization on noisy movie audio over-splits: the same character lands in several
clusters, leaving each with too little reference audio. This pass:
  1. computes a duration-weighted ECAPA centroid per cluster,
  2. greedily merges clusters whose centroids are close,
  3. rewrites speaker labels in segments.json + translations.json,
  4. builds a ~20s RMS-normalized reference per merged speaker.

Reads:  work/segments.json, work/translations.json, demucs vocals
Writes: work/refs/<speaker>.wav, updated json files
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / os.environ.get("VT_WORK", "work")
VOCALS = WORK / "demucs/htdemucs/audio_44k/vocals.wav"
REF_TARGET_SEC = 20.0
MERGE_DIST = float(os.environ.get("VT_MERGE_DIST", "0.55"))  # cosine distance below which two cluster centroids are one speaker; lower it when speakers were hand-curated
REF_RMS = 0.05  # ~ -26 dBFS, healthy level for cloning references
SR = 16000


def cluster_embeddings(segs: list[dict], wav: torch.Tensor) -> dict[str, np.ndarray]:
    enc = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(ROOT / "work/models/ecapa"),
    )
    cents: dict[str, np.ndarray] = {}
    for spk in sorted({s["speaker"] for s in segs}):
        own = [s for s in segs if s["speaker"] == spk]
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
        cents[spk] = c / np.linalg.norm(c)
    return cents


def merge_clusters(cents: dict[str, np.ndarray], durations: dict[str, float]) -> dict[str, str]:
    """Greedy merge of nearest centroids until none are closer than MERGE_DIST."""
    groups = {spk: [spk] for spk in cents}
    while True:
        keys = sorted(groups)
        best, pair = MERGE_DIST, None
        for i, a in enumerate(keys):
            for b in keys[i + 1 :]:
                d = 1 - float(np.dot(cents[a], cents[b]))
                if d < best:
                    best, pair = d, (a, b)
        if pair is None:
            break
        a, b = pair
        print(f"merging {b} into {a} (dist {best:.2f})")
        wa, wb = durations[a], durations[b]
        c = cents[a] * wa + cents[b] * wb
        cents[a] = c / np.linalg.norm(c)
        durations[a] = wa + wb
        groups[a].extend(groups.pop(b))
        del cents[b]
    mapping = {}
    for canon, members in groups.items():
        for m in members:
            mapping[m] = canon
    return mapping


def main() -> None:
    segs = json.loads((WORK / "segments.json").read_text())
    wav, sr = torchaudio.load(str(VOCALS))
    wav = torchaudio.functional.resample(wav.mean(dim=0, keepdim=True), sr, SR)

    durations = {}
    for s in segs:
        durations[s["speaker"]] = durations.get(s["speaker"], 0) + s["end"] - s["start"]
    cents = cluster_embeddings(segs, wav)
    mapping = merge_clusters(cents, durations)

    for path in (WORK / "segments.json", WORK / "translations.json"):
        data = json.loads(path.read_text())
        for s in data:
            s["speaker"] = mapping[s["speaker"]]
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    speakers = sorted({s["speaker"] for s in segs for s in [s]} | set(mapping.values()))
    (WORK / "refs").mkdir(parents=True, exist_ok=True)
    segs = json.loads((WORK / "segments.json").read_text())
    for spk in sorted({s["speaker"] for s in segs}):
        own = sorted(
            (s for s in segs if s["speaker"] == spk),
            key=lambda s: s["end"] - s["start"],
            reverse=True,
        )
        chunks, total = [], 0.0
        for s in own:
            chunks.append(wav[:, int(s["start"] * SR) : int(s["end"] * SR)])
            total += s["end"] - s["start"]
            if total >= REF_TARGET_SEC:
                break
        ref = torch.cat(chunks, dim=1)
        rms = ref.pow(2).mean().sqrt()
        if rms > 0:
            ref = (ref * (REF_RMS / rms)).clamp(-0.99, 0.99)
        out = WORK / f"refs/{spk}.wav"
        torchaudio.save(str(out), ref, SR)
        print(f"{spk}: {total:.1f}s reference from {len(chunks)} segments")


if __name__ == "__main__":
    main()
