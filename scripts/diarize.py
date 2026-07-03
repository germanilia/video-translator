"""Diarize with pyannote 3.1, split ASR segments at speaker turns, QC-reassign, detect gender.

Replaces the previous per-segment clustering. Whisper segments that span two
speakers are split at the turn boundary using word timestamps — this is what
fixes "one line spoken in the wrong voice" and "voice changes mid-character".

Requires HF_TOKEN in env (gated models: pyannote/speaker-diarization-3.1 + segmentation-3.0).

Reads:  <work>/asr.json (with word timestamps), demucs vocals
Writes: <work>/segments.json  [{id, start, end, speaker, text}]
        <work>/speakers.json  {speaker: {gender, f0_hz, total_sec}}
"""

import json
import os
from pathlib import Path

import huggingface_hub
import librosa
import numpy as np
import torch
import torchaudio

# pyannote 3.x still passes use_auth_token; huggingface_hub >= 1.0 renamed it to token.
_orig_download = huggingface_hub.hf_hub_download


def _compat_download(*args, **kwargs):
    tok = kwargs.pop("use_auth_token", None)
    if tok and "token" not in kwargs:
        kwargs["token"] = tok
    return _orig_download(*args, **kwargs)


huggingface_hub.hf_hub_download = _compat_download

# torch 2.6 defaults weights_only=True, which rejects pyannote's official
# checkpoints (they pickle TorchVersion). They come from the gated HF repo we
# explicitly trust, so restore the pre-2.6 behavior for this script only.
_orig_torch_load = torch.load


def _trusting_load(*args, **kwargs):
    kwargs["weights_only"] = False  # lightning passes True explicitly; override
    return _orig_torch_load(*args, **kwargs)


torch.load = _trusting_load

import pyannote.audio.core.pipeline as _pap  # noqa: E402

_pap.hf_hub_download = _compat_download
import pyannote.audio.core.model as _pam  # noqa: E402

if hasattr(_pam, "hf_hub_download"):
    _pam.hf_hub_download = _compat_download

from pyannote.audio import Pipeline  # noqa: E402
from speechbrain.inference.speaker import EncoderClassifier  # noqa: E402

import vt_config  # noqa: F401  (loads config.env)

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / os.environ.get("VT_WORK", "work")
VOCALS = WORK / "demucs/htdemucs/audio_44k/vocals.wav"
SR = 16000
MIN_SEG_SEC = 0.25
SCENE_GAP = 8.0
QC_OWN_SIM = 0.25  # below this similarity to own centroid, consider reassignment
QC_MARGIN = 0.10  # best other centroid must beat own by this much
F0_FEMALE = 165.0
F0_MALE = 155.0


def load_vocals() -> torch.Tensor:
    wav, sr = torchaudio.load(str(VOCALS))
    return torchaudio.functional.resample(wav.mean(dim=0, keepdim=True), sr, SR)


def run_pyannote(wav: torch.Tensor) -> list[tuple[float, float, str]]:
    pipe = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=os.environ["HF_TOKEN"]
    )
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    try:
        pipe.to(torch.device(device))
    except Exception:
        device = "cpu"
        pipe.to(torch.device(device))
    print(f"pyannote running on {device}...")
    diar = pipe({"waveform": wav, "sample_rate": SR})
    turns = [
        (turn.start, turn.end, spk)
        for turn, _, spk in diar.itertracks(yield_label=True)
    ]
    print(f"pyannote: {len(turns)} turns, {len({t[2] for t in turns})} speakers")
    return sorted(turns)


def owner_at(turns: list[tuple[float, float, str]], t: float) -> str | None:
    """Speaker whose turn covers time t; nearest turn if none covers it."""
    best, best_d = None, 1e9
    for a, b, spk in turns:
        if a <= t <= b:
            return spk
        d = min(abs(t - a), abs(t - b))
        if d < best_d:
            best, best_d = spk, d
    return best if best_d < 2.0 else None


def split_segments(asr: list[dict], turns) -> list[dict]:
    """Split each ASR segment at speaker-turn boundaries via word timestamps."""
    out = []
    for seg in asr:
        words = seg.get("words") or []
        if not words:
            spk = owner_at(turns, (seg["start"] + seg["end"]) / 2)
            if spk:
                out.append({**{k: seg[k] for k in ("start", "end", "text")}, "speaker": spk})
            continue
        groups: list[dict] = []
        for w in words:
            spk = owner_at(turns, (w["start"] + w["end"]) / 2)
            if spk is None:
                spk = groups[-1]["speaker"] if groups else None
            if spk is None:
                continue
            if groups and groups[-1]["speaker"] == spk:
                groups[-1]["end"] = w["end"]
                groups[-1]["text"] += w["word"]
            else:
                groups.append({"start": w["start"], "end": w["end"], "speaker": spk, "text": w["word"]})
        for g in groups:
            g["text"] = g["text"].strip()
            if g["text"] and g["end"] - g["start"] >= MIN_SEG_SEC:
                out.append(g)
    return out


def merge_fragments(segs: list[dict]) -> list[dict]:
    """Merge adjacent same-speaker segments with tiny gaps.

    Whisper splits sentences mid-clause; translating fragments independently
    produces incoherent Hebrew joins. Merged lines translate as one utterance.
    """
    out: list[dict] = []
    for s in segs:
        prev = out[-1] if out else None
        same = prev and s["speaker"] == prev["speaker"]
        # one-word orphans get their own (often wrong) cluster; absorb them
        prev_orphan = prev and len(prev["text"].split()) <= 1 and s["start"] - prev["end"] < 0.5
        if (prev and (same or prev_orphan)
                and s["start"] - prev["end"] < 0.35
                and s["end"] - prev["start"] < 12.0):
            prev["end"] = s["end"]
            prev["text"] = (prev["text"] + " " + s["text"]).strip()
            prev["speaker"] = s["speaker"] if prev_orphan and not same else prev["speaker"]
        else:
            out.append(dict(s))
    return out


def qc_reassign(segs: list[dict], wav: torch.Tensor) -> int:
    """Re-check every segment's speaker with ECAPA; move clear mismatches."""
    enc = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(ROOT / "work/models/ecapa")
    )

    def embed(a: float, b: float) -> np.ndarray:
        chunk = wav[:, int(a * SR) : int(b * SR)]
        if chunk.shape[1] < SR // 2:
            chunk = torch.nn.functional.pad(chunk, (0, SR // 2 - chunk.shape[1]))
        with torch.no_grad():
            e = enc.encode_batch(chunk).squeeze().numpy()
        return e / np.linalg.norm(e)

    embs = [embed(s["start"], s["end"]) for s in segs]
    cents: dict[str, np.ndarray] = {}
    for spk in {s["speaker"] for s in segs}:
        idx = [i for i, s in enumerate(segs) if s["speaker"] == spk]
        weights = [segs[i]["end"] - segs[i]["start"] for i in idx]
        c = np.average([embs[i] for i in idx], axis=0, weights=weights)
        cents[spk] = c / np.linalg.norm(c)

    moved = 0
    for i, seg in enumerate(segs):
        own = float(np.dot(embs[i], cents[seg["speaker"]]))
        best_spk, best = max(
            ((spk, float(np.dot(embs[i], c))) for spk, c in cents.items()), key=lambda kv: kv[1]
        )
        if best_spk != seg["speaker"] and own < QC_OWN_SIM and best > own + QC_MARGIN:
            seg["speaker"] = best_spk
            moved += 1
    print(f"QC: reassigned {moved} segments")
    return moved


def fix_short_replies(segs: list[dict], wav: torch.Tensor) -> int:
    """Short lines (1-2 words) often land in the wrong cluster - their embeddings
    are too weak for open clustering. Re-decide them as a RESTRICTED choice
    between the neighboring speakers' centroids, where weak evidence suffices."""
    enc = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(ROOT / "work/models/ecapa")
    )

    def embed(a: float, b: float) -> np.ndarray:
        chunk = wav[:, int(a * SR) : int(b * SR)]
        if chunk.shape[1] < SR // 2:
            chunk = torch.nn.functional.pad(chunk, (0, SR // 2 - chunk.shape[1]))
        with torch.no_grad():
            e = enc.encode_batch(chunk).squeeze().numpy()
        return e / np.linalg.norm(e)

    cents: dict[str, np.ndarray] = {}
    for spk in {s["speaker"] for s in segs}:
        own = [s for s in segs if s["speaker"] == spk and len(s["text"].split()) > 2]
        own = own or [s for s in segs if s["speaker"] == spk]
        embs = [embed(s["start"], s["end"]) for s in own[:10]]
        weights = [s["end"] - s["start"] for s in own[:10]]
        c = np.average(embs, axis=0, weights=weights)
        cents[spk] = c / np.linalg.norm(c)

    moved = 0
    for i, seg in enumerate(segs):
        if len(seg["text"].split()) > 2:
            continue
        candidates = {seg["speaker"]}
        for j in (i - 1, i + 1):
            if 0 <= j < len(segs) and abs(segs[j]["start"] - seg["end"]) < SCENE_GAP:
                candidates.add(segs[j]["speaker"])
        if len(candidates) < 2:
            continue
        e = embed(seg["start"], seg["end"])
        best = max(candidates, key=lambda spk: float(np.dot(e, cents[spk])))
        if best != seg["speaker"] and float(np.dot(e, cents[best])) > float(np.dot(e, cents[seg["speaker"]])) + 0.03:
            seg["speaker"] = best
            moved += 1
    print(f"short-reply fix: reassigned {moved} segments")
    return moved


def absorb_micro_clusters(segs: list[dict], wav: torch.Tensor) -> int:
    """Clusters with <2.5s total audio cannot produce a usable voice reference;
    fold their segments into the nearest substantial cluster by embedding."""
    enc = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(ROOT / "work/models/ecapa")
    )

    def embed(a: float, b: float) -> np.ndarray:
        chunk = wav[:, int(a * SR) : int(b * SR)]
        if chunk.shape[1] < SR // 2:
            chunk = torch.nn.functional.pad(chunk, (0, SR // 2 - chunk.shape[1]))
        with torch.no_grad():
            e = enc.encode_batch(chunk).squeeze().numpy()
        return e / np.linalg.norm(e)

    totals: dict[str, float] = {}
    for s in segs:
        totals[s["speaker"]] = totals.get(s["speaker"], 0.0) + s["end"] - s["start"]
    big = {spk for spk, t in totals.items() if t >= 2.5}
    if not big:
        return 0
    cents = {}
    for spk in big:
        own = [s for s in segs if s["speaker"] == spk][:10]
        embs = [embed(s["start"], s["end"]) for s in own]
        weights = [s["end"] - s["start"] for s in own]
        c = np.average(embs, axis=0, weights=weights)
        cents[spk] = c / np.linalg.norm(c)

    moved = 0
    for seg in segs:
        if seg["speaker"] in big:
            continue
        e = embed(seg["start"], seg["end"])
        seg["speaker"] = max(cents, key=lambda spk: float(np.dot(e, cents[spk])))
        moved += 1
    if moved:
        print(f"micro-cluster absorb: reassigned {moved} segments into substantial clusters")
    return moved


def detect_gender(segs: list[dict], wav: torch.Tensor) -> dict[str, dict]:
    """Median F0 per speaker (pooled up to 30s of their audio)."""
    speakers = {}
    for spk in sorted({s["speaker"] for s in segs}):
        own = sorted((s for s in segs if s["speaker"] == spk), key=lambda s: s["start"])
        chunks, total = [], 0.0
        for s in own:
            chunks.append(wav[0, int(s["start"] * SR) : int(s["end"] * SR)].numpy())
            total += s["end"] - s["start"]
            if total >= 30:
                break
        audio = np.concatenate(chunks) if chunks else np.zeros(SR)
        f0 = librosa.pyin(audio, fmin=65, fmax=400, sr=SR)[0]
        f0 = f0[~np.isnan(f0)]
        med = float(np.median(f0)) if len(f0) else 0.0
        gender = "female" if med >= F0_FEMALE else "male" if 0 < med <= F0_MALE else "unknown"
        speakers[spk] = {"gender": gender, "f0_hz": round(med, 1), "total_sec": round(total, 1)}
        print(f"{spk}: f0={med:.0f}Hz -> {gender} ({total:.0f}s)")
    return speakers


def main() -> None:
    asr = json.loads((WORK / "asr.json").read_text())
    wav = load_vocals()
    turns = run_pyannote(wav)
    segs = split_segments(asr, turns)
    segs = merge_fragments(segs)
    qc_reassign(segs, wav)
    fix_short_replies(segs, wav)
    absorb_micro_clusters(segs, wav)
    segs = merge_fragments(segs)  # again: label corrections can join sentence halves
    speakers = detect_gender(segs, wav)
    for i, s in enumerate(segs):
        s["id"] = i
        s["start"] = round(s["start"], 3)
        s["end"] = round(s["end"], 3)
    (WORK / "segments.json").write_text(json.dumps(segs, indent=2, ensure_ascii=False))
    (WORK / "speakers.json").write_text(json.dumps(speakers, indent=2))
    print(f"wrote {len(segs)} segments, {len(speakers)} speakers")
    # F0 medians misgender excited speech; always finish with the real classifier
    # so downstream stages can never see the crude F0 labels.
    import gender

    gender.main()


if __name__ == "__main__":
    main()
