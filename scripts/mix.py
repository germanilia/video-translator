"""Fit TTS segments to their original timing, mix over the background track, mux video.

Per segment: tempo-fit into its slot (capped), normalize to dialogue level.
Background is ducked under speech. Final mix is loudness-normalized.

Reads:  work/translations.json, work/tts/seg_*.wav,
        work/demucs/htdemucs/audio_44k/no_vocals.wav, work/clip/clip.mp4
Writes: work/output/clip_he.mp4
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

import torch
import torchaudio

from hebrew_gender import speaker_overrides

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / os.environ.get("VT_WORK", "work")
BACKGROUND = WORK / "demucs/htdemucs/audio_44k/no_vocals.wav"
VIDEO = Path(os.environ.get("VT_VIDEO", WORK / "clip/clip.mp4"))
OUT_DIR = WORK / "output"
MAX_TEMPO = 1.45
SR = 44100
VOICE_RMS = 0.085  # dialogue level; keep below original so effects/music stay present
DUCK_GAIN = 0.75  # background gain under speech (gentle duck)
DUCK_FADE_SEC = 0.08
# Demucs strips most energy into the vocals stem during dialogue, leaving a hollow
# bed. Blend a little of the original mix back in to restore effects/room tone —
# but inside dubbed line windows drop it so the original English stays inaudible.
BED_BLEND = 0.18
BED_BLEND_IN_SPEECH = 0.04
MAX_SPILL_FACTOR = 1.25  # a line may exceed its slot by at most this + 0.4s
ORIGINAL = WORK / "clip/audio_44k.wav"
VOCALS = WORK / "demucs/htdemucs/audio_44k/vocals.wav"
# Untranscribed vocal moments (grunts, screams, overlaps) have no dub line;
# pass the original vocals through there so mouths are never silent. Fill is
# masked out across each dubbed line's WHOLE original window, so a Hebrew line
# that runs shorter than the English one never lets the English tail through.
FILL_GAIN = 0.9
FILL_THRESH = 0.006  # vocal-stem RMS above this counts as "someone is vocalizing"
# In continuous dialogue, the short pauses between transcribed lines contain
# original-language tails/breaths; without bridging, fill pushes them through
# between dub lines and it reads as an "echo" of the original actor.
FILL_BRIDGE_SEC = float(os.environ.get("VT_FILL_BRIDGE_SEC", "0.8"))
START_PAD = 0.15  # delay each dub line; whisper starts run early (VAD pre-roll)
SLOT_TAIL = 0.35  # dubbed window mask extends this far past the original line end


def fit_segment(
    src: Path, slot: float, next_start: float | None, next_same_speaker: bool = True
) -> torch.Tensor:
    """Load a TTS wav, tempo-fit into its slot, resample to SR, normalize level."""
    wav, sr = torchaudio.load(str(src))
    wav = wav.mean(dim=0, keepdim=True)
    dur = wav.shape[1] / sr

    # allow limited spill past the slot (lip-sync) — but NEVER into a different
    # speaker's line: that reads as one character talking over another.
    budget = slot * MAX_SPILL_FACTOR + 0.4
    if next_start is not None:
        gap_pad = 0.15 if next_same_speaker else 0.30
        hard_cap = max(slot * 0.6, next_start - gap_pad)
        budget = min(budget, hard_cap) if not next_same_speaker else min(budget, max(slot, next_start - gap_pad))
    tempo = min(max(dur / budget, 1.0), MAX_TEMPO)

    tmp = src.with_suffix(".fit.wav")
    filters = [f"atempo={tempo:.4f}"] if tempo > 1.001 else []
    filters.append(f"aresample={SR}")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-af", ",".join(filters), "-ac", "1", str(tmp)],
        check=True,
    )
    fitted, _ = torchaudio.load(str(tmp))
    tmp.unlink()
    fitted = fitted.mean(dim=0, keepdim=True)
    rms = fitted.pow(2).mean().sqrt()
    if rms > 0:
        fitted = fitted * (VOICE_RMS / rms)
    return fitted.clamp(-0.99, 0.99)


def vocal_rms(length: int, offset_sec: float = 0.0) -> torch.Tensor:
    """50ms moving RMS of the vocals stem, aligned to the mix timeline."""
    voc, vsr = torchaudio.load(str(VOCALS))
    voc = torchaudio.functional.resample(voc, vsr, SR) if vsr != SR else voc
    a = int(offset_sec * SR)
    voc = voc.mean(dim=0)[a : a + length]
    if voc.shape[0] < length:
        voc = torch.nn.functional.pad(voc, (0, length - voc.shape[0]))
    k = int(0.05 * SR)
    cs = torch.cat([torch.zeros(1, dtype=torch.float64), voc.double().pow(2).cumsum(0)])
    rms = ((cs[k:] - cs[:-k]) / k).sqrt()
    return torch.cat([rms[:1].repeat(k // 2), rms, rms[-1:].repeat(length - rms.shape[0] - k // 2)]).float()


def speech_onset(rms: torch.Tensor, start: float, end: float, offset_sec: float = 0.0) -> float:
    """When the voice actually starts inside a segment window (whisper runs early)."""
    a = max(0, int((start - offset_sec) * SR))
    b = min(rms.shape[0], int((end - offset_sec) * SR))
    active = (rms[a:b] > FILL_THRESH).nonzero()
    if len(active) == 0:
        return start
    return offset_sec + (a + active[0].item()) / SR - 0.03


def slot_mask(segs: list[dict], length: int, offset_sec: float = 0.0) -> torch.Tensor:
    """1.0 inside any dubbed line's original time window (padded), 0.0 elsewhere."""
    mask = torch.zeros(length)
    windows = []
    for seg in segs:
        a = int((seg["start"] - offset_sec - 0.05) * SR)
        b = int((seg["end"] - offset_sec + SLOT_TAIL) * SR)
        mask[max(0, a) : min(length, b)] = 1.0
        windows.append((a, b))
    # bridge short inter-line gaps so fill/bed can't leak original-language bursts
    windows.sort()
    prev_end = None
    for a, b in windows:
        if prev_end is not None and 0 < a - prev_end < int(FILL_BRIDGE_SEC * SR):
            mask[max(0, prev_end) : min(length, a)] = 1.0
        prev_end = b if prev_end is None else max(prev_end, b)
    return _smooth(mask, DUCK_FADE_SEC).clamp(0, 1)


def load_bed(mask: torch.Tensor, offset_sec: float = 0.0) -> torch.Tensor:
    """Background bed: demucs no_vocals plus original mix — near-muted in dub windows."""
    a = int(offset_sec * SR)
    bg, bsr = torchaudio.load(str(BACKGROUND))
    bg = torchaudio.functional.resample(bg, bsr, SR) if bsr != SR else bg
    bg = bg.mean(dim=0, keepdim=True)[:, a : a + mask.shape[0]]
    orig, osr = torchaudio.load(str(ORIGINAL))
    orig = torchaudio.functional.resample(orig, osr, SR) if osr != SR else orig
    orig = orig.mean(dim=0, keepdim=True)[:, a : a + mask.shape[0]]
    n = min(bg.shape[1], orig.shape[1], mask.shape[0])
    blend = BED_BLEND * (1.0 - mask[:n]) + BED_BLEND_IN_SPEECH * mask[:n]
    return bg[:, :n] + blend.unsqueeze(0) * orig[:, :n]


def _smooth(active: torch.Tensor, fade_sec: float) -> torch.Tensor:
    """O(N) moving average of a 1-D 0/1 signal, same length out."""
    k = max(1, int(fade_sec * SR))
    cs = torch.cat([torch.zeros(1, dtype=torch.float64), active.double().cumsum(0)])
    smooth = (cs[k:] - cs[:-k]) / k
    pad_l = k // 2
    pad_r = active.shape[0] - smooth.shape[0] - pad_l
    return torch.cat([smooth[:1].repeat(pad_l), smooth, smooth[-1:].repeat(pad_r)]).float()


def duck_envelope(voice: torch.Tensor) -> torch.Tensor:
    """Per-sample background gain: DUCK_GAIN under speech, 1.0 elsewhere, smoothed."""
    smooth = _smooth((voice.abs() > 1e-4).squeeze(0), DUCK_FADE_SEC)
    return (1.0 - (1.0 - DUCK_GAIN) * smooth.clamp(0, 1)).unsqueeze(0)


def vocal_fill(mask: torch.Tensor, rms: torch.Tensor, length: int, offset_sec: float = 0.0) -> torch.Tensor:
    """Original vocals passed through only OUTSIDE dubbed line windows.

    Covers grunts, screams, and vocals whisper never transcribed — mouths never
    move in silence — while the whole window of every dubbed line stays clear of
    the original language, even when the dub runs shorter than the original.
    """
    voc, vsr = torchaudio.load(str(VOCALS))
    voc = torchaudio.functional.resample(voc, vsr, SR) if vsr != SR else voc
    a = int(offset_sec * SR)
    voc = voc.mean(dim=0, keepdim=True)[:, a : a + length]
    if voc.shape[1] < length:
        voc = torch.nn.functional.pad(voc, (0, length - voc.shape[1]))

    vocal_active = (rms > FILL_THRESH).float()
    fill_env = _smooth(vocal_active * (mask[:length] < 0.02).float(), DUCK_FADE_SEC)
    return voc * fill_env.unsqueeze(0) * FILL_GAIN


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tts-dir", default="tts", help="directory with seg_<id>.wav files, relative to work dir")
    parser.add_argument("--output", default="clip_he.mp4", help="output video filename")
    args = parser.parse_args()
    tts_dir = WORK / args.tts_dir

    segs = json.loads((WORK / "translations.json").read_text())
    speaker_overrides(WORK, segs)
    segs.sort(key=lambda s: s["start"])
    probe = torchaudio.info(str(BACKGROUND))
    total = int(probe.num_frames * SR / probe.sample_rate)
    mask = slot_mask(segs, total)
    rms = vocal_rms(total)
    bg = load_bed(mask)
    total = bg.shape[1]

    voice = torch.zeros(1, total)
    for i, seg in enumerate(segs):
        src = tts_dir / f"seg_{seg['id']:03d}.wav"
        slot = seg["end"] - seg["start"]
        place_at = speech_onset(rms, seg["start"], seg["end"])
        if i + 1 < len(segs):
            next_start = segs[i + 1]["start"] - place_at
            same = segs[i + 1]["speaker"] == seg["speaker"]
        else:
            # last line: fit within remaining runtime with NORMAL pacing rules;
            # the end-of-track fade below covers pathological overruns
            next_start = total / SR - place_at
            same = True
        fitted = fit_segment(src, slot, next_start, next_same_speaker=same)
        a = int(place_at * SR)
        b = min(a + fitted.shape[1], total)
        voice[:, a:b] += fitted[:, : b - a]
        print(f"seg {seg['id']:03d} slot {slot:5.2f}s placed {fitted.shape[1] / SR:5.2f}s @+{place_at - seg['start']:.2f}s")

    fade = min(int(0.15 * SR), total)
    voice[:, total - fade :] *= torch.linspace(1.0, 0.0, fade)
    mixed = bg * duck_envelope(voice) + voice + vocal_fill(mask, rms, total)
    peak = mixed.abs().max()
    if peak > 0.99:
        mixed = mixed / peak * 0.99

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mix_wav = OUT_DIR / f"{Path(args.output).stem}_mix.wav"
    torchaudio.save(str(mix_wav), mixed, SR)

    out = OUT_DIR / args.output
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(VIDEO), "-i", str(mix_wav),
            "-map", "0:v", "-map", "1:a",
            "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-shortest", str(out),
        ],
        check=True,
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
