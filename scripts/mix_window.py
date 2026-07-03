"""Mix and mux a time window of the dub without waiting for the full movie.

Usage:  VT_WORK=work_full VT_VIDEO=/path/movie.mp4 \
        uv run python scripts/mix_window.py --start 900 --end 1020 --output teaser_15_17.mp4
"""

import argparse
import json
import subprocess

import torch
import torchaudio

from hebrew_gender import speaker_overrides

from mix import (
    OUT_DIR, SR, VIDEO, WORK,
    duck_envelope, fit_segment, load_bed, slot_mask, speech_onset, vocal_fill, vocal_rms,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--end", type=float, required=True)
    parser.add_argument("--tts-dir", default="tts")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    segs = json.loads((WORK / "translations.json").read_text())
    speaker_overrides(WORK, segs)
    segs = sorted((s for s in segs if args.start <= s["start"] < args.end), key=lambda s: s["start"])

    total = int((args.end - args.start) * SR)
    mask = slot_mask(segs, total, offset_sec=args.start)
    rms = vocal_rms(total, offset_sec=args.start)
    bg = load_bed(mask, offset_sec=args.start)
    total = bg.shape[1]

    voice = torch.zeros(1, total)
    for i, seg in enumerate(segs):
        src = WORK / args.tts_dir / f"seg_{seg['id']:03d}.wav"
        slot = seg["end"] - seg["start"]
        place_at = speech_onset(rms, seg["start"], seg["end"], offset_sec=args.start)
        if i + 1 < len(segs):
            next_start = segs[i + 1]["start"] - place_at
            same = segs[i + 1]["speaker"] == seg["speaker"]
        else:
            next_start = args.end - place_at
            same = True
        fitted = fit_segment(src, slot, next_start, next_same_speaker=same)
        a = int((place_at - args.start) * SR)
        b = min(a + fitted.shape[1], total)
        voice[:, a:b] += fitted[:, : b - a]

    fade = min(int(0.15 * SR), total)
    voice[:, total - fade :] *= torch.linspace(1.0, 0.0, fade)
    mixed = bg * duck_envelope(voice) + voice + vocal_fill(mask, rms, total, offset_sec=args.start)
    peak = mixed.abs().max()
    if peak > 0.99:
        mixed = mixed / peak * 0.99

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mix_wav = OUT_DIR / "window_mix.wav"
    torchaudio.save(str(mix_wav), mixed, SR)

    out = OUT_DIR / args.output
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", str(args.start), "-to", str(args.end), "-i", str(VIDEO),
            "-i", str(mix_wav),
            "-map", "0:v", "-map", "1:a",
            "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-c:a", "aac", "-b:a", "192k", "-shortest", str(out),
        ],
        check=True,
    )
    mix_wav.unlink()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
