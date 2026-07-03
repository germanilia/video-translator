"""Run the full local dubbing pipeline on a complete video.

Usage:
  uv run python scripts/run_full.py "/path/to/movie.mp4" [--work work_full]

Stages are skipped when their output already exists, so the run is resumable.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import vt_config  # noqa: F401  (loads config.env into the environment)

ROOT = Path(__file__).resolve().parent.parent
CONFIG_HINT = "edit config.env in the repo root (copy config.env.example if it doesn't exist)"


def run(cmd: list[str], env: dict, label: str) -> None:
    t0 = time.time()
    print(f"\n=== {label}: {' '.join(cmd[:4])}...", flush=True)
    subprocess.run(cmd, check=True, env=env, cwd=ROOT)
    print(f"=== {label} done in {time.time() - t0:.0f}s", flush=True)


def ask(question: str, default: str = "") -> str:
    try:
        return input(question).strip() or default
    except EOFError:
        return default


def require_config(condition: bool, what: str, lines: list[str]) -> None:
    if condition:
        return
    print(f"\nMissing configuration for {what}.")
    print(f"To fix: {CONFIG_HINT} and fill in:")
    for line in lines:
        print(f"    {line}")
    sys.exit(2)


def wizard(args) -> None:
    """Ask every open decision; verify required config; explain how to fill gaps."""
    interactive = sys.stdin.isatty()
    if not (ROOT / "config.env").exists() and (ROOT / "config.env.example").exists():
        shutil.copy(ROOT / "config.env.example", ROOT / "config.env")
        print("created config.env from template - fill in keys there as needed")

    # diarization requirement applies to every run
    require_config(
        bool(os.environ.get("HF_TOKEN")),
        "speaker diarization (pyannote)",
        ["HF_TOKEN=hf_...   # free token; also accept the model terms at:",
         "#   https://hf.co/pyannote/speaker-diarization-3.1",
         "#   https://hf.co/pyannote/segmentation-3.0"],
    )

    # translation backend
    if args.translator is None:
        detected = ("pi" if shutil.which("pi") else
                    "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else
                    "openai" if os.environ.get("OPENAI_API_KEY") else None)
        if interactive:
            print("\nTranslation backend:")
            print("  [1] my AI agent already wrote <work>/translations.json (best quality)")
            print(f"  [2] remote frontier model {'(detected: ' + detected + ')' if detected else '(needs a key in config.env)'}")
            print("  [3] local Gemma via Ollama (free, offline)")
            choice = ask("choose [1/2/3, default 2 if available else 3]: ",
                         "2" if detected else "3")
            args.translator = {"1": "agent", "2": "remote", "3": "gemma"}.get(choice, "gemma")
        else:
            args.translator = "remote" if detected else "gemma"
    if args.translator == "remote":
        require_config(
            bool(shutil.which("pi") or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")),
            "remote translation",
            ["ANTHROPIC_API_KEY=sk-ant-...   # or:", "OPENAI_API_KEY=sk-...",
             "# or install+authenticate the pi CLI"],
        )

    # TTS backend
    if args.tts is None:
        if interactive:
            print("\nVoices (TTS):")
            print("  [1] free - local Chatterbox (Apple GPU, unlimited)")
            print("  [2] paid - ElevenLabs instant voice clones (closest to original voices)")
            args.tts = "elevenlabs" if ask("choose [1/2, default 1]: ", "1") == "2" else "local"
        else:
            args.tts = "local"
    if args.tts == "elevenlabs":
        require_config(
            bool(os.environ.get("ELEVENLABS_API_KEY")),
            "ElevenLabs voices",
            ["ELEVENLABS_API_KEY=...   # plan with instant voice cloning; ~40K chars per movie"],
        )

    # upscale
    if args.upscale is None:
        if interactive:
            answer = ask("\nUpscale final video 2x with Real-ESRGAN? Only worth it for low-res sources; takes hours for movies [y/N]: ").lower()
            args.upscale = "yes" if answer in ("y", "yes") else "no"
        else:
            args.upscale = "no"

    print(f"\nplan: translator={args.translator}, tts={args.tts}, upscale={args.upscale}, work={args.work}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--work", default="work_full")
    parser.add_argument("--tts", choices=["local", "elevenlabs"], default=None,
                        help="local = Chatterbox MLX (free); elevenlabs = paid API (needs ELEVENLABS_API_KEY)")
    parser.add_argument("--translator", choices=["agent", "remote", "gemma"], default=None,
                        help="agent = translations.json provided by your AI agent; remote = frontier API; gemma = local")
    parser.add_argument("--upscale", choices=["yes", "no"], default=None,
                        help="2x Real-ESRGAN upscale of the final video (movies: hours)")
    parser.add_argument("--teaser", default="", help="range to cut as teaser (HH:MM:SS-HH:MM:SS), empty to skip")
    args = parser.parse_args()

    wizard(args)

    video = Path(args.video).expanduser()
    work = ROOT / args.work
    (work / "clip").mkdir(parents=True, exist_ok=True)

    env = dict(os.environ, VT_WORK=args.work, VT_VIDEO=str(video))
    uv_py = ["uv", "run", "python"]
    mlx_py = [str(ROOT / ".venv-mlx/bin/python")]

    a16 = work / "clip/audio_16k.wav"
    a44 = work / "clip/audio_44k.wav"
    if not a44.exists():
        run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
             "-vn", "-ac", "1", "-ar", "16000", str(a16),
             "-vn", "-ac", "2", "-ar", "44100", str(a44)],
            env, "extract audio",
        )

    if not (work / "demucs/htdemucs/audio_44k/vocals.wav").exists():
        # MPS verified against CPU output (cosine sim 0.999); fallback covers unsupported ops
        demucs_env = dict(env, PYTORCH_ENABLE_MPS_FALLBACK="1")
        run(["uv", "run", "demucs", "--two-stems=vocals", "-d", "mps",
             "-o", str(work / "demucs"), str(a44)], demucs_env, "demucs separation")

    if not (work / "asr.json").exists():
        run([*mlx_py, "scripts/asr_mlx.py"], env, "transcribe (mlx-whisper)")

    if not (work / "segments.json").exists():
        run([*uv_py, "scripts/diarize.py"], env, "diarize (pyannote)")
        run([*uv_py, "scripts/gender.py"], env, "gender classification")

    if not (work / "translations.json").exists():
        if args.translator == "agent":
            sys.exit(f"translator=agent but {work / 'translations.json'} does not exist - "
                     "have your agent write it (contract in skill/SKILL.md), then rerun")
        if args.translator == "remote":
            run([*uv_py, "scripts/translate_remote.py"], env, "translate (remote frontier model)")
        else:
            run([*uv_py, "scripts/translate_ollama.py"], env, "translate (Gemma via Ollama)")
        # belt-and-braces: strip any explanation/meta junk the LLM slipped through
        run([*uv_py, "scripts/fix_translations.py"], env, "clean translations")
        # second pass: gender-agreement + grammar review of the whole script
        try:
            run([*uv_py, "scripts/review_translations.py"], env, "review translations")
        except subprocess.CalledProcessError:
            print("no remote reviewer available, skipping review pass", flush=True)

    if not (work / "refs").exists():
        run([*uv_py, "scripts/refs.py"], env, "speaker references")

    if args.tts == "elevenlabs":
        run([*uv_py, "scripts/tts_elevenlabs.py"], env, "TTS (ElevenLabs)")
        tts_dir, out_name = "tts_11labs", "movie_he_11labs.mp4"
    else:
        run([*mlx_py, "scripts/tts_mlx.py"], env, "TTS (MLX chatterbox)")
        tts_dir, out_name = "tts", "movie_he.mp4"

    out = work / f"output/{out_name}"
    if not out.exists():
        run([*uv_py, "scripts/mix.py", "--tts-dir", tts_dir, "--output", out_name], env, "mix + mux")

    if args.upscale == "yes":
        up_out = work / f"output/{Path(out_name).stem}_upscaled.mp4"
        if not up_out.exists():
            # the video track is identical across TTS backends: if the other
            # backend was already upscaled, just carry its video and swap audio
            other = "movie_he.mp4" if args.tts == "elevenlabs" else "movie_he_11labs.mp4"
            other_up = work / f"output/{Path(other).stem}_upscaled.mp4"
            if other_up.exists():
                run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(other_up), "-i", str(out),
                     "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                     "-c:a", "aac", "-b:a", "192k", "-shortest", str(up_out)],
                    env, "upscale (reuse shared video track)")
            else:
                run([*uv_py, "scripts/upscale.py", str(out), str(up_out)], env, "upscale (Real-ESRGAN 2x)")
        print(f"upscaled: {up_out}")

    if args.teaser:
        start, end = args.teaser.split("-")
        teaser = work / "output/teaser_15_17.mp4"
        run(
            ["ffmpeg", "-y", "-loglevel", "error", "-ss", start, "-to", end,
             "-i", str(out), "-c:v", "libx264", "-crf", "18", "-preset", "fast",
             "-c:a", "aac", "-b:a", "192k", str(teaser)],
            env, "teaser cut",
        )
        print(f"teaser: {teaser}")

    print(f"\nfull movie: {out}")


if __name__ == "__main__":
    sys.exit(main())
