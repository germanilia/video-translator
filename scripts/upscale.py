"""Upscale a video with Real-ESRGAN (ncnn, Apple GPU via Vulkan/MoltenVK).

Processes in chunks (extract frames -> upscale -> encode -> delete), so disk
usage stays bounded and full movies are feasible. Resumable: finished chunks
are kept and skipped on rerun.

Usage:
  uv run python scripts/upscale.py input.mp4 output.mp4 [--model realesr-animevideov3] [--scale 2]

Models / speed on M1 Max:
  realesr-animevideov3 (default): ~6 fps -> ~2 min per video-minute. Movies: overnight.
  realesrgan-x4plus: ~0.2 fps, best quality on live action - short clips only.
"""

import argparse
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "tools/realesrgan/realesrgan-ncnn-vulkan"
MODELS = ROOT / "tools/realesrgan/models"
CHUNK_SEC = 120
# frame extraction and encoding are CPU work; overlapping them with the GPU
# stage keeps the GPU continuously busy across chunks
PIPELINE_WORKERS = 3
GPU_LOCK = threading.Lock()


def probe(src: Path, entry: str) -> str:
    return subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", entry, "-of", "csv=p=0", str(src)],
        capture_output=True, text=True, check=True,
    ).stdout.strip().split(",")[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--model", default="realesr-animevideov3")
    parser.add_argument("--scale", default="2")
    args = parser.parse_args()

    src = Path(args.input).resolve()
    dst = Path(args.output).resolve()
    fps = probe(src, "stream=avg_frame_rate")
    duration = float(probe(src, "format=duration").split(",")[0] or
                     subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                     "-of", "csv=p=0", str(src)], capture_output=True, text=True).stdout.strip())

    workdir = dst.parent / f".upscale_{dst.stem}"
    workdir.mkdir(parents=True, exist_ok=True)
    n_chunks = int(duration // CHUNK_SEC) + 1
    chunk_files = [workdir / f"chunk_{c:04d}.mp4" for c in range(n_chunks)]
    done_count = threading.Lock()
    finished = [0]

    def process_chunk(c: int) -> bool:
        chunk_out = chunk_files[c]
        if chunk_out.exists():
            return True
        frames = workdir / f"in_{c:04d}"
        out_frames = workdir / f"out_{c:04d}"
        for d in (frames, out_frames):
            shutil.rmtree(d, ignore_errors=True)
            d.mkdir()
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-ss", str(c * CHUNK_SEC), "-t", str(CHUNK_SEC),
             "-i", str(src), "-vsync", "0", str(frames / "f%06d.png")],
            check=True,
        )
        if not any(frames.iterdir()):
            shutil.rmtree(frames)
            shutil.rmtree(out_frames)
            return False
        with GPU_LOCK:  # one realesrgan at a time; it saturates the GPU alone
            subprocess.run(
                [str(BIN), "-i", str(frames), "-o", str(out_frames),
                 "-n", args.model, "-s", args.scale, "-m", str(MODELS)],
                check=True, capture_output=True,
            )
        shutil.rmtree(frames)
        tmp = chunk_out.with_suffix(".tmp.mp4")
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-framerate", fps, "-i", str(out_frames / "f%06d.png"),
             "-c:v", "libx264", "-crf", "20", "-preset", "medium", "-pix_fmt", "yuv420p", str(tmp)],
            check=True,
        )
        tmp.rename(chunk_out)
        shutil.rmtree(out_frames)
        with done_count:
            finished[0] += 1
            print(f"chunk {c + 1}/{n_chunks} done ({finished[0]} this run)", flush=True)
        return True

    with ThreadPoolExecutor(max_workers=PIPELINE_WORKERS) as pool:
        results = list(pool.map(process_chunk, range(n_chunks)))
    chunk_files = [f for f, ok in zip(chunk_files, results) if ok]

    concat_list = workdir / "concat.txt"
    concat_list.write_text("".join(f"file '{f}'\n" for f in chunk_files))
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", str(concat_list), "-i", str(src),
         "-map", "0:v", "-map", "1:a?",
         "-c:v", "copy", "-c:a", "aac", "-ac", "2", "-b:a", "192k", "-shortest", str(dst)],
        check=True,
    )
    shutil.rmtree(workdir)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
