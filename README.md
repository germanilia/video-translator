# Video Translator

Dub videos into Hebrew while keeping each speaker's original voice. Runs fully
locally and free on Apple Silicon (M-series), with an optional ElevenLabs
backend for higher voice-clone fidelity. Both backends are first-class and
permanently supported.

Built for translating movies/shows for a Hebrew-speaking child: the original
music and sound effects are preserved, only the dialogue is replaced.

## Samples

Short Hebrew-dub demos from both supported TTS backends — click to play:

| Clip | Local backend (free) | ElevenLabs backend |
|---|---|---|
| Sintel scene | [🎬 Local](samples/sintel_hebrew_local.mp4) | [🎬 ElevenLabs](samples/sintel_hebrew_elevenlabs.mp4) |
| Bluey — opener | [🎬 Local](samples/bluey_opener_local.mp4) | [🎬 ElevenLabs](samples/bluey_opener_elevenlabs.mp4) |
| Bluey — Bella scene | [🎬 Local](samples/bluey_bella_local.mp4) | [🎬 ElevenLabs](samples/bluey_bella_elevenlabs.mp4) |
| Bluey — sleepover | [🎬 Local](samples/bluey_sleepover_local.mp4) | [🎬 ElevenLabs](samples/bluey_sleepover_elevenlabs.mp4) |
| TMNT trailer | [🎬 Local](samples/tmnt_trailer_hebrew_local.mp4) | [🎬 ElevenLabs](samples/tmnt_trailer_hebrew_elevenlabs.mp4) |

Original comparison clip: [🎬 Sintel source audio](samples/sintel_original.mp4)
(© Blender Foundation, CC-BY 3.0).

The paired local/ElevenLabs samples share the same video track and translation,
so the comparison is purely about the dubbed audio. Generated end-to-end by
`scripts/run_full.py` or `scripts/mix_window.py` for short demo cuts.

## How it works

```
video ──► 1. extract audio          (ffmpeg)
          2. separate vocals        (Demucs htdemucs — keeps music/effects track)
          3. transcribe             (mlx-whisper large-v3-turbo, Apple GPU)
          4. diarize speakers       (SpeechBrain ECAPA embeddings + clustering)
          5. translate EN→HE        (Gemma via Ollama | NLLB-200 local | hand-made)
          6. build voice references (per speaker, ~20s from their own lines)
          7. synthesize Hebrew      (Chatterbox multilingual MLX | ElevenLabs v3)
          8. fit + mix + mux        (tempo-fit into slots, duck background, ffmpeg)
video_he ◄┘   ... and optionally 9. upscale (Real-ESRGAN ncnn)
```

Key design decisions:

- **Clone once per speaker.** Diarization groups dialogue by speaker; each
  speaker gets one reference clip built from their own cleanest lines.
  Chatterbox clones zero-shot from the reference at generation time (no cost);
  ElevenLabs clones are created once and cached in `work/voices_11labs.json`.
- **Background preserved.** Demucs splits vocals from music/effects. The dub is
  mixed over the original background track with ducking under speech, so the
  score and sound effects survive.
- **Timing fit.** Each generated line is silence-trimmed, tempo-fitted into its
  original slot (max 1.35x speed-up, allowed to spill into pauses), and
  loudness-normalized.
- **Everything is resumable.** Every stage skips work whose output already
  exists; the full-movie runner can be killed and relaunched at any point.

## Components

| Stage | Tool | Where it runs |
|---|---|---|
| Audio extract / mux / cut | ffmpeg | CPU |
| Vocal separation | [Demucs](https://github.com/facebookresearch/demucs) `htdemucs` | Apple GPU (MPS) |
| Transcription | [mlx-whisper](https://github.com/ml-explore/mlx-examples) `whisper-large-v3-turbo` | Apple GPU |
| Diarization | [pyannote](https://github.com/pyannote/pyannote-audio) `speaker-diarization-3.1` + ECAPA QC pass | Apple GPU (MPS) |
| Gender detection | wav2vec2 voice classifier (feeds gendered Hebrew forms) | CPU |
| Translation (default) | frontier LLM via the [pi CLI](https://github.com/earendil-works/pi) print mode (scene context, gendered Hebrew) | API |
| Translation (local fallback) | Gemma via [Ollama](https://ollama.com) (`gemma4:e4b`) | Apple GPU |
| Translation (fallback) | [NLLB-200](https://huggingface.co/facebook/nllb-200-distilled-600M) distilled 600M | CPU |
| Hebrew niqqud | [dicta-onnx](https://github.com/thewh1teagle/dicta-onnx) (vowelization for correct TTS pronunciation) | CPU |
| TTS local (default) | [Chatterbox multilingual](https://github.com/resemble-ai/chatterbox) via [mlx-audio](https://github.com/Blaizzy/mlx-audio) (`mlx-community/chatterbox-fp16`) | Apple GPU |
| TTS paid (optional) | [ElevenLabs](https://elevenlabs.io) `eleven_v3` + instant voice clones | API |
| Upscaling (optional) | [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) ncnn-vulkan | Apple GPU |

Two Python environments (dependency conflicts between torch- and mlx-based stacks):

- `.venv` (uv-managed) — torch stack: Demucs, SpeechBrain, faster-whisper (CPU
  fallback), NLLB, PyTorch Chatterbox (fallback), mixing.
- `.venv-mlx` — MLX stack: mlx-audio (Chatterbox), mlx-whisper, dicta.

## Setup

```bash
# 1. System tools
brew install ffmpeg uv ollama

# 2. Main env (torch stack)
uv venv --python /opt/homebrew/bin/python3.11
uv sync            # or: uv add chatterbox-tts demucs faster-whisper speechbrain soundfile "numba>=0.59" "llvmlite>=0.42" dicta-onnx

# 3. MLX env (Apple-GPU stack)
uv venv --python /opt/homebrew/bin/python3.11 .venv-mlx
VIRTUAL_ENV=.venv-mlx uv pip install mlx-audio mlx-whisper dicta-onnx soundfile

# 4. Dicta niqqud model (~1.2GB, required for Hebrew pronunciation)
mkdir -p work/models
curl -L -o work/models/dicta-1.0.onnx \
  "https://github.com/thewh1teagle/dicta-onnx/releases/download/model-files-v1.0/dicta-1.0.onnx"

# 5. Translation model
ollama pull gemma4:e4b

# 5b. Diarization (pyannote, gated models — free):
#     create a HuggingFace token, accept terms at
#     hf.co/pyannote/speaker-diarization-3.1 and hf.co/pyannote/segmentation-3.0
export HF_TOKEN=hf_...

# 6. Optional: ElevenLabs backend
export ELEVENLABS_API_KEY=...   # paid plan with instant voice cloning

# 7. Optional: upscaler binary in tools/realesrgan/
#    https://github.com/xinntao/Real-ESRGAN/releases (macos zip), then:
#    xattr -c tools/realesrgan/realesrgan-ncnn-vulkan
```

Model weights (whisper, chatterbox, ECAPA, NLLB) download automatically on
first use into the HuggingFace cache.

Using an AI agent to operate this repo? Load [`skill/SKILL.md`](skill/SKILL.md)
into its context — it is the operating manual for the whole pipeline.

## Running

### Full video, one command — wizard

```bash
uv run python scripts/run_full.py "/path/to/movie.mp4" --work work_myvideo
```

Run interactively and a wizard walks every decision — translation backend
(your AI agent / remote frontier model / local Gemma), voices (free local /
paid ElevenLabs), and upscaling — verifying required keys at each step and
telling you exactly which line of `config.env` to fill when one is missing.
All keys and defaults live in **`config.env`** (auto-created from
[`config.env.example`](config.env.example), gitignored).

For unattended runs, pass answers as flags (no prompts):

```bash
uv run python scripts/run_full.py <video> --work work_x --translator remote --tts local --upscale no
uv run python scripts/run_full.py <video> --work work_x --translator gemma --tts elevenlabs --upscale yes
```

Produces `<work>/output/movie_he.mp4` (local) or `movie_he_11labs.mp4`
(ElevenLabs), plus an optional teaser cut (`--teaser 00:15:00-00:17:00`).
Stages are skipped when their outputs exist — safe to interrupt and relaunch.
The runner automatically cleans LLM translation output (`fix_translations.py`)
before any speech is generated.

On an M1 Max, a ~110-minute movie takes roughly **2–2.5 hours**
(Demucs ~10 min (GPU), whisper ~7 min, translation ~40 min, TTS ~1.5 h, mix ~15 min).

### Stage by stage (what run_full.py does)

All stages read/write under a work dir chosen by `VT_WORK` (default `work`):

```bash
export VT_WORK=work_full
export VT_VIDEO="/path/to/movie.mp4"

ffmpeg -i "$VT_VIDEO" -vn -ac 1 -ar 16000 $VT_WORK/clip/audio_16k.wav \
                      -vn -ac 2 -ar 44100 $VT_WORK/clip/audio_44k.wav
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run demucs --two-stems=vocals -d mps -o $VT_WORK/demucs $VT_WORK/clip/audio_44k.wav
.venv-mlx/bin/python scripts/asr_mlx.py        # -> asr.json (with word timestamps)
uv run python scripts/diarize.py               # -> segments.json (pyannote turns, split at speaker changes, QC pass)
uv run python scripts/gender.py                # -> speakers.json (voice gender -> gendered Hebrew)
uv run python scripts/translate_remote.py      # -> translations.json (remote frontier; or translate_ollama.py)
uv run python scripts/fix_translations.py      # ALWAYS: strip LLM explanations/junk
uv run python scripts/refs.py                  # -> refs/<speaker>.wav (+ cluster merge)
.venv-mlx/bin/python scripts/tts_mlx.py        # -> tts/seg_*.wav
uv run python scripts/mix.py --output movie_he.mp4
```

### Choosing the TTS backend

```bash
# Local / free (default) — Chatterbox multilingual on Apple GPU
.venv-mlx/bin/python scripts/tts_mlx.py
uv run python scripts/mix.py --tts-dir tts --output clip_he.mp4

# ElevenLabs — clones each speaker once (cached), then eleven_v3 TTS
uv run python scripts/tts_elevenlabs.py
uv run python scripts/mix.py --tts-dir tts_11labs --output clip_he_11labs.mp4
```

Both can coexist; segments land in separate dirs and `mix.py` builds whichever
you point it at. ElevenLabs sounds noticeably closer to the original voices;
Chatterbox is free and unlimited. A 2-minute scene costs ~800 ElevenLabs
characters; a full movie ~150–200K (most of a Starter plan's monthly quota).

### Choosing the translation backend

```bash
uv run python scripts/translate_remote.py   # remote frontier model — pi CLI, or ANTHROPIC_API_KEY / OPENAI_API_KEY; parallel batches
uv run python scripts/translate_ollama.py   # Gemma via Ollama — local/free fallback
uv run python scripts/translate.py          # NLLB-200 — fastest, most literal, no Ollama needed
```

`translate_remote.py` picks its provider from `VT_TRANSLATOR` (`pi` |
`anthropic` | `openai`, auto-detected from what's installed/keyed), runs
batches in parallel, batches whole scenes with context, applies speaker-gender
tags, spells numbers as Hebrew words for TTS, auto-detects providers that
return Hebrew in visual order, and falls back to Gemma per-line on failures.
`run_full.py` tries the remote backend first and falls back to Gemma.

`translate_ollama.py` checkpoints progress and falls back to existing NLLB
lines when the model returns something unparseable.

### Optional: upscale (source was 640×266)

```bash
# fast video model, ~6 fps on M1 Max (movie: overnight; teaser: ~8 min)
tools/realesrgan/realesrgan-ncnn-vulkan -i frames -o out -n realesr-animevideov3 -s 2 \
  -m tools/realesrgan/models

# max quality, ~0.2 fps — short clips only
... -n realesrgan-x4plus -s 4
```

For casual viewing, real-time playback upscaling (IINA/mpv + Anime4K shaders)
is free and instant — often the better trade for full movies.

## Work dir layout

```
work*/                       # one per source video (VT_WORK)
  clip/                      # extracted audio (+ re-encoded clip for tests)
  demucs/htdemucs/audio_44k/ # vocals.wav + no_vocals.wav
  asr.json                   # raw transcription
  segments.json              # + speaker labels
  translations.json          # + text_he
  refs/<speaker>.wav         # per-speaker cloning references
  tts/seg_<id>.wav           # generated Hebrew lines (tts_11labs/ for ElevenLabs)
  voices_11labs.json         # cached ElevenLabs voice ids (clone once, reuse)
  output/                    # final videos + mixed audio
```

## For AI Agents

If you are an LLM agent operating this repo, read [`skill/SKILL.md`](skill/SKILL.md)
first — it is the agent-agnostic operating manual (stage table, env contracts,
fix/remix recipes, upscaling, troubleshooting). Point your agent at it however
your framework loads instructions (Claude Code picks it up automatically via
`.claude/skills/`; for other agents, include `skill/SKILL.md` in context or
your rules file). The condensed contract:

**Install** (macOS Apple Silicon):
1. `brew install ffmpeg uv ollama` and `ollama pull gemma4:e4b`
2. Main env: `uv venv --python /opt/homebrew/bin/python3.11 && uv sync`
3. MLX env: `uv venv --python /opt/homebrew/bin/python3.11 .venv-mlx && VIRTUAL_ENV=.venv-mlx uv pip install mlx-audio mlx-whisper dicta-onnx soundfile`
4. Dicta niqqud model → `work/models/dicta-1.0.onnx` (URL in Setup above)

**Run — free (default)**: `uv run python scripts/run_full.py <video> --work work_x`
**Run — paid**: same command with `--tts elevenlabs`; requires `ELEVENLABS_API_KEY`
env var (never commit it), ~40K chars per 90-min movie, voice clones are cached
in `<work>/voices_11labs.json` and must never be re-cloned per run.

**Translation rule for agents**: if you are a capable multilingual LLM, write
the Hebrew translations yourself (see the contract in `skill/SKILL.md`) instead
of invoking the local Gemma model — the scripted backends are fallbacks for
unattended runs.

**Hard rules for agents:**
- Both TTS backends (local Chatterbox MLX and ElevenLabs) must always remain
  supported — never remove one.
- After every `translate_ollama.py` run, ALWAYS run `fix_translations.py`
  (the LLM sometimes emits explanations/alternatives that would be spoken
  aloud; the cleaner re-translates flagged lines and invalidates their TTS).
  `run_full.py` already chains this automatically.
- The translation prompt's strict rules (one translation per line, Hebrew
  letters only, transliterate names, no alternatives/explanations) must be
  preserved in any prompt edit.
- Mix/timing iteration never requires re-running TTS — see the mix knobs in
  the skill file; remixing a full movie takes ~5 minutes.
- The two venvs conflict (torch vs MLX) — never merge them.

## Known limitations

- **Voice similarity** is limited by zero-shot cloning from noisy movie audio;
  ElevenLabs does better than Chatterbox. Characters with very little dialogue
  get weak references.
- **Diarization over-splits** — the same character may occupy several speaker
  clusters (mitigated by a centroid-merge pass in `refs.py`). Over-splitting is
  safe (never the wrong voice), just shortens references.
- **Overlapping speech** (two characters at once) is not handled; lines are
  placed at their original start times and may collide slightly.
- **Very short exclamations** tend to come out longer than their slot.
- **Lip-sync is not attempted** — standard dub-style audio replacement.
