---
name: video-translator
description: Operate the video-translator pipeline — dub any video into Hebrew with per-speaker voice cloning. Use when asked to translate/dub a video, run the pipeline, fix translations, remix audio, or produce samples. Covers both the free local backend (Chatterbox MLX) and the paid ElevenLabs backend.
---

# Operating the video-translator pipeline

This repo dubs videos into Hebrew while cloning each speaker's original voice.
Everything runs locally on Apple Silicon except the optional ElevenLabs TTS backend.

## One-command runs

The runner is the single entry point and behaves as a WIZARD when interactive:
it asks every open decision, verifies the config each choice needs, and when a
key is missing it prints exactly which line of `config.env` to fill, then exits
so the user can complete it and rerun.

```bash
uv run python scripts/run_full.py "/path/to/video.mp4" --work work_myvideo
# Translation backend:
#   [1] my AI agent already wrote <work>/translations.json (best quality)
#   [2] remote frontier model (pi CLI / ANTHROPIC_API_KEY / OPENAI_API_KEY)
#   [3] local Gemma via Ollama (free, offline)
# Voices (TTS): [1] free local Chatterbox  [2] paid ElevenLabs
# Upscale final video 2x? [y/N]
```

ALL keys and tunables live in `config.env` at the repo root (auto-created from
`config.env.example` on first run, gitignored — never commit it). Runtime env vars
also work and override `config.env`; if the wizard says a key is missing, verify
both `env` and `config.env` before telling the user to paste secrets into a file.

## Work directory rule — one folder per video

Always create a dedicated work folder for each source video. Never reuse generic
folders like `work`, `work_full`, or another movie's folder, because resumable
stages will skip existing files and can silently mix transcripts, voices, and
outputs from different videos.

Use a short source-derived slug, for example:

```bash
uv run python scripts/run_full.py "/path/to/trailer.mp4" --work work_trailer_720p
```

When a user asks for both TTS backends, run the same dedicated work folder twice:
first with `--tts local`, then with `--tts elevenlabs`. `run_full.py` accepts only
one TTS backend per invocation; it does not produce both outputs in one run.

### Conversational wizard (when an agent drives for a human)

Ask the same questions conversationally, and at EVERY step present ALL
available selections — never pre-pick silently. Required option lists:

**1. Translation** (present all four):
   a. **Me (the agent)** — I write the Hebrew translation myself. Best quality,
      free, no setup. (Then follow the translation contract below.)
   b. **Remote frontier API** — `translate_remote.py`; needs ONE of:
      pi CLI authenticated · `ANTHROPIC_API_KEY` · `OPENAI_API_KEY` in config.env.
      Parallel batches, checkpointed.
   c. **Local Gemma via Ollama** — free, fully offline, decent quality
      (`ollama pull gemma4:e4b`).
   d. **NLLB-200** — fastest, most literal, lowest quality; no Ollama needed.

**2. Voices (TTS)** (present both):
   a. **Free local** — Chatterbox MLX voice cloning, unlimited, Apple GPU.
   b. **Paid ElevenLabs** — closest to original voices; `ELEVENLABS_API_KEY`
      in config.env or environment; ~40K characters per 90-min movie; 10 voice-slot limit.
      If the user chooses both, produce both outputs by running the same work dir twice.

**3. Upscale** (present both, with guidance):
   a. **Yes** — Real-ESRGAN 2x; recommend only for low-res sources (<720p);
      hours for a movie, resumable.
   b. **No** — recommend for already-HD sources.

For every choice that needs a key, first check whether the key is already present
in the process environment. If it is not, tell the user the exact `config.env`
line to fill and wait for their confirmation before proceeding.

For unattended runs pass answers as flags (no prompts):

```bash
uv run python scripts/run_full.py <video> --work work_x --translator remote --tts local --upscale no
uv run python scripts/run_full.py <video> --work work_x --translator agent --tts elevenlabs --upscale yes
```

`--translator agent` means YOU wrote `<work>/translations.json` (see the
translation contract below) before invoking the runner.

Output lands in `<work>/output/movie_he.mp4` (local) or `movie_he_11labs.mp4`
(ElevenLabs); upscaled variants get an `_upscaled` suffix. The upscaled VIDEO
track is shared between backends — if one backend's upscaled file exists, the
other reuses its video and only swaps audio (seconds instead of hours).
Every stage skips if its output exists — safe to rerun after any interruption.
Both backends must always remain supported; never remove one in favor of the other.

## Pipeline stages (what run_full.py executes, in order)

| # | Stage | Script | Env | Output in `<work>/` |
|---|---|---|---|---|
| 1 | extract audio | ffmpeg | — | `clip/audio_16k.wav`, `clip/audio_44k.wav` |
| 2 | separate vocals | `demucs -d mps` | `.venv` | `demucs/htdemucs/audio_44k/{vocals,no_vocals}.wav` |
| 3 | transcribe | `scripts/asr_mlx.py` | `.venv-mlx` | `asr.json` |
| 4 | diarize | `scripts/diarize.py` (pyannote; needs `HF_TOKEN`) | `.venv` | `segments.json` |
| 4b | gender | `scripts/gender.py` (voice classifier -> gendered Hebrew) | `.venv` | `speakers.json` |
| 5 | translate | `scripts/translate_remote.py` (provider-selectable remote; fallback `translate_ollama.py`) | `.venv` | `translations.json` |
| 6 | **clean translations** | `scripts/fix_translations.py` | `.venv` | repaired `translations.json` |
| 7 | speaker refs | `scripts/refs.py` | `.venv` | `refs/<speaker>.wav` |
| 8 | TTS | `scripts/tts_mlx.py` OR `scripts/tts_elevenlabs.py` | `.venv-mlx` / `.venv` | `tts/` or `tts_11labs/` |
| 9 | mix + mux | `scripts/mix.py --tts-dir <dir> --output <name>` | `.venv` | `output/*.mp4` |

All stage scripts read `VT_WORK` (work dir, default `work`) and `VT_VIDEO`
(source video path) from the environment when run individually.

## Diarization requirements

`scripts/diarize.py` uses pyannote's gated models: `HF_TOKEN` must be in env and
the token's account must have accepted terms for `pyannote/speaker-diarization-3.1`
and `pyannote/segmentation-3.0`. It splits whisper segments at speaker turns
(word timestamps required in asr.json), runs an ECAPA QC pass that reassigns
clear speaker mismatches, and `gender.py` then labels each speaker's gender so
translation produces correct gendered Hebrew forms. After re-diarizing an
existing work dir, run `scripts/reuse_11labs.py` before the ElevenLabs TTS to
carry over unchanged lines instead of paying to regenerate them.

## Translation: the operating agent SHOULD translate, not the local model

If you (the agent reading this) are a capable multilingual LLM, produce the
translations YOURSELF instead of running the local Gemma model — you are the
best translator available in this pipeline. Read `<work>/segments.json` and
`<work>/speakers.json`, then write `<work>/translations.json`:

```json
[{"id": 0, "start": 1.23, "end": 3.45, "speaker": "SPEAKER_07", "text_he": "..."}, ...]
```

Translation contract (same rules the scripted backends enforce):
- Natural, punchy SPOKEN Hebrew as a real Israeli dub for a kids' movie.
- Translate MEANING, never word-by-word: localize idioms and phrasing into what
  the character would actually say; literal calques are wrong answers.
- ONE translation per line - no alternatives, parentheses, notes, or explanations.
- Hebrew letters only; transliterate names/acronyms consistently across the movie.
- Numbers written as Hebrew words (TTS reads them aloud).
- Same spoken length as the English or shorter - never longer.
- Use each speaker's gender (speakers.json) for the speaker's own forms AND the
  ADDRESSEE's gender (the nearest other speaker in the scene) for all
  second-person forms - Hebrew genders both directions; keep scene
  continuity - translate consecutive lines together, never in isolation.
- Work in batches and checkpoint (write partial translations.json as you go);
  after finishing, run `scripts/fix_translations.py` as a lint, then REVIEW the
  whole script for gender agreement and grammar (`scripts/review_translations.py`,
  or do the review yourself with the same speaker/addressee gender map) - a
  one-shot translation always contains agreement slips.

Scripted backends, in quality order, for unattended runs or fallback:

1. `scripts/translate_remote.py` - remote frontier model, provider selectable:
   | provider | requirement | select with |
   |---|---|---|
   | pi CLI | `pi` installed + authenticated | `VT_TRANSLATOR=pi` (auto if present) |
   | Anthropic | `ANTHROPIC_API_KEY` env | `VT_TRANSLATOR=anthropic` (model: `VT_ANTHROPIC_MODEL`) |
   | OpenAI | `OPENAI_API_KEY` env | `VT_TRANSLATOR=openai` (model: `VT_OPENAI_MODEL`) |

   Batches run in PARALLEL (`VT_PI_PARALLEL`, default 6) - batches are
   independent because scene context comes from source lines. Checkpointed and
   resumable; per-line Gemma fallback on validation failures.
2. `scripts/translate_ollama.py` - local Gemma (free, always works offline).
3. `scripts/translate.py` - NLLB-200 (fast, literal).

## Transcript + dialogue review before translation (agent duty)

ASR can produce grammatically broken or semantically impossible English, especially
on trailers with music, cuts, whispered one-word lines, and overlapping dialogue.
Before translating, read `segments.json` as a complete dialogue and perform a
human transcript pass:

1. Fix obvious ASR mistakes in each segment's `text` when the intended sentence is
   clear from context. Do not invent missing plot content; only repair clear errors.
2. Fix dialogue splits/joins that make a sentence nonsensical when the surrounding
   segments show the intended sentence. Keep ids stable whenever possible.
3. Fix speaker attributions that are semantically impossible - e.g. a question and
   its answer carrying the same speaker, or a reply that must belong to the
   interlocutor.
4. Then translate from the corrected `segments.json`, not from the raw ASR text.

Edit `segments.json` directly: it is the single source of truth and all consumers
join by id. If TTS was already generated for changed lines, delete those segment
wavs in `tts/` and `tts_11labs/` so the next run regenerates them. Scripted runs
get embedding-based QC only; this semantic transcript/dialogue check is YOUR
value-add.

## MANDATORY: clean LLM translation output after every scripted run

The local LLM (Gemma via Ollama) occasionally emits explanations, alternatives
("X / Y"), parentheses, or mixed-script words instead of a clean Hebrew line.
These get spoken aloud by TTS if not removed.

**Always run `scripts/fix_translations.py` after any scripted translation backend**
(run_full.py does this automatically; do it manually for stage-by-stage runs):

```bash
VT_WORK=work_myvideo uv run python scripts/fix_translations.py
```

It flags lines containing Latin letters, `()`/`*`//` markup, meta-words,
runaway repetition, or absurd length, re-translates each flagged line with a
strict single-line prompt, and deletes the stale TTS wav of every repaired line
so the next TTS run regenerates exactly those. If you modify the translation
prompt, keep its strict rules: ONE translation per line, Hebrew letters only,
transliterate names/acronyms, never add alternatives or explanations.

## Fixing / remixing without full reruns

- **Remix only** (levels, timing, ducking changed): rerun stage 9 — minutes, not hours.
- **Re-dub one window** (preview a fix):
  `VT_WORK=... VT_VIDEO=... uv run python scripts/mix_window.py --start 900 --end 1020 --output preview.mp4`
- **Regenerate specific TTS segments**: delete `<work>/tts/seg_<id>.wav` and rerun the TTS script (it only generates missing files).
- **Mix knobs** (top of `scripts/mix.py`): `VOICE_RMS` (dialogue level), `DUCK_GAIN`
  (background under speech; higher = less ducking), `BED_BLEND` (original mix blended
  into the background — fixes hollow-sounding speech), `FILL_GAIN`/`FILL_THRESH`
  (original vocals passed through where nothing was dubbed — grunts, screams, overlaps),
  `MAX_TEMPO` and `MAX_SPILL_FACTOR` (lip-sync tightness).
## Upscaling (any output, including full movies)

```bash
# default: fast video model, 2x (e.g. 640x266 -> 1280x532)
uv run python scripts/upscale.py <input>.mp4 <output>.mp4

# options
uv run python scripts/upscale.py in.mp4 out.mp4 --scale 4                        # 4x with the fast model
uv run python scripts/upscale.py in.mp4 out.mp4 --model realesrgan-x4plus --scale 4  # max quality, SLOW
```

- Processes in 2-minute chunks (extract -> upscale -> encode -> delete), so disk
  stays bounded and **full movies are supported**. Resumable: finished chunks are
  kept in `.upscale_<name>/` next to the output and skipped on rerun.
- Speed on M1 Max: `realesr-animevideov3` ≈ 6 fps (≈2 min per video-minute — a
  90-min movie is an overnight job; run it in the background). `realesrgan-x4plus`
  ≈ 0.2 fps — short clips only, best for live-action detail.
- Original audio is carried over automatically (downmixed to stereo AAC).
- Requires the realesrgan binary in `tools/realesrgan/` (see README Setup step 7).
- Cheap alternative for casual viewing: real-time playback upscaling
  (IINA/mpv + Anime4K shaders) — zero preprocessing.

## ElevenLabs specifics (paid backend)

- Requires `ELEVENLABS_API_KEY` in env or `config.env` with instant-voice-cloning enabled. Never commit keys.
- Top-9 speakers (by line count) are cloned from `<work>/refs/`; voice ids are cached in
  `<work>/voices_11labs.json` — clones are created once and reused, never re-cloned.
- Minor speakers use stock voices (no voice slots consumed).
- Voice slots are limited (10 on Starter). Before cloning fails, check `/user/subscription`
  or the voices list; if slots are full, ask permission and delete stale `vt_movie_*`
  voices via the API before rerunning ElevenLabs TTS.
- Budget: ≈40K characters per 90-minute movie; check with the account quota endpoint first.

## Environments — two venvs, do not merge them

- `.venv` (uv project env): torch stack — demucs, speechbrain, NLLB, mixing.
- `.venv-mlx`: Apple-GPU stack — mlx-audio (Chatterbox TTS), mlx-whisper, dicta.
  Invoke via `.venv-mlx/bin/python`; the torch and mlx dependency trees conflict.

## Troubleshooting

- **TTS speaks weird meta-text** → step 6 was skipped; run `fix_translations.py`, then TTS, then remix.
- **Hollow/quiet background during dialogue** → raise `BED_BLEND` (Demucs strips energy into the discarded vocals stem during speech).
- **Mouths move, no sound** → those are untranscribed vocals (grunts/overlaps); handled by `vocal_fill` in mix.py — check `FILL_THRESH` if some are still silent.
- **Speech overruns lips** → lower `MAX_SPILL_FACTOR`, or shorten the Hebrew line in `translations.json`, delete that seg wav, regenerate, remix.
- **A speaker's voice sounds wrong** → their reference is weak; hand-cut a clean 15-30s clip of that character into `<work>/refs/<speaker>.wav` and regenerate their segments.
- **Ollama not running** → `ollama serve` / check `ollama list` shows `gemma4:e4b`.
