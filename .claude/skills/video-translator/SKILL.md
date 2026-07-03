---
name: video-translator
description: Operate the video-translator pipeline — dub any video into Hebrew with per-speaker voice cloning. Use when asked to translate/dub a video, run the pipeline, fix translations, remix audio, upscale, or produce samples. Covers both the free local backend (Chatterbox MLX) and the paid ElevenLabs backend.
---

# video-translator skill

The full operating manual lives at [`skill/SKILL.md`](../../../skill/SKILL.md)
in the repo root — read that file and follow it. It is agent-agnostic and is
the single source of truth for: pipeline stages and env contracts, free vs
paid TTS backends, the mandatory Gemma-output cleanup step, remix/repair
recipes, upscaling (including full movies), ElevenLabs slot/quota management,
and troubleshooting.
