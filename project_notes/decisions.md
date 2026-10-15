<!-- ABOUTME: Running log of project decisions for voice-agent-forge, with dates and reasons. -->
<!-- ABOUTME: Read at session start; append new decisions at the bottom. -->
# Decisions

## 2026-10-14
- **App:** web app (FastAPI + WebSocket backend, React/Vite/Tailwind frontend) for live A/B calls and an eval dashboard.
- **Moshi injection:** timeboxed (2h) test of forcing text tokens into Moshi's inner-monologue stream, with Kokoro voice handoff as a config-selectable fallback (`moshi_injection`). Reason: Moshi has no system prompt and no function calling.
- **Models:** `kyutai/moshika-mlx-q4` + `qwen3:4b` (both tool brain and cascaded LLM). Reason: 16 GB M4; q8 + 8b would swap and corrupt latency numbers.
- **Docker runtime:** Colima (fully OSS) for SearXNG.
- **Evals:** audio-mode for both pipelines (Kokoro-synthesized caller audio); text mode is a cascaded-only cheap tier. Reason: Moshi has no text input.
- **STT:** mlx-whisper instead of faster-whisper (CPU-only on Apple Silicon).
- **Tool brain hears the caller** via VAD + Whisper; Moshi's monologue only has its own speech.
- **Remote:** https://github.com/joshua-burnell-1/Voice-Agent-Demo.git (public). Scan before every push. Private scrub terms come from the `SCRUB_TERMS` env var, never tracked files.
