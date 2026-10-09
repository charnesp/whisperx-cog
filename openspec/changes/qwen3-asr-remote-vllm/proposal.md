## Why

The `qwen3-asr` backend in cog loads ~5 GB VRAM per run (full model load/unload cycle, ~4.4 s warm load) while a shared vLLM server can serve the same model on demand behind the model gateway (the GPU front door already in production for the local LLM engine/embed/retrieval/paddle), at the URL injected into cog via environment. Spike validated 2026-10-09 (throwaway vLLM 0.30.0 container, weights bind-mounted from the provisioned models.lock snapshot): multimodal `/v1/chat/completions` (audio_url base64) = 200 in 0.43 s for 10 s of audio, quality identical (same HF weights), sleep/wake_up cycle operational (VRAM 7.0 GB awake -> 1.7 GB asleep, wake-then-transcribe 0.92 s). Switching removes the static qwen VRAM from cog while keeping word alignment (Qwen ForcedAligner) and diarization (pyannote) local and unchanged: those stages depend only on the local VAD timestamps, not on the transcription engine.

## What Changes

- New remote mode for the qwen3-asr backend: transcription windows are delegated over HTTP to the vLLM server behind the model gateway (base URL and model name provided by env), the local path remains fully available (complete backward compatibility)
- The remote base URL (`QWEN_REMOTE_BASE_URL`) and model name (`QWEN_REMOTE_MODEL`) are configurable ONLY via environment: no address, port or model name hard-coded anywhere in the repository (code, tests, docs, compose, k8s); remote misconfiguration = fail-fast typed error at predictor boot
- Backend selector env: `QWEN_BACKEND=remote|local`, default `local` (zero behavioral change at deploy; flippable by env); kill-switch `ENABLE_QWEN` unchanged and effective in both modes
- Output invariant: segments produced by the remote path carry their timestamps from the LOCAL VAD chunking (start/end untouched); the ForcedAligner alignment and pyannote diarization stages downstream are unchanged
- Remote failures are explicit typed errors (`QwenRemoteError:` with a category; HTTP **500 `server_error`** on the OpenAI STT surface, **502** on the `/predictions` proxy) — never a silent fallback to the local engine
- Infra (out of repo): `the remote ASR engine` service on the existing GPU stack + `qwen3-asr` entry in the model gateway (cohabiting with retrieval, never cohabiting with the local LLM engine)

## Non-goals

- Streaming (still a non-goal of the qwen3-asr-backend change)
- Server-side diarized_json / timestamps (verbose_json unsupported for qwen3-asr on vLLM 0.30.0 — `supports_segment_timestamp=False`)
- Changing the default model for existing clients (`whisper-1` -> `BRIDGE_DEFAULT_MODEL` unchanged)
- ANY change to the faster-whisper path (tiny/large-v3/turbo bit-identical, regression acceptance criterion)
- Removing the qwen weights from the provisioned `/models` mount (they stay for the local mode and rollback)
- Log masking of the remote URL at runtime (rejected 2026-10-09: unnecessary and constraining) — the ONLY protection requirement is that no personal IP or port appears anywhere in the repository

## Capabilities

### Modified Capabilities

- `openai-stt-api`: the qwen3-asr backend gains a configurable remote mode (env-selected, env-addressed, explicit failures); the client-facing HTTP contract is unchanged

## Impact

- **Code:** `predict.py` (qwen branch: backend selection, injectable remote HTTP client, response parsing), new GPU-free module for remote helpers; `docs/` env references
- **Bridge:** `bridge/openai_compat.py` — NO change expected (the /predictions contract is untouched; verified by test)
- **Compose/k8s:** new env entries (`QWEN_BACKEND`, `QWEN_REMOTE_BASE_URL`, `QWEN_REMOTE_TIMEOUT_S`) in `docker-compose.yml` + `k8s/whisperx-stack.yaml` (bridge sync rule)
- **Docs:** `README.md`, `docs/ARCHITECTURE.md`, `docs/DATA_CONTRACTS.md`, `docs/BRIDGE.md`, `docs/OBSERVABILITY.md` (new log prefix `qwen-remote:`), `PLANS.md`
- **Infra (out of repo):** an ASR engine service on the deployment's GPU stack plus a gateway entry; the URL is injected into cog through deploy env
- **Out of scope:** streaming, server-side timestamps, default-model change, faster-whisper path changes

## Risks

- Response parsing: the multimodal reply carries a `language X<asr_text>...` prefix — parser tested against the real format (spike payload captured) and structured-output cases
- Cold start: first call after inactivity = the model gateway wake (measured: a cold boot of tens of seconds and a much shorter sleep/wake cycle) — per-request timeout default 300 s absorbs it; zero-retry posture documented (a mid-swap 5xx surfaces explicitly)
- VRAM: the remote engine's footprint is a fraction of the deployment's budget; it is scheduled to share the gateway with the retrieval models and never with the local LLM engine, exclusivity being carried by the gateway
- Remote unavailable: explicit typed `QwenRemoteError:` message (categories config/connection/timeout/http_status/parse), prediction failure at the bridge (500 `server_error` on the OpenAI STT surface / 502 on the `/predictions` proxy), no fallback and no retry at v1 (decided)
- Concurrency: VAD windows fly with a bounded pool reusing the clamped batch_size (4 default / 8 cap); VAD order preserved in the fused result
