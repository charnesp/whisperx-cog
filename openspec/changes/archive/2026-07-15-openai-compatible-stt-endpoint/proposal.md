## Why

whisperx-cog exposes a Replicate-compatible async API (`POST /predictions`, polling, JSON body with URL/data-URI audio). Clients built for the OpenAI Whisper API (`POST /v1/audio/transcriptions`, multipart file upload, synchronous `{text: "..."}` response) cannot use it without a custom adapter. Adding an OpenAI-compatible endpoint on the bridge enables drop-in use with openai-python, Hermes Agent, and other OpenAI STT clients.

## What Changes

- Add `POST /v1/audio/transcriptions` on the bridge (port 8080) accepting multipart/form-data audio uploads
- **Synchronous** request handling end-to-end: client waits for HTTP response; bridge calls Cog with a **single synchronous** `POST /predictions` (no `Prefer: respond-async`, no poll, no webhook/Redis on this path)
- Support five `response_format` values: `json`, `text`, `verbose_json`, `srt`, `vtt`
- Map OpenAI model names (`whisper-1`, etc.) to Cog `whisper_model` values
- OpenAI-shaped error responses (400, 401, 413, 422, 500, 504) with existing Bearer auth (`BRIDGE_AUTH_TOKEN`)
- **Audio handoff bridge → Cog:** encode upload as **base64 data URI** in JSON `input.audio_file` (Cog HTTP API contract); no shared volume required
- Validate file extensions against the **official OpenAI allowlist** (flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, webm)
- New env var: `OPENAI_STT_TIMEOUT_SECONDS` (default 300), `OPENAI_STT_MAX_FILE_SIZE_MB` (default 25)
- Unit/integration tests for endpoint behavior, auth, errors, model mapping, timeout, and Cog payload shape
- README section documenting the new endpoint and configuration
- No breaking changes to existing `/predictions` Replicate-compatible API

## Capabilities

### New Capabilities

- `openai-stt-api`: OpenAI-compatible speech-to-text endpoint on the bridge — multipart upload, model mapping, response format conversion, error schema, auth, file size/format limits, synchronous Cog call, and data-URI handoff

### Modified Capabilities

<!-- No existing openspec specs; bridge Replicate API behavior unchanged -->

## Impact

- **Code:** `bridge/bridge.py` (new route + multipart parsing), new `bridge/openai_compat.py` (multipart → data URI → Cog → OpenAI conversion), `README.md`, `docs/BRIDGE.md` / `docs/DATA_CONTRACTS.md` as needed
- **Dependencies:** stdlib multipart parsing preferred; no new infrastructure volumes
- **Infrastructure:** No k8s/compose volume changes required for protocol correctness
- **Tests:** New pytest module (e.g. `tests/test_openai_stt.py`) using httpx/urllib with mocked Cog
- **Dual-copy invariant:** After editing `bridge/bridge.py`, run bridge ↔ k8s sync and harness smoke/check
