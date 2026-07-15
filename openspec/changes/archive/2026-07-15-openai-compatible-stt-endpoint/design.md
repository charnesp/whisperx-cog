## Context

whisperx-cog runs three co-located containers (Cog/whisperx :5000, Redis, bridge :8080) in one network namespace. The bridge (`bridge/bridge.py`) is a stdlib `http.server` proxy implementing the Replicate-compatible async API: `POST /predictions` → Cog, webhook → Redis, `GET /predictions/<id>` for polling.

Clients expecting OpenAI Whisper (`POST /v1/audio/transcriptions`, multipart upload, synchronous `{text}`) cannot use this API without an adapter. The bridge is the correct insertion point: it is already the external HTTP entrypoint and owns auth, logging, and Cog orchestration.

**Two distinct protocol layers:**

| Layer | Protocol | Audio transport |
|-------|----------|-----------------|
| External (client → bridge) | OpenAI STT | `multipart/form-data`, field `file` |
| Internal (bridge → Cog) | Cog HTTP API (JSON only) | `input.audio_file` as **HTTPS URL** or **data URI** |

Cog's HTTP API does **not** accept multipart uploads or bare filesystem paths in JSON. Per [Cog deploy docs](https://github.com/replicate/cog/blob/main/docs/deploy.md), `Path` inputs are passed as URI strings; Cog downloads HTTP(S) URLs or decodes data URIs before `predict.py` receives a local `Path`.

## Goals / Non-Goals

**Goals:**

- Expose `POST /v1/audio/transcriptions` on the bridge with **100% OpenAI STT protocol compatibility** at the external boundary (request shape, sync semantics, response formats, error schema, extension allowlist)
- Support `response_format`: `json`, `text`, `verbose_json`, `srt`, `vtt`
- Map OpenAI model names to Cog `whisper_model` values
- Reuse existing Bearer auth (`BRIDGE_AUTH_TOKEN` / client `BRIDGE_TOKEN`)
- Hand off uploaded audio to Cog via **base64 data URI** in synchronous JSON `POST /predictions`
- Block on single synchronous Cog HTTP call within configurable timeout
- Honor `timestamp_granularities` per OpenAI rules when `response_format=verbose_json`
- Add pytest coverage for happy paths, errors, auth, mapping, timeout, Cog payload
- Document in README
- Preserve existing `/predictions` behavior unchanged

**Non-Goals:**

- Streaming transcription (`stream=true`)
- `/v1/audio/translations`
- Newer OpenAI models (`gpt-4o-transcribe`, `diarized_json`) — out of scope for v1
- Diarization via OpenAI endpoint (Cog `diarization: false` fixed)
- Webhooks or Redis on OpenAI endpoint path
- Shared volume / filesystem path handoff (not Cog API-correct)
- Changing Cog predictor pipeline or output schema

## Decisions

### 1. Module layout

| Module | Responsibility |
|--------|----------------|
| `bridge/bridge.py` | Route dispatch for `POST /v1/audio/transcriptions`; auth; delegate to handler |
| `bridge/openai_compat.py` | Multipart parse, validation, data-URI encode, sync Cog call, response conversion, errors |

**Rationale:** Keep `bridge.py` as the dual-copy source of truth with minimal diff; isolate OpenAI logic in a testable module importable from tests without starting the HTTP server.

**Alternative considered:** Separate FastAPI app — rejected to avoid new framework and duplicate server.

### 2. HTTP stack — stay on stdlib

Parse client multipart with `cgi.FieldStorage` (stdlib) or a minimal manual parser. Avoid adding FastAPI/Flask.

**Rationale:** Bridge is intentionally dependency-free (python:3.9-slim, no requirements.txt today). Stdlib multipart parsing is sufficient for the OpenAI ingress endpoint.

**Alternative:** `python-multipart` — only if stdlib parsing proves brittle in tests.

### 3. Audio handoff — data URI (primary and only path for v1)

After receiving the client's multipart `file`:

1. Read bytes into memory (max `OPENAI_STT_MAX_FILE_SIZE_MB`, default 25 MB)
2. Guess MIME from extension (`mimetypes.guess_type`)
3. Base64-encode and build `data:<mime>;base64,<payload>`
4. Pass in Cog JSON: `"audio_file": "<data uri>"`

Cog decodes the data URI to a temp file and passes a local `Path` to `predict.py`.

**Rationale:** Matches Cog HTTP API contract exactly. No shared volume, no bare paths, no multipart to Cog. Same mechanism as `cog predict -i audio_file=@local.wav` (CLI encodes to data URI internally).

**Rejected alternatives:**
- Shared volume + absolute path — not valid Cog HTTP input; confusing API layering
- `file://` URL — undocumented in Cog; not part of API contract
- POST multipart directly to Cog — Cog HTTP API is JSON-only

**Trade-off:** ~33 MB JSON body for 25 MB audio (base64 overhead) on loopback. Acceptable in co-located pod; monitor memory if needed later.

### 4. Cog prediction flow — synchronous, no poll

```
Bridge                          Cog :5000
  │ POST /predictions (JSON, no Prefer: respond-async)
  │ { "input": { "audio_file": "data:audio/ogg;base64,...", ... } }
  │ ─────────────────────────────────────────────────────────────►
  │                          (blocks until predict() completes)
  │ ◄─────────────────────────────────────────────────────────────
  │ HTTP 200 { "status": "succeeded", "output": {...} }
```

- **Do not** send `Prefer: respond-async`
- **Do not** poll `GET /predictions/{id}`
- **Do not** use webhook/Redis on this code path
- Apply `OPENAI_STT_TIMEOUT_SECONDS` as HTTP read timeout on the Cog POST

Fixed Cog input (not exposed to OpenAI client):

```python
{
  "audio_file": "data:audio/ogg;base64,...",
  "whisper_model": "<mapped>",
  "language": "<lang or null>",
  "initial_prompt": "<prompt or null>",
  "temperature": <float, default 0.0>,
  "align_output": true,
  "diarization": false,
  "batch_size": 64,
  "vad_onset": 0.500,
  "vad_offset": 0.363,
  "language_detection_min_prob": 0,
  "language_detection_max_tries": 5,
  "hotwords": null,
  "huggingface_access_token": null,
  "debug": false,
}
```

**Rationale:** OpenAI external API is synchronous; Cog supports synchronous `POST /predictions` natively ([cog.run/http](https://cog.run/http/)). Simplest correct implementation.

### 5. Model mapping

```python
MODEL_MAP = {
    "whisper-1": "large-v3-turbo",
    "large-v3": "large-v3",
    "large-v3-turbo": "large-v3-turbo",
    "tiny": "tiny",
}
```

Reject unknown models with 400 OpenAI error. Only `whisper-1` is required for OpenAI SDK compat; others are whisperx-cog extensions using OpenAI-shaped API.

### 6. Audio format allowlist — OpenAI official

Per [OpenAI Create transcription](https://platform.openai.com/docs/api-reference/audio/createTranscription):

```
flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, webm
```

Validate by file extension (case-insensitive). Reject others with HTTP 422. Do **not** use ffmpeg's full decode list — protocol compat takes precedence over ffmpeg permissiveness.

Note: `aac` is **not** in OpenAI's list (removed from earlier contract draft).

### 7. Response conversion (`openai_compat.py`)

| Format | Content-Type | Source |
|--------|--------------|--------|
| `json` | `application/json` | `{"text": join(segments.text)}` |
| `text` | `text/plain; charset=utf-8` | plain joined text |
| `verbose_json` | `application/json` | OpenAI `TranscriptionVerbose` shape — see §7 below |
| `srt` | `text/plain; charset=utf-8` | Standard SRT from segment timestamps |
| `vtt` | `text/vtt; charset=utf-8` | WebVTT header + cues |

### 7. `verbose_json` and `timestamp_granularities`

Per [OpenAI Create transcription](https://platform.openai.com/docs/api-reference/audio/createTranscription):

- `timestamp_granularities` only applies when `response_format=verbose_json`
- Allowed values: `word`, `segment` (either or both)
- Multipart field name from clients: `timestamp_granularities[]` (openai-python sends a list)

**OpenAI response layout (not nested words-in-segments):**

| Granularity requested | Response fields |
|----------------------|-----------------|
| `segment` (or default none) | Top-level `task`, `language`, `duration`, `text`, `segments[]` |
| `word` | Top-level `words[]` with `{word, start, end}` — **not** inside segments |
| both | Both `segments[]` and top-level `words[]` |

**Segment fields** (from OpenAI `TranscriptionSegment`):

| Field | Source |
|-------|--------|
| `id` | Sequential index (0, 1, …) |
| `seek` | `0` (WhisperX does not expose byte seek) |
| `start`, `end`, `text` | Cog `output.segments[]` |
| `tokens` | `[]` (WhisperX does not expose Whisper token IDs) |
| `temperature` | Request `temperature` param |
| `avg_logprob` | `0.0` placeholder (not in Cog output) |
| `compression_ratio` | `1.0` placeholder |
| `no_speech_prob` | `0.0` placeholder |

**Word fields** (from OpenAI `TranscriptionWord`): map Cog `segments[].words[]` → flat top-level `words[]` with `{word, start, end}` only — drop Cog `score` (not in OpenAI schema).

**When `response_format` ≠ `verbose_json`:** accept `timestamp_granularities` without error; no effect on response.

### 8. Multipart parsing (stdlib)

openai-python sends standard `multipart/form-data`:
- File field name: **`file`**
- Other fields: `model`, `language`, `prompt`, `temperature`, `response_format`
- Arrays: `timestamp_granularities[]` repeated parts (e.g. `word`, `segment`)

**Decision:** parse with stdlib `cgi.FieldStorage` (via WSGI-style environ dict). Handle both `timestamp_granularities` and `timestamp_granularities[]` keys. No `python-multipart` dependency unless tests fail on real SDK payloads.

### 9. openai-python SDK contract (verified)

| Expectation | Value |
|-------------|-------|
| Endpoint | `POST {base_url}/audio/transcriptions` → `/v1/audio/transcriptions` |
| Auth | `Authorization: Bearer <api_key>` |
| Request | `multipart/form-data`, field `file` = binary |
| Default response | Sync HTTP 200, JSON `{"text": "..."}` |
| Client usage | `result.text` on `Transcription` object |

### 10. OpenAI error schema

All errors return JSON:

```json
{"error": {"message": "...", "type": "invalid_request_error|authentication_error|server_error", "code": null}}
```

Map bridge auth failure on this route to `401` + `"invalid api key"` / `authentication_error` (OpenAI convention), distinct from generic `send_error(401, "Unauthorized")` on Replicate routes.

### 11. Configuration

| Env var | Default | Purpose |
|---------|---------|---------|
| `OPENAI_STT_TIMEOUT_SECONDS` | `300` | HTTP read timeout on sync Cog POST → 504 |
| `OPENAI_STT_MAX_FILE_SIZE_MB` | `25` | Max upload size → 413 |

### 12. Logging

Use existing `[bridge ext]` prefix for OpenAI endpoint requests (path, client, bytes, model, format, outcome, duration). Do **not** log data URI payloads.

### 13. Testing strategy

New `tests/test_openai_stt.py`:

- Unit tests: model map, extension allowlist, data-URI builder, response converters
- Integration tests: mock Cog sync POST; verify JSON body contains data URI in `audio_file`
- HTTP-level tests via handler with mocked Cog
- Verify openai-python SDK contract (field `file`, sync response)
- No GPU required — fits `make check` harness

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| 25 MB file → ~33 MB base64 JSON on loopback | Co-located pod; OpenAI max is 25 MB; monitor memory |
| Long sync requests block bridge thread | Acceptable for v1 (`ThreadingMixIn`); timeout env var |
| Stdlib multipart parsing edge cases | Test boundaries; fall back to `python-multipart` only if needed |
| OpenAI SDK field/Content-Type expectations | Test with openai-python snippet from spec |
| Cog sync POST blocks until GPU work done | Same as OpenAI semantics; timeout → 504 |

## Migration Plan

1. Implement `openai_compat.py` + bridge route + tests locally
2. Run `python3 scripts/sync-bridge-to-k8s.py` after `bridge.py` changes
3. Deploy updated bridge ConfigMap — no volume or Cog image changes required
4. Rollback: revert bridge code + ConfigMap

## Resolved Questions

| Question | Resolution |
|----------|------------|
| Cog `audio_file` input format | **Data URI** (or HTTPS URL). Not bare paths. Per Cog HTTP API docs. |
| Supported audio extensions | **OpenAI official list:** flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, webm |
| Sync vs async Cog POST | **Sync POST** without `Prefer: respond-async`. No poll. Matches OpenAI external sync semantics. |
| Shared volume needed? | **No** for v1. Data URI is API-correct. |
| Multipart to Cog? | **No.** Cog HTTP is JSON-only. |
| Multipart parser | **stdlib `cgi.FieldStorage`**; handle `timestamp_granularities[]` |
| openai-python SDK | Field `file`, sync `{text}`, Bearer auth — verified against SDK source |
| `verbose_json` words placement | **Top-level `words[]`**, not nested in segments (OpenAI spec) |
| Missing Whisper segment metrics | Placeholders: `tokens=[]`, `seek=0`, `avg_logprob=0.0`, `compression_ratio=1.0`, `no_speech_prob=0.0` |

## Open Questions

_All previously open questions are resolved (see Resolved Questions above and §7–§9). No blockers remain before implementation._
