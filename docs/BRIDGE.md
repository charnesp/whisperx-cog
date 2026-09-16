# Bridge

HTTP bridge for Replicate-compatible self-hosted deployment. Source of truth: **`bridge/bridge.py`** and **`bridge/openai_compat.py`**.

Full API usage: [README.md](../README.md). Architecture: [ARCHITECTURE.md](./ARCHITECTURE.md).

## Dual copies (keep in sync)

| Deployment | How loaded |
|------------|------------|
| Docker Compose | `ghcr.io/charnesp/whisperx-cog-bridge:latest` (GHCR) |
| Kubernetes | same GHCR image in `k8s/whisperx-stack.yaml` |

```bash
python3 scripts/check-bridge-sync.py    # before commit (also: make smoke)
# after editing bridge/*.py: push to main → GHCR rebuild → kubectl rollout restart
```

## Environment variables

| Container var | k8s secret / Compose | Default | Purpose |
|---------------|----------------------|---------|---------|
| `BRIDGE_AUTH_TOKEN` | `BRIDGE_TOKEN` | — | `Authorization: Bearer …` |
| `WEBHOOK_SECRET` | `WEBHOOK_SECRET` | — | Internal webhook path segment |
| `OPENAI_STT_TIMEOUT_SECONDS` | env / `.env` | `300` | Sync Cog POST read timeout (OpenAI STT) |
| `OPENAI_STT_MAX_FILE_SIZE_MB` | env / `.env` | `25` | Max upload size (OpenAI STT) |
| `HUGGINGFACE_TOKEN` | `HUGGINGFACE_TOKEN` | — | Passed to whisperx container only |
| `ENABLE_QWEN` | env / `.env` | unset = **enabled** | Kill-switch for the `qwen3-asr` backend (see [Enable gate](#enable-gate-enable_qwen) below) |

Loopback addresses (`127.0.0.1:6379`, `127.0.0.1:5000`, `localhost:8080` webhook) are fixed because Cog, Redis, and bridge share one network namespace.

## Logging

| Prefix | Scope |
|--------|--------|
| `[bridge ext]` | External clients — API requests, cache hit/miss |
| `[bridge int]` | In-pod — Redis, Cog proxy, Cog→bridge webhook |
| `[bridge]` | Process boot only |

`GET /health` and `GET /health-check` are silent. Loopback peers log as `[bridge int]`.

**Kubernetes:** `kubectl logs <pod> -c bridge`  
**Compose:** `docker compose logs bridge`

## Webhook → Redis

- URL: `http://localhost:8080/<WEBHOOK_SECRET>/webhook?id=<prediction_id>`
- RESP2 Redis client (not `redis-py`); max payload **200 MiB**; TTL **24 h**
- Requires **`Content-Length`** (chunked → **501**)

### Webhook HTTP errors

| Status | Code | Notes |
|--------|------|-------|
| 400 | `invalid_content_length` | Malformed or negative Content-Length |
| 400 | `invalid_prediction_id` | `id` not matching `^[a-zA-Z0-9_-]{1,64}$` |
| 413 | `payload_too_large` | Body > max; `Connection: close`, body drained |
| 501 | `unsupported_transfer_encoding` | Chunked not supported; best-effort drain |
| 503 | `redis_set_failed` | Redis error; JSON includes `detail`; Cog may retry |

## GET /predictions/<id>

- `id` must match `^[a-zA-Z0-9_-]{1,64}$` else **400**
- Redis hit → stored JSON immediately
- Miss or Redis read failure → proxy to Cog (logged distinctly)

## POST /predictions

- No client `webhook` → inject internal webhook + `webhook_events_filter: ["start", "completed"]`, store in Redis
- Client provides webhook → forward unchanged, **no Redis storage**

## POST /v1/audio/transcriptions (OpenAI STT)

OpenAI-compatible speech-to-text endpoint. Implemented in `bridge/openai_compat.py`, routed from `bridge/bridge.py`.

| Aspect | Behavior |
|--------|----------|
| Protocol | OpenAI `multipart/form-data` (`file`, `model`, optional `language`, `response_format`, …) |
| Semantics | **Synchronous** — blocks until Cog `predict()` completes |
| Cog call | Single `POST /predictions` JSON, **no** `Prefer: respond-async`, **no** poll, **no** Redis |
| Audio handoff | Base64 **data URI** in `input.audio_file` (Cog HTTP API contract — not bare paths, not multipart to Cog) |
| Auth errors | HTTP 401 with OpenAI error shape (`invalid api key` / `authentication_error`) |
| Logging | `[bridge ext]` — path, bytes, model, format, status, duration (no payload logging) |

Fixed Cog input on this path: `align_output: true`. Non-diarize requests set `diarization: false`; diarize requests (`model=gpt-4o-transcribe-diarize` or `response_format=diarized_json`) set `diarization: true` with `huggingface_access_token: null` — the HF token is resolved inside the **whisperx** container from `HUGGINGFACE_TOKEN` (see `hf_token.py` / `predict.py`). The bridge does **not** need `HUGGINGFACE_TOKEN` in its own env.

**Diarization path** (`gpt-4o-transcribe-diarize`):

| Parameter | Behavior |
|-----------|----------|
| `response_format=diarized_json` | Returns OpenAI `TranscriptionDiarized` with speaker segments |
| `known_speaker_names[]` | Optional — maps first N WhisperX speakers to provided names |
| `chunking_strategy` | Accepted (no-op); WhisperX VAD handles long audio |
| `prompt`, `timestamp_granularities`, `include[]=logprobs` | HTTP 400 |
| `known_speaker_references[]` | HTTP 400 — not implemented (see [PLANS.md](../PLANS.md) tech debt) |

### Enable gate (ENABLE_QWEN)

`model=qwen3-asr` is **gated at the bridge**: when the backend is disabled, the bridge refuses the request with HTTP 400 (`invalid_request_error`, message `qwen3-asr backend is disabled (ENABLE_QWEN)`) instead of forwarding a request that would fail with a Cog-side 500. `predict.py` keeps its own `qwen_enabled()` gate as defense in depth.

| `ENABLE_QWEN` (bridge env, read at request time — toggle without redeploy) | Behavior for `model=qwen3-asr` |
|---|---|
| unset (default) | **Enabled** — same default as `predict.qwen_enabled()` |
| `1` / `true` / `yes` / `on` (case-insensitive) | Enabled |
| anything else (`0`, `false`, `empty`, …) | HTTP 400 — qwen refused; whisper models unaffected |

### Qwen passthrough rules (`model=qwen3-asr`)

- **`hotwords`**: forwarded to Cog **only** on the qwen path (`QWEN_MODELS`); whisper models always send `hotwords: null` (whisper semantics unchanged). Hotwords become the Qwen transcription **context** — see [DATA_CONTRACTS.md](./DATA_CONTRACTS.md).
- **`batch_size` passthrough rule**: `batch_size` is included in the Cog input **only when the client provides it** (no hard-coded default in the bridge); the Cog predictor applies the per-model default (64 faster-whisper / 4 qwen, qwen values clamped 1-8). A non-integer value → HTTP 400 `batch_size must be an integer`.
- **no-hotwords-in-logs**: hotword content is never logged. The `[bridge ext]` request log carries `path, bytes, model, format, status, duration` only (verified in `bridge/bridge.py` `handle_openai_transcription`); inside `openai_compat.py` hotwords are read and passed through but never appear in a log call, and Qwen context truncation logs lengths only.

## Probes

- Malformed `Content-Length` on silent `GET /health-check` → treated as empty body (forwards to Cog)
- Error responses that leave socket ambiguous send **`Connection: close`** (400, 413, 501, 503 on webhook path)
