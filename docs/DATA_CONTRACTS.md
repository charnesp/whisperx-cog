# Data contracts

Shapes at system boundaries. Cog/Replicate API references: [README.md](../README.md), [AGENTS.md](../AGENTS.md).

## Prediction object (Cog / bridge / webhook)

| Field | Type | When present |
|-------|------|--------------|
| `status` | string | Always — `succeeded`, `failed`, `canceled`, `starting`, `processing` |
| `output` | object \| null | `succeeded`; null on failure |
| `error` | string | `failed` only |
| `metrics` | object | Optional — e.g. `{ "predict_time": 12.34 }` |

## Success output (`predict.py` → `Output`)

| Field | Type | Notes |
|-------|------|-------|
| `segments` | array | Utterances with `start`, `end`, `text` |
| `segments[].words` | array | Optional — alignment enabled |
| `segments[].speaker` | string | Optional — diarization enabled |
| `detected_language` | string | ISO code |
| `speaker_embeddings` | object \| null | `{ "SPEAKER_00": [float, ...] }` or null |

All floats in `output` must pass **`sanitize_for_json()`** before return (no NaN/inf).

## Bridge prediction ID

- Pattern: `^[a-zA-Z0-9_-]{1,64}$`
- Bridge generates 24-char hex id if client omits `id` on `POST /predictions`

## Prediction request input (`POST /predictions`)

Two accepted body shapes at the bridge boundary:

| Body | Content-Type | Bridge behavior |
|------|--------------|-----------------|
| JSON | `application/json` | Forwarded to Cog as-is (id + webhook injected when absent) |
| Multipart | `multipart/form-data` | Converted to Cog JSON: `file` part → `input.audio_file` (base64 data URI), other form fields merged into `input` (JSON values parsed, strings kept) |

Multipart rules: `file` extension must be in `flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, webm`; file size ≤ `OPENAI_STT_MAX_FILE_SIZE_BYTES` (default 25 MB). Cog always receives `Content-Type: application/json`.

### Error envelope (multipart only, flat convention)

```json
{"error": "missing_audio_file|unsupported_audio_format|payload_too_large|invalid_multipart_form", "detail": "..."}
```

Unit tests: `tests/test_multipart_predictions.py`

## JSON boundary rule

**Invariant:** Any value crossing the Cog HTTP / webhook boundary is processed by `json_sanitize.sanitize_for_json`. Do not return raw WhisperX tensors or un-sanitized floats from `predict()`.

Unit tests: `tests/test_json_sanitize.py`

## OpenAI STT responses (`POST /v1/audio/transcriptions`)

Bridge boundary only — not Cog/Replicate prediction objects.

### Error envelope (4xx / 5xx)

```json
{"error": {"message": "...", "type": "invalid_request_error|authentication_error|server_error", "code": null}}
```

### Success by `response_format`

| `response_format` | Content-Type | Body |
|-------------------|--------------|------|
| `json` (default) | `application/json` | `{"text": "..."}` |
| `text` | `text/plain; charset=utf-8` | plain transcribed text |
| `verbose_json` | `application/json` | OpenAI `TranscriptionVerbose` — top-level `words[]` and/or `segments[]` per `timestamp_granularities` |
| `srt` | `text/plain; charset=utf-8` | SRT subtitles |
| `vtt` | `text/vtt; charset=utf-8` | WebVTT |
| `diarized_json` | `application/json` | OpenAI `TranscriptionDiarized` — `task`, `duration`, `text`, `segments[]` with `id`, `start`, `end`, `speaker`, `text`, `type` |

`verbose_json` maps Cog `output.segments[]` to OpenAI segment fields; missing Whisper metrics use placeholders (`tokens: []`, `seek: 0`, `avg_logprob: 0.0`, etc.).

`diarized_json` groups Cog word-level `speaker` labels into OpenAI speaker turns. WhisperX `SPEAKER_00` → `A`, `SPEAKER_01` → `B`, etc.; optional `known_speaker_names[]` overrides the first N speakers. Requires `model=gpt-4o-transcribe-diarize` and whisperx `HUGGINGFACE_TOKEN` (or Cog `huggingface_access_token` input).

Unit tests: `tests/test_openai_stt.py`

## Model backends (Cog `whisper_model`)

| Cog `whisper_model` | Engine | Default `batch_size` (when omitted) | Hotwords semantics |
|---------------------|--------|-------------------------------------|--------------------|
| `tiny` / `large-v3` / `large-v3-turbo` | faster-whisper | `64` (`WHISPER_DEFAULT_BATCH`, pre-change behavior) | faster-whisper hotwords (passed to the whisper `transcribe` options) |
| `qwen3-asr` | Qwen3-ASR-1.7B (`qwen-asr` pipeline) | `resolve_qwen_batch_size()`: default `4` (`QWEN_DEFAULT_BATCH`), explicit values clamped to `1..8` (`QWEN_MAX_BATCH`), non-positive → default | **Context semantics** — hotwords are NOT passed to the ASR options: `format_qwen_context()` wraps them in the meeting-context template (`Contexte technique de la réunion. Termes, entités et noms propres attendus : <hotwords>.`) and the resulting `context` string is forwarded to the Qwen pipeline per batch. No post-filtering of the output is done. |

Qwen context rules: empty/absent hotwords → empty `context` (neutral); cap `QWEN_CONTEXT_CAP = 2000` chars applied after template assembly — truncation logs carry lengths only, never content. `ENABLE_QWEN` gates the qwen backend (see [BRIDGE.md](./BRIDGE.md) for the bridge kill-switch).

Unit tests: `tests/test_qwen_backend.py`, `tests/test_openai_stt.py`
