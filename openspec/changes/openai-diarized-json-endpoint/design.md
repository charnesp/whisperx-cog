## Context

The bridge exposes `POST /v1/audio/transcriptions` via `bridge/openai_compat.py`, mapping OpenAI multipart requests to synchronous Cog `POST /predictions` with fixed `diarization: false`. WhisperX (`predict.py`) already supports diarization when `diarization: true` and a HuggingFace token are provided — output includes `segments[].speaker` and `segments[].words[].speaker` with labels like `SPEAKER_00`.

OpenAI's diarization API (`gpt-4o-transcribe-diarize`, `response_format=diarized_json`) returns `TranscriptionDiarized`: top-level `text`, `duration`, `task: "transcribe"`, and `segments[]` where each item has `id` (string UUID-like), `speaker` (`A`, `B`, … or `known_speaker_names` values), `start`, `end`, `text`, and `type: "transcript.text.segment"`.

`HUGGINGFACE_TOKEN` is already mounted on the **whisperx** container (k8s `hf-secret`, compose `whisperx.environment`). However, `predict.py` today only uses the Cog **input** field `huggingface_access_token` — it does not fall back to `os.environ`. The OpenAI bridge path sends `huggingface_access_token: null`, so diarization is silently skipped even when the env var is set.

## Goals / Non-Goals

**Goals:**

- Support `response_format=diarized_json` with OpenAI-shaped `TranscriptionDiarized` JSON
- Support `model=gpt-4o-transcribe-diarize` as the OpenAI SDK entry point for diarized transcription
- Enable Cog `diarization: true` on the diarize path; keep `align_output: true`
- Map WhisperX speaker IDs to OpenAI letter labels (`SPEAKER_00` → `A`, `SPEAKER_01` → `B`, …)
- Honor `known_speaker_names[]` when provided (map first N WhisperX speakers to those names in order of first appearance)
- Accept `chunking_strategy` for SDK compatibility (validate allowed values; no-op — WhisperX VAD handles segmentation)
- Reject diarize-incompatible params (`prompt`, `timestamp_granularities`, `include[]=logprobs`) with 400
- Fail clearly when diarization is requested but no HF token is available (input or whisperx env)
- Unit tests with mocked Cog output containing speaker labels

**Non-Goals:**

- SSE streaming (`stream=true`) for diarized responses
- `known_speaker_references[]` speaker matching from reference audio clips
- `gpt-4o-transcribe` / `gpt-4o-mini-transcribe` (non-diarize) models
- `usage` / `logprobs` fields in diarized responses
- Changing `predict.py` pipeline or Cog output schema
- Diarization via `whisper-1` + `verbose_json` (OpenAI uses a separate model for diarize)

## Decisions

### 1. Triggering diarization

Diarization is enabled when **either**:

- `model=gpt-4o-transcribe-diarize`, or
- `response_format=diarized_json`

If both are present they must be consistent: `diarized_json` with a non-diarize model (`whisper-1`, etc.) → 400. `gpt-4o-transcribe-diarize` with `response_format` other than `json`, `text`, or `diarized_json` → 400 per OpenAI rules.

**Rationale:** Matches OpenAI SDK usage (`model=gpt-4o-transcribe-diarize`, `response_format=diarized_json`) while keeping `whisper-1` behavior unchanged.

### 2. Model mapping for diarize

```python
DIARIZE_MODEL = "gpt-4o-transcribe-diarize"

MODEL_MAP = {
    "whisper-1": "large-v3-turbo",
    "gpt-4o-transcribe-diarize": "large-v3-turbo",  # same backend, diarize flag set
    ...
}
```

**Rationale:** OpenAI clients expect the diarize model id; we alias to our best Whisper backend. No new Cog model weights required.

### 3. HuggingFace token — whisperx env fallback in `predict.py`

When `diarization: true` and `huggingface_access_token` is null/empty in Cog input, `predict.py` SHALL fall back to `os.environ.get("HUGGINGFACE_TOKEN")` (already set on the whisperx container).

If still missing after fallback → raise `RuntimeError` with a clear message (Cog `status: failed` → bridge HTTP 500). **Do not** silently skip diarization on the OpenAI path.

The bridge does **not** need `HUGGINGFACE_TOKEN` in its own env. It only sets `diarization: true` and leaves `huggingface_access_token: null` (Replicate clients can still pass the token explicitly in JSON input).

**Alternative considered:** Duplicate `HUGGINGFACE_TOKEN` on bridge and forward in Cog JSON — rejected; unnecessary secret duplication when whisperx already has the var.

### 4. Cog input on diarize path

```python
{
  "audio_file": "data:audio/...;base64,...",
  "whisper_model": "large-v3-turbo",
  "language": "<lang or null>",
  "temperature": <float>,
  "align_output": true,
  "diarization": true,
  "huggingface_access_token": null,
  "min_speakers": null,
  "max_speakers": null,
  # prompt omitted — not supported on diarize path
  ...
}
```

`initial_prompt` is **not** forwarded when diarization is enabled (OpenAI restriction).

### 5. Speaker segment conversion (WhisperX → OpenAI)

Build diarized segments by **speaker turns at word granularity**:

1. Flatten all `segments[].words[]` in order (alignment required — `align_output: true`)
2. Group consecutive words with the same `speaker` into one OpenAI segment
3. If a segment has `speaker` but no words, use segment-level `speaker` + `text` as a single segment
4. Assign `id` as stringified index (`"0"`, `"1"`, …) — OpenAI examples use opaque strings; sequential ids are sufficient for clients
5. Map speaker labels:
   - Default: order of first appearance → `A`, `B`, `C`, …
   - With `known_speaker_names[]`: first appearance order maps to `names[0]`, `names[1]`, … up to 4; overflow speakers get letter labels

**Alternative considered:** One OpenAI segment per WhisperX utterance segment — rejected because utterances can contain speaker changes mid-segment; word-level grouping matches OpenAI diarize semantics.

### 6. `convert_to_diarized_json` response shape

```json
{
  "task": "transcribe",
  "duration": <float, max segment end>,
  "text": "<joined segment texts>",
  "segments": [
    {
      "id": "0",
      "start": 0.0,
      "end": 2.5,
      "speaker": "A",
      "text": "Hello world",
      "type": "transcript.text.segment"
    }
  ]
}
```

Omit `usage` (optional in OpenAI spec). Content-Type: `application/json`.

For `response_format=json` or `text` with `model=gpt-4o-transcribe-diarize`, return plain `{"text": "..."}` or raw text (no speaker segments) — matches OpenAI behavior when diarized_json is not requested.

### 7. Parameter validation on diarize path

| Parameter | Diarize behavior |
|-----------|------------------|
| `chunking_strategy` | Accept `"auto"` or JSON `server_vad` object; ignore internally. If audio > 30s and missing, accept anyway (WhisperX handles long audio) — log warning only |
| `known_speaker_names[]` | Optional; up to 4 names |
| `known_speaker_references[]` | Reject with 400 if provided (not implemented) |
| `prompt` | 400 if non-empty |
| `timestamp_granularities` | 400 if any value sent |
| `include[]` | 400 if contains `logprobs` |
| `stream` | Ignore (same as whisper-1; no SSE in v1) |

### 8. Module changes

| File | Change |
|------|--------|
| `bridge/openai_compat.py` | `is_diarize_request()`, extended validation, `build_cog_input()` branch, `convert_to_diarized_json()`, speaker mapping helpers |
| `bridge/bridge.py` | No route change; optional log field `diarize=true` |
| `predict.py` | Env fallback `HUGGINGFACE_TOKEN` when input token null; fail hard if diarize requested but no token |
| `tests/test_openai_stt.py` | Diarized JSON conversion, validation errors, Cog payload with `diarization: true`, 503 without token |
| `docs/DATA_CONTRACTS.md` | Document `diarized_json` response shape |

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| Speaker label mapping differs from OpenAI gpt-4o-diarize model | Document that labels are WhisperX-derived; letter mapping is deterministic |
| HF token only on whisperx, not visible to bridge pre-check | `predict.py` fails hard; bridge maps Cog error to 500 |
| Diarize requests slower / more GPU memory | Existing behavior on Replicate path; `OPENAI_STT_TIMEOUT_SECONDS` already configurable |
| Word-less segments after diarize skip | Fall back to segment-level speaker; if no speakers at all, return single segment with speaker `A` and full text |
| `known_speaker_references` clients expect matching | Explicit 400 with clear message; document in README |

## Migration Plan

1. Implement `openai_compat.py` changes + tests locally
2. Deploy updated Cog image if `predict.py` changed; push bridge changes → GHCR rebuild
3. `kubectl rollout restart` / compose up
4. Verify with: `curl -F model=gpt-4o-transcribe-diarize -F response_format=diarized_json -F file=@sample.wav`
5. Rollback: revert bridge image + remove HF env from bridge if needed; non-diarize path unchanged

## Open Questions

_None blocking v1 implementation._

### TODO (follow-up) — `known_speaker_references[]`

OpenAI's `gpt-4o-transcribe-diarize` API accepts optional `known_speaker_references[]` (audio data URLs, 2–10 s each) alongside `known_speaker_names[]` to map diarized segments onto named speakers from reference clips.

**v1 behavior:** reject with HTTP 400 if the client sends `known_speaker_references[]` (or `known_speaker_references`).

**TODO for a later change:**

1. Parse multipart `known_speaker_references[]` (data URL or file parts per OpenAI SDK)
2. Determine whether WhisperX/pyannote can consume reference embeddings or clips (no API today)
3. If feasible: pass references into Cog `predict.py` and map output speakers to `known_speaker_names[]` instead of letter labels
4. Update `openai-stt-api` spec, `DATA_CONTRACTS.md`, README, and tests

Tracked in [PLANS.md](../../../PLANS.md) tech debt tracker.
