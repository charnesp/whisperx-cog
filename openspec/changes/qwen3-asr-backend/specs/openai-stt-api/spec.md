## ADDED Requirements

### Requirement: Qwen3-ASR model alias

The bridge SHALL accept `model=qwen3-asr` on `POST /v1/audio/transcriptions` and map it to Cog `whisper_model: "qwen3-asr"`. The model whitelist SHALL remain closed: any model not in `MODEL_MAP` returns HTTP 400. When `ENABLE_QWEN` is unset, `model=qwen3-asr` SHALL be accepted (kill-switch defaults to enabled). The backend SHALL be rejected with HTTP 400 and a feature-disabled message only when `ENABLE_QWEN` is explicitly set to a falsy value (`0`, `false`, empty, `no`, `off`), without requiring a redeploy.

#### Scenario: Qwen model accepted

- **WHEN** client sends `model=qwen3-asr` with a valid file and `ENABLE_QWEN=1` is set on the bridge
- **THEN** the Cog prediction input contains `whisper_model: "qwen3-asr"`

#### Scenario: Qwen model rejected when kill-switch is explicitly disabled

- **WHEN** client sends `model=qwen3-asr` and `ENABLE_QWEN` is explicitly set to a falsy value (`0`, `false`, empty, `no`, `off`)
- **THEN** the bridge returns HTTP 400 with `invalid_request_error` and a message indicating the qwen backend is disabled

#### Scenario: Qwen model accepted when kill-switch is unset

- **WHEN** client sends `model=qwen3-asr` with a valid file and `ENABLE_QWEN` is unset
- **THEN** the Cog prediction input contains `whisper_model: "qwen3-asr"` (kill-switch defaults to enabled)

#### Scenario: Unknown model still rejected

- **WHEN** client sends `model=qwen4-asr`
- **THEN** the bridge returns HTTP 400 with `{"error": {"message": "model 'qwen4-asr' not supported", "type": "invalid_request_error", "code": null}}`

### Requirement: Hotwords routed to Qwen context

When `model=qwen3-asr`, the bridge SHALL forward the client `hotwords` string as Cog input `hotwords`, and the Cog predictor SHALL pass it as the Qwen `context` system message — identical for every batch, with no accumulation across batches, and neutral when empty. When the context exceeds ~2000 characters, the predictor SHALL truncate it and log a truncation warning containing lengths only, never content. For `large-v3-turbo`, `large-v3`, and `tiny`, `hotwords` SHALL keep the existing faster-whisper semantics with the whisper path unchanged. Hotwords content SHALL never appear in bridge or predictor logs.

#### Scenario: Hotwords forwarded on qwen path

- **WHEN** client sends `model=qwen3-asr` and `hotwords="Backblaze, Supabase"`
- **THEN** the Cog prediction input contains `hotwords: "Backblaze, Supabase"` and the qwen pipeline receives it as `context`

#### Scenario: Hotwords not forwarded on whisper path

- **WHEN** client sends `model=whisper-1` and `hotwords="Backblaze, Supabase"`
- **THEN** the Cog prediction input contains `hotwords: null` (unchanged behavior)

#### Scenario: Empty hotwords is neutral

- **WHEN** client sends `model=qwen3-asr` with no `hotwords` field or an empty string
- **THEN** the qwen pipeline receives an empty/absent context and output matches the baseline run without context (bit-identical acceptance criterion)

#### Scenario: Oversized context truncated with safe log

- **WHEN** client sends `model=qwen3-asr` with `hotwords` longer than the ~2000-char context cap
- **THEN** the predictor truncates the context to the cap and logs a warning with the original and truncated lengths only (no content)

### Requirement: Per-model default batch size

The bridge SHALL include `batch_size` in the Cog prediction input only when the client provides it (no hard-coded default). On the qwen path, the predictor SHALL apply `QWEN_DEFAULT_BATCH = 4` when `batch_size` is absent and SHALL clamp explicit values to the `[1, 8]` range. The faster-whisper default batch remains unchanged.

#### Scenario: Default batch on qwen path

- **WHEN** client sends `model=qwen3-asr` without `batch_size`
- **THEN** the predictor runs the qwen pipeline with `batch_size=4`

#### Scenario: Explicit batch clamped

- **WHEN** client sends `model=qwen3-asr` with `batch_size=12`
- **THEN** the predictor runs the qwen pipeline with `batch_size=8`

#### Scenario: Batch_size omitted from Cog input

- **WHEN** client sends `model=whisper-1` without `batch_size`
- **THEN** the Cog prediction input does not contain a `batch_size` field (predictor default applies)

### Requirement: Language handling on the qwen path

When `whisper_model=qwen3-asr`, the predictor SHALL skip the whisper `detect_language` loop — Qwen performs its own language detection. A client-provided `language` SHALL be passed through as-is. The `language_detection_min_prob` / `language_detection_max_tries` inputs SHALL be ignored on this path.

#### Scenario: Language detection skipped

- **WHEN** client sends `model=qwen3-asr` without `language`
- **THEN** the predictor does not invoke the whisper `detect_language` loop and Qwen detects the language internally

#### Scenario: Language passed through

- **WHEN** client sends `model=qwen3-asr` with `language=fr`
- **THEN** the qwen pipeline receives `language: "fr"`

### Requirement: Qwen forced aligner on the qwen path

When `whisper_model=qwen3-asr` and alignment is enabled, the predictor SHALL align word-level timestamps with `Qwen/Qwen3-ForcedAligner-0.6B` (baked under `/models`), not the wav2vec2 aligner. The downstream pyannote diarization stage SHALL remain unchanged (256-dim speaker embeddings, identical output schema).

#### Scenario: Word timestamps present on qwen output

- **WHEN** client sends `model=qwen3-asr` with default `align_output=true`
- **THEN** the response contains word-level timestamps produced by the Qwen ForcedAligner

#### Scenario: Diarization unchanged after qwen alignment

- **WHEN** a qwen transcription with `diarization=true` completes
- **THEN** `assign_word_speakers` receives words from the Qwen ForcedAligner and the output speaker fields use the same schema as the whisper path

### Requirement: Qwen model baking and offline enforcement

The Cog image SHALL bake both Qwen snapshots (`Qwen/Qwen3-ASR-1.7B` and `Qwen/Qwen3-ForcedAligner-0.6B`) into `/models`, with HF revisions pinned in `models.lock`, `HF_HUB_OFFLINE=1` set in the Cog environment, and a fail-fast boot check that raises a clear error when baked weights are missing.

#### Scenario: Baked weights present

- **WHEN** the Cog image builds
- **THEN** both qwen snapshots exist under `/models` with non-empty weight files (`test -s`) matching the revisions pinned in `models.lock`

#### Scenario: Missing baked weights fail fast

- **WHEN** a qwen snapshot is absent from `/models` at model load
- **THEN** the predictor raises a clear boot-time error instead of silently downloading from HuggingFace

### Requirement: Whisper path regression invariance

The faster-whisper path SHALL remain bit-identical: for `tiny`, `large-v3`, and `large-v3-turbo`, outputs MUST be unchanged by this change, including when clients send `hotwords`.

#### Scenario: Golden-set whisper regression

- **WHEN** the golden set (fixed FR extract + real meeting extract) is replayed for `tiny`, `large-v3`, and `large-v3-turbo`
- **THEN** all outputs are bit-identical to the pre-change run

## MODIFIED Requirements

### Requirement: Model name mapping

The bridge SHALL map OpenAI model names to Cog `whisper_model` values: `whisper-1` → `large-v3-turbo`, `gpt-4o-transcribe-diarize` → `large-v3-turbo` (with diarization enabled), `large-v3` → `large-v3`, `large-v3-turbo` → `large-v3-turbo`, `tiny` → `tiny`, and additionally `qwen3-asr` → `qwen3-asr`. The whitelist SHALL remain closed: any model not in `MODEL_MAP` returns HTTP 400.

#### Scenario: whisper-1 alias

- **WHEN** client sends `model=whisper-1`
- **THEN** the Cog prediction input contains `whisper_model: "large-v3-turbo"` and `diarization: false`

#### Scenario: Unsupported model

- **WHEN** client sends `model=invalid-model`
- **THEN** the bridge returns HTTP 400 with `{"error": {"message": "model 'invalid-model' not supported", "type": "invalid_request_error", "code": null}}`

#### Scenario: qwen3-asr alias

- **WHEN** client sends `model=qwen3-asr` with `ENABLE_QWEN=1` set on the bridge
- **THEN** the Cog prediction input contains `whisper_model: "qwen3-asr"`

### Requirement: Multipart request parameters

The endpoint SHALL accept the following multipart fields: `file` (required), `model` (required), `language` (optional), `response_format` (optional, default `json`), `temperature` (optional, default `0.0`), `prompt` (optional, mapped to Cog `initial_prompt`), `timestamp_granularities` (optional), `chunking_strategy` (optional, diarize path only), `known_speaker_names` (optional, diarize path only), and `hotwords` (optional; when `model=qwen3-asr`, routed to the Qwen `context` system message). When the client provides `batch_size`, it SHALL be passed through to the Cog input (no hard-coded default); when absent, the Cog input SHALL omit `batch_size` entirely and the per-model predictor default applies.

#### Scenario: Missing file

- **WHEN** client omits the `file` field or sends an empty file
- **THEN** the bridge returns HTTP 400 with `{"error": {"message": "file is required", "type": "invalid_request_error", "code": null}}`

#### Scenario: Language parameter forwarded

- **WHEN** client sends `language=fr`
- **THEN** the bridge submits a Cog prediction with `language: "fr"`

#### Scenario: Timestamp granularities with non-verbose format

- **WHEN** client sends `timestamp_granularities` with `response_format=json` on a non-diarize request
- **THEN** the bridge processes the request without error and returns a normal json response

#### Scenario: Timestamp granularities with verbose_json

- **WHEN** client sends `response_format=verbose_json` and `timestamp_granularities[]=word`
- **THEN** the bridge returns verbose_json with a top-level `words` array

#### Scenario: Hotwords forwarded on qwen path

- **WHEN** client sends `model=qwen3-asr` and `hotwords="Backblaze, Supabase"`
- **THEN** the Cog prediction input contains `hotwords: "Backblaze, Supabase"`

#### Scenario: Hotwords not forwarded on whisper path

- **WHEN** client sends `model=whisper-1` and `hotwords="Backblaze, Supabase"`
- **THEN** the Cog prediction input contains `hotwords: null` (unchanged behavior)

#### Scenario: Batch_size forwarded when provided

- **WHEN** client sends `model=qwen3-asr` with `batch_size=6`
- **THEN** the Cog prediction input contains `batch_size: 6`

#### Scenario: Batch_size omitted when not provided

- **WHEN** client sends `model=whisper-1` without `batch_size`
- **THEN** the Cog prediction input does not contain a `batch_size` field (predictor default applies)