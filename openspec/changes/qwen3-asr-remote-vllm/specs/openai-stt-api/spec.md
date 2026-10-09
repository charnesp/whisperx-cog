## ADDED Requirements

### Requirement: Remote backend selector

The qwen3-asr transcription backend SHALL support a remote engine selected by the `QWEN_BACKEND` environment variable read at request time (never at import). Unset or empty SHALL resolve to `local` (default; operator decision 2026-10-09: zero behavioral change at deploy). The value `remote` SHALL select the remote path. Any other non-empty value SHALL resolve to `local` with a warning log naming the invalid value. The faster-whisper backends (tiny, large-v3, large-v3-turbo) and the diarize path SHALL be entirely unaffected by this selector.

#### Scenario: Default local when unset

- **WHEN** `QWEN_BACKEND` is unset and `model=qwen3-asr` is transcoded
- **THEN** the in-process local engine is used (current behavior, bit-for-bit)

#### Scenario: Invalid selector resolves local with warning

- **WHEN** `QWEN_BACKEND=rem0te` and `model=qwen3-asr` is transcoded
- **THEN** the local engine is used and a warning log names the invalid value

#### Scenario: Remote selected

- **WHEN** `QWEN_BACKEND=remote` and `model=qwen3-asr` is transcoded
- **THEN** the transcription windows are posted to the remote engine configured by `QWEN_REMOTE_BASE_URL`

### Requirement: Remote base URL is environment-only

The remote base URL SHALL come exclusively from `QWEN_REMOTE_BASE_URL`; the remote model name SHALL come exclusively from `QWEN_REMOTE_MODEL`. Neither hostname, IP address, port, nor model name of the deployment environment SHALL appear anywhere in the repository (code, tests, docs, compose, k8s); documentation SHALL use a neutral placeholder (`http://<host>:<port>/v1`). With `QWEN_BACKEND=remote` and `QWEN_REMOTE_BASE_URL` or `QWEN_REMOTE_MODEL` unset or empty, `setup()` SHALL fail fast with an explicit configuration error naming the missing variable (never a fallback constant).

#### Scenario: URL taken from environment

- **WHEN** `QWEN_BACKEND=remote` and `QWEN_REMOTE_BASE_URL=http://example.invalid:9000/v1`
- **THEN** the HTTP client posts to exactly that URL (captured by the injectable client)

#### Scenario: Missing URL or model fails at setup

- **WHEN** `QWEN_BACKEND=remote` and `QWEN_REMOTE_BASE_URL` (or `QWEN_REMOTE_MODEL`) is unset
- **THEN** `setup()` fails with an explicit configuration error naming the missing variable (no default address, no silent local fallback)

#### Scenario: Repo carries no concrete address

- **WHEN** the hygiene grep gate runs over the repository
- **THEN** no personal IP or port is found; docs use a neutral placeholder

#### Scenario: URL taken from environment

- **WHEN** `QWEN_BACKEND=remote` and `QWEN_REMOTE_BASE_URL=http://example.invalid:9000/v1`
- **THEN** the HTTP client posts to exactly that URL (captured by the injectable client)

#### Scenario: Missing URL fails fast

- **WHEN** `QWEN_BACKEND=remote` and `QWEN_REMOTE_BASE_URL` is unset
- **THEN** the call fails with an explicit configuration error naming `QWEN_REMOTE_BASE_URL` (no default address, no silent local fallback)

#### Scenario: Repo carries no concrete address

- **WHEN** the hygiene grep gate runs over the repository
- **THEN** no personal IP or port is found; docs use a neutral placeholder

### Requirement: Remote request contract

Each VAD window SHALL be transcribed by one multimodal chat completion request to `{QWEN_REMOTE_BASE_URL}/chat/completions` with `model=QWEN_REMOTE_MODEL`, `temperature=0`, a bounded `max_tokens`, audio as a `data:audio/wav;base64` URI (mono 16 kHz PCM16 wav) in `audio_url`, and the `context` system message (formatted hotwords) in EVERY request. The client SHALL never request server-side timestamps (no `verbose_json`, no `timestamp_granularities`). Window concurrency SHALL be bounded by the clamped batch_size (default 4, cap 8, values <=0 or non-integer rejected by the existing clamp helper), and results SHALL be fused in VAD window order. The `language` param, when provided, SHALL be carried as a system-message instruction.

#### Scenario: Context travels with every request

- **WHEN** a 3-window batch is transcribed with hotwords configured
- **THEN** every captured request carries the same context system message and hotword content never appears in logs

#### Scenario: No server-side timestamps requested

- **WHEN** any remote request is captured
- **THEN** its body contains no `response_format=verbose_json` and no `timestamp_granularities`

#### Scenario: Language instruction carried

- **WHEN** the client provides `language=fr`
- **THEN** every remote request's system message carries the French-language instruction

#### Scenario: Order preserved under concurrency

- **WHEN** 3 windows (start values 0.0, 30.0, 60.0) are transcribed with a pool of 2
- **THEN** the fused result segments are ordered by VAD window start regardless of completion order

### Requirement: Remote response parsing

The reply content SHALL be parsed to text only: a `language X<asr_text>text` body SHALL yield `text` (prefix never present in segments); content without the prefix (structured output) SHALL be used as-is; empty or missing content SHALL produce a typed error; `finish_reason=="length"` SHALL produce a typed error (never silently-truncated text); `language None<asr_text>` or empty transcription (silence) SHALL yield an empty segment with the same rule as the local path. The reply text SHALL NEVER determine segment timestamps.

#### Scenario: Prefixed reply parsed

- **WHEN** the reply content is `language French<asr_text>Bonjour tu va bien`
- **THEN** the segment text is `Bonjour tu va bien` and the segment start/end remain the VAD window values

#### Scenario: Empty reply is a typed error

- **WHEN** the reply content is empty
- **THEN** the batch fails with a typed error (not an empty transcript)

### Requirement: Remote failures are explicit (no fallback)

The remote configuration SHALL be resolved once in `setup()` with fail-fast on misconfiguration; in remote mode the local qwen-ASR loader SHALL never be invoked. `QWEN_REMOTE_TIMEOUT_S` (default 300) SHALL be a per-request connect+read timeout; at v1 there SHALL be zero retries (a mid-swap 502 surfaces as `qwen_remote_unavailable`, never retried). One failed window SHALL fail the whole prediction with pending futures cancelled; in-flight requests may complete but their results are discarded. Remote failures SHALL surface at the bridge boundary as a prediction failure carrying a stable grep-able message prefixed `QwenRemoteError:` with a category (config/connection/timeout/http_status/parse); messages SHALL never contain the full URL, the audio payload, or the whole reply body (truncated). There SHALL be NO silent fallback to the local engine (operator decision 2026-10-09); logs carry the prefix `qwen-remote:` with status and elapsed time, never audio or hotword content.

#### Scenario: Remote down gives explicit failure

- **WHEN** the remote engine refuses connections and `QWEN_BACKEND=remote`
- **THEN** the prediction fails with a `QwenRemoteError:` connection-category message and HTTP 502 at the bridge (no local retry, no fallback)

#### Scenario: Timeout honored

- **WHEN** the remote engine never answers within `QWEN_REMOTE_TIMEOUT_S`
- **THEN** the request fails with a `QwenRemoteError:` timeout-category message and the log records the elapsed duration

#### Scenario: Setup fail-fast on broken remote config

- **WHEN** `QWEN_BACKEND=remote` without `QWEN_REMOTE_BASE_URL` and the predictor boots
- **THEN** `setup()` raises the config-category `QwenRemoteError:` before serving any request

#### Scenario: Local ASR loader not invoked in remote mode

- **WHEN** `QWEN_BACKEND=remote` and the qwen backend is exercised
- **THEN** the local `asr_qwen` model loader is never called (ForcedAligner and pyannote stay local)

### Requirement: Kill-switch unchanged under remote mode

The `ENABLE_QWEN` kill-switch SHALL remain unchanged. It is enforced ONLY at the bridge (which this change does not modify): explicitly falsy values (`0`, `false`, empty, `no`, `off`) SHALL continue to reject `model=qwen3-asr` with HTTP 400 and a feature-disabled message before any request reaches cog, for BOTH `QWEN_BACKEND` values (the resolved backend is irrelevant to the gate). cog SHALL NOT duplicate the gate.

#### Scenario: Kill-switch precedes backend selection

- **WHEN** `ENABLE_QWEN=0` and `model=qwen3-asr` is sent (any `QWEN_BACKEND`)
- **THEN** the bridge returns HTTP 400 with the feature-disabled message and cog receives no request

### Requirement: Output invariants hold in remote mode

With `QWEN_BACKEND=remote`: segment timestamps SHALL come from the local VAD chunking only; the downstream ForcedAligner word alignment and pyannote diarization stages SHALL run unchanged; the `TranscriptionResult` and diarized output schemas SHALL be identical to the local mode. The bridge request/response contract SHALL be unchanged.

#### Scenario: Schema identity

- **WHEN** the same audio is transcribed in local mode and remote mode
- **THEN** both outputs share the same segment/word/diarization schema (bit-identity of TEXT is not required across engines; schema identity is)

#### Scenario: Alignment runs after remote transcription

- **WHEN** `align_output=true` on the remote path
- **THEN** word timestamps are produced by the Qwen ForcedAligner (unchanged code path)
