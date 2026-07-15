## ADDED Requirements

### Requirement: OpenAI transcription endpoint

The bridge SHALL expose `POST /v1/audio/transcriptions` accepting `multipart/form-data` with Bearer authentication using the existing `BRIDGE_AUTH_TOKEN`. The endpoint SHALL process requests **synchronously** — the client receives the final transcription in a single HTTP response with no polling or webhooks.

#### Scenario: Successful JSON transcription

- **WHEN** client sends `POST /v1/audio/transcriptions` with valid Bearer token, a non-empty `file` field, `model=whisper-1`, and default `response_format=json`
- **THEN** the bridge returns HTTP 200 with `Content-Type: application/json` and body `{"text": "<transcribed text>"}`

#### Scenario: Missing authentication

- **WHEN** client sends a request without `Authorization: Bearer <token>` or with an invalid token
- **THEN** the bridge returns HTTP 401 with body `{"error": {"message": "invalid api key", "type": "authentication_error", "code": null}}`

### Requirement: Multipart request parameters

The endpoint SHALL accept the following multipart fields: `file` (required), `model` (required), `language` (optional), `response_format` (optional, default `json`), `temperature` (optional, default `0.0`), `prompt` (optional, mapped to Cog `initial_prompt`), and `timestamp_granularities` (optional).

#### Scenario: Missing file

- **WHEN** client omits the `file` field or sends an empty file
- **THEN** the bridge returns HTTP 400 with `{"error": {"message": "file is required", "type": "invalid_request_error", "code": null}}`

#### Scenario: Language parameter forwarded

- **WHEN** client sends `language=fr`
- **THEN** the bridge submits a Cog prediction with `language: "fr"`

#### Scenario: Timestamp granularities with non-verbose format

- **WHEN** client sends `timestamp_granularities` with `response_format=json`
- **THEN** the bridge processes the request without error and returns a normal json response

#### Scenario: Timestamp granularities with verbose_json

- **WHEN** client sends `response_format=verbose_json` and `timestamp_granularities[]=word`
- **THEN** the bridge returns verbose_json with a top-level `words` array

### Requirement: Model name mapping

The bridge SHALL map OpenAI model names to Cog `whisper_model` values: `whisper-1` → `large-v3-turbo`, `large-v3` → `large-v3`, `large-v3-turbo` → `large-v3-turbo`, `tiny` → `tiny`.

#### Scenario: whisper-1 alias

- **WHEN** client sends `model=whisper-1`
- **THEN** the Cog prediction input contains `whisper_model: "large-v3-turbo"`

#### Scenario: Unsupported model

- **WHEN** client sends `model=invalid-model`
- **THEN** the bridge returns HTTP 400 with `{"error": {"message": "model 'invalid-model' not supported", "type": "invalid_request_error", "code": null}}`

### Requirement: Response formats

The bridge SHALL support `response_format` values: `json`, `text`, `verbose_json`, `srt`, and `vtt`.

#### Scenario: Text format

- **WHEN** client sends `response_format=text`
- **THEN** the bridge returns HTTP 200 with `Content-Type: text/plain; charset=utf-8` and the transcribed text as the raw body

#### Scenario: Verbose JSON with segment timestamps

- **WHEN** client sends `response_format=verbose_json` and `timestamp_granularities[]=segment`
- **THEN** the bridge returns HTTP 200 with JSON containing `task`, `language`, `duration`, `text`, and `segments[]` where each segment includes `id`, `seek`, `start`, `end`, `text`, `tokens`, `temperature`, `avg_logprob`, `compression_ratio`, and `no_speech_prob`

#### Scenario: Verbose JSON with word timestamps

- **WHEN** client sends `response_format=verbose_json` and `timestamp_granularities[]=word`
- **THEN** the bridge returns HTTP 200 with a top-level `words[]` array (each item `{word, start, end}`) flattened from Cog aligned segments — words are NOT nested inside `segments[]`

#### Scenario: SRT format

- **WHEN** client sends `response_format=srt`
- **THEN** the bridge returns HTTP 200 with valid SRT subtitle content derived from segment timestamps

#### Scenario: VTT format

- **WHEN** client sends `response_format=vtt`
- **THEN** the bridge returns HTTP 200 with valid WebVTT content derived from segment timestamps

### Requirement: File size and format validation

The bridge SHALL reject uploads exceeding `OPENAI_STT_MAX_FILE_SIZE_MB` (default 25 MB). The bridge SHALL reject file extensions not in the OpenAI official allowlist (`flac`, `mp3`, `mp4`, `mpeg`, `mpga`, `m4a`, `ogg`, `wav`, `webm`) with HTTP 422.

#### Scenario: Oversized file

- **WHEN** client uploads a file larger than the configured maximum
- **THEN** the bridge returns HTTP 413 with `{"error": {"message": "file exceeds 25MB limit", "type": "invalid_request_error", "code": null}}`

#### Scenario: Unsupported audio format

- **WHEN** client uploads a file with extension `aac` or other non-OpenAI extension
- **THEN** the bridge returns HTTP 422 with `{"error": {"message": "unsupported audio format", "type": "invalid_request_error", "code": null}}`

#### Scenario: Supported OpenAI format accepted

- **WHEN** client uploads a file with extension `ogg`
- **THEN** the bridge accepts the file and proceeds to transcription

### Requirement: Cog integration — synchronous JSON with data URI

The bridge SHALL submit a **synchronous** Cog `POST /predictions` (no `Prefer: respond-async`) with JSON body. The `audio_file` field SHALL be a base64 **data URI** encoding of the uploaded file. The bridge SHALL NOT poll `GET /predictions/{id}` or use webhook/Redis on this path. Cog input SHALL include `align_output: true`, `diarization: false`, and other fixed parameters as defined in the design.

#### Scenario: Data URI passed to Cog

- **WHEN** client uploads a valid ogg file
- **THEN** the bridge sends Cog JSON with `audio_file` starting with `data:audio/ogg;base64,`

#### Scenario: Cog failure

- **WHEN** Cog returns `status: failed` with an error message
- **THEN** the bridge returns HTTP 500 with `{"error": {"message": "transcription failed: <detail>", "type": "server_error", "code": null}}`

#### Scenario: Cog timeout

- **WHEN** the synchronous Cog POST exceeds `OPENAI_STT_TIMEOUT_SECONDS` (default 300)
- **THEN** the bridge returns HTTP 504 with `{"error": {"message": "transcription timed out", "type": "server_error", "code": null}}`

### Requirement: OpenAI SDK compatibility

The endpoint SHALL be compatible with the openai-python SDK when configured with `base_url="http://<host>:8080/v1"` and `api_key=<BRIDGE_TOKEN>`.

#### Scenario: openai-python client usage

- **WHEN** client calls `client.audio.transcriptions.create(model="whisper-1", file=<file>, language="fr")`
- **THEN** the request succeeds synchronously and `result.text` contains the transcribed string

### Requirement: Non-interference with Replicate API

Existing bridge routes (`POST /predictions`, `GET /predictions/<id>`, webhooks, health checks) SHALL continue to behave unchanged.

#### Scenario: Replicate predictions unchanged

- **WHEN** client sends `POST /predictions` with Replicate-compatible JSON body
- **THEN** the bridge proxies to Cog with existing webhook injection and Redis storage behavior
