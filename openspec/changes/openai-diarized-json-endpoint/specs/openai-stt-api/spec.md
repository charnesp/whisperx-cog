## ADDED Requirements

### Requirement: Diarized JSON response format

The bridge SHALL support `response_format=diarized_json` on `POST /v1/audio/transcriptions`, returning HTTP 200 with `Content-Type: application/json` and an OpenAI `TranscriptionDiarized` object containing `task` (`"transcribe"`), `duration` (float, seconds), `text` (concatenated transcript), and `segments[]` where each segment includes `id` (string), `start`, `end`, `speaker`, `text`, and `type` (`"transcript.text.segment"`).

#### Scenario: Successful diarized transcription

- **WHEN** client sends valid Bearer token, non-empty `file`, `model=gpt-4o-transcribe-diarize`, `response_format=diarized_json`, and whisperx has `HUGGINGFACE_TOKEN` configured
- **THEN** the bridge returns HTTP 200 with JSON containing `task: "transcribe"`, top-level `text`, `duration`, and `segments[]` with speaker labels and timestamps

#### Scenario: Speaker labels mapped to letters

- **WHEN** Cog output contains speakers `SPEAKER_00` and `SPEAKER_01` and client did not send `known_speaker_names`
- **THEN** diarized_json segments use OpenAI letter labels `A` and `B` respectively (by order of first appearance)

#### Scenario: Known speaker names honored

- **WHEN** client sends `known_speaker_names[]=alice` and Cog output has one speaker `SPEAKER_00`
- **THEN** diarized_json segments for that speaker use `speaker: "alice"`

### Requirement: Diarize model alias

The bridge SHALL accept `model=gpt-4o-transcribe-diarize` and map it to Cog `whisper_model: "large-v3-turbo"` with `diarization: true`.

#### Scenario: Diarize model enables Cog diarization

- **WHEN** client sends `model=gpt-4o-transcribe-diarize` and `response_format=diarized_json`
- **THEN** the Cog prediction input contains `whisper_model: "large-v3-turbo"` and `diarization: true`

#### Scenario: Diarize model with plain json response

- **WHEN** client sends `model=gpt-4o-transcribe-diarize` and `response_format=json` (default)
- **THEN** the bridge returns HTTP 200 with `{"text": "<transcribed text>"}` without speaker segments (OpenAI-compatible plain json)

### Requirement: Diarize parameter validation

On diarization requests (`model=gpt-4o-transcribe-diarize` or `response_format=diarized_json`), the bridge SHALL reject incompatible parameters with HTTP 400 and OpenAI error envelope.

#### Scenario: Prompt rejected on diarize path

- **WHEN** client sends `model=gpt-4o-transcribe-diarize` with non-empty `prompt`
- **THEN** the bridge returns HTTP 400 with `invalid_request_error`

#### Scenario: Timestamp granularities rejected on diarize path

- **WHEN** client sends `model=gpt-4o-transcribe-diarize` with `timestamp_granularities[]=word`
- **THEN** the bridge returns HTTP 400 with `invalid_request_error`

#### Scenario: Known speaker references not supported

- **WHEN** client sends `known_speaker_references[]` on a diarize request
- **THEN** the bridge returns HTTP 400 with a message indicating the parameter is not supported yet (see PLANS.md — follow-up TODO)

#### Scenario: Chunking strategy accepted

- **WHEN** client sends `model=gpt-4o-transcribe-diarize`, `response_format=diarized_json`, and `chunking_strategy=auto`
- **THEN** the bridge processes the request successfully (chunking_strategy is accepted and ignored internally)

#### Scenario: Inconsistent model and response format

- **WHEN** client sends `model=whisper-1` with `response_format=diarized_json`
- **THEN** the bridge returns HTTP 400 with `invalid_request_error`

### Requirement: HuggingFace token for diarization

When diarization is requested, Cog (`predict.py`) SHALL resolve the HuggingFace token from `huggingface_access_token` input or fall back to the whisperx process env `HUGGINGFACE_TOKEN`. If no token is available after fallback, Cog SHALL fail the prediction with a clear error (bridge maps to HTTP 500). The bridge SHALL NOT duplicate `HUGGINGFACE_TOKEN` in its own container env.

#### Scenario: Missing HF token fails prediction

- **WHEN** client requests diarized transcription and whisperx `HUGGINGFACE_TOKEN` is unset or empty and Cog input has no `huggingface_access_token`
- **THEN** Cog returns `status: failed` and the bridge returns HTTP 500 with a transcription-failed error message

#### Scenario: Whisperx env token used for diarization

- **WHEN** client requests diarized transcription, Cog input has `huggingface_access_token: null`, and whisperx has `HUGGINGFACE_TOKEN` set
- **THEN** diarization runs successfully using the whisperx env token

## MODIFIED Requirements

### Requirement: Multipart request parameters

The endpoint SHALL accept the following multipart fields: `file` (required), `model` (required), `language` (optional), `response_format` (optional, default `json`), `temperature` (optional, default `0.0`), `prompt` (optional, mapped to Cog `initial_prompt`), `timestamp_granularities` (optional), `chunking_strategy` (optional, diarize path only), and `known_speaker_names` (optional, diarize path only).

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

### Requirement: Model name mapping

The bridge SHALL map OpenAI model names to Cog `whisper_model` values: `whisper-1` → `large-v3-turbo`, `gpt-4o-transcribe-diarize` → `large-v3-turbo` (with diarization enabled), `large-v3` → `large-v3`, `large-v3-turbo` → `large-v3-turbo`, `tiny` → `tiny`.

#### Scenario: whisper-1 alias

- **WHEN** client sends `model=whisper-1`
- **THEN** the Cog prediction input contains `whisper_model: "large-v3-turbo"` and `diarization: false`

#### Scenario: Unsupported model

- **WHEN** client sends `model=invalid-model`
- **THEN** the bridge returns HTTP 400 with `{"error": {"message": "model 'invalid-model' not supported", "type": "invalid_request_error", "code": null}}`

### Requirement: Response formats

The bridge SHALL support `response_format` values: `json`, `text`, `verbose_json`, `srt`, `vtt`, and `diarized_json`.

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

### Requirement: Cog integration — synchronous JSON with data URI

The bridge SHALL submit a **synchronous** Cog `POST /predictions` (no `Prefer: respond-async`) with JSON body. The `audio_file` field SHALL be a base64 **data URI** encoding of the uploaded file. The bridge SHALL NOT poll `GET /predictions/{id}` or use webhook/Redis on this path. Cog input SHALL include `align_output: true`. Cog input SHALL set `diarization: false` for non-diarize requests and `diarization: true` for diarize requests (HF token resolved inside whisperx per HuggingFace token requirement), plus other fixed parameters as defined in the design.

#### Scenario: Data URI passed to Cog

- **WHEN** client uploads a valid ogg file
- **THEN** the bridge sends Cog JSON with `audio_file` starting with `data:audio/ogg;base64,`

#### Scenario: Non-diarize Cog input

- **WHEN** client sends `model=whisper-1` with default `response_format=json`
- **THEN** the Cog prediction input contains `diarization: false` and does not require `huggingface_access_token`

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

#### Scenario: openai-python diarized client usage

- **WHEN** client calls `client.audio.transcriptions.create(model="gpt-4o-transcribe-diarize", file=<file>, response_format="diarized_json", chunking_strategy="auto")`
- **THEN** the request succeeds synchronously and the response includes `segments` with speaker labels
