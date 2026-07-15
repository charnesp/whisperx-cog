## 1. Foundation

- [x] 1.1 Document resolved handoff decision: data URI in JSON `audio_file` per Cog HTTP API (no shared volume, no poll)
- [x] 1.2 Add env vars `OPENAI_STT_TIMEOUT_SECONDS`, `OPENAI_STT_MAX_FILE_SIZE_MB` to k8s bridge container and docker-compose

## 2. Core module — `bridge/openai_compat.py`

- [x] 2.1 Create module skeleton: `MODEL_MAP`, `OPENAI_AUDIO_EXTENSIONS` (flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, webm), env-driven timeout/max size
- [x] 2.2 Implement OpenAI error helper returning `{"error": {"message", "type", "code": null}}` with correct HTTP status codes
- [x] 2.3 Implement multipart form parser (stdlib `cgi.FieldStorage` or equivalent) extracting `file`, `model`, `language`, `response_format`, `temperature`, `prompt`, `timestamp_granularities`
- [x] 2.4 Implement request validation: file present/non-empty, model in map, file size limit, OpenAI extension allowlist
- [x] 2.5 Implement `build_audio_data_uri(file_bytes, extension)` → `data:<mime>;base64,...`
- [x] 2.6 Implement sync Cog client: build fixed input payload with data URI, `POST /predictions` without async header, HTTP read timeout from env
- [x] 2.7 Implement response converters: `json`, `text`, `verbose_json` (top-level `words[]` or `segments[]` per `timestamp_granularities`), `srt`, `vtt`
- [x] 2.8 Implement `handle_transcription_request(headers, rfile, content_length)` orchestrator

## 3. Bridge routing — `bridge/bridge.py`

- [x] 3.1 Add route dispatch in `do_POST` for exact path `/v1/audio/transcriptions` before generic proxy
- [x] 3.2 Wire OpenAI-specific auth errors (401 with OpenAI error schema) for this route
- [x] 3.3 Add `[bridge ext]` logging for OpenAI STT requests (path, bytes, model, format, status, duration — no payload logging)
- [x] 3.4 Run `python3 scripts/sync-bridge-to-k8s.py` and verify `scripts/check-bridge-sync.py` passes

## 4. Tests — `tests/test_openai_stt.py`

- [x] 4.1 Unit tests: model mapping, extension allowlist (reject aac, accept ogg), data-URI builder
- [x] 4.2 Unit tests: each response format converter (mock Cog output fixture)
- [x] 4.3 HTTP test: valid ogg, `response_format=json` → `{text: "..."}`
- [x] 4.4 HTTP test: `response_format=text` → `text/plain`
- [x] 4.5 HTTP test: `response_format=verbose_json` + `timestamp_granularities[]=word` → top-level `words[]`; `[]=segment` → `segments[]`
- [x] 4.6 HTTP test: missing file → 400
- [x] 4.7 HTTP test: invalid model → 400
- [x] 4.8 HTTP test: no auth and wrong auth → 401 with OpenAI error shape
- [x] 4.9 HTTP test: oversized file → 413
- [x] 4.10 HTTP test: unsupported extension (aac) → 422
- [x] 4.11 HTTP test: `model=whisper-1` → Cog JSON contains `whisper_model: large-v3-turbo` and data URI in `audio_file`
- [x] 4.12 HTTP test: `language=fr` forwarded to Cog input
- [x] 4.13 HTTP test: mock Cog delay exceeding timeout → 504
- [x] 4.14 HTTP test: sync Cog POST used (no GET poll, no async header)

## 5. Documentation and harness

- [x] 5.1 Add "OpenAI-compatible endpoint" section to `README.md` (curl example, model table, env vars, OpenAI extension list)
- [x] 5.2 Update `docs/BRIDGE.md` with route summary, sync flow, data-URI handoff note
- [x] 5.3 Update `docs/DATA_CONTRACTS.md` with OpenAI response shapes at bridge boundary
- [x] 5.4 Run `make -f Makefile.harness check` and fix any failures
