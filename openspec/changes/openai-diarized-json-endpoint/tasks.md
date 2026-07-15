## 1. Cog — HuggingFace token env fallback

- [x] 1.1 In `predict.py`, when `diarization: true` and `huggingface_access_token` is null/empty, fall back to `os.environ.get("HUGGINGFACE_TOKEN")`
- [x] 1.2 Fail hard (RuntimeError) when diarization requested but no token after fallback — replace silent skip + warning
- [x] 1.3 Unit test for env fallback and fail-hard behavior (mock env, no GPU)

## 2. Core — diarize detection and validation

- [x] 2.1 Add `gpt-4o-transcribe-diarize` to `MODEL_MAP` in `bridge/openai_compat.py`
- [x] 2.2 Implement `is_diarize_request()` (model or `response_format=diarized_json`)
- [x] 2.3 Extend `validate_transcription_request()` — allow `diarized_json` in `response_format`; reject inconsistent model/format pairs
- [x] 2.4 Add diarize param validation: reject `prompt`, `timestamp_granularities`, `include[]=logprobs`, `known_speaker_references[]`; accept `chunking_strategy` and `known_speaker_names[]`

## 3. Cog integration (bridge)

- [x] 3.1 Branch `build_cog_input()` — set `diarization: true` on diarize path; omit `initial_prompt` on diarize path; keep `huggingface_access_token: null`
- [x] 3.2 Keep `diarization: false` and existing behavior for non-diarize requests unchanged

## 4. Response conversion

- [x] 4.1 Add `MOCK_COG_DIARIZED_OUTPUT` fixture with `speaker` on words/segments for tests
- [x] 4.2 Implement speaker label mapper (`SPEAKER_00` → `A`, with `known_speaker_names` override)
- [x] 4.3 Implement `convert_to_diarized_json()` — word-level speaker turn grouping → OpenAI `TranscriptionDiarized` shape
- [x] 4.4 Wire `convert_cog_output()` for `response_format=diarized_json`
- [x] 4.5 Ensure `json`/`text` responses still work with `model=gpt-4o-transcribe-diarize` (no segments in body)

## 5. Tests

- [x] 5.1 Unit test: `convert_to_diarized_json` speaker mapping (letters + known names)
- [x] 5.2 Unit test: diarize validation errors (prompt, timestamp_granularities, inconsistent model)
- [x] 5.3 HTTP test: successful `diarized_json` response with mocked Cog
- [x] 5.4 HTTP test: Cog payload contains `diarization: true` and `huggingface_access_token: null`
- [x] 5.5 HTTP test: Cog failure (missing HF token) maps to bridge 500
- [x] 5.6 Run `make -f Makefile.harness check`

## 6. Documentation

- [x] 6.1 Update `docs/DATA_CONTRACTS.md` — `diarized_json` response table
- [x] 6.2 Update `docs/BRIDGE.md` — diarize path uses whisperx `HUGGINGFACE_TOKEN` (no bridge env)
- [x] 6.3 Update README OpenAI STT section with diarization example (`gpt-4o-transcribe-diarize` + `diarized_json`)

## 7. Deploy

- [ ] 7.1 Push Cog + bridge changes → image rebuilds (whisperx-cog + bridge GHCR)
- [ ] 7.2 `kubectl rollout restart` / compose up; smoke test diarized curl against running stack

## 8. Follow-up TODO (not in v1 — do not implement now)

- [x] 8.1 Document v1 limitation in README / `docs/BRIDGE.md`: `known_speaker_references[]` returns 400; link to PLANS.md
- [ ] 8.2 **Future change:** `known_speaker_references[]` — parse reference clips, integrate with WhisperX if speaker-matching API exists, map segments to `known_speaker_names[]` (see `design.md` §Open Questions)
