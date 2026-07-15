## Why

whisperx-cog already performs speaker diarization via WhisperX (`predict.py`, `diarization: true`), but the OpenAI-compatible STT endpoint (`POST /v1/audio/transcriptions`) hard-codes `diarization: false` and does not support `response_format=diarized_json` or the `gpt-4o-transcribe-diarize` model alias. Clients using the OpenAI Speech-to-Text API for speaker-aware transcripts cannot use our bridge without a custom adapter, despite the underlying capability already existing in the stack.

## What Changes

- Add `response_format=diarized_json` on `POST /v1/audio/transcriptions`, returning OpenAI `TranscriptionDiarized` shape (`duration`, `segments[]`, `task`, `text`, optional `usage`)
- Add model alias `gpt-4o-transcribe-diarize` → WhisperX pipeline with `diarization: true` (maps to `large-v3-turbo` backend, same as `whisper-1`)
- When diarization is requested, enable Cog `diarization: true` (token resolved inside whisperx via existing `HUGGINGFACE_TOKEN` env — see design §3)
- Map WhisperX speaker labels (`SPEAKER_00`, `SPEAKER_01`, …) to OpenAI labels (`A`, `B`, …); support optional `known_speaker_names[]` when provided
- Accept `chunking_strategy` on diarize requests (validate presence for OpenAI SDK compat; no-op internally — WhisperX handles long audio via VAD)
- Reject incompatible params on diarize path per OpenAI rules: `prompt`, `timestamp_granularities`, `include[]=logprobs` → 400
- Surface diarization failure when `HUGGINGFACE_TOKEN` is missing on whisperx (Cog `status: failed` → bridge HTTP 500)
- Extend unit tests and docs (`DATA_CONTRACTS.md`, `BRIDGE.md`, README OpenAI section)
- No breaking changes to existing non-diarize OpenAI STT behavior or Replicate `/predictions` API

## Capabilities

### New Capabilities

<!-- None — extends existing openai-stt-api capability -->

### Modified Capabilities

- `openai-stt-api`: Add diarized transcription — `diarized_json` response format, `gpt-4o-transcribe-diarize` model, Cog diarization integration, speaker label mapping, HF token requirement, and diarize-specific validation

## Impact

- **Code:** `bridge/openai_compat.py` (validation, Cog input, `convert_to_diarized_json`), `predict.py` (env fallback for HF token), `tests/test_openai_stt.py`
- **Infrastructure:** no change — `HUGGINGFACE_TOKEN` already on whisperx container in k8s/compose
- **Docs:** `docs/DATA_CONTRACTS.md`, `docs/BRIDGE.md`, `README.md`
- **Dual-copy / GHCR:** Bridge image rebuild after `bridge/*.py` changes; `make check`
- **Out of scope (v1):** SSE streaming (`stream=true`), `known_speaker_references[]` audio matching, `gpt-4o-transcribe` / `gpt-4o-mini-transcribe` non-diarize models

## Follow-up TODO (post-v1)

- **`known_speaker_references[]`** — OpenAI diarize clients can send up to 4 reference audio clips (data URLs) paired with `known_speaker_names[]` to label segments with known voices. v1 rejects this param with HTTP 400. **TODO:** implement when WhisperX/Cog exposes speaker-matching from reference clips (see `design.md` §Open Questions, `PLANS.md` tech debt).
