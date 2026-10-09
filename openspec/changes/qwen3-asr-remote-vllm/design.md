## Context

`predict.py` (`Predictor.predict`) on the qwen path: `whisperx.load_audio` -> VAD chunking (pyannote/silero client-side, `merge_chunks`, chunk_size=30) -> `model.transcribe(audio, batch_size, context)` per batch of windows -> `align_qwen` (Qwen3-ForcedAligner-0.6B, local) -> `diarize` (pyannote, local). Segment timestamps come from the LOCAL VAD, never from the model: that is the exact seam where transcription can be delegated to a remote engine without touching alignment/diarization.

Spike evidence 2026-10-09 (vLLM 0.30.0, weights bind-mounted ro from the models.lock v2 snapshot, `--runner=generate --gpu-memory-utilization 0.45`): multimodal chat payload `{messages:[{role:user,content:[{type:audio_url,audio_url:{url:"data:audio/wav;base64,..."}}]}]}` -> 200, content `language French<asr_text>...`; 10 s mono 16 kHz wav window = 0.43 s; `POST /sleep` 200 in 1.76 s; `POST /wake_up` 200 then transcription 0.92 s (endpoint is `/wake_up`, NOT `/wake`). vLLM 0.30.0 transcription endpoint does NOT support verbose_json for qwen3-asr (`supports_segment_timestamp=False`) — irrelevant here because cog owns timestamps via VAD.

## Goals / Non-Goals

**Goals:**
- Single branching point: the model layer of the qwen path (the per-batch `model.transcribe` equivalent); everything else unchanged
- Remote path fully env-configured: base URL (mandatory in remote mode, NEVER a code constant), timeout, backend selector; invalid selector value = default `local` + warning
- Identical output shape: same `TranscriptionResult` schema (segments start/end/text...) feeding align_qwen/diarize without modification
- Total backward compatibility: `QWEN_BACKEND=local` (or unset, default) = current behavior bit-for-bit; `ENABLE_QWEN=0` degrades both modes to a clean 400 (bridge untouched)
- GPU-free strict TDD: remote behaviors tested against HTTP doubles built as an exact mirror of the vLLM 0.30.0 source code (statuses, error envelopes, fields), never invented
- Repo hygiene: no personal IP or port anywhere in the repo (code, tests, docs, compose, k8s) — enforced mechanically

**Non-Goals:**
- Streaming; server-side timestamps/diarization; default-model change; faster-whisper path changes
- Log masking of the remote URL at runtime (explicitly rejected by the operator: unnecessary)
- Any fallback from remote to local (explicit errors only)

## Decisions

1. **Default backend = `local`** (operator decision 2026-10-09): zero behavioral change at deploy; the deploy env flips to remote when infra is ready
2. **Failures = explicit errors, no fallback** (operator decision 2026-10-09): connection refused / 5xx / unreadable body -> typed `qwen_remote_unavailable` (502 at bridge boundary), logged; never a silent degradation to the local engine
3. **Window concurrency = bounded pool reusing clamped batch_size** (4 default / 8 cap, existing helper); VAD window order preserved in the fused result
4. **Remote contract = multimodal chat/completions with audio_url data URIs** (per-VAD-window, validated by spike; keeps the `context` system message per window, unlike the transcription endpoint which carries no context and forbids verbose_json)
5. **Repo hygiene rule**: no personal IP/port hard-coded ANYWHERE in the repo (code, tests, docs, compose, k8s); documentation examples use a neutral placeholder; enforced by an exhaustive grep gate at check/CI time; the URL itself is freely loggable at runtime

### Test boundaries (TDD, docs/TESTING.md)

- Doubles: a fake vLLM server written as a mirror of the v0.30.0 code path (chat.completion envelope `choices[0].message.content`, 404/502/socket-timeout errors, content with and without the `language X<asr_text>` prefix, structured output, empty content)
- Seams: injectable HTTP client; env resolution (default/override/empty/invalid) unit-tested; `language X<asr_text>` parser unit-tested; concurrency policy unit-tested
- GPU path: manual smoke on the canary pre-merge (full transcribe->align->diarize via remote) + golden set replay in remote mode (transcript hash vs local mode)
- Regression: bridge suite green, `make -f Makefile.harness ci` exit 0, faster-whisper path untouched
