> superseded 2026-10-20 (fix/qwen-remote-quiet-chunking) : plus de plafond de tokens, découpage aux points calmes, reprise bornée

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
2. **Failures = explicit errors, no fallback** (operator decision 2026-10-09): connection refused / 5xx / unreadable body -> typed `QwenRemoteError:` (category; HTTP 500 `server_error` on the OpenAI STT surface, 502 on the `/predictions` proxy), logged; never a silent degradation to the local engine
3. **Window concurrency = bounded pool reusing clamped batch_size** (4 default / 8 cap, existing helper); window order preserved in the fused result (VAD order at v1, quiet-point plan order since 2026-10-20)
4. **Remote contract = multimodal chat/completions with audio_url data URIs** (one request per quiet-point window since 2026-10-20, per-VAD-window at v1; validated by spike; keeps the `context` system message per window, unlike the transcription endpoint which carries no context and forbids verbose_json)
5. **Repo hygiene rule**: no personal IP/port hard-coded ANYWHERE in the repo (code, tests, docs, compose, k8s); documentation examples use a neutral placeholder; enforced by an exhaustive grep gate at check/CI time; the URL itself is freely loggable at runtime
6. **Remote request identity** (B1): `QWEN_REMOTE_MODEL` env (mandatory in remote mode, no default name) selects the gateway model alias; every request pins `temperature=0` and carries NO `max_tokens` at all (operator decision 2026-10-20: a 30 s window emits 109-137 tokens, the former 120-token ceiling caused `finish_reason=="length"` failures — no replacement value, no env variable); `finish_reason=="length"` stays a typed error (never silently truncated text) and is now recoverable per window
7. **Remote timeout and bounded recovery** (B4, rewritten 2026-10-20 after the adversarial review): `QWEN_REMOTE_TIMEOUT_S` (default 300) is a per-request connect+read timeout. A failing window gets ONE retry. Only a CONTENT failure (unparseable or empty reply, `finish_reason=length`, content-related HTTP 4xx/5xx) may then be repaired by splitting the window in two halves at its quietest point, each half resolved the same way, to a maximum split depth of 2. A `timeout` or `connection` failure is NEVER split: a shorter window does not restore a dead network or a server that stopped answering, it only multiplied the wait (the former rule cost 6 x timeout per window). Request-identity statuses (400/401/403/404/405/422/429) are never split either. Whatever the shape of the failure tree, ONE INPUT window spends at most `REMOTE_MAX_ATTEMPTS_PER_WINDOW` (10, the measured cost of a complete depth-2 recovery) HTTP attempts. A window whose peak amplitude is below `SILENCE_PEAK_THRESHOLD` (0.003) sends no request at all and yields an empty text. Only a window that cannot be saved that way fails the whole prediction (pending futures cancelled with `cancel_futures=True`, in-flight requests completed but discarded). No local fallback, no partial fused output.
7b. **Quiet-point chunking** (superseded 2026-10-20): the remote path cuts windows with the method documented by vLLM ("Chunking Long Audio for Transcription") — max 30 s, cut on the quietest point of the last 1 s, RMS energy over 1600-sample (~100 ms at 16 kHz) frames, full coverage without overlap, absolute offsets — instead of pyannote VAD `merge_chunks`; the splitter is a pure numpy module (`qwen_split.py`) importable without GPU
8. **Language forcing** (B6): the client `language` param is carried to the remote path as a system-message instruction (chat/completions contract; assistant-prefill behavior of the local qwen-asr pipeline is NOT assumed to exist server-side); the parser accepts the same reply shapes as the spike; window language disagreement follows the local-mode rule (aggregates exactly as the local path does today, proven by a parity test at the canary)
9. **setup() semantics** (B3): `QWEN_BACKEND` and remote config are resolved ONCE in `setup()` with fail-fast on remote-mode misconfiguration (missing URL/model -> typed error at boot, not at first client); in remote mode the local qwen-ASR loader is NEVER invoked (local weights stay on disk untouched) while ForcedAligner and pyannote remain local as today; the `ENABLE_QWEN` gate stays where it already lives (bridge primary + the pre-existing cog `qwen_enabled()` defense-in-depth check, both unchanged) — this change adds NO new gate in cog (B2, operator decision: no added gate)
10. **Client-facing error contract** (B5): remote failures surface as stable grep-able messages prefixed `QwenRemoteError:` with a category (config/connection/timeout/http_status/parse); messages never carry the full URL (credentials in userinfo possible), the audio base64, or the whole reply body (truncated)

### Test boundaries (TDD, docs/TESTING.md)

- Doubles: a fake vLLM server written as a mirror of the v0.30.0 code path (chat.completion envelope `choices[0].message.content`, 404/502/socket-timeout errors, content with and without the `language X<asr_text>` prefix, structured output, empty content)
- Seams: injectable HTTP client; env resolution (default/override/empty/invalid) unit-tested; `language X<asr_text>` parser unit-tested; concurrency policy unit-tested
- GPU path: manual smoke on the canary pre-merge (full transcribe->align->diarize via remote) + golden set replay in remote mode (transcript hash vs local mode)
- Regression: bridge suite green, `make -f Makefile.harness ci` exit 0, faster-whisper path untouched
