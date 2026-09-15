## Why

whisperx-cog currently exposes three faster-whisper backends (`tiny`, `large-v3`, `large-v3-turbo`) on `POST /v1/audio/transcriptions` via `bridge/openai_compat.py` `MODEL_MAP`. Live benchmarks on RTX 4080 (2026-09-15, 36-minute meeting recording) show **Qwen3-ASR-1.7B** (HF `Qwen/Qwen3-ASR-1.7B`, Apache-2.0, ~3.4 GB fp16) reaching **RTFx 52 with a 4.99 GB VRAM peak at batch_size=4**, comparable throughput to `large-v3-turbo` at a fraction of the memory. More importantly, Qwen3-ASR natively supports a **`context`** system message (per-batch, no accumulation, empty string = neutral), which proved **4x better proper-noun recall** than the faster-whisper `hotwords` parameter on the same recording (Backblaze 3→12 occurrences, Supabase 0→3; faster-whisper hotwords had no measurable effect). Exposing this backend through the existing bridge parameter surface gives clients a drop-in accuracy upgrade for domain vocabulary without any API change.

## What Changes

- Add `model=qwen3-asr` to `MODEL_MAP` in `bridge/openai_compat.py`; unknown models keep returning 400 (whitelist unchanged)
- Route the existing `hotwords` bridge parameter to the Qwen `context` system message when `model=qwen3-asr` (today `hotwords` is hard-coded `None` in `build_cog_input`); for `large-v3-turbo`/`large-v3`/`tiny`, `hotwords` keeps its faster-whisper `initial_prompt`-adjacent semantics with the whisper path unchanged
- Context template (participants + technical vocabulary, identical to the validated live template), capped at ~2000 chars with a truncation log line; hotwords are never logged (client proper nouns)
- Model-dependent default batch: `qwen3-asr` without explicit `batch_size` → `QWEN_DEFAULT_BATCH = 4` (measured OOM when all VAD segments are batched at once on the shared 16 GB GPU); explicit values are clamped (cap 8). The bridge passes `batch_size` only when the client provides it
- Language: `qwen3-asr` skips the whisper `detect_language` loop (Qwen handles its own detection); a provided `language` is passed through as-is
- Alignment: the qwen path uses **Qwen3-ForcedAligner-0.6B** (HF `Qwen/Qwen3-ForcedAligner-0.6B`), not the wav2vec2 aligner (incompatible with qwen outputs); downstream pyannote diarization is unchanged (256-dim speaker embeddings, identical output format)
- Dependencies (`requirements.txt`): pin the forked pipeline `whisperx @ git+https://github.com/charnesp/whisperX@c49b26379f40863767e3c42d9afed5dc4221f54f` (upstream PR m-bain/whisperX#1401 + context patch, branch `qwen3-asr`), add `qwen-asr==0.0.6 --no-deps` plus explicit minimal deps (`transformers==4.57.6` already in prod, `soundfile`, `librosa`). **gradio/flask/vllm are forbidden** (transitive deps of qwen-asr: CVE surface, image size, pydantic/starlette pin conflicts)
- Model baking: both Qwen snapshots baked into `/models` via `cog.yaml` run steps (same wget + `test -s` pattern as the faster-whisper turbo bake), HF revisions pinned in a `models.lock` file, `HF_HUB_OFFLINE=1`, fail-fast at boot if weights are missing
- Kill-switch: `ENABLE_QWEN` env var — when disabled, `model=qwen3-asr` returns a clean HTTP 400 with no redeploy
- Deployment guardrails: the model parameter itself is the canary (existing clients sending `whisper-1` → `large-v3-turbo` see no change); rollback = previous image tag; 1–2 real meetings transcribed in duplicate (turbo vs qwen+hotwords) before any default change
- Unit tests (GPU-free) and docs (`README.md`, `docs/DATA_CONTRACTS.md`, `docs/BRIDGE.md`)
- No breaking change to the whisper path: faster-whisper bit-identical regression is an acceptance criterion

## Capabilities

### New Capabilities

<!-- None — extends existing openai-stt-api capability -->

### Modified Capabilities

- `openai-stt-api`: Add the `qwen3-asr` backend — model alias, hotwords→context routing, per-model default batch with clamping, language-detection bypass, Qwen forced aligner, kill-switch, and baked model weights

## Impact

- **Code:** `bridge/openai_compat.py` (`MODEL_MAP`, `build_cog_input()` hotwords passthrough, optional `batch_size` passthrough), `predict.py` (qwen branch: model load with explicit fp16 dtype, `QWEN_DEFAULT_BATCH`, language-loop skip, Qwen aligner), `model_paths.py` (qwen snapshot resolution), `tests/test_openai_stt.py`
- **Dependencies:** `requirements.txt` — pinned whisperx fork + `qwen-asr==0.0.6 --no-deps` + explicit minimal deps; **no gradio/flask/vllm**
- **Infrastructure:** `cog.yaml` bakes the two Qwen snapshots into `/models` (image grows ~4–5 GB); `models.lock` pins HF revisions; no k8s/compose volume changes
- **Docs:** `README.md`, `docs/DATA_CONTRACTS.md`, `docs/BRIDGE.md`
- **Out of scope (v1):** streaming, changing the default model for existing clients, qwen-specific speaker profiles, upstream PR #1401 merge handling (tracked as follow-up)

## Risks

- **Unmerged dependency chain (highest risk):** upstream PR m-bain/whisperX#1401 is still open + `qwen-asr==0.0.6` + `transformers` pins — the pinned fork commit, `--no-deps`, and `models.lock` exist precisely to freeze this chain; re-evaluate on upstream merge
- **dtype:** Qwen3-ASR must load with explicit `fp16` — default fp32 measures ~10 GB VRAM vs ~5 GB fp16
- **Concurrency:** cog runs 1 prediction at a time; two simultaneous pipelines (qwen 5 GB + whisper 4–6 GB + 2.4 GB resident) would brush the 16 GB limit — rely on existing sequential bridge/redis locking
- **Timeouts:** redis worker timeouts (`REDIS_SOCKET_TIMEOUT=120`) vs 90-minute meetings — 36 min measured ≈ 2 min of transcription (~44.8 s), so a 90-minute meeting stays within budget, but verify against the deployed timeout settings
- **Context false positives:** hotwords may be hallucinated into segments unrelated to them — measured explicitly in the golden set (recall AND false positives)
- **Secrets:** hotwords contain client proper nouns — never log them in bridge logs