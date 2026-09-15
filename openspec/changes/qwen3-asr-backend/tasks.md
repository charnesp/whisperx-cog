## 1. Bridge — model alias and kill-switch

- [ ] 1.1 RED: unit test asserting `MODEL_MAP` contains `"qwen3-asr": "qwen3-asr"` and that an unknown model still returns 400 (`invalid_request_error`)
- [ ] 1.2 GREEN: add the `qwen3-asr` entry to `MODEL_MAP` in `bridge/openai_compat.py`
- [ ] 1.3 RED: unit test — `model=qwen3-asr` with `ENABLE_QWEN` unset/empty → HTTP 400 with feature-disabled message; with `ENABLE_QWEN=1` → accepted
- [ ] 1.4 GREEN: implement the `ENABLE_QWEN` gate (env read at request time, not import time) and run `make -f Makefile.harness check`

## 2. Bridge — hotwords and batch_size passthrough

- [ ] 2.1 RED: unit test — `build_cog_input()` with `model=qwen3-asr` + `hotwords="Backblaze, Supabase"` → Cog input contains `hotwords` with that string; with `model=whisper-1` → `hotwords: None` (unchanged)
- [ ] 2.2 GREEN: implement qwen-only `hotwords` passthrough in `build_cog_input()`
- [ ] 2.3 RED: unit test — `batch_size` present in Cog input only when the client provides it; absent otherwise (both qwen and whisper models)
- [ ] 2.4 GREEN: make `batch_size` optional in `build_cog_input()` (drop the hard-coded `64`); run `make -f Makefile.harness check`

## 3. Cog — context formatting helpers (GPU-free)

- [ ] 3.1 RED: unit test for a pure `format_qwen_context(hotwords)` helper — template assembly identical to the validated live template; >2000 chars → truncated to cap with a truncation flag returned (lengths only)
- [ ] 3.2 GREEN: implement `format_qwen_context()` (no hotword content ever logged — assert log calls carry lengths only)
- [ ] 3.3 RED: unit test for a pure `clamp_batch_size(value, default)` helper — None → 4 (QWEN_DEFAULT_BATCH), explicit 12 → 8, explicit 0/negative → 4, explicit 6 → 6
- [ ] 3.4 GREEN: implement the clamp helper; run `make -f Makefile.harness check`

## 4. Cog — predict.py qwen branch (GPU path, manual smoke unless GPU CI is scoped)

- [ ] 4.1 Add `qwen3-asr` to `whisper_model` choices in `predict.py` `Input`
- [ ] 4.2 Branch model loading: `qwen3-asr` → load via the forked `whisperx.asr_qwen` with explicit `qwen_dtype="float16"`; faster-whisper load path untouched
- [ ] 4.3 On the qwen path: apply `QWEN_DEFAULT_BATCH`/clamp (helper from 3.3), skip the `detect_language` loop (pass provided `language` as-is), forward `context` to `QwenAsrPipeline.transcribe` per batch
- [ ] 4.4 Alignment on the qwen path: `Qwen/Qwen3-ForcedAligner-0.6B` from baked `/models`; pyannote diarization stage unchanged (output schema identical)
- [ ] 4.5 Boot fail-fast: missing baked qwen weights + `HF_HUB_OFFLINE=1` → clear RuntimeError at model load
- [ ] 4.6 Manual smoke on GPU: one short FR clip end-to-end (transcribe + align + diarize), word timestamps present, VRAM logged

## 5. Dependencies and baking

- [ ] 5.1 `requirements.txt`: pin `whisperx @ git+https://github.com/charnesp/whisperX@c49b26379f40863767e3c42d9afed5dc4221f54f`, add `qwen-asr==0.0.6 --no-deps`, explicit minimal deps (`transformers==4.57.6`, `soundfile`, `librosa`); verify no gradio/flask/vllm in the resolved tree
- [ ] 5.2 `models.lock`: pin HF revisions for `Qwen/Qwen3-ASR-1.7B` and `Qwen/Qwen3-ForcedAligner-0.6B`
- [ ] 5.3 `cog.yaml`: bake both snapshots into `/models` (wget + `test -s` on weights, one command per run item, same pattern as turbo); add `HF_HUB_OFFLINE=1` to the Cog environment
- [ ] 5.4 Build the Cog image; verify `/models` contents and image size delta (~4–5 GB)

## 6. Golden set (GPU, scripted, replayable)

- [ ] 6.1 Script a replayable golden-set harness: fixed FR extract + 2026-09-02 real meeting extract; turbo baseline, qwen baseline, qwen+hotwords runs
- [ ] 6.2 faster-whisper regression: `tiny`, `large-v3`, `large-v3-turbo` outputs bit-identical to pre-change
- [ ] 6.3 qwen baseline vs qwen+hotwords: proper-noun recall AND false positives (segments that should not contain the hotword names — no hallucinated insertions)
- [ ] 6.4 hotwords absent → qwen output bit-identical to the 2026-09-15 baseline run
- [ ] 6.5 End-to-end: word-level timestamps present; `assign_word_speakers` receives words from the Qwen ForcedAligner; diarized output schema unchanged
- [ ] 6.6 Peak VRAM logged < 5.5 GB at batch 4; RTFx ~50 on the 4080; record results in the PR

## 7. Documentation

- [ ] 7.1 `docs/DATA_CONTRACTS.md`: `qwen3-asr` model row, hotwords→context semantics, per-model default batch
- [ ] 7.2 `docs/BRIDGE.md`: `ENABLE_QWEN` kill-switch, batch_size passthrough rule, no-hotwords-in-logs rule
- [ ] 7.3 README OpenAI STT section: `model=qwen3-asr` example with hotwords
- [ ] 7.4 `make -f Makefile.harness check` green

## 8. Deploy (requires explicit go-ahead — out of this change's scope)

- [ ] 8.1 Push image tag; redeploy; rollback plan = previous tag (< 5 min)
- [ ] 8.2 Transcribe 1–2 real meetings in duplicate (turbo vs qwen+hotwords) before any default change; verify `REDIS_SOCKET_TIMEOUT` holds for 90-min meetings