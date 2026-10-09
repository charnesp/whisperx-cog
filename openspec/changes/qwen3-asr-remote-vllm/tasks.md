## 1. Repo leak gate (FIRST — protects every other group)

- [x] 1.1 RED: gate script + test — exhaustive regex scan of the worktree (code, tests, docs, compose, k8s, openspec) finding NO personal IP/port; RFC 2606 hosts (`*.invalid`, `example.test`) required in test fixtures; whitelist exactly `127.0.0.1`, `0.0.0.0`, `localhost`; real RED: gate FAILS if a fixture accidentally carries a concrete host (sabotage-proof: remove a host, the gate must fail)
- [x] 1.2 GREEN: wire the gate into `Makefile.harness` check/ci; `make -f Makefile.harness check` green

## 2. Env resolution + remote config (GPU-free, fail-fast at setup)

- [x] 2.1 RED: `resolve_remote_config()` — `QWEN_BACKEND` default `local` when unset; `remote` accepted; invalid value -> `local` + captured warning; `remote` without `QWEN_REMOTE_BASE_URL` -> typed `QwenRemoteError:` config error naming the variable; `QWEN_REMOTE_MODEL` mandatory in remote mode (same fail-fast); `QWEN_REMOTE_TIMEOUT_S` default 300, non-integer -> error; URL scheme limited to http/https, trailing-slash normalized, `/chat/completions` join tested; NO code constant holding any host/port/model name
- [x] 2.2 GREEN: implement `resolve_remote_config()` reading env at call time (resolved once in `setup()` fail-fast per design decision 9); run `make -f Makefile.harness check`

## 3. Remote client module (GPU-free)

- [x] 3.1 RED: tests for `qwen_remote.py` HTTP double written as a mirror of the vLLM 0.30.0 contract (chat.completion envelope `choices[0].message.content`, error shapes for connection-refused / 5xx / socket timeout / empty content): request body pins `model` from env config, `temperature=0`, bounded `max_tokens`, NEVER `response_format=verbose_json` nor `timestamp_granularities` (negative assertion)
- [x] 3.2 RED: explicit `ProxyHandler({})` decision test (urllib must NOT honor HTTP(S)_PROXY for the internal call) + bandit B310 `# nosec` justified by the scheme validation
- [x] 3.3 GREEN: implement the injectable stdlib-urllib client `transcribe_windows(windows, config, client)` posting one multimodal chat/completions per window (audio_url data URI, wav mono 16 kHz PCM16, deterministic encoder test: header/sample-rate/channels) with the `context` system message in EVERY request; client callable from a thread pool (no shared mutable state); `make -f Makefile.harness check`

## 4. Response parser (GPU-free)

- [x] 4.1 RED: parser tests — `language X<asr_text>text` -> text only; missing `<asr_text>` -> typed parse error (fail explicit, decision B6); `language None<asr_text>` / empty transcription -> empty segment (same rule as local path); empty/missing content -> typed error; multi-choice -> first choice; `finish_reason=="length"` -> typed error (never silently-truncated text)
- [x] 4.2 GREEN: implement the pure parser; run `make -f Makefile.harness check`

## 5. Window concurrency + order + failure semantics (GPU-free)

- [x] 5.1 RED: 3 windows pool 2 -> all transcribed, results ordered by VAD window start regardless of completion order; one window failing -> whole prediction fails with typed error, pending futures cancelled (`cancel_futures=True`), in-flight results discarded; pool never exceeds the clamped batch_size (`<=0` / non-integer via the existing clamp helper); zero retries: a 5xx in the mock -> immediate typed failure
- [x] 5.2 GREEN: implement the bounded-pool orchestrator (concurrency == clamped batch_size); per-request connect+read timeout from `QWEN_REMOTE_TIMEOUT_S`; run `make -f Makefile.harness check`

## 6. predict.py branching

- [x] 6.1 RED (GPU-free): with a mocked model layer — `QWEN_BACKEND=remote` routes the qwen transcription to the remote client and NEVER calls the local `asr_qwen` loader (loader mocked to count calls); `QWEN_BACKEND=local`/unset keeps the current in-process call (identity test on the untouched path); remote failure -> `QwenRemoteError:` typed message with category, no local fallback attempt; setup() fail-fast on broken remote config (test at predictor level); ForcedAligner+pyannote local stages called identically in both modes
- [x] 6.2 GREEN: minimal branching in `predict.py`; remote windows carry their LOCAL VAD start/end (server content never timestamps); `ENABLE_QWEN` gate stays bridge-only (no duplicate gate in cog); run `make -f Makefile.harness check`
- [x] 6.3 BLUE: extract shared helpers, remove duplication; full `check` green

## 7. Regression

- [x] 7.1 Full harness: `make -f Makefile.harness ci` exit 0 (bridge suite included); faster-whisper path untouched

## 8. Compose + k8s env wiring

- [x] 8.1 RED: config-shape test — compose/k8s carry the four env entries (`QWEN_BACKEND`, `QWEN_REMOTE_BASE_URL`, `QWEN_REMOTE_MODEL`, `QWEN_REMOTE_TIMEOUT_S`) with NO concrete host/model value in the files (deploy-time injection), bridge sync check stays green
- [x] 8.2 GREEN: add env entries to docker-compose.yml + k8s/whisperx-stack.yaml; `make -f Makefile.harness smoke` green

## 9. Docs (same cycle, docs-coverage-is-done)

- [x] 9.1 README.md (env table + remote mode usage, placeholder examples), ARCHITECTURE.md, DATA_CONTRACTS.md (request/response shapes + `QwenRemoteError:` taxonomy), BRIDGE.md (contract unchanged note; kill-switch bridge-only), OBSERVABILITY.md (`qwen-remote:` prefix, log contents: host, window count, durations, status), PLANS.md
- [x] 9.2 No personal IP/port/model name in any doc; leak gate covered docs too (group 1)

## 10. GPU smoke (manual, canary pre-merge — operator criteria)

- [x] 10.1 Acceptance criteria (fixed BEFORE run): same FR fixture in local vs remote mode — segment count identical (VAD-owned), word timestamps present in both, forced language respected, WER gap remote-vs-local recorded (no fixed threshold at v1, measured value recorded), cold and warm latency recorded, VRAM peak logged; the model gateway sleep/wake cycle exercised once — RUN 2026-10-09, all criteria met (results in 10.2)
- [x] 10.2 Record results in this file (annotation on the task); commit nothing on GPU hosts

### 10.2 GPU smoke results - 2026-10-09 (the shared GPU, throwaway containers, no stack touched)

Setup: one throwaway vLLM engine serving the provisioned Qwen3-ASR snapshot (no published port, an internal engine network, `--enable-sleep-mode`, dev-mode endpoints) + one throwaway runner reusing the production cog image with the group 6 sources at `/src`. Same FR fixture (60 s, 2 speakers) driven through `Predictor.predict(whisper_model="qwen3-asr", language="fr", align_output=True, diarization=True)` in both modes. Endpoint injected only via `QWEN_REMOTE_BASE_URL` (placeholder `http://<host>:<port>/v1`, never committed).

| Criterion | Local | Remote (cold, first call after wake) | Remote (warm) |
|---|---|---|---|
| Segment count | 3 | 3 | 3 |
| Segment boundaries (s) | 0.031-26.305 / 26.39-48.378 / 50.825-60.038 | identical to local | identical to local |
| Word timestamps | 142 words | 150 words | 150 words |
| Detected language | fr | fr | fr |
| Speakers (pyannote, local) | SPEAKER_00, SPEAKER_01 | same | same |
| predict() wall time | 9.40 s (transcribe 1.90 s, align 1.31 s) | 5.65 s (transcribe 1.55 s, align 2.00 s) | 5.46 s |
| Runner VRAM peak (reserved) | 4.63 GB | 2.05 GB | 2.05 GB |

- Segment count and boundaries identical in both modes (timestamps owned by the local VAD, as designed); alignment and diarization ran identically on both paths.
- WER remote-vs-local (local transcript as reference, normalised word-level Levenshtein): **7.04 %** (10 substitutions / 142 reference words); identical value cold and warm. Recorded, no threshold at v1.
- Engine-level: the remote engine woke and slept once during the run (both transitions returned 200) and the cold run was served by the woken engine; absolute GPU figures are deliberately not recorded here.
- Deviation recorded: the sleep/wake cycle was exercised on the throwaway engine directly (the model gateway's wrapper uses the same endpoints). Qwen3-ASR is NOT yet registered in the model gateway production config, so the model gateway routing for this model remains to be validated at deployment time (out of scope here, no stack touched).
- Cleanup: both throwaway containers removed, the shared GPU returned to its pre-run footing, the production stack stayed healthy (health-check 200) and no existing engine was touched.

## 11. Review + security audit closing

- [ ] 11.1 Deep review by 2 read-only subagents (code + usage), findings fixed by a dedicated fix subagent, confirmation re-review
- [ ] 11.2 Security audit (chantier gate): leak gate green (group 1), no secrets in env docs or committed fixtures, hotwords/audio never logged, remote errors explicit, zero-retry posture documented; report committed with the change
