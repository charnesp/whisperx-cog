## 1. Remote client module (GPU-free)

- [ ] 1.1 RED: tests for `qwen_remote.py` HTTP double written as a mirror of the vLLM 0.30.0 contract (chat.completion envelope, error shapes for connection-refused / 5xx / socket timeout / empty content) — real RED failures cited in commit
- [ ] 1.2 GREEN: implement the injectable HTTP client (stdlib urllib wrapper, no new dependency): single function `transcribe_windows(windows, config, client)` posting one multimodal chat/completions per window (audio_url data URI, wav mono 16 kHz) with the `context` system message in EVERY request
- [ ] 1.3 BLUE: deduplicate payload construction helpers; `make -f Makefile.harness check` green

## 2. Env resolution (GPU-free)

- [ ] 2.1 RED: `resolve_remote_config()` — `QWEN_BACKEND` default `local` when unset; `remote` accepted; invalid value -> `local` + captured warning; `remote` without `QWEN_REMOTE_BASE_URL` -> explicit configuration error carrying the variable name; `QWEN_REMOTE_TIMEOUT_S` default 300, invalid -> error; NO code constant holding any host/port
- [ ] 2.2 GREEN: implement `resolve_remote_config()` reading env at call time (never at import); run `make -f Makefile.harness check`

## 3. Response parser (GPU-free)

- [ ] 3.1 RED: parser tests — content with `language X<asr_text>text` -> text only; content without the prefix (structured output) -> text as-is; empty/missing content -> typed error; multi-choice response -> first choice, other indexes ignored (proven against vllm source paths)
- [ ] 3.2 GREEN: implement the pure parser; run `make -f Makefile.harness check`

## 4. Window concurrency + order (GPU-free)

- [ ] 4.1 RED: 3 windows with pool 2 -> all transcribed, results ordered by the VAD window order (start values), one window failing/timeouts -> typed `qwen_remote_timeout` error for the whole batch; pool never exceeds the clamped batch_size
- [ ] 4.2 GREEN: implement the bounded-pool orchestrator (concurrency == clamped batch_size, helper from qwen3-asr-backend change); run `make -f Makefile.harness check`

## 5. predict.py branching

- [ ] 5.1 RED (GPU-free): with a mocked model layer, `QWEN_BACKEND=remote` routes the qwen transcription to the remote client; `QWEN_BACKEND=local`/unset keeps the current in-process call (identity test on the untouched path); remote failure -> explicit error, no local fallback attempt
- [ ] 5.2 GREEN: minimal branching in `predict.py` reusing `align_qwen`/`diarize` unchanged; remote windows keep their LOCAL VAD start/end (server content never timestamps)
- [ ] 5.3 BLUE: extract shared helpers, remove duplication; run `make -f Makefile.harness check`

## 6. Regression + repo hygiene gate

- [ ] 6.1 Exhaustive-repo grep gate: no personal IP or port anywhere (code/tests/docs/compose/k8s); placeholders in docs use a neutral form; wire into `Makefile.harness` check/ci
- [ ] 6.2 Full harness: `make -f Makefile.harness ci` exit 0 (bridge suite included); faster-whisper path untouched

## 7. Compose + k8s env wiring

- [ ] 7.1 RED: config-shape test — compose/k8s carry the three env entries (`QWEN_BACKEND`, `QWEN_REMOTE_BASE_URL`, `QWEN_REMOTE_TIMEOUT_S`) with NO concrete host value in the file (env indirection), bridge sync check stays green
- [ ] 7.2 GREEN: add env entries to docker-compose.yml + k8s/whisperx-stack.yaml (values injected at deploy time); `make -f Makefile.harness smoke` green

## 8. Docs (same cycle, docs-coverage-is-done)

- [ ] 8.1 README.md (env table + remote mode usage), ARCHITECTURE.md (component/flow), DATA_CONTRACTS.md (remote request/response shapes + error codes), BRIDGE.md (unchanged contract note), OBSERVABILITY.md (`qwen-remote:` log prefix, error taxonomy), PLANS.md (change status)
- [ ] 8.2 No personal IP/port in any doc; examples use the neutral placeholder

## 9. GPU smoke (manual, canary pre-merge)

- [ ] 9.1 Full chain via remote on a real FR clip: transcribe (remote) -> align_qwen -> diarize; word timestamps present; VRAM peak logged; compare transcript hash vs local mode replay
- [ ] 9.2 Record results in this file (annotation on the task), commit nothing on GPU hosts

## 10. Review + security audit closing

- [ ] 10.1 Deep review by 2 read-only subagents (code + usage), findings fixed by a dedicated fix subagent, confirmation re-review
- [ ] 10.2 Security audit (chantier gate): repo leak gate green (step 6.1), no secrets in env docs, hotwords/audio never logged, remote errors explicit; report committed with the change
