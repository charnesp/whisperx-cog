# Security audit — change `qwen3-asr-remote-vllm` (group 11.2)

Operator gate for the chantier, run 2026-10-09 on the change branch
`feat/qwen3-asr-remote-vllm`. Every claim below is backed by a command that was
executed, not by intent. Nothing here was run on a GPU host and no production
stack was touched.

## 1. Leak gate (no personal IP / port in the repository)

- `python scripts/leak_gate.py .` → `OK leak_gate: no personal IP/port in the repository`, rc 0.
- The gate is wired into `make -f Makefile.harness check` (`$(PYTHON) scripts/leak_gate.py .`), so a leak fails the harness.
- Sabotage proof (fixtures assembled at runtime, no literal address in the test file): a bare `service:port` (`<label>:8000`) inside a `.env.example` and a `.md` file → `FAIL leak_gate: 2 finding(s)`, rc 1.
- Coverage extension during the fix cycle: the gate now scans `.env.example` (`.example` suffix) and flags single-label `service:port`, while whitelisting loopback, `localhost`, RFC 2606 names, `<host>`/`<port>` placeholders and internal service labels.

## 2. No concrete address or model committed

- `docker-compose.yml`: `QWEN_BACKEND=${QWEN_BACKEND:-local}`, `QWEN_REMOTE_BASE_URL` / `QWEN_REMOTE_MODEL` empty by default, `QWEN_REMOTE_TIMEOUT_S=${QWEN_REMOTE_TIMEOUT_S:-300}` — deployment-injected.
- `k8s/whisperx-stack.yaml`: the four variables are declared with empty placeholders (no literal value, not even for the backend).
- `.env.example`: templates only (`http://<host>:<port>/v1`, `<model-name>`).
- Docs name no host, port or model: the remote endpoint is always a placeholder.

## 3. No secret in the change

- a diff scan over the whole change (against the change base) for `hf_…`, `sk-…`, `password`, `passwd`, `api_key=`, `Bearer …`: only doc/source comments describing the sanitizer (no value).
- No token is read from anywhere but the environment; the diarization token comes from the container env, never from the repository.
- No committed fixture carries audio, transcript or speaker data.

## 4. Audio, hotwords and context are never logged

- The remote path emits exactly two log records: one `info` (`qwen_remote_client.py`, `qwen-remote:` line) and one `warning` (invalid `QWEN_BACKEND` in `qwen_remote.py`).
- The record is `qwen-remote: host=%s windows=%d status=%s duration_s=%.3f`; the host comes from `urlsplit(...).hostname`, i.e. **host only** — never the port, never a full URL, never userinfo.
- Neither the audio payload, the hotwords/context string, nor the transcribed text reaches any logger; covered by a dedicated test that fails if a URL, port, credential or payload appears in the captured record.
- The URL with credentials can never appear in an error either: transport reasons go through a sanitizer that strips URL and userinfo (test uses a `user:pass@host` base URL and asserts the credentials are absent).

## 5. Remote failures are explicit

- Typed taxonomy `QwenRemoteError: config|connection|timeout|http_status|parse` surfaced verbatim to the caller; no generic exception, no silent `None`.
- `http_status` bodies are truncated (96 bytes); `config` errors name the missing variable.
- No silent local fallback exists in code, and the guarantee is now tested for real (the earlier test asserted on a list nothing ever appended to — it was replaced in this cycle by a test that observes the actual local-loader seam).

## 6. Zero-retry posture

- `docs/DATA_CONTRACTS.md`: "zero retries and no silent fallback to the local engine" at v1; the spec carries the same statement.
- Implemented: one HTTP attempt per window, `cancel_futures=True` on failure, no retry loop; a 5xx is a single attempt (asserted).

## 7. Dependency triage (accepted risk)

- `PYSEC-2026-4174` (transformers 4.57.6) is ignored in `Makefile.harness` with a dated comment: the dependency is frozen, no compatible fix is available, to be revisited at the next refreeze. Recorded here as an explicitly accepted risk, not an oversight.

## 8. Review trail

- Group 11.1: two read-only reviewers (code/contract and usage/integration), a dedicated fix pass in TDD cycles, then a read-only confirmation re-review that executed every claim.
- Confirmation verdict: all findings fixed, no regression, no weakened assertion, no residual doc↔code lie blocking the merge. Three minor reserves were raised; the parser edge and the timeout-sizing wording were fixed afterwards, the third (pool-level guard-rail test) is documented as a guard-rail whose non-vacuous proof lives at the predict level.

## 9. Residual risk carried to deployment

- Qwen3-ASR is **not registered in the model gateway production config**: the GPU smoke exercised a throwaway engine directly. Validating `cog → the model gateway → vLLM` routing (alias + wake path) is a deployment prerequisite, tracked on task 10.1.
- `QWEN_REMOTE_TIMEOUT_S` is a per-operation socket timeout, not a wall-clock bound; documented, with sizing guidance against the bridge timeout.

Verdict: **gate passed** — no leak, no committed address, no secret, no payload logging, explicit errors, zero-retry documented, reviews and confirmation completed. The two residual items above are deployment-time actions, not code defects.
