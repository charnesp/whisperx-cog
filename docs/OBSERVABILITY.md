# Observability

How to observe whisperx-cog in development and production. See [ARCHITECTURE.md](./ARCHITECTURE.md) for component layout.

## Health endpoints

| Endpoint | Port | Auth | Meaning |
|----------|------|------|---------|
| `GET /health` | 8080 (bridge) | No | Bridge liveness — HTTP server up |
| `GET /health-check` | 8080 → Cog 5000 | No | Cog readiness — inspect JSON `status` |

### Cog `status` values

| Value | Meaning |
|-------|---------|
| `READY` | Accepting predictions |
| `STARTING` | Model setup in progress |
| `BUSY` | Prediction running |
| `UNHEALTHY` | Predictor `healthcheck()` failed (CUDA unavailable in this project) |
| `SETUP_FAILED`, `DEFUNCT` | Not operational |

**Kubernetes:** liveness = bridge `/health`; readiness = bridge `/health-check` (proxied).  
**Docker Compose:** container healthcheck = bridge `/health` only; poll `/health-check` manually for Cog.

## Structured logs — bridge

Filter by prefix:

```bash
# External API traffic
kubectl logs <pod> -c bridge | grep '\[bridge ext\]'
docker compose logs bridge 2>&1 | grep '\[bridge ext\]'

# Redis, Cog proxy, internal webhook
kubectl logs <pod> -c bridge | grep '\[bridge int\]'
```

Key transitions logged:

- `GET cache hit` / `GET cache miss` / `GET redis failure`
- `webhook ok` / `webhook redis_set_failed`
- `cog proxy upstream error`

## Structured logs — Cog (whisperx)

Cog stdout/stderr from the whisperx container. Prediction failures include sanitized error messages (see `json_sanitize.sanitize_error_message`).

Common failure signature:

```
Out of range float values are not JSON compliant: nan
```

→ output contained NaN before sanitization fix; verify `sanitize_for_json` runs on all return paths.

### Remote qwen3-asr logs (`qwen-remote:`)

When `QWEN_BACKEND=remote`, the whisperx container logs **one structured record per remote transcription** under the **`qwen-remote:`** prefix, carrying the remote **host** (host only — never the full URL, port or credentials), the **window count**, the **total duration** and the **status** (`ok`/`failed`). Audio payloads and hotword/context content are **never** logged (the same no-content rule as the local qwen context path). Remote failures additionally surface the typed `QwenRemoteError:` category (`config`/`connection`/`timeout`/`http_status`/`parse` — see [DATA_CONTRACTS.md](./DATA_CONTRACTS.md)).

```bash
# Remote qwen3-asr activity
docker compose logs whisperx 2>&1 | grep 'qwen-remote:'
kubectl logs <pod> -c whisperx | grep 'qwen-remote:'
```

`<pod>` is any pod carrying the `app: whisperx` label.

### Switching local ↔ remote (runbook)

`QWEN_BACKEND` is resolved **once at boot** (`Predictor.setup()` caches the config in `predict.py`), **not** per request — flipping it **requires a container restart** (`docker compose up -d --force-recreate whisperx` or `kubectl rollout restart deployment/whisperx-stack`). Reverting is the same operation with `QWEN_BACKEND=local`.

- **Fail-fast blast radius:** a broken REMOTE config (missing `QWEN_REMOTE_BASE_URL` / `QWEN_REMOTE_MODEL`, non-http(s) scheme) makes `setup()` raise, so the **whole whisperx container fails to start** — the faster-whisper models are down too. Validate both remote vars before flipping the backend.
- **Timeout sizing:** keep `QWEN_REMOTE_TIMEOUT_S` (cog) **below** the bridge's `OPENAI_STT_TIMEOUT_SECONDS` — otherwise the bridge 504 fires before cog's typed `QwenRemoteError:` can surface. Both default to 300, i.e. equal by default: plan headroom (e.g. 240 vs 300) before relying on the typed error.
- **Timeout semantics:** `QWEN_REMOTE_TIMEOUT_S` is a **per-operation socket timeout** (applied to connect, then read), not a wall-clock bound on a whole transcription: a slow multi-window batch may exceed it in total. A cold model wake \(tens of seconds\) is one idle gap absorbed by the per-request value.

Diagnose by `QwenRemoteError:` category:

| Category | Meaning / first check |
|----------|-----------------------|
| `config` | remote vars missing/invalid; the container likely never started — read the boot logs |
| `connection` | host/port/network unreachable from the pod — check the service name and DNS |
| `timeout` | engine silent past `QWEN_REMOTE_TIMEOUT_S` — check engine and gateway load and compare with `OPENAI_STT_TIMEOUT_SECONDS` |
| `http_status` | engine answered non-2xx (e.g. mid-swap) — read the engine logs |
| `parse` | reply unusable (`language X` prefix without `<asr_text>`, empty, `finish_reason=length`) — check the engine model/prompt |


## Metrics

No Prometheus/OpenTelemetry in this repo today. Operational signals:

- Cog `metrics.predict_time` in prediction JSON (when present)
- Bridge log lines include `body_bytes`, `response_bytes`, `prediction_id`

## Troubleshooting checklist

1. **`GET /health-check` → UNHEALTHY** — GPU missing or CUDA broken; check `nvidia-smi` in whisperx container.
2. **`GET /predictions/<id>` empty after success** — client supplied own webhook; use webhook URL or omit it for Redis cache.
3. **Webhook 503 `redis_set_failed`** — Redis down or payload too large; check bridge `[bridge int]` logs.
4. **Bridge / k8s drift** — run `python3 scripts/check-bridge-sync.py` (k8s must reference the GHCR bridge image).

## Harness verification (no GPU)

```bash
make -f Makefile.harness smoke   # sync + compile + yaml
make -f Makefile.harness check   # + unit tests (strict TDD gate — see docs/TESTING.md)
make -f Makefile.harness audit   # harness artifact audit
```
