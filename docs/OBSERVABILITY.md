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
make -f Makefile.harness check   # + unit tests
make -f Makefile.harness audit   # harness artifact audit
```
