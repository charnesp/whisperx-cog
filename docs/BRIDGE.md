# Bridge

HTTP bridge for Replicate-compatible self-hosted deployment. Source of truth: **`bridge/bridge.py`**.

Full API usage: [README.md](../README.md). Architecture: [ARCHITECTURE.md](./ARCHITECTURE.md).

## Dual copies (keep in sync)

| Deployment | How loaded |
|------------|------------|
| Docker Compose | Bind-mount `./bridge/bridge.py` |
| Kubernetes | ConfigMap `cog-bridge-script` in `k8s/whisperx-stack.yaml` |

```bash
python3 scripts/sync-bridge-to-k8s.py   # after editing bridge/bridge.py
python3 scripts/check-bridge-sync.py    # before commit (also: make smoke)
```

## Environment variables

| Container var | k8s secret | Compose `.env` | Purpose |
|---------------|------------|----------------|---------|
| `BRIDGE_AUTH_TOKEN` | `BRIDGE_TOKEN` | `BRIDGE_TOKEN` | `Authorization: Bearer …` |
| `WEBHOOK_SECRET` | `WEBHOOK_SECRET` | `WEBHOOK_SECRET` | Internal webhook path segment |
| `HUGGINGFACE_TOKEN` | `HUGGINGFACE_TOKEN` | `HUGGINGFACE_TOKEN` | Passed to whisperx container only |

Loopback addresses (`127.0.0.1:6379`, `127.0.0.1:5000`, `localhost:8080` webhook) are fixed because Cog, Redis, and bridge share one network namespace.

## Logging

| Prefix | Scope |
|--------|--------|
| `[bridge ext]` | External clients — API requests, cache hit/miss |
| `[bridge int]` | In-pod — Redis, Cog proxy, Cog→bridge webhook |
| `[bridge]` | Process boot only |

`GET /health` and `GET /health-check` are silent. Loopback peers log as `[bridge int]`.

**Kubernetes:** `kubectl logs <pod> -c bridge`  
**Compose:** `docker compose logs bridge`

## Webhook → Redis

- URL: `http://localhost:8080/<WEBHOOK_SECRET>/webhook?id=<prediction_id>`
- RESP2 Redis client (not `redis-py`); max payload **200 MiB**; TTL **24 h**
- Requires **`Content-Length`** (chunked → **501**)

### Webhook HTTP errors

| Status | Code | Notes |
|--------|------|-------|
| 400 | `invalid_content_length` | Malformed or negative Content-Length |
| 400 | `invalid_prediction_id` | `id` not matching `^[a-zA-Z0-9_-]{1,64}$` |
| 413 | `payload_too_large` | Body > max; `Connection: close`, body drained |
| 501 | `unsupported_transfer_encoding` | Chunked not supported; best-effort drain |
| 503 | `redis_set_failed` | Redis error; JSON includes `detail`; Cog may retry |

## GET /predictions/<id>

- `id` must match `^[a-zA-Z0-9_-]{1,64}$` else **400**
- Redis hit → stored JSON immediately
- Miss or Redis read failure → proxy to Cog (logged distinctly)

## POST /predictions

- No client `webhook` → inject internal webhook + `webhook_events_filter: ["start", "completed"]`, store in Redis
- Client provides webhook → forward unchanged, **no Redis storage**

## Probes

- Malformed `Content-Length` on silent `GET /health-check` → treated as empty body (forwards to Cog)
- Error responses that leave socket ambiguous send **`Connection: close`** (400, 413, 501, 503 on webhook path)
