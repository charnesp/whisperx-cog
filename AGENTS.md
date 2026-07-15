# AGENTS.md — Context for AI agents and maintainers

This file summarizes API behavior, gotchas, and conventions for the whisperx-cog project so that agents and humans can work on it consistently.

For user-facing setup (Kubernetes, Docker Compose, API usage), see **[README.md](./README.md)**.

## Code layout

| Path | Role |
|------|------|
| `predict.py` | Cog `Predictor`, `Output` model, `healthcheck()`, transcribe/align/diarize pipeline |
| `cog.yaml` | Cog build config; `cog_runtime: true` is **disabled** (coglet Output type limits) |
| `bridge/bridge.py` | **Source of truth** for the Replicate-compatible HTTP bridge |
| `bridge/README.md` | Maintainer pointer (sync workflow) |
| `k8s/whisperx-stack.yaml` | Kubernetes Deployment (Cog + Redis + bridge); embeds `bridge.py` in ConfigMap `cog-bridge-script` |
| `docker-compose.yml` | Docker Compose equivalent; mounts `bridge/bridge.py` directly |
| `.env.example` | Template for Compose secrets (`BRIDGE_TOKEN`, `WEBHOOK_SECRET`, `HUGGINGFACE_TOKEN`) |
| `scripts/sync-bridge-to-k8s.py` | Regenerate ConfigMap `bridge.py` block from `bridge/bridge.py` |
| `scripts/check-bridge-sync.py` | Verify standalone file matches k8s ConfigMap |

Published image: `ghcr.io/charnesp/whisperx-cog:latest`.

## Bridge script — dual copies (keep in sync)

The bridge logic lives in **`bridge/bridge.py`**. Two deployments consume it:

| Deployment | How the script is loaded |
|------------|--------------------------|
| **Docker Compose** | Volume mount `./bridge/bridge.py` → `/scripts/bridge.py` |
| **Kubernetes** | Inline in ConfigMap `cog-bridge-script`, key `bridge.py`, in `k8s/whisperx-stack.yaml` |

**Workflow when editing the bridge:**

1. Edit **`bridge/bridge.py`** only (never edit the ConfigMap block by hand unless syncing back to the standalone file).
2. Run `python3 scripts/sync-bridge-to-k8s.py` to update the Kubernetes manifest.
3. Run `python3 scripts/check-bridge-sync.py` before committing — both copies must match.

The bridge hardcodes loopback addresses (`REDIS_HOST=127.0.0.1`, `COG_URL=http://127.0.0.1:5000`, internal webhook `http://localhost:8080/...`) because Cog, Redis, and bridge share one network namespace (k8s pod or Compose `network_mode: service:bridge`).

### Environment variables

| Variable (container) | Kubernetes secret key | Docker Compose `.env` | Purpose |
|--------------------|----------------------|------------------------|---------|
| `BRIDGE_AUTH_TOKEN` | `bridge-auth-secret` → `BRIDGE_TOKEN` | `BRIDGE_TOKEN` | External API: `Authorization: Bearer …` |
| `WEBHOOK_SECRET` | `bridge-auth-secret` → `WEBHOOK_SECRET` | `WEBHOOK_SECRET` | Internal webhook path segment |
| `HUGGINGFACE_TOKEN` | `hf-secret` → `HUGGINGFACE_TOKEN` | `HUGGINGFACE_TOKEN` | WhisperX / diarization models |

### Bridge behavior

- **Logs:** Significant actions use stdout prefixes **`[bridge ext]`** (HTTP **edge**: clients **outside** the pod — `GET`/`POST` API, cache outcome for those clients) and **`[bridge int]`** (in-pod legs: **Redis**, **Cog** HTTP proxy, **Cog→bridge webhook**). **`GET /health`** and **`GET /health-check`** stay silent. **`[bridge]`** is only the process boot line. Peers on loopback (`127.0.0.1` / `::1`) are treated as in-pod, so their API calls log under **`int`** on the edge logger. Use `kubectl logs` (bridge container) or `docker compose logs bridge` for troubleshooting.
- **Webhook → Redis:** Cog POSTs completion payloads to `http://localhost:8080/<WEBHOOK_SECRET>/webhook?id=<prediction_id>`. The bridge runs a minimal **RESP2** Redis client (not `redis-py`), caps cached JSON at **200 MiB** (`REDIS_MAX_BULK_BYTES`), TTL **24 h** (`REDIS_MESSAGE_TTL`), and rejects **`Transfer-Encoding: chunked`** with **501** (Cog should send **`Content-Length`**).
- **Webhook HTTP errors:**
  - **400** `invalid_content_length` — malformed or negative `Content-Length`.
  - **400** `invalid_prediction_id` — `id` query parameter does not match `^[a-zA-Z0-9_-]{1,64}$`; body is drained when `Content-Length` was already validated.
  - **413** `payload_too_large` — body larger than the configured max; response includes **`Connection: close`**, then the bridge drains the body.
  - **501** `unsupported_transfer_encoding` — chunked body not supported; **`Connection: close`**, then a **best-effort drain** (up to 64 MiB) of unread bytes.
  - **503** `redis_set_failed` — Redis unreachable, protocol error, or unexpected reply; JSON body includes **`detail`**. With the Cog version you run, **webhook delivery may be retried on 5xx** (verify in that version’s source or by observing a second POST after a forced **503** — not guaranteed by the public Cog HTTP doc alone).
- **GET** `GET /predictions/<id>`: **`id`** must match the same **`^[a-zA-Z0-9_-]{1,64}$`** pattern; otherwise **400** `invalid_prediction_id`.
- **GET cache vs Cog:** A **Redis hit** returns the stored JSON immediately. **Cache miss** (`GET` returns Redis null bulk) or **Redis read failure** is logged distinctly; the handler then **proxies to Cog** (same status behavior as before for the API client on read errors).
- **POST /predictions:** If the client omits `webhook`, the bridge injects the internal webhook and `webhook_events_filter: ["start", "completed"]`. If the client provides its own webhook, the request is forwarded unchanged and **nothing is stored in Redis**.
- **Readiness probe:** For **`GET /health-check`** with **`silent`** proxying, a **malformed `Content-Length`** is treated as **no body** (`0`) so the probe still forwards to Cog instead of returning **400** from the bridge.
- **Keep-alive:** Error responses that may leave the socket in an ambiguous state (**400**, **413**, **501**, **503** on the webhook path) send **`Connection: close`**.

## Cog / Replicate prediction API

- **Cog HTTP API:** https://cog.run/http/
- **Replicate HTTP API:** https://replicate.com/docs/reference/http

### Prediction response format

The prediction object (response body or webhook payload) includes:

| Field     | Description |
|----------|-------------|
| `status` | `"succeeded"`, `"failed"`, `"canceled"`, or (while running) `"starting"` / `"processing"` |
| `output` | Return value of `predict()` when `status === "succeeded"`. Omitted or `null` when failed/canceled. |
| `error`  | **Only when `status === "failed"`.** String message describing the failure. |
| `metrics`| Optional object (e.g. `predict_time` in seconds). May be present on both success and failure. |

**Failed prediction example:**

```json
{
  "status": "failed",
  "error": "Out of range float values are not JSON compliant: nan",
  "output": null,
  "metrics": { "predict_time": 12.34 }
}
```

### Output schema (success)

When `status === "succeeded"`, `output` has this shape (from WhisperX transcribe + optional align + optional diarize):

```python
{
  "segments": [
    {
      "start": 0.52,           # seconds
      "end": 3.544,
      "text": " The glow deepened in the eyes of the sweet girl.",
      "words": [               # present if alignment enabled
        {
          "word": "The",
          "start": 0.52,
          "end": 0.642,
          "score": 0.796,      # alignment confidence 0–1
          "speaker": "SPEAKER_00"  # present if diarization enabled
        },
        # ...
      ],
      "speaker": "SPEAKER_00"  # present if diarization enabled
    }
  ],
  "detected_language": "en",   # ISO code
  "speaker_embeddings": {      # null if diarization disabled
    "SPEAKER_00": [ -0.09, 0.04, ... ]  # 512-dim embedding vector
  }
}
```

- **segments**: list of segments; each has `start`, `end`, `text`, optional `words` (word-level timing), optional `speaker`.
- **detected_language**: string (e.g. `"en"`).
- **speaker_embeddings**: `null` or `{ "<speaker_id>": [float, ...] }`. Vectors are sanitized for JSON (no NaN/inf).

### Webhooks (Cog)

- Events: `start`, `output`, `logs`, `completed`.
- `completed` is sent once when the prediction reaches a terminal state (`succeeded`, `canceled`, or `failed`).
- The request body is the **current prediction object** (same shape as above). So on `completed`, check `status` and use `output` or `error` accordingly.

### Cog crash handling (exceptions in `predict()`)

Cog does **not** shut down the whole process on every exception: when `predict()` raises, the runner catches the exception, marks the prediction as **failed**, and sends the webhook/response with `status: "failed"` and `error` set to the message. The worker is then available for the next prediction.

Behavior depends on the runtime:

| Situation | Old runtime (Python, default) | New runtime (Go, Cog ≥ 0.16) |
|-----------|------------------------------|------------------------------|
| Exception in `predict()` | Prediction marked `failed`, webhook/response with `error`. Worker may stay up; server can end up in a bad state or with poor logging. | Prediction marked `failed`; Go HTTP server stays up and can serve other requests. |
| Severe crash (OOM, segfault, etc.) | Server/worker can end up in a bad state or block (e.g. health check unresponsive during long predictions). | Python runner process may die; Go server continues. |
| Full container exit | Possible in extreme cases (e.g. OOM kill). | Less likely for the HTTP layer; Python subprocess can still be killed. |

To use the new runtime (recommended for better isolation): Cog ≥ 0.16 and in `cog.yaml`:

```yaml
build:
  cog_runtime: true
```

**This project keeps `cog_runtime` disabled** because coglet rejects `dict` / `list[dict]` in Output types. Ref: [Introducing a new Cog runtime](https://replicate.com/changelog/2025-07-21-cog-runtime).

## Project-specific gotchas

### JSON-serializable output (no NaN)

The Cog server sends the prediction result as JSON (e.g. to the webhook). Python’s `json.dumps()` uses `allow_nan=False` by default, so **any `float('nan')` in the output causes:**

`requests.exceptions.InvalidJSONError: Out of range float values are not JSON compliant: nan`

**Likely sources of NaN in this codebase:**

1. **`result["segments"]`** — WhisperX/alignment can produce NaN in segment or word-level `start`/`end` times (e.g. empty or very short audio, alignment edge cases).
2. **`detected_language`** — When using automatic language detection, the `probability` value could be NaN in rare cases.
3. **`speaker_embeddings`** — Embedding vectors from diarization may contain NaN.

**Fix:** Before returning from `predict()`, sanitize all float values in the payload (e.g. replace NaN with `None` or a sentinel number) so the response is JSON-serializable.

### Async predictions and webhooks

- With `Prefer: respond-async`, the server returns `202 Accepted` and processes in the background. Updates are delivered **only via webhooks**; polling for status is not supported by Cog.
- The bridge injects an internal webhook on `POST /predictions` when the client does not provide one, and stores the final prediction in Redis for `GET /predictions/<id>`. If the client provides its own webhook, the bridge does not store the result.

### Health checks

- **Cog** exposes **GET /health-check** (always 200; check JSON `status`: READY, STARTING, BUSY, SETUP_FAILED, DEFUNCT, UNHEALTHY). Optional **`healthcheck()`** on the Predictor: return `False` to set status UNHEALTHY.
- This predictor implements **`healthcheck()`** and returns `torch.cuda.is_available()` so the container is UNHEALTHY if the GPU is missing or broken.
- **Kubernetes:** bridge proxies **GET /health-check** to Cog (no auth); **readinessProbe** on bridge `:8080/health-check`, **livenessProbe** on `:8080/health`.
- **Docker Compose:** bridge **liveness** healthcheck on `GET /health` only; use manual `GET /health-check` for Cog readiness.

## References

- [Cog HTTP API](https://cog.run/http/)
- [Replicate HTTP API — Create/Get prediction](https://replicate.com/docs/reference/http)
- [WhisperX repo](https://github.com/m-bain/whisperX)
