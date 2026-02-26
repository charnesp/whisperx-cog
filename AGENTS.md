# AGENTS.md — Context for AI agents and maintainers

This file summarizes API behavior, gotchas, and conventions for the whisperx-cog project so that agents and humans can work on it consistently.

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

Ref: [Introducing a new Cog runtime](https://replicate.com/changelog/2025-07-21-cog-runtime).

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
- The bridge in `k8s/whisperx-stack.yaml` can inject an internal webhook and store the final prediction in Redis for `GET /predictions/<id>`. If the client provides its own webhook, the bridge does not store the result.

### Health checks

- **Cog** exposes **GET /health-check** (always 200; check JSON `status`: READY, STARTING, BUSY, SETUP_FAILED, DEFUNCT, UNHEALTHY). Optional **`healthcheck()`** on the Predictor: return `False` to set status UNHEALTHY.
- This predictor implements **`healthcheck()`** and returns `torch.cuda.is_available()` so the container is UNHEALTHY if the GPU is missing or broken.
- In k8s, the bridge proxies **GET /health-check** to Cog (no auth) and the Deployment uses **readinessProbe** on it; **livenessProbe** uses **GET /health** (bridge only).

## Code layout

- **`predict.py`** — Cog `Predictor`, `Output` model, `healthcheck()`, transcribe/align/diarize pipeline. Entrypoint for predictions.
- **`k8s/whisperx-stack.yaml`** — Kubernetes deployment (Cog + Redis + bridge) for a Replicate-compatible API with webhooks.

## References

- [Cog HTTP API](https://cog.run/http/)
- [Replicate HTTP API — Create/Get prediction](https://replicate.com/docs/reference/http)
- [WhisperX repo](https://github.com/m-bain/whisperX)
