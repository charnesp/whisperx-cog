# Architecture

System map for whisperx-cog. User-facing setup lives in [README.md](../README.md).

## Components

```
┌─────────────────────────────────────────────────────────────┐
│  Self-hosted stack (k8s pod or Docker Compose network)      │
│  ┌──────────┐   ┌───────┐   ┌──────────────────────────┐  │
│  │ whisperx │   │ redis │   │ bridge (Replicate API)     │  │
│  │ Cog:5000 │◄──┤ :6379 │◄──┤ :8080, loopback-only deps │  │
│  └──────────┘   └───────┘   └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
         ▲                              ▲
         │ GPU                          │ Bearer auth (external)
    predict.py                     Clients / webhooks
```

| Layer | Path | Responsibility | Must not |
|-------|------|----------------|----------|
| **Predictor** | `predict.py` | WhisperX pipeline, Cog I/O, GPU health | HTTP serving, Redis, auth |
| **JSON boundary** | `json_sanitize.py` | NaN/inf sanitization before Cog JSON response | Import torch, cog, whisperx |
| **Bridge** | `bridge/bridge.py` | Replicate-compatible proxy, webhook→Redis cache | ML inference |
| **Bridge sync** | `scripts/bridge_k8s.py`, `scripts/*-bridge*.py` | Keep `bridge.py` ↔ k8s ConfigMap aligned | Runtime HTTP |
| **Deploy** | `k8s/`, `docker-compose.yml` | Orchestration, secrets, probes | Application logic |
| **Build** | `cog.yaml`, `build.sh` | Image build, model pre-download | Request handling |

## Data flow — prediction

1. Client `POST /predictions` → **bridge** (optional webhook injection).
2. Bridge proxies → **Cog** `predict.py:Predictor.predict()`.
3. Pipeline: load audio → transcribe → optional align → optional diarize.
4. Output passed through **`sanitize_for_json()`** → Cog JSON response / webhook.
5. On `completed`, bridge stores payload in **Redis** (if internal webhook used).
6. Client `GET /predictions/<id>` → Redis hit or Cog proxy.

## Module boundaries (enforced by convention)

- **`json_sanitize.py`** — pure Python, unit-tested without GPU; sole place for JSON float safety.
- **`bridge/bridge.py`** — stdlib + loopback only; no `redis-py`; RESP2 client inline.
- **`predict.py`** — Cog entrypoint; delegates JSON safety to `json_sanitize`.

## Dual copy invariant

`bridge/bridge.py` (source of truth) must match the ConfigMap in `k8s/whisperx-stack.yaml`. See [docs/BRIDGE.md](./BRIDGE.md) and `make -f Makefile.harness smoke`.

## External dependencies

| Dependency | Used by | Notes |
|------------|---------|-------|
| Cog / Replicate HTTP API | bridge, clients | Async via webhooks |
| Hugging Face token | predict (diarization) | Secret in k8s / `.env` |
| `ghcr.io/charnesp/whisperx-cog` | k8s, compose | Built via `.github/workflows/docker-publish.yml` |

## Related docs

- [BRIDGE.md](./BRIDGE.md) — bridge behavior, errors, logging
- [OBSERVABILITY.md](./OBSERVABILITY.md) — logs, health checks, troubleshooting
- [DATA_CONTRACTS.md](./DATA_CONTRACTS.md) — prediction input/output shapes
- [AGENTS.md](../AGENTS.md) — agent entry map and harness commands
