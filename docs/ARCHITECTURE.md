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
| **Bridge sync** | `scripts/bridge_k8s.py`, `scripts/check-bridge-sync.py` | Verify k8s uses bridge GHCR image | Runtime HTTP |
| **Deploy** | `k8s/`, `docker-compose.yml` | Orchestration, secrets, probes | Application logic |
| **Build** | `cog.yaml`, `build.sh` | Cog bakes turbo to `/models`; VAD comes from whisperx package | Request handling |

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

## Remote qwen3-asr backend (optional)

The `qwen3-asr` backend runs **in-process** by default. With `QWEN_BACKEND=remote` it delegates transcription to an external OpenAI-compatible vLLM engine reached over HTTP:

- `predict.py` resolves the backend once in `setup()`: remote mode reads `QWEN_REMOTE_BASE_URL` / `QWEN_REMOTE_MODEL` / `QWEN_REMOTE_TIMEOUT_S` from env and fails fast with a typed `QwenRemoteError:` config error when the address or the model is missing.
- Each **local VAD window** is posted as one multimodal `chat/completions` request; the reply text is parsed back, while segment `start`/`end` stay owned by the **local VAD chunking** (the reply never determines timestamps).
- The downstream stages are untouched and stay **local**: the Qwen ForcedAligner word alignment and the pyannote diarization run after remote transcription exactly as in the in-process path.
- The **bridge contract** and the `ENABLE_QWEN` kill-switch are unchanged; this change adds **no new gate** in cog (the pre-existing cog `qwen_enabled()` defense-in-depth check stays as-is).
- No address, port or model name lives in the repository: the deployment injects them (`docker-compose.yml` `${QWEN_REMOTE_*}` / k8s env). See [DATA_CONTRACTS.md](./DATA_CONTRACTS.md) for the request/response shapes.

```
Clients ──► :8080 bridge ──► :5000 Cog (whisperx) ──► HTTP ──► remote engine
                                          │  (QWEN_BACKEND=remote)   (vLLM)
                                          └──► ForcedAligner + pyannote (local)
```

## Dual copy invariant

`bridge/*.py` is packaged into `ghcr.io/charnesp/whisperx-cog-bridge:latest` for Compose and k8s. See [docs/BRIDGE.md](./BRIDGE.md) and `make -f Makefile.harness smoke`.

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
- [TESTING.md](./TESTING.md) — strict TDD (RED→GREEN→REFACTOR); harness gate
- [AGENTS.md](../AGENTS.md) — agent entry map and harness commands
