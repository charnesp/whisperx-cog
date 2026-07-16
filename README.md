# whisperX on Replicate

This repo is the codebase behind the following Replicate models, which we use at [Upmeet](https://upmeet.ai):

- [victor-upmeet/whisperx](https://replicate.com/victor-upmeet/whisperx) — default choice; low-cost hardware, suits most cases
- [victor-upmeet/whisperx-a40-large](https://replicate.com/victor-upmeet/whisperx-a40-large) — for memory issues on long audio with alignment/diarization
- [victor-upmeet/whisperx-a100-80gb](https://replicate.com/victor-upmeet/whisperx-a100-80gb) — same use case, larger GPU

## Contents

- [Model information](#model-information)
- [Repository layout](#repository-layout)
- [Self-hosted deployment](#self-hosted-deployment)
  - [Architecture](#architecture)
  - [Using the API](#using-the-api)
  - [OpenAI-compatible endpoint](#openai-compatible-endpoint)
  - [Health checks](#health-checks)
  - [Prediction response format](#prediction-response-format)
  - [Bridge script maintenance](#bridge-script-maintenance)
  - [Kubernetes](#kubernetes)
  - [Docker Compose](#docker-compose)
- [Building the Cog image](#building-the-cog-image)
- [Maintainer notes](#maintainer-notes)
- [Citation](#citation)

## Model information

WhisperX provides fast automatic speech recognition (70× realtime with large-v3) with word-level timestamps and speaker diarization.

Whisper is an ASR model developed by OpenAI, trained on a large dataset of diverse audio. Whilst it produces highly accurate transcriptions, the corresponding timestamps are at the utterance level, not per word, and can be inaccurate by several seconds. OpenAI’s Whisper does not natively support batching, but WhisperX does.

The transcription model is **large-v3** from [faster-whisper](https://github.com/SYSTRAN/faster-whisper).

For implementation details, see the [WhisperX GitHub repo](https://github.com/m-bain/whisperX).

## Repository layout

| Path | Role |
|------|------|
| `predict.py` | Cog `Predictor` — transcribe, align, diarize pipeline |
| `cog.yaml` | Cog build and runtime configuration |
| `bridge/bridge.py` | **Source of truth** for the Replicate-compatible HTTP bridge (see [Bridge script maintenance](#bridge-script-maintenance)) |
| `bridge/openai_compat.py` | OpenAI STT endpoint handler (`POST /v1/audio/transcriptions`) |
| `bridge/Dockerfile` | Bridge container image (published as `ghcr.io/charnesp/whisperx-cog-bridge`) |
| `bridge/README.md` | Short pointer for bridge maintainers |
| `k8s/whisperx-stack.yaml` | Kubernetes manifest (Cog + Redis + bridge GHCR image) |
| `docker-compose.yml` | Docker Compose stack equivalent to the k8s pod |
| `.env.example` | Environment template for Docker Compose secrets |
| `AGENTS.md` | Agent entry map → `docs/` for depth |
| `Makefile.harness` | Harness: `make -f Makefile.harness smoke/check/ci` |
| `docs/` | Architecture, bridge, observability, data contracts |
| `PLANS.md` | Plans and tech-debt tracker |

Published Docker images (built from `main` via GitHub Actions):

- `ghcr.io/charnesp/whisperx-cog:latest` — WhisperX Cog predictor
- `ghcr.io/charnesp/whisperx-cog-bridge:latest` — HTTP bridge (Replicate + OpenAI STT)

## Self-hosted deployment

Run WhisperX on your own infrastructure with a **Replicate-compatible API** and **webhook-based result storage**. The stack emulates Replicate’s prediction lifecycle: create a prediction, receive webhooks for `start` and `completed`, and retrieve the result by ID.

Two deployment options are provided:

| | Kubernetes | Docker Compose |
|---|------------|----------------|
| Manifest | `k8s/whisperx-stack.yaml` | `docker-compose.yml` |
| Secrets | Kubernetes Secrets in the manifest | `.env` (from `.env.example`) |
| GPU | `nvidia.com/gpu: 1` per pod | `gpus: all` (NVIDIA Container Toolkit) |
| Typical use | Production cluster | Local dev / single host |

Both expose the same ports and API (see below).

### Architecture

Each stack runs **three components in one shared network namespace** (Kubernetes pod, or Docker Compose `network_mode: service:bridge`):

1. **whisperx** — Cog server (WhisperX) on port **5000**
2. **redis** — In-memory store for webhook payloads (prediction status and output), TTL 24 h
3. **bridge** — HTTP server on port **8080** that:
   - Proxies client requests to Cog
   - On `POST /predictions`: if the body has no `webhook`, injects an internal webhook URL so Cog sends `start` and `completed` events to the bridge; if the client already provides a webhook, forwards the payload unchanged
   - Stores webhook payloads in Redis and serves them on `GET /predictions/<id>`

```
Clients ──► :8080 bridge ──► proxy ──► :5000 Cog (whisperx)
                ▲                           │
                │         webhook (internal)│
                └─────── Redis ◄────────────┘
```

**Authentication:**

- External API: `Authorization: Bearer <BRIDGE_TOKEN>` (k8s secret `BRIDGE_TOKEN`; Compose env `BRIDGE_TOKEN` → container `BRIDGE_AUTH_TOKEN`)
- Internal Cog→bridge webhook: path segment `/<WEBHOOK_SECRET>/webhook?id=<prediction_id>` — not for external use
- `GET /health` and `GET /health-check`: no auth

**Important:** If the client supplies its own `webhook`, the bridge does **not** store the prediction in Redis, so `GET /predictions/<id>` will not return status or output — use the client webhook instead.

### Using the API

- **Endpoint:** `http://<host>:8080`
- **Create a prediction:** `POST /predictions` with JSON body (e.g. `input` with `audio` URL and options). Header: `Authorization: Bearer <BRIDGE_TOKEN>`.
- **Get result:** `GET /predictions/<id>` with the same Bearer token. Returns the cached prediction JSON once the webhook has fired (or proxies to Cog on cache miss).
- **Liveness:** `GET /health` (no auth) — bridge process only.
- **Readiness:** `GET /health-check` (no auth) — proxied to Cog; see [Health checks](#health-checks).

Async predictions (`Prefer: respond-async`) return `202 Accepted`; status updates are delivered via webhooks only (Cog does not support polling). The bridge injects an internal webhook when none is provided so results can be fetched with `GET /predictions/<id>`.

### OpenAI-compatible endpoint

The bridge also exposes an **OpenAI Whisper-compatible** speech-to-text endpoint for clients such as the [openai-python](https://github.com/openai/openai-python) SDK and Hermes Agent.

| Item | Value |
|------|-------|
| Endpoint | `POST /v1/audio/transcriptions` |
| Auth | `Authorization: Bearer <BRIDGE_TOKEN>` (same token as Replicate API) |
| Request | `multipart/form-data` with field `file` (binary audio) |
| Response | **Synchronous** — single HTTP response, no polling or webhooks |
| Default response | JSON `{"text": "..."}` |

**Example (curl):**

```bash
curl -sS http://localhost:8080/v1/audio/transcriptions \
  -H "Authorization: Bearer $BRIDGE_TOKEN" \
  -F file=@sample.ogg \
  -F model=whisper-1 \
  -F language=fr
```

**Supported `response_format` values:** `json` (default), `text`, `verbose_json`, `srt`, `vtt`, `diarized_json`.

**Diarized transcription** (speaker labels):

```bash
curl -sS http://localhost:8080/v1/audio/transcriptions \
  -H "Authorization: Bearer $BRIDGE_TOKEN" \
  -F file=@sample.wav \
  -F model=gpt-4o-transcribe-diarize \
  -F response_format=diarized_json \
  -F chunking_strategy=auto
```

Requires `HUGGINGFACE_TOKEN` on the **whisperx** container (already set in k8s/compose). The bridge does not need the token.

**Model mapping** (OpenAI name → Cog `whisper_model`):

| Client `model` | Cog `whisper_model` | Diarization |
|----------------|---------------------|-------------|
| `whisper-1` | `large-v3-turbo` | off |
| `gpt-4o-transcribe-diarize` | `large-v3-turbo` | on |
| `large-v3` | `large-v3` | off |
| `large-v3-turbo` | `large-v3-turbo` | off |
| `tiny` | `tiny` | off |

**Not supported (v1):** `known_speaker_references[]` returns HTTP 400 — see [PLANS.md](./PLANS.md) for follow-up.

**Supported audio extensions** (OpenAI official allowlist): `flac`, `mp3`, `mp4`, `mpeg`, `mpga`, `m4a`, `ogg`, `wav`, `webm`.

**Environment variables** (bridge container):

| Variable | Default | Purpose |
|----------|---------|---------|
| `OPENAI_STT_TIMEOUT_SECONDS` | `300` | Max wait for sync Cog transcription → HTTP 504 |
| `OPENAI_STT_MAX_FILE_SIZE_MB` | `25` | Max upload size → HTTP 413 |

**openai-python example:**

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key=os.environ["BRIDGE_TOKEN"])
result = client.audio.transcriptions.create(model="whisper-1", file=open("sample.ogg", "rb"))
print(result.text)

# Diarized transcription (requires HUGGINGFACE_TOKEN on whisperx container)
result = client.audio.transcriptions.create(
    model="gpt-4o-transcribe-diarize",
    file=open("sample.wav", "rb"),
    response_format="diarized_json",
    chunking_strategy="auto",
)
for seg in result.segments:
    print(seg.speaker, seg.text)
```

Audio is passed to Cog as a base64 **data URI** in JSON (`input.audio_file`); see [docs/BRIDGE.md](./docs/BRIDGE.md).

### Health checks

| Check | Path | Port | Auth | Meaning |
|-------|------|------|------|---------|
| Bridge liveness | `GET /health` | 8080 | No | Bridge HTTP server is up |
| Cog readiness | `GET /health-check` | 8080 (proxied to 5000) | No | Model server status in JSON body |

Cog’s `GET /health-check` always returns HTTP 200. Inspect JSON **`status`**:

- `READY` — accepting predictions
- `STARTING` — setup in progress
- `BUSY` — prediction running
- `SETUP_FAILED`, `DEFUNCT`, `UNHEALTHY` — not ready

This predictor implements **`healthcheck()`** and returns `True` only when CUDA is available, so Cog reports `UNHEALTHY` if the GPU is missing or broken.

Kubernetes uses bridge `GET /health` for **livenessProbe** and `GET /health-check` for **readinessProbe**. Docker Compose healthchecks bridge liveness only (`GET /health` on the bridge container).

### Prediction response format

The prediction object (API response or webhook payload on `completed`) includes:

| Field | Description |
|-------|-------------|
| `status` | `"succeeded"`, `"failed"`, `"canceled"`, or while running `"starting"` / `"processing"` |
| `output` | Return value of `predict()` when succeeded; omitted or `null` on failure |
| `error` | Present when `status === "failed"` — error message string |
| `metrics` | Optional (e.g. `predict_time` in seconds) |

**Example — failed prediction:**

```json
{
  "status": "failed",
  "error": "Out of range float values are not JSON compliant: nan",
  "output": null,
  "metrics": { "predict_time": 12.34 }
}
```

**Example — succeeded prediction:**

```json
{
  "status": "succeeded",
  "output": {
    "segments": [
      {
        "start": 0.52,
        "end": 3.544,
        "text": " The glow deepened in the eyes of the sweet girl.",
        "words": [
          {
            "word": "The",
            "start": 0.52,
            "end": 0.642,
            "score": 0.796,
            "speaker": "SPEAKER_00"
          }
        ],
        "speaker": "SPEAKER_00"
      }
    ],
    "detected_language": "en",
    "speaker_embeddings": {
      "SPEAKER_00": [-0.09, 0.04, -0.08]
    }
  },
  "metrics": { "predict_time": 5.2 }
}
```

**Output schema (when `status` is `succeeded`):**

| Field | Type | Description |
|-------|------|-------------|
| `segments` | array | Utterance list |
| `segments[].start` / `end` | number | Timestamps in seconds |
| `segments[].text` | string | Transcribed text |
| `segments[].words` | array | Word-level timing (if alignment enabled) |
| `segments[].words[].score` | number | Alignment confidence 0–1 |
| `segments[].words[].speaker` | string | Speaker id if diarization enabled |
| `segments[].speaker` | string | Segment speaker if diarization enabled |
| `detected_language` | string | ISO code (e.g. `"en"`) |
| `speaker_embeddings` | object \| null | `speaker_id` → float vector; `null` without diarization |

References: [Cog HTTP API](https://cog.run/http/), [Replicate HTTP API](https://replicate.com/docs/reference/http).

### Bridge script maintenance

The bridge is implemented in **`bridge/bridge.py`** (routing, auth, Replicate API) and **`bridge/openai_compat.py`** (OpenAI STT). These files are the **source of truth**.

They are packaged into one GHCR image consumed by Docker Compose and Kubernetes:

| Consumer | How bridge scripts are loaded |
|----------|-------------------------------|
| Docker Compose | `ghcr.io/charnesp/whisperx-cog-bridge:latest` (GHCR, built by GitHub Actions) |
| Kubernetes | same image in `k8s/whisperx-stack.yaml` |

```bash
python3 scripts/check-bridge-sync.py    # before commit (also: make smoke)
# after editing bridge/*.py: push to main → GHCR rebuild → kubectl rollout restart
```

Edit `bridge/*.py` first; the image is rebuilt on push to `main` (see `.github/workflows/bridge-docker-publish.yml`).

See [AGENTS.md](./AGENTS.md) for bridge logging prefixes, webhook error codes, Redis limits, and other implementation details.

### Kubernetes

**Prerequisites:** GPU cluster (`nvidia.com/gpu: 1`), `kubectl`, image `ghcr.io/charnesp/whisperx-cog:latest` (or your registry).

**Setup:**

1. Edit secrets in `k8s/whisperx-stack.yaml`:
   - **bridge-auth-secret:** `BRIDGE_TOKEN` (external API), `WEBHOOK_SECRET` (internal webhook path)
   - **hf-secret:** `HUGGINGFACE_TOKEN` (required for diarization models)
2. Deploy:
   ```bash
   kubectl apply -f k8s/whisperx-stack.yaml
   ```
3. Expose (LoadBalancer, Ingress, or port-forward):
   ```bash
   kubectl port-forward service/whisperx-service 8080:8080 5000:5000
   ```

**Ports (Service `whisperx-service`):**

| Port | Service |
|------|---------|
| 8080 | Replicate-compatible bridge API |
| 5000 | Cog HTTP API (optional direct access) |

**Logs:** `kubectl logs` on the **bridge** container; lines prefixed `[bridge ext]` (external clients) and `[bridge int]` (Redis, Cog proxy, internal webhook).

### Docker Compose

**Prerequisites:** Docker Compose v2, [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

**Setup:**

1. Copy and edit secrets:
   ```bash
   cp .env.example .env
   # BRIDGE_TOKEN, WEBHOOK_SECRET, HUGGINGFACE_TOKEN
   ```
2. Start:
   ```bash
   docker compose up -d
   ```
3. Check health:
   ```bash
   curl http://localhost:8080/health
   curl -H "Authorization: Bearer $BRIDGE_TOKEN" http://localhost:8080/health-check
   ```

**Ports** (defaults from `.env.example`):

| Port | Service |
|------|---------|
| 8080 | Replicate-compatible bridge API |
| 5000 | Cog HTTP API (optional direct access) |

**Logs:** `docker compose logs bridge` — same `[bridge ext]` / `[bridge int]` prefixes as Kubernetes.

Resource limits match the k8s Deployment: 12 GiB RAM, 4 CPUs for whisperx.

## Building the Cog image

Requires [Cog](https://github.com/replicate/cog) and a GPU for local runs.

```bash
cog build
cog predict -i audio=@sample.wav
```

Model weights: Cog `run:` steps bake **`large-v3-turbo`** into **`/models`** (outside `/src`, because Cog copies source *after* `run`). VAD weights ship inside the `whisperx` package (no separate download). Requests for `tiny` or `large-v3` download from HuggingFace on first use.

Local bake into `./models` (dev only):

```bash
WHISPER_BAKE_MODELS=all bash build.sh
```

FFmpeg **6** is installed once from a PPA during `cog build` (required for pyannote/torchcodec). It is not listed in `system_packages` — the default Ubuntu package is too old.

The CI workflow `.github/workflows/docker-publish.yml` builds, **verifies `/models/.../model.bin` exists**, and pushes to `ghcr.io/charnesp/whisperx-cog` on pushes to `main`.

To use the self-hosted bridge stacks with a locally built image, replace `ghcr.io/charnesp/whisperx-cog:latest` in `k8s/whisperx-stack.yaml` or `docker-compose.yml`.

## Maintainer notes

**[AGENTS.md](./AGENTS.md)** is the agent entry map; deep reference lives in **`docs/`** (architecture, bridge, observability, data contracts, [testing policy](./docs/TESTING.md)).

**Harness commands** (no GPU required):

```bash
make -f Makefile.harness smoke   # fast checks
make -f Makefile.harness ci        # full gate (CI)
```

When changing bridge behavior: edit `bridge/*.py` → push to rebuild GHCR image → `make -f Makefile.harness smoke`.

## Citation

```
@misc{bain2023whisperx,
      title={WhisperX: Time-Accurate Speech Transcription of Long-Form Audio},
      author={Max Bain and Jaesung Huh and Tengda Han and Andrew Zisserman},
      year={2023},
      eprint={2303.00747},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```
