# whisperX on Replicate

This repo is the codebase behind the following Replicate models, which we use at [Upmeet](https://upmeet.ai):

- [victor-upmeet/whisperx](https://replicate.com/victor-upmeet/whisperx) : if you don't know which model to use, use this one. It uses a low-cost hardware, which suits most cases
- [victor-upmeet/whisperx-a40-large](https://replicate.com/victor-upmeet/whisperx-a40-large) : if you encounter some memory issues with previous models, consider this one. It can happen when dealing with long audio files and performing alignment and/or diarization
- [victor-upmeet/whisperx-a100-80gb](https://replicate.com/victor-upmeet/whisperx-a100-80gb) : if you encounter some memory issues with previous models, consider this one. It can happen when dealing with long audio files and performing alignment and/or diarization

# Model Information

WhisperX provides fast automatic speech recognition (70x realtime with large-v3) with word-level timestamps and speaker diarization.

Whisper is an ASR model developed by OpenAI, trained on a large dataset of diverse audio. Whilst it does produces highly accurate transcriptions, the corresponding timestamps are at the utterance-level, not per word, and can be inaccurate by several seconds. OpenAI’s whisper does not natively support batching, but WhisperX does.

Model used is for transcription is large-v3 from faster-whisper.

For more information about WhisperX, including implementation details, see the [WhisperX github repo](https://github.com/m-bain/whisperX).

# Kubernetes deployment (Replicate-compatible API)

You can run this model on your own Kubernetes cluster with a **Replicate-compatible API** and **webhook-based result storage**. The stack emulates Replicate’s prediction lifecycle: create a prediction, get webhooks for `start` and `completed`, and poll or retrieve the result by ID.

## Architecture

The `k8s/whisperx-stack.yaml` manifest deploys a single pod with three containers:

1. **whisperx** – Cog server (WhisperX) listening on port 5000.
2. **redis** – In-memory store for webhook payloads (prediction status and output).
3. **bridge** – HTTP server on port 8080 that:
   - Proxies requests to the Cog server.
   - For `POST /predictions`: if the body does not already include `webhook` and `webhook_events_filter`, injects an internal webhook so Cog sends status updates (start, completed) to the bridge; otherwise leaves the payload as-is (classic Replicate behavior when the client provides its own webhook).
   - Stores webhook payloads in Redis and serves them on `GET /predictions/{id}`.

The **webhook and `WEBHOOK_SECRET` are only for internal communication** (Cog → bridge inside the pod). They are not meant to be called or exposed to the outside. External API access uses **`Authorization: Bearer <BRIDGE_TOKEN>`**. `/health` is unauthenticated.

## Prerequisites

- A Kubernetes cluster with GPU nodes (e.g. `nvidia.com/gpu: 1`).
- `kubectl` configured for that cluster.
- Docker image `ghcr.io/charnesp/whisperx-cog:latest` (or update the image in the Deployment to your own registry).

## Setup

1. **Edit the secrets** in `k8s/whisperx-stack.yaml` (or replace them with your own secret management):
   - **bridge-auth-secret**: set `BRIDGE_TOKEN` (for external API auth) and `WEBHOOK_SECRET` (only for the internal Cog→bridge webhook; not exposed externally).
   - **hf-secret**: set `HUGGINGFACE_TOKEN` to your Hugging Face token (required for WhisperX models).

2. **Deploy the stack:**
   ```bash
   kubectl apply -f k8s/whisperx-stack.yaml
   ```

3. **Expose the service** (e.g. via LoadBalancer, Ingress, or port-forward for testing):
   ```bash
   kubectl port-forward service/whisperx-service 8080:8080 5000:5000
   ```

## Using the API

- **Replicate-compatible endpoint:** `http://<host>:8080`
- **Create a prediction:** `POST /predictions` with a JSON body (e.g. `input` with `audio` URL and options). Use header `Authorization: Bearer <BRIDGE_TOKEN>`.
- **Get result:** `GET /predictions/<id>` with the same Bearer token. Returns the stored prediction payload (e.g. status and output once the webhook has fired).
- **Health check:** `GET /health` (no auth).

### Prediction response format (Cog / Replicate API)

The prediction object in the response (or sent to the webhook on `completed`) has this shape:

- **`status`**: `"succeeded"`, `"failed"`, or `"canceled"` (and `"starting"` / `"processing"` while running).
- **`output`**: The return value of `predict()` when `status` is `"succeeded"`. Omitted or `null` when the prediction failed or was canceled.
- **`error`**: Present when `status` is `"failed"`. Contains the error message (string) describing why the prediction failed.
- **`metrics`**: Optional object (e.g. `predict_time` in seconds). May be present even when the prediction failed.

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
          },
          { "word": "glow", "start": 0.682, "end": 0.987, "score": 0.85, "speaker": "SPEAKER_00" }
        ],
        "speaker": "SPEAKER_00"
      }
    ],
    "detected_language": "en",
    "speaker_embeddings": {
      "SPEAKER_00": [-0.09, 0.04, -0.08, ...]
    }
  },
  "metrics": { "predict_time": 5.2 }
}
```

**Output schema (when `status` is `succeeded`):**

| Field | Type | Description |
|-------|------|-------------|
| `segments` | array | List of segments (utterances). |
| `segments[].start` / `end` | number | Segment timestamps in seconds. |
| `segments[].text` | string | Transcribed text for the segment. |
| `segments[].words` | array | Word-level timing (present if alignment was run). |
| `segments[].words[].word` | string | Token text. |
| `segments[].words[].start` / `end` | number | Word timestamps in seconds. |
| `segments[].words[].score` | number | Alignment confidence (0–1). |
| `segments[].words[].speaker` | string | Speaker id (e.g. `SPEAKER_00`) if diarization was run. |
| `segments[].speaker` | string | Speaker id for the whole segment (if diarization was run). |
| `detected_language` | string | ISO language code (e.g. `"en"`). |
| `speaker_embeddings` | object \| null | Map `speaker_id` → vector of floats (embedding). `null` if diarization was not run. |

References: [Cog HTTP API](https://cog.run/http/) (POST /predictions response), [Replicate HTTP API](https://replicate.com/docs/reference/http) (Get a prediction).

The bridge only injects `webhook` and `webhook_events_filter: ["start", "completed"]` when the client does not send them; if the client already provides both, the request is forwarded unchanged. When the internal webhook is used, results are stored in Redis and available via `GET /predictions/<id>`. **If you provide your own webhook**, the bridge does not store the prediction in Redis, so **`GET /predictions/<id>` will not return the prediction status or output** — use your webhook URL to receive updates instead.

# Citation

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