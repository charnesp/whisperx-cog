## Context

The bridge (`bridge/openai_compat.py`) validates `model` against `MODEL_MAP` and forwards Cog input to `POST /predictions` with `whisper_model`, fixed `batch_size: 64`, and `hotwords: None` hard-coded. `predict.py` (`Predictor`) loads a faster-whisper model from baked `/models` paths (`model_paths.py`), runs the whisper VAD → transcribe → wav2vec2 align → pyannote diarize pipeline, and accepts `hotwords`/`batch_size` as Cog inputs.

Live measurements (2026-09-15, RTX 4080, 36-minute meeting, workspace Coder):

| Metric | large-v3-turbo | Qwen3-ASR-1.7B (batch 4) |
|--------|----------------|--------------------------|
| RTFx | ~50 | 52 |
| VRAM peak | ~4–6 GB | 4.99 GB |
| Proper nouns with hints | no effect (hotwords) | 4x more correct (Backblaze 3→12, Supabase 0→3) |

Qwen3-ASR is served through the `qwen_asr` pipeline added to whisperx by upstream PR m-bain/whisperX#1401. The fork `charnesp/whisperX` branch `qwen3-asr` (commit `c49b26379f40863767e3c42d9afed5dc4221f54f`) = PR #1401 head + a `context` patch: `QwenAsrPipeline.transcribe` accepts a `context` string and forwards it as a system message with **every** batch (no accumulation, empty string = neutral).

`qwen-asr==0.0.6` pulls gradio/flask/vllm transitively if installed bare — must be installed `--no-deps` with explicit minimal deps.

## Goals / Non-Goals

**Goals:**

- Accept `model=qwen3-asr` on `POST /v1/audio/transcriptions` (bridge whitelist; unknown models still 400)
- Route bridge `hotwords` → Qwen `context` when `model=qwen3-asr`; whisper path untouched for other models
- Per-model default batch: `QWEN_DEFAULT_BATCH = 4` when `batch_size` is absent; explicit values clamped (cap 8)
- Skip the whisper `detect_language` loop on the qwen path; pass a provided `language` through as-is
- Use `Qwen/Qwen3-ForcedAligner-0.6B` for word-level alignment on the qwen path; pyannote diarization unchanged downstream
- Load Qwen3-ASR with **explicit fp16 dtype** (fp32 default measures ~10 GB VRAM)
- Bake both Qwen snapshots into `/models` with pinned HF revisions (`models.lock`), `HF_HUB_OFFLINE=1`, fail-fast at boot
- Kill-switch `ENABLE_QWEN` env var → clean 400 without redeploy
- GPU-free unit tests for every bridge/GPU-free behavior (strict TDD per `docs/TESTING.md`)

**Non-Goals:**

- Changing the default model for existing clients (`whisper-1` → `large-v3-turbo` unchanged; the model parameter is the canary)
- Merging or closing upstream PR #1401 (fork pin is the v1 mechanism; rebase + fork removal is a tracked follow-up)
- Streaming, qwen-specific speaker profiles, or any change to faster-whisper `hotwords` semantics
- Batch > 8 (cap may be revisited after measuring batch=8 VRAM)
- Logging hotwords content (client proper nouns)

## Decisions

### 1. Model alias and whitelist

```python
MODEL_MAP = {
    "whisper-1": "large-v3-turbo",
    "gpt-4o-transcribe-diarize": "large-v3-turbo",
    "large-v3": "large-v3",
    "large-v3-turbo": "large-v3-turbo",
    "tiny": "tiny",
    "qwen3-asr": "qwen3-asr",
}
```

Unknown models keep the existing 400 `invalid_request_error`. When the `ENABLE_QWEN` env var is disabled/absent, `qwen3-asr` is rejected with the same 400 envelope (message: feature disabled) — no redeploy needed to turn the backend off.

**Rationale:** the parameter surface is unchanged for existing clients; the model name itself is the canary.

### 2. hotwords → context routing

`build_cog_input()` today sends `hotwords: None` unconditionally. Change:

- `model=qwen3-asr`: forward the client `hotwords` string as Cog input `hotwords` (Cog-side branch maps it to the Qwen `context` system message, applied identically to each batch, no accumulation, empty string = neutral). Template = participants + technical vocabulary, exactly the template validated live. Cap ~2000 chars; when truncation occurs, log a warning **with lengths only, never content**.
- Other models: `hotwords` flows to faster-whisper `asr_options["hotwords"]` as today (measured: no effect, but semantics unchanged — whisper path untouched).
- `hotwords` absent/empty → qwen behaves bit-identically to the 15/09 baseline (acceptance criterion).

**Rationale:** one parameter, model-dependent semantics, zero new multipart fields.

### 3. Per-model default batch

- Bridge: pass `batch_size` in Cog input **only when the client provides it** (today it is always `64`).
- Cog (`predict.py`): `whisper_model=qwen3-asr` and no explicit `batch_size` → `QWEN_DEFAULT_BATCH = 4` (measured OOM when all VAD segments batch at once on the shared 16 GB GPU). Explicit values clamped to `[1, 8]`.
- faster-whisper default stays `64` (unchanged).

**Rationale:** OOM protection must not depend on clients remembering the constraint; cap 8 pending a batch=8 VRAM measurement.

### 4. Language handling

On the qwen path, skip the whisper `detect_language` loop entirely (Qwen performs its own detection inside the pipeline). If the client sends `language`, it is forwarded as-is. Cog inputs `language_detection_min_prob` / `language_detection_max_tries` are ignored on this path.

### 5. Alignment and diarization

- qwen path alignment: **`Qwen/Qwen3-ForcedAligner-0.6B`** — the wav2vec2 aligner is incompatible with qwen segment outputs.
- Diarization: pyannote stage unchanged — 256-dim speaker embeddings, backend speaker profiles, output schema identical. `assign_word_speakers` must receive words from the Qwen ForcedAligner (acceptance criterion).

### 6. Dependencies

```text
whisperx @ git+https://github.com/charnesp/whisperX@c49b26379f40863767e3c42d9afed5dc4221f54f
qwen-asr==0.0.6 --no-deps
transformers==4.57.6   # already in prod
soundfile
librosa
```

gradio/flask/vllm (transitive deps of qwen-asr) are **forbidden**: CVE surface, image size, pydantic/starlette pin conflicts.

### 7. Model baking and offline enforcement

- `cog.yaml` run steps bake both snapshots into `/models` (wget + `test -s` on weights, same pattern as the turbo bake): `Qwen/Qwen3-ASR-1.7B` (~3.4 GB fp16) and `Qwen/Qwen3-ForcedAligner-0.6B`.
- HF revisions pinned in `models.lock` (two entries, not one).
- `HF_HUB_OFFLINE=1` in the Cog environment; `predict.py` fails fast at boot if baked weights are missing (no silent HF fallback for qwen).
- Image grows ~4–5 GB (host disk headroom verified).

### Test boundaries (TDD, per `docs/TESTING.md`)

- **GPU-free / strict TDD (mocked Cog):** `MODEL_MAP` entry, `ENABLE_QWEN` 400, `hotwords` passthrough only on qwen path, `batch_size` passthrough only when provided, clamp helper, context-cap truncation helper, language passthrough. RED → GREEN → REFACTOR per behavior; `make -f Makefile.harness check` after each cycle.
- **GPU path (manual smoke, no GPU CI):** fp16 load, VRAM peak, aligner/diarize end-to-end — covered by the golden set in tasks.md §6, not by unit tests.

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| Unmerged chain: PR #1401 open + qwen-asr 0.0.6 + transformers pins | Fork commit pinned in `requirements.txt`; `--no-deps` + explicit deps; `models.lock` freezes HF revisions; follow-up on upstream merge |
| fp32 default load → ~10 GB VRAM | Explicit `qwen_dtype="float16"` in the load call; VRAM asserted in golden set (< 5.5 GB logged peak) |
| Two pipelines concurrent on 16 GB GPU | cog serializes predictions (1 at a time); bridge/redis locking already sequential |
| 90-min meetings vs `REDIS_SOCKET_TIMEOUT=120` | 36 min measured ≈ 2 min transcription; verify deployed timeout before E5; bump env if needed |
| Context hallucinates hotwords into unrelated segments | Golden set measures recall AND false positives (segments without the hotwords) |
| Hotwords leak client proper nouns into logs | Never log hotwords content; truncation logs carry lengths only |
| Kill-switch forgotten | `ENABLE_QWEN` documented; disabled → clean 400 (no 500s, no redeploy) |

## Migration Plan

1. Land implementation with default unchanged; faster-whisper regression golden set must be bit-identical
2. Build/push Cog image (new weights baked); bridge unchanged in behavior for existing clients
3. Deploy via existing tag flow; rollback = previous image tag (< 5 min)
4. Canary by parameter: transcribe 1–2 real meetings in duplicate (turbo vs qwen+hotwords), compare proper nouns side by side
5. Only then consider qwen as a recommended/default model for new clients (separate decision, feu vert required)

## Open Questions

_None blocking implementation. Tracked follow-up: when upstream m-bain/whisperX#1401 merges, rebase the fork, re-pin `requirements.txt` to upstream, and delete the fork branch._