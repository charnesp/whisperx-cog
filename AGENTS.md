# AGENTS.md — Agent entry map

Short pointer file for AI agents. **Do not treat this as the full encyclopedia** — follow links into `docs/` for depth.

User-facing setup: **[README.md](./README.md)**.

## Harness commands (run before claiming done)

```bash
make -f Makefile.harness smoke   # bridge sync, compile, yaml, artifact check
make -f Makefile.harness check   # smoke + unit tests (no GPU)
make -f Makefile.harness ci        # full gate (same as CI)
make -f Makefile.harness audit     # docs / entropy audit
```

After editing **`bridge/*.py`**: push to rebuild `ghcr.io/charnesp/whisperx-cog-bridge:latest`, then `make smoke`.

## Documentation map

| Doc | Contents |
|-----|----------|
| [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md) | Components, layers, data flow, boundaries |
| [docs/BRIDGE.md](./docs/BRIDGE.md) | Bridge behavior, webhook errors, logging, sync |
| [docs/OBSERVABILITY.md](./docs/OBSERVABILITY.md) | Health checks, log prefixes, troubleshooting |
| [docs/DATA_CONTRACTS.md](./docs/DATA_CONTRACTS.md) | Prediction I/O shapes, JSON boundary rules |
| [docs/TESTING.md](./docs/TESTING.md) | **Strict TDD** (RED→GREEN→REFACTOR); OpenSpec rules in `openspec/config.yaml` |
| [PLANS.md](./PLANS.md) | Active plans, completed work, tech debt |

## Code layout

| Path | Role |
|------|------|
| `predict.py` | Cog Predictor — transcribe / align / diarize |
| `json_sanitize.py` | JSON-safe output (NaN/inf); **unit-tested, no torch** |
| `model_paths.py` | Baked `/models` vs HF fallback; **unit-tested, no torch** |
| `bridge/bridge.py` | Bridge source of truth (see dual-copy rule in docs/BRIDGE.md) |
| `k8s/whisperx-stack.yaml` | Kubernetes stack (Cog + Redis + bridge GHCR image) |
| `docker-compose.yml` | Compose stack (mounts bridge.py) |
| `scripts/check-bridge-sync.py` | Verify k8s uses bridge GHCR image |
| `Makefile.harness` | Deterministic smoke / check / ci |

Image: `ghcr.io/charnesp/whisperx-cog:latest`.

## Critical invariants

1. **Strict TDD** — failing test before production code (RED→GREEN→REFACTOR); see [docs/TESTING.md](./docs/TESTING.md). Run `make -f Makefile.harness check` after each cycle.
2. **JSON output** — all prediction floats pass `json_sanitize.sanitize_for_json` before return. NaN causes Cog webhook failure.
3. **Bridge image** — k8s and Compose use `ghcr.io/charnesp/whisperx-cog-bridge:latest`; source of truth is `bridge/*.py`; GHCR rebuild on push to `main`.
4. **cog_runtime disabled** — coglet rejects dict/list in Output; do not enable without fixing Output types.
5. **Client webhook** — if client sends own `webhook`, bridge does not store in Redis; `GET /predictions/<id>` won't work.

## Cog / Replicate API (summary)

- Prediction fields: `status`, `output`, `error` (failed only), `metrics`
- Async: `Prefer: respond-async` → 202; updates via webhooks only
- Webhook events: `start`, `completed` (bridge injects both when no client webhook)
- Exception in `predict()` → `status: failed`, worker stays up (see Cog runtime table in README)

## Health checks

- Bridge liveness: `GET /health` :8080 (no auth)
- Cog readiness: `GET /health-check` :8080 proxied; JSON `status` must be `READY`
- This predictor: `healthcheck()` = `torch.cuda.is_available()`

## References

- [Cog HTTP API](https://cog.run/http/)
- [Replicate HTTP API](https://replicate.com/docs/reference/http)
- [WhisperX](https://github.com/m-bain/whisperX)
- [Harness Engineering](https://openai.com/index/harness-engineering/)
