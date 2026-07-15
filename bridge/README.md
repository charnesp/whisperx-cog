# Bridge HTTP server

Replicate-compatible API proxy for self-hosted Cog + Redis stacks.

**Source of truth:** `bridge/bridge.py` and `bridge/openai_compat.py`.

| Deployment | How loaded |
|------------|------------|
| Docker Compose | `ghcr.io/charnesp/whisperx-cog-bridge:latest` — built from `bridge/Dockerfile` via GitHub Actions |
| Kubernetes | same GHCR image in `k8s/whisperx-stack.yaml` |

After editing here:

```bash
python3 scripts/check-bridge-sync.py     # verify k8s references GHCR image
# after edits: push to main → GHCR rebuild → kubectl rollout restart
make -f Makefile.harness smoke           # full pre-commit gate (includes sync check)
```

See [README.md](../README.md) (self-hosted deployment) and [AGENTS.md](../AGENTS.md) (bridge behavior, logging, error codes).
