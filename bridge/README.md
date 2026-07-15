# Bridge HTTP server

Replicate-compatible API proxy for self-hosted Cog + Redis stacks.

**Source of truth:** this file (`bridge/bridge.py`).

The same script is also embedded in the Kubernetes ConfigMap `cog-bridge-script` inside `k8s/whisperx-stack.yaml`. Docker Compose mounts this file directly.

After editing here:

```bash
python3 scripts/sync-bridge-to-k8s.py    # update k8s ConfigMap
python3 scripts/check-bridge-sync.py     # verify both copies match
```

See [README.md](../README.md) (self-hosted deployment) and [AGENTS.md](../AGENTS.md) (bridge behavior, logging, error codes).
