# Plans

Execution plans and tech-debt tracker for agent-first work. Architecture: [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md).

## Active

| Item | Scope | Acceptance |
|------|-------|------------|
| `openai-diarized-json-endpoint` | OpenAI `diarized_json` + `gpt-4o-transcribe-diarize` on bridge | See `openspec/changes/openai-diarized-json-endpoint/` |

## Completed

| Date | Plan | Outcome |
|------|------|---------|
| 2026-03 | Bridge hardening | RESP2 Redis client, structured logging, webhook validation |
| 2026-03 | Docker Compose stack | `docker-compose.yml`, `bridge/bridge.py` extracted, sync scripts |
| 2026-03 | Documentation pass | README + AGENTS restructure, bridge dual-copy workflow |
| 2026-03 | Harness bootstrap | `Makefile.harness`, docs/, unit tests, CI audit |

## Tech debt tracker

| Item | Priority | Notes |
|------|----------|-------|
| Enable `cog_runtime: true` | Medium | Blocked: coglet rejects dict/list in Output |
| GPU integration tests in CI | Low | `workflow_dispatch` only; no GPU on default runners |
| Prometheus metrics on bridge | Low | Logs sufficient for now; see OBSERVABILITY.md |
| Single-source k8s ConfigMap | Done | `scripts/sync-bridge-to-k8s.py` |
| OpenAI `known_speaker_references[]` | Low | Post `openai-diarized-json-endpoint`: reference audio clips → named speakers (OpenAI parity); blocked on WhisperX speaker-matching API — v1 returns 400 |

## How to add a plan

1. Add a row under **Active** with scope and acceptance criteria.
2. Link design notes in `docs/` if non-trivial.
3. On completion, move to **Completed** and note PR/commit.
