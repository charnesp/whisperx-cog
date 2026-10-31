# Testing policy

How this repo verifies behavior. Complements [Harness Engineering](https://openai.com/index/harness-engineering/) (`Makefile.harness`, `scripts/harness_audit.py`).

## Strict TDD (required)

This project **requires test-driven development** in the strict sense:

1. **RED** — write a failing test for one behavior
2. **Verify RED** — run `make -f Makefile.harness check` and confirm the new test fails for the expected reason
3. **GREEN** — write the minimal production code to pass
4. **Verify GREEN** — full test suite green
5. **REFACTOR** — clean up while staying green

**Iron law:** no production code without a failing test first. If code was written before its test, delete the code and implement again from the test.

## Scope

| Layer | TDD |
|-------|-----|
| `json_sanitize.py` | Strict TDD — `tests/test_json_sanitize.py` |
| `bridge/openai_compat.py`, `bridge/bridge.py` | Strict TDD — mock Cog, no GPU — `tests/test_openai_stt.py` |
| `scripts/bridge_k8s.py`, k8s invariants | Strict TDD — `tests/test_bridge_k8s.py` |
| `predict.py` (WhisperX / torch / GPU) | Extract pure helpers where possible and TDD them; GPU pipeline verified by manual smoke or scoped GPU CI — see [PLANS.md](../PLANS.md) |

## Harness gate

Run `make -f Makefile.harness check` (or `ci`) after every RED/GREEN cycle and before claiming done.

- stdlib `unittest` only (no pytest)
- bridge tests inject `urlopen_fn`; never start Cog or WhisperX in unit tests
- test venv needs `pyyaml` + `requests` + `numpy`:
  `requests` is used by `scripts/lock_audit.py` (and its tests); `numpy` is imported
  by `tests/test_qwen_split_energy_vectorised.py` and the `tests/test_qwen_remote_*.py`
  files, and its absence kept CI red from 2026-10-20 to 2026-10-31;
  `make ci` also needs `ruff` and `pip-audit` on PATH

## OpenSpec

Agent rules live **only** in [`openspec/config.yaml`](../openspec/config.yaml). Per-change `.openspec.yaml` is metadata (`schema`, `created`) only.

When authoring `tasks.md`:

1. Document decisions in `design.md` / delta specs.
2. For **each behavior**, order tasks **test first → implementation → refactor** (RED → GREEN → REFACTOR).
3. Group by feature slice, not “all impl then all tests”.
4. Close with `make -f Makefile.harness check`.

## Branch protection on `main`

`main` is protected since 2026-10-31, administrators included:

- no direct push: every change lands through a pull request (zero approvals required,
  so a solo maintainer is not blocked)
- required checks before merge: `harness` and `lock-audit`, on a branch up to date
  with `main` (`strict`)
- force pushes and deletions refused

Why: production follows `ghcr.io/charnesp/whisperx-cog:latest`, which is published
from `main`, so a red commit reaching `main` ends up on the GPU host. The publish
workflow also gates the build on `make -f Makefile.harness ci` (`gate` job), which
covers images built from feature branches.

## Related

- [AGENTS.md](../AGENTS.md) — harness commands
- [DATA_CONTRACTS.md](./DATA_CONTRACTS.md) — shapes asserted in tests
- [OBSERVABILITY.md](./OBSERVABILITY.md) — smoke vs check vs audit
- [openspec/config.yaml](../openspec/config.yaml) — OpenSpec TDD rules for agents
