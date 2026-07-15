# Testing policy

How this repo verifies behavior. Complements [Harness Engineering](https://openai.com/index/harness-engineering/) (`Makefile.harness`, `scripts/harness_audit.py`).

## Not strict TDD

This project **does not** use test-driven development in the strict sense (failing test first → minimal code → refactor). Do **not** require red-green-refactor cycles or delete implementation to “restart from tests.”

**Adopted model:** spec-driven changes (OpenSpec) with **test-with / test-after** unit tests on deterministic, GPU-free code paths.

| Strict TDD | This repo |
|------------|-----------|
| Test before any production code | Implementation and tests may land together |
| Watch test fail before writing code | Tests assert behavior once the API shape is known |
| One micro-cycle per behavior | OpenSpec tasks: feature work, then a dedicated tests section |

## What is required

1. **Harness gate** — run `make -f Makefile.harness check` (or `ci`) before claiming a change is done.
2. **Unit tests (no GPU)** — pure Python modules that agents and CI can run on every machine:
   - `json_sanitize.py` → `tests/test_json_sanitize.py`
   - `bridge/openai_compat.py` and bridge routing → `tests/test_openai_stt.py`
   - k8s bridge image invariant → `tests/test_bridge_k8s.py`
3. **Mocks, not Cog** — bridge tests mock Cog HTTP responses; they never start WhisperX or need CUDA.
4. **Same commit / PR** — new bridge or sanitize behavior ships with matching tests; do not defer tests to a follow-up PR unless explicitly scoped in OpenSpec non-goals.

## What is out of scope (for now)

| Area | Policy |
|------|--------|
| `predict.py` (WhisperX / torch / GPU) | No unit tests in default harness; manual or GPU CI (`workflow_dispatch`) only — see [PLANS.md](../PLANS.md) |
| End-to-end audio quality | Manual smoke against a running stack, not CI |
| pytest | Use stdlib `unittest` via `make check` |

## OpenSpec task ordering

When authoring `tasks.md` for a change:

1. Document decisions in `design.md` / delta specs first.
2. Group **implementation** tasks (modules, routes, converters).
3. Group **tests** in a dedicated section (e.g. `## N. Tests`) covering happy paths, validation errors, and HTTP boundaries with mocked Cog.
4. Close with harness: `make -f Makefile.harness check`.

Do **not** reorder tasks to enforce test-first TDD; do **not** omit the tests section.

## Adding tests

- Place files under `tests/test_*.py`; discovered by `unittest discover`.
- Prefer testing pure functions and `handle_*` orchestrators with injected `urlopen_fn` (see `tests/test_openai_stt.py`).
- Keep fixtures (`MOCK_COG_OUTPUT`, etc.) beside tests or in the module under test when shared.

## Related

- [AGENTS.md](../AGENTS.md) — harness commands
- [DATA_CONTRACTS.md](./DATA_CONTRACTS.md) — shapes asserted in tests
- [OBSERVABILITY.md](./OBSERVABILITY.md) — smoke vs check vs audit
- [openspec/config.yaml](../openspec/config.yaml) — agent rules for OpenSpec artifacts
