# Tasks - qwen-segment-resegmentation

Strict TDD per `docs/TESTING.md`: every behaviour is listed test-first.

## 1. Pure helper (RED -> GREEN)

- [x] 1.1 RED: write `tests/test_segment_resegmentation.py` (split at speaker
  change, punctuation, no fabrication, labels, merge, idempotence, result helper)
- [x] 1.2 Verify RED: `make -f Makefile.harness check` fails with
  `ModuleNotFoundError: No module named 'segment_resegmentation'` (5 errors)
- [x] 1.3 GREEN: create `segment_resegmentation.py` with `_spans`,
  `resegment_segments`, `resegment_result`
- [x] 1.4 Verify GREEN: `make -f Makefile.harness check` -> 586 tests OK
- [x] 1.5 Add the empty-text guard (window with no text is kept whole) and the
  exact-coverage guard, then re-run the gate

## 2. Wiring (RED -> GREEN by pure extraction)

- [x] 2.1 The wiring behaviour (guard + log + identity on failure) lives in
  `resegment_result`, which is unit-tested (no segments, no speaker, input not
  mutated)
- [x] 2.2 Call it in `predict.py` right after `diarize(...)`, inside
  `try/except Exception` logging `segment resegmentation skipped:`
- [x] 2.3 Compile + `make -f Makefile.harness check` green after wiring

## 3. Documentation

- [x] 3.1 `openspec/changes/qwen-segment-resegmentation/` (proposal, design,
  tasks, delta spec)
- [x] 3.2 `docs/DATA_CONTRACTS.md`: segments are speaker turns, every word
  carries its speaker
- [x] 3.3 `PLANS.md`: entry for this change

## 4. Verification before done

- [x] 4.1 `make -f Makefile.harness check` green
- [ ] 4.2 `make -f Makefile.harness ci` green (lint, pkg-audit, audit)
- [ ] 4.3 GPU smoke on a real recording: compare segment count and per-speaker
  talk time against the previous output (manual, per docs/TESTING.md)
- [ ] 4.4 Deploy: image rebuild + canary, then production (operator decision)

## 5. Cold review round (2026-10-31)

- [x] 5.1 RED: `tests/test_resegmentation_review.py` — merge must not swallow
  monologue windows, tokens spanning two words (`allez-vous`, `l'homme`), words
  with no alphanumeric content, `chars` on every segment, counters reaching the
  caller (9 tests)
- [x] 5.2 GREEN: merge capped at `MAX_TURN_S` (30 s), one token may cover up to
  three words, content-less words inherit a neighbour span, coverage computed on
  the union of spans, empty turn = block kept whole
- [x] 5.3 Predictor logs untouched windows instead of hiding them
- [x] 5.4 `tests/test_segment_resegmentation_property.py`: 300 random windows and
  100 idempotence cases
- [x] 5.5 `make -f Makefile.harness check` green (612 tests)
