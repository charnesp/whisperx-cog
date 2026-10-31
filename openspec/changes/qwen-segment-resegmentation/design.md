## Context

`predict.py` builds the transcription result, aligns it (ForcedAligner for qwen,
wav2vec2 for whisper), then diarizes with pyannote and calls
`whisperx.assign_word_speakers`. That function labels each segment with the
speaker covering most of its duration and labels each word; it does not re-cut
segments. With 30 s windows the result is one turn per window.

Measured on a real recording (2026-10-29, 1 h 25): 169 segments, 144 containing
words from two or more speakers; after re-segmentation the same file yields 560
turns and the per-speaker talk time converges with an independent turbo run
(54.0 / 19.0 / 4.9 min against 53.3 / 18.1 / 4.4 min), where the raw output
credited 63.9 min to the dominant speaker.

## Decisions

### D1 - Pure module, no torch, no GPU

`segment_resegmentation.py` takes and returns plain dicts (the JSON shapes of the
prediction output). It imports only `re` and `unicodedata`. This keeps the whole
behaviour under strict TDD in the GPU-free layer, as required by
`docs/TESTING.md` (`json_sanitize`-style scope).

### D2 - Segments follow turns, words keep their timestamps

A turn's `start`/`end` are the first word's start and the last word's end. No new
alignment pass: the words already carry ForcedAligner timestamps. The window
bounds are only used when a word lacks timestamps.

### D3 - Text is redistributed, never regenerated

Punctuation lives on the segment text, not on the words, so the text is sliced by
mapping each word to its token in the window text (accent- and case-normalised
comparison, punctuation-insensitive). Detached punctuation attaches to the
previous turn, or to the next one at the start of a window; punctuation after the
last word attaches to the last turn.

### D4 - Exact-coverage guard, and fail closed

Two guards, in this order:

1. `_spans` returns `None` unless every non-whitespace character of the text is
   covered by a word span (this is what catches a dropped detached "?").
2. The concatenation of the produced turns must render exactly the original
   characters (whitespace ignored).

When a guard fails, or when the window has no text, the original segment is
returned untouched: the service must prefer a coarse block over an invented or
truncated transcript. The only exception raised is `TypeError` on a malformed
entry (words not dicts, text not a string).

### D5 - Wiring is guarded, and visible

`predict.py` calls `resegment_result(result)` right after `diarize(...)`, inside
`try/except Exception`: on failure the raw segments are returned and a line
`segment resegmentation skipped: <type>: <message>` is printed. No silent
fallback: the reason is in the logs, and the prediction still succeeds.

### D6 - Same-speaker turns are merged across window boundaries

Two adjacent turns of the same speaker closer than `MERGE_GAP_S` (0.8 s) are
merged. Without it, a turn spanning a window boundary is split in two, and the
downstream transcript shows artificial turn changes.

## Test boundaries

- Everything is tested on plain dicts: no Cog, no torch, no audio, no GPU.
- `tests/test_segment_resegmentation.py` covers: split at speaker change (2 and 3
  speakers), single-speaker and word-less windows untouched, detached punctuation
  at both ends, character-level survival, no fabrication on word/text mismatch,
  word count preserved, malformed input raising, inherited labels written into
  words, merge and no-merge on a long gap, idempotence, and the result-level
  helper (no segments, no speaker, input not mutated).
- The GPU path itself stays on manual smoke per `docs/TESTING.md`; the wiring is
  a single guarded call whose behaviour on failure is the identity.

## Risks

- Over-segmentation if the diarization labels are noisy: mitigated by the merge
  threshold, and by the fact that turns are built from word labels rather than
  from independent boundary detection.
- A mis-aligned window is now left whole: the defect is not fixed for that
  window, but nothing is lost. `ambiguous` is counted in the returned stats.
- Consumers that assumed "one segment = one window" see more segments. The fields
  and their meaning are unchanged, and the OpenAI-compatible surface returns the
  same keys.
