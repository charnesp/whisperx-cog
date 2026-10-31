## Why

The transcriber cuts audio into windows (30 s by default) and turns each window
into ONE segment; diarization then labels that segment with its dominant
speaker. A 30 s window where three people speak therefore comes back as a single
"turn" with one label. Measured on a real 1 h 25 interview (2026-10-29): 169
segments, of which 144 carried words from at least two speakers; talk time was
attributed 63.9 min to the dominant speaker instead of 54.0 min, an 18 % error.
Every consumer of the service inherits the defect, including the OpenAI-
compatible transcription surface.

The information needed to fix it is already in the output: each word carries its
own speaker (pyannote) and its own timestamps (ForcedAligner). Only the segment
structure is wrong. `whisperx.assign_word_speakers` labels the segment by
dominant overlap and does not split it (upstream code, verified 2026-10-30), so
the split has to be done here.

## What Changes

- New pure module `segment_resegmentation.py`: rebuilds segments from speaker
  turns, redistributing the window's punctuated text over each turn.
- Segments produced after diarization SHALL follow speaker turns: `start`/`end`
  from the first/last word of the turn, `speaker` per turn, `words` restricted to
  the turn. Text is never invented, truncated or reordered: a segment whose
  word/token alignment is not exact is returned untouched.
- Exact-coverage guard: every non-whitespace character of the window text must
  belong to a turn, detached punctuation included (a naive normalized comparison
  silently dropped a detached final "?"; caught in review).
- Each word of a returned segment carries its speaker, including words that had
  none (they inherit the previous word's speaker, else the segment label).
- Wiring: one call after `diarize(...)` in `predict.py`, guarded so that any
  failure keeps the raw segments and logs the reason (a prediction never fails
  for this).
- Adjacent turns of the same speaker are merged when the gap is under 0.8 s, to
  avoid fragmenting a turn at every audio window boundary.

## Non-goals

- No change to the ASR engine, the aligner, the diarization model, or to the
  window plan: only the segment structure changes.
- No change to the HTTP contract: same fields (`segments`, `detected_language`,
  `speaker_embeddings`), same endpoint, no new parameter.
- No change to the faster-whisper path beyond the same post-processing (it
  benefits from the same split; its segments are already short, so few splits
  occur).
- No re-transcription, no new model, no GPU memory change.
- Not a substitute for word-level timestamps: the turn bounds come from the
  existing word timestamps, not from a new alignment pass.

## Capabilities

### Modified Capabilities

- `openai-stt-api`: segments returned after diarization follow speaker turns
  instead of audio windows; every word carries its speaker.
