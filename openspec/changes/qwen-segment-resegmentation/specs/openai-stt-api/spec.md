## ADDED Requirements

### Requirement: Segments follow speaker turns

After diarization, the segments returned by the predictor SHALL follow speaker
turns rather than audio windows. For each audio window whose words carry more
than one speaker, the predictor SHALL return one segment per consecutive run of
same-speaker words, with `start` and `end` taken from the first and last word of
the run, `speaker` set to the run's speaker, and `words` restricted to that run.
A window whose words carry a single speaker SHALL be returned unchanged. Every
word of a returned segment SHALL carry its speaker, including words that had
none at the input (they inherit the previous word's speaker, else the segment
label).

#### Scenario: One window, two speakers

- **WHEN** a window of text `Bonjour, comment allez-vous ? Tres bien.` carries three words labelled `SPEAKER_00` then two labelled `SPEAKER_01`
- **THEN** the output contains two segments, `Bonjour, comment allez-vous ?` for `SPEAKER_00` and `Tres bien.` for `SPEAKER_01`
- **AND** their bounds are the first and last word timestamps

#### Scenario: One window, one speaker

- **WHEN** every word of a window carries the same speaker
- **THEN** exactly one segment is returned, with the original text

#### Scenario: A monologue spanning several windows

- **WHEN** five consecutive windows carry one speaker and no silence between them
- **THEN** five segments are returned, not one long block: adjacent turns of the same speaker are only merged when the gap is under 0.8 s AND the merged turn stays within one audio window (30 s)

#### Scenario: A turn cut by a window boundary

- **WHEN** a turn of the same speaker continues in the next window after less than 0.8 s of silence
- **THEN** the two parts are merged into one segment, because the cut came from the audio windowing, not from the conversation

#### Scenario: Every segment carries its character count

- **WHEN** any segment is returned, split or not
- **THEN** it carries `chars`, its number of characters, so a consumer needs no special case

#### Scenario: Window without words

- **WHEN** a segment carries no words
- **THEN** it is returned unchanged

### Requirement: Text is never invented, truncated or reordered

The re-segmentation SHALL preserve the characters of the window text: the
concatenation of the produced turns, whitespace ignored, SHALL equal the original
window text. When the word-to-token mapping is not exact, or when the window has
no text, the original segment SHALL be returned untouched. The only error raised
SHALL be `TypeError` on a malformed entry (words not a list of dicts, text not a
string).

#### Scenario: Detached punctuation at the end of the window

- **WHEN** the window text ends with a detached `?` after the last word
- **THEN** the `?` stays in the last turn's text

#### Scenario: Alignment mismatch

- **WHEN** the window text contains a word that does not correspond to the next timestamped word
- **THEN** the segment is returned unchanged and no text is fabricated

#### Scenario: Malformed entry

- **WHEN** a segment's `words` is a list of strings
- **THEN** the helper raises `TypeError`

### Requirement: Windows left whole are reported

A window whose word/text alignment is not exact SHALL be returned untouched, and
the number of such windows SHALL be logged by the predictor as
`segment resegmentation: N/M windows kept whole (word/text alignment not exact)`.
A run where the re-segmentation did nothing SHALL NOT be indistinguishable from a
run where it succeeded.

#### Scenario: Inexact alignment

- **WHEN** a window's text cannot be mapped to its words
- **THEN** the window is returned untouched and the count of untouched windows is logged

### Requirement: Guarded wiring in the prediction path

The predictor SHALL apply the re-segmentation immediately after diarization,
inside a guard: if it raises, the raw segments SHALL be returned and the failure
SHALL be logged as `segment resegmentation skipped: <type>: <message>`. The
prediction SHALL NOT fail because of the re-segmentation. The fields of the
output (`segments`, `detected_language`, `speaker_embeddings`) and the HTTP
contract SHALL be unchanged.

#### Scenario: Re-segmentation raises

- **WHEN** the helper raises during a prediction
- **THEN** the prediction succeeds with the raw segments and a `segment resegmentation skipped:` line is logged

#### Scenario: Contract unchanged

- **WHEN** a client reads the prediction output
- **THEN** the same keys are present with the same types
