"""Re-segment ASR output at speaker changes.

Why this module exists
----------------------
The transcriber splits audio into windows (30 s by default) and receives one
text per window. Each window becomes ONE segment, and diarization then labels
that segment with its dominant speaker. A 30 s window where three people speak
is therefore a single "turn": downstream consumers cannot attribute speech.

Every word carries its own speaker (pyannote), so the information is already
there; only the segment structure is wrong. This module rebuilds segments from
speaker turns instead of audio windows, and redistributes the punctuated window
text over those turns.

Punctuation lives on the segment text, not on the words, so the text is sliced
by matching each timestamped word to its token in the text. Any segment whose
word/token alignment is not exact is left untouched: never invent, never
truncate. Pure Python, stdlib only, no torch: unit-tested on plain dicts.
"""

from __future__ import annotations

import re
import unicodedata

# Merge two consecutive turns of the same speaker when the silence between them
# is shorter than this: avoids fragmenting one turn at every audio window.
MERGE_GAP_S = 0.8


def normalize(token: str) -> str:
    """Comparable form of a token: lowercase, no accents, no punctuation."""
    s = unicodedata.normalize("NFD", token.lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9']", "", s)


def _spans(text: str, words: list[dict]):
    """Map each timestamped word to its character span in `text`.

    Returns a list of (start, end) with one entry per word, or None when the
    mapping is not exact (a significant token without a word, a word that does
    not match its token, or a character left uncovered).

    Detached punctuation never consumes a word: it is attached to the previous
    turn, or to the next one when no word has been placed yet (leading
    punctuation). Punctuation after the last word is attached to the last turn;
    without that, a detached final question mark would silently disappear.
    """
    spans: list[tuple[int, int]] = []
    pos = 0
    lead_start = None
    for w in words:
        want = normalize(w.get("word", ""))
        if not want:
            return None
        matched = None
        while pos < len(text):
            m = re.search(r"\S+", text[pos:])
            if not m:
                break
            start = pos + m.start()
            end = pos + m.end()
            pos = end
            got = normalize(text[start:end])
            if not got:
                if spans:
                    spans[-1] = (spans[-1][0], end)
                elif lead_start is None:
                    lead_start = start
                continue
            if got != want:
                return None
            matched = (lead_start if (not spans and lead_start is not None) else start, end)
            lead_start = None
            break
        if matched is None:
            return None
        spans.append(matched)

    # Detached punctuation after the last word.
    while pos < len(text):
        m = re.search(r"\S+", text[pos:])
        if not m:
            break
        start = pos + m.start()
        end = pos + m.end()
        pos = end
        if normalize(text[start:end]):
            return None
        if not spans:
            return None
        spans[-1] = (spans[-1][0], end)

    # Exact coverage: whitespace removed, every other character must be covered.
    covered = re.sub(r"\s+", "", "".join(text[s:e] for s, e in spans))
    if covered != re.sub(r"\s+", "", text):
        return None
    return spans


def resegment_segments(segments: list[dict]) -> tuple[list[dict], dict]:
    """Split segments at speaker changes.

    Doubtful content is left untouched (no exception); a MALFORMED entry does
    raise TypeError (words must be dicts, text must be a string).
    """
    stats = {"in": len(segments), "out": 0, "split": 0, "no_words": 0,
             "ambiguous": 0, "single_speaker": 0,
             "words_in": 0, "words_out": 0}
    out: list[dict] = []
    for seg in segments:
        raw_words = seg.get("words") or []
        stats["words_in"] += len(raw_words)
        text = seg.get("text") or ""
        if not isinstance(text, str) or any(not isinstance(w, dict) for w in raw_words):
            raise TypeError("malformed segment: words must be a list of dicts, text a string")

        # A word without a label inherits the previous word's speaker (temporal
        # continuity), else the segment label; the inherited label is WRITTEN
        # into the word: the contract is that every word carries its speaker.
        seg_speaker = seg.get("speaker")
        words = []
        current = seg_speaker
        for w in raw_words:
            sp = w.get("speaker") or current
            current = sp
            words.append(dict(w, speaker=sp))
        if not words or not any(w.get("speaker") for w in words):
            stats["no_words"] += 1
            out.append(seg)
            stats["words_out"] += len(words)
            continue

        runs = []
        for w in words:
            sp = w.get("speaker")
            if runs and runs[-1]["speaker"] == sp:
                runs[-1]["words"].append(w)
            else:
                runs.append({"speaker": sp, "words": [w]})

        if len(runs) < 2:
            stats["single_speaker"] += 1
            out.append(dict(seg, words=words))
            stats["words_out"] += len(words)
            continue

        spans = _spans(text, words) if text.strip() else None
        if not text.strip() or spans is None:
            # No text to distribute (or inexact alignment): keep the block whole
            # rather than emit empty turns.
            stats["ambiguous"] += 1
            out.append(dict(seg, words=words))
            stats["words_out"] += len(words)
            continue

        index = 0
        pieces = []
        for run in runs:
            first = index
            index += len(run["words"])
            piece_text = ""
            if spans is not None:
                piece_text = text[spans[first][0]:spans[index - 1][1]].strip()
            pieces.append({
                "start": run["words"][0].get("start", seg.get("start", 0.0)),
                "end": run["words"][-1].get("end", seg.get("end", 0.0)),
                "text": piece_text,
                "chars": len(piece_text),
                "speaker": run["speaker"],
                "words": run["words"],
            })

        # Second guard: the turns must render exactly the original characters.
        rebuilt = " ".join(p["text"] for p in pieces)
        if re.sub(r"\s+", "", rebuilt) != re.sub(r"\s+", "", text):
            stats["ambiguous"] += 1
            out.append(dict(seg, words=words))
            stats["words_out"] += len(words)
            continue

        stats["split"] += 1
        stats["words_out"] += len(words)
        out.extend(pieces)

    merged: list[dict] = []
    for piece in out:
        prev = merged[-1] if merged else None
        if (prev is not None and prev.get("speaker") == piece.get("speaker")
                and piece.get("start", 0.0) - prev.get("end", 0.0) <= MERGE_GAP_S
                and "words" in prev and "words" in piece):
            prev["end"] = piece.get("end", prev.get("end", 0.0))
            prev["text"] = (prev.get("text", "") + " " + piece.get("text", "")).strip()
            prev["chars"] = len(prev["text"])
            prev["words"] = prev["words"] + piece["words"]
            continue
        merged.append(dict(piece))

    stats["out"] = len(merged)
    return merged, stats


def resegment_result(result: dict) -> dict:
    """Return a copy of a prediction result with segments rebuilt per speaker turn.

    A result without segments, or without any speaker information, comes back
    unchanged. The input dict is never mutated.
    """
    if not isinstance(result, dict):
        raise TypeError("result must be a dict")
    segments = result.get("segments")
    if not isinstance(segments, list) or not segments:
        return result
    new_segments, _ = resegment_segments(segments)
    new = dict(result)
    new["segments"] = new_segments
    return new
