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
by matching each timestamped word to its token in the text. Any window whose
word/token alignment is not exact is returned untouched: never invent, never
truncate. Pure Python, stdlib only, no torch: unit-tested on plain dicts.
"""

from __future__ import annotations

import re
import unicodedata

# Two consecutive turns of the same speaker are merged when the silence between
# them is short, so a turn is not cut at every audio window boundary.
MERGE_GAP_S = 0.8

# ... but a merge never produces a turn longer than one audio window: a long
# monologue stays a series of normal segments, not one 25-minute block.
MAX_TURN_S = 30.0

# How many words one text token may cover: "allez" + "-vous" for "allez-vous",
# "l'" + "homme" for "l'homme".
MAX_WORDS_PER_TOKEN = 3


def normalize(token: str) -> str:
    """Comparable form of a token: lowercase, no accents, no punctuation."""
    s = unicodedata.normalize("NFD", token.lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9']", "", s)


def _spans(text: str, words: list[dict]):
    """Map each word to its (start, end) character span in `text`.

    Returns one span per word, or None when the mapping is not exact: a token
    with no matching word, a group of words that does not spell the token, or a
    group whose words belong to different speakers.

    Rules:
    - a word with no alphanumeric content (a lone "?", a dash) takes no text and
      inherits the span of its neighbour;
    - one text token may cover up to MAX_WORDS_PER_TOKEN words, which is how
      hyphenation ("allez-vous") and elision ("l'homme") are handled;
    - punctuation detached in the text attaches to the previous word, or to the
      next one at the start of the window; after the last word it attaches to the
      last turn, so a detached final "?" never vanishes.
    """
    count = len(words)
    spans: list = [None] * count
    pos = 0
    lead = None
    index = 0

    while index < count:
        if not normalize(words[index].get("word", "")):
            index += 1
            continue

        match = re.search(r"\S+", text[pos:])
        if not match:
            return None
        start, end = pos + match.start(), pos + match.end()
        pos = end
        token = normalize(text[start:end])
        if not token:
            placed = [i for i in range(index) if spans[i] is not None]
            if placed:
                last = placed[-1]
                spans[last] = [spans[last][0], end]
            else:
                lead = start if lead is None else lead
            continue

        for width in range(1, MAX_WORDS_PER_TOKEN + 1):
            if index + width > count:
                return None if width == 1 else None
            group = words[index:index + width]
            if any(not normalize(w.get("word", "")) for w in group):
                continue
            if "".join(normalize(w.get("word", "")) for w in group) != token:
                continue
            if len({w.get("speaker") for w in group}) > 1:
                return None
            for i in range(index, index + width):
                spans[i] = [start, end]
            if lead is not None:
                spans[index] = [lead, end]
                lead = None
            index += width
            break
        else:
            return None

    # Words with no content: inherit the previous span, else the next one.
    for i in range(count):
        if spans[i] is None:
            previous = next((spans[j] for j in range(i - 1, -1, -1) if spans[j]), None)
            following = next((spans[j] for j in range(i + 1, count) if spans[j]), None)
            spans[i] = list(previous or following or [0, 0])
    if not any(spans):
        return None

    # Detached punctuation after the last word.
    while pos < len(text):
        match = re.search(r"\S+", text[pos:])
        if not match:
            break
        start, end = pos + match.start(), pos + match.end()
        pos = end
        if normalize(text[start:end]):
            return None
        spans[-1][1] = end

    # Coverage over the union of the spans: a group of words shares one span, so
    # concatenating per word would count it twice.
    ordered = []
    for start, end in spans:
        if ordered and start < ordered[-1][1]:
            ordered[-1] = (ordered[-1][0], max(ordered[-1][1], end))
        else:
            ordered.append((start, end))
    covered = re.sub(r"\s+", "", "".join(text[s:e] for s, e in ordered))
    if covered != re.sub(r"\s+", "", text):
        return None
    return [(s, e) for s, e in spans]


def _bound(word: dict, key: str, fallback):
    """A number, or the window bound. None stays None: the caller then keeps the
    block whole rather than emit a segment the bridge cannot convert."""
    value = word.get(key)
    if isinstance(value, (int, float)):
        return value
    return fallback if isinstance(fallback, (int, float)) else None


def _with_chars(segment: dict) -> dict:
    """Every returned segment carries `chars`, so consumers need no special case."""
    if "chars" in segment:
        return segment
    return dict(segment, chars=len(segment.get("text") or ""))


def resegment_segments(segments: list[dict], stats: dict | None = None):
    """Split segments at speaker changes.

    Doubtful content is left untouched (no exception); a MALFORMED entry does
    raise TypeError (words must be dicts, text must be a string). When `stats` is
    given, it receives the counters of the run.
    """
    counters = {"in": len(segments), "out": 0, "split": 0, "no_words": 0,
                "ambiguous": 0, "single_speaker": 0,
                "words_in": 0, "words_out": 0}
    out: list[dict] = []
    for seg in segments:
        raw_words = seg.get("words") or []
        counters["words_in"] += len(raw_words)
        text = seg.get("text") or ""
        if not isinstance(text, str) or any(not isinstance(w, dict) for w in raw_words):
            raise TypeError("malformed segment: words must be a list of dicts, text a string")

        # A word without a label inherits the previous word's speaker (temporal
        # continuity), else the segment label; the inherited label is WRITTEN
        # into the word: the contract is that every word carries its speaker.
        words = []
        current = seg.get("speaker")
        for w in raw_words:
            sp = w.get("speaker") or current
            current = sp
            words.append(dict(w, speaker=sp))
        if not words or not any(w.get("speaker") for w in words):
            counters["no_words"] += 1
            out.append(_with_chars(seg))
            counters["words_out"] += len(words)
            continue

        runs = []
        for w in words:
            sp = w.get("speaker")
            if runs and runs[-1]["speaker"] == sp:
                runs[-1]["words"].append(w)
            else:
                runs.append({"speaker": sp, "words": [w]})

        if len(runs) < 2:
            counters["single_speaker"] += 1
            out.append(_with_chars(dict(seg, words=words)))
            counters["words_out"] += len(words)
            continue

        spans = _spans(text, words) if text.strip() else None
        if not text.strip() or spans is None:
            # No text to distribute (or inexact alignment): keep the block whole
            # rather than emit empty turns.
            counters["ambiguous"] += 1
            out.append(_with_chars(dict(seg, words=words)))
            counters["words_out"] += len(words)
            continue

        index = 0
        pieces = []
        bounds_ok = True
        for run in runs:
            first = index
            index += len(run["words"])
            piece_text = text[spans[first][0]:spans[index - 1][1]].strip()
            start = _bound(run["words"][0], "start", seg.get("start"))
            end = _bound(run["words"][-1], "end", seg.get("end"))
            if start is None or end is None:
                bounds_ok = False
                break
            pieces.append({
                "start": start,
                "end": end,
                "text": piece_text,
                "chars": len(piece_text),
                "speaker": run["speaker"],
                "words": run["words"],
                "_split": True,
            })

        # Second guard: the turns must render exactly the original characters,
        # no turn may be empty, and every turn must carry usable bounds (the
        # bridge compares and converts them).
        rebuilt = " ".join(p["text"] for p in pieces)
        if not bounds_ok or any(not p["text"] for p in pieces) or \
                re.sub(r"\s+", "", rebuilt) != re.sub(r"\s+", "", text):
            counters["ambiguous"] += 1
            out.append(_with_chars(dict(seg, words=words)))
            counters["words_out"] += len(words)
            continue

        counters["split"] += 1
        counters["words_out"] += len(words)
        out.extend(pieces)

    merged: list[dict] = []
    for piece in out:
        prev = merged[-1] if merged else None
        if (prev is not None and prev.get("_split") and piece.get("_split")
                and prev.get("speaker") == piece.get("speaker")
                and piece.get("start", 0.0) - prev.get("end", 0.0) <= MERGE_GAP_S
                and piece.get("end", 0.0) - prev.get("start", 0.0) <= MAX_TURN_S
                and "words" in prev and "words" in piece):
            prev["end"] = piece.get("end", prev.get("end", 0.0))
            prev["text"] = (prev.get("text", "") + " " + piece.get("text", "")).strip()
            prev["chars"] = len(prev["text"])
            prev["words"] = prev["words"] + piece["words"]
            continue
        merged.append(dict(piece))

    for piece in merged:
        piece.pop("_split", None)

    counters["out"] = len(merged)
    if stats is not None:
        stats.update(counters)
    return merged, counters


def resegment_result(result: dict, stats: dict | None = None) -> dict:
    """Return a copy of a prediction result with segments rebuilt per speaker turn.

    A result without segments comes back unchanged. The input dict is never
    mutated. When `stats` is given, the counters of the run are written into it so
    the caller can report windows left whole instead of hiding them.
    """
    if not isinstance(result, dict):
        raise TypeError("result must be a dict")
    segments = result.get("segments")
    if not isinstance(segments, list) or not segments:
        return result
    new_segments, _ = resegment_segments(segments, stats)
    new = dict(result)
    new["segments"] = new_segments
    return new
