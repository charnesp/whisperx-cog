"""Tests added after the cold review of the resegmentation change.

Each one pins a defect that the review found: the 0.8 s merge swallowing whole
monologue windows, tokens whose text does not split on word boundaries
("allez-vous", "l'homme"), words with no alphanumeric content, `chars` missing on
untouched segments, and fallback counters thrown away.
"""

import unittest

from segment_resegmentation import MERGE_GAP_S, resegment_result, resegment_segments


def word(text, start, end, speaker=None):
    return {"word": text, "start": start, "end": end, "speaker": speaker}


def window(text, words, speaker="S0", start=None, end=None):
    return {
        "start": words[0]["start"] if start is None else start,
        "end": words[-1]["end"] if end is None else end,
        "text": text,
        "speaker": speaker,
        "words": words,
    }


class MergeKeepsNormalSegments(unittest.TestCase):
    def test_monologue_windows_are_not_merged_into_one_giant_turn(self):
        windows = []
        for k in range(5):
            t = k * 30.0
            windows.append(window(
                f"phrase {k}",
                [word(f"phrase", t, t + 1.0, "S0"), word(str(k), t + 1.0, t + 2.0, "S0")],
            ))
        out, _ = resegment_segments(windows)
        self.assertEqual(len(out), 5, "five monologue windows must stay five segments")
        self.assertLessEqual(max(s["end"] - s["start"] for s in out), 30.0)

    def test_short_turns_of_the_same_speaker_are_still_merged(self):
        first = window("un", [word("un", 0.0, 1.0, "S0")])
        second = window("deux", [word("deux", 1.0 + MERGE_GAP_S / 2, 2.0, "S0")])
        out, _ = resegment_segments([first, second])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["text"], "un deux")

    def test_gap_above_threshold_prevents_the_merge(self):
        first = window("un", [word("un", 0.0, 1.0, "S0")])
        second = window("deux", [word("deux", 1.0 + MERGE_GAP_S * 2, 3.0, "S0")])
        out, _ = resegment_segments([first, second])
        self.assertEqual(len(out), 2)


class HardTokens(unittest.TestCase):
    def test_word_without_alphanumeric_content_does_not_block_the_split(self):
        seg = window("Bonjour ? le monde",
                     [word("Bonjour", 0.0, 0.5, "S0"), word("?", 0.5, 0.6, "S0"),
                      word("le", 0.6, 0.8, "S1"), word("monde", 0.8, 1.0, "S1")])
        out, stats = resegment_segments([seg])
        self.assertEqual(stats["ambiguous"], 0, "a lone '?' must not abort the split")
        self.assertEqual([s["speaker"] for s in out], ["S0", "S1"])

    def test_hyphenated_token_spans_two_words(self):
        seg = window("Bonjour allez-vous bien",
                     [word("Bonjour", 0.0, 0.5, "S0"), word("allez", 0.6, 0.9, "S0"),
                      word("-vous", 0.9, 1.1, "S0"), word("bien", 1.2, 1.4, "S1")])
        out, stats = resegment_segments([seg])
        self.assertEqual(stats["ambiguous"], 0, "allez + -vous must map to allez-vous")
        self.assertEqual([s["speaker"] for s in out], ["S0", "S1"])
        self.assertEqual(out[1]["text"], "bien")

    def test_elided_token_spans_two_words(self):
        seg = window("l'homme est la",
                     [word("l'", 0.0, 0.3, "S0"), word("homme", 0.3, 0.6, "S0"),
                      word("est", 0.7, 0.9, "S1"), word("la", 0.9, 1.1, "S1")])
        out, stats = resegment_segments([seg])
        self.assertEqual(stats["ambiguous"], 0, "l' + homme must map to l'homme")
        self.assertEqual(out[1]["text"], "est la")


class ContractOfTheOutput(unittest.TestCase):
    def test_every_returned_segment_carries_chars(self):
        split = window("a b", [word("a", 0, 1, "S0"), word("b", 1, 2, "S1")])
        mono = window("c d", [word("c", 2, 3, "S0"), word("d", 3, 4, "S0")])
        out, _ = resegment_segments([split, mono])
        for seg in out:
            self.assertIn("chars", seg)
            self.assertEqual(seg["chars"], len(seg["text"]))

    def test_counters_reach_the_caller(self):
        split = window("a b", [word("a", 0, 1, "S0"), word("b", 1, 2, "S1")])
        mono = window("c d", [word("c", 2, 3, "S0"), word("d", 3, 4, "S0")])
        stats = {}
        resegment_result({"segments": [split, mono]}, stats)
        self.assertEqual(stats["in"], 2)
        self.assertEqual(stats["split"], 1)
        self.assertEqual(stats["single_speaker"], 1)
        self.assertEqual(stats["words_in"], 4)
        self.assertEqual(stats["words_out"], 4)

    def test_ambiguous_count_is_reported_not_swallowed(self):
        seg = window("Bonjour le monde",
                     [word("Bonjour", 0, 1, "S0"), word("monde", 1, 2, "S1")])
        stats = {}
        resegment_result({"segments": [seg]}, stats)
        self.assertEqual(stats["ambiguous"], 1)


if __name__ == "__main__":
    unittest.main()
