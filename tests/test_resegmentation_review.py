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
        """The cut comes from the audio window, not from the conversation: a turn
        that continues in the next window is one turn."""
        first = window("a b", [word("a", 0.0, 1.0, "S1"), word("b", 1.0, 2.0, "S0")])
        second = window("c d", [word("c", 2.0 + MERGE_GAP_S / 2, 3.0, "S0"),
                                word("d", 3.0, 4.0, "S1")])
        out, _ = resegment_segments([first, second])
        self.assertEqual([s["text"] for s in out], ["a", "b c", "d"])
        self.assertEqual([s["speaker"] for s in out], ["S1", "S0", "S1"])

    def test_gap_above_threshold_prevents_the_merge(self):
        first = window("a b", [word("a", 0.0, 1.0, "S1"), word("b", 1.0, 2.0, "S0")])
        second = window("c d", [word("c", 2.0 + MERGE_GAP_S * 3, 3.0, "S0"),
                                word("d", 3.0, 4.0, "S1")])
        out, _ = resegment_segments([first, second])
        self.assertEqual(len(out), 4)


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

class MergeOnlyBetweenSplitPieces(unittest.TestCase):
    """The 0.8 s merge was also gluing untouched windows, whose label is the
    dominant speaker of the whole window: it widened a wrong attribution."""

    def test_two_short_monologue_windows_are_not_merged(self):
        first = window("un deux", [word("un", 0.0, 0.5, "S0"), word("deux", 0.5, 1.0, "S0")])
        second = window("trois", [word("trois", 1.1, 1.5, "S0")])
        out, _ = resegment_segments([first, second])
        self.assertEqual(len(out), 2, "untouched windows must come back unchanged")

    def test_an_ambiguous_block_is_not_glued_to_the_next_turn(self):
        # Three speakers, no word carries the token "b" -> alignment impossible.
        ambiguous = window("a b c", [word("a", 0.0, 1.0, "S0"), word("x", 1.0, 2.0, "S1"),
                                     word("c", 2.0, 3.0, "S2")])
        follow = window("alors", [word("alors", 3.0, 4.0, "S0")])
        out, stats = resegment_segments([ambiguous, follow])
        self.assertEqual(stats["ambiguous"], 1)
        self.assertEqual(out[0]["speaker"], "S0")
        self.assertEqual(out[0]["text"], "a b c", "the untouched block keeps its own text")
        self.assertEqual(len(out), 2, "and must not be widened by the next turn")

    def test_two_segments_without_any_speaker_stay_two(self):
        first = {"start": 0.0, "end": 1.0, "text": "un", "words": [{"word": "un", "start": 0.0, "end": 1.0}]}
        second = {"start": 1.1, "end": 2.0, "text": "deux", "words": [{"word": "deux", "start": 1.1, "end": 2.0}]}
        out, stats = resegment_segments([first, second])
        self.assertEqual(len(out), 2)
        self.assertEqual(stats["no_words"], 2)


class BoundsReachTheBridge(unittest.TestCase):
    """The bridge compares and converts segment bounds: a None there is a 500."""

    def test_word_without_bounds_falls_back_to_the_window(self):
        seg = {"start": 0.0, "end": 5.0, "text": "a b", "speaker": "S0",
               "words": [{"word": "a", "start": None, "end": None, "speaker": "S0"},
                         {"word": "b", "start": None, "end": None, "speaker": "S1"}]}
        out, stats = resegment_segments([seg])
        for piece in out:
            self.assertIsInstance(piece["start"], float)
            self.assertIsInstance(piece["end"], float)

    def test_no_bounds_anywhere_keeps_the_block_whole(self):
        seg = {"start": None, "end": None, "text": "a b", "speaker": "S0",
               "words": [{"word": "a", "speaker": "S0"}, {"word": "b", "speaker": "S1"}]}
        out, stats = resegment_segments([seg])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["text"], "a b")
        self.assertEqual(stats["ambiguous"], 1)
