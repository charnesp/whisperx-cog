"""Tests for segment_resegmentation (pure, no torch, no GPU).

Run: python3 -m unittest tests.test_segment_resegmentation -v
"""

import unittest

from segment_resegmentation import resegment_result, resegment_segments


def word(text, start, end, speaker=None):
    return {"word": text, "start": start, "end": end, "speaker": speaker}


def segment(text, words, speaker="SPEAKER_00", start=None, end=None):
    return {
        "start": words[0]["start"] if start is None else start,
        "end": words[-1]["end"] if end is None else end,
        "text": text,
        "speaker": speaker,
        "words": words,
    }


def strip_ws(text):
    return "".join(text.split())


class SplitAtSpeakerChange(unittest.TestCase):
    def test_two_speakers_in_one_window_become_two_turns(self):
        seg = segment(
            "Bonjour, comment allez-vous ? Tres bien.",
            [word("Bonjour", 0.0, 0.4, "SPEAKER_00"),
             word("comment", 0.4, 0.8, "SPEAKER_00"),
             word("allez-vous", 0.8, 1.2, "SPEAKER_00"),
             word("Tres", 1.5, 1.7, "SPEAKER_01"),
             word("bien", 1.7, 2.0, "SPEAKER_01")],
        )
        out, stats = resegment_segments([seg])
        self.assertEqual([s["speaker"] for s in out], ["SPEAKER_00", "SPEAKER_01"])
        self.assertEqual(out[0]["text"], "Bonjour, comment allez-vous ?")
        self.assertEqual(out[1]["text"], "Tres bien.")
        self.assertEqual((out[0]["start"], out[1]["end"]), (0.0, 2.0))
        self.assertEqual(stats["split"], 1)

    def test_three_speakers_in_one_window_become_three_turns(self):
        seg = segment("a b c", [word("a", 0.0, 0.2, "S0"),
                                word("b", 0.2, 0.4, "S1"),
                                word("c", 0.4, 0.6, "S0")], speaker="S0")
        out, _ = resegment_segments([seg])
        self.assertEqual([s["speaker"] for s in out], ["S0", "S1", "S0"])

    def test_single_speaker_window_is_untouched(self):
        seg = segment("a b", [word("a", 0.0, 0.2, "S0"), word("b", 0.2, 0.4, "S0")])
        out, stats = resegment_segments([seg])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["text"], "a b")
        self.assertEqual(stats["single_speaker"], 1)

    def test_window_without_words_is_untouched(self):
        seg = {"start": 0.0, "end": 3.0, "text": "texte", "speaker": "S0"}
        out, stats = resegment_segments([seg])
        self.assertEqual(out, [seg])
        self.assertEqual(stats["no_words"], 1)


class PunctuationIsPreserved(unittest.TestCase):
    def test_detached_final_question_mark_is_kept(self):
        seg = segment("Bonjour tout le monde ?",
                      [word("Bonjour", 0, 1, "S0"), word("tout", 1, 2, "S0"),
                       word("le", 2, 3, "S0"), word("monde", 3, 4, "S1")])
        out, stats = resegment_segments([seg])
        self.assertEqual(stats["split"], 1)
        self.assertTrue(out[-1]["text"].endswith("?"), out[-1]["text"])

    def test_detached_leading_punctuation_is_kept(self):
        seg = segment("? Bonjour tout le monde",
                      [word("Bonjour", 1, 2, "S0"), word("tout", 2, 3, "S0"),
                       word("le", 3, 4, "S1"), word("monde", 4, 5, "S1")])
        out, stats = resegment_segments([seg])
        self.assertEqual(stats["split"], 1)
        self.assertTrue(out[0]["text"].startswith("?"), out[0]["text"])

    def test_all_characters_survive_the_split(self):
        seg = segment("Un, deux : trois ! Quatre ?",
                      [word("Un", 0, 1, "S0"), word("deux", 1, 2, "S0"),
                       word("trois", 2, 3, "S1"), word("Quatre", 3, 4, "S0")])
        out, _ = resegment_segments([seg])
        self.assertEqual(strip_ws("".join(s["text"] for s in out)),
                         strip_ws(seg["text"]))


class NoFabrication(unittest.TestCase):
    def test_word_text_mismatch_leaves_segment_unchanged(self):
        seg = segment("Bonjour le monde",
                      [word("Bonjour", 0, 1, "S0"), word("monde", 1, 2, "S1")])
        out, stats = resegment_segments([seg])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["text"], seg["text"])
        self.assertEqual(stats["ambiguous"], 1)

    def test_missing_text_leaves_words_attached_to_one_segment(self):
        seg = {"start": 0.0, "end": 1.0, "text": "", "speaker": "S0",
               "words": [word("a", 0.0, 0.5, "S0"), word("b", 0.5, 1.0, "S1")]}
        out, _ = resegment_segments([seg])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["text"], "")

    def test_word_count_is_preserved(self):
        seg = segment("a b c", [word("a", 0, 1, "S0"), word("b", 1, 2, "S1"),
                                word("c", 2, 3, "S0")])
        out, stats = resegment_segments([seg])
        self.assertEqual(stats["words_in"], stats["words_out"])
        self.assertEqual(sum(len(s["words"]) for s in out), 3)

    def test_malformed_segment_raises(self):
        with self.assertRaises(TypeError):
            resegment_segments([{"start": 0, "end": 1, "text": "x", "words": ["nope"]}])


class SpeakerLabels(unittest.TestCase):
    def test_unlabelled_word_inherits_previous_speaker(self):
        seg = segment("a b c",
                      [word("a", 0, 1, "S0"), word("b", 1, 2, None),
                       word("c", 2, 3, "S1")])
        out, _ = resegment_segments([seg])
        self.assertEqual([s["speaker"] for s in out], ["S0", "S1"])
        self.assertEqual([w["speaker"] for w in out[0]["words"]], ["S0", "S0"])

    def test_untouched_segment_still_carries_speaker_on_every_word(self):
        seg = segment("a b", [word("a", 0, 1, "S0"), word("b", 1, 2, None)])
        out, _ = resegment_segments([seg])
        self.assertEqual([w["speaker"] for w in out[0]["words"]], ["S0", "S0"])


class MergeAndIdempotence(unittest.TestCase):
    def test_adjacent_turns_of_same_speaker_are_merged_across_windows(self):
        first = {"start": 0.0, "end": 1.0, "text": "un", "speaker": "S0",
                 "words": [word("un", 0.0, 1.0, "S0")]}
        second = {"start": 1.2, "end": 2.0, "text": "deux", "speaker": "S0",
                  "words": [word("deux", 1.2, 2.0, "S0")]}
        out, _ = resegment_segments([first, second])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["text"], "un deux")

    def test_long_gap_prevents_merge(self):
        first = {"start": 0.0, "end": 1.0, "text": "un", "speaker": "S0",
                 "words": [word("un", 0.0, 1.0, "S0")]}
        second = {"start": 9.0, "end": 10.0, "text": "deux", "speaker": "S0",
                  "words": [word("deux", 9.0, 10.0, "S0")]}
        out, _ = resegment_segments([first, second])
        self.assertEqual(len(out), 2)

    def test_resegmenting_twice_changes_nothing(self):
        seg = segment("a b c", [word("a", 0, 1, "S0"), word("b", 1, 2, "S1"),
                                word("c", 2, 3, "S0")])
        once, _ = resegment_segments([seg])
        twice, _ = resegment_segments(once)
        self.assertEqual([s["text"] for s in once], [s["text"] for s in twice])
        self.assertEqual([s["speaker"] for s in once], [s["speaker"] for s in twice])


class ResultLevel(unittest.TestCase):
    def test_result_without_segments_is_returned_unchanged(self):
        payload = {"detected_language": "fr", "speaker_embeddings": {"S0": [0.1]}}
        self.assertIs(resegment_result(payload), payload)

    def test_result_is_not_mutated(self):
        seg = segment("a b", [word("a", 0, 1, "S0"), word("b", 1, 2, "S1")])
        payload = {"segments": [seg], "detected_language": "fr"}
        out = resegment_result(payload)
        self.assertEqual(len(payload["segments"]), 1)
        self.assertEqual(len(out["segments"]), 2)
        self.assertEqual(out["detected_language"], "fr")

    def test_result_without_any_speaker_is_unchanged(self):
        seg = segment("a b", [word("a", 0, 1, None), word("b", 1, 2, None)], speaker=None)
        payload = {"segments": [seg]}
        out = resegment_result(payload)
        self.assertEqual(len(out["segments"]), 1)


if __name__ == "__main__":
    unittest.main()
