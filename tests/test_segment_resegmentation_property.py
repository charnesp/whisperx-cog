"""Property tests for segment_resegmentation: the guarantees must hold on any
random window, not only on the hand-written cases.

Seeded, so a failure is reproducible. No GPU, no audio, plain dicts.
"""

import random
import re
import unittest

from segment_resegmentation import resegment_segments

WORDS = ["bonjour", "le", "monde", "aujourd'hui", "oui", "non", "voila", "donc",
         "euh", "alors", "vraiment", "c'est", "d'accord", "voiture", "ligne"]
PUNCT = ["", ",", ".", "?", "!", ":", ";"]


def strip_ws(text):
    return re.sub(r"\s+", "", text)


def make_case(rng):
    speakers = [f"S{i}" for i in range(rng.randint(1, 4))]
    n = rng.randint(1, 25)
    words, tokens = [], []
    t = 0.0
    for _ in range(n):
        token = rng.choice(WORDS)
        punct = rng.choice(PUNCT)
        words.append({"word": token, "start": t, "end": t + 0.3,
                      "speaker": rng.choice(speakers)})
        tokens.append(token + punct)
        t += 0.4
    text = " ".join(tokens)
    if rng.random() < 0.3:
        text += rng.choice([".", "?", "!"])
    if rng.random() < 0.2:
        text = rng.choice(["?", "...", "- "]) + text
    return {"start": 0.0, "end": t, "text": text,
            "speaker": words[0]["speaker"], "words": words}


class PropertyHolds(unittest.TestCase):
    def test_no_character_and_no_word_is_lost_on_random_windows(self):
        rng = random.Random(20261031)
        for case_no in range(300):
            seg = make_case(rng)
            try:
                out, stats = resegment_segments([seg])
            except Exception as exc:  # noqa: BLE001 - a property run must report the case
                self.fail(f"cas {case_no}: exception {type(exc).__name__}: {exc}")

            with self.subTest(case=case_no):
                # A word is never lost nor duplicated.
                self.assertEqual(
                    sum(len(s["words"]) for s in out), len(seg["words"]),
                    "le nombre de mots doit etre conserve",
                )
                # Either the window was split and every character survived, or it
                # was left untouched: nothing in between.
                if stats["ambiguous"] or stats["no_words"] or stats["single_speaker"]:
                    self.assertEqual(len(out), 1, "un bloc non recoupe reste un bloc")
                    self.assertEqual(out[0]["text"], seg["text"],
                                     "un bloc non recoupe garde son texte")
                else:
                    self.assertEqual(
                        strip_ws("".join(s["text"] for s in out)), strip_ws(seg["text"]),
                        "aucun caractere ne doit disparaitre au recoupage",
                    )
                    for turn in out:
                        self.assertTrue(turn["text"].strip(), "aucun tour vide")
                        self.assertLessEqual(turn["start"], turn["end"])

    def test_resegmenting_a_random_window_twice_is_stable(self):
        rng = random.Random(7)
        for case_no in range(100):
            seg = make_case(rng)
            once, _ = resegment_segments([seg])
            twice, _ = resegment_segments(once)
            with self.subTest(case=case_no):
                self.assertEqual([s["text"] for s in once], [s["text"] for s in twice])
                self.assertEqual([s["speaker"] for s in once], [s["speaker"] for s in twice])


if __name__ == "__main__":
    unittest.main()
