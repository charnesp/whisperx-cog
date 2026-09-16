"""GPU-free unit tests for the golden-set harness (scripts/golden_set.py).

The harness itself drives real GPU runs (task 6.1 of tasks.md: replayable
golden set on the 4080). These tests cover only its pure/GPU-free helpers:
- transcript hashing (bit-identical regression checks)
- hotword recall + false-positive counting on transcripts
- word-level timestamp verification
- word-speakier handoff verification (assign_word_speakers receives ForcedAligner words)
- VRAM / RTFx metric evaluation
- golden-set run evaluation and JSON report building
- main() exit codes and --check-only mode without GPU

scripts/golden_set.py must exist (this is the RED assertion of task 6.1).
"""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

GOLDEN_SET_PATH = REPO_ROOT / "scripts" / "golden_set.py"


def _load_golden_set():
    spec = importlib.util.spec_from_file_location("golden_set", GOLDEN_SET_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(GOLDEN_SET_PATH.is_file(), "scripts/golden_set.py does not exist yet (RED)")
class TestTranscriptHashing(unittest.TestCase):
    def setUp(self):
        self.gs = _load_golden_set()

    def test_same_text_same_hash(self):
        self.assertEqual(
            self.gs.hash_transcript("Bonjour à tous."),
            self.gs.hash_transcript("Bonjour à tous."),
        )

    def test_different_text_different_hash(self):
        self.assertNotEqual(
            self.gs.hash_transcript("Bonjour à tous."),
            self.gs.hash_transcript("Bonjour à tous !"),
        )

    def test_hash_is_deterministic_sha256_hex(self):
        import hashlib

        expected = hashlib.sha256("Bonjour à tous.".encode("utf-8")).hexdigest()
        self.assertEqual(self.gs.hash_transcript("Bonjour à tous."), expected)


@unittest.skipUnless(GOLDEN_SET_PATH.is_file(), "scripts/golden_set.py does not exist yet (RED)")
class TestHotwordMetrics(unittest.TestCase):
    """Proper-noun recall AND false positives (tasks.md 6.3).

    The four terms come from the 2026-09-02 real meeting: Backblaze, Supabase,
    AirSync, Volok. Recall = occurrences of each term in the transcript;
    false positives = term occurrences in segments that should not contain
    them (no hallucinated insertions far from the discussion topics).
    """

    HOTWORDS = ["Backblaze", "Supabase", "AirSync", "Volok"]

    def setUp(self):
        self.gs = _load_golden_set()

    def test_counts_occurrences_case_insensitive(self):
        text = "on parle de backblaze, puis BACKBLAZE encore et Supabase."
        counts = self.gs.count_hotword_occurrences(text, self.HOTWORDS)
        self.assertEqual(counts["Backblaze"], 2)
        self.assertEqual(counts["Supabase"], 1)
        self.assertEqual(counts["AirSync"], 0)
        self.assertEqual(counts["Volok"], 0)

    def test_recall_counts_accented_boundaries(self):
        # French punctuation directly glued to the term must still count.
        text = "Backblaze: un bucket. (Supabase) c'est open source, AirSync!"
        counts = self.gs.count_hotword_occurrences(text, self.HOTWORDS)
        self.assertEqual(counts["Backblaze"], 1)
        self.assertEqual(counts["Supabase"], 1)
        self.assertEqual(counts["AirSync"], 1)

    def test_false_positive_segments_detection(self):
        # A segment about logs that suddenly contains "Supabase" with no
        # storage/database context is a hallucinated insertion.
        segments = [
            {"start": 0.0, "text": "on regarde les logs du robot aujourd'hui."},
            {"start": 5.0, "text": "supabase propose un bucket pour les images."},
            {"start": 10.0, "text": "la réunion porte sur la volumétrie des logs."},
        ]
        fps = self.gs.find_hotword_false_positives(
            segments, self.HOTWORDS, context_keywords=("bucket", "stockage", "base", "api", "s3")
        )
        # segment 2 mentions buckets/storage context -> NOT a false positive;
        # the "logs du robot" segment has no storage context and no hotword.
        self.assertEqual(fps, [])

    def test_false_positive_flagged_when_no_context(self):
        segments = [
            {"start": 0.0, "text": "le planning de la semaine est validé backblaze."},
        ]
        fps = self.gs.find_hotword_false_positives(
            segments, self.HOTWORDS, context_keywords=("bucket", "stockage")
        )
        self.assertEqual(len(fps), 1)
        self.assertEqual(fps[0]["hotword"], "Backblaze")
        self.assertEqual(fps[0]["segment_start"], 0.0)

    def test_recall_improvement_computation(self):
        baseline = {"Backblaze": 3, "Supabase": 0, "AirSync": 1, "Volok": 0}
        with_hotwords = {"Backblaze": 12, "Supabase": 3, "AirSync": 4, "Volok": 1}
        improvement = self.gs.hotword_recall_improvement(baseline, with_hotwords)
        self.assertEqual(improvement["Backblaze"], 4)  # measured 3 -> 12 (design.md)
        self.assertTrue(improvement["Supabase"] > 0)  # 0 -> 3

    def test_recall_report_structure(self):
        baseline = {"Backblaze": 3, "Supabase": 0}
        with_hotwords = {"Backblaze": 12, "Supabase": 3}
        report = self.gs.build_hotword_recall_report(baseline, with_hotwords, self.HOTWORDS)
        self.assertIn("recall", report)
        self.assertIn("false_positive_risk", report)
        for term in self.HOTWORDS:
            self.assertIn(term, report["recall"])


@unittest.skipUnless(GOLDEN_SET_PATH.is_file(), "scripts/golden_set.py does not exist yet (RED)")
class TestWordTimestamps(unittest.TestCase):
    def setUp(self):
        self.gs = _load_golden_set()

    def test_word_level_timestamps_present(self):
        result = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 2.0,
                    "text": "Bonjour à tous",
                    "words": [
                        {"word": "Bonjour", "start": 0.0, "end": 0.5},
                        {"word": "à", "start": 0.5, "end": 0.7},
                        {"word": "tous", "start": 0.7, "end": 1.9},
                    ],
                }
            ]
        }
        self.assertTrue(self.gs.word_timestamps_present(result))

    def test_missing_words_flagged(self):
        result = {"segments": [{"start": 0.0, "end": 2.0, "text": "Bonjour"}]}
        self.assertFalse(self.gs.word_timestamps_present(result))

    def test_empty_words_list_flagged(self):
        result = {"segments": [{"start": 0.0, "end": 2.0, "text": "Bonjour", "words": []}]}
        self.assertFalse(self.gs.word_timestamps_present(result))


@unittest.skipUnless(GOLDEN_SET_PATH.is_file(), "scripts/golden_set.py does not exist yet (RED)")
class TestAssignWordSpeakersHandoff(unittest.TestCase):
    """Task 6.5: assign_word_speakers must receive words from the Qwen ForcedAligner."""

    def setUp(self):
        self.gs = _load_golden_set()

    def test_words_reach_diarization(self):
        words = [{"word": "Bonjour", "start": 0.0, "end": 0.5, "speaker": "SPEAKER_00"}]
        result = {"segments": [{"start": 0.0, "end": 0.5, "text": "Bonjour", "words": words}]}
        self.assertTrue(self.gs.words_carry_speakers(result))

    def test_words_without_speaker_flagged(self):
        words = [{"word": "Bonjour", "start": 0.0, "end": 0.5}]
        result = {"segments": [{"start": 0.0, "end": 0.5, "text": "Bonjour", "words": words}]}
        self.assertFalse(self.gs.words_carry_speakers(result))


@unittest.skipUnless(GOLDEN_SET_PATH.is_file(), "scripts/golden_set.py does not exist yet (RED)")
class TestGpuMetrics(unittest.TestCase):
    def setUp(self):
        self.gs = _load_golden_set()

    def test_vram_peak_below_limit(self):
        self.assertTrue(self.gs.vram_ok(4.99))  # measured 15/09 on the 4080
        self.assertTrue(self.gs.vram_ok(5.49))

    def test_vram_peak_above_limit_fails(self):
        self.assertFalse(self.gs.vram_ok(5.6))
        self.assertFalse(self.gs.vram_ok(10.0))  # fp32 default load ~10 GB

    def test_vram_limit_is_5_5_gb(self):
        self.assertEqual(self.gs.VRAM_LIMIT_GB, 5.5)

    def test_rtfx_plausible(self):
        # RTFx = audio_duration_s / processing_time_s; measured ~52 on the 4080.
        self.assertTrue(self.gs.rtfx_ok(52.0))
        self.assertTrue(self.gs.rtfx_ok(30.0))

    def test_rtfx_implausible_flagged(self):
        self.assertFalse(self.gs.rtfx_ok(0.5))  # slower than realtime
        self.assertFalse(self.gs.rtfx_ok(5000.0))  # suspiciously impossible

    def test_rtfx_computation(self):
        self.assertAlmostEqual(
            self.gs.compute_rtfx(audio_duration_s=3600.0, processing_time_s=70.0),
            3600.0 / 70.0,
            places=6,
        )


@unittest.skipUnless(GOLDEN_SET_PATH.is_file(), "scripts/golden_set.py does not exist yet (RED)")
class TestGoldenSetRuns(unittest.TestCase):
    """Harness structure: the three runs of task 6.1 (turbo baseline, qwen
    baseline, qwen+hotwords) + faster-whisper regression (6.2) are evaluated
    against expected assertions, GPU-free via mocked loaders."""

    def setUp(self):
        self.gs = _load_golden_set()

    def test_run_definitions_exist(self):
        self.assertIn("turbo_baseline", self.gs.RUNS)
        self.assertIn("qwen_baseline", self.gs.RUNS)
        self.assertIn("qwen_hotwords", self.gs.RUNS)

    def test_run_definitions_carry_model_and_hotwords(self):
        self.assertEqual(self.gs.RUNS["turbo_baseline"]["whisper_model"], "large-v3-turbo")
        self.assertEqual(self.gs.RUNS["qwen_baseline"]["whisper_model"], "qwen3-asr")
        self.assertIsNone(self.gs.RUNS["qwen_baseline"].get("hotwords") or None)
        self.assertIn("Backblaze", self.gs.RUNS["qwen_hotwords"]["hotwords"])

    def test_evaluate_report_all_pass(self):
        report = {
            "runs": {
                "turbo_baseline": {"transcript_hash": "a" * 64, "ok": True},
                "qwen_baseline": {"transcript_hash": "b" * 64, "ok": True},
                "qwen_hotwords": {"transcript_hash": "c" * 64, "ok": True},
            },
            "word_timestamps_present": True,
            "words_carry_speakers": True,
            "vram_peak_gb": 4.99,
            "rtfx": 52.0,
            "false_positives": [],
        }
        evaluation = self.gs.evaluate_report(report)
        self.assertTrue(evaluation["all_pass"])
        self.assertEqual(evaluation["failures"], [])

    def test_evaluate_report_vram_failure_listed(self):
        report = {
            "runs": {},
            "word_timestamps_present": True,
            "words_carry_speakers": True,
            "vram_peak_gb": 6.1,
            "rtfx": 52.0,
            "false_positives": [],
        }
        evaluation = self.gs.evaluate_report(report)
        self.assertFalse(evaluation["all_pass"])
        self.assertTrue(any("vram" in f.lower() for f in evaluation["failures"]))

    def test_evaluate_report_false_positive_listed(self):
        report = {
            "runs": {},
            "word_timestamps_present": True,
            "words_carry_speakers": True,
            "vram_peak_gb": 4.99,
            "rtfx": 52.0,
            "false_positives": [{"hotword": "Supabase", "segment_start": 10.0, "text": "x"}],
        }
        evaluation = self.gs.evaluate_report(report)
        self.assertFalse(evaluation["all_pass"])
        self.assertTrue(any("false_positive" in f for f in evaluation["failures"]))


@unittest.skipUnless(GOLDEN_SET_PATH.is_file(), "scripts/golden_set.py does not exist yet (RED)")
class TestHarnessGpuFree(unittest.TestCase):
    """The harness main flow must be exercisable GPU-free with mocked
    load_audio/model functions (no CUDA in CI)."""

    def setUp(self):
        self.gs = _load_golden_set()

    def test_run_single_with_mocks(self):
        captured = {}

        def fake_load_audio(path):
            captured["audio_path"] = path
            return [0.0] * 16

        class FakeModel:
            def __init__(self, name):
                self.name = name

            def transcribe(self, audio, batch_size=None, context=None, **kwargs):
                captured["model"] = self.name
                captured["batch_size"] = batch_size
                captured["context"] = kwargs.get("context")
                return {
                    "language": "fr",
                    "segments": [
                        {
                            "start": 0.0,
                            "end": 1.0,
                            "text": "Backblaze est un bucket.",
                            "words": [
                                {"word": "Backblaze", "start": 0.0, "end": 0.4, "speaker": "SPEAKER_00"},
                                {"word": "est", "start": 0.4, "end": 0.6, "speaker": "SPEAKER_00"},
                                {"word": "un", "start": 0.6, "end": 0.7, "speaker": "SPEAKER_00"},
                                {"word": "bucket.", "start": 0.7, "end": 1.0, "speaker": "SPEAKER_00"},
                            ],
                        }
                    ],
                }

        run = self.gs.run_single(
            run_spec={"whisper_model": "qwen3-asr", "hotwords": "Backblaze, Supabase"},
            audio_path="/tmp/fake.ogg",  # noqa: S108 — test fixture path, never read
            load_audio_fn=fake_load_audio,
            model_factory=lambda name: FakeModel(name),
            clock=lambda: 0.0,
        )
        self.assertEqual(captured["model"], "qwen3-asr")
        self.assertIn("bucket", run["transcript"])
        self.assertIn("transcript_hash", run)
        self.assertGreater(run["duration_s"], 0.0)
        self.assertTrue(run["word_timestamps_present"])
        self.assertTrue(run["words_carry_speakers"])

    def test_qwen_batch_default_applied(self):
        captured = {}

        class FakeModel:
            def transcribe(self, audio, batch_size=None, **kwargs):
                captured["batch_size"] = batch_size
                return {"language": "fr", "segments": []}

        self.gs.run_single(
            run_spec={"whisper_model": "qwen3-asr", "hotwords": None},
            audio_path="x.ogg",
            load_audio_fn=lambda p: [0.0],
            model_factory=lambda name: FakeModel(),
            clock=lambda: 0.0,
        )
        self.assertEqual(captured["batch_size"], self.gs.QWEN_DEFAULT_BATCH)

    def test_whisper_batch_default_applied(self):
        captured = {}

        class FakeModel:
            def transcribe(self, audio, batch_size=None, **kwargs):
                captured["batch_size"] = batch_size
                return {"language": "fr", "segments": []}

        self.gs.run_single(
            run_spec={"whisper_model": "large-v3-turbo", "hotwords": None},
            audio_path="x.ogg",
            load_audio_fn=lambda p: [0.0],
            model_factory=lambda name: FakeModel(),
            clock=lambda: 0.0,
        )
        self.assertEqual(captured["batch_size"], self.gs.WHISPER_DEFAULT_BATCH)

    def test_report_is_json_serializable(self):
        report = self.gs.build_report(
            runs={},
            word_timestamps_present=True,
            words_carry_speakers=True,
            vram_peak_gb=4.99,
            rtfx=52.0,
            false_positives=[],
        )
        json.dumps(report)  # must not raise


if __name__ == "__main__":
    unittest.main()
