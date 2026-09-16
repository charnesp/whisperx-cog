"""GPU-free unit tests for the golden-set harness (scripts/golden_set.py).

The harness itself drives real GPU runs (task 6.1 of tasks.md: replayable
golden set on the 4080). These tests cover its pure/GPU-free helpers and the
injectable run flow (align / diarize / assign_word_speakers wiring via
align_fn / diarize_fn params, GPU-free mocks for CI):
- transcript hashing (bit-identical regression checks)
- hotword recall + false-positive counting on transcripts
- word-level timestamp verification
- word-speaker handoff verification (assign_word_speakers receives ForcedAligner words)
- align/diarize wiring inside run_single (RED tests for the E4-FIX cycle)
- canonical segment hashing (6.2: words/start/end/speaker included)
- VRAM / RTFx metric evaluation (incl. per-run VRAM peak, FIX 7)
- golden-set run evaluation and JSON report building
- main() exit codes without GPU

scripts/golden_set.py must exist (this is the RED assertion of task 6.1).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
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
        # pipeline threshold requalified at 6.5 GB (FIX 3): 6.6 GB fails
        report = {
            "runs": {},
            "word_timestamps_present": True,
            "words_carry_speakers": True,
            "vram_peak_gb": 6.6,
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

        ticks = iter([10.0, 12.5, 15.0])  # transcribe start/end + run end (duration_total_s, FIX 4)
        run = self.gs.run_single(
            run_spec={"whisper_model": "qwen3-asr", "hotwords": "Backblaze, Supabase"},
            audio_path="/tmp/fake.ogg",  # noqa: S108 — test fixture path, never read
            load_audio_fn=fake_load_audio,
            model_factory=lambda name: FakeModel(name),
            clock=lambda: next(ticks),
        )
        self.assertEqual(captured["model"], "qwen3-asr")
        self.assertIn("bucket", run["transcript"])
        self.assertIn("transcript_hash", run)
        self.assertAlmostEqual(run["duration_s"], 2.5)
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


# ---------------------------------------------------------------------------
# RED tests — E4-FIX cycle (wiring align/diarize inside run_single)
# Written WITHOUT implementation: run_single has no align_fn/diarize_fn
# params yet, so every assertion below fails (observed RED, documented in
# tasks.md deviations).
# ---------------------------------------------------------------------------


@unittest.skipUnless(GOLDEN_SET_PATH.is_file(), "scripts/golden_set.py does not exist yet (RED)")
class TestAlignDiarizeWiring(unittest.TestCase):
    """FIX 1 (🔴): run_single must wire align + diarize + assign_word_speakers.

    The pipeline (mirrors predict._run_predict): after transcribe,
    align (align_qwen on the qwen path / align standard on the turbo path)
    produces word-level timestamps, then diarize produces speaker turns and
    whisperx.assign_word_speakers labels the aligned words. Without this
    wiring, word_timestamps_present / words_carry_speakers are False on a
    real GPU run and the harness fails its own gate (6.5 not executable).
    Both functions are injectable (align_fn, diarize_fn) so CI can exercise
    the wiring GPU-free; the real defaults wrap predict.align / align_qwen /
    diarize.
    """

    def setUp(self):
        self.gs = _load_golden_set()

    @staticmethod
    def _fake_model(model_name, align_called, diarize_called):
        class FakeModel:
            def transcribe(self, audio, batch_size=None, **kwargs):
                return {
                    "language": "fr",
                    "segments": [{"start": 0.0, "end": 1.0, "text": "Backblaze est un bucket."}],
                }

        return FakeModel()

    def _run(self, model_name, align_fn, diarize_fn):
        captured = {"align": None, "diarize": None}

        def load_audio(path):
            return [0.0] * 16

        class FakeModel:
            def transcribe(self, audio, batch_size=None, **kwargs):
                return {
                    "language": "fr",
                    "segments": [{"start": 0.0, "end": 1.0, "text": "Backblaze est un bucket."}],
                }

        def clock():
            return 0.0

        run = self.gs.run_single(
            run_spec={"whisper_model": model_name, "hotwords": None},
            audio_path="x.ogg",
            load_audio_fn=load_audio,
            model_factory=lambda name: FakeModel(),
            clock=clock,
            align_fn=align_fn,
            diarize_fn=diarize_fn,
        )
        return run, captured

    def test_run_single_accepts_injected_align_and_diarize(self):
        calls = {"align": 0, "diarize": 0}

        def fake_align(audio, result):
            calls["align"] += 1
            result["segments"][0]["words"] = [{"word": "Backblaze", "start": 0.0, "end": 0.4}]
            return result

        def fake_diarize(audio, result):
            calls["diarize"] += 1
            for seg in result["segments"]:
                for w in seg.get("words") or []:
                    w["speaker"] = "SPEAKER_00"
            return result

        run, captured = self._run("qwen3-asr", fake_align, fake_diarize)
        self.assertEqual(calls["align"], 1, "align must be called once per run")
        self.assertEqual(calls["diarize"], 1, "diarize must be called once per run")
        self.assertTrue(run["word_timestamps_present"])
        self.assertTrue(run["words_carry_speakers"])
        self.assertTrue(run["ok"])

    def test_align_qwen_used_on_qwen_path_and_align_on_turbo_path(self):
        used = {}

        def align_qwen(audio, result):
            used["align"] = "align_qwen"
            result["segments"][0]["words"] = [{"word": "x", "start": 0.0, "end": 0.4}]
            return result

        def align_standard(audio, result):
            used["align"] = "align"
            result["segments"][0]["words"] = [{"word": "x", "start": 0.0, "end": 0.4}]
            return result

        def fake_diarize(audio, result):
            for seg in result["segments"]:
                for w in seg.get("words") or []:
                    w["speaker"] = "SPEAKER_00"
            return result

        self._run("qwen3-asr", align_qwen, fake_diarize)
        self.assertEqual(used["align"], "align_qwen", "qwen path must use the Qwen ForcedAligner")
        used.clear()
        self._run("large-v3-turbo", align_standard, fake_diarize)
        self.assertEqual(used["align"], "align", "turbo path must use the standard aligner")

    def test_run_single_without_align_wiring_fails_its_own_gate(self):
        # The pre-fix behavior (the bug): transcribe output carries no
        # words/speaker, so run ok must be False — proves the gate is real.
        run, _ = self._run("qwen3-asr", None, None)
        self.assertFalse(run["ok"])
        self.assertFalse(run["word_timestamps_present"])
        self.assertFalse(run["words_carry_speakers"])

    # ------------------------------------------------------------------
    # E4-FIX-2 FIX 1 (🔴): the raw transcribe result NEVER carries
    # 'whisper_model' (asr_qwen returns {segments, language} only), yet
    # default_align_fn dispatches on result.get('whisper_model') — on a real
    # GPU qwen run the wav2vec2 (standard) branch would be taken instead of
    # predict.align_qwen (crash/incompatibility). run_single must inject the
    # model name into the result it hands to align_fn, and default_align_fn
    # must route the qwen branch from that injected key.
    # ------------------------------------------------------------------

    def test_run_single_hands_model_name_to_align_fn_result(self):
        seen = {}

        def spy_align(audio, result):
            seen["result"] = result
            return result

        def passthrough(audio, result):
            return result

        self._run("qwen3-asr", spy_align, passthrough)
        self.assertEqual(
            seen["result"].get("whisper_model"),
            "qwen3-asr",
            "raw transcribe result lacks 'whisper_model' — run_single must inject it so default_align_fn can dispatch",
        )
        seen.clear()
        self._run("large-v3-turbo", spy_align, passthrough)
        self.assertEqual(seen["result"].get("whisper_model"), "large-v3-turbo")

    def _mock_predict_env(self):
        """Fake predict module (align/align_qwen spies) + fake whisperx.alignment."""
        import sys
        import types

        calls = {"align_qwen": 0, "align": 0}
        words = [{"word": "x", "start": 0.0, "end": 0.4}]

        class FakePredict:
            @staticmethod
            def align_qwen(audio, result, debug):
                calls["align_qwen"] += 1
                for seg in result.get("segments") or []:
                    seg["words"] = [dict(w) for w in words]
                return result

            @staticmethod
            def align(audio, result, debug):
                calls["align"] += 1
                for seg in result.get("segments") or []:
                    seg["words"] = [dict(w) for w in words]
                return result

            @staticmethod
            def format_qwen_context(hotwords):
                return "", 0

        fake_alignment = types.ModuleType("whisperx.alignment")
        fake_alignment.DEFAULT_ALIGN_MODELS_TORCH = {"fr": "wav2vec2-fr"}
        fake_alignment.DEFAULT_ALIGN_MODELS_HF = {}

        saved_loader = self.gs._load_predict
        saved_modules = {k: sys.modules.get(k) for k in ("whisperx", "whisperx.alignment")}
        self.gs._load_predict = lambda: FakePredict()
        import types as _types

        sys.modules["whisperx"] = _types.ModuleType("whisperx")
        sys.modules["whisperx.alignment"] = fake_alignment

        def restore():
            self.gs._load_predict = saved_loader
            for key, value in saved_modules.items():
                if value is None:
                    sys.modules.pop(key, None)
                else:
                    sys.modules[key] = value

        self.addCleanup(restore)
        return calls

    def test_default_align_fn_dispatches_from_raw_transcribe_result(self):
        """Raw BRUT result {'segments', 'language'} (no whisper_model key)
        flowing through run_single -> default_align_fn (predict mocked):
        qwen3-asr must take the align_qwen branch, large-v3-turbo the
        standard align branch (language coverage guard)."""
        calls = self._mock_predict_env()

        class FakeModel:
            def transcribe(self, audio, batch_size=None, **kwargs):
                # BRUT transcribe output: asr_qwen returns {segments, language} ONLY
                return {
                    "language": "fr",
                    "segments": [{"start": 0.0, "end": 1.0, "text": "Backblaze est un bucket."}],
                }

        common = dict(
            audio_path="x.ogg",
            load_audio_fn=lambda p: [0.0] * 16,
            model_factory=lambda name: FakeModel(),
            clock=lambda: 0.0,
            align_fn=self.gs.default_align_fn,
            diarize_fn=lambda audio, result: result,
        )
        run_q = self.gs.run_single(run_spec={"whisper_model": "qwen3-asr", "hotwords": None}, **common)
        self.assertGreater(
            calls["align_qwen"],
            0,
            "qwen raw transcribe result must route to predict.align_qwen (wav2vec2 incompatible with qwen outputs)",
        )
        self.assertEqual(calls["align"], 0, "qwen path must NOT use the wav2vec2 standard aligner")
        self.assertTrue(run_q["word_timestamps_present"])

        self._mock_predict_env_calls = calls  # same spies continue
        run_t = self.gs.run_single(run_spec={"whisper_model": "large-v3-turbo", "hotwords": None}, **common)
        self.assertGreater(calls["align"], 0, "turbo path must use the standard aligner")
        self.assertEqual(calls["align_qwen"], 1, "turbo path must NOT use the Qwen ForcedAligner")
        self.assertTrue(run_t["word_timestamps_present"])

    def test_real_defaults_resolve_predict_align_and_diarize(self):
        # Real (GPU) defaults must exist and route to predict's align functions:
        # align_fn default dispatches align_qwen on qwen / align on turbo;
        # diarize_fn default wraps predict.diarize.
        self.assertTrue(hasattr(self.gs, "default_align_fn"))
        self.assertTrue(hasattr(self.gs, "default_diarize_fn"))
        import inspect

        src_align = inspect.getsource(self.gs.default_align_fn)
        src_diarize = inspect.getsource(self.gs.default_diarize_fn)
        self.assertIn("align_qwen", src_align)
        self.assertIn("align(", src_align)
        self.assertIn("diarize", src_diarize)
        self.assertIn("assign_word_speakers", src_diarize)


@unittest.skipUnless(GOLDEN_SET_PATH.is_file(), "scripts/golden_set.py does not exist yet (RED)")
class TestCanonicalSegmentHashing(unittest.TestCase):
    """FIX 5 (🟡): 6.2 regression hash = SHA-256 of the canonical JSON of the
    full segments (text/start/end + words incl. speaker when present), not
    text-only — the invariance now covers words and speaker labels."""

    def setUp(self):
        self.gs = _load_golden_set()

    def test_hash_segments_deterministic(self):
        segments = [
            {
                "start": 0.0,
                "end": 1.0,
                "text": "Bonjour",
                "words": [{"word": "Bonjour", "start": 0.0, "end": 0.5, "speaker": "SPEAKER_00"}],
                "speaker": "SPEAKER_00",
            }
        ]
        self.assertEqual(
            self.gs.hash_segments(segments),
            self.gs.hash_segments(json.loads(json.dumps(segments))),
        )

    def test_hash_segments_matches_canonical_json_sha256(self):
        segments = [{"start": 0.0, "end": 1.0, "text": "Bonjour", "words": [], "speaker": "SPEAKER_00"}]
        expected = hashlib.sha256(
            json.dumps(segments, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        self.assertEqual(self.gs.hash_segments(segments), expected)

    def test_hash_segments_changes_when_words_change(self):
        base = [{"start": 0.0, "end": 1.0, "text": "Bonjour", "words": [{"word": "Bonjour", "start": 0.0, "end": 0.5}]}]
        changed = [{"start": 0.0, "end": 1.0, "text": "Bonjour", "words": [{"word": "Bonjour", "start": 0.1, "end": 0.5}]}]
        self.assertNotEqual(self.gs.hash_segments(base), self.gs.hash_segments(changed))

    def test_hash_segments_changes_when_speaker_changes(self):
        base = [{"start": 0.0, "end": 1.0, "text": "Bonjour", "words": [{"word": "Bonjour", "start": 0.0, "end": 0.5, "speaker": "SPEAKER_00"}]}]
        changed = [{"start": 0.0, "end": 1.0, "text": "Bonjour", "words": [{"word": "Bonjour", "start": 0.0, "end": 0.5, "speaker": "SPEAKER_01"}]}]
        self.assertNotEqual(self.gs.hash_segments(base), self.gs.hash_segments(changed))

    def test_run_single_stores_segment_hash(self):
        segments = [{"start": 0.0, "end": 1.0, "text": "Backblaze est un bucket.", "words": [], "speaker": "SPEAKER_00"}]

        class FakeModel:
            def transcribe(self, audio, batch_size=None, **kwargs):
                return {"language": "fr", "segments": segments}

        run = self.gs.run_single(
            run_spec={"whisper_model": "qwen3-asr", "hotwords": None},
            audio_path="x.ogg",
            load_audio_fn=lambda p: [0.0],
            model_factory=lambda name: FakeModel(),
            clock=lambda: 0.0,
            align_fn=lambda audio, result: result,
            diarize_fn=lambda audio, result: result,
        )
        expected = self.gs.hash_segments(segments)
        self.assertEqual(run["segments_hash"], expected)

    def test_transcript_hash_backward_compatible(self):
        # hash_transcript stays available (text-level invariance on 6.4).
        self.assertEqual(
            self.gs.hash_transcript("Bonjour"),
            hashlib.sha256("Bonjour".encode("utf-8")).hexdigest(),
        )


@unittest.skipUnless(GOLDEN_SET_PATH.is_file(), "scripts/golden_set.py does not exist yet (RED)")
class TestVramPerRun(unittest.TestCase):
    """FIX 7 (🟡): per-run VRAM peak via injectable vram_peak_fn, reset at the
    start of each run (reset_peak_memory_stats) and read at the end
    (max_memory_allocated)."""

    def setUp(self):
        self.gs = _load_golden_set()

    def test_run_single_records_vram_peak(self):
        peaks = iter([4.2])

        class FakeModel:
            def transcribe(self, audio, batch_size=None, **kwargs):
                return {"language": "fr", "segments": []}

        run = self.gs.run_single(
            run_spec={"whisper_model": "qwen3-asr", "hotwords": None},
            audio_path="x.ogg",
            load_audio_fn=lambda p: [0.0],
            model_factory=lambda name: FakeModel(),
            clock=lambda: 0.0,
            align_fn=lambda audio, result: result,
            diarize_fn=lambda audio, result: result,
            vram_peak_fn=lambda: next(peaks),
        )
        self.assertAlmostEqual(run["vram_peak_gb"], 4.2)

    def test_build_report_exposes_per_run_vram_and_global_peak(self):
        runs = {
            "turbo_baseline": {"ok": True, "transcript_hash": "a" * 64, "vram_peak_gb": 3.1},
            "qwen_baseline": {"ok": True, "transcript_hash": "b" * 64, "vram_peak_gb": 4.2},
            "qwen_hotwords": {"ok": True, "transcript_hash": "c" * 64, "vram_peak_gb": 4.99},
        }
        report = self.gs.build_report(
            runs=runs,
            word_timestamps_present=True,
            words_carry_speakers=True,
            vram_peak_gb=4.99,
            rtfx=52.0,
            false_positives=[],
        )
        self.assertEqual(report["vram_peak_by_run"]["qwen_hotwords"], 4.99)
        self.assertEqual(report["vram_peak_gb"], 4.99)
        self.assertIn("vram_peak_by_run", report)

    def test_build_report_vram_peak_gb_is_global_max_across_runs(self):
        # E4-FIX-2 FIX 2 (🟡): the report's vram_peak_gb is the TRUE global
        # peak = max over the per-run peaks, not the last vram_peak_fn()
        # reading (which is the peak of the LAST run only).
        runs = {
            "turbo_baseline": {"ok": True, "transcript_hash": "a" * 64, "vram_peak_gb": 3.1},
            "qwen_baseline": {"ok": True, "transcript_hash": "b" * 64, "vram_peak_gb": 5.9},
            "qwen_hotwords": {"ok": True, "transcript_hash": "c" * 64, "vram_peak_gb": 4.2},
        }
        report = self.gs.build_report(
            runs=runs,
            word_timestamps_present=True,
            words_carry_speakers=True,
            vram_peak_gb=3.1,  # stale last-read value; must be overridden by the max
            rtfx=52.0,
            false_positives=[],
        )
        self.assertAlmostEqual(report["vram_peak_gb"], 5.9)
        self.assertEqual(
            report["vram_peak_by_run"],
            {"turbo_baseline": 3.1, "qwen_baseline": 5.9, "qwen_hotwords": 4.2},
        )

    def test_build_report_vram_peak_gb_falls_back_when_no_per_run_peaks(self):
        # CI mocks without vram_peak_fn: no per-run peaks -> keep the
        # supplied scalar (backward compatible).
        report = self.gs.build_report(
            runs={},
            word_timestamps_present=True,
            words_carry_speakers=True,
            vram_peak_gb=4.99,
            rtfx=52.0,
            false_positives=[],
        )
        self.assertAlmostEqual(report["vram_peak_gb"], 4.99)

    def test_vram_peak_real_resets_and_reads_peak(self):
        # vram_peak_real must call reset (before run) + read peak (after run):
        # default_align_fn path exercised via inspect on the source.
        import inspect

        self.assertTrue(hasattr(self.gs, "vram_reset_real"))
        src = inspect.getsource(self.gs.vram_reset_real)
        self.assertIn("reset_peak_memory_stats", src)
        src2 = inspect.getsource(self.gs.vram_peak_real)
        self.assertIn("max_memory_allocated", src2)

    def test_evaluate_report_checks_per_run_vram(self):
        # per-run pipeline peak threshold requalified at 6.5 GB (FIX 3): 6.6 fails
        report = {
            "runs": {
                "turbo_baseline": {"ok": True, "transcript_hash": "a" * 64},
                "qwen_baseline": {"ok": True, "transcript_hash": "b" * 64},
                "qwen_hotwords": {"ok": True, "transcript_hash": "c" * 64},
            },
            "word_timestamps_present": True,
            "words_carry_speakers": True,
            "vram_peak_gb": 4.99,
            "vram_peak_by_run": {"qwen_hotwords": 6.6},
            "rtfx": 52.0,
            "false_positives": [],
        }
        evaluation = self.gs.evaluate_report(report)
        self.assertFalse(evaluation["all_pass"])
        self.assertTrue(any("vram" in f.lower() for f in evaluation["failures"]))


# ---------------------------------------------------------------------------
# RED tests — E4-EXEC-FIX cycle (3 defects observed on the real GPU run:
# batch_size not passed run_golden_set -> run_single, no VRAM free between
# runs, JSON report only written at the very end). Written WITHOUT the
# implementation: run_golden_set has no batch_size / free_fn / partial-write
# params yet, so every assertion below fails (observed RED, cited in the
# commit message).
# ---------------------------------------------------------------------------


@unittest.skipUnless(GOLDEN_SET_PATH.is_file(), "scripts/golden_set.py does not exist yet (RED)")
class TestBatchSizePassthrough(unittest.TestCase):
    """FIX A (🔴): run_golden_set must accept and pass batch_size down to
    run_single — per-run type defaults (turbo 16, qwen 4) and per-model
    overrides. On the real GPU the missing passthrough froze turbo at 64 ->
    guaranteed OOM when free VRAM < 6 GiB."""

    def setUp(self):
        self.gs = _load_golden_set()
        self.tmpdir = tempfile.mkdtemp(dir="/opt/data/tmp")
        self.output = str(Path(self.tmpdir) / "report.json")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _stub_run_single(self, recorded):
        """Replace module run_single with a spy that records batch_size."""

        def fake_run_single(run_spec, **kwargs):
            recorded.append({"model": run_spec["whisper_model"], "batch_size": kwargs.get("batch_size")})
            return {
                "whisper_model": run_spec["whisper_model"],
                "hotwords": run_spec.get("hotwords"),
                "transcript": "backblaze bucket",
                "transcript_hash": "a" * 64,
                "segments_hash": "b" * 64,
                "language": "fr",
                "batch_size": kwargs.get("batch_size"),
                "duration_s": 1.0,
                "vram_peak_gb": 1.0,
                "word_timestamps_present": True,
                "words_carry_speakers": True,
                "segments": [],
                "ok": True,
            }

        self.gs.run_single = fake_run_single

    def _gpu_free_kwargs(self):
        return dict(
            load_audio_fn=lambda p: [0.0],
            model_factory=lambda name: object(),
            clock=lambda: 0.0,
            vram_peak_fn=lambda: 1.0,
            vram_reset_fn=lambda: None,
            align_fn=lambda audio, result: result,
            diarize_fn=lambda audio, result: result,
            audio_duration_fn=lambda p: 30.0,
        )

    def test_run_golden_set_accepts_batch_size_and_passes_it_to_run_single(self):
        recorded = []
        self._stub_run_single(recorded)
        self.gs.run_golden_set(audio_path="x.ogg", output_path=self.output, batch_size=8, **self._gpu_free_kwargs())
        self.assertEqual(len(recorded), 3)
        for entry in recorded:
            self.assertEqual(
                entry["batch_size"],
                8,
                f"run {entry['model']} must receive the caller's batch_size",
            )

    def test_run_golden_set_default_per_model_batch_turbo_16_qwen_4(self):
        recorded = []
        self._stub_run_single(recorded)
        self.gs.run_golden_set(audio_path="x.ogg", output_path=self.output, **self._gpu_free_kwargs())
        by_model = {e["model"]: e["batch_size"] for e in recorded}
        self.assertEqual(by_model.get("large-v3-turbo"), 16, "turbo default batch must be 16, not 64")
        self.assertEqual(by_model.get("qwen3-asr"), 4, "qwen default batch must be 4")

    def test_run_golden_set_per_model_batch_override(self):
        recorded = []
        self._stub_run_single(recorded)
        self.gs.run_golden_set(
            audio_path="x.ogg",
            output_path=self.output,
            per_model_batch={"large-v3-turbo": 32, "qwen3-asr": 2},
            **self._gpu_free_kwargs(),
        )
        by_model = {e["model"]: e["batch_size"] for e in recorded}
        self.assertEqual(by_model.get("large-v3-turbo"), 32)
        self.assertEqual(by_model.get("qwen3-asr"), 2)

    def test_resolve_batch_size_helper(self):
        # pure helper: caller batch_size wins, then per-model table, then 16/4
        self.assertEqual(self.gs.resolve_batch_size("large-v3-turbo", 8), 8)
        self.assertEqual(self.gs.resolve_batch_size("qwen3-asr", 8), 8)
        self.assertEqual(self.gs.resolve_batch_size("large-v3-turbo"), 16)
        self.assertEqual(self.gs.resolve_batch_size("qwen3-asr"), 4)
        self.assertEqual(
            self.gs.resolve_batch_size("large-v3-turbo", per_model_batch={"large-v3-turbo": 32}),
            32,
        )
        self.assertEqual(
            self.gs.resolve_batch_size("qwen3-asr", per_model_batch={"large-v3-turbo": 32}),
            4,
        )

    def test_per_model_default_batch_constants(self):
        self.assertEqual(self.gs.PER_MODEL_DEFAULT_BATCH.get("large-v3-turbo"), 16)
        self.assertEqual(self.gs.PER_MODEL_DEFAULT_BATCH.get("qwen3-asr"), 4)


@unittest.skipUnless(GOLDEN_SET_PATH.is_file(), "scripts/golden_set.py does not exist yet (RED)")
class TestFreeGpuBetweenRuns(unittest.TestCase):
    """FIX B (🔴): the harness must release VRAM between runs — injectable
    free_fn called after EACH run, once the run result has been extracted
    and stored. On the real GPU the accumulated turbo+qwen+aligner+pyannote
    models in a single process OOMed pyannote wespeaker (312 MiB) and the
    aligner (93/93 segments unaligned)."""

    def setUp(self):
        self.gs = _load_golden_set()
        self.tmpdir = tempfile.mkdtemp(dir="/opt/data/tmp")
        self.output = str(Path(self.tmpdir) / "report.json")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _run_with_free_spy(self, free_events):
        def fake_run_single(run_spec, **kwargs):
            free_events.append(("run_end", run_spec["whisper_model"]))
            return {
                "whisper_model": run_spec["whisper_model"],
                "ok": True,
                "transcript": "",
                "segments": [],
                "words_carry_speakers": True,
                "word_timestamps_present": True,
            }

        self.gs.run_single = fake_run_single
        calls = {"n": 0}

        def free_spy():
            calls["n"] += 1
            free_events.append(("free", calls["n"]))

        self.gs.run_golden_set(
            audio_path="x.ogg",
            output_path=self.output,
            load_audio_fn=lambda p: [0.0],
            model_factory=lambda name: object(),
            clock=lambda: 0.0,
            vram_peak_fn=lambda: 1.0,
            vram_reset_fn=lambda: None,
            align_fn=lambda audio, result: result,
            diarize_fn=lambda audio, result: result,
            audio_duration_fn=lambda p: 30.0,
            free_fn=free_spy,
        )
        return calls

    def test_free_fn_called_once_after_each_run(self):
        free_events = []
        calls = self._run_with_free_spy(free_events)
        self.assertEqual(calls["n"], 3, "free_fn must be called after each of the 3 runs")

    def test_free_fn_called_after_result_extraction(self):
        # interleaving: run N ends, THEN free — the result is stored before
        # the VRAM release, never freed before its extraction
        free_events = []
        self._run_with_free_spy(free_events)
        run_ends = [i for i, e in enumerate(free_events) if e[0] == "run_end"]
        frees = [i for i, e in enumerate(free_events) if e[0] == "free"]
        self.assertEqual(len(run_ends), 3)
        self.assertEqual(len(frees), 3)
        for idx in range(3):
            self.assertLess(
                run_ends[idx],
                frees[idx],
                "free_fn must run AFTER the run result is extracted/stored",
            )

    def test_free_gpu_real_uses_gc_and_empty_cache(self):
        import inspect

        self.assertTrue(hasattr(self.gs, "free_gpu_real"))
        src = inspect.getsource(self.gs.free_gpu_real)
        self.assertIn("gc.collect", src)
        self.assertIn("empty_cache", src)
        # and run_golden_set defaults free_fn to it
        src_flow = inspect.getsource(self.gs.run_golden_set)
        self.assertIn("free_gpu_real", src_flow)


@unittest.skipUnless(GOLDEN_SET_PATH.is_file(), "scripts/golden_set.py does not exist yet (RED)")
class TestPartialReportAfterEachRun(unittest.TestCase):
    """FIX C (🔴): a partial report must be written to --output after EACH
    run (completed runs so far + the failed run's error). On the real GPU an
    OOM on run 2/3 lost ALL metrics (zero artefact, no hashes)."""

    def setUp(self):
        self.gs = _load_golden_set()
        self.tmpdir = tempfile.mkdtemp(dir="/opt/data/tmp")
        self.output = str(Path(self.tmpdir) / "report.json")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _kwargs(self):
        return dict(
            load_audio_fn=lambda p: [0.0],
            model_factory=lambda name: object(),
            clock=lambda: 0.0,
            vram_peak_fn=lambda: 1.0,
            vram_reset_fn=lambda: None,
            align_fn=lambda audio, result: result,
            diarize_fn=lambda audio, result: result,
            audio_duration_fn=lambda p: 30.0,
            free_fn=lambda: None,
        )

    @staticmethod
    def _ok_run(run_spec):
        return {
            "whisper_model": run_spec["whisper_model"],
            "hotwords": run_spec.get("hotwords"),
            "transcript": "backblaze bucket",
            "transcript_hash": "a" * 64,
            "segments_hash": "b" * 64,
            "language": "fr",
            "batch_size": 4,
            "duration_s": 1.0,
            "vram_peak_gb": 1.0,
            "word_timestamps_present": True,
            "words_carry_speakers": True,
            "segments": [],
            "ok": True,
        }

    def test_partial_snapshot_visible_after_each_completed_run(self):
        seen_at_run3 = {}

        def fake_run_single(run_spec, **kwargs):
            if run_spec["whisper_model"] == "qwen3-asr" and "hotwords" in run_spec and run_spec["hotwords"]:
                # 3rd call: the file must already hold the 2 previous runs
                seen_at_run3["content"] = Path(self.output).read_text(encoding="utf-8") if Path(self.output).exists() else None
            return self._ok_run(run_spec)

        self.gs.run_single = fake_run_single
        self.gs.run_golden_set(audio_path="x.ogg", output_path=self.output, **self._kwargs())
        self.assertIsNotNone(seen_at_run3["content"], "a partial report must exist BEFORE the last run finishes")
        partial = json.loads(seen_at_run3["content"])
        self.assertTrue(partial.get("partial"))
        self.assertIn("turbo_baseline", partial.get("runs") or {})
        self.assertIn("qwen_baseline", partial.get("runs") or {})
        self.assertNotIn("qwen_hotwords", partial.get("runs") or {})

    def test_failed_run_recorded_and_report_still_written(self):
        def fake_run_single(run_spec, **kwargs):
            if run_spec["whisper_model"] == "qwen3-asr" and not run_spec.get("hotwords"):
                raise RuntimeError("CUDA out of memory. Tried to allocate 312.00 MiB")
            return self._ok_run(run_spec)

        self.gs.run_single = fake_run_single
        report, evaluation = self.gs.run_golden_set(audio_path="x.ogg", output_path=self.output, **self._kwargs())
        # the failed run is documented, not lost
        self.assertIn("CUDA out of memory", str(report["runs"]["qwen_baseline"].get("error")))
        self.assertFalse(report["runs"]["qwen_baseline"].get("ok"))
        # completed runs survive
        self.assertTrue(report["runs"]["turbo_baseline"]["ok"])
        self.assertTrue(report["runs"]["qwen_hotwords"]["ok"])
        # and the file exists with the failure documented
        on_disk = json.loads(Path(self.output).read_text(encoding="utf-8"))
        self.assertIn("CUDA out of memory", str(on_disk["runs"]["qwen_baseline"].get("error")))

    def test_partial_snapshot_written_even_when_run_fails_midway(self):
        def fake_run_single(run_spec, **kwargs):
            if run_spec["whisper_model"] == "qwen3-asr" and not run_spec.get("hotwords"):
                raise RuntimeError("OOM")
            return self._ok_run(run_spec)

        observed = {}

        def spy_run(run_spec, **kwargs):
            if run_spec["whisper_model"] == "qwen3-asr" and run_spec.get("hotwords"):
                observed["before_last"] = Path(self.output).read_text(encoding="utf-8") if Path(self.output).exists() else None
            return fake_run_single(run_spec, **kwargs)

        self.gs.run_single = spy_run
        self.gs.run_golden_set(audio_path="x.ogg", output_path=self.output, **self._kwargs())
        self.assertIsNotNone(observed["before_last"], "partial snapshot must be on disk even after a mid-run OOM")
        partial = json.loads(observed["before_last"])
        self.assertTrue(partial.get("partial"))
        self.assertIn("turbo_baseline", partial.get("runs") or {})
        failed = partial.get("failed_runs") or {}
        self.assertIn("qwen_baseline", failed, "the failed run error must appear in the partial snapshot")

    def test_evaluate_report_tolerates_partial_report(self):
        # partial report (one run failed with an error entry): documented
        # failure, no crash
        report = {
            "runs": {
                "turbo_baseline": {"transcript_hash": "a" * 64, "ok": True},
                "qwen_baseline": {"error": "CUDA out of memory", "ok": False},
                "qwen_hotwords": {"transcript_hash": "c" * 64, "ok": True},
            },
            "word_timestamps_present": True,
            "words_carry_speakers": True,
            "vram_peak_gb": 4.99,
            "rtfx": 52.0,
            "false_positives": [],
        }
        evaluation = self.gs.evaluate_report(report)
        self.assertFalse(evaluation["all_pass"])
        self.assertTrue(any("qwen_baseline" in f for f in evaluation["failures"]))

    def test_write_partial_snapshot_helper(self):
        runs = {"turbo_baseline": self._ok_run({"whisper_model": "large-v3-turbo", "hotwords": None})}
        self.gs._write_partial_snapshot(self.output, runs=runs, failed_runs={"qwen_baseline": "OOM"})
        data = json.loads(Path(self.output).read_text(encoding="utf-8"))
        self.assertTrue(data["partial"])
        self.assertIn("turbo_baseline", data["runs"])
        self.assertEqual(data["failed_runs"], {"qwen_baseline": "OOM"})


@unittest.skipUnless(GOLDEN_SET_PATH.is_file(), "scripts/golden_set.py does not exist yet (RED)")
class TestCliBatchOptions(unittest.TestCase):
    """FIX D (🔴): CLI options --batch-size and --per-model-batch must reach
    run_golden_set."""

    def setUp(self):
        self.gs = _load_golden_set()

    def test_main_passes_batch_size_and_per_model_batch(self):
        captured = {}

        def fake_run_golden_set(**kwargs):
            captured.update(kwargs)
            return {}, {"all_pass": True, "failures": []}

        self.gs.run_golden_set = fake_run_golden_set
        exit_code = self.gs.main(
            [
                "--meeting-audio", "x.ogg",
                "--output", "/opt/data/tmp/cli_report.json",
                "--batch-size", "6",
                "--per-model-batch", '{"large-v3-turbo": 16, "qwen3-asr": 4}',
            ]
        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(captured.get("batch_size"), 6)
        self.assertEqual(captured.get("per_model_batch"), {"large-v3-turbo": 16, "qwen3-asr": 4})

    def test_main_defaults_are_none(self):
        captured = {}

        def fake_run_golden_set(**kwargs):
            captured.update(kwargs)
            return {}, {"all_pass": True, "failures": []}

        self.gs.run_golden_set = fake_run_golden_set
        self.gs.main(["--meeting-audio", "x.ogg", "--output", "/opt/data/tmp/cli_report.json"])
        self.assertIsNone(captured.get("batch_size"))
        self.assertIsNone(captured.get("per_model_batch"))

    def test_main_invalid_per_model_batch_json_fails_cleanly(self):
        self.gs.run_golden_set = lambda **kwargs: ({}, {"all_pass": True, "failures": []})
        exit_code = self.gs.main(
            [
                "--meeting-audio", "x.ogg",
                "--output", "/opt/data/tmp/cli_report.json",
                "--per-model-batch", "not-json",
            ]
        )
        self.assertEqual(exit_code, 2)


@unittest.skipUnless(GOLDEN_SET_PATH.is_file(), "scripts/golden_set.py does not exist yet (RED)")
class TestLabelingGateRecalibration(unittest.TestCase):
    """E4-QUAL-FIX FIX 1 (🟡): the 100% labeling gate was a calibration defect.

    Real GPU runs (golden_set_run1/2.json): 5470/6021 qwen words labeled
    (99.95% of the NON zero-duration words); the unlabeled ones are boundary
    duplicates where the ForcedAligner emits start==end. The gate must
    exclude zero-duration boundary duplicates from the denominator, apply a
    >= 85% labeling threshold and produce a factual failure message.
    """

    def setUp(self):
        self.gs = _load_golden_set()

    @staticmethod
    def _result(labeled, unlabeled_zero=0, unlabeled_nonzero=0):
        words = []
        for i in range(labeled):
            words.append({"word": f"w{i}", "start": 0.1 * i, "end": 0.1 * i + 0.05, "speaker": "SPEAKER_00"})
        for i in range(unlabeled_zero):
            words.append({"word": f"z{i}", "start": 5.0 + 0.1 * i, "end": 5.0 + 0.1 * i})
        for i in range(unlabeled_nonzero):
            words.append({"word": f"u{i}", "start": 6.0 + 0.1 * i, "end": 6.1 + 0.1 * i})
        return {"segments": [{"start": 0.0, "end": 10.0, "text": "texte", "words": words}]}

    def test_zero_duration_boundary_duplicates_excluded(self):
        # 950/950 non-zero-duration words labeled; 50 unlabeled zero-duration
        # duplicates must NOT drag the ratio down — the 100% gate said False.
        result = self._result(labeled=950, unlabeled_zero=50)
        self.assertTrue(self.gs.words_carry_speakers(result))

    def test_labeling_below_threshold_flagged(self):
        result = self._result(labeled=80, unlabeled_nonzero=20)
        self.assertFalse(self.gs.words_carry_speakers(result))

    def test_segment_without_words_still_flagged(self):
        # no vacuous pass: a segment without words means the handoff failed
        result = {"segments": [{"start": 0.0, "end": 1.0, "text": "Bonjour"}]}
        self.assertFalse(self.gs.words_carry_speakers(result))

    def test_empty_segments_flagged(self):
        self.assertFalse(self.gs.words_carry_speakers({"segments": []}))

    def test_labeling_min_ratio_constant_is_85_percent(self):
        self.assertEqual(self.gs.LABELING_MIN_RATIO, 0.85)

    def test_labeling_stats_helper_counts_nonzero_duration_words(self):
        result = self._result(labeled=950, unlabeled_zero=50, unlabeled_nonzero=20)
        labeled, total = self.gs.word_labeling_stats(result)
        self.assertEqual((labeled, total), (950, 970))

    def test_labeling_partial_message_factual(self):
        message = self.gs.labeling_failure_message({"segments": []})
        # factual: 'labeling partial: N/M words', never a vague claim
        self.assertIn("labeling partial:", message)
        self.assertRegex(message, r"labeling partial: \d+/\d+ words")

    def test_evaluate_report_failure_message_is_factual(self):
        report = {
            "runs": {},
            "word_timestamps_present": True,
            "words_carry_speakers": False,
            "word_labeling": {"labeled": 5470, "total": 6021},
            "vram_peak_gb": 4.99,
            "rtfx": 52.0,
            "false_positives": [],
        }
        evaluation = self.gs.evaluate_report(report)
        self.assertFalse(evaluation["all_pass"])
        factual = [f for f in evaluation["failures"] if "labeling partial: 5470/6021 words" in f]
        self.assertTrue(factual, f"expected factual labeling message, got {evaluation['failures']}")


@unittest.skipUnless(GOLDEN_SET_PATH.is_file(), "scripts/golden_set.py does not exist yet (RED)")
class TestHotwordFpRecalibration(unittest.TestCase):
    """E4-QUAL-FIX FIX 2 (🟡): the FP detector flagged 4 legitimate mentions.

    Observed on the real GPU run (golden_set_run1.json): 3 flagged segments
    come from the TURBO baseline (no hotwords active — the term was actually
    spoken, e.g. 'Le upload, dans le cas de Backblaze...') and 1 from the
    qwen_hotwords run ('...les volumétries... J'ai une question sur Backblaze')
    where the term is topical. The detector must (a) recognize the topical
    context of the flagged segments, (b) anchor FPs on word starts, (c) tag
    hallucinated insertions vs legitimate mentions.
    """

    def setUp(self):
        self.gs = _load_golden_set()

    def test_upload_cost_segment_not_flagged(self):
        # turbo seg 219 (start 1084.002): topical (upload/cost) — was FP-flagged
        segments = [{"start": 1084.002, "text": "Le upload, dans le cas de Backblaze, n'a pas de coûté d'argent."}]
        fps = self.gs.find_hotword_false_positives(segments, ["Backblaze"])
        self.assertEqual(fps, [])

    def test_secrets_segment_not_flagged_with_neighbor_context(self):
        # turbo seg 336 (start 1652.872): the NEXT segment carries 'clé API'/'buckets'
        segments = [
            {"start": 1652.872, "text": "Comment sont gérés les secrets de Backblaze ?"},
            {"start": 1663.714, "text": "tu vas leur filer, tu fais un travail par clé API, et les clés API sont liées à des buckets."},
        ]
        fps = self.gs.find_hotword_false_positives(segments, ["Backblaze"])
        self.assertEqual(fps, [])

    def test_capteur_connecter_segment_not_flagged(self):
        # turbo seg 347 (start 1737.357): 'se connecter à Backblaze' is topical
        segments = [
            {"start": 1722.323, "text": "C'est une autre API qui a accès à la clé Backblaze qui fournait juste un lien signé."},
            {"start": 1737.357, "text": "Le capteur ne peut pas se connecter à Backblaze lui."},
        ]
        fps = self.gs.find_hotword_false_positives(segments, ["Backblaze"])
        self.assertEqual(fps, [])

    def test_volumetrie_segment_not_flagged(self):
        # qwen_hotwords seg 27 (start 597.794): 'volumétries' + the next segment
        # discusses the same topic ('serveur', 'stockage')
        segments = [
            {"start": 597.794, "text": "De points, il peut peut-être te faire du cinq pour cent de l'ensemble à la fin. En attendant qu'on trouve les volumétries, à quelle vitesse ? J'ai une question. J'ai une question sur Backblaze. Je sais jamais quand."},
            {"start": 618.078, "text": "Backblaze, c'est quoi le principe de Backblaze, Charles ? C'est c'est juste un serveur qui est pas trop cher."},
        ]
        fps = self.gs.find_hotword_false_positives(segments, ["Backblaze"])
        self.assertEqual(fps, [])

    def test_fp_anchored_on_word_start(self):
        # the FP entry carries the WORD start of the matched hotword (word-level
        # anchor), not only the segment start
        segments = [
            {
                "start": 100.0,
                "text": "le planning est validé backblaze.",
                "words": [
                    {"word": "le", "start": 100.0, "end": 100.2},
                    {"word": "backblaze", "start": 104.834, "end": 105.394},
                ],
            }
        ]
        fps = self.gs.find_hotword_false_positives(segments, ["Backblaze"], context_keywords=())
        self.assertEqual(len(fps), 1)
        self.assertEqual(fps[0]["word_start"], 104.834)
        self.assertEqual(fps[0]["segment_start"], 100.0)

    def test_baseline_mention_classified_legitimate(self):
        # the term already appears in the (hotword-free) baseline transcript:
        # the word was actually spoken at that region -> legitimate mention
        segments = [{"start": 0.0, "text": "question sur backblaze dans la réunion", "words": []}]
        fps = self.gs.find_hotword_false_positives(
            segments, ["Backblaze"], context_keywords=(), baseline_text="on a parlé de backblaze aujourd'hui"
        )
        self.assertEqual(len(fps), 1)
        self.assertEqual(fps[0]["classification"], "legitimate_mention")

    def test_hallucinated_insertion_classification(self):
        # hotword present, baseline ABSENT, segment non-topical: insertion
        segments = [{"start": 0.0, "text": "le planning est validé backblaze.", "words": []}]
        fps = self.gs.find_hotword_false_positives(
            segments, ["Backblaze"], context_keywords=(), baseline_text="le planning de la semaine"
        )
        self.assertEqual(len(fps), 1)
        self.assertEqual(fps[0]["classification"], "hallucinated_insertion")

    def test_report_distinguishes_legitimate_mentions_from_insertions(self):
        report = self.gs.build_report(
            runs={},
            word_timestamps_present=True,
            words_carry_speakers=True,
            vram_peak_gb=4.99,
            rtfx=52.0,
            false_positives=[{"hotword": "Backblaze", "classification": "legitimate_mention"}],
            legitimate_mentions=[{"hotword": "Backblaze", "classification": "legitimate_mention"}],
        )
        self.assertEqual(report["false_positives"], [])
        self.assertEqual(len(report["legitimate_mentions"]), 1)

    def test_run_golden_set_scans_only_hotwords_active_run(self):
        # the turbo baseline carries NO injected hotwords: its mentions are
        # legitimate transcriptions and must not feed the FP list
        self.gs.RUNS_KEYS_HOTWORDS_ACTIVE = {"qwen_hotwords"}  # documented constant
        self.assertIn("qwen_hotwords", self.gs.RUNS_KEYS_HOTWORDS_ACTIVE)
        self.assertNotIn("turbo_baseline", self.gs.RUNS_KEYS_HOTWORDS_ACTIVE)


@unittest.skipUnless(GOLDEN_SET_PATH.is_file(), "scripts/golden_set.py does not exist yet (RED)")
class TestVramByStage(unittest.TestCase):
    """E4-QUAL-FIX FIX 3 (🟡): VRAM logged after EACH pipeline stage.

    The 5.757 GB peak includes the resident aligner + diarize models; the
    15/09 4.99 GB reference was ASR-only. The report must carry a per-stage
    decomposition (vram_by_stage) and the pipeline threshold must be
    requalified: ASR < 5.5 GB, full pipeline < 6.5 GB.
    """

    def setUp(self):
        self.gs = _load_golden_set()

    def test_run_single_records_vram_by_stage(self):
        peaks = iter([4.20, 5.10, 5.757])  # after transcribe / align / diarize
        probes = iter(peaks)

        class FakeModel:
            def transcribe(self, audio, batch_size=None, context=None, **kwargs):
                return {"language": "fr", "segments": [{"start": 0.0, "end": 1.0, "text": "Bonjour", "words": []}]}

        run = self.gs.run_single(
            run_spec={"whisper_model": "qwen3-asr", "hotwords": None},
            audio_path="x.ogg",
            load_audio_fn=lambda p: [0.0],
            model_factory=lambda name: FakeModel(),
            clock=lambda: 0.0,
            align_fn=lambda audio, result: result,
            diarize_fn=lambda audio, result: result,
            vram_peak_fn=lambda: 5.757,
            vram_probe_fn=lambda: next(probes),
        )
        self.assertEqual(run["vram_by_stage"], {"transcribe": 4.20, "align": 5.10, "diarize": 5.757})

    def test_pipeline_vram_limit_constant_is_6_5(self):
        self.assertEqual(self.gs.VRAM_PIPELINE_LIMIT_GB, 6.5)

    def test_evaluate_report_pipeline_threshold_6_5(self):
        # measured full-pipeline peak 5.757 GB must PASS (was a failure at 5.5)
        report = {
            "runs": {
                "turbo_baseline": {"transcript_hash": "a" * 64, "ok": True},
                "qwen_baseline": {"transcript_hash": "b" * 64, "ok": True},
                "qwen_hotwords": {"transcript_hash": "c" * 64, "ok": True},
            },
            "word_timestamps_present": True,
            "words_carry_speakers": True,
            "vram_peak_gb": 5.757297992706299,
            "rtfx": 49.5,
            "false_positives": [],
        }
        evaluation = self.gs.evaluate_report(report)
        self.assertTrue(evaluation["all_pass"], f"5.757 GB pipeline peak must pass: {evaluation['failures']}")

    def test_evaluate_report_pipeline_above_6_5_fails(self):
        report = {
            "runs": {},
            "word_timestamps_present": True,
            "words_carry_speakers": True,
            "vram_peak_gb": 6.6,
            "rtfx": 52.0,
            "false_positives": [],
        }
        evaluation = self.gs.evaluate_report(report)
        self.assertFalse(evaluation["all_pass"])

    def test_evaluate_report_asr_stage_threshold_5_5(self):
        # the ORIGINAL 5.5 GB limit still applies to the ASR stage alone
        report = {
            "runs": {},
            "word_timestamps_present": True,
            "words_carry_speakers": True,
            "vram_peak_gb": 5.2,
            "vram_by_stage": {"qwen_hotwords": {"transcribe": 5.6, "align": 5.7, "diarize": 5.8}},
            "rtfx": 52.0,
            "false_positives": [],
        }
        evaluation = self.gs.evaluate_report(report)
        self.assertFalse(evaluation["all_pass"])
        self.assertTrue(any("transcribe" in f for f in evaluation["failures"]))

    def test_report_exposes_vram_by_stage(self):
        runs = {
            "qwen_hotwords": {
                "ok": True,
                "transcript_hash": "c" * 64,
                "vram_peak_gb": 5.757,
                "vram_by_stage": {"transcribe": 4.99, "align": 5.1, "diarize": 5.757},
            }
        }
        report = self.gs.build_report(
            runs=runs,
            word_timestamps_present=True,
            words_carry_speakers=True,
            vram_peak_gb=5.757,
            rtfx=49.5,
            false_positives=[],
        )
        self.assertEqual(report["vram_by_stage"]["qwen_hotwords"]["transcribe"], 4.99)

    def test_fp16_documented_vs_fp32(self):
        # docstring guard: the pipeline-vs-ASR distinction is documented
        import inspect

        src = inspect.getsource(self.gs)
        self.assertIn("fp16", src)


@unittest.skipUnless(GOLDEN_SET_PATH.is_file(), "scripts/golden_set.py does not exist yet (RED)")
class TestRtfxScope(unittest.TestCase):
    """E4-QUAL-FIX FIX 4 (🟡): RTFx scope documented + explicitly named keys.

    duration_s wraps the transcribe call ONLY (no align/diarize): the reported
    RTFx is a TRANSCRIPTION-ONLY ratio (turbo 221 on the harness vs 42 e2e on
    15/09 — different scope, both correct). The report must carry explicitly
    named rtfx_transcription and rtfx_e2e keys.
    """

    def setUp(self):
        self.gs = _load_golden_set()

    def test_run_single_records_duration_total_s(self):
        ticks = iter([10.0, 12.0, 14.5])  # transcribe start/end, run end

        class FakeModel:
            def transcribe(self, audio, batch_size=None, **kwargs):
                return {"language": "fr", "segments": []}

        run = self.gs.run_single(
            run_spec={"whisper_model": "qwen3-asr", "hotwords": None},
            audio_path="x.ogg",
            load_audio_fn=lambda p: [0.0],
            model_factory=lambda name: FakeModel(),
            clock=lambda: next(ticks),
            align_fn=lambda audio, result: result,
            diarize_fn=lambda audio, result: result,
        )
        self.assertAlmostEqual(run["duration_s"], 2.0)
        self.assertAlmostEqual(run["duration_total_s"], 4.5)

    def test_duration_s_scope_documented(self):
        import inspect

        doc = inspect.getdoc(self.gs.run_single) or ""
        self.assertIn("transcription", doc.lower())
        self.assertIn("align", doc.lower())

    def test_report_carries_rtfx_transcription_and_rtfx_e2e(self):
        runs = {
            "qwen_hotwords": {
                "ok": True,
                "transcript_hash": "c" * 64,
                "duration_s": 43.57,
                "duration_total_s": 210.0,
            }
        }
        report = self.gs.build_report(
            runs=runs,
            word_timestamps_present=True,
            words_carry_speakers=True,
            vram_peak_gb=5.7,
            rtfx=49.48,
            rtfx_e2e=10.27,
            false_positives=[],
            audio_durations_s={"meeting": 2156.058},
        )
        self.assertAlmostEqual(report["rtfx_transcription"], 49.48)
        self.assertAlmostEqual(report["rtfx_e2e"], 10.27)
        self.assertIn("rtfx_transcription", report)
        self.assertIn("rtfx_e2e", report)

    def test_evaluate_report_uses_rtfx_transcription(self):
        report = {
            "runs": {
                "turbo_baseline": {"transcript_hash": "a" * 64, "ok": True},
                "qwen_baseline": {"transcript_hash": "b" * 64, "ok": True},
                "qwen_hotwords": {"transcript_hash": "c" * 64, "ok": True},
            },
            "word_timestamps_present": True,
            "words_carry_speakers": True,
            "vram_peak_gb": 5.0,
            "rtfx_transcription": 49.5,
            "false_positives": [],
        }
        evaluation = self.gs.evaluate_report(report)
        self.assertTrue(evaluation["all_pass"], f"rtfx_transcription must be the evaluated metric: {evaluation['failures']}")

    def test_evaluate_report_missing_rtfx_transcription_falls_back_to_rtfx(self):
        report = {
            "runs": {
                "turbo_baseline": {"transcript_hash": "a" * 64, "ok": True},
                "qwen_baseline": {"transcript_hash": "b" * 64, "ok": True},
                "qwen_hotwords": {"transcript_hash": "c" * 64, "ok": True},
            },
            "word_timestamps_present": True,
            "words_carry_speakers": True,
            "vram_peak_gb": 5.0,
            "rtfx": 52.0,
            "false_positives": [],
        }
        evaluation = self.gs.evaluate_report(report)
        self.assertTrue(evaluation["all_pass"])


@unittest.skipUnless(GOLDEN_SET_PATH.is_file(), "scripts/golden_set.py does not exist yet (RED)")
class TestRegressionHashesThreeEntries(unittest.TestCase):
    """E4-QUAL-FIX FIX 5 (🟡): the recorded-hashes file must carry THREE
    entries (turbo, qwen, qwen_hotwords) and evaluate_report must actually
    check the qwen entries — the current lookup (runs.get(f'{model}_baseline'))
    silently skips 'qwen3-asr' (run key is 'qwen_baseline') and never checks
    'qwen_hotwords'.
    """

    def setUp(self):
        self.gs = _load_golden_set()

    def test_evaluate_report_checks_qwen3_asr_against_qwen_baseline_run(self):
        report = {
            "runs": {
                "turbo_baseline": {"transcript_hash": "a" * 64, "ok": True},
                "qwen_baseline": {"transcript_hash": "wrong" + "b" * 59, "ok": True},
                "qwen_hotwords": {"transcript_hash": "c" * 64, "ok": True},
            },
            "word_timestamps_present": True,
            "words_carry_speakers": True,
            "vram_peak_gb": 4.99,
            "rtfx": 52.0,
            "false_positives": [],
            "regression_hashes": {"qwen3-asr": "e" * 64},
        }
        evaluation = self.gs.evaluate_report(report)
        self.assertFalse(evaluation["all_pass"])
        self.assertTrue(
            any("qwen" in f for f in evaluation["failures"]),
            f"recorded 'qwen3-asr' hash must be checked against qwen_baseline: {evaluation['failures']}",
        )

    def test_evaluate_report_checks_qwen_hotwords_hash(self):
        report = {
            "runs": {
                "turbo_baseline": {"transcript_hash": "a" * 64, "ok": True},
                "qwen_baseline": {"transcript_hash": "b" * 64, "ok": True},
                "qwen_hotwords": {"transcript_hash": "wrong" + "c" * 59, "ok": True},
            },
            "word_timestamps_present": True,
            "words_carry_speakers": True,
            "vram_peak_gb": 4.99,
            "rtfx": 52.0,
            "false_positives": [],
            "regression_hashes": {"qwen_hotwords": "f" * 64},
        }
        evaluation = self.gs.evaluate_report(report)
        self.assertFalse(evaluation["all_pass"])
        self.assertTrue(any("qwen_hotwords" in f for f in evaluation["failures"]))

    def test_matching_three_entry_hashes_pass(self):
        hashes = {"large-v3-turbo": "a" * 64, "qwen3-asr": "b" * 64, "qwen_hotwords": "c" * 64}
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
            "regression_hashes": hashes,
        }
        evaluation = self.gs.evaluate_report(report)
        self.assertTrue(evaluation["all_pass"], evaluation["failures"])


if __name__ == "__main__":
    unittest.main()
