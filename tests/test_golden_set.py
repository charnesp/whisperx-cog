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

        ticks = iter([10.0, 12.5])
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
        report = {
            "runs": {
                "turbo_baseline": {"ok": True, "transcript_hash": "a" * 64},
                "qwen_baseline": {"ok": True, "transcript_hash": "b" * 64},
                "qwen_hotwords": {"ok": True, "transcript_hash": "c" * 64},
            },
            "word_timestamps_present": True,
            "words_carry_speakers": True,
            "vram_peak_gb": 4.99,
            "vram_peak_by_run": {"qwen_hotwords": 6.0},
            "rtfx": 52.0,
            "false_positives": [],
        }
        evaluation = self.gs.evaluate_report(report)
        self.assertFalse(evaluation["all_pass"])
        self.assertTrue(any("vram" in f.lower() for f in evaluation["failures"]))


if __name__ == "__main__":
    unittest.main()
