"""GPU-free unit tests for the Qwen3-ASR backend (E3).

Covers:
- hotwords -> Qwen context (qwen path)
- hotwords -> faster-whisper asr_options (whisper path)
- batch size: QWEN_DEFAULT_BATCH=4, clamp to 8, explicit passthrough
- ENABLE_QWEN kill-switch
- MODEL_MAP accepts qwen3-asr
- hotwords absent -> empty context (baseline neutrality)
- qwen language handling (skip detect loop, pass-through)
- context truncation cap with lengths-only log

predict.py imports torch/whisperx/cog/ffmpeg at module level, which are not
installed in the test environment; tests import the qwen helpers through
importlib with stub modules registered in sys.modules first.
"""

from __future__ import annotations

import importlib
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bridge"))

import openai_compat

MODEL_MAP = openai_compat.MODEL_MAP
build_cog_input = openai_compat.build_cog_input


class _NoCtx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


class _Any:
    def __init__(self, *a, **k):
        pass


def _install_predict_stub_modules():
    """Register minimal stubs so predict.py can be imported without GPU deps."""
    stubs: dict[str, types.ModuleType] = {}

    def make_module(name: str, **attrs):
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        sys.modules[name] = mod
        stubs[name] = mod
        return mod

    class _BasePredictor:
        def setup(self):
            pass

    make_module(
        "torch",
        inference_mode=lambda: _NoCtx(),
        cuda=types.SimpleNamespace(
            empty_cache=lambda: None,
            max_memory_reserved=lambda: 0,
            is_available=lambda: False,
        ),
    )
    make_module("cog", BasePredictor=_BasePredictor, Input=_Any, Path=_Any, BaseModel=object)
    make_module(
        "ffmpeg",
        probe=lambda p: {"streams": [], "format": {}},
        Error=Exception,
        input=lambda *a, **k: None,
        output=lambda *a, **k: None,
    )
    make_module(
        "whisperx",
        load_model=lambda *a, **k: None,
        load_audio=lambda *a, **k: None,
        align=lambda *a, **k: None,
        load_align_model=lambda *a, **k: (None, None),
        assign_word_speakers=lambda *a, **k: None,
    )
    make_module("whisperx.audio", N_SAMPLES=480000, log_mel_spectrogram=lambda *a, **k: None)
    make_module("whisperx.diarize", DiarizationPipeline=lambda *a, **k: None)
    make_module("whisperx.alignment", DEFAULT_ALIGN_MODELS_TORCH={}, DEFAULT_ALIGN_MODELS_HF={})
    asr_qwen = make_module("whisperx.asr_qwen", load_model=lambda *a, **k: None)
    make_module(
        "whisperx.alignment_qwen",
        load_align_model=lambda *a, **k: (None, None),
        align=lambda *a, **k: None,
    )
    return stubs, asr_qwen


_install_predict_stub_modules()

predict = importlib.import_module("predict")


class TestModelMap(unittest.TestCase):
    def test_model_map_accepts_qwen3_asr(self):
        self.assertEqual(MODEL_MAP["qwen3-asr"], "qwen3-asr")

    def test_whisper_models_unchanged(self):
        self.assertEqual(MODEL_MAP["whisper-1"], "large-v3-turbo")
        self.assertEqual(MODEL_MAP["large-v3-turbo"], "large-v3-turbo")


class TestQwenBatchHelpers(unittest.TestCase):
    def test_default_batch_when_none(self):
        self.assertEqual(predict.resolve_qwen_batch_size(None), predict.QWEN_DEFAULT_BATCH)

    def test_default_batch_when_absent_flag(self):
        self.assertEqual(
            predict.resolve_qwen_batch_size(None, provided=False), predict.QWEN_DEFAULT_BATCH
        )
        self.assertEqual(predict.resolve_qwen_batch_size(6, provided=True), 6)

    def test_explicit_batch_clamped_to_8(self):
        self.assertEqual(predict.resolve_qwen_batch_size(12, provided=True), 8)

    def test_explicit_batch_clamped_to_1(self):
        self.assertEqual(predict.resolve_qwen_batch_size(0, provided=True), 1)
        self.assertEqual(predict.resolve_qwen_batch_size(-3, provided=True), 1)

    def test_default_batch_constant(self):
        self.assertEqual(predict.QWEN_DEFAULT_BATCH, 4)
        self.assertEqual(predict.QWEN_MAX_BATCH, 8)


class TestQwenContextHelpers(unittest.TestCase):
    def test_hotwords_become_context(self):
        ctx, truncated = predict.qwen_context_from_hotwords("Backblaze, Supabase")
        self.assertEqual(ctx, "Backblaze, Supabase")
        self.assertFalse(truncated)

    def test_hotwords_absent_neutral_context(self):
        ctx, truncated = predict.qwen_context_from_hotwords(None)
        self.assertEqual(ctx, "")
        self.assertFalse(truncated)

    def test_empty_hotwords_neutral_context(self):
        ctx, truncated = predict.qwen_context_from_hotwords("   ")
        self.assertEqual(ctx, "")
        self.assertFalse(truncated)

    def test_oversized_context_truncated_to_cap(self):
        long_hotwords = "x" * (predict.QWEN_CONTEXT_CAP + 500)
        with self.assertLogs("predict", level="WARNING") as logs:
            ctx, truncated = predict.qwen_context_from_hotwords(long_hotwords)
        self.assertEqual(len(ctx), predict.QWEN_CONTEXT_CAP)
        self.assertTrue(truncated)
        joined = "\n".join(logs.output)
        self.assertIn(str(predict.QWEN_CONTEXT_CAP + 500), joined)
        self.assertIn(str(predict.QWEN_CONTEXT_CAP), joined)
        # no hotword content in logs (lengths only)
        self.assertNotIn("xxxxx", joined)

    def test_context_cap_value(self):
        self.assertEqual(predict.QWEN_CONTEXT_CAP, 2000)


class TestEnableQwenKillSwitch(unittest.TestCase):
    def test_kill_switch_on_by_default(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertTrue(predict.qwen_enabled())

    def test_kill_switch_enabled_by_truthy_value(self):
        for val in ("1", "true", "TRUE", "yes", "on"):
            with mock.patch.dict(os.environ, {"ENABLE_QWEN": val}):
                self.assertTrue(predict.qwen_enabled())

    def test_kill_switch_disabled(self):
        for val in ("0", "false", "no", "off", ""):
            with mock.patch.dict(os.environ, {"ENABLE_QWEN": val}):
                with self.assertRaises(RuntimeError) as ctx:
                    predict.assert_qwen_enabled()
            self.assertIn("ENABLE_QWEN", str(ctx.exception))


class TestWhisperPathUnchanged(unittest.TestCase):
    def test_asr_options_hotwords_passthrough_whisper(self):
        """hotwords flows into faster-whisper asr_options on the whisper path."""
        opts = predict.build_asr_options(temperature=0.0, initial_prompt=None, hotwords="Backblaze")
        self.assertEqual(opts["hotwords"], "Backblaze")
        self.assertIsNone(opts["initial_prompt"])

    def test_asr_options_none_hotwords(self):
        opts = predict.build_asr_options(temperature=0.0, initial_prompt="p", hotwords=None)
        self.assertIsNone(opts["hotwords"])
        self.assertEqual(opts["initial_prompt"], "p")

    def test_asr_options_blank_hotwords_become_none(self):
        opts = predict.build_asr_options(temperature=0.0, initial_prompt=None, hotwords="  ")
        self.assertIsNone(opts["hotwords"])


class TestQwenLanguageHandling(unittest.TestCase):
    def test_qwen_skips_detect_language(self):
        """When model is qwen and language is None, no detect_language loop is run."""
        self.assertFalse(predict.should_detect_language("qwen3-asr", None))

    def test_qwen_provided_language_passthrough(self):
        self.assertEqual(predict.qwen_effective_language("qwen3-asr", "fr"), "fr")

    def test_qwen_no_language_returns_none(self):
        self.assertIsNone(predict.qwen_effective_language("qwen3-asr", None))

    def test_whisper_detects_language_when_unset(self):
        self.assertTrue(predict.should_detect_language("large-v3-turbo", None))

    def test_whisper_skips_detect_when_language_given(self):
        self.assertFalse(predict.should_detect_language("large-v3-turbo", "fr"))


class TestBakedQwenWeights(unittest.TestCase):
    def test_fail_fast_when_baked_weights_missing(self):
        with self.assertRaises(RuntimeError) as ctx:
            predict.assert_baked_qwen_weights("/models/nonexistent")
        self.assertIn("/models/nonexistent", str(ctx.exception))

    def test_pass_when_weight_present(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            for name in predict.QWEN_ASR_WEIGHT_FILES:
                (Path(tmp) / name).write_bytes(b"w")
            predict.assert_baked_qwen_weights(tmp)  # must not raise

    def test_fail_fast_when_one_weight_missing(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / predict.QWEN_ASR_WEIGHT_FILES[0]).write_bytes(b"w")
            with self.assertRaises(RuntimeError):
                predict.assert_baked_qwen_weights(tmp)

    def test_fail_fast_lists_expected_weight_names(self):
        with self.assertRaises(RuntimeError) as ctx:
            predict.assert_baked_qwen_weights("/models/empty")
        msg = str(ctx.exception)
        self.assertIn("safetensors", msg)


class TestQwenModelPaths(unittest.TestCase):
    def test_qwen_model_dir_resolution_baked_first(self):
        self.assertIn("/models/qwen3-asr-1.7b", predict.QWEN_MODEL_LOCAL_PATHS)

    def test_qwen_aligner_dir_resolution_baked_first(self):
        self.assertIn("/models/qwen3-forced-aligner-0.6b", predict.QWEN_ALIGNER_LOCAL_PATHS)


class TestBridgeQwenRouting(unittest.TestCase):
    def _parsed(self, model="qwen3-asr", **over):
        parsed = {
            "file_bytes": b"abc",
            "extension": "ogg",
            "whisper_model": model,
            "language": "fr",
            "prompt": None,
            "temperature": 0.0,
            "hotwords": "Backblaze, Supabase",
            "batch_size": None,
            "is_diarize": False,
            "known_speaker_names": [],
        }
        parsed.update(over)
        return parsed

    def test_hotwords_forwarded_on_qwen_path(self):
        cog_input = build_cog_input(self._parsed())
        self.assertEqual(cog_input["hotwords"], "Backblaze, Supabase")

    def test_hotwords_none_on_whisper_path(self):
        cog_input = build_cog_input(self._parsed(model="large-v3-turbo", whisper_model="large-v3-turbo"))
        self.assertIsNone(cog_input["hotwords"])

    def test_hotwords_none_when_absent(self):
        cog_input = build_cog_input(self._parsed(hotwords=None))
        self.assertIsNone(cog_input["hotwords"])

    def test_batch_size_omitted_when_not_provided(self):
        cog_input = build_cog_input(self._parsed())
        self.assertNotIn("batch_size", cog_input)

    def test_batch_size_passed_when_provided(self):
        cog_input = build_cog_input(self._parsed(batch_size=16))
        self.assertEqual(cog_input["batch_size"], 16)

    def test_batch_size_passed_on_whisper_path_too(self):
        cog_input = build_cog_input(
            self._parsed(model="large-v3-turbo", whisper_model="large-v3-turbo", batch_size=32)
        )
        self.assertEqual(cog_input["batch_size"], 32)

    def test_batch_size_omitted_on_whisper_path_when_absent(self):
        cog_input = build_cog_input(
            self._parsed(model="large-v3-turbo", whisper_model="large-v3-turbo")
        )
        self.assertNotIn("batch_size", cog_input)


if __name__ == "__main__":
    unittest.main()