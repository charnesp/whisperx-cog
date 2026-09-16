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
import io
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bridge"))

import openai_compat
from openai_compat import validate_transcription_request

MODEL_MAP = openai_compat.MODEL_MAP
build_cog_input = openai_compat.build_cog_input


def _encode_multipart(fields, files):
    """Encode a multipart/form-data body (same helper as test_openai_stt)."""
    boundary = "----testboundary"
    body = io.BytesIO()
    for name, value in fields:
        body.write(f"--{boundary}\r\n".encode())
        body.write(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        body.write(f"{value}\r\n".encode())
    for name, filename, content, content_type in files:
        body.write(f"--{boundary}\r\n".encode())
        body.write(
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode()
        )
        body.write(f"Content-Type: {content_type}\r\n\r\n".encode())
        body.write(content)
        body.write(b"\r\n")
    body.write(f"--{boundary}--\r\n".encode())
    return boundary, body.getvalue()


def _parse_multipart(body_bytes, content_type):
    fs, err = openai_compat.parse_multipart_form(
        {"Content-Type": content_type},
        io.BytesIO(body_bytes),
        len(body_bytes),
    )
    if err:
        raise AssertionError(err)
    return fs


class _NoCtx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


class _Any:
    def __init__(self, *a, **k):
        pass


def _pydantic_base_model():
    """Return a working BaseModel base for predict.Output.

    Prefer the real pydantic BaseModel when importable; otherwise use a tiny
    kwargs-recording stub so Output(...) still instantiates in the GPU-free
    test environment (predict.Output only needs attribute storage here).
    """
    try:
        from pydantic import BaseModel as _BaseModel

        return _BaseModel
    except Exception:
        pass

    class _StubBaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    return _StubBaseModel


def _install_predict_modules():
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
    make_module(
        "cog",
        BasePredictor=_BasePredictor,
        Input=_Any,
        Path=_Any,
        # BaseModel must be a real pydantic model: predict.Output subclasses it
        # with fields and is instantiated at the end of _run_predict.
        BaseModel=_pydantic_base_model(),
    )
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


_install_predict_modules()

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
        self.assertEqual(predict.resolve_qwen_batch_size(0, provided=True), 4)
        self.assertEqual(predict.resolve_qwen_batch_size(-3, provided=True), 4)

    def test_nonpositive_batch_logs_warning(self):
        with self.assertLogs("predict", level="WARNING") as logs:
            self.assertEqual(predict.resolve_qwen_batch_size(0, provided=True), 4)
        joined = "\n".join(logs.output)
        self.assertIn("0", joined)
        self.assertIn(str(predict.QWEN_DEFAULT_BATCH), joined)

    def test_clamp_above_max_logs_warning(self):
        with self.assertLogs("predict", level="WARNING") as logs:
            self.assertEqual(predict.resolve_qwen_batch_size(12, provided=True), 8)
        joined = "\n".join(logs.output)
        self.assertIn("12", joined)
        self.assertIn("8", joined)

    def test_in_range_batch_no_warning(self):
        # 6 is within [1, 8]: no clamp warning should fire.
        with mock.patch.object(predict.logger, "warning") as warn:
            self.assertEqual(predict.resolve_qwen_batch_size(6, provided=True), 6)
        warn.assert_not_called()

    def test_default_batch_constant(self):
        self.assertEqual(predict.QWEN_DEFAULT_BATCH, 4)
        self.assertEqual(predict.QWEN_MAX_BATCH, 8)


class TestQwenContextHelpers(unittest.TestCase):
    def test_hotwords_become_template_context(self):
        """FIX 3: hotwords wrapped in the design.md §2 context template."""
        ctx, truncated = predict.format_qwen_context("Backblaze, Supabase")
        self.assertIn("Backblaze, Supabase", ctx)
        self.assertIn("Contexte technique de la réunion", ctx)
        self.assertIn("Termes, entités et noms propres attendus", ctx)
        self.assertFalse(truncated)

    def test_hotwords_absent_neutral_context(self):
        ctx, truncated = predict.format_qwen_context(None)
        self.assertEqual(ctx, "")
        self.assertFalse(truncated)

    def test_empty_hotwords_neutral_context(self):
        ctx, truncated = predict.format_qwen_context("   ")
        self.assertEqual(ctx, "")
        self.assertFalse(truncated)

    def test_context_cap_value(self):
        self.assertEqual(predict.QWEN_CONTEXT_CAP, 2000)

    def test_oversized_context_truncated_to_cap(self):
        long_hotwords = "x" * (predict.QWEN_CONTEXT_CAP + 500)
        with self.assertLogs("predict", level="WARNING") as logs:
            ctx, truncated = predict.format_qwen_context(long_hotwords)
        self.assertEqual(len(ctx), predict.QWEN_CONTEXT_CAP)
        self.assertTrue(truncated)
        joined = "\n".join(logs.output)
        # lengths only: original (template-wrapped) and truncated lengths
        self.assertIn(str(len("Contexte technique de la réunion. Termes, entités et noms propres attendus : ") + predict.QWEN_CONTEXT_CAP + 500 + 1), joined)
        self.assertIn(str(predict.QWEN_CONTEXT_CAP), joined)
        # no hotword content in logs (lengths only)
        self.assertNotIn("xxxxx", joined)

    def test_context_cap_applies_after_template_assembly(self):
        # Hotwords just under the cap must still be truncated after the
        # template wrapper pushes the assembled context over 2000 chars.
        hotwords = "y" * (predict.QWEN_CONTEXT_CAP - 10)
        with self.assertLogs("predict", level="WARNING"):
            ctx, truncated = predict.format_qwen_context(hotwords)
        self.assertTrue(truncated)
        self.assertEqual(len(ctx), predict.QWEN_CONTEXT_CAP)


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


class TestWhisperPathBatchInvariant(unittest.TestCase):
    """FIX 1: whisper path without batch_size must call transcribe with 64."""

    def _run_predict_capture_transcribe(self, batch_size):
        """Drive _run_predict on the whisper path with a stubbed model, capture
        the batch_size passed to transcribe."""
        captured = {}
        model = types.SimpleNamespace(
            transcribe=lambda audio, **kw: captured.update(kw)
            or {"language": "en", "segments": []}
        )
        fake_result = {"language": "en", "segments": []}

        with mock.patch.object(
            predict.whisperx, "load_model", return_value=model
        ), mock.patch.object(
            predict.whisperx, "load_audio", return_value=[]
        ), mock.patch.object(
            predict, "get_audio_duration", return_value=1000.0
        ), mock.patch.object(
            predict, "align", side_effect=lambda *a, **k: fake_result
        ), mock.patch.object(
            predict, "diarize", side_effect=lambda *a, **k: fake_result
        ):
            predictor = predict.Predictor()
            predictor._run_predict(
                audio_file="clip.wav",
                whisper_model="large-v3-turbo",
                language="en",
                language_detection_min_prob=0,
                language_detection_max_tries=5,
                initial_prompt=None,
                hotwords=None,
                batch_size=batch_size,
                temperature=0,
                vad_onset=0.5,
                vad_offset=0.363,
                align_output=False,
                diarization=False,
                huggingface_access_token=None,
                min_speakers=None,
                max_speakers=None,
                debug=False,
            )
        return captured

    def test_whisper_without_batch_size_transcribes_with_64(self):
        captured = self._run_predict_capture_transcribe(None)
        self.assertEqual(captured["batch_size"], 64)

    def test_whisper_with_explicit_batch_size_passthrough(self):
        captured = self._run_predict_capture_transcribe(16)
        self.assertEqual(captured["batch_size"], 16)

    def test_whisper_default_batch_constant(self):
        self.assertEqual(predict.WHISPER_DEFAULT_BATCH, 64)

    def test_build_cog_input_whisper_no_batch_size_field(self):
        """Bridge: whisper request without batch_size omits the field entirely."""
        parsed = {
            "file_bytes": b"abc",
            "extension": "ogg",
            "whisper_model": "large-v3-turbo",
            "language": "en",
            "prompt": None,
            "temperature": 0.0,
            "hotwords": None,
            "batch_size": None,
            "is_diarize": False,
            "known_speaker_names": [],
        }
        cog_input = build_cog_input(parsed)
        self.assertNotIn("batch_size", cog_input)


class TestBridgeEnableQwenGate(unittest.TestCase):
    """FIX 2: ENABLE_QWEN kill-switch gated at the bridge (400, not 500)."""

    def _fs(self, model="qwen3-asr"):
        boundary, body = _encode_multipart(
            [("model", model)],
            [("file", "audio.ogg", b"abc", "audio/ogg")],
        )
        return _parse_multipart(body, f"multipart/form-data; boundary={boundary}")

    def test_kill_switch_off_returns_400(self):
        for value in ("0", "false", "False", ""):
            with mock.patch.dict(os.environ, {"ENABLE_QWEN": value}):
                parsed, err = validate_transcription_request(self._fs())
            self.assertIsNone(parsed)
            self.assertIsNotNone(err)
            status, payload = err
            self.assertEqual(status, 400)
            self.assertEqual(payload["error"]["type"], "invalid_request_error")
            self.assertIn("ENABLE_QWEN", payload["error"]["message"])

    def test_kill_switch_unset_defaults_to_enabled(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            parsed, err = validate_transcription_request(self._fs())
        self.assertIsNone(err)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["whisper_model"], "qwen3-asr")

    def test_kill_switch_on_accepts_qwen_model(self):
        for value in ("1", "true", "TRUE", "yes", "on"):
            with mock.patch.dict(os.environ, {"ENABLE_QWEN": value}):
                parsed, err = validate_transcription_request(self._fs())
            self.assertIsNone(err)
            self.assertEqual(parsed["whisper_model"], "qwen3-asr")

    def test_kill_switch_does_not_block_whisper_models(self):
        with mock.patch.dict(os.environ, {"ENABLE_QWEN": "0"}):
            parsed, err = validate_transcription_request(self._fs(model="whisper-1"))
        self.assertIsNone(err)
        self.assertEqual(parsed["whisper_model"], "large-v3-turbo")


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
