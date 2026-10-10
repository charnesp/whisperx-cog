"""GPU-free RED/GREEN tests for the predict.py remote branching (group 6).

Change: openspec/changes/qwen3-asr-remote-vllm tasks.md 6.1.
Seam under test: ``predict.Predictor._run_predict`` on the qwen path.

Requirements pinned here:
- ``QWEN_BACKEND=remote`` routes the per-window transcription through
  ``qwen_remote_client.transcribe_windows`` and the local ASR loader
  (``whisperx.asr_qwen.load_model``) is NEVER called (call-count double);
- ``QWEN_BACKEND=local``/unset keeps the current in-process path bit-for-bit
  (``asr_qwen.load_model`` -> ``model.transcribe(batch_size, context)``);
- a remote failure surfaces as the typed ``QwenRemoteError`` with no local
  fallback attempt;
- a broken remote config fails fast in ``Predictor.setup()`` (before boot);
- ``align_qwen`` + ``diarize`` run identically in both modes;
- remote segments carry the LOCAL VAD start/end (the server never timestamps);
- ``ENABLE_QWEN`` stays a single checkpoint (bridge-owned); the cog remote
  branch adds no duplicate gate.

The model layer is mocked (torch/cog/whisperx are stub modules installed by
tests/_predict_stub.py); no GPU, no real weights.
"""
from __future__ import annotations

import contextlib
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from _predict_stub import install  # noqa: E402

from qwen_remote import QwenRemoteError  # noqa: E402

predict = install()

REMOTE_CONFIG = {
    "backend": "remote",
    "base_url": "https://vllm.internal.invalid:9000/v1",
    "chat_url": "https://vllm.internal.invalid:9000/v1/chat/completions",
    "model": "qwen-vllm-test-model",
    "timeout_s": 30,
}
LOCAL_CONFIG = {
    "backend": "local",
    "base_url": None,
    "model": None,
    "timeout_s": None,
    "chat_url": None,
}

HOTWORDS = "WhisperX, PyAnnote"
EXPECTED_CONTEXT = predict.format_qwen_context(HOTWORDS)[0]

_FAKE_AUDIO = [0.0] * 160000  # 10 s at 16 kHz

# Remote path VAD windows: (start_s, end_s, samples). Server text must never
# change these timestamps.
WINDOWS = [(0.0, 2.0, [0.1, 0.2]), (2.5, 4.0, [0.3, 0.4])]
WINDOW_TEXTS = ["bonjour", "le monde"]
WINDOW_LANGS = ["fr", "fr"]  # server-detected language per window (VAD order)
EXPECTED_SEGMENTS = [
    {"text": "bonjour", "start": 0.0, "end": 2.0},
    {"text": "le monde", "start": 2.5, "end": 4.0},
]

def _asr_qwen_stub():
    """Resolve the asr_qwen stub module at CALL time.

    Other test modules re-register ``sys.modules["whisperx.asr_qwen"]`` during
    collection; predict.py imports it lazily, so the live sys.modules entry is
    the one predict actually sees. A module-level capture goes stale under
    ``unittest discover`` and silently misses the patch.
    """
    return sys.modules["whisperx.asr_qwen"]

QWEN_ARGS = dict(
    audio_file="clip.wav",
    whisper_model="qwen3-asr",
    language="fr",
    language_detection_min_prob=0,
    language_detection_max_tries=5,
    initial_prompt=None,
    hotwords=HOTWORDS,
    batch_size=None,
    temperature=0,
    vad_onset=0.5,
    vad_offset=0.363,
    align_output=True,
    diarization=True,
    huggingface_access_token="tok",
    min_speakers=None,
    max_speakers=None,
    debug=False,
)


def _local_result():
    return {"language": "fr", "segments": [{"text": "local", "start": 0.0, "end": 1.0}]}


class _RemoteBranchBase(unittest.TestCase):
    def _predictor(self, config):
        predictor = predict.Predictor()
        # setup() resolves the config once (fail-fast); here we inject the
        # resolved dict directly to unit-test the branching itself.
        predictor._qwen_remote_config = config
        return predictor

    def _common_patches(self):
        """Patches shared by every branch test (model layer fully mocked)."""
        return [
            mock.patch.object(predict.whisperx, "load_audio", return_value=_FAKE_AUDIO),
            mock.patch.object(predict, "get_audio_duration", return_value=4000.0),
            mock.patch.object(predict, "require_diarization_token", return_value="tok"),
            mock.patch.object(
                predict, "resolve_qwen_snapshot_dir", return_value="/models/qwen3-asr"
            ),
        ]

    def _stack(self, patches):
        stack = contextlib.ExitStack()
        for p in patches:
            stack.enter_context(p)
        self.addCleanup(stack.close)
        return stack


class TestRemoteRouting(_RemoteBranchBase):
    """Req 1 + 6: remote transcription via the client, local loader untouched,
    segments carry the LOCAL VAD start/end."""

    def test_remote_routes_windows_through_client_and_skips_local_loader(self):
        fake_model = types.SimpleNamespace(
            transcribe=lambda *a, **k: _local_result()
        )
        with mock.patch.object(_asr_qwen_stub(), "load_model", return_value=fake_model) as load_model, \
                mock.patch.object(predict, "align_qwen", side_effect=lambda a, r, d: r), \
                mock.patch.object(predict, "diarize", side_effect=lambda *a, **k: a[1]), \
                mock.patch.object(predict, "qwen_remote_windows", return_value=list(WINDOWS)), \
                mock.patch.object(
                    predict.qwen_remote_client, "transcribe_windows_meta",
                    return_value=(list(WINDOW_TEXTS), list(WINDOW_LANGS)),
                ) as tw:
            self._stack(self._common_patches())
            predictor = self._predictor(REMOTE_CONFIG)
            output = predictor._run_predict(**QWEN_ARGS)

        # (1) local ASR loader NEVER invoked in remote mode
        load_model.assert_not_called()
        # (1) windows go through qwen_remote_client in VAD order
        tw.assert_called_once()
        call = tw.call_args
        self.assertEqual(call.args[0], [[0.1, 0.2], [0.3, 0.4]])
        self.assertEqual(call.args[1], REMOTE_CONFIG)
        self.assertEqual(call.kwargs["batch_size"], 4)
        self.assertEqual(call.kwargs["language"], "fr")
        self.assertEqual(call.kwargs["context"], EXPECTED_CONTEXT)
        # (6) segments carry the LOCAL VAD timestamps, not the reply text
        self.assertEqual(list(output.segments), EXPECTED_SEGMENTS)

    def test_remote_uses_detected_language_when_language_none(self):
        """language=None on the remote path must surface the DETECTED language.

        Review finding P1-C: predict.py forced 'en' (`language or "en"`) and
        made the aligner load the wrong model on FR audio.
        """
        args = dict(QWEN_ARGS)
        args["language"] = None
        fake_model = types.SimpleNamespace(transcribe=lambda *a, **k: _local_result())
        with mock.patch.object(_asr_qwen_stub(), "load_model", return_value=fake_model), \
                mock.patch.object(predict, "align_qwen", side_effect=lambda a, r, d: r), \
                mock.patch.object(predict, "diarize", side_effect=lambda *a, **k: a[1]), \
                mock.patch.object(predict, "qwen_remote_windows", return_value=list(WINDOWS)), \
                mock.patch.object(
                    predict.qwen_remote_client, "transcribe_windows_meta",
                    return_value=(list(WINDOW_TEXTS), ["fr", "fr"]),
                ) as tw:
            self._stack(self._common_patches())
            predictor = self._predictor(REMOTE_CONFIG)
            output = predictor._run_predict(**args)
        self.assertEqual(output.detected_language, "fr")
        self.assertIsNone(tw.call_args.kwargs["language"])  # None still forwarded

    def test_remote_falls_back_to_en_only_without_any_detected_language(self):
        args = dict(QWEN_ARGS)
        args["language"] = None
        fake_model = types.SimpleNamespace(transcribe=lambda *a, **k: _local_result())
        with mock.patch.object(_asr_qwen_stub(), "load_model", return_value=fake_model), \
                mock.patch.object(predict, "align_qwen", side_effect=lambda a, r, d: r), \
                mock.patch.object(predict, "diarize", side_effect=lambda *a, **k: a[1]), \
                mock.patch.object(predict, "qwen_remote_windows", return_value=list(WINDOWS)), \
                mock.patch.object(
                    predict.qwen_remote_client, "transcribe_windows_meta",
                    return_value=(list(WINDOW_TEXTS), [None, None]),
                ):
            self._stack(self._common_patches())
            predictor = self._predictor(REMOTE_CONFIG)
            output = predictor._run_predict(**args)
        self.assertEqual(output.detected_language, "en")

    def test_remote_error_propagates_typed_without_local_fallback(self):
        err = QwenRemoteError("QwenRemoteError: connection - cannot reach the remote engine")
        fake_model = types.SimpleNamespace(transcribe=lambda *a, **k: _local_result())
        with mock.patch.object(_asr_qwen_stub(), "load_model", return_value=fake_model) as load_model, \
                mock.patch.object(predict, "align_qwen", side_effect=lambda a, r, d: r), \
                mock.patch.object(predict, "diarize", side_effect=lambda *a, **k: a[1]), \
                mock.patch.object(predict, "qwen_remote_windows", return_value=list(WINDOWS)), \
                mock.patch.object(
                    predict.qwen_remote_client, "transcribe_windows_meta", side_effect=err
                ) as tw:
            self._stack(self._common_patches())
            predictor = self._predictor(REMOTE_CONFIG)
            with self.assertRaises(QwenRemoteError) as ctx:
                predictor._run_predict(**QWEN_ARGS)

        self.assertIn("QwenRemoteError", str(ctx.exception))
        tw.assert_called_once()
        # (3) NO local fallback: the ASR loader is still never called
        load_model.assert_not_called()


class TestLocalPathIdentity(_RemoteBranchBase):
    """Req 2: local/unset keeps the untouched in-process path."""

    def test_local_backend_keeps_current_inprocess_path(self):
        captured = {}

        def _transcribe(audio, **kw):
            captured.update(kw)
            return _local_result()

        fake_model = types.SimpleNamespace(transcribe=_transcribe)
        with mock.patch.object(_asr_qwen_stub(), "load_model", return_value=fake_model) as load_model, \
                mock.patch.object(predict, "align_qwen", side_effect=lambda a, r, d: r), \
                mock.patch.object(predict, "diarize", side_effect=lambda *a, **k: a[1]), \
                mock.patch.object(predict.qwen_remote_client, "transcribe_windows_meta") as tw:
            self._stack(self._common_patches())
            predictor = self._predictor(LOCAL_CONFIG)
            output = predictor._run_predict(**QWEN_ARGS)

        load_model.assert_called_once()
        self.assertEqual(captured["batch_size"], 4)
        self.assertEqual(captured["context"], EXPECTED_CONTEXT)
        # remote client never touched on the local path
        tw.assert_not_called()
        self.assertEqual(list(output.segments), _local_result()["segments"])


class TestSetupFailFast(_RemoteBranchBase):
    """Req 4: broken remote config aborts setup() before serving."""

    def test_setup_fails_fast_on_broken_remote_config(self):
        with mock.patch.dict(os.environ, {"QWEN_BACKEND": "remote"}, clear=True), \
                mock.patch.object(predict, "resolve_vad_source_path", return_value=None), \
                mock.patch("predict.os.makedirs"):
            predictor = predict.Predictor()
            with self.assertRaises(QwenRemoteError) as ctx:
                predictor.setup()
        message = str(ctx.exception)
        self.assertIn("QwenRemoteError", message)
        self.assertIn("config", message)
        self.assertIn("QWEN_REMOTE_BASE_URL", message)

    def test_setup_ok_with_complete_remote_config(self):
        env = {
            "QWEN_BACKEND": "remote",
            "QWEN_REMOTE_BASE_URL": "https://vllm.internal.invalid:9000/v1",
            "QWEN_REMOTE_MODEL": "m",
        }
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(predict, "resolve_vad_source_path", return_value=None), \
                mock.patch("predict.os.makedirs"):
            predictor = predict.Predictor()
            predictor.setup()
        self.assertEqual(predictor._qwen_remote_config["backend"], "remote")


class TestAlignmentAndDiarizationParity(_RemoteBranchBase):
    """Req 5: align_qwen + diarize called identically in both modes."""

    def _capture(self, config):
        align = mock.Mock(side_effect=lambda a, r, d: r)
        diar = mock.Mock(side_effect=lambda *a, **k: a[1])
        fake_model = types.SimpleNamespace(transcribe=lambda *a, **k: _local_result())
        with mock.patch.object(_asr_qwen_stub(), "load_model", return_value=fake_model), \
                mock.patch.object(predict, "align_qwen", align), \
                mock.patch.object(predict, "diarize", diar), \
                mock.patch.object(predict, "qwen_remote_windows", return_value=list(WINDOWS)), \
                mock.patch.object(
                    predict.qwen_remote_client, "transcribe_windows_meta",
                    return_value=(list(WINDOW_TEXTS), list(WINDOW_LANGS)),
                ):
            self._stack(self._common_patches())
            predictor = self._predictor(config)
            predictor._run_predict(**QWEN_ARGS)
        return align, diar

    def test_align_and_diarize_called_identically_in_both_modes(self):
        align_local, diar_local = self._capture(LOCAL_CONFIG)
        align_remote, diar_remote = self._capture(REMOTE_CONFIG)

        for align in (align_local, align_remote):
            align.assert_called_once()
            args = align.call_args.args
            self.assertIs(args[0], _FAKE_AUDIO)
            self.assertFalse(args[2])
        for diar in (diar_local, diar_remote):
            diar.assert_called_once()
            args = diar.call_args.args
            self.assertIs(args[0], _FAKE_AUDIO)
            self.assertFalse(args[2])
            self.assertEqual(args[3], "tok")


class TestEnableQwenStaysBridgeOnly(_RemoteBranchBase):
    """Req 7: ENABLE_QWEN is a single checkpoint; the remote branch adds none."""

    def test_remote_branch_does_not_duplicate_the_gate(self):
        gate = mock.Mock()
        fake_model = types.SimpleNamespace(
            transcribe=lambda *a, **k: _local_result()
        )
        with mock.patch.object(predict, "assert_qwen_enabled", gate), \
                mock.patch.object(_asr_qwen_stub(), "load_model", return_value=fake_model), \
                mock.patch.object(predict, "align_qwen", side_effect=lambda a, r, d: r), \
                mock.patch.object(predict, "diarize", side_effect=lambda *a, **k: a[1]), \
                mock.patch.object(predict, "qwen_remote_windows", return_value=list(WINDOWS)), \
                mock.patch.object(
                    predict.qwen_remote_client, "transcribe_windows_meta",
                    return_value=(list(WINDOW_TEXTS), list(WINDOW_LANGS)),
                ):
            self._stack(self._common_patches())
            predictor = self._predictor(REMOTE_CONFIG)
            predictor._run_predict(**QWEN_ARGS)
        # exactly ONE gate checkpoint, whatever the backend
        self.assertEqual(gate.call_count, 1)

    def test_remote_does_not_bypass_kill_switch(self):
        with mock.patch.dict(os.environ, {"ENABLE_QWEN": "0"}, clear=False), \
                mock.patch.object(predict, "qwen_remote_windows") as windows, \
                mock.patch.object(predict.qwen_remote_client, "transcribe_windows_meta") as tw, \
                mock.patch.object(_asr_qwen_stub(), "load_model") as load_model:
            self._stack(self._common_patches())
            predictor = self._predictor(REMOTE_CONFIG)
            with self.assertRaises(RuntimeError) as ctx:
                predictor._run_predict(**QWEN_ARGS)
        self.assertIn("ENABLE_QWEN", str(ctx.exception))
        windows.assert_not_called()
        tw.assert_not_called()
        load_model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
