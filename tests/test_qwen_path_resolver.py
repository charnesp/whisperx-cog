"""E5-QWEN-PATH-FIX — GPU-free regression tests for the qwen path resolution.

Bug (reproduced locally + prod canary): _run_predict called
resolve_whisper_model_path(whisper_model) unconditionally, even on the qwen
branch. That resolver only knows the faster-whisper keys (ENV_OVERRIDES /
WHISPER_MODEL_HF_IDS have no 'qwen3-asr-1.7b' entry) so the qwen path crashed
with KeyError 'qwen3-asr-1.7b' before reaching its own snapshot resolution
(resolve_qwen_snapshot_dir).

Contract under test:
- the qwen branch resolves the baked snapshot WITHOUT touching
  resolve_whisper_model_path (no KeyError),
- the whisper branch still resolves through resolve_whisper_model_path,
- detect_language is never reached on qwen (should_detect_language skip),
- scripts/golden_set.py keeps the same qwen/whisper split.
"""

from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

from _predict_stub import install  # noqa: E402

predict = install()


class TestQwenPathSkipsWhisperResolver(unittest.TestCase):
    """The qwen branch of _run_predict must NOT go through the faster-whisper
    resolver (ENV_OVERRIDES/WHISPER_MODEL_HF_IDS lack the qwen key)."""

    def _run_qwen_predict(self, resolver_calls=None):
        """Drive _run_predict on the qwen path with stubbed model + snapshot.

        resolve_whisper_model_path is instrumented: every call is recorded and
        it re-raises the exact prod KeyError (the qwen path must never reach
        it). Returns (captured, snapshot_dir_used).
        """
        captured = {}
        result_qwen = {"language": "fr", "segments": []}
        model = types.SimpleNamespace(
            transcribe=lambda audio, **kw: captured.update(kw) or result_qwen
        )
        snapshot_dir_holder = {}

        def _boom(_model):
            if resolver_calls is not None:
                resolver_calls.append(_model)
            raise KeyError("qwen3-asr-1.7b")

        with tempfile.TemporaryDirectory() as tmp:
            for name in predict.QWEN_ASR_WEIGHT_FILES:
                (Path(tmp) / name).write_bytes(b"w")
            stub_asr_qwen = types.SimpleNamespace(load_model=lambda *a, **k: model)
            with mock.patch.object(
                predict, "resolve_whisper_model_path", side_effect=_boom
            ), mock.patch.object(
                predict,
                "resolve_qwen_snapshot_dir",
                side_effect=lambda *a: snapshot_dir_holder.setdefault("dir", tmp),
            ), mock.patch.dict(
                sys.modules, {"whisperx.asr_qwen": stub_asr_qwen}
            ), mock.patch.object(
                predict, "get_audio_duration", return_value=1000.0
            ):
                predictor = predict.Predictor()
                predictor._run_predict(
                    audio_file="clip.wav",
                    whisper_model="qwen3-asr",
                    language="fr",
                    language_detection_min_prob=0,
                    language_detection_max_tries=5,
                    initial_prompt=None,
                    hotwords=None,
                    batch_size=None,
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
        return captured, snapshot_dir_holder["dir"]

    def test_qwen_path_resolves_without_whisper_resolver(self):
        """RED: qwen path must resolve the baked snapshot without touching the
        faster-whisper resolver (no KeyError 'qwen3-asr-1.7b')."""
        captured, snapshot_dir = self._run_qwen_predict()
        self.assertEqual(captured["batch_size"], predict.QWEN_DEFAULT_BATCH)
        self.assertTrue(snapshot_dir)

    def test_qwen_path_never_calls_resolve_whisper_model_path(self):
        """The whisper resolver must not be called at all on the qwen path."""
        calls = []
        self._run_qwen_predict(resolver_calls=calls)
        self.assertEqual(calls, [])

    def test_detect_language_skipped_for_qwen(self):
        """should_detect_language already skips the detect loop on qwen."""
        self.assertFalse(predict.should_detect_language("qwen3-asr", None))
        self.assertTrue(predict.should_detect_language("large-v3-turbo", None))


class TestWhisperPathStillResolves(unittest.TestCase):
    """The fix must not break the faster-whisper resolution path."""

    def test_whisper_path_still_calls_resolver(self):
        """whisper path: resolve_whisper_model_path is called and its result
        reaches whisperx.load_model."""
        captured = {}
        result = {"language": "en", "segments": []}
        model = types.SimpleNamespace(
            transcribe=lambda audio, **kw: captured.update(kw) or result
        )
        resolver_args = []
        with mock.patch.object(
            predict.whisperx, "load_model", return_value=model
        ) as load_model, mock.patch.object(
            predict.whisperx, "load_audio", return_value=[]
        ), mock.patch.object(
            predict, "resolve_whisper_model_path", side_effect=resolver_args.append
        ), mock.patch.object(
            predict, "get_audio_duration", return_value=1000.0
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
                batch_size=None,
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
        self.assertEqual(resolver_args, ["large-v3-turbo"])
        self.assertEqual(load_model.call_args.kwargs.get("language"), "en")
        self.assertEqual(captured["batch_size"], 64)


class TestCallSitesSplit(unittest.TestCase):
    """Other resolve_whisper_model_path call sites must stay whisper-only."""

    def test_predict_source_guards_resolver_with_is_qwen(self):
        """predict.py must resolve whisper_arch under the non-qwen branch."""
        src = (Path(__file__).resolve().parent.parent / "predict.py").read_text()
        self.assertIn("if not is_qwen:", src)
        self.assertIn("whisper_arch = resolve_whisper_model_path(whisper_model)", src)
        # No unconditional call left.
        self.assertNotIn(
            "\n            whisper_arch = resolve_whisper_model_path(whisper_model)\n",
            src,
        )

    def test_golden_set_qwen_branch_avoids_whisper_resolver(self):
        """scripts/golden_set.py model_factory_real must not resolve qwen via
        resolve_whisper_model_path (same bug class)."""
        src = (
            Path(__file__).resolve().parent.parent / "scripts" / "golden_set.py"
        ).read_text()
        _, qwen_branch, whisper_tail = src.split('model_name == "qwen3-asr"')
        self.assertNotIn("resolve_whisper_model_path", qwen_branch)
        self.assertIn("resolve_whisper_model_path", whisper_tail)


if __name__ == "__main__":
    unittest.main()
