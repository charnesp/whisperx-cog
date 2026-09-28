"""Reusable import helper for predict.py in GPU-free tests (E5-CODE-1 T1).

The RED/GREEN registry tests must import predict.py without torch / cog /
whisperx / ffmpeg. test_qwen_backend.py registers those stubs inline; this
helper factorizes the same stub install so both test files share it.
"""

from __future__ import annotations

import importlib
import sys
import types


def _pydantic_base_model():
    """Real pydantic BaseModel when importable, else a tiny kwarg stub."""
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


class _NoCtx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


class _Any:
    def __init__(self, *a, **k):
        pass


def install():
    """Register minimal stubs so predict.py imports without GPU deps.

    Idempotent: re-calling after predict is already imported returns the
    cached module (fresh install would reset module-level constants under
    test).
    """
    cached = sys.modules.get("predict")
    if cached is not None and getattr(cached, "_from_stub_install", False):
        return cached

    def make_module(name: str, **attrs):
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        sys.modules[name] = mod
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
        # BaseModel must be a real pydantic model: predict.Output subclasses it.
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
    make_module("whisperx.asr_qwen", load_model=lambda *a, **k: None)
    make_module(
        "whisperx.alignment_qwen",
        load_align_model=lambda *a, **k: (None, None),
        align=lambda *a, **k: None,
    )

    sys.modules.pop("predict", None)
    predict = importlib.import_module("predict")
    predict.__dict__["_from_predict_stub"] = True
    return predict
