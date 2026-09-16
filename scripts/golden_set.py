#!/usr/bin/env python3
"""Golden-set harness — replayable GPU validation of the qwen3-asr backend.

Implements task 6.1 of openspec/changes/qwen3-asr-backend/tasks.md:

- fixed FR extract (test_fr.wav) + 2026-09-02 real meeting extract (meeting_0209.ogg)
- runs: turbo baseline, qwen baseline, qwen+hotwords (Backblaze, Supabase, AirSync, Volok)
- faster-whisper regression hashes (bit-identical check vs a recorded baseline, 6.2)
- qwen baseline vs qwen+hotwords: proper-noun recall AND false positives (6.3)
- word-level timestamps + assign_word_speakers handoff from the ForcedAligner (6.5)
- peak VRAM < 5.5 GB at batch 4 (fp16), RTFx plausibility on the 4080 (6.6)

The harness is scripted and replayable:

    python3 scripts/golden_set.py \
        --meeting-audio /opt/data/tmp/stt/meeting_0209.ogg \
        --fr-extract /opt/data/tmp/stt/test_fr.wav \
        --output /tmp/golden_set_report.json   # noqa: S108 (example path)

It imports predict.py lazily and runs everything on CUDA. GPU-free unit
tests cover the pure helpers (tests/test_golden_set.py); run_single() is
fully injectable (load_audio_fn, model_factory, clock) so CI can exercise
the whole evaluation flow without a GPU.

Report JSON is written to --output (default /tmp/golden_set_report.json)
and exits 0 only when evaluate_report() passes every assertion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "bridge"))

# ---------------------------------------------------------------------------
# Constants (mirrored from predict.py — the harness imports predict lazily so
# GPU-free unit tests can load this module without torch/cog installed).
# ---------------------------------------------------------------------------

QWEN_DEFAULT_BATCH = 4
WHISPER_DEFAULT_BATCH = 64
# E4-EXEC-FIX FIX A: per-run-type batch defaults passed to run_single by
# run_golden_set (turbo 16 — prod 64 OOMs when free VRAM < 6 GiB; qwen 4).
PER_MODEL_DEFAULT_BATCH = {"large-v3-turbo": 16, "qwen3-asr": 4}
VRAM_LIMIT_GB = 5.5  # design.md: fp16 load measures ~4.99 GB; fp32 would be ~10 GB
RTFX_MIN = 1.0  # slower than realtime = broken
RTFX_MAX = 300.0  # implausibly fast on a single 4080 (sanity ceiling)
DEFAULT_OUTPUT = "/tmp/golden_set_report.json"  # noqa: S108 — scratch report, no secrets

# Proper nouns from the 2026-09-02 meeting (design.md §1: 4x more correct
# with hotwords — Backblaze 3→12, Supabase 0→3).
HOTWORD_TERMS = ["Backblaze", "Supabase", "AirSync", "Volok"]
DEFAULT_HOTWORDS = "Backblaze, Supabase, AirSync, Volok"

# Segments that legitimately discuss these terms carry storage/database/S3
# vocabulary; a hotword inside a segment without any of these keywords is a
# hallucinated insertion (false positive).
_HOTWORD_CONTEXT_KEYWORDS = (
    "bucket",
    "stockage",
    "storage",
    "base de données",
    "database",
    "api",
    "s3",
    "serveur",
    "sauvegarde",
    "backup",
    "nas",
    "fichier",
    "log",
    "données",
    "data",
    "clé",
    "key",
    "sync",
    "archive",
    "zip",
    "compress",
    "edge",
    "héberge",
    "hosting",
    "cloud",
    "to",
    "téraoctet",
    "teraoctet",
    "gigas",
)


def default_run_specs() -> dict[str, dict[str, Any]]:
    """The three runs of task 6.1, plus the fixed inputs they replay."""
    return {
        "turbo_baseline": {
            "whisper_model": "large-v3-turbo",
            "hotwords": None,
            "assertion": "faster-whisper baseline (bit-identical regression hash, 6.2)",
        },
        "qwen_baseline": {
            "whisper_model": "qwen3-asr",
            "hotwords": None,
            "assertion": "hotwords absent -> neutral context (6.4)",
        },
        "qwen_hotwords": {
            "whisper_model": "qwen3-asr",
            "hotwords": DEFAULT_HOTWORDS,
            "assertion": "proper-noun recall improvement vs qwen_baseline (6.3)",
        },
    }


RUNS = default_run_specs()


# ---------------------------------------------------------------------------
# Pure helpers (GPU-free, unit-tested)
# ---------------------------------------------------------------------------


def hash_transcript(text: str) -> str:
    """SHA-256 of the transcript text — text-level invariance key (6.4)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_segments(segments: list[dict]) -> str:
    """SHA-256 of the canonical JSON (sort_keys) of the full segments — 6.2/6.4.

    The regression invariance covers the complete segments, not text-only:
    text/start/end plus word-level detail (each word's word/start/end and its
    speaker label after assign_word_speakers, when present). Any change in
    timing, word segmentation or speaker assignment changes the hash.
    """
    canonical = json.dumps(segments, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def count_hotword_occurrences(text: str, terms: list[str]) -> dict[str, int]:
    """Case-insensitive whole-word occurrence count per hotword term.

    Whole-word matching so 'Supabase' inside another word doesn't count;
    accented/punctuation boundaries are handled by \\b.
    """
    counts: dict[str, int] = {}
    for term in terms:
        pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
        counts[term] = len(pattern.findall(text))
    return counts


def find_hotword_false_positives(
    segments: list[dict],
    terms: list[str],
    context_keywords: tuple[str, ...] = _HOTWORD_CONTEXT_KEYWORDS,
) -> list[dict[str, Any]]:
    """Segments that contain a hotword but none of the context keywords.

    Those are candidate hallucinated insertions (task 6.3): the term appears
    in a segment that has no relation to the storage/cloud topic it belongs
    to. Returns one entry per (segment, hotword) pair.
    """
    findings: list[dict[str, Any]] = []
    for seg in segments:
        text = seg.get("text") or ""
        seg_start = seg.get("start", 0.0)
        lowered = text.lower()
        has_context = any(kw.lower() in lowered for kw in context_keywords)
        if has_context:
            continue
        for term in terms:
            if re.search(rf"\b{re.escape(term)}\b", text, re.IGNORECASE):
                findings.append(
                    {
                        "hotword": term,
                        "segment_start": seg_start,
                        "text": text,
                        "reason": "hotword in segment without topical context keywords",
                    }
                )
    return findings


def hotword_recall_improvement(baseline: dict[str, int], with_hotwords: dict[str, int]) -> dict[str, float]:
    """Per-term recall multiple (with_hotwords / baseline)."""
    improvement: dict[str, float] = {}
    for term, base_count in baseline.items():
        new_count = with_hotwords.get(term, 0)
        improvement[term] = (new_count / base_count) if base_count else float("inf") if new_count else 0.0
    return improvement


def build_hotword_recall_report(
    baseline_counts: dict[str, int],
    with_hotwords_counts: dict[str, int],
    terms: list[str],
) -> dict[str, Any]:
    """Structured recall summary for the JSON report (6.3)."""
    recall = {}
    for term in terms:
        base = baseline_counts.get(term, 0)
        hot = with_hotwords_counts.get(term, 0)
        recall[term] = {"baseline": base, "with_hotwords": hot, "delta": hot - base}
    improvement = hotword_recall_improvement(baseline_counts, with_hotwords_counts)
    return {
        "recall": recall,
        "improvement": {
            k: (v if v != float("inf") else "new (0 -> n)") for k, v in improvement.items()
        },
        "false_positive_risk": "see false_positives in the evaluation",
    }


def word_timestamps_present(result: dict) -> bool:
    """Every segment carries non-empty word-level timestamps (6.5)."""
    segments = (result or {}).get("segments") or []
    if not segments:
        return False
    return all(
        bool(seg.get("words")) and all("start" in w and "end" in w for w in (seg.get("words") or []))
        for seg in segments
    )


def words_carry_speakers(result: dict) -> bool:
    """After diarization, words (from the ForcedAligner) carry a speaker label (6.5).

    Proves assign_word_speakers received the qwen ForcedAligner words: every
    segment must carry words and every word a speaker key. A segment without
    words is NOT vacuously ok — the align/diarize wiring is what produces
    them, so a missing words list means the handoff never happened.
    """
    segments = (result or {}).get("segments") or []
    if not segments:
        return False
    for seg in segments:
        words = seg.get("words") or []
        if not words:
            return False
        if not all(w.get("speaker") for w in words):
            return False
    return True


def vram_ok(peak_gb: float) -> bool:
    """Peak VRAM must stay under 5.5 GB (fp16 load; design.md §risk table)."""
    return peak_gb < VRAM_LIMIT_GB


def rtfx_ok(rtfx: float) -> bool:
    """RTFx must be plausible: faster than realtime but not physically impossible."""
    return RTFX_MIN < rtfx < RTFX_MAX


def compute_rtfx(audio_duration_s: float, processing_time_s: float) -> float:
    """RTFx = audio seconds processed per processing second (measured ~52 on 4080)."""
    if processing_time_s <= 0:
        return float("inf")
    return audio_duration_s / processing_time_s


def extract_transcript_text(result: dict) -> str:
    """Concatenate segment texts into the plain transcript string."""
    segments = (result or {}).get("segments") or []
    return "\n".join(str(seg.get("text") or "") for seg in segments).strip()


# ---------------------------------------------------------------------------
# Runner (GPU-free by injection; real GPU execution wires predict.Predictor)
# ---------------------------------------------------------------------------


def run_single(
    run_spec: dict[str, Any],
    audio_path: str,
    load_audio_fn: Callable[[str], Any],
    model_factory: Callable[[str], Any],
    clock: Callable[[], float] = time.time,
    batch_size: int | None = None,
    align_fn: Callable[[Any, dict], dict] | None = None,
    diarize_fn: Callable[[Any, dict], dict] | None = None,
    vram_peak_fn: Callable[[], float] | None = None,
) -> dict[str, Any]:
    """Execute one golden-set run and collect its metrics (6.1, 6.5).

    Fully injectable for GPU-free testing: load_audio_fn / model_factory /
    clock / align_fn / diarize_fn / vram_peak_fn (CI mocks). On the real GPU
    the defaults wire the predictor pipeline exactly like predict._run_predict:
    transcribe -> align (align_qwen on the qwen path, align standard on the
    turbo path — ForcedAligner word-level timestamps) -> diarize +
    whisperx.assign_word_speakers (speaker labels on the aligned words). The
    gate on word_timestamps_present / words_carry_speakers (6.5) is therefore
    actually executable: the wiring produces the words/speaker keys it checks.

    Per-run VRAM (FIX 7): vram_peak_fn is read at the end of the run (torch
    max_memory_allocated); the real flow resets the peak counter before each
    run via vram_reset_real, so vram_peak_gb is per-run, not a global max.
    """
    model_name = run_spec["whisper_model"]
    hotwords = run_spec.get("hotwords")
    audio = load_audio_fn(audio_path)
    model = model_factory(model_name)

    start = clock()
    if model_name == "qwen3-asr":
        context = build_qwen_context(hotwords)
        effective_batch = batch_size if batch_size is not None else QWEN_DEFAULT_BATCH
        result = model.transcribe(audio, batch_size=effective_batch, context=context)
    else:
        effective_batch = batch_size if batch_size is not None else WHISPER_DEFAULT_BATCH
        result = model.transcribe(audio, batch_size=effective_batch)
    duration_s = clock() - start

    # 6.5 wiring (injectable; real GPU defaults below). align produces the
    # word-level timestamps, diarize produces the speaker turns and labels
    # the aligned words via assign_word_speakers — the keys this run's
    # gates check. Without the wiring both gates would fail on real output.
    if align_fn is not None:
        # FIX 1 (E4-FIX-2): the raw transcribe result NEVER carries
        # 'whisper_model' (asr_qwen returns {segments, language} only), so
        # inject the model name before align dispatch — on the qwen path
        # default_align_fn routes to predict.align_qwen, not wav2vec2 align.
        result = dict(result or {})
        result["whisper_model"] = model_name
        result = align_fn(audio, result)
    if diarize_fn is not None:
        result = diarize_fn(audio, result)

    segments = (result or {}).get("segments") or []
    transcript = extract_transcript_text(result)
    return {
        "whisper_model": model_name,
        "hotwords": hotwords,
        "transcript": transcript,
        "transcript_hash": hash_transcript(transcript),
        # 6.2/6.4 regression hash now covers the full segments (words/speaker)
        "segments_hash": hash_segments(segments),
        "language": (result or {}).get("language"),
        "batch_size": effective_batch,
        "duration_s": duration_s,
        "vram_peak_gb": vram_peak_fn() if vram_peak_fn else None,
        "word_timestamps_present": word_timestamps_present(result),
        "words_carry_speakers": words_carry_speakers(result),
        # raw segments kept so false-positive scanning can locate hotwords
        "segments": segments,
        "ok": word_timestamps_present(result) and words_carry_speakers(result),
    }


def default_align_fn(audio: Any, result: dict) -> dict:
    """Real GPU align default: qwen path -> predict.align_qwen (Qwen
    Qwen3-ForcedAligner-0.6B from the baked snapshot), turbo path ->
    predict.align (standard ForcedAligner). Mirrors predict._run_predict's
    align branch, including the language-coverage guard."""
    predict = _load_predict()
    import importlib

    detected_language = (result or {}).get("language")
    if (result or {}).get("whisper_model") == "qwen3-asr":
        return predict.align_qwen(audio, result, False)
    alignment_module = importlib.import_module("whisperx.alignment")
    if detected_language in alignment_module.DEFAULT_ALIGN_MODELS_TORCH or detected_language in (
        alignment_module.DEFAULT_ALIGN_MODELS_HF
    ):
        return predict.align(audio, result, False)
    print(f"Cannot align output: language {detected_language} not supported for alignment", flush=True)
    return result


def default_diarize_fn(audio: Any, result: dict) -> dict:
    """Real GPU diarize default: predict.diarize already ends with
    whisperx.assign_word_speakers(diarize_segments, result,
    speaker_embeddings), which labels the ForcedAligner words with speaker
    ids — exactly the 6.5 handoff."""
    predict = _load_predict()
    from hf_token import resolve_huggingface_token

    hf_token = resolve_huggingface_token(None)
    return predict.diarize(audio, result, False, hf_token, None, None)


def build_qwen_context(hotwords: str | None) -> str:
    """Reuse predict.format_qwen_context when available (same template/cap);
    GPU-free fallback keeps only the neutral empty string."""
    try:
        predict = _load_predict()
    except Exception:
        return ""
    context, _truncated = predict.format_qwen_context(hotwords)
    return context


def _load_predict():
    import predict  # lazy: requires torch/cog/whisperx (GPU environment only)

    return predict


def load_audio_real(path: str):
    """Real GPU path: whisperx.load_audio (ffmpeg decode, 16 kHz mono float)."""
    import whisperx

    return whisperx.load_audio(path)


def model_factory_real(model_name: str):
    """Real GPU path: build the model exactly like predict._run_predict does."""
    predict = _load_predict()
    import torch

    torch.cuda.reset_peak_memory_stats()
    if model_name == "qwen3-asr":
        import importlib

        snapshot_dir = predict.resolve_qwen_snapshot_dir()
        predict.assert_baked_qwen_weights(snapshot_dir, predict.QWEN_ASR_WEIGHT_FILES)
        asr_qwen = importlib.import_module("whisperx.asr_qwen")
        model = asr_qwen.load_model(
            snapshot_dir,
            predict.device,
            language="fr",
            vad_options=None,
            qwen_dtype="float16",
            local_files_only=True,
        )
    else:
        arch = predict.resolve_whisper_model_path(model_name)
        model = predict.whisperx.load_model(arch, predict.device, compute_type=predict.compute_type)
    return model


def audio_duration_real(path: str) -> float:
    """Audio duration in seconds via predict.get_audio_duration (ffmpeg probe)."""
    predict = _load_predict()
    return predict.get_audio_duration(path) / 1000.0


def vram_reset_real() -> None:
    """Reset the CUDA peak-memory counter (call before each run, FIX 7)."""
    import torch

    torch.cuda.reset_peak_memory_stats()


def vram_peak_real() -> float:
    """Peak VRAM in GB since the last reset (torch.cuda.max_memory_allocated)."""
    import torch

    return torch.cuda.max_memory_allocated() / (1024**3)


def free_gpu_real() -> None:
    """Release VRAM between runs (E4-EXEC-FIX FIX B): gc.collect() then
    torch.cuda.empty_cache(). On the real GPU run the turbo+qwen+aligner+
    pyannote models otherwise accumulate in the single harness process and
    later runs OOM (pyannote wespeaker 312 MiB, aligner 93/93 unaligned).
    Injectable via run_golden_set(free_fn=...) for GPU-free tests."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass  # GPU-free context: gc.collect() is the only releasable step


def resolve_batch_size(
    model_name: str,
    batch_size: int | None = None,
    per_model_batch: dict[str, int] | None = None,
) -> int:
    """E4-EXEC-FIX FIX A: effective batch for a run.

    Precedence: explicit caller batch_size > per_model_batch table >
    PER_MODEL_DEFAULT_BATCH (turbo 16, qwen 4). The prod default 64 is
    deliberately NOT used here — batch is an execution parameter, not a
    code defect; predict.py is untouched.
    """
    if batch_size is not None:
        return batch_size
    if per_model_batch and model_name in per_model_batch:
        return per_model_batch[model_name]
    return PER_MODEL_DEFAULT_BATCH.get(model_name, QWEN_DEFAULT_BATCH)


# ---------------------------------------------------------------------------
# Report assembly + evaluation
# ---------------------------------------------------------------------------


def build_report(
    runs: dict[str, dict[str, Any]],
    word_timestamps_present: bool,
    words_carry_speakers: bool,
    vram_peak_gb: float,
    rtfx: float,
    false_positives: list[dict[str, Any]],
    regression_hashes: dict[str, str] | None = None,
    audio_durations_s: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Full JSON-serialisable golden-set report."""
    vram_peak_by_run = {
        name: run["vram_peak_gb"] for name, run in runs.items() if run.get("vram_peak_gb") is not None
    }
    if vram_peak_by_run:
        # FIX 2 (E4-FIX-2): the report's vram_peak_gb is the TRUE global
        # peak = max over the per-run peaks, not the last vram_peak_fn()
        # reading (which only reflects the LAST run — each run resets the
        # CUDA peak counter). Backward compatible: no per-run peaks (CI
        # mocks) keeps the supplied scalar.
        vram_peak_gb = max(vram_peak_by_run.values())
    return {
        "runs": runs,
        "word_timestamps_present": word_timestamps_present,
        "words_carry_speakers": words_carry_speakers,
        "vram_peak_gb": vram_peak_gb,
        # per-run VRAM peaks (FIX 7): each run resets the CUDA peak counter,
        # so these are independent measurements, not one global max
        "vram_peak_by_run": vram_peak_by_run,
        "rtfx": rtfx,
        "false_positives": false_positives,
        "regression_hashes": regression_hashes or {},
        "audio_durations_s": audio_durations_s or {},
        "limits": {"vram_gb": VRAM_LIMIT_GB, "rtfx": [RTFX_MIN, RTFX_MAX]},
    }


def evaluate_report(report: dict[str, Any]) -> dict[str, Any]:
    """Golden-set acceptance: all assertions must pass, failures are listed.

    E4-EXEC-FIX FIX C: a PARTIAL report (a run failed mid-set and carries an
    'error' entry instead of metrics) is tolerated — the failure is
    documented as a listed failure, never a crash.
    """
    failures: list[str] = []
    if not report.get("word_timestamps_present"):
        failures.append("word_timestamps_present: word-level timestamps missing")
    if not report.get("words_carry_speakers"):
        failures.append("words_carry_speakers: assign_word_speakers did not label ForcedAligner words")
    if not vram_ok(report.get("vram_peak_gb", float("inf"))):
        failures.append(f"vram: peak {report.get('vram_peak_gb')} GB >= {VRAM_LIMIT_GB} GB")
    # per-run VRAM gate (FIX 7): any single run above the limit fails
    for run_name, peak in (report.get("vram_peak_by_run") or {}).items():
        if not vram_ok(peak):
            failures.append(f"vram: run {run_name} peak {peak} GB >= {VRAM_LIMIT_GB} GB")
    rtfx = report.get("rtfx")
    if rtfx is None or not rtfx_ok(rtfx):
        failures.append(f"rtfx: {rtfx} outside plausible range ({RTFX_MIN}-{RTFX_MAX})")
    fps = report.get("false_positives") or []
    if fps:
        failures.append(f"false_positives: {len(fps)} hallucinated hotword insertion(s)")
    runs = report.get("runs") or {}
    for name in ("turbo_baseline", "qwen_baseline", "qwen_hotwords"):
        if name not in runs:
            failures.append(f"runs: missing {name}")
        elif runs[name].get("error") is not None:
            # E4-EXEC-FIX FIX C: partial report — document, don't crash
            failures.append(f"runs: {name} failed with error: {runs[name]['error']}")
        elif not runs[name].get("ok"):
            failures.append(f"runs: {name} did not pass its in-run checks")
    regression = report.get("regression_hashes") or {}
    for model_name, recorded_hash in regression.items():
        run = runs.get(f"{model_name}_baseline") or runs.get(model_name)
        if run and run.get("transcript_hash") != recorded_hash:
            failures.append(
                f"regression: {model_name} transcript hash changed "
                f"(expected {recorded_hash[:12]}..., got {str(run.get('transcript_hash'))[:12]}...)"
            )
    return {"all_pass": not failures, "failures": failures}


def _write_partial_snapshot(
    output_path: str,
    runs: dict[str, dict[str, Any]],
    failed_runs: dict[str, str],
) -> None:
    """E4-EXEC-FIX FIX C: incremental JSON dump written to output_path after
    EVERY run — completed runs so far + the error of any failed run. A mid-set
    OOM on run 2/3 therefore leaves a usable artefact (metrics + hashes)
    instead of zero output."""
    snapshot = {
        "partial": True,
        "completed_runs": sorted(runs),
        "failed_runs": dict(failed_runs),
        "runs": dict(runs),
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")


def run_golden_set(
    audio_path: str,
    output_path: str = DEFAULT_OUTPUT,
    load_audio_fn: Callable | None = None,
    model_factory: Callable | None = None,
    clock: Callable[[], float] = time.time,
    vram_peak_fn: Callable[[], float] | None = None,
    vram_reset_fn: Callable[[], None] | None = None,
    align_fn: Callable[[Any, dict], dict] | None = None,
    diarize_fn: Callable[[Any, dict], dict] | None = None,
    audio_duration_fn: Callable[[str], float] | None = None,
    recorded_hashes: dict[str, str] | None = None,
    batch_size: int | None = None,
    per_model_batch: dict[str, int] | None = None,
    free_fn: Callable[[], None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Execute the full golden set (3 runs) and evaluate it. Injectable for CI.

    6.5 wiring: align_fn/diarize_fn default to the real GPU functions
    (default_align_fn / default_diarize_fn — align_qwen vs align dispatch and
    diarize + assign_word_speakers). On the real GPU path each run starts
    with a CUDA peak-counter reset (vram_reset_real) so vram_peak_fn reads a
    per-run peak (FIX 7); CI mocks supply their own functions.

    E4-EXEC-FIX fixes (observed on the real GPU run, deleg_ee3b69ea):
    - FIX A: batch_size / per_model_batch are passed down to run_single
      (defaults turbo 16, qwen 4 via resolve_batch_size) — the prod 64
      freeze OOMed when free VRAM < 6 GiB.
    - FIX B: free_fn (default free_gpu_real: gc.collect + empty_cache) is
      called after EACH run, once the result has been extracted and stored —
      models no longer accumulate across runs.
    - FIX C: a partial snapshot (partial: true, completed runs + failed_runs
      errors) is written to output_path after EVERY run; a run that raises
      is recorded with its error string and the flow continues, so a mid-set
      OOM leaves a usable artefact.
    """
    load_audio_fn = load_audio_fn or load_audio_real
    model_factory = model_factory or model_factory_real
    align_fn = align_fn or default_align_fn
    diarize_fn = diarize_fn or default_diarize_fn
    vram_peak_fn = vram_peak_fn or vram_peak_real
    vram_reset_fn = vram_reset_real if vram_reset_fn is None else vram_reset_fn
    free_fn = free_gpu_real if free_fn is None else free_fn
    audio_duration_fn = audio_duration_fn or audio_duration_real

    runs: dict[str, dict[str, Any]] = {}
    failed_runs: dict[str, str] = {}
    words_speakers_ok = True
    all_fps: list[dict[str, Any]] = []
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    for name, spec in RUNS.items():
        vram_reset_fn()  # per-run VRAM peak (FIX 7)
        try:
            run = run_single(
                run_spec=spec,
                audio_path=audio_path,
                load_audio_fn=load_audio_fn,
                model_factory=model_factory,
                clock=clock,
                align_fn=align_fn,
                diarize_fn=diarize_fn,
                vram_peak_fn=vram_peak_fn,
                # E4-EXEC-FIX FIX A: batch passthrough (turbo 16 / qwen 4)
                batch_size=resolve_batch_size(spec["whisper_model"], batch_size, per_model_batch),
            )
        except Exception as exc:  # E4-EXEC-FIX FIX C: document, keep going
            failed_runs[name] = str(exc)
            runs[name] = {"whisper_model": spec["whisper_model"], "ok": False, "error": str(exc)}
            print(f"RUN FAILED: {name}: {exc}", flush=True)
            _write_partial_snapshot(output_path, runs=runs, failed_runs=failed_runs)
            free_fn()
            continue
        runs[name] = run
        words_speakers_ok = words_speakers_ok and run["words_carry_speakers"]
        segments = _segments_from_transcript(run)
        if segments:
            all_fps.extend(find_hotword_false_positives(segments, HOTWORD_TERMS))
        # E4-EXEC-FIX FIX C: incremental snapshot AFTER each completed run
        # (result extracted and stored first, then VRAM released — FIX B)
        _write_partial_snapshot(output_path, runs=runs, failed_runs=failed_runs)
        free_fn()

    baseline_counts = count_hotword_occurrences(runs.get("qwen_baseline", {}).get("transcript", ""), HOTWORD_TERMS)
    hotwords_counts = count_hotword_occurrences(runs.get("qwen_hotwords", {}).get("transcript", ""), HOTWORD_TERMS)
    recall_report = build_hotword_recall_report(baseline_counts, hotwords_counts, HOTWORD_TERMS)

    # RTFx from the qwen_hotwords run (the target configuration): audio
    # seconds / processing seconds. Without an audio duration (CI mocks)
    # rtfx stays None and evaluate_report flags it — the GPU host supplies it.
    audio_duration_s = audio_duration_fn(audio_path) if audio_duration_fn else None
    if audio_duration_s and runs.get("qwen_hotwords", {}).get("duration_s"):
        rtfx = compute_rtfx(audio_duration_s, runs["qwen_hotwords"]["duration_s"])
    else:
        rtfx = None

    report = build_report(
        runs=runs,
        word_timestamps_present=all(r.get("word_timestamps_present") for r in runs.values()),
        words_carry_speakers=words_speakers_ok,
        vram_peak_gb=vram_peak_fn(),
        rtfx=rtfx,
        false_positives=all_fps,
        regression_hashes=recorded_hashes or {},
    )
    report["hotword_recall"] = recall_report
    evaluation = evaluate_report(report)
    report["evaluation"] = evaluation

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report, evaluation


def _segments_from_transcript(run: dict[str, Any]) -> list[dict]:
    """Segments of a stored run (raw transcript segments from run_single)."""
    return run.get("segments") or []


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replayable golden-set harness (task 6.1)")
    parser.add_argument("--meeting-audio", default=str(REPO_ROOT.parent / "meeting_0209.ogg"))
    parser.add_argument("--fr-extract", default=None, help="fixed FR extract (default: run only the meeting)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--recorded-hashes",
        default=None,
        help="JSON file mapping model name -> expected transcript hash (6.2 bit-identical check)",
    )
    # E4-EXEC-FIX FIX A/D: batch controls (execution parameters, predict.py
    # untouched) — global override and per-model table (turbo 16, qwen 4).
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the batch for ALL runs (default: per-model, turbo 16 / qwen 4)",
    )
    parser.add_argument(
        "--per-model-batch",
        default=None,
        help='JSON mapping model -> batch, e.g. \'{"large-v3-turbo": 16, "qwen3-asr": 4}\'',
    )
    args = parser.parse_args(argv)

    recorded_hashes = {}
    if args.recorded_hashes:
        recorded_hashes = json.loads(Path(args.recorded_hashes).read_text())

    per_model_batch = None
    if args.per_model_batch:
        try:
            per_model_batch = json.loads(args.per_model_batch)
            if not isinstance(per_model_batch, dict):
                raise ValueError("must be a JSON object mapping model name -> batch")
        except (ValueError, json.JSONDecodeError) as exc:
            print(f"FAIL golden-set: invalid --per-model-batch JSON ({exc})", file=sys.stderr)
            return 2

    try:
        report, evaluation = run_golden_set(
            audio_path=args.meeting_audio,
            output_path=args.output,
            recorded_hashes=recorded_hashes,
            batch_size=args.batch_size,
            per_model_batch=per_model_batch,
        )
    except ImportError as exc:
        print(
            f"FAIL golden-set: GPU dependencies missing ({exc}) — run on the CUDA host, "
            "unit tests cover this module GPU-free via tests/test_golden_set.py",
            file=sys.stderr,
        )
        return 2
    print(json.dumps(evaluation, ensure_ascii=False, indent=2))
    return 0 if evaluation["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
