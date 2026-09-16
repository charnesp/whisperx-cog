from cog import BasePredictor, Input, Path, BaseModel
from typing import Any, Optional

import gc
import importlib
import logging
import math
import os
import shutil
import warnings

# Suppress torchcodec warning from pyannote: we load audio with whisperx.load_audio
# and pass waveform to diarization; torchcodec's FFmpeg decoding is never used.
warnings.filterwarnings(
    "ignore",
    message=r"\s*torchcodec is not installed correctly",
    module="pyannote.audio.core.io",
)

from whisperx.audio import N_SAMPLES, log_mel_spectrogram
import whisperx
from whisperx.diarize import DiarizationPipeline
from json_sanitize import sanitize_error_message, sanitize_for_json
from hf_token import require_diarization_token
from model_paths import resolve_vad_source_path, resolve_whisper_model_path
from models_registry import (
    MODELS,
    local_candidates,
    resolve_key as _resolve_model_key,
)
import tempfile
import time
import torch
import ffmpeg

compute_type = "float16"  # change to "int8" if low on GPU mem (may reduce accuracy)
device = "cuda"

# ---------------------------------------------------------------------------
# Qwen3-ASR backend (qwen3-asr) — helpers and constants (GPU-free, unit-tested)
# ---------------------------------------------------------------------------
WHISPER_DEFAULT_BATCH = 64  # faster-whisper path default (pre-change behavior)

QWEN_MODEL_NAME = "qwen3-asr"
QWEN_DEFAULT_BATCH = 4  # measured OOM at larger batches on the shared 16 GB GPU
QWEN_MAX_BATCH = 8
QWEN_CONTEXT_CAP = 2000  # chars; truncation logs carry lengths only, never content
# Model constants come from the unified registry (E5-CODE-1 T1): single
# declarative source of truth, no hardcoded paths/repos here.
_QWEN_MODEL_KEY = _resolve_model_key(QWEN_MODEL_NAME)
_QWEN_ALIGNER_KEY = "qwen3-forced-aligner-0.6b"
QWEN_ASR_HF_REPO = MODELS[_QWEN_MODEL_KEY].hf_repo
QWEN_ALIGNER_HF_REPO = MODELS[_QWEN_ALIGNER_KEY].hf_repo
# Local snapshot dirs baked into /models (see models.lock revisions)
QWEN_MODEL_LOCAL_PATHS = local_candidates(_QWEN_MODEL_KEY)
QWEN_ALIGNER_LOCAL_PATHS = local_candidates(_QWEN_ALIGNER_KEY)
# Non-empty weight files that must exist in each baked snapshot
QWEN_ASR_WEIGHT_FILES = list(MODELS[_QWEN_MODEL_KEY].weight_files)
QWEN_ALIGNER_WEIGHT_FILES = list(MODELS[_QWEN_ALIGNER_KEY].weight_files)

logger = logging.getLogger(__name__)


def _env_flag_enabled(value: str | None) -> bool:
    """Truthy env values for the ENABLE_QWEN kill-switch (default: enabled)."""
    if value is None:
        return True
    return value.strip().lower() in {"1", "true", "yes", "on"}


def qwen_enabled() -> bool:
    """ENABLE_QWEN kill-switch — read at request time, default enabled."""
    return _env_flag_enabled(os.environ.get("ENABLE_QWEN"))


def assert_qwen_enabled() -> None:
    if not qwen_enabled():
        raise RuntimeError(
            "The qwen3-asr backend is disabled: set the ENABLE_QWEN environment "
            "variable to '1' (or 'true') to enable it."
        )


def resolve_qwen_batch_size(batch_size, provided: bool = True) -> int:
    """Qwen batch resolution: default 4 when absent, explicit values clamped [1, 8].

    0/negative values are corrected to the default (not clamped to 1): a
    non-positive batch is a client mistake, not an intentional single-threaded
    run. A clamp logs old/new lengths only (ints, no PII).
    """
    if not provided or batch_size is None:
        return QWEN_DEFAULT_BATCH
    try:
        value = int(batch_size)
    except (TypeError, ValueError):
        return QWEN_DEFAULT_BATCH
    if value <= 0:
        logger.warning(
            "Qwen batch_size %d invalid (non-positive), using default %d",
            value,
            QWEN_DEFAULT_BATCH,
        )
        return QWEN_DEFAULT_BATCH
    clamped = max(1, min(QWEN_MAX_BATCH, value))
    if clamped != value:
        logger.warning(
            "Qwen batch_size clamped: %d -> %d (allowed range 1-%d)",
            value,
            clamped,
            QWEN_MAX_BATCH,
        )
    return clamped


def format_qwen_context(hotwords) -> tuple[str, bool]:
    """Assemble the Qwen context string from client hotwords.

    Wraps the hotwords in the meeting-context template validated live
    (design.md §2): 'Réunion technique chez [ENTREPRISE], [CONTEXTE].
    Participants : [LISTE PARTICIPANTS]. Termes techniques : [LISTE VOCABULAIRE].'

    Known deviation from design.md §2: the OpenAI multipart contract has no
    company/participants fields, so [ENTREPRISE]/[CONTEXTE]/[LISTE PARTICIPANTS]
    cannot be filled. The simplified template keeps only the technical-vocabulary
    section that hotwords can populate. Documented as a deviation in tasks.md.

    Returns (context, truncated). Empty/absent hotwords -> empty string
    (neutral for Qwen). Content is never logged, only lengths.
    """
    words = hotwords.strip() if isinstance(hotwords, str) else ""
    if not words:
        return "", False
    context = (
        "Contexte technique de la réunion. "
        f"Termes, entités et noms propres attendus : {words}."
    )
    if len(context) > QWEN_CONTEXT_CAP:
        truncated = context[:QWEN_CONTEXT_CAP]
        # Lengths only — hotword content is never logged (client proper nouns).
        logger.warning(
            "Qwen context truncated: %d -> %d chars",
            len(context),
            len(truncated),
        )
        return truncated, True
    return context, False


def build_asr_options(temperature: float, initial_prompt, hotwords) -> dict:
    """faster-whisper asr_options — whisper path, semantics unchanged."""
    return {
        "temperatures": [temperature],
        "initial_prompt": initial_prompt,
        "hotwords": hotwords if hotwords and hotwords.strip() else None,
    }


def should_detect_language(whisper_model: str, language) -> bool:
    """Qwen detects language internally; whisper keeps the recursive detection loop."""
    if language is not None:
        return False
    return whisper_model != QWEN_MODEL_NAME


def qwen_effective_language(whisper_model: str, language):
    """Language passed through as-is on the qwen path (pipeline handles mapping)."""
    if whisper_model != QWEN_MODEL_NAME:
        return language
    if language is None:
        return None
    return str(language).strip() or None


def assert_baked_qwen_weights(snapshot_dir: str, weight_files=None) -> None:
    """Fail fast when a baked Qwen snapshot is missing weights (HF_HUB_OFFLINE=1:
    no silent HuggingFace download is possible at runtime)."""
    weight_files = weight_files or QWEN_ASR_WEIGHT_FILES
    missing = [
        name
        for name in weight_files
        if not os.path.isfile(os.path.join(snapshot_dir, name))
        or os.path.getsize(os.path.join(snapshot_dir, name)) == 0
    ]
    if missing:
        raise RuntimeError(
            f"Missing baked Qwen weights in {snapshot_dir}: {', '.join(missing)}. "
            f"HF_HUB_OFFLINE=1 forbids a runtime download from HuggingFace — "
            f"rebuild the image so cog.yaml bakes {QWEN_ASR_HF_REPO} into /models."
        )


def resolve_qwen_snapshot_dir(candidates=None) -> str:
    """Return the first existing baked/local qwen snapshot dir, else the baked path."""
    for path in candidates or QWEN_MODEL_LOCAL_PATHS:
        if os.path.isdir(path):
            return path
    return candidates[0] if candidates else QWEN_MODEL_LOCAL_PATHS[0]


def _resolve_input_default(val: Any) -> Any:
    """When predict() is called from Python (not via Cog API), omitted args get the Input()
    object (Pydantic FieldInfo) as value. Return the actual default in that case."""
    if type(val).__name__ == "FieldInfo":
        default = getattr(val, "default", val)
        if type(default).__name__ == "PydanticUndefined":
            return None
        return default
    return val


class Output(BaseModel):
    segments: Any  # list of segment dicts (start, end, text, words?, speaker?)
    detected_language: str
    speaker_embeddings: Optional[dict] = None


class Predictor(BasePredictor):
    def setup(self):
        destination_folder = "../root/.cache/torch"
        os.makedirs(destination_folder, exist_ok=True)

        source_file_path = resolve_vad_source_path()
        if source_file_path:
            destination_file_path = os.path.join(
                destination_folder, os.path.basename(source_file_path)
            )
            if not os.path.exists(destination_file_path):
                shutil.copy(source_file_path, destination_folder)

    def predict(
        self,
        audio_file: Path = Input(description="Audio file"),
        whisper_model: str = Input(
            description="Whisper ASR model: tiny (smallest), large-v3 (higher accuracy), large-v3-turbo (faster, less VRAM), or qwen3-asr (Qwen3-ASR-1.7B: best proper-noun accuracy with hotwords/context, requires ENABLE_QWEN)",
            default="large-v3-turbo",
            choices=["tiny", "large-v3", "large-v3-turbo", "qwen3-asr"],
        ),
        language: str | None = Input(
            description="ISO code of the language spoken in the audio, omit or null to perform language detection",
            default=None,
        ),
        language_detection_min_prob: float = Input(
            description="If language is not specified, then the language will be detected recursively on different "
            "parts of the file until it reaches the given probability",
            default=0,
        ),
        language_detection_max_tries: int = Input(
            description="If language is not specified, then the language will be detected following the logic of "
            "language_detection_min_prob parameter, but will stop after the given max retries. If max "
            "retries is reached, the most probable language is kept.",
            default=5,
        ),
        initial_prompt: str | None = Input(
            description="Optional text to provide as a prompt for the first window",
            default=None,
        ),
        hotwords: str | None = Input(
            description="Hotwords/hint phrases to the model (e.g. \"WhisperX, PyAnnote, GPU\"); improves recognition of rare/technical terms",
            default=None,
        ),
        batch_size: int | None = Input(
            description="Parallelization of input audio transcription. Optional: when omitted, the per-model default applies (64 for faster-whisper, 4 for qwen3-asr; qwen values are clamped to 1-8)",
            default=None,
        ),
        temperature: float = Input(
            description="Temperature to use for sampling", default=0
        ),
        vad_onset: float = Input(description="VAD onset", default=0.500),
        vad_offset: float = Input(description="VAD offset", default=0.363),
        align_output: bool = Input(
            description="Aligns whisper output to get accurate word-level timestamps",
            default=True,
        ),
        diarization: bool = Input(description="Assign speaker ID labels", default=True),
        huggingface_access_token: str | None = Input(
            description="To enable diarization, please enter your HuggingFace token (read). You need to accept "
            "the user agreement for the models specified in the README.",
            default=None,
        ),
        min_speakers: int | None = Input(
            description="Minimum number of speakers if diarization is activated (omit or null if unknown)",
            default=None,
        ),
        max_speakers: int | None = Input(
            description="Maximum number of speakers if diarization is activated (omit or null if unknown)",
            default=None,
        ),
        debug: bool = Input(
            description="Print out compute/inference times and memory usage information",
            default=False,
        ),
    ) -> Output:
        try:
            return self._run_predict(
                audio_file=audio_file,
                whisper_model=whisper_model,
                language=language,
                language_detection_min_prob=language_detection_min_prob,
                language_detection_max_tries=language_detection_max_tries,
                initial_prompt=initial_prompt,
                hotwords=hotwords,
                batch_size=batch_size,
                temperature=temperature,
                vad_onset=vad_onset,
                vad_offset=vad_offset,
                align_output=align_output,
                diarization=diarization,
                huggingface_access_token=huggingface_access_token,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                debug=debug,
            )
        except Exception as e:
            safe_msg = sanitize_error_message(str(e))
            raise RuntimeError(f"Prediction failed: {type(e).__name__}: {safe_msg}") from None

    def _run_predict(
        self,
        audio_file,
        whisper_model,
        language,
        language_detection_min_prob,
        language_detection_max_tries,
        initial_prompt,
        hotwords,
        batch_size,
        temperature,
        vad_onset,
        vad_offset,
        align_output,
        diarization,
        huggingface_access_token,
        min_speakers,
        max_speakers,
        debug,
    ) -> Output:
        with torch.inference_mode():
            # Resolve Pydantic FieldInfo → real default when predict() is called from Python
            # (e.g. run_local.py) without passing optional args
            audio_file = _resolve_input_default(audio_file)
            whisper_model = _resolve_input_default(whisper_model)
            language = _resolve_input_default(language)
            language_detection_min_prob = _resolve_input_default(language_detection_min_prob)
            language_detection_max_tries = _resolve_input_default(language_detection_max_tries)
            initial_prompt = _resolve_input_default(initial_prompt)
            hotwords = _resolve_input_default(hotwords)
            batch_size = _resolve_input_default(batch_size)
            temperature = _resolve_input_default(temperature)
            vad_onset = _resolve_input_default(vad_onset)
            vad_offset = _resolve_input_default(vad_offset)
            align_output = _resolve_input_default(align_output)
            diarization = _resolve_input_default(diarization)
            huggingface_access_token = _resolve_input_default(huggingface_access_token)
            min_speakers = _resolve_input_default(min_speakers)
            max_speakers = _resolve_input_default(max_speakers)
            debug = _resolve_input_default(debug)

            is_qwen = whisper_model == QWEN_MODEL_NAME

            if is_qwen:
                assert_qwen_enabled()

            whisper_arch = resolve_whisper_model_path(whisper_model)
            asr_options = build_asr_options(
                temperature=temperature,
                initial_prompt=initial_prompt,
                hotwords=hotwords,
            )

            vad_options = {"vad_onset": vad_onset, "vad_offset": vad_offset}

            audio_duration = get_audio_duration(audio_file)

            if (
                should_detect_language(whisper_model, language)
                and language_detection_min_prob > 0
                and audio_duration > 30000
            ):
                segments_duration_ms = 30000

                language_detection_max_tries = min(
                    language_detection_max_tries,
                    math.floor(audio_duration / segments_duration_ms),
                )

                segments_starts = distribute_segments_equally(
                    audio_duration, segments_duration_ms, language_detection_max_tries
                )

                print(
                    "Detecting languages on segments starting at "
                    + ", ".join(map(str, segments_starts)),
                    flush=True,
                )

                detected_language_details = detect_language(
                    audio_file,
                    segments_starts,
                    language_detection_min_prob,
                    language_detection_max_tries,
                    asr_options,
                    vad_options,
                    whisper_arch,
                )

                detected_language_code = detected_language_details["language"]
                detected_language_prob = detected_language_details["probability"]
                detected_language_iterations = detected_language_details["iterations"]

                print(
                    f"Detected language {detected_language_code} ({detected_language_prob:.2f}) after "
                    f"{detected_language_iterations} iterations.",
                    flush=True,
                )

                language = detected_language_details["language"]

            start_time = time.time_ns() / 1e9

            if is_qwen:
                # Qwen3-ASR path: local baked snapshot + explicit fp16 dtype
                # (default fp32 measures ~10 GB VRAM vs ~5 GB fp16). Baked
                # weights are mandatory: HF_HUB_OFFLINE=1 forbids runtime
                # downloads, so fail fast with a clear message instead.
                qwen_snapshot_dir = resolve_qwen_snapshot_dir()
                assert_baked_qwen_weights(qwen_snapshot_dir, QWEN_ASR_WEIGHT_FILES)
                print(f"Qwen ASR snapshot: {qwen_snapshot_dir}", flush=True)
                asr_qwen_module = importlib.import_module("whisperx.asr_qwen")
                language = qwen_effective_language(whisper_model, language)
                model = asr_qwen_module.load_model(
                    qwen_snapshot_dir,
                    device,
                    language=language,
                    vad_options=vad_options,
                    qwen_dtype="float16",
                    local_files_only=True,
                )
            else:
                model = whisperx.load_model(
                    whisper_arch,
                    device,
                    compute_type=compute_type,
                    language=language,
                    asr_options=asr_options,
                    vad_options=vad_options,
                )

            if debug:
                elapsed_time = time.time_ns() / 1e9 - start_time
                print(f"Duration to load model: {elapsed_time:.2f} s", flush=True)

            start_time = time.time_ns() / 1e9

            audio = whisperx.load_audio(audio_file)

            if debug:
                elapsed_time = time.time_ns() / 1e9 - start_time
                print(f"Duration to load audio: {elapsed_time:.2f} s", flush=True)

            start_time = time.time_ns() / 1e9

            if is_qwen:
                context, _truncated = format_qwen_context(hotwords)
                effective_batch = resolve_qwen_batch_size(batch_size, provided=batch_size is not None)
                result = model.transcribe(
                    audio,
                    batch_size=effective_batch,
                    context=context,
                )
            else:
                # Whisper path: batch_size None must resolve to the historical
                # faster-whisper default 64. The E3 bridge change (batch_size
                # only when provided) exposed the fork's `batch_size or
                # self._batch_size` fallback where None falls through to the
                # transformers pipeline default (effective batch 1) — pass 64
                # explicitly to keep the pre-change behavior.
                effective_whisper_batch = (
                    WHISPER_DEFAULT_BATCH if batch_size is None else batch_size
                )
                result = model.transcribe(audio, batch_size=effective_whisper_batch)
            detected_language = result["language"]

            if debug:
                elapsed_time = time.time_ns() / 1e9 - start_time
                print(f"Duration to transcribe: {elapsed_time:.2f} s", flush=True)

            gc.collect()
            torch.cuda.empty_cache()
            del model

            if align_output:
                if is_qwen:
                    result = align_qwen(audio, result, debug)
                else:
                    alignment_module = importlib.import_module("whisperx.alignment")
                    if (
                        detected_language in alignment_module.DEFAULT_ALIGN_MODELS_TORCH
                        or detected_language in alignment_module.DEFAULT_ALIGN_MODELS_HF
                    ):
                        result = align(audio, result, debug)
                    else:
                        print(
                            f"Cannot align output as language {detected_language} is not supported for alignment",
                            flush=True,
                        )

            if diarization:
                hf_token = require_diarization_token(diarization, huggingface_access_token)
                result = diarize(
                    audio,
                    result,
                    debug,
                    hf_token,
                    min_speakers,
                    max_speakers,
                )

            if debug:
                print(
                    f"max gpu memory allocated over runtime: {torch.cuda.max_memory_reserved() / (1024 ** 3):.2f} GB",
                    flush=True,
                )

        # Normalize to Output types (list, str, Optional[dict]) so schema/validation never fails
        raw_segments = result.get("segments")
        segments = sanitize_for_json(raw_segments) if raw_segments is not None else []
        if not isinstance(segments, list):
            segments = []

        raw_lang = detected_language
        if isinstance(raw_lang, dict):
            detected_language_str = str(raw_lang.get("language", ""))
        else:
            detected_language_str = str(raw_lang) if raw_lang is not None else ""

        raw_embeddings = result.get("speaker_embeddings")
        embeddings = sanitize_for_json(raw_embeddings) if raw_embeddings is not None else None
        if embeddings is not None and not isinstance(embeddings, dict):
            embeddings = None

        return Output(
            segments=segments,
            detected_language=detected_language_str,
            speaker_embeddings=embeddings,
        )


def get_audio_duration(file_path):
    probe = ffmpeg.probe(file_path)
    stream = next(
        (s for s in probe["streams"] if s["codec_type"] == "audio"), None
    )
    if stream is None:
        raise ValueError(f"No audio stream found in {file_path}")
    # Duration can be in the stream or in the format (e.g. some MP3s lack stream duration)
    duration_s = stream.get("duration")
    if duration_s is None and "format" in probe:
        duration_s = probe["format"].get("duration")
    if duration_s is None:
        raise ValueError(
            f"Cannot get duration for {file_path}: no 'duration' in stream or format. "
            "Try re-encoding the file (e.g. with ffmpeg -i in.mp3 -acodec copy out.mp3)."
        )
    return float(duration_s) * 1000


def detect_language(
    full_audio_file_path,
    segments_starts,
    language_detection_min_prob,
    language_detection_max_tries,
    asr_options,
    vad_options,
    whisper_arch,
    iteration=1,
):
    model = whisperx.load_model(
        whisper_arch,
        device,
        compute_type=compute_type,
        asr_options=asr_options,
        vad_options=vad_options,
    )

    start_ms = segments_starts[iteration - 1]

    audio_segment_file_path = extract_audio_segment(
        full_audio_file_path, start_ms, 30000
    )

    audio = whisperx.load_audio(audio_segment_file_path)

    model_n_mels = model.model.feat_kwargs.get("feature_size")
    segment = log_mel_spectrogram(
        audio[:N_SAMPLES],
        n_mels=model_n_mels if model_n_mels is not None else 80,
        padding=0 if audio.shape[0] >= N_SAMPLES else N_SAMPLES - audio.shape[0],
    )
    encoder_output = model.model.encode(segment)
    results = model.model.model.detect_language(encoder_output)
    language_token, language_probability = results[0][0]
    language = language_token[2:-2]

    print(
        f"Iteration {iteration} - Detected language: {language} ({language_probability:.2f})",
        flush=True,
    )

    audio_segment_file_path.unlink()

    gc.collect()
    torch.cuda.empty_cache()
    del model

    detected_language = {
        "language": language,
        "probability": language_probability,
        "iterations": iteration,
    }

    if (
        language_probability >= language_detection_min_prob
        or iteration >= language_detection_max_tries
    ):
        return detected_language

    next_iteration_detected_language = detect_language(
        full_audio_file_path,
        segments_starts,
        language_detection_min_prob,
        language_detection_max_tries,
        asr_options,
        vad_options,
        whisper_arch,
        iteration + 1,
    )

    if (
        next_iteration_detected_language["probability"]
        > detected_language["probability"]
    ):
        return next_iteration_detected_language

    return detected_language


def extract_audio_segment(input_file_path, start_time_ms, duration_ms):
    input_file_path = (
        Path(input_file_path)
        if not isinstance(input_file_path, Path)
        else input_file_path
    )
    file_extension = input_file_path.suffix

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        temp_file_path = Path(temp_file.name)

        print(f"Extracting from {input_file_path.name} to {temp_file.name}", flush=True)

        try:
            (
                ffmpeg.input(input_file_path, ss=start_time_ms / 1000)
                .output(temp_file.name, t=duration_ms / 1000)
                .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
            )
        except ffmpeg.Error as e:
            print("ffmpeg error occurred: ", e.stderr.decode("utf-8"), flush=True)
            raise e

    return temp_file_path


def distribute_segments_equally(total_duration, segments_duration, iterations):
    available_duration = total_duration - segments_duration

    if iterations > 1:
        spacing = available_duration // (iterations - 1)
    else:
        spacing = 0

    start_times = [i * spacing for i in range(iterations)]

    if iterations > 1:
        start_times[-1] = total_duration - segments_duration

    return start_times


def align(audio, result, debug):
    start_time = time.time_ns() / 1e9

    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"], device=device
    )
    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    if debug:
        elapsed_time = time.time_ns() / 1e9 - start_time
        print(f"Duration to align output: {elapsed_time:.2f} s", flush=True)

    gc.collect()
    torch.cuda.empty_cache()
    del model_a

    return result


def align_qwen(audio, result, debug):
    """Word-level alignment on the qwen path: Qwen3-ForcedAligner-0.6B from the
    baked /models snapshot (wav2vec2 is incompatible with qwen outputs)."""
    start_time = time.time_ns() / 1e9

    aligner_snapshot_dir = resolve_qwen_snapshot_dir(QWEN_ALIGNER_LOCAL_PATHS)
    assert_baked_qwen_weights(aligner_snapshot_dir, QWEN_ALIGNER_WEIGHT_FILES)
    print(f"Qwen forced aligner snapshot: {aligner_snapshot_dir}", flush=True)

    alignment_qwen_module = importlib.import_module("whisperx.alignment_qwen")
    model_a, metadata = alignment_qwen_module.load_align_model(
        language_code=result["language"],
        device=device,
        model_name=aligner_snapshot_dir,
        model_cache_only=True,
        qwen_dtype="float16",
    )
    result = alignment_qwen_module.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    if debug:
        elapsed_time = time.time_ns() / 1e9 - start_time
        print(f"Duration to align output: {elapsed_time:.2f} s", flush=True)

    gc.collect()
    torch.cuda.empty_cache()
    del model_a

    return result


def diarize(audio, result, debug, huggingface_access_token, min_speakers, max_speakers):
    start_time = time.time_ns() / 1e9

    diarize_model = DiarizationPipeline(
        token=huggingface_access_token, device=device
    )
    diarize_result = diarize_model(
        audio,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        return_embeddings=True,
    )
    diarize_segments, speaker_embeddings = diarize_result

    result = whisperx.assign_word_speakers(diarize_segments, result, speaker_embeddings)

    if debug:
        elapsed_time = time.time_ns() / 1e9 - start_time
        print(f"Duration to diarize segments: {elapsed_time:.2f} s", flush=True)

    gc.collect()
    torch.cuda.empty_cache()
    del diarize_model

    return result
