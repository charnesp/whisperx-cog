from cog import BasePredictor, Input, Path, BaseModel
from typing import Any, Optional
from whisperx.audio import N_SAMPLES, log_mel_spectrogram

import gc
import importlib
import math
import os
import shutil
import whisperx
from whisperx.diarize import DiarizationPipeline
import tempfile
import time
import torch
import ffmpeg

compute_type = "float16"  # change to "int8" if low on GPU mem (may reduce accuracy)
device = "cuda"

# Local paths used when models are pre-downloaded (e.g. in Docker build)
WHISPER_MODEL_LOCAL_PATHS = {
    "tiny": "./models/faster-whisper-tiny",
    "large-v3": "./models/faster-whisper-large-v3",
    "large-v3-turbo": "./models/faster-whisper-large-v3-turbo",
}
# HuggingFace repo IDs for download when local path does not exist (e.g. local dev)
WHISPER_MODEL_HF_IDS = {
    "tiny": "Systran/faster-whisper-tiny",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
}


def _resolve_input_default(val: Any) -> Any:
    """When predict() is called from Python (not via Cog API), omitted args get the Input()
    object (Pydantic FieldInfo) as value. Return the actual default in that case."""
    if type(val).__name__ == "FieldInfo":
        default = getattr(val, "default", val)
        if type(default).__name__ == "PydanticUndefined":
            return None
        return default
    return val


def _resolve_whisper_model_path(whisper_model: str) -> str:
    """Use local path if it exists, otherwise HuggingFace repo ID for download."""
    local_path = WHISPER_MODEL_LOCAL_PATHS[whisper_model]
    if os.path.isdir(local_path) and os.path.isfile(
        os.path.join(local_path, "model.bin")
    ):
        return local_path
    return WHISPER_MODEL_HF_IDS[whisper_model]


class Output(BaseModel):
    segments: Any
    detected_language: str
    speaker_embeddings: Optional[dict] = None


class Predictor(BasePredictor):
    def setup(self):
        source_folder = "./models/vad"
        destination_folder = "../root/.cache/torch"
        file_name = "whisperx-vad-segmentation.bin"

        os.makedirs(destination_folder, exist_ok=True)

        source_file_path = os.path.join(source_folder, file_name)
        if os.path.exists(source_file_path):
            destination_file_path = os.path.join(destination_folder, file_name)

            if not os.path.exists(destination_file_path):
                shutil.copy(source_file_path, destination_folder)

    def predict(
        self,
        audio_file: Path = Input(description="Audio file"),
        whisper_model: str = Input(
            description="Whisper ASR model: tiny (smallest), large-v3 (higher accuracy), or large-v3-turbo (faster, less VRAM)",
            default="large-v3-turbo",
            choices=["tiny", "large-v3", "large-v3-turbo"],
        ),
        language: str = Input(
            description="ISO code of the language spoken in the audio, specify None to perform language detection",
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
        initial_prompt: str = Input(
            description="Optional text to provide as a prompt for the first window",
            default=None,
        ),
        hotwords: str = Input(
            description="Hotwords/hint phrases to the model (e.g. \"WhisperX, PyAnnote, GPU\"); improves recognition of rare/technical terms",
            default=None,
        ),
        batch_size: int = Input(
            description="Parallelization of input audio transcription", default=64
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
        huggingface_access_token: str = Input(
            description="To enable diarization, please enter your HuggingFace token (read). You need to accept "
            "the user agreement for the models specified in the README.",
            default=None,
        ),
        min_speakers: int = Input(
            description="Minimum number of speakers if diarization is activated (leave blank if unknown)",
            default=None,
        ),
        max_speakers: int = Input(
            description="Maximum number of speakers if diarization is activated (leave blank if unknown)",
            default=None,
        ),
        debug: bool = Input(
            description="Print out compute/inference times and memory usage information",
            default=False,
        ),
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

            whisper_arch = _resolve_whisper_model_path(whisper_model)
            asr_options = {
                "temperatures": [temperature],
                "initial_prompt": initial_prompt,
                "hotwords": hotwords if hotwords and hotwords.strip() else None,
            }


            vad_options = {"vad_onset": vad_onset, "vad_offset": vad_offset}

            audio_duration = get_audio_duration(audio_file)

            if (
                language is None
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
                    + ", ".join(map(str, segments_starts))
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
                    f"{detected_language_iterations} iterations."
                )

                language = detected_language_details["language"]

            start_time = time.time_ns() / 1e9

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
                print(f"Duration to load model: {elapsed_time:.2f} s")

            start_time = time.time_ns() / 1e9

            audio = whisperx.load_audio(audio_file)

            if debug:
                elapsed_time = time.time_ns() / 1e9 - start_time
                print(f"Duration to load audio: {elapsed_time:.2f} s")

            start_time = time.time_ns() / 1e9

            result = model.transcribe(audio, batch_size=batch_size)
            detected_language = result["language"]

            if debug:
                elapsed_time = time.time_ns() / 1e9 - start_time
                print(f"Duration to transcribe: {elapsed_time:.2f} s")

            gc.collect()
            torch.cuda.empty_cache()
            del model

            if align_output:
                alignment_module = importlib.import_module("whisperx.alignment")
                if (
                    detected_language in alignment_module.DEFAULT_ALIGN_MODELS_TORCH
                    or detected_language in alignment_module.DEFAULT_ALIGN_MODELS_HF
                ):
                    result = align(audio, result, debug)
                else:
                    print(
                        f"Cannot align output as language {detected_language} is not supported for alignment"
                    )

            if diarization:
                result = diarize(
                    audio,
                    result,
                    debug,
                    huggingface_access_token,
                    min_speakers,
                    max_speakers,
                )

            if debug:
                print(
                    f"max gpu memory allocated over runtime: {torch.cuda.max_memory_reserved() / (1024 ** 3):.2f} GB"
                )

        return Output(
            segments=result["segments"],
            detected_language=detected_language,
            speaker_embeddings=result.get("speaker_embeddings"),
        )


def get_audio_duration(file_path):
    probe = ffmpeg.probe(file_path)
    stream = next(
        (stream for stream in probe["streams"] if stream["codec_type"] == "audio"), None
    )
    return float(stream["duration"]) * 1000


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
        f"Iteration {iteration} - Detected language: {language} ({language_probability:.2f})"
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

        print(f"Extracting from {input_file_path.name} to {temp_file.name}")

        try:
            (
                ffmpeg.input(input_file_path, ss=start_time_ms / 1000)
                .output(temp_file.name, t=duration_ms / 1000)
                .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
            )
        except ffmpeg.Error as e:
            print("ffmpeg error occurred: ", e.stderr.decode("utf-8"))
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
        print(f"Duration to align output: {elapsed_time:.2f} s")

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
        print(f"Duration to diarize segments: {elapsed_time:.2f} s")

    gc.collect()
    torch.cuda.empty_cache()
    del diarize_model

    return result
