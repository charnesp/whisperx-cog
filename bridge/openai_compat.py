"""OpenAI-compatible speech-to-text endpoint for the bridge.

Audio handoff to Cog uses base64 data URIs in JSON (Cog HTTP API contract).
See docs/BRIDGE.md and openspec/changes/openai-compatible-stt-endpoint/design.md.
"""

from __future__ import annotations

import base64
import io
import json
import mimetypes
import os
import re
import socket
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple

COG_URL = os.environ.get("COG_URL", "http://127.0.0.1:5000")
OPENAI_STT_TIMEOUT_SECONDS = int(os.environ.get("OPENAI_STT_TIMEOUT_SECONDS", "300"))
OPENAI_STT_MAX_FILE_SIZE_MB = int(os.environ.get("OPENAI_STT_MAX_FILE_SIZE_MB", "25"))
OPENAI_STT_MAX_FILE_SIZE_BYTES = OPENAI_STT_MAX_FILE_SIZE_MB * 1024 * 1024

MODEL_MAP = {
    "whisper-1": "large-v3-turbo",
    "large-v3": "large-v3",
    "large-v3-turbo": "large-v3-turbo",
    "tiny": "tiny",
}

OPENAI_AUDIO_EXTENSIONS = frozenset(
    {"flac", "mp3", "mp4", "mpeg", "mpga", "m4a", "ogg", "wav", "webm"}
)

EXTENSION_MIMES = {
    "flac": "audio/flac",
    "mp3": "audio/mpeg",
    "mp4": "video/mp4",
    "mpeg": "audio/mpeg",
    "mpga": "audio/mpeg",
    "m4a": "audio/mp4",
    "ogg": "audio/ogg",
    "wav": "audio/wav",
    "webm": "video/webm",
}

MOCK_COG_OUTPUT = {
    "segments": [
        {
            "start": 0.0,
            "end": 2.5,
            "text": "Hello world",
            "words": [
                {"word": "Hello", "start": 0.0, "end": 1.0, "score": 0.99},
                {"word": "world", "start": 1.0, "end": 2.5, "score": 0.98},
            ],
        },
        {
            "start": 2.5,
            "end": 5.0,
            "text": "How are you",
            "words": [
                {"word": "How", "start": 2.5, "end": 3.2, "score": 0.97},
                {"word": "are", "start": 3.2, "end": 4.0, "score": 0.96},
                {"word": "you", "start": 4.0, "end": 5.0, "score": 0.95},
            ],
        },
    ],
    "detected_language": "en",
}


class FormField:
    def __init__(
        self,
        *,
        value: Optional[str] = None,
        filename: Optional[str] = None,
        file_data: bytes = b"",
    ):
        self.value = value
        self.filename = filename
        self.file = io.BytesIO(file_data)


class MultipartForm:
    def __init__(self) -> None:
        self._fields: Dict[str, FormField | List[FormField]] = {}

    def __contains__(self, name: str) -> bool:
        return name in self._fields

    def __getitem__(self, name: str) -> FormField | List[FormField]:
        return self._fields[name]

    def add_field(self, name: str, field: FormField) -> None:
        existing = self._fields.get(name)
        if existing is None:
            self._fields[name] = field
            return
        if isinstance(existing, list):
            existing.append(field)
        else:
            self._fields[name] = [existing, field]


def _parse_content_disposition(header: str) -> Tuple[Optional[str], Optional[str]]:
    name = None
    filename = None
    for part in header.split(";"):
        part = part.strip()
        if part.lower().startswith("name="):
            name = part.split("=", 1)[1].strip().strip('"')
        elif part.lower().startswith("filename="):
            filename = part.split("=", 1)[1].strip().strip('"')
    return name, filename


def _parse_multipart_body(content_type: str, body: bytes) -> MultipartForm:
    match = re.search(r"boundary=(?P<boundary>[^;]+)", content_type, flags=re.IGNORECASE)
    if not match:
        raise ValueError("missing multipart boundary")
    boundary = match.group("boundary").strip().strip('"')
    delimiter = f"--{boundary}".encode("utf-8")
    form = MultipartForm()

    for part in body.split(delimiter):
        part = part.strip(b"\r\n")
        if not part or part == b"--":
            continue
        header_blob, _, content = part.partition(b"\r\n\r\n")
        if not header_blob:
            continue
        content = content.rstrip(b"\r\n")
        headers = header_blob.decode("utf-8", errors="replace").split("\r\n")
        disposition = next(
            (line.split(":", 1)[1].strip() for line in headers if line.lower().startswith("content-disposition:")),
            "",
        )
        name, filename = _parse_content_disposition(disposition)
        if not name:
            continue
        if filename is not None:
            form.add_field(name, FormField(filename=filename, file_data=content))
        else:
            form.add_field(name, FormField(value=content.decode("utf-8", errors="replace")))
    return form


UrlopenFn = Callable[..., Any]


def openai_error(message: str, error_type: str, status: int) -> Tuple[int, Dict[str, Any]]:
    return status, {"error": {"message": message, "type": error_type, "code": None}}


def auth_error_response() -> Tuple[int, Dict[str, Any]]:
    return openai_error("invalid api key", "authentication_error", 401)


def _field_values(fs: MultipartForm, name: str) -> List[str]:
    if name not in fs:
        return []
    field = fs[name]
    if isinstance(field, list):
        return [item.value for item in field if item.value is not None]
    if field.value is None:
        return []
    return [field.value]


def _field_value(fs: MultipartForm, name: str, default: Optional[str] = None) -> Optional[str]:
    values = _field_values(fs, name)
    if not values:
        return default
    return values[0]


def parse_multipart_form(
    headers, rfile, content_length: int
) -> Tuple[Optional[MultipartForm], Optional[Tuple[int, Dict[str, Any]]]]:
    content_type = headers.get("Content-Type") or headers.get("content-type") or ""
    if not content_type.lower().startswith("multipart/form-data"):
        return None, openai_error(
            "Content-Type must be multipart/form-data",
            "invalid_request_error",
            400,
        )

    body = rfile.read(content_length)
    try:
        fs = _parse_multipart_body(content_type, body)
    except Exception as exc:
        return None, openai_error(
            f"invalid multipart form: {exc}",
            "invalid_request_error",
            400,
        )
    return fs, None


def _extract_file(fs: MultipartForm) -> Tuple[Optional[str], bytes]:
    if "file" not in fs:
        return None, b""
    field = fs["file"]
    if isinstance(field, list):
        if not field:
            return None, b""
        field = field[0]
    filename = field.filename or ""
    data = field.file.read() if field.file else b""
    return filename, data


def _extension_from_filename(filename: str) -> str:
    if not filename or "." not in filename:
        return ""
    return filename.rsplit(".", 1)[-1].lower()


def _parse_timestamp_granularities(fs: MultipartForm) -> List[str]:
    values: List[str] = []
    for key in ("timestamp_granularities[]", "timestamp_granularities"):
        values.extend(_field_values(fs, key))
    return [v for v in values if v in ("word", "segment")]


def _parse_temperature(raw: Optional[str]) -> Tuple[Optional[float], Optional[Tuple[int, Dict[str, Any]]]]:
    if raw is None or raw == "":
        return 0.0, None
    try:
        value = float(raw)
    except ValueError:
        return None, openai_error(
            "temperature must be a number",
            "invalid_request_error",
            400,
        )
    return value, None


def validate_transcription_request(
    fs: MultipartForm,
) -> Tuple[
    Optional[Dict[str, Any]],
    Optional[Tuple[int, Dict[str, Any]]],
]:
    filename, file_bytes = _extract_file(fs)
    if not file_bytes:
        return None, openai_error("file is required", "invalid_request_error", 400)

    if len(file_bytes) > OPENAI_STT_MAX_FILE_SIZE_BYTES:
        return None, openai_error(
            f"file exceeds {OPENAI_STT_MAX_FILE_SIZE_MB}MB limit",
            "invalid_request_error",
            413,
        )

    extension = _extension_from_filename(filename or "")
    if extension not in OPENAI_AUDIO_EXTENSIONS:
        return None, openai_error("unsupported audio format", "invalid_request_error", 422)

    model = _field_value(fs, "model")
    if not model:
        return None, openai_error("model is required", "invalid_request_error", 400)
    if model not in MODEL_MAP:
        return None, openai_error(
            f"model '{model}' not supported",
            "invalid_request_error",
            400,
        )

    temperature, temp_err = _parse_temperature(_field_value(fs, "temperature"))
    if temp_err:
        return None, temp_err

    response_format = _field_value(fs, "response_format", "json") or "json"
    allowed_formats = {"json", "text", "verbose_json", "srt", "vtt"}
    if response_format not in allowed_formats:
        return None, openai_error(
            f"response_format '{response_format}' not supported",
            "invalid_request_error",
            400,
        )

    language = _field_value(fs, "language")
    prompt = _field_value(fs, "prompt")
    timestamp_granularities = _parse_timestamp_granularities(fs)

    return {
        "filename": filename,
        "file_bytes": file_bytes,
        "extension": extension,
        "model": model,
        "whisper_model": MODEL_MAP[model],
        "language": language,
        "prompt": prompt,
        "temperature": temperature,
        "response_format": response_format,
        "timestamp_granularities": timestamp_granularities,
    }, None


def build_audio_data_uri(file_bytes: bytes, extension: str) -> str:
    mime = EXTENSION_MIMES.get(extension) or mimetypes.guess_type(f"file.{extension}")[0]
    if not mime:
        mime = "application/octet-stream"
    encoded = base64.b64encode(file_bytes).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def build_cog_input(parsed: Dict[str, Any]) -> Dict[str, Any]:
    data_uri = build_audio_data_uri(parsed["file_bytes"], parsed["extension"])
    return {
        "audio_file": data_uri,
        "whisper_model": parsed["whisper_model"],
        "language": parsed["language"],
        "initial_prompt": parsed["prompt"],
        "temperature": parsed["temperature"],
        "align_output": True,
        "diarization": False,
        "batch_size": 64,
        "vad_onset": 0.500,
        "vad_offset": 0.363,
        "language_detection_min_prob": 0,
        "language_detection_max_tries": 5,
        "hotwords": None,
        "huggingface_access_token": None,
        "debug": False,
    }


def call_cog_sync(
    payload: Dict[str, Any],
    *,
    cog_url: str = COG_URL,
    timeout: int = OPENAI_STT_TIMEOUT_SECONDS,
    urlopen_fn: Optional[UrlopenFn] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[Tuple[int, Dict[str, Any]]], Optional[Dict[str, Any]]]:
    """Return (cog_output, error_response, request_meta). request_meta for tests."""
    urlopen = urlopen_fn or urllib.request.urlopen
    body = json.dumps({"input": payload}).encode("utf-8")
    req = urllib.request.Request(
        f"{cog_url}/predictions",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    request_meta = {
        "url": f"{cog_url}/predictions",
        "headers": dict(req.header_items()),
        "body": body,
        "async_header": req.has_header("Prefer"),
    }
    try:
        with urlopen(req, timeout=timeout) as res:
            data = json.loads(res.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        return None, openai_error(
            f"transcription failed: {detail or exc.reason}",
            "server_error",
            500,
        ), request_meta
    except (TimeoutError, socket.timeout, urllib.error.URLError) as exc:
        if isinstance(exc, urllib.error.URLError) and not isinstance(
            getattr(exc, "reason", None), (TimeoutError, socket.timeout)
        ):
            return None, openai_error(
                f"transcription failed: {exc.reason}",
                "server_error",
                500,
            ), request_meta
        return None, openai_error("transcription timed out", "server_error", 504), request_meta
    except Exception as exc:
        if "timed out" in str(exc).lower():
            return None, openai_error("transcription timed out", "server_error", 504), request_meta
        return None, openai_error(
            f"transcription failed: {exc}",
            "server_error",
            500,
        ), request_meta

    status = data.get("status")
    if status == "failed":
        detail = data.get("error") or "unknown error"
        return None, openai_error(f"transcription failed: {detail}", "server_error", 500), request_meta
    if status != "succeeded":
        return None, openai_error(
            f"transcription failed: unexpected status {status!r}",
            "server_error",
            500,
        ), request_meta

    output = data.get("output")
    if not isinstance(output, dict):
        return None, openai_error("transcription failed: missing output", "server_error", 500), request_meta
    return output, None, request_meta


def _join_segment_text(segments: List[Dict[str, Any]]) -> str:
    parts = [seg.get("text", "").strip() for seg in segments if seg.get("text")]
    return " ".join(parts).strip()


def _output_duration(segments: List[Dict[str, Any]]) -> float:
    if not segments:
        return 0.0
    return float(max(seg.get("end", 0.0) for seg in segments))


def _flatten_words(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    words: List[Dict[str, Any]] = []
    for seg in segments:
        for word in seg.get("words") or []:
            words.append(
                {
                    "word": word.get("word", ""),
                    "start": word.get("start", 0.0),
                    "end": word.get("end", 0.0),
                }
            )
    return words


def convert_to_json(cog_output: Dict[str, Any]) -> Dict[str, Any]:
    segments = cog_output.get("segments") or []
    return {"text": _join_segment_text(segments)}


def convert_to_text(cog_output: Dict[str, Any]) -> str:
    segments = cog_output.get("segments") or []
    return _join_segment_text(segments)


def convert_to_verbose_json(
    cog_output: Dict[str, Any],
    *,
    temperature: float = 0.0,
    timestamp_granularities: Optional[List[str]] = None,
) -> Dict[str, Any]:
    segments = cog_output.get("segments") or []
    granularities = timestamp_granularities or []
    include_segments = "segment" in granularities or not granularities
    include_words = "word" in granularities

    text = _join_segment_text(segments)
    result: Dict[str, Any] = {
        "task": "transcribe",
        "language": cog_output.get("detected_language") or "en",
        "duration": _output_duration(segments),
        "text": text,
    }

    if include_segments:
        result["segments"] = [
            {
                "id": idx,
                "seek": 0,
                "start": seg.get("start", 0.0),
                "end": seg.get("end", 0.0),
                "text": seg.get("text", ""),
                "tokens": [],
                "temperature": temperature,
                "avg_logprob": 0.0,
                "compression_ratio": 1.0,
                "no_speech_prob": 0.0,
            }
            for idx, seg in enumerate(segments)
        ]

    if include_words:
        result["words"] = _flatten_words(segments)

    return result


def _format_timestamp_srt(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis == 1000:
        secs += 1
        millis = 0
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _format_timestamp_vtt(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis == 1000:
        secs += 1
        millis = 0
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def convert_to_srt(cog_output: Dict[str, Any]) -> str:
    segments = cog_output.get("segments") or []
    blocks: List[str] = []
    for idx, seg in enumerate(segments, start=1):
        start = _format_timestamp_srt(float(seg.get("start", 0.0)))
        end = _format_timestamp_srt(float(seg.get("end", 0.0)))
        text = (seg.get("text") or "").strip()
        blocks.append(f"{idx}\n{start} --> {end}\n{text}\n")
    return "\n".join(blocks)


def convert_to_vtt(cog_output: Dict[str, Any]) -> str:
    segments = cog_output.get("segments") or []
    lines = ["WEBVTT", ""]
    for seg in segments:
        start = _format_timestamp_vtt(float(seg.get("start", 0.0)))
        end = _format_timestamp_vtt(float(seg.get("end", 0.0)))
        text = (seg.get("text") or "").strip()
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def convert_cog_output(
    cog_output: Dict[str, Any],
    response_format: str,
    *,
    temperature: float = 0.0,
    timestamp_granularities: Optional[List[str]] = None,
) -> Tuple[Dict[str, str], bytes]:
    if response_format == "json":
        return (
            {"Content-Type": "application/json"},
            json.dumps(convert_to_json(cog_output)).encode("utf-8"),
        )
    if response_format == "text":
        return (
            {"Content-Type": "text/plain; charset=utf-8"},
            convert_to_text(cog_output).encode("utf-8"),
        )
    if response_format == "verbose_json":
        payload = convert_to_verbose_json(
            cog_output,
            temperature=temperature,
            timestamp_granularities=timestamp_granularities,
        )
        return (
            {"Content-Type": "application/json"},
            json.dumps(payload).encode("utf-8"),
        )
    if response_format == "srt":
        return (
            {"Content-Type": "text/plain; charset=utf-8"},
            convert_to_srt(cog_output).encode("utf-8"),
        )
    if response_format == "vtt":
        return (
            {"Content-Type": "text/vtt; charset=utf-8"},
            convert_to_vtt(cog_output).encode("utf-8"),
        )
    return (
        {"Content-Type": "application/json"},
        json.dumps(openai_error("unsupported response format", "invalid_request_error", 400)[1]).encode(
            "utf-8"
        ),
    )


def handle_transcription_request(
    headers,
    rfile,
    content_length: int,
    *,
    cog_url: str = COG_URL,
    timeout: int = OPENAI_STT_TIMEOUT_SECONDS,
    urlopen_fn: Optional[UrlopenFn] = None,
) -> Tuple[int, Dict[str, str], bytes, Dict[str, Any]]:
    """Process an OpenAI STT request.

    Returns (status_code, response_headers, body_bytes, meta) where meta carries
    logging fields and optional request_meta from the Cog call.
    """
    meta: Dict[str, Any] = {
        "bytes": 0,
        "model": None,
        "response_format": None,
        "request_meta": None,
    }

    if content_length is None or content_length < 0:
        status, err = openai_error(
            "Content-Length must be a non-negative integer",
            "invalid_request_error",
            400,
        )
        return status, {"Content-Type": "application/json"}, json.dumps(err).encode("utf-8"), meta

    fs, parse_err = parse_multipart_form(headers, rfile, content_length)
    if parse_err:
        status, err = parse_err
        return status, {"Content-Type": "application/json"}, json.dumps(err).encode("utf-8"), meta

    parsed, validation_err = validate_transcription_request(fs)
    if validation_err:
        status, err = validation_err
        return status, {"Content-Type": "application/json"}, json.dumps(err).encode("utf-8"), meta

    assert parsed is not None
    meta["bytes"] = len(parsed["file_bytes"])
    meta["model"] = parsed["model"]
    meta["response_format"] = parsed["response_format"]

    cog_input = build_cog_input(parsed)
    cog_output, cog_err, request_meta = call_cog_sync(
        cog_input,
        cog_url=cog_url,
        timeout=timeout,
        urlopen_fn=urlopen_fn,
    )
    meta["request_meta"] = request_meta
    if cog_err:
        status, err = cog_err
        return status, {"Content-Type": "application/json"}, json.dumps(err).encode("utf-8"), meta

    assert cog_output is not None
    resp_headers, body = convert_cog_output(
        cog_output,
        parsed["response_format"],
        temperature=float(parsed["temperature"] or 0.0),
        timestamp_granularities=parsed["timestamp_granularities"],
    )
    return 200, resp_headers, body, meta
