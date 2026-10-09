"""qwen_remote_client — injectable stdlib-urllib remote client (group 3-5).

Splits from qwen_remote.py (config/env ownership stays in qwen_remote.py —
resolve_remote_config is NOT duplicated here): this module owns

- encode_wav_data_uri: deterministic float32 -> mono 16 kHz PCM16 wav ->
  data:audio/wav;base64 URI (no ffmpeg at runtime; fixed header);
- parse_asr_content (G4): the pure reply parser (prefix, None/empty,
  typed errors, first-choice, finish_reason=="length");
- transcribe_windows (G5): bounded-pool orchestrator reusing the clamped
  batch_size from predict.resolve_qwen_batch_size; VAD order preserved;
  one failed window fails the whole prediction (zero retries, explicit
  typed errors);
- perform_http_post: urllib seam — ProxyHandler({}) EXPLICIT (internal
  call must never honor HTTP(S)_PROXY; bandit B310 silenced by design
  decision: scheme is validated by qwen_remote.resolve_remote_config,
  https/http only, and the URL comes from operator env, never client
  input); per-request connect+read timeout; thread-safe (no module state).

Error taxonomy (design decision 10): every failure raises a
QwenRemoteError subclass whose message starts with
"QwenRemoteError: <category> —" (config/connection/timeout/http_status/
parse). Messages NEVER carry the full URL, the audio payload, or the
whole reply body (truncated to a prefix).
"""
from __future__ import annotations

import base64
import json
import logging
import struct
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

logger = logging.getLogger("qwen_remote")

# --- request constants (B1/B6: pinned request identity) --------------------
REMOTE_META_MAX_TOKENS = 120  # bounded, NOT configurable: window <= 30 s ASR text
REMOTE_TRANSCRIBE_TIMEOUT = 8  # unused timeout fallback; real timeout comes from config
REMOTE_SUFFIX = "/chat/completions"
DEFAULT_SYSTEM_PROMPT = (
    "Transcribe the input audio exactly as spoken; reply with the "
    "transcription text only."
)
LANGUAGE_INSTRUCTION = "Reply in French (ISO code 'fr')."
CONTEXT_INSTRUCTION = "Terms and names expected: %s."  # hotwords (B6 language prefix)
_ASR_PREFIX = "<asr_text>"
HTTP_STATUS_OK = 200
_ERROR_BODY_SNIPPET_BYTES = 96  # truncate upstream bodies in error messages

BANDIT_B310_NOCOSE = "# nosec B106/B310 — internal URL: scheme enforced upstream"


# QwenRemoteError lives in qwen_remote.py (sibling, group 2 owner);
# re-export it here so every qwen_remote_client failure passes isinstance
# checks against the SAME class used for config resolution.
from qwen_remote import QwenRemoteError  # noqa: E402,F401


class RemoteConfigError(QwenRemoteError):
    """QWEN_BACKEND/QWEN_REMOTE_* misconfiguration (category: config)."""
    category = "config"

    def __init__(self, message: str = ""):
        super().__init__(_q_message(message, self.category))


class RemoteConnectionError(QwenRemoteError):
    """DNS / refused / unreachable (category: connection)."""
    category = "connection"

    def __init__(self, message: str = ""):
        super().__init__(_q_message(message, self.category))


class RemoteTimeoutError(QwenRemoteError):
    """Per-request connect+read timeout (category: timeout)."""
    category = "timeout"

    def __init__(self, message: str = ""):
        super().__init__(_q_message(message, self.category))


class RemoteHTTPStatusError(QwenRemoteError):
    """Non-2xx from the engine, zero retries (category: http_status)."""
    category = "http_status"

    def __init__(self, status: int, body_prefix: str = ""):
        snippet = body_prefix.strip()[:_ERROR_BODY_SNIPPET_BYTES]
        suffix = f" body_prefix={snippet!r}" if snippet else ""
        super().__init__(
            _q_message(
                f"remote engine returned HTTP {status} (retry policy: NONE){suffix}",
                self.category,
            )
        )


class RemoteResponseParserError(QwenRemoteError):
    """Unusable reply body (missing/empty/asr-malformed) — category: parse."""
    category = "parse"

    def __init__(self, message: str = ""):
        super().__init__(_q_message(message, self.category))


def _q_message(text: str, category: str | None = None) -> str:
    """Normalize a raw message into the stable grep-able contract form:
    "QwenRemoteError: <category> - <detail>" (category optional for the
    root class). Message bodies NEVER embed the URL or audio payload."""
    head = "QwenRemoteError:" if category is None else f"QwenRemoteError: {category}"
    detail = f" - {text}" if text else ""
    return f"{head}{detail}"


# --- wav encoder (deterministic, no shared state) ---------------------------
def encode_wav_pcm16(samples) -> bytes:
    """Float32 samples in [-1.0, 1.0] -> mono 16 kHz PCM16 RIFF/WAVE bytes.

    Deterministic pure function: same list -> same bytes (no wall clock,
    no RNG, no environment). Clipping saturates (|[4.0]| -> 32767).
    """
    ints = []
    for value in samples:
        iv = int(round(32767.0 * float(value)))
        if iv > 32767:
            iv = 32767
        elif iv < -32767:
            iv = -32767
        ints.append(iv)
    pcm = struct.pack("<%dh" % len(ints), *ints)
    fmt = struct.pack("<I", 16) + struct.pack("<HHIIHH", 1, 1, 16000, 32000, 2, 16)
    data_size = len(pcm)
    riff_size = 4 + (8 + 16) + (8 + data_size)  # WAVE + fmt(4+16) + data(8+n)
    return b"".join(
        (
            b"RIFF",
            struct.pack("<I", riff_size),
            b"WAVE",
            b"fmt ",
            fmt,
            b"data",
            struct.pack("<I", data_size),
            pcm,
        )
    )


def encode_wav_data_uri(samples) -> str:
    """Full data:audio/wav;base64 URI for the chat multimodal payload."""
    wav = encode_wav_pcm16(samples)
    return "data:audio/wav;base64," + base64.b64encode(wav).decode("ascii")


def _audio_url(window) -> str:
    if isinstance(window, str):
        return window  # already a data URI (window pre-encoded upstream)
    return encode_wav_data_uri(window)


# --- request assembly (B1: model/temperature/max_tokens, B6: language) ------
def _system_message(language=None, context=None) -> str:
    parts = [DEFAULT_SYSTEM_PROMPT]
    if language:
        parts.append(LANGUAGE_INSTRUCTION.replace("'fr'", repr(str(language))))
    if context:
        parts.append(CONTEXT_INSTRUCTION % context)
    return " ".join(parts)


def build_chat_payload(audio_url: str, model: str, language=None, context=None) -> dict:
    """One chat/completions body for ONE window (multi-modal audio_url)."""
    return {
        "model": model,
        "temperature": 0,
        "max_tokens": REMOTE_META_MAX_TOKENS,
        "stream": False,
        "messages": [
            {"role": "system", "content": _system_message(language, context)},
            {
                "role": "user",
                "content": [{"type": "audio_url", "audio_url": {"url": audio_url}}],
            },
        ],
    }


# --- HTTP seam (G3: injectable, ProxyHandler({})-explicit, thread-safe) -----
def _build_opener():
    """ProxyHandler({}) is the DECISION: no env proxies for this call.

    Bandit B310: urllib with a user-controlled URL is normally flagged;
    here the URL comes ONLY from QWEN_REMOTE_BASE_URL (operator env,
    scheme http/https enforced by qwen_remote.resolve_remote_config),
    never from client input — silencing is justified, see design D5.
    """
    proxy_disabled = urllib.request.ProxyHandler({})
    return urllib.request.build_opener(
        proxy_disabled,  # nosec B310 — scheme validated in resolve_remote_config
    )


def perform_http_post(url: str, payload: bytes, headers: dict, timeout=None):
    """POST and return (status, text). Typed errors on all failure shapes."""
    request = urllib.request.Request(url, data=payload, method="POST")
    for key, value in headers.items():
        request.add_header(key, value)
    opener = _build_opener()
    try:
        with opener.open(request, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = b""  # a non-2xx body is best-effort (truncated into the message)
        try:
            if isinstance(getattr(exc, "fp", None), bytes):
                body = exc.fp
            else:
                body = exc.read()
        except Exception:
            body = b""
        if isinstance(body, str):
            text = body
        else:
            try:
                text = body.decode("utf-8", errors="replace")
            except Exception:
                text = ""
        raise RemoteHTTPStatusError(exc.code, text) from exc
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        if isinstance(reason, TimeoutError) or "timed out" in str(reason).lower():
            raise RemoteTimeoutError(
                f"remote engine did not answer within the per-request timeout ({reason})"
            ) from exc
        raise RemoteConnectionError(f"cannot reach the remote engine ({reason})") from exc
    except TimeoutError as exc:  # read-side timeout raised bare
        raise RemoteTimeoutError(
            f"remote engine read timed out within the per-request timeout ({exc})"
        ) from exc
    except OSError as exc:
        raise RemoteConnectionError(f"transport failure to the remote engine ({exc})") from exc


# --- reply parser (G4) ------------------------------------------------------
def parse_asr_content(content, finish_reason=None) -> str:
    """Extract the transcribed text from ONE choice's content.

    Rules (spec `Remote response parsing`):
    - `language X<asr_text>text` -> text only (prefix never leaks);
    - content without the prefix (structured output) -> as-is stripped;
    - `language None<asr_text>` / empty transcription -> "" (empty segment,
      same rule as the local path);
    - missing/whitespace-only content -> RemoteResponseParserError;
    - finish_reason == "length" -> RemoteResponseParserError (never silently
      truncated text).
    The reply text NEVER influences timestamps.
    """
    if finish_reason is not None and str(finish_reason).strip().lower() == "length":
        raise RemoteResponseParserError(
            "reply hit the max_tokens ceiling (finish_reason=length): "
            "transcription would be silently truncated — window fails"
        )
    if content is None:
        raise RemoteResponseParserError(
            "reply carries no content (choices[0].message.content missing)"
        )
    text = str(content)
    if not text.strip():
        raise RemoteResponseParserError(
            "reply content is empty: the engine produced no transcription"
        )
    index = text.find(_ASR_PREFIX)
    if index < 0:
        return text.strip()  # structured output path: content is the text
    after = text[index + len(_ASR_PREFIX):].strip()
    return after  # language None / silence -> "" (valid empty segment)


def parse_response(status: int, body: str) -> str:
    """vLLM 0.30.0 chat.completion envelope -> transcribed text of choice[0]."""
    if status != HTTP_STATUS_OK:
        raise RemoteHTTPStatusError(status, body[:_ERROR_BODY_SNIPPET_BYTES])
    try:
        payload = json.loads(body)
    except (ValueError, TypeError) as exc:
        raise RemoteResponseParserError(
            f"reply is not valid JSON chat.completion envelope ({exc})"
        ) from exc
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if not choices:
        raise RemoteResponseParserError(
            "chat.completion envelope carries no choices[] (empty_completion)"
        )
    choice = choices[0]
    finish_reason = choice.get("finish_reason")
    message = choice.get("message") or {}
    return parse_asr_content(message.get("content"), finish_reason=finish_reason)


# --- one window -------------------------------------------------------------
def transcribe_window(audio_url, config: dict, language=None, context=None) -> str:
    """ONE window -> ONE POST -> parsed text (G3 workflow, GET-safe)."""
    payload = build_chat_payload(_audio_url(audio_url), config["model"], language=language, context=context)
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    status, text = perform_http_post(
        config["chat_url"], body, {"Content-Type": "application/json"}, config.get("timeout_s")
    )
    return parse_response(status, text)


def clamp_remote_pool_size(requested, fallback: int = 4, cap: int = 8) -> int:
    """Concurrency bound: reuse the PREDICTOR clamp, never trust raw input.

    predict.resolve_qwen_batch_size already logs/normalizes; this is a
    local mirror (importing predict is GPU-forbidden here) with the same
    semantics: absent/invalid/<=0 -> default, > cap -> cap.
    """
    if requested is None:
        return fallback
    if isinstance(requested, bool) or not isinstance(requested, int):
        if isinstance(requested, float) and requested.is_integer():
            requested = int(requested)
        else:
            return fallback
    if requested <= 0:
        return fallback
    return min(requested, cap)


# --- pool orchestrator (G5) -------------------------------------------------
def transcribe_windows(
    windows,
    config: dict,
    batch_size=None,
    language=None,
    context=None,
    executor: Callable[..., ThreadPoolExecutor] | None = None,
):
    """All windows -> all texts, in INPUT (VAD) order, zero retries.

    Concurrency == clamped batch_size (predict clamp semantics: default 4,
    cap 8, <=0/invalid -> default). ONE failed window fails the WHOLE
    prediction with its typed error: pending futures are cancelled
    (cancel_futures=True), in-flight results discarded, no partial fused
    output, no retry, no local fallback.
    """
    if not windows:
        return []
    worker_count = clamp_remote_pool_size(batch_size)
    texts: list = [None] * len(windows)
    pool_executor = executor or ThreadPoolExecutor
    with pool_executor(max_workers=worker_count) as pool:
        future_to_index = {
            pool.submit(
                transcribe_window, window, config, language=language, context=context
            ): index
            for index, window in enumerate(windows)
        }
        try:
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    texts[index] = future.result()
                except QwenRemoteError:
                    pool.shutdown(wait=False, cancel_futures=True)
                    raise
                except Exception as exc:  # unexpected: wrap typed, same policy
                    pool.shutdown(wait=False, cancel_futures=True)
                    raise RemoteConnectionError(
                        f"unexpected failure on window #{index}: {exc}"
                    ) from exc
        finally:
            pass  # `with` closes the pool; cancel_futures already honored
    if any(t is None for t in texts):
        raise RemoteConnectionError(
            "a window produced no result (executor shutdown raced)"
        )
    return texts
