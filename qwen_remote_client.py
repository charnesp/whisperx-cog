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
import re
import struct
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable
from urllib.parse import urlsplit

logger = logging.getLogger("qwen_remote")

# --- request constants (B1/B6: pinned request identity) --------------------
REMOTE_META_MAX_TOKENS = 120  # bounded, NOT configurable: window <= 30 s ASR text
DEFAULT_SYSTEM_PROMPT = (
    "Transcribe the input audio exactly as spoken; reply with the "
    "transcription text only."
)
LANGUAGE_INSTRUCTION = "Reply in the language whose ISO code is %r."  # %s: caller ISO code
CONTEXT_INSTRUCTION = "Terms and names expected: %s."  # hotwords (B6 language prefix)
_ASR_PREFIX = "<asr_text>"
# Server reply prefix grammar: lowercase `language <token>` (e.g. `language French`).
# Lowercase on purpose: a normal sentence starting with the capitalised word
# "Language" is structured output, not the model's prefix.
_LANGUAGE_PREFIX = re.compile(r"^language\s+([A-Za-z][A-Za-z0-9_-]*)")
# Qwen3-ASR emits a language NAME in the reply prefix (`language French<asr_text>`)
# while the local path reports the ISO code (`fr`). Normalise to the local
# contract so remote/local results are comparable. Standard Whisper language set.
_LANGUAGE_NAME_TO_ISO = {
    "afrikaans": "af", "amharic": "am", "arabic": "ar", "assamese": "as",
    "azerbaijani": "az", "bashkir": "ba", "belarusian": "be", "bulgarian": "bg",
    "bengali": "bn", "tibetan": "bo", "breton": "br", "bosnian": "bs",
    "catalan": "ca", "czech": "cs", "welsh": "cy", "danish": "da",
    "german": "de", "greek": "el", "english": "en", "spanish": "es",
    "estonian": "et", "basque": "eu", "persian": "fa", "finnish": "fi",
    "faroese": "fo", "french": "fr", "galician": "gl", "gujarati": "gu",
    "hausa": "ha", "hawaiian": "haw", "hebrew": "he", "hindi": "hi",
    "croatian": "hr", "haitian creole": "ht", "hungarian": "hu",
    "armenian": "hy", "indonesian": "id", "icelandic": "is", "italian": "it",
    "japanese": "ja", "javanese": "jw", "georgian": "ka", "kazakh": "kk",
    "khmer": "km", "kannada": "kn", "korean": "ko", "latin": "la",
    "luxembourgish": "lb", "lingala": "ln", "lao": "lo", "lithuanian": "lt",
    "latvian": "lv", "malagasy": "mg", "maori": "mi", "macedonian": "mk",
    "malayalam": "ml", "mongolian": "mn", "marathi": "mr", "malay": "ms",
    "maltese": "mt", "myanmar": "my", "nepali": "ne", "dutch": "nl",
    "nynorsk": "nn", "norwegian": "no", "occitan": "oc", "punjabi": "pa",
    "polish": "pl", "pashto": "ps", "portuguese": "pt", "romanian": "ro",
    "russian": "ru", "sanskrit": "sa", "sindhi": "sd", "sinhala": "si",
    "slovak": "sk", "slovenian": "sl", "shona": "sn", "somali": "so",
    "albanian": "sq", "serbian": "sr", "sundanese": "su", "swedish": "sv",
    "swahili": "sw", "tamil": "ta", "telugu": "te", "tajik": "tg",
    "thai": "th", "turkmen": "tk", "tagalog": "tl", "turkish": "tr",
    "tatar": "tt", "ukrainian": "uk", "urdu": "ur", "uzbek": "uz",
    "vietnamese": "vi", "yiddish": "yi", "yoruba": "yo", "chinese": "zh",
    "cantonese": "yue",
}
HTTP_STATUS_OK = 200
_ERROR_BODY_SNIPPET_BYTES = 96  # truncate upstream bodies in error messages


# QwenRemoteError + the shared clamp resolver live in qwen_remote.py (sibling,
# group 2 owner); import them so every qwen_remote_client failure passes
# isinstance checks against the SAME class used for config resolution.
import qwen_remote  # noqa: E402
from qwen_remote import QwenRemoteError  # noqa: E402,F401


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
        parts.append(LANGUAGE_INSTRUCTION % str(language))
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


_URL_USERINFO = re.compile(r"([a-zA-Z][a-zA-Z0-9+.\-]*://)[^\s/@]+(?=@)")


def _sanitize_transport_reason(reason, url: str = "") -> str:
    """Strip credentials/URLs from a socket reason before it enters a message.

    A urllib/OS reason can embed the configured URL (userinfo password) or an
    address; the message contract forbids both (design decision 10). The exact
    configured URL is replaced first, then any userinfo credentials sitting
    before an `@` in a URL are redacted.
    """
    text = str(reason)
    if url:
        text = text.replace(url, "<redacted-url>")
    return _URL_USERINFO.sub(r"\1<redacted>", text)


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
                "remote engine did not answer within the per-request timeout "
                "(%s)" % _sanitize_transport_reason(reason, url)
            ) from exc
        raise RemoteConnectionError(
            "cannot reach the remote engine (%s)"
            % _sanitize_transport_reason(reason, url)
        ) from exc
    except TimeoutError as exc:  # read-side timeout raised bare
        raise RemoteTimeoutError(
            "remote engine read timed out within the per-request timeout (%s)"
            % _sanitize_transport_reason(exc, url)
        ) from exc
    except OSError as exc:
        raise RemoteConnectionError(
            "transport failure to the remote engine (%s)"
            % _sanitize_transport_reason(exc, url)
        ) from exc


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
        match = _LANGUAGE_PREFIX.match(text.lstrip())
        if match and _looks_like_server_language(match.group(1)):
            raise RemoteResponseParserError(
                "reply carries a server 'language X' prefix but no "
                f"{_ASR_PREFIX} marker: unparseable ASR reply"
            )
        return text.strip()  # structured output path: content is the text
    after = text[index + len(_ASR_PREFIX):].strip()
    return after  # language None / silence -> "" (valid empty segment)


def _looks_like_server_language(token) -> bool:
    """True when a `language X` head really is the Qwen3-ASR reply prefix.

    `language French` / `language None` are server prefixes; structured output
    that merely starts with the word 'language' (e.g. 'language models are
    useful') is not and must be kept as-is. Unknown tokens are therefore NOT
    treated as the prefix (narrow rejection, no over-refusal).
    """
    head = str(token).strip().lower()
    if head == "none":
        return True
    return normalize_language_code(token) is not None


def normalize_language_code(token):
    """Server prefix language token -> ISO code (None for 'None'/empty).

    Qwen3-ASR emits a language NAME (e.g. `language French<asr_text>...`) while
    the local path reports the ISO code (e.g. `fr`): normalise to the local
    contract. A token that already looks like an ISO code passes through
    lower-cased; an unknown name yields None (no made-up code).
    """
    if token is None:
        return None
    text = str(token).strip()
    if not text or text.lower() == "none":
        return None
    lowered = text.lower()
    if lowered in _LANGUAGE_NAME_TO_ISO:
        return _LANGUAGE_NAME_TO_ISO[lowered]
    if 2 <= len(text) <= 3 and text.isalpha():
        return lowered
    return None


def parse_asr_language(content):
    """Server-detected language from a reply content, normalised to ISO.

    Returns None when the reply carries no `language X` prefix (structured
    output), a `language None` marker, or a token we cannot map. Lenient by
    design: text extraction/validation is `parse_asr_content`'s job.
    """
    if content is None:
        return None
    match = _LANGUAGE_PREFIX.match(str(content).lstrip())
    if not match:
        return None
    return normalize_language_code(match.group(1))


def parse_response(status: int, body: str) -> str:
    """vLLM 0.30.0 chat.completion envelope -> transcribed text of choice[0]."""
    return parse_response_meta(status, body)[0]


def parse_response_meta(status: int, body: str):
    """Like `parse_response` but also returns the detected ISO language."""
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
    content = message.get("content")
    text = parse_asr_content(content, finish_reason=finish_reason)
    return text, parse_asr_language(content)


# --- one window -------------------------------------------------------------
def transcribe_window(audio_url, config: dict, language=None, context=None) -> str:
    """ONE window -> ONE POST -> parsed text (G3 workflow, GET-safe)."""
    return transcribe_window_meta(audio_url, config, language=language, context=context)[0]


def transcribe_window_meta(audio_url, config: dict, language=None, context=None):
    """ONE window -> ONE POST -> (parsed text, detected ISO language)."""
    payload = build_chat_payload(_audio_url(audio_url), config["model"], language=language, context=context)
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    status, text = perform_http_post(
        config["chat_url"], body, {"Content-Type": "application/json"}, config.get("timeout_s")
    )
    return parse_response_meta(status, text)


def clamp_remote_pool_size(requested, fallback: int = 4, cap: int = 8) -> int:
    """Concurrency bound: the SAME resolver as predict.resolve_qwen_batch_size.

    Delegates to the shared pure helper `qwen_remote.resolve_batch_size`
    (importing predict is GPU-forbidden here), so the remote pool size can
    never drift from the local batch clamp: absent/invalid/<=0 -> default,
    else clamped to [1, cap].
    """
    return qwen_remote.resolve_batch_size(requested, fallback, cap)


# --- structured log (P1-F: one `qwen-remote:` line per transcription) --------
def _remote_host(config) -> str:
    """Host-only label for logs: never credentials, never port, never full URL."""
    raw = ""
    if isinstance(config, dict):
        raw = config.get("base_url") or config.get("chat_url") or ""
    try:
        return urlsplit(str(raw)).hostname or ""
    except ValueError:
        return ""


def _log_remote_transcription(config, window_count, status, elapsed_s):
    """Emit ONE grep-able record. Never the URL, audio, hotwords or context."""
    logger.info(
        "qwen-remote: host=%s windows=%d status=%s duration_s=%.3f",
        _remote_host(config),
        window_count,
        status,
        elapsed_s,
    )


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

    Backwards-compatible view over `transcribe_windows_meta` (texts only).
    """
    texts, _languages = transcribe_windows_meta(
        windows,
        config,
        batch_size=batch_size,
        language=language,
        context=context,
        executor=executor,
    )
    return texts


def transcribe_windows_meta(
    windows,
    config: dict,
    batch_size=None,
    language=None,
    context=None,
    executor: Callable[..., ThreadPoolExecutor] | None = None,
):
    """All windows -> (texts, detected ISO languages), in INPUT (VAD) order.

    Concurrency == clamped batch_size (predict clamp semantics: default 4,
    cap 8, <=0/invalid -> default). ONE failed window fails the WHOLE
    prediction with its typed error: pending futures are cancelled
    (cancel_futures=True), in-flight results discarded, no partial fused
    output, no retry, no local fallback. The executor is shut down EXPLICITLY
    (never via a `with` block, whose __exit__ would re-run shutdown(wait=True)
    and undo the cancellation the typed error is meant to honour).
    """
    if not windows:
        return [], []
    started = time.time()
    worker_count = clamp_remote_pool_size(batch_size)
    texts: list = [None] * len(windows)
    languages: list = [None] * len(windows)
    pool_executor = executor or ThreadPoolExecutor
    pool = pool_executor(max_workers=worker_count)
    failed = False
    try:
        future_to_index = {
            pool.submit(
                transcribe_window_meta, window, config, language=language, context=context
            ): index
            for index, window in enumerate(windows)
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                texts[index], languages[index] = future.result()
            except QwenRemoteError:
                failed = True
                pool.shutdown(wait=False, cancel_futures=True)
                raise
            except Exception as exc:  # unexpected: wrap typed, same policy
                failed = True
                pool.shutdown(wait=False, cancel_futures=True)
                raise RemoteConnectionError(
                    f"unexpected failure on window #{index}: {exc}"
                ) from exc
    finally:
        if not failed:
            pool.shutdown(wait=True)
        _log_remote_transcription(
            config, len(windows), "failed" if failed else "ok", time.time() - started
        )
    if any(t is None for t in texts):
        raise RemoteConnectionError(
            "a window produced no result (executor shutdown raced)"
        )
    return texts, languages
