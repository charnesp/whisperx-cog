"""qwen_split — quiet-point chunking for the remote ASR path (pure numpy).

Implements the method documented by vLLM ("Chunking Long Audio for
Transcription"): instead of cutting every ``max_window_s`` seconds at an
arbitrary boundary — which can slice a word in half and wreck the
transcription — the cut is placed on the QUIETEST point found in the second
preceding the limit (RMS energy over 1600-sample frames, ~100 ms at 16 kHz).

Properties the remote path relies on:

- deterministic: same samples -> same plan (no clock, no RNG, no env);
- total coverage: the windows tile the signal, contiguous, no overlap, no
  sample lost or duplicated;
- absolute offsets: each window carries its (start, end) in seconds derived
  from its sample indices, so timestamps never depend on the engine reply;
- GPU-free and framework-free: numpy only (no whisperx, no pyannote, no torch),
  so the windowing is unit-testable without a device.

``split_window`` is the same quiet-point idea applied to ONE window: it cuts
near the midpoint, used by the client to salvage a window the engine keeps
rejecting (per-window recovery).
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

SAMPLE_RATE = 16000  # Hz, mono (the local pipeline contract)
MAX_WINDOW_S = 30.0  # seconds per remote request
QUIET_SEARCH_S = 1.0  # search the last second before the limit for the cut
ENERGY_FRAME_SAMPLES = 1600  # ~100 ms at 16 kHz, the RMS analysis frame


class Window(NamedTuple):
    """One chunk of audio: absolute bounds in seconds + its samples."""

    start: float
    end: float
    samples: np.ndarray


def _rms(samples: np.ndarray, start: int, frame: int) -> float:
    """Root-mean-square energy of one frame (float64 accumulation)."""
    block = samples[start:start + frame]
    if block.size == 0:
        return 0.0
    values = block.astype(np.float64, copy=False)
    return float(np.sqrt(np.mean(values * values)))


def frame_rms_curve(
    samples,
    start: int,
    count: int,
    energy_frame: int = ENERGY_FRAME_SAMPLES,
) -> np.ndarray:
    """RMS energy of ``count`` consecutive frames starting at ``start`` (vectorised).

    One pass over the signal: ``cumsum(x*x)`` differences give the sum of every
    sliding frame at once (the former per-frame Python loop cost ~16 000
    iterations per 30 s window, 43.5 s CPU per hour of audio). Same values as
    ``_rms`` up to float64 rounding (~1e-11), so the chosen cut index is
    unchanged. Returns a float64 array, empty when no full frame fits.
    """
    audio = np.asarray(samples)
    frame = max(1, int(energy_frame))
    start = max(0, int(start))
    total = int(audio.size)
    count = min(max(0, int(count)), max(0, total - start - frame + 1))
    if count <= 0:
        return np.empty(0, dtype=np.float64)
    values = np.asarray(audio[start:start + count + frame - 1], dtype=np.float64)
    squares = values * values
    cumulative = np.concatenate(
        (np.zeros(1, dtype=np.float64), np.cumsum(squares, dtype=np.float64))
    )
    sums = cumulative[frame:frame + count] - cumulative[:count]
    return np.sqrt(np.maximum(sums, 0.0) / float(frame))


def quietest_index(
    samples: np.ndarray,
    lo: int,
    hi: int,
    energy_frame: int = ENERGY_FRAME_SAMPLES,
) -> int:
    """Cut index of the quietest energy frame in ``[lo, hi]``.

    Every frame START inside ``[lo, hi - energy_frame]`` is scored; the cut is
    placed at the CENTRE of the winning frame, so a short silence is cut in its
    middle rather than on one of its edges. Ties resolve to the lowest index
    (deterministic). The result is always in ``]lo, hi]``.
    """
    audio = np.asarray(samples)
    total = int(audio.size)
    lo = max(0, int(lo))
    hi = min(int(hi), total)
    if hi <= lo:
        return hi
    frame = max(1, int(energy_frame))
    last = max(lo, hi - frame)
    curve = frame_rms_curve(audio, lo, last - lo + 1, frame)
    # A span shorter than one frame has a single (truncated) candidate: the
    # loop version scored it with `_rms` and kept `lo`.
    best_index = lo if curve.size == 0 else lo + int(np.argmin(curve))
    cut = best_index + frame // 2
    return min(max(cut, lo + 1), hi)


def split_audio(
    samples,
    sample_rate: int = SAMPLE_RATE,
    max_window_s: float = MAX_WINDOW_S,
    quiet_search_s: float = QUIET_SEARCH_S,
    energy_frame: int = ENERGY_FRAME_SAMPLES,
) -> list[Window]:
    """Full-coverage plan: consecutive windows of at most ``max_window_s``.

    Each limit lands on the quietest point of the preceding ``quiet_search_s``
    instead of a hard boundary. Returns ``[(start_s, end_s, samples)]`` in
    chronological order; concatenating the samples rebuilds the input exactly.
    """
    audio = np.asarray(samples)
    total = int(audio.size)
    if total == 0:
        return []
    max_len = max(1, int(round(float(max_window_s) * sample_rate)))
    search = max(1, int(round(float(quiet_search_s) * sample_rate)))
    plan: list[Window] = []
    position = 0
    while total - position > max_len:
        limit = position + max_len
        cut = quietest_index(
            audio, max(position + 1, limit - search), limit, energy_frame
        )
        if cut <= position:  # defensive: always make progress
            cut = position + 1
        if cut > limit:
            cut = limit
        plan.append(Window(position / sample_rate, cut / sample_rate, audio[position:cut]))
        position = cut
    plan.append(Window(position / sample_rate, total / sample_rate, audio[position:total]))
    return plan


def split_window(
    samples,
    sample_rate: int = SAMPLE_RATE,
    quiet_search_s: float = QUIET_SEARCH_S,
    energy_frame: int = ENERGY_FRAME_SAMPLES,
):
    """Cut ONE window in two at the quietest point around its midpoint.

    Returns ``(left, right)`` sample arrays whose concatenation is the input
    exactly, or None when the window is too short to place such a cut (the
    caller then surfaces its typed error instead of looping).
    """
    audio = np.asarray(samples)
    total = int(audio.size)
    if total < 2 * energy_frame:
        return None
    middle = total // 2
    half_search = max(1, int(round(float(quiet_search_s) * sample_rate)) // 2)
    cut = quietest_index(
        audio,
        max(1, middle - half_search),
        min(total - 1, middle + half_search),
        energy_frame,
    )
    if cut <= 0 or cut >= total:
        return None
    return audio[:cut], audio[cut:]
