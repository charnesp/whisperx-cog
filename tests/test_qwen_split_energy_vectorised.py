"""F4 RED/GREEN: the quiet-point energy scan is vectorised (numpy) and identical.

Adversarial review finding F4 (2026-10-20): ``quietest_index`` scored every frame
with a Python loop calling ``_rms`` per frame (~16 000 iterations per window,
measured 43.5 s CPU per hour of audio). The reference implementation below is
the FORMER loop version, kept verbatim in this test file: it is the oracle the
vectorised scan must match (index exactly, energy within 1e-6).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import qwen_split  # noqa: E402

SAMPLE_RATE = 16000
FRAME = qwen_split.ENERGY_FRAME_SAMPLES


# --- reference implementation: the FORMER loop version (oracle) -------------
def _loop_rms(samples, start, frame):
    block = samples[start:start + frame]
    if block.size == 0:
        return 0.0
    values = block.astype(np.float64, copy=False)
    return float(np.sqrt(np.mean(values * values)))


def _loop_quietest_index(samples, lo, hi, energy_frame=FRAME):
    total = int(samples.size)
    lo = max(0, int(lo))
    hi = min(int(hi), total)
    if hi <= lo:
        return hi
    last = max(lo, hi - energy_frame)
    best_index = lo
    best_energy = _loop_rms(samples, lo, energy_frame)
    for index in range(lo + 1, last + 1):
        energy = _loop_rms(samples, index, energy_frame)
        if energy < best_energy:
            best_energy = energy
            best_index = index
    cut = best_index + energy_frame // 2
    return min(max(cut, lo + 1), hi)


def _loop_split_audio(samples, sample_rate=SAMPLE_RATE, max_window_s=30.0,
                      quiet_search_s=1.0, energy_frame=FRAME):
    audio = np.asarray(samples)
    total = int(audio.size)
    if total == 0:
        return []
    max_len = max(1, int(round(float(max_window_s) * sample_rate)))
    search = max(1, int(round(float(quiet_search_s) * sample_rate)))
    plan = []
    position = 0
    while total - position > max_len:
        limit = position + max_len
        cut = _loop_quietest_index(
            audio, max(position + 1, limit - search), limit, energy_frame
        )
        if cut <= position:
            cut = position + 1
        if cut > limit:
            cut = limit
        plan.append((position / sample_rate, cut / sample_rate))
        position = cut
    plan.append((position / sample_rate, total / sample_rate))
    return plan


def _random_signal(seconds, seed=20261020, amplitude=0.4):
    rng = np.random.default_rng(seed)
    n = int(round(seconds * SAMPLE_RATE))
    return (rng.standard_normal(n) * amplitude).astype(np.float32)


def _speech_like_signal(seconds, seed=7):
    """Random noise with silent dips: the realistic quiet-point case."""
    audio = _random_signal(seconds, seed=seed, amplitude=0.3)
    rng = np.random.default_rng(seed + 1)
    for _ in range(int(seconds // 7) + 1):
        centre = int(rng.integers(0, max(1, len(audio) - 4000)))
        audio[centre:centre + 2400] = 0.0
    return audio


class TestVectorisedScanMatchesTheOracle(unittest.TestCase):
    def test_quietest_index_is_identical_on_a_random_signal(self):
        audio = _random_signal(20.0)
        cases = [(0, len(audio)), (1, 480000), (16000, 16000 + FRAME * 3),
                 (100000, 100000 + FRAME), (5, 5 + 2 * FRAME)]
        for lo, hi in cases:
            expected = _loop_quietest_index(audio, lo, hi)
            got = qwen_split.quietest_index(audio, lo, hi)
            self.assertEqual(got, expected, f"index drift on lo={lo} hi={hi}")

    def test_scored_energy_matches_the_loop_implementation_within_1e_6(self):
        audio = _random_signal(4.0)
        lo, hi = 10, 10 + FRAME * 40
        curve = getattr(qwen_split, "frame_rms_curve", None)
        if curve is None:
            self.skipTest("no exposed energy curve (index identity still covered)")
        energies = curve(audio, lo, hi - FRAME + 1, FRAME)
        for offset, energy in enumerate(energies):
            expected = _loop_rms(audio, lo + offset, FRAME)
            self.assertLessEqual(abs(float(energy) - expected), 1e-6)

    def test_split_audio_plan_is_identical_to_the_loop_splitter(self):
        for audio in (_random_signal(70.0), _speech_like_signal(95.0)):
            expected = _loop_split_audio(audio)
            got = [(window.start, window.end) for window in qwen_split.split_audio(audio)]
            self.assertEqual(len(got), len(expected))
            for (got_start, got_end), (want_start, want_end) in zip(got, expected):
                self.assertAlmostEqual(got_start, want_start, places=3)
                self.assertAlmostEqual(got_end, want_end, places=3)

    def test_cut_lands_on_the_synthetic_silence(self):
        audio = _random_signal(45.0, amplitude=0.2)
        audio[int(29.2 * SAMPLE_RATE):int(29.4 * SAMPLE_RATE)] = 0.0
        windows = qwen_split.split_audio(audio)
        self.assertGreaterEqual(windows[0].end, 29.2)
        self.assertLessEqual(windows[0].end, 29.4)

    def test_degenerate_windows_still_answer(self):
        audio = _random_signal(2.0)
        self.assertEqual(qwen_split.quietest_index(audio, 100, 100), 100)
        self.assertEqual(qwen_split.quietest_index(audio, 500, 100), 100)
        cut = qwen_split.quietest_index(audio, 0, FRAME - 1)  # shorter than a frame
        self.assertGreater(cut, 0)
        self.assertLessEqual(cut, FRAME - 1)


if __name__ == "__main__":
    unittest.main()
