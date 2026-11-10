"""Small audio helpers shared by the whole app."""

from __future__ import annotations

import math
import wave
from fractions import Fraction

import numpy as np
import scipy.signal

from .features import ANALYSIS_SR


def resample(y: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """High quality polyphase resampling of a mono signal."""
    y = np.asarray(y, dtype=np.float32)
    if int(sr_in) == int(sr_out) or len(y) == 0:
        return y
    frac = Fraction(int(sr_out), int(sr_in)).limit_denominator(1000)
    out = scipy.signal.resample_poly(y, frac.numerator, frac.denominator)
    return np.asarray(out, dtype=np.float32)


def to_analysis(y: np.ndarray, sr: int) -> np.ndarray:
    return resample(y, sr, ANALYSIS_SR)


def fit_length(y: np.ndarray, n: int) -> np.ndarray:
    """Cut or zero-pad to exactly n samples."""
    if len(y) == n:
        return y
    if len(y) > n:
        return y[:n]
    out = np.zeros(n, dtype=y.dtype)
    out[:len(y)] = y
    return out


def db(x: float) -> float:
    return 20.0 * math.log10(max(float(x), 1e-12))


def peak_db(y: np.ndarray) -> float:
    return db(float(np.max(np.abs(y)))) if len(y) else -240.0


def rms_db(y: np.ndarray) -> float:
    return db(float(np.sqrt(np.mean(np.square(y, dtype=np.float64))))) if len(y) else -240.0


def normalize_peak(y: np.ndarray, peak: float = 0.89) -> np.ndarray:
    m = float(np.max(np.abs(y))) if len(y) else 0.0
    if m <= 1e-9:
        return y
    return (y * (peak / m)).astype(np.float32)


def write_wav(path: str, y: np.ndarray, sr: int) -> None:
    """16 bit mono WAV (playable by every browser)."""
    data = np.clip(np.asarray(y, dtype=np.float64), -1.0, 1.0)
    pcm = (data * 32767.0).astype("<i2")
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(int(sr))
        f.writeframes(pcm.tobytes())


def envelope(y: np.ndarray, sr: int, window_ms: float = 4.0) -> np.ndarray:
    """RMS envelope sampled every sample (moving average of the power)."""
    n = max(1, int(round(window_ms / 1000.0 * sr)))
    power = np.square(np.asarray(y, dtype=np.float64))
    csum = np.concatenate([[0.0], np.cumsum(power)])
    half = n // 2
    idx = np.arange(len(y))
    lo = np.clip(idx - half, 0, len(y))
    hi = np.clip(idx + half + 1, 0, len(y))
    return np.sqrt((csum[hi] - csum[lo]) / np.maximum(hi - lo, 1))


def biquad_shelf(sr: float, f0: float, gain_db: float, high: bool):
    """RBJ shelving filter coefficients (b, a)."""
    a_ = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * f0 / sr
    cosw, sinw = math.cos(w0), math.sin(w0)
    alpha = sinw / 2.0 * math.sqrt(2.0)
    sq = 2.0 * math.sqrt(a_) * alpha
    if high:
        b0 = a_ * ((a_ + 1) + (a_ - 1) * cosw + sq)
        b1 = -2 * a_ * ((a_ - 1) + (a_ + 1) * cosw)
        b2 = a_ * ((a_ + 1) + (a_ - 1) * cosw - sq)
        a0 = (a_ + 1) - (a_ - 1) * cosw + sq
        a1 = 2 * ((a_ - 1) - (a_ + 1) * cosw)
        a2 = (a_ + 1) - (a_ - 1) * cosw - sq
    else:
        b0 = a_ * ((a_ + 1) - (a_ - 1) * cosw + sq)
        b1 = 2 * a_ * ((a_ - 1) - (a_ + 1) * cosw)
        b2 = a_ * ((a_ + 1) - (a_ - 1) * cosw - sq)
        a0 = (a_ + 1) + (a_ - 1) * cosw + sq
        a1 = -2 * ((a_ - 1) + (a_ + 1) * cosw)
        a2 = (a_ + 1) + (a_ - 1) * cosw - sq
    return np.array([b0, b1, b2]) / a0, np.array([1.0, a1 / a0, a2 / a0])
