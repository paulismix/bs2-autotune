"""Isolating the bass from a full mix.

Two methods:

* **Demucs** (optional, ``pip install demucs``): a neural source separation
  model.  By far the better result - recommended whenever the reference is a
  complete song.
* **Built-in**: no extra software.  Drums are reduced with harmonic/percussive
  separation and only the low end (where the bass dominates a mix) is kept.
  Everything above is marked as not trustworthy (see ``reliability``), so the
  brightness of the patch cannot be judged this way.

Both return the bass at the analysis sample rate and a curve that tells the
matching how much each frequency region can be trusted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import scipy.ndimage
import scipy.signal

from .audio_util import resample
from .features import ANALYSIS_SR

_REL_FREQS = 20.0 * 2.0 ** (np.arange(30) / 3.0)  # third-octave grid 20 Hz .. 16 kHz


@dataclass
class Separation:
    bass: np.ndarray  # mono, analysis sample rate
    reliability: np.ndarray  # rows of (frequency [Hz], weight 0..1)
    method: str
    note: str = ""


def demucs_available() -> bool:
    try:
        import demucs  # noqa: F401
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def _band_ratio(part: np.ndarray, whole: np.ndarray, sr: int) -> np.ndarray:
    """Energy of ``part`` relative to ``whole`` in third-octave bands."""
    n = min(len(part), len(whole))
    nper = min(4096, n)
    f, p_part = scipy.signal.welch(part[:n].astype(np.float64), sr, nperseg=nper)
    _, p_whole = scipy.signal.welch(whole[:n].astype(np.float64), sr, nperseg=nper)
    out = np.zeros(len(_REL_FREQS))
    for i, fc in enumerate(_REL_FREQS):
        sel = (f >= fc / 2 ** (1 / 6)) & (f < fc * 2 ** (1 / 6))
        if not np.any(sel):
            sel = np.array([int(np.argmin(np.abs(f - fc)))])
        out[i] = float(np.sum(p_part[sel]) / max(np.sum(p_whole[sel]), 1e-30))
    return out


def separate_demucs(data: np.ndarray, sr: int, progress: Optional[Callable[[str], None]] = None) -> Separation:
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    if progress:
        progress("loading the Demucs model (downloaded on first use)")
    model = get_model("htdemucs")
    model.eval()
    stereo = data if data.ndim == 2 else data[:, None]
    if stereo.shape[1] == 1:
        stereo = np.repeat(stereo, 2, axis=1)
    stereo = stereo[:, :2]
    msr = int(model.samplerate)
    chans = [resample(stereo[:, c], sr, msr) for c in range(2)]
    wav = torch.from_numpy(np.stack(chans).astype(np.float32))
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    wav = (wav - mean) / std
    devices = []
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")
    devices.append("cpu")
    sources = None
    last_error: Optional[Exception] = None
    for dev in devices:
        try:
            if progress:
                progress(f"separating with Demucs ({dev})")
            with torch.no_grad():
                sources = apply_model(model, wav[None], device=dev, split=True, overlap=0.25, progress=False)[0]
            break
        except Exception as exc:  # e.g. an operator that the GPU backend lacks
            last_error = exc
    if sources is None:
        raise RuntimeError(f"Demucs failed: {last_error}")
    sources = sources * std + mean
    bass = sources[model.sources.index("bass")].mean(0).cpu().numpy()
    mix = np.stack(chans).mean(0)
    bass_a = resample(bass, msr, ANALYSIS_SR)
    mix_a = resample(mix, msr, ANALYSIS_SR)
    ratio = _band_ratio(bass_a, mix_a, ANALYSIS_SR)
    weight = np.clip(ratio * 1.5, 0.1, 1.0)
    weight[_REL_FREQS <= 200.0] = 1.0
    return Separation(bass_a.astype(np.float32), np.stack([_REL_FREQS, weight], axis=1), "demucs")


def separate_builtin(data: np.ndarray, sr: int, progress: Optional[Callable[[str], None]] = None) -> Separation:
    """Low end of the mix with the drums reduced.

    Without a trained separation model the bass can only be told apart from
    the rest where it dominates: below a few hundred hertz.  Short broadband
    events (kick, snare) are reduced with harmonic/percussive separation, the
    result is low-passed and everything above is marked as not trustworthy.
    """
    mono = data.mean(axis=1) if data.ndim == 2 else data
    x = resample(mono.astype(np.float32), sr, ANALYSIS_SR).astype(np.float64)
    fs = ANALYSIS_SR
    n_fft, hop = 2048, 256
    if progress:
        progress("reducing drums")
    f, t, Z = scipy.signal.stft(x, fs, nperseg=n_fft, noverlap=n_fft - hop, window="hann")
    mag = np.abs(Z)
    # harmonic = steady over ~100 ms, percussive = smooth over ~300 Hz
    harm = scipy.ndimage.median_filter(mag, size=(1, 13))
    perc = scipy.ndimage.median_filter(mag, size=(19, 1))
    mask_h = harm ** 2 / (harm ** 2 + perc ** 2 + 1e-12)
    if progress:
        progress("keeping the low end")
    split = 1.0 / (1.0 + (f / 420.0) ** 8)  # steep low-pass around 420 Hz
    _, bass = scipy.signal.istft(Z * mask_h * split[:, None], fs, nperseg=n_fft, noverlap=n_fft - hop, window="hann")
    bass = bass[:len(x)].astype(np.float32)
    weight = np.interp(np.log2(_REL_FREQS), np.log2([20.0, 200.0, 260.0, 330.0, 420.0, 520.0, 16000.0]),
                       [1.0, 1.0, 0.7, 0.35, 0.12, 0.04, 0.04])
    note = ("The bass was isolated with the built-in method, which only hears the low end (below about 400 Hz): "
            "the brightness of the patch cannot be judged from a full mix this way. Install Demucs "
            "(pip install demucs) for a far better result, or use a passage where the bass plays alone.")
    return Separation(bass, np.stack([_REL_FREQS, weight], axis=1), "builtin", note)


def separate_bass(data: np.ndarray, sr: int, method: str = "auto",
                  progress: Optional[Callable[[str], None]] = None) -> Separation:
    """Isolate the bass; ``method`` is "auto", "demucs" or "builtin"."""
    if method in ("auto", "demucs") and demucs_available():
        try:
            return separate_demucs(data, sr, progress)
        except Exception as exc:
            if method == "demucs":
                raise
            result = separate_builtin(data, sr, progress)
            result.note = f"Demucs failed ({exc}); used the built-in method instead."
            return result
    if method == "demucs":
        raise RuntimeError("Demucs is not installed (pip install demucs).")
    return separate_builtin(data, sr, progress)
