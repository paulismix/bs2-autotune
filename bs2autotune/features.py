"""Perceptual features and the distance between a candidate and the reference.

Both sounds play the same notes with the same timing, so they can be compared
frame by frame.  Four views of the sound are compared (all in dB, after
matching the overall loudness):

* *tonal*      - mel spectrogram with a long window (harmonic balance, filter
                 position, sub oscillator),
* *transient*  - mel spectrogram with a short window (attack, filter sweeps),
* *envelope*   - the loudness contour,
* *modulation* - how much the timbre moves inside held notes (beating of
                 detuned oscillators, LFO wobble, PWM).

The oscillators of the Bass Station II run free, so their phase at note-on is
random.  A short analysis window would see that as ripple at the pitch rate;
the transient and envelope views therefore average the power over two periods
of the played note.

The distance is the weighted mean absolute difference in dB / 10, minimised
over a small time shift: "1.0" means an average error of 10 dB.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.fft
from numba import njit

ANALYSIS_SR = 32000

NoteTuple = Tuple[float, float, float]  # start [s], end [s], MIDI note


@dataclass(frozen=True)
class FeatureConfig:
    sr: int = ANALYSIS_SR
    # tonal view: long window and bands about half an ERB apart - the frequency
    # resolution of the ear (single low harmonics and anything an octave below
    # the played note are resolved, higher up several harmonics share a band)
    n_fft_a: int = 4096
    hop_a: int = 512
    n_bands_a: int = 72
    fmin_a: float = 16.0
    # transient view
    n_fft_b: int = 512
    hop_b: int = 128
    n_bands_b: int = 28
    fmin_b: float = 50.0
    fmax: float = 15000.0
    # dynamic range that is compared (dB below the loudest reference band)
    range_a: float = 60.0
    range_b: float = 50.0
    range_env: float = 50.0
    # weights of the views
    w_tonal: float = 1.0
    w_transient: float = 0.6
    w_envelope: float = 0.6
    w_modulation: float = 1.0
    # alignment
    max_shift_ms: float = 48.0
    onset_boost: float = 2.0
    onset_window_ms: float = 60.0


DEFAULT_CONFIG = FeatureConfig()

# band groups (Hz) and lags (tonal frames) of the modulation view
MOD_EDGES = (25.0, 120.0, 250.0, 500.0, 1000.0, 2500.0, 15000.0)
MOD_LAGS = (1, 3, 8)


def note_hz(note: float) -> float:
    return 440.0 * 2.0 ** ((note - 69.0) / 12.0)


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def _hz_to_erb(f):
    return 21.4 * np.log10(4.37e-3 * np.asarray(f, dtype=np.float64) + 1.0)


def _erb_to_hz(e):
    return (10.0 ** (np.asarray(e, dtype=np.float64) / 21.4) - 1.0) / 4.37e-3


def _triangular_bands(sr: int, n_fft: int, edges: np.ndarray):
    """Triangular bands from n+2 edge frequencies -> (start, length, weights, centers)."""
    n_bins = n_fft // 2 + 1
    bin_hz = sr / n_fft
    edges = np.array(edges, dtype=np.float64)
    for i in range(1, len(edges)):  # every band at least ~1.5 bins wide
        if edges[i] - edges[i - 1] < 0.75 * bin_hz:
            edges[i] = edges[i - 1] + 0.75 * bin_hz
    n_bands = len(edges) - 2
    freqs = np.arange(n_bins) * bin_hz
    start = np.zeros(n_bands, dtype=np.int64)
    length = np.zeros(n_bands, dtype=np.int64)
    rows = []
    for b in range(n_bands):
        lo, ce, hi = edges[b], edges[b + 1], edges[b + 2]
        up = (freqs - lo) / max(ce - lo, 1e-9)
        down = (hi - freqs) / max(hi - ce, 1e-9)
        w = np.maximum(0.0, np.minimum(up, down))
        nz = np.nonzero(w > 0)[0]
        if len(nz) == 0:
            nz = np.array([min(n_bins - 1, int(round(ce / bin_hz)))])
            w[nz[0]] = 1.0
        start[b] = nz[0]
        length[b] = nz[-1] - nz[0] + 1
        rows.append(w[nz[0]:nz[-1] + 1] / max(w.sum(), 1e-12))
    weights = np.zeros((n_bands, int(length.max())), dtype=np.float32)
    for b, row in enumerate(rows):
        weights[b, :len(row)] = row
    return start, length, weights, edges[1:-1].copy()


def mel_bands(sr: int, n_fft: int, n_bands: int, fmin: float, fmax: float):
    """Triangular mel filter bank in a compact form.

    Returns (start, length, weights, centers): band b averages the power bins
    start[b] .. start[b]+length[b] with weights[b, :length[b]].
    """
    fmax = min(fmax, sr / 2.0 - sr / n_fft)
    return _triangular_bands(sr, n_fft, _mel_to_hz(np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_bands + 2)))


def erb_bands(sr: int, n_fft: int, n_bands: int, fmin: float, fmax: float):
    """Triangular bands equally spaced on the ERB-rate scale (auditory resolution)."""
    fmax = min(fmax, sr / 2.0 - sr / n_fft)
    return _triangular_bands(sr, n_fft, _erb_to_hz(np.linspace(_hz_to_erb(fmin), _hz_to_erb(fmax), n_bands + 2)))


# ---------------------------------------------------------------------------
# numba helpers
# ---------------------------------------------------------------------------


@njit(cache=True, nogil=True)
def _band_power(spec, start, length, weights, out):
    """spec: (frames, bins, 2) float32 view of the complex spectrum."""
    n_frames = spec.shape[0]
    n_bands = start.shape[0]
    for t in range(n_frames):
        for b in range(n_bands):
            acc = 0.0
            s = start[b]
            for i in range(length[b]):
                re = spec[t, s + i, 0]
                im = spec[t, s + i, 1]
                acc += weights[b, i] * (re * re + im * im)
            out[t, b] = acc


@njit(cache=True, nogil=True)
def _smooth_power(power, half, out):
    """Moving average over time with a per-frame half width (frames, fractional)."""
    n_t = power.shape[0]
    n_b = power.shape[1]
    csum = np.zeros(n_t + 1, dtype=np.float64)
    for b in range(n_b):
        for t in range(n_t):
            csum[t + 1] = csum[t] + power[t, b]
        for t in range(n_t):
            h = half[t]
            if h < 0.5:
                out[t, b] = power[t, b]
                continue
            lo = t + 0.5 - h
            hi = t + 0.5 + h
            if lo < 0.0:
                lo = 0.0
            if hi > n_t:
                hi = float(n_t)
            i0 = int(lo)
            i1 = int(hi)
            c_lo = csum[i0] + (lo - i0) * (csum[i0 + 1] - csum[i0]) if i0 < n_t else csum[n_t]
            c_hi = csum[i1] + (hi - i1) * (csum[i1 + 1] - csum[i1]) if i1 < n_t else csum[n_t]
            out[t, b] = (c_hi - c_lo) / (hi - lo)


@njit(cache=True, nogil=True)
def _to_db(power, offset_db, floor_db, out):
    for t in range(power.shape[0]):
        for b in range(power.shape[1]):
            v = 10.0 * math.log10(power[t, b] + 1e-20) + offset_db
            out[t, b] = v if v > floor_db else floor_db


@njit(cache=True, nogil=True)
def _env_db(power, offset_db, floor_db, out):
    for t in range(power.shape[0]):
        acc = 0.0
        for b in range(power.shape[1]):
            acc += power[t, b]
        v = 10.0 * math.log10(acc + 1e-20) + offset_db
        out[t] = v if v > floor_db else floor_db


@njit(cache=True, nogil=True)
def _l1_shift(ref, cand, w_band, w_frame, floor_db, shift):
    """Weighted mean |ref[t] - cand[t + shift]| (missing frames = floor)."""
    n_t = ref.shape[0]
    n_b = ref.shape[1]
    n_c = cand.shape[0]
    acc = 0.0
    wsum = 0.0
    wb_total = 0.0
    for b in range(n_b):
        wb_total += w_band[b]
    for t in range(n_t):
        wf = w_frame[t]
        if wf <= 0.0:
            continue
        tc = t + shift
        row = 0.0
        if 0 <= tc < n_c:
            for b in range(n_b):
                row += w_band[b] * abs(ref[t, b] - cand[tc, b])
        else:
            for b in range(n_b):
                row += w_band[b] * abs(ref[t, b] - floor_db)
        acc += wf * row
        wsum += wf * wb_total
    return acc / wsum if wsum > 0.0 else 0.0


@njit(cache=True, nogil=True)
def _l1_shift_1d(ref, cand, w_frame, floor_db, shift):
    n_t = ref.shape[0]
    n_c = cand.shape[0]
    acc = 0.0
    wsum = 0.0
    for t in range(n_t):
        wf = w_frame[t]
        if wf <= 0.0:
            continue
        tc = t + shift
        c = cand[tc] if 0 <= tc < n_c else floor_db
        acc += wf * abs(ref[t] - c)
        wsum += wf
    return acc / wsum if wsum > 0.0 else 0.0


@njit(cache=True, nogil=True)
def _modulation(power, offset_db, floor_db, band_group, n_groups, seg_start, seg_end, lags, shift, out):
    """Movement of the timbre inside held notes.

    For every band and lag: the mean absolute change of the detrended level
    (dB) over that lag inside the sustained part of the notes.  The result is
    averaged over the bands of each band group.
    """
    n_t = power.shape[0]
    n_b = power.shape[1]
    n_l = lags.shape[0]
    count = np.zeros((n_groups, n_l), dtype=np.float64)
    acc = np.zeros((n_groups, n_l), dtype=np.float64)
    buf = np.zeros(n_t, dtype=np.float64)
    for s in range(seg_start.shape[0]):
        t0 = seg_start[s] + shift
        t1 = seg_end[s] + shift
        if t0 < 0:
            t0 = 0
        if t1 > n_t:
            t1 = n_t
        n = t1 - t0
        if n < 6:
            continue
        for b in range(n_b):
            gi = band_group[b]
            if gi < 0:
                continue
            mean = 0.0
            for i in range(n):
                v = 10.0 * math.log10(power[t0 + i, b] + 1e-20) + offset_db
                if v < floor_db:
                    v = floor_db
                buf[i] = v
                mean += v
            mean /= n
            if mean < floor_db + 15.0:
                continue  # too quiet to say anything about its movement
            # remove a straight line (decay) by least squares
            mx = (n - 1) / 2.0
            sxy = 0.0
            sxx = 0.0
            for i in range(n):
                sxy += (i - mx) * (buf[i] - mean)
                sxx += (i - mx) * (i - mx)
            slope = sxy / sxx if sxx > 0.0 else 0.0
            for i in range(n):
                buf[i] -= mean + slope * (i - mx)
            for li in range(n_l):
                lag = lags[li]
                if n <= lag + 1:
                    continue
                d = 0.0
                for i in range(n - lag):
                    x = abs(buf[i + lag] - buf[i])
                    d += x if x < 20.0 else 20.0
                acc[gi, li] += d
                count[gi, li] += n - lag
    for gi in range(n_groups):
        for li in range(n_l):
            out[gi, li] = acc[gi, li] / count[gi, li] if count[gi, li] > 0 else 0.0


# ---------------------------------------------------------------------------
# feature extraction
# ---------------------------------------------------------------------------


class _Bank:
    def __init__(self, sr, n_fft, hop, n_bands, fmin, fmax, scale="mel"):
        self.n_fft = n_fft
        self.hop = hop
        self.window = np.hanning(n_fft + 1)[:-1].astype(np.float32)
        builder = erb_bands if scale == "erb" else mel_bands
        self.start, self.length, self.weights, self.centers = builder(sr, n_fft, n_bands, fmin, fmax)
        self.n_bands = len(self.start)

    def power(self, y: np.ndarray) -> np.ndarray:
        """Mel band power, frames centred on t * hop."""
        n_fft, hop = self.n_fft, self.hop
        half = n_fft // 2
        padded = np.zeros(len(y) + n_fft, dtype=np.float32)
        padded[half:half + len(y)] = y
        frames = np.lib.stride_tricks.sliding_window_view(padded, n_fft)[::hop]
        spec = scipy.fft.rfft(frames * self.window, axis=1)
        out = np.empty((spec.shape[0], self.n_bands), dtype=np.float32)
        _band_power(spec.view(np.float32).reshape(spec.shape[0], spec.shape[1], 2),
                    self.start, self.length, self.weights, out)
        return out


    def mean_power(self, y: np.ndarray) -> np.ndarray:
        """Band power averaged over all complete frames of a short segment."""
        n_fft, hop = self.n_fft, self.hop
        if len(y) < n_fft:
            padded = np.zeros(n_fft, dtype=np.float32)
            padded[:len(y)] = y
            y = padded
        frames = np.lib.stride_tricks.sliding_window_view(np.asarray(y, dtype=np.float32), n_fft)[::hop]
        spec = scipy.fft.rfft(frames * self.window, axis=1)
        out = np.empty((spec.shape[0], self.n_bands), dtype=np.float32)
        _band_power(spec.view(np.float32).reshape(spec.shape[0], spec.shape[1], 2),
                    self.start, self.length, self.weights, out)
        return out.mean(axis=0)


_BANKS: Dict[Tuple, Tuple[_Bank, _Bank]] = {}


def _banks(cfg: FeatureConfig) -> Tuple[_Bank, _Bank]:
    key = (cfg.sr, cfg.n_fft_a, cfg.hop_a, cfg.n_bands_a, cfg.fmin_a, cfg.n_fft_b, cfg.hop_b,
           cfg.n_bands_b, cfg.fmin_b, cfg.fmax)
    if key not in _BANKS:
        _BANKS[key] = (
            _Bank(cfg.sr, cfg.n_fft_a, cfg.hop_a, cfg.n_bands_a, cfg.fmin_a, cfg.fmax, "erb"),
            _Bank(cfg.sr, cfg.n_fft_b, cfg.hop_b, cfg.n_bands_b, cfg.fmin_b, cfg.fmax, "mel"),
        )
    return _BANKS[key]


@dataclass
class Features:
    """Band power of one sound (linear power, not yet level matched)."""

    a: np.ndarray  # tonal view (frames, bands)
    b: np.ndarray  # transient view
    energy: float  # mean power inside the analysed bands (used to match the loudness)


def tonal_bank(cfg: FeatureConfig = DEFAULT_CONFIG) -> _Bank:
    return _banks(cfg)[0]


def band_energy(a: np.ndarray, cfg: FeatureConfig = DEFAULT_CONFIG) -> float:
    """Mean power of a sound inside the tonal bands.

    The loudness of two sounds is matched on this number rather than on the
    raw waveform, so that anything outside the analysed band (DC, rumble) and
    any spectral correction applied to the bands is treated consistently.
    """
    widths = _banks(cfg)[0].length.astype(np.float64)
    return float(np.mean(a.astype(np.float64) @ widths)) + 1e-20


def extract(y: np.ndarray, cfg: FeatureConfig = DEFAULT_CONFIG) -> Features:
    y = np.asarray(y, dtype=np.float32)
    bank_a, bank_b = _banks(cfg)
    a = bank_a.power(y)
    return Features(a, bank_b.power(y), band_energy(a, cfg))


FeatureSet = Union[Features, Sequence[Features]]


def _as_list(feats: FeatureSet) -> List[Features]:
    return [feats] if isinstance(feats, Features) else list(feats)


def _mean_power(items: List[Features]) -> Features:
    if len(items) == 1:
        return items[0]
    n = float(len(items))
    a = items[0].a.copy()
    b = items[0].b.copy()
    e = items[0].energy
    for it in items[1:]:
        a += it.a
        b += it.b
        e += it.energy
    return Features(a / n, b / n, e / n)


@dataclass
class Aligned:
    """Candidate features expressed like the reference (dB, level matched, shifted)."""

    a: np.ndarray
    b: np.ndarray
    env: np.ndarray
    mod: np.ndarray
    shift: int  # in transient frames

    def minus(self, other: "Aligned", limit: float = 18.0) -> "Aligned":
        """Correction that turns ``other`` into ``self`` (clipped to +-limit dB)."""
        return Aligned(np.clip(self.a - other.a, -limit, limit), np.clip(self.b - other.b, -limit, limit),
                       np.clip(self.env - other.env, -limit, limit), self.mod - other.mod, 0)


def smoothing_halfwidths(notes: Sequence[NoteTuple], n_frames: int, cfg: FeatureConfig) -> np.ndarray:
    """Per-frame half width (transient frames) of the power smoothing.

    The window spans two periods of the note sounding at that moment (which is
    also one period of a component an octave below it).
    """
    hop_s = cfg.hop_b / cfg.sr
    half = np.full(n_frames, 0.012 / hop_s, dtype=np.float32)
    if notes:
        ordered = sorted(notes, key=lambda n: n[0])
        idx = 0
        for t in range(n_frames):
            now = t * hop_s
            while idx + 1 < len(ordered) and ordered[idx + 1][0] <= now + 0.5 * hop_s:
                idx += 1
            half[t] = (1.0 / note_hz(ordered[idx][2])) / hop_s
    return np.clip(half, 1.0, 0.045 / hop_s).astype(np.float32)


class Target:
    """The reference sound prepared for fast comparisons.

    ``notes`` are the (start, end, MIDI note) triples of the phrase that both
    the reference and every candidate play.
    """

    def __init__(self, audio: np.ndarray, cfg: FeatureConfig = DEFAULT_CONFIG,
                 notes: Optional[Sequence[NoteTuple]] = None,
                 band_reliability: Optional[np.ndarray] = None):
        self.cfg = cfg
        self.n_samples = len(audio)
        self.notes: List[NoteTuple] = [(float(n[0]), float(n[1]), float(n[2])) for n in notes] if notes else []
        feats = extract(audio, cfg)
        self.features = feats
        self.energy = feats.energy
        bank_a, bank_b = _banks(cfg)
        self.centers_a = bank_a.centers
        self.centers_b = bank_b.centers
        self.half = smoothing_halfwidths(self.notes, feats.b.shape[0], cfg)
        b_smooth = np.empty_like(feats.b)
        _smooth_power(feats.b, self.half, b_smooth)

        peak_a = 10.0 * math.log10(float(feats.a.max()) + 1e-20)
        peak_b = 10.0 * math.log10(float(b_smooth.max()) + 1e-20)
        peak_e = 10.0 * math.log10(float(b_smooth.sum(axis=1).max()) + 1e-20)
        self.floor_a = peak_a - cfg.range_a
        self.floor_b = peak_b - cfg.range_b
        self.floor_e = peak_e - cfg.range_env
        self.ref_a = np.empty_like(feats.a)
        self.ref_b = np.empty_like(feats.b)
        self.ref_e = np.empty(feats.b.shape[0], dtype=np.float32)
        _to_db(feats.a, 0.0, self.floor_a, self.ref_a)
        _to_db(b_smooth, 0.0, self.floor_b, self.ref_b)
        _env_db(b_smooth, 0.0, self.floor_e, self.ref_e)

        # band weights: gently de-emphasise the top octaves, apply reliability
        def band_w(centers):
            w = np.ones(len(centers), dtype=np.float32)
            hi = centers > 5000.0
            w[hi] = (1.0 - 0.5 * np.log2(centers[hi] / 5000.0) / np.log2(cfg.fmax / 5000.0)).astype(np.float32)
            return w

        self.w_band_a = band_w(self.centers_a)
        self.w_band_b = band_w(self.centers_b)
        self.reliability_hz = None
        if band_reliability is not None:
            rel = np.asarray(band_reliability, dtype=np.float64)
            self.reliability_hz = rel  # rows of (frequency, weight)
            self.w_band_a *= np.interp(self.centers_a, rel[:, 0], rel[:, 1]).astype(np.float32)
            self.w_band_b *= np.interp(self.centers_b, rel[:, 0], rel[:, 1]).astype(np.float32)

        # frame weights: ignore what happens before the first note, boost the attacks
        t_a = np.arange(feats.a.shape[0]) * cfg.hop_a / cfg.sr
        t_b = np.arange(feats.b.shape[0]) * cfg.hop_b / cfg.sr
        start_time = min((n[0] for n in self.notes), default=0.0)
        self.w_frame_a = (t_a >= start_time - 0.04).astype(np.float32)
        self.w_frame_b = (t_b >= start_time - 0.04).astype(np.float32)
        if self.notes and cfg.onset_boost > 1.0:
            win = cfg.onset_window_ms / 1000.0
            for note in self.notes:
                sel = (t_b >= note[0] - 0.01) & (t_b <= note[0] + win)
                self.w_frame_b[sel] = cfg.onset_boost
        self.w_frame_e = self.w_frame_b.copy()
        self.max_shift = max(0, int(round(cfg.max_shift_ms / 1000.0 * cfg.sr / cfg.hop_b)))
        self.ratio = cfg.hop_a // cfg.hop_b

        # modulation view: band groups and the sustained part of every note
        edges = np.asarray(MOD_EDGES)
        self.band_group = np.full(len(self.centers_a), -1, dtype=np.int64)
        g_w = []
        for gi, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            idx = np.nonzero((self.centers_a >= lo) & (self.centers_a < hi))[0]
            self.band_group[idx] = gi
            g_w.append(float(self.w_band_a[idx].mean()) if len(idx) else 0.0)
        self.n_groups = len(edges) - 1
        self.g_weight = np.asarray(g_w, dtype=np.float64)
        frame_s = cfg.hop_a / cfg.sr
        seg_start, seg_end = [], []
        for start, end, _ in self.notes:
            if end - start >= 0.2:
                seg_start.append(int(math.ceil((start + 0.07) / frame_s)))
                seg_end.append(int(math.floor((end - 0.02) / frame_s)))
        self.seg_start = np.asarray(seg_start, dtype=np.int64)
        self.seg_end = np.asarray(seg_end, dtype=np.int64)
        self.lags = np.asarray(MOD_LAGS, dtype=np.int64)
        self.ref_mod = self._mod(feats, 0.0, 0)

    # -- pieces -------------------------------------------------------------
    def _offset(self, feats: Features) -> float:
        return 10.0 * math.log10(self.energy / feats.energy)

    def _mod(self, feats: Features, offset: float, shift_a: int) -> np.ndarray:
        out = np.zeros((self.n_groups, len(self.lags)), dtype=np.float64)
        if len(self.seg_start):
            _modulation(feats.a, offset, self.floor_a, self.band_group, self.n_groups, self.seg_start,
                        self.seg_end, self.lags, shift_a, out)
        return out

    def _db(self, feats: Features):
        off = self._offset(feats)
        a = np.empty_like(feats.a)
        b = np.empty_like(feats.b)
        e = np.empty(feats.b.shape[0], dtype=np.float32)
        b_smooth = np.empty_like(feats.b)
        half = self.half
        if len(half) != feats.b.shape[0]:
            half = np.resize(half, feats.b.shape[0])
        _smooth_power(feats.b, half, b_smooth)
        _to_db(feats.a, off, self.floor_a, a)
        _to_db(b_smooth, off, self.floor_b, b)
        _env_db(b_smooth, off, self.floor_e, e)
        return a, b, e

    def _parts(self, a, b, e, shift_b: int) -> Tuple[float, float, float]:
        sa = int(round(shift_b / self.ratio))
        la = _l1_shift(self.ref_a, a, self.w_band_a, self.w_frame_a, self.floor_a, sa) / 10.0
        lb = _l1_shift(self.ref_b, b, self.w_band_b, self.w_frame_b, self.floor_b, shift_b) / 10.0
        le = _l1_shift_1d(self.ref_e, e, self.w_frame_e, self.floor_e, shift_b) / 10.0
        return la, lb, le

    def _mod_loss(self, mod: np.ndarray) -> float:
        if mod.size == 0 or not len(self.seg_start):
            return 0.0
        diff = np.abs(mod - self.ref_mod).mean(axis=1)
        return float((diff * self.g_weight).sum() / max(self.g_weight.sum(), 1e-9)) / 10.0

    def _total(self, parts: Tuple[float, float, float], mod_loss: float) -> float:
        c = self.cfg
        return (c.w_tonal * parts[0] + c.w_transient * parts[1] + c.w_envelope * parts[2]
                + c.w_modulation * mod_loss) / (c.w_tonal + c.w_transient + c.w_envelope)

    # -- public API -----------------------------------------------------------
    def compare(self, feats: FeatureSet, shift: Optional[int] = None) -> dict:
        """Distance to the reference with its parts and the best time shift.

        ``feats`` may be several renders of the same patch (different
        oscillator phases); their power is averaged.
        """
        items = _as_list(feats)
        mean = _mean_power(items)
        a, b, e = self._db(mean)
        if shift is None:
            best = None
            for s in range(-self.max_shift, self.max_shift + 1):
                parts = self._parts(a, b, e, s)
                total = parts[0] * self.cfg.w_tonal + parts[1] * self.cfg.w_transient + parts[2] * self.cfg.w_envelope
                if best is None or total < best[0] - 1e-9:
                    best = (total, parts, s)
            parts, shift = best[1], best[2]
        else:
            parts = self._parts(a, b, e, shift)
        sa = int(round(shift / self.ratio))
        mod = np.mean([self._mod(it, self._offset(it), sa) for it in items], axis=0)
        mod_loss = self._mod_loss(mod)
        return {"loss": float(self._total(parts, mod_loss)), "tonal": float(parts[0]),
                "transient": float(parts[1]), "envelope": float(parts[2]), "modulation": float(mod_loss),
                "shift": int(shift), "shift_ms": 1000.0 * shift * self.cfg.hop_b / self.cfg.sr}

    def loss(self, feats: FeatureSet) -> float:
        return self.compare(feats)["loss"]

    def loss_audio(self, y: np.ndarray) -> float:
        return self.loss(extract(y, self.cfg))

    def aligned(self, feats: FeatureSet, shift: Optional[int] = None) -> Aligned:
        """dB features of a candidate, shifted so that they line up with the reference."""
        items = _as_list(feats)
        if shift is None:
            shift = self.compare(items)["shift"]
        a, b, e = self._db(_mean_power(items))
        sa = int(round(shift / self.ratio))
        mod = np.mean([self._mod(it, self._offset(it), sa) for it in items], axis=0)
        return Aligned(_shift_rows(a, sa, self.ref_a.shape[0], self.floor_a),
                       _shift_rows(b, shift, self.ref_b.shape[0], self.floor_b),
                       _shift_rows(e, shift, self.ref_e.shape[0], self.floor_e), mod, shift)

    def loss_aligned(self, al: Aligned, corr: Optional[Aligned] = None) -> float:
        """Distance of aligned dB features, optionally with a correction added."""
        a, b, e, mod = al.a, al.b, al.env, al.mod
        if corr is not None:
            a = np.maximum(a + corr.a, self.floor_a)
            b = np.maximum(b + corr.b, self.floor_b)
            e = np.maximum(e + corr.env, self.floor_e)
            mod = np.maximum(mod + corr.mod, 0.0)
        return float(self._total(self._parts(a, b, e, 0), self._mod_loss(mod)))

    def reference_aligned(self) -> Aligned:
        return Aligned(self.ref_a, self.ref_b, self.ref_e, self.ref_mod, 0)


def _shift_rows(x: np.ndarray, shift: int, n_out: int, fill: float) -> np.ndarray:
    """out[t] = x[t + shift] with missing rows filled."""
    out = np.full((n_out,) + x.shape[1:], fill, dtype=np.float32)
    lo = max(0, -shift)
    hi = min(n_out, x.shape[0] - shift)
    if hi > lo:
        out[lo:hi] = x[lo + shift:hi + shift]
    return out


def warmup(cfg: FeatureConfig = DEFAULT_CONFIG) -> None:
    """Compile the numba helpers."""
    y = np.zeros(cfg.sr // 2, dtype=np.float32)
    y[100] = 1.0
    t = Target(y, cfg, notes=[(0.0, 0.4, 40)])
    f = extract(y, cfg)
    t.loss_aligned(t.aligned(f))
