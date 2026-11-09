"""Measuring the instrument so that the software model follows it.

Nothing about the response curves of the Bass Station II is published: which
frequency a given filter knob position means, how long "decay 60" lasts, how
loud a triangle is compared to a sawtooth.  The calibration plays about seventy
short test notes and measures those things on the unit on the desk:

* signal level, noise floor, round-trip latency, tuning
* envelope times (attack, decay, release) and the shape of the attack
* filter cutoff curve, resonance curve, 12 dB mode, acid filter
* how far the mod envelope moves the filter, key tracking
* mixer law and the level of every waveform, the sub oscillator and noise

Every measurement is turned into a table of :class:`~bs2autotune.engine.Physics`.
A step that fails keeps the default table and is reported, it never aborts the
calibration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.signal

from .audio_util import envelope, peak_db, rms_db, to_analysis
from .device import Cancelled, Device, DeviceError
from .engine import Note, Phrase, Physics, filter_mag_db
from .features import ANALYSIS_SR
from .params import baseline_patch

PRE = 0.06  # silence before each test note [s]
ENV_NOTE = 81  # A5: a short period, so the envelope can be followed closely
TONE_NOTE = 45  # A2, 110 Hz
FILTER_NOTE = 48  # the cutoff table is defined for this note

_FIT_SR = 4.0e6  # "analog" sample rate for fitting filter responses (no frequency warping)


@dataclass
class CalibrationResult:
    physics: Physics
    latency: float = 0.0
    peak_db: float = -120.0
    noise_db: float = -120.0
    tune_cents: float = 0.0
    steps: List[dict] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"physics": self.physics.to_dict(), "latency": self.latency, "peak_db": self.peak_db,
                "noise_db": self.noise_db, "tune_cents": self.tune_cents, "steps": self.steps,
                "warnings": self.warnings}


def _log_table(points: Sequence[Tuple[float, float]], n: int, lo: float, hi: float) -> np.ndarray:
    """Table over raw values 0..n-1 through (raw, value) points, interpolated in the log domain."""
    pts = sorted((float(r), float(v)) for r, v in points if v > 0 and np.isfinite(v))
    raws = np.array([p[0] for p in pts])
    logs = np.log(np.array([p[1] for p in pts]))
    x = np.arange(n, dtype=np.float64)
    out = np.interp(x, raws, logs)
    if len(pts) >= 2:  # extend the end slopes
        s0 = (logs[1] - logs[0]) / max(raws[1] - raws[0], 1e-9)
        s1 = (logs[-1] - logs[-2]) / max(raws[-1] - raws[-2], 1e-9)
        out[x < raws[0]] = logs[0] + s0 * (x[x < raws[0]] - raws[0])
        out[x > raws[-1]] = logs[-1] + s1 * (x[x > raws[-1]] - raws[-1])
    return np.clip(np.exp(out), lo, hi)


def _lin_table(points: Sequence[Tuple[float, float]], n: int, lo: float, hi: float) -> np.ndarray:
    pts = sorted((float(r), float(v)) for r, v in points if np.isfinite(v))
    raws = np.array([p[0] for p in pts])
    vals = np.array([p[1] for p in pts])
    x = np.arange(n, dtype=np.float64)
    out = np.interp(x, raws, vals)
    if len(pts) >= 2:
        s0 = (vals[1] - vals[0]) / max(raws[1] - raws[0], 1e-9)
        s1 = (vals[-1] - vals[-2]) / max(raws[-1] - raws[-2], 1e-9)
        out[x < raws[0]] = vals[0] + s0 * (x[x < raws[0]] - raws[0])
        out[x > raws[-1]] = vals[-1] + s1 * (x[x > raws[-1]] - raws[-1])
    return np.clip(out, lo, hi)


def _attack_ratio(o: float) -> float:
    """t90 / t50 of the model attack curve for an overshoot parameter o."""
    return math.log(1.0 - 0.9 / (1.0 + o)) / math.log(1.0 - 0.5 / (1.0 + o))


def _attack_factor(o: float) -> float:
    """Attack time (0 -> 1) divided by the time to reach 50 %."""
    return math.log((1.0 + o) / o) / -math.log(1.0 - 0.5 / (1.0 + o))


class Calibrator:
    def __init__(self, device: Device, physics: Optional[Physics] = None,
                 progress: Optional[Callable[[dict], None]] = None):
        self.dev = device
        self.ph = (physics or Physics()).copy()
        self.progress = progress
        self.sr = ANALYSIS_SR
        self.result = CalibrationResult(self.ph)
        self.base = baseline_patch()
        self.base.update(osc1_wave=2, osc1_range=64, mix_osc1=255, filter_freq=255, filter_res=0,
                         amp_attack=0, amp_decay=0, amp_sustain=127, amp_release=0)
        self._step = 0
        self._total = 14
        # 1/6 octave analysis bands for filter measurements
        edges = 28.0 * 2.0 ** (np.arange(0, 56) / 6.0)
        self.band_edges = edges[edges < 0.45 * self.sr]
        self.band_freqs = np.sqrt(self.band_edges[:-1] * self.band_edges[1:])

    # -- helpers ----------------------------------------------------------------
    def _report(self, name: str, ok: bool = True, message: str = "", **data) -> None:
        self._step += 1
        step = {"name": name, "ok": bool(ok), "message": message, "data": data}
        self.result.steps.append(step)
        if not ok:
            self.result.warnings.append(f"{name}: {message}")
        if self.progress is not None:
            self.progress({"phase": "calibration", "step": self._step, "total": self._total, "name": name,
                           "ok": bool(ok), "message": message})

    def _note(self, updates: Dict[str, int], note: int, hold: float, tail: float = 0.12) -> np.ndarray:
        """Play one test note; returns latency compensated audio at the analysis rate."""
        patch = dict(self.base)
        patch.update(updates)
        phrase = Phrase([Note(PRE, PRE + hold, note)], PRE + hold + tail)
        return self.dev.capture(patch, phrase)

    def _steady(self, y: np.ndarray, hold: float, skip: float = 0.07) -> np.ndarray:
        return y[int((PRE + skip) * self.sr):int((PRE + hold - 0.01) * self.sr)]

    def _bands(self, seg: np.ndarray) -> np.ndarray:
        """Power in the analysis bands (integral of the Welch PSD)."""
        nper = min(4096, len(seg))
        f, p = scipy.signal.welch(seg.astype(np.float64), self.sr, nperseg=nper, noverlap=nper // 2)
        cum = np.concatenate([[0.0], np.cumsum(p) * (f[1] - f[0])])
        at = np.interp(self.band_edges, f + 0.5 * (f[1] - f[0]), cum[1:])
        return np.maximum(np.diff(at), 1e-30)

    def _fit(self, ratio_db: np.ndarray, valid: np.ndarray, bound_db: np.ndarray, ftype: int, slope: int,
             shape: int, k_grid: Sequence[float], k2: float, free_gain: bool) -> Tuple[float, float, float, float]:
        """Fit cutoff and resonance of the model filter to a measured response.

        ratio_db - response relative to the open filter, valid - bands where it
        was measured, bound_db - upper bound where the signal sank into the noise.
        Returns (fc, k, gain_db, error).
        """
        best = (1000.0, float(k_grid[0]), 0.0, float("inf"))
        censored = ~valid & np.isfinite(bound_db)
        if valid.sum() < 4:
            return best
        fcs = 12.0 * 2.0 ** (np.arange(0, int(24 * math.log2(26000.0 / 12.0))) / 24.0)
        for k in k_grid:
            for fc in fcs:
                h = filter_mag_db(self.band_freqs, _FIT_SR, ftype, slope, shape, fc, k, k2, self.ph.acid_comp)
                gain = float(np.median(ratio_db[valid] - h[valid])) if free_gain else 0.0
                err = float(np.mean(np.abs(ratio_db[valid] - h[valid] - gain)))
                if censored.any():
                    err += float(np.sum(np.maximum(0.0, h[censored] + gain - bound_db[censored]))) / valid.sum()
                if err < best[3]:
                    best = (float(fc), float(k), gain, err)
        return best

    def _response(self, updates: Dict[str, int], note: int = FILTER_NOTE, hold: float = 0.4):
        """Filter response with the noise source, relative to the open filter."""
        src = {"mix_osc1": 0, "mix_noise": 255}
        if getattr(self, "_open", None) is None or self._open_note != note:
            y = self._note(dict(src), note, hold)
            self._open = self._bands(self._steady(y, hold))
            self._open_note = note
        y = self._note(dict(src, **updates), note, hold)
        p = self._bands(self._steady(y, hold))
        ratio = 10.0 * np.log10(p / self._open)
        usable = self._open > self._noise_bands * 10.0  # source well above the noise
        valid = usable & (p > self._noise_bands * 2.5)
        bound = np.where(usable & ~valid, 10.0 * np.log10(self._noise_bands * 2.5 / self._open), np.nan)
        return ratio, valid, bound

    # -- steps --------------------------------------------------------------------
    def step_signal(self) -> None:
        """Level, noise floor and latency."""
        dev = self.dev
        dev.latency = 0.0
        dev.set_patch(self.base, force=True)
        silence = dev.play(Phrase([], 0.35))
        noise = float(np.sqrt(np.mean(np.square(silence[int(0.05 * dev.sr):], dtype=np.float64))))
        self.result.noise_db = 20.0 * math.log10(max(noise, 1e-9))
        self._noise_bands = self._bands(to_analysis(silence, dev.sr)[int(0.05 * self.sr):])
        lats, peaks = [], []
        for _ in range(3):
            phrase = Phrase([Note(PRE, PRE + 0.2, ENV_NOTE)], PRE + 0.45)
            raw = dev.play(phrase, extra=0.15)
            env = envelope(raw, dev.sr, 1.5)
            peak = float(env.max())
            peaks.append(float(np.max(np.abs(raw))))
            if peak < max(10.0 * noise, 2e-4):
                continue
            above = np.nonzero(env > max(0.2 * peak, 8.0 * noise))[0]
            if len(above):
                lats.append(above[0] / dev.sr - PRE)
        self.result.peak_db = 20.0 * math.log10(max(max(peaks), 1e-9))
        if not lats:
            raise DeviceError(
                "No sound arrives from the synth. Check that the MIDI channel matches, that the BS2 output is "
                "connected to the selected input of the audio interface and that the volume is up.")
        latency = float(np.median(lats))
        if latency < -0.005 or latency > 0.6:
            raise DeviceError(f"Implausible latency ({latency * 1000:.0f} ms) - is the right input selected?")
        dev.latency = max(0.0, latency - 0.0015)  # keep a little of the pre-onset in the window
        self.result.latency = dev.latency
        msg = f"latency {latency * 1000:.1f} ms, peak {self.result.peak_db:.1f} dBFS, noise {self.result.noise_db:.1f} dBFS"
        ok = True
        if self.result.peak_db > -1.0:
            ok, msg = False, msg + " - the input is clipping, turn the input gain down"
        elif self.result.peak_db < -40.0:
            ok, msg = False, msg + " - the level is very low, turn the synth volume or the input gain up"
        self._report("Signal and latency", ok, msg, latency_ms=latency * 1000.0, peak_db=self.result.peak_db,
                     noise_db=self.result.noise_db, jitter_ms=float(np.ptp(lats)) * 1000.0)

    def step_tuning(self) -> None:
        hold = 0.5
        y = self._note({}, TONE_NOTE, hold)
        seg = self._steady(y, hold)
        self._saw_rms = float(np.sqrt(np.mean(np.square(seg, dtype=np.float64))))
        n_fft = 1 << 18
        spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), n_fft))
        freqs = np.fft.rfftfreq(n_fft, 1.0 / self.sr)

        def peak_near(f0: float, width: float) -> float:
            sel = (freqs > f0 * (1 - width)) & (freqs < f0 * (1 + width))
            i = np.nonzero(sel)[0][0] + int(np.argmax(spec[sel]))
            a, b, c = spec[i - 1], spec[i], spec[i + 1]
            return float(freqs[i] + 0.5 * (a - c) / (a - 2 * b + c) * (freqs[1] - freqs[0]))

        # which octave?  the fundamental is the lowest strong peak around 55, 110 or 220 Hz
        cands = [(o, spec[(freqs > 110 * 2.0 ** o * 0.94) & (freqs < 110 * 2.0 ** o * 1.06)].max()) for o in (-1, 0, 1)]
        top = max(c[1] for c in cands)
        octave = min(o for o, v in cands if v > 0.25 * top)
        f1 = peak_near(110.0 * 2.0 ** octave, 0.06)
        f8 = peak_near(8.0 * f1, 0.02) / 8.0
        cents = 1200.0 * math.log2(f8 / (110.0 * 2.0 ** octave))
        self.ph.tune_cents = cents
        self.ph.octave_offset = float(octave)
        self.result.tune_cents = cents
        self._report("Tuning", abs(cents) < 60, f"A2 sounds at {f8:.2f} Hz ({cents:+.1f} cents"
                     + (f", {octave:+d} octave" if octave else "") + ")", cents=cents, octave=octave)

    def step_envelopes(self) -> None:
        sr = self.sr
        hold = 0.6
        win = 2.4

        def env_of(updates, hold_=hold, tail=0.12):
            y = self._note(updates, ENV_NOTE, hold_, tail)
            return envelope(y, sr, win)

        e0 = env_of({})
        plateau = float(np.median(e0[int((PRE + 0.1) * sr):int((PRE + 0.4) * sr)]))
        noise = float(np.median(e0[:int(0.04 * sr)])) + 1e-9
        if plateau < 6.0 * noise:
            self._report("Envelopes", False, "signal too weak to measure the envelopes")
            return

        def crossing(env, level):
            idx = np.nonzero(env[int(PRE * sr) - 8:] >= level)[0]
            return (idx[0] - 8) / sr if len(idx) else None

        # attack
        points, ratios = [], []
        t90 = crossing(e0, 0.9 * plateau)
        fastest = max(0.0003, (t90 or 0.002) - 0.5 * win / 1000.0)
        for raw in (25, 45, 65, 85, 100, 115):
            e = env_of({"amp_attack": raw})
            t50, t9 = crossing(e, 0.5 * plateau), crossing(e, 0.9 * plateau)
            if t50 is None or t50 < 0.004:
                continue
            points.append((raw, t50))
            if t9 is not None and t50 > 0.015:
                ratios.append(t9 / t50)
            if t50 > 0.3 * hold:
                break
        if ratios:
            r = float(np.median(ratios))
            lo, hi = 0.02, 60.0
            for _ in range(60):  # bisection: the ratio falls as the overshoot grows
                mid = math.sqrt(lo * hi)
                if _attack_ratio(mid) > r:
                    lo = mid
                else:
                    hi = mid
            self.ph.attack_overshoot = float(min(max(math.sqrt(lo * hi), 0.03), 50.0))
        factor = _attack_factor(self.ph.attack_overshoot)
        if len(points) >= 2:
            table = _log_table([(0, fastest)] + [(raw, t50 * factor) for raw, t50 in points], 128, 2e-4, 60.0)
            table[0] = fastest
            self.ph.attack_table = np.maximum.accumulate(table)
            self.ph.calibrated["attack"] = True

        def slope_tau(env, t_from, t_to):
            a, b = int(t_from * sr), int(t_to * sr)
            seg = env[a:b]
            if len(seg) < 12:
                return None
            start = float(seg[:max(3, len(seg) // 50)].max())
            keep = np.nonzero(seg < max(start * 10 ** (-38 / 20.0), 2.5 * noise))[0]
            n = keep[0] if len(keep) else len(seg)
            if n < 10:
                return 0.5 * n / sr  # gone within a few milliseconds
            t = np.arange(n) / sr
            slope = np.polyfit(t, np.log(np.maximum(seg[:n], 1e-12)), 1)[0]
            return -1.0 / slope if slope < -0.05 else None

        # decay (sustain 0) and release
        dec, rel = [], []
        for raw in (0, 20, 40, 60, 80, 100, 118):
            tau = slope_tau(env_of({"amp_sustain": 0, "amp_decay": raw}), PRE + 0.006, PRE + hold - 0.01)
            if tau is not None:
                dec.append((raw, max(tau, 3e-4)))
            e = env_of({"amp_release": raw}, hold_=0.12, tail=0.6)
            tau = slope_tau(e, PRE + 0.12 + 0.004, PRE + 0.12 + 0.58)
            if tau is not None:
                rel.append((raw, max(tau, 3e-4)))
        if len(dec) >= 3:
            self.ph.decay_table = np.maximum.accumulate(_log_table(dec, 128, 2e-4, 60.0))
            self.ph.calibrated["decay"] = True
        if len(rel) >= 3:
            self.ph.release_table = np.maximum.accumulate(_log_table(rel, 128, 2e-4, 60.0))
            self.ph.calibrated["release"] = True
        ok = len(points) >= 2 and len(dec) >= 3 and len(rel) >= 3
        self._report("Envelopes", ok,
                     f"attack 64 = {self.ph.attack_table[64] * 1000:.0f} ms, decay 64 = "
                     f"{self.ph.decay_table[64] * 1000:.0f} ms, release 64 = {self.ph.release_table[64] * 1000:.0f} ms"
                     if ok else "could not measure all envelope stages (defaults kept for the rest)",
                     attack=[[r, t * factor] for r, t in points], decay=dec, release=rel,
                     overshoot=self.ph.attack_overshoot)

    def step_sustain(self) -> None:
        hold = 0.5
        levels = []
        for raw in (127, 96, 64, 32):
            y = self._note({"amp_sustain": raw, "amp_decay": 10}, TONE_NOTE, hold)
            seg = y[int((PRE + 0.25) * self.sr):int((PRE + hold - 0.01) * self.sr)]
            levels.append((raw, float(np.sqrt(np.mean(np.square(seg, dtype=np.float64))))))
        full = levels[0][1]
        xs = [math.log(r / 127.0) for r, _ in levels[1:]]
        ys = [math.log(max(l / full, 1e-6)) for _, l in levels[1:]]
        gamma = float(np.dot(xs, ys) / np.dot(xs, xs))
        ok = 0.3 < gamma < 4.0
        if ok:
            self.ph.sustain_gamma = gamma
        self._report("Sustain level", ok, f"exponent {gamma:.2f}", gamma=gamma)

    def step_cutoff(self) -> None:
        k_grid = [2.0, 1.7, 1.41, 1.2, 1.0, 0.85, 0.7, 0.55]
        raws = (25, 45, 65, 85, 105, 125, 145, 165, 185, 205, 225, 245)
        meas = [(raw,) + self._response({"filter_freq": raw}) for raw in raws]
        best_k2, best = None, None
        for k2 in (2.0, 1.6, 1.3, 1.0, 0.8):
            fits = [self._fit(r, v, b, 0, 1, 0, k_grid, k2, False) for _, r, v, b in meas]
            mid = [f[3] for (raw, _, v, _), f in zip(meas, fits) if 60 <= raw <= 200 and v.sum() >= 6]
            score = float(np.mean(mid)) if mid else float("inf")
            if best is None or score < best[0]:
                best, best_k2 = (score, fits), k2
        fits = best[1]
        pts, ks = [], []
        for (raw, r, v, b), (fc, k, _, err) in zip(meas, fits):
            if v.sum() >= 6 and err < 6.0:
                pts.append((raw, fc))
                if 60 <= raw <= 200:
                    ks.append(k)
        if len(pts) >= 4:
            self.ph.cutoff_classic = np.log2(_log_table(pts, 256, 4.0, 40000.0))
            self.ph.damping_stage2 = float(best_k2)
            self._k0 = float(np.median(ks)) if ks else 1.41
            self.ph.track_ref_note = float(FILTER_NOTE)
            self.ph.calibrated["cutoff"] = True
        self._cutoff_points = pts
        self._report("Filter cutoff curve", len(pts) >= 4,
                     f"{len(pts)} points, {2 ** self.ph.cutoff_classic[64]:.0f} Hz at 64, "
                     f"{2 ** self.ph.cutoff_classic[128]:.0f} Hz at 128, {2 ** self.ph.cutoff_classic[192]:.0f} Hz at 192"
                     if len(pts) >= 4 else "could not measure the filter (defaults kept)",
                     points=pts, fit_error=best[0])

    def step_resonance(self) -> None:
        k_grid = list(2.0 * (0.012 / 2.0) ** (np.arange(36) / 35.0))
        pts = [(0, getattr(self, "_k0", 1.41))]
        for raw in (30, 60, 85, 105, 120):
            r, v, b = self._response({"filter_freq": 150, "filter_res": raw})
            fc, k, gain, err = self._fit(r, v, b, 0, 1, 0, k_grid, self.ph.damping_stage2, True)
            if v.sum() >= 6 and err < 6.0:
                pts.append((raw, k))
        if len(pts) >= 3:
            table = _log_table(pts, 128, 0.008, 2.0)
            self.ph.damping_classic = np.minimum.accumulate(table)
            self.ph.calibrated["resonance"] = True
        # 12 dB mode: where does the cutoff sit compared to the 24 dB mode?
        r, v, b = self._response({"filter_freq": 150, "filter_slope": 0})
        fc12, _, _, err12 = self._fit(r, v, b, 0, 0, 0, [2.0, 1.7, 1.41, 1.2, 1.0, 0.85, 0.7], 1.0, False)
        if v.sum() >= 6 and err12 < 6.0:
            self.ph.slope12_shift_oct = float(math.log2(fc12) - self.ph.cutoff_classic[150])
        self._report("Resonance", len(pts) >= 3,
                     f"Q {1 / self.ph.damping_classic[0]:.1f} at 0, {1 / self.ph.damping_classic[64]:.1f} at 64, "
                     f"{1 / self.ph.damping_classic[110]:.1f} at 110", points=pts, slope12_shift=self.ph.slope12_shift_oct)

    def step_acid(self) -> None:
        k_grid = list(np.linspace(0.0, 3.9, 27))
        pts, k0s = [], []
        for raw in (45, 85, 125, 165, 205, 245):
            r, v, b = self._response({"filter_type": 1, "filter_freq": raw})
            fc, k, gain, err = self._fit(r, v, b, 1, 1, 0, k_grid[:8], 0.0, True)
            if v.sum() >= 6 and err < 6.0:
                pts.append((raw, fc))
                k0s.append(k)
        fb = [(0, float(np.median(k0s)) if k0s else 0.0)]
        for raw in (40, 80, 110, 125):
            r, v, b = self._response({"filter_type": 1, "filter_freq": 150, "filter_res": raw})
            fc, k, gain, err = self._fit(r, v, b, 1, 1, 0, k_grid, 0.0, True)
            if v.sum() >= 6 and err < 6.0:
                fb.append((raw, k))
        ok = len(pts) >= 3
        if ok:
            self.ph.cutoff_acid = np.log2(_log_table(pts, 256, 4.0, 40000.0))
            self.ph.calibrated["acid"] = True
        if len(fb) >= 3:
            self.ph.feedback_acid = np.maximum.accumulate(_lin_table(fb, 128, 0.0, 3.95))
        self._report("Acid filter", ok, f"{len(pts)} cutoff points, {len(fb) - 1} resonance points", points=pts,
                     feedback=fb)

    def step_env_depth(self) -> None:
        mod = {"mod_attack": 0, "mod_decay": 0, "mod_sustain": 127, "filter_freq": 60}
        pts = []
        for depth in (0, 14, 28, 42):
            r, v, b = self._response(dict(mod, filter_env=64 + depth))
            fc, k, _, err = self._fit(r, v, b, 0, 1, 0, [getattr(self, "_k0", 1.41)], self.ph.damping_stage2, False)
            if v.sum() >= 6 and err < 6.0 and fc < 12000.0:
                pts.append((depth, math.log2(fc)))
        ok = len(pts) >= 3
        if ok:
            x = np.array([p[0] for p in pts], dtype=np.float64)
            yv = np.array([p[1] for p in pts])
            slope = float(np.polyfit(x, yv, 1)[0])
            ok = 0.01 < slope < 0.5
            if ok:
                self.ph.env_oct_per_unit = slope
                self.ph.calibrated["env_depth"] = True
        self._report("Envelope to filter", ok, f"full depth moves the filter {self.ph.env_oct_per_unit * 63:.1f} octaves",
                     points=pts)

    def step_tracking(self) -> None:
        def cutoff_at(note, track):
            self._open = None
            r, v, b = self._response({"filter_freq": 120, "filter_track": track}, note=note)
            fc, _, _, err = self._fit(r, v, b, 0, 1, 0, [getattr(self, "_k0", 1.41)], self.ph.damping_stage2, False)
            return math.log2(fc) if v.sum() >= 6 and err < 6.0 else None

        lo, hi = cutoff_at(36, 0), cutoff_at(60, 0)
        hi7 = cutoff_at(60, 7)
        lo7 = cutoff_at(36, 7)
        self._open = None
        ok = lo is not None and hi is not None
        works = None
        if ok:
            full = (hi - lo) / 2.0
            self.ph.track_full = float(min(max(full, 0.0), 1.5))
            self.ph.calibrated["tracking"] = True
            if hi7 is not None and lo7 is not None and full > 0.3:
                works = abs((hi7 - lo7) / 2.0) < 0.35 * full
                self.ph.calibrated["filter_track_control"] = bool(works)
        msg = f"cutoff follows the keyboard {self.ph.track_full * 100:.0f} %" if ok else "could not be measured"
        if works is False:
            msg += "; the 'filter tracking' setting did not respond (older firmware?) - it will be left alone"
        self._report("Key tracking", ok, msg, full=self.ph.track_full, control_works=works)

    def step_levels(self) -> None:
        hold = 0.35

        def rms(updates, note=TONE_NOTE):
            y = self._note(updates, note, hold)
            seg = self._steady(y, hold)
            return float(np.sqrt(np.mean(np.square(seg, dtype=np.float64)))), seg

        saw = getattr(self, "_saw_rms", None) or rms({})[0]
        # mixer law
        xs, ys = [], []
        for raw in (64, 128, 192):
            level = rms({"mix_osc1": raw})[0]
            if level > 0:
                xs.append(math.log(raw / 255.0))
                ys.append(math.log(level / saw))
        gamma = float(np.dot(xs, ys) / np.dot(xs, xs)) if xs else 1.5
        ok = 0.4 < gamma < 4.0
        if ok:
            self.ph.mix_table = (np.arange(256) / 255.0) ** gamma
            self.ph.calibrated["mixer"] = True
        # waveform levels relative to the sawtooth (unit-peak RMS: sine .707, tri .577, saw .577, square 1)
        theory = (0.7071, 0.5774, 0.5774, 1.0)
        gains = list(self.ph.wave_gain)
        for wave in (0, 1, 3):
            level = rms({"osc1_wave": wave, "osc1_pw": 64})[0]
            gains[wave] = float(gains[2] * (level / saw) * (0.5774 / theory[wave]))
        self.ph.wave_gain = tuple(min(max(g, 0.1), 4.0) for g in gains)
        # sub oscillator: level of each wave and the width of the "narrow pulse"
        sub = list(self.ph.sub_gain)
        for wave in range(3):
            level, seg = rms({"mix_osc1": 0, "mix_sub": 255, "sub_wave": wave, "sub_oct": 63}, note=57)
            if wave == 1:
                n_fft = 1 << 16
                spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), n_fft))
                freqs = np.fft.rfftfreq(n_fft, 1.0 / self.sr)
                f0 = 110.0 * 2.0 ** (self.ph.tune_cents / 1200.0 + self.ph.octave_offset)
                h1 = spec[(freqs > f0 * 0.96) & (freqs < f0 * 1.04)].max()
                h2 = spec[(freqs > 2 * f0 * 0.97) & (freqs < 2 * f0 * 1.03)].max()
                width = math.acos(min(max(h2 / max(h1, 1e-12), 0.0), 0.999)) / math.pi
                self.ph.sub_pulse_width = float(min(max(width, 0.05), 0.5))
            th = (0.7071, 2.0 * math.sqrt(self.ph.sub_pulse_width * (1.0 - self.ph.sub_pulse_width)), 1.0)[wave]
            sub[wave] = float(self.ph.wave_gain[2] * (level / saw) * (0.5774 / th))
        self.ph.sub_gain = tuple(min(max(g, 0.1), 4.0) for g in sub)
        # noise: match the density between 500 Hz and 4 kHz
        y = self._note({"mix_osc1": 0, "mix_noise": 255}, TONE_NOTE, hold)
        f, p = scipy.signal.welch(self._steady(y, hold).astype(np.float64), self.sr, nperseg=2048)
        density = float(np.mean(p[(f > 500) & (f < 4000)]))
        scale = saw / (0.5774 * self.ph.wave_gain[2])  # device units per model unit
        self.ph.noise_gain = float(min(max(math.sqrt(3.0 * density * (ANALYSIS_SR / 2.0)) / scale, 0.02), 4.0))
        self.ph.calibrated["levels"] = True
        self._report("Mixer and waveform levels", ok,
                     f"mixer exponent {gamma:.2f}; sine/tri/pulse vs saw "
                     f"{self.ph.wave_gain[0]:.2f}/{self.ph.wave_gain[1]:.2f}/{self.ph.wave_gain[3]:.2f}; "
                     f"sub pulse width {self.ph.sub_pulse_width * 100:.0f} %",
                     gamma=gamma, wave_gain=list(self.ph.wave_gain), sub_gain=list(self.ph.sub_gain),
                     noise_gain=self.ph.noise_gain)

    def step_nrpn(self) -> None:
        """Do the NRPN controlled switches respond?  (waveform select is the important one)"""
        hold = 0.3

        def brightness(updates):
            y = self._note(updates, TONE_NOTE, hold)
            seg = self._steady(y, hold)
            f, p = scipy.signal.welch(seg.astype(np.float64), self.sr, nperseg=2048)
            return float(np.sum(p[f > 600]) / max(np.sum(p), 1e-30))

        saw = brightness({"osc1_wave": 2})
        sine = brightness({"osc1_wave": 0})
        works = sine < 0.3 * saw and saw > 1e-4
        self.ph.calibrated["waveform_control"] = bool(works)
        self._report("Waveform switching (NRPN)", works,
                     "the oscillator waveform follows MIDI" if works else
                     "the oscillator waveform did not change over MIDI - waveforms must be set by hand",
                     saw=saw, sine=sine)

    # -- run ------------------------------------------------------------------------
    def run(self, quick: bool = False) -> CalibrationResult:
        steps = [self.step_signal, self.step_tuning, self.step_nrpn, self.step_envelopes, self.step_sustain,
                 self.step_cutoff, self.step_resonance, self.step_env_depth, self.step_tracking, self.step_acid,
                 self.step_levels]
        if quick:
            steps = [self.step_signal, self.step_tuning, self.step_nrpn]
        self._total = len(steps)
        self.step_signal_done = False
        for i, step in enumerate(steps):
            try:
                step()
            except (Cancelled, DeviceError):
                raise
            except Exception as exc:  # a failed measurement must not abort the calibration
                if i == 0:
                    raise
                self._report(step.__doc__.strip().split("\n")[0] if step.__doc__ else step.__name__, False,
                             f"measurement failed ({exc.__class__.__name__}: {exc})")
        self.dev.set_patch(self.base, force=True)
        self.result.physics = self.ph
        return self.result
