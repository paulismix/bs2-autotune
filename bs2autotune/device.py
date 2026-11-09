"""Devices that can play a phrase with a patch and return the recorded sound.

``HardwareDevice`` (see hardware.py) drives the real Bass Station II.
``VirtualDevice`` is a stand-in used for the demo mode and for the tests: it
runs the same kind of synth model, but with *different* physics that the tuner
does not know (other cutoff / envelope curves, other gains, extra colouring,
latency, noise, free running oscillators).  Matching a sound on it exercises
the complete workflow - calibration, model search and closed-loop refinement.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Callable, Dict, List, Optional

import numpy as np
import scipy.signal

from . import engine
from .audio_util import biquad_shelf, fit_length, to_analysis
from .engine import Phrase, Physics
from .features import ANALYSIS_SR
from .params import PARAMS, baseline_patch, normalize_patch


class DeviceError(RuntimeError):
    """Something is wrong with the instrument connection (shown to the user)."""


class Device:
    """Common interface of the real and the virtual instrument."""

    name = "device"
    sr = 48000
    is_virtual = False

    def __init__(self) -> None:
        self.latency = 0.0  # seconds between sending a note and hearing it
        self.plays = 0
        self.cancel_event: Optional[threading.Event] = None

    # -- to implement ------------------------------------------------------
    def set_patch(self, patch: Dict[str, int], force: bool = False) -> None:
        raise NotImplementedError

    def play(self, phrase: Phrase, extra: float = 0.0) -> np.ndarray:
        """Play the phrase with the current patch.

        Returns the raw recording (device sample rate); sample 0 is the moment
        the phrase was started, so the sound is delayed by the latency.
        """
        raise NotImplementedError

    def close(self) -> None:
        pass

    def dump_patch(self) -> Optional[bytes]:
        """SysEx dump of the current patch, if the device can provide one."""
        return None

    # -- shared --------------------------------------------------------------
    def capture(self, patch: Dict[str, int], phrase: Phrase, n_samples: Optional[int] = None) -> np.ndarray:
        """Set the patch, play the phrase and return latency compensated audio
        at the analysis sample rate."""
        self.set_patch(patch)
        raw = self.play(phrase, extra=self.latency + 0.05)
        skip = int(round(self.latency * self.sr))
        y = to_analysis(raw[skip:], self.sr)
        if n_samples is None:
            n_samples = int(round(phrase.duration * ANALYSIS_SR))
        return fit_length(y, n_samples)

    def check_cancel(self) -> None:
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise Cancelled()


class Cancelled(Exception):
    """Raised when the user stops a running job."""


def hidden_physics(variant: int = 0) -> Physics:
    """Physics of the virtual instrument - deliberately unlike the defaults."""
    ph = Physics()
    idx256 = np.arange(256) / 255.0
    idx128 = np.arange(128) / 127.0
    # cutoff: other range and a slightly S-shaped curve; acid sits higher
    curve = idx256 + 0.06 * np.sin(2.0 * math.pi * idx256)
    ph.cutoff_classic = np.log2(24.0) + curve * math.log2(19000.0 / 24.0)
    ph.cutoff_acid = ph.cutoff_classic + 0.45
    ph.slope12_shift_oct = 0.2
    ph.damping_classic = 1.0 / (0.75 * (26.0 / 0.75) ** (idx128 ** 1.35))
    ph.damping_stage2 = 1.0
    ph.feedback_acid = 3.3 * idx128 ** 0.9
    ph.acid_comp = 0.5
    ph.filter_sat = 0.9
    ph.env_oct_per_unit = 11.0 / 63.0
    ph.lfo2_oct_per_unit = 4.0 / 127.0
    ph.track_ref_note = 48.0
    ph.track_full = 0.96
    ph.drive_table = 1.0 + 15.0 * idx128 ** 1.4
    ph.attack_table = 0.0012 * (9.0 / 0.0012) ** (idx128 ** 1.15)
    ph.attack_table[0] = 0.0007
    ph.decay_table = 0.0016 * (2.6 / 0.0016) ** (idx128 ** 0.92)
    ph.release_table = 0.0020 * (3.2 / 0.0020) ** (idx128 ** 0.95)
    ph.attack_overshoot = 1.2
    ph.sustain_gamma = 1.25
    ph.dist_table = 1.0 + 40.0 * idx128 ** 1.7
    ph.mix_table = idx256 ** 1.0
    ph.wave_gain = (0.72, 0.8, 1.0, 1.18)
    ph.sub_gain = (1.05, 0.9, 1.15)
    ph.sub_pulse_width = 0.18
    ph.noise_gain = 0.5
    ph.ring_gain = 0.8
    ph.headroom = 0.42
    ph.osc_lp_hz = 16500.0
    ph.pitch_env_semis_per_unit = 30.0 / 63.0
    ph.lfo1_semis_per_unit = 9.0 / 127.0
    ph.env_pw_per_unit = 0.4 / 63.0
    ph.lfo2_pw_per_unit = 0.38 / 63.0
    ph.lfo_rate_table = 0.08 * (110.0 / 0.08) ** idx256
    ph.glide_table = np.concatenate([[0.0], 0.003 * (1.5 / 0.003) ** (np.arange(127) / 126.0)])
    ph.out_hp_hz = 18.0
    ph.tune_cents = 4.0
    if variant == 1:  # a second, different unit for robustness tests
        ph.cutoff_classic = np.log2(14.0) + idx256 ** 1.08 * math.log2(24000.0 / 14.0)
        ph.cutoff_acid = ph.cutoff_classic - 0.3
        ph.decay_table = ph.decay_table * 1.8
        ph.attack_table = ph.attack_table * 0.5
        ph.env_oct_per_unit = 7.5 / 63.0
        ph.mix_table = idx256 ** 2.0
        ph.tune_cents = -7.0
    return ph


class VirtualDevice(Device):
    """A simulated Bass Station II with hidden, imperfect physics."""

    name = "Virtual BS2"
    is_virtual = True

    def __init__(self, sr: int = 48000, seed: int = 1234, variant: int = 0, latency: float = 0.0187,
                 noise_db: float = -82.0, realtime: bool = False, dead_params: Optional[List[str]] = None,
                 colour: bool = True, level: float = 0.5):
        super().__init__()
        self.sr = sr
        self.true_physics = hidden_physics(variant)
        self.true_latency = latency
        self.noise = 10.0 ** (noise_db / 20.0)
        self.realtime = realtime
        self.colour = colour
        self.level = level
        self.rng = np.random.default_rng(seed)
        self.patch = baseline_patch()
        self.patch["filter_freq"] = 180  # whatever happened to be loaded before
        self.dead_params = set(dead_params or [])
        self._eq = None

    def set_patch(self, patch: Dict[str, int], force: bool = False) -> None:
        for key, value in patch.items():
            if key in PARAMS and key not in self.dead_params:
                self.patch[key] = PARAMS[key].clamp(value)

    def _colour(self, y: np.ndarray) -> np.ndarray:
        if self._eq is None:
            lo = biquad_shelf(self.sr, 95.0, 1.8, high=False)
            hi = biquad_shelf(self.sr, 5200.0, -3.5, high=True)
            lp = scipy.signal.butter(2, 17500.0, fs=self.sr)
            self._eq = (lo, hi, lp)
        y = y.astype(np.float64)
        y = y + 0.08 * (y * y - float(np.mean(y * y)))  # a little asymmetry (even harmonics)
        for b, a in self._eq:
            y = scipy.signal.lfilter(b, a, y)
        return y

    def play(self, phrase: Phrase, extra: float = 0.0) -> np.ndarray:
        self.check_cancel()
        self.plays += 1
        sr = self.sr
        latency = self.true_latency + float(self.rng.normal(0.0, 0.0006))
        total = phrase.duration + extra
        n = int(round(total * sr))
        ph = self.true_physics
        drift = float(self.rng.normal(0.0, 1.2))
        phys = ph
        P = engine.resolve(self.patch, phys, float(sr), seed=int(self.rng.integers(1, 2 ** 30)),
                           phase1=float(self.rng.random()), phase2=float(self.rng.random()))
        P[engine.I_O1_SEMI] += drift / 100.0
        P[engine.I_O2_SEMI] += (drift + float(self.rng.normal(0.0, 0.8))) / 100.0
        on, off, note = phrase.arrays(float(sr))
        dry = np.zeros(n, dtype=np.float32)
        engine._render_kernel(dry, on, off, note, P)
        y = self._colour(dry) if self.colour else dry.astype(np.float64)
        y = y * self.level
        shift = int(round(latency * sr))
        out = np.zeros(n, dtype=np.float64)
        if shift < n:
            out[shift:] = y[:n - shift]
        t = np.arange(n) / sr
        out += self.noise * self.rng.standard_normal(n)
        out += 0.5 * self.noise * np.sin(2.0 * math.pi * 50.0 * t + float(self.rng.random()) * 6.28)
        if self.realtime:
            end = time.time() + total
            while time.time() < end:
                self.check_cancel()
                time.sleep(min(0.05, max(0.0, end - time.time())))
        return np.clip(out, -1.0, 1.0).astype(np.float32)
