"""Understanding the reference recording.

The reference is turned into what the synth has to play and what it has to
sound like:

* the audio is decoded (any common format) and, for a full mix, the bass is
  isolated (see separation.py),
* the pitch of the bass line is tracked and cut into notes,
* a short excerpt with enough notes is chosen (or given by the user); its
  notes become the *phrase* that the Bass Station plays for every comparison.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.signal
from numba import njit

from .audio_util import resample
from .engine import Note, Phrase
from .features import ANALYSIS_SR, DEFAULT_CONFIG, extract

# ---------------------------------------------------------------------------
# decoding
# ---------------------------------------------------------------------------


class AudioError(RuntimeError):
    """The reference file could not be read (message is shown to the user)."""


def load_audio(path: str) -> Tuple[np.ndarray, int]:
    """Decode an audio file.  Returns (samples[n, channels] float32, sample rate)."""
    import soundfile as sf

    try:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        return data, int(sr)
    except Exception as first_error:  # not something libsndfile understands (m4a, aac ...)
        converted = _convert_to_wav(path)
        if converted is None:
            raise AudioError(
                f"Cannot read this audio file ({first_error}). Convert it to WAV, AIFF, FLAC or MP3, "
                "or install ffmpeg.") from first_error
        try:
            data, sr = sf.read(converted, dtype="float32", always_2d=True)
            return data, int(sr)
        finally:
            shutil.rmtree(os.path.dirname(converted), ignore_errors=True)


def _convert_to_wav(path: str) -> Optional[str]:
    """Use the macOS converter or ffmpeg for formats libsndfile cannot read."""
    tmp = tempfile.mkdtemp(prefix="bs2autotune_")
    out = os.path.join(tmp, "converted.wav")
    commands = []
    if shutil.which("afconvert"):
        commands.append(["afconvert", "-f", "WAVE", "-d", "LEI16", path, out])
    if shutil.which("ffmpeg"):
        commands.append(["ffmpeg", "-y", "-loglevel", "error", "-i", path, "-vn", "-f", "wav", out])
    for cmd in commands:
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=300)
            if os.path.exists(out) and os.path.getsize(out) > 44:
                return out
        except Exception:
            continue
    shutil.rmtree(tmp, ignore_errors=True)
    return None


def to_mono(data: np.ndarray) -> np.ndarray:
    return data.mean(axis=1).astype(np.float32) if data.ndim == 2 else data.astype(np.float32)


# ---------------------------------------------------------------------------
# pitch tracking (YIN difference function + Viterbi path)
# ---------------------------------------------------------------------------

_PITCH_SR = 8000
_HOP = 80  # 10 ms
_WIN = 1024
_N_CAND = 6


@njit(cache=True, nogil=True)
def _viterbi(cand_f, cand_c, n_cand, energy_ok, uv_cost, jump_cost, switch_cost):
    """Best path through per-frame pitch candidates; state n_c = unvoiced."""
    n_t, n_c = cand_f.shape
    n_s = n_c + 1
    cost = np.full((n_t, n_s), 1e9)
    back = np.zeros((n_t, n_s), dtype=np.int64)
    for s in range(n_cand[0]):
        cost[0, s] = cand_c[0, s] + (0.0 if energy_ok[0] else 1.0)
    cost[0, n_c] = uv_cost
    for t in range(1, n_t):
        for s in range(n_s):
            if s < n_c:
                if s >= n_cand[t]:
                    continue
                emit = cand_c[t, s] + (0.0 if energy_ok[t] else 1.0)
            else:
                emit = uv_cost if energy_ok[t] else 0.0
            best = 1e9
            arg = 0
            for p in range(n_s):
                c = cost[t - 1, p]
                if c >= 1e8:
                    continue
                if p < n_c and s < n_c:
                    d = abs(12.0 * math.log2(cand_f[t, s] / cand_f[t - 1, p]))
                    if d > 0.5:
                        c += jump_cost * (d if d < 12.0 else 12.0)
                elif (p < n_c) != (s < n_c):
                    c += switch_cost
                if c < best:
                    best = c
                    arg = p
            cost[t, s] = best + emit
            back[t, s] = arg
    path = np.zeros(n_t, dtype=np.int64)
    path[n_t - 1] = int(np.argmin(cost[n_t - 1]))
    for t in range(n_t - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def track_pitch(y: np.ndarray, sr: int, fmin: float = 27.0, fmax: float = 420.0) -> Tuple[np.ndarray, np.ndarray]:
    """Fundamental frequency of a monophonic bass line.

    Returns (times [s], f0 [Hz]) every 10 ms; f0 is 0 where nothing is playing.
    """
    x = resample(y, sr, _PITCH_SR).astype(np.float64)
    sos = scipy.signal.butter(4, [22.0, 1100.0], btype="bandpass", fs=_PITCH_SR, output="sos")
    x = scipy.signal.sosfiltfilt(sos, x) if len(x) > 64 else x
    tau_max = int(_PITCH_SR / fmin) + 2
    tau_min = max(2, int(_PITCH_SR / fmax))
    integ = _WIN - tau_max
    n_frames = max(1, 1 + (len(x) - 1) // _HOP)
    padded = np.zeros(n_frames * _HOP + _WIN)
    padded[_WIN // 4:_WIN // 4 + len(x)] = x  # frame t is roughly centred on t * hop
    cand_f = np.ones((n_frames, _N_CAND))
    cand_c = np.ones((n_frames, _N_CAND))
    n_cand = np.zeros(n_frames, dtype=np.int64)
    frame_energy = np.zeros(n_frames)
    n_fft = 2048
    for a in range(0, n_frames, 1024):
        b = min(n_frames, a + 1024)
        idx = (np.arange(a, b) * _HOP)[:, None] + np.arange(_WIN)[None, :]
        frames = padded[idx]
        head = frames[:, :integ]
        spec_a = np.fft.rfft(head, n_fft, axis=1)
        spec_b = np.fft.rfft(frames, n_fft, axis=1)
        corr = np.fft.irfft(np.conj(spec_a) * spec_b, n_fft, axis=1)[:, :tau_max + 1]
        csum = np.concatenate([np.zeros((b - a, 1)), np.cumsum(frames ** 2, axis=1)], axis=1)
        e0 = csum[:, integ][:, None]
        taus = np.arange(tau_max + 1)
        e_tau = csum[:, taus + integ] - csum[:, taus]
        diff = np.maximum(e0 + e_tau - 2.0 * corr, 0.0)
        running = np.cumsum(diff[:, 1:], axis=1)
        cmnd = np.ones_like(diff)
        cmnd[:, 1:] = diff[:, 1:] * taus[1:][None, :] / np.maximum(running, 1e-12)
        frame_energy[a:b] = e0[:, 0] / integ
        seg = cmnd[:, tau_min:tau_max]
        is_min = (seg[:, 1:-1] < seg[:, :-2]) & (seg[:, 1:-1] <= seg[:, 2:]) & (seg[:, 1:-1] < 0.6)
        for i in range(b - a):
            pos = np.nonzero(is_min[i])[0] + 1
            if len(pos) == 0:
                continue
            vals = seg[i, pos]
            keep = pos[np.argsort(vals)[:_N_CAND]]
            for j, p in enumerate(sorted(keep)):
                tau = p + tau_min
                y0, y1, y2 = cmnd[i, tau - 1], cmnd[i, tau], cmnd[i, tau + 1]
                denom = y0 - 2.0 * y1 + y2
                shift = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
                cand_f[a + i, j] = _PITCH_SR / (tau + min(max(shift, -1.0), 1.0))
                # a tiny preference for the shorter period keeps sub-octave errors away
                cand_c[a + i, j] = float(y1) + 0.02 * math.log2(tau / tau_min)
            n_cand[a + i] = len(keep)
    peak = float(frame_energy.max()) if n_frames else 0.0
    energy_ok = frame_energy > peak * 10.0 ** (-48.0 / 10.0)
    path = _viterbi(cand_f, cand_c, n_cand, energy_ok, 0.32, 0.03, 0.12)
    f0 = np.where(path < _N_CAND, cand_f[np.arange(n_frames), np.minimum(path, _N_CAND - 1)], 0.0)
    f0[(path >= _N_CAND) | (path >= n_cand)] = 0.0
    times = np.arange(n_frames) * _HOP / _PITCH_SR
    return times, f0


# ---------------------------------------------------------------------------
# notes
# ---------------------------------------------------------------------------


@dataclass
class RefNote:
    start: float
    end: float
    note: int
    level_db: float
    cents: float  # deviation of the played pitch from the equal tempered note

    def to_list(self) -> list:
        return [round(self.start, 4), round(self.end, 4), int(self.note), round(self.level_db, 1)]


def _onset_strength(b_power: np.ndarray) -> np.ndarray:
    """Spectral flux of the transient view (4 ms frames)."""
    floor = float(b_power.max()) * 1e-6 + 1e-20
    logp = np.log10(b_power + floor)
    lag = 3
    flux = np.zeros(len(logp))
    d = np.maximum(logp[lag:] - logp[:-lag], 0.0)
    flux[lag:] = d.sum(axis=1)
    return flux


def detect_onsets(b_power: np.ndarray, hop_s: float, sensitivity: float = 0.5) -> np.ndarray:
    """Times at which notes start.

    A note start is a broadband event: many bands rise together and the level
    jumps.  Beating between detuned oscillators and filter movement also make
    single bands rise, but far less - hence the two criteria.  ``sensitivity``
    (0..1) trades missed re-triggers against false splits.
    """
    flux = _onset_strength(b_power)
    env = 10.0 * np.log10(b_power.sum(axis=1) + 1e-20)
    back = max(2, int(round(0.032 / hop_s)))
    rise = np.zeros_like(env)
    for i in range(1, len(env)):
        rise[i] = env[i] - env[max(0, i - back):i].min()
    peaks, _ = scipy.signal.find_peaks(flux, distance=max(1, int(round(0.07 / hop_s))))
    if len(peaks) == 0:
        return np.zeros(0)
    heights = flux[peaks]
    top = np.sort(heights)[-max(1, len(heights) // 4):]
    ref = float(np.median(top))  # a typical clear onset
    strong = 0.75 - 0.5 * sensitivity  # fraction of a clear onset that is accepted on its own
    weak = 0.3 - 0.2 * sensitivity
    need_rise = 20.0 - 12.0 * sensitivity
    times = []
    for p in peaks:
        # the level jump around the flux peak
        jump = float(rise[max(0, p - 2):p + 4].max())
        if flux[p] >= strong * ref or (flux[p] >= weak * ref and jump >= need_rise):
            lo = max(0, p - int(round(0.03 / hop_s)))
            seg = flux[lo:p + 1]
            foot = lo + int(np.nonzero(seg >= 0.25 * flux[p])[0][0])
            times.append(max(0.0, foot * hop_s))
    return np.array(times)


def transcribe(y: np.ndarray, sr: int = ANALYSIS_SR, sensitivity: float = 0.4) -> Tuple[List[RefNote], dict]:
    """Notes of a monophonic bass line.  Returns (notes, details for display)."""
    cfg = DEFAULT_CONFIG
    y = resample(y, sr, cfg.sr) if sr != cfg.sr else np.asarray(y, dtype=np.float32)
    times, f0 = track_pitch(y, cfg.sr)
    feats = extract(y, cfg)
    hop_b = cfg.hop_b / cfg.sr
    env = 10.0 * np.log10(feats.b.sum(axis=1) + 1e-20)
    env_peak = float(env.max())
    onset_times = detect_onsets(feats.b, hop_b, sensitivity)

    midi = np.where(f0 > 0, 69.0 + 12.0 * np.log2(np.maximum(f0, 1e-6) / 440.0), np.nan)
    hop_p = times[1] - times[0] if len(times) > 1 else 0.01
    voiced = np.isfinite(midi)
    # global tuning of the recording relative to A = 440 Hz
    if voiced.sum() > 5:
        dev = midi[voiced] - np.round(midi[voiced])
        # circular median: robust when the tuning sits near +-50 cents
        ang = np.angle(np.mean(np.exp(2j * np.pi * dev)))
        tuning = float(ang / (2 * np.pi))
    else:
        tuning = 0.0
    semis = midi - tuning
    if voiced.sum() > 5:  # remove single-frame glitches
        filled = np.where(voiced, semis, 0.0)
        med = scipy.signal.medfilt(filled, 5)
        semis = np.where(voiced & (np.abs(med - semis) > 0.5) & (med > 0), med, semis)

    # boundaries: onsets, pitch changes, voicing changes
    notes: List[RefNote] = []
    n = len(times)
    i = 0
    while i < n:
        if not voiced[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and voiced[j + 1]:
            j += 1
        # voiced run [i, j]; split inside it
        cuts = [i]
        run_t0, run_t1 = times[i], times[j] + hop_p
        for ot in onset_times:
            if run_t0 + 0.04 < ot < run_t1 - 0.03:
                cuts.append(int(round(ot / hop_p)))
        m = i
        ref = semis[i]
        count = 0
        for t in range(i + 1, j + 1):
            if abs(semis[t] - ref) > 0.7:
                count += 1
                if count >= 3:
                    cuts.append(t - 2)
                    ref = semis[t]
                    count = 0
            else:
                count = 0
                ref = 0.9 * ref + 0.1 * semis[t]
        cuts = sorted(set(c for c in cuts if i <= c <= j))
        cuts.append(j + 1)
        merged = [cuts[0]]
        for c in cuts[1:]:
            if c - merged[-1] >= max(3, int(round(0.035 / hop_p))) or c == j + 1:
                merged.append(c)
        if merged[-1] != j + 1:
            merged[-1] = j + 1
        for a, b in zip(merged[:-1], merged[1:]):
            seg = semis[a:b]
            seg = seg[np.isfinite(seg)]
            if len(seg) < 3:
                continue
            core = seg[len(seg) // 5:] if len(seg) > 6 else seg
            pitch = float(np.median(core))
            if len(core) >= 5 and float(np.mean(np.abs(core - pitch))) > 0.6:
                continue  # no steady pitch: a drum or a glide, not a note
            start, end = times[a], times[b - 1] + hop_p
            notes.append(RefNote(float(start), float(end), int(round(pitch)), 0.0, 100.0 * (pitch - round(pitch))))
        i = j + 1

    # merge neighbours with the same pitch that have no onset between them
    merged_notes: List[RefNote] = []
    for nt in notes:
        if merged_notes:
            prev = merged_notes[-1]
            gap = nt.start - prev.end
            onset_between = bool(np.any((onset_times > prev.start + 0.04) & (onset_times < nt.start + 0.03)))
            if nt.note == prev.note and gap < 0.03 and not onset_between:
                prev.end = nt.end
                continue
            if gap < 0.03 and not onset_between and abs(nt.note - prev.note) % 12 == 0:
                # a sliver at a note boundary or a momentary octave error of the
                # pitch tracker belongs to its (longer) neighbour
                if (prev.end - prev.start) < 0.06:
                    prev.end = nt.end
                    prev.note = nt.note
                    prev.cents = nt.cents
                    continue
                if (nt.end - nt.start) < 0.09:
                    prev.end = nt.end
                    continue
        merged_notes.append(nt)

    # refine all starts to the detected onsets first (the pitch tracker needs a
    # few periods before it reports a new note, the onset detector does not)
    for nt in merged_notes:
        near = onset_times[(onset_times > nt.start - 0.06) & (onset_times < nt.start + 0.05)]
        if len(near):
            nt.start = float(near[np.argmin(np.abs(near - nt.start))])
    # then find where the held part of every note ends, and its level
    power = feats.b.sum(axis=1).astype(np.float64)
    csum = np.concatenate([[0.0], np.cumsum(power)])
    out: List[RefNote] = []
    for idx, nt in enumerate(merged_notes):
        nxt = merged_notes[idx + 1].start if idx + 1 < len(merged_notes) else len(env) * hop_b
        a = int(max(0, round(nt.start / hop_b)))
        b = int(min(len(env), round(min(nt.end + 0.25, nxt - 0.012) / hop_b)))
        if b - a < 4:
            continue
        # level averaged over one period of the note: no ripple at the pitch rate
        period = 1.0 / (440.0 * 2.0 ** ((nt.note - 69) / 12.0))
        half = max(1, int(round(0.5 * period / hop_b)))
        idxs = np.arange(a, b)
        lo = np.clip(idxs - half, 0, len(power))
        hi = np.clip(idxs + half + 1, 0, len(power))
        seg = 10.0 * np.log10((csum[hi] - csum[lo]) / np.maximum(hi - lo, 1) + 1e-20)
        pitched = max(4, int(round(0.6 * (min(nt.end, nxt) - nt.start) / hop_b)))
        level = float(np.median(seg[:min(len(seg), pitched)]))
        strong = np.nonzero(seg >= level - 10.0)[0]
        held_end = (a + int(strong[-1]) + 1) * hop_b - 0.5 * half * hop_b - 0.006 if len(strong) else nt.end
        nt.end = float(min(max(held_end, nt.start + 0.03), nxt - 0.008))
        nt.level_db = level - 10.0 * math.log10(float(power.max()) + 1e-20)
        if nt.end - nt.start >= 0.035 and nt.level_db > -40.0:
            out.append(nt)
    for nt in out:  # the analysis windows see an onset a few milliseconds early
        nt.start = float(nt.start + 0.003)
    info = {"tuning_cents": 100.0 * tuning, "onsets": [float(t) for t in onset_times]}
    return out, info


# ---------------------------------------------------------------------------
# excerpt and phrase
# ---------------------------------------------------------------------------


def choose_excerpt(notes: Sequence[RefNote], duration: float, target_len: float = 4.0,
                   max_len: float = 6.0) -> Tuple[float, float]:
    """Pick a window that contains a representative part of the line."""
    if not notes:
        return 0.0, min(duration, target_len)
    best = None
    for i, first in enumerate(notes):
        t0 = max(0.0, first.start - 0.06)
        for j in range(i, len(notes)):
            last = notes[j]
            nxt = notes[j + 1].start if j + 1 < len(notes) else duration
            t1 = min(last.end + 0.3, nxt - 0.015, duration)
            length = t1 - t0
            if length > max_len:
                break
            inside = notes[i:j + 1]
            sounding = sum(n.end - n.start for n in inside)
            pitches = len({n.note for n in inside})
            longest = max(n.end - n.start for n in inside)
            loud = float(np.mean([n.level_db for n in inside]))
            score = (min(sounding, target_len) + 0.25 * min(pitches, 4) + 0.6 * min(longest, 0.6)
                     + 0.1 * min(len(inside), 8) + 0.03 * loud - 0.4 * abs(length - target_len) / target_len)
            if length < 0.8:
                score -= 2.0
            if best is None or score > best[0]:
                best = (score, t0, t1)
    return (best[1], best[2]) if best else (0.0, min(duration, target_len))


@dataclass
class Reference:
    """Everything the matching needs to know about the reference."""

    audio: np.ndarray  # excerpt, mono, analysis sample rate
    phrase: Phrase
    notes: List[Tuple[float, float, float]]  # (start, end, MIDI note) relative to the excerpt
    tuning_cents: float = 0.0
    band_reliability: Optional[np.ndarray] = None
    start: float = 0.0  # position of the excerpt in the source file [s]
    end: float = 0.0
    transpose: int = 0


def build_reference(bass: np.ndarray, notes: Sequence[RefNote], t0: float, t1: float, tuning_cents: float = 0.0,
                    band_reliability: Optional[np.ndarray] = None, transpose: int = 0,
                    sr: int = ANALYSIS_SR) -> Reference:
    """Cut the excerpt [t0, t1] out of the (isolated) bass and build the phrase to play."""
    a, b = int(round(t0 * sr)), int(round(t1 * sr))
    audio = np.array(bass[a:b], dtype=np.float32)
    inside = [n for n in notes if n.start >= t0 - 0.02 and n.start < t1 - 0.05]
    if not inside:
        raise AudioError("No notes were found in the selected part of the reference. Select a part where the "
                         "bass plays clearly.")
    phrase_notes = []
    for n in inside:
        start = max(0.0, n.start - t0)
        end = min(n.end - t0, (t1 - t0) - 0.02)
        if end - start < 0.03:
            continue
        phrase_notes.append(Note(float(start), float(end), int(n.note + transpose), 100))
    if not phrase_notes:
        raise AudioError("The selected part is too short to contain a complete note.")
    # fade the edges so that a cut through a neighbouring note does not click
    fade = min(len(audio) // 20, int(0.008 * sr))
    if fade > 1:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        audio[:fade] *= ramp
        audio[-fade:] *= ramp[::-1]
    phrase = Phrase(phrase_notes, float(len(audio)) / sr)
    return Reference(audio, phrase, [(n.start, n.end, float(n.note)) for n in phrase_notes], tuning_cents,
                     band_reliability, float(t0), float(t1), transpose)


def note_name(midi: int) -> str:
    names = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    return f"{names[int(midi) % 12]}{int(midi) // 12 - 1}"
