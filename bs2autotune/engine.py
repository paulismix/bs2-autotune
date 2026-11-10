"""A fast software model of the Bass Station II signal path.

The model is *not* an exact emulation.  It reproduces the structure of the
instrument (two oscillators + sub, mixer, overdrive, multimode / ladder
filter, two ADSR envelopes, two LFOs, distortion, VCA) and maps the raw MIDI
values to physical quantities through the tables in :class:`Physics`.  Those
tables start as educated guesses and are replaced by measurements of the real
instrument during calibration, so the model follows the unit on the desk.

The model is used for three things:

* a broad search for a patch that sounds like the reference (thousands of
  renders, no hardware needed),
* predicting what a small parameter change will do while fine tuning on the
  hardware,
* the built-in "virtual BS2" used for the demo mode and the tests.

The audio kernel is compiled with numba and releases the GIL, so many
candidates can be rendered in parallel threads.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from numba import njit

from .params import bipolar7, bipolar8, coarse_semitones, fine_cents, normalize_patch, pulse_width_percent

# ---------------------------------------------------------------------------
# engine parameter vector layout
# ---------------------------------------------------------------------------

_names = [
    "SR",
    # oscillator 1
    "O1_WAVE", "O1_SEMI", "O1_PW", "O1_ENV_SEMI", "O1_LFO_SEMI", "O1_ENV_PW", "O1_LFO_PW", "O1_GAIN", "O1_PHASE",
    # oscillator 2
    "O2_WAVE", "O2_SEMI", "O2_PW", "O2_ENV_SEMI", "O2_LFO_SEMI", "O2_ENV_PW", "O2_LFO_PW", "O2_GAIN", "O2_PHASE",
    "SYNC",
    # sub oscillator, noise, ring
    "SUB_WAVE", "SUB_DIV", "SUB_GAIN", "SUB_PW", "NOISE_GAIN", "RING_GAIN",
    # drive stage
    "DRIVE", "HEADROOM",
    # filter
    "F_TYPE", "F_SLOPE", "F_SHAPE", "F_LOG2HZ", "F_K", "F_K2", "F_ENV_OCT", "F_LFO_OCT", "F_TRACK", "F_TRACK_REF",
    "F_FM_OCT", "F_SAT", "F_COMP",
    # envelopes
    "A_ATT", "A_DEC", "A_SUS", "A_REL", "A_TRIG",
    "M_ATT", "M_DEC", "M_SUS", "M_REL", "M_TRIG",
    "ENV_OVR",
    # distortion
    "DIST",
    # LFOs
    "L1_RATE", "L1_WAVE", "L1_DELAY", "L1_KEYSYNC", "L2_RATE", "L2_WAVE", "L2_DELAY", "L2_KEYSYNC",
    # glide and output
    "GLIDE", "OUT_HP", "OUT_GAIN", "OSC_LP", "SEED",
]
IDX: Dict[str, int] = {name: i for i, name in enumerate(_names)}
N_ENGINE = len(_names)
globals().update({f"I_{name}": i for name, i in IDX.items()})

CONTROL_RATE = 16  # samples between modulation updates


# ---------------------------------------------------------------------------
# numba kernel
# ---------------------------------------------------------------------------


@njit(cache=True, nogil=True, inline="always")
def _sat(x):
    """Smooth saturation (Pade approximation of tanh)."""
    if x > 3.0:
        return 1.0
    if x < -3.0:
        return -1.0
    x2 = x * x
    return x * (27.0 + x2) / (27.0 + 9.0 * x2)


@njit(cache=True, nogil=True, inline="always")
def _blep(t, dt):
    if t < dt:
        x = t / dt
        return x + x - x * x - 1.0
    if t > 1.0 - dt:
        x = (t - 1.0) / dt
        return x * x + x + x + 1.0
    return 0.0


@njit(cache=True, nogil=True, inline="always")
def _osc(wave, t, dt, pw):
    """Band limited oscillator sample for phase t in [0, 1)."""
    if wave == 2:  # sawtooth
        return 2.0 * t - 1.0 - _blep(t, dt)
    if wave == 3:  # pulse (DC free)
        t2 = t + pw
        if t2 >= 1.0:
            t2 -= 1.0
        return (2.0 * t - _blep(t, dt)) - (2.0 * t2 - _blep(t2, dt))
    if wave == 1:  # triangle
        return 1.0 - 4.0 * abs(t - 0.5)
    return math.sin(6.283185307179586 * t)


@njit(cache=True, nogil=True, inline="always")
def _lfo(wave, ph, sh):
    if wave == 0:  # triangle
        return 1.0 - 4.0 * abs(ph - 0.5)
    if wave == 1:  # sawtooth (falling)
        return 1.0 - 2.0 * ph
    if wave == 2:  # square
        return 1.0 if ph < 0.5 else -1.0
    return sh  # sample and hold


@njit(cache=True, nogil=True)
def _render_kernel(out, ev_on, ev_off, ev_note, P):
    sr = P[I_SR]
    n_total = out.shape[0]
    n_ev = ev_on.shape[0]
    cr = CONTROL_RATE
    dt_block = cr / sr

    o1_wave = int(P[I_O1_WAVE])
    o2_wave = int(P[I_O2_WAVE])
    sub_wave = int(P[I_SUB_WAVE])
    sub_div = P[I_SUB_DIV]
    sub_pw = P[I_SUB_PW]
    g1 = P[I_O1_GAIN]
    g2 = P[I_O2_GAIN]
    gs = P[I_SUB_GAIN]
    gn = P[I_NOISE_GAIN]
    gr = P[I_RING_GAIN]
    sync = P[I_SYNC] > 0.5
    use_o2 = (g2 > 0.0) or (gr > 0.0) or (P[I_F_FM_OCT] != 0.0)
    drive = P[I_DRIVE] * P[I_HEADROOM]
    f_type = int(P[I_F_TYPE])
    f_slope = int(P[I_F_SLOPE])
    f_shape = int(P[I_F_SHAPE])
    f_k = P[I_F_K]
    f_k2 = P[I_F_K2]
    f_fm = P[I_F_FM_OCT]
    f_sat = P[I_F_SAT]
    f_comp = P[I_F_COMP]
    dist = P[I_DIST]
    dist_in = dist * 0.25
    dist_out = 4.0 / math.sqrt(dist) if dist > 1.0 else 4.0
    out_gain = P[I_OUT_GAIN]
    hp_r = 1.0 - 6.283185307179586 * P[I_OUT_HP] / sr
    if hp_r < 0.0:
        hp_r = 0.0
    osc_lp = P[I_OSC_LP]
    lp_a = 1.0
    if osc_lp > 0.0:
        lp_a = 1.0 - math.exp(-6.283185307179586 * osc_lp / sr)

    # envelope coefficients
    ovr = P[I_ENV_OVR]
    att_ln = math.log((1.0 + ovr) / ovr)
    a_att = P[I_A_ATT]
    m_att = P[I_M_ATT]
    a_ca = math.exp(-dt_block * att_ln / a_att) if a_att > 1e-5 else 0.0
    m_ca = math.exp(-dt_block * att_ln / m_att) if m_att > 1e-5 else 0.0
    a_cd = math.exp(-dt_block / P[I_A_DEC]) if P[I_A_DEC] > 1e-5 else 0.0
    m_cd = math.exp(-dt_block / P[I_M_DEC]) if P[I_M_DEC] > 1e-5 else 0.0
    a_crel = math.exp(-dt_block / P[I_A_REL]) if P[I_A_REL] > 1e-5 else 0.0
    m_crel = math.exp(-dt_block / P[I_M_REL]) if P[I_M_REL] > 1e-5 else 0.0
    a_sus = P[I_A_SUS]
    m_sus = P[I_M_SUS]
    a_trig = int(P[I_A_TRIG])
    m_trig = int(P[I_M_TRIG])

    glide_tau = P[I_GLIDE]
    glide_c = math.exp(-dt_block / glide_tau) if glide_tau > 1e-5 else 0.0

    l1_inc = P[I_L1_RATE] * dt_block
    l2_inc = P[I_L2_RATE] * dt_block
    l1_wave = int(P[I_L1_WAVE])
    l2_wave = int(P[I_L2_WAVE])
    l1_delay = P[I_L1_DELAY]
    l2_delay = P[I_L2_DELAY]
    use_l1 = (P[I_O1_LFO_SEMI] != 0.0) or (P[I_O2_LFO_SEMI] != 0.0)
    use_l2 = (P[I_F_LFO_OCT] != 0.0) or (P[I_O1_LFO_PW] != 0.0) or (P[I_O2_LFO_PW] != 0.0)

    rng = int(P[I_SEED]) & 0x7FFFFFFF
    if rng == 0:
        rng = 0x1234567

    # state
    ph1 = P[I_O1_PHASE] % 1.0
    ph2 = P[I_O2_PHASE] % 1.0
    phs = (ph1 * sub_div) % 1.0
    inc1 = 0.0
    inc2 = 0.0
    pw1 = 0.5
    pw2 = 0.5
    a_lvl = 0.0
    a_stage = 0
    m_lvl = 0.0
    m_stage = 0
    amp = 0.0
    damp = 0.0
    g = 0.0
    dg = 0.0
    fc_block = 100.0
    s1 = 0.0
    s2 = 0.0
    s3 = 0.0
    s4 = 0.0
    l1_ph = 0.0
    l2_ph = 0.0
    l1_sh = 0.0
    l2_sh = 0.0
    l1_age = 0.0
    l2_age = 0.0
    gate = False
    off_time = -1
    pitch = 60.0
    pitch_target = 60.0
    started = False
    hp_x = 0.0
    hp_y = 0.0
    lp_z = 0.0
    ei = 0
    fc_max = 0.45 * sr
    pi_sr = 3.141592653589793 / sr

    for n in range(n_total):
        # ---- note events -------------------------------------------------
        while ei < n_ev and ev_on[ei] <= n:
            legato = gate
            pitch_target = ev_note[ei]
            if (not started) or glide_c == 0.0:
                pitch = pitch_target
            started = True
            if a_trig == 0 or not legato:
                a_stage = 1
            if m_trig == 0 or not legato:
                m_stage = 1
            if not legato:
                l1_age = 0.0
                l2_age = 0.0
            if P[I_L1_KEYSYNC] > 0.5:
                l1_ph = 0.0
            if P[I_L2_KEYSYNC] > 0.5:
                l2_ph = 0.0
            gate = True
            off_time = ev_off[ei]
            ei += 1
        if gate and n >= off_time:
            gate = False
            if a_stage != 0:
                a_stage = 3
            if m_stage != 0:
                m_stage = 3

        # ---- control rate ------------------------------------------------
        if n % cr == 0:
            # amp envelope
            if a_stage == 1:
                if a_ca == 0.0:
                    a_lvl = 1.0
                else:
                    a_lvl = (1.0 + ovr) + (a_lvl - (1.0 + ovr)) * a_ca
                if a_lvl >= 1.0:
                    a_lvl = 1.0
                    a_stage = 2
            elif a_stage == 2:
                a_lvl = a_sus + (a_lvl - a_sus) * a_cd
            elif a_stage == 3:
                a_lvl = a_lvl * a_crel
                if a_lvl < 1e-6:
                    a_lvl = 0.0
                    a_stage = 0
            # mod envelope
            if m_stage == 1:
                if m_ca == 0.0:
                    m_lvl = 1.0
                else:
                    m_lvl = (1.0 + ovr) + (m_lvl - (1.0 + ovr)) * m_ca
                if m_lvl >= 1.0:
                    m_lvl = 1.0
                    m_stage = 2
            elif m_stage == 2:
                m_lvl = m_sus + (m_lvl - m_sus) * m_cd
            elif m_stage == 3:
                m_lvl = m_lvl * m_crel
                if m_lvl < 1e-6:
                    m_lvl = 0.0
                    m_stage = 0
            damp = (a_lvl - amp) / cr

            # LFOs
            l1 = 0.0
            l2 = 0.0
            if use_l1:
                l1_ph += l1_inc
                if l1_ph >= 1.0:
                    l1_ph -= math.floor(l1_ph)
                    rng ^= (rng << 13) & 0x7FFFFFFF
                    rng ^= rng >> 17
                    rng ^= (rng << 5) & 0x7FFFFFFF
                    l1_sh = rng / 1073741824.0 - 1.0
                l1 = _lfo(l1_wave, l1_ph, l1_sh)
                l1_age += dt_block
                if l1_delay > 0.0 and l1_age < l1_delay:
                    l1 *= l1_age / l1_delay
            if use_l2:
                l2_ph += l2_inc
                if l2_ph >= 1.0:
                    l2_ph -= math.floor(l2_ph)
                    rng ^= (rng << 13) & 0x7FFFFFFF
                    rng ^= rng >> 17
                    rng ^= (rng << 5) & 0x7FFFFFFF
                    l2_sh = rng / 1073741824.0 - 1.0
                l2 = _lfo(l2_wave, l2_ph, l2_sh)
                l2_age += dt_block
                if l2_delay > 0.0 and l2_age < l2_delay:
                    l2 *= l2_age / l2_delay

            # glide
            if glide_c > 0.0:
                pitch = pitch_target + (pitch - pitch_target) * glide_c
            else:
                pitch = pitch_target

            # oscillator increments and pulse widths
            f1 = 440.0 * 2.0 ** ((pitch - 69.0 + P[I_O1_SEMI] + m_lvl * P[I_O1_ENV_SEMI]
                                   + l1 * P[I_O1_LFO_SEMI]) / 12.0)
            inc1 = f1 / sr
            if inc1 > 0.45:
                inc1 = 0.45
            pw1 = P[I_O1_PW] + m_lvl * P[I_O1_ENV_PW] + l2 * P[I_O1_LFO_PW]
            if pw1 < 0.03:
                pw1 = 0.03
            elif pw1 > 0.97:
                pw1 = 0.97
            if use_o2:
                f2 = 440.0 * 2.0 ** ((pitch - 69.0 + P[I_O2_SEMI] + m_lvl * P[I_O2_ENV_SEMI]
                                       + l1 * P[I_O2_LFO_SEMI]) / 12.0)
                inc2 = f2 / sr
                if inc2 > 0.45:
                    inc2 = 0.45
                pw2 = P[I_O2_PW] + m_lvl * P[I_O2_ENV_PW] + l2 * P[I_O2_LFO_PW]
                if pw2 < 0.03:
                    pw2 = 0.03
                elif pw2 > 0.97:
                    pw2 = 0.97

            # filter cutoff
            oct_ = (P[I_F_LOG2HZ] + m_lvl * P[I_F_ENV_OCT] + l2 * P[I_F_LFO_OCT]
                    + P[I_F_TRACK] * (pitch - P[I_F_TRACK_REF]) / 12.0)
            fc_block = 2.0 ** oct_
            if fc_block > fc_max:
                fc_block = fc_max
            elif fc_block < 5.0:
                fc_block = 5.0
            g_target = math.tan(pi_sr * fc_block)
            if n == 0:
                g = g_target
            dg = (g_target - g) / cr

        # ---- oscillators ---------------------------------------------------
        o1 = _osc(o1_wave, ph1, inc1, pw1)
        ph1 += inc1
        wrapped = False
        frac = 0.0
        if ph1 >= 1.0:
            ph1 -= 1.0
            wrapped = True
            frac = ph1 / inc1 if inc1 > 0.0 else 0.0
        mix = g1 * o1
        o2 = 0.0
        if use_o2:
            o2 = _osc(o2_wave, ph2, inc2, pw2)
            ph2 += inc2
            if ph2 >= 1.0:
                ph2 -= 1.0
            if sync and wrapped:
                ph2 = frac * inc2
            mix += g2 * o2
            if gr > 0.0:
                mix += gr * o1 * o2
        if gs > 0.0:
            incs = inc1 * sub_div
            if sub_wave == 0:
                sub = math.sin(6.283185307179586 * phs)
            elif sub_wave == 1:
                t2 = phs + sub_pw
                if t2 >= 1.0:
                    t2 -= 1.0
                sub = (2.0 * phs - _blep(phs, incs)) - (2.0 * t2 - _blep(t2, incs))
            else:
                t2 = phs + 0.5
                if t2 >= 1.0:
                    t2 -= 1.0
                sub = (2.0 * phs - _blep(phs, incs)) - (2.0 * t2 - _blep(t2, incs))
            phs += incs
            if phs >= 1.0:
                phs -= 1.0
            mix += gs * sub
        if gn > 0.0:
            rng ^= (rng << 13) & 0x7FFFFFFF
            rng ^= rng >> 17
            rng ^= (rng << 5) & 0x7FFFFFFF
            mix += gn * (rng / 1073741824.0 - 1.0)
        if osc_lp > 0.0:
            lp_z += lp_a * (mix - lp_z)
            mix = lp_z

        # ---- drive ---------------------------------------------------------
        x = _sat(mix * drive)

        # ---- filter --------------------------------------------------------
        g += dg
        gg = g
        if f_fm != 0.0:
            fc = fc_block * 2.0 ** (f_fm * o2)
            if fc > fc_max:
                fc = fc_max
            elif fc < 5.0:
                fc = 5.0
            gg = math.tan(pi_sr * fc)
        if f_type == 0:
            # state variable filter (trapezoidal), one or two 2-pole stages
            a1 = 1.0 / (1.0 + gg * (gg + f_k))
            a2 = gg * a1
            a3 = gg * a2
            v3 = x - s2
            v1 = a1 * s1 + a2 * v3
            v2 = s2 + a2 * s1 + a3 * v3
            s1 = 2.0 * v1 - s1
            s2 = 2.0 * v2 - s2
            if f_sat > 0.0:
                s1 = _sat(s1 * f_sat) / f_sat
                s2 = _sat(s2 * f_sat) / f_sat
            if f_shape == 0:
                y = v2
            elif f_shape == 1:
                y = v1 * f_k
            else:
                y = x - f_k * v1 - v2
            if f_slope == 1:
                a1 = 1.0 / (1.0 + gg * (gg + f_k2))
                a2 = gg * a1
                a3 = gg * a2
                v3 = y - s4
                v1 = a1 * s3 + a2 * v3
                v2 = s4 + a2 * s3 + a3 * v3
                s3 = 2.0 * v1 - s3
                s4 = 2.0 * v2 - s4
                if f_sat > 0.0:
                    s3 = _sat(s3 * f_sat) / f_sat
                    s4 = _sat(s4 * f_sat) / f_sat
                if f_shape == 0:
                    y = v2
                elif f_shape == 1:
                    y = v1 * f_k2
                else:
                    y = y - f_k2 * v1 - v2
        else:
            # 4-pole ladder (zero delay feedback), resonance = feedback amount
            gd = gg / (1.0 + gg)
            sg = (gd * gd * gd * s1 + gd * gd * s2 + gd * s3 + s4) / (1.0 + gg)
            u = (x * (1.0 + f_comp * f_k) - f_k * sg) / (1.0 + f_k * gd * gd * gd * gd)
            if f_sat > 0.0:
                u = _sat(u * f_sat) / f_sat
            v = (u - s1) * gd
            y1 = v + s1
            s1 = y1 + v
            v = (y1 - s2) * gd
            y2 = v + s2
            s2 = y2 + v
            v = (y2 - s3) * gd
            y3 = v + s3
            s3 = y3 + v
            v = (y3 - s4) * gd
            y = v + s4
            s4 = y + v

        # ---- distortion, VCA, output ----------------------------------------
        # distortion stage: nearly linear at zero, saturating more and more
        y = _sat(y * dist_in) * dist_out
        amp += damp
        y = y * amp
        hp_y = y - hp_x + hp_r * hp_y
        hp_x = y
        if -1e-12 < hp_y < 1e-12:
            hp_y = 0.0  # keep denormals out of the analysis
        out[n] = hp_y * out_gain


def filter_mag_db(freqs: np.ndarray, sr: float, ftype: int, slope: int, shape: int, fc: float, k: float,
                  k2: float, comp: float) -> np.ndarray:
    """Magnitude response (dB) of the model filter - the same maths as the kernel.

    ftype 0 = state variable filter (slope 0/1 = 12/24 dB, shape 0/1/2 = LP/BP/HP,
    k = damping of the resonant stage, k2 = damping of the second stage),
    ftype 1 = four pole ladder (k = feedback amount, comp = passband compensation).
    """
    fc = min(max(float(fc), 5.0), 0.45 * sr)
    x = np.tan(np.pi * np.minimum(freqs, 0.499 * sr) / sr) / math.tan(math.pi * fc / sr)
    jx = 1j * x
    if int(ftype) == 0:
        def stage(kk):
            den = 1.0 - x * x + kk * jx
            if shape == 0:
                return 1.0 / den
            if shape == 1:
                return kk * jx / den
            return -(x * x) / den

        h = stage(k)
        if int(slope) == 1:
            h = h * stage(k2)
    else:
        g4 = 1.0 / (1.0 + jx) ** 4
        h = (1.0 + comp * k) * g4 / (1.0 + k * g4)
    return 20.0 * np.log10(np.abs(h) + 1e-9)


def filter_response_db(freqs: np.ndarray, sr: float, P: np.ndarray, note: float) -> np.ndarray:
    """Magnitude response (dB) of the filter described by an engine vector, at a held note."""
    octave = P[I_F_LOG2HZ] + P[I_F_TRACK] * (note - P[I_F_TRACK_REF]) / 12.0
    return filter_mag_db(freqs, sr, int(P[I_F_TYPE]), int(P[I_F_SLOPE]), int(P[I_F_SHAPE]), 2.0 ** octave,
                         P[I_F_K], P[I_F_K2], P[I_F_COMP])


# ---------------------------------------------------------------------------
# physics: mapping raw MIDI values to physical quantities
# ---------------------------------------------------------------------------


def _exp_table(n: int, lo: float, hi: float) -> np.ndarray:
    """n values rising exponentially from lo to hi."""
    return lo * (hi / lo) ** (np.arange(n) / (n - 1.0))


def _default_cutoff_table() -> np.ndarray:
    # log2(Hz) for the raw values 0..255: about ten octaves, 16 Hz .. 22 kHz
    return np.log2(_exp_table(256, 16.0, 22000.0))


def _default_damping_table() -> np.ndarray:
    # classic filter: damping k = 1/Q, from a gentle Q of 0.6 to near self oscillation
    q = _exp_table(128, 0.6, 40.0)
    return 1.0 / q


def _default_ladder_table() -> np.ndarray:
    # acid filter: feedback amount 0 .. 3.6 (4 = self oscillation)
    return 3.6 * (np.arange(128) / 127.0) ** 1.2


def _default_attack_table() -> np.ndarray:
    t = _exp_table(128, 0.0008, 6.0)
    t[0] = 0.0005
    return t


def _default_decay_table() -> np.ndarray:
    return _exp_table(128, 0.0025, 4.0)


def _default_lfo_table() -> np.ndarray:
    return _exp_table(256, 0.05, 150.0)


def _default_glide_table() -> np.ndarray:
    t = _exp_table(128, 0.002, 2.0)
    t[0] = 0.0
    return t


def _default_gain_table(n: int, gmax: float) -> np.ndarray:
    return 1.0 + (gmax - 1.0) * (np.arange(n) / (n - 1.0)) ** 2


def _default_mix_table() -> np.ndarray:
    return (np.arange(256) / 255.0) ** 1.5


@dataclass
class Physics:
    """Raw value -> physical quantity tables of one particular instrument."""

    # tuning
    tune_cents: float = 0.0
    octave_offset: float = 0.0
    # oscillators (amplitude of each waveform relative to the sawtooth)
    wave_gain: Tuple[float, float, float, float] = (0.9, 0.9, 1.0, 0.9)
    sub_gain: Tuple[float, float, float] = (0.9, 0.9, 0.9)
    sub_pulse_width: float = 0.25
    noise_gain: float = 0.7
    ring_gain: float = 0.9
    mix_table: np.ndarray = field(default_factory=_default_mix_table)
    headroom: float = 0.3
    osc_lp_hz: float = 0.0
    # filter
    cutoff_classic: np.ndarray = field(default_factory=_default_cutoff_table)
    cutoff_acid: np.ndarray = field(default_factory=_default_cutoff_table)
    slope12_shift_oct: float = 0.0
    damping_classic: np.ndarray = field(default_factory=_default_damping_table)
    damping_stage2: float = 1.3  # damping (1/Q) of the second, non-resonant 2-pole stage of the 24 dB mode
    feedback_acid: np.ndarray = field(default_factory=_default_ladder_table)
    acid_comp: float = 0.3
    filter_sat: float = 0.0
    env_oct_per_unit: float = 9.0 / 63.0
    lfo2_oct_per_unit: float = 5.0 / 127.0
    track_ref_note: float = 60.0
    track_full: float = 1.0  # octaves of cutoff per octave of pitch at "full"
    fm_oct_per_unit: float = 3.0 / 127.0
    drive_table: np.ndarray = field(default_factory=lambda: _default_gain_table(128, 24.0))
    # envelopes
    attack_table: np.ndarray = field(default_factory=_default_attack_table)
    decay_table: np.ndarray = field(default_factory=_default_decay_table)
    release_table: np.ndarray = field(default_factory=_default_decay_table)
    attack_overshoot: float = 0.35
    sustain_gamma: float = 1.0
    # effects and modulation
    dist_table: np.ndarray = field(default_factory=lambda: _default_gain_table(128, 30.0))
    pitch_env_semis_per_unit: float = 24.0 / 63.0
    lfo1_semis_per_unit: float = 12.0 / 127.0
    env_pw_per_unit: float = 0.45 / 63.0
    lfo2_pw_per_unit: float = 0.45 / 63.0
    lfo_rate_table: np.ndarray = field(default_factory=_default_lfo_table)
    lfo_delay_table: np.ndarray = field(default_factory=lambda: np.concatenate([[0.0], _exp_table(127, 0.01, 4.0)]))
    glide_table: np.ndarray = field(default_factory=_default_glide_table)
    # output stage
    out_hp_hz: float = 12.0
    out_gain: float = 1.0
    # static colouring of the instrument's output relative to the model: gain in dB
    # at the frequencies of eq_freqs (applied to the model's spectra, not its audio)
    eq_freqs: np.ndarray = field(default_factory=lambda: 20.0 * 2.0 ** (np.arange(30) / 3.0))
    eq_db: np.ndarray = field(default_factory=lambda: np.zeros(30))
    # bookkeeping
    calibrated: Dict[str, bool] = field(default_factory=dict)

    _ARRAYS = ("mix_table", "cutoff_classic", "cutoff_acid", "damping_classic", "feedback_acid", "drive_table",
               "attack_table", "decay_table", "release_table", "dist_table", "lfo_rate_table", "lfo_delay_table",
               "glide_table")

    def to_dict(self) -> dict:
        out = {}
        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
                out[key] = [float(x) for x in value]
            elif isinstance(value, tuple):
                out[key] = [float(x) for x in value]
            else:
                out[key] = value
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "Physics":
        phys = cls()
        for key, value in data.items():
            if not hasattr(phys, key):
                continue
            current = getattr(phys, key)
            if isinstance(current, np.ndarray):
                arr = np.asarray(value, dtype=np.float64)
                if arr.shape == current.shape and np.all(np.isfinite(arr)):
                    setattr(phys, key, arr)
            elif isinstance(current, tuple):
                if len(value) == len(current):
                    setattr(phys, key, tuple(float(x) for x in value))
            elif isinstance(current, dict):
                setattr(phys, key, dict(value))
            else:
                setattr(phys, key, type(current)(value))
        return phys

    def copy(self) -> "Physics":
        return Physics.from_dict(self.to_dict())


_RANGE_OCT = {63: -1.0, 64: 0.0, 65: 1.0, 66: 2.0}


def resolve(patch: Dict[str, int], physics: Physics, sr: float, seed: int = 1,
            phase1: float = 0.13, phase2: float = 0.61) -> np.ndarray:
    """Translate a raw patch into the engine parameter vector."""
    P = np.zeros(N_ENGINE, dtype=np.float64)
    g = patch.get
    ph = physics
    P[I_SR] = sr
    tune = ph.tune_cents / 100.0 + 12.0 * ph.octave_offset

    def osc(prefix: str, base: int):
        wave = int(g(f"{prefix}_wave", 2))
        P[base + 0] = wave
        P[base + 1] = (12.0 * _RANGE_OCT.get(int(g(f"{prefix}_range", 64)), 0.0)
                       + coarse_semitones(g(f"{prefix}_coarse", 128))
                       + fine_cents(int(g(f"{prefix}_fine", 128))) / 100.0 + tune)
        P[base + 2] = pulse_width_percent(g(f"{prefix}_pw", 64)) / 100.0
        P[base + 3] = bipolar7(int(g(f"{prefix}_env_pitch", 64))) * ph.pitch_env_semis_per_unit
        P[base + 4] = bipolar8(int(g(f"{prefix}_lfo1_pitch", 128))) * ph.lfo1_semis_per_unit
        if wave == 3:
            P[base + 5] = bipolar7(int(g(f"{prefix}_env_pw", 64))) * ph.env_pw_per_unit
            P[base + 6] = bipolar7(int(g(f"{prefix}_lfo2_pw", 64))) * ph.lfo2_pw_per_unit
        return wave

    w1 = osc("osc1", I_O1_WAVE)
    w2 = osc("osc2", I_O2_WAVE)
    mix = ph.mix_table
    P[I_O1_GAIN] = mix[int(g("mix_osc1", 255))] * ph.wave_gain[w1]
    P[I_O2_GAIN] = mix[int(g("mix_osc2", 0))] * ph.wave_gain[w2]
    P[I_O1_PHASE] = phase1
    P[I_O2_PHASE] = phase2
    P[I_SYNC] = float(g("osc_sync", 0))
    sw = int(g("sub_wave", 0))
    P[I_SUB_WAVE] = sw
    P[I_SUB_DIV] = 0.5 if int(g("sub_oct", 63)) == 63 else 0.25
    P[I_SUB_GAIN] = mix[int(g("mix_sub", 0))] * ph.sub_gain[sw]
    P[I_SUB_PW] = ph.sub_pulse_width
    P[I_NOISE_GAIN] = mix[int(g("mix_noise", 0))] * ph.noise_gain
    P[I_RING_GAIN] = mix[int(g("mix_ring", 0))] * ph.ring_gain
    P[I_DRIVE] = ph.drive_table[int(g("filter_drive", 0))]
    P[I_HEADROOM] = ph.headroom
    P[I_OSC_LP] = ph.osc_lp_hz

    ftype = int(g("filter_type", 0))
    slope = int(g("filter_slope", 1))
    res = int(g("filter_res", 0))
    freq = int(g("filter_freq", 255))
    P[I_F_TYPE] = ftype
    P[I_F_SLOPE] = slope
    P[I_F_SHAPE] = int(g("filter_shape", 0))
    if ftype == 0:
        P[I_F_LOG2HZ] = ph.cutoff_classic[freq] + (ph.slope12_shift_oct if slope == 0 else 0.0)
        P[I_F_K] = ph.damping_classic[res]
        P[I_F_K2] = ph.damping_stage2
    else:
        P[I_F_LOG2HZ] = ph.cutoff_acid[freq]
        P[I_F_K] = ph.feedback_acid[res]
        P[I_F_K2] = 0.0
    P[I_F_COMP] = ph.acid_comp
    P[I_F_SAT] = ph.filter_sat
    P[I_F_ENV_OCT] = bipolar7(int(g("filter_env", 64))) * ph.env_oct_per_unit
    P[I_F_LFO_OCT] = bipolar8(int(g("filter_lfo2", 128))) * ph.lfo2_oct_per_unit
    P[I_F_TRACK] = ph.track_full * (7 - int(g("filter_track", 0))) / 7.0
    P[I_F_TRACK_REF] = ph.track_ref_note
    P[I_F_FM_OCT] = g("osc_filter_mod", 0) * ph.fm_oct_per_unit

    P[I_A_ATT] = ph.attack_table[int(g("amp_attack", 0))]
    P[I_A_DEC] = ph.decay_table[int(g("amp_decay", 0))]
    P[I_A_SUS] = (g("amp_sustain", 127) / 127.0) ** ph.sustain_gamma
    P[I_A_REL] = ph.release_table[int(g("amp_release", 0))]
    P[I_A_TRIG] = int(g("amp_trig", 0))
    P[I_M_ATT] = ph.attack_table[int(g("mod_attack", 0))]
    P[I_M_DEC] = ph.decay_table[int(g("mod_decay", 0))]
    P[I_M_SUS] = (g("mod_sustain", 127) / 127.0) ** ph.sustain_gamma
    P[I_M_REL] = ph.release_table[int(g("mod_release", 0))]
    P[I_M_TRIG] = int(g("mod_trig", 0))
    P[I_ENV_OVR] = max(0.02, ph.attack_overshoot)
    P[I_DIST] = ph.dist_table[int(g("distortion", 0))]

    P[I_L1_RATE] = ph.lfo_rate_table[int(g("lfo1_speed", 75))]
    P[I_L1_WAVE] = int(g("lfo1_wave", 0))
    P[I_L1_DELAY] = ph.lfo_delay_table[int(g("lfo1_delay", 0))]
    P[I_L1_KEYSYNC] = int(g("lfo1_keysync", 1))
    P[I_L2_RATE] = ph.lfo_rate_table[int(g("lfo2_speed", 52))]
    P[I_L2_WAVE] = int(g("lfo2_wave", 0))
    P[I_L2_DELAY] = ph.lfo_delay_table[int(g("lfo2_delay", 0))]
    P[I_L2_KEYSYNC] = int(g("lfo2_keysync", 1))
    P[I_GLIDE] = ph.glide_table[int(g("portamento", 0))]
    P[I_OUT_HP] = ph.out_hp_hz
    P[I_OUT_GAIN] = ph.out_gain
    P[I_SEED] = seed
    return P


# ---------------------------------------------------------------------------
# note sequences and rendering
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Note:
    """One note of a probe sequence (times in seconds from the start)."""

    start: float
    end: float
    note: int
    velocity: int = 100


@dataclass
class Phrase:
    """A monophonic phrase played to compare a patch with the reference."""

    notes: List[Note]
    duration: float  # total length that is rendered / recorded

    def to_dict(self) -> dict:
        return {"duration": self.duration,
                "notes": [[n.start, n.end, n.note, n.velocity] for n in self.notes]}

    @classmethod
    def from_dict(cls, data: dict) -> "Phrase":
        return cls([Note(float(a), float(b), int(c), int(d)) for a, b, c, d in data["notes"]],
                   float(data["duration"]))

    def arrays(self, sr: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        on = np.array([int(round(n.start * sr)) for n in self.notes], dtype=np.int64)
        off = np.array([int(round(n.end * sr)) for n in self.notes], dtype=np.int64)
        note = np.array([float(n.note) for n in self.notes], dtype=np.float64)
        return on, off, note


def render(patch: Dict[str, int], seq: Phrase, physics: Physics, sr: int = 32000, seed: int = 1,
           phase1: float = 0.13, phase2: float = 0.61, pad: float = 0.0) -> np.ndarray:
    """Render a sequence with the model.  Returns float32 mono audio."""
    P = resolve(patch, physics, float(sr), seed=seed, phase1=phase1, phase2=phase2)
    return render_engine(P, seq, sr, pad=pad)


def render_engine(P: np.ndarray, seq: Phrase, sr: int, pad: float = 0.0) -> np.ndarray:
    n = int(round((seq.duration + pad) * sr))
    out = np.zeros(n, dtype=np.float32)
    on, off, note = seq.arrays(float(sr))
    _render_kernel(out, on, off, note, P)
    return out


def warmup() -> None:
    """Compile the kernel (first call takes a few seconds, later calls are cached)."""
    seq = Phrase([Note(0.0, 0.01, 40)], 0.02)
    render(normalize_patch({}), seq, Physics(), 32000)
