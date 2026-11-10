"""Matching the steady timbre of held notes.

Which oscillators are on, their waveforms and octaves, the filter type and its
position: all of that shows in the spectrum of a held note, independently of
the envelopes.  This module looks at that "snapshot" spectrum only.

In the steady state the synth is (almost) linear: the output spectrum is the
spectrum of the sources multiplied by the response of the filter.  So the
spectra of all source combinations are rendered once (a *dictionary*), the
filter responses are computed analytically, and every combination of switches
and filter settings can be scored in microseconds - an exhaustive search
instead of a guess.  The best structures are then refined with real renders.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numba import njit

from . import engine
from .engine import Physics, filter_response_db
from .features import ANALYSIS_SR, Target, tonal_bank
from .model import _PHASES, two_sources
from .params import baseline_patch

STATIC_OVERRIDES: Dict[str, int] = {
    "amp_attack": 0, "amp_decay": 0, "amp_sustain": 127, "amp_release": 0,
    "filter_env": 64, "filter_lfo2": 128,
    "osc1_env_pitch": 64, "osc2_env_pitch": 64, "osc1_lfo1_pitch": 128, "osc2_lfo1_pitch": 128,
    "osc1_env_pw": 64, "osc2_env_pw": 64, "osc1_lfo2_pw": 64, "osc2_lfo2_pw": 64,
    "portamento": 0,
}

FILTER_MODES: Dict[str, Dict[str, int]] = {
    "lp24": {"filter_type": 0, "filter_slope": 1, "filter_shape": 0},
    "lp12": {"filter_type": 0, "filter_slope": 0, "filter_shape": 0},
    "acid": {"filter_type": 1, "filter_slope": 1, "filter_shape": 0},
    "bp12": {"filter_type": 0, "filter_slope": 0, "filter_shape": 1},
    "bp24": {"filter_type": 0, "filter_slope": 1, "filter_shape": 1},
    "hp12": {"filter_type": 0, "filter_slope": 0, "filter_shape": 2},
    "hp24": {"filter_type": 0, "filter_slope": 1, "filter_shape": 2},
}

_NOTE_LEN = 0.40
_SKIP = 0.06
_RANGE_DB = 50.0
_DETUNE = 140  # osc 2 fine used while judging the static timbre (+12 cents: the phases keep moving)
_PROFILE_WEIGHTS = {"late": 0.7, "early": 0.3}


@dataclass
class StaticProfile:
    note: float
    kind: str  # "late" (settled part of long notes) or "early" (first 130 ms)
    ref_db: np.ndarray  # clipped at the floor
    floor: float
    frames: int


def build_profiles(target: Target) -> List[StaticProfile]:
    """Spectra of the most informative pitch at two moments of the note.

    * late  - the settled part of the longest notes: what the patch sounds
              like once the envelopes have done their work,
    * early - the first 130 ms: the filter is still open, which exposes the
              waveforms.

    Comparing both resolves ambiguities (a triangle through an open filter
    resembles a square through a closed one in a single snapshot) and tells
    how far the filter moves.
    """
    cfg = target.cfg
    power = target.features.a
    n_frames = power.shape[0]
    frame_s = cfg.hop_a / cfg.sr
    half_win = 0.5 * cfg.n_fft_a / cfg.sr
    if not target.notes:
        return []
    by_pitch: Dict[int, List[Tuple[float, float]]] = {}
    for start, end, note in target.notes:
        by_pitch.setdefault(int(round(note)), []).append((start, end))

    def score(spans):
        long_ = sum(e - s for s, e in spans if e - s >= 0.25)
        return (long_, sum(e - s for s, e in spans))

    pitch = max(by_pitch, key=lambda k: score(by_pitch[k]))
    spans = by_pitch[pitch]
    longest = max(e - s for s, e in spans)

    def frame(t: float) -> int:
        return int(min(max(round(t / frame_s), 0), n_frames - 1))

    late_idx: List[int] = []
    early_idx: List[int] = []
    for start, end in spans:
        dur = end - start
        if dur >= 0.6 * longest:
            lo = start + max(0.15, 0.5 * dur)
            hi = end - half_win + 0.02
            if hi >= lo:
                late_idx.extend(range(int(math.ceil(lo / frame_s)), min(int(math.floor(hi / frame_s)), n_frames - 1) + 1))
            else:
                late_idx.append(frame(end - half_win))
        if dur >= 0.1:
            early_idx.append(frame(start + half_win))
    profiles = []
    for kind, idx in (("late", late_idx), ("early", early_idx)):
        if not idx:
            continue
        mean = power[np.asarray(sorted(set(idx)))].astype(np.float64).mean(axis=0)
        ref_db = 10.0 * np.log10(mean + 1e-20)
        floor = float(ref_db.max()) - _RANGE_DB
        profiles.append(StaticProfile(float(pitch), kind, np.maximum(ref_db, floor), floor, len(idx)))
    return profiles


@njit(cache=True, nogil=True)
def _score(ref, floor, strong, w, src, filt, out):
    """Distance of every (source, filter) pair to one reference profile.

    ref (B,), src (S, B) and filt (F, B) are in dB; out is (S, F).
    """
    n_b = ref.shape[0]
    n_s = src.shape[0]
    n_f = filt.shape[0]
    cnt = 0
    for b in range(n_b):
        if strong[b]:
            cnt += 1
    for i in range(n_s):
        for q in range(n_f):
            acc = 0.0
            for b in range(n_b):
                if strong[b]:
                    acc += ref[b] - (src[i, b] + filt[q, b])
            off = acc / cnt if cnt > 0 else 0.0
            d = 0.0
            for b in range(n_b):
                v = src[i, b] + filt[q, b] + off
                if v < floor:
                    v = floor
                d += w[b] * abs(ref[b] - v)
            out[i, q] = d


class StaticMatcher:
    """Distance between the steady spectrum of a patch and that of the reference."""

    def __init__(self, physics: Physics, target: Target, sr: int = ANALYSIS_SR,
                 pool_map: Optional[Callable] = None):
        self.physics = physics
        self.target = target
        self.sr = sr
        self.bank = tonal_bank(target.cfg)
        self.freqs = np.asarray(self.bank.centers, dtype=np.float64)
        w = target.w_band_a.astype(np.float64)
        self.w_band = w / w.sum() / 10.0  # weighted mean, in units of 10 dB
        self.profiles = build_profiles(target)
        self.note = self.profiles[0].note if self.profiles else 40.0
        self.eq = 10.0 ** (np.interp(np.log2(self.freqs), np.log2(physics.eq_freqs), physics.eq_db) / 10.0)
        self.n = int(round((_NOTE_LEN + 0.02) * sr))
        self.skip = int(round(_SKIP * sr))
        self._on = np.array([0], dtype=np.int64)
        self._off = np.array([int(round(_NOTE_LEN * sr))], dtype=np.int64)
        self.map = pool_map or (lambda fn, items: [fn(it) for it in items])

    # -- real renders ---------------------------------------------------------
    @staticmethod
    def static_patch(patch: Dict[str, int]) -> Dict[str, int]:
        out = dict(patch)
        out.update(STATIC_OVERRIDES)
        return out

    def _power(self, patch: Dict[str, int], note: float) -> np.ndarray:
        k = 2 if two_sources(patch) else 1
        acc = None
        for r in range(k):
            ph1, ph2 = _PHASES[r]
            P = engine.resolve(patch, self.physics, float(self.sr), seed=1 + r, phase1=ph1, phase2=ph2)
            out = np.zeros(self.n, dtype=np.float32)
            engine._render_kernel(out, self._on, self._off, np.array([note], dtype=np.float64), P)
            power = self.bank.mean_power(out[self.skip:]).astype(np.float64)
            acc = power if acc is None else acc + power
        return acc / k * self.eq

    def _distance(self, cand_db: np.ndarray, prof: StaticProfile) -> float:
        strong = prof.ref_db > prof.floor + 8.0
        if not np.any(strong):
            return 0.0
        offset = float(np.mean(prof.ref_db[strong] - cand_db[strong]))
        c = np.maximum(cand_db + offset, prof.floor)
        return float(np.sum(self.w_band * np.abs(prof.ref_db - c)))

    def loss(self, patch: Dict[str, int], kind: str = "late") -> float:
        """Distance of a real static render to one profile (diagnostics / tests)."""
        patch = self.static_patch(patch)
        for prof in self.profiles:
            if prof.kind == kind:
                cand = 10.0 * np.log10(self._power(patch, prof.note) + 1e-20)
                return self._distance(cand, prof)
        return 0.0

    # -- dictionary search ------------------------------------------------------
    def _build_atoms(self, base: Dict[str, int]) -> None:
        """Unfiltered spectra of every single source at full level."""
        note = self.note
        variants = []
        for wave in range(4):
            for rng_ in (63, 64, 65, 66):
                for pw in ((64, 44, 24, 8) if wave == 3 else (64,)):
                    variants.append((wave, rng_, pw))
        subs = [(rng_, sw, so) for rng_ in (63, 64, 65, 66) for sw in range(3) for so in (62, 63)]
        osc = self.map(lambda v: self._power(dict(base, osc1_wave=v[0], osc1_range=v[1], osc1_pw=v[2]), note),
                       variants)
        sub = self.map(lambda v: self._power(dict(base, mix_osc1=0, mix_sub=255, osc1_range=v[0], sub_wave=v[1],
                                                  sub_oct=v[2]), note), subs)
        self.osc_atoms = {v: p for v, p in zip(variants, osc)}
        self.sub_atoms = {v: p for v, p in zip(subs, sub)}
        self.osc_variants = variants

    def _source_power(self, patch: Dict[str, int]) -> np.ndarray:
        """Source spectrum of a patch as the power sum of its (independent) sources."""
        mix = self.physics.mix_table
        full = mix[255]
        power = (mix[patch["mix_osc1"]] / full) ** 2 * self.osc_atoms[
            (patch["osc1_wave"], patch["osc1_range"], patch["osc1_pw"] if patch["osc1_wave"] == 3 else 64)]
        if patch["mix_sub"] > 0:
            power = power + (mix[patch["mix_sub"]] / full) ** 2 * self.sub_atoms[
                (patch["osc1_range"], patch["sub_wave"], patch["sub_oct"])]
        if patch["mix_osc2"] > 0:
            power = power + (mix[patch["mix_osc2"]] / full) ** 2 * self.osc_atoms[
                (patch["osc2_wave"], patch["osc2_range"], patch["osc2_pw"] if patch["osc2_wave"] == 3 else 64)]
        return power

    def _filter_db(self, patch: Dict[str, int]) -> np.ndarray:
        cand = dict(patch)
        cand.update(STATIC_OVERRIDES)
        return filter_response_db(self.freqs, self.sr, engine.resolve(cand, self.physics, float(self.sr)), self.note)

    def _refine(self, patch: Dict[str, int], info: dict, open_db: np.ndarray) -> Tuple[float, Dict[str, int], dict]:
        """Fine search of the levels, the resonance and the two cutoffs around a grid result."""
        patch = dict(patch)
        info = dict(info)

        def evaluate(pt: Dict[str, int], inf: dict, cut_span: int, ress: Sequence[int]):
            src = np.ascontiguousarray((10.0 * np.log10(self._source_power(pt) + 1e-20) - open_db)[None, :])
            best = None
            for res in ress:
                total = 0.0
                cuts = {}
                for prof in self.profiles:
                    centre = inf[prof.kind]
                    cands = list(range(max(0, centre - cut_span), min(255, centre + cut_span) + 1))
                    filt = np.array([self._filter_db(dict(pt, filter_freq=c, filter_res=res)) for c in cands])
                    out = np.empty((1, len(cands)), dtype=np.float64)
                    _score(prof.ref_db, prof.floor, prof.ref_db > prof.floor + 8.0, self.w_band, src, filt, out)
                    j = int(np.argmin(out[0]))
                    cuts[prof.kind] = cands[j]
                    total += _PROFILE_WEIGHTS[prof.kind] * float(out[0, j])
                if len(self.profiles) == 1:
                    total /= _PROFILE_WEIGHTS[self.profiles[0].kind]
                if best is None or total < best[0]:
                    best = (total, res, cuts)
            return best

        def res_grid(centre: int) -> List[int]:
            return sorted({int(min(max(centre + d, 0), 118)) for d in (-10, -5, 0, 5, 10)})

        total, res, cuts = evaluate(patch, info, 6, res_grid(patch["filter_res"]))
        patch["filter_res"] = res
        info.update(cuts)
        # levels of the sources, one at a time (filter held, then re-fitted)
        for key in ("mix_sub", "mix_osc2", "mix_osc1"):
            if patch[key] == 0:
                continue
            others_on = any(patch[k] > 0 for k in ("mix_osc1", "mix_osc2", "mix_sub") if k != key)
            if not others_on:
                continue
            for level in (70, 100, 130, 160, 190, 220, 255):
                if level == patch[key]:
                    continue
                trial = dict(patch, **{key: level})
                t, r, c = evaluate(trial, info, 2, [patch["filter_res"]])
                if t < total - 1e-6:
                    total, patch, cuts = t, trial, c
                    info.update(c)
        total, res, cuts = evaluate(patch, info, 4, res_grid(patch["filter_res"]))
        patch["filter_res"] = res
        info.update(cuts)
        held = info.get("late", info.get("early"))
        patch["filter_freq"] = held
        info = {"late": held, "early": info.get("early", held)}
        penalty = (0.010 if patch["mix_osc2"] > 0 else 0.0) + (0.005 if patch["mix_sub"] > 0 else 0.0)
        return total + penalty, patch, info

    def dictionary_search(self, keep: int = 10, top_composites: int = 40,
                          progress: Optional[Callable[[int], None]] = None,
                          fixed: Optional[Dict[str, int]] = None) -> List[Tuple[float, Dict[str, int], dict]]:
        """Exhaustive search over the switches.

        Returns (loss, patch, info) triples, best first, one per structure.
        ``info`` has the cutoff (raw value) that fits the settled part of the
        notes ("late") and the one that fits their beginning ("early").
        """
        ph = self.physics
        fixed = fixed or {}
        note = self.note
        base = self.static_patch(baseline_patch())
        base.update(FILTER_MODES["lp12"])
        base.update(filter_freq=255, filter_res=0, mix_osc1=255, mix_osc2=0, mix_sub=0, osc2_fine=_DETUNE)
        self._build_atoms(base)
        if progress:
            progress(1)
        open_db = filter_response_db(self.freqs, self.sr, engine.resolve(base, ph, float(self.sr)), note)

        # --- oscillator 1 (+ sub) combinations
        comp_meta: List[Dict[str, int]] = []
        for wave, rng_, pw in self.osc_variants:
            if "osc1_wave" in fixed and wave != fixed["osc1_wave"]:
                continue
            if "osc1_range" in fixed and rng_ != fixed["osc1_range"]:
                continue
            o1 = {"osc1_wave": wave, "osc1_range": rng_, "osc1_pw": pw}
            comp_meta.append(dict(o1, mix_osc1=255, mix_sub=0))
            if fixed.get("mix_sub", 1) == 0:
                continue
            for sw in range(3):
                for so in (62, 63):
                    for l1, ls in ((255, 90), (255, 160), (255, 255), (160, 255), (90, 255)):
                        comp_meta.append(dict(o1, mix_osc1=l1, mix_sub=ls, sub_wave=sw, sub_oct=so))
        proto = dict(base)
        comp_power = np.array([self._source_power(dict(proto, **m)) for m in comp_meta])  # (C, B)

        # --- analytic filter responses on a (mode, resonance, cutoff) grid
        modes = [m for m in FILTER_MODES
                 if not any(k in fixed and fixed[k] != v for k, v in FILTER_MODES[m].items())]
        cutoffs = list(range(16, 256, 4))
        ress = [0, 15, 30, 45, 60, 75, 90, 105, 118]
        filt = np.empty((len(modes), len(ress), len(cutoffs), len(self.freqs)), dtype=np.float64)
        for mi, mode in enumerate(modes):
            for ri, res in enumerate(ress):
                for ci, cutoff in enumerate(cutoffs):
                    cand = dict(base, **FILTER_MODES[mode])
                    cand.update(filter_freq=cutoff, filter_res=res)
                    filt[mi, ri, ci] = self._filter_db(cand)
        filt_flat = np.ascontiguousarray(filt.reshape(-1, len(self.freqs)))
        n_m, n_r, n_c = len(modes), len(ress), len(cutoffs)
        w = self.w_band
        weights = dict(_PROFILE_WEIGHTS)

        def score(src_power: np.ndarray):
            """src_power (S, B) -> best loss per (source, mode, resonance) and the two cutoffs."""
            src_db = np.ascontiguousarray(10.0 * np.log10(src_power + 1e-20) - open_db[None, :])
            n_s = src_db.shape[0]
            total = np.zeros((n_s, n_m, n_r))
            cut = {}
            chunks = [(a_, min(a_ + 48, n_s)) for a_ in range(0, n_s, 48)]
            for prof in self.profiles:
                out = np.empty((n_s, filt_flat.shape[0]), dtype=np.float64)
                strong = prof.ref_db > prof.floor + 8.0

                def work(ab, prof=prof, out=out, strong=strong):
                    _score(prof.ref_db, prof.floor, strong, w, src_db[ab[0]:ab[1]], filt_flat, out[ab[0]:ab[1]])
                    return 0

                self.map(work, chunks)
                cube = out.reshape(n_s, n_m, n_r, n_c)
                total += weights[prof.kind] * cube.min(axis=3)
                cut[prof.kind] = cube.argmin(axis=3)
            if len(self.profiles) == 1:
                total /= weights[self.profiles[0].kind]
            return total, cut

        # --- stage 1: oscillator 1 (+ sub) with every filter
        total1, _ = score(comp_power)
        order = np.argsort(total1.reshape(len(comp_meta), -1).min(axis=1))[:top_composites]
        if progress:
            progress(2)

        # --- stage 2: add oscillator 2 (three levels) to the best combinations
        mix = ph.mix_table
        levels = (90, 160, 255)
        if fixed.get("mix_osc2", 1) == 0:
            levels = ()
        results: List[Tuple[float, Dict[str, int], dict]] = []
        for ci in order:
            srcs = [comp_power[ci]]
            metas: List[Dict[str, int]] = [{"mix_osc2": 0}]
            for lv in levels:
                gain = (mix[lv] / mix[255]) ** 2
                for (wave, rng_, pw) in self.osc_variants:
                    srcs.append(comp_power[ci] + gain * self.osc_atoms[(wave, rng_, pw)])
                    metas.append({"osc2_wave": wave, "osc2_range": rng_, "osc2_pw": pw, "mix_osc2": lv,
                                  "osc2_fine": _DETUNE})
            total, cut = score(np.array(srcs))
            for f_idx in np.argsort(total, axis=None)[:40]:
                si, mi, ri = np.unravel_index(int(f_idx), total.shape)
                patch = baseline_patch()
                patch.update(fixed)
                patch.update(comp_meta[ci])
                patch.update(metas[si])
                patch.update(FILTER_MODES[modes[mi]])
                late = cutoffs[int(cut["late"][si, mi, ri])] if "late" in cut else None
                early = cutoffs[int(cut["early"][si, mi, ri])] if "early" in cut else None
                held = late if late is not None else early
                patch.update(filter_freq=held, filter_res=ress[ri])
                results.append((float(total[si, mi, ri]), patch,
                                {"late": held, "early": early if early is not None else held}))
        results.sort(key=lambda item: item[0])

        # --- stage 3: the grid is coarse (a few units of cutoff move a resonance
        # peak across a harmonic, the levels come in three steps), so refine the
        # best combinations before ranking them
        seen_combo = set()
        todo = []
        for loss, patch, info in results:
            combo = (patch["osc1_wave"], patch["osc1_range"], patch["osc1_pw"], patch["mix_sub"] > 0,
                     patch["sub_wave"], patch["sub_oct"], patch["mix_osc2"] > 0, patch["osc2_wave"],
                     patch["osc2_range"], patch["osc2_pw"], patch["filter_type"], patch["filter_slope"],
                     patch["filter_shape"])
            if combo in seen_combo:
                continue
            seen_combo.add(combo)
            todo.append((patch, info))
            if len(todo) >= 120:
                break
        refined = self.map(lambda item: self._refine(item[0], item[1], open_db), todo)
        refined.sort(key=lambda item: item[0])
        if progress:
            progress(3)

        # best patch of each distinct structure
        from .search import structure_key, structure_of_patch

        out_list: List[Tuple[float, Dict[str, int], dict]] = []
        seen = set()
        for loss, patch, info in refined:
            key = structure_key(structure_of_patch(patch))
            if key in seen:
                continue
            seen.add(key)
            out_list.append((loss, patch, info))
            if len(out_list) >= keep:
                break
        return out_list
