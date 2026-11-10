"""Finding a patch with the software model (no hardware involved).

The search space has two kinds of variables:

* the *structure* of the patch - waveforms, octave ranges, which sources are
  switched on, the filter mode.  These are categorical.
* the *continuous* controls - levels, cutoff, resonance, envelopes ...

Structures are explored by sampling and by trying single changes of the best
ones; for every promising structure the continuous controls are optimised with
CMA-ES.  The budget is spent by successive halving: many structures get a
little attention, the best get most of it.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .model import Model
from .params import PARAMS, baseline_patch
from .static_match import FILTER_MODES, StaticMatcher

# ---------------------------------------------------------------------------
# search space
# ---------------------------------------------------------------------------

# structure variable -> (choices, prior probabilities)
STRUCT_VARS: Dict[str, Tuple[tuple, tuple]] = {
    "osc1_wave": ((0, 1, 2, 3), (0.1, 0.1, 0.45, 0.35)),
    "osc1_range": ((63, 64, 65, 66), (0.35, 0.45, 0.15, 0.05)),
    "osc2_on": ((0, 1), (0.5, 0.5)),
    "osc2_wave": ((0, 1, 2, 3), (0.1, 0.1, 0.45, 0.35)),
    "osc2_range": ((63, 64, 65, 66), (0.35, 0.45, 0.15, 0.05)),
    "sub_on": ((0, 1), (0.55, 0.45)),
    "sub_wave": ((0, 1, 2), (0.45, 0.2, 0.35)),
    "sub_oct": ((62, 63), (0.25, 0.75)),
    "filter_mode": (("lp24", "lp12", "acid", "bp12", "bp24", "hp12", "hp24"),
                    (0.42, 0.22, 0.2, 0.05, 0.03, 0.05, 0.03)),
    "noise_on": ((0, 1), (0.85, 0.15)),
}

Structure = Dict[str, object]


def structure_key(st: Structure) -> tuple:
    """Signature of a structure; settings of switched-off parts do not count."""
    return (
        st["osc1_wave"], st["osc1_range"],
        (st["osc2_wave"], st["osc2_range"]) if st["osc2_on"] else None,
        (st["sub_wave"], st["sub_oct"]) if st["sub_on"] else None,
        st["filter_mode"], st["noise_on"],
    )


def random_structure(rng: np.random.Generator) -> Structure:
    return {name: choices[int(rng.choice(len(choices), p=np.asarray(prior) / sum(prior)))]
            for name, (choices, prior) in STRUCT_VARS.items()}


def structure_neighbours(st: Structure) -> List[Structure]:
    """All structures that differ in exactly one (relevant) variable."""
    out = []
    for name, (choices, _) in STRUCT_VARS.items():
        if name in ("osc2_wave", "osc2_range") and not st["osc2_on"]:
            continue
        if name in ("sub_wave", "sub_oct") and not st["sub_on"]:
            continue
        for choice in choices:
            if choice != st[name]:
                nb = dict(st)
                nb[name] = choice
                out.append(nb)
    return out


def structure_of_patch(patch: Dict[str, int]) -> Structure:
    mode = "lp24"
    for name, values in FILTER_MODES.items():
        if all(patch.get(k) == v for k, v in values.items()):
            mode = name
            break
    if patch.get("filter_type") == 1:
        mode = "acid"
    return {
        "osc1_wave": patch.get("osc1_wave", 2), "osc1_range": patch.get("osc1_range", 64),
        "osc2_on": int(patch.get("mix_osc2", 0) > 0), "osc2_wave": patch.get("osc2_wave", 2),
        "osc2_range": patch.get("osc2_range", 64),
        "sub_on": int(patch.get("mix_sub", 0) > 0), "sub_wave": patch.get("sub_wave", 0),
        "sub_oct": patch.get("sub_oct", 63), "filter_mode": mode,
        "noise_on": int(patch.get("mix_noise", 0) > 0),
    }


@dataclass(frozen=True)
class Dim:
    key: str
    lo: int
    hi: int
    start: float  # typical value (normalised 0..1) used when nothing better is known


def continuous_dims(st: Structure) -> List[Dim]:
    dims = [Dim("mix_osc1", 70, 255, 0.9)]
    if st["osc1_wave"] == 3:
        dims.append(Dim("osc1_pw", 0, 64, 0.8))
    if st["osc2_on"]:
        dims.append(Dim("mix_osc2", 50, 255, 0.7))
        dims.append(Dim("osc2_fine", 88, 168, 0.5))
        if st["osc2_wave"] == 3:
            dims.append(Dim("osc2_pw", 0, 64, 0.8))
    if st["sub_on"]:
        dims.append(Dim("mix_sub", 50, 255, 0.7))
    if st["noise_on"]:
        dims.append(Dim("mix_noise", 15, 220, 0.3))
    dims += [
        Dim("filter_freq", 0, 255, 0.6),
        Dim("filter_res", 0, 118, 0.2),
        Dim("filter_env", 0, 127, 0.6),
        Dim("filter_drive", 0, 127, 0.1),
        Dim("filter_track", 0, 7, 0.0),
        Dim("mod_attack", 0, 100, 0.0),
        Dim("mod_decay", 0, 127, 0.5),
        Dim("mod_sustain", 0, 127, 0.4),
        Dim("mod_release", 0, 127, 0.3),
        Dim("amp_attack", 0, 110, 0.0),
        Dim("amp_decay", 0, 127, 0.6),
        Dim("amp_sustain", 0, 127, 0.8),
        Dim("amp_release", 0, 127, 0.2),
        Dim("distortion", 0, 127, 0.05),
    ]
    return dims


def build_patch(st: Structure, dims: Sequence[Dim], x: np.ndarray, extra: Optional[Dict[str, int]] = None) -> Dict[str, int]:
    """Structure + normalised continuous vector -> complete raw patch."""
    patch = baseline_patch()
    if extra:
        patch.update(extra)
    patch["osc1_wave"] = int(st["osc1_wave"])
    patch["osc1_range"] = int(st["osc1_range"])
    patch["mix_osc2"] = 0
    patch["mix_sub"] = 0
    patch["mix_noise"] = 0
    if st["osc2_on"]:
        patch["osc2_wave"] = int(st["osc2_wave"])
        patch["osc2_range"] = int(st["osc2_range"])
    if st["sub_on"]:
        patch["sub_wave"] = int(st["sub_wave"])
        patch["sub_oct"] = int(st["sub_oct"])
    patch.update(FILTER_MODES[str(st["filter_mode"])])
    xc = np.clip(x, 0.0, 1.0)
    for d, v in zip(dims, xc):
        patch[d.key] = int(round(d.lo + float(v) * (d.hi - d.lo)))
    return patch


def patch_vector(patch: Dict[str, int], dims: Sequence[Dim]) -> np.ndarray:
    return np.array([(min(max(patch[d.key], d.lo), d.hi) - d.lo) / float(d.hi - d.lo) for d in dims])


def complexity_penalty(patch: Dict[str, int]) -> float:
    """Tiny preference for simpler patches when the sound is equally close."""
    pen = 0.0
    if patch["mix_osc2"] > 0:
        pen += 0.010
    if patch["mix_sub"] > 0:
        pen += 0.005
    if patch["mix_noise"] > 0:
        pen += 0.006
    pen += 0.006 * patch["distortion"] / 127.0 + 0.005 * patch["filter_drive"] / 127.0
    loud = max(patch["mix_osc1"], patch["mix_osc2"], patch["mix_sub"])
    pen += 0.004 * (255 - loud) / 255.0  # keep the loudest source up (signal to noise on the hardware)
    return pen


# ---------------------------------------------------------------------------
# CMA-ES
# ---------------------------------------------------------------------------


class CMA:
    """Compact (mu/mu_w, lambda)-CMA-ES on the unit box."""

    def __init__(self, x0: np.ndarray, sigma0: float, rng: np.random.Generator, lam: Optional[int] = None):
        n = len(x0)
        self.n = n
        self.rng = rng
        self.lam = lam or (4 + int(3 * math.log(n)))
        self.mu = self.lam // 2
        w = math.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = w / w.sum()
        self.mueff = 1.0 / float(np.sum(self.w ** 2))
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0.0, math.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.chi_n = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.mean = np.array(x0, dtype=np.float64)
        self.sigma = float(sigma0)
        self.C = np.eye(n)
        self.pc = np.zeros(n)
        self.ps = np.zeros(n)
        self.B = np.eye(n)
        self.D = np.ones(n)
        self.gen = 0
        self._eigen_gen = 0
        self._ys: Optional[np.ndarray] = None

    def ask(self) -> np.ndarray:
        z = self.rng.standard_normal((self.lam, self.n))
        self._ys = (z * self.D) @ self.B.T
        return self.mean + self.sigma * self._ys

    def tell(self, fitness: np.ndarray) -> None:
        order = np.argsort(fitness)[:self.mu]
        ys = self._ys[order]
        y_w = self.w @ ys
        self.mean = np.clip(self.mean + self.sigma * y_w, -0.15, 1.15)
        inv_sqrt = (self.B / self.D) @ self.B.T
        self.ps = (1 - self.cs) * self.ps + math.sqrt(self.cs * (2 - self.cs) * self.mueff) * (inv_sqrt @ y_w)
        self.gen += 1
        hsig = (np.linalg.norm(self.ps) / math.sqrt(1 - (1 - self.cs) ** (2 * self.gen)) / self.chi_n
                < 1.4 + 2 / (self.n + 1))
        self.pc = (1 - self.cc) * self.pc + (math.sqrt(self.cc * (2 - self.cc) * self.mueff) * y_w if hsig else 0.0)
        rank_mu = (ys * self.w[:, None]).T @ ys
        self.C = ((1 - self.c1 - self.cmu) * self.C
                  + self.c1 * (np.outer(self.pc, self.pc) + (0.0 if hsig else self.cc * (2 - self.cc)) * self.C)
                  + self.cmu * rank_mu)
        self.sigma *= math.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chi_n - 1))
        self.sigma = float(min(max(self.sigma, 1e-4), 0.6))
        if self.gen - self._eigen_gen >= max(1, int(1.0 / ((self.c1 + self.cmu) * self.n * 10))):
            self._eigen_gen = self.gen
            self.C = np.triu(self.C) + np.triu(self.C, 1).T
            vals, vecs = np.linalg.eigh(self.C)
            vals = np.maximum(vals, 1e-12)
            self.D = np.sqrt(vals)
            self.B = vecs

    def spread(self) -> float:
        """Largest standard deviation of any coordinate."""
        return float(self.sigma * math.sqrt(np.max(np.diag(self.C))))


# ---------------------------------------------------------------------------
# objective with cache
# ---------------------------------------------------------------------------


class Objective:
    """Batch objective ``patches -> losses`` with a cache and bookkeeping."""

    def __init__(self, batch_fn: Callable[[List[Dict[str, int]]], Sequence[float]],
                 penalty: Optional[Callable[[Dict[str, int]], float]] = None):
        self.batch_fn = batch_fn
        self.penalty = penalty
        self.cache: Dict[tuple, float] = {}
        self.evals = 0
        self.best_loss = float("inf")
        self.best_patch: Optional[Dict[str, int]] = None
        self.history: List[Tuple[int, float]] = []

    @staticmethod
    def key(patch: Dict[str, int]) -> tuple:
        return tuple(patch[k] for k in _PATCH_KEYS)

    def __call__(self, patches: List[Dict[str, int]]) -> np.ndarray:
        keys = [self.key(p) for p in patches]
        todo: Dict[tuple, Dict[str, int]] = {}
        for k, p in zip(keys, patches):
            if k not in self.cache and k not in todo:
                todo[k] = p
        if todo:
            vals = self.batch_fn(list(todo.values()))
            for (k, p), v in zip(todo.items(), vals):
                v = float(v) + (self.penalty(p) if self.penalty else 0.0)
                self.cache[k] = v
                if v < self.best_loss:
                    self.best_loss = v
                    self.best_patch = dict(p)
            self.evals += len(todo)
            self.history.append((self.evals, self.best_loss))
        return np.array([self.cache[k] for k in keys])

    def one(self, patch: Dict[str, int]) -> float:
        return float(self([patch])[0])

    def reset(self) -> None:
        """Forget cached losses (the underlying objective has changed)."""
        self.cache.clear()
        self.best_loss = float("inf")
        self.best_patch = None


# ---------------------------------------------------------------------------
# block-wise patch optimiser
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    patch: Dict[str, int]
    loss: float

    @property
    def structure(self) -> Structure:
        return structure_of_patch(self.patch)

    def label(self) -> str:
        return describe_structure(self.patch)


def describe_structure(patch: Dict[str, int]) -> str:
    st = structure_of_patch(patch)
    waves = ("sine", "tri", "saw", "pulse")
    rng = {63: "16'", 64: "8'", 65: "4'", 66: "2'"}
    parts = [f"{waves[st['osc1_wave']]} {rng[st['osc1_range']]}"]
    if st["osc2_on"]:
        parts.append(f"+ {waves[st['osc2_wave']]} {rng[st['osc2_range']]}")
    if st["sub_on"]:
        parts.append(f"+ sub {('sine', 'pulse', 'square')[st['sub_wave']]} {st['sub_oct'] - 64}")
    if st["noise_on"]:
        parts.append("+ noise")
    parts.append(f"/ {st['filter_mode']}")
    return " ".join(parts)


@dataclass
class SearchResult:
    candidates: List[Candidate]
    evaluations: int
    seconds: float
    history: List[Tuple[int, float]] = field(default_factory=list)

    @property
    def best(self) -> Candidate:
        return self.candidates[0]


_RANGES = {d.key: (d.lo, d.hi) for d in continuous_dims(
    {"osc1_wave": 3, "osc1_range": 64, "osc2_on": 1, "osc2_wave": 3, "osc2_range": 64, "sub_on": 1,
     "sub_wave": 0, "sub_oct": 63, "filter_mode": "lp24", "noise_on": 1})}


class PatchOptimizer:
    """Builds a patch the way a sound designer does: block by block.

    Every block (amp envelope, oscillator 1, filter, filter envelope, sub,
    oscillator 2, extras) is optimised with the others held fixed; switches
    are simply enumerated, continuous controls get line searches.  A few
    sweeps over the blocks are followed by a joint CMA-ES polish.
    """

    def __init__(self, objective: Objective, seed: int = 0,
                 progress: Optional[Callable[[dict], None]] = None,
                 should_stop: Optional[Callable[[], bool]] = None,
                 frozen: Optional[Sequence[str]] = None):
        self.f = objective
        self.rng = np.random.default_rng(seed)
        self.progress = progress
        self.should_stop = should_stop or (lambda: False)
        self.frozen = set(frozen or [])
        self.phase = ""
        self.min_gain = 0.0005

    # -- helpers --------------------------------------------------------------
    def _report(self, patch: Dict[str, int], loss: float, **info) -> None:
        if self.progress is not None:
            info.update(phase=self.phase, evaluations=self.f.evals, best=loss, leader=patch)
            self.progress(info)

    def _allowed(self, update: Dict[str, int]) -> bool:
        return not any(k in self.frozen for k in update)

    def try_variants(self, patch: Dict[str, int], loss: float, variants: List[Dict[str, int]],
                     reopt: Sequence[str] = (), top: int = 3):
        """Apply each variant (dict of changes) to the patch and keep the best.

        A switch rarely pays off on its own - the controls coupled to it have
        to follow.  ``reopt`` names those controls: the ``top`` variants get a
        quick line search of each of them before the decision is made.
        """
        variants = [v for v in variants if self._allowed(v)]
        if not variants or self.should_stop():
            return patch, loss
        cands = []
        for v in variants:
            c = dict(patch)
            c.update(v)
            cands.append(c)
        losses = self.f(cands)
        order = np.argsort(losses)
        best_patch, best_loss = cands[int(order[0])], float(losses[order[0]])
        if reopt:
            for i in order[:top]:
                c, l = cands[int(i)], float(losses[i])
                if l > loss + 0.08:
                    continue  # hopeless, do not waste evaluations on it
                for key in reopt:
                    c, l = self.line(c, l, key, 0.35, points=5)
                if l < best_loss:
                    best_patch, best_loss = c, l
        if best_loss < loss - self.min_gain:
            return best_patch, best_loss
        return patch, loss

    def line(self, patch: Dict[str, int], loss: float, key: str, width: float = 1.0, points: int = 9):
        """Line search of one control inside ``width`` (fraction of its range) around the current value."""
        if key in self.frozen or self.should_stop():
            return patch, loss
        lo, hi = _RANGES.get(key, (PARAMS[key].lo, PARAMS[key].hi))
        cur = min(max(patch[key], lo), hi)
        span = (hi - lo) * width
        a = max(lo, cur - span / 2.0) if width < 1.0 else lo
        b = min(hi, cur + span / 2.0) if width < 1.0 else hi
        grid = sorted({int(round(v)) for v in np.linspace(a, b, points)} | {cur})
        step = (b - a) / max(points - 1, 1)
        best_v, best_l = cur, loss
        values = [v for v in grid if v != cur]
        for _ in range(4):
            if not values:
                break
            losses = self.f([dict(patch, **{key: v}) for v in values])
            i = int(np.argmin(losses))
            if losses[i] < best_l - 1e-9:
                best_v, best_l = values[i], float(losses[i])
            step /= 2.0
            if step < 0.75:
                break
            values = sorted({int(round(min(max(best_v + d, lo), hi))) for d in (-step, step)} - {best_v})
        if best_l < loss - self.min_gain:
            patch = dict(patch, **{key: best_v})
            return patch, best_l
        return patch, loss

    def polish(self, patch: Dict[str, int], loss: float, keys: Sequence[str], sigma: float, gens: int,
               lam: Optional[int] = None, bounds: Optional[Dict[str, Tuple[int, int]]] = None):
        """Joint CMA-ES refinement of several continuous controls around the patch.

        ``bounds`` optionally narrows the range of individual controls (a trust region).
        """
        keys = [k for k in keys if k not in self.frozen]
        if len(keys) < 2 or self.should_stop():
            return patch, loss
        full = [_RANGES.get(k, (PARAMS[k].lo, PARAMS[k].hi)) for k in keys]
        if bounds:
            full = [(max(lo, bounds[k][0]), min(hi, bounds[k][1])) if k in bounds else (lo, hi)
                    for k, (lo, hi) in zip(keys, full)]
            keep = [i for i, (lo, hi) in enumerate(full) if hi > lo]
            keys = [keys[i] for i in keep]
            full = [full[i] for i in keep]
            if len(keys) < 2:
                return patch, loss
        bounds = full
        x0 = np.array([(min(max(patch[k], lo), hi) - lo) / float(hi - lo) for k, (lo, hi) in zip(keys, bounds)])
        cma = CMA(x0, sigma, self.rng, lam=lam)
        best_patch, best_loss = patch, loss
        for g in range(gens):
            if self.should_stop() or cma.spread() < 0.003:
                break
            xs = cma.ask()
            cands = []
            for x in xs:
                c = dict(patch)
                for k, (lo, hi), v in zip(keys, bounds, np.clip(x, 0.0, 1.0)):
                    c[k] = int(round(lo + float(v) * (hi - lo)))
                cands.append(c)
            losses = self.f(cands)
            i = int(np.argmin(losses))
            if losses[i] < best_loss - 1e-9:
                best_patch, best_loss = cands[i], float(losses[i])
            cma.tell(losses + 2.0 * np.sum((xs - np.clip(xs, 0.0, 1.0)) ** 2, axis=1))
            if g % 4 == 0:
                self._report(best_patch, best_loss)
        if best_loss < loss - self.min_gain:
            return best_patch, best_loss
        return patch, loss

    # -- blocks ---------------------------------------------------------------
    def block_amp(self, patch, loss, width):
        for key in ("amp_sustain", "amp_decay", "amp_release", "amp_attack"):
            patch, loss = self.line(patch, loss, key, width)
        return patch, loss

    def block_osc1(self, patch, loss, width):
        variants = []
        for wave in range(4):
            for rng_ in (63, 64, 65, 66):
                if wave == 3:
                    for pw in (64, 44, 24, 8):
                        variants.append({"osc1_wave": wave, "osc1_range": rng_, "osc1_pw": pw})
                else:
                    variants.append({"osc1_wave": wave, "osc1_range": rng_})
        patch, loss = self.try_variants(patch, loss, variants, reopt=("filter_freq",))
        if patch["osc1_wave"] == 3:
            patch, loss = self.line(patch, loss, "osc1_pw", min(1.0, width * 2))
        return patch, loss

    def block_filter(self, patch, loss, width, modes: Optional[Sequence[str]] = None):
        cur_mode = structure_of_patch(patch)["filter_mode"]
        variants = []
        if width >= 1.0:
            cutoffs = [int(v) for v in np.linspace(25, 255, 13)]
            ress = [0, 35, 70, 105]
        else:
            c0 = patch["filter_freq"]
            cutoffs = sorted({int(min(max(c0 + d, 0), 255)) for d in (-36, -18, 0, 18, 36)})
            ress = sorted({int(min(max(patch["filter_res"] + d, 0), 118)) for d in (-25, 0, 25)})
        for mode in (modes or FILTER_MODES.keys()):
            for c in cutoffs:
                for r in ress:
                    v = dict(FILTER_MODES[mode])
                    v.update(filter_freq=c, filter_res=r)
                    variants.append(v)
        patch, loss = self.try_variants(patch, loss, variants)
        for key in ("filter_freq", "filter_res", "filter_freq"):
            patch, loss = self.line(patch, loss, key, min(1.0, width * 0.5), points=7)
        return patch, loss

    def block_filter_env(self, patch, loss, width):
        variants = []
        if width >= 1.0:
            for depth in (64, 76, 88, 100, 112, 127, 50, 30):
                for decay in (20, 40, 60, 80, 100):
                    for sus in (0, 50, 100):
                        variants.append({"filter_env": depth, "mod_decay": decay, "mod_sustain": sus})
            patch, loss = self.try_variants(patch, loss, variants)
        if patch["filter_env"] not in (63, 64) or width >= 1.0:
            for key in ("filter_env", "mod_decay", "mod_sustain", "filter_freq", "mod_attack", "mod_release"):
                patch, loss = self.line(patch, loss, key, min(1.0, width), points=7)
            if patch["filter_env"] not in (63, 64):
                patch, loss = self.polish(patch, loss, ["filter_freq", "filter_env", "mod_decay", "mod_sustain",
                                                         "filter_res"], 0.12 * min(1.0, width * 2), 10)
        return patch, loss

    def block_sub(self, patch, loss, width):
        variants = [{"mix_sub": 0}]
        for wave in range(3):
            for octv in (62, 63):
                for level in (90, 160, 255):
                    variants.append({"sub_wave": wave, "sub_oct": octv, "mix_sub": level})
        patch, loss = self.try_variants(patch, loss, variants, reopt=("mix_osc1", "filter_freq"))
        if patch["mix_sub"] > 0:
            patch, loss = self.line(patch, loss, "mix_sub", min(1.0, width * 2), points=7)
        return patch, loss

    def block_osc2(self, patch, loss, width):
        variants = [{"mix_osc2": 0}]
        for wave in range(4):
            for rng_ in (63, 64, 65, 66):
                for level in (110, 255):
                    for fine in (128, 139):
                        v = {"osc2_wave": wave, "osc2_range": rng_, "mix_osc2": level, "osc2_fine": fine}
                        if wave == 3:
                            v["osc2_pw"] = patch["osc2_pw"] if patch["mix_osc2"] > 0 else 64
                        variants.append(v)
        patch, loss = self.try_variants(patch, loss, variants, reopt=("osc2_fine", "mix_osc1", "filter_freq"))
        if patch["mix_osc2"] > 0:
            patch, loss = self.line(patch, loss, "osc2_fine", min(1.0, width * 2), points=9)
            patch, loss = self.line(patch, loss, "mix_osc2", min(1.0, width * 2), points=7)
            if patch["osc2_wave"] == 3:
                patch, loss = self.line(patch, loss, "osc2_pw", min(1.0, width * 2), points=7)
        return patch, loss

    def block_extras(self, patch, loss, width):
        patch, loss = self.try_variants(patch, loss, [{"mix_noise": v} for v in (0, 25, 60, 120)])
        patch, loss = self.try_variants(patch, loss, [{"filter_drive": v} for v in (0, 25, 50, 80, 127)],
                                        reopt=("filter_freq", "filter_res", "mix_osc1"), top=4)
        patch, loss = self.try_variants(patch, loss, [{"distortion": v} for v in (0, 25, 50, 80, 127)],
                                        reopt=("filter_res", "filter_freq", "amp_sustain"), top=4)
        patch, loss = self.try_variants(patch, loss, [{"filter_track": v} for v in (0, 2, 4, 7)],
                                        reopt=("filter_freq",))
        for key in ("filter_drive", "distortion", "mix_noise"):
            if patch[key] > 0:
                patch, loss = self.line(patch, loss, key, min(1.0, width), points=7)
        if patch["mix_osc2"] > 0 or patch["mix_sub"] > 0 or patch["filter_drive"] > 0 or patch["distortion"] > 0:
            patch, loss = self.line(patch, loss, "mix_osc1", min(1.0, width), points=7)
        return patch, loss

    def sweep(self, patch, loss, width: float):
        blocks = [("amp envelope", self.block_amp), ("oscillator 1", self.block_osc1),
                  ("filter", self.block_filter), ("filter envelope", self.block_filter_env),
                  ("sub oscillator", self.block_sub), ("oscillator 2", self.block_osc2),
                  ("drive / noise", self.block_extras), ("amp envelope", self.block_amp)]
        for name, block in blocks:
            if self.should_stop():
                break
            self.phase = name
            patch, loss = block(patch, loss, width)
            self._report(patch, loss)
        return patch, loss

    def active_keys(self, patch: Dict[str, int]) -> List[str]:
        return [d.key for d in continuous_dims(structure_of_patch(patch)) if d.key not in self.frozen]

    def optimise(self, start: Dict[str, int], sweeps: Sequence[float] = (1.0, 0.4, 0.15),
                 polish_gens: int = 30) -> Tuple[Dict[str, int], float]:
        patch = dict(start)
        loss = self.f.one(patch)
        for width in sweeps:
            before = loss
            patch, loss = self.sweep(patch, loss, width)
            if self.should_stop():
                break
            if width < 1.0 and before - loss < 0.002:
                break
        if polish_gens > 0 and not self.should_stop():
            self.phase = "joint polish"
            patch, loss = self.polish(patch, loss, self.active_keys(patch), 0.07, polish_gens)
            self._report(patch, loss)
        return patch, loss


def default_start() -> Dict[str, int]:
    patch = baseline_patch()
    patch.update(filter_freq=150, amp_release=20)
    return patch


DYNAMIC_KEYS = ("amp_attack", "amp_decay", "amp_sustain", "amp_release", "filter_freq", "filter_env",
                "mod_attack", "mod_decay", "mod_sustain", "mod_release")


class ModelSearch:
    """Global search for the patch whose model render is closest to the target.

    1. *static timbre* - an exhaustive search over the switches (waveforms,
       octaves, sub, oscillator 2, filter mode) on the spectrum of a held note
       at two moments, using a spectral dictionary (see static_match.py),
    2. *dynamics* - amp and filter envelopes for the best structures, started
       from several hypotheses ("the filter closes", "the amp decays", ...),
    3. *joint refinement* - all continuous controls together, then block-wise
       sweeps and a CMA-ES polish with phase-averaged renders.
    """

    def __init__(self, model: Model, seed: int = 0, fixed: Optional[Dict[str, int]] = None,
                 progress: Optional[Callable[[dict], None]] = None,
                 should_stop: Optional[Callable[[], bool]] = None):
        self.model = model
        self.seed = seed
        self.fixed = dict(fixed or {})
        self.progress = progress
        self.should_stop = should_stop or (lambda: False)
        self.objective = Objective(lambda patches: self.model.map(self.model.loss, patches), complexity_penalty)
        self.static = StaticMatcher(model.physics, model.target, pool_map=model.map)
        self.stage_info: Dict[str, object] = {}

    # -- stage 1 ---------------------------------------------------------------
    def _stage_static(self, effort: float) -> List[Tuple[float, Dict[str, int], dict]]:
        def report(step: int) -> None:
            if self.progress is not None:
                self.progress({"phase": "timbre", "step": step, "total": 3, "evaluations": 0,
                               "best": float("nan"), "leader": None})

        ranked = self.static.dictionary_search(keep=max(4, int(round(8 * effort))), progress=report,
                                               fixed=self.fixed)
        self.stage_info["static"] = [(round(l, 4), describe_structure(c), info) for l, c, info in ranked]
        return ranked

    # -- stage 2 ---------------------------------------------------------------
    def _hypotheses(self, patch: Dict[str, int], info: dict) -> List[Dict[str, int]]:
        """Different explanations of how the sound evolves in time."""
        ph = self.model.physics
        table = ph.cutoff_acid if patch["filter_type"] == 1 else ph.cutoff_classic
        held = float(table[info["late"]])  # cutoff that fits the settled part of the notes
        sweep = float(table[info["early"]]) - held  # how much more open the filter is at the start (octaves)

        def raw_for(log2hz: float) -> int:
            return int(np.argmin(np.abs(table - log2hz)))

        def closing(depth_oct: float, sustain: int) -> Dict[str, int]:
            """Envelope depth and base cutoff so that the cutoff settles on ``held``."""
            depth_raw = int(min(max(64 + round(depth_oct / ph.env_oct_per_unit), 0), 127))
            actual = (depth_raw - 64) * ph.env_oct_per_unit
            level = (sustain / 127.0) ** ph.sustain_gamma
            return {"filter_env": depth_raw, "filter_freq": raw_for(held - actual * level), "mod_sustain": sustain}

        common = {"amp_attack": 0, "amp_release": 25, "mod_attack": 0, "mod_release": 30, "mod_decay": 55}
        organ = {"amp_decay": 0, "amp_sustain": 127}
        hyps = [
            # the filter stands still, the amplifier shapes the note
            dict(common, filter_env=64, filter_freq=info["late"], amp_decay=70, amp_sustain=100, mod_sustain=64),
            # generic plucks
            dict(common, **organ, **closing(5.0, 45)),
            dict(common, **closing(4.5, 30), amp_decay=75, amp_sustain=60),
        ]
        if abs(sweep) > 0.3:
            # the first 130 ms average over the sweep, so the real start is further out
            hyps.append(dict(common, **organ, **closing(1.6 * sweep / 0.75, 40)))
            hyps.append(dict(common, **organ, **closing(1.4 * sweep, 0)))
            hyps.append(dict(common, **closing(1.6 * sweep / 0.75, 40), amp_decay=75, amp_sustain=60))
        return [dict(patch, **{k: v for k, v in h.items() if k not in self.fixed}) for h in hyps]

    def _timbre_keys(self, patch: Dict[str, int]) -> List[str]:
        keys = ["filter_res", "mix_osc1"]
        if patch["osc1_wave"] == 3:
            keys.append("osc1_pw")
        if patch["mix_osc2"] > 0:
            keys.append("mix_osc2")
            if patch["osc2_wave"] == 3:
                keys.append("osc2_pw")
        if patch["mix_sub"] > 0:
            keys.append("mix_sub")
        return keys

    def _quick_dynamics(self, opt: "PatchOptimizer", patch: Dict[str, int], info: dict, effort: float):
        hyps = self._hypotheses(patch, info)
        losses = self.objective(hyps)
        trials = []
        for i in np.argsort(losses)[:2]:
            h, l = opt.block_amp(hyps[int(i)], float(losses[i]), 1.0)
            h, l = opt.polish(h, l, DYNAMIC_KEYS, 0.18, max(3, int(round(5 * effort))), lam=12)
            trials.append((l, h))
        trials.sort(key=lambda item: item[0])
        return trials[0][1], trials[0][0]

    # -- the search -------------------------------------------------------------
    def run(self, effort: float = 1.0, seeds: Optional[List[Dict[str, int]]] = None,
            max_seconds: Optional[float] = None) -> SearchResult:
        t_start = time.time()

        def stop() -> bool:
            return self.should_stop() or (max_seconds is not None and time.time() - t_start > max_seconds)

        statics = self._stage_static(effort)
        opt = PatchOptimizer(self.objective, self.seed, self.progress, stop, frozen=list(self.fixed))

        # envelopes: a quick fit for every structure, a thorough one for the best
        self.model.set_precision(fine=False)
        quick: List[Tuple[float, Dict[str, int]]] = []
        for seed_patch in seeds or []:
            quick.append((self.objective.one(seed_patch), dict(seed_patch)))
        for i, (_, st, info) in enumerate(statics):
            if stop() and quick:
                break
            opt.phase = f"envelopes ({i + 1}/{len(statics)})"
            patch, loss = self._quick_dynamics(opt, st, info, effort)
            quick.append((loss, patch))
            opt._report(min(quick, key=lambda item: item[0])[1], min(q[0] for q in quick))
        quick.sort(key=lambda item: item[0])
        self.stage_info["quick"] = [(round(l, 4), describe_structure(c)) for l, c in quick]
        thorough: List[Tuple[float, Dict[str, int]]] = []
        for i, (loss, patch) in enumerate(quick[:max(1, int(round(3 * effort)))]):
            if stop() and thorough:
                break
            opt.phase = f"envelopes + timbre ({i + 1})"
            keys = list(DYNAMIC_KEYS) + self._timbre_keys(patch)
            patch, loss = opt.polish(patch, loss, keys, 0.12, max(6, int(round(22 * effort))), lam=14)
            patch, loss = opt.block_extras(patch, loss, 1.0)
            thorough.append((loss, patch))
            opt._report(patch, loss)
        thorough.sort(key=lambda item: item[0])
        self.stage_info["thorough"] = [(round(l, 4), describe_structure(c)) for l, c in thorough]

        # joint refinement with phase-averaged renders
        self.model.set_precision(fine=True)
        self.objective.reset()
        results: List[Candidate] = []
        for i, (_, patch) in enumerate(thorough):
            if i >= 2 or (stop() and results):
                results.append(Candidate(clean_patch(patch), self.objective.one(patch)))
                continue
            patch, loss = opt.optimise(patch, sweeps=(0.3, 0.12) if i == 0 else (0.3,),
                                       polish_gens=int(30 * effort) if i == 0 else int(10 * effort))
            results.append(Candidate(clean_patch(patch), loss))
        results.sort(key=lambda c: c.loss)
        unique, seen = [], set()
        for c in results:
            k = Objective.key(c.patch)
            if k not in seen:
                seen.add(k)
                unique.append(c)
        return SearchResult(unique, self.objective.evals, time.time() - t_start, list(self.objective.history))


_PATCH_KEYS = tuple(PARAMS.keys())


def clean_patch(patch: Dict[str, int]) -> Dict[str, int]:
    """Reset controls that cannot be heard to their neutral values."""
    out = dict(patch)
    base = baseline_patch()
    if out["mix_osc2"] == 0 and out["mix_ring"] == 0 and out["osc_filter_mod"] == 0 and not out["osc_sync"]:
        for key in ("osc2_wave", "osc2_range", "osc2_coarse", "osc2_fine", "osc2_pw", "osc2_env_pitch",
                    "osc2_lfo1_pitch", "osc2_env_pw", "osc2_lfo2_pw"):
            out[key] = base[key]
    if out["mix_sub"] == 0:
        out["sub_wave"] = base["sub_wave"]
        out["sub_oct"] = base["sub_oct"]
    for n in (1, 2):
        if out[f"osc{n}_wave"] != 3:
            for key in (f"osc{n}_pw", f"osc{n}_env_pw", f"osc{n}_lfo2_pw"):
                out[key] = base[key]
    if out["filter_type"] == 1:
        out["filter_slope"] = 1
        out["filter_shape"] = 0
    mod_used = (out["filter_env"] not in (63, 64) or out["osc1_env_pitch"] not in (63, 64)
                or out["osc2_env_pitch"] not in (63, 64) or out["osc1_env_pw"] not in (63, 64)
                or out["osc2_env_pw"] not in (63, 64))
    if not mod_used:
        for key in ("mod_attack", "mod_decay", "mod_sustain", "mod_release"):
            out[key] = base[key]
    if out["amp_sustain"] >= 127:
        out["amp_decay"] = base["amp_decay"]
    return out
