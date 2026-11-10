"""Closed-loop refinement on the instrument.

The model search delivers a patch that sounds right *in the model*.  The real
instrument differs from the model, so the patch is now corrected with real
measurements.  Playing the phrase takes seconds, so every measurement has to
count:

* **Model-guided steps.**  A measurement yields a complete picture of the
  sound (spectrogram, envelope ...), not just one number.  The difference
  between that picture and what the model predicted for the same patch is the
  model's error *at this point*.  Adding that error to the model's prediction
  for nearby patches gives a corrected model; its best patch inside a trust
  region is the next thing to measure.  This usually closes most of the gap
  within a handful of measurements.
* **Structure checks.**  The corrected model also proposes changes of the
  switches (waveform, octave, filter mode ...); promising ones are measured.
* **Direct search.**  When the model has nothing more to offer, single
  controls are nudged and judged by measurement alone.

The oscillators run free, so repeated measurements of one patch differ a
little; decisions take that noise into account.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .device import Device
from .engine import Phrase
from .features import Aligned, Features, Target, extract
from .model import Model
from .params import PARAMS
from .search import (Objective, PatchOptimizer, _RANGES, clean_patch, complexity_penalty, continuous_dims,
                     describe_structure, structure_key, structure_of_patch)


@dataclass
class Measured:
    """All measurements of one patch on the instrument."""

    patch: Dict[str, int]
    feats: List[Features] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)
    audio: Optional[np.ndarray] = None
    parts: Optional[dict] = None

    @property
    def n(self) -> int:
        return len(self.losses)

    @property
    def loss(self) -> float:
        return float(np.mean(self.losses)) if self.losses else float("inf")


@dataclass
class RefineResult:
    patch: Dict[str, int]
    loss: float
    parts: dict
    audio: np.ndarray
    plays: int
    history: List[dict]
    start_loss: float
    noise: float


class HardwareLoop:
    def __init__(self, device: Device, model: Model, target: Target, phrase: Phrase,
                 progress: Optional[Callable[[dict], None]] = None,
                 should_stop: Optional[Callable[[], bool]] = None, seed: int = 0,
                 frozen: Optional[Sequence[str]] = None):
        self.dev = device
        self.model = model
        self.target = target
        self.phrase = phrase
        self.progress = progress
        self.should_stop = should_stop or (lambda: False)
        self.rng = np.random.default_rng(seed)
        self.frozen = set(frozen or [])
        self.measured: Dict[tuple, Measured] = {}
        self.history: List[dict] = []
        self.plays = 0
        self.noise = 0.012  # standard deviation of repeated measurements (updated as we learn)
        self._noise_samples: List[float] = []
        self.best: Optional[Measured] = None
        self.phase = "start"
        self.clipped = False

    # -- measuring --------------------------------------------------------------
    def measure(self, patch: Dict[str, int], note: str = "") -> Measured:
        key = Objective.key(patch)
        m = self.measured.get(key)
        if m is None:
            m = self.measured[key] = Measured(dict(patch))
        y = self.dev.capture(patch, self.phrase, self.target.n_samples)
        if float(np.max(np.abs(y))) > 0.985:
            self.clipped = True
        f = extract(y, self.target.cfg)
        cmp_ = self.target.compare(f)
        m.feats.append(f)
        m.losses.append(cmp_["loss"])
        m.audio = y
        m.parts = cmp_
        self.plays += 1
        if m.n >= 2:
            self._noise_samples.append(float(np.std(m.losses, ddof=1)))
            self.noise = float(min(max(np.median(self._noise_samples), 0.004), 0.08))
        improved = self.best is None or m.loss < self.best.loss
        self.history.append({"play": self.plays, "loss": cmp_["loss"], "mean": m.loss, "phase": self.phase,
                             "note": note, "best": min(m.loss, self.best.loss) if self.best else m.loss})
        if self.progress is not None:
            self.progress({"phase": self.phase, "play": self.plays, "loss": cmp_["loss"],
                           "best": self.best.loss if self.best else m.loss, "note": note,
                           "leader": self.best.patch if self.best else patch})
        return m

    def better(self, cand: Measured, ref: Measured, z: float = 0.8) -> Optional[bool]:
        """True / False when the comparison is clear, None when more measurements are needed."""
        margin = z * self.noise * math.sqrt(1.0 / cand.n + 1.0 / ref.n)
        diff = cand.loss - ref.loss
        if diff < -margin:
            return True
        if diff > margin:
            return False
        return None

    def challenge(self, patch: Dict[str, int], note: str, max_repeats: int = 3) -> bool:
        """Measure a candidate against the current best; adopt it when it is better."""
        cand = self.measure(patch, note)
        verdict = self.better(cand, self.best)
        while verdict is None and cand.n < max_repeats and not self.should_stop():
            # too close to call: measure whichever has been measured less
            if self.best.n <= cand.n:
                self.measure(self.best.patch, "re-check best")
            else:
                self.measure(patch, note + " (repeat)")
            verdict = self.better(cand, self.best)
        if verdict is None:
            verdict = cand.loss < self.best.loss - 0.25 * self.noise
        if verdict:
            self.best = cand
        return bool(verdict)

    # -- the corrected model ------------------------------------------------------
    def _correction(self, m: Measured) -> Tuple[Aligned, int]:
        """What has to be added to the model's prediction at m.patch to get the measurement."""
        self.model.set_precision(fine=True)
        feats = self.model.features(m.patch)
        shift = self.target.compare(feats)["shift"]
        model_al = self.target.aligned(feats, shift)
        hw_al = self.target.aligned(m.feats)
        return hw_al.minus(model_al), shift

    def _corrected_objective(self, corr: Aligned, shift: int, centre: Dict[str, int], radius: float) -> Objective:
        keys = [d for d in continuous_dims(structure_of_patch(centre))]

        def penalty(patch: Dict[str, int]) -> float:
            # gentle preference for staying close to the measured point
            d = 0.0
            for dim in keys:
                span = float(dim.hi - dim.lo)
                d += ((patch[dim.key] - centre[dim.key]) / span) ** 2
            return complexity_penalty(patch) + 0.01 * d / max(radius * radius, 1e-6) / max(len(keys), 1)

        def batch(patches: List[Dict[str, int]]) -> List[float]:
            return self.model.map(
                lambda p: self.target.loss_aligned(self.target.aligned(self.model.features(p), shift), corr), patches)

        return Objective(batch, penalty)

    def _trust_bounds(self, centre: Dict[str, int], radius: float) -> Dict[str, Tuple[int, int]]:
        bounds = {}
        for dim in continuous_dims(structure_of_patch(centre)):
            span = dim.hi - dim.lo
            r = max(1, int(round(radius * span)))
            bounds[dim.key] = (max(dim.lo, centre[dim.key] - r), min(dim.hi, centre[dim.key] + r))
        return bounds

    def propose(self, radius: float, gens: int = 10) -> Tuple[Dict[str, int], float, float]:
        """Best patch of the corrected model inside the trust region around the current best."""
        centre = self.best.patch
        corr, shift = self._correction(self.best)
        obj = self._corrected_objective(corr, shift, centre, radius)
        opt = PatchOptimizer(obj, int(self.rng.integers(1 << 30)), None, self.should_stop, frozen=self.frozen)
        base = obj.one(centre)
        keys = opt.active_keys(centre)
        patch, loss = opt.polish(centre, base, keys, 0.35, gens, lam=12,
                                 bounds=self._trust_bounds(centre, radius))
        return patch, base, loss

    def propose_structure(self) -> Optional[Tuple[Dict[str, int], float]]:
        """Let the corrected model try the switches; returns (patch, predicted gain) or None."""
        centre = self.best.patch
        corr, shift = self._correction(self.best)
        obj = self._corrected_objective(corr, shift, centre, 0.3)
        opt = PatchOptimizer(obj, int(self.rng.integers(1 << 30)), None, self.should_stop, frozen=self.frozen)
        base = obj.one(centre)
        patch, loss = centre, base
        for block in (opt.block_osc1, opt.block_sub, opt.block_osc2):
            patch, loss = block(patch, loss, 0.3)
        patch, loss = opt.block_filter(patch, loss, 0.3)
        patch, loss = opt.block_extras(patch, loss, 0.3)
        if structure_key(structure_of_patch(patch)) == structure_key(structure_of_patch(centre)) and \
                all(patch[k] == centre[k] for k in ("filter_drive", "distortion", "mix_noise", "filter_track")):
            return None
        return clean_patch(patch), base - loss

    # -- direct search --------------------------------------------------------------
    def direct_search(self, budget: int) -> None:
        """Nudge single controls; keep what measurably helps."""
        self.phase = "fine tuning"
        start_plays = self.plays
        dims = [d for d in continuous_dims(structure_of_patch(self.best.patch)) if d.key not in self.frozen]
        steps = {d.key: max(1.0, 0.04 * (d.hi - d.lo)) for d in dims}
        # most influential controls first, according to the model
        self.model.set_precision(fine=True)
        centre = self.best.patch
        probes = []
        for d in dims:
            for sign in (-1, 1):
                v = int(min(max(centre[d.key] + sign * steps[d.key], d.lo), d.hi))
                probes.append(dict(centre, **{d.key: v}))
        base_feats = self.model.features(centre)
        base_al = self.target.aligned(base_feats)
        sens = {}
        vals = self.model.map(lambda p: self._model_distance(p, base_al), probes)
        for i, d in enumerate(dims):
            sens[d.key] = max(vals[2 * i], vals[2 * i + 1])
        order = sorted(dims, key=lambda d: -sens[d.key])
        direction = {d.key: 1 for d in dims}
        while self.plays - start_plays < budget and not self.should_stop():
            progress_made = False
            for d in order:
                if self.plays - start_plays >= budget or self.should_stop():
                    break
                if sens[d.key] < 0.002 or steps[d.key] < 0.75:
                    continue
                cur = self.best.patch[d.key]
                moved = False
                for sign in (direction[d.key], -direction[d.key]):
                    v = int(min(max(round(cur + sign * steps[d.key]), d.lo), d.hi))
                    if v == cur:
                        continue
                    if self.challenge(dict(self.best.patch, **{d.key: v}), f"{PARAMS[d.key].label} {v}", 2):
                        direction[d.key] = sign
                        moved = True
                        progress_made = True
                        break
                    if self.plays - start_plays >= budget:
                        break
                if not moved:
                    steps[d.key] *= 0.5
            if not progress_made and all(steps[d.key] < 0.75 or sens[d.key] < 0.002 for d in order):
                break

    def _model_distance(self, patch: Dict[str, int], base: Aligned) -> float:
        """How far the model's sound moves when the patch changes (mean dB / 10)."""
        al = self.target.aligned(self.model.features(patch), base.shift)
        return float(np.mean(np.abs(al.a - base.a)) + np.mean(np.abs(al.b - base.b))) / 20.0

    # -- the loop -----------------------------------------------------------------
    def run(self, candidates: List[Dict[str, int]], budget: int = 60) -> RefineResult:
        """Refine starting from the model's candidates; ``budget`` is the number of plays."""
        t0 = time.time()
        self.phase = "candidates"
        firsts = []
        for i, cand in enumerate(candidates[:4]):
            if self.should_stop() and firsts:
                break
            firsts.append(self.measure(cand, f"candidate {i + 1}: {describe_structure(cand)}"))
            if self.best is None or firsts[-1].loss < self.best.loss:
                self.best = firsts[-1]
        start_loss = self.best.loss
        self.measure(self.best.patch, "re-check best")
        # a second opinion on the runner-up if it is close
        firsts.sort(key=lambda m: m.loss)
        for m in firsts[:2]:
            if m is not self.best and m.loss < self.best.loss + 2.0 * self.noise:
                self.measure(m.patch, "re-check runner-up")
                if m.loss < self.best.loss:
                    self.best = m

        radius = 0.12
        fails = 0
        iteration = 0
        structure_tried = set()
        self.phase = "model-guided"
        reserve = max(6, int(0.3 * budget))  # plays kept for the final direct search
        while self.plays < budget - reserve and not self.should_stop():
            iteration += 1
            patch, base, predicted = self.propose(radius)
            gain = base - predicted
            if Objective.key(patch) == Objective.key(self.best.patch) or gain < 0.2 * self.noise:
                fails += 1
                radius *= 0.6
            else:
                if self.challenge(patch, f"model step (radius {radius:.2f}, predicted -{gain:.3f})"):
                    radius = min(0.3, radius * 1.4)
                    fails = 0
                else:
                    radius *= 0.55
                    fails += 1
            if iteration in (2, 6) or (fails >= 2 and iteration > 2 and len(structure_tried) < 3):
                key = structure_key(structure_of_patch(self.best.patch))
                if (key, iteration // 4) not in structure_tried and not self.should_stop():
                    structure_tried.add((key, iteration // 4))
                    self.phase = "structure check"
                    prop = self.propose_structure()
                    if prop is not None and prop[1] > max(0.01, self.noise):
                        if self.challenge(prop[0], f"switches: {describe_structure(prop[0])}"):
                            radius = 0.12
                            fails = 0
                    self.phase = "model-guided"
            if radius < 0.012 or fails >= 4:
                break
        if not self.should_stop():
            self.direct_search(max(0, budget - self.plays))
        if self.best.n < 2 and not self.should_stop():
            self.measure(self.best.patch, "final check")
        self.dev.set_patch(self.best.patch)
        return RefineResult(dict(self.best.patch), self.best.loss, dict(self.best.parts or {}),
                            self.best.audio, self.plays, list(self.history), start_loss, self.noise)
