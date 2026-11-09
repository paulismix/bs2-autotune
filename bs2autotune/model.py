"""Evaluating patches with the software model."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

from . import engine
from .engine import Phrase, Physics
from .features import ANALYSIS_SR, Aligned, Features, Target, extract

# oscillator start phases of the renders that are averaged when a patch has
# two free running sound sources (osc 1 + osc 2): relative phase 0, 1/3, 2/3
_PHASES = ((0.13, 0.24), (0.47, 0.913), (0.81, 0.587), (0.29, 0.79))


def two_sources(patch: Dict[str, int]) -> bool:
    """True when the relative phase of the two oscillators matters."""
    o2 = patch.get("mix_osc2", 0) > 0
    o1 = patch.get("mix_osc1", 0) > 0
    return (o1 and o2) or patch.get("mix_ring", 0) > 0


class Model:
    """The software model of the synth playing one phrase, compared to one target."""

    def __init__(self, physics: Physics, phrase: Phrase, target: Optional[Target] = None,
                 sr: int = ANALYSIS_SR, realizations: int = 3, workers: Optional[int] = None):
        self.physics = physics
        self.phrase = phrase
        self.target = target
        self.sr = sr
        self.realizations = max(1, min(realizations, len(_PHASES)))
        self.n_samples = target.n_samples if target is not None else int(round(phrase.duration * sr))
        self._on, self._off, self._note = phrase.arrays(float(sr))
        self.workers = workers or max(1, min(8, (os.cpu_count() or 2)))
        self._pool: Optional[ThreadPoolExecutor] = None
        self.evaluations = 0

    # -- rendering -----------------------------------------------------------
    def render(self, patch: Dict[str, int], realization: int = 0, seed: int = 1) -> np.ndarray:
        ph1, ph2 = _PHASES[realization % len(_PHASES)]
        P = engine.resolve(patch, self.physics, float(self.sr), seed=seed + realization, phase1=ph1, phase2=ph2)
        out = np.zeros(self.n_samples, dtype=np.float32)
        engine._render_kernel(out, self._on, self._off, self._note, P)
        return out

    def features(self, patch: Dict[str, int], realizations: Optional[int] = None) -> List[Features]:
        k = self.realizations if realizations is None else realizations
        if not two_sources(patch):
            k = 1
        cfg = self.target.cfg if self.target is not None else None
        out = []
        for r in range(k):
            y = self.render(patch, r)
            out.append(extract(y, cfg) if cfg is not None else extract(y))
        self.evaluations += 1
        return out

    # -- comparisons -----------------------------------------------------------
    def compare(self, patch: Dict[str, int]) -> dict:
        return self.target.compare(self.features(patch))

    def loss(self, patch: Dict[str, int]) -> float:
        return self.target.compare(self.features(patch))["loss"]

    def aligned(self, patch: Dict[str, int], shift: Optional[int] = None) -> Aligned:
        return self.target.aligned(self.features(patch), shift)

    # -- parallel helpers --------------------------------------------------------
    def map(self, fn: Callable, items: Sequence) -> list:
        items = list(items)
        if len(items) <= 1 or self.workers <= 1:
            return [fn(it) for it in items]
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="bs2model")
        return list(self._pool.map(fn, items))

    def loss_many(self, patches: Iterable[Dict[str, int]]) -> np.ndarray:
        return np.asarray(self.map(self.loss, list(patches)), dtype=np.float64)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None
