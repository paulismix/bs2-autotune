"""A simulated rig for testing the hardware layer without hardware.

``FakeRig`` plays three roles at once:

* a MIDI output: it decodes the controller / NRPN / note messages exactly the
  way the Bass Station II interprets them,
* the synth: the decoded patch and notes are rendered with the virtual
  instrument (hidden physics, colouring),
* an audio interface: the sound comes back in blocks, time-stamped on a clock
  that differs from the wall clock, after a fixed latency.

Time is virtual: ``SimClock.sleep_until`` advances it instantly, so a test
that "plays" minutes of audio runs in a fraction of that.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from bs2autotune import engine
from bs2autotune.device import VirtualDevice
from bs2autotune.engine import Note, Phrase
from bs2autotune.hardware import AudioBackend, Clock, MidiBackend
from bs2autotune.params import PARAM_LIST, SYSEX_HEADER, baseline_patch


class MidiDecoder:
    """Turns the raw controller stream back into parameter values (as the synth would)."""

    def __init__(self) -> None:
        self.patch: Dict[str, int] = baseline_patch()
        self.cc = {p.num: p for p in PARAM_LIST if p.kind == "cc"}
        self.msb = {p.num: p for p in PARAM_LIST if p.kind == "cc2"}
        self.lsb = {p.lsb: p for p in PARAM_LIST if p.kind == "cc2"}
        self.nrpn = {p.num: p for p in PARAM_LIST if p.kind == "nrpn"}
        self._msb_value: Dict[str, int] = {}
        self._nrpn: Optional[Tuple[int, int]] = None
        self._nrpn_msb = 127
        self._nrpn_lsb = 127
        self.unknown: List[Tuple[int, int]] = []

    def control(self, cc: int, value: int) -> None:
        if cc == 99:
            self._nrpn_msb = value
        elif cc == 98:
            self._nrpn_lsb = value
        elif cc == 101 or cc == 100:
            if value == 127:
                self._nrpn_msb = self._nrpn_lsb = 127
        elif cc == 6:
            if self._nrpn_msb == 0 and self._nrpn_lsb in self.nrpn:
                p = self.nrpn[self._nrpn_lsb]
                self.patch[p.key] = p.clamp(value)
        elif cc in self.msb:
            p = self.msb[cc]
            self._msb_value[p.key] = value
            self.patch[p.key] = p.clamp(value * 2)
        elif cc in self.lsb:
            p = self.lsb[cc]
            msb = self._msb_value.get(p.key, self.patch[p.key] >> 1)
            self.patch[p.key] = p.clamp(msb * 2 + (1 if value >= 64 else 0))
        elif cc in self.cc:
            p = self.cc[cc]
            self.patch[p.key] = p.clamp(value)
        elif cc not in (1, 64, 123, 38):
            self.unknown.append((cc, value))


class SimClock(Clock):
    def __init__(self, rig: "FakeRig") -> None:
        self.rig = rig
        self.t = 10.0

    def now(self) -> float:
        return self.t

    def sleep_until(self, t: float) -> None:
        if t > self.t:
            self.t = t
            self.rig.advance(t)


class FakeRig(MidiBackend, AudioBackend):
    def __init__(self, sr: int = 48000, latency: float = 0.0123, channel: int = 1, clock_offset: float = 123.456,
                 block: int = 256, seed: int = 7, respond_to_dump: bool = True, variant: int = 0):
        self.sr = sr
        self.latency = latency
        self.channel = channel - 1
        self.clock_offset = clock_offset
        self.block = block
        self.clock = SimClock(self)
        self.decoder = MidiDecoder()
        self.synth = VirtualDevice(sr=sr, seed=seed, variant=variant)
        self.rng = np.random.default_rng(seed)
        self._sink: Optional[Callable[[np.ndarray, float], None]] = None
        self._emitted = int(self.clock.t * sr)  # samples delivered so far (stream started at virtual time 0)
        self._take_origin: Optional[float] = None
        self._take_events: List[Tuple[float, int, int]] = []
        self._take_seed = (0.1, 0.6, 1)
        self._sounding: Optional[int] = None
        self._last_off = -1.0
        self.sent: List[List[int]] = []
        self.respond_to_dump = respond_to_dump
        self._sysex_reply: Optional[bytes] = None
        self.wrong_channel = 0

    # -- MIDI out ---------------------------------------------------------------
    def send(self, data: Sequence[int]) -> None:
        data = list(data)
        self.sent.append(data)
        status = data[0]
        if status == 0xF0:
            if tuple(data[:7]) == SYSEX_HEADER and data[7] == 0x40 and self.respond_to_dump:
                self._sysex_reply = bytes(list(SYSEX_HEADER) + [0] * 146 + [0xF7])
            return
        if status & 0x0F != self.channel:
            self.wrong_channel += 1
            return
        kind = status & 0xF0
        now = self.clock.t
        if kind == 0xB0:
            if data[1] == 123:
                self._note_off_all(now)
            else:
                self.decoder.control(data[1], data[2])
        elif kind == 0x90 and data[2] > 0:
            if self._take_origin is None or (self._sounding is None and now - self._last_off > 0.5):
                self._take_origin = now
                self._take_events = []
                self._take_seed = (float(self.rng.random()), float(self.rng.random()), int(self.rng.integers(1, 1 << 30)))
            self._take_events.append((now, 1, data[1]))
            self._sounding = data[1]
        elif kind == 0x80 or (kind == 0x90 and data[2] == 0):
            self._take_events.append((now, 0, data[1]))
            if self._sounding == data[1]:
                self._sounding = None
                self._last_off = now

    def _note_off_all(self, now: float) -> None:
        if self._sounding is not None:
            self._take_events.append((now, 0, self._sounding))
            self._sounding = None
            self._last_off = now

    def receive_sysex(self, timeout: float) -> Optional[bytes]:
        reply, self._sysex_reply = self._sysex_reply, None
        return reply

    # -- audio in ----------------------------------------------------------------
    def now(self) -> float:
        return self.clock.t + self.clock_offset

    def set_sink(self, sink) -> None:
        self._sink = sink

    def _take_audio(self, until: float) -> Optional[np.ndarray]:
        """Synth output from the take origin up to the virtual time ``until``."""
        if self._take_origin is None:
            return None
        origin = self._take_origin
        key = (origin, len(self._take_events), tuple(sorted(self.decoder.patch.items())))
        cached = getattr(self, "_cache", None)
        if cached is not None and cached[0] == key and cached[1] >= until:
            return cached[2][:max(0, int(round((until - origin) * self.sr)))]
        until_req, until = until, until + 0.6  # render ahead; reused until the next event
        notes: List[Note] = []
        open_note: Optional[Tuple[float, int]] = None
        for when, kind, note in self._take_events:
            if kind:
                if open_note is not None:
                    notes.append(Note(open_note[0] - origin, when - origin, open_note[1]))
                open_note = (when, note)
            elif open_note is not None and open_note[1] == note:
                notes.append(Note(open_note[0] - origin, when - origin, note))
                open_note = None
        if open_note is not None:
            notes.append(Note(open_note[0] - origin, until - origin + 10.0, open_note[1]))
        dur = until - origin
        if dur <= 0:
            return None
        phrase = Phrase(notes, dur)
        ph1, ph2, seed = self._take_seed
        P = engine.resolve(self.decoder.patch, self.synth.true_physics, float(self.sr), seed=seed, phase1=ph1,
                           phase2=ph2)
        n = int(round(dur * self.sr))
        dry = np.zeros(n, dtype=np.float32)
        on, off, note = phrase.arrays(float(self.sr))
        engine._render_kernel(dry, on, off, note, P)
        audio = (self.synth._colour(dry) * self.synth.level).astype(np.float32)
        self._cache = (key, until, audio)
        return audio[:max(0, int(round((until_req - origin) * self.sr)))]

    def advance(self, t: float) -> None:
        """Deliver the audio blocks that an interface would have delivered by time t."""
        target = int(t * self.sr) // self.block * self.block
        if target <= self._emitted:
            return
        n = target - self._emitted
        out = (self.synth.noise * self.rng.standard_normal(n)).astype(np.float32)
        take = self._take_audio(t)
        if take is not None:
            # stream sample i is synth time i / sr - latency
            first = self._emitted - int(round((self._take_origin + self.latency) * self.sr))
            a = max(first, 0)
            seg = take[a:first + n] if first + n > 0 else take[:0]
            start = max(-first, 0)
            out[start:start + len(seg)] += seg
        sink = self._sink
        for k in range(0, n, self.block):
            if sink is not None:
                sink(out[k:k + self.block], (self._emitted + k) / self.sr + self.clock_offset)
        self._emitted = target
