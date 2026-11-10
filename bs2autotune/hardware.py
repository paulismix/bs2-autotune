"""The real instrument: MIDI out to the Bass Station II, audio in from the interface.

Timing matters here.  Notes are sent from this thread at precise moments while
the audio interface delivers blocks on its own clock; every block carries the
time at which its first sample was captured, so the recording can be cut at
exactly the moment the phrase was started.  What remains is a constant delay
(interface + synth), which the calibration measures once.

The low-level I/O sits behind two small backends (``MidiBackend``,
``AudioBackend``) so that the same code can be exercised without hardware.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .device import Device, DeviceError
from .engine import Phrase
from .params import PARAMS, PARAM_LIST, SYSEX_REQUEST_CURRENT, fw_tuple, is_patch_dump

# ---------------------------------------------------------------------------
# backends
# ---------------------------------------------------------------------------


class Clock:
    """Wall clock used for scheduling (replaceable in tests)."""

    def now(self) -> float:
        return time.perf_counter()

    def sleep_until(self, t: float) -> None:
        while True:
            remaining = t - time.perf_counter()
            if remaining <= 0:
                return
            if remaining > 0.003:
                time.sleep(remaining - 0.002)
            # the last two milliseconds: spin for accuracy


class MidiBackend:
    def send(self, data: Sequence[int]) -> None:
        raise NotImplementedError

    def receive_sysex(self, timeout: float) -> Optional[bytes]:
        return None

    def close(self) -> None:
        pass


class AudioBackend:
    sr = 48000

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def now(self) -> float:
        """Current time on the clock that stamps the audio blocks."""
        raise NotImplementedError

    def set_sink(self, sink: Optional[Callable[[np.ndarray, float], None]]) -> None:
        """sink(block, time_of_first_sample) is called for every captured block (mono float32)."""
        raise NotImplementedError


class MidoBackend(MidiBackend):
    """MIDI through mido / python-rtmidi (CoreMIDI on macOS)."""

    def __init__(self, out_name: str, in_name: Optional[str] = None):
        try:
            import mido
        except Exception as exc:  # pragma: no cover - depends on the installation
            raise DeviceError(f"The MIDI library is not installed correctly ({exc}).") from exc
        self._mido = mido
        try:
            self.out = mido.open_output(out_name)
        except Exception as exc:
            raise DeviceError(f"Cannot open the MIDI output '{out_name}': {exc}") from exc
        self.inp = None
        if in_name:
            try:
                self.inp = mido.open_input(in_name)
            except Exception:
                self.inp = None  # dumps will not be available, everything else works

    def send(self, data: Sequence[int]) -> None:
        self.out.send(self._mido.Message.from_bytes(list(data)))

    def receive_sysex(self, timeout: float) -> Optional[bytes]:
        if self.inp is None:
            return None
        end = time.time() + timeout
        while time.time() < end:
            for msg in self.inp.iter_pending():
                if msg.type == "sysex":
                    return bytes([0xF0] + list(msg.data) + [0xF7])
            time.sleep(0.01)
        return None

    def drain(self) -> None:
        if self.inp is not None:
            for _ in self.inp.iter_pending():
                pass

    def close(self) -> None:
        for port in (self.out, self.inp):
            try:
                if port is not None:
                    port.close()
            except Exception:
                pass


class SoundDeviceBackend(AudioBackend):
    """Audio input through sounddevice / PortAudio (CoreAudio on macOS)."""

    def __init__(self, device, channel: int = 1, samplerate: Optional[int] = None):
        try:
            import sounddevice as sd
        except Exception as exc:  # pragma: no cover - depends on the installation
            raise DeviceError(f"The audio library is not available ({exc}).") from exc
        self._sd = sd
        try:
            info = sd.query_devices(device, "input")
        except Exception as exc:
            raise DeviceError(f"Audio input '{device}' not found: {exc}") from exc
        self.device = info["index"] if "index" in info else device
        self.n_channels = int(info["max_input_channels"])
        if self.n_channels < 1:
            raise DeviceError(f"'{info['name']}' has no inputs.")
        self.channel = int(min(max(channel, 1), self.n_channels))
        self.sr = int(samplerate or info["default_samplerate"] or 48000)
        self._sink: Optional[Callable[[np.ndarray, float], None]] = None
        self._stream = None
        self.status_flags: List[str] = []
        self._fallback_t0: Optional[float] = None
        self._frames = 0

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            self.status_flags.append(str(status))
        sink = self._sink
        adc = float(time_info.inputBufferAdcTime)
        if adc <= 0.0:  # host does not report capture times: count samples instead
            if self._fallback_t0 is None:
                self._fallback_t0 = self.now() - frames / self.sr
            adc = self._fallback_t0 + self._frames / self.sr
        self._frames += frames
        if sink is not None:
            sink(np.array(indata[:, self.channel - 1], dtype=np.float32), adc)

    def start(self) -> None:
        if self._stream is not None:
            return
        sd = self._sd
        try:
            self._stream = sd.InputStream(device=self.device, channels=self.n_channels, samplerate=self.sr,
                                          dtype="float32", callback=self._callback, latency="low")
            self._stream.start()
        except Exception as exc:
            self._stream = None
            raise DeviceError(f"Cannot open the audio input: {exc}") from exc

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def now(self) -> float:
        return float(self._stream.time) if self._stream is not None else time.perf_counter()

    def set_sink(self, sink) -> None:
        self._sink = sink


def list_midi_ports() -> Dict[str, List[str]]:
    try:
        import mido
        return {"outputs": list(dict.fromkeys(mido.get_output_names())),
                "inputs": list(dict.fromkeys(mido.get_input_names())), "error": ""}
    except Exception as exc:
        return {"outputs": [], "inputs": [], "error": str(exc)}


def list_audio_inputs() -> Dict[str, object]:
    try:
        import sounddevice as sd
        devices = []
        default_in = sd.default.device[0] if sd.default.device else None
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0:
                devices.append({"index": i, "name": d["name"], "channels": int(d["max_input_channels"]),
                                "samplerate": int(d["default_samplerate"]), "default": i == default_in})
        return {"devices": devices, "error": ""}
    except Exception as exc:
        return {"devices": [], "error": str(exc)}


def guess_bs2_port(names: Sequence[str]) -> Optional[str]:
    for name in names:
        if "bass station" in name.lower():
            return name
    return None


# ---------------------------------------------------------------------------
# the device
# ---------------------------------------------------------------------------


class HardwareDevice(Device):
    name = "Bass Station II"

    def __init__(self, midi: MidiBackend, audio: AudioBackend, midi_channel: int = 1, clock: Optional[Clock] = None,
                 max_firmware: Optional[str] = None, settle: float = 0.06):
        super().__init__()
        self.midi = midi
        self.audio = audio
        self.sr = audio.sr
        self.channel = int(min(max(midi_channel, 1), 16)) - 1
        self.clock = clock or Clock()
        self.max_firmware = max_firmware
        self.settle = settle
        self._sent: Dict[str, int] = {}
        self._lock = threading.Lock()
        self._blocks: List[Tuple[float, np.ndarray]] = []
        self._capturing = False
        self._last_change = 0.0
        self.audio.set_sink(self._on_block)
        self.audio.start()
        self.last_peak = 0.0

    # -- MIDI ---------------------------------------------------------------
    def _cc(self, cc: int, value: int) -> None:
        self.midi.send([0xB0 | self.channel, cc & 0x7F, value & 0x7F])

    def _known(self, key: str) -> bool:
        fw = PARAMS[key].fw
        return not (self.max_firmware and fw and fw_tuple(fw) > fw_tuple(self.max_firmware))

    def set_patch(self, patch: Dict[str, int], force: bool = False) -> None:
        with self._lock:
            changed = 0
            for p in PARAM_LIST:
                if p.key not in patch or not self._known(p.key):
                    continue
                value = p.clamp(patch[p.key])
                if not force and self._sent.get(p.key) == value:
                    continue
                for cc, val in p.messages(value):
                    self._cc(cc, val)
                    self.clock.sleep_until(self.clock.now() + 0.0012)  # do not flood the synth
                self._sent[p.key] = value
                changed += 1
            if changed:
                self._last_change = self.clock.now()

    def reset_controllers(self) -> None:
        """Wheels centred, sustain off, no hanging notes."""
        with self._lock:
            self._cc(1, 0)
            self._cc(64, 0)
            self.midi.send([0xE0 | self.channel, 0x00, 0x40])
            self._cc(123, 0)

    def note_on(self, note: int, velocity: int = 100) -> None:
        self.midi.send([0x90 | self.channel, int(note) & 0x7F, int(velocity) & 0x7F])

    def note_off(self, note: int) -> None:
        self.midi.send([0x80 | self.channel, int(note) & 0x7F, 0])

    def all_notes_off(self) -> None:
        try:
            self._cc(123, 0)
        except Exception:
            pass

    def dump_patch(self) -> Optional[bytes]:
        """Ask the synth for its current patch (SysEx); None when it does not answer."""
        with self._lock:
            drain = getattr(self.midi, "drain", None)
            if drain:
                drain()
            self.midi.send(list(SYSEX_REQUEST_CURRENT))
            data = self.midi.receive_sysex(1.5)
        return data if data is not None and is_patch_dump(data) else None

    # -- audio ---------------------------------------------------------------
    def _on_block(self, block: np.ndarray, t_first: float) -> None:
        if self._capturing:
            self._blocks.append((t_first, block))

    def play(self, phrase: Phrase, extra: float = 0.0) -> np.ndarray:
        self.check_cancel()
        self.plays += 1
        clock = self.clock
        # give the synth a moment to settle after parameter changes
        clock.sleep_until(self._last_change + self.settle)
        events: List[Tuple[float, int, int, int]] = []
        for n in phrase.notes:
            events.append((n.start, 1, n.note, n.velocity))
            events.append((n.end, 0, n.note, 0))
        events.sort(key=lambda e: (e[0], e[1]))  # at the same moment: off before on
        total = phrase.duration + extra
        self._blocks = []
        self._capturing = True
        try:
            lead = 0.03
            t0 = clock.now() + lead
            offset = self.audio.now() - clock.now()  # audio clock = wall clock + offset
            sounding: Optional[int] = None
            for when, kind, note, vel in events:
                self._wait(t0 + when)
                if kind:
                    self.note_on(note, vel)
                    sounding = note
                else:
                    self.note_off(note)
                    if sounding == note:
                        sounding = None
            self._wait(t0 + total + 0.03)
        finally:
            self._capturing = False
            self.all_notes_off()
        blocks = self._blocks
        self._blocks = []
        n = int(round(total * self.sr))
        out = np.zeros(n, dtype=np.float32)
        if not blocks:
            raise DeviceError("The audio interface delivered no samples. Is the input still connected?")
        data = np.concatenate([b for _, b in blocks])
        first = int(round((t0 + offset - blocks[0][0]) * self.sr))  # sample index of phrase time zero
        a = max(first, 0)
        seg = data[a:a + n - max(-first, 0)]
        start = max(-first, 0)
        out[start:start + len(seg)] = seg
        self.last_peak = float(np.max(np.abs(out))) if len(out) else 0.0
        return out

    def _wait(self, t: float) -> None:
        """Sleep until t, staying responsive to a stop request."""
        clock = self.clock
        while t - clock.now() > 0.1:
            self.check_cancel()
            clock.sleep_until(clock.now() + 0.05)
        clock.sleep_until(t)

    def input_level(self, seconds: float = 0.3) -> Tuple[float, float]:
        """Peak and RMS of the input right now (for the level meter)."""
        self._blocks = []
        self._capturing = True
        try:
            self.clock.sleep_until(self.clock.now() + seconds)
        finally:
            self._capturing = False
        blocks = self._blocks
        self._blocks = []
        if not blocks:
            return 0.0, 0.0
        data = np.concatenate([b for _, b in blocks])
        return float(np.max(np.abs(data))), float(np.sqrt(np.mean(np.square(data, dtype=np.float64))))

    def close(self) -> None:
        try:
            self.all_notes_off()
        finally:
            self.audio.set_sink(None)
            self.audio.stop()
            self.midi.close()


def open_hardware(midi_out: str, audio_device, input_channel: int = 1, samplerate: Optional[int] = None,
                  midi_channel: int = 1, midi_in: Optional[str] = None,
                  max_firmware: Optional[str] = None) -> HardwareDevice:
    """Open the MIDI and audio connections and return the device."""
    midi = MidoBackend(midi_out, midi_in)
    try:
        audio = SoundDeviceBackend(audio_device, input_channel, samplerate)
        return HardwareDevice(midi, audio, midi_channel, max_firmware=max_firmware)
    except Exception:
        midi.close()
        raise
