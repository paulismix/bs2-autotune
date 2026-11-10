"""The MIDI backend on the real mido / python-rtmidi stack.

A stand-in synth with two *virtual* MIDI ports takes the place of the Bass
Station II; the app side opens them by name with the classes it uses for the
real instrument.  Runs wherever the system offers virtual MIDI ports (macOS
CoreMIDI, Linux with ALSA sequencer or JACK) and is skipped elsewhere.
"""

import os
import threading
import time

import numpy as np
import pytest

from bs2autotune.engine import Note, Phrase
from bs2autotune.hardware import AudioBackend, HardwareDevice, MidoBackend, list_midi_ports
from bs2autotune.params import SYSEX_HEADER, SYSEX_REQUEST_CURRENT, normalize_patch
from tests.fakes import MidiDecoder

DUMP = bytes(list(SYSEX_HEADER) + [0x00] + list(range(1, 100)) + [0x20] * 46 + [0xF7])


class SilentAudio(AudioBackend):
    """An audio input that delivers silence in real time."""

    sr = 48000

    def __init__(self):
        self.sink = None
        self.running = False

    def start(self):
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        t = time.perf_counter()
        while self.running:
            if self.sink:
                self.sink(np.zeros(480, dtype=np.float32), t)
            t += 0.01
            time.sleep(max(0.0, t - time.perf_counter()))

    def stop(self):
        self.running = False

    def now(self):
        return time.perf_counter()

    def set_sink(self, sink):
        self.sink = sink


class StandIn:
    """Receives everything the app sends and answers the dump request like the synth."""

    def __init__(self, name):
        import mido

        self.mido = mido
        self.inp = mido.open_input(name, virtual=True)
        self.out = mido.open_output(name, virtual=True)
        self.received = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stop.is_set():
            for msg in self.inp.iter_pending():
                self.received.append(msg.bytes())
                if msg.bytes() == list(SYSEX_REQUEST_CURRENT):
                    self.out.send(self.mido.Message.from_bytes(list(DUMP)))
            time.sleep(0.0005)

    def close(self):
        self.stop.set()
        self.thread.join(1.0)
        self.inp.close()
        self.out.close()


@pytest.fixture()
def rig():
    pytest.importorskip("mido")
    pytest.importorskip("rtmidi")
    name = "BS2 Autotune test %d" % os.getpid()
    try:
        synth = StandIn(name)
    except Exception as exc:  # no MIDI system, or no virtual ports on this platform
        pytest.skip("virtual MIDI ports are not available here (%s)" % str(exc).strip().splitlines()[-1])
    time.sleep(0.3)
    ports = list_midi_ports()
    out_name = next((n for n in ports["outputs"] if name in n), None)
    in_name = next((n for n in ports["inputs"] if name in n), None)
    if not out_name or not in_name:
        synth.close()
        pytest.skip("the virtual ports are not listed (%s)" % (ports["error"] or "no error"))
    dev = HardwareDevice(MidoBackend(out_name, in_name), SilentAudio(), midi_channel=1)
    yield synth, dev
    dev.close()
    synth.close()


def test_a_complete_patch_arrives_intact(rig):
    synth, dev = rig
    patch = normalize_patch(dict(osc1_wave=3, osc1_pw=40, filter_freq=201, filter_res=77, filter_track=5,
                                 mix_sub=133, osc2_fine=141, amp_release=33, sub_fine=120, limiter=10))
    dev.set_patch(patch, force=True)
    time.sleep(0.4)
    decoder = MidiDecoder()
    for data in synth.received:
        assert data[0] == 0xB0 and len(data) == 3  # controllers on channel 1, nothing else
        decoder.control(data[1], data[2])
    assert not decoder.unknown
    assert decoder.patch == patch


def test_notes_wheels_and_the_patch_backup(rig):
    synth, dev = rig
    dev.reset_controllers()
    dev.play(Phrase([Note(0.05, 0.2, 40), Note(0.25, 0.4, 43)], 0.5), extra=0.05)
    dump = dev.dump_patch()
    time.sleep(0.2)
    notes = [m for m in synth.received if m[0] & 0xF0 in (0x80, 0x90)]
    assert notes == [[0x90, 40, 100], [0x80, 40, 0], [0x90, 43, 100], [0x80, 43, 0]]
    assert [0xE0, 0x00, 0x40] in synth.received  # pitch wheel centred
    assert [m for m in synth.received if m[0] == 0xF0] == [list(SYSEX_REQUEST_CURRENT)]
    assert dump == DUMP  # SysEx is received (and not filtered out) on the way back
