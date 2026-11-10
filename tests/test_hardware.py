"""The hardware layer, driven through the simulated rig (virtual time)."""

import numpy as np
import pytest

from bs2autotune.calibration import Calibrator
from bs2autotune.device import DeviceError, apply_tuning
from bs2autotune.engine import Note, Phrase
from bs2autotune.hardware import HardwareDevice
from bs2autotune.params import baseline_patch, normalize_patch
from tests.fakes import FakeRig


def make(latency=0.0123, channel=1, **kw):
    rig = FakeRig(latency=latency, channel=channel, **kw)
    return rig, HardwareDevice(rig, rig, midi_channel=channel, clock=rig.clock)


def test_recording_is_cut_at_the_moment_the_phrase_starts():
    rig, dev = make(latency=0.0123)
    dev.set_patch(normalize_patch({}), force=True)
    y = dev.play(Phrase([Note(0.05, 0.25, 57), Note(0.3, 0.5, 60)], 0.7), extra=0.1)
    assert len(y) == int(round(0.8 * dev.sr))
    env = np.abs(y)
    onset = np.nonzero(env > 0.2 * env.max())[0][0] / dev.sr
    assert abs(onset - (0.05 + 0.0123)) < 0.002  # note time + rig latency, despite the foreign audio clock
    assert rig.wrong_channel == 0
    dev.close()


def test_patch_changes_reach_the_synth_and_only_changes_are_sent():
    rig, dev = make()
    patch = normalize_patch(dict(osc1_wave=3, filter_freq=201, filter_res=77, filter_track=5, mix_sub=133))
    dev.set_patch(patch, force=True)
    assert rig.decoder.patch == patch
    before = len(rig.sent)
    dev.set_patch(patch)
    assert len(rig.sent) == before  # nothing changed, nothing sent
    dev.set_patch(dict(patch, filter_freq=90))
    assert len(rig.sent) == before + 2 and rig.decoder.patch["filter_freq"] == 90
    dev.close()


def test_midi_channel_is_respected():
    rig = FakeRig(channel=5)
    dev = HardwareDevice(rig, rig, midi_channel=3, clock=rig.clock)  # the synth listens on 5, we talk on 3
    dev.set_patch(dict(baseline_patch(), filter_freq=100), force=True)
    assert rig.wrong_channel > 0 and rig.decoder.patch["filter_freq"] == 255
    with pytest.raises(DeviceError):
        Calibrator(dev).step_signal()  # silence -> a clear error instead of a wrong calibration
    dev.close()


def test_sysex_dump_request():
    rig, dev = make()
    dump = dev.dump_patch()
    assert dump is not None and len(dump) == 154 and dump[0] == 0xF0 and dump[-1] == 0xF7
    assert rig.sent[-1] == [0xF0, 0x00, 0x20, 0x29, 0x00, 0x33, 0x00, 0x40, 0xF7]
    silent_rig, silent = make(respond_to_dump=False)
    assert silent.dump_patch() is None
    dev.close()
    silent.close()


def test_the_only_sysex_ever_sent_is_the_dump_request():
    # nothing the app does may write a patch slot or a bank on the synth
    rig, dev = make()
    Calibrator(dev).run(quick=True)
    dev.reset_controllers()
    dev.set_patch(dict(baseline_patch(), filter_freq=90, osc1_wave=3, paraphonic=0), force=True)
    dev.capture(baseline_patch(), Phrase([Note(0.05, 0.3, 36)], 0.6))
    dev.dump_patch()
    dev.close()
    sysex = [m for m in rig.sent if m and m[0] == 0xF0]
    assert sysex == [[0xF0, 0x00, 0x20, 0x29, 0x00, 0x33, 0x00, 0x40, 0xF7]]
    assert not hasattr(dev, "send_sysex")
    # everything else is a note, a controller or the pitch wheel on the chosen channel
    assert {m[0] & 0xF0 for m in rig.sent if m[0] != 0xF0} <= {0x80, 0x90, 0xB0, 0xE0}
    assert not any(m[0] & 0xF0 == 0xC0 for m in rig.sent)  # and never a program change


def test_signal_check_measures_latency_level_and_tuning():
    rig, dev = make(latency=0.021)
    result = Calibrator(dev).run(quick=True)
    assert abs(result.latency - (0.021 - 0.0015)) < 0.002
    assert -30 < result.peak_db < -6 and result.noise_db < -60
    assert abs(result.tune_cents - rig.synth.true_physics.tune_cents) < 1.5
    assert result.physics.calibrated.get("waveform_control") is True
    dev.close()


def test_tuning_offset_moves_both_oscillators():
    patch = normalize_patch(dict(osc2_fine=140))
    tuned = apply_tuning(patch, 15)
    assert tuned["osc1_fine"] == 143 and tuned["osc2_fine"] == 155
    assert apply_tuning(patch, 0) is patch
    assert apply_tuning(patch, 500)["osc1_fine"] == 228  # clamped to the range of the control
