"""Calibration against the virtual instrument, whose physics the app does not know."""

import numpy as np
import pytest

from bs2autotune.calibration import Calibrator
from bs2autotune.device import VirtualDevice
from bs2autotune.engine import Physics
from bs2autotune.features import Target, extract
from bs2autotune.model import Model
from bs2autotune.params import normalize_patch


@pytest.fixture(scope="module")
def calibrated():
    dev = VirtualDevice(seed=5)
    result = Calibrator(dev).run()
    return dev, result


def test_measurements_recover_the_hidden_response(calibrated):
    dev, result = calibrated
    true, ph = dev.true_physics, result.physics
    assert result.warnings == []
    assert abs(result.latency - dev.true_latency) < 0.003
    assert abs(ph.tune_cents - true.tune_cents) < 1.5
    # the cutoff table is defined at C3 (note 48)
    shift = true.track_full * (48 - true.track_ref_note) / 12.0
    for raw in (60, 100, 140, 180):
        assert abs(ph.cutoff_classic[raw] - (true.cutoff_classic[raw] + shift)) < 0.12, raw
    for raw in (30, 64, 100):
        assert abs(np.log2(ph.decay_table[raw] / true.decay_table[raw])) < 0.1, raw
        assert abs(np.log2(ph.release_table[raw] / true.release_table[raw])) < 0.1, raw
    assert abs(np.log2(ph.attack_table[64] / true.attack_table[64])) < 0.25
    assert abs(ph.env_oct_per_unit - true.env_oct_per_unit) < 0.012
    assert abs(ph.track_full - true.track_full) < 0.06
    assert abs(ph.sustain_gamma - true.sustain_gamma) < 0.1
    assert abs(ph.mix_table[128] - true.mix_table[128]) < 0.05
    for est, real in zip(ph.wave_gain, true.wave_gain):
        assert abs(est - real) < 0.1
    assert ph.calibrated.get("waveform_control") is True and ph.calibrated.get("filter_track_control") is True
    assert Physics.from_dict(ph.to_dict()).cutoff_classic[100] == pytest.approx(ph.cutoff_classic[100])


def test_calibrated_model_sounds_like_the_instrument(calibrated, phrase):
    """The same patch on the instrument and in the calibrated model: nearly as close as two takes."""
    dev, result = calibrated
    notes = [(n.start, n.end, n.note) for n in phrase.notes]
    for patch in (dict(amp_release=20),
                  dict(filter_freq=120, filter_res=70, amp_release=20),
                  dict(osc1_range=65, mix_sub=200, sub_wave=2, filter_freq=95, filter_res=45, filter_env=105,
                       mod_decay=55, mod_sustain=20, amp_decay=85, amp_sustain=70, amp_release=25)):
        patch = normalize_patch(patch)
        target = Target(dev.capture(patch, phrase), notes=notes)
        again = target.loss(extract(dev.capture(patch, phrase, target.n_samples)))
        calibrated_model = Model(result.physics, phrase, target, workers=1)
        calibrated_model.set_precision(True)
        default_model = Model(Physics(), phrase, target, workers=1)
        default_model.set_precision(True)
        assert calibrated_model.loss(patch) < max(0.09, again + 0.05)
        assert calibrated_model.loss(patch) < 0.6 * default_model.loss(patch)


def test_missing_nrpn_support_is_noticed():
    dev = VirtualDevice(seed=6, dead_params=["osc1_wave", "osc2_wave"])
    cal = Calibrator(dev)
    cal.step_signal()
    cal.step_nrpn()
    assert cal.ph.calibrated["waveform_control"] is False
    assert any("waveform" in w.lower() for w in cal.result.warnings)
