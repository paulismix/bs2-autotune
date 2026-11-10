"""The complete workflow on the virtual instrument (slow: run with --runslow)."""

import json
import os

import numpy as np
import pytest

from bs2autotune.app import App
from bs2autotune.audio_util import write_wav
from bs2autotune.calibration import Calibrator
from bs2autotune.device import VirtualDevice
from bs2autotune.features import Target, extract
from bs2autotune.model import Model
from bs2autotune.params import PARAMS, normalize_patch
from bs2autotune.refine import HardwareLoop
from bs2autotune.search import ModelSearch, describe_structure

SQUARE = dict(osc1_wave=3, osc1_pw=64, filter_type=1, filter_freq=110, filter_res=80, filter_env=95, mod_decay=45,
              mod_sustain=0, amp_decay=90, amp_sustain=100, amp_release=15, distortion=40)


@pytest.mark.slow
def test_matching_a_patch_of_the_same_instrument(phrase):
    """Reference = the instrument itself playing a hidden patch.  The match must come close to
    what two takes of that very patch differ by, and must beat the model's first guess."""
    dev = VirtualDevice(seed=5)
    physics = Calibrator(dev).run().physics
    true = normalize_patch(SQUARE)
    notes = [(n.start, n.end, n.note) for n in phrase.notes]
    target = Target(dev.capture(true, phrase), notes=notes)
    floor = float(np.mean([target.loss(extract(dev.capture(true, phrase, target.n_samples))) for _ in range(3)]))
    model = Model(physics, phrase, target, workers=4)
    found = ModelSearch(model, seed=1).run(effort=0.7)
    assert found.candidates and found.best.loss < 0.2
    loop = HardwareLoop(dev, model, target, phrase, seed=2)
    out = loop.run([c.patch for c in found.candidates], budget=40)
    model.close()
    assert out.plays <= 46
    assert out.loss <= out.start_loss + 1e-9
    assert out.loss < max(0.11, floor + 0.07), (out.loss, floor, describe_structure(out.patch))
    assert set(out.patch) == set(PARAMS)
    assert out.patch["osc1_wave"] == 3 and out.patch["filter_shape"] == 0  # a pulse through a low-pass


@pytest.mark.slow
def test_application_workflow_in_demo_mode(tmp_path, line):
    """Upload -> analyse -> match -> result files -> library, through the application layer."""
    other = VirtualDevice(seed=11, variant=1)  # the reference comes from a different "unit"
    other.latency = other.true_latency
    y = other.capture(normalize_patch(dict(osc1_range=65, mix_sub=200, sub_wave=2, filter_freq=115, filter_res=45,
                                           filter_env=105, mod_decay=55, mod_sustain=20, amp_decay=85,
                                           amp_sustain=70, amp_release=25)), line)
    wav = tmp_path / "bass.wav"
    write_wav(str(wav), y / np.abs(y).max() * 0.7, 32000)
    app = App(str(tmp_path / "data"))
    app.update_settings({"mode": "demo"})
    info = app.set_reference_file("bass.wav", wav.read_bytes())
    assert info["channels"] == 1
    app.job_analyse({"mode": "stem"})
    app.wait(120)
    assert app.job.status == "done", app.job.error
    app.job_match({"effort": "quick"})
    app.wait(900)
    assert app.job.status == "done", (app.job.error, list(app.job.log)[-3:])
    result = app.result_summary()
    assert result["loss"] < 0.25 and result["plays"] >= 10 and result["rows"]
    folder = os.path.join(app.data_dir, "sessions", result["id"])
    assert {"patch.txt", "reference.wav", "synth.wav", "reference.png", "synth.png", "result.json"} <= set(os.listdir(folder))
    assert json.load(open(os.path.join(folder, "result.json")))["patch"] == result["patch"]
    assert app.calibration["demo"] is not None  # calibrated automatically on the first match
    saved = app.library_save("Demo bass")
    assert [i["id"] for i in app.library()] == [saved["id"]]
    assert app.library_get(saved["id"])["patch"] == result["patch"]
    # a fresh start shows the last result again
    assert App(app.data_dir).result["id"] == result["id"]
