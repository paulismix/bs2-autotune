"""The synth model, the features and the distance."""

import numpy as np

from bs2autotune.engine import Note, Phrase, Physics, filter_mag_db, render
from bs2autotune.features import Target, extract
from bs2autotune.model import Model, two_sources
from bs2autotune.params import normalize_patch
from bs2autotune.search import clean_patch, describe_structure, structure_key, structure_of_patch

BASS = dict(osc1_wave=2, mix_sub=200, sub_wave=2, filter_freq=120, filter_res=40, filter_env=100, mod_decay=60,
            mod_sustain=30, amp_decay=80, amp_sustain=90, amp_release=20)


def notes_of(phrase):
    return [(n.start, n.end, n.note) for n in phrase.notes]


def test_render_is_finite_and_silent_without_notes(phrase):
    y = render(normalize_patch(BASS), phrase, Physics())
    assert y.dtype == np.float32 and len(y) == int(round(phrase.duration * 32000))
    assert np.all(np.isfinite(y)) and 0.01 < np.abs(y).max() < 4.0
    quiet = render(normalize_patch(BASS), Phrase([], 0.2), Physics())
    assert np.abs(quiet).max() == 0.0


def test_every_switch_combination_renders(phrase):
    short = Phrase([Note(0.0, 0.1, 40)], 0.15)
    for ftype, slope, shape in ((0, 0, 0), (0, 1, 0), (0, 0, 1), (0, 1, 2), (1, 1, 0)):
        for wave in range(4):
            patch = normalize_patch(dict(osc1_wave=wave, osc2_wave=3 - wave, mix_osc2=200, mix_sub=150, sub_wave=wave % 3,
                                         filter_type=ftype, filter_slope=slope, filter_shape=shape, filter_res=110,
                                         filter_freq=100, filter_drive=100, distortion=100, mix_noise=60,
                                         mix_ring=80, osc_sync=1, osc_filter_mod=60, filter_lfo2=200,
                                         osc1_lfo1_pitch=160, portamento=40))
            y = render(patch, short, Physics())
            assert np.all(np.isfinite(y)), (ftype, slope, shape, wave)


def test_pitch_follows_note_and_range():
    for note, rng, expect in ((45, 64, 110.0), (45, 63, 55.0), (57, 65, 440.0)):
        y = render(normalize_patch(dict(osc1_wave=0, osc1_range=rng)), Phrase([Note(0.0, 0.5, note)], 0.5),
                   Physics(), 32000)
        spec = np.abs(np.fft.rfft(y * np.hanning(len(y))))
        peak = np.argmax(spec) * 32000 / len(y)
        assert abs(peak - expect) < 3.0


def test_distance_is_zero_for_identical_sound_and_grows_with_the_error(phrase):
    true = normalize_patch(BASS)
    y = render(true, phrase, Physics())
    target = Target(y, notes=notes_of(phrase))
    assert target.loss(extract(y)) < 1e-6
    losses = [target.loss(extract(render(dict(true, filter_freq=v), phrase, Physics()))) for v in (120, 128, 140, 170)]
    assert losses[0] < 1e-6 and losses[0] < losses[1] < losses[2] < losses[3]
    assert target.loss(extract(render(dict(true, osc1_wave=0), phrase, Physics()))) > 0.05


def test_distance_ignores_level_and_small_time_shifts(phrase):
    true = normalize_patch(BASS)
    y = render(true, phrase, Physics())
    target = Target(y, notes=notes_of(phrase))
    assert target.loss(extract(0.2 * y)) < 1e-4
    shifted = np.concatenate([np.zeros(int(0.012 * 32000), dtype=np.float32), y])[:len(y)]
    cmp_ = target.compare(extract(shifted))
    assert cmp_["loss"] < 0.02 and abs(cmp_["shift_ms"] - 12.0) <= 4.0


def test_free_running_phase_is_only_a_small_disturbance(phrase):
    true = normalize_patch(dict(filter_freq=140, amp_release=20))
    target = Target(render(true, phrase, Physics(), phase1=0.77), notes=notes_of(phrase))
    losses = [target.loss(extract(render(true, phrase, Physics(), phase1=p))) for p in (0.1, 0.35, 0.6, 0.9)]
    assert max(losses) < 0.06
    # a real change of the sound is much larger than that
    assert target.loss(extract(render(dict(true, filter_freq=100), phrase, Physics()))) > 2 * max(losses)


def test_modulation_view_tells_detune_apart(phrase):
    reese = normalize_patch(dict(mix_osc2=255, osc2_fine=140, filter_freq=150, amp_release=20))
    target = Target(render(reese, phrase, Physics(), phase1=0.7, phase2=0.2), notes=notes_of(phrase))
    model = Model(Physics(), phrase, target, workers=1)
    model.set_precision(True)
    right = model.compare(reese)
    unison = model.compare(dict(reese, osc2_fine=128))
    assert two_sources(reese) and right["modulation"] < unison["modulation"]
    assert right["loss"] < unison["loss"]


def test_analytic_filter_response_matches_the_kernel():
    # white noise through the model filter vs. the formula used by the fast search
    ph = Physics()
    phrase = Phrase([Note(0.0, 1.9, 48)], 2.0)
    base = dict(mix_osc1=0, mix_noise=255, filter_track=7)
    open_ = render(normalize_patch(dict(base, filter_slope=0, filter_freq=255)), phrase, ph)
    freqs = np.array([100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0])

    def band_db(y):
        spec = np.abs(np.fft.rfft(y[6400:])) ** 2
        f = np.fft.rfftfreq(len(y) - 6400, 1 / 32000)
        return np.array([10 * np.log10(spec[(f > c / 1.12) & (f < c * 1.12)].mean()) for c in freqs])

    for ftype, slope, res in ((0, 1, 0), (0, 0, 60), (1, 1, 60)):
        patch = normalize_patch(dict(base, filter_type=ftype, filter_slope=slope, filter_freq=150, filter_res=res))
        measured = band_db(render(patch, phrase, ph)) - band_db(open_)
        from bs2autotune import engine
        P = engine.resolve(patch, ph, 32000.0)
        predicted = engine.filter_response_db(freqs, 32000.0, P, 48.0)
        assert np.max(np.abs(measured - predicted)) < 2.5, (ftype, slope, res, measured - predicted)
    assert filter_mag_db(np.array([10.0]), 32000.0, 0, 1, 0, 1000.0, 1.4, 1.3, 0.3)[0] > -0.5


def test_structure_helpers():
    patch = normalize_patch(dict(osc1_wave=3, osc1_pw=30, mix_sub=120, sub_wave=2, filter_type=1, osc2_wave=1))
    st = structure_of_patch(patch)
    assert st["filter_mode"] == "acid" and st["sub_on"] == 1 and st["osc2_on"] == 0
    assert "pulse 8'" in describe_structure(patch) and "acid" in describe_structure(patch)
    cleaned = clean_patch(patch)
    assert cleaned["osc2_wave"] == 2  # an oscillator that is off goes back to neutral
    assert structure_key(structure_of_patch(cleaned)) == structure_key(st)
