"""Reading the reference: pitch, notes, excerpt, isolation of the bass."""

import numpy as np
import pytest

from bs2autotune.analysis import (AudioError, build_reference, choose_excerpt, load_audio, note_name, track_pitch,
                                  transcribe)
from bs2autotune.audio_util import write_wav
from bs2autotune.device import VirtualDevice
from bs2autotune.params import normalize_patch
from bs2autotune.separation import separate_bass

SOUNDS = {
    "pluck": dict(osc1_wave=2, osc1_range=65, mix_sub=200, sub_wave=2, filter_freq=115, filter_res=45, filter_env=105,
                  mod_decay=55, mod_sustain=20, amp_decay=85, amp_sustain=70, amp_release=25),
    "sine": dict(osc1_wave=0, filter_freq=200, amp_attack=20, amp_release=40),
    "square": dict(osc1_wave=3, filter_type=1, filter_freq=130, filter_res=80, filter_env=95, mod_decay=45,
                   mod_sustain=0, amp_decay=90, amp_sustain=100, amp_release=15, distortion=40),
    # low and bright: a short analysis window sees every period of these as a click
    "bright": dict(filter_freq=150, filter_env=90, mod_decay=60, mod_sustain=30, amp_release=25),
    "open": dict(filter_freq=230, amp_release=20),
    "acid": dict(osc1_wave=2, filter_type=1, filter_freq=90, filter_res=110, filter_env=110, mod_decay=50,
                 mod_sustain=0, amp_release=20, distortion=60),
}


def recording(line, patch):
    dev = VirtualDevice(seed=11)
    dev.latency = dev.true_latency
    return dev.capture(normalize_patch(patch), line)


def test_pitch_of_plain_tones():
    sr = 32000
    t = np.arange(int(0.6 * sr)) / sr
    for hz in (41.2, 55.0, 110.0, 196.0):
        y = (0.5 * np.sin(2 * np.pi * hz * t) + 0.2 * np.sin(2 * np.pi * 2 * hz * t)).astype(np.float32)
        times, f0 = track_pitch(y, sr)
        voiced = f0[(times > 0.1) & (times < 0.5)]
        assert np.all(voiced > 0)
        assert abs(1200 * np.log2(np.median(voiced) / hz)) < 8.0


@pytest.mark.parametrize("name", sorted(SOUNDS))
def test_transcription_finds_every_note(line, name):
    y = recording(line, SOUNDS[name])
    notes, info = transcribe(y)
    assert len(notes) == len(line.notes)
    for found, true in zip(notes, line.notes):
        assert found.note == true.note
        assert abs(found.start - true.start) < 0.025  # at most one period of the lowest notes
        assert abs(found.end - true.end) < 0.05
    assert float(np.mean([abs(f.start - t.start) for f, t in zip(notes, line.notes)])) < 0.006
    assert abs(info["tuning_cents"]) < 25


def test_detuned_oscillators_do_not_shatter_the_notes(line):
    y = recording(line, dict(mix_osc2=255, osc2_fine=141, filter_freq=140, filter_res=20, amp_release=30))
    notes, _ = transcribe(y)
    assert len(line.notes) - 2 <= len(notes) <= len(line.notes) + 6
    matched = sum(1 for t in line.notes if any(abs(f.start - t.start) < 0.03 for f in notes))
    assert matched >= len(line.notes) - 1


def test_notes_that_run_into_each_other(line):
    # a long release fills the gaps: repeated notes merge, but every change of pitch
    # (the semitone B1 - C2 included) is still found
    y = recording(line, dict(osc1_wave=3, filter_freq=160, amp_release=90))
    notes, _ = transcribe(y)
    changes = [t for prev, t in zip(line.notes, line.notes[1:]) if t.note != prev.note]
    for true in changes:  # without an attack the moment of a change is known to a few periods only
        assert any(abs(f.start - true.start) < 0.045 and f.note == true.note for f in notes), true
    assert len(changes) <= len(notes) <= len(line.notes)


def other_instruments(n, sr=32000, beat=60 / 110 / 2):
    """Kick, snare, hi-hat and a chord pad on the grid of the bass line."""
    import scipy.signal

    rng = np.random.default_rng(1)
    out = np.zeros(n)
    k = 0
    while 0.3 + k * beat < n / sr - 0.5:
        i0 = int((0.3 + k * beat) * sr)
        if k % 4 == 0:  # kick: a sine that sweeps down through the range of the bass
            t = np.arange(int(0.25 * sr)) / sr
            out[i0:i0 + len(t)] += 0.7 * np.sin(2 * np.pi * np.cumsum(45 + 90 * np.exp(-t / 0.03)) / sr) * np.exp(-t / 0.09)
        if k % 8 == 4:  # snare
            t = np.arange(int(0.18 * sr)) / sr
            sos = scipy.signal.butter(2, [900, 6000], btype="band", fs=sr, output="sos")
            out[i0:i0 + len(t)] += 0.35 * scipy.signal.sosfilt(sos, rng.standard_normal(len(t))) * np.exp(-t / 0.05)
        t = np.arange(int(0.05 * sr)) / sr  # hi-hat on every eighth note
        sos = scipy.signal.butter(2, 7000, btype="high", fs=sr, output="sos")
        out[i0:i0 + len(t)] += 0.12 * scipy.signal.sosfilt(sos, rng.standard_normal(len(t))) * np.exp(-t / 0.012)
        k += 1
    t = np.arange(n) / sr
    for midi in (52, 55, 59, 64):  # a held chord of slightly detuned saws
        for detune in (-0.004, 0.004):
            out += 0.05 * scipy.signal.sawtooth(2 * np.pi * 440 * 2 ** ((midi - 69) / 12) * (1 + detune) * t)
    return out


@pytest.mark.parametrize("name", ["pluck", "open", "sine"])
def test_notes_survive_what_a_separator_leaves_behind(line, name):
    # an isolated stem still carries the rest of the band, some 20 dB down: hi-hats in the
    # middle of long notes and a kick under every fourth one must not change the notes
    bass = recording(line, SOUNDS[name]).astype(np.float64)
    bass *= 0.5 / np.abs(bass).max()
    notes, _ = transcribe((bass + 0.1 * other_instruments(len(bass))).astype(np.float32))
    assert len(notes) == len(line.notes)
    for found, true in zip(notes, line.notes):
        assert found.note == true.note and abs(found.start - true.start) < 0.03


def test_bass_and_kick_together(line):
    # much stronger leakage: the pitch is found late under a kick, the note still starts with it
    bass = recording(line, SOUNDS["open"]).astype(np.float64)
    bass *= 0.5 / np.abs(bass).max()
    notes, _ = transcribe((bass + 0.4 * other_instruments(len(bass))).astype(np.float32))
    hits = sum(1 for t in line.notes if any(abs(f.start - t.start) < 0.05 and f.note == t.note for f in notes))
    assert hits >= len(line.notes) - 2 and len(notes) <= len(line.notes) + 4


def test_excerpt_and_phrase(line):
    y = recording(line, SOUNDS["pluck"])
    notes, info = transcribe(y)
    t0, t1 = choose_excerpt(notes, len(y) / 32000)
    assert 2.0 < t1 - t0 <= 6.0
    ref = build_reference(y, notes, t0, t1, info["tuning_cents"])
    assert len(ref.audio) == int(round((t1 - t0) * 32000))
    assert 6 <= len(ref.phrase.notes) <= 24
    assert all(0.0 <= n.start < n.end <= ref.phrase.duration for n in ref.phrase.notes)
    assert ref.phrase.notes[0].start < 0.12
    with pytest.raises(AudioError):
        build_reference(y, [], 0.0, 1.0)


def test_audio_files_are_decoded(tmp_path, line):
    y = recording(line, SOUNDS["sine"])
    path = tmp_path / "ref.wav"
    write_wav(str(path), y, 32000)
    data, sr = load_audio(str(path))
    assert sr == 32000 and data.shape == (len(y), 1)
    bad = tmp_path / "broken.wav"
    bad.write_bytes(b"this is not audio")
    with pytest.raises(AudioError):
        load_audio(str(bad))
    assert note_name(28) == "E1" and note_name(60) == "C4"


def test_builtin_isolation_keeps_the_low_end(line):
    rng = np.random.default_rng(1)
    bass = recording(line, SOUNDS["sine"]).astype(np.float64)
    hiss = 0.05 * rng.standard_normal(len(bass))
    import scipy.signal
    hiss = scipy.signal.sosfilt(scipy.signal.butter(4, 2000, btype="high", fs=32000, output="sos"), hiss)
    sep = separate_bass((bass + hiss).astype(np.float32), 32000, "builtin")
    assert sep.method == "builtin" and len(sep.bass) == len(bass)
    f, p_out = scipy.signal.welch(sep.bass, 32000, nperseg=4096)
    _, p_in = scipy.signal.welch(bass + hiss, 32000, nperseg=4096)
    low, high = (f > 35) & (f < 150), f > 3000
    assert p_out[low].sum() > 0.5 * p_in[low].sum()  # the bass survives
    assert p_out[high].sum() < 0.01 * p_in[high].sum()  # the rest is gone
    weights = dict(zip(sep.reliability[:, 0].round().astype(int), sep.reliability[:, 1]))
    assert weights[80] == 1.0 and weights[1016] < 0.1


def test_demucs_integration(monkeypatch):
    """Runs only with the optional demucs package installed.  The trained weights are not
    downloaded: an untrained model exercises the whole integration, the audio is meaningless."""
    pretrained = pytest.importorskip("demucs.pretrained")
    torch = pytest.importorskip("torch")
    from demucs.htdemucs import HTDemucs

    def untrained(name, repo=None):
        assert name == "htdemucs"
        torch.manual_seed(0)
        return HTDemucs(sources=["drums", "bass", "other", "vocals"])

    monkeypatch.setattr(pretrained, "get_model", untrained)
    sr = 44100
    t = np.arange(int(2.5 * sr)) / sr
    left = 0.4 * np.sign(np.sin(2 * np.pi * 55 * t)) + 0.2 * np.sin(2 * np.pi * 880 * t)
    stereo = np.stack([left, 0.8 * left], axis=1).astype(np.float32)
    messages = []
    sep = separate_bass(stereo, sr, "demucs", messages.append)
    assert sep.method == "demucs" and sep.bass.dtype == np.float32
    assert abs(len(sep.bass) - 2.5 * 32000) < 40 and np.isfinite(sep.bass).all()
    assert sep.reliability.shape == (30, 2) and sep.reliability[:, 1].min() >= 0.1
    assert any("Demucs" in m for m in messages)
