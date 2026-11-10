# Handoff: where the project stands and what to test next

For whoever continues this work (person or assistant). Read `README.md` first; this file only adds
what is not in it: the open questions, the first tests on real hardware, and where to look when
they fail.

## State

* Branch `claude/bs2-thriller-auto-tuning-hdi0ww`. Everything is committed; no pull request yet.
* `python -m pytest` : 49 pass, 5 skipped (2 slow, 2 need virtual MIDI ports, 1 needs Demucs).
  `python -m pytest --runslow` : 51 pass, about 3 minutes.
* Never run so far: a real Bass Station II, macOS (CoreMIDI / CoreAudio), Demucs with trained weights.
* Conventions: code, comments, docs and UI text in English; conversation with the owner in Czech.
  Hard rules from the owner: the app must never write a patch slot or send a bank/patch dump, the web
  server stays on 127.0.0.1, no CDN. `tests/test_hardware.py::test_the_only_sysex_ever_sent_is_the_dump_request`
  guards the first one - keep it green.

## Tests to run on the owner's Mac, in this order

Each step names what to report back. Stop at the first failure; later steps depend on earlier ones.

1. **Install and start.** `./start.command`. Report: Python version used, any pip error, whether the
   browser opens. Then `./start.command --list-devices`: are the Bass Station II MIDI ports and the
   audio interface listed?
2. **Test suite on macOS.** `.venv/bin/python -m pip install pytest && .venv/bin/python -m pytest -rs`.
   `tests/test_real_midi.py` should now *run* (CoreMIDI has virtual ports) instead of being skipped.
   Report the summary line and any failure in full.
3. **Demo mode.** `./start.command --demo`, drop any bass line, analyse, quick match. Confirms the
   page and the numerics on this machine. Report the distance and the duration.
4. **Test signal** (hardware mode, step 1 of the page). Report the message: latency, peak, noise.
   Expected: latency 5 - 60 ms, peak around -12 dBFS, noise below -60 dBFS.
   * "No sound arrives": MIDI channel, input channel, macOS microphone permission for Terminal.
   * Latency jumping between runs by more than a few ms: suspect `SoundDeviceBackend._callback`
     (`inputBufferAdcTime` handling in `bs2autotune/hardware.py`).
5. **Calibration.** Report the whole list of steps with their ticks/warnings and both charts
   (screenshot is fine), plus `data/calibration_hardware.json`. Things to look at:
   * "Waveform switching (NRPN)": must say the waveform follows MIDI. If not, the NRPN numbers in
     `bs2autotune/params.py` or the firmware assumption are wrong.
   * Filter cutoff curve monotonic, roughly 20 Hz - 20 kHz; envelope times monotonic.
   * "Character": the model error on typical sounds after fitting. On the simulated synth it ends near
     0.06. Much above 0.15 on the real unit means the model misses something real (see below).
6. **Self-match.** Dial a simple patch by hand on the synth (one saw, filter half closed, some
   envelope), record a few bars of it through the same interface, and feed that recording to the app
   as an isolated bass line, effort Normal. This is the cleanest test of the whole chain: the answer
   exists exactly. Report distance, the patch sheet, and whether A/B sounds the same.
   Expected: below about 0.10, same waveform and filter type.
7. **Real reference, isolated bass.** A bass stem or a passage with bass alone. Report the piano roll
   (are the notes right?), distance, and a listening verdict.
8. **Full song with Demucs.** First whether Demucs installs and the weights download; then listen to
   the excerpt in step 2 (how clean is the bass?), check the notes, run a Normal match.
   Send the session folder (`data/sessions/<id>/`) if the result is poor.

For any failure the most useful material is: the *Details* log of step 3 on the page, the session
folder, and `data/calibration_hardware.json`.

## Known weak spots, most likely to matter first

* **Model versus real instrument.** All fidelity numbers come from a simulated synth of the same
  family as the model. A real unit may differ in ways calibration does not capture (envelope shapes,
  filter saturation, oscillator levels versus pitch). The refinement loop (`refine.py`) compensates,
  but if step 5/6 above show a large residual, improve `engine.py` / `calibration.py` first - a better
  model beats more plays.
* **Full mix without Demucs** is unusable with drums (distance 0.6 in the synthetic test). The
  built-in isolation is a low-pass only. Not worth more work unless Demucs turns out unusable.
* **Leakage in separated stems.** Notes survive leakage at -8 dB, the *match* does not: 0.13 at
  -20 dB, about 0.27 at -14 dB (scripts used simulated leakage, not real Demucs output). Tried and
  dropped because results were inconsistent: weighting reference-louder cells less in the distance.
  Not tried: a per-note robust (trimmed) aggregation of the distance, harmonic masking of the target.
* **Quick effort is noisy on hard references** (same case gave 0.13 - 0.29 between runs). Normal or
  Thorough for anything that matters.
* **Soft or legato sounds.** Repeated notes of the same pitch merge when a long release fills the
  gap; legato pitch changes are placed within about 40 ms. Bright detuned sounds with long release
  still get a few extra note splits.
* **Not matched at all:** LFO modulation, sync, ring modulation, pitch envelope, portamento.
  Detecting a wobble in the reference and telling the user would be a cheap first step.

## Useful commands

```bash
.venv/bin/python -m pytest                       # fast suite
.venv/bin/python -m pytest --runslow             # with end-to-end matches
.venv/bin/python -m pytest tests/test_real_midi.py -rs    # MIDI stack on virtual ports
.venv/bin/python -m bs2autotune --demo --no-browser --port 8765
```

Where things are: `README.md` ("How it works", "Development") maps every module.
