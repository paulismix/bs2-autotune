# BS2 Autotune

Give it a bass line, get the patch: **automatic sound matching for the Novation Bass Station II.**

You drop an audio file into the browser - an isolated bass line or a whole song. The app works out which
notes are played, then programs your Bass Station II over USB, plays the same notes on it, records the result
through your audio interface, compares it with the reference and keeps improving the patch until the synth
sounds as close to the reference as it can. The matched patch stays in the synth's edit buffer, ready to be
played and saved.

Everything runs locally on your Mac, with a web interface at `http://127.0.0.1:8765/`.

```
 reference audio ──► notes + excerpt ──► search in a model of the synth ──► refine on the real synth ──► patch
                                               ▲                                  │  ▲
                                               └──── calibration of your unit ────┘  └─ USB MIDI out, audio in
```

## Quick start

1. Connect the Bass Station II to the Mac by USB, and its line output to an input of your audio interface.
2. Get the code and start it:

   ```bash
   git clone https://github.com/paulismix/bs2-autotune.git
   cd bs2-autotune
   ./start.command
   ```

   You can also double-click `start.command` in Finder. The first start creates a private Python environment
   in `.venv` and installs the requirements (a few minutes); after that the app starts within seconds and
   opens the browser.
3. Follow the four steps on the page: **Connect → Reference → Match → Result**.

No synth at hand? `./start.command --demo` runs the complete workflow against a built-in virtual synth.

### Requirements

* macOS (Apple silicon or Intel). The code is plain Python and also runs on Linux; it is developed against
  macOS' CoreMIDI and CoreAudio through `python-rtmidi` and `sounddevice`.
* Python 3.10 - 3.12 (3.12 recommended: ready-made packages exist for all requirements). Install it from
  [python.org](https://www.python.org/downloads/) or with `brew install python@3.12`. If you use
  [`uv`](https://docs.astral.sh/uv/), the launcher fetches Python 3.12 by itself.
* A Bass Station II on USB, and any class compliant audio interface with a line input.
* Optional: [Demucs](https://github.com/adefossez/demucs) for full songs (see below), `ffmpeg` for unusual
  file formats (WAV, AIFF, FLAC, OGG and MP3 are read directly, M4A and others through macOS' `afconvert`).

Manual installation instead of the launcher:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m bs2autotune
```

## Before the first match

* **Cabling.** Line output of the synth into a *line* input of the interface: no instrument or microphone
  gain stage, no pad, no effects, no direct-monitoring tricks. One mono cable is all it takes.
* **Synth.** Load an initialised patch, keep the mod wheel down, turn the volume to about three quarters. The
  app sets every sound parameter it knows over MIDI; an initialised patch makes sure the few things that
  cannot be reached over MIDI are neutral too.
* **macOS microphone permission.** The first recording makes macOS ask whether the program that runs the app
  (usually Terminal) may use the microphone. Allow it - without the permission macOS hands the app silence.
  You can change it later in *System Settings → Privacy & Security → Microphone*.
* **Test signal.** Step 1 has a *Test signal* button. It plays a few notes and reports level, noise and
  latency, and tells you when the input is too hot or too quiet.
* **Calibrate once.** *Calibrate* plays about a hundred short test notes (roughly a minute and a half) and
  measures how *your* unit responds: latency, tuning, filter and envelope curves, levels, and the colour of
  the signal path. The match relies on this, so the app insists on it before the first match. Repeat it when
  you change the audio interface, its gain or the cabling.

## The four steps

**1 Connect** - choose the MIDI output, the MIDI channel (the synth's default is 1) and the audio input the
synth is connected to. A MIDI input is optional; with it the app can save a SysEx backup of the result.

**2 Reference** - drop an audio file. Tell the app whether it is an isolated bass line or a full mix, drag
over the waveform to choose the part you care about, and press *Analyse*. A selection of up to 8 seconds is
used exactly as selected; from anything longer the app picks a representative passage of about four seconds
itself. You then see the notes it found and can listen to the excerpt. If the notes are split wrongly, move
the *Note splitting* slider and analyse again.

**3 Match** - choose the effort and press *Start matching*. The app first searches in its model of your
synth (no sound), then plays candidates on the real synth and refines the best one. You can stop at any time;
the best patch so far is kept.

| Effort   | Plays on the synth | Good for                                   |
|----------|--------------------|--------------------------------------------|
| Quick    | about 24           | a first impression                         |
| Normal   | about 60           | the default                                |
| Thorough | about 130          | squeezing out the last bit, difficult sounds |

One play lasts as long as the excerpt, so a normal match of a four second excerpt takes a few minutes of
model search plus about five minutes on the synth.

**4 Result** - an A/B player for reference and synth at equal loudness, both spectrograms, the distance and
how it developed, and the patch as a sheet that lists only what differs from an initialised patch. *Send to the
synth* puts the patch into the edit buffer again, *Play the phrase* plays the excerpt's notes with it. To keep
the patch, press **Save** on the Bass Station II and choose a slot - the app never writes a slot itself. You
can also store results in the app's own library and download the patch sheet, the data and (with a MIDI input
selected) a `.syx` backup.

### Getting a faithful result

* **An isolated bass line is by far the best reference.** Use a stem, an intro or a break where the bass
  plays alone, if there is one.
* **Full songs** need the bass isolated first. With Demucs installed the app uses it automatically:

  ```bash
  .venv/bin/python -m pip install demucs      # large download; model weights are fetched on first use
  ```

  Without Demucs the built-in fallback keeps only the low end (below roughly 400 Hz) and matches that part of
  the spectrum. It finds the right notes and the body of the sound, but cannot judge its brightness.
* **Choose a passage with several different notes, short and long.** Short notes reveal the attack and the
  filter envelope, long ones the sustained tone, different pitches the filter's key tracking.
* **Tune the patch to the recording** (checkbox in step 2) if you want to play along with the original:
  old recordings are rarely at A = 440 Hz.
* **Read the distance with the label.** Below 0.08 the two are very hard to tell apart; 0.08 - 0.12 is very
  close; above 0.18 something audible is missing - usually because the reference contains things a Bass
  Station II cannot do (see below), or because other instruments leak into it.

### What is matched - and what is not

The search covers the sound engine that makes up nearly every bass patch: both oscillators (waveform, octave,
pulse width, detune, levels), the sub oscillator (waveform, octave, level), noise, the filter (Classic or
Acid, slope, low-/band-/high-pass, frequency, resonance, overdrive, envelope depth, key tracking), both
envelopes and the distortion.

Left neutral on purpose, because they cannot be identified reliably from a few seconds of audio: LFO
modulation (vibrato, filter wobble, pulse width modulation), oscillator sync, ring modulation, pitch
envelope, portamento, velocity sensitivity, arpeggiator and paraphonic mode. A reference that lives on these
is matched in its average tone only; add the movement by hand afterwards.

And of course the result is the closest *Bass Station II* sound. A layered bass, a sampled instrument or
another synth's filter character can only be approximated.

## What the app does to your synth

* It changes the **edit buffer** only - the patch currently being edited - through MIDI Control Change and
  NRPN messages, and it plays notes. Besides that it sends: mod wheel to zero, sustain pedal off, pitch wheel
  to centre and All Notes Off.
* It **never** sends a Program Change, never writes a patch slot and never sends a patch or bank dump. Your
  stored patches cannot be overwritten by the app. Saving is something you do on the synth.
* The only SysEx message it ever sends is the nine byte request for the current patch
  (`F0 00 20 29 00 33 00 40 F7`), and only when you selected a MIDI input. The answer is saved as
  `patch.syx` next to the result. This is covered by a test (`tests/test_hardware.py`).
* The web server listens on `127.0.0.1` only, refuses requests from other sites, and loads nothing from the
  internet (no CDN, no fonts, no analytics).

## How it works

1. **Reading the reference** (`analysis.py`, `separation.py`). The file is decoded and - for a mix - the
   bass is isolated. A YIN-type pitch tracker with a Viterbi path and an onset detector turn the audio into
   notes. Low, bright notes are a classic trap here: a short analysis window sees every single period as a
   click, so note starts are confirmed on a spectrum that is averaged over exactly one pitch period.
2. **A model of the synth** (`engine.py`). A fast simulation of the Bass Station II signal path: band-limited
   oscillators, sub oscillator, noise, ring modulator, mixer and overdrive, Classic (state variable) and Acid
   (diode ladder type) filters, two ADSR envelopes, LFOs, distortion. It is compiled with numba and renders
   three seconds of audio in a few milliseconds.
3. **Calibration** (`calibration.py`). The model's curves and constants are measured on your unit, so that
   a parameter value means the same in the model as on the hardware.
4. **Distance** (`features.py`). Reference and candidate are compared the way an ear would: a detailed
   spectrum over time (72 bands), the attack on a short time scale, the loudness envelope, and the amount of
   beating between detuned oscillators - after aligning them in time and loudness. The oscillators of the
   synth run free, so everything is made independent of their phase.
5. **Search in the model** (`static_match.py`, `search.py`). First the *switches* - waveforms, octaves, sub
   oscillator, filter type - are decided by matching the reference's spectrum against a dictionary of every
   combination. Then envelopes and all continuous parameters are optimised (CMA-ES and coordinate sweeps),
   starting from several different explanations of the sound so that the search does not get stuck with the
   first one.
6. **Refinement on the hardware** (`refine.py`). No model is perfect, so the best candidates are played on the
   real synth. The differences between what the model predicted and what was recorded are used to correct the
   model on the spot (space mapping), the corrected model proposes the next patch, and so on, followed by a
   direct search on the synth itself. A candidate only replaces the current best when it is better by more
   than the measured repeatability of the synth.

## Files

Everything the app stores lives in the `data` folder next to the code (change it with `--data DIR` or the
environment variable `BS2AUTOTUNE_DATA`):

```
data/settings.json                 chosen devices and options
data/calibration_hardware.json     the calibration of your unit
data/reference/                    the current reference file and its excerpt
data/sessions/<date-time>/         one folder per match:
    patch.txt                        the patch sheet (what to set on the panel, and all raw values)
    result.json                      patch, distance, history, the notes that were played
    reference.wav, synth.wav         the A/B pair at equal loudness
    reference.png, synth.png         their spectrograms
    patch.syx                        SysEx backup of the edit buffer (only with a MIDI input)
data/library/                      patches you saved in the app
```

## Command line

```
./start.command [--port 8765] [--data DIR] [--no-browser] [--demo] [--list-devices]
```

`--list-devices` prints the MIDI ports and audio inputs the app can see and exits.

## Status: what has been tested, and what has not

Honest version, because it matters for a tool that talks to hardware:

* **Tested automatically** (`python -m pytest`, plus `--runslow` for the end-to-end runs): the parameter map
  and its MIDI encoding, the synth model, the distance measure, note transcription, calibration, the search,
  the hardware refinement loop, the MIDI/audio layer and the web server. The instrument in these tests is a
  *simulated* Bass Station II that deliberately differs from the app's own model (other curves and levels,
  latency, noise, tuning drift, coloured output), connected through a simulated MIDI and audio rig with its
  own clock. On it, matching a hidden patch typically ends at a distance of 0.06 - 0.13 with the right
  oscillator and filter structure.
* **Not tested: the real thing.** The software was developed without access to a Bass Station II, so real
  CoreMIDI/CoreAudio operation and the fidelity on real hardware are unproven until you run it. The first
  things to check are *Test signal* and *Calibrate* in step 1: they verify the connection end to end, and the
  calibration reports anything on your unit that does not respond as expected (it also notices if waveform
  switching or filter tracking cannot be controlled over MIDI on your firmware, and then leaves them alone).
* **Not tested: Demucs.** The integration is written against its documented Python API, but its model
  weights could not be downloaded in the development environment.
* **The MIDI implementation** was taken from two independent community sources that agree with each other
  (see below), not from Novation's own documents. Parameters added by firmware 2.5 and 4.14 can be left out
  with the checkbox in step 1 if your unit runs older firmware.

If something does not behave as described, the log in step 3 (*Details*) and the files of the session are the
place to look - and worth attaching to a bug report.

## Troubleshooting

* **The launcher is blocked by macOS** ("cannot be opened because it is from an unidentified developer") -
  right-click `start.command` and choose *Open*, or start it from Terminal with `./start.command`. If it is
  not executable: `chmod +x start.command`.
* **Installation fails on Python 3.13 or newer** - `python-rtmidi` has ready-made packages up to Python 3.12
  only. Install Python 3.12, delete the `.venv` folder and start again.
* **No MIDI port for the synth** - check the USB cable and that the synth is on, then *Refresh devices*.
  `./start.command --list-devices` shows what the system offers.
* **"No signal"** - wrong input channel, synth volume down, or macOS denied microphone access (see above).
* **Input too hot / too quiet** - set the interface gain so that the test signal peaks around -12 dBFS, then
  calibrate again.
* **The notes in step 2 are wrong** - select a cleaner part of the recording, switch between *isolated bass*
  and *full mix*, or move the *Note splitting* slider. The match can only be as good as the notes.
* **The match is poor** - listen to the excerpt first: is it really just the bass? Then try *Thorough*. If the
  spectrograms show movement in the reference that the synth lacks, it is probably an LFO or an effect.
* **Port 8765 is taken** - `./start.command --port 8777`.
* **The page does not react during a match** - one job runs at a time; *Stop* ends it within one play.

## Development

```bash
.venv/bin/python -m pip install pytest
.venv/bin/python -m pytest              # fast tests, well under a minute
.venv/bin/python -m pytest --runslow    # plus complete matches on the virtual synth (several minutes)
```

```
bs2autotune/
  params.py        every Bass Station II parameter: CC / NRPN numbers, ranges, initial values, patch sheet
  engine.py        the synth model (numba)             features.py     the distance measure
  calibration.py   measuring a unit                    model.py        model + target, parallel rendering
  static_match.py  spectral dictionary search          search.py       optimisation in the model
  refine.py        closed loop on the instrument       device.py       device interface, virtual synth
  hardware.py      MIDI and audio I/O                  analysis.py     pitch, onsets, notes, excerpt
  separation.py    bass isolation                      app.py          jobs, settings, results, library
  server.py        local HTTP server and API           web/            the page (plain HTML, CSS, JavaScript)
tests/             unit tests, simulated rig (fakes.py), end-to-end tests
```

## Sources and credits

* MIDI implementation (CC and NRPN numbers, value ranges): the open
  [midi.guide](https://midi.guide/) dataset ([pencilresearch/midi](https://github.com/pencilresearch/midi)),
  cross-checked against François Georgy's [BS2-Web](https://github.com/francoisgeorgy/BS2-Web).
* SysEx patch layout and the "request current patch" message: François Georgy's
  [BS2-SysEx](https://github.com/francoisgeorgy/BS2-SysEx) notes.
* The matching itself (model, calibration, distance, search, refinement) was written for this project.

Novation and Bass Station are trade marks of Focusrite Audio Engineering Ltd. This is an independent,
unofficial tool; it is not affiliated with or endorsed by Novation.
