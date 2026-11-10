"""Application state and background jobs behind the web interface.

One :class:`App` holds everything: settings, the calibration of the
instrument, the loaded reference, the running job and the last result.  Long
operations (calibration, analysis, matching) run as a :class:`Job` in a worker
thread and report through an event bus that the web page listens to.
"""

from __future__ import annotations

import json
import os
import queue
import re
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

from . import __version__
from .analysis import (AudioError, RefNote, Reference, build_reference, choose_excerpt, load_audio, note_name,
                       to_mono, transcribe)
from .audio_util import normalize_peak, peak_db, resample, write_wav
from .calibration import CalibrationResult, Calibrator
from .device import Cancelled, Device, DeviceError, VirtualDevice, apply_tuning
from .engine import Note, Phrase, Physics
from .features import ANALYSIS_SR, Target, extract
from .imaging import spectrogram_png
from .model import Model
from .params import (PARAMS, baseline_patch, describe_patch, dump_with_name, normalize_patch, patch_text)
from .refine import HardwareLoop
from .search import ModelSearch, clean_patch, describe_structure
from .separation import demucs_available, separate_bass

DEFAULT_SETTINGS: Dict[str, object] = {
    "mode": "",  # "hardware" or "demo"; decided on first start
    "midi_out": "",
    "midi_in": "",
    "midi_channel": 1,
    "audio_device": "",
    "input_channel": 1,
    "samplerate": 0,  # 0 = the device default
    "firmware_features": True,
    "effort": "normal",
}

EFFORTS = {
    "quick": {"model": 0.6, "plays": 24, "label": "Quick"},
    "normal": {"model": 1.0, "plays": 60, "label": "Normal"},
    "thorough": {"model": 1.5, "plays": 130, "label": "Thorough"},
}


class Busy(RuntimeError):
    """Another job is still running."""


class UserError(RuntimeError):
    """A problem the user can fix; the message is shown as is."""


def quality_label(loss: float) -> str:
    if loss < 0.05:
        return "practically identical"
    if loss < 0.08:
        return "excellent"
    if loss < 0.12:
        return "very close"
    if loss < 0.18:
        return "close"
    if loss < 0.30:
        return "rough"
    return "poor"


class EventBus:
    def __init__(self) -> None:
        self._subs: List["queue.Queue[dict]"] = []
        self._lock = threading.Lock()

    def subscribe(self) -> "queue.Queue[dict]":
        q: "queue.Queue[dict]" = queue.Queue(maxsize=400)
        with self._lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q: "queue.Queue[dict]") -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def publish(self, event: dict) -> None:
        with self._lock:
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(event)
            except queue.Full:
                pass  # a stalled browser tab must not block the job


@dataclass
class Job:
    id: int
    kind: str
    title: str
    status: str = "running"  # running | done | error | cancelled
    phase: str = ""
    message: str = ""
    fraction: float = 0.0
    error: str = ""
    started: float = field(default_factory=time.time)
    finished: float = 0.0
    log: Deque[str] = field(default_factory=lambda: deque(maxlen=400))
    data: Dict[str, object] = field(default_factory=dict)
    cancel: threading.Event = field(default_factory=threading.Event)

    def to_dict(self) -> dict:
        return {"id": self.id, "kind": self.kind, "title": self.title, "status": self.status, "phase": self.phase,
                "message": self.message, "fraction": round(self.fraction, 4), "error": self.error,
                "elapsed": round((self.finished or time.time()) - self.started, 1), "log": list(self.log)[-60:],
                "data": self.data}


class App:
    def __init__(self, data_dir: str):
        self.data_dir = os.path.abspath(data_dir)
        for sub in ("", "reference", "sessions", "library"):
            os.makedirs(os.path.join(self.data_dir, sub), exist_ok=True)
        self.bus = EventBus()
        self._lock = threading.RLock()
        self._job_counter = 0
        self.job: Optional[Job] = None
        self._thread: Optional[threading.Thread] = None
        self.settings: Dict[str, object] = dict(DEFAULT_SETTINGS)
        self._load_settings()
        self.calibration: Dict[str, Optional[dict]] = {"hardware": None, "demo": None}
        self._load_calibrations()
        self.ref: Optional[dict] = None  # the loaded reference and its analysis
        self.result: Optional[dict] = self._last_result()
        self._demo_device: Optional[VirtualDevice] = None
        self.version = __version__

    # ------------------------------------------------------------------ paths
    def path(self, *parts: str) -> str:
        return os.path.join(self.data_dir, *parts)

    def _read_json(self, name: str) -> Optional[dict]:
        try:
            with open(self.path(name), "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _write_json(self, name: str, data: dict) -> None:
        tmp = self.path(name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1)
        os.replace(tmp, self.path(name))

    def _last_result(self) -> Optional[dict]:
        """The result of the most recent session (shown again after a restart)."""
        try:
            sessions = sorted(d for d in os.listdir(self.path("sessions")) if re.fullmatch(r"\d{8}-\d{6}", d))
        except OSError:
            return None
        for sid in reversed(sessions):
            data = self._read_json(os.path.join("sessions", sid, "result.json"))
            if data and "patch" in data and data.get("id") == sid:
                return data
        return None

    # --------------------------------------------------------------- settings
    def _load_settings(self) -> None:
        saved = self._read_json("settings.json") or {}
        for key in DEFAULT_SETTINGS:
            if key in saved and isinstance(saved[key], type(DEFAULT_SETTINGS[key])):
                self.settings[key] = saved[key]
        if self.settings["mode"] not in ("hardware", "demo"):
            from .hardware import guess_bs2_port, list_midi_ports

            ports = list_midi_ports()
            port = guess_bs2_port(ports["outputs"])
            self.settings["mode"] = "hardware" if port else "demo"
            if port:
                self.settings["midi_out"] = port
                self.settings["midi_in"] = guess_bs2_port(ports["inputs"]) or ""

    def update_settings(self, changes: dict) -> dict:
        with self._lock:
            for key, value in changes.items():
                if key not in DEFAULT_SETTINGS:
                    continue
                default = DEFAULT_SETTINGS[key]
                if isinstance(default, bool):
                    value = bool(value)
                elif isinstance(default, int):
                    value = int(value)
                else:
                    value = str(value)
                if key == "mode" and value not in ("hardware", "demo"):
                    continue
                if key == "midi_channel":
                    value = min(max(value, 1), 16)
                if key == "input_channel":
                    value = max(1, value)
                if key == "effort" and value not in EFFORTS:
                    continue
                self.settings[key] = value
            self._write_json("settings.json", self.settings)
        self.bus.publish({"type": "state"})
        return dict(self.settings)

    @property
    def mode(self) -> str:
        return str(self.settings["mode"])

    def list_devices(self) -> dict:
        from .hardware import guess_bs2_port, list_audio_inputs, list_midi_ports

        midi = list_midi_ports()
        audio = list_audio_inputs()
        return {"midi_outputs": midi["outputs"], "midi_inputs": midi["inputs"], "midi_error": midi["error"],
                "audio_inputs": audio["devices"], "audio_error": audio["error"],
                "bs2_port": guess_bs2_port(midi["outputs"]), "demucs": demucs_available()}

    def open_device(self) -> Device:
        if self.mode == "demo":
            if self._demo_device is None:
                self._demo_device = VirtualDevice(seed=2024)
            self._demo_device.cancel_event = None
            return self._demo_device
        from .hardware import open_hardware

        s = self.settings
        if not s["midi_out"]:
            raise UserError("Choose the MIDI output of the Bass Station II first.")
        if s["audio_device"] == "":
            raise UserError("Choose the audio input that the Bass Station II is connected to.")
        device: object = s["audio_device"]
        if isinstance(device, str) and device.isdigit():
            device = int(device)
        dev = open_hardware(str(s["midi_out"]), device, int(s["input_channel"]), int(s["samplerate"]) or None,
                            int(s["midi_channel"]), str(s["midi_in"]) or None,
                            None if s["firmware_features"] else "1.0")
        cal = self.calibration.get("hardware")
        if cal:
            dev.latency = float(cal.get("latency", 0.0))
        return dev

    def _close_device(self, dev: Optional[Device]) -> None:
        if dev is not None and not dev.is_virtual:
            try:
                dev.close()
            except Exception:
                pass

    # ------------------------------------------------------------ calibration
    def _cal_file(self, mode: Optional[str] = None) -> str:
        return f"calibration_{mode or self.mode}.json"

    def _load_calibrations(self) -> None:
        for mode in ("hardware", "demo"):
            data = self._read_json(self._cal_file(mode))
            if data and "physics" in data:
                self.calibration[mode] = data

    def physics(self) -> Tuple[Physics, bool]:
        """Physics of the current instrument and whether it has been calibrated."""
        cal = self.calibration.get(self.mode)
        if cal:
            return Physics.from_dict(cal["physics"]), True
        return Physics(), False

    def calibration_summary(self) -> Optional[dict]:
        cal = self.calibration.get(self.mode)
        if not cal:
            return None
        ph = cal["physics"]
        idx = list(range(0, 256, 5)) + [255]
        return {"date": cal.get("date", ""), "latency_ms": round(1000.0 * cal.get("latency", 0.0), 1),
                "peak_db": round(cal.get("peak_db", 0.0), 1), "noise_db": round(cal.get("noise_db", 0.0), 1),
                "tune_cents": round(cal.get("tune_cents", 0.0), 1), "steps": cal.get("steps", []),
                "warnings": cal.get("warnings", []), "device": cal.get("device", {}),
                "cutoff": [[i, round(min(2.0 ** ph["cutoff_classic"][i], 22000.0), 1)] for i in idx],
                "cutoff_acid": [[i, round(min(2.0 ** ph["cutoff_acid"][i], 22000.0), 1)] for i in idx],
                "attack": [[i, round(ph["attack_table"][i], 5)] for i in range(0, 128, 3)],
                "decay": [[i, round(ph["decay_table"][i], 5)] for i in range(0, 128, 3)],
                "release": [[i, round(ph["release_table"][i], 5)] for i in range(0, 128, 3)]}

    # ------------------------------------------------------------------- jobs
    def start_job(self, kind: str, title: str, fn: Callable[[Job], Optional[dict]]) -> Job:
        with self._lock:
            if self.job is not None and self.job.status == "running":
                raise Busy(f"'{self.job.title}' is still running.")
            self._job_counter += 1
            job = Job(self._job_counter, kind, title)
            self.job = job

        def run() -> None:
            try:
                fn(job)
                job.status = "cancelled" if job.cancel.is_set() and job.status == "running" else "done"
                if job.status == "done":
                    job.fraction = 1.0
            except Cancelled:
                job.status = "cancelled"
                job.message = "Stopped."
            except (UserError, DeviceError, AudioError) as exc:
                job.status = "error"
                job.error = str(exc)
            except Exception as exc:  # unexpected: keep the details for the log
                job.status = "error"
                job.error = f"{exc.__class__.__name__}: {exc}"
                job.log.append(traceback.format_exc())
            finally:
                job.finished = time.time()
                self.bus.publish({"type": "job", "job": job.to_dict()})
                self.bus.publish({"type": "state"})

        self._thread = threading.Thread(target=run, name=f"bs2job-{kind}", daemon=True)
        self._thread.start()
        self.bus.publish({"type": "job", "job": job.to_dict()})
        return job

    def stop_job(self) -> None:
        job = self.job
        if job is not None and job.status == "running":
            job.cancel.set()
            job.message = "Stopping ..."
            self.bus.publish({"type": "job", "job": job.to_dict()})

    def wait(self, timeout: Optional[float] = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def _progress(self, job: Job, phase: Optional[str] = None, message: Optional[str] = None,
                  fraction: Optional[float] = None, log: Optional[str] = None) -> None:
        if phase is not None:
            job.phase = phase
        if message is not None:
            job.message = message
        if fraction is not None:
            job.fraction = float(min(max(fraction, 0.0), 1.0))
        if log:
            job.log.append(log)
        self.bus.publish({"type": "job", "job": job.to_dict()})

    # -------------------------------------------------------- signal / calibration
    def job_test_signal(self) -> Job:
        def work(job: Job) -> None:
            self._progress(job, "Signal test", "Playing a test note ...", 0.1)
            dev = self.open_device()
            dev.cancel_event = job.cancel
            try:
                cal = Calibrator(dev, self.physics()[0])
                cal.step_signal()
                step = cal.result.steps[-1]
                job.data["signal"] = {"ok": step["ok"], "message": step["message"], **step["data"]}
                self._progress(job, "Signal test", step["message"], 1.0, step["message"])
            finally:
                self._close_device(dev)

        return self.start_job("signal", "Signal test", work)

    def job_calibrate(self) -> Job:
        mode = self.mode

        def work(job: Job) -> None:
            dev = self.open_device()
            dev.cancel_event = job.cancel
            try:
                if not dev.is_virtual:
                    dev.reset_controllers()  # type: ignore[attr-defined]

                def report(info: dict) -> None:
                    line = f"{info['name']}: {info['message']}"
                    job.data.setdefault("steps", []).append({"name": info["name"], "ok": info["ok"],
                                                              "message": info["message"]})
                    self._progress(job, "Calibration", info["name"], info["step"] / max(info["total"], 1), line)

                self._progress(job, "Calibration", "Measuring the instrument ...", 0.0)
                result = Calibrator(dev, Physics(), report).run()
                data = result.to_dict()
                data["date"] = time.strftime("%Y-%m-%d %H:%M")
                data["device"] = {"mode": mode, "midi_out": self.settings["midi_out"],
                                  "audio_device": self.settings["audio_device"],
                                  "input_channel": self.settings["input_channel"], "samplerate": dev.sr}
                self.calibration[mode] = data
                self._write_json(self._cal_file(mode), data)
                self._progress(job, "Calibration", "Calibration finished.", 1.0)
            finally:
                self._close_device(dev)

        return self.start_job("calibration", "Calibration", work)

    # ---------------------------------------------------------------- reference
    def set_reference_file(self, filename: str, payload: bytes) -> dict:
        if self.job is not None and self.job.status == "running":
            raise Busy("Wait for the running job to finish (or stop it) before loading another file.")
        name = re.sub(r"[^A-Za-z0-9._ -]", "_", os.path.basename(filename))[:120] or "reference"
        for old in os.listdir(self.path("reference")):
            try:
                os.remove(self.path("reference", old))
            except OSError:
                pass
        path = self.path("reference", name)
        with open(path, "wb") as f:
            f.write(payload)
        data, sr = load_audio(path)
        if len(data) < sr // 4:
            raise AudioError("The file is shorter than a quarter of a second.")
        mono = to_mono(data)
        n_bins = 1400
        size = max(1, len(mono) // n_bins)
        trimmed = mono[:size * (len(mono) // size)].reshape(-1, size)
        scale = float(np.max(np.abs(mono))) or 1.0
        peaks = np.stack([trimmed.min(axis=1), trimmed.max(axis=1)], axis=1) / scale
        self.ref = {"filename": name, "path": path, "sr": sr, "raw": data, "duration": len(mono) / sr,
                    "peaks": np.round(peaks * 100).astype(int).tolist(), "channels": int(data.shape[1]),
                    "peak_db": round(peak_db(mono), 1), "analysis": None}
        self.bus.publish({"type": "state"})
        return self.reference_summary()

    def reference_summary(self) -> Optional[dict]:
        ref = self.ref
        if ref is None:
            return None
        out = {k: ref[k] for k in ("filename", "sr", "duration", "peaks", "channels", "peak_db")}
        an = ref.get("analysis")
        if an:
            out["analysis"] = {k: an[k] for k in ("mode", "method", "note", "tuning_cents", "notes", "excerpt",
                                                  "range", "phrase", "version", "reliability")}
        else:
            out["analysis"] = None
        return out

    def job_analyse(self, options: dict) -> Job:
        if self.ref is None:
            raise UserError("Load a reference recording first.")
        mode = "mix" if options.get("mode") == "mix" else "stem"
        method = str(options.get("method", "auto"))
        sensitivity = float(min(max(float(options.get("sensitivity", 0.4)), 0.0), 1.0))
        region = options.get("region")
        transpose = int(options.get("transpose", 0))

        def work(job: Job) -> None:
            ref = self.ref
            sr = ref["sr"]
            raw = ref["raw"]
            duration = ref["duration"]
            t0, t1 = 0.0, duration
            explicit = False
            if region and len(region) == 2 and float(region[1]) - float(region[0]) >= 0.5:
                t0 = max(0.0, float(region[0]))
                t1 = min(duration, float(region[1]))
                explicit = (t1 - t0) <= 8.0
            # what to analyse: the selection with some context, or a manageable part of a long mix
            a0, a1 = max(0.0, t0 - 1.0), min(duration, t1 + 1.0)
            if mode == "mix" and not region and duration > 45.0:
                a0, a1 = self._loudest_low_window(to_mono(raw), sr, 30.0)
                t0, t1 = a0, a1
                self._progress(job, log=f"Long mix without a selection: analysing {a0:.0f}-{a1:.0f} s "
                                        "(where the low end is strongest).")
            seg = raw[int(a0 * sr):int(a1 * sr)]
            self._progress(job, "Analysis", "Preparing the audio ...", 0.05)
            reliability = None
            note = ""
            used = "none"
            if mode == "mix":
                sep = separate_bass(seg, sr, method, lambda m: self._progress(job, "Analysis", m, 0.2, m))
                bass, reliability, note, used = sep.bass, sep.reliability, sep.note, sep.method
            else:
                bass = resample(to_mono(seg), sr, ANALYSIS_SR)
            if job.cancel.is_set():
                raise Cancelled()
            self._progress(job, "Analysis", "Finding the notes ...", 0.6)
            notes, info = transcribe(bass, ANALYSIS_SR, sensitivity)
            if not notes:
                raise UserError("No bass notes were found in this part of the recording. Try another part, or "
                                "check that the right kind of reference (isolated bass / full mix) is selected.")
            # times are relative to the analysed segment
            r0, r1 = t0 - a0, t1 - a0
            if explicit:
                inside = [n for n in notes if r0 - 0.02 <= n.start < r1 - 0.05]
                if not inside:
                    raise UserError("The selected part contains no complete note. Select a slightly longer part.")
                e0 = max(0.0, min(r0, inside[0].start - 0.06))
                nxt = [n.start for n in notes if n.start >= r1 - 0.05]
                e1 = min(r1, (nxt[0] - 0.015) if nxt else r1, len(bass) / ANALYSIS_SR)
                e1 = max(e1, inside[-1].start + 0.1)
            else:
                pool = [n for n in notes if r0 - 0.02 <= n.start < r1]
                shifted = [RefNote(n.start - r0, n.end - r0, n.note, n.level_db, n.cents) for n in pool]
                e0, e1 = choose_excerpt(shifted, r1 - r0)
                e0, e1 = e0 + r0, e1 + r0
            reference = build_reference(bass, notes, e0, e1, info["tuning_cents"], reliability, transpose)
            target = Target(reference.audio, notes=reference.notes, band_reliability=reliability)
            version = int(time.time() * 1000) % 100000000
            write_wav(self.path("reference", "excerpt.wav"), normalize_peak(reference.audio), ANALYSIS_SR)
            with open(self.path("reference", "excerpt.png"), "wb") as f:
                f.write(spectrogram_png(target.ref_a, target.floor_a, target.floor_a + 60.0))
            ref["analysis"] = {
                "mode": mode, "method": used, "note": note, "tuning_cents": round(info["tuning_cents"], 1),
                "notes": [[round(n.start + a0, 3), round(n.end + a0, 3), n.note, round(n.level_db, 1)] for n in notes],
                "excerpt": [round(e0 + a0, 3), round(e1 + a0, 3)], "range": [round(a0, 3), round(a1, 3)],
                "phrase": [[round(n.start, 3), round(n.end, 3), n.note, note_name(n.note)]
                           for n in reference.phrase.notes],
                "reliability": reliability.tolist() if reliability is not None else None,
                "version": version, "reference": reference, "target": target,
            }
            self._progress(job, "Analysis",
                           f"{len(reference.phrase.notes)} notes in the excerpt "
                           f"({reference.end - reference.start:.1f} s).", 1.0)

        return self.start_job("analysis", "Reference analysis", work)

    @staticmethod
    def _loudest_low_window(mono: np.ndarray, sr: int, length: float) -> Tuple[float, float]:
        import scipy.signal

        dec = max(1, sr // 2000)
        low = scipy.signal.decimate(mono.astype(np.float64), dec, ftype="fir") if dec > 1 else mono
        sos = scipy.signal.butter(2, 250.0, fs=sr / dec, output="sos")
        low = scipy.signal.sosfilt(sos, low)
        per_s = int(sr / dec)
        n = len(low) // per_s
        if n <= length:
            return 0.0, len(mono) / sr
        energy = np.square(low[:n * per_s]).reshape(n, per_s).sum(axis=1)
        win = int(length)
        sums = np.convolve(energy, np.ones(win), mode="valid")
        start = int(np.argmax(sums))
        return float(start), float(start + win)

    # -------------------------------------------------------------------- match
    def job_match(self, options: dict) -> Job:
        if self.ref is None or not self.ref.get("analysis"):
            raise UserError("Load and analyse a reference first.")
        effort_key = str(options.get("effort", self.settings["effort"]))
        effort = EFFORTS.get(effort_key, EFFORTS["normal"])
        keep_tuning = bool(options.get("keep_tuning", False))
        analysis = self.ref["analysis"]
        reference: Reference = analysis["reference"]
        target: Target = analysis["target"]
        mode = self.mode

        def work(job: Job) -> None:
            t_begin = time.time()
            job.data.update(history=[], stage="prepare", candidates=[])
            self._progress(job, "Preparing", "Connecting to the instrument ...", 0.01)
            dev = self.open_device()
            dev.cancel_event = job.cancel
            model: Optional[Model] = None
            try:
                physics, calibrated = self.physics()
                if not calibrated:
                    if dev.is_virtual:
                        self._progress(job, "Preparing", "First run: calibrating the virtual synth ...", 0.02)
                        result = Calibrator(dev, Physics()).run()
                        data = result.to_dict()
                        data["date"] = time.strftime("%Y-%m-%d %H:%M")
                        data["device"] = {"mode": "demo"}
                        self.calibration["demo"] = data
                        self._write_json(self._cal_file("demo"), data)
                        physics = result.physics
                    else:
                        raise UserError("Calibrate the instrument first (step 1). The calibration teaches the app "
                                        "how this Bass Station II responds; without it the match is much worse.")
                if not dev.is_virtual:
                    dev.reset_controllers()  # type: ignore[attr-defined]
                # level and latency can change between sessions: measure them again (a few notes)
                check = Calibrator(dev, physics)
                check.step_signal()
                self._progress(job, "Preparing", check.result.steps[-1]["message"], 0.04,
                               "Signal: " + check.result.steps[-1]["message"])
                frozen: Dict[str, int] = {}
                cal_flags = physics.calibrated
                if cal_flags.get("waveform_control") is False:
                    frozen.update(osc1_wave=2, osc2_wave=2)
                    self._progress(job, log="Waveforms cannot be switched over MIDI on this unit: keeping sawtooth.")
                if cal_flags.get("filter_track_control") is False:
                    frozen.update(filter_track=0)
                dev.fine_offset = int(round(reference.tuning_cents - physics.tune_cents)) if keep_tuning else 0
                dev.fine_offset = int(min(max(dev.fine_offset, -60), 60))
                model_physics = physics.copy()
                model_physics.tune_cents = physics.tune_cents + dev.fine_offset
                model = Model(model_physics, reference.phrase, target)

                # ---- stage 1: search in the model
                job.data["stage"] = "model"
                self._progress(job, "Model search", "Trying oscillator and filter combinations ...", 0.06)
                last = [0.0]

                def model_progress(info: dict) -> None:
                    if time.time() - last[0] < 0.4:
                        return
                    last[0] = time.time()
                    phase = str(info.get("phase", ""))
                    best = info.get("best")
                    leader = info.get("leader")
                    if isinstance(best, float) and best == best:
                        job.data["model_best"] = round(best, 4)
                    if leader:
                        job.data["leader"] = describe_structure(leader)
                    elapsed = time.time() - t_begin
                    self._progress(job, "Model search", f"{phase} - {job.data.get('leader', '')}",
                                   0.06 + 0.34 * min(1.0, elapsed / (75.0 * effort["model"])))

                search = ModelSearch(model, seed=1, fixed=frozen, progress=model_progress,
                                     should_stop=job.cancel.is_set)
                found = search.run(effort=effort["model"])
                if job.cancel.is_set():
                    raise Cancelled()
                job.data["candidates"] = [{"label": c.label(), "loss": round(c.loss, 4)} for c in found.candidates[:4]]
                self._progress(job, "Model search", "Model search finished.", 0.4,
                               "Model candidates: " + "; ".join(f"{c.label()} ({c.loss:.3f})"
                                                                for c in found.candidates[:3]))

                # ---- stage 2: refine on the instrument
                job.data["stage"] = "refine"
                budget = int(effort["plays"])

                def loop_progress(info: dict) -> None:
                    job.data["history"].append([info["play"], round(info["loss"], 4), round(info["best"], 4)])
                    job.data["leader"] = describe_structure(info["leader"])
                    job.data["best"] = round(info["best"], 4)
                    self._progress(job, "Refining on the synth", f"{info['phase']}: {info['note']}",
                                   0.4 + 0.58 * min(1.0, info["play"] / budget))

                loop = HardwareLoop(dev, model, target, reference.phrase, loop_progress, job.cancel.is_set,
                                    seed=2, frozen=list(frozen))
                out = loop.run([c.patch for c in found.candidates], budget=budget)
                final = clean_patch(out.patch)
                final_tuned = apply_tuning(final, dev.fine_offset)
                dev.set_patch(final_tuned, force=True)
                dump = None
                try:
                    dump = dev.dump_patch()
                except Exception:
                    dump = None
                self._store_result(job, reference, target, final_tuned, out, found, dump, loop.clipped,
                                   time.time() - t_begin, effort_key, keep_tuning, dev.fine_offset, mode)
                self._progress(job, "Done", f"Finished: distance {out.loss:.3f} ({quality_label(out.loss)}).", 1.0)
            finally:
                if model is not None:
                    model.close()
                dev.fine_offset = 0
                self._close_device(dev)

        return self.start_job("match", "Matching", work)

    def _store_result(self, job: Job, reference: Reference, target: Target, patch: Dict[str, int], out,
                      found, dump: Optional[bytes], clipped: bool, seconds: float, effort: str, keep_tuning: bool,
                      fine_offset: int, mode: str) -> None:
        sid = time.strftime("%Y%m%d-%H%M%S")
        folder = self.path("sessions", sid)
        os.makedirs(folder, exist_ok=True)
        audio = out.audio if out.audio is not None else np.zeros(target.n_samples, dtype=np.float32)
        ref_rms = float(np.sqrt(np.mean(np.square(reference.audio, dtype=np.float64)))) + 1e-9
        syn_rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) + 1e-9
        ref_out = normalize_peak(reference.audio, 0.8)
        gain = (float(np.sqrt(np.mean(np.square(ref_out, dtype=np.float64)))) + 1e-9) / syn_rms
        syn_out = np.clip(audio * gain, -1.0, 1.0)  # same loudness for a fair A/B
        write_wav(os.path.join(folder, "reference.wav"), ref_out, ANALYSIS_SR)
        write_wav(os.path.join(folder, "synth.wav"), syn_out, ANALYSIS_SR)
        al = target.aligned(extract(audio, target.cfg))
        with open(os.path.join(folder, "reference.png"), "wb") as f:
            f.write(spectrogram_png(target.ref_a, target.floor_a, target.floor_a + 60.0))
        with open(os.path.join(folder, "synth.png"), "wb") as f:
            f.write(spectrogram_png(al.a, target.floor_a, target.floor_a + 60.0))
        title = f"BS2 Autotune {sid}"
        with open(os.path.join(folder, "patch.txt"), "w", encoding="utf-8") as f:
            f.write(patch_text(patch, title))
        if dump:
            with open(os.path.join(folder, "patch.syx"), "wb") as f:
                f.write(dump_with_name(dump, "Autotune " + sid[-6:]))
        result = {
            "id": sid, "patch": patch, "loss": round(out.loss, 4), "quality": quality_label(out.loss),
            "parts": {k: round(float(v), 4) for k, v in out.parts.items() if isinstance(v, (int, float))},
            "start_loss": round(out.start_loss, 4), "noise": round(out.noise, 4), "plays": out.plays,
            "history": [[h["play"], round(h["loss"], 4), round(h["best"], 4), h["phase"]] for h in out.history],
            "structure": describe_structure(patch), "seconds": round(seconds, 1), "effort": effort,
            "keep_tuning": keep_tuning, "fine_offset": fine_offset, "tuning_cents": round(reference.tuning_cents, 1),
            "clipped": bool(clipped), "has_syx": bool(dump), "mode": mode,
            "candidates": [{"label": c.label(), "loss": round(c.loss, 4)} for c in found.candidates[:4]],
            "phrase": reference.phrase.to_dict(), "reference_file": self.ref["filename"] if self.ref else "",
            "date": time.strftime("%Y-%m-%d %H:%M"),
        }
        with open(os.path.join(folder, "result.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=1)
        self.result = result
        job.data["result_id"] = sid

    def result_summary(self) -> Optional[dict]:
        if self.result is None:
            return None
        out = dict(self.result)
        out["rows"] = describe_patch(out["patch"])
        return out

    # ------------------------------------------------------------ patch actions
    def job_send_patch(self, patch: Dict[str, int], play: bool = False) -> Job:
        patch = normalize_patch(patch)
        phrase = self._phrase_for_audition()

        def work(job: Job) -> None:
            self._progress(job, "Sending", "Sending the patch to the synth ...", 0.2)
            dev = self.open_device()
            dev.cancel_event = job.cancel
            try:
                dev.set_patch(patch, force=True)
                if play:
                    self._progress(job, "Playing", "Playing the phrase ...", 0.5)
                    if dev.is_virtual and not self.calibration.get("demo"):
                        dev.latency = getattr(dev, "true_latency", 0.0)
                    elif dev.is_virtual:
                        dev.latency = float(self.calibration["demo"].get("latency", 0.0))
                    y = dev.capture(patch, phrase)
                    write_wav(self.path("reference", "audition.wav"), normalize_peak(y, 0.8), ANALYSIS_SR)
                    job.data["audition"] = int(time.time() * 1000) % 100000000
                self._progress(job, "Done", "Patch sent." if not play else "Played.", 1.0)
            finally:
                self._close_device(dev)

        return self.start_job("send", "Send patch", work)

    def _phrase_for_audition(self) -> Phrase:
        if self.result and self.result.get("phrase"):
            return Phrase.from_dict(self.result["phrase"])
        if self.ref and self.ref.get("analysis"):
            return self.ref["analysis"]["reference"].phrase
        notes = [Note(0.05, 0.45, 36), Note(0.5, 0.9, 36), Note(1.0, 1.4, 39), Note(1.5, 2.3, 34)]
        return Phrase(notes, 2.8)

    # ------------------------------------------------------------------ library
    def library(self) -> List[dict]:
        items = []
        for name in sorted(os.listdir(self.path("library")), reverse=True):
            if name.endswith(".json"):
                data = self._read_json(os.path.join("library", name))
                if data and "patch" in data:
                    items.append({"id": name[:-5], "name": data.get("name", name[:-5]), "date": data.get("date", ""),
                                  "loss": data.get("loss"), "structure": data.get("structure", ""),
                                  "reference": data.get("reference", "")})
        return items

    def library_save(self, name: str, patch: Optional[Dict[str, int]] = None) -> dict:
        if patch is None:
            if self.result is None:
                raise UserError("There is no result to save yet.")
            patch = self.result["patch"]
        patch = normalize_patch(patch)
        clean = re.sub(r"[^A-Za-z0-9 _-]", "", name).strip()[:40] or "patch"
        pid = f"{time.strftime('%Y%m%d-%H%M%S')}-{re.sub(r'[^A-Za-z0-9]+', '-', clean).strip('-').lower() or 'patch'}"
        data = {"name": clean, "patch": patch, "date": time.strftime("%Y-%m-%d %H:%M"),
                "structure": describe_structure(patch)}
        if self.result is not None and self.result["patch"] == patch:
            data.update(loss=self.result["loss"], reference=self.result.get("reference_file", ""),
                        session=self.result["id"])
        self._write_json(os.path.join("library", pid + ".json"), data)
        self.bus.publish({"type": "state"})
        return {"id": pid, **data}

    def library_get(self, pid: str) -> dict:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", pid):
            raise UserError("Unknown patch.")
        data = self._read_json(os.path.join("library", pid + ".json"))
        if not data:
            raise UserError("Unknown patch.")
        data["rows"] = describe_patch(data["patch"])
        data["id"] = pid
        return data

    def library_delete(self, pid: str) -> None:
        if re.fullmatch(r"[A-Za-z0-9_-]+", pid):
            try:
                os.remove(self.path("library", pid + ".json"))
            except OSError:
                pass
        self.bus.publish({"type": "state"})

    # -------------------------------------------------------------------- state
    def state(self) -> dict:
        return {
            "version": self.version, "settings": dict(self.settings), "mode": self.mode,
            "efforts": {k: {"label": v["label"], "plays": v["plays"]} for k, v in EFFORTS.items()},
            "calibration": self.calibration_summary(),
            "reference": self.reference_summary(),
            "job": self.job.to_dict() if self.job is not None else None,
            "result": self.result_summary(),
            "library": self.library(),
        }
