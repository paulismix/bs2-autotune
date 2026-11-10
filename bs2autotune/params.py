"""Novation Bass Station II parameter map.

Every parameter the app can set over MIDI is described here once: its MIDI
address, raw value range, the value it has in the factory INIT patch and how
the synth itself displays it.

Sources (all cross-checked against each other):

* midi.guide dataset (pencilresearch/midi, "Novation/Bass Station II.csv")
* francoisgeorgy/BS2-Web and BS2-SysEx (reverse engineered against hardware):
  CC/NRPN numbers, the two-CC encoding of 8-bit values, NRPN data in the
  Data Entry MSB, the SysEx layout and the factory INIT patch dump.

Raw values are what the synth stores:

* ``cc``   - one 7-bit controller, value 0..127
* ``cc2``  - an 8-bit value sent as two controllers: MSB = value >> 1 and
             LSB = 64 for odd values, 0 for even ones
* ``nrpn`` - NRPN with MSB 0; the value travels in Data Entry MSB (CC 6)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# display helpers (same conventions as the synth's own display)
# ---------------------------------------------------------------------------


def bipolar7(v: int) -> int:
    """0..127 -> -63..+63 (63 and 64 are both zero)."""
    return v - 63 if v < 64 else v - 64


def bipolar8(v: int) -> int:
    """0..255 -> -127..+127 (127 and 128 are both zero)."""
    return v - 127 if v < 128 else v - 128


def fine_cents(v: int) -> int:
    """Oscillator fine tune raw value -> cents (-100..+100)."""
    return max(-100, min(100, bipolar8(v)))


# raw value of the coarse control for every whole semitone (plateau positions)
COARSE_SEMI_TO_RAW: Dict[int, int] = {
    -12: 0, -11: 10, -10: 20, -9: 31, -8: 42, -7: 53, -6: 63, -5: 74, -4: 85,
    -3: 95, -2: 106, -1: 117, 0: 128, 1: 138, 2: 149, 3: 160, 4: 170, 5: 181,
    6: 192, 7: 202, 8: 213, 9: 224, 10: 235, 11: 245, 12: 255,
}

# (raw, semitones) anchors of the coarse control; two-wide plateaus sit on the
# whole semitones, the control is linear in between
_COARSE_ANCHORS: Tuple[Tuple[int, float], ...] = (
    (0, -12.0), (10, -11.0), (19, -10.0), (20, -10.0), (30, -9.0), (31, -9.0),
    (41, -8.0), (42, -8.0), (52, -7.0), (53, -7.0), (62, -6.0), (63, -6.0),
    (73, -5.0), (74, -5.0), (84, -4.0), (85, -4.0), (94, -3.0), (95, -3.0),
    (105, -2.0), (106, -2.0), (116, -1.0), (117, -1.0), (127, 0.0), (128, 0.0),
    (138, 1.0), (139, 1.0), (149, 2.0), (150, 2.0), (160, 3.0), (161, 3.0),
    (170, 4.0), (171, 4.0), (181, 5.0), (182, 5.0), (192, 6.0), (193, 6.0),
    (202, 7.0), (203, 7.0), (213, 8.0), (214, 8.0), (224, 9.0), (225, 9.0),
    (235, 10.0), (236, 10.0), (245, 11.0), (255, 12.0),
)


def coarse_semitones(v: float) -> float:
    """Oscillator coarse raw value (0..255) -> semitones (-12..+12)."""
    v = max(0.0, min(255.0, float(v)))
    prev = _COARSE_ANCHORS[0]
    for cur in _COARSE_ANCHORS[1:]:
        if v <= cur[0]:
            if cur[0] == prev[0]:
                return cur[1]
            t = (v - prev[0]) / (cur[0] - prev[0])
            return prev[1] + t * (cur[1] - prev[1])
        prev = cur
    return _COARSE_ANCHORS[-1][1]


def pulse_width_percent(v: float) -> float:
    """Pulse width raw value (0..127) -> percent (5..95, 64 = 50 %)."""
    return 5.0 + 90.0 * max(0.0, min(127.0, float(v))) / 127.0 - 0.4


def _fmt_plain(v: int) -> str:
    return str(int(v))


def _fmt_b7(v: int) -> str:
    return f"{bipolar7(int(v)):+d}"


def _fmt_b8(v: int) -> str:
    return f"{bipolar8(int(v)):+d}"


def _fmt_fine(v: int) -> str:
    return f"{fine_cents(int(v)):+d} ct"


def _fmt_coarse(v: int) -> str:
    return f"{coarse_semitones(v):+.1f} st"


def _fmt_pw(v: int) -> str:
    return f"{round(pulse_width_percent(v))} %"


# ---------------------------------------------------------------------------
# parameter definition
# ---------------------------------------------------------------------------

WAVES = ("sine", "triangle", "sawtooth", "pulse")
RANGES = ("16'", "8'", "4'", "2'")
SUB_WAVES = ("sine", "narrow pulse", "square")
SUB_OCTS = ("-2 oct", "-1 oct")
FILTER_TYPES = ("Classic", "Acid")
FILTER_SLOPES = ("12 dB", "24 dB")
FILTER_SHAPES = ("LP", "BP", "HP")
LFO_WAVES = ("triangle", "sawtooth", "square", "S+H")
ENV_TRIGS = ("Multi", "Single", "Autoglide")
OFF_ON = ("Off", "On")
LFO_MODES = ("Speed", "Sync")


@dataclass(frozen=True)
class Param:
    key: str
    section: str
    label: str
    kind: str  # "cc" | "cc2" | "nrpn"
    num: int
    lo: int
    hi: int
    init: int
    lsb: Optional[int] = None
    choices: Optional[Tuple[str, ...]] = None
    center: Optional[int] = None
    fw: Optional[str] = None  # minimum firmware that knows the parameter
    control: str = "knob"  # knob | switch | key (Function + key)
    fmt: Callable[[int], str] = _fmt_plain

    # -- values ------------------------------------------------------------
    def clamp(self, v: float) -> int:
        return int(max(self.lo, min(self.hi, int(round(float(v))))))

    @property
    def discrete(self) -> bool:
        return self.choices is not None

    def display(self, v: int) -> str:
        v = self.clamp(v)
        if self.choices is not None:
            idx = v - self.lo
            if 0 <= idx < len(self.choices):
                return self.choices[idx]
            return str(v)
        return self.fmt(v)

    def address(self) -> str:
        if self.kind == "cc":
            return f"CC {self.num}"
        if self.kind == "cc2":
            return f"CC {self.num}/{self.lsb}"
        return f"NRPN 0:{self.num}"

    # -- MIDI ----------------------------------------------------------------
    def messages(self, value: int) -> List[Tuple[int, int]]:
        """Controller messages (cc number, value) that set this parameter."""
        v = self.clamp(value)
        if self.kind == "cc":
            return [(self.num, v & 0x7F)]
        if self.kind == "cc2":
            assert self.lsb is not None
            return [(self.num, (v >> 1) & 0x7F), (self.lsb, 64 if v & 1 else 0)]
        if self.kind == "nrpn":
            # NRPN select (MSB 0), value in Data Entry MSB, then deselect
            # (RPN null) - the exact sequence the BS2-Web editor uses.
            return [(99, 0), (98, self.num & 0x7F), (6, v & 0x7F), (101, 127), (100, 127)]
        raise ValueError(self.kind)


def _p(key, section, label, kind, num, lo, hi, init, **kw) -> Param:
    return Param(key, section, label, kind, num, lo, hi, init, **kw)


def _osc(n: int, wave_nrpn: int, cc_range: int, coarse: Tuple[int, int], fine: Tuple[int, int],
         cc_env: int, lfo1: Tuple[int, int], cc_env_pw: int, cc_lfo2_pw: int, cc_pw: int) -> List[Param]:
    s = f"osc{n}"
    name = f"Osc {n}"
    return [
        _p(f"{s}_wave", s, f"{name} Waveform", "nrpn", wave_nrpn, 0, 3, 2, choices=WAVES, control="switch"),
        _p(f"{s}_range", s, f"{name} Range", "cc", cc_range, 63, 66, 64, choices=RANGES, control="switch"),
        _p(f"{s}_coarse", s, f"{name} Coarse", "cc2", coarse[0], 0, 255, 128, lsb=coarse[1], center=128,
           fmt=_fmt_coarse),
        _p(f"{s}_fine", s, f"{name} Fine", "cc2", fine[0], 27, 228, 128, lsb=fine[1], center=128, fmt=_fmt_fine),
        _p(f"{s}_pw", s, f"{name} Pulse Width", "cc", cc_pw, 0, 127, 64, fmt=_fmt_pw),
        _p(f"{s}_env_pitch", s, f"{name} Mod Env Depth", "cc", cc_env, 0, 127, 64, center=64, fmt=_fmt_b7),
        _p(f"{s}_lfo1_pitch", s, f"{name} LFO 1 Depth", "cc2", lfo1[0], 0, 255, 128, lsb=lfo1[1], center=128,
           fmt=_fmt_b8),
        _p(f"{s}_env_pw", s, f"{name} Mod Env PW", "cc", cc_env_pw, 0, 127, 64, center=64, fmt=_fmt_b7),
        _p(f"{s}_lfo2_pw", s, f"{name} LFO 2 PW", "cc", cc_lfo2_pw, 0, 127, 64, center=64, fmt=_fmt_b7),
    ]


PARAM_LIST: List[Param] = [
    # --- oscillators -------------------------------------------------------
    *_osc(1, 72, 70, (27, 59), (26, 58), 71, (28, 60), 72, 73, 74),
    *_osc(2, 82, 75, (30, 62), (29, 61), 76, (31, 63), 77, 78, 79),
    _p("osc_sync", "osc2", "Osc 1-2 Sync", "cc", 110, 0, 1, 0, choices=OFF_ON, control="key"),
    _p("sub_wave", "sub", "Sub Osc Wave", "cc", 80, 0, 2, 0, choices=SUB_WAVES, control="switch"),
    _p("sub_oct", "sub", "Sub Osc Octave", "cc", 81, 62, 63, 63, choices=SUB_OCTS, control="switch"),
    # --- mixer -------------------------------------------------------------
    _p("mix_osc1", "mixer", "Osc 1 Level", "cc2", 20, 0, 255, 255, lsb=52),
    _p("mix_osc2", "mixer", "Osc 2 Level", "cc2", 21, 0, 255, 0, lsb=53),
    _p("mix_sub", "mixer", "Sub Osc Level", "cc2", 22, 0, 255, 0, lsb=54),
    _p("mix_noise", "mixer", "Noise Level", "cc2", 23, 0, 255, 0, lsb=55),
    _p("mix_ring", "mixer", "Ring Mod Level", "cc2", 24, 0, 255, 0, lsb=56),
    _p("mix_ext", "mixer", "External Level", "cc2", 25, 0, 255, 0, lsb=57),
    # --- filter ------------------------------------------------------------
    _p("filter_type", "filter", "Filter Type", "cc", 83, 0, 1, 0, choices=FILTER_TYPES, control="switch"),
    _p("filter_slope", "filter", "Filter Slope", "cc", 106, 0, 1, 1, choices=FILTER_SLOPES, control="switch"),
    _p("filter_shape", "filter", "Filter Shape", "cc", 84, 0, 2, 0, choices=FILTER_SHAPES, control="switch"),
    _p("filter_freq", "filter", "Filter Frequency", "cc2", 16, 0, 255, 255, lsb=48),
    _p("filter_res", "filter", "Resonance", "cc", 82, 0, 127, 0),
    _p("filter_drive", "filter", "Overdrive", "cc", 114, 0, 127, 0),
    _p("filter_env", "filter", "Mod Env Depth", "cc", 85, 0, 127, 64, center=64, fmt=_fmt_b7),
    _p("filter_lfo2", "filter", "LFO 2 Depth", "cc2", 17, 0, 255, 128, lsb=49, center=128, fmt=_fmt_b8),
    _p("filter_track", "filter", "Filter Tracking", "nrpn", 108, 0, 7, 0, fw="2.5", control="key"),
    # --- envelopes ---------------------------------------------------------
    _p("amp_attack", "amp_env", "Amp Attack", "cc", 90, 0, 127, 0),
    _p("amp_decay", "amp_env", "Amp Decay", "cc", 91, 0, 127, 0),
    _p("amp_sustain", "amp_env", "Amp Sustain", "cc", 92, 0, 127, 127),
    _p("amp_release", "amp_env", "Amp Release", "cc", 93, 0, 127, 0),
    _p("amp_trig", "amp_env", "Amp Env Triggering", "nrpn", 73, 0, 2, 0, choices=ENV_TRIGS, control="switch"),
    _p("amp_retrig", "amp_env", "Amp Env Retrigger", "nrpn", 109, 0, 1, 0, choices=OFF_ON, fw="2.5",
       control="key"),
    _p("mod_attack", "mod_env", "Mod Attack", "cc", 102, 0, 127, 0),
    _p("mod_decay", "mod_env", "Mod Decay", "cc", 103, 0, 127, 0),
    _p("mod_sustain", "mod_env", "Mod Sustain", "cc", 104, 0, 127, 127),
    _p("mod_release", "mod_env", "Mod Release", "cc", 105, 0, 127, 0),
    _p("mod_trig", "mod_env", "Mod Env Triggering", "nrpn", 105, 0, 2, 0, choices=ENV_TRIGS, control="switch"),
    _p("mod_retrig", "mod_env", "Mod Env Retrigger", "nrpn", 110, 0, 1, 0, choices=OFF_ON, fw="2.5",
       control="key"),
    _p("vel_amp", "amp_env", "Velocity > Amp Env", "cc", 112, 0, 127, 64, center=64, control="key",
       fmt=lambda v: f"{int(v) - 64:+d}"),
    _p("vel_mod", "mod_env", "Velocity > Mod Env", "cc", 113, 0, 127, 64, center=64, control="key",
       fmt=lambda v: f"{int(v) - 64:+d}"),
    # --- LFOs --------------------------------------------------------------
    _p("lfo1_wave", "lfo1", "LFO 1 Wave", "cc", 88, 0, 3, 0, choices=LFO_WAVES, control="switch"),
    _p("lfo1_speed", "lfo1", "LFO 1 Speed", "cc2", 18, 0, 255, 75, lsb=50),
    _p("lfo1_delay", "lfo1", "LFO 1 Delay", "cc", 86, 0, 127, 0),
    _p("lfo1_slew", "lfo1", "LFO 1 Slew", "nrpn", 86, 0, 127, 0, control="key"),
    _p("lfo1_mode", "lfo1", "LFO 1 Speed/Sync", "nrpn", 88, 0, 1, 0, choices=LFO_MODES, control="key"),
    _p("lfo1_keysync", "lfo1", "LFO 1 Key Sync", "nrpn", 89, 0, 1, 0, choices=OFF_ON, control="key"),
    _p("lfo2_wave", "lfo2", "LFO 2 Wave", "cc", 89, 0, 3, 0, choices=LFO_WAVES, control="switch"),
    _p("lfo2_speed", "lfo2", "LFO 2 Speed", "cc2", 19, 0, 255, 52, lsb=51),
    _p("lfo2_delay", "lfo2", "LFO 2 Delay", "cc", 87, 0, 127, 0),
    _p("lfo2_slew", "lfo2", "LFO 2 Slew", "nrpn", 90, 0, 127, 0, control="key"),
    _p("lfo2_mode", "lfo2", "LFO 2 Speed/Sync", "nrpn", 92, 0, 1, 0, choices=LFO_MODES, control="key"),
    _p("lfo2_keysync", "lfo2", "LFO 2 Key Sync", "nrpn", 93, 0, 1, 1, choices=OFF_ON, control="key"),
    # --- effects / global --------------------------------------------------
    _p("distortion", "fx", "Distortion", "cc", 94, 0, 127, 0),
    _p("osc_filter_mod", "fx", "Osc Filter Mod", "cc", 115, 0, 127, 0),
    _p("limiter", "fx", "VCA Limit", "cc", 95, 0, 127, 0, control="key"),
    _p("portamento", "global", "Glide Time", "cc", 5, 0, 127, 0),
    _p("arp_on", "global", "Arpeggiator", "cc", 108, 0, 1, 0, choices=OFF_ON, control="switch"),
    _p("arp_latch", "global", "Arp Latch", "cc", 109, 0, 1, 0, choices=OFF_ON, control="switch"),
    _p("paraphonic", "global", "Paraphonic Mode", "nrpn", 107, 0, 1, 0, choices=OFF_ON, fw="2.5", control="key"),
    _p("osc_error", "global", "Oscillator Error", "nrpn", 111, 0, 7, 0, fw="2.5", control="key"),
    _p("sub_coarse", "sub", "Sub Osc Coarse", "nrpn", 84, 0, 127, 0, fw="4.14", control="key"),
    _p("sub_fine", "sub", "Sub Osc Fine", "nrpn", 77, 0, 127, 0, fw="4.14", control="key"),
    _p("glide_diverge", "global", "Glide Divergence", "nrpn", 113, 0, 15, 0, fw="4.14", control="key"),
    _p("amp_fixed", "amp_env", "Amp Env Fixed Duration", "nrpn", 114, 0, 1, 0, choices=OFF_ON, fw="4.14",
       control="key"),
    _p("mod_fixed", "mod_env", "Mod Env Fixed Duration", "nrpn", 115, 0, 1, 0, choices=OFF_ON, fw="4.14",
       control="key"),
    _p("overlay_bank", "global", "AFX Overlay Bank", "nrpn", 112, 0, 8, 0, fw="4.14", control="key"),
]

PARAMS: Dict[str, Param] = {p.key: p for p in PARAM_LIST}
assert len(PARAMS) == len(PARAM_LIST), "duplicate parameter key"

SECTIONS: Tuple[Tuple[str, str], ...] = (
    ("osc1", "Oscillator 1"),
    ("osc2", "Oscillator 2"),
    ("sub", "Sub Oscillator"),
    ("mixer", "Mixer"),
    ("filter", "Filter"),
    ("amp_env", "Amp Envelope"),
    ("mod_env", "Mod Envelope"),
    ("lfo1", "LFO 1"),
    ("lfo2", "LFO 2"),
    ("fx", "Effects"),
    ("global", "Global"),
)

# ---------------------------------------------------------------------------
# patches
# ---------------------------------------------------------------------------

Patch = Dict[str, int]


def init_patch() -> Patch:
    """The factory INIT patch (decoded from the factory dump)."""
    return {p.key: p.init for p in PARAM_LIST}


def baseline_patch() -> Patch:
    """INIT patch adjusted so that every evaluation is repeatable.

    The LFOs restart with every note (key sync on) so that a patch using them
    sounds the same each time it is measured.
    """
    patch = init_patch()
    patch["lfo1_keysync"] = 1
    patch["lfo2_keysync"] = 1
    return patch


def normalize_patch(patch: Dict[str, float]) -> Patch:
    """Complete a (partial) patch with baseline values and clamp everything."""
    out = baseline_patch()
    for key, value in patch.items():
        if key in PARAMS:
            out[key] = PARAMS[key].clamp(value)
    return out


def patch_messages(patch: Dict[str, int], keys: Optional[Iterable[str]] = None,
                   max_fw: Optional[str] = None) -> List[Tuple[str, List[Tuple[int, int]]]]:
    """(key, controller messages) for the given keys in a safe sending order."""
    order = list(keys) if keys is not None else [p.key for p in PARAM_LIST]
    out = []
    for key in order:
        p = PARAMS[key]
        if key not in patch:
            continue
        if max_fw is not None and p.fw is not None and fw_tuple(p.fw) > fw_tuple(max_fw):
            continue
        out.append((key, p.messages(patch[key])))
    return out


def fw_tuple(version: str) -> Tuple[int, ...]:
    return tuple(int(x) for x in version.split("."))


def describe_patch(patch: Dict[str, int], only_changed: bool = False) -> List[dict]:
    """Rows for a human readable patch sheet, grouped by panel section."""
    rows = []
    base = init_patch()
    for section, title in SECTIONS:
        for p in PARAM_LIST:
            if p.section != section or p.key not in patch:
                continue
            v = p.clamp(patch[p.key])
            changed = v != base[p.key]
            if only_changed and not changed:
                continue
            rows.append({
                "key": p.key,
                "section": section,
                "section_title": title,
                "label": p.label,
                "raw": v,
                "display": p.display(v),
                "address": p.address(),
                "control": p.control,
                "changed": changed,
                "lo": p.lo,
                "hi": p.hi,
                "init": p.init,
                "center": p.center,
                "discrete": p.discrete,
                "fw": p.fw,
            })
    return rows


def patch_text(patch: Dict[str, int], title: str = "BS2 patch") -> str:
    """Plain text patch sheet (values as the synth displays them)."""
    lines = [title, "=" * len(title), ""]
    current = None
    for row in describe_patch(patch):
        if row["section_title"] != current:
            current = row["section_title"]
            lines.append("")
            lines.append(f"[{current}]")
        mark = "*" if row["changed"] else " "
        lines.append(f" {mark} {row['label']:<26} {row['display']:>14}   (raw {row['raw']:>3}, {row['address']})")
    lines.append("")
    lines.append("* = differs from the INIT patch")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# SysEx (only what is needed to back up / restore the edit buffer)
# ---------------------------------------------------------------------------

SYSEX_HEADER: Tuple[int, ...] = (0xF0, 0x00, 0x20, 0x29, 0x00, 0x33, 0x00)
SYSEX_REQUEST_CURRENT: Tuple[int, ...] = SYSEX_HEADER + (0x40, 0xF7)
SYSEX_PATCH_LENGTH = 154
SYSEX_NAME_OFFSET = 137
SYSEX_NAME_LENGTH = 16


def is_patch_dump(data: Sequence[int]) -> bool:
    """True for a single-patch dump as sent by the synth."""
    data = list(data)
    return (
        len(data) >= 100
        and tuple(data[:6]) == SYSEX_HEADER[:6]
        and data[-1] == 0xF7
        and all(0 <= b < 0x80 for b in data[1:-1])
    )


def dump_with_name(data: Sequence[int], name: str) -> bytes:
    """Copy of a patch dump with the 16 character patch name replaced."""
    out = bytearray(data)
    if len(out) >= SYSEX_NAME_OFFSET + SYSEX_NAME_LENGTH + 1:
        clean = "".join(ch if 32 <= ord(ch) < 127 else " " for ch in name)[:SYSEX_NAME_LENGTH]
        clean = clean.ljust(SYSEX_NAME_LENGTH)
        out[SYSEX_NAME_OFFSET:SYSEX_NAME_OFFSET + SYSEX_NAME_LENGTH] = clean.encode("ascii")
    return bytes(out)
