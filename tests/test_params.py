"""The MIDI map: encoding, the factory INIT patch, display values."""

import pytest

from bs2autotune.params import (COARSE_SEMI_TO_RAW, PARAM_LIST, PARAMS, SYSEX_REQUEST_CURRENT, baseline_patch,
                                coarse_semitones, describe_patch, dump_with_name, fine_cents, init_patch,
                                is_patch_dump, normalize_patch, patch_text, pulse_width_percent)
from tests.fakes import MidiDecoder

# The factory INIT patch dump as documented by the BS2-SysEx project.
INIT_DUMP = bytes.fromhex("""
    f0 00 20 29 00 33 00 00 00 00 00 00 00 00 01 00
    4c 00 00 48 04 04 02 00 02 20 10 10 08 00 01 00
    43 40 20 00 03 7f 7c 00 00 00 00 00 0f 78 00 00
    08 20 00 00 07 78 00 00 40 00 00 0f 70 00 00 00
    00 00 12 63 10 00 00 00 00 1a 06 20 20 20 04 00
    1f 19 10 09 24 02 01 14 40 20 20 10 08 02 01 00
    40 40 10 08 04 02 01 00 40 20 00 00 00 03 10 00
    00 00 00 00 00 00 00 00 00 f7
""")

# (offset, masks) of a few fields in that dump (community reverse engineering)
FIELDS = {
    "osc1_wave": (19, (0x60,)), "osc1_pw": (19, (0x0F, 0x70)), "osc1_range": (20, (0x07, 0x78)),
    "osc1_coarse": (21, (0x07, 0x7C)), "osc1_fine": (22, (0x03, 0x7E)), "osc2_wave": (24, (0x03,)),
    "osc2_coarse": (27, (0x1F, 0x70)), "osc2_fine": (28, (0x0F, 0x78)), "mix_osc1": (37, (0x07, 0x7C)),
    "mix_osc2": (38, (0x03, 0x7E)), "mix_sub": (39, (0x01, 0x7F)), "filter_freq": (44, (0x0F, 0x78)),
    "filter_res": (45, (0x03, 0x7C)), "filter_drive": (46, (0x01, 0x7E)), "vel_amp": (49, (0x3F, 0x40)),
    "amp_attack": (50, (0x1F, 0x60)), "amp_decay": (51, (0x0F, 0x70)), "amp_sustain": (52, (0x07, 0x78)),
    "amp_release": (53, (0x03, 0x7C)), "vel_mod": (56, (0x7F,)), "mod_sustain": (59, (0x0F, 0x70)),
    "lfo1_speed": (66, (0x3F, 0x60)), "lfo2_speed": (73, (0x7F, 0x40)), "osc1_lfo1_pitch": (90, (0x3F, 0x60)),
    "filter_lfo2": (97, (0x7F, 0x40)), "osc1_env_pitch": (98, (0x1F, 0x60)), "filter_env": (105, (0x3F, 0x40)),
    "distortion": (107, (0x0F, 0x70)), "filter_track": (112, (0x07,)),
}


def _field(data, offset, masks):
    def shift(m):
        n = 0
        while m and not m & 1:
            m >>= 1
            n += 1
        return n

    if len(masks) == 1:
        return (data[offset] & masks[0]) >> shift(masks[0])
    hi = (data[offset] & masks[0]) >> shift(masks[0])
    lo = (data[offset + 1] & masks[1]) >> shift(masks[1])
    return (hi << bin(masks[1]).count("1")) | lo


def test_init_values_match_the_factory_dump():
    assert len(INIT_DUMP) == 122 and INIT_DUMP[0] == 0xF0 and INIT_DUMP[-1] == 0xF7
    patch = init_patch()
    for key, (offset, masks) in FIELDS.items():
        assert patch[key] == _field(INIT_DUMP, offset, masks), key


def test_eight_bit_values_use_two_controllers():
    p = PARAMS["filter_freq"]
    assert p.messages(201) == [(16, 100), (48, 64)]  # the documented example: 201 -> 100, 64
    assert p.messages(200) == [(16, 100), (48, 0)]
    assert PARAMS["mix_osc1"].messages(255) == [(20, 127), (52, 64)]


def test_nrpn_value_travels_in_data_entry_msb():
    assert PARAMS["osc1_wave"].messages(3) == [(99, 0), (98, 72), (6, 3), (101, 127), (100, 127)]
    assert PARAMS["filter_track"].messages(7)[:3] == [(99, 0), (98, 108), (6, 7)]


def test_every_parameter_survives_encode_decode():
    dec = MidiDecoder()
    for p in PARAM_LIST:
        for v in sorted({p.lo, p.hi, p.init, (p.lo + p.hi) // 2, min(p.hi, p.lo + 1), max(p.lo, p.hi - 1)}):
            for cc, val in p.messages(v):
                assert 0 <= cc <= 127 and 0 <= val <= 127
                dec.control(cc, val)
            assert dec.patch[p.key] == v, (p.key, v)
    assert dec.unknown == []


def test_addresses_are_unique():
    seen = set()
    for p in PARAM_LIST:
        ids = [(p.kind if p.kind == "nrpn" else "cc", p.num)]
        if p.lsb is not None:
            ids.append(("cc", p.lsb))
        for i in ids:
            assert i not in seen, (p.key, i)
            seen.add(i)


def test_display_values():
    assert fine_cents(128) == 0 and fine_cents(127) == 0 and fine_cents(228) == 100 and fine_cents(27) == -100
    assert coarse_semitones(128) == 0.0 and coarse_semitones(255) == 12.0 and coarse_semitones(0) == -12.0
    for semi, raw in COARSE_SEMI_TO_RAW.items():
        assert coarse_semitones(raw) == pytest.approx(semi, abs=1e-9)
    assert round(pulse_width_percent(64)) == 50 and round(pulse_width_percent(0)) == 5
    assert PARAMS["osc1_range"].display(63) == "16'" and PARAMS["filter_env"].display(64) == "+0"
    assert PARAMS["sub_oct"].display(62) == "-2 oct"


def test_patch_helpers():
    patch = normalize_patch({"filter_freq": 999, "osc1_wave": 3, "nonsense": 5})
    assert patch["filter_freq"] == 255 and "nonsense" not in patch and set(patch) == set(PARAMS)
    assert baseline_patch()["lfo1_keysync"] == 1
    changed = [r["key"] for r in describe_patch(patch, only_changed=True)]
    assert "osc1_wave" in changed and "filter_res" not in changed
    assert "Filter Frequency" in patch_text(patch)


def test_sysex_helpers():
    assert SYSEX_REQUEST_CURRENT == (0xF0, 0x00, 0x20, 0x29, 0x00, 0x33, 0x00, 0x40, 0xF7)
    dump = bytes([0xF0, 0, 0x20, 0x29, 0, 0x33, 0]) + bytes(146) + bytes([0xF7])
    assert is_patch_dump(dump) and not is_patch_dump(dump[:50]) and not is_patch_dump(b"\xf0\x7e\x7f\x06\x01\xf7")
    named = dump_with_name(dump, "Tuned bass")
    assert named[137:153] == b"Tuned bass      " and len(named) == 154
