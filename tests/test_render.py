"""Tests for the listening prototype's SFZ reader, sampler, and kit helpers (tools/render).

Programs are tiny synthetic .sfz files with float WAVs at 8 kHz, written to tmp_path, so
every expected sample value can be worked out by hand. Most programs use amp_veltrack=0
(velocity doesn't change the gain) and pan=-100 (a mono sample lands on the left channel
at full gain), so the left channel shows exactly which sample played and when.
"""

import math

import numpy as np
import pytest
import soundfile as sf

from src.music.spec import Kit, KitNote, load_spec
from tools.render import kits
from tools.render.sampler import FAST_OFF_S, Hit, Sampler, _pan_gains
from tools.render.sfz import SfzError, builtin_curve, note_number, read_program, velocity_gain

SR = 8000
FLAT = "amp_veltrack=0 pan=-100"  # gain 1, mono sample on the left channel only


def write_wav(path, data, sr=SR):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.asarray(data, dtype=np.float32), sr, subtype="FLOAT")


def make_program(tmp_path, text, sr=SR, **samples):
    """Write <name>.wav for each sample and kit.sfz holding `text`; read the program."""
    for name, data in samples.items():
        write_wav(tmp_path / f"{name}.wav", data, sr)
    path = tmp_path / "kit.sfz"
    path.write_text(text)
    return read_program(path)


def const(value, n=10):
    return np.full(n, value)


def impulse(lead=0, n=100, value=1.0):
    data = np.zeros(n)
    data[lead] = value
    return data


def left(audio, time_s):
    return float(audio[round(time_s * SR), 0])


# --------------------------------------------------------------------------- reading


def test_note_numbers():
    assert [note_number(t) for t in ["38", "c4", "C#4", "db4", "c-1", "g9"]] == [
        38,
        60,
        61,
        61,
        0,
        127,
    ]
    with pytest.raises(SfzError, match="not a MIDI note"):
        note_number("h2")


def test_defines_includes_and_inheritance(tmp_path):
    (tmp_path / "inc").mkdir()
    # Include paths start from the top program's folder, even inside an included file.
    (tmp_path / "inc" / "a.sfzh").write_text(
        "<group> group_label=kick pan=10\n"
        "<region> key=$KICK sample=k.wav pan=20\n"
        '#include "inc/c.sfzh"\n'
    )
    (tmp_path / "inc" / "c.sfzh").write_text("<region> key=37 sample=My Stick 01.wav\n")
    (tmp_path / "inc" / "b.sfzh").write_text(
        "<group> volume=-6\n<region> key=c4 sample=sub\\s.wav\n"
    )
    program = make_program(
        tmp_path,
        "// line comment\n"
        "#define $KICK 36 #define $VOL -3\n"
        "<control> default_path=samples/ set_cc7=127 set_hdcc10=0.25\n"
        "<global> amp_veltrack=50 /* block\ncomment */ volume=$VOL\n"
        '#include "inc/a.sfzh" #include "inc/b.sfzh"\n'
        "<region> key=38 sample=s.wav seq_length=2seq_position=2\n",
    )
    kick, stick, sub, snare = program.regions
    assert (kick.lokey, kick.hikey, kick.pan, kick.amp_veltrack, kick.volume_db) == (
        36,
        36,
        20.0,
        50.0,
        -3.0,
    )
    assert (stick.pan, stick.label) == (10.0, "kick")  # group opcodes reach every region in it
    assert stick.sample == str((tmp_path / "samples" / "My Stick 01.wav").resolve())
    assert (sub.lokey, sub.pan, sub.volume_db) == (60, 0.0, -6.0)  # a new group clears the old
    assert sub.sample == str((tmp_path / "samples" / "sub" / "s.wav").resolve())
    assert (snare.seq_length, snare.seq_position) == (2, 2)  # no space between opcodes
    assert program.cc_defaults == {7: 1.0, 10: 0.25}


def test_missing_include_is_an_error(tmp_path):
    with pytest.raises(SfzError, match="not found"):
        make_program(tmp_path, '#include "nope.sfzh"\n')


def test_curve_header(tmp_path):
    program = make_program(
        tmp_path,
        "<curve> curve_index=7 v000=0 v063=1 v127=1\n<region> key=38 sample=a.wav\n",
    )
    assert program.curve(7, 0) == 0
    assert program.curve(7, 63 / 127) == pytest.approx(1)
    assert program.curve(7, 31.5 / 127) == pytest.approx(0.5)
    assert program.curve(7, 1) == 1
    assert program.curve(0, 0.3) == 0.3  # built-ins stay available


def test_builtin_curves():
    assert [builtin_curve(i, 0.25) for i in range(5)] == [0.25, -0.5, 0.75, 0.5, 0.0625]
    with pytest.raises(SfzError):
        builtin_curve(5, 0.5)


def test_regions_a_note_on_never_plays_are_skipped(tmp_path):
    program = make_program(
        tmp_path,
        "<region> key=38 sample=a.wav\n"
        "<region> key=38 sample=a.wav trigger=release\n"
        "<region> on_locc64=1 on_hicc64=127 sample=a.wav\n"
        "<region> key=38 lochan=2 hichan=2 sample=a.wav\n",
    )
    assert len(program.regions) == 1


def test_unsupported_and_unknown_opcodes(tmp_path):
    program = make_program(
        tmp_path,
        "<region> key=38 sample=a.wav cutoff=500 tune=0 offby=1 label_cc7=Vol\n"
        "<region> key=40 sample=a.wav tune=10 loop_mode=loop_continuous\n"
        "<region> key=41 sample=a.wav pitch_keytrack=100 loop_mode=one_shot\n",
        a=const(0.5),
    )
    assert program.unsupported == {"cutoff": 1, "tune": 1, "loop_mode=loop_continuous": 1}
    assert program.unknown == {"offby": 1}
    with pytest.raises(SfzError, match="cutoff"):
        Sampler(program)


def test_keyswitches(tmp_path):
    program = make_program(
        tmp_path,
        "<global> sw_lokey=12 sw_hikey=14 sw_default=13\n"
        "<region> key=38 sw_last=12 sample=a.wav\n"
        "<region> key=38 sw_last=13 sample=b.wav\n"
        "<region> lokey=10 hikey=16 sample=a.wav\n",
    )
    assert (program.sw_default, program.sw_keys) == (13, (12, 14))
    assert program.candidates(38, 100, {}) == [1]  # only the default switch's region
    assert program.keys() == {10, 11, 15, 16, 38}  # keyswitch notes aren't sounds


# --------------------------------------------------------------------------- levels


def test_velocity_gain(tmp_path):
    program = make_program(
        tmp_path,
        "<region> key=1 sample=a.wav\n"
        "<region> key=2 sample=a.wav amp_velcurve_64=1\n"
        "<region> key=3 sample=a.wav amp_veltrack=50\n",
    )
    default, curve, half = program.regions
    assert velocity_gain(default, 64) == pytest.approx((64 / 127) ** 2)
    assert velocity_gain(default, 127) == 1
    assert velocity_gain(curve, 32) == pytest.approx(0.5)  # linear from (0, 0) to (64, 1)
    assert velocity_gain(curve, 100) == 1
    assert velocity_gain(half, 0) == pytest.approx(0.5)


def test_cc_modulations(tmp_path):
    program = make_program(
        tmp_path,
        "<control> set_cc7=127 set_hdcc89=0.5\n"
        "<global> amp_veltrack=0\n"
        "<region> key=1 sample=a.wav amplitude_oncc7=100\n"
        "<region> key=2 sample=a.wav amplitude_oncc7=100 amplitude_oncc89=200\n"
        "<region> key=3 sample=a.wav amplitude_oncc7=100 amplitude_curvecc7=4\n"
        "<region> key=4 sample=a.wav volume=-6 volume_oncc10=6 pan=-50 pan_oncc10=100 "
        "phase=invert\n",
    )
    master, mic, concave, mixed = program.regions

    def gain(r, **ccs):
        return program.level(r, 100, program.cc_values({int(k[2:]): v for k, v in ccs.items()}))

    assert gain(master) == (1.0, 0.0)
    assert gain(master, cc7=0)[0] == 0  # amplitude_oncc goes from silence to full
    assert gain(master, cc7=0.5)[0] == pytest.approx(0.5)
    assert gain(mic)[0] == pytest.approx(1.0)  # 100% x (200% x 0.5): modulations multiply
    assert gain(mic, cc7=0.5, cc89=0.25)[0] == pytest.approx(0.25)
    assert gain(concave, cc7=0.5)[0] == pytest.approx(0.25)
    g, pan = gain(mixed, cc10=1.0)
    assert (g, pan) == (pytest.approx(-1.0), 50.0)  # volume and pan modulations add
    g, pan = gain(mixed)
    assert (g, pan) == (pytest.approx(-(10 ** (-6 / 20))), -50.0)


def test_pan_law():
    h = math.sqrt(0.5)
    assert _pan_gains(1.0, 0, 1) == pytest.approx((h, h))  # mono: -3 dB constant power
    assert _pan_gains(1.0, -100, 1) == pytest.approx((1, 0))
    assert _pan_gains(1.0, 100, 1) == pytest.approx((0, 1))
    assert _pan_gains(1.0, 0, 2) == (1, 1)  # stereo: balance
    assert _pan_gains(1.0, -50, 2) == (1, 0.5)
    assert _pan_gains(0.5, 100, 2) == (0, 0.5)


# --------------------------------------------------------------------------- selection


def signs(audio, times):
    return ["+" if left(audio, t) > 0 else "-" for t in times]


def test_velocity_layers(tmp_path):
    program = make_program(
        tmp_path,
        f"<global> key=38 {FLAT}\n<region> hivel=63 sample=a.wav\n<region> lovel=64 sample=b.wav\n",
        a=const(0.5),
        b=const(-0.5),
    )
    audio = Sampler(program).render([Hit(0.0, 38, 63), Hit(0.1, 38, 64)])
    assert (left(audio, 0.0), left(audio, 0.1)) == (0.5, -0.5)


def test_round_robin_follows_time_order(tmp_path):
    program = make_program(
        tmp_path,
        f"<global> key=38 seq_length=2 {FLAT}\n"
        "<region> seq_position=1 sample=a.wav\n<region> seq_position=2 sample=b.wav\n",
        a=const(0.5),
        b=const(-0.5),
    )
    times = [0.3, 0.2, 0.1, 0.0]  # given out of order on purpose
    audio = Sampler(program).render([Hit(t, 38, 100) for t in times])
    assert signs(audio, sorted(times)) == ["+", "-", "+", "-"]


def test_random_layers_are_seeded(tmp_path):
    program = make_program(
        tmp_path,
        f"<global> key=38 {FLAT}\n"
        "<region> hirand=0.5 sample=a.wav\n<region> lorand=0.5 sample=b.wav\n",
        a=const(0.5),
        b=const(-0.5),
    )
    times = [i * 0.01 for i in range(40)]
    hits = [Hit(t, 38, 100) for t in times]
    first = Sampler(program, seed=3).render(hits)
    assert np.array_equal(first, Sampler(program, seed=3).render(hits))
    assert {abs(left(first, t)) for t in times} == {0.5}  # exactly one layer per hit
    assert set(signs(first, times)) == {"+", "-"}
    assert signs(first, times) != signs(Sampler(program, seed=4).render(hits), times)


def test_cc_conditions(tmp_path):
    text = (
        f"<global> key=46 {FLAT}\n<region> hicc4=63 sample=a.wav\n<region> locc4=64 sample=b.wav\n"
    )
    program = make_program(tmp_path, text, a=const(0.5), b=const(-0.5))
    sampler = Sampler(program)
    hits = [Hit(0.0, 46, 100, {4: 0}), Hit(0.1, 46, 100, {4: 127}), Hit(0.2, 46, 100)]
    assert signs(sampler.render(hits), [0.0, 0.1, 0.2]) == ["+", "-", "+"]  # CCs start at 0
    program = make_program(tmp_path, "<control> set_cc4=127\n" + text)
    assert signs(Sampler(program).render([Hit(0.0, 46, 100)]), [0.0]) == ["-"]


# --------------------------------------------------------------------------- timing


def test_hits_land_on_their_frame(tmp_path):
    program = make_program(tmp_path, f"<region> key=38 {FLAT} sample=a.wav\n", a=impulse())
    audio = Sampler(program).render([Hit(0.25, 38, 100)], length_s=1.0)
    assert audio.shape == (SR, 2) and audio.dtype == np.float32
    assert int(np.argmax(np.abs(audio[:, 0]))) == round(0.25 * SR)
    assert audio[round(0.25 * SR), 0] == 1.0
    assert not audio[:, 1].any()


def test_onset_alignment(tmp_path):
    lead = impulse(lead=20)
    lead[:20] = 0.001  # -60 dB of noise before the hit isn't its onset
    program = make_program(
        tmp_path,
        f"<global> key=38 {FLAT}\n<region> sample=close.wav\n<region> sample=room.wav\n",
        close=lead,
        room=impulse(lead=50, value=0.5),
    )
    hit = [Hit(0.25, 38, 100)]
    at = round(0.25 * SR)
    aligned = Sampler(program).render(hit)[:, 0]
    # The earliest mic lands on the hit; the room mic keeps its 30-frame delay.
    assert (aligned[at], aligned[at + 30]) == (1.0, 0.5)
    raw = Sampler(program, align_onsets=False).render(hit)[:, 0]
    assert (raw[at + 20], raw[at + 50]) == (1.0, 0.5)


def test_offset_end_and_envelope(tmp_path):
    program = make_program(
        tmp_path,
        f"<global> {FLAT}\n"
        "<region> key=38 offset=2 end=4 sample=ramp.wav\n"
        "<region> key=40 end=-1 sample=ramp.wav\n"
        "<region> key=42 ampeg_attack=0.01 sample=flat.wav\n",
        ramp=np.arange(10) / 10,
        flat=const(0.5, 200),
    )
    sampler = Sampler(program, align_onsets=False)
    assert sampler.render([Hit(0.0, 38, 100)])[:, 0] == pytest.approx([0.2, 0.3, 0.4])
    assert len(sampler.render([Hit(0.0, 40, 100)])) == 0  # end=-1: silent
    attack = sampler.render([Hit(0.0, 42, 100)])[:, 0]
    assert attack[40] == pytest.approx(0.25)  # halfway up an 80-frame linear attack
    assert attack[100] == 0.5


def test_mixed_sample_rates_are_refused(tmp_path):
    write_wav(tmp_path / "fast.wav", const(0.5), sr=2 * SR)
    program = make_program(
        tmp_path,
        "<region> key=38 sample=a.wav\n<region> key=40 sample=fast.wav\n",
        a=const(0.5),
    )
    with pytest.raises(SfzError, match="resampling"):
        Sampler(program).render([Hit(0.0, 40, 100)])


# --------------------------------------------------------------------------- voice stealing


def choke_program(tmp_path, extra=""):
    return make_program(
        tmp_path,
        f"<global> {FLAT}\n"
        "<region> key=46 group=1 off_by=2 sample=open.wav\n"
        "<region> key=42 group=2 sample=closed.wav\n"
        "<region> key=44 group=2 sample=*silence\n"
        "<region> key=43 group=3 sample=closed.wav\n" + extra,
        open=const(0.5, SR // 2),
        closed=impulse(n=11, value=0.25),
    )


def test_choke(tmp_path):
    sampler = Sampler(choke_program(tmp_path))
    at, fade = round(0.1 * SR), round(FAST_OFF_S * SR)
    choked = sampler.render([Hit(0.0, 46, 100), Hit(0.1, 42, 100)])[:, 0]
    assert len(choked) == at + fade  # the open hat fades out over 6 ms
    assert (choked[at - 1], choked[at]) == (0.5, 0.75)
    assert choked[-1] == pytest.approx(0.5 / fade)
    # A silent region still chokes (kits use these for "choke" keys).
    assert len(sampler.render([Hit(0.0, 46, 100), Hit(0.1, 44, 100)])) == at + fade
    # Another group doesn't.
    assert len(sampler.render([Hit(0.0, 46, 100), Hit(0.1, 43, 100)])) == SR // 2


def test_note_polyphony_spares_louder_voices(tmp_path):
    program = make_program(
        tmp_path,
        f"<region> key=38 note_polyphony=1 off_time=0.002 {FLAT} sample=a.wav\n",
        a=const(0.5, SR),
    )
    sampler = Sampler(program)
    first, second = sampler.voices([Hit(0.0, 38, 100), Hit(0.1, 38, 100)])
    assert (first.stop, first.stop_s, second.stop) == (round(0.1 * SR), 0.002, None)
    first, _ = sampler.voices([Hit(0.0, 38, 127), Hit(0.1, 38, 50)])
    assert first.stop is None


def test_group_polyphony(tmp_path):
    program = make_program(
        tmp_path,
        f"<group> group=5 polyphony=2 {FLAT}\n<region> lokey=38 hikey=40 sample=a.wav\n",
        a=const(0.5, SR),
    )
    voices = Sampler(program).voices([Hit(0.0, 38, 100), Hit(0.1, 39, 100), Hit(0.2, 40, 100)])
    assert [v.stop for v in voices] == [round(0.2 * SR), None, None]


# --------------------------------------------------------------------------- kit helpers


def tiny_kit(**note_map):
    return Kit(
        kit_id="tiny",
        type="acoustic",
        program="tiny/kit.sfz",
        samples=None,
        split=None,
        controls={7: 127, 4: 0},
        note_map={int(n[1:]): v for n, v in note_map.items()},
    )


def test_kit_hits():
    kit = tiny_kit(n38=KitNote(38, {}), n46=KitNote(46, {4: 127}), n49=KitNote(None, {}))
    hits = kits.kit_hits(kit, [(0.1, 38, 100), (0.2, 46, 90), (0.3, 49, 80)])
    assert hits == [
        Hit(0.1, 38, 100, {7: 127, 4: 0}),
        Hit(0.2, 46, 90, {7: 127, 4: 127}),  # the note's CC beats the kit's control
    ]


def test_check_kit(tmp_path):
    program = make_program(
        tmp_path,
        "<global> sw_lokey=12 sw_hikey=14 sw_default=12\n"
        "<region> key=38 sample=a.wav\n<region> key=46 locc4=64 sample=a.wav\n",
    )
    kit = tiny_kit(
        n38=KitNote(38, {}),
        n40=KitNote(12, {}),
        n37=KitNote(99, {}),
        n46=KitNote(46, {4: 127}),
        n42=KitNote(46, {}),
        n49=KitNote(None, {}),
    )
    problems = kits.check_kit(kit, program, load_spec())
    assert len(problems) == 3
    assert "GMD 37 (snare) -> key 99: the program plays nothing there" in problems
    assert "GMD 40 (snare) -> key 12 is a keyswitch, not a sound" in problems
    assert any(p.startswith("GMD 42 ") for p in problems)  # 46 needs CC4 >= 64


def test_load_program_explains_missing_kits(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_datasets"):
        kits.load_program(tiny_kit(), tmp_path)


def test_normalize():
    t = np.arange(SR) / SR
    sine = np.stack([0.3 * np.sin(2 * np.pi * 100 * t)] * 2, axis=1)
    out = kits.normalize(sine)
    assert 20 * np.log10(np.sqrt(np.mean(out**2))) == pytest.approx(kits.TARGET_RMS_DB, abs=1e-3)
    click = np.zeros((SR, 2))
    click[100] = 1.0
    peak = float(np.abs(kits.normalize(click)).max())
    assert 20 * np.log10(peak) == pytest.approx(kits.PEAK_CEILING_DB, abs=1e-3)
    silence = np.zeros((10, 2))
    assert np.array_equal(kits.normalize(silence), silence)


# --------------------------------------------------------------------------- the real kits

SPEC = load_spec()
MISSING = [k.kit_id for k in SPEC.rendering.kits if not (kits.KITS_DIR / k.program).is_file()]


@pytest.mark.skipif(bool(MISSING), reason=f"kits not downloaded: {MISSING}")
def test_real_kit_note_maps_play_sounds():
    for kit in SPEC.rendering.kits:
        assert kits.check_kit(kit, kits.load_program(kit), SPEC) == [], kit.kit_id


@pytest.mark.skipif(bool(MISSING), reason=f"kits not downloaded: {MISSING}")
def test_command_line_renders_a_kit(tmp_path):
    from tools.render.sampler import main

    args = ["--kit", "avl_black_pearl", "--note", "38", "--vel", "100", "--repeat", "1"]
    assert main([*args, "--out", str(tmp_path)]) == 0
    audio, rate = sf.read(tmp_path / "avl_black_pearl_note38.wav")
    assert rate in (44100, 48000)
    assert np.abs(audio).max() > 0.1


def test_sound_names():
    assert kits.sound_name("/k/Samples/Snare_rim/1-Snare_rim-OHL.flac") == "Snare_rim"  # DrumGizmo
    assert kits.sound_name("/k/BLACK_PEARL/38-Pearl-14-Snare-1.wav") == "Pearl-14-Snare"  # AVL
    assert kits.sound_name("/k/OH/snare2_OH_Ghost_3.wav") == "snare2"  # Salamander
    assert kits.sound_name("/k/OH/hihatSemiOpen7_OH_F_1.wav") == "hihatSemiOpen7"


def test_describe(tmp_path):
    program = make_program(
        tmp_path,
        "<global> key=38 seq_length=2\n"
        "<region> hivel=63 seq_position=1 sample=snare_1.wav\n"
        "<region> hivel=63 seq_position=2 sample=snare_2.wav\n"
        "<region> lovel=64 sample=snare_3.wav\n"
        "<region> key=40 seq_length=1 sample=*silence\n"
        "<region> key=46 seq_length=1 locc4=64 sample=hat_1.wav\n",
    )
    assert kits.describe(program, 38, {}) == ("snare", "2 layers, 2 variations")
    assert kits.describe(program, 40, {}) == ("", "no sound")
    assert kits.describe(program, 46, {}) == ("", "no sound")
    assert kits.describe(program, 46, {4: 1.0}) == ("hat", "1 layer, 1 variation")
