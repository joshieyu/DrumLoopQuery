"""Tests for config/music_spec.yaml and its loader."""

import copy

import pytest
import yaml

from src.music.spec import DEFAULT_PATH, SpecError, load_spec, parse_spec
from tools.inventory.gmd import GMD_NOTES


@pytest.fixture(scope="module")
def raw():
    return yaml.safe_load(DEFAULT_PATH.read_text())


def broken(raw, edit):
    """Deep copy of the real spec with one edit applied."""
    r = copy.deepcopy(raw)
    edit(r)
    return r


def test_project_spec_loads():
    spec = load_spec()
    assert spec.voices == ("kick", "snare", "hats")
    assert spec.same_rhythm.steps_per_beat >= 1
    assert spec.loops.bars >= 1


def test_every_documented_gmd_note_is_mapped_exactly_once():
    spec = load_spec()
    mapped = [n for g in spec.drum_map.values() for n in g.notes]
    assert len(mapped) == len(set(mapped))
    assert set(GMD_NOTES) <= set(mapped)


def test_music_lead_decisions_are_recorded():
    spec = load_spec()
    # Spec §1: open and closed hi-hats are the same voice.
    assert spec.drum_map["hihat_open"].group == spec.drum_map["hihat_closed"].group
    assert spec.is_decided("drum_map.hihat_open")
    # Spec §2: half-time and double-time are different rhythms.
    assert spec.same_rhythm.half_time_double_time_different is True
    assert spec.is_decided("same_rhythm.half_time_double_time_different")


def test_section_1_decisions():
    spec = load_spec()
    assert spec.is_decided("drum_map")
    assert spec.voice_of(44) == "hats"  # pedal hat counts as hats
    assert {spec.voice_of(n) for n in (40, 37)} == {"snare"}  # rimshot, cross-stick
    assert {spec.voice_of(n) for n in (51, 59, 53)} == {"hats"}  # ride and bell
    assert all(spec.voice_of(n) is None for n in spec.drum_map["toms"].notes)
    assert all(spec.voice_of(n) is None for n in spec.drum_map["crash"].notes)


def test_section_2_to_4_decisions():
    spec = load_spec()
    sr = spec.same_rhythm
    assert sr.steps_per_beat == 12
    assert all(spec.ghost_threshold(v) == 15 for v in spec.voices)
    assert sr.flam_merge_ms == 30
    assert sr.swing_is_different is True
    assert sr.min_hits_per_loop == {"kick": 1, "snare": 1}
    lp = spec.loops
    assert (lp.bars, lp.time_signatures, lp.start_on_downbeat) == (2, ("4-4",), True)
    assert lp.start_at_first_bar_with == ("kick", "snare")
    assert lp.drop_last_bars == 1
    assert lp.performance_tempo_bpm == (70, 160)
    assert lp.remove_duplicates_within_performance is True
    assert spec.rendering.tempo_bpm == (70, 160)
    for key in [
        "same_rhythm.steps_per_beat",
        "same_rhythm.ghost_velocity_below",
        "same_rhythm.flam_merge_ms",
        "same_rhythm.min_hits_per_loop",
        "loops.start_at_first_bar_with",
        "loops.drop_last_bars",
        "loops.performance_tempo_bpm",
        "rendering.tempo_bpm",
        "rendering.loudness_normalize",
        "rendering.keep_performer_velocities",
    ]:
        assert spec.is_decided(key), key


def test_2026_10_28_decisions():
    spec = load_spec()
    sr, lp, bb = spec.same_rhythm, spec.loops, spec.beatbox
    assert (sr.match, sr.flam_keep, sr.flam_chain) == ("exact", "loudest", True)
    assert sr.midpoint_priority == ((1,), (2,), (3, 4))
    assert (lp.boundary, lp.hop_bars) == ("nearest_step", 2)
    assert spec.rendering.max_tempo_change_pct == 20
    assert bb.avp_modes == ("personal", "fixed")
    assert bb.avp_recordings == ("isolated", "improvisation")
    assert (bb.sound_end.below_peak_db, bb.sound_end.max_ms, bb.sound_end.fade_ms) == (30, 500, 10)
    assert bb.sound_end.stop_at_next_onset is True
    assert bb.skip_velocity_below == {"kick": 0, "snare": 10, "hats": 0}
    assert (bb.open_hat_sound, bb.hats_sound) == ("hho", "hhc")
    assert (spec.taps.low, spec.taps.high, spec.taps.simultaneous) == (
        ("kick",),
        ("snare", "hats"),
        "low",
    )


def test_open_questions_stay_open():
    spec = load_spec()
    assert spec.rendering.kits == ()
    assert spec.sloppiness.swing_drift_per_bar is None
    assert spec.sloppiness.mic_eq is None
    assert set(spec.undecided()) == {
        "rendering.kits",
        "sloppiness.swing_drift_per_bar",
        "sloppiness.mic_eq",
    }


def test_voice_of_drops_and_rejects_unknown_notes():
    spec = load_spec()
    assert spec.voice_of(36) == "kick"
    for group in spec.drum_map.values():
        for note in group.notes:
            expected = None if group.group == "drop" else group.group
            assert spec.voice_of(note) == expected
    with pytest.raises(KeyError):
        spec.voice_of(0)


def test_scalar_ghost_threshold_applies_to_every_voice(raw):
    r = broken(raw, lambda r: r["same_rhythm"].update(ghost_velocity_below=35))
    spec = parse_spec(r)
    assert all(spec.ghost_threshold(v) == 35 for v in spec.voices)


def test_per_voice_ghost_thresholds(raw):
    per_voice = {"kick": 30, "snare": 40, "hats": 20}
    r = broken(raw, lambda r: r["same_rhythm"].update(ghost_velocity_below=per_voice))
    assert parse_spec(r).same_rhythm.ghost_velocity_below == per_voice


def test_undecided_lists_defaults_only():
    spec = load_spec()
    undecided = spec.undecided()
    assert "rendering.kits" in undecided
    assert "same_rhythm.steps_per_beat" not in undecided
    assert "drum_map.hihat_open" not in undecided
    assert all(spec.is_decided(u) == (u not in undecided) for u in spec.units)


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda r: r["same_rhythm"].update(stepz_per_beat=12), "unknown key"),
        (lambda r: r["loops"].pop("bars"), "missing key"),
        (lambda r: r["drum_map"]["toms"]["notes"].append(36), "already mapped"),
        (lambda r: r["drum_map"]["crash"].update(group="cymbals"), "not one of"),
        (lambda r: r["drum_map"].update(kick={"notes": [36], "group": "drop"}), "no notes map"),
        (lambda r: r["drum_map"]["kick"].update(notes=[128]), "outside"),
        (lambda r: r["same_rhythm"].update(steps_per_beat=0), "outside"),
        (lambda r: r["same_rhythm"].update(steps_per_beat=12.5), "integer"),
        (lambda r: r["same_rhythm"].update(ghost_velocity_below={"kick": 40}), "missing"),
        (lambda r: r["rendering"].update(tempo_bpm=[140, 80]), "above high"),
        (lambda r: r["sloppiness"]["drop_prob"].update(hats=[0, 1.5]), "outside"),
        (lambda r: r["beatbox"].update(simultaneous_priority=["kick", "snare"]), "every voice"),
        (lambda r: r["beatbox"].update(open_hat_sound="hho2"), "not one of"),
        (lambda r: r["taps"].update(high=["snare"]), "split the voices"),
        (lambda r: r["loops"].update(time_signatures=["4/4"]), "format"),
        (lambda r: r.update(decided=["same_rhythm.flam_window_ms"]), "do not name keys"),
        (lambda r: r["loops"].update(exclude_fills="yes"), "true or false"),
        (lambda r: r["same_rhythm"].update(flam_keep="middle"), "not one of"),
        (lambda r: r["loops"].update(hop_bars=0), "outside"),
        (lambda r: r["loops"].update(start_at_first_bar_with=["toms"]), "not one of"),
        (lambda r: r["loops"].update(performance_tempo_bpm=[160, 70]), "above high"),
        (lambda r: r["rendering"].update(max_tempo_change_pct=-5), "outside"),
        (lambda r: r["same_rhythm"].update(match="near"), "not one of"),
        (lambda r: r["same_rhythm"].update(flam_chain="yes"), "true or false"),
        (lambda r: r["same_rhythm"].update(midpoint_priority=[[1], [5]]), "don't divide"),
        (lambda r: r["same_rhythm"].update(midpoint_priority=[[1], [1, 2]]), "appears twice"),
        (lambda r: r["same_rhythm"].update(midpoint_priority=[[1], []]), "non-empty"),
        (lambda r: r["same_rhythm"].update(midpoint_priority=[[0]]), "outside"),
        (lambda r: r["loops"].update(boundary="rounded"), "not one of"),
        (lambda r: r["beatbox"].update(avp_modes=["personal", "both"]), "not one of"),
        (lambda r: r["beatbox"].update(avp_recordings=[]), "at least one"),
        (lambda r: r["beatbox"].update(avp_recordings=["live"]), "not one of"),
        (lambda r: r["beatbox"].update(hats_sound="hh"), "not one of"),
        (lambda r: r["beatbox"].update(skip_velocity_below={"snare": 40}), "missing"),
        (lambda r: r["beatbox"]["sound_end"].update(fade_ms=-1), "outside"),
        (lambda r: r["beatbox"]["sound_end"].update(fade_ms=600), "longer than max_ms"),
        (lambda r: r["beatbox"]["sound_end"].pop("max_ms"), "missing key"),
    ],
)
def test_invalid_specs_are_rejected(raw, edit, message):
    with pytest.raises(SpecError, match=message):
        parse_spec(broken(raw, edit))
