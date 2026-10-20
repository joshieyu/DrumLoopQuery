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
    assert "same_rhythm.steps_per_beat" in undecided
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
    ],
)
def test_invalid_specs_are_rejected(raw, edit, message):
    with pytest.raises(SpecError, match=message):
        parse_spec(broken(raw, edit))
