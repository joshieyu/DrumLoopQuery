"""Verify raw window boundaries, count-in skipping, and ending exclusion."""

from dataclasses import replace

import pytest

from src.library.clips import first_clip
from src.music.rhythm import Note
from src.music.spec import load_spec


def test_first_clip_skips_count_in_and_preserves_raw_hits():
    spec = load_spec()
    notes = [
        Note(42, 80, 0),  # Count-in has no kick or snare.
        Note(36, 100, 4),
        Note(38, 100, 5),
        Note(38, 1, 6),  # Quiet hits still appear in raw output.
        Note(49, 90, 7),  # Dropped label voice stays in raw output.
        Note(42, 80, 11.999),
        Note(36, 100, 12),  # Exact end is excluded.
    ]
    clip = first_clip(notes, duration_beats=16, time_signature="4-4", spec=spec)
    assert (clip.start_bar, clip.start_beat, clip.end_beat) == (1, 4, 12)
    assert clip.notes[0] == Note(36, 100, 0)
    assert clip.notes[-1].beat == pytest.approx(7.999)
    assert len(clip.notes) == 5
    assert notes[1].beat == 4  # Source events are untouched.


def test_quiet_hits_do_not_qualify_a_start():
    notes = [Note(36, 1, 0), Note(38, 1, 1), Note(36, 100, 4), Note(38, 100, 5)]
    clip = first_clip(notes, duration_beats=16, time_signature="4-4", spec=load_spec())
    assert clip.start_bar == 1


def test_ending_and_partial_bar_cannot_supply_missing_clip_length():
    notes = [Note(36, 100, 4), Note(38, 100, 5)]
    with pytest.raises(ValueError, match="Not enough complete bars"):
        first_clip(notes, duration_beats=15.9, time_signature="4-4", spec=load_spec())
    spec = load_spec()
    spec = replace(spec, loops=replace(spec.loops, drop_last_bars=0))
    assert first_clip(notes, duration_beats=15.9, time_signature="4-4", spec=spec)


def test_unknown_pitches_and_no_qualified_bar_are_explicit():
    with pytest.raises(KeyError, match="not in drum_map"):
        first_clip([Note(1, 80, 0)], duration_beats=16, time_signature="4-4", spec=load_spec())
    with pytest.raises(ValueError, match="No bar contains"):
        first_clip([Note(42, 80, 0)], duration_beats=16, time_signature="4-4", spec=load_spec())
