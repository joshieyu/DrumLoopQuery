"""Verify labeled boundaries, context, count-in skipping, stride, and deduplication."""

from dataclasses import replace

import pytest

from src.library.clips import cut_clips, first_clip
from src.music.rhythm import Note, clip_grid, pattern_id
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
        Note(36, 100, 12),  # Exact end remains as labeling context.
    ]
    clip = first_clip(notes, duration_beats=16, time_signature="4-4", bpm=120, spec=spec)
    assert (clip.start_bar, clip.start_beat, clip.end_beat) == (1, 4, 12)
    assert clip.notes[0] == Note(36, 100, 0)
    assert clip.notes[-1].beat == 8
    assert clip.grid.onsets[0, 0]
    assert len(clip.notes) == 6
    assert notes[1].beat == 4  # Source events are untouched.


def test_quiet_hits_do_not_qualify_a_start():
    notes = [Note(36, 1, 0), Note(38, 1, 1), Note(36, 100, 4), Note(38, 100, 5)]
    clip = first_clip(notes, duration_beats=16, time_signature="4-4", bpm=120, spec=load_spec())
    assert clip.start_bar == 1


def test_ending_and_partial_bar_cannot_supply_missing_clip_length():
    notes = [Note(36, 100, 4), Note(38, 100, 5)]
    with pytest.raises(ValueError, match="No eligible clip"):
        first_clip(notes, duration_beats=15.9, time_signature="4-4", bpm=120, spec=load_spec())
    spec = load_spec()
    spec = replace(spec, loops=replace(spec.loops, drop_last_bars=0))
    assert first_clip(notes, duration_beats=15.9, time_signature="4-4", bpm=120, spec=spec)


def test_early_downbeat_qualifies_start_and_uses_shared_labels():
    notes = [Note(42, 80, 0), Note(36, 100, 3.99), Note(38, 100, 5)]
    spec = load_spec()
    clip = first_clip(notes, duration_beats=16, time_signature="4-4", bpm=120, spec=spec)
    assert clip.start_bar == 1
    assert clip.notes[0].beat == pytest.approx(-0.01)
    assert clip.grid.onsets[clip.grid.voices.index("kick"), 0]
    assert clip.pattern_id == pattern_id(clip_grid(clip.notes, bpm=120, spec=spec))
    assert notes[1].beat == 3.99


def test_stride_dedup_and_minimum_content():
    notes = [
        Note(36, 100, 0),
        Note(38, 100, 1),
        Note(36, 100, 8),
        Note(38, 100, 9),
        Note(36, 100, 16),
        Note(38, 100, 18),
        Note(36, 100, 24),
    ]
    spec = load_spec()
    args = dict(duration_beats=36, time_signature="4-4", bpm=120)
    assert [c.start_bar for c in cut_clips(notes, spec=spec, **args)] == [0, 4]
    spec = replace(spec, loops=replace(spec.loops, remove_duplicates_within_performance=False))
    assert [c.start_bar for c in cut_clips(notes, spec=spec, **args)] == [0, 2, 4]
    spec = replace(spec, loops=replace(spec.loops, max_loops_per_performance=1))
    assert len(cut_clips(notes, spec=spec, **args)) == 1


def test_unknown_pitches_and_no_qualified_bar_are_explicit():
    with pytest.raises(KeyError, match="not in drum_map"):
        first_clip(
            [Note(1, 80, 0)], duration_beats=16, time_signature="4-4", bpm=120, spec=load_spec()
        )
    with pytest.raises(ValueError, match="No eligible clip"):
        first_clip(
            [Note(42, 80, 0)], duration_beats=16, time_signature="4-4", bpm=120, spec=load_spec()
        )
