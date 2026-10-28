"""Tests for the grid A/B listening tool (no GMD data needed)."""

import pretty_midi

from src.music.rhythm import Note
from src.music.spec import load_spec
from tools.listen.grid_ab import cut_loops, grid_hits, label_pitches, write_midi

SPEC = load_spec()


def bar(index, snare_beat):
    """One 4/4 bar: kick on 1, snare on `snare_beat`, 8th-note hats."""
    start = index * 4
    notes = [Note(36, 100, start), Note(38, 100, start + snare_beat)]
    return notes + [Note(42, 80, start + i / 2) for i in range(8)]


def test_cut_loops_skips_the_count_in_drops_the_ending_and_hops_two_bars():
    count_in = [Note(42, 80, i) for i in range(4)]  # hats only: not a start bar
    notes = count_in + [n for b in range(1, 8) for n in bar(b, snare_beat=1 + b % 3)]
    loops = cut_loops(notes, 32.0, 120.0, "test/1", SPEC)
    # 8 bars, last one dropped: loops start on bars 1, 3, 5 (0-based).
    assert [lp.start_bar for lp in loops] == [1, 3, 5]
    assert all(lp.grid.n_steps == 96 for lp in loops)


def test_cut_loops_removes_repeats_within_a_performance():
    notes = [n for b in range(8) for n in bar(b, snare_beat=1)]
    loops = cut_loops(notes, 32.0, 120.0, "test/2", SPEC)
    assert [lp.start_bar for lp in loops] == [0]


def test_grid_midi_lands_exactly_on_grid_steps(tmp_path):
    notes = [n for b in range(2) for n in bar(b, snare_beat=1)] + [Note(36, 90, 2 + 2 / 3)]
    (loop,) = cut_loops(notes + [n for b in range(2, 4) for n in bar(b, 3)], 16.0, 97.0, "t", SPEC)
    path = tmp_path / "grid.mid"
    write_midi(path, grid_hits(loop.grid, SPEC, velocity=100), loop.bpm, SPEC)
    pm = pretty_midi.PrettyMIDI(str(path))
    ticks = sorted(pm.time_to_tick(n.start) for n in pm.instruments[0].notes)
    lead_in = 4 * pm.resolution
    assert ticks[0] == lead_in
    assert all((t - lead_in) * SPEC.same_rhythm.steps_per_beat % pm.resolution == 0 for t in ticks)
    assert {n.pitch for n in pm.instruments[0].notes} == set(label_pitches(SPEC).values())
