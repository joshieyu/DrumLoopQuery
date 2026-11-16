from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.music.rhythm import Grid, pattern_id
from src.music.spec import load_spec

from src.eval.evaluate import EvalError
from src.eval.library_grids import clip_notes, source_grids

pretty_midi = pytest.importorskip("pretty_midi")

TEMPO = 100.0
KICK, SNARE, HAT = 36, 38, 42


def _write_midi(path, notes):
    midi = pretty_midi.PrettyMIDI(initial_tempo=TEMPO, resolution=480)
    instrument = pretty_midi.Instrument(program=0, is_drum=True)
    for pitch, beat in notes:
        start = beat * 60.0 / TEMPO
        instrument.notes.append(pretty_midi.Note(90, pitch, start, start + 0.05))
    midi.instruments.append(instrument)
    midi.write(str(path))


def _performance():
    ## four bars: kick on 1 and 3, snare on 2 and 4, hats on every 8th
    notes = []
    for bar in range(4):
        base = bar * 4
        notes += [(KICK, base + 0), (KICK, base + 2), (SNARE, base + 1), (SNARE, base + 3)]
        notes += [(HAT, base + 0.5 * k) for k in range(8)]
    ## a different kick pattern in bar 3, so start_bar matters
    notes = [n for n in notes if not (n[0] == KICK and 8 <= n[1] < 12)] + [
        (KICK, 8.5),
        (KICK, 10.5),
    ]
    return notes


def _expected_onsets(kick_beats, snare_beats, hat_beats):
    onsets = np.zeros((3, 96), dtype=bool)
    for row, beats in enumerate((kick_beats, snare_beats, hat_beats)):
        for beat in beats:
            onsets[row, round(beat * 12)] = True
    return onsets


@pytest.fixture
def spec():
    return load_spec()


@pytest.fixture
def midi_path(tmp_path):
    path = tmp_path / "performance.mid"
    _write_midi(path, _performance())
    return path


def _library(midi_path, start_bar, pattern, kits=("salamander", "muldjordkit")):
    rows = [
        (f"clip_{kit}", "perf:bar" + str(start_bar), str(midi_path), str(start_bar), pattern)
        for kit in kits
    ]
    return pd.DataFrame(
        rows, columns=["clip_id", "source_clip_id", "midi_path", "start_bar", "pattern_id"]
    )


def test_grid_is_cut_from_the_right_bars_and_matches_an_independent_pattern_id(midi_path, spec):
    ## loop = bars 0 and 1 (beats 0 to 8): the plain beat
    expected = _expected_onsets([0, 2, 4, 6], [1, 3, 5, 7], [0.5 * k for k in range(16)])
    pattern = pattern_id(Grid(spec.voices, 12, expected))
    result = source_grids(_library(midi_path, 0, pattern), spec)
    assert np.array_equal(result.onsets[0], expected)
    assert result.mismatched == []


def test_start_bar_moves_the_window(midi_path, spec):
    ## loop = bars 1 and 2 (beats 4 to 12): bar 2 has the off-beat kicks
    expected = _expected_onsets([0, 2, 4.5, 6.5], [1, 3, 5, 7], [0.5 * k for k in range(16)])
    pattern = pattern_id(Grid(spec.voices, 12, expected))
    result = source_grids(_library(midi_path, 1, pattern), spec)
    assert np.array_equal(result.onsets[0], expected)
    assert result.mismatched == []


def test_a_pattern_id_that_disagrees_with_the_midi_is_reported(midi_path, spec):
    result = source_grids(_library(midi_path, 0, "0000000000000000"), spec)
    assert result.mismatched == ["perf:bar0"]


def test_clips_of_one_source_share_one_grid_row(midi_path, spec):
    library = _library(midi_path, 0, "x")
    result = source_grids(library, spec)
    assert result.onsets.shape == (1, 3, 96)
    assert result.rows_for(library["source_clip_id"]).tolist() == [0, 0]


def test_hits_just_outside_the_loop_do_not_enter_it(spec):
    notes = [(36, 90, -0.4), (36, 90, 0.0), (36, 90, 8.0), (36, 90, 9.5)]
    kept = clip_notes([(p, v, b) for p, v, b in notes], 0, spec)
    ## beats within one beat of the loop are passed on; clip_grid then drops those that round out
    assert [round(n.beat, 1) for n in kept] == [-0.4, 0.0, 8.0]


def test_missing_columns_and_unknown_pitches_are_errors(midi_path, spec, tmp_path):
    with pytest.raises(EvalError, match="missing columns"):
        source_grids(_library(midi_path, 0, "x").drop(columns=["start_bar"]), spec)
    odd = tmp_path / "odd.mid"
    _write_midi(odd, [(KICK, 0), (SNARE, 1), (99, 2)])
    with pytest.raises(EvalError, match="pitch the spec does not know"):
        source_grids(_library(odd, 0, "x"), spec)
