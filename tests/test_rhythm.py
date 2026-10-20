"""Tests for the "same rhythm" interface: Grid and pattern_id are final; the rest is stubbed."""

import numpy as np
import pytest

from src.music.rhythm import Grid, Note, clip_grid, dualize, pattern_id
from src.music.spec import load_spec

VOICES = ("kick", "snare", "hats")


def backbeat_grid(steps_per_beat=12, bars=2):
    n = steps_per_beat * 4 * bars
    on = np.zeros((3, n), bool)
    on[0, [0, 30, 48]] = True  # kick
    on[1, [24, 72]] = True  # snare on 2 and 4
    on[2, :: steps_per_beat // 2] = True  # 8th-note hats
    return Grid(VOICES, steps_per_beat, on)


def test_pattern_id_is_stable():
    # Golden value: if this changes, every manifest's pattern IDs change with it.
    assert pattern_id(backbeat_grid()) == "39e11f03ec025eea"


def test_identical_grids_share_an_id_and_compare_equal():
    a, b = backbeat_grid(), backbeat_grid()
    assert a == b
    assert pattern_id(a) == pattern_id(b)
    assert len({a, b}) == 1


def test_one_changed_hit_changes_the_id():
    a = backbeat_grid()
    on = a.onsets.copy()
    on[2, 3] = True
    assert pattern_id(Grid(VOICES, 12, on)) != pattern_id(a)


def test_shape_and_voices_are_part_of_the_id():
    a = backbeat_grid()
    renamed = Grid(("low", "mid", "high"), 12, a.onsets)
    assert pattern_id(renamed) != pattern_id(a)
    # Same bits, read as 24 steps per beat over 1 bar instead of 12 over 2 bars.
    assert pattern_id(Grid(VOICES, 24, a.onsets)) != pattern_id(a)


def test_grid_validates_shape_and_is_read_only():
    with pytest.raises(ValueError):
        Grid(VOICES, 12, np.zeros((2, 96), bool))
    with pytest.raises(ValueError):
        Grid(VOICES, 12, np.zeros((3, 100), bool))
    g = backbeat_grid()
    assert (g.n_steps, g.beats) == (96, 8)
    with pytest.raises(ValueError):
        g.onsets[0, 1] = True


def test_unimplemented_parts_say_why():
    spec = load_spec()
    with pytest.raises(NotImplementedError, match="§2"):
        clip_grid([Note(36, 100, 0.0)], bpm=120, spec=spec)
    with pytest.raises(NotImplementedError, match="§7"):
        dualize(backbeat_grid(), spec)
