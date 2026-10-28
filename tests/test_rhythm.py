"""Tests for the "same rhythm" function: clip_grid, has_min_hits, dualize, Grid, pattern_id.

Positions are written in beats from the clip's first downbeat. At 12 steps per beat one
step is 1/12 beat; at 120 BPM a beat is 500 ms, so 1 ms is 0.002 beat.
"""

from dataclasses import replace

import numpy as np
import pytest

from src.music.rhythm import (
    TAP_HANDS,
    Grid,
    Note,
    clip_grid,
    dualize,
    has_min_hits,
    pattern_id,
)
from src.music.spec import load_spec

VOICES = ("kick", "snare", "hats")
SPEC = load_spec()
KICK, SNARE, RIMSHOT, CROSS_STICK = 36, 38, 40, 37
CLOSED_HAT, OPEN_HAT, PEDAL_HAT, RIDE, RIDE_BELL = 42, 46, 44, 51, 53
TOM, CRASH = 48, 49
TICKS = 480  # GMD's MIDI resolution


def rules(**same_rhythm):
    return replace(SPEC, same_rhythm=replace(SPEC.same_rhythm, **same_rhythm))


def steps(grid, voice):
    return np.flatnonzero(grid.onsets[grid.voices.index(voice)]).tolist()


def grid_of(notes, bpm=120, spec=SPEC):
    return clip_grid([Note(*n) for n in notes], bpm, spec)


def backbeat_grid(steps_per_beat=12, bars=2):
    n = steps_per_beat * 4 * bars
    on = np.zeros((3, n), bool)
    on[0, [0, 30, 48]] = True  # kick
    on[1, [24, 72]] = True  # snare on 2 and 4
    on[2, :: steps_per_beat // 2] = True  # 8th-note hats
    return Grid(VOICES, steps_per_beat, on)


# --------------------------------------------------------------------------- Grid and IDs


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


# --------------------------------------------------------------------------- clip_grid: grooves


def humanized_backbeat():
    """Two bars of rock beat, played a little loose (offsets up to 15 ms at 120 BPM)."""
    notes = []
    for bar in (0, 4):
        notes += [(KICK, 110, bar + 0.006), (KICK, 95, bar + 2.0 - 0.01), (KICK, 90, bar + 2.5)]
        notes += [(SNARE, 115, bar + 1.0 + 0.008), (SNARE, 118, bar + 3.0 - 0.004)]
        notes += [(CLOSED_HAT, 70 + 10 * (i % 2), bar + i / 2 + 0.01 * (-1) ** i) for i in range(8)]
    return notes


def test_humanized_backbeat_quantizes_to_the_written_beat():
    g = grid_of(humanized_backbeat())
    assert (g.voices, g.steps_per_beat, g.n_steps) == (VOICES, 12, 96)
    assert steps(g, "kick") == [0, 24, 30, 48, 72, 78]
    assert steps(g, "snare") == [12, 36, 60, 84]
    assert steps(g, "hats") == list(range(0, 96, 6))
    # Golden value: catches any later change to how clips are quantized.
    assert pattern_id(g) == "82523c050b5aceb6"


def test_shuffle_lands_on_triplet_steps():
    notes = []
    for beat in range(8):
        notes += [(RIDE, 90, beat), (RIDE, 70, beat + 2 / 3 + 0.012)]
    notes += [(KICK, 110, 0), (KICK, 100, 2 + 2 / 3), (KICK, 110, 4), (KICK, 100, 6 + 2 / 3)]
    notes += [(SNARE, 110, b) for b in (1, 3, 5, 7)]
    g = grid_of(notes, bpm=100)
    assert steps(g, "hats") == sorted([12 * b for b in range(8)] + [12 * b + 8 for b in range(8)])
    assert steps(g, "kick") == [0, 32, 48, 80]
    assert pattern_id(g) == "b9e2b8e02398865f"


def test_no_notes_gives_an_empty_grid():
    g = clip_grid([], 120, SPEC)
    assert g.onsets.shape == (3, 96) and not g.onsets.any()


# --------------------------------------------------------------------------- drum map, ghosts


def test_drum_map_groups_articulations_and_drops_toms_and_crashes():
    g = grid_of(
        [(RIMSHOT, 100, 1), (CROSS_STICK, 100, 2), (OPEN_HAT, 100, 3), (PEDAL_HAT, 100, 4)]
        + [(RIDE_BELL, 100, 5), (TOM, 120, 6), (CRASH, 120, 7)]
    )
    assert steps(g, "snare") == [12, 24]
    assert steps(g, "hats") == [36, 48, 60]
    assert not g.onsets[:, 72:].any()


def test_unknown_pitch_raises_instead_of_vanishing():
    with pytest.raises(KeyError, match="not in drum_map"):
        grid_of([(KICK, 100, 0), (0, 100, 1)])


def test_ghost_notes_below_the_threshold_are_ignored():
    assert SPEC.ghost_threshold("snare") == 15
    g = grid_of([(SNARE, 14, 1), (SNARE, 15, 2)])
    assert steps(g, "snare") == [24]


def test_ghost_thresholds_can_differ_per_voice():
    spec = rules(ghost_velocity_below={"kick": 30, "snare": 15, "hats": 15})
    g = grid_of([(KICK, 20, 0), (SNARE, 20, 1)], spec=spec)
    assert steps(g, "kick") == [] and steps(g, "snare") == [12]


# --------------------------------------------------------------------------- flams


@pytest.mark.parametrize("gap_ms, expected", [(29.9, [12]), (30.1, [12, 13])])
def test_flams_merge_only_below_the_window(gap_ms, expected):
    g = grid_of([(SNARE, 100, 1.0), (SNARE, 50, 1.0 + gap_ms * 0.002)])
    assert steps(g, "snare") == expected


def test_exactly_the_window_apart_is_not_a_flam():
    # At 125 BPM, 30 ms is exactly 1/16 beat, which floats represent exactly.
    g = grid_of([(SNARE, 100, 1.0), (SNARE, 50, 1.0625)], bpm=125)
    assert steps(g, "snare") == [12, 13]


def test_flam_window_is_in_milliseconds_so_tempo_matters():
    notes = [(SNARE, 100, 1.0), (SNARE, 50, 1.05)]  # 25 ms at 120 BPM, 50 ms at 60 BPM
    assert steps(grid_of(notes, bpm=120), "snare") == [12]
    assert steps(grid_of(notes, bpm=60), "snare") == [12, 13]


# A grace note at beat 0.95 rounds to step 11 and a main stroke 22.5 ms later (0.995) rounds
# to step 12, so the merged hit's step shows whose timing survived.
@pytest.mark.parametrize(
    "grace_velocity, main_velocity, expected",
    [(40, 110, [12]), (110, 40, [11]), (80, 80, [11])],
    ids=["loudest-later", "loudest-earlier", "tie-goes-earlier"],
)
def test_merged_flam_takes_the_loudest_hits_timing(grace_velocity, main_velocity, expected):
    g = grid_of([(SNARE, grace_velocity, 0.95), (SNARE, main_velocity, 0.995)])
    assert steps(g, "snare") == expected


def test_flam_keep_first_takes_the_earliest_hit():
    g = grid_of([(SNARE, 40, 0.95), (SNARE, 110, 0.995)], spec=rules(flam_keep="first"))
    assert steps(g, "snare") == [11]


def test_flam_keep_must_be_decided():
    with pytest.raises(ValueError, match="flam_keep"):
        grid_of([(SNARE, 100, 1)], spec=rules(flam_keep=None))


# A ruff: grace notes 20 ms apart, then the main stroke. The run spans 40 ms.
RUFF = [(SNARE, 30, 0.90), (SNARE, 45, 0.94), (SNARE, 110, 0.98)]


def test_a_chained_ruff_is_one_hit_on_the_main_stroke():
    assert steps(grid_of(RUFF), "snare") == [12]


def test_without_chaining_only_hits_near_the_first_merge():
    # The main stroke is 40 ms after the first grace note, so it stands alone.
    assert steps(grid_of(RUFF, spec=rules(flam_chain=False)), "snare") == [11, 12]


def test_different_voices_never_merge():
    g = grid_of([(KICK, 100, 0.995), (SNARE, 100, 1.0), (CLOSED_HAT, 100, 1.005)])
    assert steps(g, "kick") == steps(g, "snare") == steps(g, "hats") == [12]


def test_hat_articulations_merge_as_one_voice():
    # Pedal hat 15 ms before a ride hit: one hats hit, at the louder ride.
    g = grid_of([(PEDAL_HAT, 60, 0.95), (RIDE, 100, 0.98)])
    assert steps(g, "hats") == [12]


# --------------------------------------------------------------------------- quantizing


def test_hits_go_to_the_nearest_step():
    g = grid_of([(KICK, 100, 0.03), (SNARE, 100, 0.05), (CLOSED_HAT, 100, 1.0 - 0.04)])
    assert (steps(g, "kick"), steps(g, "snare"), steps(g, "hats")) == ([0], [1], [12])


# Exact midpoints (a tick 20 past a 40-tick step) go to the simpler position: the beat,
# then the 8th, then a 16th or triplet. Equal ranks go to the later step.
MIDPOINTS = {
    0: 0,  # beat | 12th          -> late downbeat
    1: 2,  # 12th | 12th          -> later
    2: 3,  # 12th | 16th "e"
    3: 4,  # 16th "e" | triplet   -> later
    4: 4,  # triplet | 12th
    5: 6,  # 12th | 8th "&"
    6: 6,  # 8th "&" | 12th
    7: 8,  # 12th | triplet
    8: 9,  # triplet | 16th "a"   -> later
    9: 9,  # 16th "a" | 12th
    10: 11,  # 12th | 12th        -> later
    11: 12,  # 12th | next beat   -> early downbeat
}


@pytest.mark.parametrize("lower, expected", MIDPOINTS.items())
def test_midpoints_go_to_the_simpler_position(lower, expected):
    tick = TICKS + 40 * lower + 20  # in the clip's second beat
    g = grid_of([(KICK, 100, tick / TICKS)])
    assert steps(g, "kick") == [12 + expected]


@pytest.mark.parametrize("noise", [1e-12, -1e-12])
def test_midpoint_rule_survives_float_noise(noise):
    tick = TICKS + 40 * 3 + 20
    assert steps(grid_of([(KICK, 100, tick / TICKS + noise)]), "kick") == [16]


def test_one_tick_off_the_midpoint_is_plain_rounding():
    tick = TICKS + 40 * 3 + 20  # midpoint between steps 15 and 16 -> 16 by the tie rule
    assert steps(grid_of([(KICK, 100, (tick - 1) / TICKS)]), "kick") == [15]
    assert steps(grid_of([(KICK, 100, (tick + 1) / TICKS)]), "kick") == [16]


def test_midpoint_priority_comes_from_the_spec():
    tick = TICKS + 40 * 6 + 20  # between the 8th "&" and the next 12th
    assert steps(grid_of([(KICK, 100, tick / TICKS)]), "kick") == [18]
    only_beats = rules(midpoint_priority=((1,),))
    assert steps(grid_of([(KICK, 100, tick / TICKS)], spec=only_beats), "kick") == [19]


# --------------------------------------------------------------------------- clip boundaries


def test_early_downbeat_belongs_to_this_clip():
    g = grid_of([(KICK, 100, -0.03), (SNARE, 100, -0.05)])  # 15 ms and 25 ms early
    assert steps(g, "kick") == [0]
    assert steps(g, "snare") == []  # nearer the previous clip's last step


def test_early_downbeat_of_the_next_clip_leaves_this_one():
    g = grid_of([(KICK, 100, 8 - 0.03), (SNARE, 100, 8 - 0.05)])
    assert steps(g, "kick") == []
    assert steps(g, "snare") == [95]


def test_a_midpoint_on_the_boundary_belongs_to_the_downbeat():
    half_step = 20 / TICKS
    assert steps(grid_of([(KICK, 100, -half_step)]), "kick") == [0]
    assert steps(grid_of([(KICK, 100, 8 - half_step)]), "kick") == []


def test_hits_far_outside_the_clip_are_ignored():
    g = grid_of([(KICK, 100, -3), (KICK, 100, 12), (KICK, 100, 4)])
    assert steps(g, "kick") == [48]


def cut(notes, start, margin=1.0):
    """What a loop cutter passes for the clip starting at `start` beats."""
    end = start + 8
    return [Note(p, v, b - start) for p, v, b in notes if start - margin <= b < end + margin]


def test_every_hit_lands_in_exactly_one_clip():
    rng = np.random.default_rng(0)
    perf = []
    for voice_pitch, every in [(KICK, 1.0), (SNARE, 2.0), (CLOSED_HAT, 0.5)]:
        for slot in np.arange(0, 64, every):
            perf.append((voice_pitch, 100, slot + rng.uniform(-0.12, 0.12)))
    # Hits crowding every clip boundary from both sides.
    for bar_line in range(8, 64, 8):
        perf += [(KICK, 100, bar_line - 0.04), (SNARE, 100, bar_line + 0.04)]
    # Keep same-voice hits at least 40 ms apart, so no flams merge and every hit counts.
    perf.sort(key=lambda n: (SPEC.voice_of(n[0]), n[2]))
    spaced = []
    for n in perf:
        if not (0 <= n[2] < 64):
            continue
        prev = spaced[-1] if spaced else None
        if prev and SPEC.voice_of(prev[0]) == SPEC.voice_of(n[0]) and n[2] - prev[2] < 0.08:
            continue
        spaced.append(n)
    perf = spaced
    near_a_bar_line = [b for _, _, b in perf if 0 < b < 63 and abs(b - round(b / 8) * 8) < 1 / 24]
    assert len(near_a_bar_line) >= 5  # the boundaries really get tested

    expected = {(SPEC.voice_of(p), round(b * 12)) for p, _, b in perf if 0 <= round(b * 12) < 768}
    found, total = set(), 0
    for start in range(0, 64, 8):
        g = clip_grid(cut(perf, start), 120, SPEC)
        for row, voice in enumerate(g.voices):
            for s in np.flatnonzero(g.onsets[row]):
                found.add((voice, start * 12 + int(s)))
                total += 1
    assert found == expected
    assert total == len(found)  # nothing counted in two clips


def test_a_flam_across_the_bar_line_lands_in_one_clip():
    # Grace note 30 ms (0.06 beat) before the bar line, main stroke 10 ms before it.
    perf = [(SNARE, 40, 8 - 0.06), (SNARE, 110, 8 - 0.02)]
    first = clip_grid(cut(perf, 0), 120, SPEC)
    second = clip_grid(cut(perf, 8), 120, SPEC)
    assert (steps(first, "snare"), steps(second, "snare")) == ([], [0])


# --------------------------------------------------------------------------- clip_grid: errors


@pytest.mark.parametrize("bpm", [0, -90, float("nan"), float("inf")])
def test_bpm_must_be_positive_and_finite(bpm):
    with pytest.raises(ValueError, match="bpm"):
        grid_of([(KICK, 100, 0)], bpm=bpm)


def test_note_positions_must_be_finite():
    with pytest.raises(ValueError, match="finite"):
        grid_of([(KICK, 100, float("nan"))])


def test_raw_boundary_is_not_supported():
    spec = replace(SPEC, loops=replace(SPEC.loops, boundary="raw"))
    with pytest.raises(ValueError, match="nearest_step"):
        grid_of([(KICK, 100, 0)], spec=spec)


def test_more_than_one_time_signature_is_ambiguous():
    spec = replace(SPEC, loops=replace(SPEC.loops, time_signatures=("4-4", "3-4")))
    with pytest.raises(ValueError, match="time signature"):
        grid_of([(KICK, 100, 0)], spec=spec)


# --------------------------------------------------------------------------- has_min_hits


def test_min_hits_needs_a_kick_and_a_snare():
    assert SPEC.same_rhythm.min_hits_per_loop == {"kick": 1, "snare": 1}
    assert has_min_hits(grid_of([(KICK, 100, 0), (SNARE, 100, 1)]), SPEC)
    assert not has_min_hits(grid_of([(KICK, 100, 0), (CLOSED_HAT, 100, 1)]), SPEC)
    assert not has_min_hits(grid_of([(SNARE, 100, 1), (CLOSED_HAT, 100, 1)]), SPEC)


def test_min_hits_counts_grid_hits_after_merging():
    spec = rules(min_hits_per_loop={"kick": 2})
    flam = [(KICK, 100, 0.0), (KICK, 90, 0.02)]  # 10 ms apart: one hit
    assert not has_min_hits(grid_of(flam, spec=spec), spec)
    assert has_min_hits(grid_of(flam + [(KICK, 100, 2)], spec=spec), spec)


def test_min_hits_rejects_grids_with_other_voices():
    with pytest.raises(ValueError, match="voices"):
        has_min_hits(dualize(backbeat_grid(), SPEC), SPEC)


# --------------------------------------------------------------------------- dualize


def small_grid(kick=(), snare=(), hats=()):
    on = np.zeros((3, 96), bool)
    for row, hits in enumerate([kick, snare, hats]):
        on[row, list(hits)] = True
    return Grid(VOICES, 12, on)


def test_dualize_splits_low_and_high_hands():
    g = dualize(small_grid(kick=[0, 30], snare=[24], hats=[0, 6, 24, 30, 36]), SPEC)
    assert (g.voices, g.steps_per_beat, g.n_steps) == (TAP_HANDS, 12, 96)
    assert steps(g, "low") == [0, 30]
    # Hats under a kick go (one tap on the low hand); snare and hats on 24 are one tap.
    assert steps(g, "high") == [6, 24, 36]


def test_dualize_simultaneous_high_keeps_the_high_tap():
    spec = replace(SPEC, taps=replace(SPEC.taps, simultaneous="high"))
    g = dualize(small_grid(kick=[0, 30], snare=[24], hats=[0, 6]), spec)
    assert (steps(g, "low"), steps(g, "high")) == ([30], [0, 6, 24])


def test_grooves_that_differ_only_in_snare_vs_hats_share_a_tap_pattern():
    # README §5.4: many 3-voice patterns share one 2-voice reduction.
    a = small_grid(kick=[0, 48], snare=[24, 72], hats=[12, 36])
    b = small_grid(kick=[0, 48], snare=[12, 72], hats=[24, 36])
    assert pattern_id(a) != pattern_id(b)
    assert pattern_id(dualize(a, SPEC)) == pattern_id(dualize(b, SPEC))


def test_dualized_backbeat_has_a_stable_id():
    g = dualize(grid_of(humanized_backbeat()), SPEC)
    assert steps(g, "low") == [0, 24, 30, 48, 72, 78]
    assert steps(g, "high") == [6, 12, 18, 36, 42, 54, 60, 66, 84, 90]
    assert pattern_id(g) == "1f02231533e8bd23"


def test_dualize_rejects_grids_with_other_voices():
    with pytest.raises(ValueError, match="voices"):
        dualize(Grid(("kick", "snare"), 12, np.zeros((2, 96), bool)), SPEC)
