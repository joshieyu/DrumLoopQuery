"""The "same rhythm" function: a MIDI clip plus the spec in, a quantized voice grid out.

What B (loop cutting, manifests) and D (answer key) build on:

- `clip_grid` turns one clip's MIDI hits into its voice grid, following spec §1-3.
- `has_min_hits` says whether a grid has the content a loop needs to be kept (spec §2).
- `dualize` reduces a voice grid to the two tap hands (spec §7).
- `Grid` and `pattern_id`: two clips share a pattern ID exactly when their grids are
  identical, and the ID is the same on every machine and Python version.

Typical use:

    spec = load_spec()
    grid = clip_grid(notes, bpm=clip_bpm, spec=spec)
    if has_min_hits(grid, spec):
        library_row["pattern_id"] = pattern_id(grid)
        library_row["dualized_pattern_id"] = pattern_id(dualize(grid, spec))
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from src.music.spec import MusicSpec

PATTERN_ID_VERSION = "grid-v1"  # bump if the hashed byte layout ever changes
TAP_HANDS = ("low", "high")  # row names of a dualized grid
_MIDPOINT_TOLERANCE = 1e-6  # in steps; a MIDI tick at 480 per beat is 0.025 of a 12th step


class Note(NamedTuple):
    """One drum hit of a clip, straight from the MIDI.

    beat: position in beats from the clip's first downbeat. Hits played a hair before
    the downbeat may be slightly negative; clip_grid assigns every hit to its nearest
    grid step and ignores hits that round outside the clip.
    """

    pitch: int
    velocity: int
    beat: float


@dataclass(frozen=True, eq=False)
class Grid:
    """Quantized onsets for one clip.

    onsets[v, s] is True when voice `voices[v]` has a hit on step s. Steps run from the
    clip's first downbeat at `steps_per_beat` per beat, so a 2-bar 4/4 clip at 12 steps
    per beat has 96 steps.
    """

    voices: tuple[str, ...]
    steps_per_beat: int
    onsets: np.ndarray

    def __post_init__(self) -> None:
        onsets = np.asarray(self.onsets)
        if onsets.ndim != 2 or onsets.shape[0] != len(self.voices):
            raise ValueError(
                f"onsets must have shape ({len(self.voices)}, n_steps), got {onsets.shape}"
            )
        if self.steps_per_beat < 1 or onsets.shape[1] % self.steps_per_beat:
            raise ValueError(
                f"{onsets.shape[1]} steps is not a whole number of beats at "
                f"{self.steps_per_beat} steps per beat"
            )
        onsets = onsets.astype(bool)
        onsets.setflags(write=False)
        object.__setattr__(self, "onsets", onsets)

    @property
    def n_steps(self) -> int:
        return self.onsets.shape[1]

    @property
    def beats(self) -> int:
        return self.n_steps // self.steps_per_beat

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Grid)
            and self.voices == other.voices
            and self.steps_per_beat == other.steps_per_beat
            and np.array_equal(self.onsets, other.onsets)
        )

    def __hash__(self) -> int:
        return hash(pattern_id(self))


def clip_grid(notes: Sequence[Note], bpm: float, spec: MusicSpec) -> Grid:
    """Turn one clip's MIDI hits into its quantized voice grid.

    Applies, in order and with every constant from `spec`:

    1. The drum map (spec §1): each pitch becomes a voice; dropped pitches go. Pitches
       the spec doesn't mention raise KeyError.
    2. Ghost notes: hits below `spec.ghost_threshold(voice)` go.
    3. Flams: same-voice hits closer than `same_rhythm.flam_merge_ms` (converted with
       `bpm`) become one hit at the time of the hit `same_rhythm.flam_keep` picks
       ("loudest": the loudest hit, the earlier one on a tie). With
       `same_rhythm.flam_chain`, a run of hits, each that close to the next, is one hit.
    4. Quantizing: each hit moves to its nearest step at `same_rhythm.steps_per_beat`.
       A hit exactly halfway between two steps goes to the simpler position by
       `same_rhythm.midpoint_priority`, or the later step if they rank the same.
    5. With `loops.boundary` "nearest_step", hits that round outside the clip belong to
       a neighbouring clip and are ignored.

    Pass every hit from at least half a step before the clip's first downbeat up to its
    end. Extra hits outside the clip are harmless, so a beat of margin on each side is
    the easy choice, and it lets a flam that straddles a bar line merge the same way in
    both clips.

    The clip is `spec.loops.bars` bars long. Returns a Grid with rows in `spec.voices`
    order. Two clips are the same rhythm when their grids are equal
    (`same_rhythm.match` "exact").
    """
    sr = spec.same_rhythm
    if sr.flam_keep is None:
        raise ValueError("music spec has no same_rhythm.flam_keep yet")
    if spec.loops.boundary != "nearest_step":
        raise ValueError(
            f"clip_grid needs loops.boundary 'nearest_step', not {spec.loops.boundary!r}"
        )
    if not (math.isfinite(bpm) and bpm > 0):
        raise ValueError(f"bpm must be positive and finite, got {bpm}")
    n_steps = _clip_steps(spec)

    by_voice: dict[str, list[Note]] = {v: [] for v in spec.voices}
    for note in notes:
        if not math.isfinite(note.beat):
            raise ValueError(f"Note positions must be finite, got {note}")
        voice = spec.voice_of(note.pitch)
        if voice is not None and note.velocity >= spec.ghost_threshold(voice):
            by_voice[voice].append(note)

    onsets = np.zeros((len(spec.voices), n_steps), dtype=bool)
    for row, voice in enumerate(spec.voices):
        for beat in _merge_flams(by_voice[voice], bpm, spec):
            step = _nearest_step(beat, spec)
            if 0 <= step < n_steps:
                onsets[row, step] = True
    return Grid(spec.voices, sr.steps_per_beat, onsets)


def has_min_hits(grid: Grid, spec: MusicSpec) -> bool:
    """True when the grid has at least `same_rhythm.min_hits_per_loop` hits per voice."""
    _check_voices(grid, spec)
    return all(
        int(grid.onsets[grid.voices.index(voice)].sum()) >= minimum
        for voice, minimum in spec.same_rhythm.min_hits_per_loop.items()
    )


def dualize(grid: Grid, spec: MusicSpec) -> Grid:
    """Reduce a voice grid to the two tap hands (spec §7): rows ("low", "high").

    `spec.taps.low` voices go to "low" and `spec.taps.high` voices to "high". When both
    hands have a hit on the same step, only the `spec.taps.simultaneous` hand keeps it.
    """
    _check_voices(grid, spec)
    rows = dict(zip(grid.voices, grid.onsets, strict=True))
    low = np.zeros(grid.n_steps, dtype=bool)
    high = np.zeros(grid.n_steps, dtype=bool)
    for voice in spec.taps.low:
        low |= rows[voice]
    for voice in spec.taps.high:
        high |= rows[voice]
    both = low & high
    if spec.taps.simultaneous == "low":
        high &= ~both
    else:
        low &= ~both
    return Grid(TAP_HANDS, grid.steps_per_beat, np.stack([low, high]))


def pattern_id(grid: Grid) -> str:
    """Stable 16-hex-character ID for a grid.

    Hashes the voice names, resolution, length, and the onset bits, so grids of
    different shapes or voice sets never share an ID.
    """
    h = hashlib.sha1()
    header = f"{PATTERN_ID_VERSION}|{','.join(grid.voices)}|{grid.steps_per_beat}|{grid.n_steps}|"
    h.update(header.encode("utf-8"))
    h.update(np.packbits(grid.onsets, axis=None).tobytes())
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------- helpers


def _clip_steps(spec: MusicSpec) -> int:
    """Grid length of one clip: bars x beats per bar x steps per beat."""
    if len(spec.loops.time_signatures) != 1:
        raise ValueError("clip_grid needs exactly one time signature in loops.time_signatures")
    numerator, denominator = map(int, spec.loops.time_signatures[0].split("-"))
    # Beats are quarter notes, as in the MIDI.
    steps = spec.loops.bars * numerator * 4 / denominator * spec.same_rhythm.steps_per_beat
    if not steps.is_integer():
        raise ValueError(f"{spec.loops.time_signatures[0]} bars don't fill whole grid steps")
    return int(steps)


def _merge_flams(notes: list[Note], bpm: float, spec: MusicSpec) -> list[float]:
    """Positions (in beats) of one voice's hits after flam merging."""
    sr = spec.same_rhythm
    clusters: list[list[Note]] = []
    for note in sorted(notes, key=lambda n: n.beat):
        if clusters:
            anchor = clusters[-1][-1] if sr.flam_chain else clusters[-1][0]
            if (note.beat - anchor.beat) * 60000.0 / bpm < sr.flam_merge_ms:
                clusters[-1].append(note)
                continue
        clusters.append([note])
    if sr.flam_keep == "first":
        return [c[0].beat for c in clusters]
    # "loudest"; on a tie the earliest, since each cluster is in time order.
    return [max(c, key=lambda n: n.velocity).beat for c in clusters]


def _nearest_step(beat: float, spec: MusicSpec) -> int:
    x = beat * spec.same_rhythm.steps_per_beat
    lower = math.floor(x)
    frac = x - lower
    if abs(frac - 0.5) <= _MIDPOINT_TOLERANCE:
        upper = lower + 1
        return lower if _step_rank(lower, spec) < _step_rank(upper, spec) else upper
    return lower if frac < 0.5 else lower + 1


def _step_rank(step: int, spec: MusicSpec) -> int:
    """0 for the simplest positions (the beat), larger for finer subdivisions."""
    spb = spec.same_rhythm.steps_per_beat
    position = step % spb
    for rank, subdivisions in enumerate(spec.same_rhythm.midpoint_priority):
        # The position lies on an n-per-beat grid when position / spb == k / n.
        if any(position * n % spb == 0 for n in subdivisions):
            return rank
    return len(spec.same_rhythm.midpoint_priority)


def _check_voices(grid: Grid, spec: MusicSpec) -> None:
    if grid.voices != spec.voices:
        raise ValueError(f"grid voices {grid.voices} don't match the spec's {spec.voices}")
