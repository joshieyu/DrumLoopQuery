"""The "same rhythm" function: a MIDI clip plus the spec in, a quantized voice grid out.

Interface first, so B (loop cutting, manifests) and D (answer key) can code against it
while the spec §2 decisions are still open:

- `clip_grid` and `dualize` have their final signatures but raise NotImplementedError
  until the music lead settles the grid, ghost-note, flam, and swing rules.
- `Grid` and `pattern_id` are final. Two clips share a pattern ID exactly when their grids
  are identical, and the ID is the same on every machine and Python version.

Typical use (once implemented):

    spec = load_spec()
    grid = clip_grid(notes, bpm=clip_bpm, spec=spec)
    library_row["pattern_id"] = pattern_id(grid)
    library_row["dualized_pattern_id"] = pattern_id(dualize(grid, spec))
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from src.music.spec import MusicSpec

PATTERN_ID_VERSION = "grid-v1"  # bump if the hashed byte layout ever changes


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

    Applies, in order and with every constant from `spec`: the drum map (spec §1),
    ghost-note removal, flam merging (needs `bpm`, since the window is in ms), and
    quantizing to `spec.same_rhythm.steps_per_beat` (spec §2). The clip is
    `spec.loops.bars` bars long. Returns a Grid with rows in `spec.voices` order.

    Not implemented until the spec §2 decisions are in.
    """
    raise NotImplementedError("clip_grid waits for the music spec §2 decisions")


def dualize(grid: Grid, spec: MusicSpec) -> Grid:
    """Reduce a voice grid to the two tap hands (spec §7): rows ("low", "high").

    Not implemented until the spec §7 rules are confirmed.
    """
    raise NotImplementedError("dualize waits for the music spec §7 decisions")


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
