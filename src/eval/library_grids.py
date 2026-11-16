from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from src.music.rhythm import Note, clip_grid, pattern_id
from src.music.spec import MusicSpec

from src.eval.audio_io import resolve
from src.eval.evaluate import EvalError
from src.eval.ttd_prep import read_notes

## hits this far outside the loop are passed to clip_grid, as its documentation asks, so a flam
## over a bar line merges the same way in both neighbouring clips
MARGIN_BEATS = 1.0
GRID_COLUMNS = ("source_clip_id", "midi_path", "start_bar", "pattern_id")


@dataclass(frozen=True)
class SourceGrids:
    source_ids: list[str]
    onsets: np.ndarray
    mismatched: list[str]

    def rows_for(self, source_clip_ids: Sequence[str]) -> np.ndarray:
        row = {source_id: i for i, source_id in enumerate(self.source_ids)}
        return np.array([row[s] for s in source_clip_ids], dtype=int)


def beats_per_bar(spec: MusicSpec) -> float:
    if len(spec.loops.time_signatures) != 1:
        raise EvalError("library grids need exactly one time signature in the spec")
    numerator, denominator = map(int, spec.loops.time_signatures[0].split("-"))
    return numerator * 4 / denominator


def clip_notes(notes: list[tuple[int, int, float]], start_bar: int, spec: MusicSpec) -> list[Note]:
    start = start_bar * beats_per_bar(spec)
    end = start + spec.loops.bars * beats_per_bar(spec)
    return [
        Note(pitch, velocity, beat - start)
        for pitch, velocity, beat in notes
        if start - MARGIN_BEATS <= beat < end + MARGIN_BEATS
    ]


def source_grids(library: pd.DataFrame, spec: MusicSpec) -> SourceGrids:
    missing = [c for c in GRID_COLUMNS if c not in library.columns]
    if missing:
        raise EvalError(f"library manifest: missing columns {missing}")
    unique = library.drop_duplicates("source_clip_id").sort_values("source_clip_id")
    parsed: dict[str, tuple[list[tuple[int, int, float]], float]] = {}
    onsets, mismatched = [], []
    for row in unique.itertuples():
        if row.midi_path not in parsed:
            parsed[row.midi_path] = read_notes(resolve(row.midi_path))
        notes, bpm = parsed[row.midi_path]
        try:
            grid = clip_grid(clip_notes(notes, int(float(row.start_bar)), spec), bpm, spec)
        except KeyError as error:
            raise EvalError(f"{row.midi_path}: pitch the spec does not know: {error}") from error
        onsets.append(grid.onsets)
        if pattern_id(grid) != row.pattern_id:
            mismatched.append(row.source_clip_id)
    if not onsets:
        raise EvalError("no library clips to build grids for")
    return SourceGrids(unique["source_clip_id"].tolist(), np.stack(onsets), mismatched)
