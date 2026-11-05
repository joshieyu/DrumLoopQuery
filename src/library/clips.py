"""Cut and label GMD loops using the shared music spec and rhythm functions."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from src.music.rhythm import Grid, Note, clip_grid, dualize, has_min_hits, pattern_id
from src.music.spec import MusicSpec


@dataclass(frozen=True)
class Clip:
    """One labeled window with a beat of raw context on each side.

    Context notes may have negative beats or positions beyond the window end.
    They are inputs to labeling, not instructions to render neighbouring hits.
    """

    start_bar: int
    start_beat: float
    end_beat: float
    notes: tuple[Note, ...]
    grid: Grid
    pattern_id: str
    dualized_pattern_id: str


def cut_clips(
    notes: Sequence[Note],
    *,
    duration_beats: float,
    time_signature: str,
    bpm: float,
    spec: MusicSpec,
) -> list[Clip]:
    """Extract qualifying windows at the configured stride, deduplicated locally.

    Only complete source bars are considered. Reserve the configured number of
    final complete bars, conservatively ignoring any partial trailing bar. Do not
    shift the stride to compensate for a rejected or duplicate window.
    The first bar's required voices are checked on the shared quantized grid.
    Quiet/dropped hits stay in context, but do not qualify a start.
    Unknown pitches raise through spec.voice_of rather than being silently lost.
    """
    if time_signature not in spec.loops.time_signatures:
        raise ValueError(f"Time signature {time_signature} is not allowed by the spec")
    if not math.isfinite(duration_beats) or duration_beats <= 0:
        raise ValueError("duration_beats must be positive and finite")
    if spec.loops.hop_bars is None:
        raise ValueError("loops.hop_bars must be decided before cutting loops")
    numerator, denominator = map(int, time_signature.split("-"))
    # MIDI beat coordinates are quarter notes; a whole note has four quarters.
    beats_per_bar = numerator * 4 / denominator
    complete_bars = math.floor(duration_beats / beats_per_bar)
    usable_bars = complete_bars - spec.loops.drop_last_bars
    ordered = sorted(notes, key=lambda note: (note.beat, note.pitch))
    for note in ordered:
        if not math.isfinite(note.beat):
            raise ValueError("Note positions must be finite")
        spec.voice_of(note.pitch)
    length = spec.loops.bars * beats_per_bar

    def window(bar: int) -> tuple[tuple[Note, ...], Grid]:
        start = bar * beats_per_bar
        # Week 3 handoff: a beat on either side preserves boundary flams.
        context = tuple(
            Note(n.pitch, n.velocity, n.beat - start)
            for n in ordered
            if start - 1 <= n.beat < start + length + 1
        )
        return context, clip_grid(context, bpm=bpm, spec=spec)

    required = set(spec.loops.start_at_first_bar_with)
    first_bar_steps = round(beats_per_bar * spec.same_rhythm.steps_per_beat)
    start_bar = None
    for bar in range(max(usable_bars - spec.loops.bars + 1, 0)):
        _, grid = window(bar)
        content = {
            voice for i, voice in enumerate(grid.voices) if grid.onsets[i, :first_bar_steps].any()
        }
        if required <= content:
            start_bar = bar
            break
    if start_bar is None:
        return []
    clips, seen = [], set()
    for bar in range(start_bar, usable_bars - spec.loops.bars + 1, spec.loops.hop_bars):
        context, grid = window(bar)
        if not has_min_hits(grid, spec):
            continue
        pid = pattern_id(grid)
        if spec.loops.remove_duplicates_within_performance and pid in seen:
            continue
        seen.add(pid)
        start = bar * beats_per_bar
        clips.append(
            Clip(bar, start, start + length, context, grid, pid, pattern_id(dualize(grid, spec)))
        )
        limit = spec.loops.max_loops_per_performance
        if limit is not None and len(clips) >= limit:
            break
    return clips


def first_clip(
    notes: Sequence[Note],
    *,
    duration_beats: float,
    time_signature: str,
    bpm: float,
    spec: MusicSpec,
) -> Clip:
    """Convenience entry point for the single-performance inspection report."""
    clips = cut_clips(
        notes, duration_beats=duration_beats, time_signature=time_signature, bpm=bpm, spec=spec
    )
    if not clips:
        raise ValueError("No eligible clip with sufficient complete bars and required voices")
    return clips[0]
