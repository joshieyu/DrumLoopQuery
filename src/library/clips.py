"""Extract one raw MIDI window; rhythm labels remain owned by src.music.rhythm.

Window membership uses original onset positions in [start, end). This diagnostic
cut does not quantize early hits or implement the pending rhythm-grid decisions.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from src.music.rhythm import Note
from src.music.spec import MusicSpec


@dataclass(frozen=True)
class Clip:
    """Raw hits relative to a zero-based starting bar; end_beat is exclusive."""

    start_bar: int
    start_beat: float
    end_beat: float
    notes: tuple[Note, ...]


def first_clip(
    notes: Sequence[Note], *, duration_beats: float, time_signature: str, spec: MusicSpec
) -> Clip:
    """Find the first content-qualified bar and extract a complete window there.

    Only complete source bars are considered. Reserve the configured number of
    final complete bars, conservatively ignoring any partial trailing bar. Do not
    search later starts if the first qualified bar leaves insufficient room.
    Quiet/dropped hits stay in the output, but quiet hits do not qualify a start.
    Unknown pitches raise through spec.voice_of rather than being silently lost.
    """
    if time_signature not in spec.loops.time_signatures:
        raise ValueError(f"Time signature {time_signature} is not allowed by the spec")
    if not math.isfinite(duration_beats) or duration_beats <= 0:
        raise ValueError("duration_beats must be positive and finite")
    numerator, denominator = map(int, time_signature.split("-"))
    # MIDI beat coordinates are quarter notes; a whole note has four quarters.
    beats_per_bar = numerator * 4 / denominator
    complete_bars = math.floor(duration_beats / beats_per_bar)
    usable_bars = complete_bars - spec.loops.drop_last_bars
    content: dict[int, set[str]] = {}
    ordered = sorted(notes, key=lambda note: (note.beat, note.pitch))
    for note in ordered:
        if not math.isfinite(note.beat):
            raise ValueError("Note positions must be finite")
        voice = spec.voice_of(note.pitch)
        if voice is not None and note.velocity >= spec.ghost_threshold(voice):
            bar = math.floor(note.beat / beats_per_bar)
            content.setdefault(bar, set()).add(voice)
    required = set(spec.loops.start_at_first_bar_with)
    start_bar = next(
        (bar for bar in range(max(usable_bars, 0)) if required <= content.get(bar, set())),
        None,
    )
    if start_bar is None:
        raise ValueError("No bar contains the required start voices before the ending")
    if start_bar + spec.loops.bars > usable_bars:
        raise ValueError("Not enough complete bars for a clip after reserving the ending")
    start = start_bar * beats_per_bar
    end = start + spec.loops.bars * beats_per_bar
    selected = tuple(
        Note(note.pitch, note.velocity, note.beat - start)
        for note in ordered
        if start <= note.beat < end
    )
    return Clip(start_bar=start_bar, start_beat=start, end_beat=end, notes=selected)
