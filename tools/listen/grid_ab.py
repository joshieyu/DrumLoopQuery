"""A/B the rhythm label against the drummer: random GMD loops as two MIDI files each.

    .venv/bin/python -m tools.listen.grid_ab            # 10 loops, seed 0
    .venv/bin/python -m tools.listen.grid_ab --n 20 --seed 3

For each loop it writes, to results/listening/grid_ab/:

- `NN_<performance>_bar<k>_original.mid`: every hit the drummer played in the loop,
  with toms, cymbals, ghost notes, flams, and timing untouched.
- `NN_<performance>_bar<k>_grid.mid`: the loop's rhythm label from `clip_grid`, one note
  per grid hit (kick, snare, and closed hat, all at one velocity).
- `ab_notes.csv`: one row per loop, with empty `verdict` and `notes` columns for you.

Both files start with one empty bar, so hits played a little before the loop's first
downbeat aren't cut off, and they line up when dragged into Logic together.

The loop cutting here is a listening prototype that follows the spec (start at the first
bar with the start voices, drop the ending, a new loop every `loops.hop_bars` bars). B's
pipeline in src/library is the real cutter.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pretty_midi

from src.music.rhythm import Grid, Note, clip_grid, has_min_hits, pattern_id
from src.music.spec import MusicSpec, load_spec

REPO = Path(__file__).resolve().parents[2]
GMD_DIR = REPO / "data" / "raw" / "gmd" / "groove"
OUT_DIR = REPO / "results" / "listening" / "grid_ab"
MARGIN_BEATS = 1.0  # extra hits passed to clip_grid on each side of a loop
MIDI_RESOLUTION = 480  # ticks per beat, as in GMD; divisible by every grid size we use


@dataclass(frozen=True)
class Loop:
    performance: str
    start_bar: int
    start_beat: float
    beats: float
    bpm: float
    notes: tuple[Note, ...]  # every hit within MARGIN_BEATS of the loop, beats from its start
    grid: Grid


def beats_per_bar(spec: MusicSpec) -> float:
    numerator, denominator = map(int, spec.loops.time_signatures[0].split("-"))
    return numerator * 4 / denominator


def eligible_performances(spec: MusicSpec, gmd_dir: Path = GMD_DIR) -> pd.DataFrame:
    info = pd.read_csv(gmd_dir / "info.csv")
    low, high = spec.loops.performance_tempo_bpm
    keep = info["time_signature"].isin(spec.loops.time_signatures) & info["bpm"].between(low, high)
    if spec.loops.exclude_fills:
        keep &= info["beat_type"] != "fill"
    return info[keep].sort_values("id").reset_index(drop=True)


def performance_notes(midi_path: Path) -> tuple[list[Note], float]:
    """Every drum hit in beats from the file's start, and the file's length in beats."""
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    notes = [
        Note(n.pitch, n.velocity, midi.time_to_tick(n.start) / midi.resolution)
        for inst in midi.instruments
        if inst.is_drum
        for n in inst.notes
    ]
    notes.sort(key=lambda n: (n.beat, n.pitch))
    return notes, midi.time_to_tick(midi.get_end_time()) / midi.resolution


def cut_loops(
    notes: list[Note], length_beats: float, bpm: float, performance: str, spec: MusicSpec
) -> list[Loop]:
    """All loops of one performance that pass the spec's content rule, duplicates removed."""
    bpb = beats_per_bar(spec)
    usable_bars = math.floor(length_beats / bpb) - spec.loops.drop_last_bars
    content: dict[int, set[str]] = {}
    for n in notes:
        voice = spec.voice_of(n.pitch)
        if voice is not None and n.velocity >= spec.ghost_threshold(voice):
            content.setdefault(math.floor(n.beat / bpb), set()).add(voice)
    required = set(spec.loops.start_at_first_bar_with)
    first = next((b for b in range(max(usable_bars, 0)) if required <= content.get(b, set())), None)
    if first is None:
        return []
    beats = spec.loops.bars * bpb
    loops, seen = [], set()
    for bar in range(first, usable_bars - spec.loops.bars + 1, spec.loops.hop_bars):
        start = bar * bpb
        window = tuple(
            Note(n.pitch, n.velocity, n.beat - start)
            for n in notes
            if start - MARGIN_BEATS <= n.beat < start + beats + MARGIN_BEATS
        )
        grid = clip_grid(window, bpm, spec)
        if not has_min_hits(grid, spec):
            continue
        pid = pattern_id(grid)
        if spec.loops.remove_duplicates_within_performance and pid in seen:
            continue
        seen.add(pid)
        loops.append(Loop(performance, bar, start, beats, bpm, window, grid))
    return loops


def label_pitches(spec: MusicSpec) -> dict[str, int]:
    """One MIDI note per voice for playing a grid: the first drum_map note of that voice."""
    return {
        v: next(g.notes[0] for g in spec.drum_map.values() if g.group == v) for v in spec.voices
    }


def write_midi(
    path: Path, notes: list[tuple[int, int, float]], bpm: float, spec: MusicSpec
) -> None:
    """(pitch, velocity, beat) hits into a drum MIDI file with one empty bar first."""
    pm = pretty_midi.PrettyMIDI(resolution=MIDI_RESOLUTION, initial_tempo=bpm)
    drums = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")
    sec_per_beat = 60.0 / bpm
    offset = beats_per_bar(spec)
    for pitch, velocity, beat in notes:
        start = (beat + offset) * sec_per_beat
        drums.notes.append(pretty_midi.Note(velocity, pitch, start, start + 0.1))
    pm.instruments.append(drums)
    pm.write(str(path))


def original_hits(loop: Loop, spec: MusicSpec) -> list[tuple[int, int, float]]:
    """Hits the loop owns: from half a step before its downbeat to half a step before its end."""
    half_step = 0.5 / spec.same_rhythm.steps_per_beat
    return [
        (n.pitch, n.velocity, n.beat)
        for n in loop.notes
        if -half_step <= n.beat < loop.beats - half_step
    ]


def grid_hits(grid: Grid, spec: MusicSpec, velocity: int) -> list[tuple[int, int, float]]:
    pitches = label_pitches(spec)
    return [
        (pitches[voice], velocity, step / grid.steps_per_beat)
        for row, voice in enumerate(grid.voices)
        for step in np.flatnonzero(grid.onsets[row])
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--n", type=int, default=10, help="loops to export, at most one per performance"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grid-velocity", type=int, default=100, help="velocity of every grid note")
    ap.add_argument("--gmd", type=Path, default=GMD_DIR)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    spec = load_spec()
    perfs = eligible_performances(spec, args.gmd)
    rng = np.random.default_rng(args.seed)
    rows = []
    for idx in rng.permutation(len(perfs)):
        if len(rows) == args.n:
            break
        p = perfs.iloc[idx]
        notes, length = performance_notes(args.gmd / p["midi_filename"])
        loops = cut_loops(notes, length, float(p["bpm"]), p["id"], spec)
        if not loops:
            continue
        loop = loops[rng.integers(len(loops))]
        args.out.mkdir(parents=True, exist_ok=True)
        stem = f"{len(rows) + 1:02d}_{p['id'].replace('/', '-')}_bar{loop.start_bar + 1}"
        write_midi(args.out / f"{stem}_original.mid", original_hits(loop, spec), loop.bpm, spec)
        write_midi(
            args.out / f"{stem}_grid.mid",
            grid_hits(loop.grid, spec, args.grid_velocity),
            loop.bpm,
            spec,
        )
        counts = loop.grid.onsets.sum(axis=1)
        rows.append(
            {
                "clip": len(rows) + 1,
                "performance": p["id"],
                "style": p["style"],
                "bpm": p["bpm"],
                "split": p["split"],
                "start_bar": loop.start_bar + 1,
                "original_midi": f"{stem}_original.mid",
                "grid_midi": f"{stem}_grid.mid",
                "pattern_id": pattern_id(loop.grid),
                **{f"{v}_hits": int(c) for v, c in zip(spec.voices, counts, strict=True)},
                "verdict": "",
                "notes": "",
            }
        )
    sheet = args.out / "ab_notes.csv"
    with sheet.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} loop pairs to {args.out}")
    print(
        f"Drag each *_original.mid and *_grid.mid pair into Logic; note verdicts in {sheet.name}."
    )


if __name__ == "__main__":
    main()
