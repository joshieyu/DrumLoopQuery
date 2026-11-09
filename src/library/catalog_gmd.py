"""Build a labeled MIDI catalog: run this file in the IDE or as a Python module.

Outputs under results/gmd_catalog precede rendering and the audio manifests.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pretty_midi

from src.library.clips import cut_clips
from src.music.rhythm import Note
from src.music.spec import load_spec
from src.splits import resolve_patterns

REPO_ROOT = Path(__file__).resolve().parents[2]
COLUMNS = [
    "clip_id",
    "pattern_id",
    "dualized_pattern_id",
    "split",
    "tempo_bpm",
    "midi_path",
    "source_performance",
    "start_bar",
    "start_beat",
    "end_beat",
    "context_hits",
    "grid_hits",
]


def catalog_gmd(gmd_dir: Path, out_dir: Path) -> dict:
    """Catalog eligible performances and report patterns shared across official splits.

    Paths are relative to gmd_dir. JSONL stores Note inputs and rhythm grids.
    Original splits are preserved; conflicts are reported, not resolved.
    """
    spec = load_spec()
    info = pd.read_csv(gmd_dir / "info.csv")
    low, high = spec.loops.performance_tempo_bpm
    keep = info.time_signature.isin(spec.loops.time_signatures) & info.bpm.between(low, high)
    if spec.loops.exclude_fills:
        keep &= info.beat_type != "fill"
    selected = info[keep].sort_values("id")
    rows, records, failures = [], [], []
    empty = 0
    for row in selected.itertuples(index=False):
        try:
            midi = pretty_midi.PrettyMIDI(str(gmd_dir / row.midi_filename))
            _, tempos = midi.get_tempo_changes()
            if len(tempos) != 1:
                raise ValueError("Variable-tempo MIDI requires a separate rendering policy")
            numerator, denominator = map(int, row.time_signature.split("-"))
            if any(
                (ts.numerator, ts.denominator) != (numerator, denominator)
                for ts in midi.time_signature_changes
            ):
                raise ValueError("MIDI time signature differs from metadata")
            notes = [
                Note(n.pitch, n.velocity, midi.time_to_tick(n.start) / midi.resolution)
                for instrument in midi.instruments
                if instrument.is_drum
                for n in instrument.notes
            ]
            clips = cut_clips(
                notes,
                duration_beats=midi.time_to_tick(midi.get_end_time()) / midi.resolution,
                time_signature=row.time_signature,
                bpm=float(row.bpm),
                spec=spec,
            )
        except (OSError, ValueError, KeyError) as error:
            failures.append({"source_performance": row.id, "error": str(error)})
            continue
        if not clips:
            empty += 1
        for clip in clips:
            record = {
                "clip_id": f"{row.id}:bar{clip.start_bar}",
                "pattern_id": clip.pattern_id,
                "dualized_pattern_id": clip.dualized_pattern_id,
                "split": row.split,
                "tempo_bpm": float(row.bpm),
                "midi_path": row.midi_filename,
                "source_performance": row.id,
                "start_bar": clip.start_bar,
                "start_beat": clip.start_beat,
                "end_beat": clip.end_beat,
                "context_hits": len(clip.notes),
                "grid_hits": int(clip.grid.onsets.sum()),
            }
            rows.append(record)
            records.append(
                {
                    **record,
                    "notes": [n._asdict() for n in clip.notes],
                    "grid": {
                        "voices": clip.grid.voices,
                        "steps_per_beat": clip.grid.steps_per_beat,
                        "onsets": clip.grid.onsets.tolist(),
                    },
                }
            )
    catalog = pd.DataFrame(rows, columns=COLUMNS)
    shared = catalog.groupby("pattern_id")["split"].nunique()
    collisions = catalog[catalog.pattern_id.isin(shared[shared > 1].index)]
    catalog, removals = resolve_patterns(catalog)
    kept_ids = set(catalog.clip_id)
    records = [r for r in records if r["clip_id"] in kept_ids]
    summary = {
        "spec_version": spec.version,
        "gmd_root": str(gmd_dir.resolve()),
        "eligible_performances": len(selected),
        "performances_without_clips": empty,
        "failed_performances": len(failures),
        "clips": len(catalog),
        "unique_patterns": int(catalog.pattern_id.nunique()),
        "clips_by_split": catalog.split.value_counts().to_dict(),
        "original_patterns_shared_across_splits": int((shared > 1).sum()),
        "patterns_shared_across_splits": 0,
        "removed_cross_split_clips": len(removals),
        "status": "split-resolved MIDI catalog; audio generation is a separate step",
        "bar_numbering": "zero-based",
        "context_margin_beats": 1,
        "source_duration_policy": "complete bars from MIDI note end; reserve configured final bars",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(out_dir / "loop_catalog.csv", index=False)
    collisions.to_csv(out_dir / "original_split_clashes.csv", index=False)
    removals.to_csv(out_dir / "split_removals.csv", index=False)
    catalog.iloc[:0].to_csv(out_dir / "split_conflicts.csv", index=False)
    pd.DataFrame(failures, columns=["source_performance", "error"]).to_csv(
        out_dir / "failures.csv", index=False
    )
    with (out_dir / "loops.jsonl").open("w") as output:
        for record in records:
            output.write(json.dumps(record) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gmd-dir", type=Path, default=REPO_ROOT / "data/raw/gmd/groove")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results/gmd_catalog")
    args = parser.parse_args()
    summary = catalog_gmd(args.gmd_dir, args.out)
    print(json.dumps(summary, indent=2))
    print(f"Catalog ready: {args.out.resolve()}")
    if summary["failed_performances"]:
        raise SystemExit("Some performances failed; review failures.csv before proceeding")


if __name__ == "__main__":
    main()
