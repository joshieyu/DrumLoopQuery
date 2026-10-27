"""Inspect one GMD performance and extract its first raw loop window.

Run this file from the IDE with the project's .venv interpreter, or run
``python -m src.library.inspect_midi``. No audio is synthesized or played.
See docs/data_pipeline.md for the output format and next steps.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import pandas as pd
import pretty_midi

from src.library.clips import first_clip
from src.music.rhythm import Note
from src.music.spec import load_spec

REPO_ROOT = Path(__file__).resolve().parents[2]


def inspect_performance(gmd_dir: Path, out_dir: Path, performance_id: str | None = None) -> Path:
    """Export original drum hits, without filtering, quantizing, or labeling rhythms.

    The spec filters the default performance selection only. An explicit ID must
    also meet those filters. Raw hits remain available even when their voice is
    marked 'drop' in the spec; they may still be needed for rendered drum audio.
    """
    spec = load_spec()
    info = pd.read_csv(gmd_dir / "info.csv")
    low, high = spec.loops.performance_tempo_bpm
    eligible = info[
        info["time_signature"].isin(spec.loops.time_signatures) & info["bpm"].between(low, high)
    ]
    if spec.loops.exclude_fills:
        eligible = eligible[eligible["beat_type"] != "fill"]
    if performance_id is not None:
        eligible = eligible[eligible["id"] == performance_id]
    if eligible.empty:
        raise ValueError("No eligible performance found; check the ID and music spec.")
    # Deterministic selection independent of the metadata CSV's row order.
    row = eligible.sort_values("id").iloc[0]
    midi_path = gmd_dir / row["midi_filename"]
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    rows = []
    for instrument in midi.instruments:
        if not instrument.is_drum:
            continue
        for note in instrument.notes:
            # MIDI ticks / resolution expresses beats even across tempo changes.
            tick = midi.time_to_tick(note.start)
            group = spec.note_to_group.get(note.pitch, "unknown")
            rows.append(
                {
                    "pitch": note.pitch,
                    "velocity": note.velocity,
                    "tick": tick,
                    "beat": tick / midi.resolution,
                    "start_sec": note.start,
                    "end_sec": note.end,
                    "spec_group": group,
                }
            )
    if not rows:
        raise ValueError(f"No drum hits found in {midi_path}")
    hits = pd.DataFrame(rows).sort_values(["tick", "pitch"]).reset_index(drop=True)
    times, tempos = midi.get_tempo_changes()
    metadata = {
        "performance_id": row["id"],
        "midi_path": str(midi_path.resolve()),
        "source_split": row["split"],
        "metadata_bpm": float(row["bpm"]),
        "time_signature": row["time_signature"],
        "ticks_per_beat": midi.resolution,
        "drum_hits": len(hits),
        "unknown_pitches": sorted(
            hits.loc[hits.spec_group == "unknown", "pitch"].unique().tolist()
        ),
        "spec_version": spec.version,
        "tempo_changes": [
            {"time_sec": float(t), "bpm": float(b)} for t, b in zip(times, tempos, strict=True)
        ],
    }
    # The MIDI's final note end defines the available source span for this preview.
    duration_beats = midi.time_to_tick(midi.get_end_time()) / midi.resolution
    clip_html = ""
    clip_outputs = (out_dir / "clip_hits.csv", out_dir / "clip_metadata.json")
    try:
        clip = first_clip(
            [Note(int(r.pitch), int(r.velocity), float(r.beat)) for r in hits.itertuples()],
            duration_beats=duration_beats,
            time_signature=row["time_signature"],
            spec=spec,
        )
    except (ValueError, KeyError) as error:
        # Keep the raw inspection useful for short or unsupported performances.
        clip_html = "<p>Clip unavailable: " + html.escape(str(error)) + "</p>"
        for path in clip_outputs:
            path.unlink(missing_ok=True)  # Prevent an earlier run's clip being shown as current.
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        clip_hits = pd.DataFrame(clip.notes, columns=Note._fields)
        clip_hits["source_beat"] = clip_hits["beat"] + clip.start_beat
        clip_hits["spec_group"] = clip_hits["pitch"].map(spec.note_to_group)
        clip_metadata = {
            **metadata,
            "drum_hits": len(clip.notes),
            "start_bar": clip.start_bar,
            "bar_numbering": "zero-based",
            "start_beat": clip.start_beat,
            "end_beat_exclusive": clip.end_beat,
            "duration_beats": clip.end_beat - clip.start_beat,
            "source_duration_beats": duration_beats,
            "boundary_policy": "raw onsets in [start, end); no quantization",
            "status": "raw window only; no rhythm labels or audio",
        }
        clip_hits.to_csv(clip_outputs[0], index=False)
        clip_outputs[1].write_text(json.dumps(clip_metadata, indent=2) + "\n")
        clip_html = (
            "<pre>"
            + html.escape(json.dumps(clip_metadata, indent=2))
            + "</pre>"
            + clip_hits.head(30).to_html(index=False, float_format=lambda v: f"{v:.4f}")
            + '<p><a href="clip_hits.csv">All clip hits</a> · '
            '<a href="clip_metadata.json">Clip metadata</a></p>'
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    hits.to_csv(out_dir / "hits.csv", index=False)
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    report = out_dir / "report.html"
    report.write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>GMD MIDI inspection</title><style>"
        "body{font:16px system-ui;max-width:1000px;margin:40px auto;padding:0 20px;"
        "line-height:1.5}table{border-collapse:collapse}th,td{padding:6px 12px;"
        "border:1px solid #ddd}pre{white-space:pre-wrap;background:#f5f5f5;padding:16px}"
        "</style><h1>One GMD performance</h1>"
        "<p>This inspects a performance and extracts its first raw loop window. "
        "No rhythm labels or training examples have been generated.</p>"
        "<h2>Source and metadata</h2><pre>"
        + html.escape(json.dumps(metadata, indent=2))
        + "</pre><h2>First 30 drum hits</h2>"
        "<p>Pitch identifies the drum; velocity is hit strength; beat is the position "
        "from the start of the performance. The first beat is 0. "
        "A spec_group of drop means omitted from future rhythm labels, "
        "not removed from this raw table. Unknown pitches need review.</p>"
        + hits.head(30).to_html(index=False, float_format=lambda value: f"{value:.4f}")
        + '<p><a href="hits.csv">All drum hits (CSV)</a> · '
        '<a href="metadata.json">Metadata (JSON)</a></p>'
        "<h2>First raw clip</h2><p>Clip beat positions are relative to the window start. "
        "Quiet and dropped hits remain. Raw onset boundaries are provisional; "
        "early downbeats are not reassigned to adjacent windows.</p>"
        + clip_html
        + "<h2>Next checkpoint</h2><p>Confirm boundary handling with Joshua, then "
        "connect the shared rhythm-label functions when available. Full dataset "
        "generation still needs loop stride, labels, and audio.</p></html>"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gmd-dir", type=Path, default=REPO_ROOT / "data/raw/gmd/groove")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results/midi_inspection")
    parser.add_argument("--performance-id", help="Optional exact ID from GMD info.csv")
    args = parser.parse_args()
    try:
        report = inspect_performance(args.gmd_dir, args.out, args.performance_id)
    except (FileNotFoundError, ValueError) as error:
        parser.exit(1, f"Inspection failed: {error}\n")
    print(f"Inspection ready: {report}\nOpen this HTML file in your browser.")


if __name__ == "__main__":
    main()
