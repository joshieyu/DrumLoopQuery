"""Render a small drum-library audition batch from the labeled GMD catalog.

Run this file in the IDE or ``python -m src.library.render_preview``. This
checkpoint uses the existing listening normalization and preserves pre-roll and
tails. It is not the final training library or a participant/kit split decision.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.music.rhythm import Grid, Note, clip_grid, pattern_id
from src.music.spec import MusicSpec, load_spec
from tools.listen.grid_ab import Loop, original_hits
from tools.render.kits import (
    PEAK_CEILING_DB,
    PRE_ROLL_S,
    TARGET_RMS_DB,
    check_kit,
    kit_by_id,
    kit_hits,
    load_program,
    normalize,
    write_wav,
)
from tools.render.sampler import Sampler

REPO_ROOT = Path(__file__).resolve().parents[2]


def select_records(catalog_dir: Path, *, per_split: int, spec: MusicSpec) -> list[dict]:
    """Select distinct source performances per split, excluding known conflicts.

    No split assignment is changed: shared patterns are omitted from this small
    audition only. This does not implement the final cross-split removal policy.
    """
    if per_split < 1:
        raise ValueError("per_split must be positive")
    summary = json.loads((catalog_dir / "summary.json").read_text())
    if summary["spec_version"] != spec.version:
        raise ValueError("Catalog spec version is stale; regenerate catalog_gmd first")
    if summary["failed_performances"]:
        raise ValueError("Catalog has failed performances; review failures.csv first")
    conflicts = set(pd.read_csv(catalog_dir / "split_conflicts.csv").pattern_id)
    selected, seen, counts = [], set(), {}
    with (catalog_dir / "loops.jsonl").open() as source:
        for line in source:
            row = json.loads(line)
            split, performance = row["split"], row["source_performance"]
            if row["pattern_id"] in conflicts or performance in seen:
                continue
            if counts.get(split, 0) >= per_split:
                continue
            seen.add(performance)
            counts[split] = counts.get(split, 0) + 1
            selected.append(row)
    if not selected:
        raise ValueError("No non-conflicting clips available for preview")
    return selected


def audio_events(record: dict, spec: MusicSpec) -> list[tuple[float, int, int]]:
    """Use Joshua's raw-hit ownership helper, not the whole labeling context.

    Keep original timing/velocity and dropped label categories. The listening
    helper owns the half-step onset boundary policy. Flams remain in drum audio.
    """
    notes = tuple(Note(**n) for n in record["notes"])
    grid_data = record["grid"]
    grid = Grid(
        tuple(grid_data["voices"]),
        grid_data["steps_per_beat"],
        np.asarray(grid_data["onsets"], dtype=bool),
    )
    actual = clip_grid(notes, bpm=record["tempo_bpm"], spec=spec)
    if actual != grid or pattern_id(actual) != record["pattern_id"]:
        raise ValueError(f"Stored rhythm label disagrees with current spec: {record['clip_id']}")
    loop = Loop(
        performance=record["source_performance"],
        start_bar=record["start_bar"],
        start_beat=record["start_beat"],
        beats=record["end_beat"] - record["start_beat"],
        bpm=record["tempo_bpm"],
        notes=notes,
        grid=grid,
    )
    return [
        (PRE_ROLL_S + beat * 60 / loop.bpm, pitch, velocity)
        for pitch, velocity, beat in original_hits(loop, spec)
    ]


def render_previews(
    catalog_dir: Path,
    out_dir: Path,
    *,
    kit_id: str,
    per_split: int = 3,
    seed: int = 0,
    kits_dir: Path | None = None,
) -> dict:
    """Write WAVs, a manifest with the team's base columns, and an audition page."""
    spec = load_spec()
    kit = kit_by_id(spec, kit_id)
    program = load_program(kit) if kits_dir is None else load_program(kit, kits_dir=kits_dir)
    sampler = Sampler(program, seed=seed)
    problems = check_kit(kit, program, spec)
    if problems:
        raise ValueError("Kit mapping checks failed:\n" + "\n".join(problems))
    selected = select_records(catalog_dir, per_split=per_split, spec=spec)
    catalog_summary = json.loads((catalog_dir / "summary.json").read_text())
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, sections = [], []
    for record in selected:
        bpm = record["tempo_bpm"]
        if not spec.rendering.tempo_bpm[0] <= bpm <= spec.rendering.tempo_bpm[1]:
            raise ValueError(f"Recorded tempo is outside render range: {bpm}")
        events = audio_events(record, spec)
        hits = kit_hits(kit, events)
        # Keep silence through the full pattern and permit natural one-shot tails.
        duration = PRE_ROLL_S + (record["end_beat"] - record["start_beat"]) * 60 / bpm
        raw = sampler.render(hits)
        frames = max(len(raw), round(duration * sampler.sample_rate))
        raw = np.pad(raw, ((0, frames - len(raw)), (0, 0)))
        if not np.isfinite(raw).all() or not np.any(raw):
            raise ValueError(f"Silent/nonfinite render: {record['clip_id']}")
        audio = normalize(raw)
        digest = hashlib.sha256(record["clip_id"].encode()).hexdigest()[:16]
        relative = Path("audio") / f"{kit_id}_{digest}.wav"
        path = out_dir / relative
        write_wav(path, audio, sampler.sample_rate)
        try:
            audio_path = path.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            audio_path = str(path)
        row = {
            "clip_id": f"{record['clip_id']}:{kit_id}:recorded:seed{seed}",
            "pattern_id": record["pattern_id"],
            "dualized_pattern_id": record["dualized_pattern_id"],
            "split": record["split"],
            "kit_id": kit_id,
            "tempo_bpm": bpm,
            "audio_path": audio_path,
            "midi_path": record["midi_path"],
            "source_performance": record["source_performance"],
            "start_bar": record["start_bar"],
            "kit_split": kit.split,
            "sample_rate": sampler.sample_rate,
            "channels": audio.shape[1],
            "duration_s": len(audio) / sampler.sample_rate,
            "pattern_duration_s": duration - PRE_ROLL_S,
            "pre_roll_s": PRE_ROLL_S,
            "rendered_hits": len(hits),
            "peak_dbfs": 20 * np.log10(float(np.abs(audio).max())),
            "rms_dbfs": 20 * np.log10(float(np.sqrt(np.mean(audio.astype(float) ** 2)))),
            "seed": seed,
            "status": "audition_only",
        }
        rows.append(row)
        title = html.escape(
            f"{record['split']} · {record['source_performance']} · bar {record['start_bar']}"
        )
        sections.append(
            f"<section><h2>{title}</h2><p>{bpm:g} BPM · "
            f"pattern <code>{record['pattern_id']}</code></p>"
            f'<audio controls preload="none" src="{relative.as_posix()}"></audio>'
            "<p>Listen for the first downbeat, complete drum parts, cut boundaries, "
            "and natural tails. Bar numbers start at zero.</p></section>"
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(out_dir / "library_manifest.csv", index=False)
    review = manifest[["clip_id", "source_performance", "start_bar"]].copy()
    review_path = out_dir / "listening_notes.csv"
    if review_path.exists():
        previous = pd.read_csv(review_path, keep_default_na=False)
        review = review.merge(
            previous[["clip_id", "verdict", "notes"]],
            on="clip_id",
            how="left",
            validate="one_to_one",
        ).fillna("")
    else:
        review = review.assign(verdict="", notes="")
    review.to_csv(review_path, index=False)
    summary = {
        "spec_version": spec.version,
        "kit_id": kit_id,
        "kit_split": kit.split,
        "rendered_clips": len(rows),
        "clips_by_split": manifest.split.value_counts().to_dict(),
        "seed": seed,
        "gmd_root": catalog_summary["gmd_root"],
        "audio_path_base": str(REPO_ROOT),
        "midi_path_base": catalog_summary["gmd_root"],
        "catalog_directory": str(catalog_dir.resolve()),
        "normalization": {
            "policy": "existing audition RMS with peak ceiling; not final library loudness",
            "target_rms_dbfs": TARGET_RMS_DB,
            "peak_ceiling_dbfs": PEAK_CEILING_DB,
        },
        "pre_roll_s": PRE_ROLL_S,
        "status": "audition_only; draft mapping and kit split/listening approval pending",
        "boundary_policy": "original_hits listening helper; natural timing, pre-roll, and tails",
        "conflict_policy": "exclude known cross-split pattern conflicts from preview only",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out_dir / "preview.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>Drum library preview</title><style>"
        "body{font:16px system-ui;max-width:850px;margin:40px auto;padding:0 20px;line-height:1.5}"
        "section{border-top:1px solid #ddd;padding:15px 0}h2{font-size:18px}"
        "</style><h1>Drum library: first audition batch</h1>"
        f"<p>Kit: {html.escape(kit_id)}. These are drum performances rendered from our "
        "labeled MIDI catalog, with a short lead-in and natural tails.</p>"
        "<p>Draft mapping and listening levels; these files are not the final training library. "
        "The source split is preserved, but this kit has not been assigned to a split.</p>"
        '<p><a href="library_manifest.csv">Manifest</a> · '
        '<a href="listening_notes.csv">Listening notes</a> · '
        '<a href="summary.json">Summary</a></p>' + "".join(sections) + "</html>"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=REPO_ROOT / "results/gmd_catalog")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results/library_preview")
    parser.add_argument("--kit", help="Kit ID from the spec; defaults to its first configured kit")
    parser.add_argument("--per-split", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    summary = render_previews(
        args.catalog,
        args.out,
        kit_id=args.kit or load_spec().rendering.kits[0].kit_id,
        per_split=args.per_split,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))
    print(f"Listen: {(args.out / 'preview.html').resolve()}")


if __name__ == "__main__":
    main()
