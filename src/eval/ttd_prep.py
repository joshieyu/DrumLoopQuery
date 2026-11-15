from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from src.music.rhythm import Note, clip_grid, has_min_hits, pattern_id
from src.music.spec import MusicSpec, load_spec

from src.eval.evaluate import EvalError
from src.eval.relevance import read_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "data/raw/taptamdrum/midi_files"

## the dataset's own hand mapping (hvo_sequence DUALIZATION_ROLAND_HAND_DRUM)
LEFT_PITCHES = frozenset({76, 60, 48, 72, 67})
RIGHT_PITCHES = frozenset({77, 74, 62, 86})
## a tap onset within this many grid steps of a source onset counts as the same hit
SKELETON_TOLERANCE_STEPS = 1

FOLDER = re.compile(
    r"^\[(?P<number>\d+) (?P<tested>P[P\d]*)\] "
    r"(?P<drummer>drummer\d+)-(?P<session>eval-session|session\d+)-(?P<take>\d+)_"
    r"(?P<style>.+)_(?P<bpm>\d+)_(?P<kind>beat|fill)_(?P<signature>\d+-\d+)"
    r"_best_2bar_segment_(?P<segment>\d+)$"
)
TAP_FILE = re.compile(
    r"^Participant_(?P<participant>\d+)_(?P<kind>repetition_\d+|simple|complex)\.mid$"
)


def parse_source_folder(name: str) -> dict:
    match = FOLDER.match(name)
    if not match:
        raise EvalError(f"unrecognised TapTamDrum folder name: {name}")
    fields = match.groupdict()
    session = "eval_session" if fields["session"] == "eval-session" else fields["session"]
    return {
        "number": fields["number"],
        "tested_with": fields["tested"],
        ## written the way GMD's info.csv and the library manifest write a performance id
        "gmd_performance": f"{fields['drummer']}/{session}/{fields['take']}",
        "is_eval_session": session == "eval_session",
        "style": fields["style"],
        "bpm": int(fields["bpm"]),
        "segment": int(fields["segment"]),
    }


def read_notes(path: Path) -> tuple[list[tuple[int, int, float]], float]:
    ## imported here so the rest of the package works without it
    import pretty_midi

    midi = pretty_midi.PrettyMIDI(str(path))
    tempos = midi.get_tempo_changes()[1]
    if len(tempos) != 1:
        raise EvalError(f"{path}: expected one tempo, found {len(tempos)}")
    notes = [
        (note.pitch, note.velocity, midi.time_to_tick(note.start) / midi.resolution)
        for instrument in midi.instruments
        for note in instrument.notes
    ]
    return sorted(notes, key=lambda n: (n[2], n[0])), float(tempos[0])


def _check_hands(notes: list[tuple[int, int, float]], path: Path) -> None:
    unknown = {p for p, _, _ in notes} - LEFT_PITCHES - RIGHT_PITCHES
    if unknown:
        raise EvalError(f"{path}: tap pitches not in either hand: {sorted(unknown)}")


def _voice_pitch(spec: MusicSpec, voice: str) -> int:
    return min(p for p, group in spec.note_to_group.items() if group == voice)


def source_grid(notes: list[tuple[int, int, float]], bpm: float, spec: MusicSpec):
    try:
        return clip_grid([Note(*n) for n in notes], bpm, spec)
    except KeyError as error:
        raise EvalError(f"source clip has a pitch the spec does not know: {error}") from error


def tap_steps(notes: list[tuple[int, int, float]], bpm: float, spec: MusicSpec) -> list[int]:
    ## both hands are put through the same cleanup and quantizing as a drum voice, so a
    ## tap and a source are compared on the same grid
    pitch = _voice_pitch(spec, spec.voices[0])
    grid = clip_grid([Note(pitch, velocity, beat) for _, velocity, beat in notes], bpm, spec)
    return np.flatnonzero(grid.onsets[0]).tolist()


def onset_f1(steps_a: list[int], steps_b: list[int], tolerance: int) -> float:
    if not steps_a or not steps_b:
        return 0.0
    a, b = np.array(steps_a), np.array(steps_b)
    found_a = np.mean([np.any(np.abs(b - s) <= tolerance) for s in a])
    found_b = np.mean([np.any(np.abs(a - s) <= tolerance) for s in b])
    return 0.0 if found_a + found_b == 0 else float(2 * found_a * found_b / (found_a + found_b))


def index_dataset(root: Path, spec: MusicSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    originals = sorted(root.rglob("original.mid"))
    if not originals:
        raise EvalError(f"no original.mid files under {root}")
    sources, taps = [], []
    for original in originals:
        folder = original.parent
        info = parse_source_folder(folder.name)
        notes, bpm = read_notes(original)
        grid = source_grid(notes, bpm, spec)
        steps = np.flatnonzero(grid.onsets.any(axis=0)).tolist()
        sources.append(
            {
                **info,
                "subset": folder.parent.parent.name,
                "pattern_id": pattern_id(grid),
                "meets_min_hits": has_min_hits(grid, spec),
                "n_hits": int(grid.onsets.sum()),
                "source_midi_path": original.relative_to(root).as_posix(),
            }
        )
        for tap_path in sorted(folder.glob("Participant_*.mid")):
            parsed = TAP_FILE.match(tap_path.name)
            if not parsed:
                raise EvalError(f"unrecognised tap file name: {tap_path}")
            tap_notes, tap_bpm = read_notes(tap_path)
            _check_hands(tap_notes, tap_path)
            onsets = tap_steps(tap_notes, tap_bpm, spec)
            taps.append(
                {
                    "tap_id": f"ttd{info['number']}_P{parsed['participant']}_{parsed['kind']}",
                    "number": info["number"],
                    "participant": f"P{parsed['participant']}",
                    "tap_kind": parsed["kind"],
                    "tempo_bpm": tap_bpm,
                    "n_taps": len(tap_notes),
                    "n_left": sum(p in LEFT_PITCHES for p, _, _ in tap_notes),
                    "n_right": sum(p in RIGHT_PITCHES for p, _, _ in tap_notes),
                    "skeleton_f1": onset_f1(onsets, steps, SKELETON_TOLERANCE_STEPS),
                    "tempo_matches_source": abs(tap_bpm - bpm) < 0.01,
                    "tap_midi_path": tap_path.relative_to(root).as_posix(),
                }
            )
    return pd.DataFrame(sources), pd.DataFrame(taps)


def match_sources(
    sources: pd.DataFrame,
    library: pd.DataFrame,
    pool_split: str = "test",
    pool_kits: list[str] | None = None,
) -> pd.DataFrame:
    pool = library[library["split"] == pool_split]
    if pool_kits:
        pool = pool[pool["kit_id"].isin(pool_kits)]
    if pool.empty:
        raise EvalError(f"library has no clips with split {pool_split!r} in the pool")
    clips_per_pattern = pool.groupby("pattern_id")["clip_id"].count()
    start_bars = pool.groupby("source_performance")["start_bar"].agg(
        lambda bars: {int(float(b)) for b in bars.dropna()}
    )
    elsewhere = set(library.loc[~library.index.isin(pool.index), "pattern_id"])
    out = sources.copy()
    out["performance_in_pool"] = out["gmd_performance"].isin(start_bars.index)
    out["segment_is_a_pool_start_bar"] = [
        segment in start_bars.get(performance, set())
        for performance, segment in zip(out["gmd_performance"], out["segment"], strict=True)
    ]
    out["n_pool_clips"] = out["pattern_id"].map(clips_per_pattern).fillna(0).astype(int)
    out["pattern_in_pool"] = out["n_pool_clips"] > 0
    out["pattern_in_other_splits"] = out["pattern_id"].isin(elsewhere)
    return out


def query_rows(sources: pd.DataFrame, taps: pd.DataFrame, pool_split: str = "test") -> pd.DataFrame:
    ## a tap is a query only when its source loop is in the pool. Its relevant clips are the
    ## pool clips of the SOURCE pattern: the tap is a human reduction of that loop, and a
    ## human tap never lands on the same grid steps as a rule-based reduction
    usable = sources[sources["pattern_in_pool"] & sources["meets_min_hits"]]
    merged = taps.merge(usable, on="number", how="inner", validate="many_to_one")
    return pd.DataFrame(
        {
            "query_id": merged["tap_id"],
            "pattern_id": merged["pattern_id"],
            "dualized_pattern_id": None,
            "query_type": "tap",
            "source": "taptamdrum",
            "split": pool_split,
            "participant_id": "ttd_" + merged["participant"],
            "tempo_bpm": merged["tempo_bpm"],
            "source_performance": merged["gmd_performance"],
            "start_bar": merged["segment"],
            "tap_kind": merged["tap_kind"],
            "midi_path": "data/raw/taptamdrum/midi_files/" + merged["tap_midi_path"],
            "audio_path": None,
            "relevance_key": "pattern_id",
        }
    )


def funnel(sources: pd.DataFrame, taps: pd.DataFrame, queries: pd.DataFrame) -> dict:
    usable = sources["pattern_in_pool"] & sources["meets_min_hits"]
    return {
        "sources_total": len(sources),
        "taps_total": len(taps),
        "sources_from_eval_sessions": int(sources["is_eval_session"].sum()),
        "sources_whose_performance_is_in_the_pool": int(sources["performance_in_pool"].sum()),
        "sources_whose_segment_is_a_pool_start_bar": int(
            sources["segment_is_a_pool_start_bar"].sum()
        ),
        "sources_whose_pattern_is_in_the_pool": int(sources["pattern_in_pool"].sum()),
        "sources_usable": int(usable.sum()),
        "sources_not_meeting_min_hits": int((~sources["meets_min_hits"]).sum()),
        "pool_patterns_also_in_other_splits": int(
            (sources["pattern_in_pool"] & sources["pattern_in_other_splits"]).sum()
        ),
        "tap_queries": len(queries),
        "tap_queries_by_kind": queries["tap_kind"]
        .str.replace(r"_\d+$", "", regex=True)
        .value_counts()
        .to_dict(),
        "skeleton_f1_median_all_taps": float(taps["skeleton_f1"].median()),
        "skeleton_f1_min_all_taps": float(taps["skeleton_f1"].min()),
        "taps_with_tempo_different_from_source": int((~taps["tempo_matches_source"]).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Index TapTamDrum and match its source loops to the test library."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="the midi_files folder")
    parser.add_argument(
        "--library", type=Path, default=REPO_ROOT / "manifests/library_manifest.csv"
    )
    parser.add_argument("--pool-split", default="test")
    parser.add_argument("--pool-kits", nargs="+", metavar="KIT_ID")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results/taptamdrum")
    parser.add_argument(
        "--manifest-out", type=Path, default=REPO_ROOT / "manifests/ttd_query_manifest.csv"
    )
    args = parser.parse_args()
    try:
        spec = load_spec()
        sources, taps = index_dataset(args.root, spec)
        library = read_manifest(
            args.library,
            ("clip_id", "pattern_id", "split", "source_performance", "start_bar"),
        )
        sources = match_sources(sources, library, args.pool_split, args.pool_kits)
        queries = query_rows(sources, taps, args.pool_split)
        report = {"spec_version": spec.version, **funnel(sources, taps, queries)}
    except (OSError, EvalError) as error:
        parser.exit(1, f"TapTamDrum prep failed: {error}\n")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sources.to_csv(args.out_dir / "sources.csv", index=False)
    taps.to_csv(args.out_dir / "taps.csv", index=False)
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    queries.to_csv(args.manifest_out, index=False)
    print(json.dumps(report, indent=2))
    print(f"Tap query rows: {args.manifest_out.resolve()}")


if __name__ == "__main__":
    main()
