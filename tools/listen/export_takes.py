"""Turn the recording session's usable takes into a query manifest and a zip for the team.

    .venv/bin/python -m tools.listen.export_takes

Reads the record page's log (data/recordings/team/takes.csv). Keeps the click takes the
music lead marked usable and, where a member has more than one for a beat, the latest
(spec §8, Round 2: one usable click take per beat). Writes:

- manifests/team_query_manifest.csv: one row per kept take, with query_manifest.csv's
  columns (README §12), so B can append it and D can build its answer key. The columns
  that only describe the fake-query generator stay empty.
- data/recordings/team_queries.zip: the kept WAVs at their repo paths plus the manifest,
  so unzipping it at the repo root puts every file where the manifest says.

The audio is the team's own voices: share the zip inside the team only, never commit it.
"""

from __future__ import annotations

import argparse
import csv
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TAKES = REPO / "data" / "recordings" / "team" / "takes.csv"
QUERY_MANIFEST = REPO / "manifests" / "query_manifest.csv"
OUT = REPO / "manifests" / "team_query_manifest.csv"
ZIP = REPO / "data" / "recordings" / "team_queries.zip"
# what D's answer key and manifest checks read (src/eval/relevance.py, check_manifests.py)
NEEDED = (
    "query_id",
    "pattern_id",
    "dualized_pattern_id",
    "query_type",
    "source",
    "split",
    "source_clip_id",
    "participant_id",
    "audio_path",
)


class ExportError(Exception):
    pass


def read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def select_takes(takes: list[dict]) -> list[dict]:
    """The latest usable click take for each member and beat."""
    kept: dict[tuple[str, str], dict] = {}
    for row in takes:
        if row.get("mode") != "click" or row.get("usable") != "yes":
            continue
        key = (row["participant_id"], row["source_clip_id"])
        if key not in kept or int(row["take"]) > int(kept[key]["take"]):
            kept[key] = row
    return sorted(kept.values(), key=lambda r: (r["participant_id"], r["query_id"]))


def check_rows(rows: list[dict], root: Path) -> None:
    if not rows:
        raise ExportError("no click take is marked usable yet")
    for row in rows:
        empty = [c for c in NEEDED if c != "dualized_pattern_id" and not row.get(c)]
        if empty:
            raise ExportError(f"{row.get('query_id', '?')} has no {', '.join(empty)}")
    missing = [r["audio_path"] for r in rows if not (root / r["audio_path"]).is_file()]
    if missing:
        raise ExportError(f"{len(missing)} takes have no audio here, e.g. {missing[0]}")


def write_manifest(rows: list[dict], columns: list[str], path: Path) -> None:
    lacking = [c for c in NEEDED if c not in columns]
    if lacking:
        raise ExportError(f"the query manifest header has no {', '.join(lacking)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, restval="", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def write_zip(rows: list[dict], manifest: Path, path: Path, root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(manifest, f"manifests/{manifest.name}")
        for row in rows:
            z.write(root / row["audio_path"], row["audio_path"])
    tmp.replace(path)


def summary(rows: list[dict]) -> str:
    people = sorted({r["participant_id"] for r in rows})
    each = ", ".join(f"{p} {sum(r['participant_id'] == p for r in rows)}" for p in people)
    loops = len({r["source_clip_id"] for r in rows})
    return f"{len(rows)} queries ({each}) over {loops} loops"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--takes", type=Path, default=TAKES)
    ap.add_argument(
        "--columns-from",
        type=Path,
        default=QUERY_MANIFEST,
        help="manifest whose header the output copies",
    )
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--zip", type=Path, default=ZIP)
    ap.add_argument("--root", type=Path, default=REPO, help="what audio_path is relative to")
    args = ap.parse_args(argv)
    try:
        _, takes = read_csv(args.takes)
        with args.columns_from.open(newline="") as f:
            columns = next(csv.reader(f))
        rows = select_takes(takes)
        check_rows(rows, args.root)
        write_manifest(rows, columns, args.out)
        write_zip(rows, args.out, args.zip, args.root)
    except (OSError, StopIteration, ExportError) as exc:
        print(f"Can't export the takes: {exc}", file=sys.stderr)
        return 1
    print(f"{summary(rows)}; latest usable click take per beat")
    print(f"Wrote {args.out} and {args.zip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
