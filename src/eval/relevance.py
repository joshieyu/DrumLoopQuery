"""Build the answer key, relevance.json, from the manifests and the spec's match rule.

The rule (music_spec.md §2, `same_rhythm.match: exact`): two clips are the same rhythm
exactly when their quantized grids are identical, which is when their IDs are equal.

- a **beatbox** query matches the evaluation-library clips with the same ``pattern_id``;
- a **tap** query matches the clips with the same ``dualized_pattern_id`` (the 2-voice
  reduction, so many different 3-voice patterns can share one tap pattern).

The evaluation library is the clips of ``library_manifest.csv`` whose ``split`` is the
pool split (default ``test``). Train-split clips are never candidates, so no query can be
"answered" by a clip the models trained on.

Output: ``{"query_id": ["clip_id", ...], ...}``, one line per query, sorted, so the file
diffs cleanly in git. Queries that cannot be scored (no pattern ID, or no matching clip in
the pool) are left out and counted in the report, never given an empty entry.

Run, from the repo root:

    python -m src.eval.relevance
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.evaluate import EvalError

REPO_ROOT = Path(__file__).resolve().parents[2]
ID_COLUMN = {"beatbox": "pattern_id", "tap": "dualized_pattern_id"}
LIBRARY_COLUMNS = ("clip_id", "pattern_id", "dualized_pattern_id", "split")
QUERY_COLUMNS = (
    "query_id",
    "pattern_id",
    "dualized_pattern_id",
    "query_type",
    "source",
    "split",
)


## reading


def read_manifest(path: str | Path, required: tuple[str, ...]) -> pd.DataFrame:
    """Read a manifest as text. Only an empty cell is null; whitespace is stripped."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise EvalError(f"{path}: missing columns {missing}")
    for column in frame.columns:
        stripped = frame[column].str.strip()
        frame[column] = stripped.mask(stripped == "", None)
    return frame


def _check_library(library: pd.DataFrame) -> None:
    for column in LIBRARY_COLUMNS:
        if column not in library.columns:
            raise EvalError(f"library manifest: missing column '{column}'")
        if library[column].isna().any():
            raise EvalError(f"library manifest: empty values in '{column}'")
    if library["clip_id"].duplicated().any():
        dup = library.loc[library["clip_id"].duplicated(), "clip_id"].iloc[0]
        raise EvalError(f"library manifest: duplicate clip_id, e.g. {dup}")


def _check_queries(queries: pd.DataFrame) -> None:
    for column in QUERY_COLUMNS:
        if column not in queries.columns:
            raise EvalError(f"query manifest: missing column '{column}'")
    for column in ("query_id", "query_type", "source", "split"):
        if queries[column].isna().any():
            raise EvalError(f"query manifest: empty values in '{column}'")
    if queries["query_id"].duplicated().any():
        dup = queries.loc[queries["query_id"].duplicated(), "query_id"].iloc[0]
        raise EvalError(f"query manifest: duplicate query_id, e.g. {dup}")
    unknown = sorted(set(queries["query_type"]) - set(ID_COLUMN))
    if unknown:
        raise EvalError(f"query manifest: query_type must be beatbox or tap, got {unknown}")


## building


def build_relevance(
    library: pd.DataFrame,
    queries: pd.DataFrame,
    *,
    match: str = "exact",
    pool_split: str = "test",
    query_split: str = "test",
    pool_kits: Sequence[str] | None = None,
) -> tuple[dict[str, list[str]], dict]:
    """Return (answer key, report) for the queries of `query_split`.

    The candidates are the library clips of `pool_split`, further limited to `pool_kits`
    (kit_id values) when given: the test pool is test patterns on the test kits.
    """
    if match != "exact":
        raise EvalError(
            f"same_rhythm.match is {match!r}; only 'exact' is implemented. A near-match "
            "rule needs the grids and a distance threshold."
        )
    _check_library(library)
    _check_queries(queries)
    pool = library[library["split"] == pool_split]
    if pool.empty:
        raise EvalError(f"library manifest has no clips with split == {pool_split!r}")
    if pool_kits is not None:
        if "kit_id" not in library.columns:
            raise EvalError("library manifest: --pool-kits needs a 'kit_id' column")
        absent = sorted(set(pool_kits) - set(pool["kit_id"].dropna()))
        if absent:
            raise EvalError(f"no {pool_split!r} clips on kits {absent}")
        pool = pool[pool["kit_id"].isin(pool_kits)]
    clips_by_id = {
        column: {key: sorted(group) for key, group in pool.groupby(column)["clip_id"]}
        for column in set(ID_COLUMN.values())
    }

    chosen = queries[queries["split"] == query_split]
    relevance: dict[str, list[str]] = {}
    groups: dict[str, dict] = {}
    for query in chosen.itertuples():
        key = ID_COLUMN[query.query_type]
        group = groups.setdefault(
            f"{query.query_type}/{query.source}",
            {
                "n_queries": 0,
                "n_scoreable": 0,
                "dropped_no_pattern_id": 0,
                "dropped_no_clip_in_pool": 0,
                "sizes": [],
            },
        )
        group["n_queries"] += 1
        pattern = getattr(query, key)
        if pd.isna(pattern):
            group["dropped_no_pattern_id"] += 1
            continue
        clips = clips_by_id[key].get(pattern)
        if not clips:
            group["dropped_no_clip_in_pool"] += 1
            continue
        relevance[query.query_id] = clips
        group["n_scoreable"] += 1
        group["sizes"].append(len(clips))

    for group in groups.values():
        sizes = np.array(group.pop("sizes"))
        group["relevant_set_size"] = (
            {"min": int(sizes.min()), "median": float(np.median(sizes)), "max": int(sizes.max())}
            if len(sizes)
            else None
        )

    other = library[library["split"] != pool_split]
    shared = sorted(set(pool["pattern_id"]) & set(other["pattern_id"]))
    report = {
        "match_rule": match,
        "pool_split": pool_split,
        "pool_kits": sorted(pool_kits) if pool_kits is not None else "all",
        "query_split": query_split,
        "pool_clips": len(pool),
        "pool_patterns": int(pool["pattern_id"].nunique()),
        "queries_in_manifest": len(queries),
        "queries_in_other_splits_ignored": len(queries) - len(chosen),
        "queries_with_answers": len(relevance),
        ## Test patterns that also appear in a non-pool split: leakage to decide on.
        "pool_patterns_shared_with_other_splits": len(shared),
        "by_testset": dict(sorted(groups.items())),
    }
    return dict(sorted(relevance.items())), report


def write_relevance(relevance: dict[str, list[str]], path: str | Path) -> None:
    """Write one line per query, so the file is readable and diffs well."""
    lines = [
        f"{json.dumps(query_id)}: {json.dumps(clips)}" for query_id, clips in relevance.items()
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("{\n" + ",\n".join(lines) + "\n}\n" if lines else "{}\n")


## command line


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the answer key (relevance.json) from the manifests and the spec."
    )
    parser.add_argument(
        "--library", type=Path, default=REPO_ROOT / "manifests/library_manifest.csv"
    )
    parser.add_argument("--queries", type=Path, default=REPO_ROOT / "manifests/query_manifest.csv")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "manifests/relevance.json")
    parser.add_argument(
        "--report", type=Path, default=REPO_ROOT / "results/eval/relevance_report.json"
    )
    parser.add_argument("--pool-split", default="test", help="library split used as candidates")
    parser.add_argument(
        "--pool-kits",
        nargs="+",
        metavar="KIT_ID",
        help="only these kits are candidates, e.g. --pool-kits salamander muldjordkit",
    )
    parser.add_argument("--query-split", default="test", help="query split to build entries for")
    args = parser.parse_args()
    from src.music.spec import load_spec  # after parsing, so --help works without the spec

    spec = load_spec()
    try:
        relevance, report = build_relevance(
            read_manifest(args.library, LIBRARY_COLUMNS),
            read_manifest(args.queries, QUERY_COLUMNS),
            match=spec.same_rhythm.match,
            pool_split=args.pool_split,
            query_split=args.query_split,
            pool_kits=args.pool_kits,
        )
    except (OSError, EvalError) as error:
        parser.exit(1, f"Answer key failed: {error}\n")
    report["spec_version"] = spec.version
    write_relevance(relevance, args.out)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Answer key: {args.out.resolve()}")


if __name__ == "__main__":
    main()
