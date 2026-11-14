from __future__ import annotations

import argparse
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from src.eval.evaluate import EvalError
from src.eval.relevance import LIBRARY_COLUMNS, QUERY_COLUMNS, read_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]

## columns the checks need on top of what the answer key needs
LIBRARY_EXTRA = ("kit_id", "source_clip_id")
QUERY_EXTRA = ("source_clip_id", "participant_id")
PARTICIPANT_COLUMNS = ("participant_id", "split")


class Finding(NamedTuple):
    level: str
    text: str


def _preview(values, limit: int = 3) -> str:
    shown = sorted(map(str, values))[:limit]
    return ", ".join(shown) + (", ..." if len(values) > limit else "")


def check_manifests(
    library: pd.DataFrame,
    queries: pd.DataFrame,
    participants: pd.DataFrame,
    pool_split: str = "test",
) -> list[Finding]:
    findings: list[Finding] = []

    def fail(text: str) -> None:
        findings.append(Finding("FAIL", text))

    def warn(text: str) -> None:
        findings.append(Finding("WARN", text))

    def info(text: str) -> None:
        findings.append(Finding("INFO", text))

    ## ids must be unique
    for name, frame, column in (("library", library, "clip_id"), ("queries", queries, "query_id")):
        repeated = frame.loc[frame[column].duplicated(), column]
        if len(repeated):
            fail(
                f"{name}: {len(repeated)} duplicate {column} values, e.g. {_preview(set(repeated))}"
            )

    ## test kits must never be used in another split
    kits = library.groupby("split")["kit_id"].agg(set)
    if pool_split not in kits:
        fail(f"library has no clips with split == {pool_split!r}")
        return findings
    for split, split_kits in kits.items():
        info(
            f"library split {split}: {len(split_kits)} kits, {(library.split == split).sum()} clips"
        )
    for split, split_kits in kits.items():
        shared = kits[pool_split] & split_kits
        if split != pool_split and shared:
            fail(f"kits used in both {pool_split} and {split}: {_preview(shared)}")

    ## patterns shared between the test pool and other splits are a leakage decision
    pool_patterns = set(library.loc[library.split == pool_split, "pattern_id"])
    other_patterns = set(library.loc[library.split != pool_split, "pattern_id"])
    leaked = pool_patterns & other_patterns
    if leaked:
        warn(
            f"{len(leaked)} of {len(pool_patterns)} {pool_split} patterns "
            f"({100 * len(leaked) / len(pool_patterns):.2f}%) also appear in other splits"
        )
    else:
        info(f"no {pool_split} pattern appears in another split")

    ## every query must point at a real library source clip with the same pattern and split
    by_source = library.groupby("source_clip_id").agg(
        lib_patterns=("pattern_id", "nunique"),
        lib_pattern=("pattern_id", "first"),
        lib_splits=("split", "nunique"),
        lib_split=("split", "first"),
    )
    if (by_source.lib_patterns > 1).any():
        n_multi = (by_source.lib_patterns > 1).sum()
        fail(f"{n_multi} library source clips have more than one pattern_id")
    if (by_source.lib_splits > 1).any():
        fail(
            f"{(by_source.lib_splits > 1).sum()} library source clips appear in more than one split"
        )
    linked = queries[queries.source_clip_id.notna()].join(by_source, on="source_clip_id")
    no_source = queries.source_clip_id.isna().sum()
    if no_source:
        info(f"{no_source} queries have no source_clip_id (skipped in the source-clip checks)")
    missing = linked[linked.lib_pattern.isna()]
    if len(missing):
        fail(f"{len(missing)} queries point at a source clip that is not in the library")
    found = linked[linked.lib_pattern.notna()]
    wrong_pattern = found[found.pattern_id != found.lib_pattern]
    if len(wrong_pattern):
        fail(f"{len(wrong_pattern)} queries have a different pattern_id from their source clip")
    wrong_split = found[found.split != found.lib_split]
    if len(wrong_split):
        fail(f"{len(wrong_split)} queries are in a different split from their source clip")

    ## synthetic queries must carry a pattern id
    synthetic = queries[queries.source == "synthetic"]
    if synthetic.pattern_id.isna().any():
        fail(f"{synthetic.pattern_id.isna().sum()} synthetic queries have no pattern_id")

    ## each synthetic query's participant must belong to the same split as the query
    owner = participants.set_index("participant_id")["split"]
    unknown = set(synthetic.participant_id.dropna()) - set(owner.index)
    if unknown:
        fail(f"participants not in the split file: {_preview(unknown)}")
    known = synthetic[synthetic.participant_id.isin(owner.index)]
    mismatch = known[known.participant_id.map(owner) != known.split]
    if len(mismatch):
        fail(f"{len(mismatch)} queries are in a different split from their participant")

    ## what the test set looks like
    test_queries = queries[queries.split == pool_split]
    for key, count in test_queries.groupby(["query_type", "source"]).size().items():
        info(f"{pool_split} queries {key[0]}/{key[1]}: {count}")
    for kit, count in library[library.split == pool_split].groupby("kit_id").size().items():
        info(f"{pool_split} pool kit {kit}: {count} clips")
    return findings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check the manifests before building the answer key."
    )
    parser.add_argument(
        "--library", type=Path, default=REPO_ROOT / "manifests/library_manifest.csv"
    )
    parser.add_argument("--queries", type=Path, default=REPO_ROOT / "manifests/query_manifest.csv")
    parser.add_argument(
        "--participants", type=Path, default=REPO_ROOT / "manifests/avp_participant_splits.csv"
    )
    parser.add_argument("--pool-split", default="test")
    args = parser.parse_args()
    try:
        findings = check_manifests(
            read_manifest(args.library, (*LIBRARY_COLUMNS, *LIBRARY_EXTRA)),
            read_manifest(args.queries, (*QUERY_COLUMNS, *QUERY_EXTRA)),
            read_manifest(args.participants, PARTICIPANT_COLUMNS),
            args.pool_split,
        )
    except (OSError, EvalError) as error:
        parser.exit(1, f"Manifest check failed: {error}\n")
    for finding in findings:
        print(f"{finding.level:5} {finding.text}")
    failures = sum(f.level == "FAIL" for f in findings)
    print(f"{failures} failure(s), {sum(f.level == 'WARN' for f in findings)} warning(s)")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
