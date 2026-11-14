from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.eval.evaluate import EvalError, load_relevance
from src.eval.relevance import read_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Pool:
    ## both tables are sorted by id, so every system sees the same order
    library: pd.DataFrame
    queries: pd.DataFrame


def load_pool(
    library_path: str | Path,
    queries_path: str | Path,
    relevance_path: str | Path,
    pool_split: str = "test",
    pool_kits: Sequence[str] | None = None,
) -> Pool:
    relevance = load_relevance(relevance_path)
    required = ("clip_id", "split", "audio_path", *(("kit_id",) if pool_kits else ()))
    library = read_manifest(library_path, required)
    library = library[library["split"] == pool_split]
    if pool_kits:
        library = library[library["kit_id"].isin(pool_kits)]
    if library.empty:
        raise EvalError(f"no library clips with split {pool_split!r} in the pool")
    if library["audio_path"].isna().any():
        raise EvalError("some pool clips have no audio_path")

    queries = read_manifest(queries_path, ("query_id", "audio_path"))
    absent = sorted(set(relevance) - set(queries["query_id"]))
    if absent:
        raise EvalError(
            f"{len(absent)} answer-key queries are not in the query manifest: {absent[:3]}"
        )
    queries = queries[queries["query_id"].isin(relevance)]
    if queries["audio_path"].isna().any():
        raise EvalError("some answer-key queries have no audio_path")

    ## every clip the answer key calls relevant must be a candidate
    outside = {c for clips in relevance.values() for c in clips} - set(library["clip_id"])
    if outside:
        raise EvalError(
            f"{len(outside)} relevant clips are not in the pool, e.g. {sorted(outside)[:3]}"
        )

    return Pool(
        library=library[["clip_id", "audio_path"]].sort_values("clip_id").reset_index(drop=True),
        queries=queries[["query_id", "audio_path"]].sort_values("query_id").reset_index(drop=True),
    )
