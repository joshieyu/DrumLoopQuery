"""The one scoring script: every system is scored here, from a file, the same way.

Input (README §8 and §12):

- a system file, ``.npz``, either
    * ``query_ids (Nq,)``, ``library_ids (Nl,)``, ``query_emb (Nq, d)``,
      ``library_emb (Nl, d)``, rows L2-normalized (scored by cosine similarity), or
    * ``query_ids``, ``library_ids``, ``scores (Nq, Nl)``, higher = better
      (for non-embedding systems such as B2 and B3);
- ``relevance.json``: ``{"query_id": ["clip_id", ...], ...}``, the answer key.

Metric definitions (state these in the report):

- **Rank** of a library clip: 1 + the number of clips scored strictly higher, counting
  every tied clip as ahead of it (*pessimistic* ties), so a system that returns constant
  scores gets no credit for the library's row order.
- **MRR**: mean over queries of 1 / (rank of the first relevant clip).
- **Hit@k**: fraction of queries with at least one relevant clip in the top k.
  (Not the fraction of relevant clips retrieved: with exact matching, a query's relevant
  set is the same loop on every test kit and tempo, so that fraction would mostly
  measure how many renders exist.)
- **mAP**: mean over queries of average precision over all relevant clips.
- **chance_mrr**: MRR a random ordering of the same library would get, so every number
  can be read against its floor.

Confidence intervals are percentile bootstrap intervals over queries. With a participant
grouping (``--query-manifest``) the bootstrap resamples participants instead, because one
person's queries are not independent of each other.

Run, from the repo root:

    python -m src.eval.evaluate --system results/embeddings/clap_syn_beatbox.npz \\
        --relevance manifests/relevance.json --out results/eval/clap_syn_beatbox
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KS = (1, 5, 10)
NORM_TOLERANCE = 1e-3  ## how far from 1.0 an embedding's length may be
BOOT_CHUNK = 500  ## bootstrap resamples drawn at a time, to bound memory


class EvalError(ValueError):
    """An input file is malformed or doesn't match the answer key.""" ## DON"T REMOVE THIS!!!!!!!! 


## 


def load_system(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a system file. Returns (query_ids, library_ids, scores), scores (Nq, Nl)."""
    with np.load(path, allow_pickle=False) as f:
        keys = set(f.files)
        for required in ("query_ids", "library_ids"):
            if required not in keys:
                raise EvalError(f"{path}: missing '{required}'")
        query_ids = f["query_ids"].astype(str)
        library_ids = f["library_ids"].astype(str)
        if "scores" in keys:
            scores = np.asarray(f["scores"], dtype=np.float64)
        elif {"query_emb", "library_emb"} <= keys:
            query_emb = np.asarray(f["query_emb"], dtype=np.float64)
            library_emb = np.asarray(f["library_emb"], dtype=np.float64)
            _check_embeddings(query_emb, len(query_ids), "query_emb", path)
            _check_embeddings(library_emb, len(library_ids), "library_emb", path)
            if query_emb.shape[1] != library_emb.shape[1]:
                raise EvalError(f"{path}: query and library embeddings differ in dimension")
            scores = query_emb @ library_emb.T
        else:
            raise EvalError(f"{path}: need 'scores', or 'query_emb' and 'library_emb'")
    if scores.shape != (len(query_ids), len(library_ids)):
        raise EvalError(
            f"{path}: scores have shape {scores.shape}, expected "
            f"({len(query_ids)}, {len(library_ids)})"
        )
    if not np.isfinite(scores).all():
        raise EvalError(f"{path}: scores contain NaN or infinity")
    for name, ids in (("query_ids", query_ids), ("library_ids", library_ids)):
        if len(set(ids)) != len(ids):
            raise EvalError(f"{path}: duplicate values in {name}")
    return query_ids, library_ids, scores


def _check_embeddings(emb: np.ndarray, n_rows: int, name: str, path: str | Path) -> None:
    if emb.ndim != 2 or emb.shape[0] != n_rows:
        raise EvalError(f"{path}: {name} has shape {emb.shape}, expected ({n_rows}, d)")
    if not np.isfinite(emb).all():
        raise EvalError(f"{path}: {name} contains NaN or infinity")
    worst = np.abs(np.linalg.norm(emb, axis=1) - 1.0).max()
    if worst > NORM_TOLERANCE:
        raise EvalError(
            f"{path}: {name} is not L2-normalized (a row's length is off by {worst:.3g})"
        )


def load_relevance(path: str | Path) -> dict[str, list[str]]:
    """Read the answer key: query_id -> list of relevant clip_ids."""
    with Path(path).open() as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise EvalError(f"{path}: expected a JSON object of query_id -> [clip_id, ...]")
    return {str(q): [str(c) for c in clips] for q, clips in raw.items()}


def relevant_indices(
    query_ids: np.ndarray, library_ids: np.ndarray, relevance: Mapping[str, Sequence[str]]
) -> dict[int, np.ndarray]:
    """Map each scoreable query's row to the library rows of its relevant clips.

    Queries missing from the answer key, or with no relevant clip, are left out (the
    caller reports how many). A relevant clip that isn't in the library is an error:
    the answer key must only list clips of the evaluation library.
    """
    column = {clip_id: i for i, clip_id in enumerate(library_ids)}
    out: dict[int, np.ndarray] = {}
    for row, query_id in enumerate(query_ids):
        clips = relevance.get(query_id, [])
        if not clips:
            continue
        unknown = [c for c in clips if c not in column]
        if unknown:
            raise EvalError(
                f"query {query_id}: relevant clips not in the library, e.g. {unknown[:3]}"
            )
        out[row] = np.unique([column[c] for c in clips])
    return out


## metrics


def rank_relevant(scores_row: np.ndarray, relevant_idx: np.ndarray) -> np.ndarray:
    """Pessimistic ranks (1 = best) of the relevant clips, sorted ascending.

    A clip's rank is the number of library clips scoring at least as high as it, itself
    included, so every tie counts against it.
    """
    ascending = np.sort(scores_row)
    below = np.searchsorted(ascending, scores_row[relevant_idx], side="left")
    return np.sort(len(scores_row) - below)


def query_metrics(ranks: np.ndarray, ks: Sequence[int] = DEFAULT_KS) -> dict[str, float]:
    """Metrics for one query from its relevant clips' sorted ranks."""
    first = int(ranks[0])
    out: dict[str, float] = {
        "n_relevant": len(ranks),
        "first_rank": first,
        "rr": 1.0 / first,
        "ap": float(np.mean(np.arange(1, len(ranks) + 1) / ranks)),
    }
    for k in ks:
        out[f"hit@{k}"] = float(first <= k)
    return out


@cache
def chance_rr(n_items: int, n_relevant: int) -> float:
    """Expected reciprocal rank when n_relevant of n_items are placed in random order."""
    if not 1 <= n_relevant <= n_items:
        raise ValueError(f"need 1 <= n_relevant <= n_items, got {n_relevant}, {n_items}")
    j = np.arange(0, n_items - n_relevant + 1)
    survive = np.concatenate([[1.0], np.cumprod((n_items - n_relevant - j) / (n_items - j))])
    r = np.arange(1, n_items - n_relevant + 2)
    p_first = survive[r - 1] * n_relevant / (n_items - r + 1)  # P(first relevant is at r)
    return float(np.sum(p_first / r))


def evaluate(
    system_path: str | Path,
    relevance: Mapping[str, Sequence[str]],
    ks: Sequence[int] = DEFAULT_KS,
) -> tuple[pd.DataFrame, dict]:
    """Score one system file. Returns (per-query table, bookkeeping counts)."""
    query_ids, library_ids, scores = load_system(system_path)
    relevant = relevant_indices(query_ids, library_ids, relevance)
    rows = []
    for row, idx in relevant.items():
        metrics = query_metrics(rank_relevant(scores[row], idx), ks)
        rows.append(
            {
                "query_id": query_ids[row],
                **metrics,
                "chance_rr": chance_rr(scores.shape[1], len(idx)),
            }
        )
    if not rows:
        raise EvalError(f"{system_path}: no query has a relevant clip in the answer key")
    table = pd.DataFrame(rows)
    counts = {
        "n_queries_scored": len(table),
        "n_queries_dropped": int(len(query_ids) - len(table)),
        "n_answer_key_queries_not_in_system": len(set(relevance) - set(query_ids)),
        "n_library": len(library_ids),
    }
    return table, counts


## statistics


def bootstrap_ci(
    values: Sequence[float],
    groups: Sequence[object] | None = None,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean of per-query values.

    With `groups` (e.g. participant ids), whole groups are resampled, which keeps one
    person's queries together and gives honest intervals when queries are not independent.
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n == 0:
        raise ValueError("no values to bootstrap")
    rng = np.random.default_rng(seed)
    if groups is None:
        codes, n_groups = None, n
    else:
        codes, uniques = pd.factorize(pd.Series(list(groups)))
        n_groups = len(uniques)
        sums = np.bincount(codes, weights=values, minlength=n_groups)
        sizes = np.bincount(codes, minlength=n_groups).astype(np.float64)
    means = []
    for start in range(0, n_boot, BOOT_CHUNK):
        size = min(BOOT_CHUNK, n_boot - start)
        pick = rng.integers(0, n_groups, size=(size, n_groups))
        if codes is None:
            means.append(values[pick].mean(axis=1))
        else:
            means.append(sums[pick].sum(axis=1) / sizes[pick].sum(axis=1))
    low, high = np.percentile(np.concatenate(means), [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(low), float(high)


def summarize(
    table: pd.DataFrame,
    groups: Sequence[object] | None = None,
    n_boot: int = 10_000,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Mean and 95% bootstrap interval for every reported metric."""
    names = {"rr": "mrr", "ap": "map", "chance_rr": "chance_mrr"}
    names.update({c: c for c in table.columns if c.startswith("hit@")})
    summary = {}
    for column, name in names.items():
        low, high = bootstrap_ci(table[column], groups, n_boot=n_boot, seed=seed)
        summary[name] = {"mean": float(table[column].mean()), "ci_low": low, "ci_high": high}
    return summary


## command line


def _groups_from_manifest(query_ids: pd.Series, manifest_path: Path) -> list[str]:
    """participant_id for each scored query; a query with none counts as its own group."""
    manifest = pd.read_csv(manifest_path, dtype=str)
    if not {"query_id", "participant_id"} <= set(manifest.columns):
        raise EvalError(f"{manifest_path}: needs query_id and participant_id columns")
    owner = manifest.set_index("query_id")["participant_id"]
    mapped = query_ids.map(owner)
    return mapped.fillna(query_ids).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score one retrieval system against the answer key (relevance.json)."
    )
    parser.add_argument("--system", type=Path, required=True, help="system .npz file")
    parser.add_argument("--relevance", type=Path, default=REPO_ROOT / "manifests/relevance.json")
    parser.add_argument("--out", type=Path, required=True, help="folder for the results")
    parser.add_argument(
        "--query-manifest",
        type=Path,
        help="query_manifest.csv; if given, the bootstrap resamples participants",
    )
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    try:
        table, counts = evaluate(args.system, load_relevance(args.relevance))
        groups = (
            _groups_from_manifest(table["query_id"], args.query_manifest)
            if args.query_manifest
            else None
        )
        summary = summarize(table, groups, n_boot=args.n_boot, seed=args.seed)
    except (OSError, EvalError) as error:
        parser.exit(1, f"Evaluation failed: {error}\n")
    result = {
        "system_file": str(args.system),
        **counts,
        "bootstrap": {
            "resampled": "participants" if groups is not None else "queries",
            "n_boot": args.n_boot,
            "seed": args.seed,
        },
        "tie_policy": "pessimistic",
        "hit_definition": "at least one relevant clip in the top k",
        "metrics": summary,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "per_query.csv", index=False)
    (args.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Results written to {args.out.resolve()}")


if __name__ == "__main__":
    main()
