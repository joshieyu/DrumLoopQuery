from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from src.eval.evaluate import EvalError, evaluate, load_relevance
from src.eval.relevance import LIBRARY_COLUMNS, read_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
## how many standard errors the random system may sit from chance before it counts as broken
RANDOM_TOLERANCE_SE = 4.0


def oracle_scores(
    query_ids: Sequence[str], library_ids: Sequence[str], relevance: Mapping[str, Sequence[str]]
) -> np.ndarray:
    column = {clip_id: i for i, clip_id in enumerate(library_ids)}
    scores = np.zeros((len(query_ids), len(library_ids)))
    for row, query_id in enumerate(query_ids):
        ## each relevant clip gets a slightly different score so none of them tie
        for rank, clip_id in enumerate(sorted(relevance[query_id])):
            scores[row, column[clip_id]] = 1.0 - 0.001 * rank
    return scores


def run_sanity(
    library_ids: Sequence[str],
    relevance: Mapping[str, Sequence[str]],
    out_dir: Path,
    seed: int = 0,
) -> dict:
    query_ids = sorted(relevance)
    library_ids = list(library_ids)
    rng = np.random.default_rng(seed)
    systems = {
        "oracle": oracle_scores(query_ids, library_ids, relevance),
        "random": rng.random((len(query_ids), len(library_ids))),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for name, scores in systems.items():
        path = out_dir / f"{name}.npz"
        np.savez(
            path,
            query_ids=np.array(query_ids),
            library_ids=np.array(library_ids),
            scores=scores,
        )
        table, counts = evaluate(path, relevance)
        result[name] = table
        result[f"{name}_counts"] = counts
    oracle, random = result["oracle"], result["random"]
    chance = float(random["chance_rr"].mean())
    stderr = float(random["rr"].std(ddof=1) / np.sqrt(len(random)))
    oracle_ok = bool((oracle["rr"] == 1.0).all()) and bool((oracle["hit@1"] == 1.0).all())
    gap = abs(float(random["rr"].mean()) - chance)
    random_ok = gap <= RANDOM_TOLERANCE_SE * stderr
    return {
        "n_queries": len(query_ids),
        "n_library": len(library_ids),
        "oracle_mrr": float(oracle["rr"].mean()),
        "random_mrr": float(random["rr"].mean()),
        "chance_mrr": chance,
        "random_stderr": stderr,
        "oracle_ok": oracle_ok,
        "random_ok": random_ok,
        "passed": oracle_ok and random_ok,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an oracle and a random system through the real answer key."
    )
    parser.add_argument(
        "--library", type=Path, default=REPO_ROOT / "manifests/library_manifest.csv"
    )
    parser.add_argument("--relevance", type=Path, default=REPO_ROOT / "manifests/relevance.json")
    parser.add_argument("--pool-split", default="test")
    parser.add_argument("--pool-kits", nargs="+", metavar="KIT_ID")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results/eval/sanity")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    try:
        library = read_manifest(args.library, (*LIBRARY_COLUMNS, "kit_id"))
        pool = library[library["split"] == args.pool_split]
        if args.pool_kits:
            pool = pool[pool["kit_id"].isin(args.pool_kits)]
        result = run_sanity(
            pool["clip_id"].tolist(), load_relevance(args.relevance), args.out, args.seed
        )
    except (OSError, EvalError, KeyError) as error:
        parser.exit(1, f"Sanity check failed: {error}\n")
    print(f"queries {result['n_queries']}, library clips {result['n_library']}")
    oracle_flag = "ok" if result["oracle_ok"] else "FAIL"
    random_flag = "ok" if result["random_ok"] else "FAIL"
    print(f"oracle MRR {result['oracle_mrr']:.4f}  (must be 1.0000)  {oracle_flag}")
    print(
        f"random MRR {result['random_mrr']:.4f}  chance {result['chance_mrr']:.4f} "
        f"+/- {result['random_stderr']:.4f}  {random_flag}"
    )
    print("PASSED" if result["passed"] else "FAILED")
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
