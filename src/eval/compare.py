from __future__ import annotations

import argparse
import itertools
import json
from functools import cache
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.evaluate import EvalError, bootstrap_ci

REPO_ROOT = Path(__file__).resolve().parents[2]
## the columns of each system's per_query.csv, by the name shown in the table
METRICS = {"MRR": "rr", "Hit@1": "hit@1", "Hit@5": "hit@5", "Hit@10": "hit@10", "mAP": "ap"}
HIT_KS = {"Hit@1": 1, "Hit@5": 5, "Hit@10": 10}
N_PERMUTATIONS = 20_000
PERMUTATION_CHUNK = 2_000


def load_system(path: Path) -> tuple[pd.DataFrame, int | None]:
    ## accepts the results folder of one system, or its per_query.csv
    folder = path if path.is_dir() else path.parent
    csv = path / "per_query.csv" if path.is_dir() else path
    if not csv.exists():
        raise EvalError(f"no per_query.csv at {path}")
    table = pd.read_csv(csv)
    missing = {"query_id", *METRICS.values()} - set(table.columns)
    if missing:
        raise EvalError(f"{csv}: missing columns {sorted(missing)}")
    if table["query_id"].duplicated().any():
        raise EvalError(f"{csv}: duplicate query_id values")
    summary = folder / "summary.json"
    library = json.loads(summary.read_text()).get("n_library") if summary.exists() else None
    return table.set_index("query_id").sort_index(), library


def align(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    ## every system must have been scored on exactly the same queries
    names = list(tables)
    reference = set(tables[names[0]].index)
    for name in names[1:]:
        if set(tables[name].index) != reference:
            raise EvalError(f"{name} was scored on different queries from {names[0]}")
    return tables


def groups_from_manifest(query_ids: pd.Index, manifest_path: Path, column: str) -> np.ndarray:
    ## a query with no value counts as its own group
    manifest = pd.read_csv(manifest_path, dtype=str)
    if not {"query_id", column} <= set(manifest.columns):
        raise EvalError(f"{manifest_path}: needs query_id and {column} columns")
    owner = manifest.set_index("query_id")[column]
    mapped = pd.Series(query_ids, index=query_ids).map(owner)
    return mapped.fillna(pd.Series(query_ids, index=query_ids)).to_numpy(dtype=str)


@cache
def chance_hit(n_items: int, n_relevant: int, k: int) -> float:
    ## chance of at least one relevant clip in a random top k
    if k >= n_items:
        return 1.0
    return 1.0 - comb(n_items - n_relevant, k) / comb(n_items, k)


def paired_comparison(
    a: np.ndarray, b: np.ndarray, groups: np.ndarray, n_boot: int, n_perm: int, seed: int = 0
) -> dict:
    ## the difference a - b per query, resampled and sign-flipped by whole groups of queries
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    mean = float(diff.mean())
    low, high = bootstrap_ci(diff, groups, n_boot=n_boot, seed=seed)
    codes, uniques = pd.factorize(pd.Series(groups))
    sums = np.bincount(codes, weights=diff, minlength=len(uniques))
    rng = np.random.default_rng(seed)
    extreme = done = 0
    while done < n_perm:
        size = min(PERMUTATION_CHUNK, n_perm - done)
        signs = rng.choice([-1.0, 1.0], size=(size, len(uniques)))
        extreme += int((np.abs(signs @ sums) / len(diff) >= abs(mean) - 1e-12).sum())
        done += size
    return {
        "mean_difference": mean,
        "ci_low": low,
        "ci_high": high,
        "p_value": (extreme + 1) / (n_perm + 1),
        "n_groups": len(uniques),
    }


def holm(p_values: dict[tuple[str, str], float]) -> dict[tuple[str, str], float]:
    ## Holm's step-down correction for the number of pairs compared
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[tuple[str, str], float] = {}
    running = 0.0
    for rank, (pair, p) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * p))
        adjusted[pair] = running
    return adjusted


def participant_wins(
    a: np.ndarray, b: np.ndarray, participants: np.ndarray
) -> tuple[int, int, int]:
    ## in how many participants does a beat b, on average
    frame = pd.DataFrame({"a": a, "b": b, "who": participants}).groupby("who").mean()
    return (
        int((frame.a > frame.b).sum()),
        int((frame.a < frame.b).sum()),
        int((frame.a == frame.b).sum()),
    )


def summarize_system(
    table: pd.DataFrame, groups: np.ndarray, n_boot: int, seed: int
) -> dict[str, tuple[float, float, float]]:
    out = {}
    for label, column in METRICS.items():
        low, high = bootstrap_ci(table[column], groups, n_boot=n_boot, seed=seed)
        out[label] = (float(table[column].mean()), low, high)
    return out


def chance_row(table: pd.DataFrame, n_library: int | None) -> dict[str, float | None]:
    row: dict[str, float | None] = (
        {"MRR": float(table["chance_rr"].mean())}
        if ("chance_rr" in table.columns)
        else {"MRR": None}
    )
    for label, k in HIT_KS.items():
        if n_library is None or "n_relevant" not in table.columns:
            row[label] = None
        else:
            row[label] = float(
                np.mean([chance_hit(n_library, int(r), k) for r in table["n_relevant"]])
            )
    return row


def _tex(text: str) -> str:
    return text.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def latex_table(
    summary: dict[str, dict[str, tuple[float, float, float]]], chance: dict[str, float | None]
) -> str:
    def interval(mean: float, low: float, high: float) -> str:
        return f"{mean:.3f} ({low:.3f}--{high:.3f})"

    def percent(mean: float) -> str:
        return f"{100 * mean:.1f}\\%"

    lines = [
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"System & MRR (95\% CI) & Hit@1 & Hit@5 & Hit@10 & mAP \\",
        r"\midrule",
    ]
    cells = [chance["MRR"], chance.get("Hit@1"), chance.get("Hit@5"), chance.get("Hit@10")]
    if cells[0] is not None:
        hit_cells = ["--" if c is None else percent(c) for c in cells[1:]]
        lines.append(f"Random chance & {cells[0]:.3f} & " + " & ".join(hit_cells) + r" & -- \\")
    for name, metrics in summary.items():
        lines.append(
            f"{_tex(name)} & {interval(*metrics['MRR'])} & {percent(metrics['Hit@1'][0])} & "
            f"{percent(metrics['Hit@5'][0])} & {percent(metrics['Hit@10'][0])} & "
            f"{metrics['mAP'][0]:.3f} \\\\"
        )
    return "\n".join([*lines, r"\bottomrule", r"\end{tabular}"]) + "\n"


def compare(
    tables: dict[str, pd.DataFrame],
    groups: np.ndarray,
    participants: np.ndarray,
    n_library: int | None,
    n_boot: int,
    n_perm: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    tables = align(tables)
    names = list(tables)
    summary = {name: summarize_system(tables[name], groups, n_boot, seed) for name in names}
    chance = chance_row(tables[names[0]], n_library)
    rows = [
        {"system": name, "metric": metric, "mean": m, "ci_low": lo, "ci_high": hi}
        for name, metrics in summary.items()
        for metric, (m, lo, hi) in metrics.items()
    ]
    pairs = []
    for first, second in itertools.combinations(names, 2):
        a, b = tables[first]["rr"].to_numpy(), tables[second]["rr"].to_numpy()
        result = paired_comparison(a, b, groups, n_boot, n_perm, seed)
        wins = participant_wins(a, b, participants)
        pairs.append(
            {
                "system_a": first,
                "system_b": second,
                **result,
                "participants_a_better": wins[0],
                "participants_b_better": wins[1],
                "participants_tied": wins[2],
            }
        )
    pair_table = pd.DataFrame(pairs)
    if len(pair_table):
        adjusted = holm({(r.system_a, r.system_b): r.p_value for r in pair_table.itertuples()})
        pair_table["p_value_holm"] = [
            adjusted[(r.system_a, r.system_b)] for r in pair_table.itertuples()
        ]
    return pd.DataFrame(rows), pair_table, latex_table(summary, chance)


def parse_systems(items: list[str]) -> dict[str, Path]:
    systems = {}
    for item in items:
        name, separator, path = item.partition("=")
        if not separator or not name or not path:
            raise EvalError(f"--system must look like NAME=path, got {item!r}")
        systems[name] = Path(path)
    if len(systems) < 2:
        raise EvalError("give at least two systems to compare")
    return systems


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare systems with paired significance tests.")
    parser.add_argument(
        "--system",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="a system's results folder (or per_query.csv); repeat for each system",
    )
    parser.add_argument(
        "--query-manifest", type=Path, default=REPO_ROOT / "manifests/query_manifest.csv"
    )
    parser.add_argument("--group-column", default="pattern_id")
    parser.add_argument("--participant-column", default="participant_id")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--n-perm", type=int, default=N_PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    try:
        paths = parse_systems(args.system)
        loaded = {name: load_system(path) for name, path in paths.items()}
        tables = {name: table for name, (table, _) in loaded.items()}
        n_library = next((lib for _, lib in loaded.values() if lib), None)
        index = next(iter(tables.values())).index
        groups = groups_from_manifest(index, args.query_manifest, args.group_column)
        participants = groups_from_manifest(index, args.query_manifest, args.participant_column)
        summary, pairs, tex = compare(
            tables, groups, participants, n_library, args.n_boot, args.n_perm, args.seed
        )
    except (OSError, EvalError) as error:
        parser.exit(1, f"Comparison failed: {error}\n")
    args.out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out / "summary.csv", index=False)
    pairs.to_csv(args.out / "pairs.csv", index=False)
    (args.out / "table.tex").write_text(tex)
    mrr = summary[summary["metric"] == "MRR"].set_index("system")
    print(f"{len(index)} queries, {len(np.unique(groups))} groups by {args.group_column}\n")
    print("MRR (95% CI, resampled by group)")
    for name, row in mrr.iterrows():
        print(f"  {name:28} {row['mean']:.3f}  ({row['ci_low']:.3f} to {row['ci_high']:.3f})")
    print("\nPaired differences in MRR (a minus b)")
    for r in pairs.itertuples():
        print(
            f"  {r.system_a} vs {r.system_b}: {r.mean_difference:+.3f} "
            f"({r.ci_low:+.3f} to {r.ci_high:+.3f}), Holm p = {r.p_value_holm:.4f}, "
            f"participants {r.participants_a_better}-{r.participants_b_better}"
            f"-{r.participants_tied} (a better - b better - tied)"
        )
    print(f"\nWrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
