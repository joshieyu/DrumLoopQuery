from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.compare import (
    align,
    groups_from_manifest,
    load_system,
    parse_systems,
)
from src.eval.evaluate import EvalError, bootstrap_ci

REPO_ROOT = Path(__file__).resolve().parents[2]
## label, column, and how a value becomes "badness" (bigger = messier) for each factor
FACTORS = {
    "offset": ("Timing offset (ms, absolute)", "offset_ms", np.abs),
    "jitter": ("Timing jitter (ms)", "jitter_ms", lambda v: v),
    "drift": ("Tempo drift (%, absolute)", "drift_pct", np.abs),
    "drop_prob": ("Chance of dropping a hit", "drop_prob", lambda v: v),
    "dropped_hits": ("Hits actually dropped", "dropped_hits", lambda v: v),
    "snr": ("Signal-to-noise (dB, lower is noisier)", "snr_db", lambda v: -v),
}
## the same values shown to the reader, not the "badness" version
DISPLAY = {"offset": np.abs, "drift": np.abs}
MIN_BIN_QUERIES = 30


def load_factors(manifest_path: Path, index: pd.Index) -> tuple[pd.DataFrame, pd.Series | None]:
    ## numeric factor columns (badness and display versions) plus the Personal/Fixed mode
    manifest = pd.read_csv(manifest_path).set_index("query_id")
    manifest = manifest.reindex(index)
    columns = {}
    for key, (_, column, badness) in FACTORS.items():
        if column not in manifest.columns:
            continue
        raw = pd.to_numeric(manifest[column], errors="coerce")
        columns[f"{key}"] = badness(raw)
        columns[f"{key}__shown"] = DISPLAY.get(key, lambda v: v)(raw)
    mode = manifest["mode"].astype(str) if "mode" in manifest.columns else None
    return pd.DataFrame(columns, index=index), mode


def bin_factor(badness: pd.Series, shown: pd.Series, n_bins: int):
    ## equal-count bins from cleanest to messiest; ties stay together; None if under 2 bins
    valid = badness.dropna()
    if valid.nunique() < 2:
        return None
    if valid.nunique() <= n_bins:
        order = {value: i for i, value in enumerate(sorted(valid.unique()))}
        codes = valid.map(order)
    else:
        codes = pd.qcut(valid, n_bins, labels=False, duplicates="drop")
    codes = codes.astype(int)
    if codes.nunique() < 2:
        return None
    labels = []
    for code in sorted(codes.unique()):
        values = shown[codes[codes == code].index]
        low, high = values.min(), values.max()
        labels.append(f"{low:.3g}" if low == high else f"{low:.3g} to {high:.3g}")
    return codes.reindex(badness.index), labels


def bin_scores(
    tables: dict[str, pd.DataFrame],
    groups: np.ndarray,
    codes: pd.Series,
    labels: list[str],
    factor: str,
    n_boot: int,
    seed: int,
) -> list[dict]:
    rows = []
    group_series = pd.Series(groups, index=codes.index)
    for name, table in tables.items():
        for code, label in enumerate(labels):
            mask = (codes == code).to_numpy()
            if not mask.any():
                continue
            values = table["rr"].to_numpy()[mask]
            low, high = bootstrap_ci(values, group_series[mask].to_numpy(), n_boot, seed=seed)
            rows.append(
                {
                    "factor": factor,
                    "system": name,
                    "bin": code + 1,
                    "range": label,
                    "n_queries": int(mask.sum()),
                    "mrr": float(values.mean()),
                    "ci_low": low,
                    "ci_high": high,
                    "small_bin": bool(mask.sum() < MIN_BIN_QUERIES),
                }
            )
    return rows


def _group_counts(groups: np.ndarray, n_boot: int, seed: int):
    ## how many times each whole group is drawn in each bootstrap round
    codes, uniques = pd.factorize(pd.Series(groups))
    rng = np.random.default_rng(seed)
    size = len(uniques)
    counts = rng.multinomial(size, np.full(size, 1.0 / size), size=n_boot)
    return codes, size, counts


def spearman_with_ci(
    badness: np.ndarray, rr: np.ndarray, groups: np.ndarray, n_boot: int, seed: int
) -> tuple[float, float, float]:
    ## rank correlation between messiness and reciprocal rank, intervals by resampling groups
    keep = ~np.isnan(badness)
    x = pd.Series(badness[keep]).rank().to_numpy()
    y = pd.Series(rr[keep]).rank().to_numpy()
    codes, size, counts = _group_counts(groups[keep], n_boot, seed)
    parts = np.stack([np.ones_like(x), x, y, x * x, y * y, x * y], axis=1)
    per_group = np.zeros((size, 6))
    np.add.at(per_group, codes, parts)

    def rho(totals: np.ndarray) -> np.ndarray:
        n = totals[..., 0]
        mx, my = totals[..., 1] / n, totals[..., 2] / n
        var_x = totals[..., 3] / n - mx**2
        var_y = totals[..., 4] / n - my**2
        with np.errstate(divide="ignore", invalid="ignore"):
            return (totals[..., 5] / n - mx * my) / np.sqrt(var_x * var_y)

    point = float(rho(per_group.sum(axis=0)))
    boot = rho(counts @ per_group)
    if np.isnan(boot).all():
        return point, float("nan"), float("nan")
    low, high = np.nanpercentile(boot, [2.5, 97.5])
    return point, float(low), float(high)


def joint_effects(
    badness: pd.DataFrame,
    rr: np.ndarray,
    groups: np.ndarray,
    n_boot: int,
    seed: int,
    binary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    ## change in MRR per one standard deviation of each factor, all factors in one model.
    ## binary columns (0/1, e.g. the query mode) enter as they are: their effect is the gap
    usable = [c for c in badness.columns if badness[c].nunique() > 1]
    frame = badness[usable].copy()
    extra = list(binary.columns) if binary is not None else []
    if extra:
        frame = frame.join(binary)
    keep = ~frame.isna().any(axis=1).to_numpy()
    if not usable or keep.sum() < len(usable) + len(extra) + 5:
        return pd.DataFrame(columns=["factor", "effect", "ci_low", "ci_high"])
    z = frame[usable].copy()
    z = ((z - z.mean()) / z.std(ddof=0)).join(frame[extra]).to_numpy()[keep]
    design = np.column_stack([np.ones(len(z)), z])
    y = rr[keep]
    codes, _, counts = _group_counts(groups[keep], n_boot, seed)
    point = np.linalg.lstsq(design, y, rcond=None)[0][1:]
    boot = np.empty((n_boot, len(usable) + len(extra)))
    for i in range(n_boot):
        root = np.sqrt(counts[i][codes])
        boot[i] = np.linalg.lstsq(design * root[:, None], y * root, rcond=None)[0][1:]
    low, high = np.percentile(boot, [2.5, 97.5], axis=0)
    return pd.DataFrame(
        {"factor": [*usable, *extra], "effect": point, "ci_low": low, "ci_high": high}
    )


def analyse(
    tables: dict[str, pd.DataFrame],
    groups: np.ndarray,
    factors: pd.DataFrame,
    mode: pd.Series | None,
    n_bins: int,
    n_boot: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tables = align(tables)
    bins, trends = [], []
    for key in FACTORS:
        if key not in factors.columns:
            continue
        binned = bin_factor(factors[key], factors[f"{key}__shown"], n_bins)
        if binned is None:
            continue
        codes, labels = binned
        bins += bin_scores(tables, groups, codes, labels, key, n_boot, seed)
        for name, table in tables.items():
            rho, low, high = spearman_with_ci(
                factors[key].to_numpy(), table["rr"].to_numpy(), groups, n_boot, seed
            )
            trends.append(
                {"factor": key, "system": name, "spearman": rho, "ci_low": low, "ci_high": high}
            )
    if mode is not None and mode.nunique() > 1:
        values = sorted(mode.unique())
        codes = mode.map({v: i for i, v in enumerate(values)})
        bins += bin_scores(tables, groups, codes, values, "mode", n_boot, seed)
    joint = []
    badness = factors[[k for k in FACTORS if k in factors.columns]]
    binary = None
    if mode is not None and mode.nunique() == 2:
        ## the second mode in alphabetical order is 1, so the effect reads "second minus first"
        first, second = sorted(mode.unique())
        binary = pd.DataFrame({f"mode_{second}_vs_{first}": (mode == second).astype(float)})
    for name, table in tables.items():
        result = joint_effects(
            badness, table["rr"].to_numpy(), groups, min(n_boot, 1000), seed, binary
        )
        result.insert(0, "system", name)
        joint.append(result)
    return (
        pd.DataFrame(bins),
        pd.DataFrame(trends),
        pd.concat(joint, ignore_index=True) if joint else pd.DataFrame(),
    )


def plot_bins(bins: pd.DataFrame, path: Path) -> bool:
    ## one panel per factor, one line per system; False if matplotlib is not installed
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    factors = list(dict.fromkeys(bins["factor"]))
    columns = 2
    rows = -(-len(factors) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(11, 3.2 * rows), squeeze=False)
    for ax, factor in zip(axes.flat, factors, strict=False):
        part = bins[bins["factor"] == factor]
        for name, group in part.groupby("system", sort=False):
            group = group.sort_values("bin")
            ax.plot(group["bin"], group["mrr"], marker="o", label=name)
            ax.fill_between(group["bin"], group["ci_low"], group["ci_high"], alpha=0.15)
        ticks = part.drop_duplicates("bin").sort_values("bin")
        ax.set_xticks(ticks["bin"])
        ax.set_xticklabels(ticks["range"], fontsize=7, rotation=15)
        ax.set_title(FACTORS[factor][0] if factor in FACTORS else "Query mode", fontsize=10)
        ax.set_ylabel("MRR")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
    for ax in list(axes.flat)[len(factors) :]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Retrieval quality as queries get messier (left = cleanest)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def print_report(bins: pd.DataFrame, trends: pd.DataFrame, joint: pd.DataFrame) -> None:
    for factor in dict.fromkeys(bins["factor"]):
        title = FACTORS[factor][0] if factor in FACTORS else "Query mode"
        wide = bins[bins["factor"] == factor].pivot_table(
            index=["bin", "range", "n_queries"], columns="system", values="mrr", sort=False
        )
        print(f"\nMRR by {title} (cleanest first)")
        print(wide.round(3).to_string())
    if len(trends):
        print("\nRank correlation between messiness and reciprocal rank (negative = hurts)")
        for r in trends.itertuples():
            print(
                f"  {r.factor:24} {r.system:14} {r.spearman:+.3f} "
                f"({r.ci_low:+.3f} to {r.ci_high:+.3f})"
            )
    if len(joint):
        print("\nChange in MRR per +1 SD of each factor, all factors and query mode together")
        for r in joint.itertuples():
            print(
                f"  {r.factor:24} {r.system:14} {r.effect:+.4f} "
                f"({r.ci_low:+.4f} to {r.ci_high:+.4f})"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score each system on cleaner versus messier synthetic queries."
    )
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
    parser.add_argument("--bins", type=int, default=4, help="number of equal-count bins")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    try:
        if args.bins < 2:
            raise EvalError("--bins must be at least 2")
        paths = parse_systems(args.system, minimum=1)
        tables = align({name: load_system(path)[0] for name, path in paths.items()})
        index = next(iter(tables.values())).index
        groups = groups_from_manifest(index, args.query_manifest, args.group_column)
        factors, mode = load_factors(args.query_manifest, index)
        if factors.empty and mode is None:
            raise EvalError(f"{args.query_manifest} has none of the sloppiness columns")
        bins, trends, joint = analyse(
            tables, groups, factors, mode, args.bins, args.n_boot, args.seed
        )
        if bins.empty:
            raise EvalError("no sloppiness column varies across these queries")
    except (OSError, EvalError) as error:
        parser.exit(1, f"Sloppiness analysis failed: {error}\n")
    args.out.mkdir(parents=True, exist_ok=True)
    bins.to_csv(args.out / "bins.csv", index=False)
    trends.to_csv(args.out / "trends.csv", index=False)
    joint.to_csv(args.out / "joint.csv", index=False)
    drawn = plot_bins(bins, args.out / "bins.png")
    print(f"{len(index)} queries, {len(np.unique(groups))} groups by {args.group_column}")
    print_report(bins, trends, joint)
    if not drawn:
        print("\n(matplotlib is not installed, so bins.png was not drawn: pip install matplotlib)")
    print(f"\nWrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
