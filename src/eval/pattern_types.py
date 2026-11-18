from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.compare import groups_from_manifest, load_system, parse_systems
from src.eval.evaluate import EvalError, bootstrap_ci

REPO_ROOT = Path(__file__).resolve().parents[2]
## a hit sits on a triplet position when it falls 1/3 or 2/3 of the way through a beat
SWING_SHARE = 0.3
COLORS = {
    "R1 (3-band onsets)": "#2f78d4",
    "R3 (log-mel)": "#ee6b3b",
    "Onset 3-band": "#4d4d4d",
    "Transcribe": "#8a5fbf",
}
DEFAULT_SYSTEMS = ["Onset 3-band", "Transcribe", "R1 (3-band onsets)", "R3 (log-mel)"]
BUCKETS = ["Straight", "Swung", "Sparse", "Busy"]


def pattern_features(onsets: np.ndarray, steps_per_beat: int) -> pd.DataFrame:
    ## onsets is (patterns, voices, steps). Swing share = hits on triplet positions
    ## divided by hits on the sixteenth-note or triplet grid.
    n_hits = onsets.reshape(len(onsets), -1).sum(axis=1)
    in_beat = np.arange(onsets.shape[-1]) % steps_per_beat
    sixteenth = in_beat % (steps_per_beat // 4) == 0
    triplet = in_beat % (steps_per_beat // 3) == 0
    triplet_only = triplet & ~sixteenth
    on_grid = sixteenth | triplet
    per_step = onsets.sum(axis=1)
    swung = per_step[:, triplet_only].sum(axis=1)
    clean = per_step[:, on_grid].sum(axis=1)
    share = np.divide(swung, clean, out=np.zeros(len(onsets)), where=clean > 0)
    return pd.DataFrame({"n_hits": n_hits, "swing_share": share})


def label_patterns(features: pd.DataFrame) -> pd.DataFrame:
    ## swung when enough hits sit on triplet positions; busy when above the median hit count
    out = features.copy()
    out["feel"] = np.where(out["swing_share"] >= SWING_SHARE, "Swung", "Straight")
    out["density"] = np.where(out["n_hits"] > out["n_hits"].median(), "Busy", "Sparse")
    return out


def build_pattern_table(
    library: pd.DataFrame,
    grids_fn: Callable[[pd.DataFrame], tuple[list[str], np.ndarray]],
    steps_per_beat: int,
) -> pd.DataFrame:
    ## one row per rhythm in the test pool, from one source clip each
    patterns = library.drop_duplicates("pattern_id").sort_values("pattern_id")
    source_ids, onsets = grids_fn(patterns)
    if len(source_ids) != len(patterns):
        raise EvalError("grid builder returned a different number of clips")
    features = label_patterns(pattern_features(onsets, steps_per_beat))
    features.insert(0, "pattern_id", patterns["pattern_id"].to_numpy())
    return features


def score_buckets(
    tables: dict[str, pd.DataFrame],
    patterns: pd.DataFrame,
    query_patterns: pd.Series,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    info = patterns.set_index("pattern_id")
    rows = []
    for name, table in tables.items():
        frame = pd.DataFrame({"rr": table["rr"], "pattern_id": query_patterns.reindex(table.index)})
        frame = frame.join(info[["feel", "density"]], on="pattern_id")
        for column in ("feel", "density"):
            for bucket, part in frame.groupby(column):
                low, high = bootstrap_ci(
                    part["rr"].to_numpy(), part["pattern_id"].to_numpy(), n_boot, seed=seed
                )
                rows.append(
                    {
                        "system": name,
                        "bucket": bucket,
                        "n_queries": len(part),
                        "n_patterns": part["pattern_id"].nunique(),
                        "mrr": float(part["rr"].mean()),
                        "ci_low": low,
                        "ci_high": high,
                    }
                )
    return pd.DataFrame(rows)


def plot_buckets(scores: pd.DataFrame, systems: list[str], out_pdf: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    plt.rcParams.update({"font.size": 7})
    fig, ax = plt.subplots(figsize=(3.2, 2.45))
    width = 0.8 / len(systems)
    for i, name in enumerate(systems):
        part = scores[scores["system"] == name].set_index("bucket").reindex(BUCKETS)
        x = np.arange(len(BUCKETS)) + (i - (len(systems) - 1) / 2) * width
        error = [part["mrr"] - part["ci_low"], part["ci_high"] - part["mrr"]]
        ax.bar(x, part["mrr"], width, yerr=error, label=name, color=COLORS.get(name), capsize=1)
    ax.axvline(1.5, color="#bbbbbb", lw=0.6)
    ax.set_xticks(range(len(BUCKETS)))
    ax.set_xticklabels(BUCKETS)
    ax.set_ylabel("MRR")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3, axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=5.5, frameon=False, loc="upper right")
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    fig.savefig(out_pdf.with_suffix(".png"), dpi=200)
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MRR by rhythm type: straight vs swung, sparse vs busy."
    )
    parser.add_argument("--system", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--library", type=Path, default=REPO_ROOT / "manifests/library_manifest.csv")
    parser.add_argument(
        "--query-manifest", type=Path, default=REPO_ROOT / "manifests/query_manifest.csv"
    )
    parser.add_argument("--pool-kits", nargs="+", default=["salamander", "muldjordkit"])
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results/slides/pattern_types.pdf")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    try:
        from src.eval.library_grids import source_grids
        from src.music.spec import load_spec

        spec = load_spec()
        library = pd.read_csv(args.library)
        library = library[(library["split"] == "test") & library["kit_id"].isin(args.pool_kits)]
        if library.empty:
            raise EvalError("no test-split library clips on those kits")

        def grids_fn(clips: pd.DataFrame) -> tuple[list[str], np.ndarray]:
            built = source_grids(clips, spec)
            if built.mismatched:
                raise EvalError(f"{len(built.mismatched)} library grids do not match the manifest")
            order = built.rows_for(clips["source_clip_id"].tolist())
            return clips["source_clip_id"].tolist(), built.onsets[order]

        patterns = build_pattern_table(library, grids_fn, spec.same_rhythm.steps_per_beat)
        paths = parse_systems(args.system, minimum=1)
        tables = {name: load_system(path)[0] for name, path in paths.items()}
        index = next(iter(tables.values())).index
        query_patterns = pd.Series(
            groups_from_manifest(index, args.query_manifest, "pattern_id"), index=index
        )
        scores = score_buckets(tables, patterns, query_patterns, args.n_boot, args.seed)
    except (OSError, EvalError) as error:
        parser.exit(1, f"Pattern types failed: {error}\n")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    patterns.to_csv(args.out.with_name("pattern_features.csv"), index=False)
    scores.to_csv(args.out.with_name("pattern_types.csv"), index=False)
    counts = patterns.groupby(["feel"]).size().to_dict() | patterns.groupby("density").size().to_dict()
    print(f"{len(patterns)} rhythms: {counts}")
    print(
        f"swung = at least {SWING_SHARE:.0%} of on-grid hits on triplet positions; "
        f"busy = above the median of {patterns['n_hits'].median():.0f} hits"
    )
    print(scores.pivot_table(index="system", columns="bucket", values="mrr").round(3).to_string())
    drawn = plot_buckets(scores, list(tables), args.out)
    if not drawn:
        print("(matplotlib is not installed, so the figure was not drawn: pip install matplotlib)")
    print(f"Wrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
