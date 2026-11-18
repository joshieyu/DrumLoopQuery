"""Retrieval quality by kind of rhythm: the pattern-type chart for the talk (spec §8).

    .venv/bin/python -m tools.pattern_types \\
        --system "Onset 3-band=results/eval/<onset folder>" \\
        --system "Transcribe=results/eval/<transcribe folder>" \\
        --system "R1 (3-band onsets)=results/eval/<R1 folder>" \\
        --system "R3 (log-mel)=results/eval/<R3 folder>"
    open results/figures/pattern_types/a_pattern_types.png

Each --system is NAME=PATH, where PATH is a system's results folder from
`src.eval.evaluate` (or its per_query.csv). Give them in the order above and the colors
match Dylan's other figures (Onset gray, Transcribe purple, R1 blue, R3 orange).

Every GMD test loop gets a feel (straight, swung, or none) and a density (sparse, medium
or busy) from `pattern_types` in config/music_spec.yaml, the rules the Round 2 beats
were picked with. A query takes the types of the loop it was made from (`source_clip_id`
in the query manifest). For each system and type the chart shows MRR with a 95% interval
that resamples whole people (`participant_id`), as evaluate.py does.

Writes to --out (default results/figures/pattern_types/):

- a_pattern_types.pdf, sized for the chart's slot on Dylan's results slide, and a .png
- pattern_types.csv: one row per system and type (queries, loops, performances, people,
  MRR and its interval)
- loop_types.csv: every test loop's off-beat counts, sounds, feel and density
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.compare import align, load_system
from src.eval.evaluate import EvalError, bootstrap_ci
from src.music.spec import MusicSpec, load_spec
from tools.listen.beat_picker import CATALOG_DIR, GMD_DIR, load_catalog, loop_stats

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "results" / "figures" / "pattern_types"
QUERY_MANIFEST = REPO / "manifests" / "query_manifest.csv"
FIGURE = "a_pattern_types"

# (column, value, label), left to right on the chart
TYPES = [
    ("feel", "straight", "Straight"),
    ("feel", "swung", "Swung"),
    ("density", "sparse", "Sparse"),
    ("density", "medium", "Medium"),
    ("density", "busy", "Busy"),
]
# Dylan's series colors (presentation/figures/messier_queries.pdf), in his legend order;
# a marker per system too, so the chart reads without color.
COLORS = ["#4d4d4d", "#8a5fbf", "#2f78d4", "#ee6b3b"]
MARKERS = ["o", "s", "D", "^"]
SLOT_CM = (5.97, 4.6)  # 0.41 of the 16:9 deck's text width, by the slot's 4.6 cm height


def loop_types(records: list[dict], spec: MusicSpec) -> pd.DataFrame:
    """One row per test loop: its off-beat counts, sounds, feel and density."""
    rows = []
    for record in records:
        stats = loop_stats(record, spec)
        rows.append(
            {
                "clip_id": record["clip_id"],
                "pattern_id": record["pattern_id"],
                "source_performance": record["source_performance"],
                "tempo_bpm": record["tempo_bpm"],
                "sounds": stats["sounds"],
                "straight_off": stats["straight_off"],
                "triplet_off": stats["triplet_off"],
                "other_off": stats["other_off"],
                "feel": spec.pattern_types.feel(
                    stats["straight_off"], stats["triplet_off"], stats["other_off"]
                ),
                "density": spec.pattern_types.density(stats["sounds"]),
            }
        )
    return pd.DataFrame(rows)


def query_types(query_ids: pd.Index, manifest: pd.DataFrame, loops: pd.DataFrame) -> pd.DataFrame:
    """The scored queries with their person and their source loop's types."""
    manifest = manifest.drop_duplicates("query_id").set_index("query_id")
    unknown = query_ids.difference(manifest.index)
    if len(unknown):
        raise EvalError(
            f"{len(unknown)} scored queries are not in the query manifest, e.g. {unknown[0]}"
        )
    q = manifest.loc[query_ids, ["source_clip_id", "participant_id"]]
    q = q.join(loops.set_index("clip_id"), on="source_clip_id")
    orphans = q.index[q["density"].isna()]
    if len(orphans):
        raise EvalError(
            f"{len(orphans)} queries come from loops outside the GMD test split, "
            f"e.g. {orphans[0]} ({q.loc[orphans[0], 'source_clip_id']})"
        )
    return q


def summarize(
    systems: dict[str, pd.DataFrame], queries: pd.DataFrame, n_boot: int = 10_000, seed: int = 0
) -> pd.DataFrame:
    """MRR per system and pattern type, with a 95% interval over people."""
    rows = []
    for name, table in systems.items():
        rr = table["rr"].reindex(queries.index)
        for column, value, label in TYPES:
            mask = (queries[column] == value).to_numpy()
            if not mask.any():
                continue
            part = queries[mask]
            # a query with no person counts as its own group, as in evaluate.py
            people = part["participant_id"].fillna(pd.Series(part.index, index=part.index))
            low, high = bootstrap_ci(rr[mask], people, n_boot=n_boot, seed=seed)
            rows.append(
                {
                    "system": name,
                    "kind": column,
                    "type": value,
                    "label": label,
                    "queries": int(mask.sum()),
                    "loops": part["source_clip_id"].nunique(),
                    "performances": part["source_performance"].nunique(),
                    "people": part["participant_id"].nunique(),
                    "mrr": float(rr[mask].mean()),
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return pd.DataFrame(rows)


def type_note(row: pd.Series) -> str:
    """The muted line under a type's name: how many loops, and for swung, how few grooves."""
    if row["kind"] == "feel" and row["type"] == "swung":
        return f"{row['loops']} loops\n{row['performances']} grooves"
    return f"{row['loops']} loops"


def plot(summary: pd.DataFrame, pdf: Path, png: Path) -> None:
    """Dots with 95% whiskers: one column per type, one dot per system."""
    from matplotlib.lines import Line2D

    from tools.inventory.plots import GRID, INK, INK_2, MUTED, plt

    names = list(dict.fromkeys(summary["system"]))
    present = [t for t in TYPES if ((summary["kind"] == t[0]) & (summary["type"] == t[1])).any()]
    # a wider gap between the feel and the density columns
    xs, x = [], 0.0
    for i, (kind, _, _) in enumerate(present):
        if i and kind != present[i - 1][0]:
            x += 0.45
        xs.append(x)
        x += 1.0
    width = 0.62  # of a column, shared by the systems' dots
    offsets = np.linspace(-width / 2, width / 2, len(names)) if len(names) > 1 else [0.0]

    rc = {
        "font.size": 6,
        "axes.labelsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 5.5,
        "legend.fontsize": 5.5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(SLOT_CM[0] / 2.54, SLOT_CM[1] / 2.54))
        for k, name in enumerate(names):
            part = summary[summary["system"] == name].set_index(["kind", "type"])
            px, py, lo, hi = [], [], [], []
            for (kind, value, _), cx in zip(present, xs, strict=True):
                if (kind, value) not in part.index:
                    continue
                r = part.loc[(kind, value)]
                px.append(cx + offsets[k])
                py.append(r["mrr"])
                lo.append(r["mrr"] - r["ci_low"])
                hi.append(r["ci_high"] - r["mrr"])
            color = COLORS[k % len(COLORS)]
            ax.errorbar(
                px,
                py,
                yerr=[lo, hi],
                fmt=MARKERS[k % len(MARKERS)],
                color=color,
                ecolor=color,
                elinewidth=0.8,
                capsize=0,
                markersize=3.2,
                markeredgecolor="white",
                markeredgewidth=0.4,
                label=name,
                zorder=3,
            )
        first = summary.drop_duplicates(["kind", "type"]).set_index(["kind", "type"], drop=False)
        ax.set_xticks(xs)
        ax.set_xticklabels([label for _, _, label in present], color=INK)
        ax.tick_params(axis="x", length=0, pad=2)
        for (kind, value, _), cx in zip(present, xs, strict=True):
            ax.annotate(
                type_note(first.loc[(kind, value)]),
                xy=(cx, 0),
                xycoords=("data", "axes fraction"),
                xytext=(0, -10),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=5,
                color=MUTED,
                linespacing=1.0,
            )
        groups = {}
        for (kind, _, _), cx in zip(present, xs, strict=True):
            groups.setdefault(kind, []).append(cx)
        for kind, centers in groups.items():
            ax.text(
                np.mean(centers),
                1.0,
                "Feel" if kind == "feel" else "Density",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=6,
                fontweight="bold",
                color=INK_2,
            )
        if len(groups) == 2:
            split = (xs[len(groups["feel"]) - 1] + xs[len(groups["feel"])]) / 2
            ax.axvline(split, color=GRID, linewidth=0.8, zorder=1)
        top = float(summary["ci_high"].max())
        step = 0.1 if top > 0.25 else 0.05
        ax.set_ylim(0, step * np.ceil(top * 1.05 / step))
        ax.set_yticks(np.arange(0, ax.get_ylim()[1] + 1e-9, step))
        ax.set_xlim(xs[0] - 0.6, xs[-1] + 0.6)
        ax.set_ylabel("MRR", color=INK_2, labelpad=2)
        ax.grid(axis="x", visible=False)
        ax.grid(axis="y", linewidth=0.5)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0, pad=1.5)
        handles = [
            Line2D(
                [],
                [],
                linestyle="none",
                marker=MARKERS[k % len(MARKERS)],
                markersize=3.2,
                color=COLORS[k % len(COLORS)],
                markeredgecolor="white",
                markeredgewidth=0.4,
            )
            for k in range(len(names))
        ]
        ax.legend(
            handles,
            names,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.07),
            ncol=2,
            handletextpad=0.2,
            columnspacing=0.8,
            borderaxespad=0,
            labelcolor=INK_2,
        )
        fig.subplots_adjust(left=0.13, right=0.99, top=0.77, bottom=0.2)
        pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(pdf)
        fig.savefig(png, dpi=400)
        plt.close(fig)


def parse_system(text: str) -> tuple[str, Path]:
    name, sep, path = text.partition("=")
    if not sep or not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError(f"expected NAME=PATH, got {text!r}")
    return name.strip(), Path(path.strip())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument(
        "--system",
        type=parse_system,
        action="append",
        required=True,
        help="NAME=PATH to a system's results folder or per_query.csv; repeat per system",
    )
    ap.add_argument("--query-manifest", type=Path, default=QUERY_MANIFEST)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--catalog", type=Path, default=CATALOG_DIR)
    ap.add_argument("--gmd", type=Path, default=GMD_DIR)
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    names = [name for name, _ in args.system]
    if len(set(names)) != len(names):
        ap.error("each --system needs its own NAME")
    spec = load_spec()
    try:
        systems = align({name: load_system(path)[0] for name, path in args.system})
        loops = loop_types(load_catalog(args.catalog, spec, args.gmd), spec)
        manifest = pd.read_csv(args.query_manifest, dtype=str)
        queries = query_types(systems[names[0]].index, manifest, loops)
        summary = summarize(systems, queries, n_boot=args.n_boot, seed=args.seed)
    except (OSError, EvalError) as exc:
        print(f"Can't make the chart: {exc}")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    loops.to_csv(args.out / "loop_types.csv", index=False)
    summary.to_csv(args.out / "pattern_types.csv", index=False)
    plot(summary, args.out / f"{FIGURE}.pdf", args.out / f"{FIGURE}.png")

    kinds = ""
    if "source" in manifest.columns:
        sources = manifest.drop_duplicates("query_id").set_index("query_id").loc[queries.index]
        kinds = " (" + ", ".join(f"{n} {s}" for s, n in sources["source"].value_counts().items())
        kinds += ")"
    print(
        f"{len(queries)} queries{kinds} from {queries['source_clip_id'].nunique()} loops, "
        f"{queries['participant_id'].nunique()} people"
    )
    shown = summary.pivot(index="label", columns="system", values="mrr")
    print(shown.reindex([t[2] for t in TYPES]).dropna(how="all")[names].round(3).to_string())
    print(f"Wrote {FIGURE}.pdf, {FIGURE}.png, pattern_types.csv and loop_types.csv to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
