"""Training curves for the talk: loss and validation MRR per epoch, from runs' log.csv.

    .venv/bin/python -m src.models.plot_curves results/runs/r1_v07 results/runs/r3_v07
    # -> results/figures/training_curves.png
    .venv/bin/python -m src.models.plot_curves results/runs/r1_v07 results/runs/r3_v07 \
        --labels R1 R3 --stacked --out results/figures/training_curves.pdf   # slide column

One panel for the SupCon training loss and one for validation MRR, with the MRR a random
ordering would get as a dashed line, so the gain over chance is visible. Each run is one
line, named at its end (and in the legend); the best epoch (the checkpoint kept) is
marked. --stacked puts the panels on top of each other, sharing the epoch axis, to fit
half a slide. The file type follows --out (.png, .pdf, .svg).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # files only, no window

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "results" / "figures" / "training_curves.png"

# Categorical series colors in a fixed order (validated for color-blind separation and
# contrast on a light background), plus ink and grid colors for text and axes.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
INK, INK_SOFT, GRID, CHANCE = "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"


def style_axes(ax) -> None:
    """Recessive axes: light grid behind the data, no top/right spines, soft ink."""
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_SOFT)
    ax.tick_params(colors=INK_SOFT, labelcolor=INK_SOFT)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


def plot_runs(
    run_dirs: list[Path],
    out: Path,
    title: str | None = None,
    labels: list[str] | None = None,
    stacked: bool = False,
) -> Path:
    """Draw every run's log.csv into one two-panel figure and save it (type from `out`)."""
    if len(run_dirs) > len(SERIES):
        raise ValueError(f"at most {len(SERIES)} runs per figure; got {len(run_dirs)}")
    labels = labels or [Path(r).name for r in run_dirs]
    if len(labels) != len(run_dirs):
        raise ValueError("give one label per run")
    with plt.rc_context({"font.size": 8 if stacked else 10}):
        if stacked:
            fig, (ax_loss, ax_mrr) = plt.subplots(
                2, 1, figsize=(3.4, 3.0), sharex=True, constrained_layout=True
            )
        else:
            fig, (ax_loss, ax_mrr) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        chance = None
        last_epoch = 1
        logs = [pd.read_csv(Path(run) / "log.csv") for run in run_dirs]
        # End-of-line labels: nudge them apart (highest final MRR up, lowest down).
        finals = [lg.val_mrr.iloc[-1] if "val_mrr" in lg else 0.0 for lg in logs]
        order = sorted(range(len(logs)), key=lambda i: finals[i])
        nudge = {i: 5 * (rank - (len(logs) - 1) / 2) for rank, i in enumerate(order)}
        for i, (log, label, color) in enumerate(zip(logs, labels, SERIES, strict=False)):
            last_epoch = max(last_epoch, int(log.epoch.max()))
            ax_loss.plot(log.epoch, log.train_loss, label=label, color=color, linewidth=1.5)
            if "val_mrr" in log:
                ax_mrr.plot(log.epoch, log.val_mrr, label=label, color=color, linewidth=1.5)
                best = log.loc[log.val_mrr.idxmax()]
                ax_mrr.scatter(
                    [best.epoch], [best.val_mrr], s=28, color=color, edgecolor="white", zorder=3
                )
                end = log.iloc[-1]
                ax_mrr.annotate(
                    label,
                    (end.epoch, end.val_mrr),
                    xytext=(4, nudge[i]),
                    textcoords="offset points",
                    va="center",
                    color=INK,
                )
                chance = log.val_chance_mrr.iloc[0]
        if chance is not None:
            ax_mrr.axhline(chance, color=CHANCE, linestyle="--", linewidth=1)
            ax_mrr.annotate(
                "random guessing",
                (1, chance),
                xytext=(0, 3),
                textcoords="offset points",
                va="bottom",
                color=INK_SOFT,
            )
        ax_loss.set(ylabel="training loss", title="Training loss (SupCon)")
        ax_mrr.set(xlabel="epoch", ylabel="validation MRR", title="Validation retrieval")
        ax_mrr.set_ylim(0, max(0.5, ax_mrr.get_ylim()[1]))
        ax_mrr.set_xlim(right=last_epoch * 1.12)  # room for the line labels
        if not stacked:
            ax_loss.set(xlabel="epoch")
        for ax in (ax_loss, ax_mrr):
            style_axes(ax)
        if len(run_dirs) > 1:
            ax_loss.legend(frameon=False, labelcolor=INK)
        if title:
            fig.suptitle(title, color=INK)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200)
        plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot training curves from runs' log.csv.")
    parser.add_argument("runs", type=Path, nargs="+", help="results/runs/<name> folders")
    parser.add_argument("--labels", nargs="+", help="one line name per run (default: folder)")
    parser.add_argument("--stacked", action="store_true", help="panels on top of each other")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--title", default=None)
    args = parser.parse_args(argv)
    print(f"saved {plot_runs(args.runs, args.out, args.title, args.labels, args.stacked)}")


if __name__ == "__main__":
    main()
