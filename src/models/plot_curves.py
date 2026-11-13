"""Training curves for the talk: loss and validation MRR per epoch, from runs' log.csv.

    .venv/bin/python -m src.models.plot_curves results/runs/r1_practice results/runs/r3_practice
    # -> results/figures/training_curves.png

Left panel: SupCon training loss. Right panel: validation MRR, with the MRR a random
ordering would get as a dashed line, so the gain over chance is visible. Each run is
one line, labelled with its folder name; the best epoch (the checkpoint kept) is marked.
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


def plot_runs(run_dirs: list[Path], out: Path, title: str | None = None) -> Path:
    """Draw every run's log.csv into one two-panel figure and save it as a PNG."""
    fig, (ax_loss, ax_mrr) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    chance = None
    for run in run_dirs:
        log = pd.read_csv(Path(run) / "log.csv")
        label = Path(run).name
        (line,) = ax_loss.plot(log.epoch, log.train_loss, label=label, linewidth=2)
        if "val_mrr" in log:
            ax_mrr.plot(log.epoch, log.val_mrr, label=label, color=line.get_color(), linewidth=2)
            best = log.loc[log.val_mrr.idxmax()]
            ax_mrr.scatter([best.epoch], [best.val_mrr], color=line.get_color(), zorder=3)
            chance = log.val_chance_mrr.iloc[0]
    if chance is not None:
        ax_mrr.axhline(chance, color="gray", linestyle="--", linewidth=1, label="random guessing")
    ax_loss.set(xlabel="epoch", ylabel="training loss (SupCon)", title="Training loss")
    ax_mrr.set(xlabel="epoch", ylabel="validation MRR", title="Validation retrieval", ylim=(0, 1))
    for ax in (ax_loss, ax_mrr):
        ax.grid(alpha=0.3)
        ax.legend(frameon=False)
    if title:
        fig.suptitle(title)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot training curves from runs' log.csv.")
    parser.add_argument("runs", type=Path, nargs="+", help="results/runs/<name> folders")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--title", default=None)
    args = parser.parse_args(argv)
    print(f"saved {plot_runs(args.runs, args.out, args.title)}")


if __name__ == "__main__":
    main()
