"""Front-end comparison for the talk (experiment E2): test MRR per front-end and test set.

    .venv/bin/python -m src.models.plot_frontends \
        --system r1_v07=R1 --system r3_v07=R3 \
        --testset "synth_beatbox=Fake beatbox" --testset "team_beatbox=Real beatbox (us)" \
        --out results/figures/frontends.pdf

Reads the summary.json that src.models.embed writes for each system and test set
(results/eval/<system>_<testset>/), and draws grouped bars: one group per test set, one
bar per front-end, with its 95% bootstrap interval as a whisker and the random-guessing
MRR as a dashed tick. A test set with no results yet (say, real queries not recorded)
is left out, with a note. Test sets are never pooled.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # files only, no window

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.models.plot_curves import CHANCE, INK, SERIES, style_axes  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = REPO_ROOT / "results" / "eval"
DEFAULT_OUT = REPO_ROOT / "results" / "figures" / "frontends.pdf"


def load_mrr(eval_dir: Path, system: str, testset: str) -> dict | None:
    """{mrr, low, high, chance, n} from one embed summary, or None if it doesn't exist."""
    path = eval_dir / f"{system}_{testset}" / "summary.json"
    if not path.exists():
        return None
    s = json.loads(path.read_text())
    m = s["metrics"]
    return {
        "mrr": m["mrr"]["mean"],
        "low": m["mrr"]["ci_low"],
        "high": m["mrr"]["ci_high"],
        "chance": m["chance_mrr"]["mean"],
        "n": s["n_queries_scored"],
    }


def plot_frontends(
    systems: list[tuple[str, str]],
    testsets: list[tuple[str, str]],
    out: Path,
    eval_dir: Path = EVAL_DIR,
    log=print,
) -> Path:
    """Grouped bars of test MRR with 95% intervals; `systems`/`testsets` are (id, label)."""
    if len(systems) > len(SERIES):
        raise ValueError(f"at most {len(SERIES)} systems per figure")
    found = {}
    for t, _ in testsets:
        for s, _ in systems:
            if (r := load_mrr(eval_dir, s, t)) is not None:
                found[s, t] = r
    shown = [(t, label) for t, label in testsets if any((s, t) in found for s, _ in systems)]
    for t in sorted({t for t, _ in testsets} - {t for t, _ in shown}):
        log(f"note: no results yet for test set {t!r}; left out")
    if not shown:
        raise ValueError(f"no results found under {eval_dir} for these systems and test sets")

    width = 0.8 / len(systems)
    with plt.rc_context({"font.size": 8}):
        fig, ax = plt.subplots(figsize=(1.3 + 1.3 * len(shown), 2.6), constrained_layout=True)
        for i, ((s, label), color) in enumerate(zip(systems, SERIES, strict=False)):
            xs, ys, lo, hi = [], [], [], []
            for g, (t, _) in enumerate(shown):
                if (s, t) in found:
                    r = found[s, t]
                    xs.append(g + (i - (len(systems) - 1) / 2) * width)
                    ys.append(r["mrr"])
                    lo.append(r["mrr"] - r["low"])
                    hi.append(r["high"] - r["mrr"])
            ax.bar(xs, ys, width * 0.9, color=color, label=label, edgecolor="white", linewidth=1)
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=INK, elinewidth=1, capsize=3)
            for x, y, h in zip(xs, ys, hi, strict=True):
                ax.annotate(
                    f"{y:.2f}", (x, y + h), xytext=(0, 2), textcoords="offset points",
                    ha="center", va="bottom", color=INK,
                )  # fmt: skip
        for g, (t, _) in enumerate(shown):
            chance = next(found[s, t]["chance"] for s, _ in systems if (s, t) in found)
            ax.hlines(
                chance, g - 0.45, g + 0.45, colors=CHANCE, linestyles="--", linewidth=1,
                label="random guessing" if g == 0 else None,
            )  # fmt: skip
        n = {t: max(found[s, t]["n"] for s, _ in systems if (s, t) in found) for t, _ in shown}
        ax.set_xticks(np.arange(len(shown)), [f"{label}\n({n[t]:,} queries)" for t, label in shown])
        ax.set(ylabel="test MRR (95% CI)", ylim=(0, 0.5))
        style_axes(ax)
        ax.grid(axis="x", visible=False)
        ax.legend(frameon=False, labelcolor=INK, loc="upper right")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200)
        plt.close(fig)
    return out


def _pair(text: str) -> tuple[str, str]:
    key, _, label = text.partition("=")
    return key, label or key


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Bar chart of test MRR per front-end.")
    parser.add_argument("--system", type=_pair, action="append", required=True, help="id=label")
    parser.add_argument("--testset", type=_pair, action="append", required=True, help="id=label")
    parser.add_argument("--eval-dir", type=Path, default=EVAL_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    print(f"saved {plot_frontends(args.system, args.testset, args.out, args.eval_dir)}")


if __name__ == "__main__":
    main()
