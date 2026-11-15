"""What the model hears, for the talk: one query and its matching loop, as R1 and as R3.

    .venv/bin/python -m src.models.plot_inputs \
        --query v07_978072d11c422dfb_P10_Personal_seed0 --kit salamander \
        --out results/figures/inputs.pdf

Two columns (the query, and its matching loop on --kit) and two rows: R1's three onset
bands (low, mid, high: roughly kick, snare, hats), each scaled to its own peak, and
R3's 64-band log-mel spectrogram. Beat lines come from the manifest's tempo and the
spec's lead-in. The point: a mouth and a drum kit sound nothing alike (R3), but where
the hits fall lines up (R1).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # files only, no window

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.models.config import ModelConfig, load_model_config  # noqa: E402
from src.models.data import REPO_ROOT, load_clips  # noqa: E402
from src.models.features import compute_features, load_audio  # noqa: E402
from src.models.plot_curves import GRID, INK, INK_SOFT, SERIES  # noqa: E402
from src.music.spec import load_spec  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "results" / "figures" / "inputs.pdf"
BANDS = ("low (kick)", "mid (snare)", "high (hats)")


def beat_times(tempo_bpm: float, lead_in_s: float, beats: int) -> np.ndarray:
    return lead_in_s + np.arange(beats + 1) * 60.0 / tempo_bpm


def plot_inputs(
    clips: list[tuple[str, str]],
    cfg: ModelConfig,
    out: Path,
    tempo_bpm: float,
    lead_in_s: float,
    beats: int = 8,
) -> Path:
    """`clips` is [(title, audio path), ...], one column each; saves a 2-row figure."""
    rate = cfg.audio.frame_rate
    lines = beat_times(tempo_bpm, lead_in_s, beats)
    with plt.rc_context({"font.size": 8}):
        fig, axes = plt.subplots(
            2,
            len(clips),
            figsize=(3.2 * len(clips), 3.1),
            sharex=True,
            constrained_layout=True,
            squeeze=False,
        )
        for col, (title, path) in enumerate(clips):
            audio = load_audio(path, cfg.audio.sample_rate)
            r1 = compute_features("r1", audio, cfg)
            r3 = compute_features("r3", audio, cfg)
            t = np.arange(len(r1)) / rate
            ax1, ax3 = axes[0, col], axes[1, col]
            for band, color in enumerate(SERIES[: len(BANDS)]):
                env = r1[:, band] / max(float(r1[:, band].max()), 1e-9)
                ax1.fill_between(t, band, band + 0.85 * env, color=color, linewidth=0)
            ax1.set_yticks([b + 0.4 for b in range(3)], BANDS if col == 0 else [])
            ax1.set_ylim(-0.05, 3.0)
            ax3.imshow(
                r3.T,
                origin="lower",
                aspect="auto",
                cmap="Blues",
                extent=(0, len(r3) / rate, 0, r3.shape[1]),
            )
            ax3.set_yticks([0, 32, 64] if col == 0 else [])
            ax1.set_title(title, color=INK)
            for x in lines:  # beats; on R1 only, where they read as a grid rather than gaps
                ax1.axvline(x, color=GRID, linewidth=0.6, zorder=0)
            for ax in (ax1, ax3):
                for side in ("top", "right"):
                    ax.spines[side].set_visible(False)
                ax.tick_params(colors=INK_SOFT, labelcolor=INK_SOFT)
            ax3.set_xlabel("time (s)", color=INK)
        axes[0, 0].set_ylabel("R1: onset bands", color=INK)
        axes[1, 0].set_ylabel("R3: log-mel band", color=INK)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200)
        plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="R1 and R3 inputs for a query and its loop.")
    parser.add_argument("--query", required=True, help="query_id from the query manifest")
    parser.add_argument("--kit", required=True, help="kit of the matching loop to show")
    parser.add_argument("--kit-label", help="kit name for the title (default: --kit)")
    parser.add_argument(
        "--library", type=Path, default=REPO_ROOT / "manifests/library_manifest.csv"
    )
    parser.add_argument("--queries", type=Path, default=REPO_ROOT / "manifests/query_manifest.csv")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    clips = load_clips(args.library, args.queries)
    query = clips[clips.id == args.query]
    if query.empty:
        raise SystemExit(f"no query {args.query!r} in {args.queries}")
    query = query.iloc[0]
    loops = clips[
        (clips.kind == "library")
        & (clips.pattern_id == query.pattern_id)
        & (clips.group == args.kit)
    ]
    if loops.empty:
        raise SystemExit(f"no loop of pattern {query.pattern_id} on kit {args.kit!r}")
    tempo = float(pd.read_csv(args.queries, dtype=str).set_index("query_id").tempo_bpm[args.query])
    out = plot_inputs(
        [("Fake beatbox (query)", query.audio_path),
         (f"Matching loop ({args.kit_label or args.kit})", loops.iloc[0].audio_path)],
        load_model_config(),
        args.out,
        tempo_bpm=tempo,
        lead_in_s=load_spec().loops.audio.lead_in_s,
    )  # fmt: skip
    print(f"saved {out}")


if __name__ == "__main__":
    main()
