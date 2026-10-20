"""Build the dataset inventory report for filling in docs/music_spec.md sections 1-3.

Usage:
    python -m tools.inventory.report
    python -m tools.inventory.report --ghost-thresholds 25,35,45 --flam-windows 15,25,35

Writes results/inventory/report.html plus tables/*.csv, figures/*.png, and
examples/<question>/*.mid (GMD files you can drag into Logic to listen).
Everything is descriptive; the candidate values are only there for comparison.
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from tools.inventory import avp as avp_mod
from tools.inventory import egmd as egmd_mod
from tools.inventory import gmd as gmd_mod
from tools.inventory import plots
from tools.inventory.plots import SERIES, plt

REPO_ROOT = Path(__file__).resolve().parents[2]


def int_list(s: str) -> list[int]:
    return sorted(int(x) for x in s.split(",") if x.strip())


# --------------------------------------------------------------------------- HTML page


class Page:
    def __init__(self, out: Path):
        self.out = out
        self.parts: list[str] = []
        self.toc: list[tuple[str, str]] = []
        (out / "tables").mkdir(parents=True, exist_ok=True)

    def h2(self, anchor: str, text: str) -> None:
        self.toc.append((anchor, text))
        self.parts.append(f'<h2 id="{anchor}">{html.escape(text)}</h2>')

    def h3(self, text: str) -> None:
        self.parts.append(f"<h3>{html.escape(text)}</h3>")

    def p(self, inner_html: str) -> None:
        self.parts.append(f"<p>{inner_html}</p>")

    def note(self, inner_html: str) -> None:
        self.parts.append(f'<div class="note">{inner_html}</div>')

    def raw(self, inner_html: str) -> None:
        self.parts.append(inner_html)

    def table(
        self,
        df: pd.DataFrame,
        csv: str,
        index: bool = False,
        show: int | None = 40,
        decimals: int = 1,
    ):
        df.to_csv(self.out / "tables" / f"{csv}.csv", index=index)
        shown = df if show is None else df.head(show)
        if not isinstance(shown.columns, pd.MultiIndex):
            shown = shown.rename_axis(columns=None)
        body = shown.to_html(
            index=index,
            border=0,
            classes="t",
            na_rep="",
            float_format=lambda x: f"{x:,.{decimals}f}",
        )
        body = re.sub(r"<td>(-?[\d.,]+%?)</td>", r'<td class="n">\1</td>', body)
        more = f" (first {show} of {len(df):,} rows)" if show is not None and len(df) > show else ""
        self.parts.append(
            f'<div class="tw">{body}</div>'
            f'<p class="csv"><a href="tables/{csv}.csv">{csv}.csv</a>{more}</p>'
        )

    def fig(self, name: str, caption: str) -> None:
        self.parts.append(
            f'<figure><img src="figures/{name}" alt="{html.escape(caption)}">'
            f"<figcaption>{caption}</figcaption></figure>"
        )

    def render(self, title: str, subtitle: str) -> str:
        toc = "".join(f'<li><a href="#{a}">{html.escape(t)}</a></li>' for a, t in self.toc)
        return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{ color-scheme: light; --surface:#fcfcfb; --plane:#f9f9f7; --ink:#0b0b0b;
  --ink2:#52514e; --muted:#898781; --line:#e1e0d9; --accent:#2a78d6; }}
body {{ margin:0; background:var(--plane); color:var(--ink);
  font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }}
main {{ max-width:1100px; margin:0 auto; padding:24px 16px 80px; }}
h1 {{ font-size:26px; margin:0 0 4px; }} .sub {{ color:var(--ink2); margin:0 0 24px; }}
h2 {{ font-size:20px; margin:48px 0 8px; padding-top:12px; border-top:1px solid var(--line); }}
h3 {{ font-size:16px; margin:28px 0 6px; }}
p, li {{ max-width:75ch; }} a {{ color:var(--accent); }}
nav ol {{ columns:2; padding-left:20px; }}
.note {{ background:var(--surface); border:1px solid var(--line); border-radius:8px;
  padding:10px 14px; color:var(--ink2); max-width:75ch; }}
.tw {{ overflow-x:auto; background:var(--surface); border:1px solid var(--line);
  border-radius:8px; }}
table.t {{ border-collapse:collapse; font-size:13px; font-variant-numeric:tabular-nums; }}
table.t th, table.t td {{ padding:4px 10px; border-bottom:1px solid var(--line);
  text-align:left; white-space:nowrap; }} table.t td.n {{ text-align:right; }}
table.t th {{ color:var(--ink2); font-weight:600; background:var(--plane); }}
.csv {{ font-size:12px; color:var(--muted); margin:4px 0 16px; }}
figure {{ margin:12px 0 20px; }} figure img {{ max-width:100%; height:auto;
  border:1px solid var(--line); border-radius:8px; background:var(--surface); }}
figcaption {{ font-size:13px; color:var(--ink2); max-width:75ch; margin-top:4px; }}
ul.facts li {{ margin:2px 0; }}
audio {{ height:28px; width:150px; }}
code {{ font-size:13px; background:var(--surface); padding:1px 4px; border-radius:4px; }}
</style></head><body><main>
<h1>{html.escape(title)}</h1><p class="sub">{subtitle}</p>
<nav><ol>{toc}</ol></nav>
{"".join(self.parts)}
</main></body></html>"""


# --------------------------------------------------------------------------- figures


def position_ticks(grid: int) -> tuple[list[int], list[str]]:
    marks = [(0, "1"), (1 / 4, "e"), (1 / 3, "⅓"), (1 / 2, "&"), (2 / 3, "⅔"), (3 / 4, "a")]
    ticks, labels = [], []
    for frac, lab in marks:
        if abs(frac * grid - round(frac * grid)) < 1e-9:
            ticks.append(round(frac * grid))
            labels.append(lab)
    return ticks, labels


def fig_notes(notes: pd.DataFrame, figdir: Path) -> str:
    fig, ax = plt.subplots(figsize=(7, 6))
    labels = [gmd_mod.note_label(p) for p in notes["pitch"]]
    plots.hbar(ax, labels, notes["hits"].tolist())
    ax.set_title("Hits per MIDI note, all GMD performances")
    ax.set_xlabel("hits")
    return plots.save(fig, figdir / "gmd_hits_per_note.png")


def fig_style_heatmap(share: pd.DataFrame, figdir: Path) -> str:
    fig, ax = plt.subplots(figsize=(1.0 + 0.55 * share.shape[1], 0.32 * share.shape[0] + 1.2))
    im = ax.imshow(share.to_numpy(), cmap=plots.SEQUENTIAL, aspect="auto", vmin=0)
    ax.set_xticks(range(share.shape[1]), share.columns, rotation=45, ha="right")
    ax.set_yticks(range(share.shape[0]), share.index)
    ax.grid(False)
    for (i, j), v in np.ndenumerate(share.to_numpy()):
        if v >= 0.5:
            ax.text(
                j,
                i,
                f"{v:.0f}",
                ha="center",
                va="center",
                fontsize=6.5,
                color="white" if v > share.to_numpy().max() * 0.55 else plots.INK,
            )
    ax.set_title("Share of each style's hits on each note, beats only (%)")
    fig.colorbar(im, ax=ax, shrink=0.6, label="% of style's hits")
    return plots.save(fig, figdir / "gmd_style_heatmap.png")


def fig_velocity(hits: pd.DataFrame, thresholds: list[int], figdir: Path) -> str:
    top = hits["label"].value_counts()
    top = top[top >= 1000].index[:12]
    fig, axes = plots.small_multiples(len(top), 4)
    bins = np.arange(0, 132, 4)
    for ax, lab in zip(axes, top, strict=True):
        plots.step_hist(ax, hits.loc[hits["label"] == lab, "velocity"], bins, SERIES[0])
        for i, t in enumerate(thresholds):
            plots.ref_line(ax, t, str(t), level=i)
        ax.set_title(lab, fontsize=9)
        ax.set_xlim(0, 128)
        ax.tick_params(axis="y", labelleft=False)
    fig.suptitle(
        "Velocity per note (lines: candidate ghost-note thresholds)",
        x=0.02,
        ha="left",
        fontsize=10,
        fontweight="bold",
        y=1.0,
    )
    return plots.save(fig, figdir / "gmd_velocity_per_note.png")


def fig_velocity_by_position(
    hits: pd.DataFrame, grid: int, thresholds: list[int], figdir: Path
) -> str:
    b = hits[(hits["beat_type"] == "beat") & (hits["time_signature"] == "4-4")]
    arts = ["Snare (Head)", "HH Closed", "Ride"]
    fig, axes = plt.subplots(1, len(arts), figsize=(11, 2.8))
    bins = np.arange(0, 132, 4)
    for ax, art in zip(axes, arts, strict=True):
        sub = b[b["articulation"] == art]
        pos = gmd_mod.position_classes(sub, grid)
        for color, cls in zip(SERIES, ["beats 2 & 4", "beats 1 & 3", "off-beat"], strict=False):
            plots.step_hist(ax, sub.loc[pos == cls, "velocity"], bins, color, cls, density=True)
        for i, t in enumerate(thresholds):
            plots.ref_line(ax, t, str(t), level=i)
        ax.set_title(art)
        ax.set_xlim(0, 128)
        ax.set_xlabel("velocity")
        ax.tick_params(axis="y", labelleft=False)
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.3), ncol=3)
    return plots.save(fig, figdir / "gmd_velocity_by_position.png")


def fig_positions(hits: pd.DataFrame, grid: int, figdir: Path) -> str:
    b = hits[(hits["beat_type"] == "beat") & (hits["time_signature"] == "4-4")]
    b = gmd_mod.add_grid(b, grid)
    arts = [
        "Kick",
        "Snare (Head)",
        "Snare X-Stick",
        "HH Closed",
        "HH Open",
        "HH Pedal",
        "Ride",
        "Ride (Bell)",
    ]
    fig, axes = plots.small_multiples(len(arts), 4, panel_w=2.7, panel_h=1.9)
    ticks, labels = position_ticks(grid)
    for ax, art in zip(axes, arts, strict=True):
        pos = b.loc[b["articulation"] == art, "pos_in_beat"]
        share = 100 * pos.value_counts(normalize=True).reindex(range(grid), fill_value=0)
        ax.bar(range(grid), share.to_numpy(), width=0.7, color=SERIES[0])
        ax.set_xticks(ticks, labels)
        ax.grid(axis="x", visible=False)
        ax.set_title(f"{art} ({len(pos):,})", fontsize=9)
    fig.suptitle(
        f"Where hits land inside the beat ({grid} steps per beat, % of hits), 4/4 beats",
        x=0.02,
        ha="left",
        fontsize=10,
        fontweight="bold",
        y=1.0,
    )
    return plots.save(fig, figdir / "gmd_position_in_beat.png")


def fig_swing(sw: pd.DataFrame, figdir: Path) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10, 2.8))
    for ax, col, lo, hi, title in [
        (axes[0], "pos8", 11 / 24, 17 / 24, "Off-beat 8ths: median position in the beat"),
        (axes[1], "pos16", 5 / 12, 10 / 12, "Off-beat 16ths: median position in the half beat"),
    ]:
        plots.step_hist(ax, sw[col].dropna(), np.linspace(lo, hi, 41), SERIES[0])
        plots.ref_line(ax, 0.5, "straight")
        plots.ref_line(ax, 2 / 3, "triplet", level=1)
        ax.set_title(title)
        ax.set_xlabel("position (0.5 = straight, 0.667 = triplet swing)")
        ax.set_ylabel("performances")
    return plots.save(fig, figdir / "gmd_swing.png")


def fig_gaps(gaps: pd.DataFrame, windows: list[int], figdir: Path) -> str:
    insts = [
        i
        for i in ["Kick", "Snare", "Hi-hat", "Ride", "Tom 1", "Tom 2", "Tom 3", "Crash 1"]
        if (gaps["instrument"] == i).any()
    ]
    fig, axes = plots.small_multiples(len(insts), 4)
    bins = np.arange(0, max(windows) + 32, 2)
    for ax, inst in zip(axes, insts, strict=True):
        g = gaps.loc[gaps["instrument"] == inst, "gap_ms"]
        plots.step_hist(ax, g, bins, SERIES[0])
        for i, w in enumerate(windows):
            plots.ref_line(ax, w, str(w), level=i)
        ax.set_title(f"{inst} ({len(g):,} pairs)", fontsize=9)
        ax.set_xlabel("ms")
    fig.suptitle(
        "Gaps between consecutive hits on the same instrument (lines: candidate flam windows)",
        x=0.02,
        ha="left",
        fontsize=10,
        fontweight="bold",
        y=1.0,
    )
    return plots.save(fig, figdir / "gmd_same_instrument_gaps.png")


def fig_bpm(perf: pd.DataFrame, figdir: Path) -> str:
    fig, ax = plt.subplots(figsize=(7, 2.6))
    bins = np.arange(40, 300, 5)
    for color, bt in zip(SERIES, ["beat", "fill"], strict=False):
        plots.step_hist(ax, perf.loc[perf["beat_type"] == bt, "bpm"], bins, color, bt + "s")
    ax.set_title("Tempo of GMD performances")
    ax.set_xlabel("BPM")
    ax.set_ylabel("performances")
    ax.legend()
    return plots.save(fig, figdir / "gmd_bpm.png")


def fig_open_hats(per_hit: pd.DataFrame, grid: int, figdir: Path) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10, 2.6))
    share = 100 * per_hit["pos_in_beat"].value_counts(normalize=True).reindex(
        range(grid), fill_value=0
    )
    axes[0].bar(range(grid), share.to_numpy(), width=0.7, color=SERIES[0])
    ticks, labels = position_ticks(grid)
    axes[0].set_xticks(ticks, labels)
    axes[0].grid(axis="x", visible=False)
    axes[0].set_title("Open-hat hits: position in the beat (%)")
    gaps = per_hit["gap_steps"].dropna()
    gaps = gaps[gaps <= 2 * grid]
    share = 100 * gaps.value_counts(normalize=True).reindex(range(1, 2 * grid + 1), fill_value=0)
    axes[1].bar(share.index, share.to_numpy(), width=0.7, color=SERIES[0])
    axes[1].set_xticks([s for s in range(0, 2 * grid + 1, grid // 4 or 1)])
    axes[1].grid(axis="x", visible=False)
    axes[1].set_title(f"Steps until the next hi-hat event ({grid} per beat, %)")
    return plots.save(fig, figdir / "gmd_open_hats.png")


def fig_avp_centroid(u: pd.DataFrame, figdir: Path) -> str:
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.4), sharey=True)
    bins = np.geomspace(100, 16000, 41)
    for ax, lab in zip(axes, avp_mod.LABELS, strict=True):
        for color, mode in zip(SERIES, ["Personal", "Fixed"], strict=False):
            sel = u[(u["label"] == lab) & (u["mode"] == mode) & (u["kind"] != "Improvisation")]
            plots.step_hist(ax, sel["centroid_hz"].dropna(), bins, color, mode, density=True)
        ax.set_xscale("log")
        ax.set_title(avp_mod.LABEL_NAMES[lab])
        ax.set_xlabel("spectral centroid, first 50 ms (Hz)")
        ax.tick_params(axis="y", labelleft=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, -0.02), ncol=2)
    return plots.save(fig, figdir / "avp_centroid.png")


def fig_avp_mix(mix: pd.DataFrame, figdir: Path) -> str:
    fig, axes = plt.subplots(2, 1, figsize=(10, 4.6), sharex=True)
    for ax, mode in zip(axes, ["Personal", "Fixed"], strict=True):
        m = mix.loc[mode]
        bottom = np.zeros(len(m))
        x = np.arange(len(m))
        for color, lab in zip(SERIES, avp_mod.LABELS, strict=True):
            v = m[f"share_{lab}"].to_numpy() * 100
            ax.bar(
                x,
                v,
                bottom=bottom,
                width=0.7,
                color=color,
                edgecolor=plots.SURFACE,
                linewidth=1,
                label=avp_mod.LABEL_NAMES[lab],
            )
            bottom += v
        ax.set_xticks(x, [f"P{p}" for p in m.index], fontsize=7)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, 100)
        ax.set_title(f"{mode} improvisations: share of each sound (%)")
    axes[0].legend(ncol=4, loc="upper left", bbox_to_anchor=(0, 1.32))
    return plots.save(fig, figdir / "avp_improvisation_mix.png")


def fig_avp_floor(floor: pd.DataFrame, figdir: Path) -> str:
    fig, ax = plt.subplots(figsize=(7, 2.6))
    for color, mode in zip(SERIES, ["Personal", "Fixed"], strict=False):
        plots.step_hist(
            ax,
            floor.loc[floor["mode"] == mode, "peak_to_floor_db"],
            np.arange(10, 90, 2),
            color,
            mode,
        )
    ax.set_title("AVP files: loudest 10 ms frame vs. quietest-10% frame (dB)")
    ax.set_xlabel("dB")
    ax.set_ylabel("files")
    ax.legend()
    return plots.save(fig, figdir / "avp_peak_to_floor.png")


# --------------------------------------------------------------------------- examples


def copy_examples(
    out: Path, gmd_root: Path, perf: pd.DataFrame, picks: dict[str, tuple[str, pd.Series]]
) -> list[tuple[str, str, list[str]]]:
    """Copy chosen GMD MIDI files into examples/<slug>/. picks: slug -> (description,
    Series of id -> stat). Returns (slug, description, file names) for the report."""
    root = out / "examples"
    if root.exists():
        shutil.rmtree(root)
    p = perf.set_index("id")
    listing = []
    for slug, (desc, stat) in picks.items():
        d = root / slug
        d.mkdir(parents=True)
        names = []
        for rank, (pid, value) in enumerate(stat.items(), start=1):
            src = gmd_root / p.at[pid, "midi_filename"]
            style = p.at[pid, "style"].replace("/", "-")
            name = f"{rank:02d}_{value}_{style}_{p.at[pid, 'bpm']}bpm_{pid.replace('/', '_')}.mid"
            shutil.copyfile(src, d / name)
            names.append(name)
        listing.append((slug, desc, names))
    return listing


# --------------------------------------------------------------------------- main


def build(args: argparse.Namespace) -> Path:
    out: Path = args.out
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    page = Page(out)
    grid = args.position_grid
    thr = args.ghost_thresholds
    mid_thr = thr[(len(thr) - 1) // 2]
    windows = args.flam_windows
    mid_win = windows[(len(windows) - 1) // 2]

    print("Loading GMD MIDI ...", flush=True)
    perf, hits = gmd_mod.load_gmd(args.gmd_dir)
    perf["bars"] = perf["duration_beats"] / 4
    beats44 = perf[(perf["beat_type"] == "beat") & (perf["time_signature"] == "4-4")]
    hb = hits[hits["id"].isin(beats44["id"])]

    print("Loading E-GMD metadata ...", flush=True)
    egmd = egmd_mod.load_egmd(args.egmd_dir)

    print("Loading AVP ...", flush=True)
    avp_files, utts = avp_mod.load_avp(args.avp_dir)

    # ------------------------------------------------------------------ settings
    page.note(
        "Descriptive only: nothing here is a decision. Candidate values used for comparison "
        f"(change them with command-line flags): ghost-note thresholds <b>{thr}</b>, flam "
        f"windows <b>{windows} ms</b>, grids <b>{args.grids} steps/beat</b>, position grid "
        f"<b>{grid} steps/beat</b>. Every table is also saved as CSV under "
        "<code>tables/</code>; listening examples are in <code>examples/</code>."
    )
    sources_path = args.gmd_dir.parents[1] / "sources.json"
    if sources_path.exists():
        src = pd.DataFrame(json.loads(sources_path.read_text()).values())
        page.h2("sources", "Sources")
        page.table(src[["name", "title", "version", "license", "fetched_on", "page"]], "sources")

    # ------------------------------------------------------------------ at a glance
    print("Analysing GMD ...", flush=True)
    windows2 = gmd_mod.bar_windows(hits, perf, 2, grid)
    windows1 = gmd_mod.bar_windows(hits, perf, 1, grid)
    rule_broad = gmd_mod.windows_with(windows2, [36], [38, 40, 37])
    snare_head = hits[hits["articulation"] == "Snare (Head)"]["velocity"]
    hh_closed = hits[hits["articulation"] == "HH Closed"]["velocity"]
    missing = perf[~perf["id"].isin(egmd["id"])]
    gaps = gmd_mod.same_instrument_gaps(hits, max(windows) + 30)
    page.h2("glance", "At a glance")
    facts = [
        f"GMD: {len(perf):,} performances ({(perf['beat_type'] == 'beat').sum()} beats, "
        f"{(perf['beat_type'] == 'fill').sum()} fills); {len(hits):,} hits; "
        f"{(perf['time_signature'] == '4-4').mean():.1%} in 4/4.",
        f"4/4 beats: {len(beats44)} performances, {beats44['duration'].sum() / 3600:.1f} h, "
        f"about {len(windows2):,} non-overlapping 2-bar windows "
        f"({len(windows1):,} 1-bar windows) before any filtering or deduplication.",
        f"{rule_broad.mean():.1%} of those 2-bar windows have a kick (36) and some snare "
        "articulation (38, 40 or 37).",
        "Snare head hits below each candidate threshold: "
        + ", ".join(f"&lt;{t}: {(snare_head < t).mean():.1%}" for t in thr)
        + ". Closed hi-hat (42 + 22): "
        + ", ".join(f"&lt;{t}: {(hh_closed < t).mean():.1%}" for t in thr)
        + ".",
        f"Same-instrument hit pairs closer than {mid_win} ms: "
        f"{(gaps['gap_ms'] < mid_win).sum():,} "
        f"({1000 * (gaps['gap_ms'] < mid_win).sum() / len(hits):.1f} per 1,000 hits).",
        f"E-GMD: {egmd['kit_name'].nunique()} kits, {egmd['id'].nunique():,} of the "
        f"{len(perf):,} GMD performances. Missing: {len(missing)} "
        f"({(missing['beat_type'] == 'beat').sum()} beats, "
        f"{missing.loc[missing['beat_type'] == 'beat', 'duration'].sum() / 60:.0f} min of beats).",
        f"AVP: {avp_files['participant'].nunique()} participants, {len(avp_files)} audio files, "
        f"{len(utts):,} annotated utterances; all files "
        f"{', '.join(sorted({f'{r} Hz' for r in avp_files['samplerate']}))}, "
        f"{', '.join(sorted(set(avp_files['subtype'])))}.",
    ]
    page.raw('<ul class="facts">' + "".join(f"<li>{f}</li>" for f in facts) + "</ul>")

    # ------------------------------------------------------------------ overview
    page.h2("overview", "GMD overview")
    ov = (
        perf.groupby(["beat_type", "split"])
        .agg(
            performances=("id", "size"),
            minutes=("duration", lambda s: s.sum() / 60),
            bars=("bars", "sum"),
        )
        .reset_index()
    )
    page.table(ov, "gmd_by_type_split")
    page.table(
        perf.groupby(["time_signature", "beat_type"]).size().unstack(fill_value=0).reset_index(),
        "gmd_time_signatures",
    )
    styles = (
        perf[perf["beat_type"] == "beat"]
        .groupby(["style_primary", "split"])
        .size()
        .unstack(fill_value=0)
    )
    styles["total"] = styles.sum(axis=1)
    page.h3("Styles of beat performances, by split")
    page.table(styles.sort_values("total", ascending=False).reset_index(), "gmd_styles")
    page.fig(fig_bpm(perf, figdir), "Tempo of each performance (info.csv BPM).")
    sanity = pd.DataFrame(
        {
            "check": [
                "ticks per beat",
                "files with more than one tempo event",
                "files with more than one time-signature event",
                "files whose MIDI tempo differs from info.csv BPM by > 0.5",
                "hits on undocumented MIDI notes",
            ],
            "result": [
                ", ".join(map(str, sorted(perf["ticks_per_beat"].unique()))),
                int((perf["n_tempo_events"] > 1).sum()),
                int((perf["n_timesig_events"] > 1).sum()),
                int(((perf["midi_bpm"] - perf["bpm"]).abs() > 0.5).sum()),
                int((~hits["pitch"].isin(list(gmd_mod.GMD_NOTES))).sum()),
            ],
        }
    )
    page.h3("MIDI sanity checks")
    page.table(sanity, "gmd_midi_sanity")

    # ------------------------------------------------------------------ section 1
    page.h2("s1", "Spec §1: Drum grouping")
    notes = gmd_mod.note_table(hits, perf)
    page.fig(fig_notes(notes, figdir), "Hits on each of GMD's 22 documented notes.")
    page.table(notes, "gmd_notes")
    page.fig(
        fig_style_heatmap(gmd_mod.style_share(hits), figdir),
        "Columns sum to 100%. Styles with fewer than 2,000 beat hits are left out.",
    )
    page.table(gmd_mod.style_share(hits).round(2).reset_index(), "gmd_style_share")

    page.h3("Timekeeper per performance")
    page.p(
        "For each performance, the cymbal articulation (bow and edge combined) with the "
        "most hits. Beats only."
    )
    tk = gmd_mod.timekeepers(hits).join(perf.set_index("id")[["beat_type", "style_primary"]])
    tkb = tk[tk["beat_type"] == "beat"]
    page.table(
        pd.crosstab(
            tkb["timekeeper"], tkb["style_primary"], margins=True, margins_name="total"
        ).reset_index(),
        "gmd_timekeeper_by_style",
    )
    tk.reset_index().to_csv(out / "tables" / "gmd_timekeeper_per_performance.csv", index=False)

    page.h3("Open hi-hat use")
    oh_counts, oh = gmd_mod.open_hat_followups(hits, grid)
    page.p(
        f"{len(oh):,} open-hat hits (46 + 26). {oh['same_step_closed_or_pedal'].mean():.1%} share "
        "a grid step with a closed or pedal hat. The table shows the next hi-hat event after "
        "each open hat."
    )
    page.table(oh_counts, "gmd_open_hat_next_event")
    page.fig(
        fig_open_hats(oh, grid, figdir),
        "Left: where open hats land in the beat. Right: distance to the next hi-hat event.",
    )

    # ------------------------------------------------------------------ section 2
    page.h2("s2", "Spec §2: What counts as the same rhythm")
    page.h3("Velocities and ghost-note thresholds")
    page.fig(fig_velocity(hits, thr, figdir), "Notes with at least 1,000 hits.")
    page.table(gmd_mod.velocity_table(hits, thr), "gmd_velocity")
    page.fig(
        fig_velocity_by_position(hits, grid, thr, figdir),
        "Velocity split by where the hit lands (4/4 beats). Each curve is normalized, so "
        "compare shapes, not heights.",
    )
    page.table(
        gmd_mod.velocity_by_position(
            hits, ["Snare (Head)", "HH Closed", "Ride", "Kick"], grid, thr
        ),
        "gmd_velocity_by_position",
    )

    page.h3("Flams and near-simultaneous hits")
    page.fig(
        fig_gaps(gaps, windows, figdir),
        "Includes bow/edge and head/rim pairs on the same instrument.",
    )
    per_inst = gaps.groupby("instrument")["gap_ms"]
    n_inst = hits.groupby("instrument").size()
    flam_tab = pd.DataFrame(
        {
            f"pairs < {w} ms per 1,000 hits": 1000
            * per_inst.apply(lambda s, w=w: (s < w).sum())
            / n_inst
            for w in windows
        }
    ).fillna(0)
    page.table(flam_tab.reset_index(), "gmd_flam_rates")
    pairs = (
        gaps[gaps["gap_ms"] < max(windows)]
        .groupby(["first", "second"])
        .agg(pairs=("gap_ms", "size"), median_ms=("gap_ms", "median"))
        .sort_values("pairs", ascending=False)
        .reset_index()
    )
    page.p(f"Most common note pairs closer than {max(windows)} ms:")
    page.table(pairs, "gmd_close_pairs", show=20)

    page.h3("Grid resolution and swing")
    page.fig(
        fig_positions(hits, grid, figdir),
        "Straight 16ths land on 1/e/&amp;/a; 8th-note triplets and swung 8ths on ⅓ and ⅔; "
        "swung 16ths on ⅓ and the step after ⅔. The bar count is in each title.",
    )
    page.p("How far quantizing moves each hit, for each candidate grid:")
    page.table(gmd_mod.timing_offsets(hits, args.grids), "gmd_grid_offsets")
    page.p(
        "Percent of hits that land on a grid step already taken by the same instrument "
        "(these would merge into one hit):"
    )
    page.table(gmd_mod.grid_collisions(hits, args.grids).reset_index(), "gmd_grid_collisions")
    sw = gmd_mod.swing_estimates(hits).join(perf.set_index("id")[["beat_type", "style_primary"]])
    swb = sw[sw["beat_type"] == "beat"]
    page.fig(
        fig_swing(swb, figdir),
        "One value per beat performance (kick hits left out). A value near 0.667 at the 16th "
        "level can mean swung 16ths or 8th-note triplets.",
    )
    page.table(
        swb.groupby("style_primary")
        .agg(
            performances=("pos8", "size"),
            median_pos8=("pos8", "median"),
            median_pos16=("pos16", "median"),
        )
        .sort_values("performances", ascending=False)
        .round(3)
        .reset_index(),
        "gmd_swing_by_style",
        decimals=3,
    )
    sw.reset_index().to_csv(out / "tables" / "gmd_swing_per_performance.csv", index=False)

    # ------------------------------------------------------------------ section 3
    page.h2("s3", "Spec §3: Cutting loops")
    w2 = windows2.assign(in_egmd=windows2["id"].isin(egmd["id"]))
    w1 = windows1.assign(in_egmd=windows1["id"].isin(egmd["id"]))
    avail = pd.DataFrame(
        {
            "beat performances (4/4)": beats44.groupby("split").size(),
            "2-bar windows": w2.groupby("split").size(),
            "2-bar windows in E-GMD": w2[w2["in_egmd"]].groupby("split").size(),
            "1-bar windows": w1.groupby("split").size(),
            "1-bar windows in E-GMD": w1[w1["in_egmd"]].groupby("split").size(),
        }
    )
    avail.loc["total"] = avail.sum()
    page.p(
        "Non-overlapping windows starting at beat 0, counted over the span of each "
        "performance's hits. No filtering or deduplication yet."
    )
    page.table(avail.reset_index(names="split"), "gmd_windows_available")

    variants = {
        "kick 36 + snare 38": ([36], [38]),
        "kick 36 + snare 38/40": ([36], [38, 40]),
        "kick 36 + snare 38/40/37": ([36], [38, 40, 37]),
    }
    rule_rows = []
    for name, (k, s) in variants.items():
        rule_rows.append(
            {
                "rule": name,
                "2-bar windows kept": gmd_mod.windows_with(windows2, k, s).mean() * 100,
                "1-bar windows kept": gmd_mod.windows_with(windows1, k, s).mean() * 100,
            }
        )
    page.p(
        "Percent of windows that would pass a minimum-content rule, for different ways of "
        "counting the snare:"
    )
    page.table(pd.DataFrame(rule_rows), "gmd_min_content_rules")

    fb = gmd_mod.first_bar_content(hits, perf, grid)
    page.h3("Start of each performance")
    page.p(
        f"4/4 performances whose first hit is after beat 2: "
        f"{(fb['first_hit_beat'] > 2).sum()}; whose first bar is cymbals only: "
        f"{fb['bar1_cymbals_only'].fillna(False).astype(bool).sum()}; with no kick in bar 1: "
        f"{(~fb['bar1_has_kick'].fillna(False).astype(bool)).sum()}. Examples are in "
        "<code>examples/late_first_hit</code>."
    )
    fb.reset_index().to_csv(out / "tables" / "gmd_first_bar.csv", index=False)
    fills = perf[perf["beat_type"] == "fill"]
    page.p(
        f"Fill recordings: {len(fills)}; median length {fills['bars'].median():.1f} bars "
        f"(10th–90th percentile {fills['bars'].quantile(0.1):.1f}–"
        f"{fills['bars'].quantile(0.9):.1f}). Beat recordings: median "
        f"{beats44['bars'].median():.0f} bars."
    )

    # ------------------------------------------------------------------ E-GMD
    print("Checking E-GMD MIDI against GMD ...", flush=True)
    page.h2("egmd", "E-GMD: kits and coverage (spec §4)")
    page.table(egmd_mod.kit_table(egmd), "egmd_kits", show=None)
    page.p("GMD performances included in E-GMD:")
    page.table(egmd_mod.coverage(perf, egmd), "egmd_coverage")
    missing[["id", "beat_type", "style", "bpm", "duration", "split"]].to_csv(
        out / "tables" / "egmd_missing_performances.csv", index=False
    )
    chk = egmd_mod.midi_check(args.gmd_dir, args.egmd_dir, perf, egmd, args.egmd_sample)
    by_kit = chk.groupby("kit_name").agg(
        checked=("identical", "size"), identical_pct=("identical", lambda s: 100 * s.mean())
    )
    page.p(
        f"For {chk['id'].nunique()} random performances, every kit's E-GMD MIDI was compared "
        f"with the GMD MIDI: {chk['identical'].mean():.1%} of files are note-for-note "
        "identical. Kits where some files differ:"
    )
    page.table(
        by_kit[by_kit["identical_pct"] < 100].sort_values("identical_pct").reset_index(),
        "egmd_midi_differences_by_kit",
    )
    always = egmd_mod.always_different(chk)
    page.p(
        f"{len(always)} of the checked performances differ on every kit (E-GMD has a few notes "
        "fewer). The other differences are per-kit note remaps: the kit plays some GMD notes "
        "on a different MIDI note. Most common remaps, leaving those performances out "
        "(General MIDI names: 39 hand clap, 54 tambourine, 56 cowbell):"
    )
    page.table(egmd_mod.kit_remaps(chk), "egmd_kit_remaps", show=None)
    chk.to_csv(out / "tables" / "egmd_midi_check.csv", index=False)

    # ------------------------------------------------------------------ AVP
    print("Measuring AVP utterances ...", flush=True)
    page.h2("avp", "AVP: vocal percussion sounds (spec §5–6)")
    fmt = avp_files.groupby(["samplerate", "channels", "subtype"]).size().reset_index(name="files")
    page.table(fmt, "avp_formats")
    bad = utts[~utts["label"].isin(avp_mod.LABELS) | (utts["label"] != utts["label_raw"])]
    page.p(f"Annotation issues (labels with stray spaces or blank): {len(bad)}.")
    if len(bad):
        page.table(
            bad[["participant", "mode", "kind", "onset_s", "label_raw"]].assign(
                label_raw=lambda d: d["label_raw"].map(repr)
            ),
            "avp_label_issues",
        )
    valid = utts[utts["label"].isin(avp_mod.LABELS)]
    counts = (
        valid.groupby(["participant", "mode", "label"])
        .size()
        .unstack(["mode", "label"], fill_value=0)
    )
    counts.columns = [f"{m} {lab}" for m, lab in counts.columns]
    page.h3("Utterances per participant")
    page.table(counts.reset_index(), "avp_counts", show=None)

    u = avp_mod.measure_utterances(avp_files, utts[utts["label"].isin(avp_mod.LABELS)])
    stats = (
        u[u["kind"] != "Improvisation"]
        .groupby(["mode", "label"])
        .agg(
            utterances=("label", "size"),
            peak_dbfs_median=("peak_dbfs", "median"),
            decay_ms_median=("decay_ms", "median"),
            centroid_hz_median=("centroid_hz", "median"),
            centroid_hz_p10=("centroid_hz", lambda s: s.quantile(0.1)),
            centroid_hz_p90=("centroid_hz", lambda s: s.quantile(0.9)),
        )
        .reset_index()
    )
    page.h3("Sound character per class (isolated-sound files)")
    page.table(stats, "avp_sound_stats")
    page.fig(
        fig_avp_centroid(u, figdir),
        "Brightness of each class: kicks sit low, hats high, and the overlap shows how "
        "separable the classes are by spectrum alone.",
    )
    u.drop(columns=["label_raw"]).to_csv(out / "tables" / "avp_utterances.csv", index=False)
    gaps_avp = avp_mod.repetition_gaps(utts)
    page.p("Gap between consecutive utterances inside a file (ms):")
    page.table(
        gaps_avp.groupby(["mode", "kind"])["gap_ms"]
        .describe(percentiles=[0.1, 0.5, 0.9])
        .round(0)
        .reset_index(),
        "avp_onset_gaps",
    )

    page.h3("Improvisations: which sounds people use")
    mix = avp_mod.improvisation_mix(utts[utts["label"].isin(avp_mod.LABELS)])
    page.fig(fig_avp_mix(mix, figdir), "Each bar is one participant's improvisation.")
    page.table(mix.reset_index().round(3), "avp_improvisation_mix", show=None, decimals=2)

    page.h3("Recording noise")
    floor = avp_mod.file_noise_floor(avp_files)
    noise = args.avp_dir / "noise_sample.wav"
    page.fig(
        fig_avp_floor(floor, figdir),
        "A rough SNR proxy per file: the loudest 10 ms frame against the 10th-percentile "
        "frame (usually the room between utterances). Frames of exact digital silence are "
        "left out of the floor estimate.",
    )
    page.p(
        f"Files containing exact digital silence: median {floor['zero_sample_pct'].median():.1f}% "
        f"of samples are exactly zero; {(floor['zero_sample_pct'] > 5).sum()} files have more "
        "than 5%. This suggests some editing or gating, so the gaps between utterances are not "
        "always real room tone."
    )
    page.table(
        floor.groupby("mode")[["floor_dbfs", "peak_to_floor_db", "zero_sample_pct"]]
        .describe(percentiles=[0.1, 0.5, 0.9])
        .round(1)
        .reset_index(),
        "avp_noise_floor",
    )
    if noise.exists():
        ns = avp_mod.noise_sample_stats(noise)
        rel = os.path.relpath(noise, out)
        page.p(
            f"<code>noise_sample.wav</code>: {ns['duration_s']:.1f} s, {ns['channels']} ch, "
            f"RMS {ns['rms_dbfs']:.1f} dBFS, peak {ns['peak_dbfs']:.1f} dBFS "
            f'<audio controls preload="none" src="{html.escape(rel)}"></audio>'
        )

    page.h3("Phonemes (Personal subset only)")
    pers = utts[(utts["mode"] == "Personal") & utts["label"].isin(avp_mod.LABELS)]
    for col in ["onset_phoneme", "coda_phoneme"]:
        top = pers[col].value_counts().index[:15]
        tab = pd.crosstab(pers[col].where(pers[col].isin(top), "(other)"), pers["label"])
        tab = tab.reindex(columns=avp_mod.LABELS, fill_value=0)
        tab = tab.loc[tab.sum(axis=1).sort_values(ascending=False).index]
        page.table(tab.reset_index(), f"avp_{col}", show=None)

    page.h3("Audition")
    page.p(
        "AVP's own recordings, nothing rendered. Each isolated-sound file holds about 30 "
        "repetitions of one sound."
    )
    for mode in ["Personal", "Fixed"]:
        rows = []
        for pid, g in avp_files[avp_files["mode"] == mode].groupby("participant"):
            cells = []
            for kind in avp_mod.KINDS:
                f = g[g["kind"] == kind]
                if f.empty:
                    cells.append("<td></td>")
                    continue
                rel = os.path.relpath(f["path"].iat[0], out)
                cells.append(
                    f'<td><audio controls preload="none" src="{html.escape(rel)}"></audio></td>'
                )
            rows.append(f"<tr><td>P{pid}</td>{''.join(cells)}</tr>")
        head = "".join(f"<th>{k}</th>" for k in ["", *avp_mod.KINDS])
        page.raw(
            f"<h4>{mode}</h4><div class='tw'><table class='t'><tr>{head}</tr>"
            f"{''.join(rows)}</table></div>"
        )

    # ------------------------------------------------------------------ examples
    print("Copying listening examples ...", flush=True)
    perf_i = perf.set_index("id")
    b_ids = set(beats44["id"])
    n = args.examples

    def top(series: pd.Series, k: int = n, ascending: bool = False, fmt: str = "{:.2f}"):
        s = series[series.index.isin(b_ids)].dropna().sort_values(ascending=ascending).head(k)
        return s.map(lambda v: fmt.format(v).replace(".", "p"))

    def share_below(art: str, t: int, min_hits: int) -> pd.Series:
        h = hb[hb["articulation"] == art]
        g = h.groupby("id")["velocity"]
        return (g.apply(lambda v: (v < t).mean())[g.size() >= min_hits]) * 100

    art_counts = hb.groupby(["id", "articulation"]).size().unstack(fill_value=0)
    bars = perf_i["bars"]
    window_fail = (~rule_broad).groupby(windows2["id"]).mean() * 100
    swing_mid = (swb["pos8"] - (0.5 + 2 / 3) / 2).abs()

    def fmt_pos(values: pd.Series, ids: pd.Index) -> pd.Series:
        return values.loc[ids].map(lambda v: f"{v:.3f}".replace(".", "p"))

    straight = (swb["pos8"] - 0.5).abs() + (swb["pos16"] - 0.5).abs()
    picks = {
        "snare_ghosts": (
            f"Most snare-head hits below velocity {mid_thr} (value: % of snare-head hits)",
            top(share_below("Snare (Head)", mid_thr, 50), fmt="{:.0f}pct"),
        ),
        "quiet_closed_hats": (
            f"Most closed-hat hits below velocity {mid_thr}",
            top(share_below("HH Closed", mid_thr, 100), fmt="{:.0f}pct"),
        ),
        "swung_8ths": (
            "Off-beat 8ths furthest toward triplet (value: median position)",
            top(swb["pos8"], fmt="{:.3f}"),
        ),
        "between_straight_and_swung": (
            "Off-beat 8ths halfway between straight and triplet (value: median position)",
            fmt_pos(swb["pos8"], top(swing_mid, ascending=True).index),
        ),
        "straight_reference": (
            "Off-beat 8ths and 16ths closest to straight (value: 8th position)",
            fmt_pos(swb["pos8"], top(straight, ascending=True).index),
        ),
        "swung_16ths": (
            "Off-beat 16ths furthest toward triplet (could also be triplets)",
            top(swb["pos16"], fmt="{:.3f}"),
        ),
        "ride_led": (
            "Ride is the timekeeper (value: ride share of cymbal hits)",
            top(tkb.loc[tkb["timekeeper"] == "Ride", "timekeeper_share"]),
        ),
        "pedal_hat_led": (
            "Pedal hat is the timekeeper",
            top(tkb.loc[tkb["timekeeper"] == "HH Pedal", "timekeeper_share"]),
        ),
        "open_hat_heavy": (
            "Most open-hat hits per bar",
            top(art_counts.get("HH Open", pd.Series(dtype=float)) / bars),
        ),
        "cross_stick": (
            "Highest share of snare hits played cross-stick",
            top(
                art_counts.get("Snare X-Stick", 0)
                / art_counts[["Snare (Head)", "Snare (Rim)", "Snare X-Stick"]].sum(axis=1)
            ),
        ),
        "rimshot_heavy": (
            "Highest share of snare hits played as rimshots",
            top(
                art_counts.get("Snare (Rim)", 0)
                / art_counts[["Snare (Head)", "Snare (Rim)", "Snare X-Stick"]].sum(axis=1)
            ),
        ),
        "flams": (
            f"Most same-instrument pairs closer than {mid_win} ms per bar",
            top(gaps[gaps["gap_ms"] < mid_win].groupby("id").size() / bars),
        ),
        "missing_kick_or_snare": (
            "Most 2-bar windows without a kick or any snare (value: %)",
            top(window_fail, fmt="{:.0f}pct"),
        ),
    }
    late = fb[fb["first_hit_beat"] > 2]["first_hit_beat"].sort_values(ascending=False).head(n)
    picks["late_first_hit"] = (
        "First hit after beat 2 (value: beat of first hit), any type",
        late.map(lambda v: f"{v:.1f}".replace(".", "p")),
    )
    odd = perf_i[perf_i["time_signature"] != "4-4"]["time_signature"]
    picks["odd_meter"] = ("Every performance not in 4/4", odd)
    fill_ids = perf_i[perf_i["beat_type"] == "fill"].sample(n, random_state=0).index
    picks["fills_sample"] = (
        "Random fill recordings, to hear what GMD labels a fill",
        pd.Series("fill", index=fill_ids),
    )
    listing = copy_examples(out, args.gmd_dir, perf, picks)
    pd.concat(
        [
            s.rename("value").rename_axis("id").reset_index().assign(question=k)
            for k, (_, s) in picks.items()
        ]
    ).to_csv(out / "tables" / "examples.csv", index=False)

    page.h2("examples", "Listening examples (GMD MIDI)")
    page.note(
        "Drag a folder from <code>results/inventory/examples/</code> into Logic. Notes 22 and "
        "26 (hi-hat edge) and 58 (Tom 3 rim, General MIDI vibraslap) are not standard GM drum "
        "notes, so Drum Kit Designer may play them as something else or not at all. Check the "
        "mapping before judging a groove by ear. File names start with the rank and the value "
        "that put the file on the list. Only 4/4 beat performances are used unless noted."
    )
    items = []
    for slug, desc, names in listing:
        items.append(f"<li><code>{slug}/</code>: {html.escape(desc)} ({len(names)} files)</li>")
    page.raw(f"<ul>{''.join(items)}</ul>")

    report = out / "report.html"
    report.write_text(
        page.render(
            "Dataset inventory",
            f"GMD, E-GMD and AVP, for music_spec.md sections 1–3 · generated "
            f"{dt.datetime.now():%Y-%m-%d %H:%M}",
        )
    )
    return report


def main(argv: list[str] | None = None) -> int:
    data = REPO_ROOT / "data" / "raw"
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gmd-dir", type=Path, default=data / "gmd" / "groove")
    ap.add_argument("--egmd-dir", type=Path, default=data / "egmd_midi" / "e-gmd-v1.0.0")
    ap.add_argument("--avp-dir", type=Path, default=data / "avp" / "AVP_Dataset")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "inventory")
    ap.add_argument(
        "--ghost-thresholds",
        type=int_list,
        default=[30, 40, 50],
        help="candidate velocity thresholds to compare (comma-separated)",
    )
    ap.add_argument(
        "--flam-windows",
        type=int_list,
        default=[20, 30, 40, 50],
        help="candidate flam windows in ms to compare (comma-separated)",
    )
    ap.add_argument(
        "--grids",
        type=int_list,
        default=[4, 8, 12, 24],
        help="candidate grid resolutions in steps per beat (comma-separated)",
    )
    ap.add_argument(
        "--position-grid",
        type=int,
        default=12,
        help="grid used for position-in-beat plots and bar assignment",
    )
    ap.add_argument(
        "--egmd-sample",
        type=int,
        default=100,
        help="number of performances whose E-GMD MIDI is compared with GMD",
    )
    ap.add_argument("--examples", type=int, default=5, help="MIDI examples per question")
    args = ap.parse_args(argv)
    report = build(args)
    print(f"Report written to {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
