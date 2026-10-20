"""GMD inventory: a per-hit table built from the raw MIDI, plus descriptive analyses.

Nothing here groups drums into voices or decides what counts as the same rhythm.
Hits are described by their raw MIDI note, the Roland articulation name from the
GMD documentation, and the physical instrument that name refers to.
"""

from __future__ import annotations

import re
from pathlib import Path

import mido
import numpy as np
import pandas as pd

# Dataset documentation, not a project decision: GMD's Roland TD-11 note names
# and the 9-class reduction the GMD paper used, copied from
# https://magenta.tensorflow.org/datasets/groove. Used only as display labels.
# pitch: (Roland name, physical instrument, GMD paper class)
GMD_NOTES: dict[int, tuple[str, str, str]] = {
    36: ("Kick", "Kick", "Bass (36)"),
    38: ("Snare (Head)", "Snare", "Snare (38)"),
    40: ("Snare (Rim)", "Snare", "Snare (38)"),
    37: ("Snare X-Stick", "Snare", "Snare (38)"),
    48: ("Tom 1", "Tom 1", "High Tom (50)"),
    50: ("Tom 1 (Rim)", "Tom 1", "High Tom (50)"),
    45: ("Tom 2", "Tom 2", "Low-Mid Tom (47)"),
    47: ("Tom 2 (Rim)", "Tom 2", "Low-Mid Tom (47)"),
    43: ("Tom 3 (Head)", "Tom 3", "High Floor Tom (43)"),
    58: ("Tom 3 (Rim)", "Tom 3", "High Floor Tom (43)"),
    46: ("HH Open (Bow)", "Hi-hat", "Open Hi-Hat (46)"),
    26: ("HH Open (Edge)", "Hi-hat", "Open Hi-Hat (46)"),
    42: ("HH Closed (Bow)", "Hi-hat", "Closed Hi-Hat (42)"),
    22: ("HH Closed (Edge)", "Hi-hat", "Closed Hi-Hat (42)"),
    44: ("HH Pedal", "Hi-hat", "Closed Hi-Hat (42)"),
    49: ("Crash 1 (Bow)", "Crash 1", "Crash Cymbal (49)"),
    55: ("Crash 1 (Edge)", "Crash 1", "Crash Cymbal (49)"),
    57: ("Crash 2 (Bow)", "Crash 2", "Crash Cymbal (49)"),
    52: ("Crash 2 (Edge)", "Crash 2", "Crash Cymbal (49)"),
    51: ("Ride (Bow)", "Ride", "Ride Cymbal (51)"),
    59: ("Ride (Edge)", "Ride", "Ride Cymbal (51)"),
    53: ("Ride (Bell)", "Ride", "Ride Cymbal (51)"),
}

CYMBAL_INSTRUMENTS = ["Hi-hat", "Ride", "Crash 1", "Crash 2"]
HAT_ARTICULATIONS = ["HH Closed", "HH Open", "HH Pedal"]


def articulation(name: str) -> str:
    """Roland name with the bow/edge zone dropped: 'HH Closed (Edge)' -> 'HH Closed'."""
    return re.sub(r" \((Bow|Edge)\)$", "", name)


def note_label(pitch: int) -> str:
    return f"{pitch} {GMD_NOTES[pitch][0]}" if pitch in GMD_NOTES else f"{pitch} (undocumented)"


# --------------------------------------------------------------------------- loading


def _read_midi(path: Path) -> tuple[int, list[int], list[tuple[int, int]], list[tuple]]:
    mf = mido.MidiFile(path)
    tempos, sigs, notes = [], [], []
    for track in mf.tracks:
        tick = 0
        for msg in track:
            tick += msg.time
            if msg.type == "set_tempo":
                tempos.append(msg.tempo)
            elif msg.type == "time_signature":
                sigs.append((msg.numerator, msg.denominator))
            elif msg.type == "note_on" and msg.velocity > 0:
                notes.append((tick, msg.note, msg.velocity))
    return mf.ticks_per_beat, tempos, sigs, notes


def load_gmd(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (performances, hits). `root` is the folder holding info.csv."""
    info = pd.read_csv(root / "info.csv")
    info["style_primary"] = info["style"].str.split("/").str[0]
    perf_rows, hit_frames = [], []
    for r in info.itertuples(index=False):
        tpb, tempos, sigs, notes = _read_midi(root / r.midi_filename)
        notes.sort()
        arr = np.array(notes, dtype=float).reshape(-1, 3)
        beat = arr[:, 0] / tpb
        perf_rows.append(
            {
                "id": r.id,
                "ticks_per_beat": tpb,
                "n_tempo_events": len(tempos),
                "midi_bpm": round(mido.tempo2bpm(tempos[0]), 2) if tempos else np.nan,
                "n_timesig_events": len(sigs),
                "n_hits": len(notes),
                "first_hit_beat": beat.min() if len(beat) else np.nan,
                "last_hit_beat": beat.max() if len(beat) else np.nan,
            }
        )
        hit_frames.append(
            pd.DataFrame(
                {
                    "id": r.id,
                    "tick": arr[:, 0].astype(int),
                    "pitch": arr[:, 1].astype(int),
                    "velocity": arr[:, 2].astype(int),
                    "beat": beat,
                    "sec": beat * 60.0 / r.bpm,
                }
            )
        )
    perf = info.merge(pd.DataFrame(perf_rows), on="id")
    perf["duration_beats"] = perf["duration"] * perf["bpm"] / 60.0
    hits = pd.concat(hit_frames, ignore_index=True)
    meta = perf[["id", "split", "beat_type", "style_primary", "drummer", "bpm", "time_signature"]]
    hits = hits.merge(meta, on="id")
    names = {p: GMD_NOTES.get(p, ("(undocumented)", "(undocumented)", ""))[0] for p in GMD_NOTES}
    hits["name"] = hits["pitch"].map(names).fillna("(undocumented)")
    hits["instrument"] = hits["pitch"].map({p: v[1] for p, v in GMD_NOTES.items()})
    hits["instrument"] = hits["instrument"].fillna("(undocumented)")
    hits["articulation"] = hits["name"].map(articulation)
    hits["label"] = hits["pitch"].map(note_label)
    return perf, hits


def add_grid(hits: pd.DataFrame, ticks_per_beat: int) -> pd.DataFrame:
    """Add nearest-grid columns for one candidate grid resolution."""
    q = np.rint(hits["beat"].to_numpy() * ticks_per_beat).astype(int)
    out = hits.copy()
    out["q"] = q
    out["pos_in_beat"] = q % ticks_per_beat
    out["beat_q"] = q / ticks_per_beat
    out["offset_ms"] = (hits["beat"] - out["beat_q"]) * 60000.0 / hits["bpm"]
    return out


# --------------------------------------------------------------------------- section 1


def note_table(hits: pd.DataFrame, perf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pitch, g in hits.groupby("pitch"):
        name, inst, paper = GMD_NOTES.get(pitch, ("(undocumented)", "", ""))
        rows.append(
            {
                "pitch": pitch,
                "roland_name": name,
                "instrument": inst,
                "gmd_paper_class": paper,
                "hits": len(g),
                "pct_of_all_hits": 100 * len(g) / len(hits),
                "performances_using": g["id"].nunique(),
                "hits_in_beats": int((g["beat_type"] == "beat").sum()),
                "hits_in_fills": int((g["beat_type"] == "fill").sum()),
                "train": int((g["split"] == "train").sum()),
                "validation": int((g["split"] == "validation").sum()),
                "test": int((g["split"] == "test").sum()),
                "documented": pitch in GMD_NOTES,
            }
        )
    return pd.DataFrame(rows).sort_values("hits", ascending=False).reset_index(drop=True)


def style_share(hits: pd.DataFrame, min_style_hits: int = 2000) -> pd.DataFrame:
    """Percent of each style's hits on each note (beats only)."""
    b = hits[hits["beat_type"] == "beat"]
    counts = b.groupby(["label", "style_primary"]).size().unstack(fill_value=0)
    counts = counts.loc[:, counts.sum() >= min_style_hits]
    share = 100 * counts / counts.sum()
    order = b["label"].value_counts().index
    styles = counts.sum().sort_values(ascending=False).index
    return share.reindex(index=[o for o in order if o in share.index], columns=styles)


def timekeepers(hits: pd.DataFrame) -> pd.DataFrame:
    """Per performance: which cymbal articulation carries the most hits."""
    c = hits[hits["instrument"].isin(CYMBAL_INSTRUMENTS)]
    counts = c.groupby(["id", "articulation"]).size().unstack(fill_value=0)
    out = pd.DataFrame(
        {
            "timekeeper": counts.idxmax(axis=1),
            "timekeeper_share": counts.max(axis=1) / counts.sum(axis=1),
            "cymbal_hits": counts.sum(axis=1),
        }
    )
    return out.join(counts.add_prefix("n_"))


def open_hat_followups(hits: pd.DataFrame, grid: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """For each open-hat hit: the next hi-hat event and how far away it is.

    Returns (next_event_counts, per_hit). Gaps are in grid steps of `grid` per beat.
    Hats landing on the same grid step as the open hat count as 'same step'.
    """
    h = add_grid(hits[hits["articulation"].isin(HAT_ARTICULATIONS)], grid)
    h = h.sort_values(["id", "q", "sec"])
    rows = []
    for _, g in h.groupby("id", sort=False):
        q = g["q"].to_numpy()
        art = g["articulation"].to_numpy()
        for i in np.flatnonzero(art == "HH Open"):
            same = (q == q[i]) & (np.arange(len(q)) != i)
            later = np.flatnonzero(q > q[i])
            nxt = later[0] if len(later) else None
            rows.append(
                {
                    "id": g["id"].iat[0],
                    "pos_in_beat": q[i] % grid,
                    "same_step_closed_or_pedal": bool(
                        np.any(same & np.isin(art, ["HH Closed", "HH Pedal"]))
                    ),
                    "next": art[nxt] if nxt is not None else "(none)",
                    "gap_steps": (q[nxt] - q[i]) if nxt is not None else np.nan,
                }
            )
    per_hit = pd.DataFrame(rows)
    counts = per_hit["next"].value_counts().rename_axis("next_hat_event").reset_index()
    counts["pct"] = 100 * counts["count"] / counts["count"].sum()
    return counts, per_hit


# --------------------------------------------------------------------------- section 2


def velocity_table(hits: pd.DataFrame, thresholds: list[int]) -> pd.DataFrame:
    rows = []
    for label, g in hits.groupby("label"):
        v = g["velocity"]
        row = {
            "note": label,
            "hits": len(v),
            "p5": v.quantile(0.05),
            "p25": v.quantile(0.25),
            "median": v.median(),
            "p75": v.quantile(0.75),
            "p95": v.quantile(0.95),
        }
        for t in thresholds:
            row[f"pct_below_{t}"] = 100 * (v < t).mean()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("hits", ascending=False).reset_index(drop=True)


def position_classes(hits: pd.DataFrame, grid: int) -> pd.Series:
    """Coarse metric position of each hit, from its nearest grid step.

    'beats 2 & 4' and 'beats 1 & 3' are on-the-beat hits in 4/4 by bar position;
    'off-beat' is anything not on a beat.
    """
    g = add_grid(hits, grid)
    on_beat = g["pos_in_beat"] == 0
    beat_in_bar = np.floor(g["beat_q"]).astype(int) % 4
    return pd.Series(
        np.select(
            [on_beat & beat_in_bar.isin([1, 3]), on_beat],
            ["beats 2 & 4", "beats 1 & 3"],
            default="off-beat",
        ),
        index=hits.index,
    )


def velocity_by_position(
    hits: pd.DataFrame, articulations: list[str], grid: int, thresholds: list[int]
) -> pd.DataFrame:
    b = hits[(hits["beat_type"] == "beat") & (hits["time_signature"] == "4-4")]
    b = b[b["articulation"].isin(articulations)].copy()
    b["position"] = position_classes(b, grid)
    rows = []
    for (art, pos), g in b.groupby(["articulation", "position"]):
        row = {
            "articulation": art,
            "position": pos,
            "hits": len(g),
            "median": g["velocity"].median(),
        }
        for t in thresholds:
            row[f"pct_below_{t}"] = 100 * (g["velocity"] < t).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def same_instrument_gaps(hits: pd.DataFrame, max_ms: float) -> pd.DataFrame:
    """Consecutive hits on the same physical instrument closer than `max_ms`."""
    h = hits.sort_values(["id", "instrument", "sec"])
    same = (h["id"] == h["id"].shift()) & (h["instrument"] == h["instrument"].shift())
    gap_ms = (h["sec"] - h["sec"].shift()) * 1000.0
    keep = same & (gap_ms < max_ms)
    return pd.DataFrame(
        {
            "id": h["id"][keep],
            "instrument": h["instrument"][keep],
            "first": h["label"].shift()[keep],
            "second": h["label"][keep],
            "gap_ms": gap_ms[keep],
        }
    ).reset_index(drop=True)


def grid_collisions(hits: pd.DataFrame, grids: list[int]) -> pd.DataFrame:
    """Percent of hits that land on a grid step already taken by the same instrument."""
    rows = []
    for grid in grids:
        q = np.rint(hits["beat"].to_numpy() * grid).astype(int)
        dup = pd.DataFrame({"id": hits["id"], "inst": hits["instrument"], "q": q}).duplicated()
        pct = 100 * dup.groupby(hits["instrument"]).mean()
        rows.append(pct.rename(f"{grid} steps/beat"))
    return pd.DataFrame(rows).T.sort_index().rename_axis("instrument")


def timing_offsets(hits: pd.DataFrame, grids: list[int]) -> pd.DataFrame:
    """Distribution of |distance to nearest grid step| in ms, per candidate grid."""
    rows = []
    for grid in grids:
        off = add_grid(hits, grid)["offset_ms"].abs()
        rows.append(
            {
                "grid (steps/beat)": grid,
                "median_ms": off.median(),
                "p90_ms": off.quantile(0.9),
                "p99_ms": off.quantile(0.99),
            }
        )
    return pd.DataFrame(rows)


def swing_estimates(hits: pd.DataFrame, min_hits: int = 16) -> pd.DataFrame:
    """Per performance: median position of off-beat 8ths and of off-beat 16ths.

    8th level: hits between 11/24 and 17/24 of the beat. Straight '&' sits at 0.50,
    triplet swing at 0.667. 16th level: the same idea inside each half beat, with hits
    between 5/12 and 10/12 of the half beat. Straight 'e'/'a' sit at 0.50, triplet swing
    at 0.667. 8th-note triplets also land at 0.667 at the 16th level, so a high
    16th value can mean either swung 16ths or triplets. Kick hits are left out.
    Ratio = position / (1 - position): 1.0 is straight, 2.0 is triplet swing.
    """
    h = hits[hits["instrument"] != "Kick"]
    frac = h["beat"] % 1.0
    half = (frac * 2.0) % 1.0
    e8 = frac.between(11 / 24, 17 / 24)
    e16 = half.between(5 / 12, 10 / 12)
    g8 = frac[e8].groupby(h["id"][e8])
    g16 = half[e16].groupby(h["id"][e16])
    out = pd.DataFrame(
        {
            "pos8": g8.median(),
            "n8": g8.size(),
            "pos16": g16.median(),
            "n16": g16.size(),
        }
    )
    out.loc[out["n8"] < min_hits, "pos8"] = np.nan
    out.loc[out["n16"] < min_hits, "pos16"] = np.nan
    out["ratio8"] = out["pos8"] / (1 - out["pos8"])
    out["ratio16"] = out["pos16"] / (1 - out["pos16"])
    return out


# --------------------------------------------------------------------------- section 3


def bar_windows(
    hits: pd.DataFrame, perf: pd.DataFrame, bars_per_window: int, grid: int
) -> pd.DataFrame:
    """Non-overlapping windows of `bars_per_window` 4/4 bars from beat 0, beats only.

    Hits are assigned to bars by their nearest grid step, so a downbeat played a
    hair early still belongs to the new bar. One row per window, with hit counts per
    note so different kick/snare rules can be checked afterwards.
    """
    p = perf[(perf["beat_type"] == "beat") & (perf["time_signature"] == "4-4")]
    h = add_grid(hits[hits["id"].isin(p["id"])], grid)
    h["window"] = (np.floor(h["beat_q"] / 4).astype(int)) // bars_per_window
    counts = h.groupby(["id", "window", "pitch"]).size().unstack(fill_value=0)
    counts.columns = [f"n_{c}" for c in counts.columns]
    # Include empty windows inside the span of each performance.
    last = h.groupby("id")["window"].max()
    idx = pd.MultiIndex.from_tuples(
        [(i, w) for i, n in last.items() for w in range(n + 1)], names=["id", "window"]
    )
    counts = counts.reindex(idx, fill_value=0).reset_index()
    return counts.merge(p[["id", "split", "style_primary"]], on="id")


def windows_with(windows: pd.DataFrame, kick: list[int], snare: list[int]) -> pd.Series:
    def any_of(pitches):
        cols = [f"n_{p}" for p in pitches if f"n_{p}" in windows]
        return windows[cols].sum(axis=1) > 0 if cols else pd.Series(False, index=windows.index)

    return any_of(kick) & any_of(snare)


def first_bar_content(hits: pd.DataFrame, perf: pd.DataFrame, grid: int) -> pd.DataFrame:
    """What is in bar 1 of each 4/4 performance: flags for possible count-ins or pickups."""
    p = perf[perf["time_signature"] == "4-4"]
    h = add_grid(hits[hits["id"].isin(p["id"])], grid)
    bar1 = h[h["beat_q"] < 4]
    g = bar1.groupby("id")
    out = pd.DataFrame(
        {
            "bar1_hits": g.size(),
            "bar1_has_kick": g["instrument"].apply(lambda s: (s == "Kick").any()),
            "bar1_has_snare": g["instrument"].apply(lambda s: (s == "Snare").any()),
            "bar1_cymbals_only": g["instrument"].apply(lambda s: s.isin(CYMBAL_INSTRUMENTS).all()),
        }
    )
    out = p.set_index("id")[["beat_type", "first_hit_beat"]].join(out)
    out["bar1_hits"] = out["bar1_hits"].fillna(0).astype(int)
    return out
