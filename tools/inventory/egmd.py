"""E-GMD inventory: kit list, which GMD performances it covers, and whether its MIDI
matches GMD's for the same performance."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tools.inventory.gmd import _read_midi


def load_egmd(root: Path) -> pd.DataFrame:
    return pd.read_csv(root / "e-gmd-v1.0.0.csv")


def kit_table(egmd: pd.DataFrame) -> pd.DataFrame:
    return (
        egmd.groupby("kit_name")
        .agg(sequences=("id", "nunique"), hours=("duration", lambda s: s.sum() / 3600))
        .reset_index()
        .sort_values("kit_name")
    )


def coverage(perf: pd.DataFrame, egmd: pd.DataFrame) -> pd.DataFrame:
    """GMD performances by whether E-GMD includes them."""
    p = perf.assign(in_egmd=perf["id"].isin(egmd["id"]))
    return (
        p.groupby(["beat_type", "split", "in_egmd"])
        .agg(performances=("id", "size"), minutes=("duration", lambda s: s.sum() / 60))
        .reset_index()
    )


def midi_check(
    gmd_root: Path, egmd_root: Path, perf: pd.DataFrame, egmd: pd.DataFrame, n_ids: int
) -> pd.DataFrame:
    """Compare note lists of every kit's E-GMD MIDI with the GMD MIDI, for a sample of ids."""
    ids = egmd["id"].drop_duplicates().sample(min(n_ids, egmd["id"].nunique()), random_state=0)
    gmd_files = perf.set_index("id")["midi_filename"]
    rows = []
    for i in ids:
        _, _, _, g_notes = _read_midi(gmd_root / gmd_files[i])
        g_sorted = sorted(g_notes)
        g_pitches = {n for _, n, _ in g_notes}
        for r in egmd[egmd["id"] == i].itertuples(index=False):
            _, _, _, e_notes = _read_midi(egmd_root / r.midi_filename)
            e_pitches = {n for _, n, _ in e_notes}
            rows.append(
                {
                    "id": i,
                    "kit_name": r.kit_name,
                    "identical": sorted(e_notes) == g_sorted,
                    "gmd_notes": len(g_notes),
                    "egmd_notes": len(e_notes),
                    "pitches_only_in_egmd": " ".join(map(str, sorted(e_pitches - g_pitches))),
                    "pitches_only_in_gmd": " ".join(map(str, sorted(g_pitches - e_pitches))),
                }
            )
    return pd.DataFrame(rows)


def always_different(chk: pd.DataFrame) -> pd.Index:
    """Performances whose E-GMD MIDI differs from GMD on every kit checked."""
    rate = chk.groupby("id")["identical"].mean()
    return rate[rate == 0].index


def kit_remaps(chk: pd.DataFrame, top: int = 3) -> pd.DataFrame:
    """Per kit: how often its MIDI differs from GMD, and the most common note sets that
    appear only in E-GMD or only in GMD for those files."""
    d = chk[~chk["id"].isin(always_different(chk))]
    rows = []
    for kit, g in d.groupby("kit_name"):
        diff = g[~g["identical"]]
        if diff.empty:
            continue

        def common(col: str, diff: pd.DataFrame = diff) -> str:
            vc = diff[col].fillna("").replace("", "(none)").value_counts().head(top)
            return "; ".join(f"{k} ({v})" for k, v in vc.items())

        rows.append(
            {
                "kit_name": kit,
                "files_differing": len(diff),
                "of_checked": len(g),
                "notes only in E-GMD": common("pitches_only_in_egmd"),
                "notes only in GMD": common("pitches_only_in_gmd"),
            }
        )
    return pd.DataFrame(rows).sort_values("files_differing", ascending=False)
