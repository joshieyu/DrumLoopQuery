"""Catalog annotated AVP sounds without cutting or changing audio.

Run this file from the IDE or use ``python -m src.queries.index_avp``.
See docs/data_pipeline.md for fields, validation, and the sample-boundary dependency.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tools.inventory.avp import LABEL_NAMES, LABELS, load_avp

REPO_ROOT = Path(__file__).resolve().parents[2]


def build_index(root: Path, out_dir: Path) -> dict:
    """Write a valid-onset index, issue log, counts, and provenance summary.

    Paths in the index are relative to root, recorded separately in summary.json.
    All modes and kinds are cataloged; this is not a training sample selection.
    No participant splits or audio end times are inferred.
    """
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"AVP directory not found: {root}")
    files, utterances = load_avp(root)
    if files.empty:
        raise ValueError(f"No AVP recordings found in {root}")
    if utterances.empty:
        raise ValueError("No AVP annotations found")
    keys = ["participant", "mode", "kind"]
    # The existing loader retains CSV row order within each source file.
    utterances["annotation_row"] = utterances.groupby(keys).cumcount() + 1
    table = utterances.merge(files, on=keys, how="left", validate="many_to_one")
    table["audio_path"] = table["path"].map(lambda p: p.relative_to(root).as_posix())
    table["annotation_path"] = table["path"].map(
        lambda p: p.with_suffix(".csv").relative_to(root).as_posix()
    )
    table["utterance_id"] = table["annotation_path"] + ":" + table.annotation_row.astype(str)
    table["participant_id"] = "P" + table.participant.astype(str)
    table["label"] = table.label_raw.fillna("").astype(str).str.strip()
    valid_label = table.label.isin(LABELS)
    valid_onset = (
        np.isfinite(table.onset_s) & (table.onset_s >= 0) & (table.onset_s < table.duration_s)
    )
    duplicate = table.duplicated(["audio_path", "onset_s"], keep=False)
    issues = []
    for row, label_ok, onset_ok, repeated in zip(
        table.itertuples(), valid_label, valid_onset, duplicate, strict=True
    ):
        reasons = []
        if not label_ok:
            reasons.append("blank_label" if row.label == "" else "unknown_label")
        if not onset_ok:
            reasons.append("onset_outside_audio_or_nonfinite")
        if repeated:
            reasons.append("duplicate_onset")
        if label_ok and str(row.label_raw) != row.label:
            reasons.append("label_whitespace_trimmed")
        for reason in reasons:
            issues.append(
                {
                    "audio_path": row.audio_path,
                    "annotation_row": row.annotation_row,
                    "label_raw": row.label_raw,
                    "onset_s": row.onset_s,
                    "issue": reason,
                    "excluded": not (label_ok and onset_ok and not repeated),
                }
            )
    missing_annotations = []
    for path in files.path:
        if not path.with_suffix(".csv").exists():
            relative = path.relative_to(root).as_posix()
            missing_annotations.append(relative)
            issues.append({"audio_path": relative, "issue": "missing_annotation", "excluded": True})
    index = table[valid_label & valid_onset & ~duplicate].copy()
    index["sound_class"] = index.label.map(LABEL_NAMES)
    columns = [
        "utterance_id",
        "participant_id",
        "mode",
        "kind",
        "label",
        "sound_class",
        "audio_path",
        "annotation_path",
        "annotation_row",
        "onset_s",
        "samplerate",
        "channels",
        "duration_s",
        "onset_phoneme",
        "coda_phoneme",
    ]
    index = index.sort_values(["participant", "mode", "kind", "onset_s", "annotation_row"])
    counts = index.groupby(["participant_id", "mode", "kind", "label"]).size()
    summary = {
        "dataset_root": str(root),
        "source": "https://zenodo.org/records/5036529",
        "recordings": len(files),
        "participants": int(files.participant.nunique()),
        "annotation_rows": len(table),
        "indexed_sounds": len(index),
        "excluded_rows": len(table) - len(index),
        "missing_annotation_files": missing_annotations,
        "issue_counts": pd.Series([r["issue"] for r in issues], dtype=str).value_counts().to_dict(),
        "split_status": "unassigned; do not use all participants for training",
        "audio_status": "original recordings unchanged; no samples cut or end times inferred",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    index[columns].to_csv(out_dir / "utterance_index.csv", index=False)
    pd.DataFrame(
        issues,
        columns=[
            "audio_path",
            "annotation_row",
            "label_raw",
            "onset_s",
            "issue",
            "excluded",
        ],
    ).to_csv(out_dir / "annotation_issues.csv", index=False)
    counts.rename("sounds").reset_index().to_csv(out_dir / "counts.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--avp-dir", type=Path, default=REPO_ROOT / "data/raw/avp/AVP_Dataset")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results/avp_index")
    args = parser.parse_args()
    try:
        summary = build_index(args.avp_dir, args.out)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Indexing failed: {error}\n")
    print(json.dumps(summary, indent=2))
    print(f"Index ready: {(args.out / 'utterance_index.csv').resolve()}")


if __name__ == "__main__":
    main()
