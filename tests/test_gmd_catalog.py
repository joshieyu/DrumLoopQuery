"""Check exported labels and conflicts across official source splits."""

import json

import pandas as pd
import pretty_midi

from src.library.catalog_gmd import catalog_gmd
from src.music.rhythm import Note, clip_grid, pattern_id
from src.music.spec import load_spec


def test_catalog_preserves_splits_and_reports_conflicts(tmp_path):
    root = tmp_path / "gmd"
    root.mkdir()
    rows = []
    for split in ("train", "test"):
        midi = pretty_midi.PrettyMIDI(initial_tempo=120)
        drums = pretty_midi.Instrument(program=0, is_drum=True)
        drums.notes = [
            pretty_midi.Note(100, 36, 0, 0.1),
            pretty_midi.Note(100, 38, 0.5, 0.6),
            pretty_midi.Note(80, 49, 7.5, 7.6),
        ]
        midi.instruments.append(drums)
        midi.write(str(root / f"{split}.mid"))
        rows.append(
            dict(
                id=split,
                midi_filename=f"{split}.mid",
                time_signature="4-4",
                bpm=120,
                beat_type="beat",
                split=split,
            )
        )
    pd.DataFrame(rows).to_csv(root / "info.csv", index=False)
    out = tmp_path / "out"
    summary = catalog_gmd(root, out)
    assert summary["clips"] == 1
    assert summary["failed_performances"] == 0
    assert summary["original_patterns_shared_across_splits"] == 1
    assert summary["patterns_shared_across_splits"] == 0
    assert summary["removed_cross_split_clips"] == 1
    conflicts = pd.read_csv(out / "original_split_clashes.csv")
    assert set(conflicts.split) == {"train", "test"}
    record = json.loads((out / "loops.jsonl").read_text().splitlines()[0])
    notes = [Note(**n) for n in record["notes"]]
    assert pattern_id(clip_grid(notes, bpm=120, spec=load_spec())) == record["pattern_id"]
    assert record["grid"]["voices"] == ["kick", "snare", "hats"]
