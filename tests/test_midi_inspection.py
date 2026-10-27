"""Check exported timing and preservation of raw hits on a tiny MIDI fixture."""

import json

import mido
import pandas as pd
import pytest

from src.library.inspect_midi import inspect_performance
from src.music.spec import load_spec


def test_inspection_preserves_hits_and_beats_across_tempo_changes(tmp_path):
    spec = load_spec()
    root = tmp_path / "gmd"
    root.mkdir()
    pd.DataFrame(
        [
            {
                "id": "sample",
                "midi_filename": "sample.mid",
                "time_signature": spec.loops.time_signatures[0],
                "bpm": sum(spec.loops.performance_tempo_bpm) / 2,
                "beat_type": "beat",
                "split": "test",
            }
        ]
    ).to_csv(root / "info.csv", index=False)
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.extend(
        [
            mido.MetaMessage("set_tempo", tempo=500000),
            mido.Message("note_on", channel=9, note=36, velocity=1),
            mido.Message("note_off", channel=9, note=36, time=240),
            mido.MetaMessage("set_tempo", tempo=1000000, time=240),
            mido.Message("note_on", channel=9, note=49, velocity=80, time=480),
            mido.Message("note_off", channel=9, note=49, time=240),
        ]
    )
    midi.save(root / "sample.mid")
    out = tmp_path / "out"
    report = inspect_performance(root, out)
    hits = pd.read_csv(out / "hits.csv")
    assert hits["beat"].tolist() == [0, 2]
    assert hits["start_sec"].tolist() == pytest.approx([0, 1.5])
    assert hits["velocity"].tolist() == [1, 80]  # No ghost-hit removal in raw export.
    assert hits["spec_group"].tolist() == ["kick", "drop"]
    assert json.loads((out / "metadata.json").read_text())["source_split"] == "test"
    assert report.exists()
    with pytest.raises(ValueError, match="No eligible"):
        inspect_performance(root, out, performance_id="missing")
