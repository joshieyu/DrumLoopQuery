"""Verify labeled-catalog rendering with a tiny real SFZ/sample fixture."""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from src.library import render_preview
from src.music.rhythm import Note, clip_grid, dualize, pattern_id
from src.music.spec import KitNote, load_spec


def catalog_fixture(tmp_path):
    spec = load_spec()
    notes = [
        Note(36, 100, -0.8),
        Note(36, 90, -0.01),
        Note(38, 100, 1),
        Note(49, 70, 4),
        Note(36, 80, 8.5),
    ]
    grid = clip_grid(notes, bpm=120, spec=spec)
    row = {
        "clip_id": "source:bar0",
        "source_performance": "source",
        "start_bar": 0,
        "start_beat": 0,
        "end_beat": 8,
        "tempo_bpm": 120,
        "split": "test",
        "pattern_id": pattern_id(grid),
        "dualized_pattern_id": pattern_id(dualize(grid, spec)),
        "midi_path": "source.mid",
        "notes": [n._asdict() for n in notes],
        "grid": {
            "voices": grid.voices,
            "steps_per_beat": grid.steps_per_beat,
            "onsets": grid.onsets.tolist(),
        },
    }
    folder = tmp_path / "catalog"
    folder.mkdir()
    (folder / "summary.json").write_text(
        json.dumps(
            {
                "spec_version": spec.version,
                "failed_performances": 0,
                "gmd_root": str(tmp_path),
            }
        )
    )
    (folder / "loops.jsonl").write_text(json.dumps(row) + "\n")
    pd.DataFrame(columns=["pattern_id"]).to_csv(folder / "split_conflicts.csv", index=False)
    return folder, row


def test_ownership_excludes_label_context_but_keeps_early_and_dropped_hits(tmp_path):
    _, row = catalog_fixture(tmp_path)
    events = render_preview.audio_events(row, load_spec())
    assert [pitch for _, pitch, _ in events] == [36, 38, 49]
    assert events[0][0] == pytest.approx(load_spec().loops.audio.lead_in_s - 0.005)
    row["pattern_id"] = "wrong"
    with pytest.raises(ValueError, match="disagrees"):
        render_preview.audio_events(row, load_spec())


def test_preview_renders_real_audio_and_retains_review_notes(tmp_path, monkeypatch):
    folder, _ = catalog_fixture(tmp_path)
    spec = load_spec()
    samples = tmp_path / "kits"
    samples.mkdir()
    sr = 8000
    signal = np.sin(np.arange(800) * 2 * np.pi * 80 / sr) * 0.2
    sf.write(samples / "hit.wav", signal, sr, subtype="FLOAT")
    (samples / "kit.sfz").write_text("<region> key=36 sample=hit.wav amp_veltrack=0")
    kit = replace(
        spec.rendering.kits[0],
        program="kit.sfz",
        split="test",
        samples=None,
        note_map={p: KitNote(36, {}) for p in spec.note_to_group},
    )
    spec = replace(spec, rendering=replace(spec.rendering, kits=(kit,)))
    monkeypatch.setattr(render_preview, "load_spec", lambda: spec)
    out = tmp_path / "out"
    summary = render_preview.render_previews(
        folder, out, kit_id=kit.kit_id, kits_dir=samples, per_split=1
    )
    assert summary["rendered_clips"] == 1
    manifest = pd.read_csv(out / "library_manifest.csv")
    row = manifest.iloc[0]
    audio, rate = sf.read(row.audio_path, always_2d=True)
    assert rate == sr and audio.shape[1] == 2
    assert row.pattern_duration_s == 4
    assert len(audio) >= round((4 + load_spec().loops.audio.lead_in_s) * sr)
    assert np.isfinite(audio).all() and np.any(audio)
    assert float(np.abs(audio).max()) <= 10 ** (render_preview.PEAK_CEILING_DB / 20)
    assert row.split == "test" and row.status == "audition_only"
    assert "<audio controls" in (out / "preview.html").read_text()
    review = pd.read_csv(out / "listening_notes.csv", keep_default_na=False)
    review.loc[0, "notes"] = "Keep this review"
    review.to_csv(out / "listening_notes.csv", index=False)
    render_preview.render_previews(folder, out, kit_id=kit.kit_id, kits_dir=samples, per_split=1)
    assert pd.read_csv(out / "listening_notes.csv").iloc[0].notes == "Keep this review"


def test_preview_rejects_stale_catalog_and_omits_known_conflicts(tmp_path):
    folder, row = catalog_fixture(tmp_path)
    pd.DataFrame([{"pattern_id": row["pattern_id"]}]).to_csv(
        folder / "split_conflicts.csv", index=False
    )
    with pytest.raises(ValueError, match="No non-conflicting"):
        render_preview.select_records(folder, per_split=1, spec=load_spec())
    summary = json.loads((folder / "summary.json").read_text())
    summary["spec_version"] = "old"
    (folder / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="stale"):
        render_preview.select_records(folder, per_split=1, spec=load_spec())
