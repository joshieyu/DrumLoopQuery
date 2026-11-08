"""Check beatbox simplification, single-voice provenance, and paired exports."""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from src.music.rhythm import Note, clip_grid, dualize, pattern_id
from src.music.spec import load_spec
from src.queries.beatbox_preview import beatbox_events, build_pairs, mix_query


def record(notes):
    spec = load_spec()
    grid = clip_grid(notes, bpm=120, spec=spec)
    return {
        "source_performance": "source",
        "start_bar": 0,
        "tempo_bpm": 120,
        "pattern_id": pattern_id(grid),
        "dualized_pattern_id": pattern_id(dualize(grid, spec)),
        "notes": [n._asdict() for n in notes],
        "split": "train",
    }


def test_priority_hat_mapping_thinning_and_ghosts():
    notes = [
        Note(36, 100, 0),
        Note(38, 100, 0),
        Note(42, 100, 0),
        Note(46, 100, 0.5),
        Note(38, 100, 1),
        Note(51, 100, 1),
        Note(36, 14, 3),
        Note(49, 100, 3),
    ]
    notes += [Note(42, 100, b) for b in (2, 2.25, 2.5, 2.75)]
    source = record(notes)
    spec = load_spec()
    spec = replace(spec, beatbox=replace(spec.beatbox, thin_16th_hats_to_8ths_prob=1))
    events, counts = beatbox_events(source, spec=spec, seed=0)
    assert [(e["beat"], e["label"]) for e in events] == [
        (0, "kd"),
        (0.5, "hho"),
        (1, "sd"),
        (2, "hhc"),
        (2.5, "hhc"),
    ]
    assert counts == {"thinned_hat_hits": 2, "suppressed_simultaneous": 3}
    assert source["pattern_id"] == record(notes)["pattern_id"]  # Target is not simplified.
    assert events == beatbox_events(source, spec=spec, seed=0)[0]


def sample_fixture(folder):
    folder.mkdir(parents=True)
    rows = []
    for mode in ("Personal", "Fixed"):
        for label in ("kd", "sd", "hhc", "hho"):
            name = f"P1_{mode}_{label}.wav"
            sf.write(folder / name, np.full(800, 0.1), 8000, subtype="FLOAT")
            rows.append(
                {
                    "participant_id": "P1",
                    "mode": mode,
                    "kind": "Kick",
                    "label": label,
                    "utterance_id": name,
                    "sample_path": name,
                }
            )
    return pd.DataFrame(rows)


def test_mix_keeps_participant_mode_event_timing_and_is_reproducible(tmp_path):
    samples = sample_fixture(tmp_path / "samples")
    events = [{"beat": 0, "label": "kd"}, {"beat": 1, "label": "sd"}]
    args = dict(participant="P1", mode="Personal", bpm=120, pre_roll_s=0.25, duration_s=2, seed=3)
    audio, sr, trace = mix_query(events, samples, tmp_path / "samples", **args)
    assert sr == 8000 and len(audio) == 16000
    assert [t["frame"] for t in trace] == [2000, 6000]
    assert all(t["participant_id"] == "P1" and t["mode"] == "Personal" for t in trace)
    assert not np.any(audio[:2000]) and np.any(audio[2000:2800])
    np.testing.assert_array_equal(
        audio, mix_query(events, samples, tmp_path / "samples", **args)[0]
    )
    with pytest.raises(ValueError, match="No kd"):
        mix_query(events, samples, tmp_path / "samples", **{**args, "participant": "missing"})


def test_paired_export_has_matching_labels_and_self_contained_audio(tmp_path):
    spec = load_spec()
    sample_root = tmp_path / "samples"
    samples = sample_fixture(sample_root)
    samples.to_csv(sample_root / "sample_catalog.csv", index=False)
    library = tmp_path / "library"
    catalog = tmp_path / "catalog"
    library.mkdir()
    catalog.mkdir()
    for folder in (sample_root, library, catalog):
        (folder / "summary.json").write_text(json.dumps({"spec_version": spec.version}))
    source = record([Note(36, 100, 0), Note(38, 100, 1)])
    (catalog / "loops.jsonl").write_text(json.dumps(source) + "\n")
    drum = library / "drum.wav"
    sf.write(drum, np.full((34000, 2), 0.1), 8000)
    pd.DataFrame(
        [
            {
                "clip_id": "loop_kit",
                "source_performance": "source",
                "start_bar": 0,
                "pattern_id": source["pattern_id"],
                "dualized_pattern_id": source["dualized_pattern_id"],
                "split": "train",
                "audio_path": str(drum),
                "tempo_bpm": 120,
                "pre_roll_s": 0.25,
                "pattern_duration_s": 4,
            }
        ]
    ).to_csv(library / "library_manifest.csv", index=False)
    out = tmp_path / "pairs"
    summary = build_pairs(library, catalog, sample_root, out)
    assert summary["queries"] == 2 and summary["drum_loops"] == 1
    q = pd.read_csv(out / "query_manifest.csv")
    assert set(q.pattern_id) == {source["pattern_id"]}
    assert set(q.target_clip_id) == {"loop_kit"}
    assert set(q["mode"]) == {"Personal", "Fixed"}
    assert len(list((out / "drums").glob("*.wav"))) == 1
    assert len(list((out / "queries").glob("*.wav"))) == 2
    assert (out / "preview.html").read_text().count("<audio controls") == 3
