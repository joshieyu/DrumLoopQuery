"""Check annotation validation and source identity using tiny WAV/CSV fixtures."""

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from src.queries.index_avp import build_index


def recording(root, name, annotations):
    folder = root / "Personal/Participant_3"
    folder.mkdir(parents=True, exist_ok=True)
    wav = folder / f"P3_{name}_Personal.wav"
    sf.write(wav, np.zeros(8000), 8000)
    if annotations is not None:
        wav.with_suffix(".csv").write_text(annotations)
    return wav


def test_index_validates_rows_and_retains_source_identity(tmp_path):
    root = tmp_path / "avp"
    wav = recording(root, "Kick", "0.1, kd ,b,u\n0.2,,b,u\n1.0,kd,b,u\n0.4,other,b,u\n")
    recording(root, "Snare", None)
    original = wav.read_bytes()
    out = tmp_path / "out"
    summary = build_index(root, out)
    index = pd.read_csv(out / "utterance_index.csv")
    assert len(index) == 1
    assert index.iloc[0].label == "kd"
    assert index.iloc[0].participant_id == "P3"
    assert index.iloc[0].annotation_row == 1
    assert (root / index.iloc[0].audio_path).exists()
    assert index.iloc[0].utterance_id.endswith(".csv:1")
    assert summary["excluded_rows"] == 3
    assert summary["issue_counts"]["missing_annotation"] == 1
    assert wav.read_bytes() == original
    # Rebuilding does not alter identifiers or ordering.
    build_index(root, out)
    pd.testing.assert_frame_equal(index, pd.read_csv(out / "utterance_index.csv"))


def test_duplicate_onsets_are_reported_and_empty_index_has_headers(tmp_path):
    root = tmp_path / "avp"
    recording(root, "Kick", "0.1,kd\n0.1,kd\n")
    out = tmp_path / "out"
    summary = build_index(root, out)
    assert summary["indexed_sounds"] == 0
    assert summary["issue_counts"]["duplicate_onset"] == 2
    assert "utterance_id" in pd.read_csv(out / "utterance_index.csv").columns
    with pytest.raises(FileNotFoundError, match="not found"):
        build_index(tmp_path / "missing", out)
