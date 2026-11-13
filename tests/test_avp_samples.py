"""Verify sample boundaries, channel preservation, fades, and source provenance."""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from src.music.spec import load_spec
from src.queries.extract_avp import cut_sound, extract_samples


def test_decay_is_measured_after_peak_not_at_a_waveform_zero():
    settings = load_spec().beatbox.sound_end
    audio = np.r_[np.zeros(10), np.ones(30), np.full(40, 0.01)]
    sample, cut = cut_sound(
        audio, sample_rate=1000, onset_s=0, next_onset_s=None, settings=settings
    )
    assert cut.stop_reason == "decay"
    assert cut.end_frame == 40
    assert cut.fade_frames == 10
    assert sample.shape == (40, 1)
    assert sample[-1, 0] == 0
    assert sample[20, 0] == 1
    assert np.all(audio[10:40] == 1)  # Source remains untouched.


def test_next_onset_and_duration_are_hard_caps_and_channels_survive():
    settings = load_spec().beatbox.sound_end
    audio = np.column_stack([np.ones(1000), np.full(1000, 0.5)])
    sample, cut = cut_sound(
        audio, sample_rate=1000, onset_s=0.1, next_onset_s=0.15, settings=settings
    )
    assert cut.stop_reason == "next_onset"
    assert cut.start_frame == 100 and cut.end_frame == 150
    assert sample.shape == (50, 2)
    np.testing.assert_allclose(sample[:, 1], sample[:, 0] / 2)
    sample, cut = cut_sound(
        audio, sample_rate=1000, onset_s=0, next_onset_s=None, settings=settings
    )
    assert len(sample) == 500
    assert cut.stop_reason == "max_duration"
    _, cut = cut_sound(audio, sample_rate=1000, onset_s=0.98, next_onset_s=None, settings=settings)
    assert cut.stop_reason == "recording_end"
    assert cut.end_frame == len(audio)
    settings = replace(settings, stop_at_next_onset=False)
    sample, _ = cut_sound(audio, sample_rate=1000, onset_s=0, next_onset_s=0.15, settings=settings)
    assert len(sample) == 500


def test_silent_invalid_and_short_sounds():
    settings = load_spec().beatbox.sound_end
    with pytest.raises(ValueError, match="Silent"):
        cut_sound(np.zeros(100), sample_rate=1000, onset_s=0, next_onset_s=None, settings=settings)
    with pytest.raises(ValueError, match="outside"):
        cut_sound(np.ones(100), sample_rate=1000, onset_s=0.1, next_onset_s=None, settings=settings)
    sample, cut = cut_sound(
        np.ones(100), sample_rate=1000, onset_s=0, next_onset_s=0.005, settings=settings
    )
    assert len(sample) == cut.fade_frames == 5
    assert sample[-1, 0] == 0


def test_extraction_uses_invalid_label_onset_as_boundary_and_writes_preview(tmp_path):
    root = tmp_path / "AVP_Dataset"
    source_dir = root / "Personal/Participant_3"
    source_dir.mkdir(parents=True)
    source = source_dir / "P3_Kick_Personal.wav"
    sf.write(source, np.full(8000, 0.5), 8000)
    source.with_suffix(".csv").write_text("0.1,kd,b,u\n0.2, ,b,u\n0.4,kd,b,u\n")
    original = source.read_bytes()
    out = tmp_path / "out"
    summary = extract_samples(root, out)
    catalog = pd.read_csv(out / "sample_catalog.csv")
    assert summary["extracted_samples"] == 2
    assert catalog.iloc[0].next_onset_s == 0.2
    assert catalog.iloc[0].end_frame == 1600
    assert set(catalog.participant_id) == {"P3"}
    for row in catalog.itertuples():
        sample, sr = sf.read(out / row.sample_path, always_2d=True)
        assert len(sample) == row.frames
        assert sr == row.sample_rate
        assert np.all(sample[-1] == 0)
    assert source.read_bytes() == original
    assert "<audio controls" in (out / "preview.html").read_text()
    assert json.loads((out / "summary.json").read_text())["spec_version"] == load_spec().version
