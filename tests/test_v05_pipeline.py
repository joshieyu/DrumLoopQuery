"""Check clip framing, fixed participant splits, and reproducible perturbations."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from src.audio import frame_audio
from src.music.spec import load_spec
from src.queries.augment import augment_query
from src.splits import assign_participants, resolve_patterns


def test_frame_cuts_or_pads_and_fades_inside_endpoint():
    spec = load_spec()
    audio = frame_audio(
        np.ones((2000, 2)), sample_rate=1000, duration_s=1, settings=spec.loops.audio
    )
    assert audio.shape == (1000, 2)
    assert np.all(audio[-1] == 0) and np.all(audio[500] == 1)
    padded = frame_audio(np.ones(50), sample_rate=1000, duration_s=1, settings=spec.loops.audio)
    assert len(padded) == 1000 and not np.any(padded[50:])


def test_split_resolution_keeps_all_copies_only_in_most_held_out_split():
    source = pd.DataFrame(
        {
            "pattern_id": ["a", "a", "b", "b", "b"],
            "split": ["train", "validation", "train", "test", "test"],
        }
    )
    kept, removed = resolve_patterns(source)
    assert kept.split.tolist() == ["validation", "test", "test"]
    assert removed.split.tolist() == ["train", "train"]
    people = [f"P{i}" for i in range(1, 29)]
    counts = {"train": 20, "validation": 3, "test": 5}
    a = assign_participants(people, counts=counts, seed=0)
    b = assign_participants(people[::-1], counts=counts, seed=0)
    pd.testing.assert_frame_equal(a, b)
    assert a.participant_id.is_unique and a.split.value_counts().to_dict() == counts


def test_augmentation_reproducibility_parameters_and_measured_noise(tmp_path):
    sample = np.sin(np.arange(100) * 0.2).astype(np.float32) * 0.1
    sf.write(tmp_path / "hit.wav", sample, 1000, subtype="FLOAT")
    trace = [
        {"voice": "kick", "frame": 250, "sample_path": "hit.wav"},
        {"voice": "snare", "frame": 700, "sample_path": "hit.wav"},
    ]
    spec = load_spec()
    args = dict(
        sample_rate=1000,
        duration_s=1.25,
        spec=spec,
        seed=5,
        normalization={"target_rms_dbfs": -20, "peak_ceiling_dbfs": -1},
    )
    a, params, events = augment_query(trace, tmp_path, **args)
    b, other, trace_b = augment_query(trace, tmp_path, **args)
    np.testing.assert_array_equal(a, b)
    assert params == other and events == trace_b
    assert params["measured_snr_db_before_fade"] == pytest.approx(params["snr_db"])
    assert spec.sloppiness.offset_ms[0] <= params["offset_ms"] <= spec.sloppiness.offset_ms[1]
    assert spec.sloppiness.jitter_ms[0] <= params["jitter_ms"] <= spec.sloppiness.jitter_ms[1]
    assert np.isfinite(a).all() and a[-1] == 0 and len(a) == 1250
    spec = replace(
        spec, sloppiness=replace(spec.sloppiness, drop_prob={v: (1, 1) for v in spec.voices})
    )
    with pytest.raises(ValueError, match="All events dropped"):
        augment_query(trace, tmp_path, **{**args, "spec": spec})
