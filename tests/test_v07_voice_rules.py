"""Verify speed priority and choking on actual retained placements, before noise."""

from dataclasses import replace

import numpy as np
import pytest
import soundfile as sf

from src.music.spec import load_spec
from src.queries.augment import augment_query
from src.queries.beatbox_preview import limit_event_rate
from src.queries.mixing import mix_placements


def event(voice, milliseconds):
    return {"voice": voice, "beat": milliseconds / 500}  # 120 BPM fixture


def test_speed_limit_is_priority_first_not_chronological():
    spec = load_spec()
    source = [
        event("hats", 0),
        event("kick", 100),
        event("snare", 190),
        event("snare", 300),
        event("hats", 550),
    ]
    kept = limit_event_rate(source, bpm=120, spec=spec)
    assert kept == [source[1], source[3], source[4]]
    assert source[0]["voice"] == "hats"  # Input remains unchanged.


def test_exact_minimum_is_kept_and_every_prior_event_is_checked():
    spec = load_spec()
    source = [
        event("kick", 0),
        event("kick", 120),
        event("snare", 240),
        event("hats", 489),
        event("hats", 490),
    ]
    kept = limit_event_rate(source, bpm=120, spec=spec)
    assert kept == [source[0], source[1], source[2], source[4]]
    # A future-in-time higher-priority kick also blocks an earlier hat.
    assert limit_event_rate([event("hats", 0), event("kick", 200)], bpm=120, spec=spec) == [
        event("kick", 200)
    ]


def test_choke_fades_at_actual_next_onset_and_does_not_mutate_samples():
    settings = load_spec().beatbox
    signal = np.ones(300, dtype=np.float32)
    mixed, details = mix_placements(
        [(200, signal), (100, signal)], frames=1000, sample_rate=1000, settings=settings
    )
    assert details[1]["audio_end_frame"] == 200
    assert details[1]["choke_fade_frames"] == 10
    assert details[1]["choked_at_next_sound"]
    assert mixed[199] == 0 and mixed[200] == 1
    assert mixed.max() == 1  # No summed vocal layers.
    assert np.all(signal == 1)
    # A naturally short sound stays short; a hit outside the clip cannot choke it.
    _, details = mix_placements(
        [(100, np.ones(20)), (150, signal), (1200, signal)],
        frames=1000,
        sample_rate=1000,
        settings=settings,
    )
    assert details[0]["audio_end_frame"] == 120 and not details[0]["choked_at_next_sound"]
    assert details[1]["next_played_frame"] is None


def test_choke_toggle_short_fade_and_coincident_starts():
    settings = load_spec().beatbox
    signal = np.ones(100)
    audio, info = mix_placements(
        [(0, signal), (5, signal)], frames=200, sample_rate=1000, settings=settings
    )
    assert info[0]["choke_fade_frames"] == 5 and audio[4] == 0
    audio, info = mix_placements(
        [(0, signal), (0, signal)], frames=200, sample_rate=1000, settings=settings
    )
    assert not info[0]["played"] and audio.max() == 1
    audio, _ = mix_placements(
        [(0, signal), (5, signal)],
        frames=200,
        sample_rate=1000,
        settings=replace(settings, choke_at_next_sound=False),
    )
    assert audio.max() == 2


def test_sloppy_choke_ignores_dropped_middle_hit_and_uses_jitter(tmp_path):
    spec = load_spec()
    spec = replace(
        spec,
        sloppiness=replace(
            spec.sloppiness,
            offset_ms=(0, 0),
            jitter_ms=(20, 20),
            drop_prob={"kick": (0, 0), "snare": (1, 1), "hats": (0, 0)},
        ),
    )
    sf.write(tmp_path / "hit.wav", np.ones(600) * 0.1, 1000, subtype="FLOAT")
    trace = [
        {"voice": "kick", "frame": 100, "sample_path": "hit.wav"},
        {"voice": "snare", "frame": 200, "sample_path": "hit.wav"},
        {"voice": "hats", "frame": 400, "sample_path": "hit.wav"},
    ]
    _, params, actual = augment_query(
        trace,
        tmp_path,
        sample_rate=1000,
        duration_s=1,
        spec=spec,
        seed=5,
        normalization={"target_rms_dbfs": -20, "peak_ceiling_dbfs": -1},
    )
    assert actual[1]["dropped"] and not actual[1]["played"]
    assert actual[0]["audio_end_frame"] == actual[2]["frame"]
    assert actual[0]["audio_end_frame"] != actual[2]["clean_frame"]
    assert params["measured_snr_db_before_fade"] == pytest.approx(params["snr_db"])
