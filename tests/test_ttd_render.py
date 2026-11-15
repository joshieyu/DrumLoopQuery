from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from src.music.spec import load_spec

from src.eval.evaluate import EvalError
from src.eval.pool import load_pool
from src.eval.relevance import build_relevance
from src.eval.ttd_render import (
    library_lead_in,
    pick_low_hand,
    placeholder_knocks,
    query_sample_rate,
    render_all,
    render_tap,
)

sf = pytest.importorskip("soundfile")
pretty_midi = pytest.importorskip("pretty_midi")

SR = 44_100
LEFT, RIGHT = 60, 74
ONES = np.ones(100)
KNOCKS = (ONES, 0.5 * ONES)  # low is loud, high is half as loud, so they can be told apart


def _tap(notes, low_hand="left", tempo=120.0, lead=0.25, loop_s=4.0):
    return render_tap(notes, tempo, lead, loop_s, SR, low_hand, KNOCKS)


def test_placeholder_knocks_are_deterministic_and_the_low_one_is_lower():
    low, high = placeholder_knocks(SR)
    again_low, again_high = placeholder_knocks(SR)
    assert np.array_equal(low, again_low)
    assert np.array_equal(high, again_high)
    assert np.abs(low).max() == pytest.approx(1.0)
    assert np.abs(high).max() == pytest.approx(1.0)

    def centroid(wave):
        spectrum = np.abs(np.fft.rfft(wave))
        return (np.fft.rfftfreq(len(wave), 1 / SR) * spectrum).sum() / spectrum.sum()

    assert centroid(low) < centroid(high)


def test_the_sparser_hand_gets_the_low_knock():
    left_sparse = [(LEFT, 90, 0.0), (RIGHT, 90, 1.0), (RIGHT, 90, 2.0)]
    right_sparse = [(RIGHT, 90, 0.0), (LEFT, 90, 1.0), (LEFT, 90, 2.0)]
    tie = [(LEFT, 90, 0.0), (RIGHT, 90, 1.0)]
    assert pick_low_hand(left_sparse, "density") == "left"
    assert pick_low_hand(right_sparse, "density") == "right"
    assert pick_low_hand(tie, "density") == "left"
    assert pick_low_hand(right_sparse, "left-low") == "left"
    assert pick_low_hand(left_sparse, "right-low") == "right"
    with pytest.raises(EvalError, match="hand rule"):
        pick_low_hand(tie, "random")


def test_a_tap_lands_at_the_right_time_and_the_clip_has_the_loop_length():
    ## beat 2 at 120 BPM is 1 s into the loop, so 1.25 s into the clip with a 0.25 s lead-in
    wave = _tap([(LEFT, 127, 2.0), (RIGHT, 127, 4.0)])
    assert len(wave) == round((0.25 + 4.0) * SR)
    assert np.flatnonzero(wave)[0] == round(1.25 * SR)
    assert np.abs(wave).max() == pytest.approx(0.8, abs=1e-6)


def test_velocity_and_hand_change_the_loudness_of_a_knock():
    wave = _tap([(LEFT, 127, 0.0), (LEFT, 64, 2.0), (RIGHT, 127, 4.0)])
    loud = wave[round(0.25 * SR) + 10]
    half_velocity = wave[round(1.25 * SR) + 10]
    high_knock = wave[round(2.25 * SR) + 10]
    assert half_velocity / loud == pytest.approx(64 / 127, rel=0.01)
    assert high_knock / loud == pytest.approx(0.5, rel=0.01)  # right hand is the high knock
    swapped = _tap([(LEFT, 127, 0.0), (RIGHT, 127, 4.0)], low_hand="right")
    assert swapped[round(0.25 * SR) + 10] == pytest.approx(0.5 * swapped[round(2.25 * SR) + 10])


def test_notes_outside_the_clip_are_ignored_and_an_empty_tap_is_an_error():
    wave = _tap([(LEFT, 127, 0.0), (LEFT, 127, 400.0), (LEFT, 127, -50.0)])
    assert np.count_nonzero(wave) == len(ONES)
    with pytest.raises(EvalError, match="no notes"):
        _tap([(LEFT, 127, 400.0)])


def _library(tmp_path, lead_ins=(0.25, 0.25), rates=(44100, 44100)):
    path = tmp_path / "library.csv"
    pd.DataFrame(
        {
            "clip_id": ["c1", "c2"],
            "split": ["test", "test"],
            "pattern_id": ["A", "A"],
            "audio_path": ["a.wav", "b.wav"],
            "tempo_bpm": [100.0, 100.0],
            "duration_s": [5.05, 5.05],
            "lead_in_s": lead_ins,
            "sample_rate": rates,
        }
    ).to_csv(path, index=False)
    return path


def test_lead_in_comes_from_the_library_even_when_its_clips_differ_in_sample_rate(tmp_path):
    assert library_lead_in(_library(tmp_path, rates=(44100, 48000))) == 0.25
    with pytest.raises(EvalError, match="different lead-ins"):
        library_lead_in(_library(tmp_path, lead_ins=(0.25, 0.5)))


def test_sample_rate_comes_from_the_query_manifest(tmp_path):
    path = tmp_path / "q.csv"
    pd.DataFrame({"query_id": ["a", "b"], "sample_rate": [44100, 44100]}).to_csv(path, index=False)
    assert query_sample_rate(path) == 44100
    pd.DataFrame({"query_id": ["a", "b"], "sample_rate": [44100, 48000]}).to_csv(path, index=False)
    with pytest.raises(EvalError, match="different sample rates"):
        query_sample_rate(path)


def _write_tap_midi(path, notes, tempo=100.0):
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo, resolution=480)
    instrument = pretty_midi.Instrument(program=0, is_drum=False)
    for pitch, velocity, beat in notes:
        start = beat * 60.0 / tempo
        instrument.notes.append(pretty_midi.Note(velocity, pitch, start, start + 0.05))
    midi.instruments.append(instrument)
    midi.write(str(path))


def test_render_all_writes_audio_and_a_manifest_the_baselines_can_load(tmp_path):
    midi_a = tmp_path / "a.mid"
    midi_b = tmp_path / "b.mid"
    _write_tap_midi(midi_a, [(LEFT, 90, 0), (RIGHT, 80, 1), (RIGHT, 80, 2), (RIGHT, 90, 4)])
    _write_tap_midi(midi_b, [(RIGHT, 90, 0), (LEFT, 80, 1), (LEFT, 80, 2), (LEFT, 90, 4)])
    manifest = pd.DataFrame(
        {
            "query_id": ["ttd1_P1_simple", "ttd1_P2_simple"],
            "pattern_id": ["A", "A"],
            "dualized_pattern_id": [None, None],
            "query_type": "tap",
            "source": "taptamdrum",
            "split": "test",
            "tempo_bpm": [100.0, 100.0],
            "relevance_key": "pattern_id",
            "midi_path": [str(midi_a), str(midi_b)],
            "audio_path": [None, None],
        }
    )
    out_dir = tmp_path / "audio"
    rendered = render_all(manifest, load_spec(), 0.25, SR, out_dir, KNOCKS)

    assert sorted(p.name for p in out_dir.iterdir()) == [
        "ttd1_P1_simple.wav",
        "ttd1_P2_simple.wav",
    ]
    ## 2 bars of 4/4 at 100 BPM is 4.8 s, plus the lead-in
    assert rendered["duration_s"].tolist() == pytest.approx([5.05, 5.05], abs=1e-4)
    ## each tap gives the low knock to its own sparser hand
    assert rendered["low_hand"].tolist() == ["left", "right"]
    data, rate = sf.read(out_dir / "ttd1_P1_simple.wav")
    assert rate == SR
    assert len(data) / SR == pytest.approx(5.05, abs=1e-4)

    ## the rendered manifest drives the answer key and the pool loader
    library = pd.read_csv(_library(tmp_path), dtype=str)
    library["kit_id"] = ["salamander", "muldjordkit"]
    library["dualized_pattern_id"] = "T"
    relevance, _ = build_relevance(library, rendered)
    assert relevance == {"ttd1_P1_simple": ["c1", "c2"], "ttd1_P2_simple": ["c1", "c2"]}
    (tmp_path / "relevance.json").write_text(json.dumps(relevance))
    rendered.to_csv(tmp_path / "ttd.csv", index=False)
    pool = load_pool(
        tmp_path / "library.csv", tmp_path / "ttd.csv", tmp_path / "relevance.json", timing=True
    )
    assert len(pool.queries) == 2
    assert pool.queries["duration_s"].gt(0).all()
