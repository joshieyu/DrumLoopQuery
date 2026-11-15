from __future__ import annotations

import pandas as pd
import pytest
from src.music.spec import load_spec

from src.eval.evaluate import EvalError
from src.eval.ttd_prep import (
    DEFAULT_ROOT,
    funnel,
    index_dataset,
    match_sources,
    onset_f1,
    parse_source_folder,
    query_rows,
)

pretty_midi = pytest.importorskip("pretty_midi")

BPM = 100.0
BEAT_S = 60.0 / BPM


def _write_midi(path, notes, drum):
    midi = pretty_midi.PrettyMIDI(initial_tempo=BPM, resolution=480)
    instrument = pretty_midi.Instrument(program=0, is_drum=drum)
    for pitch, velocity, beat in notes:
        start = beat * BEAT_S
        instrument.notes.append(pretty_midi.Note(velocity, pitch, start, start + 0.05))
    midi.instruments.append(instrument)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(path))


def _make_source(root, name, kick_beats, snare_beats, tap_offset_beats=0.0):
    ## kick 36, snare 38, closed hat 42 on every 8th note; the taps use the two hand pitches
    folder = root / "Repetitions" / "tested_with_four_participants" / name
    hats = [(42, 80, b / 2) for b in range(16)]
    original = [(36, 90, b) for b in kick_beats] + [(38, 90, b) for b in snare_beats] + hats
    _write_midi(folder / "original.mid", original, drum=True)
    left = [(60, 90, b + tap_offset_beats) for b in kick_beats]
    right = [(74, 80, b + tap_offset_beats) for b in snare_beats]
    for participant in (1, 2):
        _write_midi(
            folder / f"Participant_{participant}_repetition_0.mid", left + right, drum=False
        )
        _write_midi(folder / f"Participant_{participant}_simple.mid", left + right, drum=False)
    return folder


@pytest.fixture
def spec():
    return load_spec()


@pytest.fixture
def mini(tmp_path):
    root = tmp_path / "midi_files"
    _make_source(
        root,
        "[001 P1P2P3P4] drummer5-eval-session-7_pop-groove7_100_beat_4-4_best_2bar_segment_4",
        kick_beats=[0, 2, 4, 6],
        snare_beats=[1, 3, 5, 7],
    )
    _make_source(
        root,
        "[002 P1] drummer8-session2-34_latin-brazilian-songo_100_beat_4-4_best_2bar_segment_9",
        kick_beats=[0, 1.5, 4, 5.5],
        snare_beats=[2, 3, 6, 7],
    )
    return root


def test_folder_names_parse_into_library_style_ids():
    eval_case = parse_source_folder(
        "[038 P1] drummer5-eval-session-7_pop-groove7_138_beat_4-4_best_2bar_segment_6"
    )
    assert eval_case["gmd_performance"] == "drummer5/eval_session/7"
    assert eval_case["is_eval_session"]
    assert (eval_case["bpm"], eval_case["segment"], eval_case["number"]) == (138, 6, "038")
    plain = parse_source_folder(
        "[001 P1P2P3P4] drummer10-session1-9_jazz-swing_110_beat_4-4_best_2bar_segment_15"
    )
    assert plain["gmd_performance"] == "drummer10/session1/9"
    assert not plain["is_eval_session"]
    assert plain["style"] == "jazz-swing"


def test_unrecognised_folder_name_is_an_error():
    with pytest.raises(EvalError, match="unrecognised"):
        parse_source_folder("some other folder")


def test_index_reads_sources_and_taps(mini, spec):
    sources, taps = index_dataset(mini, spec)
    assert len(sources) == 2
    assert len(taps) == 8
    assert sources["meets_min_hits"].all()
    assert sources["pattern_id"].nunique() == 2
    assert taps["tap_id"].is_unique
    assert set(taps["tap_kind"]) == {"repetition_0", "simple"}
    assert taps["tempo_matches_source"].all()
    assert (taps["n_left"] == 4).all()
    assert (taps["n_right"] == 4).all()


def test_a_tap_that_keeps_the_rhythm_scores_high_and_a_shifted_one_low(mini, tmp_path, spec):
    _, taps = index_dataset(mini, spec)
    assert (taps["skeleton_f1"] > 0.5).all()
    ## a quarter beat late falls between every source onset, which sit on the 8th notes
    shifted = tmp_path / "shifted"
    _make_source(
        shifted,
        "[003 P1] drummer1-session1-1_rock_100_beat_4-4_best_2bar_segment_0",
        kick_beats=[0, 2, 4, 6],
        snare_beats=[1, 3, 5, 7],
        tap_offset_beats=0.25,
    )
    _, late = index_dataset(shifted, spec)
    assert (late["skeleton_f1"] < 0.5).all()


def test_onset_f1_tolerance():
    assert onset_f1([0, 12, 24], [0, 12, 24], 0) == 1.0
    assert onset_f1([1, 12, 24], [0, 12, 24], 0) < 1.0
    assert onset_f1([1, 13, 25], [0, 12, 24], 1) == 1.0
    assert onset_f1([], [0], 1) == 0.0


def test_tap_with_a_pitch_outside_both_hands_is_an_error(tmp_path, spec):
    folder = _make_source(
        tmp_path / "bad",
        "[001 P1] drummer1-session1-1_rock_100_beat_4-4_best_2bar_segment_0",
        kick_beats=[0, 2, 4, 6],
        snare_beats=[1, 3, 5, 7],
    )
    _write_midi(folder / "Participant_3_simple.mid", [(99, 80, 0)], drum=False)
    with pytest.raises(EvalError, match="not in either hand"):
        index_dataset(tmp_path / "bad", spec)


def _library_for(sources, kits=("salamander", "muldjordkit")):
    ## a test library holding only the first source's loop, on both test kits
    first = sources.iloc[0]
    rows = [
        (
            f"clip_{kit}",
            first["pattern_id"],
            "test",
            kit,
            first["gmd_performance"],
            first["segment"],
        )
        for kit in kits
    ]
    rows.append(("train_clip", "other", "train", "drskit", "drummer1/session1/1", 0))
    return pd.DataFrame(
        rows,
        columns=["clip_id", "pattern_id", "split", "kit_id", "source_performance", "start_bar"],
    ).astype({"start_bar": str})


def test_matching_funnel_and_query_rows(mini, spec):
    sources, taps = index_dataset(mini, spec)
    matched = match_sources(sources, _library_for(sources), pool_kits=["salamander", "muldjordkit"])
    first, second = matched.iloc[0], matched.iloc[1]
    assert first["performance_in_pool"]
    assert first["segment_is_a_pool_start_bar"]
    assert first["n_pool_clips"] == 2
    assert not second["pattern_in_pool"]
    queries = query_rows(matched, taps)
    assert len(queries) == 4  # the first source's 4 taps; the second source is not in the pool
    assert queries["query_id"].is_unique
    assert set(queries["relevance_key"]) == {"pattern_id"}
    assert set(queries["pattern_id"]) == {first["pattern_id"]}
    assert queries["dualized_pattern_id"].isna().all()
    assert queries["participant_id"].str.startswith("ttd_P").all()
    report = funnel(matched, taps, queries)
    assert report["sources_total"] == 2
    assert report["sources_usable"] == 1
    assert report["tap_queries"] == 4


def test_matching_with_no_pool_clips_is_an_error(mini, spec):
    sources, _ = index_dataset(mini, spec)
    with pytest.raises(EvalError, match="no clips"):
        match_sources(sources, _library_for(sources), pool_split="validation")


def test_no_source_matches_when_the_library_has_other_patterns(mini, spec):
    sources, taps = index_dataset(mini, spec)
    library = _library_for(sources).assign(pattern_id="not_a_real_pattern")
    matched = match_sources(sources, library)
    queries = query_rows(matched, taps)
    assert queries.empty
    assert funnel(matched, taps, queries)["tap_queries"] == 0


@pytest.mark.skipif(not DEFAULT_ROOT.exists(), reason="TapTamDrum is not downloaded")
def test_real_dataset_has_the_published_counts(spec):
    sources, taps = index_dataset(DEFAULT_ROOT, spec)
    assert len(sources) == 345
    assert len(taps) == 1116
    assert int(sources["is_eval_session"].sum()) == 28
    assert taps["tap_id"].is_unique
    assert taps["tempo_matches_source"].all()
