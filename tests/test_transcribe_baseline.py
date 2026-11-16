from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.music.rhythm import Grid, pattern_id
from src.music.spec import load_spec

from src.eval.evaluate import EvalError, evaluate
from src.eval.hit_classifier import (
    NOISE_SNRS_DB,
    HitClassifier,
    add_white_noise,
    hit_features,
)
from src.eval.onset_baseline import ANALYSIS_RATE
from src.eval.pool import Pool
from src.eval.transcribe_baseline import (
    Settings,
    detect_onsets,
    dilate,
    match_onsets,
    quantize,
    score_queries,
    tolerant_f1,
    tune,
)

sf = pytest.importorskip("soundfile")
pytest.importorskip("scipy")
pretty_midi = pytest.importorskip("pretty_midi")

RATE = ANALYSIS_RATE
VOICES = ("kick", "snare", "hats")
TEMPO = 100.0
LEAD_S = 0.25


def _raw_sound(kind, rng, rate):
    n = int(0.15 * rate)
    t = np.arange(n) / rate
    if kind == "kick":
        freq = rng.uniform(50, 90)
        return np.sin(2 * np.pi * freq * t * (1 + 1.5 * np.exp(-t / 0.02))) * np.exp(
            -t / rng.uniform(0.05, 0.09)
        )
    if kind == "snare":
        noise = rng.normal(0, 1, n) * np.exp(-t / rng.uniform(0.04, 0.07))
        return noise + 0.6 * np.sin(2 * np.pi * rng.uniform(180, 260) * t) * np.exp(-t / 0.05)
    return np.diff(rng.normal(0, 1, n + 1)) * np.exp(-t / rng.uniform(0.015, 0.03))


def make_sound(kind, rng, rate=RATE):
    ## every sound ends with a 10 ms fade, as the query generator's sounds do
    sound = _raw_sound(kind, rng, rate)
    fade = int(0.01 * rate)
    sound[-fade:] *= np.linspace(1.0, 0.0, fade)
    return sound


def place(wave, sound, onset_s, rate, amplitude=1.0):
    start = round(onset_s * rate)
    end = min(start + len(sound), len(wave))
    if 0 <= start < len(wave):
        wave[start:end] += sound[: end - start] * amplitude


@pytest.fixture(scope="module")
def spec():
    return load_spec()


@pytest.fixture(scope="module")
def classifier():
    ## trained on clean and noisy copies, as the real classifier is
    rng = np.random.default_rng(0)
    features, labels = [], []
    for row, kind in enumerate(VOICES):
        for _ in range(150):
            wave = np.zeros(RATE, dtype=np.float32)
            place(wave, make_sound(kind, rng, RATE), 0.3, RATE, rng.uniform(0.3, 1.0))
            for snr in NOISE_SNRS_DB:
                noisy = wave if snr is None else add_white_noise(wave, snr, rng)
                features.append(hit_features(noisy, RATE, 0.3))
                labels.append(row)
    return HitClassifier.fit(np.stack(features), np.array(labels), VOICES)


def _train_sounds_wave(onsets, kinds, rng, rate=RATE, length_s=6.0):
    wave = np.zeros(int(length_s * rate))
    for onset, kind in zip(onsets, kinds, strict=True):
        place(wave, make_sound(kind, rng, rate), onset, rate, rng.uniform(0.5, 1.0))
    return wave


def test_detected_times_land_on_the_true_onsets_for_every_sound_type():
    rng = np.random.default_rng(1)
    onsets = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    kinds = ["kick", "snare", "hats", "kick", "snare", "hats"]
    wave = _train_sounds_wave(onsets, kinds, rng, length_s=4.0)
    found = detect_onsets(wave, Settings())
    assert len(found) == len(onsets)
    assert np.abs(found - np.array(onsets)).max() < 0.008


def test_hits_closer_than_the_minimum_gap_count_once():
    ## the analysis window is 46 ms long, so hits must be roughly 60 ms apart to be told apart
    rng = np.random.default_rng(2)
    wave = _train_sounds_wave([0.5, 0.58], ["snare", "snare"], rng, length_s=2.0)

    def near_the_pair(settings):
        found = detect_onsets(wave, settings)
        return int(((found > 0.45) & (found < 0.65)).sum())

    assert near_the_pair(Settings(min_gap_ms=150.0)) == 1
    assert near_the_pair(Settings(min_gap_ms=30.0)) == 2


def test_a_higher_threshold_keeps_fewer_hits():
    rng = np.random.default_rng(3)
    onsets = [0.4 + 0.3 * i for i in range(10)]
    kinds = ["kick", "hats"] * 5
    wave = _train_sounds_wave(onsets, kinds, rng, length_s=4.0) * 1.0
    for i, onset in enumerate(onsets):
        if i % 2:
            place(wave, make_sound("hats", rng, RATE), onset, RATE, 0.05)
    low = detect_onsets(wave, Settings(peak_fraction=0.05))
    high = detect_onsets(wave, Settings(peak_fraction=0.9))
    assert 1 <= len(high) <= len(low)


def test_silence_has_no_onsets():
    assert len(detect_onsets(np.zeros(RATE * 2, dtype=np.float32), Settings())) == 0


def test_quantizing_puts_hits_on_the_right_steps_and_voices():
    step_s = 60.0 / TEMPO / 12
    times = LEAD_S + step_s * np.array([0, 24, 48, 95])
    grid = quantize(times, np.array([0, 1, 2, 0]), LEAD_S, TEMPO, 3, 96, 12)
    assert np.flatnonzero(grid[0]).tolist() == [0, 95]
    assert np.flatnonzero(grid[1]).tolist() == [24]
    assert np.flatnonzero(grid[2]).tolist() == [48]


def test_an_early_downbeat_stays_the_downbeat_and_outside_hits_are_dropped():
    step_s = 60.0 / TEMPO / 12
    times = LEAD_S + step_s * np.array([-0.6, -3.0, 96.0, 40.2])
    grid = quantize(times, np.zeros(4, dtype=int), LEAD_S, TEMPO, 3, 96, 12)
    assert np.flatnonzero(grid[0]).tolist() == [0, 40]


def test_dilate_widens_each_hit_by_the_tolerance():
    grid = np.zeros((1, 1, 10), dtype=bool)
    grid[0, 0, 4] = True
    assert np.flatnonzero(dilate(grid, 1)[0, 0]).tolist() == [3, 4, 5]
    assert np.flatnonzero(dilate(grid, 2)[0, 0]).tolist() == [2, 3, 4, 5, 6]


def _pattern_grid(kick, snare, hats):
    grid = np.zeros((3, 96), dtype=bool)
    for row, steps in enumerate((kick, snare, hats)):
        grid[row, list(steps)] = True
    return grid


def test_tolerant_matching_forgives_one_step_but_not_a_wrong_voice():
    library = np.stack([_pattern_grid([0, 24], [12, 36], [0, 12, 24, 36])])
    same = np.stack([_pattern_grid([0, 24], [12, 36], [0, 12, 24, 36])])
    one_off = np.stack([_pattern_grid([1, 23], [13, 35], [1, 13, 25, 37])])
    two_off = np.stack([_pattern_grid([2, 26], [14, 38], [2, 14, 26, 38])])
    wrong_voice = np.stack([_pattern_grid([12, 36], [0, 24], [0, 12, 24, 36])])
    empty = np.zeros((1, 3, 96), dtype=bool)
    assert tolerant_f1(same, library)[0, 0] == 1.0
    assert tolerant_f1(one_off, library)[0, 0] == 1.0
    assert tolerant_f1(two_off, library)[0, 0] < 0.5
    assert tolerant_f1(wrong_voice, library)[0, 0] < tolerant_f1(same, library)[0, 0]
    assert tolerant_f1(empty, library)[0, 0] == 0.0


def test_a_query_with_extra_or_missing_hits_scores_between_zero_and_one():
    library = np.stack([_pattern_grid([0, 24], [12, 36], [0, 12, 24, 36])])
    missing_hats = np.stack([_pattern_grid([0, 24], [12, 36], [0, 24])])
    score = tolerant_f1(missing_hats, library)[0, 0]
    assert 0.5 < score < 1.0


def test_matching_is_one_to_one_and_closest_first():
    annotated = np.array([1.00, 1.03])
    detected = np.array([1.02])
    assert match_onsets(detected, annotated, 0.05) == [(1, 0)]
    assert match_onsets(np.array([]), annotated, 0.05) == []
    assert match_onsets(np.array([2.0]), annotated, 0.05) == []


def test_tuning_finds_a_working_detector_and_corrects_a_timing_bias():
    rng = np.random.default_rng(4)
    recordings = []
    for _ in range(3):
        onsets = np.sort(rng.choice(np.arange(0.3, 8.0, 0.18), 20, replace=False))
        kinds = rng.choice(list(VOICES), 20)
        wave = _train_sounds_wave(onsets, kinds, rng, length_s=9.0)
        ## the annotations sit 10 ms late, as a hand-annotated dataset might
        recordings.append((wave, onsets + 0.010, np.array([VOICES.index(k) for k in kinds])))
    settings, report = tune(recordings)
    assert report["f1_within_50ms_before_delay_fix"] > 0.9
    assert report["f1_within_30ms_after_delay_fix"] > 0.9
    assert abs(report["median_error_ms_after_delay_fix"]) < 4.0
    assert settings.delay_ms == pytest.approx(26.0, abs=5.0)


def _write_midi(path, pattern):
    midi = pretty_midi.PrettyMIDI(initial_tempo=TEMPO, resolution=480)
    instrument = pretty_midi.Instrument(program=0, is_drum=True)
    for pitch, positions in zip((36, 38, 42), pattern, strict=True):
        for position in positions:
            start = position * 0.25 * 60.0 / TEMPO
            instrument.notes.append(pretty_midi.Note(90, pitch, start, start + 0.05))
    midi.instruments.append(instrument)
    midi.write(str(path))


def _random_pattern(rng):
    positions = rng.permutation(32)
    return sorted(positions[:5]), sorted(positions[5:9]), sorted(positions[9:19])


@pytest.fixture
def dataset(tmp_path, spec):
    ## 8 random loops, each as MIDI (the library side) and as played audio (the query side),
    ## with timing jitter, light noise, and a library on two kits that share one grid
    rng = np.random.default_rng(7)
    library_rows, query_rows, relevance = [], [], {}
    duration = LEAD_S + 8 * 60.0 / TEMPO
    for p in range(8):
        pattern = _random_pattern(rng)
        midi = tmp_path / f"p{p}.mid"
        _write_midi(midi, pattern)
        onsets = np.zeros((3, 96), dtype=bool)
        for row, positions in enumerate(pattern):
            onsets[row, [3 * q for q in positions]] = True
        pid = pattern_id(Grid(spec.voices, 12, onsets))
        clips = []
        for kit in ("salamander", "muldjordkit"):
            clip = f"p{p}_{kit}"
            clips.append(clip)
            library_rows.append(
                (clip, f"src{p}", str(midi), "0", pid, str(tmp_path / "unused.wav"))
            )
        for mode in ("Personal", "Fixed"):
            wave = np.zeros(int(duration * 44100))
            for kind, positions in zip(VOICES, pattern, strict=True):
                for q in positions:
                    t = LEAD_S + q * 0.25 * 60.0 / TEMPO + rng.normal(0, 0.008)
                    place(wave, make_sound(kind, rng, 44100), t, 44100, rng.uniform(0.6, 1.0))
            wave += rng.normal(0, 0.01 * np.abs(wave).max(), len(wave))
            path = tmp_path / f"q{p}_{mode}.wav"
            sf.write(path, (wave / np.abs(wave).max() * 0.8).astype(np.float32), 44100)
            query_id = f"q{p}_{mode}"
            query_rows.append((query_id, str(path), TEMPO, duration))
            relevance[query_id] = clips
    library = pd.DataFrame(
        library_rows,
        columns=["clip_id", "source_clip_id", "midi_path", "start_bar", "pattern_id", "audio_path"],
    )
    queries = pd.DataFrame(
        query_rows, columns=["query_id", "audio_path", "tempo_bpm", "duration_s"]
    )
    pool = Pool(
        library=library[["clip_id", "audio_path"]].sort_values("clip_id").reset_index(drop=True),
        queries=queries.sort_values("query_id").reset_index(drop=True),
    )
    return pool, library, relevance, tmp_path


def test_the_whole_chain_retrieves_the_right_loop(dataset, classifier, spec):
    pool, library, relevance, tmp_path = dataset
    scores, stats = score_queries(pool, library, spec, classifier, Settings())
    assert scores.shape == (16, 16)
    assert stats["library_grids_not_matching_manifest"] == 0
    assert stats["queries_with_no_hits"] == 0
    ## the same loop on two kits gets the same score, and must still rank first
    path = tmp_path / "scores.npz"
    np.savez(
        path,
        query_ids=pool.queries["query_id"].to_numpy(dtype=str),
        library_ids=pool.library["clip_id"].to_numpy(dtype=str),
        scores=scores,
    )
    table, _ = evaluate(path, relevance)
    assert table["rr"].mean() > 0.9
    assert table["hit@1"].mean() > 0.85


def test_a_library_grid_that_disagrees_with_the_manifest_stops_the_run(dataset, classifier, spec):
    pool, library, _, _ = dataset
    library.loc[library["source_clip_id"] == "src0", "pattern_id"] = "0000000000000000"
    with pytest.raises(EvalError, match="differs from the manifest"):
        score_queries(pool, library, spec, classifier, Settings())
    scores, stats = score_queries(pool, library, spec, classifier, Settings(), allow_mismatch=True)
    assert stats["library_grids_not_matching_manifest"] == 1
    assert scores.shape == (16, 16)
