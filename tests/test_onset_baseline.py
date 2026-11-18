from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from src.music.spec import load_spec

from src.eval.evaluate import EvalError, evaluate, load_relevance
from src.eval.onset_baseline import (
    ANALYSIS_RATE,
    N_BINS,
    build_scores_file,
    lagged_scores,
    lead_in_seconds,
    loop_beats,
    loop_envelope,
    onset_bands,
)
from src.eval.pool import load_pool

sf = pytest.importorskip("soundfile")
pytest.importorskip("scipy")

SR = 44_100
LEAD_S = 0.25
BURST = int(0.08 * SR)


def _bursts(kick_hz, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(BURST) / SR
    return {
        "kick": np.sin(2 * np.pi * kick_hz * t) * np.exp(-t / 0.03),
        "snare": rng.normal(0, 1, BURST) * np.exp(-t / 0.04),
        "hats": np.diff(rng.normal(0, 1, BURST + 1)) * np.exp(-t / 0.012),
    }


def _pattern(seed):
    rng = np.random.default_rng(seed)
    sixteenths = np.arange(32)
    return {
        "kick": 3 * rng.choice(sixteenths, 6, replace=False),
        "snare": 3 * rng.choice(sixteenths, 4, replace=False),
        "hats": 3 * rng.choice(sixteenths, 12, replace=False),
    }


def _render(pattern, tempo, bursts, rng, jitter_ms=0.0, snr_db=None):
    loop_s = 8 * 60 / tempo
    wave = np.zeros(int((LEAD_S + loop_s) * SR))
    step_s = 60 / tempo / 12
    for voice, steps in pattern.items():
        for step in steps:
            start = int((LEAD_S + step * step_s + rng.normal(0, jitter_ms / 1000)) * SR)
            end = min(start + BURST, len(wave))
            if start >= 0:
                wave[start:end] += bursts[voice][: end - start]
    if snr_db is not None:
        power = np.mean(wave**2)
        wave = wave + rng.normal(0, np.sqrt(power / 10 ** (snr_db / 10)), len(wave))
    return (wave / np.abs(wave).max() * 0.8).astype(np.float32)


@pytest.fixture
def dataset(tmp_path):
    ## 10 patterns at two tempos; the library has two different drum timbres, the queries a
    ## third, with timing jitter and noise
    rng = np.random.default_rng(0)
    kits = {"kit_a": _bursts(55, 1), "kit_b": _bursts(65, 2)}
    query_bursts = _bursts(85, 3)
    library_rows, query_rows, relevance = [], [], {}
    for p in range(10):
        tempo = 100.0 if p % 2 == 0 else 120.0
        pattern = _pattern(100 + p)
        duration = LEAD_S + 8 * 60 / tempo
        clips = []
        for kit, bursts in kits.items():
            path = tmp_path / "library" / f"p{p}_{kit}.wav"
            path.parent.mkdir(exist_ok=True)
            sf.write(path, _render(pattern, tempo, bursts, rng), SR)
            clips.append(f"p{p}_{kit}")
            library_rows.append((f"p{p}_{kit}", "test", kit, str(path), tempo, duration))
        for mode in ("Personal", "Fixed"):
            path = tmp_path / "queries" / f"q{p}_{mode}.wav"
            path.parent.mkdir(exist_ok=True)
            wave = _render(pattern, tempo, query_bursts, rng, jitter_ms=12, snr_db=25)
            sf.write(path, wave, SR)
            query_rows.append((f"q{p}_{mode}", str(path), tempo, duration))
            relevance[f"q{p}_{mode}"] = clips
    pd.DataFrame(
        library_rows,
        columns=["clip_id", "split", "kit_id", "audio_path", "tempo_bpm", "duration_s"],
    ).to_csv(tmp_path / "library.csv", index=False)
    pd.DataFrame(query_rows, columns=["query_id", "audio_path", "tempo_bpm", "duration_s"]).to_csv(
        tmp_path / "queries.csv", index=False
    )
    (tmp_path / "relevance.json").write_text(json.dumps(relevance))
    return tmp_path


def test_loop_is_eight_beats_in_the_current_spec():
    assert loop_beats(load_spec()) == 8.0


def test_lead_in_comes_from_the_clip_length_and_tempo():
    assert lead_in_seconds(3.728253968253968, 138.0, 8.0) == pytest.approx(0.25, abs=1e-3)
    assert lead_in_seconds(4.79, 100.0, 8.0) == 0.0  # a little short: clamped
    with pytest.raises(EvalError, match="does not fit"):
        lead_in_seconds(2.0, 100.0, 8.0)  # far too short for 8 beats
    with pytest.raises(EvalError, match="does not fit"):
        lead_in_seconds(10.0, 100.0, 8.0)  # far too much lead-in


def test_onset_curve_is_nonzero_only_near_the_onsets():
    onsets_s = [0.5, 1.0, 1.75]
    wave = np.zeros(ANALYSIS_RATE * 3, dtype=np.float32)
    rng = np.random.default_rng(0)
    for t in onsets_s:
        start = int(t * ANALYSIS_RATE)
        wave[start : start + 1000] = rng.normal(0, 0.5, 1000) * np.exp(-np.arange(1000) / 300)
    times, flux = onset_bands(wave, ANALYSIS_RATE, 1)
    active = times[flux[0] > 0]
    assert len(active) > 0
    ## a noise burst's tail keeps the curve slightly active for about 90 ms after the hit
    assert all(min(abs(t - o) for o in onsets_s) < 0.15 for t in active)
    assert all(any(abs(t - o) < 0.06 for t in active) for o in onsets_s)


def test_three_bands_separate_a_low_hit_from_a_high_hit():
    t = np.arange(2000) / ANALYSIS_RATE
    wave = np.zeros(ANALYSIS_RATE * 2, dtype=np.float32)
    wave[ANALYSIS_RATE // 2 : ANALYSIS_RATE // 2 + 2000] = np.sin(2 * np.pi * 80 * t) * np.exp(
        -t / 0.05
    )
    start = int(1.5 * ANALYSIS_RATE)
    wave[start : start + 2000] = np.sin(2 * np.pi * 6000 * t) * np.exp(-t / 0.05)
    times, flux = onset_bands(wave, ANALYSIS_RATE, 3)
    low_time = times[np.argmax(flux[0])]
    high_time = times[np.argmax(flux[2])]
    assert abs(low_time - 0.5) < 0.06
    assert abs(high_time - 1.5) < 0.06


def test_silent_audio_gives_a_zero_envelope_not_nan():
    times, flux = onset_bands(np.zeros(ANALYSIS_RATE, dtype=np.float32), ANALYSIS_RATE, 3)
    envelope = loop_envelope(times, flux, 0.0, 0.9)
    assert envelope.shape == (3, N_BINS)
    assert np.isfinite(envelope).all()
    assert not envelope.any()


def test_envelope_is_unit_length_per_band():
    rng = np.random.default_rng(0)
    times = np.linspace(0, 3, 500)
    flux = np.abs(rng.normal(size=(3, 500)))
    envelope = loop_envelope(times, flux, 0.25, 2.5)
    assert np.allclose(np.linalg.norm(envelope, axis=1), 1.0)


def test_small_shifts_are_tolerated_and_large_ones_are_not():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(1, 1, N_BINS)).astype(np.float32)
    base -= base.mean()
    base /= np.linalg.norm(base)
    near = np.roll(base, 5, axis=2)
    far = np.roll(base, 100, axis=2)
    assert lagged_scores(base, near)[0, 0] > 0.95
    assert lagged_scores(base, far)[0, 0] < 0.5


def test_pool_timing_columns_are_required_and_numeric(dataset):
    pool = load_pool(
        dataset / "library.csv", dataset / "queries.csv", dataset / "relevance.json", timing=True
    )
    assert pool.library["tempo_bpm"].dtype == float
    assert pool.queries["duration_s"].gt(0).all()
    frame = pd.read_csv(dataset / "queries.csv").drop(columns=["duration_s"])
    frame.to_csv(dataset / "queries.csv", index=False)
    with pytest.raises(EvalError, match="duration_s"):
        load_pool(
            dataset / "library.csv",
            dataset / "queries.csv",
            dataset / "relevance.json",
            timing=True,
        )


@pytest.mark.parametrize("n_bands", [1, 3])
def test_retrieves_the_right_loop_across_timbres_jitter_and_noise(dataset, n_bands):
    pool = load_pool(
        dataset / "library.csv", dataset / "queries.csv", dataset / "relevance.json", timing=True
    )
    out = build_scores_file(pool, load_spec(), n_bands, dataset / f"scores{n_bands}.npz")
    table, counts = evaluate(out, load_relevance(dataset / "relevance.json"))
    assert counts["n_queries_scored"] == 20
    assert table["rr"].mean() > 0.9
    assert table["hit@1"].mean() > 0.85


def test_a_scrambled_answer_key_scores_near_chance(dataset):
    pool = load_pool(
        dataset / "library.csv", dataset / "queries.csv", dataset / "relevance.json", timing=True
    )
    out = build_scores_file(pool, load_spec(), 1, dataset / "scores.npz")
    relevance = load_relevance(dataset / "relevance.json")
    keys = sorted(relevance)
    scrambled = {k: relevance[keys[(i + 7) % len(keys)]] for i, k in enumerate(keys)}
    table, _ = evaluate(out, scrambled)
    assert table["rr"].mean() < 0.5


def test_missing_audio_stops_before_any_analysis(dataset):
    pool = load_pool(
        dataset / "library.csv", dataset / "queries.csv", dataset / "relevance.json", timing=True
    )
    (dataset / "queries" / "q0_Fixed.wav").unlink()
    with pytest.raises(EvalError, match="audio files are missing"):
        build_scores_file(pool, load_spec(), 1, dataset / "s.npz")
