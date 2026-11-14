from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.eval.audio_io import load_mono, missing_files
from src.eval.clap_baseline import build_system_file, embed_paths
from src.eval.evaluate import EvalError, evaluate, load_relevance
from src.eval.pool import load_pool

sf = pytest.importorskip("soundfile")
pytest.importorskip("scipy")

SOURCE_RATE = 44_100
MODEL_RATE = 48_000


def _write_tone(path, freq, seconds=0.5, channels=1, seed=0):
    t = np.arange(int(SOURCE_RATE * seconds)) / SOURCE_RATE
    wave = np.sin(2 * np.pi * freq * t) + np.random.default_rng(seed).normal(0, 0.01, len(t))
    data = np.stack([wave] * channels, axis=1) if channels > 1 else wave
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, data.astype(np.float32), SOURCE_RATE)


def tone_embedder(waves):
    ## coarse spectrum of the first 0.1 s: 20 bands of 50 Hz, so each tone lands in one band
    out = []
    for wave in waves:
        magnitude = np.abs(np.fft.rfft(wave[:4800]))
        out.append([magnitude[i : i + 5].sum() for i in range(0, 100, 5)])
    return np.array(out)


@pytest.fixture
def dataset(tmp_path):
    ## 6 patterns, each a different tone, rendered on 2 kits and queried in 2 modes
    library_rows, query_rows, relevance = [], [], {}
    for p in range(6):
        freq = 225 + 100 * p
        clips = []
        for kit in ("salamander", "muldjordkit"):
            audio = tmp_path / "audio" / "library" / kit / f"p{p}.wav"
            _write_tone(audio, freq, seed=p * 10 + len(clips))
            clip_id = f"p{p}_{kit}"
            clips.append(clip_id)
            library_rows.append((clip_id, "test", kit, str(audio)))
        for mode in ("Personal", "Fixed"):
            audio = tmp_path / "audio" / "queries" / f"p{p}_{mode}.wav"
            _write_tone(audio, freq, seed=100 + p)
            query_id = f"q{p}_{mode}"
            query_rows.append((query_id, str(audio)))
            relevance[query_id] = clips
    library_rows.append(("train_clip", "train", "drskit", str(tmp_path / "unused.wav")))
    pd.DataFrame(library_rows, columns=["clip_id", "split", "kit_id", "audio_path"]).to_csv(
        tmp_path / "library.csv", index=False
    )
    pd.DataFrame(query_rows, columns=["query_id", "audio_path"]).to_csv(
        tmp_path / "queries.csv", index=False
    )
    (tmp_path / "relevance.json").write_text(json.dumps(relevance))
    return tmp_path


def test_load_mono_resamples_and_mixes_down(tmp_path):
    _write_tone(tmp_path / "a.wav", 440, seconds=1.0, channels=2)
    wave = load_mono(tmp_path / "a.wav", MODEL_RATE)
    assert wave.dtype == np.float32
    assert wave.ndim == 1
    assert len(wave) == MODEL_RATE
    assert np.isfinite(wave).all()


def test_missing_audio_is_reported(tmp_path):
    assert missing_files([tmp_path / "nope.wav"]) == [str(tmp_path / "nope.wav")]
    with pytest.raises(EvalError, match="not found"):
        load_mono(tmp_path / "nope.wav", MODEL_RATE)


def test_pool_has_only_test_clips_and_answer_key_queries_in_sorted_order(dataset):
    pool = load_pool(dataset / "library.csv", dataset / "queries.csv", dataset / "relevance.json")
    assert len(pool.library) == 12
    assert "train_clip" not in set(pool.library["clip_id"])
    assert pool.library["clip_id"].tolist() == sorted(pool.library["clip_id"])
    assert pool.queries["query_id"].tolist() == sorted(pool.queries["query_id"])


def test_pool_rejects_a_key_that_names_clips_outside_the_pool(dataset):
    with pytest.raises(EvalError, match="not in the pool"):
        load_pool(
            dataset / "library.csv",
            dataset / "queries.csv",
            dataset / "relevance.json",
            pool_kits=["salamander"],
        )


def test_pool_rejects_key_queries_missing_from_the_manifest(dataset):
    relevance = load_relevance(dataset / "relevance.json")
    relevance["ghost"] = ["p0_salamander"]
    (dataset / "relevance.json").write_text(json.dumps(relevance))
    with pytest.raises(EvalError, match="not in the query manifest"):
        load_pool(dataset / "library.csv", dataset / "queries.csv", dataset / "relevance.json")


def test_embed_paths_keeps_order_across_batches_and_normalizes(tmp_path):
    paths = []
    for i in range(10):
        data = np.full(SOURCE_RATE // 2, (i + 1) / 11, dtype=np.float32)
        sf.write(tmp_path / f"dc{i}.wav", data, SOURCE_RATE)
        paths.append(str(tmp_path / f"dc{i}.wav"))

    def embedder(waves):
        return np.array([[np.median(w), 1.0] for w in waves])

    emb = embed_paths(paths, embedder, MODEL_RATE, batch_size=4)
    assert emb.shape == (10, 2)
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-5)
    ## the ratio of the two components recovers each file's level, in file order
    assert np.allclose(emb[:, 0] / emb[:, 1], np.arange(1, 11) / 11, rtol=0.01)


def test_embed_paths_rejects_a_bad_embedder(tmp_path):
    _write_tone(tmp_path / "a.wav", 440)
    with pytest.raises(EvalError, match="shape"):
        embed_paths([str(tmp_path / "a.wav")], lambda waves: np.zeros(3), MODEL_RATE)
    with pytest.raises(EvalError, match="zero"):
        embed_paths([str(tmp_path / "a.wav")], lambda waves: np.zeros((1, 4)), MODEL_RATE)


def test_system_file_scores_perfectly_with_the_scoring_script(dataset):
    pool = load_pool(dataset / "library.csv", dataset / "queries.csv", dataset / "relevance.json")
    out = build_system_file(pool, tone_embedder, MODEL_RATE, dataset / "out" / "system.npz")
    table, counts = evaluate(out, load_relevance(dataset / "relevance.json"))
    assert counts["n_queries_scored"] == 12
    assert table["rr"].mean() == 1.0


def test_build_stops_before_embedding_when_audio_is_missing(dataset):
    pool = load_pool(dataset / "library.csv", dataset / "queries.csv", dataset / "relevance.json")
    (dataset / "audio" / "queries" / "p0_Fixed.wav").unlink()
    calls = []

    def embedder(waves):
        calls.append(len(waves))
        return np.ones((len(waves), 3))

    with pytest.raises(EvalError, match="audio files are missing"):
        build_system_file(pool, embedder, MODEL_RATE, dataset / "s.npz")
    assert calls == []
