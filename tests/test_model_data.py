"""Manifests, feature cache, and batches (src/models/data.py)."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("scipy")  # part of the optional "model" extra

import soundfile as sf  # noqa: E402

from src.models.config import load_model_config  # noqa: E402
from src.models.data import (  # noqa: E402
    LIBRARY_COLUMNS,
    QUERY_COLUMNS,
    FeatureCache,
    PatternBatchSampler,
    check_disjoint,
    load_clips,
    relevant_library,
)
from src.music.spec import load_spec  # noqa: E402

CFG = load_model_config()
SPEC = load_spec()


def write_wav(path, seconds=0.5, seed=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.random.default_rng(seed).uniform(-0.3, 0.3, int(seconds * 22050))
    sf.write(path, audio.astype(np.float32), 22050)
    return path


def manifests(tmp_path, n_patterns=3, queries_per_pattern=2):
    """Tiny manifests in README §12 format, with real (noise) audio files."""
    lib, qry = [], []
    for p in range(n_patterns):
        split = "train" if p < n_patterns - 1 else "validation"
        audio = write_wav(tmp_path / f"lib{p}.wav", seed=p)
        lib.append(
            dict.fromkeys(LIBRARY_COLUMNS, "x")
            | {
                "clip_id": f"L{p}",
                "pattern_id": f"P{p}",
                "dualized_pattern_id": f"D{p % 2}",
                "split": split,
                "kit_id": "kitA",
                "audio_path": str(audio),
            }
        )
        for k in range(queries_per_pattern):
            audio = write_wav(tmp_path / f"q{p}_{k}.wav", seed=100 + 10 * p + k)
            qry.append(
                dict.fromkeys(QUERY_COLUMNS, "x")
                | {
                    "query_id": f"Q{p}_{k}",
                    "pattern_id": f"P{p}",
                    "dualized_pattern_id": f"D{p % 2}",
                    "query_type": "tap" if k == 1 else "beatbox",
                    "participant_id": f"person{k}",
                    "split": split,
                    "audio_path": audio.name,  # relative: read from audio_root
                }
            )
    pd.DataFrame(lib).to_csv(tmp_path / "library.csv", index=False)
    pd.DataFrame(qry).to_csv(tmp_path / "queries.csv", index=False)
    return tmp_path / "library.csv", tmp_path / "queries.csv"


# --------------------------------------------------------------------------- manifests


def test_load_clips_joins_library_and_queries(tmp_path):
    lib, qry = manifests(tmp_path)
    clips = load_clips(lib, qry, audio_root=tmp_path)
    assert len(clips) == 3 + 6
    assert set(clips.kind) == {"library", "beatbox", "tap"}
    assert clips.set_index("id").loc["L0", "group"] == "kitA"
    assert clips.set_index("id").loc["Q0_1", "group"] == "person1"
    assert all(p.startswith(str(tmp_path)) for p in clips.audio_path)


def test_manifest_problems_are_reported(tmp_path):
    lib, qry = manifests(tmp_path)
    table = pd.read_csv(qry).drop(columns=["snr_db"])
    table.to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_clips(lib, tmp_path / "bad.csv")

    table = pd.read_csv(qry)
    table.loc[0, "query_type"] = "humming"
    table.to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(ValueError, match="unknown query_type"):
        load_clips(lib, tmp_path / "bad.csv")

    table = pd.read_csv(qry)
    table.loc[0, "query_id"] = "L0"
    table.to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(ValueError, match="more than once"):
        load_clips(lib, tmp_path / "bad.csv")


def test_check_disjoint_catches_a_shared_pattern(tmp_path):
    clips = load_clips(*manifests(tmp_path), audio_root=tmp_path)
    check_disjoint(clips, ["train", "validation"])
    clips.loc[clips.id == "Q2_0", "pattern_id"] = "P0"  # a validation query of a train pattern
    with pytest.raises(ValueError, match="more than one"):
        check_disjoint(clips, ["train", "validation"])


def test_relevant_library_follows_the_answer_key(tmp_path):
    clips = load_clips(*manifests(tmp_path), audio_root=tmp_path)
    library, queries = clips[clips.kind == "library"], clips[clips.kind != "library"]
    rel = relevant_library(queries, library, SPEC)
    assert rel["Q0_0"] == ["L0"]  # beatbox: same pattern_id
    assert sorted(rel["Q0_1"]) == ["L0", "L2"]  # tap: same dualized id (D0)


# --------------------------------------------------------------------------- cache


def test_feature_cache_computes_once_and_notices_changes(tmp_path):
    lib, qry = manifests(tmp_path)
    clips = load_clips(lib, qry, audio_root=tmp_path)
    path = tmp_path / "cache" / "r1.npz"
    messages = []
    cache = FeatureCache.build(clips, "r1", CFG, path, log=messages.append)
    assert len(cache) == len(clips)
    first = cache.get("Q0_0")
    assert first.dtype == np.float32 and first.shape[1] == 3

    FeatureCache.build(clips, "r1", CFG, path, log=messages.append)
    assert "reusing" in messages[-1]

    write_wav(tmp_path / "q0_0.wav", seconds=0.7, seed=999)  # re-rendered audio
    cache = FeatureCache.build(clips, "r1", CFG, path, log=messages.append)
    assert "computing" in messages[-1]
    assert cache.get("Q0_0").shape[0] > first.shape[0]


# --------------------------------------------------------------------------- batches


def toy_clips(views):
    """Clips table with the given number of views per pattern."""
    rows = [
        {"id": f"{p}_{v}", "pattern_id": p, "dualized_pattern_id": p, "kind": "library"}
        for p, n in views.items()
        for v in range(n)
    ]
    return pd.DataFrame(rows)


def test_batches_hold_patterns_times_views():
    clips = toy_clips({f"P{i}": 4 for i in range(10)} | {"lonely": 1})
    sampler = PatternBatchSampler(clips, batch_patterns=4, views_per_pattern=3, seed=0)
    assert "lonely" not in sampler.patterns  # one clip has no positive
    batches = sampler.epoch(1)
    assert len(batches) == len(sampler) == 3  # 4 + 4 + 2 patterns
    for batch in batches:
        counts = clips.iloc[batch].pattern_id.value_counts()
        assert (counts == 3).all()
        assert len(set(batch)) == len(batch)
    seen = {p for b in batches for p in clips.iloc[b].pattern_id}
    assert seen == set(sampler.patterns)


def test_batches_are_reproducible_and_change_each_epoch():
    clips = toy_clips({f"P{i}": 3 for i in range(8)})
    a = PatternBatchSampler(clips, 4, 2, seed=1)
    b = PatternBatchSampler(clips, 4, 2, seed=1)
    assert a.epoch(3) == b.epoch(3)
    assert a.epoch(3) != a.epoch(4)


def test_sampler_needs_two_usable_patterns():
    with pytest.raises(ValueError, match="at least two patterns"):
        PatternBatchSampler(toy_clips({"A": 3, "B": 1}), 4, 2)
