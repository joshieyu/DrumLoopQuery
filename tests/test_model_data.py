"""Manifests, splits and kits, feature cache, and batches (src/models/data.py)."""

import dataclasses

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
    check_kits,
    check_speakers,
    held_out_kits,
    limit_patterns,
    load_clips,
    relevant_library,
    select,
)
from src.music.spec import load_spec  # noqa: E402

CFG = load_model_config()
SPEC = load_spec()
TRAIN_KIT, TEST_KIT = "avl_black_pearl", "avl_blonde_bop"  # kit ids from the spec


def spec_with_kit_splits(splits):
    """A copy of the music spec where only the kits in `splits` are used, with those splits;
    every other kit's split is unset. The real file is untouched, and these tests don't
    depend on which kits the music lead holds out."""
    kits = tuple(dataclasses.replace(k, split=splits.get(k.kit_id)) for k in SPEC.rendering.kits)
    return dataclasses.replace(SPEC, rendering=dataclasses.replace(SPEC.rendering, kits=kits))


HELD_OUT = spec_with_kit_splits({TRAIN_KIT: "train", TEST_KIT: "test"})


def write_wav(path, seconds=0.5, seed=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.random.default_rng(seed).uniform(-0.3, 0.3, int(seconds * 22050))
    sf.write(path, audio.astype(np.float32), 22050)
    return path


def manifests(tmp_path, n_patterns=3, queries_per_pattern=2):
    """Tiny manifests in README §12 format with real (noise) audio: each pattern on two
    kits, and queries from one speaker per split."""
    lib, qry = [], []
    for p in range(n_patterns):
        split = "train" if p < n_patterns - 1 else "validation"
        for kit in (TRAIN_KIT, TEST_KIT):
            audio = write_wav(tmp_path / f"lib{p}_{kit}.wav", seed=p)
            lib.append(
                dict.fromkeys(LIBRARY_COLUMNS, "x")
                | {
                    "clip_id": f"L{p}@{kit}",
                    "pattern_id": f"P{p}",
                    "dualized_pattern_id": f"D{p % 2}",
                    "split": split,
                    "kit_id": kit,
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
                    "source": "synthetic",
                    "participant_id": f"speaker_{split}",
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
    assert len(clips) == 6 + 6
    assert set(clips.kind) == {"library", "beatbox", "tap"}
    assert set(clips.source) == {"library", "synthetic"}
    by_id = clips.set_index("id")
    assert by_id.loc[f"L0@{TRAIN_KIT}", "group"] == TRAIN_KIT
    assert by_id.loc["Q0_1", "group"] == "speaker_train"
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
    table.loc[0, "query_id"] = f"L0@{TRAIN_KIT}"
    table.to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(ValueError, match="more than once"):
        load_clips(lib, tmp_path / "bad.csv")


def test_relevant_library_follows_the_answer_key(tmp_path):
    clips = load_clips(*manifests(tmp_path), audio_root=tmp_path)
    clips = clips[clips.group != TEST_KIT]
    library, queries = clips[clips.kind == "library"], clips[clips.kind != "library"]
    rel = relevant_library(queries, library, SPEC)
    assert rel["Q0_0"] == [f"L0@{TRAIN_KIT}"]  # beatbox: same pattern_id
    assert sorted(rel["Q0_1"]) == [f"L0@{TRAIN_KIT}", f"L2@{TRAIN_KIT}"]  # tap: dualized D0


# --------------------------------------------------------------------------- splits and kits


def test_held_out_kits_never_reach_training(tmp_path):
    clips = load_clips(*manifests(tmp_path), audio_root=tmp_path)
    assert held_out_kits(HELD_OUT) == [TEST_KIT]

    train = select(clips, "train", HELD_OUT, kits="train")
    assert set(train[train.kind == "library"].group) == {TRAIN_KIT}
    assert (train.kind != "library").sum() == 4  # queries are never filtered by kit

    test_library = select(clips, "train", HELD_OUT, kits="test")
    assert set(test_library[test_library.kind == "library"].group) == {TEST_KIT}

    everything = select(clips, "train", HELD_OUT, kits="all")
    assert set(everything[everything.kind == "library"].group) == {TRAIN_KIT, TEST_KIT}
    assert len(select(clips, "train", HELD_OUT, kits=[TEST_KIT])) == len(test_library)


def test_kits_with_no_split_are_not_used(tmp_path):
    clips = load_clips(*manifests(tmp_path), audio_root=tmp_path)
    only_train = spec_with_kit_splits({TRAIN_KIT: "train"})  # TEST_KIT's split unset
    train = select(clips, "train", only_train, kits="train")
    assert set(train[train.kind == "library"].group) == {TRAIN_KIT}


def test_select_needs_held_out_kits_and_known_kits(tmp_path):
    clips = load_clips(*manifests(tmp_path), audio_root=tmp_path)
    with pytest.raises(ValueError, match="no kit is marked"):
        select(clips, "train", spec_with_kit_splits({TRAIN_KIT: "train"}), kits="test")
    with pytest.raises(ValueError, match="not in the music spec"):
        select(clips, "train", SPEC, kits=["no_such_kit"])


def test_check_kits_rejects_a_kit_the_spec_doesnt_know(tmp_path):
    clips = load_clips(*manifests(tmp_path), audio_root=tmp_path)
    check_kits(clips, SPEC)
    clips.loc[0, "group"] = "mystery_kit"
    with pytest.raises(ValueError, match="mystery_kit"):
        check_kits(clips, SPEC)


def test_check_speakers_catches_a_speaker_in_two_splits(tmp_path):
    clips = load_clips(*manifests(tmp_path), audio_root=tmp_path)
    check_speakers(clips)
    clips.loc[clips.id == "Q2_0", "group"] = "speaker_train"  # a validation query
    with pytest.raises(ValueError, match="speakers appear in more than one split"):
        check_speakers(clips)
    clips.loc[clips.kind != "library", "source"] = "practice"  # stand-ins are exempt
    check_speakers(clips)


def test_check_disjoint_catches_a_shared_pattern(tmp_path):
    clips = load_clips(*manifests(tmp_path), audio_root=tmp_path)
    check_disjoint(clips, ["train", "validation"])
    clips.loc[clips.id == "Q2_0", "pattern_id"] = "P0"  # a validation query of a train pattern
    with pytest.raises(ValueError, match="more than one"):
        check_disjoint(clips, ["train", "validation"])


def test_limit_patterns_keeps_whole_patterns():
    clips = toy_clips({f"P{i}": 3 for i in range(10)})
    small = limit_patterns(clips, 4, seed=0)
    assert small.pattern_id.nunique() == 4
    assert (small.pattern_id.value_counts() == 3).all()
    assert small.equals(limit_patterns(clips, 4, seed=0))
    assert len(limit_patterns(clips, None, seed=0)) == len(clips)


# --------------------------------------------------------------------------- cache


def test_feature_cache_computes_only_whats_missing(tmp_path):
    lib, qry = manifests(tmp_path)
    clips = load_clips(lib, qry, audio_root=tmp_path)
    root = tmp_path / "cache"
    messages = []
    cache = FeatureCache.build(clips.head(5), "r1", CFG, root, log=messages.append)
    assert "5 of 5" in messages[-1]
    first = cache.get(clips.id[0])
    assert first.dtype == np.float32 and first.shape[1] == 3

    cache = FeatureCache.build(clips, "r1", CFG, root, log=messages.append)  # a superset
    assert f"{len(clips) - 5} of {len(clips)}" in messages[-1]
    assert len(cache) == len(clips)

    FeatureCache.build(clips, "r1", CFG, root, log=messages.append)
    assert "already cached" in messages[-1]

    write_wav(tmp_path / "q0_0.wav", seconds=0.7, seed=999)  # re-rendered audio
    cache = FeatureCache.build(clips, "r1", CFG, root, log=messages.append)
    assert f"1 of {len(clips)}" in messages[-1]
    assert cache.get("Q0_0").shape[0] > first.shape[0]
    assert not list(root.glob("*.tmp.npy"))


# --------------------------------------------------------------------------- batches


def toy_clips(views, queries=None):
    """Clips table: views[p] library loops and queries[p] beatbox queries per pattern."""
    queries = queries or {}
    rows = [
        {"id": f"{p}_L{v}", "pattern_id": p, "dualized_pattern_id": p, "kind": "library"}
        for p, n in views.items()
        for v in range(n)
    ]
    rows += [
        {"id": f"{p}_Q{v}", "pattern_id": p, "dualized_pattern_id": p, "kind": "beatbox"}
        for p, n in queries.items()
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


def test_batches_pair_queries_with_loops():
    """5 loops and 1 query per pattern: at random, the query would often be left out."""
    clips = toy_clips({f"P{i}": 5 for i in range(8)}, queries={f"P{i}": 1 for i in range(8)})
    sampler = PatternBatchSampler(clips, 4, 3, seed=0, min_query_views=1)
    for epoch in range(1, 4):
        for batch in sampler.epoch(epoch):
            picked = clips.iloc[batch]
            per_pattern = picked.groupby("pattern_id").kind
            assert (per_pattern.apply(lambda k: (k == "beatbox").sum()) == 1).all()
            assert (per_pattern.apply(lambda k: (k == "library").sum()) == 2).all()


def test_batches_are_reproducible_and_change_each_epoch():
    clips = toy_clips({f"P{i}": 3 for i in range(8)})
    a = PatternBatchSampler(clips, 4, 2, seed=1)
    b = PatternBatchSampler(clips, 4, 2, seed=1)
    assert a.epoch(3) == b.epoch(3)
    assert a.epoch(3) != a.epoch(4)


def test_sampler_rejects_bad_settings():
    with pytest.raises(ValueError, match="at least two patterns"):
        PatternBatchSampler(toy_clips({"A": 3, "B": 1}), 4, 2)
    with pytest.raises(ValueError, match="min_query_views"):
        PatternBatchSampler(toy_clips({"A": 3, "B": 3}), 4, 2, min_query_views=3)
