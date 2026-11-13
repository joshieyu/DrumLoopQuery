"""End to end on tiny made-up audio: manifests -> training -> .npz -> D's score.

The library has each rhythm on two kits, one of them held out for testing in a copy of
the music spec, so these tests also check that training never touches the held-out kit.
"""

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")  # part of the optional "model" extra

import soundfile as sf  # noqa: E402

from src.eval.evaluate import evaluate  # noqa: E402
from src.models.config import load_model_config  # noqa: E402
from src.models.data import LIBRARY_COLUMNS, QUERY_COLUMNS, FeatureCache, load_clips  # noqa: E402
from src.models.embed import embed_clips, library_kits, load_run, scoring_sets  # noqa: E402
from src.models.export import write_embeddings  # noqa: E402
from src.models.plot_curves import plot_runs  # noqa: E402
from src.models.train import train, training_sets  # noqa: E402
from src.music.spec import load_spec  # noqa: E402

SR = 22050
SPEC = load_spec()
TRAIN_KIT, TEST_KIT = "avl_black_pearl", "avl_blonde_bop"
KITS = {TRAIN_KIT: (60, "train"), TEST_KIT: (90, "test")}  # kit -> (tone Hz, split)
CPU = torch.device("cpu")
QUIET = dict(log=lambda *_: None)


def spec_with_kit_splits(splits):
    """A copy of the music spec where only the kits in `splits` are used (others unset)."""
    kits = tuple(dataclasses.replace(k, split=splits.get(k.kit_id)) for k in SPEC.rendering.kits)
    return dataclasses.replace(SPEC, rendering=dataclasses.replace(SPEC.rendering, kits=kits))


HELD_OUT = spec_with_kit_splits({k: s for k, (_, s) in KITS.items()})


def hits_audio(times, freq, seconds=2.0):
    audio = np.zeros(int(seconds * SR), np.float32)
    t = np.arange(int(0.08 * SR)) / SR
    hit = (np.sin(2 * np.pi * freq * t) * np.exp(-t / 0.02)).astype(np.float32)
    for s in times:
        i = int(s * SR)
        audio[i : i + hit.size] += hit[: audio.size - i]
    return audio


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    """12 rhythms (6 train, 3 validation, 3 test), each on both kits, plus two high-pitched
    'beatbox' queries from one speaker per split."""
    root = tmp_path_factory.mktemp("e2e")
    rng = np.random.default_rng(0)
    lib, qry = [], []
    for p in range(12):
        split = "train" if p < 6 else "validation" if p < 9 else "test"
        times = np.sort(rng.uniform(0.05, 1.8, size=6))
        for kit, (freq, _) in KITS.items():
            path = root / f"lib{p}_{kit}.wav"
            sf.write(path, hits_audio(times, freq), SR)
            lib.append(
                dict.fromkeys(LIBRARY_COLUMNS, "x")
                | {
                    "clip_id": f"L{p}@{kit}",
                    "pattern_id": f"P{p}",
                    "dualized_pattern_id": f"D{p}",
                    "split": split,
                    "kit_id": kit,
                    "audio_path": str(path),
                }
            )
        for k in range(2):
            path = root / f"q{p}_{k}.wav"
            sf.write(path, hits_audio(times + rng.uniform(-0.01, 0.01, 6), 3000), SR)
            qry.append(
                dict.fromkeys(QUERY_COLUMNS, "x")
                | {
                    "query_id": f"Q{p}_{k}",
                    "pattern_id": f"P{p}",
                    "dualized_pattern_id": f"D{p}",
                    "query_type": "beatbox",
                    "source": "synthetic",
                    "participant_id": f"speaker_{split}",
                    "split": split,
                    "audio_path": str(path),
                }
            )
    pd.DataFrame(lib).to_csv(root / "library.csv", index=False)
    pd.DataFrame(qry).to_csv(root / "queries.csv", index=False)
    return root


def small_cfg(epochs):
    cfg = load_model_config()
    return dataclasses.replace(
        cfg, training=dataclasses.replace(cfg.training, epochs=epochs, batch_patterns=3)
    )


def clips_of(dataset):
    return load_clips(dataset / "library.csv", dataset / "queries.csv")


def test_train_embed_and_score_on_held_out_kits(dataset, tmp_path):
    clips = clips_of(dataset)
    run = tmp_path / "run"
    summary = train(
        cfg=small_cfg(2),
        spec=HELD_OUT,
        clips=clips,
        front_end="r1",
        out_dir=run,
        device=CPU,
        cache_root=tmp_path / "r1",
        **QUIET,
    )
    for name in ("best.pt", "last.pt", "log.csv", "run.json"):
        assert (run / name).exists()
    log = pd.read_csv(run / "log.csv")
    assert list(log.epoch) == [1, 2]
    assert {"train_loss", "val_mrr", "val_hit@1", "val_chance_mrr"} <= set(log.columns)
    saved = json.loads((run / "run.json").read_text())
    assert saved["train_kits"] == [TRAIN_KIT]  # the held-out kit never reached training
    assert saved["held_out_kits"] == [TEST_KIT]
    assert saved["train_patterns"] == 6 and summary["best_epoch"] in (1, 2)

    # Score the test split on the held-out kit only, with D's code.
    model, _, front_end, _ = load_run(run, "best", CPU)
    queries, library, key = scoring_sets(clips, HELD_OUT, "test", library_kits(HELD_OUT, "test"))
    assert key is None and set(library.group) == {TEST_KIT}
    cache = FeatureCache.build(
        pd.concat([queries, library]), front_end, small_cfg(2), tmp_path / "r1", **QUIET
    )
    path = write_embeddings(
        tmp_path / "s_t.npz",
        list(queries.id),
        embed_clips(model, queries, cache, CPU),
        list(library.id),
        embed_clips(model, library, cache, CPU),
    )
    relevance = {q: [f"L{q[1:].split('_')[0]}@{TEST_KIT}"] for q in queries.id}
    table, counts = evaluate(path, relevance)
    assert counts["n_queries_scored"] == 6 and counts["n_library"] == 3
    assert table.rr.between(1 / 3, 1).all()


def test_quick_run_limits_patterns(dataset, tmp_path):
    train_clips, val_clips = training_sets(clips_of(dataset), HELD_OUT, max_patterns=2, **QUIET)
    assert train_clips.pattern_id.nunique() == 2 and val_clips.pattern_id.nunique() == 2


def test_shared_patterns_are_left_out_of_training(dataset):
    clips = clips_of(dataset)
    clips.loc[clips.id == "Q9_0", "pattern_id"] = "P0"  # a test query of a train pattern
    messages = []
    train_clips, _ = training_sets(clips, HELD_OUT, log=messages.append)
    assert "P0" not in set(train_clips.pattern_id)
    assert "1 patterns appear in more than one split" in messages[0]


def test_training_refuses_a_speaker_in_two_splits(dataset):
    clips = clips_of(dataset)
    clips.loc[clips.id == "Q9_0", "group"] = "speaker_train"
    with pytest.raises(ValueError, match="speakers"):
        training_sets(clips, HELD_OUT, **QUIET)


def test_cross_kit_fallback(dataset):
    clips = clips_of(dataset)
    queries, library, key = scoring_sets(clips, HELD_OUT, "test", "all", query_kit=TEST_KIT)
    assert set(queries.group) == {TEST_KIT} and set(library.group) == {TRAIN_KIT}
    assert (queries.kind == "library").all()  # drum loops, embedded as loops
    assert key == {f"L{p}@{TEST_KIT}": [f"L{p}@{TRAIN_KIT}"] for p in (9, 10, 11)}


def test_library_kits_falls_back_when_none_are_held_out():
    messages = []
    nothing_held_out = spec_with_kit_splits({TRAIN_KIT: "train", TEST_KIT: "train"})
    assert library_kits(nothing_held_out, "test", log=messages.append) == "all"
    assert "holds no kit out" in messages[0]
    assert library_kits(HELD_OUT, "test") == "test"
    assert library_kits(HELD_OUT, "a, b") == ["a", "b"]


def test_reloaded_model_matches_and_curves_plot(dataset, tmp_path):
    clips = clips_of(dataset)
    run = tmp_path / "run"
    train(
        cfg=small_cfg(1),
        spec=HELD_OUT,
        clips=clips,
        front_end="r3",
        out_dir=run,
        device=CPU,
        cache_root=tmp_path / "r3",
        **QUIET,
    )
    a, *_ = load_run(run, "last")
    b, *_ = load_run(run, "last")
    some = clips[clips.group != TEST_KIT].head(4)
    cache = FeatureCache.build(some, "r3", small_cfg(1), tmp_path / "r3", **QUIET)
    np.testing.assert_allclose(
        embed_clips(a, some, cache, CPU), embed_clips(b, some, cache, CPU), atol=1e-6
    )
    png = plot_runs([run], tmp_path / "curves.png", title="test")
    assert png.exists() and png.stat().st_size > 10_000
